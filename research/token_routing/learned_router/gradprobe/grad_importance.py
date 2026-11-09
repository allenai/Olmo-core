"""Gradient token-importance probe on a frozen dense CTC checkpoint (answer-CE, teacher forced).

    python research/token_routing/learned_router/gradprobe/grad_importance.py --task nq --rung 8k --rows 2 \\
        --ckpt /data/prasann/devloss_grid/ckpts/ctc-4b-nq-full --data-root /data/prasann/devloss_grid/data --out ...

Per row, one forward + backward through the relaxed deletion path (``soft_keep``: attention +log g key bias,
GDN beta / log-decay scaled by g, removal-aware conv) with a per-token gate g = 1 on every DOCUMENT BODY token
(markers, prompt, question and answer are never dropped). ``dCE/dg_t`` is the first-order estimate of the
change in answer CE when token t is deleted: predicted harm h_t = -dCE/dg_t. A second score is grad x input
on the input embedding, h_t = -(dCE/de_t . e_t) (Taylor estimate of zeroing the embedding).

Drop curves: keep the top-k body tokens (k = rho * n_body) under each rule, HARD-delete the rest (sequence
compacted, original RoPE positions -- the grid's ``flash_ce`` path) and record CE:
  grad (signed harm), grad_abs (|dCE/dg|), gxe (signed grad x input), ig / ig_abs (integrated gradients along
  g = alpha, alpha 1 -> 0; ig4 = 4 steps), iter_abs / iter_abs_c (|grad| recomputed under the current mask at
  every step of a fine / coarse descending schedule; iter_grad = signed, opt-in), rand (seeded). Every curve point
  records ``passes``, the forward+backward passes its rule spent (1 for one-shot grad, S for IG, step count for iter). References: full, the bar
  (gold_fl20p8_noslot, its own keep fraction) and gold_only (gold documents' bodies only).
The answer is used to score tokens, so every rule except rand is an ORACLE (headroom), not deployable.

Checks (row-level, all logged): soft path at g = 1 vs the flash full CE; soft path at a binary mask vs the hard
deletion; gradient determinism (two backward passes); doc-level linearity (actual dCE of deleting each
document's body vs the summed predicted harm, Spearman / Pearson); gold-vs-non-gold AUROC of each score.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "layerskip"))
import frontier as FR  # noqa: E402
import layerskip_lib as LS  # noqa: E402

RL, TR, G = FR.RL, FR.TR, FR.G


def log(m: str) -> None:
    print(f"[gradprobe] {m}", flush=True)


def rankdata(x: np.ndarray) -> np.ndarray:
    """1-based average ranks (ties share their mean rank), as scipy.stats.rankdata (no scipy in the env)."""
    x = np.asarray(x, dtype=np.float64)
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    upper = np.cumsum(cnt)
    return ((upper - (cnt - 1) / 2.0)[inv]).astype(np.float64)


def auroc(score: np.ndarray, pos: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), ties half."""
    if pos.all() or not pos.any():
        return float("nan")
    r = rankdata(score)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def forward_ckpt(model, x, logits_to_keep, soft_keep):
    """``model(x, logits_to_keep=..., soft_keep=...)`` with per-block activation checkpointing (recompute each block
    in backward): same math, O(1 block) saved activations -- needed for the gradient pass at 32k. Mirrors
    ``layerskip_lib.forward_layerskip`` (embed -> blocks -> lm_head)."""
    from torch.utils.checkpoint import checkpoint

    ids, _, abk, pbk, lmk = model._prepare_inputs(x, None, logits_to_keep=logits_to_keep)
    assert set(abk) <= {"position_ids"}, sorted(abk)
    h = LS._embed(model, ids)
    for key, block in model.blocks.items():
        bk = {**abk, **pbk.get(int(key), {})}
        h = checkpoint(lambda hh, sk_, b=block, kw=bk: b(hh, soft_keep=sk_, **kw), h, soft_keep, use_reentrant=False)
    return model.lm_head(h, **lmk)


class Prober:
    def __init__(self, model, p, dev, ckpt_blocks: bool = False):
        self.model, self.p, self.dev = model, p, dev
        self.ckpt_blocks = ckpt_blocks
        f = p["feats"]
        body = f["is_marker"] < 0.5
        self.body_idx = f["idx"][body]  # (Nb,) absolute positions of document body tokens
        self.gold = f["gold"][body].bool().cpu().numpy()
        self.doc = f["doc"][body].cpu().numpy()
        self.j = f["j"][body].cpu().numpy()
        self.Nb = int(self.body_idx.numel())
        self._emb = None

    # ---------------------------------------------------------------- soft (relaxed) path
    def soft_ce_grad(self, z: torch.Tensor, want_grad: bool = True, ckpt=None):
        """CE under body gates z (Nb,) in [0, 1]; returns (ce, dCE/dz, dCE/de . e) (the last two None if not want_grad)."""
        p, model = self.p, self.model
        model.eval()
        model._pooled_keep_holder = None
        z = z.detach().float().clone().requires_grad_(want_grad)
        sk = torch.ones(p["T"], dtype=torch.float32, device=self.dev).index_put((self.body_idx,), z)
        store = {}

        def hook(_m, _inp, out):
            o = out.detach().requires_grad_(want_grad)
            store["e"] = o
            return o

        h = model.embeddings.register_forward_hook(hook)
        try:
            with torch.set_grad_enabled(want_grad):
                if (self.ckpt_blocks if ckpt is None else ckpt) and want_grad:
                    lg = forward_ckpt(model, p["x"], p["pred"][None], sk[None])[0].float()
                else:
                    lg = model(p["x"], logits_to_keep=p["pred"][None], soft_keep=sk[None])[0].float()
                ce = F.cross_entropy(lg, p["tgt"])
                if not want_grad:
                    return float(ce), None, None
                ce.backward()
        finally:
            h.remove()
        e = store["e"]
        gxe = (e.grad.float() * e.float()).sum(-1)[0, self.body_idx]
        return float(ce), z.grad.detach().clone(), gxe.detach()

    # ---------------------------------------------------------------- hard deletion (grid semantics)
    def hard_ce(self, keep_body: torch.Tensor):
        """keep_body (Nb,) bool -> (CE, T2, FLOPs ratio) with dropped body tokens removed from the sequence."""
        p = self.p
        real = torch.ones(p["T"], dtype=torch.bool, device=self.dev)
        real[self.body_idx[~keep_body.to(self.dev)]] = False
        it = FR.compact(p, real, torch.zeros(p["T"], dtype=torch.bool))
        ce = FR.flash_ce(self.model, it)
        return ce, it["T2"], LS.flops_ratio(p["T"], it["T2"], [0] * len(self.model.blocks))

    def topk(self, score: torch.Tensor, k: int, among: torch.Tensor = None) -> torch.Tensor:
        s = score.clone().float()
        if among is not None:
            s[~among] = -float("inf")
        keep = torch.zeros(self.Nb, dtype=torch.bool, device=s.device)
        if k > 0:
            keep[torch.topk(s, k).indices] = True
        return keep


def run_row(model, p, r, a, tok, dev, row_i):
    ck = a.ckpt_blocks == "1" or (a.ckpt_blocks == "auto" and p["T"] > 12000)
    P = Prober(model, p, dev, ckpt_blocks=ck)
    out = {"T": p["T"], "n_body": P.Nb, "n_docs": int(p["n_docs"]), "gold_docs": sorted(r["gold"] or []),
           "ce_full": p["ce_full"], "checks": {}}
    t0 = time.time()

    # --- gradients at g = 1
    ones = torch.ones(P.Nb, device=dev)
    ce1, g1, gxe1 = P.soft_ce_grad(ones)
    out["checks"]["soft_g1_vs_flash_full"] = ce1 - p["ce_full"]
    out["checks"]["grad_sec"] = time.time() - t0
    if row_i == 0 or a.checks_all_rows:
        _, g1b, gxeb = P.soft_ce_grad(ones)
        out["checks"]["grad_rel_diff_rerun"] = float((g1b - g1).norm() / g1.norm().clamp(min=1e-20))
        out["checks"]["gxe_rel_diff_rerun"] = float((gxeb - gxe1).norm() / gxe1.norm().clamp(min=1e-20))
    out["checks"]["ckpt_blocks"] = ck
    if a.check_ckpt and row_i == 0:
        # checkpointed vs plain gradient pass (memory permitting): must agree to GPU-nondeterminism level
        ce_c, g_c, gxe_c = P.soft_ce_grad(ones, ckpt=not ck)
        out["checks"]["ckpt_vs_plain"] = {"ce": ce_c - ce1, "grad_rel": float((g_c - g1).norm() / g1.norm().clamp(min=1e-20)),
                                          "gxe_rel": float((gxe_c - gxe1).norm() / gxe1.norm().clamp(min=1e-20))}
        log(f"   ckpt-vs-plain grad check: {out['checks']['ckpt_vs_plain']}")
    harm = -g1  # predicted dCE of deleting the token
    scores = {"grad_abs": g1.abs()}  # rule -> per-body-token score, re-used for the bar-budget (@bar) points
    harm_gxe = -gxe1
    # integrated gradients along the uniform path g = alpha * 1, alpha: 1 -> 0 (midpoint rule). Completeness:
    # sum_t harm_ig_t ~= CE(every body token dropped) - CE(full) -- checked against the hard deletion below.
    # one IG per step count (--ig-steps, first = the main one, named ig / ig_abs; others igS / igS_abs)
    ig_steps = [int(v) for v in str(a.ig_steps).split(",") if v]
    harm_igs = {}
    for S in ig_steps:
        ig = torch.zeros_like(g1)
        for s_ in range(S):
            _, gz, _ = P.soft_ce_grad(torch.full_like(ones, (s_ + 0.5) / S))
            ig += gz / S
        harm_igs[S] = -ig
    harm_ig = harm_igs[ig_steps[0]] if ig_steps else -g1  # no IG requested: IG-based stats fall back to the plain gradient
    if ig_steps:
        scores["ig_abs"] = harm_ig.abs()
    ce_none, _, _ = P.hard_ce(torch.zeros(P.Nb, dtype=torch.bool, device=dev))
    out["checks"]["ig_completeness"] = {"sum_harm_ig": float(harm_ig.sum()), "actual_drop_all": ce_none - p["ce_full"]}
    out["checks"]["ig_sec"] = time.time() - t0
    out["grad_stats"] = {"harm_sum": float(harm.sum()), "harm_pos_sum": float(harm.clamp(min=0).sum()),
                         "harm_neg_sum": float(harm.clamp(max=0).sum()), "harm_absmax": float(harm.abs().max()),
                         "frac_harm_neg": float((harm < 0).float().mean())}

    # --- descriptive: gold enrichment, positional structure, top tokens
    h_np, hx_np = harm.cpu().numpy(), harm_gxe.cpu().numpy()
    out["auroc_gold"] = {"grad": auroc(h_np, P.gold), "grad_abs": auroc(np.abs(h_np), P.gold),
                         "gxe": auroc(hx_np, P.gold), "ig": auroc(harm_ig.cpu().numpy(), P.gold),
                         "ig_abs": auroc(np.abs(harm_ig.cpu().numpy()), P.gold), "gold_token_frac": float(P.gold.mean())}
    pos_mass = np.clip(h_np, 0, None)
    tot = max(pos_mass.sum(), 1e-30)
    out["harm_mass"] = {"gold_docs": float(pos_mass[P.gold].sum() / tot),
                        "first8_of_doc": float(pos_mass[P.j < 8].sum() / tot),
                        "first8_frac_tokens": float((P.j < 8).mean()),
                        "gold_first8": float(pos_mass[P.gold & (P.j < 8)].sum() / tot)}
    order = np.argsort(-h_np)
    x_np = p["x"][0].cpu().numpy()
    bidx = P.body_idx.cpu().numpy()

    def tokrow(i):
        return {"piece": tok.decode([int(x_np[bidx[i]])]), "doc": int(P.doc[i]), "j": int(P.j[i]),
                "gold": bool(P.gold[i]), "harm": float(h_np[i]), "harm_gxe": float(hx_np[i])}

    out["top_tokens"] = [tokrow(i) for i in order[: a.n_show]]
    out["bottom_tokens"] = [tokrow(i) for i in order[::-1][: a.n_show]]

    # --- check: soft path at a binary mask == hard deletion (one mid-ρ grad mask)
    k_mid = int(round(0.3 * P.Nb))
    keep_mid = P.topk(harm, k_mid)
    ce_soft_bin, _, _ = P.soft_ce_grad(keep_mid.float(), want_grad=False)
    ce_hard_mid, _, _ = P.hard_ce(keep_mid)
    out["checks"]["soft_binary_vs_hard_rho0.3"] = ce_soft_bin - ce_hard_mid

    # --- doc-level linearity: delete each document body, actual vs summed predicted harm
    n_docs = (int(P.doc.max()) + 1 if P.Nb else 0) if a.doc_level else 0
    doc_act, doc_pred, doc_pred_gxe, doc_pred_ig = [], [], [], []
    for d in range(n_docs):
        m = torch.tensor(P.doc == d, device=dev)
        if not bool(m.any()):
            continue
        ce_d, _, _ = P.hard_ce(~m)
        doc_act.append(ce_d - p["ce_full"])
        doc_pred.append(float(harm[m].sum()))
        doc_pred_gxe.append(float(harm_gxe[m].sum()))
        doc_pred_ig.append(float(harm_ig[m].sum()))
    doc_act, doc_pred, doc_pred_gxe, doc_pred_ig = map(np.array, (doc_act, doc_pred, doc_pred_gxe, doc_pred_ig))
    gold_set = set(int(g) for g in (r["gold"] or []))
    if not a.doc_level:
        doc_act = doc_pred = doc_pred_gxe = doc_pred_ig = np.zeros(3)  # placeholders (stats are meaningless; skipped)
    rank_act = np.argsort(np.argsort(-doc_act))
    rank_pred = np.argsort(np.argsort(-doc_pred))
    out["doc_level"] = {
        "spearman_grad": spearman(doc_act, doc_pred), "pearson_grad": float(np.corrcoef(doc_act, doc_pred)[0, 1]),
        "spearman_gxe": spearman(doc_act, doc_pred_gxe), "spearman_ig": spearman(doc_act, doc_pred_ig),
        "pearson_ig": float(np.corrcoef(doc_act, doc_pred_ig)[0, 1]),
        "gold_rank_actual": [int(rank_act[g]) for g in sorted(gold_set) if g < len(rank_act)],
        "gold_rank_pred": [int(rank_pred[g]) for g in sorted(gold_set) if g < len(rank_pred)],
        "actual": doc_act.tolist(), "pred": doc_pred.tolist(), "pred_gxe": doc_pred_gxe.tolist(), "pred_ig": doc_pred_ig.tolist()}
    out["checks"]["doc_sec"] = time.time() - t0

    # --- drop curves
    keeps = sorted({float(k) for k in a.keeps.split(",")}, reverse=True)
    curves = {}

    def rec(name, rho, keep, passes=0):
        """passes = forward+backward passes the rule spent to produce this mask (its scoring cost)."""
        ce, T2, fl = P.hard_ce(keep)
        curves.setdefault(name, []).append({"rho": rho, "k": int(keep.sum()), "ce": ce, "dce": ce - p["ce_full"], "passes": passes,
                                            "T2": T2, "flops": fl, "gold_kept": float(keep.cpu().numpy()[P.gold].mean())
                                            if P.gold.any() else float("nan")})

    for rho in keeps:
        k = int(round(rho * P.Nb))
        rec("grad", rho, P.topk(harm, k), 1)
        rec("grad_abs", rho, P.topk(harm.abs(), k), 1)
        rec("gxe", rho, P.topk(harm_gxe, k), 1)
        for S, h in harm_igs.items():
            nm = "ig" if S == ig_steps[0] else f"ig{S}"
            rec(nm, rho, P.topk(h, k), S)
            rec(f"{nm}_abs", rho, P.topk(h.abs(), k), S)
        for s in range(a.rand_seeds):
            gen = torch.Generator().manual_seed(1000 * s + 7)
            rec(f"rand_s{s}", rho, P.topk(torch.rand(P.Nb, generator=gen).to(dev), k))
    # iterative: re-score under the current (soft = exact at binary) mask at each schedule step
    # iter_abs: fine schedule (keeps + --iter-extra); iter_abs_c: coarse (the scored keeps only). The signed
    # iterative rule (iter_grad) is off by default (--iter-signed): it collapsed the gold document in the debug rows.
    fine = sorted(set(keeps) | {float(v) for v in a.iter_extra.split(",") if v}, reverse=True)
    variants = []
    if a.iter_fine:
        variants.append(("iter_abs", lambda gz: gz.abs(), fine))
    if a.iter_coarse:
        variants.append(("iter_abs_c", lambda gz: gz.abs(), keeps))
    if a.iter_signed:
        variants.append(("iter_grad", lambda gz: -gz, fine))
    for name, fn, sched in variants:
        cur = torch.ones(P.Nb, dtype=torch.bool, device=dev)
        for n_pass, rho in enumerate(sched, 1):
            _, gz, _ = P.soft_ce_grad(cur.float())
            cur = P.topk(fn(gz), int(round(rho * P.Nb)), among=cur)
            if rho in keeps:
                rec(name, rho, cur, n_pass)
    # aX_abs: |dCE/dg| from ONE pass with every body gate at alpha = X (1 pass; X = 0.5 is ig1). Probes WHERE on the
    # 1 -> 0 path the gradient is informative (at g = 1 the CE is saturated and the gradient is mostly noise).
    out["ce_alpha"] = {}  # soft-path CE with every body gate at alpha: where along 1 -> 0 the answer CE moves
    for al in [float(v) for v in a.alphas.split(",") if v]:
        ce_al, gz, _ = P.soft_ce_grad(torch.full_like(ones, al))
        out["ce_alpha"][f"{al:g}"] = ce_al
        scores[f"a{al:g}_abs"] = gz.abs()
        for rho in keeps:
            rec(f"a{al:g}_abs", rho, P.topk(gz.abs(), int(round(rho * P.Nb))), 1)
    # aadapt_abs: per-row gate chosen where the soft-path CE drops fastest along g (forward-only probes on --adapt-grid,
    # the midpoint of the steepest adjacent drop), then ONE gradient pass there. Cost: len(grid) forwards + 1 fwd+bwd.
    if a.adapt_grid:
        grid = sorted(float(v) for v in a.adapt_grid.split(","))
        ces = [P.soft_ce_grad(torch.full_like(ones, g_), want_grad=False)[0] for g_ in grid]
        drops = [ces[i] - ces[i + 1] for i in range(len(grid) - 1)]
        i_ = int(np.argmax(drops))
        g_star = 0.5 * (grid[i_] + grid[i_ + 1])
        out["adapt"] = {"grid": grid, "ce": ces, "g_star": g_star}
        _, gz, _ = P.soft_ce_grad(torch.full_like(ones, g_star))
        scores["aadapt_abs"] = gz.abs()
        for rho in keeps:
            rec("aadapt_abs", rho, P.topk(gz.abs(), int(round(rho * P.Nb))), 1)
    # iterK_aX_abs: iterK_abs but every step scores the kept tokens at gate alpha (dropped ones at 0), not at 1
    for spec in [v for v in a.iter_alpha.split(",") if v]:
        K, al = int(spec.split("@")[0]), float(spec.split("@")[1])
        for rho in keeps:
            cur = torch.ones(P.Nb, dtype=torch.bool, device=dev)
            for i in range(1, K + 1):
                _, gz, _ = P.soft_ce_grad(cur.float() * al)
                cur = P.topk(gz.abs(), int(round(rho ** (i / K) * P.Nb)), among=cur)
            rec(f"iter{K}_a{al:g}_abs", rho, cur, K)
    # iterK_abs: for EACH target rho its own K-step geometric schedule 1 -> rho (step i keeps rho^(i/K)); cost K passes
    for K in [int(v) for v in a.iter_k.split(",") if v]:
        for rho in keeps:
            cur = torch.ones(P.Nb, dtype=torch.bool, device=dev)
            for i in range(1, K + 1):
                _, gz, _ = P.soft_ce_grad(cur.float())
                cur = P.topk(gz.abs(), int(round(rho ** (i / K) * P.Nb)), among=cur)
            rec(f"iter{K}_abs", rho, cur, K)
    # gold-first baselines over the whole rho range (uses the answer's gold docs, so an ORACLE reference): every gold
    # document body token first, then fill the budget at random (gold_rand) or by |dCE/dg| at g = 0.5 (gold_ga);
    # below the gold fraction the same secondary score picks WITHIN the gold documents.
    if P.gold.any() and a.gold_curves:
        gold_t = torch.tensor(P.gold, device=dev, dtype=torch.float32)
        _, g_half, _ = P.soft_ce_grad(torch.full_like(ones, 0.5))
        gen = torch.Generator().manual_seed(4242)
        secondary = {"gold_rand": torch.rand(P.Nb, generator=gen).to(dev), "gold_ga": g_half.abs() / g_half.abs().max().clamp(min=1e-30)}
        scores["gold_ga"] = 2.0 * gold_t + secondary["gold_ga"].clamp(0, 1)
        for nm, sec in secondary.items():
            for rho in keeps:
                rec(nm, rho, P.topk(2.0 * gold_t + sec.clamp(0, 1), int(round(rho * P.Nb))), 1 if nm == "gold_ga" else 0)
    # the bar's family, swept over x: [gold docs] + [first 8 tokens of every doc] + first/last x% of every doc's body
    # (first_last_scores + mark_doc_topk_tokens_free, exactly heuristic_real's construction; x = 20 with headers and gold
    # IS the bar). Recorded at each row's own keep ratio; the point's "x" field names the setting.
    bar_body = p["real_bar"].to(dev)[P.body_idx]
    if a.fl_sweep:
        from olmo_core.nn.attention.chunked_mask import FREE_CHUNK_ID, mark_doc_headers_free, mark_doc_topk_tokens_free
        from olmo_core.nn.pooled_soft_token import first_last_scores

        ids_ = G.RESERVED_IDS[G.FAMILY]
        xs, cid = p["x"][0].cpu(), p["cid"]
        n_docs_ = int(cid.max()) + 1
        gold_l = [int(g_) for g_ in (r["gold"] or [])]
        bidx = P.body_idx.cpu()
        for use_gold in (True, False):
            for hdr in (True, False):
                for xf in [float(v) for v in a.fl_sweep.split(",")]:
                    c = cid[None].clone()
                    if hdr:
                        c = mark_doc_headers_free(c, xs[None], doc_start_id=ids_.doc_start, doc_end_id=ids_.doc_end, stop_id=None,
                                                  stop_count=1, extra_tokens=8, cap=32)
                    if xf > 0:
                        sc = first_last_scores(xs[None], c, doc_start_id=ids_.doc_start, doc_end_id=ids_.doc_end, n_docs=n_docs_)
                        c = mark_doc_topk_tokens_free(c, xs[None], sc, doc_start_id=ids_.doc_start, doc_end_id=ids_.doc_end,
                                                      k=xf / 100.0, n_docs=n_docs_)
                    real = (c[0] == FREE_CHUNK_ID) | (cid < 0)
                    if use_gold:
                        for g_ in gold_l:
                            real |= cid == g_
                    keep = real[bidx].to(dev)
                    nm = f"{'gold_' if use_gold else ''}{'h' if hdr else ''}fl"
                    rec(nm, float(keep.float().mean()), keep)
                    curves[nm][-1]["x"] = xf
                    if use_gold and hdr and abs(xf - 20) < 1e-9:
                        out["checks"]["gold_hfl20_vs_bar_mismatch"] = int((keep != bar_body).sum())
    # every scored rule at exactly the bar's per-row body budget (paired comparison at the bar's operating point)
    k_bar = int(bar_body.sum())
    for nm, sc in scores.items():
        rec(f"{nm}@bar", k_bar / max(P.Nb, 1), P.topk(sc, k_bar))
    # references
    rec("bar", float(bar_body.float().mean()), bar_body)
    if P.gold.any():
        gk = torch.tensor(P.gold, device=dev)
        rec("gold_only", float(gk.float().mean()), gk)
    out["curves"] = curves
    out["sec"] = time.time() - t0
    out["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
    return out


def summarize(rows, keeps):
    s = {"eval_size": len(rows)}
    names = sorted({n for r in rows for n in r["curves"]})
    for n in names:
        if n.startswith("rand_s"):
            continue
        pts = {}
        for r in rows:
            for q in r["curves"].get(n, []):
                pts.setdefault(round(q["rho"], 4) if n not in ("bar", "gold_only") else "own", []).append(q)
        s[n] = {str(k): {"dce": float(np.mean([q["dce"] for q in v])),
                         "dce_se": float(np.std([q["dce"] for q in v], ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan"),
                         "rho": float(np.mean([q["rho"] for q in v])), "flops": float(np.mean([q["flops"] for q in v])),
                         "gold_kept": float(np.nanmean([q["gold_kept"] for q in v]))} for k, v in pts.items()}
    rd = {}
    for r in rows:
        for n, v in r["curves"].items():
            if n.startswith("rand_s"):
                for q in v:
                    rd.setdefault(round(q["rho"], 4), []).append(q["dce"])
    s["rand"] = {str(k): {"dce": float(np.mean(v)), "dce_sd_over_rows_seeds": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")}
                 for k, v in rd.items()}
    for key in ("auroc_gold", "doc_level"):
        s[key] = {}
        for f in rows[0][key]:
            vals = [r[key][f] for r in rows if isinstance(r[key][f], (int, float))]
            if vals:
                s[key][f] = float(np.nanmean(vals))
    return s


def load_olmo_safetensors(path, vocab, ids, stop_ids, attn_backend):
    """``ctc_devloss_grid.load_model`` for a model-only safetensors file with olmo state-dict keys (resave_model_only.py):
    same config, strict load, same soft-token attach, bf16 on GPU."""
    from safetensors.torch import load_file

    f = path if path.endswith(".safetensors") else os.path.join(path, "model.safetensors")
    model = G.build_cfg(vocab, attn_backend).build(init_device="cpu")
    model.load_state_dict(load_file(f), strict=True)
    log(f"loaded olmo safetensors {f}")
    G.attach_soft_tokens(model, ids, 42, stop_ids)
    return model.cuda().to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="nq")
    ap.add_argument("--rung", default="8k")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--row-offset", type=int, default=0, help="skip the first N rows (to shard a rung across jobs)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-format", default="hf")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tokenizer", default=os.environ.get("DEVLOSS_TOKENIZER", G.TOKENIZER_BY_FAMILY[G.FAMILY]))
    ap.add_argument("--attn-backend", default="flash_2")
    ap.add_argument("--keeps", default="0.5,0.3,0.2,0.1,0.05,0.02,0.01")
    ap.add_argument("--iter-extra", default="0.8,0.65,0.4,0.15,0.075,0.035,0.015", help="extra steps of the fine iterative schedule")
    ap.add_argument("--iter-signed", action="store_true", help="also run the signed iterative rule (iter_grad)")
    ap.add_argument("--iter-fine", type=int, default=1, help="run the fine-schedule iter_abs (0 = skip)")
    ap.add_argument("--iter-coarse", type=int, default=1, help="run the coarse-schedule iter_abs_c (0 = skip)")
    ap.add_argument("--gold-curves", type=int, default=1, help="gold_rand / gold_ga baselines over every rho (0 = off)")
    ap.add_argument("--adapt-grid", default="", help="forward-only gate grid for the adaptive rule aadapt_abs (e.g. 0.3,...,0.8)")
    ap.add_argument("--fl-sweep", default="", help="comma list of x (percent): gold/header/first-last-x%% heuristic family")
    ap.add_argument("--alphas", default="", help="comma list of gate values: one-pass |grad| at g = alpha (aX_abs)")
    ap.add_argument("--iter-alpha", default="", help="comma list K@alpha: K-step re-scoring with kept gates at alpha")
    ap.add_argument("--iter-k", default="", help="comma list K: per-target K-step geometric |grad| schedules (iterK_abs)")
    ap.add_argument("--doc-level", type=int, default=1, help="per-document deletion linearity check (0 = skip; n_docs forwards)")
    ap.add_argument("--rand-seeds", type=int, default=3)
    ap.add_argument("--ig-steps", default="16,4", help="IG step counts; the first is named ig, the others igS")
    ap.add_argument("--n-show", type=int, default=25)
    ap.add_argument("--checks-all-rows", action="store_true")
    ap.add_argument("--ckpt-blocks", default="auto", choices=["auto", "0", "1"], help="per-block activation checkpointing in "
                    "the gradient passes (auto: rows longer than 12k tokens)")
    ap.add_argument("--check-ckpt", action="store_true", help="row 0: compare the checkpointed and plain gradient passes")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune

    ids = G.RESERVED_IDS[G.FAMILY]
    vocab = G.VOCAB_BY_FAMILY[G.FAMILY]
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = TR.load_split(a.data_root, a.task, [a.rung], tok, ids, a.row_offset + a.rows)[a.row_offset:]
    log(f"{len(rows)} rows of {a.task}@{a.rung}, lens {[len(r['ids']) for r in rows]}")
    all_ids = np.concatenate([np.asarray(r["ids"], dtype=np.int64) for r in rows])
    pieces = tok.convert_ids_to_tokens(list(range(vocab)))
    stop_ids, _, tables = G.build_tables(all_ids, vocab, pieces, ids, lambda t: tok.decode([int(t)]))
    if a.ckpt_format == "olmo_st":
        model = load_olmo_safetensors(a.ckpt, vocab, ids, stop_ids, a.attn_backend)
    else:
        model = G.load_model(a.ckpt, a.ckpt_format, vocab, ids, 42, stop_ids, a.attn_backend)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    freeze_fla_length_autotune()
    dev = next(model.parameters()).device
    TR.configure(model, stop_ids, tables.to(dev))
    model.eval()
    res = []
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for i, r in enumerate(rows):
        with torch.no_grad():
            p = FR.base_prep(model, r, ids, dev)
        o = run_row(model, p, r, a, tok, dev, i)
        res.append(o)
        c, dl = o["checks"], o["doc_level"]
        log(f"row {i + 1}/{len(rows)} T {o['T']} body {o['n_body']} docs {o['n_docs']} gold {o['gold_docs']} CE_full {o['ce_full']:.4f} "
            f"| soft-g1 - flash {c['soft_g1_vs_flash_full']:+.2e}, soft-bin - hard {c['soft_binary_vs_hard_rho0.3']:+.2e}, "
            + (f"grad rerun rel {c['grad_rel_diff_rerun']:.1e}" if "grad_rel_diff_rerun" in c else "grad rerun: row 0 only"))
        log(f"   IG completeness: sum harm_ig {c['ig_completeness']['sum_harm_ig']:+.4f} vs actual drop-all "
            f"{c['ig_completeness']['actual_drop_all']:+.4f} ({c['ig_sec']:.0f}s)")
        log(f"   AUROC gold: grad {o['auroc_gold']['grad']:.3f} |grad| {o['auroc_gold']['grad_abs']:.3f} gxe {o['auroc_gold']['gxe']:.3f} "
            f"ig {o['auroc_gold']['ig']:.3f} |ig| {o['auroc_gold']['ig_abs']:.3f} "
            f"(gold tok frac {o['auroc_gold']['gold_token_frac']:.3f}); harm mass in gold {o['harm_mass']['gold_docs']:.3f}, "
            f"in first-8 {o['harm_mass']['first8_of_doc']:.3f} (tok frac {o['harm_mass']['first8_frac_tokens']:.3f})")
        log(f"   doc-level: spearman(actual, pred) grad {dl['spearman_grad']:.3f} gxe {dl['spearman_gxe']:.3f} ig {dl['spearman_ig']:.3f}; "
            f"pearson grad {dl['pearson_grad']:.3f} ig {dl['pearson_ig']:.3f}; "
            f"gold rank actual {dl['gold_rank_actual']} pred {dl['gold_rank_pred']}")
        for n, v in o["curves"].items():
            log(f"   {n:10s} " + "  ".join(f"rho {q['rho']:.3f}: dCE {q['dce']:+.4f} gold {q['gold_kept']:.2f}" for q in v))
        el = time.time() - t0
        log(f"   (peak GPU mem {o['peak_gb']:.1f} GB; {o['sec']:.0f}s this row; {el:.0f}s total, ETA {el / (i + 1) * (len(rows) - i - 1):.0f}s)")
        json.dump({"rows": res, "argv": sys.argv, "task": a.task, "rung": a.rung}, open(a.out, "w"), indent=1)
    summ = summarize(res, a.keeps)
    log("SUMMARY " + json.dumps(summ, indent=1))
    json.dump({"summary": summ, "rows": res, "argv": sys.argv, "task": a.task, "rung": a.rung}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
