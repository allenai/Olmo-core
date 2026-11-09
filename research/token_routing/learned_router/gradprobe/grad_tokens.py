"""WHICH tokens and layers make the token-dropped training gradient differ from the full-context one, and why.

    python research/token_routing/learned_router/gradprobe/grad_tokens.py --tasks nq,outlier,contradiction --rung 8k \\
        --rows 16 --ckpt /data/prasann/devloss_grid/ckpts/q35-4b-base-modelonly --ckpt-format distcp \\
        --data-root /data/prasann/devloss_grid/data --out-prefix /data/prasann/gradprobe/out/gtok_q35base

A linear layer's weight gradient is a sum of per-token outer products, dCE/dW = sum_t delta_t x_t^T (x_t: its input at
position t, delta_t: dCE/d(output_t)). So for every layer the full-context gradient G splits EXACTLY into
  D  = sum over DROPPED tokens  (signal that disappears when they are removed)
  K  = sum over KEPT tokens in the full context,   and the dropped-context gradient is
  K' = sum over the same tokens in the dropped context (their x and delta both change).
We report, per layer and for the mixer output projection (``attention.w_out``: GDN or softmax attention) and the FFN
down projection (``feed_forward.w2``): ||D||/||G||, cos(K, G), cos(K', K), cos(K', G) (= the parameter-gradient cosine
of grad_similarity.py for that matrix), and the split of the kept-token drift into FORWARD (activations changed:
K_fwd = sum delta_t x'_t^T) vs BACKWARD (the incoming signal changed: K_bwd = sum delta'_t x_t^T): cos(K_fwd, K),
cos(K_bwd, K).
Per token, the rank-1 contribution has magnitude c_t = ||delta_t|| ||x_t|| and its full-vs-dropped cosine factorizes as
cos(delta_t, delta'_t) * cos(x_t, x'_t). These are aggregated by token category (prefix / marker / gold body / other
body / dropped body / question+template suffix / answer) and, for kept body tokens, by distance to the nearest dropped
token -- plus the residual stream after every block (cos of h and of dCE/dh per token), and the top tokens decoded.

No parameter gradients are taken: the input embedding is made to require grad, so one backward per pass yields every
delta (hooks on the projection outputs) at the cost of a normal training step.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grad_importance as GI  # noqa: E402

FR, TR, G, log = GI.FR, GI.TR, GI.G, GI.log

CATS = ["prefix", "marker", "gold_body", "body", "suffix", "answer"]
DIST_BINS = [(1, 1), (2, 4), (5, 16), (17, 64), (65, 256), (257, 10**9)]
LINEARS = ("attention.w_out", "feed_forward.w2")


def cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-30))


class Tap:
    """Hooks: per block, the input x and output-grad delta of each LINEARS module, and the block output h + dCE/dh."""

    def __init__(self, model):
        self.model, self.hs = model, []
        self.store = {}
        for li, (k, block) in enumerate(model.blocks.items()):
            mods = dict(block.named_modules())
            for nm in LINEARS:
                if nm in mods:
                    self.hs.append(mods[nm].register_forward_hook(self._lin_hook((li, nm))))
            self.hs.append(block.register_forward_hook(self._lin_hook((li, "resid"), out_as_x=True)))

    def _lin_hook(self, key, out_as_x=False):
        def hook(_m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            x = o if out_as_x else inp[0]
            ent = {"x": x.detach()[0]}
            self.store[key] = ent
            if o.requires_grad:
                o.register_hook(lambda g, e=ent: e.__setitem__("d", g.detach()[0]))
        return hook

    def run(self, x, pred, tgt, pos=None):
        """One forward + backward (embedding output requires grad); returns CE and fills self.store."""
        self.store = {}
        model = self.model
        model._pooled_keep_holder = None

        def emb_hook(_m, _i, out):
            return out.detach().requires_grad_(True)

        h = model.embeddings.register_forward_hook(emb_hook)
        try:
            kw = {} if pos is None else {"position_ids": pos}
            with torch.enable_grad():
                lg = model(x, logits_to_keep=pred[None], **kw)[0].float()
                ce = F.cross_entropy(lg, tgt)
                ce.backward()
        finally:
            h.remove()
        st, self.store = self.store, {}
        return float(ce), st


def categories(p, T, dev):
    f = p["feats"]
    cat = torch.full((T,), CATS.index("suffix"), dtype=torch.long, device=dev)
    idx = f["idx"].to(dev)
    cat[: int(idx.min())] = CATS.index("prefix")
    is_m = f["is_marker"].to(dev) > 0.5
    gold = f["gold"].to(dev).bool()
    cat[idx[is_m]] = CATS.index("marker")
    cat[idx[~is_m & gold]] = CATS.index("gold_body")
    cat[idx[~is_m & ~gold]] = CATS.index("body")
    ans = torch.cat([p["pred"], p["pred"] + 1]).to(dev).clamp(max=T - 1)
    cat[ans] = CATS.index("answer")
    return cat


def analyse(st_f, st_d, kept_pos, cat, dist, n_layers, top_layers, x_tokens, tok):
    """st_f / st_d: Tap stores for the full / dropped pass; kept_pos: (T2,) full-sequence position of each dropped-pass
    row; cat: (T,) category; dist: (T,) distance to the nearest dropped token (0 = dropped)."""
    T = cat.numel()
    dropped = torch.ones(T, dtype=torch.bool, device=cat.device)
    dropped[kept_pos] = False
    out = {"layers": []}
    tops = {}
    for li in range(n_layers):
        L = {}
        for nm in LINEARS + ("resid",):
            if (li, nm) not in st_f or "d" not in st_f[(li, nm)] or "d" not in st_d.get((li, nm), {}):
                continue
            xf, df = st_f[(li, nm)]["x"].float(), st_f[(li, nm)]["d"].float()
            xd, dd = st_d[(li, nm)]["x"].float(), st_d[(li, nm)]["d"].float()
            xk, dk = xf[kept_pos], df[kept_pos]
            # per-token: magnitude of the rank-1 contribution and its full-vs-dropped cosine factors
            c_full = df.norm(dim=-1) * xf.norm(dim=-1)  # (T,)
            c_drop = dd.norm(dim=-1) * xd.norm(dim=-1)  # (T2,)
            cos_x = F.cosine_similarity(xk, xd, dim=-1)
            cos_d = F.cosine_similarity(dk, dd, dim=-1)
            r1 = cos_x * cos_d
            E = {}
            if nm != "resid":
                Gf = df.T @ xf  # (out, in) full-context weight gradient
                Dm = df[dropped].T @ xf[dropped]
                K = Gf - Dm
                Kd = dd.T @ xd
                Kf = dk.T @ xd  # old delta, new activations: forward drift only
                Kb = dd.T @ xk  # new delta, old activations: backward drift only
                E.update({"D_over_G": float(Dm.norm() / Gf.norm().clamp(min=1e-30)), "cos_K_G": cos(K, Gf),
                          "cos_Kd_K": cos(Kd, K), "cos_Kd_G": cos(Kd, Gf), "norm_Kd_over_G": float(Kd.norm() / Gf.norm().clamp(min=1e-30)),
                          "cos_Kfwd_K": cos(Kf, K), "cos_Kbwd_K": cos(Kb, K)})
                del Gf, Dm, K, Kd, Kf, Kb
            # category mass shares (full: all tokens incl. dropped; dropped pass: kept tokens) and contribution-weighted cos
            tot_f, tot_d = float(c_full.sum()), float(c_drop.sum())
            catk = cat[kept_pos]
            E["mass_full"] = {}
            E["mass_drop"] = {}
            E["r1cos_w"] = {}
            E["cos_x_med"] = {}
            E["cos_d_med"] = {}
            for ci, cn in enumerate(CATS):
                m_all = cat == ci
                E["mass_full"][cn] = float(c_full[m_all & ~dropped].sum()) / max(tot_f, 1e-30)
                mk = catk == ci
                if mk.any():
                    E["mass_drop"][cn] = float(c_drop[mk].sum()) / max(tot_d, 1e-30)
                    w = c_full[kept_pos][mk]
                    E["r1cos_w"][cn] = float((r1[mk] * w).sum() / w.sum().clamp(min=1e-30))
                    E["cos_x_med"][cn] = float(cos_x[mk].median())
                    E["cos_d_med"][cn] = float(cos_d[mk].median())
            E["mass_full"]["dropped"] = float(c_full[dropped].sum()) / max(tot_f, 1e-30)
            E["mass_full"]["dropped_gold"] = float(c_full[dropped & (cat == CATS.index("gold_body"))].sum()) / max(tot_f, 1e-30)
            # kept body tokens by distance to the nearest dropped token
            body_k = (catk == CATS.index("body")) | (catk == CATS.index("gold_body"))
            dk_ = dist[kept_pos]
            E["by_dist"] = {}
            for lo, hi in DIST_BINS:
                m = body_k & (dk_ >= lo) & (dk_ <= hi)
                if int(m.sum()) >= 5:
                    E["by_dist"][f"{lo}-{hi}"] = {"n": int(m.sum()), "cos_x_med": float(cos_x[m].median()),
                                                  "cos_d_med": float(cos_d[m].median()),
                                                  "c_ratio_med": float((c_drop[m] / c_full[kept_pos][m].clamp(min=1e-30)).median())}
            L[nm] = E
            if li in top_layers and nm == "feed_forward.w2":
                # top-12 dropped tokens by full contribution; top-12 kept tokens by lost contribution c_full * (1 - r1cos)
                def row(pos_, val):
                    return {"pos": int(pos_), "piece": tok.decode([int(x_tokens[pos_])]), "cat": CATS[int(cat[pos_])],
                            "dist": int(dist[pos_]), "v": float(val)}
                td = torch.where(dropped, c_full, torch.zeros_like(c_full))
                o1 = torch.topk(td, min(12, int(dropped.sum()))) if dropped.any() else None
                lost = c_full[kept_pos] * (1 - r1)
                o2 = torch.topk(lost, min(12, lost.numel()))
                tops[li] = {"dropped_top": [row(int(i), v) for v, i in zip(*o1)] if o1 is not None else [],
                            "kept_top_lost": [row(int(kept_pos[i]), v) for v, i in zip(*o2)],
                            "kept_top_lost_r1cos": [float(r1[i]) for i in o2.indices]}
        out["layers"].append(L)
    out["tops"] = tops
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="nq,outlier,contradiction")
    ap.add_argument("--rung", default="8k")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-format", default="hf")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tokenizer", default=os.environ.get("DEVLOSS_TOKENIZER", G.TOKENIZER_BY_FAMILY[G.FAMILY]))
    ap.add_argument("--attn-backend", default="flash_2")
    ap.add_argument("--rules", default="rand,a0.5_abs,bar")
    ap.add_argument("--keeps", default="0.5,0.1")
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune

    ids = G.RESERVED_IDS[G.FAMILY]
    vocab = G.VOCAB_BY_FAMILY[G.FAMILY]
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    tasks = a.tasks.split(",")
    rows_by = {t: TR.load_split(a.data_root, t, [a.rung], tok, ids, a.rows) for t in tasks}
    all_ids = np.concatenate([np.asarray(r["ids"], dtype=np.int64) for t in tasks for r in rows_by[t]])
    pieces = tok.convert_ids_to_tokens(list(range(vocab)))
    stop_ids, _, tables = G.build_tables(all_ids, vocab, pieces, ids, lambda t: tok.decode([int(t)]))
    if a.ckpt_format == "olmo_st":
        model = GI.load_olmo_safetensors(a.ckpt, vocab, ids, stop_ids, a.attn_backend)
    else:
        model = G.load_model(a.ckpt, a.ckpt_format, vocab, ids, 42, stop_ids, a.attn_backend)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    freeze_fla_length_autotune()
    dev = next(model.parameters()).device
    TR.configure(model, stop_ids, tables.to(dev))
    model.eval()
    n_layers = len(model.blocks)
    mixers = [type(dict(b.named_children())["attention"]).__name__ for b in model.blocks.values()]
    log(f"family {G.FAMILY}, {n_layers} layers, mixers {dict((m, mixers.count(m)) for m in set(mixers))}")
    tap = Tap(model)
    top_layers = sorted({0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1})
    keeps = [float(v) for v in a.keeps.split(",")]
    rules = a.rules.split(",")
    t0, n_tot, n_done = time.time(), sum(len(v) for v in rows_by.values()), 0
    for task in tasks:
        res, rows = [], rows_by[task]
        out_path = f"{a.out_prefix}_{task}_{a.rung}_r{len(rows)}.json"
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        for i, r in enumerate(rows):
            with torch.no_grad():
                p = FR.base_prep(model, r, ids, dev)
            T = p["T"]
            P = GI.Prober(model, p, dev, ckpt_blocks=False)
            cat = categories(p, T, dev)
            ce_f, st_f = tap.run(p["x"], p["pred"], p["tgt"])
            row = {"T": T, "n_body": P.Nb, "ce_full": p["ce_full"], "ce_tap": ce_f, "points": []}
            masks = {}
            if "a0.5_abs" in rules:
                _, gz, _ = P.soft_ce_grad(torch.full((P.Nb,), 0.5, device=dev))
            for rho in keeps:
                k = int(round(rho * P.Nb))
                if "rand" in rules:
                    masks[("rand", rho)] = P.topk(torch.rand(P.Nb, generator=torch.Generator().manual_seed(4242 + i)).to(dev), k)
                if "a0.5_abs" in rules:
                    masks[("a0.5_abs", rho)] = P.topk(gz.abs(), k)
            if "bar" in rules:
                bb = p["real_bar"].to(dev)[P.body_idx]
                masks[("bar", float(bb.float().mean()))] = bb
            for (rule, rho), keep in masks.items():
                real = torch.ones(T, dtype=torch.bool, device=dev)
                real[P.body_idx[~keep]] = False
                it = FR.compact(p, real, torch.zeros(T, dtype=torch.bool))
                ce_d, st_d = tap.run(it["x"], it["pred"], it["tgt"], it["pos"])
                kept_pos = it["pos"][0]
                # distance of every position to the nearest dropped position (0 for dropped ones)
                drop_pos = (~real).nonzero().flatten()
                ar = torch.arange(T, device=dev)
                if drop_pos.numel():
                    j = torch.searchsorted(drop_pos, ar).clamp(max=drop_pos.numel() - 1)
                    dist = torch.minimum((ar - drop_pos[(j - 1).clamp(min=0)]).abs(), (ar - drop_pos[j]).abs())
                else:
                    dist = torch.full((T,), 10**9, device=dev)
                an = analyse(st_f, st_d, kept_pos, cat, dist, n_layers, top_layers if i < 2 else [], p["x"][0], tok)
                row["points"].append({"rule": rule, "rho": rho, "ce": ce_d, "dce": ce_d - ce_f, **an})
                del st_d
            del st_f
            res.append(row)
            n_done += 1
            el = time.time() - t0
            pt = row["points"][0]["layers"]
            msg = " ".join(f"L{li}:{pt[li]['feed_forward.w2']['cos_Kd_G']:+.2f}" for li in top_layers if "feed_forward.w2" in pt[li])
            log(f"{task} row {i + 1}/{len(rows)} T {T} CE_full {p['ce_full']:.3f} (tap {ce_f:.3f}) | {row['points'][0]['rule']}@"
                f"{row['points'][0]['rho']:.2f} w2 cos(K',G) {msg} | ETA {el / n_done * (n_tot - n_done) / 60:.0f} min, "
                f"peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GB")
            json.dump({"rows": res, "argv": sys.argv, "task": task, "rung": a.rung, "mixers": mixers, "cats": CATS,
                       "family": G.FAMILY, "top_layers": top_layers}, open(out_path, "w"))


if __name__ == "__main__":
    main()
