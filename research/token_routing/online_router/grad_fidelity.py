"""Gradient fidelity of truncated training passes: how close is the PARAMETER gradient of the answer CE under a
token drop (and the mid-stack layer skip) to the full forward/backward's gradient, on held-out rows?

The online router's objective (Prasann, 2026-10-22): the truncated forward/backward should reproduce the full one,
above all its parameter gradient. Per held-out row, one full pass gives g_full; then, for each selection rule x keep
budget rho x drop point, one truncated pass gives g_drop, and we report

  cos        cos(g_drop, g_full) over all parameters (and over the transformer blocks only)
  rel_err    ||g_drop - g_full|| / ||g_full||
  norm_ratio ||g_drop|| / ||g_full||
  dce        CE_drop - CE_full
  xrow       reference: cos(g_full(row), g_full(previous row of the same task)), i.e. what "unrelated" looks like

Rules (body tokens only; markers, prompt, query and answer always stay):
  router_doc / router_row   the online router's linear scorer, fit offline here on FIT rows (same features and
                            targets as olmo_core.nn.online_token_router), per-document share / row-level top-k
  oracle_doc / oracle_row   the same selection by the TRUE full-pass target (perfect scorer of that target)
  random_doc / random_row   uniform (random_doc = the rnd30 control arm)
  first_last_doc            each document's first and last tokens (reference heuristic, gold-blind)
Drop points: input (dropped tokens skip every block; router features h4 + h4*q, target ||dL/de||) and mid8 (every
token runs blocks 0-7, dropped tokens skip 8-31; features h8 + h8*q, target ||dL/dh8||; doc rules only).
Router-target ablation (mid8, doc rules): router_<t> / oracle_<t> for the trainer's other targets (--st-otr-target)
  gxh  log |<dL/dh8, h8>|      dh  log ||h8 - h0|| (residual change through blocks 0-7; known for every token before
  the drop, so oracle_dh is realizable at train time with no backward)      mix  within-row mean of the gnorm and dh ranks

    python research/token_routing/online_router/grad_fidelity.py --ckpt <distcp dir> --shard <mt4s488> --out x.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offline_probe as OP  # noqa: E402
from offline_probe import (  # noqa: E402
    TASKS,
    build_model,
    design,
    feature_parts,
    load_rows,
    run_row,
    targets_of,
)

BUDGETS = (0.1, 0.3, 0.5)
INPUT_RULES = ("router_doc", "router_row", "oracle_doc", "oracle_row", "random_doc", "random_row", "first_last_doc")
MID_TARGETS = ("gxh", "dh", "mix")
MID_RULES = ("router_doc", "oracle_doc", "random_doc") + tuple(f"{k}_{t}_doc" for t in MID_TARGETS for k in ("router", "oracle"))
MID_LAYER = 8


def log(m):
    print(f"[grad-fid] {m}", flush=True)


def find_ckpt(path):
    if os.path.isdir(os.path.join(path, "model_and_optim")):
        return os.path.join(path, "model_and_optim")
    if path.rstrip("/").endswith("model_and_optim"):
        return path
    c = sorted(glob.glob(f"{path}/step*/model_and_optim"), key=lambda p: int(p.split("/step")[-1].split("/")[0]))
    if not c:
        def ls(d):
            try:
                return sorted(os.listdir(d))[:40]
            except OSError as exc:
                return f"<{exc}>"
        raise SystemExit(f"no model_and_optim under {path}; contents: {ls(path)}; parent: {ls(os.path.dirname(path.rstrip('/')))}")
    return c[-1]


def _ranks(x):
    return x.argsort().argsort().double() / max(1, x.numel() - 1)


def mid_targets(o, tg, b):
    """Body-token targets for the mid-stack router-target ablation (all on the full-context pass)."""
    L = MID_LAYER
    g8, h8, e = o[f"g{L}"].float(), o[f"h{L}"].float(), o["e"].float()
    t = {"gnorm": tg[f"gnorm@L{L}"][b].double(),
         "gxh": torch.log((g8 * h8).sum(-1).abs() + 1e-12)[b].double(),
         "dh": torch.log((h8 - e).norm(dim=-1) + 1e-12)[b].double()}
    t["mix"] = 0.5 * (_ranks(t["gnorm"]) + _ranks(t["dh"]))
    return t


def grad_pass(model, row, keep_pos=None, mid_layer=0):
    """Answer-CE forward+backward with parameter gradients. keep_pos (sorted kept positions): input drop, or with
    mid_layer > 0 the gather happens before block mid_layer."""
    from torch.utils.checkpoint import checkpoint

    dev = "cuda"
    ids = torch.from_numpy(row["ids"]).to(dev)
    lm = torch.from_numpy(row["lmask"]).to(dev)
    T = ids.numel()
    pred_full = torch.nonzero(lm[1:], as_tuple=True)[0]
    tgt = ids[pred_full + 1]
    fin = torch.arange(T, device=dev) if keep_pos is None else keep_pos
    pred = torch.searchsorted(fin, pred_full)
    assert bool((fin[pred] == pred_full).all()), "an answer-predicting position was dropped"
    inp = keep_pos is not None and mid_layer == 0
    x = (ids[keep_pos] if inp else ids)[None]
    _, _, abk, pbk, lmk = model._prepare_inputs(
        x, None, logits_to_keep=pred[None], position_ids=(keep_pos[None] if inp else None)
    )
    h = model.embeddings(x)
    for key, block in model.blocks.items():
        i = int(key)
        if mid_layer and keep_pos is not None and i == mid_layer:
            h = h[:, keep_pos]
            abk = {**abk, "position_ids": keep_pos[None]}
        kw = {**abk, **pbk.get(i, {})}
        h = checkpoint(lambda hh, b=block, k=kw: b(hh, **k), h, use_reentrant=False)
    ce = F.cross_entropy(model.lm_head(h, **lmk)[0].float(), tgt)
    ce.backward()
    return float(ce)


def doc_topk(score, doc, rho):
    """Per-document top ceil(rho * |doc|) by score (score/doc over body tokens)."""
    n = score.numel()
    o1 = torch.argsort(-score)
    o2 = o1[torch.argsort(doc[o1], stable=True)]
    ds = doc[o2]
    cnt = torch.bincount(doc)
    starts = torch.cumsum(cnt, 0) - cnt
    rank = torch.arange(n, device=score.device) - starts[ds]
    keep_sorted = rank < torch.ceil(rho * cnt.float()).long()[ds]
    keep = torch.zeros(n, dtype=torch.bool, device=score.device)
    keep[o2] = keep_sorted
    return keep


def row_topk(score, rho):
    k = max(1, int(math.ceil(rho * score.numel())))
    keep = torch.zeros(score.numel(), dtype=torch.bool, device=score.device)
    keep[torch.argsort(-score)[:k]] = True
    return keep


class GradStore:
    def __init__(self, model):
        self.params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    def take(self):
        g = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for _, p in self.params]
        for _, p in self.params:
            p.grad = None
        return g

    def compare(self, ga, gb):
        out = {}
        for scope in ("all", "blocks"):
            dot = na = nb = diff = 0.0
            for (n, _), a, b in zip(self.params, ga, gb):
                if scope == "blocks" and not n.startswith("blocks."):
                    continue
                a32, b32 = a.float(), b.float()
                dot += float((a32 * b32).sum())
                na += float(a32.pow(2).sum())
                nb += float(b32.pow(2).sum())
                diff += float((a32 - b32).pow(2).sum())
            out[f"cos_{scope}"] = dot / math.sqrt(max(na * nb, 1e-30))
            out[f"rel_err_{scope}"] = math.sqrt(diff / max(nb, 1e-30))
            out[f"norm_ratio_{scope}"] = math.sqrt(na / max(nb, 1e-30))
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--fit-rows", type=int, default=32)
    ap.add_argument("--test-rows", type=int, default=12)
    ap.add_argument("--ridge", type=float, default=0.01)
    ap.add_argument("--row-offset", type=int, default=0,
                    help="first row per task used; a trained checkpoint needs rows it never trained on "
                    "(e.g. 1000 on mt4s1464 for anything trained on mt4s488 / mt4s976)")
    ap.add_argument("--mid-layer", type=int, default=8, help="mid-stack drop layer L (every token runs blocks 0..L-1)")
    ap.add_argument("--tasks", default="", help="comma-separated task names from the shard's metadata (default: mt4's 4)")
    ap.add_argument("--no-input", action="store_true", help="skip the input-drop rules")
    ap.add_argument("--no-target-ablation", action="store_true", help="skip the gxh / dh / mix mid-stack rules")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    global MID_LAYER
    MID_LAYER = ML = a.mid_layer
    # the probe module reads these globals at call time: capture h_L / dL/dh_L and add the L+q feature set
    OP.HIDDEN_LAYERS = tuple(sorted(set(OP.HIDDEN_LAYERS) | {ML}))
    OP.TARGET_LAYERS = {**OP.TARGET_LAYERS, f"gnorm@L{ML}": ML}
    OP.FEATURE_SETS = {**OP.FEATURE_SETS, f"L{ML}+q": (ML, True, False)}
    tasks = [t for t in a.tasks.split(",") if t] or None
    task_names = tasks or TASKS
    mid_rules = MID_RULES if not a.no_target_ablation else ("router_doc", "oracle_doc", "random_doc")
    mid = f"mid{ML}"
    torch.manual_seed(0)
    model = build_model(find_ckpt(a.ckpt))
    fit = load_rows(a.shard, a.fit_rows, a.row_offset, tasks)
    # held-out rows: after the offline probe's fit + test rows (48 + 32), so neither probe saw them
    test = load_rows(a.shard, a.test_rows, a.row_offset + 80, tasks)

    # ---- fit the router scorers (hidden-state gradients only; parameters frozen)
    fits = {"mid8": (f"L{ML}+q", f"gnorm@L{ML}")}
    if not a.no_input:
        fits["input"] = ("L4+q", "gnorm@emb")
    if not a.no_target_ablation:
        fits.update({f"mid8_{t}": (f"L{ML}+q", t) for t in MID_TARGETS})
    stats = {k: [0, 0] for k in fits}
    t0 = time.time()
    for i, row in enumerate(fit):
        o = run_row(model, row)
        tg = targets_of(o)
        parts = feature_parts(row, o)
        b = parts["body"]
        mt = mid_targets(o, tg, b) if not a.no_target_ablation else {}
        for k, (fs, t) in fits.items():
            X = design(parts, fs)
            yt = mt[t] if t in mt else tg[t][b].double()
            y = yt - yt.mean()
            stats[k][0] = stats[k][0] + X.T @ X
            stats[k][1] = stats[k][1] + X.T @ y
        if (i + 1) % 32 == 0:
            log(f"fit {i + 1}/{len(fit)} {time.time() - t0:.0f}s")
    W = {}
    for k, (G, bvec) in stats.items():
        A = G.clone()
        A.diagonal().add_(a.ridge * float(torch.diagonal(G).mean()))
        W[k] = torch.linalg.solve(A, bvec)
    del stats

    for p in model.parameters():
        p.requires_grad_(True)
    store = GradStore(model)
    res, prev = [], {}
    t0 = time.time()
    for i, row in enumerate(test):
        # full pass: router features/targets (hidden grads) then the full parameter gradient
        for p in model.parameters():
            p.requires_grad_(False)
        o = run_row(model, row)
        tg = targets_of(o)
        parts = feature_parts(row, o)
        b = parts["body"]
        body_pos = torch.nonzero(b, as_tuple=True)[0]
        doc = parts["cid"][b].long()
        doc = doc - doc.min()
        n = body_pos.numel()
        L = body_pos.new_tensor(row["ids"].shape[0])
        # offsets within each document (first/last reference)
        cnt = torch.bincount(doc)
        starts = torch.cumsum(cnt, 0) - cnt
        j = torch.arange(n, device=b.device) - starts[doc]
        r = cnt[doc] - 1 - j
        fl_score = -torch.minimum(j, r).float() + 1e-6 * torch.rand(n, device=b.device)
        scores = {
            "mid8": {"router": design(parts, f"L{ML}+q") @ W["mid8"], "oracle": tg[f"gnorm@L{ML}"][b].double()},
        }
        if not a.no_input:
            scores["input"] = {"router": design(parts, "L4+q") @ W["input"], "oracle": tg["gnorm@emb"][b].double()}
        mt = mid_targets(o, tg, b) if not a.no_target_ablation else {}
        X8 = design(parts, f"L{ML}+q")
        for t in (MID_TARGETS if not a.no_target_ablation else ()):
            scores["mid8"][f"router_{t}"] = X8 @ W[f"mid8_{t}"]
            scores["mid8"][f"oracle_{t}"] = mt[t]
        for t in (MID_TARGETS if not a.no_target_ablation else ()):
            res.append(dict(task=row["task"], row=i, mode="tgt", rule=f"spearman_gnorm_vs_{t}", rho=1.0,
                            cos_all=float(np.corrcoef(_ranks(mt["gnorm"]).cpu(), _ranks(mt[t]).cpu())[0, 1])))
        for p in model.parameters():
            p.requires_grad_(True)
        ce_full = grad_pass(model, row)
        g_full = store.take()
        base = dict(task=row["task"], row=i, ce_full=ce_full, n_body=n, T=int(L))
        if row["task"] in prev:
            res.append(dict(base, mode="ref", rule="xrow", rho=1.0, **store.compare(g_full, prev[row["task"]])))
        gen = torch.Generator(device="cuda").manual_seed(1000 + i)
        rnd = torch.rand(n, device=b.device, generator=gen)
        for mode, rules in ((() if a.no_input else (("input", INPUT_RULES),)) + (("mid8", mid_rules),)):
            for rho in BUDGETS:
                for rule in rules:
                    kind, alloc = rule.rsplit("_", 1)
                    sc = {**scores[mode], "random": rnd, "first_last": fl_score}[kind].float()
                    keep_b = doc_topk(sc, doc, rho) if alloc == "doc" else row_topk(sc, rho)
                    keep = ~b.clone()
                    keep[body_pos[keep_b]] = True
                    keep_pos = torch.nonzero(keep, as_tuple=True)[0]
                    ce = grad_pass(model, row, keep_pos, mid_layer=(MID_LAYER if mode == "mid8" else 0))
                    g = store.take()
                    res.append(dict(base, mode=(mid if mode == "mid8" else mode), rule=rule, rho=rho, kept_frac=float(keep_b.float().mean()),
                                    dce=ce - ce_full, **store.compare(g, g_full)))
        prev = {row["task"]: g_full}  # one stored reference gradient (rows are grouped by task)
        el = time.time() - t0
        log(f"row {i + 1}/{len(test)} ({row['task']}, T={int(L)}, CE {ce_full:.3f}) {el:.0f}s, "
            f"ETA {el / (i + 1) * (len(test) - i - 1):.0f}s")

    def med(v):
        return float(np.median(v)) if v else float("nan")

    summ = []
    keys = sorted({(r["mode"], r["rule"], r["rho"]) for r in res})
    for task in list(task_names) + ["all"]:
        for mode, rule, rho in keys:
            rr = [r for r in res if r["mode"] == mode and r["rule"] == rule and r["rho"] == rho
                  and (task == "all" or r["task"] == task)]
            if not rr:
                continue
            summ.append(dict(task=task, mode=mode, rule=rule, rho=rho, eval_size=len(rr),
                             **{m: med([r[m] for r in rr if m in r]) for m in
                                ("cos_all", "cos_blocks", "rel_err_all", "norm_ratio_all", "dce", "kept_frac")}))
    json.dump(dict(mid_layer=ML, tasks=list(task_names), ckpt=a.ckpt, shard=a.shard, row_offset=a.row_offset, test_rows_per_task=a.test_rows, fit_rows_per_task=a.fit_rows,
                   rows=res, summary=summ), open(a.out, "w"), indent=1)
    log("=== medians over held-out rows (cos_all / rel_err / norm_ratio / dCE) ===")
    for s in summ:
        if s["task"] in ("all",) or s["mode"] == "ref":
            log(f"{s['task']:13s} {s['mode']:5s} {s['rule']:15s} rho={s['rho']:.1f} cos={s['cos_all']:+.3f} "
                f"(blocks {s['cos_blocks']:+.3f}) rel_err={s['rel_err_all']:.3f} norm={s['norm_ratio_all']:.3f} "
                f"dCE={s['dce']:+.3f} n={s['eval_size']}")
    for task in task_names:
        for s in summ:
            if s["task"] == task and s["rho"] == 0.3:
                log(f"{task:13s} {s['mode']:5s} {s['rule']:15s} rho=0.3 cos={s['cos_all']:+.3f} rel_err={s['rel_err_all']:.3f} "
                    f"dCE={s['dce']:+.3f}")
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
