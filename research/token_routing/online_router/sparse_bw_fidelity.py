"""Sparse-backward fidelity probe (2026-10-30): with the recipe's EXACT routed forward (two-stage 7:0.2 / 16:0.07,
router-selected, whole-model LoRA r64 over a frozen base), how close is the LoRA gradient of a compressed backward
(:mod:`olmo_core.nn.sparse_backward`) to the exact one, and what does it save?

Per held-out row: one router-routed exact pass gives g (LoRA A/B gradients, a fixed random 8M-coordinate sample);
then every variant re-runs the same forward with the compressed backward (K draws for stochastic variants):

  cos1        mean_k cos(g_k, g)                         (one row, one draw)
  cos_bar     cos(mean_k g_k, g)                         (stochastic: the K-draw mean; K-debiased like gf2)
  rel_mse     mean_k ||g_k - g||^2 / ||g||^2             (relative squared error of one draw; "variance ratio" for the
                                                           unbiased sampler, whose error is all variance)
  var         mean_k ||g_k - gbar||^2 / ||g||^2           (unbiased, K - 1)
  cos@8/@32   cos(sum over 8 / 32 rows of draw-0 g_k, sum of g)   (tasks interleaved, like training batches)
  vjp_frac    actual / dense FLOPs of the compressed linear + LoRA VJPs (what the trainer charges)
  row_frac    kept rows / rows with nonzero dy
  step_ratio  the step's training FLOPs vs a dense step (recipe formula: frozen blocks 2/3, adapters, head at the
              kept length), for the recipe, the recipe with the answer-only LM head, and the variant + answer-only head

Variants: topk{5,10,20,40}, energy{90,95,99}, sample{10,20} (K draws), and (--sketch) svd-only r{128,256,512} and
svd+sketch r{128,256,512} (sketch width = r, K draws).

    python sparse_bw_fidelity.py --ckpt <export dir or bases/..> --tag end --shard <mt4f7808> --row-offset 4000 \\
        --rows-per-task 32 --fit-offset 6000 --out /results/x.json [--sketch]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grad_fidelity2 as GF  # noqa: E402
import offline_probe as OP  # noqa: E402

SETTING = "2st_7_16"


def log(m):
    print(f"[sbw-fid] {m}", flush=True)


def variants(sketch: bool, k: int):
    from olmo_core.nn.sparse_backward import SparseBackwardConfig as C

    if sketch:
        out = [(f"svd{r}", C("sketch", rank=r, sketch=0), 1) for r in (128, 256, 512)]
        out += [(f"sk{r}", C("sketch", rank=r, sketch=r), k) for r in (128, 256, 512)]
        return out
    out = [(f"topk{int(f * 100)}", C("topk", frac=f), 1) for f in (0.05, 0.10, 0.20, 0.40)]
    out += [(f"energy{int(e * 100)}", C("energy", energy=e), 1) for e in (0.90, 0.95, 0.99)]
    out += [(f"sample{int(f * 100)}", C("sample", frac=f), k) for f in (0.10, 0.20)]
    return out


class LoraView:
    def __init__(self, model, n=8_000_000, seed=0):
        self.items = []
        named = [(nm, p) for nm, p in model.named_parameters() if "lora_" in nm]
        tot = sum(p.numel() for _, p in named)
        frac = min(1.0, n / tot)
        g = torch.Generator().manual_seed(seed)
        for nm, p in named:
            m = max(1, int(round(p.numel() * frac)))
            idx = torch.randperm(p.numel(), generator=g)[:m] if frac >= 1.0 or p.numel() < 4 * m else \
                torch.randint(p.numel(), (m,), generator=g)
            self.items.append((p, idx.cuda()))
        self.params = [p for _, p in model.named_parameters()]
        log(f"LoRA view: {len(named)} tensors, {tot / 1e6:.1f}M params, {sum(i.numel() for _, i in self.items) / 1e6:.1f}M sampled")

    def take(self):
        parts = [(p.grad.view(-1)[idx].float() if p.grad is not None else torch.zeros(idx.numel(), device="cuda"))
                 for p, idx in self.items]
        for p in self.params:
            p.grad = None
        return torch.cat(parts)


def cos(a, b):
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-30))


def step_flops(model, T, L1, L2, n_sup, r_stats):
    """(dense, recipe, recipe + answer-only head) training FLOPs of one row, by the router's formula."""
    from olmo_core.nn.early_lora import early_lora_flops_per_block

    blocks = list(model.blocks.values())
    head = float(model.lm_head.num_flops_per_token(0))
    ab = early_lora_flops_per_block(model)
    dense = sum(T * float(b.num_flops_per_token(T)) for b in blocks) + T * head
    rec = 0.0
    for i, b in enumerate(blocks):
        L = T if i < 7 else (L1 if i < 16 else L2)
        rec += L * ((2.0 / 3.0) * float(b.num_flops_per_token(L)) + ab[i])
    return dense, rec + L2 * head, rec + n_sup * head


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="ckpt")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--row-offset", type=int, default=4000)
    ap.add_argument("--rows-per-task", type=int, default=32)
    ap.add_argument("--fit-offset", type=int, default=6000)
    ap.add_argument("--fit-rows", type=int, default=24)
    ap.add_argument("--k", type=int, default=4, help="draws per stochastic variant and row")
    ap.add_argument("--sketch", action="store_true", help="the SVD / sketch variants instead of the row-sparse ones")
    ap.add_argument("--fresh-lora", action="store_true", help="install fresh r64 LoRA on all 32 blocks (base ckpt)")
    ap.add_argument("--ridge", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ls", default="")
    a = ap.parse_args()
    for d in [x for x in a.ls.split(",") if x]:
        try:
            log(f"ls {d}: {sorted(os.listdir(d))[:200]}")
        except OSError as exc:
            log(f"ls {d}: {exc}")
    from olmo_core.nn.early_lora import install_early_lora
    from olmo_core.nn.sparse_backward import install_sparse_backward, SparseBackwardConfig

    layers = [L for L, _ in GF.SETTINGS[SETTING]]
    OP.HIDDEN_LAYERS = tuple(sorted(set(OP.HIDDEN_LAYERS) | set(layers)))
    OP.TARGET_LAYERS = {**OP.TARGET_LAYERS, **{f"gnorm@L{L}": L for L in layers}}
    OP.FEATURE_SETS = {**OP.FEATURE_SETS, **{f"L{L}+q": (L, True, False) for L in layers}}
    torch.manual_seed(0)
    model, lora = GF.load_model(a.ckpt)
    if not lora:
        if not a.fresh_lora:
            raise SystemExit(f"{a.ckpt} has no LoRA adapters; pass --fresh-lora for a base checkpoint")
        install_early_lora(model, 32, 64, 32.0)  # B = 0: the recipe's step-1 state
        lora = {"n_blocks": 32, "rank": 64, "scale": 32.0, "fresh": True}
    vs = variants(a.sketch, a.k)
    install_sparse_backward(model, SparseBackwardConfig("topk", frac=1.0))
    st = model._sparse_backward
    fit = OP.load_rows(a.shard, a.fit_rows, a.fit_offset, None)
    test = OP.load_rows(a.shard, a.rows_per_task, a.row_offset, None)
    by = {}
    for r in test:
        by.setdefault(r["task"], []).append(r)
    test = [r for grp in zip(*by.values()) for r in grp]
    # ---- per-stage routers (as grad_fidelity2): ridge on h_L + h_L*q, target log ||dL/dh_L||
    for p in model.parameters():
        p.requires_grad_(False)
    st.enabled = False
    stats = {L: [0, 0] for L in layers}
    t0 = time.time()
    for row in fit:
        o = OP.run_row(model, row)
        tg = OP.targets_of(o)
        parts = OP.feature_parts(row, o)
        bb = parts["body"]
        for L in layers:
            X = OP.design(parts, f"L{L}+q")
            y = tg[f"gnorm@L{L}"][bb].double()
            stats[L][0] = stats[L][0] + X.T @ X
            stats[L][1] = stats[L][1] + X.T @ (y - y.mean())
    W = {}
    for L, (G, bvec) in stats.items():
        A = G.clone()
        A.diagonal().add_(a.ridge * float(torch.diagonal(G).mean()))
        W[L] = torch.linalg.solve(A, bvec)
    del stats
    log(f"router fit on {len(fit)} rows in {time.time() - t0:.0f}s")
    view = LoraView(model)
    groups = (8, 32)
    acc = {(n, d, G): None for n, _, K in vs for d in range(min(K, 2)) for G in groups}
    acc_ex = {G: None for G in groups}
    grp_out, rows_out = [], []
    t0 = time.time()
    for i, row in enumerate(test):
        for p in model.parameters():
            p.requires_grad_(False)
        st.enabled = False
        o = OP.run_row(model, row)
        tg = OP.targets_of(o)
        parts = OP.feature_parts(row, o)
        bm = parts["body"]
        body_pos = torch.nonzero(bm, as_tuple=True)[0]
        doc = parts["cid"][bm].long()
        doc = doc - doc.min()
        n = body_pos.numel()
        ctx = dict(b=bm, body_pos=body_pos, doc=doc, n=n,
                   router={L: OP.design(parts, f"L{L}+q") @ W[L] for L in layers}, oracle=None, first_last=None)
        gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(f"{a.tag}/{i}".encode()))
        sets, _ = GF.select("router", SETTING, ctx, gen)
        for nm, p in model.named_parameters():
            p.requires_grad_("lora_" in nm or nm.startswith("lm_head."))
        ce = GF.grad_pass(model, row, sets)
        g = view.take()
        g2 = float(g @ g)
        T = int(row["ids"].shape[0])
        L1, L2 = int(sets[0][1].numel()), int(sets[1][1].numel())
        n_sup = int(np.asarray(row["lmask"][1:]).sum())
        dense, rec, rec_sh = step_flops(model, T, L1, L2, n_sup, None)
        base = dict(tag=a.tag, task=row["task"], row=i, T=T, L1=L1, L2=L2, n_sup=n_sup, ce=ce, g_norm=math.sqrt(g2),
                    recipe_ratio=rec / dense, recipe_sh_ratio=rec_sh / dense)
        for G in groups:
            acc_ex[G] = g.clone() if acc_ex[G] is None else acc_ex[G] + g
        st.enabled = True
        for name, cfg, K in vs:
            st.cfg = cfg
            gs, fl_d, fl_a, rk, rn = [], 0.0, 0.0, 0.0, 0.0
            for kk in range(K):
                st.reset_stats()
                ce_k = GF.grad_pass(model, row, sets)
                gk = view.take()
                assert abs(ce_k - ce) < 1e-3 * max(1.0, abs(ce)), f"forward changed: {ce_k} vs {ce}"
                gs.append(gk)
                fl_d += st.flops_dense
                fl_a += st.flops_actual
                rk += st.rows_kept
                rn += st.rows_nz
                if kk < 2:
                    for G in groups:
                        key = (name, kk, G)
                        acc[key] = gk.clone() if acc[key] is None else acc[key] + gk
            gbar = sum(gs) / K
            var = (sum(float((x - gbar) @ (x - gbar)) for x in gs) / (K - 1) / g2) if K > 1 else 0.0
            gb2 = max(float(gbar @ gbar) - var * g2 / K, 1e-30)
            saved = (fl_d - fl_a) / K
            rows_out.append(dict(base, variant=name, K=K,
                                 cos1=float(np.mean([cos(x, g) for x in gs])),
                                 cos_bar=float(gbar @ g) / math.sqrt(gb2 * g2),
                                 rel_mse=float(np.mean([float((x - g) @ (x - g)) / g2 for x in gs])),
                                 var=var, norm_ratio=math.sqrt(float(gbar @ gbar) / g2),
                                 vjp_frac=fl_a / max(1.0, fl_d), row_frac=rk / max(1.0, rn),
                                 vjp_dense_over_dense=(fl_d / K) / dense,
                                 step_ratio=(rec_sh - saved) / dense))
        st.enabled = False
        for G in groups:
            if (i + 1) % G == 0:
                for (name, kk, GG), v in acc.items():
                    if GG == G and v is not None:
                        grp_out.append(dict(tag=a.tag, variant=name, draw=kk, rows=G, group=(i + 1) // G - 1,
                                            cos=cos(v, acc_ex[G])))
                        acc[(name, kk, GG)] = None
                acc_ex[G] = None
        el = time.time() - t0
        log(f"{a.tag} row {i + 1}/{len(test)} ({row['task']}, T={T}, L1={L1}, L2={L2}) {el:.0f}s, "
            f"ETA {el / (i + 1) * (len(test) - i - 1):.0f}s")
        if (i + 1) % 16 == 0 or i + 1 == len(test):
            json.dump(dict(tag=a.tag, ckpt=a.ckpt, lora=lora, shard=a.shard, row_offset=a.row_offset,
                           rows_per_task=a.rows_per_task, setting=SETTING, svd_flops=st.svd_flops,
                           rows=rows_out, groups=grp_out, summary=summarize(rows_out, grp_out)), open(a.out, "w"))
    for s in summarize(rows_out, grp_out):
        log(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()))
    log(f"wrote {a.out}")


def summarize(rows, grps):
    out = []
    med = lambda v: float(np.median(v)) if v else float("nan")  # noqa: E731
    mean = lambda v: float(np.mean(v)) if v else float("nan")  # noqa: E731
    for name in dict.fromkeys(r["variant"] for r in rows):
        rr = [r for r in rows if r["variant"] == name]
        rec = dict(variant=name, eval_size=len(rr))
        for m in ("cos1", "cos_bar", "rel_mse", "var", "norm_ratio", "vjp_frac", "row_frac"):
            rec[m] = med([r[m] for r in rr])
        for m in ("step_ratio", "recipe_ratio", "recipe_sh_ratio", "vjp_dense_over_dense"):
            rec[m] = mean([r[m] for r in rr])
        for G in (8, 32):
            rec[f"cos@{G}"] = mean([g["cos"] for g in grps if g["variant"] == name and g["rows"] == G])
        for t in dict.fromkeys(r["task"] for r in rr):
            rec[f"cos1_{t}"] = med([r["cos1"] for r in rr if r["task"] == t])
        out.append(rec)
    return out


if __name__ == "__main__":
    main()
