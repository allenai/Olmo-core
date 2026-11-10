"""CPT counterpart of the SFT gradient-concentration oracle table: the smallest share of tokens whose ATTRIBUTION gradient
(the kept tokens' rank-1 weight-gradient terms under the unchanged dense forward) reaches cos 0.95 / 0.9 with the full
step gradient, for several allocation rules. OLMo 3 7B, full CPT loss (mean next-token CE over every target).

One job = one checkpoint = one optimizer step: the rows given (16 x 32k or 8 x 64k = the CPT runs' global batch). Per
row, one dense forward + backward (frozen weights, parameter grads on so every linear saves and recomputes its input);
each linear's per-token rank-1 term dy_t x_t^T is TensorSketched (CountSketch on both sides + FFT) into one shared space
per scope: blocks 7-31 (m_hi dims) and blocks 0-6 (m_lo dims). Rankings:
  scope dn  (split 7, the SFT table's convention): score = token energy summed over blocks 7-31, cosine over the
            weights of blocks 7-31.
  scope wm  (split 0): score = energy summed over all blocks, cosine over all block weights.
Allocation rules (nested sets, prefix sums accumulated over the step's rows on the GPU):
  global    one energy threshold across every row of the step (no quota)
  row       every row keeps its own top k% (each row is one document window, so = per-document)
  blk512    every 512-token block of every row keeps its own top k% (the positional blocks of the CPT drop screen)
  random    k% uniformly at random
Writes <out>.npz (accumulated sketches per rule x grid point, counts) and <out>.json (cos curves, the smallest kept share
reaching 0.95 / 0.9 per rule and scope, and a per-row sketch self-check against the exact parameter gradient).

    python research/token_routing/grad_concentration/cpt_oracle_budget.py --ckpt <ckpt> --yarn-factor 4 \
        --dev /data/prasann/olmo3_cptdrop/data/dev32k_1024 --rows 1,2,3 --label cpt_32k --out /data/.../cpt_32k
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
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cpt_gradmap import NB, ParamGrads, run_pass  # noqa: E402
from eval_devce import build_cfg, find_distcp, load_model, load_shard, parse_yarn  # noqa: E402

SPLIT = 7
THR = np.logspace(-24, 2, 521)[::-1]  # energy thresholds, descending (kept set grows)
FRACS = np.unique(np.concatenate([np.logspace(-4, 0, 161), [1.0]]))
BLK = 512


def log(m):
    print(f"[cpt-oracle] {m}", flush=True)


class Sketch:
    """Per-token TensorSketch of every block linear's rank-1 term plus the per-token energy E[b, t].

    The tensor hook on each linear's first-forward output gives dy; the input x is taken when the checkpointed block is
    recomputed during backward (either order is handled). Attribute names match cpt_gradmap.Energy for run_pass."""

    def __init__(self, model, m_lo, m_hi, chunk=1024):
        self.on, self.phase, self.E, self.fired, self.n = False, "fwd", None, 0, 0
        self.m = {0: m_lo, 1: m_hi}
        self.chunk = chunk
        self.xs, self.gs, self.meta = {}, {}, {}
        self.S = None  # {0: (T, m_lo), 1: (T, m_hi)}
        for key, block in model.blocks.items():
            b = int(key)
            for _, mod in block.named_modules():
                if isinstance(mod, nn.Linear):
                    i = self.n
                    self.n += 1
                    sp = int(b >= SPLIT)
                    m = self.m[sp]
                    g = torch.Generator().manual_seed(1009 + i)
                    hx = torch.randint(0, m, (mod.in_features,), generator=g)
                    sx = torch.randint(0, 2, (mod.in_features,), generator=g).float() * 2 - 1
                    hy = torch.randint(0, m, (mod.out_features,), generator=g)
                    sy = torch.randint(0, 2, (mod.out_features,), generator=g).float() * 2 - 1
                    self.meta[i] = (b, sp, hx.cuda(), sx.cuda(), hy.cuda(), sy.cuda())
                    mod.register_forward_hook(self._hook(i))

    def _hook(self, i):
        def hook(mod, inp, out):
            if not self.on:
                return
            if self.phase == "fwd":
                if out.requires_grad:
                    out.register_hook(lambda g, i=i: self._on_g(i, g.detach()[0]))
                return
            x = inp[0].detach()[0]
            if i in self.gs:
                self._do(i, x, self.gs.pop(i))
            else:
                self.xs[i] = x
        return hook

    def _on_g(self, i, g):
        if i in self.xs:
            self._do(i, self.xs.pop(i), g)
        else:
            self.gs[i] = g

    @torch.no_grad()
    def _do(self, i, x, g):
        b, sp, hx, sx, hy, sy = self.meta[i]
        m = self.m[sp]
        T = x.shape[0]
        self.E[b].add_(x.float().pow(2).sum(-1) * g.float().pow(2).sum(-1))
        S = self.S[sp]
        for s in range(0, T, self.chunk):
            e = min(T, s + self.chunk)
            xc, gc = x[s:e].float() * sx, g[s:e].float() * sy
            cx = torch.zeros(e - s, m, device=x.device).index_add_(1, hx, xc)
            cy = torch.zeros(e - s, m, device=x.device).index_add_(1, hy, gc)
            S[s:e] += torch.fft.irfft(torch.fft.rfft(cx) * torch.fft.rfft(cy), n=m)
        self.fired += 1


class Acc:
    """Step accumulators for one scope: full sum, and per rule the sketch of the kept set at every grid point."""

    def __init__(self, dims, dev):
        self.F = torch.zeros(dims, device=dev, dtype=torch.float64)
        self.P = {"global": torch.zeros(len(THR), dims, device=dev, dtype=torch.float64)}
        self.cnt = {"global": np.zeros(len(THR))}
        for r in ("row", "blk512", "random"):
            self.P[r] = torch.zeros(len(FRACS), dims, device=dev, dtype=torch.float64)
            self.cnt[r] = np.zeros(len(FRACS))
        self.T = 0


def prefix_sums(V_parts, order, counts):
    """Sketch sums of the first counts[j] tokens of `order`, for every j (counts nondecreasing). V_parts: list of
    (T, m_i) tensors that make up the scope's concatenated space."""
    dims = sum(v.shape[1] for v in V_parts)
    out = torch.zeros(len(counts), dims, device=order.device, dtype=torch.float64)
    run = torch.zeros(dims, device=order.device, dtype=torch.float64)
    counts = np.asarray(counts)
    CH = 2048
    T = order.numel()
    for a in range(0, T, CH):
        b = min(T, a + CH)
        sel = (counts > a) & (counts <= b)
        if not sel.any() and counts.max() <= a:
            break
        blk = torch.cat([v[order[a:b]] for v in V_parts], 1).double()
        if sel.any():
            cs = blk.cumsum(0) + run
            js = np.flatnonzero(sel)
            out[js] = cs[torch.as_tensor(counts[js] - a - 1, device=order.device)]
        run += blk.sum(0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--yarn-factor", type=parse_yarn, required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--m-hi", type=int, default=32768)
    ap.add_argument("--m-lo", type=int, default=8192)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True, help="output path stem (.npz / .json)")
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows_idx = [int(r) for r in a.rows.split(",")]
    allrows, masks, _ = load_shard(a.dev, max(rows_idx) + 1, "olmo3_7b", 0)
    L = allrows.shape[1]
    model, load_s = load_model(a.ckpt, build_cfg("olmo3_7b", a.yarn_factor), "cuda", L)
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"loaded {find_distcp(a.ckpt)} in {load_s:.0f}s; {len(rows_idx)} rows x {L} = one step")
    sk = Sketch(model, a.m_lo, a.m_hi)
    pg = ParamGrads(model)
    dev = torch.device("cuda")
    acc = {"dn": Acc(a.m_hi, dev), "wm": Acc(a.m_lo + a.m_hi, dev)}
    gen = torch.Generator(device="cpu").manual_seed(0)
    checks, meta = [], []
    t0 = time.time()
    for k, ri in enumerate(rows_idx):
        x = torch.from_numpy(allrows[ri].astype(np.int64)).cuda()
        tgt = torch.from_numpy(masks[ri]).cuda()[1:].clone()
        T = x.numel()
        sk.S = {0: torch.zeros(T, a.m_lo, device=dev), 1: torch.zeros(T, a.m_hi, device=dev)}
        sk.xs.clear()
        sk.gs.clear()
        pg.enable(True)
        loss, _ = run_pass(model, None, sk, x, tgt, True, lambda b, g: None)
        if sk.xs or sk.gs:
            raise SystemExit(f"unpaired sketch hooks: {len(sk.xs)} x, {len(sk.gs)} g")
        pgr = pg.take()
        pg.enable(False)
        E = sk.E
        # self-check: sketch norm^2 of the row's summed gradient vs the exact parameter-gradient energy (attn + ffn)
        ex_hi, ex_lo = float(pgr[SPLIT:NB, :2].sum()), float(pgr[:SPLIT, :2].sum())
        sh, sl = float(sk.S[1].double().sum(0).pow(2).sum()), float(sk.S[0].double().sum(0).pow(2).sum())
        checks.append(dict(row=ri, hi_ratio=sh / ex_hi, lo_ratio=sl / ex_lo))
        for scope, score, parts in (("dn", E[SPLIT:].sum(0), [sk.S[1]]), ("wm", E.sum(0), [sk.S[0], sk.S[1]])):
            A = acc[scope]
            A.T += T
            A.F += torch.cat([v.double().sum(0) for v in parts])
            order = score.argsort(descending=True)
            sc_sorted = score[order].double().cpu().numpy()
            # global: counts of tokens at or above each threshold
            cg = np.searchsorted(-sc_sorted, -THR, side="right")
            A.P["global"] += prefix_sums(parts, order, cg)
            A.cnt["global"] += cg
            kf = np.minimum(T, np.ceil(FRACS * T).astype(int))
            A.P["row"] += prefix_sums(parts, order, kf)
            A.cnt["row"] += kf
            # per 512-token block: order by within-block rank, ties by score
            nb = T // BLK
            rk = torch.empty(T, dtype=torch.long, device=dev)
            sb = score[:nb * BLK].view(nb, BLK)
            ob = sb.argsort(dim=1, descending=True)
            rr = torch.empty_like(ob)
            rr.scatter_(1, ob, torch.arange(BLK, device=dev).expand(nb, BLK))
            rk[:nb * BLK] = rr.reshape(-1)
            if T > nb * BLK:
                rk[nb * BLK:] = BLK - 1
            key = rk.double() * 1e6 - torch.log(score.double().clamp(min=1e-300)) / 1e3
            ordb = key.argsort()
            kb = np.minimum(T, np.ceil(FRACS * BLK).astype(int) * nb)
            A.P["blk512"] += prefix_sums(parts, ordb, kb)
            A.cnt["blk512"] += kb
            perm = torch.randperm(T, generator=gen).to(dev)
            A.P["random"] += prefix_sums(parts, perm, kf)
            A.cnt["random"] += kf
        meta.append(dict(row=ri, loss=loss, T=T))
        sk.S = None
        el = time.time() - t0
        n = k + 1
        log(f"row {n}/{len(rows_idx)} (dev row {ri}) loss {loss:.3f}; sketch/exact norm^2 blocks 7-31 "
            f"{checks[-1]['hi_ratio']:.3f}, 0-6 {checks[-1]['lo_ratio']:.3f}; {el:.0f}s, ETA "
            f"{el / n * (len(rows_idx) - n):.0f}s; GPU peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB")
    res = dict(label=a.label, ckpt=a.ckpt, dev=a.dev, rows=meta, step_rows=len(rows_idx), T=L, m_hi=a.m_hi, m_lo=a.m_lo,
               split=SPLIT, sketch_check=checks, budgets={}, curves={})
    npz = {}
    for scope, A in acc.items():
        Fn = A.F.norm()
        for rule, P in A.P.items():
            cos = ((P @ A.F) / (P.norm(dim=1) * Fn).clamp(min=1e-300)).cpu().numpy()
            kept = A.cnt[rule] / A.T
            res["curves"][f"{scope}_{rule}"] = dict(kept=kept.tolist(), cos=cos.tolist())
            for thr in (0.9, 0.95, 0.99):
                ok = np.flatnonzero(cos >= thr)
                j = ok[np.argmin(kept[ok])] if ok.size else None
                res["budgets"][f"{scope}_{rule}_{thr}"] = None if j is None else float(kept[j])
            npz[f"{scope}_{rule}_P"] = P.float().cpu().numpy()
            npz[f"{scope}_{rule}_cnt"] = A.cnt[rule]
        npz[f"{scope}_F"] = A.F.cpu().numpy()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out + ".npz", **npz)
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    for kk, v in res["budgets"].items():
        log(f"budget {kk}: {v}")
    log(f"done in {time.time() - t0:.0f}s -> {a.out}.json")


if __name__ == "__main__":
    main()
