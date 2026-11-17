"""Cell-level (block x token) version of cpt_oracle_budget.py: the smallest share of (block, token) cells whose
ATTRIBUTION gradient reaches cos 0.95 / 0.9 with the full step gradient. A cell (b, t) = token t's rank-1 terms in all
linears of block b; its energy is E[b, t]. A cell share is directly a FLOP share of those blocks' linears.

Each block's per-token sketches go to a per-block buffer; when the block's backward is complete, its cells are ranked
by E[b, t] and folded into the step accumulators (scope dn = blocks 7-31, wm = all blocks):
  global      one cell-energy threshold over every cell of the step
  row         every row keeps its own top k% of cells (row threshold over all of its cells in the scope)
  blockrow    every (row, block) keeps its own top k% of tokens (a per-layer token quota)
  random      k% of cells uniformly at random (independently per block)
Same outputs as cpt_oracle_budget.py (<out>.json curves / budgets, sketch self-check). Run via run_cpt_oracle.sbatch
with SCRIPT=cpt_oracle_cells.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cpt_gradmap import NB, ParamGrads, run_pass  # noqa: E402
from cpt_oracle_budget import FRACS, SPLIT, THR, Sketch, prefix_sums  # noqa: E402
from eval_devce import build_cfg, find_distcp, load_model, load_shard, parse_yarn  # noqa: E402


def log(m):
    print(f"[cpt-oracle-cells] {m}", flush=True)


class CellSketch(Sketch):
    def __init__(self, model, m_lo, m_hi, on_block):
        super().__init__(model, m_lo, m_hi)
        self.per_block = {}
        for i, (b, *_r) in self.meta.items():
            self.per_block[b] = self.per_block.get(b, 0) + 1
        self.on_block = on_block
        self.buf, self.cnt = {}, {}

    @torch.no_grad()
    def _do(self, i, x, g):
        b, sp, hx, sx, hy, sy = self.meta[i]
        m = self.m[sp]
        T = x.shape[0]
        self.E[b].add_(x.float().pow(2).sum(-1) * g.float().pow(2).sum(-1))
        if b not in self.buf:
            self.buf[b] = torch.zeros(T, m, device=x.device)
            self.cnt[b] = 0
        S = self.buf[b]
        for s in range(0, T, self.chunk):
            e = min(T, s + self.chunk)
            cx = torch.zeros(e - s, m, device=x.device).index_add_(1, hx, x[s:e].float() * sx)
            cy = torch.zeros(e - s, m, device=x.device).index_add_(1, hy, g[s:e].float() * sy)
            S[s:e] += torch.fft.irfft(torch.fft.rfft(cx) * torch.fft.rfft(cy), n=m)
        self.fired += 1
        self.cnt[b] += 1
        if self.cnt[b] == self.per_block[b]:
            self.on_block(b, sp, self.buf.pop(b), self.E[b])
            self.cnt.pop(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--yarn-factor", type=parse_yarn, required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--m-hi", type=int, default=32768)
    ap.add_argument("--m-lo", type=int, default=8192)
    ap.add_argument("--span", type=int, default=0,
                    help="> 0: loss = CE of the last SPAN targets; rules select context cells (before the span), span "
                         "cells are always kept")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
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
    dev = torch.device("cuda")
    D = {"dn": a.m_hi, "wm": a.m_lo + a.m_hi}
    nblk = {"dn": NB - SPLIT, "wm": NB}
    rules_f = ("row", "blockrow", "random")
    acc = {s: {"F": torch.zeros(D[s], device=dev, dtype=torch.float64), "Fs": torch.zeros(D[s], device=dev, dtype=torch.float64),
               "global": torch.zeros(len(THR), D[s], device=dev, dtype=torch.float64), "global_cnt": np.zeros(len(THR)),
               **{r: torch.zeros(len(FRACS), D[s], device=dev, dtype=torch.float64) for r in rules_f},
               **{f"{r}_cnt": np.zeros(len(FRACS)) for r in rules_f}, "cells": 0} for s in D}
    gen = torch.Generator(device="cpu").manual_seed(0)
    state = {}

    def on_block(b, sp, Sb_all, e_all):
        T = state["s"]
        Sb, e = Sb_all[:T], e_all[:T]
        tot_s = Sb_all[T:].double().sum(0)
        order = e.argsort(descending=True)
        sc = e[order].double().cpu().numpy()
        cg = np.searchsorted(-sc, -THR, side="right")
        kf = np.minimum(T, np.ceil(FRACS * T).astype(int))
        pre_g = prefix_sums([Sb], order, cg)
        pre_f = prefix_sums([Sb], order, kf)
        pre_r = prefix_sums([Sb], torch.randperm(T, generator=gen).to(dev), kf)
        tot = Sb.double().sum(0)
        for s in D:
            if s == "dn" and b < SPLIT:
                continue
            off = 0 if s == "dn" else (0 if sp == 0 else a.m_lo)
            sl = slice(off, off + Sb.shape[1])
            A = acc[s]
            A["F"][sl] += tot
            A["Fs"][sl] += tot_s
            A["global"][:, sl] += pre_g
            A["global_cnt"] += cg
            A["blockrow"][:, sl] += pre_f
            A["blockrow_cnt"] += kf
            A["random"][:, sl] += pre_r
            A["random_cnt"] += kf
            A["cells"] += T
            state[s]["P"][:, sl] += pre_g
            state[s]["cnt"] += cg

    sk = CellSketch(model, a.m_lo, a.m_hi, on_block)
    pg = ParamGrads(model)
    checks, meta = [], []
    t0 = time.time()
    for k, ri in enumerate(rows_idx):
        x = torch.from_numpy(allrows[ri].astype(np.int64)).cuda()
        tgt = torch.from_numpy(masks[ri]).cuda()[1:].clone()
        T = x.numel()
        state["s"] = T - 1 - a.span if a.span else T
        if a.span:
            tgt &= torch.arange(T - 1, device=dev) >= state["s"]
        for s in D:
            state[s] = {"P": torch.zeros(len(THR), D[s], device=dev, dtype=torch.float64), "cnt": np.zeros(len(THR))}
        f_before = {s: (acc[s]["F"] + acc[s]["Fs"]).clone() for s in D}
        pg.enable(True)
        loss, _ = run_pass(model, None, sk, x, tgt, True, lambda b, g: None)
        if sk.xs or sk.gs or sk.buf:
            raise SystemExit(f"unpaired hooks / unfinished blocks: {len(sk.xs)} {len(sk.gs)} {list(sk.buf)}")
        pgr = pg.take()
        pg.enable(False)
        ex_hi = float(pgr[SPLIT:NB, :2].sum())
        sh = float((acc["dn"]["F"] + acc["dn"]["Fs"] - f_before["dn"]).pow(2).sum())
        checks.append(dict(row=ri, hi_ratio=sh / ex_hi))
        # per-row cell quota: the row's threshold grid point whose kept count first reaches f x the row's cells
        for s in D:
            cells = state["s"] * nblk[s]
            cnt = state[s]["cnt"]
            for fi, f in enumerate(FRACS):
                j = int(np.searchsorted(cnt, f * cells, side="left"))
                j = min(j, len(cnt) - 1)
                acc[s]["row"][fi] += state[s]["P"][j]
                acc[s]["row_cnt"][fi] += cnt[j]
        meta.append(dict(row=ri, loss=loss, T=T))
        el = time.time() - t0
        n = k + 1
        log(f"row {n}/{len(rows_idx)} (dev row {ri}) loss {loss:.3f}; sketch/exact norm^2 blocks 7-31 "
            f"{checks[-1]['hi_ratio']:.3f}; {el:.0f}s, ETA {el / n * (len(rows_idx) - n):.0f}s; "
            f"GPU peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB")
    res = dict(label=a.label, ckpt=a.ckpt, dev=a.dev, rows=meta, step_rows=len(rows_idx), T=L, m_hi=a.m_hi, m_lo=a.m_lo,
               split=SPLIT, span=a.span, unit="cells", sketch_check=[dict(c, lo_ratio=float("nan")) for c in checks], budgets={},
               curves={})
    for s, A in acc.items():
        Fall = A["F"] + A["Fs"]
        res[f"{s}_ctx_energy_share"] = float(A["F"].pow(2).sum() / Fall.pow(2).sum().clamp(min=1e-300))
        for rule in ("global",) + rules_f:
            P = A[rule]
            Pt = P + A["Fs"]
            cos = ((Pt @ Fall) / (Pt.norm(dim=1) * Fall.norm()).clamp(min=1e-300)).cpu().numpy()
            cosc = ((P @ A["F"]) / (P.norm(dim=1) * A["F"].norm()).clamp(min=1e-300)).cpu().numpy()
            kept = A[f"{rule}_cnt"] / A["cells"]
            res["curves"][f"{s}_{rule}"] = dict(kept=kept.tolist(), cos=cos.tolist(), cos_ctx=cosc.tolist())
            for thr in (0.9, 0.95, 0.99):
                for nm, cv in (("", cos), ("ctx_", cosc)):
                    ok = np.flatnonzero(cv >= thr)
                    res["budgets"][f"{nm}{s}_{rule}_{thr}"] = float(kept[ok].min()) if ok.size else None
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    for kk, v in res["budgets"].items():
        log(f"budget {kk}: {v}")
    log(f"done in {time.time() - t0:.0f}s -> {a.out}.json")


if __name__ == "__main__":
    main()
