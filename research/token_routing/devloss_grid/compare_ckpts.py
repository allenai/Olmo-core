"""Compare two checkpoints' dev-loss grids cell by cell (e.g. Qwen3.5 setA32k vs Qwen3 setA, same rows and schemes).

    python3 research/token_routing/devloss_grid/compare_ckpts.py --a <res_dir_A> --b <res_dir_B> [--names Qwen3.5,Qwen3]

Per scheme and (task, rung): paired ΔCE = mean over rows of (CE_scheme - CE_full) on the SAME rows, ± its SE, for each
checkpoint, and the difference ΔCE_B - ΔCE_A (± sqrt(SE_A² + SE_B²); rows are the same examples but the models differ,
so the two ΔCEs are treated as independent). Summary per scheme: tasks at parity (ΔCE ≤ 0.02 / 0.05 / 0.10), median
ΔCE, worst task, mean compaction -- the dev-loss-grid artifact's summary columns.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np


def load_dir(d):
    cells = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        j = json.load(open(f))
        task = j["task"].replace("ctc_", "")
        cells[(task, j["rung"])] = j
    return cells


def dce(cell, s):
    """(mean, se, n, compaction, ce_full_mean) of the paired ΔCE of scheme s vs full in one cell."""
    pr = cell["per_row"]
    if s not in pr or "full" not in pr:
        return None
    a, b = np.asarray(pr[s]["ce"], float), np.asarray(pr["full"]["ce"], float)
    m = ~(np.isnan(a) | np.isnan(b))
    d = a[m] - b[m]
    if not d.size:
        return None
    comp = float("nan")
    for k in ("ratio", "compaction", "t2_over_t", "T2_over_T"):
        if k in pr[s]:
            comp = float(np.nanmean(pr[s][k]))
            break
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(d.size)) if d.size > 1 else float("nan"), int(d.size), comp, float(b[m].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--names", default="A,B")
    ap.add_argument("--rungs", default="2k,8k,32k")
    a = ap.parse_args()
    na, nb = a.names.split(",")
    A, B = load_dir(a.a), load_dir(a.b)
    keys = sorted(set(A) & set(B), key=lambda k: (k[0], ["2k", "8k", "16k", "32k"].index(k[1]) if k[1] in ("2k", "8k", "16k", "32k") else 9))
    schemes = [s for s in next(iter(A.values()))["per_row"] if s != "full"]
    print(f"cells in both: {len(keys)}  ({', '.join(f'{t}@{r}' for t, r in keys)})\n")
    print(f"full-context CE (mean over rows): " + "; ".join(
        f"{t}@{r} {dce(A[(t, r)], schemes[0])[4]:.3f}/{dce(B[(t, r)], schemes[0])[4]:.3f}" for t, r in keys if dce(A[(t, r)], schemes[0]) and dce(B[(t, r)], schemes[0])))
    print(f"(each pair = {na}/{nb})\n")
    for rung in a.rungs.split(",") + ["all"]:
        ks = [k for k in keys if rung == "all" or k[1] == rung]
        if not ks:
            continue
        print(f"### summary, rung {rung}: tasks at parity (ΔCE ≤ 0.02 | ≤ 0.05 | ≤ 0.10), median ΔCE, worst cell — {na} vs {nb} "
              f"(⚠ eval_size 32-48 rows per cell)")
        print(f"| scheme | {na} parity | {nb} parity | {na} median ΔCE | {nb} median ΔCE | {na} worst | {nb} worst | compaction |")
        print("|---|---|---|---|---|---|---|---|")
        for s in schemes:
            row = []
            for C in (A, B):
                v = {k: dce(C[k], s) for k in ks if dce(C[k], s)}
                ds = np.array([x[0] for x in v.values()])
                worst = max(v.items(), key=lambda kv: kv[1][0]) if v else None
                row.append((v, ds, worst))
            if not row[0][0] or not row[1][0]:
                continue
            par = lambda ds: f"{(ds <= 0.02).sum()} \\| {(ds <= 0.05).sum()} \\| {(ds <= 0.10).sum()} / {len(ds)}"
            wr = lambda w: f"{w[0][0]}@{w[0][1]} {w[1][0]:+.3f}"
            comp = np.nanmean([x[3] for x in row[0][0].values()])
            print(f"| `{s}` | {par(row[0][1])} | {par(row[1][1])} | {np.median(row[0][1]):+.3f} | {np.median(row[1][1]):+.3f} | "
                  f"{wr(row[0][2])} | {wr(row[1][2])} | ×{comp:.2f} |")
        print()
    print(f"### per cell: ΔCE ± SE for {na} | {nb} | difference ({nb} − {na})")
    for s in schemes:
        print(f"\n#### `{s}`\n| task@rung | {na} ΔCE | {nb} ΔCE | {nb} − {na} |\n|---|---|---|---|")
        for k in keys:
            x, y = dce(A[k], s), dce(B[k], s)
            if not x or not y:
                continue
            se = math.sqrt(x[1] ** 2 + y[1] ** 2)
            print(f"| {k[0]}@{k[1]} (rows {x[2]}) | {x[0]:+.3f} ± {x[1]:.3f} | {y[0]:+.3f} ± {y[1]:.3f} | {y[0] - x[0]:+.3f} ± {se:.3f} |")


if __name__ == "__main__":
    main()
