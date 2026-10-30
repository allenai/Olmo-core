"""Multi-view runs vs the recipe and fresh dense at matched FLOPs (mt4 8-cell mean: 4 tasks x {8k, 2k}, 500 rows each).

Scores from ops results (record them first: ``record_beaker_evals.py --state research/token_routing/multiview/state.json
--suffix -tdmv --out <rows.jsonl>``), actual PF from each training run's Beaker log (``flop_meter/actual_pflops``,
last value). The recipe / dense curves are linear in log(PF) between their measured points; outside the curve's range
the comparison is an extrapolation and is flagged.

    python research/token_routing/multiview/collect_mv.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
BEAKER = os.path.expanduser("~/.local/bin/beaker")
#: (PF, 8-cell mean) -- recipe lora64all32r20to7 -tde f1952/f3904/f7808 and fresh dense (tokdrop_lr STATUS, 10-28/30)
RECIPE = [(345.0, 0.883), (688.0, 0.915), (1373.0, 0.924)]
DENSE = [(1531.0, 0.888), (3063.0, 0.926), (4595.0, 0.928), (6128.0, 0.931)]
CELLS = [(t, r) for t in ("contradiction", "nq", "outlier", "oolong") for r in ("8k", "2k")]


def interp(curve, pf):
    """Score of ``curve`` at ``pf`` (linear in log PF); second value says whether it extrapolated."""
    xs = [math.log(p) for p, _ in curve]
    x = math.log(pf)
    i = 0 if x <= xs[1] else (len(xs) - 2 if x >= xs[-2] else max(j for j in range(len(xs) - 1) if xs[j] <= x))
    (x0, y0), (x1, y1) = (xs[i], curve[i][1]), (xs[i + 1], curve[i + 1][1])
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0), not (xs[0] <= x <= xs[-1])


def pf_to_match(curve, score):
    """PF the curve needs to reach ``score`` (log-linear; None if above its last point)."""
    for (p0, y0), (p1, y1) in zip(curve, curve[1:]):
        if y0 <= score <= y1 and y1 > y0:
            return math.exp(math.log(p0) + (math.log(p1) - math.log(p0)) * (score - y0) / (y1 - y0))
    return None


def scores(run):
    out = subprocess.run([OPS, "result", "show", "--train-run", run, "--json"], capture_output=True, text=True,
                         timeout=120).stdout
    rows = json.loads(out) if out.strip().startswith("[") else []
    best = {}
    for r in rows:
        k = (r["task"], r["rung"])
        if k not in best or (r.get("eval_size") or 0) > best[k][1]:
            best[k] = (float(r["value"]), int(r.get("eval_size") or 0))
    return best


def beaker_pf(ex):
    log = subprocess.run([BEAKER, "experiment", "logs", ex], capture_output=True, text=True, timeout=300).stdout
    pf = re.findall(r"flop_meter/actual_pflops[ =]+([0-9.]+)", log)
    ratio = re.findall(r"flop_meter/actual_over_dense[ =]+([0-9.]+)", log)
    return (float(pf[-1]) if pf else None), (float(ratio[-1]) if ratio else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    st = json.load(open(os.path.join(HERE, "state.json")))
    out = []
    print(f"{'run':58s} {'PF':>7s} {'act/dense':>9s} {'mean8':>6s} {'recipe@PF':>9s} {'delta':>7s} {'FTM-dense':>9s}  cells")
    for run, r in st["runs"].items():
        if not r.get("ex"):
            continue
        pf, ratio = beaker_pf(r["ex"])
        sc = scores(run)
        have = [c for c in CELLS if c in sc]
        mean = sum(sc[c][0] for c in have) / len(have) if len(have) == len(CELLS) else None
        rec = interp(RECIPE, pf) if pf else (None, False)
        ftm = pf_to_match(DENSE, mean) if mean is not None else None
        row = dict(run=run, views=r["views"], rows=r["rows"], pf=pf, actual_over_dense=ratio, mean8=mean,
                   cells={f"{t}@{g}": sc[(t, g)][0] for t, g in have}, eval_sizes=sorted({sc[c][1] for c in have}),
                   recipe_at_pf=rec[0], recipe_extrapolated=rec[1],
                   delta_vs_recipe=(mean - rec[0]) if (mean is not None and rec[0] is not None) else None,
                   dense_pf_to_match=ftm, ftm_x=(ftm / pf) if (ftm and pf) else None)
        out.append(row)
        f = lambda v, w=7, d=3: (f"{v:{w}.{d}f}" if isinstance(v, float) else f"{'-':>{w}s}")
        print(f"{run:58s} {f(pf, 7, 0)} {f(ratio, 9, 3)} {f(mean, 6)} {f(rec[0], 9)}{'*' if rec[1] else ' '}"
              f"{f(row['delta_vs_recipe'])} {f(row['ftm_x'], 9, 2)}  {len(have)}/8")
    print("* recipe curve extrapolated beyond its measured PF range (345-1373 PF)")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
