"""
Large-data decay diagnosis: FLOPs-to-match (FTM) of the token-drop recipe vs dense on (a) the 8-cell generative f1
mean, (b) the unsaturated f1 cells only (oolong + outlier, 8k + 2k), (c) held-out answer CE (score_answer_ce.py).

FTM(point) = the dense FLOPs needed to reach the point's metric, read off the dense fresh-pool curve
(f1952 / f3904 / f5856 / f7808 rows/task, lr 5e-6) by linear interpolation in log(PFLOPs), divided by the point's
own FLOPs. A metric better than every dense point is reported as ``>`` the ratio at dense's best point.

    python research/token_routing/decay_fix/analyze_decay.py --ce ce1.json [--ce ce2.json ...] \\
        [--points /path/to/pareto_points.json] [--extra-runs run1,run2]
"""
import argparse
import json
import math
import os

DENSE = [f"f8k-mt4-dense-f{r}-tde-lr5e-06" for r in (1952, 3904, 5856, 7808)]
RECIPE = [f"f8k-mt4-lora64all32r20to7-f{r}-tde-lr1e-05" for r in (1952, 3904, 7808)]
CELLS = [f"{t}_{r}" for t in ("contradiction", "nq", "oolong", "outlier") for r in ("8k", "2k")]
UNSAT = [f"{t}_{r}" for t in ("oolong", "outlier") for r in ("8k", "2k")]
DEFAULT_POINTS = ("/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/research/token_routing/ds64_fast8k/"
                  "results/pareto_points.json")


def ftm(dense_xy, x, y, higher_better=True):
    """dense_xy: [(pf, metric)] sorted by pf. Returns (dense pf at metric y) / x, with a '>'/'<' flag when outside."""
    pts = sorted(dense_xy)
    s = 1.0 if higher_better else -1.0
    ys = [s * m for _, m in pts]
    yy = s * y
    if yy > max(ys):
        best = max(range(len(pts)), key=lambda i: ys[i])
        return pts[best][0] / x, ">"
    if yy <= ys[0]:
        return pts[0][0] / x, "<"
    for (p0, m0), (p1, m1) in zip(pts, pts[1:]):
        a, b = s * m0, s * m1
        if min(a, b) <= yy <= max(a, b) and a != b and b > a:
            t = (yy - a) / (b - a)
            return math.exp(math.log(p0) + t * (math.log(p1) - math.log(p0))) / x, ""
    # non-monotone dense curve: first crossing from the left
    for (p0, m0), (p1, m1) in zip(pts, pts[1:]):
        a, b = s * m0, s * m1
        if a <= yy <= b and a != b:
            t = (yy - a) / (b - a)
            return math.exp(math.log(p0) + t * (math.log(p1) - math.log(p0))) / x, "~"
    return float("nan"), "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce", action="append", default=[])
    ap.add_argument("--points", default=DEFAULT_POINTS)
    ap.add_argument("--extra-runs", default="", help="more method runs (pf from --pf name=PF if not in points)")
    ap.add_argument("--pf", action="append", default=[], help="name=PFLOPs override")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()
    P = json.load(open(a.points))
    pf = {k: v["pf"] for k, v in P["runs"].items()}
    for kv in a.pf:
        k, v = kv.split("=")
        pf[k] = float(v)
    f1 = {}
    for p in P["points"]:
        f1.setdefault(p["run"], {})[f"{p['task']}_{p['rung']}"] = p["score"]
    ce = {}
    for path in a.ce:
        d = json.load(open(path))
        for k, v in d.items():
            if not k.startswith("_") and isinstance(v, dict):
                ce[k] = v
    methods = RECIPE + [r for r in a.extra_runs.split(",") if r]
    out = {"dense": {}, "methods": {}}

    def mean(d, keys):
        vals = [d.get(k) for k in keys]
        return sum(vals) / len(vals) if all(v is not None for v in vals) else None

    def ce_mean(name, keys, field="ans_row_mean"):
        c = ce.get(name)
        if not c:
            return None
        vals = [c.get(k, {}).get(field) for k in keys]
        return sum(vals) / len(vals) if all(v is not None for v in vals) else None

    metrics = {
        "f1_8cell": (lambda n: mean(f1.get(n, {}), CELLS), True),
        "f1_unsat4": (lambda n: mean(f1.get(n, {}), UNSAT), True),
        "ce_8cell": (lambda n: ce_mean(n, CELLS), False),
        "ce_unsat4": (lambda n: ce_mean(n, UNSAT), False),
        "ce_tok_8cell": (lambda n: ce_mean(n, CELLS, "ans_tok_mean"), False),
    }
    for c in CELLS:
        metrics[f"ce_{c}"] = ((lambda cc: (lambda n: ce_mean(n, [cc])))(c), False)
    print(f"{'run':52s} {'PF':>8s} " + " ".join(f"{m:>14s}" for m in metrics))
    for n in DENSE:
        row = {m: f(n) for m, (f, _) in metrics.items()}
        out["dense"][n] = {"pf": pf.get(n), **row}
        print(f"{n:52s} {pf.get(n, float('nan')):8.0f} " + " ".join(
            f"{v:14.4f}" if v is not None else f"{'-':>14s}" for v in row.values()))
    for n in methods:
        row = {}
        for m, (f, hb) in metrics.items():
            v = f(n)
            dxy = [(pf[d], f(d)) for d in DENSE if f(d) is not None and d in pf]
            r, flag = (ftm(dxy, pf[n], v, hb) if v is not None and len(dxy) >= 2 and n in pf else (None, ""))
            row[m] = (v, r, flag)
        out["methods"][n] = {"pf": pf.get(n), **{m: {"value": v, "ftm": r, "flag": fl} for m, (v, r, fl) in row.items()}}
        print(f"{n:52s} {pf.get(n, float('nan')):8.0f} " + " ".join(
            (f"{v:7.4f} {fl}{r:4.1f}x" if v is not None and r is not None else f"{'-':>14s}") for v, r, fl in row.values()))
    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
