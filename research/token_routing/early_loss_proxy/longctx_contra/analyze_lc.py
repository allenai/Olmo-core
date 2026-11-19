"""Tables for the longctx_contra arms from checkpoint-then-score results (README here).

    python3 analyze_lc.py --res /tmp/claude-3018/elp_beaker_results [--out results.json]

Inputs (fetch_beaker_results.sh <tag>): training JSONs lcA-<arm>/<run>.json (per-step cumulative actual PF from the FLOP
meter) and scoring JSONs lcS-<arm>/score[1m]-<arm>-s<step>.json + lcS-base/score[1m]-base-s0.json (per-row CE at
probe step 0 = the loaded checkpoint). Per held-out set: CE vs PF per arm; each gold-drop arm interpolated (linear in
PF between neighbouring checkpoints) at the dense arm's checkpoint PFs, with the paired per-row difference and its SE
(rows are the same held-out rows in both arms; interpolation applied per row); FLOPs-to-match: the PF at which the
gold-drop curve first reaches dense's CE at each dense checkpoint (linear interpolation), and dense PF / that.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

PAIRS = {"256k": ("dense256b", "gdd256"), "1m": ("dense1m", "gdd1m")}
RUNS = {"dense256b": "lcA-dense256/dense256b.json", "gdd256": "lcA-gdd256/gdd256.json",
        "dense1m": "lcA-dense1m/dense1m.json", "gdd1m": "lcA-gdd1m/gdd1m.json"}


def step_pf(res, arm):
    p = os.path.join(res, RUNS[arm])
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    return {int(s["step"]): float(s["flops"]["actual_pf"]) for s in d["steps"] if s.get("flops")}


def pf_at(table, step):
    if step == 0:
        return 0.0
    if step in table:
        return table[step]
    ks = sorted(table)
    lo = max([k for k in ks if k <= step], default=None)
    hi = min([k for k in ks if k >= step], default=None)
    if lo is None or hi is None:
        return float("nan")
    return table[lo] + (table[hi] - table[lo]) * (step - lo) / max(1, hi - lo)


def scores(res, arm):
    """{step: {set: [per-row CE]}} for an arm (+ the shared base at step 0)."""
    out = {}
    files = glob.glob(os.path.join(res, f"lcS-{arm}", "score*-s*.json")) + glob.glob(os.path.join(res, "lcS-base", "score*-s0.json"))
    for f in files:
        m = re.search(r"-s(\d+)\.json$", f)
        st = int(m.group(1))
        d = json.load(open(f))
        if not d.get("probes"):
            continue
        ce = d["probes"][0]["ce"]
        for r, c in zip(d["rows"], ce):
            out.setdefault(st, {}).setdefault(r["set"], []).append(c)
    return out


def mean(x):
    return sum(x) / len(x)


def se_paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    m = mean(d)
    return math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)) if len(d) > 1 else float("nan")


def interp_rows(curve, pf):
    """curve: sorted [(pf, rows)]; rows at pf by per-row linear interpolation; also return the neighbours."""
    for (p0, r0), (p1, r1) in zip(curve, curve[1:]):
        if p0 <= pf <= p1:
            w = (pf - p0) / (p1 - p0) if p1 > p0 else 0.0
            return [a + w * (b - a) for a, b in zip(r0, r1)], (p0, p1)
    return None, None


def first_reach(curve, target):
    """Smallest PF at which the mean-CE curve reaches <= target (linear between checkpoints)."""
    pts = [(p, mean(r)) for p, r in curve]
    if pts and pts[0][1] <= target:
        return pts[0][0]
    for (p0, c0), (p1, c1) in zip(pts, pts[1:]):
        if c1 <= target < c0:
            return p0 + (p1 - p0) * (c0 - target) / (c0 - c1)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", default="/tmp/claude-3018/elp_beaker_results")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    report = {}
    for L, (dn, gd) in PAIRS.items():
        pfs = {arm: step_pf(a.res, arm) for arm in (dn, gd)}
        sc = {arm: scores(a.res, arm) for arm in (dn, gd)}
        sets = sorted({s for arm in sc for st in sc[arm] for s in sc[arm][st]})
        print(f"\n===== {L}: {dn} vs {gd} =====")
        rep = report.setdefault(L, {})
        for S in sets:
            curves = {}
            for arm in (dn, gd):
                c = sorted((pf_at(pfs[arm], st), sc[arm][st][S]) for st in sc[arm] if S in sc[arm][st]
                           if not math.isnan(pf_at(pfs[arm], st)))
                curves[arm] = c
            n = len(curves[dn][0][1]) if curves[dn] else 0
            flag = "" if n >= 500 else f"  (eval_size={n} < 500)"
            print(f"\n-- held-out {S}{flag}")
            for arm in (dn, gd):
                print(f"   {arm:10s} " + "  ".join(f"{p:7.0f}PF:{mean(r):.4f}" for p, r in curves[arm]))
            rows = []
            for p, r in curves[dn]:
                if p == 0:
                    continue
                gi, nb = interp_rows(curves[gd], p)
                pg = first_reach(curves[gd], mean(r))
                row = {"dense_pf": p, "dense_ce": mean(r), "gdd_ce_at_pf": mean(gi) if gi else None,
                       "neighbours": nb, "diff_gdd_minus_dense": (mean(gi) - mean(r)) if gi else None,
                       "se_paired": se_paired(gi, r) if gi else None,
                       "gdd_pf_to_reach_dense_ce": pg, "speedup": (p / pg) if pg else None}
                rows.append(row)
                print(f"   dense {p:7.0f} PF CE {mean(r):.4f} | gold drop at same PF: "
                      + (f"{mean(gi):.4f} (diff {row['diff_gdd_minus_dense']:+.4f} +- {row['se_paired']:.4f}; "
                         f"between ckpts {nb[0]:.0f}/{nb[1]:.0f} PF)" if gi else "n/a (outside its scored range)")
                      + (f" | gold drop reaches this CE at {pg:.0f} PF -> {p / pg:.2f}x" if pg else " | not reached"))
            rep[S] = {"eval_size": n, "curves": {k: [(p, mean(r)) for p, r in v] for k, v in curves.items()}, "matched": rows}
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
