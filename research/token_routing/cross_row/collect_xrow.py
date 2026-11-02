"""Cross-row arms vs the recipe and fresh dense at the 8k rung (primary: 4-task mean at 8k; 2k mean secondary).

Scores from ops results (record first: ``ds64_fast8k/record_beaker_evals.py --state research/token_routing/cross_row/
state.json --suffix -tdx --out <rows.jsonl>``); actual PF, realised keep and rows back-propagated from each training
run's Beaker log. FLOPs-to-match (FTM) = PF the fresh dense curve needs for the run's 8k mean (linear in log PF between
measured points, the first segment extended below 0.843) / the run's PF.

    python research/token_routing/cross_row/collect_xrow.py [--json out.json]
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
#: (PF, 8k 4-task mean): fresh dense at 1,952 / 3,904 / 5,856 / 7,808 rows/task; recipe at 1,952 / 3,904 (parent brief)
DENSE8 = [(1531.0, 0.843), (3063.0, 0.899), (4595.0, 0.901), (6128.0, 0.904)]
RECIPE8 = [(345.0, 0.845), (688.0, 0.882)]
TASKS = ("contradiction", "nq", "outlier", "oolong")


def pf_to_match(curve, score):
    for i, ((p0, y0), (p1, y1)) in enumerate(zip(curve, curve[1:])):
        if (y0 <= score <= y1 or (i == 0 and score < y0)) and y1 > y0:
            return math.exp(math.log(p0) + (math.log(p1) - math.log(p0)) * (score - y0) / (y1 - y0))
    return None


def scores(run):
    out = subprocess.run([OPS, "result", "show", "--train-run", run, "--json"], capture_output=True, text=True,
                         timeout=120).stdout
    rows = json.loads(out) if out.strip().startswith("[") else []
    best = {}
    for r in rows:
        k = (r["task"], r["rung"])
        if not str(r.get("run_id", "")).startswith(run + "-ev") or r["task"] not in TASKS:
            continue  # only the ladder eval jobs (<run>-ev<task>), not answer-CE / probe rows on the same run
        if k not in best or (r.get("eval_size") or 0) > best[k][1]:
            best[k] = (float(r["value"]), int(r.get("eval_size") or 0))
    return best


def beaker_log_stats(ex):
    log = subprocess.run([BEAKER, "experiment", "logs", ex], capture_output=True, text=True, timeout=600).stdout
    last = lambda pat: (float(m[-1]) if (m := re.findall(pat, log)) else None)  # noqa: E731
    mean = lambda pat: (sum(map(float, m)) / len(m) if (m := re.findall(pat, log)) else None)  # noqa: E731
    return dict(pf=last(r"'actual_pflops': ([0-9.]+)"), dense_pf=last(r"'dense_equivalent_pflops': ([0-9.]+)"),
                bp_frac=last(r"train/sbp/backprop_frac_cum\s+([0-9.]+)"),
                keep1=mean(r"\[online-router\] step \d+: .*? keep_frac=([0-9.]+)"),
                keep2=mean(r"\[online-router\] step \d+: .*? s1_keep_frac=([0-9.]+)"),
                row_corr=mean(r" step_row_corr=(-?[0-9.]+)"), row_cv=mean(r" step_row_keep_cv=([0-9.]+)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--extra-runs", default="", help="comma list of train_run=beaker_ex (e.g. recipe runs) to include")
    a = ap.parse_args()
    st = json.load(open(os.path.join(HERE, "state.json")))
    runs = {n: r["ex"] for n, r in st["runs"].items() if r.get("ex") and not r.get("smoke")}
    for kv in filter(None, a.extra_runs.split(",")):
        n, _, ex = kv.partition("=")
        runs[n] = ex
    out = []
    print(f"{'run':52s} {'PF':>6s} {'x/rcp':>6s} {'8k':>6s} {'2k':>6s} {'FTM8':>6s} {'bp':>5s} {'k1':>5s} {'k2':>5s}  cells")
    for run, ex in runs.items():
        s = beaker_log_stats(ex)
        sc = scores(run)
        m8 = [sc[(t, "8k")][0] for t in TASKS if (t, "8k") in sc]
        m2 = [sc[(t, "2k")][0] for t in TASKS if (t, "2k") in sc]
        mean8 = sum(m8) / 4 if len(m8) == 4 else None
        mean2 = sum(m2) / 4 if len(m2) == 4 else None
        ftm = pf_to_match(DENSE8, mean8) if mean8 is not None else None
        rows = int(re.search(r"-f(\d+)-", run).group(1))
        rec_pf = dict((r, p) for r, (p, _) in zip((1952, 3904), RECIPE8)).get(rows)
        row = dict(run=run, ex=ex, rows_per_task=rows, mean8=mean8, mean2=mean2,
                   cells={f"{t}@{g}": sc[(t, g)][0] for (t, g) in sc}, eval_sizes=sorted({v[1] for v in sc.values()}),
                   pf_over_recipe=(s["pf"] / rec_pf) if (s["pf"] and rec_pf) else None,
                   dense_pf_to_match8=ftm, ftm8=(ftm / s["pf"]) if (ftm and s["pf"]) else None, **s)
        out.append(row)
        f = lambda v, w=6, d=3: (f"{v:{w}.{d}f}" if isinstance(v, float) else f"{'-':>{w}s}")  # noqa: E731
        print(f"{run:52s} {f(s['pf'], 6, 0)} {f(row['pf_over_recipe'], 6, 2)} {f(mean8)} {f(mean2)} {f(row['ftm8'], 6, 2)}"
              f" {f(s['bp_frac'], 5, 2)} {f(s['keep1'], 5, 3)} {f(s['keep2'], 5, 3)}  {len(sc)}/8")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
