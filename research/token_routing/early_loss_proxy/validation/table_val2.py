"""Phase-2 table (loop/validate2) from ``ELPV_PHASE=2 collect_val.py`` -> results2.json: per run steps, flop-meter PF,
proxy 16k CE (record flopdrop-dynamic-drop-budget-matched.md), generation score per task at 16k and the 3-task
(contradiction / outlier / qdmatch_nq) mean at 4k / 8k / 16k with binomial SE (sqrt(sum se^2) / cells), parse rates.
No-cot, native docchunk ladder on CTC-suite rows, query after, eval_size per cell as logged. Writes table2.json.

    python3 research/token_routing/early_loss_proxy/validation/table_val2.py
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results2.json")))
TASKS3 = ["contradiction", "outlier", "qdmatch_nq"]
RUNGS = ["4k", "8k", "16k"]
PROXY = {"elpv2-p50-x4-s1683": 0.074, "elpv2-denselora-x4-s642": 0.103, "elpv2-dense-x4-s976": 0.081,
         "elpv2-denselora-x16-s1293": 0.078}


def se(p, n):
    return math.sqrt(max(p * (1 - p), 0.0) / n)


out = {}
for run in PROXY:
    t = R["train"].get(run) or {}
    cs = R["cells"].get(run, {})
    row = {"steps": t.get("steps"), "pf": t.get("pf"), "final_train_ce_last10": t.get("final_train_ce_last10"),
           "proxy_16k_ce": PROXY[run], "cells": {}, "mean3": {}}
    for k, c in cs.items():
        row["cells"][k] = {"value": c["value"], "eval_size": c["eval_size"], "se": round(se(c["value"], c["eval_size"]), 4)
                           if c["value"] is not None else None, "parse_rate": c["parse_rate"], "metric": c["metric"]}
    for g in RUNGS:
        got = [cs.get(f"{t_}@{g}") for t_ in TASKS3]
        if all(c and c["value"] is not None for c in got):
            m = sum(c["value"] for c in got) / 3
            s = math.sqrt(sum(se(c["value"], c["eval_size"]) ** 2 for c in got)) / 3
            row["mean3"][g] = [round(m, 4), round(s, 4)]
    out[run] = row
json.dump(out, open(os.path.join(HERE, "table2.json"), "w"), indent=1)
print(f"{'run':28s} {'steps':>5s} {'PF':>7s} {'proxy':>6s} " + " ".join(f"{t[:6] + '@16k':>18s}" for t in TASKS3)
      + " " + " ".join(f"{'mean3@' + g:>16s}" for g in RUNGS))
for run, r in out.items():
    cells = " ".join((lambda c: f"{c['value']:.3f}+-{c['se']:.3f} pr{c['parse_rate']}" if c and c["value"] is not None
                      else "-")(r["cells"].get(f"{t}@16k")).rjust(18) for t in TASKS3)
    means = " ".join((f"{r['mean3'][g][0]:.3f}+-{r['mean3'][g][1]:.3f}" if g in r["mean3"] else "-").rjust(16)
                     for g in RUNGS)
    pf = f"{r['pf']:.1f}" if r["pf"] else "-"
    print(f"{run:28s} {str(r['steps']):>5s} {pf:>7s} {r['proxy_16k_ce']:6.3f} {cells} {means}")
for run, r in out.items():
    print(run, {k: (c["value"], c["eval_size"], c["parse_rate"]) for k, c in sorted(r["cells"].items())})
