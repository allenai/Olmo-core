"""Comparison table from collect_val.py's results.json: aggregates with and without reorder (reorder scores ~0 with
low parse rates for every arm incl. dense full FT: the known strict-permutation-grader failure, no signal), recipe
seed mean, FTM on the dense-LoRA f4/f2/f1 curve. Writes table.json.

    python3 research/token_routing/early_loss_proxy/validation/table_val.py
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from collect_val import ftm, se  # noqa: E402

R = json.load(open(os.path.join(HERE, "results.json")))
CURVE = ["elpv-denselora-f4", "elpv-denselora-f2", "elpv-denselora-f1"]
TASKS3 = ["contradiction", "outlier", "qdmatch_nq"]
TASKS4 = TASKS3 + ["reorder"]
RUNGS = ["4k", "8k", "16k"]


def agg(run, keys):
    cs = [R["cells"][run][f"{t}@{g}"] for t, g in keys]
    m = sum(c["value"] for c in cs) / len(cs)
    s = math.sqrt(sum(se(c["value"], c["eval_size"]) ** 2 for c in cs)) / len(cs)
    return m, s


AGGS = {"all12": [(t, g) for t in TASKS4 for g in RUNGS], "noreorder9": [(t, g) for t in TASKS3 for g in RUNGS],
        **{f"8k_{n}": [(t, "8k") for t in ts] for n, ts in (("4t", TASKS4), ("3t", TASKS3))},
        **{f"16k_{n}": [(t, "16k") for t in ts] for n, ts in (("4t", TASKS4), ("3t", TASKS3))},
        **{f"4k_3t": [(t, "4k") for t in TASKS3]},
        **{f"task_{t}": [(t, g) for g in RUNGS] for t in TASKS4}}
pf = {r: R["train"][r]["pf"] for r in R["train"]}
out = {"pf": pf, "aggs": {}, "ftm": {}}
for run in R["cells"]:
    out["aggs"][run] = {k: [round(x, 4) for x in agg(run, v)] for k, v in AGGS.items()}
# recipe seed mean at f1 (both seeds at ~441.5 PF)
out["aggs"]["recipe-f1-mean2"] = {k: [round((out["aggs"]["elpv-recipe-f1"][k][0] + out["aggs"]["elpv-recipe-f1-sd1"][k][0]) / 2, 4),
                                      round(math.hypot(out["aggs"]["elpv-recipe-f1"][k][1], out["aggs"]["elpv-recipe-f1-sd1"][k][1]) / 2, 4)]
                                  for k in AGGS}
pf["recipe-f1-mean2"] = (pf["elpv-recipe-f1"] + pf["elpv-recipe-f1-sd1"]) / 2
for run in ["elpv-recipe-f1", "elpv-recipe-f1-sd1", "recipe-f1-mean2", "elpv-recipe-f2", "elpv-dense-f1"]:
    out["ftm"][run] = {}
    for k in AGGS:
        curve = [(pf[c], out["aggs"][c][k][0]) for c in CURVE]
        x, q = ftm(curve, pf[run], out["aggs"][run][k][0])
        out["ftm"][run][k] = f"{q}{x:.2f}" if q != "=" else f"{x:.2f}"
json.dump(out, open(os.path.join(HERE, "table.json"), "w"), indent=1)
keys = ["all12", "noreorder9", "8k_3t", "16k_3t", "4k_3t", "task_contradiction", "task_outlier", "task_qdmatch_nq",
        "task_reorder"]
print("run".ljust(22), "PF".rjust(7), *[k[:12].rjust(13) for k in keys])
for run in CURVE + ["elpv-dense-f1", "elpv-recipe-f2", "elpv-recipe-f1", "elpv-recipe-f1-sd1", "recipe-f1-mean2"]:
    print(run.ljust(22), f"{pf[run]:7.1f}", *[f"{out['aggs'][run][k][0]:.3f}+-{out['aggs'][run][k][1]:.3f}".rjust(13) for k in keys])
print("FTM vs dense-LoRA curve")
for run, d in out["ftm"].items():
    print(run.ljust(22), " ".join(f"{k}={d[k]}" for k in keys))
