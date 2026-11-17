"""Doc-drop results table (records/docdrop-nongold-baseline.md): one row per run in state.json (full-length runs).

Scores: ``ops result show --train-run`` (4 tasks x 8k / 2k, eval_size 500 each). Cost and memory: the node log
(``/net/<node>/data/prasann/joblogs/dd-<arm>-<job>.out``, inspection only): cumulative ``flop_meter/actual_pflops``
and peak allocated GPU memory per rank. Step cost = run PF / the dense run's PF at the same rows (full fine-tune,
LR 5e-6, same rows / steps); best-case per-step speedup = its inverse. FTM (8k-only) is read log-linearly off the
dense curve (``collect_wts.ftm``), an upper bound below its first point.

    python3 research/token_routing/docdrop/collect_docdrop.py [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "wts"))
from collect_wts import TASKS, ftm, mean8, parse_log, scores  # noqa: E402

#: dense full fine-tune (LR 5e-6) on the fresh pool: (rows/task, PF, 8k 4-task mean) -- records/tokdrop-handoff and
#: the wts record (dense f1952 PF 1531 from its Beaker log; f488 / f976 / f3904 PF from the oracle-threshold record)
DENSE_CURVE = {488: (379.0, 0.667), 976: (762.0, 0.712), 1952: (1531.0, 0.843), 3904: (3062.0, 0.899)}
REFS_8K = {"dense": 0.843, "recipe": (0.846, 0.847), "random token keep @7, 19.7%": 0.841,
           "random token keep @7, 11.7%": 0.796, "tail rule": 0.781}


def stats(run, info):
    files = glob.glob(f"/net/{info['node']}/data/prasann/joblogs/dd-{info['arm']}-{info['job']}.out")
    if not files:
        return {}
    out = parse_log(open(files[0], errors="ignore").read())
    out["done"] = out.get("done") or "DOCDROP_DONE" in open(files[0], errors="ignore").read()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    st = json.load(open(os.path.join(HERE, "state.json")))
    curve = [v for v in DENSE_CURVE.values()]
    rows = []
    for run, info in st["runs"].items():
        if info.get("steps"):
            continue
        rows_per_task = int(info["budget"].lstrip("f"))
        dense_pf = DENSE_CURVE.get(rows_per_task, (None,))[0]
        s, t = scores(run), stats(run, info)
        m8, m2 = mean8(s), mean8(s, "2k")
        f = ftm(m8, t.get("pf"), curve)
        cost = t["pf"] / dense_pf if t.get("pf") and dense_pf else None
        rows.append({"run": run, "arm": info["arm"], "seed": info["seed"], "rows_per_task": rows_per_task,
                     "pf": t.get("pf"), "step_cost": cost, "speedup": 1 / cost if cost else None,
                     "peak_gib": t.get("peak_gib"), "steps": t.get("steps"), "done": t.get("done"),
                     **{f"{k}8k": (s.get((k, "8k")) or (None,))[0] for k in TASKS},
                     **{f"{k}2k": (s.get((k, "2k")) or (None,))[0] for k in TASKS},
                     "mean8k": m8, "mean2k": m2, "ftm8k": (f"{f[0]}{f[1]:.2f}x" if f else None),
                     "eval_sizes": sorted({v[1] for v in s.values()})})
    fmt = lambda x, n=3: "-" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))  # noqa: E731
    print("| run | PF | step cost | speedup/step | peak GiB | contra | nq | outlier | oolong | 8k mean | 2k mean | FTM 8k "
          "| eval_size |")
    print("|" + "---|" * 13)
    for x in rows:
        print(f"| {x['run']} | {fmt(x['pf'], 1)} | {fmt(x['step_cost'])} | {fmt(x['speedup'], 2)} | "
              f"{fmt(x['peak_gib'], 1)} | " + " | ".join(fmt(x[f'{k}8k']) for k in TASKS)
              + f" | {fmt(x['mean8k'])} | {fmt(x['mean2k'])} | {x['ftm8k'] or '-'} | {x['eval_sizes']} |")
        print("|   2k | | | | | " + " | ".join(fmt(x[f'{k}2k']) for k in TASKS) + " | | | | |")
    if a.json:
        json.dump({"rows": rows, "dense_curve": DENSE_CURVE, "refs_8k": REFS_8K}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
