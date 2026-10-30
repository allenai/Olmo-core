"""Sparse-backward campaign (-sbw): 8-cell (4 tasks x {8k, 2k}) mean per run from ops results, plus the recipe tde
reference runs. Record new evals first: record_beaker_evals.py --suffix=-sbw --out rows.jsonl

    python3 research/token_routing/ds64_fast8k/collect_sbw.py
"""
import json, subprocess, statistics, sys
rows = json.loads(subprocess.run(["/accounts/projects/berkeleynlp/prasann/bin/ops", "result", "show", "--json"],
                                 capture_output=True, text=True).stdout)
want = lambda tr: tr and ("-sbw-" in tr or tr in ("f8k-mt4-lora64all32r20to7-f1952-tde-lr1e-05",
                                                    "f8k-mt4-lora64all32r20to7-f3904-tde-lr1e-05",
                                                    "f8k-mt4-lora64all32r20to7-f7808-tde-lr1e-05"))
by = {}
for r in rows:
    tr = r.get("train_run")
    if not want(tr) or r.get("metric") not in ("f1", "score") or r["rung"] not in ("8k", "2k"):
        continue
    by.setdefault(tr, {})[(r["task"], r["rung"])] = (float(r["value"]), r.get("eval_size"))
tasks = ["contradiction", "nq", "outlier", "oolong"]
print(f"{'run':62s} n_cells  mean8   " + "  ".join(f"{t[:6]}8k/2k" for t in tasks))
for tr in sorted(by):
    d = by[tr]
    vals = [d[(t, g)][0] for t in tasks for g in ("8k", "2k") if (t, g) in d]
    sizes = {d[k][1] for k in d}
    cells = "  ".join(f"{d.get((t,'8k'),(float('nan'),))[0]:.3f}/{d.get((t,'2k'),(float('nan'),))[0]:.3f}" for t in tasks)
    print(f"{tr:62s} {len(vals):3d}  {statistics.mean(vals) if vals else float('nan'):.4f}  {cells}  eval_size={sizes}")
