"""Collect the position-id reset ablation (launch_posid_reset.py) next to the recipe's own runs (2 seeds per budget).

Per training run: actual PF and router diagnostics (``collect_perrow_floor.parse_train``), training CE (``train/CE
loss`` lines in the Beaker log: mean over the first 50 steps, the last 25% of steps, and the final 20 steps; answer
tokens only, which no cut ever drops, so the CE is comparable between arms), and the 8k / 2k native ladder scores
from ``ops result`` (f1; oolong ``score``; eval_size 500 each).

    python research/token_routing/online_router/collect_posid_reset.py [--json out.json] [--cache DIR]
    python research/token_routing/online_router/collect_posid_reset.py --state tail_rule_state.json  # tail-rule arms
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from collect_perrow_floor import TASKS, ftm, parse_train, scores, train_log  # noqa: E402

STATE = f"{HERE}/posid_reset_state.json"
REFERENCE = {"f8k-mt4-lora64all32r20to7-f1952-tde-lr1e-05": "01M5VH5APW70Y48FW01Y7W3BVF",
             "f8k-mt4-lora64all32r20to7sd1-f1952-tde-lr1e-05": "01M6F3ZAR3PENM133YFNVHDTB5",
             "f8k-mt4-lora64all32r20to7-f3904-tde-lr1e-05": "01M5VH6FK7J23RHG272XB796M5",
             "f8k-mt4-lora64all32r20to7sd1-f3904-tde-lr1e-05": "01M62CQ8YR3Y2GHHRG0WENVP88"}


def train_ce(log):
    ce = [float(x) for x in re.findall(r"train/CE loss=([0-9.]+)", log)]
    if not ce:
        return {"steps": 0}
    q = max(1, len(ce) // 4)
    r = lambda v: round(statistics.mean(v), 4)
    return {"steps": len(ce), "ce_first50": r(ce[:50]), "ce_last25pct": r(ce[-q:]), "ce_last20": r(ce[-20:])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--cache", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--state", default=STATE, help="launcher state.json (relative paths: next to this script)")
    a = ap.parse_args()
    sp = a.state if os.path.isabs(a.state) else os.path.join(HERE, a.state)
    st = json.load(open(sp)) if os.path.exists(sp) else {"runs": {}}
    runs = dict(REFERENCE)
    runs.update({n: r["ex"] for n, r in st["runs"].items() if not r.get("max_steps")})
    sc = scores()
    table = []
    for name, ex in runs.items():
        log = train_log(ex, a.cache)
        row = {"run": name, "beaker": ex, **parse_train(log), **train_ce(log)}
        for rung in ("8k", "2k"):
            v = [sc.get((name, t, rung)) for t in TASKS]
            for t, x in zip(TASKS, v):
                row[f"{t}@{rung}"] = x[0] if x else None
                if x and (x[1] or 0) < 500:
                    row.setdefault("small_eval", []).append(f"{t}@{rung} eval_size={x[1]}")
            row[f"mean@{rung}"] = round(sum(x[0] for x in v) / 4, 4) if all(v) else None
        if row["pf"] and row["mean@8k"] is not None:
            row["ftm8k"] = ftm(row["mean@8k"], row["pf"])
        table.append(row)
    cols = (["run", "pf", "steps", "ce_first50", "ce_last25pct", "ce_last20", "mean@8k", "ftm8k", "mean@2k"]
            + [f"{t}@{r}" for r in ("8k", "2k") for t in TASKS] + ["keep_frac", "s1_keep_frac"])
    print("\t".join(cols))
    for r in table:
        print("\t".join(str(r.get(c)) for c in cols))
    if a.json:
        json.dump(table, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
