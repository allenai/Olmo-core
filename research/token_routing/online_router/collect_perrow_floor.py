"""Collect the per-row + floor ablation (launch_perrow_floor.py) next to the recipe's own runs.

Per training run: ``flop_meter/actual_pflops`` (last value in the Beaker log), the online router's logged
diagnostics averaged over steps >= 50 (realized keep after block 7 = ``keep_frac``; stage 2 relative
``s1_keep_frac`` and absolute ``s1_keep_frac_abs``; per-document gold leak ``leak_auc`` at stage 1 and
``s1_leak_auc_abs`` = absolute kept fraction after both cuts, 0.5 = density names nothing), and the 8k / 2k
native ladder scores from ``ops result`` (f1; oolong ``score``; eval_size 500 each).

    python research/token_routing/online_router/collect_perrow_floor.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = f"{HERE}/perrow_floor_state.json"
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
BEAKER = os.path.expanduser("~/.local/bin/beaker")
TASKS = ["contradiction", "nq", "outlier", "oolong"]
#: the recipe's own runs (launched by ds64_fast8k/launch_fast8k.py, router 67a28d605)
REFERENCE = {"f8k-mt4-lora64all32r20to7-f1952-tde-lr1e-05": "01M5VH5APW70Y48FW01Y7W3BVF",
             "f8k-mt4-lora64all32r20to7-f3904-tde-lr1e-05": "01M5VH6FK7J23RHG272XB796M5"}
#: dense fresh curve (rows/task -> PF, 8k mean), from the tokdrop_lr hub
DENSE = [(1952, 1531, 0.843), (3904, 3063, 0.899), (5856, 4595, 0.901), (7808, 6128, 0.904)]
DIAG = ["keep_frac", "s1_keep_frac", "s1_keep_frac_abs", "leak_auc", "s1_leak_auc_abs", "floor_extra_frac",
        "s1_floor_extra_frac"]


def train_log(ex, cache):
    path = os.path.join(cache, f"{ex}.log")
    if not os.path.exists(path):
        out = subprocess.run([BEAKER, "experiment", "logs", ex], capture_output=True, text=True, timeout=600).stdout
        open(path, "w").write(out)
    return open(path).read()


def parse_train(log):
    pf = re.findall(r"flop_meter/actual_pflops[ =]+([0-9.]+)", log)
    vals = {k: [] for k in DIAG}
    last = 0
    for m in re.finditer(r"\[online-router\] step (\d+): (.*)", log):
        step = int(m.group(1))
        last = max(last, step)
        if step < 50:
            continue
        kv = dict(t.split("=", 1) for t in m.group(2).split() if "=" in t)
        for k in DIAG:
            try:
                vals[k].append(float(kv[k]))
            except (KeyError, ValueError):
                pass
    return {"pf": float(pf[-1]) if pf else None, "last_router_step": last,
            **{k: (round(statistics.mean(v), 4) if v else None) for k, v in vals.items()}}


def scores():
    rows = json.loads(subprocess.run([OPS, "result", "show", "--json"], capture_output=True, text=True,
                                     timeout=300).stdout)
    out = {}
    for r in rows:
        if "native" not in str(r.get("eval_backend", "")) or r.get("metric") not in ("f1", "score"):
            continue
        out[(r.get("train_run"), r["task"], r["rung"])] = (r["value"], r.get("eval_size"))
    return out


def dense_at(pf):
    """Dense 8k mean at ``pf`` PF, linear in log PF between the dense fresh points (None outside the curve)."""
    import math

    for (r0, p0, v0), (r1, p1, v1) in zip(DENSE, DENSE[1:]):
        if p0 <= pf <= p1:
            t = (math.log(pf) - math.log(p0)) / (math.log(p1) - math.log(p0))
            return v0 + t * (v1 - v0)
    return None


def ftm(score, pf):
    """FLOPs-to-match: dense PF reaching ``score`` (8k mean; log-PF interpolation of the dense curve) / ``pf``.
    Below the first dense point (0.843): an upper bound '<=' 1531 / pf (dense matches it with at most 1531 PF);
    above the last: a lower bound '>=' 6128 / pf."""
    import math

    if score <= DENSE[0][2]:
        return f"<={DENSE[0][1] / pf:.2f}"
    for (r0, p0, v0), (r1, p1, v1) in zip(DENSE, DENSE[1:]):
        if v0 < score <= v1:
            t = (score - v0) / (v1 - v0)
            return f"{math.exp(math.log(p0) + t * (math.log(p1) - math.log(p0))) / pf:.2f}"
    return f">={DENSE[-1][1] / pf:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--cache", default=os.environ.get("TMPDIR", "/tmp"))
    a = ap.parse_args()
    st = json.load(open(STATE))
    runs = dict(REFERENCE)
    runs.update({n: r["ex"] for n, r in st["runs"].items() if not r.get("max_steps")})
    sc = scores()
    table = []
    for name, ex in runs.items():
        row = {"run": name, "beaker": ex, **parse_train(train_log(ex, a.cache))}
        for rung in ("8k", "2k"):
            v = [sc.get((name, t, rung)) for t in TASKS]
            for t, x in zip(TASKS, v):
                row[f"{t}@{rung}"] = x[0] if x else None
                if x and (x[1] or 0) < 500:
                    row.setdefault("small_eval", []).append(f"{t}@{rung} eval_size={x[1]}")
            row[f"mean@{rung}"] = round(sum(x[0] for x in v) / 4, 4) if all(v) else None
        if row["pf"] and row["mean@8k"] is not None:
            d = dense_at(row["pf"])
            row["dense8k_at_pf"] = round(d, 4) if d is not None else None
            row["ftm8k"] = ftm(row["mean@8k"], row["pf"])
        table.append(row)
    cols = ["run", "pf", "mean@8k", "ftm8k", "mean@2k"] + [f"{t}@8k" for t in TASKS] + DIAG
    print("\t".join(cols))
    for r in table:
        print("\t".join(str(r.get(c)) for c in cols))
    if a.json:
        json.dump(table, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
