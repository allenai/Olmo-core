"""Rows <= 16k vs rows <= 64k at matched FLOPs: how much do long eval rungs lose when training never sees long rows?

* m16 runs (mt11L16 shard, seq 16,384): native docchunk ladder scores (2k..64k, 10 tasks; eval logs parsed with
  collect_m64.eval_rows) + olmo-eval long-rung scores (r64k/r128k/r256k, 6 tasks; results_lengen/lengen.json).
* Reference for m16 dense: the 64k-trained dense LR 5e-6 curve (f16..f1, seed 0) interpolated in log PF at the m16 run's
  PF, between its TRUE neighbouring budgets. Reference for the m16 recipe: the 64k-trained recipe f4 directly (1,696 PF vs
  1,836 PF), and the 64k dense curve at the m16 recipe's PF.

    python3 research/token_routing/mt11_64k/lengen_short16.py [--record]
Writes results_lengen/short16.json and prints the tables.
"""
import argparse
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from collect_m64 import eval_rows  # noqa: E402

OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
RES = os.path.join(HERE, "results_lengen")
STATE = os.path.join(HERE, "state_lengen.json")
POINTS = ("/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.claude/worktrees/mt11-64k/research/token_routing/"
          "mt11_64k/results/points.json")
DENSE = ["m64-dense-f16-lr5e-06", "m64-dense-f8-lr5e-06", "m64-dense-f4-lr5e-06", "m64-dense-f2-lr5e-06",
         "m64-dense-f1-lr5e-06"]
RECIPE64 = "m64-lora64r20to7f16-f4-lr5e-06"
SHORT = {"m16-dense-s380-lr5e-06": "dense", "m16-dense-s770-lr5e-06": "dense",
         "m16-lora64r20to7f16-s380-lr5e-06": "recipe"}
NATIVE_RUNGS = ["2k", "8k", "16k", "32k", "64k"]
LONG_RUNGS = ["r64k", "r128k", "r256k"]
#: no-signal cells excluded from means, as in the main record (every arm fails the parser / scores ~0)
NO_SIGNAL = {("reorder", "8k"), ("reorder", "16k"), ("grouping", "32k")}


def interp_neighbours(score_of, pf, x):
    pts = sorted((pf[d], d) for d in DENSE if d in pf)
    lo, hi = [p for p in pts if p[0] <= x], [p for p in pts if p[0] >= x]
    if not lo or not hi:
        return None
    (x0, a), (x1, b) = lo[-1], hi[0]
    ya, yb = score_of(a), score_of(b)
    if ya is None or yb is None:
        return None
    if a == b:
        return ya
    w = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
    return ya + w * (yb - ya)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    pts = json.load(open(POINTS))
    pf = {t["run"]: t["pf"] for t in pts["train"] if t.get("pf")}
    pf.update(json.load(open(os.path.join(HERE, "extra_pf.json"))))
    native = {(s["train_run"], s["task"], s["rung"]): s["value"] for s in pts["scores"] if s.get("eval_size") == 500}
    # m16 native ladder scores from their eval logs
    st = json.load(open(STATE))
    new_rows = []
    for ename, e in st.get("evals", {}).items():
        if e.get("train_run") not in SHORT or not e.get("ex"):
            continue
        for r in eval_rows(ename, e):
            if r["final"] and r["eval_size"] == 500:
                native[(r["train_run"], r["task"], r["rung"])] = r["value"]
                new_rows.append(r)
    lg = json.load(open(os.path.join(RES, "lengen.json"))) if os.path.exists(os.path.join(RES, "lengen.json")) else {}
    full = {"r64k": 500, "r128k": 500, "r256k": 125}
    longs = {(r["run"], r["task"], r["rung"]): r["value"] for r in lg.get("rows", []) if r["eval_size"] == full[r["rung"]]}

    out = {"native": [], "long": []}
    for run, kind in SHORT.items():
        if run not in pf:
            continue
        for rung in NATIVE_RUNGS + LONG_RUNGS:
            src = native if rung in NATIVE_RUNGS else longs
            tasks = sorted({k[1] for k in src if k[0] == run and k[2] == rung})
            for t in tasks:
                if (t, rung) in NO_SIGNAL:
                    continue
                v = src[(run, t, rung)]
                ref_dense = interp_neighbours(lambda d: src.get((d, t, rung)), pf, pf[run])
                ref_rec = src.get((RECIPE64, t, rung)) if kind == "recipe" else None
                out["native" if rung in NATIVE_RUNGS else "long"].append(dict(
                    run=run, kind=kind, pf=pf[run], task=t, rung=rung, score=v, dense64_at_pf=ref_dense,
                    d_dense=None if ref_dense is None else v - ref_dense, recipe64=ref_rec,
                    d_recipe=None if ref_rec is None else v - ref_rec))
    json.dump(out, open(os.path.join(RES, "short16.json"), "w"), indent=1)
    for part in ("native", "long"):
        print(f"\n== {part} ==")
        for run in SHORT:
            for rung in (NATIVE_RUNGS if part == "native" else LONG_RUNGS):
                cs = [c for c in out[part] if c["run"] == run and c["rung"] == rung]
                dd = [c["d_dense"] for c in cs if c["d_dense"] is not None]
                dr = [c["d_recipe"] for c in cs if c["d_recipe"] is not None]
                if not cs:
                    continue
                print(f"  {run:34} {rung:6} n_tasks={len(cs):2}  vs64k-dense@PF "
                      f"{(sum(dd) / len(dd) if dd else float('nan')):+.3f} ({len(dd)})"
                      + (f"  vs64k-recipe {(sum(dr) / len(dr) if dr else float('nan')):+.3f} ({len(dr)})" if dr else "")
                      + "   " + " ".join(f"{c['task'][:6]}" + ("" if c["d_dense"] is None else f"{c['d_dense']:+.2f}")
                                         for c in cs))
    if a.record and new_rows:
        path = os.path.join(RES, "record_m16.jsonl")
        done_path = os.path.join(RES, "recorded_m16.json")
        done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()
        todo = [r for r in new_rows if f"{r['eval_run']}|{r['task']}|{r['rung']}" not in done]
        with open(path, "w") as f:
            for r in todo:
                f.write(json.dumps(dict(run_id=r["eval_run"], train_run=r["train_run"], experiment="mt11_64k",
                                        task=r["task"], rung=r["rung"], metric=r["metric"], value=r["value"],
                                        eval_size=r["eval_size"], cot_label="no-cot", eval_job_id=r["eval_job"],
                                        eval_backend="olmo-core-native-docchunk-ladder suite-rows (beaker)")) + "\n")
        if todo:
            p = subprocess.run([OPS, "result", "add", "--json", path, "--check"], capture_output=True, text=True)
            print((p.stdout + p.stderr).strip()[-2000:])
            if p.returncode == 0:
                done |= {f"{r['eval_run']}|{r['task']}|{r['rung']}" for r in todo}
                json.dump(sorted(done), open(done_path, "w"))


if __name__ == "__main__":
    main()
