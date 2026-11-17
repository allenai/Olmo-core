"""Collect the early_loss_proxy validation (launch_val.py state.json): training FLOPs / loss / wall-clock and the
generation-graded ladder cells, then the FLOPs-to-match (FTM) of each recipe run on the dense-LoRA f4/f2/f1 curve.

* Training (finished runs only): last ``flop_meter/actual_pflops``, last train CE, steps, Beaker started -> exited.
* Evals: ``[ladder:<task>@<rung>] <metric>=<v> (n=<N>, skipped_too_long=<S>)`` + its ``ctc metrics`` line
  (parse_rate). SE = binomial sqrt(p(1-p)/eval_size) per cell; a mean's SE = sqrt(sum se^2) / cells.
* FTM: dense-LoRA score vs log(PF) is interpolated piecewise linearly between its budget points (on the running-max
  envelope, so a non-monotone dense point cannot create two crossings); FTM = dense PF at the recipe's score / recipe
  PF. Below the cheapest dense point the value is an upper bound (``<=``), above the most expensive a lower bound
  (``>=``), as on the tokdrop_lr STATUS page.

    python3 research/token_routing/early_loss_proxy/validation/collect_val.py [--record]
"""
import argparse
import ast
import datetime as dt
import json
import math
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
BEAKER = os.path.expanduser("~/.local/bin/beaker")
CACHE = os.path.join(HERE, "logs")
SCORE = re.compile(r"\[ladder:([a-z_]+)@(\d+k)\] ([a-z0-9_]+)=(-?[0-9.]+|None) \(n=(\d+), skipped_too_long=(\d+)\)")
CTC = re.compile(r"\[ladder:([a-z_]+)@(\d+k)\] ctc metrics: (\{.*\})")
PF = re.compile(r"flop_meter/actual_pflops[\"'=: ]+([0-9.]+(?:e[+-]?\d+)?)")
CE = re.compile(r"train/CE loss[\"'=: ]+([0-9.]+(?:e[+-]?\d+)?)")
STEP = re.compile(r"\[step=(\d+)/(\d+)")
TASKS = ["contradiction", "outlier", "reorder", "qdmatch_nq"]
RUNGS = ["4k", "8k", "16k"]
DENSE_CURVE = ["elpv-denselora-f4", "elpv-denselora-f2", "elpv-denselora-f1"]


def beaker_job(ex):
    out = subprocess.run([BEAKER, "experiment", "get", ex, "--format", "json"], capture_output=True, text=True,
                         timeout=120).stdout
    try:
        return json.loads(out)[0]["jobs"][-1]["status"]
    except Exception:  # noqa: BLE001
        return {}


def log_of(ex, final):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{ex}.log")
    if os.path.exists(path):
        return open(path, errors="replace").read()
    txt = subprocess.run([BEAKER, "experiment", "logs", ex], capture_output=True, text=True, timeout=900).stdout
    if final:
        open(path, "w").write(txt)
    return txt


def _ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "")[:26]) if s else None


def train_point(name, r):
    st = beaker_job(r["ex"])
    if not st.get("finalized") or st.get("exitCode") != 0:
        return None
    log = log_of(r["ex"], True)
    pf = [float(x) for x in PF.findall(log)]
    ce = [float(x) for x in CE.findall(log)]
    steps = [int(a) for a, _ in STEP.findall(log)]
    wall = (_ts(st.get("exited")) - _ts(st.get("started"))).total_seconds() if st.get("started") else None
    return dict(run=name, arm=r["arm"], budget=r["budget"], seed=r["seed"], pf=pf[-1] if pf else None,
                final_train_ce=ce[-1] if ce else None, final_train_ce_last10=(sum(ce[-10:]) / len(ce[-10:]))
                if ce else None, steps=max(steps) if steps else None, wall_s=wall, job=r["ex"])


def eval_cells(e):
    st = beaker_job(e["ex"])
    final = bool(st.get("finalized"))
    log = log_of(e["ex"], final)
    ctc = {(t, g): ast.literal_eval(d) for t, g, d in CTC.findall(log)}
    out = []
    for t, rung, metric, v, n, skipped in SCORE.findall(log):
        t = "contradiction" if t == "contra" else t
        m = ctc.get((t, rung), {})
        out.append(dict(task=t, rung=rung, metric=metric, value=None if v == "None" else float(v),
                        eval_size=int(n), skipped_too_long_rank0=int(skipped), parse_rate=m.get("parse_rate"),
                        ctc=m, job=e["ex"], finalized=final, exit=st.get("exitCode")))
    return out


def se(p, n):
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n else float("nan")


def summarize(cells):
    """cells: {(task, rung): cell} -> means with SE for overall / per rung / per task."""
    def agg(keys):
        got = [cells[k] for k in keys if k in cells and cells[k]["value"] is not None]
        if len(got) != len(keys):
            return None
        m = sum(c["value"] for c in got) / len(got)
        s = math.sqrt(sum(se(c["value"], c["eval_size"]) ** 2 for c in got)) / len(got)
        return {"mean": round(m, 4), "se": round(s, 4), "cells": len(got)}
    allk = [(t, g) for t in TASKS for g in RUNGS]
    res = {"all12": agg(allk)}
    for g in RUNGS:
        res[f"rung_{g}"] = agg([(t, g) for t in TASKS])
    for t in TASKS:
        res[f"task_{t}"] = agg([(t, g) for g in RUNGS])
    return res


def ftm(curve, pf, score):
    """curve: [(pf, score)] ascending in pf. Returns (ftm, qualifier)."""
    env, best = [], -1e9
    for p, s in curve:
        best = max(best, s)
        env.append((p, best))
    if score <= env[0][1]:
        return env[0][0] / pf, "<="
    if score > env[-1][1]:
        return env[-1][0] / pf, ">="
    for (p0, s0), (p1, s1) in zip(env, env[1:]):
        if s0 < score <= s1:
            frac = (score - s0) / (s1 - s0)
            return math.exp(math.log(p0) + frac * (math.log(p1) - math.log(p0))) / pf, "="
    return float("nan"), "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true", help="ops result add --check every finished cell")
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    a = ap.parse_args()
    st = json.load(open(os.path.join(HERE, "state.json")))
    train = {n: train_point(n, r) for n, r in st["runs"].items() if r.get("ex")}
    cells = {}
    for ename, e in st["evals"].items():
        if not e.get("ex"):
            continue
        for c in eval_cells(e):
            if c["task"] == e["task"]:
                cells.setdefault(e["train_run"], {})[(c["task"], c["rung"])] = c
    summ = {run: summarize(cs) for run, cs in cells.items()}
    # FTM of every non-dense-LoRA run vs the dense-LoRA curve, on each aggregate
    out_ftm = {}
    curve_runs = [r for r in DENSE_CURVE if train.get(r) and r in summ]
    for run in summ:
        if run in DENSE_CURVE or not train.get(run) or len(curve_runs) < 2:
            continue
        out_ftm[run] = {}
        for key, v in summ[run].items():
            if v is None or any(summ[c].get(key) is None for c in curve_runs):
                continue
            curve = [(train[c]["pf"], summ[c][key]["mean"]) for c in curve_runs]
            x, q = ftm(curve, train[run]["pf"], v["mean"])
            out_ftm[run][key] = {"ftm": round(x, 2), "q": q}
    res = {"train": train, "summary": summ, "ftm_vs_denselora": out_ftm,
           "cells": {run: {f"{t}@{g}": c for (t, g), c in cs.items()} for run, cs in cells.items()}}
    json.dump(res, open(a.out, "w"), indent=1)
    for run, t in train.items():
        print(f"{run:24s} " + (f"PF={t['pf']:.1f} CE={t['final_train_ce']} steps={t['steps']} wall={t['wall_s']}"
                               if t else "(not finished)"))
    for run, cs in cells.items():
        print(run, " ".join(f"{t[:5]}@{g}={c['value']}(pr={c['parse_rate']},n={c['eval_size']})"
                            for (t, g), c in sorted(cs.items())))
        print("   ", json.dumps(summ[run]), json.dumps(out_ftm.get(run, {})))
    if a.record:
        for run, cs in cells.items():
            for (t, g), c in cs.items():
                if not c["finalized"] or c["value"] is None:
                    continue
                cmd = [OPS, "result", "add", "--run-id", f"{run}-ev{t}", "--train-run", run, "--experiment",
                       "tokdrop_lr", "--task", t, "--rung", g, "--metric", c["metric"], "--value", str(c["value"]),
                       "--eval-size", str(c["eval_size"]), "--cot-label", "no-cot", "--eval-backend",
                       "native-docchunk-ladder(suite-rows,qafter,ctc-grade)", "--eval-job-id", c["job"],
                       "--ckpt-path", f"/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts/{run}",
                       "--check"]
                r = subprocess.run(cmd, capture_output=True, text=True)
                print(f"[ops] {run} {t}@{g}: {(r.stdout + r.stderr).strip()[-300:]}")


if __name__ == "__main__":
    main()
