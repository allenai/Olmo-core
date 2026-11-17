"""Weighted token sampling: one table row per run (records/weighted-token-sampling.md).

Scores: `ops result show --train-run` (8 cells: 4 tasks x 8k / 2k, eval_size 500 each). Training cost and memory: the
training log (Beaker `experiment logs`, or the node log for local runs): cumulative `flop_meter` actual PF and the
actual / dense ratio (= measured step cost; per-step speedup = 1 / ratio), peak allocated GPU memory per rank
(`gpu_memory/GPU active mem (GiB)` max over steps). FTM (8k-only): dense PF that reaches the run's 8k 4-task mean on
the fresh dense curve (LR 5e-6), log-linear in PF, / the run's PF; below the lowest dense point it is an upper bound.

    python research/token_routing/wts/collect_wts.py [--md] [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
BEAKER = os.path.expanduser("~/.local/bin/beaker")
CACHE = os.path.join(HERE, "log_cache.json")
TASKS = ["contradiction", "nq", "outlier", "oolong"]
RECIPE = ["f8k-mt4-lora64all32r20to7-f1952-tde-lr1e-05", "f8k-mt4-lora64all32r20to7sd1-f1952-tde-lr1e-05",
          "f8k-mt4-lora64all32r20to7-f3904-tde-lr1e-05", "f8k-mt4-lora64all32r20to7sd1-f3904-tde-lr1e-05"]
DENSE = [f"f8k-mt4-dense-f{r}-tde-lr5e-06" for r in (488, 976, 1952, 3904, 5856, 7808)]


def sh(cmd, timeout=600):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return ""


def scores(run):
    try:
        rows = json.loads(sh([OPS, "result", "show", "--train-run", run, "--json"], 240) or "[]")
    except json.JSONDecodeError:
        return {}
    out = {}
    for r in rows:
        if r.get("task") in TASKS and r.get("rung") in ("8k", "2k"):
            out[(r["task"], r["rung"])] = (float(r["value"]), int(r.get("eval_size") or 0))
    return out


def job_of(run):
    st = json.load(open(os.path.join(HERE, "state.json"))) if os.path.exists(os.path.join(HERE, "state.json")) else {}
    if run in st.get("runs", {}):
        return st["runs"][run]["ex"], st["runs"][run].get("cluster", "beaker")
    try:
        rows = json.loads(sh([OPS, "runs", "--experiment", "tokdrop_lr", "--json"], 120) or "[]")
    except json.JSONDecodeError:
        rows = []
    exs = [r.get("job_id") for r in rows if r.get("run_id") == run]
    return (exs[-1], "beaker") if exs else (None, None)


def parse_log(log):
    pf = re.findall(r"flop-meter\] step \d+: actual ([0-9.]+) PF, dense-equivalent ([0-9.]+) PF", log)
    pf2 = re.findall(r"flop_meter/actual_pflops[ =]+([0-9.]+)", log)
    dn2 = re.findall(r"flop_meter/dense_pflops[ =]+([0-9.]+)", log)
    mem = [float(x) for x in re.findall(r"GPU active mem \(GiB\)=([0-9.]+)", log)]
    mem += [float(x) for x in re.findall(r"peak_mem_gib=([0-9.]+)", log)]
    steps = [int(s) for s in re.findall(r"\[step=(\d+)/\d+", log)]
    a = float(pf2[-1]) if pf2 else (float(pf[-1][0]) if pf else None)
    d = float(dn2[-1]) if dn2 else (float(pf[-1][1]) if pf else None)
    return {"pf": a, "dense_pf": d, "ratio": (a / d if a and d else None), "peak_gib": max(mem) if mem else None,
            "steps": max(steps) if steps else None,
            "done": ("saved model-only checkpoint" in log) or ("WTS_DONE" in log)}


def train_stats(run, cache):
    if run in cache and cache[run].get("done"):
        return cache[run]
    job, cl = job_of(run)
    if not job:
        return None
    if cl and cl.startswith("slurm:"):
        node = cl.split(":", 1)[1]
        files = glob.glob(f"/net/{node}/data/prasann/joblogs/wts-*-{job}.out")
        log = open(files[0], errors="ignore").read() if files else ""
    else:
        log = sh([BEAKER, "experiment", "logs", job])
    out = parse_log(log)
    out["job"] = job
    cache[run] = out
    return out


def mean8(sc, rung="8k"):
    v = [sc.get((t, rung)) for t in TASKS]
    return sum(x[0] for x in v) / 4 if all(v) else None


def ftm(score8, pf, curve):
    """dense PF reaching score8 (log-linear in PF along the dense curve) / pf; ('<=', x) below the first point"""
    pts = sorted((p, s) for p, s in curve if p and s is not None)
    if not pts or not pf or score8 is None:
        return None
    for (p0, s0), (p1, s1) in zip(pts, pts[1:]):
        if s0 <= score8 <= s1 and s1 > s0:
            lp = math.log(p0) + (score8 - s0) / (s1 - s0) * (math.log(p1) - math.log(p0))
            return ("=", math.exp(lp) / pf)
    if score8 < pts[0][1]:
        return ("<=", pts[0][0] / pf)
    return (">=", pts[-1][0] / pf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="", help="comma list (default: every run in state.json + the recipe)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    st = json.load(open(os.path.join(HERE, "state.json"))) if os.path.exists(os.path.join(HERE, "state.json")) else {}
    runs = a.runs.split(",") if a.runs else RECIPE + [
        r for r, v in st.get("runs", {}).items()
        if not v.get("steps") and not any(x in r for x in ("-smoke", "-memprobe", "-st"))
        and not (r + "-loc" in st.get("runs", {}))]  # full-length runs only; a local twin replaces its Beaker run
    curve = []
    for d in DENSE:
        s, t = scores(d), train_stats(d, cache)
        if t and t.get("pf"):
            curve.append((t["pf"], mean8(s)))
    rows = []
    for r in runs:
        s, t = scores(r), train_stats(r, cache) or {}
        m8, m2 = mean8(s), mean8(s, "2k")
        f = ftm(m8, t.get("pf"), curve)
        rows.append({"run": r, "pf": t.get("pf"), "ratio": t.get("ratio"),
                     "speedup": (1 / t["ratio"] if t.get("ratio") else None), "peak_gib": t.get("peak_gib"),
                     "steps": t.get("steps"), **{f"{k}8k": (s.get((k, "8k")) or (None,))[0] for k in TASKS},
                     "mean8k": m8, "mean2k": m2, "ftm8k": (f"{f[0]}{f[1]:.2f}x" if f else None),
                     "eval_sizes": sorted({v[1] for v in s.values()})})
    json.dump(cache, open(CACHE, "w"), indent=1)
    fmt = lambda x, n=3: "-" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))  # noqa: E731
    print("| run | PF | step cost (x dense) | speedup/step | peak GiB | contra | nq | outlier | oolong | 8k mean | 2k mean"
          " | FTM 8k | eval_size |")
    print("|" + "---|" * 13)
    for x in rows:
        print(f"| {x['run']} | {fmt(x['pf'], 1)} | {fmt(x['ratio'])} | {fmt(x['speedup'], 2)} | {fmt(x['peak_gib'], 1)} | "
              + " | ".join(fmt(x[f'{k}8k']) for k in TASKS)
              + f" | {fmt(x['mean8k'])} | {fmt(x['mean2k'])} | {x['ftm8k'] or '-'} | {x['eval_sizes']} |")
    print("dense curve (PF, 8k mean):", [(round(p, 1), s and round(s, 3)) for p, s in sorted(curve)])
    if a.json:
        json.dump({"rows": rows, "dense_curve": curve}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
