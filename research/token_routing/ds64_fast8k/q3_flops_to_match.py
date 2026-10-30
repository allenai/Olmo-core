"""Qwen3-4B recipe study (2026-10-29): dense-vs-recipe FLOPs-to-match on the fresh 4-task pools, next to Qwen3.5-4B.

Reads the Berkeley-slurm runs of run_q3_local.sbatch (one log per run: training + the four ladder evals) and the
trainer's flops.json; Qwen3.5 numbers come from lora_routing_2x2.json (the LoRA x routing record's tables), recomputed
here with the SAME curve code so the two models are compared one way.

Score = the 8-cell mean (contradiction / nq / outlier / oolong x 8k / 2k; each task's primary metric = the first one on
its ``[ladder:<task>@<rung>]`` line; eval_size 500 each, no-cot). Dense curve = per data size the best LR among those run
(5e-6, 1e-5), seed-averaged, made monotone (running max), interpolated linearly in log PF. "x dense" = the dense PF that
reaches the cell's score / the cell's PF; a score above the curve's top is reported as ">= top PF / cell PF".

    python research/token_routing/ds64_fast8k/q3_flops_to_match.py --out research/token_routing/ds64_fast8k/q3_flops_to_match.json
"""

import argparse
import glob
import json
import math
import os
import re
import subprocess
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = "/net/sneetches/data/prasann/joblogs"
CKPTS = "/net/sneetches/data/prasann/q3recipe/ckpts"
TASKS = ["contradiction", "nq", "outlier", "oolong"]
KEY = {"contra": "contradiction"}
HDR = re.compile(r"=== q3recipe host=\S+ RUN=(\S+) NGPU=(\d+)")
EVHDR = re.compile(r"^=== EVAL (f8k-\S+) [a-z]+ ")  # a log's pending_evals section scores ANOTHER run
LAD = re.compile(r"\[ladder:([a-z_]+)@(\d+k)\] ([a-z0-9_]+)=(-?[0-9.]+) \(n=(\d+)")
STEP = re.compile(r"^(\S+ \S+)\s.*\[step=(\d+)/(\d+)")
MEM = re.compile(r"gpu_memory/GPU (active|reserved) mem \(GiB\)=([0-9.]+)")
NAME = re.compile(r"f8k-mt4-(.+)-f(\d+)(?:st(\d+))?-q3(\d*b)?-lr([0-9.e-]+)(?:-sn)?$")  # -q3- = 4B; -q317b- = 1.7B
SIZE_OF = {None: "4b", "06b": "0.6b", "17b": "1.7b", "8b": "8b"}
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
BEAKER = os.path.expanduser("~/.local/bin/beaker")
STATE = f"{HERE}/state_q3.json"
CACHE = f"{HERE}/results_q3/beaker_logs"  # results*/ is gitignored
PF_RE = re.compile(r"flop_meter/actual_pflops[ =]+([0-9.]+)")
DENSE_PF_RE = re.compile(r"flop_meter/dense_pflops[ =]+([0-9.]+)")


def parse_log(path):
    """(run, ngpu, {evaluated run: {(task, rung): (v, n, metric)}}, mem, steps)"""
    run, ngpu, scores, mem, steps, cur = None, 0, defaultdict(dict), defaultdict(float), [], None
    for ln in open(path, errors="replace"):
        m = HDR.search(ln)
        if m:
            run, ngpu = m.group(1), int(m.group(2))
            cur = run
        m = EVHDR.match(ln)
        if m:
            cur = m.group(1)
        for m in LAD.finditer(ln):
            k = (KEY.get(m.group(1), m.group(1)), m.group(2))
            scores[cur].setdefault(k, (float(m.group(4)), int(m.group(5)), m.group(3)))  # first metric = primary
        m = MEM.search(ln)
        if m:
            mem[m.group(1)] = max(mem[m.group(1)], float(m.group(2)))
        m = STEP.match(ln)
        if m:
            steps.append((m.group(1), int(m.group(2))))
    return run, ngpu, scores, dict(mem), steps


def collect(include_beaker=True):
    runs = {}
    for path in sorted(glob.glob(f"{LOGS}/q3recipe_*.log")):
        run, ngpu, scores, mem, steps = parse_log(path)
        if not run or not NAME.match(run):
            continue
        for er, sc in scores.items():  # eval-only reruns / pending_evals add tasks to the run they scored
            if er and NAME.match(er):
                e = runs.setdefault(er, {"logs": [], "scores": {}, "mem": {}, "steps": 0})
                for k, v in sc.items():
                    e["scores"].setdefault(k, v)
        r = runs.setdefault(run, {"logs": [], "scores": {}, "mem": {}, "steps": 0})
        r["logs"].append(path)
        for k, v in mem.items():
            r["mem"][k] = max(r["mem"].get(k, 0), v)
        if steps:
            import datetime as dt
            t = [dt.datetime.strptime(s[0][:19], "%Y-%m-%d %H:%M:%S") for s in steps]
            r["steps"] = max(r["steps"], steps[-1][1])
            if len(t) > 2:
                r["s_per_step"] = round((t[-1] - t[1]).total_seconds() / max(1, steps[-1][1] - steps[1][1]), 2)
            r["ngpu"] = ngpu
        fj = f"{CKPTS}/{run}/flops.json"
        if os.path.exists(fj):
            f = json.load(open(fj))
            r["pf"], r["pf_dense_equiv"] = f["actual_pflops"], f["dense_equivalent_pflops"]
    if include_beaker:
        runs.update(beaker_runs(ops_results()))
    out = {}
    for run, r in runs.items():
        arm, rows, cut, sz, lr = NAME.match(run).groups()
        cells = {f"{t}@{g}": r["scores"].get((t, g)) for t in TASKS for g in ("8k", "2k")}
        ok = all(cells.values()) and all(v[1] >= 500 for v in cells.values())
        out[run] = dict(arm=arm, size=SIZE_OF.get(sz, sz), rows=int(rows), cut=int(cut or 0), lr=lr, pf=r.get("pf"),
                        cluster=r.get("cluster", "sneetches"),
                        pf_dense_equiv=r.get("pf_dense_equiv"), steps=r["steps"], s_per_step=r.get("s_per_step"),
                        peak_active_gib=r["mem"].get("active"), peak_reserved_gib=r["mem"].get("reserved"),
                        cells={k: (v[0] if v else None) for k, v in cells.items()},
                        eval_size={k: (v[1] if v else None) for k, v in cells.items()},
                        y=(round(sum(v[0] for v in cells.values()) / 8, 4) if ok else None), logs=r["logs"])
    return out


def _beaker_log(ex, name):
    """Finished Beaker job's log, cached (a running job is re-fetched every call)."""
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/{name}.log"
    if os.path.exists(path):
        return open(path, errors="replace").read()
    j = json.loads(subprocess.run([BEAKER, "experiment", "get", ex, "--format", "json"], capture_output=True, text=True,
                                  timeout=120).stdout or "{}")
    j = j[0] if isinstance(j, list) else j
    st = (j.get("jobs") or [{}])[0].get("status", {})
    log = subprocess.run([BEAKER, "experiment", "logs", ex], capture_output=True, text=True, timeout=600).stdout
    if st.get("exited"):
        open(path, "w").write(log)
    return log


def beaker_runs(results):
    """Beaker twins (launch_fast8k state_q3.json): scores from ops results, PF / memory / steps from the train log."""
    out = {}
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}}
    for run, r in st["runs"].items():
        if not NAME.match(run) or "st4" in run or not r.get("ex") or r.get("state") == "CANCELLED":
            continue
        log = _beaker_log(r["ex"], run)
        pf = [float(x) for x in PF_RE.findall(log)]
        dpf = [float(x) for x in DENSE_PF_RE.findall(log)]
        if not pf or "Training for" not in log or not re.search(r"\[step=(\d+)/\1[,\]]", log):
            continue  # unfinished
        mem = defaultdict(float)
        for k, v in MEM.findall(log):
            mem[k] = max(mem[k], float(v))
        steps = int(re.findall(r"\[step=(\d+)/", log)[-1])
        scores = {(t, g): v for (tr, t, g), v in results.items() if tr == run}
        out[run] = {"scores": scores, "pf": pf[-1], "pf_dense_equiv": dpf[-1] if dpf else None, "mem": dict(mem),
                    "steps": steps, "logs": [f"beaker:{r['ex']}"], "cluster": "beaker"}
    return out


def ops_results():
    """{(train_run, task, rung): (value, eval_size, metric)} -- first (primary) metric per rung, from ops results."""
    txt = subprocess.run([OPS, "result", "show", "--experiment", "tokdrop_lr", "--json"], capture_output=True, text=True,
                         timeout=120).stdout
    res = {}
    for r in (json.loads(txt) if txt.strip().startswith("[") else []):
        if "-q3-" in (r.get("train_run") or ""):
            res.setdefault((r["train_run"], r["task"], r["rung"]), (r["value"], r["eval_size"], r.get("metric")))
    return res


def record_local(out_jsonl):
    """Append the local (-sn) ladder scores to ops results (ops result add --check), skipping recorded keys."""
    have = {(k[0], k[1], k[2]) for k in ops_results()}
    rows = []
    for path in sorted(glob.glob(f"{LOGS}/q3recipe_*.log")):
        txt = open(path, errors="replace").read()
        m = HDR.search(txt)
        if not m or not NAME.match(m.group(1)):
            continue
        run = m.group(1)
        if "lora" in run and LAD.search(txt) and "adapters re-installed" not in txt:
            print(f"!!! {path}: LoRA run scored without 'adapters re-installed' -- NOT recorded")
            continue
        sha = re.search(r"CODE=\S+/code_([0-9a-f]+)", txt)
        job = re.search(r"_(?:chain)?(\d+)\.log$", path)
        cur = run
        for ln in txt.splitlines():
          me = EVHDR.match(ln)
          if me:
              cur = me.group(1)
          for mm in LAD.finditer(ln):
            t, rung, metric, v, n = mm.groups()
            t = KEY.get(t, t)
            run = cur
            if (run, t, rung) in have:
                continue
            v, n = float(v), int(n)
            se = math.sqrt(max(v * (1 - v), 1e-9) / n) if 0 <= v <= 1 else None
            rows.append(dict(run_id=f"{run}-ev{t}", train_run=run, experiment="tokdrop_lr", task=t, rung=rung,
                             metric=metric, value=v, eval_size=n, se=se and round(se, 6), cot_label="no-cot",
                             eval_backend="olmo-core-native-docchunk-ladder (sneetches)",
                             ckpt_path=f"sneetches:/data/prasann/q3recipe/ckpts/{run}",
                             source_json=f"sneetches:/data/prasann/q3recipe/ckpts/{run}/eval/{'contra' if t == 'contradiction' else t}_multirung.json",
                             git_commit=sha and sha.group(1), eval_job_id=job and f"slurm:{job.group(1)}",
                             notes=f"log sneetches:{path}"))
            have.add((run, t, rung))
    open(out_jsonl, "w").write("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} new local rows -> {out_jsonl}")
    if rows:
        r = subprocess.run([OPS, "result", "add", "--json", out_jsonl, "--check"], capture_output=True, text=True)
        print((r.stdout + r.stderr).strip()[-2000:])


def dense_curve(points):
    """points: [(pf, y)] best-LR per data size -> monotone (pf, y) sorted by pf."""
    pts = sorted(points)
    out, best = [], -1.0
    for pf, y in pts:
        best = max(best, y)
        out.append((pf, best))
    return out


def pf_for(curve, y):
    """Dense PF reaching score y (linear in log PF); (pf, flag) with flag '>=' when y is above the curve's top."""
    if y <= curve[0][1]:
        return curve[0][0], "<="
    for (p0, y0), (p1, y1) in zip(curve, curve[1:]):
        if y0 < y <= y1:
            f = (y - y0) / (y1 - y0)
            return math.exp(math.log(p0) + f * (math.log(p1) - math.log(p0))), ""
    return curve[-1][0], ">="


def dense_at(curve, pf):
    if pf <= curve[0][0]:
        return curve[0][1]
    for (p0, y0), (p1, y1) in zip(curve, curve[1:]):
        if p0 <= pf <= p1:
            return y0 + (math.log(pf) - math.log(p0)) / (math.log(p1) - math.log(p0)) * (y1 - y0)
    return curve[-1][1]


def match(curve, cell_pf, cell_y):
    pf, flag = pf_for(curve, cell_y)
    return {"x_dense": round(pf / cell_pf, 2), "flag": flag, "vs_dense": round(cell_y - dense_at(curve, cell_pf), 4)}


def q35_reference():
    d = json.load(open(f"{HERE}/lora_routing_2x2.json"))
    curve = dense_curve([(p["pf"], p["y"]) for p in d["dense_curve"]["mt4:best_5e6_1e5"]])
    ref = {}
    for t in d["tables"]:
        if t["mix"] != "mt4" or t["pool"] != "f":
            continue
        for c, name in (("A", "dense"), ("D", "recipe")):
            cell = t["cells"].get(c)
            if cell:
                ref[f"f{t['rows']}:{name}"] = dict(pf=cell["pf"], y=cell["y"], lr=cell["lr"], n_seeds=cell["n"],
                                                   stored_x=cell.get("x_best_5e6_1e5"), **match(curve, cell["pf"], cell["y"]))
    return {"curve": curve, "cells": ref}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{HERE}/q3_flops_to_match.json")
    ap.add_argument("--record", action="store_true", help="first add the local ladder scores to ops results (--check)")
    a = ap.parse_args()
    if a.record:
        record_local(f"{HERE}/results_q3/local_rows.jsonl")
    runs = collect()
    res = {"sizes": {}, "q35": q35_reference(), "runs": runs}
    for size in sorted({r["size"] for r in runs.values()}):
        by = defaultdict(list)  # dense (rows, lr) -> scores
        for r in runs.values():
            if r["size"] == size and r["arm"] == "dense" and r["y"] is not None and not r["cut"]:
                by[(r["rows"], r["lr"])].append((r["pf"], r["y"]))
        best = {}
        for (rows, lr), v in by.items():
            pf, y = sum(p for p, _ in v) / len(v), sum(y for _, y in v) / len(v)
            if rows not in best or y > best[rows][1]:
                best[rows] = (pf, y, lr)
        curve = dense_curve([(pf, y) for pf, y, _ in best.values()]) if len(best) >= 2 else None
        cells = {}
        for run, r in sorted(runs.items()):
            if r["size"] != size or r["y"] is None or r["pf"] is None or r["cut"]:
                continue
            cells[run] = {k: r[k] for k in ("arm", "size", "rows", "lr", "cluster", "pf", "y", "steps", "s_per_step",
                                            "peak_active_gib", "peak_reserved_gib")}
            cells[run].update({k: r["cells"][k] for k in ("outlier@8k", "contradiction@8k")})
            cells[run]["pf_per_step"] = round(r["pf"] / r["steps"], 3) if r["steps"] else None
            if curve:
                cells[run].update(match(curve, r["pf"], r["y"]))
        res["sizes"][size] = {"dense_curve": curve, "dense_best_lr": {str(k): v for k, v in sorted(best.items())},
                              "cells": cells}
        print(f"Qwen3-{size} dense curve (rows/task: PF, y, lr):",
              {k: (round(v[0]), v[1], v[2]) for k, v in sorted(best.items())})
        for run, c in cells.items():
            print(f"  {run:60s} PF {c['pf']:8.1f} y {c['y']:.4f} x dense {c.get('flag', '')}{c.get('x_dense')} "
                  f"vs {c.get('vs_dense')} PF/step {c['pf_per_step']} peak act {c['peak_active_gib']} GiB")
    res["q3"] = res["sizes"].get("4b")  # back-compat with the 4B-only layout
    json.dump(res, open(a.out, "w"), indent=1)
    print("Qwen3.5-4B (recomputed):")
    for k, c in res["q35"]["cells"].items():
        print(f"  {k:16s} PF {c['pf']:8.1f} y {c['y']:.4f} x dense {c['flag']}{c['x_dense']} (record {c['stored_x']})")


if __name__ == "__main__":
    main()
