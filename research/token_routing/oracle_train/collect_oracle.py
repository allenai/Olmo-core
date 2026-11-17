"""Collect oracle-run evals (Beaker eval logs, `[ladder:task@rung] metric=v (n=N`) and compare with the recipe and dense
at the same budget (f1952, from ops results). Prints a markdown table; --rows writes `ops result add` rows.

    python research/token_routing/oracle_train/collect_oracle.py [--rows rows.jsonl]
"""
import argparse
import json
import math
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BEAKER = os.path.expanduser("~/.local/bin/beaker")
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
SCORE_RE = re.compile(r"\[ladder:([a-z_]+)@(\d+k)\] ([a-z0-9_]+)=(-?[0-9.]+) \(n=(\d+)")
TASKS = ["contradiction", "nq", "outlier", "oolong"]
TASK_OF = {"contra": "contradiction"}
REFS = {"recipe": "f8k-mt4-lora64all32r20to7-f1952-tde-lr1e-05", "dense": "f8k-mt4-dense-f1952-tde-lr5e-06"}


def logs(ex):
    return subprocess.run([BEAKER, "experiment", "logs", ex], capture_output=True, text=True, timeout=180).stdout


def ref_scores(run):
    out = subprocess.run([OPS, "result", "show", "--train-run", run, "--json"], capture_output=True, text=True).stdout
    sc = {}
    try:
        rows = json.loads(out)
    except json.JSONDecodeError:
        return sc
    for r in rows:
        if r.get("metric") in ("f1", "accuracy", "acc", "score") and r.get("task") in TASKS:
            sc[(r["task"], r["rung"])] = float(r["value"])
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="")
    ap.add_argument("--only", default="", help="comma list of training runs whose rows go to --rows (avoid re-adding)")
    a = ap.parse_args()
    st = json.load(open(os.path.join(HERE, "state.json")))
    runs = {}
    rows = []
    for ename, e in st["evals"].items():
        txt = logs(e["ex"])
        for task, rung, metric, v, n in SCORE_RE.findall(txt):
            task = TASK_OF.get(task, task)
            if task != e["task"]:
                continue
            runs.setdefault(e["train_run"], {})[(task, rung)] = (float(v), int(n), metric)
            se = math.sqrt(max(float(v) * (1 - float(v)), 0) / max(int(n), 1))
            rows.append({"experiment": "tokdrop_lr", "run_id": ename, "train_run": e["train_run"], "task": task,
                         "rung": rung, "metric": metric, "value": float(v), "se": round(se, 4),
                         "eval_size": int(n), "cot_label": "no-cot", "eval_backend": "legacy_native_multirung", "eval_job_id": e["ex"], "git_commit": "00771d815",
                         "ckpt_path": f"/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts/{e['train_run']}"})
    refs = {k: ref_scores(v) for k, v in REFS.items()}
    hdr = "| run | " + " | ".join(f"{t[:6]} 8k" for t in TASKS) + " | mean 8k | " + \
        " | ".join(f"{t[:6]} 2k" for t in TASKS) + " | mean 2k |"
    print(hdr)
    print("|" + "---|" * (len(TASKS) * 2 + 3))
    allr = [(k, {kk: (vv, 500, "f1") for kk, vv in v.items()}) for k, v in refs.items()] + sorted(runs.items())
    for name, sc in allr:
        cells, means = [], {}
        for rung in ("8k", "2k"):
            vals = [sc.get((t, rung), (None,))[0] for t in TASKS]
            cells += ["-" if v is None else f"{v:.3f}" for v in vals]
            ok = [v for v in vals if v is not None]
            means[rung] = f"{sum(ok) / len(ok):.3f}" + ("" if len(ok) == len(TASKS) else f" ({len(ok)}/4)") if ok else "-"
            cells.append(means[rung])
        short = name.replace("f8k-mt4-", "").replace("-f1952-orc-lr1e-05", "")
        print(f"| {short} | " + " | ".join(cells) + " |")
    if a.rows:
        with open(a.rows, "w") as f:
            for r in rows:
                if a.only and r["train_run"] not in a.only.split(","):
                    continue
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows)} rows -> {a.rows}")


if __name__ == "__main__":
    main()
