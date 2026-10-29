"""Login-side half of the local eval path (local_eval_node.py), run every few minutes:

1. every CLAIMED cell (``<run>__<task>__<rung>``) is marked in the fast8k state.json as scored locally
   (``ex = "local"``), and its Beaker twin eval (``<run>-ev<task>-<rung>``), if one was submitted, is cancelled --
   so nothing is scored twice and the eval driver does not submit it;
2. every finished cell's ``[ladder:<task>@<rung>] <metric>=<v> (n=<N>`` lines become ops result rows
   (``ops result add --check``), exactly like record_beaker_evals.py does for Beaker logs.

    python research/token_routing/layer_drop/record_local_evals.py --claims <dir> --out rows.jsonl
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
F8K = os.path.join(HERE, "..", "ds64_fast8k")
sys.path.insert(0, F8K)
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
BEAKER = os.path.expanduser("~/.local/bin/beaker")
LINE = re.compile(r"\[ladder:([a-z_]+)@(\d+k)\] ([a-z0-9_]+)=(-?[0-9.]+) \(n=(\d+)")
KEY_TASK = {"contra": "contradiction"}


def cancel_beaker(ex):
    try:
        out = subprocess.run([BEAKER, "experiment", "get", ex, "--format", "json"], capture_output=True, text=True,
                             timeout=120).stdout
        job = json.loads(out)[0]["jobs"][-1]
        if job.get("status", {}).get("finalized") or job.get("status", {}).get("canceled"):
            return "final"
        subprocess.run([BEAKER, "job", "cancel", job["id"]], capture_output=True, text=True, timeout=120)
        return "cancelled"
    except Exception as e:  # noqa: BLE001
        return f"error {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--experiment", default="tokdrop_lr")
    a = ap.parse_args()
    import launch_fast8k as L  # merge-safe load/save of the shared state.json

    st = L.load_state()
    # cells Beaker already ran or is running: pre-claim them (marker <cell>.beaker) so the local nodes skip them
    runs_file = os.path.join(os.path.dirname(os.path.abspath(a.claims)), "runs.txt")
    for run in (open(runs_file).read().split() if os.path.exists(runs_file) else []):
        for task in ("contradiction", "nq", "outlier", "oolong"):
            for rung in ("8k", "2k"):
                e = st["evals"].get(f"{run}-ev{task}-{rung}", {})
                cell = os.path.join(a.claims, f"{run}__{task}__{rung}")
                if e.get("ex") and e["ex"] != "local" and e.get("state") in ("DONE", "R") and not os.path.exists(cell):
                    os.mkdir(cell)
                    open(cell + ".beaker", "w").write(f"{e['ex']} {e['state']}\n")
    cells = sorted(d for d in os.listdir(a.claims) if os.path.isdir(os.path.join(a.claims, d))
                   and not os.path.exists(os.path.join(a.claims, d + ".beaker")))
    changed = False
    for cell in cells:
        run, task, rung = cell.split("__")
        ename = f"{run}-ev{task}-{rung}"
        e = st["evals"].get(ename, {})
        if e.get("ex") == "local":
            continue
        note = cancel_beaker(e["ex"]) if e.get("ex") else "none"
        st["evals"][ename] = {"ex": "local", "state": "LOCAL", "train_run": run, "task": task, "beaker_twin": note}
        changed = True
        print(f"[local] {ename}: beaker twin {note}")
    if changed:
        L.save_state(st)
    out = subprocess.run([OPS, "result", "show", "--json"], capture_output=True, text=True, timeout=120).stdout
    have = {(r.get("train_run"), r["task"], r["rung"], r.get("metric"))
            for r in (json.loads(out) if out.strip().startswith("[") else [])}
    rows = []
    for cell in cells:
        # a cell's [ladder:...] line is printed only once its rung is fully scored, so the log alone is enough (the
        # .done marker is not needed: a restarted node driver does not write it for evals it did not start)
        lp = os.path.join(a.claims, cell + ".log")
        if not os.path.exists(lp):
            continue
        run, task, rung = cell.split("__")
        log = open(lp, errors="replace").read()
        for m in LINE.finditer(log):
            t, rg, metric, v, n = m.groups()
            t = KEY_TASK.get(t, t)
            if (run, t, rg, metric) in have:
                continue
            v, n = float(v), int(n)
            se = math.sqrt(max(v * (1 - v), 1e-9) / n) if 0 <= v <= 1 else None
            rows.append(dict(run_id=f"{run}-ev{task}-{rung}", train_run=run, experiment=a.experiment, task=t,
                             rung=rg, metric=metric, value=v, eval_size=n, se=se and round(se, 6),
                             cot_label="no-cot", eval_backend="olmo-core-native-docchunk-ladder (berkeley, batched)",
                             ckpt_path=f"/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts/{run}",
                             eval_job_id=f"local:{cell}"))
            have.add((run, t, rg, metric))
    with open(a.out, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(cells)} claimed cells; {len(rows)} new rows -> {a.out}")
    if rows:
        r = subprocess.run([OPS, "result", "add", "--json", a.out, "--check"], capture_output=True, text=True)
        print((r.stdout + r.stderr).strip().splitlines()[-1])


if __name__ == "__main__":
    main()
