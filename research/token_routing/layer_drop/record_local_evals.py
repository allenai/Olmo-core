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

    cells = sorted(d for d in os.listdir(a.claims) if os.path.isdir(os.path.join(a.claims, d)))
    st = L.load_state()
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
        if not os.path.exists(os.path.join(a.claims, cell + ".done")):
            continue
        run, task, rung = cell.split("__")
        log = open(os.path.join(a.claims, cell + ".log"), errors="replace").read()
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
