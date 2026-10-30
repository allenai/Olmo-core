"""Record finished Beaker ladder evals of launch_fast8k runs into ops results (the Beaker eval launcher does not).

For every eval in state.json whose name contains one of --suffix (default -tdc and -tdp) and whose (train run, task)
has no ops result yet: fetch its Beaker log, parse every `[ladder:<task>@<rung>] <metric>=<v> (n=<N>` line, and
append one row per rung. Writes a JSONL and runs `ops result add --json <file> --check`.

    python research/token_routing/ds64_fast8k/record_beaker_evals.py --out /path/rows.jsonl
"""

import argparse
import json
import math
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
BEAKER = os.path.expanduser("~/.local/bin/beaker")
CKPTS = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts"
LINE = re.compile(r"\[ladder:([a-z_]+)@(\d+k)\] ([a-z0-9_]+)=(-?[0-9.]+) \(n=(\d+)")
KEY_TASK = {"contra": "contradiction"}


def recorded():
    out = subprocess.run([OPS, "result", "show", "--json"], capture_output=True, text=True, timeout=120).stdout
    rows = json.loads(out) if out.strip().startswith("[") else []
    return {(r.get("train_run"), r["task"], r["rung"]) for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--suffix", nargs="+", default=["-tdc", "-tdp"])
    ap.add_argument("--experiment", default="tokdrop_lr")
    ap.add_argument("--state", default=os.path.join(HERE, "state.json"),
                    help="launcher state.json to read evals from (e.g. research/token_routing/multiview/state.json)")
    a = ap.parse_args()
    st = json.load(open(a.state))
    have = recorded()
    rows, fetched, empty = [], 0, 0
    for ename, e in st.get("evals", {}).items():
        run, ex = e.get("train_run"), e.get("ex")
        if not run or not ex or not any(s in ename for s in a.suffix):
            continue
        task = KEY_TASK.get(e.get("task", ""), e.get("task", ""))
        if (run, task, "8k") in have and (run, task, "2k") in have:
            continue
        log = subprocess.run([BEAKER, "experiment", "logs", ex], capture_output=True, text=True, timeout=300).stdout
        fetched += 1
        found = 0
        for m in LINE.finditer(log):
            t, rung, metric, v, n = m.groups()
            t = KEY_TASK.get(t, t)
            if (run, t, rung) in have:
                continue
            v, n = float(v), int(n)
            se = math.sqrt(max(v * (1 - v), 1e-9) / n) if 0 <= v <= 1 else None
            rows.append(dict(run_id=ename, train_run=run, experiment=a.experiment, task=t, rung=rung, metric=metric,
                             value=v, eval_size=n, se=se and round(se, 6), cot_label="no-cot",
                             eval_backend="olmo-core-native-docchunk-ladder (beaker)", ckpt_path=f"{CKPTS}/{run}",
                             eval_job_id=ex))
            have.add((run, t, rung))
            found += 1
        empty += found == 0
    with open(a.out, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    print(f"fetched {fetched} eval logs ({empty} without scores yet); {len(rows)} new rows -> {a.out}")
    if rows:
        r = subprocess.run([OPS, "result", "add", "--json", a.out, "--check"], capture_output=True, text=True)
        print((r.stdout + r.stderr).strip().splitlines()[-1])


if __name__ == "__main__":
    main()
