"""Metadata lines for a run trained OFF Beaker (cubbins / Lambda copies), in the sweep_meta.jsonl format collect_sweep.py
reads: METAJSON (summarize.py --oneline windows of oracle_meta/steps.jsonl; oracle arms only), FLOPJSON (flops.json) and
LOGJSON (parse_log of the training log: PF, steps, router keep, train CE, resume/restart counts). Stdlib only, so it runs
on a login node (reading a /net/<node> path) or on the Lambda node itself.

    python3 local_meta.py <run name> <run dir> <training log> [<log> ...] >> sweep_meta.jsonl
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_log(log: str) -> dict:
    """PF, steps, routed keep (router log) and train CE of a training log (Beaker or slurm)"""
    pf = re.findall(r"flop_meter/actual_pflops[ =]+([0-9.]+)", log)
    over = re.findall(r"flop_meter/actual_over_dense[ =]+([0-9.]+)", log)
    steps = [int(s) for s in re.findall(r"\[step=(\d+)/\d+", log)]
    keep = {}
    for s, k in re.findall(r"\[online-router\] step (\d+): dense=0 .*?keep_frac=([0-9.e-]+)", log):
        keep[int(s)] = float(k)  # last record per step (a resume replays steps)
    ce = [float(x) for x in re.findall(r"train/CE loss=([0-9.]+)", log)]
    return {"pf": float(pf[-1]) if pf else None, "over_dense": float(over[-1]) if over else None,
            "steps": max(steps) if steps else None, "keep": sorted(keep.items()), "ce": ce,
            "resumes": len(re.findall(r"Will resume training from step", log)),
            "starts": len(re.findall(r"No checkpoint found in save folder", log)),
            "done": "saved model-only checkpoint" in log}


def main():
    run, rdir, logs = sys.argv[1], sys.argv[2], sys.argv[3:]
    steps = os.path.join(rdir, "oracle_meta", "steps.jsonl")
    if os.path.exists(steps):
        out = subprocess.run([sys.executable, os.path.join(HERE, "summarize.py"), steps, "--every", "100000",
                              "--oneline", run], capture_output=True, text=True).stdout
        print(next(line for line in out.splitlines() if line.startswith("METAJSON")))
    fj = os.path.join(rdir, "flops.json")
    if os.path.exists(fj):
        print(f"FLOPJSON {run} " + json.dumps(json.load(open(fj))))
    text = "".join(open(p, errors="ignore").read() for p in logs if os.path.exists(p))
    if text:
        print(f"LOGJSON {run} " + json.dumps(parse_log(text)))


if __name__ == "__main__":
    main()
