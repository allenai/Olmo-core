"""Memory-vs-row-length sweep (memscale): one training step at micro-batch 1 on 1 GPU per (arm, length), activation
checkpointing on or off, through launch_proxy.py (one slurm job per --ac value, arms back to back, OOM = result).

    MEMTRACE=2,3 python memscale_launch.py --code /data/prasann/early_loss_proxy/code_<sha> --node mooney \\
        --ac full --lengths 16384,32768,65536,131072,262144 --arms dl,rc [--extra-arm 'name=variant,flags']
Rows: memscale_build.py pools at /data/prasann/early_loss_proxy/shards/memscale/L<L>.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from launch_proxy import LORA, RECIPE  # noqa: E402

BASE_ARMS = {"dl": ("full", LORA), "rc": ("softtoken", RECIPE)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--node", default="mooney")
    ap.add_argument("--qos", default="preemptive")
    ap.add_argument("--ac", default="full", choices=["full", "none"])
    ap.add_argument("--lengths", default="16384,32768,65536,131072,262144")
    ap.add_argument("--arms", default="dl,rc")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=VARIANT,FLAGS", help="extra base arm")
    ap.add_argument("--tag", default="memscale")
    ap.add_argument("--steps", default="3")
    ap.add_argument("--minutes", type=int, default=60)
    a = ap.parse_args()
    for x in a.arm:
        n, _, rest = x.partition("=")
        v, fl = rest.split(",", 1)
        BASE_ARMS[n] = (v, fl)
    names, cust = [], []
    for L in [int(x) for x in a.lengths.split(",")]:
        for arm in a.arms.split(","):
            var, fl = BASE_ARMS[arm]
            nm = f"{arm}{L // 1024}k{'ac' if a.ac == 'full' else 'noac'}"
            names.append(nm)
            cust += ["--custom", f"{nm}={var},1e-5,{fl} --data /data/prasann/early_loss_proxy/shards/memscale/L{L} "
                     f"--seq-len {L} --global-batch 1 --micro-batch-instances 1 --activation-checkpointing {a.ac}"]
    cmd = [sys.executable, f"{HERE}/launch_proxy.py", "--code", a.code, "--node", a.node, "--qos", a.qos,
           "--tag", a.tag, "--seeds", "0", "--steps", a.steps, "--gpus", "1", "--sets", "suite4k:1",
           "--minutes", str(a.minutes), "--arms", ",".join(names)] + cust
    print(" ".join(cmd[:20]), "...", flush=True)
    raise SystemExit(subprocess.call(cmd, env=dict(os.environ, MEMTRACE=os.environ.get("MEMTRACE", "2,3"))))


if __name__ == "__main__":
    main()
