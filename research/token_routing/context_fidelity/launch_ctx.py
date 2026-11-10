"""Launch context_probe.py shards on Beaker (jupiter, 1 GPU, urgent, minRuntime 1h, ai2/flex2 / ai2/oe-other) and register
each with ops. Sketches go to weka CTX_DIR/<label>/; analyze_context.py reads them. Gantry runs the PUSHED commit.

    python research/token_routing/context_fidelity/launch_ctx.py --labels recipe --num-shards 4
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PY = "/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python"
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
CKPTS = {"f1952": (f"{W}/ctc_suite/ckpts/f8k-mt4-dense-f1952-tde-lr5e-06", f"{W}/gradgap/adam_f1952.pt"),
         "recipe": (f"{W}/ctc_suite/ckpts/f8k-mt4-lora64all32r20to7-f1952-tde-lr1e-05", f"{W}/gradgap/adam_recipe.pt")}
CTX_DIR = f"{W}/ctxfid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="recipe")
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--shards-only", default="")
    ap.add_argument("--extra", default="")
    ap.add_argument("--minutes", type=int, default=150)
    ap.add_argument("--cluster", default="ai2/jupiter-cirrascale-2")
    a = ap.parse_args()
    for lab in a.labels.split(","):
        ck, adam = CKPTS[lab]
        only = [int(x) for x in a.shards_only.split(",")] if a.shards_only else range(a.num_shards)
        for sh in only:
            extra = (f"--ckpt {ck} --label {lab} --num-shards {a.num_shards} --shard {sh} --save-dir {CTX_DIR} "
                     f"--adam-factors {adam} {a.extra}").strip()
            cmd = [PY, "-u", f"{REPO}/research/ffn_compute/flop_scaling/beaker_bench_launch.py", "--cluster", a.cluster,
                   "--gpus", "1", "--script", "research/token_routing/context_fidelity/context_probe.py", "--extra", extra,
                   "--min-runtime", "1h", "--priority", "urgent"]
            out = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=900,
                                 env={**os.environ, "PYTHONPATH": f"{REPO}/src",
                                      "PATH": f"{os.path.expanduser('~/.local/bin')}:{os.environ['PATH']}"})
            m = re.search(r"SUBMITTED id=(\S+)", out.stdout)
            if not m:
                print(f"{lab} s{sh}: launch FAILED\n{(out.stdout + out.stderr)[-1500:]}")
                continue
            ex = m.group(1)
            run_id = f"ctxfid-{lab}-s{sh}of{a.num_shards}-{ex[-6:]}"
            r = subprocess.run([OPS, "register", "--run-id", run_id, "--experiment", "tokdrop_lr", "--cluster", "beaker",
                                "--kind", "eval", "--job-id", ex, "--expected-minutes", str(a.minutes), "--git", REPO,
                                "--notes", "context-fidelity probe (no training)"], capture_output=True, text=True)
            print(f"{lab} s{sh} -> {ex} {run_id} (ops register rc={r.returncode} {r.stderr.strip()[-200:]})")


if __name__ == "__main__":
    sys.exit(main())
