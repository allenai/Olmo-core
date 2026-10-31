"""Run the early-phase recipe training on jsteinhardt (the Beaker copy pends on jupiter): the f1952-tde recipe config
(launch_fast8k.py arm lora64all32r20to7, LR 1e-5, budget f1952 hard-stopped at step 244) through run_ctc_local.sbatch,
with permanent checkpoints at the probe steps and ephemeral resume points.

Needs, on the node: /data/prasann/gradconc/{shards/mt4f1952, ckpts/q35-4b-base-markerfix} (run_gap_local.sbatch
STAGE_ONLY=1 TRAIN_SHARDS=mt4f1952 CKNAME=q35-4b-base-markerfix) and the TRAIN-branch code (claude/grad-routing-gap-train,
which has the online router + --save-fixed-steps) exported to CODE.

    python3 research/token_routing/grad_concentration/submit_train_local.py --node sneetches --code /scratch/.../code \
        [--dependency afterok:<stage job>]
"""
import argparse
import os
import subprocess

RUN = "f8k-mt4-lora64all32r20to7-f1952st244-tdgap-lr1e-05-local"
EXTRA = ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
         "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
         "--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
         "--st-otr-early-lora-blocks 32 --attn-backend flash_2 "
         "--save-fixed-steps 0,8,16,31,61,122,244 --resume-every 20 --max-checkpoints 0")

ap = argparse.ArgumentParser()
ap.add_argument("--node", default="sneetches")
ap.add_argument("--qos", default="preemptive_high")
ap.add_argument("--code", required=True)
ap.add_argument("--dependency", default="")
a = ap.parse_args()
env = {**os.environ, "REPO": a.code, "TASK": "mt4", "DATA_SRC": "/data/prasann/gradconc/shards/mt4f1952",
       "VARIANT": "softtoken", "SCALE": "4b", "MODEL_FAMILY": "qwen3_5", "RUN": RUN, "EPOCHS": "1",
       "SEQ_LEN": "12288", "LR": "1e-05", "GLOBAL_BATCH": "16", "MICRO_BATCH": "2", "MAX_STEPS": "244", "NGPU": "2",
       "SEED": "0", "WANDB_GROUP": "f8k-q35-4b", "BASE_SRC": "/data/prasann/gradconc/ckpts/q35-4b-base-markerfix",
       "NO_FINAL_CKPT": "1", "EXTRA": EXTRA, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
cmd = ["sbatch", "--partition=jsteinhardt", f"--qos={a.qos}", "-w", a.node, "--gres=gpu:2", "--cpus-per-task=16",
       "--mem=200G", "--time=08:00:00", "-J", RUN, "--export=ALL"]
if a.dependency:
    cmd.append(f"--dependency={a.dependency}")
cmd.append(os.path.join(a.code, "recipes/train/research_trainer/run_ctc_local.sbatch"))
out = subprocess.run(cmd, capture_output=True, text=True, env=env)
print(out.stdout.strip(), out.stderr.strip()[-500:])
