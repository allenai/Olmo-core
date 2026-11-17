"""
recipe_cpt launcher: 32k continued pre-training of the marker-repaired Qwen3.5-4B base on the cpt32k_u1B shard
(1,938 steps x 16 rows x 32,257 tokens = 1.0B row tokens), Beaker, through recipes/train/research_trainer/
beaker_ctc_suite.py. Design and reasons: README.md in this folder.

Arms
  recipe   the FLOP-drop recipe (arm lora64all32r20to7 of ds64_fast8k/launch_fast8k.py): whole-model LoRA r64 /
           scale 32 (base weights and embeddings frozen, LM head trains), online two-stage token router, keep 20%
           of each pseudo-document's body after block 7 and 7% after block 16, no dense steps. CPT-only additions:
           --st-gold-blind (no gold docs in CPT rows) and --st-otr-no-query (no query; olmo3_cptdrop's CPT router
           flags). Dropped tokens leave the stack, so they carry no LM loss: ~7% of body tokens are trained per row.
  dense    full fine-tuning on the same rows (the control Prasann approved 2026-11-17).
Both: LR 2e-5, WSD (3% warmup, constant, linear decay over the last 10%), 16 rows/step (516k tokens), 8 GPUs, 1 node,
permanent checkpoints every 242 steps (124.9M tokens; kept: 242/484/968/1936 = 125M/250M/500M/1B + 726/1210/1452/1694)
plus ephemeral resume points every 121 steps.

    python launch_recipe_cpt.py --arms recipe,dense dry_run
    python launch_recipe_cpt.py --arms recipe --smoke A launch    # 20 of 30 steps, saves at 10 and 20
    python launch_recipe_cpt.py --arms recipe --smoke B launch    # --resume into the same folder: 20 -> 30
    python launch_recipe_cpt.py --arms recipe --min-runtime 7h --expected-minutes 330 launch
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
LAUNCHER = f"{REPO}/recipes/train/research_trainer/beaker_ctc_suite.py"
WEKA = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
SHARD = f"{WEKA}/recipe_cpt/shards/cpt32k_u1B"
BASE = f"{WEKA}/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"  # the base every recipe SFT run starts from
CKPTS = f"{WEKA}/ctc_suite/ckpts"
CLUSTER = os.environ.get("RCPT_CLUSTER", "ai2/jupiter-cirrascale-2")
GPUS = 8
ROWS_PER_STEP = 16
SEQ_LEN = 32768  # rows are 32,257 tokens (63 x 512 + EOS); padded to 32k
SAVE_EVERY = 242  # 242 x 16 x 32,257 = 124.9M tokens
EPHEMERAL_EVERY = 121
LR = 2e-5

RECIPE = ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
          "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
          "--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
          "--st-otr-early-lora-blocks 32")
CPT_EXTRA = "--st-gold-blind --st-otr-no-query --attn-backend flash_2"
ARMS = {"recipe": ("softtoken", f"{RECIPE} {CPT_EXTRA}"), "dense": ("full", "--attn-backend flash_2")}
#: smoke: a complete 30-step schedule (warmup 1 step); A stops at 20 with saves at 10/20, B resumes 20 -> 30
SMOKE_TOKENS = 30 * ROWS_PER_STEP * SEQ_LEN


def run_name(arm, smoke):
    return f"rcpt-q35-4b-{arm}-u1B" + ("-smoke" if smoke else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["launch", "dry_run"])
    ap.add_argument("--arms", default="recipe")
    ap.add_argument("--smoke", choices=["", "A", "B"], default="")
    ap.add_argument("--resume", action="store_true", help="relaunch into an existing save folder (manual restart)")
    ap.add_argument("--min-runtime", default="1h")
    ap.add_argument("--expected-minutes", type=float, default=None)
    ap.add_argument("--wandb-group", default="rcpt-q35-4b")
    a = ap.parse_args()
    for arm in a.arms.split(","):
        variant, extra = ARMS[arm]
        name = run_name(arm, a.smoke)
        common = f"--save-interval {SAVE_EVERY if not a.smoke else 10} --max-checkpoints 0 --scheduler wsd --wsd-decay-fraction 0.1"
        if a.smoke:
            common += f" --max-tokens {SMOKE_TOKENS} --warmup-fraction 0.04"
            if a.smoke == "A":
                common += " --max-steps 20"
            else:
                common += " --resume"
        else:
            common += f" --ephemeral-save-interval {EPHEMERAL_EVERY}" + (" --resume" if a.resume else "")
        cmd = [sys.executable, "-u", LAUNCHER, "--task", "cpt", "--variant", variant,
               "--model-family", "qwen3_5", "--model-scale", "4b", "--data-root", SHARD,
               "--run-name", name, "--exact-run-name", "--num-nodes", "1", "--num-gpus", str(GPUS),
               "--epochs", "1", "--lr", str(LR), "--cluster", CLUSTER, "--wandb-group", a.wandb_group,
               "--no-follow", "--no-compile", "--seq-len", str(SEQ_LEN), "--global-batch", str(ROWS_PER_STEP),
               "--micro-batch-instances", "1", "--base-checkpoint", BASE, "--experiment", "recipe_cpt",
               "--arm", arm, "--min-runtime", a.min_runtime, "--priority", "urgent",
               "--extra-args", f"{common} {extra}", a.mode]
        if a.expected_minutes:
            cmd[-1:-1] = ["--expected-minutes", str(a.expected_minutes)]
        print(" ".join(cmd), flush=True)
        res = subprocess.run(cmd, cwd=REPO, env=dict(os.environ, PYTHONPATH=f"{REPO}/src"), capture_output=True, text=True)
        out = res.stdout + res.stderr
        os.makedirs(f"{REPO}/research/token_routing/recipe_cpt/launch_logs", exist_ok=True)
        open(f"{REPO}/research/token_routing/recipe_cpt/launch_logs/{name}{a.smoke}.{a.mode}.log", "w").write(out)
        for line in out.splitlines():
            if line.startswith(("[ops]", "[beaker_ctc_suite] SUBMITTED", "[beaker_ctc_suite] save_folder", "[beaker_ctc_suite] url")):
                print("  " + line, flush=True)
        if res.returncode:
            print(out[-3000:], flush=True)
            raise SystemExit(f"launch failed for {name} (rc={res.returncode})")


if __name__ == "__main__":
    main()
