"""
Downstream question of recipe_cpt: does the recipe's FLOPs-to-match-dense improve when the base was CPT'd under the
same token dropping? For one exported CPT base (post_ckpt_beaker.sh -> $WEKA/recipe_cpt/bases/<tag>/model_and_optim),
train the two SFT arms of the 8k primary screen on mt4 s1952 (1,952 rows/task, 4 tasks, 16 rows/step, 2 GPUs) with
EXACTLY the flags of the reference runs, only --base-checkpoint changed:

  recipe  f8k-mt4-lora64all32r20to7-s1952-tde-lr1e-05   (softtoken, LR 1e-5, seq 12288, micro 2)
  dense   f8k-mt4-dense-s1952-tdc-lr5e-06               (full fine-tune, LR 5e-6, seq 10240, micro 2)

and score each on the 8k + 2k rungs of the four tasks (500 rows each, no-cot, chat prompt, query position after),
the same eval launch_fast8k.py runs. FTM is then read against a dense curve FROM THE SAME BASE: for a full answer also
train dense at s976 / s3904 (--dense-budgets), else compare against the original-base dense curve and say so.

    python launch_sft_from_cpt.py --base-tag rcpt-q35-4b-recipe-u1B-step242 --arms recipe,dense dry_run
    python launch_sft_from_cpt.py --base-tag rcpt-q35-4b-recipe-u1B-step242 --arms recipe,dense launch
    python launch_sft_from_cpt.py --base-tag rcpt-q35-4b-recipe-u1B-step242 --arms recipe,dense eval   # after training
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, f"{REPO}/research/token_routing/ds64_fast8k")
from launch_fast8k import RUNGS, TASK_KEY, TOKENIZER  # noqa: E402  (same eval files as the 8k screen)

W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
LAUNCHER = f"{REPO}/recipes/train/research_trainer/beaker_ctc_suite.py"
EVAL_LAUNCHER = f"{REPO}/recipes/eval/legacy_native/run_q4b_beaker_multirung_eval.py"
SHARDS = f"{W}/ds64/fast8k/shards"
CLUSTER = "ai2/jupiter-cirrascale-2"
TASKS = ["contradiction", "nq", "outlier", "oolong"]
RECIPE = ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
          "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
          "--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
          "--st-otr-early-lora-blocks 32 --attn-backend flash_2")
ARMS = {  # arm: (variant, lr, seq_len, extra)
    "recipe": ("softtoken", "1e-05", "12288", RECIPE),
    "dense": ("full", "5e-06", "10240", ""),
}


def run_name(tag, arm, budget):
    return f"sftcpt-{tag}-{arm}-{budget}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["launch", "dry_run", "eval"])
    ap.add_argument("--base-tag", required=True, help="dir name under $WEKA/recipe_cpt/bases/")
    ap.add_argument("--arms", default="recipe,dense")
    ap.add_argument("--budget", default="s1952")
    ap.add_argument("--dense-budgets", default="", help="extra dense budgets from the same base, e.g. s976,s3904")
    a = ap.parse_args()
    base = f"{W}/recipe_cpt/bases/{a.base_tag}/model_and_optim"
    jobs = [(arm, a.budget) for arm in a.arms.split(",")] + [("dense", b) for b in a.dense_budgets.split(",") if b]
    for arm, budget in jobs:
        variant, lr, seq, extra = ARMS[arm]
        name = run_name(a.base_tag, arm, budget)
        if a.mode == "eval":
            ckpt = f"{W}/ctc_suite/ckpts/{name}"
            for task in TASKS:
                cmd = [sys.executable, "-u", EVAL_LAUNCHER, f"{name}-ev{task}", CLUSTER, "--task", TASK_KEY[task],
                       "--variant", "docchunk", "--ckpt", ckpt, "--query-position", "after", "--cot-mode", "none",
                       "--tokenizer", TOKENIZER, "--ngpu", "2", "--max-test", "500", "--priority", "urgent",
                       "--max-length", "20000", "--batch-size", "8",
                       "--dc-rung-files", json.dumps({task: RUNGS[task]}), "--dc-rungs", ",".join(RUNGS[task])]
                print(" ".join(cmd), flush=True)
                subprocess.run(cmd, cwd=REPO, env=dict(os.environ, PYTHONPATH=f"{REPO}/src"))
            continue
        cmd = [sys.executable, "-u", LAUNCHER, "--task", "mt4", "--variant", variant, "--model-family", "qwen3_5",
               "--model-scale", "4b", "--data-root", f"{SHARDS}/mt4{budget}", "--run-name", name, "--exact-run-name",
               "--num-nodes", "1", "--num-gpus", "2", "--epochs", "1", "--lr", lr, "--cluster", CLUSTER,
               "--wandb-group", "sftcpt-q35-4b", "--priority", "urgent", "--min-runtime", "2h", "--no-follow",
               "--no-compile", "--seq-len", seq, "--global-batch", "16", "--micro-batch-instances", "2",
               "--base-checkpoint", base, "--experiment", "recipe_cpt", "--arm", f"sft-{arm}",
               "--expected-minutes", "90"] + (["--extra-args", extra] if extra else []) + [a.mode]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO, env=dict(os.environ, PYTHONPATH=f"{REPO}/src"))


if __name__ == "__main__":
    main()
