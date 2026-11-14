"""Launch doc-drop training runs on jsteinhardt (records/docdrop-nongold-baseline.md), experiment tokdrop_lr.

Every arm is the dense-LoRA arm (`f8k-mt4-denselora64all-*`: whole-model LoRA r64, scale 32, embeddings frozen, LM
head trained) at the recipe's LR 1e-5, 2 GPUs, micro 2, 16 rows/step, one epoch of the fresh 4-task pool. The only
change is the input: a random share of each row's NON-gold documents is removed before block 0 (gold always kept,
oolong whole; build_docdrop_shard.py).

  ddc  compact positions: the shortened rows of the doc-drop shard, trained as plain dense rows
  ddp  original positions: the FULL rows, with the same kept documents chosen by the soft-token compaction from the
       shard's keep_sidecar.json (dropped documents leave the row before block 0; kept tokens keep their
       full-context position ids)

    python research/token_routing/docdrop/launch_docdrop.py --arms ddc --code /data/prasann/docdrop/code_<sha> [--steps 8]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
STATE = f"{HERE}/state.json"
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
LORA = "--lora-rank 64 --lora-blocks 0 --lora-scale 32 --attn-backend flash_2"
SHARD = "/data/prasann/docdrop/shards/mt4{b}_dd{tag}"
FULL = "/data/prasann/oracle/shards/mt4{b}"


def arm_cfg(arm, budget, shard_tag):
    dd = SHARD.format(b=budget, tag=shard_tag)
    if arm == "ddc":
        return "full", dd, LORA
    if arm == "ddp":
        return "softtoken", FULL.format(b=budget), (
            f"{LORA} --st-gold-sidecar {dd}/keep_sidecar.json --st-n-random 0 --st-keep-mode gold_plus_random "
            "--st-drop-slots")
    raise SystemExit(f"unknown arm {arm}")


def run_name(arm, budget, lr, seed, tag):
    return f"f8k-mt4-{arm}{f'sd{seed}' if seed else ''}-{budget}-dd-lr{lr:g}{tag}-loc"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="ddc")
    ap.add_argument("--budget", default="f1952")
    ap.add_argument("--shard-tag", default="", help="suffix of the doc-drop shard dir (mt4<budget>_dd<tag>)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0, help="max steps (smoke); 0 = full epoch + model export")
    ap.add_argument("--node", default="mooney")
    ap.add_argument("--ngpu", type=int, default=2)
    ap.add_argument("--minutes", type=int, default=90)
    ap.add_argument("--resume-every", type=int, default=100)
    ap.add_argument("--code", required=True, help="staged code dir on the node (stage_code.sh)")
    ap.add_argument("--name-tag", default="")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}}
    for arm in a.arms.split(","):
        variant, data, extra = arm_cfg(arm, a.budget, a.shard_tag)
        name = run_name(arm, a.budget, a.lr, a.seed, a.name_tag or (f"-st{a.steps}" if a.steps else ""))
        if not a.steps or a.steps > a.resume_every:
            extra += f" --ephemeral-save-interval {a.resume_every}"
        env = dict(os.environ, RUN=name, VARIANT=variant, DATA=data, ARM_ARGS=extra, CODE=a.code, SEED=str(a.seed),
                   LR=str(a.lr), MAX_STEPS=str(a.steps), MINUTES=str(a.minutes), NGPU=str(a.ngpu))
        if not a.wandb:
            env["NOWANDB"] = "1"
        env.pop("SAVECK", None)
        if not a.steps:
            env["SAVECK"] = "1"
        cmd = ["sbatch", "-w", a.node, "-J", f"dd-{arm}", f"--gres=gpu:{a.ngpu}", f"{HERE}/run_docdrop_local.sbatch"]
        out = subprocess.run(cmd, env=env, capture_output=True, text=True)
        print(name, out.stdout.strip(), out.stderr.strip()[-300:])
        jid = out.stdout.strip().split()[-1] if out.returncode == 0 else None
        if jid:
            git = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
            st["runs"][name] = {"job": jid, "arm": arm, "variant": variant, "data": data, "budget": a.budget,
                                "lr": a.lr, "seed": a.seed, "node": a.node, "git": git, "code": a.code,
                                "extra": extra, "steps": a.steps}
            json.dump(st, open(STATE, "w"), indent=1)
            subprocess.run([OPS, "register", "--run-id", name, "--experiment", "tokdrop_lr", "--cluster", "slurm",
                            "--job-id", jid, "--expected-minutes", str(a.minutes), "--git", REPO])


if __name__ == "__main__":
    main()
