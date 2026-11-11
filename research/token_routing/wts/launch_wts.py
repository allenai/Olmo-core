"""Launch / evaluate weighted-token-sampling runs (records/weighted-token-sampling.md), experiment tokdrop_lr.

Every arm is the recipe ``lora64all32r20to7`` (``research/token_routing/ds64_fast8k/launch_fast8k.py``) flag for flag --
repaired Qwen3.5-4B base, whole-model LoRA r64 (scale 32), LR 1e-5, 2 GPUs, micro 2, 16 rows/step, the fresh 4-task
pool, cuts after block 7 (20%) and 16 (7%), per-document budgets -- with the stage's change on top (``--st-otr-wts``).
Every wts arm also routes its warmup step at the budget with uniform scores (``--st-otr-warmup-random``): no dense
step anywhere in training. Run tags start with ``wts``; names ``f8k-mt4-<arm>[sdN]-f<rows>-wts-lr1e-05``.

    python research/token_routing/wts/launch_wts.py --arms wtsS1 --budgets f1952 launch        # Beaker jupiter
    python research/token_routing/wts/launch_wts.py --arms wtsS1 --budgets f1952 --steps 150 local --node mooney
    python research/token_routing/wts/launch_wts.py --runs <run> eval                           # 4 tasks x 8k+2k x 500
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "research", "token_routing", "ds64_fast8k"))
sys.path.insert(0, os.path.join(REPO, "research", "token_routing", "ds64_fast2k"))
from launch_fast2k import BASE, CKPTS, W, parse_id, sh  # noqa: E402
from launch_fast8k import EVAL_LAUNCHER, MT4_TASKS, RUNGS, TASK_KEY, TOKENIZER  # noqa: E402

LAUNCHER = f"{REPO}/recipes/train/research_trainer/beaker_ctc_suite.py"
SHARDS = f"{W}/ds64/fast8k/shards"
STATE = f"{HERE}/state.json"
SUFFIX = "-wts"
WANDB_GROUP = os.environ.get("WTS_WANDB_GROUP", "f8k-wts")
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"

#: the recipe lora64all32r20to7, flag for flag (launch_fast8k.ARM_EXTRA)
RECIPE = ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
          "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
          "--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
          "--st-otr-early-lora-blocks 32 --attn-backend flash_2")
NODENSE = "--st-otr-warmup-random"


def wts(spec, stages=None):
    base = RECIPE if stages is None else RECIPE.replace("--st-otr-stages 7:0.2,16:0.07", f"--st-otr-stages {stages}")
    return f"{base} {NODENSE} --st-otr-wts {spec}"


ARMS = {
    # S0: the recipe from this branch (its step 1 trains dense, as the recipe does) and with the warmup routed
    "wtsS0": RECIPE,
    "wtsS0wr": f"{RECIPE} {NODENSE}",
    # S1: same cuts and per-document budgets, sampled (p ~ exp(score) water-filled, 10% of each budget uniform), no weights
    "wtsS1": wts("sample=1,weights=none"),
    # S2: + token weights w = 1/p: attention bias log w, DeltaNet w-fold write, both
    "wtsS2a": wts("sample=1,weights=attn"),
    "wtsS2g": wts("sample=1,weights=gdn"),
    "wtsS2b": wts("sample=1,weights=both"),
    # S3: the price rule p = clip(w_prev exp(score) / (tau sqrt(c_l)), pmin, 1), one tau per step from the FLOP controller
    # (target = the recipe's measured routed step ratio 0.216); the stage keeps become per-row caps on expected survivors
    "wtsS3": wts("sample=1,weights=both,price=0.216,center=none", stages="7:0.4,16:0.2"),
    "wtsS3d": wts("sample=1,weights=both,price=0.216,center=none,alloc=doc", stages="7:0.4,16:0.2"),
    "wtsS3r": wts("sample=1,weights=both,price=0.216,center=row", stages="7:0.4,16:0.2"),
    # round 2 (after the S1-S3 evals: outlier drops when selection is soft): sharper scores, attention-only weights
    "wtsS2aT5": wts("sample=1,weights=attn,temp=0.5"),
    "wtsS2aT2": wts("sample=1,weights=attn,temp=0.2"),
    "wtsS1T2": wts("sample=1,weights=none,temp=0.2"),
    "wtsS3aT5": wts("sample=1,weights=attn,price=0.216,center=none,temp=0.5", stages="7:0.4,16:0.2"),
    # S3 at more cuts: every full-attention block from 7 on
    "wtsS3m": wts("sample=1,weights=both,price=0.216,center=none",
                  stages="7:0.4,11:0.3,15:0.2,19:0.2,23:0.2,27:0.2"),
}


def arm_extra(arm):
    if arm in ARMS:
        return ARMS[arm]
    raise SystemExit(f"unknown arm {arm} (known: {', '.join(ARMS)})")


def run_name(arm, budget, lr, seed=0, tag=""):
    return f"f8k-mt4-{arm}{f'sd{seed}' if seed else ''}-{budget.replace('@', 'st')}{SUFFIX}-lr{lr:g}{tag}"


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}, "evals": {}}


def save(st):
    json.dump(st, open(STATE + ".tmp", "w"), indent=1)
    os.replace(STATE + ".tmp", STATE)


def git_head():
    return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def do_launch(a, mode):
    st = load()
    for arm in a.arms.split(","):
        for budget in a.budgets.split(","):
            name = run_name(arm, budget, a.lr, a.seed, a.name_tag)
            if name in st["runs"] and mode == "launch" and not a.force:
                print(f"[skip] {name} ({st['runs'][name]['ex']})")
                continue
            rows, _, steps = budget[1:].partition("@")
            data = f"{SHARDS}/mt4{budget[0]}{rows}"
            n_steps = int(steps) if steps else (int(rows) * 4) // 16
            # Beaker preempts soon after minRuntime: min(4h, 1.3 x expected) and a resume point every 100 steps
            exp_min = int(a.minutes or max(30, 0.45 * n_steps))
            minrt = f"{min(240, int(1.3 * exp_min))}m"
            cmd = [sys.executable, "-u", LAUNCHER, "--task", "mt4", "--variant", "softtoken",
                   "--model-family", "qwen3_5", "--model-scale", "4b", "--data-root", data,
                   "--run-name", name, "--exact-run-name", "--num-nodes", "1", "--num-gpus", "2",
                   "--epochs", "1", "--lr", str(a.lr), "--cluster", a.cluster,
                   "--wandb-group", WANDB_GROUP, "--priority", "urgent", "--min-runtime", minrt, "--no-follow",
                   "--no-compile", "--seq-len", str(a.seq_len), "--global-batch", "16",
                   "--micro-batch-instances", "2", "--seed", str(a.seed), "--base-checkpoint", BASE,
                   "--experiment", "tokdrop_lr"]
            if steps:
                cmd += ["--max-steps", steps]
            extra = arm_extra(arm) + (f" {a.extra}" if a.extra else "")
            if a.resume_every:
                extra = f"{extra} --ephemeral-save-interval {a.resume_every}".strip()
            cmd += ["--extra-args", extra]
            rc, out = sh(cmd + [mode])
            ex = parse_id(out)
            print(f"{name} -> rc={rc} {ex or out[-600:]}")
            if mode == "launch" and ex:
                st["runs"][name] = {"ex": ex, "arm": arm, "budget": budget, "lr": a.lr, "seed": a.seed,
                                    "cluster": "beaker", "git": git_head(), "extra": extra}
                save(st)  # beaker_ctc_suite.py registers the run with ops itself (--experiment tokdrop_lr)


def do_local(a):
    """sbatch on a jsteinhardt node (code staged on the node by stage_code.sh; base + shard on its /data)."""
    st = load()
    for arm in a.arms.split(","):
        budget = a.budgets.split(",")[0]
        name = run_name(arm, budget, a.lr, a.seed, a.name_tag or (f"-st{a.steps}" if a.steps else "")) + "-loc"
        extra = arm_extra(arm) + (f" {a.extra}" if a.extra else "")
        if a.resume_every and (not a.steps or a.steps > a.resume_every):  # preemptive QOS: requeued jobs resume
            extra = f"{extra} --ephemeral-save-interval {a.resume_every}"
        env = dict(os.environ, RUN=name, ARM_ARGS=extra, CODE=a.code, SEED=str(a.seed), LR=str(a.lr),
                   MAX_STEPS=str(a.steps or 0), MINUTES=str(a.minutes or 60), NGPU=str(a.ngpu))
        cmd = ["sbatch", "-w", a.node, "-J", f"wts-{arm}", f"--gres=gpu:{a.ngpu}", f"{HERE}/run_wts_local.sbatch"]
        out = subprocess.run(cmd, env=env, capture_output=True, text=True)
        print(name, out.stdout.strip(), out.stderr.strip()[-300:])
        jid = out.stdout.strip().split()[-1] if out.returncode == 0 else None
        if jid:
            st["runs"][name] = {"ex": jid, "arm": arm, "budget": budget, "lr": a.lr, "seed": a.seed,
                                "cluster": f"slurm:{a.node}", "git": git_head(), "extra": extra, "steps": a.steps}
            save(st)
            sh([OPS, "register", "--run-id", name, "--experiment", "tokdrop_lr", "--cluster", "slurm",
                "--job-id", jid, "--expected-minutes", str(a.minutes or 60), "--git", REPO])


def do_eval(a):
    st = load()
    for name in a.runs.split(","):
        for task in MT4_TASKS:
            ename = f"{name}-ev{task}"
            if ename in st["evals"] and not a.force:
                print(f"[skip] {ename}")
                continue
            cmd = [sys.executable, "-u", EVAL_LAUNCHER, ename, a.cluster,
                   "--task", TASK_KEY[task], "--variant", "docchunk", "--ckpt", f"{CKPTS}/{name}",
                   "--query-position", "after", "--cot-mode", "none", "--tokenizer", TOKENIZER,
                   "--ngpu", "2", "--max-test", "500", "--priority", "urgent", "--max-length", "20000",
                   "--batch-size", "8", "--dc-rung-files", json.dumps({task: RUNGS[task]}),
                   "--dc-rungs", ",".join(RUNGS[task])]
            rc, out = sh(cmd)
            ex = parse_id(out)
            print(f"eval {ename} -> {ex or 'FAILED: ' + out[-400:]}")
            if ex:
                st["evals"][ename] = {"ex": ex, "train_run": name, "task": task}
                save(st)
                sh([OPS, "register", "--run-id", ename, "--experiment", "tokdrop_lr", "--cluster", "beaker",
                    "--job-id", ex, "--kind", "eval", "--parent-run", name, "--expected-minutes", "60", "--git", REPO])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="wtsS1")
    ap.add_argument("--budgets", default="f1952", help="f<rows/task>[@<max steps>] (fresh pool shard mt4f<rows>)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--resume-every", type=int, default=100)
    ap.add_argument("--name-tag", default="")
    ap.add_argument("--extra", default="", help="appended to the arm's trainer flags")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=12288)
    ap.add_argument("--minutes", type=int, default=0, help="expected minutes (minRuntime = min(4h, 1.3x this))")
    ap.add_argument("--cluster", default="ai2/jupiter-cirrascale-2")
    ap.add_argument("--runs", default="")
    ap.add_argument("--force", action="store_true")
    # local
    ap.add_argument("--node", default="mooney")
    ap.add_argument("--ngpu", type=int, default=2)
    ap.add_argument("--steps", type=int, default=0, help="local: max steps (short proxy)")
    ap.add_argument("--code", default="", help="local: staged code dir on the node (stage_code.sh)")
    ap.add_argument("mode", choices=["launch", "dry_run", "eval", "local"])
    a = ap.parse_args()
    if a.mode == "eval":
        do_eval(a)
    elif a.mode == "local":
        if not a.code:
            raise SystemExit("--code <staged dir on the node> (stage_code.sh <node>)")
        do_local(a)
    else:
        do_launch(a, a.mode)


if __name__ == "__main__":
    main()
