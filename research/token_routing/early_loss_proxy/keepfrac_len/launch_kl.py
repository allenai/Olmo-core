"""Launch keepfrac_len arms on Beaker: one 2-GPU job per arm via ``launch_proxy_beaker.py`` (same payload, same probe).

Every arm at a length L trains on the single-length pool ``kl<L>`` (3 tasks, 1,200 rows each, 16 rows/step, one epoch
= 225 steps, the trainer's real schedule over that epoch) and probes the dense forward on ``heldout_kl.npz`` set
``kl<L>`` (48 rows per task).

    python3 keepfrac_len/launch_kl.py --len 32k --arms ref,k20,k10 --seeds 0 [--steps ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ELP = os.path.dirname(HERE)
W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/early_loss_proxy"
LENS = {"2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768, "64k": 65536}
# Trainer --seq-len per length: the short pools are rung-labelled and overshoot their label (max row 3,367 / 5,354 /
# 9,819 tokens at 2k / 4k / 8k; means 2,215 / 4,092 / 8,246), and the trainer refuses rows longer than --seq-len.
SEQ = {"2k": 4096, "4k": 6144, "8k": 10240, "16k": 16384, "32k": 32768, "64k": 65536}
SHORT = ("2k", "4k", "8k")  # held-out sets kl2k/kl4k/kl8k live in heldout_kls.npz, kl16k/32k/64k in heldout_kl.npz
L_ROOT = "/data/prasann/early_loss_proxy"  # node-local copies (mooney) for --local
LORA = "--lora-rank 64 --lora-blocks 0 --lora-scale 32 --attn-backend flash_2"


def router(k1, k2):
    """The recipe's router (launch_proxy.RECIPE) with cut 1 (after block 7) keeping k1 and cut 2 (after block 16)
    keeping k2 of the row's tokens."""
    return (f"--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac {k1:g} "
            "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
            f"--st-otr-stages 7:{k1:g},16:{k2:g} --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
            "--st-otr-early-lora-blocks 32 --attn-backend flash_2")


GRJ = "--st-gdd-keep-range 0.05,0.5 --st-gdd-whole-frac 0.15"  # loop/goldrand arm grJ


def arms(L, root=W):
    pool = f"{root}/shards/kl{L}"
    a = {
        "dft": ("full", 5e-6, "--attn-backend flash_2"),   # dense full fine-tune: the primary reference (Prasann 11-17)
        "ref": ("full", 1e-5, LORA),                       # dense LoRA reference (secondary)
        "k70": ("softtoken", 1e-5, router(0.70, 0.245)),   # higher keep (short lengths)
        "k40": ("softtoken", 1e-5, router(0.40, 0.14)),
        "k20": ("softtoken", 1e-5, router(0.20, 0.07)),    # the recipe
        "k10": ("softtoken", 1e-5, router(0.10, 0.035)),
        "k5": ("softtoken", 1e-5, router(0.05, 0.0175)),
        "k2p5": ("softtoken", 1e-5, router(0.025, 0.00875)),
        "k1p25": ("softtoken", 1e-5, router(0.0125, 0.0045)),
        # one cut changed at a time (the other at the recipe's value)
        "c1k10": ("softtoken", 1e-5, router(0.10, 0.07)),
        "c2k3p5": ("softtoken", 1e-5, router(0.20, 0.035)),
        "c1k5": ("softtoken", 1e-5, router(0.05, 0.05)),
        "c2k1p75": ("softtoken", 1e-5, router(0.20, 0.0175)),
        # experiment 2
        "grJ": ("softtoken", 1e-5, f"{router(0.20, 0.07)} {GRJ}"),
        "grJc": ("softtoken", 1e-5, f"{router(0.20, 0.07)} {GRJ} --st-gdd-only-fps {pool}/gdd_only_contradiction.json"),
    }
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--len", required=True, choices=list(LENS))
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", required=True, help="probe steps (the run hard-stops at the last one)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--micro", type=int, default=0, help="micro-batch rows per GPU (default 2 at 16k/32k, 1 at 64k)")
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--minutes", type=int, required=True, help="expected minutes per arm run (sizes minRuntime / slurm --time)")
    ap.add_argument("--ho-rows", type=int, default=48)
    ap.add_argument("--heldout", default="", help="held-out npz basename (default heldout_kls.npz at 2k/4k/8k, else heldout_kl.npz)")
    ap.add_argument("--local", default="", metavar="CODE_DIR",
                    help="run on a jsteinhardt node via ../launch_proxy.py with this staged code dir (stage_code.sh) "
                         "instead of Beaker; pool / held-out from the node's /data")
    ap.add_argument("--node", default="mooney")
    ap.add_argument("--qos", default="preemptive_high")
    ap.add_argument("--group", type=int, default=2, help="--local: arm runs back to back per slurm job")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    L = a.len
    tag = a.tag or f"kl{L}"
    A = arms(L)
    micro = a.micro or (1 if L == "64k" else 2)
    ho = a.heldout or ("heldout_kls.npz" if L in SHORT else "heldout_kl.npz")
    if a.local:
        A = arms(L, L_ROOT)
        runs = [(seed, arm) for seed in a.seeds.split(",") for arm in a.arms.split(",")]
        env = dict(os.environ, MICRO=str(micro))  # run_proxy.sbatch reads MICRO (sbatch exports the env)
        for i in range(0, len(runs), a.group):
            grp = runs[i:i + a.group]
            for seed in sorted({s for s, _ in grp}):
                names = [arm for s, arm in grp if s == seed]
                cmd = [sys.executable, f"{ELP}/launch_proxy.py", "--tag", tag, "--seeds", seed, "--arms", ",".join(names),
                       "--steps", a.steps, "--sets", f"kl{L}:{a.ho_rows}", "--heldout", f"{L_ROOT}/{ho}",
                       "--data", f"{L_ROOT}/shards/kl{L}", "--seq-len", str(SEQ[L]), "--gpus", str(a.gpus),
                       "--code", a.local, "--node", a.node, "--qos", a.qos, "--minutes", str(a.minutes * len(names))]
                for arm in names:
                    v, lr, flags = A[arm]
                    cmd += ["--custom", f"{arm}={v},{lr:g},{flags}"]
                print(f"[launch_kl local] {L} {names} sd{seed} MICRO={micro}", flush=True)
                if a.dry_run:
                    print(" ".join(cmd))
                else:
                    subprocess.run(cmd, check=True, env=env)
        return
    for seed in a.seeds.split(","):
        for arm in a.arms.split(","):
            v, lr, flags = A[arm]
            cmd = [sys.executable, f"{ELP}/launch_proxy_beaker.py", "--tag", tag, "--seeds", seed, "--arms", arm,
                   "--custom", f"{arm}={v},{lr:g},{flags}", "--steps", a.steps, "--sets", f"kl{L}:{a.ho_rows}",
                   "--heldout", f"{W}/{ho}", "--data", f"{W}/shards/kl{L}", "--seq-len", str(SEQ[L]),
                   "--micro", str(micro), "--gpus", str(a.gpus), "--minutes", str(a.minutes)]
            if a.dry_run:
                cmd.append("--dry-run")
            print(f"[launch_kl] {L} {arm} sd{seed}", flush=True)
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
