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
LENS = {"16k": 16384, "32k": 32768, "64k": 65536}
LORA = "--lora-rank 64 --lora-blocks 0 --lora-scale 32 --attn-backend flash_2"


def router(k1, k2):
    """The recipe's router (launch_proxy.RECIPE) with cut 1 (after block 7) keeping k1 and cut 2 (after block 16)
    keeping k2 of the row's tokens."""
    return (f"--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac {k1:g} "
            "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
            f"--st-otr-stages 7:{k1:g},16:{k2:g} --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
            "--st-otr-early-lora-blocks 32 --attn-backend flash_2")


GRJ = "--st-gdd-keep-range 0.05,0.5 --st-gdd-whole-frac 0.15"  # loop/goldrand arm grJ


def arms(L):
    pool = f"{W}/shards/kl{L}"
    a = {
        "dft": ("full", 5e-6, "--attn-backend flash_2"),   # dense full fine-tune: the primary reference (Prasann 11-17)
        "ref": ("full", 1e-5, LORA),                       # dense LoRA reference (secondary)
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
    ap.add_argument("--minutes", type=int, required=True, help="expected minutes per job (sizes minRuntime)")
    ap.add_argument("--ho-rows", type=int, default=48)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    L = a.len
    tag = a.tag or f"kl{L}"
    A = arms(L)
    micro = a.micro or (1 if L == "64k" else 2)
    for seed in a.seeds.split(","):
        for arm in a.arms.split(","):
            v, lr, flags = A[arm]
            cmd = [sys.executable, f"{ELP}/launch_proxy_beaker.py", "--tag", tag, "--seeds", seed, "--arms", arm,
                   "--custom", f"{arm}={v},{lr:g},{flags}", "--steps", a.steps, "--sets", f"kl{L}:{a.ho_rows}",
                   "--heldout", f"{W}/heldout_kl.npz", "--data", f"{W}/shards/kl{L}", "--seq-len", str(LENS[L]),
                   "--micro", str(micro), "--gpus", str(a.gpus), "--minutes", str(a.minutes)]
            if a.dry_run:
                cmd.append("--dry-run")
            print(f"[launch_kl] {L} {arm} sd{seed}", flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
