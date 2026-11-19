"""Launch random-doc-drop arms at 16k on Beaker (one 2 x H100 job per arm via ../launch_proxy_beaker.py), the
keepfrac_len setup on the no-gold pool: one epoch of ng16k (225 steps of 16 rows), probes on heldout_ng.npz set ng16k
(48 rows per task, eval_size 144).

Arms: dft (dense full FT) and k20 (the recipe) train on ng16k; rddA / rddB = the recipe on the variant pool
ng16k_rdd{A,B}_s<seed> (A: keep U(0.05, 0.5), B: keep U(0.25, 0.75); 15% whole rows), one fresh draw per seed.

    python3 random_doc_drop/launch_ng.py --arms dft,k20,rddA,rddB --seeds 0 [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ELP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ELP, "keepfrac_len"))
from launch_kl import router  # noqa: E402

W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/early_loss_proxy"
STEPS = "10,25,50,75,100,125,150,175,200,225"


def arms(seed):
    rec = router(0.20, 0.07)
    return {
        "dft": ("full", 5e-6, "--attn-backend flash_2", "ng16k"),
        "k20": ("softtoken", 1e-5, rec, "ng16k"),
        "rddA": ("softtoken", 1e-5, rec, f"ng16k_rddA_s{seed}"),
        "rddB": ("softtoken", 1e-5, rec, f"ng16k_rddB_s{seed}"),
        # data controls: dense full FT on the drop pool
        "dftA": ("full", 5e-6, "--attn-backend flash_2", f"ng16k_rddA_s{seed}"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--minutes", type=int, default=50)
    ap.add_argument("--tag", default="ng16k")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for seed in a.seeds.split(","):
        A = arms(seed)
        for arm in a.arms.split(","):
            v, lr, flags, pool = A[arm]
            cmd = [sys.executable, f"{ELP}/launch_proxy_beaker.py", "--tag", a.tag, "--seeds", seed, "--arms", arm,
                   "--custom", f"{arm}={v},{lr:g},{flags}", "--steps", STEPS, "--sets", "ng16k:48",
                   "--heldout", f"{W}/heldout_ng.npz", "--data", f"{W}/shards/{pool}", "--seq-len", "16384",
                   "--micro", "2", "--gpus", "2", "--minutes", str(a.minutes)]
            if a.dry_run:
                cmd.append("--dry-run")
            print(f"[launch_ng] {arm} sd{seed} pool {pool}", flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
