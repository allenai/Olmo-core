"""Launch early dense-loss proxy runs (records/early-dense-loss-proxy.md) on a jsteinhardt node, experiment tokdrop_lr.

Each arm = its full-run training flags (the ones its 1,952-rows/task result was trained with: base, LR, 16 rows/step,
micro 2, 2 GPUs, seq 12288, the fresh 4-task pool mt4f1952, the method's own forward), run for max(--steps) steps
inside ``dense_probe.py``. All arms of one call run back to back in ONE slurm job (one torchrun per arm).

    bash research/token_routing/wts/stage_code.sh mooney         # -> /data/prasann/wts/code_<sha>
    python research/token_routing/early_loss_proxy/launch_proxy.py --code /data/prasann/wts/code_<sha> \\
        --arms dense,recipe --seeds 0,1 --steps 1,2,5,10
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
STATE = f"{HERE}/state.json"
FULL = "/data/prasann/oracle/shards/mt4f1952"
DD = "/data/prasann/docdrop/shards/mt4f1952_dd{g}/keep_sidecar.json"

LORA = "--lora-rank 64 --lora-blocks 0 --lora-scale 32 --attn-backend flash_2"
#: the recipe lora64all32r20to7 (launch_wts.RECIPE / launch_fast8k), flag for flag
RECIPE = ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
          "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
          "--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
          "--st-otr-early-lora-blocks 32 --attn-backend flash_2")
#: one cut after block 7 (oracle branch RECIPE1)
RECIPE1 = RECIPE.replace("--st-otr-stages 7:0.2,16:0.07 ", "")
NODENSE = "--st-otr-warmup-random"


def rnd(pct):  # oracle branch rnd1doc<pct>: uniform per-document keep, one cut at block 7
    return RECIPE1.replace("--st-otr-keep-frac 0.2 ", f"--st-otr-keep-frac {pct / 100:g} ") + " --st-otr-score random"


def orc(t):  # oracle branch orc1doc<t> (curve logging and npz dumps off: selection unchanged)
    return f"{RECIPE1} --st-otr-score oracle --oracle-scope doc --oracle-target {t:.2f} --oracle-npz-every 0"


#: arm -> (variant, lr, trainer flags, known 8k 4-task mean at 1,952 rows/task, source run)
ARMS = {
    "dense": ("full", 5e-6, "", 0.843, "f8k-mt4-dense-f1952-tde-lr5e-06"),
    "denselora": ("full", 1e-5, LORA, 0.853, "f8k-mt4-denselora64all-f1952-tde-lr1e-05"),
    "recipe": ("softtoken", 1e-5, RECIPE, 0.8465, "f8k-mt4-lora64all32r20to7{,sd1}-f1952-tde-lr1e-05"),
    "rnd19p7": ("softtoken", 1e-5, rnd(19.7), 0.841, "f8k-mt4-rnd1doc19p7-f1952-orc-lr1e-05"),
    "rnd17p7": ("softtoken", 1e-5, rnd(17.7), 0.813, "f8k-mt4-rnd1doc17p7-f1952-orc-lr1e-05"),
    "rnd11p7": ("softtoken", 1e-5, rnd(11.7), 0.796, "f8k-mt4-rnd1doc11p7-f1952-orc-lr1e-05"),
    "rnd9p7": ("softtoken", 1e-5, rnd(9.7), 0.769, "f8k-mt4-rnd1doc9p7-f1952-orc-lr1e-05"),
    "wtsS1": ("softtoken", 1e-5, f"{RECIPE} {NODENSE} --st-otr-wts sample=1,weights=none", 0.790,
              "f8k-mt4-wtsS1-f1952-wts-lr1e-05"),
    "wtsS0wr": ("softtoken", 1e-5, f"{RECIPE} {NODENSE}", 0.837, "f8k-mt4-wtsS0wr-f1952-wts-lr1e-05"),
    "ddp": ("softtoken", 1e-5, f"{LORA} --st-gold-sidecar {DD.format(g='')} --st-n-random 0 "
            "--st-keep-mode gold_plus_random --st-drop-slots", 0.555, "f8k-mt4-ddp-f1952-dd-lr1e-05-loc"),
    "ddpg": ("softtoken", 1e-5, f"{LORA} --st-gold-sidecar {DD.format(g='g')} --st-n-random 0 "
             "--st-keep-mode gold_plus_random --st-drop-slots", 0.665, "f8k-mt4-ddpg-f1952-dd-lr1e-05-loc"),
    "tail20to7": ("softtoken", 1e-5, f"{RECIPE} --st-otr-score tail", 0.781, "lora64all32t20to7 (posid-reset branch)"),
    "tail10": ("softtoken", 1e-5, RECIPE.replace("--st-otr-stages 7:0.2,16:0.07", "--st-otr-stages 7:0.1")
               + " --st-otr-score tail", 0.747, "lora64all32t10 (posid-reset branch)"),
    "orc95": ("softtoken", 1e-5, orc(0.95), 0.850, "f8k-mt4-orc1doc-f1952-orc-lr1e-05"),
    "orc90": ("softtoken", 1e-5, orc(0.90), 0.824, "f8k-mt4-orc1doc90-f1952-orc-lr1e-05"),
    "orc85": ("softtoken", 1e-5, orc(0.85), 0.837, "f8k-mt4-orc1doc85-f1952-orc-lr1e-05"),
    "orc80": ("softtoken", 1e-5, orc(0.80), 0.721, "f8k-mt4-orc1doc80-f1952-orc-lr1e-05"),
    "orc70": ("softtoken", 1e-5, orc(0.70), 0.648, "f8k-mt4-orc1doc70-f1952-orc-lr1e-05"),
    "orc60": ("softtoken", 1e-5, orc(0.60), 0.643, "f8k-mt4-orc1doc60-f1952-orc-lr1e-05"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="dense,recipe")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", default="1,2,5,10")
    ap.add_argument("--sets", default="suite8k:16,suite2k:16,train:16")
    ap.add_argument("--heldout", default="/data/prasann/early_loss_proxy/heldout_v1.npz")
    ap.add_argument("--tag", default="v1", help="result folder results/<tag>/")
    ap.add_argument("--lr-mult", type=float, default=1.0, help="constant-LR / LR variants: multiply each arm's LR")
    ap.add_argument("--extra", default="", help="appended to every arm's trainer flags (e.g. --warmup-fraction 0)")
    ap.add_argument("--code", required=True, help="staged code dir on the node (stage_code.sh)")
    ap.add_argument("--node", default="mooney")
    ap.add_argument("--qos", default="preemptive_high")
    ap.add_argument("--minutes", type=int, default=0)
    a = ap.parse_args()
    lines = []
    for seed in a.seeds.split(","):
        for arm in a.arms.split(","):
            variant, lr, extra, _, _ = ARMS[arm]
            name = f"elp-{a.tag}-{arm}-sd{seed}"
            lines.append("\t".join([name, variant, f"{lr * a.lr_mult:g}", seed, f"{extra} {a.extra}".strip()]))
    job_dir = f"/data/prasann/early_loss_proxy/jobs"
    spec = "\n".join(lines) + "\n"
    minutes = a.minutes or max(10, 4 * len(lines))
    env = dict(os.environ, CODE=a.code, STEPS=a.steps, SETS=a.sets, HELDOUT=a.heldout, TAG=a.tag,
               SPEC_B64=__import__("base64").b64encode(spec.encode()).decode())
    cmd = ["sbatch", "-w", a.node, f"--qos={a.qos}", "-J", f"elp-{a.tag}", f"--time={minutes + 20}",
           f"{HERE}/run_proxy.sbatch"]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(out.stdout.strip(), out.stderr.strip()[-400:])
    if out.returncode:
        raise SystemExit(1)
    jid = out.stdout.strip().split()[-1]
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"jobs": {}}
    st["jobs"][jid] = {"node": a.node, "tag": a.tag, "runs": [ln.split("\t")[0] for ln in lines], "steps": a.steps,
                       "sets": a.sets, "code": a.code, "extra": a.extra, "lr_mult": a.lr_mult}
    json.dump(st, open(STATE, "w"), indent=1)
    subprocess.run([OPS, "register", "--run-id", f"elp-{a.tag}-job{jid}", "--experiment", "tokdrop_lr",
                    "--cluster", "slurm", "--job-id", jid, "--expected-minutes", str(minutes), "--git", REPO])
    print(f"job {jid}: {len(lines)} arm runs -> {a.node}:/data/prasann/early_loss_proxy/results/{a.tag}/")


if __name__ == "__main__":
    main()
