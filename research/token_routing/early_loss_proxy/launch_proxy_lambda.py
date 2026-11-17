"""Launch early-loss-proxy arms on LAMBDA (2 A100-80GB, ONE slurm job, arms back to back), experiment tokdrop_lr.

Same arguments as launch_proxy.py (arms/custom/seeds/steps/sets/stop-ce/max-train-s/lr-mult/extra/tag/minutes).
Stages the caller's committed HEAD, submits via ssh, registers with ops. Results stay on Lambda under
/accounts/projects/sewonm/prasann/ctc_suite/elp/results/<tag>/; fetch with fetch_lambda_results.sh <tag>.

    python3 research/token_routing/early_loss_proxy/launch_proxy_lambda.py --arms recipe --custom 'x=softtoken,1e-5,FLAGS' \\
        --tag mytag --steps 25,50,100 --max-train-s 600
"""
from __future__ import annotations

import argparse
import base64
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
from launch_proxy import ARMS, OPS  # noqa: E402

L = "/accounts/projects/sewonm/prasann/ctc_suite"
ROOT = f"{L}/elp"


def ssh(cmd, **kw):
    return subprocess.run(["ssh", "lambda", cmd], capture_output=True, text=True, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="dense,recipe")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", default="1,2,5,10")
    ap.add_argument("--sets", default="suite4k:16,suite8k:16,suite16k:16")
    ap.add_argument("--heldout", default=f"{ROOT}/heldout_v2.npz", help="path ON LAMBDA")
    ap.add_argument("--data", default=f"{ROOT}/shards/mtb16k", help="pool path ON LAMBDA")
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--stop-ce", type=float, default=0.0)
    ap.add_argument("--max-train-s", type=float, default=0.0)
    ap.add_argument("--custom", action="append", default=[], metavar="NAME=VARIANT,LR,FLAGS")
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--lr-mult", type=float, default=1.0)
    ap.add_argument("--extra", default="")
    ap.add_argument("--qos", default="preemptive_high",
                    help="Lambda policy: <=8 GPUs preemptive_high, <=8 preemptive, the rest normal")
    ap.add_argument("--minutes", type=int, default=0)
    ap.add_argument("--node", default="", help="optional lambda-hyperplaneNN")
    ap.add_argument("--code", default="", help="already-staged Lambda code dir (default: stage HEAD now)")
    a = ap.parse_args()
    for c in a.custom:
        name, _, rest = c.partition("=")
        variant, lr, flags = rest.split(",", 2)
        ARMS[name] = (variant, float(lr), flags.strip(), None, "custom")
    lines = []
    for seed in a.seeds.split(","):
        for arm in a.arms.split(","):
            variant, lr, extra, _, _ = ARMS[arm]
            lines.append("\t".join([f"elp-{a.tag}-{arm}-sd{seed}", variant, f"{lr * a.lr_mult:g}", seed,
                                    f"{extra} {a.extra}".strip()]))
    if subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                      text=True).stdout.strip():
        print("WARNING: uncommitted changes are NOT staged (git archive HEAD)", file=sys.stderr)
    code = a.code or subprocess.run(["bash", f"{HERE}/stage_code_lambda.sh"], capture_output=True, text=True,
                                    check=True).stdout.strip().splitlines()[-1]
    minutes = a.minutes or max(15, 6 * len(lines))
    spec_path = f"{ROOT}/jobs/{a.tag}-{os.getpid()}.tsv"
    spec = "\n".join(lines) + "\n"
    r = ssh(f"mkdir -p {ROOT}/jobs {ROOT}/logs && echo {base64.b64encode(spec.encode()).decode()} | base64 -d > {spec_path}")
    if r.returncode:
        raise SystemExit(r.stderr)
    env = dict(CODE=code, STEPS=a.steps, SETS=a.sets, HELDOUT=a.heldout, TAG=a.tag, DATA=a.data, SPEC=spec_path,
               SEQLEN=a.seq_len, NGPU=a.gpus, STOPCE=a.stop_ce or "", MAXTRAIN=a.max_train_s or "")
    envs = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items() if v != "")  # values contain commas
    sb = (f"{envs} sbatch --parsable --partition=lambda --account=site --qos={a.qos} --gres=gpu:A100:{a.gpus} --nodes=1 "
          f"--time={minutes + 20} --job-name=elp-{a.tag} --output={ROOT}/logs/elp-{a.tag}-%j.out "
          f"{'-w ' + a.node if a.node else ''} --export=ALL "
          f"{code}/research/token_routing/early_loss_proxy/run_proxy_lambda.sbatch")
    r = ssh(sb)
    print(r.stdout.strip(), r.stderr.strip()[-400:])
    if r.returncode:
        raise SystemExit(1)
    jid = r.stdout.strip().split(";")[0]
    subprocess.run([OPS, "register", "--run-id", f"elp-{a.tag}-lambda{jid}", "--experiment", "tokdrop_lr",
                    "--cluster", "lambda", "--job-id", jid, "--expected-minutes", str(minutes), "--git", REPO])
    print(f"job {jid}: {len(lines)} arm runs -> lambda:{ROOT}/results/{a.tag}/  log: {ROOT}/logs/elp-{a.tag}-{jid}.out")
    print(f"fetch: bash {HERE}/fetch_lambda_results.sh {a.tag}")


if __name__ == "__main__":
    main()
