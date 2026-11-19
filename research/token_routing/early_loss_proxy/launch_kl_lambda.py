"""Launch keepfrac_len-style arms (kl<L> single-length pools) on LAMBDA, experiment tokdrop_lr.

Lambda twin of ``keepfrac_len/launch_kl.py``: same arms (``keepfrac_len.launch_kl.arms``), same pools / held-out sets /
seq-len per length, run by ``run_proxy_lambda.sbatch`` (A100-80GB, conda-pack env, NFS inputs under LROOT/elp, run dirs
on node-local /tmp). Stages the caller's committed HEAD once (or ``--code``), submits one slurm job per ``--group``
arm runs, registers each with ops. Results: lambda:/accounts/projects/sewonm/prasann/ctc_suite/elp/results/<tag>/,
fetch with ``fetch_lambda_results.sh <tag>``.

    python3 research/token_routing/early_loss_proxy/launch_kl_lambda.py --len 8k \\
        --arms k20,k40,myarm --custom 'myarm=softtoken,1e-5,--st-... --attn-backend flash_2' \\
        --steps 10,25,50,100,150,225 --gpus 2 --minutes 60 --tag kl8k-lam [--qos preemptive_high] [--dry-run]

Lambda policy (lambda_cluster.md): at most 8 GPUs preemptive_high + 8 preemptive in total, the rest normal.
Pools staged on Lambda: kl8k, kl16k (add others with rsync from a Berkeley node's /data/prasann/early_loss_proxy).
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
sys.path.insert(0, os.path.join(HERE, "keepfrac_len"))
from launch_kl import LENS, SEQ, SHORT, arms  # noqa: E402

OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
ROOT = "/accounts/projects/sewonm/prasann/ctc_suite/elp"


def ssh(cmd):
    return subprocess.run(["ssh", "lambda", cmd], capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--len", required=True, choices=list(LENS))
    ap.add_argument("--arms", default="", help="arm names from launch_kl.arms (k20, k40, dft, ref, grJ, ...)")
    ap.add_argument("--custom", action="append", default=[], metavar="NAME=VARIANT,LR,FLAGS",
                    help="extra arm; it runs if NAME is listed in --arms (or always, if --arms is empty)")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", required=True, help="probe steps after step 0 (step 0 is always probed; the run hard-stops at the last one)")
    ap.add_argument("--tag", default="", help="results dir name (default kl<L>-lam)")
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--micro", type=int, default=0, help="micro-batch rows per GPU (default 2, 1 at 64k)")
    ap.add_argument("--ho-rows", type=int, default=48)
    ap.add_argument("--heldout", default="", help="held-out npz basename (default heldout_kls.npz at 2k/4k/8k)")
    ap.add_argument("--minutes", type=int, required=True, help="expected minutes per arm run (slurm --time, ops)")
    ap.add_argument("--group", type=int, default=1, help="arm runs back to back per slurm job")
    ap.add_argument("--qos", default="preemptive_high", choices=["preemptive_high", "preemptive", "normal"])
    ap.add_argument("--node", default="", help="optional lambda-hyperplaneNN")
    ap.add_argument("--code", default="", help="already-staged Lambda code dir (default: stage HEAD now)")
    ap.add_argument("--max-train-s", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    L = a.len
    tag = a.tag or f"kl{L}-lam"
    A = arms(L, ROOT)
    for c in a.custom:
        name, _, rest = c.partition("=")
        variant, lr, flags = rest.split(",", 2)
        A[name] = (variant, float(lr), flags.strip())
    names = [x for x in a.arms.split(",") if x] or [c.partition("=")[0] for c in a.custom]
    missing = [n for n in names if n not in A]
    if missing:
        raise SystemExit(f"unknown arms {missing}; known: {sorted(A)}")
    micro = a.micro or (1 if L == "64k" else 2)
    ho = a.heldout or ("heldout_kls.npz" if L in SHORT else "heldout_kl.npz")
    lines = []
    for seed in a.seeds.split(","):
        for n in names:
            v, lr, flags = A[n]
            lines.append("\t".join([f"elp-{tag}-{n}-sd{seed}", v, f"{lr:g}", seed, flags]))
    if subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                      text=True).stdout.strip():
        print("WARNING: uncommitted changes are NOT staged (git archive HEAD)", file=sys.stderr)
    if a.dry_run:
        code = a.code or "<staged HEAD>"
    else:
        code = a.code or subprocess.run(["bash", f"{HERE}/stage_code_lambda.sh"], capture_output=True, text=True,
                                        check=True).stdout.strip().splitlines()[-1]
    for i in range(0, len(lines), a.group):
        grp = lines[i:i + a.group]
        minutes = a.minutes * len(grp)
        spec_path = f"{ROOT}/jobs/{tag}-{os.getpid()}-{i}.tsv"
        env = dict(CODE=code, STEPS=a.steps, SETS=f"kl{L}:{a.ho_rows}", HELDOUT=f"{ROOT}/{ho}", TAG=tag,
                   DATA=f"{ROOT}/shards/kl{L}", SPEC=spec_path, SEQLEN=SEQ[L], NGPU=a.gpus, MICRO=micro,
                   MAXTRAIN=a.max_train_s or "")
        envs = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items() if v != "")
        sb = (f"{envs} sbatch --parsable --partition=lambda --account=site --qos={a.qos} --gres=gpu:A100:{a.gpus} "
              f"--nodes=1 --time={minutes + 20} --job-name=elp-{tag} --output={ROOT}/logs/elp-{tag}-%j.out "
              f"{'-w ' + a.node if a.node else ''} --export=ALL "
              f"{code}/research/token_routing/early_loss_proxy/run_proxy_lambda.sbatch")
        print(f"[launch_kl_lambda] {L} {[g.split(chr(9))[0] for g in grp]} MICRO={micro}", flush=True)
        if a.dry_run:
            print("\n".join(grp).replace("\t", " | "))
            print(sb)
            continue
        spec = "\n".join(grp) + "\n"
        r = ssh(f"mkdir -p {ROOT}/jobs {ROOT}/logs && echo {base64.b64encode(spec.encode()).decode()} | base64 -d > {spec_path}")
        if r.returncode:
            raise SystemExit(r.stderr)
        r = ssh(sb)
        if r.returncode:
            raise SystemExit(f"sbatch failed: {r.stdout} {r.stderr[-400:]}")
        jid = r.stdout.strip().split(";")[0]
        subprocess.run([OPS, "register", "--run-id", f"elp-{tag}-lambda{jid}", "--experiment", "tokdrop_lr",
                        "--cluster", "lambda", "--job-id", jid, "--expected-minutes", str(minutes), "--git", REPO])
        print(f"job {jid} -> lambda:{ROOT}/results/{tag}/  log: {ROOT}/logs/elp-{tag}-{jid}.out")
    print(f"fetch: bash {HERE}/fetch_lambda_results.sh {tag}")


if __name__ == "__main__":
    main()
