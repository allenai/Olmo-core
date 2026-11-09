"""Launch grad_importance.py jobs on Beaker (1 H100 each) against weka checkpoints, without publishing the code.

    # 1. tarball of the working tree -> private S3 (login node)
    git ls-files | tar -czf code.tgz -T - ; aws s3 cp code.tgz s3://ai2-llm/checkpoints/prasanns/_transfer/gradprobe/code_<ts>.tgz
    # 2. jobs (run from ANY clean checkout whose HEAD is pushed, e.g. a worktree at origin/prasann/landmark)
    python launch_beaker.py --launch-dir <clean pushed checkout> --code-s3 s3://.../code_<ts>.tgz \\
        --out-s3 s3://ai2-llm/checkpoints/prasanns/_transfer/gradprobe/out --models setA32k,setAbase \\
        --jobs nq:8k:16:full,nq:16k:5:len,... [--dry-run]

Each job's inline bootstrap writes the AWS creds from the beaker secrets, installs awscli, pulls and unpacks the
tarball and runs beaker_gradprobe.sh from it; outputs land at <out-s3>/<tag>.{json,log}. Every launched job is
registered with ``ops register`` (experiment learned_router). Priority urgent, workspace ai2/flex2, budget
ai2/oe-other (unallocated), clusters ai2/jupiter,ai2/ceres,ai2/titan (H100/B200; --clusters).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

GANTRY = "/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/gantry"
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
IMAGE = "tylerr/olmo-core-tch291cu128-2025-11-25"
W = "/weka/oe-training-default/ai2-llm/checkpoints"
MODELS = {
    "setA32k": f"{W}/prasanns/q35-4b-dense-ctc-setA-32k-20260923T1537/step888/model_and_optim",
    "setAbase": f"{W}/q35-4b-dense-256k-fix/step2385/model_and_optim",
    # Qwen3-4B (full attention) setA SFT, lengen arm (c): same 203,156 examples as setA32k
    "q3setA": f"{W}/prasanns/q3-4b-dense-ctc-setA-32k-20261013T2244/step886/model_and_optim",
}
# extra weka dirs synced verbatim next to a model's resave (<dst>/<model>_tokenizer/): the training-time tokenizer
EXTRA_SYNC = {"q3setA": f"{W}/prasanns/hf_tokenizers/Qwen3-4B-Base-nothink"}
PROFILES = {  # grad_importance.py flags; "full" = the 8k few-step comparison, "len" = the 16k/32k length sweep
    "full": "--ig-steps 16,4,3,2,1 --iter-k 2 --alphas 0.2,0.3,0.4,0.5,0.6,0.7,0.8 --iter-alpha 2@0.5 --iter-fine 0 --doc-level 0",
    "len": "--ig-steps 16,4,1 --iter-k 2 --alphas 0.3,0.5,0.7 --iter-fine 0 --iter-coarse 0 --doc-level 0 --rand-seeds 2",
}
EXPECTED_MIN = {"8k": 45, "16k": 40, "32k": 75}
BOOT = ('mkdir -p ~/.aws && printf "%s\\n" "$AWS_CREDS" > ~/.aws/credentials && printf "%s\\n" "$AWS_CFG" > ~/.aws/config '
        '&& export AWS_PROFILE=S3 && pip install -q awscli dataclass-extensions bettermap >/dev/null 2>&1 '
        '&& pip install -q --no-deps flash-linear-attention==0.4.1 >/dev/null 2>&1 && pip install -q einops >/dev/null 2>&1 && mkdir -p /tmp/gp_code && cd /tmp/gp_code '
        '&& aws s3 cp --quiet "$CODE_S3" code.tgz && tar -xzf code.tgz '
        '&& exec bash research/token_routing/learned_router/gradprobe/beaker_gradprobe.sh')


RESAVE = ('mkdir -p ~/.aws && printf "%s\\n" "$AWS_CREDS" > ~/.aws/credentials && printf "%s\\n" "$AWS_CFG" > ~/.aws/config '
          '&& export AWS_PROFILE=S3 && pip install -q awscli dataclass-extensions bettermap >/dev/null 2>&1 && mkdir -p /tmp/gp_code && cd /tmp/gp_code '
          '&& aws s3 cp --quiet "$CODE_S3" code.tgz && tar -xzf code.tgz && export PYTHONPATH=$PWD/src '
          '&& python -u research/token_routing/learned_router/gradprobe/resave_model_only.py "$SRC" "/tmp/rs/$NAME" '
          '&& aws s3 sync --quiet "/tmp/rs/$NAME" "$DST_S3/$NAME" '
          '&& { [ -z "${EXTRA_SRC:-}" ] || aws s3 sync --quiet "$EXTRA_SRC" "$DST_S3/${NAME}_tokenizer"; } && echo SYNC_DONE')


def launch_resave(a):
    """--resave: one CPU job per model: model-only re-save of the weka checkpoint -> <dst>/<model>/ on S3 (for running
    the probe on the Berkeley cluster when the Beaker GPU queue is stuck). Multi-cluster, eager-first."""
    for m in a.models.split(","):
        tag = f"gradprobe-resave-{m}"
        cmd = [GANTRY, "run", "--name", tag, "-w", "ai2/flex2", "-b", "ai2/oe-other",
               *[x for c in ("ai2/neptune", "ai2/ceres", "ai2/saturn", "ai2/jupiter") for x in ("--cluster", c)],
               "--gpus", "0", "--cpus", "8", "--memory", "96GiB", "--priority", "urgent", "--min-runtime", "1h",
               "--beaker-image", IMAGE,
               "--weka", "oe-training-default:/weka/oe-training-default",
               "--env-secret", "AWS_CREDS=PRASANNS_AWS_CREDENTIALS", "--env-secret", "AWS_CFG=PRASANNS_AWS_CONFIG",
               "--env", f"CODE_S3={a.code_s3}", "--env", f"SRC={MODELS[m]}", "--env", f"NAME={m}",
               "--env", f"DST_S3={a.resave}", "--env", f"EXTRA_SRC={EXTRA_SYNC.get(m, '')}",
               "--no-python", "--timeout", "0", "--yes", "--", "bash", "-c", RESAVE]
        r = subprocess.run(cmd, cwd=a.launch_dir, capture_output=True, text=True)
        mm = re.search(r"beaker\.org/ex/([0-9A-Z]{26})", r.stdout + r.stderr)
        if r.returncode != 0 or not mm:
            print(f"!!! {tag}: gantry failed (rc {r.returncode})\n{(r.stdout + r.stderr)[-1500:]}", file=sys.stderr)
            continue
        print(f"{tag}  https://beaker.org/ex/{mm.group(1)}", flush=True)
        subprocess.run([OPS, "register", "--run-id", tag, "--experiment", "learned_router", "--cluster", "beaker",
                        "--kind", "data", "--job-id", mm.group(1), "--expected-minutes", "25", "--git", a.repo], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch-dir", required=True, help="clean checkout whose HEAD is pushed (gantry clones it; unused)")
    ap.add_argument("--code-s3", required=True)
    ap.add_argument("--out-s3", required=True)
    ap.add_argument("--models", default="setA32k,setAbase")
    ap.add_argument("--jobs", default="", help="comma list task:rung:rows:profile")
    ap.add_argument("--resave", default="", help="S3 prefix: launch model-only re-save CPU jobs to <prefix>/<model>/ instead")
    ap.add_argument("--repo", default="/accounts/projects/berkeleynlp/prasann/projects/OLMo-core", help="for ops --git")
    ap.add_argument("--clusters", default="ai2/jupiter,ai2/ceres,ai2/titan", help="comma list (H100 + B200 clusters)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.resave:
        return launch_resave(a)
    for m in a.models.split(","):
        for spec in a.jobs.split(","):
            task, rung, rows, prof = spec.split(":")
            tag = f"gradprobe-{m}-{task}-{rung}-r{rows}"
            cmd = [GANTRY, "run", "--name", tag, "-w", "ai2/flex2", "-b", "ai2/oe-other",
                   *[x for c in a.clusters.split(",") for x in ("--cluster", c)],
                   "--gpus", "1", "--priority", "urgent", "--beaker-image", IMAGE,
                   "--weka", "oe-training-default:/weka/oe-training-default",
                   "--env-secret", "AWS_CREDS=PRASANNS_AWS_CREDENTIALS", "--env-secret", "AWS_CFG=PRASANNS_AWS_CONFIG",
                   "--env", f"CODE_S3={a.code_s3}", "--env", f"CKPT={MODELS[m]}", "--env", f"TASK={task}",
                   "--env", f"RUNG={rung}", "--env", f"ROWS={rows}", "--env", f"TAG={tag}", "--env", f"OUT_S3={a.out_s3}",
                   "--env", f"EXTRA={PROFILES[prof]}", "--no-python", "--timeout", "0", "--yes", "--", "bash", "-c", BOOT]
            if a.dry_run:
                print(" ".join(cmd[:6]), "...", tag, PROFILES[prof])
                continue
            r = subprocess.run(cmd, cwd=a.launch_dir, capture_output=True, text=True)
            out = r.stdout + r.stderr
            mm = re.search(r"beaker\.org/ex/([0-9A-Z]{26})", out)
            if r.returncode != 0 or not mm:
                print(f"!!! {tag}: gantry failed (rc {r.returncode})\n{out[-2000:]}", file=sys.stderr)
                continue
            ex = mm.group(1)
            print(f"{tag}  https://beaker.org/ex/{ex}", flush=True)
            subprocess.run([OPS, "register", "--run-id", tag, "--experiment", "learned_router", "--cluster", "beaker",
                            "--kind", "eval", "--job-id", ex, "--expected-minutes", str(EXPECTED_MIN[rung]),
                            "--log-path", f"{a.out_s3}/{tag}.log", "--git", a.repo], check=False)


if __name__ == "__main__":
    main()
