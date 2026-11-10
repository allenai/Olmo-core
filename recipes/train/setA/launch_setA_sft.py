"""
Every Beaker step of the setA dense SFT pair (pre-flight + training), in one file.

    # 1. CPU pre-flight: shards, drops per window, base ckpt + marker audit, packed windows/steps
    python recipes/train/setA/launch_setA_sft.py prep
    # 2. train (1 node each; 256k-2node = amandab's exact 2-node geometry, fallback)
    python recipes/train/setA/launch_setA_sft.py 32k
    python recipes/train/setA/launch_setA_sft.py 256k

    # single-task debugging / another data set or base (each flag is optional)
    python recipes/train/setA/launch_setA_sft.py 32k --tasks nq \
        --data-root /weka/.../ctc_sft_sets/<set> --base-checkpoint /weka/.../model_and_optim

    # a research arm on the canonical recipe (recipes/train/variants.py; 32k arm for most)
    python recipes/train/setA/launch_setA_sft.py 32k --variant ffnmoe \
        --variant-opts "ffn_moe_target=0.3,total_steps=<prep's packed steps>"

``--tasks`` / ``--data-root`` / ``--base-checkpoint`` are exported to the Beaker job as the env vars
``SETA_TASKS`` / ``SETA_DATA_ROOT`` / ``SETA_BASE_CKPT``, which ``_qwen35_setA_common.py`` reads (the
training scripts are driven by the olmo-core CLI, so they cannot take extra flags). A non-default
task set is added to the run name (``q35-4b-dense-ctc-setA-nq-32k-<stamp>``). ``--variant`` /
``--variant-opts`` are exported as ``SETA_VARIANT`` / ``SETA_VARIANT_OPTS`` (validated here first) and
replace ``dense`` in the run name (``q35-4b-ffnmoe-ctc-setA-32k-<stamp>``).

Every job is pinned to a full pushed commit SHA (default: this checkout's HEAD, which must be on the
remote). The data are the setA shards as built by ``recipes/data/ctc_sft/build_ctc_sft.py`` (see
_qwen35_setA_common.py).
All jobs: workspace ai2/flex2, budget ai2/oe-other (allocated via oe-scaling),
priority urgent, jupiter H100s. ``--dry-run`` prints the gantry command without submitting.
"""

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REL = "recipes/train/setA"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))  # recipes/train (variants.py, stdlib-only at import)

from _setA_defaults import (  # noqa: E402  (stdlib-only; no torch on the login node)
    DEFAULT_BASE_CHECKPOINT,
    DEFAULT_SET_ROOT,
    DEFAULT_TASKS,
    DEFAULT_VARIANT,
    ENV_BASE_CKPT,
    ENV_DATA_ROOT,
    ENV_TASKS,
    ENV_VARIANT,
    ENV_VARIANT_OPTS,
    parse_tasks,
    task_tag,
)
from variants import parse_variant_spec, setA_variant_names  # noqa: E402

GANTRY = os.environ.get(
    "GANTRY", "/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/gantry"
)

IMAGE = "tylerr/olmo-core-tch291cu128-2025-11-25"  # == OLMoCoreBeakerImage.stable
TRAIN_INSTALL = (
    "python -m pip install -e '.[beaker,eval,fla,wandb,transformers]' "
    "'torch==2.9.1+cu128' 'triton==3.5.1'"
)  # amandab's validated training install
NODES = {"32k": 1, "256k": 1, "256k-2node": 2}
CP_ARMS = {"256k", "256k-2node"}  # == ARMS entries with cp_degree > 1 (_qwen35_setA_common.py)
CPU_CLUSTERS = [
    "ai2/jupiter-cirrascale-2",
    "ai2/neptune-cirrascale",
    "ai2/ceres-cirrascale",
    "ai2/saturn-cirrascale",
]


def head_sha() -> str:
    sha = subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.run(
        ["git", "-C", HERE, "branch", "-r", "--contains", sha], capture_output=True, text=True
    ).stdout.strip()
    if not remote:
        sys.exit(
            f"HEAD {sha[:10]} is not on any remote branch -- push first (gantry runs "
            "pushed code only)"
        )
    return sha


def branch_of(sha: str) -> str:
    out = subprocess.check_output(["git", "-C", HERE, "branch", "-r", "--contains", sha], text=True)
    names = [b.strip().removeprefix("origin/") for b in out.splitlines() if "->" not in b]
    return "prasann/ctc-setA-sft" if "prasann/ctc-setA-sft" in names else names[0]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("step", choices=["prep", *NODES])
    ap.add_argument("--ref", default=None, help="full pushed SHA for prep/train (default HEAD)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="comma-separated setA tasks to train on (default: all 11); one task = single-task "
        "debugging, e.g. --tasks nq",
    )
    ap.add_argument(
        "--data-root",
        default=DEFAULT_SET_ROOT,
        help="set root; shards are read from <root>/shards_qwen35_256k/<task>/",
    )
    ap.add_argument(
        "--base-checkpoint",
        default=DEFAULT_BASE_CHECKPOINT,
        help="model_and_optim dir loaded weights-only",
    )
    ap.add_argument(
        "--variant",
        default=DEFAULT_VARIANT,
        choices=[DEFAULT_VARIANT, *[n for n in setA_variant_names() if n not in ("dense", "full")]],
        help="research arm from recipes/train/variants.py (default: dense)",
    )
    ap.add_argument(
        "--variant-opts",
        default="",
        help="the variant's options, k=v,... (';'-separated if a value holds commas); field names "
        "are train_ctc_suite's argparse dests, plus total_steps= (schedule horizon) and seed=",
    )
    args = ap.parse_args()
    if args.variant != DEFAULT_VARIANT:
        try:
            variant, _, _ = parse_variant_spec(args.variant, args.variant_opts)
        except ValueError as exc:
            ap.error(str(exc))
        if args.step in CP_ARMS and variant.cp_blocker:
            ap.error(f"--variant {args.variant} cannot run on the CP arm {args.step}: "
                     f"{variant.cp_blocker}")
    elif args.variant_opts:
        ap.error("--variant-opts needs a --variant")
    tasks = parse_tasks(args.tasks)
    unknown = [t for t in tasks if t not in DEFAULT_TASKS]
    if unknown:
        print(f"note: {unknown} are not setA tasks; prep will fail if their shards are missing")
    tag = task_tag(tasks)
    tag = f"{tag}-" if tag else ""

    stamp = datetime.now().strftime("%Y%m%dT%H%M")
    common = [
        "--workspace",
        "ai2/flex2",
        "--budget",
        "ai2/oe-other",
        "--priority",
        "urgent",
        "--beaker-image",
        IMAGE,
        "--weka",
        "oe-training-default:/weka/oe-training-default",
        "--system-python",
        "--allow-dirty",
        "--yes",
        "--timeout",
        "0",
    ]

    ref = args.ref or head_sha()
    if len(ref) != 40:
        sys.exit("--ref must be a full 40-character SHA")
    base = [
        GANTRY,
        "run",
        *common,
        "--branch",
        branch_of(ref),
        "--ref",
        ref,
        "--install",
        TRAIN_INSTALL,
        "--env-secret",
        "WANDB_API_KEY=PRASANNS_WANDB_API_KEY",
        "--env-secret",
        "BEAKER_TOKEN=PRASANNS_BEAKER_TOKEN",
        "--env",
        "PYTHONPATH=src",
        "--env",
        "OMP_NUM_THREADS=8",
        "--env",
        "NCCL_DEBUG=WARN",
        "--env",
        "OLMO_SHARED_FS=1",
        "--env",
        "PYTORCH_ALLOC_CONF=expandable_segments:True",
        "--env",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "--env",
        f"{ENV_TASKS}={','.join(tasks)}",
        "--env",
        f"{ENV_DATA_ROOT}={args.data_root}",
        "--env",
        f"{ENV_BASE_CKPT}={args.base_checkpoint}",
    ]
    if args.variant != DEFAULT_VARIANT:
        base += ["--env", f"{ENV_VARIANT}={args.variant}"]
        if args.variant_opts:
            base += ["--env", f"{ENV_VARIANT_OPTS}={args.variant_opts}"]
    if args.step == "prep":
        name = args.run_name or f"ctc-setA-sft-prep-{tag}{stamp}"
        cmd = [*base, "--name", name, "--gpus", "0", "--cpus", "32", "--shared-memory", "32GiB"]
        for c in CPU_CLUSTERS:
            cmd += ["--cluster", c]
        cmd += ["--", "python", f"{REL}/prep_setA_sft.py"]
    else:
        n = NODES[args.step]
        name = args.run_name or f"q35-4b-{args.variant}-ctc-setA-{tag}{args.step}-{stamp}"
        cmd = [
            *base,
            "--name",
            name,
            "--task-name",
            "train",
            "--cluster",
            "ai2/jupiter-cirrascale-2",
            "--gpus",
            "8",
            "--replicas",
            str(n),
            "--shared-memory",
            "10GiB",
            "--min-runtime",
            "1h",
            "--torchrun",
        ]
        if n > 1:
            cmd += ["--host-networking", "--synchronized-start-timeout", "90m"]
        cmd += [
            "--",
            f"{REL}/Qwen3.5-4B-dense-ctc-setA-{args.step}-SFT.py",
            "train",
            name,
            "ai2/jupiter-cirrascale-2",
        ]

    print(" ".join(shlex.quote(c) for c in cmd), flush=True)
    if args.dry_run:
        return
    log = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"{name}.gantry.log")
    # detached: gantry --timeout 0 returns once submitted; keep its output for the experiment URL
    with open(log, "w") as f:
        subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=False
        )
    print(open(log).read()[-1500:])


if __name__ == "__main__":
    main()
