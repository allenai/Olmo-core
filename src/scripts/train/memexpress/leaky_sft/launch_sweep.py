"""Prepare six detached Gantry commands; submit only with an explicit --submit.

All training jobs: one GPU, one replica, Jupiter, urgent, minimum runtime 1h.
A pushed source SHA containing the LeakySFT implementation is required to submit.
"""
import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from sweep_settings import ARMS, CLUSTER, DATA_ROOT, SEQUENCE_LENGTH

FAMILY = "src/scripts/train/memexpress/leaky_sft"


def build_command(arm, name, ref, output_dir):
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    return [
        "uvx",
        "--from",
        "beaker-gantry@3.7.0",
        "gantry",
        "run",
        "--name",
        name,
        "--task-name",
        "train",
        "--workspace",
        "ai2/flex2",
        "--budget",
        "ai2/oe-other",
        "--cluster",
        CLUSTER,
        "--priority",
        "urgent",
        "--min-runtime",
        "1h",
        "--gpus",
        "1",
        "--replicas",
        "1",
        "--torchrun",
        "--beaker-image",
        "tylerr/olmo-core-tch291cu128-2025-11-25",
        "--shared-memory",
        "10GiB",
        "--weka",
        "oe-training-default:/weka/oe-training-default",
        "--ref",
        ref,
        "--env-secret",
        "BEAKER_TOKEN=amandab_BEAKER_TOKEN",
        "--env-secret",
        "WANDB_API_KEY=AMANDAB_WANDB_API_KEY",
        "--env-secret",
        "HF_TOKEN=amandab_HF_TOKEN",
        "--env",
        "PYTHONPATH=src",
        "--env",
        "OLMO_SHARED_FS=1",
        "--env",
        "OMP_NUM_THREADS=8",
        "--env",
        "NCCL_DEBUG=WARN",
        "--env",
        "PYTORCH_ALLOC_CONF=expandable_segments:True",
        "--system-python",
        "--install",
        "python -m pip install -e '.[beaker,fla,wandb,transformers]' 'fla-core==0.4.2' liger-kernel",
        "--yes",
        "--no-logs",
        "--timeout",
        "0",
        "--save-spec",
        str(Path(output_dir) / f"{name}.yaml"),
        "--",
        f"{FAMILY}/Qwen3.5-4B-contradiction-32k-1ep-LeakySFT-{arm}.py",
        "train",
        name,
        CLUSTER,
    ]


def make_plan(prefix, ref, output_dir):
    return [
        dict(
            arm=arm,
            cpt_loss_scaler=alpha,
            run_name=f"{prefix}-{arm}",
            epochs=1,
            gpus=1,
            replicas=1,
            cluster=CLUSTER,
            priority="urgent",
            min_runtime="1h",
            source_ref=ref,
            status="prepared_not_submitted",
            data_root=DATA_ROOT,
            max_example_tokens=SEQUENCE_LENGTH,
            long_doc_strategy="exclude",
            data_provenance="Qwen3.5 2K–256K corpus, retaining complete examples <=32768 tokens",
            command=build_command(arm, f"{prefix}-{arm}", ref, output_dir),
        )
        for arm, alpha in ARMS.items()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="PUSHED_SHA_CONTAINING_LEAKYSFT")
    parser.add_argument(
        "--prefix",
        default="q35-leakysft-contra-xlong-filter32k-1ep-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.prefix):
        parser.error("prefix must contain only letters, digits, underscores and hyphens")
    if args.submit:
        if not re.fullmatch(r"[0-9a-f]{40}", args.ref):
            parser.error("submission requires a full pushed --ref SHA")
        # Never silently submit a remote checkout that omits the local implementation.
        paths = ["src/olmo_core", FAMILY]
        subprocess.run(["git", "diff", "--exit-code", args.ref, "--", *paths], check=True)
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "--", *paths]
        )
        if untracked.strip():
            parser.error("Commit and push the required source files before submission")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan = make_plan(args.prefix, args.ref, args.output_dir)
    (args.output_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.submit:
        with (args.output_dir / "launch-ledger.jsonl").open("a") as ledger:
            for row in plan:
                row = dict(row, status="submitting")
                ledger.write(json.dumps(row) + "\n")
                ledger.flush()
                log_path = args.output_dir / f"{row['run_name']}.log"
                with log_path.open("w") as log:
                    result = subprocess.run(row["command"], stdout=log, stderr=subprocess.STDOUT)
                row.update(
                    status="submitted" if result.returncode == 0 else "submission_failed",
                    returncode=result.returncode,
                    submission_log=str(log_path),
                )
                ledger.write(json.dumps(row) + "\n")
                ledger.flush()
                result.check_returncode()
    print(args.output_dir / "plan.json")


if __name__ == "__main__":
    main()
