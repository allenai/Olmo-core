#!/usr/bin/env python3
"""HOW TO: add a fixed PT decay to a protected 7:1 hero checkpoint.

Audience: the experiment owner and their coding agent. Read this docstring first.
This file contains a CPU-only planner and a distributed training entrypoint. It
does NOT allocate GPUs, register uploader lineages, convert models, or launch evals.
The training entrypoint is an example: it needs qualification in the production
image before committing to a full decay. CPU planning is independently usable.

SOURCE AND SCOPE
===============
Fetch this example's branch in allenai/OLMo-core:

    git fetch origin codex/hero-pt-decay-example-20261005
    git switch -c my-fixed-pt-decay origin/codex/hero-pt-decay-example-20261005

Its parent is the actual 128-GPU hero source:
    branch: codex/hero7to1-emo-128gpu-20260924
    commit: 5dcf5f96ec3d544523b674046324ffee41643e65

Use this whole checkout, not just this file copied onto current main. The hero
runtime uses Torch 2.11.0+cu130 and the pinned optimized kernels, not main's Torch
2.13 environment. Start from Beaker image 01KY8RWDARD3PR0F1T4ANGFH95 and run its
existing src/examples/olmo_ddp/olmoe3_profile_setup.sh inside the disposable worker.
The setup and olmoe3_small_hero_runtime.py verify the exact package/kernel pins.

This example supports the ORIGINAL 7:1 split-QK small PT hero, EMO or non-EMO.
It deliberately requires the saved GPU count (64 or 128), with 8 GPUs/node.
It preserves model/EMO configuration, optimizer, trainer step, data position and
per-rank RNG. It does not implement 3:1, MT, LC, SFT or GPU-count resharding.

CHOOSE A CHECKPOINT AND AN ADDITIONAL BUDGET
==========================================
Protected EMO native checkpoint aliases live at:
    /weka/olmo-3p5-checkpoints/protected/olmo35-small-hero-20260907/emo/{1T,...,14T}

These resolve to .../emo/step<STEP>/olmo-core. Use native OLMo-core checkpoints,
not HF exports: optimizer/trainer/RNG state is required. The planner resolves
aliases, checks completion markers and all saved rank audits, and records the
actual step/token count. It only reads the parent checkpoint.

The budget is ADDITIONAL decay tokens. Starting at a T boundary and appending a
fixed 100B/200B/etc. decay is intentional. This is not the historical 4T recipe
(step216000 -> step240000, which ENDED at 4.027T). No need to reproduce that recipe.

Global batch = 16,777,216 tokens/update; sequence length = 8192; peak LR = 0.0011.
Decay updates = ceil(additional_tokens / global_batch). LR decreases linearly to
zero over those updates. Nominal T aliases are approximate; use the printed
actual totals. Example: 6T is step357500 / 5,997,854,720,000 tokens. Adding 200B
takes 11,921 updates, ending at step369421 / 6,197,855,911,936 tokens.

On a machine with the checkpoint Weka mount, make a plan (stdlib Python only):

    python src/examples/olmo_ddp/olmoe3_hero_fixed_decay_example.py plan \
      --source /weka/olmo-3p5-checkpoints/protected/olmo35-small-hero-20260907/emo/6T \
      --decay-tokens 200B --arm emo --run-name my-7to1-emo-6t-plus200b \
      --output /path/on/shared/weka/my-7to1-emo-6t-plus200b.json

Use a new name AND plan for every source/budget experiment. Replace 6T with any
other protected boundary and 200B with e.g. 100B, 400B or 0.2T. Bare integers mean
tokens; B/T mean decimal billion/trillion. Omitting --output prints JSON only.
The planner never submits jobs or writes to checkpoint directories.

REGISTER THE NEW CHILD BEFORE TRAINING
=====================================
The inherited HeroAudit requires an independent uploader registration. In the
existing uploader environment, register the exact plan['registration'] using:

    import json
    from olmo_checkpoint_uploader.models import Registration
    from olmo_checkpoint_uploader.state import StateStore
    from olmoe3_small_hero_plan import CONTROL, STATE
    plan = json.load(open('/path/on/shared/weka/my-7to1-emo-6t-plus200b.json'))
    store = StateStore(CONTROL, STATE)
    # First check that this run_id is unused. Never replace another registration.
    assert not store.registration_path(plan['run_name']).exists()
    store.register(Registration(**plan['registration']))

Use the existing healthy uploader and private allenai/olmo-3p5-small bucket.
Only the NEW child is registered; do not register the protected source or change
its retention. The child has a separate prefix, two-checkpoint minimum and 1h
deletion grace. The trainer itself never prunes checkpoints. Each full checkpoint
is approximately 150GB; verify free space and uploader health before allocation.

WORKER COMMAND AND BEAKER WIRING
===============================
Reuse the hero image/setup, workspace ai2/olmo3p5-training, mounts, secret refs and
network settings. Mount olmo-3p5-checkpoints, dolma-3p5, and the volume holding the
plan. Use the current WANDB_API_KEY secret, never a literal key or parent's W&B ID.
Set worker replicas to plan['gpus']//8 and 8 GPUs/worker. Do not reuse the original
hero node entrypoint: it dispatches the constant-LR hero script.

On EVERY worker in ONE allocation, run the following after qualified setup:

    export DECAY_PLAN=/path/on/shared/weka/my-7to1-emo-6t-plus200b.json
    export DECAY_MASTER_ADDR=<reachable-hostname-of-this-allocation-rank-0>
    export DECAY_NODES=16  # 6T is 128 GPUs; use 8 for a 64-GPU source
    torchrun --nnodes="$DECAY_NODES" --nproc-per-node=8 \
      --node-rank="$BEAKER_REPLICA_RANK" \
      --master-addr="$DECAY_MASTER_ADDR" --master-port=29500 --max-restarts=0 \
      src/examples/olmo_ddp/olmoe3_hero_fixed_decay_example.py train \
      --plan "$DECAY_PLAN" --cluster ai2/holmes

Resolve rank 0 from CURRENT allocation membership; never use a previous attempt's
hostname. Existing olmoe3_profile_node.resolve_ready_leader shows how our Beaker
node wrapper does this. Configure infrastructure retries in the allocation spec.
This script selects the latest complete checkpoint in THIS child on each restart;
rank 0 broadcasts that choice to all training ranks. Partial saves are ignored.
There must be only one active allocation for a given child run at any time.

For a short first pass, append --stop-after-steps 2 on ALL workers; then run the
same command without that flag to resume. The full decay END remains max_duration
during the short pass, so LR does not prematurely decay to zero. This is an
optional startup/restore check, not a simulated crash test. Before a long run,
check restored step/tokens, matching model and optimizer, finite loss, LR and the
saved full-state checkpoint. This example has not been GPU-qualified itself.

DETAILS FOR AN AGENT ADAPTING THIS EXAMPLE
========================================
* WSD(warmup=2000, decay=end-start, decay_fraction=None, decay_min_lr=0) uses the
  RESTORED absolute step. The original warmup is already over; no fresh warmup.
* max_duration is the absolute final step, never the number of decay updates.
* Both optimizer reset flags stay False; load_optim_state/load_trainer_state True.
* The run registry, save folder, audit/storage/notification callbacks, W&B ID,
  uploader lineage and prefix must all identify the NEW child. This adapter binds
  those through hero.find_run; changing only trainer.save_folder is insufficient.
* The inherited 14T horizon is replaced for this child, allowing e.g. 14T + 200B.
* Preserve the original data/optimizer/model policy. The source model config is
  checked against the built model before training; don't silently change EMO,
  QK gains or the 7:1 layer pattern. Keep the saved GPU count for exact RNG restore.
* Source completion/audit checks are structural checks, not full payload checksums;
  use the existing protection receipts for upload/download integrity verification.
* This ends after PT decay. Conversion/evaluation and MT/LC/SFT are separate jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

BATCH = 16_777_216
LR = 0.0011
BASE_COMMIT = "5dcf5f96ec3d544523b674046324ffee41643e65"
CHILD_ROOT = Path("/weka/olmo-3p5-checkpoints/production-fixed-pt-decays")


def token_count(value: str) -> int:
    """Parse an integral token count, with optional decimal B/T suffix."""
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([bBtT]?)", value)
    if match is None:
        raise argparse.ArgumentTypeError("Use positive tokens, e.g. 200000000000, 200B or 0.2T")
    try:
        count = Decimal(match[1]) * {"": 1, "B": 10**9, "T": 10**12}[match[2].upper()]
        if count <= 0 or count != count.to_integral_value():
            raise ValueError
        return int(count)
    except (InvalidOperation, ValueError) as exc:
        raise argparse.ArgumentTypeError("Token budget must be a positive whole number") from exc


def read_json(path: Path):
    """Read a small local metadata document."""
    return json.loads(path.read_text())


def checkpoint_info(path: Path) -> dict:
    """Validate native completion and per-rank state inventory; never deserialize pickle."""
    path = path.resolve(strict=True)
    for name in (".metadata.json", "model_and_optim/.metadata", "config.json"):
        if not (path / name).is_file():
            raise ValueError(f"Incomplete native checkpoint: {path / name}")
    if read_json(path / ".metadata.json").get("ephemeral"):
        raise ValueError("An ephemeral checkpoint is not a full-state decay source")
    audit = read_json(path / "resume_audit/rank0.json")
    step, tokens, gpus = (audit[k] for k in ("step", "tokens", "gpus"))
    if gpus not in (64, 128) or step <= 2000 or tokens != step * BATCH:
        raise ValueError(f"Unexpected hero step/tokens/GPU count: {step}, {tokens}, {gpus}")
    for rank in range(gpus):
        row = read_json(path / f"resume_audit/rank{rank}.json")
        if (
            tuple(row[k] for k in ("step", "tokens", "gpus", "rank")) != (step, tokens, gpus, rank)
            or not (path / f"train/rank{rank}.pt").is_file()
        ):
            raise ValueError(f"Incomplete or inconsistent rank {rank}: {path}")
    return {"source": str(path), "start_step": step, "start_tokens": tokens, "gpus": gpus}


def make_plan(source: Path, decay_tokens: int, arm: str, name: str) -> dict:
    """Read the native source and describe an isolated fixed-budget decay."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,99}", name):
        raise ValueError("Use a unique lowercase alphanumeric/hyphen run name, <=100 characters")
    info = checkpoint_info(source)
    source = Path(info["source"])
    config = read_json(source / "config.json")
    tm = config["train_module"]
    if not tm["scheduler"]["_CLASS_"].endswith(".ConstantWithWarmup"):
        raise ValueError("Choose a non-decayed constant-LR PT checkpoint")
    if tm["optim"]["lr"] != LR or config["data_loader"]["global_batch_size"] != BATCH:
        raise ValueError("Source does not use the original hero LR/global batch")
    blocks = [config["model"]["block"], *config["model"].get("block_overrides", {}).values()]
    routers = [b["routed_experts_router"] for b in blocks if "routed_experts_router" in b]
    if not routers or any(bool(r.get("emo")) != (arm == "emo") for r in routers):
        raise ValueError("--arm disagrees with the source router/EMO configuration")
    if decay_tokens <= 0:
        raise ValueError("Decay budget must be positive")
    updates = (decay_tokens + BATCH - 1) // BATCH
    root = CHILD_ROOT / name
    if root.exists():
        raise ValueError(f"Choose an unused child run name; output already exists: {root}")
    return {
        "schema": 1,
        "base_commit": BASE_COMMIT,
        **info,
        "source_config_sha256": hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
        "run_name": name,
        "arm": arm,
        "requested_decay_tokens": decay_tokens,
        "decay_steps": updates,
        "actual_decay_tokens": updates * BATCH,
        "end_step": info["start_step"] + updates,
        "end_tokens": info["start_tokens"] + updates * BATCH,
        "global_batch_tokens": BATCH,
        "save_folder": str(root),
        "registration": {
            "run_id": name,
            "lineage_id": name,
            "checkpoint_root": str(root),
            "bucket_id": "allenai/olmo-3p5-small",
            "remote_prefix": f"fixed-pt-decays/{name}",
            "deletion_mode": "apply",
            "min_local_checkpoints": 2,
            "delete_grace_seconds": 3600,
        },
    }


@dataclass(frozen=True)
class ChildRun:
    """Supply the identity interface consumed by the inherited hero callbacks."""

    plan: dict
    smoke: bool = False

    @property
    def emo(self):
        return self.plan["arm"] == "emo"

    @property
    def arm(self):
        return self.plan["arm"]

    @property
    def run_id(self):
        return self.plan["run_name"]

    @property
    def root(self):
        return Path(self.plan["save_folder"])

    @property
    def bucket(self):
        return self.plan["registration"]["bucket_id"]

    @property
    def prefix(self):
        return self.plan["registration"]["remote_prefix"]

    def as_dict(self):
        """Record the complete source/budget/identity plan in training provenance."""
        return self.plan


def select_checkpoint(plan: dict) -> tuple[int, str]:
    """Choose the latest complete child; fail on corruption rather than rewinding it."""
    root = Path(plan["save_folder"])
    choices = [(plan["start_step"], plan["source"])]
    for path in root.glob("step*"):
        if not re.fullmatch(r"step[0-9]+", path.name) or not (path / ".metadata.json").is_file():
            continue
        info = checkpoint_info(path)
        step = info["start_step"]
        if path.is_symlink() or step != int(path.name[4:]) or info["gpus"] != plan["gpus"]:
            raise ValueError(f"Unexpected child checkpoint: {path}")
        if not plan["start_step"] < step <= plan["end_step"]:
            raise ValueError(f"Child step outside the declared decay: {path}")
        saved = read_json(path / "config.json")
        scheduler = saved["train_module"]["scheduler"]
        if (
            saved["run_name"] != plan["run_name"]
            or not scheduler["_CLASS_"].endswith(".WSD")
            or scheduler.get("decay") != plan["decay_steps"]
        ):
            raise ValueError(f"Checkpoint belongs to a different child or decay schedule: {path}")
        choices.append((step, str(path)))
    return max(choices)


def train_example(args):
    """Run the inherited full-state hero with an isolated identity and WSD schedule."""
    plan = read_json(args.plan)
    if plan["schema"] != 1 or plan["base_commit"] != BASE_COMMIT:
        raise ValueError("Unexpected plan schema/source revision")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,99}", plan["run_name"]):
        raise ValueError("Invalid child run name")
    if plan["arm"] not in ("emo", "non-emo"):
        raise ValueError("Invalid PT lineage arm")
    source = Path(plan["source"])
    info = checkpoint_info(source)
    if any(info[k] != plan[k] for k in info):
        raise ValueError("Source checkpoint metadata changed since planning")
    if (
        hashlib.sha256((source / "config.json").read_bytes()).hexdigest()
        != plan["source_config_sha256"]
    ):
        raise ValueError("Source config changed since planning")
    updates = (plan["requested_decay_tokens"] + BATCH - 1) // BATCH
    if (
        updates <= 0
        or updates != plan["decay_steps"]
        or plan["end_step"] != plan["start_step"] + updates
        or plan["end_tokens"] != plan["end_step"] * BATCH
        or plan["actual_decay_tokens"] != updates * BATCH
        or plan["global_batch_tokens"] != BATCH
        or Path(plan["save_folder"]) != CHILD_ROOT / plan["run_name"]
    ):
        raise ValueError("Inconsistent plan; regenerate it rather than editing derived fields")
    os.environ["OLMO35_HERO_GPUS"] = str(plan["gpus"])
    os.environ["OLMO35_HERO_ALLOW_CONTINUATION"] = "1"
    os.environ["OLMO35_HERO_SMOKE"] = "0"
    os.environ["WANDB_RUN_ID"] = hashlib.sha256(plan["run_name"].encode()).hexdigest()[:16]
    os.environ["WANDB_RESUME"] = "allow"

    import olmoe3_small_hero as hero
    import torch
    import torch.distributed as dist
    from olmoe3_small_hero_runtime import verify_runtime

    from olmo_core.internal.experiment import CliContext, SubCmd, train
    from olmo_core.optim.scheduler import WSD
    from olmo_core.train import (
        Duration,
        prepare_training_environment,
        teardown_training_environment,
    )
    from olmo_core.train.common import LoadStrategy
    from olmo_core.train.utils import EnvRngStates

    verify_runtime()
    prepare_training_environment(backend="cpu:gloo,cuda:nccl", shared_filesystem=True)
    if dist.get_world_size() != plan["gpus"]:
        raise ValueError("Use the source GPU count; this example does not reshard")
    child = ChildRun(plan)
    registration = read_json(hero.CONTROL / "registrations" / f"{child.run_id}.json")
    if any(registration.get(k) != v for k, v in plan["registration"].items()):
        raise ValueError("Uploader registration differs from the approved child plan")
    if (
        child.bucket != "allenai/olmo-3p5-small"
        or child.prefix != f"fixed-pt-decays/{child.run_id}"
        or registration["checkpoint_root"] != str(child.root)
        or registration["lineage_id"] != child.run_id
    ):
        raise ValueError("Uploader identity must be isolated from the parent and other children")

    def find_run(name):
        if name != child.run_id:
            raise ValueError(f"Unexpected child identity: {name}")
        return child

    hero.find_run = find_run
    hero.FINAL_STEPS = plan["end_step"]  # Also permits a decay after the 14T boundary.
    selection = [None]
    if dist.get_rank() == 0:
        child.root.mkdir(parents=True, exist_ok=True)
        identity = child.root / "fixed-decay-plan.json"
        if identity.exists():
            if read_json(identity) != plan:
                raise ValueError("This output directory belongs to a different decay plan")
        else:
            if any(child.root.iterdir()):
                raise ValueError("Refusing to adopt an existing child directory without its plan")
            with identity.open("x") as handle:
                json.dump(plan, handle, indent=2, sort_keys=True)
        selection[0] = select_checkpoint(plan)
    dist.broadcast_object_list(selection, src=0)
    start, load_path = selection[0]
    stop = plan["end_step"]
    if args.stop_after_steps is not None:
        if args.stop_after_steps <= 0:
            raise ValueError("--stop-after-steps must be positive")
        stop = min(stop, start + args.stop_after_steps)
    if start == plan["end_step"]:
        if dist.get_rank() == 0:
            print("DECAY_ALREADY_COMPLETE", start, flush=True)
        teardown_training_environment()
        return
    os.environ["OLMO35_HERO_EXPECTED_START"] = str(start)
    os.environ["OLMO35_HERO_STOP"] = str(stop)

    def equal(a, b):
        if isinstance(a, dict):
            return (
                isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
            )
        if isinstance(a, (list, tuple)):
            return isinstance(b, (list, tuple)) and len(a) == len(b) and all(map(equal, a, b))
        if torch.is_tensor(a):
            return torch.equal(a.cpu(), b.cpu())
        if hasattr(a, "shape"):
            return bool((a == b).all())
        return a == b

    @dataclass
    class DecayAudit(hero.HeroAudit):
        """Retain sampled weight/optimizer audits and additionally verify RNG/data restoration."""

        def post_checkpoint_loaded(self, path):
            super().post_checkpoint_loaded(path)
            if self.step != start:
                raise RuntimeError(f"Restored step {self.step} differs from selected step {start}")
            saved = torch.load(
                Path(path) / "train" / f"rank{dist.get_rank()}.pt",
                map_location="cpu",
                weights_only=False,
            )
            if not equal(saved["rng"], EnvRngStates.current_state().as_dict()):
                raise RuntimeError("Per-rank RNG restore mismatch")
            if not equal(saved["data_loader"], self.trainer.data_loader.state_dict()):
                raise RuntimeError("Data position restore mismatch")

        def log_metrics(self, step, metrics):
            """Keep the original finite-loss checks and verify the absolute-step decay LR."""
            super().log_metrics(step, metrics)
            for key, value in metrics.items():
                if key.startswith("optim/LR ("):
                    expected = LR * (plan["end_step"] - step) / plan["decay_steps"]
                    if not math.isclose(float(value), expected, rel_tol=1e-6, abs_tol=1e-10):
                        raise RuntimeError(
                            f"Unexpected decay LR at step {step}: {value} != {expected}"
                        )

    config = hero.config_builder()(
        CliContext(__file__, SubCmd.train, child.run_id, args.cluster, [])
    )
    actual_model = json.loads(json.dumps(config.model.as_config_dict()))
    if actual_model != read_json(source / "config.json")["model"]:
        raise ValueError("Source model differs from this 7:1 hero recipe; do not bypass this check")
    config.launch = None
    tm, tr = config.train_module, config.trainer
    tm.scheduler = WSD(warmup=2000, decay=updates, decay_fraction=None, decay_min_lr=0.0)
    tm.reset_optimizer_states_on_load = False
    tm.reset_optimizer_states_on_resume = False
    tr.load_path = load_path
    tr.load_strategy = LoadStrategy.always
    tr.load_optim_state = tr.load_trainer_state = True
    tr.max_duration = Duration.steps(plan["end_step"])
    tr.hard_stop = Duration.steps(stop)
    tr.callbacks["checkpointer"].save_interval = 500
    tr.callbacks["checkpointer"].fixed_steps = [stop]
    tr.callbacks["hero_audit"] = DecayAudit(
        output_dir=str(child.root / "audit"), run_id=child.run_id
    )
    wb = tr.callbacks["wandb"]
    wb.group = "hero-fixed-pt-decays"
    wb.tags = ["7to1", child.arm, "fixed-pt-decay", f"{plan['gpus']}g", "full-state-resume"]
    if args.stop_after_steps is not None:
        tr.no_evals = True
    if dist.get_rank() == 0:
        print(
            "FIXED_DECAY_START",
            json.dumps({**plan, "resume_step": start, "stop_step": stop}),
            flush=True,
        )
    train(config)
    teardown_training_environment()


def main():
    """Plan without training dependencies, or train inside a qualified distributed allocation."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    planner = commands.add_parser(
        "plan", help="Read source metadata and print/write a new decay plan"
    )
    planner.add_argument("--source", type=Path, required=True)
    planner.add_argument("--decay-tokens", type=token_count, required=True)
    planner.add_argument("--arm", choices=("emo", "non-emo"), required=True)
    planner.add_argument("--run-name", required=True)
    planner.add_argument("--output", type=Path)
    trainer = commands.add_parser(
        "train", help="Worker entrypoint: run under torchrun after setup/registration"
    )
    trainer.add_argument("--plan", type=Path, required=True)
    trainer.add_argument("--cluster", default="ai2/holmes")
    trainer.add_argument("--stop-after-steps", type=int)
    args = parser.parse_args()
    if args.command == "plan":
        plan = make_plan(args.source, args.decay_tokens, args.arm, args.run_name)
        serialized = json.dumps(plan, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            with args.output.open("x") as handle:
                handle.write(serialized)
        print(serialized, end="")
    else:
        train_example(args)


if __name__ == "__main__":
    main()
