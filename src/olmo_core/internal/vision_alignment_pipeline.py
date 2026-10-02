"""Run vision-alignment phases sequentially within one allocation."""

import json
import logging
import os
import signal
import subprocess
import sys

import torch
from rich import print

from olmo_core.distributed.utils import barrier, get_rank
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import file_exists, normalize_path, resource_path
from olmo_core.train import LoadStrategy, teardown_training_environment
from olmo_core.train.callbacks.multimodal import write_file_overwrite
from olmo_core.train.checkpoint import Checkpointer
from olmo_core.train.common import DurationUnit
from olmo_core.utils import prepare_cli_environment

from .experiment import CliContext, SubCmd, parse_cli_args, train
from .vision_alignment import (
    AlignmentPhase,
    VisionAlignmentExperimentConfig,
    build_config,
)

log = logging.getLogger(__name__)

_STAGES = (AlignmentPhase.bridge, AlignmentPhase.perception, AlignmentPhase.joint)
_SUCCESS_FILE = "stage_complete.json"
_WORKER_ENV = "OLMO_ALIGNMENT_PIPELINE_STAGE"


def _selected_phase(cli: CliContext) -> str:
    phase = "all"
    for arg in cli.overrides:
        if arg.startswith("--recipe.phase="):
            phase = arg.partition("=")[2]
    if phase not in ("all", *_STAGES):
        raise OLMoConfigurationError(f"Unknown alignment phase: {phase!r}")
    return phase


def _stage_cli(cli: CliContext, phase: AlignmentPhase, parent: str | None) -> CliContext:
    overrides = [arg for arg in cli.overrides if not arg.startswith("--recipe.phase=")]
    overrides.append(f"--recipe.phase={phase}")
    if parent is not None:
        overrides.append(f"--recipe.parent_checkpoint={parent}")
    return CliContext(cli.script, cli.cmd, f"{cli.run_name}-{phase}", cli.cluster, overrides)


def _endpoint(config: VisionAlignmentExperimentConfig) -> str:
    duration = config.trainer.max_duration
    if duration.unit != DurationUnit.steps or duration.value <= 0:
        raise OLMoConfigurationError("Sequential alignment requires a positive step duration")
    if config.trainer.no_checkpoints or not config.trainer.callbacks["checkpointer"].enabled:
        raise OLMoConfigurationError("Sequential alignment requires checkpoint saving")
    if config.trainer.load_strategy == LoadStrategy.never:
        raise OLMoConfigurationError("Sequential alignment requires checkpoint loading for resume")
    if config.recipe.phase != AlignmentPhase.bridge and (
        config.trainer.load_optim_state is not False
        or config.trainer.load_trainer_state is not False
    ):
        raise OLMoConfigurationError("Alignment phase handoffs must load only model weights")
    return f"{normalize_path(config.trainer.save_folder)}/step{duration.value}"


def _finalized(checkpoint: str) -> bool:
    # Model-only checkpoints cannot resume the active phase's optimizer and loader.
    return all(
        file_exists(f"{checkpoint}/{name}")
        for name in (
            Checkpointer.METADATA_FNAME,
            "model_and_optim/.metadata",
            "train/rank0.pt",
        )
    )


def _completion(config: VisionAlignmentExperimentConfig, checkpoint: str) -> dict:
    return {
        "phase": config.recipe.phase,
        "step": config.trainer.max_duration.value,
        "checkpoint": checkpoint,
        "parent_checkpoint": config.recipe.parent_checkpoint,
        "pretraining_checkpoint": config.pretraining_checkpoint,
    }


def _validate_endpoint(
    config: VisionAlignmentExperimentConfig, checkpoint: str, *, require_config: bool = True
) -> None:
    expected_step = config.trainer.max_duration.value
    if require_config or file_exists(f"{checkpoint}/config.json"):
        with resource_path(checkpoint, "config.json").open() as stream:
            saved = json.load(stream)
        recipe = saved.get("recipe", {})
        duration = saved.get("trainer", {}).get("max_duration", {})
        if (
            recipe.get("phase") != config.recipe.phase
            or recipe.get("parent_checkpoint") != config.recipe.parent_checkpoint
            or saved.get("pretraining_checkpoint") != config.pretraining_checkpoint
            or duration.get("unit") != DurationUnit.steps
            or duration.get("value") != expected_step
        ):
            raise OLMoConfigurationError(
                f"Checkpoint does not match the alignment stage: {checkpoint}"
            )
    state = torch.load(
        resource_path(checkpoint, "train/rank0.pt"), map_location="cpu", weights_only=False
    )
    if state.get("global_step") != expected_step or state.get("max_steps") != expected_step:
        raise OLMoConfigurationError(f"Checkpoint did not reach the stage horizon: {checkpoint}")


def _stage_complete(config: VisionAlignmentExperimentConfig, checkpoint: str) -> bool:
    folder = normalize_path(config.trainer.save_folder)
    if not file_exists(f"{folder}/{_SUCCESS_FILE}"):
        return False
    with resource_path(folder, _SUCCESS_FILE).open() as stream:
        completed = json.load(stream)
    if completed != _completion(config, checkpoint) or not _finalized(checkpoint):
        return False
    _validate_endpoint(config, checkpoint)
    return True


def _train_stage(config: VisionAlignmentExperimentConfig) -> None:
    checkpoint = _endpoint(config)
    completion = _completion(config, checkpoint)
    missing_config = None
    if _finalized(checkpoint):
        if not file_exists(f"{checkpoint}/config.json"):
            missing_config = config.as_config_dict()
        _validate_endpoint(config, checkpoint, require_config=missing_config is None)
        # Resume a final checkpoint whose evaluation did not finish. Do not rewrite it.
        config.trainer.callbacks["checkpointer"].enabled = False
        config.trainer.callbacks["multimodal_evaluator"].eval_on_startup = False
    trainer = train(config)
    if trainer.is_canceled or trainer.global_step != completion["step"]:
        raise RuntimeError(
            f"Alignment {config.recipe.phase} stopped at step {trainer.global_step}; "
            f"step {completion['step']} is required before continuing"
        )
    if not _finalized(checkpoint):
        raise RuntimeError(f"Alignment did not finalize its endpoint checkpoint: {checkpoint}")
    if missing_config is not None:
        # A save can finish before its config callback. Recover only the missing metadata
        # after full-state resume and successful evaluation, without rewriting model state.
        if get_rank() == 0:
            trainer.write_file("config.json", json.dumps(missing_config), dir=checkpoint)
        barrier()
    _validate_endpoint(config, checkpoint)
    barrier()
    if get_rank() == 0:
        write_file_overwrite(trainer, _SUCCESS_FILE, json.dumps(completion))
    barrier()


def _torchrun_command(cli: CliContext, index: int) -> list[str]:
    nodes = int(os.environ.get("BEAKER_REPLICA_COUNT", "1"))
    rank = int(os.environ.get("BEAKER_REPLICA_RANK", "0"))
    gpus = int(os.environ.get("BEAKER_ASSIGNED_GPU_COUNT", str(torch.cuda.device_count())))
    leader = os.environ.get("BEAKER_LEADER_REPLICA_HOSTNAME")
    if nodes > 1 and not leader:
        raise OLMoConfigurationError("Multi-node alignment requires a Beaker leader hostname")
    port = int(os.environ.get("MASTER_PORT", "29500")) + index
    if nodes < 1 or not 0 <= rank < nodes or gpus < 1 or not 1 <= port <= 65535:
        raise OLMoConfigurationError("Invalid alignment node, GPU, or rendezvous configuration")
    run_id = os.environ.get("BEAKER_EXPERIMENT_ID", cli.run_name)
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={nodes}",
        f"--nproc-per-node={gpus}",
        f"--node-rank={rank}",
        "--rdzv-backend=static",
        f"--rdzv-endpoint={leader or '127.0.0.1'}:{port}",
        f"--rdzv-id={run_id}-{index}",
        "--rdzv-conf=timeout=420",
        cli.script,
        str(SubCmd.train),
        cli.run_name,
        cli.cluster,
        *cli.overrides,
    ]


def _run_command(command: list[str]) -> None:
    env = {**os.environ, _WORKER_ENV: "1"}
    interrupted = None
    with subprocess.Popen(command, env=env, start_new_session=True) as process:

        def forward_signal(signum, _frame):
            nonlocal interrupted
            interrupted = signum
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    pass

        handlers = {
            signum: signal.signal(signum, forward_signal)
            for signum in (signal.SIGTERM, signal.SIGINT)
        }
        try:
            returncode = process.wait()
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
        if interrupted is not None:
            raise SystemExit(128 + interrupted)
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)


def _run_pipeline(cli: CliContext, bridge: VisionAlignmentExperimentConfig) -> None:
    if "RANK" in os.environ:
        raise OLMoConfigurationError(
            "Run all-stage training with python, not torchrun; the controller starts torchrun "
            "for each stage. Use --recipe.phase for an existing torchrun allocation."
        )
    parent = None
    for index, phase in enumerate(_STAGES):
        stage_cli = _stage_cli(cli, phase, parent)
        config = bridge if phase == AlignmentPhase.bridge else build_config(stage_cli)
        checkpoint = _endpoint(config)
        if _stage_complete(config, checkpoint):
            log.info("Skipping completed alignment phase %s: %s", phase, checkpoint)
        else:
            log.info("Starting alignment phase %s", phase)
            _run_command(_torchrun_command(stage_cli, index))
            if not _stage_complete(config, checkpoint):
                raise RuntimeError(f"Alignment phase {phase} did not complete successfully")
        parent = checkpoint


def main() -> None:
    """Run all alignment phases by default, or one explicitly selected phase."""
    cli = parse_cli_args()
    if _selected_phase(cli) != "all":
        config = build_config(cli)
        cli.cmd.prepare_environment(config)
        if cli.cmd == SubCmd.train and os.environ.get(_WORKER_ENV) == "1":
            try:
                _train_stage(config)
            finally:
                teardown_training_environment()
        else:
            cli.cmd.run(config)
        return

    if cli.cmd not in (SubCmd.launch, SubCmd.train, SubCmd.dry_run):
        raise OLMoConfigurationError(f"{cli.cmd} requires an explicit --recipe.phase")
    if any(
        arg.partition("=")[0]
        in {"--trainer.save_folder", "--recipe.parent_checkpoint", "--recipe", "--trainer"}
        for arg in cli.overrides
    ):
        raise OLMoConfigurationError(
            "All-stage alignment assigns separate outputs and parent checkpoints; "
            "use recipe.output_root or select one --recipe.phase"
        )
    prepare_cli_environment()
    bridge = build_config(_stage_cli(cli, AlignmentPhase.bridge, None))
    _endpoint(bridge)
    print("Alignment stages: " + " → ".join(f"{cli.run_name}-{stage}" for stage in _STAGES))
    if cli.cmd == SubCmd.launch:
        assert bridge.launch is not None
        bridge.launch.cmd = cli.remote_cmd
        bridge.launch.torchrun = False
        bridge.launch.name = bridge.launch.name.replace(
            f"{bridge.run_name}-train", f"{cli.run_name}-train", 1
        )
        bridge.launch.launch()
    elif cli.cmd == SubCmd.dry_run:
        SubCmd.dry_run.run(bridge)
        print("Perception and joint configs are resolved from their completed parent checkpoints.")
    else:
        _run_pipeline(cli, bridge)
