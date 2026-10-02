import signal
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal import experiment
from olmo_core.internal import vision_alignment_pipeline as pipeline
from olmo_core.internal.experiment import CliContext, SubCmd
from olmo_core.train import LoadStrategy
from olmo_core.train.common import Duration


@pytest.fixture
def cli():
    return CliContext(
        "src/scripts/train/Vision-Align.py",
        SubCmd.launch,
        "alignment",
        "ai2/holmes",
        ["--recipe.pretraining_checkpoint=/pretrained/step100", "--launch.min_runtime=8h"],
    )


@pytest.fixture
def entrypoint(monkeypatch, cli):
    launch = SimpleNamespace(
        name="alignment-bridge-train-uuid", cmd=[], torchrun=None, launch=Mock()
    )
    config = SimpleNamespace(
        run_name="alignment-bridge",
        recipe=SimpleNamespace(phase="bridge"),
        launch=launch,
        trainer=SimpleNamespace(
            max_duration=Duration.steps(500),
            save_folder="/outputs/alignment-bridge",
            no_checkpoints=False,
            load_strategy=LoadStrategy.if_available,
            callbacks={"checkpointer": SimpleNamespace(enabled=True)},
        ),
    )
    builder = Mock(return_value=config)
    monkeypatch.setattr(pipeline, "parse_cli_args", lambda: cli)
    monkeypatch.setattr(pipeline, "build_config", builder)
    monkeypatch.setattr(pipeline, "prepare_cli_environment", Mock())
    monkeypatch.delenv(pipeline._WORKER_ENV, raising=False)
    return config, builder


@pytest.mark.parametrize("explicit_all", [False, True])
def test_default_launch_submits_one_controller_job(cli, entrypoint, explicit_all):
    if explicit_all:
        cli.overrides.append("--recipe.phase=all")
    config, builder = entrypoint
    pipeline.main()
    assert builder.call_count == 1
    bridge_cli = builder.call_args.args[0]
    assert bridge_cli.run_name == "alignment-bridge"
    assert "--recipe.phase=bridge" in bridge_cli.overrides
    assert "--recipe.phase=all" not in bridge_cli.overrides
    assert config.launch.cmd == cli.remote_cmd
    assert config.launch.name == "alignment-train-uuid"
    assert config.launch.torchrun is False
    config.launch.launch.assert_called_once_with()


@pytest.mark.parametrize("phase", ["bridge", "perception", "joint"])
def test_explicit_phase_uses_standard_single_stage_runner(monkeypatch, cli, entrypoint, phase):
    config, builder = entrypoint
    cli.overrides.append(f"--recipe.phase={phase}")
    prepare, run = Mock(), Mock()
    monkeypatch.setattr(SubCmd, "prepare_environment", prepare)
    monkeypatch.setattr(SubCmd, "run", run)
    pipeline.main()
    builder.assert_called_once_with(cli)
    prepare.assert_called_once_with(config)
    run.assert_called_once_with(config)
    config.launch.launch.assert_not_called()


@pytest.mark.parametrize(
    "override",
    [
        "--trainer.save_folder=/shared",
        "--recipe.parent_checkpoint=/parent",
        "--recipe={}",
        "--trainer={save_folder: /shared}",
    ],
)
def test_all_stage_rejects_conflicting_handoff_overrides(cli, entrypoint, override):
    cli.overrides.append(override)
    with pytest.raises(OLMoConfigurationError, match="separate outputs"):
        pipeline.main()
    entrypoint[1].assert_not_called()


@pytest.mark.parametrize("cmd", [SubCmd.prep, SubCmd.launch_prep, SubCmd.eval_checkpoints])
def test_phase_specific_commands_require_a_phase(cli, entrypoint, cmd):
    cli.cmd = cmd
    with pytest.raises(OLMoConfigurationError, match="explicit --recipe.phase"):
        pipeline.main()
    entrypoint[1].assert_not_called()


def test_pipeline_worker_runs_completion_checks_and_tears_down(monkeypatch, cli, entrypoint):
    cli.cmd = SubCmd.train
    cli.overrides.append("--recipe.phase=bridge")
    monkeypatch.setenv(pipeline._WORKER_ENV, "1")
    prepare, train, teardown = Mock(), Mock(), Mock()
    monkeypatch.setattr(SubCmd, "prepare_environment", prepare)
    monkeypatch.setattr(pipeline, "_train_stage", train)
    monkeypatch.setattr(pipeline, "teardown_training_environment", teardown)
    pipeline.main()
    prepare.assert_called_once_with(entrypoint[0])
    train.assert_called_once_with(entrypoint[0])
    teardown.assert_called_once_with()


def test_pipeline_worker_failure_still_tears_down(monkeypatch, cli, entrypoint):
    cli.cmd = SubCmd.train
    cli.overrides.append("--recipe.phase=bridge")
    monkeypatch.setenv(pipeline._WORKER_ENV, "1")
    monkeypatch.setattr(SubCmd, "prepare_environment", Mock())
    monkeypatch.setattr(pipeline, "_train_stage", Mock(side_effect=RuntimeError("failed eval")))
    teardown = Mock()
    monkeypatch.setattr(pipeline, "teardown_training_environment", teardown)
    with pytest.raises(RuntimeError, match="failed eval"):
        pipeline.main()
    teardown.assert_called_once_with()


def test_torchrun_uses_allocation_topology_and_separate_stage_rendezvous(monkeypatch, cli):
    monkeypatch.setenv("BEAKER_REPLICA_COUNT", "2")
    monkeypatch.setenv("BEAKER_REPLICA_RANK", "1")
    monkeypatch.setenv("BEAKER_ASSIGNED_GPU_COUNT", "8")
    monkeypatch.setenv("BEAKER_LEADER_REPLICA_HOSTNAME", "leader")
    monkeypatch.setenv("BEAKER_EXPERIMENT_ID", "experiment-id")
    monkeypatch.setenv("MASTER_PORT", "29500")
    commands = [pipeline._torchrun_command(cli, index) for index in range(3)]
    for index, command in enumerate(commands):
        assert command[1:3] == ["-m", "torch.distributed.run"]
        assert "--nnodes=2" in command
        assert "--nproc-per-node=8" in command
        assert "--node-rank=1" in command
        assert f"--rdzv-endpoint=leader:{29500 + index}" in command
        assert f"--rdzv-id=experiment-id-{index}" in command
        assert command[-len(cli.overrides) :] == cli.overrides


def test_torchrun_rejects_missing_multinode_leader(monkeypatch, cli):
    monkeypatch.setenv("BEAKER_REPLICA_COUNT", "2")
    monkeypatch.delenv("BEAKER_LEADER_REPLICA_HOSTNAME", raising=False)
    with pytest.raises(OLMoConfigurationError, match="leader hostname"):
        pipeline._torchrun_command(cli, 0)


@pytest.mark.parametrize("returncode", [0, 2])
def test_child_exit_status_propagates(monkeypatch, returncode):
    process = Mock()
    process.wait.return_value = returncode
    process.__enter__ = Mock(return_value=process)
    process.__exit__ = Mock(return_value=False)
    popen = Mock(return_value=process)
    monkeypatch.setattr(pipeline.subprocess, "Popen", popen)
    if returncode:
        with pytest.raises(subprocess.CalledProcessError):
            pipeline._run_command(["worker"])
    else:
        pipeline._run_command(["worker"])
    assert popen.call_args.kwargs["env"][pipeline._WORKER_ENV] == "1"
    assert popen.call_args.kwargs["start_new_session"] is True


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_signal_forwards_to_workers_and_stops_chain_even_after_zero_exit(monkeypatch, signum):
    original_handler = signal.getsignal(signum)
    process = Mock(pid=1234)
    process.poll.return_value = None

    def wait():
        signal.getsignal(signum)(signum, None)
        return 0

    process.wait.side_effect = wait
    process.__enter__ = Mock(return_value=process)
    process.__exit__ = Mock(return_value=False)
    kill = Mock()
    monkeypatch.setattr(pipeline.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(pipeline.os, "killpg", kill)
    with pytest.raises(SystemExit) as caught:
        pipeline._run_command(["worker"])
    assert caught.value.code == 128 + signum
    kill.assert_called_once_with(1234, signum)
    assert signal.getsignal(signum) is original_handler


def test_standard_experiment_parser_keeps_cli_contract(monkeypatch):
    monkeypatch.setattr(
        experiment.sys, "argv", ["train.py", "dry_run", "run", "local", "--init_seed=1"]
    )
    assert experiment.parse_cli_args() == CliContext(
        "train.py", SubCmd.dry_run, "run", "local", ["--init_seed=1"]
    )
