import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal import vision_alignment_pipeline as pipeline
from olmo_core.internal.experiment import CliContext, SubCmd
from olmo_core.internal.vision_alignment import AlignmentPhase
from olmo_core.train import LoadStrategy
from olmo_core.train.checkpoint import Checkpointer
from olmo_core.train.common import Duration


@pytest.fixture
def cli():
    return CliContext(
        script="src/scripts/train/Vision-Align.py",
        cmd=SubCmd.train,
        run_name="alignment",
        cluster="local",
        overrides=[],
    )


@pytest.fixture
def config_factory(tmp_path):
    def build(phase=AlignmentPhase.bridge, *, steps=10, parent=None):
        config = SimpleNamespace(
            recipe=SimpleNamespace(phase=phase, parent_checkpoint=parent),
            pretraining_checkpoint=f"{tmp_path}/pretraining/step100",
            trainer=SimpleNamespace(
                save_folder=f"{tmp_path}/alignment-{phase}",
                max_duration=Duration.steps(steps),
                hard_stop=None,
                no_checkpoints=False,
                load_strategy=LoadStrategy.if_available,
                load_optim_state=False,
                load_trainer_state=False,
                callbacks={
                    "checkpointer": SimpleNamespace(enabled=True),
                    "multimodal_evaluator": SimpleNamespace(
                        eval_on_startup=True, eval_on_finish=True
                    ),
                },
            ),
        )
        config.as_config_dict = lambda: _checkpoint_config(config)
        return config

    return build


def _checkpoint_config(config):
    return {
        "recipe": {
            "phase": config.recipe.phase,
            "parent_checkpoint": config.recipe.parent_checkpoint,
        },
        "pretraining_checkpoint": config.pretraining_checkpoint,
        "trainer": {
            "max_duration": {"value": config.trainer.max_duration.value, "unit": "steps"},
            "callbacks": {
                "checkpointer": {"enabled": config.trainer.callbacks["checkpointer"].enabled},
                "multimodal_evaluator": {
                    "eval_on_startup": config.trainer.callbacks[
                        "multimodal_evaluator"
                    ].eval_on_startup,
                    "eval_on_finish": config.trainer.callbacks[
                        "multimodal_evaluator"
                    ].eval_on_finish,
                },
            },
        },
    }


def _write_checkpoint(config, *, complete=False):
    checkpoint = Path(pipeline._endpoint(config))
    (checkpoint / "model_and_optim").mkdir(parents=True, exist_ok=True)
    (checkpoint / "train").mkdir(exist_ok=True)
    (checkpoint / Checkpointer.METADATA_FNAME).write_text("{}")
    (checkpoint / "model_and_optim/.metadata").write_bytes(b"metadata")
    torch.save(
        {
            "global_step": config.trainer.max_duration.value,
            "max_steps": config.trainer.max_duration.value,
        },
        checkpoint / "train/rank0.pt",
    )
    (checkpoint / "config.json").write_text(json.dumps(_checkpoint_config(config)))
    if complete:
        (checkpoint.parent / pipeline._SUCCESS_FILE).write_text(
            json.dumps(pipeline._completion(config, str(checkpoint)))
        )
    return checkpoint


@pytest.mark.parametrize("phase", ["all", "bridge", "perception", "joint"])
def test_selected_phase(cli, phase):
    cli.overrides = [f"--recipe.phase={phase}"]
    assert pipeline._selected_phase(cli) == phase


def test_selected_phase_defaults_to_all_and_uses_last_override(cli):
    assert pipeline._selected_phase(cli) == "all"
    cli.overrides = ["--recipe.phase=invalid", "--recipe.phase=joint"]
    assert pipeline._selected_phase(cli) == "joint"


@pytest.mark.parametrize("phase", ["", "invalid", "Bridge"])
def test_selected_phase_rejects_unknown_values(cli, phase):
    cli.overrides = [f"--recipe.phase={phase}"]
    with pytest.raises(OLMoConfigurationError, match="Unknown alignment phase"):
        pipeline._selected_phase(cli)


@pytest.mark.parametrize("parent", [None, "/checkpoints/bridge/step500"])
def test_stage_cli_preserves_shared_overrides_without_mutation(cli, parent):
    original = [
        "--recipe.phase=all",
        "--recipe.output_root=/checkpoints",
        "--train_module.rank_microbatch_size=2",
        "--recipe.phase=bridge",
    ]
    cli.overrides = original.copy()
    stage = pipeline._stage_cli(cli, AlignmentPhase.perception, parent)

    assert stage is not cli
    assert stage.script == cli.script
    assert stage.cmd == cli.cmd
    assert stage.cluster == cli.cluster
    assert stage.run_name == "alignment-perception"
    assert stage.overrides == [
        "--recipe.output_root=/checkpoints",
        "--train_module.rank_microbatch_size=2",
        "--recipe.phase=perception",
        *([] if parent is None else [f"--recipe.parent_checkpoint={parent}"]),
    ]
    assert cli.overrides == original


def test_endpoint_normalizes_save_folder(config_factory):
    config = config_factory(steps=500)
    config.trainer.save_folder = "file:///checkpoints/alignment-bridge///"
    assert pipeline._endpoint(config) == "/checkpoints/alignment-bridge/step500"


@pytest.mark.parametrize(
    "duration", [Duration.steps(0), Duration.steps(-1), Duration.tokens(10), Duration.epochs(1)]
)
def test_endpoint_requires_positive_step_duration(config_factory, duration):
    config = config_factory()
    config.trainer.max_duration = duration
    with pytest.raises(OLMoConfigurationError, match="positive step duration"):
        pipeline._endpoint(config)


@pytest.mark.parametrize("disabled", ["trainer", "callback"])
def test_endpoint_requires_checkpoint_saving(config_factory, disabled):
    config = config_factory()
    if disabled == "trainer":
        config.trainer.no_checkpoints = True
    else:
        config.trainer.callbacks["checkpointer"].enabled = False
    with pytest.raises(OLMoConfigurationError, match="requires checkpoint saving"):
        pipeline._endpoint(config)


def test_endpoint_requires_checkpoint_loading(config_factory):
    config = config_factory()
    config.trainer.load_strategy = LoadStrategy.never
    with pytest.raises(OLMoConfigurationError, match="requires checkpoint loading"):
        pipeline._endpoint(config)


@pytest.mark.parametrize("phase", [AlignmentPhase.perception, AlignmentPhase.joint])
@pytest.mark.parametrize("field", ["load_optim_state", "load_trainer_state"])
@pytest.mark.parametrize("value", [True, None])
def test_endpoint_requires_model_only_stage_handoffs(config_factory, phase, field, value):
    config = config_factory(phase)
    setattr(config.trainer, field, value)
    with pytest.raises(OLMoConfigurationError, match="handoffs must load only model weights"):
        pipeline._endpoint(config)


def test_stage_complete_requires_success_marker(config_factory):
    config = config_factory()
    checkpoint = _write_checkpoint(config)
    assert pipeline._finalized(str(checkpoint))
    assert not pipeline._stage_complete(config, str(checkpoint))


def test_stage_complete_accepts_matching_finalized_checkpoint(config_factory):
    config = config_factory(AlignmentPhase.perception, parent="/bridge/step500")
    checkpoint = _write_checkpoint(config, complete=True)
    assert pipeline._stage_complete(config, str(checkpoint))


@pytest.mark.parametrize(
    "missing",
    [Checkpointer.METADATA_FNAME, "model_and_optim/.metadata", "train/rank0.pt"],
)
def test_stage_complete_rejects_incomplete_checkpoint(config_factory, missing):
    config = config_factory()
    checkpoint = _write_checkpoint(config, complete=True)
    (checkpoint / missing).unlink()
    assert not pipeline._finalized(str(checkpoint))
    assert not pipeline._stage_complete(config, str(checkpoint))


def test_stage_complete_requires_checkpoint_config(config_factory):
    config = config_factory()
    checkpoint = _write_checkpoint(config, complete=True)
    (checkpoint / "config.json").unlink()
    assert pipeline._finalized(str(checkpoint))
    with pytest.raises(FileNotFoundError):
        pipeline._stage_complete(config, str(checkpoint))


@pytest.mark.parametrize(
    ("keys", "value"),
    [
        (("recipe", "phase"), "joint"),
        (("recipe", "parent_checkpoint"), "/other-parent/step500"),
        (("pretraining_checkpoint",), "/other-pretraining/step100"),
        (("trainer", "max_duration", "unit"), "tokens"),
        (("trainer", "max_duration", "value"), 20),
    ],
)
@pytest.mark.parametrize("complete", [False, True])
def test_endpoint_metadata_mismatch_blocks_resume_and_completed_skip(
    config_factory, monkeypatch, keys, value, complete
):
    config = config_factory()
    checkpoint = _write_checkpoint(config, complete=complete)
    saved_config = _checkpoint_config(config)
    target = saved_config
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    (checkpoint / "config.json").write_text(json.dumps(saved_config))
    train = Mock()
    monkeypatch.setattr(pipeline, "train", train)

    with pytest.raises(OLMoConfigurationError, match="does not match the alignment stage"):
        if complete:
            pipeline._stage_complete(config, str(checkpoint))
        else:
            pipeline._train_stage(config)
    train.assert_not_called()
    assert config.trainer.callbacks["checkpointer"].enabled


@pytest.mark.parametrize("field", ["global_step", "max_steps"])
@pytest.mark.parametrize("step", [9, 11, None])
@pytest.mark.parametrize("complete", [False, True])
def test_endpoint_state_mismatch_blocks_resume_and_completed_skip(
    config_factory, monkeypatch, field, step, complete
):
    config = config_factory()
    checkpoint = _write_checkpoint(config, complete=complete)
    state = {"global_step": 10, "max_steps": 10}
    if step is None:
        state.pop(field)
    else:
        state[field] = step
    torch.save(state, checkpoint / "train/rank0.pt")
    train = Mock()
    monkeypatch.setattr(pipeline, "train", train)

    with pytest.raises(OLMoConfigurationError, match="did not reach the stage horizon"):
        if complete:
            pipeline._stage_complete(config, str(checkpoint))
        else:
            pipeline._train_stage(config)
    train.assert_not_called()
    assert config.trainer.callbacks["checkpointer"].enabled


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("phase", "joint"),
        ("step", 20),
        ("checkpoint", "/other/step10"),
        ("parent_checkpoint", "/other-parent/step500"),
        ("pretraining_checkpoint", "/other-pretraining/step100"),
    ],
)
def test_stage_complete_rejects_mismatched_completion(config_factory, field, value):
    config = config_factory()
    checkpoint = _write_checkpoint(config, complete=True)
    marker = checkpoint.parent / pipeline._SUCCESS_FILE
    completion = json.loads(marker.read_text())
    completion[field] = value
    marker.write_text(json.dumps(completion))
    assert not pipeline._stage_complete(config, str(checkpoint))


def test_stage_complete_rejects_malformed_completion(config_factory):
    config = config_factory()
    checkpoint = _write_checkpoint(config, complete=True)
    (checkpoint.parent / pipeline._SUCCESS_FILE).write_text("{")
    with pytest.raises(json.JSONDecodeError):
        pipeline._stage_complete(config, str(checkpoint))


def test_model_only_checkpoint_is_not_finalized(tmp_path):
    (tmp_path / ".metadata").write_bytes(b"model-only metadata")
    assert not pipeline._finalized(str(tmp_path))


def _overwrite_through_trainer(trainer, name, contents, dir=None):
    """Stand-in for the overwrite helper: the tests observe the trainer's writer."""
    kwargs = {"save_overwrite": True}
    if dir is not None:
        kwargs["dir"] = dir
    return trainer.write_file(name, contents, **kwargs)


@pytest.fixture
def stage_runner(monkeypatch, config_factory):
    config = config_factory()
    events = []
    trainer = SimpleNamespace(is_canceled=False, global_step=10, write_file=Mock())
    train = Mock(side_effect=lambda _: (events.append("fit"), trainer)[1])
    finalized = Mock(side_effect=[False, True])
    monkeypatch.setattr(pipeline, "train", train)
    monkeypatch.setattr(pipeline, "_finalized", finalized)
    monkeypatch.setattr(pipeline, "_validate_endpoint", Mock())
    monkeypatch.setattr(pipeline, "file_exists", Mock(return_value=True))
    monkeypatch.setattr(pipeline, "barrier", lambda: events.append("barrier"))
    monkeypatch.setattr(pipeline, "get_rank", lambda: 0)
    monkeypatch.setattr(pipeline, "write_file_overwrite", _overwrite_through_trainer)
    return SimpleNamespace(
        config=config, trainer=trainer, train=train, finalized=finalized, events=events
    )


def test_train_stage_marks_success_only_after_fit_and_checkpoint_finalization(stage_runner):
    runner = stage_runner

    def finalized(checkpoint):
        runner.events.append("finalized")
        assert checkpoint == pipeline._endpoint(runner.config)
        return "fit" in runner.events

    def write_file(name, contents, **kwargs):
        assert runner.events == ["finalized", "fit", "finalized", "barrier"]
        assert name == pipeline._SUCCESS_FILE
        assert json.loads(contents) == pipeline._completion(
            runner.config, pipeline._endpoint(runner.config)
        )
        assert kwargs == {"save_overwrite": True}
        runner.events.append("marker")

    runner.finalized.side_effect = finalized
    runner.trainer.write_file.side_effect = write_file
    pipeline._train_stage(runner.config)

    runner.train.assert_called_once_with(runner.config)
    runner.trainer.write_file.assert_called_once()
    assert runner.events == ["finalized", "fit", "finalized", "barrier", "marker", "barrier"]


def test_train_stage_only_rank_zero_writes_completion(stage_runner, monkeypatch):
    monkeypatch.setattr(pipeline, "get_rank", lambda: 1)
    pipeline._train_stage(stage_runner.config)
    stage_runner.trainer.write_file.assert_not_called()
    assert stage_runner.events == ["fit", "barrier", "barrier"]


@pytest.mark.parametrize("step", [0, 9, 11])
def test_train_stage_requires_exact_endpoint(stage_runner, step):
    stage_runner.trainer.global_step = step
    with pytest.raises(RuntimeError, match=f"stopped at step {step}"):
        pipeline._train_stage(stage_runner.config)
    stage_runner.trainer.write_file.assert_not_called()


def test_train_stage_does_not_mark_canceled_endpoint_complete(stage_runner):
    stage_runner.trainer.is_canceled = True
    with pytest.raises(RuntimeError, match="stopped at step 10"):
        pipeline._train_stage(stage_runner.config)
    stage_runner.trainer.write_file.assert_not_called()


def test_train_stage_does_not_mark_hard_stop_complete(stage_runner):
    stage_runner.config.trainer.hard_stop = Duration.steps(5)
    stage_runner.trainer.global_step = 5
    with pytest.raises(RuntimeError, match="stopped at step 5"):
        pipeline._train_stage(stage_runner.config)
    stage_runner.trainer.write_file.assert_not_called()


def test_train_stage_propagates_fit_failures(stage_runner):
    stage_runner.train.side_effect = RuntimeError("fit failed")
    with pytest.raises(RuntimeError, match="fit failed"):
        pipeline._train_stage(stage_runner.config)
    stage_runner.trainer.write_file.assert_not_called()


def test_train_stage_requires_finalized_endpoint_after_fit(stage_runner):
    stage_runner.finalized.side_effect = [False, False]
    with pytest.raises(RuntimeError, match="did not finalize its endpoint checkpoint"):
        pipeline._train_stage(stage_runner.config)
    stage_runner.trainer.write_file.assert_not_called()


def test_train_stage_propagates_completion_write_failure(stage_runner):
    stage_runner.trainer.write_file.side_effect = OSError("write failed")
    with pytest.raises(OSError, match="write failed"):
        pipeline._train_stage(stage_runner.config)


@pytest.mark.parametrize("missing_config", [False, True])
def test_train_stage_recovers_final_checkpoint_without_rewriting_model_state(
    config_factory, monkeypatch, missing_config
):
    config = config_factory()
    checkpoint = _write_checkpoint(config)
    config_path = checkpoint / "config.json"
    expected_config = json.loads(config_path.read_text())
    if missing_config:
        config_path.unlink()
    initial_files = {path: path.read_bytes() for path in checkpoint.rglob("*") if path.is_file()}
    writes = []
    fitted = False

    def write_file(name, contents, *, dir=None, save_overwrite=False):
        assert fitted
        output_dir = Path(dir or config.trainer.save_folder)
        if name == "config.json":
            assert dir == str(checkpoint)
            assert not save_overwrite
            assert not config_path.exists()
            assert json.loads(contents) == expected_config
        else:
            assert name == pipeline._SUCCESS_FILE
            assert save_overwrite
            assert config_path.is_file()
        (output_dir / name).write_text(contents)
        writes.append(name)

    trainer = SimpleNamespace(is_canceled=False, global_step=10, write_file=write_file)
    monkeypatch.setattr(pipeline, "write_file_overwrite", _overwrite_through_trainer)

    def train(resumed_config):
        nonlocal fitted
        assert resumed_config is config
        assert not resumed_config.trainer.callbacks["checkpointer"].enabled
        assert not resumed_config.trainer.callbacks["multimodal_evaluator"].eval_on_startup
        assert resumed_config.trainer.callbacks["multimodal_evaluator"].eval_on_finish
        assert config_path.exists() is not missing_config
        assert not (checkpoint.parent / pipeline._SUCCESS_FILE).exists()
        fitted = True
        return trainer

    monkeypatch.setattr(pipeline, "train", Mock(side_effect=train))
    monkeypatch.setattr(pipeline, "barrier", Mock())
    monkeypatch.setattr(pipeline, "get_rank", lambda: 0)
    pipeline._train_stage(config)

    assert writes == [*([] if not missing_config else ["config.json"]), pipeline._SUCCESS_FILE]
    assert {path: path.read_bytes() for path in initial_files} == initial_files
    assert pipeline._stage_complete(config, str(checkpoint))


def test_missing_config_is_not_repaired_when_final_evaluation_fails(config_factory, monkeypatch):
    config = config_factory()
    checkpoint = _write_checkpoint(config)
    (checkpoint / "config.json").unlink()
    monkeypatch.setattr(pipeline, "train", Mock(side_effect=RuntimeError("evaluation failed")))

    with pytest.raises(RuntimeError, match="evaluation failed"):
        pipeline._train_stage(config)
    assert not (checkpoint / "config.json").exists()
    assert not (checkpoint.parent / pipeline._SUCCESS_FILE).exists()


def test_missing_config_does_not_allow_incomplete_state_to_resume(config_factory, monkeypatch):
    config = config_factory()
    checkpoint = _write_checkpoint(config)
    (checkpoint / "config.json").unlink()
    torch.save({"global_step": 9, "max_steps": 10}, checkpoint / "train/rank0.pt")
    train = Mock()
    monkeypatch.setattr(pipeline, "train", train)

    with pytest.raises(OLMoConfigurationError, match="did not reach the stage horizon"):
        pipeline._train_stage(config)
    train.assert_not_called()
    assert not (checkpoint / "config.json").exists()


@pytest.fixture
def pipeline_runner(monkeypatch, cli, config_factory):
    monkeypatch.delenv("RANK", raising=False)
    configs = {
        phase: config_factory(phase, steps=steps)
        for phase, steps in zip(pipeline._STAGES, [5, 20, 15])
    }
    built = []
    launched = []
    completed = set()

    def build(stage_cli):
        phase = pipeline._selected_phase(stage_cli)
        config = configs[phase]
        config.recipe.parent_checkpoint = next(
            (
                arg.partition("=")[2]
                for arg in stage_cli.overrides
                if arg.startswith("--recipe.parent_checkpoint=")
            ),
            None,
        )
        built.append(stage_cli)
        return config

    def command(stage_cli, index):
        return [pipeline._selected_phase(stage_cli), str(index)]

    def run(command):
        launched.append(command)
        completed.add(command[0])

    monkeypatch.setattr(pipeline, "build_config", Mock(side_effect=build))
    monkeypatch.setattr(pipeline, "_torchrun_command", Mock(side_effect=command))
    monkeypatch.setattr(pipeline, "_run_command", Mock(side_effect=run))
    monkeypatch.setattr(
        pipeline,
        "_stage_complete",
        Mock(side_effect=lambda config, _: config.recipe.phase in completed),
    )
    return SimpleNamespace(
        cli=cli, configs=configs, built=built, launched=launched, completed=completed
    )


def test_pipeline_trains_all_stages_in_order_with_exact_parent_endpoints(pipeline_runner):
    runner = pipeline_runner
    pipeline._run_pipeline(runner.cli, runner.configs[AlignmentPhase.bridge])

    assert runner.launched == [["bridge", "0"], ["perception", "1"], ["joint", "2"]]
    assert [cli.run_name for cli in runner.built] == ["alignment-perception", "alignment-joint"]
    assert runner.configs[AlignmentPhase.bridge].recipe.parent_checkpoint is None
    assert runner.configs[AlignmentPhase.perception].recipe.parent_checkpoint == pipeline._endpoint(
        runner.configs[AlignmentPhase.bridge]
    )
    assert runner.configs[AlignmentPhase.joint].recipe.parent_checkpoint == pipeline._endpoint(
        runner.configs[AlignmentPhase.perception]
    )


@pytest.mark.parametrize("completed", [1, 2, 3])
def test_pipeline_skips_completed_stages_without_starting_workers(pipeline_runner, completed):
    runner = pipeline_runner
    runner.completed.update(pipeline._STAGES[:completed])
    pipeline._run_pipeline(runner.cli, runner.configs[AlignmentPhase.bridge])
    assert runner.launched == [
        [phase, str(index)] for index, phase in enumerate(pipeline._STAGES) if index >= completed
    ]
    assert runner.configs[AlignmentPhase.joint].recipe.parent_checkpoint == pipeline._endpoint(
        runner.configs[AlignmentPhase.perception]
    )


@pytest.mark.parametrize("failed_phase", pipeline._STAGES)
@pytest.mark.parametrize("raised", [False, True])
def test_pipeline_stops_after_failed_or_incomplete_stage(pipeline_runner, failed_phase, raised):
    runner = pipeline_runner

    def run(command):
        runner.launched.append(command)
        if command[0] == failed_phase:
            if raised:
                raise subprocess.CalledProcessError(1, command)
        else:
            runner.completed.add(command[0])

    pipeline._run_command.side_effect = run
    error = subprocess.CalledProcessError if raised else RuntimeError
    with pytest.raises(error):
        pipeline._run_pipeline(runner.cli, runner.configs[AlignmentPhase.bridge])
    failed_index = pipeline._STAGES.index(failed_phase)
    assert runner.launched == [
        [phase, str(index)] for index, phase in enumerate(pipeline._STAGES) if index <= failed_index
    ]
    assert len(runner.built) == failed_index
