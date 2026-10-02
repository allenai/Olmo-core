"""
Tests for the bookkeeping callbacks in :mod:`olmo_core.train.callbacks.multimodal`: the
overwriting metric saver, the same-step metric restore, the W&B auto-resume subclass and the
overwrite-capable file writer they share. The shared callbacks they subclass are unchanged.
"""

import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from olmo_core.train.callbacks.checkpointer import CheckpointRemovalStrategy
from olmo_core.train.callbacks.evaluator_callback import EvaluatorCallback
from olmo_core.train.callbacks.multimodal import (
    MultimodalBeakerCallback,
    MultimodalCheckpointerCallback,
    MultimodalEvaluatorCallback,
    MultimodalMetricSaverCallback,
    MultimodalWandBCallback,
    RestoreMetricsCallback,
    write_file_overwrite,
)
from olmo_core.train.callbacks.wandb import WANDB_API_KEY_ENV_VAR
from olmo_core.train.checkpoint import Checkpointer
from olmo_core.train.trainer import Trainer


@pytest.fixture
def metrics_trainer(tmp_path, monkeypatch):
    monkeypatch.setattr("olmo_core.train.callbacks.metric_saver.get_rank", lambda: 0)
    trainer = SimpleNamespace(
        save_folder=tmp_path / "checkpoints",
        checkpointer=Checkpointer(work_dir=tmp_path / "work", save_overwrite=False),
    )
    trainer.write_file = partial(Trainer.write_file, trainer)
    return trainer


def test_write_file_overwrite_replaces_local_files_but_not_checkpoint_policy(metrics_trainer):
    first = write_file_overwrite(metrics_trainer, "notes/marker.json", '{"a": 1}')
    second = write_file_overwrite(metrics_trainer, "notes/marker.json", '{"a": 2}')
    assert Path(first) == Path(second) == metrics_trainer.save_folder / "notes/marker.json"
    assert json.loads(Path(second).read_text()) == {"a": 2}
    # The trainer's own writer still refuses to overwrite.
    metrics_trainer.write_file("step10/marker", "checkpoint")
    with pytest.raises(FileExistsError):
        metrics_trainer.write_file("step10/marker", "replacement")
    assert metrics_trainer.checkpointer.save_overwrite is False


def test_resumed_metrics_replace_snapshots_without_overwriting_checkpoints(metrics_trainer):
    old = MultimodalMetricSaverCallback(save_interval=100)
    old.trainer = metrics_trainer
    old.log_metrics(100, {"eval/caption/CE loss": 4.0})
    old.close()
    metrics_trainer.write_file("step100/marker", "checkpoint")

    resumed = MultimodalMetricSaverCallback(save_interval=100)
    resumed.trainer = metrics_trainer
    resumed.log_metrics(100, {"eval/caption/CE loss": 2.0})
    resumed.log_metrics(100, {"eval/caption/CE gap": 0.2})
    resumed.close()
    expected = {"eval/caption/CE loss": 2.0, "eval/caption/CE gap": 0.2}
    for filename in ("metrics_step100.json", "metrics.json"):
        assert json.loads((metrics_trainer.save_folder / filename).read_text()) == expected
    with pytest.raises(FileExistsError):
        metrics_trainer.write_file("step100/marker", "replacement")
    assert (metrics_trainer.save_folder / "step100/marker").read_text() == "checkpoint"


@pytest.mark.parametrize("scheme", [None, "s3", "gs"])
def test_resume_restores_same_step_metrics_before_evaluation(metrics_trainer, monkeypatch, scheme):
    old = MultimodalMetricSaverCallback(save_interval=100)
    old.trainer = metrics_trainer
    old.log_metrics(100, {"train/CE loss": 3.0, "eval/caption/CE loss": 4.0})
    snapshot = metrics_trainer.save_folder / "metrics_step100.json"
    if scheme is not None:
        metrics_trainer.save_folder = f"{scheme}://bucket/checkpoints"
        exists = Mock(return_value=True)
        resource = Mock(return_value=snapshot)
        monkeypatch.setattr("olmo_core.train.callbacks.multimodal.file_exists", exists)
        monkeypatch.setattr("olmo_core.train.callbacks.multimodal.resource_path", resource)
    writes = []
    monkeypatch.setattr(
        "olmo_core.train.callbacks.multimodal.write_file_overwrite",
        lambda trainer, name, contents, dir=None: writes.append((name, json.loads(contents))),
    )
    metrics_trainer.global_step = 100
    metrics_trainer.checkpoint_loaded = True
    resumed = MultimodalMetricSaverCallback(save_interval=100)
    resumed.trainer = metrics_trainer
    metrics_trainer.callbacks = {"metrics": resumed}
    restore = RestoreMetricsCallback(metrics_callback="metrics")
    restore.trainer = metrics_trainer
    restore.pre_train()
    if scheme is not None:
        exists.assert_called_once_with(f"{scheme}://bucket/checkpoints/metrics_step100.json")
        resource.assert_called_once_with(f"{scheme}://bucket/checkpoints", "metrics_step100.json")
    resumed.log_metrics(100, {"eval/caption/CE loss": 2.0})
    assert resumed.metrics == {"train/CE loss": 3.0, "eval/caption/CE loss": 2.0}
    assert writes[-1] == ("metrics_step100.json", resumed.metrics)


class _MockWandB:
    def __init__(self, run_id: str = "run-123"):
        self.run = SimpleNamespace(id=run_id, path=f"entity/project/{run_id}")
        self.init_calls = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self.run


def _wandb_callback(tmp_path, wandb, *, step: int = 25) -> MultimodalWandBCallback:
    callback = MultimodalWandBCallback(
        name="bridge", project="vision-alignment", entity="test-entity", auto_resume=True
    )
    callback._wandb = wandb
    callback.trainer = SimpleNamespace(work_dir=tmp_path, global_step=step, checkpoint_loaded=False)
    return callback


def test_wandb_auto_resume_continues_existing_run(tmp_path, monkeypatch):
    monkeypatch.setenv(WANDB_API_KEY_ENV_VAR, "test-key")
    wandb = _MockWandB()
    callback = _wandb_callback(tmp_path, wandb, step=25)
    callback.load_state_dict(
        {
            "run_id": "run-123",
            "step": 20,
            "name": "bridge",
            "project": "vision-alignment",
            "entity": "test-entity",
        }
    )
    callback.pre_train()
    init = wandb.init_calls[0]
    assert init["id"] == "run-123"
    assert init["resume"] == "allow"
    assert init["allow_val_change"] is True


def test_wandb_auto_resume_rejects_different_run_identity(tmp_path, monkeypatch):
    monkeypatch.setenv(WANDB_API_KEY_ENV_VAR, "test-key")
    wandb = _MockWandB(run_id="new-run")
    callback = _wandb_callback(tmp_path, wandb)
    callback.load_state_dict(
        {
            "run_id": "old-run",
            "step": 20,
            "name": "different-name",
            "project": "vision-alignment",
            "entity": "test-entity",
        }
    )
    callback.pre_train()
    assert "id" not in wandb.init_calls[0]
    assert callback.run_id == "new-run"


def _evaluator(monkeypatch, *, step: int, **kwargs) -> MultimodalEvaluatorCallback:
    callback = MultimodalEvaluatorCallback(evaluators=[], **kwargs)
    callback.trainer = SimpleNamespace(global_step=step)
    evaluated = []
    monkeypatch.setattr(
        EvaluatorCallback, "perform_eval", lambda self, prefix="eval": evaluated.append(prefix)
    )
    callback.evaluated = evaluated  # type: ignore[attr-defined]
    return callback


def test_eval_on_finish_skips_a_step_that_was_just_evaluated(monkeypatch):
    callback = _evaluator(monkeypatch, step=500, eval_interval=500, eval_on_finish=True)
    callback.post_step()  # interval evaluation at step 500
    callback.post_train()
    assert callback.evaluated == ["eval"]


def test_eval_memory_resets_on_train_start_and_checkpoint_load(monkeypatch):
    callback = _evaluator(monkeypatch, step=500, eval_on_startup=True, eval_on_finish=True)
    callback.perform_eval()
    assert callback._last_eval_step == 500
    callback.pre_train()  # startup evaluation runs and records the step again
    assert callback.evaluated == ["eval", "eval"] and callback._last_eval_step == 500
    callback.post_checkpoint_loaded("/ckpt/step500")
    assert callback._last_eval_step is None
    callback.post_train()
    assert callback.evaluated == ["eval", "eval", "eval"]


def _checkpointer(tmp_path, monkeypatch, *, step: int, existing, **kwargs):
    callback = MultimodalCheckpointerCallback(
        save_async=False, pre_train_checkpoint=False, **kwargs
    )
    checkpointer = SimpleNamespace(
        process_group=None,
        find_checkpoints=lambda folder, ephemeral=None: iter(
            [(s, f"{folder}/step{s}") for s in existing] if ephemeral is False else []
        ),
    )
    callback.trainer = SimpleNamespace(
        global_step=step, checkpoint_loaded=True, checkpointer=checkpointer, save_folder=tmp_path
    )
    removed = []
    monkeypatch.setattr(callback, "_schedule_for_removal", lambda path: removed.append(path))
    monkeypatch.setattr("olmo_core.train.callbacks.checkpointer.is_distributed", lambda: False)
    monkeypatch.setattr("olmo_core.train.callbacks.multimodal.get_rank", lambda *a, **k: 0)
    monkeypatch.setattr("olmo_core.train.callbacks.multimodal.broadcast_object", lambda x, **k: x)
    callback.removed = removed  # type: ignore[attr-defined]
    return callback


def test_resumed_checkpointer_takes_over_permanent_checkpoints_and_trims(tmp_path, monkeypatch):
    callback = _checkpointer(
        tmp_path, monkeypatch, step=1500, existing=[500, 1000, 1500, 2000], max_checkpoints=2
    )
    callback.pre_train()
    # Steps beyond the current one belong to a longer earlier run and are left alone; of the
    # rest only the newest ``max_checkpoints`` survive.
    assert callback._checkpoints == [f"{tmp_path}/step1000", f"{tmp_path}/step1500"]
    assert callback.removed == [f"{tmp_path}/step500"]


def test_resumed_checkpointer_never_removes_with_remove_never(tmp_path, monkeypatch):
    callback = _checkpointer(
        tmp_path,
        monkeypatch,
        step=1500,
        existing=[500, 1000, 1500],
        max_checkpoints=1,
        remove=CheckpointRemovalStrategy.never,
    )
    callback.pre_train()
    assert callback._checkpoints == [] and callback.removed == []


class _ConfigStub:
    """Behaves like a resumed W&B config: a changed value raises unless allow_val_change."""

    def __init__(self, **items):
        self.items = dict(items)
        self.calls = []

    def update(self, values, allow_val_change=None):
        self.calls.append((dict(values), allow_val_change))
        for key, value in values.items():
            if key in self.items and self.items[key] != value and not allow_val_change:
                raise ValueError(f"Attempted to change value of key {key!r}")
            self.items[key] = value


def test_beaker_callback_presets_wandb_config_on_resumed_runs(monkeypatch):
    config = _ConfigStub(
        beaker_experiment_url="https://beaker.org/ex/OLD", beaker_experiment_id="OLD"
    )
    wandb_callback = MultimodalWandBCallback(
        name="bridge", project="vision-alignment", enabled=True
    )
    wandb_callback._wandb = SimpleNamespace(run=SimpleNamespace(config=config))
    callback = MultimodalBeakerCallback(enabled=True, experiment_id="NEW")
    callback.trainer = SimpleNamespace(callbacks={"wandb": wandb_callback})
    monkeypatch.setattr(callback, "_experiment_url", lambda: "https://beaker.org/ex/NEW")
    callback._preset_wandb_config()
    assert config.calls == [
        (
            {"beaker_experiment_url": "https://beaker.org/ex/NEW", "beaker_experiment_id": "NEW"},
            True,
        )
    ]
    # The base callback's plain update (no allow_val_change) is now a no-op instead of an error.
    config.update(
        {"beaker_experiment_url": "https://beaker.org/ex/NEW", "beaker_experiment_id": "NEW"}
    )
