"""Tests for UnshardedModelExportCallback."""

import json
from pathlib import Path
from unittest.mock import Mock

import torch
import torch.nn as nn
from safetensors.torch import load_file

from olmo_core.distributed.checkpoint import save_model_and_optim_state
from olmo_core.train.callbacks.unsharded_export import UnshardedModelExportCallback
from olmo_core.train.checkpoint import Checkpointer, CheckpointMetadata


def _make_checkpoint(path: Path, *, ephemeral: bool = False, config: bool = True) -> nn.Module:
    """A checkpoint directory laid out as the trainer writes one."""
    path.mkdir()
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    save_model_and_optim_state(
        path / "model_and_optim", model, torch.optim.AdamW(model.parameters())
    )
    metadata = CheckpointMetadata(ephemeral=ephemeral).as_dict(json_safe=True)
    (path / Checkpointer.METADATA_FNAME).write_text(json.dumps(metadata))
    if config:
        (path / "config.json").write_text(json.dumps({"model": {"d_model": 8}}))
    return model


def _callback(**kwargs) -> UnshardedModelExportCallback:
    trainer = Mock()
    trainer.checkpointer.write_file.side_effect = lambda dir, fname, contents: (
        Path(dir) / fname
    ).write_text(contents)
    callback = UnshardedModelExportCallback(**kwargs)
    callback.trainer = trainer
    return callback


def test_export_writes_unsharded_weights_and_config(tmp_path: Path):
    checkpoint = tmp_path / "step10"
    model = _make_checkpoint(checkpoint)

    _callback().export(str(checkpoint))

    weights = load_file(checkpoint / "model.safetensors")
    expected = model.state_dict()
    assert weights.keys() == expected.keys()
    for key, value in expected.items():
        torch.testing.assert_close(weights[key], value)
    config = (checkpoint / "olmo_core_config.json").read_text()
    assert config == (checkpoint / "config.json").read_text()
    # No temporary file is left behind.
    assert [p.name for p in checkpoint.glob("*.safetensors")] == ["model.safetensors"]


def test_ephemeral_checkpoint_is_not_exported(tmp_path: Path):
    checkpoint = tmp_path / "step10"
    _make_checkpoint(checkpoint, ephemeral=True)

    _callback().export(str(checkpoint))

    assert not (checkpoint / "model.safetensors").exists()
    assert not (checkpoint / "olmo_core_config.json").exists()


def test_checkpoint_without_config_is_not_exported(tmp_path: Path):
    checkpoint = tmp_path / "step10"
    _make_checkpoint(checkpoint, config=False)

    _callback().export(str(checkpoint))

    assert not (checkpoint / "model.safetensors").exists()


def test_saved_checkpoints_are_exported_as_background_ops(tmp_path: Path):
    callback = _callback()
    path = str(tmp_path / "step10")

    # Only queued on the saving thread; submitted from the main thread's next hook.
    callback.post_checkpoint_saved(path)
    callback.trainer.run_bookkeeping_op.assert_not_called()
    callback.post_step()
    callback.trainer.run_bookkeeping_op.assert_called_once_with(
        callback.export, path, op_name="unsharded_model_export", distributed=False
    )

    # The final checkpoint is saved in the checkpointer's 'post_train' and picked up here.
    callback.post_checkpoint_saved(str(tmp_path / "step20"))
    callback.post_train()
    assert callback.trainer.run_bookkeeping_op.call_count == 2
    callback.post_step()
    assert callback.trainer.run_bookkeeping_op.call_count == 2


def test_disabled_callback_exports_nothing(tmp_path: Path):
    callback = _callback(enabled=False)
    callback.post_checkpoint_saved(str(tmp_path / "step10"))
    callback.post_step()
    callback.post_train()
    callback.trainer.run_bookkeeping_op.assert_not_called()
