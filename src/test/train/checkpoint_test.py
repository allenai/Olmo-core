import os
import time
import weakref
from concurrent.futures import Future
from typing import cast
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

import olmo_core.train.checkpoint as checkpoint_module
from olmo_core.distributed.utils import barrier, get_rank
from olmo_core.io import dir_is_empty, file_exists, is_url, normalize_path
from olmo_core.testing import run_distributed_test
from olmo_core.train.checkpoint import Checkpointer
from olmo_core.train.train_module import BasicTrainModule, TrainModule


def run_checkpointer(base_dir, work_dir, model_factory):
    dir = f"{normalize_path(base_dir)}/{Checkpointer.checkpoint_dirname(10)}"

    if not is_url(dir):
        os.environ["OLMO_SHARED_FS"] = "1"

    checkpointer = Checkpointer(work_dir=work_dir)
    model = model_factory()
    optim = torch.optim.AdamW(model.parameters())
    train_module = BasicTrainModule(model, optim, 128)

    # Save checkpoint.
    checkpointer.save(dir, train_module, {"rank": get_rank()})
    barrier()

    assert file_exists((f"{dir}/train/rank0.pt"))
    assert file_exists((f"{dir}/train/rank1.pt"))
    assert not dir_is_empty((f"{dir}/model_and_optim"))
    assert checkpointer.dir_is_checkpoint(dir)
    assert list(checkpointer.find_checkpoints(base_dir)) == [(10, dir)]
    assert checkpointer.latest_checkpoint(base_dir) == dir

    # Load checkpoint.
    train_state = checkpointer.load(dir, train_module)
    assert train_state is not None
    assert train_state["rank"] == get_rank()


def test_checkpointer_with_local_dir(tmp_path, tiny_model_factory):
    run_distributed_test(
        run_checkpointer,
        func_args=(tmp_path / "checkpoint", tmp_path / "work_dir", tiny_model_factory),
        start_method="spawn",
    )


def test_checkpointer_with_remote_s3_dir(s3_checkpoint_dir, tmp_path, tiny_model_factory):
    from botocore.exceptions import NoCredentialsError

    try:
        dir_is_empty(s3_checkpoint_dir)
    except NoCredentialsError:
        pytest.skip("Requires AWS credentials")

    run_distributed_test(
        run_checkpointer,
        func_args=(s3_checkpoint_dir, tmp_path / "work_dir", tiny_model_factory),
        start_method="spawn",
    )


def test_checkpointer_with_remote_gcs_dir(gcs_checkpoint_dir, tmp_path, tiny_model_factory):
    from google.auth.exceptions import DefaultCredentialsError

    try:
        dir_is_empty(gcs_checkpoint_dir)
    except DefaultCredentialsError:
        pytest.skip("Requires authentication with Google Cloud")

    run_distributed_test(
        run_checkpointer,
        func_args=(gcs_checkpoint_dir, tmp_path / "work_dir", tiny_model_factory),
        start_method="spawn",
    )


def run_async_checkpointer(dir, work_dir, model_factory):
    dir = normalize_path(dir)

    if not is_url(dir):
        os.environ["OLMO_SHARED_FS"] = "1"

    checkpointer = Checkpointer(work_dir=work_dir, process_group=dist.new_group())
    model = model_factory()
    optim = torch.optim.AdamW(model.parameters())
    train_module = BasicTrainModule(model, optim, 128)

    # Save checkpoint.
    future = checkpointer.save_async(dir, train_module, {"rank": get_rank()})
    future.result()
    time.sleep(0.1)  # allow done callback to run.
    barrier()

    assert file_exists((f"{dir}/train/rank0.pt"))
    assert file_exists((f"{dir}/train/rank1.pt"))
    assert not dir_is_empty((f"{dir}/model_and_optim"))
    assert checkpointer.dir_is_checkpoint(dir)

    # Load checkpoint.
    train_state = checkpointer.load(dir, train_module)
    assert train_state is not None
    assert train_state["rank"] == get_rank()


def test_async_checkpointer_with_local_dir(tmp_path, tiny_model_factory):
    run_distributed_test(
        run_async_checkpointer,
        func_args=(tmp_path / "checkpoint", tmp_path / "work_dir", tiny_model_factory),
        start_method="spawn",
    )


def test_async_checkpointer_releases_staged_state_before_callbacks(tmp_path, monkeypatch):
    checkpointer = Checkpointer(work_dir=tmp_path)
    train_module = Mock(spec=TrainModule)
    staged_tensor = torch.empty(1)
    staged_tensor_ref = weakref.ref(staged_tensor)
    staged_state_dict = {"model": staged_tensor}
    train_module.state_dict_to_save.return_value = staged_state_dict
    del staged_tensor

    save_future: Future[None] = Future()
    observed_by_later_callback = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def fake_async_save_state_dict(_dir, state_dict, **_kwargs):
        assert state_dict is staged_state_dict
        return save_future

    monkeypatch.setattr(checkpoint_module, "async_save_state_dict", fake_async_save_state_dict)

    future = checkpointer.save_async(tmp_path / "checkpoint", train_module, {})
    future.add_done_callback(
        lambda _future: observed_by_later_callback.append(
            (dict(staged_state_dict), staged_tensor_ref() is None)
        )
    )

    assert staged_state_dict
    assert staged_tensor_ref() is not None
    assert observed_by_later_callback == []

    save_future.set_result(None)

    assert staged_state_dict == {}
    assert staged_tensor_ref() is None
    assert observed_by_later_callback == [({}, True)]


def test_async_checkpointer_with_remote_s3_dir(s3_checkpoint_dir, tmp_path, tiny_model_factory):
    from botocore.exceptions import NoCredentialsError

    try:
        dir_is_empty(s3_checkpoint_dir)
    except NoCredentialsError:
        pytest.skip("Requires AWS credentials")

    run_distributed_test(
        run_async_checkpointer,
        func_args=(s3_checkpoint_dir, tmp_path / "work_dir", tiny_model_factory),
        start_method="spawn",
    )


def test_async_checkpointer_with_remote_gcs_dir(gcs_checkpoint_dir, tmp_path, tiny_model_factory):
    from google.auth.exceptions import DefaultCredentialsError

    try:
        dir_is_empty(gcs_checkpoint_dir)
    except DefaultCredentialsError:
        pytest.skip("Requires authentication with Google Cloud")

    run_distributed_test(
        run_async_checkpointer,
        func_args=(gcs_checkpoint_dir, tmp_path / "work_dir", tiny_model_factory),
        start_method="spawn",
    )


class _RecordingDirectTrainModule:
    """Minimal stand-in exposing the direct-checkpoint API used by :class:`Checkpointer`."""

    def __init__(self, *, reset_optimizer_states_on_resume: bool):
        self.reset_optimizer_states_on_resume = reset_optimizer_states_on_resume
        self.received_reset = "unset"

    def load_state_dict_direct(self, dir, *, reset_optimizer_states_on_load=None, **kwargs):
        del dir, kwargs
        self.received_reset = reset_optimizer_states_on_load


@pytest.mark.parametrize("has_trainer_state", [True, False])
def test_checkpointer_resume_reset_only_applies_on_actual_resume(
    tmp_path, monkeypatch, has_trainer_state
):
    from torch.distributed.checkpoint.metadata import Metadata

    from olmo_core.train import checkpoint as checkpoint_module

    # Avoid needing a real distributed checkpoint on disk: the trainer-state probe is driven by the
    # presence of train/rank0.pt, and everything else is stubbed.
    monkeypatch.setattr(
        checkpoint_module, "get_checkpoint_metadata", lambda _dir: Metadata(state_dict_metadata={})
    )
    monkeypatch.setattr(checkpoint_module, "broadcast_object", lambda obj, **_kwargs: obj)

    dir = tmp_path / "checkpoint"
    (dir / "model_and_optim").mkdir(parents=True)
    if has_trainer_state:
        (dir / "train").mkdir()
        torch.save({"rank": 0}, dir / "train" / "rank0.pt")

    train_module = _RecordingDirectTrainModule(reset_optimizer_states_on_resume=True)
    checkpointer = Checkpointer(work_dir=tmp_path / "work_dir")
    # Default load_trainer_state=None: the resume reset must apply only when trainer state is found.
    # The fake only needs the duck-typed direct-checkpoint surface Checkpointer.load() probes for.
    checkpointer.load(dir, cast(TrainModule, train_module))

    assert train_module.received_reset is (True if has_trainer_state else None)
