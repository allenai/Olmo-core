from pathlib import Path
from typing import List

import pytest
import torch

from olmo_core.io import is_url
from olmo_core.testing import run_distributed_test
from olmo_core.train.callbacks.checkpointer import CheckpointerCallback
from olmo_core.train.checkpoint import Checkpointer


class StubTrainer:
    """
    Just enough of a :class:`~olmo_core.train.Trainer` for `CheckpointerCallback.post_train`:
    the save folder, the global step, and a recording `save_checkpoint`.
    """

    def __init__(self, save_folder, global_step: int):
        self.save_folder = str(save_folder)
        self.global_step = global_step
        self.saved_steps: List[int] = []

    def save_checkpoint(self, ephemeral: bool = False):
        del ephemeral
        self.saved_steps.append(self.global_step)
        return f"{self.save_folder}/step{self.global_step}"


def make_callback(trainer: StubTrainer) -> CheckpointerCallback:
    callback = CheckpointerCallback(save_async=False)
    callback.trainer = trainer  # type: ignore[assignment]
    return callback


def _in_tmp(tmp_path: Path, path: str) -> str:
    return path if is_url(path) else str(tmp_path / path)


@pytest.mark.parametrize(
    "save_folder, loaded_folder, trained_steps, expected_saved_steps",
    [
        pytest.param("ckpt", "ckpt", 0, [], id="restart_of_finished_run_saves_nothing"),
        pytest.param("ckpt", "ckpt", 2, [6], id="resume_that_trains_saves_the_final_step"),
        pytest.param("ckpt-2", "ckpt", 0, [6], id="load_from_another_folder_gets_a_final_save"),
        pytest.param(
            "ckpt", "subdir/../ckpt", 0, [], id="save_folder_spelled_differently_is_a_resume"
        ),
        pytest.param(
            "s3://bucket/run", "s3://bucket/run", 0, [], id="remote_restart_saves_nothing"
        ),
        pytest.param(
            "s3://bucket/run-2",
            "s3://bucket/run",
            0,
            [6],
            id="remote_load_from_another_folder_gets_a_final_save",
        ),
    ],
)
def test_post_train_save_after_checkpoint_loaded(
    tmp_path: Path,
    save_folder: str,
    loaded_folder: str,
    trained_steps: int,
    expected_saved_steps: List[int],
):
    # Regression test for https://github.com/allenai/OLMo-core/issues/962: a restart of a finished
    # run resumes the final checkpoint, trains no step, and must not save that step again in
    # `post_train`. A checkpoint loaded from outside the save folder (e.g. `load_path`) starts a
    # new run, whose final checkpoint must still be saved even when no step is trained.
    loaded_step = 6 - trained_steps
    trainer = StubTrainer(_in_tmp(tmp_path, save_folder), global_step=loaded_step)
    callback = make_callback(trainer)

    callback.post_checkpoint_loaded(_in_tmp(tmp_path, f"{loaded_folder}/step{loaded_step}"))
    trainer.global_step = 6
    callback.post_train()

    assert trainer.saved_steps == expected_saved_steps


TOTAL_STEPS = 6


def _fit_tiny_run(work_dir: str):
    import torch.nn as nn
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    from olmo_core.data.data_loader import DataLoaderBase
    from olmo_core.distributed.utils import get_fs_local_rank, get_rank, get_world_size
    from olmo_core.train import Duration, TrainerConfig
    from olmo_core.train.train_module import BasicTrainModule

    work = Path(work_dir)
    vocab, dim, seq, batch = 32, 8, 4, 2
    rank, world = get_rank(), get_world_size()

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, dim)
            self.out = nn.Linear(dim, vocab, bias=False)

        def forward(self, input_ids):
            return self.out(self.emb(input_ids))

    class TinyDataLoader(DataLoaderBase):
        @property
        def total_batches(self):
            return 10_000

        def state_dict(self):
            return {"batches_processed": self.batches_processed, "epoch": self._epoch}

        def load_state_dict(self, state_dict):
            self.batches_processed = state_dict["batches_processed"]
            self._epoch = state_dict["epoch"]

        def reshuffle(self, epoch=None, **kwargs):
            self._epoch = epoch or self._epoch or 1

        def _iter_batches(self):
            for i in range(self.batches_processed, self.total_batches):
                generator = torch.Generator().manual_seed(i * world + rank)
                yield {"input_ids": torch.randint(0, vocab, (batch, seq), generator=generator)}

        def get_mock_batch(self):
            return {"input_ids": torch.randint(0, vocab, (batch, seq))}

        def global_num_tokens_in_batch(self, batch):
            return self.global_batch_size

    class TinyTrainModule(BasicTrainModule):
        def num_flops_per_token(self, seq_len):
            return None

        def global_num_flops_in_batch(self, batch):
            return None

    torch.manual_seed(0)
    model = TinyModel()
    fully_shard(model, mesh=init_device_mesh("cpu", (world,)))
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    train_module = TinyTrainModule(model, optim, rank_microbatch_size=batch * seq)
    data_loader = TinyDataLoader(
        work_dir=str(work / "dataloader"),
        global_batch_size=batch * seq * world,
        dp_world_size=world,
        dp_rank=rank,
        fs_local_rank=get_fs_local_rank(),
    )
    trainer = TrainerConfig(
        save_folder=str(work / "ckpt"),
        max_duration=Duration.steps(TOTAL_STEPS),
        callbacks={"checkpointer": CheckpointerCallback(save_interval=2, save_async=False)},
    ).build(train_module, data_loader)
    trainer.fit()


def test_restart_of_finished_run_exits_cleanly(tmp_path: Path):
    # Regression test for https://github.com/allenai/OLMo-core/issues/962: rerunning the
    # identical command after a run finished must resume the final checkpoint, train nothing,
    # write nothing, and exit cleanly (instead of re-saving the final step, which fails with
    # FileExistsError under the default save_overwrite=False).
    run_distributed_test(_fit_tiny_run, world_size=2, func_args=(str(tmp_path),))

    final_checkpoint = tmp_path / "ckpt" / f"step{TOTAL_STEPS}"
    assert Checkpointer.dir_is_checkpoint(final_checkpoint)
    metadata = final_checkpoint / Checkpointer.METADATA_FNAME
    fingerprint_before = (metadata.stat().st_ino, metadata.stat().st_mtime_ns)
    step_dirs_before = sorted(p.name for p in (tmp_path / "ckpt").glob("step*"))

    # The restart: the identical command in the same save folder.
    run_distributed_test(_fit_tiny_run, world_size=2, func_args=(str(tmp_path),))

    assert sorted(p.name for p in (tmp_path / "ckpt").glob("step*")) == step_dirs_before
    assert (metadata.stat().st_ino, metadata.stat().st_mtime_ns) == fingerprint_before
