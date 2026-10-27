import logging
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import timedelta
from typing import ClassVar, List, Optional, Tuple

import torch.distributed as dist

from olmo_core.config import StrEnum
from olmo_core.distributed.utils import (
    backend_supports_cpu,
    broadcast_object,
    get_fs_local_rank,
    get_rank,
    is_distributed,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import clear_directory, is_url, join_path, remove_file
from olmo_core.utils import wait_for

from ..checkpoint import Checkpointer
from .callback import Callback

log = logging.getLogger(__name__)


class CheckpointRemovalStrategy(StrEnum):
    """
    An enumeration of the different strategies for removing old checkpoints found in the save folder.
    """

    ephemeral_only = "ephemeral_only"
    """
    Only remove checkpoints that were saved at the :data:`CheckpointerCallback.ephemeral_save_interval`.
    """

    all_non_permanent = "all_non_permanent"
    """
    Remove all non-permanent checkpoints found, including ephemeral checkpoints and also
    any other checkpoints that were not saved at the :data:`CheckpointerCallback.save_interval`.
    """

    never = "never"
    """
    Never remove any old checkpoints found in the save folder.
    """


@dataclass
class CheckpointerCallback(Callback):
    """
    Manages checkpointing during training, including writing checkpoints at set intervals
    determined by :data:`save_interval` and :data:`ephemeral_save_interval`, as well as removing
    old checkpoints found in the save folder as determined by the :data:`remove` setting.

    .. important::
        This callback gets added automatically if you don't explicitly configure it.
        If you want to override this callback you should subclass it.
    """

    priority: ClassVar[int] = 1

    save_interval: Optional[int] = 250
    """
    The interval, in steps, with which to save permanent checkoints.
    """

    ephemeral_save_interval: Optional[int] = None
    """
    The interval, in steps, with which to save temporary checkpoints. These checkpoints are removed
    each time a new checkpoint is saved.

    It can be useful to set this to a relatively frequent interval for preemptible jobs.
    """

    pre_train_checkpoint: Optional[bool] = None
    """
    Save a pretrain checkpoint. Defaults to ``True`` unless the trainer resumes from a checkpoint.
    """

    save_async: Optional[bool] = None
    """
    Save checkpoints asynchronously. Requires a separate CPU-only backend.
    Defaults to ``True`` if there is one.
    """

    remove: CheckpointRemovalStrategy = CheckpointRemovalStrategy.ephemeral_only
    """
    The strategy for removing old checkpoints found in the save folder.
    """

    ephemeral_cooldown: Optional[int] = None
    """
    The number of steps to wait after saving a checkpoint before saving another ephemeral one is allowed.
    """

    fixed_steps: Optional[List[int]] = None
    """
    A list of fixed steps at which to save additional permanent checkpoints.
    """

    max_checkpoints: Optional[int] = 3
    """
    Maximum number of permanent checkpoints to keep. When a new permanent checkpoint
    is saved and the count exceeds this limit, the oldest is removed.
    Set to ``None`` to keep all checkpoints (previous behavior).

    .. note::
        Checkpoints saved at :data:`fixed_steps` are counted toward this limit.
    """

    enabled: bool = True

    # Bookkeeping

    # NOTE: can't use type annotation here, omegaconf doesn't like it
    #  _future: Optional[Future] = None
    _future = None
    _latest_checkpoint_step: int = -1
    _latest_checkpoint_path: str = ""
    _checkpoints: List[str] = field(default_factory=list)
    _ephemeral_checkpoints: List[str] = field(default_factory=list)
    _checkpoints_to_remove: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.save_interval is not None and self.save_interval < 1:
            raise OLMoConfigurationError("'save_interval' must be at least 1")
        if self.ephemeral_save_interval is not None:
            if self.ephemeral_save_interval < 1:
                raise OLMoConfigurationError("'ephemeral_save_interval' must be at least 1")
            if (
                self.save_interval is not None
                and self.ephemeral_save_interval >= self.save_interval
            ):
                raise OLMoConfigurationError(
                    "'ephemeral_save_interval' must be less than 'save_interval'"
                )
        if self.max_checkpoints is not None and self.max_checkpoints < 1:
            raise OLMoConfigurationError("'max_checkpoints' must be at least 1")

    @property
    def checkpointer(self) -> Checkpointer:
        return self.trainer.checkpointer

    @property
    def save_folder(self) -> str:
        return self.trainer.save_folder

    @property
    def checkpoint_pending(self) -> bool:
        return self._future is not None

    def _await_last_checkpoint(self, blocking: bool = True) -> Optional[Future]:
        if (fut := self._future) is not None:
            # Wait for last async checkpoint to finish.
            if blocking or fut.done():
                fut.result()
                if get_rank() == 0:
                    # Just to be safe, make sure the checkpointer has finalized the checkpoint.
                    wait_for(
                        lambda: self.checkpointer.dir_is_checkpoint(self._latest_checkpoint_path),
                        "waiting to finalize checkpoint",
                    )
                self._future = None
                return fut
        return None

    def _save_checkpoint(self, save_async: Optional[bool] = None, ephemeral: bool = False) -> str:
        save_async = save_async if save_async is not None else self.save_async
        self._await_last_checkpoint()
        if save_async:
            path, self._future = self.trainer.save_checkpoint_async(ephemeral=ephemeral)
        else:
            path = self.trainer.save_checkpoint(ephemeral=ephemeral)
        self._latest_checkpoint_step = self.step
        self._latest_checkpoint_path = str(path)
        return str(path)

    def _remove_checkpoint(self, path: str):
        log.info(f"Removing old checkpoint at '{path}'...")

        # Remove metadata file first to invalidate the checkpoint.
        if get_rank() == 0:
            try:
                remove_file(join_path(path, self.trainer.checkpointer.METADATA_FNAME))
            except FileNotFoundError:
                pass

        if is_url(path):
            if get_rank() == 0:
                self.trainer.run_bookkeeping_op(
                    clear_directory,
                    path,
                    op_name=f"clear_directory {path}",
                    distributed=False,
                    soft_timeout=180,  # this can take a while on GCS, for example
                )
        elif get_fs_local_rank() == 0:
            self.trainer.run_bookkeeping_op(
                clear_directory, path, op_name=f"clear_directory {path}", distributed=False
            )

    def _schedule_for_removal(self, path: str):
        self._checkpoints_to_remove.append(path)

    def _remove_old_checkpoints(self):
        for path in self._checkpoints_to_remove:
            self._remove_checkpoint(path)
        self._checkpoints_to_remove.clear()

    def _trim_checkpoints(self):
        if self.max_checkpoints is not None:
            while len(self._checkpoints) > self.max_checkpoints:
                oldest_path = self._checkpoints.pop(0)
                self._schedule_for_removal(oldest_path)

    def pre_train(self):
        if not self.enabled:
            return

        if self.save_async is None:
            self.save_async = backend_supports_cpu()

        # Maybe create a new process group for async checkpointing.
        if is_distributed() and self.save_async and self.checkpointer.process_group is None:
            if not backend_supports_cpu():
                raise RuntimeError("a CPU-capable backend is required for async checkpointing")

            log.info(
                "Creating new process group for checkpointing (needed for async checkpointing)"
            )
            self.checkpointer.process_group = dist.new_group(timeout=timedelta(minutes=30))

        # Maybe save a pre-train checkpoint.
        if self.step == 0 and (
            self.pre_train_checkpoint
            or (self.pre_train_checkpoint is None and not self.trainer.checkpoint_loaded)
        ):
            self._checkpoints.append(self._save_checkpoint())

        # Collect existing ephemeral checkpoints from previous runs.
        if self.remove != CheckpointRemovalStrategy.never:
            ephemeral_checkpoints: List[Tuple[int, str]] = []

            # Only search from rank 0 to avoid hammering remote file stores with requests.
            if get_rank() == 0:
                try:
                    for step_num, path in self.checkpointer.find_checkpoints(self.save_folder):
                        if (
                            step_num == 0
                            or step_num > self.step
                            or (self.fixed_steps is not None and step_num in self.fixed_steps)
                            or (
                                self.save_interval is not None
                                and step_num % self.save_interval == 0
                            )
                        ):
                            continue
                        elif (
                            self.remove == CheckpointRemovalStrategy.ephemeral_only
                            and self.ephemeral_save_interval is not None
                            and step_num % self.ephemeral_save_interval == 0
                        ) or (self.remove == CheckpointRemovalStrategy.all_non_permanent):
                            ephemeral_checkpoints.append((step_num, path))
                except FileNotFoundError:
                    pass

            ephemeral_checkpoints = broadcast_object(ephemeral_checkpoints)

            # TODO: handle this if we ever restore callback state.
            assert not self._ephemeral_checkpoints

            self._ephemeral_checkpoints = [
                path for _, path in sorted(ephemeral_checkpoints, key=lambda x: x[0])
            ]
            for path in self._ephemeral_checkpoints:
                log.info(
                    f"Found existing ephemeral checkpoint at '{path}' which will "
                    "be removed when the next checkpoint is saved"
                )

    def _is_final_step(self) -> bool:
        """Whether this is the run's last step (the earlier of ``max_duration`` and ``hard_stop``)."""
        t = self.trainer
        try:
            cands = [t.max_steps]
            if getattr(t, "hard_stop", None) is not None:
                cands.append(t._get_max_steps(t.hard_stop))
        except Exception:  # durations that cannot be converted to steps up front
            return False
        cands = [int(c) for c in cands if c]
        return bool(cands) and self.step >= min(cands)

    def post_train_batch(self):
        if not self.enabled:
            return

        self._await_last_checkpoint(blocking=False)
        if not self.checkpoint_pending:
            self._remove_old_checkpoints()

        # An interval / fixed-step save that lands on the FINAL step is the final checkpoint (post_train reuses it):
        # save it synchronously, so no async save is still running its gloo collectives while ranks tear down
        # (2026-10-26: 8 of 10 runs whose last step was a multiple of save_interval died after the last step with a
        # gloo "Connection reset by peer"; 0 of ~21 others did).
        final_async = False if self._is_final_step() else None
        if self.fixed_steps is not None and self.step in self.fixed_steps:
            # Save permanent checkpoint.
            self._checkpoints.append(self._save_checkpoint(save_async=final_async))
            self._trim_checkpoints()
        elif self.save_interval is not None and self.step % self.save_interval == 0:
            # Save permanent checkpoint.
            self._checkpoints.append(self._save_checkpoint(save_async=final_async))
            self._trim_checkpoints()
        elif (
            self.ephemeral_save_interval is not None
            and self.step % self.ephemeral_save_interval == 0
            and not self.trainer.block_ephemeral_checkpoints
        ):
            # Maybe save ephemeral checkpoint.
            if (
                self.ephemeral_cooldown is not None
                and (self.step - self._latest_checkpoint_step) < self.ephemeral_cooldown
            ):
                return

            self._ephemeral_checkpoints.append(self._save_checkpoint(ephemeral=True))

            # Remove old ephemeral checkpoints.
            while len(self._ephemeral_checkpoints) > 1:
                oldest_path = self._ephemeral_checkpoints.pop(0)
                self._schedule_for_removal(oldest_path)

    def post_train(self):
        if not self.enabled:
            return

        if self.step > self._latest_checkpoint_step:
            self._checkpoints.append(self._save_checkpoint(save_async=False))
            self._trim_checkpoints()

        self._await_last_checkpoint()
