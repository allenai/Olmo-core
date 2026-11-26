import logging
import os
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Set, Union

import torch

from olmo_core.distributed.checkpoint import save_state_dict
from olmo_core.distributed.utils import barrier, get_local_tensor, get_rank
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import clear_directory, dir_is_empty, is_url, join_path

from .callback import Callback
from .checkpointer import CheckpointerCallback
from .evaluator_callback import EvaluatorCallback

log = logging.getLogger(__name__)


@dataclass
class ModelMergeCallback(Callback):
    """
    Averages model weights over the last ``merge_last_n_steps`` before each ``merge_step``
    and saves the result as a merged checkpoint.

    Ephemeral checkpoints are blocked during merge windows to ensure the full
    window is always re-accumulated on resume.

    .. warning::
        This callback should be enabled with intention and configured with your
        training schedule in mind. Merge steps should be configured outside of decay
        phases where possible to ensure the averaged weights reflect a stable
        training regime.
    """

    # Run before CheckpointerCallback to block ephemeral checkpoints during merge windows
    priority: ClassVar[int] = 2

    merge_step: Union[int, List[int]] = field(default_factory=list)  # type: ignore[assignment]
    """The step(s) at which to save merged checkpoint(s)."""

    merge_interval: Optional[int] = None
    """Merge every N steps. Alternative to explicit merge_step."""

    merge_last_n_steps: int = 500
    """Number of steps before each merge step to start accumulating the average."""

    output_suffix: str = "merged"
    """Suffix for merged checkpoint directory."""

    save_checkpoint: bool = True
    """Whether to write the averaged weights out as a checkpoint. With ``False`` they are only
    swapped into the model for the merged-model evaluation (the ``eval/merged`` metrics), which
    is what train modules without a model-only state dict (e.g. the OLMo DDP module) need."""

    enabled: bool = False

    # Internal state (not checkpointed)
    _accumulators: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=dict, repr=False)
    _accumulator_counts: Dict[int, int] = field(default_factory=dict, repr=False)
    _merge_steps: List[int] = field(default_factory=list, repr=False)
    _completed_merges: Set[int] = field(default_factory=set, repr=False)

    def __post_init__(self):
        if not self.enabled:
            return

        if self.merge_last_n_steps <= 0:
            raise OLMoConfigurationError(
                f"merge_last_n_steps must be positive, got {self.merge_last_n_steps}"
            )

        if self.merge_interval is not None:
            if self.merge_interval <= 0:
                raise OLMoConfigurationError(
                    f"merge_interval must be positive, got {self.merge_interval}"
                )
            # Don't set both merge_step and merge_interval
            has_merge_step = (isinstance(self.merge_step, int) and self.merge_step > 0) or (
                isinstance(self.merge_step, list) and len(self.merge_step) > 0
            )
            if has_merge_step:
                raise OLMoConfigurationError(
                    "Cannot set both merge_step and merge_interval. "
                    "If you need both, compute all steps and pass them as merge_step."
                )
            # Defer step computation to pre_train (needs max_steps from trainer)
            return

        # Convert merge_step to list
        if isinstance(self.merge_step, int):
            self._merge_steps = [self.merge_step]
        else:
            self._merge_steps = sorted(self.merge_step)

        if not self._merge_steps:
            raise OLMoConfigurationError("Either merge_step or merge_interval must be set.")

        invalid = [s for s in self._merge_steps if s <= 0]
        if invalid:
            raise OLMoConfigurationError(f"merge_step values must be positive, got: {invalid}")

    def _window_start(self, merge_step: int) -> int:
        return max(0, merge_step - self.merge_last_n_steps + 1)

    def _active_windows(self) -> List[int]:
        """Return merge steps whose windows include the current step."""
        current = self.step
        return [
            ms
            for ms in self._merge_steps
            if ms not in self._completed_merges and self._window_start(ms) <= current <= ms
        ]

    def _merged_checkpoint_path(self, step: int) -> str:
        return str(join_path(self.trainer.save_folder, f"step{step}-{self.output_suffix}"))

    def pre_train(self):
        if not self.enabled:
            return

        # Compute merge steps from interval if needed
        if self.merge_interval is not None:
            max_steps = self.trainer.max_steps
            if max_steps is None:
                raise OLMoConfigurationError(
                    "merge_interval requires max_steps to be known. "
                    "Set max_duration on the trainer."
                )
            self._merge_steps = list(range(self.merge_interval, max_steps + 1, self.merge_interval))
            if not self._merge_steps:
                log.warning(
                    f"No merge steps computed: merge_interval={self.merge_interval}, "
                    f"max_steps={max_steps}"
                )
                return

        log.info(f"ModelMergeCallback: merge_steps={self._merge_steps}")

        # Mark merge steps that are already past as completed
        current_step = self.step
        for ms in self._merge_steps:
            if ms < current_step:
                log.warning(
                    f"Current step {current_step} is past merge step {ms}. "
                    "This merge will be skipped."
                )
                self._completed_merges.add(ms)

        # Skip merges where we resumed mid-window (can't accumulate full average)
        for ms in self._merge_steps:
            if ms not in self._completed_merges and self._window_start(ms) < current_step <= ms:
                log.warning(
                    f"Resumed at step {current_step} inside merge window "
                    f"[{self._window_start(ms)}, {ms}]. "
                    f"This merge will be skipped (cannot accumulate full "
                    f"{self.merge_last_n_steps}-step average)."
                )
                self._completed_merges.add(ms)

        remaining = [ms for ms in self._merge_steps if ms not in self._completed_merges]
        log.info(f"Remaining merge steps: {remaining}")

        # Check if any merge window would be shorter than configured
        for ms in remaining:
            if ms < self.merge_last_n_steps:
                raise OLMoConfigurationError(
                    f"Merge step {ms} is less than merge_last_n_steps "
                    f"({self.merge_last_n_steps}). The merge window would only be "
                    f"{ms + 1} steps instead of {self.merge_last_n_steps}."
                )

        # Warn if any permanent checkpoint could land inside a merge window.
        # On resume from that checkpoint, the merge will be skipped.
        checkpointer = next(
            (cb for cb in self.trainer.callbacks.values() if isinstance(cb, CheckpointerCallback)),
            None,
        )
        if checkpointer and checkpointer.save_interval:
            si = checkpointer.save_interval
            for ms in remaining:
                ws = self._window_start(ms)
                first_ckpt = ((ws // si) + 1) * si
                if ws < first_ckpt < ms:
                    log.warning(
                        f"Permanent checkpoint at step {first_ckpt} falls inside "
                        f"merge window [{ws}, {ms}]. If training is interrupted and "
                        f"resumed from that checkpoint, this merge will be skipped."
                    )

    def post_train_batch(self):
        if not self.enabled:
            return

        active = self._active_windows()

        if not active:
            self.unblock_ephemeral_checkpoints()
            return

        # Copy model weights to CPU once for all active windows
        model = self.trainer.train_module.model
        model_state = {
            k: get_local_tensor(p.data.detach()).to("cpu") for k, p in model.named_parameters()
        }

        for ms in active:
            self._accumulate_weights(ms, model_state)

        # Save any windows that just completed
        for ms in active:
            if self.step == ms:
                self._save_merged_checkpoint(ms)

        # Block ephemeral checkpoints during merge windows to prevent
        # mid-window resume points that would cause the merge to be skipped.
        # Set AFTER saves so the flag is False once all windows at this step complete.
        still_active = [ms for ms in active if ms not in self._completed_merges]
        if still_active:
            self.block_ephemeral_checkpoints()
        else:
            self.unblock_ephemeral_checkpoints()

    def _accumulate_weights(self, merge_step: int, model_state: Dict[str, torch.Tensor]):
        if merge_step not in self._accumulators:
            log.info(
                f"Starting weight accumulation for merge step {merge_step} at step {self.step}"
            )
            self._accumulators[merge_step] = {
                k: torch.zeros_like(v, dtype=torch.float32, device="cpu")
                for k, v in model_state.items()
            }
            self._accumulator_counts[merge_step] = 0

        for key, value in model_state.items():
            self._accumulators[merge_step][key].add_(value.float())

        self._accumulator_counts[merge_step] += 1
        log.debug(
            f"Accumulated weights for merge step {merge_step} at step {self.step} "
            f"({self._accumulator_counts[merge_step]} total)"
        )

    @torch.no_grad()
    def _save_merged_checkpoint(self, merge_step: int):
        accumulator = self._accumulators.get(merge_step)
        count = self._accumulator_counts.get(merge_step, 0)

        if accumulator is None or count == 0:
            log.warning(f"No weights accumulated for merge step {merge_step}, cannot save")
            return

        log.info(
            f"{'Saving' if self.save_checkpoint else 'Evaluating'} merged model "
            f"(average of {count} steps) at step {merge_step}"
        )

        averaged_state: Dict[str, torch.Tensor] = {
            key: acc_val / count for key, acc_val in accumulator.items()
        }

        output_path = self._merged_checkpoint_path(merge_step)

        if self.save_checkpoint and get_rank() == 0:
            if not dir_is_empty(output_path):
                clear_directory(output_path)
            if not is_url(output_path):
                os.makedirs(output_path, exist_ok=True)

        barrier()

        # To save and evaluate correctly under FSDP, we temporarily load averaged
        # weights into the model so save_state_dict sees DTensors (with sharding
        # metadata) rather than plain tensors (which would be treated as replicated
        # and only save rank 0's data). We keep the weights loaded for evaluation
        # to avoid swapping twice.
        model = self.trainer.train_module.model
        params_dict = dict(model.named_parameters())

        original_state = {
            k: get_local_tensor(p.data.detach()).to("cpu").clone() for k, p in params_dict.items()
        }

        for name, param in params_dict.items():
            if name in averaged_state:
                local_param = get_local_tensor(param.data)
                local_param.copy_(averaged_state[name].to(local_param.device, local_param.dtype))

        barrier()

        try:
            if self.save_checkpoint:
                save_state_dict(
                    join_path(output_path, "model_and_optim"),
                    self.trainer.train_module.state_dict_to_save(optim=False),
                    process_group=self.trainer.checkpointer.process_group,
                )

                barrier()
                log.info(f"Merged checkpoint saved to: {output_path}")

            self._evaluate_merged()
        finally:
            # Restore original weights
            log.info("Restoring original model weights...")
            for name, param in params_dict.items():
                if name in original_state:
                    local_param = get_local_tensor(param.data)
                    local_param.copy_(
                        original_state[name].to(local_param.device, local_param.dtype)
                    )
            barrier()

        # Clean up
        del self._accumulators[merge_step]
        del self._accumulator_counts[merge_step]
        self._completed_merges.add(merge_step)

    def _evaluate_merged(self):
        """Run evaluations with the currently loaded (merged) model weights."""
        evaluator_callbacks = [
            cb for cb in self.trainer.callbacks.values() if isinstance(cb, EvaluatorCallback)
        ]

        if not evaluator_callbacks:
            log.info("No EvaluatorCallback found, skipping merged model evaluation")
            return

        for callback in evaluator_callbacks:
            log.info(f"Running merged model evaluation via {callback.__class__.__name__}...")
            callback.perform_eval(prefix="eval/merged")


# Utility functions for computing merge steps and required checkpoint steps for merge windows
def compute_merge_steps_from_decay_schedule(
    period_lengths: List[int],
    tokens_per_step: int,
    decay: Optional[int] = None,
    decay_fraction: Optional[float] = None,
) -> List[int]:
    """
    Compute merge steps from a decay schedule with one or more periods.
    """
    if decay is None and decay_fraction is None:
        raise ValueError("Either decay or decay_fraction must be set")

    merge_steps = []
    cumulative_tokens = 0

    for period_length in period_lengths:
        cumulative_tokens += period_length

        if decay is not None:
            decay_tokens = decay
        else:
            assert decay_fraction is not None
            decay_tokens = int(round(decay_fraction * period_length))

        pre_decay_tokens = cumulative_tokens - decay_tokens
        pre_decay_step = pre_decay_tokens // tokens_per_step
        merge_steps.append(pre_decay_step)

    return merge_steps


def compute_merge_window_starts(
    merge_steps: List[int],
    merge_last_n_steps: int,
) -> List[int]:
    """
    Compute the checkpoint steps needed at the start of each merge window.

    These steps should be passed as ``fixed_steps`` to the checkpointer so that
    a checkpoint always exists at the beginning of each merge window. Without this,
    a mid-window resume would cause the merge to be skipped.

    For overlapping windows, only the earliest start in each group is returned
    since it covers all windows in that group.
    """
    if not merge_steps:
        return []

    required_starts: List[int] = []
    prev_merge_step = -1

    for ms in sorted(merge_steps):
        start = max(0, ms - merge_last_n_steps + 1)
        # If this window starts after the previous merge step completed,
        # it's a new group and needs its own checkpoint
        if start > prev_merge_step:
            required_starts.append(start)
        prev_merge_step = ms

    return required_starts
