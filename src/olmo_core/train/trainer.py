import concurrent.futures
import logging
import math
import signal
import time
import uuid
import warnings
from collections import OrderedDict, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from numbers import Real
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import torch
import torch.distributed as dist

from ..aliases import PathOrStr
from ..data import DataLoaderBase
from ..distributed.utils import (
    all_reduce_value,
    backend_supports_cpu,
    barrier,
    broadcast_object,
    get_fs_local_rank,
    get_global_rank,
    get_local_tensor,
    get_rank,
    get_world_size,
    is_distributed,
)
from ..exceptions import OLMoConfigurationError
from ..io import (
    copy_dir,
    copy_file,
    dir_is_empty,
    file_exists,
    glob_directory,
    is_url,
    join_path,
    normalize_path,
)
from ..utils import cuda_sync_debug_mode, gc_cuda, get_default_thread_count
from .callbacks import (
    Callback,
    CheckpointerCallback,
    ConsoleLoggerCallback,
    EvaluatorCallback,
    GarbageCollectorCallback,
    SpeedMonitorCallback,
)
from .checkpoint import Checkpointer
from .common import (
    TRAIN_CE_LOSS_METRIC,
    TRAIN_PPL_METRIC,
    Duration,
    DurationUnit,
    LoadStrategy,
    MetricMergeStrategy,
    ReduceType,
    StepSkipRange,
    TrainingProgress,
)
from .train_module import TrainModule
from .utils import EnvRngStates, check_metrics_consistent, move_metrics, reduce_metrics

log = logging.getLogger(__name__)


T = TypeVar("T")


def _metric_value_to_tensor(value: Union[float, torch.Tensor]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        # Coerce Python scalars to float64 so large integer metrics (e.g. token counts) don't
        # overflow or lose precision when torch would otherwise pick int64/float32.
        if isinstance(value, Real):
            return torch.tensor(value, dtype=torch.float64)
        return torch.tensor(value)

    return get_local_tensor(value.detach()).float()


class TrainerStateDict(TypedDict):
    global_step: int
    global_train_tokens_seen: int
    global_train_petaflops: float
    max_steps: Optional[int]
    data_loader: Dict[str, Any]
    epoch: int
    world_size: int
    rng: Dict[str, Any]
    callbacks: Dict[str, Dict[str, Any]]


@dataclass
class Trainer:
    """
    Language model trainer.

    .. tip::
        Use :class:`TrainerConfig` instead of constructing this class directly.
    """

    work_dir: Path
    """
    A local working directory to use for temporary files needed during training.
    Files added to this working directory can be persisted to the :data:`save_folder` via
    :meth:`persist_working_file()`.

    .. note::
        When constructing your trainer through a :class:`TrainerConfig` this will default to
        the :data:`save_folder` if it's a local directory.
    """

    train_module: TrainModule
    """
    The train module to fit.
    """

    data_loader: DataLoaderBase
    """
    The train data loader.
    """

    device: torch.device
    """
    The default device to use. Should match the device the model is on and be appropriate for the
    main distributed backend.
    """

    save_folder: str
    """
    The folder to save all checkpoints to. Could be a local directory (if using a shared filesytem)
    or a URL.

    .. warning::
        If you try to use a local directory without a globally shared filesystem across all ranks
        you will get an error.
    """

    checkpointer: Checkpointer
    """
    The checkpointer. This is a wrapper around the functionality in
    :mod:`olmo_core.distributed.checkpoint`, which means you can use
    :func:`~olmo_core.distributed.checkpoint.unshard_checkpoint` to unshard the model and optimizer
    state from a train checkpoint after the fact.
    """

    callbacks: Dict[str, Callback]
    """
    Trainer callbacks.
    """

    max_duration: Duration
    """
    The duration to train for.

    .. important::
        The total number of training steps must be known ahead of time for various reasons such
        as setting a learning rate schedule. Therefore if your data loader's number of batches
        (:data:`~olmo_core.data.data_loader.DataLoaderBase.total_batches`) is unknown ahead of time,
        you must set the ``max_duration`` in terms of :meth:`tokens <Duration.tokens>`
        or :meth:`steps <Duration.steps>`, but not epochs.
    """

    save_overwrite: bool = False
    """
    Whether to overwrite existing files/checkpoints in the :data:`save_folder`.
    """

    load_path: Optional[PathOrStr] = None
    """
    An alternative location to load a checkpoint from if no checkpoint is found in the current :data:`save_folder`.

    This can be set to a checkpoint path or the path to a folder of checkpoints such as the :data:`save_folder`
    from a different run.
    """

    load_strategy: LoadStrategy = LoadStrategy.if_available
    """
    The strategy for loading a checkpoint prior to training.
    """

    load_trainer_state: Optional[bool] = None
    """
    Whether to load the trainer state (including dataloader state). If ``None``, this will attempt
    to load the trainer state if it exists in the checkpoint, but will will not error if it doesn't.
    """

    load_optim_state: Optional[bool] = None
    """
    Whether to load the optimizer state. If ``None``, this will attempt to load the optimizer state
    if it exists in the checkpoint, but will not error if it doesn't.
    """

    metrics_collect_interval: int = 5
    """
    How often (in steps) to collect, reduce, and pass on metrics to the
    :meth:`Callback.log_metrics <olmo_core.train.callbacks.Callback.log_metrics>` method on callbacks.

    .. note::
        Regardless of what this is set to, the
        :meth:`Callback.log_metrics <olmo_core.train.callbacks.Callback.log_metrics>` methods are still
        called with the metrics for every single step, but will be delayed according to this value.

        For example, if this is set to 5, then every 5 steps the metrics from the past 5 steps
        are collected and reduced together, then passed on to
        :meth:`Callback.log_metrics <olmo_core.train.callbacks.Callback.log_metrics>` altogether.

    .. tip::
        Increasing this can improve throughput since logging metrics always requires a host-device
        sync.
    """

    dp_process_group: Optional[dist.ProcessGroup] = None
    """
    The distributed process group for all data parallel ranks.
    """

    # Bookkeeping

    global_step: int = 0
    """
    The current step/batch. 1-based, though it's initialized to 0 before the first step.
    This does *not* reset after an epoch.
    """

    global_train_tokens_seen: int = 0
    """
    The total number of training tokens seen.
    """

    global_train_petaflops: float = 0.0
    """
    The total number of training petaflops computed.
    """

    epoch: int = 1
    """
    The current epoch (1-based).
    """

    cancel_check_interval: int = 25
    """
    The interval (in steps) to check if the run is canceled. Checking requires distributed comms,
    but if you've configured a separate CPU-only backend (like "gloo") then this shouldn't impact
    training throughput.
    """

    hard_stop: Optional[Duration] = None
    """
    Set a hard stopping point for the trainer. This is useful for ablations when you you don't
    want to do a complete training run, but you don't want to change :data:`max_duration` as to
    not affect the learning rate schedule.
    """

    async_bookkeeping: Optional[bool] = None
    """
    Do collective bookkeeping operations like reducing metrics asynchronously.
    This requires a separate CPU-only backend, and will default to ``True`` if one is available.
    """

    bookkeeping_soft_timeout: int = 30
    """
    A soft timeout (in seconds) for bookkeeping operations. If a bookkeeping operation takes longer
    than this then a warning is emitted.
    """

    # Benchmarking

    no_checkpoints: bool = False
    """
    Set this to ``True`` to disable automatic saving/loading of checkpoints altogether.
    This is useful for benchmarking.
    """

    no_evals: bool = False
    """
    Set this to ``True`` to disable evaluator callbacks.
    This is useful for benchmarking.
    """

    steps_to_skip: Optional[List[StepSkipRange]] = None
    """
    Ranges of steps to completely skip training on.
    """

    checkpoints_to_eval: Optional[List[str]] = None
    """
    Checkpoint paths (or globs) to evaluate with :meth:`eval_checkpoints`. No effect during training.
    """

    # Internal bookkeeping

    _metrics: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=OrderedDict)
    _metrics_reduce_type: Dict[str, Optional[ReduceType]] = field(default_factory=dict)
    _canceled: bool = False
    _cancel_reason: Optional[str] = None
    _canceling_rank: Optional[int] = None
    _error: Optional[BaseException] = None
    _rank_batch_size: Optional[int] = None
    _multi_thread_pool: Optional[ThreadPoolExecutor] = None
    _single_thread_pool: Optional[ThreadPoolExecutor] = None
    # maps bookkeeping operation name to an ordereddict of operation ID to operation Future
    _bookkeeping_queue: Dict[str, Dict[str, Future]] = field(
        default_factory=lambda: defaultdict(OrderedDict)
    )
    _bookkeeping_pg: Optional[dist.ProcessGroup] = None
    _blocking_ephemeral_checkpoints: Set[str] = field(repr=False, default_factory=set)
    """Callbacks that are blocking ephemeral checkpoints."""
    _checkpoint_loaded: bool = False
    _metrics_consistent: Optional[bool] = None

    def __post_init__(self):
        self.save_folder = normalize_path(self.save_folder)
        if self.load_path is not None:
            self.load_path = normalize_path(self.load_path)

        # If save folder is a local directory, make sure we're using a shared filesystem.
        if not is_url(self.save_folder) and get_fs_local_rank() != get_rank():
            raise OLMoConfigurationError(
                "Checkpointing to a local directory requires a shared filesystem. "
                "If you do have a shared filesystem please set the env var 'OLMO_SHARED_FS=1' "
                "or set 'FS_LOCAL_RANK' to the global rank for each process."
            )

        # Validate working directory.
        if is_url(self.work_dir):
            raise OLMoConfigurationError(
                f"Trainer working directory must be a local path, got a URL instead ('{self.work_dir}')."
            )
        self.work_dir = Path(normalize_path(self.work_dir))

        # Ensure save folder and working directory exist.
        if get_fs_local_rank() == 0:
            self.work_dir.mkdir(exist_ok=True, parents=True)
            if not is_url(self.save_folder):
                Path(self.save_folder).mkdir(exist_ok=True, parents=True)

        # Ensure we have necessary callbacks.
        if not self.has_callback(ConsoleLoggerCallback):
            self.callbacks.setdefault(
                "console_logger",
                ConsoleLoggerCallback(metrics_log_interval=self.metrics_collect_interval),
            )
        if not self.has_callback(CheckpointerCallback):
            self.callbacks.setdefault("checkpointer", CheckpointerCallback())
        if not self.has_callback(SpeedMonitorCallback):
            self.callbacks.setdefault("speed_monitor", SpeedMonitorCallback())
        if not self.has_callback(GarbageCollectorCallback):
            self.callbacks.setdefault("garbage_collector", GarbageCollectorCallback())

        # Set pointer to self in all callbacks.
        for callback in self.callbacks.values():
            callback.trainer = self

        # Sort callbacks by (priority, name).
        # We do this for 2 reasons: (1) to respect the priority, and (2) to ensure the callback
        # order is consistent across the process group since some callbacks make distributed
        # synchronization/communication calls.
        self._sort_callbacks()

        if self.dp_process_group is None:
            self.dp_process_group = self.train_module.dp_process_group

        # Maybe create separate process group for bookkeeping.
        if self._bookkeeping_pg is None and is_distributed():
            if self.async_bookkeeping is None:
                self.async_bookkeeping = backend_supports_cpu()
            if self.async_bookkeeping:
                if not backend_supports_cpu():
                    raise OLMoConfigurationError(
                        "A CPU-only backend is required for async bookkeeping"
                    )
                log.info("Creating new process group for async bookkeeping")
                self._bookkeeping_pg = dist.new_group()

        # Check data loader configuration.
        if self.data_loader.dp_world_size != get_world_size(self.dp_process_group):
            raise OLMoConfigurationError(
                "data loader's DP world size appears to be configured incorrectly, "
                f"got {self.data_loader.dp_world_size}, expected {get_world_size(self.dp_process_group)}."
            )
        if self.data_loader.dp_rank != get_rank(self.dp_process_group):
            raise OLMoConfigurationError(
                "data loader's DP rank appears to be configured incorrectly, "
                f"got {self.data_loader.dp_rank}, expected {get_rank(self.dp_process_group)}."
            )
        if self.data_loader.fs_local_rank != get_fs_local_rank():
            raise OLMoConfigurationError(
                "data loader's FS local rank appears to be configured incorrectly, "
                f"got {self.data_loader.fs_local_rank}, expected {get_fs_local_rank()}."
            )

        for callback in self.callbacks.values():
            callback.post_attach()

        self.train_module._attach_trainer(self)

    @property
    def block_ephemeral_checkpoints(self) -> bool:
        return len(self._blocking_ephemeral_checkpoints) > 0

    @property
    def global_batch_size(self) -> int:
        """
        Global training batch size *in tokens*.
        """
        return self.data_loader.global_batch_size

    @property
    def rank_batch_size(self) -> int:
        """
        The number of tokens in each training batch per rank.
        """
        if self._rank_batch_size is None:
            assert self.global_batch_size % get_world_size(self.dp_process_group) == 0
            self._rank_batch_size = self.global_batch_size // get_world_size(self.dp_process_group)
        return self._rank_batch_size

    @property
    def training_complete(self) -> bool:
        if self._error is not None:
            raise RuntimeError("An error occurred") from self._error

        if (
            not self._canceled
            and self.global_step > 0
            and self.global_step % self.cancel_check_interval == 0
        ):
            self.check_if_canceled()

        if self.is_canceled:
            return True
        elif self._duration_due(self.max_duration):
            return True
        elif self.hard_stop is not None and self._duration_due(self.hard_stop):
            return True
        else:
            return False

    @property
    def is_canceled(self) -> bool:
        if self._error is not None:
            raise RuntimeError("An error occurred") from self._error
        return self._canceled

    @property
    def tokens_per_batch(self) -> int:
        """
        The number of tokens in each training batch.
        """
        return self.global_batch_size

    @property
    def steps_per_epoch(self) -> Optional[int]:
        """
        The total number of training steps in an epoch, if known.
        """
        return self.data_loader.total_batches

    @property
    def tokens_per_epoch(self) -> Optional[int]:
        """
        The total number of tokens in the training dataset, minus left-overs.
        """
        if self.steps_per_epoch is not None:
            return self.steps_per_epoch * self.tokens_per_batch
        else:
            return None

    @property
    def max_steps(self) -> Optional[int]:
        """
        The maximum number of steps to train for, as determined by :data:`max_duration`.
        """
        return self._get_max_steps(self.max_duration)

    @property
    def max_tokens(self) -> Optional[int]:
        """
        The maximum number of tokens to train for, as determined by :data:`max_duration`.
        """
        return self._get_max_tokens(self.max_duration)

    def convert_duration_to_steps(self, duration: Duration) -> int:
        """Convert a duration to steps."""
        if duration.unit == DurationUnit.epochs:
            if self.steps_per_epoch is None:
                raise RuntimeError(
                    "the number of steps cannot be determined from an 'epochs' duration since "
                    "the data loader's number of batches is unknown"
                )
            return self.steps_per_epoch * duration.value
        elif duration.unit == DurationUnit.steps:
            return duration.value
        elif duration.unit == DurationUnit.tokens:
            raise RuntimeError("the number of steps cannot be determined from a 'tokens' duration")
        else:
            raise NotImplementedError(f"Unsupported duration unit: {duration.unit}")

    def _get_max_steps(self, duration: Duration) -> Optional[int]:
        if duration.unit == DurationUnit.steps:
            return duration.value
        elif duration.unit == DurationUnit.epochs:
            if self.data_loader.total_batches is None:
                return None
            max_epochs = duration.value
            # NOTE: need to cover the case where the last epoch has just ended and we've incremented
            # self.epoch.
            steps_remaining_this_epoch = (
                0
                if self.epoch > max_epochs
                else max(self.data_loader.total_batches - self.data_loader.batches_processed, 0)
            )
            steps_remaining = steps_remaining_this_epoch
            for e in range(self.epoch + 1, duration.value + 1):
                if (b := self.data_loader.batches_in_epoch(e)) is not None:
                    steps_remaining += b
                else:
                    return None
            return self.global_step + steps_remaining
        elif duration.unit == DurationUnit.tokens:
            # Need to account for a change in batch size.
            max_tokens = duration.value
            tokens_remaining = max(max_tokens - self.global_train_tokens_seen, 0)
            steps_remaining = math.ceil(tokens_remaining / self.tokens_per_batch)
            return self.global_step + steps_remaining
        else:
            raise NotImplementedError

    def _get_max_tokens(self, duration: Duration) -> Optional[int]:
        if duration.unit == DurationUnit.tokens:
            return duration.value
        else:
            max_steps = self._get_max_steps(duration)
            if max_steps is None:
                return None
            steps_remaining = max(max_steps - self.global_step, 0)
            tokens_remaining = steps_remaining * self.tokens_per_batch
            return self.global_train_tokens_seen + tokens_remaining

    @property
    def bookkeeping_device(self) -> torch.device:
        """
        The device used for collective bookkeeping (non-training) operations that can potentially.
        use a different backend.
        """
        if self.async_bookkeeping and backend_supports_cpu():
            return torch.device("cpu")
        else:
            return self.device

    @property
    def bookkeeping_pg(self) -> Optional[dist.ProcessGroup]:
        """
        The process group used for bookkeeping collectives.

        Since bookkeeping collectives might be done in a separate thread, we need a separate process
        group to avoid potential race conditions.
        """
        return self._bookkeeping_pg

    @property
    def multi_thread_pool(self) -> ThreadPoolExecutor:
        """
        A multi-threaded executor for bookkeeping tasks that don't involve distributed communication.
        """
        if self._multi_thread_pool is None:
            self._multi_thread_pool = ThreadPoolExecutor(
                max_workers=get_default_thread_count(),
                thread_name_prefix="trainer-multi-thread-pool",
            )
        return self._multi_thread_pool

    @property
    def single_thread_pool(self) -> ThreadPoolExecutor:
        """
        A single-threaded executor for bookkeeping tasks that involve distributed communication.
        """
        if self._single_thread_pool is None:
            self._single_thread_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="trainer-single-thread-pool"
            )
        return self._single_thread_pool

    @property
    def checkpoint_loaded(self) -> bool:
        """
        If a checkpoint has been loaded.
        """
        return self._checkpoint_loaded

    @property
    def training_progress(self) -> TrainingProgress:
        # Calculate total steps.
        total_steps = (
            self._get_max_steps(self.hard_stop) if self.hard_stop is not None else self.max_steps
        )
        if total_steps is not None:
            total_steps = max(total_steps, self.global_step)

        # Get current speed in batches per second.
        bps: Optional[float] = None
        tps: Optional[float] = None
        mfu: Optional[float] = None
        for callback in self._iter_callbacks():
            if isinstance(callback, SpeedMonitorCallback):
                bps = callback.bps_avg
                tps = callback.tps_avg
                mfu = callback.mfu_avg
                break

        # Estimate the remaining time.
        time_remaining: Optional[timedelta] = None
        if (
            bps is not None
            and total_steps is not None
            and (steps_remaining := (total_steps - self.global_step)) > 0
        ):
            seconds_remaining = steps_remaining / bps
            # Round to nearest minute.
            minutes_remaining = 1 + (seconds_remaining // 60)
            time_remaining = timedelta(minutes=minutes_remaining)

        return TrainingProgress(
            current_step=self.global_step,
            current_tokens=self.global_train_tokens_seen,
            total_steps=total_steps,
            time_remaining=time_remaining,
            bps=bps,
            tps=tps,
            mfu=mfu,
        )

    def get_callback_name(self, callback: Callback) -> str:
        for name, cb in self.callbacks.items():
            if cb is callback:
                return name
        raise ValueError("callback not registered with trainer!")

    def cancel_run(self, reason: str, no_sync: bool = False):
        """
        Mark the run canceled.

        :param reason: The reason for canceling.
        :param no_sync: Set this to ``True`` only if you're calling this from all ranks at the same
            time, otherwise you'll get a distributed deadlock.
        """
        if self.is_canceled:
            return

        self._canceling_rank = get_rank()
        self._cancel_reason = reason
        if no_sync:
            self._canceled = True
            log.warning(f"Run canceled from all ranks. Reason: {reason}")
            barrier()

    def check_if_canceled(self):
        """
        Asynchronously check if the run is canceled. Use :data:`is_canceled` to see the result.
        This needs to be called by all ranks at the same point in the training loop.
        """
        # NOTE: Do not set `allow_multiple` to `False` here!
        # That could result in a situation where this op is canceled on one rank while it's running
        # on another rank, leading to a deadlock.
        self.run_bookkeeping_op(self._check_if_canceled)

    def fit(self):
        """
        Fit the model, potentially loading a checkpoint first depending on the
        :data:`load_strategy`.
        """
        self._canceled = False
        self._cancel_reason = None
        self._canceling_rank = None

        # Maybe load a checkpoint.
        if (
            not self.no_checkpoints
            and not self.checkpoint_loaded
            and self.load_strategy != LoadStrategy.never
        ):
            # Try loading from the save folder first. The save folder is used for continuing
            # existing runs that failed or were preempted, so we always load trainer state and
            # optimizer state.
            self.maybe_load_checkpoint(
                self.save_folder, load_trainer_state=True, load_optim_state=True
            )

            # Then fallback to the load path, if provided.
            if self.load_path is not None:
                if not self.checkpoint_loaded:
                    self.maybe_load_checkpoint(self.load_path)
                else:
                    log.warning(
                        f"Ignoring load path ('{self.load_path}') since checkpoint was found in save folder"
                    )

            if not self.checkpoint_loaded:
                if self.load_strategy == LoadStrategy.always:
                    raise FileNotFoundError(
                        f"No checkpoint found in save folder ('{self.save_folder}') or "
                        f"load path ('{self.load_path}')"
                    )
                else:
                    log.warning(
                        f"No checkpoint found in save folder ('{self.save_folder}') or "
                        f"load path ('{self.load_path}'), will train from scratch..."
                    )

        barrier()

        log.info("Callback order:")
        for i, callback_name in enumerate(self.callbacks.keys()):
            log.info(f"  - Callback {i + 1}: {callback_name}")

        if self.max_steps is not None:
            log.info(f"Training for {self.max_steps:,d} steps")

        # Install SIGTERM + SIGINT handlers.
        og_sigterm_handler = signal.signal(signal.SIGTERM, self._handle_os_signal)
        og_sigint_handler = signal.signal(signal.SIGINT, self._handle_os_signal)

        try:
            for callback in self._iter_callbacks():
                callback.pre_train()
            self.train_module.pre_train()

            # Quick check if the run has already been canceled.
            if self.is_canceled:
                for callback in self._iter_callbacks():
                    callback.post_train()
                self._shutdown()
                return

            # Do a dry-run for compiling and catching OOMs early.
            if not self.training_complete:
                self._dry_run_batch()

            # Iterate over epochs until done.
            while not self.training_complete:
                self._fit_epoch()
        except BaseException as exc:
            self._error = exc
            log.error(f"Training failed due to:\n{type(exc).__name__}: {exc}")
            for callback in self._iter_callbacks():
                callback.on_error(exc)
            # Shutdown ungracefully, i.e. without waiting on bookkeeping ops to finish and without
            # issuing additional distributed collectives because those could result in a deadlock
            # considering we're already in a bad/inconsistent state across ranks.
            self._shutdown(gracefully=False)
            raise
        finally:
            # Restore original signal handlers.
            signal.signal(signal.SIGTERM, og_sigterm_handler)
            signal.signal(signal.SIGINT, og_sigint_handler)

        for callback in self._iter_callbacks():
            callback.post_train()

        # Wait for any bookkeeping tasks to finish.
        self._shutdown()
        log.info("Training complete")

    def eval_checkpoints(self):
        """
        Evaluate each checkpoint in :data:`checkpoints_to_eval` with the configured
        :class:`~olmo_core.train.callbacks.EvaluatorCallback`\\ s, without training.

        Loads each checkpoint in turn, runs every evaluator, and logs the results. Checkpointer and
        evaluator callbacks' ``pre_train``/``post_train`` hooks are skipped (only the evaluators'
        ``perform_eval()`` runs).

        :raises OLMoConfigurationError: If ``no_evals=True``, no ``EvaluatorCallback`` is configured,
            or ``checkpoints_to_eval`` is unset.
        """
        if self.no_evals:
            raise OLMoConfigurationError(
                f"'{self.__class__.__name__}.eval_checkpoints()' requires 'no_evals=False'"
            )

        checkpoint_paths = self._get_checkpoints_to_eval()
        evaluator_callbacks = [
            cb for cb in self._iter_callbacks() if isinstance(cb, EvaluatorCallback)
        ]
        if not evaluator_callbacks:
            raise OLMoConfigurationError(
                f"'{self.__class__.__name__}.eval_checkpoints()' requires at least one "
                f"'{EvaluatorCallback.__name__}'"
            )

        self._canceled = False
        self._cancel_reason = None
        self._canceling_rank = None

        log.info("Callback order:")
        for i, callback_name in enumerate(self.callbacks.keys()):
            log.info(f"  - Callback {i + 1}: {callback_name}")

        log.info(f"Evaluating {len(checkpoint_paths):,d} checkpoints")

        callback_blacklist = (CheckpointerCallback, EvaluatorCallback)

        # Install SIGTERM + SIGINT handlers.
        og_sigterm_handler = signal.signal(signal.SIGTERM, self._handle_os_signal)
        og_sigint_handler = signal.signal(signal.SIGINT, self._handle_os_signal)

        try:
            for callback in self._iter_callbacks():
                if not isinstance(callback, callback_blacklist):
                    callback.pre_train()
            self.train_module.pre_train()

            if self.is_canceled:
                for callback in self._iter_callbacks():
                    if not isinstance(callback, callback_blacklist):
                        callback.post_train()
                self._shutdown()
                return

            for checkpoint_num, checkpoint_path in enumerate(checkpoint_paths, start=1):
                if self.is_canceled:
                    break

                log.info(
                    f"Evaluating checkpoint {checkpoint_num:,d}/{len(checkpoint_paths):,d} "
                    f"from '{checkpoint_path}'..."
                )
                # Offline eval only needs model weights: skip optimizer state (never needed, and
                # eval-only/weights-only checkpoints have none), and leave trainer state at the
                # trainer's default (load-if-present) so it isn't required either.
                self.load_checkpoint(checkpoint_path, load_optim_state=False)

                self.record_metric("throughput/total tokens", self.global_train_tokens_seen)
                # Duck-typed so this stays train-module-agnostic (transformer train modules expose
                # these; the base TrainModule doesn't).
                num_flops_fn = getattr(self.train_module, "num_flops_per_token", None)
                max_seq_len = getattr(self.train_module, "max_sequence_length", None)
                if callable(num_flops_fn) and max_seq_len is not None:
                    num_flops_per_token = num_flops_fn(max_seq_len)
                    if num_flops_per_token is not None:
                        self.record_metric(
                            "throughput/total petaflops",
                            self.global_train_tokens_seen * num_flops_per_token / 1e15,
                        )

                for callback in evaluator_callbacks:
                    callback.perform_eval()

                self._log_metrics()
        except BaseException as exc:
            log.error(f"Checkpoint evaluation failed due to:\n{exc}")
            for callback in self._iter_callbacks():
                if not isinstance(callback, callback_blacklist):
                    callback.on_error(exc)
            for callback in self._iter_callbacks():
                callback.close()
            raise
        finally:
            # Restore original signal handlers.
            signal.signal(signal.SIGTERM, og_sigterm_handler)
            signal.signal(signal.SIGINT, og_sigint_handler)

        # Flush metrics + await queued bookkeeping ops before the post-train hooks run.
        self._log_metrics()
        self._join_bookkeeping_ops()
        for callback in self._iter_callbacks():
            if not isinstance(callback, callback_blacklist):
                callback.post_train()

        self._shutdown()
        log.info("Checkpoint evaluation complete")

    def _get_checkpoints_to_eval(self) -> List[str]:
        if not self.checkpoints_to_eval:
            raise OLMoConfigurationError(
                f"'{self.__class__.__name__}.eval_checkpoints()' requires "
                "'checkpoints_to_eval' to be set"
            )

        checkpoint_paths: List[str] = []
        error_type: Optional[str] = None
        error_msg: Optional[str] = None

        if get_rank() == 0:
            seen = set()
            try:
                for raw_path in self.checkpoints_to_eval:
                    path = normalize_path(raw_path)
                    candidate_paths = sorted(glob_directory(path)) if "*" in path else [path]
                    if not candidate_paths:
                        raise FileNotFoundError(f"No checkpoints found in '{path}'")

                    expanded_paths: List[str] = []
                    for candidate_path in candidate_paths:
                        if self.checkpointer.dir_is_checkpoint(candidate_path):
                            expanded_paths.append(candidate_path)
                        else:
                            expanded_paths.extend(
                                checkpoint_path
                                for _, checkpoint_path in sorted(
                                    self.checkpointer.find_checkpoints(candidate_path),
                                    key=lambda item: item[0],
                                )
                            )

                    if not expanded_paths:
                        raise FileNotFoundError(f"No checkpoints found in '{path}'")

                    for checkpoint_path in expanded_paths:
                        if checkpoint_path not in seen:
                            checkpoint_paths.append(checkpoint_path)
                            seen.add(checkpoint_path)
            except FileNotFoundError as exc:
                error_type = "FileNotFoundError"
                error_msg = str(exc)
            except OLMoConfigurationError as exc:
                error_type = "OLMoConfigurationError"
                error_msg = str(exc)
            except Exception as exc:
                error_type = exc.__class__.__name__
                error_msg = str(exc)

        checkpoint_paths, error_type, error_msg = broadcast_object(
            (checkpoint_paths, error_type, error_msg)
        )

        if error_type == "FileNotFoundError":
            assert error_msg is not None
            raise FileNotFoundError(error_msg)
        elif error_type == "OLMoConfigurationError":
            assert error_msg is not None
            raise OLMoConfigurationError(error_msg)
        elif error_type is not None:
            assert error_msg is not None
            raise RuntimeError(error_msg)

        if checkpoint_paths and all(
            (name := Path(path).name).startswith("step") and name[4:].isdigit()
            for path in checkpoint_paths
        ):
            checkpoint_paths = sorted(checkpoint_paths, key=lambda path: int(Path(path).name[4:]))

        return checkpoint_paths

    def _shutdown(self, gracefully: bool = True):
        if gracefully:
            self._log_metrics()
            self._join_bookkeeping_ops()
            if self._multi_thread_pool is not None:
                self._multi_thread_pool.shutdown(wait=True, cancel_futures=False)
                self._multi_thread_pool = None
            if self._single_thread_pool is not None:
                self._single_thread_pool.shutdown(wait=True, cancel_futures=False)
                self._single_thread_pool = None

        # NOTE: '.close' must be called after shutting down thread pools to ensure bookkeeping ops
        # have finished first. See https://github.com/allenai/Olmo-core/pull/546.
        for callback in self._iter_callbacks():
            callback.close()

        if gracefully:
            barrier()

        gc_cuda()

    def state_dict(self) -> TrainerStateDict:
        """
        Get the trainer state to save.
        """
        return {
            "global_step": self.global_step,
            "global_train_tokens_seen": self.global_train_tokens_seen,
            "global_train_petaflops": self.global_train_petaflops,
            "max_steps": self.max_steps,
            "data_loader": self.data_loader.state_dict(),
            "epoch": self.epoch,
            "world_size": get_world_size(),  # global world size here on purpose
            "rng": EnvRngStates.current_state().as_dict(),
            "callbacks": {k: cb.state_dict() for k, cb in self.callbacks.items()},
        }

    def load_state_dict(self, state_dict: TrainerStateDict):
        """
        Load trainer state (not model or optimizer state).
        """
        # For backwards compatibility.
        if "data_loader" not in state_dict:
            if "dataset" in state_dict:
                state_dict["data_loader"] = state_dict.pop("dataset")
                state_dict["data_loader"]["epoch"] = state_dict["epoch"]
            else:
                state_dict["data_loader"] = {
                    "dataset_type": "fsl",
                    "dataset_fingerprint_version": state_dict.pop("dataset_fingerprint_version"),
                    "dataset_fingerprint": state_dict.pop("dataset_fingerprint"),
                    "tokens_processed": state_dict["global_train_tokens_seen_this_epoch"],
                    "batches_processed": state_dict["global_train_tokens_seen_this_epoch"]
                    // self.global_batch_size,
                    "sequence_length": state_dict.pop("train_sequence_length"),
                    "max_target_sequence_length": state_dict.pop("max_train_sequence_length"),
                    "seed": state_dict["data_seed"],
                    "epoch": state_dict["epoch"],
                }

        self.data_loader.load_state_dict(state_dict["data_loader"])
        self.global_step = state_dict["global_step"]
        self.global_train_tokens_seen = state_dict["global_train_tokens_seen"]
        self.global_train_petaflops = state_dict.get("global_train_petaflops", 0.0)
        self.epoch = state_dict["epoch"]

        for cb_name, cb_state in state_dict.get("callbacks", {}).items():
            if (cb := self.callbacks.get(cb_name)) is not None:
                cb.load_state_dict(cb_state)

        log.info(f"Will resume training from step {self.global_step}, epoch {self.epoch}")

        if state_dict["world_size"] == get_world_size():  # global world size here on purpose
            rng_state = EnvRngStates.from_dict(state_dict["rng"])
            if not rng_state.restore():
                log.warning(
                    "Some RNG states were not restored due to differences in library versions"
                )
        else:
            log.warning(
                "Trainer will not restore rank RNG states since the RNG states in the checkpoint "
                "were saved with a different world size."
            )

    def load_checkpoint(
        self,
        dir: PathOrStr,
        *,
        load_trainer_state: Optional[bool] = None,
        load_optim_state: Optional[bool] = None,
        reset_optimizer_states_on_load: Optional[bool] = None,
    ):
        """
        Load a checkpoint.

        .. note::
            :meth:`fit()` may call this method automatically depending on the :data:`load_strategy`.

        :param dir: The path/URL to a checkpoint or a folder of checkpoints.
        :param load_trainer_state: Load trainer state (data loader state, RNG states, and other bookkeeping).
        :param load_optim_state: Load optimizer state in the train module.
        :param reset_optimizer_states_on_load: Override whether optimizer state should be reset for
            this checkpoint load. When left ``None``, :meth:`Checkpointer.load` defaults it to the
            train module's ``reset_optimizer_states_on_resume`` setting on an actual resume (i.e.
            only when trainer state is found in the checkpoint).
        """
        load_trainer_state = (
            self.load_trainer_state if load_trainer_state is None else load_trainer_state
        )
        load_optim_state = self.load_optim_state if load_optim_state is None else load_optim_state
        if dir == self.save_folder:
            if load_trainer_state is False:
                log.warning(
                    "Loading from save_folder with 'load_trainer_state=False' is not recommended, "
                    "since the save_folder is meant for continuing existing runs."
                )
            if load_optim_state is False:
                log.warning(
                    "Loading from save_folder with 'load_optim_state=False' is not recommended, "
                    "since the save_folder is meant for continuing existing runs."
                )

        dir = normalize_path(dir)

        # NOTE: to avoid making a ton of client requests (S3 or otherwise) we only make those
        # requests from rank 0 then scatter the result to the other ranks.
        dir_to_scatter: Optional[PathOrStr] = dir
        error: Optional[Exception] = None
        if get_rank() == 0 and not self.checkpointer.dir_is_checkpoint(dir):
            # Try to find the latest checkpoint in the directory.
            try:
                dir_to_scatter = self.checkpointer.latest_checkpoint(dir)
            except FileNotFoundError as e:  # defer raising until after the scatter
                dir_to_scatter, error = None, e
        dir_to_scatter = broadcast_object(dir_to_scatter)
        if dir_to_scatter is None:
            if error is None:
                raise FileNotFoundError(f"No checkpoints found in '{dir}'")
            raise error
        dir = dir_to_scatter

        log.info(f"Loading checkpoint from '{dir}'...")
        load_start = time.perf_counter()
        trainer_state = self.checkpointer.load(
            dir,
            self.train_module,
            load_trainer_state=load_trainer_state,
            load_optim_state=load_optim_state,
            reset_optimizer_states_on_load=reset_optimizer_states_on_load,
        )
        if trainer_state is not None:
            self.load_state_dict(cast(TrainerStateDict, trainer_state))

        self.record_metric(
            "checkpoint/load_duration_s",
            time.perf_counter() - load_start,
            reduce_type=ReduceType.max,
        )

        for callback in self._iter_callbacks():
            callback.post_checkpoint_loaded(dir)

        self._checkpoint_loaded = True
        log.info("Checkpoint successfully loaded")

    def maybe_load_checkpoint(
        self,
        dir: Optional[PathOrStr] = None,
        *,
        load_trainer_state: Optional[bool] = None,
        load_optim_state: Optional[bool] = None,
        reset_optimizer_states_on_load: Optional[bool] = None,
    ) -> bool:
        """
        Like :meth:`load_checkpoint()` but is a no-op if there is no checkpoint in the ``dir`` provided.

        .. note::
            :meth:`fit()` may call this method automatically depending on the :data:`load_strategy`.

        :returns: If a checkpoint was loaded.
        """
        if dir is None:
            dir = self.save_folder
        should_load: bool = True
        if get_rank() == 0:
            should_load = self.checkpointer.contains_checkpoint(dir)
        should_load = broadcast_object(should_load)
        if should_load:
            self.load_checkpoint(
                dir,
                load_trainer_state=load_trainer_state,
                load_optim_state=load_optim_state,
                reset_optimizer_states_on_load=reset_optimizer_states_on_load,
            )
            assert self.checkpoint_loaded
            return True
        else:
            return False

    def save_checkpoint(self, ephemeral: bool = False) -> PathOrStr:
        """
        Save a checkpoint for the current step to the :data:`save_folder`.

        :param ephemeral: Whether to mark the checkpoint as ephemeral in its metadata.
          Note that the trainer itself won't remove ephemeral checkpoints.
          That's up to the :class:`CheckpointerCallback`.

        :returns: The path/URL to the checkpoint.
        """
        dirname = self.checkpointer.checkpoint_dirname(self.global_step)
        path = join_path(self.save_folder, dirname)

        log.info(f"Saving checkpoint for step {self.global_step} to '{path}'...")
        save_start = time.perf_counter()
        # Some state (e.g. in a callback) might be tied to metrics, or depend on another bookkeeping
        # operation to be finished first.
        self._log_metrics()
        self._join_bookkeeping_ops()

        self.checkpointer.save(
            path,
            self.train_module,
            cast(Dict[str, Any], self.state_dict()),
            ephemeral=ephemeral,
        )
        self.record_metric(
            "checkpoint/save_duration_s",
            time.perf_counter() - save_start,
            reduce_type=ReduceType.max,
        )
        for callback in self._iter_callbacks():
            callback.post_checkpoint_saved(path)
        log.info("Checkpoint saved")
        return path

    def save_checkpoint_async(self, ephemeral: bool = False) -> Tuple[PathOrStr, Future]:
        """
        Save a checkpoint for the current step to the :data:`save_folder` asynchronously.

        :param ephemeral: Whether to mark the checkpoint as ephemeral in its metadata.
          Note that the trainer itself won't remove ephemeral checkpoints.
          That's up to the :class:`CheckpointerCallback`.

        :returns: The path/URL to the checkpoint and a future which will complete when the
            checkpoint is successfully saved.
        """
        step = self.global_step
        dirname = self.checkpointer.checkpoint_dirname(step)
        path = join_path(self.save_folder, dirname)

        log.info(f"Saving checkpoint for step {step} to '{path}' asynchronously...")
        save_start = time.perf_counter()
        # Some state (e.g. in a callback) might be tied to metrics, or depend on another bookkeeping
        # operation to be finished first.
        self._log_metrics()
        self._join_bookkeeping_ops()

        fut = self.checkpointer.save_async(
            path,
            self.train_module,
            cast(Dict[str, Any], self.state_dict()),
            ephemeral=ephemeral,
        )

        def callback(future: Future):
            future.result()  # ensure it finished successfully
            self.record_metric(
                "checkpoint/save_async_duration_s",
                time.perf_counter() - save_start,
                reduce_type=ReduceType.max,
            )
            for callback in self._iter_callbacks():
                callback.post_checkpoint_saved(path)
            log.info(f"Checkpoint for step {step} saved successfully")

        fut.add_done_callback(callback)

        return path, fut

    def record_metric(
        self,
        name: str,
        value: Union[float, torch.Tensor],
        reduce_type: Optional[ReduceType] = None,
        namespace: Optional[str] = None,
        merge_strategy: MetricMergeStrategy = MetricMergeStrategy.warn,
    ):
        """
        Record a new metric for the current step.

        .. seealso::
            Use :meth:`record_ce_loss()` to record the cross-entropy loss, specifically.

        :param name: The name of the metric.
        :param value: The value of the metric.
        :param reduce_type: Specifies how to reduce the metric across the distributed process group.
            ``None`` means no reduction.
        :param namespace: A namespace to record the metric under, i.g. "train" or "optim".
        :param merge_strategy: How to merge metrics when duplicates are logged.
        """
        if namespace is not None:
            name = f"{namespace.rstrip('/')}/{name.lstrip('/')}"

        value = _metric_value_to_tensor(value)

        if self.global_step not in self._metrics:
            self._metrics[self.global_step] = OrderedDict()

        step_metrics = self._metrics[self.global_step]

        if name not in step_metrics or merge_strategy == MetricMergeStrategy.latest:
            step_metrics[name] = value
        elif merge_strategy == MetricMergeStrategy.sum:
            step_metrics[name] = step_metrics[name] + value
        elif merge_strategy == MetricMergeStrategy.mean:
            step_metrics[name] = (step_metrics[name] + value) / 2
        elif merge_strategy == MetricMergeStrategy.max:
            step_metrics[name] = torch.max(step_metrics[name], value.to(step_metrics[name].device))
        elif merge_strategy == MetricMergeStrategy.min:
            step_metrics[name] = torch.min(step_metrics[name], value.to(step_metrics[name].device))
        elif merge_strategy == MetricMergeStrategy.warn:
            log.warning(
                f"Attempting to log duplicate metric '{name}' for step {(self.global_step)}. "
                "The latest value will be ignored."
            )
        elif merge_strategy == MetricMergeStrategy.oldest:
            pass
        else:
            raise NotImplementedError(merge_strategy)

        # reduce type must be consistent to avoid issues
        if name in self._metrics_reduce_type and self._metrics_reduce_type[name] != reduce_type:
            raise RuntimeError(
                f"expected '{self._metrics_reduce_type[name]}' reduce type for metric '{name}' "
                f"based on last record, but got '{reduce_type}' this time"
            )
        self._metrics_reduce_type[name] = reduce_type

    def record_ce_loss(
        self, value: Union[float, torch.Tensor], reduce_type: Optional[ReduceType] = None
    ):
        """
        Record the cross-entropy loss metric specifically.
        """
        return self.record_metric(TRAIN_CE_LOSS_METRIC, value, reduce_type=reduce_type)

    def get_metric(self, name: str, namespace: Optional[str] = None) -> Optional[torch.Tensor]:
        """
        Get the value of a metric recorded during the current step.

        .. warning::
            Metrics will only be available from the time they're recorded until the end of the
            current step.

        .. warning::
            Accessing a metric can inadvertently trigger a host-device sync, which slows down
            training.

        :param name: The name of the metric.
        """
        if self.global_step not in self._metrics:
            return None
        if namespace is not None:
            name = f"{namespace.rstrip('/')}/{name.lstrip('/')}"
        return self._metrics[self.global_step].get(name)

    def write_file(
        self, name: str, contents: Union[str, bytes], dir: Optional[PathOrStr] = None
    ) -> PathOrStr:
        """
        Write a file to the :data:`save_folder` or ``dir``, if provided.

        :param fname: The name of the file to write, relative to the :data:`save_folder` or ``dir``.
        :param contents: The contents of the file to write.
        :param dir: The path/URL to a directory to write the file to. Defaults to :data:`save_folder`.

        :returns: The path/URL of the file.
        """
        return self.checkpointer.write_file(dir or self.save_folder, name, contents)

    def persist_working_file(self, name: PathOrStr) -> PathOrStr:
        """
        Persist a file in the :data:`work_dir` by saving/uploading it to the :data:`save_folder`.

        :param name: The name/path of the file *relative* to the :data:`work_dir`.

        :returns: The full path/URL to the saved file.

        :raises FileNotFoundError: If the file can't be found.
        :raises FileExistsError: If the file already exists in the save folder and :data:`save_overwrite`
            is ``False``.
        """
        if Path(name).is_relative_to(self.work_dir):
            name = Path(name).relative_to(self.work_dir)
        source = join_path(self.work_dir, name)
        target = join_path(self.save_folder, name)
        if source != target:
            copy_file(source, target, save_overwrite=self.save_overwrite)
        elif not file_exists(source):
            raise FileNotFoundError(source)
        return target

    def persist_working_subdir(self, name: PathOrStr) -> PathOrStr:
        """
        Persist a directory in the :data:`work_dir` by saving/uploading it to the :data:`save_folder`.

        :param name: The name/path of the directory relative to the :data:`work_dir`.

        :returns: The full path/URL to the saved directory.

        :raises FileNotFoundError: If the directory can't be found.
        :raises FileExistsError: If any files in the directory already exist in the save folder and :data:`save_overwrite`
            is ``False``.
        """
        if Path(name).is_relative_to(self.work_dir):
            name = Path(name).relative_to(self.work_dir)
        source = join_path(self.work_dir, name)
        target = join_path(self.save_folder, name)
        if source != target:
            copy_dir(source, target, save_overwrite=self.save_overwrite)
        elif not Path(source).exists():
            raise FileNotFoundError(f"Source '{source}' does not exist")
        elif not Path(source).is_dir():
            raise FileNotFoundError(f"Source '{source}' exists but is not a directory")
        elif dir_is_empty(source):
            raise FileNotFoundError(f"Source directory '{source}' is empty")
        return target

    def add_callback(self, name: str, callback: Callback):
        """
        Add a callback to the trainer.
        """
        if name in self.callbacks:
            raise OLMoConfigurationError(f"A callback with name '{name}' already exists!")
        callback.trainer = self
        self.callbacks[name] = callback
        self._sort_callbacks()
        callback.post_attach()

    def has_callback(self, cb_class: Type[Callback]) -> bool:
        """
        Check if the trainer already has a registered instance of the given callback class.
        """
        for cb in self.callbacks.values():
            if isinstance(cb, cb_class):
                return True
        return False

    def _sort_callbacks(self):
        self.callbacks = OrderedDict(
            (
                (k, cb)
                for _, (k, cb) in sorted(
                    enumerate(self.callbacks.items()),
                    key=lambda x: (x[1][1].priority, -1 * x[0]),
                    reverse=True,
                )
            )
        )

    def _iter_callbacks(self) -> Iterable[Callback]:
        callbacks: Iterable[Callback] = self.callbacks.values()
        if self.no_checkpoints:
            callbacks = filter(lambda cb: not isinstance(cb, CheckpointerCallback), callbacks)
        if self.no_evals:
            callbacks = filter(lambda cb: not isinstance(cb, EvaluatorCallback), callbacks)
        return callbacks

    def _duration_due(self, duration: Duration) -> bool:
        return duration.due(
            step=self.global_step, tokens=self.global_train_tokens_seen, epoch=self.epoch
        )

    def _handle_os_signal(self, signalnum, stack_frame):
        del stack_frame

        signame: Optional[str] = None
        if signalnum == signal.SIGTERM:
            signame = "SIGTERM"
        elif signalnum == signal.SIGINT:
            signame = "SIGINT"

        msg: str
        if signame is not None:
            msg = f"{signame} received"
        else:
            msg = f"Sig({signalnum}) received"

        log.warning(msg)
        self.cancel_run(msg)

    def run_bookkeeping_op(
        self,
        op: Callable[..., T],
        *args,
        cb: Optional[Callable[[T], None]] = None,
        op_name: Optional[str] = None,
        cancel_in_progress: Optional[bool] = None,  # deprecated
        allow_multiple: bool = True,
        soft_timeout: Optional[int] = None,
        distributed: bool = True,
        **kwargs,
    ):
        """
        Run a bookkeeping operation, potentially in a background thread.

        :param op: The operation to run.
        :param args: Positional arguments to pass to the operation.
        :param kwargs: Keyword arguments to pass to the operation.
        :param cb: A callback to call with the result of the operation when it finishes.
        :param op_name: A name for the operation, used for logging, debugging, and potentially canceling
            old invocations of the same operation when ``allow_multiple`` is ``False``.
        :param allow_multiple: If ``False``, only one bookkeeping operation with the given name is allowed
            to run, so if there are other ops with the same name that are queued, those will be canceled,
            and if there's another one that's already running, the current invocation will be ignored.
        :param soft_timeout: A soft timeout, in seconds, to wait for the operation to finish. If the op
            takes longer than this a warning will be issued.
        :param distributed: This should only be set to ``False`` if the op doesn't use distributed
            communication, in which case it will be allowed to run concurrently with other ops.
        """
        if cancel_in_progress is not None:
            warnings.warn(
                "'cancel_in_progress' argument to 'Trainer.run_bookkeeping_op' is deprecated, use 'allow_multiple' instead",
                DeprecationWarning,
            )
            allow_multiple = not cancel_in_progress

        if op_name is None:
            op_name = op.__qualname__

        if soft_timeout is None:
            soft_timeout = self.bookkeeping_soft_timeout

        def wrapped_op(*args, **kwargs):
            start_time = time.perf_counter()
            assert soft_timeout is not None  # for mypy
            try:
                result = op(*args, **kwargs)
                # NOTE: invoke cb inside wrapped_op (before the future is marked as FINISHED)
                # so that _join_bookkeeping_ops() waits for it to complete. Previously cb was
                # invoked via future.add_done_callback() which runs *after* the future is
                # FINISHED, causing a race where state_dict() could capture stale callback state.
                if cb is not None:
                    cb(result)
                return result
            finally:
                if (runtime := int(time.perf_counter() - start_time)) > soft_timeout:
                    log.warning(
                        f"Bookeeping op '{op_name}' took longer than {soft_timeout} "
                        f"seconds ({runtime:,d} seconds)!"
                    )

        if not distributed or (
            self.async_bookkeeping
            and self.bookkeeping_device.type == "cpu"
            and self.bookkeeping_pg is not None
        ):
            if not allow_multiple:
                for op_id in list(self._bookkeeping_queue[op_name].keys()):
                    future = self._bookkeeping_queue[op_name][op_id]
                    if future.cancel() or future.done():
                        self._bookkeeping_queue[op_name].pop(op_id)
                    else:
                        log.warning(
                            f"Attempted to submit bookkeeping op '{op_name}' while a previous invocation was already in progress. "
                            "Since 'allow_multiple' is set to 'False' for this op, the current invocation will be canceled.\n"
                            "If you see this message frequently, the op in question may be taking longer than expected or is "
                            "being submitted too often."
                        )
                        return

            if distributed:
                future = self.single_thread_pool.submit(wrapped_op, *args, **kwargs)
            else:
                future = self.multi_thread_pool.submit(wrapped_op, *args, **kwargs)

            op_id = uuid.uuid4().hex
            self._bookkeeping_queue[op_name][op_id] = future

            def callback(fut: Future[T]):
                try:
                    fut.result()  # re-raise any exception from the op or cb
                except BaseException as e:
                    log.exception(e)
                    self._error = e
                finally:
                    # Remove the completed op from the queue.
                    assert op_name is not None  # for mypy
                    self._bookkeeping_queue[op_name].pop(op_id, None)

            future.add_done_callback(callback)
        else:
            wrapped_op(*args, **kwargs)

    def _join_bookkeeping_ops(self, timeout: Optional[float] = None):
        """
        Block until all queued bookkeeping operations are done.
        """
        futures: List[Future] = []
        for op_name, futures_dict in self._bookkeeping_queue.items():
            if futures_dict:
                log.info(
                    f"Waiting for bookkeeping ops to finish: '{op_name}' ({len(futures_dict)} ops)..."
                )
                futures.extend(futures_dict.values())
        concurrent.futures.wait(futures, timeout=timeout)
        log.info("All bookkeeping ops complete")

    def _check_if_canceled(self):
        if self._canceled:
            return

        canceling_rank = self._canceling_rank if self._canceling_rank is not None else -1
        # NOTE: this is a known host-device sync (potentially) so we don't need the warning
        with cuda_sync_debug_mode(0):
            canceling_rank = all_reduce_value(
                canceling_rank,
                self.bookkeeping_device,
                op=dist.ReduceOp.MAX,
                group=self.bookkeeping_pg,
            )
            if canceling_rank >= 0:
                cancel_reason = broadcast_object(
                    self._cancel_reason,
                    src=get_global_rank(canceling_rank, group=self.bookkeeping_pg),
                    group=self.bookkeeping_pg,
                )
                assert cancel_reason is not None
                self._canceled = True
                self._canceling_rank = canceling_rank
                self._cancel_reason = cancel_reason
                log.warning(f"Run canceled from rank {canceling_rank}. Reason: {cancel_reason}")

    def _log_metrics(self):
        if not self._metrics:
            return

        # Prep metrics to reduce by moving to bookkeeping device all at once.
        # NOTE: if training on GPU and `bookkeeping_device` is CPU, this triggers
        # host-device sync. It's unavoidable to have a host-device at some point, but we
        # prefer to do that early and then finish processing the metrics in a separate thread
        # so CUDA training can continue.
        metrics_to_reduce = move_metrics(self._metrics, self.bookkeeping_device)
        self._metrics.clear()

        if self._metrics_consistent is None:
            self._metrics_consistent = check_metrics_consistent(
                self._metrics_reduce_type,
                process_group=self.bookkeeping_pg,
            )
            if not self._metrics_consistent:
                msg = (
                    "Detected inconsistent metrics between ranks. This is expected in some cases "
                    "(like with pipeline parallelism)."
                )
                if not self.async_bookkeeping:
                    msg += " This may result in slower training speeds since you don't have async bookkeeping enabled."
                log.warning(msg)

        self.run_bookkeeping_op(
            reduce_metrics,
            metrics_to_reduce,
            self._metrics_reduce_type,
            self.bookkeeping_device,
            process_group=self.bookkeeping_pg,
            metrics_consistent=self._metrics_consistent,
            cb=self._check_and_pass_on_metrics,
        )

    def _check_and_pass_on_metrics(self, metrics: Dict[int, Dict[str, float]]):
        for step in sorted(metrics.keys()):
            # Check for nan/inf loss and add perplexity.
            if (ce_loss := metrics[step].get(TRAIN_CE_LOSS_METRIC)) is not None:
                if not math.isfinite(ce_loss):
                    raise RuntimeError(f"{ce_loss} loss encountered at step {step}")
                if ce_loss < 10:
                    metrics[step][TRAIN_PPL_METRIC] = math.exp(ce_loss)
            for callback in self._iter_callbacks():
                callback.pre_log_metrics(step, metrics[step])
            for callback in self._iter_callbacks():
                callback.log_metrics(step, metrics[step])

    def _iter_batches(self) -> Generator[Dict[str, Any], None, None]:
        data_iterator = iter(self.data_loader)

        while True:
            for callback in self._iter_callbacks():
                callback.pre_load_batch()

            try:
                batch = next(data_iterator)
                yield batch
            except StopIteration:
                break

    def _dry_run_batch(self):
        try:
            batch = self.data_loader.get_mock_batch()
        except NotImplementedError:
            return  # for backwards compatibility

        log.info("Starting forward/backward dry-run batch...")
        self.train_module.train_batch(batch, dry_run=True)
        log.info("Dry-run complete")
        self._log_ep_no_sync_symm_buffer_summary(context="after dry-run")

    def _log_ep_no_sync_symm_buffer_summary(self, *, context: str) -> None:
        log_summary = getattr(
            self.train_module,
            "log_ep_no_sync_symm_buffer_summary",
            None,
        )
        if callable(log_summary):
            log_summary(context=context)

    def _fit_epoch(self):
        self.data_loader.reshuffle(self.epoch)

        log.info(f"Starting epoch {self.epoch}...")

        for callback in self._iter_callbacks():
            callback.pre_epoch()

        self.train_module.zero_grads()

        first_batch = True
        for batch in self._iter_batches():
            # Bookkeeping.
            self.global_step += 1
            if (
                global_num_tokens := self.data_loader.global_num_tokens_in_batch(batch)
            ) is not None:
                self.global_train_tokens_seen += global_num_tokens
            if (global_num_flops := self.train_module.global_num_flops_in_batch(batch)) is not None:
                self.global_train_petaflops += global_num_flops / 1e15  # flops -> petaflops

            should_skip = False
            if self.steps_to_skip:
                for step_range in self.steps_to_skip:
                    if step_range.start <= self.global_step < step_range.stop:
                        should_skip = True
                        break

            for callback in self._iter_callbacks():
                callback.pre_step(batch)

            if should_skip:
                log.warning(f"Skipping training on step {self.global_step:,d} intentionally...")
            else:
                self.train_module.train_batch(batch)
                if self.global_step == 1:
                    self._log_ep_no_sync_symm_buffer_summary(
                        context="after train step 1 backward before optim"
                    )

                for callback in self._iter_callbacks():
                    callback.pre_optim_step()

                self.train_module.optim_step()
                self.train_module.zero_grads()
                if self.global_step == 1:
                    self._log_ep_no_sync_symm_buffer_summary(
                        context="after train step 1 optim+zero_grad before callbacks"
                    )

            for callback in self._iter_callbacks():
                callback.post_train_batch()

            for callback in self._iter_callbacks():
                callback.post_step()

            if first_batch or self.global_step % self.metrics_collect_interval == 0:
                self._log_metrics()
                if torch.cuda.is_available():
                    torch.cuda.set_sync_debug_mode("warn")

            first_batch = False

            if self.training_complete:
                # Finishing before the epoch is complete.
                # Log any remaining metrics.
                self._log_metrics()
                return

        # Log any remaining metrics.
        self._log_metrics()

        log.info("Epoch complete")

        for callback in self._iter_callbacks():
            callback.post_epoch()

        # Bookkeeping
        self.epoch += 1
        self.data_loader.reset()
