from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer

from olmo_core.config import StrEnum
from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.checkpoint import get_optimizer_state_dict
from olmo_core.distributed.utils import get_local_tensor, get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.functional.cross_entropy_loss import cross_entropy_loss
from olmo_core.utils import move_to_device

from ..common import MetricMergeStrategy, ReduceType, get_inputs_for_loss

if TYPE_CHECKING:
    from ..trainer import Trainer


class EvalBatchSizeUnit(StrEnum):
    """
    The different units for defining the size for eval batches.
    """

    tokens = "tokens"
    """
    Specify in tokens.
    """
    instances = "instances"
    """
    Specify in instances.
    """


@dataclass
class EvalBatchSpec:
    """
    Defines how eval batches should be sized.
    """

    rank_batch_size: int
    """
    The size of eval batches per rank.
    """
    batch_size_unit: EvalBatchSizeUnit = EvalBatchSizeUnit.tokens
    """
    The unit for the :data:`rank_batch_size`.
    """
    max_sequence_length: Optional[int] = None
    """
    The maximum allowed sequence length.
    """
    fixed_sequence_length: bool = False
    """
    If all batches should have a fixed sequence length at :data:`max_sequence_length` tokens.
    If this is ``True`` then ``max_sequence_length`` must be specified.
    """

    def __post_init__(self):
        if self.fixed_sequence_length and self.max_sequence_length is None:
            raise OLMoConfigurationError(
                "'max_sequence_length' must be specified when 'fixed_sequence_length=True'"
            )


class TrainModule(Stateful, metaclass=ABCMeta):
    """
    A :class:`TrainModule` is an abstraction around a :class:`~torch.nn.Module` and
    :class:`~torch.optim.Optimizer` to provide a unified API for the :class:`~olmo_core.train.Trainer`
    that's flexible enough to handle a variety of training paradigms.

    .. note::
        :class:`TrainModule` implementations are responsible for recording all necessary metrics
        like the training loss, which can be done by calling :meth:`record_metric()`.

    .. note::
        See :class:`BasicTrainModule` for a simple example implementation.
    """

    def __init__(self):
        self._trainer: Optional["Trainer"] = None

    @property
    def trainer(self) -> "Trainer":
        """
        The :class:`~olmo_core.train.Trainer` being used.

        .. warning::
            This property can only be accessed after the trainer has been attached.
        """
        if self._trainer is None:
            raise RuntimeError("trainer has not yet been assigned the train module")
        return self._trainer

    @property
    def dp_process_group(self) -> Optional[dist.ProcessGroup]:
        """
        Should return the data parallel process group if it's anything other than the default
        process group.
        """
        return None

    @property
    @abstractmethod
    def eval_batch_spec(self) -> EvalBatchSpec:
        """
        Should return the desired specification for evaluation batches.
        This is used for in-loop evaluation, for example, to determine how to build eval
        batches in a way that will work for the particular :class:`TrainModule`.
        """
        raise NotImplementedError

    def on_attach(self):
        """
        Runs as soon as the :class:`~olmo_core.train.Trainer` has been attached.
        """

    def pre_train(self):
        """
        Runs before the training loop starts and right after ``pre_train()`` has been called on all
        callbacks.
        """

    @abstractmethod
    def state_dict(self, *, optim: Optional[bool] = None) -> Dict[str, Any]:
        """
        Get the state dict to save or load.

        :param optim: If set to ``False``, optimizer state is not returned in the state dict.
        """
        raise NotImplementedError

    def state_dict_to_save(self, *, optim: Optional[bool] = None) -> Dict[str, Any]:
        """
        Can be overridden if the state dict to save should be different from the state dict to load.
        By default just returns :func:`state_dict()`.

        :param optim: If set to ``False``, optimizer state is not returned in the state dict.
        """
        return self.state_dict(optim=optim)

    def state_dict_to_load(
        self, metadata: Metadata, *, optim: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Can be overridden if the state dict to load should be different from the state dict to save.
        By default just returns :func:`state_dict()`.

        :param optim: If set to ``False``, optimizer state is not returned in the state dict.
        """
        del metadata
        return self.state_dict(optim=optim)

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load a state dict.
        """
        raise NotImplementedError

    @abstractmethod
    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        """
        Run a forward and backward pass on a training batch.
        """
        raise NotImplementedError

    @abstractmethod
    def eval_batch(self, batch: Dict[str, Any], labels: Optional[Any] = None) -> Any:
        """
        Run a forward pass on a eval batch.
        """
        raise NotImplementedError

    @abstractmethod
    def optim_step(self):
        """
        Run an optimizer step.
        """
        raise NotImplementedError

    @abstractmethod
    def zero_grads(self):
        """
        Zero-out gradients.
        """
        raise NotImplementedError

    @abstractmethod
    @lru_cache
    def num_flops_per_token(self, seq_len: int) -> Optional[int]:
        """
        Returns the number of flops per token for the given sequence length, or ``None`` if flops
        estimation is not supported.
        """
        raise NotImplementedError

    @abstractmethod
    def global_num_flops_in_batch(self, batch: Dict[str, Any]) -> Optional[int]:
        """
        Return the total (global) number of flops in the batch, or ``None`` if flops estimation
        is not supported.
        """
        raise NotImplementedError

    def record_metric(
        self,
        name: str,
        value: Union[float, torch.Tensor],
        reduce_type: Optional[ReduceType] = None,
        namespace: Optional[str] = None,
        merge_strategy: MetricMergeStrategy = MetricMergeStrategy.warn,
    ):
        """
        Record a metric. This is simply a convenience method that calls out to
        :meth:`olmo_core.train.Trainer.record_metric()`.

        .. seealso::
            Use :meth:`record_ce_loss()` to record the cross-entropy loss, specifically.
        """
        return self.trainer.record_metric(
            name, value, reduce_type=reduce_type, namespace=namespace, merge_strategy=merge_strategy
        )

    def record_ce_loss(
        self, value: Union[float, torch.Tensor], reduce_type: Optional[ReduceType] = None
    ):
        """
        Record the cross-entropy loss metric specifically.
        """
        return self.trainer.record_ce_loss(value, reduce_type=reduce_type)

    def _attach_trainer(self, trainer: "Trainer"):
        self._trainer = trainer
        self.on_attach()


class BasicTrainModule(TrainModule):
    """
    A basic :class:`TrainModule` implementation, mainly used for as an example and for testing.
    For a more practical implementation, see :class:`TransformerTrainModule`.

    :param model: The model to train.
    :param optim: The corresponding optimizer.
    :param rank_microbatch_size: The microbatch size *in tokens* per rank,
        i.e. the number of tokens to process at a time from each rank.

        .. note:: This must evenly divide into the global batch size by a factor of the data
            parallel world size. If this is less than the global batch divided by the data
            parallel world size then gradient accumulation is used.
    :param max_grad_norm: Clip gradient norms to this value.
    """

    def __init__(
        self,
        model: nn.Module,
        optim: Optimizer,
        rank_microbatch_size: int,
        max_grad_norm: Optional[float] = None,
        label_ignore_index: int = -100,
    ):
        super().__init__()
        self.model = model
        self.optim = optim
        self.rank_microbatch_size = rank_microbatch_size
        self.max_grad_norm = max_grad_norm
        self.loss_fn = cross_entropy_loss
        self.label_ignore_index = label_ignore_index

    @property
    def eval_batch_spec(self) -> EvalBatchSpec:
        return EvalBatchSpec(rank_batch_size=self.rank_microbatch_size)

    def on_attach(self):
        # Validate batch size.
        if (
            self.trainer.global_batch_size
            % (self.rank_microbatch_size * (ws := get_world_size(self.trainer.dp_process_group)))
            != 0
        ):
            raise OLMoConfigurationError(
                f"global batch size ({self.trainer.global_batch_size:,d}) must be divisible by "
                f"micro-batch size ({self.rank_microbatch_size:,d}) x DP world size ({ws})"
            )

    def state_dict(self, *, optim: Optional[bool] = None) -> Dict[str, Any]:
        sd_options = dist_cp_sd.StateDictOptions(full_state_dict=False, cpu_offload=True)
        state_dict: Dict[str, Any] = {
            "model": dist_cp_sd.get_model_state_dict(self.model, options=sd_options),
        }
        if optim is not False:
            state_dict["optim"] = get_optimizer_state_dict(
                self.model, self.optim, options=sd_options
            )
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        dist_cp_sd.set_model_state_dict(
            self.model, state_dict["model"], options=dist_cp_sd.StateDictOptions(strict=True)
        )
        if "optim" in state_dict:
            dist_cp_sd.set_optimizer_state_dict(
                self.model,
                self.optim,
                state_dict["optim"],
                options=dist_cp_sd.StateDictOptions(strict=True),
            )

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        self.model.train()

        # Move tensors to the right device.
        batch = move_to_device(batch, self.trainer.device)

        # Generate labels, calculate how many tokens are going to be use in the loss.
        if "labels" not in batch:
            batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)
        batch_num_tokens_for_loss = (batch["labels"] != self.label_ignore_index).sum()

        # Split into micro-batches.
        if self.rank_microbatch_size < (seq_len := batch["input_ids"].shape[1]):
            raise RuntimeError(
                f"Microbatch size ({self.rank_microbatch_size}) is too small relative to sequence length ({seq_len})"
            )
        micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)

        ce_batch_loss = move_to_device(torch.tensor(0.0), self.trainer.device)

        # Train one micro-batch at a time.
        for micro_batch in micro_batches:
            # Run forward pass.
            logits = self.model_forward(micro_batch)

            # shape: (batch_size * (seq_len - 1), vocab_size), (batch_size * (seq_len - 1),)
            logits_for_loss, labels_for_loss = get_inputs_for_loss(
                micro_batch,
                logits,
                label_ignore_index=self.label_ignore_index,
            )

            # Calculate loss.
            # NOTE: we use the "sum" loss reduction and then divide by 'batch_num_tokens_for_loss'
            # (the total number of tokens used in the loss across the whole batch, not just the micro batch)
            # to avoid biasing the loss in the case where micro-batches might not be the same size.
            ce_loss, _ = self.loss_fn(
                logits_for_loss,
                labels_for_loss,
                ignore_index=self.label_ignore_index,
                reduction="sum",
            )
            ce_loss.div_(batch_num_tokens_for_loss)

            # Update overall CE batch loss.
            ce_batch_loss += get_local_tensor(ce_loss.detach())

            # Run backward pass.
            ce_loss.backward()

        # In case this helps with memory utilization.
        del batch

        if dry_run:
            return

        # Record loss metrics.
        self.record_ce_loss(ce_batch_loss, ReduceType.mean)

    def eval_batch(self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None) -> Any:
        self.model.eval()
        batch = move_to_device(batch, self.trainer.device)
        with torch.no_grad():
            logits = self.model_forward(batch)
        loss: Optional[torch.Tensor] = None
        if labels is not None:
            loss, _ = self.loss_fn(
                logits,
                labels,
                ignore_index=self.label_ignore_index,
                reduction="none",
            )
        return logits, loss

    def optim_step(self):
        # Maybe clip gradients.
        if self.max_grad_norm is not None:
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            # NOTE: grad norm is already reduced over ranks, so we set `reduce_type` to `None`.
            self.trainer.record_metric(
                "total grad norm", grad_norm, reduce_type=None, namespace="optim"
            )

        # Step optimizer.
        self.optim.step()

    def zero_grads(self):
        self.optim.zero_grad(set_to_none=True)

    def model_forward(self, micro_batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(input_ids=micro_batch["input_ids"])

    @lru_cache
    def num_flops_per_token(self, seq_len: int) -> Optional[int]:
        raise NotImplementedError

    def global_num_flops_in_batch(self, batch: Dict[str, Any]) -> Optional[int]:
        raise NotImplementedError
