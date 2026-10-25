import copy
import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from torch.distributed import DeviceMesh
from torch.distributed.pipelining import PipelineStage

from olmo_core.config import Config, DType
from olmo_core.distributed.parallel import (
    ContextParallelConfig,
    DataParallelConfig,
    ExpertParallelConfig,
    PipelineP2PBackend,
    PipelineParallelConfig,
    PipelineScheduleType,
    TensorParallelConfig,
)
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention.ring import (
    RingAttentionLoadBalancerType,
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.nn.transformer import (
    Transformer,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.optim import OptimConfig
from olmo_core.optim.moe_optimizer import OLMoDDPOptimizerConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.train_module.config import TrainModuleConfig

from .pipeline.pipeline_schedule import CustomPipelineStage

if TYPE_CHECKING:
    from .ddp_train_module import OLMoDDPTrainModule
    from .pipeline_train_module import TransformerPipelineTrainModule
    from .train_module import TransformerTrainModule

log = logging.getLogger(__name__)


@beta_feature
@dataclass
class TransformerPipelineParallelConfig(PipelineParallelConfig):
    """
    Transformer-specific pipeline parallel config.
    """

    use_custom_stage_implementation: bool = False
    """
    False -> use PyTorch's ``PipelineStage`` implementation.
    True -> use :class:`~olmo_core.train.train_module.transformer.pipeline.pipeline_schedule.CustomPipelineStage`,
    which re-uses receive buffers across micro-batches.
    """

    save_schedule_plot: bool = False
    """
    If ``True``, rank 0 saves a diagnostic plot (and text dump) of the pipeline schedule when it is
    built. Requires ``matplotlib`` (the ``dev`` extra), so it is off by default.
    """

    schedule_plot_dir: Optional[str] = None
    """
    Directory for the schedule plot when :data:`save_schedule_plot` is set. Defaults to a temporary
    directory.
    """

    split_points: Optional[List[int]] = None
    """
    A list of unique, increasing block indices that define how to split the model into stages.

    For example, ``split_points = [0, 2]`` with a 4-layer model means the model will be split into
    3 stages, with the first containing just the embedding, the second containing blocks 0 and 1,
    and the third containing blocks 2 and 3 and the language modeling head.

    If not specified the split points are determined automatically based on the schedule type.
    """

    def get_split_points(self, n_layers: int) -> List[int]:
        if self.split_points is not None:
            return self.split_points

        # Multi-stage schedules support more than 2 stages per rank, but this is the default if
        # no pipeline split is specified.
        num_stages_per_rank = 1 if self.schedule.is_single_stage else 2
        total_stages = self.degree * num_stages_per_rank
        num_layers = n_layers
        if total_stages > num_layers:
            raise OLMoConfigurationError("Total stages cannot be greater than the number of layers")

        base_interval = num_layers // total_stages
        extra_layers = num_layers % total_stages

        splits: List[int] = []
        current_layer = 0
        for i in range(total_stages - 1):
            if i == 0:
                current_layer += base_interval
            else:
                # Middle stages get an extra layer if there are any remaining
                if extra_layers > 0:
                    current_layer += base_interval + 1
                    extra_layers -= 1
                else:
                    current_layer += base_interval
            splits.append(current_layer)
        log.info(f"Auto generated pipeline split points will be {splits}")
        return splits

    def split_model(
        self,
        model: Transformer,
        *,
        pp_mesh: DeviceMesh,
        device: torch.device,
        use_ddp: bool = False,
        p2p_group: Optional[dist.ProcessGroup] = None,
    ) -> Tuple[List[PipelineStage], List[Transformer]]:
        if self.p2p_backend != PipelineP2PBackend.nccl and not self.use_custom_stage_implementation:
            raise OLMoConfigurationError(
                f"p2p_backend={self.p2p_backend.value!r} requires use_custom_stage_implementation=True"
            )

        custom_schedules = {
            PipelineScheduleType.custom_1F1B,
            PipelineScheduleType.custom_interleaved_1F1B,
            PipelineScheduleType.custom_1F1B_V,
        }
        if self.schedule in custom_schedules and not self.use_custom_stage_implementation:
            # The custom schedule driver expects CustomPipelineStage (e.g. `group_size`,
            # `get_fwd_send_ops`); pairing it with torch's PipelineStage fails in pre_train / step.
            raise OLMoConfigurationError(
                f"pipeline schedule {self.schedule.value!r} requires use_custom_stage_implementation=True"
            )

        split_points = self.get_split_points(model.n_layers)
        num_stages = len(split_points) + 1

        if num_stages > 1 and model.tie_word_embeddings:
            raise NotImplementedError(
                "Pipeline parallelism with tied word embeddings is not supported: the input "
                "embeddings and LM head are placed on different pipeline stages, so they cannot "
                "share a weight."
            )

        pp_rank = pp_mesh.get_local_rank()

        def build_stage(
            stage_idx: int,
            start_layer: Optional[int],
            stop_layer: Optional[int],
            is_first: bool = False,
            is_last: bool = False,
        ) -> Tuple[PipelineStage, Transformer]:
            model_chunk = copy.deepcopy(model)
            if not is_first:
                model_chunk.embeddings = None  # type: ignore
                model_chunk.embedding_norm = None  # type: ignore

            drop_layers = start_layer is not None
            for block_idx in range(model.n_layers):
                # we keep layers in a contiguous region between start (inclusive) and stop (exclusive)
                if block_idx == start_layer:
                    drop_layers = False
                if block_idx == stop_layer:
                    drop_layers = True
                if drop_layers:
                    del model_chunk.blocks[str(block_idx)]

            if not is_last:
                model_chunk.lm_head = None  # type: ignore

            if self.use_custom_stage_implementation:
                # Custom stage implementation re-uses receive buffers across micro-batches.
                stage = CustomPipelineStage(
                    model_chunk,
                    stage_idx,
                    num_stages,
                    device,
                    is_rddp=use_ddp,
                    group=pp_mesh.get_group("pp"),
                    p2p_group=p2p_group,
                    p2p_backend=self.p2p_backend.value,
                )
            else:
                stage = PipelineStage(
                    model_chunk,
                    stage_idx,
                    num_stages,
                    device,
                    group=pp_mesh.get_group("pp"),
                )
            return stage, model_chunk

        stage_idx = pp_rank

        stages = []
        models = []
        for stage_idx in self.stage_ids_this_rank(pp_rank, num_stages):
            start_layer = split_points[stage_idx - 1] if stage_idx > 0 else None
            stop_layer = split_points[stage_idx] if stage_idx < num_stages - 1 else None
            stage, model_chunk = build_stage(
                stage_idx,
                start_layer,
                stop_layer,
                is_first=stage_idx == 0,
                is_last=stage_idx == num_stages - 1,
            )
            log.info(
                f"PP rank {pp_rank} is building stage {stage_idx} with start layer "
                f"{start_layer}, stop layer {stop_layer}: {model_chunk}"
            )
            stages.append(stage)
            models.append(model_chunk)

        return stages, models


@dataclass
class TransformerDataParallelConfig(DataParallelConfig):
    """
    Transformer-specific data parallel config.
    """

    wrapping_strategy: TransformerDataParallelWrappingStrategy = (
        TransformerDataParallelWrappingStrategy.full
    )
    """
    The wrapping strategy.
    """

    prefetch_factor: int = 0

    only_allreduce_last_microbatch: bool = True
    """
    Only synchronize gradients on the last micro-batch of a gradient-accumulation step (skip the
    reduction on intermediate micro-batches). Used by :class:`OLMoDDPTrainModule` with
    :class:`~olmo_core.nn.parallel.MultiGroupDistributedDataParallel`. The historical name predates
    normal-parameter reduce-scatter; this setting controls whichever DDP gradient collective is used.
    """

    reduce_grads_in_fp32: bool = True
    """Reduce gradients in fp32 (see :class:`~olmo_core.nn.parallel.MultiGroupDistributedDataParallel`)."""

    accumulate_grads_in_fp32: bool = True
    """Accumulate gradients in fp32 (see :class:`~olmo_core.nn.parallel.MultiGroupDistributedDataParallel`)."""

    bucket_cap_mb: Optional[int] = None
    """Gradient reduction bucket size cap in MiB (``None`` = backend default)."""

    use_reduce_scatter: bool = False
    """
    Reduce normal-parameter gradients directly into distributed-optimizer shards.

    Parameters with replicated optimizer state continue to use all-reduce. This
    option does not change the FP8WeightStore gradient synchronization path. It
    currently requires final-microbatch-only synchronization, does not support
    context parallelism, and requires the custom pipeline-stage implementation
    when pipeline parallelism is enabled.
    """


@dataclass
class TransformerTensorParallelConfig(TensorParallelConfig):
    """
    Transformer-specific tensor parallel config.
    """


@dataclass
class TransformerContextParallelConfig(ContextParallelConfig):
    """
    Transformer-specific context parallel config.
    """

    ring: RingContextParallelStyle | None = None
    uly: UlyssesContextParallelStyle | None = None

    def __post_init__(self):
        if self.ring is not None and self.uly is not None:
            raise NotImplementedError(
                "Only one of ring or ulysses can be specified. While not technically "
                "mutually exclusive, a combined context parallel style is not yet supported."
            )
        elif self.ring is None and self.uly is None:
            raise OLMoConfigurationError("One of ring or uly must be specified")

    @classmethod
    def zig_zag(cls, degree: int, head_stride: int = 1) -> "TransformerContextParallelConfig":
        return cls(
            degree=degree,
            ring=RingContextParallelStyle(
                load_balancer=RingAttentionLoadBalancerType.zig_zag,
                head_stride=head_stride,
            ),
        )

    @classmethod
    def llama3(cls, degree: int, head_stride: int = 1) -> "TransformerContextParallelConfig":
        return cls(
            degree=degree,
            ring=RingContextParallelStyle(
                load_balancer=RingAttentionLoadBalancerType.llama3,
                head_stride=head_stride,
            ),
        )

    @classmethod
    def ulysses(cls, degree: int) -> "TransformerContextParallelConfig":
        return cls(
            degree=degree,
            uly=UlyssesContextParallelStyle(),
        )


@dataclass
class TransformerExpertParallelConfig(ExpertParallelConfig):
    """
    Transformer-specific expert parallel config.
    """


@beta_feature
@dataclass
class TransformerActivationCheckpointingConfig(Config):
    """
    Defines the activation checkpointing strategy for a transformer model.
    """

    mode: TransformerActivationCheckpointingMode = TransformerActivationCheckpointingMode.full
    """
    The activation checkpointing mode.
    """

    block_interval: Optional[int] = None
    """
    Required when :data:`mode` is "selected_blocks". Determines which blocks are wrapped.
    """

    modules: Optional[List[str]] = None
    """
    Required when :data:`mode` is "selected_modules". A list of modules names to wrap for
    activation checkpointing. Globs are supported.
    """

    activation_memory_budget: Optional[float] = None
    """
    Required when :data:`mode` is "budget". Memory budget for activation checkpointing in range [0, 1].
    0 = recompute all activations, 1 = recompute none (default). Requires compilation to be enabled.

    See https://pytorch.org/blog/activation-checkpointing-techniques/ for more details.
    """

    determinism_check: str = "default"
    """
    Passed through to torch's ``checkpoint_wrapper``. "default" compares forward vs. recompute
    tensor metadata; set to "none" to skip the check. Needed for models whose recompute produces
    spurious metadata mismatches (e.g. opaque linear-attention kernels under ``torch.compile``).
    """

    def __post_init__(self):
        if (
            self.mode == TransformerActivationCheckpointingMode.selected_blocks
            and self.block_interval is None
        ):
            raise OLMoConfigurationError(
                "'block_interval' is required for 'selected_blocks' activation checkpointing"
            )
        elif (
            self.mode == TransformerActivationCheckpointingMode.selected_modules
            and self.modules is None
        ):
            raise OLMoConfigurationError(
                "'modules' is required for 'selected_modules' activation checkpointing"
            )


@dataclass
class TransformerTrainModuleConfig(TrainModuleConfig):
    """
    A configuration class for building :class:`TransformerTrainModule` or
    :class:`TransformerPipelineTrainModule` instances.

    .. seealso::
        See the :class:`TransformerTrainModule` and :class:`TransformerPipelineTrainModule`
        documentation for a description of the fields.
    """

    rank_microbatch_size: int
    max_sequence_length: int

    # Optimizer settings.

    optim: OptimConfig
    max_grad_norm: Optional[float] = None
    scheduler: Optional[Scheduler] = None

    # Model settings.

    compile_model: bool = False
    float8_config: Optional[Float8Config] = None
    pp_config: Optional[TransformerPipelineParallelConfig] = None
    dp_config: Optional[TransformerDataParallelConfig] = None
    tp_config: Optional[TransformerTensorParallelConfig] = None
    cp_config: Optional[TransformerContextParallelConfig] = None
    ep_config: Optional[TransformerExpertParallelConfig] = None
    ac_config: Optional[TransformerActivationCheckpointingConfig] = None

    # Loss function settings.

    z_loss_multiplier: Optional[float] = None

    # Checkpoint settings.

    state_dict_save_opts: Optional[Dict[str, Any]] = None
    state_dict_load_opts: Optional[Dict[str, Any]] = None
    load_key_mapping: Optional[Dict[str, str]] = None

    # Other train settings.

    autocast_precision: Optional[DType] = None
    label_ignore_index: int = -100

    def build(
        self,
        model: Transformer,
        device: Optional[torch.device] = None,
        eval_only: bool = False,
    ) -> Union["TransformerTrainModule", "TransformerPipelineTrainModule"]:
        """
        Build the corresponding :class:`TransformerTrainModule` or :class:`TransformerPipelineTrainModule.

        :param model: The :class:`~olmo_core.nn.transformer.Transformer` model to train.
        :param device: The device to train on.
        :param eval_only: If ``True``, build the train module without an optimizer (eval-only).
        """
        from .pipeline_train_module import TransformerPipelineTrainModule
        from .train_module import TransformerTrainModule

        kwargs = self.as_dict(exclude_none=True, recurse=False)
        if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
            kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
        if (state_dict_save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**state_dict_save_opts)
        if (state_dict_load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**state_dict_load_opts)

        if self.pp_config is not None:
            return TransformerPipelineTrainModule(
                model=model,
                device=device,
                eval_only=eval_only,
                **kwargs,
            )
        else:
            return TransformerTrainModule(
                model=model,
                device=device,
                eval_only=eval_only,
                **kwargs,
            )


@beta_feature
@dataclass
class TransformerPipelineTrainModuleConfig(TransformerTrainModuleConfig):
    """
    Kept for backwards compatibility, but please use :class:`TransformerTrainModuleConfig` instead.
    """

    def __post_init__(self):
        if self.pp_config is None:
            raise OLMoConfigurationError("'pp_config' is required")


@beta_feature
@dataclass
class OLMoDDPTrainModuleConfig(TrainModuleConfig):
    """
    Configuration for :class:`~olmo_core.train.train_module.transformer.ddp_train_module.OLMoDDPTrainModule`,
    the train module for the fused MoE-v2 transformer (built with the fused MoE distributed
    optimizer, :class:`~olmo_core.optim.OLMoDDPOptimizerConfig`).
    """

    rank_microbatch_size: int
    max_sequence_length: int

    # Optimizer settings.

    optim: OLMoDDPOptimizerConfig
    max_grad_norm: Optional[float] = None
    """
    Override the optimizer's gradient clipping threshold when set. If ``None``, use
    :data:`~olmo_core.optim.moe_optimizer.OLMoDDPOptimizerConfig.max_grad_norm`.
    Clipping is performed by the optimizer.
    """
    scheduler: Optional[Scheduler] = None

    # Model settings.

    compile_model: bool = False
    float8_config: Optional[Float8Config] = None
    pp_config: Optional[TransformerPipelineParallelConfig] = None
    dp_config: Optional[TransformerDataParallelConfig] = None
    tp_config: Optional[TransformerTensorParallelConfig] = None
    cp_config: Optional[TransformerContextParallelConfig] = None
    ep_config: Optional[TransformerExpertParallelConfig] = None
    ac_config: Optional[TransformerActivationCheckpointingConfig] = None

    grad_accum_in_fp32: Optional[bool] = None

    # Loss function settings.

    z_loss_multiplier: Optional[float] = None

    # Checkpoint settings.

    state_dict_save_opts: Optional[Dict[str, Any]] = None
    state_dict_load_opts: Optional[Dict[str, Any]] = None
    load_key_mapping: Optional[Dict[str, str]] = None
    reset_optimizer_states_on_load: bool = False
    reset_optimizer_states_on_resume: bool = False
    expand_shared_qk_norm_on_load: bool = False
    """Opt in to expanding shared Q/K gains and Adam moments to independent head gains.

    Applies to optimizer restores, model-only training loads, and eval-only loads.
    """

    # Other train settings.

    label_ignore_index: int = -100

    def build(
        self,
        model: Transformer,
        device: Optional[torch.device] = None,
        eval_only: bool = False,
    ) -> "OLMoDDPTrainModule":
        """
        Build the corresponding :class:`OLMoDDPTrainModule`.

        :param model: The :class:`~olmo_core.nn.transformer.Transformer` model to train.
        :param device: The device to train on.
        :param eval_only: If ``True``, build the train module without an optimizer (eval-only).
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        if (max_grad_norm := kwargs.pop("max_grad_norm", None)) is not None:
            kwargs["optim"] = replace(self.optim, max_grad_norm=max_grad_norm)

        if (state_dict_save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**state_dict_save_opts)
        if (state_dict_load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**state_dict_load_opts)

        # `grad_accum_in_fp32` is superseded by the DP config's `accumulate_grads_in_fp32`; map the
        # legacy field onto the DP config so migrated configs that set it aren't silently ignored.
        grad_accum_in_fp32 = kwargs.pop("grad_accum_in_fp32", None)
        if grad_accum_in_fp32 is not None and (dp_config := kwargs.get("dp_config")) is not None:
            kwargs["dp_config"] = replace(dp_config, accumulate_grads_in_fp32=grad_accum_in_fp32)

        return self._build_train_module(
            model=model,
            device=device,
            eval_only=eval_only,
            **kwargs,
        )

    def _build_train_module(self, **kwargs) -> "OLMoDDPTrainModule":
        """Construct the train module; subclasses can build a subclass of it instead."""
        from .ddp_train_module import OLMoDDPTrainModule

        return OLMoDDPTrainModule(**kwargs)
