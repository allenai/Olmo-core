import contextlib
import logging
import math
from dataclasses import replace
from functools import cached_property, lru_cache, partial
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.pipelining import PipelineStage
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

from olmo_core.data.utils import get_labels
from olmo_core.distributed.checkpoint import (
    get_optimizer_state_dict,
    merge_state_dicts,
    prune_state_dict,
    swap_param_keys,
)
from olmo_core.distributed.parallel import (
    PipelineSchedule,
    build_world_mesh,
    get_device_mesh_info,
    get_dp_process_group,
    get_pp_mesh,
)
from olmo_core.distributed.utils import (
    get_global_rank,
    get_local_tensor,
    get_rank,
    get_reduce_divide_factor,
    get_world_size,
)
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.transformer import Transformer
from olmo_core.optim import OptimConfig, SkipStepOptimizer
from olmo_core.optim.scheduler import Scheduler
from olmo_core.utils import (
    gc_cuda,
    get_default_device,
    log_once,
    move_to_device,
    warn_once,
)

from ...common import MetricMergeStrategy, ReduceType
from ..train_module import EvalBatchSizeUnit, EvalBatchSpec, TrainModule
from .common import parallelize_model
from .config import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
    TransformerPipelineParallelConfig,
    TransformerTensorParallelConfig,
)
from .train_module import _assert_no_unowned_fp8_weight_stores

log = logging.getLogger(__name__)


@beta_feature
class TransformerPipelineTrainModule(TrainModule):
    """
    A pipeline-parallel :class:`TrainModule` for most :class:`~olmo_core.nn.transformer.Transformer` model
    implementation provided by this library.

    .. tip::
        Use the :class:`TransformerPipelineTrainModuleConfig` to easily configure and build
        :class:`TransformerPipelineTrainModule` instances.

    :param model: The :class:`~olmo_core.nn.transformer.Transformer` model to train.
    :param optim: The corresponding optimizer config.
    :param rank_microbatch_size: The microbatch size *in tokens* per rank,
        i.e. the number of tokens to process at a time from each rank.

        .. note:: This must evenly divide into the global batch size by a factor of the data
            parallel world size. If this is less than the global batch divided by the data
            parallel world size then gradient accumulation is used.
    :param max_sequence_length: The maximum expected sequence length during training and evaluation.
    :param compile_model: Whether to compile to the model.
    :param float8_config: Float8 configuration for the model.
    :param dp_config: Data parallel configuration for the model.
    :param tp_config: Tensor parallel configuration for the model.
    :param cp_config: Context parallel configuration for the model.
    :param pp_config: Pipeline parallel configuration for the model.
    :param ac_config: Activation checkpointing configuration for the model.
    :param z_loss_multiplier: Use Z-loss with this multiplier.
    :param autocast_precision: Enable AMP with this data type.
    :param max_grad_norm: Clip gradient norms to this value.
    :param scheduler: Optional learning rate scheduler for the optimizer.
    :param device: The device to train on.
    :param state_dict_save_opts: Can be used to override the state dict options used
        when saving a checkpoint.
    :param state_dict_load_opts: Can be used to override the state dict options used
        when loading a checkpoint.
    :param load_key_mapping: Can be used to load a checkpoint where certain parameter have different names.
        This dictionary should map current keys to keys in the checkpoint to be loaded.
    """

    def __init__(
        self,
        model: Transformer,
        optim: OptimConfig,
        rank_microbatch_size: int,
        max_sequence_length: int,
        pp_config: TransformerPipelineParallelConfig,
        compile_model: bool = False,
        float8_config: Optional[Float8Config] = None,
        dp_config: Optional[TransformerDataParallelConfig] = None,
        tp_config: Optional[TransformerTensorParallelConfig] = None,
        cp_config: Optional[TransformerContextParallelConfig] = None,
        ep_config: Optional[TransformerExpertParallelConfig] = None,
        ac_config: Optional[TransformerActivationCheckpointingConfig] = None,
        z_loss_multiplier: Optional[float] = None,
        autocast_precision: Optional[torch.dtype] = None,
        max_grad_norm: Optional[float] = None,
        scheduler: Optional[Scheduler] = None,
        device: Optional[torch.device] = None,
        state_dict_save_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        state_dict_load_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        load_key_mapping: Optional[Dict[str, str]] = None,
        label_ignore_index: int = -100,
        eval_only: bool = False,
    ):
        super().__init__()

        # Validate some options.
        if rank_microbatch_size % max_sequence_length != 0:
            raise OLMoConfigurationError(
                f"'rank_microbatch_size' ({rank_microbatch_size:,d} tokens) must be divisible by "
                f"'max_sequence_length' ({max_sequence_length:,d} tokens)"
            )

        # Build world mesh.
        self.device = device or get_default_device()
        self.world_mesh = build_world_mesh(
            dp=dp_config, tp=tp_config, cp=cp_config, pp=pp_config, device_type=self.device.type
        )
        self.dp_world_size = get_world_size(self.dp_process_group)
        log.info(f"Data parallel world size = {self.dp_world_size:,d}")

        self._pp_config = pp_config
        # We'll initialize this lazily when the trainer is attached, since we need to know
        # the global batch size in order to determine the number of pipeline micro-batches.
        self._train_pp_schedule: Optional[PipelineSchedule] = None
        self._pp_stages: Optional[List[PipelineStage]] = None
        self.pp_mesh = get_pp_mesh(self.world_mesh)
        self.pp_group = self.pp_mesh.get_group()
        self.pp_group_rank = get_rank(self.pp_group)
        self.pp_group_size = get_world_size(self.pp_group)
        self.pp_prev_rank = (self.pp_group_rank - 1) % self.pp_group_size
        self.pp_next_rank = (self.pp_group_rank + 1) % self.pp_group_size
        self.pp_final_stage_rank = self._pp_config.final_stage_rank()
        pp_p2p_group = pp_config.build_p2p_process_group(self.world_mesh)

        # Capture num_flops_per_token from the full unsplit model before split_model deepcopies
        # and drops layers. Under PP each rank only holds its pipeline stage's layers, so querying
        # model_parts would undercount. When the model is on meta device (the common case for PP)
        # this reference has no memory cost.
        self._full_model_num_flops_per_token = model.num_flops_per_token

        # Split model into pipeline stages.
        stages, model_parts = pp_config.split_model(
            model,
            pp_mesh=self.pp_mesh,
            device=self.device,
            use_ddp=(self.dp_world_size > 1 and dp_config is not None and dp_config.name == "ddp"),
            p2p_group=pp_p2p_group,
        )
        self._pp_stages = stages
        log.info(
            f"Applied pipeline parallelism to the model with {get_device_mesh_info(self.pp_mesh)}"
        )

        # Parallelize model parts.
        self.model_parts: List[Transformer] = parallelize_model(
            model_parts,
            world_mesh=self.world_mesh,
            device=self.device,
            max_sequence_length=max_sequence_length,
            rank_microbatch_size=rank_microbatch_size,
            compile_model=compile_model,
            float8_config=float8_config,
            dp_config=dp_config,
            tp_config=tp_config,
            cp_config=cp_config,
            ep_config=ep_config,
            ac_config=ac_config,
            pp_enabled=True,
        )

        self._dp_config = dp_config
        self._cp_config = cp_config
        self._tp_config = tp_config
        self._ep_config = ep_config
        self.label_ignore_index = label_ignore_index
        self.z_loss_multiplier = z_loss_multiplier
        self.rank_microbatch_size = rank_microbatch_size
        self.max_sequence_length = max_sequence_length
        self.autocast_precision = autocast_precision
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.state_dict_save_opts = state_dict_save_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, cpu_offload=True
        )
        self.state_dict_load_opts = state_dict_load_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, strict=False
        )
        self.load_key_mapping = load_key_mapping
        self.eval_only = eval_only

        # Build optimizer(s) — skipped in ``eval_only`` mode (no backward, no step).
        # NOTE: parallelize_model() above still runs because PP/FSDP/HSDP wrapping is required for
        # param sharding even at eval time. Only the optimizer construction is skipped here.
        self.optimizers: List[Optimizer] = []
        if not self.eval_only:
            log.info("Building optimizer(s)...")
            for model in self.model_parts:
                _assert_no_unowned_fp8_weight_stores(model)
            self.optimizers = [optim.build(model, strict=False) for model in self.model_parts]
        else:
            log.info("Skipping optimizer build because eval_only=True")

    def _require_optimizers(self) -> List[Optimizer]:
        if not self.optimizers:
            raise RuntimeError(
                f"{type(self).__name__} was built with eval_only=True and has no optimizers"
            )
        return self.optimizers

    @property
    def model(self) -> Transformer:
        """
        Convenience accessor for the single model part when pipeline parallelism is not in use.

        Provided so callers that expect a single ``model`` attribute (e.g.
        :class:`~olmo_core.train.callbacks.speed_monitor.SpeedMonitorCallback`,
        :class:`~olmo_core.train.callbacks.hf_converter.HFConverterCallback`,
        :class:`~olmo_core.train.callbacks.model_merger.ModelMergerCallback`,
        :class:`~olmo_core.train.callbacks.gap_monitor.GAPMonitorCallback`,
        and ``Checkpointer.load_model_module``) work without branching on train-module type.

        Use :data:`model_parts` directly when iterating over all parts.

        :raises RuntimeError: If pipeline parallelism is in use (``len(model_parts) > 1``).
            ``model_parts[0]`` would only expose stage 0's weights/modules, which is silently
            wrong for callers that read state-dicts, named parameters, or do full-model export.

        .. note::
            TODO: update the callers above to iterate ``model_parts`` (or otherwise stitch stages
            together) so this property can return a coherent view when ``len(model_parts) > 1``.
            Until then, the explicit error here is preferable to silently returning a partial model.
        """
        if len(self.model_parts) > 1:
            raise RuntimeError(
                f"{type(self).__name__}.model is only valid when pipeline parallelism is "
                f"disabled (len(model_parts) == 1); got len(model_parts) == "
                f"{len(self.model_parts)}. Iterate `train_module.model_parts` instead."
            )
        return self.model_parts[0]

    @property
    def dp_process_group(self) -> Optional[dist.ProcessGroup]:
        return get_dp_process_group(self.world_mesh)

    @property
    def eval_batch_spec(self) -> EvalBatchSpec:
        # Determine the number of micro-batches.
        rank_batch_size = self.trainer.global_batch_size // get_world_size(
            self.trainer.dp_process_group
        )
        rank_batch_size_instances = rank_batch_size // self.max_sequence_length
        return EvalBatchSpec(
            rank_batch_size=rank_batch_size_instances,
            batch_size_unit=EvalBatchSizeUnit.instances,
            max_sequence_length=self.max_sequence_length,
            fixed_sequence_length=True,
        )

    @property
    def tp_enabled(self) -> bool:
        return self._tp_config is not None

    @property
    def cp_enabled(self) -> bool:
        return self._cp_config is not None

    @property
    def dp_config(self) -> Optional[TransformerDataParallelConfig]:
        return self._dp_config

    @property
    def train_pp_schedule(self) -> PipelineSchedule:
        self.trainer  # make sure trainer has been attached before trying to access this
        assert self._train_pp_schedule is not None
        return self._train_pp_schedule

    @cached_property
    def _reduce_divide_factor(self) -> float:
        return get_reduce_divide_factor(get_world_size(self.dp_process_group))

    def loss_fn(self, output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # NOTE: the output is the loss.
        del labels
        return output

    def pre_train(self):
        # Validate batch size.
        dp_ws = get_world_size(self.trainer.dp_process_group)
        if self.trainer.global_batch_size % (self.rank_microbatch_size * dp_ws) != 0:
            raise OLMoConfigurationError(
                f"global batch size ({self.trainer.global_batch_size:,d}) must be divisible by "
                f"micro-batch size ({self.rank_microbatch_size:,d}) x DP world size ({dp_ws})"
            )

        # Initialize pipeline schedule.
        assert self._train_pp_schedule is None  # make sure we don't initialize this twice
        assert self._pp_stages is not None
        pp_mesh = get_pp_mesh(self.world_mesh)
        assert pp_mesh is not None

        # Determine the number of micro-batches.
        rank_batch_size = self.trainer.global_batch_size // dp_ws
        num_microbatches = rank_batch_size // self.rank_microbatch_size

        self._train_pp_schedule = PipelineSchedule(
            model_parts=self.model_parts,  # type: ignore[arg-type]
            stages=self._pp_stages,
            pp_mesh=pp_mesh,
            schedule_name=self._pp_config.schedule,
            num_microbatches=num_microbatches,
            forward_pull_ahead_extra_activations=self._pp_config.forward_pull_ahead_extra_activations,
            save_plot=self._pp_config.save_schedule_plot,
            plot_dir=self._pp_config.schedule_plot_dir,
        )

    def state_dict(self, *, optim: Optional[bool] = None) -> Dict[str, Any]:
        if optim is None:
            optim = bool(self.optimizers)
        return self._get_state_dict(self.state_dict_save_opts, optim=optim)

    def state_dict_to_load(
        self, metadata: Metadata, *, optim: Optional[bool] = None
    ) -> Dict[str, Any]:
        has_optim_state: bool = False
        for key in metadata.state_dict_metadata.keys():
            if key.startswith("optim."):
                has_optim_state = True
                break

        if optim is None:
            if not self.optimizers:
                # eval_only: never load optim state, even if the checkpoint has it
                optim = False
            elif not has_optim_state:
                log.warning("No optimizer state found in checkpoint")
                optim = False
            else:
                optim = True

        load_opts = self.state_dict_load_opts
        if optim:
            if not has_optim_state:
                raise RuntimeError(
                    "Checkpoint does not contain optimizer state, but 'optim=True' was requested"
                )

            if "optim.param_groups.0.params" in metadata.state_dict_metadata:
                # unflattened optimizer state
                if load_opts.flatten_optimizer_state_dict:
                    log.warning(
                        "Loading checkpoint with an unflattened optimizer state even though "
                        "'flatten_optimizer_state_dict=True' in train module's 'state_dict_load_opts', "
                        "automatically switching to 'flatten_optimizer_state_dict=False'."
                    )
                    load_opts = replace(load_opts, flatten_optimizer_state_dict=False)
            else:
                # flattened optimizer state
                if not load_opts.flatten_optimizer_state_dict:
                    log.warning(
                        "Loading checkpoint with a flattened optimizer state even though "
                        "'flatten_optimizer_state_dict=False' in train module's 'state_dict_load_opts', "
                        "automatically switching to 'flatten_optimizer_state_dict=True'."
                    )
                    load_opts = replace(load_opts, flatten_optimizer_state_dict=True)

        state_dict = self._get_state_dict(load_opts, optim=optim)
        if self.load_key_mapping is not None:
            swap_param_keys(state_dict, self.load_key_mapping, metadata=metadata)

        if not load_opts.strict:
            # Remove any keys in the 'state_dict' that are not present in the checkpoint.
            pruned_keys = prune_state_dict(state_dict, set(metadata.state_dict_metadata.keys()))
            if pruned_keys:
                log.warning(f"Checkpoint is missing the following keys: {pruned_keys}")

        return state_dict

    def state_dict_to_save(self, *, optim: Optional[bool] = None) -> Dict[str, Any]:
        if optim is None:
            optim = bool(self.optimizers)
        return self._get_state_dict(self.state_dict_save_opts, optim=optim)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        load_optim = "optim" in state_dict

        if self.load_key_mapping is not None:
            swap_param_keys(state_dict, self.load_key_mapping, reverse=True, quiet=True)

        # NOTE: `dist_cp_sd.set_(model|optimizer)_state_dict()` doesn't respect `strict=False`
        # option with missing keys, so we have to handle that on our own.
        if not self.state_dict_load_opts.strict:
            flatten_optimizer_state_dict = (
                False if not load_optim else ("state" not in state_dict["optim"])
            )
            load_opts = replace(
                self.state_dict_load_opts, flatten_optimizer_state_dict=flatten_optimizer_state_dict
            )
            full_state_dict = self._get_state_dict(load_opts, optim=load_optim)
            merge_state_dicts(state_dict, full_state_dict)

        # In eval_only mode ``self.optimizers`` is empty, so we iterate model_parts directly to
        # ensure model state still gets loaded; ``load_optim`` is False in that case.
        optimizers_or_none: List[Optional[Optimizer]] = (
            list(self.optimizers) if self.optimizers else [None] * len(self.model_parts)
        )
        for model, optim in zip(self.model_parts, optimizers_or_none):
            dist_cp_sd.set_model_state_dict(
                model,
                state_dict["model"],
                options=self.state_dict_load_opts,
            )
            gc_cuda()
            if load_optim:
                assert optim is not None
                dist_cp_sd.set_optimizer_state_dict(
                    model,
                    optim,
                    state_dict["optim"],
                    options=self.state_dict_load_opts,
                )
                gc_cuda()

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        # Set model to train mode if it isn't already.
        for model in self.model_parts:
            model.train()

        # Generate labels.
        labels = batch.get("labels", get_labels(batch, label_ignore_index=self.label_ignore_index))

        # Record how many instances are going to be skipped (masked out).
        if (instance_mask := batch.get("instance_mask")) is not None and not dry_run:
            self.record_metric(
                "train/masked instances (%)", (~instance_mask).float().mean(), ReduceType.mean
            )

        # Calculate how many tokens are going to be used in the loss.
        batch_num_tokens_for_loss = (labels != self.label_ignore_index).sum().item()

        # Run pipeline schedule.
        input_ids, labels, model_kwargs = self._prepare_batch(batch, labels)
        assert labels is not None
        ce_batch_loss, z_batch_loss = self.run_pipeline(
            input_ids,
            labels,
            batch_num_tokens_for_loss,
            ignore_index=self.label_ignore_index,
            loss_reduction="sum",
            z_loss_multiplier=self.z_loss_multiplier,
            return_logits=False,
            **model_kwargs,
        )

        for model in self.model_parts:
            model.post_batch(dry_run=dry_run)

        if dry_run:
            for model in self.model_parts:
                model.reset_auxiliary_metrics()
            return

        # Record all of the losses we captured.
        # If we have a SkipStepOptimizer we'll reduce the loss (if we have the final stage) across
        # the DP process group and then asynchronously send to the ranks in the PP group.
        # NOTE: main losses will be missing for non-final stages.
        if isinstance(self.optimizers[0], SkipStepOptimizer):
            ce_batch_loss = self.reduce_send_recv(ce_batch_loss)
            for optim in self.optimizers:
                cast(SkipStepOptimizer, optim).latest_loss = ce_batch_loss
            self.record_ce_loss(ce_batch_loss)
        elif ce_batch_loss is not None:
            self.record_ce_loss(ce_batch_loss, ReduceType.mean)

        if z_batch_loss is not None:
            assert self.z_loss_multiplier
            self.record_metric(
                "Z loss",
                z_batch_loss,
                ReduceType.mean,
                namespace="train",
            )
            self.record_metric(
                "Z loss unscaled",
                z_batch_loss / self.z_loss_multiplier,
                ReduceType.mean,
                namespace="train",
            )

        # And additional metrics.
        for model in self.model_parts:
            for metric_name, (metric_val, reduction) in model.compute_auxiliary_metrics(
                reset=True,
            ).items():
                merge_strategy = MetricMergeStrategy.warn
                if reduction in (ReduceType.sum, ReduceType.mean):
                    merge_strategy = MetricMergeStrategy.sum
                elif reduction == ReduceType.max:
                    merge_strategy = MetricMergeStrategy.max
                self.record_metric(
                    metric_name,
                    metric_val,
                    reduction,
                    namespace="train",
                    merge_strategy=merge_strategy,
                )

    def reduce_send_recv(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.pp_group_rank == self.pp_final_stage_rank:
            assert x is not None
            # Reduce across DP process group.
            x.div_(self._reduce_divide_factor)
            dist.all_reduce(x, group=self.dp_process_group)
            x.div_(self.dp_world_size)
            x.mul_(self._reduce_divide_factor)
        else:
            assert x is None
            x = move_to_device(torch.empty([]), self.device)

        # Asynchronously send to previous stage rank in the PP group.
        ordered_ranks = list(self._pp_config.rank_completion_order())
        local_index = ordered_ranks.index(self.pp_group_rank)
        src_rank = None if local_index == 0 else ordered_ranks[local_index - 1]
        dst_rank = (
            None if local_index == (len(ordered_ranks) - 1) else ordered_ranks[local_index + 1]
        )

        ops: List[dist.P2POp] = []
        if src_rank is not None:
            log.debug(
                f"Rank {get_rank()} (pp group rank {self.pp_group_rank}) receiving from rank "
                f"{get_global_rank(src_rank, group=self.pp_group)} (pp group rank {src_rank})"
            )
            ops.append(dist.P2POp(dist.irecv, x, group=self.pp_group, group_peer=src_rank))
        if dst_rank is not None:
            log.debug(
                f"Rank {get_rank()} (pp group rank {self.pp_group_rank}) sending to rank "
                f"{get_global_rank(dst_rank, group=self.pp_group)} (pp group rank {dst_rank})"
            )
            ops.append(dist.P2POp(dist.isend, x, group=self.pp_group, group_peer=dst_rank))

        reqs = dist.batch_isend_irecv(ops)
        if self.pp_group_rank != self.pp_final_stage_rank:
            for req in reqs:
                req.wait()

        return x

    def eval_batch(self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None) -> Any:
        del batch, labels
        raise RuntimeError(f"{self.__class__.__name__} does not support inference")

    def optim_step(self):
        self._require_optimizers()
        # Maybe clip gradients.
        if self.max_grad_norm is not None:
            grad_norm = self._clip_grad_norm(self.max_grad_norm)
            # NOTE: grad norm is already reduced over ranks, so we set `reduce_type` to `None`.
            self.trainer.record_metric(
                "total grad norm", grad_norm, reduce_type=None, namespace="optim"
            )
            for optim in self.optimizers:
                if isinstance(optim, SkipStepOptimizer):
                    optim.latest_grad_norm = grad_norm

        # Maybe adjust learning rate.
        if self.scheduler is not None:
            for optim in self.optimizers:
                for group_idx, group in enumerate(optim.param_groups):
                    new_lr = self.scheduler.set_lr(group, self.trainer)
                    self.trainer.record_metric(
                        f"LR (group {group_idx})",
                        new_lr,
                        namespace="optim",
                        merge_strategy=MetricMergeStrategy.latest,
                    )

        # Step optimizer.
        for optim in self.optimizers:
            optim.step()
            if isinstance(optim, SkipStepOptimizer):
                self.record_metric(
                    "step skipped",
                    optim.step_skipped,
                    namespace="optim",
                    merge_strategy=MetricMergeStrategy.latest,
                )

        for model in self.model_parts:
            model.post_optim_step()

    def zero_grads(self):
        for optim in self._require_optimizers():
            optim.zero_grad(set_to_none=True)

    def run_pipeline(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        batch_num_tokens_for_loss: Union[int, float],
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Run the pipeline, returning the losses captured.
        """

        ce_batch_loss: Optional[torch.Tensor] = None
        z_batch_loss: Optional[torch.Tensor] = None

        def capture_losses(
            model: Transformer, args: Tuple[torch.Tensor, ...], output: Any
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            del args
            nonlocal ce_batch_loss
            nonlocal z_batch_loss

            if model.lm_head is not None:
                assert isinstance(output, LMOutputWithLoss)
                _, loss, ce_loss, z_loss = output
                if ce_batch_loss is None:
                    ce_batch_loss = get_local_tensor(ce_loss.detach())
                else:
                    ce_batch_loss += get_local_tensor(ce_loss.detach())

                if self.z_loss_multiplier is not None:
                    assert z_loss is not None
                    if z_batch_loss is None:
                        z_batch_loss = get_local_tensor(z_loss.detach())
                    else:
                        z_batch_loss += get_local_tensor(z_loss.detach())

                return loss.unsqueeze(0)
            else:
                assert isinstance(output, torch.Tensor)
                return output

        handles = []
        for model in self.model_parts:
            handles.append(model.register_forward_hook(capture_losses))

        with self._model_forward_context():
            self.train_pp_schedule.step(
                input_ids,
                target=labels,
                loss_div_factor=batch_num_tokens_for_loss,
                labels=labels,
                **kwargs,
            )

        for handle in handles:
            handle.remove()

        return ce_batch_loss, z_batch_loss

    @lru_cache
    def num_flops_per_token(self, seq_len: int) -> Optional[int]:
        try:
            return self._full_model_num_flops_per_token(seq_len)
        except NotImplementedError as ex:
            warn_once(f"Unable to estimate num flops per token: {ex}")
            return None

    def global_num_flops_in_batch(self, batch: Dict[str, Any]) -> Optional[int]:
        global_num_tokens = self.trainer.data_loader.global_num_tokens_in_batch(batch)
        if global_num_tokens is None:
            return None
        flops_per_token = self.num_flops_per_token(seq_len=batch["input_ids"].shape[1])
        return flops_per_token * global_num_tokens if flops_per_token is not None else None

    @contextlib.contextmanager
    def _model_forward_context(self) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            if self.autocast_precision is not None:
                stack.enter_context(torch.autocast(self.device.type, dtype=self.autocast_precision))
            yield

    def _get_state_dict(
        self, sd_options: dist_cp_sd.StateDictOptions, optim: bool = True
    ) -> Dict[str, Any]:
        state_dict: Dict[str, Any] = {
            "model": {
                k: v
                for sd in map(
                    partial(dist_cp_sd.get_model_state_dict, options=sd_options), self.model_parts
                )
                for k, v in sd.items()
            }
        }
        if optim:
            state_dict["optim"] = {
                k: v
                for sd in map(
                    partial(get_optimizer_state_dict, options=sd_options),
                    self.model_parts,
                    self._require_optimizers(),
                )
                for k, v in sd.items()
            }

        return state_dict

    def _clip_grad_norm(
        self, max_grad_norm: float, norm_type: float = 2.0, foreach: Optional[bool] = None
    ) -> torch.Tensor:
        # Adapted from https://github.com/pytorch/torchtitan/blob/2a4437014e66bcf88a3f0419b816266e6326d539/torchtitan/utils.py#L348

        parameters = [p for m in self.model_parts for p in m.parameters()]
        grads = [p.grad for p in parameters if p.grad is not None]

        total_norm = nn.utils.get_total_norm(
            grads, norm_type=norm_type, error_if_nonfinite=False, foreach=foreach
        )

        # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
        # We can simply reduce the DTensor to get the total norm in this tensor's process group
        # and then convert it to a local tensor.
        # NOTE: It has two purposes:
        #       1. to make sure the total norm is computed correctly when PP is used (see below)
        #       2. to return a reduced total_norm tensor whose .item() would return the correct value
        if isinstance(total_norm, DTensor):
            # Will reach here if any non-PP parallelism is used.
            # If only using PP, total_norm will be a local tensor.
            total_norm = total_norm.full_tensor()

        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=self.pp_group)
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=self.pp_group)
            total_norm **= 1.0 / norm_type

        torch.nn.utils.clip_grads_with_norm_(parameters, max_grad_norm, total_norm, foreach=foreach)
        return total_norm

    def _prepare_batch(
        self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        input_ids = batch.pop("input_ids")
        labels = labels if labels is not None else batch.pop("labels", None)
        kwargs: Dict[str, Any] = {}
        if "doc_lens" in batch and "max_doc_lens" in batch:
            log_once(log, "intra-document masking enabled")
            kwargs["doc_lens"] = batch["doc_lens"]
            kwargs["max_doc_lens"] = batch["max_doc_lens"]
        return input_ids, labels, kwargs
