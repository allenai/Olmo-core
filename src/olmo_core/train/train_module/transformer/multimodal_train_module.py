"""Multimodal training with globally normalized response-token losses.

:class:`MultimodalTransformerTrainModule` supports DDP, FSDP, and HSDP for
:class:`~olmo_core.nn.vision.MultimodalLM`. :class:`MultimodalOLMoDDPTrainModule` adds
multimodal batch handling to OLMoDDP's data- and expert-parallel training path. Both accept
per-token loss weights and optional independently normalized loss groups. Router auxiliary
losses use valid input tokens, independently of response-token loss weights.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
from dataclasses import dataclass
from fnmatch import fnmatch
from functools import lru_cache
from typing import (
    Any,
    ClassVar,
    Collection,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    cast,
)

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from torch.distributed.checkpoint.metadata import Metadata
from torch.optim import Optimizer

from olmo_core.aliases import PathOrStr
from olmo_core.config import DType
from olmo_core.data.utils import split_batch
from olmo_core.distributed.checkpoint import (
    RemoteFileSystemReader,
    RemoteFileSystemWriter,
)
from olmo_core.distributed.parallel import (
    DataParallelType,
    build_world_mesh,
    get_dp_model_mesh,
)
from olmo_core.distributed.utils import (
    get_local_tensor,
    get_rank,
    get_world_size,
    is_distributed,
    reduce_distributed_failure_flag,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.functional import weighted_cross_entropy_loss
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.optim import OptimConfig
from olmo_core.optim.multimodal_optimizer import MultimodalOLMoDDPOptimizer
from olmo_core.optim.scheduler import Scheduler
from olmo_core.utils import get_default_device, move_to_device, warn_once

from ...common import ReduceType
from ..config import TrainModuleConfig
from ..train_module import EvalBatchSpec, TrainModule
from .config import (
    OLMoDDPTrainModuleConfig,
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
)
from .ddp_train_module import FlatSavePlanner, OLMoDDPTrainModule, _prepare_env_for_save
from .train_module import TransformerTrainModule

log = logging.getLogger(__name__)

__all__ = [
    "MultimodalTransformerTrainModule",
    "MultimodalTransformerTrainModuleConfig",
    "MultimodalOLMoDDPTrainModule",
    "MultimodalOLMoDDPTrainModuleConfig",
]


def _mm_train_verbose_logs() -> bool:
    """Per-step batch/optim diagnostics (forces CUDA sync via ``.item()``)."""
    return os.environ.get("MM_TRAIN_VERBOSE_LOGS", "0").lower() in ("1", "true", "yes")


def _retain_embedding_gradient_rows(grad: torch.Tensor, row_ids: Tuple[int, ...]) -> torch.Tensor:
    """Return an embedding gradient with only ``row_ids`` retained."""
    row_mask = torch.zeros((grad.shape[0], 1), dtype=grad.dtype, device=grad.device)
    row_mask[list(row_ids)] = 1
    return grad * row_mask


def _matched_component_grad_norm_patterns(
    component_patterns: Mapping[str, Tuple[str, ...]], trainable_names: set[str]
) -> Dict[str, Tuple[str, ...]]:
    """Keep diagnostic components that match at least one trainable optimizer parameter."""
    return {
        component: patterns
        for component, patterns in component_patterns.items()
        if any(fnmatch(name, pattern) for name in trainable_names for pattern in patterns)
    }


def _validate_loss_group_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    if weights is None:
        return {}
    if (
        not weights
        or any(
            not isinstance(name, str) or not name or not math.isfinite(value) or value <= 0
            for name, value in weights.items()
        )
        or abs(sum(weights.values()) - 1.0) > 1e-6
    ):
        raise OLMoConfigurationError("loss_group_weights must be positive and sum to one")
    return dict(sorted(weights.items()))


def _normalize_loss_groups(
    batch: Dict[str, Any],
    group_weights: Dict[str, float],
    *,
    label_ignore_index: int,
    device: torch.device,
    dp_process_group: Optional[dist.ProcessGroup],
) -> Tuple[Dict[str, Any], torch.Tensor]:
    """Rescale annotation weights to a globally normalized sum of group objectives.

    This runs once for the full optimizer batch, before accumulation splits it. If ``M_g``
    is global active annotation mass for group ``g``, use ``w * alpha_g * M / M_g``.
    The existing global divisor and averaged DP gradients then yield
    ``sum_g alpha_g * sum_i(w_i * CE_i) / M_g``. LM z-loss uses the same weights;
    router auxiliary losses retain their independent valid-token normalization.
    """
    masks = batch.get("loss_masks")
    labels = batch.get("labels")
    names = batch.get("loss_group_names")
    shape = batch["input_ids"].shape
    error = None
    if (
        not isinstance(masks, torch.Tensor)
        or not isinstance(labels, torch.Tensor)
        or masks.shape != shape
        or labels.shape != shape
        or not isinstance(names, list)
        or len(names) != shape[0]
        or any(not isinstance(name, str) or name not in group_weights for name in names)
    ):
        error = "Group-normalized loss requires aligned labels, loss_masks, and loss_group_names"
    elif not bool(torch.isfinite(masks).all()) or bool((masks < 0).any()):
        error = "Group-normalized loss requires finite nonnegative annotation weights"
    # Even malformed metadata on one rank must fail before peers enter mass reductions.
    if reduce_distributed_failure_flag(error is not None, device, group=dp_process_group):
        raise OLMoConfigurationError(error or "Invalid loss-group batch on another DP rank")
    assert isinstance(masks, torch.Tensor) and isinstance(labels, torch.Tensor)
    assert isinstance(names, list)
    active_weights = masks.to(device=device, dtype=torch.float32) * (
        labels.to(device) != label_ignore_index
    )
    group_indices = {name: index for index, name in enumerate(group_weights)}
    row_groups = torch.tensor(
        [group_indices[name] for name in names], device=device, dtype=torch.long
    )
    local_mass = torch.stack(
        [active_weights[row_groups == index].sum() for index in range(len(group_weights))]
    )
    global_mass = local_mass.clone()
    if is_distributed():
        dist.all_reduce(global_mass, group=dp_process_group)
    if not bool(torch.isfinite(global_mass).all()) or bool((global_mass <= 0).any()):
        raise OLMoConfigurationError(
            "Every configured loss group must have positive finite supervised mass globally "
            "in every optimizer batch (ignored labels do not count)"
        )
    # Both train modules guard the divisor against zero; one clamps before dividing by
    # DP size and one afterwards. Keep the per-rank reference mass >= 1 so either path
    # preserves the objective, even with very small fractional annotation weights.
    reference_mass = global_mass.sum().clamp_min(float(get_world_size(dp_process_group)))
    coefficients = torch.tensor(list(group_weights.values()), device=device)
    scales = coefficients * reference_mass / global_mass
    normalized_batch = dict(batch)
    normalized_batch["loss_masks"] = active_weights * scales[row_groups, None]
    return normalized_batch, global_mass


def _trim_microbatch_image_padding(batch: dict[str, Any]) -> dict[str, Any]:
    """Remove unused trailing crop/pooling slots while retaining one dummy slot."""
    images = batch.get("images")
    pooled = batch.get("pooled_patches_idx")
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise OLMoConfigurationError("Image-padding trimming requires rank-4 images")
    if not isinstance(pooled, torch.Tensor) or pooled.ndim != 3:
        raise OLMoConfigurationError("Image-padding trimming requires rank-3 pooled_patches_idx")
    size, crops, patches, _ = images.shape
    if size == 0 or crops == 0 or patches == 0 or pooled.shape[0] != size or pooled.shape[1] == 0:
        raise OLMoConfigurationError(
            "Image-padding trimming requires aligned nonempty image tensors"
        )
    counts = []
    for name, limit in (("image_crop_counts", crops), ("pooled_token_counts", pooled.shape[1])):
        value = batch.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (size,)
            or value.dtype not in (torch.int32, torch.int64)
            or bool((value < 0).any())
            or bool((value > limit).any())
        ):
            raise OLMoConfigurationError(
                f"Image-padding trimming requires integer {name} with shape ({size},) "
                f"and values in [0, {limit}]"
            )
        counts.append(value.to(device=pooled.device))
    crop_counts, pooled_counts = counts
    if pooled.dtype not in (torch.int32, torch.int64) or bool((pooled < -1).any()):
        raise OLMoConfigurationError("pooled_patches_idx must contain integer patch indices or -1")
    if bool((pooled >= crop_counts[:, None, None] * patches).any()):
        raise OLMoConfigurationError(
            "Pooled patch indices reference crops outside image_crop_counts"
        )
    trailing = (
        torch.arange(pooled.shape[1], device=pooled.device)[None, :] >= pooled_counts[:, None]
    )
    if bool(((pooled >= 0) & trailing[:, :, None]).any()):
        raise OLMoConfigurationError("pooled_token_counts would discard non-padding pooled rows")
    out = dict(batch)
    out["images"] = images[:, : max(int(crop_counts.max()), 1)]
    out["pooled_patches_idx"] = pooled[:, : max(int(pooled_counts.max()), 1)]
    return out


class MultimodalTransformerTrainModule(TransformerTrainModule):
    """A :class:`TrainModule` for :class:`~olmo_core.nn.vision.MultimodalLM` stage-1 training."""

    optim: Optimizer

    def __init__(
        self,
        model: torch.nn.Module,
        optim: OptimConfig,
        rank_microbatch_size: int,
        max_sequence_length: int,
        *,
        freeze_params: Optional[List[str]] = None,
        z_loss_multiplier: Optional[float] = None,
        autocast_precision: Optional[torch.dtype] = None,
        max_grad_norm: Optional[float] = None,
        scheduler: Optional[Scheduler] = None,
        device: Optional[torch.device] = None,
        compile_model: bool = False,
        compile_vision: bool = True,
        compile_connector: bool = True,
        dp_config: Optional[TransformerDataParallelConfig] = None,
        ac_config: Optional[TransformerActivationCheckpointingConfig] = None,
        vision_activation_checkpointing: bool = True,
        connector_activation_checkpointing: bool = True,
        label_ignore_index: int = -100,
        response_logits_only: bool = False,
        loss_group_weights: Optional[Dict[str, float]] = None,
        state_dict_save_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        state_dict_load_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        load_key_mapping: Optional[Dict[str, str]] = None,
    ):
        # TransformerTrainModule initialization requires a Transformer; multimodal models apply
        # parallelism to their LM, vision encoder, and connector directly below.
        TrainModule.__init__(self)

        if rank_microbatch_size % max_sequence_length != 0:
            raise OLMoConfigurationError(
                f"'rank_microbatch_size' ({rank_microbatch_size:,d} tokens) must be divisible by "
                f"'max_sequence_length' ({max_sequence_length:,d} tokens)"
            )
        if dp_config is not None and dp_config.name not in (
            DataParallelType.ddp,
            DataParallelType.fsdp,
            DataParallelType.hsdp,
        ):
            raise OLMoConfigurationError(
                "MultimodalTransformerTrainModule only supports DDP / FSDP / HSDP data "
                f"parallelism (got dp_config.name={dp_config.name!r}); TP/CP/PP/EP of the "
                "multimodal model are not yet supported."
            )

        self.device = device or get_default_device()
        self.world_mesh = None
        if is_distributed():
            self.world_mesh = build_world_mesh(dp=dp_config, device_type=self.device.type)
        elif dp_config is not None:
            raise OLMoConfigurationError(
                "Training parallelism configs are only valid for distributed training"
            )

        # Freeze parameters (e.g. the vision encoder for stage 1) before building the
        # optimizer so frozen params are excluded from optimizer groups.
        self.freeze_params = freeze_params or []
        n_frozen = 0
        for name, p in model.named_parameters():
            if any(fnmatch(name, pat) for pat in self.freeze_params):
                p.requires_grad_(False)
                n_frozen += 1
        if self.freeze_params:
            log.info(f"Froze {n_frozen} parameter tensors matching {self.freeze_params}")

        model.to(self.device)
        if vision_activation_checkpointing and hasattr(
            model.vision, "apply_activation_checkpointing"
        ):
            model.vision.apply_activation_checkpointing()
            log.info("Applied per-block activation checkpointing to model.vision")
        if connector_activation_checkpointing and hasattr(
            model.connector, "apply_activation_checkpointing"
        ):
            model.connector.apply_activation_checkpointing()
            log.info("Applied activation checkpointing to model.connector")
        if ac_config is not None:
            model.lm.apply_activation_checkpointing(
                ac_config.mode,
                block_interval=ac_config.block_interval,
                modules=ac_config.modules,
                activation_memory_budget=ac_config.activation_memory_budget,
                determinism_check=ac_config.determinism_check,
            )
            log.info("Applied '%s' activation checkpointing to model.lm", ac_config.mode)
        if compile_model:
            log.info("Compiling model.lm blocks ...")
            model.lm.apply_compile()
            if compile_vision and hasattr(model.vision, "apply_compile"):
                log.info("Compiling vision encoder blocks ...")
                model.vision.apply_compile()
            if compile_connector and hasattr(model.connector, "apply_compile"):
                log.info("Compiling connector ...")
                model.connector.apply_compile()
        self.model = model
        self._model_mode = None

        self._dp_config = dp_config
        self._cp_config = None
        self._tp_config = None
        self._ep_config = None
        self.label_ignore_index = label_ignore_index
        self.response_logits_only = response_logits_only
        self.loss_group_weights = _validate_loss_group_weights(loss_group_weights)
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
            flatten_optimizer_state_dict=True, strict=True
        )
        self.load_key_mapping = load_key_mapping

        # Apply data parallelism IN-PLACE *before* building the optimizer: composable
        # DDP/FSDP keep the model's type, attributes, and (prefix-free) parameter names,
        # and FSDP additionally needs the optimizer built on the sharded DTensor params.
        if self.world_mesh is not None:
            assert dp_config is not None
            self._parallelize(dp_config)

        log.info("Building optimizer...")
        self.optim = optim.build(self.model, strict=True)

    def _parallelize(self, dp_config: TransformerDataParallelConfig) -> None:
        """Apply DDP (``replicate``) or FSDP (``fully_shard``) to the multimodal model
        in-place. FSDP shards the LM (the bulk of the parameters), the vision encoder,
        and the connector across the DP mesh so the 4B model + optimizer fit per GPU."""
        assert self.world_mesh is not None
        dp_mesh = get_dp_model_mesh(self.world_mesh)
        if dp_config.name == DataParallelType.ddp:
            from torch.distributed._composable.replicate import replicate

            replicate(self.model, device_mesh=dp_mesh, bucket_cap_mb=100)
        else:  # fsdp / hsdp
            from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

            param_dtype = (
                dp_config.param_dtype.as_pt() if dp_config.param_dtype is not None else None
            )
            reduce_dtype = dp_config.reduce_dtype.as_pt()
            # Shard the language model with its own (per-block) FSDP wrapping.
            self.model.lm.apply_fsdp(
                dp_mesh=dp_mesh,
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                wrapping_strategy=dp_config.wrapping_strategy,
                prefetch_factor=dp_config.prefetch_factor,
            )
            # Shard the vision encoder + connector, then the root so ``self.model`` is an
            # FSDPModule (the inherited micro-batch / gradient-sync handling keys off this).
            mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
            fully_shard(self.model.vision, mesh=dp_mesh, mp_policy=mp)
            fully_shard(self.model.connector, mesh=dp_mesh, mp_policy=mp)
            fully_shard(self.model, mesh=dp_mesh, mp_policy=mp)

    @property
    def _multimodal(self) -> torch.nn.Module:
        # ``replicate`` is applied in-place, so ``self.model`` is the MultimodalLM itself.
        return self.model

    @property
    def _lm(self) -> torch.nn.Module:
        return self._multimodal.lm

    @property
    def eval_batch_spec(self) -> EvalBatchSpec:
        return EvalBatchSpec(
            self.rank_microbatch_size, max_sequence_length=self.max_sequence_length
        )

    def _prepare_batch(  # type: ignore[override]
        self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Dict[str, Any]]:
        """Split off ``input_ids`` / ``labels`` / float ``loss_masks``; the rest
        (``images``, ``pooled_patches_idx``, ``token_type_ids``, ``subsegment_ids``,
        ``position_ids``) flows to :meth:`MultimodalLM.forward` as kwargs."""
        input_ids = batch.pop("input_ids")
        labels = labels if labels is not None else batch.pop("labels", None)
        loss_masks = batch.pop("loss_masks")
        batch.pop("pack_source_names", None)
        batch.pop("image_crop_counts", None)
        batch.pop("pooled_token_counts", None)
        batch.pop("loss_group_names", None)
        return input_ids, labels, loss_masks, batch

    def _set_model_mode(self, mode: Literal["train", "eval"]):
        super()._set_model_mode(mode)
        if mode == "train" and any(fnmatch(name, "vision.*") for name in self.freeze_params):
            self._multimodal.vision.eval()

    def _log_batch_sources(self, batch: Dict[str, Any], local_weight: torch.Tensor) -> None:
        """Log per-rank packed source names when verbose diagnostics are enabled."""
        if not _mm_train_verbose_logs():
            return
        sources = batch.get("pack_source_names")
        if sources is None:
            return
        images = batch.get("images")
        n_crops = int(images.shape[1]) if images is not None else 0
        n_im_patch = int(
            (batch["input_ids"] == self._multimodal.cfg.image_patch_token_id).sum().item()
        )
        log.info(
            "batch sources rank=%d sources=%s local_weight=%.1f im_patch=%d crops=%d shape=%s",
            get_rank(),
            sources,
            float(local_weight.item()),
            n_im_patch,
            n_crops,
            tuple(batch["input_ids"].shape),
        )

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        self._set_model_mode("train")
        if self.loss_group_weights:
            batch, _ = _normalize_loss_groups(
                batch,
                self.loss_group_weights,
                label_ignore_index=self.label_ignore_index,
                device=self.device,
                dp_process_group=self.dp_process_group,
            )

        # Global loss-weight divisor (mm_olmo BatchDivisor.global_batch): the sum of
        # positive loss weights over the whole global batch, divided by DP world size.
        # After DDP averages gradients across ranks, the effective divisor is the global
        # weight. For a single rank this is just the local weight sum.
        loss_masks = batch["loss_masks"].to(self.device).float()
        local_weight = (loss_masks * (loss_masks > 0)).sum()
        if is_distributed():
            div_factor = local_weight.clone()
            dist.all_reduce(div_factor, group=self.dp_process_group)
            div_factor = div_factor / get_world_size(self.dp_process_group)
        else:
            div_factor = local_weight
        div_factor = torch.clamp(div_factor, min=1.0)

        pack_sources = batch.get("pack_source_names")
        if not dry_run:
            self._log_batch_sources(batch, local_weight)

        ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        z_batch_loss: Optional[torch.Tensor] = (
            move_to_device(torch.tensor(0.0), self.device)
            if self.z_loss_multiplier is not None
            else None
        )
        weight_total = move_to_device(torch.tensor(0.0), self.device)

        if self.rank_microbatch_size < (seq_len := batch["input_ids"].shape[1]):
            raise RuntimeError(
                f"Microbatch size ({self.rank_microbatch_size}) is too small relative to "
                f"sequence length ({seq_len})"
            )
        micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)
        num_micro_batches = len(micro_batches)

        if get_rank() == 0 and not dry_run and _mm_train_verbose_logs():
            images = batch.get("images")
            bsz, seq_len = batch["input_ids"].shape[:2]
            n_crops = int(images.shape[1]) if images is not None else 0
            vit_bt = bsz * n_crops
            gpu_mem_gb = (
                torch.cuda.max_memory_allocated(self.device) / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            )
            log.info(
                "batch shapes: input_ids=%s crops/seq=%d vit_B*T=%d gpu_max_alloc_gb=%.2f",
                tuple(batch["input_ids"].shape),
                n_crops,
                vit_bt,
                gpu_mem_gb,
            )

        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                input_ids, labels, mb_loss_masks, model_kwargs = self._prepare_batch(micro_batch)
                assert labels is not None
                mb_loss_masks = mb_loss_masks.to(self.device).float()

                # ``labels`` / ``loss_masks`` are already next-token-aligned (shifted) by
                # the data pipeline, so no additional shift here.
                with self._model_forward_context():
                    if self.response_logits_only:
                        logits = self.model(
                            input_ids,
                            labels=None,
                            response_logits_only=True,
                            loss_masks=mb_loss_masks,
                            **model_kwargs,
                        )
                        response_mask = mb_loss_masks > 0
                        flat_logits = logits
                        flat_labels = labels.to(self.device).reshape(-1)[response_mask.reshape(-1)]
                        flat_weights = mb_loss_masks.reshape(-1)[response_mask.reshape(-1)]
                    else:
                        logits = self.model(
                            input_ids, labels=None, loss_masks=mb_loss_masks, **model_kwargs
                        )
                        vocab_size = logits.shape[-1]
                        flat_logits = logits.reshape(-1, vocab_size)
                        flat_labels = labels.to(self.device).reshape(-1)
                        flat_weights = mb_loss_masks.reshape(-1)
                        # Mask out non-loss positions from the CE target for safety.
                        flat_labels = torch.where(
                            flat_weights > 0,
                            flat_labels,
                            flat_labels.new_full((), self.label_ignore_index),
                        )

                ce_loss, z_loss = weighted_cross_entropy_loss(
                    flat_logits,
                    flat_labels,
                    flat_weights,
                    ignore_index=self.label_ignore_index,
                    compute_z_loss=self.z_loss_multiplier is not None and not dry_run,
                    z_loss_multiplier=self.z_loss_multiplier or 1e-4,
                )

                # Every rank must enter the distributed failure reduction on every
                # microbatch. Otherwise a failing rank could issue this collective while
                # healthy ranks continue into backward collectives and hang NCCL.
                local_failed = not bool(torch.isfinite(ce_loss).item())
                if reduce_distributed_failure_flag(
                    local_failed, self.device, group=self.dp_process_group
                ):
                    if local_failed:
                        n_im_patch = int(
                            (input_ids == self._multimodal.cfg.image_patch_token_id).sum()
                        )
                        raise RuntimeError(
                            f"Non-finite CE loss on rank {get_rank()}: ce={ce_loss.item()}, "
                            f"local_weight={local_weight.item():.4f}, "
                            f"logits_nan={bool(torch.isnan(logits).any())}, "
                            f"logits_inf={bool(torch.isinf(logits).any())}, "
                            f"im_patch_tokens={n_im_patch}, seq_len={input_ids.shape[1]}, "
                            f"sources={pack_sources}"
                        )
                    raise RuntimeError(
                        f"Training failed on another rank (rank {get_rank()} had finite CE)"
                    )

                if dry_run:
                    continue

                loss = ce_loss / div_factor
                if z_loss is not None:
                    loss = loss + z_loss / div_factor

                ce_batch_loss += get_local_tensor(ce_loss.detach())
                weight_total += get_local_tensor((flat_weights > 0).sum().detach()).float()
                if z_batch_loss is not None and z_loss is not None:
                    z_batch_loss += get_local_tensor(z_loss.detach())

                loss.backward()

        del batch

        # Delegate auxiliary-metric bookkeeping to the underlying Transformer.
        if hasattr(self._lm, "post_batch"):
            self._lm.post_batch(dry_run=dry_run)
        if dry_run:
            if hasattr(self._lm, "reset_auxiliary_metrics"):
                self._lm.reset_auxiliary_metrics()
            return

        # Record a per-weighted-token CE loss (comparable across steps).
        mean_ce = ce_batch_loss / torch.clamp(local_weight, min=1.0)
        self.record_ce_loss(mean_ce, ReduceType.mean)
        if z_batch_loss is not None:
            assert self.z_loss_multiplier is not None
            mean_z = z_batch_loss / torch.clamp(local_weight, min=1.0)
            self.record_metric("Z loss", mean_z, ReduceType.mean, namespace="train")

        if hasattr(self._lm, "compute_auxiliary_metrics"):
            for metric_name, (
                metric_val,
                reduction,
            ) in self._lm.compute_auxiliary_metrics(reset=True).items():
                self.record_metric(metric_name, metric_val, reduction, namespace="train")

        if not dry_run and _mm_train_verbose_logs():
            log.info(
                "train_batch rank=%d complete local_weight=%.1f",
                get_rank(),
                float(local_weight.item()),
            )

    def optim_step(self):
        if self.max_grad_norm is not None:
            grad_norm = self._clip_grad_norm(self.max_grad_norm)
            self.trainer.record_metric(
                "total grad norm", grad_norm, reduce_type=None, namespace="optim"
            )

        if self.scheduler is not None:
            for group_idx, group in enumerate(self.optim.param_groups):
                new_lr = self.scheduler.set_lr(group, self.trainer)
                self.trainer.record_metric(f"LR (group {group_idx})", new_lr, namespace="optim")

        self.optim.step()

        if hasattr(self._lm, "post_optim_step"):
            self._lm.post_optim_step()

    def eval_batch(self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None):
        raise NotImplementedError(
            "In-loop evaluation is not implemented for MultimodalTransformerTrainModule "
            "(stage-1 training runs without in-loop eval)."
        )

    @lru_cache
    def num_flops_per_token(self, seq_len: int) -> Optional[int]:
        try:
            if hasattr(self._lm, "num_flops_per_token"):
                return self._lm.num_flops_per_token(seq_len)
        except NotImplementedError as ex:
            warn_once(f"Unable to estimate num flops per token: {ex}")
        return None

    def extra_flops_per_batch(self, batch: Dict[str, Any]) -> int:
        """Vision-encoder + connector FLOPs for ``batch`` (read by the speed monitor and
        added to the per-token LM FLOPs). The ViT processes every crop in ``batch["images"]``
        — including padded / dummy crops — so we size it off that tensor's shape."""
        images = batch.get("images")
        if images is None:
            return 0
        b, t, n_patches = (
            int(images.shape[0]),
            int(images.shape[1]),
            int(images.shape[2]),
        )
        n_pooled = int((batch["input_ids"] == self._multimodal.cfg.image_patch_token_id).sum())
        return self._multimodal.image_encoder_flops(b * t, n_patches, n_pooled)


@dataclass
class MultimodalTransformerTrainModuleConfig(TrainModuleConfig):
    """Configuration for :class:`MultimodalTransformerTrainModule`."""

    rank_microbatch_size: int
    max_sequence_length: int
    optim: OptimConfig
    freeze_params: Optional[List[str]] = None
    max_grad_norm: Optional[float] = None
    scheduler: Optional[Scheduler] = None
    compile_model: bool = False
    compile_vision: bool = True
    """Also compile the vision encoder when :data:`compile_model` is enabled."""
    compile_connector: bool = True
    """Also compile the connector with dynamic shapes when :data:`compile_model` is enabled."""
    dp_config: Optional[TransformerDataParallelConfig] = None
    ac_config: Optional[TransformerActivationCheckpointingConfig] = None
    vision_activation_checkpointing: bool = True
    connector_activation_checkpointing: bool = True
    z_loss_multiplier: Optional[float] = None
    autocast_precision: Optional[DType] = None
    label_ignore_index: int = -100
    response_logits_only: bool = False
    loss_group_weights: Optional[Dict[str, float]] = None
    """Opt-in coefficients for separately normalized full-update group CE objectives.

    Positive coefficients must sum to one. Batches require one ``loss_group_names`` entry
    per homogeneous packed sequence (emitted by grouped mixture-loader quotas). Annotation
    weights remain relative within each group; ignored labels never add to its denominator.
    Every group must have positive supervised weight globally on every update. LM z-loss
    shares these weights; router auxiliary normalization is unchanged.
    """
    state_dict_save_opts: Optional[Dict[str, Any]] = None
    state_dict_load_opts: Optional[Dict[str, Any]] = None
    load_key_mapping: Optional[Dict[str, str]] = None

    def build(
        self, model: torch.nn.Module, device: Optional[torch.device] = None
    ) -> "MultimodalTransformerTrainModule":
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
            kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
        if (save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**save_opts)
        if (load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**load_opts)
        return MultimodalTransformerTrainModule(model=model, device=device, **kwargs)


class MultimodalOLMoDDPTrainModule(OLMoDDPTrainModule):
    """
    OLMoDDP data/expert-parallel training for a
    :class:`~olmo_core.nn.vision.MultimodalOLMoDDPModel`.

    The language model is trained through the unchanged :class:`OLMoDDPTrainModule` loop; this
    subclass adds what a multimodal batch needs on top of it:

    * the weighted, response-only loss divisor (the global sum of ``loss_masks``) and the router
      auxiliary-loss divisor (the global count of valid tokens), handed to the model through
      :meth:`_batch_auxiliary_loss_kwargs`;
    * freezing by parameter-name pattern, row-restricted embedding gradients, activation
      checkpointing of the vision tower and connector, and per-source data/optimizer diagnostics;
    * checkpoints that also carry the *frozen* parameters (``frozen_model.*`` entries), and loads
      that accept both those checkpoints and native text checkpoints (``module.<name>.main``
      entries without the ``lm.`` prefix), leaving freshly initialized components untouched;
    * held-out response-loss evaluation for multimodal batches, with the ordinary full-logits
      path kept for text-only downstream evaluators.

    Limitations of this layer: frozen expert weights under expert parallelism cannot be
    checkpointed (saving raises ``NotImplementedError``; the alignment recipe runs the experts
    trainable or with ``ep_config=None``), and the router token mask is not carried, so router
    statistics include padding tokens (see
    :class:`~olmo_core.nn.vision.MultimodalOLMoDDPModel`).
    """

    _FROZEN_MODEL_PARAM_KEY_PREFIX: ClassVar[str] = "frozen_model."
    _FRESH_COMPONENTS: ClassVar[Tuple[str, ...]] = ("connector", "vision")

    def __init__(
        self,
        model: torch.nn.Module,
        *args,
        freeze_params: Optional[List[str]] = None,
        vision_activation_checkpointing: bool = False,
        connector_activation_checkpointing: bool = False,
        response_logits_only: bool = False,
        diagnostics_interval: Optional[int] = None,
        train_embedding_rows: Optional[List[int]] = None,
        source_loss_mass_targets: Optional[Dict[str, float]] = None,
        loss_group_weights: Optional[Dict[str, float]] = None,
        trim_microbatch_image_padding: bool = False,
        **kwargs,
    ):
        from olmo_core.nn.vision import MultimodalOLMoDDPModel

        if not isinstance(model, MultimodalOLMoDDPModel):
            raise TypeError(
                f"{type(self).__name__} requires MultimodalOLMoDDPModel, got {type(model).__name__}"
            )
        unsupported = [
            name for name in ("tp_config", "cp_config", "pp_config") if kwargs.get(name) is not None
        ]
        if unsupported:
            raise OLMoConfigurationError(
                "Multimodal OLMoDDP currently supports data and expert parallelism only; "
                f"unset {', '.join(unsupported)}"
            )
        if model.tbo:
            raise OLMoConfigurationError(
                "Two-batch overlap is not supported for multimodal OLMoDDP"
            )
        if diagnostics_interval is not None and diagnostics_interval <= 0:
            raise OLMoConfigurationError("diagnostics_interval must be positive or None")
        self.trim_microbatch_image_padding = trim_microbatch_image_padding
        if trim_microbatch_image_padding:
            if model.cfg.vision.attention_dropout or model.cfg.vision.residual_dropout:
                raise OLMoConfigurationError(
                    "Image-padding trimming requires zero vision attention and residual dropout"
                )
            log.warning(
                "Image-padding trimming is enabled; vision FLOP estimates still use untrimmed "
                "collator shapes. Compare measured throughput and profiles, not estimated MFU."
            )

        self.freeze_params = freeze_params or []
        frozen = []
        for name, param in model.named_parameters():
            if any(fnmatch(name, pattern) for pattern in self.freeze_params):
                param.requires_grad_(False)
                frozen.append(name)
        if self.freeze_params:
            log.info("Froze %d parameter tensors matching %s", len(frozen), self.freeze_params)
        self.train_embedding_rows = tuple(sorted(train_embedding_rows or []))
        if len(self.train_embedding_rows) != len(set(self.train_embedding_rows)):
            raise OLMoConfigurationError("train_embedding_rows must contain unique IDs")
        if vision_activation_checkpointing:
            model.vision.apply_activation_checkpointing()
            log.info("Applied activation checkpointing to the vision encoder")
        if connector_activation_checkpointing:
            model.connector.apply_activation_checkpointing()
            log.info("Applied activation checkpointing to the vision connector")
        self.response_logits_only = response_logits_only
        self.loss_group_weights = _validate_loss_group_weights(loss_group_weights)
        self.diagnostics_interval = diagnostics_interval
        self.source_loss_mass_targets = dict(source_loss_mass_targets or {})
        if self.source_loss_mass_targets and (
            any(
                not math.isfinite(value) or value <= 0
                for value in self.source_loss_mass_targets.values()
            )
            or abs(sum(self.source_loss_mass_targets.values()) - 1.0) > 1e-6
        ):
            raise OLMoConfigurationError("source_loss_mass_targets must be positive and sum to one")
        self._load_key_map: Dict[str, str] = {}
        self._load_deferred: Dict[str, Any] = {}
        self._load_in_place_keys: Set[str] = set()
        self._load_resync_names: Set[str] = set()
        self._load_expansions: Dict[str, torch.Tensor] = {}
        self._load_metadata: Optional[Metadata] = None
        super().__init__(model, *args, **kwargs)

        # OLMoDDP materializes meta-device weights inside ``super().__init__`` with ``to_empty``,
        # which replaces Parameter objects. Install gradient hooks only on the final materialized
        # parameters so they run before MultiGroupDDP's post-accumulate FP32 reduction hooks.
        materialized_lm = self.multimodal_model.lm
        self._embedding_grad_hook = None
        if self.train_embedding_rows:
            embeddings = materialized_lm.embeddings
            if embeddings is None:
                raise OLMoConfigurationError("train_embedding_rows requires LM embeddings")
            if self.train_embedding_rows[0] < 0 or self.train_embedding_rows[-1] >= int(
                embeddings.weight.shape[0]
            ):
                raise OLMoConfigurationError(
                    "train_embedding_rows contains an ID outside the LM embedding table"
                )
            if not embeddings.weight.requires_grad:
                raise OLMoConfigurationError(
                    "The LM embedding parameter must remain trainable when row masking is enabled"
                )
            if (
                materialized_lm.lm_head is not None
                and materialized_lm.lm_head.w_out.weight is embeddings.weight
            ):
                raise OLMoConfigurationError(
                    "Row-masked image embeddings require untied LM input and output weights"
                )
            self._embedding_grad_hook = embeddings.weight.register_hook(
                lambda grad: _retain_embedding_gradient_rows(grad, self.train_embedding_rows)
            )
            log.info(
                "Restricted LM input-embedding gradients to rows %s", self.train_embedding_rows
            )

    @property
    def multimodal_model(self):
        """The model beneath the data-parallel wrapper."""
        model = self.model_parts[0]
        return getattr(model, "module", model)

    def _require_multimodal_optimizer(self) -> MultimodalOLMoDDPOptimizer:
        optim = self._require_optimizer()
        if not isinstance(optim, MultimodalOLMoDDPOptimizer):
            raise OLMoConfigurationError(
                "This operation needs the partial master synchronization of "
                "MultimodalOLMoDDPOptimizer; configure the train module with "
                "MultimodalOLMoDDPOptimizerConfig"
            )
        return optim

    # -- batches ------------------------------------------------------------------------------

    def _prepare_batch(
        self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        if (
            getattr(self, "trim_microbatch_image_padding", False)
            and batch.get("images") is not None
        ):
            batch = _trim_microbatch_image_padding(batch)
        input_ids, labels, model_kwargs = super()._prepare_batch(batch, labels)
        # Collator metadata that only the train module reads.
        for key in (
            "image_crop_counts",
            "pooled_token_counts",
            "loss_group_names",
            "pack_source_names",
            "metadata",
            "instance_mask",
        ):
            model_kwargs.pop(key, None)
        # The router token mask only feeds the divisor and data metrics; the LM ignores it.
        model_kwargs.pop("router_token_mask", None)
        # Response-only logits are specific to multimodal batches carrying loss weights.
        # Ordinary downstream LM evaluators do not provide ``loss_masks`` and require full
        # sequence logits, even when the training module uses response-only logits for Stage 1.
        if self.response_logits_only and "loss_masks" in batch:
            model_kwargs["response_logits_only"] = True
        return input_ids, labels, model_kwargs

    def _batch_auxiliary_loss_kwargs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Divisors for one optimizer batch, following the OLMoDDP loss-scaling convention (global
        sum, divided by the DP world size, so averaged gradients have the scale of one global
        batch): the router auxiliary losses use the valid-token count (``router_token_mask``, or
        every token), and the weighted CE uses the sum of the positive ``loss_masks``.
        """
        token_mask = batch.get("router_token_mask")
        if token_mask is not None and token_mask.shape != batch["input_ids"].shape:
            raise OLMoConfigurationError(
                "router_token_mask must match input_ids: "
                f"got {tuple(token_mask.shape)} and {tuple(batch['input_ids'].shape)}"
            )
        if token_mask is not None:
            valid_tokens = token_mask.sum(dtype=torch.long)
        else:
            valid_tokens = torch.tensor(batch["input_ids"].numel(), dtype=torch.long)
        kwargs = {"router_loss_div_factor": self._global_batch_divisor(valid_tokens)}
        if (loss_masks := batch.get("loss_masks")) is not None:
            loss_masks = loss_masks.float()
            kwargs["loss_weight_div_factor"] = self._global_batch_divisor(
                (loss_masks * (loss_masks > 0)).sum()
            )
        return kwargs

    def _global_batch_divisor(self, local_value: torch.Tensor) -> torch.Tensor:
        value = move_to_device(local_value, self.device)
        if is_distributed():
            value = value.clone()
            dist.all_reduce(value, group=self.dp_process_group)
            return value.clamp_min(1) / get_world_size(self.dp_process_group)
        return value.clamp_min(1)

    def _record_data_metrics(self, batch: Dict[str, Any]) -> None:
        """Record packing and supervision density without synchronizing CUDA."""
        token_mask = batch.get("router_token_mask")
        if token_mask is None:
            return
        if getattr(self, "source_loss_mass_targets", None) and (
            getattr(self, "_trainer", None) is None or self._diagnostics_enabled_for_step()
        ):
            self._record_source_data_metrics(batch, token_mask)
        self.record_metric(
            "packing fill", token_mask.float().mean(), ReduceType.mean, namespace="data"
        )

        if (loss_masks := batch.get("loss_masks")) is not None:
            self.record_metric(
                "response token density",
                ((loss_masks > 0) & token_mask).float().mean(),
                ReduceType.mean,
                namespace="data",
            )
        if (token_type_ids := batch.get("token_type_ids")) is not None:
            self.record_metric(
                "image token density",
                ((token_type_ids != 0) & token_mask).float().mean(),
                ReduceType.mean,
                namespace="data",
            )
        if (example_ids := batch.get("example_ids")) is not None:
            self.record_metric(
                "examples per sequence",
                (example_ids.amax(dim=1) + 1).float().mean(),
                ReduceType.mean,
                namespace="data",
            )

        if (crop_counts := batch.get("image_crop_counts")) is not None:
            self.record_metric(
                "real crops per sequence",
                crop_counts.float().mean(),
                ReduceType.mean,
                namespace="data",
            )
            images = batch.get("images")
            if images is not None:
                padded_crops = int(images.shape[1])
                utilization = crop_counts.float().sum() / max(
                    int(crop_counts.numel()) * padded_crops, 1
                )
                self.record_metric(
                    "padded crops per sequence",
                    torch.tensor(float(padded_crops), device=crop_counts.device),
                    ReduceType.mean,
                    namespace="data",
                )
                self.record_metric(
                    "crop utilization", utilization, ReduceType.mean, namespace="data"
                )
        if (pooled_counts := batch.get("pooled_token_counts")) is not None:
            self.record_metric(
                "pooled image tokens per sequence",
                pooled_counts.float().mean(),
                ReduceType.mean,
                namespace="data",
            )

    def _record_source_data_metrics(self, batch: Dict[str, Any], token_mask: torch.Tensor) -> None:
        """Record globally summed examples, tokens, and weighted loss mass by source.

        The configured mixture targets are expressed in *global supervised-loss mass*.
        Computing a ratio on each DP rank and averaging those ratios is not equivalent to
        dividing globally summed source weights by the globally summed total, especially when
        dense native-text rows and short response-only visual rows land on different ranks.
        This method therefore performs one stacked DP reduction at the diagnostics cadence and
        derives every reported share from those global sums.
        """
        packed_sources = batch.get("pack_source_names")
        example_ids = batch.get("example_ids")
        loss_masks = batch.get("loss_masks")
        labels = batch.get("labels")
        if packed_sources is None or example_ids is None or loss_masks is None or labels is None:
            raise OLMoConfigurationError(
                "Per-source telemetry requires packed source names, example IDs, labels, "
                "and loss masks"
            )
        if len(packed_sources) != int(example_ids.shape[0]):
            raise OLMoConfigurationError("Packed source metadata does not match the rank batch")
        metric_names = (
            "examples",
            "tokens",
            "positive_tokens",
            "loss_weight",
            "active_loss_weight",
        )
        stats: Dict[str, Dict[str, torch.Tensor]] = {
            source_name: {name: loss_masks.new_zeros(()) for name in metric_names}
            for source_name in self.source_loss_mass_targets
        }
        label_ignore_index = getattr(self, "label_ignore_index", -100)
        for row, source_names in enumerate(packed_sources):
            for example_id, source_name in enumerate(source_names):
                if source_name not in self.source_loss_mass_targets:
                    raise OLMoConfigurationError(
                        f"Observed unconfigured source {source_name!r} in packed telemetry"
                    )
                positions = (example_ids[row] == example_id) & token_mask[row]
                observed_stats = stats[source_name]
                active_positions = positions & (labels[row] != label_ignore_index)
                observed_stats["examples"] += 1
                observed_stats["tokens"] += positions.sum()
                observed_stats["positive_tokens"] += (
                    (loss_masks[row] > 0) & active_positions
                ).sum()
                observed_stats["loss_weight"] += (loss_masks[row] * positions).sum()
                observed_stats["active_loss_weight"] += (loss_masks[row] * active_positions).sum()

        source_names = tuple(self.source_loss_mass_targets)
        global_stats = torch.stack(
            [stats[source_name][name] for source_name in source_names for name in metric_names]
        )
        if is_distributed():
            global_stats = move_to_device(global_stats, self.device)
            dist.all_reduce(global_stats, group=self.dp_process_group)
        global_stats = global_stats.reshape(len(source_names), len(metric_names))
        stats = {
            source_name: {
                name: global_stats[source_index, metric_index]
                for metric_index, name in enumerate(metric_names)
            }
            for source_index, source_name in enumerate(source_names)
        }
        total_loss_weight = sum(
            (source_stats["loss_weight"] for source_stats in stats.values()),
            start=global_stats.new_zeros(()),
        ).clamp_min(1.0)
        for source_name, target in self.source_loss_mass_targets.items():
            metric_stats = stats[source_name]
            for metric_name, value in metric_stats.items():
                self.record_metric(
                    f"source/{source_name}/{metric_name}",
                    value,
                    # Every DP rank holds the identical already-summed tensor.
                    ReduceType.mean,
                    namespace="data",
                )
            realized_share = metric_stats["loss_weight"] / total_loss_weight
            self.record_metric(
                f"source/{source_name}/loss_mass_share",
                realized_share,
                ReduceType.mean,
                namespace="data",
            )
            # With explicit group quotas, source sampling targets are relative to their group,
            # so they cannot be compared directly with global source shares.
            if "loss_group_names" not in batch:
                self.record_metric(
                    f"source/{source_name}/loss_mass_target_abs_error",
                    (realized_share - target).abs(),
                    ReduceType.mean,
                    namespace="data",
                )

    def _diagnostics_enabled_for_step(self) -> bool:
        return bool(
            self.diagnostics_interval is not None
            and self._trainer is not None
            and self.trainer.global_step % self.diagnostics_interval == 0
        )

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        original_batch = batch
        if self.loss_group_weights:
            batch, global_mass = _normalize_loss_groups(
                batch,
                self.loss_group_weights,
                label_ignore_index=self.label_ignore_index,
                device=self.device,
                dp_process_group=self.dp_process_group,
            )
            if not dry_run:
                for index, group in enumerate(self.loss_group_weights):
                    self.record_metric(
                        f"group/{group}/active_loss_weight",
                        global_mass[index],
                        ReduceType.mean,
                        namespace="data",
                    )
                    self.record_metric(
                        f"group/{group}/objective_weight",
                        self.loss_group_weights[group],
                        ReduceType.mean,
                        namespace="data",
                    )
        if not dry_run:
            self._record_data_metrics(original_batch)
        collect_diagnostics = not dry_run and self._diagnostics_enabled_for_step()
        if collect_diagnostics:
            self.multimodal_model.set_input_diagnostics(True)
        try:
            result = super().train_batch(batch, dry_run=dry_run)
        except BaseException:
            if collect_diagnostics:
                self.multimodal_model.set_input_diagnostics(False)
            raise
        if collect_diagnostics:
            diagnostics = self.multimodal_model.pop_input_diagnostics(
                reduce_across_process_group=is_distributed(),
                process_group=self.dp_process_group,
            )
            for name, value in diagnostics.items():
                self.record_metric(name, value, reduce_type=None, namespace="multimodal")
        return result

    def optim_step(self):
        optim = self._require_optimizer()
        collect_diagnostics = self._diagnostics_enabled_for_step() and isinstance(
            optim, MultimodalOLMoDDPOptimizer
        )
        if self._diagnostics_enabled_for_step() and not collect_diagnostics:
            warn_once(
                log,
                "Component gradient-norm diagnostics need MultimodalOLMoDDPOptimizer; skipping",
            )
        if collect_diagnostics:
            assert isinstance(optim, MultimodalOLMoDDPOptimizer)
            component_patterns = {
                "vision": ("vision.*", "*vision.*"),
                "connector": ("connector.*", "*connector.*"),
                "input embeddings": ("lm.embeddings.weight", "*lm.embeddings.weight"),
                "LM output head": ("lm.lm_head.w_out.*", "*lm.lm_head.w_out.*"),
                "LM attention": ("lm.blocks.*.attention.*", "*lm.blocks.*.attention.*"),
                "LM routed experts": (
                    "lm.blocks.*.routed_experts.*",
                    "*lm.blocks.*.routed_experts.*",
                ),
                "LM shared experts": (
                    "lm.blocks.*.shared_experts.*",
                    "*lm.blocks.*.shared_experts.*",
                ),
                "LM routers": (
                    "lm.blocks.*.routed_experts_router.*",
                    "*lm.blocks.*.routed_experts_router.*",
                ),
                "LM normalization": ("lm.*norm*", "*lm.*norm*"),
            }
            trainable_names = {
                name
                for group in optim.param_groups
                for name, param in group["named_params"].items()
                if param.requires_grad
            }
            optim.set_component_grad_norm_patterns(
                _matched_component_grad_norm_patterns(component_patterns, trainable_names)
            )
        try:
            super().optim_step()
            if collect_diagnostics:
                assert isinstance(optim, MultimodalOLMoDDPOptimizer)
                for component, value in optim.latest_component_grad_norms.items():
                    self.record_metric(
                        f"{component} grad norm", value, reduce_type=None, namespace="optim"
                    )
                for group_name, value in optim.latest_clip_group_grad_norms.items():
                    component = (
                        "language model"
                        if group_name == optim.DEFAULT_CLIP_GROUP_NAME
                        else group_name
                    )
                    self.record_metric(
                        f"{component} clip group grad norm",
                        value,
                        reduce_type=None,
                        namespace="optim",
                    )
                for group_name, value in optim.latest_clip_group_coefficients.items():
                    component = (
                        "language model"
                        if group_name == optim.DEFAULT_CLIP_GROUP_NAME
                        else group_name
                    )
                    self.record_metric(
                        f"{component} clip coefficient",
                        value,
                        reduce_type=None,
                        namespace="optim",
                    )
        finally:
            if collect_diagnostics:
                assert isinstance(optim, MultimodalOLMoDDPOptimizer)
                optim.set_component_grad_norm_patterns(None)

    def extra_flops_per_batch(self, batch: Dict[str, Any]) -> int:
        """Estimate vision/connector FLOPs from untrimmed collator tensor shapes.

        This estimate does not account for optional microbatch image-padding trimming
        or subsequent cross-rank crop padding; it is not a measurement of optimized work.
        """
        images = batch.get("images")
        if images is None:
            return 0
        batch_size, crops, patches = (int(value) for value in images.shape[:3])
        pooled = int((batch["input_ids"] == self.multimodal_model.cfg.image_patch_token_id).sum())
        return self.multimodal_model.image_encoder_flops(batch_size * crops, patches, pooled)

    # -- evaluation ---------------------------------------------------------------------------

    @contextlib.contextmanager
    def _eval_batch_context(self):
        # Downstream text batches can have rank-local sequence shapes that Inductor cannot
        # lower; the eager path is correct. Keep grad mode enabled: compiled no-grad attention
        # has produced incorrect outputs on some backends, and eval_batch detaches the outputs.
        with torch.enable_grad(), torch.compiler.set_stance("force_eager"):
            yield

    def eval_batch(
        self,
        batch: Dict[str, Any],
        labels: Optional[torch.Tensor] = None,
        *,
        return_response_logits: bool = False,
    ) -> LMOutputWithLoss:
        """Evaluate multimodal response loss or delegate ordinary LM evaluation.

        Multimodal batches carry ``loss_masks`` and return the *summed* weighted response-token
        loss (the evaluator divides by the weights it sums itself) without materializing
        full-sequence logits. Text-only downstream batches do not carry ``loss_masks`` and take
        the standard OLMoDDP path so evaluators receive logits.

        :param return_response_logits: Retain logits at supervised response positions for a
            multimodal batch. This requires ``response_logits_only=True`` on the train module so
            an evaluator cannot accidentally materialize full-sequence vocabulary logits.
        """
        if "loss_masks" not in batch:
            if return_response_logits:
                raise ValueError(
                    "return_response_logits is only valid for multimodal loss-mask batches"
                )
            output = super().eval_batch(batch, labels=labels)
            assert isinstance(output, LMOutputWithLoss), "Expected LMOutputWithLoss"
            return output._replace(
                logits=output.logits.detach() if output.logits is not None else None,
                loss=output.loss.detach(),
                ce_loss=output.ce_loss.detach(),
                z_loss=output.z_loss.detach() if output.z_loss is not None else None,
            )

        if self.cp_enabled or self.tp_enabled or self.pp_enabled:
            raise RuntimeError(
                f"{self.__class__.__name__}.eval_batch() only supports the DP/EP topology"
            )
        if return_response_logits and not self.response_logits_only:
            raise RuntimeError(
                "return_response_logits requires response_logits_only=True to avoid "
                "materializing full-sequence vocabulary logits"
            )

        # EvaluatorCallback derives ordinary LM labels from input_ids, but multimodal batches
        # carry branch-aware, already-shifted labels. Prefer those and leave the original batch
        # intact so the evaluator can use its loss weights after the forward pass.
        model_batch = dict(batch)
        if (batch_labels := model_batch.pop("labels", None)) is not None:
            labels = batch_labels
        input_ids, labels, model_kwargs = self._prepare_batch(model_batch, labels)
        if labels is None:
            raise OLMoConfigurationError("Multimodal evaluation batches require labels")

        for model_part in self.model_parts:
            model_part.eval()

        try:
            with torch.enable_grad():
                output = self.model_forward_no_pipeline(
                    input_ids,
                    labels=labels,
                    ignore_index=self.label_ignore_index,
                    loss_reduction="sum",
                    return_logits=return_response_logits,
                    **model_kwargs,
                )
                assert isinstance(output, LMOutputWithLoss), "Expected LMOutputWithLoss"
                return output._replace(
                    logits=(
                        output.logits.detach()
                        if return_response_logits and output.logits is not None
                        else None
                    ),
                    loss=output.loss.detach(),
                    ce_loss=output.ce_loss.detach(),
                    z_loss=output.z_loss.detach() if output.z_loss is not None else None,
                )
        finally:
            # Router metrics from held-out data must not leak into the next training window.
            for model_part in self.model_parts:
                model_part.reset_auxiliary_metrics()

    # -- pretrained component loading ---------------------------------------------------------

    @torch.no_grad()
    def load_vision_state_dict(self, vision_state: Dict[str, torch.Tensor]) -> None:
        """Strictly load vision weights and synchronize trainable optimizer masters.

        OLMoDDP creates FP32 optimizer master parameters when the train module is built. Any
        model-only load performed afterwards must update those masters before the first optimizer
        step, otherwise that step copies the stale initialization back into the vision tower.

        :param vision_state: State dictionary in the native vision encoder format.
        """
        model = self.multimodal_model
        model.vision.load_state_dict(vision_state, strict=True)

        if self.optim is None:
            return

        vision_param_names = self._trainable_vision_param_names()
        if not vision_param_names:
            return

        optim = self._require_multimodal_optimizer()
        optim._copy_model_params_to_main_params(vision_param_names)
        optim._check_model_param_main_param_the_same(vision_param_names)

    def assert_vision_optimizer_state_synced(self) -> None:
        """Check every trainable vision tensor against its optimizer-owned FP32 master."""
        if self.optim is None:
            raise RuntimeError("Cannot check optimizer state on an eval-only train module")
        self._require_multimodal_optimizer()._check_model_param_main_param_the_same(
            self._trainable_vision_param_names()
        )

    def _trainable_vision_param_names(self) -> Set[str]:
        """Resolve every trainable vision parameter to its optimizer-owned name."""
        optim = self._require_optimizer()
        trainable_vision_params = {
            id(param) for param in self.multimodal_model.vision.parameters() if param.requires_grad
        }
        vision_param_names = {
            name
            for param_group in optim.param_groups
            for name, param in param_group["named_params"].items()
            if id(param) in trainable_vision_params
        }
        if len(vision_param_names) != len(trainable_vision_params):
            raise RuntimeError(
                "Could not map every trainable vision parameter to its optimizer master: "
                f"found {len(vision_param_names)} of {len(trainable_vision_params)}"
            )
        return vision_param_names

    @torch.no_grad()
    def reset_image_token_rows(
        self, token_ids: List[int], *, seed: int, reset_output_rows: bool = True
    ) -> None:
        """Initialize newly assigned image-token rows and update optimizer main state.

        :param token_ids: Input-embedding row IDs to initialize.
        :param seed: Initialization seed.
        :param reset_output_rows: Also initialize the same untied LM-head rows. Keep false when
            those rows already participated in the pretrained model's output softmax.
        """
        if not token_ids or len(set(token_ids)) != len(token_ids):
            raise ValueError("token_ids must be a non-empty list of unique IDs")

        model = self.multimodal_model
        lm = model.lm
        if lm.embeddings is None or lm.lm_head is None:
            raise RuntimeError("Image-token initialization requires LM embeddings and an LM head")
        if min(token_ids) < 0 or max(token_ids) >= lm.vocab_size:
            raise ValueError(
                f"Image token IDs must be within [0, {lm.vocab_size}), got {token_ids}"
            )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        row_count = len(token_ids)
        embedding_rows = torch.nn.Embedding(
            row_count, lm.d_model, device=self.device, dtype=lm.embeddings.weight.dtype
        )
        lm.init_method.init_embeddings(
            embedding_rows,
            d_model=lm.d_model,
            embed_scale=lm.embed_scale,
            std=lm.embedding_init_std if lm.embedding_init_std is not None else lm.init_std,
            generator=generator,
        )
        row_index = torch.tensor(token_ids, device=self.device, dtype=torch.long)
        lm.embeddings.weight.index_copy_(0, row_index, embedding_rows.weight)

        if reset_output_rows and lm.lm_head.w_out.weight is not lm.embeddings.weight:
            output_rows = torch.nn.Linear(
                lm.d_model,
                row_count,
                bias=False,
                device=self.device,
                dtype=lm.lm_head.w_out.weight.dtype,
            )
            lm.init_method.init_final_w_out(
                output_rows, d_model=lm.d_model, std=lm.init_std, generator=generator
            )
            lm.lm_head.w_out.weight.index_copy_(0, row_index, output_rows.weight)

        optim = self._require_multimodal_optimizer()
        reset_params = {
            name
            for group in optim.param_groups
            for name, param in group["named_params"].items()
            if param is lm.embeddings.weight
            or (reset_output_rows and param is lm.lm_head.w_out.weight)
        }
        optim._copy_model_param_rows_to_main_params(reset_params, token_ids)

    # -- checkpoints --------------------------------------------------------------------------

    def _frozen_model_param_state_dict(self) -> Dict[str, torch.Tensor]:
        """Map every frozen parameter to its stable ``frozen_model.<name>`` checkpoint key."""
        frozen_state: Dict[str, torch.Tensor] = {}
        for model_part in self.model_parts:
            for name, param in model_part.named_parameters():
                if param.requires_grad:
                    continue
                if self.ep_enabled and ".routed_experts." in name:
                    raise NotImplementedError(
                        "Checkpointing frozen expert-parallel expert weights is not supported; "
                        "train the experts or disable expert parallelism"
                    )
                key = self._FROZEN_MODEL_PARAM_KEY_PREFIX + self._strip_wrapper_prefixes(name)
                if key in frozen_state:
                    raise RuntimeError(
                        f"Duplicate frozen parameter checkpoint key '{key}'; parameter names "
                        "collide across model parts."
                    )
                frozen_state[key] = param.data
        return frozen_state

    def save_state_dict_direct(
        self,
        dir: PathOrStr,
        *,
        process_group: Optional[dist.ProcessGroup] = None,
        save_overwrite: bool = False,
        thread_count: Optional[int] = None,
        throttle_uploads: bool = False,
    ):
        """Save optimizer state, persistent model buffers, and frozen model parameters."""
        optim = self._require_optimizer()
        # Exporting EP state may release live shards; reload them after the synchronous save.
        state_dict = optim.state_dict()

        save_dict = dict(state_dict)
        for key, buffer in self._persistent_model_buffer_state_dict().items():
            assert key not in save_dict, f"Buffer key '{key}' collides with an optimizer state key"
            save_dict[key] = buffer
        for key, param in self._frozen_model_param_state_dict().items():
            assert key not in save_dict, f"Frozen key '{key}' collides with checkpoint state"
            save_dict[key] = param

        dir = _prepare_env_for_save(dir, process_group=process_group, save_overwrite=save_overwrite)
        dist_cp.state_dict_saver.save(
            save_dict,
            storage_writer=RemoteFileSystemWriter(
                dir,
                thread_count=thread_count,
                process_group=process_group,
                throttle_uploads=throttle_uploads,
            ),
            process_group=process_group,
            planner=FlatSavePlanner(dedup_save_to_lowest_rank=True),
        )

        optim.load_state_dict(state_dict, reset_optimizer_moments_on_load=False)
        torch.cuda.empty_cache()

    def load_state_dict_direct(self, dir: PathOrStr, **kwargs):
        """
        Load a checkpoint; see :meth:`OLMoDDPTrainModule.load_state_dict_direct`. The checkpoint
        metadata is kept for the duration of the load so that the key adapters can expand shared
        Q/K norm gains stored under native text keys.
        """
        from olmo_core.io import normalize_path

        reader = RemoteFileSystemReader(
            normalize_path(dir),
            thread_count=kwargs.get("thread_count"),
            pre_download=kwargs.get("pre_download", False),
            work_dir=kwargs.get("work_dir"),
        )
        self._load_metadata = reader.read_metadata()
        try:
            super().load_state_dict_direct(dir, **kwargs)
        finally:
            self._load_metadata = None

    @staticmethod
    def _strip_lm_prefix(name: str) -> Optional[str]:
        """``lm.blocks.0.w`` -> ``blocks.0.w`` (the name in a native text checkpoint)."""
        return name[len("lm.") :] if name.startswith("lm.") else None

    def _resolve_model_checkpoint_key(
        self, param_name: str, checkpoint_keys: Collection[str]
    ) -> Optional[str]:
        """
        Resolve a parameter to its checkpoint key: the standard candidates, then the native text
        checkpoint name without the ``lm.`` prefix, then the ``frozen_model.<name>`` entry of a
        multimodal checkpoint (also used by the eval-only load, which maps every parameter).
        """
        checkpoint_key = super()._resolve_model_checkpoint_key(param_name, checkpoint_keys)
        if checkpoint_key is not None:
            return checkpoint_key
        stripped = self._strip_wrapper_prefixes(param_name)
        text_name = self._strip_lm_prefix(stripped)
        if text_name is not None:
            checkpoint_key = super()._resolve_model_checkpoint_key(text_name, checkpoint_keys)
            if checkpoint_key is not None:
                return checkpoint_key
        stable_key = self._FROZEN_MODEL_PARAM_KEY_PREFIX + stripped
        return stable_key if stable_key in checkpoint_keys else None

    def _resolve_optimizer_checkpoint_key(
        self, state_key: str, checkpoint_keys: Collection[str]
    ) -> Optional[str]:
        """Map an optimizer state key onto a native text checkpoint that has no ``lm.`` prefix."""
        if state_key in checkpoint_keys:
            return state_key
        for prefix in ("model.module.lm.", "model.lm.", "module.lm.", "lm."):
            if state_key.startswith(prefix):
                candidate = prefix.replace("lm.", "", 1) + state_key[len(prefix) :]
                if candidate in checkpoint_keys:
                    return candidate
        return None

    def _allow_missing_optimizer_checkpoint_key(self, state_key: str) -> bool:
        """Freshly initialized components keep their optimizer state when a checkpoint lacks it."""
        return any(
            f".{component}." in state_key or state_key.startswith(f"{component}.")
            for component in self._FRESH_COMPONENTS
        )

    def _frozen_checkpoint_model_param_state_dict_for_load(
        self, checkpoint_keys: Collection[str]
    ) -> Dict[str, torch.nn.Parameter]:
        """Map the checkpoint keys that load straight into model parameters.

        A parameter that was frozen when a multimodal checkpoint was written is stored as
        ``frozen_model.<name>`` and is read from there whether or not it is trainable now (the
        vision tower between alignment phases, for example). A parameter that is frozen *now*
        but was trained by a native text run is stored only as a flattened FP32 optimizer main
        parameter, so it is read from that ``.main`` entry instead.
        """
        state: Dict[str, torch.nn.Parameter] = {}
        for model_part in self.model_parts:
            for name, param in model_part.named_parameters():
                stable_key = self._FROZEN_MODEL_PARAM_KEY_PREFIX + self._strip_wrapper_prefixes(
                    name
                )
                key: Optional[str] = stable_key if stable_key in checkpoint_keys else None
                if key is None and not param.requires_grad:
                    checkpoint_key = self._resolve_model_checkpoint_key(name, checkpoint_keys)
                    if checkpoint_key is not None and checkpoint_key.endswith(".main"):
                        key = checkpoint_key
                if key is None:
                    continue
                if key in state:
                    raise RuntimeError(f"Multiple frozen parameters map to checkpoint key '{key}'")
                state[key] = param
        return state

    def _frozen_checkpoint_param_state_dict_for_load(
        self, checkpoint_keys: Collection[str]
    ) -> Dict[str, torch.Tensor]:
        """Map checkpointed frozen parameters onto the tensors that receive them in place."""
        return {
            key: param.data.view(-1) if key.endswith(".main") else param
            for key, param in self._frozen_checkpoint_model_param_state_dict_for_load(
                checkpoint_keys
            ).items()
        }

    def _optimizer_state_dict_for_load(
        self, state_dict: Dict[str, Any], checkpoint_keys: Set[str]
    ) -> Dict[str, Any]:
        """
        Rewrite the optimizer state dict for the checkpoint being read: alias native text keys,
        set aside entries the checkpoint cannot have (fresh components, parameters that were
        frozen when it was written) and add the frozen parameters and ``lm.``-prefixed buffers
        so everything loads in one pass. :meth:`_optimizer_state_dict_after_load` undoes it.
        """
        self._load_key_map = {}
        self._load_deferred = {}
        self._load_in_place_keys = set()
        self._load_resync_names = set()
        self._load_expansions = {}

        frozen_in_checkpoint = {
            key[len(self._FROZEN_MODEL_PARAM_KEY_PREFIX) :]
            for key in checkpoint_keys
            if key.startswith(self._FROZEN_MODEL_PARAM_KEY_PREFIX)
        }
        # Only a masters-only load (a model-only phase handoff or a reset of the optimizer
        # moments) may leave freshly initialized components without checkpoint state. A full
        # resume loads every optimizer moment, so a missing connector/vision entry is an error.
        partial_load = all(key.endswith(".main") for key in state_dict if not key.startswith("__"))
        out: Dict[str, Any] = {}
        for key, value in state_dict.items():
            checkpoint_key = self._resolve_optimizer_checkpoint_key(key, checkpoint_keys)
            if checkpoint_key is not None:
                if checkpoint_key in out:
                    raise RuntimeError(
                        f"Multiple optimizer state keys map to checkpoint key '{checkpoint_key}'"
                    )
                out[checkpoint_key] = value
                if checkpoint_key != key:
                    self._load_key_map[checkpoint_key] = key
                continue
            param_name = self._strip_wrapper_prefixes(key.rsplit(".", 1)[0])
            if param_name in frozen_in_checkpoint:
                # Frozen when saved, trainable now: the weights arrive through the
                # ``frozen_model.*`` entry, loaded in place, and the masters are
                # resynchronized from them afterwards.
                self._load_deferred[key] = value
                self._load_resync_names.add(key.rsplit(".", 1)[0])
                continue
            if partial_load and self._allow_missing_optimizer_checkpoint_key(key):
                self._load_deferred[key] = value
                continue
            out[key] = value  # a genuinely missing key fails in the loader, as for text models

        if self.expand_shared_qk_norm_on_load and self._load_key_map:
            # The shared load path prepared Q/K gain expansions by the *current* keys, which a
            # native text checkpoint does not contain; prepare the aliased entries here (by
            # checkpoint key) and finish them in :meth:`_optimizer_state_dict_after_load`.
            self._load_expansions = self._prepare_aliased_qk_expansions(out)

        for key, tensor in self._frozen_checkpoint_param_state_dict_for_load(
            checkpoint_keys
        ).items():
            if key in out:
                raise RuntimeError(f"Frozen parameter key '{key}' collides with optimizer state")
            out[key] = tensor
            self._load_in_place_keys.add(key)

        # Persistent buffers of the LM saved by a native text run lack the ``lm.`` prefix.
        prefix = self._MODEL_BUFFER_KEY_PREFIX
        for key, buffer in self._persistent_model_buffer_state_dict().items():
            if key in checkpoint_keys or key in out:
                continue
            text_name = self._strip_lm_prefix(key[len(prefix) :])
            if text_name is not None and (alias := prefix + text_name) in checkpoint_keys:
                out[alias] = buffer
                self._load_in_place_keys.add(alias)
        return out

    def _prepare_aliased_qk_expansions(self, state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        from olmo_core.distributed.checkpoint.utils import prepare_qk_expansion

        if self._load_metadata is None:
            raise RuntimeError(
                "Q/K norm expansion needs the checkpoint metadata; load through "
                "load_state_dict_direct"
            )
        optim = self._require_optimizer()
        gain_shapes: Dict[str, Tuple[int, ...]] = {}
        for group in optim.param_groups:
            for name, param in group["named_params"].items():
                if not name.endswith((".q_norm.weight", ".k_norm.weight")) or param.ndim != 2:
                    continue
                checkpoint_key = self._load_key_map_inverse().get(f"{name}.main")
                if checkpoint_key is not None:
                    gain_shapes[checkpoint_key.rpartition(".")[0]] = tuple(param.shape)
        aliased = {key: state[key] for key in self._load_key_map}
        expansions = prepare_qk_expansion(aliased, self._load_metadata, gain_shapes)
        state.update(aliased)
        return expansions

    def _load_key_map_inverse(self) -> Dict[str, str]:
        return {current: checkpoint for checkpoint, current in self._load_key_map.items()}

    def _optimizer_state_dict_after_load(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self._load_expansions:
            from olmo_core.distributed.checkpoint.utils import finish_qk_expansion

            finish_qk_expansion(state_dict, self._load_expansions)
        out: Dict[str, Any] = {}
        for key, value in state_dict.items():
            if key in self._load_in_place_keys:
                continue  # loaded straight into the model tensor
            out[self._load_key_map.get(key, key)] = value
        out.update(self._load_deferred)
        if self._load_resync_names:
            optim = self._require_multimodal_optimizer()
            optim._copy_model_params_to_main_params(set(self._load_resync_names))
            for name in self._load_resync_names:
                out[f"{name}.main"] = optim.states[f"{name}.main"]
        self._load_key_map = {}
        self._load_deferred = {}
        self._load_in_place_keys = set()
        self._load_resync_names = set()
        self._load_expansions = {}
        return out


@dataclass
class MultimodalOLMoDDPTrainModuleConfig(OLMoDDPTrainModuleConfig):
    """Configuration for :class:`MultimodalOLMoDDPTrainModule`."""

    freeze_params: Optional[List[str]] = None
    vision_activation_checkpointing: bool = False
    connector_activation_checkpointing: bool = False
    response_logits_only: bool = False
    diagnostics_interval: Optional[int] = None
    train_embedding_rows: Optional[List[int]] = None
    """Embedding rows allowed to receive gradients; all other rows are held fixed."""
    source_loss_mass_targets: Optional[Dict[str, float]] = None
    """Optional expected source loss-mass shares that enable online delivery telemetry."""
    loss_group_weights: Optional[Dict[str, float]] = None
    """Opt-in separately normalized group CE weights; see
    :class:`MultimodalTransformerTrainModuleConfig` for the batch contract.
    """
    trim_microbatch_image_padding: bool = False
    """Opt in to removing trailing image-crop and pooled-row padding per microbatch.

    When images are supplied, requires collator ``image_crop_counts`` / ``pooled_token_counts``
    metadata and zero vision dropout. Retains at least one dummy crop and pooled row, existing
    vision collectives, and all LM token slots. FLOP estimates retain untrimmed batch shapes.
    """

    def _build_train_module(self, **kwargs) -> MultimodalOLMoDDPTrainModule:
        return MultimodalOLMoDDPTrainModule(**kwargs)
