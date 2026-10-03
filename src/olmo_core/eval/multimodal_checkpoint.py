"""Native checkpoint construction, load validation, and response-loss utilities."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from olmo_core.distributed.checkpoint import get_checkpoint_metadata
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath
from olmo_core.nn.transformer import OLMoDDPModelConfig
from olmo_core.nn.vision import MultimodalLMConfig
from olmo_core.optim import OLMoDDPOptimizerConfig
from olmo_core.train.train_module import (
    OLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)
from olmo_core.train.train_module.transformer.multimodal_train_module import (
    MultimodalOLMoDDPTrainModuleConfig,
)

WINDOWS: tuple[tuple[str, int | None], ...] = (
    ("all", None),
    ("first_1", 1),
    ("first_8", 8),
    ("first_32", 32),
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def response_ce_by_example(
    batch: Mapping[str, torch.Tensor], logits: torch.Tensor
) -> list[dict[str, Any]]:
    """Compute response-weighted cross entropy for each example and prefix window."""
    labels = batch["labels"]
    loss_masks = batch["loss_masks"]
    response_mask = loss_masks > 0
    if bool(torch.any(labels.masked_select(response_mask) == -100)):
        raise ValueError("A supervised response position has an ignored label")
    counts = response_mask.sum(dim=1)
    if bool(torch.any(counts <= 0)):
        raise ValueError("Every evaluated recipient must contain supervised response tokens")
    response_labels = labels.masked_select(response_mask)
    response_weights = loss_masks.masked_select(response_mask).float()
    if logits.ndim == 3:
        response_logits = logits.reshape(-1, logits.shape[-1])[response_mask.reshape(-1)]
    elif logits.ndim == 2 and logits.shape[0] == response_labels.numel():
        response_logits = logits
    else:
        raise ValueError(
            "Expected response-only logits with one row per supervised token, got "
            f"{tuple(logits.shape)} for {response_labels.numel()} tokens"
        )
    token_ce = F.cross_entropy(response_logits.float(), response_labels, reduction="none")
    records: list[dict[str, Any]] = []
    offset = 0
    for count_tensor in counts:
        count = int(count_tensor.item())
        ce = token_ce[offset : offset + count]
        weights = response_weights[offset : offset + count]
        windows: dict[str, float] = {}
        for name, limit in WINDOWS:
            width = count if limit is None else min(limit, count)
            selected_weights = weights[:width]
            value = (ce[:width] * selected_weights).sum() / selected_weights.sum()
            windows[name] = float(value.detach().cpu().item())
        records.append({"response_tokens": count, "windows": windows})
        offset += count
    if offset != response_labels.numel():
        raise RuntimeError("Response-logit rows were not partitioned exactly by example")
    return records


def native_checkpoint_load_coverage(train_module: Any, state_dir: Path) -> dict[str, Any]:
    """Prove that the eval-only native load maps every model parameter and buffer."""
    metadata = get_checkpoint_metadata(state_dir)
    checkpoint_keys = set(metadata.state_dict_metadata)
    required_methods = {
        "eval state": "_get_model_state_dict_for_eval_load",
        "checkpoint-key resolver": "_resolve_model_checkpoint_key",
        "frozen parameters": "_frozen_checkpoint_model_param_state_dict_for_load",
        "frozen tensors": "_frozen_checkpoint_param_state_dict_for_load",
        "persistent buffers": "_persistent_model_buffer_state_dict",
    }
    methods: dict[str, Any] = {}
    for label, name in required_methods.items():
        method = getattr(train_module, name, None)
        if not callable(method):
            raise TypeError(f"Native checkpoint load lacks the required {label} API {name}")
        methods[name] = method

    eval_state = methods["_get_model_state_dict_for_eval_load"](metadata)
    frozen_parameters = methods["_frozen_checkpoint_model_param_state_dict_for_load"](
        checkpoint_keys
    )
    frozen_tensors = methods["_frozen_checkpoint_param_state_dict_for_load"](checkpoint_keys)
    if set(frozen_parameters) != set(frozen_tensors):
        raise RuntimeError("Native frozen-parameter and frozen-tensor load keys differ")
    persistent_buffers = methods["_persistent_model_buffer_state_dict"]()
    missing_buffers = sorted(set(persistent_buffers) - checkpoint_keys)
    if missing_buffers:
        raise RuntimeError(
            "Native checkpoint is missing persistent model buffers: " f"{missing_buffers[:10]}"
        )

    for label, state in (
        ("eval", eval_state),
        ("frozen", frozen_tensors),
        ("buffer", persistent_buffers),
    ):
        for key, target in state.items():
            tensor_metadata = metadata.state_dict_metadata.get(key)
            if not isinstance(tensor_metadata, TensorStorageMetadata):
                raise TypeError(f"Native {label} load target {key!r} lacks tensor metadata")
            if tuple(target.size()) != tuple(tensor_metadata.size):
                raise RuntimeError(
                    f"Native {label} load target {key!r} shape {tuple(target.size())} differs "
                    f"from checkpoint shape {tuple(tensor_metadata.size)}"
                )
            metadata_numel = math.prod(int(size) for size in tensor_metadata.size)
            if target.numel() != metadata_numel:
                raise RuntimeError(
                    f"Native {label} load target {key!r} numel {target.numel()} differs "
                    f"from checkpoint numel {metadata_numel}"
                )

    model_parts = getattr(train_module, "model_parts", None)
    if not isinstance(model_parts, Sequence) or not model_parts:
        raise RuntimeError("Native checkpoint load does not expose non-empty model_parts")
    frozen_keys_by_parameter: dict[int, list[str]] = {}
    for key, parameter in frozen_parameters.items():
        frozen_keys_by_parameter.setdefault(id(parameter), []).append(key)

    parameter_names: dict[int, list[str]] = {}
    parameter_by_id: dict[int, Any] = {}
    for part_index, model_part in enumerate(model_parts):
        for name, parameter in model_part.named_parameters():
            parameter_names.setdefault(id(parameter), []).append(f"part{part_index}.{name}")
            parameter_by_id[id(parameter)] = parameter
    if not parameter_by_id:
        raise RuntimeError("Native checkpoint load model has no parameters")
    orphan_frozen_keys = sorted(
        key for key, parameter in frozen_parameters.items() if id(parameter) not in parameter_by_id
    )
    if orphan_frozen_keys:
        raise RuntimeError(
            "Native checkpoint frozen load targets are absent from model_parts: "
            f"{orphan_frozen_keys[:10]}"
        )

    covered_keys: set[str] = set()
    parameter_ids_by_checkpoint_key: dict[str, set[int]] = {}
    assignments: list[dict[str, Any]] = []
    missing_parameters: list[str] = []
    resolver = methods["_resolve_model_checkpoint_key"]
    for parameter_id, parameter in parameter_by_id.items():
        names = parameter_names[parameter_id]
        resolved_keys = {
            key
            for name in names
            if (key := resolver(name.split(".", 1)[1], checkpoint_keys)) is not None
        }
        resolved_keys &= set(eval_state)
        frozen_keys = set(frozen_keys_by_parameter.get(id(parameter), ()))
        if not resolved_keys and not frozen_keys:
            missing_parameters.extend(names)
            continue
        authoritative_keys = resolved_keys | frozen_keys
        if len(authoritative_keys) != 1:
            raise RuntimeError(
                f"Native model parameter {names} resolves ambiguously to "
                f"{sorted(authoritative_keys)}"
            )
        for key in authoritative_keys:
            parameter_ids_by_checkpoint_key.setdefault(key, set()).add(parameter_id)
        covered_keys.update(authoritative_keys)
        assignments.append(
            {
                "parameter_names": sorted(names),
                "checkpoint_keys": sorted(authoritative_keys),
            }
        )
    if missing_parameters:
        raise RuntimeError(
            "Native checkpoint load does not cover every model parameter; missing "
            f"{missing_parameters[:10]}"
        )
    multiply_mapped = sorted(
        key
        for key, parameter_ids in parameter_ids_by_checkpoint_key.items()
        if len(parameter_ids) > 1
    )
    if multiply_mapped:
        raise RuntimeError(
            "Native checkpoint keys resolve to multiple distinct model parameters: "
            f"{multiply_mapped[:10]}"
        )

    prepared_model_keys = set(eval_state) | set(frozen_tensors)
    unused_prepared_keys = sorted(prepared_model_keys - covered_keys)
    if unused_prepared_keys:
        raise RuntimeError(
            "Native checkpoint prepares model tensors not assigned to current parameters: "
            f"{unused_prepared_keys[:10]}"
        )

    def model_bearing_key(key: str) -> bool:
        return key.endswith(".main") or key.startswith(("frozen_model.", "model_buffer.", "model."))

    def logical_model_name(key: str) -> str:
        if key.startswith("frozen_model."):
            return key.removeprefix("frozen_model.")
        if key.endswith(".main"):
            key = key.removesuffix(".main")
            return key.removeprefix("module.")
        return key

    consumed_keys = covered_keys | set(persistent_buffers)
    unused_model_bearing = {
        key for key in checkpoint_keys - consumed_keys if model_bearing_key(key)
    }
    authoritative_main_names = {
        logical_model_name(key) for key in covered_keys if key.endswith(".main")
    }
    shadowed_frozen_keys = {
        key
        for key in unused_model_bearing
        if key.startswith("frozen_model.") and logical_model_name(key) in authoritative_main_names
    }
    unexpected_unused_model_keys = sorted(unused_model_bearing - shadowed_frozen_keys)
    if unexpected_unused_model_keys:
        raise RuntimeError(
            "Native checkpoint contains unused model-bearing keys: "
            f"{unexpected_unused_model_keys[:10]}"
        )

    report = {
        "complete": True,
        "checkpoint_key_count": len(checkpoint_keys),
        "model_parameter_count": len(parameter_by_id),
        "model_parameter_checkpoint_key_count": len(covered_keys),
        "model_parameter_checkpoint_keys_sha256": _canonical_sha256(sorted(covered_keys)),
        "model_parameter_assignments_sha256": _canonical_sha256(
            sorted(assignments, key=lambda assignment: assignment["parameter_names"])
        ),
        "eval_state_key_count": len(eval_state),
        "frozen_state_key_count": len(frozen_parameters),
        "persistent_buffer_count": len(persistent_buffers),
        "persistent_buffer_keys_sha256": _canonical_sha256(sorted(persistent_buffers)),
        "shadowed_frozen_key_count": len(shadowed_frozen_keys),
        "shadowed_frozen_keys_sha256": _canonical_sha256(sorted(shadowed_frozen_keys)),
        "unused_model_bearing_key_count": 0,
        "prepared_load_key_count": len(
            set(eval_state) | set(frozen_tensors) | set(persistent_buffers)
        ),
    }
    report["sha256"] = _canonical_sha256(report)
    return report


def native_checkpoint_load_coverage_distributed(
    train_module: Any, state_dir: Path
) -> dict[str, Any]:
    """Require every rank to produce the same complete native-load coverage report."""
    try:
        local: dict[str, Any] = {
            "ok": True,
            "report": native_checkpoint_load_coverage(train_module, state_dir),
        }
    except Exception as error:  # noqa: BLE001 - all ranks must receive every local failure.
        local = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local)
    failures = [
        f"rank {rank}: {packet.get('error')}"
        if isinstance(packet, Mapping)
        else f"rank {rank}: malformed report {packet!r}"
        for rank, packet in enumerate(gathered)
        if not isinstance(packet, Mapping) or packet.get("ok") is not True
    ]
    if failures:
        raise RuntimeError(f"Native checkpoint load coverage failed: {failures}")
    reports = [packet["report"] for packet in gathered]
    if any(report != reports[0] for report in reports[1:]):
        raise RuntimeError("Native checkpoint load coverage differs across ranks")
    return reports[0]


def checkpoint_state_dir(checkpoint: Path) -> Path:
    """Resolve a checkpoint root or an explicit model-state directory."""
    if checkpoint.name == "model_and_optim":
        return checkpoint
    nested = checkpoint / "model_and_optim"
    return nested if nested.is_dir() else checkpoint


def eval_expert_parallel_degree(raw_config: Mapping[str, Any], degree: int = 8) -> int | None:
    """
    The expert-parallel degree to evaluate a checkpoint with.

    :param raw_config: The checkpoint's saved ``config.json``.
    :param degree: The evaluation degree for checkpoints trained with expert parallelism.

    :returns: ``degree``, or ``None`` when the checkpoint trained without expert parallelism
        (``train_module.ep_config`` unset, as in the OLMo 3.5 recipes); such checkpoints are
        evaluated on the same path.
    """
    if raw_config.get("train_module", {}).get("ep_config") is None:
        return None
    return degree


def configure_lm_for_eval(
    lm_config: OLMoDDPModelConfig,
    *,
    ep_path: ExpertParallelPath = ExpertParallelPath.rowwise_nvshmem,
    expert_parallel: bool = True,
) -> None:
    """
    Select evaluation attention and expert-parallel kernels without activation recomputation.

    :param expert_parallel: Whether the model is evaluated with expert parallelism. Without it,
        the trained attention backends are kept: they are what the checkpoint was trained with
        (OLMo 3.5's attention layers use ``flash_4`` without positional encodings).
    """
    if expert_parallel:
        for block in [lm_config.block, *(lm_config.block_overrides or {}).values()]:
            if isinstance(block.sequence_mixer, AttentionConfig):
                block.sequence_mixer.backend = AttentionBackendName.flex
            if block.ep is not None:
                block.ep.path = ep_path
    lm_config.recompute_each_block = False
    lm_config.recompute_all_blocks_by_chunk = False
    lm_config.two_batch_overlap = False


def build_model_and_module_config(
    raw_config: dict[str, Any],
    *,
    ep_degree: int | None,
    max_sequence_length: int,
    rank_batch_size: int,
    ep_path: ExpertParallelPath = ExpertParallelPath.rowwise_nvshmem,
) -> tuple[torch.nn.Module, OLMoDDPTrainModuleConfig, str]:
    """
    Build native LM or multimodal evaluation configs without changing router coefficients.

    :param ep_degree: The expert-parallel degree, or ``None`` to evaluate without expert
        parallelism (see :func:`eval_expert_parallel_degree`).
    """
    model_dict = raw_config["model"]
    common = dict(
        rank_microbatch_size=rank_batch_size,
        max_sequence_length=max_sequence_length,
        optim=OLMoDDPOptimizerConfig(),  # Required by the config; not built in eval-only mode.
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            only_allreduce_last_microbatch=True,
        ),
        ep_config=(
            TransformerExpertParallelConfig(degree=ep_degree) if ep_degree is not None else None
        ),
    )

    if "lm" in model_dict and "vision" in model_dict:
        model_config = MultimodalLMConfig.from_dict(model_dict)
        if not isinstance(model_config.lm, OLMoDDPModelConfig):
            raise TypeError(
                "The multimodal checkpoint does not contain an OLMoDDP language-model config"
            )
        configure_lm_for_eval(
            model_config.lm, ep_path=ep_path, expert_parallel=ep_degree is not None
        )
        model = model_config.build(init_device="meta")
        module_config = MultimodalOLMoDDPTrainModuleConfig(
            freeze_params=["vision.*"], response_logits_only=False, **common
        )
        return model, module_config, "multimodal_stage1"

    model_config = OLMoDDPModelConfig.from_dict(model_dict)
    configure_lm_for_eval(model_config, ep_path=ep_path, expert_parallel=ep_degree is not None)
    model = model_config.build(init_device="meta")
    return model, OLMoDDPTrainModuleConfig(**common), "pretrained_lm"
