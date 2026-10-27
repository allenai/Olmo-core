"""Standard Muon math with AdamW-compatible update scaling.

https://arxiv.org/html/2502.16982v1#S2.SS2
Distributed integration lives in ``moe_optimizer``; these routines see complete
logical matrices, never flat optimizer shards.
"""

import math
from dataclasses import dataclass
from typing import Iterable

import torch


def muon_momentum(
    grad: torch.Tensor, previous: torch.Tensor, momentum: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute FP32 momentum and Nesterov direction before any BF16 conversion."""
    updated = momentum * previous.float() + grad.float()
    return updated, momentum * updated + grad.float()


def apply_muon_update(
    param: torch.Tensor,
    momentum: torch.Tensor,
    step: torch.Tensor,
    updated_momentum: torch.Tensor,
    update: torch.Tensor,
    *,
    lr: float,
    weight_decay: float,
    step_factor: torch.Tensor,
) -> None:
    """Apply an already scaled local update; a skipped step preserves all state."""
    momentum.copy_(torch.where(step_factor.bool(), updated_momentum, momentum))
    candidate = param * (1 - lr * weight_decay)
    candidate = candidate - lr * update
    param.copy_(torch.where(step_factor.bool(), candidate, param))
    step.add_(step_factor)


@dataclass(frozen=True)
class MuonLayout:
    """Logical storage view and projection axis, recovered from the owning module."""

    shape: tuple[int, ...]
    split_dim: int | None = None
    shared_experts: bool = False


def collect_muon_layouts(model_parts: Iterable[torch.nn.Module]) -> dict[str, MuonLayout]:
    """Recover v2 router and expert layouts, including DDP-prefixed parameter names."""
    from olmo_core.nn.moe.v2.routed_experts import ExpertActivation, RoutedExperts
    from olmo_core.nn.moe.v2.router import MoERouterV2
    from olmo_core.nn.moe.v2.shared_experts import SharedExperts

    layouts = {}
    for part in model_parts:
        for name, module in part.named_modules():
            prefix = f"{name}." if name else ""
            if isinstance(module, MoERouterV2):
                layouts[prefix + "weight"] = MuonLayout((module.num_experts, module.d_model))
            elif isinstance(module, SharedExperts):
                packed = module.activation == ExpertActivation.swiglu
                layouts[prefix + "w_up_gate"] = MuonLayout(
                    (module.d_model, module.num_experts, (2 if packed else 1) * module.hidden_size),
                    split_dim=-1 if packed else None,
                    shared_experts=True,
                )
            elif isinstance(module, RoutedExperts):
                packed = module.activation in (
                    ExpertActivation.swiglu,
                    ExpertActivation.gpt_oss_swiglu,
                )
                layouts[prefix + "w_up_gate"] = MuonLayout(
                    tuple(module.w_up_gate.shape), split_dim=-2 if packed else None
                )
    return layouts


def uses_muon(name: str, shape: tuple[int, ...]) -> bool:
    """Select hidden matrices; keep norms, embeddings, heads and scalar gates on AdamW."""
    parts = name.split(".")
    if (
        len(shape) == 1
        and parts[-1] == "weight"
        and any(part in {"routed_experts_router", "shared_experts_router"} for part in parts)
    ):
        raise ValueError(f"Flattened router {name} requires a MuonLayout from its owning module")
    if any("norm" in part or "conv" in part for part in parts):
        return False
    if (
        "embeddings" in parts
        or "lm_head" in parts
        # KDA/recurrent write gates produce one scalar per head, not feature vectors.
        or "w_b" in parts
        or name.endswith("bias")
        or parts[-1].startswith("b_")
    ):
        return False
    return len(shape) == 2 or (len(shape) == 3 and name.endswith(("w_up_gate", "w_down")))


def newton_schulz(matrix: torch.Tensor) -> torch.Tensor:
    """Orthogonalize independently over the last two axes with five iterations."""
    if matrix.ndim < 2:
        raise ValueError("Muon requires matrices")
    x = matrix.float()
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    if x.is_cuda:
        x = x.bfloat16()
    for a, b, c in [(3.4445, -4.7750, 2.0315)] * 5:
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return (x.mT if transposed else x).float()


def muon_direction(
    name: str, direction: torch.Tensor, rms: float = 0.2, layout: MuonLayout | None = None
) -> torch.Tensor:
    """Rescale each independent matrix; split packed SwiGLU up and gate projections."""
    if layout is not None:
        storage_shape = direction.shape
        direction = direction.reshape(layout.shape)
        if layout.shared_experts:
            direction = direction.transpose(0, 1)  # (D, E, 2H) -> (E, D, 2H)
        update = _matrix_direction(direction, rms, layout.split_dim)
        if layout.shared_experts:
            update = update.transpose(0, 1)
        return update.reshape(storage_shape)
    split_dim = (-1 if direction.ndim == 2 else -2) if name.endswith("w_up_gate") else None
    return _matrix_direction(direction, rms, split_dim)


def _matrix_direction(direction: torch.Tensor, rms: float, split_dim: int | None) -> torch.Tensor:
    if split_dim is not None:
        if direction.shape[split_dim] % 2:
            raise ValueError("Packed SwiGLU up/gate dimension must be even")
        up, gate = direction.chunk(2, dim=split_dim)
        return torch.cat(
            (_matrix_direction(up, rms, None), _matrix_direction(gate, rms, None)), dim=split_dim
        )
    scale = math.sqrt(max(direction.shape[-2:])) * rms
    # Batch independent experts to amortize kernel launches. Limit the batch for
    # larger matrices so Newton-Schulz's Gram and update temporaries stay bounded.
    if direction.ndim == 3:
        batch_size = _expert_batch_size(direction.shape[-2], direction.shape[-1])
        return torch.cat(
            [newton_schulz(chunk) * scale for chunk in direction.split(batch_size)], dim=0
        )
    return newton_schulz(direction) * scale


def _expert_batch_size(rows: int, cols: int) -> int:
    # Conservative fp32 workspace estimate, even though CUDA matmuls use bf16.
    # This bounds temporary workspace, not the full input/output allocation.
    workspace_per_expert = 4 * (3 * rows * cols + 3 * min(rows, cols) ** 2)
    return max(1, min(128, (256 * 1024**2) // workspace_per_expert))
