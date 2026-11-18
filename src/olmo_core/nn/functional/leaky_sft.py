"""LeakySFT's two disjoint token groups, including the low-memory linear-CE path."""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .cross_entropy_loss import fused_linear_cross_entropy_loss


def validate_cpt_loss_scaler(value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("cpt_loss_scaler must be finite and in [0, 1]")


def leaky_ce_sums(
    h: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    labels: torch.Tensor,
    output_mask: torch.Tensor,
    *,
    cpt_loss_scaler: float,
    ignore_index: int,
    fused: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute answer/context CE sums, with context diagnostics even at scaler zero.

    Each fused call receives one uniformly weighted group. Scaling its scalar result
    scales ALL of its precomputed gradients correctly (including projection weights).
    No assumption about Liger's unreduced backward support is needed.
    """
    flat_h = h.reshape(-1, h.shape[-1])
    targets = labels.reshape(-1)
    output = output_mask.reshape(-1)
    valid = targets != ignore_index

    def group_sum(mask: torch.Tensor) -> torch.Tensor:
        # Empty groups must be differentiable zeros, including for every head parameter.
        if not mask.any():
            zero = flat_h.sum() * 0.0 + weight.sum() * 0.0
            return zero if bias is None else zero + bias.sum() * 0.0
        selected_h, selected_labels = flat_h[mask], targets[mask]
        if fused:
            loss, _ = fused_linear_cross_entropy_loss(
                selected_h,
                weight,
                selected_labels,
                bias=bias,
                reduction="sum",
                accum_dtype=torch.float32,
            )
            return loss
        return F.cross_entropy(F.linear(selected_h, weight, bias), selected_labels, reduction="sum")

    output_sum = group_sum(valid & output)
    with torch.set_grad_enabled(torch.is_grad_enabled() and cpt_loss_scaler != 0):
        context_sum = group_sum(valid & ~output)
    return output_sum, context_sum
