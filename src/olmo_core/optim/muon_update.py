"""
Pure-tensor building blocks for Muon updates, shared by :class:`~olmo_core.optim.moe_optimizer.OLMoDDPOptimizer`.

These functions carry no optimizer state and no distributed logic, so they can be tested on
their own and reused by any optimizer that needs an orthogonalized momentum update.
"""

import math
from typing import Optional, Sequence

import torch

from .muon import MuonAdjustLRStrategy

__all__ = ["newton_schulz", "is_muon_matrix_shape", "muon_adjusted_lr"]

NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
"""Quintic Newton-Schulz coefficients from the original Muon implementation."""


def is_muon_matrix_shape(shape: Sequence[int]) -> bool:
    """
    Whether a parameter of this shape can take a Muon update: the trailing two dimensions form a
    matrix (both > 1) and any leading dimensions are treated as a batch of independent matrices
    (e.g. ``(num_experts, out, in)`` expert weights). 1D parameters and degenerate matrices such
    as depthwise conv weights ``(C, 1, k)`` are not eligible.
    """
    return len(shape) >= 2 and shape[-1] > 1 and shape[-2] > 1


def muon_adjusted_lr(
    lr: float, shape: Sequence[int], strategy: Optional[MuonAdjustLRStrategy]
) -> float:
    """
    Scale the base learning rate for a Muon update of a parameter with the given shape, matching
    the conventions of the ``dion`` package used by :class:`~olmo_core.optim.MuonConfig`.

    :param lr: The base (AdamW-style) learning rate of the param group.
    :param shape: The full parameter shape; only the trailing ``(out, in)`` dims matter.
    :param strategy: ``rms_norm`` (Moonlight: ``0.2 * sqrt(max(out, in))``), ``spectral_norm``
        (``sqrt(out / in)``), or ``None`` for no adjustment.
    """
    fan_out, fan_in = shape[-2], shape[-1]
    if strategy is None:
        return lr
    if strategy == MuonAdjustLRStrategy.rms_norm:
        return lr * 0.2 * math.sqrt(max(fan_out, fan_in))
    if strategy == MuonAdjustLRStrategy.spectral_norm:
        return lr * math.sqrt(fan_out / fan_in)
    raise ValueError(f"Unknown Muon LR adjustment strategy: {strategy}")


def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Approximate the orthogonal polar factor of each matrix in ``G`` with the quintic Newton-Schulz
    iteration, run in bfloat16. ``G`` has shape ``(..., out, in)``; leading dims are a batch of
    independent matrices and are orthogonalized separately.

    :returns: A tensor of the same shape as ``G`` in ``G``'s dtype.
    """
    assert G.ndim >= 2, "Expected a matrix or a batch of matrices"
    a, b, c = NS_COEFFICIENTS
    X = G.to(torch.bfloat16)
    # Per-matrix Frobenius normalization so the top singular value is <= 1.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    transposed = G.shape[-2] > G.shape[-1]
    if transposed:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)
