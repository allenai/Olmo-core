"""
Common ``nn`` function implementations.
"""

import torch

from .cross_entropy_loss import *

__all__ = [
    "cross_entropy_loss",
    "weighted_cross_entropy_loss",
    "fused_linear_cross_entropy_loss",
    "l2_normalize",
]


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # NOTE: could also use F.normalize(), but that doesn't work with DTensor at the moment.
    return x / torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32).type_as(x)
