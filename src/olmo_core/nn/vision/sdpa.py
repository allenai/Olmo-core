"""Memory-efficient SDPA helpers for vision modules (ViT + connector)."""

from contextlib import contextmanager
from typing import Iterator, List, Optional

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover
    SDPBackend = None  # type: ignore
    sdpa_kernel = None  # type: ignore


def vision_sdpa_backends() -> List:
    """Backends matching mm_olmo ViT (flash + efficient; exclude cuDNN math pitfalls)."""
    if SDPBackend is None:
        return []
    return [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]


@contextmanager
def vision_sdpa_context() -> Iterator[None]:
    backends = vision_sdpa_backends()
    if backends and sdpa_kernel is not None:
        with sdpa_kernel(backends):
            yield
    else:
        yield


def vision_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    with vision_sdpa_context():
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=dropout_p,
        )
