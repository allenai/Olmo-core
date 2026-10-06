"""SDPA restricted to the flash / memory-efficient kernels, as in mm_olmo's ViT."""

from contextlib import contextmanager
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


@contextmanager
def vision_sdpa_context() -> Iterator[None]:
    """Restrict SDPA to the flash and memory-efficient backends (never cuDNN / math)."""
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
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
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, dropout_p=dropout_p
        )
