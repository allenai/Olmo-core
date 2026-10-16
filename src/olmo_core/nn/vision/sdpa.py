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


def vision_sdpa_backends(*, cudnn_first: bool = False) -> List:
    """
    Backends matching mm_olmo ViT (flash + efficient; exclude cuDNN math pitfalls).

    :param cudnn_first: Put cuDNN's attention ahead of them (see
        :func:`vision_scaled_dot_product_attention`).
    """
    if SDPBackend is None:
        return []
    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    return [SDPBackend.CUDNN_ATTENTION, *backends] if cudnn_first else backends


@contextmanager
def vision_sdpa_context(*, cudnn_first: bool = False) -> Iterator[None]:
    backends = vision_sdpa_backends(cudnn_first=cudnn_first)
    if backends and sdpa_kernel is not None:
        with sdpa_kernel(backends, set_priority=cudnn_first):
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
    cudnn_first_on_sm100: bool = False,
) -> torch.Tensor:
    """
    :func:`torch.nn.functional.scaled_dot_product_attention` on the backends of
    :func:`vision_sdpa_backends`.

    :param cudnn_first_on_sm100: On Blackwell (SM100 and newer) GPUs, try cuDNN's attention
        first. PyTorch's flash attention is slow there for the ViT's head dimension (72); cuDNN
        runs the ViT's unmasked self-attention in about a third of the time with the same accuracy
        against a float32 reference.
    """
    cudnn_first = (
        cudnn_first_on_sm100 and q.is_cuda and torch.cuda.get_device_capability(q.device)[0] >= 10
    )
    with vision_sdpa_context(cudnn_first=cudnn_first):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=dropout_p,
        )
