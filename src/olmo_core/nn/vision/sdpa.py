"""SDPA restricted to the flash / memory-efficient kernels, as in mm_olmo's ViT."""

from contextlib import contextmanager
from typing import Iterator, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

CUDNN_FIRST_MIN_VERSION = (9, 13)
"""The oldest cuDNN that :func:`vision_scaled_dot_product_attention` puts first on SM100. cuDNN
9.10.2 builds no execution plan there for the ViT's attention (16 heads of 72, bfloat16, with or
without gradients), and PyTorch does not fall back; 9.13 (the runtime the training jobs load) and
9.20 run all of it (B300, 2026-10-18)."""


@torch.compiler.assume_constant_result
def cudnn_runtime_version() -> Optional[Tuple[int, int, int]]:
    """The loaded cuDNN library's version, or ``None`` without cuDNN.

    Unlike :func:`torch.backends.cudnn.version`, this does not raise when the loaded library
    differs from the one PyTorch was built against, as it does in the training image (built
    against 9.20, loading 9.13).
    """
    try:
        return tuple(torch._C._cudnn.getRuntimeVersion())  # type: ignore[return-value]
    except (AttributeError, RuntimeError):
        return None


def cudnn_first_supported(device: torch.device) -> bool:
    """Whether cuDNN's attention goes first on ``device``: an SM100 or newer GPU with a cuDNN of at
    least :data:`CUDNN_FIRST_MIN_VERSION`."""
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 10:
        return False
    version = cudnn_runtime_version()
    return version is not None and version[:2] >= CUDNN_FIRST_MIN_VERSION


@contextmanager
def vision_sdpa_context(
    device: Optional[torch.device] = None, *, cudnn_first: bool = False
) -> Iterator[None]:
    """Restrict SDPA to the flash and memory-efficient backends (never math) on CUDA.

    Other devices (e.g. CPU, where only the math kernel handles dropout) keep PyTorch's default
    kernel selection.

    :param cudnn_first: Put cuDNN's attention ahead of the other backends.
    """
    if device is not None and device.type != "cuda":
        yield
        return
    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    if cudnn_first:
        backends = [SDPBackend.CUDNN_ATTENTION, *backends]
    with sdpa_kernel(backends, set_priority=cudnn_first):
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
    :func:`torch.nn.functional.scaled_dot_product_attention` on the vision backends.

    :param cudnn_first_on_sm100: On Blackwell (SM100 and newer) GPUs, try cuDNN's attention
        first. PyTorch's flash attention is slow there for the ViT's head dimension (72); cuDNN
        runs the ViT's unmasked self-attention in about a third of the time with the same accuracy
        against a float32 reference. Older cuDNNs are skipped (:func:`cudnn_first_supported`).
    """
    cudnn_first = cudnn_first_on_sm100 and cudnn_first_supported(q.device)
    with vision_sdpa_context(q.device, cudnn_first=cudnn_first):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, dropout_p=dropout_p
        )
