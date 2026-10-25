"""
Sequence-chunked FFN sublayer (``--ffn-chunk-tokens N``): in training, the listed blocks run their FFN sublayer
``h + FFN(norm(h))`` over chunks of ``N`` tokens, each chunk under its own (non-reentrant) activation checkpoint, so the
forward + backward working set of the norm, the ``T x d_ff`` FFN intermediates and any early-LoRA adapters on the FFN
linears is ``N`` tokens long instead of the whole row.

Why (2026-10-25 max-length probes, ``records/online-token-router.md``): with full activation checkpointing the peak of a
long routed step is the recompute of ONE full-length block below the router's drop layer -- FFN intermediates, RMSNorm
upcasts, adapter activations and the autograd buffers of their gradients -- not the saved residual stream. The FFN is
per-token, so chunking it is exact (same outputs and gradients up to summation order). The cost is one more forward of
the FFN per chunked block in backward (activation recompute, which the FLOP meter does not charge, as for full AC).

Only the pre-norm :class:`~olmo_core.nn.transformer.block.TransformerBlock` layout
(``h = x + mixer(norm(x)); out = h + FFN(norm(h))``) is supported. Rows no longer than ``N`` (e.g. compacted rows past
the drop layer) take the normal path.
"""

from __future__ import annotations

import logging
from types import MethodType
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .ffn_width import parse_layer_range

log = logging.getLogger(__name__)

__all__ = ["install_ffn_chunk"]


def _ffn_sublayer(block: nn.Module, h: torch.Tensor) -> torch.Tensor:
    return block.feed_forward_residual_stream(h, block.feed_forward(block.feed_forward_norm(h)))  # type: ignore


def _chunked_forward(self: nn.Module, x: torch.Tensor, *,
                     loss_div_factor: Optional[Union[torch.Tensor, float]] = None, **kwargs) -> torch.Tensor:
    del loss_div_factor
    h = self.attention_residual_stream(x, self.attention(self.attention_norm(x), **kwargs))  # type: ignore
    n = self._ffn_chunk  # type: ignore[attr-defined]
    dim = h.dim() - 2  # sequence dim of (B, T, D) or (T, D)
    T = h.shape[dim]
    if not (self.training and torch.is_grad_enabled()) or T <= n:
        return _ffn_sublayer(self, h)
    self._ffn_chunk_calls += 1  # type: ignore[attr-defined]
    outs = [checkpoint(_ffn_sublayer, self, h.narrow(dim, s, min(n, T - s)), use_reentrant=False)
            for s in range(0, T, n)]
    return torch.cat(outs, dim=dim)


def install_ffn_chunk(model: Any, tokens: int, layers: str = "") -> List[int]:
    """Patch the forward of every listed block (default: all). Returns the block indices."""
    if tokens <= 0:
        raise ValueError(f"ffn chunk needs tokens > 0, got {tokens}")
    from .transformer.block import TransformerBlock

    blocks = list(model.blocks.values())
    ids = parse_layer_range(layers, len(blocks))
    for i in ids:
        b = blocks[i]
        if type(b) is not TransformerBlock:
            raise RuntimeError(f"ffn chunk: block {i} is {type(b).__name__}, only the pre-norm TransformerBlock "
                               "(h = x + mixer(norm(x)); out = h + FFN(norm(h))) is supported")
        b._ffn_chunk = int(tokens)
        b._ffn_chunk_calls = 0
        b.forward = MethodType(_chunked_forward, b)
    model._ffn_chunk = {"tokens": int(tokens), "layers": ids}
    log.info("[ffn-chunk] %d-token FFN sublayer chunks on %d blocks", tokens, len(ids))
    return ids
