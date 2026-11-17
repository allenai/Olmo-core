"""Chunked prefix blocks for the online token router (memscale, 2026-11-17).

The online router runs every token through blocks ``[0, drop_layer)`` and only the kept tokens
through the rest. At long rows the training peak is one of those prefix blocks' backward over the
WHOLE row. This module runs the prefix blocks chunk by chunk instead, with peak memory set by one
chunk:

* Forward: chunk ``c`` (tokens ``[s, e)``) runs blocks ``[0, drop_layer)`` inside a non-reentrant
  activation checkpoint, so only the chunk's input is kept. Gated DeltaNet state (conv windows +
  recurrent state) is carried from chunk to chunk exactly (the mixer's chunked-prefill cache),
  and each full-attention layer attends to the earlier chunks' keys / values (stored as constants
  in a row-length buffer) plus its own, causally -- the forward equals the unchunked one up to
  kernel numerics.
* Backward: autograd recomputes one chunk at a time (chunks are independent subgraphs), restoring
  that chunk's incoming state from a snapshot, and back-propagates the chunk's slice of
  dLoss/d(block-``drop_layer`` input) into the adapters. The only approximation: gradient paths
  that cross a chunk boundary inside the prefix (through the carried state or the earlier chunks'
  keys / values) are cut.

The prefix blocks run UNWRAPPED (no per-block activation checkpoint) inside the chunk checkpoint:
a stateful mixer recomputed by a nested checkpoint would see the advanced state.

Memory: O(row) buffers (block-0 input, block-``drop_layer`` input, attention K/V of the prefix's
attention layers, one GDN state snapshot per chunk) + one chunk's full activations through the
prefix blocks. FLOPs: one extra forward of the prefix blocks (the recompute), charged by the caller.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.checkpoint import checkpoint

__all__ = ["prefix_chunk_size", "run_prefix_chunked"]


def prefix_chunk_size() -> int:
    """Chunk length from ``OLMO_OTR_PREFIX_CHUNK`` (0 = off; set by the trainer's ``--prefix-chunk``)."""
    return int(os.environ.get("OLMO_OTR_PREFIX_CHUNK", "0") or 0)


def _inner(block):
    return getattr(block, "_checkpoint_wrapped_module", block)


class _PrefixKV:
    """Per attention layer: the row-length K/V buffer and the current chunk start."""

    def __init__(self, T: int):
        self.T = T
        self.k = None
        self.v = None
        self.s = 0

    def attend(self, mod, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        from flash_attn import flash_attn_func

        B, Tq = q.shape[0], q.shape[1]
        if self.k is None:  # row-length buffers, allocated at the first chunk (dtype / heads from k)
            self.k = torch.empty(B, self.T, *k.shape[2:], device=k.device, dtype=k.dtype)
            self.v = torch.empty(B, self.T, *v.shape[2:], device=v.device, dtype=v.dtype)
        s = self.s
        with torch.no_grad():
            self.k[:, s:s + Tq].copy_(k)
            self.v[:, s:s + Tq].copy_(v)
        if s > 0:
            k = torch.cat([self.k[:, :s], k], dim=1)
            v = torch.cat([self.v[:, :s], v], dim=1)
        # causal with Tk > Tq: flash-attn aligns the mask bottom-right (query i sees keys <= s + i)
        return flash_attn_func(q, k, v, causal=True, softmax_scale=mod.backend.scale)


def _slice_kwargs(kw: Dict[str, Any], B: int, T: int, s: int, e: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, val in kw.items():
        if isinstance(val, torch.Tensor) and val.dim() >= 2 and val.shape[0] == B and val.shape[1] == T:
            out[key] = val[:, s:e]
        elif isinstance(val, torch.Tensor) and val.dim() >= 1 and val.numel() > 1:
            raise NotImplementedError(f"prefix_chunk: cannot slice block kwarg {key} of shape {tuple(val.shape)}")
        elif key in ("cu_doc_lens", "cache_leftpad", "soft_keep", "attn_bias") and val is not None:
            raise NotImplementedError(f"prefix_chunk: block kwarg {key} is not supported")
        else:
            out[key] = val
    return out


def run_prefix_chunked(
    model, h: torch.Tensor, all_block_kwargs: Dict[str, Any], per_block_kwargs: Dict[int, Dict[str, Any]],
    n_prefix: int, chunk: int,
) -> torch.Tensor:
    """Run blocks ``[0, n_prefix)`` of ``model`` over ``h`` (B, T, D) in chunks of ``chunk`` tokens; see module doc."""
    from .attention import Attention
    from .attention.recurrent import GatedDeltaNet, GatedDeltaNetStateCache

    B, T, D = h.shape
    blocks = [_inner(model.blocks[str(i)]) for i in range(n_prefix)]
    kw = dict(all_block_kwargs)
    if kw.get("position_ids") is None:
        kw["position_ids"] = torch.arange(T, device=h.device)[None, :].expand(B, T)
    gdn: List[Tuple[Any, Any]] = []
    attn: List[Tuple[Any, _PrefixKV]] = []
    for blk in blocks:
        mix = getattr(blk, "attention", None)
        if isinstance(mix, GatedDeltaNet):
            if mix.state_cache is not None:
                raise RuntimeError("prefix_chunk: GatedDeltaNet already has a state cache")
            cache = GatedDeltaNetStateCache(B, mix.key_dim, mix.value_dim, mix.conv_size - 1, h.device, h.dtype)
            gdn.append((mix, cache))
        elif isinstance(mix, Attention):
            if getattr(mix, "window_size", None) is not None:
                raise NotImplementedError("prefix_chunk: sliding-window attention in the prefix")
            attn.append((mix, _PrefixKV(T)))
        else:
            raise NotImplementedError(f"prefix_chunk: sequence mixer {type(mix).__name__}")
    snaps: Dict[int, Any] = {}
    bounds = [(s, min(s + chunk, T)) for s in range(0, T, chunk)]

    def run_chunk(x: torch.Tensor, c: int) -> torch.Tensor:
        s, e = bounds[c]
        if c not in snaps:  # first (checkpointed) forward: the caches hold the state at s -> snapshot it
            snaps[c] = [None if c == 0 else tuple(t.detach().clone() for t in (
                cache.conv_state_q, cache.conv_state_k, cache.conv_state_v, cache.recurrent_state))
                for _, cache in gdn]
        # fresh tensors (not in-place copies): the previous chunk's forward wrote these buffers under autograd
        for (mix, cache), sn in zip(gdn, snaps[c]):
            if sn is None:
                for name in ("conv_state_q", "conv_state_k", "conv_state_v"):
                    setattr(cache, name, torch.zeros_like(getattr(cache, name)).detach())
                cache.recurrent_state = None
                cache.has_state = False
            else:
                cache.conv_state_q, cache.conv_state_k, cache.conv_state_v = (t.clone() for t in sn[:3])
                cache.recurrent_state = sn[3].clone()
                cache.has_state = True
        try:
            for mix, cache in gdn:
                object.__setattr__(mix, "state_cache", cache)
            for mix, pkv in attn:
                pkv.s = s
                object.__setattr__(mix, "_prefix_kv", pkv)
            kwc = _slice_kwargs(kw, B, T, s, e)
            for i, blk in enumerate(blocks):
                x = blk(x, **kwc, **_slice_kwargs(per_block_kwargs.get(i, {}), B, T, s, e))
        finally:
            for mix, _ in gdn:
                object.__setattr__(mix, "state_cache", None)
            for mix, _ in attn:
                object.__setattr__(mix, "_prefix_kv", None)
        return x

    outs = []
    for c, (s, e) in enumerate(bounds):
        x = h[:, s:e]
        if torch.is_grad_enabled():
            outs.append(checkpoint(run_chunk, x, c, use_reentrant=False))
        else:
            outs.append(run_chunk(x, c))
    out = torch.cat(outs, dim=1)
    if os.environ.get("OLMO_OTR_PREFIX_VERIFY") and not getattr(model, "_prefix_verified", False):
        model._prefix_verified = True
        with torch.no_grad():
            ref = h
            for i, blk in enumerate(blocks):
                ref = blk(ref, **all_block_kwargs, **per_block_kwargs.get(i, {}))
            d = (out.float() - ref.float()).abs()
            print(f"[prefix-chunk] verify T={T} chunk={chunk}: max|diff| {float(d.max()):.4g} mean|diff| "
                  f"{float(d.mean()):.4g} mean|ref| {float(ref.float().abs().mean()):.4g}", flush=True)
            del ref, d
    return out
