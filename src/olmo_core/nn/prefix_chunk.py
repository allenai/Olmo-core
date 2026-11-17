"""Chunked prefix blocks for the online token router (memscale, 2026-11-17).

The online router runs every token through blocks ``[0, drop_layer)`` and only the kept tokens
through the rest. At long rows the training peak is one of those prefix blocks' backward
(activation-checkpoint recompute + gradient) over the WHOLE row. Here the prefix blocks run chunk
by chunk (chunk-major: chunk 0 through blocks 0..L-1, then chunk 1, ...), each block call still
under its own activation checkpoint, so the backward recomputes one (block, chunk) at a time:

* Gated DeltaNet: each (layer, chunk) call gets a FRESH state cache built from the previous
  chunk's final conv windows + recurrent state (detached), through the ``prefix_state`` block
  kwarg -- so the activation-checkpoint recompute in backward sees exactly the same incoming state.
* Full attention: each chunk attends to the earlier chunks' keys / values (written to a row-length
  buffer, read as constants) plus its own, causally (flash-attn, bottom-right aligned).

The forward equals the unchunked one up to kernel numerics. The only approximation is in the
gradient: paths that cross a chunk boundary inside the prefix (through the carried GDN state or
the earlier chunks' K/V) are cut; each token still gets its own gradient through its own chunk.

Memory: the per-block checkpoint inputs (prefix blocks x row x d_model, as before), one K/V buffer
per prefix attention layer, the block-``drop_layer`` input, + ONE (block, chunk) recompute instead
of one (block, row). FLOPs: identical to the unchunked forward + checkpointed backward (no extra pass).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

__all__ = ["prefix_chunk_size", "run_prefix_chunked", "PrefixState"]


def prefix_chunk_size() -> int:
    """Chunk length from ``OLMO_OTR_PREFIX_CHUNK`` (0 = off; set by the trainer's ``--prefix-chunk``)."""
    return int(os.environ.get("OLMO_OTR_PREFIX_CHUNK", "0") or 0)


class _PrefixKV:
    """One attention layer's view for one chunk: the shared row-length K/V buffer and the chunk start."""

    def __init__(self, store: Dict[str, Any], s: int):
        self.store = store
        self.s = s

    def attend(self, mod, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        from flash_attn import flash_attn_func

        st, s = self.store, self.s
        B, Tq = q.shape[0], q.shape[1]
        if st.get("k") is None:  # allocated at the first chunk (dtype / heads from k)
            st["k"] = torch.empty(B, st["T"], *k.shape[2:], device=k.device, dtype=k.dtype)
            st["v"] = torch.empty(B, st["T"], *v.shape[2:], device=v.device, dtype=v.dtype)
        with torch.no_grad():
            st["k"][:, s:s + Tq].copy_(k)
            st["v"][:, s:s + Tq].copy_(v)
        if s > 0:
            k = torch.cat([st["k"][:, :s], k], dim=1)
            v = torch.cat([st["v"][:, :s], v], dim=1)
        # causal with Tk > Tq: flash-attn aligns the mask bottom-right (query i sees keys <= s + i)
        return flash_attn_func(q, k, v, causal=True, softmax_scale=mod.backend.scale)


class PrefixState:
    """Passed to every prefix block call of chunk ``c`` as the ``prefix_state`` kwarg."""

    def __init__(self, run: Dict[str, Any], c: int, s: int):
        self.run, self.c, self.s = run, c, s

    # --- Gated DeltaNet
    def gdn_cache(self, mix):
        from .attention.recurrent import GatedDeltaNetStateCache

        B = self.run["B"]
        cache = GatedDeltaNetStateCache(B, mix.key_dim, mix.value_dim, mix.conv_size - 1,
                                        self.run["device"], self.run["dtype"])
        sn = self.run["gdn"].get((id(mix), self.c))  # this chunk's incoming state (None for chunk 0)
        if sn is not None:
            cache.conv_state_q, cache.conv_state_k, cache.conv_state_v = (t.clone() for t in sn[:3])
            cache.recurrent_state = sn[3].clone()
            cache.has_state = True
        return cache

    def gdn_done(self, mix, cache) -> None:
        key = (id(mix), self.c + 1)
        if key not in self.run["gdn"] and self.c + 1 < self.run["n_chunks"]:
            self.run["gdn"][key] = tuple(t.detach().clone() for t in (
                cache.conv_state_q, cache.conv_state_k, cache.conv_state_v, cache.recurrent_state))

    # --- full attention
    def attn(self, mod) -> _PrefixKV:
        if getattr(mod, "window_size", None) is not None:
            raise NotImplementedError("prefix_chunk: sliding-window attention in the prefix")
        store = self.run["kv"].setdefault(id(mod), {"T": self.run["T"]})
        return _PrefixKV(store, self.s)


def _slice_kwargs(kw: Dict[str, Any], B: int, T: int, s: int, e: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, val in kw.items():
        if key in ("cu_doc_lens", "cache_leftpad", "soft_keep", "attn_bias", "token_weight") and val is not None:
            raise NotImplementedError(f"prefix_chunk: block kwarg {key} is not supported")
        if isinstance(val, torch.Tensor) and val.dim() >= 2 and val.shape[0] == B and val.shape[1] == T:
            out[key] = val[:, s:e]
        elif isinstance(val, torch.Tensor) and val.numel() > 1:
            raise NotImplementedError(f"prefix_chunk: cannot slice block kwarg {key} of shape {tuple(val.shape)}")
        else:
            out[key] = val
    return out


def run_prefix_chunked(
    model, h: torch.Tensor, all_block_kwargs: Dict[str, Any], per_block_kwargs: Dict[int, Dict[str, Any]],
    n_prefix: int, chunk: int,
) -> torch.Tensor:
    """Run blocks ``[0, n_prefix)`` of ``model`` over ``h`` (B, T, D) in chunks of ``chunk`` tokens; see module doc."""
    B, T, D = h.shape
    kw = dict(all_block_kwargs)
    if kw.get("position_ids") is None:
        kw["position_ids"] = torch.arange(T, device=h.device)[None, :].expand(B, T)
    # the SAME number of chunks on every rank (FSDP all-gathers once per block call; rows differ in length across
    # ranks): n = max over ranks of ceil(T / chunk), and the row is split into n near-equal chunks (each <= chunk)
    n = -(-T // chunk)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        nt = torch.tensor([n], device=h.device)
        torch.distributed.all_reduce(nt, op=torch.distributed.ReduceOp.MAX)
        n = int(nt.item())
    n = max(1, min(n, T))
    cuts = [i * T // n for i in range(n + 1)]
    bounds = [(cuts[i], cuts[i + 1]) for i in range(n)]
    run: Dict[str, Any] = {"B": B, "T": T, "device": h.device, "dtype": h.dtype, "gdn": {}, "kv": {},
                           "n_chunks": len(bounds)}
    blocks = [model.blocks[str(i)] for i in range(n_prefix)]
    offload = bool(os.environ.get("OLMO_OTR_PREFIX_OFFLOAD"))
    outs = []
    for c, (s, e) in enumerate(bounds):
        x = h[:, s:e]
        ps = PrefixState(run, c, s)
        kwc = _slice_kwargs(kw, B, T, s, e)
        for i, blk in enumerate(blocks):
            if offload and torch.is_grad_enabled():
                # the block's activation-checkpoint input (the only tensor its checkpointed forward saves) goes
                # to pinned host memory and comes back for the recompute: prefix blocks x row x d_model off the GPU
                with torch.autograd.graph.save_on_cpu(pin_memory=True):
                    x = blk(x, prefix_state=ps, **kwc, **_slice_kwargs(per_block_kwargs.get(i, {}), B, T, s, e))
            else:
                x = blk(x, prefix_state=ps, **kwc, **_slice_kwargs(per_block_kwargs.get(i, {}), B, T, s, e))
        outs.append(x)
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


def prefix_attn_kv(mod, prefix_state: Optional[PrefixState]) -> None:
    """Attention.forward hook: install (or clear) this chunk's K/V view for :meth:`Attention.sdpa`."""
    mod.__dict__["_prefix_kv"] = prefix_state.attn(mod) if prefix_state is not None else None
