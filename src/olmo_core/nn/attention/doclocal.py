"""
Training-only document-local attention for chosen full-attention blocks (``--st-doclocal-blocks``, lenscale).

In a TRAINING forward, the listed blocks' attention uses the ``chunked`` pattern of
:mod:`~olmo_core.nn.attention.chunked_mask`: a document token (markers included) attends causally to its own
document and to the FREE tokens (prompt / instruction before the documents, query, answer); FREE tokens attend
everything. Eval forwards (``model.eval()``) are untouched. Runs on FlexAttention with a block mask built from the
per-token roles, which :class:`~olmo_core.nn.transformer.Transformer` threads to the listed blocks as ``chunk_ids``
(gathered to the compacted row past an online-router cut, so the kept tokens of later blocks stay local too).

FLOPs: the meter prices each block's attention at the causal ``L (L + 1) / 2`` pairs of the row length it ran at;
:func:`pairs_saved` gives, per row, the causal pairs minus the pairs this mask allows, and the model subtracts
``factor x 12 x n_heads x head_dim x saved`` from ``_flop_extra_actual`` (``factor`` = the router's per-block share,
2/3 for a frozen-base LoRA block), i.e. only the attended pairs are charged.
"""

from typing import Iterable, List

import torch

from .chunked_mask import FREE_CHUNK_ID, PAD_CHUNK_ID, AttentionPattern, build_chunked_mask_mod

__all__ = ["enable_doclocal", "gather_roles", "pairs_saved", "doclocal_attention"]

_CACHE: dict = {}


def enable_doclocal(model, blocks: Iterable[int], pattern: str = "chunked") -> List[int]:
    """Mark the listed blocks' attention for training-only document-local masking; returns the sorted indices."""
    from . import Attention

    idx = sorted(set(int(i) for i in blocks))
    if not idx:
        raise ValueError("doclocal: no blocks given")
    if getattr(model, "_pooled_soft_tokens", None) is None:
        raise ValueError("doclocal needs the softtoken variant (marker / EOS ids come from its config)")
    for i in idx:
        att = model.blocks[str(i)].attention
        if type(att) is not Attention:
            raise ValueError(f"doclocal: block {i} is {type(att).__name__}, not a full-attention block")
        att._doclocal = True
        att._doclocal_pattern = pattern
    if pattern not in ("chunked", "standard"):
        raise ValueError(f"doclocal pattern {pattern!r}: chunked | standard (backend control: plain causal via flex)")
    model._doclocal_blocks = frozenset(idx)
    model._doclocal_pattern = pattern
    return idx


def gather_roles(roles: torch.Tensor, order: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Roles of a compacted row: ``order`` (B, W) source columns, ``valid`` (B, W) real kept columns."""
    g = torch.gather(roles, 1, order.clamp(max=roles.shape[1] - 1))
    return torch.where(valid, g, torch.full_like(g, PAD_CHUNK_ID))


@torch.no_grad()
def pairs_saved(roles: torch.Tensor) -> torch.Tensor:
    """Per row: causal pairs over the real tokens (``L (L + 1) / 2``) minus the pairs the chunked mask allows.
    Documents are contiguous runs of one role (true for the full row and for an order-preserving compaction)."""
    B, T = roles.shape
    real = roles != PAD_CHUNK_ID
    L = real.sum(dim=1).to(torch.float64)
    free = (roles == FREE_CHUNK_ID) & real
    doc = (roles >= 0) & real
    # free tokens strictly before each position, among real tokens
    free_cum = torch.cumsum(free.to(torch.float64), dim=1)
    free_before = free_cum - free.to(torch.float64)
    real_cum = torch.cumsum(real.to(torch.float64), dim=1)  # real tokens up to and including this one
    ar = torch.arange(T, device=roles.device)[None, :].expand(B, T)
    change = torch.ones_like(roles, dtype=torch.bool)
    change[:, 1:] = roles[:, 1:] != roles[:, :-1]
    start = torch.cummax(torch.where(change, ar, torch.zeros_like(ar)), dim=1).values
    rank = (ar - start).to(torch.float64)  # tokens of the same run before this one
    allowed = torch.where(free, real_cum, torch.zeros_like(real_cum))
    allowed = allowed + torch.where(doc, free_before + rank + 1.0, torch.zeros_like(real_cum))
    return L * (L + 1.0) / 2.0 - allowed.sum(dim=1)


def _block_mask(roles: torch.Tensor, T: int, pattern: str = "chunked", block_size: int = 128):
    from torch.nn.attention.flex_attention import create_block_mask

    key = (id(roles), roles.data_ptr(), roles._version, tuple(roles.shape), T, pattern)
    hit = _CACHE.get("cur")
    if hit is not None and hit[0] == key:
        return hit[1]
    mm = build_chunked_mask_mod(AttentionPattern(name=pattern), roles)
    bm = create_block_mask(mm, roles.shape[0], None, T, T, device=roles.device, BLOCK_SIZE=(block_size, block_size))
    _CACHE["cur"] = (key, bm)
    return bm


_CHECKS = {"n": 0}


@torch.no_grad()
def _self_check(q, k, v, roles, out, scale, pattern, P=1024):
    """First few calls: compare the flex output on the causal prefix [0, P) of row 0 with a dense masked SDPA, and
    log allowed-key counts per query role (free queries must see every real earlier key)."""
    import logging

    from .chunked_mask import build_chunked_allowed_mask

    P = min(P, q.shape[1])
    r = roles[:1, :P]
    allowed = build_chunked_allowed_mask(AttentionPattern(name=pattern), r).reshape(1, P, P)
    qq, kk, vv = (t[:1, :P].transpose(1, 2).float() for t in (q, k, v))
    ref = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv, attn_mask=allowed[:, None], scale=scale)
    diff = (ref.transpose(1, 2) - out[:1, :P].float()).abs().max().item()
    T = roles.shape[1]
    real = roles[0] != PAD_CHUNK_ID
    free_q = (roles[0, :P] == FREE_CHUNK_ID)
    cnt = allowed[0].sum(1).float()
    causal_real = torch.cumsum(real[:P].float(), 0)
    msg = (f"[doclocal-check] call {_CHECKS['n']} T={T} pattern={pattern} prefix={P}: max|flex - dense|={diff:.3e} "
           f"(|out| max {out[:1, :P].abs().max().item():.3e}); roles: real={int(real.sum())} free={int((roles[0] == FREE_CHUNK_ID).sum())} "
           f"doc={int((roles[0] >= 0).sum())} n_docs={int(roles[0].max()) + 1}; prefix free queries see "
           f"{cnt[free_q].sum().item():.0f}/{causal_real[free_q].sum().item():.0f} causal real keys, doc queries "
           f"{cnt[roles[0, :P] >= 0].mean().item() if bool((roles[0, :P] >= 0).any()) else 0:.1f} keys on average")
    logging.getLogger(__name__).warning(msg)
    print(msg, flush=True)


def doclocal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, roles: torch.Tensor,
                       scale: float, pattern: str = "chunked") -> torch.Tensor:
    """``q`` (B, T, H, D), ``k``/``v`` (B, T, H_kv, D) -> (B, T, H, D) under the chunked document-local mask."""
    from .document_chunked import _flex_attention

    B, T, H, _ = q.shape
    if roles.shape != (B, T):
        raise RuntimeError(f"doclocal: roles {tuple(roles.shape)} do not match the attention input {(B, T)}")
    rep = H // k.shape[2]
    if rep > 1:
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
    bm = _block_mask(roles, T, pattern)
    out = _flex_attention(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
                          v.transpose(1, 2).contiguous(), block_mask=bm, scale=scale)
    out = out.transpose(1, 2).contiguous()
    if _CHECKS["n"] < 12 and torch.is_grad_enabled():
        _self_check(q, k, v, roles, out, scale, pattern)
        _CHECKS["n"] += 1
    return out
