"""Group-partitioned attention on top of gold-document drop (2026-11-20, ``--st-attn-groups``).

In the ATTENTION layers only (Gated DeltaNet layers and FFNs keep the whole row), each training row's
documents are split, fresh every forward, into

* **hubs**: the gold documents plus ``decoys`` randomly chosen distractor documents treated identically (so the
  pattern does not mark gold), and any preamble tokens before the first document;
* ``G`` **groups**: the remaining documents, randomly partitioned into ``G`` near-equal groups.

A group token attends (causally, at its original position, RoPE already applied) to its own group and to the hubs;
a hub token attends to hubs only; the question / answer tokens after the last document (the **strip**) attend to
everything. With probability ``full_frac`` a row keeps ordinary full attention (all of it is strip).

Every token carries a group id (``attn_group_ids``, (B, T) int): ``HUB`` = 0, groups 1..G, ``STRIP`` = -1,
``PAD`` = -2 (after the row's first EOS: output zero, it feeds nothing real). Inter-document tokens take the id of
the preceding document.

Compute is a real saving, not a mask: per row, the hub set and each (hub + group g) set are gathered into their own
shorter sequences and run together in ONE ``flash_attn_varlen_func`` call (causal, bottom-right aligned); the strip
runs as a dense ``flash_attn_func`` over its prefix. Hub queries inside a group sequence are computed and discarded
(their output comes from the hub-only sequence). The same function serves the online router's chunked prefix
(queries = the tail ``[q_start, Tk)`` of the keys): every gathered query set is then the tail of its gathered key set,
so the bottom-right causal alignment stays exact.

FLOPs (:func:`charge_attn_groups`): per attention layer the meter already charged ``L (L + 1) / 2`` (query, key)
pairs per row of ``L`` real tokens; this adds ``fac x 12 x H x D x (actual - charged)`` to
``model._flop_extra_actual``, where ``actual`` counts every pair the calls above compute (repeated hub tokens, the
discarded hub queries and the strip included) and ``fac`` is the online router's per-block forward/backward share.
Charged once per forward from the group ids (not inside the attention call, which activation checkpointing re-runs).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

HUB, STRIP, PAD = 0, -1, -2

__all__ = ["HUB", "STRIP", "PAD", "AttnGroupConfig", "AttnGroupBuilder", "group_attention", "group_attention_pairs",
           "charge_attn_groups", "group_mask_reference"]


@dataclass
class AttnGroupConfig:
    groups: int = 0
    #: decoy distractor documents added to the hubs; -1 = as many as the row has gold documents
    decoys: int = -1
    full_frac: float = 0.1
    seed: int = 0
    log_every: int = 20


class AttnGroupBuilder:
    """Per-forward group ids from the (post gold-drop) token rows and the online router's gold table."""

    def __init__(self, cfg: AttnGroupConfig, *, router: Any, doc_start_id: int, doc_end_id: int, eos_id: int,
                 rank: int = 0):
        if cfg.groups < 1:
            raise ValueError("attn groups: groups must be >= 1")
        self.cfg = cfg
        self.router = router
        self.ds, self.de, self.eos = int(doc_start_id), int(doc_end_id), int(eos_id)
        self.rng = random.Random(cfg.seed * 1_000_003 + rank * 7919 + 29)
        self.rank = rank
        self.calls = 0
        self.st: Dict[str, float] = {}

    def _acc(self, k: str, v: float) -> None:
        self.st[k] = self.st.get(k, 0.0) + v

    @torch.no_grad()
    def build(self, input_ids: torch.Tensor) -> torch.Tensor:
        from .gold_grad_mask import content_fingerprint_from_row

        c = self.cfg
        self.calls += 1
        B, T = input_ids.shape
        dev = input_ids.device
        out = torch.full((B, T), PAD, dtype=torch.int32, device=dev)
        gold_tab = getattr(self.router, "gold", None) if self.router is not None else None
        for b in range(B):
            row = input_ids[b]
            is_eos = row == self.eos
            is_eos[0] = False
            L = int(is_eos.int().argmax()) + 1 if bool(is_eos.any()) else T
            self._acc("rows", 1)
            if self.rng.random() < c.full_frac:
                out[b, :L] = STRIP
                self._acc("full", 1)
                continue
            r = row[:L]
            ds_pos = torch.nonzero(r == self.ds).flatten()
            de_pos = torch.nonzero(r == self.de).flatten()
            nd = int(ds_pos.numel())
            if nd == 0 or nd != int(de_pos.numel()) or bool((de_pos < ds_pos).any()) or (
                    nd > 1 and bool((ds_pos[1:] < de_pos[:-1]).any())):
                out[b, :L] = STRIP  # no / malformed documents: full attention
                self._acc("nodocs", 1)
                continue
            gold = None
            if gold_tab is not None:
                gold = gold_tab.get(content_fingerprint_from_row(r.tolist(), self.eos))
            if gold is None:
                out[b, :L] = STRIP
                self._acc("miss", 1)
                continue
            gset = sorted({int(g) for g in gold if 0 <= int(g) < nd})
            others = [d for d in range(nd) if d not in set(gset)]
            self.rng.shuffle(others)
            k = len(gset) if c.decoys < 0 else c.decoys
            k = min(k, len(others))
            hubs = set(gset) | set(others[:k])
            rest = others[k:]
            cls = [HUB] * nd
            for i, d in enumerate(rest):
                cls[d] = 1 + (i % c.groups)
            # token -> document index: every token from a document's start marker up to the next start marker
            doc_of = torch.cumsum((r == self.ds).to(torch.int64), 0) - 1  # -1 before the first document
            cls_t = torch.tensor([HUB] + cls, dtype=torch.int32, device=dev)  # index 0 = preamble
            g = cls_t[doc_of + 1]
            g[int(de_pos[-1]) + 1:] = STRIP
            out[b, :L] = g
            self._acc("grouped", 1)
            self._acc("hub_docs", len(hubs))
            self._acc("docs", nd)
        if c.log_every and self.calls % c.log_every == 1 and self.rank == 0:
            s = self.st
            print(f"[attn-groups] call {self.calls}: rows {s.get('rows', 0):.0f} grouped {s.get('grouped', 0):.0f} "
                  f"full {s.get('full', 0):.0f} miss {s.get('miss', 0):.0f} nodocs {s.get('nodocs', 0):.0f}; "
                  f"hub docs {s.get('hub_docs', 0):.0f}/{s.get('docs', 0):.0f}, G={c.groups}", flush=True)
        return out


def _strip_runs(pos: torch.Tensor) -> List[Tuple[int, int]]:
    """Contiguous runs [a, e) of a sorted 1-D position tensor."""
    if pos.numel() == 0:
        return []
    p = pos.tolist()
    runs, a, prev = [], p[0], p[0]
    for x in p[1:]:
        if x != prev + 1:
            runs.append((a, prev + 1))
            a = x
        prev = x
    runs.append((a, prev + 1))
    return runs


def group_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gid_k: torch.Tensor, *, q_start: int = 0,
                    scale: Optional[float] = None) -> torch.Tensor:
    """Group-partitioned causal attention. ``q`` (B, Tq, Hq, D) are the key positions ``[q_start, q_start + Tq)``
    of ``k`` / ``v`` (B, Tk, Hk, D), ``Tk = q_start + Tq``; ``gid_k`` (B, Tk) group ids. Returns (B, Tq, Hq, D)."""
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    B, Tq, Hq, D = q.shape
    Tk = k.shape[1]
    if Tk != q_start + Tq:
        raise ValueError(f"group_attention: queries must be the key tail (Tk {Tk} != q_start {q_start} + Tq {Tq})")
    gid_k = gid_k.to(q.device)
    qf = q.reshape(B * Tq, Hq, D)
    kf = k.reshape(B * Tk, k.shape[2], D)
    vf = v.reshape(B * Tk, v.shape[2], D)
    qi: List[torch.Tensor] = []
    ki: List[torch.Tensor] = []
    keep: List[torch.Tensor] = []
    lq: List[int] = []
    lk: List[int] = []
    dests: List[torch.Tensor] = []
    vals: List[torch.Tensor] = []
    for b in range(B):
        gk = gid_k[b]
        gq = gk[q_start:]
        present = torch.unique(gq).tolist()
        hub_k = gk == HUB
        for g in present:
            if g < 0:
                continue
            kmask = hub_k if g == HUB else (hub_k | (gk == g))
            kpos = torch.nonzero(kmask).flatten()
            qpos = kpos[kpos >= q_start]
            qi.append(b * Tq + qpos - q_start)
            ki.append(b * Tk + kpos)
            keep.append(gk[qpos] == g)
            lq.append(int(qpos.numel()))
            lk.append(int(kpos.numel()))
        if STRIP in present:
            for a, e in _strip_runs(torch.nonzero(gq == STRIP).flatten() + q_start):
                o = flash_attn_func(q[b:b + 1, a - q_start:e - q_start], k[b:b + 1, :e], v[b:b + 1, :e],
                                    causal=True, softmax_scale=scale)
                dests.append(b * Tq + torch.arange(a - q_start, e - q_start, device=q.device))
                vals.append(o[0])
    if qi:
        dev = q.device
        cq = torch.tensor([0] + lq, device=dev, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        ck = torch.tensor([0] + lk, device=dev, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        qidx, kidx, kp = torch.cat(qi), torch.cat(ki), torch.cat(keep)
        o = flash_attn_varlen_func(qf[qidx], kf[kidx], vf[kidx], cq, ck, max(lq), max(lk),
                                   causal=True, softmax_scale=scale)
        dests.append(qidx[kp])
        vals.append(o[kp])
    out = qf.new_zeros(qf.shape)
    if dests:
        out = out.index_put((torch.cat(dests),), torch.cat(vals))
    return out.view(B, Tq, Hq, D)


@torch.no_grad()
def group_attention_pairs(gid: torch.Tensor) -> Tuple[float, float]:
    """``(actual, charged)`` (query, key) pairs summed over the rows of ``gid`` (B, T) for one attention layer:
    ``actual`` = what :func:`group_attention` computes over the whole row (``q_start = 0``), ``charged`` = the
    meter's ``L (L + 1) / 2`` per row of ``L`` real (non-PAD) tokens."""
    actual = charged = 0.0
    for b in range(gid.shape[0]):
        g = gid[b]
        L = int((g != PAD).sum())
        charged += L * (L + 1) / 2
        n_hub = int((g == HUB).sum())
        cnt = torch.bincount((g[g > 0]).long()).tolist() if bool((g > 0).any()) else []
        if n_hub:
            actual += n_hub * (n_hub + 1) / 2
        for c in cnt:
            if c:
                n = n_hub + c
                actual += n * (n + 1) / 2
        for a, e in _strip_runs(torch.nonzero(g == STRIP).flatten()):
            nq = e - a
            actual += nq * e - nq * (nq - 1) / 2
    return actual, charged


def _is_softmax_attention(mod) -> bool:
    from . import Attention

    return isinstance(mod, Attention)


def charge_attn_groups(model, gid: torch.Tensor, start: int, sign: float = 1.0) -> float:
    """Add ``sign x`` the FLOP difference (group vs full attention) of every attention layer in blocks
    ``[start, n_layers)`` for rows ``gid`` to ``model._flop_extra_actual``; returns the amount added."""
    blocks = list(model.blocks.values())
    n = len(blocks)
    otr = getattr(model, "_online_token_router", None)
    fac = otr._block_factors(n) if otr is not None else [1.0] * n
    actual, charged = group_attention_pairs(gid)
    tot = 0.0
    for i in range(start, n):
        a = getattr(blocks[i], "attention", None)
        if a is None or not _is_softmax_attention(a):
            continue
        tot += fac[i] * 12.0 * a.n_heads * a.head_dim * (actual - charged)
    tot *= sign
    model._flop_extra_actual = float(getattr(model, "_flop_extra_actual", 0.0)) + tot
    st = getattr(model, "_attn_group_stats", None)
    if st is not None:
        st["delta_flops"] = st.get("delta_flops", 0.0) + tot
    return tot


def group_mask_reference(gid: torch.Tensor) -> torch.Tensor:
    """(B, T, T) boolean allowed-mask equivalent to :func:`group_attention` (for tests): causal and
    hub->hub, group g->{hub, g}, strip->all; PAD rows see only themselves (their output is zeroed anyway)."""
    B, T = gid.shape
    qg = gid[:, :, None]
    kg = gid[:, None, :]
    causal = torch.ones(T, T, dtype=torch.bool, device=gid.device).tril()[None]
    allowed = (qg == STRIP) | ((qg == HUB) & (kg == HUB)) | ((qg > 0) & ((kg == HUB) | (kg == qg)))
    eye = torch.eye(T, dtype=torch.bool, device=gid.device)[None]
    allowed = torch.where(qg == PAD, eye, allowed)
    return allowed & causal
