"""
Per-token FFN skipping for document-body tokens (``--ffn-skip-layers``), training only.

In every listed block, document-body tokens (strictly between a ``doc_start`` and a ``doc_end`` marker) skip the
FFN sub-layer with probability ``p`` (drawn per token, block and forward): their FFN residual update is 0, so they
pass through on the residual stream, while they stay in the sequence for the mixer (attention / GatedDeltaNet).
Query, answer, prompt and marker tokens always run the FFN. The FFN runs on the gathered kept tokens only, so the
matmuls really shrink. Pad tokens skip it too (they are not charged anyway; rows are padded with EOS, so ``pad_id`` = EOS as in the FLOP meter).

- Eval / generation (``model.training == False``) runs every FFN on every token (the dense held-out probe).
- The body mask is computed from ``input_ids`` in a forward pre-hook on the model; a block whose FFN input does not
  have the ids' shape (e.g. above an online-router compaction) runs the FFN dense and is charged dense.
- The draw is seeded by (seed, step, forward counter, layer) and stored per forward, so the activation-checkpoint
  recompute sees the same mask.
- FLOPs: the holder is installed as ``model._ffn_width`` and duck-types :class:`~olmo_core.nn.ffn_width.FFNWidthHolder`
  (``credits_per_block``): a listed block's FFN is credited ``-(1 - r) x ffn_fpt`` per token, with ``r`` the realised
  share of real (non-pad) tokens that ran it. The online router multiplies that by the block's own factor (2/3 for a
  frozen-base LoRA block) per document; without the router the FLOP meter applies it per real token, scaled by the
  same frozen-base factor (``lora_factor``). The LoRA adapters on skipped tokens are still charged (conservative).
"""

from __future__ import annotations

import logging
from types import MethodType
from typing import Any, Dict, List, Optional, Sequence

import torch

log = logging.getLogger(__name__)

__all__ = ["FFNTokenSkipHolder", "install_ffn_token_skip", "parse_skip_spec"]


def parse_skip_spec(spec: str, n_layers: int) -> Dict[int, float]:
    """``"0-6:0.85,7-35:0.8"`` -> {layer: p}; a group without ``:p`` uses p = 1.0."""
    out: Dict[int, float] = {}
    for grp in [g.strip() for g in spec.split(",") if g.strip()]:
        rng, _, p = grp.partition(":")
        pv = float(p) if p else 1.0
        if not 0.0 < pv <= 1.0:
            raise ValueError(f"ffn-skip p must be in (0, 1], got {pv}")
        if "-" in rng:
            lo, hi = (int(x) for x in rng.split("-"))
            ids = range(lo, hi + 1)
        else:
            ids = [int(rng)]
        for i in ids:
            if not 0 <= i < n_layers:
                raise ValueError(f"ffn-skip layer {i} out of range [0, {n_layers})")
            out[i] = pv
    return out


class FFNTokenSkipHolder:
    token_skip = True

    def __init__(self, probs: Dict[int, float], *, doc_start_id: int, doc_end_id: int, pad_id: Optional[int],
                 seed: int = 0, skip_all_tokens: bool = False):
        self.probs = dict(probs)
        self.layers = set(probs)
        self.doc_start_id, self.doc_end_id, self.pad_id = int(doc_start_id), int(doc_end_id), pad_id
        self.seed = int(seed)
        self.skip_all_tokens = bool(skip_all_tokens)
        self.step = 0
        self._fwd = 0
        self.masks: Dict[int, torch.Tensor] = {}  # layer -> (B, T) bool RUN mask of the current forward
        self._real: Optional[torch.Tensor] = None
        self.shape: Optional[torch.Size] = None
        self.last: Dict[int, float] = {}  # layer -> realised run share (real tokens) of the last forward
        self._acc: Dict[int, List[float]] = {}  # layer -> [ran, real] summed over this step's forwards (local rank)
        self.ffn_fpt: Dict[int, float] = {}
        self.lora_factor: Dict[int, float] = {}  # set by the meter path: per-block frozen-base share (no router)

    def set_step(self, step: int) -> None:
        if int(step) != self.step:
            self.step = int(step)
            self._acc = {}

    def begin_forward(self, input_ids: torch.Tensor) -> None:
        """Draw this forward's run masks (training forwards only)."""
        self._fwd += 1
        ids = input_ids
        dev = ids.device
        is_s, is_e = ids == self.doc_start_id, ids == self.doc_end_id
        depth = torch.cumsum(is_s.to(torch.int32), dim=-1) - torch.cumsum(is_e.to(torch.int32), dim=-1)
        body = (depth > 0) & ~is_s
        real = torch.ones_like(body) if self.pad_id is None else ids != self.pad_id
        if self.skip_all_tokens:
            body = real.clone()
        cand = body | ~real
        g = torch.Generator(device=dev)
        self.masks = {}
        nreal = float(real.sum())
        for i in sorted(self.probs):
            g.manual_seed((self.seed * 1_000_003 + self.step * 7919 + self._fwd * 131 + i * 104_729) % (2**62))
            skip = cand & (torch.rand(ids.shape, generator=g, device=dev) < self.probs[i])
            run = ~skip
            self.masks[i] = run
            ran = float((run & real).sum())
            self.last[i] = ran / max(1.0, nreal)
            a = self._acc.setdefault(i, [0.0, 0.0])
            a[0] += ran
            a[1] += nreal
        self.shape = ids.shape

    def frac_of(self, layer: int, step_avg: bool = False) -> float:
        if layer not in self.layers:
            return 1.0
        if step_avg and layer in self._acc and self._acc[layer][1] > 0:
            return self._acc[layer][0] / self._acc[layer][1]
        return self.last.get(layer, 1.0)

    def credits_per_block(self, n_blocks: int, step_avg: bool = False) -> List[float]:
        """Per-block FLOP credit per token (<= 0): -(1 - r) x the FFN's cost (x ``lora_factor`` when set)."""
        return [-(1.0 - self.frac_of(i, step_avg)) * self.ffn_fpt.get(i, 0.0) * self.lora_factor.get(i, 1.0)
                for i in range(n_blocks)]


def _skip_ffn_forward(self, x: torch.Tensor) -> torch.Tensor:
    st = self._ffnskip
    holder: FFNTokenSkipHolder = st["holder"]
    run = holder.masks.get(st["layer"]) if self.training else None
    if run is None or tuple(x.shape[:-1]) != tuple(run.shape):
        if run is not None:
            holder.last[st["layer"]] = 1.0  # ran dense: charged dense
        return st["orig"](x)
    D = x.shape[-1]
    xf = x.reshape(-1, D)
    idx = run.reshape(-1).nonzero(as_tuple=True)[0]
    y = st["orig"](xf.index_select(0, idx))
    out = xf.new_zeros(xf.shape[0], y.shape[-1]).index_copy(0, idx, y.to(xf.dtype))
    return out.reshape(*x.shape[:-1], y.shape[-1])


def install_ffn_token_skip(model: Any, spec: str, *, doc_start_id: int, doc_end_id: int, pad_id: Optional[int],
                           seed: int = 0, skip_all_tokens: bool = False) -> FFNTokenSkipHolder:
    if getattr(model, "_ffn_width", None) is not None:
        raise RuntimeError("--ffn-skip-layers and --ffn-width-frac both price the FFN credit: use one")
    blocks = list(model.blocks.values())
    probs = parse_skip_spec(spec, len(blocks))
    holder = FFNTokenSkipHolder(probs, doc_start_id=doc_start_id, doc_end_id=doc_end_id, pad_id=pad_id, seed=seed,
                                skip_all_tokens=skip_all_tokens)
    for i in sorted(probs):
        ff = getattr(blocks[i], "feed_forward", None)
        if ff is None:
            raise RuntimeError(f"ffn skip: block {i} has no feed_forward")
        ff._ffnskip = {"holder": holder, "layer": i, "orig": ff.forward}
        ff.forward = MethodType(_skip_ffn_forward, ff)
        holder.ffn_fpt[i] = float(ff.num_flops_per_token(0))

    def _pre(mod, args, kwargs):
        if not mod.training:
            holder.masks = {}
            return None
        ids = args[0] if args else kwargs.get("input_ids")
        if ids is None or ids.is_floating_point():
            holder.masks = {}
            return None
        holder.begin_forward(ids)
        return None

    model.register_forward_pre_hook(_pre, with_kwargs=True)
    model._ffn_width = holder
    # the FFN share of each block's per-token cost, for the report
    for L in (4096, 8192, 16384):
        parts = []
        for i, b in enumerate(blocks):
            tot = float(b.num_flops_per_token(L))
            ffn = float(b.feed_forward.num_flops_per_token(L)) if getattr(b, "feed_forward", None) is not None else 0
            parts.append((i, type(getattr(b, "attention", None)).__name__, ffn / max(1.0, tot)))
        kinds: Dict[str, List[float]] = {}
        for _, k, s in parts:
            kinds.setdefault(k, []).append(s)
        log.info("[ffn-skip] FFN share of block FLOPs at L=%d: %s", L,
                 {k: round(sum(v) / len(v), 4) for k, v in kinds.items()})
        print(f"[ffn-skip] FFN share of block FLOPs at L={L}: "
              f"{ {k: round(sum(v) / len(v), 4) for k, v in kinds.items()} }", flush=True)
    log.info("[ffn-skip] body tokens skip the FFN: %s", probs)
    return holder
