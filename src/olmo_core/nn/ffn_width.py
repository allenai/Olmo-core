"""
Partial-MLP training (``--ffn-width-frac f``): each optimizer step and FFN layer computes a subset of ``f * d_ff``
neurons for ALL tokens -- the gate / up rows and the down columns of those neurons are gathered, so the matmuls and the
``T x d_ff`` intermediate really shrink -- and eval runs the full width.

- The subset is drawn from (seed, global step, layer) on the host, so it is identical on every rank, constant across a
  step's micro-batches and identical in the activation-checkpoint recompute (the step only changes in ``pre_step``).
- ``random`` draws ``k = round(f * d_ff)`` neurons uniformly; ``blocked`` draws ``round(f * n_blocks)`` contiguous
  blocks of ``block`` neurons (kernel-friendlier). The realised fraction ``k / d_ff`` is what is charged and rescaled.
- The FFN output (before the down bias) is scaled by ``1 / f`` (dropout-style), so the full-width eval matches the
  train-time FFN in expectation; ``rescale=False`` is the ablation.
- FLOPs: the FFN of a partial layer is charged at ``f``. The online router adds the credit in its own per-document
  accounting (it knows each block's length); without a router the FLOP meter applies it per real token.

Only the SwiGLU :class:`~olmo_core.nn.feed_forward.FeedForward` (``w1`` / ``w3`` in, ``w2`` out) is supported.
"""

from __future__ import annotations

import logging
from types import MethodType
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

__all__ = ["FFNWidthHolder", "install_ffn_width", "parse_layer_range"]


def parse_layer_range(spec: str, n_layers: int) -> List[int]:
    """``""`` = every block; ``"a-b"`` (inclusive) / ``"a,b,c"`` / mixes."""
    if not spec.strip():
        return list(range(n_layers))
    out = set()
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-"))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    bad = [i for i in out if not 0 <= i < n_layers]
    if bad:
        raise ValueError(f"ffn-width layers {bad} out of range [0, {n_layers})")
    return sorted(out)


class FFNWidthHolder:
    def __init__(self, frac: float, layers: Sequence[int], *, mode: str = "random", rescale: bool = True,
                 block: int = 128, seed: int = 0):
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"ffn width frac must be in (0, 1], got {frac}")
        if mode not in ("random", "blocked"):
            raise ValueError(f"ffn width mode must be random / blocked, got {mode!r}")
        self.frac, self.layers, self.mode, self.rescale = float(frac), set(layers), mode, bool(rescale)
        self.block, self.seed = int(block), int(seed)
        self.step = 0
        self._cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self.realised: Dict[int, float] = {}  # layer -> k / d_ff
        self.ffn_fpt: Dict[int, float] = {}  # layer -> full-width FFN FLOPs per token (forward + backward)

    def set_step(self, step: int) -> None:
        if int(step) != self.step:
            self.step = int(step)
            self._cache.clear()

    def indices(self, layer: int, hidden: int, device) -> torch.Tensor:
        key = (self.step, layer)
        idx = self._cache.get(key)
        if idx is None:
            g = torch.Generator().manual_seed((self.seed * 1_000_003 + self.step * 7919 + layer * 104_729) % (2**62))
            if self.mode == "random":
                k = max(1, int(round(self.frac * hidden)))
                sel = torch.randperm(hidden, generator=g)[:k]
            else:
                nb = (hidden + self.block - 1) // self.block
                kb = max(1, int(round(self.frac * nb)))
                blocks = torch.randperm(nb, generator=g)[:kb]
                sel = torch.cat([torch.arange(b * self.block, min((b + 1) * self.block, hidden)) for b in blocks.tolist()])
            idx = torch.sort(sel).values.to(device)
            self._cache[key] = idx
            self.realised[layer] = idx.numel() / hidden
        return idx

    def frac_of(self, layer: int) -> float:
        """Realised fraction of this layer's FFN (1.0 for layers not in the set; nominal before the first draw)."""
        if layer not in self.layers:
            return 1.0
        return self.realised.get(layer, self.frac)

    def credits_per_block(self, n_blocks: int) -> List[float]:
        """Per-block FLOP credit per token (<= 0): -(1 - f) x the FFN's full-width cost."""
        return [-(1.0 - self.frac_of(i)) * self.ffn_fpt.get(i, 0.0) for i in range(n_blocks)]


def _lin(mod: nn.Linear, x: torch.Tensor, rows: Optional[torch.Tensor] = None,
         cols: Optional[torch.Tensor] = None, add_bias: bool = True) -> torch.Tensor:
    """``mod(x)`` restricted to output ``rows`` / input ``cols`` -- including an early-LoRA adapter on ``mod``
    (:mod:`olmo_core.nn.early_lora`), which a plain ``F.linear`` on the gathered weight would silently skip."""
    w = mod.weight
    if rows is not None:
        w = w[rows]
    if cols is not None:
        w = w[:, cols]
    b = None
    if add_bias and mod.bias is not None:
        b = mod.bias[rows] if rows is not None else mod.bias
    y = F.linear(x, w, b)
    if hasattr(mod, "lora_A"):
        A = mod.lora_A if cols is None else mod.lora_A[:, cols]  # type: ignore[index]
        B = mod.lora_B if rows is None else mod.lora_B[rows]  # type: ignore[index]
        y = y + mod._lora_scale * F.linear(F.linear(x, A), B)  # type: ignore[attr-defined]
        mod._lora_calls += 1  # type: ignore[attr-defined]
    return y


def _partial_ffn_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    st = self._ffnw  # type: ignore[attr-defined]
    holder: FFNWidthHolder = st["holder"]
    if not self.training or holder.frac >= 1.0:
        return st["orig"](x)
    idx = holder.indices(st["layer"], self.hidden_size, x.device)  # type: ignore[attr-defined]
    w1, w3, w2 = self.w1, self.w3, self.w2  # type: ignore[attr-defined]
    a = _lin(w1, x, rows=idx)
    b = _lin(w3, x, rows=idx)
    z = self.activation_fn(a) * b  # type: ignore[attr-defined]
    out = _lin(w2, z, cols=idx, add_bias=False)
    if holder.rescale:
        out = out * (self.hidden_size / idx.numel())  # type: ignore[attr-defined]
    if w2.bias is not None:
        out = out + w2.bias
    return out


def install_ffn_width(model: Any, frac: float, layers: str = "", *, mode: str = "random", rescale: bool = True,
                      block: int = 128, seed: int = 0) -> FFNWidthHolder:
    """Patch the FFN of every listed block; ``model._ffn_width`` keeps the holder (the callback sets its step)."""
    blocks = list(model.blocks.values())
    layer_ids = parse_layer_range(layers, len(blocks))
    holder = FFNWidthHolder(frac, layer_ids, mode=mode, rescale=rescale, block=block, seed=seed)
    for i in layer_ids:
        ff = getattr(blocks[i], "feed_forward", None)
        if ff is None or not all(hasattr(ff, n) for n in ("w1", "w2", "w3", "activation_fn", "hidden_size")):
            raise RuntimeError(f"ffn width: block {i} has no SwiGLU FeedForward (w1 / w2 / w3)")
        ff._ffnw = {"holder": holder, "layer": i, "orig": ff.forward}
        ff.forward = MethodType(_partial_ffn_forward, ff)
        holder.ffn_fpt[i] = float(ff.num_flops_per_token(0))
    model._ffn_width = holder
    log.info("[ffn-width] frac %.3f (%s, rescale=%s) on %d blocks", frac, mode, rescale, len(layer_ids))
    return holder
