"""
Selective backprop (2026-11-02, ``tokdrop_lr`` cross-row allocation): run the forward on every row, then
back-propagate only the micro-batches whose answer loss is high.

Late in token-drop training most rows are already solved (at 7,808 rows/task the median row answer CE is 0.001 and the
top 10% of rows hold 81% of the loss), so their backward buys almost nothing. The forward is still needed to know
which rows those are; the backward (with a frozen LoRA base: the activation gradient, about one forward) is what is
saved.

The decision unit is one micro-batch INDEX across all ranks: under FSDP every backward all-gathers parameters, so all
ranks must run (or skip) the same backward. Its score is the summed answer CE of that micro-batch over ranks (one
scalar all-reduce). With micro-batch 1 on one GPU the unit is a single row.

Modes:

- ``top``: back-propagate a unit iff its score is at or above the ``1 - q`` quantile of the last ``window`` unit
  scores (a running distribution, so the cut adapts as the loss falls). Biased: it assumes skipped rows contribute
  ~0 gradient.
- ``sample``: back-propagate with probability ``p = clip(q * score / mean(window), min_p, 1)`` and weight ``1 / p``
  (unbiased for the summed loss's gradient; the draw is seeded identically on every rank).

The first ``warmup_steps`` steps back-propagate everything. If a step reaches its last unit without having run any
backward, that unit runs (an optimizer step needs a gradient). A skipped unit is charged forward-only by the FLOP meter
via the model's ``_flop_fo_delta`` (set by the mid-stack online router, which knows the per-block lengths).
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Deque, Dict, Tuple

import torch

__all__ = ["SelectiveBackpropConfig", "SelectiveBackprop"]


@dataclass
class SelectiveBackpropConfig:
    mode: str = "top"
    """``top`` (deterministic running-quantile cut) or ``sample`` (loss-proportional, 1/p reweighted)."""
    q: float = 0.3
    """Target share of units back-propagated."""
    window: int = 32
    """Units in the running score distribution."""
    min_p: float = 0.05
    """``sample``: floor on the backprop probability (bounds the 1/p weight)."""
    warmup_steps: int = 1
    """Leading optimizer steps that back-propagate every unit."""
    seed: int = 0

    def validate(self) -> None:
        if self.mode not in ("top", "sample"):
            raise ValueError(f"selective backprop mode must be top / sample, got {self.mode!r}")
        if not 0.0 < self.q <= 1.0 or self.window < 2 or not 0.0 < self.min_p <= 1.0 or self.warmup_steps < 0:
            raise ValueError("selective backprop needs 0 < q <= 1, window >= 2, 0 < min_p <= 1, warmup_steps >= 0")


class SelectiveBackprop:
    def __init__(self, cfg: SelectiveBackpropConfig):
        cfg.validate()
        self.cfg = cfg
        self.hist: Deque[float] = collections.deque(maxlen=int(cfg.window))
        self.step = 0
        self._ran = 0
        self._units = 0
        self._forced = 0
        self._weight_sum = 0.0
        self.total_units = 0
        self.total_ran = 0
        self.last_thr = float("nan")

    def begin_batch(self) -> None:
        """Call once per optimizer step (not on the trainer's dry-run batch)."""
        self.step += 1
        self._ran = self._units = self._forced = 0
        self._weight_sum = 0.0

    @staticmethod
    def unit_score(ce_sum_local: torch.Tensor) -> float:
        """Sum of this micro-batch's answer CE over ranks (identical on every rank)."""
        import torch.distributed as dist

        v = ce_sum_local.detach().to(torch.float64).reshape(1).clone()
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(v)
        return float(v.item())

    def decide(self, score: float, mb_idx: int, n_mb: int) -> Tuple[bool, float]:
        """(back-propagate this unit?, loss weight). Identical on every rank given an identical ``score``."""
        c = self.cfg
        self._units += 1
        run, w = True, 1.0
        if self.step > c.warmup_steps and len(self.hist) >= 2 and math.isfinite(score):
            h = torch.tensor(list(self.hist), dtype=torch.float64)
            if c.mode == "top":
                thr = float(torch.quantile(h, 1.0 - c.q))
                self.last_thr = thr
                run = score >= thr
            else:
                mean = float(h.mean())
                p = 1.0 if mean <= 0 else min(1.0, max(c.min_p, c.q * score / mean))
                g = torch.Generator().manual_seed((c.seed * 1_000_003 + self.step * 7919 + mb_idx * 104_729) % (2**62))
                run = bool(torch.rand((), generator=g) < p)
                w = 1.0 / p
                self.last_thr = p
        if not run and mb_idx == n_mb - 1 and self._ran == 0:
            run, w = True, 1.0  # an optimizer step needs a gradient
            self._forced += 1
        if math.isfinite(score):
            self.hist.append(score)
        if run:
            self._ran += 1
            self._weight_sum += w
        self.total_units += 1
        self.total_ran += int(run)
        return run, (w if run else 0.0)

    def metrics(self) -> Dict[str, float]:
        return {
            "sbp/backprop_frac": self._ran / max(1, self._units),
            "sbp/backprop_frac_cum": self.total_ran / max(1, self.total_units),
            "sbp/forced": float(self._forced),
            "sbp/mean_weight": self._weight_sum / max(1, self._ran),
            "sbp/last_thr": self.last_thr if math.isfinite(self.last_thr) else -1.0,
        }
