"""
Train-time whole-block skipping: a FIXED set of blocks, and/or random LayerDrop (stochastic depth).

Unlike the learned per-token router in :mod:`olmo_core.nn.block_skip`, a skipped block here is
skipped for the WHOLE micro-batch: ``Transformer._run_blocks`` does not call its forward, so the
residual stream passes through unchanged and the GPU does no work for it (no FSDP all-gather, no
forward, no backward, no reduce-scatter). That is what makes it a wall-clock lever, not just a
FLOP-meter one. GatedDeltaNet blocks of a hybrid stack are skipped whole the same way.

Training only: eval / generation forwards (``model.training == False``) always run every block.

The random draw depends only on ``(seed, step)``, so it is identical on every rank (FSDP2's
collectives stay matched: every rank skips the same blocks) and constant across the micro-batches
of one optimizer step. :class:`~olmo_core.train.callbacks.LayerDropCallback` sets the step before
each step's forwards. Until a step is set (the trainer's dry-run batch), only the fixed set is
skipped. A skipped block's parameters get no gradient that step (``grad is None``), which the
AdamW step and the grad-norm clip already skip.

Schedules for the per-block probability ``p_i`` of an eligible block ``i`` (of ``L`` blocks):

* ``uniform``: ``p_i = p``.
* ``linear``: ``p_i = p * (i + 1) / L`` (stochastic depth: deeper blocks are dropped more often).

``rescale``: a block that RUNS has its residual update scaled by ``1 / (1 - p_i)``
(``h + (block(h) - h) / (1 - p_i)``), so the expected update over the drop equals the full
update, as in inverted dropout. Off by default: fine-tuning a pretrained stack, the unscaled
update is what eval sees.
"""

import random
import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, Optional, Sequence

__all__ = ["LayerDropPlan", "flops_per_token_ran", "parse_layer_list"]

SCHEDULES = ("uniform", "linear")


def parse_layer_list(spec: Optional[str], n_layers: Optional[int] = None) -> FrozenSet[int]:
    """Parse ``"10,14,18"`` (also ``"10-12"`` ranges, negative = from the end) into block indices.

    :raises ValueError: An index outside ``[0, n_layers)`` when ``n_layers`` is given.
    """
    out = set()
    for part in (spec or "").replace(" ", "").split(","):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", part)
        if m:
            idx = range(int(m.group(1)), int(m.group(2)) + 1)
        else:
            idx = [int(part)]
        for i in idx:
            if i < 0 and n_layers is not None:
                i += n_layers
            if n_layers is not None and not 0 <= i < n_layers:
                raise ValueError(f"layer index {i} out of range for {n_layers} blocks ({spec!r})")
            out.add(i)
    return frozenset(out)


@dataclass
class LayerDropPlan:
    """Which blocks a training forward skips; see the module docstring."""

    n_layers: int
    fixed: FrozenSet[int] = frozenset()
    p: float = 0.0
    schedule: str = "uniform"
    exclude_first: int = 0
    exclude_last: int = 0
    rescale: bool = False
    seed: int = 0
    step: Optional[int] = None
    """The optimizer step whose draw the next training forwards use (set by the callback)."""
    probs: Dict[int, float] = field(default_factory=dict, init=False)
    _cache_step: Optional[int] = field(default=None, init=False, repr=False)
    _cache: FrozenSet[int] = field(default=frozenset(), init=False, repr=False)

    def __post_init__(self):
        if not 0.0 <= self.p < 1.0:
            raise ValueError(f"layer-drop p must be in [0, 1), got {self.p}")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"layer-drop schedule must be one of {SCHEDULES}, got {self.schedule!r}")
        if self.exclude_first < 0 or self.exclude_last < 0:
            raise ValueError("layer-drop exclude_first/exclude_last must be >= 0")
        bad = sorted(i for i in self.fixed if not 0 <= i < self.n_layers)
        if bad:
            raise ValueError(f"fixed skip layers {bad} out of range for {self.n_layers} blocks")
        self.fixed = frozenset(self.fixed)
        if len(self.fixed) >= self.n_layers:
            raise ValueError("cannot skip every block")
        self.probs = {}
        if self.p > 0:
            for i in range(self.exclude_first, self.n_layers - self.exclude_last):
                if i in self.fixed:
                    continue
                pi = self.p if self.schedule == "uniform" else self.p * (i + 1) / self.n_layers
                if pi > 0:
                    self.probs[i] = pi
            if not self.probs:
                raise ValueError("layer-drop p > 0 but no block is eligible (check exclude_first/last)")

    @property
    def active(self) -> bool:
        return bool(self.fixed) or bool(self.probs)

    def skipped_for_step(self, step: Optional[int]) -> FrozenSet[int]:
        """The skip set of optimizer step ``step``: the fixed set plus that step's random draw
        (a pure function of ``(seed, step)``; ``None`` = fixed set only)."""
        if step is None or not self.probs:
            return self.fixed
        rng = random.Random(f"layerdrop:{self.seed}:{int(step)}")
        drawn = {i for i in sorted(self.probs) if rng.random() < self.probs[i]}
        return self.fixed | frozenset(drawn)

    def skipped(self) -> FrozenSet[int]:
        """The skip set of the current step (cached per step)."""
        if self._cache_step != self.step or self.step is None:
            self._cache = self.skipped_for_step(self.step)
            self._cache_step = self.step
        return self._cache

    def run_scale(self, block_idx: int) -> float:
        """Residual-update multiplier of a block that RUNS (1.0 unless ``rescale``)."""
        if not self.rescale:
            return 1.0
        pi = self.probs.get(block_idx, 0.0)
        return 1.0 / (1.0 - pi) if pi > 0 else 1.0

    def expected_run_frac(self) -> float:
        """Expected fraction of blocks that run per step."""
        return 1.0 - (len(self.fixed) + sum(self.probs.values())) / self.n_layers


def flops_per_token_ran(
    model, seq_len: int, skipped: Iterable[int], factors: Optional[Dict[int, float]] = None
) -> float:
    """``model.num_flops_per_token(seq_len)`` minus the blocks in ``skipped`` (they ran neither
    forward nor backward), i.e. the per-token training FLOPs of a forward that skipped them.
    ``factors``: share of a block's forward + backward that ran (grad layer select: 2/3 = no weight
    gradient, 1/3 = forward only); blocks not listed pay 1."""
    full = int(model.num_flops_per_token(int(seq_len)))
    skipped = list(skipped)
    if not skipped and not factors:
        return full
    blocks: Sequence = list(model.blocks.values())
    out = float(full) - sum(int(blocks[i].num_flops_per_token(int(seq_len))) for i in skipped)
    for i, f in (factors or {}).items():
        if i not in skipped:
            out -= (1.0 - float(f)) * int(blocks[i].num_flops_per_token(int(seq_len)))
    return out
