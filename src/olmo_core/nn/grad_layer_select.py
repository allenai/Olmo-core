"""
Gradient-ranked layer selection (2026-10-28): each optimizer step, keep only the ``keep_frac`` share of blocks with the
most gradient signal.

**Ranking.** ``online``: an EMA of each block's parameter-gradient energy (sum of ``||g||^2`` over its TRAINABLE
params -- for a LoRA run, the adapters), updated from the steps where the block was selected; a block never measured
ranks above every measured one (so the first ``ceil(n / k)`` steps measure every block once), and ``explore_frac`` of
the slots go to random unselected blocks so stale estimates get re-measured. ``static``: a fixed per-block energy
vector (e.g. the grad-composition study's), top-k every step, no exploration (the control). The draw is a pure
function of ``(seed, step)`` and of the EMA, which is built from all-reduced energies, so every rank selects the same
blocks. ``always``: blocks that count as selected every step (e.g. the online router's drop layers).

**Modes.**

* ``update`` (update-only): every block runs forward; only the selected blocks' trainable params have
  ``requires_grad`` this step (FSDP2 copies the flag from the sharded param on every unshard), and the hidden state
  is cut (detached) at the input of the lowest selected block -- capped at ``max_cut`` -- so no backward runs below
  it. Blocks above the cut that are not selected still back-propagate the activation gradient. No train/eval
  mismatch. Charged: selected 1, unselected at or above the cut 2/3, below the cut 1/3 of the block's
  forward + backward.
* ``skip`` (forward-skip): only the selected blocks run at all; the rest are the identity (the
  :mod:`olmo_core.nn.layer_drop` path: the selector is installed as ``model._layer_drop``). Charged: 0 for a skipped
  block.

:class:`~olmo_core.train.callbacks.GradLayerSelectCallback` selects before each step, toggles ``requires_grad``,
measures the energies after backward (``pre_optim_step``) and logs the selection histogram.
"""

import math
import random
from typing import Dict, FrozenSet, List, Optional, Sequence

__all__ = ["GradLayerSelector", "MODES", "RANKINGS"]

MODES = ("update", "skip")
RANKINGS = ("online", "static")


class GradLayerSelector:
    #: layer_drop interface (``Transformer._run_blocks`` / the flop meter / the online router read these)
    rescale = False
    probs: Dict[int, float] = {}

    def __init__(
        self,
        n_layers: int,
        keep_frac: float,
        mode: str = "update",
        ranking: str = "online",
        explore_frac: float = 0.1,
        ema: float = 0.9,
        static_energy: Optional[Sequence[float]] = None,
        always: Sequence[int] = (),
        max_cut: Optional[int] = None,
        cut_input_grad: bool = False,
        seed: int = 0,
    ):
        if not 0.0 < keep_frac <= 1.0:
            raise ValueError(f"grad-select keep_frac must be in (0, 1], got {keep_frac}")
        if mode not in MODES:
            raise ValueError(f"grad-select mode must be one of {MODES}, got {mode!r}")
        if ranking not in RANKINGS:
            raise ValueError(f"grad-select ranking must be one of {RANKINGS}, got {ranking!r}")
        if not 0.0 <= explore_frac < 1.0 or not 0.0 <= ema < 1.0:
            raise ValueError("grad-select explore_frac and ema must be in [0, 1)")
        if ranking == "static" and (static_energy is None or len(static_energy) != n_layers):
            raise ValueError(f"grad-select static ranking needs {n_layers} energies, got {static_energy!r}")
        bad = [i for i in always if not 0 <= i < n_layers]
        if bad:
            raise ValueError(f"grad-select always-selected blocks {bad} out of range")
        self.n_layers = int(n_layers)
        self.k = max(1, int(round(keep_frac * n_layers)), len(set(always)))
        self.keep_frac = keep_frac
        self.mode = mode
        self.ranking = ranking
        self.explore_frac = explore_frac
        self.ema_decay = ema
        self.static_energy = list(static_energy) if static_energy is not None else None
        self.always = frozenset(int(i) for i in always)
        self.max_cut = max_cut
        self.cut_input_grad = bool(cut_input_grad)
        self.seed = int(seed)
        self.fixed: FrozenSet[int] = frozenset()  # layer_drop interface (no fixed skips)
        self.ema: List[Optional[float]] = [None] * self.n_layers
        self.counts = [0] * self.n_layers
        self.history: List[List[int]] = []
        self.step: Optional[int] = None
        self._sel: Optional[FrozenSet[int]] = None
        self._sel_step: Optional[int] = None

    # -- selection ------------------------------------------------------------------------------
    def n_explore(self) -> int:
        free = self.k - len(self.always)
        if self.ranking != "online" or self.explore_frac <= 0 or free <= 1:
            return 0
        return min(free - 1, max(1, int(round(self.explore_frac * self.k))))

    def select_for_step(self, step: Optional[int]) -> FrozenSet[int]:
        """The selected set of optimizer step ``step`` (``None`` = every block: the dry-run batch)."""
        if step is None:
            return frozenset(range(self.n_layers))
        sel = set(self.always)
        free = [i for i in range(self.n_layers) if i not in sel]
        n_exp = self.n_explore()
        n_top = self.k - len(sel) - n_exp
        if self.ranking == "static":
            score = {i: float(self.static_energy[i]) for i in free}  # type: ignore[index]
        else:
            # never-measured blocks first (+inf), then by EMA energy; ties broken by index
            score = {i: (math.inf if self.ema[i] is None else self.ema[i]) for i in free}
        ranked = sorted(free, key=lambda i: (-score[i], i))
        sel |= set(ranked[:n_top])
        if n_exp:
            rng = random.Random(f"gradsel:{self.seed}:{int(step)}")
            rest = [i for i in free if i not in sel]
            sel |= set(rng.sample(rest, min(n_exp, len(rest))))
        return frozenset(sel)

    def selected(self) -> FrozenSet[int]:
        if self._sel is None or self._sel_step != self.step:
            self._sel = self.select_for_step(self.step)
            self._sel_step = self.step
        return self._sel

    def begin_step(self, step: int) -> FrozenSet[int]:
        self.step = int(step)
        sel = self.selected()
        for i in sel:
            self.counts[i] += 1
        self.history.append(sorted(sel))
        return sel

    # -- energies -------------------------------------------------------------------------------
    def update_energies(self, energy: Dict[int, float]) -> None:
        """EMA update from this step's per-block gradient energies (only blocks that got a gradient)."""
        for i, e in energy.items():
            e = float(e)
            if not math.isfinite(e):
                continue
            self.ema[i] = e if self.ema[i] is None else self.ema_decay * self.ema[i] + (1.0 - self.ema_decay) * e

    # -- layer_drop interface (skip mode) -------------------------------------------------------
    def skipped(self) -> FrozenSet[int]:
        if self.mode != "skip" or self.step is None:
            return frozenset()
        return frozenset(range(self.n_layers)) - self.selected()

    def run_scale(self, block_idx: int) -> float:
        return 1.0

    def expected_run_frac(self) -> float:
        return self.k / self.n_layers if self.mode == "skip" else 1.0

    # -- update mode ----------------------------------------------------------------------------
    def cut(self) -> int:
        """Update mode: the block whose INPUT is detached (no backward below it); 0 = no cut."""
        if self.mode != "update" or self.step is None:
            return 0
        c = min(self.selected())
        if self.max_cut is not None:
            c = min(c, int(self.max_cut))
        return max(0, c)

    def block_factors(self) -> Dict[int, float]:
        """Share of each block's forward + backward FLOPs this step pays (blocks not listed pay 1)."""
        if self.step is None:
            return {}
        sel = self.selected()
        if self.mode == "skip":
            return {i: 0.0 for i in range(self.n_layers) if i not in sel}
        c = self.cut()
        return {i: (1.0 / 3.0 if i < c else 2.0 / 3.0) for i in range(self.n_layers) if i not in sel}

    def train_mask(self) -> Dict[int, bool]:
        """Update mode: which blocks' trainable params get a gradient this step."""
        sel = self.selected()
        return {i: (i in sel) for i in range(self.n_layers)}

    def summary(self) -> Dict[str, object]:
        steps = max(1, len(self.history))
        return {
            "mode": self.mode,
            "ranking": self.ranking,
            "keep_frac": self.keep_frac,
            "k": self.k,
            "explore": self.n_explore(),
            "always": sorted(self.always),
            "max_cut": self.max_cut,
            "steps": len(self.history),
            "selected_frac": [c / steps for c in self.counts],
            "ema": self.ema,
            "history": self.history,
        }
