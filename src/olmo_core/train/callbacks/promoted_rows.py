"""Seed the rows of tokens promoted into reserved vocabulary slots before the first step."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Tuple

import torch

from .callback import Callback

log = logging.getLogger(__name__)

__all__ = ["PromotedTokenRowsCallback", "OLMO35_THINK_ROWS"]

OLMO35_THINK_ROWS: List[Tuple[int, List[int]]] = [(100256, [14023, 771, 29]), (100266, [524, 27963, 29])]
"""``<think>`` and ``</think>`` promoted into ``<|extra_id_0|>`` / ``<|extra_id_1|>`` of
``allenai/dolma2-tokenizer-olmo35@8b97170`` by open-instruct ``--reserved_slot_tokens``, with the
ids of the pieces each used to tokenize into (``<th`` ``ink`` ``>``, ``</`` ``think`` ``>``)."""


@dataclass
class PromotedTokenRowsCallback(Callback):
    """Seed promoted-token rows once, at the start of a fresh run, never on resume.

    A run is fresh when the trainer is at step 0 after its checkpoint load: the base weights
    came from the load path (model only) and no step has trained the rows. A resumed run is past
    step 0 and keeps its trained rows. Seeding again at step 0 (a step-0 checkpoint) is a no-op,
    because the source rows have not trained either.

    After step 1 each seeded row must be strictly closer to its seeded value than to its
    pre-seed value, so an optimizer that overwrote the seed from stale masters is caught (as
    open-instruct's ``PromotedRowStepCheck``).
    """

    priority: ClassVar[int] = 3
    """After the checkpoint load, before evaluators and pre-training checkpoint saves."""

    rows: List[Tuple[int, List[int]]] = field(default_factory=lambda: list(OLMO35_THINK_ROWS))

    _before: Optional[List[torch.Tensor]] = None
    _seeded: Optional[List[torch.Tensor]] = None
    _checked: bool = False

    def _matrices(self) -> List[torch.Tensor]:
        lm = self.trainer.train_module.multimodal_model.lm  # type: ignore[attr-defined]
        matrices = [lm.embeddings.weight]
        if lm.lm_head.w_out.weight is not lm.embeddings.weight:
            matrices.append(lm.lm_head.w_out.weight)
        return matrices

    def _snapshot(self) -> List[torch.Tensor]:
        ids = [target for target, _ in self.rows]
        return [w.data[ids].detach().float().cpu().clone() for w in self._matrices()]

    def pre_train(self):
        if not self.rows or self.trainer.global_step != 0:
            if self.rows:
                log.info("Resumed at step %d: promoted rows keep their trained values", self.trainer.global_step)
            self._checked = True
            return
        self._before = self._snapshot()
        names = self.trainer.train_module.seed_token_rows_from_pieces(self.rows)  # type: ignore[attr-defined]
        self._seeded = self._snapshot()
        log.info("Seeded promoted rows %s in %s", self.rows, names)

    def post_step(self):
        if self._checked or self._seeded is None or self._before is None:
            return
        self._checked = True
        now = self._snapshot()
        for index, (cur, seeded, before) in enumerate(zip(now, self._seeded, self._before)):
            to_seed = (cur - seeded).norm(dim=-1)
            to_before = (cur - before).norm(dim=-1)
            log.info(
                "Promoted rows after step %d, matrix %d: distance to seed %s, to pre-seed %s",
                self.trainer.global_step,
                index,
                to_seed.tolist(),
                to_before.tolist(),
            )
            ok = torch.isfinite(to_seed) & torch.isfinite(to_before) & (to_seed < to_before)
            if not bool(ok.all()):
                raise RuntimeError("Promoted rows were not trained from their seed")
