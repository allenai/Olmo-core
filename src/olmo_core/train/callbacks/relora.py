import logging
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from olmo_core.distributed.utils import get_rank

from .callback import Callback

log = logging.getLogger(__name__)

__all__ = ["ReLoRACallback"]


@dataclass
class ReLoRACallback(Callback):
    """
    ReLoRA merge-and-restart for the early-LoRA adapters (:func:`olmo_core.nn.early_lora.merge_and_reinit_lora`;
    tokdrop large-data decay fix, 2026-10-30).

    After the optimizer step of every ``round(k * every_frac * max_steps)`` (k = 1, 2, ... while < max_steps) the
    adapters are folded into the frozen base, re-initialised, and their AdamW state is reset; the learning rate of
    every param group is then re-warmed linearly over ``rewarm_steps`` steps (step ``m + i`` runs at ``i / rewarm`` of
    the scheduled LR), the jagged schedule of ReLoRA. A rank-r adapter can then accumulate a higher-rank total update
    over the run at the same per-step cost. The merge is ~2 r x (base params) FLOPs per restart, charged to the
    FLOP meter through the model's ``_flop_extra_actual``.
    """

    every_frac: float = 0.25
    rewarm_steps: int = 20
    seed: int = 0
    merge_steps: List[int] = field(default_factory=list)
    _last_merge: Optional[int] = None

    def _tm(self):
        return self.trainer.train_module

    def pre_train(self):
        model = self._tm().model  # type: ignore[union-attr]
        if not getattr(model, "_early_lora", None):
            raise RuntimeError("ReLoRACallback attached but the model has no early LoRA adapters")
        if not 0.0 < self.every_frac < 1.0:
            raise ValueError(f"ReLoRA every_frac must be in (0, 1), got {self.every_frac}")
        T = int(self.trainer.max_steps or 0)
        if T <= 0:
            raise RuntimeError("ReLoRA needs a known max_steps")
        steps, k = [], 1
        while True:
            m = int(round(k * self.every_frac * T))
            if m >= T:
                break
            if m > 0 and m not in steps:
                steps.append(m)
            k += 1
        self.merge_steps = steps
        sched = getattr(self._tm(), "scheduler", None)
        if sched is None:
            raise RuntimeError("ReLoRA re-warm needs the train module's LR scheduler")
        orig = sched.set_lr
        cb = self

        def set_lr(_self, group: Dict[str, Any], trainer: Any):
            lr = orig(group, trainer)
            f = cb.lr_factor(int(trainer.global_step))
            if f < 1.0:
                import torch

                lr = lr * f
                cur = group.get(_self.lr_field)
                if isinstance(cur, torch.Tensor):
                    cur.fill_(lr)
                else:
                    group[_self.lr_field] = lr
            return lr

        sched.set_lr = types.MethodType(set_lr, sched)
        if get_rank() == 0:
            log.info("[relora] merge-and-restart after steps %s of %d, LR re-warm over %d steps", steps, T,
                     self.rewarm_steps)

    def lr_factor(self, step: int) -> float:
        if self._last_merge is None or self.rewarm_steps <= 0:
            return 1.0
        i = step - self._last_merge
        return 1.0 if i >= self.rewarm_steps or i <= 0 else i / float(self.rewarm_steps)

    def post_step(self):
        if self.step not in self.merge_steps:
            return
        from olmo_core.nn.early_lora import merge_and_reinit_lora

        tm = self._tm()
        model = tm.model  # type: ignore[union-attr]
        k = self.merge_steps.index(self.step) + 1
        out = merge_and_reinit_lora(model, getattr(tm, "optim", None), seed=self.seed, restart=k)
        model._flop_extra_actual = float(getattr(model, "_flop_extra_actual", 0.0)) + out["flops"]
        self._last_merge = int(self.step)
        self.trainer.record_metric("relora/restarts", float(k))
        self.trainer.record_metric("relora/merge_rel_delta", out["rel_delta"])
        if get_rank() == 0:
            log.info("[relora] step %d: restart %d done (|dW|/|W| = %.3e)", self.step, k, out["rel_delta"])

    def state_dict(self) -> Dict[str, Any]:
        return {"last_merge": self._last_merge}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._last_merge = state_dict.get("last_merge")
