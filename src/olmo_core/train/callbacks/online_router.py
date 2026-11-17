import logging
from dataclasses import dataclass
from typing import Any, Dict

from olmo_core.distributed.utils import get_rank

from .callback import Callback

log = logging.getLogger(__name__)

__all__ = ["OnlineRouterCallback"]


@dataclass
class OnlineRouterCallback(Callback):
    """
    Drives :class:`~olmo_core.nn.online_token_router.OnlineTokenRouter`: before every optimizer
    step it refits the router on the targets the previous step's backward measured (one collective
    per step on every rank), and it logs the router's diagnostics (``online_router/*``: realised
    keep fraction, held-out rank correlation of the scores with the measured targets on the
    exploration tokens, gold-leak AUC, weight norms).
    """

    log_every: int = 1

    def _router(self):
        return getattr(self.trainer.train_module.model, "_online_token_router", None)  # type: ignore[union-attr]

    def pre_train(self):
        router = self._router()
        if router is None:
            raise RuntimeError("OnlineRouterCallback attached but the model has no online token router")
        for r in router.chain():
            r.total_steps = int(self.trainer.max_steps) if self.trainer.max_steps else None

    def pre_step(self, batch: Dict[str, Any]):
        import torch

        router0 = self._router()
        fa = int(getattr(router0.cfg, "freeze_after", 0) or 0)
        if fa and self.trainer.global_step > fa and not router0.cfg.freeze_early:
            # warmup-then-freeze: from now on the early stack is frozen, and its grads / optimizer state are released
            from olmo_core.nn.online_token_router import freeze_and_release

            n = freeze_and_release(self.trainer.train_module.model, router0.cfg.drop_layer,
                                   getattr(self.trainer.train_module, "optim", None))
            router0.cfg.freeze_early = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if get_rank() == 0:
                log.info("[online-router] step %d: froze embeddings + blocks [0, %d) (%.0fM params) and released "
                         "their optimizer state", self.trainer.global_step, router0.cfg.drop_layer, n / 1e6)

        sched = str(getattr(router0.cfg, "freeze_schedule", "") or "")
        if sched:
            # progressive freeze: the deepest stage whose step share has passed
            from olmo_core.nn.online_token_router import freeze_and_release, parse_freeze_schedule

            total = int(router0.total_steps or self.trainer.max_steps or 0)
            want = max([n for n, f in parse_freeze_schedule(sched) if self.trainer.global_step > int(round(f * total))],
                       default=0)
            if want > router0.frozen_blocks:
                n = freeze_and_release(self.trainer.train_module.model, want,
                                       getattr(self.trainer.train_module, "optim", None))
                router0.frozen_blocks = want
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if get_rank() == 0:
                    log.info("[online-router] step %d: progressive freeze -> embeddings + blocks [0, %d) frozen "
                             "(%.0fM params), optimizer state released", self.trainer.global_step, want, n / 1e6)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        router = self._router()
        for r in router.chain():  # every pyramid stage refits from its own targets
            r.begin_step(self.trainer.global_step, model=self.trainer.train_module.model)  # type: ignore[union-attr]

    def post_step(self):
        router = self._router()
        import torch

        m = {}
        for r in router.chain():
            m.update(r.metrics())
        if torch.cuda.is_available():
            # this step's peak allocated memory on this rank (reset in pre_step)
            m["online_router/peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
        if self.log_every and self.step % self.log_every == 0:
            for k, v in m.items():
                self.trainer.record_metric(k, v)
        every = int(router.cfg.log_every)
        # also print every step that carries the full-context (dense-step) stats: they are computed on the step after a
        # dense step, which a log_every that is a multiple of dense_every never hits (offline wandb = console only)
        full = "online_router/rank_corr_fullctx" in m
        if get_rank() == 0 and every > 0 and (self.step <= 3 or self.step % every == 0 or full):
            log.info(
                "[online-router] step %d: %s w=%s",
                self.step,
                " ".join(f"{k.split('/')[-1]}={v:.4g}" for k, v in sorted(m.items())),
                router.weights_digest(),
            )
