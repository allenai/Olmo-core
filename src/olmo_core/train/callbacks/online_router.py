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

    #: the fit carried across a preemption resume (2026-10-30): without it a resumed run restarts the router with no
    #: weights, i.e. one full-context warmup step mid-run and a fresh ridge accumulation
    _STATE_KEYS = ("w", "XtX", "Xty", "n_seen", "step", "task_w", "task_XtX", "task_Xty", "task_seen", "frozen_blocks")

    def state_dict(self) -> Dict[str, Any]:
        import torch

        router = self._router()
        if router is None:
            return {}

        def cpu(v):
            if torch.is_tensor(v):
                return v.detach().cpu()
            if isinstance(v, dict):
                return {k: cpu(x) for k, x in v.items()}
            return v

        chain = [{k: cpu(getattr(r, k)) for k in self._STATE_KEYS if hasattr(r, k)} for r in router.chain()]
        return {"chain": chain, "freeze_early": bool(router.cfg.freeze_early)}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        import torch

        router = self._router()
        if router is None or not state_dict.get("chain"):
            return
        model = self.trainer.train_module.model  # type: ignore[union-attr]
        dev = next(model.parameters()).device

        def to(v):
            if torch.is_tensor(v):
                return v.to(dev)
            if isinstance(v, dict):
                return {k: to(x) for k, x in v.items()}
            return v

        chain = router.chain()
        if len(chain) != len(state_dict["chain"]):
            raise RuntimeError(f"online router resume: {len(state_dict['chain'])} saved stages vs {len(chain)} built")
        for r, d in zip(chain, state_dict["chain"]):
            for k, v in d.items():
                setattr(r, k, to(v))
            r._pending = []
        if get_rank() == 0:
            log.info("[online-router] resumed the fit of %d stage(s) at router step %s", len(chain),
                     state_dict["chain"][0].get("step"))

    def pre_train(self):
        router = self._router()
        if router is None:
            raise RuntimeError("OnlineRouterCallback attached but the model has no online token router")
        for r in router.chain():
            r.total_steps = int(self.trainer.max_steps) if self.trainer.max_steps else None

    def pre_step(self, batch: Dict[str, Any]):
        import torch

        router0 = self._router()
        de = int(getattr(router0.cfg, "diag_every", 0) or 0)
        self._diag_batch = batch if de > 0 and self.trainer.global_step % de == 0 else None
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

    def _diag(self, router, batch) -> Dict[str, float]:
        """diag_every: answer CE of the first micro-batch routed vs full context, no_grad, state restored."""
        import copy

        import torch

        tm = self.trainer.train_module
        model = tm.model  # type: ignore[union-attr]
        B = int(batch["input_ids"].shape[0])
        n = max(1, min(B, int(getattr(tm, "rank_microbatch_size", 0) or 0) // max(1, int(batch["input_ids"].shape[1]))
                       or 2))
        sub = {k: (v[:n] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == B else v) for k, v in batch.items()}
        saved_r = []
        for r in router.chain():
            d = {k: copy.copy(v) for k, v in r.__dict__.items()}
            d["_stats"] = {k: list(v) for k, v in r._stats.items()}
            saved_r.append(d)
        saved_m = {k: copy.deepcopy(getattr(model, k)) for k in ("_flop_extra_actual", "_flop_example_lens",
                                                                  "_soft_token_compaction") if hasattr(model, k)}
        out = {}
        try:
            with torch.no_grad():
                input_ids, labels, kw = tm._prepare_batch(sub)  # type: ignore[union-attr]
                nl = float((labels != tm.label_ignore_index).sum().clamp(min=1))  # type: ignore[union-attr]
                for name, force in (("routed", False), ("full", True)):
                    for r in router.chain():
                        r._force_dense = force
                    o = tm.model_forward(input_ids, labels=labels,  # type: ignore[union-attr]
                                         ignore_index=tm.label_ignore_index, loss_reduction="sum", **kw)
                    ce = getattr(o, "ce_loss", None)
                    out[f"online_router/diag_ce_{name}"] = float((ce if ce is not None else o.loss).float()) / nl
        finally:
            for r, d in zip(router.chain(), saved_r):
                r.__dict__.clear()
                r.__dict__.update(d)
                r._force_dense = False
            for k, v in saved_m.items():
                setattr(model, k, v)
        if len(out) == 2:
            out["online_router/diag_ce_gap"] = out["online_router/diag_ce_full"] - out["online_router/diag_ce_routed"]
        return out

    def post_step(self):
        router = self._router()
        import torch

        m = {}
        for r in router.chain():
            m.update(r.metrics())
        if torch.cuda.is_available():
            # this step's peak allocated memory on this rank (reset in pre_step)
            m["online_router/peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
        if getattr(self, "_diag_batch", None) is not None:
            dg = self._diag(router, self._diag_batch)
            self._diag_batch = None
            for k, v in dg.items():
                self.trainer.record_metric(k, v)
            if get_rank() == 0 and dg:
                log.info("[online-router] step %d diag: %s", self.step,
                         " ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in sorted(dg.items())))
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
