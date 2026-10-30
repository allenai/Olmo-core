"""
Multi-view training from a cached split-layer state (``--st-otr-views k``, online token router with a mid-stack drop).

The recipe (whole-model LoRA + two-stage online router, keep 20% at block 7 / 7% at block 16) spends about 45% of its
per-step FLOPs in blocks [0, 7), which run at the FULL row length. With ``k`` views a batch pays that once:

1. **View 1**: the ordinary routed forward, except that the graph is split at block ``L`` (the router's first drop
   layer): ``h_L`` is cached detached and the late stack [L, n) runs from a fresh leaf. Its backward reaches the late
   parameters and ``dL/dh_L``; the early stack's graph (activation-checkpointed block inputs) stays alive.
2. **Views 2..k**: the late stack only, from the cached ``h_L``, with a FRESH token subset at every router stage (the
   router's draw advances per forward; with ``--st-otr-score random`` each view is an independent uniform subset of
   each document's tokens). Between views the late parameters take a full optimizer step (clip, LR, AdamW).
3. **Early stack**: ONE backward through blocks [0, L) with the MEAN of the views' ``dL/dh_L``, then the trainer's
   ordinary optimizer step updates the early adapters (from that mean) and the late ones (from view ``k``).

So the late stack takes ``k`` optimizer steps per batch and the early stack one. Why one backward on the mean
rather than updating the early stack from the last view only: every view's ``dL/dh_L`` is an unbiased draw of the
early-stack gradient for a different token subset, so averaging the k of them cuts the subset-sampling variance at
no extra FLOPs (the early backward is paid once either way), while the last view alone discards k - 1 of them. The
early parameters do not change during the k views (their gradient is still unset), so the early graph from view 1
is exact for them; the only staleness is that view ``v``'s ``dL/dh_L`` was taken with late parameters that are
``v - 1`` steps newer than view 1's, which is the same order as ordinary gradient-accumulation staleness at LR 1e-5.

Cost per row (x dense): early stack forward + backward once (charged in view 1, as in the recipe) plus ``k`` x the
late stack at the kept lengths (views 2..k charged by the router's stage-0 ``mid_compact`` with the late-only formula;
they log no example lengths, so the FLOP meter's per-example dense charge counts every row once). Dense-equivalent
FLOPs therefore stay "one pass over the batch", and ``actual_pflops`` charges exactly what ran.

Steps that the router trains dense (warmup / periodic dense steps) and the trainer's dry-run batch run one view, the
ordinary way.
"""

from __future__ import annotations

import logging
from types import MethodType
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

log = logging.getLogger(__name__)

__all__ = ["enable_multiview"]

_DEBUG = bool(int(__import__("os").environ.get("MV_DEBUG", "0") or 0))


def _early_fsdp_modules(model: Any, layer: int) -> List[Any]:
    """The FSDP-wrapped blocks [0, layer) (empty without FSDP)."""
    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:  # pragma: no cover
        return []
    return [b for i, b in enumerate(model.blocks.values()) if i < layer and isinstance(b, FSDPModule)]


def _log_fsdp_states(mods: List[Any], tag: str) -> None:
    for j, m in enumerate(mods):
        try:
            stt = m._get_fsdp_state()
            g = stt._fsdp_param_group
            log.info("[multiview-debug] %s: early block %d state=%s group_state=%s unsharded=%s", tag, j,
                     stt._training_state, getattr(g, "_training_state", None), getattr(g, "is_unsharded", None))
        except Exception as e:  # noqa: BLE001
            log.info("[multiview-debug] %s: block %d: %s", tag, j, e)


def enable_multiview(train_module: Any, views: int) -> None:
    """Replace ``train_module.train_batch`` with the k-view version (``views`` > 1)."""
    if views <= 1:
        return
    model = train_module.model
    router = getattr(model, "_online_token_router", None)
    if router is None or int(router.cfg.drop_layer) <= 0:
        raise ValueError("multi-view needs the online token router with a mid-stack drop (drop_layer > 0)")
    c = router.cfg
    if c.early_nograd or c.freeze_early or c.freeze_after or c.freeze_schedule or c.mod_layers or c.target_layer \
            or c.exit_loss_weight > 0 or c.merge_dropped:
        raise ValueError("multi-view does not compose with early_nograd / freeze_* / mod_layers / target_layer / "
                         "exit_loss_weight / merge_dropped")
    if getattr(model, "_layer_drop", None) is not None or getattr(model, "_token_exit", None) is not None:
        raise ValueError("multi-view does not compose with layer drop / token exit")
    train_module._multiview = {"views": int(views), "orig": train_module.train_batch, "layer": int(c.drop_layer)}
    train_module.train_batch = MethodType(_mv_train_batch, train_module)
    log.info("[multiview] %d views per batch: blocks [0, %d) once, blocks [%d, n) + head per view", views,
             c.drop_layer, c.drop_layer)


def _reduced_loss(tm: Any, x: torch.Tensor) -> torch.Tensor:
    """The rank-mean of a per-rank batch loss (what TransformerTrainModule hands the SkipStepOptimizer)."""
    x = x.clone()
    if dist.is_available() and dist.is_initialized():
        x.div_(tm._reduce_divide_factor)
        dist.all_reduce(x)
        x.div_(tm.world_size)
        x.mul_(tm._reduce_divide_factor)
    return x


def _inner_step(tm: Any, ce: torch.Tensor) -> Optional[torch.Tensor]:
    """An optimizer step between views: what ``TransformerTrainModule.optim_step`` does, plus zero_grad."""
    from olmo_core.optim import SkipStepOptimizer

    skip = isinstance(tm.optim, SkipStepOptimizer)
    if skip:
        tm.optim.latest_loss = _reduced_loss(tm, ce)
    gn = None
    if tm.max_grad_norm is not None:
        gn = tm._clip_grad_norm(tm.max_grad_norm)
        if skip:
            tm.optim.latest_grad_norm = gn
    if tm.scheduler is not None:
        for group in tm.optim.param_groups:
            tm.scheduler.set_lr(group, tm.trainer)
    tm.optim.step()
    tm.model.post_optim_step()
    tm.optim.zero_grad(set_to_none=True)
    return gn


def _mv_train_batch(self: Any, batch: Dict[str, Any], dry_run: bool = False):
    from olmo_core.data.utils import get_labels, split_batch
    from olmo_core.distributed.utils import get_local_tensor
    from olmo_core.optim import SkipStepOptimizer
    from olmo_core.train.common import ReduceType
    from olmo_core.utils import move_to_device

    st = self._multiview
    model = self.model
    router = model._online_token_router
    k = int(st["views"])
    if dry_run or router.dense_now():
        st["last_views"] = 1
        return st["orig"](batch, dry_run=dry_run)
    if getattr(model, "_flop_pad_id", None) is None:
        raise RuntimeError("multi-view needs the per-example FLOP meter (FlopMeterCallback(per_example=True)): the "
                           "repeat views are priced by the router against per-example lengths")
    if batch.get("instance_mask") is not None:
        raise NotImplementedError("multi-view with an instance filter")
    self._set_model_mode("train")
    if "labels" not in batch:
        batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)
    n_loss = move_to_device((batch["labels"] != self.label_ignore_index).sum(), self.device)
    n_tok = batch["labels"].numel()
    self.record_metric("train/masked labels (%)", (n_tok - n_loss) / n_tok, ReduceType.mean)
    seq_len = batch["input_ids"].shape[1]
    if self.microbatch_sort_pad_id is not None and batch["input_ids"].shape[0] > 1:
        n_rows = batch["input_ids"].shape[0]
        lengths = (batch["input_ids"] != self.microbatch_sort_pad_id).sum(dim=1)
        order = torch.argsort(lengths, descending=True)
        batch = {kk: (v[order.to(v.device)] if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n_rows else v)
                 for kk, v in batch.items()}
    mbs = split_batch(batch, self.rank_microbatch_size // seq_len)
    n = len(mbs)
    early: List[Optional[torch.Tensor]] = [None] * n
    cached: List[Optional[torch.Tensor]] = [None] * n
    delta: List[Optional[torch.Tensor]] = [None] * n
    ces: List[torch.Tensor] = []
    gns: List[torch.Tensor] = []
    for v in range(k):
        ce_v = move_to_device(torch.tensor(0.0), self.device)
        for i, mb in enumerate(mbs):
            ctx: Dict[str, Any] = {"layer": st["layer"], "repeat": v > 0, "h7": cached[i]}
            with self._train_microbatch_context(i, n):
                input_ids, labels, kw = self._prepare_batch(dict(mb))  # _prepare_batch pops from the dict
                model._mv_ctx = ctx
                try:
                    _, loss, ce_loss, z_loss = self.model_forward(
                        input_ids, labels=labels, ignore_index=self.label_ignore_index, loss_reduction="sum",
                        z_loss_multiplier=self.z_loss_multiplier, loss_div_factor=n_loss, return_logits=False, **kw)
                finally:
                    model._mv_ctx = None
                ce_v += get_local_tensor(ce_loss.detach())
                del ce_loss, z_loss
                loss.backward()
                del loss
            leaf = ctx.get("leaf")
            if leaf is None or leaf.grad is None:
                raise RuntimeError("multi-view: the split-layer leaf got no gradient (did the forward reach the "
                                   "router's drop layer?)")
            if v == 0:
                early[i], cached[i] = ctx["h7_attached"], ctx["h7"]
            g = leaf.grad.float()
            delta[i] = g if delta[i] is None else delta[i].add_(g)
            del leaf, g, ctx
        ces.append(ce_v)
        if v < k - 1:
            gn = _inner_step(self, ce_v)
            if gn is not None:
                gns.append(gn.detach())
    # the early stack: one backward with the mean of the views' dL/dh_L
    early_mods = _early_fsdp_modules(model, int(st["layer"]))
    for i in range(n):
        h = early[i]
        assert h is not None and delta[i] is not None
        with self._train_microbatch_context(i, n):
            if h.requires_grad:
                if _DEBUG and i == 0:
                    _log_fsdp_states(early_mods, "before early backward")
                # FSDP2 unshards a module for backward from a tensor hook on its forward output; this graph was built
                # in view 1 and later backwards' final callbacks already resharded / finalized these modules, so
                # unshard them explicitly (each one's post-backward reshards it and reduce-scatters its gradients)
                for m in early_mods:
                    m.unshard()
                torch.autograd.backward(h.view(h.shape), (delta[i] / k).to(h.dtype))
        early[i] = cached[i] = delta[i] = None
    del batch, mbs
    st["last_views"] = k
    self.model.post_batch(dry_run=False)
    # the logged train CE is view 1's (each row's first look this step); later views' CE is logged separately
    ce0 = ces[0]
    if isinstance(self.optim, SkipStepOptimizer):
        ce0 = _reduced_loss(self, ce0)
        self.record_ce_loss(ce0)
        self.optim.latest_loss = ce0
    else:
        self.record_ce_loss(ce0, ReduceType.mean)
    self.record_metric("multiview/ce_last_view", ces[-1], ReduceType.mean)
    self.record_metric("multiview/ce_drop_first_to_last", ces[0] - ces[-1], ReduceType.mean)
    self.record_metric("multiview/views", torch.tensor(float(k), device=ces[0].device), ReduceType.mean)
    if gns:
        self.record_metric("multiview/inner_grad_norm", torch.stack(gns).mean(), None)
    for name, (val, red) in self.model.compute_auxiliary_metrics(reset=True).items():
        self.record_metric(name, val, red, namespace="train")
