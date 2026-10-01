"""Custom objectives with the transformer train modules' gradient lifecycle."""

from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch

Objective = Callable[[Any, Dict[str, Any]], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]


def train_batch_with_loss(
    module: Any,
    micro_batches: Sequence[Dict[str, Any]],
    objective: Objective,
    context_factory=None,
    *,
    reset_auxiliary_metrics: bool = False,
) -> List[Dict[str, torch.Tensor]]:
    """Accumulate a custom objective using Core's microbatch synchronization.

    The caller owns batch construction and objective normalization, and calls
    ``zero_grads()`` before this operation and ``optim_step()`` afterwards.
    ``objective(module, batch)`` owns the forward pass and returns a scalar loss
    and metrics. It must use the module's forward API and include any required
    auxiliary-loss scaling in the forward arguments. No CE labels or denominator
    are synthesized here. Metrics are detached before returning.

    Leave ``reset_auxiliary_metrics=False`` to collect and manage model auxiliary
    metrics yourself. Set it to True to discard model auxiliary metrics, clearing
    them before training and on exit (including failure). This affects reporting
    counters, not auxiliary losses or gradients. Returned objective metrics are
    copied when reset is enabled so clearing counters cannot mutate them.

    Pipeline objectives require a pipeline schedule and are not accepted by
    this non-pipeline entrypoint. Auxiliary-loss-free router balancing (``bias_gamma``)
    is rejected; auxiliary losses (``lb_loss_weight``, ``z_loss_weight``) are supported,
    including zero weights.
    """
    if getattr(module, "pp_enabled", False):
        raise NotImplementedError("Custom objectives require a non-pipeline train module")
    if not micro_batches:
        raise ValueError("A custom-objective batch must contain at least one microbatch")
    models = getattr(module, "model_parts", [module.model])
    for part_index, model in enumerate(models):
        for name, child in model.named_modules():
            if getattr(child, "bias_gamma", None) is not None:
                raise NotImplementedError(
                    "Custom objectives do not support auxiliary-loss-free router balancing "
                    f"(bias_gamma is set on model part {part_index}, module {name or '<root>'}); "
                    "the per-batch score_bias update in post_batch() is not applied on this path."
                )
    if reset_auxiliary_metrics:
        for model in models:
            model.reset_auxiliary_metrics()
    try:
        # Keep the standard train module's cached mode in sync with the model.
        # OLMoDDP has no mode cache and switches each model part directly.
        set_model_mode = getattr(module, "_set_model_mode", None)
        if set_model_mode is not None:
            set_model_mode("train")
        else:
            for model in models:
                model.train()
        metrics = []
        for index, batch in enumerate(micro_batches):
            with (
                module._train_microbatch_context(index, len(micro_batches)),
                context_factory(module, batch) if context_factory is not None else nullcontext(),
            ):
                loss, values = objective(module, batch)
                if loss.ndim != 0 or not loss.requires_grad:
                    raise ValueError("The objective must return a differentiable scalar loss")
                loss.backward()
                metrics.append(
                    {
                        name: value.detach().clone() if reset_auxiliary_metrics else value.detach()
                        for name, value in values.items()
                    }
                )
        for model in models:
            if hasattr(model, "finalize_grad_reduce"):
                model.finalize_grad_reduce()
        return metrics
    finally:
        if reset_auxiliary_metrics:
            for model in models:
                model.reset_auxiliary_metrics()
