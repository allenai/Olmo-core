"""
The float-weighted cross entropy of :mod:`olmo_core.nn.functional` computed a chunk of tokens at
a time, so the logits of a whole micro-batch are never held in memory.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from olmo_core.nn.functional import weighted_cross_entropy_loss

__all__ = ["chunked_weighted_cross_entropy_loss"]


def _project(
    hidden: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    """One chunk's logits, as the LM head's output projection computes them."""
    return F.linear(hidden, weight, bias)


class _ChunkedWeightedCrossEntropy(torch.autograd.Function):
    """
    ``weighted_cross_entropy_loss(F.linear(hidden, weight, bias), labels, loss_weights)`` with
    the same arithmetic as that one-shot path, applied ``chunk_size`` tokens at a time.

    The forward keeps no logits; the backward recomputes each chunk's logits, forms the
    gradient of the loss with respect to them in float32, casts it to the logits' dtype as the
    one-shot path does, and applies the projection's backward per chunk. The weight (and bias)
    gradient is accumulated across chunks in float32 and cast once, so it matches the one-shot
    single matmul up to float32 summation order even when the weight is bfloat16.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        labels: torch.Tensor,
        loss_weights: torch.Tensor,
        ignore_index: int,
        compute_z_loss: bool,
        z_loss_multiplier: float,
        chunk_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ce_loss = hidden.new_zeros((), dtype=torch.float32)
        z_loss = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, hidden.shape[0], chunk_size):
            end = start + chunk_size
            logits = _project(hidden[start:end], weight, bias).float()
            y = labels[start:end]
            valid = y != ignore_index
            w = loss_weights[start:end].float()
            lse = torch.logsumexp(logits, dim=-1)
            nll = lse - logits.gather(1, y.clamp(min=0).unsqueeze(1)).squeeze(1)
            ce_loss = ce_loss + torch.dot(torch.where(valid, nll, nll.new_zeros(())), w)
            if compute_z_loss:
                z_loss = z_loss + torch.dot(lse * lse, w * valid) * z_loss_multiplier
            del logits
        ctx.save_for_backward(hidden, weight, bias, labels, loss_weights)
        ctx.ignore_index = ignore_index
        ctx.compute_z_loss = compute_z_loss
        ctx.z_loss_multiplier = z_loss_multiplier
        ctx.chunk_size = chunk_size
        return ce_loss, z_loss

    @staticmethod
    def backward(ctx: Any, grad_ce: torch.Tensor, grad_z: torch.Tensor):  # type: ignore[override]
        hidden, weight, bias, labels, loss_weights = ctx.saved_tensors
        need_hidden, need_weight, need_bias = ctx.needs_input_grad[:3]
        grad_hidden = torch.empty_like(hidden) if need_hidden else None
        grad_weight = torch.zeros_like(weight, dtype=torch.float32) if need_weight else None
        grad_bias = (
            torch.zeros_like(bias, dtype=torch.float32) if need_bias and bias is not None else None
        )
        for start in range(0, hidden.shape[0], ctx.chunk_size):
            end = start + ctx.chunk_size
            h = hidden[start:end]
            logits = _project(h, weight, bias)
            logits_f = logits.float()
            y = labels[start:end]
            valid = y != ctx.ignore_index
            w = loss_weights[start:end].float()
            lse = torch.logsumexp(logits_f, dim=-1)
            softmax = torch.exp(logits_f - lse.unsqueeze(1))
            # d(sum_i w_i * nll_i) / d logits_i = w_i * (softmax_i - onehot_i) on valid rows.
            scale = torch.where(valid, w * grad_ce, w.new_zeros(()))
            if ctx.compute_z_loss:
                # d(m * sum_i w_i * lse_i^2) / d logits_i = m * w_i * 2 * lse_i * softmax_i.
                scale = scale + torch.where(
                    valid, w * (grad_z * ctx.z_loss_multiplier * 2.0) * lse, w.new_zeros(())
                )
            grad_logits = softmax * scale.unsqueeze(1)
            rows = torch.arange(y.shape[0], device=y.device)[valid]
            grad_logits[rows, y[valid]] -= (w * grad_ce)[valid]
            grad_logits = grad_logits.to(logits.dtype)
            if grad_hidden is not None:
                grad_hidden[start:end] = torch.mm(grad_logits, weight)
            if grad_weight is not None:
                grad_weight += torch.mm(grad_logits.t().float(), h.float())
            if grad_bias is not None:
                grad_bias += grad_logits.float().sum(0)
            del logits, logits_f, softmax, grad_logits
        return (
            grad_hidden,
            None if grad_weight is None else grad_weight.to(weight.dtype),
            None if grad_bias is None else grad_bias.to(bias.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def chunked_weighted_cross_entropy_loss(
    w_out: nn.Module,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    *,
    ignore_index: int = -100,
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
    chunk_size: int = 2048,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    :func:`~olmo_core.nn.functional.weighted_cross_entropy_loss` of ``w_out(hidden)``, computed
    ``chunk_size`` tokens at a time so that the logits of all tokens never exist at once.

    The loss and every gradient equal the one-shot computation up to float32 summation order
    (see :class:`_ChunkedWeightedCrossEntropy`). The peak memory is one chunk's logits and
    their float32 softmax instead of the whole micro-batch's.

    :param w_out: The LM head's output projection, a linear layer ``(N, d_model) -> (N, vocab)``.
    :param hidden: The normed hidden states to score, shape ``(N, d_model)``.
    :param labels: Target token IDs, shape ``(N,)``.
    :param loss_weights: Per-token float weights, shape ``(N,)``.
    :param ignore_index: Target value ignored by the cross entropy.
    :param compute_z_loss: Compute the weighted z-loss as well.
    :param z_loss_multiplier: The multiplier of the z-loss.
    :param chunk_size: Tokens per chunk; ``0`` scores every token at once.

    :returns: The weighted cross entropy (a scalar) and the weighted z-loss (a scalar, or
        ``None`` without ``compute_z_loss``), both summed over tokens.
    """
    if chunk_size <= 0:
        return weighted_cross_entropy_loss(
            w_out(hidden),
            labels,
            loss_weights,
            ignore_index=ignore_index,
            compute_z_loss=compute_z_loss,
            z_loss_multiplier=z_loss_multiplier,
        )
    weight = getattr(w_out, "weight", None)
    if not isinstance(weight, torch.Tensor) or isinstance(weight, DTensor):
        raise TypeError(
            "The chunked weighted loss needs an unsharded linear output projection; "
            f"got {type(w_out).__name__} with weight {type(weight).__name__}"
        )
    ce_loss, z_loss = _ChunkedWeightedCrossEntropy.apply(
        hidden,
        weight,
        getattr(w_out, "bias", None),
        labels,
        loss_weights,
        ignore_index,
        compute_z_loss,
        z_loss_multiplier,
        chunk_size,
    )
    return ce_loss, z_loss if compute_z_loss else None
