"""
The float-weighted cross entropy of :mod:`olmo_core.nn.functional` computed a chunk of tokens at
a time, so the logits of a whole micro-batch are never held in memory.
"""

from typing import Any, Callable, Dict, Optional, Tuple

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


def _mm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    ``a @ b`` with a float32 result. On the GPU the low-precision inputs feed the tensor-core
    GEMM directly and only the output is float32 (what the one-shot bfloat16 matmul accumulates
    in before rounding); elsewhere the inputs are upcast first.
    """
    if a.is_cuda and a.dtype != torch.float32:
        try:
            return torch.mm(a, b, out_dtype=torch.float32)
        except (TypeError, NotImplementedError, RuntimeError):
            pass
    return torch.mm(a.float(), b.float())


def _forward_chunk(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    ignore_index: int,
    compute_z_loss: bool,
    z_loss_multiplier: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    One chunk's partial sums of the weighted cross entropy and z-loss: its logits in float32,
    their log-sum-exp, the negative log-likelihood of the labels, each weighted and summed over
    the chunk's valid rows.
    """
    logits = _project(hidden, weight, bias).float()
    valid = labels != ignore_index
    w = loss_weights.float()
    lse = torch.logsumexp(logits, dim=-1)
    nll = lse - logits.gather(1, labels.clamp(min=0).unsqueeze(1)).squeeze(1)
    ce = torch.dot(torch.where(valid, nll, nll.new_zeros(())), w)
    if not compute_z_loss:
        return ce, None
    return ce, torch.dot(lse * lse, w * valid) * z_loss_multiplier


def _backward_chunk(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    grad_ce: torch.Tensor,
    grad_z: torch.Tensor,
    ignore_index: int,
    compute_z_loss: bool,
    z_loss_multiplier: float,
) -> torch.Tensor:
    """
    The gradient of the losses with respect to one chunk's logits, in the logits' dtype: the
    logits recomputed, their softmax in float32, scaled per row by the loss weights and the
    incoming gradients, minus the one-hot term at each valid row's label.
    """
    logits = _project(hidden, weight, bias)
    logits_f = logits.float()
    valid = labels != ignore_index
    w = loss_weights.float()
    lse = torch.logsumexp(logits_f, dim=-1)
    softmax = torch.exp(logits_f - lse.unsqueeze(1))
    # d(sum_i w_i * nll_i) / d logits_i = w_i * (softmax_i - onehot_i) on valid rows.
    scale = torch.where(valid, w * grad_ce, w.new_zeros(()))
    if compute_z_loss:
        # d(m * sum_i w_i * lse_i^2) / d logits_i = m * w_i * 2 * lse_i * softmax_i.
        scale = scale + torch.where(
            valid, w * (grad_z * z_loss_multiplier * 2.0) * lse, w.new_zeros(())
        )
    grad_logits = softmax * scale.unsqueeze(1)
    # The one-hot term, subtracted at each valid row's label (a zero for the others), as one
    # scatter so nothing here waits on the device.
    onehot = torch.where(valid, -(w * grad_ce), w.new_zeros(())).unsqueeze(1)
    grad_logits.scatter_add_(1, labels.clamp(min=0).unsqueeze(1), onehot)
    return grad_logits.to(logits.dtype)


_COMPILED: Dict[str, Callable[..., Any]] = {}


def _chunk_fn(fn: Callable[..., Any], compile: bool) -> Callable[..., Any]:
    """
    ``fn`` (one of the per-chunk computations above) or, with ``compile``, its
    :func:`torch.compile` version with static shapes, built once per process. Every chunk the
    compiled version sees has ``chunk_size`` rows (the last one is padded), so it is compiled
    once per shape/dtype/device combination rather than once per step.
    """
    if not compile:
        return fn
    compiled = _COMPILED.get(fn.__name__)
    if compiled is None:
        compiled = _COMPILED[fn.__name__] = torch.compile(fn, dynamic=False)
    return compiled


def _pad_chunk(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    rows: int,
    ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The chunk padded to ``rows`` rows with zero hidden states, ignored labels and zero loss
    weights: rows that contribute exactly zero to every partial sum and gradient.
    """
    pad = rows - hidden.shape[0]
    if pad <= 0:
        return hidden, labels, loss_weights
    return (
        F.pad(hidden, (0, 0, 0, pad)),
        F.pad(labels, (0, pad), value=ignore_index),
        F.pad(loss_weights, (0, pad)),
    )


class _ChunkedWeightedCrossEntropy(torch.autograd.Function):
    """
    ``weighted_cross_entropy_loss(F.linear(hidden, weight, bias), labels, loss_weights)`` with
    the same arithmetic as that one-shot path, applied ``chunk_size`` tokens at a time.

    The forward keeps no logits; the backward recomputes each chunk's logits, forms the
    gradient of the loss with respect to them in float32, casts it to the logits' dtype as the
    one-shot path does, and applies the projection's backward per chunk. The weight (and bias)
    gradient is accumulated across chunks in float32 (each chunk's GEMM keeps its float32
    result) and cast once, so it matches the one-shot single matmul up to float32 summation
    order even when the weight is bfloat16.

    With ``compile`` the per-chunk math (:func:`_forward_chunk`, :func:`_backward_chunk`) runs
    as one static-shape :func:`torch.compile` graph each, the last chunk padded to the chunk
    size so no step recompiles; the per-chunk GEMMs of the backward stay eager.
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
        compile: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        forward_chunk = _chunk_fn(_forward_chunk, compile)
        # Detached operands: nothing here is tracked, and a compiled graph is guarded on
        # ``requires_grad`` (a padded chunk would otherwise differ from a sliced one).
        weight_, bias_ = weight.detach(), None if bias is None else bias.detach()
        ce_loss = hidden.new_zeros((), dtype=torch.float32)
        z_loss = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, hidden.shape[0], chunk_size):
            end = start + chunk_size
            h, y, w = hidden[start:end].detach(), labels[start:end], loss_weights[start:end]
            if compile:
                h, y, w = _pad_chunk(h, y, w, chunk_size, ignore_index)
            ce, z = forward_chunk(
                h, weight_, bias_, y, w, ignore_index, compute_z_loss, z_loss_multiplier
            )
            ce_loss = ce_loss + ce
            if z is not None:
                z_loss = z_loss + z
        ctx.save_for_backward(hidden, weight, bias, labels, loss_weights)
        ctx.ignore_index = ignore_index
        ctx.compute_z_loss = compute_z_loss
        ctx.z_loss_multiplier = z_loss_multiplier
        ctx.chunk_size = chunk_size
        ctx.compile = compile
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
        backward_chunk = _chunk_fn(_backward_chunk, ctx.compile)
        weight_, bias_ = weight.detach(), None if bias is None else bias.detach()
        for start in range(0, hidden.shape[0], ctx.chunk_size):
            end = start + ctx.chunk_size
            h = hidden[start:end].detach()
            rows = h.shape[0]
            y, w = labels[start:end], loss_weights[start:end]
            if ctx.compile:
                h, y, w = _pad_chunk(h, y, w, ctx.chunk_size, ctx.ignore_index)
            grad_logits = backward_chunk(
                h,
                weight_,
                bias_,
                y,
                w,
                grad_ce,
                grad_z,
                ctx.ignore_index,
                ctx.compute_z_loss,
                ctx.z_loss_multiplier,
            )
            if grad_logits.shape[0] != rows:
                grad_logits, h = grad_logits[:rows], h[:rows]
            if grad_hidden is not None:
                grad_hidden[start:end] = torch.mm(grad_logits, weight_)
            if grad_weight is not None:
                grad_weight += _mm_fp32(grad_logits.t(), h)
            if grad_bias is not None:
                grad_bias += grad_logits.float().sum(0)
            del grad_logits
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
    compile: bool = False,
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
    :param compile: Run each chunk's element-wise math (logits to partial sums in the forward,
        recomputed logits to their gradient in the backward) as a static-shape
        :func:`torch.compile` graph; the last chunk is padded to ``chunk_size`` so the graphs
        are compiled once. The arithmetic is the eager path's; only the fusion and the
        summation order within a chunk differ.

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
        compile,
    )
    return ce_loss, z_loss if compute_z_loss else None
