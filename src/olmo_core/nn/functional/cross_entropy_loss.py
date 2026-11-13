import logging
from typing import Callable, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "cross_entropy_loss",
    "weighted_cross_entropy_loss",
    "fused_linear_cross_entropy_loss",
]

log = logging.getLogger(__name__)


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    reduction: Literal["mean", "sum", "none"] = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Cross entropy loss that optionally computes the softmax auxiliary loss (z-loss) as well.

    :param logits: Predicted unnormalized logits with shape ``(N, vocab_size)``.
    :param labels: Ground truth class indices with shape ``(N,)``.
    :param ignore_index: Specifies a target value that is ignored and does not contribute to
        the input gradient.
    :param reduction: Specifies the reduction to apply to the output.
        Can be "none", "mean", or "sum".
    :param compute_z_loss: Compute the softmax auxiliary loss as well.
    :param z_loss_multiplier: The multiplier to apply to the z-loss.

    :returns: The cross entropy loss and optionally the z-loss.
    """
    logits = logits.float()
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)

    if not compute_z_loss:
        return loss, None

    z_squared = logits.logsumexp(-1).pow(2)
    mask = labels != ignore_index
    if reduction == "mean":
        z_squared = (z_squared * mask).sum() / mask.sum()
    elif reduction == "sum":
        z_squared = (z_squared * mask).sum()

    z_loss = z_loss_multiplier * z_squared

    return loss, z_loss


def weighted_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    *,
    ignore_index: int = -100,
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Cross entropy loss with a per-token float weight, summed over tokens.

    This reproduces the loss reduction used by Molmo2 / ``mm_olmo`` training,
    where each token carries a float weight (non-zero only on response tokens,
    optionally scaled per-annotation) instead of a binary mask:

    .. code-block:: python

        ce = sum_i CE(logits_i, labels_i) * loss_weights_i

    Callers are responsible for dividing by the (global) sum of ``loss_weights``
    to obtain the final per-token-averaged loss.

    :param logits: Predicted unnormalized logits with shape ``(N, vocab_size)``.
    :param labels: Ground truth class indices with shape ``(N,)``.
    :param loss_weights: Per-token float weights with shape ``(N,)``. Tokens with
        weight ``0`` do not contribute (their ``labels`` may be anything; pair
        with ``ignore_index`` for safety).
    :param ignore_index: Target value ignored by the underlying cross entropy.
    :param compute_z_loss: Compute the (weighted) softmax auxiliary loss as well.
    :param z_loss_multiplier: The multiplier to apply to the z-loss.

    :returns: The weighted cross entropy loss (a scalar) and optionally the
        weighted z-loss (a scalar).
    """
    logits = logits.float()
    loss_weights = loss_weights.view(-1).to(logits.dtype)
    per_token = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction="none")
    ce_loss = torch.dot(per_token, loss_weights)

    if not compute_z_loss:
        return ce_loss, None

    z_weights = (labels != ignore_index).to(logits.dtype) * loss_weights
    z_squared = logits.logsumexp(-1).pow(2)
    z_loss = torch.dot(z_squared, z_weights) * z_loss_multiplier

    return ce_loss, z_loss


_fused_linear_cross_entropy_loss: Optional[Callable] = None

try:
    from liger_kernel.ops.fused_linear_cross_entropy import (  # type: ignore
        LigerFusedLinearCrossEntropyFunction,
    )

    _fused_linear_cross_entropy_loss = LigerFusedLinearCrossEntropyFunction.apply
except ImportError:
    pass
except Exception:
    log.exception("Error importing liger-kernel")


@torch._dynamo.disable()
def fused_linear_cross_entropy_loss(
    _input: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: Literal["mean", "sum", "none"] = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
    ce_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    softcap: Optional[float] = None,
    accum_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Cross entropy loss fused with the linear layer that computes the logits, which avoids materialization
    of the large logits tensor. Additionally, this function computes gradients during the forward pass,
    (valid when CrossEntropyLoss comes last), so _input and labels do not need to be stored for the backwards pass.

    :param _input: The inputs to pass through the linear layer to produce the logits ``(N, D)``.
    :param weight: The weight of the linear layer.
    :param labels: Ground truth class indices with shape ``(N,)``.
    :param bias: Optional bias for the linear layer.
    :param ignore_index: Specifies a target value that is ignored and does not contribute to
        the input gradient.
    :param reduction: Specifies the reduction to apply to the output.
        Can be "none", "mean", or "sum".
    :param compute_z_loss: Compute the softmax auxiliary loss as well.
    :param z_loss_multiplier: The multiplier to apply to the z-loss.
    :param accum_dtype: The dtype of intermediate result buffers for weight and bias gradient
        accumulations. Recommended to set `accum_dtype` to higher precision, e.g. `torch.float32`,
        if the training is unstable with original dtype. Default to performing accumulations in original dtype.

    :returns: The cross entropy loss and optionally the z-loss.
    """
    if _fused_linear_cross_entropy_loss is None:
        raise RuntimeError("'fused_linear_cross_entropy_loss' requires liger-kernel")
    ce_loss, z_loss, per_token_acc = _fused_linear_cross_entropy_loss(
        _input,
        weight,
        labels,
        bias,
        ce_weight,
        ignore_index,
        z_loss_multiplier,
        label_smoothing,
        reduction,
        softcap,
        compute_z_loss,
        accum_dtype,
    )
    del per_token_acc
    if compute_z_loss:
        return ce_loss, z_loss
    else:
        return ce_loss, None
