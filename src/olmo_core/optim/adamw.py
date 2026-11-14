from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple, Type, Union

import torch

from ..config import DType
from ..distributed.utils import get_local_tensor
from .config import OptimConfig
from .skip_step_optimizer import SkipStepOptimizer


def adamw_step(
    p: torch.Tensor,
    grad: torch.Tensor,
    *,
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: torch.Tensor,
    step_factor: torch.Tensor,
    step_increment_bugfix: bool = True,
):
    beta1, beta2 = betas

    # Perform step weight decay.
    p.mul_(1 - step_factor * (lr * weight_decay))

    # Decay the first and second moment running average coefficient.
    exp_avg.lerp_(grad.type_as(exp_avg), (step_factor * (1 - beta1)).type_as(exp_avg))
    exp_avg_sq.mul_(1 - step_factor * (1 - beta2))
    exp_avg_sq.add_(step_factor * grad * grad, alpha=1 - beta2)

    bias_correction1 = 1 - beta1 ** (step + 1)
    bias_correction2 = 1 - beta2 ** (step + 1)

    step_size = lr / bias_correction1

    denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)

    update = -step_size * torch.div(exp_avg, denom)
    update.mul_(step_factor)
    p.add_(update)
    if step_increment_bugfix:
        step.add_(step_factor)


def foreach_adamw_step(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    exp_avgs: list[torch.Tensor],
    exp_avg_sqs: list[torch.Tensor],
    steps: list[torch.Tensor],
    *,
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    step_factor: torch.Tensor,
    step_increment_bugfix: bool = True,
):
    """Perform a single AdamW update with multi-tensor (*foreach*) kernels."""
    if not params:
        return  # nothing to do

    beta1, beta2 = betas

    # Perform step weight decay.
    torch._foreach_mul_(params, 1 - step_factor * (lr * weight_decay))

    grads = [g.type_as(ea) for g, ea in zip(grads, exp_avgs)]

    # Decay the first and second moment running average coefficient.
    # foreach_lerp_ has issues when DTensor is enabled (see https://github.com/pytorch/pytorch/issues/132017).
    # Implement the lerp(a, b, w) = a + w * (b - a) with basic _foreach_mul_/add_ ops instead:
    w1 = step_factor * (1 - beta1)
    torch._foreach_mul_(exp_avgs, 1.0 - w1)
    torch._foreach_add_(exp_avgs, torch._foreach_mul(grads, w1))

    grad_squares = torch._foreach_mul(grads, grads)

    w2 = step_factor * (1 - beta2)
    torch._foreach_mul_(exp_avg_sqs, 1.0 - w2)
    torch._foreach_add_(exp_avg_sqs, torch._foreach_mul(grad_squares, w2))

    steps_t = torch.stack(steps)
    bias_corrections1 = 1 - torch.pow(beta1, steps_t + 1)
    bias_corrections2 = 1 - torch.pow(beta2, steps_t + 1)

    step_sizes = lr / bias_corrections1

    denoms = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denoms, bias_corrections2.sqrt().unbind())
    torch._foreach_add_(denoms, eps)

    updates = torch._foreach_div(exp_avgs, denoms)
    torch._foreach_mul_(updates, (-step_factor * step_sizes).unbind())
    torch._foreach_add_(params, updates)
    if step_increment_bugfix:
        torch._foreach_add_(steps, [step_factor] * len(steps))


class SkipStepAdamW(SkipStepOptimizer):
    """
    A "skip step" version of :class:`AdamW`.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        rolling_interval_length: int = 128,
        sigma_factor: int = 6,
        dtype: Optional[Union[torch.dtype, DType]] = None,
        foreach: bool = False,
        step_increment_bugfix: bool = True,
    ) -> None:
        assert lr >= 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(
            params,
            defaults,
            rolling_interval_length=rolling_interval_length,
            sigma_factor=sigma_factor,
        )
        if isinstance(dtype, DType):
            dtype = dtype.as_pt()
        self.dtype = dtype
        self.foreach = foreach
        self.stepfix = step_increment_bugfix
        self._step_skipped: Optional[torch.Tensor] = None

    @property
    def step_skipped(self) -> torch.Tensor:
        if self._step_skipped is not None:
            return self._step_skipped
        else:
            return torch.tensor(0.0)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        if self.foreach:
            self._step_foreach(closure)
        else:
            self._step(closure)

    def _step(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()

        step_factor = self.get_step_factor()
        self._step_skipped = 1 - step_factor
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p, dtype=self.dtype)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=self.dtype)

                adamw_step(
                    get_local_tensor(p),
                    get_local_tensor(p.grad),
                    lr=group["lr"],
                    betas=group["betas"],
                    eps=group["eps"],
                    weight_decay=group["weight_decay"],
                    exp_avg=get_local_tensor(state["exp_avg"]),
                    exp_avg_sq=get_local_tensor(state["exp_avg_sq"]),
                    step=state["step"],
                    step_factor=step_factor,
                    step_increment_bugfix=self.stepfix,
                )

    def _step_foreach(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()

        step_factor = self.get_step_factor()
        self._step_skipped = 1 - step_factor
        for group in self.param_groups:
            params_with_grad: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            exp_avgs: list[torch.Tensor] = []
            exp_avg_sqs: list[torch.Tensor] = []
            steps_list = []  # create list outside loops

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p, dtype=self.dtype)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=self.dtype)

                params_with_grad.append(get_local_tensor(p))
                grads.append(get_local_tensor(p.grad))
                exp_avgs.append(get_local_tensor(state["exp_avg"]))
                exp_avg_sqs.append(get_local_tensor(state["exp_avg_sq"]))
                steps_list.append(state["step"])

            if not params_with_grad:
                continue  # nothing to update in this group

            foreach_adamw_step(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                steps_list,
                lr=group["lr"],
                betas=group["betas"],
                eps=group["eps"],
                weight_decay=group["weight_decay"],
                step_factor=step_factor,
                step_increment_bugfix=self.stepfix,
            )


@OptimConfig.register("adamw")
@dataclass
class AdamWConfig(OptimConfig[torch.optim.AdamW]):
    """
    Configuration class for building an :class:`torch.optim.AdamW` optimizer.
    """

    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    foreach: Optional[bool] = None
    fused: Optional[bool] = None

    _supports_completedp: ClassVar[bool] = True

    @classmethod
    def optimizer(cls) -> Type[torch.optim.AdamW]:
        return torch.optim.AdamW


@OptimConfig.register("skip_step_adamw")
@dataclass
class SkipStepAdamWConfig(OptimConfig[SkipStepAdamW]):
    """
    Configuration class for building a :class:`SkipStepAdamW` optimizer.
    """

    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    dtype: Optional[DType] = None
    foreach: bool = True
    """
    Whether to use multi-tensor (*foreach*) kernels for the AdamW update.
    Faster than the non-foreach version.
    """

    step_increment_bugfix: bool = True
    """
    Whether or not to fix the step-incrementing bug discovered in SkipStepAdamW.

    If this flag is set to False, the step will not be incremented, which
    gives the optimizer an effective lr that is 2.2x higher than the specified lr,
    and no bias correction is applied.
    """

    rolling_interval_length: int = 128
    """
    The length of the rolling interval to use for computing the mean and standard deviation of the loss.
    """

    sigma_factor: int = 6
    """
    The number of standard deviations above the mean loss to skip a step.
    """

    _supports_completedp: ClassVar[bool] = True

    @classmethod
    def optimizer(cls) -> Type[SkipStepAdamW]:
        return SkipStepAdamW
