from typing import Any, Dict, Iterable, List, Optional, Union

import torch
from torch.optim.optimizer import Optimizer
from typing_extensions import TypeAlias

from olmo_core.utils import get_default_device, move_to_device

ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class SkipStepOptimizer(Optimizer):
    """
    A :class:`SkipStepOptimizer` is an optimizer that can skip updates when the loss or gradient
    norm for a step is above a certain threshold of standard deviations computed over a rolling
    interval.

    .. important::
        When using a :class:`SkipStepOptimizer` you must always set :data:`latest_loss` and
        :data:`latest_grad_norm` to the current loss and grad norm, respectively, *before* calling
        :meth:`step()`.

        The :class:`~olmo_core.train.train_module.TransformerTrainModule` will automatically set
        the :data:`latest_loss` and :data:`latest_grad_norm` whenever its optimizer is a subclass of
        :class:`SkipStepOptimizer`.

    .. tip::
        When implementing a :class:`SkipStepOptimizer` you should be careful to avoid host-device
        syncs. You can use :meth:`get_step_factor()` within your :meth:`step()` method to do this.
        See the implementation of :class:`SkipStepLion` for an example.
    """

    def __init__(
        self,
        params: ParamsT,
        defaults: Dict[str, Any],
        rolling_interval_length: int = 128,
        sigma_factor: int = 6,
    ) -> None:
        super().__init__(params, defaults)
        self.rolling_interval_length = rolling_interval_length
        self.sigma_factor = sigma_factor
        self._losses: List[torch.Tensor] = []
        self._grad_norms: List[torch.Tensor] = []
        self._device: Optional[torch.device] = None

    @property
    def device(self) -> torch.device:
        if self._device is None:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.numel() > 0:
                        self._device = p.device
                        break
            if self._device is None:
                self._device = get_default_device()
        return self._device

    @property
    def latest_loss(self) -> Optional[torch.Tensor]:
        if not self._losses:
            return None
        else:
            return self._losses[-1]

    @latest_loss.setter
    def latest_loss(self, loss: torch.Tensor):
        self._losses.append(loss)
        while len(self._losses) > self.rolling_interval_length + 1:
            self._losses.pop(0)

    @property
    def latest_grad_norm(self) -> Optional[torch.Tensor]:
        if not self._grad_norms:
            return None
        else:
            return self._grad_norms[-1]

    @latest_grad_norm.setter
    def latest_grad_norm(self, grad_norm: torch.Tensor):
        self._grad_norms.append(grad_norm)
        while len(self._grad_norms) > self.rolling_interval_length + 1:
            self._grad_norms.pop(0)

    @torch._dynamo.disable()
    def get_step_factor(self) -> torch.Tensor:
        """
        Returns a float tensor which will be `1.0` if the optimizer should proceed with the step
        and `0.0` if the optimizer should skip the step.

        The tensor can be used within the optimizer's step computation to essentially skip a step
        without a host-device sync.
        """
        return compute_step_factor(
            self._losses,
            self._grad_norms,
            self.rolling_interval_length,
            self.sigma_factor,
            self.device,
        )

    @property
    def step_skipped(self) -> torch.Tensor:
        """
        Returns a float tensor which will be `1.0` if the step was skipped and `0.0` otherwise.
        """
        return 1 - self.get_step_factor()


def compute_step_factor(
    losses: List[torch.Tensor],
    grad_norms: List[torch.Tensor],
    rolling_interval_length: int,
    sigma_factor: int,
    device: torch.device,
) -> torch.Tensor:
    """Shared loss/gradient-spike policy for dense and OLMoDDP optimizers."""
    if len(losses) < max(2, rolling_interval_length // 2):
        return move_to_device(torch.tensor(1.0), device)

    loss_std, loss_mean = torch.std_mean(torch.stack(losses[:-1]))
    assert losses[-1] is not None
    if grad_norms:
        assert grad_norms[-1] is not None
        grad_norm_std, grad_norm_mean = torch.std_mean(torch.stack(grad_norms[:-1]))
        step_factor = torch.logical_and(
            (losses[-1] - loss_mean) <= sigma_factor * loss_std,
            (grad_norms[-1] - grad_norm_mean) <= sigma_factor * grad_norm_std,
        )
    else:
        step_factor = (losses[-1] - loss_mean) <= sigma_factor * loss_std

    return step_factor.float()
