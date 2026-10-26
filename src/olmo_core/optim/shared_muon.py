"""Dense DTensor adapter for the same numerical Muon core used by OLMoDDP.

This correctness-first adapter gathers full matrices in FP32. It does not use
Dion's BF16 transport or independently implement the update equations.
"""

from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from .adamw import foreach_adamw_step
from .config import OptimConfig
from .muon_math import apply_muon_update, muon_direction, muon_momentum, uses_muon
from .skip_step_optimizer import SkipStepOptimizer


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _full(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _local_slice(full: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if isinstance(target, DTensor):
        shape, offset = compute_local_shape_and_global_offset(
            target.shape, target.device_mesh, target.placements
        )
        for axis, (start, length) in enumerate(zip(offset, shape)):
            full = full.narrow(axis, start, length)
    return full.contiguous()


class SharedMuon(SkipStepOptimizer):
    """Muon plus auxiliary AdamW on FP32 parameters, optionally HSDP/FSDP DTensors.

    Mixed precision forward/backward is supported through FSDP's parameter
    policy; persistent parameters must be FP32, as in the dense ladder. The
    train module supplies reduced loss and clipped gradient norm through the
    standard SkipStepOptimizer interface.
    """

    def __init__(
        self,
        params,
        lr=0.003,
        momentum=0.95,
        rms=0.2,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        rolling_interval_length=128,
        sigma_factor=6,
    ):
        if lr < 0 or not 0 <= momentum < 1 or rms <= 0:
            raise ValueError("Invalid Muon hyperparameters")
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                rms=rms,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
                algorithm="muon",
            ),
            rolling_interval_length=rolling_interval_length,
            sigma_factor=sigma_factor,
        )
        self._step_skipped = None
        for group in self.param_groups:
            if group["algorithm"] not in {"muon", "adamw"}:
                raise ValueError("Expected muon or adamw parameter group")
            for param in group["params"]:
                if param.dtype != torch.float32:
                    raise ValueError("SharedMuon requires persistent FP32 parameters")
                if group["algorithm"] == "muon" and param.ndim != 2:
                    raise ValueError("Dense Muon groups require 2D matrices")
                state = self.state[param]
                state["momentum"] = torch.zeros_like(param)
                state["step"] = torch.zeros((), device=param.device)
                if group["algorithm"] == "adamw":
                    state["variance"] = torch.zeros_like(param)

    @property
    def step_skipped(self):
        return (
            self._step_skipped
            if self._step_skipped is not None
            else torch.zeros((), device=self.device)
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        factor = self.get_step_factor()
        self._step_skipped = 1 - factor
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if group["algorithm"] == "muon":
                    momentum, direction = muon_momentum(
                        _full(param.grad).flatten(),
                        _full(state["momentum"]).flatten(),
                        group["momentum"],
                    )
                    update = muon_direction("weight", direction.reshape(param.shape), group["rms"])
                    apply_muon_update(
                        _local(param),
                        _local(state["momentum"]),
                        state["step"],
                        _local_slice(momentum.reshape(param.shape), param),
                        _local_slice(update, param),
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        step_factor=factor,
                    )
                else:
                    foreach_adamw_step(
                        [_local(param)],
                        [_local(param.grad)],
                        [_local(state["momentum"])],
                        [_local(state["variance"])],
                        [state["step"]],
                        lr=group["lr"],
                        betas=group["betas"],
                        eps=group["eps"],
                        weight_decay=group["weight_decay"],
                        step_factor=factor,
                    )
        return loss

    def state_dict(self):
        result = super().state_dict()
        # DCP keeps state/param_groups but drops arbitrary top-level keys.
        result["param_groups"][0]["skip_history"] = dict(
            losses=self._losses, grad_norms=self._grad_norms
        )
        return result

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        state_dict["param_groups"] = [dict(group) for group in state_dict["param_groups"]]
        history = state_dict["param_groups"][0].pop("skip_history", dict(losses=[], grad_norms=[]))
        super().load_state_dict(state_dict)
        self._losses = [v.to(self.device) for v in history["losses"]]
        self._grad_norms = [v.to(self.device) for v in history["grad_norms"]]
        self._step_skipped = None


@OptimConfig.register("shared_muon")
@dataclass
class SharedMuonConfig(OptimConfig[SharedMuon]):
    """Dense-ladder configuration with complete Muon/AdamW parameter routing."""

    lr: float = 0.003
    momentum: float = 0.95
    rms: float = 0.2
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    rolling_interval_length: int = 128
    sigma_factor: int = 6

    @classmethod
    def optimizer(cls):
        return SharedMuon

    def build_groups(self, model, strict=True):
        if self.group_overrides is not None or self.compile:
            raise ValueError("SharedMuon uses fixed routing and an eager shared numerical core")
        buckets: dict[str, list[torch.Tensor]] = {"muon": [], "adamw": [], "embeddings": []}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            key = (
                "embeddings"
                if "embeddings" in name.split(".")
                else ("muon" if uses_muon(name, tuple(param.shape)) else "adamw")
            )
            buckets[key].append(param)
        return [
            dict(
                params=params,
                algorithm="muon" if key == "muon" else "adamw",
                weight_decay=0.0 if key == "embeddings" else self.weight_decay,
            )
            for key, params in buckets.items()
            if params
        ]
