"""Dense DTensor adapter for the same numerical Muon core used by OLMoDDP.

The default shared backend gathers full matrices in FP32. The optional Dion
backend uses native batched tasks, BF16 transport, and Polar Express instead.
"""

from dataclasses import dataclass

import torch
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata
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
        muon_backend="shared",
    ):
        if muon_backend not in {"shared", "dion"}:
            raise ValueError(f"Unknown Muon backend: {muon_backend}")
        if muon_backend == "dion":
            if rms != 0.2:
                raise ValueError("Dion uses fixed RMS scaling 0.2")
            from .dion_backend import require_dion

            require_dion()
        self.muon_backend = muon_backend
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
                muon_backend=muon_backend,
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
        self._sync_skip_history()

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
        if self.muon_backend == "dion":
            # Dion mutates momentum before orthogonalizing. Bypass the whole
            # update on skips, including AdamW and all step counters.
            if not bool(factor.item()):
                return loss
            from .dion_backend import dion_update

            for group in self.param_groups:
                params = [p for p in group["params"] if p.grad is not None]
                if not params:
                    continue
                if group["algorithm"] == "muon":
                    dion_update(
                        params,
                        [p.grad for p in params],
                        [self.state[p]["momentum"] for p in params],
                        lr=group["lr"],
                        momentum=group["momentum"],
                        weight_decay=group["weight_decay"],
                    )
                    for p in params:
                        self.state[p]["step"].add_(1)
                else:
                    foreach_adamw_step(
                        [_local(p) for p in params],
                        [_local(p.grad) for p in params],
                        [_local(self.state[p]["momentum"]) for p in params],
                        [_local(self.state[p]["variance"]) for p in params],
                        [self.state[p]["step"] for p in params],
                        lr=group["lr"],
                        betas=group["betas"],
                        eps=group["eps"],
                        weight_decay=group["weight_decay"],
                        step_factor=factor,
                    )
            return loss
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

    def _sync_skip_history(self) -> None:
        # DCP's flattened loader enumerates live param-group keys, not just state_dict().
        self.param_groups[0]["skip_history"] = dict(
            losses=self._losses, grad_norms=self._grad_norms
        )

    def prepare_for_checkpoint_load(self, metadata: Metadata) -> None:
        """Allocate skip-history buffers matching an existing DCP checkpoint.

        DCP flattens populated lists into indexed tensor keys, but an empty list
        into a single byte-storage key. A fresh optimizer therefore needs the
        saved list lengths before the train module constructs its load template.
        This preserves the existing checkpoint format and all saved history.
        """
        for field, attr in (("losses", "_losses"), ("grad_norms", "_grad_norms")):
            histories = {}
            marker = f".skip_history.{field}"
            for key, storage in metadata.state_dict_metadata.items():
                if not key.startswith("optim.param_groups.") or marker not in key:
                    continue
                prefix, suffix = key.rsplit(marker, 1)
                entries = histories.setdefault(prefix, {})
                if suffix:
                    if not suffix.startswith(".") or not suffix[1:].isdigit():
                        raise ValueError(f"Invalid skip-history checkpoint key: {key}")
                    if not isinstance(storage, TensorStorageMetadata):
                        raise ValueError(f"Expected tensor skip-history entry: {key}")
                    entries[int(suffix[1:])] = storage
            if not histories:
                raise ValueError(f"Checkpoint is missing Muon skip history: {field}")
            entries = next(iter(histories.values()))
            if any(other != entries for other in histories.values()):
                raise ValueError(f"Inconsistent Muon skip-history metadata: {field}")
            if sorted(entries) != list(range(len(entries))):
                raise ValueError(f"Non-contiguous Muon skip history: {field}")
            setattr(
                self,
                attr,
                [
                    torch.empty(
                        entries[i].size, dtype=entries[i].properties.dtype, device=self.device
                    )
                    for i in range(len(entries))
                ],
            )
        self._sync_skip_history()

    def state_dict(self):
        # DCP keeps state/param_groups but drops arbitrary top-level keys.
        self._sync_skip_history()
        return super().state_dict()

    def load_state_dict(self, state_dict):
        for group in state_dict["param_groups"]:
            if group.get("muon_backend", "shared") != self.muon_backend:
                raise ValueError("Checkpoint Muon backend differs; start a new optimizer recipe")
        state_dict = dict(state_dict)
        state_dict["param_groups"] = [dict(group) for group in state_dict["param_groups"]]
        history = state_dict["param_groups"][0].pop("skip_history", dict(losses=[], grad_norms=[]))
        super().load_state_dict(state_dict)
        self._losses = [v.to(self.device) for v in history["losses"]]
        self._grad_norms = [v.to(self.device) for v in history["grad_norms"]]
        self._sync_skip_history()
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
    muon_backend: str = "shared"
    """``shared`` preserves the reference recipe; ``dion`` uses batched Polar Express."""

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
