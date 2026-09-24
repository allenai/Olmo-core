import logging
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Partial, Placement, Replicate, Shard
from torch.distributed.tensor.parallel import (
    PrepareModuleInput,
    PrepareModuleOutput,
    parallelize_module,
)

from olmo_core.distributed.utils import get_local_tensor

from .functional import cross_entropy_loss

log = logging.getLogger(__name__)


class _CELossFnWrapper(nn.Module):
    def __init__(
        self,
        ignore_index: int = -100,
        reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction: Literal["mean", "sum", "none"] = reduction
        self.z_loss_multiplier = z_loss_multiplier
        self.base_loss_fn = cross_entropy_loss
        self.tp_enabled = False

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if logits.shape[:-1] != labels.shape:
            raise RuntimeError(
                f"expected labels to have shape {logits.shape[:-1]}, but found {tuple(labels.shape)} instead"
            )

        # shape: (B * S, V)
        logits_for_loss = logits.view(-1, logits.shape[-1])

        # shape: (B, S) -> (B * S,)
        labels_for_loss = labels.view(-1)

        ce_loss, z_loss = self.base_loss_fn(
            logits_for_loss,
            labels_for_loss,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            compute_z_loss=self.z_loss_multiplier is not None,
            z_loss_multiplier=self.z_loss_multiplier or 1e-4,
        )

        if self.reduction == "none":
            ce_loss = ce_loss.view(labels.shape)
            if z_loss is not None:
                z_loss = z_loss.view(labels.shape)
        elif self.tp_enabled:
            ce_loss = ce_loss.unsqueeze(0)
            if z_loss is not None:
                z_loss = z_loss.unsqueeze(0)

        return ce_loss, z_loss


class CrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss.
    """

    def __init__(
        self,
        *,
        ignore_index: int = -100,
        reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        compile: bool = False,
    ):
        super().__init__()

        self.loss_fn = _CELossFnWrapper(
            ignore_index=ignore_index,
            reduction=reduction,
            z_loss_multiplier=z_loss_multiplier,
        )
        self._tp_enabled = False

        self._compile_enabled = False
        if compile:
            if torch.cuda.is_available():
                log.info("Compiling loss function...")
                self.compile()
                self._compile_enabled = True
            else:
                log.warning("Skipping loss compilation since CUDA is not available")

    @property
    def compile_enabled(self) -> bool:
        return self._compile_enabled

    @property
    def tp_enabled(self) -> bool:
        return self._tp_enabled

    @property
    def z_loss_enabled(self) -> bool:
        return self.loss_fn.z_loss_multiplier is not None

    @property
    def reduction(self) -> Literal["sum", "mean", "none"]:
        return self.loss_fn.reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        div_factor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute the CE loss and optionally Z-loss.

        :param logits: The logits of shape ``(B, S, V)``.
        :param labels: The target labels of shape ``(B, S)``.
        """
        ce_loss, z_loss = self.loss_fn(get_local_tensor(logits), get_local_tensor(labels))

        if self.reduction != "none" and ce_loss.numel() > 1:
            # This will be the same case with tensor/sequence parallel loss where we have a DTensor.
            assert self.tp_enabled
            if self.reduction == "sum":
                ce_loss = ce_loss.sum()
                if z_loss is not None:
                    z_loss = z_loss.sum()
            elif self.reduction == "mean":
                ce_loss = ce_loss.mean()
                if z_loss is not None:
                    z_loss = z_loss.mean()
            else:
                raise NotImplementedError(self.reduction)

        if div_factor is not None:
            ce_loss = ce_loss / div_factor
            if z_loss is not None:
                z_loss = z_loss / div_factor

        return ce_loss, z_loss

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layouts: Optional[Tuple[Placement, ...]] = None,
        shard_dimension: int = 1,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = False,
    ):
        desired_input_layouts: Tuple[Placement, ...]
        if input_layouts is None or len(input_layouts) == 3:
            desired_input_layouts = (Shard(shard_dimension), Shard(shard_dimension), Replicate())
        elif len(input_layouts) == 2:
            desired_input_layouts = (Shard(shard_dimension), Shard(shard_dimension))
        else:
            raise ValueError(f"expected 2 or 3 input layouts, found {len(input_layouts)}")
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
                input_layouts=input_layouts,  # type: ignore
                desired_input_layouts=desired_input_layouts,  # type: ignore
                use_local_output=False,
            ),
        )

        inner_output_layout = Shard(shard_dimension) if self.reduction == "none" else Shard(0)
        parallelize_module(
            self.loss_fn,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleOutput(
                output_layouts=(  # type: ignore
                    inner_output_layout,
                    None if not self.z_loss_enabled else inner_output_layout,
                ),
                desired_output_layouts=(  # type: ignore
                    inner_output_layout,
                    None if not self.z_loss_enabled else inner_output_layout,
                ),
                use_local_output=False,
            ),
        )

        # Reducing a sharded loss produces a partial sum, which must be reduced across ranks
        # before returning a replicated (or local) loss. Declaring it replicated skips that step.
        expected_output_layout = Shard(shard_dimension) if self.reduction == "none" else Partial()
        desired_output_layout = output_layout or Replicate()
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleOutput(
                output_layouts=(  # type: ignore
                    expected_output_layout,
                    None if not self.z_loss_enabled else expected_output_layout,
                ),
                desired_output_layouts=(  # type: ignore
                    desired_output_layout,
                    None if not self.z_loss_enabled else desired_output_layout,
                ),
                use_local_output=use_local_output,
            ),
        )

        self._tp_enabled = True
        self.loss_fn.tp_enabled = True
