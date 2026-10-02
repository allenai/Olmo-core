"""
Multimodal extensions of :class:`~olmo_core.optim.moe_optimizer.OLMoDDPOptimizer`.

A multimodal model trains a vision encoder, a connector and a language model whose gradients
live on very different scales, so the alignment recipe clips each logical scheduler group on its
own and reports per-component gradient norms. It also loads pretrained weights into the model
*after* the optimizer has materialized its FP32 masters, so it needs to resynchronize only the
parameters it touched. None of this changes the text-only optimizer: everything here lives in a
subclass that the text recipes never build.
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.optim.moe_optimizer import (
    OLMoDDPOptimizer,
    OLMoDDPOptimizerConfig,
    _assert_finite_async,
    _is_fp8_weight_store,
    assign_full_tensor_to_dtensor,
)

__all__ = ["MultimodalOLMoDDPOptimizer", "MultimodalOLMoDDPOptimizerConfig"]

log = logging.getLogger(__name__)

_DEFAULT_CLIP_GROUP = "<default>"


class MultimodalOLMoDDPOptimizer(OLMoDDPOptimizer):
    """
    :class:`~olmo_core.optim.moe_optimizer.OLMoDDPOptimizer` with per-scheduler-group gradient
    clipping, per-component gradient-norm diagnostics, a configurable foreach chunk size and
    partial model-to-master parameter synchronization.

    :param clip_grad_norm_by_scheduler_group: Clip each logical scheduler group independently.
        Physical DP and EP parameter groups with the same ``scheduler_name`` are combined before
        clipping. Parameters without a scheduler name form a private fallback group.
    :param foreach_chunk_size: Maximum number of local parameter elements updated by one foreach
        AdamW call. A smaller value reduces transient optimizer memory at the cost of launching
        more kernels.
    """

    DEFAULT_CLIP_GROUP_NAME = _DEFAULT_CLIP_GROUP

    def __init__(
        self,
        *args,
        clip_grad_norm_by_scheduler_group: bool = False,
        foreach_chunk_size: int = 600_000_000,
        **kwargs,
    ):
        if foreach_chunk_size <= 0:
            raise ValueError("foreach_chunk_size must be positive")
        super().__init__(*args, **kwargs)
        if (
            clip_grad_norm_by_scheduler_group
            and self.dense_mesh.mesh_dim_names is not None
            and "pp" in self.dense_mesh.mesh_dim_names
        ):
            raise OLMoConfigurationError(
                "Scheduler-group gradient clipping does not yet support pipeline parallelism"
            )
        self.clip_grad_norm_by_scheduler_group = clip_grad_norm_by_scheduler_group
        # Read by ``OLMoDDPOptimizer._step_foreach`` when flushing foreach chunks.
        self._foreach_chunk_threshold = foreach_chunk_size
        self.latest_component_grad_norms: Dict[str, torch.Tensor] = {}
        self.latest_clip_group_grad_norms: Dict[str, torch.Tensor] = {}
        self.latest_clip_group_coefficients: Dict[str, torch.Tensor] = {}
        self._component_grad_norm_patterns: Optional[Dict[str, Tuple[str, ...]]] = None

    @property
    def foreach_chunk_size(self) -> int:
        """Maximum number of local parameter elements updated by one foreach AdamW call."""
        return self._foreach_chunk_threshold

    def set_component_grad_norm_patterns(
        self, patterns: Optional[Dict[str, Tuple[str, ...]]]
    ) -> None:
        """
        Configure optional named-parameter patterns for the next gradient-norm report.

        :param patterns: Mapping from metric component name to ``fnmatch`` patterns, or ``None``
            to disable component diagnostics. The total clipping norm is unchanged.
        """
        if patterns is not None:
            for component, component_patterns in patterns.items():
                if not component or not component_patterns:
                    raise ValueError("Component gradient-norm patterns must be non-empty")
        self._component_grad_norm_patterns = patterns

    def _partition_main_grads(
        self, param_names: Optional[Set[str]] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        dp_grads_replicated: List[torch.Tensor] = []
        dp_grads_sharded: List[torch.Tensor] = []
        ep_dp_grads_replicated: List[torch.Tensor] = []
        ep_dp_grads_sharded: List[torch.Tensor] = []

        for param_group in self.param_groups:
            for name, param in param_group["named_params"].items():
                if not param.requires_grad or (param_names is not None and name not in param_names):
                    continue
                placements = self.states[f"{name}.main"].placements
                assert len(placements) == 1, "Expect only one placement per tensor"
                main_grad = self.main_grad[name]

                if param_group["pg"] == "dp":
                    if placements[0].is_shard():
                        dp_grads_sharded.append(main_grad)
                    else:
                        dp_grads_replicated.append(main_grad)
                elif param_group["pg"] == "ep_dp":
                    if placements[0].is_shard():
                        ep_dp_grads_sharded.append(main_grad)
                    else:
                        ep_dp_grads_replicated.append(main_grad)
        return (
            dp_grads_replicated,
            dp_grads_sharded,
            ep_dp_grads_replicated,
            ep_dp_grads_sharded,
        )

    def _compute_component_grad_norms(self) -> Dict[str, torch.Tensor]:
        patterns = self._component_grad_norm_patterns
        if patterns is None:
            return {}

        norms: Dict[str, torch.Tensor] = {}
        all_names = {
            name
            for param_group in self.param_groups
            for name, param in param_group["named_params"].items()
            if param.requires_grad
        }
        for component, component_patterns in patterns.items():
            names = {
                name
                for name in all_names
                if any(fnmatch(name, pattern) for pattern in component_patterns)
            }
            if not names:
                raise ValueError(
                    f"No trainable optimizer parameters match component {component!r} patterns "
                    f"{component_patterns!r}"
                )
            norms[component] = self._compute_total_grad_norm(*self._partition_main_grads(names))
        return norms

    def _logical_grad_clip_groups(self) -> "OrderedDict[str, List[str]]":
        """Collect parameter names into logical groups shared across DP and EP partitions."""
        groups: "OrderedDict[str, List[str]]" = OrderedDict()
        for param_group in self.param_groups:
            group_name = param_group.get("scheduler_name") or _DEFAULT_CLIP_GROUP
            names = groups.setdefault(group_name, [])
            names.extend(
                name for name, param in param_group["named_params"].items() if param.requires_grad
            )
        return groups

    def _clip_grad(self) -> torch.Tensor:
        """
        Clip gradients globally, or independently per logical scheduler group when
        ``clip_grad_norm_by_scheduler_group`` is set. See the parent method for how the norms are
        reduced across the DP, EP and PP meshes.
        """
        self.latest_component_grad_norms = self._compute_component_grad_norms()
        self.latest_clip_group_grad_norms = {}
        self.latest_clip_group_coefficients = {}
        if not self.clip_grad_norm_by_scheduler_group:
            return super()._clip_grad()

        logical_groups = self._logical_grad_clip_groups()
        ordered_names = [name for names in logical_groups.values() for name in names]
        if len(ordered_names) != len(set(ordered_names)) or set(ordered_names) != set(
            self.main_grad
        ):
            raise RuntimeError(
                "Logical gradient-clip groups must be a disjoint, exhaustive partition of "
                "optimizer gradients"
            )
        for group_name, names in logical_groups.items():
            self.latest_clip_group_grad_norms[group_name] = self._compute_total_grad_norm(
                *self._partition_main_grads(set(names))
            )
        total_grad_norm = torch.linalg.vector_norm(
            torch.stack(list(self.latest_clip_group_grad_norms.values())), ord=2
        )

        self._maybe_debug_nan_inf_grad_norm(total_grad_norm, *self._partition_main_grads())
        if self.check_nan_inf_grad:
            _assert_finite_async(total_grad_norm, "total grad norm")

        for group_name, names in logical_groups.items():
            group_norm = self.latest_clip_group_grad_norms[group_name]
            clip_coefficient = torch.clamp(self.max_grad_norm / (group_norm + 1e-6), max=1.0).to(
                group_norm.device
            )
            torch._foreach_mul_([self.main_grad[name] for name in names], clip_coefficient)
            self.latest_clip_group_coefficients[group_name] = clip_coefficient
        return total_grad_norm

    @torch.no_grad()
    def _copy_model_params_to_main_params(  # type: ignore[override]
        self, param_names: Optional[Set[str]] = None
    ) -> None:
        """
        Copy current model weights into the optimizer-owned FP32 main parameters.

        :param param_names: Optimizer parameter names to copy, e.g. after loading pretrained
            weights into one component. ``None`` copies every parameter, exactly like the parent.

        :raises KeyError: If a requested name is not an optimizer parameter.
        """
        if param_names is None:
            super()._copy_model_params_to_main_params()
            return
        copied: Set[str] = set()
        for param_group in self.param_groups:
            for name, param in param_group["named_params"].items():
                if name not in param_names:
                    continue
                if self.should_maintain_fp32_main_param:
                    assign_full_tensor_to_dtensor(
                        dst=self.states[f"{name}.main"],
                        src=param.data.float().reshape(-1),
                    )
                copied.add(name)
        if copied != param_names:
            missing = sorted(param_names - copied)
            raise KeyError(f"Optimizer does not contain requested parameter(s): {missing}")
        self._copy_main_params_to_mxfp8_weights()
        self._refresh_rowwise_fp8_caches_from_model_params()

    @torch.no_grad()
    def _copy_model_param_rows_to_main_params(
        self, param_names: Set[str], row_indices: List[int]
    ) -> None:
        """
        Copy selected rows of 2-D model parameters (e.g. freshly initialized embedding rows)
        into their FP32 main parameters.

        :raises KeyError: If a requested name is not an optimizer parameter.
        :raises ValueError: If a parameter is not a plain 2-D tensor with a flat main parameter.
        """
        copied: Set[str] = set()
        for param_group in self.param_groups:
            for name, param in param_group["named_params"].items():
                if name not in param_names:
                    continue
                if _is_fp8_weight_store(param) or param.ndim < 2:
                    raise ValueError(f"Cannot copy rows for optimizer parameter '{name}'")

                main_param = self.states[f"{name}.main"]
                if main_param.ndim != 1 or main_param.numel() != param.numel():
                    raise ValueError(
                        f"Expected a flat optimizer main parameter for '{name}', got "
                        f"shape {tuple(main_param.shape)}"
                    )

                _, global_offset = compute_local_shape_and_global_offset(
                    main_param.shape,
                    main_param.device_mesh,
                    main_param.placements,
                )
                local_main = main_param.to_local().reshape(-1)
                local_start = global_offset[0]
                local_end = local_start + local_main.numel()
                row_width = param.numel() // param.shape[0]
                flat_param = param.data.reshape(-1)

                for row in row_indices:
                    row_start = row * row_width
                    row_end = row_start + row_width
                    overlap_start = max(row_start, local_start)
                    overlap_end = min(row_end, local_end)
                    if overlap_start < overlap_end:
                        local_main[overlap_start - local_start : overlap_end - local_start].copy_(
                            flat_param[overlap_start:overlap_end]
                        )
                copied.add(name)

        if copied != param_names:
            missing = sorted(param_names - copied)
            raise KeyError(f"Optimizer does not contain requested parameter(s): {missing}")

    def _check_model_param_main_param_the_same(  # type: ignore[override]
        self, param_names: Optional[Set[str]] = None
    ) -> None:
        """
        Check that model parameters match their optimizer-owned FP32 masters.

        :param param_names: Optimizer parameter names to check. ``None`` checks every parameter.

        :raises KeyError: If a requested name is not an optimizer parameter.
        :raises ValueError: If a model parameter and its master are not close.
        """
        if param_names is None:
            super()._check_model_param_main_param_the_same()
            return
        checked: Set[str] = set()
        for param_group in self.param_groups:
            for name, param in param_group["named_params"].items():
                if name not in param_names:
                    continue
                main_param = self.states[f"{name}.main"]
                main_param_full = main_param.full_tensor().reshape(-1)
                model_param = param.data.float().reshape(-1)
                if not torch.allclose(model_param, main_param_full, atol=1e-5):
                    raise ValueError(
                        f"{name}: Model param {param} and main param {main_param} are not close"
                    )
                checked.add(name)
        if checked != param_names:
            missing = sorted(param_names - checked)
            raise KeyError(f"Optimizer does not contain requested parameter(s): {missing}")


@dataclass
class MultimodalOLMoDDPOptimizerConfig(OLMoDDPOptimizerConfig):
    """
    Configuration for :class:`MultimodalOLMoDDPOptimizer`. Every field of
    :class:`~olmo_core.optim.moe_optimizer.OLMoDDPOptimizerConfig` keeps its meaning and default.
    """

    clip_grad_norm_by_scheduler_group: bool = False
    """
    Clip gradients independently for each logical scheduler group. Physical DP and EP parameter
    groups with the same ``scheduler_name`` are combined before clipping. Parameters without a
    scheduler name form a private fallback group.
    """

    foreach_chunk_size: int = 600_000_000
    """
    Maximum number of local parameter elements updated by one foreach AdamW call. A smaller
    value reduces transient optimizer memory at the cost of launching more foreach kernels.
    """

    @classmethod
    def optimizer(cls):
        return MultimodalOLMoDDPOptimizer

    def build(self, *args: Any, **kwargs: Any) -> "MultimodalOLMoDDPOptimizer":  # type: ignore[override]
        """Build the optimizer; see :meth:`OLMoDDPOptimizerConfig.build`."""
        optim = super().build(*args, **kwargs)
        assert isinstance(optim, MultimodalOLMoDDPOptimizer)
        return optim
