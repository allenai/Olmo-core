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
from dataclasses import dataclass, field
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
    _to_local_tensor,
    assign_full_tensor_to_dtensor,
)

__all__ = ["MultimodalOLMoDDPOptimizer", "MultimodalOLMoDDPOptimizerConfig"]

log = logging.getLogger(__name__)

_DEFAULT_CLIP_GROUP = "<default>"

# Parameter names split the way ``OLMoDDPOptimizer._compute_total_grad_norm`` wants its gradients:
# DP replicated, DP sharded, EP-DP replicated, EP-DP sharded.
_NamePartition = Tuple[List[str], List[str], List[str], List[str]]


def _empty_partition() -> _NamePartition:
    return ([], [], [], [])


@dataclass
class _GradNormPlan:
    """
    Gradient-norm bookkeeping resolved once per set of trainable parameters: the mesh partition
    each parameter's main gradient belongs to, the logical clip groups and, once requested, the
    parameters behind each diagnostic component. Only the names are cached; the main gradients
    themselves are rebuilt by the optimizer every step.
    """

    trainable_names: Tuple[str, ...]
    partition_index: Dict[str, int]
    all_params: _NamePartition
    clip_groups: "OrderedDict[str, _NamePartition]"
    component_patterns: Optional[Dict[str, Tuple[str, ...]]] = None
    components: Dict[str, _NamePartition] = field(default_factory=dict)


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
        self._grad_norm_plan_cache: Optional[_GradNormPlan] = None
        self._step_local_norms: Optional[Dict[int, Optional[torch.Tensor]]] = None

    @property
    def foreach_chunk_size(self) -> int:
        """Maximum number of local parameter elements updated by one foreach AdamW call."""
        return self._foreach_chunk_threshold

    def set_component_grad_norm_patterns(
        self, patterns: Optional[Dict[str, Tuple[str, ...]]]
    ) -> None:
        """
        Configure optional named-parameter patterns for the next gradient-norm report.

        The patterns are matched against the trainable parameters once and the result is cached,
        so setting the same patterns again (after a step without them) costs nothing.

        :param patterns: Mapping from metric component name to ``fnmatch`` patterns, or ``None``
            to disable component diagnostics. The total clipping norm is unchanged.
        """
        if patterns is not None:
            for component, component_patterns in patterns.items():
                if not component or not component_patterns:
                    raise ValueError("Component gradient-norm patterns must be non-empty")
        self._component_grad_norm_patterns = patterns

    def _trainable_param_names(self) -> Tuple[str, ...]:
        """Names of the optimizer parameters that receive gradients, in parameter-group order."""
        return tuple(
            name
            for param_group in self.param_groups
            for name, param in param_group["named_params"].items()
            if param.requires_grad
        )

    def _grad_norm_plan(self) -> _GradNormPlan:
        """
        Return the gradient-norm plan for the current set of trainable parameters, resolving it
        once and reusing it until a parameter is frozen or unfrozen.

        :raises RuntimeError: If the logical clip groups are not a disjoint, exhaustive partition
            of the optimizer gradients.
        """
        trainable_names = self._trainable_param_names()
        plan = self._grad_norm_plan_cache
        if plan is not None and plan.trainable_names == trainable_names:
            return plan

        partition_index: Dict[str, int] = {}
        all_params = _empty_partition()
        clip_groups: "OrderedDict[str, _NamePartition]" = OrderedDict()
        for param_group in self.param_groups:
            group_name = param_group.get("scheduler_name") or _DEFAULT_CLIP_GROUP
            group_partition = clip_groups.setdefault(group_name, _empty_partition())
            for name, param in param_group["named_params"].items():
                if not param.requires_grad:
                    continue
                placements = self.states[f"{name}.main"].placements
                assert len(placements) == 1, "Expect only one placement per tensor"
                if param_group["pg"] == "dp":
                    index = 1 if placements[0].is_shard() else 0
                elif param_group["pg"] == "ep_dp":
                    index = 3 if placements[0].is_shard() else 2
                else:
                    raise RuntimeError(f"Unknown pg tag: {param_group['pg']}")
                if name in partition_index:
                    raise RuntimeError(f"Optimizer parameter {name!r} appears in two groups")
                partition_index[name] = index
                all_params[index].append(name)
                group_partition[index].append(name)
        if set(trainable_names) != set(self.main_grad):
            raise RuntimeError(
                "Logical gradient-clip groups must be a disjoint, exhaustive partition of "
                "optimizer gradients"
            )
        plan = _GradNormPlan(
            trainable_names=trainable_names,
            partition_index=partition_index,
            all_params=all_params,
            clip_groups=clip_groups,
        )
        self._grad_norm_plan_cache = plan
        return plan

    def _component_partitions(self, plan: _GradNormPlan) -> Dict[str, _NamePartition]:
        """
        Resolve the configured component patterns against the plan's trainable parameters, once
        per distinct set of patterns.

        :raises ValueError: If a component matches no trainable parameter.
        """
        patterns = self._component_grad_norm_patterns
        if patterns is None:
            return {}
        if plan.component_patterns == patterns:
            return plan.components
        components: Dict[str, _NamePartition] = {}
        for component, component_patterns in patterns.items():
            partition = _empty_partition()
            for name in plan.trainable_names:
                if any(fnmatch(name, pattern) for pattern in component_patterns):
                    partition[plan.partition_index[name]].append(name)
            if not any(partition):
                raise ValueError(
                    f"No trainable optimizer parameters match component {component!r} patterns "
                    f"{component_patterns!r}"
                )
            components[component] = partition
        plan.component_patterns = dict(patterns)
        plan.components = components
        return components

    def _main_grads(self, partition: _NamePartition) -> Tuple[List[torch.Tensor], ...]:
        """Materialize a name partition as the main-gradient lists the parent norm expects."""
        return tuple([self.main_grad[name] for name in names] for names in partition)

    def _compute_component_grad_norms(self) -> Dict[str, torch.Tensor]:
        components = self._component_partitions(self._grad_norm_plan())
        return {
            component: self._compute_total_grad_norm(*self._main_grads(partition))
            for component, partition in components.items()
        }

    def _logical_grad_clip_groups(self) -> "OrderedDict[str, List[str]]":
        """Collect parameter names into logical groups shared across DP and EP partitions."""
        return OrderedDict(
            (group_name, [name for names in partition for name in names])
            for group_name, partition in self._grad_norm_plan().clip_groups.items()
        )

    def _local_total_norm(self, grads: List[torch.Tensor]) -> torch.Tensor:
        """
        The parent method with a per-step memo of the per-parameter norms, so that the clip
        groups and the component diagnostics share one norm kernel per parameter. Outside
        :meth:`_clip_grad` the memo is off and this is exactly the parent method.
        """
        memo = self._step_local_norms
        if memo is None:
            return super()._local_total_norm(grads)
        norms: List[torch.Tensor] = []
        for grad in grads:
            # Main gradients are distinct tensors that stay alive for the whole step.
            key = id(grad)
            try:
                norm = memo[key]
            except KeyError:
                local_grad = _to_local_tensor(grad)
                norm = (
                    None
                    if local_grad.numel() == 0
                    else torch.linalg.vector_norm(local_grad.detach().float(), ord=2)
                )
                memo[key] = norm
            if norm is not None:
                norms.append(norm)
        if not norms:
            return torch.zeros((), device=self.device, dtype=torch.float32)
        return torch.linalg.vector_norm(torch.stack(norms), ord=2)

    def _clip_grad(self) -> torch.Tensor:
        """
        Clip gradients globally, or independently per logical scheduler group when
        ``clip_grad_norm_by_scheduler_group`` is set. See the parent method for how the norms are
        reduced across the DP, EP and PP meshes. Component gradient norms are computed only when
        patterns were set for this step, and every parameter's norm is computed once and shared
        between the clip groups and the components.
        """
        self._step_local_norms = {}
        try:
            return self._clip_grad_once()
        finally:
            self._step_local_norms = None

    def _clip_grad_once(self) -> torch.Tensor:
        plan = self._grad_norm_plan()
        self.latest_component_grad_norms = self._compute_component_grad_norms()
        self.latest_clip_group_grad_norms = {}
        self.latest_clip_group_coefficients = {}
        if not self.clip_grad_norm_by_scheduler_group:
            return super()._clip_grad()

        for group_name, partition in plan.clip_groups.items():
            self.latest_clip_group_grad_norms[group_name] = self._compute_total_grad_norm(
                *self._main_grads(partition)
            )
        total_grad_norm = torch.linalg.vector_norm(
            torch.stack(list(self.latest_clip_group_grad_norms.values())), ord=2
        )

        self._maybe_debug_nan_inf_grad_norm(total_grad_norm, *self._main_grads(plan.all_params))
        if self.check_nan_inf_grad:
            _assert_finite_async(total_grad_norm, "total grad norm")

        for group_name, partition in plan.clip_groups.items():
            group_norm = self.latest_clip_group_grad_norms[group_name]
            clip_coefficient = torch.clamp(self.max_grad_norm / (group_norm + 1e-6), max=1.0).to(
                group_norm.device
            )
            torch._foreach_mul_(
                [self.main_grad[name] for names in partition for name in names], clip_coefficient
            )
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
