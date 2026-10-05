import logging
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import (
    DTensor,
    Placement,
    Replicate,
    Shard,
    distribute_tensor,
)
from torch.distributed.tensor._utils import (
    compute_local_shape_and_global_offset,
    compute_local_stride,
)

from olmo_core._nvtx import maybe_nvtx_annotate
from olmo_core.nn.fp8_weight import FP8WeightStore
from olmo_core.utils import env_bool, get_default_device, move_to_device

from ..config import Config, DType
from ..exceptions import OLMoConfigurationError
from .adamw import foreach_adamw_step
from .config import INITIAL_LR_FIELD, LR_FIELD, OptimGroupOverride
from .grad_debug import debug_nan_inf_grad_norm

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..train.train_module import TrainModule
    from ..train.train_module.transformer.ddp_train_module import OLMoDDPTrainModule


def _to_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def _assert_finite_async(tensor: torch.Tensor, name: str) -> None:
    """
    Device-side assert that ``tensor`` is all-finite, without a host sync (so it's cheap enough to
    run every step). The process aborts asynchronously if a non-finite value is encountered.
    """
    tensor = _to_local_tensor(tensor)
    torch._assert_async(
        torch.isfinite(tensor.detach()).all(),
        f"Non-finite {name} encountered in OLMoDDPOptimizer",
    )


def _is_fp8_weight_store(param: Any) -> bool:
    return isinstance(param, FP8WeightStore)


def _rowwise_fp8_only_params_enabled(owner: Any) -> bool:
    cfg = getattr(owner, "rowwise_fp8", None)
    return (
        cfg is not None
        and getattr(cfg, "enabled", False)
        and getattr(cfg, "fp8_only_params", False)
    )


def _is_fp8_only_anchor_param(param: Any) -> bool:
    owner_ref = getattr(param, "_moe_rowwise_fp8_cache_owner", None)
    if owner_ref is None:
        return False
    owner = owner_ref() if callable(owner_ref) else owner_ref
    if owner is None:
        return False
    if param is getattr(owner, "w_up_gate", None) or param is getattr(owner, "w_down", None):
        return _rowwise_fp8_only_params_enabled(owner)
    routed_experts = getattr(owner, "routed_experts", None)
    if routed_experts is not None and (
        param is getattr(routed_experts, "w_up_gate", None)
        or param is getattr(routed_experts, "w_down", None)
    ):
        if _rowwise_fp8_only_params_enabled(routed_experts):
            return True
        return _rowwise_fp8_only_params_enabled(owner)
    shared_experts = getattr(owner, "shared_experts", None)
    if shared_experts is not None and (
        param is getattr(shared_experts, "w_up_gate", None)
        or param is getattr(shared_experts, "w_down", None)
    ):
        return _rowwise_fp8_only_params_enabled(owner)
    return False


def _is_fp8_only_expert_anchor_param(param: Any) -> bool:
    return _is_fp8_only_anchor_param(param)


@dataclass
class OLMoDDPOptimizerConfig(Config):
    """
    Configuration for :class:`OLMoDDPOptimizer`.

    Builds the distributed fused optimizer for the
    :class:`~olmo_core.nn.ddp.model.OLMoDDPModel` from AdamW settings
    (:data:`lr`, :data:`betas`, :data:`eps`, :data:`weight_decay`) and skip-step spike-detection
    settings (:data:`rolling_interval_length`,
    :data:`sigma_factor`, :data:`max_grad_norm`), plus optional per-parameter-group overrides.
    """

    group_overrides: Optional[List[OptimGroupOverride]] = None
    """
    Use this to pull out groups parameters into a separate param groups with their own options.
    """

    compile: bool = False
    """
    Compile the optimizer step.

    .. warning::
        Optimizer step compilation is still in beta and may not work with some optimizers.
        You could also see unexpected behavior and very poor performance when turning this feature
        on in the middle of a run that was previously trained without compiling the optimizer
        due to the LR being restored to a float instead of a tensor.
    """

    fixed_fields: Tuple[str, ...] = (INITIAL_LR_FIELD,)
    """
    These are fields that should not be overridden by the value in a checkpoint after
    loading optimizer state.
    """

    # TODO(optim-config-dup): the fields below mirror OLMoDDPOptimizer.__init__ (name + default),
    # so `build()` can forward them via as_dict()->kwargs. This matches the other optim configs
    # (e.g. AdamWConfig mirrors torch.optim.AdamW), but here the optimizer is ours, so the two
    # default lists can drift. Revisit for a single source of truth (the config value always wins in
    # the build path; the __init__ defaults only apply to direct construction).
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    dtype: Optional[DType] = None

    rolling_interval_length: int = 128
    """
    The length of the rolling interval to use for computing the mean and standard deviation of the loss.
    """

    sigma_factor: int = 6
    """
    The number of standard deviations above the mean loss to skip a step.
    """

    max_grad_norm: float = 1.0

    use_distributed: bool = True
    reset_optimizer_moments_on_load: bool = False
    """
    When ``True``, ignore checkpointed ``exp_avg`` and ``exp_avg_sq`` values and
    reset them to zero when restoring optimizer state.
    """

    check_nan_inf_grad: bool = True
    """
    When ``True``, device-side assert (without a host sync) that the loss and the total grad norm
    are finite each step, aborting the run on a non-finite value; note this aborts rather than
    skipping the step, so it interacts with the skip-step spike detection.
    """

    @property
    def device(self) -> torch.device:
        return get_default_device()

    def _expand_param_globs(
        self,
        go: OptimGroupOverride,
        all_params: Dict[str, Any],
        frozen_param_names: Set[str],
        g_idx: int,
        strict: bool = True,
    ) -> OptimGroupOverride:
        param_names: List[str] = []
        for pattern in go.params:
            matches = 0
            for name in list(all_params.keys()):
                if fnmatch(name, pattern):
                    param_names.append(name)
                    matches += 1

            if matches == 0:
                for name in frozen_param_names:
                    if fnmatch(name, pattern):
                        log.warning(
                            f"optim group {g_idx} override pattern '{pattern}' matches a frozen parameter and will be ignored"
                        )
                        break
                else:
                    msg = f"optim group {g_idx} override pattern '{pattern}' does not match any parameters"  # TODO: might be false alarm, param can match patterns in other groups
                    if strict:
                        raise OLMoConfigurationError(msg)
                    else:
                        log.warning(msg)

        return OptimGroupOverride(param_names, go.opts.copy())

    def build_groups(
        self, model_parts: List[nn.Module], strict: bool = True, param_filter=None
    ) -> Union[Iterable[torch.Tensor], List[Dict[str, Any]]]:
        """
        Build parameters groups.

        :param model: The model to optimize.
        :param strict: If ``True`` an error is raised if a pattern in ``group_overrides`` doesn't
            match any parameter.
        """
        all_params: Dict[str, Any] = OrderedDict()
        frozen_params: set = set()
        for part in model_parts:
            for n, p in part.named_parameters():
                if _is_fp8_only_expert_anchor_param(p):
                    continue
                if p.requires_grad:
                    if param_filter is None:  # No filter applied
                        all_params[n] = p
                    else:
                        # Apply the parameter filter
                        if param_filter(p):
                            all_params[n] = p

                else:
                    frozen_params.add(n)
            named_fp8_weight_stores = getattr(part, "named_fp8_weight_stores", None)
            if named_fp8_weight_stores is None:
                named_fp8_weight_stores = getattr(part, "named_mxfp8_expert_weights", None)
            if named_fp8_weight_stores is not None:
                logical_prefix = (
                    "module."
                    if isinstance(getattr(part, "_modules", None), dict)
                    and "module" in part._modules
                    else ""
                )
                for n, p in named_fp8_weight_stores():
                    if not getattr(p, "optimizer_enabled", False):
                        continue
                    if param_filter is None or param_filter(p):
                        all_params[f"{logical_prefix}{n}"] = p

        group_overrides = [
            self._expand_param_globs(go, all_params, frozen_params, g_idx, strict=strict)
            for g_idx, go in enumerate(self.group_overrides or [])
        ]

        # Treat no overrides as its own override group
        overriden_param_names = {name for go in group_overrides for name in go.params}
        default_override = OptimGroupOverride(
            [name for name in all_params.keys() if name not in overriden_param_names], {}
        )
        # group_overrides.append(default_override)
        group_overrides.insert(0, default_override)  # to ensure default is first

        param_groups = []
        for go in group_overrides:
            if len(go.params) > 0:
                param_groups.append(
                    {
                        "named_params": {
                            param_name: all_params[param_name] for param_name in go.params
                        },
                        **go.opts,  #
                    }
                )

        return param_groups

    @classmethod
    def optimizer(cls):
        return OLMoDDPOptimizer

    def _collect_ep_param_ids(self, model_parts: List[nn.Module]) -> Set[int]:
        """
        Collect ids() of parameters that belong to modules marked as EP-managed
        (i.e., modules having attribute `_ep_sharded` set to True).
        """
        ep_param_ids: Set[int] = set()
        for part in model_parts:
            for m in part.modules():
                if getattr(m, "_ep_sharded", False):
                    for _, p in m.named_parameters(recurse=True):
                        if _is_fp8_only_expert_anchor_param(p):
                            continue
                        ep_param_ids.add(id(p))
                    named_fp8_weight_stores = getattr(m, "named_fp8_weight_stores", None)
                    if named_fp8_weight_stores is None:
                        named_fp8_weight_stores = getattr(m, "named_mxfp8_expert_weights", None)
                    if named_fp8_weight_stores is not None:
                        for _, p in named_fp8_weight_stores():
                            if getattr(p, "optimizer_enabled", False):
                                ep_param_ids.add(id(p))
        return ep_param_ids

    def build(  # type: ignore[override]
        self,
        model_parts: List,
        train_module: Optional["TrainModule"] = None,
        strict: bool = True,
        param_filter=None,
    ) -> "OLMoDDPOptimizer":
        """
        Build the optimizer.

        :param strict: If ``True`` an error is raised if a pattern in ``group_overrides`` doesn't
            match any parameter.
        """
        from ..nn.ddp.model import OLMoDDPModel

        assert train_module is not None, "OLMoDDPOptimizerConfig.build requires a train_module"
        model_parts = cast(List[OLMoDDPModel], model_parts)
        train_module = cast("OLMoDDPTrainModule", train_module)

        # not used: train_module (was); now used to pass process groups
        kwargs = self.as_dict()
        kwargs.pop("group_overrides")
        kwargs.pop("compile")
        kwargs.pop("fixed_fields")

        # Stable parameter order (by name) for each partition, used by all ranks for packing/broadcast.
        ep_param_ids = self._collect_ep_param_ids(model_parts)

        # Build param groups for the two PGs by filtering.
        # TODO(moe-optim-group-overrides-strict): each build_groups() call sees only one partition
        # (dense vs. EP), so with strict=True a `group_overrides` pattern that matches only EP params
        # (or only dense params) raises in the other partition's pass even though it does match
        # globally. Collect matches across both partitions before enforcing strict. This becomes
        # reachable once the MoE train module drives build().
        dp_groups = self.build_groups(
            model_parts, strict=strict, param_filter=lambda p: id(p) not in ep_param_ids
        )
        for g in dp_groups:
            g["pg"] = "dp"  # type: ignore

        ep_groups = self.build_groups(
            model_parts, strict=strict, param_filter=lambda p: id(p) in ep_param_ids
        )
        for g in ep_groups:
            g["pg"] = "ep_dp"  # type: ignore

        # Concatenate, ensuring the "default" groups remain first in each partition (already ensured by build_groups()).
        all_groups: List[Dict[str, Any]] = list(dp_groups) + list(ep_groups)  # type: ignore

        from olmo_core.nn.parallel.distributed import MultiGroupDistributedDataParallel

        if isinstance(model_parts[0], MultiGroupDistributedDataParallel):
            has_grad_accum_fp32_buffer = model_parts[0]._accumulate_grads_in_fp32

        else:
            per_part = [part.has_grad_accum_fp32_buffer for part in model_parts]
            # should all have the same value
            if not all(x == per_part[0] for x in per_part):
                raise ValueError("Inconsistent `has_grad_accum_fp32_buffer` among model parts")

            has_grad_accum_fp32_buffer = per_part[0]

        optim = self.optimizer()(
            all_groups,
            dp_group=getattr(train_module, "dp_group", None),
            ep_dp_group=getattr(train_module, "ep_dp_group", None),
            world_mesh=getattr(train_module, "world_mesh", None),
            device=getattr(train_module, "device", None),
            model_has_grad_accum_fp32_buffer=has_grad_accum_fp32_buffer,
            **kwargs,
        )

        # Set 'lr' and 'initial_lr' in each group if needed.
        fixed_fields_per_group: List[Dict[str, Any]] = [{} for _ in optim.param_groups]
        for fixed_fields, group in zip(fixed_fields_per_group, optim.param_groups):
            lr: Optional[float] = None
            if LR_FIELD in group:
                lr = group[LR_FIELD]
            elif hasattr(self, LR_FIELD):
                lr = getattr(self, LR_FIELD)

            if lr is not None:
                if self.compile:
                    # 'lr' should be a tensor.
                    group[LR_FIELD] = move_to_device(torch.tensor(lr), self.device)
                else:
                    group[LR_FIELD] = lr
                group.setdefault(INITIAL_LR_FIELD, lr)

            for k in self.fixed_fields:
                if k in group:
                    fixed_fields[k] = group[k]

        log.info(
            f"Building {self.optimizer().__name__} optimizer with {len(optim.param_groups)} param group(s)..."
        )
        for g_idx, group in enumerate(optim.param_groups):
            group_param_names = "\n - ".join(group["named_params"].keys())
            group_fields_list = "\n - ".join(
                [f"{k}: {v}" for k, v in optim.param_groups[g_idx].items() if k != "named_params"]
            )
            if group_fields_list:
                log.info(
                    f"Group {g_idx}, {len(group['named_params'])} parameter(s):\n - {group_fields_list}\n - params:\n - {group_param_names}"
                )
            else:
                log.info(
                    f"Group {g_idx}, {len(group['named_params'])} parameter(s):\n - params:\n - {group_param_names}"
                )

        if self.compile:
            # Compile only the math-heavy update path. Keeping comm/copy-back eager
            # avoids Dynamo/Inductor capturing giant all_gather graphs that can OOM.
            log.info("Compiling optimizer update path (_step_foreach)...")
            optim._step_foreach = torch.compile(optim._step_foreach)

        # Register hook to reset fixed fields after loading a checkpoint.

        # optim.register_load_state_dict_post_hook(reset_fixed_fields)

        return optim


def assign_full_tensor_to_dtensor(dst: DTensor, src: torch.Tensor) -> None:
    assert dst.shape == src.shape  # global shape

    src_dt = distribute_tensor(src, dst.device_mesh, placements=dst.placements)
    dst.copy_(src_dt)


@dataclass
class _FlatModelParamSyncEntry:
    state_key: str
    param: torch.nn.Parameter
    flat_slice: torch.Tensor
    sharded_target: Optional[torch.Tensor]
    numel: int
    is_sharded: bool
    local_numel: int
    local_offset: int


@dataclass
class _FlatModelParamSyncGroup:
    tag: str
    dtype: torch.dtype
    flat_buffer: torch.Tensor
    sharded_entries: List[_FlatModelParamSyncEntry]
    replicated_entries: List[_FlatModelParamSyncEntry]
    total_sharded_local_numel: int
    process_group: Optional[ProcessGroup]
    world_size: int


# Intentionally not a torch.optim.Optimizer subclass. This object owns
# MoE-specific state layout, EP/DP sharding, gradient intake from MultiGroupDDP,
# FP8 cache refresh, and checkpoint serialization directly. Revisit the choice
# if trainer/callback integration starts needing more of the torch optimizer
# interface.
class OLMoDDPOptimizer:
    """
    Distributed fused optimizer for the
    :class:`~olmo_core.nn.ddp.model.OLMoDDPModel`.

    Keeps fp32 master copies of the parameters, reduce-scatters gradients and gathers updated
    parameters across the data-parallel and expert-parallel data-parallel process groups (DTensor-
    and expert-parallel-aware), and applies a fused per-group **AdamW** step with **skip-step**
    loss-spike detection. Built via :class:`OLMoDDPOptimizerConfig`.
    """

    LOSSES_STATE_DICT_KEY = "__moe_skip_step_losses"
    GRAD_NORMS_STATE_DICT_KEY = "__moe_skip_step_grad_norms"
    ADAM_MOMENT_STATE_SUFFIXES = ("exp_avg", "exp_avg_sq")
    MOMENT_STATE_SUFFIXES = ADAM_MOMENT_STATE_SUFFIXES

    def __init__(
        self,
        param_groups: Iterable[Dict[str, Any]],
        world_mesh: Dict[str, Optional[DeviceMesh]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        rolling_interval_length: int = 128,
        sigma_factor: int = 6,
        max_grad_norm: float = 1.0,
        dtype: Optional[Union[torch.dtype, DType]] = None,
        device: Optional[torch.device] = None,
        model_has_grad_accum_fp32_buffer: bool = False,  # whether the optimizer should expect the model to have fp32 grad accum buffers
        # --- new args for sharding across multiple PGs ---
        dp_group: Optional[ProcessGroup] = None,
        ep_dp_group: Optional[ProcessGroup] = None,
        broadcast_bucket_mb: int = 32,
        do_not_shard_tensor_smaller_than: int = 4096,
        use_distributed: bool = True,
        check_nan_inf_grad: bool = True,
        reset_optimizer_moments_on_load: bool = False,
    ) -> None:
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        self.model_has_grad_accum_fp32_buffer = model_has_grad_accum_fp32_buffer
        self.use_distributed = use_distributed
        self.reset_optimizer_moments_on_load = reset_optimizer_moments_on_load

        def _add_defaults_to_param_group(pg: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in defaults.items():
                pg.setdefault(k, v)
            return pg

        # add defaults to each param group
        param_groups = [_add_defaults_to_param_group(pg) for pg in param_groups]

        # for print info
        self._model_param_sz = 0
        self._mxfp8_logical_param_sz = 0
        self._mxfp8_cache_sz = 0
        for param_group in param_groups:
            for _name, param in param_group["named_params"].items():
                param_sz = param.numel() * param.element_size()
                self._model_param_sz += param_sz
                if _is_fp8_weight_store(param):
                    self._mxfp8_logical_param_sz += param_sz
                    for prequantized_rhs in param.iter_prequantized_caches():
                        self._mxfp8_cache_sz += (
                            prequantized_rhs.mat_b_q.numel()
                            * prequantized_rhs.mat_b_q.element_size()
                            + prequantized_rhs.scale_b.numel()
                            * prequantized_rhs.scale_b.element_size()
                        )

        # ---- Sharding context (DP and EP-DP) ----
        self._dp_group: Optional[ProcessGroup] = dp_group
        self._ep_dp_group: Optional[ProcessGroup] = ep_dp_group

        assert world_mesh["dense"] is not None, "DP mesh must be provided"

        self.dense_mesh: DeviceMesh = world_mesh["dense"]  # ('pp', 'dp')
        self.moe_mesh: Optional[DeviceMesh] = world_mesh["moe"]  # ('pp', 'ep_dp', 'ep_mp')

        self.dp_mesh = self.dense_mesh["dp"]
        self.ep_dp_mesh = self.moe_mesh["ep_dp"] if self.moe_mesh else None
        self.ep_mp_mesh = self.moe_mesh["ep_mp"] if self.moe_mesh else None
        self._ep_checkpoint_mesh_cache: Optional[DeviceMesh] = None

        self.rolling_interval_length = rolling_interval_length
        self.sigma_factor = sigma_factor
        self._losses: List[torch.Tensor] = []
        self._grad_norms: List[torch.Tensor] = []
        self._device: Optional[torch.device] = device
        self.max_grad_norm = max_grad_norm
        if isinstance(dtype, DType):
            dtype = dtype.as_pt()
        self.dtype = dtype
        self.check_nan_inf_grad = check_nan_inf_grad
        self.states_dtype = torch.float32
        self.main_grad_dtype = torch.float32

        self._step_skipped: Optional[torch.Tensor] = None
        self.do_not_shard_tensor_smaller_than = do_not_shard_tensor_smaller_than
        # MultiGroupDDP owns normal-parameter gradient synchronization. Its
        # default is all-reduce, and the opt-in reduce-scatter path still enters
        # the optimizer through _copy_model_grads_to_main_grads().
        self._use_reduce_scatter_grads = False
        self.main_grad: Dict[str, torch.Tensor] = {}
        self._flat_model_sync_groups: "OrderedDict[str, _FlatModelParamSyncGroup]" = OrderedDict()

        # check
        device = None
        has_bf16_param = False
        has_fp32_param = False
        for param_group in param_groups:
            for i, (name, param) in enumerate(param_group["named_params"].items()):
                if not param.requires_grad:
                    continue
                if device is None:
                    device = param.device
                else:
                    assert device == param.device, (
                        f"Inconsistent device found for param '{name}': "
                        f"expected {device}, got {param.device}"
                    )
                # float16 params:
                if param.type() in ["torch.cuda.HalfTensor", "torch.cuda.BFloat16Tensor"]:
                    has_bf16_param = True
                elif param.type() in ["torch.cuda.FloatTensor"]:
                    has_fp32_param = True

        if has_bf16_param and has_fp32_param:
            raise ValueError("Mixed bf16 and fp32 parameters are not supported in OLMoDDPOptimizer")

        if has_bf16_param:
            # The model only has bf16 params
            # The optimizer has to decide whether to maintain fp32 main params
            self.should_maintain_fp32_main_param = True
        else:
            # The model has its own copy of fp32 main params
            self.should_maintain_fp32_main_param = False

        for param_group in param_groups:
            for _name, param in param_group["named_params"].items():
                if _is_fp8_weight_store(param):
                    param.accumulate_wgrad_in_fp32 = self.model_has_grad_accum_fp32_buffer

        self.states: Dict[str, DTensor] = OrderedDict()

        for param_group in param_groups:
            # configure the device mesh to shard the group
            device_mesh = self._get_dp_device_mesh_for_tag(param_group["pg"])
            assert device_mesh is not None, f"Device mesh for pg tag {param_group['pg']} is None"

            # wrap each param with DTensor
            for name, param in param_group["named_params"].items():
                # flat in fp32
                num_elements = param.numel()

                # main param
                if self.should_maintain_fp32_main_param:
                    main_param = torch.zeros(num_elements, dtype=torch.float32, device=device)
                    main_param = self._distribute_tensor(main_param, device_mesh)
                    self.states[f"{name}.main"] = main_param
                else:
                    assert (
                        param.dtype == torch.float32
                    ), "Expect fp32 param when should_maintain_fp32_main_param is False"
                    # wrap in DTensor so it works with rest of the code
                    self.states[f"{name}.main"] = DTensor.from_local(
                        param.data.view(-1), device_mesh=device_mesh, placements=[Replicate()]
                    )

                # exp avg
                exp_avg = torch.zeros(num_elements, dtype=self.states_dtype, device=device)
                exp_avg = self._distribute_tensor(exp_avg, device_mesh)
                self.states[f"{name}.exp_avg"] = exp_avg

                # exp avg sq
                exp_avg_sq = torch.zeros(num_elements, dtype=self.states_dtype, device=device)
                exp_avg_sq = self._distribute_tensor(exp_avg_sq, device_mesh)
                self.states[f"{name}.exp_avg_sq"] = exp_avg_sq

                # step
                step_tensor = torch.zeros((), dtype=torch.float32, device=device)
                step_tensor = distribute_tensor(
                    step_tensor,
                    device_mesh=device_mesh,
                    placements=[Replicate()],
                )
                self.states[f"{name}.step"] = step_tensor

        self.param_groups = param_groups
        if self.should_maintain_fp32_main_param:
            self._init_flat_model_param_buffers()

        # copy model params to main params
        self._copy_model_params_to_main_params()

        if self.should_maintain_fp32_main_param:
            self._check_model_param_main_param_the_same()

        self._copy_main_params_to_mxfp8_weights()
        self._release_mxfp8_expert_anchor_storage()
        self.print_memory_summary()

        return

    def print_memory_summary(self):
        total_params = 0
        for param_group in self.param_groups:
            for name, param in param_group["named_params"].items():
                total_params += param.numel()
        log.info(f"[OLMoDDPOptimizer] Total model params: {total_params:,}")
        self._mxfp8_cache_sz = 0
        seen_mxfp8_weights: Set[int] = set()
        for param_group in self.param_groups:
            for param in param_group["named_params"].values():
                if not _is_fp8_weight_store(param) or id(param) in seen_mxfp8_weights:
                    continue
                seen_mxfp8_weights.add(id(param))
                for prequantized_rhs in param.iter_prequantized_caches():
                    self._mxfp8_cache_sz += (
                        prequantized_rhs.mat_b_q.numel() * prequantized_rhs.mat_b_q.element_size()
                        + prequantized_rhs.scale_b.numel() * prequantized_rhs.scale_b.element_size()
                    )

        # main
        def count_numel(tag: str):
            global_state_numel = 0
            local_state_numel = 0
            num_tensors_sharded = 0
            num_tensors_replicated = 0
            sharded_state_numel = 0
            replicated_state_numel = 0
            for state_key, state_val in self.states.items():
                if state_key.endswith(f".{tag}"):
                    global_state_numel += state_val.numel()
                    local_state_numel += state_val.to_local().numel()
                    if any(isinstance(p, Shard) for p in state_val.placements):
                        num_tensors_sharded += 1
                        sharded_state_numel += state_val.to_local().numel()
                    else:
                        num_tensors_replicated += 1
                        replicated_state_numel += state_val.to_local().numel()
            return (
                global_state_numel,
                local_state_numel,
                num_tensors_sharded,
                num_tensors_replicated,
                sharded_state_numel,
                replicated_state_numel,
            )

        def to_str_N_B_GB(num):
            return f"{num:,} | {num/1000**3:.4} Billion | {num * 4 /1024**3:.4} GB"

        def info_str(tag: str, stat: Tuple[int, int, int, int, int, int]):
            info_str = ""
            info_str += f"[OLMoDDPOptimizer] {tag} - Global params: {to_str_N_B_GB(stat[0])}, Local params: {to_str_N_B_GB(stat[1])}\n"
            info_str += f"    Sharded tensors: {stat[2]}, total local sharded params: {to_str_N_B_GB(stat[4])}\n"
            info_str += f"    Replicated tensors: {stat[3]}, total local replicated params: {to_str_N_B_GB(stat[5])}\n"
            return info_str

        main_stat = count_numel("main")
        exp_avg_stat = count_numel("exp_avg")
        exp_avg_sq_stat = count_numel("exp_avg_sq")

        print_str = ""

        print_str += info_str("Main param", main_stat)
        if exp_avg_stat[0] > 0:
            print_str += info_str("Exp avg", exp_avg_stat)
        if exp_avg_sq_stat[0] > 0:
            print_str += info_str("Exp avg sq", exp_avg_sq_stat)

        BYTES_IN_GB = 1024**3

        total_global_optim_gb = main_stat[0] * self.main_grad_dtype.itemsize / BYTES_IN_GB
        total_global_optim_gb += (
            (exp_avg_stat[0] + exp_avg_sq_stat[0]) * self.states_dtype.itemsize / BYTES_IN_GB
        )

        total_local_optim_gb = main_stat[1] * self.main_grad_dtype.itemsize / BYTES_IN_GB
        total_local_optim_gb += (
            (exp_avg_stat[1] + exp_avg_sq_stat[1]) * self.states_dtype.itemsize / BYTES_IN_GB
        )

        normal_model_param_gb = (self._model_param_sz - self._mxfp8_logical_param_sz) / BYTES_IN_GB
        total_model_gb = normal_model_param_gb
        total_mxfp8_cache_gb = self._mxfp8_cache_sz / BYTES_IN_GB
        print_str += f"[OLMoDDPOptimizer] Total optimizer states size: {total_global_optim_gb:.4f} GB global, {total_local_optim_gb:.4f} GB local\n"

        if self.model_has_grad_accum_fp32_buffer:
            logical_mxfp8_grad_gb = self._mxfp8_logical_param_sz / BYTES_IN_GB
            total_model_grad_gb = (
                2 * normal_model_param_gb + logical_mxfp8_grad_gb
            )  # extra fp32 grad buffer for normal params
        else:
            total_model_grad_gb = total_model_gb  # bf16 grad only
        print_str += f"[OLMoDDPOptimizer] Model params size (GB): {total_model_gb:.4f} GB, model grads size (GB): {total_model_grad_gb:.4f} GB\n"
        if self._mxfp8_logical_param_sz > 0:
            logical_mxfp8_param_gb = self._mxfp8_logical_param_sz / BYTES_IN_GB
            print_str += (
                f"[OLMoDDPOptimizer] FP8 logical bf16-equivalent params skipped from model storage: "
                f"{logical_mxfp8_param_gb:.4f} GB, FP8 RHS caches: {total_mxfp8_cache_gb:.4f} GB\n"
            )
        total_static = (
            total_local_optim_gb + total_model_gb + total_model_grad_gb + total_mxfp8_cache_gb
        )

        print_str += (
            f"[OLMoDDPOptimizer] Total estimated static memory (GB): {total_static:.4f} GB\n"
        )

        log.info(print_str)

    def _init_flat_model_param_buffers(self) -> None:
        groups_by_tag: "OrderedDict[str, List[Tuple[str, torch.nn.Parameter]]]" = OrderedDict()
        seen_param_ids: Set[int] = set()

        for param_group in self.param_groups:
            tag = param_group["pg"]
            entries = groups_by_tag.setdefault(tag, [])
            for name, param in param_group["named_params"].items():
                if _is_fp8_weight_store(param):
                    continue
                param_id = id(param)
                if param_id in seen_param_ids:
                    raise RuntimeError(
                        f"Parameter '{name}' appears multiple times in optimizer groups"
                    )
                seen_param_ids.add(param_id)
                entries.append((name, param))

        self._flat_model_sync_groups = OrderedDict()

        for tag, named_params in groups_by_tag.items():
            if not named_params:
                continue

            group_dtype = named_params[0][1].dtype
            # Preserve the 16-byte alignment used by compiled kernels when
            # they are reused across blocks. Small BF16 parameters (e.g. KDA
            # gates) can otherwise misalign the next parameter in this buffer.
            alignment_numel = 16 // named_params[0][1].element_size()
            total_numel = 0
            total_sharded_local_numel = 0
            for name, param in named_params:
                if not param.data.is_contiguous():
                    raise RuntimeError(
                        f"Flat model param buffers require contiguous parameter storage, got '{name}'"
                    )
                if param.dtype != group_dtype:
                    raise RuntimeError(
                        f"Mixed dtypes are not supported in flat model buffer group '{tag}'"
                    )
                total_numel += (
                    (param.numel() + alignment_numel - 1) // alignment_numel * alignment_numel
                )

                main_param = self.states[f"{name}.main"]
                if any(isinstance(p, Shard) for p in main_param.placements):
                    total_sharded_local_numel += main_param.to_local().numel()

            flat_buffer = torch.empty(total_numel, device=self.device, dtype=group_dtype)
            process_group = self._get_process_group_for_tag(tag)
            world_size = 1 if process_group is None else dist.get_world_size(process_group)

            sharded_entries: List[_FlatModelParamSyncEntry] = []
            replicated_entries: List[_FlatModelParamSyncEntry] = []

            global_offset = 0
            local_offset = 0
            for name, param in named_params:
                numel = param.numel()
                old_param_data = param.data
                flat_slice = flat_buffer.narrow(0, global_offset, numel)
                flat_param_view = flat_slice.view_as(old_param_data)
                flat_param_view.copy_(old_param_data)
                param.data = flat_param_view

                main_param = self.states[f"{name}.main"]
                is_sharded = any(isinstance(p, Shard) for p in main_param.placements)
                local_numel = main_param.to_local().numel() if is_sharded else 0
                sharded_target = flat_slice.view(world_size, local_numel) if is_sharded else None

                entry = _FlatModelParamSyncEntry(
                    state_key=f"{name}.main",
                    param=param,
                    flat_slice=flat_slice,
                    sharded_target=sharded_target,
                    numel=numel,
                    is_sharded=is_sharded,
                    local_numel=local_numel,
                    local_offset=local_offset,
                )
                if is_sharded:
                    sharded_entries.append(entry)
                    local_offset += local_numel
                else:
                    replicated_entries.append(entry)

                global_offset += (numel + alignment_numel - 1) // alignment_numel * alignment_numel

            self._flat_model_sync_groups[tag] = _FlatModelParamSyncGroup(
                tag=tag,
                dtype=group_dtype,
                flat_buffer=flat_buffer,
                sharded_entries=sharded_entries,
                replicated_entries=replicated_entries,
                total_sharded_local_numel=total_sharded_local_numel,
                process_group=process_group,
                world_size=world_size,
            )

        self._refresh_rowwise_fp8_caches_from_model_params()

    @torch.no_grad()
    def _copy_model_params_to_main_params(self) -> None:
        """Refresh FP32 optimizer shards after initialization or a model-only load."""
        if not self.should_maintain_fp32_main_param:
            # FP32 models already share storage with their main parameters.
            return
        for param_group in self.param_groups:
            for name, param in param_group["named_params"].items():
                assign_full_tensor_to_dtensor(
                    dst=self.states[f"{name}.main"], src=param.data.float().reshape(-1)
                )

    def _check_model_param_main_param_the_same(self):
        for param_group in self.param_groups:
            for name, param in param_group["named_params"].items():
                main_param = self.states[f"{name}.main"]
                # get global tensor from DTensor
                main_param_full = main_param.full_tensor().reshape(-1)
                model_param = param.data.float().reshape(-1)
                if not torch.allclose(model_param, main_param_full, atol=1e-5):
                    raise ValueError(
                        f"{name}: Model param {param} and main param {main_param} are not close"
                    )

    def _distribute_tensor(
        self,
        tensor,
        device_mesh,
        force_shard: bool = False,
        force_replicate: bool = False,
    ) -> DTensor:
        num_elements = tensor.numel()
        if force_shard and force_replicate:
            raise ValueError("A tensor cannot be both force-sharded and force-replicated")
        if force_shard:
            # always shard, useful for saving checkpoint
            placements = [Shard(0)]
        elif force_replicate:
            placements = [Replicate()]
        elif self.use_distributed:
            if (
                num_elements >= self.do_not_shard_tensor_smaller_than
                and num_elements % device_mesh.size(0) == 0
            ):
                # this is distributed optimizer, so each rank holds one shard of the data
                placements = [Shard(0)]
            else:
                # small tensor, do not shard
                placements = [Replicate()]
                log.info(f"[OLMoDDPOptimizer] A tensor of size {num_elements} is replicated.")
        else:
            # always no shard
            placements = [Replicate()]

        tensor_dt = distribute_tensor(
            tensor,
            device_mesh=device_mesh,
            placements=placements,
        )

        return tensor_dt

    def offload_optimizer_states(self):
        raise NotImplementedError()
        # Offload optimizer states to CPU to save GPU memory

    def reload_optimizer_states_to_device(
        self,
    ):
        raise NotImplementedError()
        # Reload optimizer states to the given device

    @property
    def device(self) -> torch.device:
        if self._device is None:
            for group in self.param_groups:
                for n, p in group["named_params"].items():
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

    @property
    def step_skipped(self) -> torch.Tensor:
        if self._step_skipped is not None:
            return self._step_skipped
        else:
            return torch.tensor(0.0)

    def normal_params_with_sharded_optimizer_state(self) -> set[torch.nn.Parameter]:
        """Return normal model parameters whose FP32 main state is sharded.

        FP8WeightStore objects use their existing optimizer-owned gradient reducer and
        are intentionally excluded from the normal-parameter DDP configuration.
        """
        params: set[torch.nn.Parameter] = set()
        for param_group in self.param_groups:
            for name, param in param_group["named_params"].items():
                if _is_fp8_weight_store(param) or not param.requires_grad:
                    continue
                main_param = self.states[f"{name}.main"]
                if (
                    any(isinstance(placement, Shard) for placement in main_param.placements)
                    and main_param.device_mesh.size(0) > 1
                ):
                    params.add(param)
        return params

    def _clip_grad(self) -> torch.Tensor:
        """
        We need to first compute the grad norm for the FULL model.
        The optimizer sees the model that's sharded across PP and EP_MP when initialized.
        Then the optimizer further shards the model across DP or EP_DP.
        At this point, replicated optimizer states have the same full reduced
        gradients on each replica rank, while sharded optimizer states have
        disjoint reduced gradient shards.

        We need to consider:
        1. PP: compute for each PP rank, then reduce across PP ranks. Apply to all grads.
        2. EP_MP: compute for each EP_MP rank, then reduce across EP_MP ranks. Apply to EP grads.
        3. DP: compute for each DP rank, don't need to reduce across DP ranks, they should be the same already. Apply to DP grads.
        4. Watch out for small tensors that are replicated instead of sharded.

        """

        # separate DP and EP_DP grads
        dp_grads_replicated = []
        dp_grads_sharded = []
        ep_dp_grads_replicated = []
        ep_dp_grads_sharded = []

        for param_group in self.param_groups:
            for name, param in param_group["named_params"].items():
                if not param.requires_grad:
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

        total_grad_norm = self._compute_total_grad_norm(
            dp_grads_replicated,
            dp_grads_sharded,
            ep_dp_grads_replicated,
            ep_dp_grads_sharded,
        )

        self._maybe_debug_nan_inf_grad_norm(
            total_grad_norm,
            dp_grads_replicated,
            dp_grads_sharded,
            ep_dp_grads_replicated,
            ep_dp_grads_sharded,
        )
        if self.check_nan_inf_grad:
            _assert_finite_async(total_grad_norm, "total grad norm")

        clip_coef = self.max_grad_norm / (total_grad_norm + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
        # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
        # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0).to(total_grad_norm.device)

        all_grads = list(self.main_grad.values())
        torch._foreach_mul_(all_grads, clip_coef_clamped)

        return total_grad_norm

    def _local_total_norm(self, grads: List[torch.Tensor]) -> torch.Tensor:
        norms: List[torch.Tensor] = []
        for grad in grads:
            local_grad = _to_local_tensor(grad)
            if local_grad.numel() == 0:
                continue
            norms.append(torch.linalg.vector_norm(local_grad.detach().float(), ord=2))
        if not norms:
            return torch.zeros((), device=self.device, dtype=torch.float32)
        return torch.linalg.vector_norm(torch.stack(norms), ord=2)

    def _compute_total_grad_norm(
        self,
        dp_grads_replicated: List[torch.Tensor],
        dp_grads_sharded: List[torch.Tensor],
        ep_dp_grads_replicated: List[torch.Tensor],
        ep_dp_grads_sharded: List[torch.Tensor],
    ) -> torch.Tensor:
        dp_grads_norm_sharded = self._local_total_norm(dp_grads_sharded)
        dp_grads_norm_replicated = self._local_total_norm(dp_grads_replicated)

        dp_grads_norm_sharded_reduced = self._reduce_norm(
            dp_grads_norm_sharded, self.dp_mesh.get_group()
        )
        dp_grad_norm = self._combine_norm(dp_grads_norm_replicated, dp_grads_norm_sharded_reduced)

        if self.moe_mesh is not None:
            ep_dp_grads_norm_sharded = self._local_total_norm(ep_dp_grads_sharded)
            ep_dp_grads_norm_replicated = self._local_total_norm(ep_dp_grads_replicated)

            ep_dp_grads_norm_sharded_reduced = self._reduce_norm(
                ep_dp_grads_norm_sharded, self.ep_dp_mesh.get_group()
            )
            ep_dp_grad_norm = self._combine_norm(
                ep_dp_grads_norm_replicated, ep_dp_grads_norm_sharded_reduced
            )

            ep_grad_norm = self._reduce_norm(ep_dp_grad_norm, self.ep_mp_mesh.get_group())
            total_grad_norm = self._combine_norm(dp_grad_norm, ep_grad_norm)
        else:
            total_grad_norm = dp_grad_norm

        assert self.dense_mesh.mesh_dim_names is not None
        if "pp" in self.dense_mesh.mesh_dim_names:
            total_grad_norm = self._reduce_norm(total_grad_norm, self.dense_mesh["pp"].get_group())
        return total_grad_norm

    def _maybe_debug_nan_inf_grad_norm(
        self,
        total_grad_norm: torch.Tensor,
        dp_grads_replicated: List[torch.Tensor],
        dp_grads_sharded: List[torch.Tensor],
        ep_dp_grads_replicated: List[torch.Tensor],
        ep_dp_grads_sharded: List[torch.Tensor],
    ) -> None:
        """
        Opt-in diagnostic (gated by ``OLMO_DDP_DEBUG_NONFINITE_GRAD``): when the total grad norm is
        non-finite, log/dump the per-parameter and per-component local grad norms to locate the
        offending parameter. This is just the optimizer-specific adapter — it reads the env knobs
        and extracts the grad data, then delegates to
        :func:`olmo_core.optim.grad_debug.debug_nan_inf_grad_norm`. A no-op unless the env var is
        set and the norm is non-finite (the extraction closures only run then).

        TODO(config-gate this diagnostic): the ad-hoc ``OLMO_DDP_DEBUG_*`` env knobs are off
        convention (core gates behavior via Config fields — cf. ``check_nan_inf_grad``). Migrate to
        a ``debug_nan_inf_grad: bool`` config field (fire on ``check_nan_inf_grad and
        debug_nan_inf_grad``) and move the rank / max_log_entries / dump-dir knobs to config fields too. The
        logged step is the trainer ``global_step`` (set on ``_debug_global_step``, not the Adam
        ``.step`` state, which lags ``global_step`` by the skip count on this skip-step optimizer);
        a clean ``global_step`` setter mirroring ``latest_loss`` would replace the raw attribute.
        """
        if not env_bool("OLMO_DDP_DEBUG_NONFINITE_GRAD"):
            return
        try:
            max_log_entries = int(os.getenv("OLMO_DDP_DEBUG_NONFINITE_GRAD_TOPK", "20"))
        except ValueError:
            max_log_entries = 20
        debug_nan_inf_grad_norm(
            _to_local_tensor(total_grad_norm.detach()),
            step=getattr(self, "_debug_global_step", -1),
            ranks_filter=os.getenv("OLMO_DDP_DEBUG_NONFINITE_GRAD_RANKS", "all"),
            max_log_entries=max_log_entries,
            dump_dir=os.getenv("OLMO_DEBUG_DUMP_DIR"),
            component_norms=lambda: {
                "dp_replicated_local": self._local_total_norm(dp_grads_replicated),
                "dp_sharded_local": self._local_total_norm(dp_grads_sharded),
                "ep_dp_replicated_local": self._local_total_norm(ep_dp_grads_replicated),
                "ep_dp_sharded_local": self._local_total_norm(ep_dp_grads_sharded),
            },
            iter_local_grads=self._iter_local_grads,
        )

    def _iter_local_grads(self) -> Iterator[Tuple[str, str, str, torch.Tensor]]:
        """Yield ``(name, param_group, placements, local_grad)`` for each param with a local grad."""
        for param_group in self.param_groups:
            for name, param in param_group["named_params"].items():
                if not param.requires_grad:
                    continue
                grad = self.main_grad.get(name)
                if grad is None:
                    continue
                local_grad = _to_local_tensor(grad.detach())
                if local_grad.numel() == 0:
                    continue
                placements = ",".join(str(p) for p in self.states[f"{name}.main"].placements)
                yield name, param_group["pg"], placements, local_grad

    def _combine_norm(self, n1, n2) -> torch.Tensor:
        return torch.sqrt(n1.square() + n2.square())

    def _reduce_norm(self, norm: torch.Tensor, pg: ProcessGroup) -> torch.Tensor:
        if norm.device.type == "cpu" and self.device.type != "cpu":
            norm = norm.to(self.device)
        norm = norm.square()
        dist.all_reduce(norm, op=dist.ReduceOp.SUM, group=pg)
        norm = norm.sqrt()
        return norm

    @torch.no_grad()
    @maybe_nvtx_annotate("OLMoDDPOptimizer.step")
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Run one optimizer step: bring gradients into the fp32 master grads, apply the fused AdamW
        update with skip-step spike detection, then gather the updated parameters back to the model.
        ``closure`` is accepted for API compatibility.
        """
        if getattr(self, "_use_reduce_scatter_grads", False):
            # Legacy optimizer-owned reducer. The active train module keeps this
            # disabled; MultiGroupDDP now owns both normal-parameter AR and RS.
            self._reduce_scatter_model_grads()
        else:
            # Active DDP-owned intake: full all-reduced gradients and local
            # reduce-scattered shards both enter through this method.
            self._copy_model_grads_to_main_grads()

        if self.check_nan_inf_grad and self.latest_loss is not None:
            _assert_finite_async(self.latest_loss, "loss")

        # _clip_grad() also asserts the total grad norm is finite when check_nan_inf_grad is set.
        total_grad_norm = self._clip_grad()

        self.latest_grad_norm = total_grad_norm

        self._step_foreach(closure)

        self._dealloc_main_grad()

        self._copy_main_params_to_model_params()

        return None

    def _get_process_group_for_tag(self, tag: str):
        if tag == "dp":
            return self._dp_group
        elif tag == "ep_dp":
            return self._ep_dp_group
        else:
            raise RuntimeError(f"Unknown pg tag: {tag}")

    def _get_dp_device_mesh_for_tag(self, tag: str):
        if tag == "dp":
            return self.dp_mesh
        elif tag == "ep_dp":
            return self.ep_dp_mesh
        else:
            raise RuntimeError(f"Unknown pg tag: {tag}")

    def _world_and_rank(self, pg: Optional[ProcessGroup]) -> Tuple[int, int]:
        if pg is None:
            return 1, 0
        return dist.get_world_size(pg), dist.get_rank(pg)

    def _state_suffixes_for_param(self, name: str) -> Tuple[str, ...]:
        return ("main", "exp_avg", "exp_avg_sq", "step")

    def _gather_sharded_flat_tensor(
        self,
        local_tensor: torch.Tensor,
        state_dt: DTensor,
        *,
        output_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if not any(isinstance(p, Shard) for p in state_dt.placements):
            if output_dtype is not None and local_tensor.dtype != output_dtype:
                return local_tensor.to(output_dtype)
            return local_tensor

        flat_global, sizes, offsets = coalesced_all_gather_flat(
            [local_tensor],
            state_dt.device_mesh.get_group(),
            output_dtype=output_dtype,
        )
        gathered = flat_global[:, offsets[0] : offsets[0] + sizes[0]].contiguous()
        return gathered.reshape(-1)

    def _load_moment_state_or_zero(
        self, state_dict: Dict[str, Any], state_key: str
    ) -> Optional[Any]:
        if state_key in state_dict:
            return state_dict.pop(state_key)

        self._ensure_local_state_storage(state_key).to_local().zero_()
        return None

    @staticmethod
    def _shares_tensor_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
        if a.numel() == 0 or b.numel() == 0:
            return False
        return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()

    def _ep_checkpoint_mesh(self) -> DeviceMesh:
        assert self.moe_mesh is not None
        # Checkpoint chunks are ordered by logical expert shard first, then the
        # optimizer shard within that expert shard. This gives a topology-neutral
        # flat tensor without materializing all EP ranks on GPU.
        if self._ep_checkpoint_mesh_cache is None:
            ep_mesh = self.moe_mesh["ep_dp", "ep_mp"]
            ranks = ep_mesh.mesh.permute(1, 0).contiguous()
            self._ep_checkpoint_mesh_cache = DeviceMesh(
                ep_mesh.device_type,
                ranks,
                mesh_dim_names=("ep_mp", "ep_dp"),
                _init_backend=False,
            )
        return self._ep_checkpoint_mesh_cache

    def _ep_checkpoint_placements(self, state_dt: DTensor) -> List[Placement]:
        assert len(state_dt.placements) == 1
        ep_dp_placement = state_dt.placements[0]
        if ep_dp_placement.is_shard():
            return [Shard(0), Shard(0)]
        return [Shard(0), Replicate()]

    def _ep_checkpoint_global_numel(self, state_dt: DTensor) -> int:
        assert self.ep_mp_mesh is not None
        return state_dt.numel() * self.ep_mp_mesh.size()

    def _is_ep_checkpoint_view(self, state_dt: DTensor, ckpt_state: DTensor) -> bool:
        expected_mesh = self._ep_checkpoint_mesh()
        return (
            torch.equal(ckpt_state.device_mesh.mesh, expected_mesh.mesh)
            and tuple(ckpt_state.device_mesh.mesh_dim_names or ())
            == tuple(expected_mesh.mesh_dim_names or ())
            and list(ckpt_state.placements) == self._ep_checkpoint_placements(state_dt)
            and ckpt_state.numel() == self._ep_checkpoint_global_numel(state_dt)
        )

    def _ep_dp_state_to_checkpoint(self, live_state_dt: DTensor) -> DTensor:
        assert self.moe_mesh is not None
        assert self.ep_dp_mesh is not None and self.ep_mp_mesh is not None
        assert len(live_state_dt.placements) == 1
        flat_local = live_state_dt.to_local().reshape(-1)
        checkpoint_mesh = self._ep_checkpoint_mesh()
        checkpoint_placements = self._ep_checkpoint_placements(live_state_dt)
        global_numel = self._ep_checkpoint_global_numel(live_state_dt)

        return DTensor.from_local(
            flat_local,
            device_mesh=checkpoint_mesh,
            placements=checkpoint_placements,
            shape=(global_numel,),
            stride=(1,),
            run_check=False,
        )

    def _load_ep_dp_state_from_checkpoint(self, state_key: str, ckpt_state: DTensor) -> None:
        assert self.moe_mesh is not None
        state_dt = self.states[state_key]
        state_local = state_dt.to_local()
        ckpt_local_direct = ckpt_state.to_local().reshape(-1)
        is_checkpoint_view = self._is_ep_checkpoint_view(state_dt, ckpt_state)
        if is_checkpoint_view and self._shares_tensor_storage(
            ckpt_local_direct, state_local.reshape(-1)
        ):
            return
        if state_local.numel() == 0:
            local_shape, _ = compute_local_shape_and_global_offset(
                state_dt.shape,
                state_dt.device_mesh,
                state_dt.placements,
            )
        else:
            local_shape = tuple(state_local.shape)
        if is_checkpoint_view and ckpt_local_direct.numel() == math.prod(local_shape):
            state_dt = self._ensure_local_state_storage(state_key)
            state_dt.to_local().copy_(ckpt_local_direct.reshape(tuple(local_shape)))
            return

        state_dt = self._ensure_local_state_storage(state_key)
        ckpt_state = ckpt_state.full_tensor().reshape(-1)
        # Slice views directly instead of re-wrapping with distribute_tensor, which
        # would allocate another local expert shard during checkpoint restore-back.
        ckpt_state = self.narrow_tensor(
            ckpt_state,
            self.moe_mesh["ep_mp"],
            [Shard(0)],
        )
        if state_dt.placements[0].is_shard():
            ckpt_local = self.narrow_tensor(
                ckpt_state,
                self.moe_mesh["ep_dp"],
                [Shard(0)],
            )
        else:
            ckpt_local = ckpt_state

        ckpt_local = ckpt_local.reshape(state_dt.to_local().shape)
        assert (
            ckpt_state.shape == state_dt.shape
        ), f"Global shape mismatch for {state_key}: {ckpt_state.shape} vs {state_dt.shape}"
        assert (
            ckpt_local.shape == state_dt.to_local().shape
        ), f"Local shape mismatch for {state_key}: {ckpt_local.shape} vs {state_dt.to_local().shape}"
        state_dt.to_local().copy_(ckpt_local)

    @maybe_nvtx_annotate("OLMoDDPOptimizer._reduce_scatter_model_grads")
    def _reduce_scatter_model_grads(self) -> None:
        for param_group in self.param_groups:
            for name, param in param_group["named_params"].items():
                is_fp8_store = _is_fp8_weight_store(param)
                if is_fp8_store:
                    model_grad_fp32 = self._get_fp8_weight_store_model_grad_fp32(name, param)
                elif self.model_has_grad_accum_fp32_buffer:
                    # the model already has a fp32 grad buffer, so the grad is already in fp32
                    # and model's bf16 grad should be None
                    if param.grad is not None:
                        raise RuntimeError(
                            "Expected model param grad to be None. Use _main_grad_fp32 to store the grad."
                        )

                    if param._main_grad_fp32 is None:
                        raise RuntimeError(
                            f"Missing _main_grad_fp32 for param '{name}'. "
                            "Grad buffers must stay bound to DDP bucket views."
                        )

                    model_grad_fp32 = param._main_grad_fp32.detach().view(
                        -1
                    )  # unsharded local shape, FP32
                else:
                    if param.grad is None:
                        raise RuntimeError(
                            f"Missing .grad for param '{name}'. "
                            "Grad buffers must stay bound to DDP bucket views."
                        )

                    # model's grad is in bf16, need to convert to fp32 for reduce-scatter
                    model_grad_fp32 = (
                        param.grad.detach().view(-1).float()
                    )  # unsharded local shape, FP32

                # prepare main param grad view
                main_param = self.states[f"{name}.main"]  # DTensor, full shape unsharded
                # depending on whether the tensor is sharded or replicated, use reduce_scatter or all-reduce
                dp_world_process_group = self._get_process_group_for_tag("dp")
                dp_world_size = (
                    1
                    if dp_world_process_group is None
                    else dist.get_world_size(dp_world_process_group)
                )
                if all(
                    isinstance(p, Shard) for p in main_param.placements
                ):  # actually main_param is always 1D flat, so it's sharded along dim 0 always
                    # reduce scatter from model grad to main param grad local
                    main_grad_local = torch.empty_like(main_param.to_local())  # local shard shape
                    dist.reduce_scatter_tensor(
                        main_grad_local,
                        model_grad_fp32,
                        group=self._get_process_group_for_tag(param_group["pg"]),
                        op=dist.ReduceOp.SUM,
                    )
                else:
                    # the tensor is replicated, use all-reduce so that all ranks have the same grad
                    # all-reduce model grad to main param grad local
                    dist.all_reduce(
                        model_grad_fp32,
                        op=dist.ReduceOp.SUM,
                        group=self._get_process_group_for_tag(param_group["pg"]),
                    )
                    main_grad_local = model_grad_fp32  # now all ranks have the same grad

                # NOTE: no matter the sum is over dp ranks or ep_dp ranks, ALWAYS divide by dp world size.
                # Explain for ep_dp grads:
                # if the EP_MP world size is X, then each EP_MP rank is already seeing X times the
                # data, hence each rank's grad is already equivalent to summing over X ranks. The above
                # reduce scatter further sums over the EP_DP ranks, which is equivalent to summing over
                # the full DP world size.
                main_grad_local.div_(dp_world_size)

                # save main param grad
                self.main_grad[name] = DTensor.from_local(
                    main_grad_local,
                    device_mesh=main_param.device_mesh,
                    placements=main_param.placements,
                )

        return

    def _get_fp8_weight_store_model_grad(self, name: str, param: FP8WeightStore) -> torch.Tensor:
        if self.model_has_grad_accum_fp32_buffer:
            if param.main_grad_fp32 is None:
                raise RuntimeError(f"Missing logical FP32 grad for FP8 weight store '{name}'")
            model_grad = param.main_grad_fp32.detach().reshape(-1)
            if model_grad.dtype != torch.float32:
                model_grad = model_grad.float()
            return model_grad

        if param.grad_bf16 is None:
            raise RuntimeError(f"Missing logical BF16 grad for FP8 weight store '{name}'")
        return param.grad_bf16.detach().reshape(-1)

    def _get_fp8_weight_store_model_grad_fp32(
        self, name: str, param: FP8WeightStore
    ) -> torch.Tensor:
        model_grad = self._get_fp8_weight_store_model_grad(name, param)
        if model_grad.dtype != torch.float32:
            model_grad = model_grad.float()
        return model_grad

    def _ep_mp_world_size_for_group(self, param_group: Dict[str, Any]) -> int:
        if self.moe_mesh is None or param_group["pg"] != "ep_dp":
            return 1
        assert self.ep_mp_mesh is not None
        return int(self.ep_mp_mesh.size())

    @staticmethod
    def _clear_fp8_weight_store_grad(param: FP8WeightStore) -> None:
        param.grad_bf16 = None
        param.main_grad_fp32 = None

    def _copy_fp8_model_grads_to_main_grads(
        self,
        param_group: Dict[str, Any],
        entries: List[Tuple[str, FP8WeightStore, torch.Tensor, DTensor]],
    ) -> None:
        if not entries:
            return

        pg = self._get_process_group_for_tag(param_group["pg"])
        pg_world_size = 1
        if dist.is_available() and dist.is_initialized():
            pg_world_size = dist.get_world_size(pg)
        average_denominator = pg_world_size * self._ep_mp_world_size_for_group(param_group)

        # The FP8-only weights are not normal DDP parameters, so their logical
        # gradients are reduced here. Sharded optimizer states only need the
        # local post-reduction shard, so use reduce-scatter instead of
        # all-reducing the full logical gradient.
        bucket_cap_bytes = 512 * 1024 * 1024
        reduce_scatter_bucket: List[Tuple[str, FP8WeightStore, torch.Tensor, DTensor]] = []
        reduce_scatter_bucket_bytes = 0
        all_reduce_bucket: List[Tuple[str, FP8WeightStore, torch.Tensor, DTensor]] = []
        all_reduce_bucket_bytes = 0

        def flush_reduce_scatter_bucket() -> None:
            nonlocal reduce_scatter_bucket
            nonlocal reduce_scatter_bucket_bytes
            if not reduce_scatter_bucket:
                return

            dtype = reduce_scatter_bucket[0][2].dtype
            device = reduce_scatter_bucket[0][2].device
            local_numels = [
                main_param.to_local().numel() for _, _, _, main_param in reduce_scatter_bucket
            ]
            total_local_numel = sum(local_numels)
            flat_input = torch.empty(
                (pg_world_size, total_local_numel),
                device=device,
                dtype=dtype,
            )

            offset = 0
            for (_name, _param, model_grad, main_param), local_numel in zip(
                reduce_scatter_bucket,
                local_numels,
            ):
                expected_numel = local_numel * pg_world_size
                if model_grad.numel() != expected_numel:
                    raise RuntimeError(
                        "FP8 reduce-scatter grad size mismatch: "
                        f"model_grad={model_grad.numel()} expected={expected_numel}"
                    )
                flat_input[:, offset : offset + local_numel].copy_(
                    model_grad.view(pg_world_size, local_numel)
                )
                offset += local_numel

            if average_denominator != 1:
                flat_input.div_(average_denominator)

            flat_output = torch.empty(total_local_numel, device=device, dtype=dtype)
            dist.reduce_scatter_tensor(
                flat_output,
                flat_input.reshape(-1),
                op=dist.ReduceOp.SUM,
                group=pg,
            )

            offset = 0
            for (name, param, _model_grad, _main_param), local_numel in zip(
                reduce_scatter_bucket,
                local_numels,
            ):
                # Views keep the reduced bucket alive until _dealloc_main_grad().
                self.main_grad[name] = flat_output[offset : offset + local_numel]
                self._clear_fp8_weight_store_grad(param)
                offset += local_numel

            reduce_scatter_bucket = []
            reduce_scatter_bucket_bytes = 0

        def flush_all_reduce_bucket() -> None:
            nonlocal all_reduce_bucket
            nonlocal all_reduce_bucket_bytes
            if not all_reduce_bucket:
                return

            if pg_world_size > 1:
                dtype = all_reduce_bucket[0][2].dtype
                device = all_reduce_bucket[0][2].device
                total_numel = sum(model_grad.numel() for _, _, model_grad, _ in all_reduce_bucket)
                flat = torch.empty(total_numel, device=device, dtype=dtype)
                offset = 0
                for _, _, model_grad, _ in all_reduce_bucket:
                    numel = model_grad.numel()
                    flat[offset : offset + numel].copy_(model_grad)
                    offset += numel
                if average_denominator != 1:
                    flat.div_(average_denominator)
                dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=pg)
                offset = 0
                for name, param, model_grad, main_param in all_reduce_bucket:
                    numel = model_grad.numel()
                    reduced_grad = flat[offset : offset + numel]
                    self.main_grad[name] = self.narrow_tensor(
                        reduced_grad,
                        main_param.device_mesh,
                        main_param.placements,
                    )
                    self._clear_fp8_weight_store_grad(param)
                    offset += numel
            else:
                for name, param, model_grad, main_param in all_reduce_bucket:
                    if average_denominator != 1:
                        model_grad.div_(average_denominator)
                    self.main_grad[name] = self.narrow_tensor(
                        model_grad,
                        main_param.device_mesh,
                        main_param.placements,
                    )
                    self._clear_fp8_weight_store_grad(param)

            all_reduce_bucket = []
            all_reduce_bucket_bytes = 0

        for name, param, model_grad, main_param in entries:
            model_grad = model_grad.reshape(-1)
            model_grad_bytes = model_grad.numel() * model_grad.element_size()
            can_reduce_scatter = pg_world_size > 1 and any(
                isinstance(p, Shard) for p in main_param.placements
            )
            if can_reduce_scatter:
                if reduce_scatter_bucket and (
                    model_grad.dtype != reduce_scatter_bucket[0][2].dtype
                    or model_grad.device != reduce_scatter_bucket[0][2].device
                    or reduce_scatter_bucket_bytes + model_grad_bytes > bucket_cap_bytes
                ):
                    flush_reduce_scatter_bucket()
                reduce_scatter_bucket.append((name, param, model_grad, main_param))
                reduce_scatter_bucket_bytes += model_grad_bytes
            else:
                if all_reduce_bucket and (
                    model_grad.dtype != all_reduce_bucket[0][2].dtype
                    or model_grad.device != all_reduce_bucket[0][2].device
                    or all_reduce_bucket_bytes + model_grad_bytes > bucket_cap_bytes
                ):
                    flush_all_reduce_bucket()
                all_reduce_bucket.append((name, param, model_grad, main_param))
                all_reduce_bucket_bytes += model_grad_bytes

        flush_reduce_scatter_bucket()
        flush_all_reduce_bucket()
        entries.clear()

    @maybe_nvtx_annotate("OLMoDDPOptimizer._copy_model_grads_to_main_grads")
    def _copy_model_grads_to_main_grads(self):
        for param_group in self.param_groups:
            fp8_entries: List[Tuple[str, FP8WeightStore, torch.Tensor, DTensor]] = []
            for name, param in param_group["named_params"].items():
                if _is_fp8_weight_store(param):
                    model_grad = self._get_fp8_weight_store_model_grad(name, param)
                    main_param = self.states[f"{name}.main"]
                    fp8_entries.append((name, param, model_grad, main_param))
                    continue

                main_param = self.states[f"{name}.main"]
                reduced_grad_shard = getattr(param, "_olmo_ddp_reduced_grad_shard", None)
                if reduced_grad_shard is not None:
                    if not any(isinstance(placement, Shard) for placement in main_param.placements):
                        raise RuntimeError(
                            f"Received a reduce-scattered gradient for replicated "
                            f"parameter '{name}'."
                        )
                    expected_numel = main_param.to_local().numel()
                    if reduced_grad_shard.numel() != expected_numel:
                        raise RuntimeError(
                            f"Reduce-scattered gradient size mismatch for '{name}': "
                            f"got {reduced_grad_shard.numel()}, expected {expected_numel}."
                        )
                    self.main_grad[name] = reduced_grad_shard.detach().view(-1)
                elif self.model_has_grad_accum_fp32_buffer:
                    # the model already has a fp32 grad buffer, so the grad is already in fp32
                    # and model's bf16 grad should be None
                    if param.grad is not None:
                        raise RuntimeError(
                            "Expected model param grad to be None. Use _main_grad_fp32 to store the grad."
                        )

                    if param._main_grad_fp32 is None:
                        raise RuntimeError(
                            f"Missing _main_grad_fp32 for param '{name}'. "
                            "Grad buffers must stay bound to DDP bucket views."
                        )

                    model_grad = param._main_grad_fp32.detach().view(
                        -1
                    )  # unsharded local shape, FP32
                else:
                    if param.grad is None:
                        raise RuntimeError(
                            f"Missing .grad for param '{name}'. "
                            "Grad buffers must stay bound to DDP bucket views."
                        )

                    # model's grad is in bf16, need to convert to fp32 for reduce-scatter
                    # model_grad_fp32 = param.grad.detach().view(-1).float() # unsharded local shape, FP32
                    model_grad = param.grad.detach().view(
                        -1
                    )  # unsharded local shape, BF16. It should be a view of the reducer bucket

                if reduced_grad_shard is None:
                    # It turns out distribute_tensor is too slow on CPU. Narrow the
                    # already-all-reduced gradient directly to the optimizer-owned view.
                    self.main_grad[name] = self.narrow_tensor(
                        model_grad, main_param.device_mesh, main_param.placements
                    )
                    del model_grad

                # MultiGroupDDP has already averaged EP params over EP-DP. Expert
                # compute saw tokens from all EP-MP ranks, so divide by EP-MP here
                # to make optimizer-consumed expert grads scale as 1 / dense DP.
                if self.moe_mesh is not None and param_group["pg"] == "ep_dp":
                    ep_mp_world_process_group = self.ep_mp_mesh.get_group()
                    ep_mp_world_size = dist.get_world_size(ep_mp_world_process_group)
                    self.main_grad[name].div_(ep_mp_world_size)
            self._copy_fp8_model_grads_to_main_grads(param_group, fp8_entries)

    def narrow_tensor(
        self, orignal: torch.Tensor, device_mesh: DeviceMesh, placements: List[Placement]
    ):
        assert len(placements) == 1, "Only support 1D sharding"
        assert device_mesh.ndim == 1, "Only support 1D device mesh"
        assert orignal.dim() == 1, "Only support 1D tensor"
        placement = placements[0]

        if placement.is_replicate():
            return orignal

        assert placement.is_shard(), "Only support shard or replicate placements"
        coord = device_mesh.get_coordinate()[0]
        ws = device_mesh.size(0)
        shard_size = orignal.numel() // ws
        start = coord * shard_size
        local_shard = orignal.narrow(0, start, shard_size)

        return local_shard

    @torch._dynamo.disable()
    @maybe_nvtx_annotate("OLMoDDPOptimizer._copy_main_params_to_model_params")
    def _copy_main_params_to_model_params(self):
        if self._flat_model_sync_groups:
            self._copy_main_params_to_flat_model_buffers()
            self._copy_main_params_to_mxfp8_weights()
            self._refresh_rowwise_fp8_caches_from_model_params()
            return

        LAUNCH_AG_THRESHOLD = 500_000_000  # X elements
        for param_group in self.param_groups:
            # initialize for coalesced all_gather
            input_dtensors: List[torch.Tensor] = []
            output_params: List[torch.Tensor] = []
            input_numel = 0
            fp8_entries: List[Tuple[str, FP8WeightStore, DTensor]] = []

            def flush_all_gather():
                nonlocal input_dtensors, output_params, input_numel
                if len(input_dtensors) == 0:
                    return
                pg = input_dtensors[0].device_mesh.get_group()
                gather_dtype = output_params[0].dtype
                input_locals = [t.to_local() for t in input_dtensors]
                flat_global, sizes, offsets = coalesced_all_gather_flat(
                    input_locals,
                    pg,
                    output_dtype=gather_dtype,
                )

                world_size = flat_global.shape[0]
                for size, off, out_param in zip(sizes, offsets, output_params):
                    out_param.data.view(world_size, size).copy_(flat_global[:, off : off + size])

                output_params.clear()
                input_dtensors.clear()
                input_numel = 0

            for name, param in param_group["named_params"].items():
                if _is_fp8_weight_store(param):
                    fp8_entries.append((name, param, self.states[f"{name}.main"]))
                    continue

                main_param = self.states[f"{name}.main"]
                if not any(isinstance(p, Shard) for p in main_param.placements):
                    # replicated tensor, directly get full tensor
                    main_param_local = main_param.to_local().reshape(param.data.shape)
                    # param.data.copy_(main_param_local.to(param.data.dtype))
                    param.data.copy_(main_param_local)
                    continue

                # check for process group
                if len(input_dtensors) > 0:
                    assert main_param.device_mesh == input_dtensors[0].device_mesh
                    assert param.dtype == output_params[0].dtype

                input_dtensors.append(main_param)
                output_params.append(param)
                input_numel += main_param.numel()

                if input_numel >= LAUNCH_AG_THRESHOLD:
                    flush_all_gather()

            # final gather
            if len(input_dtensors) > 0:
                flush_all_gather()
            self._copy_main_params_to_mxfp8_weight_entries(fp8_entries)

        self._refresh_rowwise_fp8_caches_from_model_params()
        return

    def _copy_main_params_to_mxfp8_weights(self) -> None:
        for param_group in self.param_groups:
            fp8_entries: List[Tuple[str, FP8WeightStore, DTensor]] = []
            for name, param in param_group["named_params"].items():
                if _is_fp8_weight_store(param):
                    fp8_entries.append((name, param, self.states[f"{name}.main"]))
            self._copy_main_params_to_mxfp8_weight_entries(fp8_entries)

    def _refresh_mxfp8_weight_from_full_flat(
        self,
        name: str,
        weight: FP8WeightStore,
        full_flat: torch.Tensor,
    ) -> None:
        if full_flat.numel() != weight.numel():
            raise RuntimeError(
                f"Gathered FP8 weight '{name}' has {full_flat.numel()} elements, "
                f"expected {weight.numel()}"
            )
        logical_weight = full_flat.reshape(weight.logical_shape)
        if logical_weight.dtype != torch.bfloat16:
            logical_weight = logical_weight.to(torch.bfloat16)
        weight.refresh_from_logical_weight(
            logical_weight,
            update_anchor=not weight.anchor_storage_released,
        )

    def _copy_main_params_to_mxfp8_weight_entries(
        self,
        entries: List[Tuple[str, FP8WeightStore, DTensor]],
    ) -> None:
        if not entries:
            return

        gather_threshold_elems = 250_000_000
        bucket: List[Tuple[str, FP8WeightStore, DTensor]] = []
        bucket_numel = 0

        def flush_bucket() -> None:
            nonlocal bucket
            nonlocal bucket_numel
            if not bucket:
                return

            first_main = bucket[0][2]
            pg = first_main.device_mesh.get_group()
            local_tensors = [main_param.to_local().reshape(-1) for _, _, main_param in bucket]
            flat_global, sizes, offsets = coalesced_all_gather_flat(
                local_tensors,
                pg,
                output_dtype=torch.bfloat16,
            )

            for (name, weight, _main_param), size, offset in zip(bucket, sizes, offsets):
                full_flat = flat_global[:, offset : offset + size].contiguous().reshape(-1)
                self._refresh_mxfp8_weight_from_full_flat(name, weight, full_flat)

            bucket = []
            bucket_numel = 0

        for name, weight, main_param in entries:
            local_flat = main_param.to_local().reshape(-1)
            if not any(isinstance(p, Shard) for p in main_param.placements):
                self._refresh_mxfp8_weight_from_full_flat(
                    name,
                    weight,
                    local_flat.to(torch.bfloat16),
                )
                continue

            if main_param.device_mesh.size(0) == 1:
                self._refresh_mxfp8_weight_from_full_flat(
                    name,
                    weight,
                    local_flat.to(torch.bfloat16),
                )
                continue

            if bucket:
                first_main = bucket[0][2]
                if main_param.device_mesh != first_main.device_mesh:
                    flush_bucket()
            if bucket and bucket_numel + main_param.numel() > gather_threshold_elems:
                flush_bucket()

            bucket.append((name, weight, main_param))
            bucket_numel += main_param.numel()

        flush_bucket()

    def _copy_main_param_to_mxfp8_weight(self, name: str, weight: FP8WeightStore) -> None:
        main_param = self.states[f"{name}.main"]
        local_flat = main_param.to_local().reshape(-1)
        full_flat = self._gather_sharded_flat_tensor(
            local_flat,
            main_param,
            output_dtype=torch.bfloat16,
        )
        self._refresh_mxfp8_weight_from_full_flat(name, weight, full_flat)

    def _release_mxfp8_expert_anchor_storage(self) -> None:
        seen: Set[int] = set()
        for param_group in self.param_groups:
            for param in param_group["named_params"].values():
                if not _is_fp8_weight_store(param) or id(param) in seen:
                    continue
                seen.add(id(param))
                param.release_anchor_storage()

    def _copy_main_params_to_flat_model_buffers(self) -> None:
        for sync_group in self._flat_model_sync_groups.values():
            for entry in sync_group.replicated_entries:
                main_param = self.states[entry.state_key]
                # entry.flat_slice.copy_(main_param.to_local().reshape(-1).to(sync_group.dtype))
                entry.flat_slice.copy_(main_param.to_local().reshape(-1))

            if not sync_group.sharded_entries:
                continue

            if sync_group.world_size == 1:
                for entry in sync_group.sharded_entries:
                    main_param = self.states[entry.state_key]
                    assert entry.sharded_target is not None
                    # entry.sharded_target.copy_(
                    #     main_param.to_local().reshape(1, entry.local_numel).to(sync_group.dtype)
                    # )
                    entry.sharded_target.copy_(main_param.to_local().reshape(1, entry.local_numel))
                continue

            assert sync_group.process_group is not None
            pack_buffer = torch.empty(
                sync_group.total_sharded_local_numel,
                device=self.device,
                dtype=sync_group.dtype,
            )
            gathered_buffer = torch.empty(
                sync_group.world_size * sync_group.total_sharded_local_numel,
                device=self.device,
                dtype=sync_group.dtype,
            )

            for entry in sync_group.sharded_entries:
                main_param = self.states[entry.state_key]
                pack_buffer[entry.local_offset : entry.local_offset + entry.local_numel].copy_(
                    main_param.to_local().reshape(-1)
                )

            dist.all_gather_into_tensor(
                gathered_buffer,
                pack_buffer,
                group=sync_group.process_group,
            )

            gathered_matrix = gathered_buffer.view(
                sync_group.world_size,
                sync_group.total_sharded_local_numel,
            )
            for entry in sync_group.sharded_entries:
                assert entry.sharded_target is not None
                entry.sharded_target.copy_(
                    gathered_matrix[:, entry.local_offset : entry.local_offset + entry.local_numel]
                )

    @maybe_nvtx_annotate("OLMoDDPOptimizer._refresh_rowwise_fp8_caches_from_model_params")
    def _refresh_rowwise_fp8_caches_from_model_params(self) -> None:
        owners: List[Any] = []
        seen: Set[int] = set()
        for group in self.param_groups:
            for _, param in group["named_params"].items():
                owner_ref = getattr(param, "_moe_rowwise_fp8_cache_owner", None)
                if owner_ref is None and _is_fp8_weight_store(param):
                    anchor_param = getattr(param, "anchor_param", None)
                    owner_ref = getattr(anchor_param, "_moe_rowwise_fp8_cache_owner", None)
                if owner_ref is None:
                    continue
                owner = owner_ref() if callable(owner_ref) else owner_ref
                if owner is None:
                    continue
                owner_id = id(owner)
                if owner_id in seen:
                    continue
                if not hasattr(owner, "refresh_rowwise_fp8_cache"):
                    continue
                seen.add(owner_id)
                owners.append(owner)

        for owner in owners:
            owner.refresh_rowwise_fp8_cache()

    def _dealloc_main_grad(self):
        self.main_grad.clear()

    @torch._dynamo.disable()
    def get_step_factor(self) -> torch.Tensor:
        """
        Returns a float tensor which will be `1.0` if the optimizer should proceed with the step
        and `0.0` if the optimizer should skip the step.

        The tensor can be used within the optimizer's step computation to essentially skip a step
        without a host-device sync.
        """
        if len(self._losses) < max(2, self.rolling_interval_length // 2):
            return move_to_device(torch.tensor(1.0), self.device)

        loss_std, loss_mean = torch.std_mean(torch.stack(self._losses[:-1]))
        assert self.latest_loss is not None
        if self._grad_norms:
            assert self.latest_grad_norm is not None
            grad_norm_std, grad_norm_mean = torch.std_mean(torch.stack(self._grad_norms[:-1]))
            step_factor = torch.logical_and(
                (self.latest_loss - loss_mean) <= self.sigma_factor * loss_std,
                (self.latest_grad_norm - grad_norm_mean) <= self.sigma_factor * grad_norm_std,
            )
        else:
            step_factor = (self.latest_loss - loss_mean) <= self.sigma_factor * loss_std

        return step_factor.float()

    @maybe_nvtx_annotate("OLMoDDPOptimizer._step_foreach")
    def _step_foreach(self, closure=None) -> None:
        """Performs adamw step using foreach impl, limiting chunk size to reduce memory usage."""

        if closure is not None:
            with torch.enable_grad():
                closure()

        step_factor = self.get_step_factor()  # type: ignore
        step_factor = cast(torch.Tensor, step_factor)
        self._step_skipped = 1 - step_factor

        # Allow overriding via attribute; default to X elements.
        CHUNK_ELEMS = getattr(self, "_foreach_chunk_threshold", 600_000_000)

        for group in self.param_groups:
            # Per-chunk accumulators
            main_params: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []

            exp_avgs: list[torch.Tensor] = []  # always fp32
            exp_avg_sqs: list[torch.Tensor] = []  # always fp32

            exp_avgs_original: list[
                torch.Tensor
            ] = (
                []
            )  # if states_dtype is bf16, we need to keep a reference to the original bf16 tensors
            exp_avg_sqs_original: list[torch.Tensor] = []

            steps_list: list[torch.Tensor] = []
            running_elems: int = 0

            def flush_chunk():
                nonlocal main_params, grads, exp_avgs, exp_avg_sqs, steps_list, running_elems
                if not main_params:
                    return
                foreach_adamw_step(
                    main_params,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    steps_list,
                    lr=group["lr"],
                    betas=group["betas"],
                    eps=group["eps"],
                    weight_decay=group["weight_decay"],
                    step_factor=step_factor,
                    step_increment_bugfix=True,
                )

            def reset_chunk_buffers():
                nonlocal main_params, grads, exp_avgs, exp_avg_sqs, steps_list, running_elems, exp_avgs_original, exp_avg_sqs_original
                # reset for next chunk
                main_params = []
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                exp_avgs_original = []
                exp_avg_sqs_original = []
                steps_list = []
                running_elems = 0

            def maybe_copy_back_16bit_states():
                # foreach_adamw_step makes in place updates to exp_avgs and exp_avg_sqs which are in fp32
                # so if the fp32 states are copies of bf16 states, we need to copy them back
                # otherwies, they are the original fp32 states, no need to copy back
                nonlocal exp_avgs_original, exp_avg_sqs_original, exp_avgs, exp_avg_sqs
                if self.states_dtype == torch.bfloat16:
                    for i in range(len(exp_avgs)):
                        # copy back fp32 to original bf16 tensors
                        # exp_avgs_original[i].copy_(exp_avgs[i].to(torch.bfloat16))
                        # exp_avg_sqs_original[i].copy_(exp_avg_sqs[i].to(torch.bfloat16))
                        exp_avgs_original[i].copy_(exp_avgs[i])
                        exp_avg_sqs_original[i].copy_(exp_avg_sqs[i])

            for name, model_p in group["named_params"].items():
                if not model_p.requires_grad:
                    continue

                # in adam step(), make everything local and fp32
                main_params.append(self.states[f"{name}.main"].to_local())
                grads.append(_to_local_tensor(self.main_grad[name]).float())
                if self.states_dtype == torch.bfloat16:
                    # new fp32 copy
                    exp_avgs.append(self.states[f"{name}.exp_avg"].to_local().to(torch.float32))
                    exp_avg_sqs.append(
                        self.states[f"{name}.exp_avg_sq"].to_local().to(torch.float32)
                    )

                    exp_avgs_original.append(self.states[f"{name}.exp_avg"].to_local())
                    exp_avg_sqs_original.append(self.states[f"{name}.exp_avg_sq"].to_local())
                else:
                    # original fp32
                    exp_avgs.append(self.states[f"{name}.exp_avg"].to_local())
                    exp_avg_sqs.append(self.states[f"{name}.exp_avg_sq"].to_local())
                steps_list.append(self.states[f"{name}.step"].to_local())

                running_elems += self.states[f"{name}.main"].to_local().numel()
                # Flush when we reach/exceed the threshold. It's OK to overshoot with the last add.
                if running_elems >= CHUNK_ELEMS:
                    flush_chunk()
                    maybe_copy_back_16bit_states()
                    reset_chunk_buffers()

            # Flush any tail chunk
            flush_chunk()
            maybe_copy_back_16bit_states()
            reset_chunk_buffers()

    def zero_grad(self, set_to_none=True):
        raise RuntimeError(
            "zero_grad should be called by the MoE TrainModule on the models directly now."
        )
        for group in self.param_groups:
            for n, p in group["named_params"].items():
                # clear bf16 grad
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None

                    else:
                        p.grad.detach_()
                        p.grad.zero_()

                # clear fp32 grad buffer if exists
                if self.model_has_grad_accum_fp32_buffer:
                    if getattr(p, "_main_grad_fp32", None) is not None:
                        p._main_grad_fp32 = None

    def unsharded_state_dict(self) -> dict:
        raise NotImplementedError("Removed function")

    def _install_optim_from_cpu_dtensor(self, main_sd, state1_sd, state2_sd, distribute_ep_func):
        raise NotImplementedError("Removed function")

    def _restore_rolling_stats(self, values: Any) -> List[torch.Tensor]:
        if values is None:
            log.info("No rolling stats found in checkpoint, skipping restore.")
            return []

        log.info("Restoring rolling stats from checkpoint ...")

        if isinstance(values, torch.Tensor):
            raw_values: List[Any]
            if values.ndim == 0:
                raw_values = [values]
            else:
                raw_values = [v for v in values.reshape(-1).unbind()]
        elif isinstance(values, (list, tuple)):
            raw_values = list(values)
        else:
            raw_values = [values]

        restored: List[torch.Tensor] = []
        for value in raw_values:
            if isinstance(value, torch.Tensor):
                tensor_value = value.detach().to(device=self.device, dtype=torch.float32)
                tensor_value = tensor_value.reshape(-1)[0]
            else:
                tensor_value = move_to_device(
                    torch.tensor(float(value), dtype=torch.float32), self.device
                )
            restored.append(tensor_value)

        return restored[-(self.rolling_interval_length + 1) :]

    def state_dict(self) -> dict:
        sd = {}
        for param_group in self.param_groups:
            for name, param in param_group["named_params"].items():
                if not param.requires_grad:
                    continue
                all_suffixes = self._state_suffixes_for_param(name)
                if param_group["pg"] == "ep_dp":
                    for suffix in all_suffixes:
                        state_key = f"{name}.{suffix}"
                        live_state_dt = self.states[state_key]
                        if suffix != "step":
                            sd[state_key] = self._ep_dp_state_to_checkpoint(live_state_dt)

                            state_local = live_state_dt.to_local()
                            ckpt_local = sd[state_key].to_local()
                            if not self._shares_tensor_storage(ckpt_local, state_local):
                                # Free the local shard storage while keeping DTensor metadata.
                                # TODO(moe-optim-state-dict-drops-live-state): when the checkpoint
                                # view doesn't share storage with the live shard, this drops the
                                # live optimizer state (only restored on load), so continuing to
                                # train after a checkpoint save sees empty shards. Keep the live
                                # state (e.g. copy instead of swap). This becomes reachable once the
                                # train module checkpoints mid-run.
                                empty_local = torch.empty(
                                    0, dtype=state_local.dtype, device=state_local.device
                                )
                                self.states[state_key] = DTensor.from_local(
                                    empty_local,
                                    device_mesh=live_state_dt.device_mesh,
                                    placements=live_state_dt.placements,
                                    shape=live_state_dt.shape,
                                    stride=live_state_dt.stride(),
                                    run_check=False,
                                )
                        else:  # "step"
                            sd[state_key] = live_state_dt

                else:  # DP tensor already in the right dtensor
                    for suffix in all_suffixes:
                        state_dt = self.states[f"{name}.{suffix}"]
                        sd[f"{name}.{suffix}"] = state_dt

        assert set(sd.keys()) == set(
            self.states.keys()
        ), f"State dict keys do not match live states: {set(sd.keys()) ^ set(self.states.keys())}"

        # Store rolling skip-step statistics as plain lists so they can be checkpointed as a single BYTE_IO entry.
        sd[self.LOSSES_STATE_DICT_KEY] = [float(v.detach().cpu().item()) for v in self._losses]
        sd[self.GRAD_NORMS_STATE_DICT_KEY] = [
            float(v.detach().cpu().item()) for v in self._grad_norms
        ]

        return sd

    def _ensure_local_state_storage(self, state_key: str) -> DTensor:
        state_dt = self.states[state_key]
        if state_dt.to_local().numel() != 0:
            return state_dt

        local_shape, _ = compute_local_shape_and_global_offset(
            state_dt.shape,
            state_dt.device_mesh,
            state_dt.placements,
        )
        local_stride = compute_local_stride(
            state_dt.stride(),
            state_dt.device_mesh,
            state_dt.placements,
        )
        new_local = torch.empty_strided(
            tuple(local_shape),
            tuple(local_stride),
            dtype=state_dt.dtype,
            device=self.device,
        )
        state_dt = DTensor.from_local(
            new_local,
            device_mesh=state_dt.device_mesh,
            placements=state_dt.placements,
            shape=state_dt.shape,
            stride=state_dt.stride(),
            run_check=False,
        )
        self.states[state_key] = state_dt
        return state_dt

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        strict: bool = True,
        reset_optimizer_moments_on_load: Optional[bool] = None,
    ) -> None:
        # the loaded state dict is already distributed over the DP mesh,
        # here we need to convert the DP sharded tensors to EP_MP + EP_DP sharded
        if reset_optimizer_moments_on_load is None:
            reset_optimizer_moments_on_load = self.reset_optimizer_moments_on_load

        loaded_losses = state_dict.pop(self.LOSSES_STATE_DICT_KEY, None)
        loaded_grad_norms = state_dict.pop(self.GRAD_NORMS_STATE_DICT_KEY, None)

        for param_group in self.param_groups:
            for name, param in param_group["named_params"].items():
                if not param.requires_grad:
                    continue
                all_suffixes = self._state_suffixes_for_param(name)
                if reset_optimizer_moments_on_load:
                    for suffix in self.MOMENT_STATE_SUFFIXES:
                        state_key = f"{name}.{suffix}"
                        if state_key in self.states:
                            state_dict.pop(state_key, None)
                            self._ensure_local_state_storage(state_key).to_local().zero_()

                if param_group["pg"] == "ep_dp":
                    for suffix in all_suffixes:
                        if reset_optimizer_moments_on_load and suffix in self.MOMENT_STATE_SUFFIXES:
                            continue
                        state_key = f"{name}.{suffix}"
                        state_dt = self.states[state_key]
                        if suffix != "step":
                            ckpt_state = (
                                self._load_moment_state_or_zero(state_dict, state_key)
                                if suffix in self.MOMENT_STATE_SUFFIXES
                                else state_dict.pop(state_key)
                            )
                            if ckpt_state is not None:
                                self._load_ep_dp_state_from_checkpoint(state_key, ckpt_state)
                        else:
                            ckpt_state = state_dict.pop(state_key, None)
                            if ckpt_state is not None:
                                state_dt.copy_(ckpt_state.full_tensor())
                else:
                    for suffix in all_suffixes:
                        if reset_optimizer_moments_on_load and suffix in self.MOMENT_STATE_SUFFIXES:
                            continue
                        state_key = f"{name}.{suffix}"
                        live_state = self.states[state_key]
                        if suffix in self.MOMENT_STATE_SUFFIXES:
                            ckpt_state = self._load_moment_state_or_zero(state_dict, state_key)
                            if ckpt_state is None:
                                continue
                        else:
                            ckpt_state = state_dict.pop(state_key, None)
                            if ckpt_state is None:
                                continue

                        if suffix == "step":
                            live_state.copy_(ckpt_state.full_tensor())
                        else:
                            ckpt_local = ckpt_state.to_local()
                            live_state = self._ensure_local_state_storage(state_key)
                            assert (
                                ckpt_state.shape == live_state.shape
                            ), f"Global shape mismatch {name}.{suffix}: {ckpt_state.shape} vs {live_state.shape}"
                            assert (
                                ckpt_local.shape == live_state.to_local().shape
                            ), f"Local shape mismatch {name}.{suffix}: {ckpt_local.shape} vs {live_state.to_local().shape}"
                            live_state.to_local().copy_(ckpt_local)

        self._losses = self._restore_rolling_stats(loaded_losses)
        self._grad_norms = self._restore_rolling_stats(loaded_grad_norms)

        return

    def _global_numel(self, tag: str) -> int:
        raise NotImplementedError()


@torch._dynamo.disable()
def coalesced_all_gather_flat(
    input_tensors: List[torch.Tensor],
    process_group: dist.ProcessGroup,
    output_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, List[int], List[int]]:
    """
    Coalesced all_gather for a list of 1-D tensors.

    Returns the gathered flat buffer as a `[world_size, total_elems]` tensor
    together with the per-input sizes and offsets inside the packed buffer.
    """
    if not input_tensors:
        raise ValueError("input_tensors must be non-empty")

    device = input_tensors[0].device
    input_dtype = input_tensors[0].dtype
    output_dtype = input_dtype if output_dtype is None else output_dtype
    for t in input_tensors:
        assert t.dim() == 1, "All input_tensors must be 1-D"
        assert t.device == device, "All input_tensors must be on the same device"
        assert t.dtype == input_dtype, "All input_tensors must have the same dtype"

    world_size = dist.get_world_size(process_group)
    sizes = [t.numel() for t in input_tensors]
    offsets: List[int] = []
    running = 0
    for size in sizes:
        offsets.append(running)
        running += size
    total_elems = running

    flat_local = torch.empty(total_elems, device=device, dtype=output_dtype)
    for t, off in zip(input_tensors, offsets):
        flat_local[off : off + t.numel()].copy_(t.view(-1))

    flat_global = torch.empty(world_size * total_elems, device=device, dtype=output_dtype)
    dist.all_gather_into_tensor(flat_global, flat_local, group=process_group)
    return flat_global.view(world_size, total_elems), sizes, offsets


@torch._dynamo.disable()
def coalesced_all_gather(
    input_tensors: List[torch.Tensor],
    process_group: dist.ProcessGroup,
) -> List[torch.Tensor]:
    """
    Coalesced all_gather for a list of 1-D tensors.

    Args:
        input_tensors: List of 1-D tensors. For a given index i, all ranks must
            have input_tensors[i] with the same numel, dtype, and device.
        process_group: The process group to use for all_gather (default: world group).

    Returns:
        A list of tensors, one per input tensor.
        For input_tensors[i] of shape [N_i], the output[i] has shape [world_size, N_i],
        where output[i][r] is the data from rank r.
    """
    if not input_tensors:
        return []

    flat_global, sizes, offsets = coalesced_all_gather_flat(input_tensors, process_group)

    # 3) Unpack into per-tensor gathered outputs
    gathered_outputs: List[torch.Tensor] = []
    for size, off in zip(sizes, offsets):
        # [world_size, size]
        gathered = flat_global[:, off : off + size].contiguous()
        gathered_outputs.append(gathered)

    return gathered_outputs
