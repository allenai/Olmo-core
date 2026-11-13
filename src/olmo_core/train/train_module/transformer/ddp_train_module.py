import contextlib
import dataclasses
import json
import logging
import math
import os
import time
from collections import OrderedDict
from functools import cached_property, lru_cache
from itertools import product
from typing import (
    Any,
    Callable,
    ClassVar,
    Collection,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from torch.distributed import ProcessGroup
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    Metadata,
    TensorStorageMetadata,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.pipelining import PipelineStage
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.nn.parallel import DistributedDataParallel as DDP

from olmo_core._nvtx import nvtx
from olmo_core.aliases import PathOrStr
from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.checkpoint import (
    RemoteFileSystemReader,
    RemoteFileSystemWriter,
    _prepare_env_for_save,
    contiguous_planner,
)
from olmo_core.distributed.parallel import (
    DataParallelType,
    MeshDimName,
    get_device_mesh_info,
)
from olmo_core.distributed.parallel.context_parallel import ContextParallelConfig
from olmo_core.distributed.parallel.data_parallel import DataParallelConfig
from olmo_core.distributed.parallel.expert_parallel import ExpertParallelConfig
from olmo_core.distributed.parallel.pipeline_parallel import (
    PipelineParallelConfig,
    PipelineSchedule,
)
from olmo_core.distributed.parallel.tensor_parallel import TensorParallelConfig
from olmo_core.distributed.utils import (
    backend_supports_cuda,
    get_local_tensor,
    get_rank,
    get_reduce_divide_factor,
    get_world_size,
    is_distributed,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.nn.ddp import OLMoDDPModel
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.parallel.distributed import MultiGroupDistributedDataParallel
from olmo_core.nn.transformer import Transformer
from olmo_core.optim import OLMoDDPOptimizerConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.utils import get_default_device, log_once, move_to_device

from ...common import MetricMergeStrategy, ReduceType
from ..train_module import EvalBatchSpec, TrainModule
from . import objective as objective_utils
from .config import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
    TransformerPipelineParallelConfig,
    TransformerTensorParallelConfig,
)

log = logging.getLogger(__name__)


M = TypeVar("M", bound=List[OLMoDDPModel])


def cpu_mesh_like(gpu_mesh: DeviceMesh) -> DeviceMesh:
    # gpu_mesh.mesh is a CPU int tensor of ranks with the mesh's shape
    ranks = gpu_mesh.mesh.clone()  # e.g., tensor([[0,1],[2,3]]) or nested shape
    return DeviceMesh(
        "cpu",
        ranks,  # keep the exact shape
        mesh_dim_names=gpu_mesh.mesh_dim_names,
    )


class FlatSavePlanner(DefaultSavePlanner):
    """Default planner with separately observable metadata planning cost."""

    def __init__(self, *, constant_memory_planning=False, profile=False, **kwargs):
        super().__init__(**kwargs)
        if constant_memory_planning and getattr(self, "_enable_plan_caching", False):
            raise ValueError("Constant-memory planner does not support plan caching")
        self.constant_memory_planning = constant_memory_planning
        self.profile = profile
        self.timings: Dict[str, float] = {}

    def create_local_plan(self):
        start = time.perf_counter() if self.profile else 0.0
        if self.constant_memory_planning:
            result = contiguous_planner.create_contiguous_local_save_plan(
                self.state_dict, self.is_coordinator
            )
            if self.flatten_state_dict:
                result = dataclasses.replace(result, planner_data=self.mappings)
            self.plan = result
        else:
            result = super().create_local_plan()
        if self.profile:
            self.timings["local_plan_seconds"] = time.perf_counter() - start
        return result

    def create_global_plan(self, all_plans):
        start = time.perf_counter() if self.profile else 0.0
        result = super().create_global_plan(all_plans)
        if self.profile:
            self.timings["global_plan_seconds"] = time.perf_counter() - start
        return result


class OLMoDDPTrainModule(TrainModule):
    def __init__(
        self,
        model: OLMoDDPModel,
        optim: OLMoDDPOptimizerConfig,
        rank_microbatch_size: int,
        max_sequence_length: int,
        compile_model: bool = False,
        float8_config: Optional[Float8Config] = None,
        dp_config: Optional[TransformerDataParallelConfig] = None,
        tp_config: Optional[TransformerTensorParallelConfig] = None,
        pp_config: Optional[TransformerPipelineParallelConfig] = None,
        cp_config: Optional[TransformerContextParallelConfig] = None,
        ep_config: Optional[TransformerExpertParallelConfig] = None,
        ac_config: Optional[TransformerActivationCheckpointingConfig] = None,
        z_loss_multiplier: Optional[float] = None,
        autocast_precision: Optional[torch.dtype] = None,
        scheduler: Optional[Scheduler] = None,
        device: Optional[torch.device] = None,
        state_dict_save_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        state_dict_load_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        load_key_mapping: Optional[Dict[str, str]] = None,
        reset_optimizer_states_on_load: bool = False,
        reset_optimizer_states_on_resume: bool = False,
        expand_shared_qk_norm_on_load: bool = False,
        label_ignore_index: int = -100,
        eval_only: bool = False,
    ):
        super().__init__()
        assert isinstance(model, OLMoDDPModel), "OLMoDDPTrainModule only supports OLMoDDPModel"

        ######################### Validate arguments. [BEGIN] #########################
        if rank_microbatch_size % max_sequence_length != 0:
            raise OLMoConfigurationError(
                f"'rank_microbatch_size' ({rank_microbatch_size:,d} tokens) must be divisible by "
                f"'max_sequence_length' ({max_sequence_length:,d} tokens)"
            )
        self.max_sequence_length = max_sequence_length
        self.rank_microbatch_size = rank_microbatch_size
        self.eval_only = eval_only
        self.expand_shared_qk_norm_on_load = expand_shared_qk_norm_on_load
        # Build world mesh.
        self.device = device or get_default_device()
        self.world_mesh: Dict[str, Optional[DeviceMesh]] = {}
        self.pp_group = None
        self.dp_group = None
        self.tp_group = None
        self.ep_dp_group = None
        self.ep_mp_group = None
        self.cp_group = None
        self.dense_dp_cp_group = None
        self.expert_param_group = None
        self.ep_mp_high_priority_group = None
        self.ep_mp_high_priority_groups = None

        # compatibility
        if autocast_precision is not None:
            assert False, "Autocast precision is not supported in OLMoDDPTrainModule"
        self.autocast_precision = None

        # PP related state.
        self._train_pp_schedule: Optional[PipelineSchedule] = None
        self._pp_stages: Optional[List[PipelineStage]] = None
        self.pp_group_rank = 0  # default 0
        self.pp_group_size = 1  # default 1
        self.pp_prev_rank = -1  # no previous stage
        self.pp_next_rank = -1  # no next stage
        self.pp_final_stage_rank = 0  # default 0

        if tp_config is not None:
            assert (
                tp_config.degree > 1
            ), "Tensor parallelism requires a degree > 1, otherwise use None"
            raise NotImplementedError("Tensor parallelism is not implemented")
        if pp_config is not None:
            assert (
                pp_config.degree > 1
            ), "Pipeline parallelism requires a degree > 1, otherwise use None"
            # raise NotImplementedError("Pipeline parallelism is not implemented")
        if cp_config is not None:
            assert (
                cp_config.degree > 1
            ), "Context parallelism requires a degree > 1, otherwise use None"
            if getattr(model, "tbo", False):
                raise OLMoConfigurationError(
                    "OLMoDDPTrainModule does not support context parallelism with "
                    "two-batch overlap yet"
                )
            if ep_config is not None and model.count_non_rowwise_ep_no_sync_blocks() > 0:
                raise OLMoConfigurationError(
                    "OLMoDDPTrainModule does not support context parallelism with "
                    "legacy EP no-sync yet; use rowwise EP no-sync for CP"
                )
        # if ac_config is not None:
        #     raise OLMoConfigurationError("In OLMoDDPTrainModule, activation checkpointing is controlled by the model, not the train module.")
        if float8_config is not None and float8_config.enabled:
            raise NotImplementedError("Float8 quantization is not implemented")

        assert dp_config is not None, "Data parallel config is required for OLMoDDPTrainModule"
        assert dp_config.name == "ddp", "Data parallel config must be 'ddp'"
        self.use_reduce_scatter = dp_config.use_reduce_scatter
        if self.use_reduce_scatter and cp_config is not None:
            raise OLMoConfigurationError(
                "Normal-parameter reduce-scatter is not yet supported with context "
                "parallelism. Dense gradients reduce over DP x CP, while optimizer "
                "state is currently sharded over DP only."
            )
        if self.use_reduce_scatter and not dp_config.only_allreduce_last_microbatch:
            raise OLMoConfigurationError(
                "Normal-parameter reduce-scatter requires gradient synchronization "
                "only on the final microbatch. Set "
                "dp_config.only_allreduce_last_microbatch=True."
            )
        if (
            self.use_reduce_scatter
            and pp_config is not None
            and not pp_config.use_custom_stage_implementation
        ):
            raise OLMoConfigurationError(
                "Normal-parameter reduce-scatter with pipeline parallelism requires "
                "use_custom_stage_implementation=True so non-final local backwards "
                "run under MultiGroupDDP.no_sync()."
            )

        if not dp_config.only_allreduce_last_microbatch:
            # MultiGroupDistributedDataParallel accumulates gradients in-place into flat bucket
            # views, so a bucket can be all-reduced exactly once per accumulation window. Reducing
            # on every micro-batch would re-reduce already-reduced grads (in-flight buckets are
            # still being written), leaving gradients unreduced or corrupted. Non-final micro-batches
            # must therefore stay under ``no_sync()`` and reduce only on the last one.
            raise OLMoConfigurationError(
                "OLMoDDPTrainModule requires only_allreduce_last_microbatch=True; "
                "per-micro-batch all-reduce is not supported by MultiGroupDistributedDataParallel."
            )

        if (
            state_dict_save_opts is not None
            or state_dict_load_opts is not None
            or load_key_mapping is not None
        ):
            # The direct checkpoint save/load path manages its own planners and key resolution and
            # does not consume these options, so accepting them would silently ignore requested
            # strictness / CPU offload / key remapping. Reject until they are actually wired up.
            raise OLMoConfigurationError(
                "OLMoDDPTrainModule does not support state_dict_save_opts, state_dict_load_opts, "
                "or load_key_mapping; its direct checkpoint save/load ignores them. Leave them unset."
            )

        ########################## Validate arguments. [END] ##########################

        if is_distributed():
            self._build_world_mesh(
                dp=dp_config,
                tp=tp_config,
                cp=cp_config,
                ep=ep_config,
                pp=pp_config,
                device_type=self.device.type,
            )
            self.dp_world_size = get_world_size(self.dp_process_group)
            log.info(f"Data parallel world size = {self.dp_world_size:,d}")
            assert self.world_mesh["dense"] is not None
        else:
            raise OLMoConfigurationError("Training parallelism is required for OLMoDDPTrainModule")

        self._dp_config = dp_config
        self._cp_config = cp_config
        self._tp_config = tp_config
        self._ep_config = ep_config
        self._pp_config = pp_config

        # Capture num_flops_per_token from the full unsplit model before parallelize_and_init_model
        # shards/splits it. Under PP each rank only holds its stage's layers, so querying the
        # (sharded) model parts would undercount. Uses the model-level accounting
        # (Transformer.num_flops_per_token = Σ block + lm_head); the v2 block implements the MoE
        # formula (routed + shared experts, top-k).
        self._full_model_num_flops_per_token = model.num_flops_per_token

        # Parallelize model.
        self.model_parts = self.parallelize_and_init_model(
            model,
            compile_model=compile_model,
            float8_config=float8_config,
            dp_config=dp_config,
            tp_config=tp_config,
            cp_config=cp_config,
            ep_config=ep_config,
            ac_config=ac_config,
            pp_config=pp_config,
            eval_only=eval_only,
        )

        import torch._dynamo.config as dynamo_cfg

        dynamo_cfg.recompile_limit = 64  # or any higher number you want

        if compile_model:
            self.compile_model()

        self._dp_config = dp_config
        self._cp_config = cp_config
        self._tp_config = tp_config
        self._ep_config = ep_config
        self._pp_config = pp_config

        self.label_ignore_index = label_ignore_index
        self.z_loss_multiplier = z_loss_multiplier

        self.scheduler = scheduler
        self.state_dict_save_opts = state_dict_save_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, cpu_offload=True
        )
        self.state_dict_load_opts = state_dict_load_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, strict=True
        )
        self.load_key_mapping = load_key_mapping
        self.reset_optimizer_states_on_load = reset_optimizer_states_on_load
        self.reset_optimizer_states_on_resume = reset_optimizer_states_on_resume

        self.optim = None
        if not self.eval_only:
            # Build optimizer(s).
            log.info("Building optimizer...")

            from olmo_core.optim.moe_optimizer import OLMoDDPOptimizerConfig

            assert isinstance(optim, OLMoDDPOptimizerConfig)
            optim = cast(OLMoDDPOptimizerConfig, optim)
            self.optim = optim.build(
                self.model_parts,
                self,
                strict=False,  # group_overrides might only be matched in one group, strict=False allows it to not match in one group (could match in some other group),
            )

            # MultiGroupDDP owns normal-parameter reduction. Keep the optimizer's
            # older post-backward reducer disabled so FP8WeightStore gradients stay
            # on their existing, independent synchronization path.
            if self.use_reduce_scatter:
                sharded_params = self.optim.normal_params_with_sharded_optimizer_state()
                unconfigured_params = set(sharded_params)
                for model_part in self.model_parts:
                    if not isinstance(model_part, MultiGroupDistributedDataParallel):
                        raise RuntimeError(
                            "use_reduce_scatter=True requires MultiGroupDistributedDataParallel."
                        )
                    local_sharded_params = sharded_params.intersection(
                        model_part.param_to_process_group
                    )
                    model_part.configure_reduce_scatter_params(local_sharded_params)
                    unconfigured_params.difference_update(local_sharded_params)
                if unconfigured_params:
                    raise RuntimeError(
                        "Some optimizer-sharded normal parameters are not managed by "
                        "any MultiGroupDistributedDataParallel model part."
                    )
        else:
            log.info("Skipping optimizer build because eval_only=True")

    def _require_optimizer(self):
        if self.optim is None:
            raise RuntimeError(
                f"{self.__class__.__name__} was built with eval_only=True and has no optimizer"
            )
        return self.optim

    @property
    def model(self) -> Transformer:
        if self.pp_enabled or len(self.model_parts) != 1:
            raise RuntimeError(
                f"{type(self).__name__}.model is only valid without pipeline parallelism; "
                f"got {len(self.model_parts)} model parts. Use model_parts instead."
            )
        return self.model_parts[0]

    def _cast_to_fwd_bwd_precision(self, model: Union[OLMoDDPModel, List[OLMoDDPModel]]) -> None:
        """
        Cast the model to the forward and backward precision.
        """
        if isinstance(model, list):
            for m in model:
                self._cast_to_fwd_bwd_precision(m)
        else:
            # if the model has implemented `cast_to_fwd_bwd_precision` method, call it.
            if hasattr(model, "_cast_to_fwd_bwd_precision"):
                assert callable(
                    model._cast_to_fwd_bwd_precision
                ), "model._cast_to_fwd_bwd_precision must be callable"
                model._cast_to_fwd_bwd_precision()
            else:
                # if the model doesn't have `cast_to_fwd_bwd_precision`, we need to cast the parameters directly.
                for p in model.parameters():
                    p.data = p.data.to(torch.bfloat16)

    def _build_world_mesh(
        self,
        *,
        dp: Optional[DataParallelConfig] = None,
        tp: Optional[TensorParallelConfig] = None,
        cp: Optional[ContextParallelConfig] = None,
        pp: Optional[PipelineParallelConfig] = None,
        ep: Optional[ExpertParallelConfig] = None,
        device_type: Optional[str] = None,
    ) -> None:
        """
        Build a :class:`~torch.distributed.device_mesh.DeviceMesh` suitable for the given parallel strategies.

        .. seealso::
            Pass the mesh created by this function to any of the ``get_*_mesh()`` functions in
            this module to get the right sub-mesh for a any given parallel strategy.

            - :func:`get_dp_model_mesh()` gives you the 1 or 2D sub-mesh suitable for data parallel *model*
            wrappers like FSDP(2) or DDP.
            - :func:`get_dp_mesh()` gives you the 1D sub-mesh suitable for configuring *data loaders*.
            - :func:`get_tp_mesh()` gives you the 1D sub-mesh for tensor parallelism.
            - :func:`get_cp_mesh()` gives you the 1D sub-mesh for context parallelism.
            - :func:`get_pp_mesh()` gives you the 1D sub-mesh for pipeline parallelism.
            - :func:`get_ep_mesh()` gives you the 1D sub-mesh for expert parallelism.

        .. important::
            A data parallel config is required if any other parallel config is set.

        .. important::
            Not all parallel strategies are compatible with each other.

        :param dp: Data parallel config.
        :param tp: Tensor parallel config.
        :param cp: Context parallel config.
        :param pp: Pipeline parallel config.
        :param ep: Expert parallel config.
        :param device_type: The device type.

        :returns: The world mesh with a shape compatible with the given parallel configs.
        """

        if len(self.world_mesh) > 0:
            raise RuntimeError(
                "world mesh already exists! You can only call 'build_world_mesh' once!"
            )

        device_type = device_type or get_default_device().type
        dp_world_size = get_world_size()

        # no config, Pure DP
        # if pp is None and tp is None and cp is None and dp is None and ep is None:
        #     self.world_mesh = init_device_mesh(device_type, (dp_world_size,), mesh_dim_names=(MeshDimName.dp,))
        #     return

        if dp is None:
            raise OLMoConfigurationError(
                "Data parallel config is required in addition to expert/tensor/context/pipeline parallel configs"
            )

        # Validate parallelism degrees while adjusting the dense DP degree. For MoE
        # parallel folding, expert parallelism is formed over the folded dense
        # DP x CP pool for each PP stage, not over dense DP after CP has divided it.
        if pp is not None:
            if pp.degree <= 1 or dp_world_size % pp.degree != 0:
                raise OLMoConfigurationError(
                    f"{pp.__class__.__name__}.degree must be at least 1 and divide into the world size"
                )
            dp_world_size //= pp.degree
        folded_dp_cp_world_size = dp_world_size
        if cp is not None:
            if cp.degree <= 1 or dp_world_size % cp.degree != 0:
                raise OLMoConfigurationError(
                    f"{cp.__class__.__name__}.degree must be at least 1 and divide into the world size"
                )
            dp_world_size //= cp.degree
        if tp is not None:
            if tp.degree <= 1 or dp_world_size % tp.degree != 0:
                raise OLMoConfigurationError(
                    f"{tp.__class__.__name__}.degree must be at least 1 and divide into the world size"
                )
            dp_world_size //= tp.degree
        # A negative EP degree is the "use all remaining ranks" shorthand: expert parallelism
        # spans the whole folded DP x CP pool.
        ep_degree = folded_dp_cp_world_size if ep is not None and ep.degree < 0 else None
        if ep is not None:
            if ep_degree is None:
                ep_degree = ep.degree
            if ep_degree <= 1 or folded_dp_cp_world_size % ep_degree != 0:
                raise OLMoConfigurationError(
                    f"{ep.__class__.__name__}.degree must be at least 1 and divide into the "
                    f"folded DP x CP world size, got folded_dp_cp_world_size={folded_dp_cp_world_size} and ep.degree={ep.degree}"
                )

        # Build up dense mesh dimensions. (PP, DP, CP)
        names: List[str] = []
        dims: List[int] = []

        # Pipeline parallel first.
        use_paired_pp = False  # TODO: move to config
        # TODO 2: implement paired pp properly
        if pp is not None:
            names.append(MeshDimName.pp)
            if use_paired_pp:
                dims.append(pp.degree // 2)
            else:
                dims.append(pp.degree)

        # Then data parallel.
        names.append(MeshDimName.dp)
        dims.append(dp_world_size)

        # Context parallel ranks share data-loader batches with the same DP rank,
        # but participate in dense parameter gradient synchronization.
        if cp is not None:
            names.append(MeshDimName.cp)
            dims.append(cp.degree)

        if pp is not None and use_paired_pp:
            names.append("pp_paired")
            dims.append(2)

        with torch.device("cpu"):
            mesh = torch.arange(math.prod(tuple(dims)), dtype=torch.int).view(tuple(dims))

        if pp is not None and use_paired_pp:
            r = mesh.permute(0, 2, 1).contiguous()
            pp_dim, pp_paired, dp_dim = r.shape
            r_merged = r.reshape(pp_dim * pp_paired, dp_dim)  # (pp*pp_paired, dp)
            mesh = r_merged
            names.remove("pp_paired")

        self.dense_mesh = DeviceMesh(
            device_type=device_type,
            mesh=mesh,
            mesh_dim_names=tuple(names),
        )
        # self.dense_mesh = init_device_mesh(device_type, tuple(dims), mesh_dim_names=tuple(names))
        log.info(f"Built dense_mesh {get_device_mesh_info(self.dense_mesh)}")

        if ep is None:  # EP not used
            self.moe_mesh = None
            log.info("Built moe_mesh None")

        else:
            # Build up moe mesh dimensions using MoE parallel folding. The dense
            # DP x CP pool is reinterpreted as EP_DP x EP_MP so CP does not
            # multiply the minimum GPU requirement for EP.
            names = []

            # Pipeline parallel first.
            if pp is not None:
                names.append(MeshDimName.pp)

            assert ep_degree is not None
            ep_dp_world_size = folded_dp_cp_world_size // ep_degree
            names.append(MeshDimName.ep_dp)
            names.append(MeshDimName.ep_mp)

            with torch.device("cpu"):
                dense_rank_grid = self.dense_mesh.mesh.detach().cpu().to(torch.int)
                if pp is not None:
                    pp_world_size_for_layout = pp.degree // 2 if use_paired_pp else pp.degree
                    folded_rank_grid = dense_rank_grid.reshape(
                        pp_world_size_for_layout,
                        folded_dp_cp_world_size,
                    )
                    mesh = folded_rank_grid.reshape(
                        pp_world_size_for_layout,
                        ep_dp_world_size,
                        ep_degree,
                    )
                else:
                    folded_rank_grid = dense_rank_grid.reshape(folded_dp_cp_world_size)
                    mesh = folded_rank_grid.reshape(ep_dp_world_size, ep_degree)

            device_mesh = DeviceMesh(
                device_type=device_type,
                mesh=mesh,
                mesh_dim_names=tuple(names),
            )
            self.moe_mesh = device_mesh
            # self.moe_mesh = init_device_mesh(device_type, tuple(dims), mesh_dim_names=tuple(names))

            log.info(f"Built moe_mesh{get_device_mesh_info(self.moe_mesh)}")

        # build process group

        if pp is not None:
            # self.dense_mesh['pp'] and self.moe_mesh['pp'] should be the same
            self.pp_group = self.dense_mesh["pp"].get_group()  #

        self.dp_group = self.dense_mesh["dp"].get_group()
        if cp is not None:
            self.cp_group = self.dense_mesh["cp"].get_group()
            self.dense_dp_cp_group, _dense_dp_cp_groups = self._build_mesh_dim_process_group(
                self.dense_mesh,
                (MeshDimName.dp, MeshDimName.cp),
                group_desc="moe_v2_dense_dp_cp",
            )
        else:
            self.cp_group = None
            self.dense_dp_cp_group = self.dp_group
        if self.moe_mesh is not None:
            self.ep_dp_group = self.moe_mesh["ep_dp"].get_group()
            self.ep_mp_group = self.moe_mesh["ep_mp"].get_group()
            # Under parallel folding, CP is already folded into the MoE EP-DP axis.
            self.expert_param_group = self.ep_dp_group

        self.world_mesh = {
            "dense": self.dense_mesh,
            "moe": self.moe_mesh,
            "dense_cpu": cpu_mesh_like(self.dense_mesh),
            "moe_cpu": None if self.moe_mesh is None else cpu_mesh_like(self.moe_mesh),
        }

    @staticmethod
    def _mesh_dim_rank_groups(device_mesh: DeviceMesh, dims: Sequence[str]) -> List[List[int]]:
        if device_mesh.mesh_dim_names is None:
            raise RuntimeError("could not build process groups without mesh dimension names")

        dim_names = tuple(device_mesh.mesh_dim_names)
        target_dims = tuple(str(dim) for dim in dims)
        missing_dims = [dim for dim in target_dims if dim not in dim_names]
        if missing_dims:
            raise RuntimeError(
                f"could not build process groups for dimensions {target_dims} from mesh "
                f"with dimensions {dim_names}"
            )

        target_dim_indices = [dim_names.index(dim) for dim in target_dims]
        other_dim_indices = [idx for idx in range(len(dim_names)) if idx not in target_dim_indices]

        rank_grid = device_mesh.mesh.detach().cpu().to(torch.int64)
        permuted = rank_grid.permute(*(other_dim_indices + target_dim_indices)).contiguous()
        target_shape = tuple(rank_grid.shape[idx] for idx in target_dim_indices)
        target_size = math.prod(target_shape)

        if not other_dim_indices:
            return [[int(rank) for rank in permuted.reshape(target_size).tolist()]]

        rank_groups: List[List[int]] = []
        other_shape = tuple(rank_grid.shape[idx] for idx in other_dim_indices)
        for index in product(*(range(size) for size in other_shape)):
            group = permuted[index].reshape(target_size).tolist()
            rank_groups.append([int(rank) for rank in group])
        return rank_groups

    @classmethod
    def _build_mesh_dim_process_group(
        cls,
        device_mesh: DeviceMesh,
        dims: Sequence[str],
        *,
        group_desc: str,
    ) -> Tuple[ProcessGroup, List[ProcessGroup]]:
        current_group, all_groups = dist.new_subgroups_by_enumeration(
            cls._mesh_dim_rank_groups(device_mesh, dims),
            group_desc=group_desc,
        )
        if current_group == dist.GroupMember.NON_GROUP_MEMBER:
            raise RuntimeError(
                f"Current rank is not in any process group for mesh dimensions {tuple(dims)}"
            )
        return cast(ProcessGroup, current_group), cast(List[ProcessGroup], all_groups)

    @property
    def dp_process_group(self) -> Optional[dist.ProcessGroup]:
        return self.dp_group

    @property
    def eval_batch_spec(self) -> EvalBatchSpec:
        return EvalBatchSpec(
            self.rank_microbatch_size,
            # max(self.rank_microbatch_size // 2, 1 * self.max_sequence_length),
            # 1 * self.max_sequence_length,
            max_sequence_length=self.max_sequence_length,
            #  fixed_sequence_length=self.tp_enabled,
        )

    @property
    def dp_config(self) -> TransformerDataParallelConfig:
        return self._dp_config

    @property
    def tp_enabled(self) -> bool:
        return self._tp_config is not None

    @property
    def cp_enabled(self) -> bool:
        return self._cp_config is not None

    @property
    def ep_enabled(self) -> bool:
        return self._ep_config is not None

    @property
    def pp_enabled(self) -> bool:
        return self._pp_config is not None

    @property
    def train_pp_schedule(self) -> PipelineSchedule:
        self.trainer  # make sure trainer has been attached before trying to access this
        assert self._train_pp_schedule is not None
        return self._train_pp_schedule

    @cached_property
    def world_size(self) -> int:
        return get_world_size()

    @cached_property
    def _reduce_divide_factor(self) -> float:
        return get_reduce_divide_factor(self.world_size)

    def on_attach(self):
        # Validate batch size.
        dp_ws = get_world_size(self.trainer.dp_process_group)
        if self.trainer.global_batch_size % (self.rank_microbatch_size * dp_ws) != 0:
            # raise OLMoConfigurationError(
            #     f"global batch size ({self.trainer.global_batch_size:,d}) must be divisible by "
            #     f"micro-batch size ({self.rank_microbatch_size:,d}) x DP world size ({dp_ws})"
            # )
            pass  # BUG: when batch size warmup + load checkpoint

        if self.pp_enabled:
            # Initialize pipeline schedule.
            assert self._train_pp_schedule is None  # make sure we don't initialize this twice
            assert self._pp_stages is not None
            assert self._pp_config is not None
            assert self.world_mesh["dense"] is not None
            pp_mesh = self.world_mesh["dense"]["pp"]
            assert pp_mesh is not None

            # Determine the number of micro-batches.
            rank_batch_size = self.trainer.global_batch_size // dp_ws
            num_microbatches = rank_batch_size // self.rank_microbatch_size

            self._train_pp_schedule = PipelineSchedule(
                model_parts=self.model_parts,  # type: ignore[arg-type]
                stages=self._pp_stages,
                pp_mesh=pp_mesh,
                schedule_name=self._pp_config.schedule,
                forward_pull_ahead_extra_activations=self._pp_config.forward_pull_ahead_extra_activations,
                num_microbatches=num_microbatches,
            )
            if not self._rowwise_lifetime_lease_slots_env_is_set():
                self._prewarm_ep_no_sync_symm_buffers(
                    model_parts=self.model_parts,
                    rank_microbatch_size=self.rank_microbatch_size,
                    rowwise_lifetime_lease_slots=self._estimate_pp_rowwise_lifetime_lease_slots_for_model_parts(),
                )
            self._set_ep_no_sync_symm_initialization_complete(True)

    @staticmethod
    def _rowwise_lifetime_lease_slots_env_is_set() -> bool:
        return any(
            os.getenv(name) is not None
            for name in (
                "OLMO_MOE_ROWWISE_LIFETIME_LEASE_SLOTS",
                "OLMO_MOE_ROWWISE_DISPATCH_OUT_LEASE_SLOTS",
            )
        )

    def _estimate_pp_rowwise_lifetime_lease_slots_by_stage(self) -> Dict[int, int]:
        if self._train_pp_schedule is None:
            return {}
        schedule_impl = getattr(self._train_pp_schedule, "schedule_impl", None)
        pipeline_order = getattr(schedule_impl, "pipeline_order", None)
        rank = getattr(schedule_impl, "rank", None)
        if pipeline_order is None or rank is None or rank not in pipeline_order:
            return {}

        from olmo_core.train.train_module.transformer.pipeline.pipeline_schedule import (
            PipelineActionType,
        )

        active_by_stage: Dict[int, int] = {}
        high_water_by_stage: Dict[int, int] = {}
        for action in pipeline_order[rank]:
            if action is None:
                continue
            if action.computation_type == PipelineActionType.FORWARD:
                stage_active = active_by_stage.get(action.stage_index, 0) + 1
                active_by_stage[action.stage_index] = stage_active
                high_water_by_stage[action.stage_index] = max(
                    high_water_by_stage.get(action.stage_index, 0),
                    stage_active,
                )
            elif action.computation_type == PipelineActionType.FULL_BACKWARD:
                active_by_stage[action.stage_index] = max(
                    0,
                    active_by_stage.get(action.stage_index, 0) - 1,
                )
        return {stage_idx: max(1, slots) for stage_idx, slots in high_water_by_stage.items()}

    def _estimate_pp_rowwise_lifetime_lease_slots_for_model_parts(self) -> List[int]:
        if self._pp_stages is None:
            return [1 for _ in self.model_parts]

        high_water_by_stage = self._estimate_pp_rowwise_lifetime_lease_slots_by_stage()
        fallback = max(1, int(getattr(self._train_pp_schedule, "num_microbatches", 1)))
        slots_by_part = [
            max(1, int(high_water_by_stage.get(stage.stage_index, fallback)))
            for stage in self._pp_stages
        ]
        if len(slots_by_part) != len(self.model_parts):
            raise RuntimeError(
                "Could not size rowwise lifetime lease slots per PP model part: "
                f"got {len(slots_by_part)} stages for {len(self.model_parts)} model parts"
            )
        log.info(
            "Prewarming rowwise lifetime lease slots per local PP stage: %s",
            {stage.stage_index: slots for stage, slots in zip(self._pp_stages, slots_by_part)},
        )
        return slots_by_part

    def _pp_dry_run_num_microbatches(self, full_num_microbatches: int) -> int:
        raw_value = os.getenv("OLMO_PP_DRY_RUN_MICROBATCHES")
        if raw_value is None:
            requested = 2 * max(1, self.pp_group_size)
        else:
            normalized = raw_value.strip().lower()
            if normalized in {"", "0", "false", "no", "off", "full"}:
                return full_num_microbatches
            try:
                requested = int(normalized)
            except ValueError as exc:
                raise OLMoConfigurationError(
                    "OLMO_PP_DRY_RUN_MICROBATCHES must be a positive integer, " "'0', or 'full'"
                ) from exc
            if requested <= 0:
                return full_num_microbatches

        return max(1, min(full_num_microbatches, requested))

    def _pp_dry_run_mode(self) -> str:
        # TODO: consider using the config system than environment variables for this
        raw_value = os.getenv("OLMO_PP_DRY_RUN_MODE", "independent")
        normalized = raw_value.strip().lower().replace("-", "_")
        if normalized in {
            "",
            "independent",
            "local",
            "stage",
            "stage_local",
            "accelerated",
            "hybrid",
        }:
            return "independent"
        if normalized in {"full", "legacy", "full_pp", "true_pp"}:
            return "full"
        if normalized in {
            "reduced",
            "reduced_pp",
            "reduced_true_pp",
            "debug_reduced",
            "debug_reduced_pp",
        }:
            return "reduced_true_pp"
        raise OLMoConfigurationError(
            "OLMO_PP_DRY_RUN_MODE must be one of: independent, full/true_pp, " "or reduced_true_pp"
        )

    def _pp_full_num_microbatches(self) -> int:
        assert self.pp_enabled
        assert self._train_pp_schedule is not None

        schedule_impl = getattr(self._train_pp_schedule, "schedule_impl", None)
        return int(
            getattr(schedule_impl, "_n_microbatches", self._train_pp_schedule.num_microbatches)
        )

    @staticmethod
    def _slice_pp_batch_dim(value: Any, original_batch_size: int, batch_size: int) -> Any:
        if isinstance(value, torch.Tensor) and value.size(0) == original_batch_size:
            return value[:batch_size].contiguous()
        if isinstance(value, list) and len(value) == original_batch_size:
            return value[:batch_size]
        if isinstance(value, tuple) and len(value) == original_batch_size:
            return value[:batch_size]
        return value

    def _maybe_reduce_pp_dry_run_batch(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        model_kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], Optional[int]]:
        assert self.pp_enabled

        full_num_microbatches = self._pp_full_num_microbatches()
        dry_run_num_microbatches = self._pp_dry_run_num_microbatches(full_num_microbatches)
        if dry_run_num_microbatches >= full_num_microbatches:
            return input_ids, labels, model_kwargs, None

        original_batch_size = input_ids.size(0)
        if original_batch_size % full_num_microbatches != 0:
            raise OLMoConfigurationError(
                "Cannot reduce PP dry-run microbatches without changing per-microbatch shape: "
                f"input batch size {original_batch_size} is not divisible by "
                f"full_num_microbatches={full_num_microbatches}"
            )

        instances_per_microbatch = original_batch_size // full_num_microbatches
        dry_run_batch_size = dry_run_num_microbatches * instances_per_microbatch

        log.info(
            "Reducing PP dry-run from %d to %d microbatches "
            "(rank batch %d -> %d, per-microbatch batch size %d)",
            full_num_microbatches,
            dry_run_num_microbatches,
            original_batch_size,
            dry_run_batch_size,
            instances_per_microbatch,
        )

        return (
            input_ids[:dry_run_batch_size].contiguous(),
            labels[:dry_run_batch_size].contiguous(),
            {
                key: self._slice_pp_batch_dim(value, original_batch_size, dry_run_batch_size)
                for key, value in model_kwargs.items()
            },
            dry_run_num_microbatches,
        )

    @staticmethod
    def _clear_pp_stage_dry_run_runtime(stage: Any) -> None:
        for attr_name in (
            "fwd_cache",
            "bwd_cache",
            "received_activations",
            "received_grads",
            "stage_outputs",
        ):
            runtime_state = getattr(stage, attr_name, None)
            if runtime_state is not None and hasattr(runtime_state, "clear"):
                runtime_state.clear()

        clear_step_info = getattr(stage, "clear_step_info", None)
        if callable(clear_step_info):
            clear_step_info()

    def _split_pp_dry_run_model_kwargs(
        self,
        kwargs: Dict[str, Any],
        *,
        original_batch_size: int,
        micro_batch_size: int,
        num_microbatches: int,
    ) -> List[Dict[str, Any]]:
        kwargs_mbs: List[Dict[str, Any]] = [{} for _ in range(num_microbatches)]
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.size(0) == original_batch_size:
                for mb_idx in range(num_microbatches):
                    start = mb_idx * micro_batch_size
                    end = start + micro_batch_size
                    kwargs_mbs[mb_idx][key] = value[start:end].contiguous()
            else:
                for kwargs_mb in kwargs_mbs:
                    kwargs_mb[key] = value
        return kwargs_mbs

    def _run_independent_pp_dry_run(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        batch_num_tokens_for_loss: Union[torch.Tensor, int, float],
        **kwargs,
    ) -> None:
        # TODO: this function is long. Need to explain the context and what each section does.
        assert self.pp_enabled
        assert self._pp_stages is not None

        full_num_microbatches = self._pp_full_num_microbatches()
        original_batch_size = input_ids.size(0)
        if original_batch_size % full_num_microbatches != 0:
            raise OLMoConfigurationError(
                "Cannot run independent PP dry-run without preserving per-microbatch shape: "
                f"input batch size {original_batch_size} is not divisible by "
                f"full_num_microbatches={full_num_microbatches}"
            )

        micro_batch_size = original_batch_size // full_num_microbatches
        if micro_batch_size <= 0:
            raise OLMoConfigurationError(
                "Cannot run independent PP dry-run with empty pipeline microbatches"
            )

        supported_model_kwargs = {"cp_already_sharded", "cp_original_seq_len"}
        unexpected_model_kwargs = set(kwargs) - supported_model_kwargs
        if unexpected_model_kwargs:
            raise OLMoConfigurationError(
                "Independent PP dry-run only supports the same model kwargs as the PP "
                f"schedule splitter, got unsupported keys: {sorted(unexpected_model_kwargs)}"
            )

        input_id_mbs = [
            input_ids[start : start + micro_batch_size].contiguous()
            for start in range(0, original_batch_size, micro_batch_size)
        ]
        label_mbs = [
            labels[start : start + micro_batch_size].contiguous()
            for start in range(0, original_batch_size, micro_batch_size)
        ]
        kwargs_mbs = self._split_pp_dry_run_model_kwargs(
            kwargs,
            original_batch_size=original_batch_size,
            micro_batch_size=micro_batch_size,
            num_microbatches=full_num_microbatches,
        )

        log.info(
            "Running independent PP dry-run on %d local stage(s) "
            "(microbatches=%d, per-microbatch batch size=%d, seqlen=%d)",
            len(self._pp_stages),
            full_num_microbatches,
            micro_batch_size,
            input_ids.size(1),
        )

        with self._model_forward_context():
            for stage_idx, stage in enumerate(self._pp_stages):
                self._print_dry_run_stage_progress(
                    stage_idx,
                    len(self._pp_stages),
                    mode="PP independent",
                    global_stage_idx=stage.stage_index,
                )
                self._clear_pp_stage_dry_run_runtime(stage)
                stage.prepare_step(
                    global_batch_size=original_batch_size,
                    micro_batch_size=micro_batch_size,
                    seqlen=input_ids.size(1),
                )

                try:
                    args_mb = (input_id_mbs[0],) if stage.is_first else ()
                    stage_kwargs = {
                        **kwargs_mbs[0],
                        "labels": label_mbs[0],
                        "ignore_index": self.label_ignore_index,
                        "loss_reduction": "sum",
                        "z_loss_multiplier": self.z_loss_multiplier,
                        "loss_div_factor": batch_num_tokens_for_loss,
                        "return_logits": False,
                    }
                    stage._prepare_forward_backward_meta(1, args_mb, stage_kwargs)

                    for mb_idx in range(full_num_microbatches):
                        self._print_dry_run_microbatch_progress(
                            mb_idx,
                            full_num_microbatches,
                            mode=f"PP independent stage {stage.stage_index}",
                        )
                        if stage.is_first:
                            args_mb = (input_id_mbs[mb_idx],)
                        else:
                            stage.received_activations[mb_idx] = torch.randn(
                                (
                                    micro_batch_size,
                                    input_ids.size(1),
                                    stage.d_model,
                                ),
                                device=stage.device,
                                dtype=stage.p2p_dtype,
                            )
                            args_mb = ()

                        stage_kwargs = {
                            **kwargs_mbs[mb_idx],
                            "labels": label_mbs[mb_idx],
                            "ignore_index": self.label_ignore_index,
                            "loss_reduction": "sum",
                            "z_loss_multiplier": self.z_loss_multiplier,
                            "loss_div_factor": batch_num_tokens_for_loss,
                            "return_logits": False,
                        }
                        is_last_microbatch = mb_idx == full_num_microbatches - 1
                        stage.forward_one_chunk(
                            mb_idx,
                            args_mb,
                            stage_kwargs,
                            last_forward=is_last_microbatch,
                        )
                        if not stage.is_last:
                            stage_output = stage.fwd_cache[mb_idx][1]
                            if isinstance(batch_num_tokens_for_loss, torch.Tensor):
                                grad_scale = batch_num_tokens_for_loss.detach().float().reciprocal()
                            else:
                                grad_scale = 1.0 / max(float(batch_num_tokens_for_loss), 1.0)
                            stage.received_grads[mb_idx] = torch.ones_like(stage_output).mul_(
                                grad_scale
                            )
                        stage.backward_one_chunk(
                            mb_idx,
                            last_backward=is_last_microbatch,
                        )
                finally:
                    self._clear_pp_stage_dry_run_runtime(stage)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        raise NotImplementedError("Use load_state_dict_direct instead")

    def state_dict_to_load(
        self, metadata: Metadata, *, optim: Optional[bool] = True
    ) -> Dict[str, Any]:
        raise NotImplementedError("Use load_state_dict_direct instead")

    def state_dict(self, *, optim: Optional[bool] = True) -> Dict[str, Any]:
        raise NotImplementedError("Use save_state_dict_direct or load_state_dict_direct instead")

    def save_state_dict_direct(
        self,
        dir: PathOrStr,
        *,
        process_group: Optional[dist.ProcessGroup] = None,
        save_overwrite: bool = False,
        thread_count: Optional[int] = None,
        process_count: Optional[int] = None,
        throttle_uploads: bool = False,
        compact_storage: bool = False,
        dedup_save_to_lowest_rank: bool = True,
        constant_memory_planning: bool = False,
        profile: bool = False,
    ) -> Optional[Dict[str, float]]:
        """Save native optimizer/model state and restore the live optimizer's storage.

        Defaults preserve the existing writer policy. ``compact_storage`` skips copies only
        for compact CPU storage; ``process_count`` selects the writer's existing spawn pool.
        ``constant_memory_planning`` computes metadata for ordinary contiguous shards
        arithmetically, with the default planner as fallback for other layouts.
        ``profile`` logs per-rank phases and summed worker time separately from wall time.
        CPU profiling uses wall time; CUDA profiling synchronizes this module's device.
        Profiling returns durations in seconds plus writer byte/item counts; otherwise returns None.
        """

        def timestamp():
            if profile:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                return time.perf_counter()
            return 0.0

        start = timestamp()
        timings: Dict[str, float] = {}
        optim = self._require_optimizer()
        # Validate the path and writer options before state_dict releases optimizer storage.
        dir = _prepare_env_for_save(dir, process_group=process_group, save_overwrite=save_overwrite)
        writer = RemoteFileSystemWriter(
            dir,
            thread_count=thread_count,
            process_count=process_count,
            process_group=process_group,
            throttle_uploads=throttle_uploads,
            compact_storage=compact_storage,
            profile=profile,
        )
        planner = FlatSavePlanner(
            dedup_save_to_lowest_rank=dedup_save_to_lowest_rank,
            constant_memory_planning=constant_memory_planning,
            profile=profile,
        )
        if profile:
            timings["prepare_seconds"] = timestamp() - start
        phase_start = timestamp()
        state_dict = optim.state_dict()
        if profile:
            timings["state_dict_seconds"] = timestamp() - phase_start
        try:
            phase_start = timestamp()
            save_dict = dict(state_dict)
            for key, buffer in self._persistent_model_buffer_state_dict().items():
                assert (
                    key not in save_dict
                ), f"Buffer key '{key}' collides with an optimizer state key"
                save_dict[key] = buffer
            if profile:
                timings["buffers_seconds"] = timestamp() - phase_start
            phase_start = timestamp()
            dist_cp.state_dict_saver.save(
                save_dict, storage_writer=writer, process_group=process_group, planner=planner
            )
            if profile:
                timings["distributed_save_seconds"] = timestamp() - phase_start
                timings.update(writer.timings)
                timings.update(planner.timings)
        finally:
            phase_start = timestamp()
            optim.load_state_dict(state_dict, reset_optimizer_moments_on_load=False)
            if profile:
                timings["optimizer_reload_seconds"] = timestamp() - phase_start
            phase_start = timestamp()
            torch.cuda.empty_cache()
            if profile:
                timings["empty_cache_seconds"] = timestamp() - phase_start
                timings["total_seconds"] = timestamp() - start
                log.info("checkpoint_save %s", json.dumps(timings, sort_keys=True))
        return timings if profile else None

    def load_state_dict_direct(
        self,
        dir: PathOrStr,
        *,
        process_group: Optional[dist.ProcessGroup] = None,
        pre_download: bool = False,
        work_dir: Optional[PathOrStr] = None,
        thread_count: Optional[int] = None,
        load_optim_state: Optional[bool] = True,
        reset_optimizer_states_on_load: Optional[bool] = None,
        constant_memory_planning: bool = False,
        profile: bool = False,
    ):
        from olmo_core.io import normalize_path

        if profile:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        load_start = time.perf_counter() if profile else 0.0
        load_pass_seconds = []

        def load_pass(*args, **kwargs):
            if not profile:
                return dist_cp.state_dict_loader.load(*args, **kwargs)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            phase_start = time.perf_counter()
            dist_cp.state_dict_loader.load(*args, **kwargs)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            load_pass_seconds.append(time.perf_counter() - phase_start)

        dir = normalize_path(dir)
        reader = RemoteFileSystemReader(
            dir, thread_count=thread_count, pre_download=pre_download, work_dir=work_dir
        )

        metadata = reader.read_metadata()
        checkpoint_keys = set(metadata.state_dict_metadata.keys())

        if self.eval_only:
            self._load_model_state_dict_direct(
                metadata,
                dir,
                reader,
                process_group,
                load=load_pass,
                constant_memory_planning=constant_memory_planning,
            )
        else:
            optim = self._require_optimizer()
            sd_to_load = optim.state_dict()
            has_optimizer_moments = any(
                key.endswith((".exp_avg", ".exp_avg_sq", ".muon_momentum", ".step"))
                for key in checkpoint_keys
            )
            has_optimizer_main_params = any(key.endswith(".main") for key in checkpoint_keys)
            has_model_params = any(key.startswith("model.") for key in checkpoint_keys)
            if load_optim_state is None:
                load_optim_state = has_optimizer_moments
            if reset_optimizer_states_on_load is None:
                reset_optimizer_states_on_load = self.reset_optimizer_states_on_load
            reset_optimizer_moments_on_load = getattr(
                optim, "reset_optimizer_moments_on_load", False
            )
            loaded_model_directly = False

            if not load_optim_state and has_model_params:
                log.info(
                    "Skipping optimizer state during checkpoint load; loading model weights directly"
                )
                self._load_model_state_dict_direct(
                    metadata,
                    dir,
                    reader,
                    process_group,
                    load=load_pass,
                    constant_memory_planning=constant_memory_planning,
                )
                optim._copy_model_params_to_main_params()
                optim._copy_main_params_to_mxfp8_weights()
                optim._refresh_rowwise_fp8_caches_from_model_params()
                loaded_model_directly = True
            elif not load_optim_state:
                if not has_optimizer_main_params:
                    raise RuntimeError(
                        "Checkpoint does not contain model weights or optimizer main params"
                    )
                log.info(
                    "Skipping optimizer state during checkpoint load; loading only optimizer main params"
                )
                sd_to_load = {
                    key: value for key, value in sd_to_load.items() if key.endswith(".main")
                }
            elif reset_optimizer_states_on_load:
                log.info(
                    "Resetting optimizer states during checkpoint load; loading only optimizer main params"
                )
                sd_to_load = {
                    key: value for key, value in sd_to_load.items() if key.endswith(".main")
                }
            else:
                # Backward compatibility: old checkpoints won't have rolling skip-step stats.
                for optional_key in (
                    optim.LOSSES_STATE_DICT_KEY,
                    optim.GRAD_NORMS_STATE_DICT_KEY,
                ):
                    if optional_key not in checkpoint_keys:
                        sd_to_load.pop(optional_key, None)
                        continue

                    optional_meta = metadata.state_dict_metadata[optional_key]
                    if isinstance(optional_meta, BytesStorageMetadata):
                        sd_to_load[optional_key] = []
                    elif isinstance(optional_meta, TensorStorageMetadata):
                        current_value = sd_to_load.get(optional_key)
                        if not isinstance(current_value, torch.Tensor) or tuple(
                            current_value.shape
                        ) != tuple(optional_meta.size):
                            sd_to_load[optional_key] = torch.empty(
                                tuple(optional_meta.size),
                                dtype=optional_meta.properties.dtype,
                                device=optim.device,
                            )

                if reset_optimizer_moments_on_load:
                    log.info(
                        "Resetting optimizer exp_avg and exp_avg_sq buffers during checkpoint load"
                    )
                    for key in list(sd_to_load.keys()):
                        if key.endswith(".exp_avg") or key.endswith(".exp_avg_sq"):
                            sd_to_load.pop(key)

            if not loaded_model_directly:
                from olmo_core.distributed.checkpoint.utils import (
                    finish_qk_expansion,
                    prepare_qk_expansion,
                )

                expansions = {}
                if self.expand_shared_qk_norm_on_load:
                    gain_shapes = {
                        name: tuple(param.shape)
                        for group in optim.param_groups
                        for name, param in group["named_params"].items()
                        if name.endswith((".q_norm.weight", ".k_norm.weight")) and param.ndim == 2
                    }
                    expansions = prepare_qk_expansion(sd_to_load, metadata, gain_shapes)
                load_pass(
                    sd_to_load,
                    checkpoint_id=dir,
                    storage_reader=reader,
                    process_group=process_group,
                    planner=(
                        contiguous_planner.ContiguousLoadPlanner()
                        if constant_memory_planning
                        else None
                    ),
                )

                finish_qk_expansion(sd_to_load, expansions)

                optim.load_state_dict(sd_to_load)

                # load into model params
                optim._copy_main_params_to_model_params()

        # Restore persistent model buffers (e.g. the router's score_bias) on both the eval and
        # training paths. These are model state updated outside the optimizer, so they are always
        # restored when present in the checkpoint, independent of optimizer-state handling above.
        buffers_to_load = {
            key: buffer
            for key, buffer in self._persistent_model_buffer_state_dict().items()
            if key in checkpoint_keys
        }
        if buffers_to_load:
            # Use a fresh reader for this separate load pass rather than reusing the one already
            # consumed by the optimizer/model load above.
            buffer_reader = RemoteFileSystemReader(
                dir, thread_count=thread_count, pre_download=pre_download, work_dir=work_dir
            )
            load_pass(
                buffers_to_load,
                checkpoint_id=dir,
                storage_reader=buffer_reader,
                process_group=process_group,
                planner=(
                    contiguous_planner.ContiguousLoadPlanner() if constant_memory_planning else None
                ),
            )

        torch.cuda.empty_cache()

        if profile:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            log.info(
                "checkpoint_load %s",
                json.dumps(
                    {
                        "total_seconds": time.perf_counter() - load_start,
                        "distributed_load_seconds": sum(load_pass_seconds),
                        "load_pass_seconds": load_pass_seconds,
                        "constant_memory_planning": constant_memory_planning,
                    },
                    sort_keys=True,
                ),
            )

    def _load_model_state_dict_direct(
        self,
        metadata: Metadata,
        checkpoint_id: PathOrStr,
        reader: RemoteFileSystemReader,
        process_group: Optional[ProcessGroup],
        *,
        load=None,
        constant_memory_planning: bool = False,
    ) -> None:
        from olmo_core.distributed.checkpoint.utils import (
            finish_qk_expansion,
            prepare_qk_expansion,
        )

        state = self._get_model_state_dict_for_eval_load(metadata)
        expansions = {}
        if self.expand_shared_qk_norm_on_load:
            gain_shapes = {}
            for part in self.model_parts:
                for name, param in part.named_parameters():
                    if param.ndim != 2 or not name.endswith((".q_norm.weight", ".k_norm.weight")):
                        continue
                    key = self._resolve_model_checkpoint_key(name, state.keys())
                    if key is not None:
                        gain_shapes[key.removesuffix(".main")] = tuple(param.shape)
            expansions = prepare_qk_expansion(state, metadata, gain_shapes)
        if load is None:
            load = dist_cp.state_dict_loader.load
        load(
            state,
            checkpoint_id=checkpoint_id,
            storage_reader=reader,
            process_group=process_group,
            planner=contiguous_planner.ContiguousLoadPlanner()
            if constant_memory_planning
            else None,
        )
        finish_qk_expansion(state, expansions)

    def _get_model_state_dict_for_eval_load(self, metadata: Metadata) -> Dict[str, Any]:
        model_state: Dict[str, Any] = {}
        checkpoint_keys = set(metadata.state_dict_metadata.keys())

        for model_part in self.model_parts:
            for name, param in model_part.named_parameters():
                checkpoint_key = self._resolve_model_checkpoint_key(name, checkpoint_keys)
                if checkpoint_key is None:
                    continue

                tensor_meta = metadata.state_dict_metadata[checkpoint_key]
                assert isinstance(tensor_meta, TensorStorageMetadata)
                global_numel = tensor_meta.size.numel()
                is_optimizer_main_param = checkpoint_key.endswith(".main")
                local_tensor = param.data.view(-1) if is_optimizer_main_param else param.data
                local_numel = local_tensor.numel()
                global_shape = (
                    (global_numel,) if is_optimizer_main_param else tuple(tensor_meta.size)
                )

                if (
                    self.expand_shared_qk_norm_on_load
                    and name.endswith((".q_norm.weight", ".k_norm.weight"))
                    and param.ndim == 2
                    and tuple(tensor_meta.size) == (param.shape[1],)
                ):
                    # Replicated Q/K gains are not EP shards. Keep the live destination;
                    # the load wrapper reads a temporary shared vector and expands it.
                    model_state[checkpoint_key] = local_tensor
                    continue

                if local_numel == global_numel:
                    model_state[checkpoint_key] = local_tensor
                else:
                    moe_mesh = self.world_mesh["moe"]
                    if moe_mesh is None:
                        raise RuntimeError(
                            f"Cannot load checkpoint tensor '{checkpoint_key}' for parameter "
                            f"'{name}' in eval mode with expert parallelism disabled: checkpoint "
                            f"has {global_numel:,d} elements but the model parameter has "
                            f"{local_numel:,d} elements. EP=1 eval can load EP-trained "
                            f"checkpoints when the checkpoint and model configs match; this "
                            f"shape mismatch usually means the eval config does not match the "
                            f"checkpoint."
                        )

                    ep_mp_size = moe_mesh["ep_mp"].size()
                    expected_global_numel = local_numel * ep_mp_size
                    if expected_global_numel != global_numel:
                        raise RuntimeError(
                            f"Cannot load checkpoint tensor '{checkpoint_key}' for sharded "
                            f"parameter '{name}' in eval mode: checkpoint has "
                            f"{global_numel:,d} elements, but the local parameter has "
                            f"{local_numel:,d} elements across EP-MP size {ep_mp_size:,d} "
                            f"({expected_global_numel:,d} total). This usually means the "
                            f"eval config does not match the checkpoint."
                        )

                    model_state[checkpoint_key] = DTensor.from_local(
                        local_tensor,
                        device_mesh=moe_mesh["ep_dp", "ep_mp"],
                        placements=[Replicate(), Shard(0)],
                        shape=global_shape,
                        stride=self._contiguous_stride(global_shape),
                        run_check=False,
                    )

        if not model_state:
            raise RuntimeError("Did not find any model weights to load in eval mode")

        return model_state

    @staticmethod
    def _contiguous_stride(shape: Sequence[int]) -> Tuple[int, ...]:
        stride: List[int] = []
        running = 1
        for dim in reversed(shape):
            stride.append(running)
            running *= int(dim)
        return tuple(reversed(stride))

    _MODEL_BUFFER_KEY_PREFIX: ClassVar[str] = "model_buffer."

    @staticmethod
    def _strip_wrapper_prefixes(name: str) -> str:
        while True:
            for prefix in ("module.", "_orig_mod."):
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    break
            else:
                return name

    def _persistent_model_buffer_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Map each persistent model buffer to a stable checkpoint key.

        Persistent buffers (e.g. the router's aux-loss-free ``score_bias``) hold learned state that
        is updated outside the optimizer, so they must be checkpointed alongside optimizer state.
        Keys are stripped of DDP / ``torch.compile`` wrapper prefixes so a checkpoint saved from the
        wrapped training model reloads into the unwrapped eval model and vice versa.
        """
        buffer_state: Dict[str, torch.Tensor] = {}
        for model_part in self.model_parts:
            for module_name, module in model_part.named_modules():
                for buffer_name, buffer in module._buffers.items():
                    if buffer is None or buffer_name in module._non_persistent_buffers_set:
                        continue
                    full_name = f"{module_name}.{buffer_name}" if module_name else buffer_name
                    key = self._MODEL_BUFFER_KEY_PREFIX + self._strip_wrapper_prefixes(full_name)
                    if key in buffer_state:
                        raise RuntimeError(
                            f"Duplicate persistent buffer checkpoint key '{key}'; buffer names "
                            "collide across model parts."
                        )
                    buffer_state[key] = buffer
        return buffer_state

    def _resolve_model_checkpoint_key(
        self, param_name: str, checkpoint_keys: Collection[str]
    ) -> Optional[str]:
        param_names: List[str] = []
        for candidate_name in (param_name, self._strip_wrapper_prefixes(param_name)):
            if candidate_name not in param_names:
                param_names.append(candidate_name)

        candidates: List[str] = []
        for candidate_name in param_names:
            candidates.extend(
                (
                    f"model.{candidate_name}",
                    f"model.module.{candidate_name}",
                    candidate_name,
                    f"module.{candidate_name}",
                    f"{candidate_name}.main",
                    f"module.{candidate_name}.main",
                )
            )
        for key in candidates:
            if key in checkpoint_keys:
                return key
        return None

    def _reset_dry_run_progress_timer(self) -> None:
        now = time.perf_counter()
        self._dry_run_progress_started_at = now
        self._dry_run_progress_last_log_at = now

    def _dry_run_progress_timing(self) -> Tuple[float, float]:
        now = time.perf_counter()
        started_at = getattr(self, "_dry_run_progress_started_at", now)
        last_log_at = getattr(self, "_dry_run_progress_last_log_at", started_at)
        self._dry_run_progress_last_log_at = now
        return now - last_log_at, now - started_at

    def _print_dry_run_microbatch_progress(
        self,
        microbatch_idx: int,
        num_microbatches: int,
        *,
        mode: str,
    ) -> None:
        if get_rank() == 0:
            elapsed, total = self._dry_run_progress_timing()
            print(
                f"[dry-run] {mode} microbatch {microbatch_idx + 1}/{num_microbatches} "
                f"(+{elapsed:.2f}s since last log, {total:.2f}s total)",
                flush=True,
            )

    def _print_dry_run_stage_progress(
        self,
        stage_idx: int,
        num_stages: int,
        *,
        mode: str,
        global_stage_idx: Optional[int] = None,
    ) -> None:
        if get_rank() == 0:
            elapsed, total = self._dry_run_progress_timing()
            stage_desc = f"{stage_idx + 1}/{num_stages}"
            if global_stage_idx is not None:
                stage_desc = f"{stage_desc} (global stage {global_stage_idx})"
            print(
                f"[dry-run] {mode} stage {stage_desc} "
                f"(+{elapsed:.2f}s since last log, {total:.2f}s total)",
                flush=True,
            )

    def _print_dry_run_progress_complete(self, *, mode: str) -> None:
        if get_rank() == 0:
            elapsed, total = self._dry_run_progress_timing()
            print(
                f"[dry-run] {mode} complete "
                f"(+{elapsed:.2f}s since last log, {total:.2f}s total)",
                flush=True,
            )

    def train_batch_with_loss(
        self,
        micro_batches,
        objective: objective_utils.Objective,
        context_factory=None,
        *,
        reset_auxiliary_metrics: bool = False,
    ):
        """Accumulate a caller-normalized objective with Core gradient synchronization.

        Leave ``reset_auxiliary_metrics=False`` to collect and manage model auxiliary
        metrics yourself. Set it to True to clear them before training and on exit,
        including failure, without changing auxiliary losses or gradients.
        """
        return objective_utils.train_batch_with_loss(
            self,
            micro_batches,
            objective,
            context_factory,
            reset_auxiliary_metrics=reset_auxiliary_metrics,
        )

    @nvtx.annotate("train_batch")
    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        self._require_optimizer()
        if dry_run:
            self._reset_dry_run_progress_timer()

        # Set model to train mode if it isn't already.
        for m in self.model_parts:
            m.train()

        # Generate labels.
        if "labels" not in batch:
            batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)

        # Record how many instances are going to be skipped (masked out).
        instance_mask = batch.get("instance_mask")
        if (instance_mask) is not None and not dry_run:
            self.record_metric(
                "train/masked instances (%)", (~instance_mask).float().mean(), ReduceType.mean
            )

            if (~instance_mask).all():
                print(
                    f"[Warning] rank {dist.get_rank()} All instances ({instance_mask.shape}) in the micro-batch are masked out"
                )

        #############################

        if not self.pp_enabled:
            # Calculate how many tokens are going to be used in the loss.
            batch_num_tokens_for_loss = (batch["labels"] != self.label_ignore_index).sum()

            if instance_mask is not None:
                # WARN: When we mask out instances with the instance filter, we count those tokens
                # for the loss anyways. They will count as tokens with a zero loss. This means we
                # get an artificially *low* loss for these batches. But it is really hard (and slow)
                # to do this properly in a distributed setup. We add back in the full number of tokens
                # for the loss so that each rank contributes to the loss calculation fairly.
                batch_num_tokens_for_loss += (~instance_mask).sum() * (
                    batch["labels"].shape[1] - 1
                )  # shifted labels does not count last token

            if batch_num_tokens_for_loss.item() == 0:
                print(f"[Warning] rank {dist.get_rank()} batch_num_tokens_for_loss == 0")

            batch_num_tokens_for_loss = move_to_device(batch_num_tokens_for_loss, self.device)
            if is_distributed():
                global_batch_num_tokens_for_loss = batch_num_tokens_for_loss.clone()
                dist.all_reduce(global_batch_num_tokens_for_loss, group=self.dp_process_group)
                batch_num_tokens_for_loss = global_batch_num_tokens_for_loss.clamp_min(1)
                batch_num_tokens_for_loss = batch_num_tokens_for_loss / get_world_size(
                    self.dp_process_group
                )
            else:
                batch_num_tokens_for_loss = batch_num_tokens_for_loss.clamp_min(1)

            # Batch losses to record.
            ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)
            z_batch_loss: Optional[torch.Tensor] = None
            if self.z_loss_multiplier is not None:
                z_batch_loss = move_to_device(torch.tensor(0.0), self.device)

            # Split into micro-batches.
            if self.rank_microbatch_size < (seq_len := batch["input_ids"].shape[1]):
                raise RuntimeError(
                    f"Microbatch size ({self.rank_microbatch_size}) is too small relative to sequence length ({seq_len})"
                )
            micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)
            num_micro_batches = len(micro_batches)

            # Train one micro-batch at a time.
            for micro_batch_idx, micro_batch in enumerate(micro_batches):
                if dry_run:
                    self._print_dry_run_microbatch_progress(
                        micro_batch_idx,
                        num_micro_batches,
                        mode="DDP",
                    )
                with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                    with nvtx.annotate(f"fwd_mb{micro_batch_idx}", color="blue"):
                        input_ids, labels, model_kwargs = self._prepare_batch(micro_batch)
                        # Run forward pass, get losses.
                        debug_dump_logits = self._debug_should_dump("logits")
                        lm_output = self.model_forward_no_pipeline(
                            input_ids,
                            labels=labels,
                            ignore_index=self.label_ignore_index,
                            loss_reduction="sum",
                            z_loss_multiplier=self.z_loss_multiplier,
                            loss_div_factor=batch_num_tokens_for_loss,
                            return_logits=debug_dump_logits,
                            **model_kwargs,
                        )
                        if debug_dump_logits:
                            self._debug_dump_lm_output(
                                lm_output,
                                input_ids=input_ids,
                                labels=labels,
                                micro_batch_idx=micro_batch_idx,
                            )
                        assert isinstance(lm_output, LMOutputWithLoss)
                        loss = lm_output.loss
                        ce_loss = lm_output.ce_loss
                        z_loss = lm_output.z_loss
                        del lm_output
                        # Update total batch CE and Z loss.
                        ce_batch_loss += get_local_tensor(ce_loss.detach())
                        del ce_loss
                        if z_batch_loss is not None:
                            assert z_loss is not None
                            z_batch_loss += get_local_tensor(z_loss.detach())
                            del z_loss

                    with nvtx.annotate(f"bwd_mb{micro_batch_idx}", color="red"):
                        # Run backward pass.
                        loss.backward()

            del batch  # In case this helps with memory utilization.

            for idx, model in enumerate(self.model_parts):
                model.finalize_grad_reduce()

        else:
            # pipeline parallel forward / backward
            # Run pipeline schedule.
            input_ids, labels, model_kwargs = self._prepare_batch(batch, batch["labels"])
            assert labels is not None
            dry_run_mode = self._pp_dry_run_mode() if dry_run else "full"
            dry_run_complete_mode = "PP"
            dry_run_num_microbatches = None

            # Calculate how many tokens are going to be used in the loss.
            batch_num_tokens_for_loss = (labels != self.label_ignore_index).sum()

            if instance_mask is not None and not dry_run:
                # WARN: When we mask out instances with the instance filter, we count those tokens
                # for the loss anyways. They will count as tokens with a zero loss. This means we
                # get an artificially *low* loss for these batches. But it is really hard (and slow)
                # to do this properly in a distributed setup. We add back in the full number of tokens
                # for the loss so that each rank contributes to the loss calculation fairly.
                batch_num_tokens_for_loss += (~instance_mask).sum() * (
                    batch["labels"].shape[1] - 1
                )  # shifted labels does not count last token

            if batch_num_tokens_for_loss.item() == 0:
                print(f"[Warning] rank {dist.get_rank()} batch_num_tokens_for_loss == 0")

            batch_num_tokens_for_loss = move_to_device(batch_num_tokens_for_loss, self.device)
            if is_distributed():
                global_batch_num_tokens_for_loss = batch_num_tokens_for_loss.clone()
                dist.all_reduce(global_batch_num_tokens_for_loss, group=self.dp_process_group)
                batch_num_tokens_for_loss = global_batch_num_tokens_for_loss.clamp_min(1)
                batch_num_tokens_for_loss = batch_num_tokens_for_loss / get_world_size(
                    self.dp_process_group
                )
            else:
                batch_num_tokens_for_loss = batch_num_tokens_for_loss.clamp_min(1)

            if dry_run and dry_run_mode == "reduced_true_pp":
                (
                    input_ids,
                    labels,
                    model_kwargs,
                    dry_run_num_microbatches,
                ) = self._maybe_reduce_pp_dry_run_batch(input_ids, labels, model_kwargs)
                batch_num_tokens_for_loss = (labels != self.label_ignore_index).sum()
                if batch_num_tokens_for_loss.item() == 0:
                    print(f"[Warning] rank {dist.get_rank()} batch_num_tokens_for_loss == 0")
                batch_num_tokens_for_loss = move_to_device(batch_num_tokens_for_loss, self.device)
                if is_distributed():
                    global_batch_num_tokens_for_loss = batch_num_tokens_for_loss.clone()
                    dist.all_reduce(global_batch_num_tokens_for_loss, group=self.dp_process_group)
                    batch_num_tokens_for_loss = global_batch_num_tokens_for_loss.clamp_min(1)
                    batch_num_tokens_for_loss = batch_num_tokens_for_loss / get_world_size(
                        self.dp_process_group
                    )
                else:
                    batch_num_tokens_for_loss = batch_num_tokens_for_loss.clamp_min(1)
                dry_run_complete_mode = "PP reduced true"
            elif dry_run and dry_run_mode == "independent":
                dry_run_complete_mode = "PP independent"
            elif dry_run and dry_run_mode == "full":
                dry_run_complete_mode = "PP full"

            input_ids, labels, model_kwargs = self._prepare_pipeline_context_parallel_batch(
                input_ids,
                labels,
                model_kwargs,
            )
            assert labels is not None
            ce_batch_loss = None
            z_batch_loss = None
            if dry_run and dry_run_mode == "independent":
                self._run_independent_pp_dry_run(
                    input_ids,
                    labels,
                    batch_num_tokens_for_loss,
                    **model_kwargs,
                )
            else:
                progress_callback = None
                if dry_run:
                    progress_mode = "PP reduced true" if dry_run_mode == "reduced_true_pp" else "PP"

                    def progress_callback(microbatch_idx, num_microbatches):
                        self._print_dry_run_microbatch_progress(
                            microbatch_idx,
                            num_microbatches,
                            mode=progress_mode,
                        )

                pp_outputs = self.run_pipeline(
                    input_ids,
                    labels,
                    batch_num_tokens_for_loss,
                    ignore_index=self.label_ignore_index,
                    loss_reduction="sum",
                    z_loss_multiplier=self.z_loss_multiplier,
                    return_logits=False,
                    num_microbatches=dry_run_num_microbatches,
                    progress_callback=progress_callback,
                    **model_kwargs,
                )
                # Collect losses from all micro-batches and all stages.
                for stage_outputs in pp_outputs:
                    for mb_output in stage_outputs:
                        if mb_output is None:  # non-last stage
                            continue
                        else:
                            assert isinstance(mb_output, LMOutputWithLoss)
                            _, loss, ce_loss, z_loss = mb_output

                            # ce_loss should always be not None
                            ce_batch_loss = (
                                (ce_batch_loss + ce_loss.detach())
                                if ce_batch_loss is not None
                                else ce_loss.detach()
                            )

                            # z loss is optional
                            if z_loss is not None:
                                z_batch_loss = (
                                    (z_batch_loss + z_loss.detach())
                                    if z_batch_loss is not None
                                    else z_loss.detach()
                                )

            for idx, model in enumerate(self.model_parts):
                model.finalize_grad_reduce()

        #############################

        for model in self.model_parts:
            model.post_batch(dry_run=dry_run)

        if dry_run:
            self._print_dry_run_progress_complete(
                mode=(
                    dry_run_complete_mode
                    if self.pp_enabled and "dry_run_complete_mode" in locals()
                    else "DDP"
                ),
            )
            for model in self.model_parts:
                model.reset_auxiliary_metrics()

            torch.cuda.empty_cache()
            from olmo_core.train.globals import set_global_arg

            set_global_arg("dry_run_done", True)
            return

        # Record loss metrics.

        optim = self._require_optimizer()
        with nvtx.annotate("record_metrics"):
            if self.pp_enabled:
                ce_batch_loss = self.reduce_send_recv(ce_batch_loss)
                # if ce_batch_loss > 20 or ce_batch_loss < 0.05:
                #     log.warning(f"Irregular CE loss detected: {ce_batch_loss.item()}")
                self.record_ce_loss(ce_batch_loss)
                optim.latest_loss = ce_batch_loss
            else:
                assert ce_batch_loss is not None, "CE loss should not be None"
                # Need to reduce the loss right away for the SkipStepOptimizer. NOTE: WHY?
                if is_distributed():
                    ce_batch_loss.div_(self._reduce_divide_factor)
                    dist.all_reduce(ce_batch_loss)
                    ce_batch_loss.div_(self.world_size)
                    ce_batch_loss.mul_(self._reduce_divide_factor)
                self.record_ce_loss(ce_batch_loss)
                optim.latest_loss = ce_batch_loss

            if z_batch_loss is not None:
                assert self.z_loss_multiplier is not None
                self.record_metric(
                    "Z loss",
                    z_batch_loss,
                    ReduceType.mean,
                    namespace="train",
                )
                self.record_metric(
                    "Z loss unscaled",
                    z_batch_loss / self.z_loss_multiplier,
                    ReduceType.mean,
                    namespace="train",
                )

            # And additional metrics.
            for m in self.model_parts:
                for metric_name, (metric_val, reduction) in m.compute_auxiliary_metrics(
                    reset=True
                ).items():
                    merge_strategy = MetricMergeStrategy.warn
                    if reduction in (ReduceType.sum, ReduceType.mean):
                        merge_strategy = MetricMergeStrategy.sum
                    elif reduction == ReduceType.max:
                        merge_strategy = MetricMergeStrategy.max
                    self.record_metric(
                        metric_name,
                        metric_val,
                        reduction,
                        namespace="train",
                        merge_strategy=merge_strategy,
                    )

        # dist.barrier()
        # if dist.get_rank() == 0:
        #     print("-------train batch end--------")

    def eval_batch(
        self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Optional[LMOutputWithLoss]]:
        # Currently all of our evaluators require the full logits locally,
        # but when we're using CP/TP we usually can't materialize the full logits locally (due to OOMs).
        # However we could at least support in-loop PPL evals with a little work in the evaluator
        # code to handle the sharded logits.
        if self.cp_enabled:
            raise RuntimeError(
                f"{self.__class__.__name__}.eval_batch() does not support context parallelism yet, "
                "please disable in-loop evals"
            )
        if self.tp_enabled:
            raise RuntimeError(
                f"{self.__class__.__name__}.eval_batch() does not support tensor parallelism yet, "
                "please disable in-loop evals"
            )

        input_ids, labels, model_kwargs = self._prepare_batch(batch, labels)

        for m in self.model_parts:
            m.eval()

        with self._eval_batch_context():
            if not self.pp_enabled:
                lm_output = self.model_forward_no_pipeline(
                    input_ids,
                    labels=labels,
                    ignore_index=self.label_ignore_index,
                    loss_reduction="none",
                    **model_kwargs,
                )
                assert isinstance(lm_output, LMOutputWithLoss), "Expected LMOutputWithLoss"
                return lm_output

            else:
                assert labels is not None
                lm_output = self.run_pipeline_eval(
                    input_ids,
                    labels,
                    batch_num_tokens_for_loss=None,
                    ignore_index=self.label_ignore_index,
                    loss_reduction="none",
                    **model_kwargs,
                )
                # merge lm_output from all stages and all micro-batches
                # List[List[Optional[LMOutputWithLoss]]] --> Optional[LMOutputWithLoss]
                final_lm_output: Optional[LMOutputWithLoss] = None
                logits_list = []
                loss_list = []
                ce_loss_list = []
                z_loss_list = []

                # NOTE: here we combine the multiple stages assuming they are different stages
                # with the same DP rank (e.g., stage 0 and stage 1 in X-stage PP, working on the same batch data).
                # so at max one of the stage_outputs will have non-None output.
                # TODO: this assumption might not hold in dualpipe and other schedules.
                for stage_outputs in lm_output:
                    for mb_output in stage_outputs:
                        if mb_output is None:
                            continue
                        else:
                            assert isinstance(mb_output, LMOutputWithLoss)
                            (
                                logits,  # [mb, seq_len, vocab_size]
                                loss,  # [mb, seq_len] because loss_reduction="none"
                                ce_loss,  # [mb, seq_len]
                                z_loss,  # ??
                            ) = mb_output

                            # always not None
                            loss_list.append(loss)
                            ce_loss_list.append(ce_loss)

                            # optional
                            if logits is not None:
                                logits_list.append(logits)
                            if z_loss is not None:
                                z_loss_list.append(z_loss)

                # Concatenate all logits and losses
                merged_logits = torch.cat(logits_list, dim=0) if len(logits_list) > 0 else None
                merged_loss = torch.cat(loss_list, dim=0) if len(loss_list) > 0 else None
                merged_ce_loss = torch.cat(ce_loss_list, dim=0) if len(ce_loss_list) > 0 else None
                merged_z_loss = torch.cat(z_loss_list, dim=0) if len(z_loss_list) > 0 else None

                if merged_loss is not None:
                    # has last stage output
                    assert merged_ce_loss is not None
                    final_lm_output = LMOutputWithLoss(
                        merged_logits,
                        merged_loss,
                        merged_ce_loss,
                        merged_z_loss,
                    )
                else:
                    # no last stage output
                    final_lm_output = None

                return final_lm_output

    def run_pipeline_eval(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> List[List[Optional[LMOutputWithLoss]]]:
        # the micro-batch size should be a multiple of pp degree
        def pad_dim_0_to_multiple_of(tensor: torch.Tensor, multiple: int) -> torch.Tensor:
            bsz = tensor.size(0)
            if bsz % multiple == 0:
                return tensor
            pad_size = multiple - (bsz % multiple)
            padding_tensor = (
                tensor[-1].unsqueeze(0).expand(pad_size, *tensor.size()[1:])
            )  # repeat last element
            padded_tensor = torch.cat([tensor, padding_tensor], dim=0).contiguous()
            return padded_tensor

        original_batch_size = input_ids.size(0)
        padded_input_ids = pad_dim_0_to_multiple_of(
            input_ids, self.train_pp_schedule.pp_mesh.size()
        )
        padded_labels = pad_dim_0_to_multiple_of(labels, self.train_pp_schedule.pp_mesh.size())
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor) and v.size(0) == original_batch_size:
                kwargs[k] = pad_dim_0_to_multiple_of(v, self.train_pp_schedule.pp_mesh.size())

        with self._model_forward_context():
            schedule_outputs = self.train_pp_schedule.step(
                padded_input_ids,
                target=padded_labels,
                forward_only=True,  # <<<--- eval mode
                # kwargs from now ---
                # loss_div_factor=batch_num_tokens_for_loss,
                labels=padded_labels,
                **kwargs,
            )

        # remove padding results from outputs
        for stage_idx in range(len(schedule_outputs)):
            schedule_outputs[stage_idx] = schedule_outputs[stage_idx][:original_batch_size]

        return schedule_outputs

    def _point_to_low_precision_params(self):
        for model in self.model_parts:
            for name, param in model.named_parameters():
                if hasattr(param, "_ddp_ignored") and param._ddp_ignored:
                    continue
                param.data = param._mp_param

    def _point_to_full_precision_params(self):
        for model in self.model_parts:
            for name, param in model.named_parameters():
                if hasattr(param, "_ddp_ignored") and param._ddp_ignored:
                    continue
                param.data = param._mp_param

    def _copy_full_precision_to_low_precision_params(self):
        from torch.distributed.utils import _alloc_storage

        for model in self.model_parts:
            for name, param in model.named_parameters():
                if hasattr(param, "_ddp_ignored") and param._ddp_ignored:
                    continue
                _alloc_storage(param._mp_param, param.size())
                with torch.no_grad():
                    # assert param.data == param._fp_param, "param.data should point to param._fp_param before copy"
                    assert (
                        param.data.dtype == torch.float32
                    ), "param.data should be float32 before copy"
                    param._mp_param.copy_(param.data)

    def run_pipeline(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        batch_num_tokens_for_loss: Union[int, float],
        num_microbatches: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs,
    ) -> List[List[Optional[LMOutputWithLoss]]]:
        """
        Run the pipeline, returning the losses captured.
        Returns:
            ce_batch_loss: Cross-entropy loss for the batch.
            z_batch_loss: Z loss for the batch (if applicable).
        """
        with self._model_forward_context():
            schedule_outputs = self.train_pp_schedule.step(
                input_ids,
                target=labels,
                loss_div_factor=batch_num_tokens_for_loss,
                labels=labels,
                num_microbatches=num_microbatches,
                progress_callback=progress_callback,
                **kwargs,
            )
        return schedule_outputs

    def optim_step(self):
        from olmo_core.optim.moe_optimizer import OLMoDDPOptimizer

        optim = self._require_optimizer()
        debug_dump_grads = self._debug_should_dump("grads")
        if debug_dump_grads:
            self._debug_dump_model_grads("pre_optim")
        if (
            debug_dump_grads
            or self._debug_dump_optim_grad_norms
            or self._debug_log_optim_grad_norms
        ):
            setattr(optim, "_olmo_debug_global_step", self.trainer.global_step)

        # dist.barrier()
        # if dist.get_rank() == 0:
        #     print("-------optim_step start--------")

        # Maybe adjust learning rate.
        if self.scheduler is not None:
            for group_idx, group in enumerate(optim.param_groups):
                new_lr = self.scheduler.set_lr(group, self.trainer)
                self.trainer.record_metric(f"LR (group {group_idx})", new_lr, namespace="optim")

        # Step optimizer. OLMoDDPOptimizer.step() copies the updated main params back to the model
        # params via _copy_main_params_to_model_params(), which already rebuilds every rowwise-FP8
        # cache from the fresh weights. Refreshing again here would repeat a full quantization pass
        # each step with no intervening weight update, so the refresh below is kept disabled.
        optim.step()
        # with nvtx.annotate("OLMoDDPTrainModule.refresh_rowwise_fp8_cache_after_optim", color="red"):
        #     for model in self.model_parts:
        #         model.refresh_rowwise_fp8_cache()

        # dist.barrier()
        # if dist.get_rank() == 0:
        #     print("-------optim step done--------")

        # self._copy_full_precision_to_low_precision_params()

        total_grad_norm = optim.latest_grad_norm

        if total_grad_norm is not None:
            self.trainer.record_metric(
                "total grad norm", total_grad_norm, reduce_type=None, namespace="optim"
            )

        if isinstance(optim, OLMoDDPOptimizer):
            self.record_metric("step skipped", optim.step_skipped, namespace="optim")

        for model in self.model_parts:
            model.post_optim_step()

    def zero_grads(self):
        self._require_optimizer()
        # Contract: optimizer consumes model grads but does not clear model grad buffers.
        # Keep grad-buffer lifecycle in the model wrapper so bucket views stay stable.
        for m in self.model_parts:
            m.zero_grad(set_to_none=False)

    def model_forward_no_pipeline(
        self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        """
        Run a forward pass on a micro-batch, returning the logits.
        """
        with self._model_forward_context():
            return self.model_parts[0](input_ids, labels=labels, **kwargs)

    @cached_property
    def _debug_dump_dir_cached(self) -> Optional[str]:
        return os.getenv("OLMO_DEBUG_DUMP_DIR")

    def _debug_dump_dir(self) -> Optional[str]:
        return self._debug_dump_dir_cached

    @cached_property
    def _debug_dump_run_id_cached(self) -> str:
        return os.getenv("OLMO_DEBUG_RUN_ID", "run")

    def _debug_dump_run_id(self) -> str:
        return self._debug_dump_run_id_cached

    @cached_property
    def _debug_dump_sample_size_cached(self) -> int:
        raw = os.getenv("OLMO_DEBUG_DUMP_SAMPLE_SIZE", "32")
        try:
            return max(1, int(raw))
        except ValueError:
            return 32

    def _debug_dump_sample_size(self) -> int:
        return self._debug_dump_sample_size_cached

    @cached_property
    def _debug_dump_steps(self) -> Optional[set[int]]:
        raw = os.getenv("OLMO_DEBUG_DUMP_STEPS")
        if not raw:
            return None
        try:
            return {int(part.strip()) for part in raw.split(",") if part.strip()}
        except ValueError:
            return None

    def _debug_env_flag(self, name: str, default: bool = False) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @cached_property
    def _debug_dump_optim_grad_norms(self) -> bool:
        return self._debug_env_flag("OLMO_DEBUG_DUMP_OPTIM_GRAD_NORMS", False)

    @cached_property
    def _debug_log_optim_grad_norms(self) -> bool:
        return os.getenv("OLMO_DDP_DEBUG_GRAD_NORMS") is not None

    @lru_cache(maxsize=None)
    def _debug_dump_kind_enabled(self, kind: str) -> bool:
        raw_kind = os.getenv(f"OLMO_DEBUG_DUMP_{kind.upper()}", "0").strip().lower()
        return raw_kind in {"1", "true", "yes", "on"}

    def _debug_should_dump(self, kind: str) -> bool:
        if self._debug_dump_dir() is None:
            return False
        if not self._debug_dump_kind_enabled(kind):
            return False
        steps = self._debug_dump_steps
        if steps is None:
            return True
        return self.trainer.global_step in steps

    def _debug_rank_in_filter(self, env_name: str) -> bool:
        raw = os.getenv(env_name, "all").strip().lower()
        rank = get_rank() if is_distributed() else 0
        if raw in {"all", "*"}:
            return True
        return str(rank) in {part.strip() for part in raw.split(",")}

    @torch.no_grad()
    def _debug_tensor_payload(self, tensor: torch.Tensor) -> Dict[str, Any]:
        local = get_local_tensor(tensor.detach())
        flat = local.reshape(-1)
        sample_n = self._debug_dump_sample_size()
        if flat.numel() == 0:
            sample = flat.float().cpu()
            sample_indices = torch.empty(0, dtype=torch.long)
        elif flat.numel() <= sample_n * 2:
            sample = flat.float().cpu()
            sample_indices = torch.arange(flat.numel(), dtype=torch.long)
        else:
            # Build indices on CPU in float64. CUDA linspace over very large tensors can
            # round the final index past the end before index_select sees it.
            sample_indices = (
                torch.linspace(
                    0,
                    flat.numel() - 1,
                    steps=sample_n,
                    dtype=torch.float64,
                )
                .round()
                .long()
                .clamp_(0, flat.numel() - 1)
            )
            sample = flat.index_select(0, sample_indices.to(device=flat.device)).float().cpu()
        return {
            "shape": tuple(local.shape),
            "dtype": str(local.dtype),
            "numel": local.numel(),
            "sample": sample,
            "sample_indices": sample_indices,
            "sample_sum": sample.sum(),
            "sample_abs_max": sample.abs().max() if sample.numel() > 0 else torch.tensor(0.0),
        }

    def _debug_dump_path(self, kind: str, suffix: str) -> str:
        dump_root = self._debug_dump_dir()
        assert dump_root is not None
        rank = get_rank() if is_distributed() else 0
        dump_dir = os.path.join(dump_root, self._debug_dump_run_id(), f"rank{rank:03d}")
        os.makedirs(dump_dir, exist_ok=True)
        return os.path.join(dump_dir, f"step{self.trainer.global_step:06d}_{suffix}_{kind}.pt")

    def _debug_dump_step_dir(self, suffix: str) -> str:
        dump_root = self._debug_dump_dir()
        assert dump_root is not None
        rank = get_rank() if is_distributed() else 0
        dump_dir = os.path.join(
            dump_root,
            self._debug_dump_run_id(),
            f"rank{rank:03d}",
            f"step{self.trainer.global_step:06d}_{suffix}",
        )
        os.makedirs(dump_dir, exist_ok=True)
        return dump_dir

    @staticmethod
    def _debug_safe_filename(name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)

    @torch.no_grad()
    def _debug_dump_lm_output(
        self,
        lm_output: Union[torch.Tensor, LMOutputWithLoss],
        *,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        micro_batch_idx: int,
    ) -> None:
        if not isinstance(lm_output, LMOutputWithLoss):
            return
        payload: Dict[str, Any] = {
            "kind": "logits",
            "rank": get_rank() if is_distributed() else 0,
            "step": self.trainer.global_step,
            "micro_batch_idx": micro_batch_idx,
            "input_ids": self._debug_tensor_payload(input_ids),
            "labels": None if labels is None else self._debug_tensor_payload(labels),
            "loss": self._debug_tensor_payload(lm_output.loss),
            "ce_loss": self._debug_tensor_payload(lm_output.ce_loss),
            "z_loss": None
            if lm_output.z_loss is None
            else self._debug_tensor_payload(lm_output.z_loss),
            "logits": None
            if lm_output.logits is None
            else self._debug_tensor_payload(lm_output.logits),
        }
        torch.save(payload, self._debug_dump_path("logits", f"mb{micro_batch_idx:03d}"))

    @torch.no_grad()
    def _debug_dump_model_grads(self, phase: str) -> None:
        if self._debug_env_flag("OLMO_DEBUG_DUMP_FULL_GRADS", False):
            self._debug_dump_full_model_grads(phase)
            if not self._debug_env_flag("OLMO_DEBUG_DUMP_SAMPLE_GRADS_WITH_FULL", False):
                return

        entries: List[Dict[str, Any]] = []
        for part_idx, model in enumerate(self.model_parts):
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                entry: Dict[str, Any] = {
                    "part_idx": part_idx,
                    "name": name,
                    "param_shape": tuple(param.shape),
                    "param_dtype": str(param.dtype),
                }
                if param.grad is not None:
                    entry["grad"] = self._debug_tensor_payload(param.grad)
                reduced_grad_shard = getattr(param, "_olmo_ddp_reduced_grad_shard", None)
                if reduced_grad_shard is not None:
                    entry["reduced_grad_shard"] = self._debug_tensor_payload(reduced_grad_shard)
                else:
                    main_grad = getattr(param, "_main_grad_fp32", None)
                    if main_grad is not None:
                        entry["main_grad_fp32"] = self._debug_tensor_payload(main_grad)
                if any(key in entry for key in ("grad", "main_grad_fp32", "reduced_grad_shard")):
                    entries.append(entry)

        payload = {
            "kind": "model_grads",
            "rank": get_rank() if is_distributed() else 0,
            "step": self.trainer.global_step,
            "phase": phase,
            "entries": entries,
        }
        torch.save(payload, self._debug_dump_path("model_grads", phase))

    @torch.no_grad()
    def _debug_dump_full_model_grads(self, phase: str) -> None:
        if not self._debug_rank_in_filter("OLMO_DEBUG_DUMP_FULL_GRADS_RANKS"):
            return
        rank = get_rank() if is_distributed() else 0
        step_dir = self._debug_dump_step_dir(f"{phase}_full_model_grads")
        entries: List[Dict[str, Any]] = []
        for part_idx, model in enumerate(self.model_parts):
            for param_idx, (name, param) in enumerate(model.named_parameters()):
                if not param.requires_grad:
                    continue
                tensors: List[Tuple[str, torch.Tensor]] = []
                if param.grad is not None:
                    tensors.append(("grad", param.grad))
                reduced_grad_shard = getattr(param, "_olmo_ddp_reduced_grad_shard", None)
                if reduced_grad_shard is not None:
                    tensors.append(("reduced_grad_shard", reduced_grad_shard))
                else:
                    main_grad = getattr(param, "_main_grad_fp32", None)
                    if main_grad is not None:
                        tensors.append(("main_grad_fp32", main_grad))

                for grad_name, tensor in tensors:
                    local = get_local_tensor(tensor.detach())
                    file_name = (
                        f"part{part_idx:03d}_param{param_idx:04d}_"
                        f"{grad_name}_{self._debug_safe_filename(name)}.pt"
                    )
                    torch.save(
                        {
                            "kind": "full_model_grad",
                            "rank": rank,
                            "step": self.trainer.global_step,
                            "phase": phase,
                            "part_idx": part_idx,
                            "param_idx": param_idx,
                            "name": name,
                            "grad_name": grad_name,
                            "param_shape": tuple(param.shape),
                            "param_dtype": str(param.dtype),
                            "grad_shape": tuple(local.shape),
                            "grad_dtype": str(local.dtype),
                            "tensor": local.cpu(),
                        },
                        os.path.join(step_dir, file_name),
                    )
                    entries.append(
                        {
                            "part_idx": part_idx,
                            "param_idx": param_idx,
                            "name": name,
                            "grad_name": grad_name,
                            "file": file_name,
                            "param_shape": tuple(param.shape),
                            "param_dtype": str(param.dtype),
                            "grad_shape": tuple(local.shape),
                            "grad_dtype": str(local.dtype),
                            "numel": local.numel(),
                        }
                    )

        torch.save(
            {
                "kind": "full_model_grads_manifest",
                "rank": rank,
                "step": self.trainer.global_step,
                "phase": phase,
                "entries": entries,
            },
            os.path.join(step_dir, "manifest.pt"),
        )

    @lru_cache  # type: ignore[override]
    def num_flops_per_token(self, seq_len: int) -> int:
        return int(self._full_model_num_flops_per_token(seq_len))

    def global_num_flops_in_batch(self, batch: Dict[str, Any]) -> Optional[int]:
        global_num_tokens = self.trainer.data_loader.global_num_tokens_in_batch(batch)
        if global_num_tokens is None:
            return None
        flops_per_token = self.num_flops_per_token(seq_len=batch["input_ids"].shape[1])
        return flops_per_token * global_num_tokens

    @contextlib.contextmanager
    def _train_microbatch_context(
        self, micro_batch_idx: int, num_micro_batches: int
    ) -> Generator[None, None, None]:
        is_last_mb = micro_batch_idx == num_micro_batches - 1
        with contextlib.ExitStack() as stack:
            if isinstance(self.model_parts[0], FSDPModule):
                raise OLMoConfigurationError("FSDP not supported. Use replicate()")
            elif isinstance(self.model_parts[0], DDP):
                raise OLMoConfigurationError("torch DDP not supported. Use replicate()")
            elif isinstance(self.model_parts[0], MultiGroupDistributedDataParallel):
                # only_allreduce_last_microbatch is validated to be True at construction, so every
                # non-final micro-batch accumulates under no_sync() and the single
                # finalize_grad_reduce() after the loop reduces the fully-accumulated grads once.
                if not is_last_mb:
                    stack.enter_context(self.model_parts[0].no_sync())
            elif self.dp_config.name == DataParallelType.ddp:  # temp fix
                if (
                    # If specified, only synchronize at the last microbatch.
                    not is_last_mb
                    and self.dp_config.only_allreduce_last_microbatch
                ):
                    stack.enter_context(
                        self.ddp_no_sync(self.model_parts)
                    )  # only DDP has no_sync(), can only call set_requires_gradient_sync()
            yield

    @contextlib.contextmanager
    def ddp_no_sync(self, model_parts: List[OLMoDDPModel]):
        for module in model_parts:
            assert callable(getattr(module, "set_requires_gradient_sync", None)), (
                f"{type(module).__name__} must implement set_requires_gradient_sync(flag: bool). "
                "This is automatically managed if the model is returned by torch.distributed._composable.replicate"
            )
            # non-EP modules
            module.set_requires_gradient_sync(False)  # type: ignore

            # EP managed modules
            ep_modules = [m for m in module.modules() if getattr(m, "_ep_sharded", False)]
            for m in ep_modules:
                m.set_requires_gradient_sync(False)  # type: ignore

        try:
            yield
        finally:
            for module in model_parts:
                module.set_requires_gradient_sync(True)  # type: ignore
                ep_modules = [m for m in module.modules() if getattr(m, "_ep_sharded", False)]
                for m in ep_modules:
                    m.set_requires_gradient_sync(True)  # type: ignore

    @contextlib.contextmanager
    def _eval_batch_context(self) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            yield

    @contextlib.contextmanager
    def _model_forward_context(self) -> Generator[None, None, None]:
        # No autocast context here; OLMoDDP relies on the model's own dtype handling.
        yield

    def _clip_grad_norm(
        self, max_grad_norm: float, norm_type: float = 2.0, foreach: Optional[bool] = None
    ) -> torch.Tensor:
        raise RuntimeError("Deprecated. Use optimizer's grad clipping instead.")

    def _prepare_batch(
        self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        if self.pp_enabled:
            input_ids = batch.pop("input_ids")
            labels = labels if labels is not None else batch.pop("labels", None)
            kwargs: Dict[str, Any] = {}
            if "doc_lens" in batch and "max_doc_lens" in batch:
                log_once(log, "intra-document masking enabled")
                kwargs["doc_lens"] = batch["doc_lens"]
                kwargs["max_doc_lens"] = batch["max_doc_lens"]
            return input_ids, labels, kwargs
        else:
            input_ids = batch.pop("input_ids")
            labels = labels if labels is not None else batch.pop("labels", None)
            if "doc_lens" in batch and "max_doc_lens" in batch:
                log_once(log, "intra-document masking enabled")
            return input_ids, labels, batch

    def _prepare_pipeline_context_parallel_batch(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        model_kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        if not self.cp_enabled:
            return input_ids, labels, model_kwargs

        unsupported_doc_keys = {"doc_lens", "max_doc_lens"} & set(model_kwargs)
        if unsupported_doc_keys:
            raise OLMoConfigurationError(
                "context parallelism with pipeline parallelism does not support "
                f"intra-document masking metadata yet: {sorted(unsupported_doc_keys)}"
            )

        if labels is None:
            raise OLMoConfigurationError(
                "context parallelism with pipeline parallelism requires labels to be sharded "
                "alongside input_ids"
            )

        sharding_model = self.model_parts[0]
        input_ids, labels, original_seq_len = sharding_model.prepare_cp_sequence_inputs(
            input_ids,
            labels,
            ignore_index=self.label_ignore_index,
        )

        model_kwargs = dict(model_kwargs)
        model_kwargs["cp_already_sharded"] = True
        model_kwargs["cp_original_seq_len"] = original_seq_len
        return input_ids, labels, model_kwargs

    def parallelize_and_init_model(
        self,
        model: OLMoDDPModel,  # the full model before sharding, the same on all ranks
        *,
        compile_model: bool = False,
        float8_config: Optional[Float8Config] = None,
        dp_config: Optional[TransformerDataParallelConfig] = None,
        tp_config: Optional[TransformerTensorParallelConfig] = None,
        cp_config: Optional[TransformerContextParallelConfig] = None,
        ep_config: Optional[TransformerExpertParallelConfig] = None,
        ac_config: Optional[TransformerActivationCheckpointingConfig] = None,
        pp_config: Optional[TransformerPipelineParallelConfig] = None,
        eval_only: bool = False,
    ) -> List["OLMoDDPModel"]:
        assert isinstance(model, OLMoDDPModel), "model must be an instance of OLMoDDPModel"

        if tp_config is not None:
            raise NotImplementedError("TP not supported yet")

        if pp_config is not None:
            assert (
                self.world_mesh["dense"] is not None
            ), "Dense mesh must be built before applying pipeline parallelism"
            self.pp_mesh = self.world_mesh["dense"]["pp"]
            self.pp_group = self.pp_mesh.get_group()
            self.pp_group_rank = get_rank(self.pp_group)
            self.pp_group_size = get_world_size(self.pp_group)
            self.pp_prev_rank = (self.pp_group_rank - 1) % self.pp_group_size
            self.pp_next_rank = (self.pp_group_rank + 1) % self.pp_group_size
            self.pp_final_stage_rank = pp_config.final_stage_rank()
            log.info(
                "Creating pipeline P2P process groups " "(global_rank=%d, pp_rank=%d/%d)",
                dist.get_rank(),
                self.pp_group_rank,
                self.pp_group_size,
            )
            pp_p2p_group = pp_config.build_p2p_process_group(self.world_mesh["dense"])
            log.info(
                "Created pipeline P2P process groups " "(global_rank=%d, pp_rank=%d/%d)",
                dist.get_rank(),
                self.pp_group_rank,
                self.pp_group_size,
            )
            if is_distributed():
                log.info(
                    "Synchronizing after pipeline P2P process-group creation " "(global_rank=%d)",
                    dist.get_rank(),
                )
                dist.barrier()

            # Split model into pipeline stages.
            model.purge_cuda_events()  # set event to None so that can be deepcopied

            stages_and_model_parts = pp_config.split_model(
                model,
                pp_mesh=self.pp_mesh,
                device=self.device,
                use_ddp=self.world_mesh["dense"]["dp"].size() > 1,
                p2p_group=pp_p2p_group,
            )
            stages = stages_and_model_parts[0]

            model_parts: List[OLMoDDPModel] = cast(List[OLMoDDPModel], stages_and_model_parts[1])

            for model_part in model_parts:
                assert isinstance(model_part, OLMoDDPModel)
                model_part.install_cuda_events()

            self._pp_stages = stages
            log.info(
                f"Applied pipeline parallelism to the model with {get_device_mesh_info(self.pp_mesh)}"
            )

            assert (
                self.world_mesh is not None
            ), "World mesh must be built before applying expert parallelism"
            assert (
                self.world_mesh["dense"] is not None
            ), "Dense mesh must be built before applying expert parallelism"

            for m in model_parts:
                m.apply_pp(self.pp_mesh)

        else:
            model_parts = [model]  # no PP, single part

        # Maybe apply FP8 training.
        if float8_config is not None and float8_config.enabled:
            for m in model_parts:
                m.apply_fp8(float8_config)
                log.info("Swapped linear layers to Float8 linear layers\n%s", m)

        if cp_config is not None:
            assert self.world_mesh["dense"] is not None
            cp_mesh = self.world_mesh["dense"]["cp"]
            for m in model_parts:
                m.apply_cp(cp_mesh, ring=cp_config.ring, uly=cp_config.uly)
            log.info(
                "Applied context parallelism to the model with %s",
                get_device_mesh_info(cp_mesh),
            )

        assert dp_config is not None

        if ep_config is not None:
            # EP-DP combined
            # for the dense part, replicate over DP pg
            # for the moe part, replicate over EP-DP pg
            assert (
                self.world_mesh is not None
            ), "World mesh must be built before applying expert parallelism"
            assert (
                self.world_mesh["moe"] is not None
            ), "MoE mesh must be built before applying expert parallelism"
            assert (
                self.world_mesh["dense"] is not None
            ), "Dense mesh must be built before applying expert parallelism"
            ep_mesh = self.world_mesh["moe"]
            dp_mesh = self.world_mesh["dense"]["dp"]
            dense_param_group = self.dense_dp_cp_group
            expert_param_group = self.expert_param_group
            ep_mp_group_override: Optional[ProcessGroup] = None

            use_high_priority_ep_mp_group = os.getenv(
                "OLMO_EP_MP_HIGH_PRIORITY_GROUP", "1"
            ).strip().lower() not in {"", "0", "false", "no", "off"}

            if backend_supports_cuda() and use_high_priority_ep_mp_group:
                ep_mp_dim = ep_mesh.mesh_dim_names.index(MeshDimName.ep_mp)
                ep_rank_grid = ep_mesh.mesh
                if ep_mp_dim != ep_rank_grid.ndim - 1:
                    ep_rank_grid = torch.movedim(ep_rank_grid, ep_mp_dim, -1).contiguous()
                ep_mp_rank_groups = ep_rank_grid.view(-1, ep_rank_grid.shape[-1]).tolist()
                # Keep small synchronized-EP collectives on a high-priority stream.
                # Shared experts can overlap the token-count exchange; if some EP
                # ranks arrive earlier than others, this is meant to keep that
                # overlap from delaying the collective start. Revalidate whether it
                # still buys useful overlap on current NCCL/torch versions.
                nccl_opts = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
                log.info(
                    "Creating high-priority NCCL EP-MP process groups " "(global_rank=%d)",
                    dist.get_rank(),
                )
                ep_mp_group, ep_mp_groups = dist.new_subgroups_by_enumeration(
                    ranks_per_subgroup_list=ep_mp_rank_groups,
                    backend="nccl",
                    pg_options=nccl_opts,
                    group_desc="ep_mp_high_priority",
                )
                if ep_mp_group == dist.GroupMember.NON_GROUP_MEMBER:
                    raise RuntimeError("Current rank is not in any EP-MP high-priority subgroup")
                ep_mp_group_override = cast(ProcessGroup, ep_mp_group)
                self.ep_mp_high_priority_group = ep_mp_group_override
                self.ep_mp_high_priority_groups = ep_mp_groups
                log.info("Created high-priority NCCL EP-MP process groups")
                if is_distributed():
                    log.info(
                        "Synchronizing after high-priority EP-MP process-group creation "
                        "(global_rank=%d)",
                        dist.get_rank(),
                    )
                    dist.barrier()
            elif backend_supports_cuda():
                log.info(
                    "Skipping high-priority NCCL EP-MP process-group creation because "
                    "OLMO_EP_MP_HIGH_PRIORITY_GROUP=0"
                )
            else:
                log.warning(
                    "Skipping EP-MP high-priority group creation because backend is '%s'",
                    dist.get_backend(),
                )

            share_ep_no_sync_scratch_pool = len(model_parts) > 1 and all(
                (
                    cast(OLMoDDPModel, m).recompute_each_block
                    or cast(OLMoDDPModel, m).recompute_all_blocks_by_chunk
                )
                for m in model_parts
            )
            ep_no_sync_shared_pool: Optional[Any] = None
            if share_ep_no_sync_scratch_pool:
                log.info(
                    "Sharing EP no-sync transient symmetric scratch pool across %d local PP model parts",
                    len(model_parts),
                )

            ddp_model_parts: List[OLMoDDPModel] = []
            for m in model_parts:
                if not m.is_moe:
                    raise OLMoConfigurationError("Expert parallelism is only valid for MoE models")
                returned_shared_pool = cast(OLMoDDPModel, m).apply_ep(
                    dp_mesh=dp_mesh,
                    ep_mesh=ep_mesh,
                    ep_mp_group=ep_mp_group_override,
                    param_dtype=None,
                    compile_enabled=compile_model,
                    ep_no_sync_shared_pool=(
                        ep_no_sync_shared_pool if share_ep_no_sync_scratch_pool else None
                    ),
                )
                if share_ep_no_sync_scratch_pool and returned_shared_pool is not None:
                    if ep_no_sync_shared_pool is None:
                        ep_no_sync_shared_pool = returned_shared_pool
                    elif returned_shared_pool is not ep_no_sync_shared_pool:
                        raise RuntimeError(
                            "Expected all local PP model parts to share the same EP no-sync "
                            "symmetric scratch pool"
                        )

            log.info(
                "Applied expert parallelism to the model with %s",
                get_device_mesh_info(ep_mesh),
            )
        else:
            # Pure DP (no EP)
            pass
            assert (
                self.world_mesh is not None
            ), "World mesh must be built before applying expert parallelism"
            assert (
                self.world_mesh["dense"] is not None
            ), "Dense mesh must be built before applying expert parallelism"
            # param_dtype = dp_config.param_dtype.as_pt() if dp_config.param_dtype is not None else None
            dp_mesh = self.world_mesh["dense"]["dp"]
            ep_mesh = None
            dense_param_group = self.dense_dp_cp_group
            expert_param_group = None
            # for m in model_parts:
            #     cast(OLMoDDPModel, m).apply_ddp(dp_mesh=dp_mesh, compile_enabled=compile_model, param_dtype=param_dtype)
            # log.info(f"Applied DDP to the model with {get_device_mesh_info(dp_mesh)}")

        # Maybe apply activation checkpointing.
        if ac_config is not None:
            for m in model_parts:
                m.apply_activation_checkpointing(
                    ac_config.mode,
                    block_interval=ac_config.block_interval,
                    modules=ac_config.modules,
                    activation_memory_budget=ac_config.activation_memory_budget,
                )
            log.info(f"Applied '{ac_config.mode}' activation checkpointing to the model")

        # ac_budget = 0.05
        # torch._functorch.config.activation_memory_budget = ac_budget

        self.init_model_weights(
            model_parts=model_parts,
            max_sequence_length=self.max_sequence_length,
            rank_microbatch_size=self.rank_microbatch_size,
        )

        # now wrap with DDP (requires initialized params)
        if eval_only:
            # apply_dp() currently performs the bf16 cast used for MoE forward kernels.
            # Keep that cast in eval-only mode even though we intentionally skip DDP wrapping.
            self._cast_to_fwd_bwd_precision(model_parts)
            for m in model_parts:
                m.refresh_rowwise_fp8_cache()
            log.info("Skipping DDP wrapping because eval_only=True")
            return model_parts

        ddp_model_parts = []
        for idx, m in enumerate(model_parts):
            log.info(
                "Wrapping OLMo DDP model part %d/%d with MultiGroupDistributedDataParallel",
                idx + 1,
                len(model_parts),
            )
            ddp_m = m.apply_dp(
                dp_mesh=dp_mesh,
                ep_mesh=ep_mesh,
                dense_process_group=dense_param_group,
                expert_process_group=expert_param_group,
                accumulate_grads_in_fp32=dp_config.accumulate_grads_in_fp32,
                reduce_grads_in_fp32=dp_config.reduce_grads_in_fp32,
                bucket_cap_mb=dp_config.bucket_cap_mb,
                use_reduce_scatter=dp_config.use_reduce_scatter,
            )
            log.info(
                "Wrapped OLMo DDP model part %d/%d with MultiGroupDistributedDataParallel",
                idx + 1,
                len(model_parts),
            )
            ddp_model_parts.append(ddp_m)

            if pp_config is not None:
                # update stage reference to point to the ddp wrapped model part
                assert self._pp_stages is not None
                assert self._pp_stages[idx].submod is m
                self._pp_stages[idx].submod = ddp_m

        with nvtx.annotate(
            "OLMoDDPTrainModule.refresh_rowwise_fp8_cache_before_first_step", color="red"
        ):
            for m in ddp_model_parts:
                m.refresh_rowwise_fp8_cache()

        return ddp_model_parts

    def memory_usage_estimation(self):
        pass

    def init_model_weights(
        self,
        model_parts,
        max_sequence_length: int,
        rank_microbatch_size: int,
    ):
        from olmo_core.nn.ddp import OLMoDDPModel

        self._set_ep_no_sync_symm_initialization_complete(False)

        # Materialize and init parameters.
        log.info("Initializing model weights...")
        for model_part_idx, m in enumerate(model_parts):
            m = cast(OLMoDDPModel, m)
            m.init_weights(
                max_seq_len=max_sequence_length,
                max_local_microbatch_size=rank_microbatch_size,
                device=self.device,
                world_mesh=self.world_mesh,  # only PP mesh is used, should be fine
                model_part_idx=model_part_idx,
            )
            # for n, p in m.named_parameters():
            #     print(f'{n} {p.shape}: mean={p.data.mean().item()}, std={p.data.std().item()}')

        self._prewarm_deepep_v2_runtimes(
            model_parts=model_parts,
            rank_microbatch_size=rank_microbatch_size,
        )
        self._prewarm_ep_no_sync_symm_buffers(
            model_parts=model_parts,
            rank_microbatch_size=rank_microbatch_size,
        )
        for m in model_parts:
            m.refresh_rowwise_fp8_cache()

        # PP needs one additional, schedule-aware prewarm in on_attach(). PP1 is
        # fully initialized here, so any later symmetric allocation is a bug.
        if not self.pp_enabled:
            self._set_ep_no_sync_symm_initialization_complete(True)

        return

    @staticmethod
    def _set_ep_no_sync_symm_initialization_complete(value: bool) -> None:
        from olmo_core.train.globals import set_global_arg

        set_global_arg("ep_no_sync_symm_initialization_complete", value)

    def _cp_local_rank_microbatch_size(self, rank_microbatch_size: int) -> int:
        if self._cp_config is None:
            return rank_microbatch_size

        if rank_microbatch_size % self.max_sequence_length != 0:
            raise RuntimeError(
                f"'rank_microbatch_size' ({rank_microbatch_size}) must be divisible by "
                f"'max_sequence_length' ({self.max_sequence_length})"
            )

        cp_degree = self._cp_config.degree
        sequence_batch_size = rank_microbatch_size // self.max_sequence_length
        length_multiple = cp_degree
        ring = getattr(self._cp_config, "ring", None)
        if ring is not None:
            load_balancer = getattr(ring, "load_balancer", None)
            load_balancer_name = getattr(load_balancer, "value", load_balancer)
            if load_balancer_name == "zig_zag":
                length_multiple = 2 * cp_degree

        padded_sequence_length = (
            math.ceil(self.max_sequence_length / length_multiple) * length_multiple
        )
        return sequence_batch_size * (padded_sequence_length // cp_degree)

    def _prewarm_deepep_v2_runtimes(
        self,
        *,
        model_parts,
        rank_microbatch_size: int,
    ) -> None:
        if self.world_mesh.get("moe") is None:
            return

        prewarm_rank_microbatch_size = self._cp_local_rank_microbatch_size(rank_microbatch_size)
        for model_part in model_parts:
            cast(OLMoDDPModel, model_part).prewarm_deepep_v2_runtimes(
                max_local_microbatch_size=prewarm_rank_microbatch_size,
            )

    @staticmethod
    def _ep_no_sync_symm_summary_enabled() -> bool:
        # Opt-in: this walks every symmetric tensor and queries CUDA allocator stats, so keep it off
        # for normal runs and enable it explicitly in IBGDA/debug recipes.
        raw = os.getenv("OLMO_EP_NO_SYNC_SYMM_BUFFER_SUMMARY", "0")
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _ep_no_sync_symm_prewarm_summary_enabled() -> bool:
        raw = os.getenv("OLMO_EP_NO_SYNC_SYMM_BUFFER_SUMMARY_PREWARM", "0")
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _should_log_ep_no_sync_symm_summary_rank() -> bool:
        if not dist.is_available() or not dist.is_initialized():
            return True
        ranks = (
            os.getenv("OLMO_EP_NO_SYNC_SYMM_BUFFER_SUMMARY_RANKS")
            or os.getenv("OLMO_ROWWISE_DEBUG_RANKS")
            or os.getenv("OLMO_TBO_DEBUG_RANKS")
        )
        if ranks is None:
            return dist.get_rank() == 0
        normalized = ranks.strip().lower()
        if normalized in {"all", "*"}:
            return True
        if normalized in {"", "none", "off", "0-only"}:
            return dist.get_rank() == 0
        selected_ranks = {part.strip() for part in ranks.split(",") if part.strip()}
        return str(dist.get_rank()) in selected_ranks

    @staticmethod
    def _gib(nbytes: int) -> float:
        return float(nbytes) / float(1024**3)

    def _log_ep_no_sync_symm_buffer_summary(
        self,
        *,
        model_parts,
        context: str,
    ) -> None:
        if not self._ep_no_sync_symm_summary_enabled():
            return

        typed_parts = [cast(OLMoDDPModel, m) for m in model_parts]
        unique_infos: "OrderedDict[Tuple[str, int], Tuple[int, Any]]" = OrderedDict()
        duplicate_refs = 0
        for model_part_idx, model_part in enumerate(typed_parts):
            for info in model_part.iter_ep_no_sync_symm_tensor_infos():
                if info.nbytes <= 0:
                    continue
                storage_key = (str(info.device), int(info.data_ptr))
                if storage_key in unique_infos:
                    duplicate_refs += 1
                    continue
                unique_infos[storage_key] = (model_part_idx, info)

        if not self._should_log_ep_no_sync_symm_summary_rank():
            return

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        local_rank = os.getenv("LOCAL_RANK", "?")
        total_bytes = sum(info.nbytes for _, info in unique_infos.values())
        lines = [
            (
                f"EP no-sync symmetric buffer summary ({context}) "
                f"rank={rank} local_rank={local_rank}: "
                f"unique_tensors={len(unique_infos)}, duplicate_refs={duplicate_refs}, "
                f"requested={self._gib(total_bytes):.4f} GiB"
            )
        ]

        if torch.cuda.is_available():
            device = torch.device(self.device)
            if device.type != "cuda":
                device = torch.device("cuda")
            cuda_stats = torch.cuda.memory_stats(device)

            def stat(name: str) -> int:
                return int(cuda_stats.get(name, 0))

            free_bytes, total_device_bytes = torch.cuda.mem_get_info(device)
            reserved_bytes = torch.cuda.memory_reserved(device)
            allocated_bytes = torch.cuda.memory_allocated(device)
            used_bytes = total_device_bytes - free_bytes
            non_torch_bytes = max(0, used_bytes - reserved_bytes)
            lines.append(
                "  CUDA now: "
                f"allocated={self._gib(allocated_bytes):.4f} GiB, "
                f"reserved={self._gib(reserved_bytes):.4f} GiB, "
                f"device_used={self._gib(used_bytes):.4f} GiB, "
                f"non_torch~={self._gib(non_torch_bytes):.4f} GiB"
            )
            lines.append(
                "  PyTorch stats: "
                f"allocated_peak={self._gib(stat('allocated_bytes.all.peak')):.4f} GiB, "
                f"reserved_peak={self._gib(stat('reserved_bytes.all.peak')):.4f} GiB, "
                f"active_current={self._gib(stat('active_bytes.all.current')):.4f} GiB, "
                f"active_peak={self._gib(stat('active_bytes.all.peak')):.4f} GiB"
            )
            lines.append(
                "  PyTorch fragmentation/cache: "
                f"inactive_split_current={self._gib(stat('inactive_split_bytes.all.current')):.4f} GiB, "
                f"inactive_split_peak={self._gib(stat('inactive_split_bytes.all.peak')):.4f} GiB, "
                f"segment_current={self._gib(stat('segment_bytes.all.current')):.4f} GiB, "
                f"segment_peak={self._gib(stat('segment_bytes.all.peak')):.4f} GiB, "
                f"alloc_retries={stat('num_alloc_retries')}, "
                f"ooms={stat('num_ooms')}"
            )

        if unique_infos:
            owner_totals: Dict[str, int] = {}
            owner_counts: Dict[str, int] = {}
            detail_rows = []
            for model_part_idx, info in unique_infos.values():
                owner = (
                    info.owner
                    if info.owner == "shared_pool"
                    else f"part{model_part_idx}:{info.owner}"
                )
                owner_totals[owner] = owner_totals.get(owner, 0) + info.nbytes
                owner_counts[owner] = owner_counts.get(owner, 0) + 1
                detail_rows.append((info.nbytes, model_part_idx, info))

            lines.append("  by owner:")
            for owner, nbytes in sorted(
                owner_totals.items(), key=lambda item: item[1], reverse=True
            ):
                lines.append(
                    f"    {owner}: {self._gib(nbytes):.4f} GiB " f"({owner_counts[owner]} tensors)"
                )

            try:
                detail_limit = int(os.getenv("OLMO_EP_NO_SYNC_SYMM_BUFFER_SUMMARY_LIMIT", "64"))
            except ValueError:
                detail_limit = 64
            if detail_limit != 0:
                detail_rows.sort(key=lambda row: row[0], reverse=True)
                shown_rows = detail_rows if detail_limit < 0 else detail_rows[:detail_limit]
                lines.append(f"  largest buffers: showing {len(shown_rows)}/{len(detail_rows)}")
                for nbytes, model_part_idx, info in shown_rows:
                    dtype = str(info.dtype).replace("torch.", "")
                    lines.append(
                        f"    {self._gib(nbytes):.4f} GiB "
                        f"part={model_part_idx} {info.owner}.{info.name} "
                        f"shape={info.shape} dtype={dtype} device={info.device}"
                    )

        log.info("\n".join(lines))

    def log_ep_no_sync_symm_buffer_summary(self, *, context: str) -> None:
        self._log_ep_no_sync_symm_buffer_summary(
            model_parts=self.model_parts,
            context=context,
        )

    def _prewarm_ep_no_sync_symm_buffers(
        self,
        *,
        model_parts,
        rank_microbatch_size: int,
        rowwise_lifetime_lease_slots: Optional[Union[int, Sequence[int]]] = None,
    ) -> None:
        if self.world_mesh.get("moe") is None:
            return
        from olmo_core.kernels import olmo_symm_mem

        typed_parts = [cast(OLMoDDPModel, m) for m in model_parts]
        local_counts = [m.count_ep_no_sync_blocks() for m in typed_parts]
        if not any(local_counts):
            return
        prewarm_rank_microbatch_size = self._cp_local_rank_microbatch_size(rank_microbatch_size)

        use_olmo_symm = olmo_symm_mem.is_enabled()
        if not use_olmo_symm and self.pp_enabled:
            raise OLMoConfigurationError(
                "EP no-sync with pipeline parallelism cannot use the legacy PyTorch symmetric-memory "
                "backend. PyTorch's NVSHMEM symm_mem.empty() allocates through c10d group '0' before "
                "the EP group is applied, which can hang when PP stages allocate unevenly. Use the "
                "OLMo-owned symmetric-memory backend by setting OLMO_USE_OWN_SYMM_MEM=1, or disable "
                "pipeline parallelism / EP no-sync."
            )

        if use_olmo_symm:
            prewarm_olmo = os.getenv("OLMO_OWN_SYMM_PREWARM", "1").lower() not in (
                "",
                "0",
                "false",
                "no",
                "off",
            )
            if not prewarm_olmo:
                return

        # Prewarm only the local model part's real blocks. The legacy PyTorch
        # backend is only allowed here without PP, so there is no cross-stage
        # allocation sequence to align with dummy padding.
        max_counts = list(local_counts)

        if isinstance(rowwise_lifetime_lease_slots, Sequence):
            if len(rowwise_lifetime_lease_slots) != len(typed_parts):
                raise RuntimeError(
                    "rowwise_lifetime_lease_slots sequence length must match model_parts: "
                    f"{len(rowwise_lifetime_lease_slots)} vs {len(typed_parts)}"
                )
            rowwise_slots_by_part: List[Optional[int]] = [
                int(slots) for slots in rowwise_lifetime_lease_slots
            ]
        else:
            rowwise_slots_by_part = [rowwise_lifetime_lease_slots for _ in typed_parts]

        for model_part_idx, (model_part, max_count) in enumerate(zip(typed_parts, max_counts)):
            model_part.prewarm_ep_no_sync_symm_buffers(
                max_local_microbatch_size=prewarm_rank_microbatch_size,
                pad_to_block_count=max_count,
                rowwise_lifetime_lease_slots=rowwise_slots_by_part[model_part_idx],
            )

        if self._ep_no_sync_symm_prewarm_summary_enabled():
            self._log_ep_no_sync_symm_buffer_summary(
                model_parts=typed_parts,
                context=(
                    "after prewarm, "
                    f"rank_microbatch_size={prewarm_rank_microbatch_size}, "
                    f"rowwise_lifetime_lease_slots={rowwise_lifetime_lease_slots}"
                ),
            )

    def compile_model(self):
        if torch.cuda.is_available():
            for m in self.model_parts:
                m.apply_compile()
            log.info("Applied torch.compile() to the model")
        else:
            log.warning("Skipping model compilation since CUDA is not available")

    def reduce_send_recv(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        # assert self.pp_enabled and self._pp_config is not None

        # if self.pp_group_rank == self.pp_final_stage_rank:
        #     assert x is not None
        #     # DP reduce (mean)
        #     x = x / self._reduce_divide_factor
        #     dist.all_reduce(x, group=self.dp_process_group)
        #     x = x / self.dp_world_size * self._reduce_divide_factor
        # else:
        #     assert x is None
        #     x = torch.empty([], device=self.device, dtype=torch.float32)

        # # Broadcast from final stage to all PP ranks.
        # handle = dist.broadcast(x, src=get_global_rank(self.pp_final_stage_rank, group=self.pp_group), group=self.pp_group, async_op=True)
        # handle.wait()
        # # print(f'{get_rank()} (pp group rank {self.pp_group_rank}) got {x.item()}')
        # return x

        assert (
            self.pp_enabled and self._pp_config is not None
        ), "reduce_send_recv is only valid when PP is enabled"
        if self.pp_group_rank == self.pp_final_stage_rank:
            assert x is not None
            # Reduce across the parameter replica group for dense weights. With CP this includes
            # both DP and CP ranks so scalar losses reflect the full sequence.
            reduce_group = self.dense_dp_cp_group
            dist.all_reduce(x, group=reduce_group)
            x.div_(get_world_size(reduce_group))
        else:
            assert x is None
            x = move_to_device(torch.empty([]), self.device)

        # Propagate final-stage scalar metrics through PP ranks in completion
        # order instead of broadcasting. This was inherited from older PP code;
        # it likely avoids an ordering cycle with surrounding PP collectives, but
        # the exact constraint should be revalidated before simplifying it.
        ordered_ranks = list(self._pp_config.rank_completion_order())
        local_index = ordered_ranks.index(self.pp_group_rank)
        src_rank = None if local_index == 0 else ordered_ranks[local_index - 1]
        dst_rank = (
            None if local_index == (len(ordered_ranks) - 1) else ordered_ranks[local_index + 1]
        )

        send_ops: List[dist.P2POp] = []
        recv_ops: List[dist.P2POp] = []
        if src_rank is not None:
            # print(
            #     f"Rank {get_rank()} (pp group rank {self.pp_group_rank}) receiving from rank "
            #     f"{get_global_rank(src_rank, group=self.pp_group)} (pp group rank {src_rank})"
            # )
            recv_ops.append(dist.P2POp(dist.irecv, x, group=self.pp_group, group_peer=src_rank))
        if dst_rank is not None:
            # print(
            #     f"Rank {get_rank()} (pp group rank {self.pp_group_rank}) sending to rank "
            #     f"{get_global_rank(dst_rank, group=self.pp_group)} (pp group rank {dst_rank})"
            # )
            send_ops.append(dist.P2POp(dist.isend, x, group=self.pp_group, group_peer=dst_rank))

        if len(recv_ops) > 0:
            recv_reqs = dist.batch_isend_irecv(recv_ops)
            for req in recv_reqs:
                req.wait()

        if len(send_ops) > 0:
            send_reqs = dist.batch_isend_irecv(send_ops)
            for req in send_reqs:
                req.wait()

        # print(f'{get_rank()} (pp group rank {self.pp_group_rank}) got {x.item()}')
        return x


__all__ = ["OLMoDDPTrainModule"]
