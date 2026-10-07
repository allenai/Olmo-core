import logging
import math
from dataclasses import dataclass
from typing import Tuple

import torch
from packaging.version import parse as parse_version
from torch.distributed.device_mesh import DeviceMesh

from olmo_core.distributed.parallel import (
    MeshDimName,
    get_dp_model_mesh,
    get_dp_replicate_mesh,
    get_dp_shard_mesh,
    get_tp_mesh,
    get_world_mesh,
)
from olmo_core.nn.transformer import Transformer

from .config import MatrixAwareOptimConfig, OptimConfig, OptimGroupOverride

log = logging.getLogger(__name__)


def _import_dion():
    """Import and return Dion and Dion3 from dion, raising a helpful error if not installed."""
    try:
        import dion
        
        # Extract classes dynamically to bypass namespace shadowing
        Dion = getattr(dion, "Dion")
        
        # The package aliases NorDion2 as Dion3; safely grab whichever is exposed
        Dion3 = getattr(dion, "Dion3", getattr(dion, "NorDion2", None))
        
        if Dion3 is None:
            raise ImportError("dion package found, but NorDion2/Dion3 is missing.")
            
        return Dion, Dion3
    except Exception as e:
        raise ImportError(
            "The 'dion' package is required for the Dion optimizer. "
            "Install it with: pip install git+https://github.com/microsoft/dion.git"
        ) from e


@OptimConfig.register("dion")
@dataclass
class DionConfig(MatrixAwareOptimConfig):
    """
    Configuration class for building a :class:`Dion` optimizer.

    Dion is a Muon-like optimizer that is designed to be scalable for DP-replicated, DP-sharded,
    and TP-sharded models. See https://arxiv.org/abs/2504.05295 for more details.

    Dion supports FSDP, HSDP, and TP parallelism strategies. Flattened mesh dimensions (eg. "dp_ep"
    and "dp_cp") can be supported but are currently not implemented.
    """

    lr: float = 0.01
    """
    Base learning rate. For Dion, this will be scaled based on the matrix dimensions. For AdamW,
    this is the actual learning rate and no additional scaling is done.
    """

    mu: float = 0.95
    """Momentum for Dion"""

    betas: Tuple[float, float] = (0.9, 0.95)
    """Betas for AdamW"""

    weight_decay: float = 0.1
    """Weight decay for non-embedding parameters"""

    rank_fraction: float = 1.0
    """Rank fraction for Dion. Set to 1.0 for full-rank optimization."""

    rank_multiple_of: int = 1
    """
    Round up the low-rank dimension to a multiple of this number.
    This may be useful to ensure even sharding.
    """

    @classmethod
    def optimizer(cls) -> type:
        Dion, _ = _import_dion()
        return Dion

    def default_group_overrides(self, model: torch.nn.Module) -> list[OptimGroupOverride]:
        """
        Apply Dion's parameter grouping rules.
        """
        assert isinstance(model, Transformer)
        params = self.categorize_parameters(model)

        lm_head_out: torch.nn.Linear = model.lm_head.w_out
        model_dim = lm_head_out.weight.shape[1]

        # Matrix parameters are optimized with Dion.
        matrix_override = OptimGroupOverride(params=params["matrix"], opts=dict(algorithm="dion"))

        # Vector, embedding, and lm_head parameters are optimized with AdamW.
        embed_override = OptimGroupOverride(
            params=params["embed"], opts=dict(algorithm="adamw", weight_decay=0.0)
        )
        vector_override = OptimGroupOverride(params=params["vector"], opts=dict(algorithm="adamw"))
        lm_head_override = OptimGroupOverride(
            params=params["lm_head"],
            # lr scaled by sqrt(model_dim) for lm_head is only for lion backup; see Dion paper App. D.2 and Fig. 14 for the Lion-specific 1 / sqrt(d_in) scale.
            opts=dict(algorithm="adamw", lr=self.lr ),
        )

        return [matrix_override, vector_override, embed_override, lm_head_override]

    def build_parallelism_config(self) -> dict[str, DeviceMesh | None]:
        """
        Prepare device meshes for Dion optimizer based on the parallelism configuration.

        Supports:
        - Single-device: All meshes are None
        - FSDP: outer_shard_mesh = DP mesh, replicate_mesh = None
        - HSDP: replicate_mesh = DP replicate mesh, outer_shard_mesh = DP shard mesh
        - TP: inner_shard_mesh = TP mesh (can be combined with FSDP or HSDP)

        :returns: Dictionary with 'replicate_mesh', 'outer_shard_mesh', and 'inner_shard_mesh' keys.
        """
        world_mesh = get_world_mesh()
        meshes: dict[str, DeviceMesh | None] = {
            "replicate_mesh": None,  # mesh for replicated data parallelism.
            "outer_shard_mesh": None,  # parameter sharding mesh, replicated during orthogonalization
            "inner_shard_mesh": None,  # parameter sharding mesh, remains sharded during orthogonalization
        }
        if world_mesh is None:
            return meshes

        dim_names = world_mesh.mesh_dim_names
        if dim_names is None:
            raise RuntimeError("world mesh has no dimension names")

        # Check for HSDP (has both dp_replicate and dp_shard)
        has_dp_replicate = MeshDimName.dp_replicate in dim_names
        has_dp_shard = MeshDimName.dp_shard in dim_names

        if has_dp_replicate and has_dp_shard:
            # HSDP configuration.
            # TODO(dion torch-2.13): remove once dion is torch-2.13-clean upstream. Dion + HSDP is
            # broken on torch >= 2.13: the replica-only path feeds a CPU tensor into a Triton kernel,
            # and the sharded path can hang on an ALLREDUCE until the NCCL watchdog aborts. Fail fast
            # with a clear message instead of erroring or hanging deep in the optimizer step.
            if parse_version(torch.__version__) >= parse_version("2.13"):
                raise RuntimeError(
                    "Dion with HSDP is not supported on torch >= 2.13 (upstream dion bug: a "
                    "CPU-tensor Triton error in the replica-only path and a flaky ALLREDUCE hang in "
                    "the sharded path). Use a torch < 2.13 image for Dion+HSDP runs, or switch the "
                    "data-parallel strategy (FSDP or DDP)."
                )
            meshes["replicate_mesh"] = get_dp_replicate_mesh(world_mesh)
            meshes["outer_shard_mesh"] = get_dp_shard_mesh(world_mesh)
        elif MeshDimName.dp in dim_names or any(d.startswith("dp") for d in dim_names):
            # FSDP configuration
            log.warning("Cannot determine if model is FSDP or DDP, assuming FSDP.")
            meshes["outer_shard_mesh"] = get_dp_model_mesh(world_mesh)
        if MeshDimName.tp in dim_names:
            # TP configuration
            meshes["inner_shard_mesh"] = get_tp_mesh(world_mesh)

        log.info(f"Dion parallelism_config: {meshes}")
        return meshes

    def create_optimizer(self, model: torch.nn.Module, strict: bool = True, **kwargs):
        # When using Dion, we need to set the recompile limit to 16 to avoid triggering an error
        # due to too many recompile requests. Typically, on the second recompilation, torch attempts
        # to compile a dynamic version of the op, unless dynamic=False is marked. Too many different
        # shapes passed to a compiled op with dynamic=False will trigger this error. Since we have
        # grad matrices with many different shapes, we need to set the recompile limit higher than
        # the default of 8.
        # https://docs.pytorch.org/docs/stable/compile/programming_model.recompilation.html
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 16)

        parallelism_config = self.build_parallelism_config()
        optim = self.optimizer()(
            self.build_groups(model, strict=strict),
            replicate_mesh_grad_sync=False,  # HSDP / FSDP / DDP will handle gradient sync internally
            **parallelism_config,
            **kwargs,
        )
        return optim

@OptimConfig.register("dion3")
@OptimConfig.register("nordion2")
@dataclass
class Dion3Config(DionConfig):
    """
    Configuration class for building a :class:`Dion3` optimizer (NorDion2).

    Dion3 applies fractional orthogonal updates and Gram Newton-Schulz with CuteDSL kernels.
    Supports: DDP, FSDP, HSDP.
    Does NOT support: TP, PP, EP, or CP.

    Paper: https://arxiv.org/abs/2608.11612
    """
    lr: float = 0.01 
    fraction: float = 0.25
    mu: float = 0.95
    muon_beta2: float = 0.95
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    epsilon: float = 1e-8
    adjust_lr: str = "rms_norm"
    flatten: bool = False

    @classmethod
    def optimizer(cls) -> type:
        _, Dion3 = _import_dion()
        return Dion3

    def default_group_overrides(self, model: torch.nn.Module) -> list[OptimGroupOverride]:
        assert isinstance(model, Transformer)
        params = self.categorize_parameters(model)

        # NorDion2 requires group["algorithm"] == "nordion2"
        matrix_override = OptimGroupOverride(params=params["matrix"], opts=dict(algorithm="nordion2"))
        embed_override = OptimGroupOverride(
            params=params["embed"], opts=dict(algorithm="adamw", weight_decay=0.0)
        )
        vector_override = OptimGroupOverride(params=params["vector"], opts=dict(algorithm="adamw"))
        lm_head_override = OptimGroupOverride(
            params=params["lm_head"],
            opts=dict(algorithm="adamw", lr=self.lr),
        )
        return [matrix_override, vector_override, embed_override, lm_head_override]    


    def create_optimizer(self, model: torch.nn.Module, strict: bool = True, **kwargs):
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 16)

        # Dion3 doesn't use Dion1's rank-based low-rank approximation params.
        kwargs.pop("rank_fraction", None)
        kwargs.pop("rank_multiple_of", None)

        meshes = self.build_parallelism_config()
        inner_shard_mesh = meshes.get("inner_shard_mesh")
        outer_shard_mesh = meshes.get("outer_shard_mesh")
        replicate_mesh = meshes.get("replicate_mesh")

        if inner_shard_mesh is not None and inner_shard_mesh.size() > 1:
            raise ValueError("Tensor parallel is not supported by Dion3.")

        if outer_shard_mesh is not None and outer_shard_mesh.size() > 1:
            dist_mesh = outer_shard_mesh
        elif replicate_mesh is not None and replicate_mesh.size() > 1:
            dist_mesh = replicate_mesh
        else:
            dist_mesh = None

        optim = self.optimizer()(
            self.build_groups(model, strict=strict),
            distributed_mesh=dist_mesh,
            **kwargs,
        )

        # Dion3 keeps `lr` as a live GPU tensor rather than a plain float, and unlike
        # Dion's `initial_lr`, it has no field that's protected from being overwritten
        # on checkpoint load. Without an anchor, resuming from a checkpoint silently
        # restores whatever `lr` had drifted to at save time instead of the true
        # config value, which breaks the LR schedule (see PR description for details
        # and before/after recovery plots).
        #
        # `pristine_lr` snapshots the true config-scaled LR once at construction time
        # and is excluded from `load_state_dict` so it can't be clobbered by whatever
        # was saved in the checkpoint.
        initial_lrs = []
        for group in optim.param_groups:
            raw_lr = group["lr"]
            initial_lrs.append(raw_lr.item() if isinstance(raw_lr, torch.Tensor) else float(raw_lr))

        for group, init_lr in zip(optim.param_groups, initial_lrs):
            group["pristine_lr"] = init_lr

        original_load_state_dict = optim.load_state_dict

        def patched_load_state_dict(*args, **kwargs):
            # Remove pristine_lr before loading so the distributed checkpoint loader
            # doesn't try to find a matching key on disk, then restore it afterward
            # so it survives the load untouched.
            for group in optim.param_groups:
                group.pop("pristine_lr", None)
            result = original_load_state_dict(*args, **kwargs)
            for group, init_lr in zip(optim.param_groups, initial_lrs):
                group["pristine_lr"] = init_lr
            return result

        optim.load_state_dict = patched_load_state_dict

        return optim