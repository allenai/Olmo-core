"""Optional, stateless bridge to Dion's batched Muon execution.

Optimizer adapters own persistent state and skip/checkpoint semantics. Dion owns
momentum, BF16 transport, Polar Express, scaling, and batched update execution.
"""
from collections import defaultdict
from functools import lru_cache
from typing import Optional

import torch
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor, Replicate, Shard

from .muon_math import MuonLayout


@lru_cache(maxsize=1)
def require_dion():
    """Import the pinned Dion task API only when explicitly selected."""
    try:
        from dion.muon import muon_update_megabatch_async
        from dion.opt_utils import AsyncRuntime, AsyncTask
        from dion.polar_express import polar_express
    except ImportError as exc:
        raise ImportError(
            "muon_backend='dion' requires OLMo-core's pinned Dion dependency; "
            "install the 'dion' extra using the repository's uv source pin."
        ) from exc
    # Dion specializes compiled kernels for matrix shapes, strides, and batch
    # sizes. Match the MoE trainer's allowance rather than failing at PyTorch's
    # default eight variants (the legacy Muon config also raises this limit).
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
    return muon_update_megabatch_async, AsyncRuntime, AsyncTask, polar_express


def logical_matrix_views(
    tensor: torch.Tensor, name: str, shape: tuple[int, ...], layout: Optional[MuonLayout]
) -> list[torch.Tensor]:
    """Return writable 2D views, splitting each expert's packed projections."""
    if layout is not None:
        tensor = tensor.view(layout.shape)
        if layout.shared_experts:
            tensor = tensor.transpose(0, 1)
        split_dim = layout.split_dim
    else:
        tensor = tensor.view(shape)
        split_dim = (-1 if tensor.ndim == 2 else -2) if name.endswith("w_up_gate") else None
    if tensor.ndim not in (2, 3):
        raise ValueError(f"Expected logical matrices for {name}, got {tensor.shape}")
    if split_dim is not None and tensor.shape[split_dim] % 2:
        raise ValueError(f"Packed projection dimension must be even for {name}")
    parts = tensor.chunk(2, dim=split_dim) if split_dim is not None else (tensor,)
    return [matrix for part in parts for matrix in (part.unbind(0) if part.ndim == 3 else (part,))]


@torch.no_grad()
def dion_update(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    momentums: list[torch.Tensor],
    *,
    lr,
    momentum: float,
    weight_decay,
    process_group: Optional[ProcessGroup] = None,
    max_batch_bytes: int = 256 * 1024**2,
) -> None:
    """Execute Dion's same-shape async tasks, updating caller-owned tensors in place.

    Dense DTensors use Dion's matrix-sharded all-to-all path. Plain matrices may
    use a process group to distribute orthogonalization of replicated matrices.
    Callers must bypass this function entirely on skipped steps.
    """
    update, runtime, task, polar = require_dion()
    if not (len(params) == len(grads) == len(momentums)):
        raise ValueError("Parameter, gradient, and momentum counts differ")
    buckets = defaultdict(list)
    for param, grad, mom in zip(params, grads, momentums):
        if param.ndim != 2 or param.dtype != torch.float32:
            raise ValueError("Dion Muon requires logical 2D FP32 parameters")
        mesh = None
        shard_dim = None
        if isinstance(param, DTensor):
            mesh = param.device_mesh
            # HSDP's replicated dimension must not participate in all-to-all.
            shards = [i for i, p in enumerate(param.placements) if isinstance(p, Shard)]
            if len(shards) > 1 or any(
                not isinstance(p, (Shard, Replicate)) for p in param.placements
            ):
                raise ValueError("Dion supports one matrix-sharded mesh dimension")
            if shards:
                axis = shards[0]
                pg = mesh.get_group(axis)
                shard_dim = param.placements[axis].dim
            else:
                pg = None
        else:
            pg = process_group
        if isinstance(grad, DTensor) != isinstance(param, DTensor) or isinstance(
            mom, DTensor
        ) != isinstance(param, DTensor):
            raise ValueError("Dion parameter, gradient, and momentum layouts must match")
        if isinstance(param, DTensor):
            if grad.placements != param.placements or mom.placements != param.placements:
                raise ValueError("Dion parameter, gradient, and momentum placements must match")
        buckets[(tuple(param.shape), param.device, pg, shard_dim)].append((param, grad, mom))

    def tasks():
        for (shape, device, pg, shard_dim), entries in buckets.items():
            # Bound live stacked matrices and Polar Express workspace.
            per_matrix = 4 * (6 * shape[0] * shape[1] + 3 * min(shape) ** 2)
            batch = max(1, max_batch_bytes // per_matrix)
            for start in range(0, len(entries), batch):
                chunk = entries[start : start + batch]
                yield task(
                    update(
                        X=[e[0] for e in chunk],
                        G=[e[1] for e in chunk],
                        M=[e[2] for e in chunk],
                        lr=torch.as_tensor(lr, dtype=torch.float32, device=device),
                        momentum=torch.tensor(momentum),
                        weight_decay=torch.as_tensor(
                            weight_decay, dtype=torch.float32, device=device
                        ),
                        epsilon=torch.tensor(1e-8),
                        nesterov=True,
                        flatten=False,
                        adjust_lr="rms_norm",
                        device_rank=torch.distributed.get_rank(pg) if pg else 0,
                        world_size=torch.distributed.get_world_size(pg) if pg else 1,
                        process_group=pg,
                        shard_dim=shard_dim,
                        newton_schulz_func=polar,
                        cautious_wd=False,
                    )
                )

    runtime(tasks(), max_concurrent_tasks=3).run()
