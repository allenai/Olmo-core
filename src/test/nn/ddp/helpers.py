"""Shared setup for DDP block execution and expert-parallel parity tests."""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.moe import MoERouterGatingFunction
from olmo_core.nn.moe.v2.ep_config import ExpertParallelConfig, ExpertParallelPath
from olmo_core.nn.moe.v2.fp8 import MoERowwiseFP8Config
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig


def _build_ep_mesh() -> DeviceMesh:
    world_size = dist.get_world_size()
    mesh = torch.arange(world_size, dtype=torch.int).view(1, world_size)
    return DeviceMesh(
        device_type="cuda",
        mesh=mesh,
        mesh_dim_names=("ep_dp", "ep_mp"),
    )


def _build_block(
    *,
    ep_no_sync: bool,
    ep: ExpertParallelConfig | None = None,
    ep_no_sync_capacity_factor: float = 2.0,
    d_model: int = 512,
    hidden_size: int = 1024,
    num_experts: int = 4,
    top_k: int = 1,
    num_shared_experts: int = 0,
    shared_hidden_size: int = 512,
    uniform_expert_assignment: bool = True,
    init_device: str = "cuda",
    checkpoint_combined_ep_tbo: bool = False,
    rowwise_fp8: MoERowwiseFP8Config | None = None,
    routed_rowwise_fp8: MoERowwiseFP8Config | None = None,
    ep_no_sync_use_rowwise_all_to_all: bool = False,
    ep_no_sync_rowwise_backend: str = "nvshmem",
) -> OLMoDDPTransformerBlock:
    if ep is None:
        rowwise_backend = ep_no_sync_rowwise_backend.lower()
        if rowwise_backend != "nvshmem":
            raise OLMoConfigurationError("ep_no_sync_rowwise_backend must be 'nvshmem'")
        if not ep_no_sync:
            path = ExpertParallelPath.sync_1d
        elif ep_no_sync_use_rowwise_all_to_all:
            path = ExpertParallelPath.rowwise_nvshmem
        else:
            path = ExpertParallelPath.no_sync_1d
        ep = ExpertParallelConfig(
            path=path,
            capacity_factor=ep_no_sync_capacity_factor,
            rowwise_get_nblocks=256,
            rowwise_put_nblocks=256,
            rowwise_weighted_put_nblocks=128,
            checkpoint_tbo=checkpoint_combined_ep_tbo,
        )

    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=DType.float32,
    )
    from olmo_core.nn.attention import AttentionConfig, AttentionType

    return OLMoDDPTransformerBlock(
        d_model=d_model,
        block_idx=0,
        n_layers=1,
        sequence_mixer=AttentionConfig(
            name=AttentionType.default,
            n_heads=2,
            n_kv_heads=2,
            bias=False,
            backend=AttentionBackendName.torch,
            dtype=DType.float32,
        ),
        attention_norm=layer_norm,
        routed_experts_router=MoERouterConfigV2(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            gating_function=MoERouterGatingFunction.softmax,
            uniform_expert_assignment=uniform_expert_assignment,
            lb_loss_weight=None,
            z_loss_weight=None,
            dtype=DType.float32,
        ),
        routed_experts=RoutedExpertsConfig(
            d_model=d_model,
            hidden_size=hidden_size,
            num_experts=num_experts,
            bias=False,
            dtype=DType.float32,
            rowwise_fp8=routed_rowwise_fp8,
        ),
        shared_experts=(
            SharedExpertsConfig(
                d_model=d_model,
                hidden_size=shared_hidden_size,
                num_experts=num_shared_experts,
                bias=False,
                dtype=DType.float32,
            )
            if num_shared_experts > 0
            else None
        ),
        shared_experts_router=None,
        feed_forward_norm=layer_norm,
        ep=ep,
        rowwise_fp8=rowwise_fp8,
        init_device=init_device,
    )


def _init_block_params(block: OLMoDDPTransformerBlock):
    torch.manual_seed(1234)
    with torch.no_grad():
        for p in block.parameters():
            if p.is_floating_point():
                p.normal_(mean=0.0, std=0.02)


def _install_forced_router(block: OLMoDDPTransformerBlock):
    def _make_forced_forward(router):
        def _forced_forward(local_x, scores_only, loss_div_factor=None):
            del loss_div_factor
            B, S, _ = local_x.shape
            if scores_only:
                return (
                    torch.ones(
                        B,
                        S,
                        router.num_experts,
                        device=local_x.device,
                        dtype=local_x.dtype,
                    ),
                    None,
                    None,
                    None,
                )

            expert_weights = torch.ones(
                B,
                S,
                router.top_k,
                device=local_x.device,
                dtype=local_x.dtype,
            )
            expert_indices = torch.zeros(
                B,
                S,
                router.top_k,
                device=local_x.device,
                dtype=torch.long,
            )
            batch_size_per_expert = torch.zeros(
                router.num_experts,
                device=local_x.device,
                dtype=torch.long,
            )
            batch_size_per_expert[0] = B * S * router.top_k
            return expert_weights, expert_indices, batch_size_per_expert, None

        return _forced_forward

    assert block.routed_experts_router is not None
    block.routed_experts_router.forward = _make_forced_forward(block.routed_experts_router)  # type: ignore[method-assign]
    if block.shared_experts_router is not None:
        block.shared_experts_router.forward = _make_forced_forward(block.shared_experts_router)  # type: ignore[method-assign]


def _install_deterministic_topk_router(block: OLMoDDPTransformerBlock):
    def _make_deterministic_forward(router):
        def _deterministic_forward(local_x, scores_only, loss_div_factor=None):
            del loss_div_factor
            B, S, _ = local_x.shape
            if scores_only:
                return (
                    torch.ones(
                        B,
                        S,
                        router.num_experts,
                        device=local_x.device,
                        dtype=local_x.dtype,
                    ),
                    None,
                    None,
                    None,
                )

            top_k = router.top_k
            num_experts = router.num_experts
            token_ids = torch.arange(B * S, device=local_x.device, dtype=torch.long).unsqueeze(1)
            route_offsets = torch.arange(top_k, device=local_x.device, dtype=torch.long).unsqueeze(
                0
            )
            expert_indices = (token_ids + route_offsets + dist.get_rank() * 3) % num_experts
            expert_indices = expert_indices.view(B, S, top_k)

            weights = torch.arange(1, top_k + 1, device=local_x.device, dtype=local_x.dtype)
            weights = weights / weights.sum().clamp_min(1e-6)
            expert_weights = weights.view(1, 1, top_k).expand(B, S, top_k).contiguous()

            batch_size_per_expert = torch.bincount(
                expert_indices.reshape(-1), minlength=num_experts
            ).to(dtype=torch.long)
            return expert_weights, expert_indices, batch_size_per_expert, None

        return _deterministic_forward

    assert block.routed_experts_router is not None
    block.routed_experts_router.forward = _make_deterministic_forward(block.routed_experts_router)  # type: ignore[method-assign]
    if block.shared_experts_router is not None:
        block.shared_experts_router.forward = _make_deterministic_forward(  # type: ignore[method-assign]
            block.shared_experts_router
        )
