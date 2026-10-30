from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING, Optional, Tuple, Union, cast

import torch

from olmo_core.kernels import symm_mem_vdev2d as symm_mem_vdev2d_kernels

from ...moe.utils import run_on_stream_no_compile, wait_stream_no_compile
from .checkpointing import get_rowwise_checkpoint_state
from .comm import (
    _DispatchRowwiseAutograd,
    _DispatchRowwiseFP8Autograd,
    _RowwiseCombineWeightedAutograd,
    _RowwiseCombineWeightedFP8Autograd,
    _RowwiseFP8DispatchExpertsCombineAutograd,
)
from .ep_no_sync_buffers import (
    acquire_ep_no_sync_fp8_combine_gather_lease,
    acquire_ep_no_sync_fp8_dispatch_out_lease,
    acquire_ep_no_sync_rowwise_lifetime_leases,
    compute_ep_no_sync_rank_capacity,
    get_cached_ep_no_sync_buffers,
    get_cached_ep_no_sync_rowwise_fp8_buffers,
    get_ep_no_sync_buffers,
    get_ep_no_sync_group_name,
    get_ep_no_sync_rowwise_fp8_buffers,
    get_or_init_ep_no_sync_symm_tensor,
    use_ep_no_sync_rowwise_symm_combine_gather,
    use_ep_no_sync_rowwise_symm_combine_out,
    use_ep_no_sync_rowwise_symm_dispatch_in,
)
from .ep_no_sync_common import (
    padded_local_expert_splits_for_capacity,
    rowwise_stage_debug_print,
    rowwise_stage_debug_sync,
    sync_tail_drop_allowed_splits_single_a2a,
)
from .ep_no_sync_rowwise_helpers import (
    accumulate_ep_no_sync_rowwise_metrics,
    build_rowwise_route_maps,
    should_accumulate_ep_no_sync_rowwise_metrics,
)
from .fp8 import (
    shared_experts_forward1_rowwise_fp8,
    shared_experts_forward2_rowwise_fp8,
)
from .routed_experts import requires_host_side_split_sizes, use_torch_grouped_mm

if TYPE_CHECKING:
    from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock


def _debug_tensors_enabled() -> bool:
    return os.getenv("OLMO_MOE_ROWWISE_DEBUG_TENSORS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _rowwise_ep_spans_nodes(block: "OLMoDDPTransformerBlock") -> bool:
    local_world_size = 0
    try:
        local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "0") or "0")
    except ValueError:
        local_world_size = 0
    local_cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    intra_node_limit = max(local_world_size, local_cuda_devices, 8)
    return block.ep_world_size > intra_node_limit


def _rowwise_regular_combine_put_enabled(block: "OLMoDDPTransformerBlock") -> bool:
    raw = os.getenv("OLMO_MOE_ROWWISE_COMBINE_PUT", "auto").strip().lower()
    if raw in {"auto", ""}:
        # Current measurements:
        # - single-node/NVLink regular rowwise is faster with PUT combine;
        # - inter-node/IB regular rowwise is faster with GET combine, provided
        #   the local GET destination is symmetric/NVSHMEM-visible.
        return not _rowwise_ep_spans_nodes(block)
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(
        "OLMO_MOE_ROWWISE_COMBINE_PUT must be one of "
        "auto|0|1|true|false|yes|no|on|off, got "
        f"{raw!r}"
    )


def _shared_forward1(
    block: OLMoDDPTransformerBlock,
    moe_inp: torch.Tensor,
    *,
    use_rowwise_fp8: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert block.shared_experts is not None
    if use_rowwise_fp8:
        assert block.rowwise_fp8 is not None
        return shared_experts_forward1_rowwise_fp8(
            block, moe_inp, use_fast_accum=block.rowwise_fp8.use_fast_accum
        )
    return block.shared_experts.forward1(moe_inp)


def _shared_forward2(
    block: OLMoDDPTransformerBlock,
    up: torch.Tensor,
    gate: torch.Tensor,
    weights: Optional[torch.Tensor],
    output_shape: torch.Size,
    *,
    use_rowwise_fp8: bool,
) -> torch.Tensor:
    assert block.shared_experts is not None
    if use_rowwise_fp8:
        assert block.rowwise_fp8 is not None
        shared_out = shared_experts_forward2_rowwise_fp8(
            block, up, gate, output_shape, use_fast_accum=block.rowwise_fp8.use_fast_accum
        )
    else:
        shared_out = block.shared_experts.forward2(up, gate, output_shape)
    return block._mix_shared_out(shared_out, weights, output_shape)


def combined_forward_ep_no_sync_rowwise(
    block: OLMoDDPTransformerBlock,
    x: torch.Tensor,
    *,
    activation_checkpointing: Optional[bool] = None,
    accumulate_routed_aux_loss_metrics: Optional[bool] = None,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    **kwargs,
) -> torch.Tensor:
    """Forward with EP no-sync using row-wise NVSHMEM dispatch/combine."""
    self = block
    assert self.routed_experts is not None
    assert self.routed_experts_router is not None
    assert self.ep_enabled
    assert self.num_local_routed_experts is not None
    assert use_torch_grouped_mm(), "EP no-sync implementation requires torch.grouped_mm support"
    assert (
        not requires_host_side_split_sizes()
    ), "EP no-sync implementation does not support host-side split size communication"
    group_name = get_ep_no_sync_group_name(self)
    B, S, D = x.shape
    rowwise_stage_debug_print(
        "rowwise:enter",
        block=self.block_idx,
        shape=tuple(x.shape),
        group=group_name,
    )

    block_inp = x
    del x

    segment_ids = kwargs.pop("segment_ids", None)
    attn_res_out = self._checkpointed_res_norm_attn(block_inp, **kwargs)

    kwargs.pop("max_doc_len", None)
    kwargs.pop("cu_doc_lens", None)
    moe_inp = self._prepare_moe_input(attn_res_out)
    routed_moe_inp = self._prepare_routed_moe_input(moe_inp)

    (
        local_x_global_routed_expert_weights,
        local_x_global_routed_expert_indices,
        local_batch_size_per_global_routed_expert,
        routed_expert_router_aux_loss_info,
    ) = self.routed_experts_router(
        moe_inp,
        False,
        loss_div_factor=loss_div_factor,
        **({"segment_ids": segment_ids} if segment_ids is not None else {}),
    )
    rowwise_stage_debug_print(
        "rowwise:router-exit",
        block=self.block_idx,
        numel=int(local_x_global_routed_expert_indices.numel()),
        top_k=self.routed_experts_router.top_k,
    )

    wait_stream_no_compile(
        this_stream=self.get_dense_stream(),
        other_stream=torch.cuda.current_stream(),
    )

    if self.shared_experts_router:
        (
            local_x_global_shared_expert_weights,
            _,
            _,
            _,
        ) = run_on_stream_no_compile(
            self.get_dense_stream(),
            self.shared_experts_router,
            moe_inp,
            True,
            loss_div_factor=loss_div_factor,
        )
    else:
        local_x_global_shared_expert_weights = None

    routed_in_shape = routed_moe_inp.size()
    routed_moe_inp = routed_moe_inp.view(-1, routed_in_shape[-1])
    rowwise_fp8_cfg = self.rowwise_fp8
    use_rowwise_fp8 = (
        rowwise_fp8_cfg is not None
        and rowwise_fp8_cfg.enabled
        and routed_moe_inp.device.type == "cuda"
        and self.ep.uses_rowwise_buffers
    )
    if not use_rowwise_fp8:
        rowwise_fp8_cfg = None

    num_out_tokens = local_x_global_routed_expert_indices.numel()
    num_input_tokens = routed_moe_inp.shape[0]
    top_k = self.routed_experts_router.top_k
    if activation_checkpointing is None:
        (
            activation_checkpointing,
            accumulate_routed_aux_loss_metrics,
        ) = get_rowwise_checkpoint_state()
    use_symm_dispatch_in = (not use_rowwise_fp8) and use_ep_no_sync_rowwise_symm_dispatch_in(self)
    use_symm_combine_out = (
        (not use_rowwise_fp8)
        and (not activation_checkpointing)
        and use_ep_no_sync_rowwise_symm_combine_out(self)
    )
    use_symm_combine_gather = (
        (not use_rowwise_fp8)
        and (not activation_checkpointing)
        and use_ep_no_sync_rowwise_symm_combine_gather(self)
    )
    use_regular_combine_put = (
        (not use_rowwise_fp8)
        and (not activation_checkpointing)
        and (not torch.is_grad_enabled())
        and _rowwise_regular_combine_put_enabled(self)
    )
    if use_regular_combine_put:
        use_symm_combine_gather = True
    force_scratch_lifetime_buffers = bool(
        getattr(self, "_ep_no_sync_force_scratch_lifetime_buffers", False)
    )
    if force_scratch_lifetime_buffers:
        use_symm_combine_out = False
        use_symm_combine_gather = False
        use_regular_combine_put = False
    lease_lifetime_buffers = (
        torch.is_grad_enabled()
        and not activation_checkpointing
        and not force_scratch_lifetime_buffers
    )
    lease_dispatch_out = (not use_rowwise_fp8) and lease_lifetime_buffers
    lease_combine_out = use_symm_combine_out and lease_lifetime_buffers
    lease_combine_gather = use_symm_combine_gather and lease_lifetime_buffers
    use_fused_rowwise_fp8 = (
        use_rowwise_fp8 and rowwise_fp8_cfg is not None and rowwise_fp8_cfg.fused_autograd
    )

    with torch.no_grad():
        requested_splits = local_batch_size_per_global_routed_expert.to(dtype=torch.long)
        rank_capacity = compute_ep_no_sync_rank_capacity(self, num_out_tokens)
        rowwise_stage_debug_print(
            "rowwise:sync-tail-enter",
            block=self.block_idx,
            rank_capacity=rank_capacity,
            num_out_tokens=num_out_tokens,
        )
        (
            allowed_splits,
            recv_splits_by_src_local,
            _drop_token_cnt,
            keep_from_src_dest_local,
        ) = cast(
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            sync_tail_drop_allowed_splits_single_a2a(
                self,
                requested_splits,
                rank_capacity=rank_capacity,
                return_keep_matrix=True,
            ),
        )
        rowwise_stage_debug_print(
            "rowwise:sync-tail-exit",
            block=self.block_idx,
            rank_capacity=rank_capacity,
        )
        dispatch_in_cap = num_out_tokens
        dispatch_out_cap = rank_capacity
        combine_in_cap = rank_capacity
        combine_out_cap = num_input_tokens
        if should_accumulate_ep_no_sync_rowwise_metrics(accumulate_routed_aux_loss_metrics):
            accumulate_ep_no_sync_rowwise_metrics(
                self,
                drop_token_cnt=_drop_token_cnt,
                num_out_tokens=num_out_tokens,
                recv_splits_by_src_local=recv_splits_by_src_local,
                rank_capacity=rank_capacity,
            )

    buffers = None
    dispatch_out_q: Optional[torch.Tensor] = None
    dispatch_out_scales: Optional[torch.Tensor] = None
    combine_in_q: Optional[torch.Tensor] = None
    combine_in_scales: Optional[torch.Tensor] = None
    if use_rowwise_fp8:
        assert rowwise_fp8_cfg is not None
        use_fp8_symm_dispatch_in = use_ep_no_sync_rowwise_symm_dispatch_in(self)
        use_fp8_symm_combine_gather = use_ep_no_sync_rowwise_symm_combine_gather(self)
        fp8_lease_combine_gather = use_fp8_symm_combine_gather and lease_lifetime_buffers
        rowwise_stage_debug_print(
            "rowwise:fp8-get-buffers-enter",
            block=self.block_idx,
            dispatch_out_cap=dispatch_out_cap,
            combine_in_cap=combine_in_cap,
            use_symm_dispatch_in=use_fp8_symm_dispatch_in,
            use_symm_combine_gather=use_fp8_symm_combine_gather,
            lease_lifetime=lease_lifetime_buffers,
        )
        fp8_buffers = get_cached_ep_no_sync_rowwise_fp8_buffers(
            self,
            dispatch_in_cap=num_input_tokens,
            dispatch_out_cap=dispatch_out_cap,
            combine_in_cap=combine_in_cap,
            combine_gather_cap=num_input_tokens,
            combine_gather_top_k=top_k,
            d_model=routed_moe_inp.shape[1],
            block_size=rowwise_fp8_cfg.block_size,
            device=routed_moe_inp.device,
            need_dispatch_in=use_fp8_symm_dispatch_in,
            need_dispatch_out=not lease_lifetime_buffers,
            need_combine_gather=not fp8_lease_combine_gather,
        )
        rowwise_stage_debug_print(
            "rowwise:fp8-get-buffers-cache",
            block=self.block_idx,
            cached=fp8_buffers is not None,
        )
        if fp8_buffers is None:
            fp8_buffers = get_ep_no_sync_rowwise_fp8_buffers(
                self,
                dispatch_in_cap=num_input_tokens,
                dispatch_out_cap=dispatch_out_cap,
                combine_in_cap=combine_in_cap,
                combine_gather_cap=num_input_tokens,
                combine_gather_top_k=top_k,
                d_model=routed_moe_inp.shape[1],
                block_size=rowwise_fp8_cfg.block_size,
                device=routed_moe_inp.device,
                lease_dispatch_out=False,
                lease_combine_gather=False,
                need_dispatch_in=use_fp8_symm_dispatch_in,
                need_dispatch_out=not lease_lifetime_buffers,
                need_combine_gather=not fp8_lease_combine_gather,
            )
        rowwise_stage_debug_print(
            "rowwise:fp8-get-buffers-exit",
            block=self.block_idx,
        )
        if lease_lifetime_buffers:
            rowwise_stage_debug_print(
                "rowwise:fp8-dispatch-out-lease-enter",
                block=self.block_idx,
            )
            dispatch_out_lease = acquire_ep_no_sync_fp8_dispatch_out_lease(
                self,
                dispatch_out_cap=dispatch_out_cap,
                d_model=routed_moe_inp.shape[1],
                block_size=rowwise_fp8_cfg.block_size,
                device=routed_moe_inp.device,
            )
            fp8_buffers = replace(
                fp8_buffers,
                dispatch_out_q=dispatch_out_lease.tensor("dispatch_out_q"),
                dispatch_out_scales=dispatch_out_lease.tensor("dispatch_out_scales"),
                dispatch_out_lease=dispatch_out_lease,
            )
            rowwise_stage_debug_print(
                "rowwise:fp8-dispatch-out-lease-exit",
                block=self.block_idx,
            )
            if fp8_lease_combine_gather:
                rowwise_stage_debug_print(
                    "rowwise:fp8-combine-gather-lease-enter",
                    block=self.block_idx,
                )
                combine_gather_lease = acquire_ep_no_sync_fp8_combine_gather_lease(
                    self,
                    combine_gather_cap=num_input_tokens,
                    combine_gather_top_k=top_k,
                    d_model=routed_moe_inp.shape[1],
                    block_size=rowwise_fp8_cfg.block_size,
                    device=routed_moe_inp.device,
                )
                fp8_buffers = replace(
                    fp8_buffers,
                    combine_gather_q=combine_gather_lease.tensor("combine_gather_q"),
                    combine_gather_scales=combine_gather_lease.tensor("combine_gather_scales"),
                    combine_gather_lease=combine_gather_lease,
                )
                rowwise_stage_debug_print(
                    "rowwise:fp8-combine-gather-lease-exit",
                    block=self.block_idx,
                )
        dispatch_out_q = fp8_buffers.dispatch_out_q
        dispatch_out_scales = fp8_buffers.dispatch_out_scales
        combine_in_q = fp8_buffers.combine_in_q
        combine_in_scales = fp8_buffers.combine_in_scales
    else:
        rowwise_stage_debug_print(
            "rowwise:get-buffers-enter",
            block=self.block_idx,
            dispatch_out_cap=dispatch_out_cap,
            combine_in_cap=combine_in_cap,
            d_model=routed_moe_inp.shape[-1],
        )
        buffers = get_cached_ep_no_sync_buffers(
            self,
            dispatch_in_cap=dispatch_in_cap,
            dispatch_out_cap=dispatch_out_cap,
            combine_in_cap=combine_in_cap,
            combine_out_cap=combine_out_cap,
            d_model=routed_moe_inp.shape[-1],
            dtype=routed_moe_inp.dtype,
            device=routed_moe_inp.device,
            need_dispatch_in=use_symm_dispatch_in,
            need_dispatch_meta=False,
            need_dispatch_out=not lease_dispatch_out,
            need_combine_in=True,
            need_combine_meta=False,
            need_combine_out=use_symm_combine_out and not lease_combine_out,
            need_combine_gather=use_symm_combine_gather and not lease_combine_gather,
            combine_gather_cap=num_input_tokens,
            combine_gather_top_k=top_k,
        )
        buffers_cache_hit = buffers is not None
        if buffers is None:
            buffers = get_ep_no_sync_buffers(
                self,
                dispatch_in_cap=dispatch_in_cap,
                dispatch_out_cap=dispatch_out_cap,
                combine_in_cap=combine_in_cap,
                combine_out_cap=combine_out_cap,
                d_model=routed_moe_inp.shape[-1],
                dtype=routed_moe_inp.dtype,
                device=routed_moe_inp.device,
                need_dispatch_in=use_symm_dispatch_in,
                need_dispatch_meta=False,
                need_dispatch_out=not lease_dispatch_out,
                need_combine_in=True,
                need_combine_meta=False,
                need_combine_out=use_symm_combine_out and not lease_combine_out,
                need_combine_gather=use_symm_combine_gather and not lease_combine_gather,
                combine_gather_cap=num_input_tokens,
                combine_gather_top_k=top_k,
            )
        rowwise_stage_debug_print(
            "rowwise:get-buffers-exit",
            block=self.block_idx,
            cached=buffers_cache_hit,
        )
        if lease_dispatch_out or lease_combine_out or lease_combine_gather:
            rowwise_stage_debug_print("rowwise:leases-enter", block=self.block_idx)
            leases = acquire_ep_no_sync_rowwise_lifetime_leases(
                self,
                dispatch_out_cap=dispatch_out_cap,
                combine_out_cap=combine_out_cap,
                combine_gather_cap=num_input_tokens,
                combine_gather_top_k=top_k,
                d_model=routed_moe_inp.shape[-1],
                dtype=routed_moe_inp.dtype,
                device=routed_moe_inp.device,
                need_dispatch_out=lease_dispatch_out,
                need_combine_out=lease_combine_out,
                need_combine_gather=lease_combine_gather,
            )
            buffers = replace(
                buffers,
                dispatch_out=(
                    leases.dispatch_out_lease.tensor("dispatch_out")
                    if leases.dispatch_out_lease is not None
                    else buffers.dispatch_out
                ),
                dispatch_out_is_shared=(
                    True
                    if leases.dispatch_out_lease is not None
                    else buffers.dispatch_out_is_shared
                ),
                combine_out=(
                    leases.combine_out_lease.tensor("combine_out")
                    if leases.combine_out_lease is not None
                    else buffers.combine_out
                ),
                combine_out_is_shared=(
                    True if leases.combine_out_lease is not None else buffers.combine_out_is_shared
                ),
                combine_gather=(
                    leases.combine_gather_lease.tensor("combine_gather")
                    if leases.combine_gather_lease is not None
                    else buffers.combine_gather
                ),
                dispatch_out_lease=leases.dispatch_out_lease,
                combine_out_lease=leases.combine_out_lease,
                combine_gather_lease=leases.combine_gather_lease,
            )
            rowwise_stage_debug_print("rowwise:leases-exit", block=self.block_idx)

    routing_map = local_x_global_routed_expert_indices.view(-1, top_k).int()

    with torch.no_grad():
        batch_size_per_local_expert = recv_splits_by_src_local.sum(dim=0, dtype=torch.long)
        padded_batch_size_per_local_expert = padded_local_expert_splits_for_capacity(
            recv_splits_by_src_local,
            rank_capacity=rank_capacity,
        )
        # Rowwise dispatch buffers are capacity-sized, but valid routed rows are
        # laid out densely at the front of the rank-major buffer. Keep capacity
        # padding out of grouped-mm forward/backward; the tail may contain stale
        # values and is only safe because combine kernels address explicit route
        # rows.
        expert_batch_size_per_local_expert = batch_size_per_local_expert
        routed_expert_offsets = torch.cumsum(
            expert_batch_size_per_local_expert.to(dtype=torch.int32),
            dim=0,
            dtype=torch.int32,
        )

    with torch.no_grad():
        rowwise_stage_debug_print("rowwise:build-route-enter", block=self.block_idx)
        dst_ranks, dst_rows = build_rowwise_route_maps(
            self,
            routing_map=routing_map,
            allowed_splits=allowed_splits,
            keep_from_src_dest_local=keep_from_src_dest_local,
        )
        rowwise_get_nblocks = self.ep.rowwise_get_nblocks
        rowwise_put_nblocks = self.ep.rowwise_put_nblocks
        rowwise_weighted_put_nblocks = self.ep.rowwise_weighted_put_nblocks
        rowwise_stage_debug_print(
            "rowwise:build-route-exit",
            block=self.block_idx,
            get_nblocks=rowwise_get_nblocks,
            put_nblocks=rowwise_put_nblocks,
            weighted_put_nblocks=rowwise_weighted_put_nblocks,
        )
        inverse_route_meta = None
        rowwise_combine_row_start = None
        rowwise_combine_num_rows = None
        if use_regular_combine_put:
            rowwise_stage_debug_print("rowwise:combine-put-meta-enter", block=self.block_idx)
            (
                compact_route_records,
                compact_wave_offsets,
            ) = symm_mem_vdev2d_kernels.rowwise_build_compact_route_records(
                dst_ranks,
                dst_rows,
                routing_map,
                num_local_experts=self.num_local_routed_experts,
                num_waves=1,
                nblocks=rowwise_put_nblocks,
            )
            inverse_route_meta = get_or_init_ep_no_sync_symm_tensor(
                self,
                name="rowwise_inverse_route_meta",
                shape=(rank_capacity, 2),
                dtype=torch.long,
                device=routed_moe_inp.device,
            )
            inverse_route_meta.fill_(-1)
            symm_mem_vdev2d_kernels.rowwise_inverse_route_meta_put_compact(
                inverse_route_meta,
                compact_route_records,
                compact_wave_offsets,
                src_rank=torch.distributed.get_rank(self.ep_pg),
                group_name=group_name,
                nblocks=rowwise_put_nblocks,
                pre_barrier=True,
                post_barrier=True,
                scalar_put=use_symm_dispatch_in,
            )
            rowwise_combine_row_start = batch_size_per_local_expert.new_zeros(())
            rowwise_combine_num_rows = batch_size_per_local_expert.sum()
            rowwise_stage_debug_print("rowwise:combine-put-meta-exit", block=self.block_idx)
            rowwise_stage_debug_sync("rowwise:combine-put-meta", routed_moe_inp.device)

    if self.shared_experts is not None:
        shared_out_up, shared_out_gate = run_on_stream_no_compile(
            self.get_dense_stream(),
            _shared_forward1,
            self,
            moe_inp.view(B, S, D),
            use_rowwise_fp8=use_rowwise_fp8,
        )
    else:
        shared_out_up, shared_out_gate = None, None

    route_probs = local_x_global_routed_expert_weights.view(-1, self.routed_experts_router.top_k)

    if use_fused_rowwise_fp8:
        assert rowwise_fp8_cfg is not None
        assert dispatch_out_q is not None
        assert dispatch_out_scales is not None
        assert combine_in_q is not None
        assert combine_in_scales is not None
        dispatch_in_q = fp8_buffers.dispatch_in_q if use_fp8_symm_dispatch_in else None  # type: ignore[union-attr]
        dispatch_in_scales = fp8_buffers.dispatch_in_scales if use_fp8_symm_dispatch_in else None  # type: ignore[union-attr]
        combine_gather_q = fp8_buffers.combine_gather_q if use_fp8_symm_combine_gather else None  # type: ignore[union-attr]
        combine_gather_scales = (
            fp8_buffers.combine_gather_scales if use_fp8_symm_combine_gather else None  # type: ignore[union-attr]
        )
        routed_experts = self.routed_experts
        routed_fp8_cfg = routed_experts.rowwise_fp8
        if routed_fp8_cfg is None or not routed_fp8_cfg.enabled:
            raise RuntimeError(
                "fused rowwise FP8 autograd requires routed expert rowwise_fp8 to be enabled"
            )
        up_gate_weight = routed_experts._rowwise_fp8_up_gate_weight
        down_weight = routed_experts._rowwise_fp8_down_weight
        up_gate_prequant = up_gate_weight.require_prequantized_rhs()
        up_gate_prequant_t = up_gate_weight.require_prequantized_rhs_for_dgrad()
        down_prequant = down_weight.require_prequantized_rhs()
        down_prequant_t = down_weight.require_prequantized_rhs_for_dgrad()

        up_wgrad_sink = None
        down_wgrad_sink = None
        up_wgrad_sink_transpose_last2 = False
        down_wgrad_sink_transpose_last2 = False
        up_gate_anchor = routed_experts.w_up_gate.transpose(1, 2)
        down_anchor = routed_experts.w_down
        if routed_fp8_cfg.fp8_only_params:
            up_wgrad_sink = up_gate_weight
            down_wgrad_sink = down_weight
            up_wgrad_sink_transpose_last2 = True
            up_gate_anchor = up_gate_anchor.detach()
            down_anchor = down_anchor.detach()

        local_x = _RowwiseFP8DispatchExpertsCombineAutograd.apply(
            routed_moe_inp,
            dst_ranks,
            dst_rows,
            routed_expert_offsets,
            route_probs,
            dispatch_out_q,
            dispatch_out_scales,
            combine_in_q,
            combine_in_scales,
            up_gate_anchor,
            down_anchor,
            up_gate_prequant,
            up_gate_prequant_t,
            down_prequant,
            down_prequant_t,
            fp8_buffers.dispatch_out_lease,  # type: ignore[union-attr]
            fp8_buffers.combine_gather_lease,  # type: ignore[union-attr]
            dispatch_in_q,
            dispatch_in_scales,
            combine_gather_q,
            combine_gather_scales,
            rowwise_fp8_cfg.block_size,
            rowwise_fp8_cfg.use_fast_accum,
            rowwise_fp8_cfg.fused_autograd_recompute_swiglu,
            group_name,
            self.ep_pg,
            rowwise_get_nblocks,
            rowwise_put_nblocks,
            up_wgrad_sink,
            up_wgrad_sink_transpose_last2,
            False,
            down_wgrad_sink,
            down_wgrad_sink_transpose_last2,
            False,
        )
    elif use_rowwise_fp8:
        assert rowwise_fp8_cfg is not None
        assert dispatch_out_q is not None
        assert dispatch_out_scales is not None
        dispatch_in_q = fp8_buffers.dispatch_in_q if use_fp8_symm_dispatch_in else None  # type: ignore[union-attr]
        dispatch_in_scales = fp8_buffers.dispatch_in_scales if use_fp8_symm_dispatch_in else None  # type: ignore[union-attr]
        combine_gather_q = fp8_buffers.combine_gather_q if use_fp8_symm_combine_gather else None  # type: ignore[union-attr]
        combine_gather_scales = (
            fp8_buffers.combine_gather_scales if use_fp8_symm_combine_gather else None  # type: ignore[union-attr]
        )
        dispatch_rank_major = _DispatchRowwiseFP8Autograd.apply(
            routed_moe_inp,
            dst_ranks,
            dst_rows,
            dispatch_out_q,
            dispatch_out_scales,
            fp8_buffers.dispatch_out_lease,  # type: ignore[union-attr]
            dispatch_in_q,
            dispatch_in_scales,
            combine_gather_q,
            combine_gather_scales,
            fp8_buffers.combine_gather_lease,  # type: ignore[union-attr]
            rowwise_fp8_cfg.block_size,
            group_name,
            self.ep_pg,
            rowwise_get_nblocks,
            rowwise_put_nblocks,
        )
    else:
        assert buffers is not None
        source_input_aliases_symm_input = False
        grad_out_aliases_symm_out = True
        rowwise_stage_debug_print("rowwise:dispatch-enter", block=self.block_idx)
        dispatch_rank_major = _DispatchRowwiseAutograd.apply(
            routed_moe_inp,
            buffers.dispatch_in if use_symm_dispatch_in else None,
            dst_ranks,
            dst_rows,
            buffers.dispatch_out,
            buffers.dispatch_out_lease,
            group_name,
            self.ep_pg,
            rowwise_get_nblocks,
            rowwise_put_nblocks,
            source_input_aliases_symm_input,
            grad_out_aliases_symm_out,
            True,
            False,
        )
        rowwise_stage_debug_print("rowwise:dispatch-exit", block=self.block_idx)
        rowwise_stage_debug_sync("rowwise:dispatch", routed_moe_inp.device)

    if not use_fused_rowwise_fp8:
        rowwise_stage_debug_print("rowwise:experts-enter", block=self.block_idx)
        if _debug_tensors_enabled() and self.block_idx == 0:
            self._debug_rowwise_dispatch_rank_major = dispatch_rank_major.detach()
            self._debug_rowwise_padded_batch_size_per_local_expert = (
                padded_batch_size_per_local_expert.detach()
            )
            self._debug_rowwise_recv_splits_by_src_local = recv_splits_by_src_local.detach()
            self._debug_rowwise_dst_ranks = dst_ranks.detach()
            self._debug_rowwise_dst_rows = dst_rows.detach()
        dispatch_rank_major = self.routed_experts(
            dispatch_rank_major,
            expert_batch_size_per_local_expert,
            down_proj_out=(None if use_rowwise_fp8 else buffers.combine_in.detach()),  # type: ignore[union-attr]
            up_proj_input_grad_out=(None if use_rowwise_fp8 else buffers.dispatch_out.detach()),  # type: ignore[union-attr]
            use_rowwise_fp8=use_rowwise_fp8,
            rowwise_fp8_input_q=(dispatch_out_q if use_rowwise_fp8 else None),
            rowwise_fp8_input_scales=(dispatch_out_scales if use_rowwise_fp8 else None),
        )
        rowwise_stage_debug_print("rowwise:experts-exit", block=self.block_idx)
        rowwise_stage_debug_sync("rowwise:experts", dispatch_rank_major.device)
        if _debug_tensors_enabled() and self.block_idx == 0:
            self._debug_rowwise_expert_out_rank_major = dispatch_rank_major.detach()

    wait_stream_no_compile(
        this_stream=self.get_dense_stream(),
        other_stream=torch.cuda.current_stream(),
    )

    if use_fused_rowwise_fp8:
        pass
    elif use_rowwise_fp8:
        assert rowwise_fp8_cfg is not None
        assert combine_in_q is not None
        assert combine_in_scales is not None
        combine_gather_q = fp8_buffers.combine_gather_q if use_fp8_symm_combine_gather else None  # type: ignore[union-attr]
        combine_gather_scales = (
            fp8_buffers.combine_gather_scales if use_fp8_symm_combine_gather else None  # type: ignore[union-attr]
        )
        local_x = _RowwiseCombineWeightedFP8Autograd.apply(
            dispatch_rank_major,
            dst_ranks,
            dst_rows,
            route_probs,
            combine_in_q,
            combine_in_scales,
            combine_gather_q,
            combine_gather_scales,
            rowwise_fp8_cfg.block_size,
            group_name,
            self.ep_pg,
            rowwise_get_nblocks,
            rowwise_put_nblocks,
        )
    else:
        assert buffers is not None
        expert_out_aliases_symm_expert_out = (
            dispatch_rank_major.data_ptr() == buffers.combine_in.data_ptr()
            and dispatch_rank_major.storage_offset() == buffers.combine_in.storage_offset()
            and tuple(dispatch_rank_major.shape) == tuple(buffers.combine_in.shape)
        )
        rowwise_stage_debug_print("rowwise:combine-enter", block=self.block_idx)
        if use_regular_combine_put:
            assert inverse_route_meta is not None
            assert rowwise_combine_row_start is not None
            assert rowwise_combine_num_rows is not None
            if not expert_out_aliases_symm_expert_out:
                buffers.combine_in.copy_(dispatch_rank_major)
            symm_mem_vdev2d_kernels.rowwise_combine_put(
                buffers.combine_in,
                buffers.combine_gather,
                inverse_route_meta,
                rowwise_combine_row_start,
                rowwise_combine_num_rows,
                group_name,
                nblocks=rowwise_put_nblocks,
                pre_barrier=True,
                post_barrier=True,
            )
            local_x = torch.empty(
                (num_input_tokens, routed_moe_inp.shape[-1]),
                device=routed_moe_inp.device,
                dtype=routed_moe_inp.dtype,
            )
            probs_f32 = route_probs if route_probs.dtype == torch.float32 else route_probs.float()
            symm_mem_vdev2d_kernels.rowwise_reduce_gathered_routes(
                buffers.combine_gather,
                probs_f32,
                local_x,
                route_ranks=dst_ranks,
            )
        else:
            local_x = _RowwiseCombineWeightedAutograd.apply(
                dispatch_rank_major,
                buffers.combine_in,
                buffers.combine_out if use_symm_combine_out else None,
                buffers.combine_out_lease if use_symm_combine_out else None,
                buffers.combine_gather if use_symm_combine_gather else None,
                buffers.combine_gather_lease if use_symm_combine_gather else None,
                dst_ranks,
                dst_rows,
                route_probs,
                group_name,
                self.ep_pg,
                rowwise_get_nblocks,
                rowwise_put_nblocks,
                rowwise_weighted_put_nblocks,
                expert_out_aliases_symm_expert_out,
                True,
                False,
            )
        rowwise_stage_debug_print("rowwise:combine-exit", block=self.block_idx)
        rowwise_stage_debug_sync("rowwise:combine", local_x.device)
    if _debug_tensors_enabled() and self.block_idx == 0:
        self._debug_rowwise_combined_local_x = local_x.detach()

    if self.shared_experts is not None:
        assert shared_out_up is not None
        assert shared_out_gate is not None

        mixed_shared_out = run_on_stream_no_compile(
            self.get_dense_stream(),
            _shared_forward2,
            self,
            shared_out_up,
            shared_out_gate,
            local_x_global_shared_expert_weights,
            attn_res_out.shape,
            use_rowwise_fp8=use_rowwise_fp8,
        )
    else:
        mixed_shared_out = None

    local_x = local_x.view(routed_in_shape)
    local_x = self._restore_routed_moe_output(local_x)
    wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream())

    mlp_out = self._merge_routed_and_shared(local_x, mixed_shared_out)

    final_out = self._res_norm_mlp(attn_res_out, mlp_out)
    rowwise_stage_debug_print("rowwise:exit", block=self.block_idx)
    return self._attach_routed_aux_loss(
        final_out,
        routed_expert_router_aux_loss_info,
        accumulate_metrics=accumulate_routed_aux_loss_metrics,
    )
