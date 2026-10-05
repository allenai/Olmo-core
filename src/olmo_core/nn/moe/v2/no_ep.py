from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Union, cast

import torch

from olmo_core._nvtx import nvtx

from ...moe.utils import (
    async_copy_to_cpu,
    run_on_stream_no_compile,
    wait_stream_no_compile,
)
from ..utils import moe_permute_no_compile, moe_unpermute_no_compile
from .fp8 import shared_experts_forward_rowwise_fp8
from .routed_experts import requires_host_side_split_sizes

if TYPE_CHECKING:
    from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock


def _debug_tensors_enabled() -> bool:
    return os.getenv("OLMO_MOE_ROWWISE_DEBUG_TENSORS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _shared_experts_forward(
    block: OLMoDDPTransformerBlock,
    moe_inp: torch.Tensor,
    shared_weights: Optional[torch.Tensor],
    output_shape: torch.Size,
    *,
    use_rowwise_fp8: bool,
) -> torch.Tensor:
    assert block.shared_experts is not None
    if use_rowwise_fp8:
        assert block.rowwise_fp8 is not None
        shared_out = shared_experts_forward_rowwise_fp8(
            block, moe_inp, use_fast_accum=block.rowwise_fp8.use_fast_accum
        )
    else:
        shared_out = block.shared_experts(moe_inp)
    return block._mix_shared_out(shared_out, shared_weights, output_shape)


def combined_forward_no_ep(
    block: OLMoDDPTransformerBlock,
    x: torch.Tensor,
    *,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    **kwargs,
) -> torch.Tensor:
    """Forward function without expert parallelism."""
    self = block
    assert self.routed_experts is not None
    assert self.routed_experts_router is not None

    B, S, D = x.shape

    block_inp = x
    del x

    segment_ids = kwargs.pop("segment_ids", None)
    attn_res_out = self._res_norm_attn(block_inp, **kwargs)
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

    shared_rowwise_fp8_cfg = self.rowwise_fp8
    routed_rowwise_fp8_cfg = self.routed_experts.rowwise_fp8
    # The grouped MXFP8 expert kernels are local compute kernels; they do not
    # intrinsically require an expert-parallel process group. Historically the
    # flag was only forwarded by the rowwise EP path, so pure-DP jobs always
    # took the ordinary BF16 no-EP path.
    use_shared_rowwise_fp8 = (
        shared_rowwise_fp8_cfg is not None
        and shared_rowwise_fp8_cfg.enabled
        and moe_inp.device.type == "cuda"
    )
    use_routed_rowwise_fp8 = (
        routed_rowwise_fp8_cfg is not None
        and routed_rowwise_fp8_cfg.enabled
        and routed_moe_inp.device.type == "cuda"
    )

    if requires_host_side_split_sizes() and not use_routed_rowwise_fp8:
        local_batch_size_per_global_routed_expert_cpu, copy_stream, dtoh_event = async_copy_to_cpu(
            local_batch_size_per_global_routed_expert,
            event=self._dtoh_event,
        )
    else:
        dtoh_event = None
        local_batch_size_per_global_routed_expert_cpu = None

    if self.shared_experts_router:
        (
            local_x_global_shared_expert_weights,
            _,
            _,
            _,
        ) = self.shared_experts_router(
            moe_inp,
            True,
            loss_div_factor=loss_div_factor,
        )
    else:
        local_x_global_shared_expert_weights = None

    routed_in_shape = routed_moe_inp.size()

    mixed_shared_out = None
    if self.shared_experts is not None:
        wait_stream_no_compile(
            this_stream=self.get_dense_stream(),
            other_stream=torch.cuda.current_stream(),
        )

        mixed_shared_out = run_on_stream_no_compile(
            self.get_dense_stream(),
            _shared_experts_forward,
            self,
            moe_inp,
            local_x_global_shared_expert_weights,
            attn_res_out.shape,
            use_rowwise_fp8=use_shared_rowwise_fp8,
        )

    routed_moe_inp = routed_moe_inp.view(-1, routed_in_shape[-1])

    routing_map = local_x_global_routed_expert_indices.view(
        -1, self.routed_experts_router.top_k
    ).int()
    num_out_tokens = routing_map.size(0) * self.routed_experts_router.top_k
    hidden_shape_before_permute = routed_moe_inp.shape

    with nvtx.annotate("Permute", color="green"):
        permutated_input_tokens, reversed_input_permutation_mapping = moe_permute_no_compile(
            inp=routed_moe_inp,
            routing_map=routing_map,
            num_out_tokens=num_out_tokens,
            map_type="index",
        )

    # The row count is always B * S * top_k for no-EP routing. Marking it
    # dynamic makes downstream shape-specialized kernels harder to compile and
    # can trigger Dynamo constraint violations.
    # torch._dynamo.mark_dynamic(permutated_input_tokens, 0)

    if use_routed_rowwise_fp8:
        # scaled_grouped_mm consumes device-side int32 offsets. Do not take the
        # legacy grouped_gemm CPU split-size path for MXFP8 experts.
        mlp_x = self.routed_experts(
            permutated_input_tokens,
            local_batch_size_per_global_routed_expert,
            use_rowwise_fp8=True,
        )
    elif requires_host_side_split_sizes():
        assert dtoh_event is not None
        dtoh_event = cast(torch.cuda.Event, dtoh_event)
        dtoh_event.synchronize()
        mlp_x = self.routed_experts(
            permutated_input_tokens, local_batch_size_per_global_routed_expert_cpu
        )
    else:
        mlp_x = self.routed_experts(
            permutated_input_tokens, local_batch_size_per_global_routed_expert
        )
    if _debug_tensors_enabled() and self.block_idx == 0:
        self._debug_no_ep_expert_out = mlp_x.detach()
        self._debug_no_ep_batch_size_per_expert = local_batch_size_per_global_routed_expert.detach()

    with nvtx.annotate("Unpermute", color="green"):
        unpermutated_x: torch.Tensor = moe_unpermute_no_compile(
            inp=mlp_x,
            row_id_map=reversed_input_permutation_mapping,
            restore_shape=hidden_shape_before_permute,
            map_type="index",
            merging_probs=local_x_global_routed_expert_weights.view(
                -1, self.routed_experts_router.top_k
            ),
        )
    if _debug_tensors_enabled() and self.block_idx == 0:
        self._debug_no_ep_combined_local_x = unpermutated_x.detach()

    x_moe = unpermutated_x.view(routed_in_shape)
    x_moe = self._restore_routed_moe_output(x_moe)

    wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream())

    mlp_out = self._merge_routed_and_shared(x_moe, mixed_shared_out)

    final_out = self._res_norm_mlp(attn_res_out, mlp_out)

    return self._attach_routed_aux_loss(
        final_out,
        routed_expert_router_aux_loss_info,
    )
