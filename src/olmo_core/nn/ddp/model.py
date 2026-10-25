import logging
import os
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.checkpoint import checkpoint, noop_context_fn

import olmo_core.nn.transformer
from olmo_core._nvtx import nvtx
from olmo_core.distributed.parallel import get_pp_mesh
from olmo_core.distributed.utils import hide_from_torch, unhide_from_torch
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.kernels import olmo_symm_mem
from olmo_core.kernels import symm_mem_vdev2d as symm_mem_vdev2d_kernels
from olmo_core.utils import mark_dynamic, move_to_device

from ..lm_head import LMOutputWithLoss
from ..moe.v2.checkpointing import checkpoint_recompute_context_fn
from ..moe.v2.ep_config import ExpertParallelPath
from ..moe.v2.ep_no_sync_buffers import (
    EpNoSyncSymmTensorInfo,
    _NoSyncSymmSharedPool,
    compute_ep_no_sync_rank_capacity,
    get_ep_no_sync_buffers,
    get_ep_no_sync_rowwise_fp8_buffers,
    iter_ep_no_sync_symm_tensor_infos,
    prewarm_ep_no_sync_rowwise_lifetime_leases,
    use_ep_no_sync_rowwise_symm_combine_gather,
    use_ep_no_sync_rowwise_symm_combine_out,
    use_ep_no_sync_rowwise_symm_dispatch_in,
)
from ..moe.v2.ep_no_sync_tbo_rowwise import (
    _NoSyncRowwiseTboPendingContext,
    ep_no_sync_rowwise_tbo_stage_c_launch,
    ep_no_sync_rowwise_tbo_stage_tail,
)
from .block import OLMoDDPTransformerBlock

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType
log = logging.getLogger(__name__)


recompute_context_fn = checkpoint_recompute_context_fn


def _fp32_post_grad_acc_hook(param: torch.Tensor):
    g = param.grad
    if g is None:
        return
    # upcast and accumulate in-place (no graph)

    if param._main_grad_fp32 is None:  # type: ignore[attr-defined]
        # first time init
        param._main_grad_fp32 = g.to(torch.float32)  # type: ignore[attr-defined]
    else:
        # param._main_grad_fp32.add_(g.to(torch.float32)) # type: ignore[attr-defined]
        param._main_grad_fp32.add_(g)  # type: ignore[attr-defined]
    # drop BF16 .grad to avoid double-accum & save memory
    param.grad = None


class OLMoDDPModel(olmo_core.nn.transformer.Transformer):
    """
    An MoE transformer implementation, to be used with one of the
    :class:`MoETransformerBlock` block types.

    Training currently uses BF16 activations, regardless of the parameter initialization dtype.
    Symmetric-buffer prewarming and model materialization share this precision contract.
    """

    _training_dtype: ClassVar[torch.dtype] = torch.bfloat16

    def __init__(self, *args, **kwargs):
        self.tbo = kwargs.pop("two_batch_overlap")
        self.recompute_all_blocks_by_chunk = kwargs.pop("recompute_all_blocks_by_chunk")
        self.recompute_each_block = kwargs.pop("recompute_each_block")
        self.recompute_block_keys: Optional[List[str]] = kwargs.pop("recompute_block_keys")
        self.checkpoint_tbo_dense_layers = False
        self.has_grad_accum_fp32_buffer = False  # whether the model has grad accum buffer for fp32 master grad, will be set in `attach_fp32_accum`
        self._compile_requested = False

        super().__init__(*args, **kwargs)
        self.ep_enabled = False  # default
        self._ep_modules = []
        # Historical padding tensors for single-world symmetric-memory paths:
        # every PE had to execute the same allocation sequence even when PP
        # stages owned different numbers of MoE blocks. Current OLMo-owned EP
        # paths should generally avoid needing these.
        self._ep_no_sync_dummy_symm_tensors: List[torch.Tensor] = []
        self._deepep_v2_runtime_cache: Dict[object, object] = {}

        assert not (
            self.recompute_all_blocks_by_chunk and self.recompute_each_block
        ), "Only one of recompute_all_blocks_by_chunk and recompute_each_block can be True."
        assert not (
            self.tbo and self.recompute_each_block
        ), "Cannot use TBO when recompute_each_block is True."
        # Chunk recompute wraps all blocks in one non-reentrant checkpoint, which
        # breaks DeepEP's recv_x leaf identity and drops expert grads. Only the
        # per-block reentrant path in _forward_blocks handles DeepEP correctly.
        if self.recompute_all_blocks_by_chunk and any(
            isinstance(block, OLMoDDPTransformerBlock)
            and block.has_routed_experts
            and block.ep.is_deepep
            for block in self.blocks.values()
        ):
            raise OLMoConfigurationError(
                "recompute_all_blocks_by_chunk is not supported with DeepEP expert "
                "parallelism; use recompute_each_block instead."
            )

    def named_ddp_blocks(self) -> Iterator[tuple[str, OLMoDDPTransformerBlock]]:
        for block_key, block in self.blocks.items():
            if isinstance(block, OLMoDDPTransformerBlock):
                yield block_key, block

    def named_required_ddp_blocks(self) -> Iterator[tuple[str, OLMoDDPTransformerBlock]]:
        for block_key, block in self.blocks.items():
            if not isinstance(block, OLMoDDPTransformerBlock):
                raise OLMoConfigurationError(
                    "OLMoDDPModel only supports OLMoDDPTransformerBlock blocks. "
                    "Use a shared-only OLMoDDPTransformerBlock for dense layers."
                )
            yield block_key, block

    def ddp_blocks(self) -> Iterator[OLMoDDPTransformerBlock]:
        for _, block in self.named_ddp_blocks():
            yield block

    def named_routed_blocks(self) -> Iterator[tuple[str, OLMoDDPTransformerBlock]]:
        for block_key, block in self.named_ddp_blocks():
            if block.has_routed_experts:
                yield block_key, block

    def routed_blocks(self) -> Iterator[OLMoDDPTransformerBlock]:
        for _, block in self.named_routed_blocks():
            yield block

    def named_shared_blocks(self) -> Iterator[tuple[str, OLMoDDPTransformerBlock]]:
        for block_key, block in self.named_ddp_blocks():
            if block.has_shared_experts:
                yield block_key, block

    def shared_blocks(self) -> Iterator[OLMoDDPTransformerBlock]:
        for _, block in self.named_shared_blocks():
            yield block

    def named_ep_no_sync_blocks(self) -> Iterator[tuple[str, OLMoDDPTransformerBlock]]:
        for block_key, block in self.named_routed_blocks():
            if block.ep.uses_olmo_symm:
                yield block_key, block

    def ep_no_sync_blocks(self) -> Iterator[OLMoDDPTransformerBlock]:
        for _, block in self.named_ep_no_sync_blocks():
            yield block

    def named_fp8_blocks(self) -> Iterator[tuple[str, object]]:
        for block_key, block in self.blocks.items():
            if getattr(block, "named_fp8_weight_stores", None) is not None:
                yield block_key, block
                continue
            attention = getattr(block, "attention", None)
            if getattr(attention, "named_fp8_weight_stores", None) is not None:
                yield block_key, block

    def fp8_blocks(self) -> Iterator[object]:
        for _, block in self.named_fp8_blocks():
            yield block

    @property
    def has_routed_blocks(self) -> bool:
        return any(True for _ in self.routed_blocks())

    def _check_tbo_requirements(self):
        # make sure dense blocks only appear before moe blocks
        # because TBO requires that moe blocks are consecutive
        # [dense, dense, moe, moe] is ok
        # [dense, moe, dense, moe] is not ok
        found_moe = False
        first_moe_idx = None
        for idx, block in enumerate(self.blocks.values()):
            is_routed_block = (
                isinstance(block, OLMoDDPTransformerBlock) and block.has_routed_experts
            )
            if found_moe and not is_routed_block:
                raise OLMoConfigurationError(
                    "When TBO is enabled, all dense blocks must appear before MoE blocks."
                )
            if is_routed_block and not found_moe:
                found_moe = True
                first_moe_idx = idx
            if is_routed_block:
                moe_block = cast(OLMoDDPTransformerBlock, block)
                if moe_block.ep.no_sync and moe_block.ep.shared_slots < 2:
                    raise OLMoConfigurationError(
                        "When TBO and EP no-sync are enabled, EP shared_slots must be >= 2 "
                        f"(block={moe_block.block_idx}, got {moe_block.ep.shared_slots})."
                    )
                if moe_block.ep.path != ExpertParallelPath.rowwise_nvshmem:
                    raise OLMoConfigurationError(
                        "TBO is only supported with "
                        f"EP path={ExpertParallelPath.rowwise_nvshmem!r}; "
                        f"got EP path={moe_block.ep.path!r} "
                        f"(block={moe_block.block_idx})."
                    )

        return first_moe_idx

    def purge_cuda_events(self):
        # set all events to None (so that the model can be deepcopied in PP split)
        for layer in self.ddp_blocks():
            layer.purge_cuda_events()

    def install_cuda_events(self):
        # re-install events after deepcopy
        for layer in self.ddp_blocks():
            layer.install_cuda_events()

    @property
    def is_moe(self) -> bool:
        return True

    def num_flops_per_token(self, seq_len: int) -> int:
        # TODO: under PP this returns full-model FLOPs/token via super(); confirm that's what the
        # speed monitor expects rather than the per-stage count.
        return super().num_flops_per_token(seq_len)

    def compute_auxiliary_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        from olmo_core.train.common import ReduceType

        mean_offset = 1.0
        if self.pp_enabled:
            # Change the divisor to 'world_size // pp_group_size'
            mean_offset = self._pp_group_size

        out: Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]] = {}
        for block_idx, block in self.named_routed_blocks():
            block_metrics = block.compute_metrics(reset=reset)
            for metric_name, (metric_val, reduce_type) in block_metrics.items():
                out[f"block {int(block_idx):02d}/{metric_name}"] = (metric_val, reduce_type)

                if self.pp_enabled and reduce_type == ReduceType.mean:
                    metric_val = metric_val.float() * mean_offset

                if metric_name not in out:
                    out[metric_name] = (metric_val, reduce_type)
                elif reduce_type in (ReduceType.mean, ReduceType.sum):
                    out[metric_name] = (
                        out[metric_name][0] + metric_val,
                        reduce_type,
                    )
                elif reduce_type == ReduceType.max:
                    out[metric_name] = (torch.max(out[metric_name][0], metric_val), reduce_type)
                else:
                    raise NotImplementedError(reduce_type)
        return out

    def reset_auxiliary_metrics(self):
        for block in self.routed_blocks():
            block.reset_metrics()

    def count_ep_no_sync_blocks(self) -> int:
        return sum(1 for _ in self.ep_no_sync_blocks())

    def iter_ep_no_sync_symm_tensor_infos(self) -> Iterator[EpNoSyncSymmTensorInfo]:
        for block in self.ep_no_sync_blocks():
            yield from iter_ep_no_sync_symm_tensor_infos(block)

    def count_non_rowwise_ep_no_sync_blocks(self) -> int:
        return sum(1 for block in self.ep_no_sync_blocks() if not block.ep.uses_rowwise_buffers)

    @torch.no_grad()
    def refresh_rowwise_fp8_cache(self) -> None:
        for block in self.ddp_blocks():
            block.refresh_rowwise_fp8_cache()

    def named_fp8_weight_stores(self) -> Iterator[tuple[str, object]]:
        for block_key, block in self.named_fp8_blocks():
            named_block_fp8_weight_stores = getattr(block, "named_fp8_weight_stores", None)
            if named_block_fp8_weight_stores is not None:
                for name, weight in named_block_fp8_weight_stores():
                    yield f"blocks.{block_key}.{name}", weight

            attention = getattr(block, "attention", None)
            named_attention_fp8_weight_stores = getattr(attention, "named_fp8_weight_stores", None)
            if named_attention_fp8_weight_stores is not None:
                for name, weight in named_attention_fp8_weight_stores():
                    yield f"blocks.{block_key}.attention.{name}", weight

    def named_mxfp8_expert_weights(self) -> Iterator[tuple[str, object]]:
        yield from self.named_fp8_weight_stores()

    def zero_fp8_weight_store_grads(self, set_to_none: bool = True) -> None:
        for _, weight in self.named_fp8_weight_stores():
            weight.zero_grad(set_to_none=set_to_none)  # type: ignore[attr-defined]

    def zero_mxfp8_expert_weight_grads(self, set_to_none: bool = True) -> None:
        self.zero_fp8_weight_store_grads(set_to_none=set_to_none)

    def disable_fp8_weight_anchor_grads(self) -> None:
        for block in self.ddp_blocks():
            block.disable_fp8_weight_anchor_grads()

    def disable_mxfp8_expert_anchor_grads(self) -> None:
        self.disable_fp8_weight_anchor_grads()

    def release_fp8_weight_anchor_storage(self) -> None:
        for block in self.ddp_blocks():
            block.release_fp8_weight_anchor_storage()

    def release_mxfp8_expert_anchor_storage(self) -> None:
        self.release_fp8_weight_anchor_storage()

    def sync_fp8_weight_store_grads_from_anchor(self) -> None:
        for block in self.ddp_blocks():
            block.sync_fp8_weight_store_grads_from_anchor()

    def sync_mxfp8_expert_weight_grads_from_anchor(self) -> None:
        self.sync_fp8_weight_store_grads_from_anchor()

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        self.zero_fp8_weight_store_grads(set_to_none=set_to_none)

    @torch.no_grad()
    def prewarm_deepep_v2_runtimes(
        self,
        *,
        max_local_microbatch_size: int,
    ) -> None:
        deepep_blocks = [block for block in self.routed_blocks() if block.ep.is_deepep]
        if not deepep_blocks:
            return

        param = next((p for p in self.parameters() if p.is_floating_point()), None)
        if param is None:
            raise RuntimeError("Cannot infer device for DeepEP runtime prewarm")

        # Import locally to keep the model/block module initialization order
        # simple: block.py also installs the DeepEP combined-forward method.
        from ..moe.v2.ep_deepep_v2 import prewarm_deepep_v2_runtime

        for block in deepep_blocks:
            if block.routed_experts_router is None:
                raise RuntimeError("DeepEP block is missing its routed router during prewarm")
            prewarm_deepep_v2_runtime(
                block,
                max_local_tokens=max_local_microbatch_size,
                hidden=self.d_model,
                top_k=block.routed_experts_router.top_k,
                device=param.device,
            )

    @torch.no_grad()
    def prewarm_ep_no_sync_symm_buffers(
        self,
        *,
        max_local_microbatch_size: int,
        pad_to_block_count: int,
        rowwise_lifetime_lease_slots: Optional[int] = None,
    ) -> None:
        ep_blocks = list(self.named_ep_no_sync_blocks())
        if not ep_blocks:
            if pad_to_block_count != 0:
                raise RuntimeError(
                    "Cannot pad EP no-sync symmetric allocations for a model part "
                    "with no EP no-sync blocks"
                )
            return

        first_block = ep_blocks[0][1]
        assert first_block.routed_experts_router is not None
        if first_block.ep_pg is None:
            raise RuntimeError("EP no-sync block is missing ep_pg during symmetric prewarm")

        param = next((p for p in self.parameters() if p.is_floating_point()), None)
        if param is None:
            raise RuntimeError("Cannot infer dtype/device for EP no-sync symmetric prewarm")
        # This prewarm runs before apply_dp(), which currently casts the model to
        # BF16 for forward/backward. Using param.dtype here therefore allocates
        # FP32 lifetime-lease slots that the BF16 rowwise runtime cannot reuse.
        # A non-PP dry run can hide the mismatch by resizing one slot, but PP can
        # retain several concurrent forward activations and needs every prewarmed
        # slot to match the eventual activation dtype.
        dtype = self._training_dtype
        device = param.device
        if first_block.routed_experts is None:
            raise RuntimeError("EP no-sync block is missing routed experts during prewarm")
        dummy_d_model = first_block.routed_experts.d_model
        prewarm_local_microbatch_size = max_local_microbatch_size
        if self.tbo:
            if max_local_microbatch_size % 2 != 0:
                raise RuntimeError(
                    "TBO EP no-sync symmetric prewarm requires an even local microbatch size "
                    f"(got {max_local_microbatch_size})"
                )
            prewarm_local_microbatch_size = max_local_microbatch_size // 2

        max_rank_capacity = 1

        for block_key, block in ep_blocks:
            assert block.routed_experts_router is not None
            if block.routed_experts is None:
                raise RuntimeError("EP no-sync block is missing routed experts during prewarm")
            routed_d_model = block.routed_experts.d_model
            top_k = block.routed_experts_router.top_k
            num_out_tokens = prewarm_local_microbatch_size * top_k
            rank_capacity = compute_ep_no_sync_rank_capacity(block, num_out_tokens)
            max_rank_capacity = max(max_rank_capacity, rank_capacity)
            if block.ep.uses_rowwise_buffers:
                if block.ep.rowwise_transport == "nvshmem" and device.type == "cuda":
                    symm_mem_vdev2d_kernels.preflight_rowwise_collective_launches(
                        block.ep.rowwise_get_nblocks,
                        block.ep.rowwise_put_nblocks,
                        block.ep.rowwise_weighted_put_nblocks,
                    )
                rowwise_fp8_cfg = block.rowwise_fp8
                use_rowwise_fp8 = (
                    rowwise_fp8_cfg is not None
                    and rowwise_fp8_cfg.enabled
                    and device.type == "cuda"
                )
                use_symm_dispatch_in = (
                    not use_rowwise_fp8
                ) and use_ep_no_sync_rowwise_symm_dispatch_in(block)
                use_symm_combine_out = (
                    not use_rowwise_fp8
                ) and use_ep_no_sync_rowwise_symm_combine_out(block)
                use_symm_combine_gather = (
                    not use_rowwise_fp8
                ) and use_ep_no_sync_rowwise_symm_combine_gather(block)
                use_fp8_symm_dispatch_in = (
                    use_rowwise_fp8 and use_ep_no_sync_rowwise_symm_dispatch_in(block)
                )
                use_fp8_symm_combine_gather = (
                    use_rowwise_fp8 and use_ep_no_sync_rowwise_symm_combine_gather(block)
                )
                block_is_checkpointed = (
                    self.recompute_all_blocks_by_chunk
                    or self.recompute_each_block
                    or (
                        self.recompute_block_keys is not None
                        and block_key in self.recompute_block_keys
                    )
                )
                block_uses_checkpointing = (
                    block_is_checkpointed
                    or block.checkpoint_attn
                    or block.checkpoint_permute_moe_unpermute
                )
                if (self.compile_enabled or self._compile_requested) and block_is_checkpointed:
                    # torch.compile() requires checkpoint context_fn entries to
                    # be TorchDispatchModes, so the compiled model-level
                    # checkpoint uses noop_context_fn. Give checkpointed rowwise
                    # blocks a static state instead of relying on Python-side
                    # context detection inside the compiled region. This keeps
                    # forward and recompute on the same tensor-producing path
                    # without installing a TorchDispatchMode in production.
                    #
                    # Leave metric accumulation dynamic so the original
                    # checkpointed forward records metrics while recompute does
                    # not. The rowwise transport metric helper uses the same
                    # best-effort recompute signal as the router metrics.
                    block._ep_no_sync_rowwise_static_checkpoint_state = (True, None)
                else:
                    block._ep_no_sync_rowwise_static_checkpoint_state = (
                        None if block_uses_checkpointing else (False, True)
                    )
                # The original checkpointed forward and its recompute both use
                # scratch buffers; no long-lived lease is needed. Force the
                # rowwise path into scratch-buffer mode for checkpointed blocks
                # so compiled checkpoint paths do not depend on Python-side
                # checkpoint-context detection.
                block._ep_no_sync_force_scratch_lifetime_buffers = block_is_checkpointed
                runtime_uses_lifetime_leases = not block._ep_no_sync_force_scratch_lifetime_buffers
                if not runtime_uses_lifetime_leases:
                    use_symm_combine_out = False
                    use_symm_combine_gather = False
                lease_dispatch_out = runtime_uses_lifetime_leases
                lease_combine_out = use_symm_combine_out and runtime_uses_lifetime_leases
                lease_combine_gather = (
                    use_symm_combine_gather or use_fp8_symm_combine_gather
                ) and runtime_uses_lifetime_leases
                slot_indices = range(block.ep.shared_slots) if self.tbo else (None,)
                for slot_idx in slot_indices:
                    get_ep_no_sync_buffers(
                        block,
                        dispatch_in_cap=num_out_tokens,
                        dispatch_out_cap=rank_capacity,
                        combine_in_cap=rank_capacity,
                        combine_out_cap=prewarm_local_microbatch_size,
                        d_model=routed_d_model,
                        dtype=dtype,
                        device=device,
                        slot_idx=slot_idx,
                        need_dispatch_in=use_symm_dispatch_in,
                        need_dispatch_meta=False,
                        need_dispatch_out=(not use_rowwise_fp8) and not lease_dispatch_out,
                        need_combine_in=not use_rowwise_fp8,
                        need_combine_meta=False,
                        need_combine_out=use_symm_combine_out and not lease_combine_out,
                        need_combine_gather=use_symm_combine_gather and not lease_combine_gather,
                        combine_gather_cap=prewarm_local_microbatch_size,
                        combine_gather_top_k=top_k,
                    )
                prewarm_ep_no_sync_rowwise_lifetime_leases(
                    block,
                    dispatch_out_cap=rank_capacity,
                    combine_out_cap=prewarm_local_microbatch_size,
                    combine_gather_cap=prewarm_local_microbatch_size,
                    combine_gather_top_k=top_k,
                    d_model=routed_d_model,
                    dtype=dtype,
                    device=device,
                    use_rowwise_fp8=use_rowwise_fp8,
                    block_size=(rowwise_fp8_cfg.block_size if rowwise_fp8_cfg is not None else 0),
                    need_dispatch_out=lease_dispatch_out,
                    need_combine_out=lease_combine_out,
                    need_combine_gather=lease_combine_gather,
                    num_slots=rowwise_lifetime_lease_slots,
                )
                if use_rowwise_fp8:
                    assert rowwise_fp8_cfg is not None
                    # Runtime support is asserted once inside get_ep_no_sync_rowwise_fp8_buffers.
                    get_ep_no_sync_rowwise_fp8_buffers(
                        block,
                        dispatch_in_cap=prewarm_local_microbatch_size,
                        dispatch_out_cap=rank_capacity,
                        combine_in_cap=rank_capacity,
                        combine_gather_cap=prewarm_local_microbatch_size,
                        combine_gather_top_k=top_k,
                        d_model=routed_d_model,
                        block_size=rowwise_fp8_cfg.block_size,
                        device=device,
                        lease_combine_gather=False,
                        need_dispatch_in=use_fp8_symm_dispatch_in,
                        need_dispatch_out=not lease_dispatch_out,
                        need_combine_gather=not lease_combine_gather,
                    )
            else:
                get_ep_no_sync_buffers(
                    block,
                    dispatch_in_cap=num_out_tokens,
                    dispatch_out_cap=rank_capacity,
                    combine_in_cap=rank_capacity,
                    combine_out_cap=num_out_tokens,
                    d_model=routed_d_model,
                    dtype=dtype,
                    device=device,
                )

        # Dummy padding keeps the symmetric allocation sequence aligned in older
        # single-world designs where different PP stages had different local MoE
        # block counts. In the current per-model-part prewarm path this should
        # usually be zero.
        pad_count = pad_to_block_count - len(ep_blocks)
        if pad_count < 0:
            raise RuntimeError(
                f"pad_to_block_count ({pad_to_block_count}) is smaller than local EP no-sync block count ({len(ep_blocks)})"
            )
        if pad_count:
            if olmo_symm_mem.is_enabled():
                symm_mem = None
            else:
                try:
                    import torch.distributed._symmetric_memory as symm_mem  # type: ignore[no-redef]
                except ImportError as e:
                    raise RuntimeError(
                        "EP no-sync requires torch.distributed._symmetric_memory"
                    ) from e
            for _ in range(pad_count):
                if olmo_symm_mem.is_enabled():
                    tensor = olmo_symm_mem.empty(
                        (max_rank_capacity, dummy_d_model),
                        dtype=dtype,
                        device=device,
                        group=first_block.ep_pg,
                    )
                    olmo_symm_mem.rendezvous(tensor, group=first_block.ep_pg)
                else:
                    assert symm_mem is not None
                    tensor = symm_mem.empty(
                        (max_rank_capacity, dummy_d_model),
                        dtype=dtype,
                        device=device,
                    )
                    symm_mem.rendezvous(tensor, group=first_block.ep_pg)
                self._ep_no_sync_dummy_symm_tensors.append(tensor)

    def apply_compile(self):
        super().apply_compile()
        self._tbo_last_step = torch.compile(self._tbo_last_step)  # type: ignore[method-assign]
        compile_forward_embed = os.getenv(
            "OLMO_COMPILE_FORWARD_EMBED", "1"
        ).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        if compile_forward_embed:
            self.forward_embed = torch.compile(self.forward_embed)  # type: ignore[method-assign]
        # self._forward_blocks = torch.compile(self._forward_blocks)

    def apply_ddp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        param_dtype: Optional[torch.dtype] = None,
        compile_enabled: bool = False,
        autograd_compile_enabled: bool = False,
    ):
        """
        Reject the retired DDP entry point. Use :meth:`apply_dp` instead.

        :raises NotImplementedError: Always, including when Python assertions are disabled.
        """
        raise NotImplementedError(
            "OLMoDDPModel.apply_ddp() is no longer supported; use apply_dp() instead."
        )

    def apply_ep(
        self,
        dp_mesh: DeviceMesh,
        ep_mesh: DeviceMesh,
        ep_mp_group: Optional[dist.ProcessGroup] = None,
        param_dtype: Optional[torch.dtype] = None,
        compile_enabled: bool = False,
        autograd_compile_enabled: bool = False,
        ep_no_sync_shared_pool: Optional[_NoSyncSymmSharedPool] = None,
    ) -> Optional[_NoSyncSymmSharedPool]:
        """
        Apply EP to the model.
        """
        self._compile_requested = bool(compile_enabled)
        self._deepep_v2_runtime_cache = {}

        shared_pool_to_return = ep_no_sync_shared_pool
        ep_no_sync_blocks: List[OLMoDDPTransformerBlock] = []
        deepep_blocks: List[OLMoDDPTransformerBlock] = []
        for block in self.routed_blocks():
            # ep_mp_group is the optional high-priority NCCL group for
            # synchronized EP collectives that should start promptly while shared
            # experts run. It is intentionally not used by no-sync blocks.
            # NVSHMEM symmetric-memory allocation bootstraps from c10d group
            # metadata. The high-priority NCCL EP group works for small count
            # collectives, but can hang inside symm_mem.empty() before the
            # explicit EP rendezvous runs. Use the regular DeviceMesh EP group
            # for no-sync blocks.
            block.apply_ep(ep_mesh, ep_pg=None if block.ep.no_sync else ep_mp_group)
            if block.ep.uses_olmo_symm:
                ep_no_sync_blocks.append(block)
            if block.ep.is_deepep:
                deepep_blocks.append(block)

        if ep_no_sync_blocks:
            slot_counts = {block.ep.shared_slots for block in ep_no_sync_blocks}
            if len(slot_counts) != 1:
                raise OLMoConfigurationError(
                    "All EP no-sync blocks must use the same EP shared_slots "
                    f"value, got {sorted(slot_counts)}"
                )
            shared_slots = next(iter(slot_counts))
            first_pg = ep_no_sync_blocks[0].ep_pg
            if first_pg is None:
                raise RuntimeError("EP no-sync block is missing ep_pg after apply_ep()")
            shared_pool = shared_pool_to_return
            if shared_pool is None:
                shared_pool = _NoSyncSymmSharedPool(
                    num_slots=shared_slots,
                    group=first_pg,
                )
            else:
                if shared_pool.num_slots != shared_slots:
                    raise RuntimeError(
                        "Cannot share EP no-sync symmetric scratch pool across model parts "
                        f"with different slot counts: pool={shared_pool.num_slots}, "
                        f"model_part={shared_slots}"
                    )
                if shared_pool.group is not first_pg:
                    raise RuntimeError(
                        "Cannot share EP no-sync symmetric scratch pool across model parts "
                        "with different EP process groups"
                    )
            shared_pool_to_return = shared_pool
            for block in ep_no_sync_blocks:
                if block.ep_pg is not first_pg:
                    raise RuntimeError(
                        "All EP no-sync blocks in a model part must share the same ep_pg "
                        f"(block={block.block_idx})"
                    )
                # Shared slots are transient scratch shared by model blocks.
                # Payloads saved for backward must use lifetime leases unless
                # checkpointing/recompute forces the rowwise path into scratch mode.
                block._ep_no_sync_shared_pool = shared_pool
                block._ep_no_sync_shared_slot = block.block_idx % shared_slots

        if deepep_blocks:
            first_pg = deepep_blocks[0].ep_pg
            if first_pg is None:
                raise RuntimeError("DeepEP block is missing ep_pg after apply_ep()")
            for block in deepep_blocks:
                if block.ep_pg is not first_pg:
                    raise RuntimeError(
                        "All DeepEP blocks in a model part must share the same ep_pg "
                        f"(block={block.block_idx})"
                    )
                block._deepep_v2_runtime_cache = self._deepep_v2_runtime_cache

        self.ep_enabled = True
        return shared_pool_to_return

    def apply_dp(
        self,
        dp_mesh: DeviceMesh,
        ep_mesh: Optional[DeviceMesh],
        dense_process_group: Optional[dist.ProcessGroup] = None,
        expert_process_group: Optional[dist.ProcessGroup] = None,
        accumulate_grads_in_fp32: bool = True,
        reduce_grads_in_fp32: bool = True,
        bucket_cap_mb: Optional[int] = None,
        use_reduce_scatter: bool = False,
        root_module: Optional[torch.nn.Module] = None,
    ):
        """
        Wrap the model in multi-group DDP.

        :param root_module: Wrap this module instead, when the model is nested inside it.
        """
        from olmo_core.nn.parallel.distributed import MultiGroupDistributedDataParallel

        module = self if root_module is None else root_module
        self._ep_modules = [
            m for m in module.modules() if getattr(m, "_ep_sharded", False)
        ]  # collect the ep sharded part based on `_ep_sharded` field (will be set to True in `apply_ep`)
        ep_sharded_params = set()
        for m in self._ep_modules:
            for n, p in m.named_parameters():
                ep_sharded_params.add(p)

        # TODO(dtype): broad bf16 casting is a current MoE V2 shortcut. Replace
        # this with explicit dtype ownership so FP8 state, optimizer main params,
        # and normal model params are not coupled to a blanket module cast.
        module.to(self._training_dtype)
        self.disable_mxfp8_expert_anchor_grads()

        dp_group = dense_process_group if dense_process_group is not None else dp_mesh.get_group()
        load_balancing_group = dp_mesh.get_group()

        for block in self.routed_blocks():
            assert block.routed_experts_router is not None
            block.routed_experts_router.set_load_balancing_process_group(load_balancing_group)

        if ep_mesh is None:
            epdp_group = dp_group
        else:
            epdp_group = (
                expert_process_group
                if expert_process_group is not None
                else ep_mesh["ep_dp"].get_group()
            )

        def param_process_group_fn(name: str, param: torch.nn.Parameter):
            # MoE params → EP_DP group
            if param in ep_sharded_params:
                return epdp_group

            # Dense params → DP group
            return dp_group

        init_sync_env = os.environ.get("OLMO_DDP_INIT_SYNC")
        init_sync = (
            True
            if init_sync_env is None
            else init_sync_env.strip().lower() in {"1", "true", "yes", "on"}
        )

        total_params = 0
        total_numel = 0
        ep_params = 0
        ep_numel = 0
        for param in module.parameters():
            total_params += 1
            total_numel += param.numel()
            if param in ep_sharded_params:
                ep_params += 1
                ep_numel += param.numel()
        dense_params = total_params - ep_params
        dense_numel = total_numel - ep_numel
        log.info(
            "Applying OLMo MultiGroup DDP "
            "(init_sync=%s, dense_pg_size=%s, expert_pg_size=%s, "
            "dense_params=%d/%d, expert_params=%d/%d, device=%s)",
            "enabled" if init_sync else "disabled",
            dist.get_world_size(dp_group),
            dist.get_world_size(epdp_group),
            dense_params,
            dense_numel,
            ep_params,
            ep_numel,
            self.device,
        )

        ddp_model = MultiGroupDistributedDataParallel(
            module=module,
            dim=0,  # for scatter/gather
            init_sync=init_sync,
            process_group=dp_group,
            param_process_group_fn=param_process_group_fn,
            accumulate_grads_in_fp32=accumulate_grads_in_fp32,
            reduce_grads_in_fp32=reduce_grads_in_fp32,
            bucket_cap_mb=bucket_cap_mb,
            use_reduce_scatter=use_reduce_scatter,
        )
        log.info("Finished applying OLMo MultiGroup DDP")

        from ..transformer.model import (
            _hide_cpu_inputs_from_torch,
            _unhide_cpu_inputs_from_torch,
        )

        ddp_model.register_forward_pre_hook(
            _hide_cpu_inputs_from_torch, prepend=True, with_kwargs=True
        )
        ddp_model.register_forward_pre_hook(
            _unhide_cpu_inputs_from_torch, prepend=False, with_kwargs=True
        )

        return ddp_model

    def prepare_experts_for_fsdp(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        raise OLMoConfigurationError(
            "prepare_experts_for_fsdp is not supported for OLMoDDPModel. "
            "Use prepare_experts_for_ddp instead."
        )

    def prepare_experts_for_ddp(self, world_mesh: DeviceMesh):
        # TODO: no-op today; confirm routed-expert blocks need no DDP-specific setup here
        # (unlike the FSDP path, which shards the experts).
        for block in self.routed_blocks():
            pass

    def post_batch(self, dry_run: bool = False):
        for block in self.ddp_blocks():
            block.post_batch(dry_run=dry_run)

    @torch.no_grad()
    def init_weights(
        self,
        *,
        max_seq_len: Optional[int] = None,
        max_local_microbatch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        world_mesh: Optional[Dict[str, Optional[DeviceMesh]]] = None,
        model_part_idx: int = 0,
    ) -> torch.Generator:
        """
        Initialize the model weights.

        :param max_seq_len: The maximum sequence length expected. This is used
            to warm up the RoPE cache.
        :param max_local_microbatch_size: The maximum local (rank) micro-batch size (in tokens)
            expected. This is used to warm-up some MoE cache.
        :param device: The device the local copy of the model will be trained on.
        :param world_mesh: The parallel meshes; required only when pipeline or expert parallelism
            is enabled (used to derive per-rank init seeds). Optional for standalone/non-parallel init.
        """
        from olmo_core.nn.attention import Attention

        device = device or self.device
        # TODO(dtype): materialization currently relies on the same broad bf16
        # cast as apply_dp(); replace with an explicit precision policy.
        self.to(self._training_dtype)
        self.to_empty(device=device)
        for _, module in self.named_modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()  # type: ignore

        seed = self.init_seed

        # adjust seed for PP, including
        # 1. PP stage
        # 2. model part within the PP stage (eg, interleaved 1F1B)
        if self.pp_enabled:
            assert world_mesh is not None and world_mesh["dense"] is not None
            seed += get_pp_mesh(world_mesh["dense"]).get_local_rank()
            seed += model_part_idx * 997  # random prime

        # adjust seed for EP_MP, different EP shards should have different init values
        # but within the same EP_DP group, they should share the same init value
        ep_generator = None
        if self.ep_enabled:
            assert world_mesh is not None and world_mesh["moe"]
            ep_mp_rank = world_mesh["moe"]["ep_mp"].get_local_rank()
            ep_seed = (
                seed + (1 + ep_mp_rank) * 653
            )  # random prime; +1 so that it will never be the same as the base seed
            ep_generator = torch.Generator(device).manual_seed(ep_seed)

        generator = torch.Generator(device).manual_seed(seed)

        if self.embeddings is not None:
            self.init_method.init_embeddings(
                self.embeddings,
                d_model=self.d_model,
                embed_scale=self.embed_scale,
                std=self.embedding_init_std
                if self.embedding_init_std is not None
                else self.init_std,
                generator=generator,
            )

        for _, block in self.named_required_ddp_blocks():
            # This might fail if it's wrapped.
            # v2 MoE/shared-only DDP blocks.
            att = cast(Attention, block.attention)

            # Attention weights.
            self.init_method.init_attention(
                att,
                d_model=self.d_model,
                block_idx=block.block_idx,
                num_blocks=self.n_layers,
                std=self.init_std,
                generator=generator,
            )
            # MoE/shared-expert weights.
            self.init_method.init_moe_v2(
                block,
                d_model=self.d_model,
                block_idx=block.block_idx,
                num_blocks=self.n_layers,
                std=self.init_std,
                generator=generator,
                ep_generator=ep_generator,
            )

            # Warm up RoPE cache for attention-like sequence mixers. Recurrent
            # mixers such as GDN do not have RoPE.
            rope = getattr(att, "rope", None)
            if max_seq_len is not None and rope is not None:
                rope.warmup_cache(max_seq_len, device)

        if self.lm_head is not None:
            self.init_method.init_final_w_out(
                self.lm_head.w_out, d_model=self.d_model, std=self.init_std, generator=generator
            )

        # `to_empty()` above allocated fresh storage for the embedding and LM-head params, breaking
        # any weight tie; re-establish it so a tied model doesn't silently train/save an independent
        # output matrix.
        if self.tie_word_embeddings:
            self._tie_weights()

        return generator

    def attach_fp32_accum(self):
        self.has_grad_accum_fp32_buffer = True
        for p in self.parameters():
            if not p.requires_grad:
                continue
            # persistent FP32 master grad on the same device
            p._main_grad_fp32 = None  # type: ignore[attr-defined]

            p.register_post_accumulate_grad_hook(_fp32_post_grad_acc_hook)

    def set_main_grads_to_none(self):
        for p in self.parameters():
            if not p.requires_grad:
                continue
            if hasattr(p, "_main_grad_fp32"):
                p._main_grad_fp32 = None  # type: ignore[attr-defined]
        for _, weight in self.named_fp8_weight_stores():
            weight.set_main_grad_to_none()  # type: ignore[attr-defined]

    def forward_tbo(
        self,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        **kwargs,
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        assert self.ep_enabled, "TBO requires EP to be enabled."

        (
            input_ids,
            labels,
            all_block_kwargs,
            per_block_kwargs,
            lm_head_kwargs,
        ) = self._prepare_inputs(
            input_ids,
            labels,
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
            z_loss_multiplier=z_loss_multiplier,
            loss_div_factor=loss_div_factor,
            return_logits=return_logits,
            **kwargs,
        )
        assert input_ids.size(0) % 2 == 0, "When TBO is enabled, the batch size must be even."

        # Get embeddings but pass-through for non-existent layers to allow easy
        # pipeline parallel configuration.
        h = self.embeddings(input_ids) if self.embeddings is not None else input_ids
        if self.embeddings is not None and self.embed_scale is not None:
            h = h * self.embed_scale
        if self.embedding_norm is not None:
            h = self.embedding_norm(h)

        # can only check this in every forward.
        # We cannot put this in __init__ because of PP might change model layers.
        first_moe_idx = self._check_tbo_requirements()

        # forward dense blocks
        for block_idx, (block_key, block) in enumerate(self.blocks.items()):
            is_routed_block = (
                isinstance(block, OLMoDDPTransformerBlock) and block.has_routed_experts
            )
            if is_routed_block:
                assert (
                    first_moe_idx == block_idx
                ), f"first_moe_idx {first_moe_idx}, block_idx {block_idx}, block.block_idx {block.block_idx}"
                break
            # Mark sizes as dynamic for torch.compile().
            if self.compile_enabled:
                mark_dynamic(h, (0, 1), strict=False)
            with nvtx.annotate(f"fwd_block_{block_idx}", color="blue"):
                one_block_kwargs = per_block_kwargs.get(int(block_key), {})
                if self.checkpoint_tbo_dense_layers:
                    h = checkpoint(
                        block,
                        h,
                        use_reentrant=False,
                        **one_block_kwargs,
                    )
                    h = cast(torch.Tensor, h)
                else:
                    h = block(h, **all_block_kwargs, **one_block_kwargs)

        # forward moe blocks with TBO
        x0, x1 = h.chunk(2, dim=0)  # assume even batch size
        labels0, labels1 = None, None
        if labels is not None:
            labels0, labels1 = labels.chunk(2, dim=0)
            del labels
        del h
        # Mark sizes as dynamic for torch.compile().
        if self.compile_enabled:
            mark_dynamic(x0, (0, 1), strict=False)
            mark_dynamic(x1, (0, 1), strict=False)
        x1_is_fresh = True  # x1 is always fresh in the beginning
        x1_ctx: object = {
            "x1": x1,
        }
        for block_key, block in self.named_routed_blocks():
            with nvtx.annotate(f"fwd_block_{block_key}", color="blue"):
                # Thread the block's per-block kwargs (e.g. context-parallel RoPE shards) alongside
                # the shared kwargs, matching the dense/non-TBO loops; otherwise routed blocks under
                # CP + TBO would miss their pos_sin/pos_cos/freqs_cis shards.
                one_block_kwargs = per_block_kwargs.get(int(block_key), {})
                x0, x1_ctx = block.combined_forward_rowwise_nvshmem_tbo(
                    x0, x1_ctx, x1_is_fresh, **all_block_kwargs, **one_block_kwargs
                )
                x1_is_fresh = False  # after the first TBO block, x1 is no longer fresh

        # finish x1 last steps
        h0, h1 = self._tbo_last_step(x0, x1_ctx, lm_head_kwargs, labels0, labels1)

        return self._merge_tbo_outputs(h0, h1, loss_reduction=loss_reduction)

    # @torch.compile
    def _tbo_last_step(
        self,
        x0,
        x1_ctx: object,
        lm_head_kwargs: Dict[str, Any],
        labels0: Optional[torch.Tensor],
        labels1: Optional[torch.Tensor],
    ):
        if isinstance(x1_ctx, _NoSyncRowwiseTboPendingContext):
            with nvtx.annotate("TBO-1", color="orange"):
                pending_ctx = ep_no_sync_rowwise_tbo_stage_c_launch(x1_ctx.block, x1_ctx)

            h0 = self.maybe_forward_lm_head(x0, lm_head_kwargs, labels=labels0)

            with nvtx.annotate("TBO-1", color="orange"):
                x1 = ep_no_sync_rowwise_tbo_stage_tail(x1_ctx.block, pending_ctx)

            h1 = self.maybe_forward_lm_head(x1, lm_head_kwargs, labels=labels1)
            return h0, h1

        raise RuntimeError(
            "Expected rowwise NVSHMEM TBO pending context for the final TBO step, "
            f"got type={type(x1_ctx)}"
        )

    def _merge_tbo_outputs(
        self,
        h0: Union[torch.Tensor, LMOutputWithLoss],
        h1: Union[torch.Tensor, LMOutputWithLoss],
        *,
        loss_reduction: Literal["mean", "sum", "none"],
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        if isinstance(h0, torch.Tensor):
            if not isinstance(h1, torch.Tensor):
                raise RuntimeError(f"TBO output type mismatch: {type(h0)} vs {type(h1)}")
            return torch.cat([h0, h1], dim=0)

        if not isinstance(h1, LMOutputWithLoss):
            raise RuntimeError(f"TBO output type mismatch: {type(h0)} vs {type(h1)}")

        logits: Optional[torch.Tensor]
        if h0.logits is None:
            logits = None
        else:
            if h1.logits is None:
                raise RuntimeError(
                    "TBO output type mismatch: first half has logits, second half does not"
                )
            logits = torch.cat([h0.logits, h1.logits], dim=0)

        def merge_loss(
            lhs: Optional[torch.Tensor],
            rhs: Optional[torch.Tensor],
        ) -> Optional[torch.Tensor]:
            if lhs is None:
                if rhs is not None:
                    raise RuntimeError(
                        "TBO output type mismatch: first half loss is None, second half is not"
                    )
                return None
            if rhs is None:
                raise RuntimeError(
                    "TBO output type mismatch: first half loss is set, second half is None"
                )
            if loss_reduction == "none":
                return torch.cat([lhs, rhs], dim=0)
            if loss_reduction == "mean":
                return (lhs + rhs) * 0.5
            if loss_reduction == "sum":
                return lhs + rhs
            raise NotImplementedError(loss_reduction)

        loss = merge_loss(h0.loss, h1.loss)
        ce_loss = merge_loss(h0.ce_loss, h1.ce_loss)
        z_loss = merge_loss(h0.z_loss, h1.z_loss)
        assert loss is not None
        assert ce_loss is not None
        return LMOutputWithLoss(
            logits=logits,
            loss=loss,
            ce_loss=ce_loss,
            z_loss=z_loss,
        )

    def maybe_forward_lm_head(
        self,
        x: torch.Tensor,
        lm_head_kwargs: Dict[str, Any],
        labels: Optional[torch.Tensor] = None,
    ):
        if self.lm_head is not None:
            if self.compile_enabled:
                mark_dynamic(x, (0, 1), strict=False)
                if labels is not None:
                    mark_dynamic(labels, (0, 1), strict=False)
            # NOTE: When TP is active we can't pass 'labels=None' or the hook from 'PrepareModuleInput'
            # will throw an exception.
            if labels is not None:
                lm_head_kwargs["labels"] = labels
            h0 = self.lm_head(x, **lm_head_kwargs)
        else:
            h0 = x
        return h0

    def _forward_blocks(
        self, h, all_block_kwargs: Dict[str, Any], per_block_kwargs: Dict[int, Dict[str, Any]]
    ) -> torch.Tensor:
        # Run each block.
        for block_idx, (block_key, block) in enumerate(self.blocks.items()):
            # Mark sizes as dynamic for torch.compile().
            if self.compile_enabled:
                mark_dynamic(h, (0, 1), strict=False)
            block_kwargs = per_block_kwargs.get(int(block_key), {})
            combined_kwargs = {**all_block_kwargs, **block_kwargs}
            do_not_recompute: List[int] = []
            if (self.recompute_each_block and (block_idx not in do_not_recompute)) or (
                self.recompute_block_keys and block_key in self.recompute_block_keys
            ):
                use_reentrant = (
                    isinstance(block, OLMoDDPTransformerBlock)
                    and block.has_routed_experts
                    and block.ep_enabled
                    and block.training
                    and block.ep.is_deepep
                )
                if use_reentrant:
                    # DeepEP's custom autograd function builds a private expert
                    # graph and invokes its backward from the communication
                    # backward. Non-reentrant checkpointing restores its saved
                    # leaf tensors as different wrappers, so gradients land on
                    # a different recv_x leaf. Reentrant checkpointing rebuilds
                    # the complete block and keeps the DeepEP handle, expert
                    # graph, and recv_x leaf from the same recomputed forward.
                    combined_kwargs["deepep_reentrant_checkpoint"] = True
                    # Reentrant checkpointing only reruns the block (and backprops
                    # the DeepEP expert graph) if an input requires grad. In
                    # expert-only / frozen-lower-layer runs h can be detached, which
                    # would silently drop expert grads, so pass a grad-requiring
                    # anchor; expert grads still flow through the module params.
                    grad_anchor = (
                        None
                        if h.requires_grad
                        else torch.zeros((), device=h.device, requires_grad=True)
                    )
                    h = checkpoint(
                        self._forwrad_one_block,
                        h,
                        block_key,
                        combined_kwargs,
                        grad_anchor,
                        use_reentrant=True,
                        # The user config can randomize expert assignment. The
                        # recompute must replay the same routing decisions.
                        preserve_rng_state=True,
                    )
                else:
                    h = checkpoint(
                        self._forwrad_one_block,
                        h,
                        block_key,
                        combined_kwargs,
                        use_reentrant=False,
                        context_fn=(
                            noop_context_fn if self.compile_enabled else recompute_context_fn
                        ),
                        # determinism_check='none',
                    )
                h = cast(torch.Tensor, h)
            else:
                h = self._forwrad_one_block(h, block_key, combined_kwargs)

        return h

    def _forwrad_one_block(
        self,
        h,
        block_key: str,
        block_kwargs: Dict[str, Any],
        grad_anchor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # grad_anchor only keeps reentrant checkpointing's recompute alive when h is
        # detached (see _forward_blocks); it is intentionally unused here.
        del grad_anchor
        if self.compile_enabled:
            mark_dynamic(h, (0, 1), strict=False)
        block = self.blocks[block_key]
        with nvtx.annotate(f"fwd_block_{block_key}", color="blue"):
            h = block(h, **block_kwargs)
        return h

    def forward_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Get embeddings but pass-through for non-existent layers to allow easy
        # pipeline parallel configuration.
        h = self.embeddings(input_ids) if self.embeddings is not None else input_ids
        if self.embeddings is not None and self.embed_scale is not None:
            h = h * self.embed_scale
        if self.embedding_norm is not None:
            assert (
                self.embeddings is not None
            )  # PP does not have embedding, should not have embedding norm either
            h = self.embedding_norm(h)
        return h

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        input_embeddings: Optional[torch.Tensor] = None,
        router_loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        """
        Run the transformer on the token input IDs.

        :param input_ids: The token input IDs, shape ``(batch_size, seq_len)``.
        :param input_embeddings: Used in place of ``forward_embed(input_ids)`` when given.
        :param router_loss_div_factor: Divisor for the router losses, if not ``loss_div_factor``.

        :returns: The logits if ``labels`` is ``None`` or the losses if ``labels`` is not ``None``.
        """
        if self.tbo:
            if input_embeddings is not None or router_loss_div_factor is not None:
                raise NotImplementedError(
                    "input_embeddings and router_loss_div_factor are not supported with "
                    "two-batch overlap"
                )
            return self.forward_tbo(
                input_ids,
                labels=labels,
                ignore_index=ignore_index,
                loss_reduction=loss_reduction,
                z_loss_multiplier=z_loss_multiplier,
                loss_div_factor=loss_div_factor,
                return_logits=return_logits,
                **kwargs,
            )

        (
            input_ids,
            labels,
            all_block_kwargs,
            per_block_kwargs,
            lm_head_kwargs,
        ) = self._prepare_inputs(
            input_ids,
            labels,
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
            z_loss_multiplier=z_loss_multiplier,
            loss_div_factor=loss_div_factor,
            return_logits=return_logits,
            **kwargs,
        )
        if router_loss_div_factor is not None:
            all_block_kwargs["loss_div_factor"] = move_to_device(
                router_loss_div_factor, self.device
            )

        if input_embeddings is not None:
            if self._cp_load_balancer is not None:
                raise NotImplementedError(
                    "input_embeddings are not supported with context parallelism"
                )
            h = move_to_device(input_embeddings, self.device)
        else:
            h = self.forward_embed(input_ids)

        if self.recompute_all_blocks_by_chunk:
            h = checkpoint(
                self._forward_blocks,
                h,
                all_block_kwargs,
                per_block_kwargs,
                use_reentrant=False,
                context_fn=(noop_context_fn if self.compile_enabled else recompute_context_fn),
            )
            h = cast(torch.Tensor, h)
        else:
            h = self._forward_blocks(h, all_block_kwargs, per_block_kwargs)

        # Get final logits but again pass-through in case of pipeline parallelism.
        if self.lm_head is not None:
            if self.compile_enabled:
                mark_dynamic(h, (0, 1), strict=False)
                if labels is not None:
                    mark_dynamic(labels, (0, 1), strict=False)
            # NOTE: When TP is active we can't pass 'labels=None' or the hook from 'PrepareModuleInput'
            # will throw an exception.
            if labels is not None:
                lm_head_kwargs["labels"] = labels

            with nvtx.annotate("lm_head", color="purple"):
                out = self.lm_head(h, **lm_head_kwargs)

            # check for nan
            # if torch.isnan(out.loss).any():
            #     print('nan')

        else:
            out = h
        return out


__all__ = ["OLMoDDPModel"]


def _hide_cpu_inputs_from_torch(m, args, kwargs) -> Optional[Tuple[Any, Dict[str, Any]]]:
    del m
    # Keep CPU-only metadata out of wrappers/hooks that assume every tensor kwarg
    # belongs on the module device. This came from composable-DDP/FSDP-era
    # behavior; revalidate under MultiGroupDDP before removing it.
    if (doc_lens := kwargs.get("doc_lens")) is not None:
        kwargs["doc_lens"] = hide_from_torch(doc_lens)
    return (args, kwargs)


def _unhide_cpu_inputs_from_torch(m, args, kwargs) -> Optional[Tuple[Any, Dict[str, Any]]]:
    del m
    if (doc_lens := kwargs.get("doc_lens")) is not None:
        kwargs["doc_lens"] = unhide_from_torch(doc_lens)
    return (args, kwargs)
