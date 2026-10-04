"""Qualify rounded wgrad with rowwise EP buffers and the real distributed optimizer."""

import os
from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import unhide_from_torch
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe.emo import EmoRouterConfig
from olmo_core.nn.moe.v2.ep_config import ExpertParallelConfig, ExpertParallelPath
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.transformer import (
    OLMoDDPModelConfig,
    TransformerBlockType,
    TransformerType,
)
from olmo_core.optim import OLMoDDPOptimizerConfig
from olmo_core.testing import run_distributed_test
from olmo_core.train.train_module import OLMoDDPTrainModuleConfig
from olmo_core.train.train_module.transformer import (
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)


def _build_model(width, hidden, num_experts=512, ep_path=ExpertParallelPath.rowwise_nvshmem):
    norm = LayerNormConfig(name=LayerNormType.rms, bias=False, dtype=DType.float32)
    return OLMoDDPModelConfig(
        init_seed=12536,
        d_model=width,
        n_layers=1,
        vocab_size=256,
        name=TransformerType.moe_fused_v2,
        recompute_each_block=False,
        block=OLMoDDPTransformerBlockConfig(
            name=TransformerBlockType.moe_fused_v2,
            attention=AttentionConfig(
                name=AttentionType.default,
                n_heads=width // 128,
                bias=False,
                use_flash=False,
                dtype=DType.bfloat16,
            ),
            layer_norm=norm,
            shared_experts=None,
            routed_experts=RoutedExpertsConfig(
                d_model=width,
                hidden_size=hidden,
                num_experts=num_experts,
                dtype=DType.bfloat16,
                bias=False,
            ),
            routed_experts_router=MoERouterConfigV2(
                d_model=width,
                num_experts=num_experts,
                top_k=16,
                dtype=DType.float32,
                lb_loss_weight=0.01,
                z_loss_weight=1e-5,
                global_load_balancing=True,
                emo=EmoRouterConfig(
                    eos_token_id=0,
                    min_document_expert_pool=16,
                    max_document_expert_pool=num_experts,
                ),
            ),
            ep=ExpertParallelConfig(
                path=ep_path,
                # Dropless correctness fixture; production timing retains1.25.
                capacity_factor=8.0,
                share_dispatch_out=False,
                share_combine_out=False,
            ),
        ),
        lm_head=LMHeadConfig(layer_norm=norm, bias=False, dtype=DType.bfloat16),
    ).build(init_device="meta")


def _run_ep_parity(ep_degree, width, hidden, compiled):
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    for key in (
        "OLMO_PROFILE_FP32_GRAD_ADD_VECTORIZE",
        "OLMO_PROFILE_SWIGLU_PAIRWISE",
        "OLMO_PROFILE_EMO_DOCUMENT_POOL",
        "OLMO_PROFILE_EMO_TOP16",
        "OLMO_PROFILE_RS_SINGLE_PARAM_FAST_PATH",
        "OLMO_PROFILE_ROUNDED_WGRAD_EP",
        "OLMO_EP_NO_SYNC_FORBID_RUNTIME_SYMM_ALLOC",
    ):
        os.environ[key] = "1"
    for key in (
        "OLMO_PROFILE_DDP_DEFER_REPLICATED_REDUCTIONS",
        "OLMO_PROFILE_LB_COUNT_OVERLAP",
        "OLMO_PROFILE_LB_COUNT_BATCHED",
    ):
        os.environ[key] = "0"
    stacks = []
    for enabled in (False, True):
        os.environ["OLMO_PROFILE_ROUNDED_WGRAD"] = str(int(enabled))
        config = OLMoDDPTrainModuleConfig(
            rank_microbatch_size=256,
            max_sequence_length=256,
            compile_model=compiled,
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.ddp,
                accumulate_grads_in_fp32=True,
                reduce_grads_in_fp32=True,
                use_reduce_scatter=True,
                bucket_cap_mb=1,
            ),
            ep_config=TransformerExpertParallelConfig(degree=ep_degree),
            optim=OLMoDDPOptimizerConfig(
                lr=1e-3,
                betas=(0.9, 0.95),
                max_grad_norm=1.0,
                dtype=DType.float32,
                use_distributed=True,
                compile=compiled,
            ),
        )
        tm = config.build(_build_model(width, hidden), device=device)
        assert dist.get_world_size(tm.ep_mp_group) == ep_degree
        assert dist.get_world_size(tm.ep_dp_group) == 8 // ep_degree
        assert dist.get_world_size(tm.dp_group) == 8
        assert tm.model.module.blocks["0"].routed_experts.num_local_experts == 512 // ep_degree
        stacks.append(tm)

    def compare_parameters_and_states():
        ref, new = stacks
        assert ref.optim is not None and new.optim is not None
        ref_params, new_params = dict(ref.model.named_parameters()), dict(
            new.model.named_parameters()
        )
        assert ref_params.keys() == new_params.keys()
        for name in ref_params:
            torch.testing.assert_close(
                new_params[name], ref_params[name], rtol=2e-5, atol=1e-7, msg=name
            )
        assert ref.optim.states.keys() == new.optim.states.keys()
        for name in ref.optim.states:
            torch.testing.assert_close(
                new.optim.states[name].to_local(),
                ref.optim.states[name].to_local(),
                rtol=2e-5,
                atol=1e-7,
                msg=name,
            )

    compare_parameters_and_states()
    for step in range(3):
        torch.manual_seed(913 + step + rank)
        batches = [torch.randint(1, 256, (1, 256), device=device) for _ in range(8)]
        for batch in batches:
            batch[:, 31::32] = 0
        losses = []
        for tm in stacks:
            values = []
            torch.manual_seed(781 + step + rank)
            for micro, batch in enumerate(batches):
                with tm.model.no_sync() if micro < 7 else nullcontext():
                    out = tm.model(
                        batch,
                        labels=batch.roll(-1, 1),
                        loss_reduction="sum",
                        loss_div_factor=256.0 * 8,
                        z_loss_multiplier=1e-5,
                    )
                    out.loss.backward()
                    values.append(out.loss.detach())
            tm.model.finalize_grad_reduce()
            losses.append(torch.stack(values))
            for block in tm.model.module.routed_blocks():
                dropped = block._ep_no_sync_rowwise_drop_tokens_sum
                assert dropped is not None
                assert unhide_from_torch(dropped).item() == 0
        torch.testing.assert_close(losses[1], losses[0], rtol=2e-5, atol=1e-7)
        for (name, ref), (_, new) in zip(
            stacks[0].model.named_parameters(), stacks[1].model.named_parameters()
        ):
            ref_grad = getattr(ref, "_olmo_ddp_reduced_grad_shard", ref._main_grad_fp32)
            new_grad = getattr(new, "_olmo_ddp_reduced_grad_shard", new._main_grad_fp32)
            torch.testing.assert_close(new_grad, ref_grad, rtol=2e-5, atol=1e-7, msg=name)
        for tm in stacks:
            assert tm.optim is not None
            tm.optim.step()
            assert tm.optim._step_skipped is not None
            assert not bool(tm.optim._step_skipped.item())
        compare_parameters_and_states()
        for tm in stacks:
            tm.model.zero_grad(set_to_none=(step == 1))
            for block in tm.model.module.routed_blocks():
                block.routed_experts_router.reset_metrics()
        if rank == 0:
            print(
                f"EP_PARITY ep={ep_degree} width={width} hidden={hidden} compiled={compiled} update={step+1}: losses, gradients, parameters and all Adam states pass",
                flush=True,
            )


@pytest.mark.gpu
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    "ep_degree,width,hidden", [(2, 512, 1024), (4, 512, 1024), (8, 512, 1024), (8, 768, 1536)]
)
def test_rowwise_rounded_wgrad_and_sharded_adam(ep_degree, width, hidden, compiled):
    """Use EP2/4 replicated shards and EP8, covering small and medium/large expert dimensions."""
    if torch.cuda.device_count() < 8:
        pytest.skip("requires8 CUDA GPUs")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("rounded weight-gradient kernel requires Blackwell")
    run_distributed_test(
        _run_ep_parity,
        world_size=8,
        backend="nccl",
        start_method="spawn",
        func_args=(ep_degree, width, hidden, compiled),
    )
