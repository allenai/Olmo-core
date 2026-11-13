"""Exact EP1 checkpoint -> EP2/4/8 restore and synchronous-save state retention."""

import os
from pathlib import Path
from test.nn.moe.v2.rounded_wgrad_ep_test import _build_model

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import Shard

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath
from olmo_core.optim import OLMoDDPOptimizerConfig
from olmo_core.testing import run_distributed_test
from olmo_core.train.train_module import OLMoDDPTrainModuleConfig
from olmo_core.train.train_module.transformer import (
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)


def _run_reshard(save_root, ep_degree, save_options):
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    for key in (
        "OLMO_PROFILE_ROUNDED_WGRAD",
        "OLMO_PROFILE_LB_COUNT_BATCHED",
        "OLMO_PROFILE_LB_COUNT_OVERLAP",
        "OLMO_PROFILE_DDP_DEFER_REPLICATED_REDUCTIONS",
    ):
        os.environ[key] = "0"
    os.environ["OLMO_EP_NO_SYNC_FORBID_RUNTIME_SYMM_ALLOC"] = "1"

    def build(ep):
        config = OLMoDDPTrainModuleConfig(
            rank_microbatch_size=32,
            max_sequence_length=32,
            compile_model=False,
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.ddp,
                accumulate_grads_in_fp32=True,
                reduce_grads_in_fp32=True,
                use_reduce_scatter=True,
            ),
            ep_config=TransformerExpertParallelConfig(degree=ep) if ep > 1 else None,
            optim=OLMoDDPOptimizerConfig(
                lr=1e-3,
                dtype=DType.float32,
                use_distributed=True,
                compile=False,
            ),
        )
        return config.build(
            _build_model(512, 512, num_experts=16, ep_path=ExpertParallelPath.sync_1d),
            device=torch.device("cuda", rank),
        )

    original = build(1)
    expected = {}
    with torch.no_grad():
        for index, (name, state) in enumerate(sorted(original.optim.states.items())):
            is_sharded = any(isinstance(p, Shard) for p in state.placements)
            # Distinct per-state AND per-shard values catch wrong EP slice ordering.
            value = (
                7.0
                if name.endswith(".step")
                else (index + 1 + (rank / 16 if is_sharded else 0)) * 0.001
            )
            state.to_local().fill_(value)
            expected[name] = state.full_tensor().cpu()
    original.optim._copy_main_params_to_model_params()
    original.save_state_dict_direct(Path(save_root) / "ep1", **save_options)
    for name, state in original.optim.states.items():
        torch.testing.assert_close(
            state.full_tensor().cpu(), expected[name], rtol=0, atol=0, msg=name
        )

    # Each target EP degree is parametrized into a fresh distributed process set.
    # The checkpoint layout is independent of the EP communication kernel.
    for ep in (ep_degree,):
        tm = build(ep)
        tm.load_state_dict_direct(
            Path(save_root) / "ep1",
            load_optim_state=True,
            reset_optimizer_states_on_load=False,
            constant_memory_planning=save_options.get("constant_memory_planning", False),
        )
        expert_names = {
            name
            for group in tm.optim.param_groups
            if group["pg"] == "ep_dp"
            for name in group["named_params"]
        }
        ep_rank = dist.get_rank(tm.ep_mp_group)
        assert tm.optim.states.keys() == expected.keys()
        saved_locals = {}
        for name, state in tm.optim.states.items():
            reference = expected[name]
            if name.rsplit(".", 1)[0] in expert_names and not name.endswith(".step"):
                reference = reference.chunk(ep, dim=0)[ep_rank]
            torch.testing.assert_close(
                state.full_tensor().cpu(), reference, rtol=0, atol=0, msg=name
            )
            saved_locals[name] = state.to_local().clone()
        for name, param in tm.model.named_parameters():
            master = tm.optim.states[name + ".main"].full_tensor().reshape_as(param)
            torch.testing.assert_close(param, master.to(param.dtype), rtol=0, atol=0, msg=name)

        # The direct synchronous save temporarily changes EP checkpoint views and
        # then reloads live optimizer state. Check that all local shards survive.
        tm.save_state_dict_direct(Path(save_root) / f"ep{ep}", **save_options)
        tm.zero_grads()
        for name, state in tm.optim.states.items():
            torch.testing.assert_close(
                state.to_local(), saved_locals[name], rtol=0, atol=0, msg=name
            )


@pytest.mark.gpu
@pytest.mark.parametrize("ep_degree", [2, 4, 8])
@pytest.mark.parametrize(
    "save_options",
    [
        {},
        {"compact_storage": True, "dedup_save_to_lowest_rank": False},
        {
            "compact_storage": True,
            "dedup_save_to_lowest_rank": False,
            "constant_memory_planning": True,
        },
    ],
    ids=["legacy", "balanced-compact", "constant-memory"],
)
def test_ep1_checkpoint_reshards_and_preserves_live_states(tmp_path, ep_degree, save_options):
    """Use node-local temporary checkpoints; never touch Weka or user checkpoints."""
    if torch.cuda.device_count() < ep_degree:
        pytest.skip(f"requires {ep_degree} CUDA GPUs")
    # Preserve the original eight-rank DP/EP coverage when a full node is available.
    world_size = next(size for size in (8, 4, 2) if torch.cuda.device_count() >= size)
    run_distributed_test(
        _run_reshard,
        world_size=world_size,
        backend="nccl",
        start_method="spawn",
        func_args=(str(tmp_path / "ep-reshard"), ep_degree, save_options),
    )
