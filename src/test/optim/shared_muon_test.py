import copy

import pytest
import torch
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from olmo_core.optim.moe_optimizer import OLMoDDPOptimizer
from olmo_core.optim.shared_muon import SharedMuon, SharedMuonConfig
from olmo_core.testing.distributed import run_distributed_test


def _run_parity():
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))
    for shape in [(8, 4), (4, 8), (9, 4)]:
        for placement in [Replicate(), Shard(0), Shard(1)]:
            torch.manual_seed(19)
            initial = torch.randn(shape)
            param = torch.nn.Parameter(distribute_tensor(initial.clone(), mesh, [placement]))
            dense = SharedMuon([param], lr=0.003)
            moe = object.__new__(OLMoDDPOptimizer)
            moe.muon_momentum, moe.muon_rms, moe.muon_layouts = 0.95, 0.2, {}
            moe.states = {
                "weight.main": distribute_tensor(initial.flatten().clone(), mesh, [Shard(0)]),
                "weight.exp_avg": distribute_tensor(torch.zeros(initial.numel()), mesh, [Shard(0)]),
                "weight.step": distribute_tensor(torch.zeros(()), mesh, [Replicate()]),
            }
            for factor in [1.0, 1.0, 0.0, 1.0]:
                grad = torch.randn(shape)
                param.grad = distribute_tensor(grad, mesh, [placement])
                moe.main_grad = {
                    "weight": distribute_tensor(grad.flatten(), mesh, [Shard(0)]).to_local()
                }
                dense.get_step_factor = lambda: torch.tensor(factor)
                dense.step()
                with torch.no_grad():
                    moe._step_muon(
                        "weight", shape, {"lr": 0.003, "weight_decay": 0.1}, torch.tensor(factor)
                    )
                torch.testing.assert_close(
                    param.full_tensor().flatten(),
                    moe.states["weight.main"].full_tensor(),
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(
                    dense.state[param]["momentum"].full_tensor().flatten(),
                    moe.states["weight.exp_avg"].full_tensor(),
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(
                    dense.state[param]["step"],
                    moe.states["weight.step"].full_tensor(),
                    atol=0,
                    rtol=0,
                )


def test_dtensor_adapter_matches_flat_shards():
    run_distributed_test(_run_parity, world_size=2, backend="gloo", start_method="spawn")


@pytest.mark.parametrize("algorithm,shape", [("muon", (8, 4)), ("adamw", (8,))])
def test_skip_policy_and_checkpoint_resume(algorithm, shape):
    torch.manual_seed(7)
    param = torch.nn.Parameter(torch.randn(shape))
    optim = SharedMuon([dict(params=[param], algorithm=algorithm)], rolling_interval_length=6)
    for loss in [1.0, 1.0]:
        optim.latest_loss = torch.tensor(loss)
        optim.latest_grad_norm = torch.tensor(1.0)
        param.grad = torch.ones_like(param)
        optim.step()
    saved = copy.deepcopy(optim.state_dict())
    resumed_param = torch.nn.Parameter(param.detach().clone())
    resumed = SharedMuon(
        [dict(params=[resumed_param], algorithm=algorithm)], rolling_interval_length=6
    )
    resumed.load_state_dict(saved)
    before = param.detach().clone()
    states = {k: v.clone() for k, v in optim.state[param].items()}
    for opt, p in [(optim, param), (resumed, resumed_param)]:
        opt.latest_loss = torch.tensor(100.0)
        opt.latest_grad_norm = torch.tensor(100.0)
        p.grad = torch.randn_like(p)
        assert opt.get_step_factor() == 0
        # The MoE adapter delegates to this same policy, tested against its entrypoint.
        assert OLMoDDPOptimizer.get_step_factor(opt) == 0
        opt.step()
        torch.testing.assert_close(p, before, atol=0, rtol=0)
        for k, v in states.items():
            torch.testing.assert_close(opt.state[p][k], v, atol=0, rtol=0)


def test_routing_covers_embedding_norm_and_convolutions():
    model = torch.nn.Module()
    model.embeddings = torch.nn.Embedding(16, 8)
    model.embedding_norm = torch.nn.LayerNorm(8)
    model.projection = torch.nn.Linear(8, 8)
    model.w_b = torch.nn.Linear(8, 2, bias=False)
    model.conv = torch.nn.Conv1d(8, 8, 4)
    model.lm_head = torch.nn.Linear(8, 16)
    groups = SharedMuonConfig().build_groups(model)
    routed = {id(p): g for g in groups for p in g["params"]}
    assert len(routed) == len(list(model.parameters()))
    for name, param in model.named_parameters():
        group = routed[id(param)]
        assert group["algorithm"] == ("muon" if name == "projection.weight" else "adamw")
        assert group["weight_decay"] == (0.0 if name == "embeddings.weight" else 0.1)
    assert isinstance(SharedMuonConfig().build(model), SharedMuon)


def test_dcp_preserves_skip_history_and_updates():
    model = torch.nn.Linear(4, 8, bias=False)
    original = SharedMuon(model.parameters(), rolling_interval_length=6)
    for _ in range(2):
        original.latest_loss = torch.tensor(1.0)
        original.latest_grad_norm = torch.tensor(1.0)
        model.weight.grad = torch.ones_like(model.weight)
        original.step()
    checkpoint = copy.deepcopy(get_optimizer_state_dict(model, original))
    restored_model = copy.deepcopy(model)
    restored = SharedMuon(restored_model.parameters(), rolling_interval_length=6)
    set_optimizer_state_dict(restored_model, restored, checkpoint)
    for loss in [100.0, 1.0]:
        for net, opt in [(model, original), (restored_model, restored)]:
            opt.latest_loss = torch.tensor(loss)
            opt.latest_grad_norm = torch.tensor(loss)
            net.weight.grad = torch.ones_like(net.weight)
            opt.step()
        torch.testing.assert_close(restored_model.weight, model.weight, atol=0, rtol=0)
        torch.testing.assert_close(restored.step_skipped, original.step_skipped, atol=0, rtol=0)
