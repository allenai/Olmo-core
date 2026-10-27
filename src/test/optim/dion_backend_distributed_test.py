"""Two-GPU production-backend validation: native Dion, dense, MoE layouts and skips."""
import copy
import json

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from olmo_core.optim.dion_backend import logical_matrix_views
from olmo_core.optim.moe_optimizer import OLMoDDPOptimizer
from olmo_core.optim.muon_math import MuonLayout
from olmo_core.optim.shared_muon import SharedMuon
from olmo_core.testing import requires_multi_gpu, run_distributed_test


def _run_dion_adapters():
    from dion import Muon

    device = "cuda"
    mesh = init_device_mesh(device, (dist.get_world_size(),), mesh_dim_names=("dp",))
    cases = [
        ("weight", (64, 32), None),
        ("weight", (65, 32), None),
        ("routed_experts_router.weight", (544 * 272,), MuonLayout((544, 272))),
        ("routed_experts.w_up_gate", (5, 64, 32), MuonLayout((5, 64, 32), split_dim=-2)),
        (
            "shared_experts.w_up_gate",
            (32, 5 * 64),
            MuonLayout((32, 5, 64), split_dim=-1, shared_experts=True),
        ),
        ("routed_experts.w_down", (5, 32, 64), None),
    ]
    checks = 0
    max_adapter_difference = 0.0
    for name, shape, layout in cases:
        torch.manual_seed(177)
        initial = torch.randn(shape, device=device) * 0.02
        views = logical_matrix_views(initial, name, shape, layout)
        params = [
            torch.nn.Parameter(distribute_tensor(v.contiguous(), mesh, [Shard(0)])) for v in views
        ]
        dense = SharedMuon(params, muon_backend="dion")
        native_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
        native_dense = Muon(
            native_params,
            distributed_mesh=mesh,
            lr=0.003,
            mu=0.95,
            nesterov=True,
            adjust_lr="rms_norm",
            weight_decay=0.1,
        )
        native_moe_params = [torch.nn.Parameter(v.clone()) for v in views]
        native_moe = Muon(
            native_moe_params,
            distributed_mesh=mesh.get_group(),
            lr=0.003,
            mu=0.95,
            nesterov=True,
            adjust_lr="rms_norm",
            weight_decay=0.1,
        )
        moe = object.__new__(OLMoDDPOptimizer)
        moe.algorithm = "muon_adamw"
        moe.muon_backend = "dion"
        moe.muon_momentum = 0.95
        moe.muon_rms = 0.2
        moe.muon_layouts = {name: layout} if layout else {}
        moe.states_dtype = torch.float32
        moe.states = {
            f"{name}.main": distribute_tensor(initial.flatten().clone(), mesh, [Shard(0)]),
            f"{name}.exp_avg": distribute_tensor(
                torch.zeros(initial.numel(), device=device), mesh, [Shard(0)]
            ),
            f"{name}.step": distribute_tensor(torch.zeros((), device=device), mesh, [Replicate()]),
        }
        moe.param_groups = [
            dict(
                named_params={name: torch.nn.Parameter(initial.clone())}, lr=0.003, weight_decay=0.1
            )
        ]
        for step, f in enumerate([1.0, 1.0, 0.0, 1.0]):
            factor = torch.tensor(f, device=device)
            dense.get_step_factor = lambda: factor
            moe.get_step_factor = lambda: factor
            lr = 0.003 / (step + 1)
            dense.param_groups[0]["lr"] = lr
            moe.param_groups[0]["lr"] = lr
            native_dense.param_groups[0]["lr"] = lr
            native_moe.param_groups[0]["lr"] = lr
            grad = torch.randn_like(initial) * 0.1
            for p, g in zip(params, logical_matrix_views(grad, name, shape, layout)):
                p.grad = distribute_tensor(g.contiguous(), mesh, [Shard(0)])
            moe.main_grad = {name: distribute_tensor(grad.flatten(), mesh, [Shard(0)]).to_local()}
            for p, q, r, g in zip(
                params,
                native_params,
                native_moe_params,
                logical_matrix_views(grad, name, shape, layout),
            ):
                q.grad = p.grad.detach().clone()
                r.grad = g.clone()
            dense.step()
            if f:
                native_dense.step()
                native_moe.step()
            with torch.no_grad():
                moe._step_foreach()
            full = moe.states[f"{name}.main"].full_tensor().reshape(shape)
            mom = moe.states[f"{name}.exp_avg"].full_tensor().reshape(shape)
            for p, q, r, ref, refmom in zip(
                params,
                native_params,
                native_moe_params,
                logical_matrix_views(full, name, shape, layout),
                logical_matrix_views(mom, name, shape, layout),
            ):
                torch.testing.assert_close(p.full_tensor(), q.full_tensor(), atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(ref, r, atol=1e-6, rtol=1e-6)
                max_adapter_difference = max(
                    max_adapter_difference, (p.full_tensor() - ref).abs().max().item()
                )
                torch.testing.assert_close(
                    dense.state[p]["momentum"].full_tensor(), refmom, atol=1e-6, rtol=1e-6
                )
                torch.testing.assert_close(
                    dense.state[p]["step"], moe.states[f"{name}.step"].full_tensor(), atol=0, rtol=0
                )
                checks += 1
        if dist.get_rank() == 0:
            print(
                "DION_BACKEND_CHECK " + json.dumps(dict(name=name, shape=shape, passed=True)),
                flush=True,
            )
    # Actual DCP optimizer state roundtrip on sharded parameters.
    from torch.distributed.checkpoint.state_dict import (
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )

    model = torch.nn.Module()
    model.register_parameter("weight", params[0])
    one = SharedMuon(model.parameters(), muon_backend="dion")
    model.weight.grad = torch.ones_like(model.weight)
    one.step()
    checkpoint = copy.deepcopy(get_optimizer_state_dict(model, one))
    restored = SharedMuon(model.parameters(), muon_backend="dion")
    set_optimizer_state_dict(model, restored, checkpoint)
    torch.testing.assert_close(
        restored.state[model.weight]["momentum"], one.state[model.weight]["momentum"]
    )
    if dist.get_rank() == 0:
        print(
            "DION_BACKEND_RESULT "
            + json.dumps(
                dict(
                    passed=True,
                    matrix_checks=checks,
                    sharded_dcp=True,
                    max_cross_layout_parameter_difference=max_adapter_difference,
                )
            ),
            flush=True,
        )


@requires_multi_gpu
def test_dion_adapters_match_native_distributed():
    pytest.importorskip("dion")
    run_distributed_test(_run_dion_adapters, world_size=2, backend="nccl", start_method="spawn")
