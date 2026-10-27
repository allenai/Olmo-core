import math
from types import SimpleNamespace

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from olmo_core.config import DType
from olmo_core.nn.moe.v2.routed_experts import ExpertActivation
from olmo_core.nn.moe.v2.router import MoERouterV2
from olmo_core.nn.moe.v2.shared_experts import SharedExperts
from olmo_core.optim.moe_optimizer import OLMoDDPOptimizer, OLMoDDPOptimizerConfig
from olmo_core.optim.muon_math import (
    _expert_batch_size,
    collect_muon_layouts,
    muon_direction,
    newton_schulz,
    uses_muon,
)
from olmo_core.testing.distributed import run_distributed_test


def test_parameter_routing():
    with pytest.raises(ValueError, match="requires a MuonLayout"):
        uses_muon("module.routed_experts_router.weight", (15,))
    for name, shape in [
        ("blocks.0.attention.w_q.weight", (8, 16)),
        ("blocks.0.feed_forward.routed_experts.w_down", (4, 8, 16)),
    ]:
        assert uses_muon(name, shape)
    for name, shape in [
        ("embeddings.weight", (16, 8)),
        ("lm_head.w_out.weight", (16, 8)),
        ("blocks.0.attention.q_norm.weight", (4, 8)),
        ("blocks.0.attention.q_conv1d.weight", (8, 4)),
        ("blocks.0.attention.dt_bias", (8,)),
    ]:
        assert not uses_muon(name, shape)


@pytest.mark.parametrize("prefix", ["blocks.0.attention.", "module.blocks.0.attention."])
def test_kda_projection_routing(prefix):
    for name in ("w_q", "w_k", "w_v", "w_out", "f_proj_1", "f_proj_2", "g_proj_1", "g_proj_2"):
        assert uses_muon(prefix + name + ".weight", (8, 16))
    assert not uses_muon(prefix + "w_b.weight", (8, 1024))
    assert not uses_muon(prefix + "g_proj_2.bias", (16,))
    for name in ("q_norm", "k_norm", "o_norm"):
        assert not uses_muon(prefix + name + ".weight", (8, 16))


def _real_v2_modules(
    num_shared=1, activation=ExpertActivation.swiglu, d_model=5, hidden_size=7, num_experts=3
):
    model = torch.nn.Module()
    model.routed_experts_router = MoERouterV2(
        d_model=d_model, num_experts=num_experts, top_k=1, bias=True
    )
    model.shared_experts = SharedExperts(
        d_model=d_model,
        hidden_size=hidden_size,
        num_experts=num_shared,
        bias=False,
        dtype=DType.float32,
        activation=activation,
    )
    # Exercise names from the actual optimizer's DDP-wrapped build path.
    wrapped = torch.nn.Module()
    wrapped.module = model
    wrapped.has_grad_accum_fp32_buffer = False
    return wrapped


@pytest.mark.parametrize("num_shared", [1, 3])
@pytest.mark.parametrize("activation", [ExpertActivation.swiglu, ExpertActivation.relu2])
def test_real_shared_expert_layout(num_shared, activation):
    model = _real_v2_modules(num_shared, activation)
    name = "module.shared_experts.w_up_gate"
    layout = collect_muon_layouts([model])[name]
    torch.manual_seed(23)
    direction = torch.randn_like(model.module.shared_experts.w_up_gate)
    projections = 2 if activation == ExpertActivation.swiglu else 1
    expected = torch.cat(
        [
            newton_schulz(direction[:, i * 7 : (i + 1) * 7]) * math.sqrt(7) * 0.2
            for i in range(num_shared * projections)
        ],
        dim=1,
    )
    torch.testing.assert_close(muon_direction(name, direction, layout=layout), expected)
    router = model.module.routed_experts_router
    router_layout = collect_muon_layouts([model])["module.routed_experts_router.weight"]
    assert router.weight.shape == (15,)
    assert router_layout.shape == (3, 5)
    assert uses_muon("module.routed_experts_router.weight", router_layout.shape)


def _run_real_v2_dispatch():
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))
    torch.manual_seed(31)
    model = _real_v2_modules(num_shared=3, d_model=64, hidden_size=67, num_experts=66)
    optim = OLMoDDPOptimizerConfig(
        algorithm="muon_adamw",
        lr=0.003,
        weight_decay=0.1,
        compile=False,
        use_distributed=True,
    ).build(
        [model],
        train_module=SimpleNamespace(  # type: ignore[arg-type]
            world_mesh={"dense": mesh, "moe": None}, device=torch.device("cpu")
        ),
    )
    expected = {}
    for name, param in model.named_parameters():
        if not name.endswith("bias"):
            assert any(p.is_shard() for p in optim.states[f"{name}.exp_avg"].placements)
        initial = optim.states[f"{name}.main"].full_tensor().reshape(param.shape).clone()
        grad = torch.randn_like(initial)
        state = optim.states[f"{name}.main"]
        optim.main_grad[name] = distribute_tensor(grad.flatten(), mesh, state.placements).to_local()
        if name.endswith("routed_experts_router.weight"):
            direction = newton_schulz((1.95 * grad).reshape(66, 64)) * math.sqrt(66) * 0.2
        elif name.endswith("w_up_gate"):
            direction = torch.cat(
                [
                    newton_schulz(1.95 * grad[:, i * 67 : (i + 1) * 67]) * math.sqrt(67) * 0.2
                    for i in range(6)
                ],
                dim=1,
            )
        elif name.endswith("w_down"):
            direction = torch.stack([newton_schulz(1.95 * g) * math.sqrt(67) * 0.2 for g in grad])
        else:
            adam_param = torch.nn.Parameter(initial.clone())
            adam = torch.optim.AdamW(
                [adam_param],
                lr=0.003,
                weight_decay=0.1,
                betas=optim.param_groups[0]["betas"],
                eps=optim.param_groups[0]["eps"],
            )
            adam_param.grad = grad
            adam.step()
            expected[name] = adam_param.flatten()
            continue
        expected[name] = initial.flatten() * (1 - 0.003 * 0.1) - 0.003 * direction.flatten()
    with torch.no_grad():
        optim._step_foreach()
    for name, value in expected.items():
        torch.testing.assert_close(optim.states[f"{name}.main"].full_tensor(), value)


def test_real_v2_optimizer_dispatch():
    run_distributed_test(_run_real_v2_dispatch, world_size=2, backend="gloo", start_method="spawn")


def test_five_steps_match_singular_value_polynomial():
    torch.manual_seed(17)
    matrix = torch.randn(7, 11)
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    s = s / matrix.norm()
    for a, b, c in [(3.4445, -4.7750, 2.0315)] * 5:
        s = a * s + b * s**3 + c * s**5
    expected = (u * s) @ vh
    torch.testing.assert_close(newton_schulz(matrix), expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(newton_schulz(matrix.T), expected.T, atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(newton_schulz(torch.zeros(3, 5))) == 0


def test_experts_and_packed_swiglu_are_independent():
    torch.manual_seed(5)
    packed = torch.randn(3, 8, 6)
    expected = torch.stack(
        [
            torch.cat([newton_schulz(part) * 6**0.5 * 0.2 for part in expert.chunk(2, dim=0)])
            for expert in packed
        ]
    )
    torch.testing.assert_close(muon_direction("experts.w_up_gate", packed), expected)


@pytest.mark.parametrize("transpose", [False, True])
def test_expert_batching_matches_original_with_partial_tail(transpose):
    torch.manual_seed(19)
    matrices = torch.randn(131, 7, 11)
    if transpose:
        matrices = matrices.mT
    expected = torch.cat([newton_schulz(chunk) for chunk in matrices.split(16)])
    expected *= math.sqrt(11) * 0.2
    torch.testing.assert_close(muon_direction("experts.w_down", matrices), expected)


def test_expert_batch_workspace_bound():
    assert _expert_batch_size(7, 11) == 128
    assert _expert_batch_size(544, 272) > 16
    for rows, cols in [(544, 272), (2048, 4096), (4096, 2048)]:
        batch = _expert_batch_size(rows, cols)
        estimated_workspace = batch * 4 * (3 * rows * cols + 3 * min(rows, cols) ** 2)
        assert estimated_workspace <= 256 * 1024**2
    assert _expert_batch_size(16384, 16384) == 1


def _run_sharded_muon():
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))
    torch.manual_seed(13)
    name = "experts.w_up_gate"
    shape = (3, 8, 6)  # DP shards cut through the second expert.
    initial = torch.randn(shape)
    grad = torch.randn(shape)
    for placement in (Replicate(), Shard(0)):
        optim = object.__new__(OLMoDDPOptimizer)
        optim.muon_momentum = 0.95
        optim.muon_rms = 0.2
        optim.states = {
            f"{name}.main": distribute_tensor(initial.flatten().clone(), mesh, [placement]),
            f"{name}.exp_avg": distribute_tensor(torch.zeros(initial.numel()), mesh, [placement]),
            f"{name}.step": distribute_tensor(torch.tensor(0.0), mesh, [Replicate()]),
        }
        optim.main_grad = {name: distribute_tensor(grad.flatten(), mesh, [placement]).to_local()}
        group = {"lr": 0.003, "weight_decay": 0.1}
        expected = initial.clone()
        momentum = torch.zeros_like(grad)
        for factor in (1.0, 0.0, 1.0):
            before = {key: value.full_tensor().clone() for key, value in optim.states.items()}
            optim._step_muon(name, shape, group, torch.tensor(factor))
            if factor:
                momentum = 0.95 * momentum + grad
                expected = expected * (1 - 0.003 * 0.1) - 0.003 * muon_direction(
                    name, 0.95 * momentum + grad
                )
            else:
                for key, value in before.items():
                    torch.testing.assert_close(
                        optim.states[key].full_tensor(), value, atol=0, rtol=0
                    )
            torch.testing.assert_close(
                optim.states[f"{name}.main"].full_tensor().reshape(shape), expected
            )
            torch.testing.assert_close(
                optim.states[f"{name}.exp_avg"].full_tensor().reshape(shape), momentum
            )

    # Exercise dispatch through the real optimizer, its AdamW fallback, and resume.
    def build():
        params = {
            "hidden.weight": torch.nn.Parameter(initial[0].clone()),
            "embeddings.weight": torch.nn.Parameter(initial[0].clone()),
        }
        return OLMoDDPOptimizer(
            [{"named_params": params, "pg": "dp"}],
            world_mesh={"dense": mesh, "moe": None},
            algorithm="muon_adamw",
            use_distributed=False,
            lr=0.003,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
        )

    optim = build()
    adam_param = torch.nn.Parameter(initial[0].clone())
    adam = torch.optim.AdamW([adam_param], lr=0.003, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    optim.main_grad = {name: grad[0].flatten() for name in ("hidden.weight", "embeddings.weight")}
    adam_param.grad = grad[0].clone()
    adam.step()
    with torch.no_grad():
        optim._step_foreach()
    torch.testing.assert_close(
        optim.states["embeddings.weight.main"].full_tensor().reshape(8, 6), adam_param
    )
    checkpoint = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in optim.state_dict().items()
    }
    resumed = build()
    resumed.load_state_dict(checkpoint)
    resumed.main_grad = optim.main_grad
    with torch.no_grad():
        optim._step_foreach()
        resumed._step_foreach()
    for key in optim.states:
        torch.testing.assert_close(
            optim.states[key].full_tensor(), resumed.states[key].full_tensor()
        )


def test_shards_preserve_logical_matrices_and_skip_state():
    run_distributed_test(_run_sharded_muon, world_size=2, backend="gloo", start_method="spawn")
