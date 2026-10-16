from typing import Union

import pytest
import torch

from olmo_core.testing.utils import (
    GPU_MARKS,
    requires_gpu,
    requires_te,
    requires_triton,
)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=GPU_MARKS)])
def test_core_combine_rounds_updates_in_routing_order(device):
    from olmo_core.nn.moe.v2.hf.moe_numerics import core_combine

    torch.manual_seed(73)
    values = torch.randn(5, 16, 32, device=device).bfloat16()
    probabilities = torch.rand(5, 16, device=device)
    # FP64 supplies an independent arithmetic oracle before each BF16 rounding.
    expected = torch.zeros(5, 32, dtype=torch.bfloat16, device=device)
    for slot in range(16):
        product = values[:, slot].double() * probabilities[:, slot, None].bfloat16().double()
        if device == "cuda" and torch.cuda.get_device_capability()[0] < 9:
            product = product.bfloat16().double()
        expected = (expected.double() + product).bfloat16()
    actual = core_combine(values, probabilities)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    fp32_sum = (values.float() * probabilities[..., None]).sum(1).bfloat16()
    assert not torch.equal(actual, fp32_sum)


@requires_gpu
@requires_te
@requires_triton
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_hf_core_experts_match_actual_core_and_te(dtype):
    from olmo_core.config import DType
    from olmo_core.nn.moe.utils import moe_permute_no_compile, moe_unpermute_no_compile
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import (
        Olmo3MoeCoreExperts,
        Olmo3MoeExpert,
    )
    from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig

    torch.manual_seed(83)
    N, D, inner_dim, E, K = 7, 32, 64, 8, 3
    core = RoutedExpertsConfig(
        d_model=D,
        hidden_size=inner_dim,
        num_experts=E,
        bias=False,
        dtype=DType.bfloat16 if dtype == torch.bfloat16 else DType.float32,
    ).build(init_device="cuda")
    hf = Olmo3MoeCoreExperts(Olmo3MoeExpert(D, inner_dim, "silu") for _ in range(E)).to(
        device="cuda", dtype=dtype
    )
    with torch.no_grad():
        for param in core.parameters():
            param.normal_(std=0.1)
        for index, expert in enumerate(hf):
            expert.up_proj.weight.copy_(core.w_up_gate[index, :inner_dim])
            expert.gate_proj.weight.copy_(core.w_up_gate[index, inner_dim:])
            expert.down_proj.weight.copy_(core.w_down[index].t())
        x = torch.randn(N, D, device="cuda", dtype=dtype)
        weights, indices = torch.randn(N, E, device="cuda").softmax(-1).topk(K)
        counts = torch.bincount(indices.flatten(), minlength=E).to(torch.int32)
        permuted, row_map = moe_permute_no_compile(
            inp=x, routing_map=indices.int(), num_out_tokens=N * K, map_type="index"
        )
        expected = moe_unpermute_no_compile(
            inp=core(permuted, counts),
            row_id_map=row_map,
            restore_shape=x.shape,
            merging_probs=weights,
            map_type="index",
        )
        actual = hf._forward_grouped_mm(x, indices, weights)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_core_numerics_is_serialized_and_old_configs_keep_original_mode():
    from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import (
        Olmo3MoeCoreExperts,
        Olmo3MoeCoreSharedExpert,
        Olmo3MoeSparseMLP,
    )

    assert not Olmo3MoeConfig().moe_use_core_numerics
    config = Olmo3MoeConfig(
        hidden_size=32,
        moe_intermediate_size=64,
        n_routed_experts=4,
        shared_expert_intermediate_size=64,
        moe_use_core_numerics=True,
    )
    config = Olmo3MoeConfig.from_dict(config.to_dict())
    assert isinstance(Olmo3MoeSparseMLP(config).experts, Olmo3MoeCoreExperts)
    assert isinstance(Olmo3MoeSparseMLP(config).shared_expert, Olmo3MoeCoreSharedExpert)
    with pytest.raises(ValueError, match="requires hidden_act"):
        Olmo3MoeConfig(moe_use_core_numerics=True, hidden_act="relu")


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("kind", ["dense", "shared"])
def test_hf_dense_and_shared_experts_match_core_packed_gemm(dtype, kind):
    from olmo_core.config import DType
    from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import (
        Olmo3MoeCoreSharedExpert,
        Olmo3MoeDenseMLP,
    )
    from olmo_core.nn.moe.v2.shared_experts import SharedExperts

    # Real 275M dense-first dimensions expose Ampere GEMM rounding differences
    # that the small routed-expert test does not exercise.
    torch.manual_seed(719)
    D, H = 544, 4352
    core = SharedExperts(
        d_model=D,
        hidden_size=H,
        num_experts=1,
        bias=False,
        dtype=DType.bfloat16 if dtype == torch.bfloat16 else DType.float32,
        init_device="cuda",
    )
    hf: Union[Olmo3MoeDenseMLP, Olmo3MoeCoreSharedExpert] = (
        Olmo3MoeDenseMLP(
            Olmo3MoeConfig(hidden_size=D, dense_mlp_intermediate_size=H, moe_use_core_numerics=True)
        )
        if kind == "dense"
        else Olmo3MoeCoreSharedExpert(D, H, "silu")
    )
    hf.to(device="cuda", dtype=dtype)
    with torch.no_grad():
        hf.up_proj.weight.copy_(core.w_up_gate[:, :H].t())
        hf.gate_proj.weight.copy_(core.w_up_gate[:, H:].t())
        hf.down_proj.weight.copy_(core.w_down[0].t())
        x = torch.randn(1, 60, D, device="cuda", dtype=dtype)
        torch.testing.assert_close(hf(x), core(x).squeeze(0), rtol=1e-4, atol=1e-4)


@requires_gpu
def test_core_swiglu_cuda_without_triton(monkeypatch):
    from olmo_core.nn.moe.v2.hf import moe_numerics

    monkeypatch.setattr(moe_numerics, "triton", None)
    x = torch.randn(7, 64, device="cuda", dtype=torch.bfloat16)
    up, gate = x.float().chunk(2, dim=-1)
    expected = (up * gate * torch.sigmoid(gate)).bfloat16()
    with torch.inference_mode():
        torch.testing.assert_close(moe_numerics.core_swiglu(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=GPU_MARKS)])
def test_core_combine_fullgraph_and_gradients(dtype, device):
    from olmo_core.nn.moe.v2.hf.moe_numerics import core_combine

    torch.manual_seed(73)
    values = torch.randn(5, 16, 32, dtype=dtype, device=device, requires_grad=True)
    probabilities = torch.rand(5, 16, device=device, requires_grad=True)
    expected = values.new_zeros((5, 32))
    for slot in range(16):
        value = values[:, slot].float()
        probability = probabilities[:, slot, None].to(dtype).float()
        if (
            device == "cuda"
            and dtype == torch.bfloat16
            and torch.cuda.get_device_capability()[0] < 9
        ):
            product = (value * probability).to(dtype)
            expected = (expected.float() + product.float()).to(dtype)
        else:
            expected = torch.addcmul(expected.float(), value, probability).to(dtype)
    actual = torch.compile(core_combine, fullgraph=True)(values, probabilities)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(expected, (values, probabilities), gradient)
    actual_grads = torch.autograd.grad(actual, (values, probabilities), gradient)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=GPU_MARKS)])
def test_core_experts_fullgraph_fallback(device):
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import (
        Olmo3MoeCoreExperts,
        Olmo3MoeExpert,
    )

    torch.manual_seed(83)
    experts = Olmo3MoeCoreExperts(Olmo3MoeExpert(16, 32, "silu") for _ in range(4)).to(device)
    x = torch.randn(5, 16, device=device)
    weights, indices = torch.randn(5, 4, device=device).softmax(-1).topk(2)
    with torch.no_grad():
        expected = experts(x, indices, weights)
        actual = torch.compile(experts, fullgraph=True)(x, indices, weights)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
