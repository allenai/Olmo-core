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
            Olmo3MoeConfig(
                hidden_size=D,
                dense_mlp_intermediate_size=H,
                moe_use_core_numerics=True,
                dense_layers_use_shared_expert=True,
            )
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


@pytest.mark.parametrize("kind", ["dense", "shared", "routed"])
@pytest.mark.parametrize("projection", ["up_proj", "gate_proj", "down_proj"])
@pytest.mark.parametrize("adaptation", ["adapter", "hook"])
def test_core_experts_preserve_projection_adapters_and_hooks(kind, projection, adaptation):
    from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import (
        Olmo3MoeCoreExperts,
        Olmo3MoeCoreSharedExpert,
        Olmo3MoeDenseMLP,
        Olmo3MoeExpert,
        _can_use_packed_projections,
    )
    from olmo_core.nn.moe.v2.hf.moe_numerics import core_swiglu

    class Adapter(torch.nn.Module):
        # Like an unmerged LoRA module, .weight exposes only the base weight.
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.delta = torch.nn.Parameter(torch.zeros_like(base.weight))

        @property
        def weight(self):
            return self.base.weight

        def forward(self, x):
            return self.base(x) + torch.nn.functional.linear(x, self.delta)

    torch.manual_seed(42)
    if kind == "dense":
        expert = Olmo3MoeDenseMLP(
            Olmo3MoeConfig(
                hidden_size=16,
                dense_mlp_intermediate_size=32,
                moe_use_core_numerics=True,
                dense_layers_use_shared_expert=True,
            )
        )
    elif kind == "shared":
        expert = Olmo3MoeCoreSharedExpert(16, 32, "silu")
    else:
        expert = Olmo3MoeExpert(16, 32, "silu")
    assert _can_use_packed_projections(expert)
    if adaptation == "adapter":
        adapter = Adapter(getattr(expert, projection))
        setattr(expert, projection, adapter)
        delta = adapter.delta
    else:
        layer = getattr(expert, projection)
        delta = torch.nn.Parameter(torch.zeros(layer.out_features))
        layer.register_forward_hook(lambda module, args, output: output + delta)
    assert not _can_use_packed_projections(expert)

    x = torch.randn(5, 16)
    if kind == "routed":
        experts = Olmo3MoeCoreExperts([expert])
        assert not experts._can_use_grouped_mm(x)
        indices = torch.zeros(5, 1, dtype=torch.long)
        weights = torch.ones(5, 1)

        def forward():
            return experts(x, indices, weights)

        def reference():
            return expert.down_proj(
                core_swiglu(torch.cat((expert.up_proj(x), expert.gate_proj(x)), -1))
            )

    else:

        def forward():
            return expert(x)

        def reference():
            return expert.down_proj(expert.act_fn(expert.gate_proj(x)) * expert.up_proj(x))

    before = forward().detach()
    with torch.no_grad():
        delta.fill_(0.25)
    actual = forward()
    expected = reference()
    assert not torch.equal(actual, before)
    torch.testing.assert_close(actual, expected)
    gradient = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, delta, gradient)[0]
    expected_grad = torch.autograd.grad(expected, delta, gradient)[0]
    assert actual_grad.count_nonzero() > 0
    torch.testing.assert_close(actual_grad, expected_grad)
    if kind == "routed":
        compiled = torch.compile(experts, fullgraph=True)
        torch.testing.assert_close(compiled(x, indices, weights), expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_core_numerics_reuses_custom_ops_across_module_imports(dtype, monkeypatch):
    import importlib.util
    import sys

    from olmo_core.nn.moe.v2.hf import moe_numerics

    def reject_registration(*args, **kwargs):
        pytest.fail("Importing another checkpoint must not register custom ops again")

    torch.manual_seed(91)
    values = torch.randn(5, 3, 16, dtype=dtype, requires_grad=True)
    probabilities = torch.rand(5, 3, requires_grad=True)
    compiled = torch.compile(moe_numerics.core_combine, fullgraph=True)
    expected = compiled(values, probabilities)
    gradient = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, (values, probabilities), gradient)

    with monkeypatch.context() as patch:
        patch.setattr(torch.library, "custom_op", reject_registration)
        for name in ("hf_checkpoint_a_numerics", "hf_checkpoint_b_numerics"):
            spec = importlib.util.spec_from_file_location(name, moe_numerics.__file__)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            patch.setitem(sys.modules, name, module)
            spec.loader.exec_module(module)
            assert module.core_combine is moe_numerics.core_combine
            assert module._core_combine_backward_op is moe_numerics._core_combine_backward_op
            actual = module.core_combine(values, probabilities)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual_grads = torch.autograd.grad(actual, (values, probabilities), gradient)
            for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    # A graph compiled before the imports must retain working forward/backward kernels.
    actual = compiled(values, probabilities)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grads = torch.autograd.grad(actual, (values, probabilities), gradient)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=GPU_MARKS)])
def test_core_numerics_preserves_standard_dense_projections(device, monkeypatch):
    from olmo_core.nn.feed_forward import FeedForward
    from olmo_core.nn.moe.v2.hf import modeling_olmo3moe
    from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig

    def reject_packed(*args):
        pytest.fail("Standard FeedForward layers must keep separate projections")

    monkeypatch.setattr(modeling_olmo3moe, "_core_shared_expert_forward", reject_packed)
    core = FeedForward(
        d_model=32, hidden_size=64, bias=False, dtype=torch.bfloat16, init_device=device
    )
    hf = modeling_olmo3moe.Olmo3MoeDenseMLP(
        Olmo3MoeConfig(hidden_size=32, dense_mlp_intermediate_size=64, moe_use_core_numerics=True)
    ).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        hf.gate_proj.weight.copy_(core.w1.weight)
        hf.up_proj.weight.copy_(core.w3.weight)
        hf.down_proj.weight.copy_(core.w2.weight)
        x = torch.randn(2, 7, 32, device=device, dtype=torch.bfloat16)
        torch.testing.assert_close(hf(x), core(x), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["dense", "shared", "routed"])
@pytest.mark.parametrize("offload", ["cpu", "disk"])
def test_core_experts_preserve_accelerate_offload(kind, offload, tmp_path):
    accelerate = pytest.importorskip("accelerate")
    from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import (
        Olmo3MoeCoreExperts,
        Olmo3MoeCoreSharedExpert,
        Olmo3MoeDenseMLP,
        Olmo3MoeExpert,
    )

    if kind == "dense":
        model = Olmo3MoeDenseMLP(
            Olmo3MoeConfig(
                hidden_size=16,
                dense_mlp_intermediate_size=32,
                moe_use_core_numerics=True,
                dense_layers_use_shared_expert=True,
            )
        )
    elif kind == "shared":
        model = Olmo3MoeCoreSharedExpert(16, 32, "silu")
    else:
        model = Olmo3MoeCoreExperts([Olmo3MoeExpert(16, 32, "silu")])
    x = torch.randn(5, 16)
    args = (x, torch.zeros(5, 1, dtype=torch.long), torch.ones(5, 1)) if kind == "routed" else (x,)
    with torch.no_grad():
        expected = model(*args)
        if offload == "cpu":
            accelerate.cpu_offload(model, execution_device=torch.device("cpu"))
        else:
            accelerate.disk_offload(model, str(tmp_path), execution_device=torch.device("cpu"))
        assert all(param.is_meta for param in model.parameters())
        for _ in range(2):
            torch.testing.assert_close(model(*args), expected)
            assert all(param.is_meta for param in model.parameters())


@pytest.mark.parametrize("customization", ["pre_hook", "hook", "wrapper", "subclass"])
@pytest.mark.parametrize("compile", [False, True])
def test_core_routed_experts_honor_module_calls(customization, compile, monkeypatch):
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import (
        Olmo3MoeCoreExperts,
        Olmo3MoeExpert,
        Olmo3MoeExperts,
    )

    torch.manual_seed(92)
    delta = torch.nn.Parameter(torch.full((16,), 0.25))

    class AdaptedExpert(Olmo3MoeExpert):
        def forward(self, x):
            return super().forward(x) + delta

    expert = (
        AdaptedExpert(16, 32, "silu")
        if customization == "subclass"
        else Olmo3MoeExpert(16, 32, "silu")
    )
    if customization == "pre_hook":
        expert.register_forward_pre_hook(lambda module, args: (args[0] + delta,))
    elif customization == "hook":
        expert.register_forward_hook(lambda module, args, output: output + delta)
    elif customization == "wrapper":
        original_forward = expert.forward
        monkeypatch.setattr(expert, "forward", lambda x: original_forward(x) + delta)

    experts = Olmo3MoeCoreExperts([expert])
    x = torch.randn(5, 16, requires_grad=True)
    indices = torch.zeros(5, 1, dtype=torch.long)
    weights = torch.ones(5, 1)
    # Exercise the eligibility guard even on hosts without grouped_mm support.
    monkeypatch.setattr(Olmo3MoeExperts, "_can_use_grouped_mm", lambda *args: True)
    assert not experts._can_use_grouped_mm(x)
    if compile:
        # Each case uses a different hook configuration on the same forward code.
        torch.compiler.reset()
    forward = torch.compile(experts, fullgraph=True) if compile else experts
    actual = forward(x, indices, weights)
    expected = expert(x)
    torch.testing.assert_close(actual, expected)
    gradient = torch.randn_like(actual)
    inputs = (x, delta, *expert.parameters())
    actual_grads = torch.autograd.grad(actual, inputs, gradient)
    expected_grads = torch.autograd.grad(expected, inputs, gradient)
    assert actual_grads[1].count_nonzero() > 0
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
