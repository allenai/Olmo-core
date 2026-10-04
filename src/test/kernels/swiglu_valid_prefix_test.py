from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from olmo_core.config import DType
from olmo_core.kernels import swiglu
from olmo_core.kernels.swiglu import swiglu_backward_valid_prefix, swiglu_valid_prefix
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.testing import requires_gpu, requires_triton


def test_swiglu_valid_prefix_torch_fallback():
    rows = 5
    hidden = 4
    valid_rows = 3
    x = torch.randn((rows, hidden * 2), dtype=torch.float32)
    num_elements = torch.tensor(valid_rows, dtype=torch.long)
    out = torch.full((rows, hidden), 7.0, dtype=torch.float32)

    actual = swiglu_valid_prefix(x, num_elements, out=out)
    expected = x[:valid_rows, :hidden] * F.silu(x[:valid_rows, hidden:])

    torch.testing.assert_close(actual[:valid_rows], expected)
    torch.testing.assert_close(actual[valid_rows:], torch.full_like(actual[valid_rows:], 7.0))


def _swiglu_backward_reference(x: torch.Tensor, grad_h: torch.Tensor) -> torch.Tensor:
    hidden = x.shape[-1] // 2
    up = x[:, :hidden]
    gate = x[:, hidden:]
    gate_f32 = gate.to(torch.float32)
    grad_h_f32 = grad_h.to(torch.float32)
    up_f32 = up.to(torch.float32)
    sig = torch.sigmoid(gate_f32)
    silu_gate = gate_f32 * sig
    dsilu = sig * (1.0 + gate_f32 * (1.0 - sig))
    grad_up = grad_h_f32 * silu_gate
    grad_gate = grad_h_f32 * up_f32 * dsilu
    return torch.cat((grad_up, grad_gate), dim=-1).to(dtype=x.dtype)


def test_swiglu_backward_valid_prefix_torch_fallback():
    rows = 5
    hidden = 4
    valid_rows = 3
    x = torch.randn((rows, hidden * 2), dtype=torch.float32)
    grad_h = torch.randn((rows, hidden), dtype=torch.float32)
    num_elements = torch.tensor(valid_rows, dtype=torch.long)
    out = torch.full_like(x, 7.0)

    actual = swiglu_backward_valid_prefix(x, grad_h, num_elements, out=out)
    expected = _swiglu_backward_reference(x[:valid_rows], grad_h[:valid_rows])

    torch.testing.assert_close(actual[:valid_rows], expected)
    torch.testing.assert_close(actual[valid_rows:], torch.full_like(actual[valid_rows:], 7.0))


@requires_gpu
@requires_triton
def test_swiglu_valid_prefix_matches_torch_and_leaves_tail_untouched():
    rows = 37
    hidden = 64
    valid_rows = 19
    x = torch.randn((rows, hidden * 2), device="cuda", dtype=torch.bfloat16)
    num_elements = torch.tensor(valid_rows, device="cuda", dtype=torch.long)
    out = torch.full((rows, hidden), 7.0, device="cuda", dtype=torch.bfloat16)

    actual = swiglu_valid_prefix(x, num_elements, out=out)
    expected = x[:valid_rows, :hidden] * F.silu(x[:valid_rows, hidden:])

    torch.testing.assert_close(actual[:valid_rows], expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        actual[valid_rows:],
        torch.full_like(actual[valid_rows:], 7.0),
        atol=0.0,
        rtol=0.0,
    )


@requires_gpu
@requires_triton
def test_swiglu_backward_valid_prefix_accepts_device_start_offset():
    rows = 37
    hidden = 64
    start_row = 7
    valid_rows = 19
    x = torch.randn((rows, hidden * 2), device="cuda", dtype=torch.bfloat16)
    grad_h = torch.randn((rows, hidden), device="cuda", dtype=torch.bfloat16)
    start = torch.tensor(start_row, device="cuda", dtype=torch.long)
    num_elements = torch.tensor(valid_rows, device="cuda", dtype=torch.long)
    out = torch.full_like(x, 7.0)

    actual = swiglu_backward_valid_prefix(x, grad_h, num_elements, start=start, out=out)
    expected = _swiglu_backward_reference(
        x[start_row : start_row + valid_rows],
        grad_h[start_row : start_row + valid_rows],
    )

    torch.testing.assert_close(
        actual[start_row : start_row + valid_rows],
        expected,
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        actual[:start_row],
        torch.full_like(actual[:start_row], 7.0),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        actual[start_row + valid_rows :],
        torch.full_like(actual[start_row + valid_rows :], 7.0),
        atol=0.0,
        rtol=0.0,
    )


@requires_gpu
@requires_triton
def test_swiglu_valid_prefix_accepts_device_start_offset():
    rows = 37
    hidden = 64
    start_row = 7
    valid_rows = 19
    x = torch.randn((rows, hidden * 2), device="cuda", dtype=torch.bfloat16)
    start = torch.tensor(start_row, device="cuda", dtype=torch.long)
    num_elements = torch.tensor(valid_rows, device="cuda", dtype=torch.long)
    out = torch.full((rows, hidden), 7.0, device="cuda", dtype=torch.bfloat16)

    actual = swiglu_valid_prefix(x, num_elements, start=start, out=out)
    expected = x[start_row : start_row + valid_rows, :hidden] * F.silu(
        x[start_row : start_row + valid_rows, hidden:]
    )

    torch.testing.assert_close(
        actual[start_row : start_row + valid_rows],
        expected,
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        actual[:start_row],
        torch.full_like(actual[:start_row], 7.0),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        actual[start_row + valid_rows :],
        torch.full_like(actual[start_row + valid_rows :], 7.0),
        atol=0.0,
        rtol=0.0,
    )


@requires_gpu
@requires_triton
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("valid_rows", [0, 3, 6])
def test_scoring_swiglu_matches_eager_on_rounding_regression_values(dtype, valid_rows):
    # Stable finite values expose the missing intermediate BF16/FP16 rounding
    # without relying on a broad tolerance that hides the original discrepancy.
    up = torch.tensor([1.5, -2.75, 0.25, 16.0], device="cuda", dtype=dtype)
    gate = torch.tensor([1.0, -1.0, 0.5, -2.0, 3.0, 0.125], device="cuda", dtype=dtype)
    x = torch.cat((up.expand(8, -1), gate.repeat(2)[:8, None].expand(-1, 4)), dim=-1).contiguous()
    out = torch.full((8, 4), 77.0, device="cuda", dtype=dtype)
    start = torch.tensor(1, device="cuda", dtype=torch.long)
    count = torch.tensor(valid_rows, device="cuda", dtype=torch.long)
    result = swiglu_valid_prefix(x, count, start=start, out=out, match_eager_rounding=True)
    eager = x[1 : 1 + valid_rows, :4] * F.silu(x[1 : 1 + valid_rows, 4:])
    torch.testing.assert_close(
        result[1 : 1 + valid_rows], eager, rtol=1e-6 if dtype == torch.float32 else 0, atol=0
    )
    assert torch.equal(result[:1], torch.full_like(result[:1], 77.0))
    assert torch.equal(result[1 + valid_rows :], torch.full_like(result[1 + valid_rows :], 77.0))
    if valid_rows and dtype == torch.bfloat16:
        fused = swiglu_valid_prefix(x, count, start=start)
        assert not torch.equal(fused[1 : 1 + valid_rows], eager)


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_row_specialization_config_and_cpu_fallback(mode):
    config = RoutedExpertsConfig(
        d_model=8,
        hidden_size=8,
        num_experts=2,
        bias=False,
        dtype=DType.float32,
        row_specialization=mode,
    )
    assert config.build(init_device="meta").row_specialization == mode
    assert RoutedExpertsConfig.from_dict(config.as_dict()).row_specialization == mode
    x = torch.randn(7, 16)
    actual = swiglu_valid_prefix(x, torch.tensor(5), row_specialization=mode)
    torch.testing.assert_close(actual[:5], x[:5, :8] * F.silu(x[:5, 8:]))
    with pytest.raises(ValueError, match="row_specialization"):
        swiglu_valid_prefix(x, torch.tensor(5), row_specialization="invalid")  # type: ignore[arg-type]


@requires_gpu
@requires_triton
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("eager_rounding", [False, True])
def test_dynamic_rows_exact_values_and_boundaries(dtype, eager_rounding):
    # Includes a partial column tile and capacities on both sides of the grid cap.
    for rows, valid, offset in [(1, 0, 0), (17, 9, 3), (37, 37, 0), (16385, 16380, 5)]:
        x = torch.randn(rows, 2 * 257, device="cuda", dtype=dtype)
        count = torch.tensor(valid, device="cuda", dtype=torch.long)
        start = torch.tensor(offset, device="cuda", dtype=torch.long)
        expected = torch.full((rows, 257), 7.0, device="cuda", dtype=dtype)
        actual = expected.clone()
        swiglu_valid_prefix(
            x, count, start=start, out=expected, match_eager_rounding=eager_rounding
        )
        swiglu_valid_prefix(
            x,
            count,
            start=start,
            out=actual,
            match_eager_rounding=eager_rounding,
            row_specialization="dynamic",
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(actual[:offset], torch.full_like(actual[:offset], 7.0))
        assert torch.equal(actual[offset + valid :], torch.full_like(actual[offset + valid :], 7.0))


@requires_gpu
@requires_triton
def test_dynamic_rows_do_not_compile_for_new_capacities():
    compiled = []
    original = swiglu._swiglu_valid_prefix_dynamic_rows_kernel._do_compile

    def observe(*args, **kwargs):
        compiled.append(1)
        return original(*args, **kwargs)

    # Distinct width to keep this test independent of other kernel warmups.
    with mock.patch.object(swiglu._swiglu_valid_prefix_dynamic_rows_kernel, "_do_compile", observe):
        for index, rows in enumerate([1, 16, 17, 63, 128, 257, 16385]):
            x = torch.randn(rows, 2 * 73, device="cuda", dtype=torch.bfloat16)
            count = torch.tensor(rows, device="cuda", dtype=torch.long)
            swiglu_valid_prefix(x, count, row_specialization="dynamic")
            torch.cuda.synchronize()
            if index == 0:
                first = len(compiled)
            assert len(compiled) == first, f"capacity {rows} triggered another compilation"


@requires_gpu
@requires_triton
@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_routed_dispatch_and_gradient_path(mode):
    module = RoutedExpertsConfig(
        d_model=8,
        hidden_size=8,
        num_experts=2,
        bias=False,
        dtype=DType.float32,
        row_specialization=mode,
    ).build(init_device="meta")
    x = torch.randn(19, 16, device="cuda", requires_grad=True)
    count = torch.tensor(17, device="cuda", dtype=torch.long)
    with mock.patch(
        "olmo_core.nn.moe.v2.routed_experts.swiglu_valid_prefix", wraps=swiglu_valid_prefix
    ) as kernel:
        with torch.no_grad():
            module.chunk_and_activate(x, num_elements=count)
        assert kernel.call_args.kwargs["row_specialization"] == mode
        kernel.reset_mock()
        value = module.chunk_and_activate(x, num_elements=count)
        value.sum().backward()
        kernel.assert_not_called()
    reference = x.detach().clone().requires_grad_()
    (reference[:, :8] * F.silu(reference[:, 8:])).sum().backward()
    torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)


@pytest.mark.parametrize("rounding", [False, True])
def test_routed_experts_eager_rounding_is_opt_in(rounding):
    config = RoutedExpertsConfig(
        d_model=8, hidden_size=8, num_experts=2, bias=False, dtype=DType.bfloat16
    )
    assert config.match_eager_rounding is False
    config.match_eager_rounding = rounding
    restored = RoutedExpertsConfig.from_dict(config.as_dict())
    experts = restored.build(init_device="meta")
    assert experts.match_eager_rounding is rounding
    # Exercise dispatch without a GPU; numerical kernel parity is covered above.
    x = mock.Mock(is_cuda=True, device=torch.device("cuda"))
    count = mock.Mock(device=x.device)
    with mock.patch(
        "olmo_core.nn.moe.v2.routed_experts.swiglu_valid_prefix"
    ) as kernel, torch.no_grad():
        experts.chunk_and_activate(x, num_elements=count)
    assert kernel.call_args.kwargs["match_eager_rounding"] is rounding


@requires_gpu
@requires_triton
@pytest.mark.parametrize("rounding", [False, True])
def test_routed_experts_rounding_preserves_selected_numerics(rounding):
    experts = RoutedExpertsConfig(
        d_model=8,
        hidden_size=4,
        num_experts=2,
        bias=False,
        dtype=DType.bfloat16,
        match_eager_rounding=rounding,
    ).build(init_device="meta")
    x = torch.tensor(
        [[1.5, -2.75, 0.25, 16.0, 1.0, -1.0, 0.5, -2.0]], device="cuda", dtype=torch.bfloat16
    )
    count = torch.tensor(1, device="cuda")
    with torch.no_grad():
        actual = experts.chunk_and_activate(x, num_elements=count)
    expected = x[:, :4] * F.silu(x[:, 4:]) if rounding else swiglu_valid_prefix(x, count)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_gpu
@requires_triton
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("scale", [1.0, 4.0])
@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_scoring_swiglu_randomized_eager_agreement(dtype, scale, mode):
    generator = torch.Generator(device="cuda").manual_seed(47)
    x = (torch.randn(1024, 2 * 257, device="cuda", generator=generator) * scale).to(dtype)
    count = torch.tensor(x.shape[0], device="cuda", dtype=torch.long)
    actual = swiglu_valid_prefix(x, count, match_eager_rounding=True, row_specialization=mode)
    expected = x[:, :257] * F.silu(x[:, 257:])
    # BF16 agrees exactly on this sample. FP16/FP32 can expose differences in
    # sigmoid approximations despite matching the intermediate rounding step.
    if dtype == torch.bfloat16:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif dtype == torch.float16:
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=1e-6)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
