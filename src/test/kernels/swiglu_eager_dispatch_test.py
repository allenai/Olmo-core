from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from olmo_core.config import DType
from olmo_core.kernels import swiglu
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.testing import requires_gpu, requires_triton


def _config(fused=True, dtype=DType.float32):
    return RoutedExpertsConfig(
        d_model=8,
        hidden_size=8,
        num_experts=2,
        bias=False,
        dtype=dtype,
        fused_no_grad_activation=fused,
    )


def test_eager_no_grad_option_round_trip():
    assert _config().fused_no_grad_activation is True
    restored = RoutedExpertsConfig.from_dict(_config(False).as_dict())
    assert restored.build(init_device="meta").fused_no_grad_activation is False


@requires_gpu
@requires_triton
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize("dtype", [DType.float32, DType.bfloat16])
def test_eager_no_grad_dispatch_and_gradients(fused, grad, dtype):
    module = _config(fused, dtype=dtype).build(init_device="meta")
    x = torch.randn(19, 16, device="cuda", dtype=dtype.as_pt(), requires_grad=grad)
    count = torch.tensor(17, device="cuda")
    with (
        mock.patch(
            "olmo_core.nn.moe.v2.routed_experts.swiglu_valid_prefix",
            wraps=swiglu.swiglu_valid_prefix,
        ) as kernel,
        torch.set_grad_enabled(grad),
    ):
        actual = module.chunk_and_activate(x, num_elements=count)
        assert kernel.call_count == int(fused and not grad)
        expected = x[:, :8] * F.silu(x[:, 8:])
        if not fused or grad:
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        if grad:
            actual.sum().backward()
            actual_grad = x.grad.clone()
            x.grad = None
            expected.sum().backward()
            torch.testing.assert_close(actual_grad, x.grad, atol=0, rtol=0)
