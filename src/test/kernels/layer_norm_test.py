import pytest
import torch
import torch.nn.functional as F

from olmo_core.kernels.layer_norm import layer_norm
from olmo_core.testing import requires_gpu


def _reference(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, dy: torch.Tensor):
    """Float32 LayerNorm output and gradients for inputs of any dtype."""
    x32, w32, b32 = (t.detach().float().requires_grad_() for t in (x, weight, bias))
    y = F.layer_norm(x32, (x.shape[-1],), w32, b32, 1e-6)
    y.backward(dy.float())
    assert x32.grad is not None and w32.grad is not None and b32.grad is not None
    return y.detach(), x32.grad, w32.grad, b32.grad


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 7, 4099])
@pytest.mark.parametrize("hidden", [72, 1152])
def test_layer_norm_matches_float32_reference(dtype: torch.dtype, rows: int, hidden: int):
    torch.manual_seed(0)
    device = torch.device("cuda")
    x = (torch.randn(rows, hidden, device=device) * 3 + 1).to(dtype).requires_grad_()
    weight = (1 + 0.1 * torch.randn(hidden, device=device)).to(dtype).requires_grad_()
    bias = (0.1 * torch.randn(hidden, device=device)).to(dtype).requires_grad_()
    dy = torch.randn(rows, hidden, device=device).to(dtype)

    y = layer_norm(x, weight, bias, 1e-6)
    y.backward(dy)
    y_ref, dx_ref, dw_ref, db_ref = _reference(x, weight, bias, dy)

    assert y.dtype == dtype and x.grad is not None and x.grad.dtype == dtype
    assert weight.grad is not None and bias.grad is not None
    tol = dict(atol=1e-4, rtol=1e-4) if dtype == torch.float32 else dict(atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(y.float(), y_ref, **tol)
    torch.testing.assert_close(x.grad.float(), dx_ref, **tol)
    # The parameter gradients sum over every row.
    scale = max(1.0, rows**0.5)
    torch.testing.assert_close(
        weight.grad.float(), dw_ref, atol=tol["atol"] * scale, rtol=tol["rtol"]
    )
    torch.testing.assert_close(
        bias.grad.float(), db_ref, atol=tol["atol"] * scale, rtol=tol["rtol"]
    )


@requires_gpu
def test_layer_norm_takes_strided_inputs_and_any_leading_shape():
    torch.manual_seed(0)
    device = torch.device("cuda")
    base = torch.randn(64, 3, 5, device=device).requires_grad_()
    x = base.transpose(0, 2)  # (5, 3, 64), not contiguous
    weight = torch.randn(64, device=device).requires_grad_()
    bias = torch.randn(64, device=device).requires_grad_()

    y = layer_norm(x, weight, bias)
    y.sum().backward()
    assert y.shape == x.shape
    torch.testing.assert_close(y, F.layer_norm(x, (64,), weight, bias), atol=1e-5, rtol=1e-5)
    assert base.grad is not None and base.grad.shape == base.shape


def test_layer_norm_uses_pytorch_on_cpu():
    torch.manual_seed(0)
    x = torch.randn(10, 32)
    weight, bias = torch.randn(32), torch.randn(32)
    torch.testing.assert_close(
        layer_norm(x, weight, bias, 1e-5), F.layer_norm(x, (32,), weight, bias, 1e-5)
    )
