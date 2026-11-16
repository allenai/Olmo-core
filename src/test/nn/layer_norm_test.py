import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.layer_norm import (
    CuTeRMSNorm,
    FusedRMSNorm,
    L2Norm,
    LayerNormConfig,
    LayerNormType,
    NemotronRMSNorm,
    RMSNorm,
)
from olmo_core.testing import requires_flash_attn_2, requires_gpu, requires_quack


@requires_gpu
@requires_flash_attn_2
@pytest.mark.parametrize("bias", [pytest.param(True, id="bias"), pytest.param(False, id="no-bias")])
@pytest.mark.parametrize(
    "dtype", [pytest.param(torch.float32, id="fp32"), pytest.param(torch.bfloat16, id="bf16")]
)
def test_fused_rms_norm(bias, dtype):
    dim = 64
    norm = RMSNorm(size=dim, bias=bias, init_device="cuda")
    norm_fused = FusedRMSNorm(size=dim, bias=bias, init_device="cuda")

    x = torch.randn(4, dim, device="cuda", dtype=dtype)
    y1 = norm(x)
    y2 = norm_fused(x)
    torch.testing.assert_close(y1, y2)


@requires_gpu
@requires_quack
def test_cute_rms_norm():
    dim = 64
    norm = CuTeRMSNorm(size=dim, init_device="cuda")
    norm.compile()
    ref_norm = RMSNorm(size=dim, init_device="cuda")

    x = torch.randn(4, dim, requires_grad=True, device="cuda", dtype=torch.bfloat16)
    x_ref = x.detach().clone().requires_grad_(True)
    y = norm(x)
    y_ref = ref_norm(x_ref)
    torch.testing.assert_close(y, y_ref)

    y.sum().backward()
    y_ref.sum().backward()
    assert x.grad is not None
    assert x_ref.grad is not None
    torch.testing.assert_close(x.grad, x_ref.grad)


def test_nemotron_rms_norm_matches_reference():
    dim = 64
    norm = LayerNormConfig(name=LayerNormType.nemotron_rms, eps=1e-5, bias=False).build(size=dim)
    assert isinstance(norm, NemotronRMSNorm)

    x = torch.randn(4, dim)
    with torch.no_grad():
        norm.weight.normal_()

    # Reference: fp32 variance, cast back to input dtype, then affine weight.
    ref = x.to(torch.float32)
    ref = ref * torch.rsqrt(ref.pow(2).mean(-1, keepdim=True) + norm.eps)
    ref = norm.weight * ref.to(x.dtype)

    torch.testing.assert_close(norm(x), ref)


def test_layer_norm_builder_config():
    norm = LayerNormConfig(name=LayerNormType.l2_norm).build(size=1024)
    assert isinstance(norm, L2Norm)

    with pytest.raises(OLMoConfigurationError):
        LayerNormConfig(name=LayerNormType.l2_norm, elementwise_affine=True).build(size=1024)


@pytest.mark.parametrize("norm_type", [LayerNormType.default, LayerNormType.rms])
@pytest.mark.parametrize("bias", [False, True], ids=["no-bias", "bias"])
@pytest.mark.parametrize("weight_shape", [None, (4, 16)], ids=["shared", "per-head"])
def test_layer_norm_output_scale(norm_type: LayerNormType, bias: bool, weight_shape):
    torch.manual_seed(0)
    config = LayerNormConfig(name=norm_type, bias=bias)
    norm = config.build(size=16, weight_shape=weight_shape)
    with torch.no_grad():
        norm.weight.normal_()
        if norm.bias is not None:
            norm.bias.normal_()
    scaled = config.build(size=16, weight_shape=weight_shape)
    scaled.load_state_dict(norm.state_dict())
    assert scaled.supports_output_scale()
    scaled.output_scale = 0.3

    x = torch.randn(2, 4, 16)
    y, y_scaled = norm(x), scaled(x)
    torch.testing.assert_close(y_scaled, 0.3 * y)
    y.square().sum().backward()
    y_scaled.square().sum().backward()
    # d/dw of (c * y)^2 is c^2 times d/dw of y^2.
    torch.testing.assert_close(scaled.weight.grad, 0.09 * norm.weight.grad)


def test_layer_norm_output_scale_support():
    assert not RMSNorm(size=16, elementwise_affine=False).supports_output_scale()
    assert not NemotronRMSNorm(size=16).supports_output_scale()
    assert not L2Norm(size=16).supports_output_scale()
