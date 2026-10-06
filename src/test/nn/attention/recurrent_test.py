import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.attention import (
    AttentionConfig,
    GatedDeltaNetConfig,
    NemotronMamba2Config,
)
from olmo_core.testing import requires_gpu
from olmo_core.testing.utils import requires_fla
from olmo_core.utils import seed_all


@requires_fla
@pytest.mark.parametrize(
    "recurrent_config",
    [
        pytest.param(GatedDeltaNetConfig(n_heads=8), id="default"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, n_v_heads=16), id="GVA"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, head_dim=32), id="head_dim=32"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, expand_v=1.0), id="expand_v=1.0"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, conv_size=8, conv_bias=True), id="conv_bias"),
        pytest.param(
            GatedDeltaNetConfig(n_heads=8, allow_neg_eigval=False), id="allow_neg_eigval=False"
        ),
    ],
)
def test_gated_delta_net_config_num_params(recurrent_config: GatedDeltaNetConfig):
    d_model = 512
    module = recurrent_config.build(d_model, layer_idx=0, n_layers=12, init_device="meta")

    # Make sure the estimated number of params matches the actual number of params.
    n_params = sum(p.numel() for p in module.parameters())
    assert recurrent_config.num_params(d_model) == n_params


@requires_fla
@requires_gpu
def test_gated_delta_net_fwd_bwd():
    device = "cuda"
    dtype = torch.bfloat16

    d_model, seq_len, batch_size = 256, 32, 2

    config = GatedDeltaNetConfig(n_heads=8)
    module = config.build(d_model, layer_idx=0, n_layers=12, init_device=device)

    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

    with torch.autocast(device_type=device, dtype=dtype):
        y = module(x)
        assert y.shape == x.shape

        loss = y.sum()
        loss.backward()
    assert x.grad is not None


@requires_fla
def test_gated_delta_net_num_flops_per_token():
    d_model, n_heads, seq_len = 256, 2, 8192

    gdn = GatedDeltaNetConfig(n_heads=n_heads).build(
        d_model, layer_idx=0, n_layers=1, init_device="meta"
    )
    attn = AttentionConfig(n_heads=n_heads).build(
        d_model, layer_idx=0, n_layers=1, init_device="meta"
    )

    # At long sequence lengths, recurrent layers use fewer FLOPs than quadratic attention.
    gdn_flops = gdn.num_flops_per_token(seq_len)
    attn_flops = attn.num_flops_per_token(seq_len)  # type: ignore
    # Training FLOPs apply a factor of 3 (forward + dgrad + wgrad): the base
    # 2-ops-per-MAC counts become 6, 6, and 24 respectively.
    linear_flops = 6 * sum(
        m.weight.numel() for m in (gdn.w_q, gdn.w_k, gdn.w_v, gdn.w_a, gdn.w_b, gdn.w_g, gdn.w_out)
    )
    conv_flops = 6 * gdn.conv_size * (gdn.key_dim + gdn.key_dim + gdn.value_dim)
    recurrent_flops = 24 * gdn.n_v_heads * gdn.head_k_dim * gdn.head_v_dim

    assert gdn_flops == linear_flops + conv_flops + recurrent_flops
    assert 0 < gdn_flops < attn_flops


def _small_nemotron_mamba2_config() -> NemotronMamba2Config:
    return NemotronMamba2Config(
        mamba_num_heads=4,
        mamba_head_dim=16,
        n_groups=1,
        ssm_state_size=16,
        chunk_size=8,
        dtype=DType.float32,
    )


def test_nemotron_mamba2_num_params():
    d_model = 64
    config = _small_nemotron_mamba2_config()
    module = config.build(d_model, layer_idx=0, n_layers=2, init_device="meta")
    actual = sum(p.numel() for p in module.parameters())
    assert config.num_params(d_model) == actual


def test_nemotron_mamba2_forward_runs_on_cpu():
    seed_all(0)
    d_model = 64
    module = _small_nemotron_mamba2_config().build(d_model, layer_idx=0, n_layers=2)
    x = torch.randn(2, 12, d_model)
    with torch.no_grad():
        out = module(x)
    assert out.shape == (2, 12, d_model)


def test_nemotron_mamba2_dt_bias_initialized_within_configured_range():
    seed_all(0)
    config = NemotronMamba2Config(
        mamba_num_heads=512,
        mamba_head_dim=8,
        n_groups=1,
        ssm_state_size=16,
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_floor=0.0001,
        dtype=DType.float32,
    )
    # init_weights drives the seeded generator; without it the constructor's reset_parameters uses
    # the global RNG (seeded by seed_all above), which is fine for the range check.
    module = config.build(64, layer_idx=0, n_layers=1)
    # softplus(dt_bias) must recover effective timesteps within the configured range, not the
    # softplus(1) == 1.31 that a plain ones-init would give.
    dt = torch.nn.functional.softplus(module.dt_bias.float())
    assert dt.min() >= config.time_step_floor - 1e-6
    assert dt.max() <= config.time_step_max + 1e-4


def test_nemotron_mamba2_dt_bias_uses_supplied_generator_not_global_rng():
    from olmo_core.nn.transformer.init import InitMethod

    config = _small_nemotron_mamba2_config()

    def init_dt_bias(global_seed: int) -> torch.Tensor:
        # Perturb the ambient global RNG differently each time.
        torch.manual_seed(global_seed)
        module = config.build(64, layer_idx=0, n_layers=2)
        generator = torch.Generator()
        generator.manual_seed(2026)
        module.init_weights(
            init_method=InitMethod.normal,
            d_model=64,
            block_idx=0,
            num_blocks=2,
            generator=generator,
        )
        return module.dt_bias.detach().clone()

    # An identical supplied generator must produce identical dt_bias regardless of global RNG state.
    torch.testing.assert_close(init_dt_bias(1), init_dt_bias(999))
