"""Tests for ``OLMoDDPTransformerBlock`` config accounting and construction."""

import pytest
import torch

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig, AttentionType
from olmo_core.nn.ddp.block import (
    OLMoDDPTransformerBlock,
    OLMoDDPTransformerBlockConfig,
)
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.moe import LatentMoEConfig
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.transformer import TransformerBlockType
from olmo_core.nn.transformer.init import InitMethod, nanochat_std

D_MODEL = 64


def _block_config(
    *, use_peri_norm: bool = False, latent_dim: int | None = None
) -> OLMoDDPTransformerBlockConfig:
    dtype = DType.float32
    routed_dim = D_MODEL if latent_dim is None else latent_dim
    layer_norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False, dtype=dtype)
    return OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        attention=AttentionConfig(
            name=AttentionType.default,
            n_heads=4,
            bias=False,
            backend=AttentionBackendName.torch,
            dtype=dtype,
        ),
        routed_experts=RoutedExpertsConfig(
            d_model=routed_dim, hidden_size=128, num_experts=4, bias=False, dtype=dtype
        ),
        routed_experts_router=MoERouterConfigV2(
            d_model=D_MODEL, num_experts=4, top_k=2, dtype=dtype
        ),
        shared_experts=None,
        layer_norm=layer_norm,
        use_peri_norm=use_peri_norm,
        latent_moe=(LatentMoEConfig(latent_dim=routed_dim) if latent_dim is not None else None),
    )


def _build_block(config: OLMoDDPTransformerBlockConfig) -> OLMoDDPTransformerBlock:
    block = config.build(d_model=D_MODEL, block_idx=0, n_layers=1, init_device="cpu")
    assert isinstance(block, OLMoDDPTransformerBlock)
    return block


@pytest.mark.parametrize("use_peri_norm", [False, True])
def test_block_num_params_matches_built_block(use_peri_norm: bool):
    config = _block_config(use_peri_norm=use_peri_norm)
    block = _build_block(config)
    actual = sum(p.numel() for p in block.parameters())
    assert config.num_params(D_MODEL) == actual


def test_block_num_active_params_below_total():
    # Only top_k of num_experts routed experts are active per token, so active < total.
    config = _block_config()
    assert 0 < config.num_active_params(D_MODEL) < config.num_params(D_MODEL)


def test_latent_block_dimensions_and_param_accounting():
    latent_dim = 16
    config = _block_config(latent_dim=latent_dim)
    block = _build_block(config)

    assert config.num_params(D_MODEL) == sum(p.numel() for p in block.parameters())
    assert block.routed_experts is not None
    assert block.routed_experts.d_model == latent_dim
    assert block.routed_experts_router is not None
    assert block.routed_experts_router.d_model == D_MODEL
    assert block.latent_down_proj is not None
    assert block.latent_down_proj.weight.shape == (latent_dim, D_MODEL)
    assert block.latent_up_proj is not None
    assert block.latent_up_proj.weight.shape == (D_MODEL, latent_dim)

    model_input = torch.randn(2, 3, D_MODEL)
    routed_input = block._prepare_routed_moe_input(model_input)
    assert routed_input.shape == (2, 3, latent_dim)
    assert block._restore_routed_moe_output(routed_input).shape == model_input.shape


def test_latent_block_norm_is_disabled_by_default():
    config = _block_config(latent_dim=16)
    block = _build_block(config)

    assert block.latent_up_proj_input_norm is None


def test_latent_block_enabled_norm_defaults_to_rms_norm():
    config = _block_config(latent_dim=16)
    assert config.latent_moe is not None
    config.latent_moe.up_proj_input_norm_enabled = True
    block = _build_block(config)

    assert block.latent_up_proj_input_norm is not None
    assert config.latent_moe.resolved_up_proj_input_norm().name == LayerNormType.rms
    serialized = config.latent_moe.as_config_dict()
    assert serialized["up_proj_input_norm_enabled"] is True
    assert "up_proj_input_norm" not in serialized
    restored = LatentMoEConfig.from_dict(serialized)
    assert restored.up_proj_input_norm_enabled is True
    assert restored.resolved_up_proj_input_norm().name == LayerNormType.rms
    assert config.num_params(D_MODEL) == sum(p.numel() for p in block.parameters())
    routed_out = torch.randn(2, 3, 16)
    assert block.latent_up_proj is not None
    expected = block.latent_up_proj(block.latent_up_proj_input_norm(routed_out))
    torch.testing.assert_close(block._restore_routed_moe_output(routed_out), expected)


@pytest.mark.parametrize("norm_type", [LayerNormType.default, LayerNormType.l2_norm])
def test_latent_block_accepts_configurable_up_proj_input_norm(norm_type: LayerNormType):
    config = _block_config(latent_dim=16)
    assert config.latent_moe is not None
    config.latent_moe.up_proj_input_norm_enabled = True
    config.latent_moe.up_proj_input_norm = LayerNormConfig(name=norm_type)
    block = _build_block(config)
    assert block.latent_up_proj_input_norm is not None


def test_latent_block_can_disable_up_proj_input_norm():
    config = _block_config(latent_dim=16)
    assert config.latent_moe is not None
    block = _build_block(config)
    assert block.latent_up_proj_input_norm is None

    serialized = config.latent_moe.as_config_dict()
    assert serialized["up_proj_input_norm_enabled"] is False
    restored = LatentMoEConfig.from_dict(serialized)
    assert restored.up_proj_input_norm_enabled is False


@pytest.mark.parametrize("latent_dim", [0, D_MODEL, D_MODEL * 2])
def test_latent_block_rejects_invalid_dimension(latent_dim: int):
    config = _block_config(latent_dim=latent_dim)
    with pytest.raises(OLMoConfigurationError, match="latent_dim"):
        config.build(d_model=D_MODEL, block_idx=0, n_layers=1, init_device="cpu")


def test_latent_block_projection_initialization():
    latent_dim = 16
    block = _build_block(_block_config(latent_dim=latent_dim))
    InitMethod.fan_in.init_moe_v2(
        block,
        d_model=D_MODEL,
        block_idx=0,
        num_blocks=1,
    )

    assert block.latent_down_proj is not None
    assert block.latent_up_proj is not None
    assert block.routed_experts_router is not None
    assert block.routed_experts is not None
    torch.testing.assert_close(
        block.latent_down_proj.weight.std(),
        torch.tensor(D_MODEL**-0.5),
        rtol=0.35,
        atol=0.0,
    )
    torch.testing.assert_close(
        block.latent_up_proj.weight.std(),
        torch.tensor(latent_dim**-0.5),
        rtol=0.35,
        atol=0.0,
    )
    torch.testing.assert_close(
        block.routed_experts_router.weight.std(),
        torch.tensor(D_MODEL**-0.5),
        rtol=0.35,
        atol=0.0,
    )
    torch.testing.assert_close(
        block.routed_experts.w_up_gate.std(),
        torch.tensor(latent_dim**-0.5),
        rtol=0.35,
        atol=0.0,
    )
    torch.testing.assert_close(
        block.routed_experts.w_down.std(),
        torch.tensor(block.routed_experts.hidden_size**-0.5),
        rtol=0.35,
        atol=0.0,
    )


@pytest.mark.parametrize("latent_dim", [None, 16], ids=["no-latent", "latent"])
def test_block_nanochat_initialization(latent_dim: int | None):
    config = _block_config(latent_dim=latent_dim)
    config.shared_experts = SharedExpertsConfig(
        d_model=D_MODEL, hidden_size=32, num_experts=1, bias=False, dtype=DType.float32
    )
    block = _build_block(config)
    InitMethod.nanochat.init_moe_v2(block, d_model=D_MODEL, block_idx=0, num_blocks=1)

    def assert_std(x: torch.Tensor, fan_in: int, fan_out: int):
        torch.testing.assert_close(
            x.std(), torch.tensor(nanochat_std(fan_in, fan_out)), rtol=0.1, atol=0.0
        )

    router = block.routed_experts_router
    routed = block.routed_experts
    shared = block.shared_experts
    assert router is not None and routed is not None and shared is not None
    routed_dim = D_MODEL if latent_dim is None else latent_dim
    assert routed.d_model == routed_dim

    assert_std(router.weight, D_MODEL, router.num_experts)
    assert_std(routed.w_up_gate, routed_dim, routed.hidden_size)
    assert_std(routed.w_down, routed.hidden_size, routed_dim)
    assert_std(shared.w_up_gate, D_MODEL, shared.hidden_size)
    assert_std(shared.w_down, shared.hidden_size, D_MODEL)
    for projection in (block.latent_down_proj, block.latent_up_proj):
        assert (projection is None) == (latent_dim is None)
        if projection is not None:
            assert_std(projection.weight, projection.in_features, projection.out_features)


def test_latent_block_keeps_shared_experts_in_model_space():
    config = _block_config(latent_dim=16)
    config.shared_experts = SharedExpertsConfig(
        d_model=D_MODEL,
        hidden_size=32,
        num_experts=1,
        bias=False,
        dtype=DType.float32,
    )
    block = _build_block(config)
    assert block.shared_experts is not None
    assert block.shared_experts.d_model == D_MODEL
    assert config.num_params(D_MODEL) == sum(p.numel() for p in block.parameters())


def test_non_latent_block_preserves_state_dict_keys():
    block = _build_block(_block_config())
    assert not any("latent_" in key for key in block.state_dict())


def test_latent_block_rejects_routed_config_dimension_mismatch():
    config = _block_config(latent_dim=16)
    assert config.routed_experts is not None
    config.routed_experts.d_model = 8
    with pytest.raises(OLMoConfigurationError, match="routed_experts.d_model"):
        config.build(d_model=D_MODEL, block_idx=0, n_layers=1, init_device="cpu")


def test_latent_block_rejects_model_space_router_dimension_mismatch():
    config = _block_config(latent_dim=16)
    assert config.routed_experts_router is not None
    config.routed_experts_router.d_model = 16
    with pytest.raises(OLMoConfigurationError, match="routed_experts_router.d_model"):
        config.build(d_model=D_MODEL, block_idx=0, n_layers=1, init_device="cpu")


def test_block_flops_positive():
    config = _block_config()
    assert config.flops_per_seq(D_MODEL, seqlen=512) > 0
    assert _build_block(config).num_flops_per_token(seq_len=512) > 0


def test_block_parallelism_disabled_by_default():
    block = _build_block(_block_config())
    assert block.is_moe
    assert not block.ep_enabled
    assert not block.tp_enabled


@pytest.mark.parametrize(
    "attention_alpha, feed_forward_alpha", [(None, None), (0.5, 0.25)], ids=["default", "set"]
)
def test_block_residual_alphas(attention_alpha, feed_forward_alpha):
    config = _block_config(use_peri_norm=True)
    config.attention_residual_alpha = attention_alpha
    config.feed_forward_residual_alpha = feed_forward_alpha
    block = _build_block(config)
    assert block.attention_residual_alpha == (attention_alpha or 1.0)
    assert block.feed_forward_residual_alpha == (feed_forward_alpha or 1.0)
