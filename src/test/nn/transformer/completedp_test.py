"""
Tests for the model side of CompletedP: residual multipliers, init, and parameter roles.
"""

from typing import Dict, Optional

import pytest
import torch
import torch.nn as nn

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    FusedAttentionV2,
    GateConfig,
    GateGranularity,
    KimiDeltaAttentionConfig,
)
from olmo_core.nn.attention.flash_linear_attn_api import has_fla as _has_fla
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from olmo_core.nn.ddp.block import (
    OLMoDDPTransformerBlock,
    OLMoDDPTransformerBlockConfig,
)
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe import LatentMoEConfig
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.transformer import (
    CompletedPConfig,
    OLMoDDPModelConfig,
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
    TransformerType,
)
from olmo_core.nn.transformer.completedp import CompletedPRole, named_completedp_params
from olmo_core.nn.transformer.init import InitMethod
from olmo_core.nn.transformer.mup import ParamKind, named_param_shapes
from olmo_core.optim import CompletedPOptimConfig
from olmo_core.optim.completedp import completedp_multipliers

BASE_WIDTH = 128
BASE_DEPTH = 2
INIT_STD = 0.02
NORM = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False, dtype=DType.float32)


def _dense_config(
    d_model: int = 256,
    n_layers: int = 4,
    completedp: Optional[CompletedPConfig] = None,
    **kwargs,
) -> TransformerConfig:
    return TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=512,
        n_layers=n_layers,
        n_heads=d_model // 64,
        feed_forward=FeedForwardConfig(hidden_size=4 * d_model, bias=False),
        attn_backend=AttentionBackendName.torch,
        completedp=completedp,
        **kwargs,
    )


def _ddp_block_config(
    *, d_model: int = 64, latent_dim: Optional[int] = 32, use_pre_norm: bool = False
) -> OLMoDDPTransformerBlockConfig:
    dtype = DType.float32
    routed_dim = d_model if latent_dim is None else latent_dim
    return OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        sequence_mixer=AttentionConfig(
            name=AttentionType.default,
            n_heads=4,
            bias=False,
            use_flash=False,
            qk_norm=NORM,
            use_head_qk_norm=True,
            dtype=dtype,
        ),
        routed_experts=RoutedExpertsConfig(
            d_model=routed_dim, hidden_size=2 * d_model, num_experts=4, bias=False, dtype=dtype
        ),
        routed_experts_router=MoERouterConfigV2(
            d_model=d_model, num_experts=4, top_k=2, dtype=dtype
        ),
        shared_experts=SharedExpertsConfig(
            d_model=d_model, hidden_size=d_model, num_experts=1, bias=False, dtype=dtype
        ),
        layer_norm=NORM,
        use_peri_norm=not use_pre_norm,
        use_pre_norm=use_pre_norm,
        latent_moe=None if latent_dim is None else LatentMoEConfig(latent_dim=latent_dim),
    )


def _ddp_model_config(
    d_model: int = 64, n_layers: int = 4, completedp: Optional[CompletedPConfig] = None
) -> OLMoDDPModelConfig:
    return OLMoDDPModelConfig(
        name=TransformerType.moe_fused_v2,
        d_model=d_model,
        vocab_size=128,
        n_layers=n_layers,
        block=_ddp_block_config(d_model=d_model),
        block_overrides={0: _ddp_block_config(d_model=d_model, latent_dim=None)},
        embedding_norm=NORM,
        lm_head=LMHeadConfig(layer_norm=NORM, bias=False, dtype=DType.float32),
        recompute_each_block=False,
        completedp=completedp,
    )


def _block(config: TransformerConfig) -> TransformerBlockConfig:
    assert isinstance(config.block, TransformerBlockConfig)
    return config.block


def _completedp(alpha: float = 1.0) -> CompletedPConfig:
    return CompletedPConfig(base_width=BASE_WIDTH, base_depth=BASE_DEPTH, alpha=alpha)


def _expected_init_factor(role: CompletedPRole, fan_in: Optional[int]) -> float:
    if role == CompletedPRole.hidden_matrix:
        assert fan_in is not None
        return (BASE_WIDTH / fan_in) ** 0.5
    if role in (CompletedPRole.output_matrix, CompletedPRole.hidden_readout_matrix):
        assert fan_in is not None
        return BASE_WIDTH / fan_in
    return 1.0


def _init_dense(config: TransformerConfig) -> nn.Module:
    model = config.build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))
    return model


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(base_width=0, base_depth=2), "base_width"),
        (dict(base_width=128, base_depth=0), "base_depth"),
        (dict(base_width=128, base_depth=2, alpha=0.4), "alpha"),
        (dict(base_width=128, base_depth=2, alpha=1.1), "alpha"),
    ],
)
def test_completedp_config_validation(kwargs, match):
    with pytest.raises(OLMoConfigurationError, match=match):
        CompletedPConfig(**kwargs)


@pytest.mark.parametrize("alpha", [1.0, 0.75, 0.5])
@pytest.mark.parametrize("n_layers", [2, 4, 8])
def test_residual_multiplier(alpha: float, n_layers: int):
    expected = (n_layers / BASE_DEPTH) ** -alpha
    assert _completedp(alpha).residual_multiplier(n_layers) == pytest.approx(expected)


@pytest.mark.parametrize(
    "block_name",
    [
        TransformerBlockType.default,
        TransformerBlockType.reordered_norm,
        TransformerBlockType.peri_norm,
    ],
    ids=lambda b: b.value,
)
@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_dense_blocks_get_residual_multipliers(block_name: TransformerBlockType, alpha: float):
    config = _dense_config(
        n_layers=8,
        completedp=_completedp(alpha),
        block_name=block_name,
        block_mods={3: lambda b: b},
    )
    model = config.build(init_device="meta")
    expected = (8 / BASE_DEPTH) ** -alpha
    for block in model.blocks.values():
        assert block.attention_residual_stream.alpha == pytest.approx(expected)
        assert block.feed_forward_residual_stream.alpha == pytest.approx(expected)
    # The user's config is not modified.
    assert _block(config).attention_residual_alpha is None


def test_dense_block_residual_multiplier_scales_branches():
    """With residual multiplier ``c``, each block computes ``x + c * branch(x)``."""
    torch.manual_seed(0)
    plain = _init_dense(_dense_config(n_layers=2))
    scaled = _dense_config(n_layers=8, completedp=_completedp()).build(init_device="cpu")
    scaled.blocks["0"].load_state_dict(plain.blocks["0"].state_dict())
    c = (8 / BASE_DEPTH) ** -1.0

    x = torch.randn(2, 8, 256)
    plain_block, scaled_block = plain.blocks["0"], scaled.blocks["0"]
    attn = plain_block.attention(plain_block.attention_norm(x))
    h = x + c * attn
    expected = h + c * plain_block.feed_forward(plain_block.feed_forward_norm(h))
    torch.testing.assert_close(scaled_block(x), expected)


def test_ddp_model_blocks_get_residual_multipliers():
    model = _ddp_model_config(n_layers=8, completedp=_completedp(0.5)).build(init_device="meta")
    expected = (8 / BASE_DEPTH) ** -0.5
    for block in model.blocks.values():
        assert isinstance(block, OLMoDDPTransformerBlock)
        assert block.attention_residual_alpha == pytest.approx(expected)
        assert block.feed_forward_residual_alpha == pytest.approx(expected)


@pytest.mark.parametrize("use_pre_norm", [False, True], ids=["peri-norm", "pre-norm"])
def test_ddp_block_residual_multiplier_scales_branches(use_pre_norm: bool):
    config = _ddp_block_config(use_pre_norm=use_pre_norm)
    torch.manual_seed(0)
    plain = config.build(d_model=64, block_idx=0, n_layers=1, init_device="cpu")
    config.attention_residual_alpha = 0.25
    config.feed_forward_residual_alpha = 0.5
    scaled = config.build(d_model=64, block_idx=0, n_layers=1, init_device="cpu")
    assert isinstance(plain, OLMoDDPTransformerBlock)
    assert isinstance(scaled, OLMoDDPTransformerBlock)
    InitMethod.normal.init_attention(plain.attention, d_model=64, block_idx=0, num_blocks=1)
    scaled.load_state_dict(plain.state_dict())

    # With a branch output norm, the multiplier is folded into that norm's gains.
    folded = not use_pre_norm
    assert scaled.attention_norm.output_scale == (0.25 if folded else 1.0)
    assert scaled.feed_forward_norm.output_scale == (0.5 if folded else 1.0)
    assert scaled._attention_add_alpha == (1.0 if folded else 0.25)
    assert scaled._feed_forward_add_alpha == (1.0 if folded else 0.5)

    x = torch.randn(2, 8, 64)
    torch.testing.assert_close(scaled._res_norm_attn(x) - x, 0.25 * (plain._res_norm_attn(x) - x))
    mlp_out = torch.randn(2, 8, 64)
    torch.testing.assert_close(
        scaled._res_norm_mlp(x, mlp_out) - x, 0.5 * (plain._res_norm_mlp(x, mlp_out) - x)
    )


def test_completedp_rejects_preset_residual_multiplier():
    config = _dense_config(completedp=_completedp())
    _block(config).attention_residual_alpha = 0.5
    with pytest.raises(OLMoConfigurationError, match="sets the residual multipliers"):
        config.build(init_device="meta")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        pytest.param(dict(init_method=InitMethod.nanochat), "init method", id="init-method"),
        pytest.param(dict(tie_word_embeddings=True), "tied word embeddings", id="tied"),
        pytest.param(
            dict(block_name=TransformerBlockType.default_scaled), "default_scaled", id="block"
        ),
    ],
)
def test_completedp_rejects_unsupported_models(kwargs, match):
    with pytest.raises(OLMoConfigurationError, match=match):
        _dense_config(completedp=_completedp(), **kwargs).build(init_device="meta")


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            TransformerConfig.ngpt_like(
                d_model=256, vocab_size=512, n_layers=2, n_heads=4, completedp=_completedp()
            ),
            id="normalized",
        ),
        pytest.param(
            TransformerConfig.llama_like_moe(
                d_model=256,
                vocab_size=512,
                n_layers=2,
                n_heads=4,
                num_experts=4,
                top_k=2,
                expert_hidden_size=256,
                completedp=_completedp(),
            ),
            id="moe",
        ),
    ],
)
def test_completedp_rejects_unsupported_transformer_types(config: TransformerConfig):
    with pytest.raises(OLMoConfigurationError, match="not supported for transformer type"):
        config.build(init_device="meta")


def test_named_completedp_params_dense():
    model = _dense_config(completedp=_completedp(), embedding_norm=NORM).build(init_device="meta")
    params = named_completedp_params(model)
    assert list(params) == [name for name, _ in model.named_parameters()]
    assert params["embeddings.weight"].role == CompletedPRole.input_embedding
    assert params["embedding_norm.weight"].role == CompletedPRole.input_vector
    assert params["blocks.0.attention.w_q.weight"].role == CompletedPRole.hidden_matrix
    assert params["blocks.0.attention.w_q.weight"].fan_in == 256
    assert params["blocks.0.feed_forward.w2.weight"].fan_in == 1024
    assert params["blocks.0.attention_norm.weight"].role == CompletedPRole.hidden_vector
    assert params["lm_head.w_out.weight"].role == CompletedPRole.output_matrix
    assert params["lm_head.w_out.weight"].fan_in == 256
    assert params["lm_head.norm.weight"].role == CompletedPRole.output_vector


def test_named_completedp_params_keeps_wrapper_prefix():
    class Wrapper(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

    model = Wrapper(_dense_config(completedp=_completedp()).build(init_device="meta"))
    params = named_completedp_params(model)
    assert list(params) == [name for name, _ in model.named_parameters()]
    assert params["module.embeddings.weight"].role == CompletedPRole.input_embedding
    assert params["module.lm_head.w_out.weight"].role == CompletedPRole.output_matrix


@pytest.mark.parametrize(
    "use_head_qk_norm, per_head_gains, expected",
    [
        pytest.param(False, False, CompletedPRole.hidden_vector, id="full-width"),
        pytest.param(True, False, CompletedPRole.hidden_head_shared_vector, id="head-shared"),
        pytest.param(True, True, CompletedPRole.hidden_vector, id="per-head-gains"),
    ],
)
def test_named_completedp_params_qk_norm(
    use_head_qk_norm: bool, per_head_gains: bool, expected: CompletedPRole
):
    config = _dense_config(
        completedp=_completedp(), qk_norm=True, use_head_qk_norm=use_head_qk_norm
    )
    _block(config).sequence_mixer.qk_norm_per_head_gains = per_head_gains  # type: ignore[attr-defined]
    model = config.build(init_device="meta")
    params = named_completedp_params(model)
    for name in ("q_norm", "k_norm"):
        assert params[f"blocks.0.attention.{name}.weight"].role == expected


def test_named_completedp_params_ddp_model():
    model = _ddp_model_config(completedp=_completedp()).build(init_device="meta")
    params = named_completedp_params(model)
    assert list(params) == [name for name, _ in model.named_parameters()]
    roles = {name: p.role for name, p in params.items()}
    assert roles["embedding_norm.weight"] == CompletedPRole.input_vector
    for name in (
        "blocks.1.routed_experts.w_up_gate",
        "blocks.1.routed_experts.w_down",
        "blocks.1.latent_down_proj.weight",
        "blocks.1.latent_up_proj.weight",
        "blocks.0.shared_experts.w_up_gate",
        "blocks.0.shared_experts.w_down",
    ):
        assert roles[name] == CompletedPRole.hidden_matrix, name
    # The router has one output per expert, a size that does not grow with width.
    assert roles["blocks.1.routed_experts_router.weight"] == CompletedPRole.hidden_readout_matrix
    assert params["blocks.1.routed_experts_router.weight"].fan_in == 64
    assert params["blocks.1.routed_experts.w_up_gate"].fan_in == 32
    assert params["blocks.1.routed_experts.w_down"].fan_in == 128
    assert roles["blocks.1.attention.q_norm.weight"] == CompletedPRole.hidden_head_shared_vector
    assert roles["blocks.1.attention_input_norm.weight"] == CompletedPRole.hidden_vector


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_completedp_multipliers_readout_matrices(alpha: float):
    model = _ddp_model_config(n_layers=8, completedp=_completedp(alpha)).build(init_device="meta")
    multipliers = completedp_multipliers(model, CompletedPOptimConfig())
    m_n = r = 64 / BASE_WIDTH
    m_l = 8 / BASE_DEPTH
    router = multipliers["blocks.1.routed_experts_router.weight"]
    assert router["lr"] == pytest.approx(m_l ** (alpha - 1) / r)
    assert router["eps"] == pytest.approx(m_l**-alpha)
    assert router["weight_decay"] == pytest.approx(r)
    # A hidden matrix with the same fan-in differs only in its width-scaled epsilon.
    hidden = multipliers["blocks.1.latent_down_proj.weight"]
    assert hidden["lr"] == pytest.approx(router["lr"])
    assert hidden["weight_decay"] == pytest.approx(router["weight_decay"])
    assert hidden["eps"] == pytest.approx(m_l**-alpha / m_n)


@pytest.mark.skipif(not _has_fla(), reason="fla not installed")
@pytest.mark.parametrize(
    "mixer_config",
    [
        pytest.param(KimiDeltaAttentionConfig(n_heads=4, head_dim=128), id="kda"),
        pytest.param(GatedDeltaNetConfig(n_heads=4), id="gdn"),
    ],
)
def test_named_completedp_params_linear_attention(mixer_config):
    config = _dense_config(d_model=512, completedp=_completedp())
    _block(config).sequence_mixer = mixer_config
    model = config.build(init_device="meta")
    roles = {name: p.role for name, p in named_completedp_params(model).items()}
    assert roles["blocks.0.attention.o_norm.weight"] == CompletedPRole.hidden_head_shared_vector
    for name in ("q_conv1d.weight", "A_log", "dt_bias"):
        assert roles[f"blocks.0.attention.{name}"] == CompletedPRole.hidden_vector
    assert roles["blocks.0.attention.w_out.weight"] == CompletedPRole.hidden_matrix
    if isinstance(mixer_config, KimiDeltaAttentionConfig):
        for name in ("f_proj_1", "g_proj_1"):
            role = roles[f"blocks.0.attention.{name}.weight"]
            assert role == CompletedPRole.hidden_readout_matrix
        for name in ("f_proj_2", "g_proj_2", "w_b"):
            assert roles[f"blocks.0.attention.{name}.weight"] == CompletedPRole.hidden_matrix


@pytest.mark.parametrize("d_model", [128, 256])
def test_completedp_init_scales_normal_init(d_model: int):
    """CompletedP's init equals the ``normal`` init times each parameter's width factor."""
    normal = _init_dense(_dense_config(d_model=d_model, embedding_norm=NORM))
    completedp = _init_dense(
        _dense_config(d_model=d_model, completedp=_completedp(), embedding_norm=NORM)
    )
    params = named_completedp_params(completedp)
    normal_params = dict(normal.named_parameters())
    for name, param in completedp.named_parameters():
        factor = _expected_init_factor(params[name].role, params[name].fan_in)
        expected = normal_params[name].detach().clone()
        if factor != 1.0:
            expected.mul_(factor)
        assert torch.equal(param.detach(), expected), name


def test_completedp_init_stds():
    model = _init_dense(_dense_config(d_model=256, completedp=_completedp()))
    for name, info in named_completedp_params(model).items():
        param = dict(model.named_parameters())[name].detach().float()
        if info.role in (CompletedPRole.hidden_matrix, CompletedPRole.output_matrix):
            expected = INIT_STD * _expected_init_factor(info.role, info.fan_in)
            # Truncation at +/-3 std lowers the realized std by about 1.3%.
            assert param.std().item() == pytest.approx(expected, rel=0.06), name
        elif info.role == CompletedPRole.input_embedding:
            assert param.std().item() == pytest.approx(INIT_STD, rel=0.06)
        else:
            assert torch.all(param == 1.0), name


def test_completedp_init_ddp_model():
    normal = _ddp_model_config().build(init_device="cpu")
    normal.init_weights(device=torch.device("cpu"))
    completedp = _ddp_model_config(completedp=_completedp()).build(init_device="cpu")
    completedp.init_weights(device=torch.device("cpu"))
    params = named_completedp_params(completedp)
    normal_params = dict(normal.named_parameters())
    for name, param in completedp.named_parameters():
        factor = _expected_init_factor(params[name].role, params[name].fan_in)
        expected = normal_params[name].detach().clone()
        if factor != 1.0:
            expected.mul_(factor)
        assert torch.equal(param.detach(), expected), name


def _assert_normal_init_matrices_have_init_std(module: nn.Module, std: float = INIT_STD):
    shapes = named_param_shapes(module)
    stds: Dict[str, float] = {}
    for name, param in module.named_parameters():
        if shapes[name].kind == ParamKind.matrix:
            stds[name] = param.detach().float().std().item()
    assert stds
    for name, actual in stds.items():
        assert actual == pytest.approx(std, rel=0.1), f"{name}: {actual}"


def test_normal_init_precondition_dense():
    """CompletedP rescales the ``normal`` init, which must give every matrix ``init_std``."""
    gate = GateConfig(granularity=GateGranularity.elementwise)
    model = _init_dense(_dense_config(d_model=256, qk_norm=True, gate=gate))
    _assert_normal_init_matrices_have_init_std(model)


@pytest.mark.parametrize("latent_dim", [None, 32], ids=["no-latent", "latent"])
def test_normal_init_precondition_ddp_block(latent_dim: Optional[int]):
    block = _ddp_block_config(latent_dim=latent_dim).build(
        d_model=64, block_idx=0, n_layers=1, init_device="cpu"
    )
    kwargs = dict(d_model=64, block_idx=0, num_blocks=1)
    InitMethod.normal.init_attention(block.attention, **kwargs)
    InitMethod.normal.init_moe_v2(block, **kwargs)
    _assert_normal_init_matrices_have_init_std(block)


def test_normal_init_precondition_fused_attention_v2():
    attention = FusedAttentionV2(
        d_model=256, n_heads=4, n_kv_heads=1, bias=False, backend=AttentionBackendName.torch
    )
    InitMethod.normal.init_attention(attention, d_model=256, block_idx=0, num_blocks=1)
    _assert_normal_init_matrices_have_init_std(attention)


@pytest.mark.skipif(not _has_fla(), reason="fla not installed")
@pytest.mark.parametrize(
    "mixer_config",
    [
        pytest.param(KimiDeltaAttentionConfig(n_heads=4, head_dim=128), id="kda"),
        pytest.param(GatedDeltaNetConfig(n_heads=4), id="gdn"),
    ],
)
def test_normal_init_precondition_linear_attention(mixer_config):
    mixer = mixer_config.build(512, layer_idx=0, n_layers=1, init_device="cpu")
    InitMethod.normal.init_attention(mixer, d_model=512, block_idx=0, num_blocks=1)
    _assert_normal_init_matrices_have_init_std(mixer)
