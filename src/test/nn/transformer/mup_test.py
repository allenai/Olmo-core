from typing import Optional

import pytest
import torch
import torch.nn as nn

from olmo_core.config import DType
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
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe import LatentMoEConfig, MoEConfig
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.transformer import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
    TransformerType,
)
from olmo_core.nn.transformer.init import InitMethod, nanochat_std
from olmo_core.nn.transformer.mup import ParamKind, ParamShape, named_param_shapes

VECTOR = ParamShape(ParamKind.vector)


def _matrix(fan_in: int, fan_out: int) -> ParamShape:
    return ParamShape(ParamKind.matrix, fan_in, fan_out)


def _dense_model(d_model: int = 256, vocab_size: int = 1000, **kwargs) -> nn.Module:
    return TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=2,
        n_heads=d_model // 64,
        n_kv_heads=d_model // 256,
        feed_forward=FeedForwardConfig(hidden_size=d_model * 4, bias=False),
        **kwargs,
    ).build(init_device="meta")


def _moe_model(latent_dim: Optional[int], d_model: int = 256, num_experts: int = 8):
    return TransformerConfig(
        name=TransformerType.moe,
        d_model=d_model,
        n_layers=2,
        vocab_size=1000,
        init_method=InitMethod.nanochat,
        block=TransformerBlockConfig(
            name=TransformerBlockType.moe,
            sequence_mixer=AttentionConfig(n_heads=d_model // 64),
            feed_forward_moe=MoEConfig(
                num_experts=num_experts,
                hidden_size=d_model * 2,
                shared_mlp=FeedForwardConfig(hidden_size=d_model * 4, bias=False),
                latent_moe=None if latent_dim is None else LatentMoEConfig(latent_dim=latent_dim),
            ),
            layer_norm=LayerNormConfig(name=LayerNormType.rms, bias=False),
        ),
        lm_head=LMHeadConfig(bias=False),
    ).build(init_device="cpu")


def _ddp_block(latent_dim: Optional[int], d_model: int = 64):
    dtype = DType.float32
    routed_dim = d_model if latent_dim is None else latent_dim
    norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False, dtype=dtype)
    config = OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        attention=AttentionConfig(
            name=AttentionType.default,
            n_heads=4,
            bias=False,
            use_flash=False,
            qk_norm=norm,
            use_head_qk_norm=True,
            qk_norm_per_head_gains=True,
            dtype=dtype,
        ),
        routed_experts=RoutedExpertsConfig(
            d_model=routed_dim, hidden_size=128, num_experts=4, bias=False, dtype=dtype
        ),
        routed_experts_router=MoERouterConfigV2(
            d_model=d_model, num_experts=4, top_k=2, dtype=dtype
        ),
        shared_experts=SharedExpertsConfig(
            d_model=d_model, hidden_size=32, num_experts=1, bias=False, dtype=dtype
        ),
        layer_norm=norm,
        use_peri_norm=True,
        latent_moe=None if latent_dim is None else LatentMoEConfig(latent_dim=latent_dim),
    )
    return config.build(d_model=d_model, block_idx=0, n_layers=1, init_device="cpu")


def _assert_covers_named_parameters(model: nn.Module, shapes):
    assert list(shapes) == [name for name, _ in model.named_parameters()]


@pytest.mark.parametrize(
    "granularity", [GateGranularity.headwise, GateGranularity.elementwise], ids=lambda g: g.value
)
def test_named_param_shapes_dense(granularity):
    d_model, vocab_size = 256, 1000
    model = _dense_model(d_model, vocab_size, gate=GateConfig(granularity=granularity))
    shapes = named_param_shapes(model)
    _assert_covers_named_parameters(model, shapes)

    attention = model.blocks["0"].attention
    assert shapes["embeddings.weight"] == ParamShape(ParamKind.embedding, vocab_size, d_model)
    assert shapes["lm_head.w_out.weight"] == _matrix(d_model, vocab_size)
    assert shapes["lm_head.norm.weight"] == VECTOR
    for name in ("w_q", "w_k", "w_v", "w_g", "w_out"):
        linear = getattr(attention, name)
        assert shapes[f"blocks.0.attention.{name}.weight"] == _matrix(
            linear.in_features, linear.out_features
        )
    assert shapes["blocks.0.attention.w_k.weight"].fan_out == 64
    assert shapes["blocks.0.feed_forward.w1.weight"] == _matrix(d_model, 4 * d_model)
    assert shapes["blocks.0.feed_forward.w2.weight"] == _matrix(4 * d_model, d_model)
    assert shapes["blocks.0.attention_norm.weight"] == VECTOR


def test_named_param_shapes_tied_embeddings():
    model = _dense_model(tie_word_embeddings=True)
    shapes = named_param_shapes(model)
    _assert_covers_named_parameters(model, shapes)
    assert shapes["embeddings.weight"].kind == ParamKind.embedding
    assert "lm_head.w_out.weight" not in shapes


def test_named_param_shapes_keeps_wrapper_prefix():
    class Wrapper(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

    model = Wrapper(_dense_model())
    shapes = named_param_shapes(model)
    _assert_covers_named_parameters(model, shapes)
    assert shapes["module.embeddings.weight"].kind == ParamKind.embedding


@pytest.mark.parametrize("latent_dim", [None, 64], ids=["no-latent", "latent"])
def test_named_param_shapes_moe(latent_dim: Optional[int]):
    d_model, num_experts = 256, 8
    model = _moe_model(latent_dim, d_model, num_experts)
    shapes = named_param_shapes(model)
    _assert_covers_named_parameters(model, shapes)

    expert_dim = d_model if latent_dim is None else latent_dim
    hidden = 2 * d_model
    prefix = "blocks.0.feed_forward_moe"
    assert shapes[f"{prefix}.router.weight"] == _matrix(d_model, num_experts)
    assert shapes[f"{prefix}.experts.mlp.w1"] == _matrix(expert_dim, hidden)
    assert shapes[f"{prefix}.experts.mlp.w2"] == _matrix(hidden, expert_dim)
    assert shapes[f"{prefix}.experts.mlp.w3"] == _matrix(expert_dim, hidden)
    assert shapes[f"{prefix}.shared_mlp.w2.weight"] == _matrix(4 * d_model, d_model)
    if latent_dim is not None:
        assert shapes[f"{prefix}.latent_down_proj.weight"] == _matrix(d_model, latent_dim)
        assert shapes[f"{prefix}.latent_up_proj.weight"] == _matrix(latent_dim, d_model)


@pytest.mark.parametrize("latent_dim", [None, 16], ids=["no-latent", "latent"])
def test_named_param_shapes_ddp_block(latent_dim: Optional[int]):
    d_model = 64
    block = _ddp_block(latent_dim, d_model)
    shapes = named_param_shapes(block)
    _assert_covers_named_parameters(block, shapes)

    routed_dim = d_model if latent_dim is None else latent_dim
    assert shapes["routed_experts_router.weight"] == _matrix(d_model, 4)
    assert shapes["routed_experts.w_up_gate"] == _matrix(routed_dim, 128)
    assert shapes["routed_experts.w_down"] == _matrix(128, routed_dim)
    assert shapes["shared_experts.w_up_gate"] == _matrix(d_model, 32)
    assert shapes["shared_experts.w_down"] == _matrix(32, d_model)
    # Per-head QK-norm gains are 2-D but element-wise.
    assert block.attention.q_norm.weight.ndim == 2
    assert shapes["attention.q_norm.weight"] == VECTOR


def test_named_param_shapes_fused_attention_v2():
    attention = FusedAttentionV2(
        d_model=256, n_heads=4, n_kv_heads=1, bias=False, backend=AttentionBackendName.torch
    )
    shapes = named_param_shapes(attention)
    _assert_covers_named_parameters(attention, shapes)
    assert shapes["w_qkv.weight"] == _matrix(256, 256 + 2 * 64)


@pytest.mark.skipif(not _has_fla(), reason="fla not installed")
@pytest.mark.parametrize(
    "mixer_config",
    [
        pytest.param(KimiDeltaAttentionConfig(n_heads=4, head_dim=128), id="kda"),
        pytest.param(GatedDeltaNetConfig(n_heads=4), id="gdn"),
    ],
)
def test_named_param_shapes_linear_attention(mixer_config):
    mixer = mixer_config.build(512, layer_idx=0, n_layers=1, init_device="meta")
    shapes = named_param_shapes(mixer)
    _assert_covers_named_parameters(mixer, shapes)
    for name in ("q_conv1d.weight", "A_log", "dt_bias", "o_norm.weight"):
        assert shapes[name] == VECTOR
    assert shapes["w_b.weight"] == _matrix(512, mixer.w_b.out_features)


@pytest.mark.parametrize(
    "module",
    [
        pytest.param(nn.ParameterDict({"w": nn.Parameter(torch.empty(4, 4))}), id="raw-matrix"),
        pytest.param(nn.Conv1d(4, 8, kernel_size=3), id="non-depthwise-conv"),
    ],
)
def test_named_param_shapes_raises_for_unknown_matrices(module):
    with pytest.raises(NotImplementedError, match="muP has no rule"):
        named_param_shapes(module)


def _assert_matrices_match_nanochat_init(module: nn.Module):
    shapes = named_param_shapes(module)
    for name, param in module.named_parameters():
        shape = shapes[name]
        if shape.kind != ParamKind.matrix:
            continue
        assert shape.fan_in is not None and shape.fan_out is not None
        expected = nanochat_std(shape.fan_in, shape.fan_out)
        actual = param.float().std().item()
        assert abs(actual - expected) <= 0.15 * expected, f"{name}: {actual} vs {expected}"


def test_param_shapes_match_nanochat_init_dense():
    model = _dense_model(vocab_size=128, init_method=InitMethod.nanochat)
    model.init_weights(device=torch.device("cpu"))
    _assert_matrices_match_nanochat_init(model)


@pytest.mark.parametrize("latent_dim", [None, 64], ids=["no-latent", "latent"])
def test_param_shapes_match_nanochat_init_moe(latent_dim: Optional[int]):
    model = _moe_model(latent_dim)
    model.init_weights(device=torch.device("cpu"))
    _assert_matrices_match_nanochat_init(model)


@pytest.mark.parametrize("latent_dim", [None, 16], ids=["no-latent", "latent"])
def test_param_shapes_match_nanochat_init_ddp_block(latent_dim: Optional[int]):
    d_model = 64
    block = _ddp_block(latent_dim, d_model)
    kwargs = dict(d_model=d_model, block_idx=0, num_blocks=1)
    InitMethod.nanochat.init_attention(block.attention, **kwargs)
    InitMethod.nanochat.init_moe_v2(block, **kwargs)
    _assert_matrices_match_nanochat_init(block)
