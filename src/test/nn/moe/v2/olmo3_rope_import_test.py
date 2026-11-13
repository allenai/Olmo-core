"""RoPE import contracts, including numerical parity with the bundled HF model."""

import math
from copy import deepcopy
from typing import Any

import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionType,
    KimiDeltaAttentionConfig,
)
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import (
    Olmo3MoeForCausalLM,
    Olmo3MoeRotaryEmbedding,
)
from olmo_core.nn.moe.v2.olmo3 import (
    build_olmo3_moe_config_from_hf_config,
    load_olmo3_moe_hf_state,
)
from olmo_core.nn.rope import (
    PIRoPEScalingConfig,
    RoPEScalingConfig,
    StepwiseRoPEScalingConfig,
    YaRNRoPEScalingConfig,
)


def _config(**kwargs):
    args: dict[str, Any] = dict(
        vocab_size=32,
        hidden_size=32,
        attention_hidden_size=32,
        head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        dense_layers_indices=[0, 1],
        dense_mlp_intermediate_size=24,
        dense_layers_use_shared_expert=True,
        moe_intermediate_size=12,
        shared_expert_intermediate_size=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        use_head_qk_norm=True,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=4,
        max_position_embeddings=32768,
        use_cache=False,
        pad_token_id=None,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
    )
    args.update(kwargs)
    return Olmo3MoeConfig(**args)


def _parameters(kind, theta=500000.0):
    params = dict(rope_type=kind, rope_theta=theta)
    if kind != "default":
        params["factor"] = 4.0
    if kind in ("llama3", "yarn"):
        params["original_max_position_embeddings"] = 8192
    if kind == "llama3":
        params.update(low_freq_factor=1.0, high_freq_factor=4.0)
    return params


@pytest.mark.parametrize("kind", ["default", "linear", "llama3", "yarn"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("representation", ["object", "dict", "legacy", "legacy_type"])
def test_rope_import_matches_hf_and_preserves_input(kind, nested, representation):
    params = _parameters(kind)
    if nested:
        params = {"full_attention": params, "sliding_attention": _parameters("default", 10000.0)}
    hf = _config(rope_parameters=params)
    source = hf if representation == "object" else hf.to_dict()
    if representation.startswith("legacy"):
        source["rope_scaling"] = source.pop("rope_parameters")
        if representation == "legacy_type":
            settings = source["rope_scaling"].values() if nested else [source["rope_scaling"]]
            for settings_for_type in settings:
                settings_for_type["type"] = settings_for_type.pop("rope_type")
    before = deepcopy(source.to_dict() if representation == "object" else source)
    config = build_olmo3_moe_config_from_hf_config(source)
    after = source.to_dict() if representation == "object" else source
    assert after == before
    for layer_type, block in zip(hf.layer_types, config.resolved_block_configs):
        reference = Olmo3MoeRotaryEmbedding(hf, layer_type=layer_type if nested else None)
        rope = block.sequence_mixer.rope
        assert rope.theta == (10000 if nested and layer_type == "sliding_attention" else 500000)
        native = rope.build(hf.head_dim)
        buffers = native.get_buffers(257, torch.device("cpu"))
        cos, sin = reference(torch.zeros(1, 257, hf.head_dim), torch.arange(257)[None])
        torch.testing.assert_close(buffers.pos_cos, cos[0], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(buffers.pos_sin, sin[0], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("kind", ["linear", "llama3", "yarn"])
@pytest.mark.parametrize("theta,head_dim,old_context", [(10000, 64, 8192), (500000, 128, 32768)])
def test_scaled_rope_at_long_positions(kind, theta, head_dim, old_context):
    params = _parameters(kind, theta)
    if kind != "linear":
        params["original_max_position_embeddings"] = old_context
    if kind == "llama3":
        params.update(low_freq_factor=2.0, high_freq_factor=8.0)
    if kind == "yarn":
        params.update(beta_fast=16, beta_slow=2)
    hf = _config(head_dim=head_dim, rope_parameters=params, max_position_embeddings=old_context * 4)
    config = build_olmo3_moe_config_from_hf_config(hf)
    rope = config.resolved_block_configs[0].sequence_mixer.rope
    scaling_types: dict[str, type[RoPEScalingConfig]] = {
        "linear": PIRoPEScalingConfig,
        "llama3": StepwiseRoPEScalingConfig,
        "yarn": YaRNRoPEScalingConfig,
    }
    assert isinstance(rope.scaling, scaling_types[kind])
    reference = Olmo3MoeRotaryEmbedding(hf)
    inv_freq, amplitude = rope.scaling.compute_scaled_inv_freq(theta, head_dim, torch.device("cpu"))
    torch.testing.assert_close(inv_freq, reference.inv_freq, rtol=1e-6, atol=1e-8)
    assert amplitude == reference.attention_scaling
    positions = torch.tensor([[0, 127, 8192, 32768]])
    cos, sin = reference(torch.zeros(1, 4, head_dim), positions)
    buffers = rope.build(head_dim).get_buffers(32769, torch.device("cpu"))
    torch.testing.assert_close(buffers.pos_cos[positions[0]], cos[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(buffers.pos_sin[positions[0]], sin[0], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("theta", [10000, 500000.0, 123456.5])
def test_legacy_top_level_theta_and_default(theta):
    source = _config(rope_theta=theta).to_dict()
    source.pop("rope_parameters", None)
    source.pop("rope_scaling", None)
    config = build_olmo3_moe_config_from_hf_config(source)
    assert all(block.sequence_mixer.rope.theta == theta for block in config.resolved_block_configs)
    source.pop("rope_theta")
    config = build_olmo3_moe_config_from_hf_config(source)
    assert all(block.sequence_mixer.rope.theta == 10000 for block in config.resolved_block_configs)


@pytest.mark.parametrize("empty_parameters", [{}, None])
def test_modern_parameters_take_precedence_over_legacy_fields(empty_parameters):
    source = _config().to_dict()
    source.update(
        rope_theta=12345,
        rope_scaling={"type": "dynamic", "factor": 8.0},
        rope_parameters=_parameters("linear"),
    )
    config = build_olmo3_moe_config_from_hf_config(source)
    rope = config.resolved_block_configs[0].sequence_mixer.rope
    assert rope.theta == 500000
    assert isinstance(rope.scaling, PIRoPEScalingConfig)
    # An explicitly empty modern mapping selects default RoPE, not legacy scaling.
    source["rope_parameters"] = empty_parameters
    rope = (
        build_olmo3_moe_config_from_hf_config(source).resolved_block_configs[0].sequence_mixer.rope
    )
    assert rope.theta == 12345 and rope.scaling is None


@pytest.mark.parametrize("nested", [False, True])
def test_original_context_fallbacks_match_hf(nested):
    params = _parameters("yarn")
    del params["original_max_position_embeddings"]
    if nested:
        params = {
            layer_type: deepcopy(params) for layer_type in ["full_attention", "sliding_attention"]
        }
    raw = _config(max_position_embeddings=8192).to_dict()
    raw.update(rope_parameters=params, original_max_position_embeddings=4096)
    hf = Olmo3MoeConfig(**deepcopy(raw))
    config = build_olmo3_moe_config_from_hf_config(raw)
    for layer_type, block in zip(hf.layer_types, config.resolved_block_configs):
        reference = Olmo3MoeRotaryEmbedding(hf, layer_type=layer_type if nested else None)
        rope = block.sequence_mixer.rope
        freq, amp = rope.scaling.compute_scaled_inv_freq(
            rope.theta, hf.head_dim, torch.device("cpu")
        )
        torch.testing.assert_close(
            freq, getattr(reference, f"{layer_type}_inv_freq" if nested else "inv_freq")
        )
        assert amp == reference.attention_scaling


@pytest.mark.parametrize("kind", ["dynamic", "longrope", "proportional", "unknown"])
@pytest.mark.parametrize("nested", [False, True])
def test_unsupported_scaling_is_never_silently_discarded(kind, nested):
    source = _config().to_dict()
    params = {"rope_type": kind, "factor": 2.0, "rope_theta": 500000}
    source["rope_parameters"] = (
        {"full_attention": params, "sliding_attention": _parameters("default")}
        if nested
        else params
    )
    with pytest.raises(NotImplementedError, match="Unsupported Olmo3Moe RoPE type"):
        build_olmo3_moe_config_from_hf_config(source)


@pytest.mark.parametrize(
    "parameters,exception,match",
    [
        ([], ValueError, "mapping"),
        ({"full_attention": _parameters("default")}, ValueError, "Missing RoPE"),
        (
            {"full_attention": _parameters("default"), "rope_theta": 500000},
            ValueError,
            "only configured layer types",
        ),
        ({"rope_type": "default", "type": "linear"}, ValueError, "Conflicting"),
        ({"rope_type": "default", "partial_rotary_factor": 0.5}, NotImplementedError, "Partial"),
        (
            {"rope_type": "linear", "factor": 2.0, "mrope_section": [1, 1, 2]},
            NotImplementedError,
            "Unsupported linear RoPE parameters",
        ),
        ({"rope_type": "linear"}, ValueError, "factor"),
        (
            {"rope_type": "llama3", "factor": 2.0, "low_freq_factor": 4, "high_freq_factor": 1},
            ValueError,
            "high_freq_factor",
        ),
        (
            {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": -1},
            ValueError,
            "original_max_position_embeddings",
        ),
        (
            {"rope_type": "yarn", "factor": 2.0, "beta_fast": 1, "beta_slow": 2},
            ValueError,
            "beta_fast",
        ),
    ],
)
def test_malformed_or_unrepresentable_parameters(parameters, exception, match):
    source = _config().to_dict()
    source["rope_parameters"] = parameters
    with pytest.raises(exception, match=match):
        build_olmo3_moe_config_from_hf_config(source)


@pytest.mark.parametrize("field", ["rope_theta", "factor", "low_freq_factor", "high_freq_factor"])
@pytest.mark.parametrize("value", [0, -1, True, "2", float("inf"), float("nan"), None])
def test_invalid_numeric_parameters(field, value):
    source = _config().to_dict()
    source["rope_parameters"] = _parameters("llama3") | {field: value}
    with pytest.raises(ValueError, match=field):
        build_olmo3_moe_config_from_hf_config(source)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"attention_factor": 1.7}, "attention_factor"),
        ({"mscale": 1.0, "mscale_all_dim": 1.0}, "attention_factor"),
        ({"truncate": False}, "truncate"),
        ({"beta_fast": 16.5}, "integral"),
        ({"rope_theta": 10000, "original_max_position_embeddings": 131072}, "correction range"),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_incompatible_yarn_is_rejected(overrides, match, nested):
    source = _config(head_dim=128).to_dict()
    params = _parameters("yarn") | overrides
    source["rope_parameters"] = (
        {"full_attention": params, "sliding_attention": _parameters("default")}
        if nested
        else params
    )
    with pytest.raises(NotImplementedError, match=match):
        build_olmo3_moe_config_from_hf_config(source)


def test_explicit_compatible_yarn_amplitude_and_implicit_factor():
    source = _config().to_dict()
    source["rope_parameters"] = _parameters("yarn") | {
        "factor": None,
        "attention_factor": 1 + 0.1 * math.log(4),
    }
    rope = (
        build_olmo3_moe_config_from_hf_config(source).resolved_block_configs[0].sequence_mixer.rope
    )
    assert rope.scaling.factor == 4
    assert (
        rope.scaling.get_attention_rescale_factor() == source["rope_parameters"]["attention_factor"]
    )


@pytest.mark.parametrize("disabled", [False, True])
def test_no_rope_and_linear_layers_do_not_interpret_unused_parameters(disabled):
    source = _config(
        use_rope=not disabled, layer_types=["linear_attention", "full_attention"]
    ).to_dict()
    source["rope_parameters"] = {
        "linear_attention": {"rope_type": "unknown"},
        "full_attention": {"rope_type": "unknown"} if disabled else _parameters("linear"),
    }
    config = build_olmo3_moe_config_from_hf_config(source)
    linear, full = config.resolved_block_configs
    assert isinstance(linear.sequence_mixer, KimiDeltaAttentionConfig)
    if disabled:
        assert full.sequence_mixer.rope is None
    else:
        assert isinstance(full.sequence_mixer.rope.scaling, PIRoPEScalingConfig)


@pytest.mark.parametrize("kind", ["default", "linear", "llama3", "yarn"])
@pytest.mark.parametrize("attention_type", [AttentionType.default, AttentionType.fused_v2])
def test_imported_model_logits_and_gradients_match_hf(kind, attention_type):
    # Dense blocks run on CPU and exercise real attention and imported weights without
    # requiring GPU-only MoE kernels. Different settings per layer also exercise cache isolation.
    torch.manual_seed(42)
    hf = _config(
        rope_parameters={
            "full_attention": _parameters(kind),
            "sliding_attention": _parameters("default", 10000.0),
        }
    )
    hf._attn_implementation = "eager"
    reference = Olmo3MoeForCausalLM(hf).eval()
    config = build_olmo3_moe_config_from_hf_config(
        hf,
        dtype=DType.float32,
        attention_backend=AttentionBackendName.torch,
        attention_type=attention_type,
    )
    native = config.build(init_device="cpu").eval()
    load_olmo3_moe_hf_state(native, hf, reference.state_dict())
    input_ids = torch.randint(0, hf.vocab_size, (2, 33))
    expected = reference(input_ids, use_cache=False).logits
    actual = native(input_ids)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for native_parameter, hf_parameter in (
        (native.embeddings.weight, reference.model.embed_tokens.weight),
        (native.lm_head.w_out.weight, reference.lm_head.weight),
        (
            native.blocks["0"].attention.w_out.weight,
            reference.model.layers[0].self_attn.o_proj.weight,
        ),
    ):
        assert native_parameter.grad is not None and hf_parameter.grad is not None
        torch.testing.assert_close(native_parameter.grad, hf_parameter.grad, rtol=1e-4, atol=1e-6)


def test_layer_type_settings_apply_to_dense_and_routed_blocks():
    hf = _config(
        num_hidden_layers=3,
        dense_layers_indices=[0],
        layer_types=["full_attention", "sliding_attention", "full_attention"],
        rope_parameters={
            "full_attention": _parameters("linear"),
            "sliding_attention": _parameters("default", 10000),
        },
    )
    blocks = build_olmo3_moe_config_from_hf_config(hf).resolved_block_configs
    assert [block.sequence_mixer.rope.theta for block in blocks] == [500000, 10000, 500000]
    assert [type(block.sequence_mixer.rope.scaling) for block in blocks] == [
        PIRoPEScalingConfig,
        type(None),
        PIRoPEScalingConfig,
    ]
    assert isinstance(blocks[0], OLMoDDPTransformerBlockConfig)
    assert isinstance(blocks[2], OLMoDDPTransformerBlockConfig)
    assert blocks[0].routed_experts is None
    assert blocks[2].routed_experts is not None


def test_null_yarn_beta_values_use_hf_defaults():
    hf = _config(rope_parameters=_parameters("yarn") | {"beta_fast": None, "beta_slow": None})
    rope = build_olmo3_moe_config_from_hf_config(hf).resolved_block_configs[0].sequence_mixer.rope
    reference = Olmo3MoeRotaryEmbedding(hf)
    inv_freq, amplitude = rope.scaling.compute_scaled_inv_freq(
        rope.theta, hf.head_dim, torch.device("cpu")
    )
    torch.testing.assert_close(inv_freq, reference.inv_freq)
    assert amplitude == reference.attention_scaling


@pytest.mark.parametrize("nested", [False, True])
def test_missing_parameter_theta_uses_legacy_theta_without_mutation(nested):
    source = _config(rope_theta=123456.5).to_dict()
    params = {"rope_type": "linear", "factor": 2.0}
    source["rope_parameters"] = (
        {layer_type: deepcopy(params) for layer_type in source["layer_types"]} if nested else params
    )
    before = deepcopy(source)
    blocks = build_olmo3_moe_config_from_hf_config(source).resolved_block_configs
    assert all(block.sequence_mixer.rope.theta == 123456.5 for block in blocks)
    assert source == before


@pytest.mark.parametrize("original_context", [1, 131072])
def test_yarn_rejects_both_incompatible_ramp_boundaries(original_context):
    source = _config(head_dim=128).to_dict()
    source["rope_parameters"] = _parameters("yarn", 10000) | {
        "original_max_position_embeddings": original_context
    }
    with pytest.raises(NotImplementedError, match="correction range"):
        build_olmo3_moe_config_from_hf_config(source)


def test_top_level_partial_rope_is_not_silently_dropped():
    source = _config().to_dict()
    source.update(
        partial_rotary_factor=0.5,
        rope_parameters=_parameters("linear") | {"partial_rotary_factor": 1.0},
    )
    with pytest.raises(NotImplementedError, match="Partial"):
        build_olmo3_moe_config_from_hf_config(source)


@pytest.mark.parametrize("factor", [1.0, 0.5])
def test_linear_factor_identity_or_invalid_compression(factor):
    source = _config().to_dict()
    source["rope_parameters"] = _parameters("linear") | {"factor": factor}
    if factor < 1:
        with pytest.raises(ValueError, match="at least 1"):
            build_olmo3_moe_config_from_hf_config(source)
    else:
        rope = (
            build_olmo3_moe_config_from_hf_config(source)
            .resolved_block_configs[0]
            .sequence_mixer.rope
        )
        reference = Olmo3MoeRotaryEmbedding(_config(rope_parameters=_parameters("default")))
        frequency, amplitude = rope.scaling.compute_scaled_inv_freq(
            rope.theta, source["head_dim"], torch.device("cpu")
        )
        torch.testing.assert_close(frequency, reference.inv_freq, rtol=0, atol=0)
        assert amplitude == 1.0
