"""
Roundtrip tests for the ``olmo3moe`` <-> Olmo-core MoE-v2 state-dict conversion.

These exercise the conversion functions directly on synthetic tensor dicts with a stub config, so
they don't require ``transformers`` or a built model — they verify the HF<->Olmo-core mapping is a
faithful bijection (including the fused ``w_up_gate`` split and the mixed dense/MoE layer layout).
"""

import types

import pytest
import torch

from olmo_core.nn.hf.convert import (
    convert_olmo3moe_state_from_hf,
    convert_olmo3moe_state_to_hf,
    iter_olmo3moe_state_to_hf,
)
from olmo_core.nn.hf.convert_checkpoint import _normalize_legacy_latent_moe_config
from olmo_core.testing.utils import requires_fla, requires_gpu, requires_triton


def _fake_config():
    # Distinct hidden sizes (dense/moe/shared) so a transpose or shape slip can't roundtrip by
    # accident. Layer 0 is dense, layer 1 is MoE.
    return types.SimpleNamespace(
        num_hidden_layers=2,
        n_routed_experts=3,
        hidden_size=8,
        moe_intermediate_size=5,
        shared_expert_intermediate_size=4,
        dense_layers_indices=[0],
        embed_norm=True,
        use_peri_ln=False,
    )


def test_legacy_kda_kernel_flag_is_normalized():
    config = {"block": {"sequence_mixer": {"use_cute_kernel": True}}}
    _normalize_legacy_latent_moe_config(config)
    assert config == {"block": {"sequence_mixer": {"use_experimental_kernels": True}}}
    with pytest.raises(ValueError, match="legacy and current"):
        _normalize_legacy_latent_moe_config(
            {"use_cute_kernel": True, "use_experimental_kernels": False}
        )


def _synthetic_hf_state(config):
    torch.manual_seed(0)
    d = config.hidden_size
    e = config.n_routed_experts
    h = config.moe_intermediate_size
    hs = config.shared_expert_intermediate_size
    vocab, dense_h = 6, 7
    dense = set(config.dense_layers_indices)

    hf = {
        "model.embed_tokens.weight": torch.randn(vocab, d),
        "model.norm.weight": torch.randn(d),
        "lm_head.weight": torch.randn(vocab, d),
        "model.embed_norm.weight": torch.randn(d),
    }
    for layer in range(config.num_hidden_layers):
        p = f"model.layers.{layer}."
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            hf[f"{p}self_attn.{proj}.weight"] = torch.randn(d, d)
        for norm in ("q_norm", "k_norm"):
            hf[f"{p}self_attn.{norm}.weight"] = torch.randn(d)
        hf[f"{p}post_attention_layernorm.weight"] = torch.randn(d)
        hf[f"{p}post_feedforward_layernorm.weight"] = torch.randn(d)
        if layer in dense:
            hf[f"{p}mlp.gate_proj.weight"] = torch.randn(dense_h, d)
            hf[f"{p}mlp.up_proj.weight"] = torch.randn(dense_h, d)
            hf[f"{p}mlp.down_proj.weight"] = torch.randn(d, dense_h)
        else:
            hf[f"{p}mlp.router.gate.weight"] = torch.randn(e, d)
            for expert in range(e):
                hf[f"{p}mlp.experts.{expert}.up_proj.weight"] = torch.randn(h, d)
                hf[f"{p}mlp.experts.{expert}.gate_proj.weight"] = torch.randn(h, d)
                hf[f"{p}mlp.experts.{expert}.down_proj.weight"] = torch.randn(d, h)
            hf[f"{p}mlp.shared_expert.up_proj.weight"] = torch.randn(hs, d)
            hf[f"{p}mlp.shared_expert.gate_proj.weight"] = torch.randn(hs, d)
            hf[f"{p}mlp.shared_expert.down_proj.weight"] = torch.randn(d, hs)
    return hf


def test_olmo3moe_hf_conversion_roundtrips():
    config = _fake_config()
    hf = _synthetic_hf_state(config)

    olmo = convert_olmo3moe_state_from_hf(config, hf)
    hf_roundtrip = convert_olmo3moe_state_to_hf(config, olmo)

    assert set(hf_roundtrip.keys()) == set(hf.keys())
    for key, tensor in hf.items():
        assert torch.equal(hf_roundtrip[key], tensor), f"roundtrip mismatch for '{key}'"


def test_olmo3moe_full_width_shared_dense_conversion_roundtrips():
    config = _fake_config()
    config.dense_layers_use_shared_expert = True
    config.latent_moe_dim = None
    config.dense_mlp_intermediate_size = 7
    hf = _synthetic_hf_state(config)

    olmo = convert_olmo3moe_state_from_hf(config, hf)
    assert "blocks.0.shared_experts.w_up_gate" in olmo
    assert "blocks.0.feed_forward.w1.weight" not in olmo

    hf_roundtrip = convert_olmo3moe_state_to_hf(config, olmo)
    assert set(hf_roundtrip) == set(hf)
    for key, tensor in hf.items():
        assert torch.equal(hf_roundtrip[key], tensor), f"roundtrip mismatch for '{key}'"


def _small_kda_latent_config():
    from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig

    return Olmo3MoeConfig(
        vocab_size=32,
        hidden_size=32,
        attention_hidden_size=32,
        head_dim=8,
        dense_mlp_intermediate_size=24,
        moe_intermediate_size=12,
        shared_expert_intermediate_size=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        use_head_qk_norm=True,
        use_rope=False,
        attention_gate_type="elementwise",
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        linear_allow_neg_eigval=True,
        latent_moe_dim=16,
        latent_moe_bias=False,
        layer_types=["linear_attention", "full_attention"],
        dense_layers_indices=[0],
        embed_norm=True,
        use_peri_ln=True,
    )


@requires_gpu
@requires_fla
@requires_triton
def test_olmo3moe_kda_latent_conversion_roundtrips_exactly():
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeForCausalLM

    config = _small_kda_latent_config()
    model = Olmo3MoeForCausalLM(config)
    hf = {key: value.detach().clone() for key, value in model.state_dict().items()}

    olmo = convert_olmo3moe_state_from_hf(config, hf)
    hf_roundtrip = convert_olmo3moe_state_to_hf(config, olmo)

    assert set(hf_roundtrip) == set(hf)
    for key, tensor in hf.items():
        assert torch.equal(hf_roundtrip[key], tensor), f"roundtrip mismatch for '{key}'"
    model.load_state_dict(hf_roundtrip, strict=True)


def test_olmo3moe_conversion_rejects_unexpected_source_key():
    config = _fake_config()
    hf = _synthetic_hf_state(config)
    hf["silently.ignored.weight"] = torch.randn(1)
    with pytest.raises(KeyError, match="Unexpected HF keys"):
        convert_olmo3moe_state_from_hf(config, hf)


def test_legacy_latent_moe_dimension_is_normalized():
    config = {
        "block": {
            "latent_moe": {
                "routed_expert_dim": 320,
                "bias": False,
            }
        }
    }
    _normalize_legacy_latent_moe_config(config)
    assert config["block"]["latent_moe"] == {"latent_dim": 320, "bias": False}


def _reassemble_fused_experts(fused):
    """Expand stacked per-layer expert tensors back into per-expert HF names."""
    per_expert = {}
    for name, tensor in fused.items():
        if name.endswith("mlp.experts.gate_up_proj.weight"):
            prefix = name[: -len("experts.gate_up_proj.weight")]
            hidden = tensor.shape[1] // 2
            for e in range(tensor.shape[0]):
                per_expert[f"{prefix}experts.{e}.gate_proj.weight"] = tensor[e, :hidden]
                per_expert[f"{prefix}experts.{e}.up_proj.weight"] = tensor[e, hidden:]
        elif name.endswith("mlp.experts.down_proj.weight"):
            prefix = name[: -len("experts.down_proj.weight")]
            for e in range(tensor.shape[0]):
                per_expert[f"{prefix}experts.{e}.down_proj.weight"] = tensor[e]
        else:
            per_expert[name] = tensor
    return per_expert


def test_olmo3moe_fused_expert_export_is_the_per_expert_export_stacked():
    config = _fake_config()
    hf = _synthetic_hf_state(config)
    olmo = convert_olmo3moe_state_from_hf(config, hf)

    per_expert = convert_olmo3moe_state_to_hf(config, olmo)
    fused = dict(iter_olmo3moe_state_to_hf(config, olmo, fused_experts=True))

    assert len(fused) < len(per_expert)
    n_experts = config.n_routed_experts
    moe_hidden = config.moe_intermediate_size
    routed_dim = getattr(config, "latent_moe_dim", None) or config.hidden_size
    stacked = [name for name in fused if name.endswith("mlp.experts.gate_up_proj.weight")]
    assert stacked and all(
        fused[name].shape == (n_experts, 2 * moe_hidden, routed_dim) and fused[name].is_contiguous()
        for name in stacked
    )
    down = [name for name in fused if name.endswith("mlp.experts.down_proj.weight")]
    assert down and all(
        fused[name].shape == (n_experts, routed_dim, moe_hidden) and fused[name].is_contiguous()
        for name in down
    )
    reassembled = _reassemble_fused_experts(fused)
    assert set(reassembled) == set(per_expert)
    for key, tensor in per_expert.items():
        assert torch.equal(reassembled[key], tensor), f"fused layout mismatch for '{key}'"
