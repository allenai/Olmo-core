"""
Tests for the OLMo Core <-> MaxText OLMoE3 weight mapping.

These run on synthetic numpy tensors, so they need neither JAX nor a built model. Numerical parity
against the real model forward passes is checked separately (see ``src/scripts/maxtext/``).
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from olmo_core.nn.maxtext.olmoe3 import (
    MAXTEXT_CYCLE,
    OLMoE3Geometry,
    maxtext_shapes,
    maxtext_to_olmo_core,
    normalize_olmo_core_key,
    olmo_core_shapes,
    olmo_core_to_maxtext,
    scan_maxtext,
    unscan_maxtext,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _config_810m() -> dict:
    with open(FIXTURES / "olmoe3_810m_model_config.json") as f:
        return json.load(f)


def _shrink(config: dict, *, n_layers: int = 16) -> dict:
    """
    The 810m config with every width cut down. The sizes are pairwise distinct so a transpose or
    a swapped split can't round-trip by accident.
    """
    config = copy.deepcopy(config)
    d_model, latent, hidden, experts = 24, 12, 20, 6
    config.update(d_model=d_model, vocab_size=40, n_layers=n_layers)

    def fix_block(block):
        mixer = block["sequence_mixer"]
        if mixer["type"] == "attention":
            mixer.update(n_heads=4, n_kv_heads=2, head_dim=8)
        else:
            mixer.update(n_heads=2, n_v_heads=2, head_dim=10, expand_v=1.5, conv_size=3)
        dense = block.get("routed_experts") is None
        block["shared_experts"].update(d_model=d_model, hidden_size=28 if dense else hidden)
        if not dense:
            block["routed_experts"].update(d_model=latent, hidden_size=hidden, num_experts=experts)
            block["routed_experts_router"].update(d_model=d_model, num_experts=experts)
            block["latent_moe"].update(latent_dim=latent)

    fix_block(config["block"])
    overrides = {}
    for i in range(n_layers):
        if i == 0:
            block = copy.deepcopy(config["block_overrides"]["0"])
        elif (i + 1) % MAXTEXT_CYCLE == 0:
            block = copy.deepcopy(config["block_overrides"]["7"])
        else:
            continue
        fix_block(block)
        overrides[str(i)] = block
    config["block_overrides"] = overrides
    return config


@pytest.fixture
def geometry() -> OLMoE3Geometry:
    return OLMoE3Geometry.from_olmo_core_config(_shrink(_config_810m()))


def _distinct(shapes, rng: np.random.Generator):
    """Tensors whose elements are distinct across the whole state, so any mix-up is visible."""
    total = sum(int(np.prod(s)) for s in shapes.values())
    values = rng.permutation(total).astype(np.float32)
    out, offset = {}, 0
    for k, s in shapes.items():
        n = int(np.prod(s))
        out[k] = values[offset : offset + n].reshape(s)
        offset += n
    return out


def _assert_same_values(a: dict, b: dict):
    """The two states hold exactly the same multiset of values (nothing dropped or duplicated)."""
    flat_a = np.sort(np.concatenate([v.reshape(-1) for v in a.values()]))
    flat_b = np.sort(np.concatenate([v.reshape(-1) for v in b.values()]))
    np.testing.assert_array_equal(flat_a, flat_b)


def test_810m_geometry():
    g = OLMoE3Geometry.from_olmo_core_config(_config_810m())
    g.validate_maxtext_layout()
    assert g.n_layers == 16
    assert [i for i, layer in enumerate(g.layers) if layer.attention] == [7, 15]
    assert [i for i, layer in enumerate(g.layers) if not layer.moe] == [0]
    assert (g.kda_heads, g.kda_head_dim, g.kda_head_v_dim) == (8, 128, 256)
    assert (g.attn_heads, g.attn_kv_heads, g.attn_head_dim) == (8, 4, 128)
    assert (g.num_experts, g.expert_hidden, g.latent_dim) == (512, 1024, 512)
    # Same total as MaxText's olmo35-tiny and OLMo Core's 810m rung.
    assert sum(int(np.prod(s)) for s in olmo_core_shapes(g).values()) == 12_496_341_632
    assert sum(int(np.prod(s)) for s in maxtext_shapes(g).values()) == 12_496_341_632


def test_810m_maxtext_tree_shapes():
    # A few paths copied from ``jax.eval_shape`` of MaxText's olmo35-tiny init.
    g = OLMoE3Geometry.from_olmo_core_config(_config_810m())
    shapes = maxtext_shapes(g)
    assert shapes["decoder/layers_0/shared_ffn/wi_0/kernel"] == (1024, 8192)
    assert shapes["decoder/layers_1/moe_block/wi_0"] == (512, 512, 1024)
    assert shapes["decoder/layers_1/moe_block/wo"] == (512, 1024, 512)
    assert shapes["decoder/layers_7/mixer/attention/query/kernel"] == (1024, 8, 256)
    assert shapes["decoder/layers_8/mixer/v_conv"] == (2048, 4)

    zeros = {k: np.broadcast_to(np.float32(0), s) for k, s in shapes.items()}
    scanned = {k: v.shape for k, v in scan_maxtext(zeros, g.n_layers).items()}
    assert scanned["decoder/layers_0/layer_0/shared_ffn/wi_0/kernel"] == (1024, 8192)
    assert scanned["decoder/scanned_blocks/layer_0/moe_block/wi_0"] == (512, 1, 512, 1024)
    assert scanned["decoder/scanned_blocks/layer_7/mixer/attention/query_norm/scale"] == (8, 1, 128)
    assert scanned["decoder/scanned_blocks/layer_3/mixer/A_log"] == (8, 1)
    assert len(scanned) == 444


@pytest.mark.parametrize("n_layers", [8, 16, 32])
def test_olmo_core_round_trip(n_layers):
    g = OLMoE3Geometry.from_olmo_core_config(_shrink(_config_810m(), n_layers=n_layers))
    state = _distinct(olmo_core_shapes(g), np.random.default_rng(0))
    maxtext = olmo_core_to_maxtext(state, g)
    _assert_same_values(state, maxtext)
    scanned = scan_maxtext(maxtext, g.n_layers)
    _assert_same_values(state, scanned)
    back = maxtext_to_olmo_core(unscan_maxtext(scanned), g)
    assert back.keys() == state.keys()
    for k in state:
        np.testing.assert_array_equal(back[k], state[k], err_msg=k)


def test_maxtext_round_trip(geometry):
    params = _distinct(maxtext_shapes(geometry), np.random.default_rng(1))
    back = olmo_core_to_maxtext(maxtext_to_olmo_core(params, geometry), geometry)
    assert back.keys() == params.keys()
    for k in params:
        np.testing.assert_array_equal(back[k], params[k], err_msg=k)


def test_scan_round_trip(geometry):
    params = _distinct(maxtext_shapes(geometry), np.random.default_rng(2))
    scanned = scan_maxtext(params, geometry.n_layers)
    # Layer 9 is slot 1 of the second cycle, i.e. index 0 along the scan axis.
    np.testing.assert_array_equal(
        scanned["decoder/scanned_blocks/layer_1/moe_block/wi_0"][:, 0],
        params["decoder/layers_9/moe_block/wi_0"],
    )
    np.testing.assert_array_equal(
        scanned["decoder/layers_0/layer_7/mixer/attention/ssmax_scale"],
        params["decoder/layers_7/mixer/attention/ssmax_scale"],
    )
    back = unscan_maxtext(scanned)
    assert back.keys() == params.keys()
    for k in params:
        np.testing.assert_array_equal(back[k], params[k], err_msg=k)


# The tests below check each re-layout against the computation it has to preserve, written the
# way each framework does it.


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _state(geometry, seed=3):
    rng = np.random.default_rng(seed)
    return {
        k: rng.standard_normal(s).astype(np.float64) for k, s in olmo_core_shapes(geometry).items()
    }


def test_shared_expert_semantics(geometry):
    state = _state(geometry)
    mt = olmo_core_to_maxtext(state, geometry)
    x = np.random.default_rng(4).standard_normal((5, geometry.d_model))
    for i, pre in ((0, "decoder/layers_0"), (3, "decoder/layers_3")):
        # olmo_core.nn.moe.v2.shared_experts.SharedExperts.forward with E=1.
        w_up_gate = state[f"blocks.{i}.shared_experts.w_up_gate"]
        F = w_up_gate.shape[1] // 2
        up_gate = (x @ w_up_gate).reshape(5, 1, 2, F)
        ref = (up_gate[:, 0, 0] * _silu(up_gate[:, 0, 1])) @ state[
            f"blocks.{i}.shared_experts.w_down"
        ][0]
        # MaxText MlpBlock with mlp_activations ["silu", "linear"].
        h = _silu(x @ mt[f"{pre}/shared_ffn/wi_0/kernel"]) * (
            x @ mt[f"{pre}/shared_ffn/wi_1/kernel"]
        )
        np.testing.assert_allclose(h @ mt[f"{pre}/shared_ffn/wo/kernel"], ref, rtol=1e-12)


def test_routed_expert_and_router_semantics(geometry):
    state = _state(geometry)
    mt = olmo_core_to_maxtext(state, geometry)
    g, pre, b = geometry, "decoder/layers_2", "blocks.2"
    x = np.random.default_rng(5).standard_normal((5, g.d_model))

    # Router: F.linear(x, weight.view(E, D)).
    ref_logits = x @ state[f"{b}.routed_experts_router.weight"].reshape(g.num_experts, -1).T
    np.testing.assert_allclose(x @ mt[f"{pre}/moe_block/gate/kernel"], ref_logits, rtol=1e-12)

    latent = x @ state[f"{b}.latent_down_proj.weight"].T
    np.testing.assert_allclose(latent, x @ mt[f"{pre}/latent_down/kernel"], rtol=1e-12)
    for e in range(g.num_experts):
        # RoutedExperts: gmm(x, w_up_gate, trans_b=True), chunk(2) -> up, gate, then gmm(h, w_down).
        up, gate = np.split(latent @ state[f"{b}.routed_experts.w_up_gate"][e].T, 2, axis=-1)
        ref = (up * _silu(gate)) @ state[f"{b}.routed_experts.w_down"][e]
        # MaxText RoutedMoE: silu(x @ wi_0[e]) * (x @ wi_1[e]) @ wo[e].
        h = _silu(latent @ mt[f"{pre}/moe_block/wi_0"][e]) * (
            latent @ mt[f"{pre}/moe_block/wi_1"][e]
        )
        np.testing.assert_allclose(h @ mt[f"{pre}/moe_block/wo"][e], ref, rtol=1e-12)
    y = np.random.default_rng(8).standard_normal((5, g.latent_dim))
    out = y @ state[f"{b}.latent_up_proj.weight"].T
    np.testing.assert_allclose(out, y @ mt[f"{pre}/latent_up/kernel"], rtol=1e-12)


def test_attention_projection_semantics(geometry):
    state = _state(geometry)
    mt = olmo_core_to_maxtext(state, geometry)
    g, pre, a = geometry, "decoder/layers_7/mixer/attention", "blocks.7.attention"
    H, KV, hd = g.attn_heads, g.attn_kv_heads, g.attn_head_dim
    x = np.random.default_rng(6).standard_normal((5, g.d_model))

    # OLMo Core: q = w_q(x).view(..., H, hd); the elementwise gate w_g(x) has the same layout.
    q_ref = (x @ state[f"{a}.w_q.weight"].T).reshape(5, H, hd)
    gate_ref = (x @ state[f"{a}.w_g.weight"].T).reshape(5, H, hd)
    # MaxText: one [D, H, 2 * hd] projection, split into query and gate along the last axis.
    qg = np.einsum("sd,dhk->shk", x, mt[f"{pre}/query/kernel"])
    np.testing.assert_allclose(qg[..., :hd], q_ref, rtol=1e-12)
    np.testing.assert_allclose(qg[..., hd:], gate_ref, rtol=1e-12)

    k_ref = (x @ state[f"{a}.w_k.weight"].T).reshape(5, KV, hd)
    np.testing.assert_allclose(np.einsum("sd,dhk->shk", x, mt[f"{pre}/key/kernel"]), k_ref)

    o = np.random.default_rng(7).standard_normal((5, H, hd))
    np.testing.assert_allclose(
        o.reshape(5, -1) @ mt[f"{pre}/out/kernel"], o.reshape(5, -1) @ state[f"{a}.w_out.weight"].T
    )


def test_normalize_olmo_core_key():
    name = "blocks.3.routed_experts.w_up_gate"
    for key in (name, f"model.{name}", f"module.{name}.main", f"model.module.{name}"):
        assert normalize_olmo_core_key(key) == name
    assert normalize_olmo_core_key("blocks.0.attention.A_log.main") == "blocks.0.attention.A_log"


def test_unmapped_or_missing_keys_fail(geometry):
    state = _state(geometry)
    with pytest.raises(KeyError, match="extra"):
        olmo_core_to_maxtext({**state, "blocks.0.attention.mystery": np.zeros(1)}, geometry)
    del state["blocks.7.attention.ssmax_scale"]
    with pytest.raises(KeyError, match="missing"):
        olmo_core_to_maxtext(state, geometry)


def test_wrong_shape_fails(geometry):
    state = _state(geometry)
    state["blocks.1.routed_experts.w_down"] = state["blocks.1.routed_experts.w_down"].transpose(
        0, 2, 1
    )
    with pytest.raises(ValueError, match="w_down"):
        olmo_core_to_maxtext(state, geometry)


def test_layout_maxtext_cannot_build_fails():
    config = _shrink(_config_810m())
    # Move full attention from layer 7 to layer 6.
    config["block_overrides"]["6"] = config["block_overrides"].pop("7")
    g = OLMoE3Geometry.from_olmo_core_config(config)
    with pytest.raises(ValueError, match="full attention"):
        g.validate_maxtext_layout()
