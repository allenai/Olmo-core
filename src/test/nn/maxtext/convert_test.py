"""
Tests for the OLMo Core <-> MaxText weight mappings.

These run on synthetic tensors, so they need neither JAX nor a built model. ``maxtext_test.py``
checks the mappings against MaxText itself.
"""

import copy
import json
from functools import partial
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from olmo_core.nn.maxtext import (
    MaxTextDecoderBlock,
    MaxTextModelConfig,
    convert_state_from_maxtext,
    convert_state_to_maxtext,
    get_maxtext_config,
)
from olmo_core.nn.maxtext.checkpoint import scan_params, unscan_params
from olmo_core.nn.maxtext.convert import (
    maxtext_shapes,
    normalize_olmo_core_key,
    olmo_core_shapes,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    with open(FIXTURES / name) as f:
        config = json.load(f)
    return config.get("model", config)


def shrink_hybrid(config: dict, *, n_layers: int = 16) -> dict:
    """
    The 12B hybrid MoE config with every width cut down. The sizes are pairwise distinct so a
    transpose or a swapped split can't round-trip by accident.
    """
    config = copy.deepcopy(config)
    d_model, latent, hidden, experts = 24, 12, 20, 6
    config.update(d_model=d_model, vocab_size=40, n_layers=n_layers, embed_scale=d_model**0.5)

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
        elif (i + 1) % 8 == 0:
            block = copy.deepcopy(config["block_overrides"]["7"])
        else:
            continue
        fix_block(block)
        overrides[str(i)] = block
    config["block_overrides"] = overrides
    return config


def hybrid_config(n_layers: int = 16) -> MaxTextModelConfig:
    return get_maxtext_config(
        shrink_hybrid(load_fixture("hybrid_moe_12b_model_config.json"), n_layers=n_layers)
    )


def dense_config() -> MaxTextModelConfig:
    return get_maxtext_config(load_fixture("dense_small_config.json"))


CONFIGS: Dict[str, Callable[[], MaxTextModelConfig]] = {
    "dense": dense_config,
    "hybrid_moe": hybrid_config,
}


def distinct(shapes, seed: int = 0):
    """Tensors whose elements are distinct across the whole state, so any mix-up is visible."""
    total = sum(int(np.prod(s)) for s in shapes.values())
    values = torch.from_numpy(np.random.default_rng(seed).permutation(total).astype(np.float32))
    out, offset = {}, 0
    for k, s in shapes.items():
        n = int(np.prod(s))
        out[k] = values[offset : offset + n].reshape(s)
        offset += n
    return out


def random_state(c, seed=3):
    """A float64 state for checking the mappings' semantics."""
    gen = torch.Generator().manual_seed(seed)
    return {
        k: torch.randn(s, generator=gen, dtype=torch.float64)
        for k, s in olmo_core_shapes(c).items()
    }


def assert_same_values(a: dict, b: dict):
    """The two states hold exactly the same multiset of values (nothing dropped or duplicated)."""
    flat_a = torch.sort(torch.cat([v.reshape(-1) for v in a.values()])).values
    flat_b = torch.sort(torch.cat([v.reshape(-1) for v in b.values()])).values
    torch.testing.assert_close(flat_a, flat_b, rtol=0, atol=0)


def assert_states_equal(a: dict, b: dict):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_hybrid_12b_config():
    c = get_maxtext_config(load_fixture("hybrid_moe_12b_model_config.json"))
    assert c.decoder_block == MaxTextDecoderBlock.olmoe3
    assert c.n_layers == 16
    assert [i for i, layer in enumerate(c.layers) if layer.sequence_mixer == "attention"] == [7, 15]
    assert [i for i, layer in enumerate(c.layers) if not layer.moe] == [0]
    assert c.kda is not None and c.moe is not None
    assert (c.kda.n_heads, c.kda.head_dim, c.kda.head_v_dim) == (8, 128, 256)
    assert (c.attention.n_heads, c.attention.n_kv_heads, c.attention.head_dim) == (8, 4, 128)
    assert (c.moe.num_experts, c.moe.hidden_size, c.moe.latent_dim) == (512, 1024, 512)
    assert c.scan_layout.cycle == 8
    assert sum(int(np.prod(s)) for s in olmo_core_shapes(c).values()) == 12_496_341_632
    assert sum(int(np.prod(s)) for s in maxtext_shapes(c).values()) == 12_496_341_632


def test_hybrid_12b_maxtext_tree_shapes():
    # A few paths copied from ``jax.eval_shape`` of the MaxText model.
    c = get_maxtext_config(load_fixture("hybrid_moe_12b_model_config.json"))
    shapes = maxtext_shapes(c)
    assert shapes["decoder/layers_0/shared_ffn/wi_0/kernel"] == (1024, 8192)
    assert shapes["decoder/layers_1/moe_block/wi_0"] == (512, 512, 1024)
    assert shapes["decoder/layers_1/moe_block/wo"] == (512, 1024, 512)
    assert shapes["decoder/layers_7/mixer/attention/query/kernel"] == (1024, 8, 256)
    assert shapes["decoder/layers_8/mixer/v_conv"] == (2048, 4)

    meta = {k: torch.empty(s, device="meta") for k, s in shapes.items()}
    scanned = {k: tuple(v.shape) for k, v in scan_params(meta, c.scan_layout, c.n_layers).items()}
    assert scanned["decoder/layers_0/layer_0/shared_ffn/wi_0/kernel"] == (1024, 8192)
    assert scanned["decoder/scanned_blocks/layer_0/moe_block/wi_0"] == (512, 1, 512, 1024)
    assert scanned["decoder/scanned_blocks/layer_7/mixer/attention/query_norm/scale"] == (8, 1, 128)
    assert scanned["decoder/scanned_blocks/layer_3/mixer/A_log"] == (8, 1)
    assert len(scanned) == 444


def test_dense_config():
    c = dense_config()
    assert c.decoder_block == MaxTextDecoderBlock.olmo3
    assert c.sliding_window == 8
    assert c.scan_layout.cycle == 4
    assert "sliding_window_size=8" in c.overrides()


@pytest.mark.parametrize(
    "make_config",
    [
        pytest.param(dense_config, id="dense"),
        *(pytest.param(partial(hybrid_config, n), id=f"hybrid_moe-{n}") for n in (8, 16, 32)),
    ],
)
def test_olmo_core_round_trip(make_config):
    c = make_config()
    state = distinct(olmo_core_shapes(c))
    maxtext = convert_state_to_maxtext(c, state)
    assert_same_values(state, maxtext)
    scanned = scan_params(maxtext, c.scan_layout, c.n_layers)
    assert_same_values(state, scanned)
    assert_states_equal(convert_state_from_maxtext(c, unscan_params(scanned, c.scan_layout)), state)


@pytest.mark.parametrize("name", CONFIGS)
def test_maxtext_round_trip(name):
    c = CONFIGS[name]()
    params = distinct(maxtext_shapes(c), seed=1)
    assert_states_equal(convert_state_to_maxtext(c, convert_state_from_maxtext(c, params)), params)


def test_hybrid_scan_layout():
    c = hybrid_config()
    params = distinct(maxtext_shapes(c), seed=2)
    scanned = scan_params(params, c.scan_layout, c.n_layers)
    # Layer 9 is slot 1 of the second cycle, i.e. index 0 along the scan axis.
    assert torch.equal(
        scanned["decoder/scanned_blocks/layer_1/moe_block/wi_0"][:, 0],
        params["decoder/layers_9/moe_block/wi_0"],
    )
    assert torch.equal(
        scanned["decoder/layers_0/layer_7/mixer/attention/ssmax_scale"],
        params["decoder/layers_7/mixer/attention/ssmax_scale"],
    )
    assert_states_equal(unscan_params(scanned, c.scan_layout), params)


def test_dense_scan_layout():
    c = dense_config()
    params = distinct(maxtext_shapes(c), seed=2)
    scanned = scan_params(params, c.scan_layout, c.n_layers)
    # No unrolled first cycle: layer 6 is slot 2 of the second cycle.
    assert torch.equal(
        scanned["decoder/layers/layers_2/mlp/wi_0/kernel"][:, 1],
        params["decoder/layers_6/mlp/wi_0/kernel"],
    )
    assert not any(k.startswith("decoder/layers_") for k in scanned)
    assert_states_equal(unscan_params(scanned, c.scan_layout), params)


# The tests below check each re-layout against the computation it has to preserve, written the
# way each framework does it.

silu = F.silu


def randn(*shape, seed):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def test_shared_expert_semantics():
    c = hybrid_config()
    state = random_state(c)
    mt = convert_state_to_maxtext(c, state)
    x = randn(5, c.d_model, seed=4)
    for i in (0, 3):
        pre = f"decoder/layers_{i}"
        # olmo_core.nn.moe.v2.shared_experts.SharedExperts.forward with one expert.
        w_up_gate = state[f"blocks.{i}.shared_experts.w_up_gate"]
        F = w_up_gate.shape[1] // 2
        up_gate = (x @ w_up_gate).reshape(5, 1, 2, F)
        ref = (up_gate[:, 0, 0] * silu(up_gate[:, 0, 1])) @ state[
            f"blocks.{i}.shared_experts.w_down"
        ][0]
        # MaxText MlpBlock with mlp_activations ["silu", "linear"].
        h = silu(x @ mt[f"{pre}/shared_ffn/wi_0/kernel"]) * (
            x @ mt[f"{pre}/shared_ffn/wi_1/kernel"]
        )
        torch.testing.assert_close(h @ mt[f"{pre}/shared_ffn/wo/kernel"], ref)


def test_routed_expert_and_router_semantics():
    c = hybrid_config()
    assert c.moe is not None
    state = random_state(c)
    mt = convert_state_to_maxtext(c, state)
    pre, b = "decoder/layers_2", "blocks.2"
    x = randn(5, c.d_model, seed=5)

    # Router: F.linear(x, weight.view(E, D)).
    ref_logits = x @ state[f"{b}.routed_experts_router.weight"].reshape(c.moe.num_experts, -1).T
    torch.testing.assert_close(x @ mt[f"{pre}/moe_block/gate/kernel"], ref_logits)

    latent = x @ state[f"{b}.latent_down_proj.weight"].T
    torch.testing.assert_close(latent, x @ mt[f"{pre}/latent_down/kernel"])
    for e in range(c.moe.num_experts):
        # RoutedExperts: gmm(x, w_up_gate, trans_b=True), chunk(2) -> up, gate, then gmm(h, w_down).
        up, gate = (latent @ state[f"{b}.routed_experts.w_up_gate"][e].T).chunk(2, dim=-1)
        ref = (up * silu(gate)) @ state[f"{b}.routed_experts.w_down"][e]
        # MaxText RoutedMoE: silu(x @ wi_0[e]) * (x @ wi_1[e]) @ wo[e].
        h = silu(latent @ mt[f"{pre}/moe_block/wi_0"][e]) * (
            latent @ mt[f"{pre}/moe_block/wi_1"][e]
        )
        torch.testing.assert_close(h @ mt[f"{pre}/moe_block/wo"][e], ref)
    y = randn(5, c.moe.latent_dim, seed=8)
    torch.testing.assert_close(
        y @ state[f"{b}.latent_up_proj.weight"].T, y @ mt[f"{pre}/latent_up/kernel"]
    )


def test_gated_attention_projection_semantics():
    c = hybrid_config()
    state = random_state(c)
    mt = convert_state_to_maxtext(c, state)
    pre, a = "decoder/layers_7/mixer/attention", "blocks.7.attention"
    H, KV, hd = c.attention.n_heads, c.attention.n_kv_heads, c.attention.head_dim
    x = randn(5, c.d_model, seed=6)

    # OLMo Core: q = w_q(x).view(..., H, hd); the elementwise gate w_g(x) has the same layout.
    q_ref = (x @ state[f"{a}.w_q.weight"].T).reshape(5, H, hd)
    gate_ref = (x @ state[f"{a}.w_g.weight"].T).reshape(5, H, hd)
    # MaxText: one [D, H, 2 * hd] projection, split into query and gate along the last axis.
    qg = torch.einsum("sd,dhk->shk", x, mt[f"{pre}/query/kernel"])
    torch.testing.assert_close(qg[..., :hd], q_ref)
    torch.testing.assert_close(qg[..., hd:], gate_ref)

    k_ref = (x @ state[f"{a}.w_k.weight"].T).reshape(5, KV, hd)
    torch.testing.assert_close(torch.einsum("sd,dhk->shk", x, mt[f"{pre}/key/kernel"]), k_ref)

    o = randn(5, H, hd, seed=7)
    torch.testing.assert_close(
        o.reshape(5, -1) @ mt[f"{pre}/out/kernel"], o.reshape(5, -1) @ state[f"{a}.w_out.weight"].T
    )


def test_attention_and_feed_forward_semantics():
    c = dense_config()
    state = random_state(c)
    mt = convert_state_to_maxtext(c, state)
    pre, b = "decoder/layers_1", "blocks.1"
    H, KV, hd = c.attention.n_heads, c.attention.n_kv_heads, c.attention.head_dim
    x = randn(5, c.d_model, seed=6)

    for proj, heads, name in (("w_q", H, "query"), ("w_k", KV, "key"), ("w_v", KV, "value")):
        ref = (x @ state[f"{b}.attention.{proj}.weight"].T).reshape(5, heads, hd)
        torch.testing.assert_close(
            torch.einsum("sd,dhk->shk", x, mt[f"{pre}/attention/{name}/kernel"]), ref
        )
    # MaxText's output projection contracts over [heads, head_dim].
    o = randn(5, H, hd, seed=7)
    torch.testing.assert_close(
        torch.einsum("shk,hkd->sd", o, mt[f"{pre}/attention/out/kernel"]),
        o.reshape(5, -1) @ state[f"{b}.attention.w_out.weight"].T,
    )
    # OLMo Core: w2(silu(w1(x)) * w3(x)).
    ff = f"{b}.feed_forward"
    ref = (silu(x @ state[f"{ff}.w1.weight"].T) * (x @ state[f"{ff}.w3.weight"].T)) @ state[
        f"{ff}.w2.weight"
    ].T
    h = silu(x @ mt[f"{pre}/mlp/wi_0/kernel"]) * (x @ mt[f"{pre}/mlp/wi_1/kernel"])
    torch.testing.assert_close(h @ mt[f"{pre}/mlp/wo/kernel"], ref)


def test_normalize_olmo_core_key():
    name = "blocks.3.routed_experts.w_up_gate"
    for key in (
        name,
        f"model.{name}",
        f"module.{name}.main",
        f"model.module.{name}",
        f"model._orig_mod.{name}",
    ):
        assert normalize_olmo_core_key(key) == name
    assert normalize_olmo_core_key("blocks.0.attention.A_log.main") == "blocks.0.attention.A_log"


@pytest.mark.parametrize("name", CONFIGS)
def test_unmapped_or_missing_keys_fail(name):
    c = CONFIGS[name]()
    state = random_state(c)
    with pytest.raises(KeyError, match="extra"):
        convert_state_to_maxtext(c, {**state, "blocks.0.attention.mystery": torch.zeros(1)})
    del state["blocks.1.attention.w_k.weight"]
    with pytest.raises(KeyError, match="missing"):
        convert_state_to_maxtext(c, state)


def test_wrong_shape_fails():
    c = hybrid_config()
    state = random_state(c)
    state["blocks.1.routed_experts.w_down"] = state["blocks.1.routed_experts.w_down"].transpose(
        1, 2
    )
    with pytest.raises(ValueError, match="w_down"):
        convert_state_to_maxtext(c, state)


def test_hybrid_layout_maxtext_cannot_build_fails():
    config = shrink_hybrid(load_fixture("hybrid_moe_12b_model_config.json"))
    # Full attention on layers 6 and 15: not every cycle ends in full attention.
    config["block_overrides"]["6"] = config["block_overrides"].pop("7")
    with pytest.raises(ValueError, match="cycle"):
        get_maxtext_config(config)


def test_dense_sliding_window_pattern_maxtext_cannot_build_fails():
    config = load_fixture("dense_small_config.json")
    config["block"]["sequence_mixer"]["sliding_window"]["pattern"] = [8, -1]
    with pytest.raises(ValueError, match="sliding-window pattern"):
        get_maxtext_config(config)


def test_unsupported_model_fails():
    config = load_fixture("dense_small_config.json")
    config["block"]["name"] = "default"
    with pytest.raises(NotImplementedError, match="no MaxText decoder"):
        get_maxtext_config(config)


# The settings below change the model's forward pass without changing any parameter shape, so a
# weight round-trip can't catch them. They have to be rejected (or translated) at config time.

FIXTURE_FILES = {
    "dense": "dense_small_config.json",
    "hybrid_moe": "hybrid_moe_small_config.json",
}


def moe_blocks(config: dict) -> list:
    """Every block config in ``config`` that has routed experts."""
    blocks = [config["block"], *(config.get("block_overrides") or {}).values()]
    return [b for b in blocks if b.get("routed_experts") is not None]


def test_dense_no_global_rope_fails():
    # MaxText's olmo3 decoder applies RoPE on every layer.
    config = load_fixture("dense_small_config.json")
    config["block"]["sequence_mixer"]["rope"]["no_global_rope"] = True
    with pytest.raises(ValueError, match="no_global_rope"):
        get_maxtext_config(config)


@pytest.mark.parametrize(
    "setting, value",
    [
        ("restore_weight_scale", False),
        ("normalize_expert_weights", None),
        ("normalize_expert_weights", 2.0),
        ("gating_function", "sigmoid"),
        ("expert_weight_scale", 0.5),
        ("original_top_k", 4),
        ("bias", True),
        ("bias_gamma", 1e-4),
        ("score_correction_bias", True),
        ("uniform_expert_assignment", True),
        ("random_expert_assignment", True),
        ("n_group", 2),
    ],
)
def test_hybrid_router_maxtext_cannot_build_fails(setting, value):
    # MaxText's olmoe3 router has exactly one weighting: top_k * softmax(top-k logits), with no
    # biases and no grouping.
    config = load_fixture("hybrid_moe_small_config.json")
    for block in moe_blocks(config):
        block["routed_experts_router"][setting] = value
    with pytest.raises(ValueError, match="router"):
        get_maxtext_config(config)


def test_hybrid_topk_softmax_router_is_equivalent():
    # softmax over the top-k logits is the L1-normalized gather of the full softmax, so this
    # router computes the same weights as the fixture's.
    config = load_fixture("hybrid_moe_small_config.json")
    for block in moe_blocks(config):
        block["routed_experts_router"].update(
            gating_function="topk_softmax", normalize_expert_weights=None
        )
    assert get_maxtext_config(config) == get_maxtext_config(
        load_fixture("hybrid_moe_small_config.json")
    )


@pytest.mark.parametrize("embed_scale", [None, 8.0])
def test_hybrid_embed_scale_fails(embed_scale):
    # MaxText scales by sqrt(d_model) before the embedding norm; the norm's epsilon keeps a
    # different scale from cancelling out.
    config = load_fixture("hybrid_moe_small_config.json")
    config["embed_scale"] = embed_scale
    with pytest.raises(ValueError, match="embed_scale"):
        get_maxtext_config(config)


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_lm_head_norm_epsilon_fails(name):
    config = load_fixture(FIXTURE_FILES[name])
    config["lm_head"]["layer_norm"]["eps"] = 1e-5
    with pytest.raises(ValueError, match="epsilon"):
        get_maxtext_config(config)


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_missing_lm_head_norm_fails(name):
    config = load_fixture(FIXTURE_FILES[name])
    config["lm_head"]["layer_norm"] = None
    with pytest.raises(ValueError, match="LM head"):
        get_maxtext_config(config)


def attention_blocks(config: dict) -> list:
    """Every block config in ``config`` whose sequence mixer is full attention."""
    blocks = [config["block"], *(config.get("block_overrides") or {}).values()]
    return [b for b in blocks if b["sequence_mixer"].get("type", "attention") == "attention"]


@pytest.mark.parametrize("name", FIXTURE_FILES)
@pytest.mark.parametrize("module", ["sequence_mixer", "feed_forward", "lm_head"])
def test_omitted_bias_defaults_to_true_fails(name, module):
    # ``as_config_dict()`` drops None fields, and the modules then build with bias=True.
    config = load_fixture(FIXTURE_FILES[name])
    if module == "lm_head":
        targets = [config["lm_head"]]
    elif module == "feed_forward":
        if name != "dense":
            pytest.skip("no dense feed-forward")
        targets = [config["block"]["feed_forward"]]
    else:
        targets = [b["sequence_mixer"] for b in attention_blocks(config)]
    for target in targets:
        del target["bias"]
    with pytest.raises(ValueError, match="bias"):
        get_maxtext_config(config)


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_clip_qkv_fails(name):
    config = load_fixture(FIXTURE_FILES[name])
    for block in attention_blocks(config):
        block["sequence_mixer"]["clip_qkv"] = 8.0
    with pytest.raises(ValueError, match="clip_qkv"):
        get_maxtext_config(config)


@pytest.mark.parametrize("setting", ["attention_residual_alpha", "feed_forward_residual_alpha"])
def test_dense_residual_alpha_fails(setting):
    # The overrides carry no residual scale, so a non-unit alpha would convert into a different
    # model.
    config = load_fixture("dense_small_config.json")
    config["block"][setting] = 0.5
    with pytest.raises(ValueError, match="residual"):
        get_maxtext_config(config)


@pytest.mark.parametrize("name", FIXTURE_FILES)
@pytest.mark.parametrize("where", ["block.dropout", "block.masked_dropout", "attention.dropout"])
def test_dropout_fails(name, where):
    # The overrides carry no dropout rate, so a model trained with dropout would continue
    # training without it.
    config = load_fixture(FIXTURE_FILES[name])
    blocks = [config["block"], *(config.get("block_overrides") or {}).values()]
    for block in blocks:
        if where == "attention.dropout":
            if block["sequence_mixer"].get("type", "attention") == "attention":
                block["sequence_mixer"]["dropout"] = 0.1
        else:
            block[where.split(".")[1]] = 0.1
    with pytest.raises(ValueError, match="dropout"):
        get_maxtext_config(config)


def test_dense_zero_embed_scale_fails():
    config = load_fixture("dense_small_config.json")
    config["embed_scale"] = 0.0
    with pytest.raises(ValueError, match="embedding"):
        get_maxtext_config(config)


def test_hybrid_qk_norm_epsilon_fails():
    config = load_fixture("hybrid_moe_small_config.json")
    for block in attention_blocks(config):
        block["sequence_mixer"]["qk_norm"]["eps"] = 1e-5
    with pytest.raises(ValueError, match="epsilon"):
        get_maxtext_config(config)


def test_hybrid_missing_qk_norm_fails():
    config = load_fixture("hybrid_moe_small_config.json")
    for block in attention_blocks(config):
        block["sequence_mixer"]["qk_norm"] = None
    with pytest.raises(ValueError, match="QK norm"):
        get_maxtext_config(config)


def test_hybrid_kda_norm_epsilon_fails():
    # MaxText's olmoe3 decoder hardcodes the KDA output norm's epsilon.
    config = load_fixture("hybrid_moe_small_config.json")
    for block in [config["block"], *config["block_overrides"].values()]:
        if block["sequence_mixer"]["type"] == "kimi_delta_attention":
            block["sequence_mixer"]["norm_eps"] = 1e-6
    with pytest.raises(ValueError, match="norm_eps"):
        get_maxtext_config(config)


def test_hybrid_kda_allow_neg_eigval_is_translated():
    config = load_fixture("hybrid_moe_small_config.json")
    assert "kda_allow_neg_eigval=True" in get_maxtext_config(config).overrides()
    for block in [config["block"], *config["block_overrides"].values()]:
        if block["sequence_mixer"]["type"] == "kimi_delta_attention":
            block["sequence_mixer"]["allow_neg_eigval"] = False
    assert "kda_allow_neg_eigval=False" in get_maxtext_config(config).overrides()


def test_hybrid_emo_is_translated():
    c = get_maxtext_config(load_fixture("hybrid_moe_small_config.json"))
    assert c.moe is not None and c.moe.emo is not None
    assert (c.moe.emo.min_pool, c.moe.emo.max_pool, c.moe.emo.eval_pool) == (2, 8, 8)
    assert c.moe.emo.eos_token_id == 100257
    overrides = c.overrides()
    for key in (
        "emo_enabled=True",
        "emo_min_document_expert_pool=2",
        "emo_max_document_expert_pool=8",
        "emo_eval_document_expert_pool=8",
        # The EMO router pools experts per EOS-delimited document, so MaxText's data pipeline
        # has to split documents on the same token, and only the router may see the split.
        "olmo_segment_eos_token_id=100257",
        "olmoe3_segments_for_emo_only=True",
    ):
        assert key in overrides


def test_hybrid_emo_default_eval_pool_is_resolved():
    # ``eval_document_expert_pool`` is optional; OLMo Core evaluates with the midpoint pool.
    config = load_fixture("hybrid_moe_small_config.json")
    for block in moe_blocks(config):
        block["routed_experts_router"]["emo"].update(
            min_document_expert_pool=3, max_document_expert_pool=8
        )
        del block["routed_experts_router"]["emo"]["eval_document_expert_pool"]
    c = get_maxtext_config(config)
    assert c.moe is not None and c.moe.emo is not None
    assert c.moe.emo.eval_pool == 5
    assert "emo_eval_document_expert_pool=5" in c.overrides()


def test_hybrid_without_emo():
    config = load_fixture("hybrid_moe_small_config.json")
    for block in moe_blocks(config):
        block["routed_experts_router"]["emo"] = None
    c = get_maxtext_config(config)
    assert c.moe is not None and c.moe.emo is None
    overrides = c.overrides()
    assert "emo_enabled=False" in overrides
    assert not any(k.startswith("olmo_segment_eos_token_id") for k in overrides)


def kda_blocks(config: dict) -> list:
    """Every block config in ``config`` whose sequence mixer is KDA."""
    blocks = [config["block"], *(config.get("block_overrides") or {}).values()]
    return [b for b in blocks if b["sequence_mixer"].get("type") == "kimi_delta_attention"]


@pytest.mark.parametrize("how", ["omitted", "none", "mixed"])
def test_hybrid_kda_default_n_v_heads_is_resolved(how):
    # ``n_v_heads`` is optional and defaults to ``n_heads``; saved configs may omit it, and
    # blocks may spell the same value differently.
    config = load_fixture("hybrid_moe_small_config.json")
    expected = get_maxtext_config(config)
    for i, block in enumerate(kda_blocks(config)):
        mixer = block["sequence_mixer"]
        assert mixer["n_v_heads"] == mixer["n_heads"]
        if how == "none":
            mixer["n_v_heads"] = None
        elif how == "omitted" or i == 0:
            del mixer["n_v_heads"]
    assert get_maxtext_config(config) == expected


def norm_configs(obj):
    """Every layer norm config dict nested anywhere in ``obj``."""
    if isinstance(obj, dict):
        if str(obj.get("_CLASS_", "")).endswith("LayerNormConfig"):
            yield obj
        for value in obj.values():
            yield from norm_configs(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from norm_configs(value)


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_omitted_norm_epsilon_is_resolved(name):
    # ``eps`` is optional; RMSNorm builds with 1e-5 when it's unset.
    config = load_fixture(FIXTURE_FILES[name])
    for norm in norm_configs(config):
        del norm["eps"]
    assert get_maxtext_config(config).norm_eps == 1e-5


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_omitted_norm_bias_defaults_to_true_fails(name):
    # ``bias`` is optional too, and RMSNorm builds with a bias when it's unset.
    config = load_fixture(FIXTURE_FILES[name])
    for norm in norm_configs(config):
        del norm["bias"]
    with pytest.raises(ValueError, match="bias"):
        get_maxtext_config(config)


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_non_affine_norm_fails(name):
    config = load_fixture(FIXTURE_FILES[name])
    for norm in norm_configs(config):
        norm["elementwise_affine"] = False
    with pytest.raises(ValueError, match="affine"):
        get_maxtext_config(config)


def test_hybrid_latent_up_proj_input_norm_fails():
    config = load_fixture("hybrid_moe_small_config.json")
    for block in moe_blocks(config):
        block["latent_moe"]["up_proj_input_norm_enabled"] = True
    with pytest.raises(ValueError, match="up_proj_input_norm"):
        get_maxtext_config(config)


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_extra_vocab_fails(name):
    config = load_fixture(FIXTURE_FILES[name])
    config["n_extra_vocab"] = 2
    with pytest.raises(NotImplementedError, match="n_extra_vocab"):
        get_maxtext_config(config)
