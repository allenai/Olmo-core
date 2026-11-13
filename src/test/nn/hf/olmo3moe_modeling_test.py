"""
Forward/parity test for the ``olmo3moe`` HF model + its conversion.

Builds a small ``Olmo3MoeForCausalLM`` in memory, runs its forward, then does a full
``HF -> Olmo-core -> HF`` conversion roundtrip and checks the reloaded model produces the same
logprobs. This exercises the ``modeling_olmo3moe`` forward and validates the conversion against the
model's real parameter set (strict ``load_state_dict``). Requires ``transformers``.
"""

import pytest
import torch

from olmo_core.testing.utils import (
    requires_fla,
    requires_gpu,
    requires_te,
    requires_triton,
)


def _has_olmo3moe() -> bool:
    try:
        from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import (  # noqa: F401
            Olmo3MoeForCausalLM,
        )

        return True
    except ImportError:
        return False


requires_olmo3moe = pytest.mark.skipif(
    not _has_olmo3moe(), reason="requires transformers (for the olmo3moe HF model)"
)


def _small_config():
    from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig

    # Layer 0 is dense, layer 1 is MoE (dense_layers_indices=[0]); head qk-norm on so the
    # q_norm/k_norm params the converter maps exist.
    return Olmo3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        dense_mlp_intermediate_size=16,
        shared_expert_intermediate_size=16,
        max_position_embeddings=32,
        use_head_qk_norm=True,
        dense_layers_indices=[0],
    )


def _small_kda_emo_config():
    from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig

    # Exercise the ladder's important architectural choices together: NoPE KDA plus gated full
    # attention, a dense first layer represented by the shared-expert module, and full-width EMo
    # routing (latent_moe_dim=None). At inference the EMo pool spans every expert, so its routing is
    # faithfully representable as ordinary global top-k routing in the HF implementation.
    return Olmo3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        attention_hidden_size=32,
        head_dim=8,
        dense_mlp_intermediate_size=24,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=24,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        use_head_qk_norm=True,
        use_rope=False,
        attention_gate_type="elementwise",
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        linear_allow_neg_eigval=True,
        latent_moe_dim=None,
        layer_types=["linear_attention", "full_attention"],
        dense_layers_indices=[0],
        dense_layers_use_shared_expert=True,
        embed_norm=True,
        use_peri_ln=True,
        emo_min_document_expert_pool=2,
        emo_max_document_expert_pool=4,
        emo_eval_document_expert_pool=4,
        emo_eos_token_id=63,
        global_load_balancing=True,
        use_cache=False,
    )


def _assert_logprobs_match_after_conversion_roundtrip(
    config, device: torch.device = torch.device("cpu")
):
    from olmo_core.nn.hf.convert import convert_state_from_hf, convert_state_to_hf
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeForCausalLM

    model = Olmo3MoeForCausalLM(config).to(device)
    model.eval()

    input_ids = torch.randint(
        0, config.vocab_size, (1, 8), generator=torch.Generator().manual_seed(0)
    ).to(device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        ref_logits = model(input_ids, use_cache=False).logits
    ref_logprobs = torch.log_softmax(ref_logits, dim=-1)

    hf_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    oc_state = convert_state_from_hf(config, hf_state, model_type="olmo3moe")
    hf_roundtrip = convert_state_to_hf(config, oc_state)

    model.load_state_dict(hf_roundtrip, strict=True)
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        rt_logits = model(input_ids, use_cache=False).logits
    rt_logprobs = torch.log_softmax(rt_logits, dim=-1)

    torch.testing.assert_close(rt_logprobs, ref_logprobs, rtol=1e-4, atol=1e-4)


@requires_olmo3moe
def test_olmo3moe_logprobs_match_after_conversion_roundtrip():
    config = _small_config()
    _assert_logprobs_match_after_conversion_roundtrip(config)


@requires_olmo3moe
@pytest.mark.parametrize("use_cache", [None, False, True])
def test_scalable_softmax_rejects_supplied_cache(use_cache):
    from transformers.cache_utils import DynamicCache

    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeForCausalLM

    config = _small_config()
    config.scalable_softmax = True
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    model = Olmo3MoeForCausalLM(config).eval()
    cache = DynamicCache(config=config)
    with pytest.raises(NotImplementedError, match="[Ss]calable.softmax.*cach"):
        model(torch.tensor([[1, 2, 3]]), past_key_values=cache, use_cache=use_cache)
    assert cache.get_seq_length() == 0


@requires_olmo3moe
def test_olmo3moe_kda_rejects_padding_mask_before_attention():
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import _validate_linear_attention_mask

    attention_mask = torch.tensor([[1, 1, 1, 0]])

    with pytest.raises(NotImplementedError, match="attention-mask support"):
        _validate_linear_attention_mask(attention_mask)


@requires_olmo3moe
@requires_gpu
@requires_fla
@requires_triton
def test_olmo3moe_kda_emo_logprobs_match_after_conversion_roundtrip():
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import (
        Olmo3MoeForCausalLM,
        Olmo3MoeKimiDeltaAttention,
    )

    config = _small_kda_emo_config()

    assert "linear_attention" in config.layer_types
    assert config.latent_moe_dim is None
    assert config.dense_layers_use_shared_expert is True
    assert config.emo_eval_document_expert_pool == config.n_routed_experts
    assert config.global_load_balancing is True

    model = Olmo3MoeForCausalLM(config)
    kda = next(
        module for module in model.modules() if isinstance(module, Olmo3MoeKimiDeltaAttention)
    )
    assert torch.isfinite(kda.A_log).all()
    assert torch.equal(kda.dt_bias, torch.zeros_like(kda.dt_bias))

    _assert_logprobs_match_after_conversion_roundtrip(config, torch.device("cuda"))


@requires_olmo3moe
def test_olmo3moe_experts_grouped_mm_matches_reference_loop():
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeExpert, Olmo3MoeExperts

    torch.manual_seed(0)
    num_experts, hidden_size, intermediate_size, n_tokens, top_k = 4, 32, 16, 7, 2
    experts = Olmo3MoeExperts(
        Olmo3MoeExpert(
            hidden_size=hidden_size, moe_intermediate_size=intermediate_size, hidden_act="silu"
        )
        for _ in range(num_experts)
    )

    hidden_states = torch.randn(n_tokens, hidden_size)
    topk_ids = torch.randint(0, num_experts, (n_tokens, top_k))
    topk_weights = torch.rand(n_tokens, top_k)

    # The dims satisfy the 16-byte alignment guard, but grouped_mm itself needs a new enough torch
    # (the package supports torch>=2.6, while grouped_mm's `offs=` API lands in 2.10). Skip rather
    # than fail where the op is unavailable; the forward falls back to `_forward_loop` there anyway.
    if not experts._can_use_grouped_mm(hidden_states):
        pytest.skip("torch grouped_mm unavailable")
    reference = experts._forward_loop(hidden_states, topk_ids, topk_weights)
    grouped = experts._forward_grouped_mm(hidden_states, topk_ids, topk_weights)
    torch.testing.assert_close(grouped, reference, rtol=1e-5, atol=1e-5)


@requires_olmo3moe
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_expert_combination_preserves_fp32_routing_weights(dtype):
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeExperts

    # Identity experts isolate the combination from GEMM rounding. Probabilities deliberately
    # fall between BF16 values, so rounding them before combination changes the result.
    experts = Olmo3MoeExperts([torch.nn.Identity(), torch.nn.Identity()])
    x = torch.tensor([[0.125, -0.5], [0.25, 0.75]], dtype=dtype)
    indices = torch.tensor([[0, 1], [1, 0]])
    weights = torch.tensor([[0.1732, 0.8262], [0.3511, 0.6472]], dtype=torch.float32)
    expected = (x.double() * weights.double().sum(-1, keepdim=True)).to(dtype)
    tolerance = 1e-7 if dtype == torch.float32 else 0
    torch.testing.assert_close(
        experts._forward_loop(x, indices, weights), expected, rtol=0, atol=tolerance
    )
    torch.testing.assert_close(
        experts._forward_compile_fallback(x, indices, weights), expected, rtol=0, atol=tolerance
    )


@requires_olmo3moe
@requires_te
def test_expert_combination_matches_core_unpermute():
    from olmo_core.nn.moe.utils import moe_permute_no_compile, moe_unpermute_no_compile
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeExperts

    experts = Olmo3MoeExperts([torch.nn.Identity(), torch.nn.Identity()])
    # TE's vectorized permutation requires a model-sized hidden dimension.
    x = torch.tensor([[0.125, -0.5], [0.25, 0.75]], dtype=torch.bfloat16, device="cuda").repeat(
        1, 64
    )
    indices = torch.tensor([[0, 1], [1, 0]], device="cuda")
    weights = torch.tensor([[0.1732, 0.8262], [0.3511, 0.6472]], device="cuda")
    permuted, row_id_map = moe_permute_no_compile(
        inp=x, routing_map=indices.int(), num_out_tokens=4, map_type="index"
    )
    combined = moe_unpermute_no_compile(
        inp=permuted,
        row_id_map=row_id_map,
        restore_shape=x.shape,
        map_type="index",
        merging_probs=weights,
    )
    torch.testing.assert_close(experts._forward_loop(x, indices, weights), combined, rtol=0, atol=0)


@requires_olmo3moe
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("gating", ["softmax", "sigmoid"])
@pytest.mark.parametrize("normalization", [None, 1.0])
def test_router_matches_core_fp32_scores(dtype, gating, normalization):
    from olmo_core.nn.moe.router import MoERouterGatingFunction
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeRouter
    from olmo_core.nn.moe.v2.router import MoERouterConfigV2

    torch.manual_seed(824)
    config = _small_config()
    config.gating_function = gating
    config.normalize_expert_weights = normalization
    hf = Olmo3MoeRouter(config).to(dtype).eval()
    core = (
        MoERouterConfigV2(
            d_model=config.hidden_size,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            gating_function=MoERouterGatingFunction(gating),
            normalize_expert_weights=normalization,
            restore_weight_scale=config.restore_weight_scale,
        )
        .build()
        .eval()
    )
    with torch.no_grad():
        core.weight.copy_(hf.gate.weight.flatten())
    x = torch.randn(1, 513, config.hidden_size).to(dtype)
    expected_weights, expected_indices, _, _ = core(x, False)
    for autocast in (False, True):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            weights, indices = hf(x)
        assert weights.dtype == torch.float32
        torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
        torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)


@requires_olmo3moe
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_scalable_softmax_padding_matches_individual_sequences(side, backend):
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeForCausalLM

    torch.manual_seed(831)
    config = _small_config()
    config.scalable_softmax = True
    config.use_cache = False
    config._attn_implementation = backend
    model = Olmo3MoeForCausalLM(config).eval()
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.ssmax_scale.fill_(1.3)
    sequences = [torch.tensor([3, 4, 5]), torch.tensor([6, 7, 8, 9, 10])]
    ids = torch.zeros(2, 5, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, sequence in enumerate(sequences):
        start = 5 - len(sequence) if side == "left" else 0
        ids[row, start : start + len(sequence)] = sequence
        mask[row, start : start + len(sequence)] = 1
    with torch.no_grad():
        batched = model(ids, attention_mask=mask).logits
        for row, sequence in enumerate(sequences):
            expected = model(sequence[None]).logits[0]
            torch.testing.assert_close(
                batched[row, mask[row].bool()], expected, atol=1e-6, rtol=1e-5
            )


@requires_olmo3moe
@pytest.mark.parametrize("as_mapping", [False, True])
def test_scalable_softmax_prepared_masks_require_positions(as_mapping):
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeForCausalLM

    config = _small_config()
    config.scalable_softmax = True
    config.use_cache = False
    config._attn_implementation = "eager"
    model = Olmo3MoeForCausalLM(config).eval()
    ids = torch.tensor([[3, 4, 5]])
    mask = torch.full((1, 1, 3, 3), float("-inf")).triu(1)
    if as_mapping:
        mask = {layer_type: mask for layer_type in config.layer_types}
    with pytest.raises(ValueError, match="explicit position_ids"):
        model(ids, attention_mask=mask)
    with torch.no_grad():
        actual = model(ids, attention_mask=mask, position_ids=torch.arange(3)[None]).logits
        expected = model(ids).logits
    torch.testing.assert_close(actual, expected)
