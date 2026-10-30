"""Export actual core-built hybrid models, not states synthesized by the HF implementation."""

from copy import deepcopy

import pytest
import torch

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    AttentionConfig,
    GateConfig,
    GateGranularity,
    KimiDeltaAttentionConfig,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.feed_forward import ActivationFunction, FeedForwardConfig
from olmo_core.nn.hf.checkpoint import save_hf_model
from olmo_core.nn.hf.config import get_hf_config
from olmo_core.nn.hf.convert import convert_state_from_hf, convert_state_to_hf
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe.emo import EmoRouterConfig
from olmo_core.nn.moe.moe import LatentMoEConfig
from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeForCausalLM
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.rope import PIRoPEScalingConfig, RoPEConfig
from olmo_core.nn.transformer import (
    OLMoDDPModelConfig,
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerType,
)
from olmo_core.nn.transformer.block import ReorderedNormTransformerBlock
from olmo_core.testing.utils import has_fla, requires_gpu

requires_fla = pytest.mark.skipif(not has_fla, reason="Requires flash-linear-attention")


def build_hybrid(
    *,
    latent=True,
    emo=False,
    per_head=True,
    scalable=True,
    window=None,
    device="cpu",
    kda_norm_eps=1e-5,
    latent_norm=None,
    shared_override=None,
):
    """Small generic fixture with a dense KDA layer and both sparse attention types."""
    width, experts = 128, 4
    norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False)
    kda = KimiDeltaAttentionConfig(
        n_heads=2,
        n_v_heads=2,
        head_dim=64,
        expand_v=2.0,
        allow_neg_eigval=True,
        norm_eps=kda_norm_eps,
        use_experimental_kernels=False,
    )
    attention = AttentionConfig(
        n_heads=2,
        n_kv_heads=1,
        head_dim=64,
        bias=False,
        rope=None,
        qk_norm=deepcopy(norm),
        use_head_qk_norm=True,
        qk_norm_per_head_gains=per_head,
        scalable_softmax=scalable,
        backend=AttentionBackendName.torch,
        gate=GateConfig(granularity=GateGranularity.elementwise),
        sliding_window=SlidingWindowAttentionConfig(
            pattern=[window],
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=False,
        )
        if window is not None
        else None,
    )
    shared = SharedExpertsConfig(
        d_model=width, hidden_size=128, num_experts=1, bias=False, dtype=DType.float32
    )
    sparse = OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        sequence_mixer=kda,
        layer_norm=norm,
        shared_experts=shared,
        routed_experts=RoutedExpertsConfig(
            d_model=64 if latent else width,
            hidden_size=128,
            num_experts=experts,
            bias=False,
            dtype=DType.float32,
        ),
        routed_experts_router=MoERouterConfigV2(
            d_model=width,
            num_experts=experts,
            top_k=2,
            emo=EmoRouterConfig(
                eos_token_id=63,
                min_document_expert_pool=2,
                max_document_expert_pool=experts,
                eval_document_expert_pool=experts,
            )
            if emo
            else None,
        ),
        latent_moe=LatentMoEConfig(
            latent_dim=64,
            up_proj_input_norm=latent_norm,
            up_proj_input_norm_enabled=latent_norm is not None,
        )
        if latent
        else None,
        use_peri_norm=True,
        use_pre_norm=False,
    )
    dense = deepcopy(sparse)
    dense.routed_experts = dense.routed_experts_router = dense.latent_moe = None
    full = deepcopy(sparse)
    full.sequence_mixer = attention
    overrides: dict[int, TransformerBlockConfig] = {0: dense, 1: deepcopy(sparse), 2: full}
    if shared_override is not None:
        layer, count = shared_override
        block = overrides[layer]
        assert isinstance(block, OLMoDDPTransformerBlockConfig) and block.shared_experts is not None
        block.shared_experts.num_experts = count
        block.shared_experts_router = MoERouterConfigV2(
            d_model=width, num_experts=count, top_k=count
        )
    model = OLMoDDPModelConfig(
        name=TransformerType.moe_fused_v2,
        d_model=width,
        n_layers=3,
        vocab_size=64,
        block=sparse,
        block_overrides=overrides,
        lm_head=LMHeadConfig(layer_norm=deepcopy(norm), bias=False),
        embedding_norm=deepcopy(norm),
        embed_scale=width**0.5,
        init_seed=42,
    ).build(init_device=device)
    model.init_weights(max_seq_len=32)
    # Non-uniform gains catch head/sequence broadcasting errors hidden by all-ones initialization.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".q_norm.weight" in name or ".k_norm.weight" in name or name.endswith("ssmax_scale"):
                parameter.copy_(
                    torch.linspace(0.5, 1.5, parameter.numel()).reshape(parameter.shape)
                )
    return model.eval()


def build_attention_only(*, scalable=False, emo=False, gate=None):
    """Small core-built MoE exercising the non-KDA HF exporter on CPU."""
    norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False)
    model = OLMoDDPModelConfig(
        name=TransformerType.moe_fused_v2,
        d_model=32,
        n_layers=2,
        vocab_size=64,
        block=OLMoDDPTransformerBlockConfig(
            name=TransformerBlockType.moe_fused_v2,
            sequence_mixer=AttentionConfig(
                n_heads=4,
                n_kv_heads=2,
                head_dim=8,
                bias=False,
                rope=RoPEConfig(),
                qk_norm=deepcopy(norm),
                use_head_qk_norm=True,
                scalable_softmax=scalable,
                gate=gate,
                backend=AttentionBackendName.torch,
            ),
            layer_norm=norm,
            routed_experts=RoutedExpertsConfig(
                d_model=32, hidden_size=32, num_experts=4, bias=False, dtype=DType.float32
            ),
            routed_experts_router=MoERouterConfigV2(
                d_model=32,
                num_experts=4,
                top_k=2,
                emo=EmoRouterConfig(
                    eos_token_id=63,
                    min_document_expert_pool=2,
                    max_document_expert_pool=4,
                    eval_document_expert_pool=4,
                )
                if emo
                else None,
            ),
        ),
        lm_head=LMHeadConfig(layer_norm=deepcopy(norm), bias=False),
        init_seed=42,
    ).build(init_device="cpu")
    model.init_weights(max_seq_len=8)
    return model.eval()


@pytest.mark.parametrize("scalable", [False, True])
@pytest.mark.parametrize("emo", [False, True])
def test_attention_only_export_generation_cache_policy(tmp_path, scalable, emo):
    model = build_attention_only(scalable=scalable, emo=emo)
    config = get_hf_config(model)
    assert config.use_cache is (not scalable)
    save_hf_model(tmp_path / "hf", model.state_dict(), model)
    hf = Olmo3MoeForCausalLM.from_pretrained(tmp_path / "hf").eval()
    assert hf.generation_config.use_cache is (not scalable)
    tokens = torch.tensor([[1, 2, 3]])
    generation = dict(max_new_tokens=3, do_sample=False)
    expected = hf.generate(tokens, use_cache=False, **generation)
    actual = hf.generate(tokens, **generation)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (1, 6)
    if scalable:
        with pytest.raises(NotImplementedError, match="[Ss]calable.softmax.*cach"):
            hf.generate(tokens, use_cache=True, **generation)


@pytest.mark.parametrize("layer", ["0", "1"])
@pytest.mark.parametrize("eval_pool", [None, 2])
def test_attention_only_export_rejects_restricted_emo_pool_in_every_layer(layer, eval_pool):
    model = build_attention_only(emo=True)
    emo = model.blocks[layer].routed_experts_router.emo
    emo.max_document_expert_pool = 2
    emo.eval_document_expert_pool = eval_pool
    with pytest.raises(NotImplementedError, match="eval_document_expert_pool"):
        get_hf_config(model)


@pytest.mark.parametrize(
    "builder,layer",
    [
        (build_attention_only, "0"),
        (build_attention_only, "1"),
        pytest.param(build_hybrid, "2", marks=requires_fla),
    ],
)
@pytest.mark.parametrize("norm_name", ["q_norm", "k_norm"])
@pytest.mark.parametrize(
    "norm_type,full_precision",
    [
        (LayerNormType.default, True),
        (LayerNormType.qwen_rms, True),
        (LayerNormType.nemotron_rms, True),
        (LayerNormType.rms, False),
    ],
)
def test_export_rejects_incompatible_qk_norms(builder, layer, norm_name, norm_type, full_precision):
    model = builder()
    attention = model.blocks[layer].attention
    original = getattr(attention, norm_name)
    replacement = LayerNormConfig(
        name=norm_type,
        eps=original.eps,
        bias=False,
        full_precision=full_precision,
    ).build(size=attention.head_dim, weight_shape=tuple(original.weight.shape))
    # The parameters round-trip exactly, but the normalization operation differs.
    replacement.load_state_dict(original.state_dict(), strict=True)
    setattr(attention, norm_name, replacement)
    with pytest.raises(NotImplementedError, match="Q/K.*RMSNorm"):
        get_hf_config(model)


@requires_fla
@pytest.mark.parametrize("layer", ["0", "1"])
def test_hybrid_export_rejects_heterogeneous_kda_norm_eps(layer):
    model = build_hybrid()
    model.blocks[layer].attention.o_norm.eps = 1e-3
    with pytest.raises(NotImplementedError, match="KDA.*epsilon"):
        get_hf_config(model)


@pytest.mark.parametrize(
    "builder,layers",
    [
        (build_attention_only, ("0", "1")),
        pytest.param(build_hybrid, ("1", "2"), marks=requires_fla),
    ],
)
@pytest.mark.parametrize("flag", ["uniform_expert_assignment", "random_expert_assignment"])
def test_export_rejects_forced_expert_assignments(builder, layers, flag):
    for layer in layers:
        model = builder(emo=False)
        setattr(model.blocks[layer].routed_experts_router, flag, True)
        with pytest.raises(NotImplementedError, match=flag):
            get_hf_config(model)


@requires_fla
@pytest.mark.parametrize("layer", ["0", "1"])
@pytest.mark.parametrize("conv_name", ["q_conv1d", "k_conv1d", "v_conv1d"])
def test_hybrid_export_rejects_biased_kda_convolutions(layer, conv_name):
    model = build_hybrid()
    conv = getattr(model.blocks[layer].attention, conv_name)
    conv.bias = torch.nn.Parameter(torch.zeros(conv.weight.shape[0]))
    with pytest.raises(NotImplementedError, match="KDA.*convolution.*bias"):
        get_hf_config(model)


@pytest.mark.parametrize(
    "builder,layers",
    [
        (build_attention_only, ("0", "1")),
        pytest.param(build_hybrid, ("2", "3"), marks=requires_fla),
    ],
)
@pytest.mark.parametrize("clip_qkv", [0.0, 1.0])
def test_export_rejects_clipped_attention(builder, layers, clip_qkv):
    for layer in layers:
        model = builder()
        if builder is build_hybrid:
            # Built blocks contain CUDA events, which cannot be deep-copied.
            model.blocks["3"] = build_hybrid().blocks["2"]
        model.blocks[layer].attention.clip_qkv = clip_qkv
        with pytest.raises(NotImplementedError, match="clip_qkv"):
            get_hf_config(model)


@pytest.mark.parametrize(
    "builder,layers",
    [
        (build_attention_only, ("0", "1")),
        pytest.param(build_hybrid, ("2", "3"), marks=requires_fla),
    ],
)
@pytest.mark.parametrize("change", ["scale_zero", "scale_one", "w_q", "w_k", "w_v", "w_out"])
def test_export_rejects_unrepresented_attention_operations(builder, layers, change):
    for layer in layers:
        model = builder()
        if builder is build_hybrid:
            # Built blocks contain CUDA events, which cannot be deep-copied.
            model.blocks["3"] = build_hybrid().blocks["2"]
        attention = model.blocks[layer].attention
        if change.startswith("scale"):
            attention.backend.scale = 0.0 if change == "scale_zero" else 1.0
            message = "softmax scale"
        else:
            projection = getattr(attention, change)
            projection.bias = torch.nn.Parameter(torch.ones(projection.weight.shape[0]))
            message = "attention.*bias"
        with pytest.raises(NotImplementedError, match=message):
            get_hf_config(model)


@pytest.mark.parametrize(
    "builder", [build_attention_only, pytest.param(build_hybrid, marks=requires_fla)]
)
def test_export_accepts_explicit_default_attention_scale(builder):
    model = builder()
    expected = get_hf_config(model).to_dict()
    for block in model.blocks.values():
        attention = block.attention
        if hasattr(attention, "backend"):
            attention.backend.scale = attention.head_dim**-0.5
    assert get_hf_config(model).to_dict() == expected


@requires_fla
def test_hybrid_export_preserves_nondefault_kda_norm_eps(tmp_path):
    model = build_hybrid(kda_norm_eps=1e-3)
    save_hf_model(tmp_path / "hf", model.state_dict(), model)
    hf = Olmo3MoeForCausalLM.from_pretrained(tmp_path / "hf")
    assert hf.config.linear_norm_eps == 1e-3
    for layer in (0, 1):
        assert hf.model.layers[layer].self_attn.o_norm.eps == 1e-3


@requires_fla
@pytest.mark.parametrize("norm_type", [LayerNormType.default, LayerNormType.qwen_rms])
@pytest.mark.parametrize("layer", ["1", "2"])
def test_hybrid_export_rejects_incompatible_latent_norm(norm_type, layer):
    model = build_hybrid(latent_norm=LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False))
    model.blocks[layer].latent_up_proj_input_norm = LayerNormConfig(
        name=norm_type, eps=1e-6, bias=False
    ).build(size=64)
    with pytest.raises(NotImplementedError, match="latent.*RMSNorm"):
        get_hf_config(model)


@pytest.mark.parametrize(
    "builder,path",
    [
        (build_attention_only, "lm_head.norm"),
        (build_attention_only, "blocks.1.attention_norm"),
        (build_attention_only, "blocks.1.feed_forward_norm"),
    ]
    + [
        pytest.param(build_hybrid, path, marks=requires_fla)
        for path in (
            "embedding_norm",
            "lm_head.norm",
            "blocks.0.attention_norm",
            "blocks.1.feed_forward_norm",
            "blocks.2.attention_norm",
            "blocks.0.attention_input_norm",
            "blocks.2.feed_forward_input_norm",
            "blocks.2.latent_up_proj_input_norm",
        )
    ],
)
@pytest.mark.parametrize("change", ["epsilon", "operation", "precision"])
def test_export_rejects_incompatible_model_norms(builder, path, change):
    model = builder(
        **(
            {"latent_norm": LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False)}
            if builder is build_hybrid
            else {}
        )
    )
    norm = model.get_submodule(path)
    if change == "epsilon":
        norm.eps = 1e-3
    elif change == "precision":
        norm.full_precision = False
    else:
        parent, _, name = path.rpartition(".")
        replacement = LayerNormConfig(name=LayerNormType.default, eps=1e-6, bias=False).build(
            size=norm.weight.numel()
        )
        replacement.load_state_dict(norm.state_dict())
        setattr(model.get_submodule(parent), name, replacement)
    with pytest.raises(NotImplementedError, match="RMSNorm"):
        get_hf_config(model)


@requires_fla
@pytest.mark.parametrize("layer", [0, 1, 2])
@pytest.mark.parametrize("count", [1, 2])
def test_hybrid_export_rejects_shared_expert_routing(layer, count):
    model = build_hybrid(shared_override=(layer, count))
    with pytest.raises(NotImplementedError, match="shared expert"):
        get_hf_config(model)


@pytest.mark.parametrize("granularity", [GateGranularity.headwise, GateGranularity.elementwise])
@pytest.mark.parametrize("full_precision", [False, True])
def test_attention_only_export_preserves_gate(tmp_path, granularity, full_precision):
    model = build_attention_only(
        gate=GateConfig(granularity=granularity, full_precision=full_precision)
    )
    save_hf_model(tmp_path / "hf", model.state_dict(), model)
    hf = Olmo3MoeForCausalLM.from_pretrained(tmp_path / "hf")
    assert hf.config.attention_gate_type == str(granularity)
    assert hf.config.attention_gate_full_precision == full_precision
    for idx, block in enumerate(model.blocks.values()):
        torch.testing.assert_close(
            hf.model.layers[idx].self_attn.g_proj.weight, block.attention.w_g.weight
        )


@requires_fla
@pytest.mark.parametrize("latent,emo", [(False, True), (True, True), (True, False)])
@pytest.mark.parametrize(
    "per_head,scalable", [(False, False), (True, False), (False, True), (True, True)]
)
def test_core_hybrid_exports_and_reloads(tmp_path, latent, emo, per_head, scalable):
    model = build_hybrid(
        latent=latent,
        emo=emo,
        per_head=per_head,
        scalable=scalable,
        latent_norm=LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False)
        if latent
        else None,
    )
    config = get_hf_config(model)
    assert config.latent_moe_dim == (64 if latent else None)
    assert config.qk_norm_per_head_gains == per_head and config.scalable_softmax == scalable
    assert config.emo_eos_token_id == (63 if emo else None)
    state = model.state_dict()
    converted = convert_state_to_hf(config, state)
    hf = Olmo3MoeForCausalLM(config)
    hf.load_state_dict(converted, strict=True, assign=True)
    restored = convert_state_from_hf(config, hf.state_dict(), model_type="olmo3moe")
    assert restored.keys() == state.keys()
    for key in state:
        torch.testing.assert_close(restored[key], state[key], rtol=0, atol=0, msg=key)
    save_hf_model(tmp_path / "hf", state, model)
    reloaded = Olmo3MoeForCausalLM.from_pretrained(tmp_path / "hf")
    assert reloaded.state_dict().keys() == converted.keys()
    for key, value in reloaded.state_dict().items():
        torch.testing.assert_close(
            value, converted[key], rtol=0, atol=0, check_dtype=False, msg=key
        )


@requires_fla
@requires_gpu
@pytest.mark.parametrize("emo", [False, True])
@pytest.mark.parametrize("kda_norm_eps", [1e-5, 1e-3])
@pytest.mark.parametrize("window", [None, 4])
def test_core_hybrid_hf_forward_parity(emo, kda_norm_eps, window):
    model = build_hybrid(
        emo=emo, device="cuda", kda_norm_eps=kda_norm_eps, window=window, scalable=window is None
    )
    config = get_hf_config(model)
    config._attn_implementation = "eager"
    hf = Olmo3MoeForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()
    hf.load_state_dict(convert_state_to_hf(config, model.state_dict()), strict=True)
    tokens = torch.randint(0, 63, (2, 32), device="cuda")
    with torch.no_grad():
        expected = model(tokens).float()
        actual = hf(tokens, use_cache=False).logits.float()
    # Different attention/expert kernels round in BF16; compare the model's predictive distribution.
    torch.testing.assert_close(
        actual.log_softmax(-1), expected.log_softmax(-1), atol=0.02, rtol=0.005
    )


@requires_fla
@pytest.mark.parametrize("per_head", [False, True])
@pytest.mark.parametrize("scalable", [False, True])
def test_core_hf_attention_forward_parity(per_head, scalable):
    model = build_hybrid(per_head=per_head, scalable=scalable).float()
    config = get_hf_config(model)
    config._attn_implementation = "eager"
    hf = Olmo3MoeForCausalLM(config).eval()
    hf.load_state_dict(convert_state_to_hf(config, model.state_dict()), strict=True)
    x = torch.randn(2, 7, model.d_model)
    # Sequence length differs from either head count; gains are deliberately non-uniform.
    mask = torch.full((7, 7), float("-inf")).triu(1)[None, None]
    with torch.no_grad():
        expected = model.blocks["2"].attention(x)
        actual, _ = hf.model.layers[2].self_attn(
            x,
            position_embeddings=None,
            attention_mask=mask,
            position_ids=torch.arange(7)[None],
        )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@requires_fla
def test_hybrid_export_rejects_restricted_emo_inference_pool():
    model = build_hybrid(emo=True)
    model.blocks["1"].routed_experts_router.emo.eval_document_expert_pool = 2
    with pytest.raises(NotImplementedError, match="eval_document_expert_pool"):
        get_hf_config(model)


@requires_fla
def test_hybrid_export_preserves_sliding_window_attention():
    model = build_hybrid(scalable=False, window=4)
    assert model.blocks["2"].attention.backend.window_size == (3, 0)
    config = get_hf_config(model)
    assert config.layer_types == ["linear_attention", "linear_attention", "sliding_attention"]
    assert config.sliding_window == 4
    restored = type(config).from_dict(config.to_dict())
    assert restored.layer_types == config.layer_types
    assert restored.sliding_window == config.sliding_window


@requires_fla
def test_hybrid_export_rejects_heterogeneous_sliding_windows():
    model = build_hybrid(scalable=False, window=4)
    model.blocks["3"] = build_hybrid(scalable=False, window=8).blocks["2"]
    with pytest.raises(NotImplementedError, match="one common sliding window size"):
        get_hf_config(model)


@requires_fla
def test_hybrid_hf_rejects_packed_positions():
    model = build_hybrid()
    hf = Olmo3MoeForCausalLM(get_hf_config(model))
    with pytest.raises(NotImplementedError, match="Packed/reset"):
        hf(torch.ones(1, 4, dtype=torch.long), position_ids=torch.tensor([[0, 1, 0, 1]]))


@pytest.mark.parametrize(
    "setting,value",
    [
        ("top_k", 1),
        ("original_top_k", 3),
        ("normalize_expert_weights", False),
        ("restore_weight_scale", True),
        ("gating_function", "sigmoid"),
    ],
)
def test_attention_only_rejects_later_router_settings(setting, value):
    model = build_attention_only()
    router = model.blocks["1"].routed_experts_router
    if getattr(router, setting) == value:
        value = not value
    setattr(router, setting, value)
    with pytest.raises(NotImplementedError, match="Heterogeneous router"):
        get_hf_config(model)


@pytest.mark.parametrize("change", ["disabled", "theta", "scaling"])
def test_attention_only_rejects_later_rope_settings(change):
    model = build_attention_only()
    attention = model.blocks["1"].attention
    if change == "disabled":
        attention.rope = None
    elif change == "theta":
        attention.rope.theta *= 2
    else:
        attention.rope.scaling = PIRoPEScalingConfig(factor=2)
    with pytest.raises(NotImplementedError, match="RoPE"):
        get_hf_config(model)


@pytest.mark.parametrize("layer", ["2", "3"])
def test_attention_only_rejects_non_silu_dense_layers(layer):
    model = build_attention_only()
    for idx in ("2", "3"):
        model.blocks[idx] = ReorderedNormTransformerBlock(
            d_model=32,
            block_idx=int(idx),
            n_layers=4,
            sequence_mixer=AttentionConfig(n_heads=4, head_dim=8, bias=False),
            feed_forward=FeedForwardConfig(hidden_size=32, bias=False),
            layer_norm=LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False),
        )
        model.blocks[idx].attention = deepcopy(model.blocks["0"].attention)
    get_hf_config(model)
    model.blocks[layer].feed_forward.activation_fn = ActivationFunction.gelu_tanh.build()
    with pytest.raises(NotImplementedError, match="SiLU"):
        get_hf_config(model)


def test_emo_tp_rejected_before_mutating_model():
    model = build_attention_only(emo=True)
    parameters = dict(model.named_parameters())
    with pytest.raises(OLMoConfigurationError, match="tensor parallelism"):
        model.apply_tp(None)
    assert not model._tp_enabled
    for name, parameter in model.named_parameters():
        assert parameter is parameters[name]


@requires_fla
@pytest.mark.parametrize("layer", ["0", "1"])
def test_hybrid_export_reports_experimental_backend_change(caplog, layer):
    model = build_hybrid()
    expected = get_hf_config(model).to_dict()
    # Config/export coverage only; GPU KDA tests exercise the actual supported shapes.
    model.blocks[layer].attention.use_experimental_kernels = True
    assert get_hf_config(model).to_dict() == expected
    assert f"source experimental layers [{layer}]" in caplog.text
    assert "representative production lengths" in caplog.text
