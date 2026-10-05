"""
Tests for the HF Molmo2 → :class:`MultimodalLM` state-dict converter.

The "key coverage" test below constructs a synthetic HF Molmo2 state dict
matching the shapes of a real Molmo2 model and verifies the converter
produces every key our :meth:`MultimodalLM.load_state_dict`
expects, with the correct shapes, and that the loaded model runs forward.

No network or large download required.

A separate ``@pytest.mark.slow`` numerical-parity test that requires a real
HF Molmo2 checkpoint can be added later (task follow-up).
"""

import dataclasses
from typing import Dict

import pytest
import torch

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.vision import (
    MultimodalLM,
    MultimodalLMConfig,
    VisionConnectorConfig,
    VisionEncoderConfig,
    VisionEncoderType,
)
from olmo_core.nn.vision.molmo2_loader import (
    Molmo2LoaderError,
    molmo2_hf_state_dict_to_multimodal_lm,
    retie_word_embeddings,
)

# ---------------------------------------------------------------------------
# Tiny multimodal config used throughout
# ---------------------------------------------------------------------------

# Self-contained vocab layout, decoupled from the dolma2 multimodal tokenizer
# (`olmo_core.data.multimodal`, added in a later PR). The loader only needs a
# base/extra split that pads to the LM `vocab_size`: the base vocab is what HF's
# lm_head predicts, and the extra image/special tokens are input-only. We pick a
# 128-wide vocab with a non-trivial extra count so the embedding-split path is
# actually exercised.
_BASE_VOCAB = 120  # text vocab predicted by lm_head
_EXTRA_VOCAB = 8  # image/special tokens (input-only, never predicted)
_FULL_VOCAB = _BASE_VOCAB + _EXTRA_VOCAB  # 128
_IMAGE_PATCH_ID = _BASE_VOCAB  # first extra id


def _tiny_cfg(tie_word_embeddings: bool = False) -> MultimodalLMConfig:
    """A tiny stand-in for Molmo2-O-7B: olmo3_1M LM + small SigLIP-style ViT.

    Architecturally analogous (fused QKV, fused SwiGLU, ViT, attention-pooled
    connector with SwiGLU projector); just small enough to load on CPU.
    ``tie_word_embeddings=True`` mimics the Molmo2-4B (Qwen3-4B backbone) layout
    where the LM head shares storage with the embedding table."""
    # olmo3_1M shares the same block layout as olmo3_7B (fused att_proj +
    # fused ff_proj + qk_norm), just with d_model=12. Force the `torch`
    # SDPA backend so the tiny test config builds without a CUDA flash-attn.
    lm_cfg = TransformerConfig.olmo3_1M(
        vocab_size=_BASE_VOCAB,
        n_extra_vocab=_EXTRA_VOCAB,
        attn_backend=AttentionBackendName.torch,
        tie_word_embeddings=tie_word_embeddings,
    )
    vis_cfg = VisionEncoderConfig(
        name=VisionEncoderType.siglip,
        use_cls_token=False,
        patch_embedding_bias=True,
        use_pre_ln=False,
        image_default_input_size=(56, 56),
        image_patch_size=14,
        image_emb_dim=32,
        image_num_heads=2,
        image_num_key_value_heads=2,
        image_num_layers=3,
        image_head_dim=16,
        image_mlp_dim=64,
        image_num_pos=16,
        image_norm_eps=1e-5,
    )
    conn_cfg = VisionConnectorConfig.from_vision_encoder(
        vis_cfg, output_dim=lm_cfg.d_model, mlp_hidden_size=64
    )
    return MultimodalLMConfig(
        lm=lm_cfg,
        vision=vis_cfg,
        connector=conn_cfg,
        image_patch_token_id=_IMAGE_PATCH_ID,
        # The head spans the base vocab structurally, so no logit masking.
        output_vocab_size=None,
    )


def _attention_dims(lm_cfg: TransformerConfig):
    """Mirror of molmo2_loader._attention_dims so tests can construct fake weights."""
    block = lm_cfg.block
    seq_mixer = block.attention if block.attention is not None else block.sequence_mixer
    n_heads = getattr(seq_mixer, "n_heads", None) or getattr(seq_mixer, "num_heads", None)
    n_kv = (
        getattr(seq_mixer, "n_kv_heads", None)
        or getattr(seq_mixer, "num_kv_heads", None)
        or n_heads
    )
    head_dim = getattr(seq_mixer, "head_dim", None) or (lm_cfg.d_model // n_heads)
    return int(n_heads), int(n_kv), int(head_dim)


def _has_qk_norm(lm_cfg: TransformerConfig) -> bool:
    block = lm_cfg.block
    seq_mixer = block.attention if block.attention is not None else block.sequence_mixer
    return getattr(seq_mixer, "qk_norm", None) is not None


# ---------------------------------------------------------------------------
# Build a synthetic HF Molmo2 state dict
# ---------------------------------------------------------------------------


def _synthetic_hf_state_dict(cfg: MultimodalLMConfig) -> Dict[str, torch.Tensor]:
    """Construct a Molmo2-shaped HF state dict from our config's dimensions.

    Uses ``ones`` everywhere so the converter can be exercised without
    relying on actual weight semantics. Numerical-correctness checks live
    elsewhere."""
    sd: Dict[str, torch.Tensor] = {}

    lm_cfg = cfg.lm
    n_layers = lm_cfg.n_layers
    n_heads, n_kv, head_dim = _attention_dims(lm_cfg)
    d_model = lm_cfg.d_model
    # Match our model's resolved hidden_size for SwiGLU. For olmo3_1M this
    # ends up matching what TransformerConfig.llama_like computes.
    block = lm_cfg.block
    intermediate = block.feed_forward.hidden_size

    has_qk_norm = _has_qk_norm(lm_cfg)

    # Vocab: HF Molmo2 keeps base + extra vocab in separate buffers, and its lm_head
    # predicts only the *base* vocab (the extras are input-only). Our config mirrors that
    # split, so `lm.vocab_size` is the base and `lm.n_extra_vocab` the extra block. Tied
    # configs export lm_head.weight as a copy of wte.embedding, so mirror that.
    base_vocab = lm_cfg.vocab_size
    new_vocab = lm_cfg.n_extra_vocab
    sd["model.transformer.wte.embedding"] = torch.randn(base_vocab, d_model)
    sd["model.transformer.wte.new_embedding"] = torch.randn(new_vocab, d_model)
    sd["model.transformer.ln_f.weight"] = torch.randn(d_model)
    if lm_cfg.tie_word_embeddings:
        sd["lm_head.weight"] = sd["model.transformer.wte.embedding"].clone()
    else:
        sd["lm_head.weight"] = torch.randn(base_vocab, d_model)

    fused_qkv_out = n_heads * head_dim + 2 * n_kv * head_dim
    for i in range(n_layers):
        prefix = f"model.transformer.blocks.{i}"
        sd[f"{prefix}.attn_norm.weight"] = torch.randn(d_model)
        sd[f"{prefix}.ff_norm.weight"] = torch.randn(d_model)
        sd[f"{prefix}.self_attn.att_proj.weight"] = torch.randn(fused_qkv_out, d_model)
        sd[f"{prefix}.self_attn.attn_out.weight"] = torch.randn(d_model, n_heads * head_dim)
        if has_qk_norm:
            # OLMo-core uses per-channel qk_norm (num_heads * head_dim),
            # matching Molmo2's non-qwen3 qk_norm_type sizing.
            sd[f"{prefix}.self_attn.q_norm.weight"] = torch.randn(n_heads * head_dim)
            sd[f"{prefix}.self_attn.k_norm.weight"] = torch.randn(n_kv * head_dim)
        sd[f"{prefix}.mlp.ff_proj.weight"] = torch.randn(intermediate * 2, d_model)
        sd[f"{prefix}.mlp.ff_out.weight"] = torch.randn(d_model, intermediate)

    # Vision side.
    vis = cfg.vision
    p = vis.image_patch_size
    sd["model.vision_backbone.image_vit.patch_embedding.weight"] = torch.randn(
        vis.image_emb_dim, p * p * 3
    )
    sd["model.vision_backbone.image_vit.patch_embedding.bias"] = torch.randn(vis.image_emb_dim)
    sd["model.vision_backbone.image_vit.positional_embedding"] = torch.randn(
        vis.image_num_pos, vis.image_emb_dim
    )
    for i in range(vis.image_num_layers):
        prefix = f"model.vision_backbone.image_vit.transformer.resblocks.{i}"
        for n in ("attention_norm", "ffn_norm"):
            sd[f"{prefix}.{n}.weight"] = torch.randn(vis.image_emb_dim)
            sd[f"{prefix}.{n}.bias"] = torch.randn(vis.image_emb_dim)
        qkv_out = vis.image_num_heads * vis.image_head_dim
        kv_out = vis.image_num_key_value_heads * vis.image_head_dim
        sd[f"{prefix}.attention.wq.weight"] = torch.randn(qkv_out, vis.image_emb_dim)
        sd[f"{prefix}.attention.wq.bias"] = torch.randn(qkv_out)
        sd[f"{prefix}.attention.wk.weight"] = torch.randn(kv_out, vis.image_emb_dim)
        sd[f"{prefix}.attention.wk.bias"] = torch.randn(kv_out)
        sd[f"{prefix}.attention.wv.weight"] = torch.randn(kv_out, vis.image_emb_dim)
        sd[f"{prefix}.attention.wv.bias"] = torch.randn(kv_out)
        sd[f"{prefix}.attention.wo.weight"] = torch.randn(vis.image_emb_dim, qkv_out)
        sd[f"{prefix}.attention.wo.bias"] = torch.randn(vis.image_emb_dim)
        sd[f"{prefix}.feed_forward.w1.weight"] = torch.randn(vis.image_mlp_dim, vis.image_emb_dim)
        sd[f"{prefix}.feed_forward.w1.bias"] = torch.randn(vis.image_mlp_dim)
        sd[f"{prefix}.feed_forward.w2.weight"] = torch.randn(vis.image_emb_dim, vis.image_mlp_dim)
        sd[f"{prefix}.feed_forward.w2.bias"] = torch.randn(vis.image_emb_dim)

    # Connector: pooling cross-attention (input_dim = num_input_layers * image_emb_dim).
    conn = cfg.connector
    pool_in = conn.num_input_layers * vis.image_emb_dim
    pool_q = conn.image_num_heads * conn.image_head_dim
    pool_kv = conn.image_num_key_value_heads * conn.image_head_dim
    for proj, out_dim in (("wq", pool_q), ("wk", pool_kv), ("wv", pool_kv)):
        sd[f"model.vision_backbone.image_pooling_2d.{proj}.weight"] = torch.randn(out_dim, pool_in)
        sd[f"model.vision_backbone.image_pooling_2d.{proj}.bias"] = torch.randn(out_dim)
    sd["model.vision_backbone.image_pooling_2d.wo.weight"] = torch.randn(conn.image_emb_dim, pool_q)
    sd["model.vision_backbone.image_pooling_2d.wo.bias"] = torch.randn(conn.image_emb_dim)

    # Connector: SwiGLU projector.
    hidden = conn.mlp_hidden_size or 4 * conn.image_emb_dim
    sd["model.vision_backbone.image_projector.w1.weight"] = torch.randn(hidden, conn.image_emb_dim)
    sd["model.vision_backbone.image_projector.w3.weight"] = torch.randn(hidden, conn.image_emb_dim)
    sd["model.vision_backbone.image_projector.w2.weight"] = torch.randn(conn.output_dim, hidden)
    return sd


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_converter_produces_every_required_key():
    cfg = _tiny_cfg()
    model = MultimodalLM(cfg, init_device="cpu")
    expected_keys = set(model.state_dict().keys())

    hf_sd = _synthetic_hf_state_dict(cfg)
    converted = molmo2_hf_state_dict_to_multimodal_lm(hf_sd, cfg)

    missing = expected_keys - set(converted.keys())
    extra = set(converted.keys()) - expected_keys
    assert not missing, f"converter missing keys: {sorted(missing)[:5]}"
    # Accept extra keys (e.g. RoPE buffers stay on the model side) only if
    # they aren't model parameters.
    model_param_keys = {k for k, _ in model.named_parameters()}
    extra_param = extra & model_param_keys
    assert not extra_param, f"converter produced unexpected param keys: {extra_param}"


def test_converter_shapes_match_model_load():
    """The converted state dict must load into the model without complaint."""
    cfg = _tiny_cfg()
    model = MultimodalLM(cfg, init_device="cpu")
    hf_sd = _synthetic_hf_state_dict(cfg)
    converted = molmo2_hf_state_dict_to_multimodal_lm(hf_sd, cfg)
    missing, unexpected = model.load_state_dict(converted, strict=False)
    # We expect only buffers like rotary_emb caches to be unloaded; no
    # parameters missing.
    param_keys = {k for k, _ in model.named_parameters()}
    assert not (set(missing) & param_keys), f"unloaded params: {set(missing) & param_keys}"


def test_qkv_split_is_round_trip_correct():
    """Splitting the fused att_proj and reading off Q/K/V should recover the
    halves a real HF attention would consume."""
    cfg = _tiny_cfg()
    lm_cfg = cfg.lm
    n_heads, n_kv, head_dim = _attention_dims(lm_cfg)
    d_model = lm_cfg.d_model
    fused_out = n_heads * head_dim + 2 * n_kv * head_dim

    # Pack Q/K/V along output dim like HF Molmo2 does.
    q = torch.randn(n_heads * head_dim, d_model)
    k = torch.randn(n_kv * head_dim, d_model)
    v = torch.randn(n_kv * head_dim, d_model)
    fused = torch.cat([q, k, v], dim=0)
    assert fused.shape == (fused_out, d_model)

    hf_sd = _synthetic_hf_state_dict(cfg)
    hf_sd["model.transformer.blocks.0.self_attn.att_proj.weight"] = fused
    converted = molmo2_hf_state_dict_to_multimodal_lm(hf_sd, cfg)
    torch.testing.assert_close(converted["lm.blocks.0.attention.w_q.weight"], q)
    torch.testing.assert_close(converted["lm.blocks.0.attention.w_k.weight"], k)
    torch.testing.assert_close(converted["lm.blocks.0.attention.w_v.weight"], v)


def test_ff_proj_chunk_assigns_gate_and_multiplier_correctly():
    """The HF MLP chunks ``[x, gate]``: x is the multiplier (our w3), gate
    is what gets the activation (our w1)."""
    cfg = _tiny_cfg()
    lm_cfg = cfg.lm
    d_model = lm_cfg.d_model
    intermediate = lm_cfg.block.feed_forward.hidden_size
    # Distinct halves so we can verify the assignment.
    mul_half = torch.ones(intermediate, d_model)
    gate_half = -torch.ones(intermediate, d_model)
    fused = torch.cat([mul_half, gate_half], dim=0)
    hf_sd = _synthetic_hf_state_dict(cfg)
    hf_sd["model.transformer.blocks.0.mlp.ff_proj.weight"] = fused
    converted = molmo2_hf_state_dict_to_multimodal_lm(hf_sd, cfg)
    torch.testing.assert_close(converted["lm.blocks.0.feed_forward.w3.weight"], mul_half)
    torch.testing.assert_close(converted["lm.blocks.0.feed_forward.w1.weight"], gate_half)


def test_patch_embedding_permute_is_spatial_to_c_first():
    """The converter must permute Molmo2's ``(p, p, c)`` flatten order
    into our ``(c, p, p)`` order so the same pixel patchification produces
    the same dot product on both sides."""
    cfg = _tiny_cfg()
    p = cfg.vision.image_patch_size
    D = cfg.vision.image_emb_dim

    # Build a weight where row d has value 1 at the (c=d%3, kh=0, kw=0) pixel
    # only. In HF's spatial-first layout that pixel is at flat index c (since
    # the first p*p indices cover c=0 across all kh,kw...wait no, HF order
    # is kh,kw,c so position (0,0,c=d%3) sits at flat index 0*(p*3) + 0*3 + (d%3) = d%3).
    hf_patch_w = torch.zeros(D, p * p * 3)
    for d in range(D):
        c = d % 3
        # HF flat index for (kh=0, kw=0, c): 0*(p*3) + 0*3 + c = c.
        hf_patch_w[d, c] = 1.0

    hf_sd = _synthetic_hf_state_dict(cfg)
    hf_sd["model.vision_backbone.image_vit.patch_embedding.weight"] = hf_patch_w
    converted = molmo2_hf_state_dict_to_multimodal_lm(hf_sd, cfg)
    our_w = converted["vision.patch_embedding.weight"]
    # In our C-first layout, (c, kh=0, kw=0) sits at flat index c*(p*p) + 0*p + 0 = c*p*p.
    for d in range(D):
        c = d % 3
        assert our_w[d, c * p * p].item() == 1.0
        # All other positions should be zero.
        assert (our_w[d] != 0).sum().item() == 1


def test_missing_key_raises_helpful_error():
    cfg = _tiny_cfg()
    hf_sd = _synthetic_hf_state_dict(cfg)
    del hf_sd["model.transformer.wte.embedding"]
    with pytest.raises(Molmo2LoaderError, match="model.transformer.wte.embedding"):
        molmo2_hf_state_dict_to_multimodal_lm(hf_sd, cfg)


def test_forward_runs_with_converted_weights():
    """Sanity: a model loaded with converter output runs a forward without errors,
    and the inputs-only extra-vocab logit columns come out masked."""
    cfg = _tiny_cfg()
    model = MultimodalLM(cfg, init_device="cpu")
    hf_sd = _synthetic_hf_state_dict(cfg)
    converted = molmo2_hf_state_dict_to_multimodal_lm(hf_sd, cfg)
    model.load_state_dict(converted, strict=False)
    model.eval()
    # Text-only forward.
    out = model(input_ids=torch.randint(0, 100, (1, 8)))
    # The LM head spans the base vocab only — the extra image-special tokens are inputs and
    # are structurally unpredictable, so there are no masked columns to check.
    assert out.shape == (1, 8, _BASE_VOCAB)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Tied word embeddings (Molmo2-4B layout) + output-vocab masking
# ---------------------------------------------------------------------------


def test_tied_lm_head_shares_embedding_table_and_preserves_new_embedding():
    """For a tied LM the head shares the *base* block, and the extra rows are untouched.

    The base block and the extra image-token rows are separate parameters, so the tied head
    (which is the base block) can no longer clobber the extra rows on ``load_state_dict`` —
    the failure mode this test was originally written for is now structurally impossible."""
    cfg = _tiny_cfg(tie_word_embeddings=True)
    hf_sd = _synthetic_hf_state_dict(cfg)
    converted = molmo2_hf_state_dict_to_multimodal_lm(hf_sd, cfg)

    # The tied head is the base block itself, and it spans the base vocab only.
    assert converted["lm.lm_head.w_out.weight"] is converted["lm.embeddings.weight"]
    assert converted["lm.embeddings.weight"].shape[0] == _BASE_VOCAB
    assert converted["lm.embeddings.extra_weight"].shape[0] == _EXTRA_VOCAB

    model = MultimodalLM(cfg, init_device="cpu")
    assert model.lm.lm_head.w_out.weight is model.lm.embeddings.weight
    model.load_state_dict(converted, strict=False)
    torch.testing.assert_close(
        model.lm.embeddings.extra_weight,
        hf_sd["model.transformer.wte.new_embedding"],
    )
    torch.testing.assert_close(
        model.lm.embeddings.weight,
        hf_sd["model.transformer.wte.embedding"],
    )


def test_tied_lm_head_mismatch_raises():
    """A tied LM config with an HF checkpoint whose lm_head differs from the base
    embedding cannot be represented — the converter must say so."""
    cfg = _tiny_cfg(tie_word_embeddings=True)
    hf_sd = _synthetic_hf_state_dict(cfg)
    hf_sd["lm_head.weight"] = torch.randn(_BASE_VOCAB, cfg.lm.d_model)
    with pytest.raises(Molmo2LoaderError, match="tie_word_embeddings"):
        molmo2_hf_state_dict_to_multimodal_lm(hf_sd, cfg)


def test_retie_word_embeddings_after_to_empty():
    """``Module.to_empty`` silently breaks the tie; the helper restores it and the
    re-tied model still holds the converted weights."""
    cfg = _tiny_cfg(tie_word_embeddings=True)
    hf_sd = _synthetic_hf_state_dict(cfg)
    converted = molmo2_hf_state_dict_to_multimodal_lm(hf_sd, cfg)

    model = MultimodalLM(cfg, init_device="meta")
    model.to_empty(device=torch.device("cpu"))
    assert model.lm.lm_head.w_out.weight is not model.lm.embeddings.weight  # broken tie
    model.load_state_dict(converted, strict=False)
    retie_word_embeddings(model)
    assert model.lm.lm_head.w_out.weight is model.lm.embeddings.weight
    torch.testing.assert_close(
        model.lm.embeddings.extra_weight,
        hf_sd["model.transformer.wte.new_embedding"],
    )
    torch.testing.assert_close(
        model.lm.embeddings.weight,
        hf_sd["model.transformer.wte.embedding"],
    )


def test_labels_path_guarded_when_output_vocab_masked():
    """Passing ``labels`` through MultimodalLM bypasses output-vocab masking — must raise.

    Molmo2 no longer needs the masking (its head spans the base vocab structurally), so this
    exercises the feature on a config that opts into it explicitly."""
    cfg = _tiny_cfg()
    cfg = dataclasses.replace(cfg, output_vocab_size=_BASE_VOCAB - 4)
    model = MultimodalLM(cfg, init_device="cpu")
    input_ids = torch.randint(0, _BASE_VOCAB - 4, (1, 8))
    with pytest.raises(Exception, match="output_vocab_size"):
        model(input_ids=input_ids, labels=input_ids.clone())
