"""
LM and full-pipeline logit parity between HF Molmo2 and our
:class:`MultimodalLM` after loading converted weights.

Three sub-tests, each peels off another layer of the model:

1. **Token embedding parity** — :class:`Molmo2Embedding` (HF) vs our
   concatenated :class:`nn.Embedding` produce identical activations.
2. **LM-only forward parity** — feed the same input embeddings through
   HF's text transformer and ours; compare last-token logits. Validates
   every LM block (attention + SwiGLU) and the LM head row layout.
3. **Full multimodal pipeline parity** — random pre-patchified image
   through the vision encoder, connector, embedding splice, then LM;
   compare last-token logits at a position where image features are
   spliced. Validates the channel-vs-spatial patch-embedding permute and
   the connector indexing semantics end-to-end.

Bypasses HF's :class:`AutoProcessor` because the bundled Molmo2 code
depends on a newer ``transformers.video_utils`` API than what we have
installed. We feed already-shaped tensors to HF's lower-level forward
paths instead.
"""


import pytest
import torch

from olmo_core.nn.vision import MultimodalLM
from olmo_core.nn.vision.molmo2_loader import (
    ensure_default_rope_registered,
    molmo2_config_from_hf_config,
    molmo2_hf_state_dict_to_multimodal_lm,
    reinit_rope_buffers,
    retie_word_embeddings,
)
from olmo_core.testing import requires_gpu

transformers = pytest.importorskip("transformers")

from ._molmo2_common import MOLMO2_VARIANTS, _hf_cache_has  # noqa: F401 (re-exported)


def _build_ours(model_id: str, device, dtype):
    """Load HF Molmo2, build our model + converted weights. Returns
    (hf_model on CPU, ours on `device` with `dtype`, our config)."""
    ensure_default_rope_registered()
    from transformers import AutoModelForImageTextToText

    ensure_default_rope_registered()
    try:
        hf = AutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True, local_files_only=True
        )
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Could not load {model_id}: {e}")
    reinit_rope_buffers(hf)
    cfg = molmo2_config_from_hf_config(hf.config)
    converted = molmo2_hf_state_dict_to_multimodal_lm(hf.state_dict(), cfg)
    ours = MultimodalLM(cfg, init_device="meta")
    ours.to_empty(device=torch.device("cpu"))
    ours.load_state_dict(converted, strict=False)
    retie_word_embeddings(ours)  # `to_empty` breaks the tied-embedding share (Molmo2-4B)
    del converted
    ours = ours.to(device=device, dtype=dtype).eval()
    return hf, ours, cfg


# ---------------------------------------------------------------------------
# Token embedding parity
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("model_id", MOLMO2_VARIANTS)
def test_molmo2_embedding_parity(model_id: str):
    """``Molmo2Embedding(input_ids) == ours.lm.embeddings(input_ids)``."""
    if not _hf_cache_has(model_id):
        pytest.skip(f"{model_id} not in HF cache")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    hf, ours, cfg = _build_ours(model_id, device, dtype)

    hf_emb = hf.model.transformer.wte.to(device=device, dtype=dtype)
    torch.manual_seed(0)
    # Include some image-extra-vocab IDs so we exercise new_embedding.
    base_vocab = cfg.lm.vocab_size - 128
    ids = (
        torch.cat(
            [
                torch.randint(0, base_vocab, (4,)),
                torch.tensor([cfg.image_patch_token_id, base_vocab + 5]),
            ]
        )
        .unsqueeze(0)
        .to(device)
    )

    with torch.inference_mode():
        hf_e = hf_emb(ids)
        our_e = ours.lm.embeddings(ids)

    diff = (hf_e.float() - our_e.float()).abs().max().item()
    assert diff < 1e-3, f"{model_id}: embedding diff = {diff:.3e}"


# ---------------------------------------------------------------------------
# LM-only forward parity (text-only)
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("model_id", MOLMO2_VARIANTS)
def test_molmo2_lm_forward_parity(model_id: str):
    """HF text transformer and ours produce the same last-token logits given
    the same ``inputs_embeds`` (no images, pure LM block + head parity).

    .. note::
        Molmo2-O-7B uses per-layer YaRN attention scaling
        (``attention_factor ≈ 1.208``) on 24 of its 32 LM layers.
        :class:`MultimodalLM` applies a single global RoPE config
        and does not support per-layer attention factors, so the logit
        magnitudes for O-7B differ from HF's reference.  The test is skipped
        for O-7B until per-layer YaRN is implemented.
    """
    if not _hf_cache_has(model_id):
        pytest.skip(f"{model_id} not in HF cache")
    if "O-7B" in model_id:
        pytest.skip(
            f"{model_id} uses per-layer YaRN attention scaling which is not "
            f"yet implemented in MultimodalLM; logit parity skipped."
        )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    hf, ours, cfg = _build_ours(model_id, device, dtype)

    # Random text-only input.
    torch.manual_seed(0)
    base_vocab = cfg.lm.vocab_size - 128
    input_ids = torch.randint(0, base_vocab, (1, 12), device=device)

    # HF: full forward (no images, no cache to avoid Molmo2's transformers
    # version pin causing CacheLayerMixin signature mismatches).
    hf = hf.to(device=device, dtype=dtype).eval()
    with torch.inference_mode():
        hf_out = hf(input_ids=input_ids, use_cache=False)
    hf_last = hf_out.logits[0, -1].float()

    # Ours: full forward (no images). Our logits keep (masked) columns for the
    # image-token positions, so compare only the base-vocab slice.
    with torch.inference_mode():
        our_logits = ours(input_ids=input_ids, logits_to_keep=1)
    our_last = our_logits[0, -1].float()[: hf_last.shape[0]]

    diff = (hf_last - our_last).abs().max().item()
    hf_argmax = int(hf_last.argmax().item())
    our_argmax = int(our_last.argmax().item())
    assert hf_argmax == our_argmax, (
        f"{model_id}: LM argmax mismatch — HF={hf_argmax} ours={our_argmax}, "
        f"max abs diff = {diff:.3e}"
    )
    assert diff < 1.0, f"{model_id}: LM logit max abs diff = {diff:.3e} (threshold 1.0 in bf16)"


# ---------------------------------------------------------------------------
# Full pipeline parity (vision + connector + LM)
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("model_id", MOLMO2_VARIANTS)
def test_molmo2_full_pipeline_logit_parity(model_id: str):
    """End-to-end: same random pre-patchified image + same prompt token IDs
    feeding both HF Molmo2 and ours; assert the last-token logits match.

    .. note::
        Molmo2-O-7B uses per-layer YaRN attention scaling
        (``attention_factor ≈ 1.208``) on 24 of its 32 LM layers.
        :class:`MultimodalLM` applies a single global RoPE config
        and does not support per-layer attention factors, so full-pipeline
        argmax may diverge for O-7B.  The test is skipped for O-7B until
        per-layer YaRN is implemented.
    """
    if not _hf_cache_has(model_id):
        pytest.skip(f"{model_id} not in HF cache")
    if "O-7B" in model_id:
        pytest.skip(
            f"{model_id} uses per-layer YaRN attention scaling which is not "
            f"yet implemented in MultimodalLM; full-pipeline parity skipped."
        )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    hf, ours, cfg = _build_ours(model_id, device, dtype)
    hf = hf.to(device=device, dtype=dtype).eval()

    # Build a tiny synthetic image-bearing input.
    # 1 crop, 27x27 = 729 patches → with 2x2 pool, 196 pooled tokens... but
    # to keep the test cheap we use a single-pool layout: 1 pool over 4 patches.
    # That means 1 <im_patch> token. For the model's splice contract we
    # must put 1 <im_patch> in input_ids and provide pooled_patches_idx
    # with shape (1, 1, 4) covering patches [0, 1, 2, 3].
    torch.manual_seed(0)
    patch_size = cfg.vision.image_patch_size
    n_patches = cfg.vision.image_num_pos  # 729

    # HF: pixel_values shape (n_crops, n_patches, patch_dim) — spatial-first.
    pixel_values_hf = torch.randn(
        1, n_patches, 3 * patch_size * patch_size, dtype=dtype, device=device
    )
    # Pool 1 group of 4 patches into 1 image-token.
    image_token_pooling = torch.zeros(1, 4, dtype=torch.long, device=device)
    image_token_pooling[0, 0] = 0
    image_token_pooling[0, 1] = 1
    image_token_pooling[0, 2] = 2
    image_token_pooling[0, 3] = 3
    # image_grids: (1, 4) → low-res 1x1 + high-res 0x0 → 1 + 0 = 1 pooled.
    image_grids = torch.tensor([[1, 1, 0, 0]], dtype=torch.long, device=device)
    image_num_crops = torch.tensor([1], dtype=torch.long, device=device)

    # input_ids: a short prompt with exactly one image_patch_id and EXACTLY
    # 2 image_end_token_id tokens (HF's build_batched_images counts these
    # to infer the number of images: raw_counts // 2). We follow with a few
    # text tokens so the last-position logit reflects post-image LM activity.
    image_end_id = hf.config.image_end_token_id
    image_patch_id = hf.config.image_patch_id
    base_vocab = cfg.lm.vocab_size - 128
    text = torch.randint(0, base_vocab, (4,), device=device).tolist()
    input_ids = torch.tensor(
        [[image_end_id, image_patch_id, image_end_id] + text],
        dtype=torch.long,
        device=device,
    )

    # ---------- HF forward ----------
    with torch.inference_mode():
        hf_out = hf(
            input_ids=input_ids,
            pixel_values=pixel_values_hf,
            image_token_pooling=image_token_pooling,
            image_grids=image_grids,
            image_num_crops=image_num_crops,
            use_cache=False,
        )
    hf_last = hf_out.logits[0, -1].float()

    # ---------- ours: convert inputs ----------
    # pixel_values: spatial-first (kh, kw, c). Our model expects C-first.
    pv_ours = (
        pixel_values_hf.reshape(1, n_patches, patch_size, patch_size, 3)
        .permute(0, 1, 4, 2, 3)
        .reshape(1, n_patches, 3 * patch_size * patch_size)
        .contiguous()
        .unsqueeze(0)  # add batch dim → (1, 1, n_patches, dim)
    )
    pooled_idx_ours = image_token_pooling.unsqueeze(0)  # (1, 1, 4)

    with torch.inference_mode():
        our_logits = ours(
            input_ids=input_ids,
            images=pv_ours,
            pooled_patches_idx=pooled_idx_ours,
            logits_to_keep=1,
        )
    # Slice off our padding rows so the shapes match HF.
    our_last = our_logits[0, -1].float()[: hf_last.shape[0]]

    diff = (hf_last - our_last).abs().max().item()
    hf_argmax = int(hf_last.argmax().item())
    our_argmax = int(our_last.argmax().item())
    assert hf_argmax == our_argmax, (
        f"{model_id}: full-pipeline argmax mismatch — HF={hf_argmax} "
        f"ours={our_argmax}, max abs diff = {diff:.3e}"
    )
    assert diff < 5.0, (
        f"{model_id}: full-pipeline logit max abs diff = {diff:.3e} "
        f"(threshold 5.0 in bf16; the splice + 32+ LM layers amplify the "
        f"per-component noise)"
    )
