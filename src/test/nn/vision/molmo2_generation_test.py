"""
End-to-end generation test for the HF Molmo2 → :class:`MultimodalLM` converter.

Loads Molmo2-4B from the local HF cache, converts its weights with our loader,
then performs a short greedy-decode run on a synthetic image to verify that:

1. Both the HF reference model and our model produce **identical** token IDs
   for the first :data:`_N_TOKENS` decode steps.
2. The decoded text is non-empty, contains ASCII letters, and matches between
   the two models — demonstrating that the full vision+connector+LM pipeline
   works end-to-end without :mod:`torchvision`.

The test uses a **PIL-based preprocessor** that replaces the torchvision resize
in :class:`Molmo2ImageProcessor` with :meth:`PIL.Image.Image.resize`.  The
image is a tiny synthetic gradient (no network access needed).  Because both
the HF and our models receive the same pre-normalised patch tensors, any
difference in generated token IDs is a real bug in the converter.

Auto-skips if ``allenai/Molmo2-4B`` is not in the local HF cache.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from olmo_core.nn.vision import MultimodalLM
from olmo_core.nn.vision.molmo2_loader import (
    ensure_default_rope_registered,
    molmo2_config_from_hf_config,
    molmo2_hf_state_dict_to_multimodal_lm,
    reinit_rope_buffers,
    retie_word_embeddings,
)
from olmo_core.testing import requires_gpu

from ._molmo2_common import _hf_cache_has  # noqa: F401

transformers = pytest.importorskip("transformers")

# Only test the smallest variant to keep peak memory reasonable.
_MODEL_ID = "allenai/Molmo2-4B"

# How many tokens to decode (enough to see structure, small enough to be fast).
_N_TOKENS = 8

# ──────────────────────────── preprocessor constants ────────────────────────
# Taken from Molmo2-4B preprocessor_config.json.  Fixed across all Molmo2
# variants that use the SigLIP2-SO400M/14 vision encoder.
_PATCH_SIZE = 14
_IMAGE_SIZE = 378  # 27 × 27 patches per crop
_POOL_H = 2
_POOL_W = 2
# After 2×2 pooling over 27 patches with ceil-padding:
#   ceil(27/2) = 14  → 14×14 = 196 low-res image tokens
_GRID_H = 14
_GRID_W = 14

# Molmo2 special token IDs (same across all variants).
_IM_PATCH_ID = 151938  # <im_patch>
_IM_COL_ID = 151939  # <im_col>
_IM_START_ID = 151936  # <im_start>   (high-res section boundary)
_LOW_RES_IM_START_ID = 151940  # <low_res_im_start>
_IM_END_ID = 151937  # <im_end>
_IMAGE_PLACEHOLDER_ID = 151941  # <|image|>  (replaced in token sequence)


# ────────────────────────────── helpers ─────────────────────────────────────


def _arange_for_pooling(idx_arr: np.ndarray, pool_h: int, pool_w: int) -> np.ndarray:
    """Pad *idx_arr* to be divisible by pool dims, then return ``(ph, pw, pool_h*pool_w)``
    pooling indices. Mirrors the ``arange_for_pooling`` function in
    ``image_processing_molmo2.py`` but uses plain NumPy instead of einops."""
    h, w = idx_arr.shape
    h_pad = pool_h * ((h + pool_h - 1) // pool_h) - h
    w_pad = pool_w * ((w + pool_w - 1) // pool_w) - w
    idx_arr = np.pad(
        idx_arr,
        [[h_pad // 2, (h_pad + 1) // 2], [w_pad // 2, (w_pad + 1) // 2]],
        mode="constant",
        constant_values=-1,
    )
    ph = (h + pool_h - 1) // pool_h
    pw = (w + pool_w - 1) // pool_w
    # einops "(h dh) (w dw) -> h w (dh dw)" equivalent:
    idx_arr = idx_arr.reshape(ph, pool_h, pw, pool_w).transpose(0, 2, 1, 3)
    return idx_arr.reshape(ph, pw, pool_h * pool_w)


def _preprocess_image_pil(pil_img: Image.Image, dtype: torch.dtype, device: torch.device):
    """Single-crop (global-view-only) Molmo2 image preprocessing using PIL.

    Replaces ``Molmo2ImageProcessor`` (which requires torchvision for resize)
    with a pure-PIL implementation for the common square 378×378 case.

    :returns:
        ``(pixel_values_hf, images_ours, pooling_idx, image_grids, image_num_crops)``

        * ``pixel_values_hf``  – shape ``(1, 729, 588)``, spatial-first, for HF.
        * ``images_ours``      – shape ``(1, 1, 729, 588)``, C-first, for ours.
        * ``pooling_idx``      – shape ``(1, 196, 4)``, patch indices.
        * ``image_grids``      – shape ``(1, 4)`` = ``[[14, 14, 0, 0]]``.
        * ``image_num_crops``  – shape ``(1,)`` = ``[1]``.
    """
    img = pil_img.convert("RGB").resize((_IMAGE_SIZE, _IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    # SigLIP2 normalise: (pixel − 0.5) / 0.5
    arr = (arr - 0.5) / 0.5  # (378, 378, 3)

    h_p = _IMAGE_SIZE // _PATCH_SIZE  # 27
    w_p = _IMAGE_SIZE // _PATCH_SIZE  # 27

    # Reshape to (h_p, w_p, kH, kW, C) — layout shared by both orderings.
    arr5 = arr.reshape(h_p, _PATCH_SIZE, w_p, _PATCH_SIZE, 3).transpose(0, 2, 1, 3, 4)

    # HF expects spatial-first: flatten kH, kW, C → (n, kH·kW·C)
    patches_sf = arr5.reshape(h_p * w_p, _PATCH_SIZE * _PATCH_SIZE * 3)

    # Ours expects C-first: (h_p, w_p, C, kH, kW) → flatten → (n, C·kH·kW)
    patches_cf = arr5.transpose(0, 1, 4, 2, 3).reshape(h_p * w_p, 3 * _PATCH_SIZE * _PATCH_SIZE)

    # Pooling indices: 27×27 → 14×14, each group covers 4 original patches.
    idx_arr = np.arange(h_p * w_p, dtype=np.int32).reshape(h_p, w_p)
    pool_idx = _arange_for_pooling(idx_arr, _POOL_H, _POOL_W)  # (14, 14, 4)
    pooling_idx = pool_idx.reshape(-1, _POOL_H * _POOL_W)  # (196, 4)

    def _ft(a: np.ndarray) -> torch.Tensor:
        """Float tensor (patches)."""
        return torch.from_numpy(a).to(dtype=dtype, device=device)

    def _it(a: np.ndarray) -> torch.Tensor:
        """Integer index tensor."""
        return torch.from_numpy(a.astype(np.int64)).to(device=device)

    pixel_values_hf = _ft(patches_sf[np.newaxis])  # (1, 729, 588)
    images_ours = _ft(patches_cf[np.newaxis, np.newaxis])  # (1, 1, 729, 588)
    pooling_idx_t = _it(pooling_idx).unsqueeze(0)  # (1, 196, 4) int64
    image_grids = torch.tensor([[_GRID_H, _GRID_W, 0, 0]], dtype=torch.long, device=device)
    image_num_crops = torch.tensor([1], dtype=torch.long, device=device)

    return pixel_values_hf, images_ours, pooling_idx_t, image_grids, image_num_crops


def _build_image_token_ids() -> list:
    """Return the expanded image token ID sequence for one global-only crop.

    Structure:
    ``<low_res_im_start>``  ``(``_GRID_W_ × ``<im_patch>`` + ``<im_col>``) × _GRID_H_``
    ``<im_end>``  ``<im_start>`` ``<im_end>``

    This produces exactly ``_GRID_H * _GRID_W = 196`` ``<im_patch>`` tokens and
    exactly **two** ``<im_end>`` tokens (required by ``build_batched_images``).
    """
    tokens: list = [_LOW_RES_IM_START_ID]
    for _ in range(_GRID_H):
        tokens += [_IM_PATCH_ID] * _GRID_W
        tokens += [_IM_COL_ID]
    tokens += [_IM_END_ID, _IM_START_ID, _IM_END_ID]
    return tokens


def _build_input_ids(tok, image_tokens: list, device: torch.device) -> torch.Tensor:
    """Build ``input_ids`` for an image + short question.

    Uses the tokenizer's chat template, then replaces the ``<|image|>``
    placeholder token with the full expanded image token sequence.
    """
    text = tok.apply_chat_template(
        [{"role": "user", "content": "<|image|>What colors do you see?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids: list = tok.encode(text, add_special_tokens=False)

    # Splice the image tokens in place of the <|image|> placeholder.
    try:
        img_pos = ids.index(_IMAGE_PLACEHOLDER_ID)
    except ValueError as exc:
        raise RuntimeError(
            f"<|image|> placeholder (ID {_IMAGE_PLACEHOLDER_ID}) not found in tokenised "
            f"prompt. ids={ids[:10]}…"
        ) from exc
    ids = ids[:img_pos] + image_tokens + ids[img_pos + 1 :]
    return torch.tensor([ids], dtype=torch.long, device=device)


def _greedy_decode(model_fn, input_ids: torch.Tensor, n_tokens: int) -> list:
    """Run a manual greedy-decode loop for *n_tokens* steps.

    *model_fn* must accept ``input_ids`` as its first positional argument and
    return a logits tensor of shape ``(B, S, V)``.  All extra keyword
    arguments (e.g., image tensors) are closed over inside *model_fn*.
    """
    gen_ids = input_ids.clone()
    generated: list = []
    with torch.inference_mode():
        for _ in range(n_tokens):
            logits = model_fn(gen_ids)
            next_id = int(logits[0, -1].argmax().item())
            generated.append(next_id)
            gen_ids = torch.cat(
                [gen_ids, torch.tensor([[next_id]], dtype=torch.long, device=gen_ids.device)],
                dim=1,
            )
    return generated


# ─────────────────────────────── test ───────────────────────────────────────


@requires_gpu
def test_molmo2_generation_parity():
    """Convert HF Molmo2-4B weights, then greedy-decode a synthetic image+question
    and verify our model's output tokens match the HF reference exactly.

    The test image is a small synthetic RGB gradient (no internet access needed).
    Both models receive identical preprocessed patches — HF in spatial-first
    order, ours in C-first order — so any token mismatch indicates a real
    converter or forward-pass bug.
    """
    if not _hf_cache_has(_MODEL_ID):
        pytest.skip(f"{_MODEL_ID} not in HF cache")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # ── Load HF model ────────────────────────────────────────────────────────
    ensure_default_rope_registered()
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    try:
        hf = AutoModelForImageTextToText.from_pretrained(
            _MODEL_ID, trust_remote_code=True, local_files_only=True
        )
        tok = AutoTokenizer.from_pretrained(
            _MODEL_ID, trust_remote_code=True, local_files_only=True
        )
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Could not load {_MODEL_ID}: {e}")

    reinit_rope_buffers(hf)

    # ── Build our model from converted weights ────────────────────────────────
    cfg = molmo2_config_from_hf_config(hf.config)
    converted = molmo2_hf_state_dict_to_multimodal_lm(hf.state_dict(), cfg)
    ours = MultimodalLM(cfg, init_device="meta")
    ours.to_empty(device=torch.device("cpu"))
    missing, unexpected = ours.load_state_dict(converted, strict=False)
    retie_word_embeddings(ours)  # `to_empty` breaks the tied-embedding share (Molmo2-4B)
    del converted
    param_keys = {k for k, _ in ours.named_parameters()}
    assert not (
        set(missing) & param_keys
    ), f"Missing params: {sorted(set(missing) & param_keys)[:5]}"
    assert not (
        set(unexpected) & param_keys
    ), f"Unexpected params: {sorted(set(unexpected) & param_keys)[:5]}"

    hf = hf.to(device=device, dtype=dtype).eval()
    ours = ours.to(device=device, dtype=dtype).eval()

    # ── Synthetic test image: 64×64 RGB gradient, PIL-preprocessed ───────────
    np.random.seed(42)
    gradient = np.zeros((64, 64, 3), dtype=np.uint8)
    gradient[:, :, 0] = np.linspace(0, 255, 64, dtype=np.uint8)  # red ramp
    gradient[:, :, 1] = np.linspace(0, 200, 64, dtype=np.uint8)[np.newaxis, :].T  # green ramp
    gradient[:, :, 2] = 100  # flat blue
    pil_img = Image.fromarray(gradient)

    pixel_values_hf, images_ours, pooling_idx, image_grids, image_num_crops = _preprocess_image_pil(
        pil_img, dtype=dtype, device=device
    )

    # ── Build shared input_ids ────────────────────────────────────────────────
    image_tokens = _build_image_token_ids()
    input_ids = _build_input_ids(tok, image_tokens, device=device)

    # Sanity: correct number of <im_patch> and <im_end> tokens
    n_im_patch = (input_ids == _IM_PATCH_ID).sum().item()
    n_im_end = (input_ids == _IM_END_ID).sum().item()
    assert (
        n_im_patch == _GRID_H * _GRID_W
    ), f"Expected {_GRID_H * _GRID_W} <im_patch> tokens, got {n_im_patch}"
    assert n_im_end == 2, f"Expected 2 <im_end> tokens, got {n_im_end}"

    # ── HF greedy decode ─────────────────────────────────────────────────────
    def _hf_fn(ids: torch.Tensor) -> torch.Tensor:
        out = hf(
            input_ids=ids,
            pixel_values=pixel_values_hf,
            image_token_pooling=pooling_idx.squeeze(0),  # HF expects (n_pooled, pool_k)
            image_grids=image_grids,
            image_num_crops=image_num_crops,
            use_cache=False,
        )
        return out.logits

    hf_tokens = _greedy_decode(_hf_fn, input_ids, _N_TOKENS)

    # ── Our greedy decode ─────────────────────────────────────────────────────
    def _our_fn(ids: torch.Tensor) -> torch.Tensor:
        return ours(
            input_ids=ids,
            images=images_ours,
            pooled_patches_idx=pooling_idx,
        )

    our_tokens = _greedy_decode(_our_fn, input_ids, _N_TOKENS)

    # ── Decode for readability ────────────────────────────────────────────────
    hf_text = tok.decode(hf_tokens, skip_special_tokens=True)
    our_text = tok.decode(our_tokens, skip_special_tokens=True)

    # ── Assertions ───────────────────────────────────────────────────────────
    assert hf_tokens == our_tokens, (
        f"Generated token IDs mismatch after {_N_TOKENS} steps:\n"
        f"  HF  tokens = {hf_tokens}\n"
        f"  ours tokens = {our_tokens}\n"
        f"  HF  text = {hf_text!r}\n"
        f"  ours text = {our_text!r}"
    )
    # Check the decoded text is non-trivial (contains at least one alphabetic char).
    assert any(
        c.isalpha() for c in our_text
    ), f"Decoded text looks empty or non-linguistic: {our_text!r}"


@requires_gpu
def test_molmo2_image_bidirectional_attention_parity():
    """Pin HF Molmo2's bidirectional image-token attention.

    With multiple image tokens, HF attends **bidirectionally** among them
    (driven by ``token_type_ids``) while text stays causal. This test feeds the
    same 196-image-token input to HF and to our converted model and checks the
    last-token logits:

    * passing ``token_type_ids`` reproduces HF's logits (bidirectional), and
    * omitting it (pure causal) is measurably *worse* — which is exactly the gap
      this feature closes. The second assertion guards against the test silently
      regressing to a causal-vs-causal comparison (the blind spot of the
      single-image-token full-pipeline parity test).
    """
    if not _hf_cache_has(_MODEL_ID):
        pytest.skip(f"{_MODEL_ID} not in HF cache")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    ensure_default_rope_registered()
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    try:
        hf = AutoModelForImageTextToText.from_pretrained(
            _MODEL_ID, trust_remote_code=True, local_files_only=True
        )
        tok = AutoTokenizer.from_pretrained(
            _MODEL_ID, trust_remote_code=True, local_files_only=True
        )
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Could not load {_MODEL_ID}: {e}")

    reinit_rope_buffers(hf)

    cfg = molmo2_config_from_hf_config(hf.config)
    ours = MultimodalLM(cfg, init_device="meta")
    ours.to_empty(device=torch.device("cpu"))
    ours.load_state_dict(molmo2_hf_state_dict_to_multimodal_lm(hf.state_dict(), cfg), strict=False)
    retie_word_embeddings(ours)  # `to_empty` breaks the tied-embedding share (Molmo2-4B)
    hf = hf.to(device=device, dtype=dtype).eval()
    ours = ours.to(device=device, dtype=dtype).eval()

    # Synthetic image + prompt → 196 <im_patch> tokens.
    np.random.seed(42)
    gradient = np.zeros((64, 64, 3), dtype=np.uint8)
    gradient[:, :, 0] = np.linspace(0, 255, 64, dtype=np.uint8)
    gradient[:, :, 1] = np.linspace(0, 200, 64, dtype=np.uint8)[np.newaxis, :].T
    gradient[:, :, 2] = 100
    pil_img = Image.fromarray(gradient)
    pixel_values_hf, images_ours, pooling_idx, image_grids, image_num_crops = _preprocess_image_pil(
        pil_img, dtype=dtype, device=device
    )
    input_ids = _build_input_ids(tok, _build_image_token_ids(), device=device)

    # token_type_ids: 1 for image structural tokens (matches HF's image_token_ids
    # set), 0 for text. This is what triggers HF's bidirectional image mask.
    image_id_set = torch.tensor(
        [_IM_PATCH_ID, _IM_COL_ID, _IM_START_ID, _LOW_RES_IM_START_ID, _IM_END_ID],
        device=device,
    )
    token_type_ids = torch.isin(input_ids, image_id_set).long()
    assert int(token_type_ids.sum()) > 1, "need >1 image token to exercise bidirectionality"

    # HF reference WITH bidirectional image attention.
    with torch.inference_mode():
        hf_logits = (
            hf(
                input_ids=input_ids,
                pixel_values=pixel_values_hf,
                image_token_pooling=pooling_idx.squeeze(0),
                image_grids=image_grids,
                image_num_crops=image_num_crops,
                token_type_ids=token_type_ids,
                use_cache=False,
            )
            .logits[0, -1]
            .float()
        )
    base_vocab = hf_logits.shape[0]

    with torch.inference_mode():
        our_bidir = ours(
            input_ids=input_ids,
            images=images_ours,
            pooled_patches_idx=pooling_idx,
            token_type_ids=token_type_ids,
            logits_to_keep=1,
        )[0, -1].float()[:base_vocab]
        our_causal = ours(
            input_ids=input_ids,
            images=images_ours,
            pooled_patches_idx=pooling_idx,
            logits_to_keep=1,
        )[0, -1].float()[:base_vocab]

    bidir_diff = (hf_logits - our_bidir).abs().max().item()
    causal_diff = (hf_logits - our_causal).abs().max().item()

    # (1) Bidirectional reproduces HF.
    assert bidir_diff < 5.0, (
        f"bidirectional logits diverge from HF: max abs diff = {bidir_diff:.3e} "
        f"(threshold 5.0 in bf16)"
    )
    # (2) The gap is real: pure-causal is measurably worse than bidirectional.
    #     If this fails, the test has regressed to causal-vs-causal and no longer
    #     exercises image-token bidirectional attention.
    assert causal_diff > bidir_diff, (
        f"causal logits match HF as well as bidirectional (causal={causal_diff:.3e}, "
        f"bidir={bidir_diff:.3e}); test no longer pins the bidirectional gap"
    )
