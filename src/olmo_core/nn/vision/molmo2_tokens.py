"""Shared Molmo2 token constants and image-token-sequence construction.

These are the *pure* pieces of the Molmo2 token layout (no tokenizer, model, or
PIL dependency) so they can be reused by both the inference/eval example scripts
and the training data pipeline without either importing from the other.

The constants match the released ``allenai/Molmo2-*`` tokenizers (Qwen2.5 base
vocab plus the Molmo2 image-special tokens) and the SigLIP2-SO400M/14 multi-crop
preprocessor.
"""

from __future__ import annotations

# ── Preprocessor constants (SigLIP2-SO400M/14, 378×378, multi-crop) ──────────
PATCH_SIZE = 14
IMAGE_SIZE = 378  # 27 × 27 patches per crop
N_PATCHES = 27  # IMAGE_SIZE // PATCH_SIZE
N_PATCHES_SQ = 729  # N_PATCHES ** 2
PATCH_DIM = 588  # 3 * PATCH_SIZE ** 2
POOL_H = 2
POOL_W = 2
DEFAULT_MAX_CROPS = 8
OVERLAP_MARGINS = (4, 4)

# ── Molmo2 special token IDs (identical across 4B / 8B variants) ─────────────
IM_PATCH_ID = 151938  # <im_patch>
IM_COL_ID = 151939  # <im_col>
IM_START_ID = 151936  # <im_start>
LOW_RES_IM_START_ID = 151940  # <low_res_im_start>
IM_END_ID = 151937  # <im_end>
IMAGE_PLACEHOLDER_ID = 151941  # <|image|>

# Image structural tokens that attend bidirectionally in HF Molmo2 (token_type_ids==1).
# Matches the processor's IMAGE_TOKENS set for the image (non-video) path.
IMAGE_TOKEN_IDS = frozenset({IM_PATCH_ID, IM_COL_ID, IM_START_ID, LOW_RES_IM_START_ID, IM_END_ID})

DEFAULT_MODEL_ID = "allenai/Molmo2-4B"
EOS_TOKEN_ID = (
    151643  # <|endoftext|> — the PAD token; Molmo2's real eos_token is <|im_end|> (151645)
)
IM_END_TURN_ID = 151645  # Qwen2.5 <|im_end|> (chat end-of-turn)


def build_image_token_ids(
    resized_h: int,
    resized_w: int,
    h: int,
    w: int,
    low_res_col_tokens: bool = False,
) -> list[int]:
    """Return the expanded image token-ID sequence for a multi-crop image.

    Token structure (matching ``Molmo2Processor.get_image_tokens`` — the
    low-res/global section has **no** ``<im_col>`` separators)::

        <low_res_im_start>
        resized_h × resized_w × <im_patch>
        <im_end> <im_start>
        h × (w × <im_patch> + <im_col>)
        <im_end>

    ``low_res_col_tokens=True`` reproduces the legacy ``pixmo_cap_eval``
    layout that also put ``<im_col>`` separators in the low-res section.

    :param resized_h: Pooled height (in tokens) of the low-res/global crop.
    :param resized_w: Pooled width (in tokens) of the low-res/global crop.
    :param h: Number of high-res crop rows (in pooled tokens).
    :param w: Number of high-res crop columns (in pooled tokens).
    :param low_res_col_tokens: Emit ``<im_col>`` separators in the low-res
        section too (legacy layout).

    :returns: The flat list of image token IDs.
    """
    tokens: list[int] = [LOW_RES_IM_START_ID]
    if low_res_col_tokens:
        for _ in range(resized_h):
            tokens += [IM_PATCH_ID] * resized_w + [IM_COL_ID]
    else:
        tokens += [IM_PATCH_ID] * (resized_h * resized_w)
    tokens += [IM_END_ID, IM_START_ID]
    for _ in range(h):
        tokens += [IM_PATCH_ID] * w + [IM_COL_ID]
    tokens += [IM_END_ID]
    return tokens
