"""Shared Molmo2 token constants and image-token-sequence construction.

These are the *pure* pieces of the Molmo2 token layout (no tokenizer, model, or
PIL dependency) so they can be reused by both the inference/eval example scripts
and the training data pipeline without either importing from the other.

The constants match the released ``allenai/Molmo2-*`` tokenizers (Qwen2.5 base
vocab plus the Molmo2 image-special tokens) and the SigLIP2-SO400M/14 multi-crop
preprocessor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from olmo_core.config import Config

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

IM_START_TOKEN = "<im_start>"
IM_END_TOKEN = "<im_end>"
IM_PATCH_TOKEN = "<im_patch>"
IM_COL_TOKEN = "<im_col>"
LOW_RES_IM_START_TOKEN = "<low_res_im_start>"
IMAGE_PLACEHOLDER_TOKEN = "<|image|>"
IM_END_TURN_TOKEN = "<|im_end|>"

# Preserve the released Molmo2 ordering when these tokens need to be appended to another
# tokenizer. The s002 tokenizer has 74 padded model-vocabulary rows, so these six additions
# do not require resizing its embedding or LM-head tensors.
IMAGE_SPECIAL_TOKENS = (
    IM_START_TOKEN,
    IM_END_TOKEN,
    IM_PATCH_TOKEN,
    IM_COL_TOKEN,
    LOW_RES_IM_START_TOKEN,
    IMAGE_PLACEHOLDER_TOKEN,
)


@dataclass
class Molmo2TokenIds(Config):
    """Token IDs used by Molmo2 image sequences and chat formatting."""

    im_start_id: int = IM_START_ID
    im_end_id: int = IM_END_ID
    im_patch_id: int = IM_PATCH_ID
    im_col_id: int = IM_COL_ID
    low_res_im_start_id: int = LOW_RES_IM_START_ID
    image_placeholder_id: int = IMAGE_PLACEHOLDER_ID
    im_end_turn_id: int = IM_END_TURN_ID

    @property
    def image_token_ids(self) -> frozenset[int]:
        """IDs that receive bidirectional image-token attention."""
        return frozenset(
            {
                self.im_start_id,
                self.im_end_id,
                self.im_patch_id,
                self.im_col_id,
                self.low_res_im_start_id,
            }
        )


DEFAULT_MOLMO2_TOKEN_IDS = Molmo2TokenIds()


def prepare_molmo2_tokenizer(
    tokenizer: Any,
    *,
    model_vocab_size: Optional[int] = None,
) -> Molmo2TokenIds:
    """Add Molmo2 image tokens to ``tokenizer`` and return their resolved IDs.

    Existing tokens are retained at their current IDs. New image tokens are appended in
    :data:`IMAGE_SPECIAL_TOKENS` order. ``<|im_end|>`` must already be supplied by the
    tokenizer's chat vocabulary; it is not added because changing the chat template is a
    separate model-format decision.

    :param tokenizer: A Hugging Face-compatible tokenizer.
    :param model_vocab_size: Optional fixed embedding-table size. When provided, reject an
        adapted tokenizer that would require resizing model weights.
    """
    tokenizer.add_tokens(list(IMAGE_SPECIAL_TOKENS), special_tokens=True)

    vocab = tokenizer.get_vocab()

    def require_id(token: str) -> int:
        if token not in vocab:
            raise ValueError(f"Tokenizer does not contain required token {token!r}")
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        if token_id < 0:
            raise ValueError(f"Tokenizer returned invalid ID {token_id} for {token!r}")
        return token_id

    token_ids = Molmo2TokenIds(
        im_start_id=require_id(IM_START_TOKEN),
        im_end_id=require_id(IM_END_TOKEN),
        im_patch_id=require_id(IM_PATCH_TOKEN),
        im_col_id=require_id(IM_COL_TOKEN),
        low_res_im_start_id=require_id(LOW_RES_IM_START_TOKEN),
        image_placeholder_id=require_id(IMAGE_PLACEHOLDER_TOKEN),
        im_end_turn_id=require_id(IM_END_TURN_TOKEN),
    )
    resolved_ids = [
        token_ids.im_start_id,
        token_ids.im_end_id,
        token_ids.im_patch_id,
        token_ids.im_col_id,
        token_ids.low_res_im_start_id,
        token_ids.image_placeholder_id,
        token_ids.im_end_turn_id,
    ]
    if len(set(resolved_ids)) != len(resolved_ids):
        raise ValueError(f"Molmo2 special tokens did not resolve to unique IDs: {resolved_ids}")
    if model_vocab_size is not None and (max_token_id := max(vocab.values())) >= model_vocab_size:
        raise ValueError(
            f"Adapted tokenizer requires token ID {max_token_id:,d}, but the model "
            f"vocabulary has only {model_vocab_size:,d} rows"
        )
    return token_ids


def build_image_token_ids(
    resized_h: int,
    resized_w: int,
    h: int,
    w: int,
    low_res_col_tokens: bool = False,
    token_ids: Optional[Molmo2TokenIds] = None,
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
    ids = token_ids or DEFAULT_MOLMO2_TOKEN_IDS
    tokens: list[int] = [ids.low_res_im_start_id]
    if low_res_col_tokens:
        for _ in range(resized_h):
            tokens += [ids.im_patch_id] * resized_w + [ids.im_col_id]
    else:
        tokens += [ids.im_patch_id] * (resized_h * resized_w)
    tokens += [ids.im_end_id, ids.im_start_id]
    for _ in range(h):
        tokens += [ids.im_patch_id] * w + [ids.im_col_id]
    tokens += [ids.im_end_id]
    return tokens
