"""Pretraining-native text layout for s002 multimodal Stage 1 examples.

s002 was pretrained on Dolma2 documents, not role-tagged conversations. Its tokenizer has no
separate BOS token, so the EOS token marks the document boundary at the beginning of each example.
Molmo's ``message_format=none`` convention then leaves the first message unchanged and prefixes
every later message with one space.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds, build_image_token_ids

__all__ = [
    "branch_context_ids",
    "document_prompt_ids",
    "image_prefix_ids",
    "message_ids",
    "response_ids",
]


def message_ids(tokenizer, text: str, *, first: bool) -> List[int]:
    """Tokenize one plain document message using Molmo's non-role separator rule.

    :param tokenizer: Language-model tokenizer.
    :param text: Message text.
    :param first: Whether this is the first text message in the document or branch.

    :returns: Token IDs without tokenizer-added special tokens.
    """
    return tokenizer.encode(text if first else " " + text, add_special_tokens=False)


def response_ids(tokenizer, text: str) -> List[int]:
    """Tokenize a response as the second plain message, including its leading space."""
    return message_ids(tokenizer, text, first=False)


def branch_context_ids(tokenizer, prompt: str) -> List[int]:
    """Tokenize an unlabelled task prompt as the first message of a branch."""
    return message_ids(tokenizer, prompt, first=True)


def document_prompt_ids(
    tokenizer,
    prompt: str,
    *,
    image_ids: Sequence[int] = (),
) -> List[int]:
    """Build an s002 inference prompt from a document boundary, images, and plain text.

    :param tokenizer: s002/Dolma2 tokenizer.
    :param prompt: Unlabelled task prompt.
    :param image_ids: Optional expanded Molmo image-token blocks.

    :returns: ``[EOS] + image_ids + prompt_ids``.
    """
    if tokenizer.eos_token_id is None:
        raise ValueError("The s002 document layout requires an EOS token")
    return [int(tokenizer.eos_token_id), *image_ids, *branch_context_ids(tokenizer, prompt)]


def image_prefix_ids(
    tokenizer,
    image_grid: np.ndarray,
    *,
    token_ids: Optional[Molmo2TokenIds] = None,
) -> List[int]:
    """Build ``[EOS document boundary] + [expanded Molmo image block]``.

    :param tokenizer: s002/Dolma2 tokenizer.
    :param image_grid: ``(resized_h, resized_w, crop_h, crop_w)`` image grid.
    :param token_ids: Optional model-specific image token IDs.

    :returns: Shared non-loss prefix token IDs.
    """
    if tokenizer.eos_token_id is None:
        raise ValueError("The s002 document layout requires an EOS token")
    resized_h, resized_w, height, width = (int(image_grid[i]) for i in range(4))
    return [int(tokenizer.eos_token_id)] + build_image_token_ids(
        resized_h,
        resized_w,
        height,
        width,
        token_ids=token_ids,
    )
