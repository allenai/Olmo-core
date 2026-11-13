"""qwen3 message layout helpers for Molmo2 multimodal training.

mm_olmo trains with ``message_format="qwen3"``: the first user message begins with
``<|im_start|>user\\n`` before the image block (no leading BOS). Multi-branch
examples share ``user_header + image`` and repeat a full user turn per branch;
single-branch examples keep the image and prompt in one user message (suffix only).
"""

from __future__ import annotations

from typing import List

import numpy as np

from olmo_core.nn.vision.molmo2_tokens import build_image_token_ids

__all__ = [
    "branch_context_ids",
    "followup_turn_context_ids",
    "image_prefix_ids",
    "multi_image_prefix_ids",
    "user_header_ids",
    "user_turn_continuation_ids",
    "user_turn_ids",
    "user_turn_suffix_ids",
]


def user_turn_ids(tokenizer, question: str) -> List[int]:
    """Full non-loss user turn + assistant header (``<|im_start|>user … assistant\\n``)."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def user_header_ids(tokenizer) -> List[int]:
    """Token ids for ``<|im_start|>user\\n`` at the start of the first user message."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=False,
        add_generation_prompt=True,
    )
    marker = "<|im_end|>"
    if marker not in text:
        raise ValueError(f"expected {marker!r} in empty user chat template, got {text!r}")
    return tokenizer.encode(text[: text.index(marker)], add_special_tokens=False)


def user_turn_suffix_ids(tokenizer, question: str) -> List[int]:
    """Question + end-of-user + assistant header, for use after an image in the same turn."""
    full = user_turn_ids(tokenizer, question)
    header = user_header_ids(tokenizer)
    if full[: len(header)] != header:
        raise ValueError("user turn does not start with the expected qwen3 user header")
    return full[len(header) :]


def image_prefix_ids(tokenizer, image_grid: np.ndarray) -> List[int]:
    """Shared qwen3 prefix: ``<|im_start|>user\\n`` + expanded image token block."""
    resized_h, resized_w, h, w = (int(image_grid[i]) for i in range(4))
    return user_header_ids(tokenizer) + build_image_token_ids(resized_h, resized_w, h, w)


def multi_image_prefix_ids(tokenizer, image_grids: List[np.ndarray]) -> List[int]:
    """Shared qwen3 prefix for several images.

    mm_olmo's ``MultiImagePreprocessor`` prepends the text ``"Image {i+1}"`` to each
    image's token block when the example holds more than one image (nothing extra for
    a single image), and the blocks follow each other directly inside the first user
    message: ``<|im_start|>user\\n Image 1 <blocks> Image 2 <blocks> ... {question}``.
    """
    ids = user_header_ids(tokenizer)
    multi = len(image_grids) > 1
    for i, grid in enumerate(image_grids):
        if multi:
            ids = ids + tokenizer.encode(f"Image {i + 1}", add_special_tokens=False)
        resized_h, resized_w, h, w = (int(grid[j]) for j in range(4))
        ids = ids + build_image_token_ids(resized_h, resized_w, h, w)
    return ids


def user_turn_continuation_ids(tokenizer) -> List[int]:
    """Token ids for ``<|im_end|>\\n<|im_start|>user\\n`` between branched user turns."""
    text = "<|im_end|>\n<|im_start|>user\n"
    return tokenizer.encode(text, add_special_tokens=False)


def followup_turn_context_ids(tokenizer, question: str) -> List[int]:
    """Context of turn 2+ in a multi-turn conversation.

    mm_olmo's qwen3 ``apply_chat_template`` prefixes a user message that follows an
    assistant message with ``<|im_end|>\\n`` (closing the previous assistant turn,
    ``preprocessor_utils.py:267``), then the regular user turn follows:
    ``<|im_end|>\\n<|im_start|>user\\n{q}<|im_end|>\\n<|im_start|>assistant\\n``.
    """
    return tokenizer.encode("<|im_end|>\n", add_special_tokens=False) + user_turn_ids(
        tokenizer, question
    )


def branch_context_ids(
    tokenizer, prompt: str, *, branch_index: int = 0, multi_branch: bool
) -> List[int]:
    """Branch context tokens for qwen3 layout (suffix after shared image prefix)."""
    del branch_index, multi_branch
    return user_turn_suffix_ids(tokenizer, prompt)
