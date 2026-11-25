"""Unit tests for qwen3 multimodal token layout helpers."""

from __future__ import annotations

import numpy as np
import pytest

from olmo_core.data.multimodal.qwen3_layout import (
    branch_context_ids,
    image_prefix_ids,
    user_header_ids,
    user_turn_ids,
    user_turn_suffix_ids,
)
from olmo_core.nn.vision.molmo2_tokens import (
    IM_END_ID,
    LOW_RES_IM_START_ID,
    Molmo2TokenIds,
)


@pytest.fixture(scope="module")
def tokenizer():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("allenai/Molmo2-4B", trust_remote_code=True)


def test_user_header_starts_turn(tokenizer):
    header = user_header_ids(tokenizer)
    full = user_turn_ids(tokenizer, "hello")
    assert full[: len(header)] == header
    assert tokenizer.decode(header).startswith("<|im_start|>user")


def test_suffix_recovers_full_user_turn(tokenizer):
    question = "text_vqa: what is this?"
    assert user_header_ids(tokenizer) + user_turn_suffix_ids(tokenizer, question) == user_turn_ids(
        tokenizer, question
    )


def test_image_prefix_has_no_bos(tokenizer):
    grid = np.array([1, 1, 1, 1], dtype=np.int64)
    prefix = image_prefix_ids(tokenizer, grid)
    bos = tokenizer.bos_token_id or tokenizer.eos_token_id
    assert prefix[0] != bos
    assert prefix[: len(user_header_ids(tokenizer))] == user_header_ids(tokenizer)
    assert LOW_RES_IM_START_ID in prefix
    assert prefix[-1] == IM_END_ID


def test_image_prefix_honors_s002_token_ids(tokenizer):
    token_ids = Molmo2TokenIds(
        im_start_id=100278,
        im_end_id=100279,
        im_patch_id=100280,
        im_col_id=100281,
        low_res_im_start_id=100282,
        image_placeholder_id=100283,
        im_end_turn_id=100265,
    )
    prefix = image_prefix_ids(
        tokenizer,
        np.array([1, 1, 1, 1], dtype=np.int64),
        token_ids=token_ids,
    )
    image_tokens = prefix[len(user_header_ids(tokenizer)) :]
    assert image_tokens == [100282, 100280, 100279, 100278, 100280, 100281, 100279]
    assert max(image_tokens) < 100352


def test_single_branch_uses_suffix(tokenizer):
    q = "Where is the cat?"
    assert branch_context_ids(
        tokenizer, q, branch_index=0, multi_branch=False
    ) == user_turn_suffix_ids(tokenizer, q)


def test_multi_branch_uses_suffix_for_all_branches(tokenizer):
    q = "Where is the cat?"
    for branch_index in (0, 1):
        assert branch_context_ids(tokenizer, q, branch_index=branch_index, multi_branch=True) == (
            user_turn_suffix_ids(tokenizer, q)
        )
