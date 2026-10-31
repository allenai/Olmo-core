"""olmo35 layout against open-instruct's C11 tokenization (fixture from
``src/scripts/make_olmo35_c11_fixture.py``)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from olmo_core.data.multimodal.olmo35_layout import (
    OLMO35_TOKEN_IDS,
    OLMO35_TOKENIZER,
    OLMO35_TOKENIZER_REVISION,
    encode_conversation,
    image_block_ids,
    load_olmo35_tokenizer,
    tokenize_conversation,
)

FIXTURE = Path(__file__).parents[2] / "fixtures" / "olmo35_c11" / "rows.json.gz"
EOS = 100257
IM_END = 100265
NEWLINE = 198


@pytest.fixture(scope="module")
def tokenizer():
    pytest.importorskip("transformers")

    try:
        return load_olmo35_tokenizer()
    except OSError as exc:
        pytest.skip(f"{OLMO35_TOKENIZER} unavailable: {exc}")


@pytest.fixture(scope="module")
def rows():
    with gzip.open(FIXTURE, "rt") as f:
        fixture = json.load(f)
    assert (fixture["tokenizer"], fixture["revision"]) == (
        OLMO35_TOKENIZER,
        OLMO35_TOKENIZER_REVISION,
    )
    return fixture["rows"]


def _runs(mask: np.ndarray):
    """(start, end) of each run of True."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def test_fixture_covers_multi_turn_and_tools(rows):
    assert {row["source"] for row in rows} == {"dolci", "nemo", "sci"}
    assert any(sum(m["role"] == "assistant" for m in row["messages"]) > 1 for row in rows)
    assert any(any(m.get("tool_calls") for m in row["messages"]) for row in rows)


def test_text_rows_match_open_instruct(tokenizer, rows):
    for row in rows:
        input_ids, label_mask = tokenize_conversation(
            tokenizer, row["messages"], tools=row["tools"]
        )
        assert input_ids.tolist() == row["input_ids"], row["id"]
        assert label_mask.astype(int).tolist() == row["label_mask"], row["id"]


def test_text_example_targets_are_the_next_label(tokenizer, rows):
    for row in rows:
        example = encode_conversation(tokenizer, row["messages"], tools=row["tools"])
        mask = np.asarray(row["label_mask"], dtype=bool)
        ids = np.asarray(row["input_ids"])
        assert example["input_ids"].tolist() == row["input_ids"]
        np.testing.assert_array_equal(example["loss_masks"][:-1], mask[1:])
        assert example["loss_masks"][-1] == 0
        np.testing.assert_array_equal(example["labels"][:-1], np.where(mask[1:], ids[1:], -100))
        assert not example["token_type_ids"].any()


def _multi_turn(rows):
    return next(
        row
        for row in rows
        if sum(m["role"] == "assistant" for m in row["messages"]) > 1 and not row["tools"]
    )


def _image_ids(tokenizer, n_images: int):
    grid = np.array([2, 2, 2, 2], dtype=np.int64)
    return image_block_ids(tokenizer, [grid] * n_images, token_ids=OLMO35_TOKEN_IDS)


@pytest.mark.parametrize("n_images", [1, 2])
def test_image_row_keeps_text_tokens_and_labels(tokenizer, rows, n_images):
    row = _multi_turn(rows)
    image = _image_ids(tokenizer, n_images)
    example = encode_conversation(tokenizer, row["messages"], image_ids=image)
    ids = example["input_ids"]
    # The image block sits right after the first "<|im_start|>user\n".
    header = tokenizer("<|im_start|>user\n", add_special_tokens=False)["input_ids"]
    text = np.asarray(row["input_ids"])
    at = next(i for i in range(len(text)) if text[i : i + len(header)].tolist() == header) + len(
        header
    )
    assert ids[:at].tolist() == text[:at].tolist()
    assert ids[at : at + len(image)].tolist() == image
    assert ids[at + len(image) :].tolist() == text[at:].tolist()
    # No image token is a label or a target; the text keeps its labels.
    is_image = np.isin(ids, list(OLMO35_TOKEN_IDS.image_token_ids))
    assert example["token_type_ids"].astype(bool).tolist() == is_image.tolist()
    assert not example["loss_masks"][at - 1 : at + len(image)].any()
    mask = np.asarray(row["label_mask"], dtype=bool)
    expected_targets = np.concatenate([mask[1:at], np.zeros(len(image), bool), mask[at:]])
    np.testing.assert_array_equal(example["loss_masks"][:-1].astype(bool), expected_targets)


def test_image_row_labels_cover_exactly_the_assistant_turns(tokenizer, rows):
    row = _multi_turn(rows)
    example = encode_conversation(tokenizer, row["messages"], image_ids=_image_ids(tokenizer, 1))
    ids = example["input_ids"]
    is_label = np.zeros(len(ids), dtype=bool)
    is_label[1:] = example["loss_masks"][:-1] > 0
    assistants = [m for m in row["messages"] if m["role"] == "assistant"]
    runs = _runs(is_label)
    assert len(runs) == len(assistants)
    for (start, end), message in zip(runs, assistants):
        # Each run ends with the turn's EOS and holds the turn's text; the user, system and
        # header tokens around it, and the image, are outside every run.
        assert ids[end - 1] == EOS
        decoded = tokenizer.decode(ids[start:end].tolist())
        # Inline reasoning opens with "<think>", which the generation prompt already holds.
        content = message["content"].strip().removeprefix("<think>")
        assert content[:200] in decoded
        before = tokenizer.decode(ids[max(0, start - 8) : start].tolist())
        assert before.endswith(
            ("<|im_start|>assistant\n<th", "<|im_start|>assistant\n<think")
        ), before


def test_synthetic_turns_and_empty_think_block(tokenizer):
    messages = [
        {"role": "user", "content": "What is in the image?"},
        {"role": "assistant", "content": "A red bicycle."},
        {"role": "user", "content": "What color is it?"},
        {"role": "assistant", "content": "Red."},
    ]
    example = encode_conversation(tokenizer, messages, image_ids=_image_ids(tokenizer, 1))
    ids = example["input_ids"]
    is_label = np.zeros(len(ids), dtype=bool)
    is_label[1:] = example["loss_masks"][:-1] > 0
    labelled = [tokenizer.decode(ids[s:e].tolist()) for s, e in _runs(is_label)]
    # The template writes an empty think block before an answer without reasoning; its
    # "></" piece straddles the generation prompt and is a label, as in open-instruct.
    assert labelled == ["></think>A red bicycle.<|endoftext|>", "></think>Red.<|endoftext|>"]
    assert "What color is it?" not in "".join(labelled)


def test_tool_call_turn_ends_with_im_end(tokenizer, rows):
    row = next(row for row in rows if any(m.get("tool_calls") for m in row["messages"]))
    input_ids, label_mask = tokenize_conversation(tokenizer, row["messages"], tools=row["tools"])
    ends = [input_ids[end - 2 : end].tolist() for _, end in _runs(label_mask)]
    # A tool-call turn's span is the template's render through that turn, which ends in
    # "<|im_end|>\n": open-instruct trains the newline too. Other turns end with EOS.
    assert [IM_END, NEWLINE] in ends and ends[-1][-1] == EOS


def test_image_without_user_turn_is_refused(tokenizer):
    with pytest.raises(ValueError, match="user turn"):
        encode_conversation(
            tokenizer,
            [{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}],
            image_ids=_image_ids(tokenizer, 1),
        )


def test_tokenizer_keeps_the_serialized_pre_tokenizer(tokenizer):
    # "!\n\n" is one piece under tokenizer.json's Split regex; transformers 5's rebuilt
    # GPT2Tokenizer backend splits it in two.
    pre_tokenizer = json.loads(tokenizer.backend_tokenizer.to_str())["pre_tokenizer"]
    assert pre_tokenizer["type"] == "Sequence"
    assert len(tokenizer("Hello!\n\nWorld", add_special_tokens=False)["input_ids"]) == 3
