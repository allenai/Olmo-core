"""Tests for SFT formatting and image-only-v9 registry."""

import numpy as np
import pytest

from olmo_core.data.multimodal import build_branched_sequence
from olmo_core.data.multimodal.message_weight import (
    MessageWeight,
    apply_message_weight_to_loss_masks,
    loss_token_weighting_for_build,
)
from olmo_core.data.multimodal.sft_formatter import SftFormatter


def test_message_weight_scalar():
    masks = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    out = apply_message_weight_to_loss_masks(masks, None, MessageWeight(weight=0.2))
    np.testing.assert_allclose(out, [0.0, 0.2, 0.2, 0.0])


def test_loss_token_weighting_for_build():
    assert loss_token_weighting_for_build(MessageWeight()) == "none"
    assert (
        loss_token_weighting_for_build(MessageWeight(root_subsegments=True)) == "root_subsegments"
    )
    assert (
        loss_token_weighting_for_build(MessageWeight(root_subsegments=True, root_length=True))
        == "root_subsegments_root_tokens"
    )


def test_build_branched_sequence_root_length_single_branch():
    out = build_branched_sequence(
        [100, 151938],
        [([10, 11], [20, 21])],
        eos_id=1,
        loss_token_weighting="root_subsegments_root_tokens",
    )
    expected = 2.0 / np.sqrt(len([20, 21]) + 1)
    nz = out["loss_masks"] > 0
    np.testing.assert_allclose(out["loss_masks"][nz], expected, rtol=1e-5)


def test_sft_formatter_vqa_short_answer():
    fmt = SftFormatter(seed=0)
    turns = fmt.format_turns(
        {"style": "text_vqa", "question": "What color?", "answers": ["red", "red", "blue"]},
        index=0,
    )
    assert turns[0][0] == "text_vqa: What color?"
    assert turns[0][1] == "red"


def test_sft_formatter_demo_no_style_prefix():
    fmt = SftFormatter(seed=0)
    turns = fmt.format_turns(
        {"style": "user_qa", "question": "Hi?", "answer": "Hello"},
        index=0,
    )
    assert turns[0][0] == "Hi?"


def test_format_vqa_short_passes_answer():
    from olmo_core.data.multimodal.academic.formatters import format_vqa_short

    fmt = SftFormatter(seed=0)
    formatted = format_vqa_short(
        {"question": "Q?", "answer": "42", "image": "img.jpg"},
        style="tabwmp_da",
    )
    turns = fmt.format_turns(formatted, index=0)
    assert turns[0][1] == "42"


def test_format_turns_vqa2_multi_message_list():
    fmt = SftFormatter(seed=0)
    turns = fmt.format_turns(
        {
            "image": "img.jpg",
            "message_list": [
                {"question": "Q1?", "answers": ["a", "a"], "style": "vqa2"},
                {"question": "Q2?", "answers": ["b"], "style": "vqa2"},
            ],
        },
        index=0,
    )
    assert len(turns) == 2
    assert turns[0][1] == "a"
    assert turns[1][1] == "b"


def test_format_cosyn_exp_chain_of_thought_and_explanation():
    fmt = SftFormatter(seed=0)
    turns = fmt.format_turns(
        {
            "image": "img.jpg",
            "message_list": [
                {
                    "question": "What is the value?",
                    "explanation": "The bar shows 42.",
                    "answer": "42",
                    "style": "cosyn_chart_exp",
                }
            ],
        },
        index=0,
    )
    assert len(turns) == 1
    assert turns[0][0].startswith("cosyn_chart_exp:")
    assert "Provide reasoning steps" in turns[0][0]
    assert turns[0][1] == "The bar shows 42. Answer: 42"


def test_pixmo_clocks_style_prefix():
    fmt = SftFormatter(seed=0)
    turns = fmt.format_turns(
        {
            "style": "clocks",
            "prompt": "What time is being shown?",
            "text": "The time shown is 3:00",
        },
        index=0,
    )
    assert turns[0][0].startswith("clocks:")


def test_format_cosyn_document_style_prefix():
    fmt = SftFormatter(seed=0)
    turns = fmt.format_turns(
        {
            "image": "img.jpg",
            "message_list": [
                {"question": "Who wrote it?", "answer": "Alice", "style": "cosyn_document"},
                {"question": "When?", "answer": "2020", "style": "cosyn_document"},
            ],
        },
        index=0,
    )
    assert len(turns) == 2
    assert turns[0][0].startswith("cosyn_document:")
    assert turns[1][0].startswith("cosyn_document:")


def test_format_turns_message_list_applies_style_prefix():
    fmt = SftFormatter(seed=0)
    turns = fmt.format_turns(
        {
            "image": "img.jpg",
            "message_list": [
                {"question": "Q1?", "answer": "A1", "style": "plot_qa"},
                {"question": "Q2?", "answer": "A2", "style": "plot_qa"},
            ],
        },
        index=0,
    )
    assert len(turns) == 2
    assert turns[0][0].startswith("plot_qa:")
    assert turns[1][0].startswith("plot_qa:")


def test_encode_sft_example_branch_shuffle_is_deterministic():
    pytest.importorskip("transformers")
    from PIL import Image
    from transformers import AutoTokenizer

    from olmo_core.data.multimodal.message_sequence import encode_sft_example

    tokenizer = AutoTokenizer.from_pretrained("allenai/Molmo2-4B", trust_remote_code=True)
    image = Image.new("RGB", (64, 64), color=(128, 64, 32))
    turns = [("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3")]
    a = encode_sft_example(tokenizer, image, turns, seed=42)
    b = encode_sft_example(tokenizer, image, turns, seed=42)
    c = encode_sft_example(tokenizer, image, turns, seed=43)
    np.testing.assert_array_equal(a["input_ids"], b["input_ids"])
    assert not np.array_equal(a["input_ids"], c["input_ids"])


def test_chart_qa_weighted_loss_mask():
    from olmo_core.data.multimodal.academic.registry import _format_chart_qa_weighted
    from olmo_core.data.multimodal.message_sequence import encode_sft_example

    pytest.importorskip("transformers")
    from PIL import Image
    from transformers import AutoTokenizer

    ex = {
        "image": Image.new("RGB", (64, 64)),
        "question": "What is the value?",
        "answers": ["42"],
        "metadata": {"is_human": True, "example_id": "test"},
    }
    formatted = _format_chart_qa_weighted(ex, np.random.RandomState(0), "train")
    assert abs(formatted["weight"] - (2 * 20901 / 28299)) < 1e-6

    tokenizer = AutoTokenizer.from_pretrained("allenai/Molmo2-4B", trust_remote_code=True)
    turns = SftFormatter(seed=0).format_turns(formatted, index=0)
    base = encode_sft_example(tokenizer, formatted["image"], turns, seed=0)
    weighted = encode_sft_example(
        tokenizer,
        formatted["image"],
        turns,
        message_weight=formatted["weight"],
        seed=0,
    )
    nz = base["loss_masks"] > 0
    assert nz.any()
    ratio = weighted["loss_masks"][nz][0] / base["loss_masks"][nz][0]
    np.testing.assert_allclose(ratio, formatted["weight"], rtol=1e-4)
