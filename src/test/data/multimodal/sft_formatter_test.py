"""Tests for SFT formatting and image-only-v9 registry."""

import numpy as np
import pytest

from olmo_core.data.multimodal import build_branched_sequence
from olmo_core.data.multimodal.message_weight import (
    MessageWeight,
    apply_message_weight_to_loss_masks,
    loss_token_weighting_for_build,
)
from olmo_core.data.multimodal.mixture_weights import compute_flat_mixture_weights
from olmo_core.data.multimodal.mixtures.image_only_v9 import IMAGE_ONLY_V9_SUBMIXTURES
from olmo_core.data.multimodal.sequence_builder import example_rng
from olmo_core.data.multimodal.sft_common import (
    SFT_MESSAGE_FORMATS,
    get_example_with_skip,
    sft_example_rng,
    truncate_example,
    validate_sft_message_format,
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


def test_mixture_weights_sum_to_one():
    lengths = {src.name: 1000 for g in IMAGE_ONLY_V9_SUBMIXTURES for src in g.datasets}
    flat = compute_flat_mixture_weights(IMAGE_ONLY_V9_SUBMIXTURES, lengths)
    assert len(flat) == 43
    assert abs(sum(w for _, w in flat) - 1.0) < 1e-6


def test_debug_mixture_subset_weights_renormalize():
    from olmo_core.data.multimodal.mixtures.image_only_v9 import DEBUG_MIXTURE_DATASETS

    lengths = {src.name: 1000 for g in IMAGE_ONLY_V9_SUBMIXTURES for src in g.datasets}
    flat = compute_flat_mixture_weights(IMAGE_ONLY_V9_SUBMIXTURES, lengths)
    allowed = set(DEBUG_MIXTURE_DATASETS)
    subset = [(name, w) for name, w in flat if name in allowed]
    assert len(subset) == len(DEBUG_MIXTURE_DATASETS)
    norm = sum(w for _, w in subset)
    subset = [(name, w / norm) for name, w in subset]
    assert abs(sum(w for _, w in subset) - 1.0) < 1e-6


def test_image_only_v9_has_43_datasets():
    # 4 demo + 33 academic (incl. 3 mantis + 6 multidoc) + 5 pointing + tulu4
    names = [src.name for g in IMAGE_ONLY_V9_SUBMIXTURES for src in g.datasets]
    assert len(names) == 43
    assert len(set(names)) == 43


def test_message_formats_include_qwen_and_native_document_layouts():
    assert SFT_MESSAGE_FORMATS == ("qwen3", "document")
    assert validate_sft_message_format("qwen3") == "qwen3"
    assert validate_sft_message_format("document") == "document"
    with pytest.raises(ValueError, match="Unknown message_format"):
        validate_sft_message_format("qwen2")


def test_qwen_rng_preserves_seed_and_document_rng_tracks_epoch():
    qwen = sft_example_rng(17, 23, 7, "qwen3")
    expected = example_rng(17, 23)
    np.testing.assert_array_equal(
        qwen.randint(2**31, size=16), expected.randint(2**31, size=16)
    )

    assert not np.array_equal(
        sft_example_rng(17, 23, 7, "qwen3").randint(2**31, size=16),
        sft_example_rng(999, 23, 7, "qwen3").randint(2**31, size=16),
    )

    document_epoch0 = sft_example_rng(17, 23, 0, "document")
    document_epoch1 = sft_example_rng(17, 23, 1, "document")
    assert not np.array_equal(
        document_epoch0.randint(2**31, size=16),
        document_epoch1.randint(2**31, size=16),
    )


def test_example_rng_preserves_seeded_molmo2_stream():
    expected = np.random.RandomState(17 * 195172 + 23).randint(2**31, size=16)
    np.testing.assert_array_equal(example_rng(17, 23).randint(2**31, size=16), expected)


@pytest.mark.parametrize("epoch", [None, 0, 3])
def test_bad_row_skip_supports_single_argument_and_epoch_aware_datasets(epoch):
    calls = []

    def build(index, *args):
        calls.append((index, args))
        if index == 1:
            raise ValueError("Invalid example")
        return {"row": index}

    class Dataset:
        _build = staticmethod(build)

    assert get_example_with_skip(Dataset(), 1, 3, epoch) == {"row": 2}
    expected_args = () if epoch is None else (epoch,)
    assert calls == [(1, expected_args), (2, expected_args)]


def test_truncate_example_uses_model_specific_image_patch_id():
    seq = {
        "input_ids": np.array([10, 11, 1234], dtype=np.int64),
        "labels": np.array([11, 12, 13], dtype=np.int64),
        "loss_masks": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "position_ids": np.arange(3, dtype=np.int64),
    }
    with pytest.raises(ValueError, match="drop <im_patch>"):
        truncate_example(seq, 2, image_patch_token_id=1234)


@pytest.mark.parametrize("max_len", [8])
def test_truncation_preserves_metadata_and_non_token_fields(max_len):
    token_fields = {
        "input_ids": np.arange(10, 18),
        "labels": np.arange(11, 19),
        "loss_masks": np.ones(8),
        "position_ids": np.arange(8),
    }
    other_fields = {
        "metadata": {"example_id": "row-1", "answers": ["answer"]},
        "source": "academic",
        "weight": 1.0,
        "optional": None,
        "references": list(range(8)),
        "scalar": np.array(1),
        "images": np.zeros((8, 2, 2)),
        "pooled_patches_idx": np.zeros((8, 4), dtype=np.int64),
    }
    example = {**token_fields, **other_fields}
    out = truncate_example(example, max_len)

    for key, value in token_fields.items():
        np.testing.assert_array_equal(out[key], value[:max_len])
        assert len(example[key]) == 8
    for key, value in other_fields.items():
        assert out[key] is value
    assert set(out) == set(example)


def test_truncate_recomputes_surviving_root_subsegment_weight():
    root_weight = 1.0 / np.sqrt(2.0)
    seq = {
        "input_ids": np.array([10, 20, 21, 30, 31, 32]),
        "labels": np.array([20, 21, 2, 31, 32, 2]),
        "loss_masks": np.array([0.0, root_weight, root_weight, 0.0, root_weight, root_weight]),
        "position_ids": np.arange(6),
        "token_type_ids": np.zeros(6, dtype=np.int64),
        "subsegment_ids": np.array([10000, 0, 0, 1, 1, 1]),
    }

    out = truncate_example(seq, 4, recompute_root_subsegments=True)

    # Branch 1 contributes only context before the limit, so vendor preprocessing omits it
    # and removes the original two-branch 1/sqrt(2) scaling from branch 0.
    assert len(out["input_ids"]) == 3
    np.testing.assert_allclose(out["loss_masks"], [0.0, 1.0, 1.0])


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
