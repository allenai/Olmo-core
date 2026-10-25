"""Multi-image processing / formatting tests (mm_olmo MultiImagePreprocessor parity)."""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from olmo_core.data.multimodal.grounding import (
    format_multi_image_points_tag,
    multi_image_pointing_answer,
)
from olmo_core.data.multimodal.message_sequence import encode_sft_example
from olmo_core.data.multimodal.sft_formatter import SftFormatter


@pytest.fixture(scope="module")
def tokenizer():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("allenai/Molmo2-4B", trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer unavailable: {e}")


def _img(w=64, h=48, color=(255, 0, 0)):
    return Image.new("RGB", (w, h), color)


# ---------------------------------------------------------------------------
# Point serialization
# ---------------------------------------------------------------------------


def test_multi_image_points_tag_continuing_indices_and_sep():
    tag = format_multi_image_points_tag([(1, [(0.5, 0.5), (0.1, 0.2)]), (3, [(0.9, 0.9)])], "cat")
    # Per-image points sorted by (x, y); point ids continue across images; image
    # groups joined by ";" (html-v2); image indices are 1-based display indices.
    assert tag == '<points coords="1 1 100 200 2 500 500;3 3 900 900">cat</points>'


def test_multi_image_pointing_answer_modes():
    pts = [(1, [(0.5, 0.5)]), (2, [(0.25, 0.75)])]
    tag = format_multi_image_points_tag(pts, "dog")
    assert multi_image_pointing_answer(pts, "dog", "multi_image_pointing") == tag
    assert (
        multi_image_pointing_answer(pts, "dog", "multi_image_point_then_count")
        == f"Counting the {tag} shows a total of 2."
    )
    assert multi_image_pointing_answer([], "dog", "multi_image_pointing") == "There are none."


# ---------------------------------------------------------------------------
# Formatter branch
# ---------------------------------------------------------------------------


def _multi_points_example():
    return {
        "style": "multi_image_pointing",
        "labels": ["Dog", "Dog"],
        "normalized_labels": ["dog", "dog"],
        "points": [[{"x": 50.0, "y": 50.0}], [{"x": 25.0, "y": 75.0}]],
        "point_scale": 100,
        "clip_points": True,
    }


def test_formatter_multi_image_pointing_produces_question_and_points():
    fmt = SftFormatter(seed=0)
    ex = _multi_points_example()
    turns = fmt.format_turns(ex, is_training=True, rng=np.random.RandomState(0))
    assert len(turns) == 1
    q, a = turns[0]
    assert "dog" in q.lower()
    assert a == "There are none." or "<points coords=" in a


def test_formatter_multi_image_pointing_no_label_match():
    fmt = SftFormatter(seed=0)
    ex = _multi_points_example()
    ex["points"] = [[], []]
    q, a = fmt.format_turns(ex, is_training=True, rng=np.random.RandomState(0))[0]
    assert a == "There are none."


def test_formatter_multi_image_pointing_deterministic():
    fmt = SftFormatter(seed=0)
    t1 = fmt.format_turns(_multi_points_example(), rng=np.random.RandomState(7))
    t2 = fmt.format_turns(_multi_points_example(), rng=np.random.RandomState(7))
    assert t1 == t2


# ---------------------------------------------------------------------------
# Multi-image encoding
# ---------------------------------------------------------------------------


def test_encode_two_images_prefixes_and_offsets(tokenizer):
    from olmo_core.nn.vision.molmo2_tokens import IM_PATCH_ID

    seq = encode_sft_example(
        tokenizer,
        [_img(64, 48), _img(96, 64, color=(0, 255, 0))],
        [("What differs?", "The color.")],
        max_crops=4,
    )
    text = tokenizer.decode(seq["input_ids"])
    assert "Image 1" in text and "Image 2" in text
    # Every pooled row indexes into the concatenated crop-patch axis.
    total_patches = seq["images"].shape[0] * seq["images"].shape[1]
    valid = seq["pooled_patches_idx"][seq["pooled_patches_idx"] >= 0]
    assert valid.max() < total_patches
    # One <im_patch> token per pooled row (the MultimodalLM splice invariant).
    n_pool_tokens = int((seq["input_ids"] == IM_PATCH_ID).sum())
    assert n_pool_tokens == seq["pooled_patches_idx"].shape[0]


def test_encode_single_image_has_no_image_prefix(tokenizer):
    seq = encode_sft_example(tokenizer, _img(), [("Describe.", "A red square.")], max_crops=4)
    text = tokenizer.decode(seq["input_ids"])
    assert "Image 1" not in text


def test_encode_single_image_unchanged_by_multi_support(tokenizer):
    """List-of-one and bare image must produce identical sequences."""
    a = encode_sft_example(tokenizer, _img(), [("Q?", "A.")], max_crops=4)
    b = encode_sft_example(tokenizer, [_img()], [("Q?", "A.")], max_crops=4)
    assert np.array_equal(a["input_ids"], b["input_ids"])
    assert np.array_equal(a["pooled_patches_idx"], b["pooled_patches_idx"])
    assert np.array_equal(a["images"], b["images"])


def test_encode_truncates_to_max_images(tokenizer):
    seq = encode_sft_example(
        tokenizer,
        [_img() for _ in range(4)],
        [("Q?", "A.")],
        max_crops=2,
        max_images=2,
    )
    text = tokenizer.decode(seq["input_ids"])
    assert "Image 2" in text and "Image 3" not in text


# ---------------------------------------------------------------------------
# Dataset smoke tests (weka data; skipped when unavailable)
# ---------------------------------------------------------------------------

_WEKA = "/weka/oe-training-default/mm-olmo/torch_datasets"


def _weka_or_skip(path: str):
    if not os.path.exists(path):
        pytest.skip(f"{path} not available")


@pytest.mark.parametrize("subset", ["nlvr2", "spot-the-diff"])
def test_mantis_dataset_smoke(tokenizer, subset):
    _weka_or_skip(f"{_WEKA}/academic_datasets/mantis-instruct/{subset}")
    from olmo_core.data.multimodal.multi_image_datasets import (
        MantisInstructDatasetConfig,
    )

    ds = MantisInstructDatasetConfig(subset=subset).build(tokenizer)
    assert len(ds) > 0
    seq = ds[0]
    assert seq["images"].shape[0] > 0
    assert (seq["loss_masks"] > 0).any()


def test_cosyn_multidoc_dataset_smoke(tokenizer):
    _weka_or_skip(f"{_WEKA}/pixmo_datasets/pixmo_docs_multi/chart_metadata_v3.json")
    from olmo_core.data.multimodal.multi_image_datasets import (
        CoSynMultiDocDatasetConfig,
    )

    ds = CoSynMultiDocDatasetConfig(doc_type="chart").build(tokenizer)
    assert len(ds) > 0
    seq = ds[0]
    assert (seq["loss_masks"] > 0).any()


def test_correction_qa_dataset_smoke(tokenizer):
    _weka_or_skip(f"{_WEKA}/pixmo_datasets/correction-qa/train-records.json")
    from olmo_core.data.multimodal.multi_image_datasets import CorrectionQaDatasetConfig

    ds = CorrectionQaDatasetConfig().build(tokenizer)
    assert len(ds) > 0
    seq = ds[0]
    text = tokenizer.decode(seq["input_ids"])
    assert "Image 1" in text  # multi_image_only guarantees >= 2 images
    assert (seq["loss_masks"] > 0).any()


def test_pixmo_multi_points_dataset_smoke(tokenizer):
    _weka_or_skip(f"{_WEKA}/pixmo_datasets/pixmo-multi-points")
    from olmo_core.data.multimodal.multi_image_datasets import (
        PixMoMultiPointsDatasetConfig,
    )

    ds = PixMoMultiPointsDatasetConfig(message_weight=0.2).build(tokenizer)
    assert len(ds) > 0
    seq = ds[1]
    assert (seq["loss_masks"] >= 0).all()
