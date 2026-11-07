"""``annotation_sampling="documents"``: every annotation of a drawn image as its own document,
the documents sharing the image's crops."""

import numpy as np
import pytest
import torch
from PIL import Image

from olmo_core.data.multimodal import message_sequence
from olmo_core.data.multimodal.collator import MultimodalCollatorConfig
from olmo_core.data.multimodal.packing import pack_examples

from .stage1_document_mode_test import (
    DOLMA2_IDS,
    _assert_document_example,
    _points_v2,
    _Tok,
    _text_rich,
)

_TURNS = [
    ("Point to the cup.", "cup at 1"),
    ("Point to the spoon.", "spoon there"),
    ("Where is the plate?", "plate here ok"),
]
_TOKEN_KEYS = ("input_ids", "labels", "loss_masks", "position_ids", "token_type_ids")


def _encode(turns, **kwargs):
    kwargs.setdefault("loss_token_weighting", "none")
    return message_sequence.encode_sft_example(
        _Tok(),
        Image.new("RGB", (64, 48), color=(200, 30, 30)),
        turns,
        max_crops=1,
        token_ids=DOLMA2_IDS,
        message_format="document",
        shuffle_rng=np.random.RandomState(7),
        **kwargs,
    )


def _documents(example):
    ids = example["document_ids"]
    assert np.all(np.diff(ids) >= 0) and ids[0] == 0, "documents must be contiguous and in order"
    return [ids == d for d in range(int(ids.max()) + 1)]


def test_documents_are_the_single_annotation_examples_sharing_one_image():
    docs = _encode(_TURNS, branches_as_documents=True)
    single = [_encode([turn]) for turn in _TURNS]
    selections = _documents(docs)
    assert len(selections) == len(_TURNS)
    matched = []
    for sel in selections:
        (k,) = [
            i
            for i, s in enumerate(single)
            if np.array_equal(s["input_ids"], docs["input_ids"][sel])
        ]
        matched.append(k)
        for key in _TOKEN_KEYS:
            np.testing.assert_array_equal(docs[key][sel], single[k][key], err_msg=key)
    assert sorted(matched) == list(range(len(_TURNS)))
    # One copy of the crops; every document's image block pools them.
    np.testing.assert_array_equal(docs["images"], single[0]["images"])
    np.testing.assert_array_equal(
        docs["pooled_patches_idx"], np.tile(single[0]["pooled_patches_idx"], (len(_TURNS), 1))
    )
    assert (docs["input_ids"] == DOLMA2_IDS.im_patch_id).sum() == len(docs["pooled_patches_idx"])
    assert "subsegment_ids" not in docs


def test_documents_carry_the_sibling_layouts_loss():
    docs = _encode(_TURNS, branches_as_documents=True)
    siblings = _encode(_TURNS)
    assert docs["loss_masks"].sum() == pytest.approx(siblings["loss_masks"].sum())
    np.testing.assert_array_equal(
        np.sort(docs["labels"][docs["loss_masks"] > 0]),
        np.sort(siblings["labels"][siblings["loss_masks"] > 0]),
    )


def test_token_budget_drops_whole_trailing_documents():
    full = _encode(_TURNS, branches_as_documents=True)
    lengths = [int(sel.sum()) for sel in _documents(full)]
    trimmed = _encode(_TURNS, branches_as_documents=True, document_token_budget=sum(lengths) - 1)
    assert len(_documents(trimmed)) == 2
    np.testing.assert_array_equal(trimmed["input_ids"], full["input_ids"][: sum(lengths[:2])])
    assert len(trimmed["pooled_patches_idx"]) == 2 * len(full["pooled_patches_idx"]) // 3
    # The first document stays even when it alone exceeds the budget.
    first = _encode(_TURNS, branches_as_documents=True, document_token_budget=1)
    assert len(first["input_ids"]) == lengths[0]


def test_one_annotation_stays_a_plain_example():
    plain = _encode(_TURNS[:1])
    docs = _encode(_TURNS[:1], branches_as_documents=True)
    assert set(docs) == set(plain)
    for key, value in plain.items():
        np.testing.assert_array_equal(docs[key], value)


def test_documents_refuse_branch_scaled_weighting():
    with pytest.raises(ValueError, match="loss_token_weighting='none'"):
        _encode(_TURNS, branches_as_documents=True, loss_token_weighting="root_subsegments")


@pytest.mark.parametrize("build, n_annotations", [(_text_rich, 3), (_points_v2, 4)])
def test_sources_build_one_document_per_annotation(tmp_path, build, n_annotations):
    ds = build(
        tmp_path,
        token_ids=DOLMA2_IDS,
        message_format="document",
        annotation_sampling="documents",
        loss_token_weighting="none",
        max_sequence_length=65536,
    )
    one = build(
        tmp_path,
        token_ids=DOLMA2_IDS,
        message_format="document",
        annotation_sampling="one",
        loss_token_weighting="none",
        max_sequence_length=65536,
    )
    for index in range(len(ds)):
        example = ds.get(index, 0)
        selections = _documents(example)
        assert len(selections) == n_annotations
        n_crops = len(one.get(index, 0)["images"])
        assert len(example["images"]) == n_crops  # crops once, not per document
        for sel in selections:
            _assert_document_example({key: example[key][sel] for key in _TOKEN_KEYS})


def _document_example():
    docs = _encode(_TURNS, branches_as_documents=True)
    return docs, [int(sel.sum()) for sel in _documents(docs)]


def test_packing_numbers_every_document_of_the_pack():
    docs, lengths = _document_example()
    plain = _encode(_TURNS[:1])
    packed = pack_examples([plain, docs, plain])
    n = len(plain["input_ids"])
    expected = np.concatenate(
        [np.zeros(n)]
        + [np.full(length, 1 + d) for d, length in enumerate(lengths)]
        + [np.full(n, 1 + len(lengths))]
    )
    np.testing.assert_array_equal(packed["document_ids"], expected)
    np.testing.assert_array_equal(
        packed["example_ids"],
        np.concatenate([np.zeros(n), np.ones(sum(lengths)), np.full(n, 2)]),
    )
    # Plain packs keep their old fields.
    assert "document_ids" not in pack_examples([plain, plain])


def test_collator_pads_document_ids():
    docs, lengths = _document_example()
    plain = _encode(_TURNS[:1])
    batch = MultimodalCollatorConfig(pad_token_id=0).build()(
        [pack_examples([docs]), pack_examples([plain, plain])]
    )
    ids = batch["document_ids"]
    assert ids.dtype == torch.int64 and ids.shape == batch["input_ids"].shape
    total = sum(lengths)
    np.testing.assert_array_equal(
        ids[0, :total].numpy(), np.repeat(np.arange(len(lengths)), lengths)
    )
    assert (ids[0, total:] == -1).all()
    # A pack without documents numbers its examples.
    n = len(plain["input_ids"])
    np.testing.assert_array_equal(ids[1, : 2 * n].numpy(), np.repeat([0, 1], n))
    assert (ids[1, 2 * n :] == -1).all()
