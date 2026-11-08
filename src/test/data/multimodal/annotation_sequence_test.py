"""``annotation_sampling="sequence"``: every annotation of a drawn image after one copy of the
image, in one document, each annotation closed by an EOS token."""

import numpy as np
import pytest

from olmo_core.data.multimodal.packing import pack_examples

from .annotation_documents_test import _TOKEN_KEYS, _TURNS, _documents, _encode
from .stage1_document_mode_test import DOLMA2_IDS, _points_v2, _Tok, _text_rich

_EOS = _Tok().eos_token_id


def _image_tokens(example):
    return int(np.isin(example["input_ids"], list(DOLMA2_IDS.image_token_ids)).sum())


def test_sequence_holds_one_image_then_every_document_body_in_order():
    seq = _encode(_TURNS, branches_as_sequence=True)
    docs = _encode(_TURNS, branches_as_documents=True)
    selections = _documents(docs)
    assert "document_ids" not in seq and "subsegment_ids" not in seq
    # The same shuffled order; the first document is prefix + its body.
    first = docs["input_ids"][selections[0]]
    image_and_boundary = int(np.flatnonzero(first == DOLMA2_IDS.im_end_id)[-1]) + 1
    expected = [first]
    for sel in selections[1:]:
        expected += [np.array([_EOS]), docs["input_ids"][sel][image_and_boundary:]]
    np.testing.assert_array_equal(seq["input_ids"], np.concatenate(expected))
    # One image block, pooled once.
    assert _image_tokens(seq) == _image_tokens(_encode(_TURNS[:1]))
    np.testing.assert_array_equal(seq["images"], docs["images"])
    np.testing.assert_array_equal(
        seq["pooled_patches_idx"], docs["pooled_patches_idx"][: len(seq["pooled_patches_idx"])]
    )
    assert len(seq["pooled_patches_idx"]) * len(_TURNS) == len(docs["pooled_patches_idx"])
    np.testing.assert_array_equal(seq["position_ids"], np.arange(len(seq["input_ids"])))


def test_sequence_keeps_every_documents_loss():
    seq = _encode(_TURNS, branches_as_sequence=True)
    docs = _encode(_TURNS, branches_as_documents=True)
    assert seq["loss_masks"].sum() == pytest.approx(docs["loss_masks"].sum())
    np.testing.assert_array_equal(
        np.sort(seq["labels"][seq["loss_masks"] > 0]),
        np.sort(docs["labels"][docs["loss_masks"] > 0]),
    )
    # Every inserted EOS is an input token whose prediction carries loss, and the context after
    # it carries none.
    eos_inputs = np.flatnonzero(seq["input_ids"] == _EOS)[1:]  # [0] is the document boundary
    assert len(eos_inputs) == len(_TURNS) - 1
    for i in eos_inputs:
        assert seq["labels"][i - 1] == _EOS and seq["loss_masks"][i - 1] > 0
        assert seq["loss_masks"][i] == 0
    assert seq["labels"][-1] == _EOS and seq["loss_masks"][-1] > 0


def test_sequence_scales_with_message_weight_like_documents():
    seq = _encode(_TURNS, branches_as_sequence=True, message_weight=0.5)
    docs = _encode(_TURNS, branches_as_documents=True, message_weight=0.5)
    assert seq["loss_masks"].sum() == pytest.approx(docs["loss_masks"].sum())


def test_sequence_token_budget_drops_whole_trailing_annotations():
    full = _encode(_TURNS, branches_as_sequence=True)
    eos_inputs = np.flatnonzero(full["input_ids"] == _EOS)[1:]
    # Budget that fits the first two annotations (with the EOS that closes the second) only.
    trimmed = _encode(
        _TURNS, branches_as_sequence=True, document_token_budget=int(eos_inputs[1]) + 1
    )
    np.testing.assert_array_equal(trimmed["input_ids"], full["input_ids"][: eos_inputs[1]])
    first = _encode(_TURNS, branches_as_sequence=True, document_token_budget=1)
    np.testing.assert_array_equal(first["input_ids"], full["input_ids"][: eos_inputs[0]])


def test_one_annotation_stays_a_plain_example():
    plain = _encode(_TURNS[:1])
    seq = _encode(_TURNS[:1], branches_as_sequence=True)
    assert set(seq) == set(plain)
    for key, value in plain.items():
        np.testing.assert_array_equal(seq[key], value)


def test_sequence_refuses_branch_scaled_weighting_and_documents():
    with pytest.raises(ValueError, match="loss_token_weighting='none'"):
        _encode(_TURNS, branches_as_sequence=True, loss_token_weighting="root_subsegments")
    with pytest.raises(ValueError, match="exclusive"):
        _encode(_TURNS, branches_as_sequence=True, branches_as_documents=True)


@pytest.mark.parametrize("build, n_annotations", [(_text_rich, 3), (_points_v2, 4)])
def test_sources_build_one_sequence_per_image(tmp_path, build, n_annotations):
    kwargs = dict(
        token_ids=DOLMA2_IDS,
        message_format="document",
        loss_token_weighting="none",
        max_sequence_length=65536,
    )
    seq_ds = build(tmp_path, annotation_sampling="sequence", **kwargs)
    doc_ds = build(tmp_path, annotation_sampling="documents", **kwargs)
    for index in range(len(seq_ds)):
        seq, docs = seq_ds.get(index, 0), doc_ds.get(index, 0)
        assert "document_ids" not in seq
        assert len(_documents(docs)) == n_annotations
        assert (np.flatnonzero(seq["input_ids"] == _EOS)[1:]).size == n_annotations - 1
        assert seq["loss_masks"].sum() == pytest.approx(docs["loss_masks"].sum())
        assert len(seq["images"]) == len(docs["images"])
        assert _image_tokens(seq) * n_annotations == _image_tokens(docs)
        for key in _TOKEN_KEYS:
            assert len(seq[key]) == len(seq["input_ids"])


def test_packed_sequences_are_isolated_by_example():
    seq = _encode(_TURNS, branches_as_sequence=True)
    plain = _encode(_TURNS[:1])
    packed = pack_examples([plain, seq, plain])
    assert "document_ids" not in packed
    n = len(plain["input_ids"])
    np.testing.assert_array_equal(
        packed["example_ids"],
        np.concatenate([np.zeros(n), np.ones(len(seq["input_ids"])), np.full(n, 2)]),
    )
