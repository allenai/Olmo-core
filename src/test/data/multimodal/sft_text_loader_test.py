"""Text-only SFT batches for the multimodal trainer: open-instruct's order, and one EMo document
(and one attention document) per conversation.

Uses the open-instruct fixture cache of ``sft_numpy_cache_test.py``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from olmo_core.data.multimodal.sft_numpy_cache import SftNumpyCacheConfig
from olmo_core.data.multimodal.sft_text_loader import (
    SftTextDataLoaderConfig,
    batch_sha256,
    open_instruct_step_indices,
)
from olmo_core.nn.vision.multimodal import (
    document_lengths_from_example_ids,
    segment_ids_from_example_ids,
)
from olmo_core.ops.moe import segment_ids_from_eos

FIXTURE = Path(__file__).parents[2] / "fixtures" / "olmo35_c11" / "cache.npz"
EOS = 100257
PAD = 100277


@pytest.fixture(scope="module")
def reference():
    with np.load(FIXTURE) as f:
        return {key: f[key] for key in f.files}


@pytest.fixture
def numpy_dir(tmp_path, reference):
    out = tmp_path / "numpy"
    out.mkdir()
    for name in reference["file_names"]:
        (out / str(name)).write_bytes(reference[f"file:{name}"].tobytes())
    return out


def _loader(tmp_path, numpy_dir, reference, seed=34521, instances_per_batch=2):
    sequence_length = int(reference["sequence_length"])
    cache = SftNumpyCacheConfig(
        numpy_dir=str(numpy_dir), work_dir=str(tmp_path / "work"), sequence_length=sequence_length
    )
    loader = SftTextDataLoaderConfig(
        cache=cache,
        global_batch_size=instances_per_batch * sequence_length,
        seed=seed,
        work_dir=str(tmp_path / "loader"),
        num_workers=0,
    ).build()
    loader.reshuffle(epoch=1)
    return loader


def test_order_is_open_instructs(tmp_path, numpy_dir, reference):
    loader = _loader(tmp_path, numpy_dir, reference)
    batches = list(loader)
    expected = open_instruct_step_indices(
        len(reference["input_ids"]), seed=34521, instances_per_step=2, steps=len(batches)
    )
    assert [b["index"].tolist() for b in batches] == expected.tolist()


def test_order_is_reproducible(tmp_path, numpy_dir, reference):
    first = [batch_sha256(b) for b in _loader(tmp_path / "a", numpy_dir, reference)]
    second = [batch_sha256(b) for b in _loader(tmp_path / "b", numpy_dir, reference)]
    assert first == second and len(set(first)) == len(first)


def test_each_conversation_is_one_emo_and_attention_document(tmp_path, numpy_dir, reference):
    """The model takes EMo segments and attention documents from ``document_ids`` (via
    ``example_ids``). Each must be one whole cache row, though rows hold EOS after every
    assistant turn, where an EOS scan would split them."""
    loader = _loader(tmp_path, numpy_dir, reference)
    doc_counts = reference["doc_counts"]
    doc_lens = np.split(reference["doc_lens"], np.cumsum(doc_counts)[:-1])
    eos_would_split = 0
    for batch in loader:
        ids = batch["input_ids"]
        document_ids = batch["document_ids"]
        emo = segment_ids_from_example_ids(document_ids)
        lengths, _ = document_lengths_from_example_ids(document_ids)
        for row, index in enumerate(batch["index"].tolist()):
            expected = [int(n) for n in doc_lens[index] if n > 0]
            got = [int(n) for n in lengths[row].tolist() if n > 0]
            assert got == expected
            # One EMo segment per document, in order.
            changes = int((emo[row, 1:] != emo[row, :-1]).sum()) + 1
            assert changes == len(expected)
            eos = segment_ids_from_eos(ids[row : row + 1], EOS)
            eos_would_split += int((eos[0, 1:] != eos[0, :-1]).sum()) + 1 - changes
    assert eos_would_split > 0, "the fixture must hold multi-turn rows with EOS inside them"


def test_batch_format(tmp_path, numpy_dir, reference):
    batch = next(iter(_loader(tmp_path, numpy_dir, reference)))
    sequence_length = int(reference["sequence_length"])
    for key in ("input_ids", "labels", "loss_masks", "example_ids", "document_ids"):
        assert batch[key].shape == (2, sequence_length), key
    assert int(batch["image_crop_counts"].sum()) == 0
    assert batch["loss_masks"].dtype == torch.float32
