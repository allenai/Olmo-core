"""
Document boundaries for packed SFT data in the layout open-instruct writes: one raw token array, one
raw label-mask array and one ``(start,end)`` metadata row per conversation.

The chat templates end every non-tool assistant turn with EOS, so a multi-turn conversation carries
EOS tokens in its interior, and they end tool-call turns with ``<|im_end|>``, so a conversation that
stops on a tool call carries no EOS at all. Scanning for EOS therefore splits the first kind of row
and merges the second into its neighbour. Only the metadata knows where the rows are.
"""

import gzip
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from olmo_core.data import (
    LongDocStrategy,
    NumpyPackedFSLDataset,
    NumpyPackedFSLDatasetConfig,
    TokenizerConfig,
)

EOS = 100257
PAD = 100277
IM_START = 100264
IM_END = 100265
NL = 198
VOCAB_SIZE = 100352

# Each row is (token_ids, label_mask); assistant tokens are the trainable ones.
# A multi-turn conversation: the first assistant turn ends in EOS, mid-row.
MULTI_TURN = (
    [IM_START, 10, IM_END, NL, IM_START, 11, EOS, IM_START, 12, IM_END, NL, IM_START, 13, EOS],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
)
# A conversation that stops on a tool call: it ends in `<|im_end|>\n` and has no EOS.
TOOL_CALL = (
    [IM_START, 20, IM_END, NL, IM_START, 21, IM_END, NL],
    [0, 0, 0, 0, 0, 1, 1, 0],
)
# An ordinary single-turn conversation.
SINGLE_TURN = (
    [IM_START, 30, IM_END, NL, IM_START, 31, EOS],
    [0, 0, 0, 0, 0, 1, 1],
)
ROWS = [MULTI_TURN, TOOL_CALL, SINGLE_TURN]

# What the default (EOS-scan) path produces for ROWS at sequence length 32, recorded at OLMo-core
# 89e7dcb7, before `use_array_if_local` was plumbed through. The multi-turn row is split at its
# mid-row EOS into [0, 7) and [7, 14); the tool-call row is merged with the row after it, [14, 29).
EXPECTED_DEFAULT_SPANS = [[14, 29], [0, 7], [7, 14]]
EXPECTED_DEFAULT_FINGERPRINT = "2c75a3f0a74a3172f2e15045565fcc0faeb591a65e876ed5f5f7d81331ffca12"
EXPECTED_DEFAULT_INPUT_IDS = [TOOL_CALL[0] + SINGLE_TURN[0] + MULTI_TURN[0] + [PAD] * 3]
EXPECTED_DEFAULT_LABEL_MASK = [TOOL_CALL[1] + SINGLE_TURN[1] + MULTI_TURN[1] + [0] * 3]
EXPECTED_DEFAULT_DOC_LENS = [[15, 7, 7, 3]]


def _write_part(
    directory: Path, part: int, rows: List[Tuple[List[int], List[int]]]
) -> Tuple[Path, Path]:
    """Write one part the way open-instruct's numpy writer does."""
    token_path = directory / f"token_ids_part_{part:04d}.npy"
    label_path = directory / f"labels_mask_part_{part:04d}.npy"
    np.array([t for tokens, _ in rows for t in tokens], dtype=np.uint32).tofile(token_path)
    np.array([m for _, mask in rows for m in mask], dtype=np.bool_).tofile(label_path)
    with gzip.open(directory / f"token_ids_part_{part:04d}.csv.gz", "wt") as f:
        start = 0
        for tokens, _ in rows:
            f.write(f"{start},{start + len(tokens)}\n")
            start += len(tokens)
    return token_path, label_path


def _build(directory: Path, sequence_length: int, **kwargs) -> NumpyPackedFSLDataset:
    """Build the dataset the way open-instruct's olmo-core SFT path does."""
    ds = NumpyPackedFSLDatasetConfig(
        tokenizer=TokenizerConfig(vocab_size=VOCAB_SIZE, eos_token_id=EOS, pad_token_id=PAD),
        work_dir=str(directory / "work"),
        paths=[str(directory / "token_ids_part_*.npy")],
        expand_glob=True,
        label_mask_paths=[str(directory / "labels_mask_part_*.npy")],
        generate_doc_lengths=True,
        long_doc_strategy=LongDocStrategy.truncate,
        sequence_length=sequence_length,
        **kwargs,
    ).build()
    assert isinstance(ds, NumpyPackedFSLDataset)
    ds.prepare()
    return ds


def _document_indices(ds: NumpyPackedFSLDataset) -> List[List[int]]:
    """The packed ``(start, end)`` document spans of every source, in packing order."""
    spans = []
    for path in ds.paths:
        indices = np.fromfile(ds._get_document_indices_path(path), dtype=ds.indices_dtype)
        spans.append(indices.reshape(-1, 2).tolist())
    return spans


def _segments(item: Dict) -> List[Tuple[List[int], List[int]]]:
    """Split an instance into ``(token_ids, label_mask)`` segments along its ``doc_lens``."""
    out, start = [], 0
    for length in item["doc_lens"].tolist():
        out.append(
            (
                item["input_ids"][start : start + length].tolist(),
                item["label_mask"][start : start + length].int().tolist(),
            )
        )
        start += length
    assert start == item["input_ids"].numel()
    return out


def _real_segments(ds: NumpyPackedFSLDataset) -> List[Tuple[List[int], List[int]]]:
    """The non-padding segments of every instance. Padding must be one trailing segment."""
    out = []
    for i in range(len(ds)):
        segments = _segments(ds[i])
        if segments[-1][0][0] == PAD:
            tokens, mask = segments.pop()
            assert set(tokens) == {PAD} and not any(mask)
        assert all(PAD not in tokens for tokens, _ in segments)
        out.extend(segments)
    return out


@pytest.mark.parametrize("sequence_length", [16, 32])
def test_metadata_boundaries_keep_each_conversation_one_document(
    tmp_path: Path, sequence_length: int
):
    _write_part(tmp_path, 0, ROWS)
    ds = _build(tmp_path, sequence_length, use_array_if_local=False)

    # Packing: exactly the metadata rows, so neither the mid-row EOS nor the missing EOS moves a
    # boundary.
    (spans,) = _document_indices(ds)
    assert sorted(spans) == [[0, 14], [14, 22], [22, 29]]

    # doc_lens: every real segment of every instance is one whole conversation, with its labels.
    # Trailing padding is one final segment, as it is for the EOS scan.
    documents = _real_segments(ds)
    assert sorted(documents) == sorted(ROWS)


def test_metadata_boundaries_match_metadata_row_count(tmp_path: Path):
    parts = [ROWS, [TOOL_CALL, TOOL_CALL, MULTI_TURN], [SINGLE_TURN]]
    for part, rows in enumerate(parts):
        _write_part(tmp_path, part, rows)
    ds = _build(tmp_path, 32, use_array_if_local=False)

    for rows, spans in zip(parts, _document_indices(ds)):
        assert len(spans) == len(rows)
    assert sorted(_real_segments(ds)) == sorted(row for rows in parts for row in rows)


def test_default_boundaries_unchanged(tmp_path: Path):
    """With the option left unset, packing and every batch field are what they were before it
    existed: the EOS scan splits the multi-turn row in two and merges the tool-call row into the
    single-turn row after it. These values were recorded at OLMo-core 89e7dcb7, before the option
    was added, and that commit passes this test unchanged."""
    _write_part(tmp_path, 0, ROWS)
    ds = _build(tmp_path, 32)

    (spans,) = _document_indices(ds)
    assert spans == EXPECTED_DEFAULT_SPANS
    assert ds.fingerprint == EXPECTED_DEFAULT_FINGERPRINT
    items = [ds[i] for i in range(len(ds))]
    assert [item["input_ids"].tolist() for item in items] == EXPECTED_DEFAULT_INPUT_IDS
    assert [item["label_mask"].int().tolist() for item in items] == EXPECTED_DEFAULT_LABEL_MASK
    assert [item["doc_lens"].tolist() for item in items] == EXPECTED_DEFAULT_DOC_LENS
    # The cache file names do not depend on the option either, so existing caches are reused.
    for name, get_path in [
        ("document-indices", ds._get_document_indices_path),
        ("instance-offsets", ds._get_instance_offsets_path),
        ("documents-by-instance", ds._get_docs_by_instance_path),
    ]:
        assert get_path(ds.paths[0]) == ds._get_indices_path(
            name, ds.paths[0], extra_ids=(LongDocStrategy.truncate, ds.indices_dtype.__name__)
        )
