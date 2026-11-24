"""The C11 numpy-cache reader against open-instruct's packed instances (fixture from
``src/scripts/make_olmo35_c11_fixture.py``: a cache written by open-instruct's
``convert_hf_to_numpy_sft`` with row-aligned parts, packed by its ``_numpy_dataset_config``
with metadata boundaries at OLMo-core 2f74ec30)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from olmo_core.data.multimodal.sft_numpy_cache import (
    TOKEN_IDS_METADATA_GLOB,
    SftNumpyCacheConfig,
    check_row_aligned_cache,
)
from olmo_core.exceptions import OLMoConfigurationError

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


def _doc_lens(reference):
    return np.split(reference["doc_lens"], np.cumsum(reference["doc_counts"])[:-1])


def _row_lengths(numpy_dir: Path):
    lengths = []
    for path in sorted(numpy_dir.glob(TOKEN_IDS_METADATA_GLOB)):
        with gzip.open(path, "rt") as f:
            lengths += [int(end) - int(start) for start, end in (line.split(",") for line in f)]
    return lengths


def test_instances_match_open_instruct(tmp_path, numpy_dir, reference):
    sequence_length = int(reference["sequence_length"])
    dataset = SftNumpyCacheConfig(
        numpy_dir=str(numpy_dir), work_dir=str(tmp_path / "work"), sequence_length=sequence_length
    ).build()
    assert len(dataset) == len(reference["input_ids"])
    for index, (ids, mask, doc_lens) in enumerate(
        zip(reference["input_ids"], reference["label_mask"], _doc_lens(reference))
    ):
        example = dataset[index]
        assert example["input_ids"].tolist() == ids.tolist()
        np.testing.assert_array_equal(example["loss_masks"][:-1].astype(bool), mask[1:])
        assert example["loss_masks"][-1] == 0
        np.testing.assert_array_equal(example["labels"][:-1], np.where(mask[1:], ids[1:], -100))
        np.testing.assert_array_equal(
            example["document_ids"], np.repeat(np.arange(len(doc_lens)), doc_lens)
        )
        assert example["images"].shape[0] == 0


def test_no_document_is_split_at_eos(tmp_path, numpy_dir, reference):
    """Every document is one whole cache row, even when the row holds EOS mid-way (each
    assistant turn of a multi-turn row ends with EOS)."""
    sequence_length = int(reference["sequence_length"])
    dataset = SftNumpyCacheConfig(
        numpy_dir=str(numpy_dir), work_dir=str(tmp_path / "work"), sequence_length=sequence_length
    ).build()
    rows = sorted(min(n, sequence_length) for n in _row_lengths(numpy_dir))
    documents = []
    interior_eos = 0
    for index in range(len(dataset)):
        example = dataset[index]
        ids, doc_ids = example["input_ids"], example["document_ids"]
        for doc in np.unique(doc_ids):
            span = ids[doc_ids == doc]
            if (span == PAD).all():
                continue
            documents.append(len(span))
            interior_eos += int((span[:-1] == EOS).sum())
            assert span[-1] in (EOS, PAD) or len(span) == sequence_length
    assert sorted(documents) == rows
    assert interior_eos > 0, "the fixture must hold rows with EOS inside them"


def test_mid_row_cut_cache_is_refused(numpy_dir):
    stats_path = numpy_dir / "dataset_statistics.json"
    stats = json.loads(stats_path.read_text())
    stats["configuration"]["row_aligned_parts"] = False
    stats_path.write_text(json.dumps(stats))
    with pytest.raises(OLMoConfigurationError, match="row_aligned_parts"):
        check_row_aligned_cache(str(numpy_dir))


def _write_replay_dir(path: Path, row_lengths, *, seed: int = 0, metadata: bool = True) -> Path:
    """A long-context replay dir as the current text SFT recipe's: one document plus EOS per row,
    every position a label, one metadata line per row, uint32 tokens (Dolma2's vocabulary)."""
    rng = np.random.RandomState(seed)
    path.mkdir()
    tokens, lines, start = [], [], 0
    for length in row_lengths:
        row = rng.randint(1000, 90000, size=length - 1).tolist() + [EOS]
        tokens += row
        lines.append(f"{start},{start + length}")
        start += length
    (path / "token_ids_part_0000.npy").write_bytes(np.asarray(tokens, dtype=np.uint32).tobytes())
    (path / "labels_mask_part_0000.npy").write_bytes(np.ones(len(tokens), dtype=np.uint8).tobytes())
    if metadata:
        with gzip.open(path / "token_ids_part_0000.csv.gz", "wt") as f:
            f.write("\n".join(lines) + "\n")
    return path


def test_extra_numpy_dir_adds_replay_instances(tmp_path, numpy_dir, reference):
    sequence_length = int(reference["sequence_length"])
    replay = _write_replay_dir(tmp_path / "replay", [sequence_length // 2, sequence_length // 3])
    base = SftNumpyCacheConfig(
        numpy_dir=str(numpy_dir), work_dir=str(tmp_path / "w0"), sequence_length=sequence_length
    ).build()
    with_replay = SftNumpyCacheConfig(
        numpy_dir=str(numpy_dir),
        work_dir=str(tmp_path / "w1"),
        sequence_length=sequence_length,
        extra_numpy_dirs=[str(replay)],
    ).build()
    assert len(with_replay) > len(base)
    # The cache's instances are unchanged; the replay's rows train every position but the last.
    for index in range(len(base)):
        np.testing.assert_array_equal(base[index]["input_ids"], with_replay[index]["input_ids"])
    replay_ids = set(np.frombuffer((replay / "token_ids_part_0000.npy").read_bytes(), np.uint32))
    found = False
    for index in range(len(base), len(with_replay)):
        example = with_replay[index]
        doc = example["document_ids"]
        for d in np.unique(doc):
            ids = example["input_ids"][doc == d]
            if ids[-1] == EOS and set(ids.tolist()) <= replay_ids:
                found = True
                assert (example["loss_masks"][doc == d][:-1] > 0).all()
    assert found


def test_extra_numpy_dir_without_metadata_is_refused(tmp_path, numpy_dir, reference):
    replay = _write_replay_dir(tmp_path / "replay", [100, 200], metadata=False)
    with pytest.raises(OLMoConfigurationError, match="metadata"):
        SftNumpyCacheConfig(
            numpy_dir=str(numpy_dir),
            work_dir=str(tmp_path / "w"),
            sequence_length=int(reference["sequence_length"]),
            extra_numpy_dirs=[str(replay)],
        ).build()
