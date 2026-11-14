from pathlib import Path

import numpy as np
import pytest

from olmo_core.data.composable import *


def test_sampling_instance_source(tmp_path: Path):
    sequence_length = 8
    source = SamplingInstanceSource(
        ConcatAndChunkInstanceSource(
            InMemoryTokenSource(list(range(64)), work_dir=tmp_path),
            sequence_length=sequence_length,
            work_dir=tmp_path,
        ),
        ConcatAndChunkInstanceSource(
            InMemoryTokenSource(list(range(64, 128)), work_dir=tmp_path),
            sequence_length=sequence_length,
            work_dir=tmp_path,
        ),
        max_instances=8,
        work_dir=tmp_path,
        seed=None,
    )
    assert source.source_sample_sizes == (4, 4)
    assert len(source) == 8

    assert list(source[0]["input_ids"]) == list(range(0, 8))
    assert list(source[4]["input_ids"]) == list(range(64, 64 + 8))


def test_sampling_instance_source_with_oversampling(tmp_path: Path):
    sequence_length = 8
    source = SamplingInstanceSource(
        ConcatAndChunkInstanceSource(
            InMemoryTokenSource(list(range(64)), work_dir=tmp_path),
            sequence_length=sequence_length,
            work_dir=tmp_path,
        ),
        ConcatAndChunkInstanceSource(
            InMemoryTokenSource(list(range(64, 128)), work_dir=tmp_path),
            sequence_length=sequence_length,
            work_dir=tmp_path,
        ),
        max_instances=24,
        work_dir=tmp_path,
        seed=None,
    )
    assert source.source_sample_sizes == (12, 12)
    assert len(source) == 24
    assert list(source[0]["input_ids"]) == list(range(0, 8))
    assert list(source[12]["input_ids"]) == list(range(64, 64 + 8))


def test_sampling_instance_source_with_random_sampling(tmp_path: Path, seed: int = 42):
    sequence_length = 8
    source = SamplingInstanceSource(
        ConcatAndChunkInstanceSource(
            InMemoryTokenSource(list(range(64)), work_dir=tmp_path),
            sequence_length=sequence_length,
            work_dir=tmp_path,
        ),
        ConcatAndChunkInstanceSource(
            InMemoryTokenSource(list(range(64, 128)), work_dir=tmp_path),
            sequence_length=sequence_length,
            work_dir=tmp_path,
        ),
        max_instances=8,
        seed=seed,
        work_dir=tmp_path,
    )
    assert source.source_sample_sizes == (4, 4)
    assert len(source) == 8

    for idx in range(4):
        for token in source[idx]["input_ids"]:
            assert 0 <= token < 64
    for idx in range(4, 8):
        for token in source[idx]["input_ids"]:
            assert 64 <= token < 128


def test_sampling_instance_source_with_random_oversampling(tmp_path: Path, seed: int = 42):
    sequence_length = 8
    source = SamplingInstanceSource(
        ConcatAndChunkInstanceSource(
            InMemoryTokenSource(list(range(64)), work_dir=tmp_path),
            sequence_length=sequence_length,
            work_dir=tmp_path,
        ),
        ConcatAndChunkInstanceSource(
            InMemoryTokenSource(list(range(64, 128)), work_dir=tmp_path),
            sequence_length=sequence_length,
            work_dir=tmp_path,
        ),
        max_instances=24,
        seed=seed,
        work_dir=tmp_path,
    )
    assert source.source_sample_sizes == (12, 12)
    assert len(source) == 24

    for idx in range(12):
        for token in source[idx]["input_ids"]:
            assert 0 <= token < 64
    for idx in range(12, 24):
        for token in source[idx]["input_ids"]:
            assert 64 <= token < 128


@pytest.mark.parametrize("seed", [None, 42])
@pytest.mark.parametrize("max_tokens", [32, 96])
def test_sampling_instance_source_with_sequence_length_warmup(tmp_path, seed, max_tokens):
    def sample(sequence_length):
        work_dir = tmp_path / str(sequence_length)
        source = SamplingInstanceSource(
            ConcatAndChunkInstanceSource(
                InMemoryTokenSource(list(range(64)), work_dir=work_dir),
                sequence_length=sequence_length,
                max_sequence_length=8,
                work_dir=work_dir,
            ),
            max_tokens=max_tokens,
            seed=seed,
            work_dir=work_dir,
        )
        return [token for i in range(len(source)) for token in source[i]["input_ids"]]

    expected = sample(8)
    assert len(expected) == max_tokens
    assert sample(4) == expected
    assert sample(2) == expected


def test_sampling_instance_source_rebuilds_legacy_index_dtype(tmp_path):
    source = ConcatAndChunkInstanceSource(
        InMemoryTokenSource(list(range(64)), work_dir=tmp_path),
        sequence_length=4,
        max_sequence_length=8,
        work_dir=tmp_path,
    )

    def sample():
        return SamplingInstanceSource(source, max_tokens=32, seed=None, work_dir=tmp_path)

    sampled = sample()
    index_path = next(sampled.work_dir.glob("*-indices.npy"))
    # Older chunk expansion wrote int64 indices, but the reader expects uint32.
    np.arange(8, dtype=np.int64).tofile(index_path)

    restored = sample()
    tokens = [token for i in range(len(restored)) for token in restored[i]["input_ids"]]
    assert tokens == list(range(32))
    assert index_path.stat().st_size == 8 * np.dtype(np.uint32).itemsize
