import gzip
import math
from pathlib import Path
from typing import List

import numpy as np
import pytest

from olmo_core.data import (
    LongDocStrategy,
    NumpyFSLDataset,
    NumpyFSLDatasetConfig,
    NumpyPackedFSLDataset,
    NumpyPackedFSLDatasetConfig,
    NumpyPaddedFSLDataset,
    NumpyVSLDataset,
    TokenizerConfig,
    numpy_dataset,
)
from olmo_core.data.numpy_dataset import (
    NumpyFSLDatasetMixture,
    NumpyInterleavedFSLDataset,
)
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDatasetConfig,
    SourceMixtureList,
)
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.data.utils import (
    get_document_indices,
    get_document_lengths,
    write_document_indices,
)
from olmo_core.io import get_file_size

from .utils import mk_mmaps


def test_numpy_fsl_dataset(tmp_path: Path):
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(16,))
    mmap1[:] = list(range(16))
    mmap1.flush()

    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(16,))
    mmap2[:] = list(range(16, 32))
    mmap2.flush()

    ds = NumpyFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=4,
        pad_token_id=-1,
        eos_token_id=-1,
        vocab_size=32_000,
    )
    assert ds[0]["input_ids"].tolist() == [0, 1, 2, 3]
    assert ds[1]["input_ids"].tolist() == [4, 5, 6, 7]
    assert ds[7]["input_ids"].tolist() == [28, 29, 30, 31]
    assert len(ds) == 8


def test_numpy_fsl_dataset_doc_lengths(tmp_path: Path):
    data1 = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(len(data1),))
    mmap1[:] = data1
    mmap1.flush()

    data2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 21, 22, 0]
    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(len(data2),))
    mmap2[:] = data2
    mmap2.flush()

    ds = NumpyFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=4,
        pad_token_id=-1,
        eos_token_id=0,
        vocab_size=32_000,
        generate_doc_lengths=True,
    )
    assert ds[0]["input_ids"].tolist() == [1, 0, 2, 3]
    assert ds[0]["doc_lens"].tolist() == [2, 2]
    assert ds[1]["input_ids"].tolist() == [4, 5, 6, 7]
    assert ds[1]["doc_lens"].tolist() == [4]
    assert ds[5]["input_ids"].tolist() == [19, 20, 0, 21]
    assert ds[5]["doc_lens"].tolist() == [3, 1]
    assert len(ds) == 6


def test_numpy_fsl_dataset_doc_lengths_with_bos(tmp_path: Path):
    data1 = [0, 1, 100, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(len(data1),))
    mmap1[:] = data1
    mmap1.flush()

    data2 = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 100, 0, 21, 22, 100]
    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(len(data2),))
    mmap2[:] = data2
    mmap2.flush()

    ds = NumpyFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=4,
        pad_token_id=-1,
        eos_token_id=100,
        vocab_size=32_000,
        generate_doc_lengths=True,
        bos_token_id=0,
    )
    assert ds[0]["input_ids"].tolist() == [0, 1, 100, 0]
    assert ds[0]["doc_lens"].tolist() == [3, 1]
    assert ds[1]["input_ids"].tolist() == [2, 3, 4, 5]
    assert ds[1]["doc_lens"].tolist() == [4]
    assert ds[6]["input_ids"].tolist() == [0, 21, 22, 100]
    assert ds[6]["doc_lens"].tolist() == [4]
    assert len(ds) == 7


def test_numpy_fsl_dataset_doc_lengths_same_bos_and_eos(tmp_path: Path):
    data1 = [0, 1, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(len(data1),))
    mmap1[:] = data1
    mmap1.flush()

    data2 = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 0, 21, 22, 0]
    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(len(data2),))
    mmap2[:] = data2
    mmap2.flush()

    ds = NumpyFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=4,
        pad_token_id=-1,
        eos_token_id=0,
        vocab_size=32_000,
        generate_doc_lengths=True,
        bos_token_id=0,
    )
    assert ds[0]["input_ids"].tolist() == [0, 1, 0, 0]
    assert ds[0]["doc_lens"].tolist() == [3, 1]
    assert ds[1]["input_ids"].tolist() == [2, 3, 4, 5]
    assert ds[1]["doc_lens"].tolist() == [4]
    assert ds[6]["input_ids"].tolist() == [0, 21, 22, 0]
    assert ds[6]["doc_lens"].tolist() == [4]
    assert len(ds) == 7


def test_numpy_fsl_dataset_with_label_mask(tmp_path: Path):
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(16,))
    mmap1[:] = list(range(16))
    mmap1.flush()

    mmap1_mask = np.memmap(tmp_path / "mmap1_mask.npy", mode="w+", dtype=np.bool_, shape=(16,))
    mmap1_mask[:] = True
    mmap1_mask[7] = False
    mmap1_mask.flush()

    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(16,))
    mmap2[:] = list(range(16, 32))
    mmap2.flush()

    mmap2_mask = np.memmap(tmp_path / "mmap2_mask.npy", mode="w+", dtype=np.bool_, shape=(16,))
    mmap2_mask[:] = True
    mmap2_mask[0:3] = False
    mmap2_mask.flush()

    ds = NumpyFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=4,
        pad_token_id=-1,
        eos_token_id=-1,
        vocab_size=32_000,
        label_mask_paths=[tmp_path / "mmap1_mask.npy", tmp_path / "mmap2_mask.npy"],
    )
    assert len(ds) == 8

    assert ds[0]["input_ids"].tolist() == [0, 1, 2, 3]
    assert ds[0]["label_mask"].tolist() == [True, True, True, True]

    assert ds[1]["input_ids"].tolist() == [4, 5, 6, 7]
    assert ds[1]["label_mask"].tolist() == [True, True, True, False]

    assert ds[4]["input_ids"].tolist() == [16, 17, 18, 19]
    assert ds[4]["label_mask"].tolist() == [False, False, False, True]

    assert ds[7]["input_ids"].tolist() == [28, 29, 30, 31]
    assert ds[7]["label_mask"].tolist() == [True, True, True, True]


def test_numpy_padded_fsl_dataset(tmp_path: Path):
    data1 = [1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 0]
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(len(data1),))
    mmap1[:] = data1
    mmap1.flush()

    data2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 21, 22, 0]
    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(len(data2),))
    mmap2[:] = data2
    mmap2.flush()

    ds = NumpyPaddedFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=8,
        pad_token_id=0,
        eos_token_id=0,
        vocab_size=32_000,
    )
    ds.prepare()
    assert ds[0]["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 0]
    assert ds[0]["label_mask"].tolist() == [True] * 8
    assert ds[1]["input_ids"].tolist() == [8, 9, 10, 0, 0, 0, 0, 0]
    assert ds[1]["label_mask"].tolist() == [True] * 4 + [False] * 4
    assert ds[2]["input_ids"].tolist() == [11, 12, 13, 14, 15, 16, 17, 18]
    assert ds[3]["input_ids"].tolist() == [21, 22, 0, 0, 0, 0, 0, 0]
    assert len(ds) == 4


def test_numpy_padded_fsl_dataset_with_label_mask(tmp_path: Path):
    data1 = [1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 0]
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(len(data1),))
    mmap1[:] = data1
    mmap1.flush()

    data1_mask = [False, True, True, True, True, True, True, True] + [True, True, True, True]
    mmap1_mask = np.memmap(
        tmp_path / "mmap1_mask.npy", mode="w+", dtype=np.bool_, shape=(len(data1_mask),)
    )
    mmap1_mask[:] = data1_mask
    mmap1_mask.flush()

    data2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 21, 22, 0]
    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(len(data2),))
    mmap2[:] = data2
    mmap2.flush()

    data2_mask = [True, True, True, True, True, True, True, True, True, True, True] + [
        True,
        True,
        True,
    ]
    mmap2_mask = np.memmap(
        tmp_path / "mmap2_mask.npy", mode="w+", dtype=np.bool_, shape=(len(data2_mask),)
    )
    mmap2_mask[:] = data2_mask
    mmap2_mask.flush()

    ds = NumpyPaddedFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=8,
        pad_token_id=0,
        eos_token_id=0,
        vocab_size=32_000,
        label_mask_paths=[tmp_path / "mmap1_mask.npy", tmp_path / "mmap2_mask.npy"],
    )

    ds.prepare()
    assert len(ds) == 4

    assert ds[0]["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 0]
    assert ds[0]["label_mask"].tolist() == [False] + [True] * 7

    assert ds[1]["input_ids"].tolist() == [8, 9, 10, 0, 0, 0, 0, 0]
    assert ds[1]["label_mask"].tolist() == [True] * 4 + [False] * 4

    assert ds[2]["input_ids"].tolist() == [11, 12, 13, 14, 15, 16, 17, 18]
    assert ds[3]["input_ids"].tolist() == [21, 22, 0, 0, 0, 0, 0, 0]


@pytest.mark.parametrize("long_doc_strategy", [LongDocStrategy.truncate, LongDocStrategy.fragment])
def test_numpy_packed_fsl_dataset(tmp_path: Path, long_doc_strategy):
    data1 = np.array(
        [1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 1, 2, 0]
    )
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(len(data1),))
    mmap1[:] = data1
    mmap1.flush()

    data2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 0]
    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(len(data2),))
    mmap2[:] = data2
    mmap2.flush()

    ds = NumpyPackedFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=8,
        pad_token_id=-1,
        eos_token_id=0,
        vocab_size=32_000,
        long_doc_strategy=long_doc_strategy,
    )
    ds.prepare()
    assert len(ds) == 6

    assert ds[0]["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 0]
    assert ds[0]["label_mask"].tolist() == [True] * 8

    assert ds[1]["input_ids"].tolist() == [1, 2, 3, 4, 5, 0, -1, -1]
    assert ds[1]["label_mask"].tolist() == [True] * 6 + [False] * 2

    assert ds[3]["input_ids"].tolist() == [1, 2, 3, 0, 1, 2, 0, -1]
    assert ds[3]["label_mask"].tolist() == [True] * 7 + [False]

    if long_doc_strategy == LongDocStrategy.truncate:
        assert ds[5]["input_ids"].tolist() == [1, 2, 0, -1, -1, -1, -1, -1]
        assert ds[5]["label_mask"].tolist() == [True] * 3 + [False] * 5
    elif long_doc_strategy == LongDocStrategy.fragment:
        assert ds[5]["input_ids"].tolist() == [9, 10, 0, 1, 2, 0, -1, -1]
        assert ds[5]["label_mask"].tolist() == [True] * 6 + [False] * 2
    else:
        raise ValueError(long_doc_strategy)


@pytest.mark.parametrize("long_doc_strategy", [LongDocStrategy.truncate, LongDocStrategy.fragment])
def test_numpy_packed_fsl_dataset_with_label_mask(tmp_path: Path, long_doc_strategy):
    data1 = [1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 1, 2, 0]
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(len(data1),))
    mmap1[:] = data1
    mmap1.flush()

    data1_mask = [True] * len(data1)
    data1_mask[1] = False
    mmap1_mask = np.memmap(
        tmp_path / "mmap1_mask.npy", mode="w+", dtype=np.bool_, shape=(len(data1_mask),)
    )
    mmap1_mask[:] = data1_mask
    mmap1_mask.flush()

    data2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 0]
    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(len(data2),))
    mmap2[:] = data2
    mmap2.flush()

    data2_mask = [True] * len(data2)
    data2_mask[8] = False
    mmap2_mask = np.memmap(
        tmp_path / "mmap2_mask.npy", mode="w+", dtype=np.bool_, shape=(len(data2_mask),)
    )
    mmap2_mask[:] = data2_mask
    mmap2_mask.flush()

    ds = NumpyPackedFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=8,
        pad_token_id=-1,
        eos_token_id=0,
        vocab_size=32_000,
        label_mask_paths=[tmp_path / "mmap1_mask.npy", tmp_path / "mmap2_mask.npy"],
        long_doc_strategy=long_doc_strategy,
    )
    ds.prepare()
    assert len(ds) == 6

    assert ds[0]["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 0]
    assert ds[0]["label_mask"].tolist() == [True, False] + [True] * 6

    assert ds[1]["input_ids"].tolist() == [1, 2, 3, 4, 5, 0, -1, -1]
    assert ds[1]["label_mask"].tolist() == [True] * 6 + [False] * 2

    assert ds[3]["input_ids"].tolist() == [1, 2, 3, 0, 1, 2, 0, -1]
    assert ds[3]["label_mask"].tolist() == [True] * 7 + [False]

    if long_doc_strategy == LongDocStrategy.truncate:
        assert ds[5]["input_ids"].tolist() == [1, 2, 0, -1, -1, -1, -1, -1]
        assert ds[5]["label_mask"].tolist() == [True] * 3 + [False] * 5
    elif long_doc_strategy == LongDocStrategy.fragment:
        assert ds[5]["input_ids"].tolist() == [9, 10, 0, 1, 2, 0, -1, -1]
        assert ds[5]["label_mask"].tolist() == [False, True] + [True] * 4 + [False] * 2
    else:
        raise ValueError(long_doc_strategy)


def test_numpy_packed_fsl_dataset_with_source_grouping(tmp_path: Path):
    data1 = np.array(
        [1, 2, 3, 0, 11, 12, 13, 14, 15, 0, 21, 22, 23, 24, 25, 0, 31, 32, 33, 0, 41, 42, 0],
    )
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(len(data1),))
    mmap1[:] = data1
    mmap1.flush()

    data2 = [51, 52, 0, 61, 62, 63, 64, 65, 66, 67, 0]
    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(len(data2),))
    mmap2[:] = data2
    mmap2.flush()

    data3 = [71, 72, 73, 74, 75, 76, 77, 78, 79, 0, 81, 82, 0]
    mmap3 = np.memmap(tmp_path / "mmap3.npy", mode="w+", dtype=np.uint16, shape=(len(data3),))
    mmap3[:] = data3
    mmap3.flush()

    data4 = [91, 92, 93, 94, 0]
    mmap4 = np.memmap(tmp_path / "mmap4.npy", mode="w+", dtype=np.uint16, shape=(len(data4),))
    mmap4[:] = data4
    mmap4.flush()

    ds = NumpyPackedFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        tmp_path / "mmap3.npy",
        tmp_path / "mmap4.npy",
        sequence_length=8,
        pad_token_id=-1,
        eos_token_id=0,
        vocab_size=32_000,
        source_group_size=2,
    )
    ds.prepare()

    # NOTE: potentially brittle test here!
    # Hard-coding exactly what the instances should be to ensure it's deterministic.
    assert len(ds) == 7
    assert ds[0]["input_ids"].tolist() == [61, 62, 63, 64, 65, 66, 67, 0]
    assert ds[1]["input_ids"].tolist() == [11, 12, 13, 14, 15, 0, -1, -1]
    assert ds[2]["input_ids"].tolist() == [21, 22, 23, 24, 25, 0, -1, -1]
    assert ds[3]["input_ids"].tolist() == [1, 2, 3, 0, 31, 32, 33, 0]
    assert ds[4]["input_ids"].tolist() == [41, 42, 0, 51, 52, 0, -1, -1]
    assert ds[5]["input_ids"].tolist() == [71, 72, 73, 74, 75, 76, 77, 78]
    assert ds[6]["input_ids"].tolist() == [91, 92, 93, 94, 0, 81, 82, 0]


def test_numpy_fsl_mixture_dataset(tmp_path: Path):
    # NOTE: At small token counts the take_ratio can be finicky so we test at small but real world-ish scale
    npdtype = np.uint16
    seed = 42
    mmap1 = mk_mmaps(tmp_path, "mmap1", 1, 20 * 1000, npdtype, eos=0, seed=seed)
    mmap2 = mk_mmaps(tmp_path, "mmap2", 1, 20 * 1000, npdtype, eos=0, seed=seed * 2)

    sequence_length = 4
    tokenizer = TokenizerConfig(
        vocab_size=32_000,
        eos_token_id=0,
        pad_token_id=-1,
    )

    bsz = 32
    max_tokens = 10_000

    mixture_config = SourceMixtureDatasetConfig(
        render_tables=False,
        requested_tokens=max_tokens,
        source_list=SourceMixtureList(
            [
                SourceMixtureConfig(
                    source_name="mmap1",
                    paths=[str(i[0]) for i in mmap1],
                    target_ratio=0.8,
                ),
                SourceMixtureConfig(
                    source_name="mmap2",
                    paths=[str(i[0]) for i in mmap2],
                    target_ratio=0.2,
                ),
            ]
        ),
        seed=seed,
        global_batch_size=sequence_length * bsz,
    )

    ds = NumpyFSLDatasetConfig(
        source_mixture_config=mixture_config,
        sequence_length=sequence_length,
        tokenizer=tokenizer,
        dtype=NumpyDatasetDType.uint16,
        include_instance_metadata=False,
    ).build()
    ds.prepare()

    first_ds_item = ds[0]["input_ids"].tolist()

    # NOTE: This is commented out until we fix behavior of the source mixture dataset
    # first_src_sequence = mmap1[0][1][:sequence_length].tolist()
    # Note that changing the seed here could result in the inclusion of the first sequence from the mock data.
    # assert not np.array_equal(first_src_sequence, first_ds_item)
    expected = "aff421"
    assert ds.fingerprint.endswith(
        expected
    ), f"Fingerprint mismatch, expected {expected}, got {ds.fingerprint[-6:]}...Do you need to update expected fingerprint?"
    assert first_ds_item == [56423, 24546, 15796, 52203]  # stable because we pass a seed
    assert ds.num_tokens == 10_112  # oversamples to handle rounding error
    assert len(ds) == 2528
    assert len(ds) / bsz >= math.ceil(max_tokens / (sequence_length * bsz))


@pytest.mark.parametrize(
    "mmap1_size, mmap2_size, expected_fingerprint",
    [
        (10 * 1000 + 7, 20 * 1000 + 1, "feabd0"),
        (10 * 1000, 20 * 1000, "9dc4c8"),
    ],
)
def test_numpy_fsl_mixture_dataset_with_repetition(
    tmp_path: Path, mmap1_size: int, mmap2_size: int, expected_fingerprint: str
):
    # NOTE: At small token counts the take_ratio can be finicky so we test at small but real world-ish scale
    npdtype = np.uint16
    seed = 42
    # Only 10k tokens in mmap1 so we have to upsample to meet 0.8 target below
    mmap1 = mk_mmaps(
        tmp_path=tmp_path,
        prefix="mmap1",
        num_files=1,
        size=mmap1_size,
        dtype=npdtype,
        eos=0,
        seed=72,
    )
    mmap2 = mk_mmaps(
        tmp_path=tmp_path,
        prefix="mmap2",
        num_files=1,
        size=mmap2_size,
        dtype=npdtype,
        eos=0,
        seed=27,
    )

    sequence_length = 4
    tokenizer = TokenizerConfig(
        vocab_size=32_000,
        eos_token_id=0,
        pad_token_id=-1,
    )

    source1_paths = [str(i[0]) for i in mmap1] * 2  # duplicate the paths

    bsz = 32
    max_tokens = 40_000

    mixture_config = SourceMixtureDatasetConfig(
        render_tables=False,
        requested_tokens=max_tokens,
        source_list=SourceMixtureList(
            [
                SourceMixtureConfig(
                    source_name="mmap1",
                    paths=source1_paths,
                    target_ratio=0.8,
                    max_repetition_ratio=2.0,
                ),
                SourceMixtureConfig(
                    source_name="mmap2",
                    paths=[str(i[0]) for i in mmap2],
                    target_ratio=0.2,
                ),
            ]
        ),
        seed=seed,
        global_batch_size=sequence_length * bsz,  # 10k sequences of length 4
    )

    ds = NumpyFSLDatasetConfig(
        source_mixture_config=mixture_config,
        sequence_length=sequence_length,
        tokenizer=tokenizer,
        dtype=NumpyDatasetDType.uint16,
        include_instance_metadata=False,
    ).build()
    ds.prepare()

    assert ds.fingerprint.endswith(
        expected_fingerprint
    ), f"Fingerprint mismatch, expected {expected_fingerprint}, got {ds.fingerprint[-6:]}...Do you need to update expected fingerprint?"

    first_ds_item = ds[0]["input_ids"].tolist()
    # NOTE: This is commented out until we fix behavior of the source mixture dataset
    # first_src_sequence = mmap1[0][1][:sequence_length].tolist()
    # Note that changing the seed here could result in the inclusion of the first sequence from the mock data.
    # assert not np.array_equal(first_src_sequence, first_ds_item)
    assert first_ds_item == [
        12761,
        6996,
        63252,
        65373,
    ]  # stable because we pass a seed
    assert ds.num_tokens == 40_064  # oversamples to handle rounding error
    assert len(ds) / bsz == math.ceil(max_tokens / (sequence_length * bsz))

    # Iterate through dataset to verify all instances have correct length
    for idx in range(len(ds)):
        instance = ds[idx]
        assert (
            len(instance["input_ids"]) == sequence_length
        ), f"Instance {idx} has incorrect length: {len(instance['input_ids'])} != {sequence_length}"


def write_data_file(data: List[int], path: Path, dtype, eos_token_id: int):
    path.parent.mkdir(exist_ok=True, parents=True)
    mmap = np.memmap(path, mode="w+", dtype=dtype, shape=(len(data),))
    mmap[:] = data
    mmap.flush()
    write_document_indices(path, dtype=dtype, eos_token_id=eos_token_id)


def test_numpy_vsl_dataset(tmp_path: Path):
    eos_token_id = 0
    pad_token_id = 0
    dtype = np.uint32

    # Write some fake data.
    data1 = [1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 0]
    data1_path = tmp_path / "data" / "part-1-00000.npy"
    write_data_file(data1, data1_path, dtype, eos_token_id)
    assert get_document_indices(data1_path) == [(0, 9), (9, len(data1))]

    data2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]
    data2_path = tmp_path / "data" / "part-2-00000.npy"
    write_data_file(data2, data2_path, dtype, eos_token_id)
    assert get_document_indices(data2_path) == [(0, len(data2))]

    ds = NumpyVSLDataset(
        data1_path,
        data2_path,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        vocab_size=32_000,
        max_sequence_length=8,
        min_sequence_length=2,
        dtype=dtype,
    )
    ds.work_dir = tmp_path
    ds.prepare()

    assert len(ds) == 5
    assert ds[0]["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert ds[1]["input_ids"].tolist() == [1, 2, 3, 4]
    assert ds[2]["input_ids"].tolist() == [5, 0]
    assert ds[3]["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert ds[4]["input_ids"].tolist() == [9, 10]

    assert ds.get_instance_lengths().tolist() == [8, 4, 2, 8, 2]
    buckets = ds.get_instance_buckets()
    assert len(buckets) == 3  # for each power of 2 from 2**1 = 2 through 2**3 = 8
    assert buckets[0][1].tolist() == [2, 4]  # instances of length 2
    assert buckets[1][1].tolist() == [1]  # instances of length 4
    assert buckets[2][1].tolist() == [0, 3]  # instances of length 8

    assert ds.instances_per_bucket == ((2, 2), (4, 1), (8, 2))


def test_numpy_interleaved_fsl_dataset(tmp_path: Path):
    data1 = [1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 0]
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(len(data1),))
    mmap1[:] = data1
    mmap1.flush()

    data2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 21, 22, 23, 24, 25, 0]
    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(len(data2),))
    mmap2[:] = data2
    mmap2.flush()

    ds = NumpyInterleavedFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=16,
        pad_token_id=-1,
        eos_token_id=0,
        vocab_size=32_000,
        seed=2,
        docs_per_instance=2,
        chunks_per_doc=4,
    )
    ds.work_dir = tmp_path
    ds.prepare()

    assert ds[0]["input_ids"].tolist() == [
        21,
        22,
        11,
        12,
        23,
        13,
        14,
        24,
        15,
        16,
        25,
        17,
        18,
        0,
        -1,
        -1,
    ]
    assert ds[0]["label_mask"].tolist() == [True] * 14 + [False] * 2
    assert ds[1]["input_ids"].tolist() == [1, 2, 8, 3, 4, 9, 5, 6, 10, 7, 0, -1, -1, -1, -1, -1]
    assert ds[1]["label_mask"].tolist() == [True] * 11 + [False] * 5
    assert len(ds) == 2


def test_numpy_interleaved_fsl_dataset_with_label_mask(tmp_path: Path):
    data1 = [1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 0]
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(len(data1),))
    mmap1[:] = data1
    mmap1.flush()

    data1_mask = [False, True, True, True, True, True, True, True] + [True, True, True, True]
    mmap1_mask = np.memmap(
        tmp_path / "mmap1_mask.npy", mode="w+", dtype=np.bool_, shape=(len(data1_mask),)
    )
    mmap1_mask[:] = data1_mask
    mmap1_mask.flush()

    data2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 21, 22, 23, 24, 25, 0]
    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(len(data2),))
    mmap2[:] = data2
    mmap2.flush()

    data2_mask = [True, True, True, True, True, True, True, True, True, True, True] + [
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    mmap2_mask = np.memmap(
        tmp_path / "mmap2_mask.npy", mode="w+", dtype=np.bool_, shape=(len(data2_mask),)
    )
    mmap2_mask[:] = data2_mask
    mmap2_mask.flush()

    ds = NumpyInterleavedFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=16,
        pad_token_id=-1,
        eos_token_id=0,
        vocab_size=32_000,
        seed=2,
        docs_per_instance=2,
        chunks_per_doc=4,
        label_mask_paths=[tmp_path / "mmap1_mask.npy", tmp_path / "mmap2_mask.npy"],
    )

    ds.work_dir = tmp_path
    ds.prepare()

    assert ds[0]["input_ids"].tolist() == [
        21,
        22,
        11,
        12,
        23,
        13,
        14,
        24,
        15,
        16,
        25,
        17,
        18,
        0,
        -1,
        -1,
    ]
    assert ds[0]["label_mask"].tolist() == [True] * 14 + [False] * 2
    assert ds[1]["input_ids"].tolist() == [1, 2, 8, 3, 4, 9, 5, 6, 10, 7, 0, -1, -1, -1, -1, -1]
    assert ds[1]["label_mask"].tolist() == [False] + [True] * 10 + [False] * 5
    assert len(ds) == 2


def test_numpy_interleaved_fsl_dataset_with_bos_token(tmp_path: Path):
    data1 = [99, 1, 2, 3, 4, 5, 6, 7, 0, 99, 8, 9, 10, 0]
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(len(data1),))
    mmap1[:] = data1
    mmap1.flush()

    data2 = [99, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 99, 21, 22, 23, 24, 25, 0]
    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(len(data2),))
    mmap2[:] = data2
    mmap2.flush()

    ds = NumpyInterleavedFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=16,
        pad_token_id=-1,
        eos_token_id=0,
        vocab_size=32_000,
        seed=2,
        docs_per_instance=2,
        chunks_per_doc=4,
        bos_token_id=99,
    )
    ds.work_dir = tmp_path
    ds.prepare()

    assert ds[0]["input_ids"].tolist() == [
        99,
        21,
        22,
        11,
        12,
        23,
        13,
        14,
        24,
        15,
        16,
        25,
        17,
        0,
        -1,
        -1,
    ]
    assert ds[0]["label_mask"].tolist() == [True] * 14 + [False] * 2
    assert ds[1]["input_ids"].tolist() == [99, 1, 2, 8, 3, 4, 9, 5, 6, 10, 7, 0, -1, -1, -1, -1]
    assert ds[1]["label_mask"].tolist() == [True] * 12 + [False] * 4
    assert len(ds) == 2


def test_numpy_interleaved_fsl_dataset_with_bos_token_and_label_mask(tmp_path: Path):
    data1 = [99, 1, 2, 3, 4, 5, 6, 7, 0, 99, 8, 9, 10, 0]
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(len(data1),))
    mmap1[:] = data1
    mmap1.flush()

    data1_mask = [True, False, True, True, True, True, True, True, True] + [
        True,
        True,
        True,
        True,
        True,
    ]
    mmap1_mask = np.memmap(
        tmp_path / "mmap1_mask.npy", mode="w+", dtype=np.bool_, shape=(len(data1_mask),)
    )
    mmap1_mask[:] = data1_mask
    mmap1_mask.flush()

    data2 = [99, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 99, 21, 22, 23, 24, 25, 0]
    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(len(data2),))
    mmap2[:] = data2
    mmap2.flush()

    data2_mask = [True, True, True, True, True, True, True, True, True, True, True, True] + [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    mmap2_mask = np.memmap(
        tmp_path / "mmap2_mask.npy", mode="w+", dtype=np.bool_, shape=(len(data2_mask),)
    )
    mmap2_mask[:] = data2_mask
    mmap2_mask.flush()

    ds = NumpyInterleavedFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=16,
        pad_token_id=-1,
        eos_token_id=0,
        vocab_size=32_000,
        seed=2,
        docs_per_instance=2,
        chunks_per_doc=4,
        label_mask_paths=[tmp_path / "mmap1_mask.npy", tmp_path / "mmap2_mask.npy"],
        bos_token_id=99,
    )

    ds.work_dir = tmp_path
    ds.prepare()

    assert ds[0]["input_ids"].tolist() == [
        99,
        21,
        22,
        11,
        12,
        23,
        13,
        14,
        24,
        15,
        16,
        25,
        17,
        0,
        -1,
        -1,
    ]
    assert ds[0]["label_mask"].tolist() == [True] * 14 + [False] * 2
    assert ds[1]["input_ids"].tolist() == [99, 1, 2, 8, 3, 4, 9, 5, 6, 10, 7, 0, -1, -1, -1, -1]
    assert ds[1]["label_mask"].tolist() == [True] + [False] + [True] * 10 + [False] * 4
    assert len(ds) == 2


def test_numpy_interleaved_fsl_dataset_with_interleaving_exempt_paths(tmp_path: Path):
    data1 = [1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 0]
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(len(data1),))
    mmap1[:] = data1
    mmap1.flush()

    data2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 21, 22, 23, 24, 25, 0]
    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(len(data2),))
    mmap2[:] = data2
    mmap2.flush()

    ds = NumpyInterleavedFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=16,
        pad_token_id=-1,
        eos_token_id=0,
        vocab_size=32_000,
        seed=3,
        docs_per_instance=2,
        chunks_per_doc=4,
        interleaving_exempt_paths=[tmp_path / "mmap1.npy"],
    )
    ds.work_dir = tmp_path
    ds.prepare()

    assert ds[0]["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 0] + [-1] * 8
    assert ds[1]["input_ids"].tolist() == [8, 9, 10, 0] + [-1] * 12

    assert ds[2]["input_ids"].tolist() == [
        21,
        22,
        11,
        12,
        23,
        13,
        14,
        24,
        15,
        16,
        25,
        17,
        18,
        0,
        -1,
        -1,
    ]
    assert ds[2]["label_mask"].tolist() == [True] * 14 + [False] * 2
    assert len(ds) == 3


def test_guess_dtype():
    config = NumpyFSLDatasetConfig(paths=[], sequence_length=1024, tokenizer=TokenizerConfig.gpt2())
    assert config.get_dtype() == np.uint16

    config = NumpyFSLDatasetConfig(
        paths=[], sequence_length=1024, tokenizer=TokenizerConfig.dolma2()
    )
    assert config.get_dtype() == np.uint32


def test_numpy_packed_fsl_dataset_use_array_if_local_not_served_from_stale_cache(tmp_path: Path):
    """Preparing with the inferred boundaries and then again with the metadata boundaries, in the
    same work_dir, must not serve the second run from the first run's packing caches."""
    # [1, 2, 3, 4] lost its EOS to truncation by its producer; then [5, 6, 0].
    data = [1, 2, 3, 4, 5, 6, 0]
    data_path = tmp_path / "mmap1.npy"
    mmap = np.memmap(data_path, mode="w+", dtype=np.uint16, shape=(len(data),))
    mmap[:] = data
    mmap.flush()
    with gzip.open(data_path.with_suffix(".csv.gz"), mode="wt") as f:
        f.write("0,4\n4,7\n")

    work_dir = tmp_path / "work"

    def prepare(use_array_if_local):
        ds = NumpyPackedFSLDataset(
            data_path,
            sequence_length=4,
            pad_token_id=-1,
            eos_token_id=0,
            vocab_size=32_000,
            use_array_if_local=use_array_if_local,
        )
        ds.work_dir = work_dir
        ds.prepare()
        return sum(int((ds[i]["label_mask"]).sum()) for i in range(len(ds)))

    # Inferred boundaries merge the two documents and truncate to 4, dropping the second.
    assert prepare(None) == 4
    # Same work_dir, so the caches from the run above are present. The metadata boundaries must
    # still be honored rather than the stale packing cache being reused.
    assert prepare(False) == len(data)


def test_numpy_packed_fsl_dataset_doc_lens_follow_metadata_boundaries(tmp_path: Path):
    """`doc_lens` drives the block-diagonal attention mask, so with the metadata boundaries it
    must not be re-derived by scanning for EOS: the document that lost its terminator would merge
    with the next one and attention would cross a real boundary."""
    # [1, 2, 3, 4] lost its EOS to truncation by its producer; then [5, 6, 7, 0].
    data = [1, 2, 3, 4, 5, 6, 7, 0]
    data_path = tmp_path / "mmap1.npy"
    mmap = np.memmap(data_path, mode="w+", dtype=np.uint16, shape=(len(data),))
    mmap[:] = data
    mmap.flush()
    with gzip.open(data_path.with_suffix(".csv.gz"), mode="wt") as f:
        f.write("0,4\n4,8\n")

    def doc_lens(use_array_if_local, sequence_length):
        ds = NumpyPackedFSLDataset(
            data_path,
            sequence_length=sequence_length,
            pad_token_id=-1,
            eos_token_id=0,
            vocab_size=32_000,
            generate_doc_lengths=True,
            use_array_if_local=use_array_if_local,
        )
        ds.work_dir = tmp_path / f"work-{use_array_if_local}-{sequence_length}"
        ds.prepare()
        return [ds[i]["doc_lens"].tolist() for i in range(len(ds))]

    # Both documents fill one instance exactly. The metadata boundaries keep them apart; the EOS
    # scan sees one 8-token span, so attention would cross the boundary.
    assert doc_lens(False, 8) == [[4, 4]]
    assert doc_lens(None, 8) == [[8]]
    # Trailing padding stays a final segment, matching `get_document_lengths`.
    assert doc_lens(False, 16) == [[4, 4, 8]]

    # A remote source reads the metadata boundaries whatever `use_array_if_local` says, so the
    # strategy has to follow the effective boundary source rather than the raw option.
    def packed_from_metadata(use_array_if_local, paths):
        ds = NumpyPackedFSLDataset(
            data_path,
            sequence_length=8,
            pad_token_id=-1,
            eos_token_id=0,
            vocab_size=32_000,
            use_array_if_local=use_array_if_local,
        )
        return ds._packed_from_metadata_boundaries(paths)

    assert packed_from_metadata(True, ["s3://bucket/mmap1.npy"]) is True
    assert packed_from_metadata(None, ["s3://bucket/mmap1.npy"]) is True
    assert packed_from_metadata(True, [data_path, "s3://bucket/other.npy"]) is True
    assert packed_from_metadata(True, [data_path]) is False
    assert packed_from_metadata(None, [data_path]) is False
    assert packed_from_metadata(False, [data_path]) is True


@pytest.mark.parametrize("pad_token_id", [-1, 0, 9])
@pytest.mark.parametrize("bos_token_id", [None, 0, 9])
def test_numpy_packed_fsl_dataset_doc_lens_padding_matches_get_document_lengths(
    tmp_path: Path, pad_token_id, bos_token_id
):
    """Preserve scanner padding segments, including EOS padding and BOS transitions."""
    bos = bos_token_id if bos_token_id is not None else 9
    data = [bos, 1, 2, 0, bos, 3, 4, 0]
    data_path = tmp_path / "mmap1.npy"
    mmap = np.memmap(data_path, mode="w+", dtype=np.uint16, shape=(len(data),))
    mmap[:] = data
    mmap.flush()
    with gzip.open(data_path.with_suffix(".csv.gz"), mode="wt") as f:
        f.write("0,4\n4,8\n")

    ds = NumpyPackedFSLDataset(
        data_path,
        sequence_length=16,
        pad_token_id=pad_token_id,
        eos_token_id=0,
        bos_token_id=bos_token_id,
        vocab_size=32_000,
        generate_doc_lengths=True,
        use_array_if_local=False,
    )
    ds.work_dir = tmp_path / "work"
    ds.prepare()
    item = ds[0]
    assert (
        item["doc_lens"].tolist()
        == get_document_lengths(
            item["input_ids"], eos_token_id=0, bos_token_id=bos_token_id
        ).tolist()
    )


@pytest.mark.parametrize("source_group_size", [1, 2])
def test_numpy_packed_fsl_dataset_metadata_correction_invalidates_cache(
    tmp_path, source_group_size
):
    paths = [tmp_path / f"mmap{i}.npy" for i in range(2)]
    for path in paths:
        np.array([1, 2, 3, 4, 5, 6, 7, 0], dtype=np.uint16).tofile(path)
        path.with_suffix(".csv.gz").write_bytes(
            gzip.compress(b"0,4\n4,8\n", compresslevel=0, mtime=0)
        )
    metadata_path = paths[0].with_suffix(".csv.gz")
    metadata_path.write_bytes(gzip.compress(b"0,3\n3,8\n", compresslevel=0, mtime=0))
    old_size = metadata_path.stat().st_size

    def prepare():
        ds = NumpyPackedFSLDatasetConfig(
            paths=[str(tmp_path / "mmap*.npy")],
            expand_glob=True,
            tokenizer=TokenizerConfig(vocab_size=32_000, eos_token_id=0, pad_token_id=-1),
            sequence_length=4,
            source_group_size=source_group_size,
            use_array_if_local=False,
            work_dir=str(tmp_path / "work"),
        ).build()
        assert isinstance(ds, NumpyPackedFSLDataset)
        ds.prepare()
        return ds

    before = prepare()
    fingerprint = before.fingerprint
    cache_paths = [
        before._get_document_indices_path(*before.paths[:source_group_size]),
        before._get_instance_offsets_path(*before.paths[:source_group_size]),
        before._get_docs_by_instance_path(*before.paths[:source_group_size]),
    ]
    assert sum(int(item["label_mask"].sum()) for item in before) == 15

    # Correct only the sidecar, keeping both its size and the token array unchanged.
    metadata_path.write_bytes(gzip.compress(b"0,4\n4,8\n", compresslevel=0, mtime=0))
    assert metadata_path.stat().st_size == old_size
    after = prepare()
    assert sum(int(item["label_mask"].sum()) for item in after) == 16
    assert after.fingerprint != fingerprint
    assert all(
        old != new
        for old, new in zip(
            cache_paths,
            [
                after._get_document_indices_path(*after.paths[:source_group_size]),
                after._get_instance_offsets_path(*after.paths[:source_group_size]),
                after._get_docs_by_instance_path(*after.paths[:source_group_size]),
            ],
        )
    )
    assert prepare().fingerprint == after.fingerprint


@pytest.mark.parametrize("use_array_if_local", [None, True, False])
def test_numpy_packed_fsl_dataset_metadata_cache_remote_and_mixed(
    tmp_path, monkeypatch, use_array_if_local
):
    local = tmp_path / "local.npy"
    remote = "https://example.com/remote.npy"
    sidecar = tmp_path / "remote.csv.gz"
    sidecar.write_bytes(gzip.compress(b"0,4\n4,8\n", mtime=0))
    reads = []

    def resource_path(folder, fname):
        reads.append((folder, fname))
        return sidecar

    monkeypatch.setattr(numpy_dataset, "resource_path", resource_path)
    monkeypatch.setattr(numpy_dataset, "get_file_size", lambda path: 16)

    def dataset():
        ds = NumpyPackedFSLDataset(
            local,
            remote,
            sequence_length=4,
            pad_token_id=-1,
            eos_token_id=0,
            vocab_size=32_000,
            source_group_size=2,
            use_array_if_local=use_array_if_local,
        )
        ds.work_dir = tmp_path / "work"
        return ds

    before = dataset()
    fingerprint = before.fingerprint
    cache = before._get_document_indices_path(local, remote)
    assert ("https://example.com", "remote.csv.gz") in reads
    assert ((str(tmp_path), "local.csv.gz") in reads) is (use_array_if_local is False)
    reads.clear()
    before._get_instance_offsets_path(local, remote)
    before._get_docs_by_instance_path(local, remote)
    assert not reads  # Hashes are reused; indexing must not fetch the sidecars again.
    sidecar.write_bytes(gzip.compress(b"0,3\n3,8\n", mtime=0))
    after = dataset()
    assert after.fingerprint != fingerprint
    assert after._get_document_indices_path(local, remote) != cache


def test_numpy_fsl_mixture_sizes_shared_index_for_largest_duplicate(tmp_path: Path):
    # A path duplicated in a mixture shares one indices file, but each occurrence keeps its own
    # token allocation in `path_offset_index`. If a later occurrence has a larger allocation than
    # the first, the shared file must be sized for the larger one, or reads for that occurrence's
    # tail run past the end of the generated file.
    npdtype = np.uint16
    seq_len = 4
    ((path, _),) = mk_mmaps(tmp_path, "dup", 1, 20 * 1000, npdtype, eos=0, seq_length=seq_len)

    small_tokens = 10 * seq_len  # occurrence idx=0
    large_tokens = 40 * seq_len  # occurrence idx=1 (the larger, later duplicate)

    ds = NumpyFSLDatasetMixture(
        path,
        path,
        path_offset_index={(str(path), 0): small_tokens, (str(path), 1): large_tokens},
        seed=42,
        sequence_length=seq_len,
        pad_token_id=-1,
        eos_token_id=0,
        vocab_size=32_000,
        dtype=npdtype,
    )
    ds.work_dir = tmp_path
    ds.prepare()

    # The shared index file holds enough instances for the larger occurrence.
    item_size = ds.indices_dtype(0).itemsize
    file_instances = get_file_size(ds._get_instance_indices_path(path)) // (item_size * 2)
    assert file_instances >= large_tokens // seq_len

    # The last global instance falls in the larger occurrence and must be readable (out of bounds
    # before the fix, which sized the file from the first occurrence's smaller allocation).
    assert len(ds) == small_tokens // seq_len + large_tokens // seq_len
    assert len(ds[len(ds) - 1]["input_ids"]) == seq_len
