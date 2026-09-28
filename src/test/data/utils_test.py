import gzip
from collections import namedtuple

import numpy as np
import pytest
import torch

from olmo_core.data.utils import (
    InstancePacker,
    SegmentTree,
    attention_mask_to_cache_leftpad,
    bucket_documents,
    get_cumulative_document_lengths,
    get_document_lengths,
    iter_batched,
    iter_document_indices,
    melt_batch,
    pack_documents_into_instances,
    segment_documents_into_instances,
    write_document_indices,
)


@pytest.mark.limit_memory("265 KB")
def test_segment_documents_into_instances(tmp_path):
    data = [1, 2, 3, 4, 0, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 0] * 1000
    data_path = tmp_path / "data.npy"
    max_sequence_length = 4
    mmap = np.memmap(data_path, mode="w+", dtype=np.uint16, shape=(len(data),))
    indices_path = tmp_path / "indices.npy"
    mmap[:] = data
    mmap.flush()

    eos = 0
    dtype = np.uint16
    sample = (2, 42)

    results = []
    for _ in range(10):
        results.append(
            segment_documents_into_instances(
                path=data_path,
                target=indices_path,
                max_sequence_length=max_sequence_length,
                eos_token_id=eos,
                dtype=dtype,
                sample=sample,
            )
        )

    assert all([r[1] == 2 for r in results])


def test_iter_document_indices(tmp_path):
    data = [1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 0]
    data_path = tmp_path / "data.npy"
    mmap = np.memmap(data_path, mode="w+", dtype=np.uint16, shape=(len(data),))
    mmap[:] = data
    mmap.flush()
    write_document_indices(data_path, dtype=np.uint16, eos_token_id=0)
    assert list(
        iter_document_indices(data_path, eos_token_id=0, dtype=np.uint16, use_array_if_local=False)
    ) == [(0, 9), (9, len(data))]
    assert list(
        iter_document_indices(data_path, eos_token_id=0, dtype=np.uint16, use_array_if_local=True)
    ) == [(0, 9), (9, len(data))]


@pytest.mark.parametrize("bos_token_id, eos_token_id", [(98, 99), (99, 99)])
def test_iter_document_indices_with_bos_token_id(tmp_path, bos_token_id: int, eos_token_id: int):
    data = [bos_token_id, 2, 3, 4, 5, 6, 7, 8, eos_token_id, bos_token_id, 2, 3, 4, 5, eos_token_id]
    data_path = tmp_path / "data.npy"
    mmap = np.memmap(data_path, mode="w+", dtype=np.uint16, shape=(len(data),))
    mmap[:] = data
    mmap.flush()
    assert list(
        iter_document_indices(
            data_path,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            dtype=np.uint16,
            use_array_if_local=True,
        )
    ) == [(0, 9), (9, len(data))]


def test_melt_batch():
    batch = {
        "input_ids": torch.randint(0, 32, (2, 12)),
        "index": torch.tensor([0, 1]),
        "metadata": [{"path": "x"}, {"path": "y"}],
    }
    new_batch = melt_batch(batch, 4)
    assert new_batch.keys() == batch.keys()
    assert new_batch["input_ids"].shape == (6, 4)
    assert new_batch["index"].tolist() == [0, 0, 0, 1, 1, 1]
    assert new_batch["metadata"] == [{"path": "x"}] * 3 + [{"path": "y"}] * 3


def test_get_document_lengths():
    eos_token_id = 50279

    # Should work when the instance starts with EOS token.
    assert get_document_lengths(
        torch.tensor([eos_token_id, 3, 4, 5, 5, eos_token_id, 6, 5, eos_token_id, 3, 5]),
        eos_token_id=eos_token_id,
    ).tolist() == [1, 5, 3, 2]

    # Should work when the instance ends with EOS token.
    assert get_document_lengths(
        torch.tensor([3, 4, 5, 5, eos_token_id, 6, 5, eos_token_id, 3, 5, eos_token_id]),
        eos_token_id=eos_token_id,
    ).tolist() == [5, 3, 3]

    # Should work when the instance BOS token is provided.
    bos_token_id = 50278
    assert get_document_lengths(
        torch.tensor(
            [
                bos_token_id,
                3,
                4,
                5,
                5,
                eos_token_id,
                bos_token_id,
                6,
                5,
                eos_token_id,
                bos_token_id,
                3,
                5,
                eos_token_id,
            ]
        ),
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
    ).tolist() == [6, 4, 4]

    # Should work when the instance ends with BOS token.
    bos_token_id = 50278
    assert get_document_lengths(
        torch.tensor(
            [bos_token_id, 3, 4, 5, 5, eos_token_id, bos_token_id, 6, 5, eos_token_id, bos_token_id]
        ),
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
    ).tolist() == [6, 4, 1]

    # Should work when the instance BOS token is provided and is same as EOS token.
    bos_token_id = eos_token_id
    assert get_document_lengths(
        torch.tensor(
            [
                bos_token_id,
                3,
                4,
                5,
                5,
                eos_token_id,
                bos_token_id,
                6,
                5,
                eos_token_id,
                bos_token_id,
                3,
                5,
                eos_token_id,
            ]
        ),
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
    ).tolist() == [6, 4, 4]


def test_get_cumulative_document_lengths():
    assert get_cumulative_document_lengths(
        torch.tensor(
            [
                [1, 5, 3, 2, 0],
                [5, 3, 3, 0, 0],
            ],
            dtype=torch.int32,
        )
    ).tolist() == [0, 1, 6, 9, 11, 16, 19, 22]


def test_iter_batched():
    instances = [
        {"input_ids": torch.tensor([1, 1, 1, 1])},
        {"input_ids": torch.tensor([2, 2, 2, 2])},
        {"input_ids": torch.tensor([3, 3])},
        {"input_ids": torch.tensor([4, 4])},
        {"input_ids": torch.tensor([5, 5])},
        {"input_ids": torch.tensor([6, 6])},
        {"input_ids": torch.tensor([7, 7])},
    ]
    batches = list(iter_batched(instances, 8))
    assert len(batches[0]) == 2
    assert len(batches[1]) == 4
    assert len(batches[2]) == 1


def test_bucket_documents(tmp_path):
    buckets = [2, 4]
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 0, 1, 2, 0])

    data_path = tmp_path / "data.npy"
    mmap = np.memmap(data_path, dtype=np.uint16, mode="w+", shape=data.shape)
    mmap[:] = data
    mmap.flush()

    write_document_indices(data_path, dtype=np.uint16, eos_token_id=0)

    n_og_docs, n_new_docs = bucket_documents(
        data_path, tmp_path / "buckets.npy", buckets=buckets, eos_token_id=0, dtype=np.uint16
    )
    assert n_og_docs == 3
    assert n_new_docs == 6

    buckets = (
        np.memmap(tmp_path / "buckets.npy", mode="r", dtype=np.uint32).reshape((-1, 2)).tolist()
    )
    assert buckets == [[0, 4], [4, 8], [8, 12], [13, 17], [17, 19], [19, 21]]


def test_segment_tree():
    seg_tree = SegmentTree(8)
    assert seg_tree.root_node.weight == 8

    leaf = seg_tree.query(8)  # leaf_id=7, weight=8
    assert leaf.leaf_id == 7
    assert leaf.weight == 8

    seg_tree.leaf_nodes[1].update(2)
    seg_tree.leaf_nodes[3].update(4)

    leaf = seg_tree.query(3)  # leaf_id=3, weight=4
    assert leaf.leaf_id == 3
    assert leaf.weight == 4


def test_instance_packer():
    # Follows the example from appendix (B) in https://arxiv.org/pdf/2404.10830
    packer = InstancePacker(8)
    assert packer._pack_document(0, 8) == 0
    assert packer._pack_document(1, 6) == 1
    assert packer._pack_document(2, 6) == 2
    assert packer._pack_document(3, 4) == 3
    assert packer._pack_document(4, 3) == 3

    # And here we extend the example...
    assert packer._pack_document(5, 2) == 1
    assert packer._pack_document(6, 3) == 4


def test_pack_documents_into_instances(tmp_path):
    data = [1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 1, 2, 0]
    data_path = tmp_path / "data.npy"
    mmap = np.memmap(data_path, dtype=np.uint16, mode="w+", shape=(len(data),))
    mmap[:] = data
    mmap.flush()

    instances, document_indices, total_tokens = pack_documents_into_instances(
        data_path, max_sequence_length=8, eos_token_id=0, dtype=np.uint16
    )
    assert instances == [[0], [1], [2], [3, 4]]
    assert document_indices[0].tolist() == [0, 8]
    assert total_tokens == len(data)


AttentionMaskTestCase = namedtuple(
    "AttentionMaskTestCase", ["shape", "valid_starts", "expected_seq_lens"]
)


@pytest.mark.parametrize(
    "test_case",
    [
        # Mixed padding lengths
        AttentionMaskTestCase(shape=(2, 10), valid_starts=[3, 5], expected_seq_lens=[7, 5]),
        # No padding (all valid tokens)
        AttentionMaskTestCase(shape=(2, 8), valid_starts=[0, 0], expected_seq_lens=[8, 8]),
        # Different sequence lengths with padding
        AttentionMaskTestCase(shape=(3, 12), valid_starts=[2, 4, 7], expected_seq_lens=[10, 8, 5]),
        # Single sequence
        AttentionMaskTestCase(shape=(1, 15), valid_starts=[5], expected_seq_lens=[10]),
        # All padding except last token
        AttentionMaskTestCase(shape=(2, 20), valid_starts=[19, 18], expected_seq_lens=[1, 2]),
    ],
)
def test_attention_mask_to_cache_leftpad_valid_cases(test_case):
    batch_size, seq_len = test_case.shape
    valid_starts = test_case.valid_starts
    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    for i, start in enumerate(valid_starts):
        attention_mask[i, start:] = True

    cache_leftpad = attention_mask_to_cache_leftpad(attention_mask)

    assert cache_leftpad.tolist() == valid_starts


@pytest.mark.parametrize(
    "invalid_mask",
    [
        torch.tensor([[1, 0, 1, 1, 1], [1, 1, 0, 1, 1]], dtype=torch.bool),
        torch.tensor([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]], dtype=torch.bool),
        torch.tensor([[1, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool),
    ],
)
def test_attention_mask_to_cache_leftpad_invalid_masks(invalid_mask):
    with pytest.raises(ValueError, match="prefix padding"):
        attention_mask_to_cache_leftpad(invalid_mask)


def _write_metadata(data_path, boundaries):
    """Write a boundaries metadata file directly, as a producer that knows its own row
    boundaries would, rather than deriving them by scanning for EOS."""
    metadata_path = data_path.with_suffix(".csv.gz")
    with gzip.open(metadata_path, mode="wt") as f:
        for start_idx, end_idx in boundaries:
            f.write(f"{start_idx},{end_idx}\n")
    return metadata_path


def test_iter_document_indices_non_eos_terminated_document(tmp_path):
    """A document that is not EOS-terminated merges with the next one when boundaries are
    inferred from the token array. Reading the metadata file instead recovers the truth."""
    # Two documents: [1, 2, 3] was truncated by its producer and lost its EOS, then [4, 5, 6, 0].
    data = [1, 2, 3, 4, 5, 6, 0]
    data_path = tmp_path / "data.npy"
    mmap = np.memmap(data_path, mode="w+", dtype=np.uint16, shape=(len(data),))
    mmap[:] = data
    mmap.flush()
    _write_metadata(data_path, [(0, 3), (3, 7)])

    # Inferred from the array, the un-terminated document swallows the one after it.
    assert list(
        iter_document_indices(data_path, eos_token_id=0, dtype=np.uint16, use_array_if_local=True)
    ) == [(0, 7)]
    # The metadata file has the real boundaries.
    assert list(
        iter_document_indices(data_path, eos_token_id=0, dtype=np.uint16, use_array_if_local=False)
    ) == [(0, 3), (3, 7)]


def test_pack_documents_into_instances_use_array_if_local(tmp_path):
    """Capping document length on top of a merged span silently discards the tokens past the
    cap, so the second document never reaches training. Reading the metadata keeps them."""
    # [1, 2, 3, 4] was truncated to the cap by its producer and lost its EOS; then [5, 6, 0].
    data = [1, 2, 3, 4, 5, 6, 0]
    data_path = tmp_path / "data.npy"
    mmap = np.memmap(data_path, mode="w+", dtype=np.uint16, shape=(len(data),))
    mmap[:] = data
    mmap.flush()
    _write_metadata(data_path, [(0, 4), (4, 7)])

    def total_tokens_packed(use_array_if_local):
        _, _, total_tokens = pack_documents_into_instances(
            data_path,
            max_sequence_length=4,  # the packer requires a power of two
            eos_token_id=0,
            dtype=np.uint16,
            use_array_if_local=use_array_if_local,
        )
        return total_tokens

    # Merged into one 7-token span, truncated to 4: the whole second document is dropped.
    assert total_tokens_packed(True) == 4
    # Real boundaries: both documents survive.
    assert total_tokens_packed(False) == len(data)


def test_segment_documents_into_instances_use_array_if_local(tmp_path):
    data = [1, 2, 3, 4, 5, 6, 0]
    data_path = tmp_path / "data.npy"
    mmap = np.memmap(data_path, mode="w+", dtype=np.uint16, shape=(len(data),))
    mmap[:] = data
    mmap.flush()
    _write_metadata(data_path, [(0, 3), (3, 7)])

    def num_docs(use_array_if_local):
        total_og_docs, _ = segment_documents_into_instances(
            data_path,
            tmp_path / f"indices-{use_array_if_local}.npy",
            max_sequence_length=3,
            eos_token_id=0,
            dtype=np.uint16,
            use_array_if_local=use_array_if_local,
        )
        return total_og_docs

    assert num_docs(True) == 1
    assert num_docs(False) == 2
