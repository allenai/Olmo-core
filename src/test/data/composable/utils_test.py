import numpy as np

from olmo_core.data.composable.utils import (
    SEED_NOT_SET,
    build_global_indices,
    calculate_sample_sizes,
    reset_composable_seed,
    resolve_seed,
    sample_global_indices,
    set_composable_seed,
)


def test_calculate_sample_sizes():
    # Case 1: imbalanced sources but we want a balanced ratio without oversampling, so we
    # under-sample from the larger of the two sources.
    assert calculate_sample_sizes(
        [8, 16],
        [0.5, 0.5],
        [1.0, 1.0],
    ).tolist() == [8, 8]

    # Case 2: a similar situation but with three sources and a different target ratio.
    assert calculate_sample_sizes(
        [8, 8, 16],
        [0.3, 0.2, 0.5],
        [1.0, 1.0, 1.0],
    ).tolist() == [8, 5, 13]

    # Case 3: imbalanced sources but we want a balanced ratio with oversampling allowed, so we
    # "meet in the middle" by oversampling the smaller of the two sources while undersampling the
    # larger one to maintain the same number of tokens that we started with.
    assert calculate_sample_sizes(
        [8, 16],
        [0.5, 0.5],
        [2.0, 1.0],
    ).tolist() == [12, 12]

    # Case 4: a similar situation but this time our target size is small enough that we don't need
    # to repeat any tokens to meet it.
    assert calculate_sample_sizes(
        [8, 16],
        [0.5, 0.5],
        [1.0, 1.0],
        target_size=16,
    ).tolist() == [8, 8]


def test_build_global_indices(seed: int = 42):
    total_tokens = 24
    max_sequence_length = 8

    sequence_length = 4
    indices1 = build_global_indices(
        total_tokens // sequence_length,
        sequence_length=sequence_length,
        max_sequence_length=max_sequence_length,
        seed=seed,
    )
    assert indices1.size == 6
    assert indices1[0:2].tolist() in [[0, 1], [2, 3], [4, 5]]
    assert indices1[2:4].tolist() in [[0, 1], [2, 3], [4, 5]]
    assert indices1[4:6].tolist() in [[0, 1], [2, 3], [4, 5]]

    sequence_length = 8
    indices2 = build_global_indices(
        total_tokens // sequence_length,
        sequence_length=sequence_length,
        max_sequence_length=max_sequence_length,
        seed=seed,
    )
    assert (np.repeat(indices2, 2) == indices1 // 2).all()


def test_set_and_resolve_seed():
    try:
        # Seeds explicitly set to None should always resolve to None.
        assert resolve_seed(None) is None

        # Before 'set_seed()' has been called, 'SEED_NOT_SET' should always resolve to a concrete 0.
        assert resolve_seed(SEED_NOT_SET) is not SEED_NOT_SET
        assert resolve_seed(SEED_NOT_SET) == 0

        # Now set the set.
        set_composable_seed(1234)

        # Seeds explicitly set to None should always resolve to None.
        assert resolve_seed(None) is None

        # After 'set_seed()' has been called, 'SEED_NOT_SET' should resolve to a new seed.
        seed = resolve_seed(SEED_NOT_SET)
        assert isinstance(seed, int)
        assert seed is not SEED_NOT_SET

        # And we should get something different each time.
        seed2 = resolve_seed(SEED_NOT_SET)
        assert isinstance(seed2, int)
        assert seed2 != seed
    finally:
        reset_composable_seed()


def test_sample_global_indices():
    # One instance per chunk: a random subset of distinct indices.
    indices = sample_global_indices(
        1_000_000, 500, sequence_length=8, max_sequence_length=8, seed=0
    )
    assert indices.dtype == np.uint32
    assert len(indices) == len(set(indices.tolist())) == 500
    assert indices.max() < 1_000_000
    assert indices.max() >= 500  # not just the first instances

    # Several instances per chunk: chunks stay whole and contiguous.
    indices = sample_global_indices(1000, 40, sequence_length=4, max_sequence_length=16, seed=0)
    chunks = indices.reshape(-1, 4)
    assert (chunks[:, 0] % 4 == 0).all()
    assert (np.diff(chunks, axis=1) == 1).all()
    assert len(set(chunks[:, 0].tolist())) == 10

    # Taking every instance is a permutation.
    indices = sample_global_indices(64, 64, sequence_length=8, max_sequence_length=16, seed=1)
    assert sorted(indices.tolist()) == list(range(64))
