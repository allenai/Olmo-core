import numpy as np
import pytest

from olmo_core.data.composable.utils import (
    SEED_NOT_SET,
    build_global_indices,
    calculate_sample_sizes,
    reset_composable_seed,
    resolve_seed,
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


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64])
def test_build_global_indices(dtype, seed: int = 42):
    total_tokens = 24
    max_sequence_length = 8

    sequence_length = 4
    indices1 = build_global_indices(
        total_tokens // sequence_length,
        sequence_length=sequence_length,
        max_sequence_length=max_sequence_length,
        seed=seed,
        dtype=dtype,
    )
    assert indices1.dtype == dtype
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
        dtype=dtype,
    )
    assert indices2.dtype == dtype
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
