"""CPU tests for whole-batch prefetching on a background thread: identical batch order, a
checkpoint that describes consumed batches only, error propagation and thread shutdown."""

import copy
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from olmo_core.data.multimodal import mixture_data_loader
from olmo_core.data.multimodal.batch_prefetch import BatchPrefetcher, BatchPrefetchStats
from olmo_core.data.multimodal.collator import MultimodalCollator
from olmo_core.data.multimodal.mixture_data_loader import (
    MixtureDataLoader,
    MixtureDataLoaderConfig,
)
from olmo_core.exceptions import OLMoConfigurationError

THREAD_NAME = "mixture-batch-prefetch"


class _Dataset:
    def __init__(self, tag, *, fail_at=None):
        self.tag = tag
        self.fail_at = fail_at
        self.content_fingerprint = f"batch-prefetch-{tag}"

    def __len__(self):
        return 29

    def get(self, index, epoch):
        if self.fail_at is not None and index == self.fail_at:
            raise ValueError(f"synthetic failure at {self.tag}[{index}]")
        size = 5 + index % 7
        crops = index % 3
        tag = self.tag + index + 1000 * epoch
        tokens = np.full(size, tag, dtype=np.int64)
        return {
            "input_ids": tokens,
            "labels": tokens + 1,
            "loss_masks": np.asarray([0.0] + [0.5] * (size - 1), dtype=np.float32),
            "position_ids": np.arange(size, dtype=np.int64),
            "token_type_ids": np.asarray([1] * crops + [0] * (size - crops), dtype=np.int64),
            "subsegment_ids": np.arange(size, dtype=np.int64) // 3,
            "images": np.full((crops, 4, 3), tag / 2.0, dtype=np.float32),
            "pooled_patches_idx": np.arange(crops * 2, dtype=np.int64).reshape(crops, 2),
        }

    def __getitem__(self, index):
        return self.get(index, 0)


def _loader(
    path,
    *,
    depth=0,
    workers=0,
    grouped=False,
    continuous=True,
    rank=0,
    fail_at=None,
    max_errors=0,
):
    return MixtureDataLoader(
        [_Dataset(100, fail_at=fail_at), _Dataset(200)],
        [0.4, 0.6],
        MultimodalCollator(pad_token_id=0, pad_sequence_length=64, batch_metadata=True),
        work_dir=path,
        global_batch_size=8 * 64,
        seed=31,
        epoch_instances=2400,
        pack=True,
        pack_max_crops=8,
        pack_buffer_size=4,
        pack_image_weight=30.0,
        continuous_stream=continuous,
        prefetch_workers=workers,
        batch_prefetch_depth=depth,
        max_consecutive_data_errors=max_errors,
        max_total_data_errors=max_errors,
        dp_world_size=2,
        dp_rank=rank,
        dataset_names=["caption", "transcript"],
        source_groups={"caption": "image", "transcript": "text"} if grouped else None,
        group_sequence_quotas={"image": 4, "text": 4} if grouped else None,
    )


def _assert_equal(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, np.ndarray):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_equal(left, right)
    else:
        assert actual == expected


def _take(loader, n, *, epoch=1):
    """Consume ``n`` batches from a fresh iterator and return them with the loader's state."""
    loader.reshuffle(epoch=epoch)
    iterator = iter(loader)
    try:
        batches = [next(iterator) for _ in range(n)]
        state = copy.deepcopy(loader.state_dict())
    finally:
        iterator.close()
    return batches, state


def _no_prefetch_threads():
    return not any(t.name == THREAD_NAME and t.is_alive() for t in threading.enumerate())


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("workers", [0, 4])
@pytest.mark.parametrize("depth", [1, 2, 5])
def test_prefetched_batches_match_the_synchronous_order(tmp_path, grouped, workers, depth):
    """The yielded sequence with batch prefetching is the thread-free (no example threads, no
    batch thread) sequence, batch for batch and field for field."""
    expected, expected_state = _take(_loader(tmp_path, grouped=grouped), 12)
    loader = _loader(tmp_path, depth=depth, workers=workers, grouped=grouped)
    actual, actual_state = _take(loader, 12)
    _assert_equal(actual, expected)
    _assert_equal(actual_state, expected_state)
    assert loader.batch_prefetch_stats.batches_consumed == 12
    assert loader.batch_prefetch_stats.batches_produced >= 12
    assert _no_prefetch_threads()


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("consumed", [1, 3, 7])
def test_checkpoint_after_consumed_batches_resumes_the_thread_free_stream(
    tmp_path, grouped, consumed
):
    """Save after ``consumed`` batches with prefetching on (the producer has run ahead), restore
    into a fresh loader with prefetching off, and the next batches are the thread-free loader's
    batches ``consumed + 1, ...``; and the other way round."""
    reference, _ = _take(_loader(tmp_path, grouped=grouped), consumed + 6)
    for save_depth, resume_depth in [(3, 0), (0, 3), (2, 2)]:
        saver = _loader(tmp_path, depth=save_depth, grouped=grouped)
        _, state = _take(saver, consumed)
        assert state["batches_processed"] == consumed
        resumed = _loader(tmp_path, depth=resume_depth, grouped=grouped)
        resumed.load_state_dict(state)
        following, _ = _take(resumed, 6)
        _assert_equal(following, reference[consumed:])
    assert _no_prefetch_threads()


def test_live_state_describes_consumed_batches_only(tmp_path):
    """While the iterator is live and the queue is full, ``state_dict`` is the synchronous
    loader's state after exactly the consumed batches, and closing the iterator keeps it."""
    sync_states = []
    sync = _loader(tmp_path)
    sync.reshuffle(epoch=1)
    sync_iterator = iter(sync)
    sync_states.append(copy.deepcopy(sync.state_dict()))
    for _ in range(4):
        next(sync_iterator)
        sync_states.append(copy.deepcopy(sync.state_dict()))
    sync_iterator.close()

    loader = _loader(tmp_path, depth=3)
    loader.reshuffle(epoch=1)
    iterator = iter(loader)
    _assert_equal(loader.state_dict(), sync_states[0])
    for k in range(1, 5):
        next(iterator)
        # Let the producer fill its queue so its own state is well ahead of the consumer's.
        deadline = threading.Event()
        while loader.batch_prefetch_stats.batches_produced < k + 3 and not deadline.wait(0.01):
            pass
        assert loader.batch_prefetch_stats.batches_produced >= k + 3
        _assert_equal(loader.state_dict(), sync_states[k])
        assert loader.state_dict()["batches_processed"] == k
    iterator.close()
    assert _no_prefetch_threads()
    _assert_equal(loader.state_dict(), sync_states[4])
    _assert_equal(sync.state_dict(), sync_states[4])


def test_producer_errors_reach_the_consumer_and_the_thread_stops(tmp_path):
    reference_error = None
    sync = _loader(tmp_path, fail_at=3)
    sync.reshuffle(epoch=1)
    sync_iterator = iter(sync)
    sync_batches = []
    try:
        for _ in range(40):
            sync_batches.append(next(sync_iterator))
    except ValueError as error:
        reference_error = error
    finally:
        sync_iterator.close()
    assert reference_error is not None and sync_batches

    loader = _loader(tmp_path, depth=2, fail_at=3)
    loader.reshuffle(epoch=1)
    iterator = iter(loader)
    batches = []
    with pytest.raises(ValueError, match="synthetic failure"):
        for _ in range(40):
            batches.append(next(iterator))
    iterator.close()
    _assert_equal(batches, sync_batches)
    assert _no_prefetch_threads()


def test_closing_the_iterator_shuts_the_thread_down_and_depth_zero_uses_no_thread(
    tmp_path, monkeypatch
):
    created = []
    native = mixture_data_loader.BatchPrefetcher

    def record(*args, **kwargs):
        prefetcher = native(*args, **kwargs)
        created.append(prefetcher)
        return prefetcher

    monkeypatch.setattr(mixture_data_loader, "BatchPrefetcher", record)
    loader = _loader(tmp_path, depth=0)
    _take(loader, 2)
    assert created == [] and _no_prefetch_threads()

    loader = _loader(tmp_path, depth=2)
    loader.reshuffle(epoch=1)
    iterator = iter(loader)
    next(iterator)
    assert len(created) == 1 and not _no_prefetch_threads()
    iterator.close()
    assert _no_prefetch_threads()
    # The same loader can start a new iterator (a new epoch, or the trainer's next fit).
    _take(loader, 2)
    assert len(created) == 2 and _no_prefetch_threads()


@pytest.mark.parametrize("depth", [2])
def test_epoch_mode_sequence_and_exhaustion_match(tmp_path, depth):
    """The non-continuous (epoch-shuffled) mode ends its epoch on the producer: the consumer
    sees the same batches and then ``StopIteration``."""
    sync = _loader(tmp_path, continuous=False)
    sync.reshuffle(epoch=1)
    expected = list(sync)
    assert 0 < len(expected) == sync.total_batches
    loader = _loader(tmp_path, depth=depth, continuous=False)
    loader.reshuffle(epoch=1)
    actual = list(loader)
    _assert_equal(actual, expected)
    assert loader.batches_processed == sync.batches_processed
    assert _no_prefetch_threads()


def test_batch_prefetcher_unit_behaviour():
    stats = BatchPrefetchStats()
    counter = iter(range(100))

    def make():
        for value in counter:
            yield value

    prefetcher = BatchPrefetcher(make, lambda: "state", depth=2, stats=stats)
    assert next(prefetcher) == (0, "state")
    assert next(prefetcher) == (1, "state")
    prefetcher.close()
    prefetcher.close()
    assert stats.batches_consumed == 2 and 2 <= stats.batches_produced <= 5
    assert next(counter) == stats.batches_produced
    with pytest.raises(StopIteration):
        next(prefetcher)

    def finite():
        yield 1

    prefetcher = BatchPrefetcher(finite, lambda: None, depth=1)
    assert list(prefetcher) == [(1, None)]
    prefetcher.close()
    with pytest.raises(ValueError, match="depth"):
        BatchPrefetcher(finite, lambda: None, depth=0)


@pytest.mark.parametrize("depth", [-1, True, "2"])
def test_invalid_batch_prefetch_depth_rejected(tmp_path, depth):
    with pytest.raises(OLMoConfigurationError, match="batch_prefetch_depth"):
        _loader(tmp_path, depth=depth)


def test_batch_prefetch_depth_config_default_off_and_round_trip(tmp_path):
    config = MixtureDataLoaderConfig(global_batch_size=128, sequence_length=16, work_dir="x")
    assert config.batch_prefetch_depth == 0
    config = MixtureDataLoaderConfig(
        global_batch_size=128, sequence_length=16, work_dir=str(tmp_path), batch_prefetch_depth=2
    )
    restored = MixtureDataLoaderConfig.from_dict(config.as_config_dict())
    assert restored == config
    dataset: Any = SimpleNamespace(
        datasets=[_Dataset(100)],
        weights=[1.0],
        names=["caption"],
        tokenizer=SimpleNamespace(pad_token_id=0),
    )
    loader = restored.build(dataset)
    assert loader.batch_prefetch_depth == 2
    # Group children are driven by the parent's thread; they never start one of their own.
    grouped = _loader(tmp_path, depth=2, grouped=True)
    assert all(child.batch_prefetch_depth == 0 for child in grouped._group_loaders.values())
