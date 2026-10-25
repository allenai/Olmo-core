"""CPU tests for prepared-example read-ahead configuration and exact loader resume."""

import copy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from olmo_core.data.multimodal import mixture_data_loader
from olmo_core.data.multimodal.collator import MultimodalCollator
from olmo_core.data.multimodal.mixture_data_loader import (
    MixtureDataLoader,
    MixtureDataLoaderConfig,
)
from olmo_core.exceptions import OLMoConfigurationError


class _Dataset:
    def __init__(self, tag):
        self.tag = tag
        self.content_fingerprint = f"prefetch-config-{tag}"

    def __len__(self):
        return 31

    def get(self, index, epoch):
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


def _loader(path, *, depth=None, workers=8, grouped=False, buffer_size=4, rank=0):
    return MixtureDataLoader(
        [_Dataset(100), _Dataset(200)],
        [0.4, 0.6],
        MultimodalCollator(pad_token_id=0, pad_sequence_length=64),
        work_dir=path,
        global_batch_size=8 * 64,
        seed=31,
        epoch_instances=2400,
        pack=True,
        pack_max_crops=8,
        pack_buffer_size=buffer_size,
        pack_image_weight=30.0,
        continuous_stream=buffer_size > 0,
        prefetch_workers=workers,
        prefetch_max_in_flight=depth,
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


@pytest.mark.parametrize("depth", [None, 64])
def test_prefetch_config_round_trip_and_build(tmp_path, depth):
    config = MixtureDataLoaderConfig(
        global_batch_size=128,
        sequence_length=16,
        work_dir=str(tmp_path),
        prefetch_workers=4,
        prefetch_max_in_flight=depth,
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
    assert loader.prefetch_workers == 4
    assert loader.prefetch_max_in_flight == depth
    legacy = config.as_config_dict()
    if depth is None:
        assert "prefetch_max_in_flight" not in legacy
    else:
        assert legacy.pop("prefetch_max_in_flight") == depth
    assert MixtureDataLoaderConfig.from_dict(legacy).prefetch_max_in_flight is None


@pytest.mark.parametrize("depth", [0, "64"])
def test_invalid_prefetch_depth_rejected_even_without_workers(tmp_path, depth):
    with pytest.raises(OLMoConfigurationError, match="prefetch_max_in_flight"):
        _loader(tmp_path, depth=depth, workers=0)


@pytest.mark.parametrize("depth", [64])
@pytest.mark.parametrize("grouped", [True])
@pytest.mark.parametrize("buffer_size", [4])
def test_prefetch_depth_reaches_both_streams_and_group_children(
    tmp_path, monkeypatch, depth, grouped, buffer_size
):
    calls = []
    native_prefetch = mixture_data_loader.prefetch_map

    def record(fn, values, *, num_workers, max_in_flight=None):
        calls.append((num_workers, max_in_flight))
        return native_prefetch(fn, values, num_workers=num_workers, max_in_flight=max_in_flight)

    monkeypatch.setattr(mixture_data_loader, "prefetch_map", record)
    loader = _loader(tmp_path, depth=depth, grouped=grouped, buffer_size=buffer_size)
    for child in loader._group_loaders.values():
        assert child.prefetch_max_in_flight == depth
    loader.reshuffle(epoch=1)
    iterator = iter(loader)
    try:
        next(iterator)
    finally:
        iterator.close()
    assert calls == [(8, depth)] * (2 if grouped else 1)


@pytest.mark.parametrize("grouped", [False])
@pytest.mark.parametrize("buffer_size", [4])
@pytest.mark.parametrize("rank", [0])
def test_native_prefetch_preserves_every_batch_field_and_state(
    tmp_path, grouped, buffer_size, rank
):
    expected = None
    expected_state = None
    for workers, depth in [(0, None), (8, None), (8, 16), (4, 64), (4, 128)]:
        loader = _loader(
            tmp_path,
            workers=workers,
            depth=depth,
            grouped=grouped,
            buffer_size=buffer_size,
            rank=rank,
        )
        loader.reshuffle(epoch=2)
        iterator = iter(loader)
        try:
            batches = [next(iterator) for _ in range(4)]
            state = copy.deepcopy(loader.state_dict())
            assert {"images", "pooled_patches_idx", "pack_source_names"} <= batches[0].keys()
            if expected is None:
                expected, expected_state = batches, state
            else:
                _assert_equal(batches, expected)
                _assert_equal(state, expected_state)
        finally:
            iterator.close()
