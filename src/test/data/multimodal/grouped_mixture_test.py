"""CPU tests for explicit per-update mixture group exposure quotas."""


import numpy as np
import pytest
import torch

from olmo_core.data.multimodal.collator import MultimodalCollator
from olmo_core.data.multimodal.mixture_data_loader import (
    MixtureDataLoader,
    MixtureDataLoaderConfig,
)
from olmo_core.exceptions import OLMoConfigurationError


class _Dataset:
    def __init__(self, tag):
        self.tag = tag
        self.content_fingerprint = f"source-{tag}"

    def __len__(self):
        return 19

    def get(self, index, epoch):
        size = 3 + index % 3
        tokens = np.full(size, self.tag + index + 1000 * epoch, dtype=np.int64)
        return {
            "input_ids": tokens,
            "labels": tokens.copy(),
            "loss_masks": np.full(size, 0.5, dtype=np.float32),
            "position_ids": np.arange(size, dtype=np.int64),
            "token_type_ids": np.zeros(size, dtype=np.int64),
            "images": np.zeros((0, 2, 3), dtype=np.float32),
            "pooled_patches_idx": np.zeros((0, 1), dtype=np.int64),
        }


def _loader(tmp_path, *, pack=True, buffer_size=4, workers=0, rank=0, **kwargs):
    values = {
        "work_dir": tmp_path,
        "global_batch_size": 8 * 16,
        "seed": 17,
        "epoch_instances": 240,
        "pack": pack,
        "pack_max_crops": 2,
        "pack_buffer_size": buffer_size,
        "continuous_stream": pack and buffer_size > 0,
        "prefetch_workers": workers,
        "dp_world_size": 2,
        "dp_rank": rank,
        "dataset_names": ["arbitrary_parent_replay", "caption", "pointing"],
        "source_groups": {
            "arbitrary_parent_replay": "text",
            "caption": "vision",
            "pointing": "vision",
        },
        "group_sequence_quotas": {"vision": 6, "text": 2},
    }
    values.update(kwargs)
    return MixtureDataLoader(
        [_Dataset(10), _Dataset(100), _Dataset(200)],
        [0.001, 0.399, 0.6],
        MultimodalCollator(pad_token_id=0, pad_sequence_length=16),
        **values,
    )


def _assert_batches_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], torch.Tensor):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        else:
            assert actual[key] == expected[key]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_groups": {}},
        {"source_groups": {"caption": "vision"}},
        {"group_sequence_quotas": {"text": 1, "vision": 7}},
        {"group_sequence_quotas": {"text": 2, "vision": 4}},
        {"group_sequence_quotas": {"text": 0, "vision": 8}},
        {"group_sequence_quotas": {"text": 2, "other": 6}},
    ],
)
def test_grouped_quota_validation(tmp_path, kwargs):
    with pytest.raises(OLMoConfigurationError):
        _loader(tmp_path, **kwargs)


def test_quota_config_round_trip_and_legacy_default(tmp_path):
    config = MixtureDataLoaderConfig(
        global_batch_size=128,
        sequence_length=16,
        work_dir=str(tmp_path),
        source_groups={"replay": "text", "images": "vision"},
        group_sequence_quotas={"text": 2, "vision": 6},
    )
    restored = MixtureDataLoaderConfig.from_dict(config.as_config_dict())
    assert restored == config
    legacy = _loader(tmp_path, source_groups=None, group_sequence_quotas=None)
    assert "grouped_version" not in legacy.state_dict()
    assert "loss_group_names" not in legacy.get_mock_batch()
    grouped = _loader(tmp_path)
    assert grouped.get_mock_batch()["loss_group_names"] == ["text", "vision", "vision", "vision"]
    grouped.global_batch_size = 128
    with pytest.raises(OLMoConfigurationError, match="dynamic batch-size"):
        grouped.global_batch_size = 256
