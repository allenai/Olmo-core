import json
import random
from pathlib import Path

import numpy as np
import pytest

from olmo_core.data.multimodal.pretraining_replay import PretrainingReplayConfig
from olmo_core.data.numpy_dataset import InstanceFilterConfig, NumpyFSLDatasetConfig
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDatasetConfig,
    SourceMixtureList,
)
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.exceptions import OLMoConfigurationError


def _config(path: Path, **kwargs):
    return NumpyFSLDatasetConfig(
        tokenizer=TokenizerConfig(vocab_size=128, eos_token_id=126, pad_token_id=127),
        paths=[str(path)],
        sequence_length=4,
        dtype=NumpyDatasetDType.uint16,
        **kwargs,
    )


def _save_checkpoint(path: Path, dataset: NumpyFSLDatasetConfig, paths=None):
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"dataset": dataset.as_config_dict()}))
    if paths is not None:
        (path / "data_paths.txt").write_text("\n".join(map(str, paths)))


@pytest.mark.parametrize("dtype", [NumpyDatasetDType.uint32])
def test_native_tokens_labels_and_dtype(tmp_path, dtype):
    path = tmp_path / "tokens.npy"
    tokens = np.array([1, 126, 3, 127, 5, 6, 7, 8], dtype=dtype.as_np_dtype())
    tokens.tofile(path)
    config = _config(path)
    config.dtype = dtype
    replay = PretrainingReplayConfig(dataset=config).build()
    assert len(replay) == 2
    first = replay.get(0, epoch=3)
    np.testing.assert_array_equal(first["input_ids"], tokens[:4])
    np.testing.assert_array_equal(first["labels"], [126, 3, 127, -100])
    np.testing.assert_array_equal(first["loss_masks"], [1, 1, 1, 0])
    np.testing.assert_array_equal(first["position_ids"], [0, 1, 2, 3])
    np.testing.assert_array_equal(first["token_type_ids"], [0, 0, 0, 0])
    assert first["images"].shape[0] == 0
    assert first["pooled_patches_idx"].shape[0] == 0
    np.testing.assert_array_equal(replay[-1]["input_ids"], tokens[4:])


def test_mask_shift_and_repetition_filter(tmp_path):
    path = tmp_path / "tokens.npy"
    mask_path = tmp_path / "masks.npy"
    np.array([1, 2, 3, 4, 5, 5, 5, 5], dtype=np.uint16).tofile(path)
    np.array([True, False, True, True, False, False, False, False]).tofile(mask_path)
    config = _config(
        path,
        label_mask_paths=[str(mask_path)],
        instance_filter_config=InstanceFilterConfig(
            repetition_min_period=1, repetition_max_period=2, repetition_max_count=3
        ),
    )
    replay = PretrainingReplayConfig(dataset=config).build()
    np.testing.assert_array_equal(replay[0]["labels"], [-100, 3, 4, -100])
    np.testing.assert_array_equal(replay[0]["loss_masks"], [0, 1, 1, 0])
    filtered = replay[1]
    np.testing.assert_array_equal(filtered["labels"], [-100] * 4)
    np.testing.assert_array_equal(filtered["loss_masks"], [1, 1, 1, 0])
    assert filtered["metadata"]["instance_filter_valid"] is False


@pytest.mark.parametrize("seed", [17])
@pytest.mark.parametrize("validation_size", [7])
def test_replay_splits_are_disjoint_complete_and_repeatable(tmp_path, seed, validation_size):
    path = tmp_path / "tokens.npy"
    np.arange(80, dtype=np.uint16).tofile(path)
    config = _config(path)
    train_config = PretrainingReplayConfig(
        dataset=config, split="train", validation_size=validation_size, split_seed=seed
    )
    validation_config = train_config.copy()
    validation_config.split = "validation"
    train = train_config.build()
    validation = validation_config.build()
    train_starts = [int(train[i]["input_ids"][0]) for i in range(len(train))]
    validation_starts = [int(validation[i]["input_ids"][0]) for i in range(len(validation))]
    assert len(train) == 20 - validation_size
    assert len(validation) == validation_size
    assert not set(train_starts) & set(validation_starts)
    assert sorted(train_starts + validation_starts) == list(range(0, 80, 4))
    assert train_starts == sorted(train_starts)
    assert validation_starts == [
        i * 4 for i in random.Random(seed).sample(range(20), validation_size)
    ]
    for dataset in (train, validation):
        np.testing.assert_array_equal(
            dataset[-1]["input_ids"], dataset[len(dataset) - 1]["input_ids"]
        )
        for invalid in (len(dataset), -len(dataset) - 1):
            with pytest.raises(IndexError):
                dataset[invalid]
    assert train.fingerprint != validation.fingerprint
    assert train.fingerprint == train_config.build().fingerprint
    restored = PretrainingReplayConfig.from_dict(validation_config.as_config_dict())
    assert restored.build().fingerprint == validation.fingerprint


def test_saved_paths_precede_mix_without_preparing_corpus(tmp_path, monkeypatch):
    config = _config(tmp_path / "unused.npy")
    config.paths = None
    config.mix = "not-a-current-mix"
    config.mix_base_dir = "s3://pretraining-data"
    checkpoint = tmp_path / "checkpoint"
    paths = ["s3://pretraining-data/b.npy", "s3://pretraining-data/a.npy"]
    _save_checkpoint(checkpoint, config, paths)

    def unexpected_build(*args, **kwargs):
        pytest.fail("Resolving a replay config must not prepare token arrays")

    monkeypatch.setattr(NumpyFSLDatasetConfig, "build", unexpected_build)
    resolved = PretrainingReplayConfig(checkpoint=str(checkpoint)).resolve_dataset()
    assert resolved.paths == paths
    assert resolved.mix is None
    assert resolved.tokenizer == config.tokenizer
    assert resolved.dtype == config.dtype


def test_source_mixture_preserves_sampling_allocation(tmp_path, monkeypatch):
    import concurrent.futures
    from functools import partial

    monkeypatch.setattr(
        concurrent.futures,
        "ProcessPoolExecutor",
        partial(concurrent.futures.ProcessPoolExecutor, max_workers=1),
    )
    a, b = tmp_path / "a.npy", tmp_path / "b.npy"
    np.tile(np.array([1, 2, 3, 126], dtype=np.uint16), 16).tofile(a)
    np.tile(np.array([11, 12, 13, 126], dtype=np.uint16), 8).tofile(b)
    config = _config(a, work_dir=str(tmp_path / "cache"))
    config.paths = None
    config.source_mixture_config = SourceMixtureDatasetConfig(
        source_list=SourceMixtureList(
            sources=[
                SourceMixtureConfig("a", 0.25, [str(a)], max_source_fraction=0.5),
                SourceMixtureConfig("b", 0.75, [str(b)], max_repetition_ratio=2.0),
            ]
        ),
        requested_tokens=64,
        global_batch_size=16,
        seed=73,
        render_tables=False,
    )
    standard = config.build()
    standard.prepare()
    checkpoint = tmp_path / "checkpoint"
    _save_checkpoint(checkpoint, config, standard.paths)
    replay_config = PretrainingReplayConfig(checkpoint=str(checkpoint))
    resolved = replay_config.resolve_dataset()
    assert resolved.paths is None
    assert resolved.source_mixture_config.as_config_dict() == (
        config.source_mixture_config.as_config_dict()
    )
    replay = replay_config.build()
    assert len(replay) == len(standard) == 16
    for i in range(len(replay)):
        np.testing.assert_array_equal(replay[i]["input_ids"], standard[i]["input_ids"])
    assert sum(replay[i]["input_ids"][0] == 1 for i in range(len(replay))) == 4
    split_config = replay_config.copy()
    split_config.split = "train"
    with pytest.raises(OLMoConfigurationError, match="Weighted mixtures can repeat"):
        split_config.resolve_dataset()
    (checkpoint / "data_paths.txt").write_text(f"{b}\n{a}")
    with pytest.raises(OLMoConfigurationError, match="Rebuilt source mixture differs"):
        replay_config.build()


def _zero_allocation_mixture_config(tmp_path):
    unused, a, b = (tmp_path / name for name in ("unused.npy", "a.npy", "b.npy"))
    np.array([1, 2, 3, 126], dtype=np.uint16).tofile(unused)
    np.tile(np.array([5, 5, 5, 5, 5, 5, 5, 126], dtype=np.uint16), 8).tofile(a)
    np.tile(np.array([11, 12, 13, 126], dtype=np.uint16), 16).tofile(b)
    config = _config(
        unused,
        work_dir=str(tmp_path / "cache"),
        instance_filter_config=InstanceFilterConfig(
            repetition_min_period=1, repetition_max_period=2, repetition_max_count=3
        ),
    )
    config.paths = None
    config.source_mixture_config = SourceMixtureDatasetConfig(
        source_list=SourceMixtureList(
            sources=[
                SourceMixtureConfig("zero", 0.0, [str(unused), str(a)]),
                SourceMixtureConfig("positive", 1.0, [str(a), str(b), str(a)]),
            ]
        ),
        requested_tokens=48,
        global_batch_size=4,
        render_tables=False,
    )
    paths = list(map(str, [unused, a, a, b, a]))
    config.metadata = [{"occurrence": i} for i in range(len(paths))]
    return config, paths


def test_alignment_ancestry_and_cycle_detection(tmp_path):
    parent = tmp_path / "pretraining"
    config = _config(tmp_path / "tokens.npy")
    _save_checkpoint(parent, config)
    alignment = tmp_path / "alignment"
    alignment.mkdir()
    (alignment / "config.json").write_text(json.dumps({"pretraining_checkpoint": str(parent)}))
    replay = PretrainingReplayConfig(checkpoint=str(alignment))
    assert replay.resolve_dataset() == config
    (parent / "config.json").write_text(json.dumps({"pretraining_checkpoint": str(alignment)}))
    with pytest.raises(OLMoConfigurationError, match="Cycle"):
        replay.resolve_dataset()


def test_replay_config_roundtrip_and_window_override(tmp_path):
    config = _config(tmp_path / "tokens.npy", max_target_sequence_length=4)
    replay = PretrainingReplayConfig(dataset=config, sequence_length=8, work_dir=str(tmp_path))
    reconstructed = PretrainingReplayConfig.from_dict(replay.as_config_dict())
    assert reconstructed == replay
    resolved = reconstructed.resolve_dataset()
    assert resolved.sequence_length == resolved.max_target_sequence_length == 8
    assert resolved.work_dir == str(tmp_path)
    assert config.sequence_length == 4
    assert config.work_dir is None
    replay.sequence_length = 1
    with pytest.raises(OLMoConfigurationError, match="at least two"):
        replay.resolve_dataset()


def test_resume_fingerprint_covers_filter_and_sequence(tmp_path):
    path = tmp_path / "tokens.npy"
    np.arange(32, dtype=np.uint16).tofile(path)
    config = _config(path, max_target_sequence_length=8)
    initial = PretrainingReplayConfig(dataset=config).build()
    longer = PretrainingReplayConfig(dataset=config, sequence_length=8).build()
    assert initial.dataset.fingerprint == longer.dataset.fingerprint
    assert initial.fingerprint != longer.fingerprint
    config.instance_filter_config = InstanceFilterConfig()
    filtered = PretrainingReplayConfig(dataset=config).build()
    assert filtered.fingerprint != initial.fingerprint
    other_cache = PretrainingReplayConfig(dataset=config, work_dir=str(tmp_path / "cache")).build()
    assert filtered.fingerprint == other_cache.fingerprint
