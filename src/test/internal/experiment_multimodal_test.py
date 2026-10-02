import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from olmo_core.data import DataLoaderConfig, TokenizerConfig
from olmo_core.data.multimodal.alignment import (
    MultimodalDatasetMixture,
    MultimodalMixtureConfig,
)
from olmo_core.data.multimodal.mixture_data_loader import MixtureDataLoaderConfig
from olmo_core.data.multimodal.pixmo_cap import PixMoCapDatasetConfig
from olmo_core.internal.experiment import ExperimentConfig, SubCmd, _build_data_loader
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.vision import (
    Molmo2TokenIds,
    MultimodalLMConfig,
    VisionConnectorConfig,
    VisionEncoderConfig,
)
from olmo_core.optim import AdamWConfig
from olmo_core.train import TrainerConfig
from olmo_core.train.train_module.transformer.multimodal_train_module import (
    MultimodalTransformerTrainModuleConfig,
)


def _example(value):
    return {
        "input_ids": np.full(3, value, dtype=np.int64),
        "labels": np.full(3, value, dtype=np.int64),
        "loss_masks": np.ones(3, dtype=np.float32),
        "position_ids": np.arange(3, dtype=np.int64),
        "token_type_ids": np.zeros(3, dtype=np.int64),
        "images": np.zeros((0, 4, 8), dtype=np.float32),
        "pooled_patches_idx": np.zeros((0, 4), dtype=np.int64),
    }


@pytest.fixture
def mixture():
    return MultimodalDatasetMixture(
        names=["caption", "text"],
        datasets=[[_example(1)] * 8, [_example(2)] * 8],
        weights=[0.25, 0.75],
        tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=3),
        token_ids=Molmo2TokenIds(),
    )


def test_mixture_loader_builds_through_shared_runner(tmp_path, monkeypatch, mixture):
    dataset = MultimodalMixtureConfig(tokenizer=TokenizerConfig.dolma2())
    build = Mock(return_value=mixture)
    monkeypatch.setattr(dataset, "build", build)
    loader_config = MixtureDataLoaderConfig(
        global_batch_size=16, sequence_length=4, work_dir=str(tmp_path), seed=19
    )
    group = object()
    get_world_size = Mock(return_value=2)
    get_rank = Mock(return_value=1)
    monkeypatch.setattr(
        "olmo_core.data.multimodal.mixture_data_loader.get_world_size", get_world_size
    )
    monkeypatch.setattr("olmo_core.data.multimodal.mixture_data_loader.get_rank", get_rank)

    loader = _build_data_loader(
        SimpleNamespace(dataset=dataset, data_loader=loader_config), dp_process_group=group
    )
    loader.reshuffle(epoch=1)
    batch = next(iter(loader))

    build.assert_called_once_with()
    get_world_size.assert_called_once_with(group)
    get_rank.assert_called_once_with(group)
    assert loader.dp_rank == 1
    assert loader.dataset_names == ["caption", "text"]
    assert loader.weights == [0.25, 0.75]
    assert batch["input_ids"].shape == (2, 4)
    assert batch["input_ids"][:, -1].tolist() == [0, 0]
    assert batch["loss_masks"][:, -1].tolist() == [0.0, 0.0]
    assert not loader.allow_legacy_state_without_dataset_fingerprints


class _MultimodalExperimentConfig(ExperimentConfig):
    """The shared experiment config with multimodal component types (as the recipes declare)."""

    model: MultimodalLMConfig
    dataset: MultimodalMixtureConfig
    data_loader: MixtureDataLoaderConfig
    train_module: MultimodalTransformerTrainModuleConfig


def test_multimodal_experiment_round_trip_and_dry_run(tmp_path):
    lm = TransformerConfig.olmo2_1M(vocab_size=512)
    vision = VisionEncoderConfig()
    config = _MultimodalExperimentConfig(
        run_name="alignment",
        launch=None,
        model=MultimodalLMConfig(
            lm=lm,
            vision=vision,
            connector=VisionConnectorConfig.from_vision_encoder(vision, output_dim=lm.d_model),
        ),
        dataset=MultimodalMixtureConfig(
            tokenizer=TokenizerConfig.dolma2(),
            sources={"caption": PixMoCapDatasetConfig(dataset_path="synthetic")},
            target_loss_mass={"caption": 1.0},
            mean_loss_weight={"caption": 3.0},
        ),
        data_loader=MixtureDataLoaderConfig(
            global_batch_size=8, sequence_length=4, work_dir=str(tmp_path)
        ),
        train_module=MultimodalTransformerTrainModuleConfig(
            rank_microbatch_size=4, max_sequence_length=4, optim=AdamWConfig()
        ),
        trainer=TrainerConfig(save_folder=str(tmp_path)),
    )
    # ``as_config_dict`` omits the ``None`` launch of a local run; it is restored explicitly.
    restored = _MultimodalExperimentConfig.from_dict({"launch": None, **config.as_config_dict()})
    restored = restored.merge(["--data_loader.pack=true", "--model.image_patch_token_id=500"])

    assert isinstance(restored.model, MultimodalLMConfig)
    assert isinstance(restored.dataset, MultimodalMixtureConfig)
    assert isinstance(restored.data_loader, MixtureDataLoaderConfig)
    assert restored.data_loader.pack
    assert restored.model.image_patch_token_id == 500
    assert isinstance(
        DataLoaderConfig.from_dict(config.data_loader.as_config_dict()), MixtureDataLoaderConfig
    )
    SubCmd.dry_run.run(restored)


def test_text_runner_import_does_not_require_vision_dependencies():
    script = """
import sys
sys.modules.update({name: None for name in ("PIL", "transformers", "datasets")})
import olmo_core.internal.experiment
assert "olmo_core.data.multimodal" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
