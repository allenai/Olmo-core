from datetime import timedelta
from unittest.mock import Mock

import pytest

from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.internal import experiment
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.train_module import TransformerTrainModuleConfig


@pytest.mark.parametrize("command", [experiment.SubCmd.train, experiment.SubCmd.eval_checkpoints])
def test_process_group_timeout_survives_config_roundtrip(command, monkeypatch):
    # Only environment setup is exercised, before any model/data construction.
    config = experiment.ExperimentConfig(
        run_name="cold-cache",
        launch=None,
        model=TransformerConfig.llama_like(
            d_model=64,
            vocab_size=128,
            n_layers=1,
            n_heads=2,
            feed_forward=FeedForwardConfig(hidden_size=128, bias=False),
        ),
        dataset=NumpyFSLDatasetConfig(
            paths=["unused.npy"], sequence_length=64, tokenizer=TokenizerConfig.dolma2()
        ),
        data_loader=NumpyDataLoaderConfig(global_batch_size=128, seed=0),
        train_module=TransformerTrainModuleConfig(
            rank_microbatch_size=128, max_sequence_length=64, optim=AdamWConfig()
        ),
        trainer=TrainerConfig(save_folder="unused", max_duration=Duration.steps(2)),
    ).merge(["process_group_timeout_seconds=7200"])
    restored = experiment.ExperimentConfig.from_dict(
        config.as_dict(include_class_name=True, json_safe=True)
    )
    prepare = Mock()
    monkeypatch.setattr(experiment, "prepare_training_environment", prepare)
    command.prepare_environment(restored)
    prepare.assert_called_once_with(backend=config.backend, timeout=timedelta(hours=2))
