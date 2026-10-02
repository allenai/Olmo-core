from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.distributed as dist

from olmo_core.config import Config
from olmo_core.data import TokenizerConfig
from olmo_core.data.multimodal.alignment import MultimodalMixtureConfig
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.vision import Molmo2TokenIds
from olmo_core.train.callbacks.checkpointer import CheckpointerCallback
from olmo_core.train.callbacks.evaluator_callback import EvaluatorCallback
from olmo_core.train.callbacks.multimodal import (
    InitializeMultimodalModelCallback,
    MultimodalEvaluatorCallbackConfig,
)
from olmo_core.train.common import Duration
from olmo_core.train.train_module.transformer.multimodal_train_module import (
    MultimodalOLMoDDPTrainModule,
)


@pytest.fixture
def bootstrap(monkeypatch):
    callback = InitializeMultimodalModelCallback(
        language_checkpoint="/checkpoints/language/step100",
        vision_model_id="google/siglip2-so400m-patch14-384",
        image_token_ids=[100, 101],
        vision_revision="test-revision",
        cache_dir="/cache/huggingface",
        seed=42,
        load_threads=2,
    )
    train_module = Mock(spec=MultimodalOLMoDDPTrainModule)
    vision_loader = Mock(return_value={"vision": object()})
    monkeypatch.setattr(
        "olmo_core.train.callbacks.multimodal._load_siglip_vision_state", vision_loader
    )
    callback.trainer = SimpleNamespace(
        checkpoint_loaded=False, train_module=train_module, global_step=0
    )
    return callback, train_module, vision_loader


@pytest.mark.parametrize("step", [17])
def test_restored_checkpoint_is_never_reinitialized(bootstrap, step):
    callback, train_module, vision_loader = bootstrap
    callback.trainer.checkpoint_loaded = True
    callback.trainer.global_step = step

    callback.pre_train()

    assert train_module.mock_calls == []
    vision_loader.assert_not_called()


@pytest.mark.parametrize("suffix", [""])
def test_fresh_run_loads_components_and_resets_only_input_rows(bootstrap, suffix):
    callback, train_module, vision_loader = bootstrap
    callback.language_checkpoint += suffix

    callback.pre_train()

    train_module.load_state_dict_direct.assert_called_once_with(
        "/checkpoints/language/step100/model_and_optim",
        process_group=dist.group.WORLD,
        thread_count=2,
        load_optim_state=False,
    )
    vision_loader.assert_called_once_with(
        callback.vision_model_id,
        revision="test-revision",
        cache_dir="/cache/huggingface",
        n_blocks=train_module.multimodal_model.cfg.vision.image_num_layers,
    )
    train_module.load_vision_state_dict.assert_called_once_with(vision_loader.return_value)
    train_module.reset_image_token_rows.assert_called_once_with(
        [100, 101], seed=42, reset_output_rows=False
    )
    assert [call[0] for call in train_module.mock_calls] == [
        "load_state_dict_direct",
        "load_vision_state_dict",
        "assert_vision_optimizer_state_synced",
        "reset_image_token_rows",
    ]
    assert callback.priority > CheckpointerCallback.priority
    assert callback.priority > EvaluatorCallback.priority


@pytest.fixture
def eval_setup(tmp_path, monkeypatch):
    dataset_config = MultimodalMixtureConfig(
        tokenizer=TokenizerConfig.dolma2(),
        sources={"caption": Config()},
        target_loss_mass={"caption": 1.0},
    )
    example = {
        "input_ids": np.ones(3, dtype=np.int64),
        "labels": np.ones(3, dtype=np.int64),
        "loss_masks": np.ones(3, dtype=np.float32),
        "position_ids": np.arange(3, dtype=np.int64),
        "token_type_ids": np.zeros(3, dtype=np.int64),
        "images": np.ones((1, 4, 8), dtype=np.float32),
        "pooled_patches_idx": np.arange(4, dtype=np.int64).reshape(1, 4),
    }
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    token_ids = Molmo2TokenIds()
    monkeypatch.setattr(
        dataset_config, "build_tokenizer", Mock(return_value=(tokenizer, token_ids))
    )
    monkeypatch.setattr(
        dataset_config, "build_sources", Mock(return_value={"caption": [example] * 8})
    )
    monkeypatch.setattr(dataset_config, "build", Mock(side_effect=AssertionError))
    monkeypatch.setattr("olmo_core.train.callbacks.multimodal.get_world_size", lambda _: 2)
    monkeypatch.setattr("olmo_core.train.callbacks.multimodal.get_rank", lambda _: 1)
    trainer = SimpleNamespace(dp_process_group=None, device=torch.device("cpu"), work_dir=tmp_path)
    config = MultimodalEvaluatorCallbackConfig(
        eval_dataset=dataset_config,
        sequence_length=4,
        rank_batch_size=2,
        examples_per_source=8,
        blank_image_sources=["caption"],
    )
    return config, trainer


def test_multimodal_evaluators_pair_identical_heldout_examples(eval_setup):
    config, trainer = eval_setup
    callback = config.build(trainer)

    assert callback.eval_duration == Duration.steps(2)
    assert callback.eval_interval == 500
    assert callback.eval_on_startup and callback.eval_on_finish
    actual, blank = callback.evaluators
    assert actual.name == "caption-validation"
    assert blank.name == "caption-blank-image"
    assert actual.batches.dp_rank == 1
    assert actual.batches.dp_world_size == 2
    first_batch = next(iter(actual))
    blank_batch = next(iter(blank))
    repeated_batch = next(iter(actual))
    assert first_batch["input_ids"].shape == (2, 4)
    assert first_batch["images"].count_nonzero() > 0
    assert blank_batch["images"].count_nonzero() == 0
    for field in ("input_ids", "labels", "loss_masks", "pooled_patches_idx"):
        assert torch.equal(first_batch[field], blank_batch[field])
        assert torch.equal(first_batch[field], repeated_batch[field])
    config.eval_dataset.build.assert_not_called()


@pytest.mark.parametrize("cancel_after_first_eval", [True])
def test_multimodal_evaluators_forward_cancel_after_first_eval(eval_setup, cancel_after_first_eval):
    config, trainer = eval_setup
    assert config.cancel_after_first_eval is False
    config.cancel_after_first_eval = cancel_after_first_eval

    callback = config.build(trainer)

    assert callback.cancel_after_first_eval is cancel_after_first_eval
    assert callback.eval_on_startup is True


def test_matched_image_evaluators_pair_recipients_and_limit_each_population(eval_setup):
    config, trainer = eval_setup
    template = config.eval_dataset.build_sources.return_value["caption"][0]
    rows = [{**template, "images": template["images"] * (i + 1)} for i in range(8)]
    config.eval_dataset.build_sources.return_value = {"caption": rows}
    config.matched_image_sources = ["caption"]
    config.matched_image_examples = 4
    config.matched_image_candidates = 8
    config.early_response_tokens = 2
    callback = config.build(trainer)
    ordinary, blank, correct, wrong, early_correct, early_wrong = callback.evaluators

    assert len(list(ordinary)) == len(list(blank)) == 2
    assert len(list(correct)) == len(list(wrong)) == 1
    assert len(list(early_correct)) == len(list(early_wrong)) == 1
    correct_batch = next(iter(correct))
    wrong_batch = next(iter(wrong))
    assert not torch.equal(correct_batch["images"], wrong_batch["images"])
    for name in correct_batch:
        if name != "images":
            assert torch.equal(correct_batch[name], wrong_batch[name])
    prefix_batch = next(iter(early_correct))
    assert prefix_batch["loss_masks"].count_nonzero().item() == 4
    assert torch.equal(prefix_batch["labels"], correct_batch["labels"])
    assert torch.equal(prefix_batch["input_ids"], correct_batch["input_ids"])
    assert early_wrong.reference_evaluator is early_correct
    assert wrong.reference_evaluator is correct

    correct.update_metrics(correct_batch, torch.tensor(6.0), None)
    wrong.update_metrics(wrong_batch, torch.tensor(12.0), None)
    torch.testing.assert_close(wrong.compute_metrics()["CE gap"], torch.tensor(1.0))


def test_matched_image_pairing_fails_closed_on_duplicate_images(eval_setup):
    config, trainer = eval_setup
    config.matched_image_sources = ["caption"]
    config.matched_image_examples = 4
    config.matched_image_candidates = 8
    with pytest.raises(OLMoConfigurationError, match="distinct exact-geometry"):
        config.build(trainer)


def test_nonleader_receives_pair_indices_without_decoding_candidates(eval_setup, monkeypatch):
    config, trainer = eval_setup
    config.matched_image_sources = ["caption"]
    config.matched_image_examples = 4
    config.matched_image_candidates = 8
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    prepare = Mock(side_effect=AssertionError("Nonleader must not prepare image candidates"))
    broadcast = Mock(return_value={"pairs": [(0, 1), (1, 0), (2, 3), (3, 2)]})
    monkeypatch.setattr("olmo_core.train.callbacks.multimodal.build_bounded_image_pairs", prepare)
    monkeypatch.setattr("olmo_core.train.callbacks.multimodal.broadcast_object", broadcast)
    callback = config.build(trainer)
    assert len(callback.evaluators) == 6
    prepare.assert_not_called()
    broadcast.assert_called_once_with(None, src=0, group=None)
