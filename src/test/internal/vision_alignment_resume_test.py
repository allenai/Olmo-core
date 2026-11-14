"""Resuming a multimodal run keeps the tokenizer split and loss calibration it trained with."""

import json
from types import SimpleNamespace

import pytest

from olmo_core.data import TokenizerConfig
from olmo_core.data.multimodal.alignment import MultimodalMixtureConfig
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.vision_alignment import keep_resumed_tokenizer_split

MEANS = {"pixmo_caption": 28.5, "text_midtraining": 65535.0}
LEGACY_MEANS = {"pixmo_caption": 28.8, "text_midtraining": 65535.0}
TARGETS = {"pixmo_caption": 0.12, "text_midtraining": 0.88}
LEGACY_TARGETS = {"pixmo_caption": 0.11, "text_midtraining": 0.89}


def _legacy(means=LEGACY_MEANS, targets=LEGACY_TARGETS):
    """A saved dataset section of a run on the earlier split (no tokenizer_json_split)."""
    return {"mean_loss_weight": means, "target_loss_mass": targets}


def _config(save_folder):
    mixture = MultimodalMixtureConfig(
        tokenizer=TokenizerConfig.dolma2(),
        mean_loss_weight=dict(MEANS),
        target_loss_mass=dict(TARGETS),
    )
    evaluator = SimpleNamespace(
        eval_dataset=MultimodalMixtureConfig(tokenizer=TokenizerConfig.dolma2())
    )
    return SimpleNamespace(
        dataset=mixture,
        train_module=SimpleNamespace(source_loss_mass_targets=dict(TARGETS)),
        trainer=SimpleNamespace(
            save_folder=str(save_folder), callbacks={"multimodal_evaluator": evaluator}
        ),
    )


def _checkpoint(folder, step, dataset, train_module=None):
    """A complete checkpoint; ``dataset=None`` leaves out its config, as a save interrupted
    before the config callback does."""
    path = folder / f"step{step}"
    (path / "train").mkdir(parents=True)
    (path / "model_and_optim").mkdir()
    (path / "train" / "rank0.pt").write_bytes(b"")
    (path / "model_and_optim" / ".metadata").write_bytes(b"")
    (path / ".metadata.json").write_text(json.dumps({"ephemeral": False}))
    if dataset is not None:
        if train_module is None and "target_loss_mass" in dataset:
            train_module = {"source_loss_mass_targets": dataset["target_loss_mass"]}
        saved = {"dataset": dataset, "train_module": train_module or {}}
        (path / "config.json").write_text(json.dumps(saved))


def _evaluator_split(config):
    return config.trainer.callbacks["multimodal_evaluator"].eval_dataset.tokenizer_json_split


def test_a_new_run_keeps_the_default(tmp_path):
    config = _config(tmp_path / "missing")
    keep_resumed_tokenizer_split(config)
    assert config.dataset.tokenizer_json_split is True
    assert config.dataset.mean_loss_weight == MEANS
    assert _evaluator_split(config) is True


def test_a_run_saved_before_the_setting_resumes_with_its_split_and_means(tmp_path):
    _checkpoint(tmp_path, 100, _legacy())
    config = _config(tmp_path)
    keep_resumed_tokenizer_split(config)
    assert config.dataset.tokenizer_json_split is False
    assert config.dataset.mean_loss_weight == LEGACY_MEANS
    assert config.dataset.target_loss_mass == LEGACY_TARGETS
    assert config.train_module.source_loss_mass_targets == LEGACY_TARGETS
    assert _evaluator_split(config) is False


def test_a_legacy_run_without_saved_targets_is_rejected(tmp_path):
    _checkpoint(tmp_path, 100, {"mean_loss_weight": LEGACY_MEANS})
    with pytest.raises(OLMoConfigurationError, match="target_loss_mass"):
        keep_resumed_tokenizer_split(_config(tmp_path))


def test_a_run_with_the_setting_resumes_unchanged(tmp_path):
    _checkpoint(tmp_path, 100, _legacy())
    _checkpoint(
        tmp_path,
        200,
        {"tokenizer_json_split": True, "mean_loss_weight": MEANS, "target_loss_mass": TARGETS},
    )
    config = _config(tmp_path)
    keep_resumed_tokenizer_split(config)
    assert config.dataset.tokenizer_json_split is True
    assert config.dataset.mean_loss_weight == MEANS


def test_a_legacy_run_whose_saved_means_do_not_match_is_rejected(tmp_path):
    _checkpoint(tmp_path, 100, _legacy(means={"other_source": 1.0}))
    with pytest.raises(OLMoConfigurationError, match="tokenizer_json_split=False"):
        keep_resumed_tokenizer_split(_config(tmp_path))


def test_a_checkpoint_without_its_config_yields_to_the_newest_one_with_it(tmp_path):
    _checkpoint(tmp_path, 100, _legacy())
    _checkpoint(tmp_path, 200, None)
    config = _config(tmp_path)
    keep_resumed_tokenizer_split(config)
    assert config.dataset.tokenizer_json_split is False
    assert config.dataset.mean_loss_weight == LEGACY_MEANS


def test_checkpoints_without_any_config_keep_the_config(tmp_path):
    _checkpoint(tmp_path, 200, None)
    config = _config(tmp_path)
    keep_resumed_tokenizer_split(config)
    assert config.dataset.tokenizer_json_split is True
    assert config.dataset.mean_loss_weight == MEANS


def test_a_checkpoint_config_without_a_dataset_section_gives_nothing_to_recover(tmp_path):
    _checkpoint(tmp_path, 100, None)
    (tmp_path / "step100" / "config.json").write_text(json.dumps({"model": {}}))
    config = _config(tmp_path)
    keep_resumed_tokenizer_split(config)
    assert config.dataset.tokenizer_json_split is True
    assert config.dataset.mean_loss_weight == MEANS


def test_a_legacy_run_resumed_with_the_split_already_off_still_gets_its_means(tmp_path):
    """``--dataset.tokenizer_json_split=false`` alone leaves the code's means, calibrated for the
    new split; the checkpoint's own means must still replace them."""
    _checkpoint(tmp_path, 100, _legacy())
    config = _config(tmp_path)
    config.dataset.tokenizer_json_split = False
    keep_resumed_tokenizer_split(config)
    assert config.dataset.tokenizer_json_split is False
    assert config.dataset.mean_loss_weight == LEGACY_MEANS
    assert _evaluator_split(config) is False


def test_a_new_split_checkpoint_resumes_with_it(tmp_path):
    _checkpoint(
        tmp_path,
        100,
        {"tokenizer_json_split": True, "mean_loss_weight": MEANS, "target_loss_mass": TARGETS},
    )
    config = _config(tmp_path)
    config.dataset.tokenizer_json_split = False
    keep_resumed_tokenizer_split(config)
    assert config.dataset.tokenizer_json_split is True
    assert config.dataset.mean_loss_weight == MEANS
