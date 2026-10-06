"""Multi-phase handoffs and policies of the alignment recipe (perception and joint)."""

import json

import pytest

from olmo_core.data import NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.multimodal.alignment import MultimodalMixtureConfig
from olmo_core.data.multimodal.pixmo_cap import PixMoCapDatasetConfig
from olmo_core.data.multimodal.pretraining_replay import PretrainingReplayConfig
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal import vision_alignment
from olmo_core.internal.vision_alignment import VisionAlignmentExperimentConfig
from olmo_core.internal.vision_alignment_data import (
    ALIGNMENT_MEAN_LOSS_WEIGHTS,
    DEFAULT_ALIGNMENT_ARTIFACT_ROOT,
)
from olmo_core.nn.transformer import OLMoDDPModelConfig
from olmo_core.nn.vision import MultimodalLMConfig
from olmo_core.train import LoadStrategy
from olmo_core.train.callbacks.metric_saver import MetricSaverCallback
from olmo_core.train.callbacks.multimodal import (
    InitializeMultimodalModelCallback,
    RestoreMetricsCallback,
)


def test_all_phases_roundtrip_and_checkpoint_handoff(alignment_recipe):
    recipe = alignment_recipe
    bridge = recipe.build()
    bridge_checkpoint = recipe.save(bridge)
    perception = recipe.build("perception", bridge_checkpoint)
    perception_checkpoint = recipe.save(perception)
    joint = recipe.build("joint", perception_checkpoint)

    assert isinstance(
        bridge.trainer.callbacks["initialize_multimodal"], InitializeMultimodalModelCallback
    )
    assert bridge.trainer.load_strategy == LoadStrategy.if_available
    assert bridge.trainer.load_path is None
    assert bridge.trainer.load_optim_state is None
    assert bridge.trainer.load_trainer_state is None
    for config, parent in ((perception, bridge_checkpoint), (joint, perception_checkpoint)):
        assert "initialize_multimodal" not in config.trainer.callbacks
        assert config.trainer.load_path == str(parent)
        assert config.trainer.load_strategy == LoadStrategy.always
        assert config.trainer.load_optim_state is False
        assert config.trainer.load_trainer_state is False

    for config in (bridge, perception, joint):
        restored = VisionAlignmentExperimentConfig.from_dict(
            json.loads(json.dumps(config.as_config_dict()))
        )
        assert restored == config
        assert restored.pretraining_checkpoint == str(recipe.base)
        assert isinstance(restored.model, MultimodalLMConfig)
        assert isinstance(restored.model.lm, OLMoDDPModelConfig)
    initializer = bridge.trainer.callbacks["initialize_multimodal"]
    assert initializer.language_checkpoint == str(recipe.base)
    assert initializer.seed == bridge.init_seed
    assert "vision.*" not in perception.train_module.freeze_params
    replay = joint.dataset.sources["native_text_replay"]
    assert isinstance(replay, PretrainingReplayConfig)
    assert replay.checkpoint == str(recipe.base)
    assert replay.sequence_length == joint.data_loader.sequence_length == 8192
    assert joint.dataset.target_loss_mass["native_text_replay"] == 0.35
    holdout = joint.trainer.callbacks["multimodal_evaluator"].eval_dataset.sources[
        "native_text_holdout"
    ]
    assert replay.split == "train"
    assert holdout.split == "validation"
    expected_holdout = replay.copy()
    expected_holdout.split = "validation"
    assert holdout == expected_holdout
    assert holdout.validation_size == 1024
    assert "native_text_replay" not in bridge.dataset.sources
    assert "native_text_replay" not in perception.dataset.sources


@pytest.mark.parametrize(
    "phase,microbatch,max_checkpoints,eval_batch_size",
    [("bridge", 4, 2, 4), ("perception", 4, 2, 4), ("joint", 2, 3, 1)],
)
def test_phase_loader_checkpoint_and_evaluation_defaults(
    alignment_recipe, phase, microbatch, max_checkpoints, eval_batch_size
):
    parent = None
    for stage in vision_alignment.AlignmentPhase:
        config = alignment_recipe.build(stage, parent)
        if stage == phase:
            break
        parent = alignment_recipe.save(config)
    loader, module, trainer = config.data_loader, config.train_module, config.trainer
    assert loader.sequence_length == module.max_sequence_length == 8192
    assert loader.global_batch_size == 128 * 8192
    assert module.rank_microbatch_size == microbatch * 8192
    assert module.ep_config.degree == 8
    assert loader.pack and loader.pack_max_crops == 64
    assert loader.pack_buffer_size == 48 and loader.prefetch_workers == 8
    assert loader.prefetch_max_in_flight is loader.defer_packed_image_copy is None
    assert loader.max_consecutive_data_errors == loader.max_total_data_errors == 0
    assert loader.seed == 95818 and config.init_seed == 6198
    assert not module.trim_microbatch_image_padding
    assert module.train_embedding_rows == vision_alignment._image_token_rows(
        alignment_recipe.token_ids
    )
    assert not module.reset_optimizer_states_on_load
    assert not module.reset_optimizer_states_on_resume
    assert module.diagnostics_interval == trainer.metrics_collect_interval == 1
    assert all(
        source.max_sequence_length == 8192
        for source in config.dataset.sources.values()
        if not isinstance(source, PretrainingReplayConfig)
    )

    callbacks = trainer.callbacks
    assert callbacks["beaker"] is not None
    assert all(
        type(callback).__module__.startswith("olmo_core.") for callback in callbacks.values()
    )
    wandb = callbacks["wandb"]
    assert wandb.entity is None and wandb.project == "vision-alignment" and wandb.auto_resume
    metrics, restore = callbacks["metrics"], callbacks["restore_metrics"]
    assert isinstance(metrics, MetricSaverCallback) and metrics.save_interval == 1
    assert metrics.final_metrics_fname == "metrics-final.json"
    assert isinstance(restore, RestoreMetricsCallback) and restore.metrics_callback == "metrics"
    assert sum(isinstance(callback, MetricSaverCallback) for callback in callbacks.values()) == 1
    assert sum(isinstance(callback, RestoreMetricsCallback) for callback in callbacks.values()) == 1
    checkpointer = callbacks["checkpointer"]
    assert checkpointer.enabled and not checkpointer.save_async
    assert checkpointer.save_interval == 500 and checkpointer.ephemeral_save_interval == 50
    assert checkpointer.max_checkpoints == max_checkpoints
    assert checkpointer.pre_train_checkpoint is False and checkpointer.fixed_steps is None
    evaluator = callbacks["multimodal_evaluator"]
    assert evaluator.eval_interval == 500 and evaluator.eval_on_startup and evaluator.eval_on_finish
    assert evaluator.sequence_length == 8192 and evaluator.rank_batch_size == eval_batch_size
    assert evaluator.examples_per_source == evaluator.matched_image_examples == 64
    assert evaluator.matched_image_candidates == 512 and evaluator.early_response_tokens == 8
    assert (
        evaluator.blank_image_sources
        == evaluator.matched_image_sources
        == [
            "pixmo_caption",
            "pixmo_transcript",
        ]
    )
    for source in evaluator.eval_dataset.sources.values():
        assert source.split == "validation"
        if isinstance(source, PretrainingReplayConfig):
            assert source.sequence_length == 8192
        else:
            assert source.max_sequence_length == (8192 if phase == "joint" else 2560)
            assert source.max_crops == 8


@pytest.mark.parametrize("phase", ["perception", "joint"])
def test_rejects_bare_language_checkpoint_as_phase_parent(alignment_recipe, phase):
    with pytest.raises(OLMoConfigurationError, match="requires a .* checkpoint"):
        alignment_recipe.build(phase, alignment_recipe.base)


def test_rejects_wrong_previous_phase(alignment_recipe):
    bridge = alignment_recipe.save(alignment_recipe.build())
    perception = alignment_recipe.save(alignment_recipe.build("perception", bridge))
    with pytest.raises(OLMoConfigurationError, match="requires a bridge checkpoint"):
        alignment_recipe.build("perception", perception)


def test_bridge_requires_one_bare_pretraining_checkpoint(alignment_recipe):
    with pytest.raises(OLMoConfigurationError, match="Bridge requires"):
        alignment_recipe.build(parent=alignment_recipe.base)
    with pytest.raises(OLMoConfigurationError, match="Bridge requires"):
        alignment_recipe.build(overrides=["--recipe.pretraining_checkpoint=null"])


def test_bridge_accepts_a_different_pretraining_checkpoint(alignment_recipe, tmp_path):
    other = tmp_path / "other-pretraining"
    other.mkdir()
    (other / "config.json").write_text((alignment_recipe.base / "config.json").read_text())
    config = alignment_recipe.build(overrides=[f"--recipe.pretraining_checkpoint={other}"])
    assert config.pretraining_checkpoint == str(other)
    assert config.trainer.callbacks["initialize_multimodal"].language_checkpoint == str(other)


def test_phase_parent_retains_pretraining_ancestry(alignment_recipe):
    bridge = alignment_recipe.save(alignment_recipe.build())
    with pytest.raises(OLMoConfigurationError, match="differs from phase parent"):
        alignment_recipe.build(
            "perception", bridge, overrides=["--recipe.pretraining_checkpoint=/different/model"]
        )


@pytest.mark.parametrize("sequence_length", [None, 8192])
def test_config_build_does_not_open_corpus_or_model_weights(
    alignment_recipe, monkeypatch, sequence_length
):
    def unexpected_build(*args, **kwargs):
        pytest.fail("Config construction must not build datasets or model weights")

    monkeypatch.setattr(NumpyFSLDatasetConfig, "build", unexpected_build)
    monkeypatch.setattr(PretrainingReplayConfig, "build", unexpected_build)
    monkeypatch.setattr(PixMoCapDatasetConfig, "build", unexpected_build)
    monkeypatch.setattr(MultimodalLMConfig, "build", unexpected_build)
    overrides = [] if sequence_length is None else [f"--recipe.sequence_length={sequence_length}"]
    bridge = alignment_recipe.build(overrides=overrides)
    perception = alignment_recipe.build(
        "perception", alignment_recipe.save(bridge), overrides=overrides
    )
    joint = alignment_recipe.build("joint", alignment_recipe.save(perception), overrides=overrides)
    assert joint.trainer.max_duration.value == 1500


def test_step_limit_override_does_not_shorten_joint_lr_horizon(alignment_recipe):
    bridge = alignment_recipe.save(alignment_recipe.build())
    perception = alignment_recipe.save(alignment_recipe.build("perception", bridge))
    joint = alignment_recipe.build(
        "joint", perception, overrides=["--trainer.max_duration.value=1000"]
    )
    assert joint.trainer.max_duration.value == 1000
    assert joint.train_module.scheduler.default.t_max == 1500
    assert joint.train_module.scheduler.schedulers["connector"].t_max == 1500


def test_different_loader_and_train_contexts_are_rejected(alignment_recipe):
    with pytest.raises(OLMoConfigurationError, match="sequence lengths must agree"):
        alignment_recipe.build(overrides=["--data_loader.sequence_length=1024"])


@pytest.mark.parametrize(
    "phase,sequence_length,microbatch_instances",
    [("bridge", 4096, 4), ("perception", 4096, 4), ("joint", 4096, 2)],
)
def test_recipe_context_override_updates_all_components(
    alignment_recipe, phase, sequence_length, microbatch_instances
):
    parent = None
    if phase != "bridge":
        parent = alignment_recipe.save(alignment_recipe.build())
    if phase == "joint":
        parent = alignment_recipe.save(alignment_recipe.build("perception", parent))
    baseline = alignment_recipe.build(phase, parent)
    overrides = [f"--recipe.sequence_length={sequence_length}"]
    if phase == "joint":
        overrides.append(f"--dataset.mean_loss_weight.native_text_replay={sequence_length - 1}")
    config = alignment_recipe.build(phase, parent, overrides=overrides)

    assert config.data_loader.sequence_length == sequence_length
    assert config.data_loader.global_batch_size == 128 * sequence_length
    assert config.train_module.max_sequence_length == sequence_length
    assert config.train_module.rank_microbatch_size == microbatch_instances * sequence_length
    evaluator = config.trainer.callbacks["multimodal_evaluator"]
    assert evaluator.sequence_length == sequence_length
    assert evaluator.rank_batch_size == (1 if phase == "joint" else microbatch_instances)
    for dataset in (config.dataset, evaluator.eval_dataset):
        for source in dataset.sources.values():
            if isinstance(source, PretrainingReplayConfig):
                assert source.sequence_length == sequence_length
            else:
                expected_length = (
                    min(sequence_length, 2560)
                    if phase != "joint" and dataset is evaluator.eval_dataset
                    else sequence_length
                )
                assert source.max_sequence_length == expected_length
    assert config.train_module.freeze_params == baseline.train_module.freeze_params
    assert config.train_module.scheduler == baseline.train_module.scheduler
    assert config.model == baseline.model
    assert config.trainer.max_duration == baseline.trainer.max_duration
    assert VisionAlignmentExperimentConfig.from_dict(config.as_config_dict()) == config
    original = json.loads((alignment_recipe.base / "config.json").read_text())
    assert original["dataset"]["sequence_length"] == 8192


@pytest.mark.parametrize("length", ["0", "-1", "1", "true", "8192.5"])
def test_recipe_context_override_rejects_invalid_length(alignment_recipe, length):
    with pytest.raises(OLMoConfigurationError, match="sequence_length"):
        alignment_recipe.build(overrides=[f"--recipe.sequence_length={length}"])


def test_recipe_context_override_requires_fresh_visual_calibration(alignment_recipe):
    overrides = [f"--recipe.artifact_root={DEFAULT_ALIGNMENT_ARTIFACT_ROOT}"]
    baseline = alignment_recipe.build(include_means=False, overrides=overrides)
    assert baseline.dataset.mean_loss_weight == ALIGNMENT_MEAN_LOSS_WEIGHTS["bridge"]
    with pytest.raises(
        OLMoConfigurationError,
        match=r"Changing recipe.sequence_length.*pixmo_caption.*pixmo_transcript",
    ):
        alignment_recipe.build(
            include_means=False,
            overrides=[*overrides, "--recipe.sequence_length=4096"],
        )
    assert ALIGNMENT_MEAN_LOSS_WEIGHTS["bridge"] == baseline.dataset.mean_loss_weight


def test_explicit_default_context_keeps_existing_visual_calibration(alignment_recipe):
    config = alignment_recipe.build(
        include_means=False,
        overrides=[
            f"--recipe.artifact_root={DEFAULT_ALIGNMENT_ARTIFACT_ROOT}",
            "--recipe.sequence_length=8192",
        ],
    )
    assert config.dataset.mean_loss_weight == ALIGNMENT_MEAN_LOSS_WEIGHTS["bridge"]


def test_changed_joint_context_derives_exact_unmasked_replay_calibration(alignment_recipe):
    bridge = alignment_recipe.save(alignment_recipe.build())
    perception = alignment_recipe.save(alignment_recipe.build("perception", bridge))
    visual_means = {
        name: mean
        for name, mean in ALIGNMENT_MEAN_LOSS_WEIGHTS["joint"].items()
        if name != "native_text_replay"
    }
    config = alignment_recipe.build(
        "joint",
        perception,
        include_means=False,
        overrides=[
            "--recipe.sequence_length=4096",
            f"--dataset.mean_loss_weight={json.dumps(visual_means)}",
        ],
    )
    assert config.dataset.mean_loss_weight["native_text_replay"] == 4095
    assert config.dataset.sampling_weights()["native_text_replay"] > 0


def test_unmasked_replay_rejects_stale_context_calibration(alignment_recipe):
    bridge = alignment_recipe.save(alignment_recipe.build())
    perception = alignment_recipe.save(alignment_recipe.build("perception", bridge))
    with pytest.raises(OLMoConfigurationError, match="exact mean_loss_weight 4095"):
        alignment_recipe.build("joint", perception, overrides=["--recipe.sequence_length=4096"])


def test_changed_source_requires_its_calibration(alignment_recipe):
    with pytest.raises(OLMoConfigurationError, match="changed sources:.*pixmo_caption"):
        alignment_recipe.build(
            include_means=False,
            overrides=[
                "--dataset.sources.pixmo_caption.max_crops=4",
                "--dataset.mean_loss_weight.pixmo_transcript=30",
            ],
        )
    assert alignment_recipe.constructed_sources[0]["pixmo_caption"].max_crops == 8


def test_changed_source_accepts_whole_calibration_mapping_without_mutating_defaults(
    alignment_recipe,
):
    means = {"pixmo_caption": 12.5, "pixmo_transcript": 30.0}
    config = alignment_recipe.build(
        include_means=False,
        overrides=[
            "--dataset.sources.pixmo_caption.max_crops=4",
            f"--dataset.mean_loss_weight={json.dumps(means)}",
        ],
    )
    assert config.dataset.mean_loss_weight == means
    assert config.dataset.sources["pixmo_caption"].max_crops == 4
    original = alignment_recipe.constructed_sources[0]["pixmo_caption"]
    assert original.max_crops == 8
    assert original is not config.dataset.sources["pixmo_caption"]
    assert ALIGNMENT_MEAN_LOSS_WEIGHTS["bridge"]["pixmo_caption"] != 12.5


@pytest.mark.parametrize("phase", ["bridge", "perception"])
@pytest.mark.parametrize("overlap", ["same", "ancestor", "descendant", "alias"])
def test_output_must_not_overlap_parent(alignment_recipe, phase, overlap):
    parent = (
        alignment_recipe.base
        if phase == "bridge"
        else alignment_recipe.save(alignment_recipe.build())
    )
    paths = {
        "same": str(parent),
        "ancestor": str(parent.parent),
        "descendant": str(parent / "new-phase"),
        "alias": f"{parent.parent}/./{parent.name}",
    }
    with pytest.raises(OLMoConfigurationError, match="separate output folder"):
        alignment_recipe.build(
            phase,
            parent=parent if phase == "perception" else None,
            overrides=[f"--trainer.save_folder={paths[overlap]}"],
        )


@pytest.mark.parametrize(
    "override",
    ["--dataset.tokenizer.eos_token_id=7", "--dataset.tokenizer_revision=different"],
)
def test_component_tokenizer_override_is_rejected(alignment_recipe, override):
    with pytest.raises(OLMoConfigurationError, match="Select the tokenizer through"):
        alignment_recipe.build(overrides=[override])


@pytest.mark.parametrize(
    "override",
    [
        "--trainer.callbacks.multimodal_evaluator.eval_dataset.tokenizer.eos_token_id=7",
        "--trainer.callbacks.multimodal_evaluator.eval_dataset.tokenizer_revision=different",
        "--trainer.callbacks.multimodal_evaluator.eval_dataset.model_vocab_size=100353",
        "--trainer.callbacks.multimodal_evaluator.sequence_length=4096",
    ],
)
def test_evaluation_config_must_match_training(alignment_recipe, override):
    with pytest.raises(OLMoConfigurationError, match="Evaluation"):
        alignment_recipe.build(overrides=[override])


def test_replay_window_must_match_loader(alignment_recipe):
    bridge = alignment_recipe.save(alignment_recipe.build())
    perception = alignment_recipe.save(alignment_recipe.build("perception", bridge))
    with pytest.raises(OLMoConfigurationError, match="Replay and data-loader sequence lengths"):
        alignment_recipe.build(
            "joint",
            perception,
            overrides=[
                "--dataset.sources.native_text_replay.sequence_length=4096",
                "--dataset.mean_loss_weight.native_text_replay=4095",
            ],
        )


def test_phase_handoffs_inherit_parent_tokenizer_revision(alignment_recipe):
    bridge = alignment_recipe.build(overrides=["--recipe.tokenizer_revision=custom-revision"])
    perception = alignment_recipe.build("perception", alignment_recipe.save(bridge))
    joint = alignment_recipe.build("joint", alignment_recipe.save(perception))
    assert perception.dataset.tokenizer_revision == "custom-revision"
    assert joint.dataset.tokenizer_revision == "custom-revision"


def test_phase_handoff_rejects_changed_tokenizer_revision(alignment_recipe):
    bridge = alignment_recipe.save(
        alignment_recipe.build(overrides=["--recipe.tokenizer_revision=parent-revision"])
    )
    with pytest.raises(OLMoConfigurationError, match="revision differs from the phase parent"):
        alignment_recipe.build(
            "perception", bridge, overrides=["--recipe.tokenizer_revision=child-revision"]
        )


@pytest.mark.parametrize("field,value", [("identifier", "other/tokenizer"), ("eos_token_id", 7)])
def test_phase_handoff_rejects_parent_tokenizer_mismatch(alignment_recipe, field, value):
    bridge = alignment_recipe.build()
    setattr(bridge.dataset.tokenizer, field, value)
    with pytest.raises(OLMoConfigurationError, match="parent tokenizer differs"):
        alignment_recipe.build("perception", alignment_recipe.save(bridge))


def test_legacy_parent_tokenizer_metadata_is_checked():
    tokenizer = TokenizerConfig.dolma2()
    recipe = vision_alignment.VisionAlignmentRecipeConfig()
    parent = {
        "artifacts": {"tokenizer_id": tokenizer.identifier, "tokenizer_revision": "legacy-pin"}
    }
    assert vision_alignment._resolve_tokenizer_revision(recipe, parent, tokenizer) == "legacy-pin"
    parent["artifacts"]["tokenizer_id"] = "other/tokenizer"
    with pytest.raises(OLMoConfigurationError, match="parent tokenizer differs"):
        vision_alignment._resolve_tokenizer_revision(recipe, parent, tokenizer)


def test_unpinned_parent_revision_is_not_replaced_by_a_default_pin():
    tokenizer = TokenizerConfig.dolma2()
    recipe = vision_alignment.VisionAlignmentRecipeConfig()
    parent = {"dataset": {"tokenizer": tokenizer.as_config_dict(), "tokenizer_revision": None}}
    assert vision_alignment._resolve_tokenizer_revision(recipe, parent, tokenizer) is None


@pytest.mark.parametrize("supply_mean", [False, True])
def test_masked_parent_replay_requires_calibrated_mean(alignment_recipe, supply_mean):
    base_config_path = alignment_recipe.base / "config.json"
    base = json.loads(base_config_path.read_text())
    base["dataset"]["label_mask_paths"] = ["s3://pretraining/masks.npy"]
    base_config_path.write_text(json.dumps(base))
    bridge = alignment_recipe.save(alignment_recipe.build())
    perception = alignment_recipe.save(alignment_recipe.build("perception", bridge))
    overrides = [f"--recipe.artifact_root={DEFAULT_ALIGNMENT_ARTIFACT_ROOT}"]
    if supply_mean:
        overrides.append("--dataset.mean_loss_weight.native_text_replay=4095.5")
        joint = alignment_recipe.build(
            "joint", perception, include_means=False, overrides=overrides
        )
        assert joint.dataset.mean_loss_weight["native_text_replay"] == 4095.5
    else:
        with pytest.raises(ValueError, match="missing mean weights.*native_text_replay"):
            alignment_recipe.build("joint", perception, include_means=False, overrides=overrides)


def test_tied_embedding_override_fails_during_config_build(alignment_recipe):
    with pytest.raises(OLMoConfigurationError, match="require untied LM input and output weights"):
        alignment_recipe.build(overrides=["--model.lm.tie_word_embeddings=true"])


def test_holdout_uses_final_replay_config(alignment_recipe):
    bridge = alignment_recipe.save(alignment_recipe.build())
    perception = alignment_recipe.save(alignment_recipe.build("perception", bridge))
    joint = alignment_recipe.build(
        "joint",
        perception,
        overrides=["--dataset.sources.native_text_replay.split_seed=42"],
    )
    train = joint.dataset.sources["native_text_replay"]
    holdout = joint.trainer.callbacks["multimodal_evaluator"].eval_dataset.sources[
        "native_text_holdout"
    ]
    assert train.split_seed == holdout.split_seed == 42
    assert holdout is not train


@pytest.mark.parametrize(
    "override",
    [
        "--recipe.text_validation_size=-1",
        "--recipe.text_validation_size=32",
        "--dataset.sources.native_text_replay.split=all",
    ],
)
def test_invalid_recipe_holdout_is_rejected(alignment_recipe, override):
    bridge = alignment_recipe.save(alignment_recipe.build())
    perception = alignment_recipe.save(alignment_recipe.build("perception", bridge))
    with pytest.raises(OLMoConfigurationError):
        alignment_recipe.build("joint", perception, overrides=[override])


def test_automatic_holdout_can_be_disabled_for_separate_validation(alignment_recipe):
    bridge = alignment_recipe.save(alignment_recipe.build())
    perception = alignment_recipe.save(alignment_recipe.build("perception", bridge))
    joint = alignment_recipe.build(
        "joint", perception, overrides=["--recipe.text_validation_size=0"]
    )
    assert joint.dataset.sources["native_text_replay"].split == "all"
    assert "native_text_holdout" not in (
        joint.trainer.callbacks["multimodal_evaluator"].eval_dataset.sources
    )


@pytest.mark.parametrize("default_weight", [None, 0.0, 0.02])
def test_router_load_balancing_is_inherited_through_phase_handoffs(
    alignment_recipe, default_weight
):
    from olmo_core.nn.moe.v2.router import MoERouterConfigV2

    config_path = alignment_recipe.base / "config.json"
    base = json.loads(config_path.read_text())
    lm = OLMoDDPModelConfig.from_dict(base["model"])
    lm.block.routed_experts_router = MoERouterConfigV2(
        d_model=64, num_experts=8, top_k=2, lb_loss_weight=default_weight
    )
    lm.block_overrides = {1: lm.block.copy()}
    lm.block_overrides[1].routed_experts_router.lb_loss_weight = 0.03
    base["model"] = lm.as_config_dict()
    config_path.write_text(json.dumps(base))

    bridge = alignment_recipe.build(overrides=["--recipe.restore_pretraining_router_lb=true"])
    perception = alignment_recipe.build("perception", alignment_recipe.save(bridge))
    joint = alignment_recipe.build("joint", alignment_recipe.save(perception))
    for config in (bridge, perception, joint):
        assert config.model.lm.block.routed_experts_router.lb_loss_weight == default_weight
        assert config.model.lm.block_overrides[1].routed_experts_router.lb_loss_weight == 0.03


@pytest.mark.parametrize("weight", [0.0, 0.025])
def test_router_load_balancing_override_is_noop_without_routed_experts(alignment_recipe, weight):
    inherited = alignment_recipe.build()
    overridden = alignment_recipe.build(overrides=[f"--recipe.router_lb_loss_weight={weight}"])
    assert overridden.model == inherited.model
    assert overridden.train_module == inherited.train_module


def test_router_load_balancing_override_persists_until_explicitly_reenabled(alignment_recipe):
    from olmo_core.nn.moe.v2.router import MoERouterConfigV2

    config_path = alignment_recipe.base / "config.json"
    base = json.loads(config_path.read_text())
    lm = OLMoDDPModelConfig.from_dict(base["model"])
    lm.block.routed_experts_router = MoERouterConfigV2(
        d_model=64, num_experts=8, top_k=2, lb_loss_weight=0.02, z_loss_weight=0.003
    )
    base["model"] = lm.as_config_dict()
    config_path.write_text(json.dumps(base))
    bridge = alignment_recipe.build(overrides=["--recipe.router_lb_loss_weight=0"])
    perception = alignment_recipe.build("perception", alignment_recipe.save(bridge))
    joint = alignment_recipe.build(
        "joint",
        alignment_recipe.save(perception),
        overrides=["--recipe.router_lb_loss_weight=0.02"],
    )
    assert bridge.model.lm.block.routed_experts_router.lb_loss_weight == 0.0
    assert perception.recipe.router_lb_loss_weight is None
    assert perception.model.lm.block.routed_experts_router.lb_loss_weight == 0.0
    assert joint.model.lm.block.routed_experts_router.lb_loss_weight == 0.02
    for config in (bridge, perception, joint):
        assert config.model.lm.block.routed_experts_router.z_loss_weight == 0.003


@pytest.mark.parametrize("weight", ["-0.01", ".nan", ".inf", "-.inf"])
def test_router_load_balancing_override_rejects_invalid_weights(alignment_recipe, weight):
    with pytest.raises(OLMoConfigurationError, match="finite and nonnegative"):
        alignment_recipe.build(overrides=[f"--recipe.router_lb_loss_weight={weight}"])


@pytest.mark.parametrize("phase", ["bridge", "perception", "joint"])
@pytest.mark.parametrize("default_weight", [None, 0.0, 0.02])
def test_restore_pretraining_router_lb_only_changes_coefficients(
    alignment_recipe, phase, default_weight
):
    original = alignment_recipe.set_router_coefficients(default_weight)
    parent = None
    if phase != "bridge":
        bridge = alignment_recipe.build(overrides=["--recipe.router_lb_loss_weight=0"])
        for block in bridge.model.lm.resolved_block_configs:
            block.ep.capacity_factor = 8.0
            block.routed_experts_router.z_loss_weight = 0.009
        parent = alignment_recipe.save(bridge)
        if phase == "joint":
            parent = alignment_recipe.save(alignment_recipe.build("perception", parent))
    inherited = alignment_recipe.build(
        phase, parent, overrides=["--recipe.restore_pretraining_router_lb=false"]
    )
    restored = alignment_recipe.build(
        phase, parent, overrides=["--recipe.restore_pretraining_router_lb=true"]
    )
    expected = inherited.model.copy()
    for block, pretrained in zip(
        expected.lm.resolved_block_configs, original.resolved_block_configs
    ):
        block.routed_experts_router.lb_loss_weight = pretrained.routed_experts_router.lb_loss_weight
    assert restored.model == expected
    assert restored.train_module == inherited.train_module
    assert restored.dataset == inherited.dataset
    assert restored.trainer == inherited.trainer
    assert restored.recipe.restore_pretraining_router_lb
    assert not inherited.recipe.restore_pretraining_router_lb
    assert restored.recipe.router_lb_loss_weight is None
    assert VisionAlignmentExperimentConfig.from_dict(restored.as_config_dict()) == restored
    if phase != "bridge":
        assert all(
            block.routed_experts_router.lb_loss_weight == 0
            for block in inherited.model.lm.resolved_block_configs
        )


@pytest.mark.parametrize(
    "override",
    [
        "--recipe.router_lb_loss_weight=0",
        "--recipe.router_lb_loss_weight=0.025",
        "--model.lm.block.routed_experts_router.lb_loss_weight=0",
    ],
)
def test_restore_pretraining_router_lb_rejects_conflicting_overrides(alignment_recipe, override):
    with pytest.raises(OLMoConfigurationError, match="mutually exclusive"):
        alignment_recipe.build(overrides=["--recipe.restore_pretraining_router_lb=true", override])


@pytest.mark.parametrize(
    "override",
    [
        "--model.lm.n_layers=5",
        "--model.lm.d_model=96",
        "--model.lm.block.routed_experts_router.num_experts=16",
        "--model.lm.block.routed_experts_router.top_k=1",
        "--model.lm.block.routed_experts_router.gating_function=sigmoid",
        "--model.lm.block.routed_experts_router=null",
        "--model.lm.block.use_pre_norm=true",
    ],
)
def test_restore_pretraining_router_lb_checks_final_cli_topology(alignment_recipe, override):
    alignment_recipe.set_router_coefficients(0.02)
    parent = alignment_recipe.save(
        alignment_recipe.build(overrides=["--recipe.router_lb_loss_weight=0"])
    )
    with pytest.raises(OLMoConfigurationError, match="topology"):
        alignment_recipe.build(
            "perception",
            parent,
            overrides=["--recipe.restore_pretraining_router_lb=true", override],
        )


def test_restore_pretraining_router_lb_resolves_aliases_and_override_layout(alignment_recipe):
    pretrained = alignment_recipe.set_router_coefficients(0.02)
    current = pretrained.copy()
    current.block_overrides = None  # Same resolved architecture, compacted LB-off metadata.
    current.block.routed_experts_router.lb_loss_weight = 0
    current.block.routed_experts_router.z_loss_weight = 0.009
    current.block.ep.capacity_factor = 8.0
    expected_blocks = [block.copy() for block in current.resolved_block_configs]
    for block, original in zip(expected_blocks, pretrained.resolved_block_configs):
        block.routed_experts_router.lb_loss_weight = original.routed_experts_router.lb_loss_weight
    original = pretrained.copy()
    vision_alignment._restore_pretraining_router_lb(current, pretrained)
    assert current.resolved_block_configs == expected_blocks
    assert pretrained == original


def test_restore_pretraining_router_lb_validates_all_layers_before_mutating(alignment_recipe):
    pretrained = alignment_recipe.set_router_coefficients(0.02)
    current = pretrained.copy()
    current.block.routed_experts_router.lb_loss_weight = 0
    current.block_overrides[3].routed_experts_router.top_k = 1
    before = current.copy()
    with pytest.raises(OLMoConfigurationError, match="layer 3"):
        vision_alignment._restore_pretraining_router_lb(current, pretrained)
    assert current == before


def test_restore_pretraining_router_lb_is_metadata_only(alignment_recipe, monkeypatch):
    alignment_recipe.set_router_coefficients(0.02)

    def forbidden(*args, **kwargs):
        pytest.fail("Restoration must not build models or corpus datasets")

    monkeypatch.setattr(OLMoDDPModelConfig, "build", forbidden)
    monkeypatch.setattr(MultimodalMixtureConfig, "build", forbidden)
    config = alignment_recipe.build(overrides=["--recipe.restore_pretraining_router_lb=true"])
    assert config.model.lm.block.routed_experts_router.lb_loss_weight == 0.02


@pytest.mark.parametrize("weight", [-0.01, float("nan"), float("inf")])
def test_restore_pretraining_router_lb_rejects_invalid_original_coefficients(
    alignment_recipe, weight
):
    pretrained = alignment_recipe.set_router_coefficients(0.02)
    current = pretrained.copy()
    pretrained.block_overrides[3].routed_experts_router.lb_loss_weight = weight
    with pytest.raises(OLMoConfigurationError, match="Invalid pretrained router LB"):
        vision_alignment._restore_pretraining_router_lb(current, pretrained)


def test_restore_pretraining_router_lb_without_routed_experts_is_noop(alignment_recipe):
    inherited = alignment_recipe.build()
    restored = alignment_recipe.build(overrides=["--recipe.restore_pretraining_router_lb=true"])
    assert restored.model == inherited.model
