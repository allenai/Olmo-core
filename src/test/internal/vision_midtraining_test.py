import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from olmo_core.config import Config
from olmo_core.data import NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.multimodal.alignment import MultimodalMixtureConfig
from olmo_core.data.multimodal.pixmo_cap import PixMoCapDatasetConfig
from olmo_core.data.multimodal.pretraining_replay import PretrainingReplayConfig
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDatasetConfig,
    SourceMixtureList,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal import (
    vision_alignment,
    vision_midtraining,
    vision_midtraining_data,
)
from olmo_core.internal.experiment import CliContext, SubCmd
from olmo_core.internal.vision_midtraining import (
    MixedMidtrainingExperimentConfig,
    build_config,
)
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.transformer import OLMoDDPModelConfig
from olmo_core.nn.vision import (
    Molmo2TokenIds,
    MultimodalLMConfig,
    VisionConnectorConfig,
    VisionEncoderConfig,
)
from olmo_core.optim import Scheduler
from olmo_core.train import Duration, LoadStrategy


@pytest.fixture
def mixed_recipe(tmp_path, monkeypatch):
    tokenizer = TokenizerConfig.dolma2()
    lm = OLMoDDPModelConfig(
        d_model=64,
        vocab_size=100352,
        n_layers=3,
        block=OLMoDDPTransformerBlockConfig(sequence_mixer=AttentionConfig(n_heads=4)),
        lm_head=LMHeadConfig(),
    )
    vision = VisionEncoderConfig()
    model = MultimodalLMConfig(
        lm=lm,
        vision=vision,
        connector=VisionConnectorConfig.from_vision_encoder(vision, output_dim=lm.d_model),
        image_patch_token_id=100280,
    )
    token_ids = Molmo2TokenIds(
        im_start_id=100278,
        im_end_id=100279,
        im_patch_id=100280,
        im_col_id=100281,
        low_res_im_start_id=100282,
        image_placeholder_id=100283,
        im_end_turn_id=100264,
    )
    revision = "5292e5d6c0f40b67cc765fe41bec991cf4345b5c"
    parent = tmp_path / "joint" / "step12000"
    parent.mkdir(parents=True)
    metadata = {
        "recipe": {"phase": "joint"},
        "model": model.as_config_dict(),
        "dataset": {"tokenizer": tokenizer.as_config_dict(), "tokenizer_revision": revision},
    }
    (parent / "config.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(
        MultimodalMixtureConfig, "build_tokenizer", Mock(return_value=(tokenizer, token_ids))
    )

    def visual_sources(sequence_length=8192, max_crops=8, **kwargs):
        return {
            name: PixMoCapDatasetConfig(
                dataset_path=f"{tmp_path}/{name}",
                max_sequence_length=sequence_length,
                max_crops=max_crops,
            )
            for name in vision_midtraining_data.DEFAULT_VISUAL_MEAN_LOSS_WEIGHTS
        }

    monkeypatch.setattr(vision_midtraining, "build_visual_sources", visual_sources)

    def build(*overrides):
        return build_config(
            CliContext(
                script="src/scripts/train/Mixed-Midtraining.py",
                cmd=SubCmd.dry_run,
                run_name="mixed-test",
                cluster="local",
                overrides=[
                    f"--recipe.parent_checkpoint={parent}",
                    f"--recipe.output_root={tmp_path}/outputs",
                    f"--recipe.work_dir={tmp_path}/cache",
                    *overrides,
                ],
            )
        )

    return SimpleNamespace(
        build=build, parent=parent, metadata=metadata, model=model, tokenizer=tokenizer
    )


def test_mixed_recipe_defaults_and_roundtrip(mixed_recipe):
    config = mixed_recipe.build()
    # ``as_config_dict`` omits the ``None`` launch of a local run; it is restored explicitly.
    restored = MixedMidtrainingExperimentConfig.from_dict(
        {"launch": None, **config.as_config_dict()}
    )
    assert restored == config
    assert config.recipe.text_loss_share == 0.9
    assert config.launch is None
    assert config.dataset.target_loss_mass["text_midtraining"] == 0.9
    assert len(config.dataset.sources) == 9
    assert config.train_module.source_loss_mass_targets == config.dataset.target_loss_mass
    assert config.data_loader.global_batch_size == 1048576
    assert config.data_loader.sequence_length == config.train_module.max_sequence_length == 8192
    assert config.train_module.rank_microbatch_size == 16384
    assert config.data_loader.global_batch_size // (16 * 16384) == 4
    assert config.data_loader.pack and config.data_loader.pack_buffer_size == 48
    assert config.data_loader.pack_max_crops == 64
    assert config.data_loader.source_groups is None
    assert config.data_loader.group_sequence_quotas is None
    assert config.train_module.loss_group_weights is None
    assert config.data_loader.prefetch_workers == 8
    assert config.data_loader.max_consecutive_data_errors == 0
    assert config.data_loader.max_total_data_errors == 0
    assert config.trainer.max_duration.value == 50000297984
    assert config.trainer.max_duration.unit == "tokens"
    assert config.trainer.load_path == str(mixed_recipe.parent)
    assert config.trainer.load_strategy == LoadStrategy.always
    assert config.trainer.load_optim_state is False
    assert config.trainer.load_trainer_state is False
    assert not config.trainer.save_overwrite


@pytest.mark.parametrize("text_loss_share", [0.9, 1.0])
def test_step_zero_resume_does_not_resave_checkpoint(mixed_recipe, monkeypatch, text_loss_share):
    config = mixed_recipe.build(f"--recipe.text_loss_share={text_loss_share}")
    callback = config.trainer.callbacks["checkpointer"]
    trainer = Mock(global_step=0, checkpoint_loaded=True, save_folder=config.trainer.save_folder)
    path = f"{trainer.save_folder}/step0"
    trainer.checkpointer.find_checkpoints.return_value = [(0, path)]
    callback.trainer = trainer
    monkeypatch.setattr("olmo_core.train.callbacks.checkpointer.is_distributed", lambda: False)
    monkeypatch.setattr("olmo_core.train.callbacks.checkpointer.get_rank", lambda: 0)
    monkeypatch.setattr("olmo_core.train.callbacks.checkpointer.broadcast_object", lambda x: x)

    callback.pre_train()

    trainer.save_checkpoint.assert_not_called()
    trainer.save_checkpoint_async.assert_not_called()
    assert callback._checkpoints == [path]


@pytest.mark.parametrize("text_loss_share", [0.9, 1.0])
def test_checkpoint_loading_cannot_be_disabled(mixed_recipe, text_loss_share):
    with pytest.raises(OLMoConfigurationError, match="trainer.no_checkpoints"):
        mixed_recipe.build(
            f"--recipe.text_loss_share={text_loss_share}", "--trainer.no_checkpoints=true"
        )


@pytest.mark.parametrize("text_loss_share", [0.9, 1.0])
def test_checkpoint_writes_can_be_disabled(mixed_recipe, text_loss_share):
    config = mixed_recipe.build(
        f"--recipe.text_loss_share={text_loss_share}",
        "--trainer.callbacks.checkpointer.enabled=false",
    )
    assert config.trainer.callbacks["checkpointer"].enabled is False
    assert config.trainer.no_checkpoints is False
    assert config.trainer.load_path == str(mixed_recipe.parent)
    assert config.trainer.load_strategy == LoadStrategy.always
    assert config.trainer.load_optim_state is False
    assert config.trainer.load_trainer_state is False


def test_mixed_packing_override_preserves_loss_allocation(mixed_recipe):
    config = mixed_recipe.build()
    smaller_packs = mixed_recipe.build("--data_loader.pack_max_crops=16")
    assert smaller_packs.data_loader.pack_max_crops == 16
    assert smaller_packs.dataset == config.dataset
    assert smaller_packs.train_module == config.train_module


def test_hard_stop_preserves_full_schedule_and_resume_configuration(mixed_recipe):
    first = mixed_recipe.build(
        '--trainer.hard_stop={"value":50,"unit":"steps"}',
        "--trainer.callbacks.checkpointer.fixed_steps=[50]",
    )
    continued = mixed_recipe.build(
        '--trainer.hard_stop={"value":500,"unit":"steps"}',
        "--trainer.callbacks.checkpointer.fixed_steps=[50]",
    )
    assert first.trainer.hard_stop.value == 50
    assert continued.trainer.hard_stop.value == 500
    assert first.trainer.max_duration == continued.trainer.max_duration
    assert continued.trainer.max_duration.value == 50_000_297_984
    assert first.train_module.scheduler == continued.train_module.scheduler
    assert first.dataset == continued.dataset
    assert first.data_loader == continued.data_loader
    assert first.trainer.save_folder == continued.trainer.save_folder
    assert first.trainer.load_path == continued.trainer.load_path
    assert continued.trainer.callbacks["checkpointer"].fixed_steps == [50]


def test_mixed_recipe_keeps_explicit_61_source_text_allocation(mixed_recipe, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Config construction must not allocate sources or open training arrays")

    monkeypatch.setattr(SourceMixtureDatasetConfig, "build", forbidden)
    monkeypatch.setattr(NumpyFSLDatasetConfig, "build", forbidden)
    monkeypatch.setattr(PretrainingReplayConfig, "build", forbidden)
    monkeypatch.setattr(SourceMixtureConfig, "resolved_paths", property(forbidden))
    config = mixed_recipe.build()
    replay = config.dataset.sources["text_midtraining"]
    assert isinstance(replay, PretrainingReplayConfig)
    assert replay.checkpoint is None and replay.split == "all"
    text = replay.dataset
    assert text.tokenizer == mixed_recipe.tokenizer
    assert text.sequence_length == text.max_target_sequence_length == 8192
    assert not text.generate_doc_lengths
    assert text.source_mixture_config.requested_tokens == 50000297984
    assert text.source_mixture_config.global_batch_size == 1048576
    assert text.source_mixture_config.seed == 1337
    source_list = text.source_mixture_config.source_list
    assert len(source_list.sources) == 61
    names = {source.source_name for source in source_list.sources}
    assert {"instruction-new-format", "flan"} <= names
    assert text.instance_filter_config.repetition_min_period == 1
    assert text.instance_filter_config.repetition_max_period == 13
    assert text.instance_filter_config.repetition_max_count == 32
    assert config.dataset.mean_loss_weight["text_midtraining"] == 8191


def test_t100_keeps_text_and_nonvision_optimization_without_visual_access(
    mixed_recipe, monkeypatch
):
    mixed = mixed_recipe.build()

    def forbidden(*args, **kwargs):
        pytest.fail("Text-only recipe must not construct or calibrate visual sources")

    monkeypatch.setattr(vision_midtraining, "build_visual_sources", forbidden)
    text = mixed_recipe.build("--recipe.text_loss_share=1")
    assert list(text.dataset.sources) == ["text_midtraining"]
    assert text.dataset.target_loss_mass == {"text_midtraining": 1.0}
    assert text.dataset.mean_loss_weight == {"text_midtraining": 8191}
    assert text.dataset.sources["text_midtraining"] == mixed.dataset.sources["text_midtraining"]
    assert text.data_loader == mixed.data_loader
    assert text.train_module.rank_microbatch_size == mixed.train_module.rank_microbatch_size
    assert text.model == mixed.model
    expected_optim = mixed.train_module.optim.copy()
    expected_optim.group_overrides[1].opts["lr"] = 0.0
    assert text.train_module.optim == expected_optim
    assert text.train_module.scheduler == mixed.train_module.scheduler
    assert text.train_module.freeze_params == ["vision.*"]
    assert not text.train_module.vision_activation_checkpointing
    assert (
        MixedMidtrainingExperimentConfig.from_dict({"launch": None, **text.as_config_dict()})
        == text
    )


def test_mixed_recipe_preserves_parent_architecture_and_router(mixed_recipe):
    mixed_recipe.model.lm.block.routed_experts_router = MoERouterConfigV2(
        d_model=64, num_experts=8, top_k=2, lb_loss_weight=0.005, z_loss_weight=0.0002
    )
    mixed_recipe.metadata["model"] = mixed_recipe.model.as_config_dict()
    (mixed_recipe.parent / "config.json").write_text(json.dumps(mixed_recipe.metadata))
    config = mixed_recipe.build()
    parent = json.loads((mixed_recipe.parent / "config.json").read_text())
    assert parent == json.loads(json.dumps(mixed_recipe.metadata))
    assert config.model.lm.n_layers == mixed_recipe.model.lm.n_layers
    assert config.model.lm.d_model == mixed_recipe.model.lm.d_model
    assert config.model.connector == mixed_recipe.model.connector
    assert config.model.vision == mixed_recipe.model.vision
    assert (
        config.model.lm.block.routed_experts_router
        == mixed_recipe.model.lm.block.routed_experts_router
    )


@pytest.mark.parametrize("text_loss_share", [0.5, 0.9, 1.0])
def test_mixed_recipe_optimizer_contract(mixed_recipe, text_loss_share):
    config = mixed_recipe.build(f"--recipe.text_loss_share={text_loss_share}")
    module = config.train_module
    text_only = text_loss_share == 1.0
    assert module.freeze_params == (["vision.*"] if text_only else [])
    assert module.train_embedding_rows is None
    assert module.vision_activation_checkpointing is not text_only
    assert not module.connector_activation_checkpointing and module.response_logits_only
    assert module.ep_config.degree == 8 and module.compile_model
    assert module.optim.lr == 1e-5 and module.optim.weight_decay == 0.1
    assert module.optim.betas == (0.9, 0.95) and module.optim.eps == 1e-8
    assert module.optim.clip_grad_norm_by_scheduler_group
    assert module.optim.max_grad_norm == module.max_grad_norm == 1.0
    assert module.optim.sigma_factor == 12
    assert module.optim.group_overrides[0].opts == {
        "lr": 2e-5,
        "weight_decay": 0.0,
        "scheduler_name": "connector",
    }
    assert module.optim.group_overrides[1].opts == {
        "lr": 0.0 if text_only else 1e-6,
        "weight_decay": 0.0,
        "scheduler_name": "vision",
    }
    schedules = [module.scheduler.default, *module.scheduler.schedulers.values()]
    for schedule in schedules:
        assert schedule.units == "tokens"
        assert schedule.warmup == 209715200
        assert schedule.t_max == 50000297984
        assert schedule.alpha_f == 0.1


def test_frozen_vision_control_uses_component_overrides(mixed_recipe):
    config = mixed_recipe.build(
        '--train_module.freeze_params=["vision.*"]',
        "--train_module.vision_activation_checkpointing=false",
    )
    assert config.train_module.freeze_params == ["vision.*"]
    assert not config.train_module.vision_activation_checkpointing
    assert config.recipe.text_loss_share == 0.9


def test_short_stop_does_not_change_data_or_schedule(mixed_recipe):
    baseline = mixed_recipe.build()
    short = mixed_recipe.build(
        "--trainer.max_duration.unit=steps", "--trainer.max_duration.value=2"
    )
    assert short.dataset == baseline.dataset
    assert short.train_module.scheduler == baseline.train_module.scheduler
    assert short.trainer.max_duration.value == 2


@pytest.mark.parametrize("microbatch_sequences", [1, 4])
def test_microbatch_override_preserves_global_batch_and_schedule(
    mixed_recipe, microbatch_sequences
):
    baseline = mixed_recipe.build()
    config = mixed_recipe.build(
        f"--train_module.rank_microbatch_size={microbatch_sequences * 8192}"
    )
    assert config.train_module.rank_microbatch_size == microbatch_sequences * 8192
    assert config.data_loader == baseline.data_loader
    assert config.dataset == baseline.dataset
    assert config.train_module.optim == baseline.train_module.optim
    assert config.train_module.scheduler == baseline.train_module.scheduler
    assert config.trainer.max_duration == baseline.trainer.max_duration


@pytest.mark.parametrize("share", ["true", "-0.1", "1.1", "nan", "inf"])
def test_invalid_text_share_is_rejected(mixed_recipe, share):
    with pytest.raises((OLMoConfigurationError, ValueError, TypeError)):
        mixed_recipe.build(f"--recipe.text_loss_share={share}")


def test_output_cannot_overlap_parent(mixed_recipe):
    with pytest.raises(OLMoConfigurationError, match="output|folder|parent"):
        mixed_recipe.build(f"--trainer.save_folder={mixed_recipe.parent}")


def test_context_override_requires_visual_recalibration(mixed_recipe):
    with pytest.raises(OLMoConfigurationError, match="calibrat|mean_loss_weight"):
        mixed_recipe.build("--recipe.sequence_length=4096")


def test_component_overrides_are_supported_and_update_loss_targets(mixed_recipe):
    config = mixed_recipe.build(
        "--train_module.optim.lr=0.00003",
        "--dataset.mean_loss_weight.pixmo_cap=500",
        "--data_loader.prefetch_workers=4",
    )
    assert config.train_module.optim.lr == 3e-5
    assert config.data_loader.prefetch_workers == 4
    assert config.dataset.mean_loss_weight["pixmo_cap"] == 500
    assert config.train_module.source_loss_mass_targets == config.dataset.target_loss_mass
    assert config.dataset.target_loss_mass["text_midtraining"] == 0.9
    assert sum(config.dataset.target_loss_mass.values()) == pytest.approx(1)


def test_only_native_config_classes_are_serialized(mixed_recipe):
    config = mixed_recipe.build()

    def inspect(value):
        if isinstance(value, Config):
            assert type(value).__module__.startswith("olmo_core.")

    config.apply(inspect)
    checkpointer = config.trainer.callbacks["checkpointer"]
    assert checkpointer.fixed_steps is None
    assert checkpointer.save_interval == 10000
    assert checkpointer.ephemeral_save_interval == 500
    assert checkpointer.max_checkpoints == 2
    assert checkpointer.pre_train_checkpoint is None
    assert not config.train_module.reset_optimizer_states_on_load
    assert not config.train_module.reset_optimizer_states_on_resume
    assert config.trainer.callbacks["wandb"].auto_resume


def test_legacy_alignment_parent_resolves_tokenizer_from_ancestry(mixed_recipe, monkeypatch):
    metadata = mixed_recipe.metadata.copy()
    metadata.pop("recipe")
    dataset = metadata.pop("dataset")
    metadata["vision_alignment"] = {"phase": "joint"}
    metadata["artifacts"] = {
        "base_checkpoint": "/original/pretraining/stepN",
        "tokenizer_id": mixed_recipe.tokenizer.identifier,
        "tokenizer_revision": dataset["tokenizer_revision"],
    }

    def read(checkpoint):
        if checkpoint == str(mixed_recipe.parent):
            return metadata
        assert checkpoint == "/original/pretraining/stepN"
        return {"dataset": {"tokenizer": mixed_recipe.tokenizer.as_config_dict()}}

    monkeypatch.setattr(vision_midtraining, "_read_checkpoint_config", read)
    config = mixed_recipe.build()
    assert config.dataset.tokenizer == mixed_recipe.tokenizer
    assert config.dataset.tokenizer_revision == dataset["tokenizer_revision"]
    assert config.pretraining_checkpoint == "/original/pretraining/stepN"


@pytest.mark.parametrize("override", ["--recipe.text_loss_share", "--recipe.text-loss-share=true"])
def test_boolean_share_cannot_bypass_numeric_validation(mixed_recipe, override):
    with pytest.raises(OLMoConfigurationError, match="boolean"):
        mixed_recipe.build(override)


def test_native_dashed_overrides_and_batch_sizing(mixed_recipe):
    config = mixed_recipe.build(
        "--recipe.text-loss-share=1", "--data-loader.global-batch-size=2097152"
    )
    budget = ((50_000_000_000 + 2097151) // 2097152) * 2097152
    assert config.data_loader.global_batch_size == 2097152
    assert config.trainer.max_duration.value == budget
    source_mix = config.dataset.sources["text_midtraining"].dataset.source_mixture_config
    assert source_mix.requested_tokens == budget
    assert source_mix.global_batch_size == 2097152
    assert config.train_module.scheduler.default.t_max == budget
    assert config.train_module.scheduler.default.warmup == 200 * 2097152


def test_changed_source_requires_explicit_calibration(mixed_recipe):
    change = "--dataset.sources.pixmo_cap.max_crops=4"
    with pytest.raises(OLMoConfigurationError, match="pixmo_cap"):
        mixed_recipe.build(change)
    config = mixed_recipe.build(change, "--dataset.mean-loss-weight.pixmo_cap=400")
    assert config.dataset.sources["pixmo_cap"].max_crops == 4
    assert config.dataset.mean_loss_weight["pixmo_cap"] == 400


def test_visual_groups_support_source_replacement_without_default_artifacts(
    mixed_recipe, monkeypatch
):
    original = mixed_recipe.build()
    sources = {
        "text_midtraining": original.dataset.sources["text_midtraining"].as_config_dict(),
        "caption": original.dataset.sources["pixmo_cap"].as_config_dict(),
        "document": original.dataset.sources["ocr_document"].as_config_dict(),
    }

    def forbidden(*args, **kwargs):
        pytest.fail("A fully supplied source map must not require default visual artifacts")

    monkeypatch.setattr(vision_midtraining, "build_visual_sources", forbidden)
    config = mixed_recipe.build(
        f"--dataset.sources={json.dumps(sources)}",
        '--dataset.mean_loss_weight={"text_midtraining":8191,"caption":100,"document":200}',
        '--recipe.visual_example_weights={"caption":1}',
        '--recipe.visual_loss_shares={"document":0.25}',
    )
    assert config.dataset.target_loss_mass == pytest.approx(
        {"text_midtraining": 0.9, "caption": 0.075, "document": 0.025}
    )
    assert config.train_module.source_loss_mass_targets == config.dataset.target_loss_mass
    assert set(config.dataset.sampling_weights()) == set(sources)


def test_fixed_sequence_quotas_cannot_override_loss_share_policy(mixed_recipe):
    with pytest.raises(OLMoConfigurationError, match="quotas"):
        mixed_recipe.build('--data_loader.group_sequence_quotas={"text":128}')


def test_complete_dataset_override_includes_its_calibration(mixed_recipe, monkeypatch):
    original = mixed_recipe.build()

    def forbidden(*args, **kwargs):
        pytest.fail("A complete dataset override must not require default visual artifacts")

    monkeypatch.setattr(vision_midtraining, "build_visual_sources", forbidden)
    config = mixed_recipe.build(f"--dataset={json.dumps(original.dataset.as_config_dict())}")
    assert config.dataset == original.dataset


def test_complete_dataset_override_rejects_conflicting_loss_targets(mixed_recipe):
    dataset = mixed_recipe.build().dataset.as_config_dict()
    dataset["target_loss_mass"]["text_midtraining"] = 0.5
    with pytest.raises(OLMoConfigurationError, match="recipe.text_loss_share"):
        mixed_recipe.build(f"--dataset={json.dumps(dataset)}")


def test_whole_recipe_override_is_rejected_before_reading_parent(mixed_recipe, monkeypatch):
    recipe = mixed_recipe.build().recipe.as_config_dict()
    recipe["max_tokens"] = 1000
    read = Mock(side_effect=AssertionError("Parent must not be read"))
    monkeypatch.setattr(vision_midtraining, "_read_checkpoint_config", read)
    with pytest.raises(OLMoConfigurationError, match="dotted --recipe.FIELD"):
        mixed_recipe.build(f"--recipe={json.dumps(recipe)}")
    read.assert_not_called()


@pytest.mark.parametrize(
    "mismatch,error",
    [("missing", ValueError), ("extra", ValueError), ("sequence_length", OLMoConfigurationError)],
)
def test_inconsistent_visual_sources_are_rejected(mixed_recipe, mismatch, error):
    dataset = mixed_recipe.build().dataset.as_config_dict()
    if mismatch == "missing":
        dataset["sources"].pop("ocr_document")
    elif mismatch == "extra":
        dataset["sources"]["extra"] = dataset["sources"]["ocr_document"].copy()
        dataset["mean_loss_weight"]["extra"] = dataset["mean_loss_weight"]["ocr_document"]
    else:
        dataset["sources"]["ocr_document"]["max_sequence_length"] = 4096
    with pytest.raises(error):
        mixed_recipe.build(f"--dataset={json.dumps(dataset)}")


@pytest.mark.parametrize("text_loss_share", [0.9, 1.0])
def test_launch_uses_standard_two_node_alignment_settings(
    mixed_recipe, monkeypatch, text_loss_share
):
    from gantry.api import GitRepoState

    factory = Mock(
        return_value=BeakerLaunchConfig(
            name="mixed-test",
            cmd=["train"],
            git=GitRepoState(
                repo="allenai/OLMo-core",
                repo_url="https://github.com/allenai/OLMo-core",
                ref="a" * 40,
                branch="vision-moe",
            ),
        )
    )
    monkeypatch.setattr(vision_alignment, "build_launch_config", factory)
    config = build_config(
        CliContext(
            script="src/scripts/train/Mixed-Midtraining.py",
            cmd=SubCmd.launch,
            run_name="mixed-test",
            cluster="ai2/holmes",
            overrides=[
                f"--recipe.parent_checkpoint={mixed_recipe.parent}",
                f"--recipe.text_loss_share={text_loss_share}",
                "--recipe.work_dir=/tmp/mixed-data-cache",
            ],
        )
    )
    assert factory.call_args.kwargs["workspace"] == "ai2/oe-olmo3p5-mt"
    assert factory.call_args.kwargs["budget"] == "ai2/oe-other"
    assert config.launch.num_nodes == 2
    assert factory.call_args.kwargs["num_nodes"] == 2
    assert config.data_loader.sequence_length == 8192
    assert config.data_loader.global_batch_size == 128 * 8192
    assert config.train_module.rank_microbatch_size == 2 * 8192
    assert (
        config.data_loader.global_batch_size // (16 * config.train_module.rank_microbatch_size) == 4
    )
    assert config.data_loader.pack_max_crops == 64
    assert factory.call_args.kwargs["step_soft_timeout"] is None
    assert config.launch.priority == "urgent"
    assert config.launch.min_runtime == "8h"
    assert config.launch.shared_memory == "32GiB"
    assert config.launch.step_timeout is None
    assert config.launch.cmd == ["train"]
    assert config.launch.aws_config_secret is None
    assert config.launch.aws_credentials_secret is None
    env = {item.name: item.value for item in config.launch.env_vars}
    assert env["OLMO_CORE_DATA_VERIFICATION_CACHE_DIR"] == "/tmp/mixed-data-cache/data-verification"
    assert "OLMO_CORE_FS_CACHE_DIR" not in env
    secrets = {item.name: item.secret for item in config.launch.env_secrets}
    assert secrets["BEAKER_TOKEN"] == "jasonr_BEAKER_TOKEN"
    assert secrets["WANDB_API_KEY"] == "jasonr_WANDB_API_KEY"
    assert len(secrets) == len(config.launch.env_secrets)


@pytest.mark.parametrize(
    "override",
    [
        "--recipe.visual_example_weights.pixmo_cap=true",
        '--recipe.visual_example_weights={"pixmo_cap":true}',
        "--recipe.visual_loss_shares.ocr_document=true",
        "--dataset.mean_loss_weight.pixmo_cap=true",
        '--dataset.mean_loss_weight={"pixmo_cap":true}',
    ],
)
def test_numeric_visual_weights_reject_boolean_overrides(mixed_recipe, override):
    with pytest.raises(OLMoConfigurationError, match="boolean"):
        mixed_recipe.build(override)


def test_native_parent_with_another_tokenizer_requires_explicit_text(mixed_recipe):
    tokenizer = mixed_recipe.tokenizer.copy()
    tokenizer.identifier = "test/compatible-tokenizer"
    mixed_recipe.metadata["dataset"]["tokenizer"] = tokenizer.as_config_dict()
    (mixed_recipe.parent / "config.json").write_text(json.dumps(mixed_recipe.metadata))
    with pytest.raises(OLMoConfigurationError, match="Dolma2|text_dataset"):
        mixed_recipe.build("--recipe.text_loss_share=1")
    text = NumpyFSLDatasetConfig(
        paths=["/unused/text.npy"], tokenizer=tokenizer, sequence_length=8192
    )
    config = mixed_recipe.build(
        "--recipe.text_loss_share=1",
        f"--recipe.text_dataset={json.dumps(text.as_config_dict())}",
    )
    assert config.dataset.tokenizer == tokenizer
    assert config.dataset.sources["text_midtraining"].dataset == text


def test_masked_text_requires_calibration_only_in_mixed_runs(mixed_recipe):
    text = NumpyFSLDatasetConfig(
        paths=["/unused/text.npy"],
        label_mask_paths=["/unused/masks.npy"],
        tokenizer=mixed_recipe.tokenizer.copy(),
        sequence_length=8192,
    )
    override = f"--recipe.text_dataset={json.dumps(text.as_config_dict())}"
    with pytest.raises(OLMoConfigurationError, match="Masked text"):
        mixed_recipe.build(override)
    mixed = mixed_recipe.build(override, "--dataset.mean_loss_weight.text_midtraining=2048")
    assert mixed.dataset.mean_loss_weight["text_midtraining"] == 2048
    assert mixed.dataset.sources["text_midtraining"].dataset == text
    text_only = mixed_recipe.build(override, "--recipe.text_loss_share=1")
    assert text_only.dataset.mean_loss_weight == {"text_midtraining": 1}
    assert text_only.dataset.sources["text_midtraining"].dataset == text


def test_custom_artifact_roots_do_not_reopen_default_provenance(mixed_recipe, monkeypatch):
    original_factory = vision_midtraining.build_visual_sources
    means = dict(vision_midtraining_data.DEFAULT_VISUAL_MEAN_LOSS_WEIGHTS)
    means["text_midtraining"] = 8191

    def custom_factory(*args, **kwargs):
        assert kwargs["alignment_artifact_root"] == "/custom/alignment"
        assert kwargs["midtraining_artifact_root"] == "/custom/mixed"
        return original_factory(*args, **kwargs)

    monkeypatch.setattr(vision_midtraining, "build_visual_sources", custom_factory)
    config = mixed_recipe.build(
        "--recipe.alignment_artifact_root=/custom/alignment",
        "--recipe.midtraining_artifact_root=/custom/mixed",
        f"--dataset.mean_loss_weight={json.dumps(means)}",
    )
    assert config.dataset.target_loss_mass["text_midtraining"] == 0.9


TEXT_FIXTURE = Path(__file__).parent.parent / "fixtures" / "olmo35_text_midtraining_config.json"


@pytest.fixture
def text_config(tmp_path):
    """The text team's OLMo 3.5 mid-training config, with a small source mixture as its data."""
    text = json.loads(TEXT_FIXTURE.read_text())
    text["dataset"] = NumpyFSLDatasetConfig.from_src_mix(
        SourceMixtureDatasetConfig(
            source_list=SourceMixtureList(
                sources=[
                    SourceMixtureConfig(
                        source_name="web", target_ratio=0.7, paths=["gs://bucket/web/*.npy"]
                    ),
                    SourceMixtureConfig(
                        source_name="code", target_ratio=0.3, paths=["gs://bucket/code/*.npy"]
                    ),
                ]
            ),
            requested_tokens=text["trainer"]["max_duration"]["value"],
            global_batch_size=text["data_loader"]["global_batch_size"],
            processes=16,
            seed=1387822106,
        ),
        tokenizer=TokenizerConfig.dolma2(),
        sequence_length=8192,
        work_dir="/text-team/dataset-cache",
    ).as_config_dict()
    path = tmp_path / "text_config.json"
    path.write_text(json.dumps(text))
    return SimpleNamespace(path=path, config=text)


def test_text_config_inherits_the_text_midtraining_recipe(mixed_recipe, text_config):
    text = text_config.config
    config = mixed_recipe.build(f"--recipe.text_config={text_config.path}")
    text_loader, text_module = text["data_loader"], text["train_module"]
    batch = text_loader["global_batch_size"]
    budget = -(-text["trainer"]["max_duration"]["value"] // batch) * batch

    # Batch, budget and loader workers.
    assert config.data_loader.global_batch_size == batch == 1024 * 8192
    assert config.data_loader.prefetch_workers == text_loader["num_workers"]
    assert config.trainer.max_duration == Duration.tokens(budget)

    # The text mixture, with its cache in this recipe's work directory.
    replay = config.dataset.sources["text_midtraining"]
    expected = NumpyFSLDatasetConfig.from_dict(text["dataset"])
    assert (
        replay.dataset.source_mixture_config.source_list
        == expected.source_mixture_config.source_list
    )
    assert replay.dataset.source_mixture_config.requested_tokens == budget
    assert replay.dataset.work_dir == config.recipe.work_dir != expected.work_dir
    assert config.dataset.target_loss_mass["text_midtraining"] == 0.9

    # LM optimization from the text config; connector and vision scale with the LM rate.
    optim, lr = config.train_module.optim, text_module["optim"]["lr"]
    assert optim.lr == lr
    groups = {tuple(group.params): group.opts for group in optim.group_overrides}
    # From an alignment checkpoint: connector and vision at half the LM's LR, weight decay as the
    # text config's (no override), on the text config's schedule.
    assert groups[("*connector.*",)]["lr"] == lr / 2
    assert groups[("*vision.*",)]["lr"] == lr / 2
    assert "weight_decay" not in groups[("*connector.*",)]
    assert "weight_decay" not in groups[("*vision.*",)]
    for group in text_module["optim"]["group_overrides"]:
        assert groups[tuple(group["params"])] == group["opts"]
    for name in ("betas", "eps", "weight_decay", "sigma_factor", "compile"):
        assert getattr(optim, name) == (
            tuple(text_module["optim"][name]) if name == "betas" else text_module["optim"][name]
        )
    scheduler = Scheduler.from_dict(text_module["scheduler"])
    assert config.train_module.scheduler.default == scheduler
    assert all(s == scheduler for s in config.train_module.scheduler.schedulers.values())
    assert config.train_module.ep_config is None
    assert config.train_module.z_loss_multiplier == text_module["z_loss_multiplier"]
    assert config.train_module.compile_model == text_module["compile_model"]

    # Checkpoint cadence from the text run.
    checkpointer = config.trainer.callbacks["checkpointer"]
    text_checkpointer = text["trainer"]["callbacks"]["checkpointer"]
    assert checkpointer.save_interval == text_checkpointer["save_interval"]
    assert checkpointer.ephemeral_save_interval == text_checkpointer["ephemeral_save_interval"]


def test_text_config_launch_keeps_two_nodes(mixed_recipe, text_config, monkeypatch):
    from gantry.api import GitRepoState

    factory = Mock(
        return_value=BeakerLaunchConfig(
            name="mixed-test",
            cmd=["train"],
            git=GitRepoState(
                repo="allenai/OLMo-core",
                repo_url="https://github.com/allenai/OLMo-core",
                ref="a" * 40,
                branch="vision",
            ),
        )
    )
    monkeypatch.setattr(vision_alignment, "build_launch_config", factory)
    config = build_config(
        CliContext(
            script="src/scripts/train/Mixed-Midtraining.py",
            cmd=SubCmd.launch,
            run_name="mixed-test",
            cluster="ai2/holmes",
            overrides=[
                f"--recipe.parent_checkpoint={mixed_recipe.parent}",
                f"--recipe.text_config={text_config.path}",
                "--recipe.work_dir=/tmp/mixed-data-cache",
            ],
        )
    )
    text_launch = text_config.config["launch"]
    assert config.launch.num_nodes == 2
    assert config.launch.beaker_image == text_launch["beaker_image"]
    assert factory.call_args.kwargs["workspace"] == "ai2/oe-olmo3p5-mt"


def _write_text_lm_checkpoint(path, lm):
    path.mkdir(parents=True)
    config = {
        "model": lm.as_config_dict(),
        "dataset": {"tokenizer": TokenizerConfig.dolma2().as_config_dict()},
    }
    (path / "config.json").write_text(json.dumps(config))
    return path


@pytest.mark.parametrize("phase", ["bridge", "perception", "joint"])
def test_parent_can_be_any_alignment_stage(mixed_recipe, tmp_path, phase):
    """Any stage's endpoint can start midtraining (stage ablations). Bridge and perception switch
    router load balancing off while the LM is frozen; midtraining trains the LM, so it restores
    the text LM's coefficients. Joint already restored them."""
    pretrained = mixed_recipe.model.lm.copy()
    pretrained.block.routed_experts_router = MoERouterConfigV2(
        d_model=64, num_experts=8, top_k=2, lb_loss_weight=0.005
    )
    ancestry = _write_text_lm_checkpoint(tmp_path / "text-lm" / "step100", pretrained)
    mixed_recipe.model.lm.block.routed_experts_router = MoERouterConfigV2(
        d_model=64, num_experts=8, top_k=2, lb_loss_weight=0.0
    )
    mixed_recipe.metadata["model"] = mixed_recipe.model.as_config_dict()
    mixed_recipe.metadata["recipe"] = {"phase": phase}
    mixed_recipe.metadata["pretraining_checkpoint"] = str(ancestry)
    (mixed_recipe.parent / "config.json").write_text(json.dumps(mixed_recipe.metadata))

    config = mixed_recipe.build()

    assert config.alignment_phase == phase
    assert config.pretraining_checkpoint == str(ancestry)
    router = config.model.lm.block.routed_experts_router
    assert router.lb_loss_weight == (0.0 if phase == "joint" else 0.005)
    assert config.trainer.load_path == str(mixed_recipe.parent)
    assert "initialize_multimodal" not in config.trainer.callbacks


def test_parent_must_be_an_alignment_checkpoint(mixed_recipe):
    mixed_recipe.metadata["recipe"] = {"phase": "stage1"}
    (mixed_recipe.parent / "config.json").write_text(json.dumps(mixed_recipe.metadata))
    with pytest.raises(OLMoConfigurationError, match="vision-alignment checkpoint"):
        mixed_recipe.build()


def _build_from_text_lm(tmp_path, *overrides):
    return build_config(
        CliContext(
            script="src/scripts/train/Mixed-Midtraining.py",
            cmd=SubCmd.dry_run,
            run_name="mixed-test",
            cluster="local",
            overrides=[
                f"--recipe.output_root={tmp_path}/outputs",
                f"--recipe.work_dir={tmp_path}/cache",
                *overrides,
            ],
        )
    )


def test_midtraining_without_alignment_starts_from_the_text_lm(mixed_recipe, tmp_path):
    lm = mixed_recipe.model.lm
    checkpoint = _write_text_lm_checkpoint(tmp_path / "text-lm" / "step100", lm)

    config = _build_from_text_lm(tmp_path, f"--recipe.pretraining_checkpoint={checkpoint}")

    assert config.alignment_phase is None
    assert config.pretraining_checkpoint == str(checkpoint)
    assert config.model.lm.d_model == lm.d_model and config.model.lm.n_layers == lm.n_layers
    assert config.model.image_patch_token_id == 100280
    expected = MultimodalLMConfig.molmo2_vision_stack(config.model.lm, image_patch_token_id=100280)
    expected.compile_loss = True  # the recipe's default, on top of alignment's stack
    assert config.model == expected
    # The initialization callback loads the LM and the vision encoder; resume uses the run's own
    # checkpoints.
    assert config.trainer.load_path is None
    assert config.trainer.load_strategy == LoadStrategy.if_available
    init = config.trainer.callbacks["initialize_multimodal"]
    assert init.language_checkpoint == str(checkpoint)
    assert init.vision_model_id == config.recipe.vision_model_id
    assert init.image_token_ids == [100278, 100279, 100280, 100281, 100282, 100283]
    # Everything trains, as when starting from an alignment checkpoint.
    assert config.train_module.freeze_params == []
    assert config.dataset.target_loss_mass["text_midtraining"] == 0.9


@pytest.mark.parametrize("both", [True, False])
def test_exactly_one_starting_checkpoint(mixed_recipe, tmp_path, both):
    checkpoint = _write_text_lm_checkpoint(tmp_path / "text-lm" / "step100", mixed_recipe.model.lm)
    overrides = [f"--recipe.parent_checkpoint={mixed_recipe.parent}"] if both else []
    if both:
        overrides.append(f"--recipe.pretraining_checkpoint={checkpoint}")
    with pytest.raises(OLMoConfigurationError, match="exactly one"):
        _build_from_text_lm(tmp_path, *overrides)


def test_text_config_in_memory_matches_the_file(mixed_recipe, text_config):
    """A launcher that builds the text config itself (scaling-ladders) gets the same result as
    the saved config.json."""
    from_file = mixed_recipe.build(f"--recipe.text_config={text_config.path}")
    in_memory = build_config(
        CliContext(
            script="src/scripts/train/Mixed-Midtraining.py",
            cmd=SubCmd.dry_run,
            run_name="mixed-test",
            cluster="local",
            overrides=[
                f"--recipe.parent_checkpoint={mixed_recipe.parent}",
                f"--recipe.output_root={from_file.recipe.output_root}",
                f"--recipe.work_dir={from_file.recipe.work_dir}",
            ],
        ),
        text_config=text_config.config,
    )
    in_memory.recipe.text_config = from_file.recipe.text_config
    assert in_memory == from_file


def test_text_config_cannot_come_from_both(mixed_recipe, text_config):
    with pytest.raises(OLMoConfigurationError, match="in memory or as recipe.text_config"):
        build_config(
            CliContext(
                script="src/scripts/train/Mixed-Midtraining.py",
                cmd=SubCmd.dry_run,
                run_name="mixed-test",
                cluster="local",
                overrides=[
                    f"--recipe.parent_checkpoint={mixed_recipe.parent}",
                    f"--recipe.text_config={text_config.path}",
                ],
            ),
            text_config=text_config.config,
        )


def test_a_launcher_can_supply_its_own_launch_config(mixed_recipe, text_config):
    from gantry.api import GitRepoState

    theirs = BeakerLaunchConfig(
        name="ladders-mixed",
        cmd=["ladders/olmoe3/workloads/mixed_midtraining.py", "train"],
        num_nodes=1,
        git=GitRepoState(
            repo="allenai/scaling-ladders",
            repo_url="https://github.com/allenai/scaling-ladders",
            ref="a" * 40,
            branch="main",
        ),
    )
    config = build_config(
        CliContext(
            script="ladders/olmoe3/workloads/mixed_midtraining.py",
            cmd=SubCmd.dry_run,
            run_name="mixed-test",
            cluster="ai2/holmes",
            overrides=[f"--recipe.parent_checkpoint={mixed_recipe.parent}"],
        ),
        text_config=text_config.config,
        launch=theirs,
    )
    assert config.launch.cmd == theirs.cmd and config.launch.git == theirs.git
    assert config.launch.num_nodes == 2 and theirs.num_nodes == 1


@pytest.mark.parametrize("visual_data", ["stage1_v3", "midtraining"])
def test_stage1_v3_vision_installs_its_packages(
    mixed_recipe, text_config, monkeypatch, visual_data
):
    """The v3 Stage-1 sources render PDFs and read HDF5, which the text image lacks; with them the
    launch installs the packages alignment installs for its stage1_v3 data."""
    from gantry.api import GitRepoState

    from olmo_core.internal.vision_alignment import _STAGE1_V3_POST_SETUP
    from olmo_core.internal.vision_alignment_data import STAGE1_V3_MEAN_LOSS_WEIGHTS

    monkeypatch.setattr(
        vision_midtraining,
        "build_stage1_v3_sources",
        lambda phase, sequence_length, artifact_root: {
            name: PixMoCapDatasetConfig(
                dataset_path=f"/data/{name}", max_sequence_length=sequence_length
            )
            for name in STAGE1_V3_MEAN_LOSS_WEIGHTS
        },
    )
    theirs = BeakerLaunchConfig(
        name="ladders-mixed",
        cmd=["train"],
        post_setup="python -m build_ext",
        git=GitRepoState(
            repo="allenai/scaling-ladders",
            repo_url="https://github.com/allenai/scaling-ladders",
            ref="a" * 40,
            branch="main",
        ),
    )
    cli = CliContext(
        script="ladders/olmoe3/workloads/mixed_midtraining.py",
        cmd=SubCmd.dry_run,
        run_name="mixed-test",
        cluster="ai2/holmes",
        overrides=[
            f"--recipe.parent_checkpoint={mixed_recipe.parent}",
            f"--recipe.visual_data={visual_data}",
        ],
    )
    config = build_config(cli, text_config=text_config.config, launch=theirs)
    if visual_data == "stage1_v3":
        assert config.launch.post_setup == f"python -m build_ext && {_STAGE1_V3_POST_SETUP}"
    else:
        assert config.launch.post_setup == "python -m build_ext"
    assert theirs.post_setup == "python -m build_ext"


def test_prefetch_workers_is_a_recipe_knob(mixed_recipe, text_config):
    default = mixed_recipe.build(f"--recipe.text_config={text_config.path}")
    assert default.data_loader.prefetch_workers == text_config.config["data_loader"]["num_workers"]
    more = mixed_recipe.build(
        f"--recipe.text_config={text_config.path}", "--recipe.prefetch_workers=32"
    )
    assert more.data_loader.prefetch_workers == 32
    assert default.data_loader.prefetch_max_in_flight is None
    deeper = mixed_recipe.build(
        f"--recipe.text_config={text_config.path}", "--recipe.prefetch_max_in_flight=128"
    )
    assert deeper.data_loader.prefetch_max_in_flight == 128


def test_text_config_from_the_text_lm_scales_the_fresh_components(
    mixed_recipe, text_config, tmp_path
):
    """From the text LM: the fresh connector at 10x the LM's LR, the pretrained vision encoder at
    a fifth; weight decay as the text config's."""
    # The checkpoint's LM is the text config's (the recipe checks they match).
    lm = OLMoDDPModelConfig.from_dict(text_config.config["model"])
    checkpoint = _write_text_lm_checkpoint(tmp_path / "text-lm" / "step100", lm)
    config = _build_from_text_lm(
        tmp_path,
        f"--recipe.pretraining_checkpoint={checkpoint}",
        f"--recipe.text_config={text_config.path}",
    )
    lr = text_config.config["train_module"]["optim"]["lr"]
    assert config.train_module.optim.lr == lr
    groups = {
        tuple(group.params): group.opts for group in config.train_module.optim.group_overrides
    }
    assert groups[("*connector.*",)]["lr"] == lr * 10
    assert groups[("*vision.*",)]["lr"] == lr / 5
    assert "weight_decay" not in groups[("*connector.*",)]
    assert "weight_decay" not in groups[("*vision.*",)]
    assert (
        config.train_module.optim.weight_decay
        == text_config.config["train_module"]["optim"]["weight_decay"]
    )


def test_text_config_duration_in_steps_sizes_the_budget(mixed_recipe, text_config, tmp_path):
    """A step-based text duration (short runs, as microanneal.py accepts) becomes steps x batch."""
    text = json.loads(text_config.path.read_text())
    text["trainer"]["max_duration"] = {
        "value": 150,
        "unit": "steps",
        "_CLASS_": "olmo_core.train.common.Duration",
    }
    path = tmp_path / "text_steps.json"
    path.write_text(json.dumps(text))
    config = mixed_recipe.build(f"--recipe.text_config={path}")
    batch = text["data_loader"]["global_batch_size"]
    assert config.trainer.max_duration == Duration.tokens(150 * batch)
    assert (
        config.dataset.sources["text_midtraining"].dataset.source_mixture_config.requested_tokens
        == 150 * batch
    )


@pytest.mark.parametrize("document_mode", [None, False])
def test_document_mode_off_uses_the_text_teams_kda_kernels(
    mixed_recipe, text_config, tmp_path, document_mode
):
    from olmo_core.nn.attention.kda import KimiDeltaAttentionConfig

    lm = OLMoDDPModelConfig.from_dict(text_config.config["model"])
    checkpoint = _write_text_lm_checkpoint(tmp_path / "text-lm" / "step100", lm)
    overrides = [
        f"--recipe.pretraining_checkpoint={checkpoint}",
        f"--recipe.text_config={text_config.path}",
    ]
    if document_mode is not None:
        overrides.append(f"--recipe.document_mode={document_mode}")
    config = _build_from_text_lm(tmp_path, *overrides)
    mixers = [
        block.sequence_mixer
        for block in [config.model.lm.block, *(config.model.lm.block_overrides or {}).values()]
        if isinstance(block.sequence_mixer, KimiDeltaAttentionConfig)
    ]
    assert mixers, "the OLMo 3.5 fixture has KDA layers"
    if document_mode is None:
        assert config.model.document_mode is None
        assert all(m.use_experimental_kernels is False for m in mixers)  # FLA, boundaries
    else:
        assert config.model.document_mode is False
        assert all(m.use_experimental_kernels is True for m in mixers)  # kernel_fun, no boundaries


@pytest.mark.parametrize("compile_loss", [None, False])
def test_the_chunked_loss_is_compiled_unless_the_recipe_says_otherwise(
    text_config, tmp_path, compile_loss
):
    lm = OLMoDDPModelConfig.from_dict(text_config.config["model"])
    checkpoint = _write_text_lm_checkpoint(tmp_path / "text-lm" / "step100", lm)
    overrides = [
        f"--recipe.pretraining_checkpoint={checkpoint}",
        f"--recipe.text_config={text_config.path}",
    ]
    if compile_loss is not None:
        overrides.append(f"--recipe.compile_loss={compile_loss}")
    config = _build_from_text_lm(tmp_path, *overrides)
    assert config.model.loss_chunk_size > 0
    assert config.model.compile_loss is (compile_loss is None)
