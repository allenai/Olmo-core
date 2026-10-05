import json

import pytest

from olmo_core.data.multimodal.pixmo_cap import PixMoCapDatasetConfig
from olmo_core.data.multimodal.pretraining_replay import PretrainingReplayConfig
from olmo_core.internal.vision_alignment import VisionAlignmentExperimentConfig
from olmo_core.train import Duration


def _perception_checkpoint(recipe):
    bridge = recipe.save(recipe.build())
    return recipe.save(recipe.build("perception", bridge))


def test_joint_group_loss_is_separate_from_sequence_quotas(alignment_recipe):
    config = alignment_recipe.build("joint", _perception_checkpoint(alignment_recipe))
    loader = config.data_loader
    assert loader.source_groups == {
        name: "text" if isinstance(source, PretrainingReplayConfig) else "vision"
        for name, source in config.dataset.sources.items()
    }
    assert loader.group_sequence_quotas == {"text": 16, "vision": 112}
    assert sum(loader.group_sequence_quotas.values()) * loader.sequence_length == (
        loader.global_batch_size
    )
    assert config.train_module.loss_group_weights == {"text": 0.35, "vision": 0.65}
    assert config.train_module.source_loss_mass_targets == pytest.approx(
        config.dataset.target_loss_mass
    )
    assert config.dataset.target_loss_mass["native_text_replay"] == 0.35
    assert config.dataset.mean_loss_weight["native_text_replay"] == 8191


def test_joint_groups_include_final_custom_source(alignment_recipe):
    parent = _perception_checkpoint(alignment_recipe)
    custom = PixMoCapDatasetConfig(
        dataset_path="/custom/captions", max_sequence_length=8192
    ).as_config_dict()
    config = alignment_recipe.build(
        "joint",
        parent,
        overrides=[
            f"--dataset.sources.custom_caption={json.dumps(custom)}",
            "--dataset.target_loss_mass.custom_caption=0.01",
            "--dataset.target_loss_mass.pixmo_caption=0.27",
            "--dataset.mean_loss_weight.custom_caption=25",
        ],
    )
    assert set(config.data_loader.source_groups) == set(config.dataset.sources)
    assert config.data_loader.source_groups["custom_caption"] == "vision"
    assert config.data_loader.source_groups["native_text_replay"] == "text"
    assert config.data_loader.group_sequence_quotas == {"text": 16, "vision": 112}
    assert config.train_module.loss_group_weights == {"text": 0.35, "vision": 0.65}
    assert config.dataset.sampling_weights()["custom_caption"] > 0


def test_joint_optimizer_and_complete_cosine_schedule(alignment_recipe):
    config = alignment_recipe.build("joint", _perception_checkpoint(alignment_recipe))
    module = config.train_module
    assert config.trainer.max_duration == Duration.steps(1500)
    assert config.trainer.hard_stop is None
    assert module.freeze_params == ["lm.lm_head.w_out.weight"]
    assert module.vision_activation_checkpointing and not module.connector_activation_checkpointing
    assert module.response_logits_only and module.compile_model
    assert module.z_loss_multiplier == 1e-4
    assert module.optim.lr == 1e-6
    assert [(group.params, group.opts) for group in module.optim.group_overrides] == [
        (
            ["*lm.embeddings.weight"],
            {"lr": 2e-5, "weight_decay": 0.0, "scheduler_name": "connector"},
        ),
        (["*connector.*"], {"lr": 2e-5, "weight_decay": 0.0, "scheduler_name": "connector"}),
        (["*vision.*"], {"lr": 2e-6, "weight_decay": 0.0, "scheduler_name": "vision"}),
    ]
    for schedule, peak, warmup in [
        (module.scheduler.schedulers["connector"], 2e-5, 50),
        (module.scheduler.schedulers["vision"], 2e-6, 100),
        (module.scheduler.default, 1e-6, 100),
    ]:
        assert schedule.warmup == warmup and schedule.t_max == 1500
        assert schedule.alpha_f == 0.1
        assert schedule.get_lr(peak, warmup, 1500) == pytest.approx(peak)
        assert schedule.get_lr(peak, 1500, 1500) == pytest.approx(peak * 0.1)


@pytest.mark.parametrize("default_weight", [None, 0.0, 0.015, 0.025])
def test_joint_restores_original_per_layer_lb_without_architecture_repair(
    alignment_recipe, default_weight
):
    original = alignment_recipe.set_router_coefficients(default_weight)
    parent_path = _perception_checkpoint(alignment_recipe)
    parent = VisionAlignmentExperimentConfig.from_dict(
        json.loads((parent_path / "config.json").read_text())
    )
    config = alignment_recipe.build("joint", parent_path)
    expected = parent.model.copy()
    for block, pretrained in zip(
        expected.lm.resolved_block_configs, original.resolved_block_configs
    ):
        block.routed_experts_router.lb_loss_weight = pretrained.routed_experts_router.lb_loss_weight
    assert config.model == expected
    for block, pretrained in zip(
        config.model.lm.resolved_block_configs, original.resolved_block_configs
    ):
        assert (
            block.routed_experts_router.lb_loss_weight
            == pretrained.routed_experts_router.lb_loss_weight
        )
        assert block.routed_experts_router.z_loss_weight == 0.003
        assert block.ep == pretrained.ep
    assert config.recipe.restore_pretraining_router_lb


@pytest.mark.parametrize(
    "override",
    ["--recipe.restore_pretraining_router_lb=false", "--recipe.router_lb_loss_weight=0"],
)
def test_joint_permits_explicit_lb_off(alignment_recipe, override):
    alignment_recipe.set_router_coefficients(0.015)
    config = alignment_recipe.build(
        "joint", _perception_checkpoint(alignment_recipe), overrides=[override]
    )
    assert all(
        block.routed_experts_router.lb_loss_weight == 0
        for block in config.model.lm.resolved_block_configs
    )


def test_joint_null_lb_restoration_uses_phase_default(alignment_recipe):
    original = alignment_recipe.set_router_coefficients(0.015)
    config = alignment_recipe.build(
        "joint",
        _perception_checkpoint(alignment_recipe),
        overrides=["--recipe.restore_pretraining_router_lb=null"],
    )
    assert config.recipe.restore_pretraining_router_lb
    assert all(
        block.routed_experts_router.lb_loss_weight
        == pretrained.routed_experts_router.lb_loss_weight
        for block, pretrained in zip(
            config.model.lm.resolved_block_configs, original.resolved_block_configs
        )
    )
