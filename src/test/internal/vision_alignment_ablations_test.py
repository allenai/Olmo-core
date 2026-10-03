"""Stage ablations of the alignment recipe: ``recipe.steps`` and phases started from the text LM."""

import json

import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal import vision_alignment
from olmo_core.internal.vision_alignment import VisionAlignmentExperimentConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.transformer import OLMoDDPModelConfig
from olmo_core.train import LoadStrategy
from olmo_core.train.callbacks.multimodal import InitializeMultimodalModelCallback


def _chain(alignment_recipe, phase, overrides=()):
    """Build ``phase`` through the default bridge -> perception -> joint handoffs (parents are
    saved once per test)."""
    parents = alignment_recipe.__dict__.setdefault("chain_parents", {None: None})
    previous = None
    for stage in vision_alignment.AlignmentPhase:
        if stage == phase:
            return alignment_recipe.build(stage, parents[previous], overrides=overrides)
        if stage not in parents:
            parents[stage] = alignment_recipe.save(alignment_recipe.build(stage, parents[previous]))
        previous = stage
    raise AssertionError(phase)


def _horizons(config) -> dict[str, int]:
    scheduler = config.train_module.scheduler
    return {
        "connector": scheduler.schedulers["connector"].t_max,
        "vision": scheduler.schedulers["vision"].t_max,
        "lm": scheduler.default.t_max,
    }


def _warmups(config) -> dict[str, int]:
    scheduler = config.train_module.scheduler
    return {
        "connector": scheduler.schedulers["connector"].warmup,
        "vision": scheduler.schedulers["vision"].warmup,
        "lm": scheduler.default.warmup,
    }


@pytest.mark.parametrize(
    "phase,steps,horizons",
    [
        ("bridge", 1000, {"connector": 500, "vision": 1000, "lm": 1000}),
        ("perception", 2500, {"connector": 2500, "vision": 2500, "lm": 2500}),
        ("perception", 3500, {"connector": 3500, "vision": 3500, "lm": 3500}),
        ("joint", 3500, {"connector": 3500, "vision": 3500, "lm": 3500}),
    ],
)
def test_steps_sets_duration_and_every_schedule_horizon(alignment_recipe, phase, steps, horizons):
    default = _chain(alignment_recipe, phase)
    config = _chain(alignment_recipe, phase, overrides=[f"--recipe.steps={steps}"])
    assert config.trainer.max_duration.value == steps
    assert _horizons(config) == horizons
    assert _warmups(config) == _warmups(default)
    for name in ("connector", "vision"):
        assert (
            config.train_module.scheduler.schedulers[name].alpha_f
            == default.train_module.scheduler.schedulers[name].alpha_f
        )
    assert (
        config.train_module.scheduler.default.alpha_f
        == default.train_module.scheduler.default.alpha_f
    )


@pytest.mark.parametrize("phase", ["bridge", "perception", "joint"])
def test_default_steps_leave_the_phase_unchanged(alignment_recipe, phase):
    default = _chain(alignment_recipe, phase)
    assert default.recipe.steps is None
    assert default.trainer.max_duration.value == vision_alignment._PHASES[phase].steps
    explicit = _chain(
        alignment_recipe,
        phase,
        overrides=[f"--recipe.steps={vision_alignment._PHASES[phase].steps}"],
    )
    explicit.recipe.steps = None
    assert explicit == default


@pytest.mark.parametrize("steps", [0, -5])
def test_steps_must_be_positive(alignment_recipe, steps):
    with pytest.raises(OLMoConfigurationError, match="recipe.steps"):
        alignment_recipe.build(overrides=[f"--recipe.steps={steps}"])


@pytest.mark.parametrize("phase", ["perception", "joint"])
def test_phase_starts_from_the_pretraining_checkpoint_like_bridge(alignment_recipe, phase):
    bridge = alignment_recipe.build()
    chained = _chain(alignment_recipe, phase)
    fresh = alignment_recipe.build(phase)

    # Initialized exactly like bridge: text LM weights, SigLIP, fresh connector and image rows.
    assert fresh.recipe.parent_checkpoint is None
    assert fresh.pretraining_checkpoint == str(alignment_recipe.base)
    assert fresh.trainer.load_path is None
    assert fresh.trainer.load_strategy == LoadStrategy.if_available
    assert fresh.trainer.load_optim_state is fresh.trainer.load_trainer_state is None
    initializer = fresh.trainer.callbacks["initialize_multimodal"]
    assert isinstance(initializer, InitializeMultimodalModelCallback)
    assert initializer == bridge.trainer.callbacks["initialize_multimodal"]
    assert initializer.language_checkpoint == str(alignment_recipe.base)
    assert fresh.model == chained.model
    # The phase's own policy: what trains, the rates, schedules and image-token rows.
    assert fresh.train_module == chained.train_module
    assert fresh.train_module.freeze_params == list(vision_alignment._PHASES[phase].freeze_params)
    assert fresh.train_module.train_embedding_rows == vision_alignment._image_token_rows(
        alignment_recipe.token_ids
    )
    assert fresh.dataset == chained.dataset
    assert fresh.data_loader == chained.data_loader
    restored = VisionAlignmentExperimentConfig.from_dict(
        json.loads(json.dumps(fresh.as_config_dict()))
    )
    assert restored == fresh


def test_perception_from_the_pretraining_checkpoint_hands_off_to_joint(alignment_recipe):
    perception = alignment_recipe.build("perception", overrides=["--recipe.steps=2500"])
    joint = alignment_recipe.build("joint", alignment_recipe.save(perception))
    assert joint.pretraining_checkpoint == str(alignment_recipe.base)
    assert "initialize_multimodal" not in joint.trainer.callbacks
    assert joint.trainer.load_strategy == LoadStrategy.always


@pytest.mark.parametrize("phase", ["perception", "joint"])
def test_bridge_parent_with_more_steps(alignment_recipe, phase):
    bridge = alignment_recipe.save(alignment_recipe.build())
    config = alignment_recipe.build(phase, bridge, overrides=["--recipe.steps=3500"])
    assert config.trainer.load_path == str(bridge)
    assert config.trainer.max_duration.value == 3500
    assert "initialize_multimodal" not in config.trainer.callbacks
    assert config.train_module.freeze_params == list(vision_alignment._PHASES[phase].freeze_params)


def test_phase_without_any_checkpoint_is_rejected(alignment_recipe):
    with pytest.raises(OLMoConfigurationError, match="recipe.parent_checkpoint"):
        alignment_recipe.build("perception", overrides=["--recipe.pretraining_checkpoint=null"])


def test_router_load_balancing_from_the_pretraining_checkpoint(alignment_recipe):
    config_path = alignment_recipe.base / "config.json"
    base = json.loads(config_path.read_text())
    lm = OLMoDDPModelConfig.from_dict(base["model"])
    lm.block.routed_experts_router = MoERouterConfigV2(
        d_model=64, num_experts=8, top_k=2, lb_loss_weight=0.02
    )
    base["model"] = lm.as_config_dict()
    config_path.write_text(json.dumps(base))
    # Frozen LM: off, as perception inherits it from bridge; joint restores pretraining.
    assert (
        alignment_recipe.build("perception").model.lm.block.routed_experts_router.lb_loss_weight
        == 0.0
    )
    joint = alignment_recipe.build("joint")
    assert joint.recipe.restore_pretraining_router_lb
    assert joint.model.lm.block.routed_experts_router.lb_loss_weight == 0.02
