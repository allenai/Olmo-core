import json

import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.vision_alignment_data import (
    ALIGNMENT_MEAN_LOSS_WEIGHTS,
    DEFAULT_ALIGNMENT_ARTIFACT_ROOT,
    build_visual_sources,
)
from olmo_core.train import Duration

BRIDGE_MEANS = {
    "pixmo_caption": 28.829104933305643,
    "pixmo_transcript": 29.77422616124386,
}


def test_bridge_default_loss_calibration(alignment_recipe):
    config = alignment_recipe.build(
        include_means=False,
        overrides=[f"--recipe.artifact_root={DEFAULT_ALIGNMENT_ARTIFACT_ROOT}"],
    )
    assert list(config.dataset.sources) == ["pixmo_caption", "pixmo_transcript"]
    assert ALIGNMENT_MEAN_LOSS_WEIGHTS["bridge"] == config.dataset.mean_loss_weight == BRIDGE_MEANS
    assert config.dataset.target_loss_mass == {"pixmo_caption": 0.7, "pixmo_transcript": 0.3}
    assert config.train_module.source_loss_mass_targets == config.dataset.target_loss_mass
    unnormalized = {
        name: target / BRIDGE_MEANS[name]
        for name, target in config.dataset.target_loss_mass.items()
    }
    expected = {name: value / sum(unnormalized.values()) for name, value in unnormalized.items()}
    assert config.dataset.sampling_weights() == pytest.approx(expected)


def test_bridge_nondefault_context_requires_calibration(alignment_recipe):
    overrides = [
        f"--recipe.artifact_root={DEFAULT_ALIGNMENT_ARTIFACT_ROOT}",
        "--recipe.sequence_length=2560",
    ]
    with pytest.raises(OLMoConfigurationError, match="calibrated dataset.mean_loss_weight"):
        alignment_recipe.build(include_means=False, overrides=overrides)
    short_means = {"pixmo_caption": 29.144221610185923, "pixmo_transcript": 29.720018727071874}
    config = alignment_recipe.build(
        include_means=False,
        overrides=[*overrides, f"--dataset.mean_loss_weight={json.dumps(short_means)}"],
    )
    assert config.dataset.mean_loss_weight == short_means
    assert config.data_loader.sequence_length == 2560
    assert ALIGNMENT_MEAN_LOSS_WEIGHTS["bridge"] == BRIDGE_MEANS


@pytest.mark.parametrize("split", ["train", "validation"])
def test_bridge_visual_source_definitions(split):
    sources = build_visual_sources("bridge", 8192, split=split)
    assert list(sources) == ["pixmo_caption", "pixmo_transcript"]
    for name, source in sources.items():
        assert (
            source.dataset_path
            == f"{DEFAULT_ALIGNMENT_ARTIFACT_ROOT}/pixmo-cap-content-disjoint-v1/dataset"
        )
        assert source.split == split and source.require_split
        assert source.max_sequence_length == 8192 and source.max_crops == 8
        assert source.loss_token_weighting == "root_subsegments_root_tokens"
        assert source.message_format == "document"
        assert not source.style_length_conditioning
        assert source.seed == 0
        assert source.mode == ("caption" if name == "pixmo_caption" else "transcript")
    assert sources["pixmo_caption"].fixed_prompt == "Description:"
    assert sources["pixmo_transcript"].fixed_prompt == "Transcript:"
    assert sources["pixmo_transcript"].require_transcript


def test_bridge_model_preserves_parent_architecture(alignment_recipe):
    original = alignment_recipe.set_router_coefficients(0.02)
    config = alignment_recipe.build()
    # Execution settings (attention backend, expert dispatch, recomputation) are the
    # checkpoint's own; only the router LB policy changes and two-batch overlap is off.
    expected = original.copy()
    expected.two_batch_overlap = False
    for block in [expected.block, *expected.block_overrides.values()]:
        block.routed_experts_router.lb_loss_weight = 0.0
    assert config.model.lm == expected
    assert config.model.image_patch_token_id == alignment_recipe.token_ids.im_patch_id
    saved = json.loads((alignment_recipe.base / "config.json").read_text())
    assert saved["model"] == json.loads(json.dumps(original.as_config_dict()))


def test_bridge_optimizer_and_freezing(alignment_recipe):
    config = alignment_recipe.build()
    module = config.train_module
    assert module.freeze_params == [
        "vision.*",
        "lm.embedding_norm.*",
        "lm.blocks.*",
        "lm.lm_head.*",
    ]
    assert not module.vision_activation_checkpointing
    assert module.connector_activation_checkpointing and module.response_logits_only
    assert module.compile_model and module.ep_config.degree == 8
    assert module.z_loss_multiplier == 1e-4
    optim = module.optim
    assert optim.lr == 2e-4
    assert optim.betas == (0.9, 0.95)
    assert optim.eps == 1e-6 and optim.weight_decay == 0.0
    assert optim.max_grad_norm == module.max_grad_norm == 1.0
    assert optim.clip_grad_norm_by_scheduler_group and optim.check_nan_inf_grad
    assert optim.sigma_factor == 12 and optim.foreach_chunk_size == 50_000_000
    assert not optim.compile
    assert [(group.params, group.opts) for group in optim.group_overrides] == [
        (
            ["*lm.embeddings.weight"],
            {"lr": 2e-4, "weight_decay": 0.0, "scheduler_name": "connector"},
        ),
        (["*connector.*"], {"lr": 2e-4, "weight_decay": 0.0, "scheduler_name": "connector"}),
        (["*vision.*"], {"lr": 0.0, "weight_decay": 0.0, "scheduler_name": "vision"}),
    ]


def test_bridge_schedule_decays_before_training_stop(alignment_recipe):
    config = alignment_recipe.build()
    assert config.trainer.max_duration == Duration.steps(500)
    scheduler = config.train_module.scheduler
    connector = scheduler.schedulers["connector"]
    assert connector.warmup == 100 and connector.t_max == 250
    assert connector.alpha_f == 0.1
    assert connector.get_lr(2e-4, 50, 500) == pytest.approx(1e-4)
    assert connector.get_lr(2e-4, 100, 500) == pytest.approx(2e-4)
    assert connector.get_lr(2e-4, 250, 500) == pytest.approx(2e-5)
    assert connector.get_lr(2e-4, 500, 500) == pytest.approx(2e-5)
    assert scheduler.default.warmup == scheduler.schedulers["vision"].warmup == 100
    assert scheduler.default.t_max == scheduler.schedulers["vision"].t_max == 500
    shorter = alignment_recipe.build(overrides=["--trainer.max_duration.value=250"])
    assert shorter.train_module.scheduler == scheduler
