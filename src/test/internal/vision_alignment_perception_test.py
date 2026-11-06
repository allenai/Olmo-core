import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.vision_alignment_data import DEFAULT_ALIGNMENT_ARTIFACT_ROOT
from olmo_core.train import Duration

PERCEPTION_MEANS = {
    "audited_alignment": 18.696413921716157,
    "cosyn_point": 19.90814993873937,
    "ocr_document": 4.103901697322726,
    "pixmo_caption": 28.26835112122353,
    "pixmo_points_basic": 33.4985158557829,
    "pixmo_points_high_frequency": 27.439346029539593,
    "pixmo_transcript": 29.593976828677114,
    "scalar_count": 3.464101552963257,
}


def test_perception_default_loss_calibration(alignment_recipe):
    bridge = alignment_recipe.save(alignment_recipe.build())
    config = alignment_recipe.build(
        "perception",
        bridge,
        include_means=False,
        overrides=[f"--recipe.artifact_root={DEFAULT_ALIGNMENT_ARTIFACT_ROOT}"],
    )
    assert config.dataset.mean_loss_weight == PERCEPTION_MEANS
    assert len(config.dataset.sources) == 8
    weights = {
        name: target / PERCEPTION_MEANS[name]
        for name, target in config.dataset.target_loss_mass.items()
    }
    assert config.dataset.sampling_weights() == pytest.approx(
        {name: value / sum(weights.values()) for name, value in weights.items()}
    )


def test_perception_schedule_matches_training_duration(alignment_recipe):
    bridge = alignment_recipe.save(alignment_recipe.build())
    config = alignment_recipe.build("perception", bridge)
    assert config.trainer.max_duration == Duration.steps(2000)
    scheduler = config.train_module.scheduler
    for schedule, warmup, peak in [
        (scheduler.schedulers["connector"], 100, 5e-5),
        (scheduler.schedulers["vision"], 250, 3e-6),
        (scheduler.default, 250, 5e-5),
    ]:
        assert schedule.t_max == 2000 and schedule.warmup == warmup
        assert schedule.alpha_f == 0.1
        assert schedule.get_lr(peak, warmup, 2000) == pytest.approx(peak)
        assert schedule.get_lr(peak, 2000, 2000) == pytest.approx(peak * 0.1)
    shorter = alignment_recipe.build(
        "perception", bridge, overrides=["--trainer.max_duration.value=1000"]
    )
    assert shorter.train_module.scheduler == scheduler


def test_perception_nondefault_context_requires_calibration(alignment_recipe):
    bridge = alignment_recipe.save(alignment_recipe.build())
    with pytest.raises(OLMoConfigurationError, match="calibrated dataset.mean_loss_weight"):
        alignment_recipe.build(
            "perception",
            bridge,
            include_means=False,
            overrides=[
                f"--recipe.artifact_root={DEFAULT_ALIGNMENT_ARTIFACT_ROOT}",
                "--recipe.sequence_length=2560",
            ],
        )
