"""The multimodal SFT recipe's derived schedule: text exposure fixed, image dose as the knob."""

import math

import pytest

from olmo_core.internal.olmo35_mm_sft import (
    C12_HALF_PASS,
    C12_INSTANCES,
    REPLAY_INSTANCES,
    TEXT_SOURCE,
    Olmo35MultimodalSFTRecipeConfig,
    derive_schedule,
    text_totals,
)


def test_text_exposure_is_half_pass_plus_replay():
    total, drawn, mean = text_totals(Olmo35MultimodalSFTRecipeConfig())
    assert total == C12_INSTANCES + REPLAY_INSTANCES
    assert drawn == round(C12_HALF_PASS * total / C12_INSTANCES)
    assert 50_000 < mean < 65_536


def test_text_only_schedule():
    s = derive_schedule(
        text_instances=294_560,
        text_mean_loss_weight=58_000.0,
        image_dose_tokens=0,
        image_stats={},
        sequence_length=65_536,
        instances_per_step=16,
    )
    assert s["text_loss_share"] == 1.0 and s["target_loss_mass"] == {TEXT_SOURCE: 1.0}
    assert s["steps"] == 18_410


def test_image_dose_sets_share_and_steps():
    stats = {"a": (1.0, 100.0, 1000.0), "b": (3.0, 300.0, 2000.0)}
    s = derive_schedule(
        text_instances=294_560,
        text_mean_loss_weight=58_000.0,
        image_dose_tokens=3.5e9,
        image_stats=stats,
        sequence_length=65_536,
        instances_per_step=16,
        image_pack_fill=1.0,
    )
    # Equal per-example probability (share / mean is 0.01 for both), so 1,500 tokens and
    # 200 loss per image example on average.
    assert s["image_tokens_per_example"] == pytest.approx(1500.0)
    assert s["image_loss_per_example"] == pytest.approx(200.0)
    image_mass = 3.5e9 / 1500.0 * 200.0
    text_mass = 294_560 * 58_000.0
    assert s["text_loss_share"] == pytest.approx(text_mass / (text_mass + image_mass))
    t = s["target_loss_mass"]
    assert sum(t.values()) == pytest.approx(1.0)
    assert t["b"] == pytest.approx(3 * t["a"])
    assert s["steps"] == math.ceil((294_560 + 3.5e9 / 65_536) / 16)
