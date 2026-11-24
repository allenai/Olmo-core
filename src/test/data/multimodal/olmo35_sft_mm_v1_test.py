"""Stage-2 image data for Olmo 3.5 multimodal SFT (version 1)."""

from __future__ import annotations

import math
import os

import pytest

from olmo_core.data.multimodal.mixtures import olmo35_sft_mm_v1 as v1
from olmo_core.data.multimodal.mixtures.image_only_v9 import IMAGE_ONLY_V9_SUBMIXTURES
from olmo_core.data.multimodal.olmo35_layout import OLMO35_TOKEN_IDS

STAGE1_SAME_TASK_CAP = 0.25
"""At most this share of the image-data tokens may come from images stage 1 trained on with the
same task (the anchors and the pointing and counting sources)."""


def test_membership():
    v9 = [s.name for g in IMAGE_ONLY_V9_SUBMIXTURES if g.name != "nlp" for s in g.datasets]
    assert len(v9) == 42 and len(v1.KEPT) == 32
    assert set(v1.KEPT) == set(v9) - set(v1.DROPPED)
    assert "tulu4" not in v1.KEPT
    assert set(v1.ANCHOR_TOKENS) <= set(v1.KEPT)
    for table in (v1.V9_EXAMPLE_RATES, v1.MEAN_TOKENS, v1.MEAN_LOSS_WEIGHTS):
        assert list(table) == list(v1.KEPT)
        assert all(value > 0 and math.isfinite(value) for value in table.values())


def test_dose_spends_the_tokens_at_molmo2_rates():
    examples = v1.expected_examples()
    tokens = {name: n * v1.MEAN_TOKENS[name] for name, n in examples.items()}
    assert sum(tokens.values()) == pytest.approx(v1.DEFAULT_DOSE_TOKENS, rel=1e-9)
    for name, dose in v1.ANCHOR_TOKENS.items():
        assert tokens[name] == pytest.approx(dose, rel=1e-9)
    rest = [name for name in v1.KEPT if name not in v1.ANCHOR_TOKENS]
    ratio = examples[rest[0]] / v1.V9_EXAMPLE_RATES[rest[0]]
    for name in rest:
        assert examples[name] / v1.V9_EXAMPLE_RATES[name] == pytest.approx(ratio, rel=1e-9)
    # Doubling the dose doubles every non-anchor source.
    doubled = v1.expected_examples(2 * v1.DEFAULT_DOSE_TOKENS)
    total = 2 * v1.DEFAULT_DOSE_TOKENS - sum(v1.ANCHOR_TOKENS.values())
    once = v1.DEFAULT_DOSE_TOKENS - sum(v1.ANCHOR_TOKENS.values())
    assert doubled[rest[0]] / examples[rest[0]] == pytest.approx(total / once, rel=1e-9)


def test_stage1_same_task_share_is_capped():
    examples = v1.expected_examples()
    same = sum(
        n * v1.MEAN_TOKENS[name]
        for name, n in examples.items()
        if v1.SOURCE_CLASS.get(name) in ("anchor", "stage-1 images, same task")
    )
    assert same / v1.DEFAULT_DOSE_TOKENS <= STAGE1_SAME_TASK_CAP
    assert not any(v1.SOURCE_CLASS.get(name) == "dropped" for name in v1.KEPT)


def test_sources_render_olmo35_at_the_sft_length():
    sources = v1.build_sources()
    assert list(sources) == list(v1.KEPT)
    for name, config in sources.items():
        assert config.message_format == "olmo35", name  # type: ignore[attr-defined]
        assert config.token_ids == OLMO35_TOKEN_IDS, name  # type: ignore[attr-defined]
        # The mixture refuses a source whose max_sequence_length differs from the recipe's.
        assert getattr(config, "max_sequence_length", v1.SEQUENCE_LENGTH) == v1.SEQUENCE_LENGTH
    for name in v1.DROPPED:
        with pytest.raises(KeyError):
            v1.build_source_config(name)


def test_loss_mass_targets_accept_the_v1_tables():
    from olmo_core.internal.vision_midtraining_data import (
        TEXT_SOURCE_NAME,
        loss_mass_targets,
    )

    examples = v1.expected_examples()
    visual = v1.visual_loss_mass()
    assert visual == pytest.approx(
        sum(n * v1.MEAN_LOSS_WEIGHTS[name] for name, n in examples.items())
    )
    text_mass = 15.5e9  # the order of a half pass of the text SFT data
    share = text_mass / (text_mass + visual)
    targets = loss_mass_targets(
        {**v1.MEAN_LOSS_WEIGHTS, TEXT_SOURCE_NAME: 58_000.0},
        target_text_loss_mass=share,
        visual_example_weights=examples,
    )
    assert targets[TEXT_SOURCE_NAME] == pytest.approx(share)
    assert sum(targets.values()) == pytest.approx(1.0)
    # The sampler draws target / mean examples: the v1 example ratios come back.
    draws = {name: targets[name] / v1.MEAN_LOSS_WEIGHTS[name] for name in v1.KEPT}
    for name in v1.KEPT:
        assert draws[name] / draws["text_vqa"] == pytest.approx(
            examples[name] / examples["text_vqa"], rel=1e-9
        )


def test_one_source_builds_an_olmo35_example():
    from olmo_core.data.multimodal.paths import ACADEMIC_DATASETS

    if not os.path.isdir(ACADEMIC_DATASETS):
        pytest.skip("stage-2 data not mounted")
    pytest.importorskip("datasets")
    from olmo_core.data.multimodal.olmo35_layout import load_olmo35_sft_tokenizer

    try:
        tokenizer, _ = load_olmo35_sft_tokenizer()
    except OSError as exc:
        pytest.skip(f"tokenizer unavailable: {exc}")
    example = v1.build_source_config("text_vqa").build(tokenizer)[0]
    assert example["token_type_ids"].any()
    supervised = example["labels"][example["loss_masks"] > 0]
    assert supervised[0] == 100266 and supervised[-1] == tokenizer.eos_token_id
