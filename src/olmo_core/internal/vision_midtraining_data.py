"""Visual replay and text/vision allocation for mixed midtraining."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from olmo_core.config import Config
from olmo_core.data.multimodal.alignment import MultimodalSourceConfig
from olmo_core.data.multimodal.pixmo_cap import PixMoCapDatasetConfig
from olmo_core.data.multimodal.pixmo_points import (
    CoSynPointDatasetConfig,
    PixMoCountDatasetConfig,
    PixMoPointsDatasetConfig,
)

from .vision_alignment_data import DEFAULT_ALIGNMENT_ARTIFACT_ROOT
from .vision_alignment_data import build_visual_sources as build_alignment_sources

DEFAULT_MIDTRAINING_ARTIFACT_ROOT = (
    "/weka/oe-training-default/rustin/experiments/vision-moe/vision-midtraining/artifacts"
)
TEXT_SOURCE_NAME = "text_midtraining"

DEFAULT_VISUAL_EXAMPLE_WEIGHTS = {
    "pixmo_cap": 0.70 * 0.666666666667,
    "pixmo_points_basic": 0.70 * 0.122201237231,
    "pixmo_count": 0.70 * 0.058384427650,
    "pixmo_points_high_frequency": 0.70 * 0.096695558356,
    "cosyn_point": 0.70 * 0.056052110097,
    "pixmo_count_scalar": 0.10,
    "ocr_document": 0.12,
    "audited_alignment": 0.08,
}
"""Conditional visual example shares, including explicit scalar-count and OCR exposure."""

DEFAULT_VISUAL_LOSS_SHARES: dict[str, float] = {}
"""Optional reserved visual loss shares; default sources are all example-weighted."""

DEFAULT_VISUAL_MEAN_LOSS_WEIGHTS = {
    "audited_alignment": 104.0703125,
    "cosyn_point": 100.984375,
    "ocr_document": 4.65625,
    "pixmo_cap": 443.8046875,
    "pixmo_count": 37.4609375,
    "pixmo_count_scalar": 3.0,
    "pixmo_points_basic": 342.359375,
    "pixmo_points_high_frequency": 250.4921875,
}
"""128-example estimates (seed 6198), documented in ``visual_calibration_v1.json``.

Valid for the default selected 8K/eight-crop document sources, explicit grounding prompts,
separate scalar counting and unweighted response loss with Dolma2 revision
``5292e5d6c0f40b67cc765fe41bec991cf4345b5c``. These are bounded estimates, not corpus-exact
means; recalibrate after changing the tokenizer or source serialization.
"""


def build_visual_sources(
    sequence_length: int = 8192,
    max_crops: int = 8,
    alignment_artifact_root: str = DEFAULT_ALIGNMENT_ARTIFACT_ROOT,
    midtraining_artifact_root: str = DEFAULT_MIDTRAINING_ARTIFACT_ROOT,
    *,
    split: str = "train",
) -> dict[str, Config]:
    """Build caption, pointing/counting, OCR and audited alignment source configs.

    This constructs configs only. Caption and Basic-point training retain alignment's prepared
    row exclusions. OCR and audited sources retain its physical splits, quality filters, and
    train/validation exclusions, but use midtraining's unweighted response-token loss.
    Model-specific image token IDs are supplied by the mixture builder.

    :param sequence_length: Maximum serialized example length.
    :param max_crops: Maximum local image crops.
    :param alignment_artifact_root: Prepared alignment datasets and selection manifests.
    :param midtraining_artifact_root: Prepared grounded-count dataset directory.
    :param split: Logical ``train`` or ``validation`` split.
    :returns: Dataset configs keyed by source name.
    """
    if type(sequence_length) is not int or sequence_length < 2:
        raise ValueError("sequence_length must be an integer of at least two")
    if type(max_crops) is not int or max_crops < 1:
        raise ValueError("max_crops must be a positive integer")
    if split not in ("train", "validation"):
        raise ValueError(f"Unknown visual split {split!r}")

    common: dict[str, Any] = {
        "split": split,
        "require_split": True,
        "max_crops": max_crops,
        "max_sequence_length": sequence_length,
        "loss_token_weighting": "none",
        "message_format": "document",
        "seed": 0,
    }
    count_common = {
        **common,
        "dataset_path": str(
            Path(midtraining_artifact_root) / "pixmo-count-grounded-holdout-v1/dataset"
        ),
        "split": "grounded_validation" if split == "validation" else "train",
    }
    sources: dict[str, Config] = {
        "pixmo_cap": PixMoCapDatasetConfig(
            dataset_path=str(
                Path(alignment_artifact_root) / "pixmo-cap-content-disjoint-v1/dataset"
            ),
            mode="transcript_and_caption",
            style_length_conditioning=True,
            **common,
        ),
        "pixmo_points_basic": PixMoPointsDatasetConfig(
            kind="basic",
            counting="both",
            both_mode="duplicate",
            explicit_grounding_prompts=True,
            **common,
        ),
        "pixmo_count": PixMoCountDatasetConfig(
            mode="grounded",
            counting="both",
            explicit_grounding_prompts=True,
            scalar_count_replay=False,
            **count_common,
        ),
        "pixmo_count_scalar": PixMoCountDatasetConfig(
            mode="scalar_count",
            **count_common,
        ),
        "pixmo_points_high_frequency": PixMoPointsDatasetConfig(
            kind="high_frequency",
            counting="both",
            both_mode="duplicate",
            explicit_grounding_prompts=True,
            **common,
        ),
        "cosyn_point": CoSynPointDatasetConfig(explicit_grounding_prompts=True, **common),
    }
    import olmo_core.internal.vision_alignment_phases  # noqa: F401  (registers the joint builders)

    alignment = build_alignment_sources(
        "joint", sequence_length, alignment_artifact_root, split=split
    )
    if split == "train":
        for name, alignment_name, repeat in (
            ("pixmo_cap", "pixmo_caption", 1),
            ("pixmo_points_basic", "pixmo_points_basic", 2),
        ):
            selected = alignment[alignment_name]
            assert isinstance(selected, MultimodalSourceConfig)
            selected_dataset = sources[name].replace(
                split=selected.dataset.split  # type: ignore[attr-defined]
            )
            sources[name] = selected.replace(
                dataset=selected_dataset,
                selection_repeat=repeat,
            )
    for name in ("ocr_document", "audited_alignment"):
        selected = alignment[name]
        assert isinstance(selected, MultimodalSourceConfig)
        sources[name] = selected.replace(
            dataset=selected.dataset.replace(max_crops=max_crops, loss_token_weighting="none")
        )
    return dict(sorted(sources.items()))


def _validate_text_share(value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Text loss mass must be finite and between zero and one")


def _validate_positive_mapping(values: Mapping[str, float], name: str) -> None:
    for source, value in values.items():
        if not isinstance(source, str) or not source:
            raise ValueError(f"{name} must use nonempty source names")
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} for {source!r} must be finite and positive")


def loss_mass_targets(
    mean_loss_weight: Mapping[str, float] | None = None,
    *,
    target_text_loss_mass: float = 0.9,
    visual_example_weights: Mapping[str, float] | None = None,
    visual_loss_shares: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Allocate text/vision loss while retaining configurable visual example ratios.

    ``visual_loss_shares`` reserves conditional visual loss for named sources. The remainder
    goes to ``visual_example_weights`` sources, retaining their relative example frequencies.
    These disjoint groups can contain any source names. Returned targets work with the shared
    mixture's ``target / mean`` sampler; zero-mass modalities are omitted.

    :param mean_loss_weight: Mean summed loss weights for the active sources.
    :param target_text_loss_mass: Text's share of expected loss mass, including endpoints.
    :param visual_example_weights: Relative example weights within the remaining visual mass.
    :param visual_loss_shares: Named sources' shares of aggregate visual loss mass.
    :returns: Expected loss-mass targets keyed by active source name.
    """
    _validate_text_share(target_text_loss_mass)
    if target_text_loss_mass == 1:
        return {TEXT_SOURCE_NAME: 1.0}

    examples = dict(
        DEFAULT_VISUAL_EXAMPLE_WEIGHTS if visual_example_weights is None else visual_example_weights
    )
    shares = dict(DEFAULT_VISUAL_LOSS_SHARES if visual_loss_shares is None else visual_loss_shares)
    _validate_positive_mapping(examples, "Visual example weight")
    _validate_positive_mapping(shares, "Visual loss share")
    if set(examples) & set(shares) or TEXT_SOURCE_NAME in examples or TEXT_SOURCE_NAME in shares:
        raise ValueError(
            "Visual weight groups must be disjoint and cannot contain text_midtraining"
        )
    reserved = sum(shares.values())
    if examples:
        if reserved >= 1:
            raise ValueError(
                "Visual loss shares must leave positive mass for example-weighted sources"
            )
    elif not math.isclose(reserved, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Without example-weighted sources, visual loss shares must sum to one")

    means = {} if mean_loss_weight is None else dict(mean_loss_weight)
    expected = set(examples) | set(shares)
    if target_text_loss_mass > 0:
        expected.add(TEXT_SOURCE_NAME)
    if set(means) != expected:
        raise ValueError("Mean loss weights must cover exactly the active weighted sources")
    _validate_positive_mapping(means, "Mean loss weight")

    visual_mass = 1 - target_text_loss_mass
    example_mass = visual_mass * (1 - reserved)
    weighted_means = {name: weight * means[name] for name, weight in examples.items()}
    total = sum(weighted_means.values())
    if examples and (not math.isfinite(total) or total <= 0):
        raise ValueError("Example-weighted visual loss must have a finite positive total")
    targets = {name: example_mass * value / total for name, value in weighted_means.items()}
    targets.update({name: visual_mass * share for name, share in shares.items()})
    if target_text_loss_mass > 0:
        targets[TEXT_SOURCE_NAME] = target_text_loss_mass
    return dict(sorted(targets.items()))
