"""Default visual sources for bridge, perception, and joint alignment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from olmo_core.config import Config
from olmo_core.data.multimodal.pixmo_cap import PixMoCapDatasetConfig

DEFAULT_ALIGNMENT_ARTIFACT_ROOT = (
    "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/artifacts"
)

ALIGNMENT_LOSS_TARGETS: dict[str, dict[str, float]] = {
    "bridge": {"pixmo_caption": 0.70, "pixmo_transcript": 0.30},
    "perception": {
        "pixmo_caption": 0.45,
        "pixmo_transcript": 0.20,
        "pixmo_points_basic": 0.10,
        "pixmo_points_high_frequency": 0.02,
        "cosyn_point": 0.03,
        "ocr_document": 0.10,
        "scalar_count": 0.05,
        "audited_alignment": 0.05,
    },
    "joint": {
        "native_text_replay": 0.35,
        "pixmo_caption": 0.28,
        "pixmo_transcript": 0.12,
        "pixmo_points_basic": 0.05,
        "pixmo_points_high_frequency": 0.01,
        "cosyn_point": 0.02,
        "ocr_document": 0.08,
        "count_numeric": 0.04,
        "audited_alignment": 0.05,
    },
}

ALIGNMENT_MEAN_LOSS_WEIGHTS: dict[str, dict[str, float]] = {
    "bridge": {
        "pixmo_caption": 28.829104933305643,
        "pixmo_transcript": 29.77422616124386,
    },
    "perception": {
        "audited_alignment": 18.860581716464367,
        "cosyn_point": 19.9274699697271,
        "ocr_document": 4.121176112443209,
        "pixmo_caption": 28.593906218127813,
        "pixmo_points_basic": 33.50284379025106,
        "pixmo_points_high_frequency": 27.439346029539593,
        "pixmo_transcript": 29.704384807148017,
        "scalar_count": 3.464101552963257,
    },
    "joint": {
        "audited_alignment": 18.266647445358103,
        "cosyn_point": 20.201689065172104,
        "count_numeric": 3.464101552963257,
        "native_text_replay": 8191.0,
        "ocr_document": 4.189212302095257,
        "pixmo_caption": 29.160138869367074,
        "pixmo_points_basic": 33.28980797799886,
        "pixmo_points_high_frequency": 28.820370947258198,
        "pixmo_transcript": 29.65188992714684,
    },
}
"""Loss-weight calibration at 8,192 tokens for each phase."""


def build_visual_sources(
    phase: str,
    sequence_length: int,
    artifact_root: str = DEFAULT_ALIGNMENT_ARTIFACT_ROOT,
    *,
    split: str = "train",
) -> dict[str, Config]:
    """Configure visual sources and their prepared image-disjoint selections.

    Source configs support recipe and CLI overrides. This function reads selection metadata
    without loading datasets or images.

    :param phase: ``bridge``, ``perception``, or ``joint``.
    :param sequence_length: Maximum serialized example length.
    :param artifact_root: Directory containing the prepared alignment datasets and selections.
    :param split: Logical ``train`` or ``validation`` split.
    :returns: Dataset configs keyed by source name.
    """
    if phase not in ALIGNMENT_LOSS_TARGETS:
        raise ValueError(f"Unknown alignment phase {phase!r}")
    if split not in ("train", "validation"):
        raise ValueError(f"Unknown alignment split {split!r}")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    root = Path(artifact_root)
    common: dict[str, Any] = {
        "max_crops": 8,
        "max_sequence_length": sequence_length,
        "loss_token_weighting": "root_subsegments_root_tokens",
        "message_format": "document",
        "seed": 0,
    }

    sources: dict[str, Config] = {
        "pixmo_caption": PixMoCapDatasetConfig(
            dataset_path=str(root / "pixmo-cap-content-disjoint-v1/dataset"),
            split=split,
            require_split=True,
            mode="caption",
            fixed_prompt="Description:",
            style_length_conditioning=False,
            **common,
        ),
        "pixmo_transcript": PixMoCapDatasetConfig(
            dataset_path=str(root / "pixmo-cap-content-disjoint-v1/dataset"),
            split=split,
            require_split=True,
            mode="transcript",
            require_transcript=True,
            fixed_prompt="Transcript:",
            style_length_conditioning=False,
            **common,
        ),
    }
    if phase == "bridge":
        return sources
    extend = _PHASE_SOURCE_EXTENSIONS.get(phase)
    if extend is None:
        raise ValueError(
            f"Visual sources for the {phase!r} phase are provided by the alignment phases layer "
            "(olmo_core.internal.vision_alignment_phases); import it to register them."
        )
    return extend(sources, root, common, phase, split)


_PHASE_SOURCE_EXTENSIONS: dict[str, Callable[..., dict[str, Config]]] = {}
"""Perception and joint source builders, registered by the phases layer so the bridge path never
imports their data adapters."""


def register_phase_sources(phase: str, builder: Callable[..., dict[str, Config]]) -> None:
    """Register the visual-source builder of a non-bridge phase."""
    _PHASE_SOURCE_EXTENSIONS[phase] = builder
