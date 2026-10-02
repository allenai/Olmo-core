"""Perception and joint visual sources for the vision-alignment recipe.

Importing this module registers the perception and joint source builders with
:mod:`olmo_core.internal.vision_alignment_data`; the bridge phase needs none of the data
adapters used here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from olmo_core.config import Config
from olmo_core.data.multimodal.alignment import MultimodalSourceConfig
from olmo_core.data.multimodal.pixmo_points import (
    CoSynPointDatasetConfig,
    PixMoCountDatasetConfig,
    PixMoPointsDatasetConfig,
)
from olmo_core.data.multimodal.vision_alignment_perception import (
    VisionAlignmentAuditedAlignmentDatasetConfig,
    VisionAlignmentOcrDocumentDatasetConfig,
)

from .vision_alignment_data import register_phase_sources


def extend_visual_sources(
    sources: dict[str, Config], root: Path, common: dict[str, Any], phase: str, split: str
) -> dict[str, Config]:
    """Add the perception/joint sources to the bridge ``sources`` and apply the prepared,
    image-disjoint selections."""
    # Perception and joint sources live in the phases-and-evals layer; the bridge path never
    # imports them.

    selection_root = root / "perception-provenance-v2"
    with (selection_root / "vision-alignment-perception-provenance.json").open() as stream:
        manifest = json.load(stream)
    spec = manifest["source_spec"]
    count_name = "count_numeric" if phase == "joint" else "scalar_count"
    sources.update(
        {
            "pixmo_points_basic": PixMoPointsDatasetConfig(
                require_split=True,
                kind="basic",
                counting=False,
                both_mode="per_annotation",
                **common,
            ),
            "pixmo_points_high_frequency": PixMoPointsDatasetConfig(
                require_split=True,
                kind="high_frequency",
                counting=False,
                both_mode="per_annotation",
                **common,
            ),
            "cosyn_point": CoSynPointDatasetConfig(require_split=True, **common),
            count_name: PixMoCountDatasetConfig(
                require_split=True, mode="scalar_count", counting="both", **common
            ),
            "ocr_document": VisionAlignmentOcrDocumentDatasetConfig(
                source_names=tuple(spec["ocr_source_names"]), **common
            ),
            "audited_alignment": VisionAlignmentAuditedAlignmentDatasetConfig(
                root=spec["finevision_root"],
                visualweb_path=str(
                    root / "finevision-materialization-v1/visualwebinstruct-filtered"
                ),
                geo170k_path=str(root / "finevision-materialization-v1/geo170k-align"),
                visualweb_fingerprint=spec["finevision_visualweb_fingerprint"],
                geo170k_fingerprint=spec["finevision_geo170k_fingerprint"],
                **common,
            ),
        }
    )
    selected: dict[str, Config] = {}
    other_split = "validation" if split == "train" else "train"
    for name, config in sorted(sources.items()):
        source_name = "scalar_count" if name == "count_numeric" else name
        splits = manifest["sources"][source_name]
        selection = splits[split]
        config.split = selection["physical_split"]  # type: ignore[attr-defined]
        other = splits[other_split]
        excluded = []
        if selection["physical_split"] == other["physical_split"]:
            excluded.append(str(selection_root / other["selection"]["path"]))
        selected[name] = MultimodalSourceConfig(
            dataset=config,
            selection_path=str(selection_root / selection["selection"]["path"]),
            excluded_selection_paths=excluded,
        )
    return selected


register_phase_sources("perception", extend_visual_sources)
register_phase_sources("joint", extend_visual_sources)
