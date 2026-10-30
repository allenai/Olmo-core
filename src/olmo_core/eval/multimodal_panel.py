"""Prepared source panels for matched multimodal checkpoint evaluations."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from olmo_core.data.multimodal.alignment import (
    MultimodalSourceConfig,
    _load_indices,
    _SelectedDataset,
)
from olmo_core.train.callbacks.multimodal import MultimodalEvaluatorCallbackConfig

OCR_COMPONENTS = {
    "ocr_text_vqa": "text_vqa",
    "ocr_doc_qa": "doc_qa",
    "ocr_info_qa": "info_qa",
    "ocr_chart_qa": "chart_qa_weighted",
}
SCORE_SOURCES = {
    "scalar_count": "scalar_count",
    "pixmo_points_basic": "pixmo_points_basic",
    **{name: "ocr_document" for name in OCR_COMPONENTS},
    "pixmo_caption": "pixmo_caption",
}


@dataclass
class SavedPanelSourceConfig(MultimodalSourceConfig):
    """Select prepared rows without changing source formatting or random-number indices.

    ``added_validation_indices`` records previously admitted validation rows outside the
    original selection. Reading a panel never adds rows or rebuilds content inventories.
    """

    row_indices: list[int] = field(default_factory=list)
    added_validation_indices: list[int] = field(default_factory=list)

    def build(self, tokenizer: Any) -> Any:
        """Build the source and select its recorded original row indices."""
        indices = self._load_panel_indices()
        base = MultimodalSourceConfig(dataset=self.dataset).build(tokenizer)
        return _SelectedDataset(base, indices, self.dataset)

    def _load_panel_indices(self) -> np.ndarray:
        if self.selection_path is None:
            raise ValueError("A comparison panel requires a saved held-out selection")
        indices = np.asarray(self.row_indices, dtype=np.int64)
        if not len(indices) or len(np.unique(indices)) != len(indices) or np.any(indices < 0):
            raise ValueError("Panel rows must be nonempty, nonnegative, and unique")
        saved = _load_indices(self.selection_path)
        additions = set(self.added_validation_indices)
        if additions and getattr(self.dataset, "split", None) != "validation":
            raise ValueError("Panel additions must use the original validation split")
        if not set(indices.tolist()) <= set(saved.tolist()) | additions:
            raise ValueError("Panel row is absent from the saved selection and explicit additions")
        if additions - set(indices.tolist()) or additions & set(saved.tolist()):
            raise ValueError("Panel additions must be new rows used in this panel")
        for path in self.excluded_selection_paths:
            if np.intersect1d(indices, _load_indices(path)).size:
                raise ValueError("Panel overlaps an excluded training selection")
        return indices


def load_frozen_evaluator(
    path: Path,
) -> tuple[MultimodalEvaluatorCallbackConfig, dict[str, Any]]:
    """Load a prepared panel without opening datasets, selections, or content inventories.

    Saved configs using the earlier panel class path are adapted before deserialization;
    their source definitions and row identities are unchanged.
    """
    payload = json.loads(path.read_text())
    if payload.get("version") != 1:
        raise ValueError("Unsupported frozen comparison panel version")
    config = payload["evaluator"]
    for source in config["eval_dataset"]["sources"].values():
        if source.get("_CLASS_") == (
            "configs.vision_moe.vision_alignment.decoded_comparison.panel.SavedPanelSourceConfig"
        ):
            source["_CLASS_"] = "olmo_core.eval.multimodal_panel.SavedPanelSourceConfig"
    evaluator = MultimodalEvaluatorCallbackConfig.from_dict(config)
    manifest = payload["manifest"]
    examples = manifest["examples_per_source"]
    if (
        manifest.get("version") != 2
        or not isinstance(examples, int)
        or not 0 < examples <= 512
        or examples != evaluator.examples_per_source
        or list(evaluator.eval_dataset.sources) != list(SCORE_SOURCES)
        or list(manifest["panels"]) != list(SCORE_SOURCES)
        or evaluator.blank_image_sources
        or evaluator.matched_image_sources
    ):
        raise ValueError("Frozen panel config and manifest definitions differ")
    for name, source in evaluator.eval_dataset.sources.items():
        rows = manifest["panels"][name]["rows"]
        indices = [row["base_source_index"] for row in rows]
        if (
            [row["panel_index"] for row in rows] != list(range(examples))
            or any(not isinstance(index, int) or index < 0 for index in indices)
            or len(set(indices)) != examples
        ):
            raise ValueError(f"Frozen panel {name} has invalid or duplicate rows")
        if name in OCR_COMPONENTS and (
            not isinstance(source, SavedPanelSourceConfig)
            or source.row_indices != indices
            or source.added_validation_indices
            != [row["base_source_index"] for row in rows if row["added_to_saved_panel"]]
            or manifest["panels"][name]["component"] != OCR_COMPONENTS[name]
        ):
            raise ValueError(f"Frozen OCR panel {name} config and row identities differ")
    return evaluator, manifest


def validate_panel_rows(rows: list[dict[str, Any]], source: str, manifest: dict[str, Any]) -> None:
    """Check decoded or selected row identities against their frozen panel positions."""
    expected = manifest["panels"][source]["rows"]
    for row in rows:
        index = row["panel_index"]
        if (
            type(index) is not int
            or not 0 <= index < len(expected)
            or row.get("source", source) != source
            or row["base_source_index"] != expected[index]["base_source_index"]
        ):
            raise ValueError(f"Frozen panel {source} row identities differ")


def validate_frozen_panel_indices(
    evaluator: MultimodalEvaluatorCallbackConfig, manifest: dict[str, Any]
) -> None:
    """Verify the ordered selection lists without opening datasets or decoding examples."""
    examples = evaluator.examples_per_source
    for name, source in evaluator.eval_dataset.sources.items():
        if isinstance(source, SavedPanelSourceConfig):
            indices = source._load_panel_indices()
        elif isinstance(source, MultimodalSourceConfig):
            repeat = source.selection_repeat
            if type(repeat) is not int or repeat < 1:
                raise ValueError("selection_repeat must be a positive integer")
            if source.selection_path is None:
                if repeat != 1 or source.excluded_selection_paths:
                    raise ValueError("Repeated or excluded selections require selection_path")
                indices = np.arange(examples)
            else:
                indices = _load_indices(source.selection_path)
                for path in source.excluded_selection_paths:
                    if np.intersect1d(indices, _load_indices(path), assume_unique=True).size:
                        raise ValueError(f"Frozen panel {name} overlaps an excluded selection")
                indices = (indices[:examples, None] * repeat + np.arange(repeat)).reshape(-1)
        else:
            raise ValueError(f"Frozen panel {name} requires a prepared source selection")
        if len(indices) < examples:
            raise ValueError(f"Frozen panel {name} selection has too few rows")
        validate_panel_rows(
            [
                {"panel_index": index, "base_source_index": int(base_index)}
                for index, base_index in enumerate(indices[:examples])
            ],
            name,
            manifest,
        )
