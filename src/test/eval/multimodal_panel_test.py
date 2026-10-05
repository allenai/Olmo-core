import importlib
import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from olmo_core.config import Config
from olmo_core.data.multimodal.alignment import (
    MultimodalMixtureConfig,
    MultimodalSourceConfig,
)
from olmo_core.data.multimodal.collator import MultimodalCollator
from olmo_core.data.multimodal.sequence_builder import build_branched_sequence
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.eval import multimodal_panel as panel
from olmo_core.eval.multimodal_decoding import prepare_prompts
from olmo_core.train.callbacks.multimodal import MultimodalEvaluatorCallbackConfig


@dataclass
class SourceConfig(Config):
    split: str = "validation"

    def build(self, tokenizer):
        return Source(self)


class Source:
    def __init__(self, config):
        self.config = config

    def __len__(self):
        return 100

    def get(self, index, epoch=0):
        result = build_branched_sequence(
            prefix_ids=[1, 9],
            branches=[([5], [index + 20, epoch + 30]), ([6], [80])],
            eos_id=1,
            image_token_ids=frozenset({9}),
            loss_token_weighting="root_subsegments_root_tokens",
        )
        result["images"] = np.full((1, 1, 3), index, dtype=np.float32)
        result["pooled_patches_idx"] = np.zeros((1, 1), dtype=np.int64)
        return result

    def raw_image_references(self, index):
        return (f"image-{index}",)


@pytest.fixture
def frozen_panel(tmp_path):
    selected = tmp_path / "saved.indices"
    selected.write_text("3\n27\n41\n")
    sources, panels = {}, {}
    for name in panel.SCORE_SOURCES:
        source = MultimodalSourceConfig(dataset=SourceConfig(), selection_path=str(selected))
        if name in panel.OCR_COMPONENTS:
            source = panel.SavedPanelSourceConfig(
                dataset=source.dataset,
                selection_path=source.selection_path,
                row_indices=[3, 27],
            )
        sources[name] = source
        panels[name] = {
            "component": panel.OCR_COMPONENTS.get(name),
            "rows": [
                {
                    "panel_index": index,
                    "base_source_index": row,
                    "added_to_saved_panel": False,
                }
                for index, row in enumerate([3, 27])
            ],
        }
    evaluator = MultimodalEvaluatorCallbackConfig(
        eval_dataset=MultimodalMixtureConfig(tokenizer=TokenizerConfig.dolma2(), sources=sources),
        sequence_length=2560,
        examples_per_source=2,
        rank_batch_size=4,
        blank_image_sources=[],
        matched_image_sources=[],
    )
    payload = {
        "version": 1,
        "evaluator": evaluator.as_config_dict(),
        "manifest": {"version": 2, "examples_per_source": 2, "panels": panels},
    }
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(payload))
    return path, payload, evaluator


@pytest.mark.parametrize("legacy", [False, True])
def test_frozen_panel_loads_without_archive_imports_or_dataset_reads(
    frozen_panel, monkeypatch, legacy
):
    path, payload, expected = frozen_panel
    if legacy:
        for name in panel.OCR_COMPONENTS:
            payload["evaluator"]["eval_dataset"]["sources"][name][
                "_CLASS_"
            ] = "configs.vision_moe.vision_alignment.decoded_comparison.panel.SavedPanelSourceConfig"
        path.write_text(json.dumps(payload))
    before = path.read_bytes()
    import_module = importlib.import_module

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith("configs.")
        return import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", guarded_import)
    monkeypatch.setattr(SourceConfig, "build", lambda *args: pytest.fail("Opened source data"))
    monkeypatch.setattr(panel, "_load_indices", lambda *args: pytest.fail("Opened selections"))
    actual, manifest = panel.load_frozen_evaluator(path)
    assert actual == expected
    assert manifest == payload["manifest"]
    assert path.read_bytes() == before
    assert all(
        type(actual.eval_dataset.sources[name]) is panel.SavedPanelSourceConfig
        for name in panel.OCR_COMPONENTS
    )


@pytest.mark.parametrize("corruption", ["version", "count", "duplicate", "component", "indices"])
def test_frozen_panel_rejects_inconsistent_definitions(frozen_panel, corruption):
    path, payload, _ = frozen_panel
    manifest = payload["manifest"]
    if corruption == "version":
        payload["version"] = 10
    elif corruption == "count":
        manifest["examples_per_source"] = 3
    elif corruption == "duplicate":
        manifest["panels"]["scalar_count"]["rows"][1]["base_source_index"] = 3
    elif corruption == "component":
        manifest["panels"]["ocr_info_qa"]["component"] = "text_vqa"
    else:
        manifest["panels"]["ocr_info_qa"]["rows"][0]["base_source_index"] = 31
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        panel.load_frozen_evaluator(path)


def test_frozen_panel_indices_validate_all_sources_without_building_data(frozen_panel, monkeypatch):
    path, payload, _ = frozen_panel
    evaluator, manifest = panel.load_frozen_evaluator(path)
    monkeypatch.setattr(SourceConfig, "build", lambda *args: pytest.fail("Opened source data"))
    panel.validate_frozen_panel_indices(evaluator, manifest)
    assert manifest == payload["manifest"]


@pytest.mark.parametrize("source", ["scalar_count", "pixmo_points_basic", "pixmo_caption"])
@pytest.mark.parametrize("change", ["manifest", "reorder", "selection"])
def test_non_ocr_panel_rejects_stale_or_reordered_indices(frozen_panel, source, change):
    path, payload, _ = frozen_panel
    rows = payload["manifest"]["panels"][source]["rows"]
    if change == "manifest":
        rows[0]["base_source_index"] = 31
    elif change == "reorder":
        rows[0]["base_source_index"], rows[1]["base_source_index"] = 27, 3
    else:
        selected = path.with_name(f"{source}.indices")
        selected.write_text("27\n3\n41\n")
        payload["evaluator"]["eval_dataset"]["sources"][source]["selection_path"] = str(selected)
    path.write_text(json.dumps(payload))
    evaluator, manifest = panel.load_frozen_evaluator(path)
    with pytest.raises(ValueError, match=f"Frozen panel {source} row identities differ"):
        panel.validate_frozen_panel_indices(evaluator, manifest)


@pytest.mark.parametrize("repeat", [1, 2, 3])
def test_frozen_panel_indices_preserve_selection_repeat_mapping(frozen_panel, repeat):
    path, _, _ = frozen_panel
    evaluator, manifest = panel.load_frozen_evaluator(path)
    evaluator.eval_dataset.sources["scalar_count"].selection_repeat = repeat
    expected = [3, 27] if repeat == 1 else [3 * repeat, 3 * repeat + 1]
    for row, base_index in zip(manifest["panels"]["scalar_count"]["rows"], expected):
        row["base_source_index"] = base_index
    panel.validate_frozen_panel_indices(evaluator, manifest)


@pytest.mark.parametrize("source", list(panel.SCORE_SOURCES))
def test_panel_rows_reject_wrong_base_identity_or_source(frozen_panel, source):
    _, payload, _ = frozen_panel
    manifest = payload["manifest"]
    row = {"source": source, "panel_index": 1, "base_source_index": 27}
    panel.validate_panel_rows([row], source, manifest)
    with pytest.raises(ValueError, match="row identities differ"):
        panel.validate_panel_rows([{**row, "base_source_index": 3}], source, manifest)
    with pytest.raises(ValueError, match="row identities differ"):
        panel.validate_panel_rows([{**row, "source": "another-source"}], source, manifest)


@pytest.mark.parametrize("source", ["scalar_count", "ocr_doc_qa"])
def test_frozen_panel_indices_reject_short_or_unapproved_selections(frozen_panel, source):
    path, _, _ = frozen_panel
    evaluator, manifest = panel.load_frozen_evaluator(path)
    selected = path.with_name(f"{source}.indices")
    selected.write_text("3\n")
    evaluator.eval_dataset.sources[source].selection_path = str(selected)
    with pytest.raises(ValueError, match="too few rows|absent from the saved selection"):
        panel.validate_frozen_panel_indices(evaluator, manifest)


def test_saved_selection_retains_tokens_masks_images_and_first_reference(frozen_panel):
    path, _, _ = frozen_panel
    config = panel.load_frozen_evaluator(path)[0].eval_dataset.sources["ocr_doc_qa"]
    dataset, original = config.build(None), config.dataset.build(None)
    selected, expected = dataset.get(1, epoch=3), original.get(27, epoch=3)
    assert selected.keys() == expected.keys()
    for key in selected:
        np.testing.assert_array_equal(selected[key], expected[key])
    assert dataset.raw_image_references(1) == ("image-27",)

    batch = MultimodalCollator(pad_token_id=0, pad_sequence_length=32)([selected])
    tokenizer = SimpleNamespace(
        eos_token_id=1,
        pad_token_id=0,
        decode=lambda ids, **kwargs: " ".join(map(str, ids)),
    )
    inputs, lengths, records = prepare_prompts(
        batch, tokenizer, [{"panel_index": 1, "base_source_index": 27}], "ocr_doc_qa"
    )
    assert lengths == [3]
    assert records[0]["prompt_token_ids"] == [1, 9, 5]
    assert records[0]["reference_token_ids"] == [47, 33, 1]
    assert records[0]["reference_complete"]
    assert inputs["input_ids"][0, 3:].count_nonzero() == 0
    assert inputs["router_token_mask"][0].sum() == 3


@pytest.mark.parametrize("indices", [[], [3, 3], [-1], [42]])
def test_saved_selection_rejects_invalid_or_unapproved_rows(frozen_panel, indices):
    _, _, evaluator = frozen_panel
    config = evaluator.eval_dataset.sources["ocr_doc_qa"]
    config.row_indices = indices
    with pytest.raises(ValueError):
        config.build(None)


def test_saved_selection_requires_disjoint_validation_additions(frozen_panel, tmp_path):
    _, _, evaluator = frozen_panel
    config = evaluator.eval_dataset.sources["ocr_doc_qa"]
    config.row_indices = [3, 4]
    config.added_validation_indices = [4]
    excluded = tmp_path / "train.indices"
    excluded.write_text("4\n")
    config.excluded_selection_paths = [str(excluded)]
    with pytest.raises(ValueError, match="excluded training"):
        config.build(None)
    config.dataset.split = "train"
    with pytest.raises(ValueError, match="original validation"):
        config.build(None)
