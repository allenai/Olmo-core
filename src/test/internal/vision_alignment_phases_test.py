"""Perception/joint visual sources of the alignment recipe (the phases layer)."""

import json

import pytest

from olmo_core.data.multimodal.alignment import MultimodalSourceConfig
from olmo_core.data.multimodal.pixmo_cap import PixMoCapDatasetConfig
from olmo_core.internal import (  # noqa: F401  (registers the builders)
    vision_alignment_phases,
)
from olmo_core.internal.vision_alignment_data import (
    ALIGNMENT_LOSS_TARGETS,
    ALIGNMENT_MEAN_LOSS_WEIGHTS,
    build_visual_sources,
)


@pytest.fixture
def artifacts(tmp_path):
    sources = {}
    for name in ALIGNMENT_LOSS_TARGETS["perception"]:
        sources[name] = {
            split: {
                "physical_split": "train" if name == "audited_alignment" else split,
                "selection": {"path": f"selections/{name}-{split}.indices"},
            }
            for split in ("train", "validation")
        }
    selection_root = tmp_path / "perception-provenance-v2"
    selection_root.mkdir()
    (selection_root / "vision-alignment-perception-provenance.json").write_text(
        json.dumps(
            {
                "source_spec": {
                    "finevision_root": "/finevision",
                    "finevision_visualweb_fingerprint": "prepared-visualweb",
                    "finevision_geo170k_fingerprint": "prepared-geo170k",
                    "ocr_source_names": ["text_vqa", "doc_qa", "info_qa", "chart_qa_weighted"],
                },
                "sources": sources,
            }
        )
    )
    return str(tmp_path)


@pytest.mark.parametrize("phase,length", [("bridge", 2560), ("perception", 2560), ("joint", 8192)])
def test_visual_recipe_configs_and_calibration_names(artifacts, phase, length):
    sources = build_visual_sources(phase, length, artifacts)
    expected = set(ALIGNMENT_LOSS_TARGETS[phase]) - {"native_text_replay"}
    assert set(sources) == expected
    assert list(sources) == sorted(sources)
    assert set(ALIGNMENT_MEAN_LOSS_WEIGHTS[phase]) == set(ALIGNMENT_LOSS_TARGETS[phase])
    assert sum(ALIGNMENT_LOSS_TARGETS[phase].values()) == pytest.approx(1.0)
    for source in sources.values():
        config = source.dataset if isinstance(source, MultimodalSourceConfig) else source
        assert config.max_sequence_length == length
        assert config.message_format == "document"
        assert config.loss_token_weighting == "root_subsegments_root_tokens"
    if phase == "bridge":
        assert isinstance(sources["pixmo_caption"], PixMoCapDatasetConfig)
        assert sources["pixmo_transcript"].require_transcript
    else:
        assert sources["pixmo_points_basic"].dataset.counting is False
        assert sources["pixmo_points_basic"].dataset.both_mode == "per_annotation"
        count_name = "count_numeric" if phase == "joint" else "scalar_count"
        assert sources[count_name].dataset.mode == "scalar_count"
        assert sources[count_name].selection_path.endswith("scalar_count-train.indices")


def test_validation_uses_physical_splits_without_equating_unrelated_row_indices(artifacts):
    sources = build_visual_sources("joint", 8192, artifacts, split="validation")
    for name, source in sources.items():
        if name == "audited_alignment":
            assert source.dataset.split == "train"
            assert source.excluded_selection_paths[0].endswith("audited_alignment-train.indices")
        else:
            assert source.dataset.split == "validation"
            assert source.excluded_selection_paths == []
        assert source.selection_path.endswith("-validation.indices")
