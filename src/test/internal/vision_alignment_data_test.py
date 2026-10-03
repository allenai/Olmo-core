import hashlib
from pathlib import Path

import pytest

from olmo_core.data.multimodal.pixmo_cap import PixMoCapDatasetConfig
from olmo_core.internal import vision_alignment_data
from olmo_core.internal.vision_alignment_data import (
    ALIGNMENT_ARTIFACT_MANIFESTS,
    ALIGNMENT_LOSS_TARGETS,
    ALIGNMENT_MEAN_LOSS_WEIGHTS,
    ALIGNMENT_ONE_ANNOTATION_MEAN_LOSS_WEIGHTS,
    DEFAULT_ALIGNMENT_ARTIFACT_ROOT,
    build_visual_sources,
    has_calibrated_artifacts,
)


def test_bridge_visual_recipe_configs_and_calibration_names(tmp_path):
    sources = build_visual_sources("bridge", 2560, str(tmp_path))
    assert set(sources) == set(ALIGNMENT_LOSS_TARGETS["bridge"])
    assert set(sources) == set(ALIGNMENT_MEAN_LOSS_WEIGHTS["bridge"])
    for name, source in sources.items():
        assert isinstance(source, PixMoCapDatasetConfig), name
        assert source.max_sequence_length == 2560
        assert source.message_format == "document"
        assert source.require_split is True
    assert sources["pixmo_caption"].mode == "caption"
    assert sources["pixmo_transcript"].mode == "transcript"
    assert sources["pixmo_transcript"].require_transcript is True


def test_bridge_requires_no_perception_metadata(tmp_path):
    sources = build_visual_sources("bridge", 2560, str(tmp_path), split="validation")
    assert sources["pixmo_caption"].split == "validation"
    assert sources["pixmo_caption"].dataset_path.startswith(str(tmp_path))


def test_one_annotation_calibration_covers_only_known_sources():
    for phase, means in ALIGNMENT_ONE_ANNOTATION_MEAN_LOSS_WEIGHTS.items():
        assert set(means) <= set(ALIGNMENT_MEAN_LOSS_WEIGHTS[phase])
        # One annotation carries less supervised weight than all of an image's annotations.
        assert all(
            0 < mean < ALIGNMENT_MEAN_LOSS_WEIGHTS[phase][name] for name, mean in means.items()
        )


def test_a_byte_identical_artifact_copy_keeps_the_calibration(tmp_path, monkeypatch):
    contents = {
        "pixmo-cap-content-disjoint-v1/build-state.json": b"captions",
        "perception-provenance-v2/build-state.json": b"perception",
    }
    for name, content in contents.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(content)
    monkeypatch.setattr(
        vision_alignment_data,
        "ALIGNMENT_ARTIFACT_MANIFESTS",
        {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()},
    )
    assert has_calibrated_artifacts(str(tmp_path), "perception")
    assert has_calibrated_artifacts(DEFAULT_ALIGNMENT_ARTIFACT_ROOT, "joint")
    (tmp_path / "perception-provenance-v2/build-state.json").write_bytes(b"rebuilt")
    assert not has_calibrated_artifacts(str(tmp_path), "perception")
    assert has_calibrated_artifacts(str(tmp_path), "bridge")  # reads only the caption artifacts
    assert not has_calibrated_artifacts(str(tmp_path / "elsewhere"), "bridge")


@pytest.mark.skipif(
    not Path(DEFAULT_ALIGNMENT_ARTIFACT_ROOT).is_dir(), reason="default artifacts not mounted"
)
def test_pinned_manifests_are_the_default_artifacts():
    for name, digest in ALIGNMENT_ARTIFACT_MANIFESTS.items():
        path = Path(DEFAULT_ALIGNMENT_ARTIFACT_ROOT) / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
