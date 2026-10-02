from olmo_core.data.multimodal.pixmo_cap import PixMoCapDatasetConfig
from olmo_core.internal.vision_alignment_data import (
    ALIGNMENT_LOSS_TARGETS,
    ALIGNMENT_MEAN_LOSS_WEIGHTS,
    build_visual_sources,
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
