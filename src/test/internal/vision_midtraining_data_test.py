import json
from pathlib import Path

import pytest

from olmo_core.data.multimodal.alignment import MultimodalSourceConfig
from olmo_core.data.multimodal.mixture_weights import sampling_weights_from_loss_mass
from olmo_core.data.multimodal.pixmo_cap import PixMoCapDatasetConfig
from olmo_core.data.multimodal.pixmo_points import PixMoPointsDatasetConfig
from olmo_core.data.multimodal.vision_alignment_perception import (
    VisionAlignmentAuditedAlignmentDatasetConfig,
    VisionAlignmentOcrDocumentDatasetConfig,
)
from olmo_core.internal.vision_alignment_data import ALIGNMENT_LOSS_TARGETS
from olmo_core.internal.vision_midtraining_data import (
    DEFAULT_VISUAL_EXAMPLE_WEIGHTS,
    DEFAULT_VISUAL_LOSS_SHARES,
    DEFAULT_VISUAL_MEAN_LOSS_WEIGHTS,
    build_visual_sources,
    loss_mass_targets,
)


def _expected_loss_mass(probabilities, means):
    masses = {name: probability * means[name] for name, probability in probabilities.items()}
    total = sum(masses.values())
    return {name: mass / total for name, mass in masses.items()}


@pytest.fixture
def artifacts(tmp_path):
    sources = {
        name: {
            split: {
                "physical_split": (
                    "train"
                    if name in ("audited_alignment", "pixmo_caption", "pixmo_points_basic")
                    else split
                ),
                "selection": {"path": f"selections/{name}-{split}.indices"},
            }
            for split in ("train", "validation")
        }
        for name in ALIGNMENT_LOSS_TARGETS["perception"]
    }
    selection_root = tmp_path / "perception-provenance-v2"
    (selection_root / "selections").mkdir(parents=True)
    for splits in sources.values():
        for split, selection in splits.items():
            (selection_root / selection["selection"]["path"]).write_text(
                "2\n0\n" if split == "train" else "1\n"
            )
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


@pytest.fixture
def means():
    names = [*DEFAULT_VISUAL_EXAMPLE_WEIGHTS, *DEFAULT_VISUAL_LOSS_SHARES, "text_midtraining"]
    return {name: float(2**i) for i, name in enumerate(names)}


@pytest.mark.parametrize("length,crops", [(8192, 8), (4096, 4)])
@pytest.mark.parametrize("split", ["train", "validation"])
def test_visual_sources_preserve_formats_and_selections(artifacts, length, crops, split):
    sources = build_visual_sources(
        sequence_length=length,
        max_crops=crops,
        alignment_artifact_root=artifacts,
        midtraining_artifact_root=artifacts,
        split=split,
    )
    assert set(sources) == set(DEFAULT_VISUAL_EXAMPLE_WEIGHTS) | set(DEFAULT_VISUAL_LOSS_SHARES)
    assert len(sources) == 8
    selected = {
        name: source
        for name, source in sources.items()
        if isinstance(source, MultimodalSourceConfig)
    }
    assert set(selected) == {"ocr_document", "audited_alignment"} | (
        {"pixmo_cap", "pixmo_points_basic"} if split == "train" else set()
    )
    if split == "train":
        assert selected["pixmo_cap"].selection_repeat == 1
        assert selected["pixmo_points_basic"].selection_repeat == 2
        for name, alignment_name in (
            ("pixmo_cap", "pixmo_caption"),
            ("pixmo_points_basic", "pixmo_points_basic"),
        ):
            source = selected[name]
            assert source.dataset.split == "train"
            assert source.selection_path.endswith(f"{alignment_name}-train.indices")
            assert len(source.excluded_selection_paths) == 1
            assert source.excluded_selection_paths[0].endswith(
                f"{alignment_name}-validation.indices"
            )
    datasets = {
        name: source.dataset if isinstance(source, MultimodalSourceConfig) else source
        for name, source in sources.items()
    }
    for config in datasets.values():
        assert config.max_sequence_length == length
        assert config.max_crops == crops
        assert config.message_format == "document"
        assert config.loss_token_weighting == "none"

    ocr = sources["ocr_document"]
    assert isinstance(ocr, MultimodalSourceConfig)
    assert isinstance(ocr.dataset, VisionAlignmentOcrDocumentDatasetConfig)
    assert ocr.dataset.source_names == ("text_vqa", "doc_qa", "info_qa", "chart_qa_weighted")
    assert ocr.dataset.split == split
    assert ocr.dataset.skip_bad_rows is False
    assert ocr.selection_path.endswith(f"ocr_document-{split}.indices")
    assert ocr.excluded_selection_paths == []

    audited = sources["audited_alignment"]
    assert isinstance(audited, MultimodalSourceConfig)
    assert isinstance(audited.dataset, VisionAlignmentAuditedAlignmentDatasetConfig)
    assert audited.dataset.split == "train"
    assert audited.dataset.visualweb_path.endswith("visualwebinstruct-filtered")
    assert audited.dataset.geo170k_path.endswith("geo170k-align")
    assert audited.dataset.visualweb_fingerprint == "prepared-visualweb"
    assert audited.dataset.geo170k_fingerprint == "prepared-geo170k"
    assert audited.dataset.min_formatting == 4
    assert audited.dataset.min_visual_dependency == 4
    assert audited.dataset.min_relevance == 4
    assert audited.selection_path.endswith(f"audited_alignment-{split}.indices")
    other_split = "validation" if split == "train" else "train"
    assert len(audited.excluded_selection_paths) == 1
    assert audited.excluded_selection_paths[0].endswith(f"audited_alignment-{other_split}.indices")

    caption = datasets["pixmo_cap"]
    assert caption.split == split
    assert caption.mode == "transcript_and_caption"
    assert caption.style_length_conditioning is True
    assert sources["pixmo_count"].scalar_count_replay is False
    assert sources["pixmo_count"].mode == "grounded"
    assert sources["pixmo_count"].explicit_grounding_prompts is True
    assert sources["pixmo_count"].split == (
        "grounded_validation" if split == "validation" else "train"
    )
    scalar = sources["pixmo_count_scalar"]
    assert scalar.mode == "scalar_count"
    assert scalar.scalar_count_replay is False
    assert scalar.explicit_grounding_prompts is False
    assert scalar.dataset_path == sources["pixmo_count"].dataset_path
    assert scalar.split == sources["pixmo_count"].split
    assert sources["cosyn_point"].explicit_grounding_prompts is True
    for name in ("pixmo_points_basic", "pixmo_points_high_frequency"):
        points = datasets[name]
        assert points.split == split
        assert points.counting == "both"
        assert points.both_mode == "duplicate"
        assert points.explicit_grounding_prompts is True


@pytest.mark.parametrize(
    "name,config_class,repeat,indices",
    [
        ("pixmo_cap", PixMoCapDatasetConfig, 1, [2, 0]),
        ("pixmo_points_basic", PixMoPointsDatasetConfig, 2, [4, 5, 0, 1]),
    ],
)
def test_training_selections_preserve_rows_and_duplicate_styles(
    artifacts, monkeypatch, name, config_class, repeat, indices
):
    reads = []

    class Rows:
        def __len__(self):
            return 6 * repeat

        def get(self, index, epoch=0):
            reads.append(index)
            return {"row": index, "epoch": epoch}

        def raw_image_references(self, index):
            reads.append(index)
            return (f"image-{index // repeat}",)

    monkeypatch.setattr(config_class, "build", lambda self, tokenizer: Rows())
    source = build_visual_sources(alignment_artifact_root=artifacts)[name]
    assert isinstance(source, MultimodalSourceConfig)
    restored = MultimodalSourceConfig.from_dict(source.as_config_dict())
    assert restored == source
    for config in (source, restored):
        reads.clear()
        selected = config.build(object())
        assert reads == []
        assert selected.indices.tolist() == indices
        assert selected.get(1, epoch=7) == {"row": indices[1], "epoch": 7}
        assert [selected.raw_image_references(i) for i in range(len(selected))] == [
            (f"image-{row}",) for row in (2, 0) for _ in range(repeat)
        ]


@pytest.mark.parametrize("target_text", [0.0, 0.5, 0.8, 0.9])
@pytest.mark.parametrize("mean_scale", [0.01, 1.0, 100.0])
def test_two_stage_mixer_retains_loss_shares_and_example_ratios(means, target_text, mean_scale):
    means = {name: value * mean_scale ** (i % 3) for i, (name, value) in enumerate(means.items())}
    if target_text == 0:
        means.pop("text_midtraining")
    targets = loss_mass_targets(means, target_text_loss_mass=target_text)
    probabilities = sampling_weights_from_loss_mass(targets, means)
    actual_mass = _expected_loss_mass(probabilities, means)
    assert sum(targets.values()) == pytest.approx(1.0)
    assert actual_mass.get("text_midtraining", 0.0) == pytest.approx(target_text)
    visual_mass = 1.0 - target_text
    example_mass = sum(actual_mass[name] for name in DEFAULT_VISUAL_EXAMPLE_WEIGHTS)
    assert example_mass == pytest.approx(visual_mass)
    example_probability = sum(probabilities[name] for name in DEFAULT_VISUAL_EXAMPLE_WEIGHTS)
    example_weight = sum(DEFAULT_VISUAL_EXAMPLE_WEIGHTS.values())
    for name, weight in DEFAULT_VISUAL_EXAMPLE_WEIGHTS.items():
        assert probabilities[name] / example_probability == pytest.approx(weight / example_weight)
    assert probabilities["ocr_document"] / example_probability == pytest.approx(0.12)
    assert probabilities["pixmo_count_scalar"] / example_probability == pytest.approx(0.10)
    assert probabilities["audited_alignment"] / example_probability == pytest.approx(0.08)


def test_default_visual_example_mix_preserves_core_ratios():
    assert DEFAULT_VISUAL_LOSS_SHARES == {}
    core_weights = {
        "pixmo_cap": 0.666666666667,
        "pixmo_points_basic": 0.122201237231,
        "pixmo_count": 0.058384427650,
        "pixmo_points_high_frequency": 0.096695558356,
        "cosyn_point": 0.056052110097,
    }
    assert sum(DEFAULT_VISUAL_EXAMPLE_WEIGHTS.values()) == pytest.approx(1.0)
    assert sum(DEFAULT_VISUAL_EXAMPLE_WEIGHTS[name] for name in core_weights) == pytest.approx(0.7)
    for name, weight in core_weights.items():
        assert DEFAULT_VISUAL_EXAMPLE_WEIGHTS[name] == pytest.approx(0.7 * weight)


def _example_weights_without_reserved_sources():
    return {
        name: weight
        for name, weight in DEFAULT_VISUAL_EXAMPLE_WEIGHTS.items()
        if name not in ("ocr_document", "audited_alignment")
    }


def test_explicit_visual_loss_shares(means):
    examples = _example_weights_without_reserved_sources()
    targets = loss_mass_targets(
        means,
        visual_example_weights=examples,
        visual_loss_shares={"ocr_document": 0.25, "audited_alignment": 0.15},
    )
    assert targets["text_midtraining"] == pytest.approx(0.9)
    assert targets["ocr_document"] == pytest.approx(0.025)
    assert targets["audited_alignment"] == pytest.approx(0.015)
    assert sum(targets[name] for name in examples) == pytest.approx(0.06)


def test_reserved_loss_share_does_not_silently_override_default_example_weight(means):
    with pytest.raises(ValueError, match="must be disjoint"):
        loss_mass_targets(means, visual_loss_shares={"ocr_document": 0.25})


@pytest.mark.parametrize("bad_value", [0, -1, True, float("inf"), float("nan")])
def test_reject_invalid_calibration_means(means, bad_value):
    means["ocr_document"] = bad_value
    with pytest.raises(ValueError):
        loss_mass_targets(means)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_reject_mismatched_calibration_sources(means, change):
    if change == "missing":
        means.pop("ocr_document")
    else:
        means["native_text_replay"] = 100.0
    with pytest.raises(ValueError):
        loss_mass_targets(means)


@pytest.mark.parametrize("ratio", [-0.1, 1.1, True, False, float("inf"), float("nan")])
def test_reject_invalid_text_ratios(means, ratio):
    with pytest.raises(ValueError):
        loss_mass_targets(means, target_text_loss_mass=ratio)


@pytest.mark.parametrize(
    "shares",
    [
        {},
        {"ocr_document": 0.1},
        {"ocr_document": 0.1, "audited_alignment": 0.1, "unknown": 0.1},
        {"ocr_document": 0, "audited_alignment": 0.1},
        {"ocr_document": -0.1, "audited_alignment": 0.1},
        {"ocr_document": True, "audited_alignment": 0.1},
        {"ocr_document": float("nan"), "audited_alignment": 0.1},
        {"ocr_document": 0.5, "audited_alignment": 0.5},
        {"ocr_document": 0.8, "audited_alignment": 0.4},
    ],
)
def test_reject_invalid_or_uncalibrated_visual_shares(means, shares):
    with pytest.raises(ValueError):
        loss_mass_targets(
            means,
            visual_example_weights=_example_weights_without_reserved_sources(),
            visual_loss_shares=shares,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sequence_length": 1},
        {"sequence_length": 8192.0},
        {"max_crops": 0},
        {"max_crops": True},
        {"split": "all"},
    ],
)
def test_reject_invalid_visual_config(artifacts, kwargs):
    with pytest.raises(ValueError):
        build_visual_sources(alignment_artifact_root=artifacts, **kwargs)


def test_default_visual_means_match_bounded_calibration():
    calibration_path = (
        Path(__file__).resolve().parents[3]
        / "configs/vision_moe/vision_midtraining/visual_calibration_v1.json"
    )
    calibration = json.loads(calibration_path.read_text())
    assert DEFAULT_VISUAL_MEAN_LOSS_WEIGHTS == calibration["mean_loss_weight"]


@pytest.mark.parametrize("text_share", [0.0, 0.5, 0.9])
def test_custom_visual_groups_and_endpoint(text_share):
    means = {"captions": 200.0, "points": 100.0, "documents": 5.0}
    if text_share > 0:
        means["text_midtraining"] = 8191.0
    targets = loss_mass_targets(
        means,
        target_text_loss_mass=text_share,
        visual_example_weights={"captions": 3.0, "points": 1.0},
        visual_loss_shares={"documents": 0.2},
    )
    probabilities = sampling_weights_from_loss_mass(targets, means)
    actual = _expected_loss_mass(probabilities, means)
    assert actual.get("text_midtraining", 0.0) == pytest.approx(text_share)
    assert actual["documents"] == pytest.approx((1.0 - text_share) * 0.2)
    assert probabilities["captions"] / probabilities["points"] == pytest.approx(3.0)
    assert sum(actual.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "examples,shares",
    [({"visual": 2.0}, {}), ({}, {"visual": 1.0})],
)
def test_single_visual_group(examples, shares):
    means = {"visual": 10.0, "text_midtraining": 8191.0}
    assert loss_mass_targets(
        means, visual_example_weights=examples, visual_loss_shares=shares
    ) == pytest.approx({"visual": 0.1, "text_midtraining": 0.9})


@pytest.mark.parametrize(
    "examples,shares",
    [
        ({"visual": 0.0}, {}),
        ({"visual": True}, {}),
        ({"visual": float("nan")}, {}),
        ({"visual": float("inf")}, {}),
        ({"visual": 1.0}, {"visual": 0.1}),
        ({"text_midtraining": 1.0}, {}),
        ({}, {"text_midtraining": 1.0}),
        ({}, {}),
        ({}, {"visual": 0.5}),
        ({}, {"visual": 1.1}),
        ({"": 1.0}, {}),
    ],
)
def test_invalid_visual_weight_groups(examples, shares):
    with pytest.raises(ValueError):
        loss_mass_targets(
            {"visual": 10.0, "text_midtraining": 8191.0},
            visual_example_weights=examples,
            visual_loss_shares=shares,
        )


@pytest.mark.parametrize("means", [None, {"unused_visual": float("nan")}])
def test_text_only_target_needs_no_visual_calibration(means):
    assert loss_mass_targets(
        means,
        target_text_loss_mass=1.0,
        visual_example_weights={"unused_visual": float("nan")},
        visual_loss_shares={"unused_visual": -1.0},
    ) == {"text_midtraining": 1.0}
