"""Three-phase override-table check of the OLMo 3.5 recipe (perception and joint handoffs).


OLMo 3.5 support in the alignment recipe: with ``recipe.text_config`` every text-side setting is
inherited from the text team's resolved mid-training config and the alignment phase differs from
it exactly by :data:`~olmo_core.internal.vision_alignment.MULTIMODAL_OVERRIDES`; without it the
LM config comes from the checkpoint through the legacy normalizer with EMO cleared.
"""

import dataclasses
import fnmatch
import json
from pathlib import Path

import pytest

from olmo_core.data.multimodal.alignment import MultimodalSourceConfig
from olmo_core.data.multimodal.pixmo_points import (
    CoSynPointDatasetConfig,
    PixMoPointsDatasetConfig,
)
from olmo_core.internal import vision_alignment
from olmo_core.internal.vision_alignment import (
    MULTIMODAL_OVERRIDES,
    VisionAlignmentExperimentConfig,
)
from olmo_core.internal.vision_alignment_data import (
    ALIGNMENT_MEAN_LOSS_WEIGHTS,
    ALIGNMENT_ONE_ANNOTATION_MEAN_LOSS_WEIGHTS,
    DEFAULT_ALIGNMENT_ARTIFACT_ROOT,
)
from olmo_core.nn.transformer import OLMoDDPModelConfig
from olmo_core.train import TrainerConfig
from olmo_core.train.train_module.transformer.config import OLMoDDPTrainModuleConfig

FIXTURE = Path(__file__).parent.parent / "fixtures" / "olmo35_text_midtraining_config.json"


@pytest.fixture
def text_config() -> dict:
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def hero_checkpoint(alignment_recipe, text_config):
    """The alignment fixture's pretraining checkpoint rewritten as the OLMo 3.5 8T checkpoint's
    config.json: the text LM with the legacy ``use_cute_kernel`` key and EMO routing."""
    saved = json.loads((alignment_recipe.base / "config.json").read_text())
    model = json.loads(json.dumps(text_config["model"]))
    for block in [model["block"], *model["block_overrides"].values()]:
        mixer = block["sequence_mixer"]
        if "use_experimental_kernels" in mixer:
            mixer["use_cute_kernel"] = mixer.pop("use_experimental_kernels")
        router = block.get("routed_experts_router")
        if router is not None:
            router["emo"] = {"pool_size": 8}
    saved["model"] = model
    (alignment_recipe.base / "config.json").write_text(json.dumps(saved))
    return alignment_recipe.base


def _flatten(value, prefix=""):
    out = {}
    if isinstance(value, dict) and value:
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}{key}."))
    else:
        out[prefix.rstrip(".")] = json.dumps(value, sort_keys=True)
    return out


def _differing_keys(text: dict, multimodal: dict) -> set[str]:
    """Keys whose values differ between the text config and the multimodal config, with the
    text LM compared against ``model.lm``."""
    # Top-level settings the experiment config of this tree does not define cannot be
    # inherited yet (they arrive with newer text-side code); they are inherited once present.
    known = {f.name for f in dataclasses.fields(VisionAlignmentExperimentConfig)}
    text = {key: value for key, value in text.items() if key in known}
    text_model = text.pop("model")
    flat_text = _flatten(text)
    flat_text.update(_flatten({"model": {"lm": text_model}}))
    flat_mm = _flatten(multimodal)
    return {key for key in set(flat_text) | set(flat_mm) if flat_text.get(key) != flat_mm.get(key)}


def _round_trip(text_config: dict) -> dict:
    """The text config as this tree's classes serialize it, so comparisons see only real
    differences (not key order or defaults filled in by newer classes)."""
    text = json.loads(json.dumps(text_config))
    text["model"] = OLMoDDPModelConfig.from_dict(text["model"]).as_config_dict()
    text["train_module"] = OLMoDDPTrainModuleConfig.from_dict(text["train_module"]).as_config_dict()
    text["trainer"] = TrainerConfig.from_dict(text["trainer"]).as_config_dict()
    return text


def _hero_phases(alignment_recipe) -> dict[str, dict]:
    """Resolved configs of the three phases built from the text config, through the handoffs."""
    override = f"--recipe.text_config={FIXTURE}"
    configs, parent = {}, None
    for phase in ("bridge", "perception", "joint"):
        config = alignment_recipe.build(phase, parent, overrides=[override])
        configs[phase] = config.as_config_dict()
        parent = alignment_recipe.save(config)
    return configs


def _covered(key: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(key, pattern) or key.startswith(pattern + ".")


def test_all_hero_phases_differ_from_the_text_config_only_by_the_override_table(
    alignment_recipe, hero_checkpoint, text_config
):
    text = _round_trip(text_config)
    differing_by_phase = {
        phase: _differing_keys(text, config)
        for phase, config in _hero_phases(alignment_recipe).items()
    }
    for phase, differing in differing_by_phase.items():
        uncovered = sorted(
            key
            for key in differing
            if not any(_covered(key, pattern) for pattern in MULTIMODAL_OVERRIDES)
        )
        assert not uncovered, f"{phase}: differences not in MULTIMODAL_OVERRIDES: {uncovered}"
    # Every table entry explains a real difference in at least one phase (e.g. the joint
    # microbatch and loss split differ while bridge matches the text values).
    all_differing = set().union(*differing_by_phase.values())
    stale = sorted(
        pattern
        for pattern in MULTIMODAL_OVERRIDES
        if not any(_covered(key, pattern) for key in all_differing)
    )
    assert not stale, f"MULTIMODAL_OVERRIDES entries without a difference: {stale}"


@pytest.fixture
def pointing_sources(monkeypatch):
    """Give the fixture's perception/joint mixtures their real multi-annotation source types."""
    stand_in = vision_alignment.build_visual_sources

    def visual_sources(phase, sequence_length, artifact_root, split="train"):
        sources = stand_in(phase, sequence_length, artifact_root, split=split)
        if "cosyn_point" in sources:
            sources["cosyn_point"] = MultimodalSourceConfig(
                dataset=CoSynPointDatasetConfig(split=split),
                selection_path=f"{artifact_root}/cosyn_point.npy",
            )
            for name, kind in (
                ("pixmo_points_basic", "basic"),
                ("pixmo_points_high_frequency", "high_frequency"),
            ):
                sources[name] = PixMoPointsDatasetConfig(split=split, kind=kind)
        return sources

    monkeypatch.setattr(vision_alignment, "build_visual_sources", visual_sources)


def _annotation_sampling(sources) -> dict[str, str]:
    configs = {name: getattr(source, "dataset", source) for name, source in sources.items()}
    return {
        name: config.annotation_sampling
        for name, config in configs.items()
        if hasattr(config, "annotation_sampling")
    }


@pytest.mark.parametrize("document_mode", [True, False])
def test_document_mode_samples_one_annotation_with_its_calibration(
    alignment_recipe, pointing_sources, request, document_mode
):
    if document_mode:
        request.getfixturevalue("hero_checkpoint")
    overrides = [f"--recipe.artifact_root={DEFAULT_ALIGNMENT_ARTIFACT_ROOT}"]
    parent = alignment_recipe.save(alignment_recipe.build(overrides=overrides))
    for phase in ("perception", "joint"):
        config = alignment_recipe.build(phase, parent, include_means=False, overrides=overrides)
        sampling = "one" if document_mode else "all"
        evaluator = config.trainer.callbacks["multimodal_evaluator"]
        for sources in (config.dataset.sources, evaluator.eval_dataset.sources):
            assert _annotation_sampling(sources) == {
                "cosyn_point": sampling,
                "pixmo_points_basic": sampling,
                "pixmo_points_high_frequency": sampling,
            }
        means = dict(ALIGNMENT_MEAN_LOSS_WEIGHTS[phase])
        if document_mode:
            means.update(ALIGNMENT_ONE_ANNOTATION_MEAN_LOSS_WEIGHTS[phase])
        assert config.dataset.mean_loss_weight == means
        parent = alignment_recipe.save(config)


def _write_caption_manifest(root: Path) -> None:
    """The prepared caption selections the stage-1 v3 caption sources read."""
    folder = root / "perception-provenance-v2"
    folder.mkdir(parents=True, exist_ok=True)

    def entry(split):
        return {"physical_split": split, "selection": {"path": f"selections/{split}.indices"}}

    sources = {
        name: {"train": entry("train"), "validation": entry("validation")}
        for name in ("pixmo_caption", "pixmo_transcript")
    }
    (folder / "vision-alignment-perception-provenance.json").write_text(
        json.dumps({"sources": sources})
    )


@pytest.mark.parametrize("document_mode", [True, False])
def test_stage1_v3_data_switch(alignment_recipe, request, tmp_path, document_mode):
    from olmo_core.exceptions import OLMoConfigurationError
    from olmo_core.internal.vision_alignment_data import (
        STAGE1_V3_LOSS_TARGETS,
        STAGE1_V3_MEAN_LOSS_WEIGHTS,
        STAGE1_V3_SOURCES,
    )

    if document_mode:
        request.getfixturevalue("hero_checkpoint")
    _write_caption_manifest(tmp_path / "artifacts")
    v3 = ["--recipe.data=stage1_v3"]
    with pytest.raises(OLMoConfigurationError, match="bridge is caption-only"):
        alignment_recipe.build(overrides=v3)
    parent = alignment_recipe.save(alignment_recipe.build())
    if not document_mode:
        # The shipped means are for one annotation per example.
        with pytest.raises(OLMoConfigurationError, match="supply dataset.mean_loss_weight"):
            alignment_recipe.build("perception", parent, include_means=False, overrides=v3)
        return
    for phase in ("perception", "joint"):
        config = alignment_recipe.build(phase, parent, include_means=False, overrides=v3)
        sources = dict(config.dataset.sources)
        targets = dict(config.dataset.target_loss_mass)
        if phase == "joint":
            assert sources.pop("native_text_replay") is not None
            assert targets.pop("native_text_replay") == 0.35
            assert sum(targets.values()) == pytest.approx(0.65)
            assert config.data_loader.group_sequence_quotas == {"text": 16, "vision": 112}
        assert set(sources) == set(STAGE1_V3_SOURCES)
        total = sum(STAGE1_V3_LOSS_TARGETS.values())
        for name, value in targets.items():
            assert value / sum(targets.values()) == pytest.approx(
                STAGE1_V3_LOSS_TARGETS[name] / total
            )
        means = {k: v for k, v in config.dataset.mean_loss_weight.items() if k in sources}
        assert means == STAGE1_V3_MEAN_LOSS_WEIGHTS
        sampling = _annotation_sampling(sources)
        assert set(sampling.values()) == {"one"}
        assert {"pixmo_points_v2", "pixmo_count_v2", "cosyn_point_v2", "plot_qa"} <= set(sampling)
        for name, source in sources.items():
            dataset = getattr(source, "dataset", source)
            assert dataset.message_format == "document", name
            assert dataset.loss_token_weighting == "none", name
            assert dataset.max_sequence_length == 8192, name
        caption = sources["pixmo_caption"].dataset
        assert caption.style_tag and caption.fixed_prompt is None and caption.mode == "caption"
        evaluator = config.trainer.callbacks["multimodal_evaluator"]
        assert {"pixmo_caption", "v3_long_caption", "v3_transcript"} <= set(
            evaluator.eval_dataset.sources
        )
        parent = alignment_recipe.save(config)
