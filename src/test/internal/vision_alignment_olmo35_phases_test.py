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


@pytest.fixture
def stage1(alignment_recipe, hero_checkpoint, tmp_path):
    """Build the stage1 phase on the OLMo 3.5 checkpoint with the v3 data."""
    _write_caption_manifest(tmp_path / "artifacts")

    def build(overrides=()):
        return alignment_recipe.build(
            "stage1", include_means=False, overrides=["--recipe.data=stage1_v3", *overrides]
        )

    return build


def _covered(key: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(key, pattern) or key.startswith(pattern + ".")


def test_all_hero_phases_differ_from_the_text_config_only_by_the_override_table(
    alignment_recipe, hero_checkpoint, text_config, stage1
):
    text = _round_trip(text_config)
    configs = _hero_phases(alignment_recipe)
    configs["stage1"] = stage1([f"--recipe.text_config={FIXTURE}"]).as_config_dict()
    differing_by_phase = {phase: _differing_keys(text, config) for phase, config in configs.items()}
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


def _group_lrs(config) -> dict[str, float]:
    optim = config.train_module.optim
    lrs = {o.opts["scheduler_name"]: o.opts["lr"] for o in optim.group_overrides}
    return {"connector": lrs["connector"], "vision": lrs["vision"], "lm": optim.lr}


def test_stage1_defaults(stage1, alignment_recipe):
    from olmo_core.internal.vision_alignment_data import (
        STAGE1_V3_LOSS_TARGETS,
        STAGE1_V3_SOURCES,
    )
    from olmo_core.train import LoadStrategy

    config = stage1()
    # Started from the text LM like bridge, never from a parent phase.
    assert config.recipe.parent_checkpoint is None and config.trainer.load_path is None
    assert config.trainer.load_strategy == LoadStrategy.if_available
    assert "initialize_multimodal" in config.trainer.callbacks
    # Molmo2-Stage1's learning rates and warmups over the token-matched step budget.
    assert config.trainer.max_duration.value == 15_625
    scheduler = config.train_module.scheduler
    schedules = {"lm": scheduler.default, **scheduler.schedulers}
    assert {name: (s.warmup, s.t_max, s.alpha_f) for name, s in schedules.items()} == {
        "connector": (200, 15_625, 0.1),
        "vision": (2000, 15_625, 0.1),
        "lm": (2000, 15_625, 0.1),
    }
    assert _group_lrs(config) == {"connector": 2e-4, "vision": 6e-6, "lm": 2e-5}
    # Every component trains; only the image-token rows of the embeddings.
    assert config.train_module.freeze_params == []
    assert config.train_module.train_embedding_rows == vision_alignment._image_token_rows(
        alignment_recipe.token_ids
    )
    assert config.recipe.restore_pretraining_router_lb
    assert config.model.lm.block.routed_experts_router.lb_loss_weight == 0.01
    # Visual data only, at the v3 run's full loss shares.
    assert set(config.dataset.sources) == set(STAGE1_V3_SOURCES)
    total = sum(STAGE1_V3_LOSS_TARGETS.values())
    assert config.train_module.source_loss_mass_targets == pytest.approx(
        {name: value / total for name, value in STAGE1_V3_LOSS_TARGETS.items()}
    )
    assert config.data_loader.group_sequence_quotas is None
    assert config.data_loader.source_groups is None
    assert config.train_module.loss_group_weights is None
    loader, module = config.data_loader, config.train_module
    assert loader.global_batch_size == 128 * 8192
    assert module.rank_microbatch_size == 4 * 8192
    # A permanent checkpoint at each quarter, none pruned.
    checkpointer = config.trainer.callbacks["checkpointer"]
    assert checkpointer.fixed_steps == [3906, 7812, 11719, 15625]
    assert checkpointer.max_checkpoints == 4
    assert (checkpointer.save_interval, checkpointer.ephemeral_save_interval) == (15_625, 250)
    restored = VisionAlignmentExperimentConfig.from_dict(
        json.loads(json.dumps(config.as_config_dict()))
    )
    assert restored == config


def test_stage1_steps_set_the_horizons_and_checkpoint_quarters(stage1):
    config = stage1(["--recipe.steps=4000"])
    scheduler = config.train_module.scheduler
    assert config.trainer.max_duration.value == 4000
    assert {s.t_max for s in (scheduler.default, *scheduler.schedulers.values())} == {4000}
    assert config.trainer.callbacks["checkpointer"].fixed_steps == [1000, 2000, 3000, 4000]


@pytest.mark.parametrize(
    "overrides,lrs,frozen,lb_loss_weight",
    [
        (["--recipe.lm_lr=1e-6"], {"connector": 2e-4, "vision": 6e-6, "lm": 1e-6}, [], 0.01),
        (
            ["--recipe.lm_lr=0"],
            {"connector": 2e-4, "vision": 6e-6, "lm": 2e-4},
            ["lm.embedding_norm.*", "lm.blocks.*", "lm.lm_head.*"],
            0.0,
        ),
        (
            ["--recipe.vision_lr=0", "--recipe.connector_lr=1e-4"],
            {"connector": 1e-4, "vision": 0.0, "lm": 2e-5},
            ["vision.*"],
            0.01,
        ),
    ],
)
def test_stage1_learning_rate_ablations(stage1, overrides, lrs, frozen, lb_loss_weight):
    config = stage1(overrides)
    assert _group_lrs(config) == lrs
    assert config.train_module.freeze_params == frozen
    assert config.model.lm.block.routed_experts_router.lb_loss_weight == lb_loss_weight
    assert config.recipe.restore_pretraining_router_lb == (lb_loss_weight > 0)


@pytest.mark.parametrize(
    "overrides,match",
    [
        (["--recipe.lm_lr=-1e-5"], "finite and nonnegative"),
        (["--recipe.connector_lr=0"], "connector_lr must be positive"),
    ],
)
def test_stage1_rejects_invalid_learning_rates(stage1, overrides, match):
    from olmo_core.exceptions import OLMoConfigurationError

    with pytest.raises(OLMoConfigurationError, match=match):
        stage1(overrides)


def test_stage1_guards(alignment_recipe, stage1):
    from olmo_core.exceptions import OLMoConfigurationError

    stage1()  # writes the caption manifest
    with pytest.raises(OLMoConfigurationError, match="trains on recipe.data=stage1_v3"):
        alignment_recipe.build("stage1", include_means=False)
    parent = alignment_recipe.save(alignment_recipe.build())
    with pytest.raises(OLMoConfigurationError, match="Stage 1 starts from the text LM"):
        alignment_recipe.build(
            "stage1", parent, include_means=False, overrides=["--recipe.data=stage1_v3"]
        )
    with pytest.raises(OLMoConfigurationError, match="apply to the stage1 phase only"):
        alignment_recipe.build(overrides=["--recipe.lm_lr=1e-6"])
