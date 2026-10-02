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

from olmo_core.internal.vision_alignment import (
    MULTIMODAL_OVERRIDES,
    VisionAlignmentExperimentConfig,
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
