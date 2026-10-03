"""The pointing prompt form must follow the checkpoint's family, not the dataset.

Stage 1 trains with the released ``Molmo2-4B-Pretrain`` family (``prompt_templates="none"`` +
``system_prompt="style_and_length_v2"``): the question is the bare lowercased label behind a
``"<style>:"`` prefix. Stage 2 (``image_only_v9``) uses ``uber_model_v2`` templates with no
prefix. Training on one and evaluating on the other is out of distribution: a stage-1 4B
checkpoint trained on the stage-2 form scored 0.706 f1 on ``pixmo_point_eval_v3_mp`` against
0.815 for the released Pretrain checkpoint, losing most of it on abstention (zero-slice f1
0.718 vs 0.913) because the "Please say 'There are none.'" instruction exists only in the
stage-2 template.
"""

import numpy as np
import pytest

from olmo_core.data.multimodal.sft_formatter import SftFormatter

STAGE1 = {"prompt_templates": "none", "system_prompt": "style_and_length_v2"}


class _PromptTok:
    """Minimal tokenizer that records the user turns it templates."""

    eos_token_id = 1
    bos_token_id = 0

    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.prompts.append(messages[0]["content"])
        text = f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def encode(self, text, add_special_tokens=False):
        return [(ord(c) % 90) + 10 for c in text]


POINTS = [{"x": 10.5, "y": 20.5}, {"x": 30.0, "y": 40.0}]


def _turn(style: str, points, **family):
    fmt = SftFormatter(seed=0, **family)
    example = {"style": style, "label": "Leather Earmuff", "points": points, "point_scale": 100}
    return fmt.format_turns(example, index=3, rng=np.random.RandomState(0))[0]


@pytest.mark.parametrize("style", ["pointing", "point_count"])
def test_stage1_family_is_the_terse_prefixed_form(style: str):
    """mm_olmo ``data_formatter.py:1770-1779`` plus the ``"<style>:"`` prefix."""
    user, _ = _turn(style, POINTS, **STAGE1)
    assert user == f"{style}: leather earmuff"


@pytest.mark.parametrize("style", ["pointing", "point_count"])
def test_stage2_family_is_unchanged(style: str):
    """The default must stay the templated, unprefixed stage-2 form."""
    user, _ = _turn(style, POINTS)
    assert not user.startswith(f"{style}:")
    assert "leather earmuff" in user.lower()
    assert len(user) > len(f"{style}: leather earmuff")


# ---------------------------------------------------------------------------
# CoSyn pointing: its own `cosyn_point:` tag, on a question instead of a label
# ---------------------------------------------------------------------------


def _write_cosyn_point(tmp_path):
    """A one-row `cosyn-point` fixture: the columns `CoSynPointDataset` reads."""
    from datasets import Dataset, DatasetDict
    from PIL import Image

    img = tmp_path / "doc.png"
    Image.new("RGB", (64, 48), color=(10, 40, 90)).save(img)
    rows = {
        "image": [str(img)],
        "questions": [["Find the button that submits the form."]],
        "answer_points": [[{"x": [50.0], "y": [25.0]}]],
        "names": [["submit button"]],
    }
    path = tmp_path / "cosyn-point"
    DatasetDict({"train": Dataset.from_dict(rows)}).save_to_disk(str(path))
    return str(path)


def _cosyn_turn(tmp_path, **family):
    from olmo_core.data.multimodal import CoSynPointDatasetConfig

    tok = _PromptTok()
    path = _write_cosyn_point(tmp_path)
    CoSynPointDatasetConfig(dataset_path=path, max_crops=1, **family).build(tok)[0]
    return [p for p in tok.prompts if p]


def test_cosyn_point_carries_its_own_tag_in_stage1(tmp_path):
    """CoSyn pointing's question is an English request from the data, not an object name, so
    under the stage-1 family it gets mm_olmo's `cosyn_point:` tag, and `pointing:` is always
    followed by an object name."""
    from olmo_core.data.multimodal.pixmo_points import COSYN_POINT_STYLE

    assert COSYN_POINT_STYLE == "cosyn_point"
    assert _cosyn_turn(tmp_path, **STAGE1) == [
        "cosyn_point: Find the button that submits the form."
    ]


def _write_cosyn_point_v2(tmp_path):
    """A one-row audited CoSyn fixture (`cosyn-point-v2-masks` columns): three questions, of which
    the second failed the audit with `clear_error` and the third with `error`."""
    from datasets import Dataset, DatasetDict
    from PIL import Image

    img = tmp_path / "doc2.png"
    Image.new("RGB", (64, 48), color=(10, 40, 90)).save(img)
    rows = {
        "id": ["doc2"],
        "image": [str(img)],
        "questions": [["Find the title.", "Find the date.", "Find the signature."]],
        "answer_points": [[{"x": [50.0], "y": [25.0]}] * 3],
        "names": [["title", "date", "signature"]],
        "audit_result": [["correct", "clear_error", "error"]],
        "masks": [[["rle"], [], ["rle"]]],  # segmentation only: must be ignored
        "mask_f1": [[1.0, 0.0, 1.0]],
        "mask_size": [[48, 64]],
    }
    path = tmp_path / "cosyn-point-v2-masks"
    DatasetDict({"train": Dataset.from_dict(rows)}).save_to_disk(str(path))
    return str(path)


def test_audited_cosyn_tags_failed_questions_aux_cosyn_point(tmp_path):
    """With `audit_style`, the questions that failed the audit keep training behind their own tag;
    the masks never become messages."""
    from olmo_core.data.multimodal import CoSynPointDatasetConfig

    tok = _PromptTok()
    ex = CoSynPointDatasetConfig(
        dataset_path=_write_cosyn_point_v2(tmp_path),
        audit_style="aux_cosyn_point",
        max_crops=1,
        **STAGE1,
    ).build(tok)[0]
    assert ex["loss_masks"].sum() > 0
    assert sorted(p for p in tok.prompts if p) == [
        "aux_cosyn_point: Find the date.",
        "aux_cosyn_point: Find the signature.",
        "cosyn_point: Find the title.",
    ]


def test_audited_cosyn_without_audit_style_tags_every_question_alike(tmp_path):
    from olmo_core.data.multimodal import CoSynPointDatasetConfig

    tok = _PromptTok()
    CoSynPointDatasetConfig(
        dataset_path=_write_cosyn_point_v2(tmp_path), max_crops=1, **STAGE1
    ).build(tok)[0]
    assert {p.split(":")[0] for p in tok.prompts if p} == {"cosyn_point"}


def test_audit_style_needs_an_audited_build(tmp_path):
    from olmo_core.data.multimodal import CoSynPointDatasetConfig
    from olmo_core.exceptions import OLMoConfigurationError

    with pytest.raises(OLMoConfigurationError, match="audit_result"):
        CoSynPointDatasetConfig(
            dataset_path=_write_cosyn_point(tmp_path), audit_style="aux_cosyn_point"
        ).build(_PromptTok())


def test_stage1_v2_pointing_uses_the_audited_cosyn_build():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_s1a", "src/scripts/train/Molmo2-Stage1.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_s1a"] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    from olmo_core.data.multimodal.pixmo_points import COSYN_POINT_V2_PATH

    assert mod.POINTING_V2_COSYN_AUDIT_STYLE == "aux_cosyn_point"
    assert COSYN_POINT_V2_PATH.endswith("pixmo_datasets/cosyn-point-v2-masks")
    from types import SimpleNamespace

    config = SimpleNamespace(pointing_v2=None, count_v2=None, gui_syn=None, gui_pointing=False)
    cosyn = dict(mod._v2_pointing_sources(config))["cosyn_point_v2"]
    assert cosyn.dataset_path == COSYN_POINT_V2_PATH
    assert cosyn.audit_style == "aux_cosyn_point"


def test_target_never_carries_the_prefix():
    """Only the prompt is prefixed; the answer's label stays bare in both families.

    The 08/14 checkpoint echoed ``pointing:`` back inside ``<points>...</points>`` at eval,
    which is what a model does when the prefix is unfamiliar -- not something the targets ever
    contained.
    """
    for family in ({}, STAGE1):
        _, target = _turn("pointing", POINTS, **family)
        assert target == '<points coords="1 1 105 205 2 300 400">leather earmuff</points>'


def test_absent_label_abstains_in_both_families():
    """Zero points must yield the abstention string regardless of prompt family."""
    for family in ({}, STAGE1):
        _, target = _turn("pointing", [], **family)
        assert target == "There are none."


def test_stage1_weights_pointing_tokens_like_the_released_run():
    """Stage 1 must not leave the pointing datasets on `root_subsegments`.

    The dataset classes default to `root_subsegments`, which scales an example's loss by
    1/sqrt(n_labels). The released ``Molmo2-4B-Pretrain`` sets no weighting anywhere
    (``mm_preprocessor.loss_token_weighting: None``, ``message_weight: None`` on all four
    pointing entries), so every response token counts equally. Our pointing rows average ~12.5
    labels and counting ~2.8, so the default gave that data ~3.5x / ~1.7x less gradient weight
    per token than captions -- and the factor does not cancel out of the global
    ``sum(CE*w)/sum(w)`` divisor when branch counts differ across examples.
    """
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_s1", "src/scripts/train/Molmo2-Stage1.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_s1"] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    assert mod.POINTING_DATASET_KWARGS["loss_token_weighting"] == "none"
    assert mod.POINTING_DATASET_KWARGS["prompt_templates"] == "none"
    assert mod.POINTING_DATASET_KWARGS["system_prompt"] == "style_and_length_v2"


def test_root_subsegments_downweights_multi_label_examples():
    """Pin the mechanism: 1/sqrt(n) per token, so a 12-label example is weighted ~0.29."""
    import numpy as np

    from olmo_core.data.multimodal.message_weight import (
        MessageWeight,
        apply_message_weight_to_loss_masks,
    )

    for n, expected in ((1, 1.0), (4, 0.5), (12, 1 / 12**0.5)):
        sub = np.repeat(np.arange(1, n + 1), 10).astype(np.int64)
        masks = np.ones(10 * n, dtype=np.float32)
        out = apply_message_weight_to_loss_masks(
            masks, sub, MessageWeight(root_subsegments=True), branch_scaling_already_applied=False
        )
        assert abs(float(out[0]) - expected) < 1e-4, f"{n} labels -> {out[0]}"


def test_caption_message_weight_scales_loss_mass():
    """`message_weight` must scale caption loss mass, and default to released behaviour.

    Stage 1 sets captions to 1.25. The released ``Molmo2-4B-Pretrain`` leaves every source at 1.0
    (``message_weight: None`` on all six datasets), which puts captions at ~77.5% of the
    ``sum(CE*w)/sum(w)`` loss mass; 1.25 lifts that to ~81%. Measured over two-seed baselines:
    dense_caption avg 57.271 -> 57.808 (seed spread 0.085) for pointing costs of ~0.009-0.013 f1
    (seed spreads 0.008-0.011).
    """
    import numpy as np

    from olmo_core.data.multimodal.message_weight import (
        MessageWeight,
        apply_message_weight_to_loss_masks,
    )

    sub = np.repeat([1, 2], 20).astype(np.int64)
    base = apply_message_weight_to_loss_masks(
        np.ones(40, dtype=np.float32),
        sub,
        MessageWeight.from_string("none").with_overrides(None),
        branch_scaling_already_applied=True,
    )
    w125 = apply_message_weight_to_loss_masks(
        np.ones(40, dtype=np.float32),
        sub,
        MessageWeight.from_string("none").with_overrides(1.25),
        branch_scaling_already_applied=True,
    )
    assert abs(float(base[0]) - 1.0) < 1e-6
    assert abs(float(w125[0]) - 1.25) < 1e-6
    assert abs(float(w125.sum() / base.sum()) - 1.25) < 1e-6


def test_stage1_sets_caption_message_weight():
    """The stage-1 default must be 1.25, and it must be overridable rather than hardcoded."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_s1c", "src/scripts/train/Molmo2-Stage1.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_s1c"] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    assert mod.CAPTION_MESSAGE_WEIGHT == 1.25
    src = open("src/scripts/train/Molmo2-Stage1.py").read()
    # the value must reach the dataset config through the override-readable name, not a literal
    assert "message_weight=caption_message_weight" in src
    assert "caption_message_weight" in src


def test_read_override_accepts_dashed_names():
    """`_read_override` must normalize hyphens the way `Config.merge` does.

    `_clean_opt` turns `--caption-message-weight=1.0` into `caption_message_weight`, so the merger
    accepts the dashed spelling. A reader comparing the raw name would miss it and build the dataset
    config from the default while `merge` set the top-level field to the requested value -- a silent
    divergence, and it affected the pre-existing `model_size` / `train_vit` / `init_from` readers
    too.
    """
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_s1o", "src/scripts/train/Molmo2-Stage1.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_s1o"] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    for spelling in ("--caption_message_weight=1.0", "--caption-message-weight=1.0"):
        assert mod._read_float_override([spelling], "caption_message_weight", 1.25) == 1.0
    # unrelated keys still ignored
    assert mod._read_float_override(["--other=3"], "caption_message_weight", 1.25) == 1.25


def test_sft_demo_mode_threads_message_weight():
    """`message_weight` must reach the sft_demo path, which stage 2 uses.

    `PixMoCapDataset.__getitem__` returns early for `mode="sft_demo"` (built that way in
    mixtures/image_only_v9.py), so the branched path's scaling never runs and the option would be
    silently ignored for that mode.
    """
    src = open("src/olmo_core/data/multimodal/pixmo_cap.py").read()
    demo = src.split("def _getitem_sft_demo")[1].split("\ndef ")[0]
    assert "message_weight=" in demo, "sft_demo path does not pass message_weight"
    assert "self.config.message_weight" in demo
