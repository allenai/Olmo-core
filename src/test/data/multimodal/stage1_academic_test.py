"""The stage-1 academic QA group, its prompt family, and the TallyQA held-out filter."""

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

from olmo_core.data.multimodal import Stage1AcademicDatasetConfig
from olmo_core.data.multimodal.academic import registry
from olmo_core.data.multimodal.academic.registry import (
    ACADEMIC_REGISTRY,
    STAGE2_EVAL_TRAIN_SETS,
    AcademicSpec,
    build_academic_data,
)
from olmo_core.data.multimodal.mixtures import stage1_academic as acad_mix
from olmo_core.data.multimodal.pixmo_points_v2 import STAGE1_PROMPT_FAMILY
from olmo_core.data.multimodal.sft_formatter import SftFormatter
from olmo_core.exceptions import OLMoConfigurationError

STAGE1 = STAGE1_PROMPT_FAMILY
STAGE2 = {"prompt_templates": "uber_model_v2", "system_prompt": "demo_or_style_v2"}


class _FakeTok:
    """Minimal tokenizer for CPU tests (chat template + char-level encode)."""

    eos_token_id = 1
    bos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def encode(self, text, add_special_tokens=False):
        return [(ord(c) % 90) + 10 for c in text]


def _turns(example, family):
    return SftFormatter(seed=0, **family).format_turns(
        example, index=0, rng=np.random.RandomState(0)
    )


# ---------------------------------------------------------------------------
# The stage-1 prompt family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("style", ["dv_qa", "figure_qa", "plot_qa", "cosyn_document"])
def test_stage1_family_tags_the_bare_question(style):
    """Every style is tagged, as mm_olmo's molmo3 stage-1 family tags every non-caption style;
    before, a non-pointing style under the stage-1 family went in with no tag at all."""
    ex = {"image": "img.jpg", "message_list": [{"question": "Q1?", "answer": "A1", "style": style}]}
    assert _turns(ex, STAGE1) == [(f"{style}: Q1?", "A1")]
    # The stage-2 family tags these styles too (they are not demo styles), with the same string.
    assert _turns(ex, STAGE2) == [(f"{style}: Q1?", "A1")]


def test_stage1_family_keeps_the_pointing_form():
    ex = {
        "style": "pointing",
        "label": "Mug",
        "points": [{"x": 10.0, "y": 20.0}],
        "point_scale": 100,
    }
    assert _turns(ex, STAGE1)[0][0] == "pointing: mug"
    assert not _turns(ex, STAGE2)[0][0].startswith("pointing:")  # a demo style in stage 2


def test_stage1_family_clocks_keeps_its_fixed_prompt():
    ex = {"style": "clocks", "prompt": "What time is being shown?", "text": "3:02 PM"}
    assert _turns(ex, STAGE1) == [("clocks: What time is being shown?", "3:02 PM")]


def test_stage1_family_explanation_style_sends_the_bare_question():
    """mm_olmo applies the chain-of-thought instruction only in its templated families; under
    ``prompt_templates="none"`` the ``*_exp`` tag is the only thing asking for the explanation."""
    msg = {
        "question": "What is the value?",
        "explanation": "The bar shows 42.",
        "answer": "42",
        "style": "cosyn_chart_exp",
    }
    ex = {"image": "img.jpg", "message_list": [msg]}
    assert _turns(ex, STAGE1) == [
        ("cosyn_chart_exp: What is the value?", "The bar shows 42. Answer: 42")
    ]
    stage2 = _turns(ex, STAGE2)[0]
    assert "Provide reasoning steps" in stage2[0] and stage2[1] == "The bar shows 42. Answer: 42"


def test_stage1_family_refuses_multiple_choice():
    ex = {"question": "Which?", "options": ["a", "b"], "answer_idx": 0, "style": "science_qa"}
    with pytest.raises(NotImplementedError, match="multiple-choice"):
        _turns(ex, STAGE1)
    _turns(ex, STAGE2)  # stage 2 templates it as before


# ---------------------------------------------------------------------------
# TallyQA: no COCO val2017 image
# ---------------------------------------------------------------------------


@pytest.fixture
def _clear_heldout_caches():
    registry.coco_val2017_ids.cache_clear()
    registry.vg_coco_ids.cache_clear()
    build_academic_data.cache_clear()
    yield
    registry.coco_val2017_ids.cache_clear()
    registry.vg_coco_ids.cache_clear()
    build_academic_data.cache_clear()


def _write_tally_tree(tmp_path, monkeypatch):
    tally = tmp_path / "tally_qa"
    tally.mkdir()
    images = {
        "train2014/COCO_train2014_000000000001.jpg": "keep: a COCO train image",
        "val2014/COCO_val2014_000000000002.jpg": "drop: a val2017 image under its 2014 name",
        "val2014/COCO_val2014_000000000003.jpg": "keep: a val2014 image that is not in val2017",
        "VG_100K/10.jpg": "drop: VG 10 is COCO 2, a val2017 image",
        "VG_100K_2/11.jpg": "keep: VG 11 is not a COCO image",
        "VG_100K/12.jpg": "keep: VG 12 is COCO 5, not in val2017",
    }
    rows = [
        {"image": im, "image_id": i, "question": f"How many {i}?", "answer": i, "question_id": i}
        for i, im in enumerate(images)
    ]
    for split in ("train", "test"):
        (tally / f"{split}.json").write_text(json.dumps(rows))
    val2017 = tmp_path / "captions_val2017.json"
    val2017.write_text(json.dumps({"images": [{"id": 2}, {"id": 4}], "annotations": []}))
    vg = tmp_path / "image_data.json"
    vg.write_text(
        json.dumps(
            [
                {"image_id": 10, "coco_id": 2},
                {"image_id": 11, "coco_id": None},
                {"image_id": 12, "coco_id": 5},
            ]
        )
    )
    monkeypatch.setattr(registry, "TALLY_QA_SOURCE", str(tally))
    monkeypatch.setattr(registry, "COCO_VAL2017_ANNOTATIONS", str(val2017))
    monkeypatch.setattr(registry, "VG_IMAGE_DATA", str(vg))
    return images


def test_tally_qa_train_drops_coco_val2017_images(tmp_path, monkeypatch, _clear_heldout_caches):
    images = _write_tally_tree(tmp_path, monkeypatch)
    kept = {"/".join(r["image"].split("/")[-2:]) for r in registry._load_tally_qa("train")}
    assert kept == {im for im, why in images.items() if why.startswith("keep")}
    # Only the train split is filtered; the others are returned whole.
    assert len(registry._load_tally_qa("test")) == len(images)


@pytest.mark.parametrize("missing", ["COCO_VAL2017_ANNOTATIONS", "VG_IMAGE_DATA"])
def test_tally_qa_train_fails_without_its_heldout_sets(
    tmp_path, monkeypatch, _clear_heldout_caches, missing
):
    """A guard that cannot be checked fails the build rather than passing silently."""
    _write_tally_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(registry, missing, str(tmp_path / "missing.json"))
    with pytest.raises(OLMoConfigurationError, match="held-out set"):
        registry._load_tally_qa("train")


@pytest.mark.skipif(
    not (
        os.path.exists(os.path.join(registry.TALLY_QA_SOURCE, "train.json"))
        and os.path.exists(registry.COCO_VAL2017_ANNOTATIONS)
        and os.path.exists(registry.VG_IMAGE_DATA)
    ),
    reason="TallyQA / COCO / Visual Genome data not available",
)
def test_tally_qa_train_real_data(_clear_heldout_caches):
    """Measured on weka: 5,114 of the 132,981 train images are dropped (4,052 filed as COCO
    val2014 images and 1,062 Visual Genome images that are val2017 images). The other 28,581
    val2014 files are kept."""
    rows = registry._load_tally_qa("train")
    assert len(rows) == 127_867
    val2017 = registry.coco_val2017_ids()
    assert len(val2017) == 5_000
    n_val2014 = 0
    for r in rows:
        src, fname = r["image"].split("/")[-2:]
        assert registry._tally_coco_id(f"{src}/{fname}") not in val2017
        n_val2014 += src == "val2014"
    assert n_val2014 == 28_581


# ---------------------------------------------------------------------------
# Stage1AcademicDataset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(STAGE2_EVAL_TRAIN_SETS))
def test_stage1_dataset_refuses_stage2_eval_train_sets(name):
    """Refused before any data is read, so this needs no data on disk."""
    with pytest.raises(OLMoConfigurationError, match="stage-2 eval benchmark"):
        Stage1AcademicDatasetConfig(name=name).build(_FakeTok())


def test_stage1_dataset_rejects_unknown_names_and_bad_caps():
    with pytest.raises(OLMoConfigurationError, match="Unknown academic dataset"):
        Stage1AcademicDatasetConfig(name="not_a_dataset").build(_FakeTok())
    with pytest.raises(OLMoConfigurationError, match="max_questions"):
        Stage1AcademicDatasetConfig(name="dv_qa", max_questions=0).build(_FakeTok())


@pytest.fixture
def _many_question_source(monkeypatch):
    """A registry source with one 64x64 image carrying 30 PlotQA-style questions."""
    name = "fake_many_qa"
    rows = [
        {
            "image": np.full((64, 64, 3), 127, dtype=np.uint8),
            "questions": [f"Q{i}?" for i in range(30)],
            "answers": [f"A{i}" for i in range(30)],
        }
    ]
    monkeypatch.setitem(
        ACADEMIC_REGISTRY,
        name,
        AcademicSpec(name, lambda split: rows, registry._format_plot_qa_molmo),
    )
    build_academic_data.cache_clear()
    yield name
    build_academic_data.cache_clear()


def test_stage1_dataset_formats_with_the_stage1_family(_many_question_source):
    ds = Stage1AcademicDatasetConfig(name=_many_question_source).build(_FakeTok())
    _, branches, _ = ds.format_row(0, ds.epoch_rng(0))
    assert len(branches) == 30
    assert branches[0] == [("plot_qa: Q0?", "A0")]
    out = ds[0]
    assert out["input_ids"].shape == out["loss_masks"].shape


def test_default_max_questions_applies_without_the_recipe(_many_question_source, monkeypatch):
    """A config built directly, not through the recipe helper, still gets the source's cap, so a
    PlotQA-like source cannot overflow the sequence and be silently tail-truncated. An explicit
    value wins."""
    from olmo_core.data.multimodal import academic_dataset

    monkeypatch.setitem(academic_dataset.STAGE1_DEFAULT_MAX_QUESTIONS, _many_question_source, 7)
    ds = Stage1AcademicDatasetConfig(name=_many_question_source).build(_FakeTok())
    assert len(ds.format_row(0, ds.epoch_rng(0))[1]) == 7
    ds = Stage1AcademicDatasetConfig(name=_many_question_source, max_questions=12).build(_FakeTok())
    assert len(ds.format_row(0, ds.epoch_rng(0))[1]) == 12


def test_max_questions_rotates_across_epochs(_many_question_source):
    """The cap draws a subset per epoch, so over a run an image's other questions are reached;
    pinned to one draw, the rest would never be trained on."""
    ds = Stage1AcademicDatasetConfig(name=_many_question_source, max_questions=5).build(_FakeTok())
    seen = []
    for epoch in range(6):
        ds.set_epoch(epoch)
        _, branches, _ = ds.format_row(0, ds.epoch_rng(0))
        assert len(branches) == 5
        seen.append(tuple(b[0][0] for b in branches))
    assert len(set(seen)) > 1, f"questions never rotated across epochs: {seen}"
    assert len({q for s in seen for q in s}) > 5
    # Deterministic within an epoch, and identical for a resumed run that replays it.
    ds.set_epoch(3)
    a = ds[0]
    ds.set_epoch(0)
    ds.set_epoch(3)
    np.testing.assert_array_equal(a["input_ids"], ds[0]["input_ids"])


# ---------------------------------------------------------------------------
# The stage-1 group
# ---------------------------------------------------------------------------


def test_stage2_eval_train_sets_are_pinned():
    """The benchmarks the stage-2 checkpoints are evaluated on (mm_olmo ``eval_molmo2.py``), plus
    TallyQA for its shared images. Changing this list changes what stage 1 may train on."""
    assert set(STAGE2_EVAL_TRAIN_SETS) == {
        "coco_2014_vqa_multi",
        "text_vqa",
        "chart_qa_weighted",
        "doc_qa",
        "info_qa",
        "ai2_diagram_v2_mix_transparent",
        "a_okvqa_mc",
        "a_okvqa_da",
        "tally_qa",
    }
    assert set(STAGE2_EVAL_TRAIN_SETS) <= set(ACADEMIC_REGISTRY)


def test_no_stage1_source_is_a_stage2_eval_train_set():
    names = set(acad_mix.STAGE1_ACADEMIC_SOURCE_NAMES)
    assert not names & set(STAGE2_EVAL_TRAIN_SETS)
    assert names <= set(ACADEMIC_REGISTRY)
    with pytest.raises(OLMoConfigurationError, match="stage-2 eval benchmark"):
        acad_mix.build_stage1_academic_source("tally_qa", _FakeTok())
    with pytest.raises(OLMoConfigurationError, match="Unknown stage-1 academic source"):
        acad_mix.build_stage1_academic_source("okvqa", _FakeTok())


def test_default_sources():
    assert acad_mix.DEFAULT_ACADEMIC_SOURCES == (
        "cosyn_chart_exp",
        "cosyn_chemical_exp",
        "cosyn_diagram_exp",
        "cosyn_document",
        "cosyn_math_exp",
        "cosyn_music_exp",
        "cosyn_table_exp",
        "dv_qa",
        "figure_qa",
        "plot_qa",
    )
    from olmo_core.data.multimodal.academic_dataset import STAGE1_DEFAULT_MAX_QUESTIONS

    assert STAGE1_DEFAULT_MAX_QUESTIONS == {"plot_qa": 20}


# ---------------------------------------------------------------------------
# PixMo-Clocks: a group of its own
# ---------------------------------------------------------------------------


@pytest.fixture
def _fake_clocks(monkeypatch):
    """The clocks registry entry, with one in-memory row in the shape the real formatter returns."""

    def formatter(row, rng, split):
        return {
            "image": np.full((64, 64, 3), 255, dtype=np.uint8),
            "prompt": "What time is being shown?",
            "text": row["text"],
            "style": "clocks",
            "metadata": {},
        }

    rows = [{"text": "The time shown is 3:02 PM"}]
    monkeypatch.setitem(
        ACADEMIC_REGISTRY,
        acad_mix.CLOCKS_SOURCE,
        AcademicSpec(acad_mix.CLOCKS_SOURCE, lambda split: rows, formatter),
    )
    build_academic_data.cache_clear()
    yield
    build_academic_data.cache_clear()


def test_clocks_group_asks_the_question(_fake_clocks):
    """The user turn keeps the question behind the tag, as for the academic sources: stage 1 may be
    the only place the model learns to read clocks, so the question text is trained too."""
    ds = acad_mix.build_stage1_clocks_source(_FakeTok())
    _, branches, _ = ds.format_row(0, ds.epoch_rng(0))
    assert branches == [[("clocks: What time is being shown?", "The time shown is 3:02 PM")]]
    out = ds[0]
    assert out["input_ids"].shape == out["loss_masks"].shape


def test_clocks_is_not_an_academic_source():
    assert acad_mix.CLOCKS_SOURCE not in acad_mix.STAGE1_ACADEMIC_SOURCE_NAMES
    with pytest.raises(OLMoConfigurationError, match="own group"):
        acad_mix.build_stage1_academic_source("pixmo_clocks", _FakeTok())
    with pytest.raises(OLMoConfigurationError, match="own group"):
        acad_mix.academic_weighting_sizes(["pixmo_clocks"], [1])


@pytest.mark.skipif(
    not os.path.exists(os.path.join(registry.PIXMO_DATASETS, "clocks", "train.jsonl")),
    reason="PixMo-Clocks data not available",
)
def test_clocks_group_real_data():
    build_academic_data.cache_clear()
    ds = acad_mix.build_stage1_clocks_source(_FakeTok())
    assert len(ds) == 800_269
    _, branches, _ = ds.format_row(0, ds.epoch_rng(0))
    (((user, answer),),) = branches
    assert user == "clocks: What time is being shown?"
    assert answer.startswith("The time")


#: Training rows (one per image) measured on weka.
REAL_SIZES = {
    "cosyn_chart_exp": 116_814,
    "cosyn_chemical_exp": 8_942,
    "cosyn_diagram_exp": 34_963,
    "cosyn_document": 71_282,
    "cosyn_math_exp": 66_714,
    "cosyn_music_exp": 11_969,
    "cosyn_table_exp": 46_518,
    "dv_qa": 200_000,
    "figure_qa": 100_000,
    "plot_qa": 157_070,
}
TEMPLATED = ("dv_qa", "figure_qa", "plot_qa")


def test_academic_weighting_sizes_cap_the_templated_charts():
    names = list(REAL_SIZES)
    assert acad_mix.academic_weighting_sizes(names, [REAL_SIZES[n] for n in names]) == [
        116_814,
        8_942,
        34_963,
        71_282,
        66_714,
        11_969,
        46_518,
        10_000,
        10_000,
        20_000,
    ]
    with pytest.raises(OLMoConfigurationError, match="stage-2 eval benchmark"):
        acad_mix.academic_weighting_sizes(["tally_qa"], [1])


# ---------------------------------------------------------------------------
# Molmo2-Stage1.py
# ---------------------------------------------------------------------------


def _load_stage1_module():
    try:
        import olmo_core.internal.common  # noqa: F401  (needs a recent beaker-py)
    except ImportError as e:  # pragma: no cover - env-dependent
        pytest.skip(f"Molmo2-Stage1.py imports fail here: {e}")
    spec = importlib.util.spec_from_file_location(
        "_stage1_academic", "src/scripts/train/Molmo2-Stage1.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_academic"] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


def _data_config(**kw):
    from types import SimpleNamespace

    from olmo_core.data.multimodal import (
        OcrCaptionTarsDatasetConfig,
        OlmOcrMixDatasetConfig,
        TextRichCaptionDatasetConfig,
    )
    from olmo_core.data.multimodal.mixtures.ocr import DEFAULT_OCR_SOURCES

    fields = dict(
        recipe="v2",
        pointing_data="v2",
        pointing_rate=0.25,
        nlp_rate=0.0,
        ocr_rate=0.25,
        ocr_sources=DEFAULT_OCR_SOURCES,
        olmocr=OlmOcrMixDatasetConfig(),
        ocr_tars=OcrCaptionTarsDatasetConfig(),
        text_rich=TextRichCaptionDatasetConfig(),
        academic_rate=0.1,
        academic_sources=acad_mix.DEFAULT_ACADEMIC_SOURCES,
        clock_rate=0.0,
    )
    fields.update(kw)
    return SimpleNamespace(**fields)


def test_stage1_academic_group_wiring():
    mod = _load_stage1_module()
    assert mod.ACADEMIC_RATE == 0.0  # off by default: an ablation arm
    assert mod.ACADEMIC_SOURCES == acad_mix.DEFAULT_ACADEMIC_SOURCES
    fields = set(mod.ExperimentConfig.__dataclass_fields__)
    assert {"academic_rate", "academic_sources", "clock_rate"} <= fields
    assert mod.CLOCK_RATE == 0.0
    for r in ("v1", "v2"):
        assert mod.RECIPES[r]["academic_rate"] == mod.RECIPES[r]["clock_rate"] == 0.0
    mod.validate_data_config(_data_config())


def test_v3_recipe():
    """v3 is v2 plus the academic QA group and the clock group, with more OCR, for 50k steps:
    caption 0.33, pointing 0.16, OCR 0.34, academic 0.14, clocks 0.03, on the v2 pointing sources
    and no text-only data."""
    mod = _load_stage1_module()
    v3 = mod.RECIPES["v3"]
    assert v3 == dict(
        pointing_rate=0.16,
        nlp_rate=0.0,
        ocr_rate=0.34,
        academic_rate=0.14,
        clock_rate=0.03,
        pointing_data="v2",
    )
    groups = ("pointing_rate", "nlp_rate", "ocr_rate", "academic_rate", "clock_rate")
    assert 1.0 - sum(v3[k] for k in groups) == pytest.approx(0.33)
    assert mod.resolve_recipe(["--recipe=v3"]) == ("v3", v3)
    # v3's rates are set for 50k steps; the other recipes keep the script's 32k.
    assert mod.RECIPE_MAX_STEPS == {"v3": 50_000}
    assert mod.MAX_STEPS == 32_000
    mod.validate_data_config(_data_config(recipe="v3", **v3))
    # With the real sizes, the ten default sources share exactly the recipe's academic rate.
    names = list(acad_mix.DEFAULT_ACADEMIC_SOURCES)
    frac = mod._academic_fractions(names, [REAL_SIZES[n] for n in names])
    assert v3["academic_rate"] * frac.sum() == pytest.approx(0.14)


@pytest.mark.parametrize("rate", [0.0, 0.1])
def test_stage1_refuses_stage2_eval_train_sets(rate):
    """Refused whatever the rate, so a refused set cannot sit in a saved config."""
    mod = _load_stage1_module()
    for name in ("tally_qa", "coco_2014_vqa_multi", "doc_qa"):
        with pytest.raises(OLMoConfigurationError, match="stage-2 eval benchmarks"):
            mod.validate_data_config(
                _data_config(academic_rate=rate, academic_sources=("dv_qa", name))
            )


def test_stage1_academic_validation():
    mod = _load_stage1_module()
    with pytest.raises(OLMoConfigurationError, match="Unknown academic_sources"):
        mod.validate_data_config(_data_config(academic_sources=("okvqa",)))
    with pytest.raises(OLMoConfigurationError, match="duplicates"):
        mod.validate_data_config(_data_config(academic_sources=("dv_qa", "dv_qa")))
    with pytest.raises(OLMoConfigurationError, match="at least one entry"):
        mod.validate_data_config(_data_config(academic_sources=()))
    with pytest.raises(OLMoConfigurationError, match="exceeds 1"):
        mod.validate_data_config(_data_config(academic_rate=0.6))
    with pytest.raises(OLMoConfigurationError, match=">= 0"):
        mod.validate_data_config(_data_config(academic_rate=-0.1))
    # PixMo-Clocks is its own group, not an academic source.
    with pytest.raises(OLMoConfigurationError, match="own group"):
        mod.validate_data_config(_data_config(academic_sources=("dv_qa", "pixmo_clocks")))
    # The clock rate counts toward the total and must be >= 0.
    with pytest.raises(OLMoConfigurationError, match="exceeds 1"):
        mod.validate_data_config(_data_config(clock_rate=0.5))
    with pytest.raises(OLMoConfigurationError, match=">= 0"):
        mod.validate_data_config(_data_config(clock_rate=-0.01))
    mod.validate_data_config(_data_config(clock_rate=0.01))


def test_stage1_academic_fractions():
    """The default split with the real sizes: CoSyn 81.2%, the templated chart sets 18.8% (44.0%
    without their row caps)."""
    mod = _load_stage1_module()
    names = list(acad_mix.DEFAULT_ACADEMIC_SOURCES)
    frac = mod._academic_fractions(names, [REAL_SIZES[n] for n in names])
    share = dict(zip(names, frac))
    np.testing.assert_allclose(frac.sum(), 1.0)
    np.testing.assert_allclose(sum(share[n] for n in TEMPLATED), 0.188, atol=5e-4)
    np.testing.assert_allclose(
        sum(f for n, f in share.items() if n.startswith("cosyn_")), 0.812, atol=5e-4
    )
    np.testing.assert_allclose(
        share["cosyn_chart_exp"] / share["dv_qa"], np.sqrt(116_814 / 10_000), rtol=1e-9
    )
    uncapped = np.sqrt([REAL_SIZES[n] for n in names])
    share_uncapped = dict(zip(names, uncapped / uncapped.sum()))
    np.testing.assert_allclose(sum(share_uncapped[n] for n in TEMPLATED), 0.440, atol=5e-4)
    with pytest.raises(OLMoConfigurationError, match="dv_qa"):
        mod._academic_fractions(["dv_qa"], [0])  # an empty source is named
