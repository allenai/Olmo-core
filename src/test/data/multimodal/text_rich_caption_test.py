"""CPU tests for the three-level figure-caption source (mm_olmo ``TextRichCaptionConfig`` port):
one branch per caption level, mm_olmo's style names, a tag-only user turn, train-split-only
loading, config validation, truncation, and the OCR-registry / Molmo2-Stage1 wiring."""

import importlib.util
import sys

import numpy as np
import pytest
from datasets import Dataset, DatasetDict

from olmo_core.data.multimodal import (
    OcrCaptionTarsDatasetConfig,
    OlmOcrMixDatasetConfig,
    TextRichCaptionDatasetConfig,
)
from olmo_core.data.multimodal.mixtures import ocr as ocr_mix
from olmo_core.data.multimodal.text_rich_caption import (
    CAPTION_LEVELS,
    CATEGORIES,
    level_style,
)
from olmo_core.exceptions import OLMoConfigurationError

# Three captions of one row, in the proportions of the real corpus (a chart row ships roughly
# 150 / 650 / 2300 characters).
HIGH = "An area chart of monthly sign-ups for seven activities."
MID = "A centered, filled-area chart sits inside a rounded white card on a pale blue page. " * 4
LOW = "At the top of the page a large centered title reads Seasonal Distribution. " * 12


class _FakeTok:
    """Minimal tokenizer for CPU tests: records the user turns it templates."""

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


# ---------------------------------------------------------------------------
# Fixture: a tiny on-disk copy of the corpus layout
# ---------------------------------------------------------------------------


def _write_corpus(tmp_path, category="chart", rows=None, columns=None, validation=None):
    """A corpus root: ``hf/<category>`` splits plus the images they point at.

    ``columns`` restricts which caption columns are written; ``validation`` adds a held-out
    split with those row ids.
    """
    from PIL import Image

    if rows is None:
        rows = [
            {"id": "ex0", "high_level": HIGH, "mid_level": MID, "low_level": LOW},
            {"id": "ex1", "high_level": HIGH, "mid_level": MID, "low_level": LOW},
        ]

    def records(items):
        out = []
        for row in items:
            relpath = f"{category}/{row['id']}/{row['id']}.png"
            img = tmp_path / relpath
            img.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 48), color=(200, 30, 30)).save(img)
            rec = {"id": row["id"], "category": category, "image_relpath": relpath}
            for lvl in CAPTION_LEVELS:
                if (columns is None or lvl in columns) and lvl in row:
                    rec[lvl] = row[lvl]
            out.append(rec)
        return out

    splits = {"train": Dataset.from_list(records(rows))}
    if validation:
        held = [
            {"id": i, "high_level": HIGH, "mid_level": MID, "low_level": LOW} for i in validation
        ]
        splits["validation"] = Dataset.from_list(records(held))
    DatasetDict(splits).save_to_disk(str(tmp_path / "hf" / category))
    return str(tmp_path)


def _cfg(root, **kw):
    kw.setdefault("category", "chart")
    kw.setdefault("max_crops", 1)
    return TextRichCaptionDatasetConfig(dataset_path=root, **kw)


# ---------------------------------------------------------------------------
# Three levels, three branches
# ---------------------------------------------------------------------------


def test_level_styles_are_mm_olmo_names():
    assert CAPTION_LEVELS == ("high_level", "mid_level", "low_level")
    assert [level_style(lvl) for lvl in CAPTION_LEVELS] == [
        "fig_caption_high",
        "fig_caption_mid",
        "fig_caption_low",
    ]


def test_emits_one_branch_per_level_in_order(tmp_path):
    ds = _cfg(_write_corpus(tmp_path)).build(_FakeTok())
    turns = ds.turns(ds._data[0])
    assert [t[1] for t in turns] == [HIGH, MID, LOW]
    assert [t[0] for t in turns] == [
        "fig_caption_high:",
        "fig_caption_mid:",
        "fig_caption_low:",
    ]


def test_all_three_captions_are_supervised(tmp_path):
    """The summary and the dense read-out are most of the text; a mid-level-only source would
    carry less than half of it."""
    ds = _cfg(_write_corpus(tmp_path)).build(_FakeTok())
    total = sum(len(t[1]) for t in ds.turns(ds._data[0]))
    assert total == len(HIGH) + len(MID) + len(LOW)
    assert total > 2 * len(MID)


def test_user_turn_is_the_tag_with_no_question(tmp_path):
    """General OCR data: the prompt names the task and asks nothing."""
    tok = _FakeTok()
    ds = _cfg(_write_corpus(tmp_path)).build(tok)
    _ = ds[0]
    asked = sorted(p for p in tok.prompts if p)
    assert asked == sorted(f"{level_style(lvl)}:" for lvl in CAPTION_LEVELS)


def test_levels_can_be_narrowed(tmp_path):
    ds = _cfg(_write_corpus(tmp_path), levels=("low_level",)).build(_FakeTok())
    assert [t[1] for t in ds.turns(ds._data[0])] == [LOW]


def test_blank_level_is_dropped(tmp_path):
    rows = [{"id": "ex0", "high_level": "", "mid_level": MID, "low_level": LOW}]
    ds = _cfg(_write_corpus(tmp_path, rows=rows)).build(_FakeTok())
    assert [t[1] for t in ds.turns(ds._data[0])] == [MID, LOW]


def test_row_with_no_caption_is_skipped_not_raised(tmp_path):
    rows = [
        {"id": "ex0", "high_level": "", "mid_level": "  ", "low_level": ""},
        {"id": "ex1", "high_level": HIGH, "mid_level": MID, "low_level": LOW},
    ]
    ds = _cfg(_write_corpus(tmp_path, rows=rows)).build(_FakeTok())
    with pytest.raises(ValueError, match="no non-empty caption"):
        ds.turns(ds._data[0])
    # ...but fetching it substitutes the next usable row instead of spending the error budget.
    np.testing.assert_array_equal(ds[0]["input_ids"], ds[1]["input_ids"])


# ---------------------------------------------------------------------------
# Train split only
# ---------------------------------------------------------------------------


def test_only_the_train_split_is_read(tmp_path):
    root = _write_corpus(tmp_path, validation=["held0", "held1", "held2"])
    ds = _cfg(root).build(_FakeTok())
    assert len(ds) == 2
    assert {ds._data[i]["id"] for i in range(len(ds))} == {"ex0", "ex1"}
    assert "split" not in TextRichCaptionDatasetConfig.__dataclass_fields__


def test_build_without_a_train_split_fails(tmp_path):
    from PIL import Image

    (tmp_path / "chart" / "v0").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(tmp_path / "chart" / "v0" / "v0.png")
    rec = {"id": "v0", "image_relpath": "chart/v0/v0.png"}
    rec.update({lvl: "text" for lvl in CAPTION_LEVELS})
    DatasetDict({"validation": Dataset.from_list([rec])}).save_to_disk(
        str(tmp_path / "hf" / "chart")
    )
    with pytest.raises(OLMoConfigurationError, match="no 'train' split"):
        _cfg(str(tmp_path)).build(_FakeTok())


# ---------------------------------------------------------------------------
# Paths, validation, loading
# ---------------------------------------------------------------------------


def test_paths_resolve_under_the_root(tmp_path):
    root = _write_corpus(tmp_path)
    cfg = _cfg(root)
    assert cfg.hf_path == f"{root}/hf/chart"
    ds = cfg.build(_FakeTok())
    assert ds.image_path(ds._data[0]) == f"{root}/chart/ex0/ex0.png"


@pytest.mark.parametrize("category", CATEGORIES)
def test_every_category_builds(tmp_path, category):
    root = _write_corpus(tmp_path / category, category=category)
    assert len(_cfg(root, category=category).build(_FakeTok())) == 2


@pytest.mark.parametrize(
    "kw",
    [
        {"category": "receipts"},
        {"levels": ()},
        {"levels": ("medium_level",)},
        {"levels": ("mid_level", "mid_level")},
    ],
)
def test_config_validation_rejects_bad_values(kw):
    with pytest.raises(OLMoConfigurationError):
        TextRichCaptionDatasetConfig(**kw).validate()


def test_missing_level_column_is_named(tmp_path):
    root = _write_corpus(tmp_path, columns=("high_level", "mid_level"))
    with pytest.raises(OLMoConfigurationError, match="low_level"):
        _cfg(root).build(_FakeTok())
    assert len(_cfg(root, levels=("high_level", "mid_level")).build(_FakeTok())) == 2


# ---------------------------------------------------------------------------
# Built examples
# ---------------------------------------------------------------------------


def test_example_has_three_branches_that_all_carry_loss(tmp_path):
    seq = _cfg(_write_corpus(tmp_path)).build(_FakeTok())[0]
    assert seq["loss_masks"].shape == seq["input_ids"].shape
    branch_ids = sorted(set(seq["subsegment_ids"].tolist()))
    assert len(branch_ids) == 4  # shared image prefix + three branches
    for b in branch_ids[:3]:
        assert (seq["loss_masks"][seq["subsegment_ids"] == b] > 0).any(), b


def test_example_is_deterministic_within_an_epoch(tmp_path):
    ds = _cfg(_write_corpus(tmp_path)).build(_FakeTok())
    np.testing.assert_array_equal(ds[0]["input_ids"], ds[0]["input_ids"])


def test_max_sequence_length_truncates(tmp_path):
    root = _write_corpus(tmp_path)
    full = _cfg(root).build(_FakeTok())[0]
    cut = len(full["input_ids"]) - 32
    short = _cfg(root, max_sequence_length=cut).build(_FakeTok())[0]
    assert len(short["input_ids"]) == cut
    assert (short["loss_masks"] > 0).any()


# ---------------------------------------------------------------------------
# OCR registry + Molmo2-Stage1 wiring
# ---------------------------------------------------------------------------


def test_build_ocr_source_routes_by_category(tmp_path):
    root = _write_corpus(tmp_path / "doc", category="doc")
    ds = ocr_mix.build_ocr_source(
        "text_rich_doc",
        _FakeTok(),
        olmocr=OlmOcrMixDatasetConfig(),
        tars=OcrCaptionTarsDatasetConfig(dataset_path="unused"),
        text_rich=_cfg(root),
    )
    assert ds.config.category == "doc"
    assert len(ds) == 2


def _load_stage1_module():
    try:
        import olmo_core.train  # noqa: F401
    except Exception as e:  # pragma: no cover - depends on the install
        pytest.skip(f"Molmo2-Stage1.py imports fail here: {e}")
    spec = importlib.util.spec_from_file_location(
        "_stage1_text_rich", "src/scripts/train/Molmo2-Stage1.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_text_rich"] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


def test_stage1_carries_a_figure_caption_template():
    mod = _load_stage1_module()
    assert "text_rich" in mod.ExperimentConfig.__dataclass_fields__
    assert set(ocr_mix.TEXT_RICH_SOURCES) <= set(mod.OCR_SOURCES)
    assert mod.OCR_RATE == 0.0  # the OCR group stays opt-in
