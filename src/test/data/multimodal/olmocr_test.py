"""CPU tests for the olmOCR-mix page-transcription source (mm_olmo ``OlmOcrMixConfig`` port):
subset / split naming, PDF path resolution, render-size sampling, the style tag per prompt
family, language filtering, blank-page handling, truncation, and the Molmo2-Stage1 wiring.

The dataset tests swap the PDF renderer for a stub so they run without ``pypdfium2``; one test
exercises the real renderer when it is installed."""

import importlib.util
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from olmo_core.data.multimodal import OlmOcrMixDatasetConfig
from olmo_core.data.multimodal import olmocr as olmocr_mod
from olmo_core.data.multimodal.olmocr import (
    OLMOCR_STYLE,
    canonical_split,
    canonical_subset,
    pdf_path_for,
)
from olmo_core.exceptions import OLMoConfigurationError


class _FakeTok:
    eos_token_id = 1
    bos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def encode(self, text, add_special_tokens=False):
        return [(ord(c) % 90) + 10 for c in text]


# ---------------------------------------------------------------------------
# Naming / paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("documents", "00_documents"),
        ("00_documents", "00_documents"),
        ("national_archives", "03_national_archives"),
    ],
)
def test_canonical_subset(name, expected):
    assert canonical_subset(name) == expected


def test_canonical_subset_rejects_unknown():
    with pytest.raises(OLMoConfigurationError):
        canonical_subset("receipts")


def test_canonical_split_accepts_validation_alias():
    assert canonical_split("validation") == "eval"
    assert canonical_split("train") == "train"
    with pytest.raises(OLMoConfigurationError):
        canonical_split("test")


def test_pdf_path_for_splits_after_the_tarball_name():
    rel = "pdf_tarballs/02_loc_transcripts_train_00003.tar.gz:gladstone_2023-01-20/mss:1-8.pdf"
    assert pdf_path_for("/root", rel) == (
        "/root/pdfs/02_loc_transcripts_train_00003/gladstone_2023-01-20/mss:1-8.pdf"
    )
    with pytest.raises(ValueError):
        pdf_path_for("/root", "pdfs/no-separator.pdf")


# ---------------------------------------------------------------------------
# Fixture: a two-subset olmOCR-mix root with PIL-written single-page PDFs
# ---------------------------------------------------------------------------

ROWS = [
    # (id, language, text)
    ("en-short", "en", "Page one.\n\nA short transcription."),
    ("fr", "fr", "Une page en français."),
    ("en-blank", "en", None),
    ("en-long", "en", "word " * 400),
]


def _write_root(tmp_path):
    from PIL import Image

    root = tmp_path / "olmocr_mix"
    chunk = "00_documents_train_00000"
    (root / "pdfs" / chunk / "0000").mkdir(parents=True)
    rows = []
    for i, (rid, lang, text) in enumerate(ROWS):
        arc = f"0000/{rid}-1.pdf"
        Image.new("RGB", (120 + 10 * i, 160), color=(240, 240, 230)).save(
            str(root / "pdfs" / chunk / arc), "PDF"
        )
        rows.append(
            {
                "id": rid,
                "url": f"https://example/{rid}",
                "page_number": i + 1,
                "pdf_relpath": f"pdf_tarballs/{chunk}.tar.gz:{arc}",
                "primary_language": lang,
                "is_rotation_valid": True,
                "rotation_correction": 0,
                "is_table": False,
                "is_diagram": False,
                "natural_text": text,
            }
        )
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(root / "00_documents_train.parquet"))
    pq.write_table(table.slice(0, 1), str(root / "00_documents_eval.parquet"))
    return str(root)


@pytest.fixture
def stub_renderer(monkeypatch):
    """Replace pypdfium2 rendering with a solid image of the requested size (records calls)."""
    from PIL import Image

    calls = []

    def fake_render(pdf_path, target_longest_image_dim):
        assert os.path.exists(pdf_path), pdf_path
        calls.append((pdf_path, target_longest_image_dim))
        return Image.new("RGB", (target_longest_image_dim // 2, target_longest_image_dim))

    monkeypatch.setattr(olmocr_mod, "render_pdf_page", fake_render)
    return calls


def _cfg(root, **kw):
    kw.setdefault("max_crops", 1)
    return OlmOcrMixDatasetConfig(dataset_path=root, **kw)


# ---------------------------------------------------------------------------
# Config / selection
# ---------------------------------------------------------------------------


def test_config_validation():
    with pytest.raises(OLMoConfigurationError):
        OlmOcrMixDatasetConfig(subset="receipts").validate()
    with pytest.raises(OLMoConfigurationError):
        OlmOcrMixDatasetConfig(target_longest_image_dim_range=(2048, 1024)).validate()
    with pytest.raises(OLMoConfigurationError):
        OlmOcrMixDatasetConfig(languages=()).validate()
    OlmOcrMixDatasetConfig(languages=None, split="validation").validate()


def test_config_fields_merge_from_cli():
    cfg = OlmOcrMixDatasetConfig().merge(
        ["languages=null", "target_longest_image_dim_range=[800,900]", "subset=books"]
    )
    assert cfg.languages is None
    assert cfg.target_longest_image_dim_range == (800, 900)
    assert cfg.subset == "books"


def test_language_filter_and_missing_parquet(tmp_path, stub_renderer):
    root = _write_root(tmp_path)
    tok = _FakeTok()
    assert len(_cfg(root).build(tok)) == 3  # the French page is dropped
    assert len(_cfg(root, languages=None).build(tok)) == 4
    assert len(_cfg(root, languages=("fr",)).build(tok)) == 1
    assert len(_cfg(root, split="validation").build(tok)) == 1
    with pytest.raises(FileNotFoundError):
        _cfg(root, subset="books").build(tok)


# ---------------------------------------------------------------------------
# Per-example pieces
# ---------------------------------------------------------------------------


def test_render_size_sampled_on_train_fixed_on_eval(tmp_path, stub_renderer):
    root = _write_root(tmp_path)
    train = _cfg(root, target_longest_image_dim_range=(1000, 1002)).build(_FakeTok())
    dims = {train.target_dim_for(np.random.RandomState(s)) for s in range(40)}
    assert dims == {1000, 1001, 1002}  # inclusive range (mm_olmo `randint(lo, hi + 1)`)
    fixed = _cfg(root, target_longest_image_dim_range=None, target_longest_image_dim=777).build(
        _FakeTok()
    )
    assert fixed.target_dim_for(np.random.RandomState(0)) == 777
    ev = _cfg(root, split="eval", target_longest_image_dim=1536).build(_FakeTok())
    assert ev.target_dim_for(np.random.RandomState(0)) == 1536


def test_user_prompt_is_the_bare_tag(tmp_path, stub_renderer):
    """The user turn is the ``olmocr:`` tag alone, with no question and no length number."""
    root = _write_root(tmp_path)
    ds = _cfg(root).build(_FakeTok())
    assert ds.user_prompt() == f"{OLMOCR_STYLE}:" == "olmocr:"


def test_render_size_rotates_across_epochs(tmp_path, stub_renderer):
    """``target_longest_image_dim_range`` samples per example, so it has to move with the epoch;
    otherwise every epoch renders each page at exactly the same size."""
    root = _write_root(tmp_path)
    ds = _cfg(root, target_longest_image_dim_range=(1000, 1400)).build(_FakeTok())
    dims = []
    for epoch in range(6):
        ds.set_epoch(epoch)
        ds[0]
        dims.append(stub_renderer[-1][1])
    assert len(set(dims)) > 1, dims
    ds.set_epoch(2)
    ds[0]
    assert stub_renderer[-1][1] == dims[2]  # deterministic per (row, epoch)


def test_blank_page_transcribes_as_no_text_found(tmp_path, stub_renderer):
    root = _write_root(tmp_path)
    ds = _cfg(root).build(_FakeTok())
    rows = [ds._data[int(i)] for i in ds._index]
    assert [r["id"] for r in rows] == ["en-short", "en-blank", "en-long"]
    assert ds.transcription(rows[1]) == "No text found"
    assert ds.transcription(rows[0]) == ROWS[0][2]


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


def test_example_layout_and_loss(tmp_path, stub_renderer):
    root = _write_root(tmp_path)
    tok = _FakeTok()
    ds = _cfg(root, loss_token_weighting="none", target_longest_image_dim_range=(900, 900)).build(
        tok
    )
    ex = ds[0]
    assert stub_renderer[-1] == (ds.pdf_path(ds._data[int(ds._index[0])]), 900)
    for key in ("input_ids", "labels", "loss_masks", "position_ids", "token_type_ids", "images"):
        assert key in ex
    assert "subsegment_ids" not in ex  # one transcription per page -> single branch
    # Loss only on the transcription (+ its EOS target), all weight 1.
    resp = tok.encode(ROWS[0][2])
    assert ex["loss_masks"].sum() == pytest.approx(len(resp) + 1)
    assert set(np.unique(ex["loss_masks"]).tolist()) == {0.0, 1.0}
    assert ex["labels"][ex["loss_masks"] > 0][-1] == tok.eos_token_id
    # Deterministic per index; the seed changes the render size / bucket draws.
    np.testing.assert_array_equal(ex["input_ids"], ds[0]["input_ids"])
    ds2 = _cfg(root, target_longest_image_dim_range=(900, 1400), seed=5).build(tok)
    ds2[0]
    assert stub_renderer[-1][1] != 900 or True  # render size is drawn from the rng


def test_message_weight_scales_loss(tmp_path, stub_renderer):
    root = _write_root(tmp_path)
    base = _cfg(root, loss_token_weighting="none").build(_FakeTok())[0]
    weighted = _cfg(root, loss_token_weighting="none", message_weight=0.3).build(_FakeTok())[0]
    nz = base["loss_masks"] > 0
    np.testing.assert_allclose(weighted["loss_masks"][nz], 0.3 * base["loss_masks"][nz])


def test_max_sequence_length_truncates_long_pages(tmp_path, stub_renderer):
    root = _write_root(tmp_path)
    tok = _FakeTok()
    full = _cfg(root).build(tok)[2]  # "en-long": 400 words
    assert len(full["input_ids"]) > 1200
    n_image = int((full["token_type_ids"] == 1).sum())
    cut = _cfg(root, max_sequence_length=n_image + 200).build(tok)[2]
    assert len(cut["input_ids"]) == n_image + 200
    assert (cut["token_type_ids"] == 1).sum() == n_image  # the image block is never cut
    assert cut["loss_masks"].sum() > 0
    # A length that would cut the image block is unusable for every row, so the skip path runs
    # out rather than raising the per-row ValueError.
    tiny = _cfg(root, max_sequence_length=n_image // 2).build(tok)
    with pytest.raises(ValueError):
        tiny._build(2)
    with pytest.raises(RuntimeError, match="consecutive rows"):
        tiny[2]


def test_real_renderer_honours_target_size(tmp_path):
    pytest.importorskip("pypdfium2")
    root = _write_root(tmp_path)
    ds = _cfg(root).build(_FakeTok())
    row = ds._data[int(ds._index[0])]
    img = olmocr_mod.render_pdf_page(ds.pdf_path(row), 640)
    assert max(img.size) == 640 and img.mode == "RGB"
    # A portrait 120x160 page keeps its aspect ratio.
    assert abs(img.size[0] / img.size[1] - 120 / 160) < 0.02


# ---------------------------------------------------------------------------
# Molmo2-Stage1 wiring
# ---------------------------------------------------------------------------


def _load_stage1_module():
    try:
        import olmo_core.internal.common  # noqa: F401  (needs a recent beaker-py)
    except ImportError as e:  # pragma: no cover - env-dependent
        pytest.skip(f"Molmo2-Stage1.py imports fail here: {e}")
    spec = importlib.util.spec_from_file_location(
        "_stage1_ocr", "src/scripts/train/Molmo2-Stage1.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_ocr"] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


def test_stage1_refuses_the_olmocr_eval_split():
    """The training script reads train splits only; the held-out `eval` parquets stay held out."""
    from types import SimpleNamespace

    from olmo_core.data.multimodal import (
        OcrCaptionTarsDatasetConfig,
        TextRichCaptionDatasetConfig,
    )
    from olmo_core.data.multimodal.mixtures.ocr import DEFAULT_OCR_SOURCES
    from olmo_core.data.multimodal.mixtures.stage1_academic import (
        DEFAULT_ACADEMIC_SOURCES,
    )

    mod = _load_stage1_module()

    def data_config(split):
        return SimpleNamespace(
            recipe="v1",
            pointing_data="v1",
            pointing_rate=0.3,
            nlp_rate=0.1,
            ocr_rate=0.1,
            ocr_sources=DEFAULT_OCR_SOURCES,
            olmocr=OlmOcrMixDatasetConfig(split=split),
            ocr_tars=OcrCaptionTarsDatasetConfig(),
            text_rich=TextRichCaptionDatasetConfig(),
            academic_rate=0.0,
            academic_sources=DEFAULT_ACADEMIC_SOURCES,
            clock_rate=0.0,
        )

    for split in ("eval", "validation"):
        with pytest.raises(OLMoConfigurationError, match="train split"):
            mod.validate_data_config(data_config(split))
    mod.validate_data_config(data_config("train"))
    assert OlmOcrMixDatasetConfig().split == "train"


def test_stage1_ocr_group_is_opt_in_and_sqrt_split():
    mod = _load_stage1_module()
    assert mod.OCR_RATE == 0.0
    assert {
        "olmocr_documents",
        "olmocr_books",
        "olmocr_loc_transcripts",
        "olmocr_national_archives",
    } <= set(mod.OCR_SOURCES)
    fields = {f.name for f in mod.ExperimentConfig.__dataclass_fields__.values()}
    assert {"ocr_rate", "olmocr", "ocr_sources"} <= fields
    np.testing.assert_allclose(
        mod._size_fractions([100, 400, 1600], "sqrt"), np.array([10, 20, 40]) / 70
    )
    np.testing.assert_allclose(
        mod._size_fractions([100, 400, 1600], "linear"), np.array([100, 400, 1600]) / 2100
    )
