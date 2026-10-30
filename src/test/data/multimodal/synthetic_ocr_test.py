"""CPU tests for the synthetic OCR sources: the visual-order target of the NVIDIA set, the receipt
target, train-split-only reading, the style tags and the OCR registry wiring."""

import io
import json
import pickle
from typing import Any, Dict

import numpy as np
import pytest

from olmo_core.data.multimodal import (
    NvidiaSynthOcrDatasetConfig,
    SyntheticReceiptsDatasetConfig,
)
from olmo_core.data.multimodal.mixtures import ocr as ocr_mix
from olmo_core.data.multimodal.synthetic_ocr import (
    layout_text,
    prepare_synthetic_receipts,
    receipt_text,
)
from olmo_core.exceptions import OLMoConfigurationError


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


def _jpeg(seed: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64 + seed, 48), color=(seed * 30 % 255, 80, 140)).save(buf, format="JPEG")
    return buf.getvalue()


def _box(text, x, y, w, h):
    return {"text": text, "bbox": [x, y, w, h]}


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def test_layout_text_reads_rows_top_down_and_left_to_right():
    ann = {
        "line_bboxes": [
            _box("second line", 10, 60, 100, 20),
            _box("right", 200, 12, 50, 20),
            _box("left", 10, 10, 50, 20),
        ]
    }
    assert layout_text(ann) == "left right second line"


def test_layout_text_vertical_word_joins_only_the_line_through_its_centre():
    # A tall vertical word spanning three lines must not merge them into one row.
    ann = {
        "line_bboxes": [
            _box("games, with all", 300, 100, 200, 20),
            _box("quirks, have made", 300, 140, 200, 20),
            _box("setup a bit", 300, 180, 200, 20),
            _box("available", 50, 90, 20, 120),  # centre y = 150, on the "quirks" line
        ]
    }
    assert layout_text(ann) == "games, with all available quirks, have made setup a bit"


def test_layout_text_falls_back_to_word_boxes_and_skips_blanks():
    ann = {
        "word_bboxes": [_box("b", 50, 0, 10, 10), _box("a", 0, 0, 10, 10), _box(" ", 5, 5, 1, 1)]
    }
    assert layout_text(ann) == "a b"
    assert layout_text({"line_bboxes": []}) == ""


def test_receipt_text_collapses_padding_and_drops_rules():
    full = (
        "JANSEN STORES\n"
        "------------------------------------------------\n"
        "DIS MEN JUN                               $12.99\n"
        "  2 x $1.99\n"
        "\n"
        "================\n"
        "TOTAL                                     $45.69\n"
    )
    assert receipt_text(full) == "JANSEN STORES\nDIS MEN JUN $12.99\n2 x $1.99\nTOTAL $45.69"


# ---------------------------------------------------------------------------
# NVIDIA OCR-Synthetic (HDF5)
# ---------------------------------------------------------------------------


def _write_h5(path, samples):
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        vb = h5py.vlen_dtype(np.dtype("uint8"))
        images = f.create_dataset("images", (len(samples),), dtype=vb)
        for i, (seed, _) in enumerate(samples):
            images[i] = np.frombuffer(_jpeg(seed), dtype=np.uint8)
        f.create_dataset(
            "annotations",
            data=[json.dumps(ann) for _, ann in samples],
            dtype=h5py.string_dtype(),
        )
        f.create_dataset(
            "labels",
            data=[" ".join(b["text"] for b in ann["line_bboxes"]) for _, ann in samples],
            dtype=h5py.string_dtype(),
        )


def _nvidia_root(tmp_path):
    root = tmp_path / "nvidia"
    a = {"line_bboxes": [_box("world", 60, 5, 40, 10), _box("hello", 5, 5, 40, 10)]}
    b = {"line_bboxes": [_box("second", 5, 40, 40, 10), _box("first", 5, 5, 40, 10)]}
    empty: Dict[str, list] = {"line_bboxes": []}
    _write_h5(root / "en" / "train" / "train_000.h5", [(1, a), (2, empty)])
    _write_h5(root / "en" / "train" / "train_001.h5", [(3, b)])
    # Held-out and non-English files must never be read.
    _write_h5(root / "en" / "validation" / "validation_000.h5", [(4, a)] * 5)
    _write_h5(root / "ja" / "train" / "train_000.h5", [(5, a)] * 7)
    return str(root)


def test_nvidia_reads_only_english_train_files(tmp_path):
    ds = NvidiaSynthOcrDatasetConfig(dataset_path=_nvidia_root(tmp_path), max_crops=1).build(
        _PromptTok()
    )
    assert len(ds) == 3
    assert [p.split("/")[-3:] for p in ds.files] == [
        ["en", "train", "train_000.h5"],
        ["en", "train", "train_001.h5"],
    ]
    assert ds.locate(2) == (ds.files[1], 0)


def test_nvidia_example_is_synth_ocr_tag_and_visual_order(tmp_path):
    tok = _PromptTok()
    ds = NvidiaSynthOcrDatasetConfig(dataset_path=_nvidia_root(tmp_path), max_crops=1).build(tok)
    assert layout_text(ds.read(0)[1]) == "hello world"  # the label says "world hello"
    ex = ds[0]
    assert ex["loss_masks"].sum() > 0
    assert [p for p in tok.prompts if p] == ["synth_ocr:"]


def test_nvidia_skips_an_image_with_no_text(tmp_path):
    ds = NvidiaSynthOcrDatasetConfig(dataset_path=_nvidia_root(tmp_path), max_crops=1).build(
        _PromptTok()
    )
    np.testing.assert_array_equal(ds[1]["input_ids"], ds[2]["input_ids"])


def test_nvidia_pickles_without_open_handles(tmp_path):
    ds = NvidiaSynthOcrDatasetConfig(dataset_path=_nvidia_root(tmp_path), max_crops=1).build(
        _PromptTok()
    )
    ds.read(0)
    assert ds._handles
    clone = pickle.loads(pickle.dumps(ds))
    assert clone._handles == {}
    assert layout_text(clone.read(2)[1]) == "first second"


def test_nvidia_missing_files_is_a_clear_error(tmp_path):
    pytest.importorskip("h5py")
    with pytest.raises(OLMoConfigurationError, match="en/train"):
        NvidiaSynthOcrDatasetConfig(dataset_path=str(tmp_path)).build(_PromptTok())


# ---------------------------------------------------------------------------
# Synthetic receipts (parquet -> Arrow)
# ---------------------------------------------------------------------------


def _write_receipts(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "receipts"
    (root / "data").mkdir(parents=True)

    def table(rows):
        return pa.table(
            {
                "id": [r[0] for r in rows],
                "image_clean": [{"bytes": _jpeg(0), "path": None} for _ in rows],
                "image_photo": [{"bytes": _jpeg(i + 1), "path": None} for i, _ in enumerate(rows)],
                "full_text": [f"SHOP {r[0]}\n-----\nTOTAL      $1.00" for r in rows],
                "locale": [r[1] for r in rows],
                "split_policy": [r[2] for r in rows],
            }
        )

    pq.write_table(
        table([("t0", "US", "train"), ("t1", "DE", "train"), ("t2", "UK", "train")]),
        str(root / "data" / "train-00000-of-00001.parquet"),
    )
    pq.write_table(
        table([("e0", "US", "eval"), ("e1", "UK", "eval")]),
        str(root / "data" / "eval-00000-of-00001.parquet"),
    )
    return str(root)


def test_prepare_keeps_english_train_rows_only(tmp_path):
    from olmo_core.data.multimodal.dataset_compat import load_from_disk_compat

    root = _write_receipts(tmp_path)
    out = str(tmp_path / "prepared")
    assert prepare_synthetic_receipts(root, out) == 2
    data = load_from_disk_compat(out)
    assert sorted(data["id"]) == ["t0", "t2"]
    assert set(data["locale"]) == {"US", "UK"}
    assert isinstance(data[0]["image_photo"], bytes)


def test_receipt_example_is_receipt_ocr_tag(tmp_path):
    root = _write_receipts(tmp_path)
    prepare_synthetic_receipts(root, str(tmp_path / "receipts" / "en_train_arrow"))
    tok = _PromptTok()
    ds = SyntheticReceiptsDatasetConfig(dataset_path=root, max_crops=1).build(tok)
    assert len(ds) == 2
    assert ds[0]["loss_masks"].sum() > 0
    assert [p for p in tok.prompts if p] == ["receipt_ocr:"]


def test_receipts_without_prepared_dir_is_a_clear_error(tmp_path):
    root = _write_receipts(tmp_path)
    with pytest.raises(OLMoConfigurationError, match="prepare_synthetic_receipts"):
        SyntheticReceiptsDatasetConfig(dataset_path=root).build(_PromptTok())


def test_receipts_refuse_a_non_english_prepared_dir(tmp_path):
    root = _write_receipts(tmp_path)
    prepare_synthetic_receipts(
        root, str(tmp_path / "receipts" / "en_train_arrow"), locales=("US", "DE")
    )
    with pytest.raises(OLMoConfigurationError, match="locales"):
        SyntheticReceiptsDatasetConfig(dataset_path=root).build(_PromptTok())


# ---------------------------------------------------------------------------
# OCR registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["nvidia_synth_en", "synth_receipts_en"])
def test_synthetic_sources_are_default_transcription_sources(name):
    assert name in ocr_mix.DEFAULT_OCR_SOURCES
    assert ocr_mix.ocr_task(name) == ocr_mix.TRANSCRIPTION


def test_build_ocr_source_uses_the_synthetic_configs(tmp_path):
    root = _write_receipts(tmp_path)
    prepare_synthetic_receipts(root, str(tmp_path / "receipts" / "en_train_arrow"))
    templates: Dict[str, Any] = dict(
        olmocr=ocr_mix.OlmOcrMixDatasetConfig(),
        tars=ocr_mix.OcrCaptionTarsDatasetConfig(),
        text_rich=ocr_mix.TextRichCaptionDatasetConfig(),
        nvidia_synth=NvidiaSynthOcrDatasetConfig(dataset_path=_nvidia_root(tmp_path), max_crops=1),
        receipts=SyntheticReceiptsDatasetConfig(dataset_path=root, max_crops=1),
    )
    assert len(ocr_mix.build_ocr_source("nvidia_synth_en", _PromptTok(), **templates)) == 3
    assert len(ocr_mix.build_ocr_source("synth_receipts_en", _PromptTok(), **templates)) == 2


# ---------------------------------------------------------------------------
# Weighting-size cap
# ---------------------------------------------------------------------------


def test_synthetic_sizes_are_capped_at_the_largest_real_source_of_their_task():
    names = ["olmocr_documents", "textocr", "nvidia_synth_en", "synth_receipts_en", "text_rich_doc"]
    sizes = [218_831, 21_749, 1_460_304, 16_437, 438_267]
    assert ocr_mix.ocr_weighting_sizes(names, sizes) == [
        218_831,
        21_749,
        218_831,  # capped at olmocr_documents, the largest real transcription source
        16_437,  # already below the cap
        438_267,  # a figure-caption source is not a transcription source's cap
    ]


def test_synthetic_sizes_are_uncapped_without_a_real_source_of_their_task():
    names = ["nvidia_synth_en", "synth_receipts_en", "text_rich_doc"]
    sizes = [1_460_304, 16_437, 438_267]
    assert ocr_mix.ocr_weighting_sizes(names, sizes) == sizes
