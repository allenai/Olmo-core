"""CPU tests for the image-text shards source (:mod:`olmo_core.data.multimodal.image_text_shards`):
the index filters, reading an image from its tar by offset, the example layout and loss, the
per-(row, epoch) streams, truncation, unusable rows, the high-resolution draw, and the stage-1
builder. Each test writes its own tiny dataset: one tar of images and a parquet index."""

import io
import os
import tarfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from olmo_core.data.multimodal import ImageTextShardsDatasetConfig
from olmo_core.data.multimodal.image_text_shards import (
    IMAGE_TEXT_SHARDS_STYLE,
    index_files,
)
from olmo_core.exceptions import OLMoConfigurationError


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


# ---------------------------------------------------------------------------
# Fixture: a tar of images and its index
# ---------------------------------------------------------------------------

LONG = " ".join(f"word{i}" for i in range(400))

# id, kind, split, keep, target, image format, (width, height)
ROWS = [
    ("eq", "equation", "train", True, r"\[ x^2 + y^2 = z^2 \]", "PNG", (64, 48)),
    ("tab", "table", "train", False, "<table><tr><td>1</td></tr></table>", "JPEG", (64, 48)),
    ("page", "page", "train", True, LONG, "PNG", (64, 48)),
    ("held", "page", "heldout", True, "held out page", "PNG", (64, 48)),
    ("out", "page", "excluded", True, "benchmark page", "PNG", (64, 48)),
]


def _image_file(fmt, size, color):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format=fmt)
    return buffer.getvalue()


def _write_dataset(tmp_path, rows=ROWS, *, parts=False, payloads=None, name="ds"):
    """Write ``rows`` as one tar and an index (``index/part-*.parquet`` when ``parts``).

    :param payloads: Optional ``id -> bytes`` stored in place of that row's image file.

    :returns: The dataset directory and the stored bytes by id.
    """
    root = tmp_path / name
    (root / "shards").mkdir(parents=True)
    stored = {}
    with tarfile.open(root / "shards" / "00000.tar", "w") as tar:
        for k, (rid, _, _, _, _, fmt, size) in enumerate(rows):
            data = (payloads or {}).get(rid) or _image_file(fmt, size, (40 * k, 30, 30))
            info = tarfile.TarInfo(f"{rid}.{fmt.lower()}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            stored[rid] = data
    with tarfile.open(root / "shards" / "00000.tar") as tar:
        members = {m.name.split(".")[0]: m for m in tar.getmembers()}
    table = pa.table(
        {
            "id": [r[0] for r in rows],
            "shard": ["00000.tar"] * len(rows),
            "offset": pa.array([members[r[0]].offset_data for r in rows], pa.int64()),
            "size": pa.array([members[r[0]].size for r in rows], pa.int64()),
            "width": pa.array([r[6][0] for r in rows], pa.int32()),
            "height": pa.array([r[6][1] for r in rows], pa.int32()),
            "target": [r[4] for r in rows],
            "kind": [r[1] for r in rows],
            "split": [r[2] for r in rows],
            "keep": [r[3] for r in rows],
            "tier": ["Tier1"] * len(rows),
        }
    )
    if parts:
        (root / "index").mkdir()
        half = len(rows) // 2
        pq.write_table(table.slice(0, half), str(root / "index" / "part-a.parquet"))
        pq.write_table(table.slice(half), str(root / "index" / "part-b.parquet"))
    else:
        pq.write_table(table, str(root / "index.parquet"))
    return str(root), stored


def _cfg(root, **kw):
    kw.setdefault("max_crops", 1)
    return ImageTextShardsDatasetConfig(dataset_path=root, **kw)


def _targets(ds):
    return [ds.row(i)["target"] for i in range(len(ds))]


def _contains(seq, sub) -> bool:
    return any(seq[i : i + len(sub)] == sub for i in range(len(seq) - len(sub) + 1))


# ---------------------------------------------------------------------------
# Config / index
# ---------------------------------------------------------------------------


def test_config_validation(tmp_path):
    root, _ = _write_dataset(tmp_path)
    for bad in (
        dict(dataset_path=""),
        dict(split="excluded"),
        dict(kinds=()),
        dict(keep_column=""),
        dict(style=""),
        dict(max_crops=0),
        dict(p_high_res=1.5),
        dict(high_res_max_crops=0),
    ):
        with pytest.raises(OLMoConfigurationError):
            ImageTextShardsDatasetConfig(**{"dataset_path": root, **bad}).validate()
    cfg = ImageTextShardsDatasetConfig().merge(
        [f"dataset_path={root}", "kinds=[equation,table]", "keep_column=keep", "max_crops=4"]
    )
    assert cfg.kinds == ("equation", "table") and cfg.keep_column == "keep" and cfg.max_crops == 4
    assert cfg.style == IMAGE_TEXT_SHARDS_STYLE == "olmocr" and cfg.loss_token_weighting == "none"


def test_index_filters(tmp_path):
    root, _ = _write_dataset(tmp_path)
    tok = _FakeTok()
    # Training rows only: the held-out and the excluded row are never indexed.
    assert _targets(_cfg(root).build(tok)) == [ROWS[0][4], ROWS[1][4], ROWS[2][4]]
    assert _targets(_cfg(root, kinds=("equation", "table")).build(tok)) == [ROWS[0][4], ROWS[1][4]]
    assert _targets(_cfg(root, keep_column="keep").build(tok)) == [ROWS[0][4], ROWS[2][4]]
    assert _targets(_cfg(root, keep_column="keep", kinds=("table",)).build(tok)) == []
    assert _targets(_cfg(root, split="heldout").build(tok)) == ["held out page"]
    with pytest.raises(OLMoConfigurationError, match="no_such"):
        _cfg(root, keep_column="no_such").build(tok)
    with pytest.raises(FileNotFoundError, match="index"):
        _cfg(str(tmp_path / "missing")).build(tok)


def test_index_parts_are_read_in_name_order(tmp_path):
    single, _ = _write_dataset(tmp_path, name="single")
    split, _ = _write_dataset(tmp_path, parts=True, name="split")
    assert [os.path.basename(p) for p in index_files(split)] == ["part-a.parquet", "part-b.parquet"]
    assert index_files(single) == [os.path.join(single, "index.parquet")]
    tok = _FakeTok()
    a, b = _cfg(single).build(tok), _cfg(split).build(tok)
    assert _targets(a) == _targets(b)
    np.testing.assert_array_equal(a[0]["input_ids"], b[0]["input_ids"])


def test_index_requires_the_format_columns(tmp_path):
    root, _ = _write_dataset(tmp_path)
    path = os.path.join(root, "index.parquet")
    pq.write_table(pq.read_table(path).drop(["offset"]), path)
    with pytest.raises(OLMoConfigurationError, match="offset"):
        _cfg(root).build(_FakeTok())


def test_image_is_read_from_the_tar_by_offset(tmp_path):
    root, stored = _write_dataset(tmp_path)
    ds = _cfg(root).build(_FakeTok())
    for i, rid in enumerate(("eq", "tab", "page")):
        row = ds.row(i)
        assert row["shard"] == "00000.tar" and ds.image_bytes(row) == stored[rid]
        assert ds.image(row).size == (64, 48)
    with pytest.raises(IndexError):
        ds.row(3)


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


def test_example_layout_and_loss(tmp_path):
    root, _ = _write_dataset(tmp_path)
    tok = _FakeTok()
    ds = _cfg(root).build(tok)
    ex = ds[0]
    for key in ("input_ids", "labels", "loss_masks", "position_ids", "token_type_ids", "images"):
        assert key in ex
    assert "subsegment_ids" not in ex  # one target per image -> single branch
    # The user turn is the bare style tag, with no question.
    assert ds.user_prompt() == "olmocr:"
    assert _contains(ex["input_ids"].tolist(), tok.encode("olmocr:"))
    # Loss only on the target (+ its EOS), all weight 1; the target is used verbatim.
    resp = tok.encode(ROWS[0][4])
    assert ex["loss_masks"].sum() == pytest.approx(len(resp) + 1)
    assert set(np.unique(ex["loss_masks"]).tolist()) == {0.0, 1.0}
    assert ex["labels"][ex["loss_masks"] > 0][-1] == tok.eos_token_id
    assert ex["labels"][ex["loss_masks"] > 0][:-1].tolist() == resp
    assert ex["images"].shape[0] == 2  # max_crops=1 -> the global crop + one local crop

    styled = _cfg(root, style="equation_ocr").build(tok)
    assert styled.user_prompt() == "equation_ocr:"
    assert _contains(styled[0]["input_ids"].tolist(), tok.encode("equation_ocr:"))
    weighted = _cfg(root, message_weight=0.3).build(tok)[0]
    nz = ex["loss_masks"] > 0
    np.testing.assert_allclose(weighted["loss_masks"][nz], 0.3 * ex["loss_masks"][nz])


def test_document_format_uses_the_lm_image_tokens(tmp_path):
    from .stage1_document_mode_test import DOLMA2_IDS, _assert_document_example, _Tok

    root, _ = _write_dataset(tmp_path)
    ds = _cfg(
        root, token_ids=DOLMA2_IDS, message_format="document", max_sequence_length=4096
    ).build(_Tok())
    for index in range(len(ds)):
        _assert_document_example(ds.get(index, 0))


def test_example_is_deterministic_per_row_and_epoch(tmp_path):
    big = [(r[0], r[1], r[2], r[3], r[4], r[5], (1200, 900)) for r in ROWS]
    root, _ = _write_dataset(tmp_path, big)
    tok = _FakeTok()
    ds = _cfg(root, max_crops=2, high_res_max_crops=6, p_high_res=0.5).build(tok)
    crops = []
    for epoch in range(24):
        a, b = ds.get(0, epoch), ds.get(0, epoch)
        np.testing.assert_array_equal(a["input_ids"], b["input_ids"])
        np.testing.assert_array_equal(a["images"], b["images"])
        crops.append(a["images"].shape[0])
    assert set(crops) == {3, 7}  # the draw follows the epoch
    # `set_epoch` (the epoch-shuffled loader) and `get` (the stream loader) agree.
    ds.set_epoch(5)
    np.testing.assert_array_equal(ds[0]["input_ids"], ds.get(0, 5)["input_ids"])
    other = _cfg(root, max_crops=2, high_res_max_crops=6, p_high_res=0.5, seed=1).build(tok)
    assert [other.get(0, e)["images"].shape[0] for e in range(24)] != crops


def test_high_res_share_draws_the_larger_crop_budget(tmp_path):
    """``p_high_res`` crops an image with ``high_res_max_crops`` on the example's stream; zero
    leaves the example as it was, and the target and its loss weight do not change."""
    big = [(r[0], r[1], r[2], r[3], r[4], r[5], (1200, 900)) for r in ROWS]
    root, _ = _write_dataset(tmp_path, big)
    tok = _FakeTok()

    def build(**kw):
        kw.setdefault("high_res_max_crops", 6)
        return _cfg(root, max_crops=2, **kw).build(tok)

    low, off, high = (
        build(high_res_max_crops=24)[0],
        build(p_high_res=0.0)[0],
        build(p_high_res=1.0)[0],
    )
    assert low["images"].shape[0] == 3  # two tiles and the global crop
    np.testing.assert_array_equal(low["input_ids"], off["input_ids"])
    np.testing.assert_array_equal(low["images"], off["images"])
    assert high["images"].shape[0] == 7  # six tiles and the global crop
    assert len(high["input_ids"]) > len(low["input_ids"])
    assert high["loss_masks"].sum() == pytest.approx(low["loss_masks"].sum())


def test_max_sequence_length_truncates_long_targets(tmp_path):
    root, _ = _write_dataset(tmp_path)
    tok = _FakeTok()
    full = _cfg(root).build(tok)[2]  # the 400-word page
    assert len(full["input_ids"]) > 1200
    n_image = int((full["token_type_ids"] == 1).sum())
    cut = _cfg(root, max_sequence_length=n_image + 200).build(tok)[2]
    assert len(cut["input_ids"]) == n_image + 200
    assert (cut["token_type_ids"] == 1).sum() == n_image  # the image block is never cut
    assert cut["loss_masks"].sum() > 0
    tiny = _cfg(root, max_sequence_length=n_image // 2).build(tok)
    with pytest.raises(ValueError):
        tiny._build(2)
    with pytest.raises(RuntimeError, match="consecutive rows"):
        tiny[2]


def test_unusable_rows_are_skipped(tmp_path):
    """A row whose bytes are not an image, whose target is empty or blank, or whose offset runs
    past the tar is replaced by the next usable row, deterministically."""
    rows = [
        ("broken", "page", "train", True, "text of a broken image", "PNG", (64, 48)),
        ("empty", "page", "train", True, "", "PNG", (64, 48)),
        ("blank", "page", "train", True, "  \n ", "PNG", (64, 48)),
        ("good", "page", "train", True, "a good page", "PNG", (64, 48)),
    ]
    root, _ = _write_dataset(tmp_path, rows, payloads={"broken": b"not an image at all"})
    tok = _FakeTok()
    ds = _cfg(root).build(tok)
    assert len(ds) == 4
    good = ds._build(3)
    for i, error in ((0, Exception), (1, ValueError), (2, ValueError)):
        with pytest.raises(error):
            ds._build(i)
        np.testing.assert_array_equal(ds[i]["input_ids"], good["input_ids"])
        np.testing.assert_array_equal(ds.get(i, 2)["input_ids"], ds.get(3, 2)["input_ids"])
    assert ds._warned > 0

    # An index whose offsets point past the end of the tar: a short read, never a wrong image.
    path = os.path.join(root, "index.parquet")
    table = pq.read_table(path)
    column = table.schema.get_field_index("offset")
    moved = pa.array([10**9] + table.column("offset").to_pylist()[1:], pa.int64())
    pq.write_table(table.set_column(column, "offset", moved), path)
    short = _cfg(root).build(tok)
    with pytest.raises(RuntimeError, match="read 0 of"):
        short.image_bytes(short.row(0))


# ---------------------------------------------------------------------------
# Stage-1 builder
# ---------------------------------------------------------------------------


def test_stage1_builder_reads_a_dataset_with_the_olmocr_settings(tmp_path):
    from olmo_core.internal.vision_alignment_data import build_stage1_shard_source

    root, _ = _write_dataset(tmp_path)
    cfg = build_stage1_shard_source(
        262144, root, keep_column="keep", kinds=["equation", "page"], max_crops=4
    )
    cfg.validate()
    assert cfg.split == "train" and cfg.kinds == ("equation", "page") and cfg.keep_column == "keep"
    assert cfg.max_crops == 4 and cfg.max_sequence_length == 262144 and cfg.style == "olmocr"
    assert cfg.message_format == "document" and cfg.loss_token_weighting == "none"
    assert cfg.p_high_res == 0.0 and cfg.high_res_max_crops == 24
    plain = build_stage1_shard_source(65536, root)
    assert plain.kinds is None and plain.keep_column is None and plain.max_crops == 8
    for args, kw in (
        ((0, root), {}),
        ((8, ""), {}),
        ((8, root), {"kinds": ()}),
        ((8, root), {"max_crops": 0}),
    ):
        with pytest.raises(ValueError):
            build_stage1_shard_source(*args, **kw)
