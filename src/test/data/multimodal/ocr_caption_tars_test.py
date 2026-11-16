"""CPU tests for the webdataset-style caption tars (oe-encoder ``*_v6_tars`` OCR sources): shard
indexing and its cache, sample reads, ``<text>`` stripping, the style tag, example layout,
truncation, and the OCR source registry / Molmo2-Stage1 wiring."""

import importlib.util
import io
import json
import os
import sys
import tarfile
from typing import Any, Dict, List

import numpy as np
import pytest

from olmo_core.data.multimodal import (
    OcrCaptionTarsDatasetConfig,
    TarShardIndex,
    TextRichCaptionDatasetConfig,
)
from olmo_core.data.multimodal import ocr_caption_tars as ct
from olmo_core.data.multimodal.mixtures import ocr as ocr_mix
from olmo_core.data.multimodal.olmocr import OlmOcrMixDatasetConfig
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
# Fixture: two shards of (image, json) pairs, plus an orphan image and a stray member
# ---------------------------------------------------------------------------

SAMPLES = [
    # (shard, key, ext, caption)
    (0, "a_000", "png", "<text>FELIX PRIVAT DBU 889</text>"),
    (0, "a_001", "jpg", "A chart with three bars."),
    (0, "a_002", "jpg", "<text>\n  multi\nline  \n</text>"),
    (1, "b_000", "png", "<text></text>"),  # empty after stripping
    (1, "b_001", "jpg", "word " * 400),
]


def _png_bytes(seed, ext):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (48 + 4 * seed, 40), color=(seed * 40 % 255, 90, 120)).save(
        buf, format="PNG" if ext == "png" else "JPEG"
    )
    return buf.getvalue()


def _write_tars(tmp_path, extra_key: str = ""):
    root = tmp_path / "toy_v6_tars"
    root.mkdir()
    shards = [tarfile.open(str(root / f"toy-w0{i}-00000.tar"), "w") for i in range(2)]

    def add(tf, name, data):
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    samples = list(SAMPLES)
    if extra_key:
        samples.append((0, extra_key, "png", "<text>unicode key</text>"))
    for i, (shard, key, ext, caption) in enumerate(samples):
        add(shards[shard], f"{key}.{ext}", _png_bytes(i, ext))
        meta = {"caption": caption, "dense_caption": caption, "n_words": str(i)}
        add(shards[shard], f"{key}.json", json.dumps(meta).encode())
    add(shards[1], "orphan.jpg", _png_bytes(9, "jpg"))  # no json -> skipped
    add(shards[1], "README.txt", b"not a sample")  # unexpected extension -> ignored
    for tf in shards:
        tf.close()
    return str(root)


def _cfg(root, tmp_path, **kw):
    kw.setdefault("style", "textocr")
    kw.setdefault("max_crops", 1)
    kw.setdefault("index_cache_dir", str(tmp_path / "index_cache"))
    return OcrCaptionTarsDatasetConfig(dataset_path=root, **kw)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def test_index_pairs_members_and_skips_orphans(tmp_path):
    root = _write_tars(tmp_path)
    idx = TarShardIndex.build(TarShardIndex.list_shards(root))
    assert len(idx) == 5
    assert [str(k) for k in idx.keys] == [s[1] for s in SAMPLES]
    assert idx.shard_idx.tolist() == [s[0] for s in SAMPLES]
    img, meta = idx.read_sample(2)
    assert json.loads(meta)["caption"] == SAMPLES[2][3]
    from PIL import Image

    assert Image.open(io.BytesIO(img)).size == (48 + 4 * 2, 40)


def test_index_handles_non_ascii_keys(tmp_path):
    """Keys come from tar member names, which nothing constrains to ASCII. ``np.bytes_`` encodes
    str through the ASCII codec, so one non-ASCII name would raise ``UnicodeEncodeError`` from
    ``__init__`` -- at mixture-build time, where the loader's per-example error tolerance does not
    apply, after paying the whole shard scan."""
    root = _write_tars(tmp_path, extra_key="café_señor_日本語")
    idx = TarShardIndex.build(TarShardIndex.list_shards(root))
    assert "café_señor_日本語" in [str(k) for k in idx.keys]
    cache = str(tmp_path / "cache_unicode")
    TarShardIndex.load_or_build(root, cache_dir=cache)  # survives the npz round trip
    reloaded = TarShardIndex.load_or_build(root, cache_dir=cache)
    assert "café_señor_日本語" in [str(k) for k in reloaded.keys]
    ds = _cfg(root, tmp_path).build(_FakeTok())
    assert "café_señor_日本語" in [ds.key(i) for i in range(len(ds))]


def test_index_cache_name_carries_the_format_version(tmp_path):
    """The v1 cache stored ASCII bytes for ``keys``; a stale one must be ignored, not mis-read."""
    root = _write_tars(tmp_path)
    cache = str(tmp_path / "cache_ver")
    TarShardIndex.load_or_build(root, cache_dir=cache)
    (name,) = os.listdir(cache)
    assert f"-v{ct.INDEX_FORMAT_VERSION}-" in name


def test_index_save_is_atomic_and_uniquely_named(tmp_path, monkeypatch):
    """Two writers must not share a temp path: ranks on different hosts can collide on pid."""
    root = _write_tars(tmp_path)
    idx = TarShardIndex.build(TarShardIndex.list_shards(root))
    seen = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append(src)
        return real_replace(src, dst)

    monkeypatch.setattr(ct.os, "replace", spy)
    target = str(tmp_path / "c" / "idx.npz")
    idx.save(target)
    idx.save(target)
    assert len(set(seen)) == 2 and all(s != target for s in seen)
    assert os.listdir(tmp_path / "c") == ["idx.npz"]  # no temp left behind

    # A failed write leaves no temp file either.
    monkeypatch.setattr(ct.np, "savez", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        idx.save(target)
    assert os.listdir(tmp_path / "c") == ["idx.npz"]


def test_index_cache_roundtrip_and_reuse(tmp_path, monkeypatch):
    root = _write_tars(tmp_path)
    cache = str(tmp_path / "cache")
    idx = TarShardIndex.load_or_build(root, cache_dir=cache)
    files = os.listdir(cache)
    assert len(files) == 1 and files[0].startswith("toy_v6_tars-") and files[0].endswith(".npz")
    # Second load must come from the cache: scanning is forbidden.
    monkeypatch.setattr(ct, "_scan_shard", lambda path: pytest.fail("rescanned " + path))
    again = TarShardIndex.load_or_build(root, cache_dir=cache)
    assert again.shards == idx.shards
    np.testing.assert_array_equal(again.keys, idx.keys)
    np.testing.assert_array_equal(again.offsets, idx.offsets)
    assert not any(f.startswith("toy_v6_tars-") and ".tmp-" in f for f in os.listdir(cache))


def test_index_cache_invalidates_when_shards_change(tmp_path):
    root = _write_tars(tmp_path)
    cache = str(tmp_path / "cache")
    TarShardIndex.load_or_build(root, cache_dir=cache)
    with tarfile.open(os.path.join(root, "toy-w02-00000.tar"), "w") as tf:
        data = _png_bytes(3, "jpg")
        info = tarfile.TarInfo("c_000.jpg")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
        meta = json.dumps({"caption": "x"}).encode()
        info = tarfile.TarInfo("c_000.json")
        info.size = len(meta)
        tf.addfile(info, io.BytesIO(meta))
    idx = TarShardIndex.load_or_build(root, cache_dir=cache)
    assert len(idx) == 6 and len(os.listdir(cache)) == 2


def test_index_pairs_each_key_once(tmp_path):
    """The scan de-duplicates member stems through a set now (list membership made it O(n^2));
    a repeated stem must still yield exactly one sample, at its last offsets."""
    root = tmp_path / "dup_tars"
    root.mkdir()
    with tarfile.open(str(root / "dup-00000.tar"), "w") as tf:

        def add(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        for _ in range(2):
            add("k.png", _png_bytes(1, "png"))
            add("k.json", json.dumps({"caption": "x"}).encode())
    idx = TarShardIndex.build(TarShardIndex.list_shards(str(root)))
    assert [str(k) for k in idx.keys] == ["k"]


def test_missing_shards_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        TarShardIndex.list_shards(str(tmp_path))


# ---------------------------------------------------------------------------
# Text / prompt
# ---------------------------------------------------------------------------


def test_strip_text_tags():
    assert ct.strip_text_tags("<text>FELIX 889</text>") == "FELIX 889"
    assert ct.strip_text_tags("<text>\n a\nb \n</text>") == "a\nb"
    assert ct.strip_text_tags("plain caption") == "plain caption"
    assert ct.strip_text_tags("<text>keep <b>inner</b> tags</text>") == "keep <b>inner</b> tags"


def test_config_validation(tmp_path):
    with pytest.raises(OLMoConfigurationError):
        OcrCaptionTarsDatasetConfig().validate()  # no dataset_path
    with pytest.raises(OLMoConfigurationError):
        OcrCaptionTarsDatasetConfig(dataset_path="/x", style="").validate()
    OcrCaptionTarsDatasetConfig(dataset_path="/x").validate()


def test_dataset_text_and_prompt(tmp_path):
    root = _write_tars(tmp_path)
    ds = _cfg(root, tmp_path).build(_FakeTok())
    assert len(ds) == 5
    assert ds.key(1) == "a_001"
    assert ds.text({"caption": "<text>FELIX</text>"}) == "FELIX"
    assert (
        _cfg(root, tmp_path, strip_text_tags=False)
        .build(_FakeTok())
        .text({"caption": "<text>FELIX</text>"})
        == "<text>FELIX</text>"
    )
    with pytest.raises(ValueError):
        ds.text({"caption": "<text></text>"})
    with pytest.raises(ValueError):
        ds.text({"dense_caption": "x"})
    # The user turn is the bare tag, with no length number.
    from olmo_core.data.multimodal.pixmo_cap import style_tag_prompt

    assert style_tag_prompt("textocr") == "textocr:"


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


def test_example_layout(tmp_path):
    root = _write_tars(tmp_path)
    tok = _FakeTok()
    ds = _cfg(root, tmp_path, loss_token_weighting="none").build(tok)
    ex = ds[0]
    for key in ("input_ids", "labels", "loss_masks", "position_ids", "token_type_ids", "images"):
        assert key in ex
    assert "subsegment_ids" not in ex
    text_ids = ex["input_ids"][ex["token_type_ids"] == 0].tolist()
    prompt_ids = tok.encode("textocr:")
    assert any(text_ids[i : i + len(prompt_ids)] == prompt_ids for i in range(len(text_ids)))
    resp = tok.encode("FELIX PRIVAT DBU 889")  # tags stripped
    assert ex["loss_masks"].sum() == pytest.approx(len(resp) + 1)
    assert ex["labels"][ex["loss_masks"] > 0][-1] == tok.eos_token_id
    np.testing.assert_array_equal(ex["input_ids"], ds[0]["input_ids"])  # deterministic


def test_blank_target_is_skipped_not_raised(tmp_path):
    """An unusable row must not raise out of ``__getitem__``: that spends the mixture loader's
    error budget and a run of them inside one shard aborts training. The next usable row is
    substituted instead."""
    root = _write_tars(tmp_path)
    tok = _FakeTok()
    ds = _cfg(root, tmp_path, loss_token_weighting="none").build(tok)
    assert ds.text({"caption": "<text>x</text>"}) == "x"
    with pytest.raises(ValueError):  # the underlying row is still unusable
        ds._build(3)
    np.testing.assert_array_equal(ds[3]["input_ids"], ds[4]["input_ids"])  # skipped to row 4


def test_unusable_rows_eventually_raise(tmp_path, monkeypatch):
    """Skipping is bounded: an all-broken source must still fail loudly rather than spin."""
    from olmo_core.data.multimodal import sft_common

    root = _write_tars(tmp_path)
    ds = _cfg(root, tmp_path).build(_FakeTok())
    monkeypatch.setattr(ds, "_build", lambda i: (_ for _ in ()).throw(ValueError("nope")))
    with pytest.raises(RuntimeError, match="consecutive rows"):
        ds[0]
    assert sft_common.MAX_ROW_SKIP > 1


def test_message_weight_and_truncation(tmp_path):
    root = _write_tars(tmp_path)
    tok = _FakeTok()
    base = _cfg(root, tmp_path, loss_token_weighting="none").build(tok)[1]
    weighted = _cfg(root, tmp_path, loss_token_weighting="none", message_weight=0.25).build(tok)[1]
    nz = base["loss_masks"] > 0
    np.testing.assert_allclose(weighted["loss_masks"][nz], 0.25 * base["loss_masks"][nz])
    full = _cfg(root, tmp_path).build(tok)[4]
    n_image = int((full["token_type_ids"] == 1).sum())
    assert len(full["input_ids"]) > n_image + 1000
    cut = _cfg(root, tmp_path, max_sequence_length=n_image + 150).build(tok)[4]
    assert len(cut["input_ids"]) == n_image + 150
    assert (cut["token_type_ids"] == 1).sum() == n_image


# ---------------------------------------------------------------------------
# OCR registry + Molmo2-Stage1 wiring
# ---------------------------------------------------------------------------


def test_ocr_registry_shape():
    names = ocr_mix.OCR_SOURCE_NAMES
    assert len(names) == len(set(names)) == 4 + 5 + 6 + 2
    assert set(ocr_mix.OLMOCR_MIX_SOURCES) <= set(names)
    assert set(ocr_mix.TEXT_RICH_SOURCES) == {
        f"text_rich_{c}" for c in ("chart", "diagram", "doc", "graphic", "table")
    }
    assert set(ocr_mix.DUPLICATE_OLMOCR_SOURCES) == {"s2pdf", "iabooks"}
    # Default = train splits only: no re-rendered duplicates, no source of unverifiable split.
    assert set(ocr_mix.SPLIT_UNVERIFIED_SOURCES) == {"hiertext", "cocotext", "ubertext"}
    assert set(ocr_mix.DEFAULT_OCR_SOURCES) == (
        set(ocr_mix.OLMOCR_MIX_SOURCES)
        | set(ocr_mix.TEXT_RICH_SOURCES)
        | {"textocr"}
        | set(ocr_mix.SYNTHETIC_OCR_SOURCES)
    )
    # Every tar source is a transcription source: <text>-wrapped text under one of two styles.
    styles = {src.style for src in ocr_mix.OCR_TAR_SOURCES.values()}
    assert styles == {ocr_mix.OLMOCR_STYLE, ocr_mix.TEXTOCR_STYLE}
    assert all(src.strip_text_tags for src in ocr_mix.OCR_TAR_SOURCES.values())
    with pytest.raises(OLMoConfigurationError):
        ocr_mix.build_ocr_source(
            "nope",
            _FakeTok(),
            olmocr=OlmOcrMixDatasetConfig(),
            tars=OcrCaptionTarsDatasetConfig(),
            text_rich=TextRichCaptionDatasetConfig(),
        )


def test_build_ocr_source_fills_tar_template(tmp_path):
    root = tmp_path / "oe"
    (root / "scene_text_tars").mkdir(parents=True)
    os.rename(_write_tars(tmp_path), str(root / "scene_text_tars" / "cocotext_v6_tars"))
    tars = OcrCaptionTarsDatasetConfig(max_crops=1, index_cache_dir=str(tmp_path / "cache"))
    ds = ocr_mix.build_ocr_source(
        "cocotext",
        _FakeTok(),
        olmocr=OlmOcrMixDatasetConfig(),
        tars=tars,
        text_rich=TextRichCaptionDatasetConfig(),
        data_root=str(root),
    )
    assert ds.config.style == "textocr" and ds.config.strip_text_tags is True
    assert ds.config.dataset_path == str(root / "scene_text_tars" / "cocotext_v6_tars")
    assert ds.config.max_crops == 1 and len(ds) == 5


def _load_stage1_module():
    try:
        import olmo_core.internal.common  # noqa: F401  (needs a recent beaker-py)
    except ImportError as e:  # pragma: no cover - env-dependent
        pytest.skip(f"Molmo2-Stage1.py imports fail here: {e}")
    spec = importlib.util.spec_from_file_location(
        "_stage1_ocr_tars", "src/scripts/train/Molmo2-Stage1.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_ocr_tars"] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


def test_stage1_ocr_group_wiring():
    mod = _load_stage1_module()
    assert mod.OCR_RATE == 0.0
    assert mod.OCR_SOURCES == ocr_mix.DEFAULT_OCR_SOURCES
    fields = {f.name for f in mod.ExperimentConfig.__dataclass_fields__.values()}
    assert {"ocr_rate", "ocr_sources", "olmocr", "ocr_tars", "ocr_data_root"} <= fields


def test_ocr_group_is_general_ocr_only():
    """Images from VQA datasets (the Cambrian subsets) and caption-target sources (TextCaps) are
    left for a later stage; so is the single-caption build the three-level one supersedes."""
    for name in ocr_mix.OCR_SOURCE_NAMES:
        assert not name.startswith("cambrian_") and name != "textcaps", name
    assert not any(src.relpath.startswith("text_rich") for src in ocr_mix.OCR_TAR_SOURCES.values())


def test_ocr_rate_is_split_by_task_before_size():
    names = ["olmocr_documents", "olmocr_books", "text_rich_chart", "textocr"]
    assert [ocr_mix.ocr_task(n) for n in names] == [
        ocr_mix.TRANSCRIPTION,
        ocr_mix.TRANSCRIPTION,
        ocr_mix.FIGURE_CAPTION,
        ocr_mix.TRANSCRIPTION,
    ]
    assert ocr_mix.ocr_task_shares(names) == [0.5, 0.5, 0.5, 0.5]
    # A single task takes the whole rate.
    assert ocr_mix.ocr_task_shares(["olmocr_books", "textocr"]) == [1.0, 1.0]
    assert ocr_mix.ocr_task_shares(["text_rich_doc"]) == [1.0]
    with pytest.raises(OLMoConfigurationError):
        ocr_mix.ocr_task("cambrian_ocr_vqa")


def test_stage1_ocr_fractions_give_each_task_half():
    mod = _load_stage1_module()
    names = ["olmocr_documents", "olmocr_books", "text_rich_chart", "text_rich_doc", "textocr"]
    sizes = [400, 100, 90_000, 10_000, 100]
    frac = mod._ocr_fractions(names, sizes)
    np.testing.assert_allclose(frac.sum(), 1.0)
    np.testing.assert_allclose(frac[[0, 1, 4]].sum(), 0.5)  # transcription
    np.testing.assert_allclose(frac[[2, 3]].sum(), 0.5)  # figure captions, however large
    np.testing.assert_allclose(frac[[0, 1, 4]], 0.5 * np.array([20, 10, 10]) / 40)  # sqrt inside
    np.testing.assert_allclose(frac[[2, 3]], 0.5 * np.array([300, 100]) / 400)
    with pytest.raises(OLMoConfigurationError, match="text_rich_doc"):
        mod._ocr_fractions(names, [400, 100, 90_000, 0, 100])  # an empty source is named


def _data_config(**kw):
    """The data-mixture fields `validate_data_config` reads, without the Beaker launch that the
    full `build_config` resolves (which needs cluster access and a pushed commit)."""
    from types import SimpleNamespace

    from olmo_core.data.multimodal.mixtures.stage1_academic import (
        DEFAULT_ACADEMIC_SOURCES,
    )

    fields = dict(
        recipe="v1",
        pointing_data="v1",
        pointing_rate=0.3,
        nlp_rate=0.1,
        ocr_rate=0.1,
        ocr_sources=ocr_mix.DEFAULT_OCR_SOURCES,
        olmocr=OlmOcrMixDatasetConfig(),
        ocr_tars=OcrCaptionTarsDatasetConfig(),
        text_rich=TextRichCaptionDatasetConfig(),
        academic_rate=0.0,
        academic_sources=DEFAULT_ACADEMIC_SOURCES,
        clock_rate=0.0,
    )
    fields.update(kw)
    return SimpleNamespace(**fields)


def test_stage1_refuses_per_source_ocr_tar_overrides():
    """One template serves every tar source, so ``build_ocr_source`` overwrites these three.
    Accepting them would record the value in the saved run config and then ignore it."""
    mod = _load_stage1_module()
    overrides: List[Dict[str, Any]] = [
        {"dataset_path": "/somewhere/else"},
        {"style": "long_caption"},
        {"strip_text_tags": False},
    ]
    for override in overrides:
        with pytest.raises(OLMoConfigurationError, match="per OCR source"):
            mod.validate_data_config(_data_config(ocr_tars=OcrCaptionTarsDatasetConfig(**override)))
    # The same holds for the other two templates' per-source fields.
    kws: List[Dict[str, Any]] = [
        {"olmocr": OlmOcrMixDatasetConfig(subset="books")},
        {"text_rich": TextRichCaptionDatasetConfig(category="doc")},
    ]
    for kw in kws:
        with pytest.raises(OLMoConfigurationError, match="per OCR source"):
            mod.validate_data_config(_data_config(**kw))
    mod.validate_data_config(_data_config())  # the untouched template passes
    # The supported way to relocate the tree is a top-level field, and it reaches the sources.
    assert "ocr_data_root" in mod.ExperimentConfig.__dataclass_fields__
    tars = ocr_mix.OCR_TAR_SOURCES["cocotext"]
    assert ocr_mix.os.path.join("/my/tars", tars.relpath).startswith("/my/tars")


def test_stage1_warns_when_a_source_of_unverified_split_is_selected(caplog):
    mod = _load_stage1_module()
    with caplog.at_level("WARNING"):
        mod.validate_data_config(_data_config(ocr_sources=("olmocr_books", "cocotext")))
    assert "cocotext" in caplog.text and "training split" in caplog.text


# ---------------------------------------------------------------------------
# Data recipes
# ---------------------------------------------------------------------------


def test_stage1_recipes():
    """v1 is the released pretrain mixture; v2 is caption 0.5, v2 pointing and counting 0.25,
    OCR 0.25 and no text-only data."""
    mod = _load_stage1_module()
    v1, v2 = mod.RECIPES["v1"], mod.RECIPES["v2"]
    assert mod.RECIPE == "v1"
    assert v1 == dict(
        pointing_rate=0.3,
        nlp_rate=0.1,
        ocr_rate=0.0,
        academic_rate=0.0,
        clock_rate=0.0,
        pointing_data="v1",
    )
    assert v2 == dict(
        pointing_rate=0.25,
        nlp_rate=0.0,
        ocr_rate=0.25,
        academic_rate=0.0,
        clock_rate=0.0,
        pointing_data="v2",
    )
    groups = ("pointing_rate", "nlp_rate", "ocr_rate", "academic_rate", "clock_rate")
    for r in (v1, v2):
        caption = 1.0 - sum(r[k] for k in groups)
        assert caption == pytest.approx(0.6 if r is v1 else 0.5)
    # The ExperimentConfig defaults are the v1 recipe, so a config built without `--recipe`
    # (and the dataclass defaults a test builds) agree with it.
    fields = mod.ExperimentConfig.__dataclass_fields__
    assert {k: fields[k].default for k in v1} == v1
    assert fields["recipe"].default == "v1"


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ([], "v1"),
        (["--recipe=v2"], "v2"),
        (["--recipe=V2"], "v2"),
        (["--recipe=v3"], "v3"),
        (["--recipe=v2", "--recipe=v1"], "v1"),  # later wins, as in `merge`
        (["--ocr_rate=0.1"], "v1"),
    ],
)
def test_resolve_recipe(overrides, expected):
    mod = _load_stage1_module()
    name, fields = mod.resolve_recipe(overrides)
    assert name == expected
    assert fields == mod.RECIPES[expected]
    fields["ocr_rate"] = 0.9  # a copy: the registry itself must not change
    assert mod.RECIPES[expected]["ocr_rate"] != 0.9


def test_unknown_recipe_is_refused():
    mod = _load_stage1_module()
    with pytest.raises(OLMoConfigurationError, match="recipe"):
        mod.resolve_recipe(["--recipe=v9"])
    with pytest.raises(OLMoConfigurationError, match="recipe"):
        mod.validate_data_config(_data_config(recipe="v9"))


def test_v2_recipe_validates_and_splits_its_groups():
    """The v2 recipe's rates pass validation, and with the real source sizes the caption, pointing
    and OCR groups get 0.5 / 0.25 / 0.25 of the mixture."""
    mod = _load_stage1_module()
    mod.validate_data_config(_data_config(recipe="v2", **mod.RECIPES["v2"]))
    pointing = mod._pointing_group_fractions([220_290, 36_901, 68_051], "v2")
    ocr_names = list(ocr_mix.DEFAULT_OCR_SOURCES)
    sizes = {
        # olmOCR-mix sizes after the default English filter, as a run sees them.
        "olmocr_documents": 218_831,
        "olmocr_books": 15_134,
        "olmocr_loc_transcripts": 9_712,
        "olmocr_national_archives": 9_810,
        "text_rich_chart": 353_439,
        "text_rich_diagram": 142_534,
        "text_rich_doc": 438_267,
        "text_rich_graphic": 87_212,
        "text_rich_table": 315_515,
        "textocr": 21_749,
        "nvidia_synth_en": 1_460_304,
        "synth_receipts_en": 16_437,
    }
    ocr = mod._ocr_fractions(ocr_names, [sizes[n] for n in ocr_names])
    r = mod.RECIPES["v2"]
    assert r["pointing_rate"] * pointing.sum() == pytest.approx(0.25)
    assert r["ocr_rate"] * ocr.sum() == pytest.approx(0.25)
    # The NVIDIA synthetic set is weighted as if no larger than olmOCR-mix documents.
    share = dict(zip(ocr_names, r["ocr_rate"] * ocr))
    assert share["nvidia_synth_en"] == pytest.approx(share["olmocr_documents"])
    assert share["nvidia_synth_en"] == pytest.approx(0.0382, abs=1e-4)
