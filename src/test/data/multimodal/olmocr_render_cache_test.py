"""CPU tests for the olmOCR-mix persistent render cache: entry keys, the hit path returning
exactly the live render (lossless formats), the fallbacks (missing or corrupt entry, read-only
cache), examples unchanged by the cache, and the mixture loader's order being the same with and
without the cache and with and without prefetch threads.

The real renderer is used wherever pixels matter; those tests skip without ``pypdfium2``."""

import copy
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from olmo_core.data.multimodal.collator import MultimodalCollator
from olmo_core.data.multimodal.mixture_data_loader import MixtureDataLoader
from olmo_core.data.multimodal.olmocr import (
    RENDER_CACHE_FORMATS,
    OlmOcrMixDatasetConfig,
    RenderCache,
    render_pdf_page,
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


ROWS = [
    ("en-a", "en", "Page one.\n\nA short transcription."),
    ("en-b", "en", "Second page, with a little more text on it."),
    ("fr", "fr", "Une page en français."),
    ("en-blank", "en", None),
    ("en-c", "en", "word " * 50),
]


def _write_root(tmp_path, name="olmocr_mix"):
    """A one-subset olmOCR-mix root whose single-page PDFs hold textured images, so a lossy
    re-encoding is visibly different from the render."""
    from PIL import Image

    root = tmp_path / name
    chunk = "00_documents_train_00000"
    (root / "pdfs" / chunk / "0000").mkdir(parents=True)
    rows = []
    rng = np.random.RandomState(0)
    for i, (rid, lang, text) in enumerate(ROWS):
        arc = f"0000/{rid}-1.pdf"
        pixels = rng.randint(0, 256, size=(160, 120 + 10 * i, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(str(root / "pdfs" / chunk / arc), "PDF")
        rows.append(
            {
                "id": rid,
                "url": f"https://example/{rid}",
                "page_number": i + 1,
                "pdf_relpath": f"pdf_tarballs/{chunk}.tar.gz:{arc}",
                "primary_language": lang,
                "natural_text": text,
            }
        )
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(root / "00_documents_train.parquet"))
    pq.write_table(table.slice(0, 2), str(root / "00_documents_eval.parquet"))
    return str(root)


def _cfg(root, **kw):
    kw.setdefault("max_crops", 1)
    kw.setdefault("target_longest_image_dim_range", (300, 420))
    return OlmOcrMixDatasetConfig(dataset_path=root, **kw)


# ---------------------------------------------------------------------------
# Config and keys
# ---------------------------------------------------------------------------


def test_config_validation_and_cli_merge():
    OlmOcrMixDatasetConfig().validate()  # no cache by default
    with pytest.raises(OLMoConfigurationError, match="render_cache_format"):
        OlmOcrMixDatasetConfig(render_cache_dir="/c", render_cache_format="tiff").validate()
    with pytest.raises(OLMoConfigurationError):
        RenderCache("/c", image_format="bmp")
    cfg = OlmOcrMixDatasetConfig().merge(
        ["render_cache_dir=/c/olmocr", "render_cache_format=png", "render_cache_write=false"]
    )
    assert cfg.render_cache_dir == "/c/olmocr"
    assert cfg.render_cache_format == "png"
    assert cfg.render_cache_write is False
    assert set(RENDER_CACHE_FORMATS) == {"webp", "png", "jpeg"}


def test_cache_paths_mirror_the_pdf_tree(tmp_path):
    root = str(tmp_path / "mix")
    cache = RenderCache(str(tmp_path / "cache"), root=root)
    pdf = os.path.join(root, "pdfs", "00_documents_train_00000", "0000", "a:b-1.pdf")
    assert cache.path_for(pdf, 0, 1536) == str(
        tmp_path
        / "cache"
        / "pdfs"
        / "00_documents_train_00000"
        / "0000"
        / "a:b-1.pdf.p0.d1536.webp"
    )
    # The key carries the page and the render size, and the extension follows the format.
    assert cache.path_for(pdf, 0, 1537) != cache.path_for(pdf, 0, 1536)
    assert cache.path_for(pdf, 1, 1536) != cache.path_for(pdf, 0, 1536)
    assert (
        RenderCache("/c", image_format="png", root=root)
        .path_for(pdf, 0, 1024)
        .endswith("a:b-1.pdf.p0.d1024.png")
    )
    assert RenderCache("/c", image_format="jpeg", root=root).path_for(pdf, 0, 1024).endswith(".jpg")
    # A PDF outside the root is keyed by its absolute path.
    outside = RenderCache("/c", root=root).path_for("/elsewhere/x.pdf", 0, 640)
    assert outside == "/c/elsewhere/x.pdf.p0.d640.webp"
    assert RenderCache("/c").path_for(pdf, 0, 640) == "/c" + pdf + ".p0.d640.webp"


# ---------------------------------------------------------------------------
# Hit path == live render; fallbacks
# ---------------------------------------------------------------------------


def _pdf_paths(root):
    ds = _cfg(root, languages=None).build(_FakeTok())
    return [ds.pdf_path(ds._data[int(i)]) for i in ds._index]


@pytest.mark.parametrize("image_format", ["webp", "png"])
def test_cached_page_equals_live_render(tmp_path, image_format):
    pytest.importorskip("pypdfium2")
    root = _write_root(tmp_path)
    cache = RenderCache(str(tmp_path / "cache"), image_format=image_format, root=root)
    pdfs = _pdf_paths(root)
    for pdf in pdfs[:3]:
        for dim in (256, 333, 640):
            live = render_pdf_page(pdf, dim)
            first = render_pdf_page(pdf, dim, cache=cache)  # miss: render + store
            assert os.path.exists(cache.path_for(pdf, 0, dim))
            second = render_pdf_page(pdf, dim, cache=cache)  # hit: decode
            for image in (first, second):
                assert image.mode == "RGB" and image.size == live.size
                assert np.array_equal(np.asarray(image), np.asarray(live))
    stats = cache.stats()
    assert (stats["hits"], stats["misses"]) == (9, 9)
    assert stats["hit_mean_s"] > 0 and stats["render_mean_s"] > 0


def test_jpeg_entries_are_lossy_but_close(tmp_path):
    pytest.importorskip("pypdfium2")
    root = _write_root(tmp_path)
    cache = RenderCache(str(tmp_path / "cache"), image_format="jpeg", root=root)
    assert not cache.lossless
    pdf = _pdf_paths(root)[0]
    live = np.asarray(render_pdf_page(pdf, 300)).astype(int)
    render_pdf_page(pdf, 300, cache=cache)
    hit = np.asarray(render_pdf_page(pdf, 300, cache=cache)).astype(int)
    assert hit.shape == live.shape
    assert not np.array_equal(hit, live)
    assert np.abs(hit - live).mean() < 8


def test_missing_corrupt_and_unwritable_entries_fall_back_to_rendering(tmp_path, caplog):
    pytest.importorskip("pypdfium2")
    root = _write_root(tmp_path)
    pdf = _pdf_paths(root)[1]
    live = np.asarray(render_pdf_page(pdf, 320))

    # A read-only cache never gets an entry, and every call renders.
    ro = RenderCache(str(tmp_path / "ro"), root=root, write=False)
    for _ in range(2):
        assert np.array_equal(np.asarray(render_pdf_page(pdf, 320, cache=ro)), live)
    assert not os.path.exists(ro.path_for(pdf, 0, 320))
    assert ro.stats()["misses"] == 2 and ro.stats()["hits"] == 0

    # A corrupt entry is ignored (one warning), the page is rendered and the entry rewritten.
    rw = RenderCache(str(tmp_path / "rw"), root=root)
    path = rw.path_for(pdf, 0, 320)
    os.makedirs(os.path.dirname(path))
    with open(path, "wb") as f:
        f.write(b"not an image")
    with caplog.at_level("WARNING"):
        assert np.array_equal(np.asarray(render_pdf_page(pdf, 320, cache=rw)), live)
    assert any("cannot read" in r.message for r in caplog.records)
    assert np.array_equal(np.asarray(render_pdf_page(pdf, 320, cache=rw)), live)
    assert rw.stats()["hits"] == 1

    # A cache directory that cannot be created warns once and keeps rendering.
    blocked = tmp_path / "blocked"
    blocked.write_text("a file, not a directory")
    bad = RenderCache(str(blocked), root=root)
    caplog.clear()
    with caplog.at_level("WARNING"):
        for _ in range(2):
            assert np.array_equal(np.asarray(render_pdf_page(pdf, 320, cache=bad)), live)
    assert sum("cannot write" in r.message for r in caplog.records) == 1


# ---------------------------------------------------------------------------
# Dataset examples and loader order are unchanged by the cache
# ---------------------------------------------------------------------------


def _assert_equal(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, np.ndarray):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_equal(left, right)
    else:
        assert actual == expected


def test_dataset_examples_identical_with_and_without_cache(tmp_path):
    pytest.importorskip("pypdfium2")
    root = _write_root(tmp_path)
    tok = _FakeTok()
    plain = _cfg(root).build(tok)
    cached = _cfg(root, render_cache_dir=str(tmp_path / "cache")).build(tok)
    assert plain.render_cache is None and cached.render_cache is not None
    for epoch in (0, 1):
        for i in range(len(plain)):
            expected = plain.get(i, epoch)
            _assert_equal(cached.get(i, epoch), expected)  # miss
            _assert_equal(cached.get(i, epoch), expected)  # hit
    stats = cached.render_cache.stats()
    assert stats["misses"] == 2 * len(plain) and stats["hits"] == 2 * len(plain)


def _loader(root, tmp_path, *, workers, cache_dir=None, write=True):
    tok = _FakeTok()
    kw = {}
    if cache_dir is not None:
        kw = {"render_cache_dir": cache_dir, "render_cache_write": write}
    datasets = [
        _cfg(root, split="train", **kw).build(tok),
        _cfg(root, split="eval", seed=3, **kw).build(tok),
    ]
    return MixtureDataLoader(
        datasets,
        [0.7, 0.3],
        MultimodalCollator(pad_token_id=0, pad_sequence_length=1024),
        work_dir=str(tmp_path / "work"),
        global_batch_size=2 * 1024,
        seed=11,
        epoch_instances=400,
        pack=True,
        pack_max_crops=4,
        pack_buffer_size=4,
        continuous_stream=True,
        prefetch_workers=workers,
        prefetch_max_in_flight=16 if workers else None,
        dataset_names=["train", "eval"],
    )


def _prewarm(root, cache_dir, epochs=48):
    """Render every page of both splits at the sizes the training path draws for the first
    ``epochs`` source epochs, so a loader over the cache hits whatever its read-ahead touches."""
    tok = _FakeTok()
    cache = RenderCache(cache_dir, root=root)
    for ds in (_cfg(root, split="train").build(tok), _cfg(root, split="eval", seed=3).build(tok)):
        for i in range(len(ds)):
            pdf = ds.pdf_path(ds._data[int(ds._index[i])])
            for dim in {ds.target_dim_for(ds.epoch_rng(i, epoch)) for epoch in range(epochs)}:
                render_pdf_page(pdf, dim, cache=cache)


def test_mixture_order_matches_synchronous_loading_with_and_without_cache(tmp_path):
    """The batches (and the loader's resume state) are the same whether examples are built on
    the calling thread or on a prefetch pool, and whether the pages come from the renderer, a
    cold cache (misses that are rendered and stored) or a warm cache (every page a hit)."""
    pytest.importorskip("pypdfium2")
    root = _write_root(tmp_path)
    cold_dir, warm_dir = str(tmp_path / "cold"), str(tmp_path / "warm")
    _prewarm(root, warm_dir)
    expected = None
    expected_state = None
    for label, workers, cache_dir in (
        ("sync", 0, None),
        ("threads", 8, None),
        ("threads-cold-cache", 8, cold_dir),
        ("threads-warm-cache", 8, warm_dir),
        ("sync-warm-cache", 0, warm_dir),
    ):
        loader = _loader(root, tmp_path, workers=workers, cache_dir=cache_dir)
        loader.reshuffle(epoch=1)
        iterator = iter(loader)
        try:
            batches = [next(iterator) for _ in range(6)]
            state = copy.deepcopy(loader.state_dict())
        finally:
            iterator.close()
        if expected is None:
            expected, expected_state = batches, state
        else:
            _assert_equal(batches, expected)
            _assert_equal(state, expected_state)
        if cache_dir is not None:
            stats = {
                name: ds.render_cache.stats()
                for name, ds in zip(loader.dataset_names, loader.datasets)
            }
            # The pool's read-ahead renders a few refs past the consumed ones, so a cold cache
            # only has to show misses; the pre-warmed one must never render.
            if label == "threads-cold-cache":
                assert all(s["misses"] > 0 for s in stats.values()), stats
            else:
                assert all(s["misses"] == 0 and s["hits"] > 0 for s in stats.values()), (
                    label,
                    stats,
                )
