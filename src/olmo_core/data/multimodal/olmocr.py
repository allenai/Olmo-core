"""olmOCR-mix page transcription for Molmo2 stage-1.

Port of mm_olmo's ``OlmOcrMixConfig`` (``olmo/data/olmocr_datasets.py``), one of the two OCR
groups in its molmo3 stage-1 mixture (``launch_scripts/train_molmo3_stage1.py``,
``_base_mixture``). ``allenai/olmOCR-mix-1025`` holds, per subset (``documents``, ``books``,
``loc_transcripts``, ``national_archives``) and split (``train`` / ``eval``), one parquet of page
records plus the source PDFs. mm_olmo's ``download`` fetches those and expands every tarball
into ``pdfs/<chunk>/<arcname>``; this module reads that layout -- already materialised on weka
at :data:`~olmo_core.data.multimodal.paths.OLMOCR_MIX` -- and does not download.

Each example is one rendered page and its ``natural_text`` transcription. There is no question:
the user turn is just the style tag, the bare ``"olmocr:"`` (mm_olmo's name; its formatter has no
template for this style), and the assistant turn is the transcription
(``"No text found"`` for blank pages). Pages are rasterised on the fly with ``pypdfium2`` at a
longest side sampled from ``target_longest_image_dim_range`` for training (mm_olmo: 1024-2048)
and fixed (1536) otherwise, following olmOCR's own per-page DPI rule.

Transcriptions run long (documents pages: median ~580 tokens, p99 ~2900 with the Molmo2
tokenizer), so ``max_sequence_length`` should be set to the training sequence length; the
sequence is then tail-truncated like mm_olmo's preprocessor does.

Rendering is the slow part of an example (tens of ms for born-digital ``documents`` pages, up to
seconds for scanned ``national_archives`` pages), and it serialises behind a lock. An optional
persistent :class:`RenderCache` (``render_cache_dir``) stores each rendered page losslessly,
keyed by (PDF path, page, longest side), so a pre-rendered page (see
``src/scripts/prerender_olmocr_pages.py``) is decoded instead of rasterised; a page missing from
the cache is rendered as before.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds

from .message_sequence import encode_sft_example
from .paths import OLMOCR_MIX
from .pixmo_cap import style_tag_prompt
from .sft_common import (
    EpochSeededExamples,
    SftMessageFormat,
    get_example_with_skip,
    load_hf_dataset,
    truncate_for_format,
)

__all__ = [
    "OLMOCR_STYLE",
    "OLMOCR_SUBSETS",
    "OLMOCR_SPLITS",
    "OlmOcrMixDatasetConfig",
    "OlmOcrMixDataset",
    "RENDER_CACHE_FORMATS",
    "RenderCache",
    "canonical_subset",
    "canonical_split",
    "render_pdf_page",
]

log = logging.getLogger(__name__)

#: Style of page transcription in olmOCR's output form (reading order, HTML tables, ``\( \)`` /
#: ``\[ \]`` math); the user turn is this tag alone, ``"olmocr:"``, mm_olmo's name for it.
OLMOCR_STYLE = "olmocr"

#: Hub config names. The numeric prefix is part of the parquet / tarball filenames.
OLMOCR_SUBSETS: Tuple[str, ...] = (
    "00_documents",
    "01_books",
    "02_loc_transcripts",
    "03_national_archives",
)
OLMOCR_SPLITS: Tuple[str, ...] = ("train", "eval")

#: Columns the dataset reads; everything else in the parquet is provenance / flags.
_COLUMNS = ["id", "url", "page_number", "pdf_relpath", "primary_language", "natural_text"]


def canonical_subset(subset: str) -> str:
    """Accept either the hub name (``00_documents``) or the bare one (``documents``)."""
    if subset in OLMOCR_SUBSETS:
        return subset
    matches = [s for s in OLMOCR_SUBSETS if s.split("_", 1)[1] == subset]
    if not matches:
        bare = [s.split("_", 1)[1] for s in OLMOCR_SUBSETS]
        raise OLMoConfigurationError(
            f"Unknown olmOCR-mix subset {subset!r}, expected one of {bare} "
            f"(or a full name from {list(OLMOCR_SUBSETS)})"
        )
    return matches[0]


def canonical_split(split: str) -> str:
    """The hub calls the held-out split ``eval``; accept the repo's usual ``validation`` too."""
    if split == "validation":
        return "eval"
    if split not in OLMOCR_SPLITS:
        raise OLMoConfigurationError(
            f"Unknown olmOCR-mix split {split!r}, expected one of {list(OLMOCR_SPLITS)} "
            "or 'validation'"
        )
    return split


_TARBALL_SEP = ".tar.gz:"


def pdf_path_for(root: str, pdf_relpath: str) -> str:
    """Resolve a row's ``pdf_relpath`` to the expanded PDF.

    ``pdf_tarballs/00_documents_train_00000.tar.gz:0000/abc-1.pdf`` ->
    ``<root>/pdfs/00_documents_train_00000/0000/abc-1.pdf``. The separator is the colon right
    after the tarball name, so the split is on ``".tar.gz:"``: an arcname is a path and may
    itself contain a colon (mm_olmo's ``rsplit(":")`` would then cut inside it).
    """
    if _TARBALL_SEP not in pdf_relpath:
        raise ValueError(f"pdf_relpath {pdf_relpath!r} has no '{_TARBALL_SEP}' separator")
    tar_part, arcname = pdf_relpath.split(_TARBALL_SEP, 1)
    return os.path.join(root, "pdfs", os.path.basename(tar_part), arcname)


_PDFIUM_LOCK = threading.Lock()

#: Image formats a :class:`RenderCache` can store pages in. ``webp`` and ``png`` are lossless
#: (the decoded page is pixel-identical to the live render); ``jpeg`` (quality 95) is not.
RENDER_CACHE_FORMATS: Tuple[str, ...] = ("webp", "png", "jpeg")

# Encoder settings per format, chosen on 1536px olmOCR-mix pages: lossless WebP at effort 1 is
# about half the bytes of PNG (270 KB vs 490 KB for a documents page, 1.1 MB vs 1.7 MB for a
# national-archives scan) and decodes as fast (14 / 32 ms); PNG at level 1 encodes fastest.
_RENDER_CACHE_ENCODERS: Dict[str, Dict[str, Any]] = {
    "webp": {"format": "WEBP", "lossless": True, "quality": 0, "method": 1},
    "png": {"format": "PNG", "compress_level": 1},
    "jpeg": {"format": "JPEG", "quality": 95},
}
_RENDER_CACHE_EXTENSIONS: Dict[str, str] = {"webp": "webp", "png": "png", "jpeg": "jpg"}


class RenderCache:
    """A persistent store of rendered PDF pages, keyed by (PDF path, page, longest side).

    A cached page is decoded instead of rasterised, which is both faster (the WebP decode of a
    page takes 15-30 ms; the render 20 ms to 2 s) and lock-free, so a slow page no longer stalls
    the loader's whole read-ahead window. Entries are written atomically (temporary file, then
    ``os.replace``), so a reader never sees a partial file and concurrent writers of one key
    just overwrite each other with identical content. Every failure on the cache path -- an
    unreadable or corrupt entry, a directory that cannot be written -- falls back to the live
    render and warns once; the cache changes timing only, never the example.

    :param cache_dir: Directory holding the cache tree.
    :param image_format: One of :data:`RENDER_CACHE_FORMATS`.
    :param root: Directory the PDF paths are keyed relative to (``dataset_path``), so the cache
        tree mirrors ``pdfs/<chunk>/<arcname>``; a PDF outside it is keyed by its absolute path.
    :param write: Whether a page rendered on a cache miss is stored.
    """

    def __init__(
        self,
        cache_dir: str,
        image_format: str = "webp",
        root: Optional[str] = None,
        write: bool = True,
    ):
        if image_format not in RENDER_CACHE_FORMATS:
            raise OLMoConfigurationError(
                f"render cache format must be one of {RENDER_CACHE_FORMATS}, got {image_format!r}"
            )
        self.cache_dir = cache_dir
        self.image_format = image_format
        self.root = root
        self.write = write
        self.hits = 0
        self.misses = 0
        self.hit_seconds = 0.0
        self.render_seconds = 0.0
        self.store_seconds = 0.0
        self._stats_lock = threading.Lock()
        self._warned_load = False
        self._warned_store = False

    @property
    def lossless(self) -> bool:
        """Whether a cached page decodes to exactly the pixels of the live render."""
        return self.image_format != "jpeg"

    def path_for(self, pdf_path: str, page: int, target_longest_image_dim: int) -> str:
        """The cache entry of ``page`` of ``pdf_path`` rendered at ``target_longest_image_dim``."""
        key = os.path.abspath(pdf_path)
        if self.root is not None:
            rel = os.path.relpath(key, os.path.abspath(self.root))
            if not rel.startswith(os.pardir):
                key = rel
        key = key.lstrip(os.sep)
        ext = _RENDER_CACHE_EXTENSIONS[self.image_format]
        return os.path.join(self.cache_dir, f"{key}.p{page}.d{target_longest_image_dim}.{ext}")

    def load(self, path: str) -> Optional["PILImage"]:
        """The cached RGB image at ``path``, or ``None`` when there is no usable entry."""
        from PIL import Image

        if not os.path.exists(path):
            return None
        try:
            with Image.open(path) as image:
                return image.convert("RGB")
        except Exception as e:  # a truncated or corrupt entry: render instead
            if not self._warned_load:
                self._warned_load = True
                log.warning("olmOCR render cache: cannot read %s (%s); rendering instead", path, e)
            return None

    def store(self, path: str, image: "PILImage") -> bool:
        """Write ``image`` to ``path`` atomically. Returns whether it was written."""
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            image.save(tmp, **_RENDER_CACHE_ENCODERS[self.image_format])
            os.replace(tmp, path)
            return True
        except Exception as e:
            if not self._warned_store:
                self._warned_store = True
                log.warning("olmOCR render cache: cannot write %s (%s)", path, e)
            try:
                os.remove(tmp)
            except OSError:
                pass
            return False

    def record(self, hit: bool, seconds: float, store_seconds: float = 0.0) -> None:
        """Account one page: a hit (decode time) or a miss (render time, plus the store)."""
        with self._stats_lock:
            if hit:
                self.hits += 1
                self.hit_seconds += seconds
            else:
                self.misses += 1
                self.render_seconds += seconds
                self.store_seconds += store_seconds

    def stats(self) -> Dict[str, float]:
        """Counts and mean seconds per page of the hit and miss paths so far."""
        with self._stats_lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_mean_s": self.hit_seconds / max(self.hits, 1),
                "render_mean_s": self.render_seconds / max(self.misses, 1),
                "store_mean_s": self.store_seconds / max(self.misses, 1),
            }


def _render_pdf_page(pdf_path: str, target_longest_image_dim: int) -> "PILImage":
    try:
        import pypdfium2
    except ImportError as e:
        raise ImportError(
            "olmOCR-mix stores source PDFs, so rendering a page needs pypdfium2 "
            "(`pip install pypdfium2`)."
        ) from e

    with _PDFIUM_LOCK:
        pdf = pypdfium2.PdfDocument(pdf_path)
        try:
            if len(pdf) != 1:
                raise RuntimeError(
                    f"{pdf_path}: expected a pre-split single-page PDF, got {len(pdf)}"
                )
            page = pdf[0]
            longest_dim = max(page.get_size())  # (width, height) in PDF points
            scale = target_longest_image_dim / longest_dim
            return page.render(scale=scale).to_pil().convert("RGB")
        finally:
            pdf.close()


def render_pdf_page(
    pdf_path: str, target_longest_image_dim: int, cache: Optional[RenderCache] = None
) -> "PILImage":
    """Render a single-page PDF to an RGB PIL image whose longest side is
    ``target_longest_image_dim`` pixels.

    Follows olmOCR's convention (``render_pdf_to_base64png``): the DPI is chosen per page as
    ``target * 72 / longest_mediabox_dim_in_points``, i.e. a render ``scale`` of
    ``target / longest_dim`` -- page sizes vary, so no fixed DPI. olmOCR rasterises with poppler,
    so fonts / antialiasing can differ slightly from pypdfium2's output.

    The shipped files are single-page extracts (the row's ``page_number`` is provenance in the
    original document), so only page 0 exists. pypdfium2 is not thread-safe; the render is
    serialised behind a lock because :class:`~.mixture_data_loader.MixtureDataLoader` prefetches
    examples on a thread pool.

    :param cache: With a :class:`RenderCache`, a cached entry for (``pdf_path``, page 0,
        ``target_longest_image_dim``) is decoded instead (no lock); otherwise the page is
        rendered and, if the cache writes, stored for next time.

    :raises ImportError: If ``pypdfium2`` is not installed (``pip install pypdfium2``).
    :raises RuntimeError: If the PDF has more than one page.
    """
    if cache is None:
        return _render_pdf_page(pdf_path, target_longest_image_dim)
    path = cache.path_for(pdf_path, 0, target_longest_image_dim)
    t0 = time.perf_counter()
    image = cache.load(path)
    if image is not None:
        cache.record(True, time.perf_counter() - t0)
        return image
    t0 = time.perf_counter()
    image = _render_pdf_page(pdf_path, target_longest_image_dim)
    rendered = time.perf_counter() - t0
    if cache.write:
        cache.store(path, image)
    cache.record(False, rendered, time.perf_counter() - t0 - rendered)
    return image


@dataclass
class OlmOcrMixDatasetConfig(Config):
    """One subset of olmOCR-mix-1025: transcribe a rendered PDF page (mm_olmo
    ``OlmOcrMixConfig``). Field names follow mm_olmo's."""

    subset: str = "documents"
    """``documents`` / ``books`` / ``loc_transcripts`` / ``national_archives`` (hub names with the
    numeric prefix are accepted too)."""

    split: str = "train"
    """``train`` or ``eval`` (``validation`` is an alias)."""

    dataset_path: str = OLMOCR_MIX
    """Root holding ``<subset>_<split>.parquet`` and the expanded ``pdfs/`` tree."""

    target_longest_image_dim_range: Optional[Tuple[int, int]] = (1024, 2048)
    """Training-time render size: the longest side is drawn uniformly from this inclusive range
    per example. ``None`` always renders at ``target_longest_image_dim``."""

    target_longest_image_dim: int = 1536
    """Render size (longest side) for the eval split, or for training when no range is set."""

    languages: Optional[Tuple[str, ...]] = ("en",)
    """Keep only rows whose ``primary_language`` (ISO 639-1; English is ~94% of the corpus) is
    listed. ``None`` keeps every language."""

    max_crops: int = 8
    max_sequence_length: Optional[int] = None
    """Tail-truncate the built sequence to this many tokens (the image block always fits; an
    example left without loss tokens is rejected, and the loader skips it). Set it to the
    training sequence length: long pages otherwise overflow it."""

    loss_token_weighting: str = "none"
    """``"none"`` weights every response token equally, like the stage-1 caption source."""
    message_weight: Optional[float] = None
    """Scalar loss multiplier for this source (mm_olmo's ``ocr_weight``)."""
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)
    """Image token IDs of the selected language-model tokenizer (set by the mixture)."""
    message_format: SftMessageFormat = "qwen3"
    """``"qwen3"`` (the released Molmo2 chat layout) or ``"document"`` (plain pretraining
    documents, for a language model trained without a chat template)."""

    seed: int = 0

    render_cache_dir: Optional[str] = None
    """Directory of a persistent :class:`RenderCache` of rendered pages. A page already in it
    (pre-rendered with ``src/scripts/prerender_olmocr_pages.py``, or stored by an earlier miss)
    is decoded instead of rasterised; a missing page is rendered as without a cache. ``None``
    (the default) always renders. A lossless format keeps the examples identical, so a cache may
    be switched on or off across a resume."""
    render_cache_format: str = "webp"
    """Entry format, one of :data:`RENDER_CACHE_FORMATS`; ``webp`` and ``png`` are lossless."""
    render_cache_write: bool = True
    """Whether a page rendered on a cache miss is stored (encoding a page costs 50-300 ms on the
    loader thread). The training-time render size is drawn per (page, source epoch), so a miss
    stored during training is reused only when that draw recurs (a replay of the same epoch)."""

    def validate(self):
        canonical_subset(self.subset)
        canonical_split(self.split)
        if self.render_cache_format not in RENDER_CACHE_FORMATS:
            raise OLMoConfigurationError(
                f"render_cache_format must be one of {RENDER_CACHE_FORMATS}, "
                f"got {self.render_cache_format!r}"
            )
        if self.target_longest_image_dim_range is not None:
            lo, hi = self.target_longest_image_dim_range
            if lo <= 0 or hi < lo:
                raise OLMoConfigurationError(
                    "target_longest_image_dim_range must be a positive (lo, hi) with lo <= hi, "
                    f"got {(lo, hi)}"
                )
        if self.target_longest_image_dim <= 0:
            raise OLMoConfigurationError("target_longest_image_dim must be positive")
        if self.languages is not None and len(self.languages) == 0:
            raise OLMoConfigurationError(
                "languages=() would filter out every row; use None to keep all languages"
            )

    def build(self, tokenizer) -> "OlmOcrMixDataset":
        self.validate()
        return OlmOcrMixDataset(self, tokenizer)


class OlmOcrMixDataset(EpochSeededExamples):
    """Map-style dataset over the (language-filtered) pages of one olmOCR-mix subset."""

    def __init__(self, config: OlmOcrMixDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.subset = canonical_subset(config.subset)
        self.split = canonical_split(config.split)
        self.parquet_path = os.path.join(config.dataset_path, f"{self.subset}_{self.split}.parquet")
        if not os.path.exists(self.parquet_path):
            raise FileNotFoundError(
                f"{self.parquet_path} not found; materialise olmOCR-mix with mm_olmo's "
                f"`OlmOcrMixConfig.download(subsets=[{config.subset!r}])` or point "
                "`dataset_path` at it"
            )
        # The parquet is one file, so `split="train"` here is just `load_dataset`'s name for it.
        self._data = load_hf_dataset(self.parquet_path, split="train", keep_columns=_COLUMNS)
        self._index = self._build_index()
        self._warned = 0
        self.render_cache: Optional[RenderCache] = None
        if config.render_cache_dir is not None:
            self.render_cache = RenderCache(
                config.render_cache_dir,
                image_format=config.render_cache_format,
                root=config.dataset_path,
                write=config.render_cache_write,
            )
        log.info(
            "olmOCR-mix %s/%s: %d of %d pages kept (languages=%s)",
            self.subset,
            self.split,
            len(self._index),
            len(self._data),
            config.languages,
        )

    def _build_index(self) -> np.ndarray:
        """Rows passing the ``languages`` filter (mm_olmo's build-time ``ds.filter``), computed
        on the Arrow column so no cache file is written."""
        if self.config.languages is None:
            return np.arange(len(self._data))
        import pyarrow as pa
        import pyarrow.compute as pc

        keep = pc.is_in(
            self._data.data.column("primary_language"),
            value_set=pa.array(list(self.config.languages)),
        )
        mask = pc.fill_null(keep, False).to_numpy(zero_copy_only=False).astype(bool)
        return np.flatnonzero(mask)

    def __len__(self) -> int:
        return len(self._index)

    # -- per-example pieces (mm_olmo `format_example` + the formatter's system prompt) --------

    def target_dim_for(self, rng: np.random.RandomState) -> int:
        """Render target for one example: sampled on train when a range is set, fixed otherwise
        (mm_olmo ``target_dim_for``)."""
        cfg = self.config
        if cfg.target_longest_image_dim_range is None or self.split != "train":
            return cfg.target_longest_image_dim
        lo, hi = cfg.target_longest_image_dim_range
        return int(rng.randint(lo, hi + 1))  # numpy's randint excludes `high`

    def pdf_path(self, row: Dict[str, Any]) -> str:
        return pdf_path_for(self.config.dataset_path, row["pdf_relpath"])

    @staticmethod
    def transcription(row: Dict[str, Any]) -> str:
        """The target text; blank pages are transcribed as ``"No text found"`` (mm_olmo)."""
        return row["natural_text"] or "No text found"

    def user_prompt(self) -> str:
        """The user turn: only the style tag, since the ``olmocr`` style has no question
        (:func:`~.pixmo_cap.style_tag_prompt`)."""
        return style_tag_prompt(OLMOCR_STYLE)

    # -- example ---------------------------------------------------------------------------

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Build page ``index``, deterministically skipping unusable rows.

        A page whose PDF fails to render, or whose transcription leaves no loss tokens after
        truncation, must not raise out of here: it would spend the mixture loader's error budget
        and a run of them would abort training. Same policy as the other sources; see
        :func:`~olmo_core.data.multimodal.sft_common.get_example_with_skip`.
        """
        return get_example_with_skip(self, index, len(self))

    def _build(self, i: int, epoch: Optional[int] = None) -> Dict[str, np.ndarray]:
        cfg = self.config
        row = self._data[int(self._index[i])]
        # Per (row, epoch): `target_dim_for` samples a render size, which should vary by epoch.
        rng = self.epoch_rng(i, epoch)
        # mm_olmo draw order: the render size in `format_example`, then the formatter's prefix.
        target_dim = self.target_dim_for(rng)
        text = self.transcription(row)
        image = render_pdf_page(self.pdf_path(row), target_dim, cache=self.render_cache)
        prompt = self.user_prompt()
        # One image, one (tag, transcription) turn: the shared message encoder builds exactly the
        # stage-1 single-branch layout (user header + image block + tag, then the response).
        seq = encode_sft_example(
            self.tokenizer,
            image,
            [(prompt, text)],
            max_crops=cfg.max_crops,
            loss_token_weighting=cfg.loss_token_weighting,
            message_weight=cfg.message_weight,
            token_ids=cfg.token_ids,
            message_format=cfg.message_format,
            shuffle_rng=rng,
        )
        return truncate_for_format(seq, cfg.max_sequence_length, cfg.token_ids, cfg.message_format)
