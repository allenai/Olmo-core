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
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError

from .message_sequence import encode_sft_example
from .paths import OLMOCR_MIX
from .pixmo_cap import style_tag_prompt
from .sft_common import (
    EpochSeededExamples,
    get_example_with_skip,
    load_hf_dataset,
    truncate_example,
)

__all__ = [
    "OLMOCR_STYLE",
    "OLMOCR_SUBSETS",
    "OLMOCR_SPLITS",
    "OlmOcrMixDatasetConfig",
    "OlmOcrMixDataset",
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


def render_pdf_page(pdf_path: str, target_longest_image_dim: int):
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

    :raises ImportError: If ``pypdfium2`` is not installed (``pip install pypdfium2``).
    :raises RuntimeError: If the PDF has more than one page.
    """
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

    seed: int = 0

    def validate(self):
        canonical_subset(self.subset)
        canonical_split(self.split)
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

    def _build(self, i: int) -> Dict[str, np.ndarray]:
        cfg = self.config
        row = self._data[int(self._index[i])]
        # Per (row, epoch): `target_dim_for` samples a render size, which should vary by epoch.
        rng = self.epoch_rng(i)
        # mm_olmo draw order: the render size in `format_example`, then the formatter's prefix.
        target_dim = self.target_dim_for(rng)
        text = self.transcription(row)
        image = render_pdf_page(self.pdf_path(row), target_dim)
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
            shuffle_rng=rng,
        )
        if cfg.max_sequence_length is not None:
            seq = truncate_example(seq, cfg.max_sequence_length)
        return seq
