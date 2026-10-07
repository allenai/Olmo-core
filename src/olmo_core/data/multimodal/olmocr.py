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
and fixed (1536) otherwise, following olmOCR's own per-page DPI rule. pypdfium2 is not
thread-safe: by default pages render one at a time under a process-wide lock, and with
``render_workers > 0`` they render in parallel in small helper processes
(:class:`PdfRenderPool`; the same pypdfium2 call, identical pixels).

Transcriptions run long (documents pages: median ~580 tokens, p99 ~2900 with the Molmo2
tokenizer), so ``max_sequence_length`` should be set to the training sequence length; the
sequence is then tail-truncated like mm_olmo's preprocessor does.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
    "PdfRenderPool",
    "canonical_subset",
    "canonical_split",
    "get_render_pool",
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

#: Command that starts one render helper (see :mod:`.olmocr_render_worker`); the worker file is
#: run by path so the helper imports pdfium only, not ``olmo_core`` / ``torch``.
_WORKER_COMMAND: List[str] = [
    sys.executable,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "olmocr_render_worker.py"),
]


def render_pdf_page(pdf_path: str, target_longest_image_dim: int, render_workers: int = 0):
    """Render a single-page PDF to an RGB PIL image whose longest side is
    ``target_longest_image_dim`` pixels.

    Follows olmOCR's convention (``render_pdf_to_base64png``): the DPI is chosen per page as
    ``target * 72 / longest_mediabox_dim_in_points``, i.e. a render ``scale`` of
    ``target / longest_dim`` -- page sizes vary, so no fixed DPI. olmOCR rasterises with poppler,
    so fonts / antialiasing can differ slightly from pypdfium2's output.

    The shipped files are single-page extracts (the row's ``page_number`` is provenance in the
    original document), so only page 0 exists. pypdfium2 is not thread-safe, and
    :class:`~.mixture_data_loader.MixtureDataLoader` prefetches examples on a thread pool, so
    with ``render_workers == 0`` the render is serialised behind a process-wide lock, and with
    ``render_workers > 0`` it runs in one of that many helper processes (:class:`PdfRenderPool`,
    shared by every caller in this process) so pages render in parallel. Both paths make the
    same pypdfium2 call and return identical pixels; the lock is the fallback when the helpers
    cannot start.

    :param render_workers: Helper processes to render in, ``0`` for in-process rendering.

    :raises ImportError: If ``pypdfium2`` is not installed (``pip install pypdfium2``).
    :raises RuntimeError: If the PDF has more than one page.
    """
    from PIL import Image

    if render_workers > 0:
        rendered = get_render_pool(render_workers).render(pdf_path, target_longest_image_dim)
        if rendered is not None:
            width, height, pixels = rendered
            return Image.frombytes("RGB", (width, height), pixels)

    try:
        import pypdfium2  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "olmOCR-mix stores source PDFs, so rendering a page needs pypdfium2 "
            "(`pip install pypdfium2`)."
        ) from e

    from .olmocr_render_worker import render_rgb

    with _PDFIUM_LOCK:
        width, height, pixels = render_rgb(pdf_path, target_longest_image_dim)
    return Image.frombytes("RGB", (width, height), pixels)


class PdfRenderPool:
    """Up to ``max_workers`` helper processes rendering PDF pages for this process.

    Each helper is a copy of :mod:`.olmocr_render_worker` started lazily on first demand (an idle
    pool costs nothing) and used by one loader thread at a time, so renders of different pages
    proceed in parallel and never hold the GIL or pypdfium2's lock in the trainer. A request
    whose helper dies is retried once on a fresh helper; after ``max_workers`` such deaths
    (or if a helper cannot be started at all) the pool disables itself and
    :meth:`render` returns ``None``, which sends callers to the in-process lock.

    :param max_workers: Helper processes to run at most.
    :param command: The helper's command line, for tests; defaults to the module's worker.
    """

    def __init__(self, max_workers: int, command: Optional[List[str]] = None):
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.max_workers = max_workers
        self.command = list(_WORKER_COMMAND if command is None else command)
        # Idle helpers; a ``None`` in it tells waiters the pool is disabled.
        self._idle: "queue.Queue[Optional[subprocess.Popen]]" = queue.Queue()
        self._lock = threading.Lock()
        self._started = 0  # helpers started and still counted as live
        self._deaths = 0
        self._disabled = False
        self._procs: List[subprocess.Popen] = []

    @property
    def disabled(self) -> bool:
        """Whether the pool has given up on its helpers (callers render in-process)."""
        return self._disabled

    @property
    def num_started(self) -> int:
        """Helpers started so far (live or not), for tests and logs."""
        return len(self._procs)

    def _start(self) -> Optional[subprocess.Popen]:
        try:
            proc = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None
            )
        except OSError as e:
            log.warning(
                "olmOCR render helper %s could not start (%s); rendering pages in-process",
                self.command,
                e,
            )
            return None
        self._procs.append(proc)
        return proc

    def _acquire(self) -> Optional[subprocess.Popen]:
        """An idle helper, a new one while under ``max_workers``, else wait for one."""
        with self._lock:
            if self._disabled:
                return None
            try:
                return self._idle.get_nowait()
            except queue.Empty:
                pass
            if self._started < self.max_workers:
                proc = self._start()  # under the lock: one starter at a time
                if proc is None:
                    self._disabled = True
                    return None
                self._started += 1
                return proc
        proc = self._idle.get()
        if proc is None:  # the pool was disabled while waiting; leave the wake-up for the next
            self._idle.put(None)
        return proc

    def _discard(self, proc: subprocess.Popen) -> None:
        """Drop a dead (or misbehaving) helper and count the death."""
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        try:
            proc.kill()
        except OSError:
            pass
        proc.wait()
        with self._lock:
            self._started -= 1
            self._deaths += 1
            if self._deaths >= self.max_workers and not self._disabled:
                self._disabled = True
                self._idle.put(None)  # wake threads waiting for a helper that will not come
                log.warning(
                    "olmOCR render helpers died %d times (last exit code %s); "
                    "rendering pages in-process from now on",
                    self._deaths,
                    proc.returncode,
                )

    @staticmethod
    def _request(
        proc: subprocess.Popen, pdf_path: str, target_longest_image_dim: int
    ) -> Tuple[int, int, bytes]:
        assert proc.stdin is not None and proc.stdout is not None
        request = {"pdf_path": pdf_path, "target": int(target_longest_image_dim)}
        proc.stdin.write(json.dumps(request).encode("utf-8") + b"\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            raise EOFError("render helper closed its output")
        header = json.loads(line)
        if not header.get("ok"):
            raise RuntimeError(header.get("error", "render failed in the helper"))
        size = int(header["size"])
        pixels = proc.stdout.read(size)
        if len(pixels) != size:
            raise EOFError("render helper closed its output mid-image")
        return int(header["width"]), int(header["height"]), pixels

    def render(
        self, pdf_path: str, target_longest_image_dim: int
    ) -> Optional[Tuple[int, int, bytes]]:
        """Render in a helper, as :func:`~.olmocr_render_worker.render_rgb` would in-process.

        :returns: ``(width, height, rgb_bytes)``, or ``None`` when the pool is disabled and the
            caller should render in-process.
        :raises RuntimeError: When the helper could not render the page (its error message).
        """
        for _attempt in range(2):
            proc = self._acquire()
            if proc is None:
                return None
            try:
                result = self._request(proc, pdf_path, target_longest_image_dim)
            except RuntimeError:
                self._idle.put(proc)  # the helper is healthy; the page is not
                raise
            except (OSError, EOFError, ValueError):
                self._discard(proc)
                continue
            self._idle.put(proc)
            return result
        return None

    def close(self) -> None:
        """Stop every helper (they also exit on their own when the trainer's stdin pipe closes)."""
        with self._lock:
            self._disabled = True
            self._idle.put(None)
            procs, self._procs = self._procs, []
        for proc in procs:
            for stream in (proc.stdin, proc.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
            if proc.poll() is None:
                try:
                    proc.kill()
                except OSError:
                    pass
            proc.wait()


_POOLS: Dict[int, PdfRenderPool] = {}
_POOLS_LOCK = threading.Lock()


def get_render_pool(max_workers: int) -> PdfRenderPool:
    """The process's shared :class:`PdfRenderPool` of ``max_workers`` helpers, created on first
    use. Every olmOCR source of a run asks for the same size, so a run has one pool."""
    with _POOLS_LOCK:
        pool = _POOLS.get(max_workers)
        if pool is None:
            pool = _POOLS[max_workers] = PdfRenderPool(max_workers)
            log.info("olmOCR pages render in up to %d helper processes", max_workers)
        return pool


def _close_render_pools() -> None:
    with _POOLS_LOCK:
        pools = list(_POOLS.values())
        _POOLS.clear()
    for pool in pools:
        pool.close()


atexit.register(_close_render_pools)


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

    render_workers: int = 0
    """Helper processes that rasterise pages. ``0`` renders in the loader thread under a
    process-wide lock (pypdfium2 is not thread-safe), so concurrent pages queue behind each
    other: a national-archives page takes about a second. ``N > 0`` renders in up to ``N`` small
    helper processes shared by every olmOCR source of this process (:class:`PdfRenderPool`), so
    pages render in parallel and the loader threads only wait on a pipe. The pixels are
    identical either way; the lock is the fallback when the helpers cannot start."""

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
        if self.render_workers < 0:
            raise OLMoConfigurationError("render_workers must be >= 0 (0 renders in-process)")
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

    def _build(self, i: int, epoch: Optional[int] = None) -> Dict[str, np.ndarray]:
        cfg = self.config
        row = self._data[int(self._index[i])]
        # Per (row, epoch): `target_dim_for` samples a render size, which should vary by epoch.
        rng = self.epoch_rng(i, epoch)
        # mm_olmo draw order: the render size in `format_example`, then the formatter's prefix.
        target_dim = self.target_dim_for(rng)
        text = self.transcription(row)
        image = render_pdf_page(self.pdf_path(row), target_dim, render_workers=cfg.render_workers)
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
