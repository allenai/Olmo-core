"""PDFA pages with LightOn transcripts: page transcription and whole-document tasks.

The data is a build of ``pixparse/pdfa-eng-wds`` (web PDFs) joined with the page transcripts of
``lightonai/LightOnOCR-mix-0126`` (written by Qwen3-VL-235B), at :data:`~.paths.PDFA_PAGES`:

* ``index.parquet``: one row per document with its page count, split, and byte offsets;
* ``pdfa/pdfa-NNNN.tar``: the PDF bytes, read with one ``pread`` at ``pdf_off``;
* ``lighton/lighton-XXX.jsonl``: one line per document with its transcribed pages, read with one
  ``pread`` at ``lo_off``. Not every page has a transcript (about 93% do).

Pages are rasterised on the fly with ``pypdfium2``, like the olmOCR sources. One source is one
task (:data:`PDFA_TASKS`):

``page_ocr``
    One page, transcribed. Every transcribed page of every document is one example.
``transcribe_page``
    A run of consecutive pages, each written after its page label (``"Page 12:"``); the answer
    transcribes one of them.
``missing_page``
    The same run with one page left out; the answer is that page's text.
``find_page``
    The same run and a line quoted from one page, found on no other page of the run; the answer
    is the page number.
``page_order``
    A run of consecutive pages in shuffled order, labelled ``"Image k"``; the answer lists the
    image numbers in reading order.

For the multi-page tasks an example is one document, and each epoch draws a new run of pages and
a new target from it. Page labels are text because the language model's full-attention layers
carry no position encoding.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds

from .message_sequence import encode_sft_example
from .olmocr import _PDFIUM_LOCK
from .paths import PDFA_PAGES
from .sft_common import (
    EpochSeededExamples,
    SftMessageFormat,
    get_example_with_skip,
    truncate_for_format,
)

__all__ = [
    "PDFA_TASKS",
    "PdfaPagesDatasetConfig",
    "PdfaPagesDataset",
    "normalize_lighton_text",
    "quotable_lines",
    "render_pdf_pages",
]

log = logging.getLogger(__name__)

PDFA_TASKS: Tuple[str, ...] = (
    "page_ocr",
    "transcribe_page",
    "missing_page",
    "find_page",
    "page_order",
)

#: Upper bound on the image tokens of one page per crop (measured: 982-1,208 at 8 crops plus the
#: global view), used to size a run of pages before anything is rendered.
_TOKENS_PER_CROP = 140
#: Tokens kept free for the prompt and the answer when sizing a run of pages.
_ANSWER_RESERVE = 3500

_PROMPTS: Dict[str, Tuple[str, ...]] = {
    "page_ocr": (
        "Transcribe this page to Markdown.",
        "Write out the text of this page in Markdown.",
        "Convert this page to Markdown, in reading order.",
    ),
    "transcribe_page": (
        "Transcribe page {page} to Markdown.",
        "Write out the text of page {page} in Markdown.",
        "Convert page {page} of this document to Markdown, in reading order.",
    ),
    "missing_page": (
        "Page {page} of this document is missing. Write the text of page {page} in Markdown.",
        "Page {page} is not shown. Write out what page {page} says, in Markdown.",
    ),
    "find_page": (
        'Question: Which page contains this line? "{line}"\nAnswer:',
        'Question: On which page does this line appear? "{line}"\nAnswer:',
    ),
    "page_order": (
        "Question: These images are consecutive pages of one document, in shuffled order. "
        "List the image numbers in reading order.\nAnswer:",
        "Question: The pages of this document were shuffled. In which order should the images "
        "be read? Give the image numbers.\nAnswer:",
    ),
}

_STYLE_ATTR = re.compile(r'\s+(?:style|class|border|cellpadding|cellspacing)="[^"]*"')
_WRAPPER_TAG = re.compile(r"</?(?:div|span)\b[^>]*>")
_IMAGE_MARK = re.compile(r"!\[[^\]]*\]\([^)]*\)|<!--.*?-->|<img\b[^>]*>", re.S)
_TEACHER_NOTE = re.compile(
    r"(?im)^[ \t>*_(\[]*note:.*(?:the image contains|not transcribed|not included in the "
    r"(?:markdown|transcription)|graphical element).*$"
)
_BLANK_RUN = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")
_MARKUP = re.compile(r"<[^>]+>|[#*_`|>]+|\$+|\\[()\[\]]")
_SPACE = re.compile(r"\s+")


def normalize_lighton_text(text: str) -> str:
    """Clean one LightOn page transcript for use as a target.

    Removes what the labelling model added beyond the page's text: inline CSS and table
    attributes, ``<div>`` / ``<span>`` layout wrappers (their content is kept), image
    placeholders (``![image](image_1.png)``, ``<!-- Image: ... -->``) and its own notes about
    what it left out (``"Note: The image contains a watermark ..."``).

    :param text: The ``content`` field of a LightOnOCR-mix row.

    :returns: The cleaned transcript; empty if nothing is left.
    """
    text = _IMAGE_MARK.sub("", text)
    text = _TEACHER_NOTE.sub("", text)
    text = _STYLE_ATTR.sub("", text)
    text = _WRAPPER_TAG.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _BLANK_RUN.sub("\n\n", text).strip()


def _plain(text: str) -> str:
    """Text with markup removed, case folded and whitespace collapsed (for line matching)."""
    return _SPACE.sub(" ", _MARKUP.sub(" ", text)).strip().casefold()


def quotable_lines(text: str, others: Sequence[str]) -> List[str]:
    """Lines of ``text`` that identify its page among ``others``.

    A line qualifies if, with markup removed, it has 30 to 160 characters and at least five
    words, and its case-folded form occurs in none of the other pages' texts. Repeated headers
    and footers therefore never qualify.

    :param text: The target page's transcript.
    :param others: The texts of every other page in the run.

    :returns: The qualifying lines, markup removed, in page order.
    """
    haystack = [_plain(o) for o in others]
    out = []
    for line in text.split("\n"):
        line = _SPACE.sub(" ", _MARKUP.sub(" ", line)).strip()
        if not 30 <= len(line) <= 160 or len(line.split()) < 5 or '"' in line:
            continue
        folded = line.casefold()
        if not any(folded in h for h in haystack):
            out.append(line)
    return out


def render_pdf_pages(pdf_bytes: bytes, pages: Sequence[int], target_longest_image_dim: int):
    """Render pages of a PDF to RGB PIL images whose longest side is the target.

    :param pdf_bytes: The PDF file's bytes.
    :param pages: Zero-based page indices, in the order to return them.
    :param target_longest_image_dim: Longest side of each rendered page, in pixels.

    :returns: One image per requested page.
    """
    import pypdfium2

    with _PDFIUM_LOCK:
        pdf = pypdfium2.PdfDocument(pdf_bytes)
        try:
            out = []
            for index in pages:
                page = pdf[int(index)]
                scale = target_longest_image_dim / max(page.get_size())
                out.append(page.render(scale=scale).to_pil().convert("RGB"))
            return out
        finally:
            pdf.close()


@dataclass
class PdfaPagesDatasetConfig(Config):
    """One task over the PDFA pages build (see the module docstring)."""

    task: str = "page_ocr"
    """One of :data:`PDFA_TASKS`."""

    split: str = "train"
    """``train`` or ``heldout`` (documents never trained on, for the long-document checks)."""

    dataset_path: str = PDFA_PAGES
    """Root holding ``index.parquet``, ``pdfa/`` and ``lighton/``."""

    min_doc_pages: int = 1
    """Keep only documents with at least this many pages."""

    window_pages: Tuple[int, int] = (4, 32)
    """Multi-page tasks: the run length is drawn uniformly from this inclusive range, then
    limited by the document, by ``max_sequence_length`` and (``page_order``) by 20."""

    min_coverage: float = 0.8
    """Multi-page tasks: keep documents with a transcript for at least this share of pages."""

    target_longest_image_dim_range: Optional[Tuple[int, int]] = (1024, 2048)
    """Training-time render size: the longest side is drawn uniformly from this inclusive range
    per example. ``None``, or the ``heldout`` split, renders at ``target_longest_image_dim``."""

    target_longest_image_dim: int = 1536

    max_crops: int = 8
    context_max_crops: Optional[int] = None
    """``transcribe_page``: crop budget of the pages that are not the target (default
    ``max_crops``). The other tasks read every page at ``max_crops``."""

    max_sequence_length: Optional[int] = None
    """Longest example in tokens. A run of pages is sized to fit; a transcript is tail-truncated."""

    loss_token_weighting: str = "none"
    message_weight: Optional[float] = None
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)
    """Image token IDs of the selected language-model tokenizer (set by the mixture)."""
    message_format: SftMessageFormat = "qwen3"
    seed: int = 0

    def validate(self):
        if self.task not in PDFA_TASKS:
            raise OLMoConfigurationError(f"task must be one of {PDFA_TASKS}, got {self.task!r}")
        if self.split not in ("train", "heldout"):
            raise OLMoConfigurationError(f"split must be train or heldout, got {self.split!r}")
        lo, hi = self.window_pages
        if lo < 3 or hi < lo:
            raise OLMoConfigurationError(
                f"window_pages must be (lo >= 3, hi >= lo), got {(lo, hi)}"
            )
        if self.target_longest_image_dim_range is not None:
            lo, hi = self.target_longest_image_dim_range
            if lo <= 0 or hi < lo:
                raise OLMoConfigurationError(
                    f"target_longest_image_dim_range must be a positive (lo, hi), got {(lo, hi)}"
                )

    def build(self, tokenizer) -> "PdfaPagesDataset":
        self.validate()
        return PdfaPagesDataset(self, tokenizer)


class PdfaPagesDataset(EpochSeededExamples):
    """Map-style dataset over the PDFA pages build for one task."""

    def __init__(self, config: PdfaPagesDatasetConfig, tokenizer):
        import pyarrow.parquet as pq

        self.config = config
        self.tokenizer = tokenizer
        self.multi_page = config.task != "page_ocr"
        index_path = os.path.join(config.dataset_path, "index.parquet")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"{index_path} not found; see v7-recipe/long-docs/README.md")
        columns = ["key", "shard", "pages", "pdf_off", "pdf_size", "split"]
        columns += ["lo_bucket", "lo_off", "lo_size", "lo_pages"]
        table = pq.read_table(index_path, columns=columns).to_pydict()
        pages = np.asarray(table["pages"], dtype=np.int64)
        lo_pages = np.asarray(table["lo_pages"], dtype=np.int64)
        keep = (np.asarray(table["split"]) == config.split) & (lo_pages > 0)
        keep &= pages >= config.min_doc_pages
        if self.multi_page:
            keep &= pages >= config.window_pages[0]
            keep &= lo_pages >= config.min_coverage * pages
        rows = np.flatnonzero(keep)
        self._docs = {
            name: np.asarray(table[name])[rows]
            for name in ("shard", "pages", "pdf_off", "pdf_size", "lo_bucket", "lo_off", "lo_size")
        }
        self._keys = [table["key"][i] for i in rows]
        self._page_ends = np.cumsum(lo_pages[rows])  # page_ocr: example -> document
        self._warned = 0
        log.info(
            "PDFA pages %s/%s: %d documents, %d examples",
            config.task,
            config.split,
            len(rows),
            len(self),
        )

    def __len__(self) -> int:
        if self.multi_page:
            return len(self._keys)
        return int(self._page_ends[-1]) if len(self._page_ends) else 0

    # -- reading ---------------------------------------------------------------------------

    @staticmethod
    def _pread(path: str, offset: int, size: int) -> bytes:
        fd = os.open(path, os.O_RDONLY)
        try:
            return os.pread(fd, size, offset)
        finally:
            os.close(fd)

    def pdf_bytes(self, doc: int) -> bytes:
        """The PDF of document ``doc`` (an index into the kept documents)."""
        d = self._docs
        path = os.path.join(
            self.config.dataset_path, "pdfa", f"pdfa-{int(d['shard'][doc]):04d}.tar"
        )
        return self._pread(path, int(d["pdf_off"][doc]), int(d["pdf_size"][doc]))

    def transcripts(self, doc: int) -> Dict[int, str]:
        """Zero-based page index -> raw LightOn transcript, for the transcribed pages."""
        d = self._docs
        path = os.path.join(
            self.config.dataset_path, "lighton", f"lighton-{int(d['lo_bucket'][doc]):03d}.jsonl"
        )
        record = json.loads(self._pread(path, int(d["lo_off"][doc]), int(d["lo_size"][doc])))
        if record["key"] != self._keys[doc]:
            raise RuntimeError(f"LightOn record {record['key']} read for {self._keys[doc]}")
        return {int(p[0]): p[1] for p in record["pages"]}

    # -- example ---------------------------------------------------------------------------

    def target_dim_for(self, rng: np.random.RandomState) -> int:
        """Render size of one example: sampled on train when a range is set, fixed otherwise."""
        cfg = self.config
        if cfg.target_longest_image_dim_range is None or cfg.split != "train":
            return cfg.target_longest_image_dim
        lo, hi = cfg.target_longest_image_dim_range
        return int(rng.randint(lo, hi + 1))

    def max_window(self) -> int:
        """Most pages a run can hold within ``max_sequence_length`` (an upper-bound estimate)."""
        cfg = self.config
        hi = cfg.window_pages[1] if cfg.task != "page_order" else min(cfg.window_pages[1], 20)
        if cfg.max_sequence_length is None:
            return hi
        crops = (
            cfg.max_crops
            if cfg.task != "transcribe_page"
            else (cfg.context_max_crops or cfg.max_crops)
        )
        per_page = _TOKENS_PER_CROP * (crops + 1) + 8
        return max(3, min(hi, (cfg.max_sequence_length - _ANSWER_RESERVE) // per_page))

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Build example ``index``, deterministically skipping unusable rows
        (:func:`~olmo_core.data.multimodal.sft_common.get_example_with_skip`)."""
        return get_example_with_skip(self, index, len(self))

    def _plan(self, i: int, rng: np.random.RandomState) -> Dict[str, Any]:
        """Choose the document, the pages shown, their labels and the (prompt, answer) turn."""
        cfg = self.config
        prompts = _PROMPTS[cfg.task]
        if not self.multi_page:
            doc = int(np.searchsorted(self._page_ends, i, side="right"))
            nth = i - (int(self._page_ends[doc - 1]) if doc else 0)
            texts = self.transcripts(doc)
            page = sorted(texts)[nth]
            answer = normalize_lighton_text(texts[page])
            if not answer:
                raise ValueError("empty transcript")
            return dict(
                doc=doc,
                pages=[page],
                labels=None,
                crops=None,
                target=page,
                turn=(prompts[rng.randint(len(prompts))], answer),
            )

        doc = i
        n_pages = int(self._docs["pages"][doc])
        width = min(
            int(rng.randint(cfg.window_pages[0], cfg.window_pages[1] + 1)),
            n_pages,
            self.max_window(),
        )
        start = int(rng.randint(0, n_pages - width + 1))
        window = list(range(start, start + width))
        raw = self.transcripts(doc)
        texts = {p: normalize_lighton_text(raw[p]) for p in window if p in raw}
        texts = {p: t for p, t in texts.items() if t}
        if not texts:
            raise ValueError("no transcribed page in the run")
        prompt = prompts[rng.randint(len(prompts))]
        candidates = sorted(texts)
        labels: Optional[List[str]] = [f"Page {p + 1}:" for p in window]
        crops: Optional[List[int]] = None

        if cfg.task == "page_order":
            order = rng.permutation(width)  # image k shows page window[order[k]]
            shown = [window[j] for j in order]
            answer = ", ".join(str(int(k) + 1) for k in np.argsort(order))
            return dict(
                doc=doc, pages=shown, labels=None, crops=None, target=None, turn=(prompt, answer)
            )

        if cfg.task == "find_page":
            rng.shuffle(candidates)
            for target in candidates:
                lines = quotable_lines(texts[target], [t for p, t in texts.items() if p != target])
                if lines:
                    line = lines[rng.randint(len(lines))]
                    return dict(
                        doc=doc,
                        pages=window,
                        labels=labels,
                        crops=None,
                        target=target,
                        turn=(prompt.format(line=line), f"Page {target + 1}"),
                    )
            raise ValueError("no page of the run has a line found on no other page")

        target = candidates[rng.randint(len(candidates))]
        prompt = prompt.format(page=target + 1)
        if cfg.task == "missing_page":
            shown = [p for p in window if p != target]
            labels = [f"Page {p + 1}:" for p in shown]
            return dict(
                doc=doc,
                pages=shown,
                labels=labels,
                crops=None,
                target=target,
                turn=(prompt, texts[target]),
            )
        if cfg.context_max_crops is not None:
            crops = [cfg.max_crops if p == target else cfg.context_max_crops for p in window]
        return dict(
            doc=doc,
            pages=window,
            labels=labels,
            crops=crops,
            target=target,
            turn=(prompt, texts[target]),
        )

    def _build(self, i: int, epoch: Optional[int] = None) -> Dict[str, np.ndarray]:
        cfg = self.config
        rng = self.epoch_rng(i, epoch)
        target_dim = self.target_dim_for(rng)
        plan = self._plan(i, rng)
        images = render_pdf_pages(self.pdf_bytes(plan["doc"]), plan["pages"], target_dim)
        seq = encode_sft_example(
            self.tokenizer,
            images,
            [plan["turn"]],
            max_crops=cfg.max_crops,
            max_images=len(images),
            image_labels=plan["labels"],
            per_image_max_crops=plan["crops"],
            loss_token_weighting=cfg.loss_token_weighting,
            message_weight=cfg.message_weight,
            token_ids=cfg.token_ids,
            message_format=cfg.message_format,
            shuffle_rng=rng,
        )
        return truncate_for_format(seq, cfg.max_sequence_length, cfg.token_ids, cfg.message_format)
