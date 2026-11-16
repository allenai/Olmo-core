"""The OCR source group for Molmo2 stage-1 (``Molmo2-Stage1.py --ocr_rate``).

Every source is an image -> free-text task whose user turn is a style tag and nothing else: no
question, no instruction. There are two tasks, and the group's rate is divided between them
before it is divided among sources (:data:`OCR_TASK_SHARES`):

* **transcription** -- write out the text in the image.
  The four olmOCR-mix-1025 subsets (style ``olmocr``; PDF pages rendered at load time, see
  :class:`~olmo_core.data.multimodal.olmocr.OlmOcrMixDatasetConfig`), the scene-text tars
  (style ``textocr``; TextOCR, plus HierText / COCO-Text / UberText, which take TextOCR's form:
  the text in a photo as snippets joined by spaces), and two synthetic English sets (see
  :mod:`~olmo_core.data.multimodal.synthetic_ocr`): NVIDIA's SynthDoG-style scattered text
  (style ``synth_ocr``) and thermal-printer receipts (style ``receipt_ocr``), each with a tag of
  its own.
* **figure captions** -- describe a text-rich figure at three altitudes.
  The five ``text_rich_*`` categories (styles ``fig_caption_{high,mid,low}``; see
  :class:`~olmo_core.data.multimodal.text_rich_caption.TextRichCaptionDatasetConfig`).

olmOCR-mix and the figure captions are mm_olmo's two molmo3 stage-1 OCR groups
(``train_molmo3_stage1._base_mixture``: ``olmo_ocr`` and ``figure_ocr``, 0.075 each), which is why
the two tasks split the rate evenly. Within a task the split is by sqrt(size), mm_olmo's default
``root_size_factor``. A single flat sqrt(size) split would instead hand the figure captions 73%
of the rate, because their five categories are each larger than all of olmOCR-mix.

**What is deliberately not here.** Sources whose images come from VQA datasets (the Cambrian
subsets) or whose target is a caption rather than the image's text (TextCaps) are not general OCR
data and are left for a later stage. The single-caption ``text_rich_caption_v6_tars`` build is
superseded by the three-level build above, which holds the same images.

**Train splits only.** A source is in :data:`DEFAULT_OCR_SOURCES` only if it is known to hold
training data and nothing else:

* olmOCR-mix reads ``<subset>_train.parquet``; the training script refuses any other split.
* The figure captions read the ``train`` split of mm_olmo's build, which holds 2,048 rows per
  category out as ``validation``; the dataset has no option to read them.
* TextOCR keys are OpenImages ids: 1,200 sampled from the first, middle and last shard are all
  TextVQA *train* images, none of its 3,166 val or 3,289 test images.
* HierText, COCO-Text and UberText have official val / test splits, but their tar keys are
  synthetic (``cocotext_0000000``) and the JSON carries no split, so which splits the tars hold
  cannot be established from the data. Registered, not default
  (:data:`SPLIT_UNVERIFIED_SOURCES`).

``s2pdf`` and ``iabooks`` are the SAME pages as olmOCR-mix ``documents`` / ``books`` train (97.4%
/ 99.4% of their page ids, and every one of their documents; none of the eval pages), rendered
and transcribed by a different pipeline. Registered so either rendering can be chosen, but not
default, so a page is not counted twice.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from olmo_core.data.multimodal.ocr_caption_tars import OcrCaptionTarsDatasetConfig
from olmo_core.data.multimodal.olmocr import OLMOCR_STYLE, OlmOcrMixDatasetConfig
from olmo_core.data.multimodal.paths import OE_ENCODER_DATA
from olmo_core.data.multimodal.synthetic_ocr import (
    NvidiaSynthOcrDatasetConfig,
    SyntheticReceiptsDatasetConfig,
)
from olmo_core.data.multimodal.text_rich_caption import (
    CATEGORIES as TEXT_RICH_CATEGORIES,
)
from olmo_core.data.multimodal.text_rich_caption import TextRichCaptionDatasetConfig
from olmo_core.exceptions import OLMoConfigurationError

__all__ = [
    "OcrTarSource",
    "OCR_TAR_SOURCES",
    "OLMOCR_MIX_SOURCES",
    "TEXT_RICH_SOURCES",
    "SYNTHETIC_OCR_SOURCES",
    "OCR_SOURCE_NAMES",
    "DUPLICATE_OLMOCR_SOURCES",
    "SPLIT_UNVERIFIED_SOURCES",
    "DEFAULT_OCR_SOURCES",
    "TRANSCRIPTION",
    "FIGURE_CAPTION",
    "OCR_TASK_SHARES",
    "ocr_task",
    "ocr_task_shares",
    "ocr_weighting_sizes",
    "OLMOCR_STYLE",
    "TEXTOCR_STYLE",
    "build_ocr_source",
]

#: Scene-text transcription in TextOCR's form: every piece of text in a photo, joined by spaces,
#: with no layout. The non-default scene-text sources are tagged the same way.
TEXTOCR_STYLE = "textocr"


@dataclass(frozen=True)
class OcrTarSource:
    """One oe-encoder caption-tars source."""

    relpath: str
    """Shard directory under :data:`~olmo_core.data.multimodal.paths.OE_ENCODER_DATA`."""
    style: str
    strip_text_tags: bool
    """Whether the tars wrap the text in ``<text>...</text>`` (transcription-type sources)."""


#: Transcription tars: the target is the text visible in the image, ``<text>``-wrapped.
OCR_TAR_SOURCES: Dict[str, OcrTarSource] = {
    # olmOCR pages, pre-rendered; duplicates of olmOCR-mix documents / books (see module doc).
    "s2pdf": OcrTarSource("olmocr_v6_tars/s2pdf", OLMOCR_STYLE, True),
    "iabooks": OcrTarSource("olmocr_v6_tars/iabooks", OLMOCR_STYLE, True),
    # Scene text.
    "textocr": OcrTarSource("textocr_v6_tars", TEXTOCR_STYLE, True),
    "hiertext": OcrTarSource("scene_text_tars/hiertext_v6_tars", TEXTOCR_STYLE, True),
    "cocotext": OcrTarSource("scene_text_tars/cocotext_v6_tars", TEXTOCR_STYLE, True),
    "ubertext": OcrTarSource("scene_text_tars/ubertext_v6_tars", TEXTOCR_STYLE, True),
}

#: olmOCR-mix sources: group name -> ``OlmOcrMixDatasetConfig.subset``.
OLMOCR_MIX_SOURCES: Dict[str, str] = {
    "olmocr_documents": "documents",
    "olmocr_books": "books",
    "olmocr_loc_transcripts": "loc_transcripts",
    "olmocr_national_archives": "national_archives",
}

#: Figure-caption sources: group name -> ``TextRichCaptionDatasetConfig.category``.
TEXT_RICH_SOURCES: Dict[str, str] = {f"text_rich_{c}": c for c in TEXT_RICH_CATEGORIES}

#: Synthetic transcription sources (English train splits); each has its own config template.
SYNTHETIC_OCR_SOURCES: Tuple[str, ...] = ("nvidia_synth_en", "synth_receipts_en")

OCR_SOURCE_NAMES: Tuple[str, ...] = (
    tuple(OLMOCR_MIX_SOURCES)
    + tuple(TEXT_RICH_SOURCES)
    + tuple(OCR_TAR_SOURCES)
    + SYNTHETIC_OCR_SOURCES
)

#: Tar sources whose pages are already in an olmOCR-mix train subset (see module doc).
DUPLICATE_OLMOCR_SOURCES: Dict[str, str] = {"s2pdf": "olmocr_documents", "iabooks": "olmocr_books"}

#: Sources whose tars cannot be shown to hold only a training split (see module doc). Usable
#: through ``--ocr_sources``, but a default run never reads them.
SPLIT_UNVERIFIED_SOURCES: Tuple[str, ...] = ("hiertext", "cocotext", "ubertext")

DEFAULT_OCR_SOURCES: Tuple[str, ...] = tuple(
    n
    for n in OCR_SOURCE_NAMES
    if n not in DUPLICATE_OLMOCR_SOURCES and n not in SPLIT_UNVERIFIED_SOURCES
)

TRANSCRIPTION = "transcription"
FIGURE_CAPTION = "figure_caption"

#: How the group's rate is divided between the two tasks before any source sees it: mm_olmo's
#: ``olmo_ocr`` / ``figure_ocr`` groups at 0.075 each. Renormalised over the tasks a run selects.
OCR_TASK_SHARES: Dict[str, float] = {TRANSCRIPTION: 0.5, FIGURE_CAPTION: 0.5}


def ocr_task(name: str) -> str:
    """The task (:data:`TRANSCRIPTION` or :data:`FIGURE_CAPTION`) of an OCR source."""
    if name in TEXT_RICH_SOURCES:
        return FIGURE_CAPTION
    if name in OLMOCR_MIX_SOURCES or name in OCR_TAR_SOURCES or name in SYNTHETIC_OCR_SOURCES:
        return TRANSCRIPTION
    raise OLMoConfigurationError(f"Unknown OCR source {name!r}; expected one of {OCR_SOURCE_NAMES}")


def ocr_task_shares(names: Sequence[str]) -> List[float]:
    """For each source in ``names``, the share of the OCR rate that its *task* receives.

    The shares of the tasks present are renormalised to sum to 1, so selecting sources of a single
    task gives that task the whole rate. Dividing a task's share among its own sources (by size)
    is the caller's job.

    :param names: OCR source names.

    :returns: One task share per name, in order. Sources of the same task repeat the same value.
    """
    tasks = [ocr_task(n) for n in names]
    total = sum(OCR_TASK_SHARES[t] for t in set(tasks))
    return [OCR_TASK_SHARES[t] / total for t in tasks]


def ocr_weighting_sizes(names: Sequence[str], sizes: Sequence[int]) -> List[int]:
    """The sizes the OCR split weights its sources by.

    A synthetic source (:data:`SYNTHETIC_OCR_SOURCES`) counts as no larger than the largest real
    source of the same task in ``names``: its row count says how much data was generated, not how
    much it is worth, and uncapped the 1.46M NVIDIA images would take the biggest share of
    transcription on size alone (6.6% of the v2 mixture, more than the four olmOCR-mix subsets
    together). A task with no real source among ``names`` is left uncapped.

    :param names: OCR source names.
    :param sizes: Their row counts, parallel to ``names``.

    :returns: The sizes to weight by, parallel to ``names``.
    """
    tasks = [ocr_task(n) for n in names]
    caps: Dict[str, int] = {}
    for name, task, size in zip(names, tasks, sizes):
        if name not in SYNTHETIC_OCR_SOURCES:
            caps[task] = max(caps.get(task, 0), int(size))
    return [
        min(int(size), caps[task]) if name in SYNTHETIC_OCR_SOURCES and task in caps else int(size)
        for name, task, size in zip(names, tasks, sizes)
    ]


def build_ocr_source(
    name: str,
    tokenizer,
    *,
    olmocr: OlmOcrMixDatasetConfig,
    tars: OcrCaptionTarsDatasetConfig,
    text_rich: TextRichCaptionDatasetConfig,
    nvidia_synth: Optional[NvidiaSynthOcrDatasetConfig] = None,
    receipts: Optional[SyntheticReceiptsDatasetConfig] = None,
    data_root: str = OE_ENCODER_DATA,
):
    """Build one OCR source by name from the template configs.

    :param olmocr: template for the olmOCR-mix sources; its ``subset`` is overridden.
    :param tars: template for the caption-tars sources; ``dataset_path``, ``style`` and
        ``strip_text_tags`` are overridden from :data:`OCR_TAR_SOURCES`.
    :param text_rich: template for the figure-caption sources; its ``category`` is overridden.
    :param nvidia_synth: config of ``nvidia_synth_en``; defaults if not given.
    :param receipts: config of ``synth_receipts_en``; defaults if not given.
    :param data_root: where the oe-encoder tar directories live.
    """
    if name == "nvidia_synth_en":
        return (nvidia_synth or NvidiaSynthOcrDatasetConfig()).build(tokenizer)
    if name == "synth_receipts_en":
        return (receipts or SyntheticReceiptsDatasetConfig()).build(tokenizer)
    if name in OLMOCR_MIX_SOURCES:
        return olmocr.replace(subset=OLMOCR_MIX_SOURCES[name]).build(tokenizer)
    if name in TEXT_RICH_SOURCES:
        return text_rich.replace(category=TEXT_RICH_SOURCES[name]).build(tokenizer)
    if name in OCR_TAR_SOURCES:
        src = OCR_TAR_SOURCES[name]
        return tars.replace(
            dataset_path=os.path.join(data_root, src.relpath),
            style=src.style,
            strip_text_tags=src.strip_text_tags,
        ).build(tokenizer)
    raise OLMoConfigurationError(f"Unknown OCR source {name!r}; expected one of {OCR_SOURCE_NAMES}")
