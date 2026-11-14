"""Synthetic OCR sources for Molmo2 stage-1: rendered text with exact transcriptions.

Two sources, English only, train splits only:

* :class:`NvidiaSynthOcrDataset` -- ``nvidia/OCR-Synthetic-Multilingual-v1`` (CC BY 4.0), the
  SynthDoG-style data behind Nemotron OCR v2. Words and short phrases are scattered over photos
  or flat colours, often rotated or vertical, so it is scene-like text, not a page. It has a tag
  of its own, ``synth_ocr``, and its target is the text joined by spaces,
  the form TextOCR's answers take. The dataset's own
  ``labels`` field is NOT used as the target: it joins the words in the order of the sentence
  they were cut from, which only 16% of images show visually (measured on 496 validation images),
  so it would train the model to recite text in an order it cannot see. The target is rebuilt
  from the line boxes in visual order instead (:func:`layout_text`).
* :class:`SyntheticReceiptsDataset` -- ``albertobarnabo/synthetic-receipts-ocr`` (Apache 2.0),
  thermal-printer receipts, of which the US and UK locales are English. It has a tag of its own,
  ``receipt_ocr``, and its target is the printed lines (:func:`receipt_text`).
  The photo-degraded image is used (perspective, lighting, blur, JPEG), the closer one to real
  receipts. The parquet row groups hold several hundred inline images each, so the English train
  rows are first converted once into a memory-mapped Arrow dataset
  (:func:`prepare_synthetic_receipts`).

Both only ever read their ``train`` split: NVIDIA's ``validation`` / ``test`` files and the
receipts' ``eval`` parquets are never globbed.
"""

from __future__ import annotations

import glob
import io
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError

from .message_sequence import encode_sft_example
from .paths import NVIDIA_SYNTH_OCR, SYNTH_RECEIPTS_OCR
from .pixmo_cap import style_tag_prompt
from .sft_common import EpochSeededExamples, get_example_with_skip, truncate_example

__all__ = [
    "NvidiaSynthOcrDatasetConfig",
    "NvidiaSynthOcrDataset",
    "SyntheticReceiptsDatasetConfig",
    "SyntheticReceiptsDataset",
    "RECEIPT_LOCALES",
    "SYNTH_OCR_STYLE",
    "RECEIPT_OCR_STYLE",
    "layout_text",
    "receipt_text",
    "prepare_synthetic_receipts",
]

log = logging.getLogger(__name__)

#: Style of the NVIDIA set: its own tag, so its synthetic scattered text is learned apart from
#: TextOCR's real photos.
SYNTH_OCR_STYLE = "synth_ocr"

#: Style of the receipts: its own tag, so a receipt transcription is learned apart from
#: olmOCR's page transcriptions.
RECEIPT_OCR_STYLE = "receipt_ocr"

#: The English locales of the receipts dataset.
RECEIPT_LOCALES: Tuple[str, ...] = ("US", "UK")

_RECEIPT_COLUMNS = ("id", "image_photo", "full_text", "locale")


def _run_example(ds, tokenizer, image, prompt: str, text: str, cfg, index: int):
    seq = encode_sft_example(
        tokenizer,
        image,
        [(prompt, text)],
        max_crops=cfg.max_crops,
        loss_token_weighting=cfg.loss_token_weighting,
        message_weight=cfg.message_weight,
        shuffle_rng=ds.epoch_rng(index),
    )
    if cfg.max_sequence_length is not None:
        seq = truncate_example(seq, cfg.max_sequence_length)
    return seq


# ---------------------------------------------------------------------------
# NVIDIA OCR-Synthetic-Multilingual-v1 (English)
# ---------------------------------------------------------------------------


def layout_text(annotation: Dict[str, Any]) -> str:
    """The text of one annotated image in visual reading order, joined by spaces.

    Line boxes are grouped into rows by their vertical centres (a box joins a row when its centre
    is within half the height of the shortest box involved), rows are read top to bottom and the
    boxes in a row left to right. Using the shortest height keeps a tall, vertical word from
    swallowing the several lines it spans: it joins only the line through its centre. Falls back
    to the word boxes when an image has no line boxes.

    :param annotation: One decoded ``annotations`` entry (``line_bboxes`` / ``word_bboxes``).

    :returns: The text, or ``""`` when the image has none.
    """
    boxes = annotation.get("line_bboxes") or annotation.get("word_bboxes") or []
    items = []
    for b in boxes:
        text = (b.get("text") or "").strip()
        if not text:
            continue
        x, y, _, h = (float(v) for v in b["bbox"])
        items.append((y + h / 2.0, max(h, 1.0), x, text))
    items.sort(key=lambda it: it[0])
    rows: List[List[Tuple[float, float, float, str]]] = []
    for it in items:
        if rows:
            last = rows[-1]
            cy = sum(r[0] for r in last) / len(last)
            if abs(it[0] - cy) < 0.5 * min(it[1], min(r[1] for r in last)):
                last.append(it)
                continue
        rows.append([it])
    return " ".join(it[3] for row in rows for it in sorted(row, key=lambda it: it[2]))


@dataclass
class NvidiaSynthOcrDatasetConfig(Config):
    """``nvidia/OCR-Synthetic-Multilingual-v1``, English train split, as ``synth_ocr`` examples."""

    dataset_path: str = NVIDIA_SYNTH_OCR
    """The dataset's local copy; ``en/train/*.h5`` is read under it and nothing else."""

    max_crops: int = 8
    max_sequence_length: Optional[int] = None
    """Tail-truncate the built sequence to this many tokens (see
    :func:`~.sft_common.truncate_example`)."""
    loss_token_weighting: str = "none"
    """``"none"`` weights every response token equally, like the other OCR sources."""
    message_weight: Optional[float] = None
    seed: int = 0

    def validate(self):
        if not self.dataset_path:
            raise OLMoConfigurationError("dataset_path must point at the dataset's local copy")

    def build(self, tokenizer) -> "NvidiaSynthOcrDataset":
        self.validate()
        return NvidiaSynthOcrDataset(self, tokenizer)


def _import_h5py():
    try:
        import h5py
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The NVIDIA synthetic OCR source reads HDF5 files and needs `h5py` "
            "(`pip install h5py`; the Molmo2-Stage1 launch installs it)."
        ) from e
    return h5py


class NvidiaSynthOcrDataset(EpochSeededExamples):
    """Map-style dataset over the samples of the English train ``.h5`` files."""

    def __init__(self, config: NvidiaSynthOcrDatasetConfig, tokenizer):
        h5py = _import_h5py()
        self.config = config
        self.tokenizer = tokenizer
        self.files = sorted(glob.glob(os.path.join(config.dataset_path, "en", "train", "*.h5")))
        if not self.files:
            raise OLMoConfigurationError(
                f"no en/train/*.h5 files under {config.dataset_path!r}; download them with "
                "`hf download nvidia/OCR-Synthetic-Multilingual-v1 --repo-type dataset "
                "--include 'en/train/*.h5' --local-dir <dataset_path>`"
            )

        # Opening a file costs about a second on weka, so the 63 files are counted in parallel.
        def count(path: str) -> int:
            with h5py.File(path, "r") as f:
                return int(f["labels"].shape[0])

        with ThreadPoolExecutor(max_workers=min(16, len(self.files))) as pool:
            counts = list(pool.map(count, self.files))
        self._starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self._handles: Dict[str, Any] = {}
        self._pid = os.getpid()
        log.info(
            "NVIDIA synthetic OCR (en/train): %d samples in %d files",
            len(self),
            len(self.files),
        )

    def __len__(self) -> int:
        return int(self._starts[-1])

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def _file(self, path: str):
        # h5py handles do not survive a fork: reopen them in each worker process.
        if os.getpid() != self._pid:
            self._handles = {}
            self._pid = os.getpid()
        handle = self._handles.get(path)
        if handle is None:
            handle = _import_h5py().File(path, "r")
            self._handles[path] = handle
        return handle

    def locate(self, i: int) -> Tuple[str, int]:
        """``(file, row)`` of sample ``i``."""
        k = int(np.searchsorted(self._starts, i, side="right")) - 1
        return self.files[k], int(i - self._starts[k])

    def read(self, i: int) -> Tuple[bytes, Dict[str, Any]]:
        """``(jpeg_bytes, annotation)`` of sample ``i``."""
        import json

        path, row = self.locate(i)
        f = self._file(path)
        return f["images"][row].tobytes(), json.loads(f["annotations"][row])

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Build row ``index``, deterministically skipping rows with no usable target; see
        :func:`~olmo_core.data.multimodal.sft_common.get_example_with_skip`."""
        return get_example_with_skip(self, index, len(self))

    def _build(self, i: int) -> Dict[str, np.ndarray]:
        from PIL import Image

        image_bytes, annotation = self.read(i)
        text = layout_text(annotation)
        if not text:
            raise ValueError("image has no text")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        prompt = style_tag_prompt(SYNTH_OCR_STYLE)
        return _run_example(self, self.tokenizer, image, prompt, text, self.config, i)


# ---------------------------------------------------------------------------
# Synthetic receipts (English)
# ---------------------------------------------------------------------------

_SEPARATOR_LINE = re.compile(r"^[-=_*~.#\s]{3,}$")


def receipt_text(full_text: str) -> str:
    """A receipt's printed lines as a page transcription.

    The dataset's ``full_text`` pads each line with spaces to align the price column of a
    monospaced printer. The padding is collapsed to one space (``DIS MEN JUN   $12.99`` ->
    ``DIS MEN JUN $12.99``), lines made only of rule characters (``-----``) are dropped, as are
    blank lines.
    """
    lines = []
    for line in full_text.splitlines():
        line = re.sub(r"[ \t]{2,}", " ", line).strip()
        if not line or _SEPARATOR_LINE.match(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def prepare_synthetic_receipts(
    source_dir: str, out_dir: str, *, locales: Sequence[str] = RECEIPT_LOCALES
) -> int:
    """Convert the English train rows of the receipts parquets into an Arrow dataset.

    Reads ``<source_dir>/data/train-*.parquet`` only (never the ``eval`` parquets), keeps rows of
    ``locales`` whose ``split_policy`` is ``train``, and saves ``id`` / ``image_photo`` (the JPEG
    bytes) / ``full_text`` / ``locale`` with ``save_to_disk`` to ``out_dir``.

    :returns: The number of rows written.
    """
    import datasets
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(source_dir, "data", "train-*.parquet")))
    if not files:
        raise OLMoConfigurationError(f"no data/train-*.parquet files under {source_dir!r}")
    tables = []
    for path in files:
        t = pq.read_table(path, columns=list(_RECEIPT_COLUMNS) + ["split_policy"])
        keep = [
            loc in locales and pol == "train"
            for loc, pol in zip(
                t.column("locale").to_pylist(), t.column("split_policy").to_pylist()
            )
        ]
        t = t.filter(pa.array(keep)).drop(["split_policy"])
        image = t.column("image_photo")
        t = t.set_column(
            t.column_names.index("image_photo"),
            "image_photo",
            pa.array(
                [x["bytes"] if isinstance(x, dict) else x for x in image.to_pylist()], pa.binary()
            ),
        )
        tables.append(t)
    table = pa.concat_tables(tables)
    datasets.Dataset(table).save_to_disk(out_dir)
    return table.num_rows


@dataclass
class SyntheticReceiptsDatasetConfig(Config):
    """``albertobarnabo/synthetic-receipts-ocr``, English train receipts, as ``receipt_ocr``
    examples."""

    dataset_path: str = SYNTH_RECEIPTS_OCR
    """The dataset's local copy (``data/train-*.parquet``)."""
    prepared_subdir: str = "en_train_arrow"
    """Where under ``dataset_path`` the English train rows are kept as an Arrow dataset, built once
    by :func:`prepare_synthetic_receipts`."""

    max_crops: int = 8
    max_sequence_length: Optional[int] = None
    """Tail-truncate the built sequence to this many tokens (see
    :func:`~.sft_common.truncate_example`)."""
    loss_token_weighting: str = "none"
    """``"none"`` weights every response token equally, like the other OCR sources."""
    message_weight: Optional[float] = None
    seed: int = 0

    def validate(self):
        if not self.dataset_path:
            raise OLMoConfigurationError("dataset_path must point at the dataset's local copy")
        if not self.prepared_subdir or os.path.isabs(self.prepared_subdir):
            raise OLMoConfigurationError("prepared_subdir must be a relative directory name")

    def build(self, tokenizer) -> "SyntheticReceiptsDataset":
        self.validate()
        return SyntheticReceiptsDataset(self, tokenizer)


class SyntheticReceiptsDataset(EpochSeededExamples):
    """Map-style dataset over the English train receipts."""

    def __init__(self, config: SyntheticReceiptsDatasetConfig, tokenizer):
        from .dataset_compat import load_from_disk_compat

        self.config = config
        self.tokenizer = tokenizer
        prepared = os.path.join(config.dataset_path, config.prepared_subdir)
        if not os.path.isdir(prepared):
            # Not built here: every rank would race to write the same directory.
            raise OLMoConfigurationError(
                f'{prepared!r} does not exist; build it once with `python -c "from '
                "olmo_core.data.multimodal.synthetic_ocr import prepare_synthetic_receipts; "
                f"prepare_synthetic_receipts('{config.dataset_path}', '{prepared}')\"`"
            )
        self._data = load_from_disk_compat(prepared)
        locales = set(self._data.unique("locale"))
        if not locales <= set(RECEIPT_LOCALES):
            raise OLMoConfigurationError(
                f"{prepared!r} holds locales {sorted(locales)}; expected only {RECEIPT_LOCALES}"
            )
        log.info("synthetic receipts (en/train): %d receipts", len(self._data))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Build row ``index``, deterministically skipping rows with no usable target; see
        :func:`~olmo_core.data.multimodal.sft_common.get_example_with_skip`."""
        return get_example_with_skip(self, index, len(self))

    def _build(self, i: int) -> Dict[str, np.ndarray]:
        from PIL import Image

        row = self._data[i]
        text = receipt_text(row["full_text"])
        if not text:
            raise ValueError("receipt has no text")
        image = Image.open(io.BytesIO(row["image_photo"])).convert("RGB")
        prompt = style_tag_prompt(RECEIPT_OCR_STYLE)
        return _run_example(self, self.tokenizer, image, prompt, text, self.config, i)
