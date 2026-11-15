"""Image-text shards: one stored image and the text to transcribe it to.

A common on-disk layout for transcription builds (equation and table crops, document pages), so
that one source reads all of them. A dataset is a directory with

* ``shards/<name>.tar``: plain tar files whose members are image files (PNG or JPEG), stored as
  they are;
* ``index.parquet``, or ``index/part-*.parquet`` for a large set: one row per example with ``id``,
  ``shard`` (a file name under ``shards/``), ``offset`` (the byte offset of the image *data* inside
  the tar, not of its header) and ``size`` (image bytes), ``width`` / ``height``, ``target`` (the
  text to train on, already in the olmOCR-mix convention), ``kind`` (``equation`` / ``table`` /
  ``page``), ``split`` (``train`` / ``heldout`` / ``excluded``) and any number of other columns,
  such as a builder's own boolean verdict.

Each example is one image and its ``target``. As for the olmOCR pages (:mod:`.olmocr`) there is no
question: the user turn is only the style tag (the bare ``"olmocr:"`` by default) and the assistant
turn is the target, used verbatim. The image is read with one ``pread`` at ``offset`` and is not
resized here; the crop budget does that, as for every other source.

Only rows of the selected ``split`` are indexed (``excluded`` rows can never be), and, when set,
only rows of the selected ``kinds`` whose ``keep_column`` is true. Rows that cannot be used (an
image that does not decode, an empty target, a target that leaves no loss tokens after truncation)
are skipped deterministically (:func:`~.sft_common.get_example_with_skip`).

The filtered index is held in memory as Arrow: ``shard``, ``offset``, ``size`` and ``target`` of
the kept rows only, about 150 bytes a row for short targets.
"""

from __future__ import annotations

import glob
import io
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds

from .message_sequence import encode_sft_example
from .pixmo_cap import style_tag_prompt
from .sft_common import (
    EpochSeededExamples,
    SftMessageFormat,
    get_example_with_skip,
    truncate_for_format,
)

__all__ = [
    "IMAGE_TEXT_SHARDS_SPLITS",
    "IMAGE_TEXT_SHARDS_STYLE",
    "ImageTextShardsDatasetConfig",
    "ImageTextShardsDataset",
    "index_files",
]

log = logging.getLogger(__name__)

IMAGE_TEXT_SHARDS_STYLE = "olmocr"
"""Default style tag: the olmOCR pages' (the targets follow the olmOCR-mix convention)."""

IMAGE_TEXT_SHARDS_SPLITS = ("train", "heldout")
"""Splits that can be read. ``excluded`` rows (benchmark overlap and the like) never are."""

_REQUIRED_COLUMNS = ("shard", "offset", "size", "target", "kind", "split")


def index_files(dataset_path: str) -> List[str]:
    """The index of a dataset: ``index.parquet``, else the sorted ``index/part-*.parquet``.

    :param dataset_path: The dataset directory.

    :raises FileNotFoundError: If the directory has neither.
    """
    single = os.path.join(dataset_path, "index.parquet")
    if os.path.exists(single):
        return [single]
    parts = sorted(glob.glob(os.path.join(dataset_path, "index", "part-*.parquet")))
    if not parts:
        raise FileNotFoundError(
            f"{dataset_path}: no index.parquet and no index/part-*.parquet (image-text shards)"
        )
    return parts


@dataclass
class ImageTextShardsDatasetConfig(Config):
    """One image-text shards dataset (see the module docstring), or a selection of its rows."""

    dataset_path: str = ""
    """The dataset directory, holding the index and ``shards/``."""

    split: str = "train"
    """``train`` or ``heldout`` (rows kept out of training, for checks)."""

    kinds: Optional[Tuple[str, ...]] = None
    """Keep only rows whose ``kind`` is listed (``equation`` / ``table`` / ``page``). ``None``
    keeps every kind."""

    keep_column: Optional[str] = None
    """Name of a boolean index column; only rows where it is true are kept. ``None`` keeps every
    row of the split and kinds."""

    style: str = IMAGE_TEXT_SHARDS_STYLE
    """Style tag of the user turn, written as the bare ``"<style>:"``
    (:func:`~.pixmo_cap.style_tag_prompt`)."""

    max_crops: int = 8
    p_high_res: float = 0.0
    """Probability that a training example is cropped with :attr:`high_res_max_crops` instead of
    :attr:`max_crops`, drawn on the example's ``(row, epoch)`` stream. Zero draws nothing."""
    high_res_max_crops: int = 24
    max_sequence_length: Optional[int] = None
    """Tail-truncate the built sequence to this many tokens (the image block always fits; an
    example left without loss tokens is rejected, and the loader skips it)."""

    loss_token_weighting: str = "none"
    """``"none"`` weights every response token equally, like the other stage-1 OCR sources."""
    message_weight: Optional[float] = None
    """Scalar loss multiplier for this source."""
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)
    """Image token IDs of the selected language-model tokenizer (set by the mixture)."""
    message_format: SftMessageFormat = "qwen3"
    """``"qwen3"`` (the released Molmo2 chat layout) or ``"document"`` (plain pretraining
    documents, for a language model trained without a chat template)."""

    seed: int = 0

    def validate(self):
        if not self.dataset_path:
            raise OLMoConfigurationError("dataset_path must name an image-text shards directory")
        if self.split not in IMAGE_TEXT_SHARDS_SPLITS:
            raise OLMoConfigurationError(
                f"split must be one of {IMAGE_TEXT_SHARDS_SPLITS}, got {self.split!r}"
            )
        if self.kinds is not None and (len(self.kinds) == 0 or not all(self.kinds)):
            raise OLMoConfigurationError(
                "kinds=() would filter out every row; use None to keep every kind"
            )
        if self.keep_column is not None and not self.keep_column:
            raise OLMoConfigurationError("keep_column must be None or name a column")
        if not self.style:
            raise OLMoConfigurationError("style must name a style tag")
        if self.max_crops < 1:
            raise OLMoConfigurationError("max_crops must be at least 1")
        if not 0.0 <= self.p_high_res <= 1.0:
            raise OLMoConfigurationError("p_high_res must be in [0, 1]")
        if self.high_res_max_crops < 1:
            raise OLMoConfigurationError("high_res_max_crops must be at least 1")

    def build(self, tokenizer) -> "ImageTextShardsDataset":
        self.validate()
        return ImageTextShardsDataset(self, tokenizer)


class ImageTextShardsDataset(EpochSeededExamples):
    """Map-style dataset over the selected rows of an image-text shards dataset."""

    def __init__(self, config: ImageTextShardsDatasetConfig, tokenizer):
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        self.config = config
        self.tokenizer = tokenizer
        paths = index_files(config.dataset_path)
        filters: List[Tuple[str, str, Any]] = [("split", "=", config.split)]
        if config.kinds is not None:
            filters.append(("kind", "in", list(config.kinds)))
        if config.keep_column is not None:
            filters.append((config.keep_column, "=", True))
        tables = []
        total = 0
        for path in paths:
            # Checked per file: the parts of a large index are written by separate tasks.
            parquet = pq.ParquetFile(path)
            names = set(parquet.schema_arrow.names)
            total += parquet.metadata.num_rows
            missing = [c for c in _REQUIRED_COLUMNS if c not in names]
            if config.keep_column is not None and config.keep_column not in names:
                missing.append(config.keep_column)
            if missing:
                raise OLMoConfigurationError(f"{path} has no column(s) {missing}")
            # One file at a time, in name order: the row order must not depend on threads.
            table = pq.read_table(
                path, columns=["shard", "offset", "size", "target"], filters=filters
            )
            tables.append(
                pa.table(
                    {
                        # A few thousand shard names over millions of rows.
                        "shard": pc.dictionary_encode(table.column("shard").cast(pa.string())),
                        "offset": table.column("offset").cast(pa.int64()),
                        "size": table.column("size").cast(pa.int64()),
                        "target": table.column("target").cast(pa.large_string()),
                    }
                )
            )
        self._rows = pa.concat_tables(tables).unify_dictionaries()
        self._warned = 0
        log.info(
            "Image-text shards %s (%s): %d of %d rows kept (kinds=%s, keep_column=%s)",
            config.dataset_path,
            config.split,
            len(self),
            total,
            config.kinds,
            config.keep_column,
        )

    def __len__(self) -> int:
        return self._rows.num_rows

    # -- reading ---------------------------------------------------------------------------

    def row(self, i: int) -> Dict[str, Any]:
        """``shard``, ``offset``, ``size`` and ``target`` of kept row ``i``."""
        if not 0 <= i < len(self):
            raise IndexError(i)
        return self._rows.slice(i, 1).to_pylist()[0]

    def image_bytes(self, row: Dict[str, Any]) -> bytes:
        """The stored image file of ``row``, read with one ``pread``."""
        path = os.path.join(self.config.dataset_path, "shards", row["shard"])
        size = int(row["size"])
        fd = os.open(path, os.O_RDONLY)
        try:
            data = os.pread(fd, size, int(row["offset"]))
        finally:
            os.close(fd)
        if len(data) != size:
            raise RuntimeError(f"{path}: read {len(data)} of {size} bytes at {row['offset']}")
        return data

    def image(self, row: Dict[str, Any]):
        """The decoded image of ``row``; raises if the bytes are not an image."""
        from PIL import Image

        image = Image.open(io.BytesIO(self.image_bytes(row)))
        image.load()  # decode here, so that a damaged file fails inside `_build`
        return image

    def user_prompt(self) -> str:
        """The user turn: only the style tag, since there is no question."""
        return style_tag_prompt(self.config.style)

    # -- example ---------------------------------------------------------------------------

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Build row ``index``, deterministically skipping unusable rows (see the module
        docstring and :func:`~olmo_core.data.multimodal.sft_common.get_example_with_skip`)."""
        return get_example_with_skip(self, index, len(self))

    def _build(self, i: int, epoch: Optional[int] = None) -> Dict[str, np.ndarray]:
        cfg = self.config
        row = self.row(i)
        text = row["target"]
        if not text or not text.strip():
            raise ValueError(f"row {i} of {cfg.dataset_path} has an empty target")
        # Per (row, epoch): the high-resolution draw should vary by epoch.
        rng = self.epoch_rng(i, epoch)
        # One image, one (tag, target) turn: the shared message encoder builds exactly the
        # stage-1 single-branch layout (user header + image block + tag, then the response).
        seq = encode_sft_example(
            self.tokenizer,
            self.image(row),
            [(self.user_prompt(), text)],
            max_crops=cfg.max_crops,
            p_high_res=cfg.p_high_res,
            high_res_max_crops=cfg.high_res_max_crops,
            loss_token_weighting=cfg.loss_token_weighting,
            message_weight=cfg.message_weight,
            token_ids=cfg.token_ids,
            message_format=cfg.message_format,
            shuffle_rng=rng,
        )
        return truncate_for_format(seq, cfg.max_sequence_length, cfg.token_ids, cfg.message_format)
