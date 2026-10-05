"""FineVision instruction-tuning datasets (``HuggingFaceM4/FineVision``).

One loader for any FineVision config, since they all share the same row schema:

* ``images`` — a list of images.
* ``texts`` — a list of ``{"user": ..., "assistant": ...}`` turns; the user side holds the
  instruction, the assistant side the supervision target.
* ``source`` plus four quality signals with per-turn ratings and a per-row minimum:
  ``formatting``, ``visual_dependency``, ``image_correspondence`` and ``relevance``. The
  ``*_min`` columns are exposed as optional filters on :class:`FineVisionDatasetConfig`.

A row is assembled as ONE sequential conversation branch (loss on every assistant turn,
one EOS target at the end) by
:func:`~olmo_core.data.multimodal.message_sequence.encode_sft_example`, i.e. the same
qwen3 layout as every other stage-2 source: no BOS, the image token block(s) inside the
first user turn, ``Image {i+1}`` prefixes when a row carries several images.

Configs verified against the copies on weka (see :data:`FINEVISION_ROOT`):

===========================  =========  ================  =========================
config                       rows       ``<image>`` mark  notes
===========================  =========  ================  =========================
``visualwebinstruct(filtered)``  263,581  leading, 1/row  web visual instruction
``mavis_math_rule_geo``           99,986  leading, 1/row  synthetic geometry w/ CoT
``mavis_math_metagen``            87,348  **none**        synthetic math w/ CoT
``geo170k(align)``                35,297  **none**        geometry caption/alignment
``geo170k(qa)``                   12,101  **none**        geometry multiple-choice
===========================  =========  ================  =========================

Every one of those is single-turn with exactly one image per row. Note that three of them
carry **no** ``<image>`` marker at all: the image block is positioned from the ``images``
column, not from the marker, so both layouts work identically (any marker present is
stripped so it is never tokenized as literal text).

.. warning::
    ``image_correspondence_min`` is very low on the synthetic-geometry configs — 75% of
    ``geo170k(qa)`` and 63% of ``mavis_math_metagen`` rows score 1 — so a naive
    ``min_image_correspondence=4`` discards most of them. Prefer
    ``min_visual_dependency``, which is 4-5 for ~95% of those rows.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from olmo_core.config import Config

from .message_sequence import encode_sft_example
from .sequence_builder import example_rng
from .sft_common import (
    decode_pil_image,
    get_example_with_skip,
    load_hf_dataset,
    strip_image_placeholders,
    truncate_example,
)

__all__ = [
    "FineVisionDatasetConfig",
    "FineVisionDataset",
    "VisualWebInstructDatasetConfig",
    "FINEVISION_ROOT",
]

log = logging.getLogger(__name__)

FINEVISION_ROOT = "/weka/oe-training-default/mm-olmo/hf_datasets/HuggingFaceM4___FineVision"
"""Directory holding one subdirectory of parquet shards per downloaded FineVision config."""

_QUALITY_COLUMNS = {
    "min_formatting": "formatting_min",
    "min_visual_dependency": "visual_dependency_min",
    "min_image_correspondence": "image_correspondence_min",
    "min_relevance": "relevance_min",
}


@dataclass
class FineVisionDatasetConfig(Config):
    """Configuration for :class:`FineVisionDataset`."""

    config_name: str = "visualwebinstruct(filtered)"
    """FineVision config to load, e.g. ``"mavis_math_rule_geo"`` or ``"geo170k(qa)"``.
    Resolved against :attr:`root` unless :attr:`dataset_path` is set."""

    root: str = FINEVISION_ROOT
    """Directory containing one subdirectory per FineVision config."""

    dataset_path: Optional[str] = None
    """Explicit parquet directory / glob / file, or a ``save_to_disk`` Arrow directory.
    Overrides :attr:`root` + :attr:`config_name` when set."""

    split: str = "train"

    texts_column: str = "texts"
    """Column holding the list of ``{"user", "assistant"}`` turns."""

    images_column: str = "images"

    min_formatting: Optional[int] = None
    """Keep rows with ``formatting_min >=`` this (1-5; how well-formed the answer is)."""

    min_visual_dependency: Optional[int] = None
    """Keep rows with ``visual_dependency_min >=`` this (1-5; does the answer need the
    image?). Usually the most effective filter for image-grounded training."""

    min_image_correspondence: Optional[int] = None
    """Keep rows with ``image_correspondence_min >=`` this (1-5). See the module-level
    warning before using this on the synthetic-geometry configs."""

    min_relevance: Optional[int] = None
    """Keep rows with ``relevance_min >=`` this (1-5)."""

    max_crops: int = 8
    """Max high-res crops *per image*. Rows with several images cost a multiple of this."""

    max_images: int = 5
    """Max images per row (extra images are dropped, matching the stage-2 budget)."""

    max_sequence_length: int = 4096
    loss_token_weighting: str = "root_subsegments"
    seed: int = 0

    def resolved_path(self) -> str:
        """The directory this config will read.

        :returns: :attr:`dataset_path` if set, else ``root/config_name``.
        """
        if self.dataset_path is not None:
            return self.dataset_path
        return os.path.join(self.root, self.config_name)

    def build(self, tokenizer) -> "FineVisionDataset":
        """Construct the dataset.

        :param tokenizer: A Molmo2 chat tokenizer.
        """
        return FineVisionDataset(self, tokenizer)


@dataclass
class VisualWebInstructDatasetConfig(FineVisionDatasetConfig):
    """:class:`FineVisionDatasetConfig` pinned to ``visualwebinstruct(filtered)``."""

    config_name: str = "visualwebinstruct(filtered)"


class FineVisionDataset:
    """Map-style dataset yielding packed FineVision instruction examples."""

    def __init__(self, config: FineVisionDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self._data = load_hf_dataset(
            config.resolved_path(),
            config.split,
            keep_columns=[config.texts_column, config.images_column]
            + list(_QUALITY_COLUMNS.values()),
        )
        self._index = self._build_index()
        self._warned = 0

    def _build_index(self) -> Optional[np.ndarray]:
        """Row positions kept after the quality filters (``None`` when none are active)."""
        cfg = self.config
        active = {
            column: getattr(cfg, attr)
            for attr, column in _QUALITY_COLUMNS.items()
            if getattr(cfg, attr) is not None
        }
        if not active:
            return None

        keep = np.ones(len(self._data), dtype=bool)
        for column, threshold in active.items():
            if column not in self._data.column_names:
                log.warning("FineVision: no %r column; ignoring that filter", column)
                continue
            values = np.array(
                [-np.inf if v is None else float(v) for v in self._data[column]],
                dtype=np.float64,
            )
            keep &= values >= threshold
        index = np.nonzero(keep)[0]
        log.info(
            "FineVision[%s]: kept %d / %d rows after quality filtering (%s)",
            cfg.config_name,
            len(index),
            len(self._data),
            ", ".join(f"{c}>={t}" for c, t in active.items()),
        )
        return index

    def __len__(self) -> int:
        return len(self._data) if self._index is None else len(self._index)

    def _row(self, i: int) -> Dict:
        pos = int(i if self._index is None else self._index[i])
        return self._data[pos]

    def _build(self, i: int) -> Dict[str, np.ndarray]:
        cfg = self.config
        row = self._row(i)

        turns: List[Tuple[str, str]] = []
        for turn in row[cfg.texts_column] or []:
            # Any inline <image> marker is stripped: the image is supplied as an explicit
            # token block inside the first user turn instead. Several configs carry no
            # marker at all, which is equivalent here since the block comes from the
            # `images` column.
            user = strip_image_placeholders(turn.get("user"))
            assistant = (turn.get("assistant") or "").strip()
            if not user or not assistant:
                continue
            turns.append((user, assistant))
        if not turns:
            raise ValueError("no usable (user, assistant) turn in row")

        raw_images = row.get(cfg.images_column) or []
        pil_images = [decode_pil_image(im) for im in raw_images if im is not None]

        # One sequential conversation branch (turn k attends earlier turns).
        seq = encode_sft_example(
            self.tokenizer,
            pil_images,
            [turns],
            max_crops=cfg.max_crops,
            max_images=cfg.max_images,
            loss_token_weighting=cfg.loss_token_weighting,
            shuffle_rng=example_rng(cfg.seed, i),
        )
        return truncate_example(seq, cfg.max_sequence_length)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Build the example at ``index``, skipping ahead over unusable rows.

        See :func:`~olmo_core.data.multimodal.sft_common.get_example_with_skip`.
        """
        return get_example_with_skip(self, index, len(self))


# Backwards-compatible alias: the loader used to be VisualWebInstruct-specific.
VisualWebInstructDataset = FineVisionDataset
