"""Three-level captions for synthetic text-rich images (mm_olmo's ``figure_ocr`` group).

Ports mm_olmo's ``TextRichCaptionConfig`` (``olmo/data/text_rich_caption_datasets.py``), one of
the two OCR groups its molmo3 stage-1 mixture spends 0.075 on each
(``launch_scripts/train_molmo3_stage1.py``, ``_base_mixture``; the other is olmOCR-mix, see
:mod:`.olmocr`). One row is one rendered chart / diagram / document / graphic / table, and it
carries three captions of decreasing altitude, which become the three branches of one example:

===============  ======================  ==========================================
field            style                   what it is
===============  ======================  ==========================================
``high_level``   ``fig_caption_high``    one- or two-sentence summary (~150 chars)
``mid_level``    ``fig_caption_mid``     layout and content description (~700 chars)
``low_level``    ``fig_caption_low``     dense description quoting the text (2-4k chars)
===============  ======================  ==========================================

The style names are this repo's. mm_olmo calls them ``ocr_caption_<level>``; they were renamed
because all three are descriptions of a figure, not OCR of it, and the name should say so.

All three on one image is the point of the group: the model learns to read the same page at
several altitudes, and ``low_level``, the read-out of everything on the page, is the part closest
to transcription. As with every OCR source the user turn is the style tag alone, with no question.

Only the ``train`` split is ever read. The build holds 2,048 rows per category out as
``validation``; this is a training source and has no way to reach them. Images stay on disk: a
row holds an ``image_relpath`` under
:data:`~olmo_core.data.multimodal.paths.TEXT_RICH_CAPTION`, not pixels.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError

from .message_sequence import encode_sft_example
from .paths import TEXT_RICH_CAPTION
from .pixmo_cap import style_tag_prompt
from .pixmo_points import _load_split, _open_image
from .sft_common import EpochSeededExamples, get_example_with_skip, truncate_example

__all__ = [
    "CATEGORIES",
    "CAPTION_LEVELS",
    "level_style",
    "TextRichCaptionDatasetConfig",
    "TextRichCaptionDataset",
]

log = logging.getLogger(__name__)

#: The five image kinds the corpus is built from (mm_olmo ``CATEGORIES``).
CATEGORIES: Tuple[str, ...] = ("chart", "diagram", "doc", "graphic", "table")

#: The caption altitudes, coarse to fine, in the order mm_olmo emits them
#: (``TextRichCaptionConfig.format_example``).
CAPTION_LEVELS: Tuple[str, ...] = ("high_level", "mid_level", "low_level")


def level_style(level: str) -> str:
    """The style name of one caption level: ``high_level`` -> ``fig_caption_high``."""
    return f"fig_caption_{level.removesuffix('_level')}"


@dataclass
class TextRichCaptionDatasetConfig(Config):
    """One category of mm_olmo's ``text_rich_caption``."""

    category: str = "chart"
    """One of :data:`CATEGORIES`."""

    dataset_path: str = TEXT_RICH_CAPTION
    """Corpus root: ``hf/<category>`` holds the rows, ``<category>/<id>/<id>.png`` the images."""

    levels: Tuple[str, ...] = CAPTION_LEVELS
    """Caption levels to emit, each as its own branch. All three is what mm_olmo trains;
    narrowing this is an ablation, not a parity setting."""

    max_crops: int = 8
    max_sequence_length: Optional[int] = None
    """Tail-truncate the built sequence to this many tokens (see
    :func:`~.sft_common.truncate_example`). Three branches of a dense page run long, so set it
    to the training sequence length."""

    loss_token_weighting: str = "none"
    """``"none"`` weights every response token equally, like the stage-1 caption source."""
    message_weight: Optional[float] = None
    """Scalar loss multiplier for this source (mm_olmo's ``ocr_weight``, unset in its stage 1)."""

    seed: int = 0

    def validate(self):
        if self.category not in CATEGORIES:
            raise OLMoConfigurationError(f"category={self.category!r} is not one of {CATEGORIES}")
        if not self.levels:
            raise OLMoConfigurationError("levels must name at least one caption level")
        unknown = [lvl for lvl in self.levels if lvl not in CAPTION_LEVELS]
        if unknown:
            raise OLMoConfigurationError(
                f"levels names {unknown}, which are not caption levels of this corpus "
                f"({list(CAPTION_LEVELS)})"
            )
        if len(set(self.levels)) != len(self.levels):
            raise OLMoConfigurationError(f"levels has duplicates: {self.levels}")

    @property
    def hf_path(self) -> str:
        """Where this category's rows live."""
        return os.path.join(self.dataset_path, "hf", self.category)

    def build(self, tokenizer) -> "TextRichCaptionDataset":
        self.validate()
        return TextRichCaptionDataset(self, tokenizer)


class TextRichCaptionDataset(EpochSeededExamples):
    """Map-style dataset over one category's train rows: image -> one branch per level."""

    def __init__(self, config: TextRichCaptionDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        data = _load_split(config.hf_path, "train")
        if hasattr(data, "keys"):
            # `_load_split` hands back the whole DatasetDict when the split is missing. Reading
            # "whatever is there" could pull in held-out rows, so a build without `train` fails.
            raise OLMoConfigurationError(
                f"{config.hf_path} has no 'train' split (it has {list(data.keys())})"
            )
        self._data = data
        self._warned = 0
        missing = [lvl for lvl in config.levels if lvl not in self._data.column_names]
        if missing:
            raise OLMoConfigurationError(
                f"{config.hf_path} has no {missing} column(s), so levels={list(config.levels)} "
                f"cannot be built; it ships {sorted(self._data.column_names)}"
            )
        log.info(
            "text_rich_caption/%s (train): %d images x %d level(s) %s",
            config.category,
            len(self._data),
            len(config.levels),
            list(config.levels),
        )

    def __len__(self) -> int:
        return len(self._data)

    def image_path(self, row: Dict[str, Any]) -> str:
        """Absolute path of a row's rendered image."""
        return os.path.join(self.config.dataset_path, row["image_relpath"])

    def turns(self, row: Dict[str, Any]) -> List[Tuple[str, str]]:
        """The ``(user, assistant)`` pairs of one row, one per configured level.

        A level whose caption is blank is dropped, since its branch would carry no loss tokens.

        :raises ValueError: If no level of the row has any text.
        """
        cfg = self.config
        turns: List[Tuple[str, str]] = []
        for level in cfg.levels:
            text = row[level]
            if not isinstance(text, str) or not text.strip():
                continue
            prompt = style_tag_prompt(level_style(level))
            turns.append((prompt, text))
        if not turns:
            raise ValueError(f"row {row.get('id')!r} has no non-empty caption in {cfg.levels}")
        return turns

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Build row ``index``, deterministically skipping rows with no usable target (a row with
        every caption blank, an undecodable image, a truncation that leaves no loss tokens), so
        they do not spend the mixture loader's error budget. Same policy as the other OCR
        sources; see :func:`~olmo_core.data.multimodal.sft_common.get_example_with_skip`."""
        return get_example_with_skip(self, index, len(self))

    def _build(self, i: int) -> Dict[str, np.ndarray]:
        cfg = self.config
        row = self._data[i]
        rng = self.epoch_rng(i)
        image = _open_image(self.image_path(row)).convert("RGB")
        seq = encode_sft_example(
            self.tokenizer,
            image,
            self.turns(row),
            max_crops=cfg.max_crops,
            loss_token_weighting=cfg.loss_token_weighting,
            message_weight=cfg.message_weight,
            shuffle_rng=rng,
        )
        if cfg.max_sequence_length is not None:
            seq = truncate_example(seq, cfg.max_sequence_length)
        return seq
