"""MolmoPoint-GUISyn GUI pointing for Molmo2 stage-1.

Ports mm_olmo's ``Molmo2SyntheticPointConfig`` (``molmo2_syn_point``,
``olmo/data/academic_datasets_manual.py``) over `allenai/MolmoPoint-GUISyn
<https://huggingface.co/datasets/allenai/MolmoPoint-GUISyn>`_: synthetic desktop, mobile and web
screenshots rendered from LLM-written HTML, with every visible UI element annotated by a name
("Close Google tab button"), five natural-language interaction intents ("Click the X to close
Google") and its bounding box. Train split, revision ``24bb3e99``: desktop 16,232 images (about
89 elements each, median 1920x1080), mobile 10,130 (about 22, median 393x844), web 9,830 (about
40, median 1440x900).

Each image is one multi-branch example; each sampled element is one branch whose answer is a
single point at the element's box center, labelled with its lowercased name. The question is,
with probability :attr:`GuiSynDatasetConfig.p_intent` (mm_olmo's 0.8), one of the element's
intents under its own ``"gui_point:"`` tag (:data:`GUI_POINT_STYLE`), and otherwise the
lowercased name under ``"pointing:"``. The intents are English requests, not object names, so
they get a tag of their own for the same reason CoSyn's questions get ``"cosyn_point:"``:
``pointing:`` keeps meaning "an object name follows".

Differences from mm_olmo, all deliberate:

* **At most** :attr:`~GuiSynDatasetConfig.max_elements` **elements per image and epoch**, drawn
  afresh each epoch. mm_olmo trains every element of an image as one example; at ~89 elements a
  desktop screenshot would not fit the stage-1 sequence length.
* **Element names are lowercased**, in the question and in the answer label, like every other
  stage-1 pointing label. mm_olmo keeps the name's case (``label_cased``) and, on intent
  branches, labels the answer with the intent text.
* **The resolution is the stage-1 one** (``max_crops=8``): a 1920x1080 desktop shot with
  12-pixel buttons is downscaled, so the smallest targets lose detail. mm_olmo's GUI-only
  mixtures force its high-resolution crops instead.

The data is mm_olmo's copy on weka (:data:`GUI_SYN_ROOT`), read straight from its Arrow shards
at a pinned revision, with no network access. Only the train split is read; the 256-image
validation split of each part stays held out.
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError

from .paths import MOLMO_DATA_DIR
from .pixmo_points import _build_example
from .pixmo_points_v2 import STAGE1_PROMPT_FAMILY, _rows_with_any
from .sft_common import EpochSeededExamples
from .sft_formatter import SftFormatter

__all__ = [
    "GUI_POINT_STYLE",
    "GUI_SYN_HF_REPO",
    "GUI_SYN_REVISION",
    "GUI_SYN_PARTS",
    "GUI_SYN_ROOT",
    "GuiSynDatasetConfig",
    "GuiSynDataset",
    "load_gui_syn_part",
]

log = logging.getLogger(__name__)

#: The style an element's intent is asked under: ``"gui_point: Click the X to close Google"``.
GUI_POINT_STYLE = "gui_point"

GUI_SYN_HF_REPO = "allenai/MolmoPoint-GUISyn"
#: The revision the copy on weka holds (the Hub's ``main`` as of 2026-09-30).
GUI_SYN_REVISION = "24bb3e990bb1e48796d80ade9dbc858dda695e75"
GUI_SYN_PARTS: Tuple[str, ...] = ("desktop", "mobile", "web")

#: mm_olmo's copy of GUISyn on weka, as ``datasets.load_dataset`` laid it out:
#: ``<part>/0.0.0/<revision>/molmo_point-gui_syn-<split>*.arrow``.
GUI_SYN_ROOT = os.path.join(MOLMO_DATA_DIR, "hf_datasets", "allenai___molmo_point-gui_syn")
_SHARD_PREFIX = "molmo_point-gui_syn"


def load_gui_syn_part(
    dataset_path: str, part: str, revision: str = GUI_SYN_REVISION, split: str = "train"
):
    """One part of GUISyn as a ``datasets.Dataset``, memory-mapped from its Arrow shards under
    ``dataset_path`` (laid out like :data:`GUI_SYN_ROOT`).

    :raises OLMoConfigurationError: if there is no shard of ``split`` for that part and revision.
    """
    import datasets

    shard_dir = os.path.join(dataset_path, part, "0.0.0", revision)
    files = sorted(
        glob.glob(os.path.join(shard_dir, f"{_SHARD_PREFIX}-{split}.arrow"))
        + glob.glob(os.path.join(shard_dir, f"{_SHARD_PREFIX}-{split}-*.arrow"))
    )
    if not files:
        raise OLMoConfigurationError(
            f"No {split!r} shards of {GUI_SYN_HF_REPO} ({part!r}, revision {revision}) under "
            f"{shard_dir!r}."
        )
    return datasets.concatenate_datasets([datasets.Dataset.from_file(f) for f in files])


@dataclass
class GuiSynDatasetConfig(Config):
    """mm_olmo ``Molmo2SyntheticPointConfig`` (``molmo2_syn_point``), stage-1 form."""

    dataset_path: str = GUI_SYN_ROOT
    """Where GUISyn is on disk (see :data:`GUI_SYN_ROOT`)."""

    revision: str = GUI_SYN_REVISION
    """The dataset revision to read."""

    parts: Tuple[str, ...] = GUI_SYN_PARTS
    """Which of ``desktop`` / ``mobile`` / ``web`` to train on."""

    max_elements: int = 16
    """Elements sampled per image and epoch (all of them when an image has fewer)."""

    p_intent: float = 0.8
    """Probability that an element with intents is asked by one of them (``gui_point:``) rather
    than by its name (``pointing:``); an element with no name is always asked by an intent
    (mm_olmo ``p_intent``)."""

    max_crops: int = 8
    loss_token_weighting: str = "none"
    """``"none"`` weights every response token equally, like the other v2 pointing sources."""
    message_weight: Optional[float] = None
    seed: int = 0

    def validate(self):
        if not self.parts:
            raise OLMoConfigurationError("parts must name at least one GUISyn part")
        unknown = [p for p in self.parts if p not in GUI_SYN_PARTS]
        if unknown:
            raise OLMoConfigurationError(f"parts {unknown} are not in {GUI_SYN_PARTS}")
        if len(set(self.parts)) != len(self.parts):
            raise OLMoConfigurationError(f"parts has duplicates: {self.parts}")
        if self.max_elements < 1:
            raise OLMoConfigurationError("max_elements must be at least 1")
        if not 0.0 <= self.p_intent <= 1.0:
            raise OLMoConfigurationError("p_intent must be in [0, 1]")

    def build(self, tokenizer) -> "GuiSynDataset":
        self.validate()
        return GuiSynDataset(self, tokenizer)


def _usable(name: Optional[str], intents: Optional[List[str]]) -> bool:
    return bool(name and name.strip()) or bool(intents)


class GuiSynDataset(EpochSeededExamples):
    """Map-style dataset over the GUISyn screenshots that have at least one usable element (a
    name or an intent)."""

    def __init__(self, config: GuiSynDatasetConfig, tokenizer):
        import datasets

        self.config = config
        self.tokenizer = tokenizer
        parts = [
            load_gui_syn_part(config.dataset_path, part, config.revision).select_columns(
                ["image", "annotation"]
            )
            for part in config.parts
        ]
        self._data = datasets.concatenate_datasets(parts)
        self._index = self._build_index()
        log.info(
            "GUISyn %s: %d of %d images have a usable element",
            "/".join(config.parts),
            len(self._index),
            len(self._data),
        )

    def _build_index(self) -> np.ndarray:
        import pyarrow.compute as pc

        def _usable_flat(flat) -> np.ndarray:
            name = pc.fill_null(pc.utf8_trim_whitespace(flat.field("name")), "")
            has_name = pc.invert(pc.equal(name, "")).to_numpy(zero_copy_only=False)
            n_intents = pc.fill_null(pc.list_value_length(flat.field("intent")), 0)
            return has_name | (n_intents.to_numpy(zero_copy_only=False) > 0)

        rows = _rows_with_any(self._data.data.column("annotation"), _usable_flat)
        return np.flatnonzero(rows)

    def __len__(self) -> int:
        return len(self._index)

    def format_row(
        self, annotations: List[Dict[str, Any]], image_size: Tuple[int, int], rng
    ) -> List[Dict[str, Any]]:
        """One image's branch messages: up to ``max_elements`` usable elements, each a formatter
        sub-example with one point at its box center (0-100 percent coordinates)."""
        cfg = self.config
        usable = [a for a in annotations if _usable(a.get("name"), a.get("intent"))]
        if not usable:
            raise ValueError("GUISyn row has no usable element")
        if len(usable) > cfg.max_elements:
            picked = rng.choice(len(usable), size=cfg.max_elements, replace=False)
            usable = [usable[int(j)] for j in picked]

        width, height = image_size
        messages: List[Dict[str, Any]] = []
        for anno in usable:
            name = (anno.get("name") or "").strip()
            intents = [str(x) for x in anno.get("intent") or []]
            point = np.array(
                [[anno["x_center"] / width * 100.0, anno["y_center"] / height * 100.0]]
            )
            msg: Dict[str, Any] = dict(points=point, point_scale=100, clip_points=True)
            # mm_olmo's draw order: the intent/name coin only when there is a name to fall back
            # on, then the intent itself.
            if intents and (not name or rng.random() < cfg.p_intent):
                question = intents[rng.randint(len(intents))]
                msg.update(style=GUI_POINT_STYLE, question=question, label=name or question)
            else:
                msg.update(style="pointing", label=name)
            messages.append(msg)
        return messages

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        cfg = self.config
        row = self._data[int(self._index[i])]
        image = row["image"]
        # Per (row, epoch): the element sample has to rotate across epochs.
        rng = self.epoch_rng(i)
        messages = self.format_row(row["annotation"], image.size, rng)
        fmt = SftFormatter(seed=cfg.seed, **STAGE1_PROMPT_FAMILY)
        branches = [fmt.format_turns(msg, index=i, rng=rng)[0] for msg in messages]
        return _build_example(
            self.tokenizer,
            image,
            branches,
            max_crops=cfg.max_crops,
            loss_token_weighting=cfg.loss_token_weighting,
            message_weight=cfg.message_weight,
            shuffle_rng=rng,
        )
