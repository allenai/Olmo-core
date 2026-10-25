"""Multi-image SFT datasets (mm_olmo all-v9 image slice).

Four source families whose examples carry several images:

* **Mantis-Instruct** (``mantis_instruct_{subset}_multi_only``) — multiple-choice QA
  over 2+ images (mm_olmo ``MantisInstructConfig``, non-flat, MC style).
* **CorrectionQa** (``correction_qa_multi_only_max5``) — human QA grouped by image
  set, 2-5 images (mm_olmo ``CorrectionQaConfig(multi_image_only=True, max_images=5)``).
* **CoSyn multi-document** (``cosyn_multidoc_{doc_type}_exp``) — several synthetic
  document images with explanation answers (mm_olmo ``CoSynMultiDocsConfig``).
* **PixMo multi-points** (``pixmo_multi_points``) — pointing/counting across several
  images (mm_olmo ``PixMoMultiPointsConfig``).

All route through the shared :class:`~olmo_core.data.multimodal.sft_formatter.SftFormatter`
and :func:`~olmo_core.data.multimodal.message_sequence.encode_sft_example` (which handles
``Image {i+1}`` prefixes, crop concatenation, and pooled-index offsets).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from os.path import join
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from olmo_core.config import Config

from .dataset_compat import load_from_disk_compat
from .detect_counting_question import is_pixmo_point_and_count_question
from .message_sequence import encode_sft_example
from .paths import ACADEMIC_DATASETS, PIXMO_DATASETS, TORCH_DATASETS
from .pixmo_ama import NO_POINT_PREFIX
from .sequence_builder import example_rng
from .sft_formatter import SftFormatter

__all__ = [
    "MantisInstructDatasetConfig",
    "MantisInstructDataset",
    "CorrectionQaDatasetConfig",
    "CorrectionQaDataset",
    "CoSynMultiDocDatasetConfig",
    "CoSynMultiDocDataset",
    "PixMoMultiPointsDatasetConfig",
    "PixMoMultiPointsDataset",
]

_CORRECTION_URL_PREFIX = (
    "https://explore-multimodal-datasets.s3.us-west-2.amazonaws.com/correction-urls"
)


def _open_images(paths) -> List[Image.Image]:
    return [Image.open(p).convert("RGB") for p in paths]


class _MultiImageSftDataset:
    """Shared __getitem__: format rows -> SftFormatter turns -> multi-image encode."""

    #: subclasses set these
    config: Any
    tokenizer: Any
    _formatter: SftFormatter
    _rows: List[Any]

    def __len__(self) -> int:
        return len(self._rows)

    def _format_row(self, row, rng: np.random.RandomState) -> Dict[str, Any]:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        rng = example_rng(self.config.seed, index)
        formatted = self._format_row(self._rows[index], rng)
        turns = self._formatter.format_branches(formatted, index=index, rng=rng)
        images = _open_images(formatted["image"])
        return encode_sft_example(
            self.tokenizer,
            images,
            turns,
            max_crops=self.config.max_crops,
            max_images=self.config.max_images,
            p_high_res=getattr(self.config, "p_high_res", 0.0),
            loss_token_weighting=self.config.loss_token_weighting,
            message_weight=self.config.message_weight,
            shuffle_rng=rng,
        )


@dataclass
class MantisInstructDatasetConfig(Config):
    """``mantis_instruct_{subset}_multi_only`` (mm_olmo ``MantisInstructConfig``)."""

    subset: str = "nlvr2"
    multi_image_only: bool = True
    max_crops: int = 8
    max_images: int = 5
    loss_token_weighting: str = "root_subsegments_root_tokens"
    message_weight: Optional[float] = None
    seed: int = 0

    def build(self, tokenizer) -> "MantisInstructDataset":
        return MantisInstructDataset(self, tokenizer)


class MantisInstructDataset(_MultiImageSftDataset):
    def __init__(self, config: MantisInstructDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._formatter = SftFormatter(seed=config.seed)
        ds = load_from_disk_compat(join(ACADEMIC_DATASETS, "mantis-instruct", config.subset))
        data = ds["train"] if hasattr(ds, "keys") and "train" in ds else ds
        self._data = data
        if config.multi_image_only:
            n_images = [len(x) for x in data["images"]]
            self._rows = [i for i, n in enumerate(n_images) if n > 1]
        else:
            self._rows = list(range(len(data)))

    @staticmethod
    def _shuffle_options(options: List[str], answer_idx: int, rng: np.random.RandomState):
        """mm_olmo ``MantisInstructConfig.shuffle_options``."""
        perm = rng.permutation(len(options))
        shuffled = [options[i] for i in perm]
        inverse = np.empty_like(perm)
        inverse[perm] = np.arange(len(perm))
        return shuffled, int(inverse[answer_idx])

    def _format_row(self, row_idx: int, rng: np.random.RandomState) -> Dict[str, Any]:
        ex = self._data[int(row_idx)]
        style = "mantis_instruct_mc"
        messages = []
        for i, question in enumerate(ex["mc_question"]):
            options, answer_idx = self._shuffle_options(
                ex["choices"][i], int(ex["correct_choice_idx"][i]), rng
            )
            messages.append(
                dict(question=question, options=options, answer_idx=answer_idx, style=style)
            )
        return dict(
            image=list(ex["images"]),
            message_list=messages,
            metadata=dict(example_id=ex["example_id"], subset=ex["subset"]),
        )


@dataclass
class CorrectionQaDatasetConfig(Config):
    """``correction_qa_multi_only_max5`` (mm_olmo ``CorrectionQaConfig``)."""

    multi_image_only: bool = True
    max_images: int = 5
    prefix_how_many: bool = True
    max_crops: int = 8
    loss_token_weighting: str = "root_subsegments_root_tokens"
    message_weight: Optional[float] = None
    seed: int = 0

    def build(self, tokenizer) -> "CorrectionQaDataset":
        return CorrectionQaDataset(self, tokenizer)


class CorrectionQaDataset(_MultiImageSftDataset):
    def __init__(self, config: CorrectionQaDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._formatter = SftFormatter(seed=config.seed)
        with open(join(PIXMO_DATASETS, "correction-qa", "train-records.json")) as f:
            records = json.load(f)
        grouped: Dict[Tuple[str, ...], List[dict]] = defaultdict(list)
        for record in records:
            if "imageUrls" in record:
                grouped[tuple(record["imageUrls"])].append(record)
            else:
                grouped[tuple([record["imageUrl"]])].append(record)
        rows: List[Any] = list(grouped.items())
        if config.multi_image_only:
            rows = [(k, v) for k, v in rows if len(k) > 1]
        if config.max_images:
            rows = [(k, v) for k, v in rows if len(k) <= config.max_images]
        self._rows = list(rows)

    def _format_row(self, row, rng: np.random.RandomState) -> Dict[str, Any]:
        image_urls, records = row
        dst_prefix = join(TORCH_DATASETS, "correction_images")
        images = [url.replace(_CORRECTION_URL_PREFIX, dst_prefix) for url in image_urls]
        messages = []
        for record in records:
            q, a = record["question"], record["answer"]
            if self.config.prefix_how_many and is_pixmo_point_and_count_question(q, a):
                q = NO_POINT_PREFIX[rng.randint(0, len(NO_POINT_PREFIX))] + q
            messages.append(dict(question=q, answer=a, style="correction_qa"))
        return dict(image=images, message_list=messages)


@dataclass
class CoSynMultiDocDatasetConfig(Config):
    """``cosyn_multidoc_{doc_type}_exp`` (mm_olmo ``CoSynMultiDocsConfig``)."""

    doc_type: str = "chart"
    use_exp: bool = True
    max_images: int = 5
    max_crops: int = 8
    loss_token_weighting: str = "root_subsegments_root_tokens"
    message_weight: Optional[float] = None
    seed: int = 0

    def build(self, tokenizer) -> "CoSynMultiDocDataset":
        return CoSynMultiDocDataset(self, tokenizer)


class CoSynMultiDocDataset(_MultiImageSftDataset):
    SRC = join(PIXMO_DATASETS, "pixmo_docs_multi")

    def __init__(self, config: CoSynMultiDocDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._formatter = SftFormatter(seed=config.seed)
        with open(join(self.SRC, f"{config.doc_type}_metadata_v3.json")) as f:
            metadata = json.load(f)
        rows = {k: v for k, v in metadata.items() if v is not None and v > 0}
        if config.max_images is not None:
            rows = {k: v for k, v in rows.items() if v <= config.max_images}
        self._rows = list(rows)

    def _format_row(self, folder: str, rng: np.random.RandomState) -> Dict[str, Any]:
        src = join(self.SRC, self.config.doc_type, folder)
        images = [f for f in os.listdir(src) if f.endswith(".png")]
        assert len(images) > 0
        images.sort(key=lambda x: int(x.split(".")[0].split("-")[-1]))
        images = [join(src, f) for f in images]
        with open(join(src, "qa.json")) as f:
            qas = json.load(f)["raw"]
        style = f"cosyn_{self.config.doc_type}"
        if self.config.use_exp:
            style += "_exp"
            message_list = [
                dict(
                    question=q["question"],
                    answer=q["answer"],
                    explanation=q["reasoning"],
                    style=style,
                )
                for q in qas
            ]
        else:
            message_list = [
                dict(question=q["question"], answer=q["answer"], style=style) for q in qas
            ]
        return dict(image=images, message_list=message_list, metadata=dict(image_paths=images))


@dataclass
class PixMoMultiPointsDatasetConfig(Config):
    """``pixmo_multi_points`` (mm_olmo ``PixMoMultiPointsConfig``)."""

    styles: Tuple[str, ...] = ("multi_image_pointing", "multi_image_point_then_count")
    max_images: int = 5
    max_crops: int = 8
    p_high_res: float = 0.0
    loss_token_weighting: str = "none"
    message_weight: Optional[float] = None
    seed: int = 0

    def build(self, tokenizer) -> "PixMoMultiPointsDataset":
        return PixMoMultiPointsDataset(self, tokenizer)


class PixMoMultiPointsDataset(_MultiImageSftDataset):
    def __init__(self, config: PixMoMultiPointsDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._formatter = SftFormatter(seed=config.seed)
        ds = load_from_disk_compat(join(PIXMO_DATASETS, "pixmo-multi-points"))
        self._data = ds["train"] if hasattr(ds, "keys") and "train" in ds else ds
        self._rows = list(range(len(self._data)))

    def _format_row(self, row_idx: int, rng: np.random.RandomState) -> Dict[str, Any]:
        ex = dict(self._data[int(row_idx)])
        ex["style"] = rng.choice(list(self.config.styles))
        ex["image"] = list(ex["images"])
        ex["point_scale"] = 100
        ex["clip_points"] = True
        ex["metadata"] = dict(image_paths=ex["images"])
        return ex
