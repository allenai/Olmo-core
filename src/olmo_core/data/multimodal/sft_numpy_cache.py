"""Text examples from an open-instruct numpy SFT cache, packed as open-instruct packs them.

open-instruct's text SFT (``olmo_core_finetune.py``) tokenizes the mixture once into a numpy
cache (``token_ids_part_*.npy``, ``labels_mask_part_*.npy`` and one ``(start, end)`` metadata
line per row) and packs it with :class:`~olmo_core.data.numpy_dataset.NumpyPackedFSLDataset`.
C11 does so with ``--row_aligned_parts`` and ``--document_boundaries_from_metadata``: each row
is one document whose boundaries come from the metadata, never from EOS. That matters because
the Olmo 3.5 template ends every assistant turn with EOS, so an EOS scan would split a
multi-turn row into documents, the later ones without their prompt.

:class:`SftNumpyCacheConfig` builds the same packed dataset (the configuration of open-instruct's
``_numpy_dataset_config``) and turns instance ``i`` into one text example of the multimodal
mixture, as :class:`~olmo_core.data.multimodal.pretraining_replay.ComposableTextReplayDataset`
does for mid-training text: the same tokens and label mask, with ``document_ids`` from the
instance's ``doc_lens``.
"""

from __future__ import annotations

import glob
import gzip
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

from olmo_core.config import Config
from olmo_core.data.numpy_dataset import (
    NumpyPackedFSLDataset,
    NumpyPackedFSLDatasetConfig,
)
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.data.types import LongDocStrategy
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.vision.molmo2_tokens import N_PATCHES_SQ, PATCH_DIM, POOL_H, POOL_W

from .pretraining_replay import MAX_DOCUMENT_SEGMENTS

__all__ = ["SftNumpyCacheConfig", "SftNumpyCacheDataset"]

TOKEN_IDS_NPY_GLOB = "token_ids_part_*.npy"
LABELS_MASK_NPY_GLOB = "labels_mask_part_*.npy"
TOKEN_IDS_METADATA_GLOB = "token_ids_part_*.csv.gz"
ROW_ALIGNED_PARTS_KEY = "row_aligned_parts"
"""File names and statistics key of open-instruct's ``numpy_dataset_conversion``."""


def check_row_aligned_cache(numpy_dir: str) -> None:
    """Raise unless ``numpy_dir`` was written with row-aligned parts: one metadata line per row
    (open-instruct's ``_check_row_aligned_cache``). A mid-row-cut cache has two lines for each
    cut row, which metadata boundaries would train as two documents.

    :param numpy_dir: The cache directory.
    """
    stats_path = os.path.join(numpy_dir, "dataset_statistics.json")
    with open(stats_path) as f:
        stats = json.load(f)
    if not stats["configuration"].get(ROW_ALIGNED_PARTS_KEY, False):
        raise OLMoConfigurationError(f"{numpy_dir} was not written with --row_aligned_parts")
    num_lines = 0
    for path in glob.glob(os.path.join(numpy_dir, TOKEN_IDS_METADATA_GLOB)):
        with gzip.open(path, "rt") as f:
            num_lines += sum(1 for _ in f)
    num_rows = stats["overall_statistics"]["total_instances"]
    if num_lines != num_rows:
        raise OLMoConfigurationError(
            f"{numpy_dir} has {num_lines:,} metadata lines for {num_rows:,} rows, so some row "
            "spans two parts"
        )


@dataclass
class SftNumpyCacheConfig(Config):
    """An open-instruct numpy SFT cache (plus any ``--extra_numpy_dirs``) as text examples."""

    numpy_dir: str
    """The cache directory open-instruct trained from."""
    work_dir: str
    """Where the packing indexes are written."""
    sequence_length: int = 65_536
    extra_numpy_dirs: List[str] = field(default_factory=list)
    """open-instruct's ``--extra_numpy_dirs``, in the same order."""
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig.dolma2)
    """open-instruct's ``to_oc_tokenizer_config``, which is ``dolma2()`` for the Olmo 3.5
    tokenizer."""
    document_masking: bool = True
    """Give each example ``document_ids``, one per row, as C11's ``DOCB=1`` trains."""
    max_document_segments: Optional[int] = MAX_DOCUMENT_SEGMENTS
    """Refuse an instance with more documents than the KDA kernel takes. Unlike the replay's
    cap, documents are never merged: that would change attention against the text run."""

    def dataset_config(self) -> NumpyPackedFSLDatasetConfig:
        """The packed dataset open-instruct's ``_numpy_dataset_config`` builds for this cache."""
        dirs = [self.numpy_dir, *self.extra_numpy_dirs]
        return NumpyPackedFSLDatasetConfig(
            tokenizer=self.tokenizer,
            work_dir=self.work_dir,
            paths=[os.path.join(d, TOKEN_IDS_NPY_GLOB) for d in dirs],
            expand_glob=True,
            label_mask_paths=[os.path.join(d, LABELS_MASK_NPY_GLOB) for d in dirs],
            generate_doc_lengths=True,
            long_doc_strategy=LongDocStrategy.truncate,
            sequence_length=self.sequence_length,
            use_array_if_local=False,
        )

    def build(self, tokenizer=None) -> "SftNumpyCacheDataset":
        """Build and prepare the dataset (``tokenizer`` is accepted for the mixture interface)."""
        del tokenizer
        check_row_aligned_cache(self.numpy_dir)
        dataset = self.dataset_config().build()
        assert isinstance(dataset, NumpyPackedFSLDataset)
        dataset.prepare()
        return SftNumpyCacheDataset(
            dataset,
            document_masking=self.document_masking,
            max_document_segments=self.max_document_segments,
        )


class SftNumpyCacheDataset:
    """Packed SFT instances as text examples (labels, loss weights, empty image arrays)."""

    def __init__(
        self,
        dataset: NumpyPackedFSLDataset,
        *,
        document_masking: bool = True,
        max_document_segments: Optional[int] = MAX_DOCUMENT_SEGMENTS,
    ):
        self.dataset = dataset
        self.sequence_length = dataset.sequence_length
        self.document_masking = document_masking
        self.max_document_segments = max_document_segments

    @property
    def fingerprint(self) -> str:
        """Identify the packed dataset and the conversion settings for loader resume checks."""
        settings = [self.dataset.fingerprint, "sft_numpy_cache", self.document_masking]
        return hashlib.sha256(json.dumps(settings).encode()).hexdigest()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get(index)

    def get(self, index: int, epoch: int = 0) -> dict[str, Any]:
        """
        Return packed instance ``index`` as a text example.

        :param index: Instance index, as open-instruct's data loader indexes the same dataset.
        :param epoch: Accepted for the mixture-loader interface; source ordering is external.
        """
        del epoch
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"{index} is out of bounds for an SFT cache of {len(self)} instances")
        instance = self.dataset[index]
        tokens = instance["input_ids"].numpy().astype(np.int64)
        is_target = instance["label_mask"].numpy().astype(np.bool_)
        # Position t predicts token t + 1.
        targets = is_target[1:]
        labels = np.full(len(tokens), -100, dtype=np.int64)
        labels[:-1] = np.where(targets, tokens[1:], -100)
        loss_masks = np.zeros(len(tokens), dtype=np.float32)
        loss_masks[:-1] = targets
        valid_instance = bool(instance.get("instance_mask", True))
        if not valid_instance:
            # As ComposableTextReplayDataset: no loss, but the L - 1 denominator weight stays.
            labels.fill(-100)
            loss_masks[:-1] = 1.0
        example: dict[str, Any] = {
            "input_ids": tokens,
            "labels": labels,
            "loss_masks": loss_masks,
            "position_ids": np.arange(len(tokens), dtype=np.int64),
            "token_type_ids": np.zeros(len(tokens), dtype=np.int64),
            "images": np.zeros((0, N_PATCHES_SQ, PATCH_DIM), dtype=np.float32),
            "pooled_patches_idx": np.full((0, POOL_H * POOL_W), -1, dtype=np.int64),
            "metadata": {"instance_filter_valid": valid_instance},
        }
        if self.document_masking:
            doc_lens = instance["doc_lens"].numpy().astype(np.int64)
            if int(doc_lens.sum()) != len(tokens):
                raise RuntimeError(f"Instance {index}: doc_lens do not cover the instance")
            if (
                self.max_document_segments is not None
                and len(doc_lens) > self.max_document_segments
            ):
                raise RuntimeError(
                    f"Instance {index} has {len(doc_lens):_} documents, over the "
                    f"{self.max_document_segments:_} the KDA kernel takes"
                )
            example["document_ids"] = np.repeat(np.arange(len(doc_lens), dtype=np.int64), doc_lens)
        return example
