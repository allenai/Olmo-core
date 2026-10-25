"""Adapt a language model's pretraining dataset for multimodal text replay."""

from __future__ import annotations

import hashlib
import json
import random
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import numpy as np

from olmo_core.config import Config
from olmo_core.data.composable import InstanceSource, InstanceSourceConfig
from olmo_core.data.numpy_dataset import (
    NumpyDatasetBase,
    NumpyFSLDataset,
    NumpyFSLDatasetConfig,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import file_exists, is_url, resource_path
from olmo_core.nn.vision.molmo2_tokens import N_PATCHES_SQ, PATCH_DIM, POOL_H, POOL_W

__all__ = [
    "ComposableTextReplayConfig",
    "ComposableTextReplayDataset",
    "PretrainingReplayConfig",
    "PretrainingReplayDataset",
]


def _checkpoint_dataset(checkpoint: str) -> tuple[dict[str, Any], str]:
    visited = set()
    while True:
        checkpoint = (
            checkpoint.rstrip("/") if is_url(checkpoint) else str(Path(checkpoint).resolve())
        )
        if checkpoint in visited:
            raise OLMoConfigurationError("Cycle in checkpoint pretraining_checkpoint references")
        visited.add(checkpoint)
        with resource_path(checkpoint, "config.json").open() as f:
            config = json.load(f)
        if parent := config.get("pretraining_checkpoint"):
            checkpoint = str(parent)
            continue
        dataset = config.get("dataset")
        if not isinstance(dataset, dict):
            raise OLMoConfigurationError(
                f"Checkpoint {checkpoint!r} does not record a pretraining dataset. "
                "Set PretrainingReplayConfig.dataset explicitly."
            )
        return dataset, checkpoint


def _align_metadata_with_selected_paths(config: NumpyFSLDatasetConfig) -> None:
    """
    Keep per-occurrence ``metadata`` aligned with a weighted source mixture that drops
    zero-allocation paths.

    A saved pretraining config may list one metadata entry per *unfiltered* path occurrence
    (the list ``data_paths.txt`` was written from), while the rebuilt mixture only keeps the
    occurrences that received tokens. Filter the sidecar the same way, in place, so the standard
    dataset builder sees one entry per path it loads.
    """
    metadata = config.metadata
    if config.source_mixture_config is None or not isinstance(metadata, list):
        return
    mixture = config.source_mixture_config.build(
        npdtype=config.get_dtype(), sequence_length=config.sequence_length
    )
    # The standard builder always drops zero-token occurrences from the mixture it loads.
    path_tokens = [item for outcome in mixture.sources for item in outcome.path_tokens]
    positions = [i for i, item in enumerate(path_tokens) if item.tokens > 0]
    if len(metadata) == len(path_tokens) != len(positions):
        config.metadata = [metadata[i] for i in positions]


@dataclass
class PretrainingReplayConfig(Config):
    """
    Reuse a checkpoint's fixed-length NumPy text dataset, or provide one explicitly.

    Inference reads only ``config.json`` and, when available, ``data_paths.txt``. Alignment
    checkpoints can refer to the original language model through ``pretraining_checkpoint``.
    Weighted source mixtures retain their standard allocation and sampling configuration;
    corpus preparation happens only in :meth:`build`.

    Composable datasets and document-isolated attention are not inferred. Supply an explicit
    :class:`~olmo_core.data.numpy_dataset.NumpyFSLDatasetConfig` for these checkpoints.
    """

    checkpoint: str | None = None
    """The language-model checkpoint or an alignment checkpoint that records its ancestry."""

    dataset: NumpyFSLDatasetConfig | None = None
    """An explicit dataset takes precedence over checkpoint metadata, including saved paths."""

    sequence_length: int | None = None
    """Optional replay window length; defaults to the pretraining window length."""

    work_dir: str | None = None
    """Optional override for the standard dataset preparation cache."""

    split: Literal["all", "train", "validation"] = "all"
    """Optionally withhold native windows from replay for a complementary validation split."""

    validation_size: int = 1024
    """Number of withheld windows. Used only when ``split`` is not ``all``."""

    split_seed: int = 6198
    """Seed for the bounded validation sample; train and validation must use the same seed."""

    def resolve_dataset(self) -> NumpyFSLDatasetConfig:
        """Resolve the dataset and tokenizer without reading or preparing token arrays."""
        return self._resolve_dataset()[0]

    def _resolve_dataset(self) -> tuple[NumpyFSLDatasetConfig, list[str] | None]:
        if self.split not in ("all", "train", "validation"):
            raise OLMoConfigurationError(f"Unknown replay split: {self.split!r}")
        if self.split != "all" and self.validation_size < 1:
            raise OLMoConfigurationError("Replay validation_size must be positive")
        saved_paths = None
        if self.dataset is not None:
            dataset = self.dataset.copy()
        else:
            if self.checkpoint is None:
                raise OLMoConfigurationError("Set either a replay checkpoint or a dataset")
            raw, checkpoint = _checkpoint_dataset(self.checkpoint)
            class_name = raw.get(Config.CLASS_NAME_FIELD)
            if class_name not in (
                None,
                f"{NumpyFSLDatasetConfig.__module__}.NumpyFSLDatasetConfig",
            ):
                raise OLMoConfigurationError(
                    f"Cannot infer text replay from {class_name!r}. "
                    "Set PretrainingReplayConfig.dataset explicitly."
                )
            dataset = NumpyFSLDatasetConfig.from_dict(raw)
            if file_exists(f"{checkpoint}/data_paths.txt"):
                saved_paths = resource_path(checkpoint, "data_paths.txt").read_text().splitlines()
                if not saved_paths or any(not path.strip() for path in saved_paths):
                    raise OLMoConfigurationError("Checkpoint data_paths.txt contains empty paths")
                if dataset.source_mixture_config is None:
                    # Saved paths are already expanded and permuted. Apply the same permutation
                    # only to sidecars, then disable the second path shuffle.
                    order = list(range(len(saved_paths)))
                    if dataset.source_permutation_seed is not None:
                        random.Random(dataset.source_permutation_seed).shuffle(order)
                    for field_name in ("metadata", "label_mask_paths"):
                        values = getattr(dataset, field_name)
                        if values is not None:
                            if len(values) != len(order) or (
                                field_name == "label_mask_paths" and dataset.expand_glob
                            ):
                                raise OLMoConfigurationError(
                                    f"Cannot align saved paths with {field_name}; "
                                    "set the replay dataset explicitly."
                                )
                            setattr(dataset, field_name, [values[i] for i in order])
                    dataset.paths = saved_paths
                    dataset.mix = None
                    dataset.mix_base_dir = None
                    dataset.expand_glob = False
                    dataset.source_permutation_seed = None

        if type(dataset) is not NumpyFSLDatasetConfig:
            raise OLMoConfigurationError("Text replay requires a NumpyFSLDatasetConfig")
        if dataset.generate_doc_lengths:
            raise OLMoConfigurationError(
                "Text replay does not support document-isolated attention "
                "(generate_doc_lengths=True)"
            )
        if self.sequence_length is not None:
            dataset.sequence_length = self.sequence_length
            target = dataset.max_target_sequence_length
            if target is not None and (
                self.sequence_length <= 0 or target % self.sequence_length != 0
            ):
                dataset.max_target_sequence_length = self.sequence_length
        if dataset.sequence_length < 2:
            raise OLMoConfigurationError("Text replay needs at least two tokens per instance")
        if self.work_dir is not None:
            dataset.work_dir = self.work_dir
        dataset.validate()
        if self.split != "all" and dataset.source_mixture_config is not None:
            raise OLMoConfigurationError(
                "Automatic replay validation requires an unweighted fixed-length dataset. "
                "Weighted mixtures can repeat physical windows; use split='all' with a "
                "separately disjoint validation corpus."
            )
        return dataset, saved_paths

    def build(self, tokenizer=None) -> PretrainingReplayDataset:
        """
        Build and prepare the standard dataset, then adapt its examples for multimodal training.

        :param tokenizer: Optional runtime tokenizer whose native EOS and padding IDs must agree.
        """
        config, saved_paths = self._resolve_dataset()
        if tokenizer is not None:
            for name in ("eos_token_id", "pad_token_id"):
                if getattr(tokenizer, name, None) != getattr(config.tokenizer, name):
                    raise OLMoConfigurationError(
                        f"Replay tokenizer {name} differs from pretraining"
                    )
        _align_metadata_with_selected_paths(config)
        dataset = config.build()
        # A path list cannot represent weighted allocations. The rebuilt mixture drops
        # zero-allocation occurrences; older checkpoints saved the unfiltered list, so accept
        # either, retaining allocation order, duplicates, and per-path metadata.
        if config.source_mixture_config is not None and saved_paths is not None:
            paths = list(map(str, dataset.paths))
            if paths != saved_paths:
                mixture = config.source_mixture_config.build(
                    npdtype=config.get_dtype(), sequence_length=config.sequence_length
                )
                unfiltered_paths = [
                    str(item.path) for outcome in mixture.sources for item in outcome.path_tokens
                ]
                if unfiltered_paths != saved_paths:
                    raise OLMoConfigurationError(
                        "Rebuilt source mixture differs from checkpoint data_paths.txt. "
                        "Set an explicit replay dataset if changing the corpus is intended."
                    )
        dataset.prepare()
        validation_indices = []
        if self.split != "all":
            if type(dataset) is not NumpyFSLDataset:
                raise OLMoConfigurationError("Replay splitting requires plain fixed-length windows")
            # Aliased files could otherwise put the same window in both splits.
            identities = []
            for path in dataset.paths:
                if is_url(str(path)):
                    parsed = urlsplit(str(path))
                    scheme = "object" if parsed.scheme in ("s3", "gs") else parsed.scheme
                    identities.append((scheme, parsed.netloc, parsed.path))
                else:
                    identities.append(("file", "", str(Path(path).resolve())))
            if len(set(identities)) != len(identities):
                raise OLMoConfigurationError(
                    "Replay splitting requires unique physical paths, including storage aliases"
                )
            if self.validation_size >= len(dataset):
                raise OLMoConfigurationError("Replay validation split must leave training windows")
            # Sampling a range stores only the held-out indices, not a corpus-sized permutation.
            validation_indices = random.Random(self.split_seed).sample(
                range(len(dataset)), self.validation_size
            )
        return PretrainingReplayDataset(dataset, config, self.split, validation_indices)


class PretrainingReplayDataset:
    """Fixed-length native token windows with next-token labels and empty image arrays."""

    def __init__(
        self,
        dataset: NumpyDatasetBase,
        config: NumpyFSLDatasetConfig,
        split: str = "all",
        validation_indices: Sequence[int] = (),
    ):
        self.dataset = dataset
        self.config = config.copy()
        self.sequence_length = config.sequence_length
        self.split = split
        self._validation_indices = tuple(validation_indices)
        self._exclusion_offsets = tuple(
            index - i for i, index in enumerate(sorted(validation_indices))
        )

    @property
    def fingerprint(self) -> str:
        """Identify the underlying arrays and replay settings for loader resume checks."""
        settings = self.config.as_dict(
            exclude_none=True, exclude_private_fields=True, exclude={"work_dir"}, json_safe=True
        )
        if self.split != "all":
            settings["replay_split"] = [self.split, self._validation_indices]
        encoded = json.dumps([self.dataset.fingerprint, settings], sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def __len__(self) -> int:
        if self.split == "validation":
            return len(self._validation_indices)
        if self.split == "train":
            return len(self.dataset) - len(self._validation_indices)
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get(index)

    def get(self, index: int, epoch: int = 0) -> dict[str, Any]:
        """
        Return a native token window without retokenization or injected special tokens.

        :param index: Index in the standard pretraining dataset.
        :param epoch: Accepted for the mixture-loader interface; source ordering is external.
        """
        del epoch
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"{index} is out of bounds for replay of size {len(self)}")
        if self.split == "validation":
            index = self._validation_indices[index]
        elif self.split == "train":
            index += bisect_right(self._exclusion_offsets, index)
        item = self.dataset[index]
        tokens = np.asarray(item["input_ids"], dtype=np.int64)
        labels = np.full(len(tokens), -100, dtype=np.int64)
        labels[:-1] = tokens[1:]
        loss_masks = np.ones(len(tokens), dtype=np.float32)
        loss_masks[-1] = 0.0
        if "label_mask" in item:
            loss_masks[:-1] = np.asarray(item["label_mask"], dtype=np.bool_)[1:]
            labels[loss_masks == 0] = -100
        valid_instance = bool(item.get("instance_mask", True))
        if not valid_instance:
            # Filtered rows contribute zero loss but retain their L-1 denominator weight,
            # matching OLMoDDP training and the multimodal weighted-loss convention.
            labels.fill(-100)
            loss_masks[:-1] = 1.0
        return {
            "input_ids": tokens,
            "labels": labels,
            "loss_masks": loss_masks,
            "position_ids": np.arange(len(tokens), dtype=np.int64),
            "token_type_ids": np.zeros(len(tokens), dtype=np.int64),
            "images": np.zeros((0, N_PATCHES_SQ, PATCH_DIM), dtype=np.float32),
            "pooled_patches_idx": np.full((0, POOL_H * POOL_W), -1, dtype=np.int64),
            "metadata": {**item.get("metadata", {}), "instance_filter_valid": valid_instance},
        }


MAX_DOCUMENT_SEGMENTS = 4096
"""FLA's KDA forward (``chunk_gated_delta_rule_fwd_h``) launches a grid of (cdiv(V, BV), N * HV),
with N the attention segments of the micro-batch and HV = 8 heads; CUDA caps that dimension at
65,535, so more than 8,191 segments fail with "Triton Error [CUDA]: invalid argument" (both mixed
256K runs at step 8,472). The text stage's cap, ``hero_mt_lc_256k.MAX_DOC_SEGMENTS``: a 2x margin."""


def cap_segments(lengths: np.ndarray, max_segments: int) -> np.ndarray:
    """
    Merge consecutive segment lengths so there are at most ``max_segments``.

    Degenerate rows (thousands of short EOS-separated segments) have their consecutive segments
    grouped until each group holds at least ceil(total / max_segments) tokens, the text stage's
    ``hero_mt_lc_256k._cap_segments``. Rows within the cap are returned unchanged.
    """
    if len(lengths) <= max_segments:
        return lengths
    min_group = -(-int(lengths.sum()) // max_segments)
    groups = []
    current = 0
    for length in lengths.tolist():
        current += length
        if current >= min_group:
            groups.append(current)
            current = 0
    if current:
        groups.append(current)
    return np.asarray(groups, dtype=lengths.dtype)


@dataclass
class ComposableTextReplayConfig(Config):
    """
    Text replay from a composable instance source of fixed-length text instances, such as the
    text team's merged mid-training + long-context stream (scaling-ladders
    ``merged_mt_lc.py``), so a multimodal mixture can train on exactly that text.

    Each instance becomes one text example, the way :class:`PretrainingReplayDataset` turns a
    NumPy window into one. An instance may carry:

    - ``label_mask``: which tokens are targets (``False`` positions are neither predicted nor
      counted in the loss denominator);
    - ``instance_mask``: ``False`` for a fully filtered instance, which contributes no loss but
      keeps its ``L - 1`` denominator weight (the OLMoDDP and multimodal convention);
    - an integer under :attr:`denominator_extra_key`: positions the text run counts in the loss
      denominator although they are not targets (a repetition-filtered chunk of a concatenated
      instance). The denominator only counts positions, so the first that many non-target
      positions are counted with zero loss, which gives the same loss as the text run.

    PAD tokens in the input are fed as EOS and never predicted when both ids are set: a model
    that never saw PAD has a near-zero PAD embedding, and long PAD runs make the first gradient
    non-finite (as the merged stage's own loader does).
    """

    source: InstanceSourceConfig
    """The instance source; it must produce instances of one fixed length."""
    work_dir: str
    """Where the source's prepared indexes live (the text run's ``data_loader.work_dir``)."""
    pad_token_id: int | None = None
    eos_token_id: int | None = None
    denominator_extra_key: str = "loss_denominator_extra_tokens"
    document_masking: bool = False
    """Split each instance into attention segments (``document_ids``) as the text team's 256K
    stage does (scaling-ladders ``hero_mt_lc_256k._document_lengths``): a segment ends after every
    EOS and, in instances of :attr:`chunked_source`, at every :attr:`chunk_length` boundary; a run
    of padding (PAD fed as EOS, never a target) stays one segment. Needs ``eos_token_id``."""
    chunked_source: int | None = None
    """Child index, in an interleaved source with ``locate``, whose instances are concatenated
    chunks (the MT side: 32 unrelated 8K chunks per 256K instance)."""
    chunk_length: int | None = None
    """Length of those chunks."""
    max_document_segments: int | None = MAX_DOCUMENT_SEGMENTS
    """With :attr:`document_masking`, the most segments an instance keeps: an instance over it has
    its consecutive segments merged greedily (see :func:`cap_segments`), as the text stage's
    ``hero_mt_lc_256k._cap_segments``. ``None`` keeps every segment."""

    def build(self, tokenizer=None) -> "ComposableTextReplayDataset":
        """Build the replay dataset (``tokenizer`` is accepted for the mixture interface)."""
        del tokenizer
        if (self.pad_token_id is None) != (self.eos_token_id is None):
            raise OLMoConfigurationError("Set both pad_token_id and eos_token_id, or neither")
        if self.document_masking and self.eos_token_id is None:
            raise OLMoConfigurationError("document_masking needs eos_token_id")
        if (self.chunked_source is None) != (self.chunk_length is None):
            raise OLMoConfigurationError("Set both chunked_source and chunk_length, or neither")
        return ComposableTextReplayDataset(
            self.source.build(self.work_dir),
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            denominator_extra_key=self.denominator_extra_key,
            document_masking=self.document_masking,
            chunked_source=self.chunked_source,
            chunk_length=self.chunk_length,
            max_document_segments=self.max_document_segments,
        )


class ComposableTextReplayDataset:
    """Text examples (labels, loss weights, empty image arrays) from composable instances."""

    def __init__(
        self,
        source: InstanceSource,
        *,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        denominator_extra_key: str = "loss_denominator_extra_tokens",
        document_masking: bool = False,
        chunked_source: int | None = None,
        chunk_length: int | None = None,
        max_document_segments: int | None = MAX_DOCUMENT_SEGMENTS,
    ):
        if source.sequence_length != source.max_sequence_length:
            raise OLMoConfigurationError("Composable text replay requires fixed-length instances")
        self.source = source
        self.sequence_length = source.sequence_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.denominator_extra_key = denominator_extra_key
        self.document_masking = document_masking
        self.chunked_source = chunked_source
        self.chunk_length = chunk_length
        self.max_document_segments = max_document_segments
        if document_masking and chunked_source is not None and not hasattr(source, "locate"):
            raise OLMoConfigurationError(
                "chunked_source needs an interleaved source that can locate its instances"
            )

    @property
    def fingerprint(self) -> str:
        """Identify the source and the conversion settings for loader resume checks."""
        settings = [
            self.source.fingerprint,
            self.pad_token_id,
            self.eos_token_id,
            self.denominator_extra_key,
        ]
        if self.document_masking:
            settings += ["document_masking", self.chunked_source, self.chunk_length]
        return hashlib.sha256(json.dumps(settings).encode()).hexdigest()

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get(index)

    def get(self, index: int, epoch: int = 0) -> dict[str, Any]:
        """
        Return instance ``index`` as a text example.

        :param index: Index in the instance source.
        :param epoch: Accepted for the mixture-loader interface; source ordering is external.
        """
        del epoch
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"{index} is out of bounds for replay of size {len(self)}")
        instance = self.source[index]
        tokens = np.array(instance["input_ids"], dtype=np.int64).reshape(-1)
        if len(tokens) != self.sequence_length:
            raise RuntimeError(
                f"Instance {index} has {len(tokens):_} tokens; expected {self.sequence_length:_}"
            )
        label_mask = instance.get("label_mask")
        is_target = (
            np.ones(len(tokens), dtype=np.bool_)
            if label_mask is None
            else np.array(label_mask, dtype=np.bool_).reshape(-1)
        )
        if len(is_target) != len(tokens):
            raise RuntimeError(f"Instance {index} has a label mask of the wrong length")
        # Position t predicts token t + 1.
        targets = is_target[1:].copy()
        extra_tokens: Any = instance.get(self.denominator_extra_key)
        extra = int(extra_tokens or 0)
        counted_only = np.zeros(len(tokens) - 1, dtype=np.bool_)
        if extra > 0:
            candidates = np.flatnonzero(~targets)
            if len(candidates) < extra:
                raise RuntimeError(
                    f"Instance {index} asks for {extra:_} extra denominator positions but has "
                    f"only {len(candidates):_} non-target positions"
                )
            counted_only[candidates[:extra]] = True
        token_is_target = is_target.copy()
        if self.pad_token_id is not None:
            is_pad = tokens == self.pad_token_id
            if is_pad.any():
                assert self.eos_token_id is not None
                tokens[is_pad] = self.eos_token_id
                targets &= ~is_pad[1:]
                token_is_target &= ~is_pad
        labels = np.full(len(tokens), -100, dtype=np.int64)
        labels[:-1] = np.where(targets, tokens[1:], -100)
        loss_masks = np.zeros(len(tokens), dtype=np.float32)
        loss_masks[:-1] = (targets | counted_only).astype(np.float32)
        valid_instance = bool(instance.get("instance_mask", True))
        if not valid_instance:
            # As PretrainingReplayDataset: no loss, but the L - 1 denominator weight stays.
            labels.fill(-100)
            loss_masks[:-1] = 1.0
        return {
            "input_ids": tokens,
            "labels": labels,
            "loss_masks": loss_masks,
            "position_ids": np.arange(len(tokens), dtype=np.int64),
            "token_type_ids": np.zeros(len(tokens), dtype=np.int64),
            "images": np.zeros((0, N_PATCHES_SQ, PATCH_DIM), dtype=np.float32),
            "pooled_patches_idx": np.full((0, POOL_H * POOL_W), -1, dtype=np.int64),
            "metadata": {"instance_filter_valid": valid_instance},
            **(
                {"document_ids": self._document_ids(index, tokens, token_is_target)}
                if self.document_masking
                else {}
            ),
        }

    def _document_ids(self, index: int, tokens: np.ndarray, is_target: np.ndarray) -> np.ndarray:
        """Per-token segment ids, the segments of ``hero_mt_lc_256k._document_lengths``: ends
        after every EOS and at every chunk boundary of a chunked instance; a run of padding (EOS
        fed for PAD, not a target) is one segment, but chunk boundaries inside it stay."""
        n = len(tokens)
        chunk = None
        if self.chunked_source is not None:
            source_index, _ = self.source.locate(index)  # type: ignore[attr-defined]
            if source_index == self.chunked_source:
                chunk = self.chunk_length
        is_eos = tokens == self.eos_token_id
        ends = np.zeros(n + 1, dtype=np.bool_)
        ends[1:] = is_eos
        if chunk is not None:
            ends[np.arange(chunk, n, chunk)] = True
        ends[0] = ends[n] = True
        # ``is_target`` is the token-level label mask (PAD excluded), as the text loader's.
        pad = is_eos & ~is_target
        inside_pad_run = np.zeros(n + 1, dtype=np.bool_)
        inside_pad_run[1:n] = pad[:-1] & pad[1:]
        if chunk is not None:
            inside_pad_run[np.arange(chunk, n, chunk)] = False
        ends &= ~inside_pad_run
        lengths = np.diff(np.flatnonzero(ends))
        if self.max_document_segments is not None:
            lengths = cap_segments(lengths, self.max_document_segments)
        return np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
