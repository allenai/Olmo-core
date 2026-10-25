"""Configurable multimodal sources and supervised-loss mixtures."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from cached_path import cached_path

from olmo_core.config import Config
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds, prepare_molmo2_tokenizer

from .mixture_weights import sampling_weights_from_loss_mass

__all__ = [
    "MultimodalDatasetMixture",
    "MultimodalMixtureConfig",
    "MultimodalSourceConfig",
]


def _load_indices(path: str) -> np.ndarray:
    with cached_path(path).open("rb") as stream:
        if path.endswith(".npy"):
            indices = np.load(stream, allow_pickle=False)
        else:
            indices = np.asarray([int(line) for line in stream if line.strip()], dtype=np.int64)
    if indices.ndim != 1 or indices.dtype.kind not in ("i", "u") or not len(indices):
        raise ValueError(f"Selection {path!r} must contain a nonempty vector of integer indices")
    if np.any(indices < 0) or np.any(indices > np.iinfo(np.int64).max):
        raise ValueError(f"Selection {path!r} contains invalid indices")
    indices = indices.astype(np.int64, copy=False)
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"Selection {path!r} contains duplicate indices")
    return indices


class _SelectedDataset:
    """Select logical rows while retaining the base dataset's epoch-dependent formatting."""

    content_fingerprint_version = "selected-multimodal-dataset-v1"

    def __init__(self, dataset: Any, indices: np.ndarray, dataset_config: Config):
        if np.any(indices >= len(dataset)):
            raise ValueError("Selection contains an index outside the base dataset")
        self._dataset = dataset
        self.indices = indices
        self.config = getattr(dataset, "config", dataset_config)
        fingerprint = getattr(dataset, "content_fingerprint", None)
        if fingerprint is None:
            fingerprint = getattr(dataset, "fingerprint", None)
        if callable(fingerprint):
            fingerprint = fingerprint()
        if fingerprint is not None and (not isinstance(fingerprint, str) or not fingerprint):
            raise ValueError("Base dataset fingerprint must be a nonempty string")
        # The resume identity covers source configuration and row selection, not Python files.
        digest = hashlib.sha256(
            json.dumps(
                {"config": dataset_config.as_config_dict(), "fingerprint": fingerprint},
                sort_keys=True,
                allow_nan=False,
            ).encode()
        )
        digest.update(indices.astype("<i8", copy=False).tobytes())
        self.content_fingerprint = digest.hexdigest()

    def __len__(self) -> int:
        return len(self.indices)

    def _raw_index(self, index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise IndexError(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return int(self.indices[index])

    def __getitem__(self, index: int) -> Any:
        return self.get(index, 0)

    def get(self, index: int, epoch: int = 0) -> Any:
        """Return the selected base row using the current source epoch."""
        raw_index = self._raw_index(index)
        get = getattr(self._dataset, "get", None)
        return get(raw_index, epoch) if callable(get) else self._dataset[raw_index]

    def raw_image_references(self, index: int) -> tuple[Any, ...]:
        """Return image references from the selected base row."""
        return tuple(self._dataset.raw_image_references(self._raw_index(index)))


@dataclass
class MultimodalSourceConfig(Config):
    """Apply an optional prepared row selection to any map-style dataset config.

    The nested config must implement ``build(tokenizer)``. Use its concrete config class
    directly in a mixture when no row selection is needed.
    """

    dataset: Config
    selection_path: str | None = None
    """Newline-delimited integer indices or a one-dimensional integer ``.npy`` file."""
    excluded_selection_paths: list[str] = field(default_factory=list)
    """Selections from the same base dataset that must be disjoint from this selection."""
    selection_repeat: int = 1
    """Consecutive logical rows per prepared index; index ``i`` selects ``repeat*i + offset``."""

    def build(self, tokenizer: Any) -> Any:
        """Build the source and apply its row selection without reading example contents."""
        if type(self.selection_repeat) is not int or self.selection_repeat < 1:
            raise ValueError("selection_repeat must be a positive integer")
        if self.excluded_selection_paths and self.selection_path is None:
            raise ValueError("Excluded selections require selection_path")
        if self.selection_repeat != 1 and self.selection_path is None:
            raise ValueError("selection_repeat requires selection_path")
        build = getattr(self.dataset, "build", None)
        if not callable(build):
            raise TypeError("Source dataset config must implement build(tokenizer)")
        if self.selection_path is None:
            return build(tokenizer)
        indices = _load_indices(self.selection_path)
        for excluded_path in self.excluded_selection_paths:
            if np.intersect1d(indices, _load_indices(excluded_path), assume_unique=True).size:
                raise ValueError(
                    f"Selection {self.selection_path!r} overlaps excluded selection "
                    f"{excluded_path!r}"
                )
        dataset = build(tokenizer)
        if self.selection_repeat != 1:
            if np.any(indices >= len(dataset) // self.selection_repeat):
                raise ValueError("Selection contains an index outside the base dataset")
            indices = (
                indices[:, None] * self.selection_repeat + np.arange(self.selection_repeat)
            ).reshape(-1)
        return _SelectedDataset(dataset, indices, self.dataset)


@dataclass
class MultimodalDatasetMixture:
    """Built sources and example-sampling weights consumed by the mixture data loader."""

    names: list[str]
    datasets: list[Any]
    weights: list[float]
    tokenizer: Any
    token_ids: Molmo2TokenIds


@dataclass
class MultimodalMixtureConfig(Config):
    """Build named map-style sources and calibrate their supervised-loss allocation.

    ``sources`` accepts any config implementing ``build(tokenizer)``; concrete nested
    config classes are retained by :meth:`Config.as_config_dict`. Source insertion order
    defines the loader's source order and must remain unchanged when resuming a run.
    """

    tokenizer: TokenizerConfig
    sources: dict[str, Config] = field(default_factory=dict)
    target_loss_mass: dict[str, float] = field(default_factory=dict)
    """Desired relative supervised-loss mass, not example-sampling probabilities."""
    mean_loss_weight: dict[str, float] = field(default_factory=dict)
    """Measured mean sum of loss weights per example from each configured source."""
    model_vocab_size: int | None = None
    """Embedding-table size, including any reserved rows for image tokens."""
    tokenizer_revision: str | None = None
    tokenizer_cache_dir: str | None = None

    def validate(self):
        """Validate source names and loss targets without opening datasets."""
        if not self.sources or any(not isinstance(name, str) or not name for name in self.sources):
            raise ValueError("sources must contain nonempty source names")
        if set(self.sources) != set(self.target_loss_mass):
            raise ValueError("sources and target_loss_mass must contain identical names")
        if any(
            not math.isfinite(float(value)) or value <= 0
            for value in self.target_loss_mass.values()
        ):
            raise ValueError("target_loss_mass values must be finite and positive")
        if self.model_vocab_size is not None and self.model_vocab_size < self.tokenizer.vocab_size:
            raise ValueError(
                "model_vocab_size cannot be smaller than the base tokenizer vocabulary"
            )

    def build_tokenizer(self) -> tuple[Any, Molmo2TokenIds]:
        """Load the configured tokenizer and add image tokens within the model vocabulary."""
        from transformers import AutoTokenizer, PreTrainedConfig
        from transformers.models.auto.tokenization_auto import get_tokenizer_config

        if self.tokenizer.identifier is None:
            raise ValueError("Multimodal data requires a tokenizer identifier")
        metadata = get_tokenizer_config(
            self.tokenizer.identifier,
            revision=self.tokenizer_revision,
            cache_dir=self.tokenizer_cache_dir,
        )
        # Self-describing tokenizer repositories need not include a model config.json.
        config = PreTrainedConfig() if metadata.get("tokenizer_class") else None
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer.identifier,
            config=config,
            revision=self.tokenizer_revision,
            cache_dir=self.tokenizer_cache_dir,
            use_fast=False,
        )
        for name in ("eos_token_id", "pad_token_id"):
            if getattr(tokenizer, name) != getattr(self.tokenizer, name):
                raise ValueError(f"Loaded tokenizer {name} differs from the configured tokenizer")
        token_ids = prepare_molmo2_tokenizer(tokenizer, model_vocab_size=self.model_vocab_size)
        return tokenizer, token_ids

    def build_sources(self, tokenizer: Any, token_ids: Molmo2TokenIds) -> dict[str, Any]:
        """Build sources for calibration or training without changing their saved configs."""
        self.validate()

        def assign_token_ids(config: Config):
            if hasattr(config, "token_ids"):
                config.token_ids = token_ids  # type: ignore[attr-defined]

        datasets = {}
        for name, source in self.sources.items():
            config = source.copy()
            config.apply(assign_token_ids)
            build = getattr(config, "build", None)
            if not callable(build):
                raise TypeError(f"Source {name!r} must implement build(tokenizer)")
            dataset = build(tokenizer)
            if len(dataset) == 0:
                raise ValueError(f"Source {name!r} is empty")
            datasets[name] = dataset
        return datasets

    def sampling_weights(self) -> dict[str, float]:
        """Return calibrated example probabilities, requiring one mean per source."""
        self.validate()
        return sampling_weights_from_loss_mass(self.target_loss_mass, self.mean_loss_weight)

    def estimate_mean_loss_weights(
        self, samples_per_source: int = 128, seed: int = 0
    ) -> dict[str, float]:
        """Estimate calibration means from a bounded, reproducible sample of each source.

        Each source uses unique sampled row indices and formatting epoch zero. The same
        source receives the same probe indices when other sources are added or reordered.
        This returns new means without changing the config or requiring existing means.

        :param samples_per_source: Maximum number of examples to preprocess per source.
        :param seed: Nonnegative sampling seed.
        """
        if (
            isinstance(samples_per_source, bool)
            or not isinstance(samples_per_source, int)
            or samples_per_source <= 0
        ):
            raise ValueError("samples_per_source must be a positive integer")
        tokenizer, token_ids = self.build_tokenizer()
        means = {}
        for name, dataset in self.build_sources(tokenizer, token_ids).items():
            indices = np.random.default_rng(seed).choice(
                len(dataset), min(samples_per_source, len(dataset)), replace=False
            )
            get = getattr(dataset, "get", None)
            weights = []
            for index in indices:
                example = get(int(index), 0) if callable(get) else dataset[int(index)]
                weight = float(np.asarray(example["loss_masks"], dtype=np.float64).sum())
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError(
                        f"Source {name!r} row {index} has nonpositive or nonfinite loss weight"
                    )
                weights.append(weight)
            means[name] = float(np.mean(weights))
        return means

    def build(
        self,
        *,
        tokenizer: Any = None,
        token_ids: Molmo2TokenIds | None = None,
    ) -> MultimodalDatasetMixture:
        """Build a mixture, optionally reusing an already prepared tokenizer."""
        weights = self.sampling_weights()
        if (tokenizer is None) != (token_ids is None):
            raise ValueError("Provide both tokenizer and token_ids, or neither")
        if tokenizer is None:
            tokenizer, token_ids = self.build_tokenizer()
        assert token_ids is not None
        sources = self.build_sources(tokenizer, token_ids)
        return MultimodalDatasetMixture(
            names=list(sources),
            datasets=list(sources.values()),
            weights=[weights[name] for name in sources],
            tokenizer=tokenizer,
            token_ids=token_ids,
        )
