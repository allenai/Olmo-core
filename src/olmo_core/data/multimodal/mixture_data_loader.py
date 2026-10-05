"""Weighted multi-source mixture data loader for Molmo2 stage-1.

Drives the :class:`~olmo_core.train.Trainer` over several map-style multimodal datasets
sampled by per-source weights — the OLMo-core analogue of mm_olmo's ``SubMixture`` /
``IterableDatasetMixture``. Used for the caption + pointing + NLP stage-1 mixture.

Each epoch interleaves examples by drawing a source per slot from ``weights`` (multinomial)
and cycling through a shuffled permutation of that source, so each source contributes
roughly ``weight`` of the examples. Batches are reported in *tokens*
(``instances × pad_sequence_length``) like :class:`MultimodalDataLoader`.

With ``continuous_stream=True`` the loader instead follows Molmo2's continuous multinomial
source stream and per-source shuffled epochs (no artificial OLMo-core epoch boundary) and
packs through a buffered packer whose cursor and buffer are saved in :meth:`state_dict`, so a
checkpoint resumes the exact example stream.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch.distributed as dist

from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.exceptions import OLMoConfigurationError

from ..data_loader import DataLoaderBase, DataLoaderConfig
from .collator import MultimodalCollator
from .packing import (
    _select_buffered_pack_indices,
    example_crop_count,
    iter_dynamic_packs,
    iter_packs,
    pack_examples,
)
from .prefetch import prefetch_map
from .rng import make_random_state

if TYPE_CHECKING:
    from .alignment import MultimodalDatasetMixture

log = logging.getLogger(__name__)

DEFAULT_MAX_CONSECUTIVE_DATA_ERRORS = 10
DEFAULT_MAX_TOTAL_DATA_ERRORS = 1000
PACKED_LOADER_STATE_VERSION = 5
LEGACY_PACKED_LOADER_STATE_VERSIONS = (3, 4)

__all__ = ["MixtureDataLoader", "MixtureDataLoaderConfig"]

ExampleRef = Tuple[int, int, int]
LoadedExample = Tuple[ExampleRef, Optional[Dict[str, Any]], Optional[Exception]]


@DataLoaderConfig.register("multimodal_mixture")
@dataclass
class MixtureDataLoaderConfig(DataLoaderConfig["MixtureDataLoader"]):
    """Configuration for a stateful, token-batched multimodal mixture loader."""

    global_batch_size: int
    """Global batch size in padded token positions."""

    sequence_length: int
    """Fixed length to which each packed sequence is padded."""

    work_dir: str
    seed: int = 0
    epoch_instances: Optional[int] = None
    pack: bool = False
    pack_max_crops: Optional[int] = None
    pack_buffer_size: int = 48
    pack_image_weight: float = 30.0
    continuous_stream: bool = False
    """Use Molmo2's continuous source stream with an exactly resumable buffered packer
    (requires ``pack``, a positive ``pack_buffer_size`` and ``pack_max_crops``)."""
    batch_metadata: bool = False
    """Have the collator emit ``router_token_mask``, ``image_crop_counts`` and
    ``pooled_token_counts`` for the OLMoDDP multimodal train module."""
    est_tokens_per_example: int = 1400
    prefetch_workers: int = 0
    prefetch_max_in_flight: int | None = None
    """Maximum submitted-but-unconsumed examples per source group.

    Defaults to ``max(2 * prefetch_workers, 4)`` when prefetching is enabled.
    This bounds prepared-example read-ahead without changing packing or resume state.
    """
    defer_packed_image_copy: bool | None = None
    """Copy packed float32 images directly into the final batch; disabled by default."""
    max_consecutive_data_errors: int = DEFAULT_MAX_CONSECUTIVE_DATA_ERRORS
    max_total_data_errors: int = DEFAULT_MAX_TOTAL_DATA_ERRORS
    allow_legacy_state_without_dataset_fingerprints: bool = False
    source_groups: Optional[Dict[str, str]] = None
    """Optional assignment of every source name to an independently packed group."""
    group_sequence_quotas: Optional[Dict[str, int]] = None
    """Global padded sequence slots per group in every optimizer batch.

    Quotas must sum to ``global_batch_size / sequence_length`` and each be a positive
    multiple of the DP world size. These are exposure quotas, not supervised-token or
    loss-weight fractions. Both this field and ``source_groups`` must be set together.
    """

    def build(
        self,
        dataset: "MultimodalDatasetMixture",
        *,
        dp_process_group: Optional[dist.ProcessGroup] = None,
    ) -> "MixtureDataLoader":
        """Build the loader for the given data-parallel process group.

        :param dataset: Built source datasets, weights, and tokenizer.
        :param dp_process_group: The data-parallel process group.
        """
        dp_world_size = get_world_size(dp_process_group)
        if (
            self.sequence_length <= 0
            or self.global_batch_size <= 0
            or self.global_batch_size % (self.sequence_length * dp_world_size)
        ):
            raise OLMoConfigurationError(
                "global_batch_size must be a positive multiple of sequence_length "
                "times the data-parallel world size"
            )
        pad_token_id = dataset.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = dataset.tokenizer.eos_token_id
        if pad_token_id is None:
            raise OLMoConfigurationError("The tokenizer must define a pad or EOS token")

        return MixtureDataLoader(
            dataset.datasets,
            dataset.weights,
            MultimodalCollator(
                pad_token_id=pad_token_id,
                pad_sequence_length=self.sequence_length,
                batch_metadata=self.batch_metadata,
            ),
            work_dir=self.work_dir,
            global_batch_size=self.global_batch_size,
            seed=self.seed,
            epoch_instances=self.epoch_instances,
            pack=self.pack,
            pack_max_crops=self.pack_max_crops,
            pack_buffer_size=self.pack_buffer_size,
            pack_image_weight=self.pack_image_weight,
            continuous_stream=self.continuous_stream,
            est_tokens_per_example=self.est_tokens_per_example,
            prefetch_workers=self.prefetch_workers,
            prefetch_max_in_flight=self.prefetch_max_in_flight,
            defer_packed_image_copy=(
                False if self.defer_packed_image_copy is None else self.defer_packed_image_copy
            ),
            max_consecutive_data_errors=self.max_consecutive_data_errors,
            max_total_data_errors=self.max_total_data_errors,
            dp_world_size=dp_world_size,
            dp_rank=get_rank(dp_process_group),
            dataset_names=dataset.names,
            allow_legacy_state_without_dataset_fingerprints=(
                self.allow_legacy_state_without_dataset_fingerprints
            ),
            source_groups=self.source_groups,
            group_sequence_quotas=self.group_sequence_quotas,
        )


class _OrderedExampleStream(Iterator[Tuple[ExampleRef, Dict[str, Any]]]):
    """Load the exact Molmo2 mixture stream in order and track its consumed cursor."""

    def __init__(
        self,
        loader: "MixtureDataLoader",
        *,
        refs_consumed: int = 0,
    ):
        if refs_consumed < 0:
            raise OLMoConfigurationError("refs_consumed must be non-negative")
        self.loader = loader
        self.refs_consumed = refs_consumed
        self._results = iter(
            prefetch_map(
                loader._try_load_ref,
                loader._rank_refs_from_cursor(refs_consumed),
                num_workers=loader.prefetch_workers,
                max_in_flight=loader.prefetch_max_in_flight,
            )
        )

    def __iter__(self) -> "_OrderedExampleStream":
        return self

    def __next__(self) -> Tuple[ExampleRef, Dict[str, Any]]:
        while True:
            ref, example, error = next(self._results)
            # Count refs only as their ordered results are consumed. Thread-pool read-ahead is
            # intentionally excluded so a checkpoint can resume from this exact cursor.
            self.refs_consumed += 1
            if error is None:
                assert example is not None
                self.loader._consecutive_data_errors = 0
                return ref, example
            self.loader._handle_data_error(ref, error)

    def close(self):
        close = getattr(self._results, "close", None)
        if close is not None:
            close()


class _BufferedPackingIterator(Iterator[Dict[str, Any]]):
    """Stateful form of the buffered packer used by the mixture loader."""

    def __init__(
        self,
        example_stream: _OrderedExampleStream,
        *,
        seq_len: int,
        max_crops_per_pack: int,
        buffer_size: int,
        image_weight: float = 1.0,
        defer_image_copy: bool = False,
        buffer: Optional[Sequence[Tuple[ExampleRef, Dict[str, Any]]]] = None,
        packs_emitted: int = 0,
    ):
        self.example_stream = example_stream
        self.seq_len = seq_len
        self.max_crops_per_pack = max_crops_per_pack
        self.buffer_size = buffer_size
        self.image_weight = image_weight
        self.defer_image_copy = defer_image_copy
        self.buffer = list(buffer or [])
        self.packs_emitted = packs_emitted

    def __iter__(self) -> "_BufferedPackingIterator":
        return self

    def __next__(self) -> Dict[str, Any]:
        while True:
            ref, example = next(self.example_stream)
            length = len(example["input_ids"])
            crops = example_crop_count(example)
            token_granularity = max(1, self.seq_len // 512)
            at_token_capacity = (length + token_granularity - 1) // token_granularity >= (
                self.seq_len + token_granularity - 1
            ) // token_granularity
            if at_token_capacity or crops > self.max_crops_per_pack:
                self.packs_emitted += 1
                return pack_examples([example], defer_image_copy=self.defer_image_copy)
            if len(self.buffer) < self.buffer_size:
                self.buffer.append((ref, example))
                continue

            selected = _select_buffered_pack_indices(
                [len(buffered_example["input_ids"]) for _, buffered_example in self.buffer],
                [example_crop_count(buffered_example) for _, buffered_example in self.buffer],
                self.seq_len,
                self.max_crops_per_pack,
                self.image_weight,
            )
            if not selected:
                raise RuntimeError(
                    "Buffered packer could not select an example within its constraints"
                )
            # Emit in buffer order, like Molmo2's DynamicSolver and :class:`DynamicPacker`.
            packed = [self.buffer[i][1] for i in sorted(selected)]
            for i in sorted(selected, reverse=True):
                self.buffer.pop(i)
            self.buffer.append((ref, example))
            self.packs_emitted += 1
            return pack_examples(packed, defer_image_copy=self.defer_image_copy)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "refs_consumed": self.example_stream.refs_consumed,
            "buffer_refs": [ref for ref, _ in self.buffer],
            "packs_emitted": self.packs_emitted,
        }

    def close(self):
        self.example_stream.close()


class MixtureDataLoader(DataLoaderBase):
    """Sample examples across multiple datasets by ``weights`` and collate into batches.

    :param datasets: the source datasets (each map-style, ``__getitem__`` -> example dict).
    :param weights: per-source sampling weights (need not sum to 1; normalized internally).
    :param collator: must have a fixed ``pad_sequence_length`` (token-based batching).
    :param global_batch_size: global batch size in *tokens* (= global instances × seq len).
    :param epoch_instances: number of (global) instances that make up one epoch; defaults to
        the sum of the source lengths.
    :param continuous_stream: Follow Molmo2's continuous multinomial source stream (sources
        epoch independently, ``total_batches`` is ``None``) and pack with an exactly resumable
        buffered packer. Requires ``pack=True``, a positive ``pack_buffer_size`` and
        ``pack_max_crops``. The default keeps the epoch-shuffled reference order and
        :func:`~olmo_core.data.multimodal.packing.iter_dynamic_packs` used by the Molmo2
        training scripts; checkpointed cursor state, dataset fingerprints, source epochs and
        ``defer_packed_image_copy`` apply only to this mode.
    :param prefetch_max_in_flight: Maximum prepared examples in flight per source group.
        Defaults to ``max(2 * prefetch_workers, 4)``. Ignored when prefetching is disabled.
    :param defer_packed_image_copy: Copy packed float32 images directly into the final batch.
    :param allow_legacy_state_without_dataset_fingerprints: Allow restoring version 3 or 4
        buffered-packing cursor state, which predates per-source content fingerprints. This
        remains enabled by default for existing recipes and emits a warning. New recipes that
        require exact content validation should disable it explicitly. State from before
        buffered cursor support is still replayed from the beginning for backwards
        compatibility; that fallback is not an exact content-validated resume.
    :param source_groups: Optional source-name to group-name mapping, covering all sources.
    :param group_sequence_quotas: Global padded sequence slots per group per optimizer batch.
        Each quota must be divisible by DP size, giving every rank at least one sequence
        from every group. Groups are packed independently and sampling weights only choose
        sources within a group. Explicit quotas do not support dynamic batch-size changes.
    """

    _epoch: Optional[int]

    def __init__(
        self,
        datasets: Sequence,
        weights: Sequence[float],
        collator: MultimodalCollator,
        *,
        work_dir,
        global_batch_size: int,
        seed: int = 0,
        epoch_instances: Optional[int] = None,
        pack: bool = False,
        pack_max_crops: Optional[int] = None,
        pack_buffer_size: int = 48,
        pack_image_weight: float = 30.0,
        continuous_stream: bool = False,
        est_tokens_per_example: int = 1400,
        prefetch_workers: int = 0,
        prefetch_max_in_flight: int | None = None,
        defer_packed_image_copy: bool = False,
        max_consecutive_data_errors: int = DEFAULT_MAX_CONSECUTIVE_DATA_ERRORS,
        max_total_data_errors: int = DEFAULT_MAX_TOTAL_DATA_ERRORS,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: Optional[int] = None,
        dataset_names: Optional[Sequence[str]] = None,
        allow_legacy_state_without_dataset_fingerprints: bool = True,
        source_groups: Optional[Dict[str, str]] = None,
        group_sequence_quotas: Optional[Dict[str, int]] = None,
    ):
        super().__init__(
            work_dir=work_dir,
            global_batch_size=global_batch_size,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            fs_local_rank=fs_local_rank,
        )
        if collator.pad_sequence_length is None:
            raise OLMoConfigurationError(
                "MixtureDataLoader requires the collator to have a fixed `pad_sequence_length`."
            )
        if len(datasets) != len(weights) or not datasets:
            raise OLMoConfigurationError(
                "datasets and weights must be non-empty and the same length"
            )
        if continuous_stream:
            if not pack or pack_buffer_size <= 0 or pack_max_crops is None:
                raise OLMoConfigurationError(
                    "continuous_stream requires pack=True, a positive pack_buffer_size and "
                    "pack_max_crops"
                )
            if not math.isfinite(pack_image_weight) or pack_image_weight < 0:
                raise OLMoConfigurationError("pack_image_weight must be finite and non-negative")
        elif defer_packed_image_copy:
            raise OLMoConfigurationError("defer_packed_image_copy requires continuous_stream=True")
        if prefetch_max_in_flight is not None and (
            isinstance(prefetch_max_in_flight, bool)
            or not isinstance(prefetch_max_in_flight, int)
            or prefetch_max_in_flight <= 0
        ):
            raise OLMoConfigurationError("prefetch_max_in_flight must be a positive integer")
        if not isinstance(defer_packed_image_copy, bool):
            raise OLMoConfigurationError("defer_packed_image_copy must be a boolean")
        self.datasets = list(datasets)
        if dataset_names is None:
            self.dataset_names = [str(i) for i in range(len(datasets))]
        elif len(dataset_names) != len(datasets):
            raise OLMoConfigurationError(
                "dataset_names must be the same length as datasets when provided"
            )
        else:
            self.dataset_names = list(dataset_names)
        self.continuous_stream = continuous_stream
        self.dataset_fingerprints = [
            self._dataset_fingerprint(dataset, name) if continuous_stream else None
            for dataset, name in zip(self.datasets, self.dataset_names)
        ]
        self.allow_legacy_state_without_dataset_fingerprints = (
            allow_legacy_state_without_dataset_fingerprints
        )
        w = np.asarray(weights, dtype=np.float64)
        if continuous_stream and (
            not np.isfinite(w).all() or (w <= 0).any() or not math.isfinite(float(w.sum()))
        ):
            raise OLMoConfigurationError("Mixture weights must be finite and strictly positive")
        self.weights = (w / w.sum()).tolist()
        # Molmo2 casts the normalized rates to float32 in IterableDatasetMixture. Keep that
        # exact dtype because it determines the multinomial source-choice boundaries.
        self._sampling_weights = np.asarray(self.weights, dtype=np.float32)
        self.collator = collator
        self.seed = seed
        self.seq_len = collator.pad_sequence_length
        self.pack = pack
        self.pack_max_crops = pack_max_crops
        self.pack_buffer_size = pack_buffer_size
        self.pack_image_weight = pack_image_weight
        self.est_tokens_per_example = est_tokens_per_example
        self.prefetch_workers = prefetch_workers
        self.prefetch_max_in_flight = prefetch_max_in_flight
        self.defer_packed_image_copy = defer_packed_image_copy
        self.max_consecutive_data_errors = max_consecutive_data_errors
        self.max_total_data_errors = max_total_data_errors
        self._consecutive_data_errors = 0
        self._total_data_errors = 0
        self._sizes = [len(d) for d in self.datasets]
        self.epoch_instances = epoch_instances or sum(self._sizes)
        self._order: Optional[List[ExampleRef]] = None
        self._active_packer: Optional[_BufferedPackingIterator] = None
        self._packing_state: Optional[Dict[str, Any]] = None
        self._group_loaders: Dict[str, MixtureDataLoader] = {}
        self.source_groups = dict(source_groups or {})
        self.group_sequence_quotas = dict(group_sequence_quotas or {})
        if source_groups is not None or group_sequence_quotas is not None:
            if not self.source_groups or not self.group_sequence_quotas:
                raise OLMoConfigurationError(
                    "source_groups and group_sequence_quotas must be nonempty and set together"
                )
            if len(set(self.dataset_names)) != len(self.dataset_names) or set(
                self.source_groups
            ) != set(self.dataset_names):
                raise OLMoConfigurationError(
                    "source_groups must assign every unique dataset name exactly once"
                )
            if any(
                not isinstance(group, str) or not group for group in self.source_groups.values()
            ) or set(self.source_groups.values()) != set(self.group_sequence_quotas):
                raise OLMoConfigurationError(
                    "source_groups must cover exactly the nonempty group_sequence_quotas names"
                )
            if (
                any(
                    isinstance(quota, bool)
                    or not isinstance(quota, int)
                    or quota <= 0
                    or quota % self.dp_world_size
                    for quota in self.group_sequence_quotas.values()
                )
                or sum(self.group_sequence_quotas.values()) * self.seq_len != global_batch_size
            ):
                raise OLMoConfigurationError(
                    "Group sequence quotas must be positive multiples of the DP world size "
                    "and sum to global_batch_size / sequence_length"
                )
            # Reuse the ordinary source stream, error handling, packing, and checkpointing.
            # Sorting group names makes dictionary insertion order irrelevant. Source order
            # within each group remains the explicitly configured dataset order.
            for group in sorted(self.group_sequence_quotas):
                indices = [
                    i
                    for i, name in enumerate(self.dataset_names)
                    if self.source_groups[name] == group
                ]
                self._group_loaders[group] = MixtureDataLoader(
                    [self.datasets[i] for i in indices],
                    [self.weights[i] for i in indices],
                    self.collator,
                    work_dir=self.work_dir,
                    global_batch_size=self.group_sequence_quotas[group] * self.seq_len,
                    seed=self.seed,
                    epoch_instances=self.epoch_instances,
                    pack=self.pack,
                    pack_max_crops=self.pack_max_crops,
                    pack_buffer_size=self.pack_buffer_size,
                    pack_image_weight=self.pack_image_weight,
                    continuous_stream=self.continuous_stream,
                    est_tokens_per_example=self.est_tokens_per_example,
                    prefetch_workers=self.prefetch_workers,
                    prefetch_max_in_flight=self.prefetch_max_in_flight,
                    defer_packed_image_copy=self.defer_packed_image_copy,
                    max_consecutive_data_errors=self.max_consecutive_data_errors,
                    max_total_data_errors=self.max_total_data_errors,
                    dp_world_size=self.dp_world_size,
                    dp_rank=self.dp_rank,
                    fs_local_rank=self.fs_local_rank,
                    dataset_names=[self.dataset_names[i] for i in indices],
                    allow_legacy_state_without_dataset_fingerprints=(
                        self.allow_legacy_state_without_dataset_fingerprints
                    ),
                )

    @staticmethod
    def _dataset_fingerprint(dataset: Any, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Return a stable content identity advertised by a mixture source, if any.

        ``content_fingerprint`` is the preferred protocol. ``fingerprint`` is accepted for
        existing OLMo datasets, including ``PretrainingReplayDataset``. Fingerprint values
        must be non-empty strings so an accidentally unstable or unserializable value cannot
        silently weaken checkpoint validation.
        """
        fingerprint = None
        for attribute in ("content_fingerprint", "fingerprint"):
            fingerprint = getattr(dataset, attribute, None)
            if fingerprint is not None:
                break
        if fingerprint is None:
            return None
        if callable(fingerprint):
            fingerprint = fingerprint()
        if not isinstance(fingerprint, str) or not fingerprint:
            raise OLMoConfigurationError(
                f"Mixture dataset {dataset_name!r} advertises an invalid content fingerprint: "
                f"{fingerprint!r}"
            )

        version = getattr(dataset, "content_fingerprint_version", None)
        if version is None:
            version = getattr(dataset, "fingerprint_version", None)
        if callable(version):
            version = version()
        if version is not None and (not isinstance(version, str) or not version):
            raise OLMoConfigurationError(
                f"Mixture dataset {dataset_name!r} advertises an invalid fingerprint version: "
                f"{version!r}"
            )
        dataset_type = f"{type(dataset).__module__}.{type(dataset).__qualname__}"
        return {"type": dataset_type, "version": version, "value": fingerprint}

    def _validate_dataset_fingerprints(self, saved_fingerprints: Any) -> None:
        """Require every version-5 source identity to match the current mixture exactly."""
        if not isinstance(saved_fingerprints, (list, tuple)):
            raise OLMoConfigurationError(
                "Packed-loader version-5 state is missing its dataset_fingerprints list"
            )
        if len(saved_fingerprints) != len(self.dataset_fingerprints):
            raise OLMoConfigurationError(
                "Packed-loader resume state contains "
                f"{len(saved_fingerprints)} dataset fingerprints, but the current loader "
                f"has {len(self.dataset_fingerprints)} sources"
            )
        for dataset_name, saved, current in zip(
            self.dataset_names, saved_fingerprints, self.dataset_fingerprints
        ):
            if saved != current:
                raise OLMoConfigurationError(
                    f"Packed-loader dataset content fingerprint changed for source "
                    f"{dataset_name!r}: checkpoint has {saved!r}, current dataset has "
                    f"{current!r}"
                )

    @property
    def _global_instances(self) -> int:
        return self.global_batch_size // self.seq_len

    @DataLoaderBase.global_batch_size.setter  # type: ignore[attr-defined]
    def global_batch_size(self, new_global_batch_size: int):
        if (
            getattr(self, "_group_loaders", None)
            and new_global_batch_size != self.global_batch_size
        ):
            raise OLMoConfigurationError(
                "Explicit group sequence quotas do not support dynamic batch-size changes"
            )
        DataLoaderBase.global_batch_size.__set__(self, new_global_batch_size)  # type: ignore[attr-defined]

    @property
    def _rank_instances(self) -> int:
        return self.rank_batch_size // self.seq_len

    @property
    def total_batches(self) -> Optional[int]:
        if self.continuous_stream:
            # Molmo2's packed IterableDatasetMixture is infinite and the trainer is bounded
            # by steps. Returning None keeps one continuous source RNG/count stream for the
            # whole run instead of introducing artificial OLMo-core epoch boundaries.
            return None
        if self.pack:
            # Examples are packed several-per-sequence, so an epoch is fewer batches. Estimate
            # the pack count from the average real length (exact count is data-dependent; the
            # cycled packer keeps ranks in sync and the run is step-bounded regardless).
            est_packs = max(1, (self.epoch_instances * self.est_tokens_per_example) // self.seq_len)
            return est_packs // self._global_instances
        return self.epoch_instances // self._global_instances

    @property
    def total_data_errors(self) -> int:
        """Return the cumulative number of data errors consumed by this rank."""
        if self._group_loaders:
            return sum(loader.total_data_errors for loader in self._group_loaders.values())
        return self._total_data_errors

    def reshuffle(self, epoch: Optional[int] = None, **kwargs):
        if epoch is not None:
            self._epoch = epoch
        epoch = self._epoch if self._epoch is not None else 1
        # Sources that *sample* inside ``__getitem__`` (a few of an image's negatives, a render
        # size) need the epoch, or every epoch redraws the same subset and the rest of each pool
        # is never trained on: reshuffling references alone cannot rotate a within-example draw.
        # See :class:`~olmo_core.data.multimodal.sft_common.EpochSeededExamples`.
        for dataset in self.datasets:
            set_epoch = getattr(dataset, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch)
        if self._group_loaders:
            for loader in self._group_loaders.values():
                loader.reshuffle(epoch=epoch, **kwargs)
            return
        if self.continuous_stream:
            self._order = None
            return
        rng = np.random.RandomState(self.seed + epoch)
        # Number of example refs to draw. When packing, an epoch consumes ~all examples
        # (several per packed sequence), so draw a full epoch of examples; otherwise draw
        # exactly enough to fill ``total_batches`` of one-example-per-slot batches.
        n = (
            self.epoch_instances
            if self.pack
            else (self.total_batches or 0) * self._global_instances
        )
        # Per-source shuffled cycles (sampling within a source without replacement until
        # exhausted, then reshuffle — covers sources smaller than their sampled count).
        # References carry source epoch 0: this mode loads examples with ``dataset[index]``.
        perms = [rng.permutation(s) if s else np.array([], dtype=int) for s in self._sizes]
        cursors = [0] * len(self.datasets)
        src_choices = rng.choice(len(self.datasets), size=n, p=self.weights)
        order: List[ExampleRef] = []
        for src in src_choices:
            size = self._sizes[src]
            if size == 0:
                continue
            if cursors[src] >= size:
                perms[src] = rng.permutation(size)
                cursors[src] = 0
            order.append((int(src), int(perms[src][cursors[src]]), 0))
            cursors[src] += 1
        self._order = order

    def _iter_batches(self) -> Iterable[Dict[str, Any]]:
        if self._group_loaders:
            iterators = {
                group: iter(loader._iter_example_batches())
                for group, loader in self._group_loaders.items()
            }
            try:
                total_batches = self.total_batches
                steps = (
                    itertools.count(self.batches_processed)
                    if total_batches is None
                    else range(self.batches_processed, total_batches)
                )
                for _ in steps:
                    examples = []
                    group_names = []
                    for group, loader in self._group_loaders.items():
                        group_examples = next(iterators[group])
                        if self.total_data_errors > self.max_total_data_errors:
                            raise OLMoConfigurationError(
                                "Exceeded aggregate grouped-loader data error tolerance "
                                f"(total_data_errors={self.total_data_errors}, "
                                f"max_total_data_errors={self.max_total_data_errors})"
                            )
                        examples.extend(group_examples)
                        group_names.extend([group] * len(group_examples))
                    batch = self.collator(examples)
                    batch["loss_group_names"] = group_names
                    for loader in self._group_loaders.values():
                        loader.batches_processed += 1
                    yield batch
            finally:
                for iterator in iterators.values():
                    close = getattr(iterator, "close", None)
                    if close is not None:
                        close()
            return
        for examples in self._iter_example_batches():
            yield self.collator(examples)

    def _iter_example_batches(self) -> Iterable[List[Dict[str, Any]]]:
        """Yield uncollated sequence batches, also reused by explicit group quotas."""
        ri = self._rank_instances
        if self.continuous_stream:
            packer = self._build_buffered_packer()
            self._active_packer = packer
            try:
                if self._packing_state is None:
                    packs_to_replay = self.batches_processed * ri
                    if packs_to_replay:
                        log.warning(
                            "Packed-loader checkpoint has no cursor state; replaying %d "
                            "previously consumed packs once for backwards compatibility",
                            packs_to_replay,
                        )
                    for _ in range(packs_to_replay):
                        next(packer)
                while True:
                    yield [next(packer) for _ in range(ri)]
            finally:
                self._packing_state = self._buffered_packing_state(packer)
                self._active_packer = None
                packer.close()
            return

        if self._order is None:
            raise RuntimeError("call reshuffle() before iterating")
        n_batches = self.total_batches or 0
        if self.pack:
            rank_refs = self._order[self.dp_rank :: self.dp_world_size]
            gen = self._pack_stream(rank_refs)
            for _ in range(self.batches_processed * ri):  # resume: replay consumed packs
                next(gen)
            for _ in range(self.batches_processed, n_batches):
                yield [next(gen) for _ in range(ri)]
            return
        gi = self._global_instances
        for b in range(self.batches_processed, n_batches):
            global_slice = self._order[b * gi : (b + 1) * gi]
            rank_slice = global_slice[self.dp_rank * ri : (self.dp_rank + 1) * ri]
            ref_iter: Iterator = itertools.chain(rank_slice, itertools.cycle(self._order))
            examples = [self._load_example(ref_iter) for _ in range(ri)]
            yield examples

    def _try_load_example(self, ref) -> Dict[str, Any]:
        src_idx, example_idx, source_epoch = ref
        dataset = self.datasets[src_idx]
        get = getattr(dataset, "get", None) if self.continuous_stream else None
        ex = get(example_idx, source_epoch) if get is not None else dataset[example_idx]
        out = dict(ex)
        out["_source_name"] = self.dataset_names[src_idx]
        return out

    def _try_load_ref(self, ref: ExampleRef) -> LoadedExample:
        """Load one specific ref without mutating error counters in a worker thread."""
        try:
            return ref, self._try_load_example(ref), None
        except Exception as error:
            return ref, None, error

    def _rank_refs_from_cursor(self, refs_consumed: int = 0) -> Iterator[ExampleRef]:
        """Yield this rank's slice of Molmo2's continuous global reference stream.

        Source selection exactly follows ``IterableDatasetMixture``: MT19937 seeded once,
        float32 multinomial rates, a global per-source count, and a source permutation from
        ``make_random_state(seed, source_epoch, 1)``. Vectorized fast-forwarding makes exact
        checkpoint resume practical without storing or replaying preprocessed examples.
        """
        if refs_consumed < 0:
            raise OLMoConfigurationError("refs_consumed must be non-negative")
        if any(size <= 0 for size in self._sizes):
            raise OLMoConfigurationError("Every sampled mixture source must be non-empty")

        rng = np.random.RandomState(self.seed)
        counts = np.zeros(len(self.datasets), dtype=np.int64)
        global_cursor = self.dp_rank + refs_consumed * self.dp_world_size
        remaining = global_cursor
        while remaining:
            chunk_size = min(remaining, 1_000_000)
            choices = rng.choice(len(self.datasets), size=chunk_size, p=self._sampling_weights)
            counts += np.bincount(choices, minlength=len(self.datasets))
            remaining -= chunk_size

        shuffled_orders: List[Tuple[Optional[int], Optional[np.ndarray]]] = [
            (None, None) for _ in self.datasets
        ]
        while True:
            # Generate this rank's global slot plus the intervening slots assigned to the
            # other ranks. Batched choice is bit-identical to repeated scalar choice for
            # RandomState and leaves the MT19937 state at the same position.
            choices = np.asarray(
                rng.choice(
                    len(self.datasets),
                    size=self.dp_world_size,
                    p=self._sampling_weights,
                ),
                dtype=np.int64,
            )
            source = int(choices[0])
            source_count = int(counts[source])
            counts += np.bincount(choices, minlength=len(self.datasets))

            size = self._sizes[source]
            source_epoch = source_count // size
            shuffled_for, shuffled_order = shuffled_orders[source]
            if shuffled_for != source_epoch:
                shuffled_order = np.arange(size, dtype=np.int32)
                make_random_state(self.seed, source_epoch, 1).shuffle(shuffled_order)
                shuffled_orders[source] = (source_epoch, shuffled_order)
            assert shuffled_order is not None
            yield source, int(shuffled_order[source_count % size]), source_epoch

    def _handle_data_error(self, ref: ExampleRef, error: Exception):
        """Apply data-error tolerance in reference order on the consumer thread."""
        self._consecutive_data_errors += 1
        self._total_data_errors += 1
        src_idx, example_idx, source_epoch = ref
        context = (
            f"{self.dataset_names[src_idx]}[{example_idx}] at source epoch {source_epoch} "
            f"(consecutive_data_errors={self._consecutive_data_errors}, "
            f"total_data_errors={self._total_data_errors})"
        )
        if (
            self._consecutive_data_errors > self.max_consecutive_data_errors
            or self._total_data_errors > self.max_total_data_errors
        ):
            message = f"Exceeded data error tolerance loading {context}: {error!r}"
            error.add_note(message)
            log.error(message)
            raise error
        log.warning(
            "Skipping %s[%d] at source epoch %d after error "
            "(consecutive_data_errors=%d, total_data_errors=%d): %r",
            self.dataset_names[src_idx],
            example_idx,
            source_epoch,
            self._consecutive_data_errors,
            self._total_data_errors,
            error,
        )

    def _load_example(self, ref_iter: Iterator) -> Dict[str, Any]:
        """Load the next valid example, skipping refs that fail formatting (mm_olmo parity)."""
        while True:
            ref = next(ref_iter)
            try:
                out = self._try_load_example(ref)
                self._consecutive_data_errors = 0
                return out
            except Exception as error:
                self._handle_data_error(ref, error)

    def _example_stream(self, rank_refs: Sequence) -> Iterator[Dict[str, Any]]:
        """Infinite stream of example dicts for this rank: cycle the refs, load each example
        (heavy image preprocessing) on a background thread pool when ``prefetch_workers > 0``
        so it overlaps the GPU step, yielding in order to keep packing deterministic."""
        if not rank_refs:
            raise OLMoConfigurationError("No examples are available for this data-parallel rank")
        results = iter(
            prefetch_map(
                self._try_load_ref,
                itertools.cycle(rank_refs),
                num_workers=self.prefetch_workers,
                max_in_flight=self.prefetch_max_in_flight,
            )
        )
        try:
            while True:
                ref, example, error = next(results)
                if error is None:
                    assert example is not None
                    self._consecutive_data_errors = 0
                    yield example
                else:
                    self._handle_data_error(ref, error)
        finally:
            close = getattr(results, "close", None)
            if close is not None:
                close()

    def _pack_stream(self, refs: Sequence[ExampleRef]) -> Iterator[Dict[str, Any]]:
        """Packed-example stream over cycled ``refs`` (the default, epoch-shuffled mode).

        With ``pack_max_crops`` set this is mm_olmo's stage-2 packer: a buffered
        2D-knapsack over (text tokens, image crops) that freely mixes text-only and
        image examples in one pack. Without it, the legacy token-only next-fit
        (stage-1 behaviour) is used. ``flush=False`` because the ref stream is
        infinite (cycled).
        """
        stream = self._example_stream(refs)
        if self.pack_max_crops is not None:
            return iter_dynamic_packs(
                stream,
                self.seq_len,
                max_crops_per_pack=self.pack_max_crops,
                buffer_size=self.pack_buffer_size,
                image_weight=self.pack_image_weight,
                flush=False,
            )
        return iter_packs(stream, self.seq_len)

    def _build_buffered_packer(self) -> _BufferedPackingIterator:
        if self.pack_max_crops is None:
            raise OLMoConfigurationError("Buffered packing requires pack_max_crops")
        if self._packing_state is None:
            return _BufferedPackingIterator(
                _OrderedExampleStream(self),
                seq_len=self.seq_len,
                max_crops_per_pack=self.pack_max_crops,
                buffer_size=self.pack_buffer_size,
                image_weight=self.pack_image_weight,
                defer_image_copy=self.defer_packed_image_copy,
            )

        state = self._packing_state
        state_version = int(state.get("version", 0))
        if state_version in LEGACY_PACKED_LOADER_STATE_VERSIONS:
            if not self.allow_legacy_state_without_dataset_fingerprints:
                raise OLMoConfigurationError(
                    f"Packed-loader state version {state_version} predates dataset content "
                    "fingerprints. Set "
                    "allow_legacy_state_without_dataset_fingerprints=True to resume it "
                    "without content validation."
                )
            log.warning(
                "Restoring legacy packed-loader state version %d without validating dataset "
                "content fingerprints because "
                "allow_legacy_state_without_dataset_fingerprints=True",
                state_version,
            )
        if state_version == 3:
            if self.pack_image_weight != 1.0:
                raise OLMoConfigurationError(
                    "Version-3 packed-loader state implies pack_image_weight=1.0"
                )
        elif state_version not in (4, PACKED_LOADER_STATE_VERSION):
            raise OLMoConfigurationError(f"Unsupported packed-loader state version {state_version}")
        expected = {
            "epoch": self._epoch,
            "seed": self.seed,
            "dp_world_size": self.dp_world_size,
            "dp_rank": self.dp_rank,
            "rank_instances": self._rank_instances,
            "seq_len": self.seq_len,
            "pack_buffer_size": self.pack_buffer_size,
            "pack_max_crops": self.pack_max_crops,
            "dataset_sizes": self._sizes,
            "dataset_names": self.dataset_names,
            "weights": self.weights,
        }
        if state_version >= 4:
            expected["pack_image_weight"] = self.pack_image_weight
        if state_version >= PACKED_LOADER_STATE_VERSION:
            self._validate_dataset_fingerprints(state.get("dataset_fingerprints"))
        for key, expected_value in expected.items():
            if state.get(key) != expected_value:
                raise OLMoConfigurationError(
                    f"Packed-loader resume state has {key}={state.get(key)!r}, "
                    f"but the current loader requires {expected_value!r}"
                )

        refs_consumed = int(state["refs_consumed"])
        if refs_consumed < 0:
            raise OLMoConfigurationError(
                "Packed-loader resume state contains a negative reference cursor"
            )
        packs_emitted = int(state["packs_emitted"])
        expected_packs = self.batches_processed * self._rank_instances
        if packs_emitted != expected_packs:
            raise OLMoConfigurationError(
                f"Packed-loader resume state contains {packs_emitted} packs, but "
                f"batches_processed requires {expected_packs}"
            )
        buffer_refs = [self._ref_from_state(ref) for ref in state["buffer_refs"]]
        if len(buffer_refs) > self.pack_buffer_size:
            raise OLMoConfigurationError(
                "Packed-loader checkpoint buffer exceeds the configured buffer size"
            )
        buffer: List[Tuple[ExampleRef, Dict[str, Any]]] = []
        for ref in buffer_refs:
            try:
                buffer.append((ref, self._try_load_example(ref)))
            except Exception as error:
                error.add_note(
                    f"Failed to restore buffered example {ref}; exact packed-loader "
                    "resume requires every saved buffer ref to remain loadable"
                )
                raise

        return _BufferedPackingIterator(
            _OrderedExampleStream(self, refs_consumed=refs_consumed),
            seq_len=self.seq_len,
            max_crops_per_pack=self.pack_max_crops,
            buffer_size=self.pack_buffer_size,
            image_weight=self.pack_image_weight,
            defer_image_copy=self.defer_packed_image_copy,
            buffer=buffer,
            packs_emitted=packs_emitted,
        )

    @staticmethod
    def _ref_from_state(value: Any) -> ExampleRef:
        try:
            src_idx, example_idx, source_epoch = value
        except (TypeError, ValueError) as error:
            raise OLMoConfigurationError(
                f"Invalid example reference in packed-loader state: {value!r}"
            ) from error
        return int(src_idx), int(example_idx), int(source_epoch)

    def _buffered_packing_state(self, packer: _BufferedPackingIterator) -> Dict[str, Any]:
        state = packer.state_dict()
        state.update(
            {
                "version": PACKED_LOADER_STATE_VERSION,
                "epoch": self._epoch,
                "seed": self.seed,
                "dp_world_size": self.dp_world_size,
                "dp_rank": self.dp_rank,
                "rank_instances": self._rank_instances,
                "seq_len": self.seq_len,
                "pack_buffer_size": self.pack_buffer_size,
                "pack_max_crops": self.pack_max_crops,
                "pack_image_weight": self.pack_image_weight,
                "dataset_sizes": self._sizes,
                "dataset_names": self.dataset_names,
                "dataset_fingerprints": self.dataset_fingerprints,
                "weights": self.weights,
            }
        )
        return state

    def get_mock_batch(self) -> Dict[str, Any]:
        if self._group_loaders:
            examples = []
            group_names = []
            for group, loader in self._group_loaders.items():
                src = next(i for i, size in enumerate(loader._sizes) if size)
                for i in range(loader._rank_instances):
                    example = loader._try_load_example((src, i % loader._sizes[src], 0))
                    examples.append(
                        pack_examples([example], defer_image_copy=self.defer_packed_image_copy)
                        if self.pack
                        else example
                    )
                    group_names.append(group)
            batch = self.collator(examples)
            batch["loss_group_names"] = group_names
            return batch
        ri = max(self._rank_instances, 1)
        # Pull from the first non-empty source.
        src = next((i for i, s in enumerate(self._sizes) if s), 0)
        size = max(self._sizes[src], 1)
        if self.pack:
            refs = [(src, i % size, 0) for i in range(max(ri * 4, 4))]
            gen = (
                iter_packs(
                    self._example_stream(refs),
                    self.seq_len,
                    max_crops_per_pack=self.pack_max_crops,
                    buffer_size=self.pack_buffer_size,
                    image_weight=self.pack_image_weight,
                    defer_image_copy=self.defer_packed_image_copy,
                )
                if self.continuous_stream
                else self._pack_stream(refs)
            )
            return self.collator([next(gen) for _ in range(ri)])
        examples = [self.datasets[src][i % size] for i in range(ri)]
        return self.collator(examples)

    def global_num_tokens_in_batch(self, batch: Dict[str, Any]) -> Optional[int]:
        del batch
        return self.global_batch_size

    def state_dict(self) -> Dict[str, Any]:
        if self._group_loaders:
            return {
                "grouped_version": 1,
                "batches_processed": self.batches_processed,
                "epoch": self._epoch,
                "source_groups": self.source_groups,
                "group_sequence_quotas": self.group_sequence_quotas,
                "grouped_config": self._grouped_resume_config(),
                "groups": {
                    group: loader.state_dict() for group, loader in self._group_loaders.items()
                },
            }
        state: Dict[str, Any] = {
            "batches_processed": self.batches_processed,
            "epoch": self._epoch,
            "seed": self.seed,
            "consecutive_data_errors": self._consecutive_data_errors,
            "total_data_errors": self._total_data_errors,
        }
        if self._active_packer is not None:
            state["packing_state"] = self._buffered_packing_state(self._active_packer)
        elif self._packing_state is not None:
            state["packing_state"] = self._packing_state
        return state

    def _grouped_resume_config(self) -> Dict[str, Any]:
        """Identity checks for all grouped modes, including non-buffered loaders."""
        return {
            "seed": self.seed,
            "dp_world_size": self.dp_world_size,
            "dp_rank": self.dp_rank,
            "global_batch_size": self.global_batch_size,
            "seq_len": self.seq_len,
            "epoch_instances": self.epoch_instances,
            "pack": self.pack,
            "pack_buffer_size": self.pack_buffer_size,
            "pack_max_crops": self.pack_max_crops,
            "pack_image_weight": self.pack_image_weight,
            "est_tokens_per_example": self.est_tokens_per_example,
            "dataset_names": self.dataset_names,
            "dataset_sizes": self._sizes,
            "dataset_fingerprints": self.dataset_fingerprints,
            "weights": self.weights,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if self._group_loaders:
            if (
                state_dict.get("grouped_version") != 1
                or any(
                    state_dict.get(key) != expected
                    for key, expected in (
                        ("source_groups", self.source_groups),
                        ("group_sequence_quotas", self.group_sequence_quotas),
                        ("grouped_config", self._grouped_resume_config()),
                    )
                )
                or set(state_dict.get("groups", {})) != set(self._group_loaders)
            ):
                raise OLMoConfigurationError(
                    "Grouped-loader resume requires matching source groups and sequence quotas"
                )
            self.batches_processed = int(state_dict["batches_processed"])
            self._epoch = state_dict["epoch"]
            for group, loader in self._group_loaders.items():
                loader.load_state_dict(state_dict["groups"][group])
                if (
                    loader.batches_processed != self.batches_processed
                    or loader._epoch != self._epoch
                ):
                    raise OLMoConfigurationError("Grouped-loader child batch/epoch cursor mismatch")
            return
        if "grouped_version" in state_dict:
            raise OLMoConfigurationError("Cannot load grouped-loader state without group quotas")
        self.batches_processed = state_dict.get("batches_processed", 0)
        epoch = state_dict.get("epoch")
        self._epoch = None if epoch is None else int(epoch)
        self.seed = state_dict.get("seed", self.seed)
        self._consecutive_data_errors = state_dict.get("consecutive_data_errors", 0)
        self._total_data_errors = state_dict.get("total_data_errors", 0)
        self._packing_state = state_dict.get("packing_state")

    def reset(self):
        super().reset()
        self._packing_state = None
        for loader in self._group_loaders.values():
            loader.reset()
