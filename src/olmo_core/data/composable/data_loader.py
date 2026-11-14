import dataclasses
import functools as ft
import logging
import math
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

import olmo_core.distributed.utils as dist_utils
from olmo_core.aliases import PathOrStr
from olmo_core.config import Config, StrEnum
from olmo_core.distributed.parallel import get_dp_process_group
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import get_default_device, roundrobin, threaded_generator

from ..collator import DataCollator
from ..data_loader import DataLoaderConfig, TextDataLoaderBase
from ..tokenizer import TokenizerConfig
from ..utils import (
    find_periodic_sequences,
    get_document_lengths,
    get_rng,
    iter_batched,
    memmap_to_write,
)
from .instance_source import InstanceSource
from .utils import (
    SEED_NOT_SET,
    as_tensor,
    build_global_indices,
    format_fname_from_fields,
    resolve_seed,
)

log = logging.getLogger(__name__)


@dataclass
class InstanceFilterConfig(Config):
    """Config for instance filtering."""

    repetition_max_period: int = 13
    repetition_min_period: int = 1
    repetition_max_count: int = 32


class ShuffleStrategy(StrEnum):
    """Defines how the data is shuffled."""

    inter_source = "inter_source"
    """Shuffle across all sources as if they were one big source."""
    intra_source = "intra_source"
    """
    Shuffle within each source, then concatenate the sources in order. This can be used to create
    a data curriculum.
    """
    interleaved_source = "interleaved_source"
    """
    Shuffle within each source and then interleave instances from each source.
    """


@DataLoaderConfig.register("composable")
@dataclass
class ComposableDataLoaderConfig(DataLoaderConfig["ComposableDataLoader"]):
    """
    A configuration class for building :class:`ComposableDataLoader` data loaders.
    """

    tokenizer: Optional[TokenizerConfig] = None
    global_batch_size: Optional[int] = None
    seed: int = dataclasses.field(default_factory=lambda: resolve_seed(SEED_NOT_SET))
    work_dir: Optional[str] = None
    shuffle: bool = True
    shuffle_strategy: Optional[ShuffleStrategy] = None
    sources_per_epoch: int = -1
    num_threads: Optional[int] = None
    num_workers: int = 0
    prefetch_factor: Optional[int] = None
    target_device_type: Optional[str] = None
    generate_doc_lengths: bool = False
    instance_filter_config: Optional[InstanceFilterConfig] = None
    display_source_visualization: bool = True

    def __post_init__(self, *args):
        del args
        if self.sources_per_epoch == 0 or self.sources_per_epoch < -1:
            raise OLMoConfigurationError(
                "'sources_per_epoch' must be -1 (for all sources) or a positive integer."
            )
        if not self.shuffle and self.shuffle_strategy is not None:
            raise OLMoConfigurationError("'shuffle_strategy' cannot be set if 'shuffle' is False.")

    def build(
        self,
        *sources: InstanceSource,
        collator: Optional[DataCollator] = None,
        work_dir: Optional[PathOrStr] = None,
        mesh: Optional[dist.DeviceMesh] = None,
        dp_process_group: Optional[dist.ProcessGroup] = None,
        tokenizer: Optional[TokenizerConfig] = None,
        global_batch_size: Optional[int] = None,
    ) -> "ComposableDataLoader":
        """
        Construct the :class:`ComposableDataLoader`.

        :param sources: The instance sources.
        :param collator: An optional data collator. If not provided, a default will be created.
        :param work_dir: A working directory for caching.
        :param mesh: An optional ``DeviceMesh`` that defines the data parallel dimensions. Ideally
            you should create this mesh using :func:`~olmo_core.distributed.parallel.build_world_mesh()`.
            Alternatively you can pass the ``dp_process_group`` instead.
        :param dp_process_group: The data parallel process group.
        """
        if not sources:
            raise OLMoConfigurationError("At least one 'source' must be provided.")

        if dp_process_group is None and mesh is not None:
            dp_process_group = get_dp_process_group(mesh)

        work_dir = work_dir or self.work_dir
        if work_dir is None:
            raise OLMoConfigurationError("'work_dir' must be specified.")

        tokenizer = tokenizer if tokenizer is not None else self.tokenizer
        if tokenizer is None:
            raise OLMoConfigurationError("'tokenizer' must be specified.")

        global_batch_size = (
            global_batch_size if global_batch_size is not None else self.global_batch_size
        )
        if global_batch_size is None:
            raise OLMoConfigurationError("'global_batch_size' must be specified.")

        return ComposableDataLoader(
            *sources,
            collator=collator
            or DataCollator(
                pad_token_id=tokenizer.pad_token_id, vocab_size=tokenizer.padded_vocab_size()
            ),
            tokenizer=tokenizer,
            work_dir=work_dir,
            global_batch_size=global_batch_size,
            dp_world_size=dist_utils.get_world_size(dp_process_group),
            dp_rank=dist_utils.get_rank(dp_process_group),
            fs_local_rank=dist_utils.get_fs_local_rank(),
            seed=self.seed,
            shuffle=self.shuffle,
            shuffle_strategy=self.shuffle_strategy,
            sources_per_epoch=self.sources_per_epoch,
            num_threads=self.num_threads,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            target_device_type=self.target_device_type or get_default_device().type,
            generate_doc_lengths=self.generate_doc_lengths,
            instance_filter_config=self.instance_filter_config,
            display_source_visualization=self.display_source_visualization,
        )


class ComposableDataLoader(TextDataLoaderBase):
    """
    A data loader for composable instance sources.

    :param sources: One or more instance sources to draw data from. All sources must have the same
      ``sequence_length`` and ``max_sequence_length``.
    :param collator: The data collator to use to form batches.
    :param tokenizer: The config of the tokenizer used to create the underlying data.
    :param work_dir: A common local working directory that can be used for caching.
    :param global_batch_size: The total batch size (in tokens) across all data parallel ranks.
    :param dp_world_size: The number of data parallel ranks.
    :param dp_rank: The data parallel rank of the current process.
    :param fs_local_rank: The local rank of the current process with respect to filesystem access
      of the working directory.
    :param seed: The random seed to use when shuffling data.
    :param shuffle: Whether to shuffle data at the start of each epoch.
    :param shuffle_strategy: How to shuffle the data. Defaults to :data:`ShuffleStrategy.inter_source`.
    :param sources_per_epoch: The number of sources to use per epoch. If -1, all sources are used.
    :param num_threads: The number of threads to use for loading data within each worker process.
    :param num_workers: The number of worker processes to use for loading data.
    :param prefetch_factor: The number of batches to prefetch from each worker process.
    :param target_device_type: The type of device that batches will be sent to, typically either "cpu" or "cuda".
    :param generate_doc_lengths: Whether to generate document lengths for each instance needed for
      intra-document masking.
    :param instance_filter_config: Optional configuration for filtering instances based on
      long sequences of repeated ngrams.
    :param display_source_visualization: Whether to display a visualization of each source
      to stdout from rank 0.
    """

    Config = ComposableDataLoaderConfig

    def __init__(
        self,
        *sources: InstanceSource,
        collator: DataCollator,
        tokenizer: TokenizerConfig,
        work_dir: PathOrStr,
        global_batch_size: int,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: Optional[int] = None,
        seed: int = SEED_NOT_SET,
        shuffle: bool = True,
        shuffle_strategy: Optional[ShuffleStrategy] = None,
        sources_per_epoch: int = -1,
        num_threads: Optional[int] = None,
        num_workers: int = 0,
        prefetch_factor: Optional[int] = None,
        target_device_type: str = "cpu",
        generate_doc_lengths: bool = False,
        instance_filter_config: Optional[InstanceFilterConfig] = None,
        display_source_visualization: bool = True,
    ):
        if not sources:
            raise OLMoConfigurationError("'sources' must contain at least one InstanceSource.")

        if sources_per_epoch == 0 or sources_per_epoch < -1:
            raise OLMoConfigurationError(
                "'sources_per_epoch' must be -1 (for all sources) or a positive integer."
            )

        if sources_per_epoch > 0 and len(sources) % sources_per_epoch != 0:
            raise OLMoConfigurationError(
                "'sources_per_epoch' must evenly divide into the number of sources."
            )

        if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != collator.pad_token_id:
            raise OLMoConfigurationError(
                "'tokenizer.pad_token_id' must match 'collator.pad_token_id'."
            )

        if not shuffle and shuffle_strategy is not None:
            raise OLMoConfigurationError("'shuffle_strategy' cannot be set if 'shuffle' is False.")

        if shuffle and shuffle_strategy is None:
            shuffle_strategy = ShuffleStrategy.inter_source

        super().__init__(
            collator=collator,
            work_dir=work_dir,
            global_batch_size=global_batch_size,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            fs_local_rank=fs_local_rank,
        )
        self.tokenizer = tokenizer
        self.seed = resolve_seed(seed)
        self.shuffle = shuffle
        self.shuffle_strategy = shuffle_strategy
        self.sources_per_epoch = sources_per_epoch if sources_per_epoch > 0 else len(sources)
        self.sources = tuple(sources)
        self.sequence_length = self.sources[0].sequence_length
        self.max_sequence_length = self.sources[0].max_sequence_length
        for i, source in enumerate(self.sources):
            if source.sequence_length != self.sequence_length:
                raise OLMoConfigurationError("All sources must have the same 'sequence_length'.")
            if source.max_sequence_length != self.max_sequence_length:
                raise OLMoConfigurationError(
                    "All sources must have the same 'max_sequence_length'."
                )
            if source.sequence_length != source.max_sequence_length:
                # NOTE: To guarantee the same data order with `self.max_sequence_length` fixed but `self.sequence_length`
                # changing, we need `len(source) // (source.max_sequence_length // source.sequence_length)` to remain constant.
                # For example, if `sequence_length` is half of `max_sequence_length`, then `len(source)`
                # should double. This check wouldn't catch all possible violations, but should catch some.
                if len(source) % (source.max_sequence_length // source.sequence_length) != 0:
                    raise OLMoConfigurationError(
                        "Each source must have a number of instances that is a multiple of "
                        "'max_sequence_length // sequence_length' when 'sequence_length' != "
                        "'max_sequence_length'. "
                        f"Source {i} does not meet this condition: {source}"
                    )
        if self.max_sequence_length % self.sequence_length != 0:
            raise OLMoConfigurationError(
                "'max_sequence_length' must be a multiple of 'sequence_length'."
            )
        if self.global_batch_size % self.sequence_length != 0:
            raise OLMoConfigurationError(
                "'global_batch_size' must be a multiple of 'sequence_length'."
            )
        self.num_threads = num_threads
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.target_device_type = target_device_type
        self.generate_doc_lengths = generate_doc_lengths
        self.instance_filter_config = instance_filter_config
        self._global_indices: Optional[np.ndarray] = None

        if display_source_visualization and dist_utils.get_rank() == 0:
            print()
            for source in sources:
                source.visualize()
                print()

    @property
    def sources_for_this_epoch(self) -> Tuple[InstanceSource, ...]:
        return self.sources_for_epoch(self._epoch or 1)

    def sources_for_epoch(self, epoch: int) -> Tuple[InstanceSource, ...]:
        assert self.sources_per_epoch > 0
        assert len(self.sources) % self.sources_per_epoch == 0
        num_groups = len(self.sources) // self.sources_per_epoch
        start_offset = ((epoch - 1) % num_groups) * self.sources_per_epoch
        return self.sources[start_offset : start_offset + self.sources_per_epoch]

    @ft.cached_property
    def source_fingerprints(self) -> Tuple[str, ...]:
        return tuple(source.fingerprint for source in self.sources)

    @property
    def total_instances(self) -> int:
        return self.instances_in_epoch(self._epoch or 1)

    def instances_in_epoch(self, epoch: int) -> int:
        sources_in_epoch = self.sources_for_epoch(epoch)
        if self.shuffle and self.shuffle_strategy == ShuffleStrategy.interleaved_source:
            # When interleaving sources, we need to make sure that each source contributes
            # equally to the total number of instances, so we take the minimum length
            # across all sources and multiply by the number of sources.
            min_length = min(len(source) for source in sources_in_epoch)
            return min_length * len(sources_in_epoch)
        else:
            return sum(len(source) for source in sources_in_epoch)

    @property
    def total_tokens(self) -> int:
        return self.total_instances * self.sequence_length

    @property
    def total_batches(self) -> Optional[int]:
        return self.batches_in_epoch(self._epoch or 1)

    def batches_in_epoch(self, epoch: int) -> Optional[int]:
        return self.instances_in_epoch(epoch) // (self.global_batch_size // self.sequence_length)

    @property
    def worker_info(self):
        return torch.utils.data.get_worker_info()

    def state_dict(self) -> Dict[str, Any]:
        return dict(
            source_fingerprints=self.source_fingerprints,
            batches_processed=self.batches_processed,
            tokens_processed=self.tokens_processed,
            global_batch_size=self.global_batch_size,
            sequence_length=self.sequence_length,
            max_sequence_length=self.max_sequence_length,
            shuffle=self.shuffle,
            shuffle_strategy=self.shuffle_strategy,
            sources_per_epoch=self.sources_per_epoch,
            seed=self.seed,
            epoch=self._epoch,
        )

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if state_dict["sources_per_epoch"] != self.sources_per_epoch:
            raise RuntimeError(
                "Restoring data loader state with a different 'sources_per_epoch' is not supported!"
            )

        if state_dict["source_fingerprints"] != self.source_fingerprints:
            # We're allowed to append more sources provided 'sources_per_epoch' hasn't changed
            # (checked above), the existing sources haven't changed, and we haven't already
            # iterated over the original set of sources.
            num_og_sources = len(state_dict["source_fingerprints"])
            if (
                num_og_sources < len(self.source_fingerprints)
                and state_dict["source_fingerprints"] == self.source_fingerprints[:num_og_sources]
            ):
                max_epochs = num_og_sources // self.sources_per_epoch
                if state_dict["epoch"] is not None and state_dict["epoch"] > max_epochs:
                    raise RuntimeError(
                        "Restoring data loader state after appending new sources can only be done "
                        "when the original sources haven't been iterated over yet!"
                    )
            else:
                raise RuntimeError(
                    "Restoring data loader state from different dataset source is not supported (fingerprints don't match)!"
                )

        if state_dict["max_sequence_length"] != self.max_sequence_length:
            raise RuntimeError(
                "Restoring data loading state with a different 'max_sequence_length' is not supported!"
            )

        if state_dict["shuffle_strategy"] != self.shuffle_strategy:
            raise RuntimeError(
                "Restoring data loading state with a different 'shuffle_strategy' is not supported!"
            )

        if state_dict["shuffle"] != self.shuffle:
            raise RuntimeError(
                "Restoring data loading state with a different shuffle setting is not supported!"
            )

        # Account for change in batch size / sequence length.
        if state_dict["tokens_processed"] % self.global_batch_size != 0:
            raise RuntimeError(
                f"Cannot resume after {state_dict['tokens_processed']:,d} tokens with a global "
                f"batch size of {self.global_batch_size:,d}: the checkpoint must fall on a "
                "batch boundary to avoid duplicating or losing data."
            )
        self.tokens_processed = state_dict["tokens_processed"]
        self.batches_processed = self.tokens_processed // self.global_batch_size

        if state_dict["seed"] != self.seed and self.shuffle:
            log.warning(
                "Restoring data loading state with a different data seed, "
                "will use data seed from state dict for data order consistency."
            )
            self.seed = state_dict["seed"]

        self._epoch = state_dict["epoch"] or self._epoch

        log.info(
            f"Data loader will resume from batch {self.batches_processed:,d}/{self.total_batches:,d} "
            f"based on batch size of {self.global_batch_size:,d} tokens"
        )

    def reshuffle(self, epoch: Optional[int] = None, **kwargs):
        del kwargs
        if epoch is None:
            epoch = 1 if self._epoch is None else self._epoch + 1
        if epoch <= 0:
            raise ValueError(f"'epoch' must be at least 1, got {epoch}")
        self._epoch = epoch
        self.build_and_save_global_indices()

    def _iter_batches(self) -> Iterable[Dict[str, Any]]:
        # If we're already at the end of epoch we can skip creating the iterator.
        if self.total_batches is not None and self.batches_processed >= self.total_batches:
            yield from ()
            return

        def _build_batch_iterator():
            return iter(
                torch.utils.data.DataLoader(
                    _IterableDataLoaderWrapper(self),
                    batch_size=None,
                    num_workers=self.num_workers,
                    pin_memory=self.target_device_type == "cuda" and self.num_workers > 0,
                    prefetch_factor=self.prefetch_factor,
                    persistent_workers=False,
                    timeout=0,
                ),
            )

        current_global_batch_size = self.global_batch_size
        batch_iterator = _build_batch_iterator()
        while (batch := next(batch_iterator, None)) is not None:
            yield batch

            # If batch size has changed, re-initialize the workers.
            # NOTE: base class handles the logic of adjusting `self.batches_processed` when
            # `self.global_batch_size` is changed through the property setter.
            if current_global_batch_size != self.global_batch_size:
                if self.num_workers > 0:
                    log.info("Batch size has changed, reinitializing data loading workers...")
                current_global_batch_size = self.global_batch_size
                batch_iterator = _build_batch_iterator()

    def get_mock_batch(self) -> Dict[str, Any]:
        rng = get_rng(self.seed + self.dp_rank)
        num_instances = self.rank_batch_size // self.sequence_length
        indices = rng.integers(0, self.total_instances, num_instances)
        instances = [self.get_instance(idx) for idx in indices]
        return self.collator(instances)

    def get_instance(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            idx = self.total_instances + idx
        source_start_offset = 0
        for source in self.sources_for_this_epoch:
            source_end_offset = source_start_offset + len(source)

            if source_start_offset <= idx < source_end_offset:
                out: Dict[str, Any] = {"index": idx}
                if source.label is not None:
                    out["metadata"] = {"source": source.label}

                instance = source[idx - source_start_offset]

                input_ids = as_tensor(instance["input_ids"])
                out["input_ids"] = input_ids

                if (label_mask := instance.get("label_mask")) is not None:
                    out["label_mask"] = as_tensor(label_mask)

                if self.generate_doc_lengths:
                    out["doc_lens"] = get_document_lengths(
                        input_ids,
                        self.tokenizer.eos_token_id,
                        bos_token_id=self.tokenizer.bos_token_id,
                    )

                if self.instance_filter_config is not None:
                    instance_mask = True
                    for m in find_periodic_sequences(
                        input_ids.numpy(),
                        max_period=self.instance_filter_config.repetition_max_period,
                        min_period=self.instance_filter_config.repetition_min_period,
                    ):
                        if m.times >= self.instance_filter_config.repetition_max_count:
                            instance_mask = False
                            break
                    out["instance_mask"] = instance_mask

                return out

            source_start_offset = source_end_offset
        raise IndexError(f"Index {idx} out of range for {self.total_instances} instances")

    @property
    def global_indices_file(self) -> Path:
        global_indices_fname = format_fname_from_fields(
            "global_indices",
            seed=self.seed if self.shuffle else None,
            epoch=self.epoch if self.shuffle else None,
            shuffle=self.shuffle_strategy if self.shuffle else None,
            size=self.total_instances,
            seq_len=self.sequence_length,
            max_seq_len=self.max_sequence_length,
            v=1,  # tick if logic changes
        )
        return self.work_dir / f"{global_indices_fname}.npy"

    def get_global_indices(self) -> np.ndarray:
        if self._global_indices is not None:
            return self._global_indices
        if not self.global_indices_file.is_file():
            raise RuntimeError("Missing global indices file, did you forget to call 'reshuffle()'?")
        return np.memmap(self.global_indices_file, mode="r", dtype=np.uint32)  # type: ignore

    def get_local_indices(self) -> np.ndarray:
        indices = self.get_global_indices()

        # Remove tail of data to make it evenly divisible.
        instances_per_batch = self.global_batch_size // self.sequence_length
        total_size = instances_per_batch * (self.total_instances // instances_per_batch)
        indices = indices[:total_size]

        # Slice up by batch.
        # shape: (global num batches, global num instances per batch)
        indices = indices.reshape(-1, instances_per_batch)

        # Offset by the number of batches already processed.
        if self.batches_processed > 0:
            indices = indices[self.batches_processed :]

        # Slice batches by data loader worker rank to avoid duplicates.
        if (worker_info := self.worker_info) is not None:
            # Note that each data loading worker gathers a whole batch at a time, and the workers
            # are called round-robin by rank. So to slice these up in a way that preserves order, regardless
            # of the number of workers, we give worker 0 the first batch, worker 1 the second batch, etc.
            indices = indices[worker_info.id :: worker_info.num_workers]

        # Finally slice batches into micro batches for the local DP rank.
        indices = indices[:, self.dp_rank :: self.dp_world_size].reshape((-1,))

        return indices

    def build_and_save_global_indices(self):
        self._global_indices = None
        if self.fs_local_rank == 0:
            if self.global_indices_file.is_file():
                log.info(
                    f"Using existing global indices file for seed {self.seed} and epoch {self.epoch} "
                    f"at:\n'{self.global_indices_file}'"
                )
            else:
                log.info(
                    f"Saving global data order indices for seed {self.seed} and epoch {self.epoch} "
                    f"to:\n'{self.global_indices_file}'..."
                )
                global_indices = self._build_global_indices()
                assert len(global_indices) < np.iinfo(np.uint32).max
                with memmap_to_write(
                    self.global_indices_file, shape=global_indices.shape, dtype=np.uint32
                ) as global_indices_mmap:
                    global_indices_mmap[:] = global_indices
                log.info(f"Global data order indices saved to:\n'{self.global_indices_file}'")
        dist_utils.barrier()

    def _build_global_indices(self) -> np.ndarray:
        dtype = np.uint32
        if not self.shuffle or self.shuffle_strategy == ShuffleStrategy.inter_source:
            return build_global_indices(
                self.total_instances,
                sequence_length=self.sequence_length,
                max_sequence_length=self.max_sequence_length,
                seed=(self.seed + self.epoch) if self.shuffle else None,
                dtype=dtype,
            )
        elif self.shuffle_strategy == ShuffleStrategy.intra_source:
            indices_per_source = []
            offset = 0
            for i, source in enumerate(self.sources_for_this_epoch):
                indices = build_global_indices(
                    len(source),
                    sequence_length=self.sequence_length,
                    max_sequence_length=self.max_sequence_length,
                    seed=self.seed + self.epoch + i,
                    dtype=dtype,
                )
                indices_per_source.append(indices + offset)
                offset += len(source)
            return np.concatenate(indices_per_source)
        elif self.shuffle_strategy == ShuffleStrategy.interleaved_source:
            source_size = min(len(source) for source in self.sources_for_this_epoch)
            num_sources = len(self.sources_for_this_epoch)
            interleaved_indices = np.empty((num_sources * source_size,), dtype=np.uint32)
            offset = 0
            for i, source in enumerate(self.sources_for_this_epoch):
                indices = build_global_indices(
                    source_size,
                    sequence_length=self.sequence_length,
                    max_sequence_length=self.max_sequence_length,
                    seed=self.seed + self.epoch + i,
                    dtype=dtype,
                )
                interleaved_indices[i::num_sources] = indices + offset
                offset += len(source)
            return interleaved_indices
        elif self.shuffle_strategy is None:
            raise NotImplementedError(f"Unknown shuffle strategy: {self.shuffle_strategy}")
        else:
            raise RuntimeError("shouldn't get here")


class _IterableDataLoaderWrapper(torch.utils.data.IterableDataset[Dict[str, Any]]):
    def __init__(self, data_loader: ComposableDataLoader):
        self.data_loader = data_loader

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over the local rank+worker instances.
        """
        num_threads = self.data_loader.num_threads
        if self.data_loader.worker_info is None and self.data_loader.num_threads is None:
            # If `num_threads` hasn't been specified and we're not using multiprocessing we'll
            # try to guess a good number of threads.
            num_threads = 4

        # Potentially slice by threads.
        instance_iterator: Iterator[Dict[str, Any]]
        if num_threads:
            # In order to stay ahead of training the total queue size (sum across all threads)
            # should be bigger than the maximum number of instances per batch locally.
            max_instances_per_rank: int
            max_instances_per_rank = (
                self.data_loader.rank_batch_size // self.data_loader.sequence_length
            )

            queue_size = math.ceil(max_instances_per_rank * 2 / num_threads)

            thread_generators = []
            for i in range(num_threads):
                # NOTE: `_get_local_instance_indices` might return an iterator, so we have to
                # create a unique one for each thread otherwise it would be exhausted prematurely
                # and give the wrong order.
                indices = self.data_loader.get_local_indices()
                generator = (
                    self.data_loader.get_instance(int(idx))
                    for idx in islice(indices, i, None, num_threads)
                )
                thread_generators.append(
                    threaded_generator(
                        generator, maxsize=queue_size, thread_name=f"data thread {i}"
                    )
                )

            instance_iterator = roundrobin(*thread_generators)
        else:
            indices = self.data_loader.get_local_indices()
            instance_iterator = (self.data_loader.get_instance(int(idx)) for idx in indices)

        return (
            self.data_loader.collator(batch)
            for batch in iter_batched(instance_iterator, self.data_loader.rank_batch_size)
        )
