import concurrent.futures
import functools as ft
import hashlib
import logging
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Sequence, Tuple

import numpy as np

import olmo_core.distributed.utils as dist_utils
import olmo_core.io as io
from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoConfigurationError

from ..tokenizer import TokenizerConfig
from ..types import LongDocStrategy, NumpyDatasetDType
from ..utils import (
    InstancePacker,
    chunked,
    load_array_slice_into_tensor,
    run_worker_func,
    write_array_to_disk,
)
from .instance_source import Instance, InstanceSource, InstanceSourceConfig
from .numpy_document_source import NumpyDocumentSource
from .token_source import DocumentSource, DocumentSourceConfig
from .utils import as_ndarray, path_map, resolve_seed

log = logging.getLogger(__name__)


@dataclass
class PackingInstanceSourceConfig(InstanceSourceConfig):
    """Config for :class:`PackingInstanceSource`."""

    sources: List[DocumentSourceConfig]
    sequence_length: int
    tokenizer: TokenizerConfig
    max_sequence_length: Optional[int] = None
    long_doc_strategy: LongDocStrategy = LongDocStrategy.truncate
    source_group_size: int = 1
    label: Optional[str] = None

    @classmethod
    def from_npy(
        cls,
        *npy_paths: str,
        tokenizer: TokenizerConfig,
        sequence_length: int,
        max_sequence_length: Optional[int] = None,
        dtype: Optional[NumpyDatasetDType] = None,
        source_permutation_seed: Optional[int] = None,
        source_group_size: int = 1,
        label_mask_paths: Optional[List[str]] = None,
        expand_glob: Optional[bool] = None,
        label: Optional[str] = None,
        long_doc_strategy: LongDocStrategy = LongDocStrategy.truncate,
    ) -> "PackingInstanceSourceConfig":
        """
        Create a :class:`PackingInstanceSourceConfig` from one or more tokenized ``.npy`` source files.
        """
        return cls(
            sources=[
                NumpyDocumentSource.Config(
                    source_paths=list(npy_paths),
                    tokenizer=tokenizer,
                    dtype=dtype,
                    source_permutation_seed=resolve_seed(source_permutation_seed),
                    source_group_size=source_group_size,
                    label_mask_paths=label_mask_paths,
                    expand_glob=expand_glob,
                )
            ],
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            max_sequence_length=max_sequence_length,
            source_group_size=1,
            label=label,
            long_doc_strategy=long_doc_strategy,
        )

    def build(self, work_dir: PathOrStr) -> "PackingInstanceSource":
        return PackingInstanceSource(
            *[source for source_config in self.sources for source in source_config.build(work_dir)],
            sequence_length=self.sequence_length,
            max_sequence_length=self.max_sequence_length,
            work_dir=work_dir,
            tokenizer=self.tokenizer,
            long_doc_strategy=self.long_doc_strategy,
            source_group_size=self.source_group_size,
            label=self.label,
        )


class PackingInstanceSource(InstanceSource):
    """
    Like the :class:`~olmo_core.data.numpy_dataset.NumpyPackedFSLDataset`, this instance source
    packs documents from each :class:`DocumentSource` into instances using the Optimized Best-Fit Decreasing (OBFD)
    algorithm described in `Fewer Truncations Improve Language Modeling <https://arxiv.org/pdf/2404.10830>`_.
    The resulting instances will all have exactly ``sequence_length`` tokens, using padding if needed.

    .. note::
        By default OBFD is applied to each source separately since source files from the Dolma toolkit
        are usually large enough for OBFD to achieve very good compactness (minimal padding tokens)
        and so that we can parallelize the packing. However, you can pack instances from multiple
        consecutive sources together by setting ``source_group_size`` to a value greater than 1.

    :param sources: Sources of documents to pack.
    :param sequence_length: The sequence length of each instance, i.e. the maximum number of tokens
        that can be packed into each instance.
    :param tokenizer: The tokenizer configuration.
    :param max_sequence_length: This must be equal to ``sequence_length`` if given.
    :param long_doc_strategy: The strategy to use for documents longer than ``sequence_length``.
    :param source_group_size: The number of consecutive sources to pack together.
    """

    Config = PackingInstanceSourceConfig

    DISPLAY_ICON = "\ueb29"

    def __init__(
        self,
        *sources: DocumentSource,
        sequence_length: int,
        work_dir: PathOrStr,
        tokenizer: TokenizerConfig,
        max_sequence_length: Optional[int] = None,
        long_doc_strategy: LongDocStrategy = LongDocStrategy.truncate,
        source_group_size: int = 1,
        label: Optional[str] = None,
    ):
        super().__init__(
            sequence_length=sequence_length,
            max_sequence_length=max_sequence_length,
            work_dir=work_dir,
            label=label,
        )
        if self.sequence_length != self.max_sequence_length:
            raise OLMoConfigurationError(
                f"'{self.__class__.__name__}' requires 'sequence_length' to be equal to 'max_sequence_length'."
            )
        self._sources = sources
        self._tokenizer = tokenizer
        self._long_doc_strategy = long_doc_strategy
        self._source_group_size = source_group_size
        self._source_groups = tuple(
            tuple(group) for group in chunked(self.sources, self.source_group_size)
        )
        if self.fs_local_rank == 0:
            log.info("Packing document into instances...")
            self._pack_all_documents_into_instances()
        dist_utils.barrier()

    @property
    def sources(self) -> Tuple[DocumentSource, ...]:
        return self._sources

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.pad_token_id

    @property
    def long_doc_strategy(self) -> LongDocStrategy:
        return self._long_doc_strategy

    @property
    def source_group_size(self) -> int:
        return self._source_group_size

    @property
    def source_groups(self) -> Tuple[Tuple[DocumentSource, ...], ...]:
        return self._source_groups

    @ft.cached_property
    def source_group_instance_offsets(self) -> Tuple[Tuple[int, int], ...]:
        item_size = np.uint64(0).itemsize
        num_instances_per_group = path_map(
            lambda path: io.get_file_size(path) // (item_size * 2),
            [self._get_instance_offsets_path(*sources) for sources in self.source_groups],
        )
        array_instance_offsets = []
        start_offset = 0
        for num_instances in num_instances_per_group:
            array_instance_offsets.append((start_offset, start_offset + num_instances))
            start_offset += num_instances
        return tuple(array_instance_offsets)

    @property
    def num_instances(self) -> int:
        return self.source_group_instance_offsets[-1][1]

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update(
            (
                f"class={self.__class__.__name__},"
                f"{self.sequence_length=},"
                f"{self.long_doc_strategy=},"
                f"{self.source_group_size=},"
            ).encode()
        )
        for source in self.sources:
            sha256_hash.update(f"source={source.fingerprint},".encode())
        return sha256_hash.hexdigest()

    def children(self):
        return self.sources

    def __len__(self) -> int:
        return self.source_group_instance_offsets[-1][1]

    def __getitem__(self, idx: int) -> Instance:
        idx = self.validate_index(idx)

        # The index of the source group.
        source_group_index: Optional[int] = None
        # The instance index within the source group.
        instance_index: Optional[int] = None
        for i, (instance_offset_start, instance_offset_end) in enumerate(
            self.source_group_instance_offsets
        ):
            if instance_offset_start <= idx < instance_offset_end:
                source_group_index = i
                instance_index = idx - instance_offset_start
                break

        if source_group_index is None or instance_index is None:
            raise IndexError(f"{idx} is out of bounds for source of size {len(self)}")

        sources = self.source_groups[source_group_index]
        document_indices_path = self._get_document_indices_path(*sources)
        instance_offsets_path = self._get_instance_offsets_path(*sources)
        docs_by_instance_path = self._get_docs_by_instance_path(*sources)

        # Load start and end document indices corresponding to instance.
        instance_indices = load_array_slice_into_tensor(
            instance_offsets_path,
            instance_index * 2,
            instance_index * 2 + 2,
            np.uint64,
        ).tolist()
        instance_start, instance_end = instance_indices

        # Load document IDs corresponding to instance.
        document_ids = load_array_slice_into_tensor(
            docs_by_instance_path,
            instance_start,
            instance_end,
            np.uint64,
        ).tolist()

        # Load token IDs and label masks for each document.
        document_token_ids: List[np.ndarray] = []
        document_label_masks: List[np.ndarray] = []
        for document_id in document_ids:
            document_indices = load_array_slice_into_tensor(
                document_indices_path, document_id * 2, document_id * 2 + 2, np.uint64
            ).tolist()
            document_start, document_end = document_indices

            # Pick out the right source from the source group by comparing the starting
            # index (in tokens) of the document to the starting index of each source within the group.
            source: DocumentSource
            source_start = 0
            for source in sources:
                source_size = source.num_tokens
                if source_start <= document_start < (source_start + source_size):
                    document_start -= source_start
                    document_end -= source_start
                    break
                else:
                    source_start += source_size
            else:
                raise RuntimeError("we shouldn't be here!")

            assert source is not None
            token_range = source.get_token_range(document_start, document_end)
            document_token_ids.append(as_ndarray(token_range["input_ids"]))
            if "label_mask" in token_range:
                document_label_masks.append(as_ndarray(token_range["label_mask"]))
            else:
                document_label_masks.append(np.ones_like(document_token_ids[-1], dtype=np.bool_))

        # Combine token IDs and maybe label masks for each document.
        input_ids = np.concatenate(document_token_ids, dtype=np.int_)
        label_mask = np.concatenate(document_label_masks)

        # Pad to target sequence length.
        pad_shape = (0, self.sequence_length - input_ids.size)
        loss_valid_mask = np.pad(np.ones(input_ids.size, dtype=np.bool_), pad_shape)
        label_mask = np.pad(label_mask, pad_shape, constant_values=False)
        input_ids = np.pad(input_ids, pad_shape, constant_values=self.pad_token_id)

        return {
            "input_ids": typing.cast(Sequence[int], input_ids),
            "label_mask": typing.cast(Sequence[bool], label_mask),
            "loss_valid_mask": typing.cast(Sequence[bool], loss_valid_mask),
        }

    def _get_indices_path(self, name: str, *sources: DocumentSource, ext="npy") -> Path:
        sources_fingerprint = hashlib.sha256()
        sources_fingerprint.update((f"{self.sequence_length=},{self.long_doc_strategy=},").encode())
        for source in sources:
            sources_fingerprint.update(source.fingerprint.encode())
        filename = f"{name}-{sources_fingerprint.hexdigest()}.{ext}"
        return self.work_dir / filename

    def _get_document_indices_path(self, *sources: DocumentSource) -> Path:
        return self._get_indices_path("document-indices", *sources)

    def _get_instance_offsets_path(self, *sources: DocumentSource) -> Path:
        return self._get_indices_path("instance-offsets", *sources)

    def _get_docs_by_instance_path(self, *sources: DocumentSource) -> Path:
        return self._get_indices_path("documents-by-instance", *sources)

    def _pack_documents_into_instances(self, *sources: DocumentSource) -> Tuple[int, int]:
        document_indices_path = self._get_document_indices_path(*sources)
        instance_offsets_path = self._get_instance_offsets_path(*sources)
        docs_by_instance_path = self._get_docs_by_instance_path(*sources)

        def doc_idx_gen() -> Generator[int, None, None]:
            start_offset = 0
            for source in sources:
                for start_idx, end_idx in source.get_document_offsets():
                    if end_idx - start_idx > self.sequence_length:
                        if self.long_doc_strategy == LongDocStrategy.truncate:
                            yield start_offset + start_idx
                            yield start_offset + start_idx + self.sequence_length
                        elif self.long_doc_strategy == LongDocStrategy.fragment:
                            for new_start_idx in range(start_idx, end_idx, self.sequence_length):
                                yield start_offset + new_start_idx
                                yield start_offset + min(
                                    end_idx, new_start_idx + self.sequence_length
                                )
                        elif self.long_doc_strategy == LongDocStrategy.exclude:
                            continue  # drop the document entirely
                        else:
                            raise NotImplementedError(self.long_doc_strategy)
                    else:
                        yield start_offset + start_idx
                        yield start_offset + end_idx
                start_offset += source.num_tokens

        document_indices = np.fromiter(doc_idx_gen(), dtype=np.uint64).reshape(-1, 2)
        instance_packer = InstancePacker(self.sequence_length)
        instances, document_indices, total_tokens = instance_packer.pack_documents(document_indices)

        instance_start_offset = 0
        instance_offsets_list: List[int] = []
        documents_by_instance_list: List[int] = []
        for instance in instances:
            instance_offsets_list.append(instance_start_offset)
            instance_offsets_list.append(instance_start_offset + len(instance))
            instance_start_offset += len(instance)
            documents_by_instance_list.extend(instance)

        # shape: (num_instances * 2,)
        instance_offsets = np.array(instance_offsets_list, dtype=np.uint64)
        # shape: (num_documents,)
        docs_by_instance = np.array(documents_by_instance_list, dtype=np.uint64)

        write_array_to_disk(document_indices.reshape(-1), document_indices_path)
        write_array_to_disk(instance_offsets, instance_offsets_path)
        write_array_to_disk(docs_by_instance, docs_by_instance_path)

        return len(instances), total_tokens

    def _pack_all_documents_into_instances(self):
        # Collect all sources that need to be packed (no cache hit).
        sources_needed: List[Tuple[DocumentSource, ...]] = []
        for sources in self.source_groups:
            document_indices_path = self._get_document_indices_path(*sources)
            instance_offsets_path = self._get_instance_offsets_path(*sources)
            docs_by_instance_path = self._get_docs_by_instance_path(*sources)
            if (
                document_indices_path.is_file()
                and instance_offsets_path.is_file()
                and docs_by_instance_path.is_file()
            ):
                log.info(f"Reusing cached packing results for {sources}")
            elif sources not in sources_needed:
                sources_needed.append(sources)

        if sources_needed:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = []
                for sources in sources_needed:
                    log.info(f"Packing documents from {sources} into instances...")
                    future = executor.submit(
                        run_worker_func,
                        self._pack_documents_into_instances,
                        *sources,
                    )
                    futures.append(future)

                concurrent.futures.wait(futures, return_when="FIRST_EXCEPTION")

                # Log results.
                for sources, future in zip(sources_needed, futures):
                    total_instances, total_tokens = future.result()
                    total_padding = self.sequence_length * total_instances - total_tokens
                    avg_padding = total_padding / total_instances
                    log.info(
                        f"Packed {total_tokens:,} tokens from {sources} into {total_instances:,d} instances "
                        f"of sequence length {self.sequence_length:,d} using an average of "
                        f"{avg_padding:.1f} padding tokens per instance."
                    )
