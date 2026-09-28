from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import math
import os
import random
import tempfile
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property, partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from olmo_core.exceptions import OLMoConfigurationError, OLMoEnvironmentError

from ..aliases import PathOrStr
from ..config import Config, StrEnum
from ..distributed.utils import barrier, get_fs_local_rank
from ..io import (
    _get_s3_client,
    deterministic_glob_directory,
    get_file_size,
    is_url,
    normalize_path,
    resource_path,
)
from .mixes import DataMix, DataMixBase
from .source_mixture import SourceMixtureDatasetConfig
from .tokenizer import TokenizerConfig
from .types import LongDocStrategy, NumpyDatasetDType, NumpyUIntTypes
from .utils import (
    bucket_documents,
    chunk_array,
    chunked,
    divide_into_buckets,
    find_periodic_sequences,
    get_doc_lengths_from_indices,
    get_document_lengths,
    get_rng,
    load_array_slice_into_tensor,
    memmap_to_write,
    pack_documents_into_instances,
    run_worker_func,
    segment_documents_into_instances,
    write_array_to_disk,
)

__all__ = [
    "NumpyDatasetBase",
    "NumpyFSLDatasetBase",
    "NumpyFSLDataset",
    "NumpyFSLDatasetMixture",
    "NumpyPaddedFSLDataset",
    "NumpyPackedFSLDataset",
    "NumpyInterleavedFSLDataset",
    "VSLCurriculum",
    "VSLNaturalCurriculum",
    "VSLGrowthCurriculum",
    "VSLGrowP2Curriculum",
    "VSLGrowLinearCurriculum",
    "NumpyVSLDataset",
    "NumpyDatasetConfig",
    "NumpyFSLDatasetConfig",
    "NumpyPaddedFSLDatasetConfig",
    "NumpyPackedFSLDatasetConfig",
    "NumpyInterleavedFSLDatasetConfig",
    "NumpyVSLDatasetConfig",
    "VSLCurriculumType",
    "VSLCurriculumConfig",
]


log = logging.getLogger(__name__)


T = TypeVar("T")


def _get_data_prep_max_workers() -> Optional[int]:
    raw_value = os.environ.get("OLMO_DATA_PREP_WORKERS")
    if not raw_value:
        return None

    try:
        max_workers = int(raw_value)
    except ValueError as e:
        raise OLMoEnvironmentError("'OLMO_DATA_PREP_WORKERS' must be a positive integer") from e

    if max_workers <= 0:
        raise OLMoEnvironmentError("'OLMO_DATA_PREP_WORKERS' must be a positive integer")
    return max_workers


def _get_data_prep_progress_interval() -> float:
    raw_value = os.environ.get("OLMO_DATA_PREP_PROGRESS_INTERVAL")
    if not raw_value:
        return 30.0

    try:
        interval = float(raw_value)
    except ValueError as e:
        raise OLMoEnvironmentError(
            "'OLMO_DATA_PREP_PROGRESS_INTERVAL' must be a non-negative number"
        ) from e

    if interval < 0:
        raise OLMoEnvironmentError(
            "'OLMO_DATA_PREP_PROGRESS_INTERVAL' must be a non-negative number"
        )
    return interval


def _get_data_prep_use_array_if_local() -> Optional[bool]:
    raw_value = os.environ.get("OLMO_DATA_PREP_USE_ARRAY_IF_LOCAL")
    if not raw_value or raw_value == "auto":
        return None

    value = raw_value.lower()
    if value in ("1", "true", "yes", "y", "array", "arrays"):
        return True
    if value in ("0", "false", "no", "n", "metadata", "csv"):
        return False

    raise OLMoEnvironmentError("'OLMO_DATA_PREP_USE_ARRAY_IF_LOCAL' must be one of true/false/auto")


def _format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def _log_data_prep_progress(
    label: str,
    *,
    completed: int,
    total: int,
    started_at: float,
    latest_path: Optional[PathOrStr] = None,
    unit: str = "files",
):
    elapsed = max(time.monotonic() - started_at, 1e-6)
    rate = completed / elapsed
    remaining = total - completed
    eta = remaining / rate if rate > 0 else math.inf
    message = (
        f"{label}: {completed:,d}/{total:,d} complete "
        f"({completed / total:.1%}, {rate:.2f} {unit}/s, "
        f"elapsed {_format_elapsed(elapsed)}, eta {_format_elapsed(eta) if math.isfinite(eta) else 'unknown'})"
    )
    if latest_path is not None:
        message += f"; latest '{latest_path}'"
    log.info(message)


@dataclass
class InstanceFilterConfig(Config):
    repetition_max_period: int = 13
    repetition_min_period: int = 1
    repetition_max_count: int = 32


class NumpyDatasetBase(ABC):
    """
    An abstract base class for datasets backed by numpy arrays on disk of token IDs.

    In general the instances that these datasets produce are sequences of token IDs from one
    or more numpy arrays, sometimes with additional metadata attached.
    The way those instances are formed depends on the implementation details of the subclass.

    .. warning::
        When using :class:`NumpyDatasetBase` implementations in a distributed setting be sure
        that the :data:`work_dir` is shared among all local ranks and :data:`fs_local_rank` is set
        accordingly. Once those fields are set you should then call :meth:`prepare()` in the
        main process before doing anything else.

    .. tip::
        Use the dataset config helpers (e.g. :class:`NumpyFSLDatasetConfig`) to configure and
        construct datasets instead of constructing them directly.
    """

    def __init__(
        self,
        *paths: PathOrStr,
        pad_token_id: int,
        eos_token_id: int,
        vocab_size: int,
        dtype: NumpyUIntTypes = np.uint16,
        bos_token_id: Optional[int] = None,
    ):
        if not paths:
            raise OLMoConfigurationError("At least one path is required")

        self._array_paths = tuple(paths)
        self._pad_token_id = pad_token_id
        self._eos_token_id = eos_token_id
        self._bos_token_id = bos_token_id
        self._vocab_size = vocab_size
        self._dtype = dtype
        self._fs_local_rank = get_fs_local_rank()
        self._work_dir: Optional[Path] = None
        self._work_dir_set = False
        self._array_file_sizes: Optional[Tuple[int, ...]] = None

    @property
    @abstractmethod
    def max_sequence_length(self) -> int:
        """
        The maximum sequence length of any instances generated by this dataset.
        """
        raise NotImplementedError

    @property
    def paths(self) -> Tuple[PathOrStr, ...]:
        """
        Paths and/or URLs to the numpy arrays.
        """
        return self._array_paths

    @property
    def file_sizes(self) -> Tuple[int, ...]:
        """
        The size, in bytes, of each numpy array.
        """
        if self._array_file_sizes is None:
            self._array_file_sizes = tuple(self.map(lambda path, _: get_file_size(path)))
        return self._array_file_sizes

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._bos_token_id

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def dtype(
        self,
    ) -> NumpyUIntTypes:
        """
        The numpy datatype of the arrays.
        """
        return self._dtype

    @property
    def fingerprint_version(self) -> str:
        """
        The version of the :data:`fingerprint`.
        """
        return "v2.0"

    @property
    def fingerprint_fields(self) -> Tuple[str, ...]:
        """
        Extra values to include when calculating the data contents :data:`fingerprint`.
        """
        return ("vocab_size", "pad_token_id", "eos_token_id", "dtype", "bos_token_id")

    @property
    def fingerprint(self) -> str:
        """
        Used to compare the contents of a dataset.
        """
        sha256_hash = hashlib.sha256()
        sha256_hash.update(f"class={self.__class__.__name__}".encode())
        for field_name in self.fingerprint_fields:
            field_value = getattr(self, field_name)
            sha256_hash.update(f"{field_name}={field_value},".encode())
        for path, size in zip(self.paths, self.file_sizes):
            sha256_hash.update(f"path={os.path.basename(path)},size={size},".encode())
        return sha256_hash.hexdigest()

    @property
    def fs_local_rank(self) -> int:
        return self._fs_local_rank

    @fs_local_rank.setter
    def fs_local_rank(self, fs_local_rank: int):
        self._fs_local_rank = fs_local_rank

    @property
    def work_dir(self) -> Path:
        if self._work_dir is not None:
            return self._work_dir
        else:
            return Path(tempfile.gettempdir())

    @work_dir.setter
    def work_dir(self, work_dir: PathOrStr):
        if is_url(work_dir):
            raise OLMoConfigurationError(
                f"'work_dir' should be a local path, not a URL ('{work_dir}')."
            )
        self._work_dir = Path(normalize_path(work_dir))
        self._work_dir_set = True

    @property
    def work_dir_set(self) -> bool:
        """
        Check if the working directory was explicitly set.
        """
        return self._work_dir_set

    @property
    def num_tokens(self) -> int:
        """
        Get the total number of tokens in the dataset.
        """
        raise NotImplementedError

    def _get_file_size(self, path: PathOrStr):
        path_idx = self.paths.index(path)
        return self.file_sizes[path_idx]

    def _warmup_clients(self):
        # Maybe create client up front to work around a threading issue in boto.
        if any(str(p).startswith("s3://") for p in self.paths):
            _get_s3_client("s3")

        if any(str(p).startswith("r2://") for p in self.paths):
            try:
                _get_s3_client("r2")
            except OLMoEnvironmentError:
                # R2 might not be needed, so ignore this error. We will get an error
                # later if R2 is needed.
                pass

        if any(str(p).startswith("weka://") for p in self.paths):
            try:
                _get_s3_client("weka")
            except OLMoEnvironmentError:
                # Weka might not be needed, so ignore this error. We will get an error
                # later if Weka is needed.
                pass

    def map(
        self,
        func: Callable[[PathOrStr, int], T],
        *,
        max_workers: Optional[int] = None,
        method: Literal["threads", "processes"] = "threads",
        _paths: Optional[Sequence[PathOrStr]] = None,
    ) -> List[T]:
        """
        Call a function on each path in the dataset, returning a list of the results, in order.

        :param func: The function to map to the paths and their indices.
        :param max_workers: The number of workers threads/processes. Set to 0 to execute synchronously
            in the main thread/process.
        :param method: Whether to use multi-threading or multi-processing.

        :returns: The results, in the same order as :data:`paths`.
        """
        paths = _paths or self.paths

        if max_workers == 0:
            return [func(path, idx) for idx, path in enumerate(paths)]

        executor_class: Union[
            Type[concurrent.futures.ThreadPoolExecutor],
            Type[concurrent.futures.ProcessPoolExecutor],
        ]
        if method == "threads":
            self._warmup_clients()
            executor_class = concurrent.futures.ThreadPoolExecutor
        elif method == "processes":
            executor_class = concurrent.futures.ProcessPoolExecutor
        else:
            raise ValueError(method)

        with executor_class(max_workers=max_workers) as executor:
            futures = [executor.submit(func, path, idx) for idx, path in enumerate(paths)]

        return [future.result() for future in futures]

    def prepare(self):
        """
        Perform any necessary preparation.

        .. warning::
            Be sure to set :data:`work_dir` properly before calling this and only call this from the
            main process (not a worker process).
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Get the number of instances in the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get an instance from the dataset. At a minimum this will contain the field "input_ids", a
        integer tensor of token IDs.
        """
        raise NotImplementedError

    def _validate_instance(
        self, input_ids: torch.Tensor, instance_filter_config: InstanceFilterConfig
    ) -> bool:
        for m in find_periodic_sequences(
            input_ids.numpy(),
            max_period=instance_filter_config.repetition_max_period,
            min_period=instance_filter_config.repetition_min_period,
        ):
            if m.times >= instance_filter_config.repetition_max_count:
                return False
        return True


class NumpyFSLDatasetBase(NumpyDatasetBase, Dataset[Dict[str, Any]]):
    """
    A base class for fixed sequence length (FSL) numpy array-backed datasets.
    """

    def __init__(
        self,
        *paths: PathOrStr,
        sequence_length: int,
        pad_token_id: int,
        eos_token_id: int,
        vocab_size: int,
        dtype: NumpyUIntTypes = np.uint16,
        metadata: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        include_instance_metadata: Optional[bool] = None,
        generate_doc_lengths: bool = False,
        bos_token_id: Optional[int] = None,
        instance_filter_config: Optional[InstanceFilterConfig] = None,
        label_mask_paths: Optional[List[PathOrStr]] = None,
    ):
        if include_instance_metadata is None and metadata:
            include_instance_metadata = True

        if isinstance(metadata, list):
            if len(metadata) != len(paths):
                raise OLMoConfigurationError(
                    "'metadata' should have the same length as the number of file paths"
                )
        else:
            metadata = [metadata or {}] * len(paths)

        if label_mask_paths is not None and len(label_mask_paths) != len(paths):
            raise OLMoConfigurationError(
                "There must be the same number of 'label_mask_paths' as there are 'paths'"
            )

        super().__init__(
            *paths,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            dtype=dtype,
            bos_token_id=bos_token_id,
        )
        self._metadata = tuple(metadata)
        self._sequence_length = sequence_length
        self._include_instance_metadata = include_instance_metadata
        self._generate_doc_lengths = generate_doc_lengths
        self.instance_filter_config = instance_filter_config
        self._label_mask_paths = label_mask_paths
        self._label_mask_path_to_source_path: Dict[PathOrStr, PathOrStr] = {}
        if self._label_mask_paths:
            for label_mask_path, source_path in zip(self._label_mask_paths, self._array_paths):
                self._label_mask_path_to_source_path[label_mask_path] = source_path

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @property
    def max_sequence_length(self) -> int:
        return self.sequence_length

    @property
    def max_target_sequence_length(self) -> Optional[int]:
        return None

    def _get_indices_path(
        self, name: str, *source_paths: PathOrStr, extra_ids: Optional[Sequence[str]] = None
    ) -> Path:
        sha256_hash = hashlib.sha256()
        for source_path in source_paths:
            # NOTE: the pre-processed data file names are based on the corresponding source (token IDs) file name,
            # so to get the right instance indices file name for a label mask file, we need to map
            # the label mask file name to its corresponding source file name.
            if source_path in self._label_mask_path_to_source_path:
                source_path = self._label_mask_path_to_source_path[source_path]
            sha256_hash.update(str(source_path).encode())
            sha256_hash.update(str(self._get_file_size(source_path)).encode())
        for extra_id in extra_ids or []:
            sha256_hash.update(extra_id.encode())
        path_hash = sha256_hash.hexdigest()
        return self.work_dir / "dataset-common" / f"{name}-{self.sequence_length}-{path_hash}.npy"


class NumpyFSLDataset(NumpyFSLDatasetBase):
    """
    A fixed sequence length (FSL) numpy array-backed dataset.

    In this implementation the token IDs from all arrays are concatenated together and then chunked
    into contiguous blocks of ``sequence_length`` tokens to create instances. Therefore documents
    may be split over multiple instances.

    .. seealso::
        :class:`NumpyPaddedFSLDataset`

    .. important::
        If the length of an array is not a multiple of ``sequence_length`` or
        ``max_target_sequence_length`` the remainder of the tokens will be ignored.

    .. important::
        No special tokens are added to the input IDs so it's assumed that if you want
        EOS tokens between documents, for example, those will already be in the array.

    :param paths: Paths or URLs to numpy token ID arrays.
    :param sequence_length: The number of tokens to chunk together into a single instance.
        Generally this should correspond to your model's maximum input length.
    :param pad_token_id: The ID of the padding token.
    :param eos_token_id: The ID of the EOS token.
    :param dtype: The numpy datatype of the arrays.
    :param metadata: Metadata to add to each item. This should be a dictionary or a list of dictionaries
        with the same number of items as there are paths.
    :param include_instance_metadata: If ``True`` (the default), each instance returned from
        :meth:`__getitem__()` will include the metadata from its source.
    :param max_target_sequence_length: Optional upper bound used when precomputing cached offsets.
        If you're planning a sequence-length warm-up, set this to the final chunk size so future
        datasets with larger ``sequence_length`` values can reuse the exact same document ordering.
        The current dataset still returns ``sequence_length``-token windows; this hint simply keeps
        token boundaries and cache files deterministic across warm-up stages. Leave ``None`` if you
        won't rebuild at a larger length.
    """

    def __init__(
        self,
        *paths: PathOrStr,
        sequence_length: int,
        pad_token_id: int,
        eos_token_id: int,
        vocab_size: int,
        dtype: NumpyUIntTypes = np.uint16,
        metadata: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        include_instance_metadata: Optional[bool] = None,
        generate_doc_lengths: bool = False,
        bos_token_id: Optional[int] = None,
        max_target_sequence_length: Optional[int] = None,
        instance_filter_config: Optional[InstanceFilterConfig] = None,
        label_mask_paths: Optional[List[PathOrStr]] = None,
    ):
        super().__init__(
            *paths,
            sequence_length=sequence_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            dtype=dtype,
            metadata=metadata,
            include_instance_metadata=include_instance_metadata,
            generate_doc_lengths=generate_doc_lengths,
            bos_token_id=bos_token_id,
            instance_filter_config=instance_filter_config,
            label_mask_paths=label_mask_paths,
        )

        if max_target_sequence_length is not None and (
            max_target_sequence_length < sequence_length
            or max_target_sequence_length % sequence_length != 0
        ):
            raise OLMoConfigurationError(
                f"'max_target_sequence_length'({max_target_sequence_length}) should be a multiple of 'sequence_length'({sequence_length})"
            )

        self._max_target_sequence_length = max_target_sequence_length
        self._array_offsets: Optional[Tuple[Tuple[int, int], ...]] = None
        self._num_instances: Optional[int] = None

    @property
    def fingerprint_fields(self) -> Tuple[str, ...]:
        return (
            "vocab_size",
            "pad_token_id",
            "eos_token_id",
            "dtype",
            "max_target_sequence_length",
            "bos_token_id",
        )

    @property
    def num_tokens(self) -> int:
        return len(self) * self.sequence_length

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @property
    def max_sequence_length(self) -> int:
        return self.sequence_length

    @property
    def max_target_sequence_length(self) -> Optional[int]:
        return self._max_target_sequence_length

    @property
    def file_sizes(self) -> Tuple[int, ...]:
        """
        The size, in bytes, of each numpy array.
        """
        return self._sizes_and_offsets[0]

    @property
    def offsets(self) -> Tuple[Tuple[int, int], ...]:
        """
        Gives the global start and end instance indices for each data file in the dataset.
        """
        return self._sizes_and_offsets[1]

    @property
    def metadata(self) -> Tuple[Dict[str, Any], ...]:
        return self._metadata

    def prepare(self):
        len(self)

    def __len__(self) -> int:
        if self._num_instances is None:
            self._num_instances = self.offsets[-1][1]
        return self._num_instances

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = int(index)  # in case this is a numpy int type.
        pos_index = index if index >= 0 else len(self) + index

        # The index of the array within 'self.paths'.
        array_index: Optional[int] = None
        # The index within the corresponding array.
        array_local_index: Optional[int] = None
        for i, (offset_start, offset_end) in enumerate(self.offsets):
            if offset_start <= pos_index < offset_end:
                array_index = i
                array_local_index = pos_index - offset_start
                break

        if array_index is None or array_local_index is None:
            raise IndexError(f"{index} is out of bounds for dataset of size {len(self)}")

        # Read the data from file.
        input_ids = self._read_chunk_from_array(self.paths[array_index], array_local_index)
        out: Dict[str, Any] = {"input_ids": input_ids}

        if self._label_mask_paths is not None:
            label_mask = self._read_chunk_from_array(
                self._label_mask_paths[array_index], array_local_index, dtype=np.bool_
            )
            out["label_mask"] = label_mask

        if self.instance_filter_config is not None:
            out["instance_mask"] = self._validate_instance(input_ids, self.instance_filter_config)

        if self._include_instance_metadata:
            metadata = self._metadata[array_index]
            out["metadata"] = deepcopy(metadata)

        if self._generate_doc_lengths:
            out["doc_lens"] = get_document_lengths(
                input_ids, self.eos_token_id, bos_token_id=self.bos_token_id
            )

        return out

    @property
    def _sizes_and_offsets(self) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, int], ...]]:
        if self._array_offsets is None or self._array_file_sizes is None:
            array_sizes: List[int] = []
            array_offsets: List[Tuple[int, int]] = []
            array_file_sizes: List[int] = []
            item_size = self.dtype(0).itemsize

            start_offset = 0
            for size, length in self.map(self._get_file_size_and_length):
                array_sizes.append(size // item_size)
                end_offset = start_offset + length
                array_offsets.append((start_offset, end_offset))
                array_file_sizes.append(size)
                start_offset += length

            self._array_offsets = tuple(array_offsets)
            self._array_file_sizes = tuple(array_file_sizes)

            mask_item_size = np.bool_(True).itemsize
            if self._label_mask_paths is not None:
                for i, (size, _) in enumerate(
                    self.map(
                        partial(self._get_file_size_and_length, dtype=np.bool_),
                        _paths=self._label_mask_paths,
                    )
                ):
                    size = size // mask_item_size
                    if array_sizes[i] != size:
                        raise RuntimeError(
                            f"mismatch between size of source file ('{self._array_paths[i]}', {array_sizes[i]:,d}) and "
                            f"size of corresponding label mask file ('{self._label_mask_paths[i]}', {size:,d})"
                        )

        return self._array_file_sizes, self._array_offsets

    def _read_chunk_from_array(self, path: PathOrStr, index: int, dtype=None) -> torch.Tensor:
        start_idx = index * self.sequence_length
        return load_array_slice_into_tensor(
            path,
            start_idx,
            start_idx + self.sequence_length,
            dtype or self.dtype,
        )

    def _get_file_size_and_length(self, path: PathOrStr, idx: int, dtype=None) -> Tuple[int, int]:
        del idx
        dtype = dtype or self.dtype
        item_size = dtype(0).itemsize
        file_size = get_file_size(path)
        if (
            self.max_target_sequence_length is None
            or self.max_target_sequence_length == self.sequence_length
        ):
            return file_size, file_size // (item_size * self.sequence_length)
        elif self.max_target_sequence_length > self.sequence_length:
            num_max_seq_len_instances = file_size // (item_size * self.max_target_sequence_length)
            return (
                file_size,
                num_max_seq_len_instances
                * (self.max_target_sequence_length // self.sequence_length),
            )
        else:
            raise RuntimeError("invalid 'max_target_sequence_length' or 'sequence_length'")


class NumpyFSLDatasetMixture(NumpyFSLDataset):
    """
    A version of :class:`NumpyFSLDataset` built from a mixture of sources and their expected token ratios relative to each other.
    A ``path_offset_index`` is used to determine the number of instances to retain from a path when constructing the local indices.
    """

    def __init__(
        self,
        *paths: PathOrStr,
        path_offset_index: Dict[Tuple[str, int], int],
        seed: int,
        sequence_length: int,
        pad_token_id: int,
        eos_token_id: int,
        vocab_size: int,
        dtype: NumpyUIntTypes = np.uint16,
        metadata: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        include_instance_metadata: Optional[bool] = None,
        generate_doc_lengths: bool = False,
        bos_token_id: Optional[int] = None,
        max_target_sequence_length: Optional[int] = None,
        instance_filter_config: Optional[InstanceFilterConfig] = None,
    ):
        if max_target_sequence_length is not None and (
            max_target_sequence_length < sequence_length
            or max_target_sequence_length % sequence_length != 0
        ):
            raise OLMoConfigurationError(
                f"'max_target_sequence_length'({max_target_sequence_length}) should be a multiple of 'sequence_length'({sequence_length})"
            )

        super().__init__(
            *paths,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            dtype=dtype,
            sequence_length=sequence_length,
            metadata=metadata,
            include_instance_metadata=include_instance_metadata,
            generate_doc_lengths=generate_doc_lengths,
            bos_token_id=bos_token_id,
            max_target_sequence_length=max_target_sequence_length,
            instance_filter_config=instance_filter_config,
        )
        self._num_instances: Optional[int] = None
        self._array_offsets: Optional[Tuple[Tuple[int, int], ...]] = None
        self._lengths_dtype: Optional[NumpyUIntTypes] = None
        self._instances_per_bucket: Optional[Tuple[Tuple[int, int], ...]] = None
        self._path_offset_index = path_offset_index
        self._seed = seed

    @property
    def indices_dtype(
        self,
    ) -> NumpyUIntTypes:
        return np.uint32

    def prepare(self):
        if self.fs_local_rank == 0:
            log.info("Gathering indices...")
            self._write_document_indices()
        barrier()
        len(self)

    def _get_instance_indices_path(self, source_path: PathOrStr) -> Path:
        return self._get_indices_path(
            "mixture-instance-indices", source_path, extra_ids=(self.indices_dtype.__name__,)
        )

    def _write_document_indices(self):
        # A path can appear multiple times in a mixture (e.g. upsampling / repetition). Every
        # occurrence shares the same path-derived indices file but keeps its own token allocation
        # in ``_path_offset_index`` (Hamilton rounding can give occurrences different counts), so
        # size the shared file for the largest allocation across occurrences; otherwise a later,
        # larger occurrence would read past the generated tail.
        max_instances_by_path: Dict[str, int] = {}
        ordered_paths: List[PathOrStr] = []
        for idx, path in enumerate(self.paths):
            key = str(path)
            instances = self._path_offset_index[(key, idx)] // self.sequence_length
            if key in max_instances_by_path:
                max_instances_by_path[key] = max(max_instances_by_path[key], instances)
            else:
                max_instances_by_path[key] = instances
                ordered_paths.append(path)

        paths_needed: List[PathOrStr] = []
        num_reused = 0
        for path in ordered_paths:
            indices_path = self._get_instance_indices_path(path)
            if indices_path.is_file():
                num_reused += 1
                log.info(f"Reusing mixture instance indices for '{path}' at:\n'{indices_path}'")
            else:
                paths_needed.append(path)

        num_zero_instance = 0
        if paths_needed:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=_get_data_prep_max_workers()
            ) as executor:
                future_to_path: Dict[concurrent.futures.Future, PathOrStr] = {}
                for path in paths_needed:
                    indices_path = self._get_instance_indices_path(path)
                    log.info(f"Gathering instance indices for '{path}'...")
                    # NOTE: We limit the number of instances by total target token count // sequence length
                    max_instances = max_instances_by_path[str(path)]

                    # Sampling from small npy files can result in 0 instance indices.
                    # We skip processing these to avoid writing empty mmapped files.
                    if max_instances > 0:
                        future = executor.submit(
                            run_worker_func,
                            segment_documents_into_instances,
                            path,
                            indices_path,
                            max_sequence_length=self.sequence_length,
                            eos_token_id=self.eos_token_id,
                            dtype=self.dtype,
                            indices_dtype=self.indices_dtype,
                            sample=(max_instances, self._seed),
                            use_array_if_local=_get_data_prep_use_array_if_local(),
                        )
                        future_to_path[future] = path
                    else:
                        num_zero_instance += 1

                total_futures = len(future_to_path)
                log.info(
                    "Mixture instance index prep summary: "
                    f"{num_reused:,d} reusable, {total_futures:,d} to create, "
                    f"{num_zero_instance:,d} skipped with zero requested instances "
                    f"out of {len(self.paths):,d} selected paths"
                )

                if total_futures:
                    pending = set(future_to_path)
                    completed = 0
                    started_at = time.monotonic()
                    last_log_at = started_at
                    progress_interval = _get_data_prep_progress_interval()

                    while pending:
                        timeout = progress_interval if progress_interval > 0 else None
                        done, pending = concurrent.futures.wait(
                            pending,
                            timeout=timeout,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )

                        if not done:
                            _log_data_prep_progress(
                                "Mixture instance index prep",
                                completed=completed,
                                total=total_futures,
                                started_at=started_at,
                            )
                            last_log_at = time.monotonic()
                            continue

                        latest_path: Optional[PathOrStr] = None
                        for future in done:
                            path = future_to_path[future]
                            latest_path = path
                            _, total_instances = future.result()
                            completed += 1
                            log.info(
                                f"Created {total_instances:,d} instances of sequence length up to "
                                f"{self.sequence_length} from '{path}'"
                            )

                        now = time.monotonic()
                        if (
                            completed == total_futures
                            or progress_interval == 0
                            or now - last_log_at >= progress_interval
                        ):
                            _log_data_prep_progress(
                                "Mixture instance index prep",
                                completed=completed,
                                total=total_futures,
                                started_at=started_at,
                                latest_path=latest_path,
                            )
                            last_log_at = now

    # def _read_chunk_from_array(self, path: PathOrStr, index: int) -> torch.Tensor:
    #     indices_path = self._get_instance_indices_path(path)
    #     indices = load_array_slice_into_tensor(
    #         indices_path, index * 2, index * 2 + 2, self.indices_dtype
    #     )
    #     start_idx, end_idx = indices
    #     data = load_array_slice_into_tensor(path, int(start_idx), int(end_idx), self.dtype)
    #     return data

    def _get_file_size_and_length(self, path: PathOrStr, idx: int, dtype=None) -> Tuple[int, int]:
        dtype = dtype or self.dtype
        item_size = dtype(0).itemsize
        file_size = self._get_size_from_offset_index((path, idx))
        if (
            self.max_target_sequence_length is None
            or self.max_target_sequence_length == self.sequence_length
        ):
            return file_size, file_size // (item_size * self.sequence_length)
        elif self.max_target_sequence_length > self.sequence_length:
            num_max_seq_len_instances = file_size // (item_size * self.max_target_sequence_length)
            return (
                file_size,
                num_max_seq_len_instances
                * (self.max_target_sequence_length // self.sequence_length),
            )
        else:
            raise RuntimeError("invalid 'max_target_sequence_length' or 'sequence_length'")

    def _get_size_from_offset_index(self, path_index: Tuple[PathOrStr, int]) -> int:
        try:
            path, idx = path_index
            # Get size in bytes from tokens in the supplied index * itemsize
            return self._path_offset_index[(str(path), idx)] * self.dtype(0).itemsize
        except KeyError:
            raise OLMoEnvironmentError(f"Item not found in path index @ {path_index}")


class NumpyPaddedFSLDataset(NumpyFSLDataset):
    """
    An FSL dataset that creates a single instance from each document.
    The resulting instances will all have exactly ``sequence_length`` tokens, using padding if needed.
    """

    def __init__(
        self,
        *paths: PathOrStr,
        sequence_length: int,
        pad_token_id: int,
        eos_token_id: int,
        vocab_size: int,
        dtype: NumpyUIntTypes = np.uint16,
        bos_token_id: Optional[int] = None,
        metadata: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        include_instance_metadata: Optional[bool] = None,
        instance_filter_config: Optional[InstanceFilterConfig] = None,
        label_mask_paths: Optional[List[PathOrStr]] = None,
    ):
        super().__init__(
            *paths,
            sequence_length=sequence_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            dtype=dtype,
            metadata=metadata,
            include_instance_metadata=include_instance_metadata,
            bos_token_id=bos_token_id,
            instance_filter_config=instance_filter_config,
            label_mask_paths=label_mask_paths,
        )
        self._array_instance_offsets: Optional[Tuple[Tuple[int, int], ...]] = None

    @property
    def fingerprint_fields(self) -> Tuple[str, ...]:
        return (
            "vocab_size",
            "pad_token_id",
            "eos_token_id",
            "dtype",
            "max_target_sequence_length",
            "bos_token_id",
            "sequence_length",
        )

    @property
    def offsets(self) -> Tuple[Tuple[int, int], ...]:
        if self._array_instance_offsets is None:
            item_size = self.indices_dtype(0).itemsize
            num_instances_per_path = self.map(
                lambda path, _: get_file_size(self._get_instance_indices_path(path))
                // (item_size * 2)
            )
            array_instance_offsets = []
            start_offset = 0
            for num_instances in num_instances_per_path:
                array_instance_offsets.append((start_offset, start_offset + num_instances))
                start_offset += num_instances
            self._array_instance_offsets = tuple(array_instance_offsets)
        return self._array_instance_offsets

    @property
    def indices_dtype(
        self,
    ) -> NumpyUIntTypes:
        return np.uint32

    def prepare(self):
        if self.fs_local_rank == 0:
            log.info("Gathering dataset document indices...")
            self._write_instance_indices()
        barrier()
        len(self)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = super().__getitem__(index)
        pad_shape = (0, self.sequence_length - len(item["input_ids"]))
        if "label_mask" in item:
            item["label_mask"] = F.pad(item["label_mask"], pad_shape, value=False)
        else:
            item["label_mask"] = F.pad(
                torch.ones_like(item["input_ids"], dtype=torch.bool), pad_shape, value=False
            )
        item["input_ids"] = F.pad(item["input_ids"], pad_shape, value=self.pad_token_id)
        return item

    def _read_chunk_from_array(self, path: PathOrStr, index: int, dtype=None) -> torch.Tensor:
        indices_path = self._get_instance_indices_path(path)
        indices = load_array_slice_into_tensor(
            indices_path, index * 2, index * 2 + 2, self.indices_dtype
        )
        start_idx, end_idx = indices
        data = load_array_slice_into_tensor(path, int(start_idx), int(end_idx), dtype or self.dtype)
        return data

    def _get_instance_indices_path(self, source_path: PathOrStr) -> Path:
        return self._get_indices_path("instance-indices", source_path)

    def _write_instance_indices(self):
        paths_needed: List[PathOrStr] = []
        for path in self.paths:
            indices_path = self._get_instance_indices_path(path)
            if indices_path.is_file():
                log.info(f"Reusing instance indices for '{path}' at:\n'{indices_path}'")
            elif path not in paths_needed:
                paths_needed.append(path)

        if paths_needed:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=_get_data_prep_max_workers()
            ) as executor:
                futures = []
                for path in paths_needed:
                    indices_path = self._get_instance_indices_path(path)
                    log.info(f"Gathering instance indices for '{path}'...")
                    future = executor.submit(
                        run_worker_func,
                        segment_documents_into_instances,
                        path,
                        indices_path,
                        max_sequence_length=self.sequence_length,
                        eos_token_id=self.eos_token_id,
                        dtype=self.dtype,
                        indices_dtype=self.indices_dtype,
                        use_array_if_local=_get_data_prep_use_array_if_local(),
                    )
                    futures.append(future)

                concurrent.futures.wait(futures, return_when="FIRST_EXCEPTION")

                # Log results.
                for path, future in zip(paths_needed, futures):
                    _, total_instances = future.result()
                    log.info(
                        f"Created {total_instances:,d} instances of sequence length up to "
                        f"{self.sequence_length} from '{path}'"
                    )


class NumpyPackedFSLDataset(NumpyFSLDatasetBase):
    """
    An FSL dataset that packs documents into instances using the Optimized Best-Fit Decreasing (OBFD)
    algorithm described in `Fewer Truncations Improve Language Modeling <https://arxiv.org/pdf/2404.10830>`_.
    The resulting instances will all have exactly ``sequence_length`` tokens, using padding if needed.

    .. note::
        By default OBFD is applied to each source file separately since source files from the Dolma toolkit
        are usually large enough for OBFD to achieve very good compactness (minimal padding tokens)
        and so that we can parallelize the packing. However, you can pack instances from multiple
        consecutive source files together by setting ``source_group_size`` to a value greater than 1.

    .. tip::
        Although this shares much of its option plumbing with :class:`NumpyFSLDataset`, it bypasses
        that subclass and derives from :class:`NumpyFSLDatasetBase` so it can provide its own packing
        caches, offsets, and item materialisation logic. Subclassing :class:`NumpyFSLDataset` would
        require overriding nearly every behavior defined there.
    """

    def __init__(
        self,
        *paths: PathOrStr,
        sequence_length: int,
        pad_token_id: int,
        eos_token_id: int,
        vocab_size: int,
        dtype: NumpyUIntTypes = np.uint16,
        metadata: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        include_instance_metadata: Optional[bool] = None,
        generate_doc_lengths: bool = False,
        bos_token_id: Optional[int] = None,
        instance_filter_config: Optional[InstanceFilterConfig] = None,
        label_mask_paths: Optional[List[PathOrStr]] = None,
        long_doc_strategy: LongDocStrategy = LongDocStrategy.truncate,
        source_group_size: int = 1,
        use_array_if_local: Optional[bool] = None,
    ):
        super().__init__(
            *paths,
            sequence_length=sequence_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            dtype=dtype,
            metadata=metadata,
            include_instance_metadata=include_instance_metadata,
            generate_doc_lengths=generate_doc_lengths,
            bos_token_id=bos_token_id,
            instance_filter_config=instance_filter_config,
            label_mask_paths=label_mask_paths,
        )

        assert source_group_size >= 1

        self._long_doc_strategy = long_doc_strategy
        self._source_group_size = source_group_size
        self._use_array_if_local = use_array_if_local

        self._source_path_groups = list(chunked(self.paths, self.source_group_size))
        self._label_mask_path_groups: Optional[List[List[PathOrStr]]] = None
        self._metadata_groups = list(chunked(self._metadata, self.source_group_size))

        if self._label_mask_paths:
            self._label_mask_path_groups = list(
                chunked(self._label_mask_paths, self.source_group_size)
            )

        self._source_sizes: Optional[List[int]] = None
        self._source_size_groups: Optional[List[List[int]]] = None
        self._source_instance_offsets: Optional[Tuple[Tuple[int, int], ...]] = None
        self._num_instances: Optional[int] = None

    @property
    def fingerprint_fields(self) -> Tuple[str, ...]:
        fields: Tuple[str, ...] = (
            "vocab_size",
            "pad_token_id",
            "eos_token_id",
            "dtype",
            "long_doc_strategy",
            "bos_token_id",
            "sequence_length",
        )
        # For backwards compat, only add this when it's not the default.
        if self._source_group_size > 1:
            fields = fields + ("source_group_size",)
        # Likewise: this changes which document boundaries are used, and therefore the resulting
        # instances, so it has to participate in the fingerprint or a cached index built with the
        # other boundary source would be reused.
        if self._use_array_if_local is not None:
            fields = fields + ("use_array_if_local",)
        if self._packed_from_metadata_boundaries(self.paths):
            fields = fields + ("_metadata_fingerprint",)
        return fields

    @property
    def long_doc_strategy(self) -> LongDocStrategy:
        return self._long_doc_strategy

    @property
    def source_group_size(self) -> int:
        return self._source_group_size

    @property
    def use_array_if_local(self) -> Optional[bool]:
        return self._use_array_if_local

    @property
    def indices_dtype(
        self,
    ) -> NumpyUIntTypes:
        return np.uint64

    @property
    def source_instance_offsets(self) -> Tuple[Tuple[int, int], ...]:
        if self._source_instance_offsets is None:
            item_size = self.indices_dtype(0).itemsize
            num_instances_per_group = self.map(
                lambda path, _: get_file_size(path) // (item_size * 2),
                _paths=[
                    self._get_instance_offsets_path(*paths)
                    for paths in chunked(self.paths, self.source_group_size)
                ],
            )
            array_instance_offsets = []
            start_offset = 0
            for num_instances in num_instances_per_group:
                array_instance_offsets.append((start_offset, start_offset + num_instances))
                start_offset += num_instances
            self._source_instance_offsets = tuple(array_instance_offsets)
        return self._source_instance_offsets

    @property
    def source_sizes(self) -> List[int]:
        if self._source_sizes is None:
            item_size = self.dtype(0).itemsize
            self._source_sizes = self.map(lambda path, _: get_file_size(path) // item_size)
        return self._source_sizes

    @property
    def source_size_groups(self) -> List[List[int]]:
        if self._source_size_groups is None:
            self._source_size_groups = list(chunked(self.source_sizes, self.source_group_size))
        return self._source_size_groups

    def prepare(self):
        if self.fs_local_rank == 0:
            log.info("Packing document into instances...")
            self._pack_all_documents_into_instances()
        barrier()
        len(self)

    def __len__(self) -> int:
        if self._num_instances is None:
            self._num_instances = self.source_instance_offsets[-1][1]
        return self._num_instances

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = int(index)  # in case this is a numpy int type.
        index = index if index >= 0 else len(self) + index

        # The index of the source group.
        source_group_index: Optional[int] = None
        # The instance index within the source group.
        instance_index: Optional[int] = None
        for i, (instance_offset_start, instance_offset_end) in enumerate(
            self.source_instance_offsets
        ):
            if instance_offset_start <= index < instance_offset_end:
                source_group_index = i
                instance_index = index - instance_offset_start
                break

        if source_group_index is None or instance_index is None:
            raise IndexError(f"{index} is out of bounds for dataset of size {len(self)}")

        # All npy source file paths within the group.
        source_paths = self._source_path_groups[source_group_index]
        # The number of tokens in each npy source file within the group.
        source_sizes = self.source_size_groups[source_group_index]
        # All label mask paths for the group.
        label_mask_paths = (
            None
            if self._label_mask_path_groups is None
            else self._label_mask_path_groups[source_group_index]
        )

        document_indices_path = self._get_document_indices_path(*source_paths)
        instance_offsets_path = self._get_instance_offsets_path(*source_paths)
        docs_by_instance_path = self._get_docs_by_instance_path(*source_paths)

        # Load start and end document indices corresponding to instance.
        instance_indices = load_array_slice_into_tensor(
            instance_offsets_path,
            instance_index * 2,
            instance_index * 2 + 2,
            self.indices_dtype,
        ).tolist()
        instance_start, instance_end = instance_indices

        # Load document IDs corresponding to instance.
        document_ids = load_array_slice_into_tensor(
            docs_by_instance_path,
            instance_start,
            instance_end,
            self.indices_dtype,
        ).tolist()

        # Load token IDs and maybe label masks for each document.
        document_token_ids: List[torch.Tensor] = []
        document_label_masks: Optional[List[torch.Tensor]] = (
            None if label_mask_paths is None else []
        )
        for document_id in document_ids:
            document_indices = load_array_slice_into_tensor(
                document_indices_path, document_id * 2, document_id * 2 + 2, self.indices_dtype
            ).tolist()
            document_start, document_end = document_indices

            # Pick out the right source file from the source group by comparing the starting
            # index (in tokens) of the document to the starting index of each source within the group.
            source_path: Optional[PathOrStr] = None
            label_mask_path: Optional[PathOrStr] = None
            source_start = 0
            for i, (source_path, source_size) in enumerate(zip(source_paths, source_sizes)):
                if source_start <= document_start < (source_start + source_size):
                    document_start -= source_start
                    document_end -= source_start
                    if label_mask_paths is not None:
                        label_mask_path = label_mask_paths[i]
                    break
                else:
                    source_start += source_size
            else:
                raise RuntimeError("we shouldn't be here!")

            assert source_path is not None
            document_token_ids.append(
                load_array_slice_into_tensor(source_path, document_start, document_end, self.dtype)
            )
            if label_mask_path is not None:
                assert document_label_masks is not None
                document_label_masks.append(
                    load_array_slice_into_tensor(
                        label_mask_path, document_start, document_end, np.bool_
                    )
                )

        # Combine token IDs and maybe label masks for each document.
        input_ids = torch.cat(document_token_ids)
        label_mask = None if document_label_masks is None else torch.cat(document_label_masks)

        # Pad to target sequence length.
        pad_shape = (0, self.sequence_length - input_ids.numel())
        if label_mask is not None:
            label_mask = F.pad(label_mask, pad_shape, value=False)
        else:
            label_mask = F.pad(torch.ones_like(input_ids, dtype=torch.bool), pad_shape, value=False)
        input_ids = F.pad(input_ids, pad_shape, value=self.pad_token_id)

        # Prepare final output.
        out: Dict[str, Any] = {"input_ids": input_ids, "label_mask": label_mask}
        if self.instance_filter_config is not None:
            out["instance_mask"] = self._validate_instance(input_ids, self.instance_filter_config)
        if self._include_instance_metadata:
            metadata = self._metadata_groups[source_group_index]
            out["metadata"] = deepcopy(metadata)
        if self._generate_doc_lengths:
            if self._packed_from_metadata_boundaries(source_paths):
                # Rescanning would merge a document that lacks a terminator into the one after it
                # -- exactly the documents this setting exists to support. `doc_lens` drives the
                # block-diagonal attention mask, so that would let tokens attend across a real
                # document boundary. The loaded documents give the lengths exactly.
                doc_lens = [document.numel() for document in document_token_ids]
                if (padding := input_ids.numel() - sum(doc_lens)) > 0:
                    # Keep the scanner's padding segmentation, including one segment per EOS
                    # padding token. With BOS, include the final real token to detect a boundary
                    # between the last document and padding (e.g. when padding is BOS).
                    if self.bos_token_id is None:
                        doc_lens.extend(
                            get_document_lengths(input_ids[-padding:], self.eos_token_id).tolist()
                        )
                    else:
                        tail_lens = get_document_lengths(
                            input_ids[-padding - 1 :], self.eos_token_id, self.bos_token_id
                        ).tolist()
                        doc_lens[-1] += tail_lens[0] - 1
                        doc_lens.extend(tail_lens[1:])
                out["doc_lens"] = torch.tensor(doc_lens, dtype=torch.int32)
            else:
                out["doc_lens"] = get_document_lengths(
                    input_ids, self.eos_token_id, bos_token_id=self.bos_token_id
                )
        return out

    def _packed_from_metadata_boundaries(self, source_paths: Sequence[PathOrStr]) -> bool:
        """Whether packing took its document boundaries from the metadata file.

        `iter_document_indices` reads the metadata for any URL source whatever
        `use_array_if_local` says, so the effective boundary source -- not the raw option --
        decides whether re-deriving boundaries by scanning for EOS would disagree with how the
        instance was packed.
        """
        return self._use_array_if_local is False or any(is_url(path) for path in source_paths)

    def _get_metadata_hash(self, source_path: PathOrStr, _: int) -> Optional[str]:
        if not self._packed_from_metadata_boundaries([source_path]):
            return None
        metadata_path = resource_path(
            os.path.dirname(source_path), os.path.basename(source_path).replace(".npy", ".csv.gz")
        )
        digest = hashlib.sha256()
        with metadata_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @cached_property
    def _metadata_hashes(self) -> Dict[PathOrStr, str]:
        # Snapshot once per dataset, like file_sizes: no metadata I/O in __getitem__.
        # Hash contents so same-size corrections also invalidate cached packing results.
        return {
            path: digest
            for path, digest in zip(self.paths, self.map(self._get_metadata_hash))
            if digest is not None
        }

    @property
    def _metadata_fingerprint(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(
            (os.path.basename(path), digest) for path, digest in self._metadata_hashes.items()
        )

    def _packing_cache_extra_ids(self, source_paths: Sequence[PathOrStr]) -> Tuple[str, ...]:
        # These key the on-disk packing caches, which are looked up by path and never compared
        # against `fingerprint`. Anything that changes the packing result therefore has to appear
        # here as well, or a stale cache from a previous run would be reused.
        extra_ids: Tuple[str, ...] = (self._long_doc_strategy, self.indices_dtype.__name__)
        # For backwards compat, only add this when it's not the default, so existing caches
        # built before the option existed are still found.
        if self._use_array_if_local is not None:
            extra_ids = extra_ids + (f"use_array_if_local={self._use_array_if_local}",)
        if self._packed_from_metadata_boundaries(source_paths):
            extra_ids += tuple(
                f"metadata={path},sha256={self._metadata_hashes[path]}"
                for path in source_paths
                if path in self._metadata_hashes
            )
        return extra_ids

    def _get_document_indices_path(self, *source_paths: PathOrStr) -> Path:
        return self._get_indices_path(
            "document-indices",
            *source_paths,
            extra_ids=self._packing_cache_extra_ids(source_paths),
        )

    def _get_instance_offsets_path(self, *source_paths: PathOrStr) -> Path:
        return self._get_indices_path(
            "instance-offsets",
            *source_paths,
            extra_ids=self._packing_cache_extra_ids(source_paths),
        )

    def _get_docs_by_instance_path(self, *source_paths: PathOrStr) -> Path:
        return self._get_indices_path(
            "documents-by-instance",
            *source_paths,
            extra_ids=self._packing_cache_extra_ids(source_paths),
        )

    def _pack_documents_from_source_into_instances(
        self, *source_paths: PathOrStr
    ) -> Tuple[int, int]:
        document_indices_path = self._get_document_indices_path(*source_paths)
        instance_offsets_path = self._get_instance_offsets_path(*source_paths)
        docs_by_instance_path = self._get_docs_by_instance_path(*source_paths)

        instances, document_indices, total_tokens = pack_documents_into_instances(
            *source_paths,
            max_sequence_length=self.sequence_length,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            dtype=self.dtype,
            indices_dtype=self.indices_dtype,
            long_doc_strategy=self._long_doc_strategy,
            use_array_if_local=self._use_array_if_local,
        )
        document_indices = document_indices.reshape(-1)

        instance_start_offset = 0
        instance_offsets_list: List[int] = []
        documents_by_instance_list: List[int] = []
        for instance in instances:
            instance_offsets_list.append(instance_start_offset)
            instance_offsets_list.append(instance_start_offset + len(instance))
            instance_start_offset += len(instance)
            documents_by_instance_list.extend(instance)

        # shape: (num_instances * 2,)
        instance_offsets = np.array(instance_offsets_list, dtype=self.indices_dtype)
        # shape: (num_documents,)
        docs_by_instance = np.array(documents_by_instance_list, dtype=self.indices_dtype)

        write_array_to_disk(document_indices, document_indices_path)
        write_array_to_disk(instance_offsets, instance_offsets_path)
        write_array_to_disk(docs_by_instance, docs_by_instance_path)

        return len(instances), total_tokens

    def _pack_all_documents_into_instances(self):
        # Collect all sources that need to be packed (no cache hit).
        sources_needed: List[List[PathOrStr]] = []
        for source_paths in chunked(self.paths, self.source_group_size):
            document_indices_path = self._get_document_indices_path(*source_paths)
            instance_offsets_path = self._get_instance_offsets_path(*source_paths)
            docs_by_instance_path = self._get_docs_by_instance_path(*source_paths)
            if (
                document_indices_path.is_file()
                and instance_offsets_path.is_file()
                and docs_by_instance_path.is_file()
            ):
                log.info(f"Reusing cached packing results for {source_paths}")
            elif source_paths not in sources_needed:
                sources_needed.append(source_paths)

        if sources_needed:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=_get_data_prep_max_workers()
            ) as executor:
                future_to_paths: Dict[concurrent.futures.Future, List[PathOrStr]] = {}
                for source_paths in sources_needed:
                    log.info(f"Packing documents from {source_paths} into instances...")
                    future = executor.submit(
                        run_worker_func,
                        self._pack_documents_from_source_into_instances,
                        *source_paths,
                    )
                    future_to_paths[future] = source_paths

                total_futures = len(future_to_paths)
                log.info(
                    "Packed FSL data prep summary: %d source group(s) to pack "
                    "out of %d group(s), source_group_size=%d, sequence_length=%d",
                    total_futures,
                    len(self._source_path_groups),
                    self.source_group_size,
                    self.sequence_length,
                )

                pending = set(future_to_paths)
                completed = 0
                started_at = time.monotonic()
                last_log_at = started_at
                progress_interval = _get_data_prep_progress_interval()

                while pending:
                    timeout = progress_interval if progress_interval > 0 else None
                    done, pending = concurrent.futures.wait(
                        pending,
                        timeout=timeout,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )

                    if not done:
                        _log_data_prep_progress(
                            "Packed FSL data prep",
                            completed=completed,
                            total=total_futures,
                            started_at=started_at,
                            unit="groups",
                        )
                        last_log_at = time.monotonic()
                        continue

                    latest_path: Optional[PathOrStr] = None
                    for future in done:
                        source_paths = future_to_paths[future]
                        latest_path = source_paths[0] if source_paths else None
                        total_instances, total_tokens = future.result()
                        completed += 1
                        total_padding = self.sequence_length * total_instances - total_tokens
                        avg_padding = total_padding / total_instances if total_instances else 0.0
                        log.info(
                            f"Packed {total_tokens:,} tokens from {source_paths} into {total_instances:,d} instances "
                            f"of sequence length {self.sequence_length:,d} using an average of "
                            f"{avg_padding:.1f} padding tokens per instance."
                        )

                    now = time.monotonic()
                    if (
                        completed == total_futures
                        or progress_interval == 0
                        or now - last_log_at >= progress_interval
                    ):
                        _log_data_prep_progress(
                            "Packed FSL data prep",
                            completed=completed,
                            total=total_futures,
                            started_at=started_at,
                            latest_path=latest_path,
                            unit="groups",
                        )
                        last_log_at = now


class NumpyInterleavedFSLDataset(NumpyPaddedFSLDataset):
    """
    A version of :class:`NumpyPaddedFSLDataset` that creates a single instance by chunking documents and
    interleaving these chunks. The resulting instances may be padded out to ``sequence_length``.
    """

    def __init__(
        self,
        *paths: PathOrStr,
        sequence_length: int,
        pad_token_id: int,
        eos_token_id: int,
        vocab_size: int,
        seed: int,
        docs_per_instance: int,
        chunks_per_doc: int,
        dtype: NumpyUIntTypes = np.uint16,
        metadata: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        include_instance_metadata: Optional[bool] = None,
        instance_filter_config: Optional[InstanceFilterConfig] = None,
        label_mask_paths: Optional[List[PathOrStr]] = None,
        bos_token_id: Optional[int] = None,
        interleaving_exempt_paths: Optional[List[PathOrStr]] = None,
    ):
        if sequence_length % docs_per_instance != 0:
            raise OLMoConfigurationError(
                "'sequence_length' must be a multiple of 'docs_per_instance'"
            )
        if sequence_length % chunks_per_doc != 0:
            raise OLMoConfigurationError("'sequence_length' must be a multiple of 'chunks_per_doc'")

        super().__init__(
            *paths,
            sequence_length=sequence_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            dtype=dtype,
            bos_token_id=bos_token_id,
            metadata=metadata,
            include_instance_metadata=include_instance_metadata,
            instance_filter_config=instance_filter_config,
            label_mask_paths=label_mask_paths,
        )
        self._docs_per_instance = docs_per_instance
        self._chunks_per_doc = chunks_per_doc
        self._seed = seed
        self._interleaving_exempt_paths = interleaving_exempt_paths
        self._num_interleaving_exempt_instances: Optional[int] = None
        self._num_interleavable_instances: Optional[int] = None

    @property
    def fingerprint_fields(self) -> Tuple[str, ...]:
        return (
            "vocab_size",
            "pad_token_id",
            "eos_token_id",
            "dtype",
            "_docs_per_instance",
            "_seed",
            "_interleaving_exempt_paths",
            "max_target_sequence_length",
            "bos_token_id",
            "sequence_length",
        )

    def __len__(self) -> int:
        if self._num_instances is None:
            item_size = self.indices_dtype(0).itemsize

            interleavable_indices_path = self._get_interleaveable_indices_path()
            num_interleavable_instances = get_file_size(interleavable_indices_path) // item_size

            interleaving_exempt_indices_path = self._get_interleaving_exempt_indices_path()
            num_interleaving_exempt_instances = (
                (get_file_size(interleaving_exempt_indices_path) // item_size)
                if interleaving_exempt_indices_path.is_file()
                else 0
            )

            self._num_interleavable_instances = num_interleavable_instances
            self._num_interleaving_exempt_instances = num_interleaving_exempt_instances
            self._num_instances = (
                num_interleavable_instances // self._docs_per_instance
                + num_interleaving_exempt_instances
            )
        return self._num_instances

    def prepare(self):
        if self.fs_local_rank == 0:
            log.info("Gathering dataset document and interleaving indices...")
            self._write_instance_indices()
            self._write_docs_interleaving_indices()
        barrier()
        len(self)

    def _write_docs_interleaving_indices(self):
        interleavable_indices_path = self._get_interleaveable_indices_path()
        interleaving_exempt_indices_path = self._get_interleaving_exempt_indices_path()
        if interleavable_indices_path.is_file() and (
            self._interleaving_exempt_paths is None or interleaving_exempt_indices_path.is_file()
        ):
            log.info(
                f"Reusing all document interleaving indices at:\n'{interleavable_indices_path}'"
            )
        else:
            log.info(
                f"Generating all document interleaving indices to:\n'{interleavable_indices_path}..."
            )

            if self._interleaving_exempt_paths:
                interleaving_exempt_doc_indices = [
                    instance_num
                    for i_offset, (start, end) in enumerate(self.offsets)
                    for instance_num in range(start, end)
                    if self.paths[i_offset] in self._interleaving_exempt_paths
                ]

                with memmap_to_write(
                    interleaving_exempt_indices_path,
                    dtype=self.indices_dtype,
                    shape=(len(interleaving_exempt_doc_indices),),
                ) as interleaving_exempt_indices:
                    interleaving_exempt_indices[:] = interleaving_exempt_doc_indices

                interleavable_doc_indices = sorted(
                    set(range(self.offsets[-1][1])) - set(interleaving_exempt_doc_indices)
                )
            else:
                interleavable_doc_indices = list(range(self.offsets[-1][1]))

            with memmap_to_write(
                interleavable_indices_path,
                dtype=self.indices_dtype,
                shape=(len(interleavable_doc_indices),),
            ) as interleavable_indices:
                interleavable_indices[:] = get_rng(self._seed).permutation(
                    interleavable_doc_indices
                )

    def _remove_special_tokens_and_interleave(
        self,
        tensors: List[torch.Tensor],
        tensors_non_special_indices: List[Tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        cleaned_tensors: List[torch.Tensor] = [
            tensor[non_special_indices]
            for tensor, non_special_indices in zip(tensors, tensors_non_special_indices)
        ]

        chunked_tensors = [
            cleaned_tensor.tensor_split(self._chunks_per_doc) for cleaned_tensor in cleaned_tensors
        ]
        return torch.cat(
            [
                chunked_tensor[i]
                for i in range(self._chunks_per_doc)
                for chunked_tensor in chunked_tensors
            ]
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = int(index)  # in case this is a numpy int type.
        pos_index = index if index >= 0 else len(self) + index

        assert self._num_interleaving_exempt_instances is not None
        if self._interleaving_exempt_paths and pos_index < self._num_interleaving_exempt_instances:
            interleaving_exempt_indices_path = self._get_interleaving_exempt_indices_path()
            doc_index = load_array_slice_into_tensor(
                interleaving_exempt_indices_path,
                pos_index,
                pos_index + 1,
                self.indices_dtype,
            ).tolist()[0]

            return super().__getitem__(doc_index)

        pos_index -= self._num_interleaving_exempt_instances
        assert self._num_interleavable_instances is not None
        assert pos_index < self._num_interleavable_instances

        interleaving_indices_path = self._get_interleaveable_indices_path()
        interleaving_indices = load_array_slice_into_tensor(
            interleaving_indices_path,
            pos_index * self._docs_per_instance,
            pos_index * self._docs_per_instance + self._docs_per_instance,
            self.indices_dtype,
        ).tolist()

        docs: List[Dict[str, Any]] = []
        for doc_index in interleaving_indices:
            doc = super().__getitem__(doc_index)

            # Shrink the documents down, so that interleaving them does not exceed the sequence length.
            doc["input_ids"] = doc["input_ids"][: self.sequence_length // self._docs_per_instance]
            doc["label_mask"] = doc["label_mask"][: self.sequence_length // self._docs_per_instance]

            docs.append(doc)

        for doc in docs:
            if doc.keys() != docs[0].keys():
                raise RuntimeError(
                    f"Trying to interleave documents when dataset docs have different keys: {docs[0].keys()}, {doc.keys()}."
                )

        item: Dict[str, Any] = {}

        docs_non_special_token_indices = []
        for doc in docs:
            special_tokens_mask = torch.logical_or(
                doc["input_ids"] == self.pad_token_id,
                doc["input_ids"] == self.eos_token_id,
            )
            if self.bos_token_id is not None:
                special_tokens_mask = torch.logical_or(
                    special_tokens_mask,
                    doc["input_ids"] == self.bos_token_id,
                )

            non_special_token_indices = torch.nonzero(
                torch.logical_not(special_tokens_mask),
                as_tuple=True,
            )
            docs_non_special_token_indices.append(non_special_token_indices)

        item["input_ids"] = self._remove_special_tokens_and_interleave(
            [doc["input_ids"] for doc in docs], docs_non_special_token_indices
        )
        item["label_mask"] = self._remove_special_tokens_and_interleave(
            [doc["label_mask"] for doc in docs], docs_non_special_token_indices
        )

        # Add bos and tokens if there is space after interleaving.
        if self.bos_token_id is not None and len(item["input_ids"]) < self.sequence_length:
            item["input_ids"] = F.pad(item["input_ids"], (1, 0), value=self.bos_token_id)
            item["label_mask"] = F.pad(item["label_mask"], (1, 0), value=True)
        if len(item["input_ids"]) < self.sequence_length:
            item["input_ids"] = F.pad(item["input_ids"], (0, 1), value=self.eos_token_id)
            item["label_mask"] = F.pad(item["label_mask"], (0, 1), value=True)

        pad_shape = (0, self.sequence_length - len(item["input_ids"]))
        item["input_ids"] = F.pad(item["input_ids"], pad_shape, value=self.pad_token_id)
        item["label_mask"] = F.pad(item["label_mask"], pad_shape, value=False)

        if "instance_mask" in docs[0]:
            item["instance_mask"] = all([doc["instance_mask"] for doc in docs])

        if "metadata" in docs[0]:
            metadata = docs[0]["metadata"]
            for doc in docs:
                doc_metadata = docs[0]["metadata"]
                if metadata != doc_metadata:
                    raise RuntimeError(
                        f"Trying to interleave documents when dataset docs have different metadata: {metadata}, {doc_metadata}."
                    )
            item["metadata"] = metadata

        if "doc_lens" in docs[0]:
            raise RuntimeError("Document lengths unexpectedly found.")

        return item

    def _get_instance_indices_path(self, source_path: PathOrStr) -> Path:
        return self._get_indices_path(
            "instance-indices", source_path, extra_ids=(str(self._docs_per_instance),)
        )

    def _get_interleaveable_indices_path(self) -> Path:
        return self.work_dir / f"dataset-{self.fingerprint}" / "interleavable-docs-indices.npy"

    def _get_interleaving_exempt_indices_path(self) -> Path:
        return (
            self.work_dir / f"dataset-{self.fingerprint}" / "interleaving-exempt-docs-indices.npy"
        )


@dataclass
class VSLCurriculum:
    """
    Base class for variable sequence length curriculums. These determine the sampling
    probability of batches from each bucket throughout training with a :class:`NumpyVSLDataset`.
    """

    @abstractmethod
    def batches_per_bucket(
        self, dataset: NumpyVSLDataset, global_batch_size: int
    ) -> List[Tuple[int, int]]:
        raise NotImplementedError

    @abstractmethod
    def get_batch_indices(
        self, batches_per_bucket: Sequence[Tuple[int, int]], seed: int
    ) -> np.ndarray:
        raise NotImplementedError

    def get_total_batches(self, batches_per_bucket: Sequence[Tuple[int, int]]) -> int:
        return sum([batches for _, batches in batches_per_bucket])

    def log_buckets(
        self,
        dataset: NumpyVSLDataset,
        global_batch_size: int,
        batches_per_bucket: Sequence[Tuple[int, int]],
    ):
        natural_batches_per_bucket = VSLNaturalCurriculum().batches_per_bucket(
            dataset, global_batch_size
        )
        for i, (seq_len, num_batches) in enumerate(batches_per_bucket):
            num_natural_batches = natural_batches_per_bucket[i][1]
            if num_batches != num_natural_batches:
                log.info(
                    f"- bucket {i}:   sequence length {seq_len:>6d} => {num_batches:>6d} batches "
                    f"used ({num_natural_batches:d} total)"
                )
            else:
                log.info(
                    f"- bucket {i}:   sequence length {seq_len:>6d} => {num_batches:>6d} batches"
                )

    @property
    @abstractmethod
    def short_str(self) -> str:
        """
        Return a unique human-readable identifier for the instance.
        """
        raise NotImplementedError


@dataclass
class VSLNaturalCurriculum(VSLCurriculum):
    """
    Implements the natural curriculum from
    `Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum
    <https://arxiv.org/pdf/2405.13226>`_.
    """

    def batches_per_bucket(
        self, dataset: NumpyVSLDataset, global_batch_size: int
    ) -> List[Tuple[int, int]]:
        batches_per_bucket = []
        for seq_len, num_instances in dataset.instances_per_bucket:
            instances_per_batch = global_batch_size // seq_len
            batches = num_instances // instances_per_batch
            batches_per_bucket.append((seq_len, batches))
        return batches_per_bucket

    def get_batch_indices(
        self, batches_per_bucket: Sequence[Tuple[int, int]], seed: int
    ) -> np.ndarray:
        total_batches = self.get_total_batches(batches_per_bucket)
        batch_indices = np.arange(total_batches, dtype=np.uint32)
        rng = get_rng(seed)
        # Put a batch with the largest sequence length first to catch OOMs early.
        idx = rng.integers(total_batches - batches_per_bucket[-1][1], total_batches)
        batch = batch_indices[idx]
        batch_indices[idx] = batch_indices[0]
        batch_indices[0] = batch
        rng.shuffle(batch_indices[1:])
        return batch_indices

    @property
    def short_str(self) -> str:
        return "vsl-natural"


@dataclass
class VSLGrowthCurriculum(VSLCurriculum):
    """
    A base class for growth curriculums, like :class:`VSLGrowP2Curriculum` and :class:`VSLGrowLinearCurriculum`.
    """

    num_cycles: int = 8
    """
    The number of cycles in the curriculum.
    """
    balanced: bool = False
    """
    Whether or not to balance the number of batches in each bucket.

    .. note::
        Balancing the number of batches requires dropping more data.
    """

    def batches_per_bucket(
        self, dataset: NumpyVSLDataset, global_batch_size: int
    ) -> List[Tuple[int, int]]:
        actual_batches_per_bucket = VSLNaturalCurriculum().batches_per_bucket(
            dataset, global_batch_size
        )
        if self.balanced:
            batches_per_bucket = min([batches for _, batches in actual_batches_per_bucket])
            batches_per_bucket = self.num_cycles * (batches_per_bucket // self.num_cycles)
            return [(seq_len, batches_per_bucket) for seq_len, _ in actual_batches_per_bucket]
        else:
            return [
                (seq_len, self.num_cycles * (batches_per_bucket // self.num_cycles))
                for seq_len, batches_per_bucket in actual_batches_per_bucket
            ]

    def get_cycle_distribution(
        self, indices: np.ndarray, batches_per_bucket: Sequence[Tuple[int, int]], cycle: int = 0
    ) -> List[List[int]]:
        cycle_length = indices.shape[0] // self.num_cycles
        cycle_indices = indices[cycle * cycle_length : (cycle * cycle_length) + cycle_length]
        distribution: List[List[int]] = []
        for subcycle in np.array_split(cycle_indices, len(batches_per_bucket)):
            distribution.append([])
            bucket_offset_start = 0
            bucket_offset_end = 0
            for _, num_batches in batches_per_bucket:
                bucket_offset_end += num_batches
                count = ((subcycle >= bucket_offset_start) & (subcycle < bucket_offset_end)).sum()
                distribution[-1].append(count)
                bucket_offset_start += num_batches
        return distribution

    def get_batch_indices(
        self, batches_per_bucket: Sequence[Tuple[int, int]], seed: int
    ) -> np.ndarray:
        # Shortest sequence length first.
        assert batches_per_bucket[0][0] < batches_per_bucket[-1][0]

        rng = get_rng(seed)
        num_buckets = len(batches_per_bucket)

        log.info(f"Constructing {self.__class__.__name__} curriculum with {num_buckets} buckets")

        cycles: List[np.ndarray] = []
        for cycle in range(self.num_cycles):
            # Now we need to chunk the batch indices *within* each bucket in this cycle into the batch
            # indices for each sub-cycle.
            # At the same time we'll translate those *within* bucket indices into global batch indices
            # by adding the right offset for each bucket.
            all_bucket_subcycle_batches: List[List[np.ndarray]] = []
            for bucket in range(num_buckets):
                # This is how many batches we'll pull from this bucket for each cycle.
                batch_counts_per_cycle_this_bucket = divide_into_buckets(
                    batches_per_bucket[bucket][1], self.num_cycles
                )
                # These are the batch indices *within* this bucket that we'll use for this cycle.
                batches_this_cycle_this_bucket = chunk_array(
                    np.arange(0, batches_per_bucket[bucket][1], dtype=np.uint32),
                    batch_counts_per_cycle_this_bucket,
                )[cycle]
                bucket_offset = sum([b for _, b in batches_per_bucket[:bucket]])
                bucket_subcycle_batch_counts = self._get_num_bucket_batches_for_cycle(
                    bucket, num_buckets, batch_counts_per_cycle_this_bucket[cycle]
                )
                bucket_subcycle_batches = chunk_array(
                    bucket_offset + batches_this_cycle_this_bucket, bucket_subcycle_batch_counts
                )
                all_bucket_subcycle_batches.append(bucket_subcycle_batches)

            # Now we'll build each full syb-cycle by concatenating all of the bucket sub-cycle batches
            # together and shuffling.
            all_subsycles: List[np.ndarray] = []
            for subcycle in range(num_buckets):
                subsycle_batches: List[np.ndarray] = []
                for bucket in range(num_buckets):
                    subsycle_batches.append(all_bucket_subcycle_batches[bucket][subcycle])
                res = np.concatenate(subsycle_batches)
                rng.shuffle(res)
                all_subsycles.append(res)
            del all_bucket_subcycle_batches

            # Finally we can concatenate all of the subsycles together to form the complete cycle.
            cycles.append(np.concatenate(all_subsycles))
            del all_subsycles

        indices = np.concatenate(cycles)
        del cycles

        # Make sure the very first batch has the longest sequence length (is from the last bucket).
        # That way OOMs should happen right away.
        final_bucket_start = sum([b for _, b in batches_per_bucket[:-1]])
        first_long_seq_len_batch = np.argmax(indices >= final_bucket_start)
        batch = indices[first_long_seq_len_batch]
        indices[first_long_seq_len_batch] = indices[0]
        indices[0] = batch

        assert indices.shape[0] == self.get_total_batches(batches_per_bucket)
        return indices

    @classmethod
    @abstractmethod
    def _get_bucket_odds_for_cycle(cls, bucket_idx: int, num_buckets: int) -> List[int]:
        raise NotImplementedError

    @classmethod
    def _get_num_bucket_batches_for_cycle(
        cls, bucket_idx: int, num_buckets: int, num_batches: int
    ) -> List[int]:
        odds = cls._get_bucket_odds_for_cycle(bucket_idx, num_buckets)
        divisor = sum(odds)
        props = [o / divisor for o in odds]
        out = []
        total = 0
        for p in props:
            n = round(p * num_batches)
            total += n
            out.append(n)
        if total < num_batches:
            out[-1] += num_batches - total
        return out


@dataclass
class VSLGrowP2Curriculum(VSLGrowthCurriculum):
    """
    Implements the "Grow-P2" curriculum from
    `Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum
    <https://arxiv.org/pdf/2405.13226>`_.
    """

    @classmethod
    def _get_bucket_odds_for_cycle(cls, bucket_idx: int, num_buckets: int) -> List[int]:
        all_odds = []
        start_odds = num_buckets - bucket_idx
        for cycle in range(num_buckets):
            exp = (
                start_odds + cycle
                if start_odds + cycle <= num_buckets
                else start_odds - ((start_odds + cycle) % num_buckets)
            )
            all_odds.append(2 ** (exp - 1))
        return all_odds

    @property
    def short_str(self) -> str:
        return f"vsl-grow-p2-{self.num_cycles}-cycle{'-balanced' if self.balanced else ''}"


@dataclass
class VSLGrowLinearCurriculum(VSLGrowthCurriculum):
    """
    Implements the "Grow-Linear" curriculum from
    `Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum
    <https://arxiv.org/pdf/2405.13226>`_.
    """

    @classmethod
    def _get_bucket_odds_for_cycle(cls, bucket_idx: int, num_buckets: int) -> List[int]:
        all_odds = []
        start_odds = num_buckets - bucket_idx
        for cycle in range(num_buckets):
            odds = (
                start_odds + cycle
                if start_odds + cycle <= num_buckets
                else start_odds - ((start_odds + cycle) % num_buckets)
            )
            all_odds.append(odds)
        return all_odds

    @property
    def short_str(self) -> str:
        return f"vsl-grow-linear-{self.num_cycles}-cycle{'-balanced' if self.balanced else ''}"


class NumpyVSLDataset(NumpyDatasetBase, Dataset[Dict[str, Any]]):
    """
    A variable sequence length (VSL) numpy array-backed dataset. This is used to inject a sequence
    length-based curriculum during training as introduced in
    `Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum
    <https://arxiv.org/pdf/2405.13226>`_.

    This dataset creates instances of token IDs with lengths that are powers of 2
    between ``min_sequence_length`` (which must be a power of 2) and ``max_sequence_length``
    (also a power a 2). Some tokens will be discarded unless ``min_sequence_length`` is 1.

    .. important::
        No special tokens are added to the input IDs so it's assumed that if you want
        EOS tokens between documents, for example, those will already be in the array.

    :param paths: Paths or URLs to numpy token ID arrays.
    :param pad_token_id: The ID of the padding token.
    :param eos_token_id: The ID of the EOS token.
    :param max_sequence_length: The maximum allowed sequence length. A power of 2, e.g. '4096'.
    :param min_sequence_length: The minimum allowed sequence length. A power of 2, e.g. '256'.
    :param curriculum: The variable sequence length curriculum. Determines the sampling
        probability of batches from each bucket throughout training.
    :param dtype: The numpy datatype of the arrays.
    :param metadata: Metadata to add to each item. This should be a dictionary or a list of dictionaries
        with the same number of items as there are paths.
    :param include_instance_metadata: If ``True`` (the default), each instance returned from
        :meth:`__getitem__()` will include the metadata from its source.
    """

    def __init__(
        self,
        *paths: PathOrStr,
        pad_token_id: int,
        eos_token_id: int,
        vocab_size: int,
        max_sequence_length: int,
        min_sequence_length: int = 256,
        curriculum: Optional[VSLCurriculum] = None,
        dtype: NumpyUIntTypes = np.uint16,
        metadata: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        include_instance_metadata: Optional[bool] = None,
        instance_filter_config: Optional[InstanceFilterConfig] = None,
    ):
        if math.log(max_sequence_length, 2) % 1 != 0:
            raise OLMoConfigurationError("'max_sequence_length' must be a power of 2")

        if math.log(min_sequence_length, 2) % 1 != 0:
            raise OLMoConfigurationError("'min_sequence_length' must be a power of 2")

        if max_sequence_length <= min_sequence_length:
            raise OLMoConfigurationError(
                "'max_sequence_length' should be bigger than 'min_sequence_length'"
            )

        if include_instance_metadata is None and metadata:
            include_instance_metadata = True

        if isinstance(metadata, list):
            if len(metadata) != len(paths):
                raise OLMoConfigurationError(
                    "'metadata' should have the same length as the number of file paths"
                )
        else:
            metadata = [metadata or {}] * len(paths)

        super().__init__(
            *paths,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            dtype=dtype,
        )
        self._metadata = metadata
        self._include_instance_metadata = include_instance_metadata
        self._max_sequence_length = max_sequence_length
        self._min_sequence_length = min_sequence_length
        self._curriculum = curriculum or VSLNaturalCurriculum()
        self._num_instances: Optional[int] = None
        self._array_offsets: Optional[Tuple[Tuple[int, int], ...]] = None
        self._lengths_dtype: Optional[NumpyUIntTypes] = None
        self._instances_per_bucket: Optional[Tuple[Tuple[int, int], ...]] = None
        self.instance_filter_config = instance_filter_config

    @property
    def fingerprint_fields(self) -> Tuple[str, ...]:
        """
        Extra values to include when calculating the data contents :data:`fingerprint`.
        """
        return (
            "vocab_size",
            "pad_token_id",
            "eos_token_id",
            "dtype",
            "min_sequence_length",
            "max_sequence_length",
            "curriculum",
        )

    @property
    def max_sequence_length(self) -> int:
        return self._max_sequence_length

    @property
    def min_sequence_length(self) -> int:
        return self._min_sequence_length

    @property
    def curriculum(self) -> VSLCurriculum:
        return self._curriculum

    @property
    def all_sequence_lengths(self) -> List[int]:
        min_exp = int(math.log(self.min_sequence_length, 2))
        max_exp = int(math.log(self.max_sequence_length, 2))
        return [2**exp for exp in range(min_exp, max_exp + 1)]

    @property
    def offsets(self) -> Tuple[Tuple[int, int], ...]:
        """
        Gives the global start and end instance indices for each data file in the dataset.
        """
        if self._array_offsets is None:
            array_offsets = []
            item_size = self.indices_dtype(0).itemsize
            start_offset = 0
            for path in self.paths:
                doc_indices_path = self._get_document_indices_path(path)
                instances_in_file = (get_file_size(doc_indices_path) // item_size) // 2
                end_offset = start_offset + instances_in_file
                array_offsets.append((start_offset, end_offset))
                start_offset += instances_in_file
            self._array_offsets = tuple(array_offsets)
        return self._array_offsets

    def prepare(self):
        if self.fs_local_rank == 0:
            log.info("Gathering dataset document indices and buckets...")
            self._write_document_indices()
            self._write_instance_lengths()
            self._write_instance_buckets(self.get_instance_lengths())
        barrier()
        len(self)

    def __len__(self):
        if self._num_instances is None:
            self._num_instances = self.offsets[-1][1]
        return self._num_instances

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = int(index)  # in case this is a numpy int type.
        pos_index = index if index >= 0 else len(self) + index

        # The index of the array within 'self.paths'.
        array_index: Optional[int] = None
        # The index within the corresponding array.
        array_local_index: Optional[int] = None
        for i, (offset_start, offset_end) in enumerate(self.offsets):
            if offset_start <= pos_index < offset_end:
                array_index = i
                array_local_index = pos_index - offset_start
                break

        if array_index is None or array_local_index is None:
            raise IndexError(f"{index} is out of bounds for dataset of size {len(self)}")

        # Read the data from file.
        input_ids = self._read_chunk_from_array(self.paths[array_index], array_local_index)
        out: Dict[str, Any] = {"input_ids": input_ids}

        if self.instance_filter_config is not None:
            out["instance_mask"] = self._validate_instance(input_ids, self.instance_filter_config)

        if self._include_instance_metadata:
            metadata = self._metadata[array_index]
            out["metadata"] = deepcopy(metadata)

        return out

    def _read_chunk_from_array(self, path: PathOrStr, index: int) -> torch.Tensor:
        indices_path = self._get_document_indices_path(path)
        indices = load_array_slice_into_tensor(
            indices_path, index * 2, index * 2 + 2, self.indices_dtype
        )
        start_idx, end_idx = indices
        data = load_array_slice_into_tensor(path, int(start_idx), int(end_idx), self.dtype)
        return data

    def _get_document_indices_path(self, path: PathOrStr) -> Path:
        sha256_hash = hashlib.sha256()
        sha256_hash.update(str(path).encode())
        sha256_hash.update(str(self._get_file_size(path)).encode())
        for seq_len in self.all_sequence_lengths:
            sha256_hash.update(str(seq_len).encode())
        path_hash = sha256_hash.hexdigest()
        return self.work_dir / "dataset-common" / f"bucketed-doc-indices-{path_hash}.npy"

    def _get_instance_lengths_path(self) -> Path:
        return self.work_dir / f"dataset-{self.fingerprint}" / "instance-lengths.npy"

    def _get_instance_bucket_path(self, seq_len: int) -> Path:
        return self.work_dir / f"dataset-{self.fingerprint}" / f"bucket{seq_len}-indices.npy"

    def _write_document_indices(self):
        paths_needed: List[PathOrStr] = []
        for path in self.paths:
            indices_path = self._get_document_indices_path(path)
            if indices_path.is_file():
                log.info(f"Reusing document indices for '{path}' at:\n'{indices_path}'")
            elif path not in paths_needed:
                paths_needed.append(path)

        if paths_needed:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=_get_data_prep_max_workers()
            ) as executor:
                futures = []
                for path in paths_needed:
                    indices_path = self._get_document_indices_path(path)
                    log.info(f"Gathering document indices for '{path}'...")
                    future = executor.submit(
                        run_worker_func,
                        bucket_documents,
                        path,
                        indices_path,
                        buckets=self.all_sequence_lengths,
                        eos_token_id=self.eos_token_id,
                        dtype=self.dtype,
                        indices_dtype=self.indices_dtype,
                    )
                    futures.append(future)

                concurrent.futures.wait(futures, return_when="FIRST_EXCEPTION")

                # Log results.
                for path, future in zip(paths_needed, futures):
                    total_og_docs, total_bucketed_docs = future.result()
                    log.info(
                        f"Created {total_bucketed_docs:,d} bucketed documents by sequence length from "
                        f"{total_og_docs:,d} original documents in '{path}'"
                    )

    def _write_instance_lengths(self):
        instance_lengths_path = self._get_instance_lengths_path()
        if instance_lengths_path.is_file():
            log.info(f"Reusing all instance lengths at:\n'{instance_lengths_path}'")
        else:
            log.info(f"Gathering all instance lengths to:\n'{instance_lengths_path}...")
            with memmap_to_write(
                instance_lengths_path, dtype=self.lengths_dtype, shape=(len(self),)
            ) as instance_lengths:
                for path, (offset_start, offset_end) in zip(self.paths, self.offsets):
                    indices_path = self._get_document_indices_path(path)
                    indices_mmap = np.memmap(indices_path, dtype=self.indices_dtype, mode="r")
                    instance_lengths[offset_start:offset_end] = get_doc_lengths_from_indices(
                        indices_mmap
                    )
                    del indices_mmap

    def _write_instance_buckets(self, instance_lengths: np.ndarray):
        for seq_len in self.all_sequence_lengths:
            bucket_path = self._get_instance_bucket_path(seq_len)
            if bucket_path.is_file():
                log.info(
                    f"Reusing instance indices for seq len {seq_len} bucket at:\n'{bucket_path}'"
                )
            else:
                log.info(f"Gathering instance indices for seq len {seq_len} bucket...")
                bucket_path.parent.mkdir(exist_ok=True, parents=True)
                instance_indices = (instance_lengths == seq_len).nonzero()[0]
                with memmap_to_write(
                    bucket_path,
                    dtype=self.indices_dtype,
                    shape=instance_indices.shape,
                ) as bucket:
                    bucket[:] = instance_indices
                log.info(
                    f"Instance indices for seq len {seq_len} bucket written to:\n'{bucket_path}'"
                )

    def get_instance_lengths(self) -> np.ndarray:
        """
        Get a numpy memory-mapped array with the length of every instance in the dataset.
        """
        return np.memmap(self._get_instance_lengths_path(), dtype=self.lengths_dtype, mode="r")

    def get_instance_bucket(self, seq_len: int) -> np.ndarray:
        """
        Get the instance indices in a bucket.
        """
        return np.memmap(
            self._get_instance_bucket_path(seq_len), dtype=self.indices_dtype, mode="r"
        )

    def get_instance_buckets(self) -> List[Tuple[int, np.ndarray]]:
        """
        Get the buckets of instance indices that all have the same length.
        The buckets will be sorted from smallest sequence length to longest.
        """
        buckets = []
        for seq_len in self.all_sequence_lengths:
            buckets.append((seq_len, self.get_instance_bucket(seq_len)))
        return buckets

    @property
    def instances_per_bucket(self) -> Tuple[Tuple[int, int], ...]:
        """
        The number of instances in each bucket.
        """
        if self._instances_per_bucket is None:
            instances_per_bucket = []
            item_size = self.indices_dtype(0).itemsize
            for seq_len in self.all_sequence_lengths:
                instances_per_bucket.append(
                    (seq_len, get_file_size(self._get_instance_bucket_path(seq_len)) // item_size)
                )
            self._instances_per_bucket = tuple(instances_per_bucket)
        return self._instances_per_bucket

    @property
    def indices_dtype(self) -> NumpyUIntTypes:
        return np.uint32

    @property
    def lengths_dtype(self) -> NumpyUIntTypes:
        if self._lengths_dtype is None:
            for dtype in (
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ):
                if (self.max_sequence_length - 1) <= np.iinfo(dtype).max:
                    self._lengths_dtype = dtype
                    break
            assert self._lengths_dtype is not None
        return self._lengths_dtype


class VSLCurriculumType(StrEnum):
    """
    An enumeration of the different VSL curriculum implementations.
    """

    natural = "natural"
    """
    The natural curriculum ➡️ :class:`VSLNaturalCurriculum`.
    """

    grow_p2 = "grow_p2"
    """
    The "Grow-P2" curriculum ➡️ :class:`VSLGrowP2Curriculum`.
    """

    grow_linear = "grow_linear"
    """
    The "Grow-Linear" curriculum ➡️ :class:`VSLGrowLinearCurriculum`.
    """


@dataclass
class VSLCurriculumConfig(Config):
    name: VSLCurriculumType = VSLCurriculumType.natural
    num_cycles: Optional[int] = None
    balanced: Optional[bool] = None

    def validate(self):
        if self.name == VSLCurriculumType.natural:
            self.num_cycles = None
            self.balanced = None

    def build(self) -> VSLCurriculum:
        """
        Build the VSL curriculum.
        """
        if self.name == VSLCurriculumType.natural:
            if self.num_cycles is not None:
                raise OLMoConfigurationError(
                    f"'num_cycles' is not a valid field for the {self.name} curriculum"
                )
            if self.balanced is not None:
                raise OLMoConfigurationError(
                    f"'balanced' is not a valid field for the {self.name} curriculum"
                )
            return VSLNaturalCurriculum()

        if self.name in (VSLCurriculumType.grow_p2, VSLCurriculumType.grow_linear):
            if self.num_cycles is None:
                raise OLMoConfigurationError(
                    f"'num_cycles' is required for the {self.name} curriculum"
                )
            if self.balanced is None:
                raise OLMoConfigurationError(
                    f"'balanced' is required for the {self.name} curriculum"
                )

            if self.name == VSLCurriculumType.grow_p2:
                return VSLGrowP2Curriculum(num_cycles=self.num_cycles, balanced=self.balanced)
            else:  # grow_linear
                return VSLGrowLinearCurriculum(num_cycles=self.num_cycles, balanced=self.balanced)

        raise NotImplementedError(self.name)


NumpyDatasetConfigT = TypeVar("NumpyDatasetConfigT", bound="NumpyDatasetConfig")


@dataclass(kw_only=True)
class NumpyDatasetConfig(Config, ABC):
    """
    Abstract base configuration class for numpy-based datasets.

    This abstract base class provides common configuration options and utilities
    for creating :class:`NumpyDatasetBase` datasets.
    """

    tokenizer: TokenizerConfig
    """
    The tokenizer config.
    """
    paths: Optional[List[str]] = None
    """
    The paths/URLs to the numpy token ID arrays.
    """
    mix: Optional[Union[str, DataMixBase]] = None
    """
    The name of a data mix (e.g. ``"dolma17"``).
    """
    mix_base_dir: Optional[str] = None
    """
    The base directory of the data mix.
    """
    expand_glob: bool = False
    """
    If True, treat the :data:`paths` as globs.
    """
    dtype: Optional[NumpyDatasetDType] = None
    """
    The numpy datatype of the token ID arrays.
    """
    metadata: Optional[List[Dict[str, Any]]] = None
    """
    Metadata for the numpy arrays.
    """
    include_instance_metadata: bool = True
    """
    Whether or not to include the :data:`metadata` in the instances returned from
    :meth:`NumpyDatasetBase.__getitem__()`.
    """
    instance_filter_config: Optional[InstanceFilterConfig] = None
    """
    The instance filter config (aka the "ngram filter") that will be applied to the dataset. This
    can be used to filter out instances with too many repeated token ngrams.
    """
    source_permutation_seed: Optional[int] = None
    """
    Used to shuffle the source files before handing off to the dataset class.
    """
    work_dir: Optional[str] = None
    """
    The dataset working directory. This is used to cache working files like shuffled indices,
    instance buckets, etc.

    .. tip::
        You can save a lot of time and disk space by setting this to a common directory across
        all of you runs.
    """
    ignore_fingerprint_mismatch: bool = False
    """
    If True, ignore dataset fingerprint mismatches when loading from a checkpoint.
    This is used when intentionally switching to a different dataset mix.
    """

    @abstractmethod
    def build(self) -> NumpyDatasetBase:
        """
        Build and return a NumpyDatasetBase instance from this configuration.

        :returns: The constructed dataset instance.
        """
        raise NotImplementedError

    def get_dtype(self) -> NumpyUIntTypes:
        if self.dtype is not None:
            return NumpyDatasetDType(self.dtype).as_np_dtype()

        for dtype in (
            NumpyDatasetDType.uint8,
            NumpyDatasetDType.uint16,
            NumpyDatasetDType.uint32,
            NumpyDatasetDType.uint64,
        ):
            if (self.tokenizer.vocab_size - 1) <= np.iinfo(dtype.as_np_dtype()).max:
                log.info(f"Assuming dtype '{dtype}' based on vocab size")
                return dtype.as_np_dtype()

        raise ValueError("vocab size too big!")

    def _expand_globs(self, patterns: Sequence[str]) -> List[str]:
        expanded: List[str] = []
        for pattern in patterns:
            log.info(f"Expanding '{pattern}'...")
            matches = deterministic_glob_directory(pattern)
            if not matches:
                error_msg = f"Pattern '{pattern}' did not match any files"
                # Add helpful hint for mix-0625 which has unavailable files
                if "0625" in pattern:
                    error_msg += (
                        "\n\nNOTE: Some files in OLMo-mix-0625 are not available. "
                        "If you are resuming training from a checkpoint that used mix-0625, you will need to "
                        "switch to a newer mix such as OLMo-mix-0925. To continue training with a different "
                        "dataset mix, set 'ignore_fingerprint_mismatch=True' in your NumpyDataLoaderConfig "
                        "to bypass the fingerprint mismatch error. This will probably result in a different data order!"
                    )
                raise FileNotFoundError(error_msg)
            for match in matches:
                log.info(f" - '{match}'")
            expanded.extend(matches)
        return expanded

    def _resolve_paths_metadata(
        self,
        *,
        allow_mix: bool,
        label_mask_paths: Optional[Sequence[PathOrStr]] = None,
    ) -> Tuple[List[str], Optional[List[Dict[str, Any]]], Optional[List[PathOrStr]]]:
        if self.paths is not None and self.mix is not None:
            raise OLMoConfigurationError("Only one of 'paths' or 'mix' can be set")

        metadata: Optional[List[Dict[str, Any]]] = self.metadata
        resolved_label_masks: Optional[List[PathOrStr]] = None

        if self.paths is not None:
            raw_paths = [str(path) for path in self.paths]
            if self.expand_glob:
                paths = self._expand_globs(raw_paths)
                if label_mask_paths is not None:
                    mask_patterns = [str(path) for path in label_mask_paths]
                    expanded_masks = self._expand_globs(mask_patterns)
                    resolved_label_masks = [cast(PathOrStr, mask) for mask in expanded_masks]
            else:
                paths = raw_paths
                if label_mask_paths is not None:
                    resolved_label_masks = [cast(PathOrStr, path) for path in label_mask_paths]
        else:
            if self.mix is None:
                raise OLMoConfigurationError("Either 'paths' or 'mix' must be set")
            if not allow_mix:
                raise OLMoConfigurationError("'mix' is not supported for this dataset type")
            if self.mix_base_dir is None:
                raise OLMoConfigurationError(
                    "'mix_base_dir' is required to build a dataset from a mix"
                )
            if self.tokenizer.identifier is None:
                raise OLMoConfigurationError(
                    "Missing tokenizer identifier required to construct data mix"
                )
            mix = self.mix
            if not isinstance(mix, DataMixBase):
                mix = DataMix(mix)
            paths, labels = mix.build(self.mix_base_dir, self.tokenizer.identifier)
            paths = [str(path) for path in paths]
            if metadata is None:
                metadata = [{"label": label} for label in labels]
            if label_mask_paths is not None:
                resolved_label_masks = [cast(PathOrStr, path) for path in label_mask_paths]

        if self.source_permutation_seed is not None:
            order = list(range(len(paths)))
            rng = random.Random(self.source_permutation_seed)
            rng.shuffle(order)
            paths = [paths[i] for i in order]
            if metadata is not None:
                metadata = [metadata[i] for i in order]
            if resolved_label_masks is not None:
                resolved_label_masks = [resolved_label_masks[i] for i in order]

        return paths, metadata, resolved_label_masks

    def _finalize(self, dataset: NumpyDatasetBase) -> NumpyDatasetBase:
        if self.work_dir is not None:
            dataset.work_dir = Path(self.work_dir)
        return dataset

    @classmethod
    def glob(
        cls: Type[NumpyDatasetConfigT], *glob_paths: str, **kwargs: Any
    ) -> NumpyDatasetConfigT:
        """
        Initialize a dataset config with glob paths.

        .. note::
            Globs are not expanded until :meth:`build()` is called.
            If any of the globs don't expand to any matches a :class:`FileNotFoundError`
            error is raised

        :param glob_paths: The glob patterns.
        :returns: A new dataset config.
        """
        return cls(paths=list(glob_paths), mix=None, mix_base_dir=None, expand_glob=True, **kwargs)

    @classmethod
    def from_data_mix(
        cls: Type[NumpyDatasetConfigT],
        mix: Union[str, DataMixBase],
        *,
        tokenizer: TokenizerConfig,
        **kwargs: Any,
    ) -> NumpyDatasetConfigT:
        """
        Initialize a dataset config from an official data mix.

        :param mix: The data mix.
        :param tokenizer: The tokenizer config.
        :returns: A new dataset config.
        """
        if tokenizer.identifier is None:
            raise OLMoConfigurationError(
                "Missing tokenizer identifier required to construct data mix"
            )
        return cls(mix=mix, paths=None, tokenizer=tokenizer, **kwargs)


@dataclass
class NumpyFSLDatasetConfig(NumpyDatasetConfig):
    sequence_length: int
    """
    The length of a single instance. Generally this should correspond to your model's maximum input length.
    """
    max_target_sequence_length: Optional[int] = None
    """
    Optional upper bound used when precomputing cached offsets.

    If you're planning a sequence-length warm-up, set this to the final chunk size so future
    datasets with larger ``sequence_length`` values can reuse the exact same document ordering.
    The current dataset still returns ``sequence_length``-token windows; this hint simply keeps
    token boundaries and cache files deterministic across warm-up stages. Leave ``None`` if you
    won't rebuild at a larger length.
    """
    generate_doc_lengths: bool = False
    """
    Include individual document lengths in the instances returned from
    :meth:`NumpyDatasetBase.__getitem__()`.
    """
    label_mask_paths: Optional[List[str]] = None
    """
    The paths/URLs to numpy bool files indicating which tokens should be masked.
    """
    source_mixture_config: Optional[SourceMixtureDatasetConfig] = None
    """
    A source mixture dataset config. If set, the dataset will be built from a mixture of sources.
    """

    @classmethod
    def from_src_mix(
        cls, src_mix: SourceMixtureDatasetConfig, **kwargs: Any
    ) -> NumpyFSLDatasetConfig:
        """
        Initialize a dataset config from a custom fine-grained data mix.

        :param src_mix: The fine-grained SourceMixtureDatasetConfig.
        :returns: A new dataset config.
        """
        return cls(source_mixture_config=src_mix, paths=None, mix=None, mix_base_dir=None, **kwargs)

    def validate(self):
        if self.sequence_length <= 0:
            raise OLMoConfigurationError("'sequence_length' must be positive")
        if self.source_mixture_config is not None:
            if self.paths is not None or self.mix is not None:
                raise OLMoConfigurationError(
                    "Specify only one of 'paths', 'mix', or 'source_mixture_config'"
                )
            if self.label_mask_paths is not None:
                raise OLMoConfigurationError(
                    "'label_mask_paths' is not supported alongside 'source_mixture_config'"
                )

    def build(self) -> NumpyDatasetBase:
        self.validate()

        if self.source_mixture_config is not None:
            mixture = self.source_mixture_config.build(
                npdtype=self.get_dtype(), sequence_length=self.sequence_length
            )
            # Drop paths that were sampled to zero tokens so the mixture's path index stays tight:
            # they produce no instances downstream, and the prep loop would otherwise carry them as
            # no-ops. to_paths()/to_index() filter together, so their idx ordering stays aligned.
            mixture.filter_zero_token_paths = True
            dataset = NumpyFSLDatasetMixture(
                *mixture.to_paths(),
                seed=self.source_mixture_config.seed,
                path_offset_index=mixture.to_index(),
                sequence_length=self.sequence_length,
                max_target_sequence_length=self.max_target_sequence_length,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                vocab_size=self.tokenizer.vocab_size,
                dtype=self.get_dtype(),
                metadata=self.metadata,
                include_instance_metadata=self.include_instance_metadata,
                generate_doc_lengths=self.generate_doc_lengths,
                bos_token_id=self.tokenizer.bos_token_id,
                instance_filter_config=self.instance_filter_config,
            )
            return self._finalize(dataset)

        paths, metadata, label_masks = self._resolve_paths_metadata(
            allow_mix=True, label_mask_paths=self.label_mask_paths
        )

        dataset = NumpyFSLDataset(
            *paths,
            sequence_length=self.sequence_length,
            max_target_sequence_length=self.max_target_sequence_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            vocab_size=self.tokenizer.vocab_size,
            dtype=self.get_dtype(),
            metadata=metadata,
            include_instance_metadata=self.include_instance_metadata,
            generate_doc_lengths=self.generate_doc_lengths,
            bos_token_id=self.tokenizer.bos_token_id,
            instance_filter_config=self.instance_filter_config,
            label_mask_paths=label_masks,
        )
        return self._finalize(dataset)


@dataclass(kw_only=True)
class NumpyPaddedFSLDatasetConfig(NumpyDatasetConfig):
    sequence_length: int
    """
    The length of a single instance. Generally this should correspond to your model's maximum input length.
    """
    label_mask_paths: Optional[List[str]] = None
    """
    The paths/URLs to numpy bool files indicating which tokens should be masked.
    """

    def validate(self):
        if self.sequence_length <= 0:
            raise OLMoConfigurationError("'sequence_length' must be positive")

    def build(self) -> NumpyDatasetBase:
        self.validate()
        paths, metadata, label_masks = self._resolve_paths_metadata(
            allow_mix=True, label_mask_paths=self.label_mask_paths
        )
        dataset = NumpyPaddedFSLDataset(
            *paths,
            sequence_length=self.sequence_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            vocab_size=self.tokenizer.vocab_size,
            dtype=self.get_dtype(),
            bos_token_id=self.tokenizer.bos_token_id,
            metadata=metadata,
            include_instance_metadata=self.include_instance_metadata,
            instance_filter_config=self.instance_filter_config,
            label_mask_paths=label_masks,
        )
        return self._finalize(dataset)


@dataclass(kw_only=True)
class NumpyPackedFSLDatasetConfig(NumpyDatasetConfig):
    sequence_length: int
    """
    The length of a single instance. Generally this should correspond to your model's maximum input length.
    """
    generate_doc_lengths: bool = False
    """
    Include individual document lengths in the instances returned from
    :meth:`NumpyDatasetBase.__getitem__()`.
    """
    label_mask_paths: Optional[List[str]] = None
    """
    The paths/URLs to numpy bool files indicating which tokens should be masked.
    """
    long_doc_strategy: LongDocStrategy = LongDocStrategy.truncate
    """
    The strategy to use for handling long documents.
    """
    source_group_size: int = 1
    """
    The number of source npy files to process together when packing.
    """
    use_array_if_local: Optional[bool] = None
    """
    Where to get document boundaries from. Leave as ``None`` to infer them by scanning the token
    array for the EOS token (the historical default). Set to ``False`` to read them from the source
    metadata file instead, which is authoritative and required for correctness when the producer may
    emit documents that are not EOS-terminated -- see the warning on
    :func:`~olmo_core.data.utils.iter_document_indices`.
    """

    def validate(self):
        if self.sequence_length <= 0:
            raise OLMoConfigurationError("'sequence_length' must be positive")
        if self.source_group_size < 1:
            raise OLMoConfigurationError("'source_group_size' must be at least 1")

    def build(self) -> NumpyDatasetBase:
        self.validate()

        paths, metadata, label_masks = self._resolve_paths_metadata(
            allow_mix=True, label_mask_paths=self.label_mask_paths
        )

        dataset = NumpyPackedFSLDataset(
            *paths,
            sequence_length=self.sequence_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            vocab_size=self.tokenizer.vocab_size,
            dtype=self.get_dtype(),
            metadata=metadata,
            include_instance_metadata=self.include_instance_metadata,
            generate_doc_lengths=self.generate_doc_lengths,
            bos_token_id=self.tokenizer.bos_token_id,
            instance_filter_config=self.instance_filter_config,
            long_doc_strategy=self.long_doc_strategy,
            label_mask_paths=label_masks,
            source_group_size=self.source_group_size,
            use_array_if_local=self.use_array_if_local,
        )
        return self._finalize(dataset)


@dataclass(kw_only=True)
class NumpyInterleavedFSLDatasetConfig(NumpyDatasetConfig):
    sequence_length: int
    """
    The length of a single instance. Generally this should correspond to your model's maximum input length.
    """
    docs_per_instance: int
    """
    The number of documents to include in each instance.
    """
    chunks_per_doc: int
    """
    The number of chunks to include in each document.
    """
    seed: int
    """
    The seed to use for the random number generator.
    """
    label_mask_paths: Optional[List[str]] = None
    """
    The paths/URLs to numpy bool files indicating which tokens should be masked.
    """
    interleaving_exempt_paths: Optional[List[str]] = None
    """
    The paths/URLs to numpy bool files indicating which tokens should be exempt from interleaving.
    """

    def validate(self):
        if self.sequence_length <= 0:
            raise OLMoConfigurationError("'sequence_length' must be positive")
        if self.docs_per_instance <= 0:
            raise OLMoConfigurationError("'docs_per_instance' must be positive")
        if self.chunks_per_doc <= 0:
            raise OLMoConfigurationError("'chunks_per_doc' must be positive")

    def build(self) -> NumpyDatasetBase:
        self.validate()

        paths, metadata, label_masks = self._resolve_paths_metadata(
            allow_mix=True, label_mask_paths=self.label_mask_paths
        )

        interleaving_exempt_paths = cast(Optional[List[PathOrStr]], self.interleaving_exempt_paths)

        dataset = NumpyInterleavedFSLDataset(
            *paths,
            sequence_length=self.sequence_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            vocab_size=self.tokenizer.vocab_size,
            seed=self.seed,
            docs_per_instance=self.docs_per_instance,
            chunks_per_doc=self.chunks_per_doc,
            dtype=self.get_dtype(),
            metadata=metadata,
            include_instance_metadata=self.include_instance_metadata,
            instance_filter_config=self.instance_filter_config,
            label_mask_paths=label_masks,
            bos_token_id=self.tokenizer.bos_token_id,
            interleaving_exempt_paths=interleaving_exempt_paths,
        )
        return self._finalize(dataset)


@dataclass(kw_only=True)
class NumpyVSLDatasetConfig(NumpyDatasetConfig):
    max_sequence_length: int
    """
    The maximum sequence length. Generally this should correspond to your model's maximum input length.
    """
    min_sequence_length: int
    """
    The minimum sequence length.
    """
    vsl_curriculum: Optional[VSLCurriculumConfig] = None
    """
    The VSL curriculum config.
    """

    def validate(self):
        if self.max_sequence_length <= 0:
            raise OLMoConfigurationError("'max_sequence_length' must be positive")
        if self.min_sequence_length <= 0:
            raise OLMoConfigurationError("'min_sequence_length' must be positive")
        if self.min_sequence_length > self.max_sequence_length:
            raise OLMoConfigurationError(
                "'min_sequence_length' cannot exceed 'max_sequence_length'"
            )
        if self.tokenizer.bos_token_id is not None:
            raise OLMoConfigurationError("'bos_token_id' is not supported for the VSL dataset")
        if self.vsl_curriculum is not None:
            self.vsl_curriculum.validate()

    def build(self) -> NumpyDatasetBase:
        self.validate()

        paths, metadata, _ = self._resolve_paths_metadata(allow_mix=True)

        dataset = NumpyVSLDataset(
            *paths,
            max_sequence_length=self.max_sequence_length,
            min_sequence_length=self.min_sequence_length,
            curriculum=None if self.vsl_curriculum is None else self.vsl_curriculum.build(),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            vocab_size=self.tokenizer.vocab_size,
            dtype=self.get_dtype(),
            metadata=metadata,
            include_instance_metadata=self.include_instance_metadata,
            instance_filter_config=self.instance_filter_config,
        )
        return self._finalize(dataset)
