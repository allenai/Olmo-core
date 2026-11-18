import functools as ft
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

from olmo_core.aliases import PathOrStr

from ..tokenizer import TokenizerConfig
from ..types import NumpyDatasetDType
from .instance_source import Instance, InstanceSource, InstanceSourceConfig
from .numpy_document_source import NumpyDocumentSource
from .token_source import TokenSource, TokenSourceConfig
from .utils import resolve_seed


@dataclass
class ConcatAndChunkInstanceSourceConfig(InstanceSourceConfig):
    """
    Config for :class:`ConcatAndChunkInstanceSource`.
    """

    sources: List[TokenSourceConfig]
    sequence_length: int
    max_sequence_length: Optional[int] = None
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
    ) -> "ConcatAndChunkInstanceSourceConfig":
        """
        Create a :class:`ConcatAndChunkInstanceSourceConfig` from one or more tokenized ``.npy`` source files.
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
            sequence_length=sequence_length,
            max_sequence_length=max_sequence_length,
            label=label,
        )

    def build(self, work_dir: PathOrStr) -> "ConcatAndChunkInstanceSource":
        return ConcatAndChunkInstanceSource(
            *[source for source_config in self.sources for source in source_config.build(work_dir)],
            sequence_length=self.sequence_length,
            max_sequence_length=self.max_sequence_length,
            work_dir=work_dir,
            label=self.label,
        )


class ConcatAndChunkInstanceSource(InstanceSource):
    """
    The basic instance source that simply chunks up token sources without regard for
    document boundaries, just like the :class:`~olmo_core.data.numpy_dataset.NumpyFSLDataset`.
    """

    Config = ConcatAndChunkInstanceSourceConfig
    DISPLAY_ICON = "\uf51e"

    def __init__(
        self,
        *sources: TokenSource,
        sequence_length: int,
        work_dir: PathOrStr,
        max_sequence_length: Optional[int] = None,
        label: Optional[str] = None,
    ):
        super().__init__(
            sequence_length=sequence_length,
            max_sequence_length=max_sequence_length,
            work_dir=work_dir,
            label=label,
        )
        self._sources = sources

    @property
    def sources(self) -> Tuple[TokenSource, ...]:
        return self._sources

    @ft.cached_property
    def num_instances(self) -> int:
        return sum(
            (
                (source.num_tokens // self.max_sequence_length)
                * (self.max_sequence_length // self.sequence_length)
                for source in self.sources
            )
        )

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        # NOTE: counterintuitively, we don't take 'self.sequence_length' into account when computing
        # the fingerprint because only 'self.max_sequence_length' will affect data order, and this source
        # allows for changing the sequence length in the middle of an epoch.
        sha256_hash.update(
            (f"class={self.__class__.__name__},{self.max_sequence_length=},").encode()
        )
        for source in self.sources:
            sha256_hash.update(f"source={source.fingerprint},".encode())
        return sha256_hash.hexdigest()

    def __len__(self) -> int:
        return self.num_instances

    def __getitem__(self, idx: int) -> Instance:
        idx = self.validate_index(idx)
        source_start_offset = 0
        for source in self.sources:
            source_end_offset = source_start_offset + (
                source.num_tokens // self.max_sequence_length
            ) * (self.max_sequence_length // self.sequence_length)

            if source_start_offset <= idx < source_end_offset:
                start_idx = (idx - source_start_offset) * self.sequence_length
                instance = source.get_token_range(start_idx, start_idx + self.sequence_length)
                return {**instance, "loss_valid_mask": [True] * self.sequence_length}

            source_start_offset = source_end_offset
        raise IndexError(f"Index {idx} out of range for {self.num_instances} instances.")

    def children(self):
        return self.sources
