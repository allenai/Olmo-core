import functools as ft
import hashlib
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator, List, Optional, Sequence, Tuple, TypedDict

from typing_extensions import NotRequired

import olmo_core.io as io
from olmo_core.aliases import PathOrStr
from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError

from .source_abc import SourceABC
from .utils import SEED_NOT_SET, resolve_seed

if TYPE_CHECKING:
    from .sampling_instance_source import (
        SamplingInstanceSource,
        SamplingInstanceSourceConfig,
    )
    from .sliced_instance_source import SlicedInstanceSource


class Instance(TypedDict):
    """
    An instance is just a dictionary that should include ``input_ids`` and optionally a
    corresponding ``label_mask``.
    """

    input_ids: Sequence[int]
    """The token IDs for this instance."""
    label_mask: NotRequired[Sequence[bool]]
    """An optional mask indicating which tokens should contribute to the SFT loss."""
    loss_valid_mask: NotRequired[Sequence[bool]]
    """Real prediction targets, including prompts but excluding structural/padding positions."""
    doc_lens: NotRequired[Sequence[int]]
    """Optional explicit per-document lengths within this instance (they must sum to the instance
    length). When present, the data loader uses these for intra-document masking instead of deriving
    document boundaries from EOS tokens. Used by sources that pack pre-segmented documents (e.g.
    :class:`~olmo_core.data.composable.landmark_packing_instance_source.LandmarkPackingInstanceSource`)."""


class InstanceSource(SourceABC):
    """
    An abstract base class for a source of instances, usually consumed by a :class:`ComposableDataLoader`.
    It essentially represents an array of instances, where each instance is a sequence of
    ``sequence_length`` tokens.

    :param sequence_length: The length of each sequence (instance) to produce.
    :param max_sequence_length: For sources that support this. If you intend to increase the sequence
      length in the middle of an epoch, you should set this to the maximum sequence length that you'll
      train on to guarantee that you can restart the run with the same data order after changing sequence length.
      Care needs to be taken when implementing this in a subclass to ensure that the exact same tokens
      will be produced when `sequence_length` is changed but `max_sequence_length` is fixed.
    """

    def __init__(
        self,
        *,
        work_dir: PathOrStr,
        sequence_length: int,
        max_sequence_length: Optional[int] = None,
        label: Optional[str] = None,
    ):
        super().__init__(work_dir=work_dir, label=label)
        if io.is_url(work_dir):
            raise OLMoConfigurationError(
                f"'work_dir' should be a local path, not a URL ('{work_dir}')."
            )
        assert sequence_length > 0
        if max_sequence_length is not None:
            assert max_sequence_length > 0
            if sequence_length > max_sequence_length:
                raise OLMoConfigurationError(
                    "'sequence_length' cannot be greater than 'max_sequence_length'."
                )
            if max_sequence_length % sequence_length != 0:
                raise OLMoConfigurationError(
                    "'max_sequence_length' must be a multiple of 'sequence_length'."
                )
        self._sequence_length = sequence_length
        self._max_sequence_length = max_sequence_length or sequence_length

    @property
    def sequence_length(self) -> int:
        """The sequence length of each instance that this source will produce."""
        return self._sequence_length

    @property
    def max_sequence_length(self) -> int:
        """
        Typically the same as ``sequence_length`` though in some cases it can be greater, such
        as when the sequence length will be increased in the middle of an epoch.
        """
        return self._max_sequence_length

    @property
    def num_tokens(self) -> int:
        return len(self) * self.sequence_length

    @abstractmethod
    def __len__(self) -> int:
        """The number of instances available from this source."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Instance:
        """Get an instance by index."""
        raise NotImplementedError

    def __iter__(self) -> Generator[Instance, None, None]:
        """Iterate over all instances in the source."""
        for i in range(len(self)):
            yield self[i]

    def validate_index(self, idx: int) -> int:
        idx = int(idx)
        if idx < 0:
            idx = len(self) + idx
        if not (0 <= idx < len(self)):
            raise IndexError(
                f"Index {idx} is out of bounds for source {self} with {len(self):,d} instances."
            )
        return idx

    def __add__(self, other: "InstanceSource") -> "ConcatenatedInstanceSource":
        """Add two instance sources together into a :class:`ConcatenatedInstanceSource`."""
        if isinstance(other, InstanceSource):
            return ConcatenatedInstanceSource(self, other, work_dir=self.common_work_dir)
        else:
            raise TypeError(f"Cannot add {type(self)} with {type(other)}.")

    def __mul__(self, factor: float) -> "SamplingInstanceSource":
        """Re-size this source by a given factor by sampling instances from it."""
        if isinstance(factor, (float, int)):
            return self.resize(factor)
        else:
            raise TypeError(f"Cannot multiply {type(self)} with {type(factor)}.")

    def sample(
        self,
        *,
        max_tokens: Optional[int] = None,
        max_instances: Optional[int] = None,
        seed: Optional[int] = SEED_NOT_SET,
    ) -> "SamplingInstanceSource":
        """
        Sample instances from this source.

        .. seealso::
            - :meth:`resize()`
            - :meth:`split()`

        :param max_tokens: The maximum number of tokens to sample from this source.
          Mutually exclusive with ``max_instances``.
        :param max_instances: The maximum number of instances to sample from this source.
          Mutually exclusive with ``max_tokens``.
        :param seed: A random seed for sampling. If ``None``, no shuffling is done and instances
          are taken in order.
        """
        from .sampling_instance_source import SamplingInstanceSource

        return SamplingInstanceSource(
            self,
            max_tokens=max_tokens,
            max_instances=max_instances,
            seed=resolve_seed(seed),
            work_dir=self.common_work_dir,
        )

    def resize(self, factor: float, seed: Optional[int] = SEED_NOT_SET) -> "SamplingInstanceSource":
        """
        Re-size this source by a given factor by sampling instances from it.

        .. seealso::
            - :meth:`sample()`
            - :meth:`split()`

        :param factor: The factor by which to resize this source.
        :param seed: A random seed for sampling.
        """
        assert factor > 0
        return self.sample(max_tokens=int(self.num_tokens * factor), seed=seed)

    def split(
        self, ratio: float, seed: Optional[int] = None
    ) -> Tuple["SlicedInstanceSource", "SlicedInstanceSource"]:
        """
        Split this source into two disjoint sources according to the given ratio.

        :param ratio: The ratio of the first split to original source. E.g., ``0.8`` means
          the first split will have 80% of the instances and the second split will have 20%.
        :param seed: A seed to use to randomize the split.
        """
        from .sliced_instance_source import SlicedInstanceSource

        assert 0 < ratio < 1
        split_idx = int(
            ((ratio * self.num_tokens) // self.max_sequence_length)
            * (self.max_sequence_length // self.sequence_length)
        )

        return (
            SlicedInstanceSource(
                self, slice(0, split_idx), seed=seed, work_dir=self.common_work_dir
            ),
            SlicedInstanceSource(
                self, slice(split_idx, None), seed=seed, work_dir=self.common_work_dir
            ),
        )

    def random_split(
        self, ratio: float, seed: int = SEED_NOT_SET
    ) -> Tuple["SlicedInstanceSource", "SlicedInstanceSource"]:
        """
        Like :meth:`split()` but always a random split.
        """
        return self.split(ratio, seed=seed)

    def visualize(self, icons: bool = True):
        """
        Print a visualization of this source and its children, recursively.

        :param icons: Whether to use icons in the visualization.

           .. important::
               Some icons used in the visualization require a Nerd Font to render properly.
        """
        from .visualize import visualize_source

        visualize_source(self, icons=icons)


@dataclass
class InstanceSourceConfig(Config):
    """A base config class for configuring and building an :class:`InstanceSource`."""

    @abstractmethod
    def build(self, work_dir: PathOrStr) -> InstanceSource:
        """Build the :class:`InstanceSource`."""
        raise NotImplementedError

    def __add__(self, other: "InstanceSourceConfig") -> "ConcatenatedInstanceSourceConfig":
        """Add two instance source configs together into a :class:`ConcatenatedInstanceSourceConfig`."""
        if isinstance(other, InstanceSourceConfig):
            return ConcatenatedInstanceSourceConfig(sources=[self, other])
        else:
            raise TypeError(f"Cannot add {type(self)} with {type(other)}.")

    def __mul__(self, factor: float) -> "SamplingInstanceSourceConfig":
        """Re-size this source by a given factor by sampling instances from it."""
        if isinstance(factor, (float, int)):
            return self.resize(factor)
        else:
            raise TypeError(f"Cannot multiply {type(self)} with {type(factor)}.")

    def sample(
        self,
        *,
        max_tokens: Optional[int] = None,
        max_instances: Optional[int] = None,
        seed: Optional[int] = SEED_NOT_SET,
    ) -> "SamplingInstanceSourceConfig":
        """
        Sample instances from this source.

        :param max_tokens: The maximum number of tokens to sample from this source.
          Mutually exclusive with ``max_instances``.
        :param max_instances: The maximum number of instances to sample from this source.
          Mutually exclusive with ``max_tokens``.
        :param seed: A random seed for sampling. If ``None``, no shuffling is done and instances
          are taken in order.
        """
        from .sampling_instance_source import SamplingInstanceSourceConfig

        return SamplingInstanceSourceConfig(
            sources=[self],
            max_tokens=max_tokens,
            max_instances=max_instances,
            seed=resolve_seed(seed),
        )

    def resize(
        self, factor: float, seed: Optional[int] = SEED_NOT_SET
    ) -> "SamplingInstanceSourceConfig":
        """
        Re-size this source by a given factor by sampling instances from it.

        :param factor: The factor by which to resize this source.
        :param seed: A random seed for sampling.
        """
        from .sampling_instance_source import SamplingInstanceSourceConfig

        assert factor > 0
        return SamplingInstanceSourceConfig(
            sources=[self],
            factor=factor,
            seed=resolve_seed(seed),
        )

    def split(
        self, ratio: float, seed: Optional[int] = None
    ) -> Tuple["SplitInstanceSourceConfig", "SplitInstanceSourceConfig"]:
        """
        Split this source into two disjoint sources according to the given ratio.

        :param ratio: The ratio of the first split to original source. E.g., ``0.8`` means
          the first split will have 80% of the instances and the second split will have 20%.
        :param seed: A seed to use to randomize the split.
        """
        seed = resolve_seed(seed)
        return SplitInstanceSourceConfig(
            source=self,
            ratio=ratio,
            idx=0,
            seed=seed,
        ), SplitInstanceSourceConfig(source=self, ratio=ratio, idx=1, seed=seed)

    def random_split(
        self, ratio: float, seed: int = SEED_NOT_SET
    ) -> Tuple["SplitInstanceSourceConfig", "SplitInstanceSourceConfig"]:
        """
        Like :meth:`split()` but always a random split.
        """
        return self.split(ratio, seed=seed)


@dataclass
class SplitInstanceSourceConfig(InstanceSourceConfig):
    """A base config class for configuring and building a split :class:`InstanceSource`."""

    source: InstanceSourceConfig
    ratio: float
    idx: int
    seed: Optional[int] = None

    def __post_init__(self):
        assert 0 < self.ratio < 1
        assert self.idx in (0, 1)
        self.seed = resolve_seed(self.seed)

    def build(self, work_dir: PathOrStr) -> InstanceSource:
        from .sliced_instance_source import SlicedInstanceSource

        source = self.source.build(work_dir)
        split_idx = int(self.ratio * len(source))
        seed = resolve_seed(self.seed)
        if self.idx == 0:
            return SlicedInstanceSource(source, slice(0, split_idx), seed=seed, work_dir=work_dir)
        elif self.idx == 1:
            return SlicedInstanceSource(
                source, slice(split_idx, None), seed=seed, work_dir=work_dir
            )
        else:
            raise ValueError(f"Invalid split index: {self.idx}")


@dataclass
class ConcatenatedInstanceSourceConfig(InstanceSourceConfig):
    """A config for a :class:`ConcatenatedInstanceSource`."""

    sources: List[InstanceSourceConfig]

    def build(self, work_dir: PathOrStr) -> "ConcatenatedInstanceSource":
        return ConcatenatedInstanceSource(
            *[source.build(work_dir=work_dir) for source in self.sources],
            work_dir=work_dir,
        )


class ConcatenatedInstanceSource(InstanceSource):
    """
    An instance source that concatenates multiple instance sources together end-to-end.
    """

    Config = ConcatenatedInstanceSourceConfig

    DISPLAY_ICON = "\uf51e"

    def __init__(
        self,
        *sources: InstanceSource,
        work_dir: PathOrStr,
        label: Optional[str] = None,
    ):
        if len(sources) == 0:
            raise OLMoConfigurationError("At least one source must be provided.")

        sequence_length = sources[0].sequence_length
        max_sequence_length = sources[0].max_sequence_length
        for source in sources:
            if source.sequence_length != sequence_length:
                raise OLMoConfigurationError("All sources must have the same sequence length.")
            if source.max_sequence_length != max_sequence_length:
                raise OLMoConfigurationError("All sources must have the same max sequence length.")

        super().__init__(
            work_dir=work_dir,
            sequence_length=sequence_length,
            max_sequence_length=max_sequence_length,
            label=label,
        )

        unraveled_sources: List[InstanceSource] = []
        for source in sources:
            if isinstance(source, ConcatenatedInstanceSource):
                unraveled_sources.extend(source.sources)
            else:
                unraveled_sources.append(source)
        self._sources = tuple(unraveled_sources)

    @property
    def sources(self) -> Tuple[InstanceSource, ...]:
        return self._sources

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update((f"class={self.__class__.__name__},").encode())
        for source in self.sources:
            sha256_hash.update(f"source={source.fingerprint},".encode())
        return sha256_hash.hexdigest()

    def __len__(self) -> int:
        return sum(len(source) for source in self.sources)

    def __getitem__(self, idx: int) -> Instance:
        idx = self.validate_index(idx)
        source_start_offset = 0
        for source in self.sources:
            source_end_offset = source_start_offset + len(source)
            if source_start_offset <= idx < source_end_offset:
                return source[idx - source_start_offset]
            source_start_offset = source_end_offset
        raise IndexError(f"{idx} is out of bounds for source of size {len(self)}")

    def children(self):
        return self.sources
