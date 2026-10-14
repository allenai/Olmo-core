import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from itertools import chain
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich.text import Text

from olmo_core.aliases import PathOrStr
from olmo_core.config import Config
from olmo_core.data.types import NumpyUIntTypes
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import deterministic_glob_directory, file_exists, get_file_size

__all__ = [
    "SourceMixtureConfig",
    "SourceMixtureList",
    "SourceMixtureDatasetConfig",
]

log = logging.getLogger(__name__)


@dataclass
class SourceMixtureConfig(Config):
    """
    Configuration for a single data source within a mixture.

    This class defines how a data source should be sampled and weighted when
    creating a training dataset from multiple sources. It allows control over
    the target proportion, repetition limits, and maximum usage fraction of
    the source data.
    """

    source_name: str
    """
    The name of the source.
    """
    target_ratio: float
    """
    The target ratio of the source in the mixture.
    """
    paths: List[str]
    """
    A list of paths to the source data.
    """
    max_repetition_ratio: float = 1.0
    """
    The maximum ratio of repetitions of the source data to include in the mixture.
    This can be used to upsample the source data by setting the repetition ratio > 1.
    """
    max_source_fraction: float = 1.0
    """
    The maximum ratio of the source data to include in the mixture.
    """

    _resolved_paths: Optional[List[str]] = None

    def validate(self):
        if self.target_ratio:
            if not 0 < self.target_ratio <= 1:
                raise OLMoConfigurationError("target_ratio must be > 0 and <= 1")
            if not 0 < self.max_source_fraction <= 1:
                raise OLMoConfigurationError("max_source_fraction must > 0 and <= 1")

        if self.max_repetition_ratio < 1:
            raise OLMoConfigurationError("max_repetition_ratio must be >= 1")

        if not self.paths:
            raise OLMoConfigurationError("paths must not be empty")

        if not 0 <= self.max_source_fraction <= 1:
            raise OLMoConfigurationError("max_source_fraction must be in the range [0, 1]")

    @property
    def resolved_paths(self) -> List[str]:
        """
        Resolve the paths, expanding any globs and validating existence.
        Caches the result after the first access.
        """
        if self._resolved_paths is not None:
            return self._resolved_paths

        resolved: List[str] = []
        for path in self.paths:
            path_str = str(path)
            if "*" in path_str:
                matches = deterministic_glob_directory(path_str)
                if not matches:
                    error_msg = f"Glob pattern '{path_str}' did not match any files"
                    # Add helpful hint for mix-0625 which has unavailable files
                    if "0625" in path_str:
                        error_msg += (
                            "\n\nNOTE: Some files in OLMo-mix-0625 are not available. "
                            "If you are resuming training from a checkpoint that used mix-0625, you will need to "
                            "switch to a newer mix such as OLMo-mix-0925. To continue training with a different "
                            "dataset mix, set 'ignore_fingerprint_mismatch=True' in your NumpyDataLoaderConfig "
                            "to bypass the fingerprint mismatch error. This will probably result in a different data order!"
                        )
                    raise FileNotFoundError(error_msg)
                resolved.extend(matches)
            else:
                if not file_exists(path_str):
                    error_msg = f"Path '{path_str}' does not exist"
                    # Add helpful hint for mix-0625 which has unavailable files
                    if "0625" in path_str:
                        error_msg += (
                            "\n\nNOTE: Some files in OLMo-mix-0625 are not available. "
                            "If you are resuming training from a checkpoint that used mix-0625, you will need to "
                            "switch to a newer mix such as OLMo-mix-0925. To continue training with a different "
                            "dataset mix, set 'ignore_fingerprint_mismatch=True' in your NumpyDataLoaderConfig "
                            "to bypass the fingerprint mismatch error. This will probably result in a different data order!"
                        )
                    raise FileNotFoundError(error_msg)
                resolved.append(path_str)

        self._resolved_paths = resolved
        return resolved


@dataclass
class SourceMixtureList(Config):
    """
    A list of source configurations for building a mixture dataset.
    This class ensures that the target ratios of the sources sum to 1.0.

    The purpose of this class is to make managing sources independent from the details of
    materializing those sources with SourceMixtureDatasetConfig.build().

    With this separation, we can define a list of sources in a YAML file without also needing to
    specify parameters like requested_tokens, global_batch_size, or processes.
    """

    sources: List[SourceMixtureConfig]

    def validate(self):
        if not self.sources:
            raise OLMoConfigurationError("sources must not be empty")

        summed_weights = np.sum([source.target_ratio for source in self.sources])

        if not np.allclose(summed_weights, 1.0):
            raise OLMoConfigurationError(f"target_ratios must sum to 1.0, got {summed_weights}")


@dataclass
class SourceTokenDetails:
    """
    A class to hold intermediate selection details for a mixture source.
    """

    config: SourceMixtureConfig
    """
    The configuration object associated with the source.
    """
    population: int
    """
    The total number of tokens available for the source.
    """
    num_selected: int
    """
    The number of tokens to select for the source.
    """

    def for_table(self, total_selected_tokens: int) -> Dict:
        observed_source_ratio = self.num_selected / self.population if self.population else 0.0
        return {
            "source_name": self.config.source_name,
            "source_population": f"{self.population}",
            "num_selected": f"{self.num_selected}",
            "target_ratio": str(self.config.target_ratio),
            "max_repetion_ratio": str(self.config.max_repetition_ratio),
            "max_source_fraction": str(self.config.max_source_fraction),
            "observed_source_ratio": f"{observed_source_ratio:.4}",
            "observed_global_ratio": f"{(self.num_selected / total_selected_tokens):.4}",
        }


@dataclass
class SourcePathTokens:
    path: str
    tokens: int
    max_tokens: int


def _fractional_repetition_capacity_by_path(
    token_counts_by_path: List[int], sequence_length: int, max_repetition_ratio: float
) -> Tuple[int, List[int]]:
    """Return full copies and the final fractional copy's per-path sequence capacities."""
    path_instances = [tokens // sequence_length for tokens in token_counts_by_path]
    repetition_ratio = Fraction(str(max_repetition_ratio))
    full_repetitions = math.floor(repetition_ratio)
    fractional_ratio = repetition_ratio - full_repetitions
    total_instances = sum(path_instances)
    fractional_total = (
        math.floor(total_instances * repetition_ratio) - full_repetitions * total_instances
    )
    fractional_exact = [instances * fractional_ratio for instances in path_instances]
    fractional_by_path = [math.floor(instances) for instances in fractional_exact]
    remaining = fractional_total - sum(fractional_by_path)
    remainder_order = sorted(
        range(len(path_instances)),
        key=lambda idx: (-(fractional_exact[idx] - fractional_by_path[idx]), idx),
    )
    for idx in remainder_order[:remaining]:
        fractional_by_path[idx] += 1

    return full_repetitions, fractional_by_path


@dataclass
class SourceMixtureOutcome:
    name: str
    """
    The name of the source.
    """
    path_tokens: List[SourcePathTokens]
    """
    A list of paths and the associated token counts.
    """


@dataclass
class SourceMixtureDataset:  # Note: "dataset" naming is a bit inconsistent with the rest of the codebase
    """
    A container for a fractionalized mixture of data sources. Do not construct directly,
    use :class:`SourceMixtureDatasetConfig` instead.

    See also :class:`~olmo_core.data.numpy_dataset.NumpyFSLDatasetMixture`, the downstream
    consumer of this dataset.
    """

    sources: List[SourceMixtureOutcome]
    """
    A list of sources and their associated paths and token counts.
    """

    filter_zero_token_paths: bool = False
    """
    If ``True``, drop paths that contribute zero tokens from :meth:`to_index` and :meth:`to_paths`.
    Defaults to ``False`` to preserve the historical path indexing (zero-token paths produce no
    instances downstream anyway, but keeping them keeps the ``(path, idx)`` numbering stable).
    """

    def selected_path_tokens(self) -> List[SourcePathTokens]:
        """
        Flatten the mixture to its per-path token counts, optionally dropping zero-token paths
        (see :data:`filter_zero_token_paths`). Both :meth:`to_index` and :meth:`to_paths` route
        through this so their ``idx`` ordering stays aligned.
        """
        path_tokens = list(chain.from_iterable(outcome.path_tokens for outcome in self.sources))
        if self.filter_zero_token_paths:
            path_tokens = [item for item in path_tokens if item.tokens > 0]
        return path_tokens

    def to_index(self) -> Dict[Tuple[str, int], int]:
        """
        Convert the dataset to an indexed array of dict((int, path), int).
        """
        return {
            (str(item.path), idx): item.tokens
            for idx, item in enumerate(self.selected_path_tokens())
        }

    def to_paths(self) -> List[PathOrStr]:
        """
        Convert the dataset to a list of paths while maintaining stable ordering.
        """
        return [item.path for item in self.selected_path_tokens()]


@dataclass
class SourceMixtureDatasetConfig(Config):
    """
    Configuration for building a dataset from a fractionalized mixture of sources.

    This class manages the creation of training datasets by combining multiple data sources
    according to specified target ratios. It handles token counting, source selection,
    and ensures the final mixture meets the requested dataset size while maintaining
    the desired proportions across sources.

    The build process will:
    1. Count available tokens in each source
    2. Calculate token allocations based on target ratios
    3. Validate that sources have sufficient data
    4. Generate a mixture that respects repetition and fraction limits
    """

    source_list: SourceMixtureList
    """
    A list of source configurations contained in a SourceMixtureList.
    """
    requested_tokens: int
    """
    The desired dataset size, in tokens. This is used to determine the number of tokens to select from each source.
    The total dataset size will be greater than or equal to this value, depending on rounding.
    """
    global_batch_size: int
    """
    The global batch size for training, in tokens. Used to determine the total number of requested instances.
    """
    processes: int = 1
    """
    The number of processes to use for counting tokens in parallel.
    """
    seed: int = 42
    """
    The seed used to generate the dataset. Specifically this seed is used when sampling the actual
    instances to use from each source.
    """
    render_tables: bool = True
    """
    Whether to render tables of the mixture outcome.
    """
    quiet: bool = False

    def validate(self):
        if self.requested_tokens <= 0:
            raise OLMoConfigurationError("requested_tokens must be > 0")
        self.source_list.validate()

    def build(self, *, npdtype: NumpyUIntTypes, sequence_length: int) -> SourceMixtureDataset:
        self.validate()
        available_tokens_by_source: Dict[str, int] = {}
        token_counts_by_source: List[List[int]] = []

        log.info("---------------------------------------------------------")
        log.info("Generating a source mixture from configurations:")
        log.info(self.source_list.sources)

        # Count the number of tokens available for each source
        for source_config in self.source_list.sources:
            log.info(f"Counting tokens for source: {source_config.source_name}")
            token_counts = self._count_tokens_for_paths(
                paths=cast(List[PathOrStr], source_config.resolved_paths),
                source=source_config.source_name,
                npdtype=npdtype,
            )
            token_counts_by_source.append(token_counts)
            available_tokens_by_source[source_config.source_name] = sum(token_counts)

        training_steps = math.ceil(self.requested_tokens / self.global_batch_size)
        assert (
            self.global_batch_size % sequence_length == 0
        ), "global_batch_size must be multiple of sequence_length"
        num_instances_per_batch = self.global_batch_size // sequence_length
        requested_instances = training_steps * num_instances_per_batch

        source_ratios = [Fraction(str(source.target_ratio)) for source in self.source_list.sources]
        total_ratio = sum(source_ratios)
        # Apply Hamilton's method at source granularity before respecting path capacities.
        source_instances = [
            int(requested_instances * ratio / total_ratio) for ratio in source_ratios
        ]
        source_remainders = [
            requested_instances * ratio / total_ratio - instances
            for ratio, instances in zip(source_ratios, source_instances)
        ]
        source_caps = [
            int(
                available_tokens_by_source[source.source_name]
                * Fraction(str(source.max_source_fraction))
                * Fraction(str(source.max_repetition_ratio))
            )
            for source in self.source_list.sources
        ]
        source_instance_caps = []
        for idx, source_config in enumerate(self.source_list.sources):
            full_repetitions, fractional_repetitions = _fractional_repetition_capacity_by_path(
                token_counts_by_path=token_counts_by_source[idx],
                sequence_length=sequence_length,
                max_repetition_ratio=source_config.max_repetition_ratio,
            )
            max_repetition_instances = full_repetitions * sum(
                tokens // sequence_length for tokens in token_counts_by_source[idx]
            ) + sum(fractional_repetitions)
            source_instance_caps.append(
                min(source_caps[idx] // sequence_length, max_repetition_instances)
            )
        remaining_instances = requested_instances - sum(source_instances)
        for _, _, idx in sorted(
            (
                (
                    source_remainders[idx],
                    source_instances[idx] + 1 <= source_instance_caps[idx],
                    idx,
                )
                for idx in range(len(source_instances))
            ),
            reverse=True,
        )[:remaining_instances]:
            source_instances[idx] += 1

        tokens_details_by_source: List[SourceTokenDetails] = []

        # Pad each source target before checking capacity and materializing repetitions.
        for idx, source_config in enumerate(self.source_list.sources):
            num_for_source = available_tokens_by_source[source_config.source_name]
            needed_for_source = source_instances[idx] * sequence_length
            max_for_source = source_caps[idx]

            # Ensure that each source can meet its batch-rounded target.
            if max_for_source < needed_for_source:
                raise OLMoConfigurationError(
                    "Insufficient tokens for source to complete requested training steps: "
                    f"{source_config.source_name} has {max_for_source}, needs {needed_for_source}"
                )
            if source_instances[idx] > source_instance_caps[idx]:
                raise OLMoConfigurationError(
                    "Insufficient source capacity to complete requested training steps: "
                    f"{source_config.source_name} needs {source_instances[idx]} instances, "
                    f"but configured paths provide at most {source_instance_caps[idx]} instances"
                )

            tokens_details_by_source.append(
                SourceTokenDetails(
                    config=source_config,
                    population=num_for_source,
                    num_selected=needed_for_source,
                )
            )

        completed: List[SourceMixtureOutcome] = []
        tokens_per_path_per_source: Dict[str, List[SourcePathTokens]] = {}
        for source_idx, source in enumerate(tokens_details_by_source):
            source_path_tokens = self.get_paths_and_tokens_for_source(
                source_config=source.config,
                token_details=source,
                npdtype=npdtype,
                token_counts_by_path=token_counts_by_source[source_idx],
                sequence_length=sequence_length,
            )
            requested_source_instances = source_instances[source_idx]
            full_repetitions, fractional_repetitions = _fractional_repetition_capacity_by_path(
                token_counts_by_path=token_counts_by_source[source_idx],
                sequence_length=sequence_length,
                max_repetition_ratio=source.config.max_repetition_ratio,
            )
            path_count = len(source.config.resolved_paths)
            materialized_repetitions = len(source_path_tokens) // path_count
            additional_repetitions = max(
                0, math.ceil(source.config.max_repetition_ratio) - materialized_repetitions
            )
            path_capacity = sum(
                path_token.max_tokens // sequence_length for path_token in source_path_tokens
            )
            if path_capacity < requested_source_instances and additional_repetitions:
                for _ in range(additional_repetitions):
                    for path_idx, path in enumerate(source.config.resolved_paths):
                        if materialized_repetitions < full_repetitions:
                            max_tokens = token_counts_by_source[source_idx][path_idx]
                        elif materialized_repetitions == full_repetitions:
                            max_tokens = fractional_repetitions[path_idx] * sequence_length
                        else:
                            max_tokens = 0
                        source_path_tokens.append(
                            SourcePathTokens(path=path, tokens=0, max_tokens=max_tokens)
                        )
                        path_capacity += max_tokens // sequence_length
                    materialized_repetitions += 1
                    if path_capacity >= requested_source_instances:
                        break
            tokens_per_path_per_source[source.config.source_name] = source_path_tokens

        # Round within each source so path capacity cannot shift another source's share.
        for source_idx, source in enumerate(tokens_details_by_source):
            source_path_tokens = tokens_per_path_per_source[source.config.source_name]
            requested_source_instances = source_instances[source_idx]
            path_count = len(source.config.resolved_paths)
            int_instances = []
            remainders = []
            path_fractions = []
            for path_token in source_path_tokens:
                max_instances = path_token.max_tokens // sequence_length
                desired_instances = path_token.tokens // sequence_length
                base_instances = min(int(desired_instances), max_instances)
                int_instances.append(base_instances)
                fraction = (path_token.tokens % sequence_length) / sequence_length
                path_fractions.append(fraction)
                remainders.append(fraction if base_instances < max_instances else 0.0)

            additional_instances_needed = requested_source_instances - sum(int_instances)
            if additional_instances_needed < 0:
                # Trim later repetitions first so per-path ceiling cannot discard unique data.
                while additional_instances_needed < 0:
                    candidates = sorted(
                        (-(idx // path_count), path_fractions[idx], idx)
                        for idx, instances in enumerate(int_instances)
                        if instances > 0
                    )
                    for _, _, idx in candidates[:-additional_instances_needed]:
                        int_instances[idx] -= 1
                        additional_instances_needed += 1

            while additional_instances_needed > 0:
                eligible_indices = [
                    idx
                    for idx in range(len(int_instances))
                    if int_instances[idx] < source_path_tokens[idx].max_tokens // sequence_length
                ]
                if not eligible_indices:
                    break

                # Exhaust earlier occurrences before spending capacity on repetitions.
                repetition_idx = min(idx // path_count for idx in eligible_indices)
                eligible_indices = [
                    idx for idx in eligible_indices if idx // path_count == repetition_idx
                ]

                # Refill evenly after a path reaches its capacity.
                base = additional_instances_needed // len(eligible_indices)
                if base:
                    for idx in eligible_indices:
                        max_instances = source_path_tokens[idx].max_tokens // sequence_length
                        int_instances[idx] += min(base, max_instances - int_instances[idx])
                else:
                    candidates_with_remainders = [
                        (remainders[idx], idx) for idx in eligible_indices
                    ]
                    candidates_with_remainders.sort(reverse=True)
                    for _, idx in candidates_with_remainders[:additional_instances_needed]:
                        int_instances[idx] += 1

                additional_instances_needed = requested_source_instances - sum(int_instances)

            if additional_instances_needed > 0:
                max_possible_instances = sum(
                    path_token.max_tokens // sequence_length for path_token in source_path_tokens
                )
                raise OLMoConfigurationError(
                    "Insufficient source capacity to complete requested training steps: "
                    f"{source.config.source_name} needs {requested_source_instances} instances, "
                    f"but configured paths provide at most {max_possible_instances}"
                )

            for path_token, instances in zip(source_path_tokens, int_instances):
                path_token.tokens = instances * sequence_length

        final_token_distribution: Dict[str, float] = {}
        for source_name, source_path_tokens in tokens_per_path_per_source.items():
            completed.append(
                SourceMixtureOutcome(
                    name=source_name,
                    path_tokens=source_path_tokens,
                )
            )
            final_token_distribution[source_name] = sum(path.tokens for path in source_path_tokens)

        total_tokens = sum(final_token_distribution.values())
        final_token_distribution = {
            k: v / total_tokens for k, v in final_token_distribution.items()
        }

        if self.render_tables:
            self.render_mixture_outcome_tables(tokens_details_by_source)

        for outcome in completed:
            for item in outcome.path_tokens:
                log.info(f"Selected {item.tokens} tokens from {outcome.name} at {item.path}")

        token_difference = total_tokens - self.requested_tokens
        percent_difference = (token_difference / self.requested_tokens) * 100
        log.info(
            f"Total tokens in mixture: {total_tokens} "
            f"(requested: {self.requested_tokens}, diff: {token_difference:+} tokens, "
            f"{percent_difference:+.2f}%)"
        )

        original_token_distribution = {
            source_config.source_name: source_config.target_ratio
            for source_config in self.source_list.sources
        }
        for source_name, ratio in original_token_distribution.items():
            diff = np.abs(final_token_distribution.get(source_name, 0) - ratio)
            log.info(f"{source_name}: {diff:.4f} difference from target ratio {ratio:.4f}")

        return SourceMixtureDataset(sources=completed)

    def get_paths_and_tokens_for_source(
        self,
        source_config: SourceMixtureConfig,
        token_details: SourceTokenDetails,
        npdtype: NumpyUIntTypes,
        token_counts_by_path: Optional[List[int]] = None,
        *,
        sequence_length: int = 1,
    ) -> List[SourcePathTokens]:
        """
        Get the paths and resulting token count for a source.

        :param token_counts_by_path: Optional token counts in the same order as the resolved paths.
            If provided, these are reused instead of counting the files again.
        :param sequence_length: Number of tokens per complete sequence.
        """
        path_tokens: List[SourcePathTokens] = []

        resolved_paths = source_config.resolved_paths
        if token_counts_by_path is None:
            token_counts_by_path = [
                self._count_tokens_for_file(path, npdtype) for path in resolved_paths
            ]
        elif len(token_counts_by_path) != len(resolved_paths):
            raise ValueError("token_counts_by_path must match the resolved paths")
        available_token_counts = token_counts_by_path

        full_repetitions, fractional_repetitions = _fractional_repetition_capacity_by_path(
            token_counts_by_path=available_token_counts,
            sequence_length=sequence_length,
            max_repetition_ratio=source_config.max_repetition_ratio,
        )

        def max_tokens_for_path(path_idx: int, repetition_idx: int) -> int:
            if repetition_idx < full_repetitions:
                return available_token_counts[path_idx]
            if repetition_idx == full_repetitions:
                return fractional_repetitions[path_idx] * sequence_length
            return 0

        if token_details.population == 0:
            return [
                SourcePathTokens(path=path, tokens=0, max_tokens=tokens)
                for path, tokens in zip(resolved_paths, available_token_counts)
            ]

        take_ratio = token_details.num_selected / token_details.population

        # When we need more than 1 repetition of the source data we have a take ration > 1
        if take_ratio > 1:
            take_ratios = []
            remaining = take_ratio

            while remaining > 0:
                chunk = min(1.0, remaining)
                take_ratios.append(chunk)
                remaining -= chunk

            for repetition_idx, ratio in enumerate(take_ratios):
                for path_idx, (path, available_tokens) in enumerate(
                    zip(resolved_paths, available_token_counts)
                ):
                    tokens_to_keep = int(math.ceil(available_tokens * ratio))
                    path_tokens.append(
                        SourcePathTokens(
                            path=path,
                            tokens=tokens_to_keep,
                            max_tokens=max_tokens_for_path(path_idx, repetition_idx),
                        )
                    )

            return path_tokens

        for path_idx, (path, available_tokens) in enumerate(
            zip(resolved_paths, available_token_counts)
        ):
            tokens_to_keep = int(math.ceil(available_tokens * take_ratio))
            path_tokens.append(
                SourcePathTokens(
                    path=path,
                    tokens=tokens_to_keep,
                    max_tokens=max_tokens_for_path(path_idx, 0),
                )
            )

        return path_tokens

    def _count_tokens_for_paths(
        self, paths: List[PathOrStr], source: Optional[str], npdtype: NumpyUIntTypes
    ) -> List[int]:
        """
        Count the number of tokens for each source file in parallel.

        Args:
            paths: The source files whose tokens should be counted.
            source: The source name for progress reporting.
            npdtype: The data type of the source tokens.

        Returns:
            Token counts in input order, including duplicate paths.
        """

        with ThreadPoolExecutor(max_workers=self.processes) as executor:
            futures = []
            for path in paths:
                futures.append(
                    executor.submit(self._count_tokens_for_file, path=path, npdtype=npdtype)
                )
            future_indices = {future: idx for idx, future in enumerate(futures)}
            results = [0] * len(futures)

            with Progress(disable=self.quiet) as progress:
                task = progress.add_task(
                    f"Counting available tokens for source: {source}", total=len(futures)
                )
                for future in as_completed(futures):
                    progress.update(task, advance=1)
                    results[future_indices[future]] = future.result()

            return results

    def _count_tokens_for_file(self, path: PathOrStr, npdtype: NumpyUIntTypes) -> int:
        return self._bytes_to_tokens(get_file_size(path), npdtype=npdtype)

    def _bytes_to_tokens(self, num_bytes: int, npdtype: NumpyUIntTypes) -> int:
        """
        Convert bytes to tokens based on the dtype.
        """
        return num_bytes // npdtype(int(0)).itemsize

    def render_mixture_outcome_tables(self, results: List[SourceTokenDetails]) -> None:
        """
        Render tables enumerating the global and per-source mixture outcomes.
        """
        selected_tokens = sum(item.num_selected for item in results)
        source_rows = [item.for_table(selected_tokens) for item in results]
        source_headers = source_rows[0].keys()

        source_table = Table(title="Outcome by source")
        for header in source_headers:
            source_table.add_column(header)

        for row in source_rows:
            source_table.add_row(*[row[header] for header in source_headers])

        log.info(self.table_to_text(source_table))

        total_tokens = sum([item.population for item in results])
        observed_global_ratio = f"{(selected_tokens / total_tokens):.4}"

        global_table = Table(title="Global outcome")
        global_headers = [
            "total_tokens",
            "selected_tokens",
            "observed_global_ratio",
        ]

        for header in global_headers:
            global_table.add_column(header)

        global_table.add_row(f"{total_tokens:.2e}", f"{selected_tokens:.2e}", observed_global_ratio)
        log.info(self.table_to_text(global_table))

    def table_to_text(self, table: Table) -> Text:
        """Generate an ascii formatted presentation of a Rich table
        Eliminates column styling
        """
        console = Console(width=250)
        with console.capture() as capture:
            table.width = 250
            console.print(table)

        return Text.from_ansi(capture.get())
