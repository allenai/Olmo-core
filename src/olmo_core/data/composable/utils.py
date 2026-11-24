import concurrent.futures
import warnings
from typing import Callable, List, Literal, Optional, Sequence, Type, TypeVar, Union

import numpy as np
import torch

import olmo_core.io as io
from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoConfigurationError, OLMoEnvironmentError

from ..types import NumpyUIntTypes
from ..utils import get_rng


def _warmup_clients(paths: Sequence[PathOrStr]):
    # Maybe create client up front to work around a threading issue in boto.
    if any(str(p).startswith("s3://") for p in paths):
        io._get_s3_client("s3")

    if any(str(p).startswith("r2://") for p in paths):
        try:
            io._get_s3_client("r2")
        except OLMoEnvironmentError:
            # R2 might not be needed, so ignore this error. We will get an error
            # later if R2 is needed.
            pass

    if any(str(p).startswith("weka://") for p in paths):
        try:
            io._get_s3_client("weka")
        except OLMoEnvironmentError:
            # Weka might not be needed, so ignore this error. We will get an error
            # later if Weka is needed.
            pass


T = TypeVar("T")


def path_map(
    func: Callable[[PathOrStr], T],
    paths: Sequence[PathOrStr],
    *,
    max_workers: Optional[int] = None,
    method: Literal["threads", "processes"] = "threads",
) -> List[T]:
    """
    Call a function on each path, returning a list of the results, in order.

    :param func: The function to map to the paths and their indices.
    :param max_workers: The number of workers threads/processes. Set to 0 to execute synchronously
        in the main thread/process.
    :param method: Whether to use multi-threading or multi-processing.

    :returns: The results, in the same order as :data:`paths`.
    """
    if max_workers == 0 or len(paths) <= 1:
        return [func(path) for path in paths]

    executor_class: Union[
        Type[concurrent.futures.ThreadPoolExecutor],
        Type[concurrent.futures.ProcessPoolExecutor],
    ]
    if method == "threads":
        _warmup_clients(paths)
        executor_class = concurrent.futures.ThreadPoolExecutor
    elif method == "processes":
        executor_class = concurrent.futures.ProcessPoolExecutor
    else:
        raise ValueError(method)

    with executor_class(max_workers=max_workers) as executor:
        futures = [executor.submit(func, path) for path in paths]

    return [future.result() for future in futures]


def format_fname_from_fields(prefix: str, **fields) -> str:
    parts = [prefix]
    for key in sorted(fields):
        value = fields[key]
        if value is not None:
            parts.append(f"{key}{value}")
    return "_".join(parts)


def format_token_count(n: int) -> str:
    if n >= 1_000_000_000_000:
        return f"{n / 1_000_000_000_000:.1f}T"
    elif n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    else:
        return str(n)


def as_ndarray(array: Union[Sequence[int], Sequence[bool]]) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    elif isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    else:
        return np.array(array)


def as_tensor(array: Union[Sequence[int], Sequence[bool]]) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array
    elif isinstance(array, np.ndarray):
        if array.dtype == np.bool_:
            return torch.tensor(array, device="cpu")
        else:
            return torch.tensor(array.astype(np.int_), dtype=torch.long, device="cpu")
    else:
        return torch.tensor(array, device="cpu")


def calculate_sample_sizes(
    source_sizes: Sequence[int],
    target_ratios: Sequence[float],
    max_repetition_factors: Sequence[float],
    target_size: Optional[int] = None,
    labels: Optional[Sequence[str]] = None,
    unit: str = "tokens",
) -> np.ndarray:
    """
    Calculate the number of items needed to sample from each source in order to match the target ratios.
    """
    assert len(source_sizes) == len(target_ratios) == len(max_repetition_factors)
    if labels is not None:
        assert len(labels) == len(source_sizes)

    ratios = np.array(target_ratios)
    sizes = np.array(source_sizes)
    max_repetition_factors_ = np.array(max_repetition_factors)

    assert (ratios > 0.0).all(), f"All ratios must be positive! Got {target_ratios}"
    assert (
        max_repetition_factors_ >= 1.0
    ).all(), f"All max repetition factors must be at least 1.0! Got {max_repetition_factors}"
    assert (sizes > 0).all(), f"All source sizes must be positive! Got {sizes}"

    strict = True
    if target_size is None:
        strict = False
        target_size = sizes.sum()

    # Normalize ratios.
    ratio_total = ratios.sum()
    if not np.allclose(ratio_total, 1.0):
        ratios = ratios / ratio_total
        new_ratio_summary_lines = []
        for i in range(len(ratios)):
            label_str: str
            if labels is not None:
                label_str = f"'{labels[i]}'"
            else:
                label_str = f"{i}"
            new_ratio_summary_lines.append(
                f" ❯ Source {label_str}: target ratio adjusted from {target_ratios[i]} to {ratios[i]}"
            )
        new_ratio_summary = "\n".join(new_ratio_summary_lines)
        warnings.warn(
            f"Target mixing ratios don't sum to 1. They will be normalized as follows:\n{new_ratio_summary}",
            UserWarning,
        )

    # Determine the number of items to sample from each source.
    # This is tricky because the sources may have different sizes, yet we want to stay
    # true to the sampling ratios while minimizing the number of dropped or over-sampled items.
    # To that end, the optimal natural distribution of items over sources is the one that
    # matches the target sampling ratios. We'll call that the 'ideal_sample_sizes'.
    ideal_sample_sizes = target_size * ratios
    # But since the actual (natural) distribution probably differs from the ideal one, it's
    # not possible to match the target ratios without some dropping or oversampling.
    # So we first calculate how much oversampling/repetition is needed per source, and then cap that
    # according to the given `max_repetitions_per_source`.
    max_repetition_factors_needed = np.maximum(ideal_sample_sizes / sizes, 1.0)
    repetition_factors_to_use = np.minimum(max_repetition_factors_, max_repetition_factors_needed)
    # Now we can adjust sizes based on the repetitions needed.
    sizes_to_use = sizes * repetition_factors_to_use
    # Lastly, we need to adjust the ideal sample sizes down until by the smallest common factor
    # that would result in all sample sizes being less than or equal to the number of items available
    # from the corresponding source. We can calculate that factor by finding the source with the
    # largest relative difference between its available size (number of items after oversampling) and
    # its ideal sample size, and taking that ratio.
    adjustment_factor = min(1.0, (sizes_to_use / ideal_sample_sizes).min())
    actual_sample_sizes = ideal_sample_sizes * adjustment_factor

    # Sanity check.
    # Sample sizes should stay true to target ratios.
    actual_ratios = actual_sample_sizes / actual_sample_sizes.sum()
    assert np.allclose(
        ratios, actual_ratios
    ), f"expected ratios: {ratios}, actual ratios: {actual_ratios}"
    # And sample sizes shouldn't be larger than the number of items available.
    actual_sample_sizes_int = actual_sample_sizes.astype(np.uint64)
    assert (actual_sample_sizes_int <= sizes_to_use).all()
    assert target_size is not None
    actual_size = actual_sample_sizes.sum()
    if strict and not np.allclose(target_size, actual_size):
        idx_of_max_diff = np.argmax(max_repetition_factors_needed - max_repetition_factors_)
        if labels is not None:
            label_str = f"with label '{labels[idx_of_max_diff]}'"
        else:
            label_str = f"with index {idx_of_max_diff}"
        required_sample_size = int(ideal_sample_sizes[idx_of_max_diff])
        provided_sample_size = int(sizes[idx_of_max_diff] * max_repetition_factors[idx_of_max_diff])
        raise OLMoConfigurationError(
            f"Unable to meet target size of {int(target_size):,d} {unit} with the given "
            f"source ratios and max repetition factors. The best we can do is {int(actual_size):,d} {unit}. "
            f"The source with the biggest discrepancy between its required sample size "
            f"(~{required_sample_size:,d} {unit}, {100 * ratios[idx_of_max_diff]:.1f}% of mix) and "
            f"the size it can provide after accounting for the max repetition factor "
            f"({sizes[idx_of_max_diff]:,d} x {max_repetition_factors[idx_of_max_diff]:.2f} ~= {provided_sample_size:,d} {unit}) "
            f"is the source {label_str}. Consider either decreasing the target size of the mix "
            f"to {int(actual_size):,d} {unit} or increasing the max repetition factor for that source "
            f"to {max_repetition_factors_needed[idx_of_max_diff]:.2f}."
        )

    return actual_sample_sizes_int


def build_global_indices(
    total_instances: int,
    *,
    sequence_length: int,
    max_sequence_length: int,
    seed: Optional[int],
    dtype: NumpyUIntTypes = np.uint32,
) -> np.ndarray:
    """
    Build global (as opposed to rank-local) instance indices as a numpy array, in a way that
    preserves the order of data when ``max_sequence_length`` is fixed but ``sequence_length`` changes.
    """
    assert total_instances < np.iinfo(dtype).max
    assert max_sequence_length % sequence_length == 0
    chunk_size = max_sequence_length // sequence_length
    # Length of dataset would be calculated incorrectly if this didn't hold.
    assert total_instances % chunk_size == 0

    # NOTE: To guarantee the same data order with `self.max_sequence_length` fixed but `self.sequence_length`
    # changing, we need `self.total_instances // chunk_size` to remain constant.
    # This is ensured by requiring `self.max_sequence_length` is a multiple of `self.sequence_length`
    # and assuming that `self.total_instances` is proportional to `chunk_size`, i.e.
    # if `self.sequence_length` is half of `self.max_sequence_length`, then `self.total_instances`
    # should double. This takes some care when implementing an `InstanceSource` to ensure that
    # excess tokens are dropped in a way that respects `self.max_sequence_length`, not `self.sequence_length`.
    chunk_indices = np.arange(total_instances // chunk_size, dtype=dtype)

    # Deterministically shuffle based on epoch and seed
    if seed is not None:
        rng = get_rng(seed)
        rng.shuffle(chunk_indices)

    if chunk_size == 1:
        return chunk_indices

    indices = np.repeat(chunk_indices * chunk_size, chunk_size)
    indices = indices.reshape((-1, chunk_size)) + np.arange(0, chunk_size, dtype=dtype).reshape(
        (1, -1)
    )
    indices = indices.reshape(-1)
    return indices


class _NOT_SET_INT_TYPE(int):
    pass


SEED_NOT_SET = _NOT_SET_INT_TYPE()
"""
A placeholder for the default seed, which can be changed by calling :func:`set_composable_seed()`.
"""

_SEED_RNG: Optional[np.random.Generator] = None


S = TypeVar("S", int, None, Optional[int])


def resolve_seed(default: S) -> S:
    global _SEED_RNG
    if default is SEED_NOT_SET:
        if _SEED_RNG is None:
            return 0  # type: ignore[return-type]
        else:
            return int(_SEED_RNG.integers(0, 2**31 - 1))  # type: ignore[return-type]
    else:
        return default


def set_composable_seed(seed: int):
    """
    Set the global seed for the composable module.
    """
    global _SEED_RNG

    _SEED_RNG = get_rng(seed)


def reset_composable_seed():
    """
    Reset the global seed for the composable module.
    """
    global _SEED_RNG

    _SEED_RNG = None
