"""Mixture sampling weights (port of mm_olmo SubMixture rate math)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["DatasetSource", "SubMixture", "compute_flat_mixture_weights"]


@dataclass
class DatasetSource:
    name: str
    sampling_rate: Optional[float] = None
    root_size_factor: Optional[Union[int, float]] = None
    message_weight: Optional[float] = None
    override_p_high_res: Optional[float] = None


@dataclass
class SubMixture:
    name: str
    rate: float
    datasets: Sequence[DatasetSource]


def _dataset_size_factor(source: DatasetSource, dataset_len: int) -> float:
    """mm_olmo root-size score (data_loader.py:264-271), all four branches."""
    if source.root_size_factor == 0:
        return 1.0
    if source.root_size_factor is None:
        return float(np.sqrt(max(dataset_len, 1)))
    if source.root_size_factor < 1:
        return float(np.sqrt(dataset_len * source.root_size_factor))
    return float(np.sqrt(source.root_size_factor))


def compute_flat_mixture_weights(
    groups: Sequence[SubMixture],
    dataset_lengths: dict[str, int],
) -> List[Tuple[str, float]]:
    """Return normalized (dataset_name, global_rate) pairs."""
    flat: List[Tuple[str, float]] = []
    for group in groups:
        if group.rate <= 0 or not group.datasets:
            continue
        factors = []
        for src in group.datasets:
            frac = _dataset_size_factor(src, dataset_lengths[src.name])
            if src.sampling_rate is not None:
                frac *= src.sampling_rate
            factors.append(frac)
        total = sum(factors)
        for src, frac in zip(group.datasets, factors):
            flat.append((src.name, group.rate * (frac / total)))
    norm = sum(w for _, w in flat)
    return [(name, w / norm) for name, w in flat]
