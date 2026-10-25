"""Example-sampling weights for dataset groups and supervised-loss targets."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "DatasetSource",
    "SubMixture",
    "compute_flat_mixture_weights",
    "sampling_weights_from_loss_mass",
]


def sampling_weights_from_loss_mass(
    target_loss_mass: Mapping[str, float],
    mean_loss_weight: Mapping[str, float],
) -> dict[str, float]:
    """Convert supervised-loss targets to example-sampling probabilities.

    For target loss mass :math:`t_i` and mean per-example loss weight :math:`m_i`, source
    ``i`` receives probability proportional to :math:`t_i / m_i`.

    :param target_loss_mass: Desired supervised-loss mass by source.
    :param mean_loss_weight: Estimated mean ``sum(loss_masks)`` per example by source.
    :returns: Normalized probabilities in the target mapping's source order.
    :raises ValueError: If mappings are empty, have different keys, or contain nonfinite or
        nonpositive values.
    """
    for name, values in (
        ("target_loss_mass", target_loss_mass),
        ("mean_loss_weight", mean_loss_weight),
    ):
        if not values:
            raise ValueError(f"{name} must not be empty")
        invalid = {
            key: value
            for key, value in values.items()
            if not math.isfinite(float(value)) or value <= 0
        }
        if invalid:
            raise ValueError(f"{name} values must be positive, got {invalid}")
    if set(target_loss_mass) != set(mean_loss_weight):
        missing = sorted(set(target_loss_mass) - set(mean_loss_weight))
        extra = sorted(set(mean_loss_weight) - set(target_loss_mass))
        raise ValueError(
            "Loss-mass calibration source mismatch: "
            f"missing mean weights for {missing}, unexpected means for {extra}"
        )

    def normalize(values: Mapping[str, float]) -> dict[str, float]:
        total = float(sum(values.values()))
        if total <= 0:
            raise ValueError("Cannot normalize a mapping with non-positive total mass")
        return {key: float(value) / total for key, value in values.items()}

    target = normalize(target_loss_mass)
    return normalize(
        {source: target[source] / float(mean_loss_weight[source]) for source in target}
    )


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
