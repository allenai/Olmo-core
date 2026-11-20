"""Deterministic per-example random-number generation for multimodal datasets."""

from __future__ import annotations

import numpy as np

__all__ = ["make_random_state"]


def make_random_state(seed: int, *seeds: int) -> np.random.RandomState:
    """Build the same independent ``RandomState`` stream used by Molmo2 preprocessing.

    :param seed: Root seed, normally the example index.
    :param seeds: Additional deterministic coordinates, normally the source epoch.

    :returns: A legacy ``RandomState`` backed by MT19937.
    """
    seed_sequence = np.random.SeedSequence(seed, spawn_key=seeds)
    return np.random.RandomState(np.random.MT19937(seed_sequence))
