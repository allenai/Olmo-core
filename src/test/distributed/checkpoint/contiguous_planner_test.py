"""Metadata parity includes uneven, empty and repeatedly sharded dimensions."""

import itertools
import random

import pytest
import torch
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import _StridedShard

from olmo_core.distributed.checkpoint.contiguous_planner import (
    contiguous_shard_metadata,
)


def test_metadata_matches_pytorch_for_contiguous_layouts():
    rng = random.Random(17)
    cases = 0
    for shape in ((0,), (1,), (7,), (16,), (0, 7), (7, 0), (11, 9), (7, 5, 3)):
        for mesh_shape in ((1,), (2,), (4,), (2, 2), (2, 3), (2, 2, 2)):
            choices = [Replicate(), *(Shard(dim) for dim in range(len(shape)))]
            for _ in range(8):
                placements = tuple(rng.choice(choices) for _ in mesh_shape)
                for coordinate in itertools.product(*(range(n) for n in mesh_shape)):
                    expected = _compute_local_shape_and_global_offset(
                        shape, mesh_shape, coordinate, placements
                    )
                    assert (
                        contiguous_shard_metadata(shape, mesh_shape, coordinate, placements)
                        == expected
                    )
                    cases += 1
    assert cases >= 1000


@pytest.mark.parametrize("placement", [Partial(), _StridedShard(0, split_factor=2), Shard(-1)])
def test_unsupported_placements_fall_back(placement):
    assert contiguous_shard_metadata((8,), (2,), (0,), (placement,)) is None


def test_large_metadata_does_not_allocate_tensor_indices(monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError("Metadata calculation allocated an index tensor")

    monkeypatch.setattr(torch, "arange", forbid)
    monkeypatch.setattr(torch, "tensor", forbid)
    assert contiguous_shard_metadata((10**12,), (8, 4), (7, 3), (Shard(0), Shard(0))) == (
        (31_250_000_000,),
        (968_750_000_000,),
    )
