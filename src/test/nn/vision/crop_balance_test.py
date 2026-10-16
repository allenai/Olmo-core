"""
Tests for :mod:`olmo_core.nn.vision.crop_balance`: the share plan, and balanced encoding against
each rank encoding its own crops (features and summed gradients). The distributed tests spawn
their processes: forked children of a test worker that already ran CPU kernels can hang in OpenMP.
"""

from typing import List

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from olmo_core.nn.vision.crop_balance import balanced_encode, plan_crop_balance
from olmo_core.testing import BACKENDS, run_distributed_test

# Crops per rank, by world size. Ranks without crops still encode a placeholder crop.
_COUNTS = {
    1: [5],
    2: [7, 1],
    3: [9, 1, 0],
    4: [380, 1, 1, 200],
}


@pytest.mark.parametrize(
    "counts",
    [[5], [1, 1], [380, 0, 1, 1, 200, 0, 0, 1], [0, 0, 7], [3, 3, 3], [1, 0, 0, 0], [0, 0, 0]],
)
def test_plan_shares_are_even_and_keep_each_rank_in_order(counts: List[int]):
    world_size = len(counts)
    send = plan_crop_balance(counts, world_size)
    assert [sum(row) for row in send] == counts
    shares = [sum(send[src][dst] for src in range(world_size)) for dst in range(world_size)]
    assert sum(shares) == sum(counts)
    assert max(shares) - min(shares) <= 1

    # Route crop ids out and back as the two all-to-alls do: every rank gets its own crops back
    # in their original order.
    crops = [[(src, i) for i in range(n)] for src, n in enumerate(counts)]
    received = []
    for dst in range(world_size):
        block = []
        for src in range(world_size):
            start = sum(send[src][:dst])
            block.append(crops[src][start : start + send[src][dst]])
        received.append(block)
    returned = [
        [item for dst in range(world_size) for item in received[dst][src]]
        for src in range(world_size)
    ]
    assert returned == crops


def test_plan_gives_every_rank_a_crop_when_every_rank_holds_one():
    send = plan_crop_balance([1, 1, 1, 1, 40], 5)
    assert all(sum(send[src][dst] for src in range(5)) >= 1 for dst in range(5))


def test_plan_rejects_bad_counts():
    with pytest.raises(ValueError, match="Expected 3"):
        plan_crop_balance([1, 2], 3)
    with pytest.raises(ValueError, match="non-negative"):
        plan_crop_balance([1, -1], 2)


def _encoder(frozen: bool) -> nn.Module:
    torch.manual_seed(0)
    encoder = nn.Sequential(nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 4))
    for param in encoder.parameters():
        param.requires_grad_(not frozen)
    return encoder


def _summed_grads(module: nn.Module) -> List[torch.Tensor]:
    grads = []
    for param in module.parameters():
        assert param.grad is not None
        grad = param.grad.detach().clone()
        dist.all_reduce(grad)
        grads.append(grad)
    return grads


def run_balanced_encode_matches_local(frozen: bool):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = (
        torch.device("cuda", rank) if "nccl" in str(dist.get_backend()) else torch.device("cpu")
    )
    counts = _COUNTS[world_size]
    torch.manual_seed(100 + rank)
    crops = torch.randn(max(counts[rank], 1), 6, device=device)
    weights = torch.randn(crops.shape[0], 4, device=device)
    group = dist.new_group()

    local_encoder = _encoder(frozen).to(device)
    expected = local_encoder(crops)
    balanced_encoder = _encoder(frozen).to(device)
    actual, gathered = balanced_encode(balanced_encoder, crops, group)

    assert gathered == [max(n, 1) for n in counts]
    torch.testing.assert_close(actual, expected)
    if frozen:
        assert not actual.requires_grad
        return
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    for got, want in zip(_summed_grads(balanced_encoder), _summed_grads(local_encoder)):
        torch.testing.assert_close(got, want)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("frozen", [False, True], ids=["trainable", "frozen"])
def test_balanced_encode_matches_each_rank_encoding_its_own_crops(backend: str, frozen: bool):
    run_distributed_test(
        run_balanced_encode_matches_local,
        world_size=4 if backend == "gloo" else 2,
        backend=backend,
        start_method="spawn",
        func_args=(frozen,),
    )


def run_balanced_encode_rejects_a_rank_without_crops():
    group = dist.new_group()
    crops = torch.randn(0 if dist.get_rank() == 0 else 3, 6)
    with pytest.raises(ValueError, match="at least one crop"):
        balanced_encode(_encoder(False), crops, group)


def test_balanced_encode_rejects_a_rank_without_crops():
    run_distributed_test(
        run_balanced_encode_rejects_a_rank_without_crops, world_size=2, start_method="spawn"
    )
