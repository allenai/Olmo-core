"""
Share the vision encoder's work evenly among data-parallel ranks.

The image crops of a packed multimodal batch are unevenly spread: one rank's micro-batch can hold
hundreds of crops while another's is text only. When the language model keeps its ranks in step
(an MoE router all-reduces its load-balancing statistics in every routed layer), every rank then
waits for the one with the most crops. The encoder treats every crop independently, so
:func:`balanced_encode` sends the crops around for each rank to encode an equal share and returns
the features, and in the backward pass their gradients, to the ranks the crops came from. The
features and the encoder's gradients summed over ranks are those of unbalanced encoding up to
floating-point rounding.
"""

from typing import Callable, List, Sequence, Tuple

import torch
import torch.distributed as dist

__all__ = ["plan_crop_balance", "balanced_encode"]


def plan_crop_balance(counts: Sequence[int], world_size: int) -> List[List[int]]:
    """
    Split all ranks' crops, taken in rank order, into ``world_size`` contiguous shares that differ
    by at most one crop.

    :param counts: Crops each rank holds, indexed by rank.
    :param world_size: Number of ranks.

    :returns: ``send[src][dst]``, the number of ``src``'s crops that ``dst`` encodes. Row ``src``
        sums to ``counts[src]`` and column ``dst`` to ``dst``'s share. Each source sends a
        contiguous block of its crops to each destination, in destination order, so concatenating
        the returned features over destinations restores the source's crop order.

    :raises ValueError: If ``counts`` does not hold ``world_size`` non-negative entries.
    """
    if len(counts) != world_size:
        raise ValueError(f"Expected {world_size} crop counts, got {len(counts)}")
    if any(count < 0 for count in counts):
        raise ValueError(f"Crop counts must be non-negative, got {list(counts)}")
    total = sum(counts)
    shares = [
        total // world_size + (1 if dst < total % world_size else 0) for dst in range(world_size)
    ]
    send = [[0] * world_size for _ in range(world_size)]
    src, left = 0, counts[0] if world_size else 0
    for dst, share in enumerate(shares):
        need = share
        while need:
            while left == 0:
                src += 1
                left = counts[src]
            take = min(need, left)
            send[src][dst] += take
            left -= take
            need -= take
    return send


class _CropAllToAll(torch.autograd.Function):
    """``all_to_all_single`` along the crop axis; the backward sends the gradients back."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        output_splits: List[int],
        input_splits: List[int],
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.output_splits = output_splits
        ctx.input_splits = input_splits
        ctx.group = group
        out = x.new_empty((sum(output_splits),) + tuple(x.shape[1:]))
        dist.all_to_all_single(out, x.contiguous(), output_splits, input_splits, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        grad_in = grad_out.new_empty((sum(ctx.input_splits),) + tuple(grad_out.shape[1:]))
        dist.all_to_all_single(
            grad_in,
            grad_out.contiguous(),
            ctx.input_splits,
            ctx.output_splits,
            group=ctx.group,
        )
        return grad_in, None, None, None


def balanced_encode(
    encode: Callable[[torch.Tensor], torch.Tensor],
    crops: torch.Tensor,
    group: dist.ProcessGroup,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Encode ``crops`` with the work shared evenly among the ranks of ``group``.

    Every rank of ``group`` must call this at the same point and hold at least one crop, so that
    every rank encodes at least one crop and its encoder parameters take part in the backward
    pass.

    :param encode: Maps ``(n, ...)`` crops to ``(n, ...)`` features, each row depending only on
        its own crop.
    :param crops: This rank's crops, ``(n_local, ...)``, shaped alike beyond the first dimension
        on every rank.
    :param group: The ranks that share the work.

    :returns: The features of this rank's own crops, in order, and every rank's crop count.

    :raises ValueError: If any rank holds no crop.
    """
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    local = torch.tensor([crops.shape[0]], device=crops.device, dtype=torch.int64)
    gathered = torch.empty(world_size, device=crops.device, dtype=torch.int64)
    dist.all_gather_into_tensor(gathered, local, group=group)
    # One device-to-host sync per call: the split sizes must be known on the host.
    counts = [int(count) for count in gathered.tolist()]
    if min(counts) < 1:
        raise ValueError(f"Every rank must hold at least one crop to balance, got {counts}")
    send = plan_crop_balance(counts, world_size)
    input_splits = send[rank]
    output_splits = [send[src][rank] for src in range(world_size)]
    received = _CropAllToAll.apply(crops, output_splits, input_splits, group)
    features = encode(received)
    if features.shape[0] != received.shape[0]:
        raise ValueError(
            f"The encoder returned {features.shape[0]} rows for {received.shape[0]} crops"
        )
    # What this rank received from each source goes back to that source.
    return _CropAllToAll.apply(features, input_splits, output_splits, group), counts
