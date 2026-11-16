from typing import Literal, Optional

import pytest
import torch
from torch.distributed import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from olmo_core.distributed.utils import get_local_tensor, get_world_size
from olmo_core.nn.cross_entropy_loss import CrossEntropyLoss
from olmo_core.testing import requires_multi_gpu, run_distributed_test
from olmo_core.utils import get_default_device


def compute_loss(
    loss_fn: CrossEntropyLoss,
    logits: torch.Tensor,
    labels: torch.Tensor,
    batch_num_tokens_for_loss: Optional[torch.Tensor],
) -> torch.Tensor:
    ce_loss, z_loss = loss_fn(logits, labels, batch_num_tokens_for_loss)

    loss = ce_loss
    if z_loss is not None:
        loss = loss + z_loss

    if loss_fn.reduction != "none":
        assert loss.shape == tuple(), f"{loss}"
    else:
        assert loss.shape == labels.shape
        loss = loss.sum()

    return loss


def run_cross_entropy_loss_parallel(
    compile: bool,
    reduction: Literal["sum", "mean", "none"],
    z_loss_multiplier: Optional[float],
    logits: torch.Tensor,
    labels: torch.Tensor,
    batch_num_tokens_for_loss: Optional[torch.Tensor],
    grad: torch.Tensor,
    expected_loss: torch.Tensor,
):
    # Init device mesh.
    tp_mesh = init_device_mesh("cuda", (get_world_size(),), mesh_dim_names=("tp",))

    # Put tensors on target device and potentially distributed over the device mesh .
    logits = distribute_tensor(
        logits.to(device=get_default_device()).requires_grad_(),
        device_mesh=tp_mesh,
        placements=(Shard(1),),
    )
    labels = distribute_tensor(
        labels.to(device=get_default_device()), device_mesh=tp_mesh, placements=(Shard(1),)
    )
    if batch_num_tokens_for_loss is not None:
        batch_num_tokens_for_loss = batch_num_tokens_for_loss.to(device=get_default_device())
    grad = distribute_tensor(
        grad.to(device=get_default_device()), device_mesh=tp_mesh, placements=(Shard(1),)
    )
    expected_loss = expected_loss.to(device=get_default_device())

    # Initialize loss and apply parallelism.
    loss_fn = CrossEntropyLoss(
        reduction=reduction, compile=compile, z_loss_multiplier=z_loss_multiplier
    )
    loss_fn.apply_tp(tp_mesh, use_local_output=True)

    # Get loss.
    loss = compute_loss(loss_fn, logits, labels, batch_num_tokens_for_loss)

    # Check loss.
    torch.testing.assert_close(loss.detach(), expected_loss)

    # Trigger backward pass.
    loss.backward()
    assert logits.grad is not None

    # Check gradients.
    torch.testing.assert_close(get_local_tensor(logits.grad), get_local_tensor(grad))


@pytest.mark.parametrize(
    "compile, reduction, ignore_shard",
    [
        pytest.param(False, "mean", False, id="default-mean"),
        pytest.param(False, "mean", True, id="mean-ignored-shard"),
        pytest.param(False, "sum", False, id="default-sum"),
        pytest.param(False, "none", False, id="default-none"),
    ],
)
@pytest.mark.parametrize("normalize_loss", [False, True])
@pytest.mark.parametrize("z_loss_multiplier", [None, 1e-4])
@requires_multi_gpu
def test_cross_entropy_loss_parallel(
    compile: bool,
    ignore_shard: bool,
    reduction: Literal["sum", "mean", "none"],
    normalize_loss: bool,
    z_loss_multiplier: Optional[float],
):
    B, S, V = 4, 16, 256

    loss_fn = CrossEntropyLoss(
        reduction=reduction, compile=compile, z_loss_multiplier=z_loss_multiplier
    )

    labels = torch.randint(0, V, (B, S), device="cuda")
    logits = torch.randn(B, S, V, device="cuda", requires_grad=True)
    labels[0][2] = -100
    labels[2][9] = -100
    labels[3][12] = -100
    if ignore_shard:
        labels[:, : S // 2] = -100
    batch_num_tokens_for_loss = (labels != -100).sum() if normalize_loss else None

    # Get loss.
    loss = compute_loss(loss_fn, logits, labels, batch_num_tokens_for_loss)

    # Trigger backward pass.
    loss.backward()
    assert logits.grad is not None

    run_distributed_test(
        run_cross_entropy_loss_parallel,
        world_size=2,
        backend="nccl",
        start_method="spawn",
        func_args=(
            compile,
            reduction,
            z_loss_multiplier,
            logits.detach().cpu(),
            labels.detach().cpu(),
            batch_num_tokens_for_loss.detach().cpu()
            if batch_num_tokens_for_loss is not None
            else None,
            logits.grad.detach().cpu(),
            loss.detach().cpu(),
        ),
    )
