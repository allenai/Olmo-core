from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.utils import get_local_tensor, get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.lm_head import LMHeadConfig, LMHeadType, LMLossImplementation
from olmo_core.testing import requires_gpu, requires_multi_gpu, run_distributed_test
from olmo_core.utils import get_default_device, record_flops, seed_all


def test_lm_head_builder_config():
    lm_head = LMHeadConfig(name=LMHeadType.default).build(d_model=64, vocab_size=128)
    assert lm_head.w_out.bias is not None

    lm_head = LMHeadConfig(name=LMHeadType.default, bias=False).build(d_model=64, vocab_size=128)
    assert lm_head.w_out.bias is None

    with pytest.raises(OLMoConfigurationError):
        LMHeadConfig(name=LMHeadType.normalized, bias=True).build(d_model=64, vocab_size=128)


@requires_gpu
def test_lm_head_fused_linear_loss(
    d_model: int = 256,
    vocab_size: int = 1024,
    z_loss_multiplier: float = 1e-3,
    loss_reduction: str = "sum",
):
    seed_all(42)
    device = torch.device("cuda")

    config1 = LMHeadConfig(loss_implementation=LMLossImplementation.default, bias=False)
    lm_head1 = config1.build(d_model=d_model, vocab_size=vocab_size, init_device="cuda")
    config2 = LMHeadConfig(loss_implementation=LMLossImplementation.fused_linear, bias=False)
    lm_head2 = config2.build(d_model=d_model, vocab_size=vocab_size, init_device="cuda")

    lm_head2.load_state_dict(lm_head1.state_dict())

    B, S = 2, 32
    inputs1 = torch.randn(B, S, d_model, device=device, requires_grad=True)
    inputs2 = inputs1.detach().clone().requires_grad_(True)
    labels = torch.randint(0, vocab_size, (B, S), device=device)
    loss_div_factor = B * S

    output1 = lm_head1(
        inputs1,
        labels=labels,
        loss_div_factor=loss_div_factor,
        loss_reduction=loss_reduction,
        z_loss_multiplier=z_loss_multiplier,
    )
    _, loss1, ce_loss1, z_loss1 = output1
    loss1.backward()
    assert inputs1.grad is not None

    output2 = lm_head2(
        inputs2,
        labels=labels,
        loss_div_factor=loss_div_factor,
        loss_reduction=loss_reduction,
        z_loss_multiplier=z_loss_multiplier,
    )
    _, loss2, ce_loss2, z_loss2 = output2
    loss2.backward()
    assert inputs2.grad is not None

    torch.testing.assert_close(loss1, loss2)
    torch.testing.assert_close(ce_loss1, ce_loss2)
    torch.testing.assert_close(z_loss1, z_loss2)
    torch.testing.assert_close(inputs1.grad, inputs2.grad)


def run_lm_head_tp(
    *,
    checkpoint_dir: Path,
    config: LMHeadConfig,
    d_model: int,
    vocab_size: int,
    loss_reduction: str,
    loss_div_factor: float,
    z_loss_multiplier: float,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    loss: torch.Tensor,
    ce_loss: torch.Tensor,
    z_loss: torch.Tensor,
    grad: torch.Tensor,
):
    seed_all(42)
    device = get_default_device()
    tp_mesh = dist.init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("tp",))

    lm_head = config.build(d_model=d_model, vocab_size=vocab_size, init_device="meta")
    lm_head.apply_tp(tp_mesh, input_layouts=(Shard(1), Replicate()))
    lm_head.to_empty(device=device)

    load_model_and_optim_state(checkpoint_dir, lm_head)

    inputs = inputs.to(device=device).requires_grad_(True)
    local_inputs = distribute_tensor(inputs, device_mesh=tp_mesh, placements=(Shard(1),))
    local_output = lm_head(
        local_inputs,
        labels=labels,
        loss_div_factor=loss_div_factor,
        loss_reduction=loss_reduction,
        z_loss_multiplier=z_loss_multiplier,
    )
    _, local_loss, local_ce_loss, local_z_loss = local_output

    # The loss for optimizing ('local_loss') will have been reduced across the TP group, but not the losses
    # for logging ('local_ce_loss' and 'local_z_loss').
    dist.all_reduce(local_ce_loss)
    local_ce_loss.div_(get_world_size())
    dist.all_reduce(local_z_loss)
    local_z_loss.div_(get_world_size())

    local_loss.backward()
    assert local_inputs.grad is not None

    torch.testing.assert_close(get_local_tensor(local_loss), loss.to(device=device))
    torch.testing.assert_close(get_local_tensor(local_ce_loss), ce_loss.to(device=device))
    torch.testing.assert_close(get_local_tensor(local_z_loss), z_loss.to(device=device))
    torch.testing.assert_close(
        get_local_tensor(local_inputs.grad),
        get_local_tensor(
            distribute_tensor(grad.to(device=device), device_mesh=tp_mesh, placements=(Shard(1),))
        ),
    )


@requires_multi_gpu
@pytest.mark.parametrize("head_type", [LMHeadType.default])
@pytest.mark.parametrize(
    "loss_implementation", [LMLossImplementation.default, LMLossImplementation.fused_linear]
)
@pytest.mark.parametrize("layer_norm", [LayerNormConfig(bias=False), None])
@pytest.mark.parametrize("d_model", [64])
@pytest.mark.parametrize("vocab_size", [128])
@pytest.mark.parametrize("loss_reduction", ["sum"])
def test_lm_head_tp(
    tmp_path: Path,
    head_type: LMHeadType,
    loss_implementation: LMLossImplementation,
    layer_norm: Optional[LayerNormConfig],
    d_model: int,
    vocab_size: int,
    loss_reduction: str,
    z_loss_multiplier: float = 1e-2,
):
    seed_all(42)
    device = torch.device("cuda")

    checkpoint_dir = tmp_path / "checkpoint"

    config = LMHeadConfig(
        name=head_type, loss_implementation=loss_implementation, bias=False, layer_norm=layer_norm
    )
    lm_head = config.build(d_model=d_model, vocab_size=vocab_size, init_device="cuda")
    save_model_and_optim_state(checkpoint_dir, lm_head)

    B, S = 2, 32
    inputs = torch.randn(B, S, d_model, device=device, requires_grad=True)
    labels = torch.randint(0, vocab_size, (B, S), device=device)
    loss_div_factor = B * S

    output = lm_head(
        inputs,
        labels=labels,
        loss_div_factor=loss_div_factor,
        loss_reduction=loss_reduction,
        z_loss_multiplier=z_loss_multiplier,
    )
    _, loss, ce_loss, z_loss = output
    loss.backward()
    assert inputs.grad is not None

    run_distributed_test(
        run_lm_head_tp,
        backend="nccl",
        start_method="spawn",
        func_kwargs=dict(
            checkpoint_dir=checkpoint_dir,
            config=config,
            d_model=d_model,
            vocab_size=vocab_size,
            loss_reduction=loss_reduction,
            loss_div_factor=loss_div_factor,
            z_loss_multiplier=z_loss_multiplier,
            inputs=inputs.detach().cpu(),
            labels=labels.detach().cpu(),
            loss=loss.detach().cpu(),
            ce_loss=ce_loss.detach().cpu(),
            z_loss=z_loss.detach().cpu(),
            grad=inputs.grad.detach().cpu(),
        ),
    )


@requires_gpu
@pytest.mark.parametrize("head_type", [LMHeadType.default, LMHeadType.normalized])
def test_lm_head_logits_to_keep(head_type):
    seed_all(42)
    device = torch.device("cuda")
    d_model, vocab_size = 256, 1024
    B, S = 2, 32

    config = LMHeadConfig(name=head_type, loss_implementation=LMLossImplementation.default)
    lm_head = config.build(d_model=d_model, vocab_size=vocab_size, init_device="cuda")

    inputs = torch.randn(B, S, d_model, device=device)
    labels = torch.randint(0, vocab_size, (B, S), device=device)

    # Test integer logits_to_keep (keep last N positions)
    logits_to_keep = 8
    output = lm_head(inputs, labels=labels, logits_to_keep=logits_to_keep, return_logits=True)
    assert output.logits.shape == (B, logits_to_keep, vocab_size)
    output_ref = lm_head(
        inputs[:, -logits_to_keep:], labels=labels[:, -logits_to_keep:], return_logits=True
    )
    assert torch.allclose(output.logits, output_ref.logits) if "output_ref" in locals() else True
    assert torch.allclose(output.loss, output_ref.loss) if "output_ref" in locals() else True

    # Test tensor logits_to_keep (keep specific positions)
    positions = torch.tensor([[5, 10, 15, 20], [8, 12, 16, 24]], device=device)
    output = lm_head(inputs, labels=labels, logits_to_keep=positions, return_logits=True)
    assert output.logits.shape == (B, 4, vocab_size)

    # Test logits_to_keep=0 (keep all)
    output = lm_head(inputs, labels=labels, logits_to_keep=0, return_logits=True)
    assert output.logits.shape == (B, S, vocab_size)

    # Test inference mode
    logits = lm_head(inputs, logits_to_keep=logits_to_keep)
    assert logits.shape == (B, logits_to_keep, vocab_size)


@requires_gpu
@pytest.mark.parametrize("head_type", [LMHeadType.default, LMHeadType.normalized])
@pytest.mark.parametrize(
    "loss_implementation", [LMLossImplementation.default, LMLossImplementation.fused_linear]
)
def test_lm_head_response_logits_only(head_type, loss_implementation):
    seed_all(42)
    device = torch.device("cuda")
    d_model, vocab_size = 256, 1024
    B, S = 2, 32

    if (
        head_type == LMHeadType.normalized
        and loss_implementation == LMLossImplementation.fused_linear
    ):
        pytest.skip("NormalizedLMHead does not support fused_linear")

    config = LMHeadConfig(name=head_type, loss_implementation=loss_implementation)
    lm_head = config.build(d_model=d_model, vocab_size=vocab_size, init_device="cuda")

    inputs = torch.randn(B, S, d_model, device=device)
    labels = torch.randint(0, vocab_size, (B, S), device=device)

    response_mask = torch.zeros(B, S, dtype=torch.bool, device=device)
    response_mask[0, [3, 7, 15, 22]] = True
    response_mask[1, [1, 9, 18]] = True

    output_full = lm_head(inputs, labels=labels, return_logits=True)
    full_logits = output_full.logits
    ref_logits = full_logits.view(-1, vocab_size)[response_mask.view(-1)]

    narrow_logits = lm_head(
        inputs,
        labels=None,
        response_logits_only=True,
        response_mask=response_mask,
    )
    torch.testing.assert_close(narrow_logits, ref_logits)

    output_narrow = lm_head(
        inputs,
        labels=labels,
        response_logits_only=True,
        response_mask=response_mask,
        return_logits=True,
    )
    if output_narrow.logits is not None:
        torch.testing.assert_close(output_narrow.logits, ref_logits)
    torch.testing.assert_close(output_narrow.loss, output_full.loss)


@pytest.mark.parametrize("head_type", [LMHeadType.default, LMHeadType.normalized])
@pytest.mark.parametrize(
    "loss_implementation", [LMLossImplementation.default, LMLossImplementation.fused_linear]
)
def test_lm_head_num_flops_per_token(
    head_type: LMHeadType, loss_implementation: LMLossImplementation
):
    seed_all(0)

    d_model = 128
    seq_len = 32
    batch_size = 1
    vocab_size = 1024

    config = LMHeadConfig(name=head_type, loss_implementation=loss_implementation)
    lm_head = config.build(d_model=d_model, vocab_size=vocab_size, init_device="cpu")

    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    actual_flops = record_flops(lm_head, x, with_backward=True)
    actual_flops_per_token = actual_flops / seq_len

    estimated_flops_per_token = lm_head.num_flops_per_token(seq_len)

    tolerance = 0.02
    relative_error = (
        abs(estimated_flops_per_token - actual_flops_per_token) / actual_flops_per_token
    )
    assert relative_error < tolerance, (
        f"Estimated FLOPs ({estimated_flops_per_token}) differs too much from actual ({actual_flops_per_token}), "
        f"{relative_error=:.2%}, {tolerance=:.2%}"
    )
