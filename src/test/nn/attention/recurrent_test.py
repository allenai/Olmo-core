from test.nn.attention.attention_test import BF16_ATOL, BF16_RTOL
from typing import Any, Dict

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from olmo_core.config import DType
from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.utils import get_full_tensor, get_rank, get_world_size
from olmo_core.nn.attention import (
    AttentionConfig,
    GatedDeltaNetConfig,
    NemotronMamba2Config,
)
from olmo_core.nn.attention.recurrent import GatedDeltaNet
from olmo_core.nn.attention.ring import UlyssesContextParallelStyle
from olmo_core.testing import requires_gpu, run_distributed_test
from olmo_core.testing.utils import requires_fla, requires_multi_gpu
from olmo_core.utils import get_default_device, seed_all


@requires_fla
@pytest.mark.parametrize(
    "recurrent_config",
    [
        pytest.param(GatedDeltaNetConfig(n_heads=8), id="default"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, n_v_heads=16), id="GVA"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, head_dim=32), id="head_dim=32"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, expand_v=1.0), id="expand_v=1.0"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, conv_size=8, conv_bias=True), id="conv_bias"),
        pytest.param(
            GatedDeltaNetConfig(n_heads=8, allow_neg_eigval=False), id="allow_neg_eigval=False"
        ),
    ],
)
def test_gated_delta_net_config_num_params(recurrent_config: GatedDeltaNetConfig):
    d_model = 512
    module = recurrent_config.build(d_model, layer_idx=0, n_layers=12, init_device="meta")

    # Make sure the estimated number of params matches the actual number of params.
    n_params = sum(p.numel() for p in module.parameters())
    assert recurrent_config.num_params(d_model) == n_params


@requires_fla
@requires_gpu
def test_gated_delta_net_fwd_bwd():
    device = "cuda"
    dtype = torch.bfloat16

    d_model, seq_len, batch_size = 256, 32, 2

    config = GatedDeltaNetConfig(n_heads=8)
    module = config.build(d_model, layer_idx=0, n_layers=12, init_device=device)

    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

    with torch.autocast(device_type=device, dtype=dtype):
        y = module(x)
        assert y.shape == x.shape

        loss = y.sum()
        loss.backward()
    assert x.grad is not None


@requires_fla
def test_gated_delta_net_num_flops_per_token():
    d_model, n_heads, seq_len = 256, 2, 8192

    gdn = GatedDeltaNetConfig(n_heads=n_heads).build(
        d_model, layer_idx=0, n_layers=1, init_device="meta"
    )
    attn = AttentionConfig(n_heads=n_heads).build(
        d_model, layer_idx=0, n_layers=1, init_device="meta"
    )

    # At long sequence lengths, recurrent layers use fewer FLOPs than quadratic attention.
    gdn_flops = gdn.num_flops_per_token(seq_len)
    attn_flops = attn.num_flops_per_token(seq_len)  # type: ignore
    # Training FLOPs apply a factor of 3 (forward + dgrad + wgrad): the base
    # 2-ops-per-MAC counts become 6, 6, and 24 respectively.
    linear_flops = 6 * sum(
        m.weight.numel() for m in (gdn.w_q, gdn.w_k, gdn.w_v, gdn.w_a, gdn.w_b, gdn.w_g, gdn.w_out)
    )
    conv_flops = 6 * gdn.conv_size * (gdn.key_dim + gdn.key_dim + gdn.value_dim)
    recurrent_flops = 24 * gdn.n_v_heads * gdn.head_k_dim * gdn.head_v_dim

    assert gdn_flops == linear_flops + conv_flops + recurrent_flops
    assert 0 < gdn_flops < attn_flops


def _run_context_parallel_gdn_ulysses(
    checkpoint_dir: str,
    inputs_path: str,
    outputs_path: str,
    gdn_kwargs: Dict[str, Any],
):
    device = get_default_device()
    mesh = init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("cp",))

    gdn = GatedDeltaNet(init_device=device.type, **gdn_kwargs)
    gdn.apply_cp(mesh["cp"], uly=UlyssesContextParallelStyle())
    load_model_and_optim_state(checkpoint_dir, gdn)

    # Load the input and split it across ranks on the sequence dimension.
    x = torch.load(inputs_path, map_location=device)
    rank, world_size = get_rank(), get_world_size()
    chunk_size = x.size(1) // world_size
    x_local = x[:, rank * chunk_size : (rank + 1) * chunk_size, :]

    with torch.autocast(device.type, dtype=x_local.dtype):
        local_y = gdn(x_local)
    y = DTensor.from_local(local_y, mesh, (Shard(1),))

    og_y = torch.load(outputs_path, map_location=device)
    tol_scale = 2  # requires slightly more tolerance than default
    torch.testing.assert_close(
        og_y, get_full_tensor(y), rtol=BF16_RTOL * tol_scale, atol=BF16_ATOL * tol_scale
    )


@requires_multi_gpu
@requires_fla
def test_context_parallel_gdn_ulysses(tmp_path):
    seed_all(0)
    device = get_default_device()

    # n_heads must be divisible by CP degree (world_size=2).
    gdn_kwargs: Dict[str, Any] = {"d_model": 128, "n_heads": 8}
    gdn = GatedDeltaNet(init_device=device.type, **gdn_kwargs)

    bs, seq_len = 2, 64
    x = torch.randn(bs, seq_len, gdn_kwargs["d_model"], device=device, dtype=torch.bfloat16)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        y = gdn(x)

    outputs_path = tmp_path / "gdn_y.pt"
    torch.save(y, outputs_path)
    inputs_path = tmp_path / "gdn_x.pt"
    torch.save(x, inputs_path)
    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, gdn)

    run_distributed_test(
        _run_context_parallel_gdn_ulysses,
        backend="nccl",
        start_method="spawn",
        func_args=(checkpoint_dir, inputs_path, outputs_path, gdn_kwargs),
    )


def _small_nemotron_mamba2_config() -> NemotronMamba2Config:
    return NemotronMamba2Config(
        mamba_num_heads=4,
        mamba_head_dim=16,
        n_groups=1,
        ssm_state_size=16,
        chunk_size=8,
        dtype=DType.float32,
    )


def test_nemotron_mamba2_num_params():
    d_model = 64
    config = _small_nemotron_mamba2_config()
    module = config.build(d_model, layer_idx=0, n_layers=2, init_device="meta")
    actual = sum(p.numel() for p in module.parameters())
    assert config.num_params(d_model) == actual


def test_nemotron_mamba2_forward_runs_on_cpu():
    seed_all(0)
    d_model = 64
    module = _small_nemotron_mamba2_config().build(d_model, layer_idx=0, n_layers=2)
    x = torch.randn(2, 12, d_model)
    with torch.no_grad():
        out = module(x)
    assert out.shape == (2, 12, d_model)


def test_nemotron_mamba2_dt_bias_initialized_within_configured_range():
    seed_all(0)
    config = NemotronMamba2Config(
        mamba_num_heads=512,
        mamba_head_dim=8,
        n_groups=1,
        ssm_state_size=16,
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_floor=0.0001,
        dtype=DType.float32,
    )
    # init_weights drives the seeded generator; without it the constructor's reset_parameters uses
    # the global RNG (seeded by seed_all above), which is fine for the range check.
    module = config.build(64, layer_idx=0, n_layers=1)
    # softplus(dt_bias) must recover effective timesteps within the configured range, not the
    # softplus(1) == 1.31 that a plain ones-init would give.
    dt = torch.nn.functional.softplus(module.dt_bias.float())
    assert dt.min() >= config.time_step_floor - 1e-6
    assert dt.max() <= config.time_step_max + 1e-4


def test_nemotron_mamba2_dt_bias_uses_supplied_generator_not_global_rng():
    from olmo_core.nn.transformer.init import InitMethod

    config = _small_nemotron_mamba2_config()

    def init_dt_bias(global_seed: int) -> torch.Tensor:
        # Perturb the ambient global RNG differently each time.
        torch.manual_seed(global_seed)
        module = config.build(64, layer_idx=0, n_layers=2)
        generator = torch.Generator()
        generator.manual_seed(2026)
        module.init_weights(
            init_method=InitMethod.normal,
            d_model=64,
            block_idx=0,
            num_blocks=2,
            generator=generator,
        )
        return module.dt_bias.detach().clone()

    # An identical supplied generator must produce identical dt_bias regardless of global RNG state.
    torch.testing.assert_close(init_dt_bias(1), init_dt_bias(999))
