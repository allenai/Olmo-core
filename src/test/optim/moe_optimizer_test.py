import pytest
import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.transformer import CompletedPConfig, TransformerConfig
from olmo_core.optim.completedp import CompletedPOptimConfig, completedp_multipliers
from olmo_core.optim.config import OptimGroupOverride
from olmo_core.optim.moe_optimizer import OLMoDDPOptimizer, OLMoDDPOptimizerConfig
from olmo_core.optim.mup import MuPConfig
from olmo_core.testing.distributed import run_distributed_test


def _run_refresh_main_params():
    mesh = init_device_mesh("cpu", (2,))
    param = nn.Parameter(torch.arange(12, dtype=torch.bfloat16).reshape(3, 4))
    optim = object.__new__(OLMoDDPOptimizer)
    optim.should_maintain_fp32_main_param = True
    optim.param_groups = [{"named_params": {"weight": param}}]
    for placement in (Replicate(), Shard(0)):
        main = distribute_tensor(torch.full((12,), -1.0), mesh, [placement])
        moment = distribute_tensor(torch.full((12,), 0.5), mesh, [placement])
        optim.states = {"weight.main": main, "weight.exp_avg": moment}
        optim._copy_model_params_to_main_params()
        assert optim.states["weight.main"] is main
        torch.testing.assert_close(main.full_tensor(), torch.arange(12).float(), rtol=0, atol=0)
        torch.testing.assert_close(moment.full_tensor(), torch.full((12,), 0.5), rtol=0, atol=0)
        torch.testing.assert_close(
            param, torch.arange(12, dtype=torch.bfloat16).reshape(3, 4), rtol=0, atol=0
        )


def test_refresh_main_params_after_model_load():
    run_distributed_test(
        _run_refresh_main_params, world_size=2, backend="gloo", start_method="spawn"
    )


def _run_aligned_flat_model_buffers():
    mesh = init_device_mesh("cpu", (2,))
    params = {
        "gate": nn.Parameter(torch.arange(4, dtype=torch.bfloat16)),
        "norm": nn.Parameter(torch.arange(544, dtype=torch.bfloat16)),
        "scale": nn.Parameter(torch.arange(3, dtype=torch.bfloat16)),
        "projection": nn.Parameter(torch.arange(24, dtype=torch.bfloat16).reshape(4, 6)),
    }
    original = {name: param.detach().clone() for name, param in params.items()}
    optim = object.__new__(OLMoDDPOptimizer)
    optim._device = torch.device("cpu")
    optim._dp_group = mesh.get_group()
    optim.param_groups = [{"pg": "dp", "named_params": params}]
    optim.states = {
        f"{name}.main": distribute_tensor(
            param.detach().float().reshape(-1),
            mesh,
            [Shard(0) if name in ("norm", "projection") else Replicate()],
        )
        for name, param in params.items()
    }
    optim._init_flat_model_param_buffers()
    for name, param in params.items():
        assert param.data_ptr() % 16 == 0, name
        torch.testing.assert_close(param, original[name], rtol=0, atol=0)

    # Padding in model storage must not change packed all-gather offsets or
    # the logical shapes/values used by optimizer state and checkpoints.
    for main in optim.states.values():
        main.to_local().add_(8)
    optim._copy_main_params_to_flat_model_buffers()
    for name, param in params.items():
        torch.testing.assert_close(param, original[name] + 8, rtol=0, atol=0)


def test_flat_model_buffers_preserve_alignment_and_sharded_sync():
    run_distributed_test(
        _run_aligned_flat_model_buffers, world_size=2, backend="gloo", start_method="spawn"
    )


def test_build_groups_applies_overrides():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    config = OLMoDDPOptimizerConfig(
        lr=1e-3,
        group_overrides=[OptimGroupOverride(params=["*bias*"], opts={"weight_decay": 0.0})],
    )
    groups = config.build_groups([model])
    assert isinstance(groups, list)

    # Every parameter lands in exactly one group.
    grouped = [p for g in groups for p in g["named_params"].values()]
    assert len(grouped) == len(list(model.parameters()))

    # The bias parameters land in a group carrying the override option.
    bias_ids = {id(p) for n, p in model.named_parameters() if "bias" in n}
    override_group = next(g for g in groups if g.get("weight_decay") == 0.0)
    assert {id(p) for p in override_group["named_params"].values()} == bias_ids


def test_build_groups_applies_mup_with_wrapper_prefix():
    class Wrapper(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

    model = Wrapper(nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 8)))
    config = OLMoDDPOptimizerConfig(
        lr=1e-3,
        weight_decay=0.1,
        mup=MuPConfig(base_width=8),
        group_overrides=[OptimGroupOverride(params=["*bias*"], opts={"weight_decay": 0.0})],
    )
    groups = config.build_groups([model])

    by_name = {name: g for g in groups for name in g["named_params"]}
    assert set(by_name) == {name for name, _ in model.named_parameters()}
    assert by_name["module.0.weight"]["lr"] == pytest.approx(1e-3)
    assert by_name["module.1.weight"]["lr"] == pytest.approx(0.5e-3)
    for name in ("module.0.bias", "module.1.bias"):
        assert by_name[name]["lr"] == pytest.approx(1e-3)
        assert by_name[name]["weight_decay"] == 0.0
    # The default group's splits still come first, and leave weight decay to the optimizer default.
    assert "weight_decay" not in groups[0]
    assert "weight_decay" not in by_name["module.1.weight"]


def test_build_groups_applies_completedp_with_wrapper_prefix():
    class Wrapper(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

    model = Wrapper(
        TransformerConfig.llama_like(
            d_model=128,
            vocab_size=256,
            n_layers=4,
            n_heads=2,
            feed_forward=FeedForwardConfig(hidden_size=512, bias=False),
            completedp=CompletedPConfig(base_width=64, base_depth=2),
        ).build(init_device="meta")
    )
    completedp = CompletedPOptimConfig(base_batch_size=1, batch_size=4)
    config = OLMoDDPOptimizerConfig(
        lr=1e-3,
        eps=1e-8,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        completedp=completedp,
        group_overrides=[
            OptimGroupOverride(params=["*embeddings.weight"], opts={"weight_decay": 0.0})
        ],
    )
    groups = config.build_groups([model])

    by_name = {name: g for g in groups for name in g["named_params"]}
    assert set(by_name) == {name for name, _ in model.named_parameters()}
    multipliers = completedp_multipliers(model, completedp)
    for name, group in by_name.items():
        assert name.startswith("module.")
        assert group["lr"] == pytest.approx(1e-3 * multipliers[name]["lr"])
        assert group["eps"] == pytest.approx(1e-8 * multipliers[name]["eps"])
        assert group["betas"] == pytest.approx((0.6, 0.8))
    assert by_name["module.embeddings.weight"]["weight_decay"] == 0.0
    w2 = "module.blocks.0.feed_forward.w2.weight"
    assert by_name[w2]["weight_decay"] == pytest.approx(0.1 * multipliers[w2]["weight_decay"])


def test_build_groups_requires_completedp_for_completedp_model():
    model = TransformerConfig.llama_like(
        d_model=128,
        vocab_size=256,
        n_layers=2,
        n_heads=2,
        completedp=CompletedPConfig(base_width=64, base_depth=2),
    ).build(init_device="meta")
    with pytest.raises(OLMoConfigurationError, match="needs 'completedp' set"):
        OLMoDDPOptimizerConfig(lr=1e-3).build_groups([model])
