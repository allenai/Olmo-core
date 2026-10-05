import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from olmo_core.optim.config import OptimGroupOverride
from olmo_core.optim.moe_optimizer import OLMoDDPOptimizer, OLMoDDPOptimizerConfig
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
