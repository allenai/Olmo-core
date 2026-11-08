from copy import deepcopy

import pytest
import safetensors.torch
import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from olmo_core.distributed.checkpoint import (
    UnshardStrategy,
    async_save_model_and_optim_state,
    get_optimizer_state_dict,
    load_keys,
    load_model_and_optim_state,
    merge_state_dicts,
    prune_state_dict,
    save_model_and_optim_state,
    save_state_dict,
    unshard_checkpoint,
)
from olmo_core.optim import SkipStepAdamW
from olmo_core.testing import BACKENDS, requires_multi_gpu, run_distributed_test
from olmo_core.utils import get_default_device


def run_save_and_load_torch_fsdp_model(dir, model_factory, model_data_factory, use_orig_params):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    fsdp_model = FSDP(model_factory().cuda(), use_orig_params=use_orig_params)
    optim = torch.optim.AdamW(fsdp_model.parameters())

    # Take a train step to initialize optimizer state.
    fsdp_model(model_data_factory().cuda()).sum().backward()
    optim.step()

    # Save checkpoint.
    save_model_and_optim_state(dir, fsdp_model, optim)
    dist.barrier()

    # Now create a new fsdp model and load that state.
    fsdp_model2 = FSDP(model_factory().cuda(), use_orig_params=use_orig_params)
    optim2 = torch.optim.AdamW(fsdp_model2.parameters())
    load_model_and_optim_state(dir, fsdp_model2, optim2)

    # Check model parameters.
    with FSDP.summon_full_params(fsdp_model, recurse=True), FSDP.summon_full_params(
        fsdp_model2, recurse=True
    ):
        torch.testing.assert_close(fsdp_model.state_dict(), fsdp_model2.state_dict())

    # Check optimizer state.
    for (p1_name, p1), (p2_name, p2) in zip(
        fsdp_model.named_parameters(), fsdp_model2.named_parameters()
    ):
        assert p1_name == p2_name
        torch.testing.assert_close(
            optim.state[p1],
            optim2.state[p2],
            msg=lambda m: f"State for '{p1_name}' does not match. {m}",
        )


@requires_multi_gpu
@pytest.mark.parametrize(
    "use_orig_params",
    (
        pytest.param(True, id="use_orig_params=True"),
        pytest.param(False, id="use_orig_params=False"),
    ),
)
def test_save_and_load_torch_fsdp_model(
    tmp_path,
    tiny_model_factory,
    tiny_model_data_factory,
    use_orig_params,
):
    run_distributed_test(
        run_save_and_load_torch_fsdp_model,
        backend="nccl",
        start_method="spawn",
        func_args=(
            tmp_path,
            tiny_model_factory,
            tiny_model_data_factory,
            use_orig_params,
        ),
    )


class FeedForward(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.dim = dim
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, dim)
        self.w3 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def run_save_and_load_tensor_parallel_model(dir, take_step_before_checkpoint, run_async):
    tp_mesh = init_device_mesh(get_default_device().type, (dist.get_world_size(),))

    feed_forward = FeedForward().to(get_default_device())

    # Save a checkpoint from the unsharded model.
    save_state_dict(dir / "unsharded", {"model": feed_forward.state_dict()})

    parallelize_module(
        feed_forward,
        tp_mesh,
        {
            # by default ColwiseParallel input layouts is replicated
            # and RowwiseParallel output layouts is replicated
            "w1": ColwiseParallel(),
            "w2": RowwiseParallel(),
            "w3": ColwiseParallel(),
        },
    )
    optim = torch.optim.AdamW(feed_forward.parameters())

    if take_step_before_checkpoint:
        # Take a forward and backward pass.
        feed_forward(
            torch.rand((2, feed_forward.dim), device=get_default_device())
        ).sum().backward()

        # Take an optimizer step.
        optim.step()
        optim.zero_grad(set_to_none=True)

    # Save checkpoint.
    if run_async:
        async_save_model_and_optim_state(dir / "sharded", feed_forward, optim).result()
    else:
        save_model_and_optim_state(dir / "sharded", feed_forward, optim)

    # Create another sharded model, load the checkpoint and make sure the state matches.
    feed_forward2 = FeedForward().to(get_default_device())
    parallelize_module(
        feed_forward2,
        tp_mesh,
        {
            # by default ColwiseParallel input layouts is replicated
            # and RowwiseParallel output layouts is replicated
            "w1": ColwiseParallel(),
            "w2": RowwiseParallel(),
            "w3": ColwiseParallel(),
        },
    )
    optim2 = torch.optim.AdamW(feed_forward2.parameters())
    load_model_and_optim_state(dir / "sharded", feed_forward2, optim2)
    torch.testing.assert_close(feed_forward.state_dict(), feed_forward2.state_dict())
    torch.testing.assert_close(optim.state_dict(), optim2.state_dict())

    # Now load the checkpoint with a different topology, in this case an unsharded model.
    unsharded_feed_forward = FeedForward().to(get_default_device())
    unsharded_optim = torch.optim.AdamW(unsharded_feed_forward.parameters())
    load_model_and_optim_state(dir / "sharded", unsharded_feed_forward, unsharded_optim)

    # Now make sure we can load the checkpoint saved from the original unsharded model into both models.
    load_model_and_optim_state(dir / "unsharded", unsharded_feed_forward)
    load_model_and_optim_state(dir / "unsharded", feed_forward2)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "take_step_before_checkpoint",
    [pytest.param(True, id="after-step"), pytest.param(False, id="pre-step")],
)
@pytest.mark.parametrize(
    "run_async",
    [pytest.param(True, id="async"), pytest.param(False, id="sync")],
)
def test_save_and_load_tensor_parallel_model(
    backend, tmp_path, take_step_before_checkpoint, run_async
):
    run_distributed_test(
        run_save_and_load_tensor_parallel_model,
        backend=backend,
        start_method="spawn",
        func_args=(tmp_path, take_step_before_checkpoint, run_async),
    )


def run_save_sharded_checkpoint(dir):
    tp_mesh = init_device_mesh(get_default_device().type, (dist.get_world_size(),))

    feed_forward = FeedForward().to(get_default_device())
    parallelize_module(
        feed_forward,
        tp_mesh,
        {
            # by default ColwiseParallel input layouts is replicated
            # and RowwiseParallel output layouts is replicated
            "w1": ColwiseParallel(),
            "w2": RowwiseParallel(),
            "w3": ColwiseParallel(),
        },
    )
    optim = torch.optim.AdamW(feed_forward.parameters())

    # Take a forward and backward pass.
    feed_forward(torch.rand((2, feed_forward.dim), device=get_default_device())).sum().backward()

    # Take an optimizer step.
    optim.step()
    optim.zero_grad(set_to_none=True)

    save_model_and_optim_state(dir, feed_forward, optim)


@pytest.mark.parametrize("backend", BACKENDS)
def test_unshard_checkpoint(backend, tmp_path):
    sharded_checkpoint_dir = tmp_path / "sharded"
    unsharded_checkpoint_dir = tmp_path / "unsharded"

    run_distributed_test(
        run_save_sharded_checkpoint,
        backend=backend,
        func_args=(sharded_checkpoint_dir,),
        start_method="spawn",
    )

    assert sharded_checkpoint_dir.is_dir()

    # Unshard with regular PyTorch format.
    model_path_pt, optim_path_pt = unshard_checkpoint(
        sharded_checkpoint_dir,
        unsharded_checkpoint_dir,
    )
    assert model_path_pt.is_file()
    assert model_path_pt.suffix == ".pt"
    assert optim_path_pt is not None
    assert optim_path_pt.is_file()
    assert optim_path_pt.suffix == ".pt"

    model_state_pt = torch.load(model_path_pt, map_location="cpu", weights_only=True)
    assert model_state_pt
    assert model_state_pt.keys() == {
        "w1.weight",
        "w1.bias",
        "w2.weight",
        "w2.bias",
        "w3.weight",
        "w3.bias",
    }
    assert model_state_pt["w1.weight"].shape == (16, 16)

    optim_state = torch.load(optim_path_pt, map_location="cpu", weights_only=False)
    assert optim_state
    assert optim_state.keys() == {"param_groups", "state"}
    assert optim_state["state"].keys() == {
        "w1.weight",
        "w1.bias",
        "w2.weight",
        "w2.bias",
        "w3.weight",
        "w3.bias",
    }
    assert optim_state["state"]["w1.weight"].keys() == {"step", "exp_avg", "exp_avg_sq"}
    assert optim_state["state"]["w1.weight"]["exp_avg"].shape == (16, 16)

    # Unshard model with safetensors format.
    model_path_st, optim_path_st = unshard_checkpoint(
        sharded_checkpoint_dir, unsharded_checkpoint_dir, optim=False, use_safetensors=True
    )
    assert model_path_st.is_file()
    assert model_path_st.suffix == ".safetensors"
    assert optim_path_st is None

    model_state_st = safetensors.torch.load_file(model_path_st)
    torch.testing.assert_close(model_state_pt, model_state_st)

    # Unshard model with safetensors format, one file per tensor.
    model_dir_st, optim_dir_st = unshard_checkpoint(
        sharded_checkpoint_dir,
        unsharded_checkpoint_dir / "one_file_per_tensor",
        optim=False,
        use_safetensors=True,
        unshard_strategy=UnshardStrategy.one_file_per_tensor(),
    )
    assert model_dir_st.is_dir()
    assert optim_dir_st is None

    combined_model_state = {}
    for path in model_dir_st.iterdir():
        assert path.suffix == ".safetensors"
        combined_model_state.update(safetensors.torch.load_file(path))
    torch.testing.assert_close(combined_model_state, model_state_st)

    # Unshard model with safetensors format, multiple tensors per file by size.
    model_dir_st, optim_dir_st = unshard_checkpoint(
        sharded_checkpoint_dir,
        unsharded_checkpoint_dir / "chunks",
        optim=False,
        use_safetensors=True,
        unshard_strategy=UnshardStrategy.chunks(1_000),
    )
    assert model_dir_st.is_dir()
    assert optim_dir_st is None

    combined_model_state = {}
    for path in model_dir_st.iterdir():
        assert path.suffix == ".safetensors"
        combined_model_state.update(safetensors.torch.load_file(path))
    torch.testing.assert_close(combined_model_state, model_state_st)

    # Try loading specific keys.
    tensors = list(load_keys(sharded_checkpoint_dir, ["model.w1.weight", "model.w2.bias"]))
    assert len(tensors) == 2
    assert tensors[0].shape == (16, 16)
    assert tensors[1].shape == (16,)


def run_load_checkpoint_with_missing_keys(dir):
    class FF1(nn.Module):
        def __init__(self, dim: int = 16):
            super().__init__()
            self.dim = dim
            self.w1 = nn.Linear(dim, dim)
            self.w2 = nn.Linear(dim, dim)
            self.w3 = nn.Linear(dim, dim)

        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

    class FF2(nn.Module):
        def __init__(self, dim: int = 16):
            super().__init__()
            self.dim = dim
            self.w1 = nn.Linear(dim, dim)
            self.w2 = nn.Linear(dim, dim)
            self.w3 = nn.Linear(dim, dim)
            self.w4 = nn.Linear(dim, dim)

        def forward(self, x):
            return self.w4(self.w2(F.silu(self.w1(x)) * self.w3(x)))

    ff1 = FF1()
    ff2 = FF2()

    save_model_and_optim_state(dir, ff1)

    # NOTE: this raises a `RuntimeError`, but for some reason catching a `RuntimeError` doesn't
    # work here so we catch `BaseException` instead.
    with pytest.raises(
        BaseException, match="Missing key in checkpoint state_dict: model.w4.weight."
    ):
        load_model_and_optim_state(dir, ff2)


def test_load_checkpoint_with_missing_keys(tmp_path):
    run_distributed_test(
        run_load_checkpoint_with_missing_keys,
        backend="gloo",
        func_args=(tmp_path,),
    )


def run_load_checkpoint_with_extra_keys(dir):
    class FF1(nn.Module):
        def __init__(self, dim: int = 16):
            super().__init__()
            self.dim = dim
            self.w1 = nn.Linear(dim, dim)
            self.w2 = nn.Linear(dim, dim)
            self.w3 = nn.Linear(dim, dim)

        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

    class FF2(nn.Module):
        def __init__(self, dim: int = 16):
            super().__init__()
            self.dim = dim
            self.w1 = nn.Linear(dim, dim)
            self.w3 = nn.Linear(dim, dim)

        def forward(self, x):
            return F.silu(self.w1(x)) * self.w3(x)

    ff1 = FF1()
    ff2 = FF2()

    save_model_and_optim_state(dir, ff1)

    # NOTE: this raises a `RuntimeError`, but for some reason catching a `RuntimeError` doesn't
    # work here so we catch `BaseException` instead.
    with pytest.raises(
        BaseException, match='Unexpected key(s) in state_dict: "w2.weight", "w2.bias".'
    ):
        load_model_and_optim_state(dir, ff2)


@pytest.mark.xfail(reason="current limitation with torch.distributed.checkpoint module")
def test_load_checkpoint_with_extra_keys(tmp_path):
    run_distributed_test(
        run_load_checkpoint_with_extra_keys,
        backend="gloo",
        func_args=(tmp_path,),
    )


def run_load_checkpoint_with_different_keys(dir, flatten_optim_state: bool):
    class FF1(nn.Module):
        def __init__(self, dim: int = 16):
            super().__init__()
            self.dim = dim
            self.w1 = nn.Linear(dim, dim, bias=False)
            self.w2 = nn.Linear(dim, dim, bias=False)
            self.w3 = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

    class FF2(nn.Module):
        def __init__(self, dim: int = 16):
            super().__init__()
            self.dim = dim
            self.w1 = nn.Linear(dim, dim, bias=False)
            self.fc_out = nn.Linear(dim, dim, bias=False)
            self.w3 = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            return self.fc_out(F.silu(self.w1(x)) * self.w3(x))

    ff1 = FF1()
    optim1 = torch.optim.AdamW(ff1.parameters())
    ff2 = FF2()
    optim2 = torch.optim.AdamW(ff2.parameters())

    save_model_and_optim_state(dir, ff1, optim1, flatten_optimizer_state=flatten_optim_state)
    load_model_and_optim_state(
        dir,
        ff2,
        optim2,
        key_mapping={"fc_out.weight": "w2.weight"},
        flatten_optimizer_state=flatten_optim_state,
    )


@pytest.mark.parametrize(
    "flatten_optim_state",
    [pytest.param(True, id="flat-optim"), pytest.param(False, id="nested-optim")],
)
def test_load_checkpoint_with_different_keys(tmp_path, flatten_optim_state):
    run_distributed_test(
        run_load_checkpoint_with_different_keys,
        backend="gloo",
        start_method="spawn",
        func_args=(tmp_path, flatten_optim_state),
    )


def test_prune_state_dict():
    state_dict = {
        "model": {
            "a": 1,
            "b": 2,
        },
        "optim": {
            "c": 1,
            "d": 2,
        },
    }
    pruned_keys = prune_state_dict(state_dict, {"model.a", "optim.c"})
    assert pruned_keys == {"model.b", "optim.d"}
    assert state_dict == {
        "model": {
            "a": 1,
        },
        "optim": {
            "c": 1,
        },
    }


def test_merge_state_dicts():
    state_dict = {
        "model": {
            "a": 1,
            "b": 2,
        },
        "optim": {
            "c": 1,
            "d": 2,
        },
    }
    merge_state_dicts(state_dict, {"model": {"e": 3}})
    assert state_dict["model"]["e"] == 3


OPTIM_FACTORIES = [
    pytest.param(lambda params: torch.optim.AdamW(params, lr=1e-2), id="adamw"),
    pytest.param(lambda params: SkipStepAdamW(params, lr=1e-2), id="skip-step-adamw"),
    pytest.param(
        lambda params: SkipStepAdamW(params, lr=1e-2, foreach=True), id="skip-step-adamw-foreach"
    ),
]


def _train_steps(model: nn.Module, optim: torch.optim.Optimizer, n: int):
    torch.manual_seed(1)
    for _ in range(n):
        optim.zero_grad(set_to_none=True)
        model(torch.randn(8, 16)).pow(2).sum().backward()
        optim.step()


def _iter_steps(optim_state_dict):
    if "state" in optim_state_dict:
        for param_state in optim_state_dict["state"].values():
            yield param_state["step"]
    else:
        for key, value in optim_state_dict.items():
            if key.startswith("state.") and key.endswith(".step"):
                yield value


@pytest.mark.parametrize("optim_factory", OPTIM_FACTORIES)
@pytest.mark.parametrize(
    "flatten_optim_state",
    [pytest.param(True, id="flat-optim"), pytest.param(False, id="nested-optim")],
)
def test_get_optimizer_state_dict_does_not_advance_fresh_optimizer(
    optim_factory, flatten_optim_state
):
    torch.manual_seed(0)
    model = nn.Linear(16, 16)
    optim = optim_factory(model.parameters())
    optim.param_groups[0]["lr"] = 3e-3

    optim_state_dict = get_optimizer_state_dict(
        model,
        optim,
        options=dist_cp_sd.StateDictOptions(flatten_optimizer_state_dict=flatten_optim_state),
    )

    for param_state in optim.state.values():
        assert float(param_state["step"]) == 0
    steps = list(_iter_steps(optim_state_dict))
    assert len(steps) == 2
    assert all(float(step) == 0 for step in steps)
    if flatten_optim_state:
        assert optim_state_dict["param_groups.weight.lr"] == 3e-3
    else:
        assert optim_state_dict["param_groups"][0]["lr"] == 3e-3


@pytest.mark.parametrize("optim_factory", OPTIM_FACTORIES)
def test_get_optimizer_state_dict_on_fresh_optimizer_keeps_trajectory(optim_factory):
    torch.manual_seed(0)
    model1 = nn.Linear(16, 16)
    model2 = deepcopy(model1)
    optim1 = optim_factory(model1.parameters())
    optim2 = optim_factory(model2.parameters())

    get_optimizer_state_dict(model1, optim1)
    _train_steps(model1, optim1, 3)
    _train_steps(model2, optim2, 3)

    torch.testing.assert_close(model1.state_dict(), model2.state_dict(), rtol=0, atol=0)
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert float(optim1.state[p1]["step"]) == float(optim2.state[p2]["step"]) == 3


@pytest.mark.parametrize("optim_factory", OPTIM_FACTORIES)
def test_get_optimizer_state_dict_leaves_initialized_optimizer_alone(optim_factory):
    torch.manual_seed(0)
    model = nn.Linear(16, 16)
    optim = optim_factory(model.parameters())
    _train_steps(model, optim, 1)

    optim_state_dict = get_optimizer_state_dict(model, optim)

    expected = dist_cp_sd.get_optimizer_state_dict(model, optim)
    torch.testing.assert_close(optim_state_dict["state"], expected["state"], rtol=0, atol=0)
    assert optim_state_dict["param_groups"] == expected["param_groups"]
    assert all(float(step) == 1 for step in _iter_steps(optim_state_dict))


class _GroupStepSGD(torch.optim.Optimizer):
    """
    Keeps a step count in each param group (like Dion) and an int step per parameter.
    """

    def __init__(self, params, lr: float = 1e-2):
        super().__init__(params, dict(lr=lr, step=0))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            group["step"] += 1
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                state["step"] = state.get("step", 0) + 1
                p.add_(p.grad, alpha=-group["lr"])


@pytest.mark.parametrize(
    "flatten_optim_state",
    [pytest.param(True, id="flat-optim"), pytest.param(False, id="nested-optim")],
)
def test_get_optimizer_state_dict_resets_group_steps_and_copied_state(
    monkeypatch, flatten_optim_state
):
    # Make torch return copies, as it does for sharded models, so the live optimizer and the
    # returned state dict have to be reset separately.
    torch_get_optimizer_state_dict = dist_cp_sd.get_optimizer_state_dict
    monkeypatch.setattr(
        dist_cp_sd,
        "get_optimizer_state_dict",
        lambda *args, **kwargs: deepcopy(torch_get_optimizer_state_dict(*args, **kwargs)),
    )

    model = nn.Linear(16, 16)
    optim = _GroupStepSGD(model.parameters())
    optim_state_dict = get_optimizer_state_dict(
        model,
        optim,
        options=dist_cp_sd.StateDictOptions(flatten_optimizer_state_dict=flatten_optim_state),
    )

    assert [group["step"] for group in optim.param_groups] == [0]
    assert [state["step"] for state in optim.state.values()] == [0, 0]
    if flatten_optim_state:
        steps = {k: v for k, v in optim_state_dict.items() if k.endswith(".step")}
        assert steps == {
            "state.weight.step": 0,
            "state.bias.step": 0,
            "param_groups.weight.step": 0,
            "param_groups.bias.step": 0,
        }
    else:
        assert [group["step"] for group in optim_state_dict["param_groups"]] == [0]
        assert list(_iter_steps(optim_state_dict)) == [0, 0]


def run_resume_from_step_zero_checkpoint(dir):
    torch.manual_seed(0)
    model1 = nn.Linear(16, 16)
    model2 = deepcopy(model1)
    model3 = deepcopy(model1)
    optim1 = torch.optim.AdamW(model1.parameters(), lr=1e-2)
    optim2 = torch.optim.AdamW(model2.parameters(), lr=1e-2)
    optim3 = torch.optim.AdamW(model3.parameters(), lr=1e-2)

    # A step-0 checkpoint, then continue training without reloading.
    save_model_and_optim_state(dir, model1, optim1)
    _train_steps(model1, optim1, 3)

    # Resume from that checkpoint into a fresh model and optimizer.
    load_model_and_optim_state(dir, model2, optim2)
    _train_steps(model2, optim2, 3)

    # Reference: never checkpointed.
    _train_steps(model3, optim3, 3)

    torch.testing.assert_close(model1.state_dict(), model3.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(model2.state_dict(), model3.state_dict(), rtol=0, atol=0)


def test_resume_from_step_zero_checkpoint(tmp_path):
    run_distributed_test(
        run_resume_from_step_zero_checkpoint,
        backend="gloo",
        start_method="spawn",
        func_args=(tmp_path,),
    )
