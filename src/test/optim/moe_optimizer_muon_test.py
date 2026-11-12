"""
Tests for the Muon path of :class:`olmo_core.optim.moe_optimizer.OLMoDDPOptimizer`.

The distributed tests build a real optimizer on a 2-rank CPU (gloo) mesh and compare every
parameter after ``step()`` against a plain-tensor reference of the dion Muon / AdamW math,
covering both orthogonalization paths: local (shard holds whole matrices) and all-gather.
"""

import math
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from olmo_core.optim.adamw import foreach_adamw_step
from olmo_core.optim.config import OptimGroupOverride
from olmo_core.optim.moe_optimizer import OLMoDDPOptimizer, OLMoDDPOptimizerConfig
from olmo_core.optim.muon import MuonAdjustLRStrategy
from olmo_core.optim.muon_update import (
    is_muon_matrix_shape,
    muon_adjusted_lr,
    newton_schulz,
)
from olmo_core.testing.distributed import run_distributed_test

# ---------------------------------------------------------------------------
# Pure-tensor helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((512,), False),  # vector
        ((8, 512), True),  # dense matrix
        ((64, 1024, 512), True),  # batch of expert matrices
        ((512, 1, 4), False),  # depthwise conv weight
        ((3, 1), False),  # degenerate matrix
    ],
)
def test_is_muon_matrix_shape(shape, expected):
    assert is_muon_matrix_shape(shape) is expected


def test_muon_adjusted_lr():
    lr = 0.01
    assert muon_adjusted_lr(lr, (1024, 512), MuonAdjustLRStrategy.rms_norm) == pytest.approx(
        lr * 0.2 * math.sqrt(1024)
    )
    assert muon_adjusted_lr(lr, (1024, 512), MuonAdjustLRStrategy.spectral_norm) == pytest.approx(
        lr * math.sqrt(2.0)
    )
    assert muon_adjusted_lr(lr, (1024, 512), None) == lr
    # Only the trailing matrix dims matter for batched (expert) weights.
    assert muon_adjusted_lr(lr, (8, 1024, 512), MuonAdjustLRStrategy.rms_norm) == muon_adjusted_lr(
        lr, (1024, 512), MuonAdjustLRStrategy.rms_norm
    )


@pytest.mark.parametrize("shape", [(16, 32), (32, 16), (4, 16, 32)])
def test_newton_schulz_approximates_polar_factor(shape):
    # Rectangular Gaussian matrices are well conditioned, so five Newton-Schulz steps bring every
    # singular value close to 1 (a square Gaussian matrix has near-zero singular values, which
    # the quintic iteration cannot lift in five steps).
    torch.manual_seed(0)
    G = torch.randn(*shape)
    orth = newton_schulz(G)
    assert orth.shape == G.shape and orth.dtype == G.dtype

    # Singular values are driven towards 1 and the singular vectors match the exact polar factor.
    svals = torch.linalg.svdvals(orth)
    assert svals.min() > 0.6 and svals.max() < 1.3, svals
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    polar = U @ Vh
    cos = (orth * polar).sum(dim=(-2, -1)) / (orth.norm(dim=(-2, -1)) * polar.norm(dim=(-2, -1)))
    assert (cos > 0.9).all(), cos

    # Each matrix of a batch is orthogonalized independently of the others.
    if G.ndim == 3:
        for i in range(G.shape[0]):
            torch.testing.assert_close(newton_schulz(G[i]), orth[i], rtol=2e-2, atol=1e-3)


def test_select_muon_params_by_group_and_shape():
    groups = [
        {
            "use_muon": True,
            "named_params": {
                "attn.weight": nn.Parameter(torch.zeros(8, 8)),
                "experts.w": nn.Parameter(torch.zeros(4, 8, 8)),
                "norm.weight": nn.Parameter(torch.zeros(8)),
                "conv.weight": nn.Parameter(torch.zeros(8, 1, 4)),
                "frozen.weight": nn.Parameter(torch.zeros(8, 8), requires_grad=False),
            },
        },
        {
            "use_muon": False,
            "named_params": {"embed.weight": nn.Parameter(torch.zeros(8, 8))},
        },
    ]
    assert OLMoDDPOptimizer._select_muon_params(groups) == {"attn.weight", "experts.w"}


# ---------------------------------------------------------------------------
# Distributed step tests
# ---------------------------------------------------------------------------

LR = 0.02
WD = 0.1
MU = 0.9
BETAS = (0.9, 0.95)
EPS = 1e-8
MAX_GRAD_NORM = 1.0

# (name, shape, group) -- 'muon' group is the default (use_muon=True) group, 'adam' is an override.
PARAM_SPECS: List[Tuple[str, Tuple[int, ...], str]] = [
    ("attn.weight", (16, 24), "muon"),  # dense matrix: a 2-rank shard splits rows -> gather path
    ("experts.w", (4, 8, 6), "muon"),  # 4 expert matrices: 2 per rank -> local path
    ("norm.weight", (24,), "muon"),  # vector in a Muon group -> AdamW
    ("conv.weight", (24, 1, 4), "muon"),  # not matrix-shaped -> AdamW
    ("embed.weight", (10, 24), "adam"),  # explicit AdamW group, no weight decay
]
MUON_NAMES = ("attn.weight", "experts.w")  # ordered: ranks must issue collectives in step
SHARDED_NAMES = ("attn.weight", "experts.w", "embed.weight")


def _make_params(seed: int) -> Dict[str, nn.Parameter]:
    gen = torch.Generator().manual_seed(seed)
    return {
        name: nn.Parameter(torch.randn(*shape, generator=gen)) for name, shape, _ in PARAM_SPECS
    }


def _make_grads(seed: int) -> Dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {name: torch.randn(*shape, generator=gen) for name, shape, _ in PARAM_SPECS}


def _build_optimizer(params: Dict[str, nn.Parameter], mesh, sharded: bool) -> OLMoDDPOptimizer:
    """
    Build the optimizer the way ``OLMoDDPOptimizerConfig.build`` does (minus the train module),
    optionally re-laying out the larger parameters' states as 1D shards across the mesh.
    """
    config = OLMoDDPOptimizerConfig(
        lr=LR,
        betas=BETAS,
        eps=EPS,
        weight_decay=WD,
        max_grad_norm=MAX_GRAD_NORM,
        use_muon=True,
        muon_momentum=MU,
        muon_nesterov=True,
        check_nan_inf_grad=False,
        group_overrides=[
            OptimGroupOverride(
                params=["embed.weight"], opts={"use_muon": False, "weight_decay": 0.0}
            )
        ],
    )
    model = nn.Module()
    for name, param in params.items():
        module_name, attr = name.split(".")
        if not hasattr(model, module_name):
            model.add_module(module_name, nn.Module())
        getattr(model, module_name).register_parameter(attr, param)
    groups = config.build_groups([model])
    for group in groups:
        group["pg"] = "dp"
    kwargs = config.as_dict()
    for key in ("group_overrides", "compile", "fixed_fields"):
        kwargs.pop(key)
    optim = OLMoDDPOptimizer(
        groups,
        world_mesh={"dense": mesh, "moe": None},
        dp_group=mesh.get_group(),
        device=torch.device("cpu"),
        **kwargs,
    )
    if sharded:
        for name in SHARDED_NAMES:
            for suffix in optim._state_suffixes_for_param(name):
                if suffix == "step":
                    continue
                key = f"{name}.{suffix}"
                optim.states[key] = distribute_tensor(
                    optim.states[key].full_tensor().clone(), mesh, [Shard(0)]
                )
    return optim


def _reference_step(
    params: Dict[str, torch.Tensor],
    grads: Dict[str, torch.Tensor],
    muon_momentum: Dict[str, torch.Tensor],
    adam_state: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> None:
    """One step of the dion Muon / AdamW math on full tensors, in place."""
    total_norm = torch.sqrt(sum(g.float().norm() ** 2 for g in grads.values()))
    clip = torch.clamp(MAX_GRAD_NORM / (total_norm + 1e-6), max=1.0)
    grads = {name: g * clip for name, g in grads.items()}
    group_of = {name: group for name, _, group in PARAM_SPECS}
    for name, p in params.items():
        g = grads[name]
        wd = 0.0 if group_of[name] == "adam" else WD
        if name in MUON_NAMES:
            m = muon_momentum[name]
            m.mul_(MU).add_(g)
            u = g + MU * m  # nesterov
            o = newton_schulz(u.view(-1, p.shape[-2], p.shape[-1])).view_as(p)
            p.mul_(1 - LR * wd)
            p.sub_(muon_adjusted_lr(LR, p.shape, MuonAdjustLRStrategy.rms_norm) * o)
        else:
            exp_avg, exp_avg_sq, step = adam_state[name]
            foreach_adamw_step(
                [p],
                [g],
                [exp_avg],
                [exp_avg_sq],
                [step],
                lr=LR,
                betas=BETAS,
                eps=EPS,
                weight_decay=wd,
                step_factor=torch.tensor(1.0),
            )


def _run_steps(sharded: bool, num_steps: int = 2):
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))
    params = _make_params(seed=1)
    optim = _build_optimizer(params, mesh, sharded=sharded)

    # Muon params carry a momentum buffer and no AdamW moments; everything else is AdamW.
    for name, _, _ in PARAM_SPECS:
        has_muon = f"{name}.muon_momentum" in optim.states
        has_adam = f"{name}.exp_avg" in optim.states and f"{name}.exp_avg_sq" in optim.states
        assert has_muon == (name in MUON_NAMES) and has_adam == (name not in MUON_NAMES), name

    ref_params = {name: p.detach().clone() for name, p in params.items()}
    ref_momentum = {name: torch.zeros_like(ref_params[name]) for name in MUON_NAMES}
    ref_adam = {
        name: (torch.zeros_like(p), torch.zeros_like(p), torch.zeros(()))
        for name, p in ref_params.items()
        if name not in MUON_NAMES
    }
    for step_idx in range(num_steps):
        grads = _make_grads(seed=100 + step_idx)  # identical on every rank, like all-reduced grads
        for name, param in params.items():
            param.grad = grads[name].clone()
        optim.latest_loss = torch.tensor(1.0)
        optim.step()
        _reference_step(ref_params, grads, ref_momentum, ref_adam)

        for name, param in params.items():
            torch.testing.assert_close(param.detach(), ref_params[name], rtol=1e-5, atol=1e-6)
        for name in MUON_NAMES:
            momentum = optim.states[f"{name}.muon_momentum"].full_tensor().view_as(ref_params[name])
            torch.testing.assert_close(momentum, ref_momentum[name], rtol=1e-5, atol=1e-6)
            assert optim.states[f"{name}.step"].full_tensor().item() == step_idx + 1


@pytest.mark.parametrize("sharded", [False, True])
def test_muon_step_matches_reference(sharded: bool):
    run_distributed_test(
        _run_steps, world_size=2, backend="gloo", start_method="spawn", func_args=(sharded,)
    )


def _run_skipped_step():
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))
    params = _make_params(seed=1)
    optim = _build_optimizer(params, mesh, sharded=True)
    optim.get_step_factor = lambda: torch.tensor(0.0)  # type: ignore[method-assign]
    before = {name: p.detach().clone() for name, p in params.items()}
    for name, param in params.items():
        param.grad = torch.randn_like(param)
    optim.step()
    for name, param in params.items():
        torch.testing.assert_close(param.detach(), before[name], rtol=0, atol=0)
    for name in MUON_NAMES:
        assert optim.states[f"{name}.muon_momentum"].full_tensor().abs().sum() == 0
        assert optim.states[f"{name}.step"].full_tensor().item() == 0


def test_skipped_step_leaves_muon_state_untouched():
    run_distributed_test(_run_skipped_step, world_size=2, backend="gloo", start_method="spawn")


def _run_state_dict_roundtrip():
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))
    params = _make_params(seed=1)
    optim = _build_optimizer(params, mesh, sharded=True)
    for name, param in params.items():
        param.grad = _make_grads(seed=7)[name]
    optim.latest_loss = torch.tensor(1.0)
    optim.step()

    sd = optim.state_dict()
    assert "attn.weight.muon_momentum" in sd and "attn.weight.exp_avg" not in sd
    assert "norm.weight.exp_avg" in sd and "norm.weight.muon_momentum" not in sd

    # A fresh optimizer restores the exact states.
    fresh = _build_optimizer(_make_params(seed=2), mesh, sharded=True)
    fresh.load_state_dict({k: v for k, v in sd.items()})
    for key, state in optim.states.items():
        torch.testing.assert_close(
            fresh.states[key].full_tensor(), state.full_tensor(), rtol=0, atol=0
        )

    # A checkpoint without the Muon momentum (e.g. written by AdamW) zero-initializes it.
    partial = {k: v for k, v in sd.items() if not k.endswith(".muon_momentum")}
    fresh.load_state_dict(partial)
    for name in MUON_NAMES:
        assert fresh.states[f"{name}.muon_momentum"].full_tensor().abs().sum() == 0
        torch.testing.assert_close(
            fresh.states[f"{name}.main"].full_tensor(),
            optim.states[f"{name}.main"].full_tensor(),
            rtol=0,
            atol=0,
        )

    # reset_optimizer_moments_on_load covers the Muon momentum too.
    assert "muon_momentum" in OLMoDDPOptimizer.MOMENT_STATE_SUFFIXES
    fresh.load_state_dict({k: v for k, v in sd.items()}, reset_optimizer_moments_on_load=True)
    for name in MUON_NAMES:
        assert fresh.states[f"{name}.muon_momentum"].full_tensor().abs().sum() == 0


def test_muon_state_dict_roundtrip():
    run_distributed_test(
        _run_state_dict_roundtrip, world_size=2, backend="gloo", start_method="spawn"
    )


# ---------------------------------------------------------------------------
# Lockstep comparison against the dion Muon optimizer
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Embedding -> dense matrix -> batch of expert matrices -> vector gain -> LM head."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 16)
        self.attn = nn.Linear(16, 16, bias=False)
        self.experts = nn.Parameter(torch.randn(4, 16, 16) * 0.1)
        self.norm = nn.Parameter(torch.ones(16))
        self.head = nn.Linear(16, 32, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = x + torch.tanh(self.attn(x))
        x = x + torch.tanh(torch.einsum("bsd,edh->ebsh", x, self.experts)).mean(dim=0)
        logits = self.head(x * self.norm)
        return nn.functional.cross_entropy(logits.flatten(0, 1), tokens.flatten())


def _run_lockstep_vs_dion(num_steps: int = 6):
    dion = pytest.importorskip("dion")
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))
    torch.manual_seed(0)
    model = _TinyModel()
    muon_names = ("attn.weight", "experts")
    adam_names = ("embed.weight", "norm", "head.weight")
    params = dict(model.named_parameters())

    # Ours: Muon for the matrix params, AdamW (no decay) for the rest, no grad clipping.
    config = OLMoDDPOptimizerConfig(
        lr=LR,
        betas=BETAS,
        eps=EPS,
        weight_decay=WD,
        max_grad_norm=1e9,
        use_muon=True,
        muon_momentum=MU,
        muon_nesterov=True,
        check_nan_inf_grad=False,
        group_overrides=[
            OptimGroupOverride(
                params=list(adam_names), opts={"use_muon": False, "weight_decay": 0.0}
            )
        ],
    )
    groups = config.build_groups([model])
    for group in groups:
        group["pg"] = "dp"
    kwargs = config.as_dict()
    for key in ("group_overrides", "compile", "fixed_fields"):
        kwargs.pop(key)
    ours = OLMoDDPOptimizer(
        groups, world_mesh={"dense": mesh, "moe": None}, dp_group=mesh.get_group(), **kwargs
    )
    for name in muon_names:
        for suffix in ours._state_suffixes_for_param(name):
            if suffix != "step":
                key = f"{name}.{suffix}"
                ours.states[key] = distribute_tensor(
                    ours.states[key].full_tensor().clone(), mesh, [Shard(0)]
                )
    assert ours._muon_param_names == set(muon_names)

    # dion: same hyperparameters, our Newton-Schulz kernel, so only the surrounding update logic
    # (momentum, Nesterov, weight decay, LR adjustment, sharding) is under test here.
    ref_params = {name: p.detach().clone().requires_grad_(True) for name, p in params.items()}
    theirs = dion.Muon(
        [
            {"params": [ref_params[n] for n in muon_names], "algorithm": "muon"},
            {
                "params": [ref_params[n] for n in adam_names],
                "algorithm": "adamw",
                "weight_decay": 0.0,
            },
        ],
        lr=LR,
        mu=MU,
        betas=BETAS,
        weight_decay=WD,
        epsilon=EPS,
        nesterov=True,
        adjust_lr="rms_norm",
        use_polar_express=False,
        newton_schulz_func=lambda X, epsilon: newton_schulz(X, eps=epsilon),
    )

    gen = torch.Generator().manual_seed(1)
    for _ in range(num_steps):
        tokens = torch.randint(0, 32, (2, 8), generator=gen)
        model.zero_grad(set_to_none=True)
        model(tokens).backward()
        for name, p in params.items():
            ref_params[name].grad = p.grad.detach().clone()
        ours.latest_loss = torch.tensor(1.0)
        ours.step()
        theirs.step()
        for name in muon_names:
            torch.testing.assert_close(
                params[name].detach(), ref_params[name].detach(), rtol=1e-4, atol=1e-4
            )


def test_muon_matches_dion_lockstep():
    run_distributed_test(_run_lockstep_vs_dion, world_size=2, backend="gloo", start_method="spawn")
