"""
Correctness tests for Ulysses context parallelism (CP) in the recurrent sequence mixers,
:class:`~olmo_core.nn.attention.recurrent.GatedDeltaNet` and
:class:`~olmo_core.nn.attention.kda.KimiDeltaAttention`.

Each rank holds a contiguous slice of the sequence for the per-token projections; an all-to-all
switches to head parallelism so the causal convolutions and the recurrent kernel see the full
sequence for ``heads / cp_degree`` heads, and a second all-to-all switches back. The parity test
runs a single-process reference on the full sequence and checks, on every CP rank, that the
activations at each stage boundary and at the output match the reference's slice, that
parameters sliced per rank carry zero gradient outside the rank's slice, and that gradients
summed over the CP group equal the full-sequence gradients.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

import olmo_core.nn.attention.kda as kda_module
import olmo_core.nn.attention.recurrent as recurrent_module
from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.nn.attention.kda import KimiDeltaAttention
from olmo_core.nn.attention.recurrent import GatedDeltaNet
from olmo_core.nn.attention.ring import UlyssesContextParallelStyle
from olmo_core.nn.transformer.init import InitMethod
from olmo_core.testing import run_distributed_test
from olmo_core.testing.utils import requires_fla, requires_multi_gpu
from olmo_core.utils import get_default_device, seed_all

D_MODEL = 128
N_HEADS = 8


@dataclass
class Tolerance:
    """
    ``rel_fro`` bounds the relative Frobenius error; ``rtol``/``atol`` bound the max abs error
    relative to the largest magnitude in the expected tensor.
    """

    rel_fro: float
    rtol: float
    atol: float


# Forward activations are bf16 under autocast; gradients are accumulated in fp32 but flow through
# bf16 activations, and the CP path sums per-rank contributions in a different order.
FWD_TOL = Tolerance(rel_fro=2e-2, rtol=5e-2, atol=1e-2)
GRAD_TOL = Tolerance(rel_fro=3e-2, rtol=5e-2, atol=1e-4)


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, tol: Tolerance):
    actual = actual.detach().float()
    expected = expected.detach().float().to(actual.device)
    assert actual.shape == expected.shape, f"{name}: shape {actual.shape} != {expected.shape}"
    diff = actual - expected
    rel_fro = (diff.norm() / expected.norm().clamp_min(1e-6)).item()
    max_abs = diff.abs().max().item()
    scale = expected.abs().max().clamp_min(1e-6).item()
    assert rel_fro <= tol.rel_fro and max_abs <= tol.atol + tol.rtol * scale, (
        f"{name}: rel_fro={rel_fro:.3e} (tol {tol.rel_fro:.1e}), "
        f"max_abs={max_abs:.3e} vs allowed {tol.atol + tol.rtol * scale:.3e} (scale {scale:.3e})"
    )


@dataclass
class MixerSpec:
    name: str
    kwargs: Dict[str, Any]

    def build(self, device: str, **overrides: Any):
        kwargs = {**self.kwargs, **overrides}
        if self.name == "gdn":
            return GatedDeltaNet(init_device=device, **kwargs)
        elif self.name == "kda":
            return KimiDeltaAttention(init_device=device, **kwargs)
        raise NotImplementedError(self.name)


MIXERS: Dict[str, MixerSpec] = {
    "gdn": MixerSpec("gdn", {"d_model": D_MODEL, "n_heads": N_HEADS}),
    "kda": MixerSpec("kda", {"d_model": D_MODEL, "n_heads": N_HEADS, "allow_neg_eigval": True}),
}


@dataclass
class Case:
    batch_size: int
    seq_len: int
    doc_lens: Optional[List[int]] = None
    """Document lengths that sum to ``seq_len``; packed inputs require ``batch_size == 1``."""
    overrides: Dict[str, Any] = field(default_factory=dict)
    mixers: tuple = ("gdn", "kda")

    def cu_doc_lens(self, device) -> Optional[torch.Tensor]:
        if self.doc_lens is None:
            return None
        assert self.batch_size == 1 and sum(self.doc_lens) == self.seq_len
        return torch.tensor(
            [0, *torch.tensor(self.doc_lens).cumsum(0).tolist()], dtype=torch.int32, device=device
        )


CASES: Dict[str, Case] = {
    # Two instances, no document boundaries.
    "batch2": Case(batch_size=2, seq_len=128),
    # Packed documents whose boundaries (100, 156) do not line up with the CP split at
    # 128 (degree 2) or 64/128/192 (degree 4), so a document straddles ranks.
    "docs": Case(batch_size=1, seq_len=256, doc_lens=[100, 56, 100]),
    # Grouped value heads: q/k heads are repeated onto v heads after the exchange.
    "gva": Case(batch_size=1, seq_len=128, overrides={"n_v_heads": 2 * N_HEADS}, mixers=("gdn",)),
}


def _init_real_weights(module, seed: int = 777):
    """Real init so decays are in the strong regime; ``torch.empty`` leaves A_log as garbage."""
    generator = torch.Generator(device=module.A_log.device).manual_seed(seed)
    with torch.no_grad():
        module.init_weights(
            init_method=InitMethod.normal,
            d_model=module.d_model,
            block_idx=0,
            num_blocks=2,
            generator=generator,
        )


def _expected_param_slices(module, cp_rank: int, cp_world_size: int) -> Dict[str, slice]:
    """
    The dim-0 slice of every parameter that a rank should touch under Ulysses CP. Anything
    outside these slices must receive zero gradient on that rank.
    """
    slices: Dict[str, slice] = {}
    for conv_name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        conv = getattr(module, conv_name)
        local = conv.hidden_size // cp_world_size
        conv_slice = slice(cp_rank * local, (cp_rank + 1) * local)
        slices[f"{conv_name}.weight"] = conv_slice
        if conv.bias is not None:
            slices[f"{conv_name}.bias"] = conv_slice
    if isinstance(module, KimiDeltaAttention):
        local_heads = module.n_heads // cp_world_size
        slices["A_log"] = slice(cp_rank * local_heads, (cp_rank + 1) * local_heads)
        local_gate = module.gate_dim // cp_world_size
        slices["dt_bias"] = slice(cp_rank * local_gate, (cp_rank + 1) * local_gate)
    return slices


# Dimension along which each recorded stage tensor is partitioned across CP ranks after the
# exchange: channels for the conv tensors, heads for the kernel tensors, and the parameter axis
# for the kernel's per-head / per-channel gate parameters.
_STAGE_SPLIT_DIM = {
    "q_conv1d.in": -1,
    "q_conv1d.out": -1,
    "k_conv1d.in": -1,
    "k_conv1d.out": -1,
    "v_conv1d.in": -1,
    "v_conv1d.out": -1,
    "kernel.q": 2,
    "kernel.k": 2,
    "kernel.v": 2,
    "kernel.g": 2,
    "kernel.beta": 2,
    "kernel.out": 2,
    "kernel.A_log": 0,
    "kernel.dt_bias": 0,
}


class _StageRecorder:
    """
    Captures the tensors at every stage boundary inside a mixer's forward pass: the inputs and
    outputs of the three causal convolutions, and the inputs and output of the recurrent kernel.
    Under CP these are the post-exchange, head-parallel tensors, so each one should equal the
    reference's slice along :data:`_STAGE_SPLIT_DIM`.
    """

    _KERNEL_KEYS = ("q", "k", "v", "g", "beta", "A_log", "dt_bias")

    def __init__(self, module):
        self.module = module
        self.stages: Dict[str, torch.Tensor] = {}
        self._handles: List[Any] = []
        self._patch: Any = None

    def _record(self, name: str, value: Any):
        if isinstance(value, torch.Tensor):
            self.stages[name] = value.detach().float().cpu()

    def __enter__(self) -> "_StageRecorder":
        for conv_name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            conv = getattr(self.module, conv_name)
            self._handles.append(
                conv.register_forward_pre_hook(
                    lambda m, args, kwargs, n=conv_name: self._record(
                        f"{n}.in", kwargs.get("x", args[0] if args else None)
                    ),
                    with_kwargs=True,
                )
            )
            self._handles.append(
                conv.register_forward_hook(
                    lambda m, args, out, n=conv_name: self._record(f"{n}.out", out)
                )
            )

        target: types.ModuleType
        if isinstance(self.module, KimiDeltaAttention):
            target, fn_name = kda_module, "dispatch_chunk_kda"
        else:
            target, fn_name = recurrent_module, "dispatch_chunk_gated_delta_rule"
        original = getattr(target, fn_name)

        def wrapped(*args, **kwargs):
            for key in self._KERNEL_KEYS:
                if key in kwargs:
                    self._record(f"kernel.{key}", kwargs[key])
            out = original(*args, **kwargs)
            self._record("kernel.out", out[0])
            return out

        self._patch = patch.object(target, fn_name, wrapped)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        if self._patch is not None:
            self._patch.stop()


def _rank_slice(tensor: torch.Tensor, dim: int, rank: int, world_size: int) -> torch.Tensor:
    size = tensor.shape[dim]
    assert size % world_size == 0, (tensor.shape, dim, world_size)
    per_rank = size // world_size
    return tensor.narrow(dim, rank * per_rank, per_rank)


def _build_reference(
    mixer_name: str, case_name: str, ref_path: Path, checkpoint_dir: Path, device: torch.device
):
    """Run the full sequence in one process and save inputs, outputs, stages, and gradients."""
    spec, case = MIXERS[mixer_name], CASES[case_name]
    seed_all(0)
    module = spec.build(device.type, **case.overrides)
    _init_real_weights(module)

    x = torch.randn(
        case.batch_size, case.seq_len, D_MODEL, device=device, dtype=torch.bfloat16
    ).requires_grad_(True)
    # A fixed random weight makes the loss a plain sum over tokens, so it splits exactly across
    # the sequence shards.
    loss_weight = torch.randn(case.batch_size, case.seq_len, D_MODEL, device=device)
    cu_doc_lens = case.cu_doc_lens(device)

    with _StageRecorder(module) as recorder, torch.autocast(device.type, dtype=torch.bfloat16):
        y = module(x, cu_doc_lens=cu_doc_lens)
    (y.float() * loss_weight).sum().backward()
    assert x.grad is not None
    assert set(recorder.stages) >= {"q_conv1d.in", "v_conv1d.out", "kernel.q", "kernel.out"}

    grads = {}
    for name, param in module.named_parameters():
        assert param.grad is not None, f"reference produced no gradient for {name}"
        grads[name] = param.grad.detach().float().cpu()
    torch.save(
        {
            "x": x.detach().cpu(),
            "loss_weight": loss_weight.cpu(),
            "cu_doc_lens": None if cu_doc_lens is None else cu_doc_lens.cpu(),
            "y": y.detach().cpu(),
            "x_grad": x.grad.detach().cpu(),
            "grads": grads,
            "stages": recorder.stages,
        },
        ref_path,
    )
    save_model_and_optim_state(checkpoint_dir, module)


def _run_ulysses_cp_parity(checkpoint_dir: str, ref_path: str, mixer_name: str, case_name: str):
    spec, case = MIXERS[mixer_name], CASES[case_name]
    device = get_default_device()
    rank, world_size = get_rank(), get_world_size()
    mesh = init_device_mesh(device.type, (world_size,), mesh_dim_names=("cp",))
    cp_group = mesh["cp"].get_group()

    module = spec.build(device.type, **case.overrides)
    module.apply_cp(mesh["cp"], uly=UlyssesContextParallelStyle())
    load_model_and_optim_state(checkpoint_dir, module)

    ref = torch.load(ref_path, map_location=device)
    seq_len = ref["x"].size(1)
    assert seq_len % world_size == 0
    t_local = seq_len // world_size
    local = slice(rank * t_local, (rank + 1) * t_local)

    # Contiguous sequence shard, as the Ulysses load balancer produces. Document boundaries
    # stay global under Ulysses because the layer sees the full sequence after the exchange.
    x_local = ref["x"][:, local].clone().requires_grad_(True)
    cu_doc_lens = ref["cu_doc_lens"]

    with _StageRecorder(module) as recorder, torch.autocast(device.type, dtype=torch.bfloat16):
        y_local = module(x_local, cu_doc_lens=cu_doc_lens)
    assert y_local.shape == x_local.shape, y_local.shape

    # Every stage boundary inside the layer: the post-exchange tensor on this rank must equal
    # the reference's slice of heads / channels for this rank.
    assert set(recorder.stages) == set(ref["stages"]), (
        f"rank{rank} recorded stages {sorted(recorder.stages)} but reference has "
        f"{sorted(ref['stages'])}"
    )
    for stage_name, ref_tensor in ref["stages"].items():
        expected = _rank_slice(ref_tensor, _STAGE_SPLIT_DIM[stage_name], rank, world_size)
        _assert_close(
            f"rank{rank} stage {stage_name}", recorder.stages[stage_name], expected, FWD_TOL
        )

    _assert_close(f"rank{rank} output", y_local, ref["y"][:, local], FWD_TOL)

    (y_local.float() * ref["loss_weight"][:, local]).sum().backward()

    # Sliced parameters must only receive gradient inside this rank's slice.
    for name, param_slice in _expected_param_slices(module, rank, world_size).items():
        grad = module.get_parameter(name).grad
        assert grad is not None, name
        outside = torch.ones(grad.shape[0], dtype=torch.bool, device=grad.device)
        outside[param_slice] = False
        assert (grad[outside] == 0).all(), f"rank{rank} {name}: gradient leaked outside CP slice"
        assert (grad[param_slice] != 0).any(), f"rank{rank} {name}: no gradient inside CP slice"

    # Summed over the CP group, per-rank gradients must reproduce the full-sequence gradients.
    for name, param in module.named_parameters():
        assert param.grad is not None, f"rank{rank} produced no gradient for {name}"
        grad = param.grad.detach().float().clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=cp_group)
        _assert_close(f"grad {name}", grad, ref["grads"][name], GRAD_TOL)

    assert x_local.grad is not None
    _assert_close(f"rank{rank} x.grad", x_local.grad, ref["x_grad"][:, local], GRAD_TOL)


def _parity_params():
    return [
        pytest.param(mixer_name, case_name, id=f"{mixer_name}-{case_name}")
        for mixer_name in MIXERS
        for case_name, case in CASES.items()
        if mixer_name in case.mixers
    ]


@requires_multi_gpu
@requires_fla
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("mixer_name,case_name", _parity_params())
def test_ulysses_cp_forward_backward_parity(
    tmp_path, mixer_name: str, case_name: str, world_size: int
):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")

    ref_path = tmp_path / "reference.pt"
    checkpoint_dir = tmp_path / "checkpoint"
    _build_reference(mixer_name, case_name, ref_path, checkpoint_dir, torch.device("cuda"))

    run_distributed_test(
        _run_ulysses_cp_parity,
        world_size=world_size,
        backend="nccl",
        start_method="spawn",
        func_args=(str(checkpoint_dir), str(ref_path), mixer_name, case_name),
    )
