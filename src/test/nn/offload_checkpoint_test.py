import pytest
import torch
import torch.nn as nn

from olmo_core.nn.offload_checkpoint import OffloadInputCheckpointWrapper
from olmo_core.testing import requires_gpu


class _Block(nn.Module):
    def __init__(self, d: int = 16):
        super().__init__()
        self.w1 = nn.Linear(d, 4 * d)
        self.w2 = nn.Linear(4 * d, d)

    def forward(self, x: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
        return x + scale * self.w2(torch.nn.functional.gelu(self.w1(x)))


def _grads(model: nn.Module):
    # named_parameters keeps the wrapper prefix (as torch's checkpoint_wrapper); state_dict strips it
    return {
        n.replace("_checkpoint_wrapped_module.", ""): p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }


def _run(device: str, input_requires_grad: bool):
    torch.manual_seed(0)
    ref = nn.ModuleList([_Block() for _ in range(3)]).to(device)
    off = nn.ModuleList([OffloadInputCheckpointWrapper(_Block()) for _ in range(3)]).to(device)
    off.load_state_dict(ref.state_dict())
    x = torch.randn(2, 5, 16, device=device)
    outs = []
    for blocks in (ref, off):
        h = x.clone().requires_grad_(input_requires_grad)
        inp = h
        for b in blocks:
            h = b(h, scale=0.5)
        h.square().mean().backward()
        outs.append((h.detach(), _grads(blocks), inp.grad))
    (h_ref, g_ref, xg_ref), (h_off, g_off, xg_off) = outs
    torch.testing.assert_close(h_off, h_ref)
    assert set(g_off) == set(g_ref) and len(g_ref) == 12
    for n in g_ref:
        torch.testing.assert_close(g_off[n], g_ref[n])
    if input_requires_grad:
        torch.testing.assert_close(xg_off, xg_ref)
    else:
        assert xg_ref is None and xg_off is None


@pytest.mark.parametrize("input_requires_grad", [True, False])
def test_offload_checkpoint_matches_plain_cpu(input_requires_grad):
    _run("cpu", input_requires_grad)


@requires_gpu
@pytest.mark.parametrize("input_requires_grad", [True, False])
def test_offload_checkpoint_matches_plain_gpu(input_requires_grad):
    _run("cuda", input_requires_grad)


def test_offload_checkpoint_keeps_parameter_names():
    ref = _Block()
    off = OffloadInputCheckpointWrapper(_Block())
    assert set(off.state_dict().keys()) == set(ref.state_dict().keys())


def test_offload_checkpoint_no_grad_path():
    off = OffloadInputCheckpointWrapper(_Block())
    with torch.no_grad():
        y = off(torch.randn(1, 3, 16))
    assert not y.requires_grad
