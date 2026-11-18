"""Ulysses query scaling preserves global document positions and gradients."""

from types import SimpleNamespace

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import Attention, AttentionBackendName
from olmo_core.nn.attention.ring import UlyssesContextParallelStyle


@pytest.mark.parametrize("cp", [2, 4])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_ulysses_scaling_and_gradients_match_full_sequence(cp, packed, dtype, monkeypatch):
    torch.manual_seed(93)
    batch, length, heads, width = 2, 32, 4, 4
    reference = Attention(
        d_model=heads * width,
        n_heads=heads,
        scalable_softmax=True,
        backend=AttentionBackendName.torch,
        dtype=dtype,
    )
    assert reference.ssmax_scale is not None
    with torch.no_grad():
        reference.ssmax_scale.copy_(torch.tensor([0.5, 0.8, 1.2, 2.0], dtype=dtype))
    q = torch.randn(batch, length, heads, width, dtype=dtype, requires_grad=True)
    boundaries = torch.tensor([0, 5, 19, 32, 45, 64], dtype=torch.int32) if packed else None
    weights = torch.randn_like(q)
    expected = reference._apply_scalable_softmax(q, boundaries)
    (expected * weights).sum().backward()
    local_length = length // cp
    scale_grads = []
    for rank in range(cp):
        attention = Attention(
            d_model=heads * width,
            n_heads=heads,
            scalable_softmax=True,
            backend=AttentionBackendName.torch,
            dtype=dtype,
        )
        attention.load_state_dict(reference.state_dict())
        mesh = SimpleNamespace(get_local_rank=lambda: rank, size=lambda: cp)

        def enable_cp(*args, **kwargs):
            attention.backend.cp_enabled = True
            attention.backend.uly = kwargs["uly"]

        # This unit test isolates query scaling; GPU integration tests cover
        # the backend's actual all-to-all and full model forward/backward.
        monkeypatch.setattr(attention.backend, "apply_cp", enable_cp)
        attention.apply_cp(mesh, uly=UlyssesContextParallelStyle())
        start, end = rank * local_length, (rank + 1) * local_length
        local = q.detach()[:, start:end].clone().requires_grad_()
        actual = attention._apply_scalable_softmax(local, boundaries)
        torch.testing.assert_close(actual, expected[:, start:end], rtol=0, atol=0)
        (actual * weights[:, start:end]).sum().backward()
        torch.testing.assert_close(local.grad, q.grad[:, start:end], rtol=0, atol=0)
        assert attention.ssmax_scale is not None
        scale_grads.append(attention.ssmax_scale.grad)
    rtol, atol = (0.02, 0.5) if dtype == torch.bfloat16 else (1e-12, 1e-12)
    torch.testing.assert_close(
        torch.stack(scale_grads).sum(0), reference.ssmax_scale.grad, rtol=rtol, atol=atol
    )


def test_scalable_softmax_still_rejects_ring_cp():
    attention = Attention(d_model=8, n_heads=2, scalable_softmax=True)
    with pytest.raises(OLMoConfigurationError, match="requires Ulysses"):
        attention.apply_cp(None, ring=object())
