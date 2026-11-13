import copy
from typing import Any

import pytest
import torch

from olmo_core.nn import OutputDiscardCheckpoint
from olmo_core.nn import output_discard_checkpoint as odc_module
from olmo_core.nn.feed_forward import FeedForward


def test_output_discard_checkpoint_discards_and_restores_storage():
    torch.manual_seed(11)
    ckpt = OutputDiscardCheckpoint()

    submodule = torch.nn.Linear(512, 512)
    x = torch.randn(8, 512, requires_grad=True)

    y = ckpt.checkpoint(submodule, x)
    assert isinstance(y, torch.Tensor)

    # Make a grad-producing tensor that does not require reading y in backward.
    # This exercises hook-triggered recompute + storage restoration without
    # relying on full graph stitching.
    add_bias = torch.nn.Parameter(torch.randn_like(y))
    z = y + add_bias

    ckpt.discard_output_and_register_recompute(z)
    assert ckpt.outputs is not None
    assert ckpt.outputs[0].untyped_storage().nbytes() == 0

    loss = z.square().sum()
    loss.backward()

    assert y.untyped_storage().nbytes() > 0
    assert ckpt.outputs is None
    assert ckpt._ctx is None
    assert add_bias.grad is not None


@pytest.mark.parametrize("case", ["linear-chain", "3d-view", "python-fallback"])
def test_output_discard_checkpoint_linear_backward(case, monkeypatch):
    """Discarded outputs retain gradient parity, including saved views and Python restoration."""
    if case == "3d-view":
        # Linear saves a 2D view of its 3D input. Restoration must mutate the existing
        # StorageImpl in place so that the saved view sees the restored bytes.
        torch.manual_seed(3)
        submodule_ref = torch.nn.Sequential(
            torch.nn.Linear(64, 256, bias=False),
            torch.nn.SiLU(),
        )
        next_layer_ref = torch.nn.Linear(256, 64, bias=False)
        x_ref = torch.randn(2, 8, 64, requires_grad=True)
    else:
        if case == "python-fallback":
            # Exercise the fallback even when a working C++ toolchain is available.
            monkeypatch.setattr(odc_module._SHARED_STORAGE_LOADER, "_load", lambda: None)
            torch.manual_seed(13)
            batch_size, width = 4, 256
        else:
            torch.manual_seed(7)
            batch_size, width = 8, 512
        submodule_ref = torch.nn.Sequential(
            torch.nn.Linear(width, 2 * width),
            torch.nn.GELU(),
            torch.nn.Linear(2 * width, width),
        )
        next_layer_ref = torch.nn.Linear(width, width // 2)
        x_ref = torch.randn(batch_size, width, requires_grad=True)

    submodule_ckpt = copy.deepcopy(submodule_ref)
    next_layer_ckpt = copy.deepcopy(next_layer_ref)
    x_ckpt = x_ref.detach().clone().requires_grad_(True)

    z_ref = next_layer_ref(submodule_ref(x_ref))
    z_ref.square().mean().backward()

    ckpt = OutputDiscardCheckpoint()
    y_ckpt = ckpt.checkpoint(submodule_ckpt, x_ckpt)
    z_ckpt = next_layer_ckpt(y_ckpt)
    torch.testing.assert_close(z_ckpt, z_ref)
    ckpt.discard_output_and_register_recompute(z_ckpt)
    assert y_ckpt.untyped_storage().nbytes() == 0
    z_ckpt.square().mean().backward()

    assert y_ckpt.untyped_storage().nbytes() > 0
    assert x_ref.grad is not None
    assert x_ckpt.grad is not None
    torch.testing.assert_close(x_ckpt.grad, x_ref.grad, atol=1e-6, rtol=1e-6)
    for module_ref, module_ckpt in (
        (submodule_ref, submodule_ckpt),
        (next_layer_ref, next_layer_ckpt),
    ):
        for p_ref, p_ckpt in zip(module_ref.parameters(), module_ckpt.parameters()):
            assert p_ref.grad is not None
            assert p_ckpt.grad is not None
            torch.testing.assert_close(p_ckpt.grad, p_ref.grad, atol=1e-6, rtol=1e-6)


def test_share_storage_cpp_extension_rebinds_in_place():
    """
    Build the C++ ``share_storage`` extension and verify it rebinds ``dst``'s
    bytes to ``src``'s storage by mutating ``dst``'s existing ``StorageImpl`` in
    place: a view taken before the call sees the new data, and ``dst`` ends up
    sharing ``src``'s underlying buffer (distinct from the copy-based fallback).

    Skipped on machines without a working C++ toolchain (the build returns
    ``None``); the fallback path is covered separately.
    """
    loader = odc_module._SharedStorageLoader()
    share_storage = loader._load()
    if share_storage is None:
        pytest.skip(f"C++ share_storage extension unavailable: {loader._build_error!r}")
    assert share_storage is not None  # narrow for mypy past pytest.skip

    dst = torch.arange(8, dtype=torch.float32)
    dst_view = dst.view(2, 4)  # shares dst's StorageImpl
    src = torch.full((8,), 5.0, dtype=torch.float32)

    # Simulate the discard: free dst's bytes (the view shares the same storage).
    dst.untyped_storage().resize_(0)
    assert dst_view.untyped_storage().nbytes() == 0

    share_storage(dst, src)

    torch.testing.assert_close(dst, src)
    # The view, taken before the rebind, sees the new data through the shared
    # StorageImpl -- the property the Python fallback was fixed to match.
    torch.testing.assert_close(dst_view, src.view(2, 4))
    # C++ path shares src's buffer rather than copying into a fresh one.
    assert dst.untyped_storage().data_ptr() == src.untyped_storage().data_ptr()


class _ODCFeedForward(FeedForward):
    """
    SwiGLU :class:`FeedForward` with the fat ``activation(w1(x)) * w3(x)``
    intermediate -- the one ``w2`` saves for backward -- wrapped by
    :class:`OutputDiscardCheckpoint`. Drop-in replacement for the baseline.
    """

    def _gated(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation_fn(self.w1(x)) * self.w3(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and x.requires_grad:
            ckpt = OutputDiscardCheckpoint()
            h = ckpt.checkpoint(self._gated, x)
            y = self.w2(h)
            ckpt.discard_output_and_register_recompute(y)
            return y
        return self.w2(self._gated(x))


@pytest.mark.parametrize("n_layers", [1, 4])
def test_output_discard_checkpoint_ffn_integration(n_layers):
    """
    Wire :class:`OutputDiscardCheckpoint` into the real OLMo SwiGLU
    :class:`FeedForward` and assert output + gradient parity vs an unmodified
    stack. ``n_layers > 1`` exercises per-block ODC instances and verifies the
    recompute hooks fire in the correct order during a chained backward.
    """
    ffn_kwargs: dict[str, Any] = {
        "d_model": 64,
        "hidden_size": 128,
        "bias": False,
        "init_device": "cpu",
    }

    torch.manual_seed(0)
    baseline = torch.nn.Sequential(*[FeedForward(**ffn_kwargs) for _ in range(n_layers)])
    odc = torch.nn.Sequential(*[_ODCFeedForward(**ffn_kwargs) for _ in range(n_layers)])
    odc.load_state_dict(copy.deepcopy(baseline.state_dict()))

    torch.manual_seed(1)
    x_base = torch.randn(2, 8, 64, requires_grad=True)
    x_odc = x_base.detach().clone().requires_grad_(True)

    baseline(x_base).square().mean().backward()
    y_odc = odc(x_odc)
    # Storage of every block's fat intermediate is freed before backward.
    y_odc.square().mean().backward()

    assert x_base.grad is not None and x_odc.grad is not None
    torch.testing.assert_close(x_odc.grad, x_base.grad, atol=1e-5, rtol=1e-5)
    for p_base, p_odc in zip(baseline.parameters(), odc.parameters()):
        assert p_base.grad is not None and p_odc.grad is not None
        torch.testing.assert_close(p_odc.grad, p_base.grad, atol=1e-5, rtol=1e-5)
