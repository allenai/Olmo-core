from copy import deepcopy

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from olmo_core.nn.moe.utils import run_on_stream_no_compile, wait_stream_no_compile
from olmo_core.testing import requires_gpu


@requires_gpu
def test_compiled_side_stream_backward_after_separate_compiled_loss():
    # The separate loss graph replaces Dynamo's stream registry. On Torch 2.13,
    # tracing the stream context itself makes the preceding graph's backward fail.
    torch.manual_seed(44)
    stream = torch.cuda.Stream()

    def compute(x, w):
        assert torch.compiler.is_compiling(), "Side-stream math must stay compiled"
        return torch.sin(x @ w)

    def forward(x, w):
        wait_stream_no_compile(stream, torch.cuda.current_stream())
        y = run_on_stream_no_compile(stream, compute, x, w)
        wait_stream_no_compile(torch.cuda.current_stream(), stream)
        return x + y

    compiled_forward = torch.compile(forward)
    compiled_loss = torch.compile(lambda x: x.square().mean())
    for _ in range(2):
        x = torch.randn(16, 32, device="cuda", requires_grad=True)
        w = torch.randn(32, 32, device="cuda", requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        wr = w.detach().clone().requires_grad_()
        ref = (xr + torch.sin(xr @ wr)).square().mean()
        ref.backward()
        loss = compiled_loss(compiled_forward(x, w))
        loss.backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(loss, ref, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(x.grad, xr.grad, rtol=1e-4, atol=2e-6)
        torch.testing.assert_close(w.grad, wr.grad, rtol=1e-4, atol=2e-6)


@requires_gpu
@pytest.mark.parametrize("recompute", [False, True])
def test_compiled_split_shared_experts_backward(recompute):
    from test.nn.ddp.block_no_sync_test import _build_block, _init_block_params

    from olmo_core.nn.moe.v2.ep_no_sync_rowwise import (
        _shared_forward1,
        _shared_forward2,
    )

    block = _build_block(
        ep_no_sync=True,
        d_model=128,
        hidden_size=256,
        num_shared_experts=1,
        shared_hidden_size=256,
        init_device="cuda",
    )
    _init_block_params(block)
    block.purge_cuda_events()
    reference = deepcopy(block)
    block.install_cuda_events()
    reference.install_cuda_events()
    assert block.shared_experts is not None and reference.shared_experts is not None
    stream = block.get_dense_stream()

    def forward(x):
        wait_stream_no_compile(stream, torch.cuda.current_stream())
        up, gate = run_on_stream_no_compile(
            stream, _shared_forward1, block, x, use_rowwise_fp8=False
        )
        routed = x.sin()
        wait_stream_no_compile(stream, torch.cuda.current_stream())
        shared = run_on_stream_no_compile(
            stream, _shared_forward2, block, up, gate, None, x.shape, use_rowwise_fp8=False
        )
        wait_stream_no_compile(torch.cuda.current_stream(), stream)
        return routed + shared

    compiled = torch.compile(forward)
    compiled_loss = torch.compile(lambda y: y.square().mean())
    for _ in range(2):
        block.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        x = torch.randn(1, 16, 128, device="cuda", requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        expected = xr.sin() + reference.shared_experts(xr).squeeze(0)
        expected.square().mean().backward()
        actual = checkpoint(compiled, x, use_reentrant=False) if recompute else compiled(x)
        compiled_loss(actual).backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-6)
        torch.testing.assert_close(x.grad, xr.grad, rtol=1e-4, atol=2e-6)
        for param, ref in zip(
            block.shared_experts.parameters(), reference.shared_experts.parameters()
        ):
            torch.testing.assert_close(param.grad, ref.grad, rtol=1e-4, atol=2e-6)
