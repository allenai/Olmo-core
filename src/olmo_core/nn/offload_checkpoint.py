"""
Activation checkpointing that keeps each block's INPUT in pinned host memory instead of on the GPU.

With full activation checkpointing the only activations held across the forward pass are the block inputs: one
``(B, T, d)`` tensor per block, ``n_layers x T x d x 2`` bytes per row in bf16 (8 GiB per 32k tokens for a 7B with
d = 4096 and 32 blocks). At long context that stack, not the parameters or optimizer state, sets the maximum trainable
sequence length. This wrapper moves it to the host: the forward runs the block under ``no_grad`` and copies its input
to pinned CPU memory (non-blocking, stream-ordered); the backward copies it back, re-runs the block with grad and
backpropagates through it -- the same recompute as full checkpointing, so FLOPs and results are unchanged and only
PCIe traffic is added (one copy each way per block per step).

Torch's non-reentrant ``checkpoint_wrapper`` cannot do this: its recompute closure holds the GPU input alive, out of
reach of saved-tensor hooks. This is a reentrant-style :class:`torch.autograd.Function`, as torch's
``use_reentrant=True`` checkpoint, so FSDP2's per-module pre-backward (unshard) and post-backward (reduce-scatter)
hooks fire around it as they do for reentrant checkpointing. A zero-size dummy input that requires grad keeps the
backward firing when the block input itself does not require grad (frozen embeddings, e.g. a whole-model LoRA).

Parameter names are unchanged: the wrapper subclasses torch's ``ActivationWrapper`` and keeps its
``_checkpoint_wrapped_module`` prefix handling, exactly as ``checkpoint_wrapper``.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import ActivationWrapper

__all__ = ["OffloadInputCheckpointWrapper", "offload_input_checkpoint_wrapper"]


class _OffloadInputCheckpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, dummy: torch.Tensor):
        del dummy
        ctx.run_fn = run_fn
        ctx.device = x.device
        ctx.x_requires_grad = x.requires_grad
        if x.is_cuda:
            host = torch.empty(x.shape, dtype=x.dtype, device="cpu", pin_memory=True)
            host.copy_(x.detach(), non_blocking=True)
        else:
            host = x.detach().clone()
        ctx.host = host
        with torch.no_grad():
            out = run_fn(x)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x = ctx.host.to(ctx.device, non_blocking=True).detach().requires_grad_(True)
        ctx.host = None
        with torch.enable_grad():
            out = ctx.run_fn(x)
        torch.autograd.backward(out, grad_out)
        return None, (x.grad if ctx.x_requires_grad else None), None


class OffloadInputCheckpointWrapper(ActivationWrapper):
    """Full activation checkpointing of ``module`` with its input held on the host (see the module docstring).
    The wrapped module must be called as ``module(x, **kwargs)`` and return one tensor shaped like ``x``."""

    def __init__(self, mod: nn.Module):
        super().__init__(mod)
        self._dummy = None

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        mod = self._checkpoint_wrapped_module
        if not torch.is_grad_enabled():
            return mod(x, *args, **kwargs)

        def run_fn(inp: torch.Tensor) -> torch.Tensor:
            return mod(inp, *args, **kwargs)

        if self._dummy is None or self._dummy.device != x.device:
            self._dummy = torch.empty(0, device=x.device, requires_grad=True)
        return _OffloadInputCheckpoint.apply(run_fn, x, self._dummy)


def offload_input_checkpoint_wrapper(module: nn.Module) -> OffloadInputCheckpointWrapper:
    return OffloadInputCheckpointWrapper(module)
