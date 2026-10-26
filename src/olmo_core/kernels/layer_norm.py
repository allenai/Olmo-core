"""
LayerNorm with an affine weight and bias as Triton kernels, for any number of rows.

PyTorch's CUDA LayerNorm reduces the weight and bias gradients in a separate kernel that runs far
below memory bandwidth at hundreds of thousands of rows, and ``torch.compile`` only fixes that for
static shapes. The vision encoder's row count changes with every micro-batch's crop count, so the
kernels here are never specialized on it: the forward normalizes ``BLOCK_M`` rows per program, and
the backward runs a fixed number of programs per SM that each walk a strided set of row blocks,
write the input gradient and keep float32 partial sums of the weight and bias gradients, which one
``sum`` over the programs reduces (deterministically: no atomics).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton may be unavailable in some test envs.
    triton = None
    tl = None

__all__ = ["layer_norm"]


# Tuned on B300 for 179K x 1152 bf16 rows (the vision encoder's): forward 207 us, backward 698 us,
# against 448 us and 3,034 us for PyTorch's LayerNorm.
_FWD_BLOCK_M = 2
_FWD_NUM_WARPS = 4
_BWD_BLOCK_M = 4
_BWD_NUM_WARPS = 4
_BWD_PROGRAMS_PER_SM = 4
# Wider rows would need a looped reduction; they take PyTorch's LayerNorm instead.
_MAX_HIDDEN = 8192

_SM_COUNT: Dict[torch.device, int] = {}


if triton is not None:

    @triton.jit
    def _layer_norm_fwd_kernel(
        x_ptr,
        y_ptr,
        weight_ptr,
        bias_ptr,
        mean_ptr,
        rstd_ptr,
        rows,
        hidden,
        eps,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row_idx = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        col_idx = tl.arange(0, BLOCK_N)
        row_mask = row_idx < rows
        col_mask = col_idx < hidden
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = row_idx[:, None].to(tl.int64) * hidden + col_idx[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1) / hidden
        centered = tl.where(mask, x - mean[:, None], 0.0)
        rstd = tl.rsqrt(tl.sum(centered * centered, axis=1) / hidden + eps)
        weight = tl.load(weight_ptr + col_idx, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + col_idx, mask=col_mask, other=0.0).to(tl.float32)
        y = centered * rstd[:, None] * weight[None, :] + bias[None, :]
        tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=mask)
        tl.store(mean_ptr + row_idx, mean, mask=row_mask)
        tl.store(rstd_ptr + row_idx, rstd, mask=row_mask)

    @triton.jit
    def _layer_norm_bwd_kernel(
        x_ptr,
        dy_ptr,
        dx_ptr,
        weight_ptr,
        mean_ptr,
        rstd_ptr,
        dweight_partial_ptr,
        dbias_partial_ptr,
        rows,
        hidden,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_programs = tl.num_programs(0)
        col_idx = tl.arange(0, BLOCK_N)
        col_mask = col_idx < hidden
        weight = tl.load(weight_ptr + col_idx, mask=col_mask, other=0.0).to(tl.float32)
        dweight = tl.zeros([BLOCK_N], dtype=tl.float32)
        dbias = tl.zeros([BLOCK_N], dtype=tl.float32)
        for start in range(pid * BLOCK_M, rows, num_programs * BLOCK_M):
            row_idx = start + tl.arange(0, BLOCK_M)
            row_mask = row_idx < rows
            mask = row_mask[:, None] & col_mask[None, :]
            offsets = row_idx[:, None].to(tl.int64) * hidden + col_idx[None, :]
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            mean = tl.load(mean_ptr + row_idx, mask=row_mask, other=0.0)
            rstd = tl.load(rstd_ptr + row_idx, mask=row_mask, other=0.0)
            x_hat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
            weighted_dy = weight[None, :] * dy
            c1 = tl.sum(x_hat * weighted_dy, axis=1) / hidden
            c2 = tl.sum(weighted_dy, axis=1) / hidden
            dx = (weighted_dy - (x_hat * c1[:, None] + c2[:, None])) * rstd[:, None]
            tl.store(dx_ptr + offsets, dx.to(dx_ptr.dtype.element_ty), mask=mask)
            dweight += tl.sum(dy * x_hat, axis=0)
            dbias += tl.sum(dy, axis=0)
        tl.store(dweight_partial_ptr + pid * hidden + col_idx, dweight, mask=col_mask)
        tl.store(dbias_partial_ptr + pid * hidden + col_idx, dbias, mask=col_mask)


def _sm_count(device: torch.device) -> int:
    if device not in _SM_COUNT:
        _SM_COUNT[device] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SM_COUNT[device]


class _LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):  # type: ignore[override]
        assert triton is not None
        shape = x.shape
        x2d = x.reshape(-1, shape[-1]).contiguous()
        rows, hidden = x2d.shape
        y = torch.empty_like(x2d)
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
        _layer_norm_fwd_kernel[(triton.cdiv(rows, _FWD_BLOCK_M),)](
            x2d,
            y,
            weight,
            bias,
            mean,
            rstd,
            rows,
            hidden,
            eps,
            BLOCK_M=_FWD_BLOCK_M,
            BLOCK_N=triton.next_power_of_2(hidden),
            num_warps=_FWD_NUM_WARPS,
        )
        ctx.save_for_backward(x2d, weight, mean, rstd)
        ctx.shape = shape
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):  # type: ignore[override]
        assert triton is not None
        x2d, weight, mean, rstd = ctx.saved_tensors
        rows, hidden = x2d.shape
        dy2d = dy.reshape(-1, hidden).contiguous()
        dx = torch.empty_like(x2d)
        programs = min(
            triton.cdiv(rows, _BWD_BLOCK_M), _BWD_PROGRAMS_PER_SM * _sm_count(x2d.device)
        )
        dweight_partial = torch.empty((programs, hidden), device=x2d.device, dtype=torch.float32)
        dbias_partial = torch.empty((programs, hidden), device=x2d.device, dtype=torch.float32)
        _layer_norm_bwd_kernel[(programs,)](
            x2d,
            dy2d,
            dx,
            weight,
            mean,
            rstd,
            dweight_partial,
            dbias_partial,
            rows,
            hidden,
            BLOCK_M=_BWD_BLOCK_M,
            BLOCK_N=triton.next_power_of_2(hidden),
            num_warps=_BWD_NUM_WARPS,
        )
        dweight = dweight_partial.sum(0).to(weight.dtype)
        dbias = dbias_partial.sum(0).to(weight.dtype)
        return dx.view(ctx.shape), dweight, dbias, None


def layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """
    LayerNorm over the last dimension of ``x``, as ``F.layer_norm(x, (hidden,), weight, bias, eps)``.

    CUDA inputs up to 8,192 wide run the Triton kernels (statistics and gradients in float32,
    outputs in ``x``'s dtype); anything else, and any input when Triton is unavailable, uses
    ``F.layer_norm``.

    :param x: Input of shape ``(..., hidden)``.
    :param weight: Scale of shape ``(hidden,)``.
    :param bias: Shift of shape ``(hidden,)``.
    :param eps: Added to the variance for numerical stability.
    """
    hidden = x.shape[-1]
    if triton is None or not x.is_cuda or hidden > _MAX_HIDDEN or x.numel() == 0:
        return F.layer_norm(x, (hidden,), weight, bias, eps)
    return _LayerNormFunction.apply(x, weight, bias, eps)
