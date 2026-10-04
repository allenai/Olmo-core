from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton may be unavailable in some test envs.
    triton = None
    tl = None


_VALID_PREFIX_SWIGLU_BLOCK_M = 16
_VALID_PREFIX_SWIGLU_BLOCK_N = 256
_VALID_PREFIX_SWIGLU_ROW_PROGRAMS = 1024
_VALID_PREFIX_SWIGLU_NUM_WARPS = 4
_VALID_PREFIX_SWIGLU_NUM_STAGES = 1


if triton is not None:

    @triton.jit
    def _swiglu_valid_prefix_kernel(
        x_ptr,
        x_stride_0,
        x_stride_1,
        start_elements_ptr,
        num_elements_ptr,
        out_ptr,
        out_stride_0,
        out_stride_1,
        rows: tl.constexpr,
        hidden: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        ROW_PROGRAMS: tl.constexpr,
        HAS_START_TENSOR: tl.constexpr,
        START_ROW: tl.constexpr,
        MATCH_EAGER_ROUNDING: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        start_row_base = tl.load(start_elements_ptr) if HAS_START_TENSOR else START_ROW
        end_row = start_row_base + tl.load(num_elements_ptr)

        col_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        col_mask = col_idx < hidden
        row_start = start_row_base + pid_m * BLOCK_M
        row_stride: tl.constexpr = ROW_PROGRAMS * BLOCK_M

        while row_start < end_row:
            row_idx = row_start + tl.arange(0, BLOCK_M)[:, None]
            mask = (row_idx < end_row) & (row_idx < rows) & col_mask

            up_offsets = row_idx * x_stride_0 + col_idx * x_stride_1
            gate_offsets = row_idx * x_stride_0 + (col_idx + hidden) * x_stride_1
            up = tl.load(x_ptr + up_offsets, mask=mask, other=0.0).to(tl.float32)
            gate = tl.load(x_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
            if MATCH_EAGER_ROUNDING:
                # PyTorch's separate silu and multiply materialize silu in the
                # input dtype. Scoring must retain that training-time rounding.
                silu = (gate * tl.sigmoid(gate)).to(x_ptr.dtype.element_ty)
                y = up * silu.to(tl.float32)
            else:
                y = up * gate * tl.sigmoid(gate)

            out_offsets = row_idx * out_stride_0 + col_idx * out_stride_1
            tl.store(out_ptr + out_offsets, y, mask=mask)
            row_start += row_stride

    @triton.jit(do_not_specialize=["rows"])
    def _swiglu_valid_prefix_dynamic_rows_kernel(
        x_ptr,
        x_stride_0,
        x_stride_1,
        start_elements_ptr,
        num_elements_ptr,
        out_ptr,
        out_stride_0,
        out_stride_1,
        rows,
        hidden: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_START_TENSOR: tl.constexpr,
        START_ROW: tl.constexpr,
        MATCH_EAGER_ROUNDING: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        start_row_base = tl.load(start_elements_ptr) if HAS_START_TENSOR else START_ROW
        end_row = start_row_base + tl.load(num_elements_ptr)

        col_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        col_mask = col_idx < hidden
        row_start = start_row_base + pid_m * BLOCK_M
        row_stride = tl.num_programs(0) * BLOCK_M

        while row_start < end_row:
            row_idx = row_start + tl.arange(0, BLOCK_M)[:, None]
            mask = (row_idx < end_row) & (row_idx < rows) & col_mask

            up_offsets = row_idx * x_stride_0 + col_idx * x_stride_1
            gate_offsets = row_idx * x_stride_0 + (col_idx + hidden) * x_stride_1
            up = tl.load(x_ptr + up_offsets, mask=mask, other=0.0).to(tl.float32)
            gate = tl.load(x_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
            if MATCH_EAGER_ROUNDING:
                # PyTorch's separate silu and multiply materialize silu in the
                # input dtype. Scoring must retain that training-time rounding.
                silu = (gate * tl.sigmoid(gate)).to(x_ptr.dtype.element_ty)
                y = up * silu.to(tl.float32)
            else:
                y = up * gate * tl.sigmoid(gate)

            out_offsets = row_idx * out_stride_0 + col_idx * out_stride_1
            tl.store(out_ptr + out_offsets, y, mask=mask)
            row_start += row_stride

    @triton.jit
    def _swiglu_backward_valid_prefix_kernel(
        x_ptr,
        x_stride_0,
        x_stride_1,
        grad_h_ptr,
        grad_h_stride_0,
        grad_h_stride_1,
        start_elements_ptr,
        num_elements_ptr,
        out_ptr,
        out_stride_0,
        out_stride_1,
        rows: tl.constexpr,
        hidden: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        ROW_PROGRAMS: tl.constexpr,
        HAS_START_TENSOR: tl.constexpr,
        START_ROW: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        start_row_base = tl.load(start_elements_ptr) if HAS_START_TENSOR else START_ROW
        end_row = start_row_base + tl.load(num_elements_ptr)

        col_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        col_mask = col_idx < hidden
        row_start = start_row_base + pid_m * BLOCK_M
        row_stride: tl.constexpr = ROW_PROGRAMS * BLOCK_M

        while row_start < end_row:
            row_idx = row_start + tl.arange(0, BLOCK_M)[:, None]
            mask = (row_idx < end_row) & (row_idx < rows) & col_mask

            up_offsets = row_idx * x_stride_0 + col_idx * x_stride_1
            gate_offsets = row_idx * x_stride_0 + (col_idx + hidden) * x_stride_1
            grad_h_offsets = row_idx * grad_h_stride_0 + col_idx * grad_h_stride_1

            up = tl.load(x_ptr + up_offsets, mask=mask, other=0.0).to(tl.float32)
            gate = tl.load(x_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
            grad_h = tl.load(grad_h_ptr + grad_h_offsets, mask=mask, other=0.0).to(tl.float32)

            sig = tl.sigmoid(gate)
            silu_gate = gate * sig
            dsilu = sig * (1.0 + gate * (1.0 - sig))
            grad_up = grad_h * silu_gate
            grad_gate = grad_h * up * dsilu

            out_up_offsets = row_idx * out_stride_0 + col_idx * out_stride_1
            out_gate_offsets = row_idx * out_stride_0 + (col_idx + hidden) * out_stride_1
            tl.store(out_ptr + out_up_offsets, grad_up, mask=mask)
            tl.store(out_ptr + out_gate_offsets, grad_gate, mask=mask)
            row_start += row_stride


def _swiglu_valid_prefix_torch(
    x: torch.Tensor,
    num_elements: torch.Tensor,
    out: torch.Tensor,
    start: torch.Tensor | int | None,
) -> torch.Tensor:
    hidden = x.shape[1] // 2
    if start is None:
        start = 0
    valid_x = x[start : start + num_elements]
    valid_out = valid_x[:, :hidden] * F.silu(valid_x[:, hidden:])
    out[start : start + num_elements].copy_(valid_out)
    return out


def _swiglu_backward_valid_prefix_torch(
    x: torch.Tensor,
    grad_h: torch.Tensor,
    num_elements: torch.Tensor,
    out: torch.Tensor,
    start: torch.Tensor | int | None,
) -> torch.Tensor:
    hidden = x.shape[1] // 2
    if start is None:
        start = 0
    valid_x = x[start : start + num_elements]
    valid_grad_h = grad_h[start : start + num_elements]
    up = valid_x[:, :hidden]
    gate = valid_x[:, hidden:]
    gate_f32 = gate.to(torch.float32)
    grad_h_f32 = valid_grad_h.to(torch.float32)
    up_f32 = up.to(torch.float32)
    sig = torch.sigmoid(gate_f32)
    silu_gate = gate_f32 * sig
    dsilu = sig * (1.0 + gate_f32 * (1.0 - sig))
    grad_up = grad_h_f32 * silu_gate
    grad_gate = grad_h_f32 * up_f32 * dsilu
    out[start : start + num_elements].copy_(
        torch.cat((grad_up, grad_gate), dim=-1).to(dtype=x.dtype)
    )
    return out


def swiglu_valid_prefix(
    x: torch.Tensor,
    num_elements: torch.Tensor,
    *,
    start: torch.Tensor | int | None = None,
    out: Optional[torch.Tensor] = None,
    match_eager_rounding: bool = False,
    row_specialization: Literal["static", "dynamic"] = "static",
    block_m: int = _VALID_PREFIX_SWIGLU_BLOCK_M,
    block_n: int = _VALID_PREFIX_SWIGLU_BLOCK_N,
    row_programs: int = _VALID_PREFIX_SWIGLU_ROW_PROGRAMS,
    num_warps: int = _VALID_PREFIX_SWIGLU_NUM_WARPS,
    num_stages: int = _VALID_PREFIX_SWIGLU_NUM_STAGES,
) -> torch.Tensor:
    """
    Compute ``up * silu(gate)`` for ``x[start:start + num_elements]``.

    ``num_elements`` is a device scalar, so callers can pass a GPU-computed
    valid-row count without synchronizing to the host. ``start`` can also be a
    device scalar. The returned/output tensor has capacity shape
    ``[x.shape[0], x.shape[1] // 2]``; rows outside the requested range are
    intentionally left untouched. With ``match_eager_rounding=True``, the CUDA
    kernel rounds ``silu(gate)`` to the input dtype before multiplication, as
    the eager training expression does. This matches the rounding boundary,
    not necessarily every output bit: Triton and PyTorch sigmoid approximations
    can still differ, particularly in FP16/FP32. The default preserves the fused
    arithmetic used by explicit fused forward/backward callers.

    ``row_specialization="static"`` preserves the capacity-specialized kernel.
    ``"dynamic"`` reuses a forward kernel across row capacities, while retaining
    specialization on hidden width, tile sizes, offsets, and arithmetic mode.
    This option does not change the backward kernel or register an autograd rule.
    """
    if row_specialization not in ("static", "dynamic"):
        raise ValueError("row_specialization must be 'static' or 'dynamic'")
    if x.ndim != 2:
        raise ValueError(f"Expected x rank-2 [M, 2H], got {tuple(x.shape)}")
    if x.shape[1] % 2 != 0:
        raise ValueError(f"Expected even last dim for [M, 2H], got {tuple(x.shape)}")
    if num_elements.numel() != 1:
        raise ValueError(
            f"num_elements must be a scalar tensor, got shape={tuple(num_elements.shape)}"
        )
    if num_elements.dtype not in (torch.int32, torch.int64, torch.long):
        raise ValueError(f"num_elements must be int32/int64, got {num_elements.dtype}")
    if isinstance(start, torch.Tensor):
        if start.numel() != 1:
            raise ValueError(f"start must be a scalar tensor, got shape={tuple(start.shape)}")
        if start.dtype not in (torch.int32, torch.int64, torch.long):
            raise ValueError(f"start must be int32/int64, got {start.dtype}")
    elif start is not None and not isinstance(start, int):
        raise ValueError(
            f"start must be None, an int, or a scalar tensor, got {type(start).__name__}"
        )

    rows = x.shape[0]
    hidden = x.shape[1] // 2
    if out is None:
        out = torch.empty((rows, hidden), device=x.device, dtype=x.dtype)
    else:
        if tuple(out.shape) != (rows, hidden):
            raise ValueError(
                f"out shape mismatch: expected {(rows, hidden)}, got {tuple(out.shape)}"
            )
        if out.device != x.device:
            raise ValueError(f"out device must match x device: {out.device} vs {x.device}")
        if out.dtype != x.dtype:
            raise ValueError(f"out dtype must match x dtype: {out.dtype} vs {x.dtype}")

    if not x.is_cuda:
        return _swiglu_valid_prefix_torch(x, num_elements, out, start)
    if triton is None or tl is None:
        raise RuntimeError("Triton is required for CUDA swiglu_valid_prefix")
    if num_elements.device != x.device:
        raise ValueError(
            f"num_elements device must match x device: {num_elements.device} vs {x.device}"
        )
    if isinstance(start, torch.Tensor) and start.device != x.device:
        raise ValueError(f"start device must match x device: {start.device} vs {x.device}")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported dtype for swiglu_valid_prefix: {x.dtype}")
    if not x.is_contiguous():
        raise ValueError("swiglu_valid_prefix requires contiguous x")
    if not out.is_contiguous():
        raise ValueError("swiglu_valid_prefix requires contiguous out")
    if block_m <= 0 or block_n <= 0 or row_programs <= 0:
        raise ValueError(
            "block_m, block_n, and row_programs must all be positive "
            f"(got {block_m}, {block_n}, {row_programs})"
        )

    row_grid = min(triton.cdiv(rows, block_m), int(row_programs))
    col_grid = triton.cdiv(hidden, block_n)
    has_start_tensor = isinstance(start, torch.Tensor)
    start_ptr = start if has_start_tensor else num_elements
    start_row = 0 if start is None or has_start_tensor else int(start)
    kernel = (
        _swiglu_valid_prefix_kernel
        if row_specialization == "static"
        else _swiglu_valid_prefix_dynamic_rows_kernel
    )
    capacity_options = {"ROW_PROGRAMS": int(row_grid)} if row_specialization == "static" else {}
    kernel[(row_grid, col_grid)](
        x,
        x.stride(0),
        x.stride(1),
        start_ptr,
        num_elements,
        out,
        out.stride(0),
        out.stride(1),
        rows,
        hidden,
        BLOCK_M=int(block_m),
        BLOCK_N=int(block_n),
        **capacity_options,
        HAS_START_TENSOR=has_start_tensor,
        START_ROW=start_row,
        MATCH_EAGER_ROUNDING=match_eager_rounding,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


def swiglu_backward_valid_prefix(
    x: torch.Tensor,
    grad_h: torch.Tensor,
    num_elements: torch.Tensor,
    *,
    start: torch.Tensor | int | None = None,
    out: Optional[torch.Tensor] = None,
    block_m: int = _VALID_PREFIX_SWIGLU_BLOCK_M,
    block_n: int = _VALID_PREFIX_SWIGLU_BLOCK_N,
    row_programs: int = _VALID_PREFIX_SWIGLU_ROW_PROGRAMS,
    num_warps: int = _VALID_PREFIX_SWIGLU_NUM_WARPS,
    num_stages: int = _VALID_PREFIX_SWIGLU_NUM_STAGES,
) -> torch.Tensor:
    """
    Compute SwiGLU backward for ``x[start:start + num_elements]``.

    ``x`` is the forward linear1 output ``[M, 2H]`` and ``grad_h`` is the
    activation-output gradient ``[M, H]``. The returned/output tensor has
    capacity shape ``[M, 2H]``; rows outside the requested range are left
    untouched.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected x rank-2 [M, 2H], got {tuple(x.shape)}")
    if x.shape[1] % 2 != 0:
        raise ValueError(f"Expected even last dim for [M, 2H], got {tuple(x.shape)}")
    hidden = x.shape[1] // 2
    if grad_h.ndim != 2 or tuple(grad_h.shape) != (x.shape[0], hidden):
        raise ValueError(
            f"grad_h must be [M, H] matching x, got {tuple(grad_h.shape)} "
            f"for x={tuple(x.shape)}"
        )
    if num_elements.numel() != 1:
        raise ValueError(
            f"num_elements must be a scalar tensor, got shape={tuple(num_elements.shape)}"
        )
    if num_elements.dtype not in (torch.int32, torch.int64, torch.long):
        raise ValueError(f"num_elements must be int32/int64, got {num_elements.dtype}")
    if isinstance(start, torch.Tensor):
        if start.numel() != 1:
            raise ValueError(f"start must be a scalar tensor, got shape={tuple(start.shape)}")
        if start.dtype not in (torch.int32, torch.int64, torch.long):
            raise ValueError(f"start must be int32/int64, got {start.dtype}")
    elif start is not None and not isinstance(start, int):
        raise ValueError(
            f"start must be None, an int, or a scalar tensor, got {type(start).__name__}"
        )

    rows = x.shape[0]
    if out is None:
        out = torch.empty_like(x)
    else:
        if tuple(out.shape) != tuple(x.shape):
            raise ValueError(
                f"out shape mismatch: expected {tuple(x.shape)}, got {tuple(out.shape)}"
            )
        if out.device != x.device:
            raise ValueError(f"out device must match x device: {out.device} vs {x.device}")
        if out.dtype != x.dtype:
            raise ValueError(f"out dtype must match x dtype: {out.dtype} vs {x.dtype}")

    if not x.is_cuda:
        return _swiglu_backward_valid_prefix_torch(x, grad_h, num_elements, out, start)
    if triton is None or tl is None:
        raise RuntimeError("Triton is required for CUDA swiglu_backward_valid_prefix")
    if grad_h.device != x.device:
        raise ValueError(f"grad_h device must match x device: {grad_h.device} vs {x.device}")
    if num_elements.device != x.device:
        raise ValueError(
            f"num_elements device must match x device: {num_elements.device} vs {x.device}"
        )
    if isinstance(start, torch.Tensor) and start.device != x.device:
        raise ValueError(f"start device must match x device: {start.device}")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported dtype for swiglu_backward_valid_prefix: {x.dtype}")
    if grad_h.dtype != x.dtype:
        raise ValueError(f"grad_h dtype must match x dtype: {grad_h.dtype} vs {x.dtype}")
    if not x.is_contiguous():
        raise ValueError("swiglu_backward_valid_prefix requires contiguous x")
    if not grad_h.is_contiguous():
        raise ValueError("swiglu_backward_valid_prefix requires contiguous grad_h")
    if not out.is_contiguous():
        raise ValueError("swiglu_backward_valid_prefix requires contiguous out")
    if block_m <= 0 or block_n <= 0 or row_programs <= 0:
        raise ValueError(
            "block_m, block_n, and row_programs must all be positive "
            f"(got {block_m}, {block_n}, {row_programs})"
        )

    row_grid = min(triton.cdiv(rows, block_m), int(row_programs))
    col_grid = triton.cdiv(hidden, block_n)
    has_start_tensor = isinstance(start, torch.Tensor)
    start_ptr = start if has_start_tensor else num_elements
    start_row = 0 if start is None or has_start_tensor else int(start)
    _swiglu_backward_valid_prefix_kernel[(row_grid, col_grid)](
        x,
        x.stride(0),
        x.stride(1),
        grad_h,
        grad_h.stride(0),
        grad_h.stride(1),
        start_ptr,
        num_elements,
        out,
        out.stride(0),
        out.stride(1),
        rows,
        hidden,
        BLOCK_M=int(block_m),
        BLOCK_N=int(block_n),
        ROW_PROGRAMS=int(row_grid),
        HAS_START_TENSOR=has_start_tensor,
        START_ROW=start_row,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


__all__ = ["swiglu_valid_prefix", "swiglu_backward_valid_prefix"]
