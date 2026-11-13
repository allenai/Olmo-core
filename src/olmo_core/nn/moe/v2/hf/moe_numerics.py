"""Numerics of OLMoDDP's no-EP expert forward, without a TransformerEngine dependency."""

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only Transformers installations.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _swiglu_kernel(x, y, rows: tl.constexpr, hidden: tl.constexpr, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row = index // hidden
        col = index % hidden
        up = tl.load(x + row * (2 * hidden) + col, row < rows, 0).to(tl.float32)
        gate = tl.load(x + row * (2 * hidden) + hidden + col, row < rows, 0).to(tl.float32)
        value = up * gate * tl.sigmoid(gate)
        tl.store(y + index, value, row < rows)


def core_swiglu(up_gate: torch.Tensor) -> torch.Tensor:
    """Apply the core inference kernel's single-rounding SwiGLU to packed [up, gate]."""
    if up_gate.is_cuda and not torch.is_grad_enabled() and triton is not None:
        up_gate = up_gate.contiguous()
        rows, width = up_gate.shape
        hidden = width // 2
        output = up_gate.new_empty((rows, hidden))
        if rows:
            _swiglu_kernel[(triton.cdiv(rows * hidden, 1024),)](
                up_gate, output, rows, hidden, BLOCK=1024
            )
        return output
    up, gate = up_gate.float().chunk(2, dim=-1)
    return (up * gate * torch.sigmoid(gate)).to(up_gate.dtype)


def _core_combine_impl(values: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    """Reproduce TE's index-map combination in routing-slot order.

    TE 2.18 converts probabilities to the activation dtype and rounds every
    update to that dtype. On pre-SM90 GPUs, CUDA's BF16 multiply and add each
    lower to an FMA and round separately; on SM90+ they contract into one FMA.
    CPU uses the SM90+ arithmetic. The opaque custom op preserves these rounding
    boundaries under full-graph compilation.
    ``values`` is [tokens, top_k, hidden]; probabilities remain FP32 up to this op.
    """
    dtype = values.dtype
    separate_rounding = (
        dtype == torch.bfloat16
        and values.is_cuda
        and torch.cuda.get_device_capability(values.device)[0] < 9
    )
    probabilities = probabilities.to(dtype).float()
    values = values.float()
    output = values.new_zeros((values.shape[0], values.shape[2])).to(dtype)
    for slot in range(values.shape[1]):
        if separate_rounding:
            product = (values[:, slot] * probabilities[:, slot, None]).to(dtype)
            output = (output.float() + product.float()).to(dtype)
        else:
            output = torch.addcmul(
                output.float(), values[:, slot], probabilities[:, slot, None]
            ).to(dtype)
    return output


def _core_combine_fake(values, probabilities):
    return values.new_empty((values.shape[0], values.shape[2]))


def _core_combine_setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _core_combine_backward_impl(
    values: torch.Tensor, probabilities: torch.Tensor, grad_output: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Match autograd through the activation-dtype casts in the forward.
    grad = grad_output.float()[:, None, :]
    grad_values = (grad * probabilities.to(values.dtype).float()[..., None]).to(values.dtype)
    grad_probabilities = (grad * values.float()).sum(-1).to(values.dtype).to(probabilities.dtype)
    return grad_values, grad_probabilities


def _core_combine_backward_fake(values, probabilities, grad_output):
    return torch.empty_like(values), torch.empty_like(probabilities)


def _core_combine_backward(ctx, grad_output):
    return _core_combine_backward_op(*ctx.saved_tensors, grad_output)


# HF remote-code loading imports this file under a different module name for
# each checkpoint/revision. Operator names are process-global: register each
# implementation (including fake/autograd kernels) only once, then reuse it.
if not hasattr(torch.ops.olmo3_moe_hf, "core_combine_backward"):
    _backward_op = torch.library.custom_op("olmo3_moe_hf::core_combine_backward", mutates_args=())(
        _core_combine_backward_impl
    )
    _backward_op.register_fake(_core_combine_backward_fake)
_core_combine_backward_op = torch.ops.olmo3_moe_hf.core_combine_backward.default

if not hasattr(torch.ops.olmo3_moe_hf, "core_combine"):
    _combine_op = torch.library.custom_op("olmo3_moe_hf::core_combine", mutates_args=())(
        _core_combine_impl
    )
    _combine_op.register_fake(_core_combine_fake)
    _combine_op.register_autograd(_core_combine_backward, setup_context=_core_combine_setup_context)
core_combine = torch.ops.olmo3_moe_hf.core_combine.default
