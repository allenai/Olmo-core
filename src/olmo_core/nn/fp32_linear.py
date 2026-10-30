"""Low-precision linear projection with FP32 output and ordinary first-order gradients."""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class _FP32Output(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias):
        ctx.save_for_backward(inputs, weight)
        ctx.bias_dtype = None if bias is None else bias.dtype
        output = torch.mm(
            inputs.reshape(-1, inputs.shape[-1]), weight.T, out_dtype=torch.float32
        ).reshape(*inputs.shape[:-1], weight.shape[0])
        return output if bias is None else output + bias.float()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        inputs, weight = ctx.saved_tensors
        # Preserve ordinary mixed-precision linear backward: the upstream gradient
        # is rounded to the operand dtype before the GEMMs and bias reduction.
        # Forward FP32 logits do not imply FP32 gradient GEMMs.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            grad = grad_output.reshape(-1, weight.shape[0]).to(inputs.dtype)
            grad_input = grad_weight = grad_bias = None
            if ctx.needs_input_grad[0]:
                grad_input = torch.mm(grad, weight).reshape(inputs.shape)
            if ctx.needs_input_grad[1]:
                grad_weight = torch.mm(grad.T, inputs.reshape(-1, inputs.shape[-1]))
            if ctx.needs_input_grad[2]:
                grad_bias = grad.sum(0).to(ctx.bias_dtype)
        return grad_input, grad_weight, grad_bias


class FP32OutputLinear(nn.Linear):
    """Linear layer retaining FP32 logits with low-precision GEMM operands.

    CUDA BF16/FP16 uses an FP32-output GEMM and a first-order custom backward
    with ordinary BF16/FP16 gradient GEMMs and bias reduction. Autocast selects operand precision;
    parameter storage and state-dict names are unchanged. CPU and FP32 operands
    use a differentiable FP32 reference projection. Tensor parallelism and
    higher-order gradients are not supported by the low-precision CUDA path.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias: Optional[torch.Tensor] = self.bias
        device_type = input.device.type
        if torch.is_autocast_enabled(device_type):
            dtype = torch.get_autocast_dtype(device_type)
            input, weight = input.to(dtype), weight.to(dtype)
            bias = None if bias is None else bias.to(dtype)
        with torch.autocast(device_type=device_type, enabled=False):
            if input.is_cuda and input.dtype in (torch.bfloat16, torch.float16):
                return _FP32Output.apply(input, weight, bias)
            return F.linear(input.float(), weight.float(), None if bias is None else bias.float())
