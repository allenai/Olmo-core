"""
Notes on torch grouped_mm backend behavior (PyTorch 2.10):

- torch.nn.functional.grouped_mm(...) calls aten::_grouped_mm.
- CUDA path in aten/native/cuda/GroupedBlas.cpp chooses:
  1) Fast path: at::cuda::detail::bf16bf16_grouped_mm(...)
  2) Fallback: _grouped_mm_fallback(...)

Fast path conditions (CUDA):
- mat_a dtype == bf16
- mat_b dtype == bf16
- out_dtype (or default) == bf16
- _scaled_mm_allowed_device(sm90_only=true, sm100_only=true) is true
  (in 2.10 this means device major == 9 or 10 only)
- grouped-mm CUTLASS kernels are built
  (!USE_ROCM, !Windows, CUDA_VERSION >= 12.0)

Fallback behavior:
- Uses offs.cpu() and loops over groups with mm_out/bmm_out.
- This can introduce D2H sync for device offs.

Implication:
- On CUDA devices outside the fastpath gate (e.g. major 12 in PyTorch 2.10),
  grouped_mm falls back even if offs is on GPU.
"""

import os

try:
    import grouped_gemm  # type: ignore
    import grouped_gemm.ops  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - import guard
    grouped_gemm = None  # type: ignore[assignment]
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Literal, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from olmo_core._nvtx import nvtx
from olmo_core.config import Config, DType, StrEnum
from olmo_core.kernels import (
    OlmoMXFP8Tensor,
    ScaledGroupedMMPrequantizedLHS,
    ScaledGroupedMMPrequantizedRHS,
)
from olmo_core.kernels import grouped_mm as grouped_mm_with_buffers
from olmo_core.kernels import (
    grouped_mm_row_offset,
    scaled_grouped_mm_q,
    scaled_grouped_mm_q_fp8_weight,
)
from olmo_core.kernels.mxfp8_utils import (
    dequantize_rows_from_mxfp8,
    swiglu_quantize_rows_from_mxfp8,
    swiglu_quantize_rows_to_mxfp8,
)
from olmo_core.kernels.swiglu import swiglu_valid_prefix
from olmo_core.nn.fp8_weight import FP8WeightCacheSpec, FP8WeightStore

from .fp8 import MoERowwiseFP8Config, normalize_rowwise_fp8_config


class ExpertActivation(StrEnum):
    swiglu = "swiglu"
    relu2 = "relu2"
    gpt_oss_swiglu = "gpt_oss_swiglu"


def _debug_is_inf_or_nan(x):
    return torch.logical_or(~torch.isfinite(x), torch.isnan(x))


def _debug_get_row_indices_for_nan_or_inf(x):
    return torch.where(_debug_is_inf_or_nan(x).any(dim=-1))[0]


def _debug_get_row_indices_for_nan_or_inf_before_end(x, end):
    naninf_row_indices = _debug_get_row_indices_for_nan_or_inf(x)
    return naninf_row_indices[naninf_row_indices < end]


def _identity_fp8_rhs(weight: torch.Tensor) -> torch.Tensor:
    return weight


def _transpose_last2_fp8_rhs(weight: torch.Tensor) -> torch.Tensor:
    return weight.transpose(-2, -1)


_ROUTED_UP_GATE_FP8_CACHE_SPECS = (
    FP8WeightCacheSpec("rhs", _transpose_last2_fp8_rhs),
    FP8WeightCacheSpec("rhs_for_dgrad", _identity_fp8_rhs),
)
_ROUTED_DOWN_FP8_CACHE_SPECS = (
    FP8WeightCacheSpec("rhs", _identity_fp8_rhs),
    FP8WeightCacheSpec("rhs_for_dgrad", _transpose_last2_fp8_rhs),
)


class _SwiGLUQuantizeRowsMXFP8Autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, up_gate: torch.Tensor, block_size: int):  # type: ignore[override]
        h, h_q, h_scales = swiglu_quantize_rows_to_mxfp8(
            up_gate,
            block_size=int(block_size),
        )
        if up_gate.requires_grad:
            ctx.save_for_backward(up_gate)
        else:
            ctx.save_for_backward()
        return h, h_q, h_scales

    @staticmethod
    def backward(ctx, grad_h: torch.Tensor, grad_h_q: torch.Tensor, grad_h_scales: torch.Tensor):  # type: ignore[override]
        del grad_h_q, grad_h_scales
        if not ctx.needs_input_grad[0]:
            return None, None
        if grad_h is None:
            up_gate = ctx.saved_tensors[0]
            return torch.zeros_like(up_gate), None

        (up_gate,) = ctx.saved_tensors
        grad_up_gate = swiglu_backward_grad_up_gate(up_gate, grad_h)
        return OlmoMXFP8Tensor.from_hp(grad_up_gate), None


def _swiglu_backward_grad_up_gate_impl(up_gate: torch.Tensor, grad_h: torch.Tensor) -> torch.Tensor:
    hidden = up_gate.shape[-1] // 2
    up = up_gate[:, :hidden]
    gate = up_gate[:, hidden:]

    gate_f32 = gate.to(torch.float32)
    grad_h_f32 = grad_h.to(torch.float32)
    up_f32 = up.to(torch.float32)

    sig = torch.sigmoid(gate_f32)
    silu_gate = gate_f32 * sig
    dsilu = sig * (1.0 + gate_f32 * (1.0 - sig))

    grad_up = grad_h_f32 * silu_gate
    grad_gate = grad_h_f32 * up_f32 * dsilu
    return torch.cat((grad_up, grad_gate), dim=-1).to(dtype=up_gate.dtype)


_swiglu_backward_grad_up_gate_compiled = torch.compile(
    _swiglu_backward_grad_up_gate_impl,
    fullgraph=True,
    dynamic=False,
)


def swiglu_backward_grad_up_gate(up_gate: torch.Tensor, grad_h: torch.Tensor) -> torch.Tensor:
    if (
        up_gate.is_cuda
        and grad_h.is_cuda
        and os.getenv("OLMO_MXFP8_COMPILE_SWIGLU_BWD", "1").strip().lower()
        in {"1", "true", "yes", "on"}
    ):
        return _swiglu_backward_grad_up_gate_compiled(up_gate, grad_h)
    return _swiglu_backward_grad_up_gate_impl(up_gate, grad_h)


@torch.compiler.disable
def swiglu_quantize_rows_mxfp8_autograd(
    up_gate: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _SwiGLUQuantizeRowsMXFP8Autograd.apply(up_gate, int(block_size))


class _SwiGLUQuantizeRowsFromMXFP8Autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, up_gate_q: torch.Tensor, up_gate_scales: torch.Tensor, block_size: int):  # type: ignore[override]
        h, h_q, h_scales = swiglu_quantize_rows_from_mxfp8(
            up_gate_q,
            up_gate_scales,
            block_size=int(block_size),
        )
        if up_gate_q.requires_grad:
            ctx.save_for_backward(up_gate_q, up_gate_scales)
            ctx.block_size = int(block_size)
        else:
            ctx.save_for_backward()
            ctx.block_size = int(block_size)
        return h, h_q, h_scales

    @staticmethod
    def backward(ctx, grad_h: torch.Tensor, grad_h_q: torch.Tensor, grad_h_scales: torch.Tensor):  # type: ignore[override]
        del grad_h_q, grad_h_scales
        if not ctx.needs_input_grad[0]:
            return None, None, None

        up_gate_q, up_gate_scales = ctx.saved_tensors
        if grad_h is None:
            return (
                torch.zeros(
                    up_gate_q.shape,
                    dtype=torch.bfloat16,
                    device=up_gate_q.device,
                ),
                None,
                None,
            )

        up_gate = dequantize_rows_from_mxfp8(
            up_gate_q,
            up_gate_scales,
            block_size=int(ctx.block_size),
            out_dtype=torch.bfloat16,
        )
        hidden = up_gate.shape[-1] // 2
        up = up_gate[:, :hidden]
        gate = up_gate[:, hidden:]

        gate_f32 = gate.to(torch.float32)
        grad_h_f32 = grad_h.to(torch.float32)
        up_f32 = up.to(torch.float32)

        sig = torch.sigmoid(gate_f32)
        silu_gate = gate_f32 * sig
        dsilu = sig * (1.0 + gate_f32 * (1.0 - sig))

        grad_up = grad_h_f32 * silu_gate
        grad_gate = grad_h_f32 * up_f32 * dsilu
        grad_up_gate = torch.cat((grad_up, grad_gate), dim=-1).to(dtype=torch.bfloat16)
        return grad_up_gate, None, None


@torch.compiler.disable
def gmm_no_compile(a, b, batch_sizes, trans_b=False):
    return grouped_gemm.ops.gmm(a, b, batch_sizes, trans_b)


def gmm(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_b: bool = False,
    out: Optional[torch.Tensor] = None,
    input_grad_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if use_torch_grouped_mm():
        # torch.nn.functional.grouped_mm has no trans_b argument.
        # It expects mat_b to be (num_groups, K, N), so we transpose when
        # emulating grouped_gemm(..., trans_b=True).
        b_grouped_mm = b.transpose(1, 2) if trans_b else b
        offs = torch.cumsum(batch_sizes.to(dtype=torch.int32), dim=0, dtype=torch.int32)
        if out is not None or input_grad_out is not None:
            out_tensor = grouped_mm_with_buffers(
                a,
                b_grouped_mm,
                offs=offs,
                out=out,
                input_grad_out=input_grad_out,
            )
        else:
            out_tensor = F.grouped_mm(a, b_grouped_mm, offs=offs)
        # WARNING: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grouped_mm.html
        # "offs[i] marks the end of group i and offs[-1] must be strictly less than the total length of that operand’s sliced dimension"
        # so padded positions are always necessary? I think "strictly less than" is a documentation mistake

        # BIG NOTE: grouped_mm's returned value containes uninitialized values for the padded positions (pos > offsets[-1]), which can be NaN and may cause later ops to produce NaN in valid positions.
        # if _debug_get_row_indices_for_nan_or_inf_before_end(out, batch_sizes.sum()).numel() > 0:
        #     raise RuntimeError(f"NaN or Inf detected in grouped_mm output in valid tokens. batch_sizes={batch_sizes} offs={offs} out={out}")
        return out_tensor

    if out is not None or input_grad_out is not None:
        raise RuntimeError("gmm(out=..., input_grad_out=...) requires torch grouped_mm backend")
    return gmm_no_compile(a, b, batch_sizes, trans_b)


@torch.compiler.disable
def scaled_grouped_mm_q_no_compile(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    *,
    offs: torch.Tensor,
    use_fast_accum: bool,
    prequantized_lhs: Optional[ScaledGroupedMMPrequantizedLHS] = None,
    prequantized_rhs: Optional[ScaledGroupedMMPrequantizedRHS] = None,
    prequantized_rhs_for_dgrad: Optional[ScaledGroupedMMPrequantizedRHS] = None,
) -> torch.Tensor:
    kwargs = dict(
        offs=offs,
        use_fast_accum=use_fast_accum,
        prequantized_lhs=prequantized_lhs,
        prequantized_rhs=prequantized_rhs,
    )
    if prequantized_rhs_for_dgrad is not None:
        kwargs["prequantized_rhs_for_dgrad"] = prequantized_rhs_for_dgrad
    return scaled_grouped_mm_q(
        mat_a,
        mat_b,
        **kwargs,
    )


@torch.compiler.disable
def scaled_grouped_mm_q_fp8_weight_no_compile(
    mat_a: torch.Tensor,
    grad_anchor: torch.Tensor,
    *,
    offs: torch.Tensor,
    use_fast_accum: bool,
    prequantized_lhs: Optional[ScaledGroupedMMPrequantizedLHS] = None,
    prequantized_rhs: ScaledGroupedMMPrequantizedRHS,
    prequantized_rhs_for_dgrad: ScaledGroupedMMPrequantizedRHS,
    wgrad_sink: Optional[FP8WeightStore] = None,
    wgrad_sink_transpose_last2: bool = False,
    wgrad_sink_squeeze_first_dim: bool = False,
) -> torch.Tensor:
    kwargs = dict(
        offs=offs,
        use_fast_accum=use_fast_accum,
        prequantized_lhs=prequantized_lhs,
        prequantized_rhs=prequantized_rhs,
        prequantized_rhs_for_dgrad=prequantized_rhs_for_dgrad,
    )
    if wgrad_sink is not None:
        kwargs["wgrad_sink"] = wgrad_sink
        kwargs["wgrad_sink_transpose_last2"] = wgrad_sink_transpose_last2
        kwargs["wgrad_sink_squeeze_first_dim"] = wgrad_sink_squeeze_first_dim
    return scaled_grouped_mm_q_fp8_weight(
        mat_a,
        grad_anchor,
        **kwargs,
    )


# if env variable OLMO_USE_TORCH_GROUPED_MM is set, use its value to determine whether to use torch grouped_mm;
USE_TORCH_GROUPED_MM: Optional[bool] = None
env_val = os.getenv("OLMO_USE_TORCH_GROUPED_MM")
if env_val is not None:
    if env_val.lower() in ("1", "true", "yes"):
        USE_TORCH_GROUPED_MM = True
    elif env_val.lower() in ("0", "false", "no"):
        USE_TORCH_GROUPED_MM = False
    else:
        raise ValueError(
            f"Invalid value for OLMO_USE_TORCH_GROUPED_MM: {env_val}. Expected one of (1, 0, true, false, yes, no)."
        )
else:
    # otherwise, use feature detection and version gate.
    USE_TORCH_GROUPED_MM = None


def use_torch_grouped_mm():
    global USE_TORCH_GROUPED_MM
    if USE_TORCH_GROUPED_MM is not None:
        return USE_TORCH_GROUPED_MM

    torch_version = torch.__version__.split("+")[0]  # strip local build suffix, e.g. +cu128
    try:
        major_str, minor_str, *_ = torch_version.split(".")
        major, minor = int(major_str), int(minor_str)
        meets_version_gate = major > 2 or (major == 2 and minor >= 10)
    except (ValueError, TypeError):
        # Fall back to feature detection on unusual version strings.
        meets_version_gate = hasattr(F, "grouped_mm")

    # grouped_mm was added in torch 2.10; hasattr keeps this robust to local builds.
    USE_TORCH_GROUPED_MM = meets_version_gate and hasattr(F, "grouped_mm")
    return USE_TORCH_GROUPED_MM


REQUIRES_HOST_SIDE_SPLIT_SIZES = None  # cache the result of whether host-side split sizes are required, since it does not change during runtime and checking it requires parsing torch version every time.


def requires_host_side_split_sizes():
    # read from cache if available
    global REQUIRES_HOST_SIDE_SPLIT_SIZES
    if REQUIRES_HOST_SIDE_SPLIT_SIZES is not None:
        return REQUIRES_HOST_SIDE_SPLIT_SIZES

    # grouped_gemm cublas mode requires host-side split sizes, grouped_mm does not.
    REQUIRES_HOST_SIDE_SPLIT_SIZES = not use_torch_grouped_mm()

    return REQUIRES_HOST_SIDE_SPLIT_SIZES


@dataclass
class RoutedExpertsConfig(Config):
    """Configuration for routed experts in a MoE block."""

    # Input (and output) dimension of the experts
    d_model: int

    # Hidden (intermediate) dimension of the experts
    hidden_size: int

    # Number of routed experts
    num_experts: int

    # Whether to use bias in the experts
    bias: bool

    # default dtype for the experts
    dtype: DType

    # Optional FP8 config used by rowwise EP no-sync path.
    rowwise_fp8: Optional[MoERowwiseFP8Config] = None
    activation: ExpertActivation = ExpertActivation.swiglu
    activation_alpha: float = 1.702
    activation_limit: Optional[float] = None
    # Forward valid-prefix kernel only; direct wave/backward callers keep their defaults.
    row_specialization: Literal["static", "dynamic"] = "static"
    match_eager_rounding: bool = False
    """Round SiLU before multiplication in no-gradient SwiGLU to match eager training."""

    def build(
        self,
        init_device: str = "cpu",
    ) -> "RoutedExperts":
        kwargs = self.as_dict()
        return RoutedExperts(init_device=init_device, **kwargs)

    def num_params(self) -> int:
        """
        The number of params that the module will have once built.

        :param d_model: The model dimensionality.
        """

        up_factor = (
            2
            if self.activation in (ExpertActivation.swiglu, ExpertActivation.gpt_oss_swiglu)
            else 1
        )
        params = (up_factor + 1) * self.d_model * self.hidden_size  # up[/gate], down
        if self.bias:
            params += up_factor * self.hidden_size  # up[/gate] bias
            params += self.d_model  # down bias

        params *= self.num_experts  # for each expert

        return params

    def num_active_params(self, top_k: int) -> int:
        """
        The number of params that the module will have once built, given the top_k experts.

        :param top_k: The number of experts to use.
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if top_k > self.num_experts:
            raise ValueError(
                f"top_k ({top_k}) cannot be greater than num_experts ({self.num_experts})"
            )

        up_factor = (
            2
            if self.activation in (ExpertActivation.swiglu, ExpertActivation.gpt_oss_swiglu)
            else 1
        )
        params = (up_factor + 1) * self.d_model * self.hidden_size  # up[/gate], down
        if self.bias:
            params += up_factor * self.hidden_size  # up[/gate] bias
            params += self.d_model  # down bias

        params *= top_k  # for each expert

        return params


class RoutedExperts(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        num_experts: int,
        bias: bool,
        dtype: DType,
        rowwise_fp8: Optional[MoERowwiseFP8Config] = None,
        activation: ExpertActivation = ExpertActivation.swiglu,
        activation_alpha: float = 1.702,
        activation_limit: Optional[float] = None,
        init_device: str = "cpu",
        row_specialization: Literal["static", "dynamic"] = "static",
        match_eager_rounding: bool = False,
    ):
        super().__init__()
        if row_specialization not in ("static", "dynamic"):
            raise ValueError("row_specialization must be static or dynamic")
        self.row_specialization = row_specialization
        self.match_eager_rounding = match_eager_rounding
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.activation = ExpertActivation(activation)
        self._profile_pairwise_swiglu = os.environ.get("OLMO_PROFILE_SWIGLU_PAIRWISE", "0") == "1"
        self._profile_rounded_wgrad = os.environ.get("OLMO_PROFILE_ROUNDED_WGRAD", "0") == "1"
        self.activation_alpha = float(activation_alpha)
        self.activation_limit = None if activation_limit is None else float(activation_limit)
        self.bias = bias
        up_factor = (
            2
            if self.activation in (ExpertActivation.swiglu, ExpertActivation.gpt_oss_swiglu)
            else 1
        )
        self.w_up_gate = nn.Parameter(
            torch.empty(
                num_experts,
                up_factor * hidden_size,
                d_model,
                dtype=dtype.as_pt(),
                device=init_device,
            ),
        )

        self.w_down = nn.Parameter(
            torch.empty(num_experts, hidden_size, d_model, dtype=dtype.as_pt(), device=init_device),
        )
        self.b_up_gate = (
            nn.Parameter(
                torch.empty(
                    num_experts,
                    up_factor * hidden_size,
                    dtype=dtype.as_pt(),
                    device=init_device,
                ),
            )
            if bias
            else None
        )
        self.b_down = (
            nn.Parameter(
                torch.empty(
                    num_experts,
                    d_model,
                    dtype=dtype.as_pt(),
                    device=init_device,
                ),
            )
            if bias
            else None
        )
        owner_ref = weakref.ref(self)
        self.w_up_gate._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
        self.w_down._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]

        # assume no ep in init
        self.num_local_experts: int = num_experts
        self.ep_dim: int = 1
        self.ep_rank: int = 0
        self.rowwise_fp8 = normalize_rowwise_fp8_config(rowwise_fp8)
        if self.rowwise_fp8 is not None and self.rowwise_fp8.enabled:
            # Config-only check here; runtime CUDA support is asserted at FP8 buffer build.
            self.rowwise_fp8.validate()
        self._rowwise_fp8_up_gate_prequant: Optional[ScaledGroupedMMPrequantizedRHS] = None
        self._rowwise_fp8_down_prequant: Optional[ScaledGroupedMMPrequantizedRHS] = None
        self._rowwise_fp8_up_gate_prequant_t: Optional[ScaledGroupedMMPrequantizedRHS] = None
        self._rowwise_fp8_down_prequant_t: Optional[ScaledGroupedMMPrequantizedRHS] = None
        self._rowwise_fp8_weight_versions: Optional[tuple[int, int]] = None
        self._rowwise_fp8_up_gate_weight = FP8WeightStore(
            logical_name="w_up_gate",
            logical_shape=tuple(self.w_up_gate.shape),
            cache_specs=_ROUTED_UP_GATE_FP8_CACHE_SPECS,
            anchor_param=self.w_up_gate,
            optimizer_enabled=(
                self.rowwise_fp8 is not None
                and self.rowwise_fp8.enabled
                and self.rowwise_fp8.fp8_only_params
            ),
        )
        self._rowwise_fp8_down_weight = FP8WeightStore(
            logical_name="w_down",
            logical_shape=tuple(self.w_down.shape),
            cache_specs=_ROUTED_DOWN_FP8_CACHE_SPECS,
            anchor_param=self.w_down,
            optimizer_enabled=(
                self.rowwise_fp8 is not None
                and self.rowwise_fp8.enabled
                and self.rowwise_fp8.fp8_only_params
            ),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Standalone default; the model-level init (``init_moe_v2``) may override the weights
        # with a depth-scaled std. Biases are not touched by that init, so zero them here.
        std = 0.02
        for w in (self.w_up_gate, self.w_down):
            nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)
        for b in (self.b_up_gate, self.b_down):
            if b is not None:
                nn.init.zeros_(b)

    def _sync_rowwise_fp8_weight_anchors(self) -> None:
        fp8_only_params = (
            self.rowwise_fp8 is not None
            and self.rowwise_fp8.enabled
            and self.rowwise_fp8.fp8_only_params
        )
        self._rowwise_fp8_up_gate_weight.anchor_param = self.w_up_gate
        self._rowwise_fp8_up_gate_weight.logical_shape = tuple(self.w_up_gate.shape)
        self._rowwise_fp8_up_gate_weight.optimizer_enabled = fp8_only_params
        self._rowwise_fp8_down_weight.anchor_param = self.w_down
        self._rowwise_fp8_down_weight.logical_shape = tuple(self.w_down.shape)
        self._rowwise_fp8_down_weight.optimizer_enabled = fp8_only_params
        if fp8_only_params:
            self.w_up_gate.requires_grad_(False)
            self.w_down.requires_grad_(False)

    def disable_mxfp8_expert_anchor_grads(self) -> None:
        cfg = self.rowwise_fp8
        if cfg is None or not cfg.enabled or not cfg.fp8_only_params:
            return
        self._sync_rowwise_fp8_weight_anchors()
        self.w_up_gate.grad = None
        self.w_down.grad = None
        if hasattr(self.w_up_gate, "_main_grad_fp32"):
            self.w_up_gate._main_grad_fp32 = None  # type: ignore[attr-defined]
        if hasattr(self.w_down, "_main_grad_fp32"):
            self.w_down._main_grad_fp32 = None  # type: ignore[attr-defined]

    def release_mxfp8_expert_anchor_storage(self) -> None:
        cfg = self.rowwise_fp8
        if cfg is None or not cfg.enabled or not cfg.fp8_only_params:
            return
        self._sync_rowwise_fp8_weight_anchors()
        self._rowwise_fp8_up_gate_weight.release_anchor_storage()
        self._rowwise_fp8_down_weight.release_anchor_storage()

    def named_fp8_weight_stores(self) -> Iterator[tuple[str, FP8WeightStore]]:
        self._sync_rowwise_fp8_weight_anchors()
        yield "w_up_gate", self._rowwise_fp8_up_gate_weight
        yield "w_down", self._rowwise_fp8_down_weight

    def named_mxfp8_expert_weights(self) -> Iterator[tuple[str, FP8WeightStore]]:
        yield from self.named_fp8_weight_stores()

    def zero_mxfp8_expert_weight_grads(self, set_to_none: bool = True) -> None:
        for _, weight in self.named_mxfp8_expert_weights():
            weight.zero_grad(set_to_none=set_to_none)

    def set_mxfp8_expert_weight_main_grads_to_none(self) -> None:
        for _, weight in self.named_mxfp8_expert_weights():
            weight.set_main_grad_to_none()

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        self.zero_mxfp8_expert_weight_grads(set_to_none=set_to_none)

    def _sync_mxfp8_weight_grad_from_anchor(
        self,
        weight: FP8WeightStore,
        anchor: torch.nn.Parameter,
    ) -> None:
        anchor_grad = getattr(anchor, "_main_grad_fp32", None)
        if anchor_grad is None:
            anchor_grad = anchor.grad
        if anchor_grad is None:
            return
        weight.replace_wgrad(anchor_grad)

    def sync_mxfp8_expert_weight_grads_from_anchor(self) -> None:
        cfg = self.rowwise_fp8
        if cfg is None or not cfg.enabled or not cfg.fp8_only_params:
            return
        self._sync_mxfp8_weight_grad_from_anchor(
            self._rowwise_fp8_up_gate_weight,
            self.w_up_gate,
        )
        self._sync_mxfp8_weight_grad_from_anchor(
            self._rowwise_fp8_down_weight,
            self.w_down,
        )

    def invalidate_rowwise_fp8_cache(self) -> None:
        self._rowwise_fp8_up_gate_prequant = None
        self._rowwise_fp8_down_prequant = None
        self._rowwise_fp8_up_gate_prequant_t = None
        self._rowwise_fp8_down_prequant_t = None
        self._rowwise_fp8_weight_versions = None
        self._rowwise_fp8_up_gate_weight.invalidate()
        self._rowwise_fp8_down_weight.invalidate()

    @torch.no_grad()
    def refresh_rowwise_fp8_cache(self) -> None:
        cfg = self.rowwise_fp8
        if cfg is None or not cfg.enabled:
            self.invalidate_rowwise_fp8_cache()
            return
        self._sync_rowwise_fp8_weight_anchors()
        if cfg.fp8_only_params and (
            self._rowwise_fp8_up_gate_weight.anchor_storage_released
            or self._rowwise_fp8_down_weight.anchor_storage_released
        ):
            if (
                self._rowwise_fp8_up_gate_weight.prequantized_rhs is None
                or self._rowwise_fp8_up_gate_weight.prequantized_rhs_for_dgrad is None
                or self._rowwise_fp8_down_weight.prequantized_rhs is None
                or self._rowwise_fp8_down_weight.prequantized_rhs_for_dgrad is None
            ):
                raise RuntimeError(
                    "Cannot refresh rowwise FP8 routed expert caches from released bf16 anchors. "
                    "The optimizer must refresh MXFP8 stores directly from fp32 main params."
                )
            # See below: narrow the store's widened union back to the grouped-mm variant.
            self._rowwise_fp8_up_gate_prequant = cast(
                Optional[ScaledGroupedMMPrequantizedRHS],
                self._rowwise_fp8_up_gate_weight.prequantized_rhs,
            )
            self._rowwise_fp8_up_gate_prequant_t = cast(
                Optional[ScaledGroupedMMPrequantizedRHS],
                self._rowwise_fp8_up_gate_weight.prequantized_rhs_for_dgrad,
            )
            self._rowwise_fp8_down_prequant = cast(
                Optional[ScaledGroupedMMPrequantizedRHS],
                self._rowwise_fp8_down_weight.prequantized_rhs,
            )
            self._rowwise_fp8_down_prequant_t = cast(
                Optional[ScaledGroupedMMPrequantizedRHS],
                self._rowwise_fp8_down_weight.prequantized_rhs_for_dgrad,
            )
            self._rowwise_fp8_weight_versions = None
            return
        if self.w_up_gate.device.type != "cuda" or self.w_down.device.type != "cuda":
            self.invalidate_rowwise_fp8_cache()
            return
        self._rowwise_fp8_up_gate_weight.refresh_from_logical_weight(
            self.w_up_gate,
            version_tensors=(self.w_up_gate,),
        )
        self._rowwise_fp8_down_weight.refresh_from_logical_weight(
            self.w_down,
            version_tensors=(self.w_down,),
        )
        # The store's prequantized RHS type widened to a union (to also back MXFP8 linear); these
        # routed-expert stores only ever hold the grouped-mm variant, so narrow back to it.
        self._rowwise_fp8_up_gate_prequant = cast(
            Optional[ScaledGroupedMMPrequantizedRHS],
            self._rowwise_fp8_up_gate_weight.prequantized_rhs,
        )
        self._rowwise_fp8_up_gate_prequant_t = cast(
            Optional[ScaledGroupedMMPrequantizedRHS],
            self._rowwise_fp8_up_gate_weight.prequantized_rhs_for_dgrad,
        )
        self._rowwise_fp8_down_prequant = cast(
            Optional[ScaledGroupedMMPrequantizedRHS],
            self._rowwise_fp8_down_weight.prequantized_rhs,
        )
        self._rowwise_fp8_down_prequant_t = cast(
            Optional[ScaledGroupedMMPrequantizedRHS],
            self._rowwise_fp8_down_weight.prequantized_rhs_for_dgrad,
        )
        self._rowwise_fp8_weight_versions = (
            int(self.w_up_gate._version),
            int(self.w_down._version),
        )

    def _use_rowwise_fp8(self, x: torch.Tensor, *, enabled: bool) -> bool:
        if not enabled:
            return False
        cfg = self.rowwise_fp8
        if cfg is None or not cfg.enabled:
            return False
        if x.device.type != "cuda":
            return False
        if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return False
        return True

    def _forward_rowwise_fp8(
        self,
        x: torch.Tensor,
        batch_size_per_expert_tensor: torch.Tensor,
        *,
        prequantized_input_q: Optional[torch.Tensor] = None,
        prequantized_input_scales: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.activation != ExpertActivation.swiglu:
            raise RuntimeError("rowwise FP8 routed experts currently require swiglu activation")
        if self.b_up_gate is not None or self.b_down is not None:
            raise RuntimeError("rowwise FP8 routed experts do not support expert biases")
        cfg = self.rowwise_fp8
        assert cfg is not None

        up_gate_weight = self._rowwise_fp8_up_gate_weight
        down_weight = self._rowwise_fp8_down_weight
        up_gate_prequant = up_gate_weight.require_prequantized_rhs()
        up_gate_prequant_t = up_gate_weight.require_prequantized_rhs_for_dgrad()
        down_prequant = down_weight.require_prequantized_rhs()
        down_prequant_t = down_weight.require_prequantized_rhs_for_dgrad()
        offs = torch.cumsum(
            batch_size_per_expert_tensor.to(dtype=torch.int32), dim=0, dtype=torch.int32
        )
        prequantized_lhs = None
        if prequantized_input_q is not None and prequantized_input_scales is not None:
            prequantized_lhs = ScaledGroupedMMPrequantizedLHS(
                mat_a_q=prequantized_input_q,
                scale_a=prequantized_input_scales,
                mat_a_shape=tuple(x.shape),
                scales_are_blocked=False,
            )

        up_kwargs = dict(
            offs=offs,
            use_fast_accum=cfg.use_fast_accum,
            prequantized_lhs=prequantized_lhs,
            prequantized_rhs=up_gate_prequant,
            prequantized_rhs_for_dgrad=up_gate_prequant_t,
        )
        if cfg.fp8_only_params:
            up_kwargs["wgrad_sink"] = up_gate_weight
            up_kwargs["wgrad_sink_transpose_last2"] = True
        up_gate_anchor = self.w_up_gate.transpose(1, 2)
        if cfg.fp8_only_params:
            up_gate_anchor = up_gate_anchor.detach()
        up_gate = scaled_grouped_mm_q_fp8_weight_no_compile(
            x,
            up_gate_anchor,
            **up_kwargs,
        )
        up_gate = cast(torch.Tensor, up_gate)
        h, h_q, h_scales = swiglu_quantize_rows_mxfp8_autograd(
            up_gate,
            int(cfg.block_size),
        )
        h_prequantized_lhs = ScaledGroupedMMPrequantizedLHS(
            mat_a_q=h_q,
            scale_a=h_scales,
            mat_a_shape=tuple(h.shape),
            scales_are_blocked=False,
        )

        down_kwargs = dict(
            offs=offs,
            use_fast_accum=cfg.use_fast_accum,
            prequantized_lhs=h_prequantized_lhs,
            prequantized_rhs=down_prequant,
            prequantized_rhs_for_dgrad=down_prequant_t,
        )
        if cfg.fp8_only_params:
            down_kwargs["wgrad_sink"] = down_weight
        down_anchor = self.w_down.detach() if cfg.fp8_only_params else self.w_down
        down = scaled_grouped_mm_q_fp8_weight_no_compile(
            h,
            down_anchor,
            **down_kwargs,
        )

        return cast(torch.Tensor, down)

    # @torch.compiler.disable(recursive=False)
    @nvtx.annotate("RoutedExperts.forward", color="blue")
    def forward(
        self,
        x: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
        *,
        down_proj_out: Optional[torch.Tensor] = None,
        up_proj_input_grad_out: Optional[torch.Tensor] = None,
        row_weights: Optional[torch.Tensor] = None,
        use_rowwise_fp8: bool = False,
        rowwise_fp8_input_q: Optional[torch.Tensor] = None,
        rowwise_fp8_input_scales: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        `batch_size_per_expert` specifies the number of tokens in x for each expert.
        """

        assert isinstance(
            batch_size_per_expert, torch.Tensor
        ), "only accept Tensor for batch_size_per_expert"

        if (down_proj_out is not None or row_weights is not None) and (
            self.b_up_gate is not None or self.b_down is not None
        ):
            # The rowwise EP paths pass capacity-padded buffers with only the valid per-expert
            # counts, and the down output aliases a symmetric combine buffer. Expert-bias
            # expansion doesn't match that padded/aliased layout, so require bias-free experts.
            raise NotImplementedError(
                "Routed-expert biases are not supported on the rowwise EP path "
                "(down_proj_out / row_weights); use bias-free routed experts there."
            )

        use_rowwise_fp8_runtime = self._use_rowwise_fp8(x, enabled=use_rowwise_fp8)
        if requires_host_side_split_sizes() and not use_rowwise_fp8_runtime:
            # CPU-side split sizes are required by grouped_gemm cublas mode.
            # grouped_gemm CUTLASS mode can accept device-side split sizes, but it is slow.
            # Always assume grouped_gemm runs in cublas mode.
            assert (
                batch_size_per_expert.device.type == "cpu"
            ), "batch_size_per_expert must be on cpu"
            batch_size_per_expert_tensor = batch_size_per_expert.to(
                dtype=torch.int64
            )  # int64 required for grouped_gemm
        else:
            assert (
                batch_size_per_expert.device.type == "cuda"
            ), "batch_size_per_expert expected to be on GPU"
            # grouped_mm expects int32 offsets derived from split sizes.
            batch_size_per_expert_tensor = batch_size_per_expert.to(dtype=torch.int32)

        if x.numel() == 0:
            if down_proj_out is not None:
                return down_proj_out
            return x

        if use_rowwise_fp8_runtime:
            if row_weights is not None:
                raise RuntimeError(
                    "row_weights are not supported with rowwise FP8 routed experts yet"
                )
            if self.activation != ExpertActivation.swiglu:
                raise RuntimeError("rowwise FP8 routed experts currently require swiglu activation")
            return self._forward_rowwise_fp8(
                x,
                batch_size_per_expert_tensor,
                prequantized_input_q=rowwise_fp8_input_q,
                prequantized_input_scales=rowwise_fp8_input_scales,
            )
        if (
            self.rowwise_fp8 is not None
            and self.rowwise_fp8.enabled
            and self.rowwise_fp8.fp8_only_params
            and (
                self._rowwise_fp8_up_gate_weight.anchor_storage_released
                or self._rowwise_fp8_down_weight.anchor_storage_released
            )
        ):
            raise RuntimeError(
                "RoutedExperts with fp8_only_params=True cannot fall back to bf16 grouped-mm "
                "after bf16 expert anchor storage has been released"
            )

        w_up_gate = self.w_up_gate  # (E, H, 2D)
        w_down = self.w_down  # (E, H, D)

        projection: Callable[..., Any] = gmm
        if self._profile_rounded_wgrad and torch.is_grad_enabled():
            from olmo_core.ops.rounded_wgrad import rounded_weight_gmm

            ep_qualification = os.environ.get("OLMO_PROFILE_ROUNDED_WGRAD_EP", "0") == "1"
            if not ep_qualification and (
                self.ep_dim != 1 or down_proj_out is not None or up_proj_input_grad_out is not None
            ):
                raise RuntimeError(
                    "Rounded weight-gradient EP/buffer support requires the explicit "
                    "OLMO_PROFILE_ROUNDED_WGRAD_EP qualification flag"
                )

            def rounded_projection(a, b, counts, trans_b=False, **kwargs):
                return rounded_weight_gmm(a, b, counts, trans_b, **kwargs)

            projection = rounded_projection

        # up (+ gate) projection
        up_gate = projection(
            x,
            w_up_gate,
            batch_size_per_expert_tensor,
            trans_b=True,
            input_grad_out=up_proj_input_grad_out,
        )  # -> (BS, 2H)

        up_gate = cast(torch.Tensor, up_gate)  # ensure type is Tensor
        if self.b_up_gate is not None:
            up_gate = up_gate + self._expand_expert_bias(
                self.b_up_gate,
                batch_size_per_expert_tensor,
                output_size=up_gate.shape[0],
            )

        num_valid_rows = batch_size_per_expert_tensor.sum()
        h = self.chunk_and_activate(up_gate, num_elements=num_valid_rows)  # -> (BS, H)
        if row_weights is not None:
            if row_weights.numel() != h.shape[0]:
                raise RuntimeError(
                    "row_weights must have one value per routed row: "
                    f"weights={tuple(row_weights.shape)} rows={h.shape[0]}"
                )
            h = h * row_weights.reshape(-1, 1).to(dtype=h.dtype)

        # down projection
        down = projection(
            h,
            w_down,
            batch_size_per_expert_tensor,
            trans_b=False,
            out=down_proj_out,
        )  # -> (BS, D)
        if self.b_down is not None:
            down = cast(torch.Tensor, down) + self._expand_expert_bias(
                self.b_down,
                batch_size_per_expert_tensor,
                output_size=down.shape[0],
            )

        return cast(torch.Tensor, down)  # ensure type is Tensor

    @nvtx.annotate("RoutedExperts.forward_row_offset", color="blue")
    def forward_row_offset(
        self,
        x: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
        *,
        expert_start: int,
        expert_end: int,
        row_start: torch.Tensor,
        down_proj_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward a contiguous local-expert window in its canonical rowwise-EP rows.

        ``row_start`` is a CUDA scalar holding the first canonical row for
        ``expert_start``. The grouped-MM data-prep kernel shifts A/C pointers by
        that scalar, so no activation repack is needed for wave execution.
        """
        if torch.is_grad_enabled():
            raise RuntimeError("forward_row_offset is forward-only and requires no_grad")
        if requires_host_side_split_sizes():
            raise RuntimeError("forward_row_offset requires the torch grouped_mm backend")
        if not use_torch_grouped_mm():
            raise RuntimeError("forward_row_offset requires torch grouped_mm support")
        if self.activation != ExpertActivation.swiglu:
            raise RuntimeError("forward_row_offset currently requires swiglu activation")
        if self.b_up_gate is not None or self.b_down is not None:
            raise RuntimeError("forward_row_offset does not support expert biases")
        if self.rowwise_fp8 is not None and self.rowwise_fp8.enabled:
            raise RuntimeError("forward_row_offset does not support rowwise FP8 experts yet")
        if not x.is_cuda:
            raise RuntimeError("forward_row_offset requires CUDA input")
        if batch_size_per_expert.device != x.device:
            raise RuntimeError("batch_size_per_expert must be on the same CUDA device as x")
        if row_start.device != x.device or row_start.numel() != 1:
            raise RuntimeError("row_start must be a CUDA scalar on the same device as x")
        if expert_start < 0 or expert_end > self.num_local_experts or expert_start >= expert_end:
            raise RuntimeError(
                f"invalid expert range [{expert_start}, {expert_end}) for {self.num_local_experts} local experts"
            )

        batch_size_per_expert_tensor = batch_size_per_expert.to(dtype=torch.int32)
        batch_size_window = batch_size_per_expert_tensor[int(expert_start) : int(expert_end)]
        num_valid_rows = batch_size_window.sum()

        up_gate = torch.empty(
            (x.shape[0], self.w_up_gate.shape[1]),
            device=x.device,
            dtype=x.dtype,
        )
        grouped_mm_row_offset(
            x,
            self.w_up_gate[int(expert_start) : int(expert_end)].transpose(1, 2),
            batch_size_window,
            row_start=row_start,
            out=up_gate,
        )

        h = self.chunk_and_activate(
            up_gate,
            num_elements=num_valid_rows,
            start=row_start,
        )
        if down_proj_out is None:
            down_proj_out = torch.empty(
                (x.shape[0], self.w_down.shape[2]),
                device=x.device,
                dtype=x.dtype,
            )
        grouped_mm_row_offset(
            h,
            self.w_down[int(expert_start) : int(expert_end)],
            batch_size_window,
            row_start=row_start,
            out=down_proj_out,
        )
        return down_proj_out

    def act_and_down(
        self, up_gate: torch.Tensor, batch_size_per_expert_tensor: torch.Tensor
    ) -> torch.Tensor:
        # swiglu + down projection
        # so that it apply activation checkpointing if needed
        if self.b_up_gate is not None:
            up_gate = up_gate + self._expand_expert_bias(
                self.b_up_gate,
                batch_size_per_expert_tensor,
                output_size=up_gate.shape[0],
            )
        num_valid_rows = batch_size_per_expert_tensor.sum()
        h = self.chunk_and_activate(up_gate, num_elements=num_valid_rows)  # -> (BS, H)

        down = gmm(h, self.w_down, batch_size_per_expert_tensor, trans_b=False)  # -> (BS, H)
        if self.b_down is not None:
            down = down + self._expand_expert_bias(
                self.b_down,
                batch_size_per_expert_tensor,
                output_size=down.shape[0],
            )
        return down

    @staticmethod
    def _expand_expert_bias(
        bias: torch.Tensor,
        batch_size_per_expert_tensor: torch.Tensor,
        *,
        output_size: Optional[int] = None,
    ) -> torch.Tensor:
        repeats = batch_size_per_expert_tensor.to(device=bias.device, dtype=torch.long)
        return torch.repeat_interleave(bias, repeats, dim=0, output_size=output_size)

    def chunk_and_activate(
        self,
        up_gate: torch.Tensor,
        *,
        num_elements: Optional[torch.Tensor] = None,
        start: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        # Forward-only EP can skip padded tail rows with a custom kernel while
        # keeping training on the autograd-backed PyTorch activation.
        if self.activation == ExpertActivation.swiglu:
            if (
                self._profile_pairwise_swiglu
                and torch.compiler.is_compiling()
                and torch.is_grad_enabled()
                and up_gate.is_cuda
                and up_gate.dtype == torch.bfloat16
                and up_gate.ndim == 2
                and up_gate.is_contiguous()
            ):
                from olmo_core.ops.swiglu_pairwise import pairwise_swiglu

                return pairwise_swiglu(up_gate)
            if (
                num_elements is not None
                and up_gate.is_cuda
                and num_elements.device == up_gate.device
                and not torch.is_grad_enabled()
            ):
                return swiglu_valid_prefix(
                    up_gate,
                    num_elements,
                    start=start,
                    match_eager_rounding=self.match_eager_rounding,
                    row_specialization=self.row_specialization,
                )
            up, gate = up_gate.chunk(2, dim=-1)
            h = up * F.silu(gate)  # -> (BS, H)
        elif self.activation == ExpertActivation.gpt_oss_swiglu:
            gate = up_gate[..., ::2]
            up = up_gate[..., 1::2]
            if self.activation_limit is not None:
                gate = gate.clamp(max=self.activation_limit)
                up = up.clamp(min=-self.activation_limit, max=self.activation_limit)
            h = (up + 1.0) * gate * torch.sigmoid(gate * self.activation_alpha)
        elif self.activation == ExpertActivation.relu2:
            h = F.relu(up_gate).square()
        else:
            raise NotImplementedError(self.activation)
        return h

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        # shard dim 0 to ep_mp, replicate on ep_dp mesh
        self.ep_mesh = ep_mesh["ep_dp", "ep_mp"]
        # with torch.no_grad():  # just to avoid tracking the rebind below
        self.ep_dim = ep_mesh["ep_mp"].size()
        self.ep_rank = ep_mesh["ep_mp"].get_local_rank()

        assert (
            self.num_experts % self.ep_dim == 0
        ), "num_experts must be divisible by the number of expert partitions"
        self.num_local_experts = self.num_experts // self.ep_dim

        up_factor = (
            2
            if self.activation in (ExpertActivation.swiglu, ExpertActivation.gpt_oss_swiglu)
            else 1
        )
        self.w_up_gate = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                up_factor * self.hidden_size,
                self.d_model,
                dtype=self.w_up_gate.dtype,
                device=self.w_up_gate.device,
            ),
        )

        self.w_down = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.hidden_size,
                self.d_model,
                dtype=self.w_down.dtype,
                device=self.w_down.device,
            ),
        )
        if self.bias:
            assert self.b_up_gate is not None and self.b_down is not None
            self.b_up_gate = nn.Parameter(
                torch.empty(
                    self.num_local_experts,
                    up_factor * self.hidden_size,
                    dtype=self.b_up_gate.dtype,
                    device=self.b_up_gate.device,
                ),
            )
            self.b_down = nn.Parameter(
                torch.empty(
                    self.num_local_experts,
                    self.d_model,
                    dtype=self.b_down.dtype,
                    device=self.b_down.device,
                ),
            )
        owner_ref = weakref.ref(self)
        self.w_up_gate._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
        self.w_down._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
        self._rowwise_fp8_up_gate_weight = FP8WeightStore(
            logical_name="w_up_gate",
            logical_shape=tuple(self.w_up_gate.shape),
            cache_specs=_ROUTED_UP_GATE_FP8_CACHE_SPECS,
            anchor_param=self.w_up_gate,
            optimizer_enabled=(
                self.rowwise_fp8 is not None
                and self.rowwise_fp8.enabled
                and self.rowwise_fp8.fp8_only_params
            ),
        )
        self._rowwise_fp8_down_weight = FP8WeightStore(
            logical_name="w_down",
            logical_shape=tuple(self.w_down.shape),
            cache_specs=_ROUTED_DOWN_FP8_CACHE_SPECS,
            anchor_param=self.w_down,
            optimizer_enabled=(
                self.rowwise_fp8 is not None
                and self.rowwise_fp8.enabled
                and self.rowwise_fp8.fp8_only_params
            ),
        )
        self.invalidate_rowwise_fp8_cache()

        self._ep_sharded = True

    def extra_repr(self):
        return f"num_experts={self.num_experts}, hidden_size={self.hidden_size}, d_model={self.d_model}"
