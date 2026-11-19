from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from olmo_core.nn.attention.flash_linear_attn_api import dispatch_causal_conv1d

__all__ = ["CausalConv1d", "int32_safe_tokens"]

#: Largest element offset a 32-bit index can hold. fla's Triton kernels (0.4.x) compute the
#: within-row offset ``t * row_width`` in int32, so a (seq_len x row_width) operand must stay below it.
_INT32_LIMIT = 2**31 - 1


def int32_safe_tokens(row_width: int, multiple: int = 1024, limit: int = _INT32_LIMIT) -> int:
    """The longest segment (a multiple of ``multiple`` tokens) whose ``seq_len * row_width`` offsets stay below ``limit``."""
    return max(multiple, (int(limit) // max(1, int(row_width))) // multiple * multiple)


class CausalConv1d(nn.Conv1d):
    """
    CausalConv1d (aka short convolution) layer for efficient causal convolution operations.
    This implements a depthwise separable 1D convolution with causal padding.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        backend: Literal["triton", "cuda"] = "triton",
        dtype: torch.dtype | None = None,
        init_device: str = "cpu",
        activation: Literal["silu", "swish"] | None = "silu",
        use_experimental_kernels: bool = False,
    ):
        """
        :param hidden_size: Number of input/output channels (must be equal for depthwise conv).
        :param kernel_size: Size of the convolution kernel.
        :param bias: Whether to include learnable bias.
        :param backend: Backend implementation ('triton' or 'cuda').
        :param dtype: The data type of the convolution weights and bias.
        :param init_device: The device to initialize the parameters on, e.g. "cpu", "meta".
        :param activation: Activation function ('silu' or 'swish').
        :param use_experimental_kernels: **Experimental.** Whether to run the fused short-conv kernels
            from the ``kernel-fun`` package instead of FLA's, see
            :func:`~olmo_core.nn.attention.flash_linear_attn_api.dispatch_causal_conv1d`.
            Requires the package, installed with the ``kernel-fun`` extra.
        """
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
            device=init_device,
            dtype=dtype,
        )
        self.hidden_size = hidden_size
        self.backend = backend
        self.activation = activation
        self.use_experimental_kernels = use_experimental_kernels
        self.cp_enabled = False

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            ``batch_size`` must be 1 if ``cu_seqlens`` is provided.
            When CP is enabled, input should be channel-parallel: ``(batch_size, seq_len, hidden_size/CP)``.
        :param cu_seqlens: Cumulative sequence lengths for variable-length sequences.
            Shape: ``(num_seqs + 1,)``.
        :returns: Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
            When CP is enabled, output is channel-parallel: ``(batch_size, seq_len, hidden_size/CP)``.
        """
        weight = self.weight
        bias = self.bias

        if self.cp_enabled:
            weight = weight[self._cp_channel_slice]
            if bias is not None:
                bias = bias[self._cp_channel_slice]

        seg = int32_safe_tokens(x.shape[-1]) if getattr(self, "_int32_safe", False) else 0
        if seg and cu_seqlens is None and x.dim() == 3 and x.shape[1] > seg:
            # Long-sequence guard (``enable_int32_safe_long_seq``): fla's conv kernel indexes a row
            # with 32-bit offsets (t * C), which wrap once seq_len * channels >= 2^31 (Qwen3.5-4B's
            # fused q|k|v conv: 8192 channels -> 262,144 tokens) and silently read garbage. Run it
            # on segments below that bound, each with the previous ``kernel_size - 1`` inputs
            # prepended and their outputs dropped, which is exactly the full causal convolution.
            W = self.kernel_size[0]
            outs = []
            T = x.shape[1]
            for s0 in range(0, T, seg):
                a = max(0, s0 - (W - 1))
                y = dispatch_causal_conv1d(
                    x=x[:, a : min(s0 + seg, T)],
                    weight=weight.squeeze(1),
                    bias=bias,
                    activation=self.activation,
                    backend=self.backend,
                    cu_seqlens=None,
                    use_experimental_kernels=self.use_experimental_kernels,
                )[0]
                outs.append(y[:, s0 - a :])
            return torch.cat(outs, dim=1)

        output = dispatch_causal_conv1d(
            x=x,
            weight=weight.squeeze(1),
            bias=bias,
            activation=self.activation,
            backend=self.backend,
            cu_seqlens=cu_seqlens,
            use_experimental_kernels=self.use_experimental_kernels,
        )
        return output[0]

    def forward_soft_keep(self, x: torch.Tensor, keep: torch.Tensor, chunk: int = 128) -> torch.Tensor:
        """
        The causal conv under DIFFERENTIABLE token removal (``GatedDeltaNet(soft_keep=...)``).

        Removing token ``s`` from the row changes every later token's window: it holds the previous
        ``kernel_size - 1`` *kept* tokens. That window is written with registers
        ``R_r(t) = p_t * R_{r-1}(t-1) + (1 - p_t) * R_r(t-1)`` (``R_0(t) = x_t``), i.e. ``R_r(t)`` is the
        r-th most recent kept input at or before ``t``, and the output is
        ``act(w[K-1] x_t + sum_r w[K-1-r] R_r(t-1) + bias)``. For ``p in {0, 1}`` this is EXACTLY the
        conv of the compacted row at the kept positions; in between it interpolates smoothly and is
        differentiable in ``p``. Each register is a first-order scalar-gated scan, computed in
        ``chunk``-sized blocks with non-positive exponents only (stable); activation-checkpointed so
        the (T, C) float32 intermediates are not kept for backward. No CP / ``cu_seqlens`` support.

        :param x: ``(B, T, C)`` conv input.
        :param keep: ``(B, T)`` keep probability in ``[0, 1]``.
        :returns: ``(B, T, C)`` in ``x.dtype``.
        """
        from torch.utils.checkpoint import checkpoint

        weight, bias = self._local_weight_bias()
        return checkpoint(_soft_keep_conv, x, keep, weight.squeeze(1), bias, self.activation, chunk, use_reentrant=False)

    @property
    def state_width(self) -> int:
        """The width of the cached conv state needed for single-step decoding: ``kernel_size - 1``."""
        return self.kernel_size[0] - 1

    def _local_weight_bias(self):
        weight = self.weight
        bias = self.bias
        if self.cp_enabled:
            weight = weight[self._cp_channel_slice]
            if bias is not None:
                bias = bias[self._cp_channel_slice]
        return weight, bias

    def prefill_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Capture the conv state for cached decoding from a prefill input.

        :param x: The prefill conv input of shape ``(batch_size, seq_len, hidden_size)`` (the same
            tensor passed to :meth:`forward`).
        :returns: The last ``kernel_size - 1`` inputs, channel-first and left-padded with zeros if
            ``seq_len < kernel_size - 1``, of shape ``(batch_size, hidden_size, kernel_size - 1)``.
        """
        w = self.state_width
        xt = x.transpose(1, 2)  # (B, hidden, seq_len)
        if xt.shape[-1] < w:
            xt = F.pad(xt, (w - xt.shape[-1], 0))
        return xt[:, :, -w:].contiguous()

    def forward_with_state(self, x: torch.Tensor, conv_state: torch.Tensor) -> torch.Tensor:
        """
        Multi-token causal convolution that *continues* from a cached window, for chunked prefill.

        :meth:`forward` left-pads with zeros, which is correct only when ``x`` starts at position 0.
        Feeding a long prompt in chunks means every chunk after the first must instead see the real
        preceding ``kernel_size - 1`` inputs, or each chunk boundary silently gets a zero-padded
        convolution. This prepends the cached window, convolves, and drops the window's outputs, so
        the result is identical to convolving the whole prompt at once. ``conv_state`` is updated in
        place to this chunk's trailing window, ready for the next chunk (or for :meth:`step`).

        :param x: Conv input of shape ``(batch_size, seq_len, hidden_size)``.
        :param conv_state: Cached window of shape ``(batch_size, hidden_size, kernel_size - 1)``,
            updated in place.

        :returns: Conv output of shape ``(batch_size, seq_len, hidden_size)``.
        """
        w = self.state_width
        if w == 0:
            return self.forward(x)
        prefix = conv_state.transpose(1, 2).to(x.dtype)  # (B, kernel_size - 1, hidden)
        # The new window is the tail of [prefix ; x], not of x alone: a chunk shorter than the
        # window still has to carry the older inputs forward.
        combined = torch.cat([prefix, x], dim=1)
        conv_state.copy_(self.prefill_state(combined))
        # dispatch_causal_conv1d zero-pads the front, so the first ``w`` outputs belong to the
        # prepended window and are discarded; the rest are exactly the chunk's outputs.
        return self.forward(combined)[:, w:]

    def step(self, x_t: torch.Tensor, conv_state: torch.Tensor) -> torch.Tensor:
        """
        Single-step causal convolution for cached decoding, updating ``conv_state`` in place.

        Mirrors the reference ``causal_conv1d_update``: the new input is concatenated onto the
        cached window, convolved, and the trailing ``kernel_size - 1`` inputs are written back.

        :param x_t: The new conv input of shape ``(batch_size, 1, hidden_size)``.
        :param conv_state: The cached window of shape ``(batch_size, hidden_size, kernel_size - 1)``,
            updated in place.
        :returns: The conv output of shape ``(batch_size, 1, hidden_size)``.
        """
        weight, bias = self._local_weight_bias()
        hidden_size = weight.shape[0]
        xt = x_t.transpose(1, 2)  # (B, hidden, 1)
        window = torch.cat([conv_state, xt], dim=-1)  # (B, hidden, kernel_size)
        conv_state.copy_(window[:, :, -self.state_width :])
        out = F.conv1d(
            window.to(weight.dtype), weight, bias, padding=0, groups=hidden_size
        )  # (B, hidden, 1)
        if self.activation in ("silu", "swish"):
            out = F.silu(out)
        return out.transpose(1, 2).to(x_t.dtype)  # (B, 1, hidden)

    def apply_cp(self, cp_mesh: DeviceMesh):
        """
        Configure convolution for Ulysses-style (channel-parallel) context parallelism.

        Instead of sharding parameters (which conflicts with ``FSDP``), we keep the full
        parameters and slice to the local ``C/CP`` channels during forward based on CP rank.
        Since convolutions tend to have a small number of parameters, the extra memory overhead
        of keeping the full parameters on each rank is minimal.

        :param cp_mesh: The context parallel device mesh.
        """
        if cp_mesh.size() == 1:
            return

        local_channels = self.hidden_size // cp_mesh.size()
        start = cp_mesh.get_local_rank() * local_channels
        self._cp_channel_slice = slice(start, start + local_channels)
        self.cp_enabled = True


def _gated_scan(a_log: torch.Tensor, b: torch.Tensor, x: torch.Tensor, chunk: int) -> torch.Tensor:
    """``h_t = exp(a_log_t) * h_{t-1} + b_t * x_t`` (``h_{-1} = 0``) over ``(B, T, C)`` float32, blockwise:
    intra-block decay matrices ``exp(cum_t - cum_s)`` (s <= t) plus a block-level carry matmul."""
    B, T, C = x.shape
    pad = (-T) % chunk
    if pad:
        a_log = F.pad(a_log, (0, pad))
        b = F.pad(b, (0, pad))
        x = F.pad(x, (0, 0, 0, pad))
    n = (T + pad) // chunk
    cum = a_log.view(B, n, chunk).cumsum(-1)  # (B, n, L), non-increasing
    tri = torch.ones(chunk, chunk, dtype=torch.bool, device=x.device).tril()
    D = torch.exp((cum[..., :, None] - cum[..., None, :]).masked_fill(~tri, float("-inf")))  # (B, n, L, L)
    intra = D @ (b[..., None] * x).view(B, n, chunk, C)  # (B, n, L, C)
    g_end = cum[..., -1].cumsum(-1)  # (B, n) total log-decay through the end of each block
    trib = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
    E = torch.exp((g_end[..., :, None] - g_end[..., None, :]).masked_fill(~trib, float("-inf")))  # (B, n, n)
    carry = E @ intra[:, :, -1]  # (B, n, C) state at the end of each block
    h_prev = F.pad(carry, (0, 0, 1, 0))[:, :-1]  # state entering each block
    out = intra + torch.exp(cum)[..., None] * h_prev[:, :, None, :]
    return out.reshape(B, n * chunk, C)[:, :T]


def _soft_keep_conv(x, keep, w, bias, activation, chunk):
    K = w.shape[-1]
    dt = torch.float64 if x.dtype == torch.float64 else torch.float32
    p = keep.to(dt)
    a_log = torch.log(torch.clamp(1.0 - p, min=1e-6))  # kept (p=1) -> ~no carry-over; dropped (p=0) -> 0
    src = x.to(dt)
    wf = w.to(dt)
    y = src * wf[:, K - 1]
    for r in range(1, K):
        reg = _gated_scan(a_log, p, src, chunk)  # r-th most recent kept input at or before t
        reg_prev = F.pad(reg, (0, 0, 1, 0))[:, :-1]  # ... before t
        y = y + reg_prev * wf[:, K - 1 - r]
        src = reg_prev
    if bias is not None:
        y = y + bias.to(dt)
    if activation in ("silu", "swish"):
        y = F.silu(y)
    elif activation is not None:
        raise ValueError(activation)
    return y.to(x.dtype)
