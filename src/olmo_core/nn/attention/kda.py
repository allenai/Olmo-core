"""Native Olmo-core integration for Kimi Delta Attention (KDA)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Placement

from olmo_core.config import DType
from olmo_core.distributed.parallel.context_parallel import (
    all_to_all_cp2hp,
    all_to_all_single_cp2hp,
    all_to_all_single_hp2cp,
)
from olmo_core.nn.attention.base import SequenceMixer, SequenceMixerConfig
from olmo_core.nn.attention.flash_linear_attn_api import (
    dispatch_chunk_kda,
    has_fla,
    require_kernel_fun,
)
from olmo_core.nn.attention.ring import (
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.nn.buffer_cache import BufferCache
from olmo_core.nn.convolution import CausalConv1d
from olmo_core.nn.feed_forward import ActivationFunction
from olmo_core.utils import log_once

if TYPE_CHECKING:
    from olmo_core.nn.transformer.init import InitMethod

log = logging.getLogger(__name__)


class KimiDeltaAttention(SequenceMixer):
    """Kimi Delta Attention using Moonshot's released Triton training kernel.

    This follows the parameterization of the released Kimi-Linear checkpoint:
    KDA uses a vector-valued decay for every key channel and a scalar delta
    gate for every value head. The surrounding Olmo-core adapter retains packed
    document convolutions, initialization, and sequence-mixer interfaces.

    .. warning::
        ``use_experimental_kernels`` opts both the KDA chunk kernel and the short
        convolutions into the **experimental** kernels from the ``kernel-fun`` package. See
        :class:`KimiDeltaAttentionConfig` for what opting in entails.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_v_heads: int | None = None,
        head_dim: int | None = None,
        expand_v: float = 1.0,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        use_experimental_kernels: bool = False,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ) -> None:
        super().__init__()
        if not has_fla():
            raise RuntimeError(
                "KimiDeltaAttention requires flash-linear-attention with fla.ops.kda"
            )
        if use_experimental_kernels:
            require_kernel_fun()
        from fla.modules import FusedRMSNormGated

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_v_heads = n_v_heads if n_v_heads is not None else n_heads
        self.head_dim = head_dim if head_dim is not None else d_model // n_heads
        self.expand_v = expand_v
        self.allow_neg_eigval = allow_neg_eigval
        self.conv_size = conv_size
        self.use_experimental_kernels = use_experimental_kernels
        if use_experimental_kernels:
            log_once(
                log,
                "KDA is running with the EXPERIMENTAL kernels from kernel-fun "
                "(use_experimental_kernels=True): the KDA chunk kernel and the fused "
                "short-conv kernels behind the Q/K/V convolutions. These are new and are "
                "not numerically identical to FLA's kernels. The KDA kernel only engages "
                "on Blackwell at chunk size 64 without packed-document cu_seqlens; every "
                "other shape falls back to FLA — which the kernels log, with the reason, "
                "once per process. See the kernel_fun.kda and kernel_fun.cconv packages "
                "for the supported box.",
                level=logging.WARNING,
            )

        self.head_k_dim = self.head_dim
        self.head_v_dim = int(self.head_dim * expand_v)
        self.key_dim = self.n_heads * self.head_k_dim
        self.value_dim = self.n_v_heads * self.head_v_dim
        self.gate_dim = self.n_heads * self.head_k_dim

        if not math.isclose(self.head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError("expand_v must produce an integer value-head dimension")
        if self.n_v_heads != self.n_heads:
            raise ValueError("the pinned Kimi reference layer requires n_v_heads == n_heads")
        if self.head_k_dim > 256:
            raise ValueError("FLA's KDA chunk kernel requires head_dim <= 256")

        factory = {"dtype": dtype, "device": init_device}
        self.w_q = nn.Linear(d_model, self.key_dim, bias=False, **factory)
        self.w_k = nn.Linear(d_model, self.key_dim, bias=False, **factory)
        self.w_v = nn.Linear(d_model, self.value_dim, bias=False, **factory)

        # Kimi's low-rank projection produces one decay logit per key channel.
        self.f_proj_1 = nn.Linear(d_model, self.head_v_dim, bias=False, **factory)
        self.f_proj_2 = nn.Linear(self.head_v_dim, self.gate_dim, bias=False, **factory)
        self.w_b = nn.Linear(d_model, self.n_heads, bias=False, **factory)
        self.A_log = nn.Parameter(
            torch.empty(self.n_heads, dtype=torch.float32, device=init_device)
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.gate_dim, dtype=torch.float32, device=init_device)
        )

        self.q_conv1d = CausalConv1d(
            hidden_size=self.key_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            activation=ActivationFunction.silu.value,
            dtype=dtype,
            init_device=init_device,
            use_experimental_kernels=use_experimental_kernels,
        )
        self.k_conv1d = CausalConv1d(
            hidden_size=self.key_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            activation=ActivationFunction.silu.value,
            dtype=dtype,
            init_device=init_device,
            use_experimental_kernels=use_experimental_kernels,
        )
        self.v_conv1d = CausalConv1d(
            hidden_size=self.value_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            activation=ActivationFunction.silu.value,
            dtype=dtype,
            init_device=init_device,
            use_experimental_kernels=use_experimental_kernels,
        )

        self.g_proj_1 = nn.Linear(d_model, self.head_v_dim, bias=False, **factory)
        self.g_proj_2 = nn.Linear(self.head_v_dim, self.value_dim, bias=True, **factory)
        self.o_norm = FusedRMSNormGated(
            self.head_v_dim,
            eps=norm_eps,
            activation="sigmoid",
            device=torch.device(init_device),
            dtype=dtype,
        )
        self.w_out = nn.Linear(self.value_dim, d_model, bias=False, **factory)

        # Ulysses context-parallel state, populated by :meth:`apply_cp`.
        self.cp_enabled = False
        self.uly: UlyssesContextParallelStyle | None = None
        self._cp_mesh: DeviceMesh | None = None
        self._cp_group: torch.distributed.ProcessGroup | None = None
        self._cp_head_slice = slice(None)
        self._cp_gate_slice = slice(None)

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        batch_size, seq_len, _ = x.shape
        output_shape = (batch_size, seq_len, self.d_model)
        if self.cp_enabled and cu_doc_lens is not None and batch_size > 1:
            raise RuntimeError(
                "Batch size must equal 1 when using context parallelism with "
                f"intra-document masking (got {batch_size} instances)"
            )
        if cu_doc_lens is not None and batch_size > 1:
            # FLA's variable-length KDA and causal-convolution kernels represent packed
            # sequences as a single flattened batch, with document boundaries supplied by
            # ``cu_seqlens``. Olmo-core's mask builder likewise flattens boundaries across the
            # entire batch, so flatten the payload here and restore its shape after KDA.
            x = x.reshape(1, batch_size * seq_len, self.d_model)
            batch_size, seq_len = 1, batch_size * seq_len

        # Per-token projections run on this rank's local tokens with all heads.
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        raw_decay = self.f_proj_2(self.f_proj_1(x))
        beta = self.w_b(x).float().sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        A_log, dt_bias = self.A_log, self.dt_bias
        if self.cp_enabled and self.uly is not None:
            assert self._cp_group is not None
            # Ulysses CP: switch from sequence-parallel to head-parallel so that the
            # convolutions and the recurrent scan see the full sequence for a slice of heads.
            # q, k, and raw_decay share ``key_dim`` channels, so they ride one collective.
            # [B, T_local, C] -> [B, T_total, C/CP]
            q, k, raw_decay = all_to_all_cp2hp([q, k, raw_decay], self._cp_group)
            v = all_to_all_single_cp2hp(v, self._cp_group)
            # [B, T_local, H] -> [B, T_total, H/CP]
            beta = all_to_all_single_cp2hp(beta, self._cp_group)
            # The gate is applied inside the kernel, after the exchange, so its per-head and
            # per-channel parameters must be restricted to this rank's heads.
            A_log = A_log[self._cp_head_slice]
            dt_bias = dt_bias[self._cp_gate_slice]

        q = self.q_conv1d(x=q, cu_seqlens=cu_doc_lens)
        k = self.k_conv1d(x=k, cu_seqlens=cu_doc_lens)
        v = self.v_conv1d(x=v, cu_seqlens=cu_doc_lens)

        # Under CP this is the full sequence length and the head count below is local.
        T = q.size(1)
        q = q.view(batch_size, T, -1, self.head_k_dim)
        k = k.view(batch_size, T, -1, self.head_k_dim)
        v = v.view(batch_size, T, -1, self.head_v_dim)
        raw_decay = raw_decay.view(batch_size, T, -1, self.head_k_dim)

        # No kernel-fun version log here. The package logs `kernel_fun.versions()` itself,
        # once per process, from inside its `torch.compiler.disable`d entry points — free,
        # where this frame is compiled and a call here cost two graph breaks. It also
        # raised `TypeError: unhashable type: 'dict'`: `log_once` is `lru_cache`d and
        # `versions()` returns a dict.
        o, _ = dispatch_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=raw_decay,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            cu_seqlens=cu_doc_lens,
            use_experimental_kernels=self.use_experimental_kernels,
        )

        if self.cp_enabled and self.uly is not None:
            assert self._cp_group is not None
            # [B, T_total, H/CP, D] -> [B, T_local, H, D]
            o = all_to_all_single_hp2cp(o, self._cp_group)

        output_gate = self.g_proj_2(self.g_proj_1(x)).view(
            batch_size, seq_len, self.n_v_heads, self.head_v_dim
        )
        return self.w_out(self.o_norm(o, output_gate).view(batch_size, seq_len, -1)).view(
            output_shape
        )

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Placement | None = None,
        output_layout: Placement | None = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ) -> None:
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("Tensor parallelism is not yet implemented for KDA")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: RingContextParallelStyle | None = None,
        uly: UlyssesContextParallelStyle | None = None,
    ) -> None:
        """
        Prepare the layer for Ulysses-style context parallelism.

        Each rank keeps its contiguous slice of the sequence for the per-token projections.
        Inside `forward`, an all-to-all switches to head parallelism so the causal
        convolutions and the recurrent kernel run over the full sequence for
        ``n_heads / cp_degree`` heads, and a second all-to-all switches back.

        :param cp_mesh: The context parallel device mesh.
        :param ring: Ring context parallelism, which is not supported for KDA.
        :param uly: The Ulysses context parallel style.
        """
        if ring is not None:
            raise NotImplementedError(
                "Ring context parallelism is not supported for KimiDeltaAttention"
            )
        assert uly is not None

        cp_world_size = cp_mesh.size()
        if cp_world_size == 1:
            return

        # Ulysses CP partitions heads across ranks. Every sliced quantity (key/value/gate
        # channels) is ``n_heads`` times a head dim, so this one check covers them all.
        if self.n_heads % cp_world_size != 0:
            raise ValueError(
                f"KimiDeltaAttention n_heads ({self.n_heads}) must be divisible by the "
                f"context parallel degree ({cp_world_size})"
            )

        cp_rank = cp_mesh.get_local_rank()
        local_heads = self.n_heads // cp_world_size
        local_gate = self.gate_dim // cp_world_size
        self._cp_head_slice = slice(cp_rank * local_heads, (cp_rank + 1) * local_heads)
        self._cp_gate_slice = slice(cp_rank * local_gate, (cp_rank + 1) * local_gate)

        self.uly = uly
        self._cp_mesh = cp_mesh
        self._cp_group = cp_mesh.get_group()
        self.cp_enabled = True

        self.q_conv1d.apply_cp(cp_mesh)
        self.k_conv1d.apply_cp(cp_mesh)
        self.v_conv1d.apply_cp(cp_mesh)

    @torch.no_grad()
    def init_weights(
        self,
        *,
        init_method: InitMethod,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: torch.Generator | None = None,
    ) -> None:
        from olmo_core.nn.transformer.init import InitMethod, init_linear

        if init_method == InitMethod.fan_in:
            raise NotImplementedError(f"init method '{init_method}' is not supported for KDA")
        if init_method == InitMethod.normalized:
            std = d_model**-0.5

        for linear in (
            self.w_q,
            self.w_k,
            self.w_v,
            self.f_proj_1,
            self.f_proj_2,
            self.w_b,
            self.g_proj_1,
            self.g_proj_2,
        ):
            init_linear(linear, std=std, generator=generator)
        for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
            init_linear(conv, std=std, generator=generator)

        self.A_log.copy_(nn.init.uniform_(self.A_log, a=1, b=16, generator=generator).log())
        # Match FLA 0.4.1's released KDA layer initialization.
        self.dt_bias.zero_()
        self.o_norm.reset_parameters()

        if init_method == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5
        elif init_method == InitMethod.normalized:
            std = std / (2 * num_blocks) ** 0.5
        init_linear(self.w_out, std=std, generator=generator)

    def num_flops_per_token(self, seq_len: int) -> int:
        del seq_len
        training_factor = 3
        linear_flops = (
            2
            * training_factor
            * sum(
                module.weight.numel()
                for module in (
                    self.w_q,
                    self.w_k,
                    self.w_v,
                    self.f_proj_1,
                    self.f_proj_2,
                    self.w_b,
                    self.g_proj_1,
                    self.g_proj_2,
                    self.w_out,
                )
            )
        )
        conv_flops = 2 * training_factor * self.conv_size * (self.key_dim * 2 + self.value_dim)
        state_size = self.n_v_heads * self.head_k_dim * self.head_v_dim
        # Decay, delta read/erase/write, and recurrent read all touch the KxV state.
        recurrent_flops = 2 * training_factor * 5 * state_size
        return int(linear_flops + conv_flops + recurrent_flops)


@SequenceMixerConfig.register("kimi_delta_attention")
@dataclass
class KimiDeltaAttentionConfig(SequenceMixerConfig[KimiDeltaAttention]):
    """Configuration for :class:`KimiDeltaAttention`.

    :param n_heads: The number of key/query heads.
    :param n_v_heads: The number of value heads. The pinned KDA kernel currently requires this
        to equal ``n_heads``. Defaults to ``n_heads`` when unset.
    :param head_dim: The key/query head dimension. Defaults to ``d_model // n_heads``.
    :param expand_v: Multiplier applied to ``head_dim`` to determine the value head dimension.
    :param allow_neg_eigval: Whether to scale the delta gate to allow negative eigenvalues.
    :param conv_size: The kernel size of the causal convolutions applied to Q, K, and V.
    :param conv_bias: Whether the causal convolutions include bias parameters.
    :param norm_eps: Epsilon used by the gated RMS normalization on the output.
    :param use_experimental_kernels: **Experimental.** Whether to use the kernels from the
        ``kernel-fun`` package instead of FLA's: the CuTe/Triton KDA kernels
        (:func:`kernel_fun.kda.chunk_kda`) for the fixed-length chunk path, and the fused
        short-conv kernels (:func:`kernel_fun.cconv.causal_conv1d`) for the layer's three
        causal convolutions. This single flag controls both. Requires the package,
        installed with the ``kernel-fun`` extra (``pip install
        'ai2-olmo-core[kernel-fun]'``). The training image must also provide the CUDA 13
        CuTe DSL build (tested with ``nvidia-cutlass-dsl[cu13]==4.5.3``), which needs a
        CUDA 13-compatible NVIDIA driver. Install it explicitly with ``pip install
        'nvidia-cutlass-dsl[cu13]==4.5.3'`` when preparing the image; the extra leaves
        the image's CUDA stack unchanged. The CUDA 12 compiler cannot lower the KDA
        MMA backward; building the layer with it raises before training starts. Each kernel
        only takes effect on the hardware/shapes it supports (Blackwell, chunk-size-64, no
        packed-document ``cu_seqlens`` for KDA; Hopper and up, no bias, no packed-document
        ``cu_seqlens`` for the conv); otherwise the layer silently falls back to FLA.

        These kernels are faster but newer and far less exercised than FLA's: they are
        not bit-identical to FLA's monolith, so loss curves will not match a run with this
        turned off. Swapped in are the forward scan+readout, the gate activation (fused
        into the cumsum rather than run as eager fp32 ops), and four of the backward's
        seven stages; the rest are FLA's own kernels at FLA's own stage boundaries. At the
        production shape this measured 1.54x on the op. Leave it off unless you are
        deliberately testing the kernels, and check the ``kernel-fun`` lines in the
        training log to confirm they engaged — the fallback is silent by design and reads
        as a correct 1.00x.
    :param dtype: The parameter dtype.
    """

    n_heads: int = 16
    n_v_heads: int | None = None
    head_dim: int | None = None
    expand_v: float = 1.0
    allow_neg_eigval: bool = False
    conv_size: int = 4
    conv_bias: bool = False
    norm_eps: float = 1e-5
    use_experimental_kernels: bool = False
    dtype: DType = DType.float32

    def num_params(self, d_model: int) -> int:
        n_v_heads = self.n_v_heads or self.n_heads
        head_dim = self.head_dim or d_model // self.n_heads
        head_v_dim = int(head_dim * self.expand_v)
        key_dim = self.n_heads * head_dim
        value_dim = n_v_heads * head_v_dim
        gate_dim = self.n_heads * head_dim

        params = 0
        params += d_model * key_dim * 2  # q, k
        params += d_model * value_dim  # v
        params += d_model * head_v_dim + head_v_dim * gate_dim  # decay bottleneck
        params += d_model * self.n_heads  # scalar delta gate
        params += self.n_heads + gate_dim  # A_log, dt_bias
        params += self.conv_size * (key_dim * 2 + value_dim)
        if self.conv_bias:
            params += key_dim * 2 + value_dim
        params += d_model * head_v_dim + head_v_dim * value_dim + value_dim  # output gate
        params += head_v_dim  # RMSNorm weight
        params += value_dim * d_model  # output projection
        return params

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: BufferCache | None = None,
    ) -> KimiDeltaAttention:
        del layer_idx, n_layers, cache
        return KimiDeltaAttention(
            d_model=d_model,
            n_heads=self.n_heads,
            n_v_heads=self.n_v_heads,
            head_dim=self.head_dim,
            expand_v=self.expand_v,
            allow_neg_eigval=self.allow_neg_eigval,
            conv_size=self.conv_size,
            conv_bias=self.conv_bias,
            norm_eps=self.norm_eps,
            use_experimental_kernels=self.use_experimental_kernels,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )
