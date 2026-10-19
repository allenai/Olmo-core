import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Placement, distribute_tensor
from torch.nn import functional as F

from olmo_core.config import DType
from olmo_core.distributed.parallel.context_parallel import (
    all_to_all_cp2hp,
    all_to_all_single_cp2hp,
    all_to_all_single_hp2cp,
)
from olmo_core.nn.attention.base import SequenceMixer, SequenceMixerConfig
from olmo_core.nn.attention.flash_linear_attn_api import (
    dispatch_chunk_gated_delta_rule,
    has_fla,
)
from olmo_core.nn.attention.ring import (
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.nn.buffer_cache import BufferCache
from olmo_core.nn.convolution import CausalConv1d
from olmo_core.nn.feed_forward import ActivationFunction

if TYPE_CHECKING:
    from olmo_core.nn.transformer.init import InitMethod


class GatedDeltaNet(SequenceMixer):
    """
    The layer implementation for `Gated Delta Networks <https://arxiv.org/abs/2412.06464>`_.

    Modified from: https://github.com/fla-org/flash-linear-attention/blob/3cf180339b8a1cbad823f553541cd531d18670ea/fla/layers/gated_deltanet.py#L34

    This is a linear attention variant that uses a gated delta rule for recurrent
    state updates, providing efficient O(n) sequence modeling.

    :param d_model: The model hidden size.
    :param n_heads: The number of attention heads.
    :param n_v_heads: The number of value heads. If ``None``, defaults to ``n_heads``.
        GVA is applied if ``n_v_heads`` > ``n_heads``.
    :param head_dim: The dimension of each head. If ``None``, defaults to ``d_model // n_heads``.
    :param expand_v: The expansion ratio for the value dim. Default: 2.0.
    :param allow_neg_eigval: Allow negative eigenvalues. Default: ``True``. If set to ``True``, the beta
        will be multiplied by 2. See reference: `Unlocking State-Tracking in Linear RNNs Through Negative
        Eigenvalues <https://arxiv.org/abs/2411.12537>`_.
    :param conv_size: The kernel size of the short convolution. Default: 4.
    :param conv_bias: Whether to use bias in the short convolution. Default: ``False``.
    :param norm_eps: The epsilon value for the normalization layer. Default: 1e-5.
    :param dtype: The default data type to use for parameters.
    :param init_device: The device to initialize weights on.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_v_heads: int | None = None,
        head_dim: int | None = None,
        expand_v: float = 2.0,
        allow_neg_eigval: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        assert has_fla()
        from fla.modules import FusedRMSNormGated

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_v_heads = n_v_heads if n_v_heads is not None else n_heads
        self.head_dim = head_dim if head_dim is not None else d_model // n_heads
        self.expand_v = expand_v
        self.allow_neg_eigval = allow_neg_eigval
        self.conv_size = conv_size

        self.head_k_dim = self.head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.n_heads * self.head_k_dim)
        self.value_dim = int(self.n_v_heads * self.head_v_dim)

        # Consistency checks: ensure expand_v produces integer dimensions
        assert math.isclose(self.n_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5)
        assert math.isclose(self.head_dim * expand_v, self.head_v_dim, rel_tol=1e-5)
        assert self.n_v_heads >= self.n_heads and self.n_v_heads % self.n_heads == 0

        self.w_q = nn.Linear(d_model, self.key_dim, bias=False, dtype=dtype, device=init_device)
        self.w_k = nn.Linear(d_model, self.key_dim, bias=False, dtype=dtype, device=init_device)
        self.w_v = nn.Linear(d_model, self.value_dim, bias=False, dtype=dtype, device=init_device)
        self.w_a = nn.Linear(d_model, self.n_v_heads, bias=False, dtype=dtype, device=init_device)
        self.w_b = nn.Linear(d_model, self.n_v_heads, bias=False, dtype=dtype, device=init_device)

        self.A_log = nn.Parameter(torch.empty(self.n_v_heads, dtype=dtype, device=init_device))
        self.dt_bias = nn.Parameter(torch.empty(self.n_v_heads, dtype=dtype, device=init_device))

        self.q_conv1d = CausalConv1d(
            hidden_size=self.key_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            activation=ActivationFunction.silu.value,
            dtype=dtype,
            init_device=init_device,
        )
        self.k_conv1d = CausalConv1d(
            hidden_size=self.key_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            activation=ActivationFunction.silu.value,
            dtype=dtype,
            init_device=init_device,
        )
        self.v_conv1d = CausalConv1d(
            hidden_size=self.value_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            activation=ActivationFunction.silu.value,
            dtype=dtype,
            init_device=init_device,
        )
        self.w_g = nn.Linear(d_model, self.value_dim, bias=False, dtype=dtype, device=init_device)
        self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps, device=init_device)  # type: ignore
        self.w_out = nn.Linear(self.value_dim, d_model, bias=False, dtype=dtype, device=init_device)

        self.cp_enabled = False

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply gated delta network sequence mixing to the input.

        :param x: The input of shape ``(batch_size, seq_len, d_model)``.
        :param cu_doc_lens: Cumulative document lengths in the input ``x``, a 1D
            :class:`torch.int32` tensor that should always have one more element than there
            are documents (the first element in the tensor should always be ``0``).

        :returns: The output with shape ``(batch_size, seq_len, d_model)``.
        """
        del kwargs  # Ignore any extra kwargs passed from attention interface
        B, T_og, _ = x.shape

        # shape: (batch_size, seq_len, n_heads * head_k_dim),
        #        (batch_size, seq_len, n_heads * head_k_dim),
        #        (batch_size, seq_len, n_v_heads * head_v_dim)
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        beta = self.w_b(x).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0
        g = -self.A_log.float().exp() * F.softplus(self.w_a(x).float() + self.dt_bias)

        if self.cp_enabled and self.uly is not None:
            assert self._cp_group is not None
            # [B, T_local, C] -> [B, T_total, C/CP]
            q, k = all_to_all_cp2hp([q, k], self._cp_group)
            v = all_to_all_single_cp2hp(v, self._cp_group)
            g, beta = all_to_all_cp2hp([g, beta], self._cp_group)

        q = self.q_conv1d(x=q, cu_seqlens=cu_doc_lens)
        k = self.k_conv1d(x=k, cu_seqlens=cu_doc_lens)
        v = self.v_conv1d(x=v, cu_seqlens=cu_doc_lens)

        T = q.size(1)
        q = q.view(B, T, -1, self.head_k_dim)
        k = k.view(B, T, -1, self.head_k_dim)
        v = v.view(B, T, -1, self.head_v_dim)

        if self.n_v_heads > self.n_heads:
            repeat_factor = self.n_v_heads // self.n_heads
            q = q.repeat_interleave(repeat_factor, dim=-2)
            k = k.repeat_interleave(repeat_factor, dim=-2)

        o, _ = dispatch_chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=cu_doc_lens, use_qk_l2norm_in_kernel=True
        )

        if self.cp_enabled and self.uly is not None:
            assert self._cp_group is not None
            # [B, T, H/CP, D] -> [B, T/CP, H, D]
            o = all_to_all_single_hp2cp(o, self._cp_group)

        g = self.w_g(x).view(B, T, -1, self.head_v_dim)

        # shape: (batch_size, seq_len, d_model)
        return self.w_out(self.o_norm(o, g).view(B, T_og, -1))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("Tensor parallelism is not yet implemented for GatedDeltaNet")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        if ring is not None:
            raise NotImplementedError("Ring context parallelism is not supported for GatedDeltaNet")
        assert uly is not None

        cp_world_size = cp_mesh.size()
        if cp_world_size == 1:
            return

        # Ulysses CP requires divisibility by CP world size for:
        # 1. n_v_heads - for head partitioning in the recurrent kernel
        # 2. key_dim and value_dim - for channel partitioning in the conv layers
        assert self.n_v_heads % cp_world_size == 0
        assert self.key_dim % cp_world_size == 0
        assert self.value_dim % cp_world_size == 0

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
        init_method: "InitMethod",
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        from olmo_core.nn.transformer.init import (
            InitMethod,
            init_linear,
            init_linear_nanochat,
        )

        if init_method == InitMethod.fan_in:
            raise NotImplementedError(
                f"init method '{init_method}' is not supported for GatedDeltaNet"
            )

        if init_method == InitMethod.normalized:
            std = d_model**-0.5

        for w in (
            self.w_q,
            self.w_k,
            self.w_v,
            self.w_a,
            self.w_b,
            self.w_g,
            self.q_conv1d,
            self.k_conv1d,
            self.v_conv1d,
        ):
            if init_method == InitMethod.nanochat:
                init_linear_nanochat(w, generator=generator)
            else:
                init_linear(w, std=std, generator=generator)

        self.A_log.copy_(nn.init.uniform_(self.A_log, a=0, b=16, generator=generator).log())
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            nn.init.uniform_(self.dt_bias, generator=generator)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias.copy_(inv_dt)

        if init_method == InitMethod.nanochat:
            init_linear_nanochat(self.w_out, generator=generator)
            return
        if init_method == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5
        elif init_method == InitMethod.normalized:
            std = std / (2 * num_blocks) ** 0.5

        init_linear(self.w_out, std=std, generator=generator)

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Compute FLOPs per token for Gated Delta Net.

        This accounts for:
        - Linear projections (w_q, w_k, w_v, w_a, w_b, w_g, w_out)
        - Short convolutions (q, k, v)
        - Gated delta rule recurrent computation
        - Gated RMS normalization
        """
        del seq_len
        # Training FLOPs include the forward pass plus backward dgrad and wgrad
        # work, matching the convention used by the other sequence mixers.
        training_factor = 3

        # Linear projection FLOPs (2 ops per multiply-add)
        linear_flops = (
            2
            * training_factor
            * sum(
                m.weight.numel()
                for m in (self.w_q, self.w_k, self.w_v, self.w_a, self.w_b, self.w_g, self.w_out)
            )
        )

        # Short convolution FLOPs (2 ops per multiply-add, kernel_size taps per output)
        conv_flops = (
            2
            * training_factor
            * self.conv_size
            * (self.key_dim + self.key_dim + self.value_dim)  # q_conv1d  # k_conv1d  # v_conv1d
        )

        # Gated delta rule recurrent computation per token:
        # - Outer product k ⊗ v: n_v_heads * head_k_dim * head_v_dim
        # - State decay: n_v_heads * head_k_dim * head_v_dim
        # - Beta scaling: n_v_heads * head_k_dim * head_v_dim
        # - Query-state matmul: n_v_heads * head_k_dim * head_v_dim
        # Each is 2 FLOPs per element (multiply-add or similar)
        state_size = self.n_v_heads * self.head_k_dim * self.head_v_dim
        recurrent_flops = 2 * training_factor * 4 * state_size

        return int(linear_flops + conv_flops + recurrent_flops)


@SequenceMixerConfig.register("gated_delta_net")
@dataclass
class GatedDeltaNetConfig(SequenceMixerConfig[GatedDeltaNet]):
    """
    Configuration for :class:`GatedDeltaNet`.

    See :class:`GatedDeltaNet` for a description of the configuration options.
    """

    n_heads: int = 16
    """
    The number of attention heads.
    """
    n_v_heads: Optional[int] = None
    """
    The number of value heads. If ``None``, defaults to ``n_heads``.
    If ``n_v_heads`` > ``n_heads``, GVA (Grouped Value Attention) is applied.

    GVA is preferred over GQA for linear RNNs like GDN because the recurrent state
    has shape ``(n_v_heads, head_k_dim, head_v_dim)``. Unlike softmax attention where
    the KV cache grows with sequence length (motivating GQA to reduce it), the linear
    RNN state is constant size regardless of sequence length. Since there's no memory
    scaling issue to solve, we instead can opt to increase the state size to improve the model's
    capacity to compress long-range context. Increasing ``n_v_heads`` directly
    increases this fixed state size.
    """
    head_dim: Optional[int] = None
    """
    The dimension of each head. If ``None``, defaults to ``d_model // n_heads``.
    """
    expand_v: float = 2.0
    """
    The expansion ratio for the value dimension (``head_v_dim = head_dim * expand_v``).
    Like ``n_v_heads``, this increases the constant-size recurrent state, improving
    capacity without memory scaling concerns.
    """
    allow_neg_eigval: bool = True
    """
    Allow negative eigenvalues in the recurrent dynamics.
    """
    conv_size: int = 4
    """
    The kernel size of the short convolution.
    """
    conv_bias: bool = False
    """
    Whether to use bias in the short convolution.
    """
    norm_eps: float = 1e-5
    """
    The epsilon value for the normalization layer.
    """
    dtype: DType = DType.float32
    """
    The default data type to use for parameters.
    """

    def num_params(self, d_model: int) -> int:
        """
        The number of params that the GatedDeltaNet will have once built.

        :param d_model: The model dimensionality.
        """
        n_heads = self.n_heads
        n_v_heads = self.n_v_heads or n_heads
        head_dim = self.head_dim or d_model // n_heads
        head_v_dim = int(head_dim * self.expand_v)
        key_dim = n_heads * head_dim
        value_dim = n_v_heads * head_v_dim

        params = 0

        # Linear projections: w_q, w_k, w_v, w_a, w_b, w_g, w_out
        params += d_model * key_dim  # w_q
        params += d_model * key_dim  # w_k
        params += d_model * value_dim  # w_v
        params += d_model * n_v_heads  # w_a
        params += d_model * n_v_heads  # w_b
        params += d_model * value_dim  # w_g
        params += value_dim * d_model  # w_out

        # A_log and dt_bias parameters
        params += n_v_heads  # A_log
        params += n_v_heads  # dt_bias

        # Short convolutions (kernel_size * hidden_size for each)
        params += self.conv_size * key_dim  # q_conv1d
        params += self.conv_size * key_dim  # k_conv1d
        params += self.conv_size * value_dim  # v_conv1d
        if self.conv_bias:
            params += key_dim  # q_conv1d bias
            params += key_dim  # k_conv1d bias
            params += value_dim  # v_conv1d bias

        # FusedRMSNormGated (weight only, no bias)
        params += head_v_dim  # o_norm

        return params

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> GatedDeltaNet:
        """
        Build the GatedDeltaNet module.

        :param d_model: The model dimensionality.
        :param layer_idx: The layer index (unused).
        :param n_layers: The total number of layers (unused).
        :param init_device: The device to initialize the parameters on, e.g. "cpu", "meta".
        :param cache: Optional buffer cache (unused).
        """
        del layer_idx, n_layers, cache  # Unused

        return GatedDeltaNet(
            d_model=d_model,
            n_heads=self.n_heads,
            n_v_heads=self.n_v_heads,
            head_dim=self.head_dim,
            expand_v=self.expand_v,
            allow_neg_eigval=self.allow_neg_eigval,
            conv_size=self.conv_size,
            conv_bias=self.conv_bias,
            norm_eps=self.norm_eps,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )


def _copy_into_tensor_or_dtensor(dst: torch.Tensor, src: torch.Tensor) -> None:
    if isinstance(dst, DTensor):
        dst.copy_(distribute_tensor(src, dst.device_mesh, placements=dst.placements))
    else:
        dst.copy_(src)


def _pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
    pad_shape = (
        (0, 0, 0, 0, 0, pad_size, 0, 0)
        if len(input_tensor.shape) == 4
        else (0, 0, 0, pad_size, 0, 0)
    )
    return F.pad(input_tensor, pad_shape, mode="constant", value=0)


def _reshape_into_chunks(
    input_tensor: torch.Tensor, pad_size: int, chunk_size: int
) -> torch.Tensor:
    input_tensor = _pad_tensor_by_size(input_tensor, pad_size)
    if len(input_tensor.shape) == 3:
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    return input_tensor.reshape(
        input_tensor.shape[0],
        -1,
        chunk_size,
        input_tensor.shape[2],
        input_tensor.shape[3],
    )


def _segment_sum(input_tensor: torch.Tensor) -> torch.Tensor:
    chunk_size = input_tensor.size(-1)
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool),
        diagonal=-1,
    )
    input_tensor = input_tensor.masked_fill(~mask, 0)
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)

    mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool),
        diagonal=0,
    )
    return tensor_segsum.masked_fill(~mask, -torch.inf)


class NemotronRMSNormGated(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        group_size: int,
        eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=init_device))
        self.variance_epsilon = eps
        self.group_size = group_size

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(
        self, hidden_states: torch.Tensor, gate: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        *prefix_dims, last_dim = hidden_states.shape
        group_count = last_dim // self.group_size
        hidden_states_group = hidden_states.view(*prefix_dims, group_count, self.group_size)
        variance = hidden_states_group.pow(2).mean(-1, keepdim=True)
        hidden_states_group = hidden_states_group * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states_group.view(*prefix_dims, group_count * self.group_size)
        return self.weight * hidden_states.to(input_dtype)


@SequenceMixerConfig.register("nemotron_mamba2")
@dataclass
class NemotronMamba2Config(SequenceMixerConfig["NemotronMamba2Mixer"]):
    ssm_state_size: int = 128
    conv_kernel: int = 4
    mamba_num_heads: int = 64
    mamba_head_dim: int = 64
    n_groups: int = 8
    chunk_size: int = 128
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 0.0001
    use_conv_bias: bool = True
    use_bias: bool = False
    layer_norm_epsilon: float = 1e-5
    dtype: DType = DType.bfloat16

    def num_params(self, d_model: int) -> int:
        intermediate = self.mamba_num_heads * self.mamba_head_dim
        conv_dim = intermediate + 2 * self.n_groups * self.ssm_state_size
        projection_size = intermediate + conv_dim + self.mamba_num_heads
        params = d_model * projection_size
        if self.use_bias:
            params += projection_size
        params += conv_dim * self.conv_kernel
        if self.use_conv_bias:
            params += conv_dim
        params += 3 * self.mamba_num_heads
        params += intermediate
        params += intermediate * d_model
        if self.use_bias:
            params += d_model
        return params

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "NemotronMamba2Mixer":
        del n_layers, cache
        return NemotronMamba2Mixer(
            d_model=d_model,
            layer_idx=layer_idx,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
            **self.as_dict(exclude={"dtype"}),
        )


class NemotronMamba2Mixer(SequenceMixer):
    def __init__(
        self,
        *,
        d_model: int,
        layer_idx: int,
        ssm_state_size: int,
        conv_kernel: int,
        mamba_num_heads: int,
        mamba_head_dim: int,
        n_groups: int,
        chunk_size: int,
        time_step_min: float,
        time_step_max: float,
        time_step_floor: float,
        use_conv_bias: bool,
        use_bias: bool,
        layer_norm_epsilon: float,
        dtype: torch.dtype,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.ssm_state_size = ssm_state_size
        self.conv_kernel_size = conv_kernel
        self.num_heads = mamba_num_heads
        self.head_dim = mamba_head_dim
        self.intermediate_size = mamba_num_heads * mamba_head_dim
        self.n_groups = n_groups
        self.chunk_size = chunk_size
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_floor = time_step_floor
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            d_model, projection_size, bias=use_bias, dtype=dtype, device=init_device
        )
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            dtype=dtype,
            device=init_device,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads, dtype=dtype, device=init_device))
        self.A_log = nn.Parameter(torch.empty(self.num_heads, dtype=dtype, device=init_device))
        self.D = nn.Parameter(torch.ones(self.num_heads, dtype=dtype, device=init_device))
        self.norm = NemotronRMSNormGated(
            self.intermediate_size,
            group_size=self.intermediate_size // self.n_groups,
            eps=layer_norm_epsilon,
            dtype=dtype,
            init_device=init_device,
        )
        self.out_proj = nn.Linear(
            self.intermediate_size, d_model, bias=use_bias, dtype=dtype, device=init_device
        )
        self.reset_parameters()

    def reset_parameters(self, generator: Optional[torch.Generator] = None) -> None:
        with torch.no_grad():
            if self.A_log.device.type != "meta":
                values = (
                    torch.arange(
                        1, self.num_heads + 1, device=self.A_log.device, dtype=torch.float32
                    )
                    .log()
                    .to(dtype=self.A_log.dtype)
                )
                _copy_into_tensor_or_dtensor(self.A_log, values)
            if self.dt_bias.device.type != "meta":
                # Match the Nemotron-H / Mamba2 timestep init: sample dt log-uniformly in
                # [time_step_min, time_step_max], clamp by time_step_floor, and store the
                # inverse-softplus so softplus(dt_bias) recovers dt in the configured range. The
                # generator (when supplied by init_weights) keeps this tied to the model init seed
                # rather than the ambient global RNG.
                dt = torch.exp(
                    torch.rand(
                        self.num_heads,
                        device=self.dt_bias.device,
                        dtype=torch.float32,
                        generator=generator,
                    )
                    * (math.log(self.time_step_max) - math.log(self.time_step_min))
                    + math.log(self.time_step_min)
                ).clamp(min=self.time_step_floor)
                inv_dt = dt + torch.log(-torch.expm1(-dt))
                _copy_into_tensor_or_dtensor(self.dt_bias, inv_dt.to(dtype=self.dt_bias.dtype))
            nn.init.ones_(self.D)
            self.norm.reset_parameters()

    def forward(
        self,
        input_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        if attention_mask is not None:
            input_states = (input_states * attention_mask[:, :, None]).to(dtype)
        projected_states = self.in_proj(input_states)
        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.n_groups * self.ssm_state_size
            - self.num_heads
        ) // 2
        _, _, gate, hidden_states, dt = projected_states.split(
            [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )

        hidden_states = F.silu(
            self.conv1d(hidden_states.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        )
        if attention_mask is not None:
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)

        hidden_states, B, C = torch.split(
            hidden_states,
            [
                self.intermediate_size,
                self.n_groups * self.ssm_state_size,
                self.n_groups * self.ssm_state_size,
            ],
            dim=-1,
        )
        A = -torch.exp(self.A_log.float())
        dt = F.softplus(dt + self.dt_bias)
        dt = torch.clamp(dt, self.time_step_min)
        hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
        B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        B = B.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
        C = C.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
        pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

        D_residual = self.D[..., None] * _pad_tensor_by_size(hidden_states, pad_size)

        hidden_states = hidden_states * dt[..., None]
        A = A.to(hidden_states.dtype) * dt

        hidden_states, A, B, C = [
            _reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, A, B, C)
        ]

        A = A.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A, dim=-1)
        L = torch.exp(_segment_sum(A))

        G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]
        G = G_intermediate.sum(dim=-1)
        M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
        M = M_intermediate.sum(dim=-1)
        Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(3)

        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        B_decay_contraction = B * decay_states.permute(0, 2, 3, 1)[..., None]
        states = (
            (
                B_decay_contraction.permute(0, 1, 3, 2, 4)[..., None]
                * hidden_states.permute(0, 1, 3, 2, 4)[..., None, :]
            )
            .sum(dim=3)
            .permute(0, 1, 2, 4, 3)
        )
        previous_states = torch.zeros_like(states[:, :1])
        states = torch.cat([previous_states, states], dim=1)
        decay_chunk = torch.exp(_segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))

        states_permuted = states.permute(0, 2, 1, 3, 4)
        result = (decay_chunk[..., None, None] * states_permuted[:, :, None, ...]).sum(dim=2)
        new_states = result.permute(0, 2, 1, 3, 4)
        states = new_states[:, :-1]

        state_decay_out = torch.exp(A_cumsum)
        C_times_states = C[..., None, :] * states[:, :, None, ...]
        state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
        Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]

        y = Y_diag + Y_off
        y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)
        y = y + D_residual
        if pad_size > 0:
            y = y[:, :seq_len, :, :]
        y = y.reshape(batch_size, seq_len, -1)

        scan_output = self.norm(y, gate)
        return self.out_proj(scan_output.to(dtype))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("Tensor parallelism is not implemented for NemotronMamba2Mixer")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        del cp_mesh, ring, uly
        raise NotImplementedError("Context parallelism is not implemented for NemotronMamba2Mixer")

    def init_weights(
        self,
        *,
        init_method: "InitMethod",
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        from olmo_core.nn.transformer.init import InitMethod, init_linear

        if init_method == InitMethod.nanochat:
            raise NotImplementedError(
                f"init method '{init_method}' is not supported for NemotronMamba2Mixer"
            )

        del d_model
        init_linear(self.in_proj, std=std, generator=generator)
        init_linear(self.conv1d, std=std, generator=generator)
        out_std = std
        if init_method in (InitMethod.llama, InitMethod.normalized):
            out_std = std / math.sqrt(2 * num_blocks)
        elif init_method == InitMethod.llama_depth:
            out_std = std / math.sqrt(2 * (block_idx + 1))
        init_linear(self.out_proj, std=out_std, generator=generator)
        self.reset_parameters(generator=generator)

    def num_flops_per_token(self, seq_len: int) -> int:
        del seq_len
        return 0
