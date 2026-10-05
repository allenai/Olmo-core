import logging
import math
import os
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import torch
import torch.nn as nn
from torch.autograd.graph import saved_tensors_hooks
from torch.distributed import DeviceMesh
from torch.distributed.tensor import (
    DTensor,
    Placement,
    Replicate,
    Shard,
    distribute_tensor,
)
from torch.distributed.tensor.parallel import parallelize_module

from olmo_core.config import Config, DType, StrEnum
from olmo_core.distributed.parallel.tensor_parallel import SequenceParallel
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention.base import SequenceMixer, SequenceMixerConfig
from olmo_core.nn.attention.kda import KimiDeltaAttention, KimiDeltaAttentionConfig
from olmo_core.nn.attention.kv_cache import KVCacheManager
from olmo_core.nn.attention.recurrent import (
    GatedDeltaNet,
    GatedDeltaNetConfig,
    NemotronMamba2Config,
    NemotronMamba2Mixer,
)

from ..buffer_cache import BufferCache
from ..config import ModuleConfig
from ..functional import l2_normalize
from ..layer_norm import LayerNorm, LayerNormConfig
from ..mxfp8_linear import MXFP8Linear
from ..output_discard_checkpoint import OutputDiscardCheckpoint
from ..rope import ComplexRotaryEmbedding, RoPEConfig, RoPEType, RotaryEmbedding
from ..utils import get_tp_wrappers
from . import flash_attn_api
from .backend import (
    AttentionBackend,
    AttentionBackendName,
    FlashAttention2Backend,
    FlashAttention3Backend,
    FlashAttention4Backend,
    TEAttentionBackend,
    TorchAttentionBackend,
)
from .ring import (
    RingAttentionLlama3LoadBalancer,
    RingAttentionLoadBalancer,
    RingAttentionLoadBalancerType,
    RingAttentionZigZagLoadBalancer,
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
    UlyssesLoadBalancer,
)

if TYPE_CHECKING:
    from olmo_core.nn.transformer.init import InitMethod

__all__ = [
    "SlidingWindowAttentionConfig",
    "GateGranularity",
    "GateConfig",
    "AttentionType",
    "AttentionBackendName",
    "AttentionBackend",
    "TorchAttentionBackend",
    "FlashAttention2Backend",
    "FlashAttention3Backend",
    "FlashAttention4Backend",
    "TEAttentionBackend",
    "AttentionConfig",
    "Attention",
    "FusedAttentionV2",
    "NormalizedAttention",
    "RingAttentionLoadBalancerType",
    "RingAttentionLoadBalancer",
    "RingAttentionZigZagLoadBalancer",
    "RingAttentionLlama3LoadBalancer",
    "UlyssesLoadBalancer",
    "RingContextParallelStyle",
    "UlyssesContextParallelStyle",
    "GatedDeltaNetConfig",
    "GatedDeltaNet",
    "KimiDeltaAttentionConfig",
    "KimiDeltaAttention",
    "NemotronMamba2Config",
    "NemotronMamba2Mixer",
]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _MXFP8SavedTensor:
    qdata: torch.Tensor
    scales: torch.Tensor
    shape: torch.Size
    dtype: torch.dtype
    name: str


def _can_save_tensor_as_mxfp8(t: torch.Tensor) -> bool:
    return (
        t.is_cuda
        and t.dtype == torch.bfloat16
        and t.ndim >= 2
        and t.shape[-1] % 32 == 0
        and t.requires_grad
    )


def _set_saved_activation_name(t: torch.Tensor, name: str) -> torch.Tensor:
    t._olmo_saved_activation_name = name  # type: ignore[attr-defined]
    if os.getenv("OLMO_EP_NO_SYNC_SAVED_ACTIVATIONS_DEBUG"):
        try:
            from olmo_core.nn.moe.v2.activation_debug import (
                record_named_saved_activation,
            )

            record_named_saved_activation(t, name)
        except Exception:
            pass
    return t


def _record_saved_activation_debug(t: torch.Tensor, name: str) -> None:
    if not os.getenv("OLMO_EP_NO_SYNC_SAVED_ACTIVATIONS_DEBUG"):
        return

    try:
        from olmo_core.nn.moe.v2.activation_debug import record_named_saved_activation

        record_named_saved_activation(t, name)
    except Exception:
        pass


def _pack_mxfp8_saved_tensor(t: torch.Tensor, *, name: str) -> _MXFP8SavedTensor:
    from olmo_core.kernels.mxfp8_utils import quantize_rows_to_mxfp8

    t_2d = t.reshape(-1, t.shape[-1])
    qdata, scales = quantize_rows_to_mxfp8(t_2d, block_size=32)
    _set_saved_activation_name(qdata, f"{name}.mxfp8_qdata")
    _set_saved_activation_name(scales, f"{name}.mxfp8_scales")
    return _MXFP8SavedTensor(
        qdata=qdata,
        scales=scales,
        shape=t.shape,
        dtype=t.dtype,
        name=name,
    )


def _unpack_mxfp8_saved_tensor(x: _MXFP8SavedTensor) -> torch.Tensor:
    from olmo_core.kernels.mxfp8_utils import dequantize_rows_from_mxfp8

    t_2d = dequantize_rows_from_mxfp8(
        x.qdata,
        x.scales,
        block_size=32,
        out_dtype=x.dtype,
    )
    return t_2d.view(x.shape)


class _MXFP8SavedQKVHooks:
    def __init__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        pack_counter: list[int],
    ) -> None:
        # Match by storage pointer, not tensor identity. The attention backend transposes (and, for
        # GQA, may repeat) q/k/v into new tensor objects before SDPA, and autograd saves *those*
        # derived tensors. View transforms such as transpose share storage with the originals, so a
        # storage-pointer key still recognizes them; identity matching would miss all of them and
        # the pack would silently no-op. (A GQA repeat that copies k/v gets fresh storage and is
        # not matched -- acceptable, and it never mis-packs an unrelated tensor.)
        self.target_names = {
            q.untyped_storage().data_ptr(): "attention.q",
            k.untyped_storage().data_ptr(): "attention.k",
            v.untyped_storage().data_ptr(): "attention.v",
        }
        self.pack_counter = pack_counter

    def pack(self, t: torch.Tensor) -> Any:
        name = self.target_names.get(t.untyped_storage().data_ptr())
        if name is None:
            _record_saved_activation_debug(t, "attention.sdpa.saved_passthrough")
            return t
        if not _can_save_tensor_as_mxfp8(t):
            return t

        self.pack_counter[0] += 1
        return _pack_mxfp8_saved_tensor(t, name=name)

    def unpack(self, x: Any) -> torch.Tensor:
        if not isinstance(x, _MXFP8SavedTensor):
            return x

        return _unpack_mxfp8_saved_tensor(x)


@torch.compiler.disable(reason="MXFP8 saved-QKV hooks close over per-forward tensors")
def _mxfp8_saved_qkv_hooks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    pack_counter: list[int],
):
    hooks = _MXFP8SavedQKVHooks(q, k, v, pack_counter=pack_counter)
    return saved_tensors_hooks(hooks.pack, hooks.unpack)


class GateGranularity(StrEnum):
    headwise = "headwise"
    """Head-wise gating: one gate value per attention head, broadcast across head dimension."""
    elementwise = "elementwise"
    """Element-wise gating: one gate value per output element."""


@dataclass
class GateConfig(Config):
    granularity: GateGranularity = GateGranularity.headwise
    """The granularity of gating to use."""
    full_precision: bool = True
    """Whether to always apply gating in full precision regardless of the input data type."""


@dataclass
class SlidingWindowAttentionConfig(Config):
    pattern: List[int]
    """
    The pattern of window sizes to use for attention, repeated to cover all layers.
    A value of -1 indicates full attention. For example, a pattern of ``[4096, 4096, 4096, -1]``
    means that for each set of 4 layers, the first 3 will use a window size of 4096,
    and the last layer will use full attention.
    """

    force_full_attention_on_first_layer: bool = True
    """
    If `True`, the first transformer layer will always use full attention, regardless of the pattern.
    """

    force_full_attention_on_last_layer: bool = True
    """
    If `True`, the last transformer layer will always use full attention, regardless of the pattern.
    """

    def _get_window_size(self, layer_idx: int, n_layers: int) -> int:
        """
        Get the window size for a given layer, returning -1 for full attention.
        """
        if self.force_full_attention_on_first_layer and layer_idx == 0:
            return -1
        if self.force_full_attention_on_last_layer and layer_idx == (n_layers - 1):
            return -1

        # Adjust the layer index if the first layer is special-cased to full attention
        # (in which case the pattern is applied starting from the second layer)
        effective_layer_idx = layer_idx
        if self.force_full_attention_on_first_layer:
            effective_layer_idx -= 1

        window_size = self.pattern[effective_layer_idx % len(self.pattern)]
        if window_size <= 0 and window_size != -1:
            raise OLMoConfigurationError(
                f"Sliding window size must be positive or -1 (got {window_size})"
            )
        return window_size

    def should_use_swa(self, layer_idx: int, n_layers: int) -> bool:
        """
        Returns `True` if the given layer uses sliding window attention.
        """
        return self._get_window_size(layer_idx, n_layers) != -1

    def get_window_size(self, layer_idx: int, n_layers: int) -> int:
        """
        Get the sliding window size for a given layer.
        """
        window_size = self._get_window_size(layer_idx, n_layers)
        if window_size == -1:
            raise ValueError(f"Layer {layer_idx} is not configured for sliding window attention.")
        return window_size


def _resolve_attention_backend(
    backend: Optional[AttentionBackendName], use_flash: Optional[bool] = None
) -> Optional[AttentionBackendName]:
    """Resolve the deprecated flag without overriding an explicit backend or SWA auto-selection."""
    if backend is not None:
        backend = AttentionBackendName(backend)
    if use_flash:
        if backend is not None and backend != AttentionBackendName.flash_2:
            raise OLMoConfigurationError(
                f"'use_flash' is only compatible with 'flash_2' backend (got '{backend}')"
            )
        elif backend is None:
            warnings.warn(
                "'use_flash' is deprecated, use 'backend=flash_2' instead",
                DeprecationWarning,
                stacklevel=3,
            )
            backend = AttentionBackendName.flash_2
    return backend


class AttentionType(StrEnum):
    """
    An enumeration of the different attention implementations.
    """

    default = "default"
    """
    ➡️ :class:`Attention`
    """
    fused = "fused"
    """
    Legacy spelling for :class:`FusedAttentionV2`, retained for loading old configs.
    Fused RoPE is replaced with regular RoPE, which is not numerically identical.
    """
    fused_v2 = "fused_v2"
    """
    ➡️ :class:`FusedAttentionV2`
    """
    normalized = "normalized"
    """
    ➡️ :class:`NormalizedAttention`
    """


@SequenceMixerConfig.register("attention")
@dataclass
class AttentionConfig(SequenceMixerConfig["SequenceMixer"]):
    """
    A configuration class for easily building any of the different attention modules.

    See the individual :class:`Attention` subclasses for a description of the configuration options.
    """

    name: AttentionType = AttentionType.default
    """
    The name of the implementation.
    """
    n_heads: int = 16
    n_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    bias: Optional[bool] = None
    gate: Optional[GateConfig] = None
    rope: Optional[RoPEConfig] = None
    clip_qkv: Optional[float] = None
    qk_norm: Optional[LayerNormConfig] = None
    dropout: Optional[float] = None
    use_flash: Optional[bool] = None
    """Deprecated input retained for old configs; use :data:`backend` in new configs."""
    backend: Optional[AttentionBackendName] = None
    dtype: DType = DType.float32
    sliding_window: Optional[SlidingWindowAttentionConfig] = None
    use_head_qk_norm: Optional[bool] = None
    scalable_softmax: bool = False
    """Scalable softmax; incompatible with CP, KV caching, and sliding window attention."""
    qk_norm_per_head_gains: Optional[bool] = None
    attention_sinks: bool = False
    """
    Add a per-head learnable "attention sink" logit (as in GPT-OSS). Only supported by the default
    attention with the torch backend.
    """
    mxfp8_projections: Optional[bool] = None
    """
    Shorthand for using :class:`MXFP8Linear` for both of :class:`FusedAttentionV2`'s packed QKV and
    output projections. Only supported by ``fused_v2`` attention.
    """
    mxfp8_qkv_projection: Optional[bool] = None
    """
    Use :class:`MXFP8Linear` for :class:`FusedAttentionV2`'s packed QKV projection.
    """
    mxfp8_out_projection: Optional[bool] = None
    """
    Use :class:`MXFP8Linear` for :class:`FusedAttentionV2`'s output projection.
    """
    use_recompute_qkv_prep: Optional[bool] = None
    """
    Recompute the Q/K/V preparation in the backward pass to save activation memory. Supported by the
    ``default`` and ``fused_v2`` attention implementations.
    """
    mxfp8_save_qkv_for_backward: Optional[bool] = None
    """
    Save Q/K/V for backward as MXFP8 to reduce the saved-activation footprint. Supported by the
    ``default`` and ``fused_v2`` attention implementations.

    .. note::
        The flash backends save Q/K/V unmodified, so all three are packed. The ``torch`` backend
        transposes Q/K/V (a view -> still packed) but for GQA (``n_kv_heads < n_heads``) it also
        repeats K/V into fresh storage before SDPA, so only Q is packed for GQA on that backend.
    """

    def num_params(self, d_model: int) -> int:
        """
        The number of params that the attention implementation will have once built.

        :param d_model: The model dimensionality.
        """
        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads or n_heads
        head_dim = self.head_dim or d_model // n_heads
        bias = self.bias if self.bias is not None else self.name != AttentionType.normalized

        params = 0

        # Block attention Q projection.
        params += d_model * n_heads * head_dim
        if bias:
            params += n_heads * head_dim

        # Block attention KV projections.
        params += 2 * d_model * n_kv_heads * head_dim
        if bias:
            params += 2 * n_kv_heads * head_dim

        # Block attention QK norm.
        if self.qk_norm is not None:
            if self.use_head_qk_norm and not self.qk_norm_per_head_gains:
                params += 2 * self.qk_norm.num_params(head_dim)
            else:
                params += self.qk_norm.num_params(n_heads * head_dim)  # q_norm
                params += self.qk_norm.num_params(n_kv_heads * head_dim)  # k_norm

        # Block attention out.
        params += n_heads * head_dim * d_model
        if bias:
            params += d_model

        # Block attention gate projection.
        if self.gate is not None:
            if self.gate.granularity == GateGranularity.headwise:
                params += d_model * n_heads
                if bias:
                    params += n_heads
            elif self.gate.granularity == GateGranularity.elementwise:
                params += d_model * (n_heads * head_dim)
                if bias:
                    params += n_heads * head_dim

        # Block QK scaling factors.
        if self.name == AttentionType.normalized:
            params += n_heads * head_dim
            params += n_kv_heads * head_dim

        # Per-head attention-sink logits.
        if self.attention_sinks:
            params += n_heads

        # Per-head scalable-softmax factors.
        if self.scalable_softmax:
            params += n_heads

        return params

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "SequenceMixer":
        """
        Build the corresponding attention module.

        :param d_model: The model dimensionality.
        :param init_device: The device to initialize the parameters on, e.g. "cpu", "meta".
        """
        if self.name == AttentionType.fused:
            warnings.warn(
                "The 'fused' attention implementation has been removed; using FusedAttentionV2. "
                "Checkpoint parameter names and shapes are preserved, but regular RoPE is "
                "not numerically identical to fused RoPE.",
                UserWarning,
                stacklevel=2,
            )
            rope = self.rope
            if rope is not None and rope.name == RoPEType.fused:
                rope = replace(rope, name=RoPEType.default)
            return replace(
                self,
                name=AttentionType.fused_v2,
                rope=rope,
                backend=self.backend or AttentionBackendName.flash_2,
                use_flash=None,  # The original implementation ignored this flag.
            ).build(
                d_model,
                layer_idx=layer_idx,
                n_layers=n_layers,
                init_device=init_device,
                cache=cache,
            )

        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")

        sliding_window_config: Optional[SlidingWindowAttentionConfig] = kwargs.pop(
            "sliding_window", None
        )
        if sliding_window_config is not None and sliding_window_config.should_use_swa(
            layer_idx, n_layers
        ):
            if self.scalable_softmax:
                raise OLMoConfigurationError(
                    "'scalable_softmax' is not supported with sliding window attention"
                )
            kwargs["window_size"] = sliding_window_config.get_window_size(layer_idx, n_layers)
        else:  # global (non-SWA) layer
            rope_config: Optional[RoPEConfig] = kwargs.get("rope")
            if rope_config is not None and rope_config.no_global_rope:
                kwargs["rope"] = None

        kwargs.update(
            dtype=kwargs.pop("dtype").as_pt(),
            d_model=d_model,
            init_device=init_device,
            cache=cache,
        )

        # Attention sinks are only wired up for the default attention; drop the flag otherwise so
        # the other implementations don't see an unexpected keyword argument.
        if not kwargs.get("attention_sinks", False):
            kwargs.pop("attention_sinks", None)
        elif self.name != AttentionType.default:
            raise OLMoConfigurationError("attention_sinks are only supported by default attention")

        # The MXFP8 packed-projection options are only wired up for fused_v2 attention; route them
        # there and reject them for any other implementation. A disabled (falsy) flag is a no-op, so
        # only reject when one is actually enabled.
        fused_v2_kwargs = {
            key: kwargs.pop(key)
            for key in ("mxfp8_projections", "mxfp8_qkv_projection", "mxfp8_out_projection")
            if key in kwargs
        }
        if any(fused_v2_kwargs.values()) and self.name != AttentionType.fused_v2:
            enabled = sorted(key for key, value in fused_v2_kwargs.items() if value)
            raise OLMoConfigurationError(f"{enabled} are only supported by fused_v2 attention")

        # QKV recompute / MXFP8-save are honored by the shared Attention.forward, so they apply to
        # the default and fused_v2 implementations (normalized overrides forward). As above,
        # only reject an enabled flag.
        shared_forward_kwargs = {
            key: kwargs.pop(key)
            for key in ("use_recompute_qkv_prep", "mxfp8_save_qkv_for_backward")
            if key in kwargs
        }
        if any(shared_forward_kwargs.values()) and self.name not in (
            AttentionType.default,
            AttentionType.fused_v2,
        ):
            enabled = sorted(key for key, value in shared_forward_kwargs.items() if value)
            raise OLMoConfigurationError(
                f"{enabled} are only supported by default and fused_v2 attention"
            )

        if self.name != AttentionType.default:
            for key in ("scalable_softmax", "qk_norm_per_head_gains"):
                if kwargs.pop(key, False):
                    raise OLMoConfigurationError(
                        f"'{key}' is not supported with {self.name} attention"
                    )

        try:
            if self.name == "default":
                return Attention(**kwargs, **shared_forward_kwargs)
            elif self.name == "fused_v2":
                return FusedAttentionV2(**kwargs, **fused_v2_kwargs, **shared_forward_kwargs)
            elif self.name == "normalized":
                if "window_size" in kwargs:
                    raise OLMoConfigurationError(
                        "'window_size' is not supported with normalized attention"
                    )
                return NormalizedAttention(**kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e


def _causal_attention_positions(seq_len: int, window_size: Optional[int] = None) -> int:
    """
    The number of attended ``(query, key)`` pairs across a sequence under causal masking, optionally
    capped by a sliding window.

    For full causal attention this is the triangle ``seq_len * (seq_len + 1) // 2`` (each query
    attends to itself and all earlier tokens). With a sliding ``window_size`` each query attends to
    at most ``window_size`` positions, so the early queries still form a triangle while the rest
    contribute a flat ``window_size`` each.

    :param seq_len: The sequence length.
    :param window_size: The sliding-window size, or ``None`` for full causal attention.
    """
    if window_size is None or window_size >= seq_len:
        return seq_len * (seq_len + 1) // 2
    return window_size * (window_size + 1) // 2 + (seq_len - window_size) * window_size


class Attention(SequenceMixer):
    """
    An implementation of multi-head self-attention with support for multi-query (MQA)
    and grouped-query (GQA) attention.

    Intra-document masking is also supported by passing in the
    ``max_doc_len`` and ``cu_doc_lens`` parameters to :meth:`forward()`. This requires
    a backend that supports it, like the flash backend.

    .. seealso::
        :class:`FusedAttentionV2` for a packed QKV projection.

    :param d_model: The model hidden size.
    :param n_heads: The number of attention heads.
    :param n_kv_heads: The number of key and value heads, if different.
    :param bias: Include biases with linear layers.
    :param gate: Configuration for attention gating. If None, no gating is applied.
    :param rope: The config for RoPE, if RoPE should be used.
    :param clip_qkv: Clip QKV to this value, if set.
    :param qk_norm: Configuration a layer norm for queries and keys.
    :param use_head_qk_norm: Apply the QK norm head-wise, i.e. to each head separately with
        normalization statistics computed over ``head_dim`` instead of the full hidden dimension.
    :param qk_norm_per_head_gains: Give each head its own norm gain (and bias) parameters instead
        of sharing them across heads. Requires ``use_head_qk_norm=True``.
    :param dropout: Dropout probability.
    :param use_flash: Deprecated, use ``backend="flash_2"`` instead.
    :param backend: The attention backend to use. If not set, it will be chosen automatically.
    :param scalable_softmax: Use Scalable-Softmax with a learned scale for each query head.
        Context parallelism, KV caching, and sliding window attention are unsupported.
    :param dtype: The default data type to use for parameters.
    :param init_device: The device to initialize weights on.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        bias: bool = True,
        gate: Optional[GateConfig] = None,
        rope: Optional[RoPEConfig] = None,
        clip_qkv: Optional[float] = None,
        qk_norm: Optional[LayerNormConfig] = None,
        dropout: float = 0.0,
        softmax_scale: Optional[float] = None,
        use_flash: Optional[bool] = None,
        backend: Optional[AttentionBackendName] = None,
        window_size: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
        use_head_qk_norm: bool = False,
        attention_sinks: bool = False,
        use_recompute_qkv_prep: bool = False,
        mxfp8_save_qkv_for_backward: bool = False,
        scalable_softmax: bool = False,
        qk_norm_per_head_gains: bool = False,
    ):
        super().__init__()

        self.use_recompute_qkv_prep = use_recompute_qkv_prep
        self.mxfp8_save_qkv_for_backward = mxfp8_save_qkv_for_backward
        self._mxfp8_saved_qkv_for_backward_last_pack_count = 0
        if qk_norm_per_head_gains and not use_head_qk_norm:
            raise OLMoConfigurationError(
                "'qk_norm_per_head_gains' requires 'use_head_qk_norm=True'"
            )

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.d_model = d_model
        # Some models (e.g. Qwen3) use explicit head_dim that differs from d_model // n_heads.
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            self.head_dim = d_model // n_heads
        self.w_q = nn.Linear(
            d_model, n_heads * self.head_dim, bias=bias, dtype=dtype, device=init_device
        )
        self.w_k = nn.Linear(
            d_model, self.n_kv_heads * self.head_dim, bias=bias, dtype=dtype, device=init_device
        )
        self.w_v = nn.Linear(
            d_model, self.n_kv_heads * self.head_dim, bias=bias, dtype=dtype, device=init_device
        )
        self.w_out = nn.Linear(
            n_heads * self.head_dim, d_model, bias=bias, dtype=dtype, device=init_device
        )

        self.gate = gate
        self.w_g: Optional[nn.Linear] = None
        if gate is not None:
            if gate.granularity == GateGranularity.headwise:
                self.w_g = nn.Linear(
                    d_model, self.n_heads, bias=bias, dtype=dtype, device=init_device
                )
            elif gate.granularity == GateGranularity.elementwise:
                self.w_g = nn.Linear(
                    d_model,
                    self.n_heads * self.head_dim,
                    bias=bias,
                    dtype=dtype,
                    device=init_device,
                )

        self.clip_qkv = clip_qkv
        self.use_head_qk_norm = use_head_qk_norm
        self.scalable_softmax = scalable_softmax
        self.ssmax_scale: Optional[nn.Parameter] = None
        if scalable_softmax:
            if window_size is not None:
                raise OLMoConfigurationError(
                    "'scalable_softmax' is not supported with sliding window attention"
                )
            self.ssmax_scale = nn.Parameter(torch.ones(n_heads, dtype=dtype, device=init_device))

        # Per-head learnable attention-sink logits (GPT-OSS). See :meth:`sdpa`.
        self.sinks: Optional[nn.Parameter] = (
            nn.Parameter(torch.empty(n_heads, dtype=dtype, device=init_device))
            if attention_sinks
            else None
        )

        self.q_norm: Optional[LayerNorm] = None
        self.k_norm: Optional[LayerNorm] = None
        if qk_norm is not None:
            if use_head_qk_norm:
                q_weight_shape: Optional[Tuple[int, ...]] = None
                k_weight_shape: Optional[Tuple[int, ...]] = None
                if qk_norm_per_head_gains:
                    q_weight_shape = (n_heads, self.head_dim)
                    k_weight_shape = (self.n_kv_heads, self.head_dim)
                self.q_norm = qk_norm.build(
                    size=self.head_dim, init_device=init_device, weight_shape=q_weight_shape
                )
                self.k_norm = qk_norm.build(
                    size=self.head_dim, init_device=init_device, weight_shape=k_weight_shape
                )
            else:
                self.q_norm = qk_norm.build(size=n_heads * self.head_dim, init_device=init_device)
                self.k_norm = qk_norm.build(
                    size=self.n_kv_heads * self.head_dim, init_device=init_device
                )

        self.rope: Optional[Union[RotaryEmbedding, ComplexRotaryEmbedding]] = None
        if rope is not None:
            if rope.name == "fused":
                raise OLMoConfigurationError(
                    f"fused RoPE is not compatible with {self.__class__.__name__}"
                )
            rope_class = rope.build(self.head_dim, cache=cache)
            assert isinstance(rope_class, (RotaryEmbedding, ComplexRotaryEmbedding))
            self.rope = rope_class

        backend = _resolve_attention_backend(backend, use_flash)

        # Translate window size so that we only look left, not right.
        self.window_size = window_size
        window_size_tuple: Tuple[int, int] = (-1, -1)
        if window_size is not None:
            if window_size <= 0:
                raise OLMoConfigurationError(f"'window_size' must be positive (got {window_size})")

            if backend is None and flash_attn_api.has_flash_attn_2():
                # note: flash_3, flash_4, and te backends are faster than flash_2 and also support SWA
                backend = AttentionBackendName.flash_2

            # Window size is [i - window_size[0], i + window_size[1]] inclusive
            window_size_tuple = (window_size - 1, 0)

        if backend is None:
            backend = AttentionBackendName.torch

        # Reject an unsupported sinks/backend combination at construction (before the CPU fallback
        # below) rather than deferring the failure to the first forward pass.
        if attention_sinks and backend != AttentionBackendName.torch:
            raise OLMoConfigurationError(
                f"attention_sinks are only supported by the torch attention backend (got '{backend}')"
            )

        if not torch.cuda.is_available() and backend != AttentionBackendName.torch:
            warnings.warn(
                f"Backend is set to {backend}, but GPUs are not available. Defaulting to torch."
            )
            backend = AttentionBackendName.torch

        backend.assert_supported()
        log.info(f"Using attention backend '{backend}'")
        self.backend = backend.build(
            head_dim=self.head_dim,
            n_heads=n_heads,
            n_kv_heads=self.n_kv_heads,
            scale=softmax_scale,
            dropout_p=dropout,
            window_size=window_size_tuple,
            cache=cache,
        )
        self.kv_cache_manager: Optional[KVCacheManager] = None

    @property
    def cp_enabled(self) -> bool:
        return self.backend.cp_enabled

    def sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        cache_leftpad: Optional[torch.Tensor] = None,
        or_mask: Optional[torch.Tensor] = None,
        and_mask: Optional[torch.Tensor] = None,
        flex_attn_is_image: Optional[torch.Tensor] = None,
        flex_attn_subsegment_ids: Optional[torch.Tensor] = None,
        flex_attn_example_ids: Optional[torch.Tensor] = None,
        flex_attn_block_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.record_leftpad(cache_leftpad)
        if or_mask is not None and not self.backend.SUPPORTS_OR_MASK:
            raise NotImplementedError(
                f"'{type(self.backend).__name__}' does not support `or_mask` "
                "(e.g. bidirectional image-token attention); use the 'torch' attention backend."
            )
        if and_mask is not None and not self.backend.SUPPORTS_AND_MASK:
            raise NotImplementedError(
                f"'{type(self.backend).__name__}' does not support `and_mask` "
                "(e.g. subsegment / branch isolation); use the 'torch' attention backend."
            )
        # shape: (batch_size, seq_len, n_heads, head_dim)
        backend_kwargs: Dict[str, Optional[torch.Tensor]] = dict(
            or_mask=or_mask,
            and_mask=and_mask,
        )
        from olmo_core.nn.attention.backend import FlexAttentionBackend

        if isinstance(self.backend, FlexAttentionBackend):
            backend_kwargs.update(
                flex_attn_is_image=flex_attn_is_image,
                flex_attn_subsegment_ids=flex_attn_subsegment_ids,
                flex_attn_example_ids=flex_attn_example_ids,
                flex_attn_block_mask=flex_attn_block_mask,
            )
        att = self.backend(
            (q, k, v),
            cu_doc_lens=cu_doc_lens,
            cu_doc_lens_q=cu_doc_lens_q,
            cu_doc_lens_k=cu_doc_lens_k,
            max_doc_len=max_doc_len,
            max_doc_len_q=max_doc_len_q,
            max_doc_len_k=max_doc_len_k,
            local_k_slice=local_k_slice,
            kv_cache_manager=self.kv_cache_manager,
            sinks=self.sinks,
            **backend_kwargs,
        )
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.update_seqlen(q.shape[1])
        return att

    def _apply_scalable_softmax(
        self,
        q: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.scalable_softmax:
            return q
        if self.cp_enabled:
            raise NotImplementedError("Scalable-Softmax is not supported with context parallelism")
        if self.kv_cache_manager is not None:
            raise NotImplementedError("Scalable-Softmax is not supported with KV caching")

        if cu_doc_lens is None:
            visible_lengths = torch.arange(1, q.shape[1] + 1, device=q.device)
            visible_lengths = visible_lengths.unsqueeze(0).expand(q.shape[0], -1)
        else:
            boundaries = cu_doc_lens.to(device=q.device)
            token_indices = torch.arange(
                q.shape[0] * q.shape[1], device=q.device, dtype=boundaries.dtype
            )
            document_indices = torch.searchsorted(boundaries[1:], token_indices, right=True)
            document_starts = boundaries[document_indices]
            visible_lengths = (token_indices - document_starts + 1).view(q.shape[0], q.shape[1])

        assert self.ssmax_scale is not None
        ssmax_scale = self.ssmax_scale
        if isinstance(ssmax_scale, DTensor):
            ssmax_scale = ssmax_scale.to_local()
        scale = visible_lengths.log().to(q.dtype).unsqueeze(-1)
        scale = scale * ssmax_scale.to(q.dtype).view(1, 1, -1)
        return q * scale.unsqueeze(-1)

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        start_pos: Optional[int],
        pos_sin: Optional[torch.Tensor],
        pos_cos: Optional[torch.Tensor],
        freqs_cis: Optional[torch.Tensor],
        cu_doc_lens: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.rope is not None
        rope_kwargs = {}
        if cu_doc_lens is not None:
            if not isinstance(self.rope, RotaryEmbedding):
                raise NotImplementedError(
                    "Intra-document RoPE (cu_doc_lens) is only supported by RotaryEmbedding; "
                    f"got {type(self.rope).__name__}"
                )
            rope_kwargs["cu_doc_lens"] = cu_doc_lens
        if position_ids is not None:
            if not isinstance(self.rope, RotaryEmbedding):
                raise NotImplementedError(
                    "Explicit `position_ids` RoPE is only supported by RotaryEmbedding; "
                    f"got {type(self.rope).__name__}"
                )
            rope_kwargs["position_ids"] = position_ids
        return self.rope(
            q,
            k,
            head_first=False,
            start_pos=start_pos,
            pos_sin=pos_sin,
            pos_cos=pos_cos,
            freqs_cis=freqs_cis,
            **rope_kwargs,
        )

    def _prepare_qkv(
        self,
        x: torch.Tensor,
        *,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        start_pos: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project the input into per-head Q/K/V (with clip, QK norm, and RoPE applied), returning
        tensors shaped ``(batch_size, seq_len, n_heads (local), head_dim)``. Subclasses can override
        this to change the projection layout (e.g. a packed QKV projection) while reusing
        :meth:`forward`.
        """
        B, T, _ = x.shape

        # shape: (batch_size, seq_len, n_heads * head_dim),
        #        (batch_size, seq_len, n_kv_heads * head_dim),
        #        (batch_size, seq_len, n_kv_heads * head_dim)
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        if self.clip_qkv is not None:
            q.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
            k.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
            v.clamp_(min=-self.clip_qkv, max=self.clip_qkv)

        if not self.use_head_qk_norm:
            if self.q_norm is not None:
                q = self.q_norm(q)
            if self.k_norm is not None:
                k = self.k_norm(k)

        # NOTE: use -1 instead of `n_heads` / `n_kv_heads` to infer actual local size when
        # using tensor parallelism.
        # shape: (batch_size, seq_len, n_heads (local), head_dim)
        q = q.view(B, T, -1, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads (local), head_dim)
        k = k.view(B, T, -1, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads (local), head_dim)
        v = v.view(B, T, -1, self.head_dim)

        if self.use_head_qk_norm:
            # NOTE: with per-head gains ('qk_norm_per_head_gains=True') the norm weight has shape
            # (n_heads, head_dim), and correctness relies on applying the norm *after* the
            # head-wise view above so that head 'h' broadcasts against gain row 'h'. Under tensor
            # parallelism this still holds: the norm runs in the sequence-sharded region where all
            # heads are present locally and the weight is replicated.
            if self.q_norm is not None:
                q = self.q_norm(q)
            if self.k_norm is not None:
                k = self.k_norm(k)

        if self.rope is not None:
            # In context-parallel mode we must be given pre-sharded buffers
            if self.cp_enabled and pos_sin is None and pos_cos is None and freqs_cis is None:
                raise RuntimeError(
                    "RoPE buffers must be passed through to attention after being properly "
                    "sharded by the context parallel load balancer"
                )

            q, k = self._apply_rope(
                q, k, start_pos, pos_sin, pos_cos, freqs_cis, cu_doc_lens, position_ids
            )

        q = self._apply_scalable_softmax(q, cu_doc_lens)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        cache_leftpad: Optional[torch.Tensor] = None,
        or_mask: Optional[torch.Tensor] = None,
        and_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        flex_attn_is_image: Optional[torch.Tensor] = None,
        flex_attn_subsegment_ids: Optional[torch.Tensor] = None,
        flex_attn_example_ids: Optional[torch.Tensor] = None,
        flex_attn_block_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply attention to the input.

        :param x: The input of shape ``(batch_size, seq_len, d_model)``.
        :param cu_doc_lens: Cumulative document lengths in the input ``x``, a 1D
            :class:`torch.int32` tensor that should always have one more element than there
            are documents (the first element in the tensor should always be ``0``).
            Required together with ``max_doc_len`` when using intra-document masking.
        :param max_doc_len: The maximum document length in the input ``x``.
            Required together with ``cu_doc_lens`` when using intra-document masking.

        :returns: The output of attention with shape ``(batch_size, seq_len, d_model)``.
        """
        B, T, _ = x.shape

        start_pos = self.kv_cache_manager.current_position() if self.kv_cache_manager else None

        # Optionally recompute Q/K/V in backward (trading compute for activation memory) by wrapping
        # the projection in an OutputDiscardCheckpoint.
        qkv_checkpoint: Optional[OutputDiscardCheckpoint] = None
        if torch.is_grad_enabled() and x.requires_grad and self.use_recompute_qkv_prep:
            qkv_checkpoint = OutputDiscardCheckpoint()
            q, k, v = cast(
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                qkv_checkpoint.checkpoint(
                    self._prepare_qkv,
                    x,
                    pos_sin=pos_sin,
                    pos_cos=pos_cos,
                    freqs_cis=freqs_cis,
                    start_pos=start_pos,
                    cu_doc_lens=cu_doc_lens,
                    position_ids=position_ids,
                ),
            )
        else:
            q, k, v = self._prepare_qkv(
                x,
                pos_sin=pos_sin,
                pos_cos=pos_cos,
                freqs_cis=freqs_cis,
                start_pos=start_pos,
                cu_doc_lens=cu_doc_lens,
                position_ids=position_ids,
            )

        # Optionally save Q/K/V for backward as MXFP8 to reduce the saved-activation footprint.
        self._mxfp8_saved_qkv_for_backward_last_pack_count = 0
        qkv_save_counter = [0]
        if (
            torch.is_grad_enabled()
            and self.mxfp8_save_qkv_for_backward
            and any(t.requires_grad for t in (q, k, v))
        ):
            qkv_save_context: Any = _mxfp8_saved_qkv_hooks(q, k, v, pack_counter=qkv_save_counter)
        else:
            qkv_save_context = nullcontext()

        # shape: (batch_size, seq_len, n_heads, head_dim)
        with qkv_save_context:
            att = self.sdpa(
                q,
                k,
                v,
                cu_doc_lens=cu_doc_lens,
                cu_doc_lens_q=cu_doc_lens_q,
                cu_doc_lens_k=cu_doc_lens_k,
                max_doc_len=max_doc_len,
                max_doc_len_q=max_doc_len_q,
                max_doc_len_k=max_doc_len_k,
                local_k_slice=local_k_slice,
                cache_leftpad=cache_leftpad,
                or_mask=or_mask,
                and_mask=and_mask,
                flex_attn_is_image=flex_attn_is_image,
                flex_attn_subsegment_ids=flex_attn_subsegment_ids,
                flex_attn_example_ids=flex_attn_example_ids,
                flex_attn_block_mask=flex_attn_block_mask,
            )
        self._mxfp8_saved_qkv_for_backward_last_pack_count = qkv_save_counter[0]
        if qkv_checkpoint is not None:
            # Recompute Q/K/V before attention backward consumes the discarded activations.
            qkv_checkpoint.discard_output_and_register_recompute(att)

        if self.gate is not None:
            assert self.w_g is not None
            g = self.w_g(x)
            if self.gate.full_precision:
                g = g.float()
            gate_values = torch.sigmoid(g).to(att.dtype)
            if self.gate.granularity == GateGranularity.headwise:
                # head-wise gating is broadcast across head_dim
                # shape: (batch_size, seq_len, n_heads, head_dim)
                att = att * gate_values.unsqueeze(-1)
            elif self.gate.granularity == GateGranularity.elementwise:
                att = att.view(B, T, -1) * gate_values
                # the following att.view op is redundant (a no-op)

        # shape: (batch_size, seq_len, d_model)
        att = att.view(B, T, -1)

        # shape: (batch_size, seq_len, d_model)
        return self.w_out(att)

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        rowwise_parallel, colwise_parallel, prepare_module_input = get_tp_wrappers(
            float8_enabled=float8_enabled
        )

        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=prepare_module_input(
                input_layouts=None if input_layout is None else (input_layout,),
                desired_input_layouts=(Replicate(),),
            ),
        )

        plan = {
            "w_q": colwise_parallel(
                output_layouts=None if self.q_norm is None else Shard(1),
                use_local_output=self.q_norm is None,
            ),
            "w_k": colwise_parallel(
                output_layouts=None if self.k_norm is None else Shard(1),
                use_local_output=self.k_norm is None,
            ),
            "w_v": colwise_parallel(),
            "w_out": rowwise_parallel(
                output_layouts=output_layout, use_local_output=use_local_output
            ),
        }

        if self.w_g is not None:
            plan["w_g"] = colwise_parallel()

        if self.q_norm is not None:
            # if full-dim norm: output is sharded on the embedding dimension (B, T, E [sharded])
            #    which will be reshaped into (B, T, H [sharded], D)
            # if head-wise norm: output is sharded on the head dimension (B, T, H [sharded], D)
            plan["q_norm"] = SequenceParallel(use_local_output=True, output_layouts=Shard(2))

        if self.ssmax_scale is not None:
            self.register_parameter(
                "ssmax_scale",
                nn.Parameter(distribute_tensor(self.ssmax_scale, tp_mesh, [Shard(0)])),
            )
        if self.k_norm is not None:
            plan["k_norm"] = SequenceParallel(use_local_output=True, output_layouts=Shard(2))

        parallelize_module(
            module=self,
            device_mesh=tp_mesh,
            parallelize_plan=plan,
        )

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        """
        Prepare the module for context-parallelism (ring attention).

        .. important::
            This requires a backend that supports CP, such as "flash_2" or "te".

        :param cp_mesh: The context parallel device sub-mesh.
        :param ring: The ring context parallel style.
        :param uly: The ulysses context parallel style.
        """
        if self.scalable_softmax:
            raise OLMoConfigurationError(
                "Scalable-Softmax is not supported with context parallelism"
            )
        self.backend.apply_cp(cp_mesh, ring=ring, uly=uly)

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

        if self.ssmax_scale is not None:
            nn.init.ones_(self.ssmax_scale)

        # Compute std for Q/K/V initialization
        if init_method == InitMethod.fan_in:
            # For fan_in, use 1/√d_in based on actual weight shape (ignores base std parameter)
            # Each projection may have different output dims (n_heads * head_dim vs n_kv_heads * head_dim)
            # but they all have the same input dim
            for w in (self.w_q, self.w_k, self.w_v):
                w_std = w.in_features**-0.5
                init_linear(w, std=w_std, generator=generator)
        else:
            if init_method == InitMethod.normalized:
                std = d_model**-0.5
            for w in (self.w_q, self.w_k, self.w_v):
                init_linear(w, std=std, generator=generator)

        # Initialize attention gate projection if present
        if self.w_g is not None:
            if init_method == InitMethod.fan_in:
                g_std = self.w_g.in_features**-0.5
            else:
                g_std = std
            init_linear(self.w_g, std=g_std, generator=generator)

        # Compute std for w_out initialization
        if init_method == InitMethod.fan_in:
            std = self.w_out.in_features**-0.5
        elif init_method == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5
        elif init_method == InitMethod.normalized:
            std = std / (2 * num_blocks) ** 0.5

        init_linear(self.w_out, std=std, generator=generator)

        if self.sinks is not None:
            nn.init.normal_(self.sinks, mean=0.0, std=std, generator=generator)

    def init_kv_cache_manager(self, batch_size: int, max_seq_len: int):
        """
        Initialize the kv cache manager for attention. When the kv cache manager exists,
        kv caching will be used during the forward pass. This should only be called during inference.

        :param batch_size: The batch size for the cache.
        :param max_seq_len: The maximum sequence length for the cache.
        """
        if self.scalable_softmax:
            raise OLMoConfigurationError("Scalable-Softmax is not supported with KV caching")
        self.backend.assert_supports_kv_cache()

        self.kv_cache_manager = KVCacheManager(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            device=self.w_k.weight.device,
        )

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        This accounts for:
        - Linear projections (Q, K, V, output, and gating if enabled)
        - Attention computation (QK^T and softmax(QK^T) @ V)
        - Sliding window attention (reduced effective sequence length)
        """
        # 6 FLOPs per parameter (2 ops * 3 for forward+backward)
        param_flops = 6 * sum(p.numel() for p in self.parameters())

        # Attention computation (QK^T and Attn*V).
        # 12x multiplier: 2 matmuls * 2 ops each * 3 for forward+backward.
        # Note that flash attention technically uses more flops (14x multiplier) due to recomputation,
        # however, we just compute the idealized flops for SDPA.
        #
        # Historical note: this previously counted ``12 * n_heads * head_dim * effective_seq_len``
        # with ``effective_seq_len = min(window_size, seq_len)``, which ignored causal masking and
        # used a flat sliding-window cap. That overcounted full-attention layers by ~2x (no causal
        # /2) and only roughly approximated sliding-window layers. We now count the exact causal /
        # sliding-window ``(query, key)`` pairs, so reported model TFLOPs / MFU for full-attention
        # runs drop on the attention-compute term relative to logs produced before this change.
        attention_positions = _causal_attention_positions(seq_len, self.window_size or None)
        attn_flops = 12 * self.n_heads * self.head_dim * attention_positions // seq_len

        return param_flops + attn_flops


@beta_feature
class NormalizedAttention(Attention):
    """
    An nGPT attention implementation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        rope: Optional[RoPEConfig] = None,
        qk_norm: Optional[LayerNormConfig] = None,
        use_flash: Optional[bool] = None,
        backend: Optional[AttentionBackendName] = None,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            rope=rope,
            qk_norm=qk_norm,
            use_flash=use_flash,
            backend=backend,
            softmax_scale=math.sqrt(d_model // n_heads),
            bias=False,
            dtype=dtype,
            init_device=init_device,
            cache=cache,
        )

        self.sq_init_value = 1.0
        self.sq_init_scaling = 1.0 / math.sqrt(d_model)
        self.sq = nn.Parameter(
            torch.empty(self.head_dim * self.n_heads, dtype=dtype, device=init_device)
        )

        self.sk_init_value = 1.0
        self.sk_init_scaling = 1.0 / math.sqrt(d_model)
        self.sk = nn.Parameter(
            torch.empty(self.head_dim * self.n_kv_heads, dtype=dtype, device=init_device)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.sq)
        nn.init.ones_(self.sk)
        with torch.no_grad():
            self.sq.mul_(self.sq_init_scaling)
            self.sk.mul_(self.sk_init_scaling)

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        cache_leftpad: Optional[torch.Tensor] = None,
        or_mask: Optional[torch.Tensor] = None,
        and_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cache_leftpad:
            raise NotImplementedError(
                "cache_leftpad is not supported for the normalized attention variant"
            )

        B, T, _ = x.shape

        # shape: (batch_size, seq_len, n_heads * head_dim),
        #        (batch_size, seq_len, n_kv_heads * head_dim),
        #        (batch_size, seq_len, n_kv_heads * head_dim)
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        sq = (self.sq * (self.sq_init_value / self.sq_init_scaling)).view(1, 1, -1)
        q = sq * q

        sk = (self.sk * (self.sk_init_value / self.sk_init_scaling)).view(1, 1, -1)
        k = sk * k

        # shape: (batch_size, seq_len, n_heads, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads, head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads, head_dim)
        v = v.view(B, T, self.n_kv_heads, self.head_dim)

        if self.rope is not None:
            if self.cp_enabled and pos_sin is None and pos_cos is None and freqs_cis is None:
                raise RuntimeError(
                    "RoPE buffers must be passed through to attention after being properly "
                    "sharded by the context parallel load balancer"
                )

            start_pos = self.kv_cache_manager.current_position() if self.kv_cache_manager else None
            q, k = self._apply_rope(
                q, k, start_pos, pos_sin, pos_cos, freqs_cis, cu_doc_lens, position_ids
            )

        # shape: (batch_size, seq_len, n_heads, head_dim)
        att = self.sdpa(
            q,
            k,
            v,
            cu_doc_lens=cu_doc_lens,
            cu_doc_lens_q=cu_doc_lens_q,
            cu_doc_lens_k=cu_doc_lens_k,
            max_doc_len=max_doc_len,
            max_doc_len_q=max_doc_len_q,
            max_doc_len_k=max_doc_len_k,
            local_k_slice=local_k_slice,
            cache_leftpad=cache_leftpad,
            or_mask=or_mask,
            and_mask=and_mask,
        )

        # shape: (batch_size, seq_len, d_model)
        att = att.view(B, T, -1)

        # shape: (batch_size, seq_len, d_model)
        return self.w_out(att)

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled

        raise NotImplementedError("TP is not implemented yet for the normalized attention variant")

    @torch.no_grad()
    def normalize_matrices(self):
        """
        Normalize the weights in all matrices. This should be called after each optimizer step, which
        the :class:`~olmo_core.train.train_module.TransformerTrainModule` will handle for you.
        """
        self._normalize_matrix(self.w_q.weight)
        self._normalize_matrix(self.w_k.weight)
        self._normalize_matrix(self.w_v.weight)
        self._normalize_matrix(self.w_out.weight, dim=0)

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(l2_normalize(w, dim=dim))


class FusedAttentionV2(Attention):
    """
    A packed-projection variant of :class:`Attention`.

    This keeps the regular attention backend contract by unpacking Q/K/V after a
    single packed projection, so it can support features such as GQA, QK norm,
    gating, RoPE, sliding window attention, and KV caching without requiring a
    packed-QKV attention kernel.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        bias: bool = True,
        gate: Optional[GateConfig] = None,
        rope: Optional[RoPEConfig] = None,
        clip_qkv: Optional[float] = None,
        qk_norm: Optional[LayerNormConfig] = None,
        dropout: float = 0.0,
        softmax_scale: Optional[float] = None,
        use_flash: Optional[bool] = None,
        backend: Optional[AttentionBackendName] = None,
        window_size: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
        use_head_qk_norm: bool = False,
        mxfp8_projections: bool = False,
        mxfp8_qkv_projection: Optional[bool] = None,
        mxfp8_out_projection: Optional[bool] = None,
        use_recompute_qkv_prep: bool = False,
        mxfp8_save_qkv_for_backward: bool = False,
    ):
        nn.Module.__init__(self)

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.d_model = d_model
        self.n_rep = self.n_heads // self.n_kv_heads
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            self.head_dim = d_model // n_heads
        self.sinks = None

        self.q_dim = n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim
        qkv_dim = self.q_dim + 2 * self.kv_dim
        if mxfp8_qkv_projection is None:
            mxfp8_qkv_projection = mxfp8_projections
        if mxfp8_out_projection is None:
            mxfp8_out_projection = mxfp8_projections
        self.mxfp8_projections = bool(mxfp8_qkv_projection or mxfp8_out_projection)
        self.mxfp8_qkv_projection = bool(mxfp8_qkv_projection)
        self.mxfp8_out_projection = bool(mxfp8_out_projection)
        if self.mxfp8_qkv_projection:
            self._validate_mxfp8_projection_shape("w_qkv", d_model, qkv_dim)
            self.w_qkv = MXFP8Linear(
                d_model,
                qkv_dim,
                bias=bias,
                dtype=dtype,
                device=init_device,
                save_wgrad_input="mxfp8",
            )
        else:
            self.w_qkv = nn.Linear(
                d_model,
                qkv_dim,
                bias=bias,
                dtype=dtype,
                device=init_device,
            )

        if self.mxfp8_out_projection:
            self._validate_mxfp8_projection_shape("w_out", self.q_dim, d_model)
            self.w_out = MXFP8Linear(
                self.q_dim,
                d_model,
                bias=bias,
                dtype=dtype,
                device=init_device,
                # SDPA already saves its bf16 output for backward, so saving the w_out input as
                # MXFP8 too would just keep an extra copy of the activations and waste memory.
                save_wgrad_input="bf16",
            )
        else:
            self.w_out = nn.Linear(self.q_dim, d_model, bias=bias, dtype=dtype, device=init_device)

        self.gate = gate
        self.w_g: Optional[nn.Linear] = None
        if gate is not None:
            if gate.granularity == GateGranularity.headwise:
                self.w_g = nn.Linear(
                    d_model, self.n_heads, bias=bias, dtype=dtype, device=init_device
                )
            elif gate.granularity == GateGranularity.elementwise:
                self.w_g = nn.Linear(
                    d_model,
                    self.q_dim,
                    bias=bias,
                    dtype=dtype,
                    device=init_device,
                )

        self.clip_qkv = clip_qkv
        self.use_head_qk_norm = use_head_qk_norm
        self.use_recompute_qkv_prep = use_recompute_qkv_prep
        self.mxfp8_save_qkv_for_backward = mxfp8_save_qkv_for_backward
        self._mxfp8_saved_qkv_for_backward_last_pack_count = 0

        self.q_norm: Optional[LayerNorm] = None
        self.k_norm: Optional[LayerNorm] = None
        if qk_norm is not None:
            if use_head_qk_norm:
                self.q_norm = qk_norm.build(size=self.head_dim, init_device=init_device)
                self.k_norm = qk_norm.build(size=self.head_dim, init_device=init_device)
            else:
                self.q_norm = qk_norm.build(size=self.q_dim, init_device=init_device)
                self.k_norm = qk_norm.build(size=self.kv_dim, init_device=init_device)

        self.rope: Optional[Union[RotaryEmbedding, ComplexRotaryEmbedding]] = None
        if rope is not None:
            if rope.name == "fused":
                raise OLMoConfigurationError(
                    f"fused RoPE is not compatible with {self.__class__.__name__}"
                )
            rope_class = rope.build(self.head_dim, cache=cache)
            assert isinstance(rope_class, (RotaryEmbedding, ComplexRotaryEmbedding))
            self.rope = rope_class

        backend = _resolve_attention_backend(backend, use_flash)

        self.window_size = window_size
        window_size_tuple: Tuple[int, int] = (-1, -1)
        if window_size is not None:
            if window_size <= 0:
                raise OLMoConfigurationError(f"'window_size' must be positive (got {window_size})")

            if backend is None and flash_attn_api.has_flash_attn_2():
                backend = AttentionBackendName.flash_2

            window_size_tuple = (window_size - 1, 0)

        if backend is None:
            backend = AttentionBackendName.torch

        if not torch.cuda.is_available() and backend != AttentionBackendName.torch:
            warnings.warn(
                f"Backend is set to {backend}, but GPUs are not available. Defaulting to torch."
            )
            backend = AttentionBackendName.torch

        backend.assert_supported()
        log.info(f"Using attention backend '{backend}'")
        self.backend = backend.build(
            head_dim=self.head_dim,
            n_heads=n_heads,
            n_kv_heads=self.n_kv_heads,
            scale=softmax_scale,
            dropout_p=dropout,
            window_size=window_size_tuple,
            cache=cache,
        )
        self.kv_cache_manager: Optional[KVCacheManager] = None

    @staticmethod
    def _validate_mxfp8_projection_shape(name: str, in_features: int, out_features: int) -> None:
        if in_features % 32 != 0 or out_features % 32 != 0:
            raise OLMoConfigurationError(
                f"MXFP8 FusedAttentionV2 projection '{name}' requires in/out features "
                f"to be divisible by 32, got in_features={in_features}, "
                f"out_features={out_features}"
            )

    def named_fp8_weight_stores(self) -> Iterator[tuple[str, object]]:
        for module_name in ("w_qkv", "w_out"):
            module = getattr(self, module_name)
            named_weight_stores = getattr(module, "named_fp8_weight_stores", None)
            if named_weight_stores is None:
                continue
            for name, weight in named_weight_stores():
                yield f"{module_name}.{name}", weight

    def named_mxfp8_attention_weights(self) -> Iterator[tuple[str, object]]:
        yield from self.named_fp8_weight_stores()

    def zero_mxfp8_attention_weight_grads(self, set_to_none: bool = True) -> None:
        for module in (self.w_qkv, self.w_out):
            zero_grads = getattr(module, "zero_mxfp8_weight_grads", None)
            if zero_grads is not None:
                zero_grads(set_to_none=set_to_none)

    def set_mxfp8_attention_weight_main_grads_to_none(self) -> None:
        for module in (self.w_qkv, self.w_out):
            set_main_grads_to_none = getattr(module, "set_mxfp8_weight_main_grads_to_none", None)
            if set_main_grads_to_none is not None:
                set_main_grads_to_none()

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        self.zero_mxfp8_attention_weight_grads(set_to_none=set_to_none)

    def disable_mxfp8_attention_anchor_grads(self) -> None:
        for module in (self.w_qkv, self.w_out):
            disable_grads = getattr(module, "disable_mxfp8_anchor_grads", None)
            if disable_grads is not None:
                disable_grads()

    def release_mxfp8_attention_anchor_storage(self) -> None:
        for module in (self.w_qkv, self.w_out):
            release_storage = getattr(module, "release_mxfp8_anchor_storage", None)
            if release_storage is not None:
                release_storage()

    @torch.no_grad()
    def refresh_mxfp8_attention_cache(self) -> None:
        for module in (self.w_qkv, self.w_out):
            refresh_cache = getattr(module, "refresh_mxfp8_cache", None)
            if refresh_cache is not None:
                refresh_cache()

    def invalidate_mxfp8_attention_cache(self) -> None:
        for module in (self.w_qkv, self.w_out):
            invalidate_cache = getattr(module, "invalidate_mxfp8_cache", None)
            if invalidate_cache is not None:
                invalidate_cache()

    def _prepare_qkv(
        self,
        x: torch.Tensor,
        *,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        start_pos: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape

        qkv = self.w_qkv(x)

        if self.clip_qkv is not None:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)

        q, k, v = qkv.split((self.q_dim, self.kv_dim, self.kv_dim), dim=-1)

        if not self.use_head_qk_norm:
            if self.q_norm is not None:
                q = self.q_norm(q)
            if self.k_norm is not None:
                k = self.k_norm(k)

        # NOTE: use -1 instead of `n_heads` / `n_kv_heads` to infer actual local size when
        # using tensor parallelism.
        # shape: (batch_size, seq_len, n_heads (local), head_dim)
        q = q.view(B, T, -1, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads (local), head_dim)
        k = k.view(B, T, -1, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads (local), head_dim)
        v = v.view(B, T, -1, self.head_dim)

        if self.use_head_qk_norm:
            if self.q_norm is not None:
                q = self.q_norm(q)
            if self.k_norm is not None:
                k = self.k_norm(k)

        if self.rope is not None:
            if self.cp_enabled and pos_sin is None and pos_cos is None and freqs_cis is None:
                raise RuntimeError(
                    "RoPE buffers must be passed through to attention after being properly "
                    "sharded by the context parallel load balancer"
                )

            q, k = self._apply_rope(
                q, k, start_pos, pos_sin, pos_cos, freqs_cis, cu_doc_lens, position_ids
            )

        return q, k, v

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled

        raise NotImplementedError("TP is not implemented yet for FusedAttentionV2")

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

        if init_method == InitMethod.fan_in:
            std = self.w_qkv.in_features**-0.5
        elif init_method == InitMethod.normalized:
            std = d_model**-0.5

        init_linear(self.w_qkv, std=std, generator=generator)

        if self.w_g is not None:
            if init_method == InitMethod.fan_in:
                g_std = self.w_g.in_features**-0.5
            else:
                g_std = std
            init_linear(self.w_g, std=g_std, generator=generator)

        if init_method == InitMethod.fan_in:
            std = self.w_out.in_features**-0.5
        elif init_method == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5
        elif init_method == InitMethod.normalized:
            std = std / (2 * num_blocks) ** 0.5

        init_linear(self.w_out, std=std, generator=generator)

    def init_kv_cache_manager(self, batch_size: int, max_seq_len: int):
        self.backend.assert_supports_kv_cache()

        self.kv_cache_manager = KVCacheManager(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            device=self.w_qkv.weight.device,
        )
