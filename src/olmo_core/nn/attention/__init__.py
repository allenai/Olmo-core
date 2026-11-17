import logging
import math
import os
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Set, Tuple, Union, cast

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
from olmo_core.distributed.parallel.context_parallel import (
    all_to_all_cp2hp,
    all_to_all_single_cp2hp,
    all_to_all_single_hp2cp,
)
from olmo_core.distributed.parallel.tensor_parallel import SequenceParallel
from olmo_core.distributed.utils import get_rank, get_world_size
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
from .landmark import (
    build_block_doc_id,
    build_local_packed_position_ids,
    landmark_grouped_softmax,
    repeat_kv,
)
from .landmark_kernel import fused_landmark_attention, has_landmark_kernel
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
    "AttentionTypePatternConfig",
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
    "LandmarkAttention",
    "FastLandmarkAttention",
    "FastCompressiveLandmarkAttention",
    "CompressiveGQAGroupedAttention",
    "MultiCompressiveLandmarkAttention",
    "SparseLandmarkAttention",
    "DocumentLandmarkAttention",
    "DocumentCompressiveLandmarkAttention",
    "MultiLandmarkAttention",
    "DocumentMultiLandmarkAttention",
    "DocumentChunkedAttention",
    "DilatedSlidingWindowAttention",
    "AttentionPattern",
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
    landmark = "landmark"
    """
    ➡️ :class:`LandmarkAttention`
    """

    fast_landmark = "fast_landmark"
    """
    ➡️ :class:`FastLandmarkAttention` (landmark attention with the optimized FA2-style kernel)
    """

    fast_compressive_landmark = "fast_compressive_landmark"
    """
    ➡️ :class:`FastCompressiveLandmarkAttention` (fast landmark attention that also folds each
    block's landmark token into the attention output -- a compressed summary of the block)
    """

    compressive_gqa_grouped = "compressive_gqa_grouped"
    """
    ➡️ :class:`CompressiveGQAGroupedAttention` (compressive landmark attention where each past block's
    cross-block gate is computed from the MEAN of the KV group's landmark scores for that block, so a
    GQA group's query heads share one block rescaling; within-block + local stay per-head)
    """

    multi_compressive_landmark = "multi_compressive_landmark"
    """
    ➡️ :class:`MultiCompressiveLandmarkAttention` (compressive landmark attention with multiple
    landmark tokens per block; each block's cross-block gate is a ``landmark_gate_pool`` -- ``"mean"``
    or ``"max"`` -- pool over its landmark scores, and every landmark folds its value into the block's
    within-block summary. ``block_size = mem_freq + num_landmarks`` is held to a power of two)
    """

    sparse_landmark = "sparse_landmark"
    """
    ➡️ :class:`SparseLandmarkAttention` (sparse landmark-only-across-chunks attention)
    """

    shared_vector_landmark = "shared_vector_landmark"
    """
    ➡️ :class:`SharedVectorLandmarkAttention` (fast landmark attention that appends a learned,
    per-block positional vector to every value before the attention aggregation)
    """

    document_landmark = "document_landmark"
    """
    ➡️ :class:`DocumentLandmarkAttention` (landmark version of document-chunked attention: normal
    landmark grouped softmax within a document, plus a landmark bridge across documents)
    """

    multi_landmark = "multi_landmark"
    """
    ➡️ :class:`MultiLandmarkAttention` (landmark grouped softmax with multiple landmark tokens per
    block, pooled into each block's gate via ``landmark_pool``)
    """

    document_multi_landmark = "document_multi_landmark"
    """
    ➡️ :class:`DocumentMultiLandmarkAttention` (document-chunked multi-landmark attention)
    """

    document_compressive_landmark = "document_compressive_landmark"
    """
    ➡️ :class:`DocumentCompressiveLandmarkAttention` (compressive version of document-chunked
    attention: the compressive landmark grouped softmax -- each past block's landmark token also
    contributes its value as a compressed summary -- plus the chunked-document mask)
    """

    document_chunked = "document_chunked"
    """
    ➡️ :class:`DocumentChunkedAttention` (dense full attention restricted by the chunked-document
    mask: context chunks isolated, FREE query/answer bridges across them -- the non-landmark dense
    analogue of ``document_landmark``)
    """

    pooled_doc_kv = "pooled_doc_kv"
    """
    ➡️ :class:`PooledDocKVAttention` (full causal attention where most context documents' per-token
    K/V are replaced -- for queries outside the document -- by a single mean-pooled KV slot with a
    ``+log(doc_len)`` logit bias; gold documents plus a random subset of negatives keep real KV.
    A train-time-only compression: inference is ordinary full attention)
    """

    dilated_sliding_window = "dilated_sliding_window"
    """
    ➡️ :class:`DilatedSlidingWindowAttention` (dilated causal sliding window whose dilation stride
    rotates with transformer depth -- "Hierarchical K": each layer attends only ``K`` keys spaced
    ``base**(layer_idx % num_configs)`` apart, so attention stays O(N*K) while the receptive field of a
    stack of layers grows geometrically)
    """

    summary_token = "summary_token"
    """
    ➡️ :class:`SummaryTokenAttention` (per-document **summary-token** masking: each context document
    is followed by a short run of summary tokens, and a document may read only itself plus the summary
    runs of earlier documents, while the trailing query reads the summary runs but no raw document
    content -- see :mod:`olmo_core.nn.attention.summary_mask` for the rule and its levers)
    """


@dataclass
class AttentionTypePatternConfig(Config):
    """
    Specifies the :class:`AttentionType` to use on a per-layer basis, mirroring the format of
    :class:`SlidingWindowAttentionConfig`. The ``pattern`` is repeated to cover all layers, so a
    single model can mix attention implementations (e.g. full attention with landmark variants).

    The shared attention parameters (``mem_freq``, ``num_landmarks``, etc.) are taken from the
    enclosing :class:`AttentionConfig`; this pattern only selects the *type* per layer.
    """

    pattern: List[AttentionType]
    """
    The pattern of attention types to use, repeated to cover all layers. For example, a pattern of
    ``[fast_landmark, fast_landmark, fast_landmark, sparse_landmark]`` means that for each set of 4
    layers, the first 3 use fast landmark attention and the last uses sparse landmark attention.
    """

    force_full_attention_on_first_layer: bool = False
    """
    If `True`, the first transformer layer will always use :data:`AttentionType.default` (full
    attention), regardless of the pattern.
    """

    force_full_attention_on_last_layer: bool = False
    """
    If `True`, the last transformer layer will always use :data:`AttentionType.default` (full
    attention), regardless of the pattern.
    """

    def get_type(self, layer_idx: int, n_layers: int) -> AttentionType:
        """
        Get the :class:`AttentionType` for a given layer.
        """
        if self.force_full_attention_on_first_layer and layer_idx == 0:
            return AttentionType.default
        if self.force_full_attention_on_last_layer and layer_idx == (n_layers - 1):
            return AttentionType.default

        # Adjust the layer index if the first layer is special-cased to full attention
        # (in which case the pattern is applied starting from the second layer).
        effective_layer_idx = layer_idx
        if self.force_full_attention_on_first_layer:
            effective_layer_idx -= 1

        return AttentionType(self.pattern[effective_layer_idx % len(self.pattern)])

    def landmark_types(self) -> Set[AttentionType]:
        """
        The set of landmark attention types that appear anywhere in the pattern.
        """
        return {t for t in self.pattern if t in _LANDMARK_ATTENTION_TYPES}


_LANDMARK_ATTENTION_TYPES = (
    AttentionType.landmark,
    AttentionType.fast_landmark,
    AttentionType.fast_compressive_landmark,
    AttentionType.compressive_gqa_grouped,
    AttentionType.multi_compressive_landmark,
    AttentionType.sparse_landmark,
    AttentionType.shared_vector_landmark,
    AttentionType.document_landmark,
    AttentionType.document_compressive_landmark,
    AttentionType.multi_landmark,
    AttentionType.document_multi_landmark,
)

# Landmark variants that accept the ``num_landmarks`` (and, for the multi-landmark ones,
# ``landmark_pool``) config fields.
_NUM_LANDMARKS_TYPES = (
    AttentionType.sparse_landmark,
    AttentionType.multi_landmark,
    AttentionType.document_multi_landmark,
    AttentionType.multi_compressive_landmark,
)
_MULTI_LANDMARK_TYPES = (
    AttentionType.multi_landmark,
    AttentionType.document_multi_landmark,
)


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
    layer_types: Optional[AttentionTypePatternConfig] = None
    """
    Optionally select the :class:`AttentionType` on a per-layer basis. When set, this overrides
    :data:`name` for each layer (which is then only used as a fallback). The shared landmark
    parameters below (``mem_freq``, ``num_landmarks``, etc.) apply to whichever layers resolve to a
    landmark variant.
    """
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
    mem_freq: Optional[int] = None
    """
    The number of regular tokens between landmark tokens, used only by
    :class:`LandmarkAttention` (``name="landmark"``). The landmark block size is ``mem_freq + 1``.
    """
    landmark_use_kernel: Optional[bool] = None
    """
    For :class:`LandmarkAttention` only: use the fused Triton kernel instead of the eager path.
    Defaults to ``False`` (eager). See :class:`LandmarkAttention` for the caveats.
    """
    num_landmarks: Optional[int] = None
    """
    Number of landmark tokens per block/chunk (the last ``num_landmarks`` tokens). Supported by
    :class:`SparseLandmarkAttention` (``name="sparse_landmark"``), :class:`MultiLandmarkAttention`
    (``name="multi_landmark"``), :class:`DocumentMultiLandmarkAttention`
    (``name="document_multi_landmark"``), and :class:`MultiCompressiveLandmarkAttention`
    (``name="multi_compressive_landmark"``). Defaults to 1.
    """
    landmark_pool: Optional[str] = None
    """
    For the multi-landmark variants (``name="multi_landmark"`` / ``"document_multi_landmark"``) only:
    how to pool a block's landmark probabilities into its gate -- ``"sum"`` (marginal mass, the
    default) or ``"max"`` (best-matching landmark). See :class:`MultiLandmarkAttention`. This pools
    *post-softmax probabilities*; the compressive variant uses ``landmark_gate_pool`` instead, which
    pools the pre-softmax *logits*.
    """
    landmark_gate_pool: Optional[str] = None
    """
    For :class:`MultiCompressiveLandmarkAttention` (``name="multi_compressive_landmark"``) only: how
    to pool a block's landmark *logits* into its cross-block gate -- ``"mean"`` (average affinity, the
    default) or ``"max"`` (best-matching landmark). See :class:`MultiCompressiveLandmarkAttention`.
    """
    nonselected_landmark_mass: Optional[float] = None
    """
    For :class:`FastCompressiveLandmarkAttention` (``name="fast_compressive_landmark"``) only: the
    fraction of attention mass reserved at top-k decode time for the landmark tokens of the
    non-selected blocks. Defaults to 0.1. See :class:`FastCompressiveLandmarkAttention`.
    """
    group_landmark_selection: Optional[str] = None
    """
    For :class:`FastCompressiveLandmarkAttention` (``name="fast_compressive_landmark"``) or
    :class:`DocumentCompressiveLandmarkAttention` (``name="document_compressive_landmark"``) only:
    under GQA, share top-k landmark-block selection across each KV group's query heads via ``"mean"``
    or ``"max"`` aggregation of their scores, instead of each head retrieving independently. Defaults
    to ``None`` (unchanged per-head selection). See :class:`FastCompressiveLandmarkAttention`.
    """
    gate_temperature: Optional[bool] = None
    """
    For :class:`FastCompressiveLandmarkAttention` (``name="fast_compressive_landmark"``) only: adds a
    per-layer learnable temperature on the cross-block gate softmax (initialized as a no-op).
    Defaults to ``None``/``False`` (no temperature parameter). See
    :class:`FastCompressiveLandmarkAttention`.
    """
    vec_dim: Optional[int] = None
    """
    For :class:`SharedVectorLandmarkAttention` (``name="shared_vector_landmark"``) only: the length
    of the learned per-block positional vector appended to every value. Defaults to 32. The per-head
    attention output becomes ``head_dim + vec_dim`` and ``w_out`` is widened to match.
    """
    cross_doc_mode: Optional[str] = None
    """
    For :class:`DocumentLandmarkAttention` (``name="document_landmark"``) only: which
    :class:`CrossDocumentMaskType` policy governs cross-document attention. Defaults to
    ``"document_chunked"`` (the landmark bridge across documents).
    """
    dilation_n: Optional[int] = None
    """
    For :class:`DocumentChunkedAttention` with ``cross_doc_mode="hierarchical_dilated"`` only: the
    number of documents a context query attends per layer (itself + ``n-1`` strided predecessors).
    Defaults to 2.
    """
    dilation_m: Optional[int] = None
    """
    For :class:`DocumentChunkedAttention` with ``cross_doc_mode="hierarchical_dilated"`` only: the
    dilation base ``m >= 1``. At cycle position ``p = ell % dilation_cycle`` the stride is ``m**p``
    (saturated within the cycle once it spans all history), so the receptive span is ``(n-1)*m**p``
    documents. Defaults to 2.
    """
    dilation_cycle: Optional[int] = None
    """
    For the document-chunked family with ``cross_doc_mode="hierarchical_dilated"`` only: the rotation
    period ``L`` -- the dilation stride rotates with depth as ``layer_idx % L`` (the "Hierarchical K"
    schedule), so deep layers revisit the fine-grained small-stride patterns. ``None`` (default) uses
    the pattern default of 3. (Note the pure-saturation schedule ``dilation_cycle=None`` at the
    :class:`~olmo_core.nn.attention.chunked_mask.AttentionPattern` level is deprecated.)
    """
    dilation_max_docs: Optional[int] = None
    """
    For :class:`DocumentChunkedAttention` with ``cross_doc_mode="hierarchical_dilated"`` only: an
    optional fixed document count for the saturation cap. ``None`` (default) computes the cap per
    sequence from the actual number of context chunks.
    """
    doc_keep_prob: Optional[float] = None
    """
    For :class:`DocumentChunkedAttention` with ``cross_doc_mode="random_doc"`` only: each context
    document attends to itself + a random subset of the strictly-earlier documents, each kept with this
    Bernoulli probability (~this fraction of previous docs on average). Defaults to 0.1. The keep set is
    a deterministic seeded hash of the ``(query_doc, key_doc)`` pair (fixed random sparsity graph over
    document indices; vary :attr:`random_doc_seed`).
    """
    random_doc_seed: Optional[int] = None

    random_doc_per_example: Optional[bool] = None
    """
    ``random_doc``: give EVERY EXAMPLE its own random sparsity graph (still deterministic
    per example) instead of one graph shared across all examples. The ablation for "does the
    model need a *stable* mask, or just sparse connectivity?".
    """
    """
    For :class:`DocumentChunkedAttention` with ``cross_doc_mode="random_doc"`` only: seed for the
    per-document-pair keep hash. Defaults to 42.
    """
    summary_every_k: Optional[int] = None
    """
    For :class:`DocumentChunkedAttention` with ``cross_doc_mode="summary_attention"`` only: the **cell
    size** -- how many context documents precede each summary span. The tokenized layout must match
    (``summary_every_k`` docs then one summary span per cell), so chunk indices run on a stride of
    ``summary_every_k + 1``. Defaults to 10.
    """
    summary_bandwidth: Optional[int] = None
    """
    For :class:`DocumentChunkedAttention` with ``cross_doc_mode="summary_attention"`` only: the **relay
    bandwidth** -- how many of each earlier summary span's leading tokens a later chunk may attend. The
    dose knob of the bandwidth ladder; the data is identical across rungs, only visibility changes.
    ``0`` (default) removes the relay entirely, reproducing a pure cell-blocks mask (the floor control).
    """
    summary_relay: Optional[bool] = None
    """
    For :class:`DocumentChunkedAttention` with ``cross_doc_mode="summary_attention"`` only: whether a
    summary span may read its own cell's documents. ``True`` (default) is the treatment. ``False`` is
    the **placebo** -- the span keeps its position, tokens, and every edge into it, but reads nothing,
    so it provably carries zero document content.
    """
    gold_decoys: Optional[int] = None
    """
    For :class:`DocumentChunkedAttention` with ``cross_doc_mode="gold_hop_controlled"`` only:
    distance-matched non-gold decoy pairs per gold pair, given the identical edit so the arm's
    structural signature stops naming the gold pair. Measured: ``0`` leaves ``hop_inf`` at graph-only
    precision@3 16.2% (66x chance); ``12`` cuts it to 2.0% (8x) and makes ``hop2`` / ``hop_inf``
    leak-matched, which is what the ``hop2 - hop_inf`` contrast requires.
    """
    gold_hops: Optional[int] = None
    """
    For :class:`DocumentChunkedAttention` with ``cross_doc_mode="gold_hop_controlled"`` only: which arm
    of the multi-hop gold-routing ladder -- ``1`` (gold edge forced present), ``2`` / ``3`` (gold edge
    **deleted**, shortest gold path forced to exactly that length), or ``-1``
    (:data:`~olmo_core.nn.attention.gold_hop_mask.GOLD_HOPS_INF`: gold edge deleted and every path cut,
    the leak-matched control). Recorded here so the checkpoint's ``config.json`` names the arm; the
    per-example graph is built at runtime by
    :func:`~olmo_core.nn.attention.gold_hop_mask.install_gold_hop_mask`, which refuses to install if the
    two disagree.
    """
    n_summary_tokens: Optional[int] = None
    """
    For :class:`SummaryTokenAttention` (``name="summary_token"``) only: how many summary tokens follow
    each context document. **Must match the tokenized data** -- the launcher cross-checks it against
    the shard metadata rather than trusting it.
    """
    summary_visible_tokens: Optional[int] = None
    """
    For :class:`SummaryTokenAttention` only: how many *leading* tokens of each summary run a later
    document may read. ``None`` (default) means all of them. Lowering it is the compression-dose
    ladder -- the tokenized data is unchanged across rungs, only visibility varies -- and ``0`` removes
    the relay entirely, which is the floor control and is reachable only by asking for it explicitly.
    """
    summaries_read_own_document: Optional[bool] = None
    """
    For :class:`SummaryTokenAttention` only: whether a document's summary run may read that document's
    content. ``True`` (default) is the treatment. ``False`` is the **placebo** -- the run keeps its
    position, its tokens and every edge into it, but reads nothing, so it provably carries zero
    document content.
    """
    summaries_read_earlier_summaries: Optional[bool] = None
    """
    For :class:`SummaryTokenAttention` only: whether a summary run may read earlier summary runs.
    ``True`` (default) gives a relay chain, so information can propagate transitively past its
    immediate neighbour.
    """
    query_reads_documents: Optional[bool] = None
    """
    For :class:`SummaryTokenAttention` only: whether the trailing query/answer is an unrestricted
    reader. ``False`` (default) is the treatment -- the query sees the instruction, its own span and
    the summary runs, but no raw document content. ``True`` restricts only document-to-document
    attention.
    """
    flex_block_size: Optional[int] = None
    """
    For :class:`DocumentChunkedAttention` only: the FlexAttention ``create_block_mask`` block size (the
    granularity at which fully-masked blocks are skipped). ``None`` (default) uses FlexAttention's
    default of 128. Shrinking it (e.g. 64 / 32) lets sub-128-token chunks realize their block-sparsity
    at the cost of higher kernel overhead; profile the crossover per chunk-size distribution.
    """
    full_attention_layers: Optional[List[int]] = None
    """
    For :class:`DocumentChunkedAttention` (``name="document_chunked"``) only: layer indices that use
    plain full (causal) attention instead of the chunked mask -- a hybrid where these ``N`` layers see
    the whole sequence while the rest stay document-chunked. Negative indices count from the end
    (``-1`` = last layer). ``None`` / empty (default) -> every layer uses the chunked mask.
    """
    pooled_keep_prob: Optional[float] = None
    """
    For :class:`PooledDocKVAttention` (``name="pooled_doc_kv"``) only: the fallback Bernoulli
    probability that a context document keeps its real per-token KV when no gold keep-set hook
    (:func:`~olmo_core.nn.attention.pooled_doc_kv.install_pooled_doc_keep`) is active. Defaults to
    0.1. The draw is a deterministic seeded hash of the example's chunk layout and the document
    index, so it is identical across layers and epochs.
    """
    pooled_keep_seed: Optional[int] = None
    """
    For :class:`PooledDocKVAttention` only: seed for the fallback keep draw. Defaults to 42.
    """
    pooled_len_bias: Optional[bool] = None
    """
    For :class:`PooledDocKVAttention` only: add ``+log(doc_len)`` to each pooled slot's attention
    logit, making the slot exactly equivalent to ``doc_len`` copies of the document's mean KV entry
    (the principled "sum of attention mass" form). Defaults to ``True``. ``False`` gives the slot
    the mass of a single average token.
    """
    dilated_window_k: Optional[int] = None
    """
    For :class:`DilatedSlidingWindowAttention` (``name="dilated_sliding_window"``) only: the number of
    keys ``K`` each query attends per layer (its own position + ``K-1`` strided predecessors). Defaults
    to 3.
    """
    dilated_window_num_configs: Optional[int] = None
    """
    For :class:`DilatedSlidingWindowAttention` only: the rotation cycle length ``L`` (number of
    distinct dilation configs before the rotation resets). Defaults to 3.
    """
    dilated_window_base: Optional[int] = None
    """
    For :class:`DilatedSlidingWindowAttention` only: the dilation base ``m``. At transformer layer
    ``ell`` the dilation stride is ``m ** (ell % L)``, so successive layers attend keys spaced
    ``1, m, m**2, ...`` apart. Defaults to 2.
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
        # Shared-vector landmark: the separate w_out_vec branch (n_heads * vec_dim -> d_model) plus
        # the per-head weight_landmark map and base vector.
        if self.name == AttentionType.shared_vector_landmark:
            vec_dim = self.vec_dim if self.vec_dim is not None else 32
            params += n_heads * vec_dim * d_model  # w_out_vec
            if bias:
                params += d_model  # w_out_vec bias
            params += n_heads * head_dim * vec_dim  # weight_landmark
            params += n_heads * vec_dim  # base

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

        # Resolve the effective attention type for this layer. When a per-layer pattern is
        # configured it overrides ``name``; otherwise ``name`` applies to every layer.
        layer_types_config: Optional[AttentionTypePatternConfig] = kwargs.pop("layer_types", None)
        if layer_types_config is not None:
            effective_name = layer_types_config.get_type(layer_idx, n_layers)
            # The full set of attention types this config may produce across all layers, used for
            # validating the shared landmark parameters below.
            possible_types: Set[AttentionType] = {
                AttentionType(t) for t in layer_types_config.pattern
            }
            if (
                layer_types_config.force_full_attention_on_first_layer
                or layer_types_config.force_full_attention_on_last_layer
            ):
                possible_types.add(AttentionType.default)
        else:
            effective_name = self.name
            possible_types = {self.name}

        is_landmark = effective_name in _LANDMARK_ATTENTION_TYPES

        sliding_window_config: Optional[SlidingWindowAttentionConfig] = kwargs.pop(
            "sliding_window", None
        )
        use_swa = sliding_window_config is not None and sliding_window_config.should_use_swa(
            layer_idx, n_layers
        )
        if use_swa and is_landmark:
            raise OLMoConfigurationError(
                f"layer {layer_idx} resolves to landmark attention ('{effective_name}') but sliding "
                "window attention is also configured for it; these are mutually exclusive"
            )
        if use_swa:
            assert sliding_window_config is not None
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

        mem_freq = kwargs.pop("mem_freq", None)
        landmark_use_kernel = kwargs.pop("landmark_use_kernel", None)
        num_landmarks = kwargs.pop("num_landmarks", None)
        landmark_pool = kwargs.pop("landmark_pool", None)
        landmark_gate_pool = kwargs.pop("landmark_gate_pool", None)
        nonselected_landmark_mass = kwargs.pop("nonselected_landmark_mass", None)
        group_landmark_selection = kwargs.pop("group_landmark_selection", None)
        gate_temperature = kwargs.pop("gate_temperature", None)
        vec_dim = kwargs.pop("vec_dim", None)
        cross_doc_mode = kwargs.pop("cross_doc_mode", None)
        dilation_n = kwargs.pop("dilation_n", None)
        dilation_m = kwargs.pop("dilation_m", None)
        dilation_cycle = kwargs.pop("dilation_cycle", None)
        dilation_max_docs = kwargs.pop("dilation_max_docs", None)
        doc_keep_prob = kwargs.pop("doc_keep_prob", None)
        random_doc_seed = kwargs.pop("random_doc_seed", None)
        random_doc_per_example = kwargs.pop("random_doc_per_example", None)
        summary_every_k = kwargs.pop("summary_every_k", None)
        summary_bandwidth = kwargs.pop("summary_bandwidth", None)
        summary_relay = kwargs.pop("summary_relay", None)
        n_summary_tokens = kwargs.pop("n_summary_tokens", None)
        summary_visible_tokens = kwargs.pop("summary_visible_tokens", None)
        summaries_read_own_document = kwargs.pop("summaries_read_own_document", None)
        summaries_read_earlier_summaries = kwargs.pop("summaries_read_earlier_summaries", None)
        query_reads_documents = kwargs.pop("query_reads_documents", None)
        gold_hops = kwargs.pop("gold_hops", None)
        gold_decoys = kwargs.pop("gold_decoys", None)
        flex_block_size = kwargs.pop("flex_block_size", None)
        pooled_keep_prob = kwargs.pop("pooled_keep_prob", None)
        pooled_keep_seed = kwargs.pop("pooled_keep_seed", None)
        pooled_len_bias = kwargs.pop("pooled_len_bias", None)
        dilated_window_k = kwargs.pop("dilated_window_k", None)
        dilated_window_num_configs = kwargs.pop("dilated_window_num_configs", None)
        dilated_window_base = kwargs.pop("dilated_window_base", None)
        full_attention_layers = kwargs.pop("full_attention_layers", None)
        if mem_freq is not None and not (possible_types & set(_LANDMARK_ATTENTION_TYPES)):
            raise OLMoConfigurationError(
                "'mem_freq' is only supported with landmark attention variants "
                f"(no landmark layer is configured; got name='{self.name}')"
            )
        if landmark_use_kernel is not None and not (
            possible_types
            & {
                AttentionType.landmark,
                AttentionType.document_landmark,
                AttentionType.shared_vector_landmark,
                AttentionType.compressive_gqa_grouped,
                AttentionType.document_compressive_landmark,
            }
        ):
            raise OLMoConfigurationError(
                "'landmark_use_kernel' is only supported with landmark, document_landmark, "
                "shared_vector_landmark, compressive_gqa_grouped, or document_compressive_landmark "
                f"attention (got name='{self.name}')"
            )
        if vec_dim is not None and AttentionType.shared_vector_landmark not in possible_types:
            raise OLMoConfigurationError(
                "'vec_dim' is only supported with shared_vector_landmark attention "
                f"(got name='{self.name}')"
            )
        if num_landmarks is not None and not (possible_types & set(_NUM_LANDMARKS_TYPES)):
            raise OLMoConfigurationError(
                "'num_landmarks' is only supported with sparse_landmark, multi_landmark, "
                "document_multi_landmark, or multi_compressive_landmark attention "
                f"(got name='{self.name}')"
            )
        if landmark_gate_pool is not None and not (
            possible_types & {AttentionType.multi_compressive_landmark}
        ):
            raise OLMoConfigurationError(
                "'landmark_gate_pool' is only supported with multi_compressive_landmark attention "
                f"(got name='{self.name}')"
            )
        if landmark_pool is not None and not (possible_types & set(_MULTI_LANDMARK_TYPES)):
            raise OLMoConfigurationError(
                "'landmark_pool' is only supported with multi_landmark or document_multi_landmark "
                f"attention (got name='{self.name}')"
            )
        if cross_doc_mode is not None and not (
            possible_types
            & {
                AttentionType.document_landmark,
                AttentionType.document_compressive_landmark,
                AttentionType.document_multi_landmark,
                AttentionType.document_chunked,
            }
        ):
            raise OLMoConfigurationError(
                "'cross_doc_mode' is only supported with document_landmark, "
                "document_compressive_landmark, document_multi_landmark, or document_chunked "
                f"attention (got name='{self.name}')"
            )
        # ``hierarchical_dilated`` is a cross-document visibility policy that is orthogonal to the
        # attention mechanism, so the dilation knobs apply to every document-chunked family (dense /
        # landmark / compressive).
        _DOC_CHUNKED_TYPES = {
            AttentionType.document_chunked,
            AttentionType.document_landmark,
            AttentionType.document_compressive_landmark,
        }
        if (
            dilation_n is not None
            or dilation_m is not None
            or dilation_cycle is not None
            or dilation_max_docs is not None
        ) and not (possible_types & _DOC_CHUNKED_TYPES):
            raise OLMoConfigurationError(
                "'dilation_n' / 'dilation_m' / 'dilation_cycle' / 'dilation_max_docs' are only "
                "supported with document_chunked, document_landmark, or "
                f"document_compressive_landmark attention (got name='{self.name}')"
            )
        _COMPRESSIVE_TYPES = {
            AttentionType.fast_compressive_landmark,
            AttentionType.document_compressive_landmark,
            AttentionType.compressive_gqa_grouped,
            AttentionType.multi_compressive_landmark,
        }
        if nonselected_landmark_mass is not None and not (possible_types & _COMPRESSIVE_TYPES):
            raise OLMoConfigurationError(
                "'nonselected_landmark_mass' is only supported with the compressive landmark variants "
                "(fast_compressive_landmark, document_compressive_landmark, compressive_gqa_grouped, "
                f"multi_compressive_landmark); got name='{self.name}'"
            )
        if group_landmark_selection is not None and not (possible_types & _COMPRESSIVE_TYPES):
            raise OLMoConfigurationError(
                "'group_landmark_selection' is only supported with the compressive landmark variants "
                "(fast_compressive_landmark, document_compressive_landmark, compressive_gqa_grouped, "
                f"multi_compressive_landmark); got name='{self.name}'"
            )
        if gate_temperature is not None and not (
            possible_types & {AttentionType.fast_compressive_landmark}
        ):
            raise OLMoConfigurationError(
                "'gate_temperature' is only supported with fast_compressive_landmark attention; "
                f"got name='{self.name}'"
            )
        if (
            dilated_window_k is not None
            or dilated_window_num_configs is not None
            or dilated_window_base is not None
        ) and AttentionType.dilated_sliding_window not in possible_types:
            raise OLMoConfigurationError(
                "'dilated_window_k' / 'dilated_window_num_configs' / 'dilated_window_base' are only "
                f"supported with dilated_sliding_window attention (got name='{self.name}')"
            )
        if full_attention_layers is not None and not (
            possible_types & {AttentionType.document_chunked, AttentionType.pooled_doc_kv}
        ):
            raise OLMoConfigurationError(
                "'full_attention_layers' (hybrid full/chunked layers) is only supported with "
                f"document_chunked or pooled_doc_kv attention (got name='{self.name}')"
            )
        if (
            pooled_keep_prob is not None
            or pooled_keep_seed is not None
            or pooled_len_bias is not None
        ) and AttentionType.pooled_doc_kv not in possible_types:
            raise OLMoConfigurationError(
                "'pooled_keep_prob' / 'pooled_keep_seed' / 'pooled_len_bias' are only supported "
                f"with pooled_doc_kv attention (got name='{self.name}')"
            )
        if (
            summary_every_k is not None
            or summary_bandwidth is not None
            or summary_relay is not None
        ) and not (possible_types & _DOC_CHUNKED_TYPES):
            raise OLMoConfigurationError(
                "'summary_every_k' / 'summary_bandwidth' / 'summary_relay' are only supported with "
                "document_chunked, document_landmark, or document_compressive_landmark attention "
                f"(got name='{self.name}')"
            )
        if (
            n_summary_tokens is not None
            or summary_visible_tokens is not None
            or summaries_read_own_document is not None
            or summaries_read_earlier_summaries is not None
            or query_reads_documents is not None
        ) and AttentionType.summary_token not in possible_types:
            raise OLMoConfigurationError(
                "'n_summary_tokens' / 'summary_visible_tokens' / 'summaries_read_own_document' / "
                "'summaries_read_earlier_summaries' / 'query_reads_documents' are only supported "
                f"with summary_token attention (got name='{self.name}'). Note these are NOT the "
                "'summary_every_k' / 'summary_bandwidth' / 'summary_relay' knobs of the "
                "document_chunked 'summary_attention' pattern, which is a different mask."
            )
        if (
            gold_hops is not None or gold_decoys is not None
        ) and AttentionType.document_chunked not in possible_types:
            raise OLMoConfigurationError(
                "'gold_hops' / 'gold_decoys' (the multi-hop gold-routing ladder) are only supported "
                "with "
                f"document_chunked attention (got name='{self.name}')"
            )

        # Attention sinks are only wired up for the default attention; drop the flag otherwise so
        # the other implementations don't see an unexpected keyword argument.
        if not kwargs.get("attention_sinks", False):
            kwargs.pop("attention_sinks", None)
        elif effective_name != AttentionType.default:
            raise OLMoConfigurationError("attention_sinks are only supported by default attention")

        # The MXFP8 packed-projection options are only wired up for fused_v2 attention; route them
        # there and reject them for any other implementation. A disabled (falsy) flag is a no-op, so
        # only reject when one is actually enabled.
        fused_v2_kwargs = {
            key: kwargs.pop(key)
            for key in ("mxfp8_projections", "mxfp8_qkv_projection", "mxfp8_out_projection")
            if key in kwargs
        }
        if any(fused_v2_kwargs.values()) and effective_name != AttentionType.fused_v2:
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
        if any(shared_forward_kwargs.values()) and effective_name not in (
            AttentionType.default,
            AttentionType.fused_v2,
        ):
            enabled = sorted(key for key, value in shared_forward_kwargs.items() if value)
            raise OLMoConfigurationError(
                f"{enabled} are only supported by default and fused_v2 attention"
            )

        if effective_name != AttentionType.default:
            for key in ("scalable_softmax", "qk_norm_per_head_gains"):
                if kwargs.pop(key, False):
                    raise OLMoConfigurationError(
                        f"'{key}' is not supported with {effective_name} attention"
                    )

        try:
            if effective_name == "default":
                return Attention(**kwargs, **shared_forward_kwargs)
            elif effective_name == "fused_v2":
                return FusedAttentionV2(**kwargs, **fused_v2_kwargs, **shared_forward_kwargs)
            elif effective_name == "normalized":
                if "window_size" in kwargs:
                    raise OLMoConfigurationError(
                        "'window_size' is not supported with normalized attention"
                    )
                return NormalizedAttention(**kwargs)
            elif effective_name == "landmark":
                if mem_freq is None:
                    raise OLMoConfigurationError("landmark attention requires 'mem_freq' to be set")
                if landmark_use_kernel is not None:
                    kwargs["use_kernel"] = landmark_use_kernel
                return LandmarkAttention(mem_freq=mem_freq, **kwargs)
            elif effective_name == "fast_landmark":
                if mem_freq is None:
                    raise OLMoConfigurationError(
                        "fast_landmark attention requires 'mem_freq' to be set"
                    )
                return FastLandmarkAttention(mem_freq=mem_freq, **kwargs)
            elif effective_name == "fast_compressive_landmark":
                if mem_freq is None:
                    raise OLMoConfigurationError(
                        "fast_compressive_landmark attention requires 'mem_freq' to be set"
                    )
                if nonselected_landmark_mass is not None:
                    kwargs["nonselected_landmark_mass"] = nonselected_landmark_mass
                if group_landmark_selection is not None:
                    kwargs["group_landmark_selection"] = group_landmark_selection
                if gate_temperature is not None:
                    kwargs["gate_temperature"] = gate_temperature
                return FastCompressiveLandmarkAttention(mem_freq=mem_freq, **kwargs)
            elif effective_name == "compressive_gqa_grouped":
                if mem_freq is None:
                    raise OLMoConfigurationError(
                        "compressive_gqa_grouped attention requires 'mem_freq' to be set"
                    )
                if nonselected_landmark_mass is not None:
                    kwargs["nonselected_landmark_mass"] = nonselected_landmark_mass
                if group_landmark_selection is not None:
                    kwargs["group_landmark_selection"] = group_landmark_selection
                if landmark_use_kernel is not None:
                    kwargs["use_kernel"] = landmark_use_kernel
                return CompressiveGQAGroupedAttention(mem_freq=mem_freq, **kwargs)
            elif effective_name == "multi_compressive_landmark":
                if mem_freq is None:
                    raise OLMoConfigurationError(
                        "multi_compressive_landmark attention requires 'mem_freq' to be set"
                    )
                if num_landmarks is not None:
                    kwargs["num_landmarks"] = num_landmarks
                if landmark_gate_pool is not None:
                    kwargs["landmark_gate_pool"] = landmark_gate_pool
                if nonselected_landmark_mass is not None:
                    kwargs["nonselected_landmark_mass"] = nonselected_landmark_mass
                if group_landmark_selection is not None:
                    kwargs["group_landmark_selection"] = group_landmark_selection
                return MultiCompressiveLandmarkAttention(mem_freq=mem_freq, **kwargs)
            elif effective_name == "sparse_landmark":
                if mem_freq is None:
                    raise OLMoConfigurationError(
                        "sparse_landmark attention requires 'mem_freq' to be set"
                    )
                if num_landmarks is not None:
                    kwargs["num_landmarks"] = num_landmarks
                return SparseLandmarkAttention(mem_freq=mem_freq, **kwargs)
            elif effective_name == "shared_vector_landmark":
                if mem_freq is None:
                    raise OLMoConfigurationError(
                        "shared_vector_landmark attention requires 'mem_freq' to be set"
                    )
                if vec_dim is not None:
                    kwargs["vec_dim"] = vec_dim
                if landmark_use_kernel is not None:
                    kwargs["use_kernel"] = landmark_use_kernel
                return SharedVectorLandmarkAttention(mem_freq=mem_freq, **kwargs)
            elif effective_name == "document_landmark":
                if mem_freq is None:
                    raise OLMoConfigurationError(
                        "document_landmark attention requires 'mem_freq' to be set"
                    )
                if cross_doc_mode is not None:
                    kwargs["cross_doc_mode"] = cross_doc_mode
                if landmark_use_kernel is not None:
                    kwargs["use_kernel"] = landmark_use_kernel
                # Per-layer dilation stride for the "hierarchical_dilated" cross_doc_mode.
                if dilation_n is not None:
                    kwargs["dilation_n"] = dilation_n
                if dilation_m is not None:
                    kwargs["dilation_m"] = dilation_m
                if dilation_cycle is not None:
                    kwargs["dilation_cycle"] = dilation_cycle
                if dilation_max_docs is not None:
                    kwargs["dilation_max_docs"] = dilation_max_docs
                kwargs["layer_idx"] = layer_idx
                kwargs["n_layers"] = n_layers
                return DocumentLandmarkAttention(mem_freq=mem_freq, **kwargs)
            elif effective_name == "document_compressive_landmark":
                if mem_freq is None:
                    raise OLMoConfigurationError(
                        "document_compressive_landmark attention requires 'mem_freq' to be set"
                    )
                if cross_doc_mode is not None:
                    kwargs["cross_doc_mode"] = cross_doc_mode
                if nonselected_landmark_mass is not None:
                    kwargs["nonselected_landmark_mass"] = nonselected_landmark_mass
                if group_landmark_selection is not None:
                    kwargs["group_landmark_selection"] = group_landmark_selection
                if landmark_use_kernel is not None:
                    kwargs["use_kernel"] = landmark_use_kernel
                # Per-layer dilation stride for the "hierarchical_dilated" cross_doc_mode.
                if dilation_n is not None:
                    kwargs["dilation_n"] = dilation_n
                if dilation_m is not None:
                    kwargs["dilation_m"] = dilation_m
                if dilation_cycle is not None:
                    kwargs["dilation_cycle"] = dilation_cycle
                if dilation_max_docs is not None:
                    kwargs["dilation_max_docs"] = dilation_max_docs
                kwargs["layer_idx"] = layer_idx
                kwargs["n_layers"] = n_layers
                return DocumentCompressiveLandmarkAttention(mem_freq=mem_freq, **kwargs)
            elif effective_name == "multi_landmark":
                if mem_freq is None:
                    raise OLMoConfigurationError(
                        "multi_landmark attention requires 'mem_freq' to be set"
                    )
                if num_landmarks is not None:
                    kwargs["num_landmarks"] = num_landmarks
                if landmark_pool is not None:
                    kwargs["landmark_pool"] = landmark_pool
                return MultiLandmarkAttention(mem_freq=mem_freq, **kwargs)
            elif effective_name == "document_multi_landmark":
                if mem_freq is None:
                    raise OLMoConfigurationError(
                        "document_multi_landmark attention requires 'mem_freq' to be set"
                    )
                if num_landmarks is not None:
                    kwargs["num_landmarks"] = num_landmarks
                if landmark_pool is not None:
                    kwargs["landmark_pool"] = landmark_pool
                if cross_doc_mode is not None:
                    kwargs["cross_doc_mode"] = cross_doc_mode
                return DocumentMultiLandmarkAttention(mem_freq=mem_freq, **kwargs)
            elif effective_name == "document_chunked":
                # Dense (non-landmark) chunked attention -- no mem_freq.
                if cross_doc_mode is not None:
                    kwargs["cross_doc_mode"] = cross_doc_mode
                if dilation_n is not None:
                    kwargs["dilation_n"] = dilation_n
                if dilation_m is not None:
                    kwargs["dilation_m"] = dilation_m
                if dilation_cycle is not None:
                    kwargs["dilation_cycle"] = dilation_cycle
                if dilation_max_docs is not None:
                    kwargs["dilation_max_docs"] = dilation_max_docs
                if doc_keep_prob is not None:
                    kwargs["doc_keep_prob"] = doc_keep_prob
                if random_doc_seed is not None:
                    kwargs["random_seed"] = random_doc_seed
                if random_doc_per_example is not None:
                    kwargs["random_doc_per_example"] = random_doc_per_example
                # Cell size / relay bandwidth / relay on-off for the "summary_attention" pattern.
                if summary_every_k is not None:
                    kwargs["summary_every_k"] = summary_every_k
                if summary_bandwidth is not None:
                    kwargs["summary_bandwidth"] = summary_bandwidth
                if summary_relay is not None:
                    kwargs["summary_relay"] = summary_relay
                # The arm of the "gold_hop_controlled" ladder (recorded; the graph arrives via the hook).
                if gold_hops is not None:
                    kwargs["gold_hops"] = gold_hops
                if gold_decoys is not None:
                    kwargs["gold_decoys"] = gold_decoys
                if flex_block_size is not None:
                    kwargs["flex_block_size"] = flex_block_size
                if full_attention_layers is not None:
                    kwargs["full_attention_layers"] = full_attention_layers
                # The hierarchical-dilated pattern picks its stride from the layer index.
                kwargs["layer_idx"] = layer_idx
                kwargs["n_layers"] = n_layers
                return DocumentChunkedAttention(**kwargs)
            elif effective_name == "pooled_doc_kv":
                # Train-time per-document KV pooling; the non-pooled topology is plain causal.
                if pooled_keep_prob is not None:
                    kwargs["keep_prob"] = pooled_keep_prob
                if pooled_keep_seed is not None:
                    kwargs["keep_seed"] = pooled_keep_seed
                if pooled_len_bias is not None:
                    kwargs["len_bias"] = pooled_len_bias
                if flex_block_size is not None:
                    kwargs["flex_block_size"] = flex_block_size
                if full_attention_layers is not None:
                    kwargs["full_attention_layers"] = full_attention_layers
                kwargs["layer_idx"] = layer_idx
                kwargs["n_layers"] = n_layers
                return PooledDocKVAttention(**kwargs)
            elif effective_name == "dilated_sliding_window":
                # Dilated causal sliding window with a per-layer rotating dilation stride.
                if dilated_window_k is not None:
                    kwargs["window"] = dilated_window_k
                if dilated_window_num_configs is not None:
                    kwargs["num_configs"] = dilated_window_num_configs
                if dilated_window_base is not None:
                    kwargs["base"] = dilated_window_base
                if flex_block_size is not None:
                    kwargs["flex_block_size"] = flex_block_size
                # The dilation stride is picked from the layer index.
                kwargs["layer_idx"] = layer_idx
                kwargs["n_layers"] = n_layers
                return DilatedSlidingWindowAttention(**kwargs)
            elif effective_name == "summary_token":
                # Per-document summary-token masking. The levers are named for what they control;
                # every default is the treatment, so no arm is silently the floor control.
                if n_summary_tokens is not None:
                    kwargs["n_summary_tokens"] = n_summary_tokens
                if summary_visible_tokens is not None:
                    kwargs["summary_visible_tokens"] = summary_visible_tokens
                if summaries_read_own_document is not None:
                    kwargs["summaries_read_own_document"] = summaries_read_own_document
                if summaries_read_earlier_summaries is not None:
                    kwargs["summaries_read_earlier_summaries"] = summaries_read_earlier_summaries
                if query_reads_documents is not None:
                    kwargs["query_reads_documents"] = query_reads_documents
                if flex_block_size is not None:
                    kwargs["flex_block_size"] = flex_block_size
                return SummaryTokenAttention(**kwargs)
            else:
                raise NotImplementedError(effective_name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{effective_name}' {self.__class__.__name__}, {e}"
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
    ) -> torch.Tensor:
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.record_leftpad(cache_leftpad)
        # shape: (batch_size, seq_len, n_heads, head_dim)
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
        if cu_doc_lens is not None or position_ids is not None:
            if not isinstance(self.rope, RotaryEmbedding):
                raise NotImplementedError(
                    "Intra-document RoPE (cu_doc_lens / position_ids) is only supported by "
                    f"RotaryEmbedding; got {type(self.rope).__name__}"
                )
            if cu_doc_lens is not None:
                rope_kwargs["cu_doc_lens"] = cu_doc_lens
            if position_ids is not None:
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
        Compute the query, key, and value tensors from the input, applying QKV clipping,
        QK-norm, and RoPE. This is the shared pre-attention logic used by :meth:`forward`
        (and subclasses such as :class:`LandmarkAttention`).

        :returns: ``(q, k, v)`` with shapes ``(batch_size, seq_len, n_heads (local), head_dim)``,
            ``(batch_size, seq_len, n_kv_heads (local), head_dim)``, and
            ``(batch_size, seq_len, n_kv_heads (local), head_dim)`` respectively.
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

            # Under Ulysses CP with sequence packing, RoPE is applied on this rank's contiguous
            # shard (length ``T``) *before* the all-to-all gathers the full sequence, while
            # ``cu_doc_lens`` still describes the full (un-sharded) sequence. Feeding that full
            # ``cu_doc_lens`` to RoPE's per-document branch indexes a shard-sized position table with
            # full-sequence offsets, which overflows (e.g. an out-of-bounds CUDA gather). Convert to
            # explicit per-document positions for the local shard instead -- correct even for
            # documents that straddle a rank boundary -- mirroring :class:`LandmarkAttention`. The
            # full ``cu_doc_lens`` is still used for masking inside the backend, after the gather.
            rope_cu_doc_lens = cu_doc_lens
            if (
                cu_doc_lens is not None
                and position_ids is None
                and self.cp_enabled
                and self.backend.uly is not None
            ):
                cp_pg = self.backend.cp_pg
                assert cp_pg is not None
                position_ids = build_local_packed_position_ids(
                    cu_doc_lens, B, T, get_rank(cp_pg), get_world_size(cp_pg)
                )
            # Explicit ``position_ids`` always win for RoPE and the two are mutually exclusive in
            # RotaryEmbedding, so drop ``cu_doc_lens`` from the RoPE call whenever positions were
            # given -- by the CP branch above, or by a caller that already knows each token's
            # intra-document position. The compacted block-skip path
            # (:func:`olmo_core.nn.block_skip._compact_kwargs`) is the second kind: it gathers the
            # kept tokens, so their positions are no longer implied by the (re-derived)
            # ``cu_doc_lens``, which still describes the compacted document boundaries the BACKEND
            # masks with.
            if position_ids is not None:
                rope_cu_doc_lens = None

            # Explicit per-token ``position_ids`` (e.g. ragged right-padded landmark decode, where each
            # row's single token sits at a different absolute position) override the scalar KV-cache
            # ``start_pos``; the two are mutually exclusive in RoPE.
            # ``start_pos`` may be passed in by :meth:`forward` (upstream's recompute path); when it
            # is not, fall back to the KV cache position, as before.
            if position_ids is not None:
                start_pos = None
            elif start_pos is None and self.kv_cache_manager is not None:
                start_pos = self.kv_cache_manager.current_position()
            q, k = self._apply_rope(
                q, k, start_pos, pos_sin, pos_cos, freqs_cis, rope_cu_doc_lens, position_ids
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
        chunk_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        aux_capture: Optional[dict] = None,
        kv_grad_mask: Optional[torch.Tensor] = None,
        soft_kv_override: Optional[dict] = None,
        block_keep: Optional[torch.Tensor] = None,
        kv_route_p: Optional[torch.Tensor] = None,
        kv_route_keep: Optional[torch.Tensor] = None,
        soft_keep: Optional[torch.Tensor] = None,
        summary_roles: Optional[torch.Tensor] = None,
        causal_example: Optional[torch.Tensor] = None,
        token_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply attention to the input.

        :param x: The input of shape ``(batch_size, seq_len, d_model)``.
        :param position_ids: Optional explicit per-token positions ``(batch_size, seq_len)`` for
            RoPE (e.g. soft-token pooled sequences where kept tokens keep their original
            positions). ``None`` (default) uses contiguous positions. On a sliding-window layer,
            non-contiguous positions also move the window to position space (key visible iff
            ``0 <= pos_q - pos_k < window_size``), see :mod:`.position_window`.
        :param attn_bias: Optional ``(B, 1, T, T)`` additive attention bias replacing the causal
            mask (soft-token aux matching: position-causality + shadow-column blocking). Runs a
            direct masked SDPA instead of the configured backend.
        :param aux_capture: Optional shared dict; when given, post-RoPE q/k/v slices at the
            requested indices are appended to ``aux_capture["layers"]`` for the soft-token
            attention-contribution matching loss.
        :param kv_grad_mask: Optional ``(B, T)`` bool; ``False`` columns have their K/V DETACHED
            (static-KV semantics for pooled soft tokens: forward identical, no LM gradient
            through those columns).
        :param block_keep: Optional ``(B, T)`` bool from :mod:`olmo_core.nn.block_skip`: ``False``
            tokens skip this block and must not be keys here (ANDed into the KV router's keep).
        :param soft_keep: Optional ``(B, T)`` float keep probability in ``[0, 1]`` (differentiable
            token removal): every key ``j`` gets the additive logit bias ``log(p_j + 1e-8)`` on top
            of the causal mask, for all queries, through the masked-SDPA path -- exact removal of
            key ``j`` at ``p_j = 0`` (up to the 1e-8 floor) and a no-op at ``p_j = 1``; gradients
            flow to ``p``. ``None`` (default) leaves the layer untouched.
        :param soft_kv_override: Optional dict replacing this layer's K/V at pooled soft-token
            columns with precomputed oracle slots (see :mod:`olmo_core.nn.oracle_slot`). Keys:
            ``rows``/``cols`` ``(S,)`` batch/column indices, ``pos`` ``(S,)`` absolute doc-center
            positions, ``k`` ``(S, n_kv_heads, head_dim)`` center-frame keys (rotated to ``pos``
            here with this layer's own RoPE), and ``v`` ``(S, n_kv_heads, head_dim)``. The
            override values are constants, so these columns are static KV regardless of
            ``kv_grad_mask``.
        :param cu_doc_lens: Cumulative document lengths in the input ``x``, a 1D
            :class:`torch.int32` tensor that should always have one more element than there
            are documents (the first element in the tensor should always be ``0``).
            Required together with ``max_doc_len`` when using intra-document masking.
        :param max_doc_len: The maximum document length in the input ``x``.
            Required together with ``cu_doc_lens`` when using intra-document masking.
        :param chunk_ids: **Accepted and ignored.** When a model enables document-chunked
            attention, :class:`~olmo_core.nn.transformer.Transformer` threads ``chunk_ids`` to
            *every* block, so a model that mixes chunked and non-chunked sequence mixers requires
            the non-chunked ones to tolerate it -- exactly as
            :class:`~olmo_core.nn.attention.recurrent.GatedDeltaNet` already does via ``**kwargs``
            for the Qwen3.5 hybrid. Plain attention has no chunked mask, so the roles carry no
            meaning here. This is what lets Olmo 3's sliding-window layers stay untouched while its
            full-attention layers carry the document-chunk mask.
        :param summary_roles: **Accepted and ignored**, for the same reason as ``chunk_ids``: when a
            model enables summary-token attention the per-token roles are threaded to every block, and
            on a hybrid such as Qwen3.5 only a quarter of the layers are attention at all. See
            :class:`~olmo_core.nn.attention.summary_token.SummaryTokenAttention`, which consumes them.
        :param causal_example: **Accepted and ignored**; the per-example mask-mixture arm, threaded
            alongside ``summary_roles``.
        :param token_weight: Optional ``(B, T)`` float ``w >= 1`` (weighted token sampling): key ``j`` stands for
            ``w_j`` identical copies of itself, i.e. the additive logit bias ``log w_j`` for every query, on top of
            the causal mask (exact: ``w`` copies of a key multiply its softmax mass by ``w``). Runs the masked-SDPA
            path (flash-attn has no additive bias). All-ones weights leave the layer on its normal path.

        :returns: The output of attention with shape ``(batch_size, seq_len, d_model)``.
        """
        # Not applicable to plain attention; see the parameter docs above.
        del chunk_ids, summary_roles, causal_example
        B, T, _ = x.shape

        # shape: (batch_size, seq_len, n_heads (local), head_dim),
        #        (batch_size, seq_len, n_kv_heads (local), head_dim),
        #        (batch_size, seq_len, n_kv_heads (local), head_dim)
        # ``position_ids`` is only forwarded when given, so that subclasses overriding
        # ``_prepare_qkv`` with upstream's signature (e.g. :class:`FusedAttentionV2`) keep working.
        prepare_kwargs: Dict[str, Any] = dict(
            pos_sin=pos_sin,
            pos_cos=pos_cos,
            freqs_cis=freqs_cis,
            cu_doc_lens=cu_doc_lens,
            start_pos=(
                None
                if position_ids is not None
                else (self.kv_cache_manager.current_position() if self.kv_cache_manager else None)
            ),
        )
        if position_ids is not None:
            prepare_kwargs["position_ids"] = position_ids

        # Optionally recompute Q/K/V in backward (trading compute for activation memory) by wrapping
        # the projection in an OutputDiscardCheckpoint.
        qkv_checkpoint: Optional[OutputDiscardCheckpoint] = None
        if torch.is_grad_enabled() and x.requires_grad and self.use_recompute_qkv_prep:
            qkv_checkpoint = OutputDiscardCheckpoint()
            q, k, v = cast(
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                qkv_checkpoint.checkpoint(self._prepare_qkv, x, **prepare_kwargs),
            )
        else:
            q, k, v = self._prepare_qkv(x, **prepare_kwargs)

        if soft_kv_override is not None and soft_kv_override["rows"].numel() > 0:
            # Oracle slots: rotate the cached center-frame keys to each slot's doc-center position
            # with this layer's own RoPE (buffers already warmed by the main _prepare_qkv call),
            # then overwrite the slot columns' K/V. Values are position-free.
            assert self.rope is not None and isinstance(self.rope, RotaryEmbedding)
            ovr_k = soft_kv_override["k"].to(k.dtype)
            _, ovr_k = self.rope(
                ovr_k[None],
                ovr_k[None],
                head_first=False,
                position_ids=soft_kv_override["pos"][None],
            )
            k = torch.index_put(
                k, (soft_kv_override["rows"], soft_kv_override["cols"]), ovr_k[0].to(k.dtype)
            )
            v = torch.index_put(
                v,
                (soft_kv_override["rows"], soft_kv_override["cols"]),
                soft_kv_override["v"].to(v.dtype),
            )
            if soft_kv_override.get("bias") is not None:
                # Per-slot scalar logit bias (the constant part of the doc's log-mass, which a
                # bias-free linear slot cannot carry). Requires the additive-bias SDPA path.
                if attn_bias is None:
                    raise RuntimeError(
                        "soft_kv_override with slot biases requires attn_bias (the "
                        "position-causal masked-SDPA path)"
                    )
                col_bias = torch.zeros((B, 1, 1, T), dtype=attn_bias.dtype, device=attn_bias.device)
                col_bias[soft_kv_override["rows"], 0, 0, soft_kv_override["cols"]] = (
                    soft_kv_override["bias"].to(attn_bias.dtype)
                )
                attn_bias = attn_bias + col_bias

        if kv_grad_mask is not None:
            from .gold_grad_mask import detach_kv

            k, v = detach_kv(k, v, kv_grad_mask)

        if aux_capture is not None:
            aux_capture["layers"].append(
                (
                    q[aux_capture["rows_q"], aux_capture["cols_q"]],
                    k[aux_capture["rows_kv"], aux_capture["cols_kv"]],
                    v[aux_capture["rows_kv"], aux_capture["cols_kv"]],
                    k[aux_capture["rows_sh"], aux_capture["cols_sh"]],
                    v[aux_capture["rows_sh"], aux_capture["cols_sh"]],
                )
            )

        if token_weight is not None and getattr(self, "_token_weight_on", True):
            tw = token_weight.to(device=q.device, dtype=torch.float64 if q.dtype == torch.float64 else torch.float32)
            if bool((tw != 1.0).any()):
                lw = torch.log(tw)[:, None, None, :]  # (B, 1, 1, T)
                if attn_bias is None:
                    causal = torch.full((T, T), float("-inf"), device=q.device, dtype=tw.dtype).triu(1)
                    attn_bias = causal[None, None] + lw
                else:
                    attn_bias = attn_bias + lw.to(attn_bias.dtype)

        if soft_keep is not None:
            sk_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            sk = torch.log(soft_keep.to(device=q.device, dtype=sk_dtype) + 1e-8)  # (B, T)
            causal = torch.full((T, T), float("-inf"), device=q.device, dtype=sk_dtype).triu(1)
            sk_bias = causal[None, None] + sk[:, None, None, :]  # (B, 1, T, T)
            attn_bias = sk_bias if attn_bias is None else attn_bias + sk_bias

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

        with qkv_save_context:
            kv_route = getattr(self, "_kv_route", None)
            if (
                (kv_route is not None and kv_route["holder"].enabled)
                or block_keep is not None
                or kv_route_p is not None
            ) and attn_bias is None:
                # Learned KV-cache allocation: keep/drop each key at this layer (kv_route.py). Also the
                # path for block skipping's key mask (``block_keep``: tokens skipping this block are
                # not keys here), with or without a KV router on the layer.
                from .kv_route import kv_route_attention

                att = kv_route_attention(
                    self,
                    x,
                    q,
                    k,
                    v,
                    cu_doc_lens=cu_doc_lens,
                    max_doc_len=max_doc_len,
                    cache_leftpad=cache_leftpad,
                    block_keep=block_keep,
                    kv_route_p=kv_route_p,
                    kv_route_keep=kv_route_keep,
                )
            elif attn_bias is not None:
                # Soft-token aux path: position-causal + shadow-blocked masked SDPA.
                from ..pooled_soft_token import masked_sdpa

                if self.window_size is not None:
                    # The additive bias replaces the backend, window included: fold the sliding
                    # window back in, measured in (original) positions.
                    from .position_window import apply_position_window_to_bias

                    attn_bias = apply_position_window_to_bias(
                        attn_bias, position_ids, self.window_size
                    )
                att = masked_sdpa(q, k, v, attn_bias, self.head_dim**-0.5)
            elif self._use_position_window(position_ids, cu_doc_lens, cu_doc_lens_q, local_k_slice):
                # Compacted row (dropped tokens, survivors keep their original positions): the
                # backends window by INDEX, which after dropping reaches ~W / keep original tokens
                # back. Window by position instead (position_window.py). Contiguous positions never
                # get here, so dense training keeps the backend path untouched.
                from .position_window import position_window_attention

                assert position_ids is not None and self.window_size is not None
                att = position_window_attention(
                    q,
                    k,
                    v,
                    position_ids,
                    self.window_size,
                    cu_doc_lens=cu_doc_lens,
                    scale=self.backend.scale,
                )
            else:
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
                )

        self._mxfp8_saved_qkv_for_backward_last_pack_count = qkv_save_counter[0]
        if qkv_checkpoint is not None:
            # Recompute Q/K/V before attention backward consumes the discarded activations.
            qkv_checkpoint.discard_output_and_register_recompute(att)

        # shape: (batch_size, seq_len, n_heads * head_dim)
        att = att.view(B, T, -1)

        # shape: (batch_size, seq_len, n_heads * head_dim)
        att = self._apply_gate(att, x)

        # shape: (batch_size, seq_len, d_model)
        return self.w_out(att)

    def _use_position_window(
        self,
        position_ids: Optional[torch.Tensor],
        cu_doc_lens: Optional[torch.Tensor],
        cu_doc_lens_q: Optional[torch.Tensor],
        local_k_slice: Optional[slice],
    ) -> bool:
        """
        Whether this forward must window by ORIGINAL position (:mod:`.position_window`): a sliding
        window layer given non-contiguous ``position_ids`` (a compacted row), outside KV-cache
        decode and context parallelism. Full-attention layers and dense rows return ``False``
        without touching ``position_ids``.
        """
        if self.window_size is None or position_ids is None:
            return False
        if (
            self.kv_cache_manager is not None
            or cu_doc_lens_q is not None
            or local_k_slice is not None
            or self.cp_enabled
        ):
            return False
        from .position_window import needs_position_window

        if not needs_position_window(position_ids, cu_doc_lens):
            return False
        if self.sinks is not None:
            raise NotImplementedError("attention sinks are not supported with position windows")
        if self.training and self.backend.dropout_p > 0.0:
            raise NotImplementedError("attention dropout is not supported with position windows")
        return True

    def _apply_gate(self, att: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the optional output gate (``att * sigmoid(w_g(x))``) to the attention output, just
        before the output projection. This is the shared gating logic used by :meth:`forward` and by
        the landmark attention variants, all of which compute their attention output and then call
        :attr:`w_out`. When no gate is configured this is a no-op (returns ``att`` unchanged), so
        non-gated models are byte-for-byte unaffected.

        :param att: The attention output of shape ``(batch_size, seq_len, n_heads * head_dim)``
            (n_heads being the *local* head count under tensor parallelism).
        :param x: The block input of shape ``(batch_size, seq_len, d_model)``, used to compute the
            gate.

        :returns: The gated attention output, same shape as ``att``.
        """
        if self.gate is None:
            return att
        assert self.w_g is not None
        B, T, _ = x.shape
        g = self.w_g(x)
        if self.gate.full_precision:
            g = g.float()
        gate_values = torch.sigmoid(g).to(att.dtype)
        if self.gate.granularity == GateGranularity.headwise:
            # head-wise gating: one value per (local) head, broadcast across head_dim. Use -1 for the
            # head dim so this stays correct when heads are sharded under tensor parallelism.
            att = (att.view(B, T, -1, self.head_dim) * gate_values.unsqueeze(-1)).view(B, T, -1)
        else:  # elementwise: one value per output element.
            att = att * gate_values
        return att

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

        :raises OLMoConfigurationError: If the Ulysses CP degree is incompatible with the head
            counts of this module.
        """
        if self.scalable_softmax:
            raise OLMoConfigurationError(
                "Scalable-Softmax is not supported with context parallelism"
            )
        if uly is not None:
            # Checked here rather than left to the collective, which would otherwise fail deep
            # inside a '.view()' with an unhelpful shape error. n_kv_heads need not divide the CP
            # degree: all_to_all_qkv_cp2hp replicates the KV heads when it doesn't.
            cp_size = cp_mesh.size()
            if self.n_heads % cp_size != 0:
                raise OLMoConfigurationError(
                    f"Ulysses CP degree ({cp_size}) must divide n_heads ({self.n_heads})"
                )
            if self.n_kv_heads % cp_size != 0 and self.n_heads % self.n_kv_heads != 0:
                raise OLMoConfigurationError(
                    f"Ulysses CP degree ({cp_size}) does not divide n_kv_heads "
                    f"({self.n_kv_heads}), so KV heads must be replicated, but n_kv_heads does not "
                    f"divide n_heads ({self.n_heads})"
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
        # ``seq_len=0`` is how the FLOP meter / joint budget ask for the parameter-only term.
        if seq_len <= 0:
            return param_flops
        attention_positions = _causal_attention_positions(seq_len, self.window_size or None)
        attn_flops = 12 * self.n_heads * self.head_dim * attention_positions // seq_len

        return param_flops + attn_flops


class LandmarkAttention(Attention):
    """
    Landmark attention, a drop-in :class:`Attention` variant.

    Landmark attention inserts a special "landmark" token after every ``mem_freq`` regular tokens,
    dividing the sequence into blocks of ``block_size = mem_freq + 1`` tokens, and computes a grouped
    (two-level) softmax so that a query attends to a block's tokens gated by the attention weight
    assigned to that block's landmark. The projections, QK-norm, RoPE, optional output gating, weight
    init, and tensor parallel plan are all inherited from :class:`Attention`. When a
    :class:`GateConfig` is supplied the gate is applied to the landmark attention output
    (``att * sigmoid(w_g(x))``) exactly as in the base attention, which lets landmark attention drop
    into gated models like Qwen3.5.

    Two forward implementations are available, selected by ``use_kernel``:

    - **Eager** (default): a dense :func:`~olmo_core.nn.attention.landmark.landmark_grouped_softmax`
      that is fully autograd-differentiable, so it provides a working forward *and backward* on both
      CPU and GPU. This is ``O(T^2)`` in memory.
    - **Fused kernel** (``use_kernel=True``): the Triton kernel
      (:func:`~olmo_core.nn.attention.landmark_kernel.fused_landmark_attention`), which is flash-style
      (``O(T)`` memory) and requires landmark tokens at fixed periodic positions
      (``pos % block_size == block_size - 1``) and ``mem_freq >= 15``. Its forward and backward are
      validated against the eager path (gradients match exactly in fp32). This is the memory-efficient
      path for long-context training; it is CUDA + triton only.

    .. note::
        Intra-document masking for sequence packing (``cu_doc_lens``) is supported on both the
        **eager** and **fused-kernel** paths: documents are masked block-diagonally so a query never
        attends across a document boundary, and RoPE positions reset per document. This requires every
        packed document length to be a multiple of the landmark ``block_size`` (so the periodic
        ``is_mem`` pattern stays valid). Packing also composes with **Ulysses context parallelism**:
        each rank applies per-document RoPE to its sequence slice (documents straddling a rank
        boundary stay positionally continuous; see
        :func:`~olmo_core.nn.attention.landmark.build_local_packed_position_ids`) before the
        all-to-all gathers the full sequence, on which the block-diagonal document mask is built.
        Generation / KV-caching and long-context landmark retrieval are not yet supported.

    :param mem_freq: The number of regular tokens between landmark tokens. The landmark block size
        is ``mem_freq + 1``.
    :param use_kernel: Use the fused Triton kernel instead of the eager path. Defaults to ``False``.

    See :class:`Attention` for the remaining parameters.
    """

    def __init__(
        self,
        *,
        mem_freq: int,
        use_kernel: bool = False,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ):
        if kwargs.get("window_size") is not None:
            raise OLMoConfigurationError(
                "LandmarkAttention does not support sliding window attention"
            )
        super().__init__(softmax_scale=softmax_scale, **kwargs)
        if mem_freq is None or mem_freq < 1:
            raise OLMoConfigurationError(
                f"LandmarkAttention requires mem_freq >= 1 (got {mem_freq})"
            )
        self.mem_freq = mem_freq
        self.block_size = mem_freq + 1
        self.use_kernel = use_kernel
        self.softmax_scale = softmax_scale if softmax_scale is not None else self.head_dim**-0.5
        # Inference-only hard top-k landmark retrieval (None = exact: attend all allowed blocks).
        # When set, the eager grouped-softmax keeps only the ``_eval_top_k`` highest-scoring landmark
        # blocks per query and gives every other past block zero weight (the landmark paper's decode).
        self._eval_top_k: Optional[int] = None
        # Ulysses context-parallel state, populated by ``apply_cp``. Unlike the base ``Attention``,
        # landmark attention has its own forward and does not route through ``self.backend``, so it
        # performs the Ulysses all-to-all itself (see ``forward``).
        self._cp_pg: Optional[torch.distributed.ProcessGroup] = None
        self._cp_world_size: int = 1

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        """
        Prepare landmark attention for Ulysses context parallelism.

        Ring/zigzag CP splits the sequence into per-rank chunks, which breaks the landmark grouped
        softmax (a query must see the landmark tokens of *all* preceding blocks). Ulysses CP instead
        splits the heads: each rank gathers the full sequence (with ``n_heads / cp_degree`` heads)
        via an all-to-all, so the grouped softmax sees the complete sequence. We therefore only
        support Ulysses, and we perform the all-to-all in :meth:`forward` rather than in
        ``self.backend`` (which landmark attention never uses).

        :param cp_mesh: The context parallel device sub-mesh.
        :param ring: Must be ``None``; ring CP is not supported.
        :param uly: The Ulysses context parallel style.

        :raises OLMoConfigurationError: If ring CP is requested, or if the number of (KV) heads is
            not divisible by the CP degree.
        """
        if ring is not None:
            raise OLMoConfigurationError(
                "LandmarkAttention only supports Ulysses context parallelism, not ring/zigzag CP "
                "(which splits the sequence and breaks the landmark grouped softmax). Use "
                "TransformerContextParallelConfig.ulysses(...)."
            )
        if uly is None:
            raise ValueError("One of 'ring' or 'uly' must be specified")

        cp_size = cp_mesh.size()
        if self.n_heads % cp_size != 0:
            raise OLMoConfigurationError(
                f"Ulysses CP degree ({cp_size}) must divide n_heads ({self.n_heads})"
            )
        if self.n_kv_heads % cp_size != 0:
            raise OLMoConfigurationError(
                f"Ulysses CP degree ({cp_size}) must divide n_kv_heads ({self.n_kv_heads}); "
                f"landmark attention does not support replicating KV heads across CP ranks"
            )

        self._cp_pg = cp_mesh.get_group()
        self._cp_world_size = cp_size

    @property
    def cp_enabled(self) -> bool:
        return self._cp_pg is not None

    @torch.compiler.disable
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
    ) -> torch.Tensor:
        """
        Apply landmark attention to the input of shape ``(batch_size, seq_len, d_model)``.
        """
        if any(
            v is not None
            for v in (
                cu_doc_lens_q,
                cu_doc_lens_k,
                max_doc_len_q,
                max_doc_len_k,
                local_k_slice,
            )
        ):
            raise NotImplementedError(
                "Landmark attention supports symmetric intra-document masking via 'cu_doc_lens' "
                "only; the cross-attention variants (cu_doc_lens_q/k, max_doc_len_q/k, "
                "local_k_slice) are not supported"
            )
        if cache_leftpad is not None or self.kv_cache_manager is not None:
            raise NotImplementedError(
                "KV-caching / generation is not yet supported with landmark attention"
            )

        # ``T_local`` is the per-rank sequence length: the full sequence when CP is disabled, or the
        # CP shard (``T / cp_degree``) under Ulysses CP. RoPE is applied below in ``_prepare_qkv``
        # using the (already CP-sharded) positional buffers, i.e. with correct *global* positions,
        # before the Ulysses all-to-all gathers the full sequence.
        B, T_local, _ = x.shape

        # Per-document RoPE for sequence packing. Without CP the local shard *is* the full sequence,
        # so the standard ``cu_doc_lens`` RoPE path (positions reset to 0 per document) applies. Under
        # Ulysses CP the shard is a contiguous slice of the full sequence while ``cu_doc_lens`` still
        # describes the full sequence, and RoPE runs on the slice *before* the all-to-all gather; so
        # we pass explicit per-document positions for this rank's slice -- correct even for documents
        # that straddle a rank boundary (their positions stay continuous across the boundary).
        rope_cu_doc_lens, position_ids = cu_doc_lens, None
        if cu_doc_lens is not None and self.cp_enabled:
            assert self._cp_pg is not None
            position_ids = build_local_packed_position_ids(
                cu_doc_lens, B, T_local, get_rank(self._cp_pg), self._cp_world_size
            )
            rope_cu_doc_lens = None

        # shape: (B, T_local, n_heads, head_dim), (B, T_local, n_kv_heads, head_dim) x2
        q, k, v = self._prepare_qkv(
            x,
            pos_sin=pos_sin,
            pos_cos=pos_cos,
            freqs_cis=freqs_cis,
            cu_doc_lens=rope_cu_doc_lens,
            position_ids=position_ids,
        )

        if self.cp_enabled:
            assert self._cp_pg is not None
            # Ulysses: gather the full sequence and scatter the heads across CP ranks.
            # (B, T/CP, H, D) -> (B, T, H/CP, D); likewise for the KV heads.
            q = all_to_all_single_cp2hp(q, self._cp_pg)
            k, v = all_to_all_cp2hp([k, v], self._cp_pg)

        # Each rank now holds the full sequence ``T`` (= ``T_local`` without CP).
        T = q.shape[1]
        if T % self.block_size != 0:
            raise OLMoConfigurationError(
                f"Sequence length ({T}) must be a multiple of the landmark block size "
                f"(mem_freq + 1 = {self.block_size})."
            )

        # -> (B, n_heads, T, head_dim), expanding KV heads for GQA. Under CP the head counts here
        # are the per-rank (n_heads / cp_degree) counts; their ratio (n_rep) is unchanged.
        n_rep = q.shape[2] // k.shape[2]
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), n_rep)
        v = repeat_kv(v.transpose(1, 2), n_rep)

        if self.use_kernel:
            if not has_landmark_kernel():
                raise RuntimeError(
                    "LandmarkAttention was built with 'use_kernel=True' but the fused Triton kernel "
                    "is unavailable (install 'triton' and run on a CUDA device)."
                )
            if self.block_size < 16:
                # The kernel tiles by ``block_size`` and uses ``tl.dot``, which requires tile
                # dimensions >= 16.
                raise OLMoConfigurationError(
                    f"The fused landmark kernel requires mem_freq >= 15 (block size "
                    f"{self.block_size} < 16); tl.dot needs tile dimensions of at least 16."
                )
            is_mem = (torch.arange(T, device=q.device) % self.block_size) == (self.block_size - 1)
            # Per-block document ids for sequence packing (None for the single-document path).
            doc_id = (
                build_block_doc_id(cu_doc_lens, B, T, self.block_size)
                if cu_doc_lens is not None
                else None
            )
            # shape: (B, n_heads, T, head_dim)
            att = fused_landmark_attention(
                q,
                k,
                v,
                is_mem,
                sm_scale=self.softmax_scale,
                block_size=self.block_size,
                doc_id=doc_id,
            )
        else:
            # Eager grouped-softmax path: works on CPU/GPU and is fully autograd-differentiable,
            # so it provides a working training backward without the fused kernel.
            # shape: (B, n_heads, T, head_dim)
            att = self._eager_forward(q, k, v, cu_doc_lens=cu_doc_lens)

        # shape: (B, T, n_heads (local), head_dim)
        att = att.transpose(1, 2)

        if self.cp_enabled:
            assert self._cp_pg is not None
            # Ulysses: scatter the sequence back and gather the heads.
            # (B, T, H/CP, D) -> (B, T/CP, H, D)
            att = all_to_all_single_hp2cp(att.contiguous(), self._cp_pg)

        # shape: (B, T_local, n_heads * head_dim)
        att = att.contiguous().view(B, T_local, -1)

        # shape: (B, T_local, n_heads * head_dim)
        att = self._apply_gate(att, x)

        # shape: (B, T_local, d_model)
        return self.w_out(att)

    def _eager_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, n_heads, T, _ = q.shape
        attn_mask, is_mem, last_section_mask = self._landmark_masks(
            T, q.device, q.dtype, cu_doc_lens=cu_doc_lens, batch_size=B
        )

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.softmax_scale
        attn = attn + attn_mask
        attn = torch.maximum(
            attn, torch.tensor(torch.finfo(attn.dtype).min, device=attn.device, dtype=attn.dtype)
        )

        # Inference top-k landmark retrieval: keep only the top-k landmark (block-gate) columns per
        # query; the grouped softmax then zeroes every non-selected block's content.
        if self._eval_top_k is not None:
            attn = self._apply_topk_landmark_retrieval(attn, T)

        probs = landmark_grouped_softmax(
            attn,
            dim=-1,
            is_mem=is_mem.expand(B, n_heads, T, T),
            last_section_mask=last_section_mask.expand(B, 1, T, T),
        ).to(q.dtype)

        # shape: (B, n_heads, T, head_dim)
        return torch.matmul(probs, v)

    def set_eval_top_k(self, top_k: Optional[int]) -> None:
        """
        Enable (or clear, with ``None``) inference-only hard top-k landmark retrieval. At each forward,
        every query keeps only the ``top_k`` highest-scoring landmark blocks and gives all other past
        blocks zero attention weight. Training is unaffected (callers only set this at eval time).
        """
        if top_k is not None and top_k < 1:
            raise OLMoConfigurationError(f"top_k must be >= 1 or None (got {top_k})")
        self._eval_top_k = top_k

    def _apply_topk_landmark_retrieval(self, attn: torch.Tensor, T: int) -> torch.Tensor:
        """
        Mask all but the ``self._eval_top_k`` highest-scoring landmark columns (block gates) per query
        to ``finfo.min``. Landmark columns are the block-end positions; already-masked landmarks
        (``finfo.min`` from the additive mask -- the query's own block or out-of-document blocks under
        chunked masking) rank lowest and are never selected.

        :param attn: Masked attention logits ``(B, n_heads, T, T)``.
        :param T: Sequence length.
        """
        top_k = self._eval_top_k
        assert top_k is not None
        lm_cols = torch.arange(self.block_size - 1, T, self.block_size, device=attn.device)
        if lm_cols.numel() <= top_k:
            return attn
        lm_scores = attn[..., lm_cols]  # (B, n_heads, T, n_landmarks)
        keep = torch.zeros_like(lm_scores, dtype=torch.bool)
        keep.scatter_(-1, lm_scores.topk(top_k, dim=-1).indices, True)
        finfo_min = torch.finfo(attn.dtype).min
        attn = attn.clone()
        attn[..., lm_cols] = lm_scores.masked_fill(~keep, finfo_min)
        return attn

    def _landmark_masks(
        self,
        T: int,
        device: torch.device,
        dtype: torch.dtype,
        cu_doc_lens: Optional[torch.Tensor] = None,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build the additive causal attention mask, the landmark mask (``is_mem``), and the
        ``last_section_mask`` for the eager grouped-softmax path. Without ``cu_doc_lens`` these are
        batch-independent (leading dim 1) and depend only on ``T`` and the landmark block size; with
        ``cu_doc_lens`` the additive mask gains a batch dimension because each batch element can have
        its own document layout.

        :param cu_doc_lens: Optional 1D tensor of cumulative document lengths for sequence packing.
            This follows the same *flattened-over-batch* convention as the flash-attention backend
            (see :func:`olmo_core.data.utils.get_cumulative_document_lengths`): it is the cumulative
            sum of every document length across the whole ``(batch_size, T)`` micro-batch, i.e.
            ``[0, ..., batch_size * T]``, with an implicit boundary at every multiple of ``T`` (since
            each instance is exactly ``T`` tokens). An additive block-diagonal document mask is
            folded into the causal mask so a query never attends across a document boundary (nor to
            another document's landmark tokens). Because the landmark ``is_mem`` pattern is the
            *global* periodic one, every document boundary must fall on a landmark block boundary,
            i.e. each cumulative length must be a multiple of ``block_size``.
        :param batch_size: The number of sequences in the micro-batch. Used to reshape the flattened
            ``cu_doc_lens`` into per-sequence document ids.

        :raises OLMoConfigurationError: If a document boundary is not a multiple of ``block_size``,
            or ``cu_doc_lens`` does not describe exactly ``batch_size * T`` tokens.
        """
        block_size = self.block_size
        finfo_min = torch.finfo(dtype).min

        # additive causal mask, shape (1, 1, T, T): 0 on/below the diagonal, -inf above.
        causal = torch.full((T, T), finfo_min, dtype=dtype, device=device)
        attn_mask = torch.triu(causal, diagonal=1)[None, None].clone()

        if cu_doc_lens is not None:
            # Fold in a block-diagonal document mask: keys in a different document than the query
            # are set to -inf. The periodic ``is_mem`` pattern (below) is computed over global
            # positions, so each document boundary must coincide with a block boundary or the
            # landmark tokens of one document would be mis-grouped with another's.
            boundaries = cu_doc_lens.to(device=device, dtype=torch.long)
            if bool((boundaries % block_size != 0).any().item()):
                raise OLMoConfigurationError(
                    f"Landmark packing requires every document boundary to be a multiple of the "
                    f"landmark block size (mem_freq + 1 = {block_size}), but got "
                    f"cu_doc_lens={cu_doc_lens.tolist()}. This almost always means the document "
                    f"lengths came from the wrong source: EOS-derived 'generate_doc_lengths' or a "
                    f"generic token packer (PackingInstanceSource / ConcatAndChunk) do NOT produce "
                    f"block-aligned landmark documents. Build packed landmark SFT data with "
                    f"LandmarkPackingInstanceSource (it inserts landmarks per document and emits "
                    f"block-aligned doc_lens) and set generate_doc_lengths=False."
                )
            total = batch_size * T
            if int(boundaries[-1].item()) != total:
                raise OLMoConfigurationError(
                    f"cu_doc_lens must describe exactly batch_size * T = {total} tokens "
                    f"(flattened over the batch), but its last entry is {int(boundaries[-1].item())}."
                )
            # Per-token document id over the flattened batch, shape (batch_size * T,): document 0,
            # then 1, etc. ``searchsorted`` maps each flat position to the count of boundaries at or
            # below it. Reshape to (batch_size, T): the implicit boundary at every multiple of T (each
            # instance is exactly T tokens) means a document never straddles two batch elements.
            interior = boundaries[(boundaries > 0) & (boundaries < total)]
            positions = torch.arange(total, device=device)
            doc_id = torch.searchsorted(interior, positions, right=True).view(batch_size, T)
            # (B, 1, T, T): True where a query and key are in different documents.
            cross_doc = doc_id[:, None, :, None] != doc_id[:, None, None, :]
            attn_mask = attn_mask.masked_fill(cross_doc, finfo_min)

        # shape: (1, 1, 1, T)
        is_mem = ((torch.arange(T, device=device) % block_size) == (block_size - 1)).view(
            1, 1, 1, T
        )
        mem_ids = torch.where(attn_mask < -1, -1, torch.cumsum(is_mem, -1) - is_mem.int())
        last_section_mask = torch.amax(mem_ids, -1, keepdim=True) == mem_ids
        # Mask landmark tokens that fall in the query's own (last) section.
        attn_mask.masked_fill_(last_section_mask & is_mem, finfo_min)
        last_section_mask = last_section_mask.logical_and(attn_mask > -1)
        is_mem = is_mem.logical_and(attn_mask > -1)

        return attn_mask, is_mem, last_section_mask


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
            q, k = self._apply_rope(q, k, start_pos, pos_sin, pos_cos, freqs_cis, cu_doc_lens)

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

            q, k = self._apply_rope(q, k, start_pos, pos_sin, pos_cos, freqs_cis, cu_doc_lens)

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


# New landmark-attention sequence mixers. These live in their own modules and subclass ``Attention``
# (defined above), so they are imported at the end of this package to avoid a circular import; the
# ``AttentionConfig.build`` branches above reference them by name at call time.
from .chunked_mask import AttentionPattern  # noqa: E402
from .dilated_window import DilatedSlidingWindowAttention  # noqa: E402
from .document_chunked import DocumentChunkedAttention  # noqa: E402
from .landmark_compressive import FastCompressiveLandmarkAttention  # noqa: E402
from .landmark_compressive_gqa import CompressiveGQAGroupedAttention  # noqa: E402
from .landmark_document import DocumentLandmarkAttention  # noqa: E402
from .landmark_document_compressive import (  # noqa: E402
    DocumentCompressiveLandmarkAttention,
)
from .landmark_fast import FastLandmarkAttention  # noqa: E402
from .landmark_multi import (  # noqa: E402
    DocumentMultiLandmarkAttention,
    MultiLandmarkAttention,
)
from .landmark_multi_compressive import MultiCompressiveLandmarkAttention  # noqa: E402
from .landmark_shared_vector import SharedVectorLandmarkAttention  # noqa: E402
from .landmark_sparse import SparseLandmarkAttention  # noqa: E402
from .pooled_doc_kv import PooledDocKVAttention  # noqa: E402
from .summary_token import SummaryTokenAttention  # noqa: E402
