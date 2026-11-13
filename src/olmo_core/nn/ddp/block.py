import os
import threading
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterator, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Placement
from torch.utils.checkpoint import checkpoint

import olmo_core.nn.transformer.block

try:
    import torch.distributed._symmetric_memory as _symm_mem
except ImportError:
    _symm_mem = None  # type: ignore[assignment]

from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.kernels import ScaledGroupedMMPrequantizedRHS, olmo_symm_mem
from olmo_core.kernels import symm_mem_vdev2d as symm_mem_vdev2d_kernels
from olmo_core.nn.fp8_weight import FP8WeightCacheSpec, FP8WeightStore
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerBlockType
from olmo_core.ops import attach_auxiliary_loss
from olmo_core.utils import get_or_init_stream

from ..attention.base import SequenceMixerConfig
from ..buffer_cache import BufferCache
from ..layer_norm import LayerNorm, LayerNormConfig
from ..moe.moe import LatentMoEConfig
from ..moe.v2.activation_debug import (
    EP_NO_SYNC_SAVED_ACTIVATIONS_DEBUG_ENABLED,
    maybe_dump_ep_no_sync_saved_activations,
)
from ..moe.v2.checkpointing import (
    get_rowwise_checkpoint_state,
    is_checkpoint_recomputing,
)
from ..moe.v2.ep_config import ExpertParallelConfig, ExpertParallelPath
from ..moe.v2.ep_no_sync_1d import (
    combined_forward_ep_no_sync_1d as _combined_forward_ep_no_sync_1d,
)
from ..moe.v2.ep_no_sync_buffers import (
    _NoSyncSymmSharedPool,
    resolve_ep_no_sync_rowwise_symm_options,
)
from ..moe.v2.ep_no_sync_rowwise import (
    combined_forward_ep_no_sync_rowwise as _combined_forward_ep_no_sync_rowwise,
)

# The experimental ``rowwise_wave`` and ``deepep_v2`` backends are imported lazily at their
# dispatch sites below: those modules land in a later PR and ``ExpertParallelConfig`` rejects
# the paths in the meantime, so the module-level import here must not require them.
from ..moe.v2.ep_no_sync_rowwise_helpers import (
    add_ep_no_sync_rowwise_metrics,
    reset_ep_no_sync_rowwise_metrics,
)
from ..moe.v2.ep_no_sync_tbo_rowwise import (
    combined_forward_ep_no_sync_tbo_rowwise as _combined_forward_ep_no_sync_tbo_rowwise,
)
from ..moe.v2.ep_sync_1d import combined_forward_ep_1d as _combined_forward_ep_1d
from ..moe.v2.fp8 import MoERowwiseFP8Config
from ..moe.v2.fp8 import invalidate_rowwise_fp8_cache as _invalidate_rowwise_fp8_cache
from ..moe.v2.fp8 import normalize_rowwise_fp8_config
from ..moe.v2.fp8 import refresh_rowwise_fp8_cache as _refresh_rowwise_fp8_cache
from ..moe.v2.fp8 import shared_experts_forward_rowwise_fp8
from ..moe.v2.no_ep import combined_forward_no_ep as _combined_forward_no_ep
from ..moe.v2.routed_experts import RoutedExperts, RoutedExpertsConfig
from ..moe.v2.router import MoERouterConfigV2, MoERouterV2
from ..moe.v2.shared_experts import SharedExperts, SharedExpertsConfig

# Process-local cache for the EP->group "0" symmetric-memory alias.
# This makes repeated calls from multiple MoE blocks idempotent when they use
# the same EP group.
_EP_SYMM_GROUP0_ALIAS_LOCK = threading.Lock()
_EP_SYMM_GROUP0_ALIAS_RANKS: Optional[Tuple[int, ...]] = None


def _shared_up_gate_rhs(weight: torch.Tensor) -> torch.Tensor:
    return weight.unsqueeze(0)


def _shared_up_gate_rhs_for_dgrad(weight: torch.Tensor) -> torch.Tensor:
    return weight.transpose(0, 1).unsqueeze(0)


def _shared_down_rhs(weight: torch.Tensor) -> torch.Tensor:
    return weight


def _shared_down_rhs_for_dgrad(weight: torch.Tensor) -> torch.Tensor:
    return weight.transpose(1, 2)


_SHARED_UP_GATE_FP8_CACHE_SPECS = (
    FP8WeightCacheSpec("rhs", _shared_up_gate_rhs),
    FP8WeightCacheSpec("rhs_for_dgrad", _shared_up_gate_rhs_for_dgrad),
)
_SHARED_DOWN_FP8_CACHE_SPECS = (
    FP8WeightCacheSpec("rhs", _shared_down_rhs),
    FP8WeightCacheSpec("rhs_for_dgrad", _shared_down_rhs_for_dgrad),
)


@dataclass
class OLMoDDPTransformerBlockConfig(TransformerBlockConfig):
    """
    Configuration for a :class:`OLMoDDPTransformerBlock`.

    Unlike :class:`~olmo_core.nn.transformer.config.TransformerBlockConfig`, this block does not use
    the ``feed_forward`` / ``feed_forward_moe`` fields; the feed-forward sublayer is described by
    :data:`routed_experts` (+ :data:`routed_experts_router`) and the optional :data:`shared_experts`
    (+ :data:`shared_experts_router`). The inherited ``layer_norm`` field is used for both the
    attention and feed-forward norms.
    """

    shared_experts: Optional[SharedExpertsConfig] = None
    """
    The optional shared (always-on) experts applied to every token.
    """

    routed_experts: Optional[RoutedExpertsConfig] = None
    """
    The routed experts that each token is dispatched to via :data:`routed_experts_router`.
    """

    shared_experts_router: Optional[MoERouterConfigV2] = None
    """
    The router that produces the (gated) combine weights for the shared experts. Only needed when
    there is more than one shared expert.
    """

    routed_experts_router: Optional[MoERouterConfigV2] = None
    """
    The router that selects the top-k routed experts for each token and produces their combine
    weights.
    """

    latent_moe: Optional[LatentMoEConfig] = None
    """Optional projections that run only the routed MoE branch in a latent dimension."""

    use_peri_norm: bool = False
    """
    Apply a "peri-norm" — an additional input (pre) norm on the attention and feed-forward
    sublayers, in addition to the branch norms.
    """

    use_pre_norm: bool = False
    """
    Use ``layer_norm`` as the sublayer *input* (pre) norm rather than the post-sublayer branch norm.

    .. note::
        ``use_pre_norm`` / ``use_peri_norm`` overload how the attention/feed-forward norms are
        placed, which makes checkpoint conversion fragile across model families. A future cleanup
        may replace them with explicit placement fields and stable ``pre_*``/``post_*`` norm module
        names (with back-compat aliases).
    """

    checkpoint_attn: bool = False
    """
    Activation-checkpoint the attention sublayer.
    """

    checkpoint_permute_moe_unpermute: bool = False
    """
    Activation-checkpoint the permute → experts → unpermute region of the MoE sublayer.
    """

    checkpoint_second_unpermute: bool = False
    """
    Activation-checkpoint the second (combine) unpermute.
    """

    ep: Optional[ExpertParallelConfig] = None
    """
    Expert-parallel dispatch configuration. Selects the EP family (sync / no-sync, 1D / rowwise);
    ``None`` keeps the block on the no-expert-parallel path.
    """

    rowwise_fp8: Optional[MoERowwiseFP8Config] = None
    """
    Optional rowwise-FP8 configuration for the experts (beta).
    """

    def build(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> olmo_core.nn.transformer.block.TransformerBlockBase:
        assert self.feed_forward is None and self.feed_forward_moe is None, (
            "OLMoDDPTransformerBlock does not support `feed_forward` or `feed_forward_moe`. "
            "Use `shared_experts` for dense/shared MLPs and `routed_experts` for routed MoE."
        )
        if self.latent_moe is not None:
            if self.routed_experts is None or self.routed_experts_router is None:
                raise OLMoConfigurationError(
                    "latent_moe requires routed_experts and routed_experts_router"
                )
            latent_dim = self.latent_moe.latent_dim
            if not 0 < latent_dim < d_model:
                raise OLMoConfigurationError(
                    "latent_moe.latent_dim must be greater than 0 and less than "
                    f"d_model ({d_model}), got {latent_dim}"
                )
            if self.routed_experts.d_model != latent_dim:
                raise OLMoConfigurationError(
                    "latent_moe.latent_dim must equal routed_experts.d_model "
                    f"({latent_dim} != {self.routed_experts.d_model})"
                )
            if self.routed_experts_router.d_model != d_model:
                raise OLMoConfigurationError(
                    "routed_experts_router.d_model must equal block d_model when latent_moe is "
                    f"enabled ({self.routed_experts_router.d_model} != {d_model})"
                )
            if self.shared_experts is not None and self.shared_experts.d_model != d_model:
                raise OLMoConfigurationError(
                    "shared_experts.d_model must equal block d_model when latent_moe is enabled "
                    f"({self.shared_experts.d_model} != {d_model})"
                )
            if (
                self.shared_experts_router is not None
                and self.shared_experts_router.d_model != d_model
            ):
                raise OLMoConfigurationError(
                    "shared_experts_router.d_model must equal block d_model when latent_moe is "
                    f"enabled ({self.shared_experts_router.d_model} != {d_model})"
                )

        kwargs = self.as_dict(exclude_none=False, recurse=False)
        kwargs.pop("name")
        kwargs.pop("feed_forward")  # from parent config
        kwargs.pop("feed_forward_moe")  # from parent config
        # `masked_dropout` (from parent config) is only implemented by the default block.
        if kwargs.pop("masked_dropout"):
            raise OLMoConfigurationError(
                f"'masked_dropout' is only supported by the '{TransformerBlockType.default}' "
                f"block type, got '{self.name}'"
            )
        # The block constructor takes the split attention/feed-forward norms; build both from
        # the single inherited `layer_norm` field (two identical norm modules).
        layer_norm = kwargs.pop("layer_norm")
        if layer_norm is None:
            raise OLMoConfigurationError(
                "OLMoDDPTransformerBlockConfig requires `layer_norm` to be set."
            )
        kwargs["attention_norm"] = layer_norm
        kwargs["feed_forward_norm"] = layer_norm
        kwargs.update(
            d_model=d_model,
            block_idx=block_idx,
            n_layers=n_layers,
            init_device=init_device,
            cache=cache,
        )

        if self.name == TransformerBlockType.moe_fused_v2:
            return OLMoDDPTransformerBlock(**kwargs)
        else:
            raise NotImplementedError(self.name)

    def num_params(self, d_model: int) -> int:
        block_params = 0

        block_params += self.sequence_mixer.num_params(d_model)
        # `layer_norm` is used for both the attention and feed-forward norms.
        if self.layer_norm is not None:
            block_params += self.layer_norm.num_params(d_model)
        if self.shared_experts is not None:
            block_params += self.shared_experts.num_params()
        if self.routed_experts is not None:
            block_params += self.routed_experts.num_params()
        if self.routed_experts_router is not None:
            block_params += self.routed_experts_router.num_params()
        if self.shared_experts_router is not None:
            block_params += self.shared_experts_router.num_params()
        if self.latent_moe is not None:
            block_params += self.latent_moe.num_params(d_model)
        if self.layer_norm is not None:
            block_params += self.layer_norm.num_params(d_model)
        if self.use_peri_norm and self.layer_norm is not None:
            # Peri-norm adds a post-attention and a post-feed-forward norm.
            block_params += 2 * self.layer_norm.num_params(d_model)

        return block_params

    def num_active_params(self, d_model: int) -> int:
        block_params = 0

        block_params += self.sequence_mixer.num_params(d_model)
        # `layer_norm` is used for both the attention and feed-forward norms.
        if self.layer_norm is not None:
            block_params += self.layer_norm.num_params(d_model)
        if self.shared_experts is not None:
            block_params += self.shared_experts.num_params()
        if self.routed_experts is not None:
            assert self.routed_experts_router is not None, "routed_experts must have a router"
            block_params += self.routed_experts.num_active_params(self.routed_experts_router.top_k)
        if self.routed_experts_router is not None:
            block_params += self.routed_experts_router.num_params()
        if self.shared_experts_router is not None:
            block_params += self.shared_experts_router.num_params()
        if self.latent_moe is not None:
            block_params += self.latent_moe.num_params(d_model)
        if self.layer_norm is not None:
            block_params += self.layer_norm.num_params(d_model)
        if self.use_peri_norm and self.layer_norm is not None:
            # Peri-norm adds a post-attention and a post-feed-forward norm.
            block_params += 2 * self.layer_norm.num_params(d_model)

        return block_params

    def flops_per_seq(
        self,
        d_model: int,
        seqlen: int,
        *,
        layer_idx: Optional[int] = None,
        n_layers: Optional[int] = None,
    ) -> int:
        flops = 0

        # attention
        if hasattr(self.sequence_mixer, "flops_per_seq"):
            flops += self.sequence_mixer.flops_per_seq(d_model, seqlen)  # type: ignore[attr-defined]
        else:
            flops += (
                self.sequence_mixer.build(
                    d_model,
                    layer_idx=layer_idx if layer_idx is not None else 0,
                    n_layers=n_layers if n_layers is not None else 1,
                    init_device="meta",
                ).num_flops_per_token(seqlen)
                * seqlen
            )

        # routers
        if self.routed_experts_router is not None:
            flops += (
                6
                * seqlen
                * self.routed_experts_router.d_model
                * self.routed_experts_router.num_experts
            )
        if self.shared_experts_router is not None:
            flops += (
                6
                * seqlen
                * self.shared_experts_router.d_model
                * self.shared_experts_router.num_experts
            )

        # routed experts
        # (seq_len, d_model) * (d_model, expert_hidden_size) * top_k
        # x3 for swiglu (up, down, gate)
        # x3 for fwd+bwd
        # x2 for GEMM
        if self.routed_experts is not None:
            assert self.routed_experts_router is not None, "routed_experts must have a router"
            flops += (
                (3 * 3 * 2)
                * seqlen
                * self.routed_experts.d_model
                * self.routed_experts.hidden_size
                * self.routed_experts_router.top_k
            )
        if self.latent_moe is not None:
            # Both model<->latent projections are used for every routed token.
            flops += 12 * seqlen * d_model * self.latent_moe.latent_dim

        # shared experts
        # (seq_len, d_model) * (d_model, expert_hidden_size) * num_experts
        # x3 for swiglu (up, down, gate)
        # x3 for fwd+bwd
        # x2 for GEMM
        if self.shared_experts is not None:
            flops += (
                (3 * 3 * 2)
                * seqlen
                * d_model
                * self.shared_experts.hidden_size
                * self.shared_experts.num_experts
            )

        return flops


if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType


class OLMoDDPTransformerBlock(olmo_core.nn.transformer.block.TransformerBlockBase):
    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        sequence_mixer: SequenceMixerConfig,
        attention_norm: LayerNormConfig,
        routed_experts_router: Optional[MoERouterConfigV2],
        shared_experts_router: Optional[MoERouterConfigV2],
        shared_experts: Optional[SharedExpertsConfig],
        routed_experts: Optional[RoutedExpertsConfig],
        feed_forward_norm: LayerNormConfig,
        use_peri_norm: bool = False,
        use_pre_norm: bool = False,
        dropout: float = 0.0,
        attention_residual_alpha: Optional[float] = None,
        feed_forward_residual_alpha: Optional[float] = None,
        checkpoint_attn=False,
        checkpoint_permute_moe_unpermute=False,
        checkpoint_second_unpermute=False,
        ep: Optional[ExpertParallelConfig] = None,
        rowwise_fp8: Optional[MoERowwiseFP8Config] = None,
        latent_moe: Optional[LatentMoEConfig] = None,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(n_layers=n_layers)

        assert dropout == 0.0 or dropout is None, "OLMoDDPTransformerBlock does not support dropout"
        self.d_model = d_model
        self.block_idx = block_idx
        self.latent_down_proj: Optional[torch.nn.Linear]
        self.latent_up_proj_input_norm: Optional[LayerNorm]
        self.latent_up_proj: Optional[torch.nn.Linear]
        if latent_moe is None:
            self.latent_down_proj = None
            self.latent_up_proj_input_norm = None
            self.latent_up_proj = None
        else:
            if routed_experts is None:
                raise OLMoConfigurationError("latent_moe requires routed_experts")
            if routed_experts_router is None:
                raise OLMoConfigurationError("latent_moe requires routed_experts_router")
            if not 0 < latent_moe.latent_dim < d_model:
                raise OLMoConfigurationError(
                    "latent_moe.latent_dim must be greater than 0 and less than "
                    f"d_model ({d_model}), got {latent_moe.latent_dim}"
                )
            if routed_experts.d_model != latent_moe.latent_dim:
                raise OLMoConfigurationError(
                    "latent_moe.latent_dim must equal routed_experts.d_model"
                )
            if routed_experts_router.d_model != d_model:
                raise OLMoConfigurationError(
                    "routed_experts_router.d_model must equal block d_model when latent_moe is "
                    "enabled"
                )
            self.latent_down_proj = torch.nn.Linear(
                d_model,
                latent_moe.latent_dim,
                bias=latent_moe.bias,
                dtype=routed_experts.dtype.as_pt(),
                device=init_device,
            )
            self.latent_up_proj_input_norm = (
                latent_moe.resolved_up_proj_input_norm().build(
                    latent_moe.latent_dim, init_device=init_device
                )
                if latent_moe.up_proj_input_norm_enabled
                else None
            )
            self.latent_up_proj = torch.nn.Linear(
                latent_moe.latent_dim,
                d_model,
                bias=latent_moe.bias,
                dtype=routed_experts.dtype.as_pt(),
                device=init_device,
            )

        if attention_residual_alpha is not None:
            raise OLMoConfigurationError(
                "OLMoDDPTransformerBlock does not support attention_residual_alpha"
            )
        if feed_forward_residual_alpha is not None:
            raise OLMoConfigurationError(
                "OLMoDDPTransformerBlock does not support feed_forward_residual_alpha"
            )

        self.routed_experts: Optional[RoutedExperts]
        self.routed_experts_router: Optional[MoERouterV2]
        self.shared_experts: Optional[SharedExperts]
        self.shared_experts_router: Optional[MoERouterV2]
        self.rowwise_fp8 = normalize_rowwise_fp8_config(rowwise_fp8)
        if self.rowwise_fp8 is not None and self.rowwise_fp8.enabled:
            # Config-only check here; runtime CUDA support is asserted at FP8 buffer build.
            self.rowwise_fp8.validate()
        self._shared_rowwise_fp8_up_prequant: Optional[ScaledGroupedMMPrequantizedRHS] = None
        self._shared_rowwise_fp8_down_prequant: Optional[ScaledGroupedMMPrequantizedRHS] = None
        self._shared_rowwise_fp8_up_prequant_t: Optional[ScaledGroupedMMPrequantizedRHS] = None
        self._shared_rowwise_fp8_down_prequant_t: Optional[ScaledGroupedMMPrequantizedRHS] = None
        self._shared_rowwise_fp8_weight_versions: Optional[Tuple[int, int]] = None
        self._shared_rowwise_fp8_up_gate_weight: Optional[FP8WeightStore] = None
        self._shared_rowwise_fp8_down_weight: Optional[FP8WeightStore] = None
        if use_peri_norm and use_pre_norm:
            raise OLMoConfigurationError("use_peri_norm and use_pre_norm are mutually exclusive")
        self.use_peri_norm = use_peri_norm
        self.use_pre_norm = use_pre_norm

        ######## START: Attention ########
        self.attention = sequence_mixer.build(
            d_model, layer_idx=block_idx, n_layers=n_layers, init_device=init_device, cache=cache
        )
        self.attention_norm = attention_norm.build(d_model, init_device=init_device)
        self.attention_input_norm = None
        if self.use_peri_norm:
            self.attention_input_norm = attention_norm.build(d_model, init_device=init_device)

        ######## END: Attention ########

        ######## START: MLP ########
        assert (routed_experts is not None) or (
            shared_experts is not None
        ), "At least one of routed_experts or shared_experts must be provided"

        #### Optional: routed experts ####
        if routed_experts:
            # Routed Experts enabled
            assert (
                routed_experts_router is not None
            ), "Need routed_experts_router when using routed experts"
            routed_experts.rowwise_fp8 = normalize_rowwise_fp8_config(routed_experts.rowwise_fp8)
            if self.rowwise_fp8 is not None and routed_experts.rowwise_fp8 is None:
                routed_experts.rowwise_fp8 = self.rowwise_fp8
            self.routed_experts = routed_experts.build(init_device=init_device)
            owner_ref = weakref.ref(self)
            self.routed_experts.w_up_gate._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
            self.routed_experts.w_down._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
            self.routed_experts_router = routed_experts_router.build(init_device=init_device)
        else:
            # Routed Experts not enabled
            assert (
                routed_experts_router is None
            ), "Should not set routed_experts_router when not using routed experts"
            self.routed_experts = None
            self.routed_experts_router = None
        #### END: Optional: routed experts ####

        #### Optional: shared experts ####
        if shared_experts:
            # Shared Experts enabled
            self.shared_experts = shared_experts.build(init_device=init_device)
            owner_ref = weakref.ref(self)
            self.shared_experts.w_up_gate._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
            self.shared_experts.w_down._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
            self._shared_rowwise_fp8_up_gate_weight = FP8WeightStore(
                logical_name="shared_experts.w_up_gate",
                logical_shape=tuple(self.shared_experts.w_up_gate.shape),
                cache_specs=_SHARED_UP_GATE_FP8_CACHE_SPECS,
                anchor_param=self.shared_experts.w_up_gate,
                optimizer_enabled=(
                    self.rowwise_fp8 is not None
                    and self.rowwise_fp8.enabled
                    and self.rowwise_fp8.fp8_only_params
                ),
            )
            self._shared_rowwise_fp8_down_weight = FP8WeightStore(
                logical_name="shared_experts.w_down",
                logical_shape=tuple(self.shared_experts.w_down.shape),
                cache_specs=_SHARED_DOWN_FP8_CACHE_SPECS,
                anchor_param=self.shared_experts.w_down,
                optimizer_enabled=(
                    self.rowwise_fp8 is not None
                    and self.rowwise_fp8.enabled
                    and self.rowwise_fp8.fp8_only_params
                ),
            )
            # Shared Experts Router
            if shared_experts_router is not None:
                self.shared_experts_router = shared_experts_router.build(init_device=init_device)
            else:
                if shared_experts.num_experts > 1:
                    raise OLMoConfigurationError(
                        "Need shared_experts_router when using more than one shared expert"
                    )
                self.shared_experts_router = None
        else:
            # Shared Experts not enabled
            assert (
                shared_experts_router is None
            ), "Should not set shared_experts_router when not using shared experts"
            self.shared_experts = None
            self.shared_experts_router = None
        #### END: Optional: shared experts ####

        self.feed_forward_norm = feed_forward_norm.build(d_model, init_device=init_device)
        self.feed_forward_input_norm = None
        if self.use_peri_norm:
            self.feed_forward_input_norm = feed_forward_norm.build(d_model, init_device=init_device)
        ######## END: MLP ########

        self.ep_pg = None
        self._ep_enabled = False
        self.tp_pg = None
        self._tp_enabled = False

        # CUDA events for the EP device<->host comm overlap. `torch.cuda.Event()` requires CUDA, so
        # allocate them only when CUDA is available; on a CUDA-less machine they stay None so the
        # block can be *constructed* on CPU (every forward path that uses them is GPU-only). When
        # CUDA is available they are allocated once, up front, so the object ids stay stable for
        # torch.compile across all forward paths (EP and no-EP).
        #
        # The first set serves the single-batch paths (ep_sync_1d / no_ep) and the first batch of
        # two-batch overlap (TBO); the ``_tbo1`` set is the second overlapped TBO batch.
        self._dtoh_event: torch.cuda.Event = None  # type: ignore[assignment]
        self._dtoh_event_send: torch.cuda.Event = None  # type: ignore[assignment]
        self._dtoh_event_recv: torch.cuda.Event = None  # type: ignore[assignment]
        self._before_rev_all2all_event: torch.cuda.Event = None  # type: ignore[assignment]
        self._dtoh_event_tbo1: torch.cuda.Event = None  # type: ignore[assignment]
        self._dtoh_event_send_tbo1: torch.cuda.Event = None  # type: ignore[assignment]
        self._dtoh_event_recv_tbo1: torch.cuda.Event = None  # type: ignore[assignment]
        self._before_rev_all2all_event_tbo1: torch.cuda.Event = None  # type: ignore[assignment]
        if torch.cuda.is_available():
            self.install_cuda_events()

        self.num_local_routed_experts: Optional[int] = (
            self.routed_experts.num_experts if self.routed_experts else None
        )

        self.checkpoint_attn = checkpoint_attn
        self.checkpoint_permute_moe_unpermute = checkpoint_permute_moe_unpermute
        self.checkpoint_second_unpermute = checkpoint_second_unpermute
        self.ep = ep.copy(deep=True) if ep is not None else ExpertParallelConfig()
        self.ep.validate()
        self._ep_symm_group_name: Optional[str] = None
        self._ep_no_sync_symm_cache: Dict[str, torch.Tensor] = {}
        self._ep_no_sync_static_buffer_cache: Dict[Tuple[object, ...], object] = {}
        self._ep_no_sync_rowwise_fp8_static_buffer_cache: Dict[Tuple[object, ...], object] = {}
        self._ep_no_sync_rowwise_static_checkpoint_state: Optional[
            Tuple[bool, Optional[bool]]
        ] = None
        self._ep_no_sync_force_scratch_lifetime_buffers = False
        self._ep_no_sync_symm_lease_pools: Dict[str, object] = {}
        self._ep_no_sync_last_debug: Dict[str, torch.Tensor] = {}
        self._ep_no_sync_shared_pool: Optional[_NoSyncSymmSharedPool] = None
        self._ep_no_sync_shared_slot: int = 0
        self._ep_no_sync_te_backend_warned: bool = False
        self._deepep_v2_runtime: Optional[object] = None
        self._deepep_v2_runtime_cache: Optional[dict[object, object]] = None
        # EP no-sync tail-drop metrics, populated by rowwise and DeepEP paths.
        self._ep_no_sync_rowwise_drop_tokens_sum: Optional[torch.Tensor] = None
        self._ep_no_sync_rowwise_total_tokens_sum: Optional[torch.Tensor] = None
        self._ep_no_sync_rowwise_symm_util_max: Optional[torch.Tensor] = None
        # self._ep_no_sync_forward_call_count: int = 0

    def invalidate_rowwise_fp8_cache(self) -> None:
        _invalidate_rowwise_fp8_cache(self)

    def refresh_rowwise_fp8_cache(self) -> None:
        _refresh_rowwise_fp8_cache(self)

    def _shared_fp8_only_params_enabled(self) -> bool:
        cfg = self.rowwise_fp8
        return cfg is not None and cfg.enabled and cfg.fp8_only_params

    def _sync_shared_rowwise_fp8_weight_anchors(self) -> None:
        if (
            self.shared_experts is None
            or self._shared_rowwise_fp8_up_gate_weight is None
            or self._shared_rowwise_fp8_down_weight is None
        ):
            return
        fp8_only_params = self._shared_fp8_only_params_enabled()
        owner_ref = weakref.ref(self)
        self.shared_experts.w_up_gate._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
        self.shared_experts.w_down._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
        self._shared_rowwise_fp8_up_gate_weight.anchor_param = self.shared_experts.w_up_gate
        self._shared_rowwise_fp8_up_gate_weight.logical_shape = tuple(
            self.shared_experts.w_up_gate.shape
        )
        self._shared_rowwise_fp8_up_gate_weight.optimizer_enabled = fp8_only_params
        self._shared_rowwise_fp8_down_weight.anchor_param = self.shared_experts.w_down
        self._shared_rowwise_fp8_down_weight.logical_shape = tuple(self.shared_experts.w_down.shape)
        self._shared_rowwise_fp8_down_weight.optimizer_enabled = fp8_only_params
        if fp8_only_params:
            self.shared_experts.w_up_gate.requires_grad_(False)
            self.shared_experts.w_down.requires_grad_(False)

    def named_fp8_weight_stores(self) -> Iterator[tuple[str, FP8WeightStore]]:
        if self.routed_experts is not None:
            for name, weight in self.routed_experts.named_fp8_weight_stores():
                yield f"routed_experts.{name}", weight
        self._sync_shared_rowwise_fp8_weight_anchors()
        if self._shared_rowwise_fp8_up_gate_weight is not None:
            yield "shared_experts.w_up_gate", self._shared_rowwise_fp8_up_gate_weight
        if self._shared_rowwise_fp8_down_weight is not None:
            yield "shared_experts.w_down", self._shared_rowwise_fp8_down_weight

    def named_mxfp8_expert_weights(self) -> Iterator[tuple[str, object]]:
        yield from self.named_fp8_weight_stores()

    def zero_fp8_weight_store_grads(self, set_to_none: bool = True) -> None:
        for _, weight in self.named_fp8_weight_stores():
            weight.zero_grad(set_to_none=set_to_none)

    def zero_mxfp8_expert_weight_grads(self, set_to_none: bool = True) -> None:
        self.zero_fp8_weight_store_grads(set_to_none=set_to_none)

    def set_fp8_weight_store_main_grads_to_none(self) -> None:
        for _, weight in self.named_fp8_weight_stores():
            weight.set_main_grad_to_none()

    def set_mxfp8_expert_weight_main_grads_to_none(self) -> None:
        self.set_fp8_weight_store_main_grads_to_none()

    def disable_fp8_weight_anchor_grads(self) -> None:
        if self.routed_experts is not None:
            self.routed_experts.disable_mxfp8_expert_anchor_grads()
        if self._shared_fp8_only_params_enabled() and self.shared_experts is not None:
            self._sync_shared_rowwise_fp8_weight_anchors()
            self.shared_experts.w_up_gate.grad = None
            self.shared_experts.w_down.grad = None
            if hasattr(self.shared_experts.w_up_gate, "_main_grad_fp32"):
                self.shared_experts.w_up_gate._main_grad_fp32 = None  # type: ignore[attr-defined]
            if hasattr(self.shared_experts.w_down, "_main_grad_fp32"):
                self.shared_experts.w_down._main_grad_fp32 = None  # type: ignore[attr-defined]

    def disable_mxfp8_expert_anchor_grads(self) -> None:
        self.disable_fp8_weight_anchor_grads()

    def release_fp8_weight_anchor_storage(self) -> None:
        if self.routed_experts is not None:
            self.routed_experts.release_mxfp8_expert_anchor_storage()
        if not self._shared_fp8_only_params_enabled() or self.shared_experts is None:
            return
        self._sync_shared_rowwise_fp8_weight_anchors()
        if self._shared_rowwise_fp8_up_gate_weight is not None:
            self._shared_rowwise_fp8_up_gate_weight.release_anchor_storage()
        if self._shared_rowwise_fp8_down_weight is not None:
            self._shared_rowwise_fp8_down_weight.release_anchor_storage()
        self.shared_experts._fp8_anchor_storage_released = True

    def release_mxfp8_expert_anchor_storage(self) -> None:
        self.release_fp8_weight_anchor_storage()

    def _sync_fp8_weight_store_grad_from_anchor(
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

    def sync_fp8_weight_store_grads_from_anchor(self) -> None:
        if self.routed_experts is not None:
            self.routed_experts.sync_mxfp8_expert_weight_grads_from_anchor()
        if (
            self._shared_fp8_only_params_enabled()
            and self.shared_experts is not None
            and self._shared_rowwise_fp8_up_gate_weight is not None
            and self._shared_rowwise_fp8_down_weight is not None
        ):
            self._sync_fp8_weight_store_grad_from_anchor(
                self._shared_rowwise_fp8_up_gate_weight,
                self.shared_experts.w_up_gate,
            )
            self._sync_fp8_weight_store_grad_from_anchor(
                self._shared_rowwise_fp8_down_weight,
                self.shared_experts.w_down,
            )

    def sync_mxfp8_expert_weight_grads_from_anchor(self) -> None:
        self.sync_fp8_weight_store_grads_from_anchor()

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        self.zero_mxfp8_expert_weight_grads(set_to_none=set_to_none)

    def purge_cuda_events(self):
        # set all events to None (so that the model can be deepcopied)
        self._dtoh_event = None  # type: ignore[assignment]
        self._dtoh_event_send = None  # type: ignore[assignment]
        self._dtoh_event_recv = None  # type: ignore[assignment]
        self._before_rev_all2all_event = None  # type: ignore[assignment]

        self._dtoh_event_tbo1 = None  # type: ignore[assignment]
        self._dtoh_event_send_tbo1 = None  # type: ignore[assignment]
        self._dtoh_event_recv_tbo1 = None  # type: ignore[assignment]
        self._before_rev_all2all_event_tbo1 = None  # type: ignore[assignment]

    def install_cuda_events(self):
        self._dtoh_event = cast(torch.cuda.Event, torch.cuda.Event())
        self._dtoh_event_send = cast(torch.cuda.Event, torch.cuda.Event())
        self._dtoh_event_recv = cast(torch.cuda.Event, torch.cuda.Event())
        self._before_rev_all2all_event = cast(torch.cuda.Event, torch.cuda.Event())

        self._dtoh_event_tbo1 = cast(torch.cuda.Event, torch.cuda.Event())
        self._dtoh_event_send_tbo1 = cast(torch.cuda.Event, torch.cuda.Event())
        self._dtoh_event_recv_tbo1 = cast(torch.cuda.Event, torch.cuda.Event())
        self._before_rev_all2all_event_tbo1 = cast(torch.cuda.Event, torch.cuda.Event())

    def compute_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        from olmo_core.train.common import ReduceType

        # compute shared and routed experts metrics
        # metrics_shared = self.shared_experts.compute_metrics(reset=reset)
        if self.routed_experts_router:
            metrics_routed = self.routed_experts_router.compute_metrics(reset=reset)
        else:
            metrics_routed = {}
        out = dict(metrics_routed)

        add_ep_no_sync_rowwise_metrics(self, out, ReduceType)

        if reset:
            reset_ep_no_sync_rowwise_metrics(self)

        # metrics = {
        #     "shared": metrics_shared,
        #     "routed": metrics_routed,
        # }
        return out

    def reset_metrics(self):
        # if self.shared_experts_router:
        #     self.shared_experts_router.reset_metrics()
        if self.routed_experts_router:
            self.routed_experts_router.reset_metrics()
        reset_ep_no_sync_rowwise_metrics(self)

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Per-token forward+backward FLOPs estimate for this block.

        Mirrors :meth:`OLMoDDPTransformerBlockConfig.flops_per_seq` divided by ``seqlen``
        — every term in that formula is linear in ``seqlen``, so this is exact.
        """
        d_model = self.d_model
        flops = 0

        # attention
        flops += self.attention.num_flops_per_token(seq_len)

        # Routers always consume model-space activations; only routed expert payloads may use a
        # latent width.
        if self.routed_experts_router is not None:
            flops += 6 * self.routed_experts_router.d_model * self.routed_experts_router.num_experts
        if self.shared_experts_router is not None:
            flops += 6 * self.shared_experts_router.d_model * self.shared_experts_router.num_experts

        # routed experts: top_k active per token; SwiGLU has 3 matmuls; fwd+bwd x3; GEMM x2.
        if self.routed_experts is not None:
            assert self.routed_experts_router is not None
            flops += (
                (3 * 3 * 2)
                * self.routed_experts.d_model
                * self.routed_experts.hidden_size
                * self.routed_experts_router.top_k
            )

        # shared experts: all `num_experts` are always active per token.
        if self.shared_experts is not None:
            flops += (
                (3 * 3 * 2)
                * d_model
                * self.shared_experts.hidden_size
                * self.shared_experts.num_experts
            )

        for projection in (self.latent_down_proj, self.latent_up_proj):
            if projection is not None:
                flops += 6 * projection.weight.numel()

        return flops

    @property
    def has_routed_experts(self) -> bool:
        return self.routed_experts is not None

    @property
    def has_shared_experts(self) -> bool:
        return self.shared_experts is not None

    @property
    def is_shared_only(self) -> bool:
        return self.has_shared_experts and not self.has_routed_experts

    @property
    def is_moe(self) -> bool:
        return self.has_routed_experts

    @property
    def ep_enabled(self) -> bool:
        return self._ep_enabled

    @property
    def tp_enabled(self) -> bool:
        return self._tp_enabled

    def get_dense_stream(self, for_x1=False) -> torch.cuda.Stream:
        if for_x1:  # not used for now
            return get_or_init_stream(id="dense_x1", priority=20)
        else:
            return get_or_init_stream(id="dense", priority=20)

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if not self.has_routed_experts:
            return self.combined_forward_shared_only(x, loss_div_factor=loss_div_factor, **kwargs)

        if not self.ep_enabled:
            return self.combined_forward_no_ep(x, loss_div_factor=loss_div_factor, **kwargs)

        # In eval mode, different ranks might get different input token counts,
        # and no-sync can freeze. Fall back to the synced EP path there.
        if not (self.ep.no_sync and self.training):
            return self.combined_forward_ep_1d(x, loss_div_factor=loss_div_factor, **kwargs)

        if self.ep.path == ExpertParallelPath.rowwise_wave:
            no_sync_forward = self.combined_forward_ep_no_sync_rowwise_wave
        elif self.ep.path == ExpertParallelPath.rowwise_nvshmem:
            no_sync_forward = self.combined_forward_ep_no_sync_rowwise
        elif self.ep.path == ExpertParallelPath.deepep_v2:
            no_sync_forward = self.combined_forward_ep_deepep_v2
        elif self.ep.path == ExpertParallelPath.no_sync_1d:
            no_sync_forward = self.combined_forward_ep_no_sync_1d
        else:
            raise RuntimeError(f"Unsupported EP no-sync path {self.ep.path!r}")

        # Keep the disabled-by-default activation diagnostic out of the compiled hot path. Its
        # per-layer block_idx key otherwise forces one Dynamo graph per block (and per grad mode
        # under reentrant checkpoint), eventually sending unmatched block forwards to eager.
        if EP_NO_SYNC_SAVED_ACTIVATIONS_DEBUG_ENABLED:
            debug_out = maybe_dump_ep_no_sync_saved_activations(
                self,
                x,
                loss_div_factor=loss_div_factor,
                forward_kwargs=kwargs,
                no_sync_forward=no_sync_forward,
            )
            if debug_out is not None:
                return debug_out
        return no_sync_forward(x, loss_div_factor=loss_div_factor, **kwargs)

    def apply_pp(self, pp_mesh: DeviceMesh):
        pass  # nothing to do

    def _ensure_ep_no_sync_symm_backend(self):
        if olmo_symm_mem.is_enabled():
            if not torch.cuda.is_available():
                raise RuntimeError("EP no-sync requires CUDA")
            return

        if _symm_mem is None:
            raise RuntimeError(
                "EP no-sync requires torch.distributed._symmetric_memory, but it is unavailable"
            )

        if not torch.cuda.is_available():
            raise RuntimeError("EP no-sync requires CUDA")

        device = torch.device("cuda", torch.cuda.current_device())
        current_backend = _symm_mem.get_backend(device)
        if current_backend is not None and current_backend.upper() == "NVSHMEM":
            return

        if not _symm_mem.is_nvshmem_available():
            raise RuntimeError(
                "EP no-sync requires NVSHMEM-backed symmetric memory for "
                "all_to_all_vdev, but NVSHMEM is not available in this "
                "PyTorch build/environment."
            )

        try:
            _symm_mem.set_backend("NVSHMEM")
        except Exception as e:
            try:
                backend_after = _symm_mem.get_backend(device)
            except Exception:
                backend_after = None
            raise RuntimeError(
                "EP no-sync requires NVSHMEM-backed symmetric memory for "
                "all_to_all_vdev. Failed to switch backend to NVSHMEM "
                f"(current={current_backend}, after_error={backend_after}): {e}. "
                "Call torch.distributed._symmetric_memory.set_backend('NVSHMEM') "
                "before any symmetric-memory allocations."
            ) from e

    def _try_alias_ep_group_as_world_for_symm_mem(self) -> bool:
        """
        Try to alias EP group metadata as symmetric-memory group "0".

        NVSHMEM-backed ``symm_mem.empty()`` bootstraps through group "0" before
        the later tensor rendezvous sees the EP process group. Each process can
        safely alias its local EP island as group "0" before the first symmetric
        allocation.
        """
        global _EP_SYMM_GROUP0_ALIAS_RANKS
        if _symm_mem is None or self.ep_pg is None:
            return False
        if dist.group.WORLD is None:
            return False
        if self.ep_pg.group_name == dist.group.WORLD.group_name:
            return True

        try:
            import torch.distributed.distributed_c10d as c10d
        except Exception:
            return False

        try:
            alias_ranks = tuple(sorted(c10d._world.pg_group_ranks[self.ep_pg].keys()))
            with _EP_SYMM_GROUP0_ALIAS_LOCK:
                # Idempotent success: alias already installed for the same EP group.
                if _EP_SYMM_GROUP0_ALIAS_RANKS == alias_ranks:
                    return True

                # If group "0" is already registered by a different context,
                # do not overwrite global process state.
                if _symm_mem.is_symm_mem_enabled_for_group("0"):
                    return False

                if not self._register_process_group_for_symm_mem(self.ep_pg, "0"):
                    return False
                _EP_SYMM_GROUP0_ALIAS_RANKS = alias_ranks
                return True
        except Exception:
            return False

    @staticmethod
    def _register_process_group_for_symm_mem(
        group: dist.ProcessGroup,
        group_name: str,
    ) -> bool:
        if _symm_mem is None:
            return False
        if _symm_mem.is_symm_mem_enabled_for_group(group_name):
            return True
        try:
            import torch.distributed.distributed_c10d as c10d
            from torch._C._distributed_c10d import _SymmetricMemory
        except Exception:
            return False

        try:
            global_ranks = sorted(c10d._world.pg_group_ranks[group].keys())
            global_ranks_str = "_".join(map(str, global_ranks))
            store = c10d.PrefixStore(
                f"symmetric_memory-{global_ranks_str}",
                c10d._get_process_group_store(group),
            )
            _SymmetricMemory.set_group_info(
                group_name,
                dist.get_rank(group),
                dist.get_world_size(group),
                store,
            )
            group_to_store = getattr(_symm_mem, "_group_name_to_store", None)
            if isinstance(group_to_store, dict):
                group_to_store[group_name] = store
            return True
        except Exception:
            return False

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        assert (
            self.routed_experts is not None
        ), "ep can only be applied when routed_experts is enabled"
        ep_mp_mesh = ep_mesh["ep_mp"]
        ep_pg = kwargs.get("ep_pg")
        self.ep_mesh = ep_mesh
        self.routed_experts.apply_ep(ep_mesh)
        owner_ref = weakref.ref(self)
        self.routed_experts.w_up_gate._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
        self.routed_experts.w_down._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
        _invalidate_rowwise_fp8_cache(self)
        self.num_local_routed_experts = self.routed_experts.num_local_experts
        self._ep_enabled = True
        self.ep_pg = ep_pg if ep_pg is not None else ep_mp_mesh.get_group()
        self._deepep_v2_runtime = None
        self._deepep_v2_runtime_cache = None

        if self.ep.uses_olmo_symm:
            if olmo_symm_mem.is_enabled() and not self.ep.uses_rowwise_buffers:
                raise RuntimeError(
                    "OLMo-owned symmetric memory currently supports only the rowwise "
                    "or wave/Mega EP no-sync paths. Set OLMO_USE_OWN_SYMM_MEM=0 to use the legacy "
                    "torch.ops.symm_mem.all_to_all_vdev path."
                )
            if _symm_mem is None and not olmo_symm_mem.is_enabled():
                raise RuntimeError(
                    "EP no-sync requires torch.distributed._symmetric_memory, but it is unavailable"
                )
            self._ensure_ep_no_sync_symm_backend()
            assert self.ep_pg is not None
            group_name = self.ep_pg.group_name
            assert (
                dist.group.WORLD is not None
            ), "torch.distributed.group.WORLD must be initialized for EP no-sync to work"
            world_group_name = dist.group.WORLD.group_name
            alias_group0_active = False
            try:
                if olmo_symm_mem.is_enabled():
                    olmo_symm_mem.register_group(
                        self.ep_pg, device=torch.device("cuda", torch.cuda.current_device())
                    )
                else:
                    if not self._register_process_group_for_symm_mem(self.ep_pg, group_name):
                        raise RuntimeError(
                            f"Failed to register EP group '{group_name}' for symmetric memory support"
                        )
                    if os.getenv("OLMO_EP_NO_SYNC_ALIAS_GROUP0", "1").lower() not in (
                        "",
                        "0",
                        "false",
                        "no",
                    ):
                        alias_group0_active = self._try_alias_ep_group_as_world_for_symm_mem()
                        if not alias_group0_active:
                            raise RuntimeError(
                                f"Failed to alias EP group '{group_name}' as group '0' for symmetric memory support"
                            )

            except Exception as e:
                raise RuntimeError(
                    f"Failed to enable symmetric memory for EP group '{group_name}' "
                    f"(world='{world_group_name}', alias_group0_active={alias_group0_active}) "
                    f"(block={self.block_idx}, rank={get_rank(self.ep_pg)}): {e}"
                ) from e
            self._ep_symm_group_name = group_name
            if self.ep.uses_rowwise_buffers:
                resolve_ep_no_sync_rowwise_symm_options(self)
                if self.ep.rowwise_transport == "nvshmem":
                    symm_mem_vdev2d_kernels.preflight_rowwise_collective_launches(
                        self.ep.rowwise_get_nblocks,
                        self.ep.rowwise_put_nblocks,
                        self.ep.rowwise_weighted_put_nblocks,
                    )
            self._ep_no_sync_symm_cache.clear()
            self._ep_no_sync_static_buffer_cache.clear()
            self._ep_no_sync_rowwise_fp8_static_buffer_cache.clear()
            self._ep_no_sync_symm_lease_pools.clear()
            self._ep_no_sync_shared_pool = None
            self._ep_no_sync_shared_slot = 0
            self._ep_no_sync_te_backend_warned = False

    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        raise NotImplementedError("TP is not supported in MoEFusedV1TransformerBlock")

    def apply_cp(self, cp_mesh: DeviceMesh, ring=None, uly=None):
        self.attention.apply_cp(cp_mesh, ring=ring, uly=uly)
        if self.routed_experts_router is not None:
            self.routed_experts_router.apply_cp(cp_mesh)
        if self.shared_experts_router is not None:
            self.shared_experts_router.apply_cp(cp_mesh)

    def apply_fsdp(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError("FSDP is not supported in OLMoDDPTransformerBlock")

    def apply_compile(self):
        self.compile(
            fullgraph=False,
            # dynamic=False
        )

        # NOTE: the rowwise NVSHMEM TBO forward is called by the outer model
        # directly, so compile it here as well.
        self.combined_forward_rowwise_nvshmem_tbo = torch.compile(  # type: ignore[method-assign]
            self.combined_forward_rowwise_nvshmem_tbo
        )
        self._res_norm_attn = torch.compile(self._res_norm_attn)  # type: ignore[method-assign]

    @property
    def ep_world_size(self) -> int:
        if self.ep_pg is not None:
            return get_world_size(self.ep_pg)
        else:
            return 1

    def _attach_routed_aux_loss(
        self,
        x: torch.Tensor,
        routed_expert_router_aux_loss_info: Optional[Tuple[object, ...]],
        *,
        accumulate_metrics: Optional[bool] = None,
    ) -> torch.Tensor:
        if routed_expert_router_aux_loss_info is None:
            return x
        assert self.routed_experts_router is not None
        if accumulate_metrics is None:
            accumulate_metrics = not is_checkpoint_recomputing()
        routed_expert_router_aux_loss = self.routed_experts_router.compute_aux_loss(
            *routed_expert_router_aux_loss_info,
            accumulate_metrics=accumulate_metrics,
        )
        if routed_expert_router_aux_loss is None:
            return x
        return attach_auxiliary_loss(x, routed_expert_router_aux_loss)

    @torch.compiler.disable
    def sync_dtoh_event(self):
        assert self._dtoh_event is not None
        dtoh_event = cast(torch.cuda.Event, self._dtoh_event)
        dtoh_event.synchronize()

    # -------------------------------------------------------------------------
    # Forward path entry points
    # -------------------------------------------------------------------------
    def combined_forward_shared_only(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward for shared-only blocks, equivalent to a dense DDP-stack MLP."""
        assert self.routed_experts is None
        assert self.routed_experts_router is None
        assert self.shared_experts is not None

        block_inp = x
        del x

        attn_res_out = self._checkpointed_res_norm_attn(block_inp, **kwargs)
        kwargs.pop("max_doc_len", None)
        kwargs.pop("cu_doc_lens", None)
        mlp_inp = self._prepare_moe_input(attn_res_out)

        if self.shared_experts_router:
            shared_expert_weights, _, _, _ = self.shared_experts_router(
                mlp_inp,
                True,
                loss_div_factor=loss_div_factor,
            )
        else:
            shared_expert_weights = None

        rowwise_fp8_cfg = self.rowwise_fp8
        # Rowwise FP8 requires CUDA; on CPU (smoke/eval/materialize) fall back to the dense path.
        use_rowwise_fp8 = (
            rowwise_fp8_cfg is not None
            and rowwise_fp8_cfg.enabled
            and mlp_inp.device.type == "cuda"
        )

        if use_rowwise_fp8:
            assert rowwise_fp8_cfg is not None
            shared_out = shared_experts_forward_rowwise_fp8(
                self,
                mlp_inp,
                use_fast_accum=rowwise_fp8_cfg.use_fast_accum,
            )
        else:
            shared_out = self.shared_experts(mlp_inp)
        mlp_out = self._mix_shared_out(
            shared_out,
            shared_expert_weights,
            attn_res_out.shape,
        )

        return self._res_norm_mlp(attn_res_out, mlp_out)

    def combined_forward_no_ep(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return _combined_forward_no_ep(
            self,
            x,
            loss_div_factor=loss_div_factor,
            **kwargs,
        )

    def combined_forward_ep_no_sync_1d(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return _combined_forward_ep_no_sync_1d(
            self,
            x,
            loss_div_factor=loss_div_factor,
            **kwargs,
        )

    def combined_forward_ep_no_sync_rowwise(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        checkpoint_state = self._ep_no_sync_rowwise_static_checkpoint_state
        if checkpoint_state is None:
            checkpoint_state = get_rowwise_checkpoint_state()
        activation_checkpointing, accumulate_routed_aux_loss_metrics = checkpoint_state
        return _combined_forward_ep_no_sync_rowwise(
            self,
            x,
            activation_checkpointing=activation_checkpointing,
            accumulate_routed_aux_loss_metrics=accumulate_routed_aux_loss_metrics,
            loss_div_factor=loss_div_factor,
            **kwargs,
        )

    def combined_forward_ep_no_sync_rowwise_wave(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        checkpoint_state = self._ep_no_sync_rowwise_static_checkpoint_state
        if checkpoint_state is None:
            checkpoint_state = get_rowwise_checkpoint_state()
        activation_checkpointing, accumulate_routed_aux_loss_metrics = checkpoint_state
        from ..moe.v2.ep_no_sync_rowwise_wave import (
            combined_forward_ep_no_sync_rowwise_wave as _combined_forward_ep_no_sync_rowwise_wave,
        )

        return _combined_forward_ep_no_sync_rowwise_wave(
            self,
            x,
            activation_checkpointing=activation_checkpointing,
            accumulate_routed_aux_loss_metrics=accumulate_routed_aux_loss_metrics,
            loss_div_factor=loss_div_factor,
            **kwargs,
        )

    def combined_forward_ep_deepep_v2(
        self,
        x: torch.Tensor,
        *,
        deepep_reentrant_checkpoint: bool = False,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        checkpoint_state = self._ep_no_sync_rowwise_static_checkpoint_state
        if checkpoint_state is None:
            checkpoint_state = get_rowwise_checkpoint_state()
        _, accumulate_routed_aux_loss_metrics = checkpoint_state
        from ..moe.v2.ep_deepep_v2 import (
            combined_forward_ep_deepep_v2 as _combined_forward_ep_deepep_v2,
        )

        return _combined_forward_ep_deepep_v2(
            self,
            x,
            accumulate_routed_aux_loss_metrics=accumulate_routed_aux_loss_metrics,
            accumulate_router_aux_loss_metrics=(True if deepep_reentrant_checkpoint else None),
            loss_div_factor=loss_div_factor,
            **kwargs,
        )

    # TODO(ep-tbo-dispatch): this two-batch-overlap forward (and its checkpointed wrapper below) is
    # currently unreachable — the block's no-sync dispatch selects on ep.path only and never routes
    # to it, so ep.schedule=tbo is rejected by ep_config.validate() as "not yet supported". Wire a
    # path==rowwise_nvshmem & schedule==tbo branch into the forward dispatch, then lift that
    # rejection.
    def combined_forward_rowwise_nvshmem_tbo(
        self,
        x0: torch.Tensor,
        x1_ctx: object,
        x1_is_fresh: bool,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, object]:
        if self.latent_down_proj is not None:
            raise NotImplementedError("LatentMoE is not supported with rowwise NVSHMEM TBO yet")
        if self.ep.path != ExpertParallelPath.rowwise_nvshmem:
            raise RuntimeError(
                "EP TBO is only supported with "
                f"path={ExpertParallelPath.rowwise_nvshmem!r}; "
                f"got path={self.ep.path!r}"
            )
        return _combined_forward_ep_no_sync_tbo_rowwise(
            self,
            x0,
            x1_ctx,
            x1_is_fresh,
            loss_div_factor=loss_div_factor,
            **kwargs,
        )

    def combined_forward_ep_1d(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return _combined_forward_ep_1d(
            self,
            x,
            loss_div_factor=loss_div_factor,
            **kwargs,
        )

    def checkpointed_combined_forward_rowwise_nvshmem_tbo(
        self,
        x0: torch.Tensor,
        x1_ctx: object,
        x1_is_fresh: bool,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, object]:
        if self.ep.checkpoint_tbo:
            out = checkpoint(
                self.combined_forward_rowwise_nvshmem_tbo,
                x0,
                x1_ctx,
                x1_is_fresh,
                loss_div_factor=loss_div_factor,
                use_reentrant=False,
                **kwargs,
            )
            return cast(Tuple[torch.Tensor, object], out)
        else:
            return self.combined_forward_rowwise_nvshmem_tbo(
                x0,
                x1_ctx,
                x1_is_fresh,
                loss_div_factor=loss_div_factor,
                **kwargs,
            )

    # -------------------------------------------------------------------------
    # Shared forward helpers
    # -------------------------------------------------------------------------
    def _prepare_moe_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_peri_norm:
            assert self.feed_forward_input_norm is not None
            return self.feed_forward_input_norm(x)
        if self.use_pre_norm:
            return self.feed_forward_norm(x)
        return x

    def _prepare_routed_moe_input(self, model_input: torch.Tensor) -> torch.Tensor:
        if self.latent_down_proj is None:
            return model_input
        return self.latent_down_proj(model_input)

    def _restore_routed_moe_output(self, routed_output: torch.Tensor) -> torch.Tensor:
        if self.latent_up_proj is None:
            return routed_output
        if self.latent_up_proj_input_norm is not None:
            routed_output = self.latent_up_proj_input_norm(routed_output)
        return self.latent_up_proj(routed_output)

    def _merge_routed_and_shared(
        self,
        routed_out: torch.Tensor,
        mixed_shared_out: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.shared_experts is None:
            return routed_out
        if mixed_shared_out is None:
            raise RuntimeError("shared_experts is enabled but mixed_shared_out is missing")
        return routed_out + mixed_shared_out

    def _mix_shared_out(
        self,
        shared_out: torch.Tensor,
        shared_expert_weights: Optional[torch.Tensor],
        output_shape: torch.Size,
    ) -> torch.Tensor:
        assert self.shared_experts is not None
        B, S, D = output_shape
        if self.shared_experts_router is None:
            if self.shared_experts.num_experts != 1:
                raise RuntimeError("shared_experts_router is required for multiple shared experts")
            return shared_out.squeeze(0)

        if shared_expert_weights is None:
            raise RuntimeError(
                "shared_experts_router is enabled but shared expert weights are missing"
            )
        _, _, E_s = shared_expert_weights.shape
        return (
            torch.bmm(
                shared_expert_weights.to(shared_out.dtype).reshape(B * S, 1, E_s),
                shared_out.permute(1, 2, 0, 3).contiguous().view(B * S, E_s, D),
            )
            .squeeze(1)
            .view(B, S, D)
        )

    def _res_norm_mlp(self, residual: torch.Tensor, mlp_out: torch.Tensor) -> torch.Tensor:
        if self.use_pre_norm:
            return residual + mlp_out
        return residual + self.feed_forward_norm(mlp_out)

    def _res_norm_attn(self, block_inp, **kwargs) -> torch.Tensor:
        attn_in = block_inp
        if self.use_peri_norm:
            assert self.attention_input_norm is not None
            attn_in = self.attention_input_norm(block_inp)
        elif self.use_pre_norm:
            attn_in = self.attention_norm(block_inp)
            return block_inp + self.attention(attn_in, **kwargs)
        attn_res_out = block_inp + self.attention_norm(self.attention(attn_in, **kwargs))
        return attn_res_out

    def _checkpointed_res_norm_attn(self, block_inp, **kwargs) -> torch.Tensor:
        if self.checkpoint_attn:
            out = checkpoint(
                self._res_norm_attn,
                block_inp,
                use_reentrant=False,
                **kwargs,
            )
            return cast(torch.Tensor, out)
        else:
            return self._res_norm_attn(block_inp, **kwargs)

    def post_batch(self, dry_run: bool = False):
        """
        Should be called right after the final backward of a complete batch but before the optimizer step.
        """
        if self.shared_experts_router:
            self.shared_experts_router.post_batch(dry_run=dry_run)
        if self.routed_experts_router:
            self.routed_experts_router.post_batch(dry_run=dry_run)


__all__ = [
    "OLMoDDPTransformerBlock",
    "OLMoDDPTransformerBlockConfig",
]
