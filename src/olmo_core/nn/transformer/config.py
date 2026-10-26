import logging
import math
from collections.abc import Callable
from dataclasses import InitVar, dataclass, field
from fnmatch import fnmatch
from itertools import cycle, islice
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from olmo_core.config import UNSET, DType, StrEnum
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention.base import SequenceMixerConfig
from olmo_core.utils import ensure_multiple_of

from ..attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    AttentionTypePatternConfig,
    GateConfig,
    GateGranularity,
    SlidingWindowAttentionConfig,
    _resolve_attention_backend,
)
from ..attention.recurrent import GatedDeltaNetConfig
from ..buffer_cache import BufferCache
from ..config import ModelConfig, ModuleConfig
from ..feed_forward import ActivationFunction, FeedForwardConfig, FeedForwardType
from ..layer_norm import LayerNormConfig, LayerNormType
from ..lm_head import LMHeadConfig, LMHeadType
from ..moe import MoEConfig, MoERouterConfig, MoEType
from ..rope import PIRoPEScalingConfig, RoPEConfig, RoPEScalingConfig, RoPEType
from .init import InitMethod

if TYPE_CHECKING:
    from .block import TransformerBlockBase
    from .model import Transformer

log = logging.getLogger(__name__)


class TransformerDataParallelWrappingStrategy(StrEnum):
    """
    An enumeration of the different wrapping strategy for the data parallel implementations.
    """

    full = "full"
    """
    Wrap each block and the LM head (only applies to FSDP).
    """

    blocks = "blocks"
    """
    Like full but the LM head is not wrapped separately (only applies to FSDP).
    """

    fine_grained = "fine_grained"
    """
    Wrap certain modules within each block in addition to wrapping each block (only applies to FSDP).
    """


@beta_feature
class TransformerActivationCheckpointingMode(StrEnum):
    """
    An enumeration of the different activation checkpointing modes.
    """

    full = "full"
    """Checkpoint every block."""
    full_offload = "full_offload"
    """Checkpoint every block and hold each block's input in pinned host memory (``olmo_core.nn.offload_checkpoint``)."""
    selected_blocks = "selected_blocks"
    """Checkpoint only selected blocks."""
    selected_modules = "selected_modules"
    """Checkpoint only selected modules."""
    selected_ops = "selected_ops"
    """Checkpoint only a specific set of operations."""
    budget = "budget"
    """Checkpoint based on a budget."""


class TransformerType(StrEnum):
    """
    An enumeration of transformer implementations.
    """

    default = "default"
    """
    ➡️ :class:`Transformer`
    """

    normalized = "normalized"
    """
    ➡️ :class:`NormalizedTransformer` (nGPT)
    """

    moe = "moe"
    """
    ➡️ :class:`MoETransformer`
    """

    moe_fused_v2 = "moe_fused_v2"
    """
    ➡️ :class:`OLMoDDPModel`
    """


class TransformerBlockType(StrEnum):
    """
    An enumeration of the different transformer block implementations.
    """

    default = "default"
    """
    ➡️ :class:`TransformerBlock`
    """

    default_scaled = "default_scaled"
    """
    ➡️ :class:`LayerNormScaledTransformerBlock` (applies LayerNorm Scaling)
    """

    reordered_norm = "reordered_norm"
    """
    ➡️ :class:`ReorderedNormTransformerBlock`
    """

    peri_norm = "peri_norm"
    """
    ➡️ :class:`PeriNormTransformerBlock`
    """

    normalized = "normalized"
    """
    ➡️ :class:`NormalizedTransformerBlock`
    """

    moe = "moe"
    """
    ➡️ :class:`MoETransformerBlock`
    """

    moe_reordered_norm = "moe_reordered_norm"
    """
    ➡️ :class:`MoEReorderedNormTransformerBlock`
    """

    moe_hybrid = "moe_hybrid"
    """
    ➡️ :class:`MoEHybridTransformerBlock`
    """

    moe_hybrid_reordered_norm = "moe_hybrid_reordered_norm"
    """
    ➡️ :class:`MoEHybridReorderedNormTransformerBlock`
    """

    moe_fused_v2 = "moe_fused_v2"
    """
    ➡️ :class:`OLMoDDPTransformerBlock`
    """


@dataclass
class TransformerBlockConfig(ModuleConfig):
    """
    A configuration class for easily building transformer blocks.
    """

    sequence_mixer: SequenceMixerConfig = field(default=UNSET)
    """
    The sequence mixer config (e.g. attention, recurrent, convolution, etc.).
    """
    attention: InitVar[Optional[AttentionConfig]] = None
    """
    .. deprecated::
        Use :data:`sequence_mixer` instead. This field is only kept for backwards compatibility
        with old configs that used ``attention: AttentionConfig``.
    """
    layer_norm: Optional[LayerNormConfig] = None
    """
    The layer norm config.
    """
    feed_forward: Optional[FeedForwardConfig] = None
    """
    The feed-forward config, required for non-MoE blocks.
    """
    feed_forward_moe: Optional[MoEConfig] = None
    """
    The config for the MoE feed-forward layer. Required for MoE blocks.
    """
    name: TransformerBlockType = TransformerBlockType.default
    """
    The block type.
    """
    dropout: Optional[float] = None
    """
    Dropout probability.
    """
    attention_residual_alpha: Optional[float] = None
    """
    A scaling factor applied to the attention/recurrent output before adding it to the residual stream.
    """
    feed_forward_residual_alpha: Optional[float] = None
    """
    A scaling factor applied to the feed-forward (MLP) output before adding it to the residual stream.
    """

    def __post_init__(self, attention: Optional[AttentionConfig] = None):
        # Handle backwards compatibility: old configs used `attention` instead of `sequence_mixer`.
        if attention is not None:
            if self.sequence_mixer is not UNSET:
                raise OLMoConfigurationError(
                    "Cannot specify both 'attention' and 'sequence_mixer' in TransformerBlockConfig. "
                    "Use 'sequence_mixer' only (the 'attention' field is deprecated)."
                )
            self.sequence_mixer = attention
        if self.sequence_mixer is UNSET:
            raise OLMoConfigurationError(
                "TransformerBlockConfig requires 'sequence_mixer' to be set."
            )

    def build(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "TransformerBlockBase":
        from .block import (
            LayerNormScaledTransformerBlock,
            MoEHybridReorderedNormTransformerBlock,
            MoEHybridTransformerBlock,
            MoEReorderedNormTransformerBlock,
            MoETransformerBlock,
            NormalizedTransformerBlock,
            PeriNormTransformerBlock,
            ReorderedNormTransformerBlock,
            TransformerBlock,
        )

        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
        kwargs.update(
            d_model=d_model,
            block_idx=block_idx,
            n_layers=n_layers,
            init_device=init_device,
            cache=cache,
        )

        try:
            if self.name == TransformerBlockType.default:
                return TransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.default_scaled:
                return LayerNormScaledTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.reordered_norm:
                return ReorderedNormTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.peri_norm:
                return PeriNormTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.normalized:
                return NormalizedTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.moe:
                return MoETransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.moe_reordered_norm:
                return MoEReorderedNormTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.moe_hybrid:
                return MoEHybridTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.moe_hybrid_reordered_norm:
                return MoEHybridReorderedNormTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.moe_fused_v2:
                from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock

                return OLMoDDPTransformerBlock(**kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e

    def num_params(self, d_model: int) -> int:
        block_params = 0

        # Block attn and MLP scaling factors.
        if self.name == TransformerBlockType.normalized:
            block_params += 2 * d_model

        # Block attention params.
        block_params += self.sequence_mixer.num_params(d_model)
        if self.layer_norm is not None:
            block_params += self.layer_norm.num_params(d_model)

        # Block feed forward (dense and/or sparse).
        if self.feed_forward is not None:
            block_params += self.feed_forward.num_params(d_model)
            if self.layer_norm is not None:
                block_params += self.layer_norm.num_params(d_model)
        if self.feed_forward_moe is not None:
            block_params += self.feed_forward_moe.num_params(d_model)
            if self.layer_norm is not None:
                block_params += self.layer_norm.num_params(d_model)

        # Two extra norms for Peri-LN block type.
        if self.name == TransformerBlockType.peri_norm:
            assert self.layer_norm is not None
            block_params += 2 * self.layer_norm.num_params(d_model)

        return block_params

    def num_active_params(self, d_model: int) -> int:
        num_params = self.num_params(d_model)
        if self.feed_forward_moe is None:
            return num_params

        num_inactive_params = self.feed_forward_moe.num_params(
            d_model
        ) - self.feed_forward_moe.num_active_params(d_model)
        return num_params - num_inactive_params


@dataclass
class TransformerConfig(ModelConfig):
    """
    A config for easily building transformer models.

    :param name: The name of the implementation.

    See :class:`Transformer` for a description of the other parameters.
    """

    d_model: int
    vocab_size: int
    n_layers: int
    block: TransformerBlockConfig | dict[str, TransformerBlockConfig]
    lm_head: LMHeadConfig
    embedding_norm: Optional[LayerNormConfig] = None
    name: TransformerType = TransformerType.default
    dtype: DType = DType.float32
    init_method: InitMethod = InitMethod.normal
    init_seed: int = 0
    init_std: float = 0.02
    embedding_init_std: Optional[float] = None
    freeze_params: Optional[List[str]] = None
    block_pattern: Optional[List[str]] = None
    document_chunk_attention: Optional[Dict[str, Any]] = None
    """
    Enable runtime ``chunk_id`` reconstruction for document-chunked attention -- either
    :class:`~olmo_core.nn.attention.DocumentLandmarkAttention` (landmark) or
    :class:`~olmo_core.nn.attention.DocumentChunkedAttention` (dense). A dict of
    ``{"doc_start_id", "doc_end_id", "eos_id", "mode"?}`` passed to
    :meth:`~olmo_core.nn.transformer.Transformer.enable_document_chunk_attention` after the model is
    built. Requires the model be uniformly a document-chunked variant and rules out context parallelism.
    """
    summary_token_attention: Optional[Dict[str, Any]] = None
    """
    Enable runtime ``summary_roles`` construction for
    :class:`~olmo_core.nn.attention.summary_token.SummaryTokenAttention`. A dict of
    ``{"doc_start_id", "doc_end_id", "summary_token_id", "eos_id", "pad_id"?, mixture params...}``
    passed to :meth:`~olmo_core.nn.transformer.Transformer.enable_summary_token_attention` after the
    model is built.

    Unlike :attr:`document_chunk_attention` this does **not** require the model be uniformly a
    summary-token variant -- on a hybrid only the attention layers are masked, which is a deliberate
    scoping choice and is logged when enabled.
    """
    block_overrides: Optional[Dict[int, TransformerBlockConfig]] = None
    embed_scale: Optional[float] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.tie_word_embeddings and self.name == TransformerType.normalized:
            raise OLMoConfigurationError(
                "Tying word embeddings is not supported with the normalized transformer"
            )
        validate_block_resolution_config(
            n_layers=self.n_layers,
            block=self.block,
            block_pattern=self.block_pattern,
            block_overrides=self.block_overrides,
        )
        if self.block_pattern is not None and self.n_layers % len(self.block_pattern) != 0:
            log.warning(
                "`n_layers` (%d) is not divisible by the length of `block_pattern` (%d). "
                "The pattern will be cycled and truncated to fit `n_layers`, so the last "
                "cycle will be incomplete.",
                self.n_layers,
                len(self.block_pattern),
            )

    def build(
        self,
        *,
        init_device: str = "cpu",
    ) -> "Transformer":
        """
        Build the model corresponding to this config.

        :param init_device: The device to put the parameters on during initialization. In a
            distributed setting it usually makes sense to set this to "meta".
        """
        from .model import MoETransformer, NormalizedTransformer, Transformer

        log.info(
            f"Building transformer with {self.num_params:,d} total params, "
            f"{self.num_non_embedding_params:,d} non-embedding params"
        )
        model: Transformer
        if self.name == TransformerType.default:
            model = Transformer(
                d_model=self.d_model,
                vocab_size=self.vocab_size,
                n_layers=self.n_layers,
                block=self.block,
                embedding_norm=self.embedding_norm,
                lm_head=self.lm_head,
                dtype=self.dtype.as_pt(),
                init_method=self.init_method,
                init_device=init_device,
                init_seed=self.init_seed,
                init_std=self.init_std,
                embedding_init_std=self.embedding_init_std,
                block_overrides=self.block_overrides,
                block_pattern=self.block_pattern,
                embed_scale=self.embed_scale,
                tie_word_embeddings=self.tie_word_embeddings,
            )
        elif self.name == TransformerType.normalized:
            assert self.embedding_norm is None
            model = NormalizedTransformer(
                d_model=self.d_model,
                vocab_size=self.vocab_size,
                n_layers=self.n_layers,
                block=self.block,
                lm_head=self.lm_head,
                dtype=self.dtype.as_pt(),
                init_method=self.init_method,
                init_device=init_device,
                init_seed=self.init_seed,
                init_std=self.init_std,
                embedding_init_std=self.embedding_init_std,
                block_overrides=self.block_overrides,
                block_pattern=self.block_pattern,
            )
        elif self.name == TransformerType.moe:
            model = MoETransformer(
                d_model=self.d_model,
                vocab_size=self.vocab_size,
                n_layers=self.n_layers,
                block=self.block,
                embedding_norm=self.embedding_norm,
                lm_head=self.lm_head,
                dtype=self.dtype.as_pt(),
                init_method=self.init_method,
                init_device=init_device,
                init_seed=self.init_seed,
                init_std=self.init_std,
                embedding_init_std=self.embedding_init_std,
                block_overrides=self.block_overrides,
                block_pattern=self.block_pattern,
                tie_word_embeddings=self.tie_word_embeddings,
            )
        elif self.name == TransformerType.moe_fused_v2:
            raise RuntimeError("Use OLMoDDPModelConfig")
        else:
            raise NotImplementedError(self.name)

        if self.freeze_params:
            for name, param in model.named_parameters():
                for pattern in self.freeze_params:
                    if fnmatch(name, pattern):
                        param.requires_grad = False
                        log.info(f"Param '{name}' will be frozen")
                        break
                else:
                    log.info(f"Param '{name}' will be trainable")

        if self.document_chunk_attention is not None:
            model.enable_document_chunk_attention(**self.document_chunk_attention)

        if self.summary_token_attention is not None:
            model.enable_summary_token_attention(**self.summary_token_attention)

        log.info("%s", model)
        log.info(
            f"Built model with:\n"
            f"- {model.num_params:,d} total params\n"
            f"- {model.num_non_embedding_params:,d} non-embedding params\n"
            f"- {model.num_trainable_params:,d} trainable params"
        )

        return model

    @property
    def resolved_block_configs(self) -> list[TransformerBlockConfig]:
        return resolve_block_configs(
            n_layers=self.n_layers,
            block=self.block,
            block_pattern=self.block_pattern,
            block_overrides=self.block_overrides,
        )

    @property
    def num_params(self) -> int:
        """
        The total number of parameters that a model from this config would have.
        """
        num_params = 0

        # Embedding params.
        num_params += self.d_model * self.vocab_size
        if self.embedding_norm is not None:
            num_params += self.embedding_norm.num_params(self.d_model)

        # All block params.
        for block_config in self.resolved_block_configs:
            num_params += block_config.num_params(self.d_model)

        # LM head.
        num_params += self.lm_head.num_params(self.d_model, self.vocab_size)

        # The LM head weight is shared with the embeddings when tied.
        if self.tie_word_embeddings:
            num_params -= self.d_model * self.vocab_size

        return num_params

    @property
    def num_active_params(self) -> int:
        """
        The total number of active parameters that a model from this config would have.
        """
        num_active_params = 0

        # Embedding params.
        num_active_params += self.d_model * self.vocab_size
        if self.embedding_norm is not None:
            num_active_params += self.embedding_norm.num_params(self.d_model)

        # All block active params.
        for block_config in self.resolved_block_configs:
            num_active_params += block_config.num_active_params(self.d_model)

        # LM head.
        num_active_params += self.lm_head.num_params(self.d_model, self.vocab_size)

        # The LM head weight is shared with the embeddings when tied.
        if self.tie_word_embeddings:
            num_active_params -= self.d_model * self.vocab_size

        return num_active_params

    @property
    def num_non_embedding_params(self) -> int:
        """
        The number of parameters excluding embedding parameters.
        """
        return self.num_params - self.d_model * self.vocab_size

    @property
    def num_active_non_embedding_params(self) -> int:
        """
        The number of active parameters excluding embedding parameters.
        """
        return self.num_active_params - self.d_model * self.vocab_size

    @classmethod
    def olmo2_1M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=12,
            hidden_size_multiplier=1.0,
            n_layers=kwargs.pop("n_layers", 4),
            n_heads=kwargs.pop("n_heads", 4),
            head_dim=kwargs.pop("head_dim", 4),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_14M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=128,
            n_layers=kwargs.pop("n_layers", 4),
            n_heads=kwargs.pop("n_heads", 8),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_30M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=256,
            n_layers=kwargs.pop("n_layers", 4),
            n_heads=kwargs.pop("n_heads", 8),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_60M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=384,
            hidden_size_multiplier=1.5,
            n_layers=kwargs.pop("n_layers", 8),
            n_heads=kwargs.pop("n_heads", 8),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_100M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 100M OLMo2 model config.
        """
        return cls.llama_like(
            d_model=512,
            hidden_size_multiplier=1.5,
            n_layers=kwargs.pop("n_layers", 12),
            n_heads=kwargs.pop("n_heads", 8),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_190M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=768,
            hidden_size_multiplier=1.5,
            n_layers=kwargs.pop("n_layers", 12),
            n_heads=kwargs.pop("n_heads", 12),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_370M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=1024,
            hidden_size_multiplier=1.5,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_600M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=kwargs.pop("d_model", 1344),
            hidden_size_multiplier=1.5,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_760M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=1536,
            hidden_size_multiplier=1.5,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_1B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1B OLMo2 model config.

        This is different from the OLMo 1B from the old OLMo trainer.
        """
        return cls.llama2_1B(
            vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            hidden_size_multiplier=1.5,
            **kwargs,
        )

    @classmethod
    def olmo2_1B_v2(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1B OLMo2 model config.

        This matches the OLMo 1B from the old OLMo trainer.
        """
        return cls.llama2_1B(
            vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            n_layers=kwargs.pop("n_layers", 16),
            hidden_size_multiplier=kwargs.pop("hidden_size_multiplier", 1.5),
            **kwargs,
        )

    @classmethod
    def olmo2_3B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 3B OLMo2 model config.
        """
        return cls.llama_like(
            d_model=3328,
            hidden_size_multiplier=1.5,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_7B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 7B OLMo2 model config.
        """
        return cls.llama2_7B(
            vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_13B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 13B OLMo2 model config.
        """
        return cls.llama2_13B(
            vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_32B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 32B OLMo2 model config.
        """
        d_model = 5120
        return cls.llama_like(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=kwargs.pop("n_layers", 64),
            n_heads=kwargs.pop("n_heads", 40),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            hidden_size_multiple_of=kwargs.pop("hidden_size_multiple_of", 512),
            hidden_size_multiplier=kwargs.pop("hidden_size_multiplier", 27648 / (8 * d_model / 3)),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo3_1M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        config = cls.olmo2_1M(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_14M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        config = cls.olmo2_14M(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_30M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        config = cls.olmo2_30M(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_60M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        config = cls.olmo2_60M(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_100M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 100M OLMo3 model config.
        """
        config = cls.olmo2_100M(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_190M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 190M OLMo3 model config.
        """
        config = cls.olmo2_190M(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_370M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 370M OLMo3 model config.
        """
        config = cls.olmo2_370M(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_600M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 600M OLMo3 model config.
        """
        config = cls.olmo2_600M(
            vocab_size=vocab_size,
            d_model=kwargs.pop("d_model", 1280),
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_760M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 760M OLMo3 model config.
        """
        config = cls.olmo2_760M(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_1B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1B OLMo3 model config.
        """
        config = cls.olmo2_1B_v2(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_3B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 3B OLMo3 model config.
        """
        config = cls.olmo2_3B(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_7B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 7B OLMo3 model config.
        """
        config = cls.olmo2_7B(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_13B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 13B OLMo3 model config.
        """
        config = cls.olmo2_13B(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_32B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 32B OLMo3 model config.
        """
        config = cls.olmo2_32B(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def smallmoe(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        d_model = kwargs.pop("d_model", 768)
        return cls.llama_like(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 12),
            n_heads=kwargs.pop("n_heads", 12),
            name=kwargs.pop("name", TransformerType.moe),
            block_name=kwargs.pop("block_name", TransformerBlockType.moe_reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            feed_forward_moe=MoEConfig(
                name=MoEType.default,
                num_experts=32,
                hidden_size=int(0.5 * d_model),
                router=MoERouterConfig(top_k=4),
                shared_mlp=FeedForwardConfig(hidden_size=d_model * 2),
                lb_loss_weight=0.01,
                z_loss_weight=0.001,
            ),
        )

    @classmethod
    def small_hybrid_moe(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        d_model = kwargs.pop("d_model", 768)
        return cls.llama_like(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 12),
            n_heads=kwargs.pop("n_heads", 12),
            name=kwargs.pop("name", TransformerType.moe),
            block_name=kwargs.pop("block_name", TransformerBlockType.moe_hybrid_reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            feed_forward=FeedForwardConfig(hidden_size=d_model * 2, bias=False),
            feed_forward_moe=MoEConfig(
                name=MoEType.default,
                num_experts=32,
                hidden_size=int(0.5 * d_model),
                router=MoERouterConfig(top_k=4),
                lb_loss_weight=0.01,
                z_loss_weight=0.001,
            ),
        )

    @classmethod
    def olmoe_1B_7B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        d_model = kwargs.pop("d_model", 2048)
        return cls.llama_like(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            name=kwargs.pop("name", TransformerType.moe),
            block_name=kwargs.pop("block_name", TransformerBlockType.moe_reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            feed_forward_moe=MoEConfig(
                name=MoEType.dropless,
                num_experts=64,
                hidden_size=int(0.5 * d_model),
                router=MoERouterConfig(top_k=8),
                lb_loss_weight=0.01,
                z_loss_weight=0.001,
            ),
        )

    @classmethod
    def ngpt_271M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 271M nGPT model config.
        """
        return cls.ngpt_like(
            d_model=1024,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            **kwargs,
        )

    @classmethod
    def ngpt_1B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1B nGPT model config.
        """
        return cls.ngpt_like(
            d_model=2048,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 18),
            n_heads=kwargs.pop("n_heads", 16),
            **kwargs,
        )

    @classmethod
    def llama2_271M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 271M Llama2-like model config.
        """
        return cls.llama_like(
            d_model=1024,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 8),
            rope_theta=kwargs.pop("rope_theta", 10_000),
            **kwargs,
        )

    @classmethod
    def llama2_1B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1B Llama2-like model config.

        Note: Llama2 doesn't have a 1B. We made this up.
        """
        return cls.llama_like(
            d_model=2048,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 18),
            n_heads=kwargs.pop("n_heads", 16),
            rope_theta=kwargs.pop("rope_theta", 10_000),
            **kwargs,
        )

    @classmethod
    def llama2_7B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 7B Llama2-like model config.
        """
        return cls.llama_like(
            d_model=4096,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 32),
            n_heads=kwargs.pop("n_heads", 32),
            rope_theta=kwargs.pop("rope_theta", 10_000),
            **kwargs,
        )

    @classmethod
    def llama2_13B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 7B Llama2-like model config.
        """
        return cls.llama_like(
            d_model=5120,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 40),
            n_heads=kwargs.pop("n_heads", 40),
            rope_theta=kwargs.pop("rope_theta", 10_000),
            **kwargs,
        )

    @classmethod
    def llama2_26B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 26B Llama2-like model config.
        """
        return cls.llama_like(
            d_model=5120,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 80),
            n_heads=kwargs.pop("n_heads", 40),
            rope_theta=kwargs.pop("rope_theta", 10_000),
            **kwargs,
        )

    @classmethod
    def llama2_70B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 70B Llama2-like model config.
        """
        return cls.llama_like(
            d_model=8192,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 80),
            n_heads=kwargs.pop("n_heads", 64),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            rope_theta=kwargs.pop("rope_theta", 10_000),
            hidden_size_multiplier=1.3,
            hidden_size_multiple_of=4096,
            **kwargs,
        )

    @classmethod
    def llama3_1B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1B Llama3-like model config.
        """
        return cls.llama_like(
            d_model=2048,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 32),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            hidden_size_multiplier=1.5,
            **kwargs,
        )

    @classmethod
    def llama3_8B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        An 8B Llama3-like model config.
        """
        return cls.llama_like(
            d_model=4096,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 32),
            n_heads=kwargs.pop("n_heads", 32),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            hidden_size_multiplier=1.3,
            hidden_size_multiple_of=1024,
            **kwargs,
        )

    @classmethod
    def llama3_70B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 70B Llama3-like model config.
        """
        return cls.llama_like(
            d_model=8196,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 80),
            n_heads=kwargs.pop("n_heads", 64),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            hidden_size_multiplier=1.3,
            hidden_size_multiple_of=4096,
            **kwargs,
        )

    @classmethod
    def llama3_405B(
        cls,
        vocab_size: int,
        **kwargs,
    ) -> "TransformerConfig":
        """
        A 405B Llama3-like model config.
        """
        return cls.llama_like(
            d_model=16384,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 126),
            n_heads=kwargs.pop("n_heads", 128),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            hidden_size_multiplier=1.2,
            hidden_size_multiple_of=4096,
            **kwargs,
        )

    @classmethod
    def gemma3_1B(cls, vocab_size: int = 262208, **kwargs) -> "TransformerConfig":
        """
        Gemma 3 1B model config.
        """
        return cls.gemma3_like(
            d_model=2304,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 26),
            n_heads=kwargs.pop("n_heads", 8),
            n_kv_heads=kwargs.pop("n_kv_heads", 4),
            hidden_size=kwargs.pop("hidden_size", 9216),
            **kwargs,
        )

    @classmethod
    def qwen3_0_6B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=1024,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 28),
            n_heads=kwargs.pop("n_heads", 16),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            head_dim=kwargs.pop("head_dim", 128),
            rope_theta=kwargs.pop("rope_theta", 1_000_000),
            rope_full_precision=kwargs.pop("rope_full_precision", False),
            layer_norm_eps=1e-6,
            layer_norm_name=LayerNormType.qwen_rms,
            qk_norm=kwargs.pop("qk_norm", True),
            use_head_qk_norm=kwargs.pop("use_head_qk_norm", True),
            feed_forward=FeedForwardConfig(
                hidden_size=3072, bias=False, dtype=kwargs.get("dtype", DType.float32)
            ),
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", False),  # see qwen3_5_like
            **kwargs,
        )

    @classmethod
    def gemma3_4B(cls, vocab_size: int = 262208, **kwargs) -> "TransformerConfig":
        """
        Gemma 3 4B model config.
        """
        return cls.gemma3_like(
            d_model=2560,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 34),
            n_heads=kwargs.pop("n_heads", 16),
            n_kv_heads=kwargs.pop("n_kv_heads", 4),
            hidden_size=kwargs.pop("hidden_size", 10240),
            **kwargs,
        )

    @classmethod
    def qwen3_1_7B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=2048,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 28),
            n_heads=kwargs.pop("n_heads", 16),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            head_dim=kwargs.pop("head_dim", 128),
            rope_theta=kwargs.pop("rope_theta", 1_000_000),
            rope_full_precision=kwargs.pop("rope_full_precision", False),
            layer_norm_eps=1e-6,
            layer_norm_name=LayerNormType.qwen_rms,
            qk_norm=kwargs.pop("qk_norm", True),
            use_head_qk_norm=kwargs.pop("use_head_qk_norm", True),
            feed_forward=FeedForwardConfig(
                hidden_size=6144, bias=False, dtype=kwargs.get("dtype", DType.float32)
            ),
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", False),  # see qwen3_5_like
            **kwargs,
        )

    @classmethod
    def gemma3_12B(cls, vocab_size: int = 262208, **kwargs) -> "TransformerConfig":
        """
        Gemma 3 12B model config.
        """
        return cls.gemma3_like(
            d_model=3840,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 48),
            n_heads=kwargs.pop("n_heads", 24),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            hidden_size=kwargs.pop("hidden_size", 15360),
            **kwargs,
        )

    @classmethod
    def qwen3_4B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=2560,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 36),
            n_heads=kwargs.pop("n_heads", 32),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            head_dim=kwargs.pop("head_dim", 128),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            rope_theta=kwargs.pop("rope_theta", 1_000_000),
            rope_full_precision=kwargs.pop("rope_full_precision", False),
            layer_norm_eps=1e-6,
            layer_norm_name=LayerNormType.qwen_rms,
            qk_norm=kwargs.pop("qk_norm", True),
            use_head_qk_norm=kwargs.pop("use_head_qk_norm", True),
            feed_forward=FeedForwardConfig(
                hidden_size=9728, bias=False, dtype=kwargs.get("dtype", DType.float32)
            ),
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", False),  # see qwen3_5_like
            **kwargs,
        )

    @classmethod
    def gemma3_27B(cls, vocab_size: int = 262208, **kwargs) -> "TransformerConfig":
        """
        Gemma 3 27B model config.
        """
        return cls.gemma3_like(
            d_model=5376,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 62),
            n_heads=kwargs.pop("n_heads", 32),
            n_kv_heads=kwargs.pop("n_kv_heads", 16),
            hidden_size=kwargs.pop("hidden_size", 21504),
            **kwargs,
        )

    @classmethod
    def qwen3_8B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=4096,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 36),
            n_heads=kwargs.pop("n_heads", 32),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            head_dim=kwargs.pop("head_dim", 128),
            rope_theta=kwargs.pop("rope_theta", 1_000_000),
            rope_full_precision=kwargs.pop("rope_full_precision", False),
            layer_norm_eps=1e-6,
            layer_norm_name=LayerNormType.qwen_rms,
            qk_norm=kwargs.pop("qk_norm", True),
            use_head_qk_norm=kwargs.pop("use_head_qk_norm", True),
            feed_forward=FeedForwardConfig(
                hidden_size=12288, bias=False, dtype=kwargs.get("dtype", DType.float32)
            ),
            **kwargs,
        )

    @classmethod
    def qwen3_14B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=5120,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 48),
            n_heads=kwargs.pop("n_heads", 40),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            head_dim=kwargs.pop("head_dim", 128),
            rope_theta=kwargs.pop("rope_theta", 1_000_000),
            rope_full_precision=kwargs.pop("rope_full_precision", False),
            layer_norm_eps=1e-6,
            layer_norm_name=LayerNormType.qwen_rms,
            qk_norm=kwargs.pop("qk_norm", True),
            use_head_qk_norm=kwargs.pop("use_head_qk_norm", True),
            feed_forward=FeedForwardConfig(
                hidden_size=17408, bias=False, dtype=kwargs.get("dtype", DType.float32)
            ),
            **kwargs,
        )

    @classmethod
    def qwen3_32B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=5120,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 64),
            n_heads=kwargs.pop("n_heads", 40),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            head_dim=kwargs.pop("head_dim", 128),
            rope_theta=kwargs.pop("rope_theta", 1_000_000),
            rope_full_precision=kwargs.pop("rope_full_precision", False),
            layer_norm_eps=1e-6,
            layer_norm_name=LayerNormType.qwen_rms,
            qk_norm=kwargs.pop("qk_norm", True),
            use_head_qk_norm=kwargs.pop("use_head_qk_norm", True),
            feed_forward=FeedForwardConfig(
                hidden_size=25600, bias=False, dtype=kwargs.get("dtype", DType.float32)
            ),
            **kwargs,
        )

    @classmethod
    def qwen3_5_0_8B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.qwen3_5_like(
            d_model=1024,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 24),
            n_heads=kwargs.pop("n_heads", 8),
            n_kv_heads=kwargs.pop("n_kv_heads", 2),
            head_dim=kwargs.pop("head_dim", 256),
            intermediate_size=kwargs.pop("intermediate_size", 3584),
            linear_num_key_heads=kwargs.pop("linear_num_key_heads", 16),
            linear_num_value_heads=kwargs.pop("linear_num_value_heads", 16),
            **kwargs,
        )

    @classmethod
    def qwen3_5_2B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.qwen3_5_like(
            d_model=2048,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 24),
            n_heads=kwargs.pop("n_heads", 8),
            n_kv_heads=kwargs.pop("n_kv_heads", 2),
            head_dim=kwargs.pop("head_dim", 256),
            intermediate_size=kwargs.pop("intermediate_size", 6144),
            linear_num_key_heads=kwargs.pop("linear_num_key_heads", 16),
            linear_num_value_heads=kwargs.pop("linear_num_value_heads", 16),
            **kwargs,
        )

    @classmethod
    def qwen3_5_4B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.qwen3_5_like(
            d_model=2560,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 32),
            n_heads=kwargs.pop("n_heads", 16),
            n_kv_heads=kwargs.pop("n_kv_heads", 4),
            head_dim=kwargs.pop("head_dim", 256),
            intermediate_size=kwargs.pop("intermediate_size", 9216),
            linear_num_key_heads=kwargs.pop("linear_num_key_heads", 16),
            linear_num_value_heads=kwargs.pop("linear_num_value_heads", 32),
            **kwargs,
        )

    @classmethod
    def qwen3_5_9B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.qwen3_5_like(
            d_model=4096,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 32),
            n_heads=kwargs.pop("n_heads", 16),
            n_kv_heads=kwargs.pop("n_kv_heads", 4),
            head_dim=kwargs.pop("head_dim", 256),
            intermediate_size=kwargs.pop("intermediate_size", 12288),
            linear_num_key_heads=kwargs.pop("linear_num_key_heads", 16),
            linear_num_value_heads=kwargs.pop("linear_num_value_heads", 32),
            **kwargs,
        )

    @classmethod
    def qwen3_5_27B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.qwen3_5_like(
            d_model=5120,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 64),
            n_heads=kwargs.pop("n_heads", 24),
            n_kv_heads=kwargs.pop("n_kv_heads", 4),
            head_dim=kwargs.pop("head_dim", 256),
            intermediate_size=kwargs.pop("intermediate_size", 17408),
            linear_num_key_heads=kwargs.pop("linear_num_key_heads", 16),
            linear_num_value_heads=kwargs.pop("linear_num_value_heads", 48),
            **kwargs,
        )

    @classmethod
    def qwen3_5_like(
        cls,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_conv_kernel_dim: int = 4,
        rope_theta: int = 10_000_000,
        partial_rotary_factor: float = 0.25,
        layer_norm_eps: float = 1e-6,
        fused_ops: bool = False,
        use_flash: Optional[bool] = None,  # Deprecated; use attn_backend instead.
        attn_backend: Optional[AttentionBackendName] = None,
        document_chunked: bool = False,
        full_attention_layers: Optional[List[int]] = None,
        cross_doc_mode: Optional[str] = None,
        dilation_n: Optional[int] = None,
        dilation_m: Optional[int] = None,
        dilation_cycle: Optional[int] = None,
        dilation_max_docs: Optional[int] = None,
        doc_keep_prob: Optional[float] = None,
        random_doc_seed: Optional[int] = None,
        random_doc_per_example: Optional[bool] = None,
        flex_block_size: Optional[int] = None,
        dtype: DType = DType.float32,
        **kwargs,
    ) -> "TransformerConfig":
        """
        Create a Qwen3.5-like hybrid model configuration.

        Qwen3.5 dense models combine Gated DeltaNet (linear attention) layers with
        full attention layers in a 3:1 ratio. Both layer types use pre-norm blocks with
        Qwen-style RMS normalization, per-head QK norm and output gating on full-attention
        layers, and partial RoPE (25% of head dimension by default).

        When ``document_chunked=True`` the **full-attention** blocks use
        :class:`~olmo_core.nn.attention.DocumentChunkedAttention` (the document-chunked mask
        family) instead of plain causal attention, while the Gated DeltaNet (linear-attention)
        blocks are left untouched -- they are inherently unmasked and simply ignore the runtime
        ``chunk_ids`` threaded to every block. This mirrors ``llama_like``'s document-chunked
        knobs (``cross_doc_mode`` and the ``dilation_*`` / ``doc_keep_prob`` / ``random_doc_seed`` /
        ``flex_block_size`` / ``full_attention_layers`` parameters) so the same contradiction /
        long-context document-chunked SFT recipe works on the hybrid Qwen3.5 models. The boundary
        token ids and the optional mask-mixing schedule are supplied separately by setting
        :attr:`TransformerConfig.document_chunk_attention` on the returned config (see
        :meth:`~olmo_core.nn.transformer.Transformer.enable_document_chunk_attention`).
        """
        uses_document_chunked = document_chunked
        if cross_doc_mode is not None and not uses_document_chunked:
            raise OLMoConfigurationError(
                "'cross_doc_mode' is only valid when 'document_chunked=True'."
            )
        if full_attention_layers is not None and not uses_document_chunked:
            raise OLMoConfigurationError(
                "'full_attention_layers' (hybrid full/chunked layers) is only valid with "
                "'document_chunked=True'."
            )
        _hier = cross_doc_mode == "hierarchical_dilated"
        if (
            dilation_n is not None
            or dilation_m is not None
            or dilation_cycle is not None
            or dilation_max_docs is not None
        ) and not (uses_document_chunked and _hier):
            raise OLMoConfigurationError(
                "'dilation_n' / 'dilation_m' / 'dilation_cycle' / 'dilation_max_docs' are only valid "
                "with document_chunked=True and cross_doc_mode='hierarchical_dilated'."
            )
        _rand = cross_doc_mode == "random_doc"
        if (
            doc_keep_prob is not None
            or random_doc_seed is not None
            or random_doc_per_example is not None
        ) and not (uses_document_chunked and _rand):
            raise OLMoConfigurationError(
                "'doc_keep_prob' / 'random_doc_seed' / 'random_doc_per_example' are only valid "
                "with document_chunked=True and "
                "cross_doc_mode='random_doc'."
            )
        layer_norm = LayerNormConfig(
            name=LayerNormType.qwen_rms,
            eps=layer_norm_eps,
            bias=False,
            dtype=dtype,
        )

        gdn_block = TransformerBlockConfig(
            name=TransformerBlockType.default,
            sequence_mixer=GatedDeltaNetConfig(
                n_heads=linear_num_key_heads,
                n_v_heads=linear_num_value_heads,
                head_dim=linear_key_head_dim,
                expand_v=linear_value_head_dim / linear_key_head_dim,
                allow_neg_eigval=False,
                conv_size=linear_conv_kernel_dim,
                norm_eps=layer_norm_eps,
                dtype=dtype,
            ),
            feed_forward=FeedForwardConfig(hidden_size=intermediate_size, bias=False, dtype=dtype),
            layer_norm=layer_norm,
        )

        # Document-chunked mask family (applied ONLY to the full-attention blocks; the Gated
        # DeltaNet blocks are linear attention and stay untouched). Mirrors ``llama_like``.
        att_type = (
            AttentionType.document_chunked if uses_document_chunked else AttentionType.default
        )
        attn_block = TransformerBlockConfig(
            name=TransformerBlockType.default,
            sequence_mixer=AttentionConfig(
                name=att_type,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                bias=False,
                rope=RoPEConfig(
                    name=RoPEType.default,
                    theta=rope_theta,
                    full_precision=kwargs.pop("rope_full_precision", False),
                    partial_rotary_factor=partial_rotary_factor,
                ),
                gate=GateConfig(granularity=GateGranularity.elementwise),
                qk_norm=layer_norm,
                use_head_qk_norm=True,
                backend=_resolve_attention_backend(attn_backend, use_flash),
                # Only pass the document-chunked knobs on the relevant cross_doc_mode (matches
                # ``llama_like``); config.py's downstream builders assert the pairing otherwise.
                cross_doc_mode=cross_doc_mode if uses_document_chunked else None,
                dilation_n=dilation_n if (uses_document_chunked and _hier) else None,
                dilation_m=dilation_m if (uses_document_chunked and _hier) else None,
                dilation_cycle=dilation_cycle if (uses_document_chunked and _hier) else None,
                dilation_max_docs=dilation_max_docs if (uses_document_chunked and _hier) else None,
                doc_keep_prob=doc_keep_prob if (uses_document_chunked and _rand) else None,
                random_doc_seed=random_doc_seed if (uses_document_chunked and _rand) else None,
                random_doc_per_example=(
                    random_doc_per_example if (uses_document_chunked and _rand) else None
                ),
                flex_block_size=flex_block_size if uses_document_chunked else None,
                full_attention_layers=full_attention_layers if uses_document_chunked else None,
                dtype=dtype,
            ),
            feed_forward=FeedForwardConfig(hidden_size=intermediate_size, bias=False, dtype=dtype),
            layer_norm=layer_norm,
        )

        return cls(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            block={"gdn": gdn_block, "attn": attn_block},
            block_pattern=["gdn", "gdn", "gdn", "attn"],
            lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
            dtype=dtype,
            # Upstream defaults this to True (HF Qwen3.5 ties its embeddings). Our Qwen3.5
            # checkpoints (0.8B/2B/4B dense, landmark, GDN hybrid, setA) were trained with SEPARATE
            # embedding and LM-head weights, and tying would make one overwrite the other on load,
            # so the default stays False here. Pass ``tie_word_embeddings=True`` explicitly to opt in.
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", False),
            **kwargs,
        )

    @classmethod
    def llama_like(
        cls,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        gate: Optional[GateConfig] = None,
        qk_norm: bool = False,
        use_head_qk_norm: bool = False,
        layer_norm_eps: float = 1e-5,
        layer_norm_name: Optional[LayerNormType] = None,
        rope_theta: int = 500_000,
        rope_type: Optional[RoPEType] = None,
        rope_full_precision: bool = True,
        no_global_rope: bool = False,
        hidden_size_multiple_of: int = 256,
        hidden_size_multiplier: Optional[float] = None,
        fused_ops: bool = False,
        use_flash: Optional[bool] = None,  # Deprecated; use attn_backend instead.
        attn_backend: Optional[AttentionBackendName] = None,
        sliding_window: Optional[SlidingWindowAttentionConfig] = None,
        landmark: bool = False,
        mem_freq: Optional[int] = None,
        landmark_use_kernel: bool = False,
        fast_landmark: bool = False,
        fast_compressive_landmark: bool = False,
        compressive_gqa_grouped: bool = False,
        multi_compressive_landmark: bool = False,
        landmark_gate_pool: Optional[str] = None,
        nonselected_landmark_mass: Optional[float] = None,
        group_landmark_selection: Optional[str] = None,
        gate_temperature: Optional[bool] = None,
        shared_vector_landmark: bool = False,
        vec_dim: Optional[int] = None,
        sparse_landmark: bool = False,
        num_landmarks: Optional[int] = None,
        document_landmark: bool = False,
        document_compressive: bool = False,
        document_chunked: bool = False,
        full_attention_layers: Optional[List[int]] = None,
        cross_doc_mode: Optional[str] = None,
        dilation_n: Optional[int] = None,
        dilation_m: Optional[int] = None,
        dilation_cycle: Optional[int] = None,
        dilation_max_docs: Optional[int] = None,
        doc_keep_prob: Optional[float] = None,
        random_doc_seed: Optional[int] = None,
        random_doc_per_example: Optional[bool] = None,
        summary_every_k: Optional[int] = None,
        summary_bandwidth: Optional[int] = None,
        summary_relay: Optional[bool] = None,
        gold_hops: Optional[int] = None,
        gold_decoys: Optional[int] = None,
        flex_block_size: Optional[int] = None,
        dilated_sliding_window: bool = False,
        dilated_window_k: Optional[int] = None,
        dilated_window_num_configs: Optional[int] = None,
        dilated_window_base: Optional[int] = None,
        layer_types: Optional[AttentionTypePatternConfig] = None,
        block_name: TransformerBlockType = TransformerBlockType.default,
        block_mods: Optional[
            Dict[int, Callable[[TransformerBlockConfig], TransformerBlockConfig]]
        ] = None,
        dtype: DType = DType.float32,
        rope_scaling: Optional[RoPEScalingConfig] = None,
        feed_forward: Optional[FeedForwardConfig] = None,
        feed_forward_moe: Optional[MoEConfig] = None,
        **kwargs,
    ) -> "TransformerConfig":
        """
        Create a Llama-like model configuration.

        :param hidden_size_multiple_of: Ensure the FFN hidden size is a multiple of this value.
        :param hidden_size_multiplier: Custom multiplier for the FFN hidden size.
        :param fused_ops: Use fused operations where possible.
        :param layer_norm_name: Override the layer norm implementation. Defaults to
            :data:`LayerNormType.fused_rms` when ``fused_ops=True``, otherwise
            :data:`LayerNormType.rms`.
        :param block_mods: A dictionary of block indices to functions that take the base block config and return a modified block config.
        :param layer_types: Optionally select the attention :class:`AttentionType` per layer via an
            :class:`AttentionTypePatternConfig` (e.g. to mix full attention with landmark variants).
            Mutually exclusive with the uniform ``landmark`` / ``fast_landmark`` / ``sparse_landmark``
            flags; shared landmark params (``mem_freq`` / ``num_landmarks``) still apply.
        :param dtype: The default data type to use for all parameters.
        """
        # Resolve hidden size of FFN in blocks.
        hidden_size = int(8 * d_model / 3)
        if hidden_size_multiplier is not None:
            hidden_size = int(hidden_size_multiplier * hidden_size)
        hidden_size = ensure_multiple_of(hidden_size, hidden_size_multiple_of)

        # Configure global layer norm.
        if layer_norm_name is None:
            layer_norm_name = LayerNormType.fused_rms if fused_ops else LayerNormType.rms
        layer_norm = LayerNormConfig(
            name=layer_norm_name,
            eps=layer_norm_eps,
            bias=False,
            dtype=dtype,
        )

        # Decide on attention/rope implementations.
        att_type = AttentionType.default
        if rope_type is None:
            rope_type = RoPEType.default
            if fused_ops and n_kv_heads is None:
                att_type = AttentionType.fused_v2
                use_flash = None  # The original fused implementation ignored this flag.
                if attn_backend is None:
                    attn_backend = AttentionBackendName.flash_2

        if (
            sum(
                [
                    landmark,
                    fast_landmark,
                    fast_compressive_landmark,
                    compressive_gqa_grouped,
                    multi_compressive_landmark,
                    shared_vector_landmark,
                    sparse_landmark,
                    document_landmark,
                    document_compressive,
                ]
            )
            > 1
        ):
            raise OLMoConfigurationError(
                "Only one of 'landmark', 'fast_landmark', 'fast_compressive_landmark', "
                "'compressive_gqa_grouped', 'multi_compressive_landmark', 'shared_vector_landmark', "
                "'sparse_landmark', 'document_landmark', 'document_compressive' may be set."
            )

        uses_uniform_landmark = (
            landmark
            or fast_landmark
            or fast_compressive_landmark
            or compressive_gqa_grouped
            or multi_compressive_landmark
            or shared_vector_landmark
            or sparse_landmark
            or document_landmark
            or document_compressive
        )
        if document_chunked and (uses_uniform_landmark or layer_types is not None):
            raise OLMoConfigurationError(
                "'document_chunked' (dense chunked attention) cannot be combined with a landmark "
                "variant or 'layer_types'."
            )
        if full_attention_layers is not None and not document_chunked:
            raise OLMoConfigurationError(
                "'full_attention_layers' (hybrid full/chunked layers) is only valid with "
                "'document_chunked=True'."
            )
        if dilated_sliding_window:
            raise NotImplementedError(
                "'dilated_sliding_window' (the positional 'Hierarchical K' rotating-dilation window) "
                "is DEPRECATED and no longer supported. Use the document-chunked hierarchical-dilated "
                "pattern instead: document_chunked=True with cross_doc_mode='hierarchical_dilated' "
                "(rotating per-layer dilation via 'dilation_cycle')."
            )
        if dilated_sliding_window and (
            uses_uniform_landmark or document_chunked or layer_types is not None
        ):
            raise OLMoConfigurationError(
                "'dilated_sliding_window' cannot be combined with a landmark variant, "
                "'document_chunked', or 'layer_types'."
            )
        if dilated_sliding_window and sliding_window is not None:
            raise OLMoConfigurationError(
                "'dilated_sliding_window' has its own dilated-window mask and cannot be combined "
                "with 'sliding_window'."
            )
        if (
            dilated_window_k is not None
            or dilated_window_num_configs is not None
            or dilated_window_base is not None
        ) and not dilated_sliding_window:
            raise OLMoConfigurationError(
                "'dilated_window_k' / 'dilated_window_num_configs' / 'dilated_window_base' are only "
                "valid with 'dilated_sliding_window=True'."
            )
        # ``hierarchical_dilated`` is a cross-document visibility policy orthogonal to the attention
        # mechanism, so the dilation knobs apply to every document-chunked family.
        uses_document_chunked_family = document_chunked or document_landmark or document_compressive
        if (
            dilation_n is not None
            or dilation_m is not None
            or dilation_cycle is not None
            or dilation_max_docs is not None
        ) and not uses_document_chunked_family:
            raise OLMoConfigurationError(
                "'dilation_n' / 'dilation_m' / 'dilation_cycle' / 'dilation_max_docs' are only valid "
                "with a document-chunked family ('document_chunked' / 'document_landmark' / "
                "'document_compressive', with cross_doc_mode='hierarchical_dilated')."
            )
        if (
            summary_every_k is not None
            or summary_bandwidth is not None
            or summary_relay is not None
        ) and not document_chunked:
            raise OLMoConfigurationError(
                "'summary_every_k' / 'summary_bandwidth' / 'summary_relay' are only valid with "
                "'document_chunked=True' and cross_doc_mode='summary_attention'."
            )
        if (gold_hops is not None or gold_decoys is not None) and not document_chunked:
            raise OLMoConfigurationError(
                "'gold_hops' / 'gold_decoys' are only valid with 'document_chunked=True' and "
                "cross_doc_mode='gold_hop_controlled'."
            )

        pattern_landmark_types = layer_types.landmark_types() if layer_types is not None else set()
        pattern_has_plain_landmark = AttentionType.landmark in pattern_landmark_types
        pattern_has_sparse_landmark = AttentionType.sparse_landmark in pattern_landmark_types
        pattern_has_document_landmark = AttentionType.document_landmark in pattern_landmark_types
        pattern_has_compressive_landmark = (
            AttentionType.fast_compressive_landmark in pattern_landmark_types
        )
        pattern_has_shared_vector_landmark = (
            AttentionType.shared_vector_landmark in pattern_landmark_types
        )
        uses_compressive_landmark = (
            fast_compressive_landmark
            or compressive_gqa_grouped
            or multi_compressive_landmark
            or document_compressive
            or pattern_has_compressive_landmark
        )
        uses_shared_vector_landmark = shared_vector_landmark or pattern_has_shared_vector_landmark
        if nonselected_landmark_mass is not None and not uses_compressive_landmark:
            raise OLMoConfigurationError(
                "'nonselected_landmark_mass' is only valid with fast_compressive_landmark, "
                "compressive_gqa_grouped, multi_compressive_landmark, or document_compressive "
                "attention."
            )
        if group_landmark_selection is not None and not uses_compressive_landmark:
            raise OLMoConfigurationError(
                "'group_landmark_selection' is only valid with fast_compressive_landmark, "
                "compressive_gqa_grouped, multi_compressive_landmark, or document_compressive "
                "attention."
            )
        if gate_temperature is not None and not fast_compressive_landmark:
            raise OLMoConfigurationError(
                "'gate_temperature' is only valid with fast_compressive_landmark attention."
            )
        if landmark_gate_pool is not None and not multi_compressive_landmark:
            raise OLMoConfigurationError(
                "'landmark_gate_pool' is only valid with multi_compressive_landmark attention."
            )
        if vec_dim is not None and not uses_shared_vector_landmark:
            raise OLMoConfigurationError(
                "'vec_dim' is only valid with shared_vector_landmark attention."
            )

        if layer_types is not None:
            if uses_uniform_landmark:
                raise OLMoConfigurationError(
                    "'layer_types' selects the attention type per layer and cannot be combined "
                    "with the uniform 'landmark' / 'fast_landmark' / 'fast_compressive_landmark' / "
                    "'sparse_landmark' flags."
                )
            if pattern_landmark_types and mem_freq is None:
                raise OLMoConfigurationError(
                    "'mem_freq' must be set when 'layer_types' includes a landmark attention variant."
                )
            if not pattern_landmark_types and mem_freq is not None:
                raise OLMoConfigurationError(
                    "'mem_freq' is only valid when 'layer_types' includes a landmark attention variant."
                )
            if pattern_landmark_types and sliding_window is not None:
                raise OLMoConfigurationError(
                    "Landmark attention is not compatible with sliding window attention."
                )
            if num_landmarks is not None and not pattern_has_sparse_landmark:
                raise OLMoConfigurationError(
                    "'num_landmarks' is only valid when 'layer_types' includes 'sparse_landmark'."
                )
            if cross_doc_mode is not None and not pattern_has_document_landmark:
                raise OLMoConfigurationError(
                    "'cross_doc_mode' is only valid when 'layer_types' includes 'document_landmark'."
                )
        elif uses_uniform_landmark:
            if mem_freq is None:
                raise OLMoConfigurationError(
                    "'mem_freq' must be set for a landmark attention variant."
                )
            if sliding_window is not None:
                raise OLMoConfigurationError(
                    "Landmark attention is not compatible with sliding window attention."
                )
            if landmark:
                att_type = AttentionType.landmark
            elif fast_landmark:
                att_type = AttentionType.fast_landmark
            elif fast_compressive_landmark:
                att_type = AttentionType.fast_compressive_landmark
            elif compressive_gqa_grouped:
                att_type = AttentionType.compressive_gqa_grouped
            elif multi_compressive_landmark:
                att_type = AttentionType.multi_compressive_landmark
            elif shared_vector_landmark:
                att_type = AttentionType.shared_vector_landmark
            elif sparse_landmark:
                att_type = AttentionType.sparse_landmark
            elif document_compressive:
                att_type = AttentionType.document_compressive_landmark
            else:
                att_type = AttentionType.document_landmark
            if num_landmarks is not None and not (sparse_landmark or multi_compressive_landmark):
                raise OLMoConfigurationError(
                    "'num_landmarks' is only valid when 'sparse_landmark=True' or "
                    "'multi_compressive_landmark=True'."
                )
            if cross_doc_mode is not None and not (document_landmark or document_compressive):
                raise OLMoConfigurationError(
                    "'cross_doc_mode' is only valid when 'document_landmark=True' or "
                    "'document_compressive=True'."
                )
        else:
            if mem_freq is not None:
                raise OLMoConfigurationError(
                    "'mem_freq' is only valid with a landmark attention variant "
                    "(landmark / fast_landmark / sparse_landmark / document_landmark)."
                )
            if num_landmarks is not None:
                raise OLMoConfigurationError(
                    "'num_landmarks' is only valid when 'sparse_landmark=True'."
                )
            if document_chunked:
                att_type = AttentionType.document_chunked
            elif dilated_sliding_window:
                att_type = AttentionType.dilated_sliding_window
            elif cross_doc_mode is not None:
                raise OLMoConfigurationError(
                    "'cross_doc_mode' is only valid when 'document_landmark=True' or "
                    "'document_chunked=True'."
                )

        # Feed-forward.
        if feed_forward is None and feed_forward_moe is None:
            feed_forward = FeedForwardConfig(hidden_size=hidden_size, bias=False, dtype=dtype)

        # Configure blocks.
        block = TransformerBlockConfig(
            name=block_name,
            sequence_mixer=AttentionConfig(
                name=att_type,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                bias=False,
                rope=RoPEConfig(
                    name=rope_type,
                    theta=rope_theta,
                    full_precision=rope_full_precision,
                    no_global_rope=no_global_rope,
                    scaling=rope_scaling,
                ),
                gate=gate,
                qk_norm=layer_norm if qk_norm else None,
                use_head_qk_norm=use_head_qk_norm if qk_norm else None,
                backend=_resolve_attention_backend(attn_backend, use_flash),
                sliding_window=sliding_window,
                layer_types=layer_types,
                mem_freq=mem_freq,
                landmark_use_kernel=(
                    landmark_use_kernel
                    if (
                        landmark
                        or document_landmark
                        or document_compressive
                        or pattern_has_plain_landmark
                        or uses_shared_vector_landmark
                    )
                    else None
                ),
                num_landmarks=(
                    num_landmarks
                    if (
                        sparse_landmark or pattern_has_sparse_landmark or multi_compressive_landmark
                    )
                    else None
                ),
                landmark_gate_pool=(landmark_gate_pool if multi_compressive_landmark else None),
                nonselected_landmark_mass=(
                    nonselected_landmark_mass if uses_compressive_landmark else None
                ),
                group_landmark_selection=(
                    group_landmark_selection if uses_compressive_landmark else None
                ),
                gate_temperature=(gate_temperature if fast_compressive_landmark else None),
                vec_dim=(vec_dim if uses_shared_vector_landmark else None),
                cross_doc_mode=(
                    cross_doc_mode
                    if (
                        document_landmark
                        or document_compressive
                        or document_chunked
                        or pattern_has_document_landmark
                    )
                    else None
                ),
                dilation_n=dilation_n if uses_document_chunked_family else None,
                dilation_m=dilation_m if uses_document_chunked_family else None,
                dilation_cycle=dilation_cycle if uses_document_chunked_family else None,
                dilation_max_docs=dilation_max_docs if uses_document_chunked_family else None,
                # random_doc (dense DocumentChunkedAttention with cross_doc_mode="random_doc"): each doc
                # attends itself + a seeded-random ~doc_keep_prob subset of earlier docs.
                doc_keep_prob=doc_keep_prob if document_chunked else None,
                random_doc_seed=random_doc_seed if document_chunked else None,
                random_doc_per_example=random_doc_per_example if document_chunked else None,
                # summary_attention (dense DocumentChunkedAttention with
                # cross_doc_mode="summary_attention"): cell size, relay bandwidth, and relay on/off.
                summary_every_k=summary_every_k if document_chunked else None,
                summary_bandwidth=summary_bandwidth if document_chunked else None,
                summary_relay=summary_relay if document_chunked else None,
                # gold_hop_controlled (dense DocumentChunkedAttention with
                # cross_doc_mode="gold_hop_controlled"): which arm of the multi-hop gold-routing
                # ladder. Recorded here so config.json names the arm; the per-example gold-edited graph
                # is supplied at runtime by gold_hop_mask.install_gold_hop_mask.
                gold_hops=gold_hops if document_chunked else None,
                gold_decoys=gold_decoys if document_chunked else None,
                # FlexAttention block-mask granularity (dense DocumentChunkedAttention only). Smaller
                # (e.g. 32) lets sub-128-token chunks realize block-sparsity; ignored by landmark
                # variants (no flex path).
                flex_block_size=flex_block_size if document_chunked else None,
                # Hybrid full/chunked: these layer indices run plain causal attention (dense
                # DocumentChunkedAttention only).
                full_attention_layers=full_attention_layers if document_chunked else None,
                # Dilated causal sliding window with per-layer rotating dilation ("Hierarchical K").
                dilated_window_k=dilated_window_k if dilated_sliding_window else None,
                dilated_window_num_configs=(
                    dilated_window_num_configs if dilated_sliding_window else None
                ),
                dilated_window_base=dilated_window_base if dilated_sliding_window else None,
                dtype=dtype,
            ),
            feed_forward=feed_forward,
            feed_forward_moe=feed_forward_moe,
            layer_norm=layer_norm,
        )

        if block_mods and kwargs.get("block_overrides"):
            raise OLMoConfigurationError(
                "`block_mods` and `block_overrides` cannot be used together."
            )
        block_overrides = None
        if block_mods:
            block_overrides = {i: block_mods[i](block.copy()) for i in block_mods}
        elif kwargs.get("block_overrides"):
            block_overrides = kwargs.get("block_overrides")

        return cls(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            block=block,
            lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
            dtype=dtype,
            block_overrides=block_overrides,
            **kwargs,
        )

    @classmethod
    def llama_like_moe(
        cls,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        num_experts: int,
        top_k: int,
        expert_hidden_size: int,
        shared_expert_hidden_size: Optional[int] = None,
        dropless: bool = False,
        capacity_factor: Optional[float] = None,
        lb_loss_weight: float = 0.01,
        z_loss_weight: Optional[float] = 0.001,
        reordered_norm: bool = False,
        hybrid: bool = False,
        **kwargs,
    ) -> "TransformerConfig":
        block_name: TransformerBlockType
        if reordered_norm:
            block_name = (
                TransformerBlockType.moe_hybrid_reordered_norm
                if hybrid
                else TransformerBlockType.moe_reordered_norm
            )
        else:
            block_name = TransformerBlockType.moe_hybrid if hybrid else TransformerBlockType.moe
        return cls.llama_like(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            n_heads=n_heads,
            name=TransformerType.moe,
            block_name=block_name,
            qk_norm=kwargs.pop("qk_norm", reordered_norm),
            feed_forward_moe=MoEConfig(
                name=MoEType.default if not dropless else MoEType.dropless,
                num_experts=num_experts,
                hidden_size=expert_hidden_size,
                capacity_factor=capacity_factor,
                router=MoERouterConfig(top_k=top_k),
                shared_mlp=(
                    None
                    if shared_expert_hidden_size is None
                    else FeedForwardConfig(hidden_size=shared_expert_hidden_size, bias=False)
                ),
                lb_loss_weight=lb_loss_weight,
                z_loss_weight=z_loss_weight,
            ),
            **kwargs,
        )

    @classmethod
    def ngpt_like(
        cls,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        qk_norm: bool = True,
        rope_theta: int = 500_000,
        hidden_size_multiple_of: int = 256,
        hidden_size_multiplier: Optional[float] = None,
        use_flash: bool = False,  # Deprecated; use attn_backend instead.
        attn_backend: Optional[AttentionBackendName] = None,
        dtype: DType = DType.float32,
        **kwargs,
    ) -> "TransformerConfig":
        """
        Create an nGPT-like model configuration.
        """
        # Resolve hidden size of FFN in blocks.
        hidden_size = int(8 * d_model / 3)
        if hidden_size_multiplier is not None:
            hidden_size = int(hidden_size_multiplier * hidden_size)
        hidden_size = ensure_multiple_of(hidden_size, hidden_size_multiple_of)

        # Configure blocks.
        block = TransformerBlockConfig(
            name=TransformerBlockType.normalized,
            sequence_mixer=AttentionConfig(
                name=AttentionType.normalized,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                qk_norm=None if not qk_norm else LayerNormConfig(name=LayerNormType.l2_norm),
                rope=RoPEConfig(name=RoPEType.default, theta=rope_theta),
                backend=_resolve_attention_backend(attn_backend, use_flash),
                dtype=dtype,
            ),
            feed_forward=FeedForwardConfig(
                name=FeedForwardType.normalized, hidden_size=hidden_size, dtype=dtype
            ),
        )

        return cls(
            name=TransformerType.normalized,
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            block=block,
            lm_head=LMHeadConfig(name=LMHeadType.normalized, dtype=dtype),
            dtype=dtype,
            init_method=InitMethod.normalized,
            **kwargs,
        )

    @classmethod
    def gemma3_like(
        cls,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        hidden_size: int,
        head_dim: Optional[int] = None,
        gate: Optional[GateConfig] = None,
        activation: ActivationFunction = ActivationFunction.gelu_tanh,
        local_window_size: int = 1024,
        local_rope_theta: int = 10_000,
        global_rope_theta: int = 1_000_000,
        global_rope_linear_scaling_factor: float = 0.0,
        global_layer_interval: int = 6,
        layer_norm_eps: float = 1e-6,
        fused_ops: bool = False,
        use_flash: Optional[bool] = None,  # Deprecated; use attn_backend instead.
        attn_backend: Optional[AttentionBackendName] = None,
        document_chunked: bool = False,
        cross_doc_mode: Optional[str] = None,
        flex_block_size: Optional[int] = None,
        dtype: DType = DType.float32,
        **kwargs,
    ) -> "TransformerConfig":
        """
        Create a Gemma 3-like model configuration.

        Gemma 3 features:
        - Hybrid local/global attention: 5 local layers with sliding window, then 1 global layer
        - Dual RoPE frequencies: local layers use 10K, global layers use 1M
        - QK-norm for attention score stabilization
        - GeGLU activation (GELU with tanh approximation)

        :param local_window_size: Sliding window size for local attention layers.
        :param local_rope_theta: RoPE base frequency for local attention layers.
        :param global_rope_theta: RoPE base frequency for global attention layers.
        :param global_rope_linear_scaling_factor: Position-interpolation (HF ``rope_type="linear"``)
            factor applied to the **global** layers only, matching Gemma 3's
            ``text_config.rope_scaling`` (``factor=8`` for the released 4B/12B/27B checkpoints).
            ``0`` (default) leaves RoPE unscaled.
        :param global_layer_interval: Number of layers per pattern cycle (default 6 = 5 local + 1 global).
        :param document_chunked: Use :class:`~olmo_core.nn.attention.DocumentChunkedAttention` on the
            **global** (full-attention) layers. The local layers keep plain sliding-window attention:
            :class:`DocumentChunkedAttention` rejects ``window_size`` outright (the chunked mask and a
            sliding window are two different key restrictions), so the chunked mask can only live on
            the global layers. This mirrors the Qwen3.5 hybrid, where the chunked mask applies to the
            full-attention blocks and the GDN blocks are left alone.
        :param cross_doc_mode: The chunked-attention pattern for the global layers; only valid with
            ``document_chunked=True``.
        :param flex_block_size: FlexAttention block size for the chunked global layers.
        """
        if not document_chunked and (cross_doc_mode is not None or flex_block_size is not None):
            raise OLMoConfigurationError(
                "'cross_doc_mode' / 'flex_block_size' are only valid with 'document_chunked=True'."
            )
        layer_norm = LayerNormConfig(
            name=LayerNormType.fused_rms if fused_ops else LayerNormType.rms,
            eps=layer_norm_eps,
            bias=False,
            dtype=dtype,
        )

        local_block = TransformerBlockConfig(
            name=TransformerBlockType.peri_norm,
            sequence_mixer=AttentionConfig(
                name=AttentionType.default,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                bias=False,
                rope=RoPEConfig(name=RoPEType.default, theta=local_rope_theta),
                gate=gate,
                qk_norm=layer_norm,
                use_head_qk_norm=True,
                backend=_resolve_attention_backend(attn_backend, use_flash),
                sliding_window=SlidingWindowAttentionConfig(
                    pattern=[local_window_size],  # Always apply SWA on local_block
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=False,
                ),
                dtype=dtype,
            ),
            feed_forward=FeedForwardConfig(
                hidden_size=hidden_size,
                bias=False,
                dtype=dtype,
                activation=activation,
            ),
            layer_norm=layer_norm,
        )

        global_block = local_block.copy()
        sequence_mixer = cast(AttentionConfig, global_block.sequence_mixer.copy())
        sequence_mixer.rope = RoPEConfig(
            name=RoPEType.default,
            theta=global_rope_theta,
            scaling=(
                PIRoPEScalingConfig(factor=float(global_rope_linear_scaling_factor))
                if global_rope_linear_scaling_factor and global_rope_linear_scaling_factor > 1
                else None
            ),
        )
        sequence_mixer.sliding_window = None
        if document_chunked:
            sequence_mixer.name = AttentionType.document_chunked
            sequence_mixer.cross_doc_mode = cross_doc_mode
            sequence_mixer.flex_block_size = flex_block_size
        global_block.sequence_mixer = sequence_mixer

        blocks = {"local": local_block, "global": global_block}
        block_pattern = ["local"] * (global_layer_interval - 1) + ["global"]

        return cls(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            block=blocks,
            lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
            dtype=dtype,
            block_pattern=block_pattern,
            embed_scale=math.sqrt(d_model),
            **kwargs,
        )

    def with_rope_scaling(
        self, rope_scaling: RoPEScalingConfig, full_attn_layers_only: bool = True
    ) -> "TransformerConfig":
        """
        Return a copy of this config with the given RoPE scaling scheme applied.
        """
        new_config = self.copy()
        if isinstance(new_config.block, dict):
            raise OLMoConfigurationError(
                "Cannot use `with_rope_scaling` with a hybrid model with named blocks."
            )
        assert isinstance(
            new_config.block.sequence_mixer, AttentionConfig
        ), "Sequence mixer must be an attention config for RoPE scaling"
        if new_config.block.sequence_mixer.rope is None:
            raise ValueError("Cannot apply RoPE scaling to a model without RoPE.")
        if new_config.block_overrides:
            raise ValueError("Cannot apply RoPE scaling when block_overrides are already set.")

        def apply_scaling(block_config: TransformerBlockConfig) -> None:
            assert isinstance(block_config.sequence_mixer, AttentionConfig)
            rope_config = block_config.sequence_mixer.rope
            if rope_config is None:
                raise ValueError("Cannot apply RoPE scaling to a layer without RoPE.")
            rope_config = rope_config.copy()
            rope_config.scaling = rope_scaling
            block_config.sequence_mixer.rope = rope_config

        if not full_attn_layers_only:
            apply_scaling(new_config.block)
            return new_config

        # Add rope scaling only to layers that use full attention (no sliding window and no
        # non-default attention type from the layer_types pattern). We supply "block_overrides"
        # for the layers we want to scale.
        overrides: Dict[int, TransformerBlockConfig] = {}
        for i in range(new_config.n_layers):
            sliding_window_cfg = new_config.block.sequence_mixer.sliding_window
            if sliding_window_cfg and sliding_window_cfg.should_use_swa(i, new_config.n_layers):
                continue
            layer_types_cfg = new_config.block.sequence_mixer.layer_types
            if (
                layer_types_cfg is not None
                and layer_types_cfg.get_type(i, new_config.n_layers) != AttentionType.default
            ):
                continue
            block_copy = new_config.block.copy()
            apply_scaling(block_copy)
            overrides[i] = block_copy

        new_config.block_overrides = overrides or None
        return new_config


@dataclass
class OLMoDDPModelConfig(TransformerConfig):
    """
    A :class:`TransformerConfig` for the fused MoE-v2 model
    (:class:`~olmo_core.nn.ddp.model.OLMoDDPModel`).
    """

    two_batch_overlap: bool = False
    """
    Overlap compute with the all-to-all communication when expert parallelism is enabled by
    splitting each micro-batch into two halves. The micro-batch size must be a multiple of 2.
    """

    recompute_all_blocks_by_chunk: bool = False
    """
    Recompute all blocks as a single chunk rather than per-layer. Reduces the activation memory
    held in early pipeline-parallel stages; should not be used without pipeline parallelism as it
    adds recomputation overhead for no benefit.
    """

    recompute_each_block: bool = False
    """
    Recompute each block individually. Keeps only one block's activations live at a time at the
    cost of extra recomputation. Works with or without pipeline parallelism, but is incompatible
    with :data:`two_batch_overlap`.
    """

    recompute_block_keys: Optional[List[str]] = None
    """
    Restrict block-level recomputation to the named submodules of each block.
    """

    def build(
        self,
        *,
        init_device: str = "cpu",
    ) -> "Transformer":
        """
        Build the model corresponding to this config.

        :param init_device: The device to put the parameters on during initialization. In a
            distributed setting it usually makes sense to set this to "meta".
        """
        from .model import Transformer

        log.info(
            f"Building transformer with {self.num_params:,d} total params, "
            f"{self.num_non_embedding_params:,d} non-embedding params"
        )
        model: Transformer
        if self.name == TransformerType.moe_fused_v2:
            from olmo_core.nn.ddp.model import OLMoDDPModel

            model = OLMoDDPModel(
                d_model=self.d_model,
                vocab_size=self.vocab_size,
                n_layers=self.n_layers,
                block=self.block,
                lm_head=self.lm_head,
                dtype=self.dtype.as_pt(),
                init_method=self.init_method,
                init_device=init_device,
                init_seed=self.init_seed,
                init_std=self.init_std,
                block_overrides=self.block_overrides,
                two_batch_overlap=self.two_batch_overlap,
                recompute_all_blocks_by_chunk=self.recompute_all_blocks_by_chunk,
                recompute_each_block=self.recompute_each_block,
                recompute_block_keys=self.recompute_block_keys,
                embedding_norm=self.embedding_norm,
                embedding_init_std=self.embedding_init_std,
                embed_scale=self.embed_scale,
                block_pattern=self.block_pattern,
                tie_word_embeddings=self.tie_word_embeddings,
            )
        else:
            raise NotImplementedError(self.name)

        if self.freeze_params:
            for name, param in model.named_parameters():
                for pattern in self.freeze_params:
                    if fnmatch(name, pattern):
                        param.requires_grad = False
                        log.info(f"Param '{name}' will be frozen")
                        break
                else:
                    log.info(f"Param '{name}' will be trainable")

        log.info("%s", model)
        return model


def validate_block_resolution_config(
    n_layers: int,
    block: TransformerBlockConfig | dict[str, TransformerBlockConfig],
    block_pattern: list[str] | None = None,
    block_overrides: dict[int, TransformerBlockConfig] | None = None,
) -> None:
    if not isinstance(block, dict):
        if block_pattern is not None:
            raise OLMoConfigurationError(
                "`block_pattern` is not supported when `block` is not a dict of named blocks."
            )
        return

    if not block_pattern:
        raise OLMoConfigurationError(
            "`block_pattern` must be provided and non-empty when `block` is a dict of named blocks."
        )
    if block_overrides is not None:
        raise OLMoConfigurationError(
            "`block_overrides` is not supported when `block` is a dict of named blocks; "
            "use `block_pattern` to control per-layer block selection."
        )

    available_block_names = set(block.keys())
    missing_block_names = set(block_pattern) - available_block_names
    if missing_block_names:
        raise OLMoConfigurationError(
            "Every name in `block_pattern` must exist in `block`. "
            f"Unknown names: {missing_block_names}. Available names: {available_block_names}."
        )


def resolve_block_configs(
    n_layers: int,
    block: TransformerBlockConfig | dict[str, TransformerBlockConfig],
    block_pattern: list[str] | None = None,
    block_overrides: dict[int, TransformerBlockConfig] | None = None,
) -> list[TransformerBlockConfig]:
    """Resolve the block configuration for each layer."""
    validate_block_resolution_config(
        n_layers=n_layers,
        block=block,
        block_pattern=block_pattern,
        block_overrides=block_overrides,
    )

    block_configs: list[TransformerBlockConfig]
    if isinstance(block, dict):
        # Named-block configuration.
        assert block_pattern is not None
        assert block_overrides is None
        full_pattern = list(islice(cycle(block_pattern), n_layers))
        block_configs = [block[name] for name in full_pattern]
    else:
        # Single-block with manual override configuration.
        assert block_pattern is None
        block_configs = [block] * n_layers
        if block_overrides is not None:
            for block_idx, override in block_overrides.items():
                block_configs[block_idx] = override

    assert len(block_configs) == n_layers
    return block_configs
