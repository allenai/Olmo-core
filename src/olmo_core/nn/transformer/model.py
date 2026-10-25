import logging
from collections import defaultdict
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

import olmo_core.ops.moe as moe_ops
from olmo_core.data.utils import get_cumulative_document_lengths
from olmo_core.distributed.parallel import get_pp_mesh
from olmo_core.distributed.utils import hide_from_torch, unhide_from_torch
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention.ring import (
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.utils import get_default_device, mark_dynamic, move_to_device

from ..attention import (
    Attention,
    RingAttentionLoadBalancer,
    SequenceMixer,
)
from ..attention.chunked_mask import (
    MASK_MIX_SCHEDULES,
    build_chunk_ids_from_tokens,
    causal_example_flags,
    collapse_roles_to_causal,
    mask_mix_standard_prob,
)
from ..attention.summary_mask import ROLE_EXAMPLE_ID, build_summary_roles
from ..attention.summary_token import SummaryTokenAttention
from ..buffer_cache import BufferCache
from ..functional import l2_normalize
from ..layer_norm import LayerNormConfig
from ..lm_head import LMHeadConfig, LMLossImplementation, LMOutputWithLoss
from ..moe import MoEBase
from ..rope import RoPEBuffers, RotaryEmbeddingBase
from ..utils import selective_checkpointing_context_fn
from .block import (
    MoETransformerBlock,
    NormalizedTransformerBlock,
    TransformerBlock,
    TransformerBlockBase,
)
from .config import (
    TransformerActivationCheckpointingMode,
    TransformerBlockConfig,
    TransformerDataParallelWrappingStrategy,
    resolve_block_configs,
)
from .init import InitMethod

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType

__all__ = [
    "Transformer",
    "NormalizedTransformer",
    "MoETransformer",
    "TransformerDataParallelWrappingStrategy",
    "TransformerActivationCheckpointingMode",
]


log = logging.getLogger(__name__)


class Transformer(nn.Module):
    """
    A typical "Llama-style" transformer implementation.

    :param d_model: The model dimensionality.
    :param vocab_size: The vocab size.
    :param n_layers: The number of transformer layers/blocks.
    :param block: The block configuration. Can be a single block config or a dict of named blocks.
    :param layer_norm: The layer norm config for the final layer norm.
    :param bias: Whether to use a bias in the final linear layer.
    :param dtype: The datatype to use for the linear output layer.
    :param init_device: The device used when initializing parameters.
    :param init_seed: The seed used when initializing parameters.
    :param init_std: The standard deviation used when initializing parameters.
    :param embedding_init_std: The standard deviation used when initializing the embeddings.
    :param block_overrides: Overrides for specific blocks. Not supported if `block` is a dict of named blocks.
    :param block_pattern: The pattern of blocks to use. Required if `block` is a dict of named blocks.
    :param embed_scale: The scale factor for the embeddings.
    """

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        block: TransformerBlockConfig | dict[str, TransformerBlockConfig],
        lm_head: LMHeadConfig,
        embedding_norm: Optional[LayerNormConfig] = None,
        dtype: torch.dtype = torch.float32,
        init_method: InitMethod = InitMethod.normal,
        init_device: str = "cpu",
        init_seed: int = 0,
        init_std: float = 0.02,
        embedding_init_std: Optional[float] = None,
        block_overrides: Optional[Dict[int, TransformerBlockConfig]] = None,
        block_pattern: Optional[List[str]] = None,
        embed_scale: Optional[float] = None,
        tie_word_embeddings: bool = False,
    ):
        super().__init__()

        cache = BufferCache()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dtype = dtype
        self.embed_scale = embed_scale
        # Optional document-chunked-attention config (see ``enable_document_chunk_attention``). When
        # set, ``_prepare_inputs`` reconstructs per-token ``chunk_ids`` from the boundary tokens and
        # forwards them to every block (for :class:`DocumentLandmarkAttention`).
        self._document_chunk_attention: Optional[Dict[str, Any]] = None
        # Soft-token document pooling ("B1"): set by ``enable_pooled_soft_tokens``.
        self._pooled_soft_tokens: Optional[Dict[str, Any]] = None
        self.pooled_projector: Optional[nn.Module] = None
        self._pooled_keep_holder: Optional[Any] = None
        # Role-gated FFN ("flexible-compute FFN"): set by ``enable_role_gated_ffn``.
        self._role_gated_ffn: Optional[Dict[str, Any]] = None
        # Nested-width FFN mixture (learned router): set by ``enable_nested_ffn_moe``.
        self._nested_ffn_moe: Optional[Dict[str, Any]] = None
        # Random per-token FFN dropping for drop-CPT: set by ``enable_ffn_token_drop``.
        self._ffn_token_drop: Optional[Dict[str, Any]] = None
        # Per-layer KV-cache allocation router: set by ``enable_kv_route``.
        self._kv_route: Optional[Dict[str, Any]] = None
        # Joint FFN+attention budget over both routers: set by ``joint_budget.install_joint_budget``.
        self._joint_budget: Optional[Dict[str, Any]] = None
        # Per-token block skipping (mixture-of-depths router): set by ``enable_block_skip``.
        self._block_skip: Optional[Dict[str, Any]] = None
        # Train-time whole-block skipping (fixed set / LayerDrop): set by ``enable_layer_drop``.
        self._layer_drop: Optional[Any] = None
        # Token exit (input-drop keep rule, dropped tokens leave at a layer): ``enable_token_exit``.
        self._token_exit: Optional[Any] = None
        # Deliver router budget gradients through each block's output instead of a separate loss
        # term (olmo_core.nn.budget_attach): required under activation checkpointing, where the
        # separate term forces every block to be recomputed at the start of backward.
        self.budget_attach: bool = False
        # Optional summary-token-attention config (see ``enable_summary_token_attention``). When set,
        # ``_prepare_inputs`` derives per-token ``summary_roles`` and the per-example
        # ``causal_example`` arm and forwards both to every block (for
        # :class:`~olmo_core.nn.attention.summary_token.SummaryTokenAttention`).
        self._summary_token_attention: Optional[Dict[str, Any]] = None
        # Which arm of the mask mixture to serve at INFERENCE; see ``set_summary_eval_mask_mode``.
        self._summary_eval_mask_mode: str = "causal"

        self.embeddings = nn.Embedding(vocab_size, d_model, dtype=dtype, device=init_device)
        self.embedding_norm = (
            None
            if embedding_norm is None
            else embedding_norm.build(
                d_model,
                init_device=init_device,
            )
        )

        block_configs: List[TransformerBlockConfig] = resolve_block_configs(
            n_layers=n_layers,
            block=block,
            block_pattern=block_pattern,
            block_overrides=block_overrides,
        )

        self.blocks = nn.ModuleDict()
        for block_idx in range(n_layers):
            self.blocks[str(block_idx)] = self._validate_block(
                block_configs[block_idx].build(
                    d_model=d_model,
                    block_idx=block_idx,
                    n_layers=n_layers,
                    init_device=init_device,
                    cache=cache,
                )
            )
        self.lm_head = lm_head.build(
            d_model=d_model, vocab_size=vocab_size, init_device=init_device
        )

        self.tie_word_embeddings = tie_word_embeddings
        if tie_word_embeddings:
            self._tie_weights()

        self.init_device = init_device
        self.init_method = InitMethod(init_method)
        self.init_seed = init_seed
        self.init_std = init_std
        self.embedding_init_std = embedding_init_std

        self._cache = cache
        self._pp_enabled = False
        self._pp_group_size = 1
        self._fp8_enabled = False
        self._precompute_float8_dynamic_scale_for_fsdp = False
        self._compile_enabled = False
        self._device: Optional[torch.device] = None
        self._cp_load_balancer: Optional[RingAttentionLoadBalancer] = None
        self._tp_enabled = False
        self._tp_mesh: Optional[DeviceMesh] = None
        self._fsdp_enabled = False

        # Cache the value of these properties up-front in case the parameters are removed
        # later, like for pipeline parallelism.
        self.num_params
        self.num_non_embedding_params

    def _tie_weights(self) -> None:
        if self.embeddings is None or self.lm_head is None:
            raise OLMoConfigurationError(
                "Cannot tie word embeddings without both embeddings and an LM head"
            )
        if self.lm_head.w_out.bias is not None:
            raise OLMoConfigurationError("Cannot tie word embeddings when the LM head uses a bias")
        self.lm_head.w_out.weight = self.embeddings.weight

    def _validate_block(self, block: TransformerBlockBase) -> TransformerBlockBase:
        return block

    def enable_document_chunk_attention(
        self,
        doc_start_id: int,
        doc_end_id: int,
        eos_id: int,
        mode: str = "chunked",
        pad_id: Optional[int] = None,
        standard_mix_prob: float = 0.0,
        mix_start_p: float = 0.0,
        mix_end_p: float = 0.0,
        mix_total_forwards: int = 0,
        mix_seed: int = 42,
        mix_log_interval: int = 500,
    ) -> None:
        """
        Enable runtime ``chunk_ids`` reconstruction for document-chunked landmark attention.

        When enabled, :meth:`forward` reconstructs a per-token ``chunk_id`` role tensor from the
        ``<|doc_start|>`` / ``<|doc_end|>`` boundary tokens (and the EOS pad terminator) on every
        step and forwards it to each block, so :class:`~olmo_core.nn.attention.DocumentLandmarkAttention`
        can build its chunked mask. All attention layers must accept a ``chunk_ids`` keyword (i.e. the
        model is uniformly ``document_landmark``); mixing with other attention types via this path is
        not supported. Context parallelism with chunked attention is not yet supported.

        **Mask mixing** (optional): during *training*, each forward independently collapses a random
        subset of examples from the chunked mask to plain causal (full) attention, by setting their
        roles to all-FREE (see :func:`~olmo_core.nn.attention.chunked_mask.collapse_roles_to_causal`).
        The per-example probability ``p`` is either static (``standard_mix_prob``) or a linear
        curriculum (``mix_start_p -> mix_end_p`` over ``mix_total_forwards`` forwards). ``p == 0`` (all
        defaults) is a no-op -- pure-chunked training stays bit-identical. Static and curriculum mixing
        are mutually exclusive.

        :param doc_start_id: The ``<|doc_start|>`` token id.
        :param doc_end_id: The ``<|doc_end|>`` token id.
        :param eos_id: The EOS / document terminator (everything after the first one is padding).
        :param mode: ``"chunked"`` (no SINK) or ``"modified_swa"`` (mark the instruction prefix SINK).
        :param pad_id: Optional dedicated interior-padding id (window fill for the landmark variant);
            when set, those positions reconstruct to ``PAD`` (non-attendable) rather than ``FREE``.
        :param standard_mix_prob: Static per-example mask-mix probability (constant every forward).
        :param mix_start_p: Curriculum start probability (at forward 0).
        :param mix_end_p: Curriculum end probability (at ``mix_total_forwards``).
        :param mix_total_forwards: Number of forwards over which the curriculum anneals linearly.
        :param mix_seed: Base seed for the deterministic per-(forward, example) mix coin.
        :param mix_log_interval: Log a ``[curriculum]`` line (current ``p`` + cumulative collapse
            count) every this many training forwards.

        :raises OLMoConfigurationError: If static and curriculum mixing are both requested, or a
            curriculum is requested without ``mix_total_forwards > 0``.
        """
        curriculum = mix_start_p > 0.0 or mix_end_p > 0.0
        if standard_mix_prob > 0.0 and curriculum:
            raise OLMoConfigurationError(
                "Mask mixing: 'standard_mix_prob' (static) is mutually exclusive with the curriculum "
                "('mix_start_p' / 'mix_end_p'); set only one."
            )
        if curriculum and mix_total_forwards <= 0:
            raise OLMoConfigurationError(
                "Mask-mix curriculum ('mix_start_p' / 'mix_end_p') requires 'mix_total_forwards' > 0 "
                "(the number of forwards over which p anneals)."
            )
        mix: Optional[Dict[str, Any]] = None
        if standard_mix_prob > 0.0 or curriculum:
            mix = {
                "standard_mix_prob": float(standard_mix_prob),
                "mix_start_p": float(mix_start_p),
                "mix_end_p": float(mix_end_p),
                "mix_total_forwards": int(mix_total_forwards),
                "mix_seed": int(mix_seed),
                "log_interval": max(1, int(mix_log_interval)),
                "forward_idx": 0,
                "n_collapsed": 0,
            }
        self._document_chunk_attention = {
            "doc_start_id": int(doc_start_id),
            "doc_end_id": int(doc_end_id),
            "eos_id": int(eos_id),
            "mode": mode,
            "pad_id": None if pad_id is None else int(pad_id),
            "mix": mix,
        }

    def enable_pooled_soft_tokens(
        self,
        doc_start_id: int,
        doc_end_id: int,
        eos_id: int,
        *,
        placeholder_id: int,
        keep_prob: float = 0.1,
        keep_seed: int = 42,
        projector_hidden: Optional[int] = None,
        aux_match_weight: float = 0.0,
        aux_queries: int = 16,
        aux_max_shadows: int = 8,
        detach_soft_kv: bool = False,
        len_bias: bool = False,
        len_bias_scale: float = 1.0,
        len_bias_extra: float = 0.0,
        distill_prob: float = 0.0,
        distill_weight: float = 1.0,
        distill_layer_stride: int = 4,
        oracle_cache: Optional[Any] = None,
        header_stop_id: Optional[int] = None,
        header_stop_count: int = 1,
        header_cap: int = 32,
        header_extra_tokens: int = 0,
        keep_token_rule: str = "none",
        keep_token_k: float = 0,
        keep_token_weights: Optional[Dict[str, float]] = None,
        keep_token_sd: Optional[Dict[str, float]] = None,
        keep_token_tables: Optional[Any] = None,
        keep_token_log_every: int = 50,
        detach_soft_gdn: bool = True,
        drop_slots: bool = False,
        mix_start_p: float = 0.0,
        mix_end_p: float = 0.0,
        mix_total_calls: int = 0,
        slot_mode: str = "mean",
        slot_stop_ids: Optional[Sequence[int]] = None,
        slot_centroid: str = "row",
        cat_keep: Optional[Dict[str, Any]] = None,
        pad_multiple: int = 1,
        span_len: int = 12,
        kept_doc_k: Optional[float] = None,
        keep_token_min: int = 0,
        positional_block_size: int = 0,
        keep_recent: int = 0,
        keep_ramp: Optional[Tuple[float, float]] = None,
        online_router: Optional[Any] = None,
    ) -> None:
        """
        Enable train-time soft-token document pooling ("B1"; see :mod:`olmo_core.nn.pooled_soft_token`).

        During **training** forwards, each context document (identified from the
        ``<|doc_start|>``/``<|doc_end|>`` markers) is either kept (real tokens) or replaced by a
        single soft token -- the :class:`~olmo_core.nn.pooled_soft_token.PooledDocProjector` output
        on the document's mean input embedding -- at the document's center position, with original
        ``position_ids`` threaded to RoPE. The keep set is gold + random negatives via
        :func:`~olmo_core.nn.attention.pooled_doc_kv.install_pooled_doc_keep`, or a seeded random
        ``keep_prob`` fraction as fallback. Eval/generation forwards are untouched (full attention
        over real tokens -- the transfer condition), as are rows without markers.

        Creates the ``pooled_projector`` submodule -- call this BEFORE building the optimizer, and
        call ``pooled_projector.reset_parameters()`` after loading a base checkpoint without
        projector keys. Not supported together with context/pipeline parallelism or
        ``enable_document_chunk_attention``.

        :param placeholder_id: Token id emitted at soft slots (embedding is overwritten; use a
            repaired reserved id, e.g. the landmark id).
        :param header_stop_id: When given, each document's header -- the tokens after
            ``<|doc_start|>`` through the ``header_stop_count``-th occurrence of this id (at most
            ``header_cap`` tokens) -- stays REAL and only the body is pooled
            (:func:`~olmo_core.nn.attention.chunked_mask.mark_doc_headers_free`). The eval-side
            construction that reproduces full attention on contradiction (``":"``, count 1) and
            oolong (``":"``, count 3); see records/pooled-doc-kv-attention.md (2026-09-08).
        :param header_extra_tokens: ALSO keep the next ``header_extra_tokens`` body tokens (the
            ones right after the header) real, so a pooled doc's compacted row becomes
            ``<header><first header_extra_tokens body tokens><SLOT>``. ``0`` (default) is
            bit-identical to header-only behaviour. Works even without ``header_stop_id`` (the
            "header" is then zero-length, so the extra tokens are simply the document's first
            ``header_extra_tokens`` tokens after ``<|doc_start|>``). If the document's remaining
            body has fewer than ``header_extra_tokens`` tokens, the WHOLE document -- including
            ``<|doc_end|>`` -- is freed (real), so it produces no soft-token slot at all. Counted
            by position, not distance, so it is never truncated by ``header_cap`` -- no need to
            enlarge that when raising this. See
            :func:`~olmo_core.nn.attention.chunked_mask.mark_doc_headers_free`.
        :param keep_token_rule: WHICH body tokens of a pooled document stay real.

            * ``"none"`` (default) -- no per-token keeping; bit-identical to before this existed.
            * ``"first"`` -- the first ``keep_token_k`` body tokens, i.e. exactly
              ``header_extra_tokens=keep_token_k`` (it is implemented as that call, so the two are
              bit-identical by construction). Mutually exclusive with ``header_extra_tokens``.
            * ``"first_last"`` -- the first ``k // 2`` and last ``k - k // 2`` body tokens
              (:func:`~olmo_core.nn.pooled_soft_token.first_last_scores`). Same cost as ``"first"``,
              and it beats it on every rung of the frozen-model probe (ΔCE +0.044 vs +0.049 and
              R@gold 0.927 vs 0.818 at 2k, genF1 0.800 vs 0.787 at 32k --
              ``records/outlier-realtoken-parity-probe.md``). Needs no feature tables.
            * ``"rule"`` -- the ``keep_token_k`` highest-scoring body tokens PER DOCUMENT under the
              cheap feature rule of
              :func:`~olmo_core.nn.pooled_soft_token.keep_token_scores` (idf, relative position,
              first-sentence, capitalisation, digit, piece length), kept at their original
              positions and so non-contiguous
              (:func:`~olmo_core.nn.attention.chunked_mask.mark_doc_topk_tokens_free`). Ties go to
              the earlier position; a document with ``<= k`` body tokens is kept whole. Needs
              ``keep_token_tables``.

            The eval-side probe this ports says the *choice* is the whole effect: at k=8 on outlier
            the same budget spent at random is WORSE than keeping nothing (CE 0.558 vs 0.470),
            ``first`` reaches 0.263, and this rule reaches 0.071 -- the gradient oracle's 0.072
            (``records/outlier-saliency-preview-probe.md`` 5d).
        :param keep_token_k: Tokens kept real per pooled document (``0`` = rule off).
        :param keep_token_weights: Feature -> weight for ``"rule"``
            (:data:`~olmo_core.nn.pooled_soft_token.DEFAULT_KEEP_TOKEN_WEIGHTS` when ``None``).
        :param keep_token_sd: Optional per-feature standard deviations the weights are divided by
            (for a ridge fitted on standardised features).
        :param keep_token_tables: :class:`~olmo_core.nn.pooled_soft_token.KeepTokenTables` with the
            vocab-sized idf / piece tables. Required for ``"rule"``.
        :param keep_token_log_every: Log the realised keep-token cost (mean real tokens per pooled
            document, real-token fraction) every N compactions; the first 5 are always logged.
        :param mix_start_p: **Compression-mixing curriculum.** Probability that a row trains
            UNCOMPRESSED (no pooling at all), annealed linearly to ``mix_end_p`` over
            ``mix_total_calls`` training forwards
            (:func:`~olmo_core.nn.attention.pooled_doc_kv.anneal_p_full`). This is the GOLD-BLIND
            arm of the curriculum: it applies to the seeded ``keep_prob`` fallback draw. When the
            :func:`~olmo_core.nn.attention.pooled_doc_kv.install_pooled_doc_keep` hook is installed
            the hook's own curriculum is used instead and these are ignored (the hook sees gold, so
            it must make the same decision). ``0.0`` = no curriculum, the historical behaviour.
        :param mix_end_p: End point of the anneal (see ``mix_start_p``).
        :param mix_total_calls: Forwards over which to anneal; ``0`` holds ``mix_start_p`` constant.
        :param slot_mode: How the pooled slot's feature vector is built from the document's input
            embeddings (:func:`~olmo_core.nn.pooled_soft_token.apply_slot_mode`).

            * ``"mean"`` (default, the historical behaviour) -- the plain mean.
            * ``"cmean"`` -- the mean over CONTENT tokens only, i.e. dropping ``slot_stop_ids``
              (punctuation/whitespace, the markers/pad, and the top-K most frequent corpus ids).
            * ``"cent_cmean"`` -- ``cmean`` minus the row centroid, rescaled to the row's mean
              real-token embedding norm.

            A document's mean input embedding is ~94% shared common-word mass, which is why the
            plain-mean slot is unreadable for outlier (every document sits at cosine 0.93-0.94
            from the corpus centroid, gold and non-gold alike). Removing that mass is free and
            lifts the oracle readout 3.2x/4.5x over the ``k/n`` floor at 2k/8k -- see
            ``records/outlier-richer-slot-probe.md``. This has to happen in the slot CONSTRUCTION,
            not in training: with ``detach_soft_kv`` (the default) the projector gets no LM
            gradient on pooled slots, so the slot stays whatever this makes it for the whole run.
        :param slot_stop_ids: The stop set for ``cmean``/``cent_cmean``, built once at startup by
            :func:`~olmo_core.nn.pooled_soft_token.build_slot_stop_ids`. Required (and only used)
            when ``slot_mode != "mean"``.
        :param slot_centroid: Which centre ``cent_cmean`` subtracts. Only ``"row"`` (the mean over
            all of that row's document content tokens, computed online, no extra pass) is
            implemented; the eval-side probe used the corpus centroid, which a row of 14-56
            documents from the same corpus estimates.
        :param drop_slots: Emit NO slot token for pooled documents (their tokens simply vanish
            from the compacted row; kept tokens keep their original positions). Slot-less
            compaction has no slot K/V or recurrent write to detach, so ``detach_soft_kv`` /
            ``detach_soft_gdn`` become no-ops.
        :param detach_soft_gdn: With ``detach_soft_kv``, ALSO sever the backward through the
            slots' recurrent-mixer write channels (``k``/``v`` before the conv, ``beta``, ``g``)
            on GatedDeltaNet layers, so a slot influences no parameter's gradient anywhere: its
            attention K/V and GDN writes are detached, and its own query/FFN path is then a dead
            end. ``False`` = the pre-2026-09-08 behaviour (GDN co-adaptation through slots).
        :param span_len: ``keep_token_rule="spans"``: length of the random phrase-sized spans.
        :param kept_doc_k: Two-tier keep. When set, documents in the keep set (gold + random decoys)
            are NOT kept whole: they get the keep-token rule at this larger ``k`` (tier A) while every
            other document gets ``keep_token_k`` (tier B). ``None`` keeps them whole (historical).
        :param keep_token_min: Floor for a fractional ``keep_token_k``: every document keeps at least
            this many body tokens (top by the same scores), i.e. ``max(keep_token_min, k * |body|)``.
            Short documents stay (nearly) whole; long ones keep the fraction. 0 = off.
        :param pad_multiple: Round the compacted width up to a multiple of this. The width is
            data-dependent and FLA's GDN kernels autotune at every new one (~25-50 s); ``1`` keeps
            the exact width.
        :param positional_block_size: ``> 0`` = **positional-block mode** for MARKER-FREE rows
            (plain-text pretraining windows): a row's "documents" are consecutive blocks of this
            many positions (a leading EOS at position 0 is kept real and not in any block; the
            last partial block is a block too; see
            :func:`~olmo_core.nn.attention.chunked_mask.build_positional_block_chunk_ids`)
            instead of ``doc_start_id``/``doc_end_id`` spans, which are then unused. Every keep
            rule applies unchanged per block (``keep_prob`` whole blocks, ``keep_token_rule`` /
            ``keep_token_k`` per block, ``header_extra_tokens`` = the first N tokens of every
            block). The random whole-block draw is salted with a hash of the row's tokens, since
            the block layout alone is the same for every row. Needs the gold-blind path (no
            :func:`~olmo_core.nn.attention.pooled_doc_kv.install_pooled_doc_keep` hook); refuses
            ``header_stop_id`` (no header to anchor) and paired distillation. ``0`` (default) =
            marker documents, bit-identical to before.
        :param keep_recent: Positional-block mode only. ``> 0`` = the last ``keep_recent`` content
            tokens of every row are always kept (dense recent context); the keep rule applies to
            the tokens before them, so a block that straddles the boundary is ruled on its
            earlier part only. ``0`` (default) = off.
        :param keep_ramp: Positional-block mode only. ``(a, b)`` = the per-block keep fraction
            ramps linearly over a row's blocks, from ``a`` on the first block to ``b`` on the last
            (far context compressed hardest when ``a < b``). It replaces ``keep_token_k`` when a
            per-block token rule is active (``first_last`` / ``first`` / ``rule`` / ``spans``),
            else ``keep_prob`` (whole-block draw). ``None`` (default) = the scalar fraction.
        """
        if self._document_chunk_attention is not None:
            raise OLMoConfigurationError(
                "enable_pooled_soft_tokens is mutually exclusive with "
                "enable_document_chunk_attention (the compacted sequence is plain causal)."
            )
        from ..pooled_soft_token import KEEP_TOKEN_RULES, SLOT_MODES, PooledDocProjector

        if keep_token_rule not in KEEP_TOKEN_RULES:
            raise OLMoConfigurationError(
                f"unknown keep_token_rule {keep_token_rule!r} (expected one of {KEEP_TOKEN_RULES})"
            )
        if keep_token_rule != "none":
            if keep_token_k <= 0:
                raise OLMoConfigurationError(
                    f"keep_token_rule={keep_token_rule!r} needs keep_token_k > 0"
                )
            # first_last / rule on top of a fixed real prefix is the devloss grid's gold_fl20p8
            # (the prefix keeps the doc id; the K budget applies to the rest of the body). Only
            # 'first' duplicates the prefix.
            if header_extra_tokens > 0 and keep_token_rule == "first":
                raise OLMoConfigurationError(
                    "keep_token_rule='first' IS header_extra_tokens; set only one (first_last / "
                    "rule may be combined with a header_extra_tokens prefix)."
                )
        if (keep_token_rule == "online") != (online_router is not None):
            raise OLMoConfigurationError(
                "keep_token_rule='online' and online_router (an "
                "olmo_core.nn.online_token_router.OnlineTokenRouter) go together"
            )
        # learned online keep scorer (keep_token_rule="online"); see olmo_core.nn.online_token_router
        self._online_token_router = online_router
        if online_router is not None:
            # router overhead FLOPs (per rank, this step), read and reset by FlopMeterCallback
            self._flop_extra_actual = 0.0
        if keep_token_rule == "rule" and keep_token_tables is None:
            raise OLMoConfigurationError(
                "keep_token_rule='rule' needs keep_token_tables (build them with "
                "olmo_core.nn.pooled_soft_token.build_token_idf / build_token_piece_tables)"
            )
        if slot_mode not in SLOT_MODES:
            raise OLMoConfigurationError(
                f"unknown slot_mode {slot_mode!r} (expected one of {SLOT_MODES})"
            )
        if positional_block_size < 0:
            raise OLMoConfigurationError("positional_block_size must be >= 0")
        if (keep_recent > 0 or keep_ramp is not None) and positional_block_size <= 0:
            raise OLMoConfigurationError(
                "keep_recent / keep_ramp are position-aware rules of positional-block mode; "
                "set positional_block_size > 0"
            )
        if keep_recent < 0:
            raise OLMoConfigurationError("keep_recent must be >= 0")
        if keep_ramp is not None:
            keep_ramp = (float(keep_ramp[0]), float(keep_ramp[1]))
            if not all(0.0 <= v <= 1.0 for v in keep_ramp):
                raise OLMoConfigurationError(f"keep_ramp fractions must be in [0, 1], got {keep_ramp}")
            if kept_doc_k is not None or cat_keep is not None:
                raise OLMoConfigurationError("keep_ramp does not combine with kept_doc_k / cat_keep")
        if positional_block_size > 0:
            bad_pos = [
                name
                for name, on in (
                    ("header_stop_id (positional blocks have no header)", header_stop_id is not None),
                    ("paired distillation (distill_prob > 0)", distill_prob > 0.0),
                    ("oracle_cache", oracle_cache is not None),
                )
                if on
            ]
            if bad_pos:
                raise OLMoConfigurationError(
                    f"positional_block_size={positional_block_size} does not support: "
                    f"{', '.join(bad_pos)}"
                )
        if slot_mode != "mean" and not slot_stop_ids:
            raise OLMoConfigurationError(
                f"slot_mode={slot_mode!r} needs a non-empty slot_stop_ids (build it with "
                "olmo_core.nn.pooled_soft_token.build_slot_stop_ids)"
            )

        emb_weight = self.embeddings.weight  # type: ignore[union-attr]
        self.pooled_projector = PooledDocProjector(
            self.d_model,
            hidden=projector_hidden,
            dtype=emb_weight.dtype,
            init_device="meta" if emb_weight.device.type == "meta" else str(emb_weight.device),
        )
        self._pooled_soft_tokens = {
            "doc_start_id": int(doc_start_id),
            "doc_end_id": int(doc_end_id),
            "eos_id": int(eos_id),
            "placeholder_id": int(placeholder_id),
            "keep_prob": float(keep_prob),
            "keep_seed": int(keep_seed),
            # Compression-mixing curriculum for the GOLD-BLIND keep draw (see anneal_p_full). The
            # gold-aware hook carries its own copy; ``mix_calls`` counts training compactions on
            # this rank and is the anneal's clock.
            "mix_start_p": float(mix_start_p),
            "mix_end_p": float(mix_end_p),
            "mix_total_calls": int(mix_total_calls),
            "mix_calls": 0,
            "mix_debug_calls": 12,
            # Aux attention-contribution matching (see pooled_soft_token.aux_matching_loss): train
            # the projector ONLINE so a soft token's per-layer KV reproduces its doc's real
            # attention behavior, using the keep set as the supervision source. > 0 enables
            # shadows + the position-causal masked attention path.
            "aux_match_weight": float(aux_match_weight),
            "aux_queries": int(aux_queries),
            "aux_max_shadows": int(aux_max_shadows),
            # Treat pooled slots as STATIC KV: sever the LM loss's backward through the slot
            # columns' K/V at EVERY layer (and their input injection), so the main model receives
            # task gradient only through real tokens and cannot co-invent a private "summary
            # language" (the co-drift channel). The projector then trains ONLY via the aux
            # shadow objective (frozen if aux_match_weight == 0).
            "detach_soft_kv": bool(detach_soft_kv),
            # +log(doc_len) on every pooled slot's logit (pooled_soft_token.add_soft_len_bias):
            # restores a diffuse document's softmax MASS; needs the additive-bias SDPA path.
            "len_bias": bool(len_bias),
            "len_bias_scale": float(len_bias_scale),
            "len_bias_extra": float(len_bias_extra),
            # Paired consistency distillation: with probability distill_prob a training forward
            # runs BOTH the full pass (LM gradient -> protects the full-attention pathway from
            # co-drift) and the compressed pass, matching the student's hidden states at the
            # divergence layers (probe: L16+) to the DETACHED teacher at answer positions.
            # The coin is seeded on a shared forward counter so every FSDP rank takes the same
            # branch (asymmetric double-forwards would desynchronize the collectives).
            "distill_prob": float(distill_prob),
            "distill_weight": float(distill_weight),
            "distill_layers": sorted(
                set(range(int(self.n_layers * 0.4), self.n_layers, max(1, distill_layer_stride)))
                | {self.n_layers - 1}
            ),
            "_distill_counter": 0,
            # Oracle slot cache (olmo_core.nn.oracle_slot.OracleSlotCache): when set, pooled
            # slots' per-layer K/V are OVERRIDDEN with precomputed oracle log-mass slots instead
            # of whatever the network computes from the projected soft token -- the
            # maximal-fidelity static slot. Docs missing from the cache fall back to the
            # projector path.
            "oracle_cache": oracle_cache,
            "_oracle_stats": {"hits": 0, "misses": 0, "calls": 0},
            # Header-real pooling: the question's exact-match tokens (ids, dates) stay real.
            "header_stop_id": None if header_stop_id is None else int(header_stop_id),
            "header_stop_count": int(header_stop_count),
            "header_cap": int(header_cap),
            "header_extra_tokens": int(header_extra_tokens),
            # Keep-token rule: WHICH body tokens of a pooled doc stay real (none/first/rule). The
            # "first" branch is literally mark_doc_headers_free(extra_tokens=k), so it is
            # bit-identical to --st-header-extra-tokens; "rule" adds a non-contiguous per-document
            # top-k over cheap token features (pooled_soft_token.keep_token_scores).
            "keep_token_rule": str(keep_token_rule),
            "keep_token_k": float(keep_token_k),
            "keep_token_weights": (
                None if keep_token_weights is None else dict(keep_token_weights)
            ),
            "keep_token_sd": None if keep_token_sd is None else dict(keep_token_sd),
            "keep_token_tables": keep_token_tables,
            "keep_token_log_every": int(keep_token_log_every),
            "_keep_token_calls": 0,
            "detach_soft_gdn": bool(detach_soft_gdn),
            # Slot construction (pooled_soft_token.apply_slot_mode). "mean" is the historical
            # path and is bit-identical to it; the others need ``slot_stop_ids``. The vocab-sized
            # bool mask is built lazily on the first compaction (the embedding may be on meta
            # here) and cached under "slot_stop_mask".
            "slot_mode": slot_mode,
            # drop_slots=True: pooled documents get NO slot at all -- their tokens vanish from the
            # compacted row (pooled_soft_token.compact_pooled_rows). Slot-less training then has
            # nothing to detach: the detach paths see zero slot columns and are no-ops.
            "drop_slots": bool(drop_slots),
            # Round the compacted width up to a multiple of this (compact_pooled_rows): bounds the
            # distinct widths FLA's GDN kernels autotune for. 1 = exact width (historical).
            "pad_multiple": int(pad_multiple),
            "span_len": int(span_len),
            "kept_doc_k": None if kept_doc_k is None else float(kept_doc_k),
            "keep_token_min": int(keep_token_min),
            # Positional-block mode (marker-free rows): documents = fixed position blocks.
            "positional_block_size": int(positional_block_size),
            # Position-aware keep rules (positional-block mode): always-kept recent tail and a
            # linear per-block keep-fraction ramp over the row.
            "keep_recent": int(keep_recent),
            "keep_ramp": keep_ramp,
            "slot_stop_ids": None if slot_stop_ids is None else [int(t) for t in slot_stop_ids],
            "slot_stop_mask": None,
            "slot_centroid": str(slot_centroid),
            "_slot_stats": {"calls": 0, "fallback_docs": 0},
            # WHOLE-CATEGORY keep policy (olmo_core.nn.attention.doc_categories). When set, the
            # keep mask comes from clustering the DOCUMENTS' OWN SLOT VECTORS instead of from the
            # per-document keep_prob draw or the gold hook's mask, and a category is kept or
            # pooled as a unit -- never partially. Keys: mode, n_small, n_decoy,
            # decoy_max_size_mult, n_cats, cats_random, threshold_rule, log_every.
            "cat_keep": None if cat_keep is None else dict(cat_keep),
            "_cat_calls": 0,
        }
        for mod in self.modules():
            if hasattr(mod, "detach_masked_writes"):
                mod.detach_masked_writes = bool(detach_soft_kv and detach_soft_gdn)

    def enable_token_exit(self, exit_layer: int, *, log_every: int = 50) -> None:
        """
        Enable token exit (see :mod:`olmo_core.nn.token_exit`) on top of the positional-block
        input drop: the tokens the soft-token keep rule drops run blocks ``[0, exit_layer)`` on the
        full row and are compacted out (original positions kept) at the input of block
        ``exit_layer``; kept tokens run every block; dropped tokens' labels get no loss.
        ``exit_layer = 0`` is the plain input drop. Training forwards only (eval is untouched, as
        for the input drop). No new parameters.

        Call AFTER :meth:`enable_pooled_soft_tokens` (positional-block mode, ``drop_slots``).
        """
        from ..token_exit import TokenExitHolder

        cfg = self._pooled_soft_tokens
        if cfg is None or int(cfg.get("positional_block_size", 0)) <= 0:
            raise OLMoConfigurationError(
                "token exit reuses the positional-block keep rule: call enable_pooled_soft_tokens("
                "positional_block_size > 0, drop_slots=True) first"
            )
        bad = [
            name
            for name, on in (
                ("slots (set drop_slots)", not cfg.get("drop_slots", False)),
                ("aux matching", cfg["aux_match_weight"] > 0.0),
                ("len_bias", bool(cfg.get("len_bias"))),
                ("paired distillation", cfg["distill_prob"] > 0.0),
                ("oracle cache", cfg.get("oracle_cache") is not None),
            )
            if on
        ]
        if bad:
            raise OLMoConfigurationError(f"token exit does not support: {', '.join(bad)}")
        if not 0 <= int(exit_layer) < len(self.blocks):
            raise OLMoConfigurationError(
                f"token exit layer {exit_layer} outside the block stack [0, {len(self.blocks)})"
            )
        self._token_exit = TokenExitHolder(exit_layer=int(exit_layer), log_every=log_every)
        log.info("Token exit enabled: dropped tokens leave at the input of block %d", exit_layer)

    def enable_role_gated_ffn(
        self,
        doc_start_id: int,
        doc_end_id: int,
        eos_id: int,
        *,
        start_layer: int = 4,
        pad_id: Optional[int] = None,
    ) -> None:
        """
        Enable role-gated FFN compute (see :mod:`olmo_core.nn.role_gated_ffn`): context-document
        tokens (between ``<|doc_start|>``/``<|doc_end|>`` markers) skip the full FFN from
        ``start_layer`` on -- identity residual -- while free/query/answer tokens (and generated
        tokens at decode time) keep it. The gate applies to EVERY forward (train, eval, prefill),
        so training and inference see identical routing; no new parameters, so base checkpoints
        load untouched.

        Call BEFORE building the optimizer / applying data parallelism.

        :param start_layer: First gated layer (earlier layers keep the full FFN everywhere).
        """
        from ..role_gated_ffn import RoleGateHolder, install_role_gated_ffn

        holder = RoleGateHolder()
        gated = install_role_gated_ffn(self.blocks, holder, start_layer=start_layer)
        if not gated:
            raise OLMoConfigurationError("enable_role_gated_ffn gated no blocks")
        self._role_gated_ffn = {
            "doc_start_id": int(doc_start_id),
            "doc_end_id": int(doc_end_id),
            "eos_id": int(eos_id),
            "pad_id": pad_id,
            "start_layer": int(start_layer),
            "holder": holder,
        }
        log.info("Role-gated FFN enabled on %d blocks (start_layer=%d)", len(gated), start_layer)

    def enable_nested_ffn_moe(
        self,
        *,
        start_layer: int = 4,
        divisors: Sequence[float] = (1, 4, 16, 64),
        include_null: bool = True,
        target_cost: float = 0.05,
        budget_weight: float = 1.0,
        hinge_power: int = 1,
        two_sided: bool = False,
        target_anneal_calls: int = 0,
        explore_prob: float = 0.0,
        explore_anneal_calls: int = 0,
        recon_frac: float = 0.0,
        recon_weight: float = 0.0,
        entropy_weight: float = 0.0,
        seed: int = 0,
        layer_curriculum_calls: int = 0,
        width_multiple: int = 8,
        trainable_width: int = 0,
    ) -> None:
        """
        Enable the nested-width FFN mixture (see :mod:`olmo_core.nn.nested_ffn_moe`): from
        ``start_layer`` on, a learned per-token router picks one of several nested FFN widths
        (down to a zero-cost null rung), and a budget hinge loss pushes the mean per-token FFN
        cost under ``target_cost``.

        Adds a small router and per-rung gains per gated block -- NEW state-dict keys, so the base
        checkpoint must be re-saved with them (``bake_ffn_moe_into_base.py``). The router is
        initialized to select the full rung with probability ~1, so an enabled but untrained model
        reproduces its base exactly.

        Call BEFORE building the optimizer / applying data parallelism.

        :param start_layer: First routed layer.
        :param divisors: Cost divisors for the rungs, e.g. ``(1, 4, 16, 64)``.
        :param include_null: Append a zero-compute null rung.
        :param target_cost: Mean per-token FFN cost the budget hinge allows for free.
        :param budget_weight: Weight of the budget hinge.
        :param recon_frac: Fraction of tokens carrying a local full-FFN reconstruction target.
        :param seed: Base seed for the exploration draws (kept deterministic per forward so
            activation-checkpoint recompute reproduces the routing).
        :param layer_curriculum_calls: If > 0, routing opens from the last layer downward to
            ``start_layer`` linearly over this many forwards (see
            :meth:`NestedFFNHolder.current_min_layer`).
        :param width_multiple: Rung widths are floored to a multiple of this (minimum = this).
            ``1`` allows a single-hidden-unit rung, e.g. divisor 9728 on Qwen3-4B.

        :raises OLMoConfigurationError: If no blocks were routed.
        """
        from ..nested_ffn_moe import (
            NestedFFNHolder,
            install_nested_ffn_moe,
            resolve_rung_widths,
        )

        first_block = next(iter(self.blocks.values()))
        hidden_size = first_block.feed_forward.w1.out_features  # type: ignore[union-attr]
        widths, costs = resolve_rung_widths(
            hidden_size, divisors, include_null=include_null, multiple_of=width_multiple
        )
        holder = NestedFFNHolder(
            costs,
            target_cost=target_cost,
            budget_weight=budget_weight,
            hinge_power=hinge_power,
            two_sided=two_sided,
            target_anneal_calls=target_anneal_calls,
            explore_prob=explore_prob,
            explore_anneal_calls=explore_anneal_calls,
            recon_frac=recon_frac,
            recon_weight=recon_weight,
            entropy_weight=entropy_weight,
            seed=seed,
            start_layer=start_layer,
            n_layers=len(self.blocks),
            layer_curriculum_calls=layer_curriculum_calls,
        )
        routed = install_nested_ffn_moe(
            self.blocks,
            holder,
            start_layer=start_layer,
            widths=widths,
            costs=costs,
            trainable_width=trainable_width,
        )
        if not routed:
            raise OLMoConfigurationError("enable_nested_ffn_moe routed no blocks")
        self._nested_ffn_moe = {
            "start_layer": int(start_layer),
            "widths": widths,
            "costs": costs,
            "holder": holder,
        }
        log.info(
            "Nested-FFN MoE enabled on %d blocks (start_layer=%d) rungs=%s costs=%s target=%.4f",
            len(routed),
            start_layer,
            widths,
            [round(c, 5) for c in costs],
            target_cost,
        )

    def enable_ffn_token_drop(
        self,
        *,
        max_rate: float,
        layer_prob: float = 0.0,
        start_layer: int = 1,
        seed: int = 0,
        rank: int = 0,
    ) -> None:
        """
        Enable random per-token FFN dropping during training (see
        :mod:`olmo_core.nn.ffn_token_drop`): each token skips a block's FFN (identity residual)
        with a per-row rate ``r ~ U[0, max_rate]``, and each (row, layer) is dropped whole with
        probability ``layer_prob``. Training-only; adds no parameters, so the checkpoint stays a
        plain dense checkpoint.

        Call BEFORE applying data parallelism.

        :param max_rate: Upper end of the per-row token drop rate.
        :param layer_prob: Probability of dropping a layer's FFN for a whole row.
        :param start_layer: First layer that may drop.
        :param seed: Base seed for the draws.
        :param rank: Data-parallel rank (mixed into the seed).

        :raises OLMoConfigurationError: If no blocks were patched.
        """
        from ..ffn_token_drop import FFNTokenDropHolder, install_ffn_token_drop

        holder = FFNTokenDropHolder(
            max_rate=max_rate, layer_prob=layer_prob, seed=seed, rank=rank
        )
        patched = install_ffn_token_drop(self.blocks, holder, start_layer=start_layer)
        if not patched:
            raise OLMoConfigurationError("enable_ffn_token_drop patched no blocks")
        self._ffn_token_drop = {
            "max_rate": float(max_rate),
            "layer_prob": float(layer_prob),
            "start_layer": int(start_layer),
            "holder": holder,
        }
        log.info(
            "FFN token drop enabled on %d blocks (start_layer=%d, max_rate=%.3f, layer_prob=%.3f)",
            len(patched),
            start_layer,
            max_rate,
            layer_prob,
        )

    def enable_kv_route(
        self,
        *,
        start_layer: int = 0,
        target: float = 0.5,
        budget_weight: float = 1.0,
        two_sided: bool = True,
        target_anneal_calls: int = 0,
        explore_prob: float = 0.0,
        explore_anneal_calls: int = 0,
        seed: int = 0,
        router_location: str = "root",
    ) -> None:
        """
        Enable learned per-layer KV-cache allocation (see :mod:`olmo_core.nn.attention.kv_route`):
        every plain full-attention layer at or after ``start_layer`` gets a per-token keep/drop
        router; dropped keys leave that layer's cache, and a budget term pulls the mean keep
        fraction (over tokens and routed layers) to ``target``.

        Adds NEW state-dict keys (``blocks.<i>.attention._kvr_router.w.*``), initialised to keep
        everything so an enabled-but-untrained model reproduces its base exactly. Call BEFORE
        building the optimizer / applying data parallelism.

        :raises OLMoConfigurationError: If no attention layer was routed.
        """
        from ..attention.kv_route import KVRouteHolder, install_kv_route

        holder = KVRouteHolder(
            target=target,
            budget_weight=budget_weight,
            two_sided=two_sided,
            target_anneal_calls=target_anneal_calls,
            explore_prob=explore_prob,
            explore_anneal_calls=explore_anneal_calls,
            seed=seed,
            start_layer=start_layer,
            n_layers=len(self.blocks),
        )
        routed = install_kv_route(
            self.blocks, holder, start_layer=start_layer, owner=self, location=router_location
        )
        if not routed:
            raise OLMoConfigurationError("enable_kv_route routed no attention layers")
        self._kv_route = {
            "start_layer": int(start_layer),
            "routed": routed,
            "holder": holder,
            "router_location": router_location,
        }
        log.info(
            "KV routing enabled on %d attention layers %s (start_layer=%d) target=%.3f",
            len(routed),
            routed,
            start_layer,
            target,
        )

    def enable_block_skip(
        self,
        *,
        start_layer: int = 0,
        target: float = 0.5,
        budget_weight: float = 1.0,
        two_sided: bool = True,
        target_anneal_calls: int = 0,
        seed: int = 0,
        compact: bool = False,
        capacity_frac: Optional[float] = None,
    ) -> None:
        """
        Enable learned per-token block skipping (see :mod:`olmo_core.nn.block_skip`): every block at
        or after ``start_layer`` gets a router deciding per token whether the block runs; skipped
        tokens pass the residual stream unchanged and are not keys in that block's attention (in a
        GatedDeltaNet block: do not write the recurrent state). Adds
        NEW state-dict keys ``bskip_routers.<i>.w.*`` (on the model root) initialised to run everything.

        :param compact: Gather the kept tokens into a shorter sequence instead of masking, so the
            skipped work is really not done. This is what makes block skipping a wall-clock lever
            rather than a FLOP-meter one -- and it changes what "skipped" means for the
            GatedDeltaNet short convolution. Read the caveat in :mod:`olmo_core.nn.block_skip`
            before enabling it.
        :param capacity_frac: Fixed per-block capacity (mixture-of-depths): run the top
            ``capacity_frac`` of tokens by router logit instead of thresholding. Makes the
            compacted shape static, which removes one host sync per block. Changes the router's
            semantics -- see :func:`olmo_core.nn.block_skip.block_skip_decide`.

        :raises OLMoConfigurationError: If no block was routed.
        """
        from ..block_skip import BlockSkipHolder, install_block_skip

        holder = BlockSkipHolder(
            target=target,
            budget_weight=budget_weight,
            two_sided=two_sided,
            target_anneal_calls=target_anneal_calls,
            seed=seed,
            start_layer=start_layer,
            n_layers=len(self.blocks),
            compact=compact,
            capacity_frac=capacity_frac,
        )
        routed = install_block_skip(self.blocks, holder, start_layer=start_layer, owner=self)
        if not routed:
            raise OLMoConfigurationError("enable_block_skip routed no blocks")
        self._block_skip = {"start_layer": int(start_layer), "routed": routed, "holder": holder}
        log.info(
            "Block skipping enabled on %d blocks (start_layer=%d) target=%.3f compact=%s",
            len(routed),
            start_layer,
            target,
            compact,
        )

    def enable_layer_drop(
        self,
        *,
        fixed: Optional[Set[int]] = None,
        p: float = 0.0,
        schedule: str = "uniform",
        exclude_first: int = 0,
        exclude_last: int = 0,
        rescale: bool = False,
        seed: int = 0,
    ) -> None:
        """
        Enable train-time whole-block skipping (see :mod:`olmo_core.nn.layer_drop`): the ``fixed``
        blocks are skipped on every training forward, and each other eligible block is skipped
        with probability ``p`` per optimizer step (drawn from ``(seed, step)``, identical on all
        ranks; the step comes from :class:`~olmo_core.train.callbacks.LayerDropCallback`). A
        skipped block's forward is not called. Eval forwards run every block.

        :raises ValueError: On an out-of-range layer, ``p`` outside ``[0, 1)``, or nothing to skip.
        """
        from ..layer_drop import LayerDropPlan

        plan = LayerDropPlan(
            n_layers=len(self.blocks),
            fixed=frozenset(fixed or ()),
            p=float(p),
            schedule=schedule,
            exclude_first=int(exclude_first),
            exclude_last=int(exclude_last),
            rescale=bool(rescale),
            seed=int(seed),
        )
        if not plan.active:
            raise OLMoConfigurationError("enable_layer_drop: no fixed layers and p == 0")
        self._layer_drop = plan
        log.info(
            "Layer drop enabled: fixed=%s p=%.3f schedule=%s eligible=%d rescale=%s "
            "(expected run frac %.3f)",
            sorted(plan.fixed),
            plan.p,
            schedule,
            len(plan.probs),
            rescale,
            plan.expected_run_frac(),
        )

    def _attach_block_budget(self, h: torch.Tensor, block_idx: int) -> torch.Tensor:
        """
        Hand this block's router expectations their budget gradient through ``h`` (see
        :mod:`olmo_core.nn.budget_attach`). Coefficients are the budget derivatives linearised
        with the previous forward's values; zero on the first forward.
        """
        from ..budget_attach import attach_budget_grads

        nffn = self._nested_ffn_moe
        kvr = self._kv_route
        bsk = self._block_skip
        jb = self._joint_budget
        terms: List[torch.Tensor] = []
        coefs: List[float] = []

        def _this_layer(holder, attr):
            vals = getattr(holder, attr)
            return [e for li, e in zip(holder._exp_layers, vals) if int(li) == block_idx]

        if jb is not None:
            if "last_cost" not in jb:
                return h
            sign = 1.0 if jb["last_cost"] > jb["last_target"] else -1.0
            lam = jb["weight"] * sign
            sh = jb["per_block"][block_idx]
            keep_prev = bsk["holder"].last_exp_per_layer.get(block_idx, 1.0) if bsk else 1.0
            ffn_prev = nffn["holder"].last_exp_per_layer.get(block_idx, 1.0) if nffn else 1.0
            kv_prev = kvr["holder"].last_exp_per_layer.get(block_idx, 1.0) if kvr else 1.0
            if nffn is not None:
                for e in _this_layer(nffn["holder"], "_exp_costs"):
                    terms.append(e)
                    coefs.append(lam * sh["ffn"] * keep_prev)
            if kvr is not None:
                for e in _this_layer(kvr["holder"], "_exp_keep"):
                    terms.append(e)
                    coefs.append(lam * sh["attn"] * keep_prev)
            if bsk is not None:
                for e in _this_layer(bsk["holder"], "_exp_keep"):
                    terms.append(e)
                    coefs.append(lam * (sh["proj"] + sh["ffn"] * ffn_prev + sh["attn"] * kv_prev))
        else:
            for cfg, attr in ((nffn, "_exp_costs"), (kvr, "_exp_keep"), (bsk, "_exp_keep")):
                if cfg is None:
                    continue
                holder = cfg["holder"]
                if holder.budget_weight <= 0 or holder.last_exp_mean is None:
                    continue
                gap = holder.last_exp_mean - holder.current_target()
                if gap <= 0 and not getattr(holder, "two_sided", True):
                    continue
                n_layers = max(1, len(holder.last_exp_per_layer))
                sign = 1.0 if gap > 0 else -1.0
                for e in _this_layer(holder, attr):
                    terms.append(e)
                    coefs.append(holder.budget_weight * sign / n_layers)
        return attach_budget_grads(h, terms, coefs)

    def _set_role_gate_mask(self, input_ids: torch.Tensor) -> None:
        """Recompute the FFN gate mask from the (possibly compacted) token stream."""
        cfg = self._role_gated_ffn
        assert cfg is not None
        holder = cfg["holder"]
        if input_ids.is_floating_point():
            holder.clear()  # PP hidden states: cannot derive roles here
            return
        chunk_ids = build_chunk_ids_from_tokens(
            input_ids,
            doc_start_id=cfg["doc_start_id"],
            doc_end_id=cfg["doc_end_id"],
            eos_id=cfg["eos_id"],
            mode="chunked",
            pad_id=cfg["pad_id"],
        )
        holder.set_from_chunk_ids(move_to_device(chunk_ids, self.device))

    def _record_example_lens(self, dense_lens: List[int], actual_lens: List[int]) -> None:
        """Per-example lengths of this forward for the FLOP meter (dense = the real example,
        actual = what ran, e.g. the compacted example). Read and reset by ``FlopMeterCallback``."""
        st = getattr(self, "_flop_example_lens", None)
        if st is None:
            st = self._flop_example_lens = {"dense": [], "actual": []}
        st["dense"].extend(dense_lens)
        st["actual"].extend(actual_lens)

    def _check_soft_token_segment_mode(self) -> None:
        """Packed-row soft tokens re-pack the compacted examples into one varlen row, which is only
        defined for slot-less compaction with no extra attention bias or paired teacher pass."""
        cfg = self._pooled_soft_tokens
        assert cfg is not None
        bad = [
            name
            for name, on in (
                ("slots (set drop_slots)", not cfg.get("drop_slots", False)),
                ("aux matching", cfg["aux_match_weight"] > 0.0),
                ("len_bias", bool(cfg.get("len_bias"))),
                ("paired distillation", cfg["distill_prob"] > 0.0),
                ("oracle cache", cfg.get("oracle_cache") is not None),
            )
            if on
        ]
        if bad:
            raise OLMoConfigurationError(
                f"pooled soft tokens on PACKED rows (doc_lens) do not support: {', '.join(bad)}"
            )

    @staticmethod
    def _positional_keep_features(
        chunk_ids: torch.Tensor, cfg: Dict[str, Any]
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Position-aware keep rules of positional-block mode (``keep_ramp`` / ``keep_recent``).

        :returns: ``(ramp_frac, chunk_ids)`` -- ``ramp_frac`` is ``(B, n_blocks)`` float32, block
            ``d`` of a row with ``n`` blocks getting ``a + (b - a) * d / (n - 1)`` (``None`` when no
            ramp is configured); ``chunk_ids`` has the row's last ``keep_recent`` content tokens
            re-labelled ``FREE`` (always kept). The ramp is laid over the blocks BEFORE the recent
            tail is freed, so a block's fraction depends only on where it sits in the row.
        """
        from ..attention.chunked_mask import FREE_CHUNK_ID, PAD_CHUNK_ID

        ramp = cfg.get("keep_ramp")
        recent = int(cfg.get("keep_recent", 0))
        ramp_frac = None
        cid = chunk_ids.to(torch.long)
        if ramp is not None:
            n_blk = int(cid.max().item()) + 1
            if n_blk > 0:
                nb = (cid.max(dim=1, keepdim=True).values + 1).clamp(min=1).to(torch.float32)
                d = torch.arange(n_blk, device=chunk_ids.device, dtype=torch.float32)[None, :]
                t = (d / (nb - 1.0).clamp(min=1.0)).clamp(max=1.0)
                ramp_frac = ramp[0] + (ramp[1] - ramp[0]) * t
        if recent > 0:
            S = chunk_ids.shape[1]
            pos = torch.arange(S, device=chunk_ids.device)[None, :].expand_as(cid)
            content = cid != PAD_CHUNK_ID
            end = torch.where(content, pos, torch.full_like(pos, -1)).max(dim=1, keepdim=True).values
            tail = (pos > end - recent) & (cid >= 0)
            chunk_ids = torch.where(tail, torch.full_like(chunk_ids, FREE_CHUNK_ID), chunk_ids)
        return ramp_frac, chunk_ids

    def _compact_pooled_soft_tokens(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        ignore_index: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[Tuple]]:
        """
        Apply soft-token compaction for this training forward. Returns
        ``(input_ids, labels, position_ids, soft_inject)`` -- unchanged inputs and ``None``s when
        the batch has no context documents. ``soft_inject = (rows, cols, mean_embeds)`` is consumed
        after the embedding lookup.
        """
        from ..attention.pooled_doc_kv import anneal_p_full, resolve_keep_docs
        from ..pooled_soft_token import apply_slot_mode, compact_pooled_rows

        cfg = self._pooled_soft_tokens
        assert cfg is not None
        if self._cp_load_balancer is not None or self._pp_enabled:
            raise OLMoConfigurationError(
                "pooled soft tokens are not supported with context/pipeline parallelism"
            )
        block = int(cfg.get("positional_block_size", 0))
        row_salt = None
        if block > 0:
            # Marker-free rows: documents are fixed position blocks. No token is a marker, so
            # every downstream helper gets ids that never match (-1) and treats each block's
            # tokens as its body.
            from ..attention.chunked_mask import build_positional_block_chunk_ids

            if self._pooled_keep_holder is not None:
                raise OLMoConfigurationError(
                    "positional_block_size needs the gold-blind keep path; a gold keep hook "
                    "(install_pooled_doc_keep) is installed"
                )
            chunk_ids = build_positional_block_chunk_ids(
                input_ids, block_size=block, eos_id=cfg["eos_id"]
            )
            doc_start_id = doc_end_id = -1
            ramp_frac, chunk_ids = self._positional_keep_features(chunk_ids, cfg)
            # The block layout is the same for every row, so the seeded whole-block draw needs a
            # per-row salt or every row keeps the same blocks: a position-weighted token hash.
            w = torch.arange(1, input_ids.shape[1] + 1, dtype=torch.int64, device=input_ids.device)
            row_salt = ((input_ids.to(torch.int64) + 1) * w).sum(dim=-1)
        else:
            ramp_frac = None
            doc_start_id, doc_end_id = cfg["doc_start_id"], cfg["doc_end_id"]
            chunk_ids = build_chunk_ids_from_tokens(
                input_ids,
                doc_start_id=doc_start_id,
                doc_end_id=doc_end_id,
                eos_id=cfg["eos_id"],
                mode="chunked",
            )
        n_docs = int(chunk_ids.max().item()) + 1
        if n_docs <= 0:
            return None
        if ramp_frac is not None:
            ramp_frac = ramp_frac[:, :n_docs]
        # Which body tokens stay REAL. "first" IS --st-header-extra-tokens (same call, so
        # bit-identical); "rule" adds a non-contiguous per-document top-k afterwards.
        keep_rule = cfg.get("keep_token_rule", "none")
        extra_tokens = int(cfg.get("header_extra_tokens", 0))
        # "first" with 0 < k < 1 is a per-document FRACTION (first ceil(k*|body|) tokens) and goes
        # through the top-k path with first_scores; integer k stays on the header call below.
        # Positional blocks have no <|doc_start|> to count from, so integer "first" goes through
        # the same top-k path (first_scores keeps the first k tokens of every block).
        frac_first = keep_rule == "first" and (
            block > 0 or 0.0 < float(cfg["keep_token_k"]) < 1.0
        )
        if keep_rule == "first" and not frac_first:
            extra_tokens = int(cfg["keep_token_k"])
        pre_free_chunk_ids = chunk_ids if keep_rule != "none" else None
        if block > 0 and extra_tokens > 0:
            # header_extra_tokens per positional block: the first N tokens of every block stay
            # real (a block of <= N tokens is kept whole).
            from ..attention.chunked_mask import FREE_CHUNK_ID

            pos = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(chunk_ids)
            lead = (input_ids[:, :1] == cfg["eos_id"]).to(pos.dtype)
            head = (chunk_ids >= 0) & ((pos - lead) % block < extra_tokens)
            chunk_ids = torch.where(head, torch.full_like(chunk_ids, FREE_CHUNK_ID), chunk_ids)
        elif cfg.get("header_stop_id") is not None or extra_tokens > 0:
            from ..attention.chunked_mask import mark_doc_headers_free

            chunk_ids = mark_doc_headers_free(
                chunk_ids,
                input_ids,
                doc_start_id=doc_start_id,
                doc_end_id=doc_end_id,
                stop_id=cfg["header_stop_id"],
                stop_count=cfg["header_stop_count"],
                extra_tokens=extra_tokens,
                cap=cfg["header_cap"],
            )
        hdr_chunk_ids, tier_scores = chunk_ids, None
        if keep_rule in ("rule", "first_last", "spans") or frac_first:
            from ..attention.chunked_mask import mark_doc_topk_tokens_free
            from ..pooled_soft_token import (
                DEFAULT_KEEP_TOKEN_WEIGHTS,
                first_last_scores,
                first_scores,
                keep_token_scores,
                span_scores,
            )

            if keep_rule == "spans":
                scores = span_scores(input_ids, span_len=int(cfg.get("span_len", 12)))
            elif frac_first:
                scores = first_scores(
                    input_ids,
                    chunk_ids,
                    doc_start_id=doc_start_id,
                    doc_end_id=doc_end_id,
                    n_docs=n_docs,
                )
            elif keep_rule == "first_last":
                scores = first_last_scores(
                    input_ids,
                    chunk_ids,
                    doc_start_id=doc_start_id,
                    doc_end_id=doc_end_id,
                    n_docs=n_docs,
                )
            else:
                scores = keep_token_scores(
                    input_ids,
                    chunk_ids,
                    tables=cfg["keep_token_tables"],
                    weights=cfg.get("keep_token_weights") or DEFAULT_KEEP_TOKEN_WEIGHTS,
                    sd=cfg.get("keep_token_sd"),
                    doc_start_id=doc_start_id,
                    doc_end_id=doc_end_id,
                    n_docs=n_docs,
                )
            tier_scores = scores
            hdr_only = chunk_ids
            chunk_ids = mark_doc_topk_tokens_free(
                chunk_ids,
                input_ids,
                scores,
                doc_start_id=doc_start_id,
                doc_end_id=doc_end_id,
                k=float(cfg["keep_token_k"]),
                n_docs=n_docs,
                k_doc_frac=ramp_frac,
            )
            ramp_frac = None  # consumed by the token rule; whole-block keep_prob stays scalar
            if int(cfg.get("keep_token_min", 0)) > 0:
                # floor: also free each document's top keep_token_min tokens by the SAME scores, so
                # the union is the top max(keep_token_min, k * |body|)
                from ..attention.chunked_mask import FREE_CHUNK_ID

                floor_ids = mark_doc_topk_tokens_free(
                    hdr_only,
                    input_ids,
                    scores,
                    doc_start_id=cfg["doc_start_id"],
                    doc_end_id=cfg["doc_end_id"],
                    k=float(cfg["keep_token_min"]),
                    n_docs=n_docs,
                )
                chunk_ids = torch.where(
                    floor_ids == FREE_CHUNK_ID, torch.full_like(chunk_ids, FREE_CHUNK_ID), chunk_ids
                )
        elif keep_rule == "online":
            # learned per-token keep mask, refit every step (olmo_core.nn.online_token_router);
            # markers always stay real, so a document dropped whole leaves an empty marker pair
            from ..attention.chunked_mask import mark_positions_free

            chunk_ids = mark_positions_free(
                chunk_ids,
                input_ids,
                self._online_token_router.select(self, input_ids, chunk_ids, labels, ignore_index),
                doc_start_id=doc_start_id,
                doc_end_id=doc_end_id,
                n_docs=n_docs,
                free_markers=True,
            )
        elif keep_rule == "custom":
            # caller-supplied per-token keep mask (saliency / attention-budget selectors that the
            # model cannot score from token features alone); see chunked_mask.mark_positions_free
            from ..attention.chunked_mask import mark_positions_free

            chunk_ids = mark_positions_free(
                chunk_ids,
                input_ids,
                cfg["keep_token_mask"],
                doc_start_id=doc_start_id,
                doc_end_id=doc_end_id,
                n_docs=n_docs,
                free_markers=cfg.get("keep_token_mask_markers", False) or False,
            )
        # Compression-mixing curriculum on the GOLD-BLIND path: with probability ``p_full`` a row
        # trains uncompressed. With the gold-aware hook installed the hook already applied it
        # (it needs the gold table), so this stays off to avoid applying it twice.
        mix_on = (cfg.get("mix_start_p", 0.0) > 0.0 or cfg.get("mix_end_p", 0.0) > 0.0) and (
            self._pooled_keep_holder is None
        )
        p_full = (
            anneal_p_full(
                cfg["mix_calls"],
                mix_start_p=cfg["mix_start_p"],
                mix_end_p=cfg["mix_end_p"],
                mix_total_calls=cfg["mix_total_calls"],
            )
            if mix_on
            else 0.0
        )
        # NB the keep set is resolved AFTER the slot vectors below, not here: a whole-category
        # policy (cfg["cat_keep"]) decides which documents to keep FROM those vectors.
        # Mean input embedding per pooled doc (from the ORIGINAL row), the projector's feature.
        emb = self.embeddings(input_ids)  # type: ignore[misc]
        B, T, D = emb.shape
        cid = chunk_ids.to(torch.long)
        is_ctx = cid >= 0
        flat = (torch.arange(B, device=emb.device)[:, None] * n_docs + cid.clamp(min=0)).reshape(
            -1
        )[is_ctx.reshape(-1)]
        sums = torch.zeros(B * n_docs, D, dtype=emb.dtype, device=emb.device).index_add(
            0, flat, emb.reshape(B * T, D)[is_ctx.reshape(-1)]
        )
        counts = (
            torch.zeros(B * n_docs, dtype=torch.float32, device=emb.device)
            .index_add(0, flat, torch.ones_like(flat, dtype=torch.float32))
            .clamp(min=1.0)
        )
        doc_means = (sums / counts.unsqueeze(-1).to(emb.dtype)).reshape(B, n_docs, D)
        if cfg.get("slot_mode", "mean") != "mean":
            # Content-only (and optionally centred) slot: the plain mean is ~94% shared
            # common-word mass, which is what makes it unreadable -- records/outlier-richer-slot-probe.md.
            mask = cfg.get("slot_stop_mask")
            if mask is None or mask.device != emb.device:
                V = int(self.embeddings.weight.shape[0])  # type: ignore[union-attr]
                mask = torch.zeros(V, dtype=torch.bool, device=emb.device)
                ids_t = torch.tensor(cfg["slot_stop_ids"], dtype=torch.long, device=emb.device)
                mask[ids_t[ids_t < V]] = True
                cfg["slot_stop_mask"] = mask
            doc_means, n_fallback = apply_slot_mode(
                emb,
                input_ids,
                chunk_ids,
                n_docs,
                doc_means,
                mode=cfg["slot_mode"],
                stop_mask=mask,
                centroid=cfg.get("slot_centroid", "row"),
            )
            st_slot = cfg["_slot_stats"]
            st_slot["calls"] += 1
            st_slot["fallback_docs"] += n_fallback
            if n_fallback and st_slot["calls"] % 100 == 1:
                log.warning(
                    "[soft-slot] %s: %d documents lost every token to the stop set so far and "
                    "fell back to the plain mean (%d compactions)",
                    cfg["slot_mode"],
                    st_slot["fallback_docs"],
                    st_slot["calls"],
                )

        cat_cfg = cfg.get("cat_keep")
        if cat_cfg is None:
            keep = resolve_keep_docs(
                chunk_ids,
                n_docs,
                holder=self._pooled_keep_holder,
                keep_prob=cfg["keep_prob"] if ramp_frac is None else ramp_frac,
                keep_seed=cfg["keep_seed"],
                mix_p_full=p_full,
                mix_call=cfg["mix_calls"] if mix_on else 0,
                row_salt=row_salt,
            )
        else:
            # Whole-category keep: cluster the documents by the SAME slot vectors the pooled slot
            # carries, then keep or pool each category as a unit. For "gold_plus_wholecats" the
            # gold-sidecar hook's mask is read as the GOLD set (the trainer installs that hook with
            # n_random=0 precisely so it carries gold and nothing else); "smallcat_keep" is
            # gold-blind and ignores the holder entirely.
            from ..attention.doc_categories import resolve_category_keep

            doc_valid = (
                torch.zeros(B, n_docs, dtype=torch.bool, device=doc_means.device)
                .reshape(-1)
                .index_fill_(0, flat, True)
                .reshape(B, n_docs)
            )
            gold_mask = None
            if cat_cfg["mode"] == "gold_plus_wholecats":
                holder = self._pooled_keep_holder
                gold_mask = None if holder is None else holder.keep_docs
                if gold_mask is None:
                    raise OLMoConfigurationError(
                        "cat_keep mode 'gold_plus_wholecats' needs the gold-sidecar keep hook "
                        "(install_pooled_doc_keep); no holder mask was set for this forward."
                    )
                gold_mask = gold_mask.to(device=doc_means.device, dtype=torch.bool)
                if gold_mask.shape[0] == 1 and B > 1:
                    gold_mask = gold_mask.expand(B, -1)
                if gold_mask.shape[1] < n_docs:
                    gold_mask = torch.cat(
                        [
                            gold_mask,
                            torch.zeros(
                                B,
                                n_docs - gold_mask.shape[1],
                                dtype=torch.bool,
                                device=doc_means.device,
                            ),
                        ],
                        dim=1,
                    )
                gold_mask = gold_mask[:, :n_docs]
            cfg["_cat_calls"] += 1
            keep, cat_stats = resolve_category_keep(
                doc_means.detach().float(),
                doc_valid,
                mode=cat_cfg["mode"],
                gold_docs=gold_mask,
                n_small=cat_cfg.get("n_small", 3),
                n_decoy=cat_cfg.get("n_decoy", 1),
                decoy_max_size_mult=cat_cfg.get("decoy_max_size_mult", 2.0),
                n_cats=cat_cfg.get("n_cats", 3),
                cats_random=cat_cfg.get("cats_random", False),
                threshold_rule=cat_cfg.get("threshold_rule", "gap"),
                seed=cfg["keep_seed"],
                call=cfg["_cat_calls"],
                doc_counts=counts.reshape(B, n_docs),
            )
            n_call = cfg["_cat_calls"]
            every = int(cat_cfg.get("log_every", 50))
            if n_call <= 5 or (every > 0 and n_call % every == 0):
                # The FLOP cost of this policy is `real_token_frac`, and it is DATA-DEPENDENT (a
                # decoy category's size varies row to row), so it has to be logged, not assumed.
                print(
                    f"[cat-keep] call#{n_call} {cat_cfg['mode']}: B={B} n_docs<={n_docs} "
                    f"cats={cat_stats['n_cats_mean']:.1f} kept_docs={cat_stats['kept_docs_mean']:.1f} "
                    f"kept_sizes={cat_stats['kept_sizes']} "
                    f"real_token_frac={cat_stats['real_token_frac']} "
                    f"decoy_skipped={cat_stats['decoy_skipped']} "
                    f"saturated_rows={cat_stats['saturated_rows']}",
                    flush=True,
                )
        if mix_on:
            cfg["mix_calls"] += 1
            calls, B_ = cfg["mix_calls"], keep.shape[0]
            if calls <= cfg["mix_debug_calls"] or calls % 100 == 0:
                # Same ``[pooled-kv] ... p_full=`` shape as the gold-aware hook, so the job-log
                # greps in debug/ds64 and records/ see the curriculum on either path.
                print(
                    f"[pooled-kv] call#{calls} gold-blind: B={B_} n_docs<={n_docs} "
                    f"keep_prob={cfg['keep_prob']} pooled_docs={int((~keep).sum().item())} "
                    f"p_full={p_full:.2f} mixed={int(keep.all(dim=1).sum().item())}",
                    flush=True,
                )

        if keep_rule == "online":
            keep = torch.zeros_like(keep)  # no document is kept whole: the router chose the tokens
        if cfg.get("kept_doc_k") is not None and tier_scores is not None:
            # Two-tier keep: the keep set (gold + decoys) is not kept whole but gets the same rule at
            # the larger kept_doc_k; every other document keeps keep_token_k (already marked above).
            from ..attention.chunked_mask import FREE_CHUNK_ID, mark_doc_topk_tokens_free

            c_a = mark_doc_topk_tokens_free(
                hdr_chunk_ids,
                input_ids,
                tier_scores,
                doc_start_id=doc_start_id,
                doc_end_id=doc_end_id,
                k=float(cfg["kept_doc_k"]),
                n_docs=n_docs,
            )
            cid0 = hdr_chunk_ids.to(torch.long)
            keep_b = keep[:, :n_docs].to(torch.bool)
            tier_a = (cid0 >= 0) & keep_b.gather(1, cid0.clamp(min=0))
            free_a = (c_a == FREE_CHUNK_ID) & tier_a
            chunk_ids = torch.where(free_a, torch.full_like(chunk_ids, FREE_CHUNK_ID), chunk_ids)
            keep = torch.zeros_like(keep)
        cb = compact_pooled_rows(
            input_ids,
            labels,
            chunk_ids,
            keep,
            placeholder_id=cfg["placeholder_id"],
            pad_token_id=cfg["eos_id"],
            ignore_index=ignore_index,
            add_shadows=cfg["aux_match_weight"] > 0.0,
            max_shadows_per_row=cfg["aux_max_shadows"],
            drop_slots=bool(cfg.get("drop_slots", False)),
            pad_multiple=1 if cfg.get("_segment_mode") else int(cfg.get("pad_multiple", 1)),
        )
        if pre_free_chunk_ids is not None:
            cfg["_keep_token_calls"] += 1
            n_call = cfg["_keep_token_calls"]
            every = int(cfg.get("keep_token_log_every", 50))
            if n_call <= 5 or (every > 0 and n_call % every == 0):
                # The realised cost of a keep-token rule is DATA-dependent (short documents are
                # kept whole, so the mean is <= k and the FLOP saving is not 1/k of anything):
                # it has to be measured, not assumed. Freed-per-document is counted against the
                # PRE-free roles, which still carry each freed token's document id.
                from ..attention.chunked_mask import PAD_CHUNK_ID

                pre_cid = pre_free_chunk_ids.to(torch.long)
                freed = (chunk_ids < 0) & (pre_cid >= 0)
                gid_full = torch.arange(B, device=input_ids.device)[
                    :, None
                ] * n_docs + pre_cid.clamp(min=0)
                per_doc = torch.zeros(
                    B * n_docs, dtype=torch.float32, device=input_ids.device
                ).index_add_(
                    0, gid_full[freed], torch.ones(int(freed.sum()), device=input_ids.device)
                )
                present = (
                    torch.zeros(B * n_docs, dtype=torch.bool, device=input_ids.device)
                    .index_fill_(0, gid_full[pre_cid >= 0], True)
                    .reshape(B, n_docs)
                )
                pooled = present & ~keep[:, :n_docs]
                n_pooled = int(pooled.sum().item())
                kept_per_doc = (
                    float(per_doc.reshape(B, n_docs)[pooled].mean().item()) if n_pooled else 0.0
                )
                n_in = int((chunk_ids != PAD_CHUNK_ID).sum().item())
                n_real = int(cb.row_lens.sum().item()) - int(cb.soft_rows.numel())
                print(
                    f"[keep-token] call#{n_call} rule={keep_rule} k={cfg['keep_token_k']}: "
                    f"B={B} pooled_docs={n_pooled} real_tokens_per_pooled_doc={kept_per_doc:.2f} "
                    f"real_token_frac={n_real / max(1, n_in):.4f}",
                    flush=True,
                )

        # Projector features for pooled slots AND (when aux is on) the shadow candidates -- both
        # are P(mean input embeds of the doc), same distribution, same module.
        inj_rows = torch.cat([cb.soft_rows, cb.shadow_rows])
        inj_cols = torch.cat([cb.soft_cols, cb.shadow_cols])
        inj_docs = torch.cat([cb.soft_docs, cb.shadow_docs])
        mean_feats = doc_means[inj_rows, inj_docs]

        # Oracle slot lookup: hash each pooled doc's token span, gather its cached per-layer
        # slots. Docs missing from the cache stay on the projector path.
        oracle_ovr = None
        cache = cfg.get("oracle_cache")
        if cache is not None and cb.soft_rows.numel() > 0:
            import numpy as np

            from ..oracle_slot import doc_hash64

            ids_np = input_ids.cpu().numpy()
            cid_np = cid.cpu().numpy()
            s_rows = cb.soft_rows.cpu().numpy()
            s_docs = cb.soft_docs.cpu().numpy()
            row_order: Dict[int, Tuple[Any, Any]] = {}
            hashes = []
            for b, d in zip(s_rows.tolist(), s_docs.tolist()):
                if b not in row_order:
                    order = np.argsort(cid_np[b], kind="stable")
                    row_order[b] = (order, cid_np[b][order])
                order, srt = row_order[b]
                span_lo = int(np.searchsorted(srt, d))
                span_hi = int(np.searchsorted(srt, d, side="right"))
                hashes.append(doc_hash64(ids_np[b][order[span_lo:span_hi]]))
            idx = cache.lookup(hashes)
            found = idx >= 0
            st = cfg["_oracle_stats"]
            st["hits"] += int(found.sum())
            st["misses"] += int((~found).sum())
            st["calls"] += 1
            if st["misses"] > 0 and st["calls"] % 100 == 1:
                log.warning(
                    "[oracle-slot] cache misses so far: %d / %d slots (missing docs fall back "
                    "to the projector)",
                    st["misses"],
                    st["hits"] + st["misses"],
                )
            if st["calls"] >= 10 and st["hits"] == 0:
                raise RuntimeError(
                    "[oracle-slot] 0 cache hits after 10 forwards -- the doc-hash scheme "
                    "almost certainly mismatches the cache builder; refusing to silently "
                    "train plain B1"
                )
            if found.any():
                sel = torch.from_numpy(np.flatnonzero(found)).to(cb.soft_rows.device)
                slots, biases = cache.gather(idx[found])
                rows = cb.soft_rows[sel]
                cols = cb.soft_cols[sel]
                oracle_ovr = {
                    "rows": rows,
                    "cols": cols,
                    "pos": cb.position_ids[rows, cols],
                    "slots": slots.to(input_ids.device, non_blocking=True),
                    "biases": biases.to(input_ids.device, non_blocking=True),
                }
        return cb, (inj_rows, inj_cols, mean_feats), oracle_ovr

    def _run_blocks(
        self,
        h: torch.Tensor,
        all_block_kwargs: Dict[str, Any],
        per_block_kwargs: Dict[int, Dict[str, Any]],
        capture_layers: Optional[Set[int]] = None,
        token_exit: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Run the block stack; optionally capture (attached) hidden states at given layer indices.

        ``token_exit`` (:mod:`olmo_core.nn.token_exit`): ``{"layer": L, "idx": (B, T') kept
        columns, "all_block_kwargs": ..., "per_block_kwargs": ...}`` -- blocks ``< L`` run on the
        full row with the FULL-row kwargs given there, ``h`` is gathered to the kept columns at the
        input of block ``L``, and the remaining blocks use the (compacted) kwargs passed in.
        """
        captured: Dict[int, torch.Tensor] = {}
        otr = getattr(self, "_online_token_router", None) if self.training else None
        otr_layer = otr.cfg.drop_layer if otr is not None else 0
        # pyramid: every stage's drop layer -> its router (one entry without stages)
        otr_stages = {r.cfg.drop_layer: r for r in otr.chain()} if otr_layer > 0 else {}
        # online router mixture-of-depths: the listed blocks run on the router's kept tokens only
        otr_mod = frozenset(otr.mod_layers) if otr_layer > 0 and getattr(otr, "mod_layers", None) else frozenset()
        otr_tap = int(getattr(otr.cfg, "target_layer", 0) or 0) if otr_layer > 0 else 0
        ld = self._layer_drop if self.training else None
        ld_skip = ld.skipped() if ld is not None else frozenset()
        # online router early_nograd: on routed forwards blocks [0, otr_layer) run forward-only
        otr_nograd = otr_layer > 0 and bool(getattr(otr, "_nograd_fwd", False))
        # the no-grad prefix ends at the drop layer, or earlier when a progressive freeze has frozen fewer blocks
        otr_nograd_end = int(getattr(otr, "_nograd_until", 0) or otr_layer) if otr_nograd else -1
        grad_was = torch.is_grad_enabled()
        if otr_nograd:
            torch.set_grad_enabled(False)
        for block_key, block in self.blocks.items():
            block_idx = int(block_key)
            if otr_nograd and block_idx == otr_nograd_end:
                torch.set_grad_enabled(grad_was)
                # a fresh leaf: the router's target hook at the split reads dL/dh at or above here
                h = h.detach().requires_grad_(grad_was)
            if block_idx in otr_stages:
                # online router, mid-stack drop: dropped tokens skip blocks [otr_layer, n_layers)
                h, all_block_kwargs = otr_stages[block_idx].mid_compact(self, h, all_block_kwargs)
            if otr_tap and block_idx == otr_tap:
                h = otr.tap_target(self, h, block_idx)  # type: ignore[union-attr]
            if block_idx in ld_skip:
                # whole-block skip (olmo_core.nn.layer_drop): identity, forward never called
                if capture_layers is not None and block_idx in capture_layers:
                    captured[block_idx] = h
                continue
            h_in = h
            block_kwargs = per_block_kwargs.get(block_idx, {})
            if token_exit is not None:
                if block_idx == token_exit["layer"]:
                    from ..token_exit import gather_kept

                    h = gather_kept(h, token_exit["idx"])
                elif block_idx < token_exit["layer"]:
                    if self.compile_enabled:
                        mark_dynamic(h, (0, 1), strict=False)
                    h = block(
                        h,
                        **token_exit["all_block_kwargs"],
                        **token_exit["per_block_kwargs"].get(block_idx, {}),
                    )
                    if capture_layers is not None and block_idx in capture_layers:
                        captured[block_idx] = h
                    continue
            if self.compile_enabled:
                mark_dynamic(h, (0, 1), strict=False)
            # Routers evaluated on the block INPUT, outside the checkpoint region (router_fn.py):
            # a router inside the region kept every layer's norm intermediates alive under FSDP2.
            extra: Dict[str, Any] = {}
            attn_mod = getattr(block, "attention", None)
            kv_cfg = getattr(attn_mod, "_kv_route", None)
            if (
                kv_cfg is not None
                and kv_cfg["holder"].enabled
                and kv_cfg.get("location", "attention") == "root"
            ):
                from ..attention.kv_route import kv_route_decide

                extra.update(kv_route_decide(self, attn_mod, h))
            if getattr(block, "_bskip", None) is not None:
                from ..block_skip import block_skip_decide, block_skip_forward

                p_skip, keep_skip = block_skip_decide(block, h)
                h = block_skip_forward(
                    block, h, {**all_block_kwargs, **block_kwargs, **extra}, p_skip, keep_skip
                )
            elif block_idx in otr_mod:
                h = otr.mod_block(block, h, {**all_block_kwargs, **block_kwargs, **extra})  # type: ignore[union-attr]
            else:
                h = block(h, **all_block_kwargs, **block_kwargs, **extra)
            if ld is not None and ld.rescale:
                scale = ld.run_scale(block_idx)
                if scale != 1.0:
                    h = h_in + (h - h_in) * scale
            if self.budget_attach and h.requires_grad:
                h = self._attach_block_budget(h, block_idx)
            if capture_layers is not None and block_idx in capture_layers:
                captured[block_idx] = h
        torch.set_grad_enabled(grad_was)  # (no-op unless early_nograd never reached its split layer)
        return h, captured

    def enable_summary_token_attention(
        self,
        doc_start_id: int,
        doc_end_id: int,
        summary_token_id: int,
        eos_id: int,
        pad_id: Optional[int] = None,
        standard_mix_prob: float = 0.0,
        mix_start_p: float = 0.0,
        mix_end_p: float = 0.0,
        mix_total_forwards: int = 0,
        mix_schedule: str = "linear",
        mix_step_frac: float = 0.5,
        mix_seed: int = 42,
        mix_log_interval: int = 500,
    ) -> None:
        """
        Enable per-token ``summary_roles`` construction for
        :class:`~olmo_core.nn.attention.summary_token.SummaryTokenAttention`.

        When enabled, :meth:`forward` derives the roles from the token stream on every step (see
        :func:`~olmo_core.nn.attention.summary_mask.build_summary_roles`) and forwards them, plus the
        per-example mask-mixture arm, to every block.

        .. note::
            Every sequence mixer must *tolerate* the ``summary_roles`` / ``causal_example`` keywords,
            but they need not consume them. On a hybrid such as Qwen3.5
            (``block_pattern=["gdn","gdn","gdn","attn"]``) only the 8 attention layers of 32 are
            masked; the 24 :class:`~olmo_core.nn.attention.recurrent.GatedDeltaNet` layers absorb the
            keyword via ``**kwargs`` and are an **unrestricted cross-document channel**. That is a
            deliberate scoping choice for the SummTokenSFT experiment, not an oversight -- but it
            means results describe what the *attention* layers route, not the whole model. The
            realized split is logged at :meth:`enable_summary_token_attention` time so it lands in the
            saved config rather than having to be rediscovered.

        **Mask mixing** (optional): during *training*, each example is independently assigned to the
        causal arm with probability ``p`` from
        :func:`~olmo_core.nn.attention.chunked_mask.mask_mix_standard_prob` -- either static
        (``standard_mix_prob``) or a curriculum (``mix_start_p -> mix_end_p`` over
        ``mix_total_forwards``, ``"linear"`` or ``"step"``). ``p == 0`` (all defaults) means every
        example is masked.

        :param doc_start_id: The ``<|box_start|>`` token id.
        :param doc_end_id: The ``<|box_end|>`` token id.
        :param summary_token_id: The ``<|summ|>`` token id appended after each context document.
        :param eos_id: The EOS / example terminator.
        :param pad_id: Optional dedicated padding id. Needed whenever the tokenizer ties ``pad`` to
            ``eos`` (Qwen3.5 does), or tail padding cannot be told from content.
        :param standard_mix_prob: Static per-example causal-arm probability.
        :param mix_start_p: Curriculum start probability (at forward 0).
        :param mix_end_p: Curriculum end probability (at ``mix_total_forwards``).
        :param mix_total_forwards: Forwards over which the curriculum runs. **This must be the number
            of forwards a single rank performs**, i.e.
            ``(n_instances * epochs) // (world_size * micro_batch_instances)`` -- getting it wrong
            leaves ``p`` short of ``mix_end_p`` and has silently voided prior arms.
        :param mix_schedule: ``"linear"`` or ``"step"``; see
            :data:`~olmo_core.nn.attention.chunked_mask.MASK_MIX_SCHEDULES`.
        :param mix_step_frac: ``"step"`` only -- where in training the switch happens.
        :param mix_seed: Base seed for the deterministic per-(forward, example) coin.
        :param mix_log_interval: Log a ``[curriculum]`` line every this many training forwards.

        :raises OLMoConfigurationError: If static and curriculum mixing are both requested, or a
            curriculum is requested without ``mix_total_forwards > 0``.
        """
        curriculum = mix_start_p > 0.0 or mix_end_p > 0.0
        if standard_mix_prob > 0.0 and curriculum:
            raise OLMoConfigurationError(
                "Mask mixing: 'standard_mix_prob' (static) is mutually exclusive with the curriculum "
                "('mix_start_p' / 'mix_end_p'); set only one."
            )
        if curriculum and mix_total_forwards <= 0:
            raise OLMoConfigurationError(
                "Mask-mix curriculum ('mix_start_p' / 'mix_end_p') requires 'mix_total_forwards' > 0 "
                "(the number of forwards ONE RANK performs)."
            )
        if mix_schedule not in MASK_MIX_SCHEDULES:
            raise OLMoConfigurationError(
                f"Unknown mix_schedule {mix_schedule!r}; expected {MASK_MIX_SCHEDULES}"
            )

        # Record how much of the model the mask actually covers. On a hybrid this is a minority of
        # layers, and that fact belongs in the saved config rather than in someone's memory.
        n_mixers = 0
        n_summary_layers = 0
        for block in self.blocks.values():
            mixer = getattr(block, "attention", None)
            if mixer is None:
                continue
            n_mixers += 1
            if isinstance(mixer, SummaryTokenAttention):
                n_summary_layers += 1
        if n_summary_layers == 0:
            raise OLMoConfigurationError(
                "enable_summary_token_attention() was called but no layer uses "
                "AttentionType.summary_token, so the mask would be a silent no-op."
            )
        unmasked = n_mixers - n_summary_layers
        log.info(
            f"[summary-token] masking {n_summary_layers} of {n_mixers} sequence mixers"
            + (
                f"; the other {unmasked} ignore the roles and are an UNRESTRICTED cross-document "
                "channel. Results describe what the attention layers route, not the whole model."
                if unmasked
                else " (every layer is masked)."
            )
        )

        mix: Optional[Dict[str, Any]] = None
        if standard_mix_prob > 0.0 or curriculum:
            mix = {
                "standard_mix_prob": float(standard_mix_prob),
                "mix_start_p": float(mix_start_p),
                "mix_end_p": float(mix_end_p),
                "mix_total_forwards": int(mix_total_forwards),
                "mix_schedule": str(mix_schedule),
                "mix_step_frac": float(mix_step_frac),
                "mix_seed": int(mix_seed),
                "log_interval": max(1, int(mix_log_interval)),
                "forward_idx": 0,
            }
        self._summary_token_attention = {
            "doc_start_id": int(doc_start_id),
            "doc_end_id": int(doc_end_id),
            "summary_token_id": int(summary_token_id),
            "eos_id": int(eos_id),
            "pad_id": None if pad_id is None else int(pad_id),
            "n_summary_layers": n_summary_layers,
            "n_sequence_mixers": n_mixers,
            "mix": mix,
        }

    def set_summary_eval_mask_mode(self, mode: str) -> str:
        """
        Choose which arm of the summary-token mask mixture is served at **inference**.

        The mixture coin in :meth:`forward` is drawn during training only, so at inference the arm is
        not a per-example random variable but a single decision about how to serve the model -- and it
        has to be made explicitly, because the two arms are different masks:

        - ``"causal"`` (default): every example is on the causal arm, i.e. plain causal attention with
          the ``<|summ|>`` tokens present as ordinary tokens. This is the arm that models trained with
          ``standard_mix_prob=1.0``, or with a curriculum ending at ``mix_end_p=1.0``, actually saw.
        - ``"restricted"``: no example is on the causal arm, so the summary mask applies in full and
          the trailing query reads only summaries and its own document (subject to the layer's
          :class:`~olmo_core.nn.attention.summary_token.SummaryMaskSpec`). This measures what the
          summary tokens compressed, and is the condition to use when the question is whether the
          model can answer from summaries alone.

        Defaulting to ``"causal"`` makes the served mask match the trained one for the arms that end
        up fully causal; serving ``"restricted"`` to those is a train/test mismatch that reads as a
        capability result. Training is unaffected either way -- this is consulted only when
        ``self.training`` is ``False``.

        :param mode: ``"causal"`` or ``"restricted"``.

        :returns: The mode that was set.

        :raises OLMoConfigurationError: If ``mode`` is not one of the two arms.
        """
        if mode not in ("causal", "restricted"):
            raise OLMoConfigurationError(
                f"Unknown summary eval mask mode {mode!r}; expected 'causal' or 'restricted'."
            )
        self._summary_eval_mask_mode = mode
        return mode

    def set_landmark_eval_top_k(self, top_k: Optional[int]) -> int:
        """
        Enable inference-only hard top-k landmark retrieval on every eager landmark attention layer
        (e.g. :class:`~olmo_core.nn.attention.DocumentLandmarkAttention`): each query attends only the
        ``top_k`` highest-scoring landmark blocks. ``None`` restores exact (all-block) attention.
        Training is unaffected. Returns the number of layers updated.

        :param top_k: Number of landmark blocks to keep per query, or ``None`` for exact.
        """
        n = 0
        for module in self.modules():
            if hasattr(module, "set_eval_top_k") and callable(module.set_eval_top_k):
                module.set_eval_top_k(top_k)
                n += 1
        return n

    def compute_auxiliary_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        del reset
        return {}

    def reset_auxiliary_metrics(self):
        pass

    @property
    def pp_enabled(self) -> bool:
        return self._pp_enabled

    @property
    def fp8_enabled(self) -> bool:
        return self._fp8_enabled

    @property
    def tp_enabled(self) -> bool:
        return self._tp_enabled

    @property
    def fsdp_enabled(self) -> bool:
        return self._fsdp_enabled

    @property
    def is_moe(self) -> bool:
        return False

    @property
    def device(self) -> torch.device:
        if self._device is None:
            for p in self.parameters():
                if p.numel() > 0:
                    self._device = p.device
                    break
            else:
                self._device = get_default_device()
        return self._device

    @property
    def compile_enabled(self) -> bool:
        return self._compile_enabled

    def get_rope_buffers(
        self, seq_len: int, device: Optional[torch.device] = None
    ) -> Dict[int, Optional[RoPEBuffers]]:
        """
        Get the RoPE buffers to pass to each layer.
        """
        if device is None:
            device = self.device
        rope_buffers = {}
        for key, block in self.blocks.items():
            if isinstance(block.attention, Attention):
                rope = cast(Optional[RotaryEmbeddingBase], block.attention.rope)
                rope_buffers[int(key)] = None if rope is None else rope.get_buffers(seq_len, device)
            else:
                rope_buffers[int(key)] = None
        return rope_buffers

    def prepare_cp_sequence_inputs(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """
        Shard sequence inputs for context parallelism before entering a pipeline schedule.

        Pipeline-parallel stages exchange hidden activations with already-local CP sequence
        lengths. The model still needs the original full sequence length later so each stage can
        build and shard its own RoPE buffers consistently.

        :returns: The CP-sharded ``input_ids``, the CP-sharded ``labels`` (or ``None``), and the
            original (pre-shard) sequence length.
        """
        cp_load_balancer = self._cp_load_balancer
        if cp_load_balancer is None:
            raise OLMoConfigurationError(
                "prepare_cp_sequence_inputs() requires context parallelism to be applied first"
            )
        if input_ids.dim() < 2:
            raise ValueError("'input_ids' must have a batch and sequence dimension")
        if labels is not None and labels.shape[:2] != input_ids.shape[:2]:
            raise ValueError(
                f"'labels' shape {tuple(labels.shape)} does not match input batch/sequence "
                f"shape {tuple(input_ids.shape[:2])}"
            )

        original_seq_len = input_ids.shape[1]
        inputs = [input_ids]
        seq_dims = [1]
        pad_values: List[Union[int, float]] = [0]

        if labels is not None:
            inputs.append(labels)
            seq_dims.append(1)
            pad_values.append(ignore_index)

        sharded_inputs = cp_load_balancer.batch_shard(
            inputs=inputs,
            seq_dims=seq_dims,
            pad_values=pad_values,
        )
        sharded_labels = sharded_inputs[1] if labels is not None else None
        return sharded_inputs[0], sharded_labels, original_seq_len

    @torch.no_grad()
    def init_weights(
        self,
        *,
        max_seq_len: Optional[int] = None,
        max_local_microbatch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        world_mesh: Optional[DeviceMesh] = None,
        model_part_idx: int = 0,
    ) -> torch.Generator:
        """
        Initialize the model weights.

        :param max_seq_len: The maximum sequence length expected. This is used
            to warm up the RoPE cache.
        :param max_local_microbatch_size: The maximum local (rank) micro-batch size (in tokens)
            expected. This is used to warm-up some MoE cache.
        :param device: The device the local copy of the model will be trained on.
        :param model_part_idx: The local index of this model part on the current rank.
            With interleaved pipeline schedules a single rank can own multiple model
            chunks, and each must receive a distinct seed; otherwise their parameters
            would be identical.
        """
        device = device or self.device
        self.to_empty(device=device)

        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()  # type: ignore

        seed = self.init_seed
        if world_mesh is not None and self.pp_enabled:
            pp_mesh = get_pp_mesh(world_mesh)
            seed += pp_mesh.get_local_rank() + model_part_idx * pp_mesh.size()

        generator = torch.Generator(device).manual_seed(seed)

        if self.embeddings is not None:
            self.init_method.init_embeddings(
                self.embeddings,
                d_model=self.d_model,
                embed_scale=self.embed_scale,
                std=(
                    self.embedding_init_std
                    if self.embedding_init_std is not None
                    else self.init_std
                ),
                generator=generator,
            )

        # Re-establish weight tying since `to_empty` above allocates fresh storage.
        if self.tie_word_embeddings:
            self._tie_weights()

        for block in self.blocks.values():
            # This might fail if it's wrapped.
            #  assert isinstance(block, TransformerBlock)
            block = cast(TransformerBlock, block)
            att = cast(SequenceMixer, block.attention)

            # Attention weights.
            self.init_method.init_attention(
                att,
                d_model=self.d_model,
                block_idx=block.block_idx,
                num_blocks=self.n_layers,
                std=self.init_std,
                generator=generator,
            )

            # Feed-forward weights.
            if hasattr(block, "feed_forward"):
                self.init_method.init_feed_forward(
                    block.feed_forward,
                    d_model=self.d_model,
                    block_idx=block.block_idx,
                    num_blocks=self.n_layers,
                    std=self.init_std,
                    generator=generator,
                )

            # MoE weights.
            if hasattr(block, "feed_forward_moe"):
                block = cast(MoETransformerBlock, block)
                if max_local_microbatch_size is not None:
                    block.feed_forward_moe.warmup_cache(max_local_microbatch_size)
                self.init_method.init_feed_forward_moe(
                    block.feed_forward_moe,
                    d_model=self.d_model,
                    block_idx=block.block_idx,
                    num_blocks=self.n_layers,
                    std=self.init_std,
                    generator=generator,
                )

            # Fused MoE-v2 weights.
            from ..ddp.block import OLMoDDPTransformerBlock

            if isinstance(block, OLMoDDPTransformerBlock):
                self.init_method.init_moe_v2(
                    block,
                    d_model=self.d_model,
                    block_idx=block.block_idx,
                    num_blocks=self.n_layers,
                    std=self.init_std,
                    generator=generator,
                )

            if isinstance(att, Attention):
                # Warm up attention backend cache.
                if max_seq_len is not None and att.backend is not None:
                    att.backend.warmup_cache(max_seq_len, device)

                # Warm up RoPE cache.
                if max_seq_len is not None and att.rope is not None:
                    att.rope.warmup_cache(max_seq_len, device)

        if self.lm_head is not None and not self.tie_word_embeddings:
            self.init_method.init_final_w_out(
                self.lm_head.w_out,
                d_model=self.d_model,
                std=self.init_std,
                generator=generator,
            )

        return generator

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Dict[str, Any],
        Dict[int, Dict[str, Any]],
        Dict[str, Any],
    ]:
        # NOTE: with pipeline parallelism input_ids might actually be an intermediate output,
        # so we have to be careful here.
        B, S = input_ids.shape[:2]

        # Context-parallel inputs may already be sequence-sharded by the caller (e.g. the pipeline
        # train module shards the batch before the pipeline schedule). In that case we must not
        # shard input_ids/labels again, but we still shard the RoPE buffers, which are built from the
        # original (pre-shard) sequence length carried in 'cp_original_seq_len'.
        cp_already_sharded = kwargs.pop("cp_already_sharded", False)
        cp_original_seq_len = kwargs.pop("cp_original_seq_len", None)
        if cp_original_seq_len is not None:
            cp_original_seq_len = int(cp_original_seq_len)
            if cp_original_seq_len <= 0:
                raise ValueError("'cp_original_seq_len' must be positive")
            if not cp_already_sharded:
                raise OLMoConfigurationError(
                    "'cp_original_seq_len' is only valid when 'cp_already_sharded=True'"
                )
        if cp_already_sharded and cp_original_seq_len is None:
            raise OLMoConfigurationError(
                "'cp_already_sharded=True' requires 'cp_original_seq_len' so RoPE buffers can "
                "be sharded from the full sequence length"
            )

        all_block_kwargs: Dict[str, Any] = {}
        per_block_kwargs: Dict[int, Dict[str, Any]] = defaultdict(dict)
        lm_head_kwargs: Dict[str, Any] = dict(
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
            z_loss_multiplier=z_loss_multiplier,
            return_logits=return_logits,
            logits_to_keep=logits_to_keep,
        )

        if loss_div_factor is not None:
            loss_div_factor = move_to_device(loss_div_factor, self.device)
            lm_head_kwargs["loss_div_factor"] = loss_div_factor
            all_block_kwargs["loss_div_factor"] = loss_div_factor

        # Prepare document length inputs.
        max_doc_len: Optional[int] = None
        cu_doc_lens: Optional[torch.Tensor] = None
        doc_lens: Optional[torch.Tensor] = None
        cache_leftpad: Optional[torch.Tensor] = kwargs.pop("cache_leftpad", None)

        # Explicit per-token positions (soft-token pooling: kept tokens keep their ORIGINAL
        # positions so RoPE geometry matches full-attention inference). Threaded to every block.
        if (position_ids := kwargs.pop("position_ids", None)) is not None:
            if self._cp_load_balancer is not None:
                raise NotImplementedError("position_ids is not supported with context parallelism")
            all_block_kwargs["position_ids"] = move_to_device(position_ids, self.device)
        # Soft-token aux matching: the position-causal masked-attention bias + the shared per-layer
        # capture dict (each Attention appends its q/k/v slices for the matching loss).
        if (attn_bias := kwargs.pop("attn_bias", None)) is not None:
            all_block_kwargs["attn_bias"] = move_to_device(attn_bias, self.device)
        if (aux_capture := kwargs.pop("aux_capture", None)) is not None:
            all_block_kwargs["aux_capture"] = aux_capture
        if (kv_grad_mask := kwargs.pop("kv_grad_mask", None)) is not None:
            all_block_kwargs["kv_grad_mask"] = move_to_device(kv_grad_mask, self.device)
        if (soft_keep := kwargs.pop("soft_keep", None)) is not None:
            # Differentiable token removal (debug/learned_router, 2026-09-27): a (B, T) keep
            # probability every sequence mixer applies without compacting the row -- attention as
            # a +log(p) key bias, GatedDeltaNet as p-scaled q/k/v/beta/g. Absent -> untouched.
            all_block_kwargs["soft_keep"] = move_to_device(soft_keep, self.device)
        # Oracle slot K/V overrides (soft-token pooling): one gathered slot stack for the batch,
        # sliced per layer into per-block kwargs (each Attention rotates + injects its own slice).
        if (oracle_ovr := kwargs.pop("soft_kv_override_layers", None)) is not None:
            slots = move_to_device(oracle_ovr["slots"], self.device)
            biases = move_to_device(oracle_ovr["biases"], self.device)
            rows = move_to_device(oracle_ovr["rows"], self.device)
            cols = move_to_device(oracle_ovr["cols"], self.device)
            slot_pos = move_to_device(oracle_ovr["pos"], self.device)
            for li in range(slots.shape[1]):
                per_block_kwargs[li]["soft_kv_override"] = {
                    "rows": rows,
                    "cols": cols,
                    "pos": slot_pos,
                    "k": slots[:, li, 0],
                    "v": slots[:, li, 1],
                    "bias": biases[:, li],
                }

        if (doc_lens := kwargs.pop("doc_lens", None)) is not None and (
            max_doc_lens := kwargs.pop("max_doc_lens", None)
        ) is not None:
            max_doc_len = max(max_doc_lens)
            cu_doc_lens = get_cumulative_document_lengths(doc_lens)

        if self._document_chunk_attention is not None and self._cp_load_balancer is not None:
            raise NotImplementedError(
                "Document-chunked landmark attention (enable_document_chunk_attention) is not yet "
                "supported together with context parallelism (chunk_ids are not sequence-sharded)."
            )

        # Summary-token attention: derive the per-token roles and the per-example mixture arm from the
        # FULL, unsharded token stream, before the context-parallel sharding below. Under Ulysses the
        # roles deliberately stay full-length: the all-to-all inside
        # :class:`~olmo_core.nn.attention.summary_token.SummaryTokenAttention` restores the whole
        # sequence (with partitioned heads) before the mask is applied, so a sharded roles tensor
        # would be the wrong object.
        summary_roles: Optional[torch.Tensor] = None
        causal_example: Optional[torch.Tensor] = None
        if self._summary_token_attention is not None:
            st_cfg = self._summary_token_attention
            summary_roles = build_summary_roles(
                input_ids,
                doc_start_id=st_cfg["doc_start_id"],
                doc_end_id=st_cfg["doc_end_id"],
                summary_token_id=st_cfg["summary_token_id"],
                eos_id=st_cfg["eos_id"],
                pad_id=st_cfg.get("pad_id"),
            )
            # Mask mixing (training only): assign each example to the causal or the masked arm with a
            # scheduled, seeded probability. Unlike the chunked path this does not rewrite the roles;
            # the arm is passed to the mask as its own input. Kept eager (python-seeded coin plus a
            # counter) -- exclude from torch.compile.
            mix = st_cfg.get("mix")
            if mix is not None and self.training:
                idx = mix["forward_idx"]
                p = mask_mix_standard_prob(
                    idx,
                    standard_mix_prob=mix["standard_mix_prob"],
                    mix_start_p=mix["mix_start_p"],
                    mix_end_p=mix["mix_end_p"],
                    mix_total_forwards=mix["mix_total_forwards"],
                    mix_schedule=mix["mix_schedule"],
                    mix_step_frac=mix["mix_step_frac"],
                )
                mix["forward_idx"] = idx + 1
                # One coin per packed EXAMPLE, not per instance: an instance may hold several
                # examples, and drawing once for the row would make the realized causal fraction
                # depend on how the packer happened to bin them. The flags are then scattered to
                # tokens by example_id, which is the form the mask consumes.
                example_id = summary_roles[:, ROLE_EXAMPLE_ID]
                n_examples = int(example_id.max().item()) + 1  # -1 everywhere (all pad) -> 0
                n_examples = max(1, n_examples)
                flags = causal_example_flags(
                    input_ids.shape[0] * n_examples,
                    p,
                    forward_idx=mix["forward_idx"],
                    mix_seed=mix["mix_seed"],
                ).view(input_ids.shape[0], n_examples)
                causal_example = torch.gather(
                    flags.to(example_id.device), 1, example_id.clamp_min(0).long()
                )
                if p > 0.0 and mix["forward_idx"] % mix["log_interval"] == 0:
                    log.info(
                        f"[curriculum] forward={mix['forward_idx']} p_causal={p:.3f} "
                        f"causal_this_batch={int(flags.sum())}/{flags.numel()} examples "
                        f"({input_ids.shape[0]} instances x {n_examples})"
                    )
            elif not self.training and self._summary_eval_mask_mode == "causal":
                # Inference: the arm is a serving decision, not a coin. Mark every token causal so the
                # mask reduces to plain causal attention (see ``set_summary_eval_mask_mode``). Leaving
                # this None -- as every forward did before the mode existed -- silently served the
                # fully restricted mask to models that trained entirely on the causal arm.
                causal_example = torch.ones(
                    input_ids.shape, dtype=torch.bool, device=input_ids.device
                )

        # Shard inputs and RoPE buffers on sequence dimension if using context parallelism.
        if (cp_load_balancer := self._cp_load_balancer) is not None:
            inputs: List[torch.Tensor] = []
            seq_dims: List[int] = []
            pad_values: List[Union[int, float]] = []
            keys: List[str] = []

            if cp_already_sharded:
                input_ids = move_to_device(input_ids, self.device)
                labels = move_to_device(labels, self.device)
                if cu_doc_lens is not None or max_doc_len is not None:
                    raise OLMoConfigurationError(
                        "context parallel inputs that are already sharded cannot also pass "
                        "'doc_lens'/'max_doc_lens'; pre-sharded intra-document masking metadata "
                        "is not implemented yet"
                    )
            else:
                inputs.append(input_ids)
                seq_dims.append(1)
                pad_values.append(0)
                keys.append("input_ids")

            rope_seq_len = cp_original_seq_len if cp_already_sharded else S

            # NOTE: initialize buffer(s) on CPU to avoid possible host-device sync when sharding.
            for block_idx, rope_buffers in self.get_rope_buffers(
                rope_seq_len, torch.device("cpu")
            ).items():
                if rope_buffers is not None:
                    # Also shard RoPE buffers based on the context parallelism load balancer.
                    if rope_buffers.pos_sin is not None:
                        inputs.append(rope_buffers.pos_sin)
                        seq_dims.append(0)
                        pad_values.append(0.0)
                        keys.append(f"block_{block_idx}.pos_sin")
                    if rope_buffers.pos_cos is not None:
                        inputs.append(rope_buffers.pos_cos)
                        seq_dims.append(0)
                        pad_values.append(0.0)
                        keys.append(f"block_{block_idx}.pos_cos")
                    if rope_buffers.freqs_cis is not None:
                        inputs.append(rope_buffers.freqs_cis)
                        seq_dims.append(0)
                        pad_values.append(0.0)
                        keys.append(f"block_{block_idx}.freqs_cis")

            if labels is not None and not cp_already_sharded:
                inputs.append(labels)
                seq_dims.append(1)
                pad_values.append(ignore_index)
                keys.append("labels")

            if cache_leftpad is not None:
                raise NotImplementedError("cache_leftpad is not supported with context parallelism")

            if cu_doc_lens is not None:
                # NOTE: Can only shard properly here if 'input_ids' is flat, i.e. a single instance.
                # TODO: (epwalsh) We could just flatten all of the inputs here, but then we risk going
                # beyond the model's maximum sequence length, which might be okay at least
                # with relative positional encodings, but then again if you're resorting to context
                # parallelism you can probably only fit a single instance at a time anyway.
                if B != 1:
                    raise RuntimeError(
                        f"Rank micro-batches must consist of a single instance when using "
                        f"context parallelism with intra-document masking (got {B} instances)"
                    )
                inputs, additional_inputs = cp_load_balancer.batch_shard_by_document(
                    inputs=inputs,
                    seq_dims=seq_dims,
                    cu_doc_lens=cu_doc_lens,
                    pad_values=pad_values,
                    length_multiple=16,
                )
                for key, value in additional_inputs.items():
                    all_block_kwargs[key] = move_to_device(value, self.device)

            else:
                if inputs:
                    inputs = cp_load_balancer.batch_shard(
                        inputs=inputs,
                        seq_dims=seq_dims,
                        pad_values=pad_values,
                    )

            for key, value in zip(keys, inputs):
                if key.startswith("block_"):
                    block_key, key = key.split(".", 1)
                    block_idx = int(block_key.replace("block_", ""))
                    per_block_kwargs[block_idx][key] = move_to_device(value, self.device)
                else:
                    all_block_kwargs[key] = move_to_device(value, self.device)

            if not cp_already_sharded:
                input_ids = all_block_kwargs.pop("input_ids")
                labels = all_block_kwargs.pop("labels", None)
        else:
            if cp_already_sharded:
                raise OLMoConfigurationError(
                    "'cp_already_sharded=True' requires context parallelism to be applied first"
                )
            input_ids = move_to_device(input_ids, self.device)
            labels = move_to_device(labels, self.device)

            if (max_doc_len is not None or cu_doc_lens is not None) and cache_leftpad is not None:
                raise ValueError("max_doc_len/cu_doc_lens and cache_leftpad are mutually exclusive")
            if max_doc_len is not None or cu_doc_lens is not None:
                all_block_kwargs["max_doc_len"] = max_doc_len
                all_block_kwargs["cu_doc_lens"] = move_to_device(cu_doc_lens, self.device)
            if cache_leftpad is not None:
                all_block_kwargs["cache_leftpad"] = move_to_device(cache_leftpad, self.device)

            # Document-chunked landmark attention: reconstruct per-token chunk_ids from the boundary
            # tokens and forward them to every block (see ``enable_document_chunk_attention``).
            if self._document_chunk_attention is not None:
                cfg = self._document_chunk_attention
                chunk_ids = build_chunk_ids_from_tokens(
                    input_ids,
                    doc_start_id=cfg["doc_start_id"],
                    doc_end_id=cfg["doc_end_id"],
                    eos_id=cfg["eos_id"],
                    mode=cfg["mode"],
                    pad_id=cfg.get("pad_id"),
                )
                # Mask mixing (training only): with a scheduled, seeded probability, collapse a random
                # subset of examples to plain causal by neutralizing their roles to all-FREE. Runs once
                # per forward on the shared chunk_ids (before they are threaded to every block), so all
                # layers and any AC recompute see the same mixed roles. p == 0 (no mix configured, or a
                # forward where no example is drawn) leaves chunk_ids untouched -> bit-identical to
                # pure chunked. Kept eager (python-seeded coin + counter) -- exclude from torch.compile.
                mix = cfg.get("mix")
                if mix is not None and self.training:
                    idx = mix["forward_idx"]
                    p = mask_mix_standard_prob(
                        idx,
                        standard_mix_prob=mix["standard_mix_prob"],
                        mix_start_p=mix["mix_start_p"],
                        mix_end_p=mix["mix_end_p"],
                        mix_total_forwards=mix["mix_total_forwards"],
                    )
                    mix["forward_idx"] = idx + 1
                    new_chunk_ids = collapse_roles_to_causal(
                        chunk_ids, p, forward_idx=mix["forward_idx"], mix_seed=mix["mix_seed"]
                    )
                    if (
                        new_chunk_ids is not chunk_ids
                    ):  # at least one example collapsed this forward
                        mix["n_collapsed"] += 1
                    chunk_ids = new_chunk_ids
                    if p > 0.0 and mix["forward_idx"] % mix["log_interval"] == 0:
                        log.info(
                            f"[curriculum] forward={mix['forward_idx']} p_standard={p:.3f} "
                            f"collapsed_forwards={mix['n_collapsed']}"
                        )
                all_block_kwargs["chunk_ids"] = move_to_device(chunk_ids, self.device)

        # Threaded to every block in both the CP and non-CP paths, and deliberately NOT sequence-
        # sharded (see the construction above).
        if summary_roles is not None:
            all_block_kwargs["summary_roles"] = move_to_device(summary_roles, self.device)
            if causal_example is not None:
                all_block_kwargs["causal_example"] = move_to_device(causal_example, self.device)

        emo_blocks = []
        emo_eos_token_ids = set()
        for block_idx, block in self.blocks.items():
            router = getattr(block, "routed_experts_router", None)
            if router is not None and getattr(router, "requires_segment_ids", False):
                emo_blocks.append(int(block_idx))
                emo_eos_token_ids.add(router.eos_token_id)
        if emo_blocks:
            if self._pp_enabled:
                raise OLMoConfigurationError(
                    "EMO routing does not currently support pipeline parallelism because "
                    "token-derived segment IDs are not carried between pipeline stages"
                )
            if self._tp_enabled:
                raise OLMoConfigurationError("EMO routing does not support tensor parallelism")
            if self._cp_load_balancer is not None:
                raise OLMoConfigurationError(
                    "EMO routing does not currently support context parallelism"
                )
            if len(emo_eos_token_ids) != 1:
                raise OLMoConfigurationError(
                    "All EMO routers in a model must use the same eos_token_id"
                )
            segment_ids = moe_ops.segment_ids_from_eos(input_ids, emo_eos_token_ids.pop())
            for block_idx in emo_blocks:
                per_block_kwargs[block_idx]["segment_ids"] = segment_ids

        if "cu_doc_lens" in all_block_kwargs:
            mark_dynamic(all_block_kwargs["cu_doc_lens"], 0, strict=False)  # type: ignore[arg-type]

        return (
            input_ids,
            labels,
            all_block_kwargs,
            per_block_kwargs,
            lm_head_kwargs,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        input_embeddings: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        """
        Run the transformer on the token input IDs.

        :param input_ids: The token input IDs, shape ``(batch_size, seq_len)``.
        :param input_embeddings: Pre-computed embeddings to use instead of looking up
            ``input_ids`` in the embedding table, shape
            ``(batch_size, seq_len, d_model)``.  When provided the embedding lookup,
            scale, and norm steps are all skipped.  Intended for multimodal use-cases
            where image features have already been spliced into the embedding sequence.
            Not supported with context parallelism.
        :param labels: The token labels, shape ``(batch_size, seq_len)``.
        :param ignore_index: The index to ignore in the loss computation. Default is -100.
        :param loss_reduction: The reduction method for the loss. Can be "mean", "sum", or "none".
        :param z_loss_multiplier: Optional multiplier for the z-loss regularization term.
        :param loss_div_factor: Optional divisor for the loss, can be a scalar or tensor.
        :param return_logits: Whether to return logits along with the loss when labels are provided.
        :param logits_to_keep: Number of positions to keep from the end of the sequence (if int),
            or tensor specifying which positions to keep. Default is 0 (keep all).

        :returns: The logits if ``labels`` is ``None`` or the losses if ``labels`` is not ``None``.
        """
        if input_embeddings is not None and self._cp_load_balancer is not None:
            raise RuntimeError(
                "`input_embeddings` is not supported with context parallelism: `_prepare_inputs` "
                "shards `input_ids`/`labels`/RoPE while `input_embeddings` stays full-size, which "
                "would misalign the hidden states."
            )

        # Soft-token document pooling ("B1"): during training, compact the sequence (pooled docs ->
        # one placeholder each) BEFORE anything else sees it. Eval/generation forwards untouched.
        soft_inject: Optional[Tuple] = None
        aux_ctx: Optional[Dict[str, Any]] = None
        distill_teacher: Optional[Dict[str, Any]] = None
        track_lens = (
            getattr(self, "_flop_pad_id", None) is not None
            and self.training
            and not input_ids.is_floating_point()
        )
        if track_lens and self._pooled_soft_tokens is None:
            from ..pooled_soft_token import example_lengths

            lens = example_lengths(input_ids, kwargs.get("doc_lens"), self._flop_pad_id)
            self._record_example_lens(lens, lens)
        # Token exit: keep the FULL row (and its kwargs) for blocks [0, L); the soft-token
        # compaction below decides which tokens survive past block L.
        tx_full: Optional[Tuple[torch.Tensor, Dict[str, Any]]] = None
        tx_idx: Optional[torch.Tensor] = None
        if (
            self._token_exit is not None
            and self._pooled_soft_tokens is not None
            and self.training
            and not input_ids.is_floating_point()
        ):
            if kwargs.get("doc_lens") is not None or input_embeddings is not None:
                raise NotImplementedError(
                    "token exit runs on padded one-example rows (no doc_lens packing, no "
                    "input_embeddings)"
                )
            tx_full = (input_ids, dict(kwargs))
        if (
            self._pooled_soft_tokens is not None
            and self.training
            and not input_ids.is_floating_point()  # PP passes hidden states; only act on token ids
        ):
            # Paired-distillation coin: synchronized across ranks via the shared forward counter.
            pst0 = self._pooled_soft_tokens
            paired = False
            if pst0["distill_prob"] > 0.0 and labels is not None:
                import random as _random

                cnt = pst0["_distill_counter"]
                pst0["_distill_counter"] = cnt + 1
                paired = (
                    _random.Random(f"distill:{pst0['keep_seed']}:{cnt}").random()
                    < pst0["distill_prob"]
                )
            if paired:
                # TEACHER pass: the ORIGINAL row, plain full attention, WITH LM gradient.
                t_ids, t_labels, t_abk, t_pbk, t_lmk = self._prepare_inputs(
                    input_ids,
                    labels,
                    ignore_index=ignore_index,
                    loss_reduction=loss_reduction,
                    z_loss_multiplier=z_loss_multiplier,
                    loss_div_factor=loss_div_factor,
                    return_logits=False,
                )
                h_t = self.embeddings(t_ids)  # type: ignore[misc]
                if self.embed_scale is not None:
                    h_t = h_t * self.embed_scale
                if self.embedding_norm is not None:
                    h_t = self.embedding_norm(h_t)
                h_t, t_caps = self._run_blocks(
                    h_t, t_abk, t_pbk, capture_layers=set(pst0["distill_layers"])
                )
                t_lmk["labels"] = t_labels
                t_out = self.lm_head(h_t, **t_lmk)  # type: ignore[misc]
                # Distill positions: every labeled (answer) position PLUS sampled FREE-region
                # positions -- free tokens survive compaction, so both passes have them, and
                # matching across the whole row constrains the pathway far more per teacher
                # than answer positions alone (the 32k signal-density fix).
                ans_r, ans_c = (t_labels != ignore_index).nonzero(as_tuple=True)
                roles_t = build_chunk_ids_from_tokens(
                    t_ids,
                    doc_start_id=pst0["doc_start_id"],
                    doc_end_id=pst0["doc_end_id"],
                    eos_id=pst0["eos_id"],
                    mode="chunked",
                )
                free_r, free_c = (roles_t == -1).nonzero(as_tuple=True)
                n_extra = min(96 * t_ids.shape[0], int(free_r.numel()))
                if n_extra > 0:
                    sel_f = torch.randperm(free_r.numel(), device=free_r.device)[:n_extra]
                    d_rows = torch.cat([ans_r, free_r[sel_f]])
                    d_pos = torch.cat([ans_c, free_c[sel_f]])
                else:
                    d_rows, d_pos = ans_r, ans_c
                distill_teacher = {
                    "loss_full": t_out.loss,
                    "rows": d_rows,
                    "pos": d_pos,
                    "caps": {li: t_caps[li][d_rows, d_pos].detach() for li in t_caps},
                }
            n_tokens_in = int(input_ids.numel())
            labels_in = labels
            # PACKED rows (doc_lens given): unpack to one example per row so the per-example gold
            # fingerprints and the compaction apply unchanged, then re-pack the compacted examples
            # into one varlen row below (repack_compacted_rows).
            packed = kwargs.get("doc_lens") is not None
            pst0["_segment_mode"] = packed
            if packed:
                from ..pooled_soft_token import unpack_packed_rows

                self._check_soft_token_segment_mode()
                packed_orig = (input_ids, labels, kwargs.pop("doc_lens"), kwargs.pop("max_doc_lens", None))
                up = unpack_packed_rows(
                    input_ids, labels, packed_orig[2], pad_id=pst0["eos_id"], ignore_index=ignore_index
                )
                input_ids, labels, dense_lens = up.input_ids, up.labels, up.lengths
                holder = self._pooled_keep_holder
                if holder is not None and getattr(holder, "keep_docs_fn", None) is not None:
                    holder.keep_docs = holder.keep_docs_fn(input_ids)
            elif track_lens:
                from ..pooled_soft_token import example_lengths

                dense_lens = example_lengths(input_ids, None, pst0["eos_id"])
            compacted = self._compact_pooled_soft_tokens(input_ids, labels, ignore_index)
            if compacted is None and packed:
                input_ids, labels = packed_orig[0], packed_orig[1]
                kwargs["doc_lens"], kwargs["max_doc_lens"] = packed_orig[2], packed_orig[3]
            if compacted is None and track_lens:
                self._record_example_lens(dense_lens, dense_lens)
            if compacted is not None:
                cb, soft_inject, oracle_ovr = compacted
                if tx_full is not None:
                    # Slot-less compaction keeps tokens in order at their original positions, so
                    # the compacted row's position_ids ARE the kept columns of the full row (pad
                    # columns point at T - 1: no label, after every real token).
                    assert soft_inject is not None and soft_inject[0].numel() == 0
                    soft_inject = None
                    tx_idx = cb.position_ids
                    self._token_exit.record(  # type: ignore[union-attr]
                        int(tx_full[0].numel()), int(cb.row_lens.sum().item())
                    )
                input_ids, labels = cb.input_ids, cb.labels
                if track_lens:
                    self._record_example_lens(dense_lens, [int(x) for x in cb.row_lens.tolist()])
                if packed:
                    from ..pooled_soft_token import repack_compacted_rows

                    input_ids, labels, rp_pos, rp_dl, rp_mdl = repack_compacted_rows(
                        cb,
                        pad_id=pst0["eos_id"],
                        ignore_index=ignore_index,
                        pad_multiple=int(pst0.get("pad_multiple", 1)),
                    )
                    kwargs["doc_lens"], kwargs["max_doc_lens"] = rp_dl, rp_mdl
                # Compaction accounting for the FLOP meter (accumulated across microbatches; the
                # FlopMeterCallback reads and resets it every step).
                stats = getattr(self, "_soft_token_compaction", None)
                if stats is None:
                    stats = self._soft_token_compaction = {
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "rows": 0,
                    }
                stats["tokens_in"] += n_tokens_in
                stats["tokens_out"] += int(input_ids.numel())
                stats["rows"] += int(input_ids.shape[0])
                if labels_in is not None and labels is not None:
                    # Live labels before/after compaction. The train module's loss divisor counts
                    # them BEFORE compaction, so the logged CE is (true CE on kept labels) x
                    # labels_out / labels_in; FlopMeterCallback logs that ratio.
                    stats["labels_in"] = stats.get("labels_in", 0) + int(
                        (labels_in != ignore_index).sum().item()
                    )
                    stats["labels_out"] = stats.get("labels_out", 0) + int(
                        (labels != ignore_index).sum().item()
                    )
                kwargs["position_ids"] = rp_pos if packed else cb.position_ids
                if oracle_ovr is not None:
                    kwargs["soft_kv_override_layers"] = oracle_ovr
                    # Slot biases need the additive-bias SDPA path: position-causality is
                    # equivalent to sequence-causality on the compacted row (content is sorted
                    # by original position), so this only changes the kernel, not the math.
                    from ..pooled_soft_token import build_position_causal_bias

                    kwargs.setdefault(
                        "attn_bias",
                        build_position_causal_bias(
                            cb,
                            dtype=self.embeddings.weight.dtype,  # type: ignore[union-attr]
                            device=input_ids.device,
                        ),
                    )
                pst = self._pooled_soft_tokens
                if pst["detach_soft_kv"] and cb.soft_rows.numel() > 0:
                    kv_grad_mask = torch.ones_like(cb.input_ids, dtype=torch.bool)
                    kv_grad_mask[cb.soft_rows, cb.soft_cols] = False
                    kwargs["kv_grad_mask"] = kv_grad_mask
                if (
                    pst["aux_match_weight"] > 0.0
                    and cb.shadow_rows.numel() > 0
                    and labels is not None
                ):
                    from ..pooled_soft_token import build_position_causal_bias

                    kwargs["attn_bias"] = build_position_causal_bias(
                        cb, dtype=self.embeddings.weight.dtype, device=input_ids.device  # type: ignore[union-attr]
                    )
                    lab_rows, lab_cols = (labels != ignore_index).nonzero(as_tuple=True)
                    if lab_rows.numel() > 0:
                        nq = min(pst["aux_queries"] * cb.input_ids.shape[0], int(lab_rows.numel()))
                        sel = torch.randperm(lab_rows.numel(), device=lab_rows.device)[:nq]
                        valid = cb.shadow_doc_cols >= 0
                        n_sh = cb.shadow_rows.numel()
                        aux_ctx = {
                            "rows_q": lab_rows[sel],
                            "cols_q": lab_cols[sel],
                            "rows_kv": cb.shadow_rows[:, None].expand_as(cb.shadow_doc_cols)[valid],
                            "cols_kv": cb.shadow_doc_cols[valid],
                            "doc_of_kv": torch.arange(n_sh, device=lab_rows.device)[
                                :, None
                            ].expand_as(cb.shadow_doc_cols)[valid],
                            "rows_sh": cb.shadow_rows,
                            "cols_sh": cb.shadow_cols,
                            "log_len": cb.shadow_log_len,
                            "layers": [],
                        }
                        kwargs["aux_capture"] = aux_ctx
                if pst.get("len_bias") and cb.soft_rows.numel() > 0:
                    from ..pooled_soft_token import add_soft_len_bias, build_position_causal_bias

                    if kwargs.get("attn_bias") is None:
                        kwargs["attn_bias"] = build_position_causal_bias(
                            cb, dtype=self.embeddings.weight.dtype, device=input_ids.device  # type: ignore[union-attr]
                        )
                    kwargs["attn_bias"] = add_soft_len_bias(
                        kwargs["attn_bias"],
                        cb,
                        scale=pst.get("len_bias_scale", 1.0),
                        extra=pst.get("len_bias_extra", 0.0),
                    )
        # Role-gated FFN: gate mask from the FINAL token stream (post-compaction when the
        # soft-token path rewrote input_ids), so kept-doc tokens are gated in compacted rows too.
        if self._role_gated_ffn is not None:
            self._set_role_gate_mask(input_ids)
        # Nested-FFN router: reset per-forward accumulators and advance the budget/exploration
        # schedules. Loss terms are only collected when we are actually computing a loss.
        if self._nested_ffn_moe is not None:
            self._nested_ffn_moe["holder"].begin_forward(collect_loss=labels is not None)
        if self._ffn_token_drop is not None:
            self._ffn_token_drop["holder"].begin_forward(training=self.training)
        if self._kv_route is not None:
            self._kv_route["holder"].begin_forward(collect_loss=labels is not None)
        if self._block_skip is not None:
            self._block_skip["holder"].begin_forward(collect_loss=labels is not None)

        (
            input_ids,
            labels,
            all_block_kwargs,
            per_block_kwargs,
            lm_head_kwargs,
        ) = self._prepare_inputs(
            input_ids,
            labels,
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
            z_loss_multiplier=z_loss_multiplier,
            loss_div_factor=loss_div_factor,
            return_logits=return_logits,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        tx_plan: Optional[Dict[str, Any]] = None
        if tx_idx is not None:
            assert tx_full is not None and self._token_exit is not None
            # Full-row block kwargs for blocks [0, L): no position_ids (contiguous), so every
            # attention layer takes its normal backend path there.
            _, _, tx_abk, tx_pbk, _ = self._prepare_inputs(
                tx_full[0],
                None,
                ignore_index=ignore_index,
                loss_reduction=loss_reduction,
                z_loss_multiplier=z_loss_multiplier,
                loss_div_factor=loss_div_factor,
                return_logits=return_logits,
                logits_to_keep=logits_to_keep,
                **tx_full[1],
            )
            tx_plan = {
                "layer": self._token_exit.exit_layer,
                "idx": move_to_device(tx_idx, self.device),
                "all_block_kwargs": tx_abk,
                "per_block_kwargs": tx_pbk,
            }
            input_ids = move_to_device(tx_full[0], self.device)  # the embedding sees the full row

        # Get embeddings but pass-through for non-existent layers to allow easy
        # pipeline parallel configuration.
        if input_embeddings is not None:
            h = move_to_device(input_embeddings, self.device)
        else:
            h = self.embeddings(input_ids) if self.embeddings is not None else input_ids
            if getattr(self, "_online_token_router", None) is not None and self.training:
                # online router targets: per-token gradient at the (compacted) embedding output
                h = self._online_token_router.attach_target_hook(
                    self, h, all_block_kwargs.get("position_ids")
                )
            if soft_inject is not None:
                # Overwrite the placeholder embeddings with the projector's soft doc tokens. Under
                # detach_soft_kv the POOLED slots are injected detached (static-KV semantics; the
                # per-layer K/V cut happens in Attention via kv_grad_mask) while SHADOW candidates
                # stay attached so the aux objective can train the projector.
                s_rows, s_cols, mean_feats = soft_inject
                assert self.pooled_projector is not None
                h = h.clone()
                soft_vecs = self.pooled_projector(mean_feats).to(h.dtype)
                if self._pooled_soft_tokens["detach_soft_kv"]:  # type: ignore[index]
                    n_slots = s_rows.shape[0] - (
                        aux_ctx["rows_sh"].shape[0] if aux_ctx is not None else 0
                    )
                    soft_vecs = torch.cat(
                        [soft_vecs[:n_slots].detach(), soft_vecs[n_slots:]], dim=0
                    )
                h[s_rows, s_cols] = soft_vecs
            if self.embeddings is not None and self.embed_scale is not None:
                h = h * self.embed_scale
            if self.embedding_norm is not None:
                h = self.embedding_norm(h)

        # Run each block (capturing student hiddens at the distill layers on paired forwards).
        cap_set = (
            set(self._pooled_soft_tokens["distill_layers"])  # type: ignore[index]
            if distill_teacher is not None
            else None
        )
        h, s_caps = self._run_blocks(
            h, all_block_kwargs, per_block_kwargs, capture_layers=cap_set, token_exit=tx_plan
        )
        if getattr(self, "_online_token_router", None) is not None and self.training:
            labels = self._online_token_router.mid_labels(labels)  # no-op unless a mid-stack drop ran

        # Aux attention-contribution matching loss (soft-token pooling; see aux_matching_loss).
        aux_loss: Optional[torch.Tensor] = None
        if aux_ctx is not None and aux_ctx["layers"]:
            from ..pooled_soft_token import aux_matching_loss

            first_block = next(iter(self.blocks.values()))
            scale = first_block.attention.head_dim**-0.5  # type: ignore[union-attr]
            aux_loss = aux_matching_loss(
                [
                    (qs, kd, vd, aux_ctx["doc_of_kv"], ksh, vsh)
                    for (qs, kd, vd, ksh, vsh) in aux_ctx["layers"]
                ],
                q_rows=aux_ctx["rows_q"],
                shadow_rows=aux_ctx["rows_sh"],
                shadow_log_len=aux_ctx["log_len"],
                scale=scale,
            )

        # Get final logits but again pass-through in case of pipeline parallelism.
        if self.lm_head is not None:
            if self.compile_enabled:
                mark_dynamic(h, (0, 1), strict=False)
                if labels is not None:
                    mark_dynamic(labels, (0, 1), strict=False)
            # NOTE: When TP is active we can't pass 'labels=None' or the hook from 'PrepareModuleInput'
            # will throw an exception.
            if labels is not None:
                lm_head_kwargs["labels"] = labels
            out = self.lm_head(h, **lm_head_kwargs)
            if isinstance(out, LMOutputWithLoss) and out.loss is not None:
                loss = out.loss
                otr_x = getattr(self, "_online_token_router", None) if self.training else None
                x_pack = otr_x.pop_exit() if otr_x is not None else None
                if x_pack is not None and bool((x_pack[1] != lm_head_kwargs.get("ignore_index", -100)).any()):
                    # online router early-exit loss: the dropped tokens' split-layer states through the LM head
                    x_kw = {k: v for k, v in lm_head_kwargs.items() if k not in ("labels", "logits_to_keep")}
                    x_kw["return_logits"] = False
                    x_out = self.lm_head(x_pack[0][None], labels=x_pack[1][None], **x_kw)
                    loss = loss + otr_x.cfg.exit_loss_weight * x_out.loss.to(loss.dtype)
                if aux_loss is not None:
                    w = self._pooled_soft_tokens["aux_match_weight"]  # type: ignore[index]
                    loss = loss + w * aux_loss.to(loss.dtype)
                attach = self.budget_attach  # gradients already delivered per block
                if self._nested_ffn_moe is not None:
                    nffn_loss = self._nested_ffn_moe["holder"].regularization_loss()
                    if nffn_loss is not None and not attach:
                        loss = loss + nffn_loss.to(loss.dtype)
                if self._kv_route is not None:
                    kvr_loss = self._kv_route["holder"].regularization_loss()
                    if kvr_loss is not None and not attach:
                        loss = loss + kvr_loss.to(loss.dtype)
                if self._block_skip is not None:
                    bs_loss = self._block_skip["holder"].regularization_loss()
                    if bs_loss is not None and not attach:
                        loss = loss + bs_loss.to(loss.dtype)
                if self._joint_budget is not None:
                    from ..joint_budget import joint_budget_loss

                    jb_loss = joint_budget_loss(self)  # also records last_cost / last_target
                    if jb_loss is not None and not attach:
                        loss = loss + jb_loss.to(loss.dtype)
                if distill_teacher is not None and labels is not None:
                    # Map teacher (row, ORIGINAL position) -> student compacted column via the
                    # per-row ascending position_ids (free/answer tokens survive compaction).
                    pos_ids_s = all_block_kwargs.get("position_ids")
                    s_r = distill_teacher["rows"]
                    if pos_ids_s is not None:
                        s_c = (
                            torch.searchsorted(
                                pos_ids_s[s_r].contiguous(),
                                distill_teacher["pos"][:, None].contiguous(),
                            )
                            .squeeze(-1)
                            .clamp(max=pos_ids_s.shape[1] - 1)
                        )
                    else:
                        s_c = distill_teacher["pos"]
                    terms = []
                    for li, t_h in distill_teacher["caps"].items():
                        s_h = s_caps[li][s_r, s_c].float()
                        t_f = t_h.float()
                        terms.append(
                            ((s_h - t_f) ** 2).sum(-1).mean()
                            / t_f.pow(2).sum(-1).mean().clamp(min=1e-6)
                        )
                    d_loss = torch.stack(terms).mean()
                    w_d = self._pooled_soft_tokens["distill_weight"]  # type: ignore[index]
                    loss = loss + distill_teacher["loss_full"] + w_d * d_loss.to(loss.dtype)
                if loss is not out.loss:
                    out = LMOutputWithLoss(out.logits, loss, out.ce_loss, out.z_loss)
            return out
        else:
            return h

    def apply_fp8(self, float8_config: Float8Config):
        """
        Use an FP8 recipe on most linear layers.
        """
        if not float8_config.enabled:
            return

        modules_to_ignore = set()
        if self.lm_head is not None:
            modules_to_ignore.add("lm_head.w_out")
        if float8_config.modules_to_ignore is not None:
            modules_to_ignore.update(float8_config.modules_to_ignore)

        float8_config.apply_float8_linear(self, modules_to_ignore=modules_to_ignore)

        self._fp8_enabled = True
        self._precompute_float8_dynamic_scale_for_fsdp = (
            float8_config.should_precompute_float8_dynamic_scale_for_fsdp
        )

    def apply_pp(self, pp_mesh: DeviceMesh):
        """
        Prepare the model for pipeline parallelism after it's been split into stages.
        """
        for block in self.blocks.values():
            router = getattr(block, "routed_experts_router", None)
            if router is not None and getattr(router, "requires_segment_ids", False):
                raise OLMoConfigurationError(
                    "EMO routing does not currently support pipeline parallelism because "
                    "token-derived segment IDs are not carried between pipeline stages"
                )
        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_pp(pp_mesh)
        self._pp_enabled = True
        self._pp_group_size = pp_mesh.size()

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: Optional[bool] = None):
        """
        Apply tensor parallelism to the model.

        :param loss_parallel: Set to ``True`` if parallelizing the loss function as well.
        :param float8_enabled: Set this to ``True`` if training with float8 linear layers.
        """
        for block in self.blocks.values():
            router = getattr(block, "routed_experts_router", None)
            if router is not None and getattr(router, "requires_segment_ids", False):
                raise OLMoConfigurationError(
                    "EMO routing does not currently support tensor parallelism because "
                    "document pools require the complete token sequence"
                )

        if self.tie_word_embeddings and (
            self.lm_head is None
            or self.lm_head.loss_implementation == LMLossImplementation.fused_linear
        ):
            raise NotImplementedError(
                "Tensor parallelism with tied word embeddings requires the default loss "
                "implementation; the fused-linear loss replicates the LM head weight, which is "
                "incompatible with the vocab-sharded embedding."
            )

        if float8_enabled is None:
            float8_enabled = self.fp8_enabled
        elif not float8_enabled and self.fp8_enabled:
            raise OLMoConfigurationError(
                "Got 'float8_enabled=False', but FP8 has already been enabled"
            )

        if self.embeddings is not None:
            parallelize_module(
                self.embeddings,
                device_mesh=tp_mesh,
                parallelize_plan=RowwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                    use_local_output=False,
                ),
            )
        if self.embedding_norm is not None:
            parallelize_module(
                self.embedding_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
            )

        # Apply tensor/sequence parallelism to every transformer block.
        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_tp(tp_mesh, input_layout=Shard(1), float8_enabled=float8_enabled)

        if self.lm_head is not None:
            self.lm_head.apply_tp(tp_mesh, input_layouts=(Shard(1), Replicate()))

        # The embedding (RowwiseParallel) and the LM head (ColwiseParallel) both shard their
        # weight along the vocab dimension, so re-point the head at the embedding's sharded
        # parameter to restore the tie that `parallelize_module` broke.
        if self.tie_word_embeddings and self.embeddings is not None and self.lm_head is not None:
            self._tie_weights()

        self._tp_enabled = True
        self._tp_mesh = tp_mesh

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: RingContextParallelStyle | None = None,
        uly: UlyssesContextParallelStyle | None = None,
    ):
        """
        Prepare the model for context-parallelism (CP).

        :param cp_mesh: The CP device mesh.
        :param ring: The ring context parallel style.
        :param uly: The ulysses context parallel style.
        """
        if ring is not None:
            self._cp_load_balancer = ring.load_balancer.build(cp_mesh)
        elif uly is not None:
            self._cp_load_balancer = uly.load_balancer.build(cp_mesh)

        for block in self.blocks.values():
            cast(TransformerBlockBase, block).apply_cp(cp_mesh, ring=ring, uly=uly)
        if self.lm_head is not None:
            self.lm_head.apply_cp(cp_mesh)

    def apply_activation_checkpointing(
        self,
        mode: TransformerActivationCheckpointingMode,
        block_interval: Optional[int] = None,
        modules: Optional[List[str]] = None,
        activation_memory_budget: Optional[float] = None,
        determinism_check: str = "default",
    ):
        """
        Apply activation checkpointing to the model.

        :param mode: Determines how to apply activation checkpointing.
        :param block_interval: Required when :data:`mode` is "selected_blocks". Determines
            which blocks are wrapped.
        :param modules: Required when :data:`mode` is "selected_modules". A list of modules names
            to wrap for activation checkpointing. Globs are supported.
        :param activation_memory_budget: The memory budget for activation checkpointing in the range
            [0, 1]. 0 corresponds to the memory usage when recomputing all activations, and 1
            corresponds to the memory usage when recomputing no activations (which is the default).
            Requires compilation to be enabled.
        :param determinism_check: Passed through to torch's ``checkpoint_wrapper``. "default" compares
            forward vs. recompute tensor metadata; "none" skips the check.
        """

        if mode == TransformerActivationCheckpointingMode.budget:
            if activation_memory_budget is None:
                raise ValueError("'activation_memory_budget' is required for 'budget' mode")
            if activation_memory_budget < 0 or activation_memory_budget > 1:
                raise ValueError("'activation_memory_budget' must be in the range [0, 1]")
            torch._functorch.config.activation_memory_budget = activation_memory_budget
            return

        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper as ptd_checkpoint_wrapper,
        )

        if (
            mode == TransformerActivationCheckpointingMode.selected_blocks
            and block_interval is None
        ):
            raise ValueError("'block_interval' is required for 'selected_blocks' mode")

        if mode == TransformerActivationCheckpointingMode.selected_modules and modules is None:
            raise ValueError("'modules' is required for 'selected_modules' mode")

        # EMO samples a routed-expert pool size for each document. Recompute must replay those
        # samples so it routes through the same experts as the original checkpointed forward.
        # TODO: also preserve RNG state if dropout is active.
        preserve_rng_state = any(
            getattr(
                getattr(block, "routed_experts_router", None),
                "requires_segment_ids",
                False,
            )
            for block in self.blocks.values()
        )

        if mode == TransformerActivationCheckpointingMode.selected_modules:
            from fnmatch import fnmatch

            assert modules is not None
            wrapped_modules: Set[str] = set()
            for name, module in self.named_modules():
                for pattern in modules:
                    if fnmatch(name, pattern):
                        break
                else:
                    continue

                if isinstance(module, MoEBase):
                    raise OLMoConfigurationError(
                        "Wrapping an entire MoE module for activation checkpointing is not supported. "
                        "Please try a finer-grained wrapping strategy."
                    )

                # NOTE: have to be careful not to try to wrap submodules of modules that have been wrapped.
                parent_name = ".".join(name.split(".")[:-1])
                if parent_name in wrapped_modules:
                    continue

                parent = self if not parent_name else self.get_submodule(parent_name)
                module = ptd_checkpoint_wrapper(
                    module,
                    preserve_rng_state=preserve_rng_state,
                    determinism_check=determinism_check,
                )
                parent.register_module(name.split(".")[-1], module)
                log.info(f"Wrapped '{name}' for activation checkpointing")
                wrapped_modules.add(name)
        else:
            for block_idx, block in enumerate(self.blocks.values()):
                if mode == TransformerActivationCheckpointingMode.selected_blocks:
                    assert block_interval is not None
                    if block_idx % block_interval == 0:
                        if isinstance(block, MoETransformerBlock):
                            raise OLMoConfigurationError(
                                "Wrapping MoE blocks for activation checkpointing is not supported."
                            )
                        block = ptd_checkpoint_wrapper(
                            block,
                            preserve_rng_state=preserve_rng_state,
                            determinism_check=determinism_check,
                        )
                elif mode == TransformerActivationCheckpointingMode.full:
                    if isinstance(block, MoETransformerBlock):
                        raise OLMoConfigurationError(
                            "Wrapping MoE blocks for activation checkpointing is not supported."
                        )
                    block = ptd_checkpoint_wrapper(
                        block,
                        preserve_rng_state=preserve_rng_state,
                        determinism_check=determinism_check,
                    )
                elif mode == TransformerActivationCheckpointingMode.selected_ops:
                    block = ptd_checkpoint_wrapper(
                        block,
                        context_fn=selective_checkpointing_context_fn,
                        preserve_rng_state=preserve_rng_state,
                        determinism_check=determinism_check,
                    )

                self.blocks.register_module(str(block_idx), block)

    def apply_compile(self):
        """
        Apply ``torch.compile()`` to each transformer block, which makes compilation efficient
        due to repeated structure.

        .. warning::
            This must be called after :meth:`apply_activation_checkpointing()` but
            before :meth:`apply_fsdp()` or :meth:`apply_ddp()`.
        """
        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_compile()

        if self.lm_head is not None:
            self.lm_head.compile(fullgraph=False)

        torch.compiler.config.dynamic_sources += "L['kwargs']['max_doc_len'],"
        self._compile_enabled = True

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        param_dtype: Optional[torch.dtype] = None,
        reduce_dtype: torch.dtype = torch.float32,
        pp_enabled: bool = False,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
    ):
        """
        Apply FSDP(2) to the model.

        .. warning::
            This should generally be called last if using any other parallelism strategies or optimizations
            like :meth:`apply_compile()`.

        :param dp_mesh: The model data parallel device mesh.
        :param param_dtype: The data type to materialize params in. Defaults to the current param dtype.
        :param reduce_dtype: The data type for gradient reduction.
        :pp_enabled: If pipeline parallelism is also enabled.
        :prefetch_factor: For tuning the prefetch settings. 0 is the default, and higher values result
            in more aggressive prefetching.
        :wrapping_strategy: The wrapping strategy.
        """
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype or self.dtype, reduce_dtype=reduce_dtype
        )
        # NOTE: blocks get a policy with 'cast_forward_inputs' DISABLED, which is required for
        # correctness whenever activation checkpointing wraps a block (the AC wrapper sits
        # *outside* the block, so FSDP's pre-forward hook runs *inside* the checkpointed region).
        # FSDP2 only casts forward inputs in 'FSDPState._pre_forward', which early-returns when
        # the state is 'PRE_BACKWARD' -- i.e. exactly during the AC recompute. So any float32
        # tensor passed to a block as an argument reaches the block as 'param_dtype' in the
        # forward but as float32 in the recompute. Under context parallelism that's precisely the
        # CP-sharded RoPE buffers ('pos_sin'/'pos_cos', built in fp32 by '_prepare_inputs'), and
        # since torch.compile guards on input dtypes the recompute recompiles the block into a
        # differently-partitioned graph that saves a different sequence of tensors. The result is
        #   CheckpointError: torch.utils.checkpoint: Recomputed values for the following tensors
        #   have different metadata than during the forward pass.
        # on the very first backward (seen on 128k-context CP=2 runs; see
        # 'debug/cp_ac_rope_dtype/').
        # Disabling the cast is safe: the hidden state entering a block is already in
        # 'param_dtype' (the embedding layer is sharded with the same policy and every layer norm
        # preserves its input dtype), and the RoPE buffers are meant to stay in fp32 anyway --
        # 'RotaryEmbedding.forward' casts them itself with 'type_as'.
        block_mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype or self.dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=False,
        )
        fsdp_config = dict(mesh=dp_mesh, mp_policy=mp_policy)
        # For PP, do not reshard after forward to avoid per-microbatch all-gathers,
        # which can be expensive and non-overlapped
        reshard_after_forward = False if pp_enabled else True

        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_fsdp(
                dp_mesh=dp_mesh,
                prefetch_factor=prefetch_factor,
                wrapping_strategy=wrapping_strategy,
                reshard_after_forward=reshard_after_forward,
                mp_policy=block_mp_policy,
            )

        # When weights are tied the embeddings and LM head share a parameter, so they must
        # stay in the same FSDP group (the root) rather than being sharded separately.
        if self.embeddings is not None and not self.tie_word_embeddings:
            fully_shard(
                self.embeddings,
                reshard_after_forward=reshard_after_forward,
                **fsdp_config,
            )
            # Embedding params are not needed for backwards computation.
            cast(FSDPModule, self.embeddings).set_unshard_in_backward(False)

        if wrapping_strategy != TransformerDataParallelWrappingStrategy.blocks:
            if self.embedding_norm is not None:
                fully_shard(self.embedding_norm, **fsdp_config)
            if self.lm_head is not None and not self.tie_word_embeddings:
                fully_shard(self.lm_head, reshard_after_forward=False, **fsdp_config)

        fully_shard(self, reshard_after_forward=reshard_after_forward, **fsdp_config)
        # Some inputs need to be on CPU initially, but FSDP will move everything to model's
        # device if we don't hide it.
        self.register_forward_pre_hook(_hide_cpu_inputs_from_torch, prepend=True, with_kwargs=True)
        self.register_forward_pre_hook(
            _unhide_cpu_inputs_from_torch, prepend=False, with_kwargs=True
        )

        if prefetch_factor > 0:
            blocks = cast(List[FSDPModule], list(self.blocks.values()))
            for i in range(len(blocks)):
                block = blocks[i]
                if i + 1 < len(blocks):
                    block.set_modules_to_forward_prefetch(blocks[i + 1 : i + 1 + prefetch_factor])
                elif isinstance(self.lm_head, FSDPModule):
                    block.set_modules_to_forward_prefetch([self.lm_head])

        self._fsdp_enabled = True

    def apply_ddp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        param_dtype: Optional[torch.dtype] = None,
        compile_enabled: bool = False,
        autograd_compile_enabled: bool = False,
    ):
        """
        Apply DDP to the model.
        """
        from torch.distributed._composable.replicate import replicate

        # Cast model explicitly to the specified dtype before applying DDP
        target_dtype = param_dtype or self.dtype
        if target_dtype != self.dtype:
            self.to(dtype=target_dtype)

        # Adapted from
        # https://github.com/pytorch/torchtitan/blob/90c889e972b56b9faadebbb78fc985dedc537ed9/torchtitan/parallelisms/parallelize_llama.py#L328
        if compile_enabled:
            if autograd_compile_enabled:
                torch._dynamo.config.optimize_ddp = "python_reducer_without_compiled_forward"  # type: ignore
            else:
                torch._dynamo.config.optimize_ddp = "ddp_optimizer"  # type: ignore

        replicate(self, device_mesh=dp_mesh, bucket_cap_mb=100)
        # Some inputs need to be on CPU initially, but DDP will move everything to model's
        # device if we don't hide it.
        self.register_forward_pre_hook(_hide_cpu_inputs_from_torch, prepend=True, with_kwargs=True)
        self.register_forward_pre_hook(
            _unhide_cpu_inputs_from_torch, prepend=False, with_kwargs=True
        )

    @cached_property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @cached_property
    def num_non_embedding_params(self) -> int:
        return self.num_params - self.embeddings.weight.numel()

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Returns the idealized number of flops per token for the given sequence length. Purposefully
        does not account for wasted flops due to padding, recomputation, etc.
        """
        flops_per_token = 0
        blocks = cast(List[TransformerBlockBase], list(self.blocks.values()))
        for block in blocks:
            flops_per_token += block.num_flops_per_token(seq_len)
        if self.lm_head is not None:
            flops_per_token += self.lm_head.num_flops_per_token(seq_len)
        return flops_per_token

    def post_batch(self, dry_run: bool = False):
        """
        Should be called right after the final backward of a complete batch but before the optimizer step.
        """
        del dry_run

    def post_optim_step(self):
        """
        Should be called right after an optimizer step.
        """
        if self.fp8_enabled and self._precompute_float8_dynamic_scale_for_fsdp:
            from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

            precompute_float8_dynamic_scale_for_fsdp(self)


@beta_feature
class NormalizedTransformer(Transformer):
    """
    A nGPT transformer implementation, to be used with the :class:`NormalizedTransformerBlock` block
    type.
    """

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        block: TransformerBlockConfig | dict[str, TransformerBlockConfig],
        lm_head: LMHeadConfig,
        dtype: torch.dtype = torch.float32,
        init_method: InitMethod = InitMethod.normalized,
        init_device: str = "cpu",
        init_seed: int = 0,
        init_std: float = 0.02,
        embedding_init_std: Optional[float] = None,
        block_overrides: Optional[Dict[int, TransformerBlockConfig]] = None,
        block_pattern: Optional[List[str]] = None,
    ):
        super().__init__(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            block=block,
            lm_head=lm_head,
            dtype=dtype,
            init_method=init_method,
            init_device=init_device,
            init_seed=init_seed,
            init_std=init_std,
            embedding_init_std=embedding_init_std,
            block_overrides=block_overrides,
            block_pattern=block_pattern,
        )

    def _validate_block(self, block: TransformerBlockBase) -> TransformerBlockBase:
        if not isinstance(block, NormalizedTransformerBlock):
            raise OLMoConfigurationError(
                f"'{self.__class__.__name__}' requires a '{NormalizedTransformerBlock.__name__}' block"
            )
        return block

    @torch.no_grad()
    def init_weights(self, *args, **kwargs) -> torch.Generator:
        generator = super().init_weights(*args, **kwargs)
        self.normalize_matrices()
        return generator

    @torch.no_grad()
    def normalize_matrices(self):
        """
        Normalize the weights in all matrices. This should be called after each optimizer step, which
        the :class:`~olmo_core.train.train_module.TransformerTrainModule` will handle for you.
        """
        if self.embeddings is not None:
            self._normalize_matrix(self.embeddings.weight)

        for block in self.blocks.values():
            if hasattr(block, "normalize_matrices"):
                block.normalize_matrices()  # type: ignore

        if self.lm_head is not None:
            self.lm_head.normalize_matrices()  # type: ignore

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(l2_normalize(w, dim=dim))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        float8_enabled: Optional[bool] = None,
    ):
        del tp_mesh, float8_enabled

        raise NotImplementedError(
            "TP is not implemented yet for the normalized transformer variant"
        )

    def apply_compile(self):
        super().apply_compile()
        self.normalize_matrices = torch.compile(self.normalize_matrices)

    def post_optim_step(self):
        super().post_optim_step()
        self.normalize_matrices()


@beta_feature
class MoETransformer(Transformer):
    """
    An MoE transformer implementation, to be used with one of the
    :class:`MoETransformerBlock` block types.
    """

    @property
    def is_moe(self) -> bool:
        return True

    def compute_auxiliary_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        from olmo_core.train.common import ReduceType

        mean_offset = 1.0
        if self.pp_enabled:
            # Change the divisor to 'world_size // pp_group_size'
            mean_offset = self._pp_group_size

        out: Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]] = {}
        for block_idx, block in self.blocks.items():
            if not block.is_moe:
                continue
            block = cast(MoETransformerBlock, block)
            block_metrics = block.compute_metrics(reset=reset)
            for metric_name, (metric_val, reduce_type) in block_metrics.items():
                out[f"block {int(block_idx):02d}/{metric_name}"] = (
                    metric_val,
                    reduce_type,
                )

                if self.pp_enabled and reduce_type == ReduceType.mean:
                    metric_val = metric_val.float() * mean_offset

                if metric_name not in out:
                    out[metric_name] = (metric_val, reduce_type)
                elif reduce_type in (ReduceType.mean, ReduceType.sum):
                    out[metric_name] = (
                        out[metric_name][0] + metric_val,
                        reduce_type,
                    )
                elif reduce_type == ReduceType.max:
                    out[metric_name] = (
                        torch.max(out[metric_name][0], metric_val),
                        reduce_type,
                    )
                else:
                    raise NotImplementedError(reduce_type)
        return out

    def reset_auxiliary_metrics(self):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            cast(MoETransformerBlock, block).reset_metrics()

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            block = cast(MoETransformerBlock, block)
            block.apply_ep(ep_mesh, **kwargs)

    def prepare_experts_for_fsdp(
        self,
        world_mesh: DeviceMesh,
        param_dtype: Optional[torch.dtype] = None,
        reduce_dtype: torch.dtype = torch.float32,
        pp_enabled: bool = False,
    ):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            block = cast(MoETransformerBlock, block)
            reshard_after_forward = True
            if pp_enabled or block.ep_enabled or block.tp_enabled:
                reshard_after_forward = False
            block.feed_forward_moe.prepare_experts_for_fsdp(
                world_mesh=world_mesh,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=param_dtype or self.dtype, reduce_dtype=reduce_dtype
                ),
                reshard_after_forward=reshard_after_forward,
            )

    def prepare_experts_for_ddp(self, world_mesh: DeviceMesh):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            cast(MoETransformerBlock, block).feed_forward_moe.prepare_experts_for_ddp(
                world_mesh=world_mesh,
            )

    def post_batch(self, dry_run: bool = False):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            block = cast(MoETransformerBlock, block)
            block.feed_forward_moe.post_batch(dry_run=dry_run)


def _hide_cpu_inputs_from_torch(m, args, kwargs) -> Optional[Tuple[Any, Dict[str, Any]]]:
    del m
    if (doc_lens := kwargs.get("doc_lens")) is not None:
        kwargs["doc_lens"] = hide_from_torch(doc_lens)
    return (args, kwargs)


def _unhide_cpu_inputs_from_torch(m, args, kwargs) -> Optional[Tuple[Any, Dict[str, Any]]]:
    del m
    if (doc_lens := kwargs.get("doc_lens")) is not None:
        kwargs["doc_lens"] = unhide_from_torch(doc_lens)
    return (args, kwargs)
