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
    FusedAttention,
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
from ..lm_head import LeakyLMOutputWithLoss, LMHeadConfig, LMOutputWithLoss
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
            if isinstance(block.attention, (Attention, FusedAttention)):
                rope = cast(Optional[RotaryEmbeddingBase], block.attention.rope)
                rope_buffers[int(key)] = None if rope is None else rope.get_buffers(seq_len, device)
            else:
                rope_buffers[int(key)] = None
        return rope_buffers

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
                std=self.embedding_init_std
                if self.embedding_init_std is not None
                else self.init_std,
                generator=generator,
            )

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

            if isinstance(att, (Attention, FusedAttention)):
                # Warm up attention backend cache.
                if max_seq_len is not None and att.backend is not None:
                    att.backend.warmup_cache(max_seq_len, device)

                # Warm up RoPE cache.
                if max_seq_len is not None and att.rope is not None:
                    att.rope.warmup_cache(max_seq_len, device)

        if self.lm_head is not None:
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

        output_mask = kwargs.pop("output_mask", None)
        cpt_loss_scaler = kwargs.pop("cpt_loss_scaler", None)

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
            inputs = [input_ids]
            seq_dims = [1]
            pad_values: List[Union[int, float]] = [0]
            keys = ["input_ids"]

            # NOTE: initialize buffer(s) on CPU to avoid possible host-device sync when sharding.
            for block_idx, rope_buffers in self.get_rope_buffers(S, torch.device("cpu")).items():
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

            if labels is not None:
                inputs.append(labels)
                seq_dims.append(1)
                pad_values.append(ignore_index)
                keys.append("labels")

            if output_mask is not None:
                inputs.append(output_mask)
                seq_dims.append(1)
                pad_values.append(False)
                keys.append("output_mask")

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

            input_ids = all_block_kwargs.pop("input_ids")
            labels = all_block_kwargs.pop("labels", None)
            output_mask = all_block_kwargs.pop("output_mask", None)
        else:
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

        if "cu_doc_lens" in all_block_kwargs:
            mark_dynamic(all_block_kwargs["cu_doc_lens"], 0, strict=False)  # type: ignore[arg-type]

        if cpt_loss_scaler is not None:
            lm_head_kwargs["cpt_loss_scaler"] = cpt_loss_scaler
            lm_head_kwargs["output_mask"] = move_to_device(output_mask, self.device)

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
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[torch.Tensor, LMOutputWithLoss, LeakyLMOutputWithLoss]:
        """
        Run the transformer on the token input IDs.

        :param input_ids: The token input IDs, shape ``(batch_size, seq_len)``.
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

        # Get embeddings but pass-through for non-existent layers to allow easy
        # pipeline parallel configuration.
        h = self.embeddings(input_ids) if self.embeddings is not None else input_ids
        if self.embeddings is not None and self.embed_scale is not None:
            h = h * self.embed_scale
        if self.embedding_norm is not None:
            h = self.embedding_norm(h)

        # Run each block.
        for block_key, block in self.blocks.items():
            block_idx = int(block_key)
            block_kwargs = per_block_kwargs.get(block_idx, {})
            # Mark sizes as dynamic for torch.compile().
            if self.compile_enabled:
                mark_dynamic(h, (0, 1), strict=False)
            h = block(h, **all_block_kwargs, **block_kwargs)

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
            return self.lm_head(h, **lm_head_kwargs)
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

        # TODO: only preserve RNG state if dropout is active
        preserve_rng_state = False

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
                module = ptd_checkpoint_wrapper(module, preserve_rng_state=preserve_rng_state)
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
                        block = ptd_checkpoint_wrapper(block, preserve_rng_state=preserve_rng_state)
                elif mode == TransformerActivationCheckpointingMode.full:
                    if isinstance(block, MoETransformerBlock):
                        raise OLMoConfigurationError(
                            "Wrapping MoE blocks for activation checkpointing is not supported."
                        )
                    block = ptd_checkpoint_wrapper(block, preserve_rng_state=preserve_rng_state)
                elif mode == TransformerActivationCheckpointingMode.selected_ops:
                    block = ptd_checkpoint_wrapper(
                        block,
                        context_fn=selective_checkpointing_context_fn,
                        preserve_rng_state=preserve_rng_state,
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
                mp_policy=mp_policy,
            )

        if self.embeddings is not None:
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
            if self.lm_head is not None:
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
