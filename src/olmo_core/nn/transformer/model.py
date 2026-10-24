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
from olmo_core.nn.embedding import SplitVocabEmbedding
from olmo_core.nn.residual_stream import ResidualStream
from olmo_core.utils import get_default_device, mark_dynamic, move_to_device

from ..attention import Attention, RingAttentionLoadBalancer, SequenceMixer
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
from .init import InitMethod, _apply_init

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
        n_extra_vocab: int = 0,
    ):
        super().__init__()

        cache = BufferCache()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dtype = dtype
        self.embed_scale = embed_scale

        # With extra tokens the table is split into two parameters (see
        # :class:`SplitVocabEmbedding`): ``vocab_size`` is then the *base* vocab, and the
        # tied LM head spans only that base — the extra tokens are inputs, never targets.
        self.n_extra_vocab = n_extra_vocab
        self.embeddings: nn.Module
        if n_extra_vocab > 0:
            self.embeddings = SplitVocabEmbedding(
                vocab_size, n_extra_vocab, d_model, dtype=dtype, device=init_device
            )
        else:
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
            if isinstance(self.embeddings, SplitVocabEmbedding):
                # mm_olmo initialises the added rows from `new_embedding_init_range`, which
                # matches our `init_std` default of 0.02.
                _apply_init(
                    nn.init.normal_,
                    self.embeddings.extra_weight,
                    generator=generator,
                    std=(
                        self.embedding_init_std
                        if self.embedding_init_std is not None
                        else self.init_std
                    ),
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
        response_logits_only: bool = False,
        response_mask: Optional[torch.Tensor] = None,
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

        if response_mask is not None:
            response_mask = move_to_device(response_mask, self.device)
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
            response_logits_only=response_logits_only,
            response_mask=response_mask,
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
        response_logits_only: bool = False,
        response_mask: Optional[torch.Tensor] = None,
        or_mask: Optional[torch.Tensor] = None,
        and_mask: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        flex_attn_is_image: Optional[torch.Tensor] = None,
        flex_attn_subsegment_ids: Optional[torch.Tensor] = None,
        flex_attn_example_ids: Optional[torch.Tensor] = None,
        flex_attn_block_mask: Optional[torch.Tensor] = None,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
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
        :param response_logits_only: If True, only compute logits at ``response_mask`` positions.
        :param response_mask: Boolean mask ``(batch_size, seq_len)`` for ``response_logits_only``.

        :returns: The logits if ``labels`` is ``None`` or the losses if ``labels`` is not ``None``.
        """
        if input_embeddings is not None and self._cp_load_balancer is not None:
            raise RuntimeError(
                "`input_embeddings` is not supported with context parallelism: `_prepare_inputs` "
                "shards `input_ids`/`labels`/RoPE while `input_embeddings` stays full-size, which "
                "would misalign the hidden states."
            )
        if or_mask is not None and self._cp_load_balancer is not None:
            raise RuntimeError(
                "`or_mask` is not supported with context parallelism: the full-size "
                "(seq, seq) mask would misalign with the sequence-sharded hidden states."
            )
        if and_mask is not None and self._cp_load_balancer is not None:
            raise RuntimeError(
                "`and_mask` is not supported with context parallelism: the full-size "
                "(seq, seq) mask would misalign with the sequence-sharded hidden states."
            )
        if position_ids is not None and self._cp_load_balancer is not None:
            raise RuntimeError(
                "Explicit `position_ids` are not supported with context parallelism, "
                "which shards/derives its own RoPE positions."
            )

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
            response_logits_only=response_logits_only,
            response_mask=response_mask,
            **kwargs,
        )

        # Bidirectional / custom attention allow-mask (e.g. for image tokens),
        # OR'd onto the causal base inside each block's attention. Passed through
        # to every block; only the dense SDPA backend honors it.
        if or_mask is not None:
            all_block_kwargs["or_mask"] = move_to_device(or_mask, self.device)

        # Restrictive (AND) attention mask, e.g. subsegment / branch isolation in packed
        # multi-annotation multimodal data. AND'd onto the (causal | or_mask) base inside
        # each block's attention; only the dense SDPA backend honors it.
        if and_mask is not None:
            all_block_kwargs["and_mask"] = move_to_device(and_mask, self.device)

        if flex_attn_is_image is not None:
            all_block_kwargs["flex_attn_is_image"] = move_to_device(flex_attn_is_image, self.device)
        if flex_attn_subsegment_ids is not None:
            all_block_kwargs["flex_attn_subsegment_ids"] = move_to_device(
                flex_attn_subsegment_ids, self.device
            )
        if flex_attn_example_ids is not None:
            all_block_kwargs["flex_attn_example_ids"] = move_to_device(
                flex_attn_example_ids, self.device
            )
        if flex_attn_block_mask is not None:
            all_block_kwargs["flex_attn_block_mask"] = flex_attn_block_mask

        # Explicit per-token RoPE positions (e.g. parallel branches that share an
        # overlapping position range). Passed through to every attention block.
        if position_ids is not None:
            all_block_kwargs["position_ids"] = move_to_device(position_ids, self.device)

        # Externally-provided RoPE buffers (e.g. M-RoPE / multimodal 3D rotary). When
        # supplied, every attention block uses these instead of computing 1D RoPE from
        # sequential positions. Sequence-mixers that ignore RoPE (e.g. GatedDeltaNet)
        # simply drop these via ``**kwargs``. Not supported with context parallelism,
        # which shards/derives its own RoPE buffers.
        if pos_sin is not None or pos_cos is not None:
            if self._cp_load_balancer is not None:
                raise RuntimeError(
                    "External `pos_sin`/`pos_cos` (e.g. M-RoPE) are not supported with "
                    "context parallelism."
                )
            if pos_sin is not None:
                all_block_kwargs["pos_sin"] = move_to_device(pos_sin, self.device)
            if pos_cos is not None:
                all_block_kwargs["pos_cos"] = move_to_device(pos_cos, self.device)

        # Get embeddings but pass-through for non-existent layers to allow easy
        # pipeline parallel configuration.
        if input_embeddings is not None:
            h = move_to_device(input_embeddings, self.device)
        else:
            h = self.embeddings(input_ids) if self.embeddings is not None else input_ids
            if self.embeddings is not None and self.embed_scale is not None:
                h = h * self.embed_scale
            if self.embedding_norm is not None:
                h = self.embedding_norm(h)

        # Per-token residual dropout mask (Molmo2's `response_residual_dropout`). Only
        # forwarded when supplied, so blocks that do not accept it are unaffected.
        if drop_mask is not None:
            all_block_kwargs["drop_mask"] = drop_mask

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

    def _dropout_is_active(self) -> bool:
        """
        Whether any submodule applies dropout, and therefore whether activation checkpointing has
        to preserve the RNG state across recomputation.

        Covers both :class:`~olmo_core.nn.residual_stream.ResidualStream` (whose ``masked_dropout``
        drives Molmo2's ``response_residual_dropout``) and plain :class:`torch.nn.Dropout`.
        """
        for module in self.modules():
            if isinstance(module, ResidualStream):
                if module.p > 0.0 or module.masked_dropout > 0.0:
                    return True
            elif isinstance(module, nn.Dropout) and module.p > 0.0:
                return True
        return False

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

        # Recomputation must replay the *same* random draws the forward pass used, otherwise the
        # gradients are taken with respect to a different sample than the loss was. Saving and
        # restoring the RNG state costs a little per checkpointed block, so only pay it when needed:
        # - Dropout is active (mirrors mm_olmo's `llm_activation_checkpoint_function`).
        # - EMO samples a routed-expert pool size for each document, and recompute must route
        #   through the same experts as the original checkpointed forward.
        preserve_rng_state = self._dropout_is_active() or any(
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
