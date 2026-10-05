import os
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from olmo_core.config import Config, DType
from olmo_core.distributed.utils import barrier, is_distributed
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.embedding import SplitVocabEmbedding
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerConfig
from olmo_core.nn.vision.config import VisionEncoderConfig
from olmo_core.nn.vision.connector import (
    ImagePoolingType,
    ImageProjectorType,
    VisionConnectorConfig,
)
from olmo_core.nn.vision.molmo2_tokens import IM_PATCH_ID

__all__ = [
    "MOLMO2_BASE_VOCAB_SIZE",
    "MOLMO2_N_EXTRA_TOKENS",
    "MOLMO2_VOCAB_SIZE",
    "MultimodalLMConfig",
    "MultimodalLM",
]

MOLMO2_BASE_VOCAB_SIZE = 151_936
"""Molmo2's base text vocab (Qwen3's) — the number of IDs the model may *predict*."""

MOLMO2_N_EXTRA_TOKENS = 128
"""Molmo2's inputs-only image-special tokens, held in a separate embedding block."""

MOLMO2_VOCAB_SIZE = MOLMO2_BASE_VOCAB_SIZE + MOLMO2_N_EXTRA_TOKENS
"""Molmo2's total *input* vocab (base + extra); the LM head spans only the base."""


def _molmo2_attn_backend():
    """Attention backend for the Molmo2 LM, matching the HF loader's rule.

    The multimodal masks (bidirectional image tokens + subsegment branch isolation) only run
    on the dense ``torch`` backend or the fused ``flex`` one; ``OLMO2_FLEX_ATTN=1`` selects
    FlexAttention (much faster for these masks, needs a GPU + ``torch.compile``).
    """
    from olmo_core.nn.attention import AttentionBackendName

    return (
        AttentionBackendName.flex
        if os.environ.get("OLMO2_FLEX_ATTN") == "1"
        else AttentionBackendName.torch
    )


@dataclass
class MultimodalLMConfig(Config):
    """
    Configuration for a multimodal (vision-language) transformer.

    Composes a language model (:class:`~olmo_core.nn.transformer.TransformerConfig`),
    a vision encoder (:class:`~olmo_core.nn.vision.VisionEncoderConfig`), and
    a vision-to-language connector
    (:class:`~olmo_core.nn.vision.VisionConnectorConfig`) into a single module
    that splices projected image features into the LM embedding stream.

    Example::

        lm_cfg = TransformerConfig.olmo2_1M(vocab_size=50000)
        vis_cfg = VisionEncoderConfig()           # CLIP ViT-L/14-336
        conn_cfg = VisionConnectorConfig.from_vision_encoder(vis_cfg, output_dim=lm_cfg.d_model)
        cfg = MultimodalLMConfig(
            lm=lm_cfg, vision=vis_cfg, connector=conn_cfg, image_patch_token_id=49152,
        )
        model = cfg.build()
    """

    lm: TransformerConfig
    """Language model configuration."""

    vision: VisionEncoderConfig
    """Vision encoder configuration."""

    connector: VisionConnectorConfig
    """Vision-to-language connector configuration."""

    image_patch_token_id: int = 0
    """
    LM vocabulary ID of the ``<im_patch>`` placeholder token. Positions in
    ``input_ids`` matching this ID receive projected image features via ``+=``
    during the forward pass. The data preprocessor is responsible for ensuring
    the number of occurrences of this token in the sequence matches the number
    of pooled image features the connector produces.
    """

    vit_layers: Tuple[int, ...] = (-1,)
    """
    Indices of the ViT hidden-state layers to extract and concatenate before
    the connector. Negative indices count from the last layer. For example,
    ``(-1,)`` uses only the final layer; a two-layer selection such as ``(-2, -9)``
    requires :attr:`connector.num_input_layers` to be ``2``.
    """

    output_vocab_size: Optional[int] = None
    """
    Number of token IDs the model may *predict*. Molmo2 extends the base text vocab
    with extra image-special tokens (``<im_patch>``, ``<im_start>``, …) that are
    **inputs-only**: HF Molmo2's ``lm_head`` covers just the base vocab, so the extra
    IDs can never be sampled and never enter the softmax. Our LM head spans the full
    ``lm.vocab_size``, so when this is set (to the base vocab size), the forward pass
    masks logit columns ``>= output_vocab_size`` to the dtype minimum — they contribute
    exactly ``0`` to the softmax and receive exactly ``0`` gradient, reproducing a
    base-vocab-only head for loss, sampling, and (tied-embedding) training dynamics.
    ``None`` (default) disables masking.
    """

    @classmethod
    def _molmo2_like(
        cls,
        lm: TransformerConfig,
        *,
        connector_mlp_hidden_size: int,
        response_residual_dropout: float = 0.0,
        **kwargs,
    ) -> "MultimodalLMConfig":
        """
        Shared Molmo2 body: a SigLIP2-SO400M/14-378 ViT truncated to 25 of its 27 blocks
        and a two-layer attention-pooling connector. Identical across the released
        variants; only the LM and the connector's MLP width differ.

        :param lm: The already-built language-model config.
        :param connector_mlp_hidden_size: HF ``adapter_config.intermediate_size``. Equal to
            the LM's feed-forward hidden size in every released variant, but a distinct
            field, so it is passed explicitly rather than derived from ``lm``.
        """
        from olmo_core.nn.vision.config import VisionEncoderType

        # mm_olmo `llm.response_residual_dropout`: drop this fraction of the residual on
        # *response* tokens only (`residual_dropout` stays 0 for prompt and image tokens).
        # Defaults to 0 so the factory keeps matching the HF-derived config, which describes
        # the released weights for inference; training scripts opt in (stage 1 uses 0.1).
        if response_residual_dropout:
            if not isinstance(lm.block, TransformerBlockConfig):
                raise OLMoConfigurationError(
                    "response_residual_dropout needs a single block config, got a per-block dict"
                )
            lm.block.masked_dropout = response_residual_dropout

        # Molmo2 keeps only blocks 0..24 of SigLIP2's 27 — everything past the deepest
        # feature layer (``vit_layers`` max 24) is dropped. ``name`` is a label only (the
        # SigLIP variant is selected by use_cls_token / patch_embedding_bias / use_pre_ln);
        # we use ``siglip`` so this is byte-identical to the HF-derived config.
        vision = replace(
            VisionEncoderConfig.siglip2_so400m_patch14_378(),
            name=VisionEncoderType.siglip,
            image_num_layers=25,
        )
        return cls(
            lm=lm,
            vision=vision,
            connector=VisionConnectorConfig(
                image_emb_dim=vision.image_emb_dim,
                image_num_heads=vision.image_num_heads,
                image_num_key_value_heads=vision.image_num_key_value_heads,
                image_head_dim=vision.image_head_dim,
                output_dim=lm.d_model,
                num_input_layers=2,
                pooling_type=ImagePoolingType.attention_meanq,
                pooling_attention_mask=True,
                projector_type=ImageProjectorType.mlp,
                mlp_hidden_size=connector_mlp_hidden_size,
            ),
            image_patch_token_id=IM_PATCH_ID,
            vit_layers=(24, 18),
            # The head spans the base vocab structurally, so no logit masking is required.
            output_vocab_size=None,
            **kwargs,
        )

    @classmethod
    def molmo2_4B(
        cls, *, rope_theta: int = 5_000_000, response_residual_dropout: float = 0.0, **kwargs
    ) -> "MultimodalLMConfig":
        """
        Molmo2-4B architecture: a Qwen3-4B LM plus the shared Molmo2 vision stack.

        Equivalent to
        :func:`~olmo_core.nn.vision.molmo2_loader.molmo2_config_from_hf_config` applied to
        ``allenai/Molmo2-4B``, but without reading anything from the Hugging Face Hub — so
        training from base checkpoints has no dependency on the released Molmo2 repo (whose
        remote code also needs ``trust_remote_code=True``).

        :param rope_theta: RoPE base. Defaults to ``5_000_000``, matching the *released*
            Molmo2-4B weights. When initialising the LM from **base** Qwen3-4B, pass
            ``1_000_000`` instead — that is what those weights were trained with (and what
            mm_olmo's stage-1 ``QWEN3_4B`` config uses). Note Molmo2-4B is the only released
            variant whose base differs from its Qwen3 backbone's.
        """
        return cls._molmo2_like(
            TransformerConfig.qwen3_4B(
                vocab_size=MOLMO2_BASE_VOCAB_SIZE,
                n_extra_vocab=MOLMO2_N_EXTRA_TOKENS,
                rope_theta=rope_theta,
                # The released pretrain configs set `rope_full_precision: true`.
                rope_full_precision=True,
                attn_backend=_molmo2_attn_backend(),
                dtype=DType.float32,
            ),
            connector_mlp_hidden_size=9728,
            response_residual_dropout=response_residual_dropout,
            **kwargs,
        )

    @classmethod
    def molmo2_8B(
        cls, *, rope_theta: int = 1_000_000, response_residual_dropout: float = 0.0, **kwargs
    ) -> "MultimodalLMConfig":
        """
        Molmo2-8B architecture: a Qwen3-8B LM plus the shared Molmo2 vision stack.

        Equivalent to
        :func:`~olmo_core.nn.vision.molmo2_loader.molmo2_config_from_hf_config` applied to
        ``allenai/Molmo2-8B``, without reading from the Hugging Face Hub.

        :param rope_theta: RoPE base. ``1_000_000`` for both the released Molmo2-8B weights
            and base Qwen3-8B, so unlike :meth:`molmo2_4B` the default is correct for either
            weight source. Note this variant is **not** weight-tied (neither is Qwen3-8B).
        """
        return cls._molmo2_like(
            TransformerConfig.qwen3_8B(
                vocab_size=MOLMO2_BASE_VOCAB_SIZE,
                n_extra_vocab=MOLMO2_N_EXTRA_TOKENS,
                rope_theta=rope_theta,
                # The released pretrain configs set `rope_full_precision: true`.
                rope_full_precision=True,
                attn_backend=_molmo2_attn_backend(),
                dtype=DType.float32,
            ),
            connector_mlp_hidden_size=12288,
            response_residual_dropout=response_residual_dropout,
            **kwargs,
        )

    def build(self, init_device: str = "cpu") -> "MultimodalLM":
        """
        Instantiate the multimodal model on ``init_device``.

        :param init_device: Device string (e.g. ``"cpu"``, ``"meta"``).
        :returns: A :class:`MultimodalLM`.
        """
        return MultimodalLM(self, init_device=init_device)


class MultimodalLM(nn.Module):
    """
    Vision-language model: vision encoder + connector + language model.

    Forward pass flow:

    1. Look up LM token embeddings for ``input_ids``.
    2. If images are provided, run them through the vision tower, extract the
       configured ViT layers, optionally strip the CLS / register prefix, and
       gather/pool/project via the connector to produce one feature per
       ``<im_patch>`` placeholder token.
    3. Add the projected image features back into the LM embedding sequence
       at every position where ``input_ids == image_patch_token_id``.
    4. Run the LM with the modified embeddings.

    :param cfg: Multimodal model configuration.
    :param init_device: Device on which to initialise parameters.
    """

    def __init__(self, cfg: MultimodalLMConfig, init_device: str = "cpu"):
        super().__init__()
        if cfg.output_vocab_size is not None and not (
            0 < cfg.output_vocab_size <= cfg.lm.vocab_size
        ):
            raise OLMoConfigurationError(
                f"output_vocab_size ({cfg.output_vocab_size}) must be in "
                f"(0, lm.vocab_size={cfg.lm.vocab_size}]"
            )
        self.cfg = cfg
        self.lm = cfg.lm.build(init_device=init_device)
        # Cached so `forward` only builds a drop mask when some block will consume it.
        self._masked_residual_dropout = float(getattr(cfg.lm.block, "masked_dropout", 0.0) or 0.0)
        self.vision = cfg.vision.build(init_device=init_device)
        self.connector = cfg.connector.build(init_device=init_device)

    # -- model introspection (mirrors the Transformer API used by the trainer / callbacks) --

    @property
    def num_params(self) -> int:
        """Total number of parameters (LM + vision + connector)."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_non_embedding_params(self) -> int:
        """All parameters excluding the LM token-embedding table (the vision encoder
        and connector have no token embeddings, so they count in full)."""
        vision_connector = sum(
            p.numel() for m in (self.vision, self.connector) for p in m.parameters()
        )
        return self.lm.num_non_embedding_params + vision_connector

    @property
    def is_moe(self) -> bool:
        return self.lm.is_moe

    def num_flops_per_token(self, seq_len: int) -> int:
        """Idealized FLOPs/token for the language model only. The vision encoder /
        connector run per-image (not per-token), so their FLOPs are reported separately
        via :meth:`image_encoder_flops` and added to the throughput/MFU accounting."""
        return self.lm.num_flops_per_token(seq_len)

    @property
    def _n_vision_params(self) -> int:
        if not hasattr(self, "_n_vision_params_cache"):
            self._n_vision_params_cache = sum(p.numel() for p in self.vision.parameters())
        return self._n_vision_params_cache

    @property
    def _n_connector_params(self) -> int:
        if not hasattr(self, "_n_connector_params_cache"):
            self._n_connector_params_cache = sum(p.numel() for p in self.connector.parameters())
        return self._n_connector_params_cache

    def image_encoder_flops(
        self, n_crops: int, n_patches_per_crop: int, n_pooled_tokens: int
    ) -> int:
        """Idealized FLOPs for the vision half of one batch, for MFU accounting.

        The ViT processes every (padded) crop in the batch, so ``n_crops`` should be the
        full ``B * n_crops`` of the images tensor. The encoder is **frozen** → forward-only
        (2 FLOPs/param/patch for the linear layers, plus the attention score+context
        quadratic ``4·L·P·d`` per patch). The connector is **trained** → 6 FLOPs/param
        (fwd+bwd) per pooled output token.

        :param n_crops: total number of image crops processed by the ViT this batch.
        :param n_patches_per_crop: patches per crop fed to the ViT (``P``).
        :param n_pooled_tokens: number of pooled ``<im_patch>`` tokens produced.
        """
        d = self.cfg.vision.image_emb_dim
        n_layers = self.cfg.vision.image_num_layers
        n_raw = n_crops * n_patches_per_crop
        vit = n_raw * (2 * self._n_vision_params + 4 * n_layers * n_patches_per_crop * d)
        connector = n_pooled_tokens * 6 * self._n_connector_params
        return int(vit + connector)

    def _vit_crop_microbatch(self) -> int:
        """Max crops per ViT forward (0 = no chunking). Env: ``VIT_CROP_MICROBATCH``."""
        raw = os.environ.get("VIT_CROP_MICROBATCH", "16")
        return int(raw)

    def _vit_forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """Run ViT on ``images`` ``(B*T, N, patch_dim)`` and return ``(B*T, n_patches, dim)``."""
        hidden_states: List[torch.Tensor] = self.vision(images)
        selected = [hidden_states[i] for i in self.cfg.vit_layers]
        features = torch.cat(selected, dim=-1) if len(selected) > 1 else selected[0]
        num_prefix = getattr(self.vision, "num_prefix_tokens", 0)
        if num_prefix > 0:
            features = features[:, num_prefix:]
        return features

    def _encode_images(
        self,
        images: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode pre-patchified images into LM-space pooled features.

        :param images: Shape ``(B, n_crops, n_patches, patch_dim)``.
        :param pooled_patches_idx: Shape ``(B, n_pooled, pool_size)`` —
            indices into the flattened ``(n_crops * n_patches)`` patch axis
            for each pool group, with ``-1`` marking padded slots.
        :returns: Shape ``(B, n_pooled, lm_d_model)``.
        """
        B, T, N, _ = images.shape
        microbatch = self._vit_crop_microbatch()

        # Pad crop axis to the DP max so every rank runs the same number of ViT
        # microbatch chunks (avoids FSDP collective desync when ``n_crops`` differs).
        if is_distributed():
            t_local = torch.tensor([T], device=images.device, dtype=torch.int32)
            t_max = t_local.clone()
            dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
            t_pad = int(t_max.item())
            if t_pad > T:
                pad = torch.zeros(
                    (B, t_pad - T, N, images.shape[-1]),
                    device=images.device,
                    dtype=images.dtype,
                )
                images = torch.cat([images, pad], dim=1)
                T = t_pad

        if microbatch <= 0 or T <= microbatch:
            features = self._vit_forward_features(images.reshape(B * T, N, -1))
            features = features.reshape(B, T * features.shape[1], features.shape[-1])
        else:
            parts: List[torch.Tensor] = []
            for start in range(0, T, microbatch):
                end = min(start + microbatch, T)
                chunk = images[:, start:end].reshape(B * (end - start), N, -1)
                chunk_features = self._vit_forward_features(chunk)
                parts.append(chunk_features.reshape(B, (end - start) * chunk_features.shape[1], -1))
            features = torch.cat(parts, dim=1)

        return self.connector(features, pooled_patches_idx)

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        pooled_patches_idx: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        subsegment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        example_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        """
        Run the vision-language forward pass.

        :param input_ids: Token IDs, shape ``(B, seq_len)``. Positions equal to
            :attr:`cfg.image_patch_token_id` will be overwritten (via ``+=``)
            with projected image features.
        :param images: Pre-patchified image patches, shape
            ``(B, n_crops, n_patches, patch_dim)``. Pass ``None`` for
            text-only batches.
        :param pooled_patches_idx: Per-group patch indices,
            shape ``(B, n_pooled, pool_size)``. Required when *images* is not
            ``None``. ``n_pooled`` must equal the number of
            ``<im_patch>`` tokens per sequence.
        :param labels: Target token IDs, shape ``(B, seq_len)``.
        :param token_type_ids: Optional ``(B, seq_len)`` tensor marking image tokens
            (non-zero) vs. text tokens (zero). When provided, image tokens attend to
            each other **bidirectionally** (causal order is ignored among image
            tokens) while text stays causal, matching HF Molmo2's attention mask.
            Requires the dense (``"torch"``) attention backend.
        :param subsegment_ids: Optional ``(B, seq_len)`` int tensor marking subsegments
            for packed multi-annotation data: a shared prefix (``ATTEND_ALL`` id) that
            branches into several mutually-isolated response branches (one id each). A
            query may only attend to keys with a ``>=`` subsegment id (matching mm_olmo's
            ``attention_mask & (subseg_q <= subseg_k)``). Requires ``position_ids`` and
            the dense (``"torch"``) attention backend.
        :param position_ids: Optional ``(B, seq_len)`` int tensor of explicit RoPE
            positions. Required with ``subsegment_ids`` (parallel branches share an
            overlapping position range and cannot use sequential positions).
        :returns: Logits or loss (same as
            :meth:`~olmo_core.nn.transformer.Transformer.forward`).
        """
        if subsegment_ids is not None and position_ids is None:
            raise ValueError("`position_ids` is required when `subsegment_ids` is provided")

        response_logits_only = kwargs.pop("response_logits_only", False)
        loss_masks = kwargs.pop("loss_masks", None)
        response_mask: Optional[torch.Tensor] = None
        if response_logits_only:
            if loss_masks is None:
                raise ValueError("`loss_masks` is required when `response_logits_only=True`")
            response_mask = loss_masks > 0

        # Per-token residual dropout (mm_olmo `response_residual_dropout`): the mask selects
        # *response* tokens, so prompt and image tokens keep the plain `dropout` rate. Only
        # built when a block actually asks for it, and only while training.
        drop_mask: Optional[torch.Tensor] = None
        if self.training and self._masked_residual_dropout > 0.0:
            if loss_masks is None:
                raise ValueError(
                    "`loss_masks` is required when the LM uses `masked_dropout` "
                    "(response_residual_dropout)"
                )
            drop_mask = (loss_masks > 0).to(dtype=torch.bool)

        if labels is not None and self.cfg.output_vocab_size is not None:
            # The LM head would compute the loss internally over the full (padded) vocab,
            # bypassing the output-vocab masking below and shifting the softmax
            # denominator relative to mm_olmo / HF Molmo2 (whose lm_head has no columns
            # for the extra image-special tokens).
            raise OLMoConfigurationError(
                "`labels` cannot be passed through MultimodalLM when `output_vocab_size` "
                "is set: compute the loss externally on the (masked) logits instead, as "
                "MultimodalTransformerTrainModule does."
            )

        assert (
            self.lm.embeddings is not None
        ), "MultimodalLM requires the LM to have an embedding table"

        device = self.lm.device
        input_ids = input_ids.to(device)
        if labels is not None:
            labels = labels.to(device)

        use_flex_attn = os.environ.get("OLMO2_FLEX_ATTN") == "1"
        or_mask: Optional[torch.Tensor] = None
        and_mask: Optional[torch.Tensor] = None
        flex_attn_block_mask = None
        flex_mask_kwargs: Optional[dict] = None
        if use_flex_attn:
            B, S = input_ids.shape
            flex_is_image = token_type_ids.to(device) != 0 if token_type_ids is not None else None
            flex_subseg = subsegment_ids.to(device) if subsegment_ids is not None else None
            flex_eid = example_ids.to(device) if example_ids is not None else None
            flex_mask_kwargs = dict(
                B=B,
                S=S,
                device=device,
                is_image=flex_is_image,
                subsegment_ids=flex_subseg,
                example_id=flex_eid,
            )

        # Compute LM token embeddings with any configured scale / norm. We embed here
        # (rather than inside ``self.lm``) so image features can be spliced in below.
        # Under FSDP the embedding weight is a sharded ``DTensor`` that only the LM's own
        # forward would unshard, so gather it to a full tensor for the lookup (a no-op for
        # DDP / single-GPU where the weight is already a plain tensor).
        emb = self.lm.embeddings

        def _full(weight: torch.Tensor) -> torch.Tensor:
            return weight.full_tensor() if isinstance(weight, DTensor) else weight

        if isinstance(emb, SplitVocabEmbedding):
            # The image-special token IDs live in the *extra* block, so the lookup must span
            # both parameters — `emb.weight` alone covers only the base vocab. Gradients still
            # reach both blocks through the concatenation.
            emb_weight = torch.cat([_full(emb.weight), _full(emb.extra_weight)], dim=0)
        else:
            emb_weight = _full(emb.weight)
        h = F.embedding(input_ids, emb_weight, padding_idx=emb.padding_idx)
        if self.lm.embed_scale is not None:
            h = h * self.lm.embed_scale
        if self.lm.embedding_norm is not None:
            h = self.lm.embedding_norm(h)

        if images is not None:
            if pooled_patches_idx is None:
                raise ValueError("`pooled_patches_idx` is required when `images` is provided")

            images = images.to(device)
            pooled_patches_idx = pooled_patches_idx.to(device)

            image_features = self._encode_images(images, pooled_patches_idx)  # (B, n_pooled, d)

            # Tie the connector output into the autograd graph on *every* forward that ran
            # the vision path, even when no rows are spliced below (e.g. an all-text
            # microbatch handed a dummy zero crop by the collator). This adds exactly 0 to
            # the activations but keeps the connector's FSDP reduce-scatter — and the vision
            # all-gather — firing on every rank each step, so collectives stay in lockstep
            # across ranks regardless of how text-only vs image examples are distributed.
            h = h + 0.0 * image_features.sum().to(h.dtype)

            # ViT may run extra crop microbatches when ``n_crops`` differs across DP ranks;
            # sync before the LM FSDP forward so all-gather collectives stay aligned.
            if is_distributed():
                barrier()

            # Keep only valid pooled rows (a row is padding iff *all* its patch
            # indices are -1, e.g. added by a batch collator to equalize ``n_pooled``
            # across examples). Selecting in row-major order keeps each example's
            # features aligned with its ``<im_patch>`` positions, so batches with a
            # variable number of image tokens per example work. For unpadded / B=1
            # inputs every row is valid and this is a no-op.
            valid_rows = (pooled_patches_idx >= 0).any(dim=-1)  # (B, n_pooled)
            image_features = image_features[valid_rows]  # (total_valid, d)

            # Splice into LM embeddings at every <im_patch> position.
            is_image_patch = input_ids.view(-1) == self.cfg.image_patch_token_id
            n_patches_in_seq = int(is_image_patch.sum())
            n_features = image_features.shape[0]
            if n_patches_in_seq != n_features:
                raise ValueError(
                    f"Number of <im_patch> tokens in input_ids ({n_patches_in_seq}) does not "
                    f"match the number of projected image features ({n_features}). The data "
                    f"preprocessor must insert exactly one <im_patch> per pooled feature."
                )
            d = h.shape[-1]
            # ``.contiguous()`` guards against a non-contiguous ``h`` (e.g. from a fused
            # embedding_norm), for which ``.view()`` would raise. ``flat`` is a view of
            # the contiguous ``h``, so the in-place add below propagates back into ``h``.
            h = h.contiguous()
            flat = h.view(-1, d)
            flat[is_image_patch] = flat[is_image_patch] + image_features.reshape(-1, d)

        # Build flex BlockMask after vision so the mask is not resident during ViT.
        if flex_mask_kwargs is not None:
            from olmo_core.nn.attention.backend import FlexAttentionBackend

            flex_attn_block_mask = FlexAttentionBackend.build_block_mask_from_vectors(
                **flex_mask_kwargs
            )

        if not use_flex_attn:
            # Dense (B, S, S) masks for the torch SDPA backend only.
            if token_type_ids is not None:
                is_image = token_type_ids.to(device) != 0  # (B, S)
                or_mask = (is_image[:, :, None] & is_image[:, None, :]).unsqueeze(1)

            seg_rule: Optional[torch.Tensor] = None
            if subsegment_ids is not None:
                seg = subsegment_ids.to(device)
                seg_rule = seg[:, :, None] <= seg[:, None, :]
            if example_ids is not None:
                eid = example_ids.to(device)
                same_example = eid[:, :, None] == eid[:, None, :]
                combined = same_example & seg_rule if seg_rule is not None else same_example
                and_mask = combined.unsqueeze(1)
            elif seg_rule is not None:
                and_mask = seg_rule.unsqueeze(1)

        if position_ids is not None:
            position_ids = position_ids.to(device)

        out = self.lm(
            input_ids,
            input_embeddings=h,
            labels=labels,
            or_mask=or_mask,
            and_mask=and_mask,
            flex_attn_block_mask=flex_attn_block_mask,
            position_ids=position_ids,
            response_logits_only=response_logits_only,
            response_mask=response_mask,
            drop_mask=drop_mask,
            **kwargs,
        )

        # Mask the logit columns of the inputs-only image-special tokens (see
        # :attr:`MultimodalLMConfig.output_vocab_size`). ``finfo.min`` underflows to
        # exactly 0 in the softmax, so loss / sampling / gradients match a
        # base-vocab-only head bit-for-bit (mm_olmo computes logits against the base
        # embedding table only, even under weight tying). Applies to both the full
        # ``(B, S, V)`` logits and the ``(N_response, V)`` response-only logits.
        output_vocab_size = self.cfg.output_vocab_size
        if (
            output_vocab_size is not None
            and isinstance(out, torch.Tensor)
            and out.shape[-1] > output_vocab_size
        ):
            out[..., output_vocab_size:] = torch.finfo(out.dtype).min
        return out
