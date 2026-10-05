"""Vision-language models with pooled image features in the language-model embedding stream."""

import os
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from olmo_core.config import Config, DType
from olmo_core.distributed.utils import barrier, is_distributed
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.embedding import SplitVocabEmbedding
from olmo_core.nn.functional import weighted_cross_entropy_loss
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerConfig
from olmo_core.nn.vision.config import VisionEncoderConfig
from olmo_core.nn.vision.connector import (
    ImagePoolingType,
    ImageProjectorType,
    VisionConnectorConfig,
)
from olmo_core.nn.vision.molmo2_tokens import IM_PATCH_ID
from olmo_core.utils import move_to_device

__all__ = [
    "MOLMO2_BASE_VOCAB_SIZE",
    "MOLMO2_N_EXTRA_TOKENS",
    "MOLMO2_VOCAB_SIZE",
    "MultimodalLMConfig",
    "MultimodalLM",
    "MultimodalOLMoDDPModel",
]

MOLMO2_BASE_VOCAB_SIZE = 151_936
"""Molmo2's base text vocab (Qwen3's) — the number of IDs the model may *predict*."""

MOLMO2_N_EXTRA_TOKENS = 128
"""Molmo2's inputs-only image-special tokens, held in a separate embedding block."""

MOLMO2_VOCAB_SIZE = MOLMO2_BASE_VOCAB_SIZE + MOLMO2_N_EXTRA_TOKENS
"""Molmo2's total *input* vocab (base + extra); the LM head spans only the base."""


_INPUT_DIAGNOSTIC_NAMES = (
    "text embedding RMS",
    "connector output RMS",
    "spliced image embedding RMS",
)


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
        image_patch_token_id: int = IM_PATCH_ID,
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
        :param image_patch_token_id: Token id whose positions receive the image features;
            Molmo2's own ``<im_patch>`` id by default, a spare row of the LM's vocabulary
            when the LM is not a Molmo2 one.
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
            image_patch_token_id=image_patch_token_id,
            vit_layers=(24, 18),
            # The head spans the base vocab structurally, so no logit masking is required.
            output_vocab_size=None,
            **kwargs,
        )

    @classmethod
    def molmo2_vision_stack(
        cls, lm: TransformerConfig, *, image_patch_token_id: int, **kwargs
    ) -> "MultimodalLMConfig":
        """
        Molmo2's vision tower and connector around an already-built language model.

        The vision encoder and the connector are those of :meth:`molmo2_4B` (SigLIP2-SO400M/14
        truncated to 25 blocks, two-layer attention-pooling connector with a 9,728-wide MLP),
        with the connector projecting into ``lm.d_model``. This is the configuration for
        aligning a vision tower to a language model from another checkpoint; the connector
        is freshly initialised rather than loaded from a Molmo2 checkpoint.

        :param lm: The already-built language-model config.
        :param image_patch_token_id: The token id in ``lm``'s vocabulary that marks image
            patch positions.
        """
        return cls._molmo2_like(
            lm,
            connector_mlp_hidden_size=9728,
            image_patch_token_id=image_patch_token_id,
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
        :returns: A :class:`MultimodalLM`, or a :class:`MultimodalOLMoDDPModel` when the language
            model is an :class:`~olmo_core.nn.transformer.OLMoDDPModelConfig`.
        """
        from olmo_core.nn.transformer import OLMoDDPModelConfig

        if isinstance(self.lm, OLMoDDPModelConfig):
            return MultimodalOLMoDDPModel(self, init_device=init_device)
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
        self._collect_input_diagnostics = False
        self._input_diagnostic_sums: Dict[str, torch.Tensor] = {}
        self._input_diagnostic_counts: Dict[str, int] = {}
        # Resolved lazily from the LM's sequence mixers; tests may set it directly.
        self._document_mode: Optional[bool] = None

    def uses_document_boundaries(self) -> bool:
        """
        Whether packed examples are isolated through document boundaries (``doc_lens``) instead
        of attention masks.

        Recurrent sequence mixers such as :class:`~olmo_core.nn.attention.kda.KimiDeltaAttention`
        cannot apply an attention mask; they reset their state at document boundaries instead
        (``cu_doc_lens``), and the attention layers of such language models take the same
        boundaries through their variable-length kernels. When any LM block uses such a mixer,
        :meth:`forward` derives one document per packed example from ``example_ids`` and sends no
        masks or positions. Image tokens are then causal like text.
        """
        if self._document_mode is None:
            from olmo_core.nn.attention.kda import KimiDeltaAttention

            self._document_mode = any(
                isinstance(module, KimiDeltaAttention) for module in self.lm.modules()
            )
        return self._document_mode

    # -- input-scale diagnostics ------------------------------------------------------------

    def set_input_diagnostics(self, enabled: bool) -> None:
        """Enable or disable detached embedding-scale diagnostics for the next forwards."""
        self._collect_input_diagnostics = enabled
        self._input_diagnostic_sums.clear()
        self._input_diagnostic_counts.clear()

    @torch.no_grad()
    def _record_input_diagnostic(self, name: str, values: torch.Tensor) -> None:
        if not self._collect_input_diagnostics or values.numel() == 0:
            return
        sum_squares = values.detach().float().square().sum()
        if name in self._input_diagnostic_sums:
            self._input_diagnostic_sums[name] += sum_squares
            self._input_diagnostic_counts[name] += values.numel()
        else:
            self._input_diagnostic_sums[name] = sum_squares
            self._input_diagnostic_counts[name] = values.numel()

    @torch.no_grad()
    def pop_input_diagnostics(
        self,
        *,
        reduce_across_process_group: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Return the accumulated RMS values and disable collection.

        :param reduce_across_process_group: Reduce sums and counts before computing each RMS.
        :param process_group: The process group to reduce across. ``None`` uses the default group.
        """
        if reduce_across_process_group:
            if not is_distributed():
                raise RuntimeError("cannot reduce input diagnostics outside distributed training")
            device = next(self.parameters()).device
            n_metrics = len(_INPUT_DIAGNOSTIC_NAMES)
            stats = torch.zeros(2 * n_metrics, dtype=torch.float64, device=device)
            for idx, name in enumerate(_INPUT_DIAGNOSTIC_NAMES):
                if name in self._input_diagnostic_sums:
                    stats[idx] = self._input_diagnostic_sums[name].double()
                    stats[n_metrics + idx] = self._input_diagnostic_counts[name]
            self.set_input_diagnostics(False)
            dist.all_reduce(stats, group=process_group)
            return {
                name: torch.sqrt(stats[idx] / stats[n_metrics + idx])
                for idx, name in enumerate(_INPUT_DIAGNOSTIC_NAMES)
                if stats[n_metrics + idx] > 0
            }

        diagnostics = {
            name: torch.sqrt(sum_squares / self._input_diagnostic_counts[name])
            for name, sum_squares in self._input_diagnostic_sums.items()
        }
        self.set_input_diagnostics(False)
        return diagnostics

    # -- model introspection (mirrors the Transformer API used by the trainer / callbacks) --

    @property
    def num_params(self) -> int:
        """Total number of parameters (LM + vision + connector)."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        """Number of parameters with gradient computation enabled."""
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
        """Whether the language model uses mixture-of-experts layers."""
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
        full ``B * n_crops`` of the images tensor. A frozen encoder is forward-only (2
        FLOPs/param/patch for the linear layers, plus the attention score+context quadratic
        ``4·L·P·d`` per patch); a trainable encoder uses the forward-plus-backward multipliers
        (6 and 12). The connector is **trained** → 6 FLOPs/param (fwd+bwd) per pooled output
        token.

        :param n_crops: total number of image crops processed by the ViT this batch.
        :param n_patches_per_crop: patches per crop fed to the ViT (``P``).
        :param n_pooled_tokens: number of pooled ``<im_patch>`` tokens produced.
        """
        d = self.cfg.vision.image_emb_dim
        n_layers = self.cfg.vision.image_num_layers
        n_raw = n_crops * n_patches_per_crop
        vision_multiplier = (
            6 if any(param.requires_grad for param in self.vision.parameters()) else 2
        )
        attention_multiplier = 12 if vision_multiplier == 6 else 4
        vit = n_raw * (
            vision_multiplier * self._n_vision_params
            + attention_multiplier * n_layers * n_patches_per_crop * d
        )
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

    def encode_images(
        self,
        images: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode images into the compact ``(total_valid_pooled_rows, d_model)`` features that
        :meth:`forward` splices at the ``<im_patch>`` positions, so they can be computed once and
        reused across autoregressive forwards (``encoded_image_features``).

        :param images: Shape ``(B, n_crops, n_patches, patch_dim)``.
        :param pooled_patches_idx: Shape ``(B, n_pooled, pool_size)``; a row whose indices are
            all ``-1`` is collator padding and is dropped.
        """
        device = self.lm.device
        images = images.to(device)
        pooled_patches_idx = pooled_patches_idx.to(device)
        image_features = self._encode_images(images, pooled_patches_idx)  # (B, n_pooled, d)
        self._record_input_diagnostic("connector output RMS", image_features)

        # ViT may run extra crop microbatches when ``n_crops`` differs across DP ranks;
        # sync before the LM forward so all-gather collectives stay aligned.
        if is_distributed():
            barrier()

        # Keep only valid pooled rows (a row is padding iff *all* its patch indices are -1,
        # e.g. added by a batch collator to equalize ``n_pooled`` across examples). Selecting in
        # row-major order keeps each example's features aligned with its ``<im_patch>``
        # positions, so batches with a variable number of image tokens per example work. For
        # unpadded / B=1 inputs every row is valid and this is a no-op.
        valid_rows = (pooled_patches_idx >= 0).any(dim=-1)  # (B, n_pooled)
        return image_features[valid_rows]  # (total_valid, d)

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        pooled_patches_idx: Optional[torch.Tensor] = None,
        encoded_image_features: Optional[torch.Tensor] = None,
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
            ``None``. The number of non-padding pooled rows must equal the number of
            ``<im_patch>`` tokens per sequence.
        :param encoded_image_features: Optional output of :meth:`encode_images`, reused across
            autoregressive forwards; mutually exclusive with ``images``.
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

        document_mode = self.uses_document_boundaries()
        if document_mode:
            # The packer always emits ``subsegment_ids`` (a constant id for examples without
            # branches), so only real sibling branches, which need an attention mask, are
            # rejected; otherwise the ids carry nothing beyond the example boundaries.
            if subsegment_ids is not None and has_sibling_branches(subsegment_ids, example_ids):
                raise ValueError(
                    "Sibling-branch packing (`subsegment_ids` with several branches in one "
                    "example) needs an attention mask, which the LM's recurrent sequence mixers "
                    "cannot apply; pack one document per example instead."
                )
            subsegment_ids = None
            # Recurrent mixers carry no positions and the attention layers of such LMs use no
            # RoPE, so explicit positions have nothing to act on.
            position_ids = None
            doc_lens, max_doc_lens = document_lengths_from_example_ids(
                example_ids if example_ids is not None else torch.zeros_like(input_ids)
            )
            kwargs["doc_lens"] = doc_lens
            kwargs["max_doc_lens"] = max_doc_lens

        use_flex_attn = os.environ.get("OLMO2_FLEX_ATTN") == "1" and not document_mode
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

        if images is not None and encoded_image_features is not None:
            raise ValueError("Pass either `images` or `encoded_image_features`, not both")

        image_features = encoded_image_features
        if images is not None:
            if pooled_patches_idx is None:
                raise ValueError("`pooled_patches_idx` is required when `images` is provided")
            image_features = self.encode_images(images, pooled_patches_idx)

            # Tie the connector output into the autograd graph on *every* forward that ran
            # the vision path, even when no rows are spliced below (e.g. an all-text
            # microbatch handed a dummy zero crop by the collator). This adds exactly 0 to
            # the activations but keeps the connector's FSDP reduce-scatter — and the vision
            # all-gather — firing on every rank each step, so collectives stay in lockstep
            # across ranks regardless of how text-only vs image examples are distributed.
            h = h + 0.0 * image_features.sum().to(h.dtype)

        is_image_patch: Optional[torch.Tensor] = None
        if image_features is not None:
            # Splice into LM embeddings at every <im_patch> position.
            image_features = image_features.to(device)
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
            image_features = image_features.to(flat.dtype)
            flat[is_image_patch] = flat[is_image_patch] + image_features.reshape(-1, d)

        if self._collect_input_diagnostics:
            valid_tokens = torch.ones_like(input_ids, dtype=torch.bool)
            if token_type_ids is not None:
                valid_tokens = valid_tokens & (token_type_ids.to(device) == 0)
            elif is_image_patch is not None:
                valid_tokens = (valid_tokens.view(-1) & ~is_image_patch).view_as(input_ids)
            self._record_input_diagnostic("text embedding RMS", h[valid_tokens])
            if is_image_patch is not None:
                self._record_input_diagnostic(
                    "spliced image embedding RMS", h.reshape(-1, h.shape[-1])[is_image_patch]
                )

        # Build flex BlockMask after vision so the mask is not resident during ViT.
        if flex_mask_kwargs is not None:
            from olmo_core.nn.attention.backend import FlexAttentionBackend

            flex_attn_block_mask = FlexAttentionBackend.build_block_mask_from_vectors(
                **flex_mask_kwargs
            )

        if not use_flex_attn and not document_mode:
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


def has_sibling_branches(
    subsegment_ids: torch.Tensor, example_ids: Optional[torch.Tensor] = None
) -> bool:
    """
    Whether any packed example carries more than one branch.

    :param subsegment_ids: ``(B, S)`` per-token subsegment ids; the shared prefix of a branched
        example uses ``ATTEND_ALL_SUBSEGMENT_ID`` and does not count as a branch.
    :param example_ids: ``(B, S)`` per-token example ids (``-1`` = padding); ``None`` treats
        each row as one example.
    """
    from olmo_core.data.multimodal.sequence_builder import ATTEND_ALL_SUBSEGMENT_ID

    subsegment_ids = subsegment_ids.cpu()
    rows = example_ids.cpu() if example_ids is not None else torch.zeros_like(subsegment_ids)
    for row_subseg, row_examples in zip(subsegment_ids, rows):
        for example in torch.unique(row_examples):
            if int(example) < 0:
                continue
            ids = torch.unique(row_subseg[row_examples == example])
            if int((ids != ATTEND_ALL_SUBSEGMENT_ID).sum()) > 1:
                return True
    return False


def document_lengths_from_example_ids(example_ids: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
    """
    Turn per-token packed-example ids into per-row document lengths.

    Every run of equal ids in a row is one document, including a trailing padding run
    (``-1``), so the lengths of a row sum to the sequence length and each padded position is
    isolated from every example. Ids are read on the CPU: the result feeds kernel launch
    arguments (``max_doc_lens``) that must be Python integers.

    :param example_ids: Integer tensor of shape ``(B, seq_len)``.

    :returns: ``doc_lens`` of shape ``(B, max_docs)`` (``int32``, zero-padded) as consumed by
        :func:`~olmo_core.data.utils.get_cumulative_document_lengths`, and the longest document
        of each row.

    :raises ValueError: If an id occurs in two separate runs of a row, which is how
        sibling-branch packing interleaves examples; such batches need attention masks.
    """
    if example_ids.dim() != 2:
        raise ValueError(f"example_ids must be (B, seq_len), got shape {tuple(example_ids.shape)}")
    rows = example_ids.detach().cpu().to(torch.long)
    lengths: List[List[int]] = []
    for row in rows:
        if row.numel() == 0:
            lengths.append([])
            continue
        change = torch.ones(row.numel(), dtype=torch.bool)
        change[1:] = row[1:] != row[:-1]
        starts = torch.nonzero(change, as_tuple=False).flatten()
        ends = torch.cat([starts[1:], torch.tensor([row.numel()])])
        if len(starts) != len(torch.unique(row)):
            raise ValueError(
                "example_ids must be contiguous per example (one document each); interleaved "
                "ids indicate sibling-branch packing, which needs attention masks"
            )
        lengths.append((ends - starts).tolist())
    max_docs = max((len(row) for row in lengths), default=0)
    doc_lens = torch.zeros(len(lengths), max_docs, dtype=torch.int32)
    for index, row in enumerate(lengths):
        if row:
            doc_lens[index, : len(row)] = torch.tensor(row, dtype=torch.int32)
    return doc_lens, [max(row) if row else 0 for row in lengths]


class MultimodalOLMoDDPModel(MultimodalLM):
    """
    A :class:`MultimodalLM` whose language model is an
    :class:`~olmo_core.nn.ddp.model.OLMoDDPModel`, trainable by
    :class:`~olmo_core.train.train_module.transformer.multimodal_train_module.MultimodalOLMoDDPTrainModule`.

    The OLMoDDP train module owns data/expert parallelism and its fused optimizer, and only
    talks to the language model through a small runtime surface. This adapter keeps the
    multimodal model compositional and forwards that surface to the LM, wrapping the *whole*
    model (vision encoder, connector and LM) in the LM's multi-group DDP. Pipeline, tensor and
    context parallelism are deliberately not exposed: multimodal masks and image embeddings need
    the full sequence on each rank.

    Given ``labels`` and ``loss_masks``, the forward computes the multimodal objective itself: a
    float-weighted, response-only cross entropy normalized by the global loss weight, plus the
    LM's z-loss over the same positions. Given ``labels`` alone (a text-only batch) it defers to
    the language model's plain per-token loss, and without labels it returns logits like the
    parent.

    The router token mask of #868 is **not** carried: ``router_token_mask`` only feeds the train
    module's router-loss divisor and data metrics, while the routers themselves see every
    position, so their load-balancing and z-loss statistics include packed padding tokens. The
    bridge phase trains with a zero load-balancing weight; this matters once the language model
    itself trains (perception/joint and mid-training).
    """

    _olmo_ddp_compatible = True

    def __init__(self, cfg: MultimodalLMConfig, init_device: str = "cpu"):
        super().__init__(cfg, init_device=init_device)
        from olmo_core.nn.ddp import OLMoDDPModel

        if not isinstance(self.lm, OLMoDDPModel):
            raise TypeError(
                f"{type(self).__name__} requires an OLMoDDPModel language model, "
                f"got {type(self.lm).__name__}"
            )

    def encode_images(
        self,
        images: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode images with the pixels cast to the vision tower's parameter dtype.

        The OLMoDDP train module casts the whole wrapped model (vision encoder, connector and
        LM) to its training dtype, while the collator delivers float32 pixels, so the cast is
        required here; it is a no-op when the tower already runs in the pixels' dtype.
        """
        vision_dtype = next(self.vision.parameters()).dtype
        return super().encode_images(images.to(dtype=vision_dtype), pooled_patches_idx)

    @property
    def _olmo_lm(self):
        from olmo_core.nn.ddp import OLMoDDPModel

        return cast(OLMoDDPModel, self.lm)

    # -- runtime surface the OLMoDDP train module reads ---------------------------------------

    @property
    def tbo(self) -> bool:
        """Whether the language model uses two-batch overlap."""
        return self._olmo_lm.tbo

    @property
    def recompute_all_blocks_by_chunk(self) -> bool:
        """Whether activation recomputation groups LM blocks into chunks."""
        return self._olmo_lm.recompute_all_blocks_by_chunk

    @property
    def recompute_each_block(self) -> bool:
        """Whether each LM block is independently recomputed during backward."""
        return self._olmo_lm.recompute_each_block

    @property
    def has_grad_accum_fp32_buffer(self) -> bool:
        """Whether the language model accumulates gradients in FP32 buffers."""
        return self._olmo_lm.has_grad_accum_fp32_buffer

    @property
    def device(self) -> torch.device:
        """The language model's device."""
        return self._olmo_lm.device

    def train(self, mode: bool = True):
        """Set training mode while keeping a fully frozen vision encoder in evaluation mode."""
        super().train(mode)
        if mode and not any(p.requires_grad for p in self.vision.parameters()):
            self.vision.eval()
        return self

    def purge_cuda_events(self) -> None:
        """Release language-model CUDA event handles before copying or partitioning it."""
        self._olmo_lm.purge_cuda_events()

    def install_cuda_events(self) -> None:
        """Install language-model CUDA event handles for distributed execution."""
        self._olmo_lm.install_cuda_events()

    def count_non_rowwise_ep_no_sync_blocks(self) -> int:
        """Count no-sync expert-parallel blocks that do not use rowwise dispatch."""
        return self._olmo_lm.count_non_rowwise_ep_no_sync_blocks()

    def count_ep_no_sync_blocks(self) -> int:
        """Count language-model blocks using no-sync expert parallelism."""
        return self._olmo_lm.count_ep_no_sync_blocks()

    def iter_ep_no_sync_symm_tensor_infos(self) -> Iterator[Any]:
        """Yield symmetric-buffer metadata from no-sync expert-parallel blocks."""
        yield from self._olmo_lm.iter_ep_no_sync_symm_tensor_infos()

    def apply_fp8(self, *args, **kwargs) -> None:
        """Configure FP8 computation in the language model."""
        self._olmo_lm.apply_fp8(*args, **kwargs)

    def apply_cp(self, *args, **kwargs) -> None:
        """Reject context parallelism, which is unsupported for multimodal batches."""
        raise NotImplementedError("Context parallelism is not supported for multimodal OLMoDDP")

    def apply_pp(self, *args, **kwargs) -> None:
        """Reject pipeline parallelism, which is unsupported for multimodal batches."""
        raise NotImplementedError("Pipeline parallelism is not supported for multimodal OLMoDDP")

    def apply_ep(self, *args, **kwargs):
        """Apply expert parallelism to the language model."""
        return self._olmo_lm.apply_ep(*args, **kwargs)

    def apply_activation_checkpointing(self, *args, **kwargs) -> None:
        """Configure activation checkpointing in the language model."""
        self._olmo_lm.apply_activation_checkpointing(*args, **kwargs)

    def apply_compile(self) -> None:
        """Compile the language-model blocks."""
        self._olmo_lm.apply_compile()

    def apply_dp(self, *args, **kwargs):
        """Wrap the complete multimodal model (not just the LM) in data parallelism."""
        return self._olmo_lm.apply_dp(*args, root_module=self, **kwargs)

    @torch.no_grad()
    def init_weights(self, *args, device: Optional[torch.device] = None, **kwargs):
        """Initialize LM parameters and materialize the vision encoder and connector."""
        device = device or self._olmo_lm.device
        generator = self._olmo_lm.init_weights(*args, device=device, **kwargs)
        for module in (self.vision, self.connector):
            module.to_empty(device=device)
            module.reset_parameters()
        return generator

    def prewarm_deepep_v2_runtimes(self, *args, **kwargs) -> None:
        """Initialize DeepEP runtimes used by the language model."""
        self._olmo_lm.prewarm_deepep_v2_runtimes(*args, **kwargs)

    def prewarm_ep_no_sync_symm_buffers(self, *args, **kwargs) -> None:
        """Prewarm the language model's expert-parallel symmetric buffers."""
        self._olmo_lm.prewarm_ep_no_sync_symm_buffers(*args, **kwargs)

    def refresh_rowwise_fp8_cache(self) -> None:
        """Rebuild language-model FP8 caches from current parameter values."""
        self._olmo_lm.refresh_rowwise_fp8_cache()

    def named_fp8_weight_stores(self) -> Iterator[Tuple[str, object]]:
        """Yield FP8 weight stores with multimodal-model parameter names."""
        for name, weight in self._olmo_lm.named_fp8_weight_stores():
            yield f"lm.{name}", weight

    def named_mxfp8_expert_weights(self) -> Iterator[Tuple[str, object]]:
        """Yield FP8 weight stores using the expert-weight compatibility interface."""
        yield from self.named_fp8_weight_stores()

    def zero_grad(self, *args, **kwargs) -> None:
        """Clear gradients of every component through the language model's buffers."""
        self._olmo_lm.zero_grad(*args, **kwargs)
        for module in (self.vision, self.connector):
            for param in module.parameters():
                param.grad = None

    def post_batch(self, dry_run: bool = False) -> None:
        """Run language-model cleanup after an optimizer batch."""
        self._olmo_lm.post_batch(dry_run=dry_run)

    def post_optim_step(self) -> None:
        """Update language-model state after the optimizer step."""
        self._olmo_lm.post_optim_step()

    def compute_auxiliary_metrics(self, reset: bool = True) -> Dict[str, Tuple[torch.Tensor, Any]]:
        """Return accumulated language-model auxiliary metrics and their reduction types."""
        return self._olmo_lm.compute_auxiliary_metrics(reset=reset)

    def reset_auxiliary_metrics(self) -> None:
        """Clear accumulated language-model auxiliary metrics."""
        self._olmo_lm.reset_auxiliary_metrics()

    # -- forward ------------------------------------------------------------------------------

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        loss_masks: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        loss_weight_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        response_logits_only: bool = False,
        router_token_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        """
        Run the vision-language forward pass.

        Without ``labels`` this returns logits exactly like :meth:`MultimodalLM.forward`. With
        ``labels`` it computes the multimodal training objective from response-position logits:

        .. code-block:: python

            ce = sum_i loss_masks_i * CE(logits_i, labels_i) / loss_weight_div_factor

        so that, after DDP gradient averaging, the objective is normalized by the global loss
        weight (the train module supplies ``loss_weight_div_factor`` as the global weight sum
        divided by the DP world size, matching the LM-only convention for ``loss_div_factor``).
        The z-loss uses the same weights. Router auxiliary losses keep their own divisor
        (``router_loss_div_factor``, forwarded to the LM).

        :param loss_masks: Float per-token loss weights, shape ``(B, seq_len)``, aligned with
            ``labels``. Required with ``labels``; ``0`` marks unsupervised positions.
        :param loss_div_factor: The LM-only divisor (label count). Accepted for interface
            compatibility and used only when ``loss_weight_div_factor`` is not given.
        :param loss_weight_div_factor: Divisor for the weighted CE and z-loss.
        :param response_logits_only: Compute logits only where ``loss_masks > 0``. Always on
            when ``labels`` are given; opt-in otherwise.
        :param router_token_mask: Accepted from the collator for the train module (router loss
            divisor, data metrics) and dropped here: the router token mask of #868 is not
            carried, so the language model's routers see every position and their
            load-balancing and z-loss statistics include padding tokens (see the class docstring).
        """
        del router_token_mask
        if loss_masks is not None:
            loss_masks = loss_masks.to(device=self.lm.device, dtype=torch.float32)
        if labels is None:
            return super().forward(
                input_ids,
                loss_masks=loss_masks,
                response_logits_only=response_logits_only,
                **kwargs,
            )
        if loss_masks is None:
            # A text-only batch (e.g. a downstream evaluator's): the language model computes
            # the plain per-token cross entropy with the LM-only divisor and reduction.
            return super().forward(
                input_ids,
                labels=labels,
                ignore_index=ignore_index,
                loss_reduction=loss_reduction,
                z_loss_multiplier=z_loss_multiplier,
                loss_div_factor=loss_div_factor,
                return_logits=return_logits,
                response_logits_only=response_logits_only,
                **kwargs,
            )
        if loss_reduction != "sum":
            raise ValueError(
                f"{type(self).__name__} computes a weighted sum loss; got "
                f"loss_reduction={loss_reduction!r}"
            )
        if self.cfg.output_vocab_size is not None:
            raise OLMoConfigurationError(
                "output_vocab_size is not supported by the multimodal OLMoDDP loss"
            )
        labels = labels.to(self.lm.device)
        response_mask = loss_masks > 0
        # Response logits are ``(N_response, vocab)`` in row-major mask order.
        logits = super().forward(
            input_ids, loss_masks=loss_masks, response_logits_only=True, **kwargs
        )
        assert isinstance(logits, torch.Tensor)
        flat_labels = labels.reshape(-1)[response_mask.reshape(-1)]
        flat_weights = loss_masks.reshape(-1)[response_mask.reshape(-1)]
        ce_loss, z_loss = weighted_cross_entropy_loss(
            logits,
            flat_labels,
            flat_weights,
            ignore_index=ignore_index,
            compute_z_loss=z_loss_multiplier is not None,
            z_loss_multiplier=z_loss_multiplier or 0.0,
        )
        div_factor = (
            loss_weight_div_factor if loss_weight_div_factor is not None else loss_div_factor
        )
        if div_factor is not None:
            div_factor = move_to_device(div_factor, ce_loss.device)
            ce_loss = ce_loss / div_factor
            if z_loss is not None:
                z_loss = z_loss / div_factor
        loss = ce_loss if z_loss is None else ce_loss + z_loss
        return LMOutputWithLoss(
            logits=logits if return_logits else None, loss=loss, ce_loss=ce_loss, z_loss=z_loss
        )
