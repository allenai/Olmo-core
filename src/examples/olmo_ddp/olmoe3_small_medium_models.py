"""Canonical small and medium OLMoE3 production-candidate model shapes.

These are the exact model shapes used by the September 2026 systems
qualification runs. Keep this module free of launcher and data concerns so it
can be imported directly by profiling tools.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

from olmo_core.config import DType
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    GateConfig,
    GateGranularity,
    KimiDeltaAttentionConfig,
)
from olmo_core.nn.ddp import OLMoDDPTransformerBlockConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe import (
    EmoRouterConfig,
    LatentMoEConfig,
    MoELoadBalancingLossGranularity,
    MoERouterGatingFunction,
)
from olmo_core.nn.moe.v2.ep_config import ExpertParallelConfig, ExpertParallelPath
from olmo_core.nn.moe.v2.fp8 import MoERowwiseFP8Config
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.transformer import (
    InitMethod,
    OLMoDDPModelConfig,
    TransformerBlockType,
    TransformerType,
)

VOCAB_SIZE = 100_352
HEAD_DIM = 128
NUM_EXPERTS = 512
TOP_K = 16
MODEL_SIZES = ("small", "medium")


@dataclass(frozen=True)
class Geometry:
    """One controlled family member and its guarded parameter counts."""

    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    expected_active_params: int
    expected_active_non_embedding_params: int
    expected_total_params: int

    @property
    def latent_dim(self) -> int:
        return self.d_model // 2

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        # Layer 0 is dense+KDA. The remaining blocks follow the 7:1 KDA:FA cadence.
        return tuple(range(7, self.n_layers, 8))


GEOMETRIES: dict[str, Geometry] = {
    "small": Geometry(
        d_model=1024,
        n_layers=16,
        n_heads=8,
        n_kv_heads=4,
        expected_active_params=794_230_912,
        expected_active_non_embedding_params=691_470_464,
        expected_total_params=12_496_339_072,
    ),
    "medium": Geometry(
        d_model=1536,
        n_layers=24,
        n_heads=16,
        n_kv_heads=8,
        expected_active_params=2_387_524_992,
        expected_active_non_embedding_params=2_233_384_320,
        expected_total_params=42_759_798_144,
    ),
}


def geometry(model_size: str) -> Geometry:
    """Return the guarded geometry for ``model_size``."""

    try:
        return GEOMETRIES[model_size.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown model size {model_size!r}; choose from {MODEL_SIZES}") from exc


def _norm() -> LayerNormConfig:
    return LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=DType.float32,
    )


def _kda(g: Geometry) -> KimiDeltaAttentionConfig:
    return KimiDeltaAttentionConfig(
        n_heads=g.n_heads,
        n_v_heads=g.n_heads,
        head_dim=HEAD_DIM,
        expand_v=2.0,
        allow_neg_eigval=True,
        conv_size=4,
        conv_bias=False,
        norm_eps=1e-5,
        use_cute_kernel=True,
        dtype=DType.float32,
    )


def _attention(g: Geometry, norm: LayerNormConfig) -> AttentionConfig:
    return AttentionConfig(
        name=AttentionType.default,
        n_heads=g.n_heads,
        n_kv_heads=g.n_kv_heads,
        head_dim=HEAD_DIM,
        bias=False,
        gate=GateConfig(granularity=GateGranularity.elementwise, full_precision=True),
        rope=None,
        qk_norm=deepcopy(norm),
        backend=AttentionBackendName.flash_4,
        scalable_softmax=True,
        dtype=DType.float32,
        use_head_qk_norm=True,
    )


def _shared(g: Geometry, *, hidden_size: int | None = None) -> SharedExpertsConfig:
    return SharedExpertsConfig(
        d_model=g.d_model,
        hidden_size=hidden_size or g.d_model,
        num_experts=1,
        bias=False,
        dtype=DType.float32,
    )


def _moe_block(
    g: Geometry,
    norm: LayerNormConfig,
    sequence_mixer: KimiDeltaAttentionConfig | AttentionConfig,
    emo: EmoRouterConfig,
) -> OLMoDDPTransformerBlockConfig:
    rowwise_fp8 = MoERowwiseFP8Config(
        enabled=False,
        fused_autograd_recompute_swiglu=False,
    )
    return OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        sequence_mixer=sequence_mixer,
        layer_norm=deepcopy(norm),
        shared_experts=_shared(g),
        routed_experts=RoutedExpertsConfig(
            d_model=g.latent_dim,
            hidden_size=g.d_model,
            num_experts=NUM_EXPERTS,
            bias=False,
            dtype=DType.float32,
            rowwise_fp8=rowwise_fp8,
        ),
        routed_experts_router=MoERouterConfigV2(
            d_model=g.d_model,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            bias=False,
            normalize_expert_weights=1.0,
            gating_function=MoERouterGatingFunction.softmax,
            dtype=DType.float32,
            lb_loss_weight=0.01,
            lb_loss_granularity=MoELoadBalancingLossGranularity.local_batch,
            z_loss_weight=1e-5,
            restore_weight_scale=True,
            use_recompute_fp32_cast=False,
            global_load_balancing=True,
            emo=deepcopy(emo),
        ),
        latent_moe=LatentMoEConfig(
            latent_dim=g.latent_dim,
            up_proj_input_norm_enabled=False,
        ),
        use_peri_norm=True,
        use_pre_norm=False,
        checkpoint_attn=False,
        checkpoint_permute_moe_unpermute=False,
        checkpoint_second_unpermute=False,
        ep=ExpertParallelConfig(
            path=ExpertParallelPath.rowwise_nvshmem,
            capacity_factor=1.25,
            share_dispatch_out=False,
            share_combine_out=False,
        ),
        rowwise_fp8=rowwise_fp8,
    )


def _dense_first(g: Geometry, norm: LayerNormConfig) -> OLMoDDPTransformerBlockConfig:
    return OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        sequence_mixer=_kda(g),
        layer_norm=deepcopy(norm),
        shared_experts=_shared(g, hidden_size=8 * g.d_model),
        use_peri_norm=True,
        use_pre_norm=False,
        checkpoint_attn=False,
        checkpoint_permute_moe_unpermute=False,
        checkpoint_second_unpermute=False,
    )


def build_model_config(
    model_size: str,
    *,
    eos_token_id: int,
    vocab_size: int = VOCAB_SIZE,
) -> OLMoDDPModelConfig:
    """Build the exact BF16/EMO production candidate for ``model_size``."""

    g = geometry(model_size)
    norm = _norm()
    emo = EmoRouterConfig(
        eos_token_id=eos_token_id,
        min_document_expert_pool=16,
        max_document_expert_pool=NUM_EXPERTS,
        eval_document_expert_pool=NUM_EXPERTS,
    )
    emo.validate_for_router(num_experts=NUM_EXPERTS, top_k=TOP_K)
    default_block = _moe_block(g, norm, _kda(g), emo)
    attention_block = _moe_block(g, norm, _attention(g, norm), emo)
    model = OLMoDDPModelConfig(
        name=TransformerType.moe_fused_v2,
        d_model=g.d_model,
        vocab_size=vocab_size,
        n_layers=g.n_layers,
        block=default_block,
        block_overrides={
            0: _dense_first(g, norm),
            **{index: deepcopy(attention_block) for index in g.full_attention_layers},
        },
        lm_head=LMHeadConfig(
            layer_norm=deepcopy(norm),
            bias=False,
            dtype=DType.float32,
        ),
        embedding_norm=deepcopy(norm),
        dtype=DType.float32,
        init_method=InitMethod.normal,
        init_seed=0,
        init_std=0.02,
        embed_scale=math.sqrt(g.d_model),
        tie_word_embeddings=False,
        two_batch_overlap=False,
        recompute_all_blocks_by_chunk=False,
        recompute_each_block=False,
    )
    model.validate()

    if vocab_size == VOCAB_SIZE:
        actual = (
            model.num_active_params,
            model.num_active_non_embedding_params,
            model.num_params,
        )
        expected = (
            g.expected_active_params,
            g.expected_active_non_embedding_params,
            g.expected_total_params,
        )
        if actual != expected:
            raise ValueError(f"{model_size} parameter-count drift: {actual=} != {expected=}")
    return model
