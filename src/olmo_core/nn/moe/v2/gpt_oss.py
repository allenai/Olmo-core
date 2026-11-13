"""
Config builders for GPT-OSS (fused-MoE) models on the Olmo-core fused-MoE-v2 stack.

.. warning::
    These builders map GPT-OSS architecture hyperparameters onto Olmo-core configs. There is no
    GPT-OSS checkpoint conversion path in :mod:`olmo_core.nn.hf.convert` yet, so a config produced
    here cannot load a HuggingFace GPT-OSS checkpoint.
    :func:`build_gpt_oss_20b_config_from_hf_config` maps HF *config* hyperparameters only.

    These builders are accessed via this module path (``olmo_core.nn.moe.v2.gpt_oss``); they are
    not re-exported from the package namespace.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from olmo_core.config import DType
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation
from olmo_core.nn.moe import MoELoadBalancingLossGranularity, MoERouterGatingFunction
from olmo_core.nn.moe.v2.ep_config import ExpertParallelConfig, ExpertParallelPath
from olmo_core.nn.moe.v2.routed_experts import ExpertActivation, RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.rope import RoPEConfig, RoPEType, YaRNRoPEScalingConfig
from olmo_core.nn.transformer import TransformerBlockType, TransformerType
from olmo_core.nn.transformer.config import OLMoDDPModelConfig, TransformerBlockConfig

GPT_OSS_SLIDING_ATTENTION = "sliding_attention"
GPT_OSS_FULL_ATTENTION = "full_attention"
GPT_OSS_LAYER_PATTERN = (GPT_OSS_SLIDING_ATTENTION, GPT_OSS_FULL_ATTENTION)


def _resolve_ep_config(
    ep: ExpertParallelConfig | None,
    *,
    ep_path: ExpertParallelPath | str,
    ep_capacity_factor: float,
) -> ExpertParallelConfig:
    if ep is None:
        ep = ExpertParallelConfig(
            path=ExpertParallelPath(ep_path),
            capacity_factor=ep_capacity_factor,
        )
    else:
        ep = deepcopy(ep)
    ep.validate()
    return ep


def _resolve_layer_types(layer_types: Sequence[str], n_layers: int) -> tuple[str, ...]:
    if n_layers <= 0:
        raise ValueError(f"n_layers must be positive, got {n_layers}")
    if len(layer_types) == 0:
        raise ValueError("layer_types must be non-empty")
    if len(layer_types) == n_layers:
        return tuple(layer_types)
    if len(layer_types) > n_layers:
        return tuple(layer_types[:n_layers])
    if n_layers % len(layer_types) != 0:
        raise ValueError(
            f"n_layers ({n_layers}) must be divisible by layer_types length ({len(layer_types)})"
        )
    return tuple(layer_types) * (n_layers // len(layer_types))


def _resolve_rope_parameters(hf_config: Mapping[str, Any]) -> Mapping[str, Any]:
    rope_parameters = hf_config.get("rope_parameters")
    if rope_parameters is not None:
        return rope_parameters
    rope_scaling = hf_config.get("rope_scaling")
    if rope_scaling is not None:
        params = dict(rope_scaling)
        if "rope_theta" not in params and "rope_theta" in hf_config:
            params["rope_theta"] = hf_config["rope_theta"]
        return params
    return {
        "rope_type": "yarn",
        "factor": 32.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "original_max_position_embeddings": 4096,
        "rope_theta": hf_config.get("rope_theta", 150000),
    }


def get_gpt_oss_20b_config_overrides(hf_config: Mapping[str, Any]) -> dict[str, Any]:
    """
    Extract :func:`build_gpt_oss_20b_config` keyword arguments from a HuggingFace GPT-OSS config.
    """
    rope_parameters = _resolve_rope_parameters(hf_config)
    return {
        "vocab_size": int(hf_config["vocab_size"]),
        "d_model": int(hf_config["hidden_size"]),
        "n_layers": int(hf_config["num_hidden_layers"]),
        "num_attention_heads": int(hf_config["num_attention_heads"]),
        "num_key_value_heads": int(hf_config["num_key_value_heads"]),
        "attention_head_dim": int(hf_config["head_dim"]),
        "rope_theta": int(rope_parameters.get("rope_theta", hf_config.get("rope_theta", 150000))),
        "rope_factor": float(rope_parameters.get("factor", 32.0)),
        "rope_beta_fast": int(float(rope_parameters.get("beta_fast", 32.0))),
        "rope_beta_slow": int(float(rope_parameters.get("beta_slow", 1.0))),
        "rope_original_max_position_embeddings": int(
            rope_parameters.get("original_max_position_embeddings", 4096)
        ),
        "sliding_window": int(hf_config["sliding_window"]),
        "num_experts": int(hf_config["num_local_experts"]),
        "num_experts_per_tok": int(hf_config["num_experts_per_tok"]),
        "moe_intermediate_size": int(hf_config["intermediate_size"]),
        "rms_norm_eps": float(hf_config["rms_norm_eps"]),
        "router_aux_loss_weight": float(hf_config.get("router_aux_loss_coef", 0.9)),
        "swiglu_limit": float(hf_config.get("swiglu_limit", 7.0)),
        "layer_types": tuple(hf_config.get("layer_types", GPT_OSS_LAYER_PATTERN)),
    }


def _make_attention_block(
    *,
    common_block_kwargs: dict[str, Any],
    num_attention_heads: int,
    num_key_value_heads: int,
    attention_head_dim: int,
    rope_theta: int,
    rope_factor: float,
    rope_beta_fast: int,
    rope_beta_slow: int,
    rope_original_max_position_embeddings: int,
    sliding_window: int | None,
    attention_backend: AttentionBackendName,
    dtype: DType,
) -> OLMoDDPTransformerBlockConfig:
    return OLMoDDPTransformerBlockConfig(
        sequence_mixer=AttentionConfig(
            name=AttentionType.default,
            n_heads=num_attention_heads,
            n_kv_heads=num_key_value_heads,
            head_dim=attention_head_dim,
            bias=True,
            rope=RoPEConfig(
                name=RoPEType.default,
                theta=int(rope_theta),
                # GPT-OSS rotates the full head dim, so partial_rotary_factor is 1.0.
                partial_rotary_factor=1.0,
                full_precision=False,
                scaling=YaRNRoPEScalingConfig(
                    factor=rope_factor,
                    beta_fast=rope_beta_fast,
                    beta_slow=rope_beta_slow,
                    old_context_len=rope_original_max_position_embeddings,
                ),
            ),
            backend=attention_backend,
            dtype=dtype,
            attention_sinks=True,
            sliding_window=None
            if sliding_window is None
            else SlidingWindowAttentionConfig(
                pattern=[sliding_window],
                force_full_attention_on_first_layer=False,
                force_full_attention_on_last_layer=False,
            ),
        ),
        **{k: deepcopy(v) for k, v in common_block_kwargs.items()},
    )


def build_gpt_oss_20b_config(
    *,
    vocab_size: int = 201088,
    d_model: int = 2880,
    n_layers: int = 24,
    num_attention_heads: int = 64,
    num_key_value_heads: int = 8,
    attention_head_dim: int = 64,
    rope_theta: int = 150000,
    rope_factor: float = 32.0,
    rope_beta_fast: int = 32,
    rope_beta_slow: int = 1,
    rope_original_max_position_embeddings: int = 4096,
    sliding_window: int = 128,
    num_experts: int = 32,
    num_experts_per_tok: int = 4,
    moe_intermediate_size: int = 2880,
    rms_norm_eps: float = 1e-5,
    swiglu_limit: float = 7.0,
    layer_types: Sequence[str] = GPT_OSS_LAYER_PATTERN,
    dtype: DType = DType.bfloat16,
    init_seed: int = 2026,
    init_std: float = 0.02,
    attention_backend: AttentionBackendName = AttentionBackendName.torch,
    router_aux_loss_weight: float | None = 0.9,
    router_z_loss_weight: float | None = None,
    compile_friendly_recompute: bool = False,
    ep: ExpertParallelConfig | None = None,
    ep_path: ExpertParallelPath | str = ExpertParallelPath.sync_1d,
    ep_capacity_factor: float = 1.25,
) -> OLMoDDPModelConfig:
    """
    Build a GPT-OSS-20B model config on the fused-MoE-v2 stack. Defaults match the 20B architecture;
    override any keyword to build a scaled variant.
    """
    layer_types = _resolve_layer_types(layer_types, n_layers)
    unsupported = set(layer_types) - {GPT_OSS_SLIDING_ATTENTION, GPT_OSS_FULL_ATTENTION}
    if unsupported:
        raise ValueError(f"Unsupported gpt-oss layer_types: {sorted(unsupported)}")

    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=rms_norm_eps,
        bias=False,
        dtype=dtype,
    )
    routed_experts = RoutedExpertsConfig(
        d_model=d_model,
        hidden_size=moe_intermediate_size,
        num_experts=num_experts,
        bias=True,
        dtype=dtype,
        activation=ExpertActivation.gpt_oss_swiglu,
        activation_alpha=1.702,
        activation_limit=swiglu_limit,
    )
    routed_router = MoERouterConfigV2(
        d_model=d_model,
        num_experts=num_experts,
        top_k=num_experts_per_tok,
        bias=True,
        gating_function=MoERouterGatingFunction.topk_softmax,
        normalize_expert_weights=None,
        lb_loss_weight=router_aux_loss_weight,
        z_loss_weight=router_z_loss_weight,
        lb_loss_granularity=MoELoadBalancingLossGranularity.instance,
        dtype=dtype,
    )
    common_block_kwargs = dict(
        name=TransformerBlockType.moe_fused_v2,
        use_pre_norm=True,
        use_peri_norm=False,
        ep=_resolve_ep_config(ep, ep_path=ep_path, ep_capacity_factor=ep_capacity_factor),
        checkpoint_attn=compile_friendly_recompute,
        checkpoint_permute_moe_unpermute=compile_friendly_recompute,
        checkpoint_second_unpermute=compile_friendly_recompute,
        # core's block config takes a single layer_norm (the attention/feed-forward split is a
        # constructor-only detail); both norms are the same config here.
        layer_norm=layer_norm,
        routed_experts=deepcopy(routed_experts),
        routed_experts_router=deepcopy(routed_router),
        shared_experts=None,
        shared_experts_router=None,
    )

    def _block(sliding_window_size: int | None) -> OLMoDDPTransformerBlockConfig:
        return _make_attention_block(
            common_block_kwargs=common_block_kwargs,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_head_dim=attention_head_dim,
            rope_theta=rope_theta,
            rope_factor=rope_factor,
            rope_beta_fast=rope_beta_fast,
            rope_beta_slow=rope_beta_slow,
            rope_original_max_position_embeddings=rope_original_max_position_embeddings,
            sliding_window=sliding_window_size,
            attention_backend=attention_backend,
            dtype=dtype,
        )

    blocks = {
        GPT_OSS_SLIDING_ATTENTION: _block(sliding_window),
        GPT_OSS_FULL_ATTENTION: _block(None),
    }

    # Annotate with the base config type: OLMoDDPModelConfig.block is invariant over the dict value
    # type, so the concrete OLMoDDPTransformerBlockConfig subtype won't unify with it.
    block: TransformerBlockConfig | dict[str, TransformerBlockConfig]
    block_pattern: list[str] | None
    if len(set(layer_types)) == 1:
        block = blocks[layer_types[0]]
        block_pattern = None
    else:
        block = dict(blocks)
        block_pattern = list(layer_types)

    config = OLMoDDPModelConfig(
        init_seed=init_seed,
        init_std=init_std,
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        block=block,
        block_pattern=block_pattern,
        embedding_norm=None,
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
        name=TransformerType.moe_fused_v2,
        dtype=dtype,
        two_batch_overlap=False,
        recompute_each_block=compile_friendly_recompute,
        recompute_all_blocks_by_chunk=False,
    )
    config.lm_head.loss_implementation = LMLossImplementation.default
    return config


def build_gpt_oss_20b_config_from_hf_config(
    hf_config: Mapping[str, Any],
    **overrides: Any,
) -> OLMoDDPModelConfig:
    """
    Build a GPT-OSS config from a HuggingFace GPT-OSS *config* mapping.

    .. warning::
        This maps config values only; it does **not** convert HuggingFace checkpoint weights. The
        fused MoE experts/router layout (and the attention-sink logits) have no converter in
        :mod:`olmo_core.nn.hf.convert`, so the resulting config cannot load a HuggingFace GPT-OSS
        checkpoint.

    .. # TODO(gpt-oss-hf-checkpoint-convert): add a GPT-OSS-aware HF state converter (router +
        routed experts + attention sinks + sliding/full attention) and a strict-load + logit-parity
        test, then drop this warning.
    """
    kwargs = get_gpt_oss_20b_config_overrides(hf_config)
    kwargs.update(overrides)
    return build_gpt_oss_20b_config(**kwargs)


def build_debug_gpt_oss_20b_config(
    *,
    vocab_size: int,
    n_layers: int = 4,
    d_model: int = 256,
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
    moe_intermediate_size: int = 128,
    dtype: DType = DType.bfloat16,
    attention_backend: AttentionBackendName = AttentionBackendName.torch,
    **kwargs: Any,
) -> OLMoDDPModelConfig:
    """
    Build a small GPT-OSS config for tests/debugging using the interleaved sliding/full pattern.
    """
    return build_gpt_oss_20b_config(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=1,
        attention_head_dim=max(1, d_model // 4),
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        layer_types=_resolve_layer_types(GPT_OSS_LAYER_PATTERN, n_layers),
        dtype=dtype,
        attention_backend=attention_backend,
        **kwargs,
    )
