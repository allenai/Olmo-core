import logging
from typing import Any, Dict, List, Optional

import torch.nn.functional as F
from transformers import Olmo2Config, PretrainedConfig

from olmo_core.doc_utils import beta_feature
from olmo_core.nn.attention import Attention, GateGranularity, KimiDeltaAttention
from olmo_core.nn.attention.recurrent import GatedDeltaNet
from olmo_core.nn.layer_norm import FusedRMSNorm, LayerNorm, RMSNorm
from olmo_core.nn.moe.mlp import DroplessMoEMLP, MoEMLP
from olmo_core.nn.moe.router import MoERouterGatingFunction
from olmo_core.nn.rope import RoPEScalingConfig
from olmo_core.nn.transformer.block import (
    MoEReorderedNormTransformerBlock,
    ReorderedNormTransformerBlock,
    TransformerBlock,
)
from olmo_core.nn.transformer.model import (
    MoETransformer,
    NormalizedTransformer,
    Transformer,
)

log = logging.getLogger(__name__)

try:
    from olmo_core.nn.ddp.model import OLMoDDPModel  # type: ignore
    from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import (
        Olmo3MoeConfig,  # type: ignore
    )
except ImportError:
    Olmo3MoeConfig = None  # type: ignore[assignment,misc]
    OLMoDDPModel = None  # type: ignore[assignment,misc]

try:
    from transformers import FlexOlmoConfig  # type: ignore
except ImportError:
    FlexOlmoConfig = None

try:
    from transformers import Olmo3Config  # type: ignore
except ImportError:
    Olmo3Config = None


def _validate_olmo3moe_router_selection(router: Any) -> None:
    """Reject router behavior that the HF Olmo3Moe implementation cannot reproduce."""
    emo = getattr(router, "emo", None)
    if emo is not None and emo.eval_pool_size() != router.num_experts:
        raise NotImplementedError(
            "HF EMo export currently requires eval_document_expert_pool=num_experts."
        )
    unsupported_modifiers = []
    for flag in ("uniform_expert_assignment", "random_expert_assignment"):
        if getattr(router, flag, False):
            unsupported_modifiers.append(flag)
    if router.bias_gamma is not None:
        unsupported_modifiers.append("bias_gamma")
    if router.score_correction_bias:
        unsupported_modifiers.append("score_correction_bias")
    if router.gating_function not in (
        MoERouterGatingFunction.softmax,
        MoERouterGatingFunction.sigmoid,
    ):
        unsupported_modifiers.append(f"gating_function={router.gating_function.value}")
    if router.n_group is not None or router.topk_group is not None:
        unsupported_modifiers.append("grouped routing (n_group/topk_group)")
    if router.expert_weight_scale is not None:
        unsupported_modifiers.append("expert_weight_scale")
    if (
        router.gating_function == MoERouterGatingFunction.sigmoid
        and router.sigmoid_stability_epsilon != 1e-7
    ):
        unsupported_modifiers.append(
            f"sigmoid_stability_epsilon={router.sigmoid_stability_epsilon}"
        )
    if unsupported_modifiers:
        raise NotImplementedError(
            f"Exporting olmo3moe with router selection modifiers "
            f"({', '.join(unsupported_modifiers)}) is not supported."
        )


def _get_flex_olmo_config(model: MoETransformer) -> PretrainedConfig:
    blocks = list(model.blocks.values())
    for block in blocks:
        if not isinstance(block, MoEReorderedNormTransformerBlock):
            raise NotImplementedError(
                f"Block is not a {MoEReorderedNormTransformerBlock.__name__}, unable to build HF config for {model.__class__.__name__}"
            )

        if not isinstance(block.experts.mlp, (DroplessMoEMLP, MoEMLP)):
            raise NotImplementedError(
                f"MoE mlp is not a {DroplessMoEMLP.__name__} or {MoEMLP.__name__}, unable to build HF config for {model.__class__.__name__}"
            )

        if not isinstance(block.attention, Attention):
            raise NotImplementedError(
                f"Attention is not a {Attention.__name__}, unable to build HF config for {model.__class__.__name__}"
            )
        if block.attention.rope is None:
            raise NotImplementedError(
                f"Attention does not use rope, unable to build HF config for {model.__class__.__name__}"
            )

    block = blocks[0]
    assert isinstance(block, MoEReorderedNormTransformerBlock)
    assert isinstance(block.attention, Attention)
    assert block.attention.rope is not None

    if FlexOlmoConfig is None:
        raise RuntimeError("The installed transformers version does not support FlexOlmo")

    return FlexOlmoConfig(
        vocab_size=model.vocab_size,
        hidden_size=model.d_model,
        intermediate_size=block.feed_forward_moe.experts.mlp.hidden_size,
        num_hidden_layers=model.n_layers,
        num_attention_heads=block.attention.n_heads,
        num_key_value_heads=block.attention.n_kv_heads,
        hidden_act="silu",
        max_position_embeddings=-1,
        attention_bias=block.attention.w_out.bias is not None,
        rope_theta=block.attention.rope.theta,
        pad_token_id=None,  # type: ignore
        bos_token_id=None,
        eos_token_id=None,  # type: ignore
        rms_norm_eps=block.feed_forward_norm.eps,
        num_experts_per_tok=block.feed_forward_moe.router.top_k,
        num_experts=block.feed_forward_moe.router.num_experts,
        tie_word_embeddings=model.tie_word_embeddings,
    )


def _register_olmo3moe_auto_classes() -> None:
    """
    Register the standalone ``olmo3moe`` config/model with transformers' ``Auto*`` mappings.

    transformers ships no ``olmo3moe`` architecture. The in-memory ``Auto*.register`` calls let
    ``AutoModelForCausalLM.from_config`` resolve it while exporting a checkpoint. The
    ``register_for_auto_class`` calls additionally persist an ``auto_map`` into the exported
    ``config.json`` and bundle the model code alongside it, so a fresh process can reload the
    checkpoint with ``trust_remote_code=True``. In-memory registration is idempotent —
    transformers raises :class:`ValueError` on a duplicate.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeForCausalLM

    try:
        AutoConfig.register("olmo3moe", Olmo3MoeConfig)
    except ValueError:
        pass  # already registered
    try:
        AutoModelForCausalLM.register(Olmo3MoeConfig, Olmo3MoeForCausalLM)
    except ValueError:
        pass  # already registered

    Olmo3MoeConfig.register_for_auto_class("AutoConfig")
    Olmo3MoeForCausalLM.register_for_auto_class("AutoModelForCausalLM")


def _validate_olmo3moe_norm(
    norm: Optional[LayerNorm], *, name: str, eps: float, weight_shape: tuple
) -> None:
    # HF applies both normalization and gains in FP32 before casting back. Other
    # RMSNorm variants can round activations or gains earlier in the computation.
    if norm is None or type(norm) not in (RMSNorm, FusedRMSNorm) or not norm.full_precision:
        raise NotImplementedError(f"HF {name} requires full-precision RMSNorm or FusedRMSNorm.")
    assert norm is not None
    if norm.weight is None or tuple(norm.weight.shape) != weight_shape or norm.bias is not None:
        raise NotImplementedError(
            f"HF {name} requires bias-free RMSNorm with gains {weight_shape}."
        )
    if norm.eps != eps:
        raise NotImplementedError(f"HF {name} requires the model RMSNorm epsilon ({eps}).")


def _validate_olmo3moe_model_norms(model: "OLMoDDPModel", eps: float) -> None:
    norms = [("lm_head.norm", model.lm_head.norm, model.d_model)]
    if model.embedding_norm is not None:
        norms.append(("embedding_norm", model.embedding_norm, model.d_model))
    for idx, block in model.blocks.items():
        for name in ("attention_norm", "feed_forward_norm"):
            norms.append((f"blocks.{idx}.{name}", getattr(block, name), model.d_model))
        for name in (
            "attention_input_norm",
            "feed_forward_input_norm",
            "latent_up_proj_input_norm",
        ):
            norm = getattr(block, name, None)
            if norm is not None:
                width = (
                    block.latent_up_proj.in_features
                    if name == "latent_up_proj_input_norm"
                    else model.d_model
                )
                norms.append((f"blocks.{idx}.{name}", norm, width))
    for name, norm, width in norms:
        _validate_olmo3moe_norm(norm, name=name, eps=eps, weight_shape=(width,))


def _validate_olmo3moe_shared_experts(block: Any) -> None:
    shared = getattr(block, "shared_experts", None)
    if shared is not None and shared.num_experts != 1:
        raise NotImplementedError("HF export supports exactly one shared expert per block.")
    if getattr(block, "shared_experts_router", None) is not None:
        raise NotImplementedError("HF export does not support a shared expert router.")


def _olmo3moe_attention_gate(attention: Attention) -> tuple[Optional[str], bool]:
    if attention.gate is None:
        return None, True
    if attention.gate.granularity not in (GateGranularity.headwise, GateGranularity.elementwise):
        raise NotImplementedError(f"Unsupported attention gate {attention.gate.granularity!r}.")
    return str(attention.gate.granularity), attention.gate.full_precision


def _olmo3moe_attention_signature(attention: Attention, rms_norm_eps: float) -> tuple:
    """Validate Q/K normalization and describe the HF model's attention configuration."""
    if attention.clip_qkv is not None:
        raise NotImplementedError("HF export does not support attention clip_qkv.")
    if attention.backend.scale not in (None, attention.head_dim**-0.5):
        raise NotImplementedError("HF export does not support a custom attention softmax scale.")
    if any(
        projection.bias is not None
        for projection in (attention.w_q, attention.w_k, attention.w_v, attention.w_out)
    ):
        raise NotImplementedError("HF export does not support attention projection biases.")
    if not attention.use_head_qk_norm or attention.q_norm is None or attention.k_norm is None:
        raise NotImplementedError("HF export requires head-wise QK norm.")
    per_head = attention.q_norm.weight is not None and attention.q_norm.weight.ndim == 2
    for norm, heads in (
        (attention.q_norm, attention.n_heads),
        (attention.k_norm, attention.n_kv_heads),
    ):
        shape = (heads, attention.head_dim) if per_head else (attention.head_dim,)
        _validate_olmo3moe_norm(norm, name="Q/K norm", eps=rms_norm_eps, weight_shape=shape)
    return (
        attention.n_heads,
        attention.n_kv_heads,
        attention.head_dim,
        per_head,
        attention.scalable_softmax,
        attention.q_norm.eps,
        attention.k_norm.eps,
        *_olmo3moe_attention_gate(attention),
    )


def _olmo3moe_kda_norm_eps(attention: KimiDeltaAttention) -> float:
    eps = getattr(attention.o_norm, "eps", getattr(attention.o_norm, "variance_epsilon", None))
    if eps is None:
        raise NotImplementedError("Unable to determine the KDA output-norm epsilon for HF export.")
    return float(eps)


def _get_olmo3moe_config(model: "OLMoDDPModel") -> PretrainedConfig:
    from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock

    if any(isinstance(block.attention, KimiDeltaAttention) for block in model.blocks.values()):
        return _get_olmo3moe_kda_emo_config(model)

    if Olmo3MoeConfig is None:
        raise RuntimeError(
            "Building an Olmo3MoeConfig requires the olmo3moe HF model files "
            "(olmo_core.nn.moe.v2.hf)."
        )

    _register_olmo3moe_auto_classes()

    blocks = list(model.blocks.values())

    # Identify the dense (non-MoE) layers and pick a representative MoE and dense block.
    dense_layers_indices: List[int] = []
    moe_block: Optional[OLMoDDPTransformerBlock] = None
    dense_block: Optional[TransformerBlock] = None
    shared_dense_block: Optional[OLMoDDPTransformerBlock] = None
    for idx, block in enumerate(blocks):
        if isinstance(block, OLMoDDPTransformerBlock):
            if block.routed_experts is None:
                dense_layers_indices.append(idx)
                shared_dense_block = block
            elif moe_block is None:
                moe_block = block
        else:
            # olmo3moe places the layernorms after attention/MLP (reordered norm); a standard
            # pre-norm dense block would export with its norms in the wrong position.
            if not isinstance(block, ReorderedNormTransformerBlock):
                raise NotImplementedError(
                    f"Exporting olmo3moe requires reordered-norm dense blocks, got "
                    f"{type(block).__name__}."
                )
            dense_layers_indices.append(idx)
            if dense_block is None:
                dense_block = block

    if moe_block is None:
        raise NotImplementedError(
            f"No {OLMoDDPTransformerBlock.__name__} found, unable to build HF config for "
            f"{model.__class__.__name__}"
        )

    use_peri_ln = moe_block.use_peri_norm
    if any(
        block.use_peri_norm != use_peri_ln or block.use_pre_norm
        for block in blocks
        if isinstance(block, OLMoDDPTransformerBlock)
    ):
        raise NotImplementedError("HF export requires homogeneous peri-LN and no pre-norm blocks.")
    if use_peri_ln and dense_block is not None:
        raise NotImplementedError("Peri-LN export requires shared-expert dense blocks.")
    if dense_block is not None and shared_dense_block is not None:
        raise NotImplementedError(
            "HF export cannot mix shared-expert and feed-forward dense layouts."
        )

    attention = moe_block.attention
    if not isinstance(attention, Attention):
        raise NotImplementedError(
            f"Attention is not a {Attention.__name__}, unable to build HF config for "
            f"{model.__class__.__name__}"
        )
    if attention.rope is None:
        raise NotImplementedError(
            f"Attention does not use rope, unable to build HF config for "
            f"{model.__class__.__name__}"
        )
    # The olmo3moe converter only round-trips head-wise QK-norm, unscaled RoPE, and bias-free
    # attention; reject anything else rather than silently exporting a divergent model.
    if attention.rope.scaling is not None:
        raise NotImplementedError("Exporting olmo3moe with scaled RoPE is not supported.")
    if not attention.use_head_qk_norm or attention.q_norm is None:
        raise NotImplementedError(
            "Exporting olmo3moe requires head-wise QK-norm (use_head_qk_norm=True); other "
            "QK-norm configurations are not supported."
        )
    rms_norm_eps = moe_block.feed_forward_norm.eps
    _validate_olmo3moe_model_norms(model, rms_norm_eps)
    attention_signature = _olmo3moe_attention_signature(attention, rms_norm_eps)
    for block in blocks:
        _validate_olmo3moe_shared_experts(block)
        if not isinstance(block.attention, Attention):
            raise NotImplementedError("HF export requires Attention in every non-KDA layer.")
        rope = block.attention.rope
        if rope is None or rope.theta != attention.rope.theta or rope.scaling is not None:
            raise NotImplementedError("Non-KDA HF export requires identical unscaled RoPE.")
        if _olmo3moe_attention_signature(block.attention, rms_norm_eps) != attention_signature:
            raise NotImplementedError("Heterogeneous attention configurations are unsupported.")

    if moe_block.routed_experts is None or moe_block.routed_experts_router is None:
        raise NotImplementedError("MoE block is missing routed experts or its router.")

    routed_experts = moe_block.routed_experts
    router = moe_block.routed_experts_router
    # Selection modifiers change which experts a token routes to at inference. The HF Olmo3Moe
    # router only implements plain softmax/sigmoid gating with no score-bias or group-masking
    # path, so exporting any of these would silently diverge (or crash on the first HF forward).
    router_signature = None
    for block in blocks:
        if isinstance(block, OLMoDDPTransformerBlock) and block.routed_experts_router is not None:
            block_router = block.routed_experts_router
            _validate_olmo3moe_router_selection(block_router)
            block_emo = getattr(block_router, "emo", None)
            signature = (
                block_router.num_experts,
                block_router.top_k,
                block_router.original_top_k,
                block_router.gating_function,
                block_router.normalize_expert_weights,
                block_router.restore_weight_scale,
                block_emo,
            )
            if router_signature is None:
                router_signature = signature
            elif signature != router_signature:
                raise NotImplementedError("Heterogeneous router configurations are unsupported.")

    # The HF olmo3moe router/expert linears are bias-free and the converter only copies
    # contiguous SwiGLU up/gate weights, so biased or non-SwiGLU experts can't be represented.
    if router.bias is not None:
        raise NotImplementedError("Exporting olmo3moe with a biased router is not supported.")
    if routed_experts.bias:
        raise NotImplementedError("Exporting olmo3moe with biased routed experts is not supported.")
    if routed_experts.activation.value != "swiglu":
        raise NotImplementedError(
            f"Exporting olmo3moe with routed-expert activation "
            f"{routed_experts.activation.value!r} is not supported (only SwiGLU)."
        )
    shared_experts = moe_block.shared_experts
    if shared_experts is not None and shared_experts.activation.value != "swiglu":
        raise NotImplementedError(
            f"Exporting olmo3moe with shared-expert activation "
            f"{shared_experts.activation.value!r} is not supported (only SwiGLU)."
        )

    # Dense MLP intermediate size, if there are any dense layers.
    dense_mlp_intermediate_size: Optional[int] = None
    for idx in dense_layers_indices:
        block = blocks[idx]
        if isinstance(block, OLMoDDPTransformerBlock):
            shared = block.shared_experts
            if shared is None or shared.activation.value != "swiglu":
                raise NotImplementedError(
                    "HF dense export requires a bias-free SwiGLU shared expert."
                )
            hidden_size = shared.hidden_size
        else:
            feed_forward = block.feed_forward
            if feed_forward.activation_fn is not F.silu:
                raise NotImplementedError(
                    "HF export requires SiLU in every dense feed-forward layer."
                )
            if any(
                proj.bias is not None
                for proj in (feed_forward.w1, feed_forward.w2, feed_forward.w3)
            ):
                raise NotImplementedError(
                    "HF export does not support biased dense feed-forward layers."
                )
            hidden_size = feed_forward.hidden_size
        if dense_mlp_intermediate_size is None:
            dense_mlp_intermediate_size = hidden_size
        elif hidden_size != dense_mlp_intermediate_size:
            raise NotImplementedError("Heterogeneous dense feed-forward sizes are unsupported.")

    # Shared experts (optional). The HF model has a single shared expert.
    shared_expert_intermediate_size: Optional[int] = None
    if moe_block.shared_experts is not None:
        if moe_block.shared_experts.num_experts > 1:
            raise NotImplementedError(
                "Exporting olmo3moe with more than one shared expert is not supported."
            )
        shared_expert_intermediate_size = moe_block.shared_experts.hidden_size

    # Sliding window: Olmo-core stores a per-layer window on the attention backend; a value of
    # (-1, -1) means full attention. HF expects a value one larger than the flash-attention window
    # (which excludes the current position); see the OLMo 3 handling in `get_hf_config`.
    layer_types: List[str] = []
    window_sizes = set()
    for block in blocks:
        window = block.attention.backend.window_size
        if window != (-1, -1):
            layer_types.append("sliding_attention")
            window_sizes.add(window[0])
        else:
            layer_types.append("full_attention")

    if len(window_sizes) > 1:
        raise ValueError(
            "All sliding window attention layers must have the same window size for "
            f"Olmo3MoeConfig. Found different window sizes: {window_sizes}."
        )
    sliding_window = (window_sizes.pop() + 1) if window_sizes else attention.head_dim

    attention_hidden_size = attention.n_heads * attention.head_dim
    gate_type, gate_full_precision = _olmo3moe_attention_gate(attention)
    emo = getattr(router, "emo", None)

    return Olmo3MoeConfig(
        vocab_size=model.vocab_size,
        hidden_size=model.d_model,
        attention_hidden_size=attention_hidden_size,
        head_dim=attention.head_dim,
        dense_mlp_intermediate_size=dense_mlp_intermediate_size,
        moe_intermediate_size=routed_experts.hidden_size,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        n_routed_experts=routed_experts.num_experts,
        num_experts_per_tok=router.top_k,
        original_num_experts_per_tok=router.original_top_k,
        num_hidden_layers=model.n_layers,
        num_attention_heads=attention.n_heads,
        num_key_value_heads=attention.n_kv_heads,
        hidden_act="silu",
        gating_function=str(router.gating_function),
        normalize_expert_weights=router.normalize_expert_weights,
        restore_weight_scale=router.restore_weight_scale,
        max_position_embeddings=-1,
        attention_bias=attention.w_out.bias is not None,
        rope_theta=attention.rope.theta,
        rope_scaling=None,
        rms_norm_eps=moe_block.feed_forward_norm.eps,
        use_head_qk_norm=attention.use_head_qk_norm,
        qk_norm_per_head_gains=attention.q_norm.weight.ndim == 2,
        scalable_softmax=attention.scalable_softmax,
        attention_gate_type=gate_type,
        attention_gate_full_precision=gate_full_precision,
        use_cache=not attention.scalable_softmax,
        sliding_window=sliding_window,
        layer_types=layer_types,
        dense_layers_indices=dense_layers_indices,
        dense_layers_use_shared_expert=shared_dense_block is not None,
        embed_scale=model.embed_scale if model.embed_scale is not None else 1.0,
        embed_norm=model.embedding_norm is not None,
        use_peri_ln=use_peri_ln,
        emo_min_document_expert_pool=emo.min_document_expert_pool if emo is not None else None,
        emo_max_document_expert_pool=emo.max_document_expert_pool if emo is not None else None,
        emo_eval_document_expert_pool=emo.eval_pool_size() if emo is not None else None,
        emo_eos_token_id=emo.eos_token_id if emo is not None else None,
        emo_routing_mode="full_pool" if emo is not None and emo.full_pool else None,
        emo_source_config=emo.source_config if emo is not None else None,
        pad_token_id=None,  # type: ignore
        bos_token_id=None,
        eos_token_id=None,  # type: ignore
        tie_word_embeddings=model.tie_word_embeddings,
    )


def _get_olmo3moe_kda_emo_config(model: "OLMoDDPModel") -> PretrainedConfig:
    """Build a fail-closed HF config for hybrid KDA MoE, with optional latent projections/EMO."""

    from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock

    if Olmo3MoeConfig is None:
        raise RuntimeError("The olmo3moe HF model files are unavailable.")
    _register_olmo3moe_auto_classes()

    blocks = list(model.blocks.values())
    if not blocks or not all(isinstance(block, OLMoDDPTransformerBlock) for block in blocks):
        raise NotImplementedError("KDA + EMo export requires OLMoDDP blocks at every layer.")

    dense_layers_indices = [idx for idx, block in enumerate(blocks) if block.routed_experts is None]
    sparse_blocks = [block for block in blocks if block.routed_experts is not None]
    kda_blocks = [block for block in blocks if isinstance(block.attention, KimiDeltaAttention)]
    attention_blocks = [block for block in blocks if isinstance(block.attention, Attention)]
    if not dense_layers_indices or not sparse_blocks or not kda_blocks or not attention_blocks:
        raise NotImplementedError(
            "KDA + EMo export requires dense, sparse, KDA, and full-attention layers."
        )

    representative = sparse_blocks[0]
    _validate_olmo3moe_model_norms(model, representative.feed_forward_norm.eps)
    routed_experts = representative.routed_experts
    router = representative.routed_experts_router
    assert routed_experts is not None and router is not None
    sparse_signature = None
    for block in sparse_blocks:
        assert block.routed_experts is not None and block.routed_experts_router is not None
        block_router = block.routed_experts_router
        block_experts = block.routed_experts
        _validate_olmo3moe_router_selection(block_router)
        emo = getattr(block_router, "emo", None)
        latent = block.latent_down_proj
        latent_norm = block.latent_up_proj_input_norm
        if block_router.bias is not None:
            raise NotImplementedError("Exporting KDA + EMo with a biased router is unsupported.")
        if block_experts.bias:
            raise NotImplementedError(
                "Exporting KDA + EMo with biased routed experts is unsupported."
            )
        if block_experts.activation.value != "swiglu":
            raise NotImplementedError(
                "Exporting KDA + EMo requires SwiGLU routed experts, got "
                f"{block_experts.activation.value!r}."
            )
        signature = (
            block_experts.d_model,
            block_experts.hidden_size,
            block_experts.num_experts,
            block_router.top_k,
            block_router.original_top_k,
            block_router.gating_function,
            block_router.normalize_expert_weights,
            block_router.restore_weight_scale,
            block_router.global_load_balancing,
            emo.min_document_expert_pool if emo is not None else None,
            emo.max_document_expert_pool if emo is not None else None,
            emo.eval_pool_size() if emo is not None else None,
            emo.eos_token_id if emo is not None else None,
            emo.full_pool if emo is not None else None,
            emo.source_config if emo is not None else None,
            latent.out_features if latent is not None else None,
            latent.bias is not None if latent is not None else False,
            latent_norm is not None,
            block.shared_experts.hidden_size if block.shared_experts is not None else None,
        )
        if sparse_signature is None:
            sparse_signature = signature
        elif signature != sparse_signature:
            raise NotImplementedError("Heterogeneous sparse EMo layers are unsupported.")

    for block in blocks:
        _validate_olmo3moe_shared_experts(block)
        if block.shared_experts is not None and block.shared_experts.activation.value != "swiglu":
            raise NotImplementedError(
                "Exporting KDA + EMo requires SwiGLU shared experts, got "
                f"{block.shared_experts.activation.value!r}."
            )

    experimental_layers = [
        idx
        for idx, block in enumerate(blocks)
        if isinstance(block.attention, KimiDeltaAttention)
        and block.attention.use_experimental_kernels
    ]
    if experimental_layers:
        # Kernel choice is an execution backend, not a different model architecture.
        # Export intentionally uses portable FLA; exact state conversion does not
        # certify forward equivalence at shapes that engage experimental kernels.
        log.warning(
            "HF export uses standard FLA KDA/convolution kernels for source experimental "
            "layers %s. Kernels are not numerically identical; validate full-model forward "
            "agreement at representative production lengths, not only tensor round trips "
            "or short fallback sequences.",
            experimental_layers,
        )

    kda = kda_blocks[0].attention
    assert isinstance(kda, KimiDeltaAttention)
    for block in kda_blocks:
        mixer = block.attention
        assert isinstance(mixer, KimiDeltaAttention)
        if any(conv.bias is not None for conv in (mixer.q_conv1d, mixer.k_conv1d, mixer.v_conv1d)):
            raise NotImplementedError("KDA HF export does not support convolution bias.")
    kda_norm_eps = _olmo3moe_kda_norm_eps(kda)
    kda_signature = (
        kda.n_heads,
        kda.n_v_heads,
        kda.head_k_dim,
        kda.head_v_dim,
        kda.conv_size,
        kda.allow_neg_eigval,
        kda_norm_eps,
    )
    for block in kda_blocks[1:]:
        other = block.attention
        assert isinstance(other, KimiDeltaAttention)
        if (
            other.n_heads,
            other.n_v_heads,
            other.head_k_dim,
            other.head_v_dim,
            other.conv_size,
            other.allow_neg_eigval,
            _olmo3moe_kda_norm_eps(other),
        ) != kda_signature:
            raise NotImplementedError(
                "Heterogeneous KDA layer shapes or output-norm epsilons are unsupported."
            )

    attention = attention_blocks[0].attention
    assert isinstance(attention, Attention)
    rms_norm_eps = representative.feed_forward_norm.eps
    attention_signature = _olmo3moe_attention_signature(attention, rms_norm_eps)
    assert attention.q_norm is not None and attention.k_norm is not None
    if any(
        _olmo3moe_attention_signature(block.attention, rms_norm_eps) != attention_signature
        for block in attention_blocks
    ):
        raise NotImplementedError("Heterogeneous full-attention configurations are unsupported.")
    ropes = [block.attention.rope for block in attention_blocks]
    if any(rope is None for rope in ropes) != all(rope is None for rope in ropes):
        raise NotImplementedError("Full-attention layers must consistently enable or disable RoPE.")
    rope_theta = None
    rope_scaling = None
    if attention.rope is not None:
        rope_theta = attention.rope.theta
        if any(rope is None or rope.theta != rope_theta for rope in ropes):
            raise NotImplementedError(
                "Heterogeneous full-attention RoPE theta values are unsupported."
            )
        rope_scaling = _get_and_validate_rope_scaling_config(attention_blocks)
    if attention.q_norm is None or attention.k_norm is None or not attention.use_head_qk_norm:
        raise NotImplementedError("HF export requires head-wise QK norm.")

    gate_type, gate_full_precision = _olmo3moe_attention_gate(attention)

    if any(not block.use_peri_norm or block.use_pre_norm for block in blocks):
        raise NotImplementedError("KDA + EMo export requires peri-norm without pre-norm.")

    dense_hidden_sizes = {
        blocks[idx].shared_experts.hidden_size
        for idx in dense_layers_indices
        if blocks[idx].shared_experts is not None
    }
    if len(dense_hidden_sizes) != 1:
        raise NotImplementedError("Dense layers must share one MLP width.")
    shared_hidden = (
        representative.shared_experts.hidden_size
        if representative.shared_experts is not None
        else None
    )
    layer_types = []
    window_sizes = set()
    for block in blocks:
        if isinstance(block.attention, KimiDeltaAttention):
            layer_types.append("linear_attention")
        elif block.attention.backend.window_size != (-1, -1):
            layer_types.append("sliding_attention")
            window_sizes.add(block.attention.backend.window_size[0])
        else:
            layer_types.append("full_attention")
    if len(window_sizes) > 1:
        raise NotImplementedError("Hybrid KDA HF export requires one common sliding window size.")
    window_kwargs = {"sliding_window": window_sizes.pop() + 1} if window_sizes else {}
    latent = representative.latent_down_proj
    emo = getattr(router, "emo", None)

    return Olmo3MoeConfig(
        vocab_size=model.vocab_size,
        hidden_size=model.d_model,
        attention_hidden_size=attention.n_heads * attention.head_dim,
        head_dim=attention.head_dim,
        dense_mlp_intermediate_size=next(iter(dense_hidden_sizes)),
        moe_intermediate_size=routed_experts.hidden_size,
        shared_expert_intermediate_size=shared_hidden,
        n_routed_experts=routed_experts.num_experts,
        num_experts_per_tok=router.top_k,
        original_num_experts_per_tok=router.original_top_k,
        num_hidden_layers=model.n_layers,
        num_attention_heads=attention.n_heads,
        num_key_value_heads=attention.n_kv_heads,
        gating_function=str(router.gating_function),
        normalize_expert_weights=router.normalize_expert_weights,
        restore_weight_scale=router.restore_weight_scale,
        max_position_embeddings=-1,
        use_head_qk_norm=True,
        qk_norm_per_head_gains=attention.q_norm.weight.ndim == 2,
        scalable_softmax=attention.scalable_softmax,
        use_rope=attention.rope is not None,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        attention_gate_type=gate_type,
        attention_gate_full_precision=gate_full_precision,
        linear_num_key_heads=kda.n_heads,
        linear_num_value_heads=kda.n_v_heads,
        linear_key_head_dim=kda.head_k_dim,
        linear_value_head_dim=kda.head_v_dim,
        linear_conv_kernel_dim=kda.conv_size,
        linear_allow_neg_eigval=kda.allow_neg_eigval,
        linear_norm_eps=kda_norm_eps,
        latent_moe_dim=latent.out_features if latent is not None else None,
        latent_moe_bias=latent.bias is not None if latent is not None else False,
        latent_moe_up_proj_input_norm=representative.latent_up_proj_input_norm is not None,
        **window_kwargs,
        layer_types=layer_types,
        dense_layers_indices=dense_layers_indices,
        dense_layers_use_shared_expert=True,
        embed_scale=model.embed_scale if model.embed_scale is not None else 1.0,
        embed_norm=model.embedding_norm is not None,
        use_peri_ln=True,
        rms_norm_eps=representative.feed_forward_norm.eps,
        emo_min_document_expert_pool=emo.min_document_expert_pool if emo is not None else None,
        emo_max_document_expert_pool=emo.max_document_expert_pool if emo is not None else None,
        emo_eval_document_expert_pool=emo.eval_pool_size() if emo is not None else None,
        emo_eos_token_id=emo.eos_token_id if emo is not None else None,
        emo_routing_mode="full_pool" if emo is not None and emo.full_pool else None,
        emo_source_config=emo.source_config if emo is not None else None,
        global_load_balancing=router.global_load_balancing,
        use_cache=False,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=emo.eos_token_id if emo is not None else None,
        tie_word_embeddings=model.tie_word_embeddings,
    )


@beta_feature
def get_hf_config(model: Transformer) -> PretrainedConfig:
    if OLMoDDPModel is not None and isinstance(model, OLMoDDPModel):
        return _get_olmo3moe_config(model)

    if isinstance(model, NormalizedTransformer):
        raise NotImplementedError(
            f"Building HF config not implemented for {model.__class__.__name__}"
        )

    if isinstance(model, MoETransformer):
        return _get_flex_olmo_config(model)

    blocks = list(model.blocks.values())
    first_block = blocks[0]
    if not isinstance(first_block, ReorderedNormTransformerBlock):
        raise NotImplementedError(
            f"Block is not a {ReorderedNormTransformerBlock.__name__}, unable to build HF config for {model.__class__.__name__}"
        )

    if not isinstance(first_block.attention, Attention):
        raise NotImplementedError(
            f"Attention is not a {Attention.__name__}, unable to build HF config for {model.__class__.__name__}"
        )
    if first_block.attention.backend is None:
        raise ValueError("Attention backend is not set.")

    has_rope = first_block.attention.rope is not None

    if has_rope:
        rope_scaling = _get_and_validate_rope_scaling_config(blocks)
        rope_theta = first_block.attention.rope.theta
    else:
        rope_scaling = None
        rope_theta = None

    # Extract common configuration parameters
    common_config_args = {
        "vocab_size": model.vocab_size,
        "hidden_size": model.d_model,
        "intermediate_size": first_block.feed_forward.hidden_size,
        "num_hidden_layers": model.n_layers,
        "num_attention_heads": first_block.attention.n_heads,
        "num_key_value_heads": first_block.attention.n_kv_heads,
        "hidden_act": "silu",
        "max_position_embeddings": -1,
        "attention_bias": first_block.attention.w_out.bias is not None,
        "rope_theta": rope_theta,
        "rope_scaling": rope_scaling,
        "pad_token_id": None,
        "bos_token_id": None,
        "eos_token_id": None,
        "rms_norm_eps": first_block.feed_forward_norm.eps,
        "tie_word_embeddings": model.tie_word_embeddings,
    }

    # The OLMo 3 model family is identical to the OLMo 2 model family, except:
    # - Sliding window attention is used for 3 out of 4 layers.
    # - RoPE scaling is not applied to sliding window attention layers.
    # Therefore, if any layer uses sliding window attention, we assume the model is OLMo 3.
    # Identify layers that use sliding window attention.
    sliding_window_blocks = [
        block for block in blocks if block.attention.backend.window_size != (-1, -1)
    ]

    if sliding_window_blocks:
        if Olmo3Config is None:
            raise RuntimeError("The installed transformers version does not support Olmo3")

        found_window_sizes = {
            block.attention.backend.window_size[0] for block in sliding_window_blocks
        }

        if len(found_window_sizes) > 1:
            raise ValueError(
                "All sliding window attention layers must have the same window size for "
                f"OLMo3Config. Found different window sizes: {found_window_sizes}."
            )

        # This sliding window sizes value is configured to be fed to flash_attention -
        # it is one smaller than the actual window size because FA implicitly includes the
        # current position in the window. HF expects a value one larger than this and will
        # manually adjust the window size down by 1 for FA.
        # See https://github.com/huggingface/transformers/pull/40163
        common_window_size_value = found_window_sizes.pop()

        olmo3_specific_args = {
            "sliding_window": common_window_size_value + 1,
            "layer_types": [
                "sliding_attention"
                if block.attention.backend.window_size != (-1, -1)
                else "full_attention"
                for block in blocks
            ],
        }
        return Olmo3Config(**common_config_args, **olmo3_specific_args)
    else:
        return Olmo2Config(**common_config_args)


def _get_and_validate_rope_scaling_config(blocks) -> dict | None:
    """
    Validate RoPE scaling configuration across transformer blocks.

    :param blocks: The list of transformer blocks to validate.
    :returns: The validated RoPE scaling config dict for HF, or None if no scaling.
    :raises NotImplementedError: If RoPE scaling is applied to sliding window layers or if
                               full attention layers have different RoPE scaling configs.
    """
    # Separate full attention layers from sliding window layers
    full_attention_layers = [
        (idx, block)
        for idx, block in enumerate(blocks)
        if block.attention.backend.window_size == (-1, -1)
    ]
    sliding_window_layers = [
        (idx, block)
        for idx, block in enumerate(blocks)
        if block.attention.backend.window_size != (-1, -1)
    ]

    # Check for RoPE scaling on sliding window layers (not allowed)
    sliding_with_scaling = [
        (idx, block)
        for idx, block in sliding_window_layers
        if block.attention.rope is not None and block.attention.rope.scaling is not None
    ]
    if sliding_with_scaling:
        sliding_indices = [idx for idx, _ in sliding_with_scaling]
        raise NotImplementedError(
            f"RoPE scaling is configured on sliding window attention layers {sliding_indices}, "
            f"but HuggingFace only supports RoPE scaling on full attention layers. "
            f"Please remove RoPE scaling from sliding window layers or convert them to full attention."
        )

    # Collect RoPE scaling configs from full attention layers only
    full_layers_with_scaling = [
        (idx, block)
        for idx, block in full_attention_layers
        if block.attention.rope is not None and block.attention.rope.scaling is not None
    ]
    if not full_layers_with_scaling:
        return None

    rope_scaling_configs: list[RoPEScalingConfig] = [
        block.attention.rope.scaling for _, block in full_layers_with_scaling
    ]

    # Validate that all full attention layers with RoPE scaling use the same configuration
    first_config = rope_scaling_configs[0]
    first_config_dict = first_config.to_hf_config()

    for i, rope_config in enumerate(rope_scaling_configs[1:], 1):
        config_dict = rope_config.to_hf_config()
        if config_dict != first_config_dict:
            scaling_indices = [idx for idx, _ in full_layers_with_scaling]
            raise NotImplementedError(
                f"Full attention layers have different RoPE scaling configurations but HuggingFace "
                "only supports a single RoPE scaling configuration per model. "
                f"Full attention layers with scaling: {scaling_indices}. "
                f"First config: {first_config_dict}, Different config at layer {i}: {config_dict}"
            )

    return first_config_dict


# ---------------------------------------------------------------------------
# Hybrid model helpers
# ---------------------------------------------------------------------------


@beta_feature
def is_olmo_hybrid_model(model: Transformer) -> bool:
    """Return ``True`` if the model has both :class:`GatedDeltaNet` and :class:`Attention` layers."""
    has_gdn = False
    has_attn = False
    for block in model.blocks.values():
        if isinstance(block.attention, GatedDeltaNet):
            has_gdn = True
        elif isinstance(block.attention, Attention):
            has_attn = True
        if has_gdn and has_attn:
            return True
    return False


@beta_feature
def get_hybrid_layer_types(model: Transformer) -> List[str]:
    """
    Return a per-layer type list for a hybrid model.

    Each entry is ``"linear_attention"`` (GDN) or ``"full_attention"`` (standard attention),
    matching the HF ``olmo_hybrid`` config format.
    """
    layer_types: List[str] = []
    for idx, block in model.blocks.items():
        if isinstance(block.attention, GatedDeltaNet):
            layer_types.append("linear_attention")
        elif isinstance(block.attention, Attention):
            layer_types.append("full_attention")
        else:
            raise ValueError(f"Unknown sequence mixer type at layer {idx}: {type(block.attention)}")
    return layer_types


def _get_hybrid_rope_scaling(model: Transformer, layer_types: List[str]) -> Optional[dict]:
    """
    Extract the RoPE scaling config from attention blocks.  GDN layers are skipped
    because they don't use RoPE.
    """
    attn_blocks = [
        (int(idx), block)
        for idx, block in model.blocks.items()
        if layer_types[int(idx)] == "full_attention"
    ]

    layers_with_scaling = [
        (idx, block)
        for idx, block in attn_blocks
        if block.attention.rope is not None and block.attention.rope.scaling is not None
    ]
    if not layers_with_scaling:
        return None

    first_config = layers_with_scaling[0][1].attention.rope.scaling.to_hf_config()
    for idx, block in layers_with_scaling[1:]:
        cfg = block.attention.rope.scaling.to_hf_config()
        if cfg != first_config:
            raise NotImplementedError(
                f"Inconsistent RoPE scaling configs. First: {first_config}, Layer {idx}: {cfg}"
            )
    return first_config


@beta_feature
def get_hybrid_hf_config(
    model: Transformer,
    layer_types: List[str],
    max_seq_len: int = 65536,
) -> Dict[str, Any]:
    """
    Build the ``config.json`` dict for a HF ``olmo_hybrid`` model.

    Returns a plain dict (not :class:`PretrainedConfig`) to avoid a hard dependency
    on a specific ``transformers`` version.

    :param model: The Olmo-core hybrid transformer model.
    :param layer_types: Per-layer type list from :func:`get_hybrid_layer_types`.
    :param max_seq_len: Maximum sequence length for ``max_position_embeddings``.
    """
    blocks = list(model.blocks.values())

    attn_block: Optional[TransformerBlock] = None
    gdn_block: Optional[TransformerBlock] = None
    for lt, block in zip(layer_types, blocks):
        if lt == "full_attention" and attn_block is None:
            attn_block = block
        elif lt == "linear_attention" and gdn_block is None:
            gdn_block = block

    if attn_block is None:
        raise ValueError("Hybrid model must have at least one attention layer")
    if gdn_block is None:
        raise ValueError("Hybrid model must have at least one GDN layer")

    attn: Attention = attn_block.attention
    gdn: GatedDeltaNet = gdn_block.attention

    # RoPE (from attention blocks only)
    rope_parameters: Optional[dict] = None
    if attn.rope is not None:
        rope_theta = float(attn.rope.theta)
        rope_scaling = _get_hybrid_rope_scaling(model, layer_types)
        rope_parameters = {"rope_theta": rope_theta}
        if rope_scaling:
            rope_parameters.update(rope_scaling)
        else:
            rope_parameters["rope_type"] = "default"
        log.info(f"RoPE: {rope_parameters}")
    else:
        log.info("No RoPE configured")

    # Warn if GDN blocks are post-norm but HF expects pre-norm.
    if isinstance(gdn_block, ReorderedNormTransformerBlock):
        log.warning(
            "GDN block uses post-norm (ReorderedNormTransformerBlock) but HF olmo_hybrid "
            "expects pre-norm for linear_attention layers. The conversion will proceed, but "
            "outputs may not match exactly."
        )

    config: Dict[str, Any] = {
        "model_type": "olmo_hybrid",
        "architectures": ["OlmoHybridForCausalLM"],
        # Standard transformer fields
        "vocab_size": model.vocab_size,
        "hidden_size": model.d_model,
        "intermediate_size": attn_block.feed_forward.hidden_size,
        "num_hidden_layers": len(blocks),
        "num_attention_heads": attn.n_heads,
        "num_key_value_heads": attn.n_kv_heads,
        "hidden_act": "silu",
        "max_position_embeddings": max_seq_len,
        "initializer_range": 0.02,
        "use_cache": True,
        "attention_bias": attn.w_out.bias is not None,
        "attention_dropout": 0.0,
        "rms_norm_eps": attn_block.feed_forward_norm.eps,  # todo: revisit
        "tie_word_embeddings": model.tie_word_embeddings,
        # Hybrid layer configuration
        "layer_types": layer_types,
        # GDN (linear attention) parameters
        "linear_num_key_heads": gdn.n_heads,
        "linear_num_value_heads": gdn.n_v_heads,
        "linear_key_head_dim": gdn.head_k_dim,
        "linear_value_head_dim": gdn.head_v_dim,
        "linear_conv_kernel_dim": gdn.conv_size,
        "linear_allow_neg_eigval": gdn.allow_neg_eigval,
        # Token IDs (updated later after tokenizer is saved)
        "pad_token_id": None,
        "bos_token_id": None,
        "eos_token_id": None,
    }

    if rope_parameters is not None:
        config["rope_parameters"] = rope_parameters
    else:
        config["rope_theta"] = None

    return config
