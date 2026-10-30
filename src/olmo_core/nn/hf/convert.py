import logging
import re
from typing import Any, Dict, Iterator, List, Mapping

import torch
from transformers import PretrainedConfig

from olmo_core.doc_utils import beta_feature
from olmo_core.nn.conversion.state_converter import StateConverter
from olmo_core.nn.conversion.state_mapping import (
    StateMappingTemplate,
    StateType,
    TemplatePlaceholder,
)

log = logging.getLogger(__name__)

LAYER = TemplatePlaceholder.LAYER
EXPERT = TemplatePlaceholder.EXPERT


#: Map of Hugging Face weight keys to OLMo Core weight keys, that is used to determine how HF state
#: maps to OLMo Core state. Different HF models may use different names for a given OLMo
#: Core state. You may configure this to change how HF state maps to OLMo Core state.
#:
#: This map only captures one-to-one mappings from HF to OLMo Core. For many-to-many mappings
#: or mappings that require additional manipulation of state, see
#: :data:`HF_TO_OLMO_CORE_TEMPLATE_MAPPINGS`. If a given HF key can refer to different OLMo Core
#: states depending on the HF model, see :data:`MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_WEIGHT_MAPPINGS`.
HF_TO_OLMO_CORE_WEIGHT_MAPPINGS: Dict[str, str] = {
    "model.embed_tokens.weight": "embeddings.weight",
    "model.norm.weight": "lm_head.norm.weight",
    "lm_head.weight": "lm_head.w_out.weight",
    # Attention.
    f"model.layers.{LAYER}.self_attn.q_proj.weight": f"blocks.{LAYER}.attention.w_q.weight",
    f"model.layers.{LAYER}.self_attn.k_proj.weight": f"blocks.{LAYER}.attention.w_k.weight",
    f"model.layers.{LAYER}.self_attn.v_proj.weight": f"blocks.{LAYER}.attention.w_v.weight",
    f"model.layers.{LAYER}.self_attn.o_proj.weight": f"blocks.{LAYER}.attention.w_out.weight",
    # MLP.
    f"model.layers.{LAYER}.mlp.gate_proj.weight": f"blocks.{LAYER}.feed_forward.w1.weight",
    f"model.layers.{LAYER}.mlp.down_proj.weight": f"blocks.{LAYER}.feed_forward.w2.weight",
    f"model.layers.{LAYER}.mlp.up_proj.weight": f"blocks.{LAYER}.feed_forward.w3.weight",
    # Layer norms.
    f"model.layers.{LAYER}.input_layernorm.weight": f"blocks.{LAYER}.attention_norm.weight",
    f"model.layers.{LAYER}.post_attention_layernorm.weight": f"blocks.{LAYER}.attention_norm.weight",
    f"model.layers.{LAYER}.post_feedforward_layernorm.weight": f"blocks.{LAYER}.feed_forward_norm.weight",
    f"model.layers.{LAYER}.self_attn.q_norm.weight": f"blocks.{LAYER}.attention.q_norm.weight",
    f"model.layers.{LAYER}.self_attn.k_norm.weight": f"blocks.{LAYER}.attention.k_norm.weight",
    # MoEMLP.
    f"model.layers.{LAYER}.mlp.gate.weight": f"blocks.{LAYER}.feed_forward_moe.router.weight",
    f"model.layers.{LAYER}.mlp.experts.{EXPERT}.gate_proj.weight": f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w1",
    f"model.layers.{LAYER}.mlp.experts.{EXPERT}.down_proj.weight": f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w2",
    f"model.layers.{LAYER}.mlp.experts.{EXPERT}.up_proj.weight": f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w3",
}


#: Map of Hugging Face module keys to OLMo Core module keys, that is used to determine how HF state
#: maps to OLMo Core state. Different HF models may use different names for a given OLMo
#: Core state. You may configure this to change how HF state maps to OLMo Core state.
#:
#: This map only captures one-to-one mappings from HF to OLMo Core. For many-to-many mappings
#: or mappings that require additional manipulation of state, see
#: :data:`HF_TO_OLMO_CORE_TEMPLATE_MAPPINGS`. If a given HF key can refer to different OLMo Core
#: states depending on the HF model, see :data:`MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_MODULE_MAPPINGS`.
HF_TO_OLMO_CORE_MODULE_MAPPINGS: Dict[str, str] = {
    "model.embed_tokens": "embeddings",
    "model.norm": "lm_head.norm",
    "lm_head": "lm_head.w_out",
    # Attention.
    f"model.layers.{LAYER}.self_attn.q_proj": f"blocks.{LAYER}.attention.w_q",
    f"model.layers.{LAYER}.self_attn.k_proj": f"blocks.{LAYER}.attention.w_k",
    f"model.layers.{LAYER}.self_attn.v_proj": f"blocks.{LAYER}.attention.w_v",
    f"model.layers.{LAYER}.self_attn.o_proj": f"blocks.{LAYER}.attention.w_out",
    # MLP.
    f"model.layers.{LAYER}.mlp.gate_proj": f"blocks.{LAYER}.feed_forward.w1",
    f"model.layers.{LAYER}.mlp.down_proj": f"blocks.{LAYER}.feed_forward.w2",
    f"model.layers.{LAYER}.mlp.up_proj": f"blocks.{LAYER}.feed_forward.w3",
    # Layer norms.
    f"model.layers.{LAYER}.input_layernorm": f"blocks.{LAYER}.attention_norm",
    f"model.layers.{LAYER}.post_attention_layernorm": f"blocks.{LAYER}.attention_norm",
    f"model.layers.{LAYER}.post_feedforward_layernorm": f"blocks.{LAYER}.feed_forward_norm",
    f"model.layers.{LAYER}.self_attn.q_norm": f"blocks.{LAYER}.attention.q_norm",
    f"model.layers.{LAYER}.self_attn.k_norm": f"blocks.{LAYER}.attention.k_norm",
    # MoEMLP.
    f"model.layers.{LAYER}.mlp": f"blocks.{LAYER}.feed_forward_moe",
    f"model.layers.{LAYER}.post_moe_norm": f"blocks.{LAYER}.feed_forward_moe_norm",
    f"model.layers.{LAYER}.mlp.gate": f"blocks.{LAYER}.feed_forward_moe.router",
    # Indices are not part of the original OLMo Core state but can be introduced during conversion for aide debugging.
    f"model.layers.{LAYER}.mlp.gate.indices": f"blocks.{LAYER}.feed_forward_moe.router.indices",
    f"model.layers.{LAYER}.mlp.shared_mlp": f"blocks.{LAYER}.feed_forward_moe.shared_mlp",
    f"model.layers.{LAYER}.mlp.shared_mlp.gate_proj": f"blocks.{LAYER}.feed_forward_moe.shared_mlp.w1",
    f"model.layers.{LAYER}.mlp.shared_mlp.down_proj": f"blocks.{LAYER}.feed_forward_moe.shared_mlp.w2",
    f"model.layers.{LAYER}.mlp.shared_mlp.up_proj": f"blocks.{LAYER}.feed_forward_moe.shared_mlp.w3",
}


#: Map of Hugging Face weight keys to OLMo Core weight keys. This map captures overrides of the standard
#: one-to-one mappings in :data:`HF_TO_OLMO_CORE_WEIGHT_MAPPINGS`, in case a given HF key can refer to
#: different OLMo Core states depending on the HF model architecture. You may configure this to change
#: how HF state maps to OLMo Core state.
MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_WEIGHT_MAPPINGS: Dict[str, Dict[str, str]] = {
    "llama": {
        f"model.layers.{LAYER}.post_attention_layernorm.weight": f"blocks.{LAYER}.feed_forward_norm.weight"
    },
    "gemma3_text": {
        f"model.layers.{LAYER}.post_attention_layernorm.weight": f"blocks.{LAYER}.post_attention_norm.weight",
        f"model.layers.{LAYER}.pre_feedforward_layernorm.weight": f"blocks.{LAYER}.feed_forward_norm.weight",
        f"model.layers.{LAYER}.post_feedforward_layernorm.weight": f"blocks.{LAYER}.post_feed_forward_norm.weight",
    },
    "qwen3": {
        f"model.layers.{LAYER}.post_attention_layernorm.weight": f"blocks.{LAYER}.feed_forward_norm.weight"
    },
    "qwen3_5_text": {
        f"model.layers.{LAYER}.post_attention_layernorm.weight": f"blocks.{LAYER}.feed_forward_norm.weight"
    },
}

#: Map of Hugging Face module keys to OLMo Core module keys. This map captures overrides of the standard
#: one-to-one mappings in :data:`HF_TO_OLMO_CORE_MODULE_MAPPINGS`, in case a given HF key can refer to
#: different OLMo Core states depending on the HF model architecture. You may configure this to change
#: how HF state maps to OLMo Core state.
MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_MODULE_MAPPINGS: Dict[str, Dict[str, str]] = {
    "llama": {
        f"model.layers.{LAYER}.post_attention_layernorm": f"blocks.{LAYER}.feed_forward_norm"
    },
    "gemma3_text": {
        f"model.layers.{LAYER}.post_attention_layernorm": f"blocks.{LAYER}.post_attention_norm",
        f"model.layers.{LAYER}.pre_feedforward_layernorm": f"blocks.{LAYER}.feed_forward_norm",
        f"model.layers.{LAYER}.post_feedforward_layernorm": f"blocks.{LAYER}.post_feed_forward_norm",
    },
    "qwen3": {
        f"model.layers.{LAYER}.post_attention_layernorm": f"blocks.{LAYER}.feed_forward_norm"
    },
    "qwen3_5_text": {
        f"model.layers.{LAYER}.post_attention_layernorm": f"blocks.{LAYER}.feed_forward_norm"
    },
}


#: Map of Hugging Face keys to OLMo Core keys, that is used to determine how HF state
#: maps to OLMo Core state. Different HF models may use different names for a given OLMo
#: Core state. You may configure this to change how HF state maps to OLMo Core state.
#:
#: This map captures many-to-many mappings from HF to OLMo Core and mappings that require
#: additional manipulation of state (e.g. merging dimensions).
#: For simple one-to-one mappings from HF to OLMo Core, see
#: :data:`HF_TO_OLMO_CORE_MAPPINGS`.
HF_TO_OLMO_CORE_TEMPLATE_MAPPINGS: Dict[str, StateMappingTemplate] = {
    f"model.layers.{LAYER}.mlp.experts.{EXPERT}.gate_proj.weight": StateMappingTemplate(
        f"model.layers.{LAYER}.mlp.experts.{EXPERT}.gate_proj.weight",
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w1",
        source_key_per_placeholder=TemplatePlaceholder.EXPERT,
        source_concat_dim=1,
        dims_permutation=(1, 0),
    ),
    f"model.layers.{LAYER}.mlp.experts.{EXPERT}.down_proj.weight": StateMappingTemplate(
        f"model.layers.{LAYER}.mlp.experts.{EXPERT}.down_proj.weight",
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w2",
        source_key_per_placeholder=TemplatePlaceholder.EXPERT,
        source_concat_dim=1,
        dims_permutation=(1, 0),
    ),
    f"model.layers.{LAYER}.mlp.experts.{EXPERT}.up_proj.weight": StateMappingTemplate(
        f"model.layers.{LAYER}.mlp.experts.{EXPERT}.up_proj.weight",
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w3",
        source_key_per_placeholder=TemplatePlaceholder.EXPERT,
        source_concat_dim=1,
        dims_permutation=(1, 0),
    ),
    f"model.layers.{LAYER}.mlp.gate.weight": StateMappingTemplate(
        f"model.layers.{LAYER}.mlp.gate.weight",
        f"blocks.{LAYER}.feed_forward_moe.router.weight",
        flatten_dims=(0, 1),
    ),
}


#: Map of OLMo Core weight keys to Hugging Face weight keys, that is used to determine how OLMo Core state
#: maps to HF state. You may configure this to change how OLMo Core state maps to HF state.
#:
#: This map only captures one-to-one mappings from OLMo Core to HF. For many-to-many mappings
#: or mappings that require additional manipulation of state, see :data:`OLMO_CORE_TO_HF_TEMPLATE_MAPPINGS`.
OLMO_CORE_TO_HF_WEIGHT_MAPPINGS: Dict[str, str] = {
    "embeddings.weight": "model.embed_tokens.weight",
    "lm_head.norm.weight": "model.norm.weight",
    "lm_head.w_out.weight": "lm_head.weight",
    # Attention.
    f"blocks.{LAYER}.attention.w_q.weight": f"model.layers.{LAYER}.self_attn.q_proj.weight",
    f"blocks.{LAYER}.attention.w_k.weight": f"model.layers.{LAYER}.self_attn.k_proj.weight",
    f"blocks.{LAYER}.attention.w_v.weight": f"model.layers.{LAYER}.self_attn.v_proj.weight",
    f"blocks.{LAYER}.attention.w_out.weight": f"model.layers.{LAYER}.self_attn.o_proj.weight",
    # MLP.
    f"blocks.{LAYER}.feed_forward.w1.weight": f"model.layers.{LAYER}.mlp.gate_proj.weight",
    f"blocks.{LAYER}.feed_forward.w2.weight": f"model.layers.{LAYER}.mlp.down_proj.weight",
    f"blocks.{LAYER}.feed_forward.w3.weight": f"model.layers.{LAYER}.mlp.up_proj.weight",
    # Layer norms.
    f"blocks.{LAYER}.attention_norm.weight": f"model.layers.{LAYER}.post_attention_layernorm.weight",
    f"blocks.{LAYER}.feed_forward_norm.weight": f"model.layers.{LAYER}.post_feedforward_layernorm.weight",
    f"blocks.{LAYER}.attention.q_norm.weight": f"model.layers.{LAYER}.self_attn.q_norm.weight",
    f"blocks.{LAYER}.attention.k_norm.weight": f"model.layers.{LAYER}.self_attn.k_norm.weight",
    # MoEMLP.
    f"blocks.{LAYER}.feed_forward_moe.router.weight": f"model.layers.{LAYER}.mlp.gate.weight",
    f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w1": f"model.layers.{LAYER}.mlp.experts.{EXPERT}.gate_proj.weight",
    f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w2": f"model.layers.{LAYER}.mlp.experts.{EXPERT}.down_proj.weight",
    f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w3": f"model.layers.{LAYER}.mlp.experts.{EXPERT}.up_proj.weight",
}


#: Map of OLMo Core module keys to Hugging Face module keys, that is used to determine how OLMo Core state
#: maps to HF state. You may configure this to change how OLMo Core state maps to HF state.
#:
#: This map only captures one-to-one mappings from OLMo Core to HF. For many-to-many mappings
#: or mappings that require additional manipulation of state, see :data:`OLMO_CORE_TO_HF_TEMPLATE_MAPPINGS`.
OLMO_CORE_TO_HF_MODULE_MAPPINGS: Dict[str, str] = {
    "embeddings": "model.embed_tokens",
    "lm_head.norm": "model.norm",
    "lm_head.w_out": "lm_head",
    # Attention.
    f"blocks.{LAYER}.attention.w_q": f"model.layers.{LAYER}.self_attn.q_proj",
    f"blocks.{LAYER}.attention.w_k": f"model.layers.{LAYER}.self_attn.k_proj",
    f"blocks.{LAYER}.attention.w_v": f"model.layers.{LAYER}.self_attn.v_proj",
    f"blocks.{LAYER}.attention.w_out": f"model.layers.{LAYER}.self_attn.o_proj",
    # MLP.
    f"blocks.{LAYER}.feed_forward": f"model.layers.{LAYER}.mlp",
    f"blocks.{LAYER}.feed_forward.w1": f"model.layers.{LAYER}.mlp.gate_proj",
    f"blocks.{LAYER}.feed_forward.w2": f"model.layers.{LAYER}.mlp.down_proj",
    f"blocks.{LAYER}.feed_forward.w3": f"model.layers.{LAYER}.mlp.up_proj",
    # Layer norms.
    f"blocks.{LAYER}.attention_norm": f"model.layers.{LAYER}.post_attention_layernorm",
    f"blocks.{LAYER}.feed_forward_norm": f"model.layers.{LAYER}.post_feedforward_layernorm",
    f"blocks.{LAYER}.attention.q_norm": f"model.layers.{LAYER}.self_attn.q_norm",
    f"blocks.{LAYER}.attention.k_norm": f"model.layers.{LAYER}.self_attn.k_norm",
    # MoEMLP.
    f"blocks.{LAYER}.feed_forward_moe": f"model.layers.{LAYER}.mlp",
    f"blocks.{LAYER}.feed_forward_moe_norm": f"model.layers.{LAYER}.post_moe_norm",
    f"blocks.{LAYER}.feed_forward_moe.router": f"model.layers.{LAYER}.mlp.gate",
    # Indices are not part of the original OLMo Core state but can be introduced during conversion for aide debugging.
    f"blocks.{LAYER}.feed_forward_moe.router.indices": f"model.layers.{LAYER}.mlp.gate.indices",
    f"blocks.{LAYER}.feed_forward_moe.shared_mlp": f"model.layers.{LAYER}.mlp.shared_mlp",
    f"blocks.{LAYER}.feed_forward_moe.shared_mlp.w1": f"model.layers.{LAYER}.mlp.shared_mlp.gate_proj",
    f"blocks.{LAYER}.feed_forward_moe.shared_mlp.w2": f"model.layers.{LAYER}.mlp.shared_mlp.down_proj",
    f"blocks.{LAYER}.feed_forward_moe.shared_mlp.w3": f"model.layers.{LAYER}.mlp.shared_mlp.up_proj",
}


#: Map of OLMo Core keys to Hugging Face keys, that is used to determine how OLMo Core state
#: maps to HF state. You may configure this to change how OLMo Core state maps to HF state.
#:
#: This map captures many-to-many mappings from OLMo Core to HF and mappings that require
#: additional manipulation of state (e.g. merging dimensions).
#: For simple one-to-one mappings from OLMo Core to HF, see
#: :data:`OLMO_CORE_TO_HF_MAPPINGS`.
OLMO_CORE_TO_HF_TEMPLATE_MAPPINGS: Dict[str, StateMappingTemplate] = {
    f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w1": StateMappingTemplate(
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w1",
        f"model.layers.{LAYER}.mlp.experts.{EXPERT}.gate_proj.weight",
        dest_key_per_placeholder=TemplatePlaceholder.EXPERT,
        dims_permutation=(1, 0),
        dest_chunk_dim=1,
    ),
    f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w2": StateMappingTemplate(
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w2",
        f"model.layers.{LAYER}.mlp.experts.{EXPERT}.down_proj.weight",
        dest_key_per_placeholder=TemplatePlaceholder.EXPERT,
        dims_permutation=(1, 0),
        dest_chunk_dim=1,
    ),
    f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w3": StateMappingTemplate(
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w3",
        f"model.layers.{LAYER}.mlp.experts.{EXPERT}.up_proj.weight",
        dest_key_per_placeholder=TemplatePlaceholder.EXPERT,
        dims_permutation=(1, 0),
        dest_chunk_dim=1,
    ),
    f"blocks.{LAYER}.feed_forward_moe.router.weight": StateMappingTemplate(
        f"blocks.{LAYER}.feed_forward_moe.router.weight",
        f"model.layers.{LAYER}.mlp.gate.weight",
        unflatten_dim=(0, (TemplatePlaceholder.EXPERT, -1)),
    ),
}


#: Map of OLMo Core keys to Hugging Face keys, that is used to determine how OLMo Core state
#: maps to HF state. This map captures overrides of the standard mappings in
#: :data:`OLMO_CORE_TO_HF_TEMPLATE_MAPPINGS`, in case a given OLMo Core key can refer to
#: different HF states depending on the HF model. You may configure this to change how OLMo Core
#: state maps to HF state.
MODEL_TYPE_SPECIFIC_OLMO_CORE_TO_HF_TEMPLATE_MAPPINGS: Dict[
    str, Dict[str, StateMappingTemplate]
] = {
    "flex_olmo": {
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w1": StateMappingTemplate(
            f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w1",
            f"model.layers.{LAYER}.mlp.experts.{EXPERT}.gate_proj.weight",
            dest_key_per_placeholder=TemplatePlaceholder.EXPERT,
            dims_permutation=(1, 0),
            dest_chunk_dim=1,
        ),
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w2": StateMappingTemplate(
            f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w2",
            f"model.layers.{LAYER}.mlp.experts.{EXPERT}.down_proj.weight",
            dest_key_per_placeholder=TemplatePlaceholder.EXPERT,
            dims_permutation=(1, 0),
            dest_chunk_dim=1,
        ),
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w3": StateMappingTemplate(
            f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w3",
            f"model.layers.{LAYER}.mlp.experts.{EXPERT}.up_proj.weight",
            dest_key_per_placeholder=TemplatePlaceholder.EXPERT,
            dims_permutation=(1, 0),
            dest_chunk_dim=1,
        ),
        f"blocks.{LAYER}.feed_forward_moe.router.weight": StateMappingTemplate(
            f"blocks.{LAYER}.feed_forward_moe.router.weight",
            f"model.layers.{LAYER}.mlp.gate.weight",
            unflatten_dim=(0, (TemplatePlaceholder.EXPERT, -1)),
        ),
    },
    "llama": {
        f"blocks.{LAYER}.attention_norm.weight": StateMappingTemplate(
            f"blocks.{LAYER}.attention_norm.weight",
            f"model.layers.{LAYER}.input_layernorm.weight",
            state_type=StateType.weight,
        ),
        f"blocks.{LAYER}.feed_forward_norm.weight": StateMappingTemplate(
            f"blocks.{LAYER}.feed_forward_norm.weight",
            f"model.layers.{LAYER}.post_attention_layernorm.weight",
            state_type=StateType.weight,
        ),
    },
    "qwen3": {
        f"blocks.{LAYER}.attention_norm.weight": StateMappingTemplate(
            f"blocks.{LAYER}.attention_norm.weight",
            f"model.layers.{LAYER}.input_layernorm.weight",
            state_type=StateType.weight,
        ),
        f"blocks.{LAYER}.feed_forward_norm.weight": StateMappingTemplate(
            f"blocks.{LAYER}.feed_forward_norm.weight",
            f"model.layers.{LAYER}.post_attention_layernorm.weight",
            state_type=StateType.weight,
        ),
    },
    "gemma3_text": {
        f"blocks.{LAYER}.attention_norm.weight": StateMappingTemplate(
            f"blocks.{LAYER}.attention_norm.weight",
            f"model.layers.{LAYER}.input_layernorm.weight",
            state_type=StateType.weight,
        ),
        f"blocks.{LAYER}.post_attention_norm.weight": StateMappingTemplate(
            f"blocks.{LAYER}.post_attention_norm.weight",
            f"model.layers.{LAYER}.post_attention_layernorm.weight",
            state_type=StateType.weight,
        ),
        f"blocks.{LAYER}.feed_forward_norm.weight": StateMappingTemplate(
            f"blocks.{LAYER}.feed_forward_norm.weight",
            f"model.layers.{LAYER}.pre_feedforward_layernorm.weight",
            state_type=StateType.weight,
        ),
        f"blocks.{LAYER}.post_feed_forward_norm.weight": StateMappingTemplate(
            f"blocks.{LAYER}.post_feed_forward_norm.weight",
            f"model.layers.{LAYER}.post_feedforward_layernorm.weight",
            state_type=StateType.weight,
        ),
    },
}


def _get_hf_model_to_olmo_core_one_to_one_templates(
    model_type: str | None = None,
) -> List[StateMappingTemplate]:
    mapping_templates = {
        hf_key: StateMappingTemplate(hf_key, olmo_core_key, state_type=StateType.weight)
        for hf_key, olmo_core_key in HF_TO_OLMO_CORE_WEIGHT_MAPPINGS.items()
    }

    for hf_key, olmo_core_key in HF_TO_OLMO_CORE_MODULE_MAPPINGS.items():
        mapping_templates[hf_key] = StateMappingTemplate(
            hf_key, olmo_core_key, state_type=StateType.module
        )

    if model_type in MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_WEIGHT_MAPPINGS:
        model_type_specific_mapping_templates = {
            hf_key: StateMappingTemplate(hf_key, olmo_core_key, state_type=StateType.weight)
            for hf_key, olmo_core_key in MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_WEIGHT_MAPPINGS[
                model_type
            ].items()
        }
        mapping_templates.update(model_type_specific_mapping_templates)

    if model_type in MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_MODULE_MAPPINGS:
        model_type_specific_mapping_templates = {
            hf_key: StateMappingTemplate(hf_key, olmo_core_key, state_type=StateType.module)
            for hf_key, olmo_core_key in MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_MODULE_MAPPINGS[
                model_type
            ].items()
        }
        mapping_templates.update(model_type_specific_mapping_templates)

    return list(mapping_templates.values())


def _get_converter_from_hf(model_type: str | None = None) -> StateConverter:
    mapping_templates = _get_hf_model_to_olmo_core_one_to_one_templates(model_type)
    mapping_templates += list(HF_TO_OLMO_CORE_TEMPLATE_MAPPINGS.values())
    return StateConverter(mapping_templates)


@beta_feature
def get_converter_from_hf(model_type: str | None = None) -> StateConverter:
    return _get_converter_from_hf(model_type=model_type)


def _apply_gemma3_norm_transform(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform Gemma 3 norm weights from HF format to OLMo format.

    HF Gemma 3 uses zero-initialized RMSNorm weights with `hidden_states * (1 + weight)`,
    while OLMo uses ones-initialized weights with `hidden_states * weight`.

    This function adds 1 to all norm weights to convert between the two conventions.
    """
    norm_patterns = [
        "attention_norm.weight",
        "post_attention_norm.weight",
        "feed_forward_norm.weight",
        "post_feed_forward_norm.weight",
        "lm_head.norm.weight",
        "q_norm.weight",
        "k_norm.weight",
    ]

    for key, value in state.items():
        if any(pattern in key for pattern in norm_patterns):
            if isinstance(value, torch.Tensor):
                state[key] = value + 1.0

    return state


def _apply_qwen3_5_norm_transform(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform Qwen3.5 norm weights from HF format to OLMo format.

    HF Qwen3.5 uses zero-initialized RMSNorm weights with ``hidden_states * (1 + weight)``,
    while OLMo uses ones-initialized weights with ``hidden_states * weight``.
    """
    norm_patterns = [
        "attention_norm.weight",
        "feed_forward_norm.weight",
        "lm_head.norm.weight",
        "q_norm.weight",
        "k_norm.weight",
    ]

    for key, value in state.items():
        if any(pattern in key for pattern in norm_patterns):
            if isinstance(value, torch.Tensor):
                state[key] = value + 1.0

    return state


def _get_qwen3_5_text_config(config: PretrainedConfig) -> PretrainedConfig:
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return text_config
    return config


def _get_hf_state_value(hf_state: Dict[str, Any], *keys: str) -> Any:
    """
    Return the first matching value from *keys*, raising if none are present.
    """
    for key in keys:
        if key in hf_state:
            return hf_state[key]
    raise KeyError(keys[0])


def _normalize_qwen3_5_hf_state(hf_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize HF state dict keys to ``model.layers.*`` / shared text-model keys.

    Handles multimodal checkpoints that nest the text decoder under prefixes like
    ``model.language_model.``.

    Also aliases legacy GDN key names (``o_proj`` / ``o_norm``) to the names used by
    recent Hugging Face checkpoints (``out_proj`` / ``norm``).
    """
    normalized: Dict[str, Any] = {}
    text_suffixes = (
        "embed_tokens.weight",
        "norm.weight",
        "layers.",
    )

    for key, value in hf_state.items():
        if any(skip in key for skip in ("visual.", "vision_tower", "multi_modal")):
            continue

        model_key: str | None = None
        if ".layers." in key:
            idx = key.index(".layers.")
            model_key = "model" + key[idx:]
        elif key.endswith("embed_tokens.weight"):
            model_key = "model.embed_tokens.weight"
        elif key.endswith("norm.weight") and "layers." not in key:
            model_key = "model.norm.weight"
        elif key == "lm_head.weight" or key.endswith(".lm_head.weight"):
            model_key = "lm_head.weight"

        if model_key is None:
            continue
        if not any(s in model_key for s in text_suffixes) and model_key != "lm_head.weight":
            continue
        normalized[model_key] = value
        if "linear_attn.out_proj." in model_key:
            normalized[model_key.replace("out_proj", "o_proj")] = value
        if "linear_attn.norm.weight" in model_key:
            normalized[model_key.replace("linear_attn.norm", "linear_attn.o_norm")] = value

    return normalized


def _convert_qwen3_5_gdn_layer(
    hf_state: Dict[str, Any],
    layer_idx: int,
    key_dim: int,
    value_dim: int,
) -> Dict[str, Any]:
    prefix = f"model.layers.{layer_idx}."
    olmo_prefix = f"blocks.{layer_idx}."
    state: Dict[str, Any] = {}

    in_proj_qkv = hf_state[f"{prefix}linear_attn.in_proj_qkv.weight"]
    state[f"{olmo_prefix}attention.w_q.weight"] = in_proj_qkv[:key_dim]
    state[f"{olmo_prefix}attention.w_k.weight"] = in_proj_qkv[key_dim : 2 * key_dim]
    state[f"{olmo_prefix}attention.w_v.weight"] = in_proj_qkv[2 * key_dim : 2 * key_dim + value_dim]

    state[f"{olmo_prefix}attention.w_g.weight"] = hf_state[f"{prefix}linear_attn.in_proj_z.weight"]
    state[f"{olmo_prefix}attention.w_a.weight"] = hf_state[f"{prefix}linear_attn.in_proj_a.weight"]
    state[f"{olmo_prefix}attention.w_b.weight"] = hf_state[f"{prefix}linear_attn.in_proj_b.weight"]
    state[f"{olmo_prefix}attention.w_out.weight"] = _get_hf_state_value(
        hf_state,
        f"{prefix}linear_attn.o_proj.weight",
        f"{prefix}linear_attn.out_proj.weight",
    )

    conv_weight = hf_state[f"{prefix}linear_attn.conv1d.weight"]
    state[f"{olmo_prefix}attention.q_conv1d.weight"] = conv_weight[:key_dim]
    state[f"{olmo_prefix}attention.k_conv1d.weight"] = conv_weight[key_dim : 2 * key_dim]
    state[f"{olmo_prefix}attention.v_conv1d.weight"] = conv_weight[
        2 * key_dim : 2 * key_dim + value_dim
    ]

    state[f"{olmo_prefix}attention.o_norm.weight"] = _get_hf_state_value(
        hf_state,
        f"{prefix}linear_attn.o_norm.weight",
        f"{prefix}linear_attn.norm.weight",
    )
    state[f"{olmo_prefix}attention.A_log"] = hf_state[f"{prefix}linear_attn.A_log"]
    state[f"{olmo_prefix}attention.dt_bias"] = hf_state[f"{prefix}linear_attn.dt_bias"]

    state[f"{olmo_prefix}attention_norm.weight"] = hf_state[f"{prefix}input_layernorm.weight"]
    state[f"{olmo_prefix}feed_forward_norm.weight"] = hf_state[
        f"{prefix}post_attention_layernorm.weight"
    ]
    state[f"{olmo_prefix}feed_forward.w1.weight"] = hf_state[f"{prefix}mlp.gate_proj.weight"]
    state[f"{olmo_prefix}feed_forward.w2.weight"] = hf_state[f"{prefix}mlp.down_proj.weight"]
    state[f"{olmo_prefix}feed_forward.w3.weight"] = hf_state[f"{prefix}mlp.up_proj.weight"]

    return state


def _split_qwen3_5_q_proj(
    q_proj: torch.Tensor, n_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split fused HF ``q_proj`` weights into separate OLMo ``w_q`` and ``w_g`` weights.

    Hugging Face interleaves query and gate projections per head as
    ``[q_0, g_0, q_1, g_1, ...]``, while OLMo stores them as ``[q_0, q_1, ..., g_0, g_1, ...]``.
    """
    hidden_size = q_proj.shape[1]
    q_gate = q_proj.view(n_heads, 2, head_dim, hidden_size)
    w_q = q_gate[:, 0].reshape(n_heads * head_dim, hidden_size)
    w_g = q_gate[:, 1].reshape(n_heads * head_dim, hidden_size)
    return w_q, w_g


def _convert_qwen3_5_attn_layer(
    hf_state: Dict[str, Any],
    layer_idx: int,
    n_heads: int,
    head_dim: int,
) -> Dict[str, Any]:
    prefix = f"model.layers.{layer_idx}."
    olmo_prefix = f"blocks.{layer_idx}."
    state: Dict[str, Any] = {}

    q_proj = hf_state[f"{prefix}self_attn.q_proj.weight"]
    w_q, w_g = _split_qwen3_5_q_proj(q_proj, n_heads, head_dim)
    state[f"{olmo_prefix}attention.w_q.weight"] = w_q
    state[f"{olmo_prefix}attention.w_g.weight"] = w_g

    state[f"{olmo_prefix}attention.w_k.weight"] = hf_state[f"{prefix}self_attn.k_proj.weight"]
    state[f"{olmo_prefix}attention.w_v.weight"] = hf_state[f"{prefix}self_attn.v_proj.weight"]
    state[f"{olmo_prefix}attention.w_out.weight"] = hf_state[f"{prefix}self_attn.o_proj.weight"]
    state[f"{olmo_prefix}attention.q_norm.weight"] = hf_state[f"{prefix}self_attn.q_norm.weight"]
    state[f"{olmo_prefix}attention.k_norm.weight"] = hf_state[f"{prefix}self_attn.k_norm.weight"]

    state[f"{olmo_prefix}attention_norm.weight"] = hf_state[f"{prefix}input_layernorm.weight"]
    state[f"{olmo_prefix}feed_forward_norm.weight"] = hf_state[
        f"{prefix}post_attention_layernorm.weight"
    ]
    state[f"{olmo_prefix}feed_forward.w1.weight"] = hf_state[f"{prefix}mlp.gate_proj.weight"]
    state[f"{olmo_prefix}feed_forward.w2.weight"] = hf_state[f"{prefix}mlp.down_proj.weight"]
    state[f"{olmo_prefix}feed_forward.w3.weight"] = hf_state[f"{prefix}mlp.up_proj.weight"]

    return state


@beta_feature
def convert_qwen3_5_state_from_hf(
    config: PretrainedConfig,
    hf_state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert a Hugging Face Qwen3.5 text model state dict to Olmo-core format.

    Handles the hybrid Gated DeltaNet + full-attention architecture, including fused
    projection and convolution weight layouts that differ from the standard converter.

    :param config: The Hugging Face config (``Qwen3_5Config`` or ``Qwen3_5TextConfig``).
    :param hf_state: A model state dict in HF format.
    """
    text_config = _get_qwen3_5_text_config(config)
    hf_state = _normalize_qwen3_5_hf_state(hf_state)

    layer_types: List[str] = list(text_config.layer_types)
    n_layers: int = text_config.num_hidden_layers

    key_dim = text_config.linear_num_key_heads * text_config.linear_key_head_dim
    value_dim = text_config.linear_num_value_heads * text_config.linear_value_head_dim
    head_dim: int = text_config.head_dim
    n_heads: int = text_config.num_attention_heads

    olmo_state: Dict[str, Any] = {}
    if "model.embed_tokens.weight" in hf_state:
        olmo_state["embeddings.weight"] = hf_state["model.embed_tokens.weight"]
    if "model.norm.weight" in hf_state:
        olmo_state["lm_head.norm.weight"] = hf_state["model.norm.weight"]
    if "lm_head.weight" in hf_state:
        olmo_state["lm_head.w_out.weight"] = hf_state["lm_head.weight"]
    elif (
        getattr(text_config, "tie_word_embeddings", False)
        and "model.embed_tokens.weight" in hf_state
    ):
        olmo_state["lm_head.w_out.weight"] = hf_state["model.embed_tokens.weight"]

    for layer_idx in range(n_layers):
        if layer_types[layer_idx] == "linear_attention":
            olmo_state.update(_convert_qwen3_5_gdn_layer(hf_state, layer_idx, key_dim, value_dim))
        elif layer_types[layer_idx] == "full_attention":
            olmo_state.update(_convert_qwen3_5_attn_layer(hf_state, layer_idx, n_heads, head_dim))
        else:
            raise ValueError(f"Unknown layer type {layer_types[layer_idx]!r} at layer {layer_idx}")

    return _apply_qwen3_5_norm_transform(olmo_state)


# ---------------------------------------------------------------------------
# olmo3moe (MoE-v2) conversion.
#
# The MoE-v2 routed experts store a *fused* up+gate projection weight
# ``blocks.{i}.routed_experts.w_up_gate`` of shape ``(E, 2H, D)`` (the first ``H``
# rows are the up-projection, the next ``H`` are the gate-projection), plus a
# per-expert down-projection ``blocks.{i}.routed_experts.w_down`` of shape
# ``(E, H, D)``. HF ``olmo3moe`` instead expects a separate ``nn.Linear`` per
# expert: ``gate_proj``/``up_proj`` of shape ``(H, D)`` and ``down_proj`` of
# shape ``(D, H)``.
#
# The :class:`StateMappingTemplate` pipeline (cat -> unflatten -> permute ->
# flatten -> chunk) cannot express this fused up/gate split: isolating the up
# (or gate) half requires a slice, and fanning that half out per-expert requires
# a second, independent split along a different dimension. There is no slice op
# and only one ``dest_chunk_dim``/``EXPERT`` expansion is available per template.
# So, following the standalone-converter precedent used for Qwen3.5
# (:func:`convert_qwen3_5_state_from_hf`), olmo3moe uses dedicated functions.
#
# These mirror ``convert_checkpoint.py`` from the standalone MoE-v2 HF converter.
# Norm handling only covers the default reordered-norm scheme (``use_peri_ln`` is
# False): ``attention_norm`` -> ``post_attention_layernorm`` and
# ``feed_forward_norm`` -> ``post_feedforward_layernorm``, uniformly across dense
# and MoE layers. The peri-LN scheme is not supported here because it maps the
# same Olmo-core norm key (``attention_norm``) to different HF keys depending on
# whether the layer is dense or MoE, which cannot be expressed uniformly.
# ---------------------------------------------------------------------------


def _olmo3moe_dense_layer_indices(config: PretrainedConfig) -> set:
    indices = getattr(config, "dense_layers_indices", None)
    if indices is None:
        return set()
    return set(indices)


def _require_exact_state_keys(state: Mapping[str, Any], used: set[str], *, state_name: str) -> None:
    unexpected = sorted(set(state) - used)
    if unexpected:
        raise KeyError(f"Unexpected {state_name} keys: {unexpected}")


def _take(state: Mapping[str, Any], used: set[str], key: str) -> Any:
    used.add(key)
    return state[key]


def _olmo3moe_is_kda_layer(config: PretrainedConfig, layer_idx: int) -> bool:
    layer_types = getattr(config, "layer_types", None) or []
    return bool(layer_types and layer_types[layer_idx] == "linear_attention")


def _convert_kda_layer_from_hf(
    hf_state: Dict[str, Any],
    used: set[str],
    prefix: str,
    olmo_prefix: str,
) -> Dict[str, Any]:
    key_map = {
        "q_proj.weight": "w_q.weight",
        "k_proj.weight": "w_k.weight",
        "v_proj.weight": "w_v.weight",
        "f_proj_1.weight": "f_proj_1.weight",
        "f_proj_2.weight": "f_proj_2.weight",
        "beta_proj.weight": "w_b.weight",
        "A_log": "A_log",
        "dt_bias": "dt_bias",
        "q_conv1d.weight": "q_conv1d.weight",
        "k_conv1d.weight": "k_conv1d.weight",
        "v_conv1d.weight": "v_conv1d.weight",
        "g_proj_1.weight": "g_proj_1.weight",
        "g_proj_2.weight": "g_proj_2.weight",
        "g_proj_2.bias": "g_proj_2.bias",
        "o_norm.weight": "o_norm.weight",
        "o_proj.weight": "w_out.weight",
    }
    return {
        f"{olmo_prefix}attention.{olmo_suffix}": _take(
            hf_state, used, f"{prefix}self_attn.{hf_suffix}"
        )
        for hf_suffix, olmo_suffix in key_map.items()
    }


def _convert_kda_layer_to_hf(
    olmo_state: Mapping[str, Any],
    used: set[str],
    prefix: str,
    olmo_prefix: str,
) -> Dict[str, Any]:
    key_map = {
        "w_q.weight": "q_proj.weight",
        "w_k.weight": "k_proj.weight",
        "w_v.weight": "v_proj.weight",
        "f_proj_1.weight": "f_proj_1.weight",
        "f_proj_2.weight": "f_proj_2.weight",
        "w_b.weight": "beta_proj.weight",
        "A_log": "A_log",
        "dt_bias": "dt_bias",
        "q_conv1d.weight": "q_conv1d.weight",
        "k_conv1d.weight": "k_conv1d.weight",
        "v_conv1d.weight": "v_conv1d.weight",
        "g_proj_1.weight": "g_proj_1.weight",
        "g_proj_2.weight": "g_proj_2.weight",
        "g_proj_2.bias": "g_proj_2.bias",
        "o_norm.weight": "o_norm.weight",
        "w_out.weight": "o_proj.weight",
    }
    return {
        f"{prefix}self_attn.{hf_suffix}": _take(
            olmo_state, used, f"{olmo_prefix}attention.{olmo_suffix}"
        )
        for olmo_suffix, hf_suffix in key_map.items()
    }


@beta_feature
def convert_olmo3moe_state_from_hf(
    config: PretrainedConfig,
    hf_state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert a Hugging Face ``olmo3moe`` state dict to Olmo-core MoE-v2 format.

    Handles the fused ``w_up_gate`` routed-expert layout and the mixed dense / MoE
    layer structure that the template-based converter cannot express.

    :param config: The Hugging Face ``Olmo3MoeConfig``.
    :param hf_state: A model state dict in HF format.
    """
    n_layers: int = config.num_hidden_layers
    n_experts: int = config.n_routed_experts
    dense_indices = _olmo3moe_dense_layer_indices(config)
    has_shared = getattr(config, "shared_expert_intermediate_size", None) is not None
    latent_dim = getattr(config, "latent_moe_dim", None)
    dense_layers_use_shared_expert = getattr(config, "dense_layers_use_shared_expert", False)
    peri_ln = getattr(config, "use_peri_ln", False)
    used: set[str] = set()

    olmo_state: Dict[str, Any] = {}

    olmo_state["embeddings.weight"] = _take(hf_state, used, "model.embed_tokens.weight")
    olmo_state["lm_head.norm.weight"] = _take(hf_state, used, "model.norm.weight")
    # With tied embeddings transformers omits ``lm_head.weight``; fall back to the input embeddings.
    if "lm_head.weight" in hf_state:
        olmo_state["lm_head.w_out.weight"] = _take(hf_state, used, "lm_head.weight")
    else:
        olmo_state["lm_head.w_out.weight"] = hf_state["model.embed_tokens.weight"]
    if getattr(config, "embed_norm", False):
        olmo_state["embedding_norm.weight"] = _take(hf_state, used, "model.embed_norm.weight")

    for layer_idx in range(n_layers):
        prefix = f"model.layers.{layer_idx}."
        olmo_prefix = f"blocks.{layer_idx}."

        if _olmo3moe_is_kda_layer(config, layer_idx):
            olmo_state.update(_convert_kda_layer_from_hf(hf_state, used, prefix, olmo_prefix))
        else:
            attention_map = {
                "q_proj.weight": "w_q.weight",
                "k_proj.weight": "w_k.weight",
                "v_proj.weight": "w_v.weight",
                "o_proj.weight": "w_out.weight",
                "q_norm.weight": "q_norm.weight",
                "k_norm.weight": "k_norm.weight",
            }
            if getattr(config, "attention_gate_type", None) is not None:
                attention_map["g_proj.weight"] = "w_g.weight"
            if getattr(config, "scalable_softmax", False):
                attention_map["ssmax_scale"] = "ssmax_scale"
            for hf_suffix, olmo_suffix in attention_map.items():
                olmo_state[f"{olmo_prefix}attention.{olmo_suffix}"] = _take(
                    hf_state, used, f"{prefix}self_attn.{hf_suffix}"
                )

        olmo_state[f"{olmo_prefix}attention_norm.weight"] = _take(
            hf_state, used, f"{prefix}post_attention_layernorm.weight"
        )
        olmo_state[f"{olmo_prefix}feed_forward_norm.weight"] = _take(
            hf_state, used, f"{prefix}post_feedforward_layernorm.weight"
        )
        if peri_ln:
            olmo_state[f"{olmo_prefix}attention_input_norm.weight"] = _take(
                hf_state, used, f"{prefix}pre_attention_layernorm.weight"
            )
            olmo_state[f"{olmo_prefix}feed_forward_input_norm.weight"] = _take(
                hf_state, used, f"{prefix}pre_feedforward_layernorm.weight"
            )

        if layer_idx in dense_indices:
            gate = _take(hf_state, used, f"{prefix}mlp.gate_proj.weight")
            down = _take(hf_state, used, f"{prefix}mlp.down_proj.weight")
            up = _take(hf_state, used, f"{prefix}mlp.up_proj.weight")
            if not dense_layers_use_shared_expert:
                olmo_state[f"{olmo_prefix}feed_forward.w1.weight"] = gate
                olmo_state[f"{olmo_prefix}feed_forward.w2.weight"] = down
                olmo_state[f"{olmo_prefix}feed_forward.w3.weight"] = up
            else:
                olmo_state[f"{olmo_prefix}shared_experts.w_up_gate"] = torch.cat(
                    [up.T, gate.T], dim=1
                ).contiguous()
                olmo_state[f"{olmo_prefix}shared_experts.w_down"] = down.T.unsqueeze(0).contiguous()
            continue

        olmo_state[f"{olmo_prefix}routed_experts_router.weight"] = _take(
            hf_state, used, f"{prefix}mlp.router.gate.weight"
        ).reshape(-1)

        up_list = [
            _take(hf_state, used, f"{prefix}mlp.experts.{e}.up_proj.weight")
            for e in range(n_experts)
        ]
        gate_list = [
            _take(hf_state, used, f"{prefix}mlp.experts.{e}.gate_proj.weight")
            for e in range(n_experts)
        ]
        down_list = [
            _take(hf_state, used, f"{prefix}mlp.experts.{e}.down_proj.weight")
            for e in range(n_experts)
        ]

        w_up = torch.stack(up_list, dim=0)
        w_gate = torch.stack(gate_list, dim=0)
        olmo_state[f"{olmo_prefix}routed_experts.w_up_gate"] = torch.cat(
            [w_up, w_gate], dim=1
        ).contiguous()
        olmo_state[f"{olmo_prefix}routed_experts.w_down"] = torch.stack(
            [d.T for d in down_list], dim=0
        ).contiguous()

        if latent_dim is not None:
            olmo_state[f"{olmo_prefix}latent_down_proj.weight"] = _take(
                hf_state, used, f"{prefix}mlp.latent_down_proj.weight"
            )
            olmo_state[f"{olmo_prefix}latent_up_proj.weight"] = _take(
                hf_state, used, f"{prefix}mlp.latent_up_proj.weight"
            )
            if getattr(config, "latent_moe_bias", False):
                olmo_state[f"{olmo_prefix}latent_down_proj.bias"] = _take(
                    hf_state, used, f"{prefix}mlp.latent_down_proj.bias"
                )
                olmo_state[f"{olmo_prefix}latent_up_proj.bias"] = _take(
                    hf_state, used, f"{prefix}mlp.latent_up_proj.bias"
                )
            if getattr(config, "latent_moe_up_proj_input_norm", False):
                olmo_state[f"{olmo_prefix}latent_up_proj_input_norm.weight"] = _take(
                    hf_state,
                    used,
                    f"{prefix}mlp.latent_up_proj_input_norm.weight",
                )

        if has_shared:
            shared_up = _take(hf_state, used, f"{prefix}mlp.shared_expert.up_proj.weight")
            shared_gate = _take(hf_state, used, f"{prefix}mlp.shared_expert.gate_proj.weight")
            olmo_state[f"{olmo_prefix}shared_experts.w_up_gate"] = torch.cat(
                [shared_up.T, shared_gate.T], dim=1
            ).contiguous()
            shared_down = _take(hf_state, used, f"{prefix}mlp.shared_expert.down_proj.weight")
            olmo_state[f"{olmo_prefix}shared_experts.w_down"] = shared_down.T.unsqueeze(
                0
            ).contiguous()

    _require_exact_state_keys(hf_state, used, state_name="HF")
    return olmo_state


@beta_feature
def convert_olmo3moe_state_to_hf(
    config: PretrainedConfig, olmo_core_state: Dict[str, Any]
) -> Dict[str, Any]:
    """Materialize the exact HF state; use the iterator for streaming publication."""
    return dict(iter_olmo3moe_state_to_hf(config, olmo_core_state))


def iter_olmo3moe_state_to_hf(
    config: PretrainedConfig,
    olmo_core_state: Mapping[str, Any],
    *,
    fused_experts: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    """
    Yield an unsharded MoE-v2 state in HF format without retaining all layers.

    A lazy input mapping may gather parameters on demand. Expert views retain
    only the current layer slabs; the consumer controls output bucket lifetime.

    Inverse of :func:`convert_olmo3moe_state_from_hf`. Splits the fused
    ``w_up_gate`` routed-expert weight into per-expert HF ``up_proj``/``gate_proj``
    and reshapes ``w_down`` into per-expert HF ``down_proj``.

    With ``fused_experts`` the routed experts of each layer are yielded as two
    stacked tensors instead of ``3 * n_routed_experts`` slices:
    ``mlp.experts.gate_up_proj.weight`` of shape ``[E, 2 * moe_hidden, d]`` with
    gate rows before up rows, and ``mlp.experts.down_proj.weight`` of shape
    ``[E, d, moe_hidden]``. That is the serving engine's fused expert layout, for
    online weight publication; it is not the HF checkpoint format.

    :param config: The Hugging Face ``Olmo3MoeConfig``.
    :param olmo_core_state: An unsharded Olmo-core model state dict.
    :param fused_experts: Yield stacked per-layer expert tensors.
    """
    n_layers: int = config.num_hidden_layers
    n_experts: int = config.n_routed_experts
    d_model: int = config.hidden_size
    routed_d_model: int = getattr(config, "latent_moe_dim", None) or d_model
    moe_hidden: int = config.moe_intermediate_size
    dense_indices = _olmo3moe_dense_layer_indices(config)
    has_shared = getattr(config, "shared_expert_intermediate_size", None) is not None
    shared_hidden = getattr(config, "shared_expert_intermediate_size", None)
    latent_dim = getattr(config, "latent_moe_dim", None)
    dense_layers_use_shared_expert = getattr(config, "dense_layers_use_shared_expert", False)
    peri_ln = getattr(config, "use_peri_ln", False)
    used: set[str] = set()

    yield "model.embed_tokens.weight", _take(olmo_core_state, used, "embeddings.weight")
    yield "model.norm.weight", _take(olmo_core_state, used, "lm_head.norm.weight")
    yield "lm_head.weight", _take(olmo_core_state, used, "lm_head.w_out.weight")
    if getattr(config, "embed_norm", False):
        yield "model.embed_norm.weight", _take(olmo_core_state, used, "embedding_norm.weight")

    for layer_idx in range(n_layers):
        prefix = f"model.layers.{layer_idx}."
        olmo_prefix = f"blocks.{layer_idx}."

        if _olmo3moe_is_kda_layer(config, layer_idx):
            yield from _convert_kda_layer_to_hf(olmo_core_state, used, prefix, olmo_prefix).items()
        else:
            attention_map = {
                "w_q.weight": "q_proj.weight",
                "w_k.weight": "k_proj.weight",
                "w_v.weight": "v_proj.weight",
                "w_out.weight": "o_proj.weight",
                "q_norm.weight": "q_norm.weight",
                "k_norm.weight": "k_norm.weight",
            }
            if getattr(config, "attention_gate_type", None) is not None:
                attention_map["w_g.weight"] = "g_proj.weight"
            if getattr(config, "scalable_softmax", False):
                attention_map["ssmax_scale"] = "ssmax_scale"
            for olmo_suffix, hf_suffix in attention_map.items():
                yield (
                    f"{prefix}self_attn.{hf_suffix}",
                    _take(olmo_core_state, used, f"{olmo_prefix}attention.{olmo_suffix}"),
                )

        yield (
            f"{prefix}post_attention_layernorm.weight",
            _take(olmo_core_state, used, f"{olmo_prefix}attention_norm.weight"),
        )
        yield (
            f"{prefix}post_feedforward_layernorm.weight",
            _take(olmo_core_state, used, f"{olmo_prefix}feed_forward_norm.weight"),
        )
        if peri_ln:
            yield (
                f"{prefix}pre_attention_layernorm.weight",
                _take(olmo_core_state, used, f"{olmo_prefix}attention_input_norm.weight"),
            )
            yield (
                f"{prefix}pre_feedforward_layernorm.weight",
                _take(olmo_core_state, used, f"{olmo_prefix}feed_forward_input_norm.weight"),
            )

        if layer_idx in dense_indices:
            if not dense_layers_use_shared_expert:
                yield (
                    f"{prefix}mlp.gate_proj.weight",
                    _take(olmo_core_state, used, f"{olmo_prefix}feed_forward.w1.weight"),
                )
                yield (
                    f"{prefix}mlp.down_proj.weight",
                    _take(olmo_core_state, used, f"{olmo_prefix}feed_forward.w2.weight"),
                )
                yield (
                    f"{prefix}mlp.up_proj.weight",
                    _take(olmo_core_state, used, f"{olmo_prefix}feed_forward.w3.weight"),
                )
            else:
                dense_hidden = config.dense_mlp_intermediate_size
                w_up_gate = _take(
                    olmo_core_state,
                    used,
                    f"{olmo_prefix}shared_experts.w_up_gate",
                ).reshape(d_model, 2 * dense_hidden)
                yield f"{prefix}mlp.up_proj.weight", w_up_gate[:, :dense_hidden].T.contiguous()
                yield f"{prefix}mlp.gate_proj.weight", w_up_gate[:, dense_hidden:].T.contiguous()
                yield (
                    f"{prefix}mlp.down_proj.weight",
                    _take(olmo_core_state, used, f"{olmo_prefix}shared_experts.w_down")
                    .reshape(dense_hidden, d_model)
                    .T.contiguous(),
                )
            continue

        yield (
            f"{prefix}mlp.router.gate.weight",
            _take(olmo_core_state, used, f"{olmo_prefix}routed_experts_router.weight").reshape(
                n_experts, d_model
            ),
        )

        w_up_gate = _take(olmo_core_state, used, f"{olmo_prefix}routed_experts.w_up_gate").reshape(
            n_experts, 2 * moe_hidden, routed_d_model
        )
        w_up = w_up_gate[:, :moe_hidden, :]
        w_gate = w_up_gate[:, moe_hidden:, :]
        w_down = _take(olmo_core_state, used, f"{olmo_prefix}routed_experts.w_down").reshape(
            n_experts, moe_hidden, routed_d_model
        )
        if fused_experts:
            yield (
                f"{prefix}mlp.experts.gate_up_proj.weight",
                torch.cat((w_gate, w_up), dim=1).contiguous(),
            )
            yield f"{prefix}mlp.experts.down_proj.weight", w_down.transpose(1, 2).contiguous()
        else:
            for e in range(n_experts):
                yield f"{prefix}mlp.experts.{e}.up_proj.weight", w_up[e].contiguous()
                yield f"{prefix}mlp.experts.{e}.gate_proj.weight", w_gate[e].contiguous()
                yield f"{prefix}mlp.experts.{e}.down_proj.weight", w_down[e].T.contiguous()

        del w_up_gate, w_up, w_gate, w_down

        if latent_dim is not None:
            yield (
                f"{prefix}mlp.latent_down_proj.weight",
                _take(olmo_core_state, used, f"{olmo_prefix}latent_down_proj.weight"),
            )
            yield (
                f"{prefix}mlp.latent_up_proj.weight",
                _take(olmo_core_state, used, f"{olmo_prefix}latent_up_proj.weight"),
            )
            if getattr(config, "latent_moe_bias", False):
                yield (
                    f"{prefix}mlp.latent_down_proj.bias",
                    _take(olmo_core_state, used, f"{olmo_prefix}latent_down_proj.bias"),
                )
                yield (
                    f"{prefix}mlp.latent_up_proj.bias",
                    _take(olmo_core_state, used, f"{olmo_prefix}latent_up_proj.bias"),
                )
            if getattr(config, "latent_moe_up_proj_input_norm", False):
                yield (
                    f"{prefix}mlp.latent_up_proj_input_norm.weight",
                    _take(olmo_core_state, used, f"{olmo_prefix}latent_up_proj_input_norm.weight"),
                )

        if has_shared:
            assert shared_hidden is not None
            shared_up_gate = _take(
                olmo_core_state, used, f"{olmo_prefix}shared_experts.w_up_gate"
            ).reshape(d_model, 2 * shared_hidden)
            shared_up = shared_up_gate[:, :shared_hidden]
            shared_gate = shared_up_gate[:, shared_hidden:]
            yield f"{prefix}mlp.shared_expert.up_proj.weight", shared_up.T.contiguous()
            yield f"{prefix}mlp.shared_expert.gate_proj.weight", shared_gate.T.contiguous()
            shared_down = _take(
                olmo_core_state, used, f"{olmo_prefix}shared_experts.w_down"
            ).reshape(shared_hidden, d_model)
            yield f"{prefix}mlp.shared_expert.down_proj.weight", shared_down.T.contiguous()

    _require_exact_state_keys(olmo_core_state, used, state_name="Olmo-core")


def _convert_state(
    config: PretrainedConfig,
    state: Dict[str, Any],
    converter: StateConverter,
) -> Dict[str, Any]:
    if not hasattr(config, "num_hidden_layers"):
        raise ValueError(f"Number of hidden layers missing in HF config: {config}")
    n_layers: int = config.num_hidden_layers
    n_experts: int | None = getattr(config, "num_experts", None)

    placeholder_bounds = {
        TemplatePlaceholder.LAYER: n_layers,
    }
    if n_experts:
        placeholder_bounds[TemplatePlaceholder.EXPERT] = n_experts

    return converter.convert(state, placeholder_bounds)


@beta_feature
def convert_state_from_hf(
    config: PretrainedConfig,
    hf_state: Dict[str, Any],
    *,
    model_type: str | None = None,
) -> Dict[str, Any]:
    """
    Converts a model state dict in Hugging Face transformers format into an unsharded state dict of
    OLMo Core format.

    :param config: The Hugging Face config for the model
    :param hf_state: A model state dict in HF format.
    :param model_type: The model type of the HF model.
    """

    converter = _get_converter_from_hf(model_type=model_type)

    if model_type in {"qwen3_5", "qwen3_5_text"}:
        return convert_qwen3_5_state_from_hf(config, hf_state)

    if model_type == "olmo3moe":
        return convert_olmo3moe_state_from_hf(config, hf_state)

    converted_state = _convert_state(config, hf_state, converter)

    if model_type == "gemma3_text":
        converted_state = _apply_gemma3_norm_transform(converted_state)

    if config.tie_word_embeddings:
        converted_state["lm_head.w_out.weight"] = converted_state["embeddings.weight"]

    return converted_state


def _get_converter_to_hf(model_type: str | None = None) -> StateConverter:
    mapping_templates = {
        olmo_core_key: StateMappingTemplate(olmo_core_key, hf_key, state_type=StateType.module)
        for olmo_core_key, hf_key in OLMO_CORE_TO_HF_MODULE_MAPPINGS.items()
    }
    mapping_templates.update(
        {
            olmo_core_key: StateMappingTemplate(olmo_core_key, hf_key, state_type=StateType.weight)
            for olmo_core_key, hf_key in OLMO_CORE_TO_HF_WEIGHT_MAPPINGS.items()
        }
    )
    mapping_templates.update(OLMO_CORE_TO_HF_TEMPLATE_MAPPINGS)

    if model_type:
        mapping_templates.update(
            MODEL_TYPE_SPECIFIC_OLMO_CORE_TO_HF_TEMPLATE_MAPPINGS.get(model_type, {})
        )

    return StateConverter(list(mapping_templates.values()))


@beta_feature
def get_converter_to_hf(model_type: str | None = None) -> StateConverter:
    return _get_converter_to_hf(model_type)


def _apply_gemma3_norm_inverse_transform(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inverse of :func:`_apply_gemma3_norm_transform`: subtracts 1 from norm weights
    so that an Olmo-core checkpoint can be exported back into HF Gemma 3 format.
    """
    norm_patterns = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "model.norm.weight",
        "q_norm.weight",
        "k_norm.weight",
    ]

    for key, value in state.items():
        if any(pattern in key for pattern in norm_patterns):
            if isinstance(value, torch.Tensor):
                state[key] = value - 1.0

    return state


@beta_feature
def convert_state_to_hf(
    config: PretrainedConfig, olmo_core_state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Converts an *unsharded* model state dict of OLMo Core format into Hugging Face transformers format.

    :param config: The Hugging Face config for the model
    :param olmo_core_state: An unsharded OLMo Core model state dict. None of the states can be
        :class:`DTensor` or :class:`ShardedTensor`
    """

    if config.model_type == "olmo3moe":
        return convert_olmo3moe_state_to_hf(config, olmo_core_state)

    converter = _get_converter_to_hf(config.model_type)

    converted_state = _convert_state(config, olmo_core_state, converter)

    if config.model_type == "gemma3_text":
        converted_state = _apply_gemma3_norm_inverse_transform(converted_state)

    return converted_state


# ---------------------------------------------------------------------------
# Hybrid (GDN + attention) per-layer key maps.
#
# These can't use the LAYER template because GDN and attention layers sharing
# the same Olmo-core prefix (``blocks.{i}.attention.*``) need different HF
# prefixes (``linear_attn.*`` vs ``self_attn.*``).
# ---------------------------------------------------------------------------

#: GDN layers: Olmo-core ``blocks.{i}.attention.*`` -> HF ``model.layers.{i}.linear_attn.*``.
#: These layers use pre-norm in HF (input_layernorm before the sequence mixer).
HYBRID_GDN_LAYER_KEY_MAP: Dict[str, str] = {
    "attention.w_q.weight": "linear_attn.q_proj.weight",
    "attention.w_k.weight": "linear_attn.k_proj.weight",
    "attention.w_v.weight": "linear_attn.v_proj.weight",
    "attention.w_a.weight": "linear_attn.a_proj.weight",
    "attention.w_b.weight": "linear_attn.b_proj.weight",
    "attention.w_g.weight": "linear_attn.g_proj.weight",
    "attention.w_out.weight": "linear_attn.out_proj.weight",
    "attention.q_conv1d.weight": "linear_attn.q_conv1d.weight",
    "attention.k_conv1d.weight": "linear_attn.k_conv1d.weight",
    "attention.v_conv1d.weight": "linear_attn.v_conv1d.weight",
    "attention.o_norm.weight": "linear_attn.norm.weight",
    "attention.A_log": "linear_attn.A_log",
    "attention.dt_bias": "linear_attn.dt_bias",
    "attention_norm.weight": "input_layernorm.weight",
    "feed_forward_norm.weight": "post_attention_layernorm.weight",
    "feed_forward.w1.weight": "mlp.gate_proj.weight",
    "feed_forward.w2.weight": "mlp.down_proj.weight",
    "feed_forward.w3.weight": "mlp.up_proj.weight",
}

#: Attention layers: Olmo-core ``blocks.{i}.attention.*`` -> HF ``model.layers.{i}.self_attn.*``.
#: These layers use post-norm in HF (layernorm after the sequence mixer and after the MLP).
HYBRID_ATTN_LAYER_KEY_MAP: Dict[str, str] = {
    "attention.w_q.weight": "self_attn.q_proj.weight",
    "attention.w_k.weight": "self_attn.k_proj.weight",
    "attention.w_v.weight": "self_attn.v_proj.weight",
    "attention.w_out.weight": "self_attn.o_proj.weight",
    "attention.q_norm.weight": "self_attn.q_norm.weight",
    "attention.k_norm.weight": "self_attn.k_norm.weight",
    "attention_norm.weight": "post_attention_layernorm.weight",
    "feed_forward_norm.weight": "post_feedforward_layernorm.weight",
    "feed_forward.w1.weight": "mlp.gate_proj.weight",
    "feed_forward.w2.weight": "mlp.down_proj.weight",
    "feed_forward.w3.weight": "mlp.up_proj.weight",
}

#: Non-block keys shared across all hybrid models.
HYBRID_SHARED_KEY_MAP: Dict[str, str] = {
    "embeddings.weight": "model.embed_tokens.weight",
    "lm_head.norm.weight": "model.norm.weight",
    "lm_head.w_out.weight": "lm_head.weight",
}

_HYBRID_BLOCK_KEY_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")


@beta_feature
def convert_hybrid_state_to_hf(
    state_dict: Dict[str, Any],
    layer_types: List[str],
) -> Dict[str, Any]:
    """
    Convert an Olmo-core hybrid state dict to HF ``olmo_hybrid`` format.

    Uses :data:`HYBRID_SHARED_KEY_MAP` for non-block keys, and per-layer
    :data:`HYBRID_GDN_LAYER_KEY_MAP` / :data:`HYBRID_ATTN_LAYER_KEY_MAP`
    based on *layer_types*.

    :param state_dict: An unsharded Olmo-core model state dict.
    :param layer_types: Per-layer type list (``"linear_attention"`` or ``"full_attention"``).
    """
    hf_state: Dict[str, Any] = {}

    for olmo_key, value in state_dict.items():
        # Try shared (non-block) keys first.
        if olmo_key in HYBRID_SHARED_KEY_MAP:
            hf_state[HYBRID_SHARED_KEY_MAP[olmo_key]] = value
            continue

        m = _HYBRID_BLOCK_KEY_RE.match(olmo_key)
        if m is None:
            raise KeyError(f"Unmapped key: {olmo_key}")

        layer_idx = int(m.group(1))
        suffix = m.group(2)

        key_map = (
            HYBRID_GDN_LAYER_KEY_MAP
            if layer_types[layer_idx] == "linear_attention"
            else HYBRID_ATTN_LAYER_KEY_MAP
        )
        if suffix not in key_map:
            raise KeyError(
                f"Unmapped block suffix for layer {layer_idx} "
                f"(type={layer_types[layer_idx]!r}): {olmo_key}"
            )

        hf_key = f"model.layers.{layer_idx}.{key_map[suffix]}"
        hf_state[hf_key] = value

    return hf_state
