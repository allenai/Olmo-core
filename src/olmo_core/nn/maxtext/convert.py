"""
Weight mappings between OLMo Core and MaxText, built on :class:`~olmo_core.nn.conversion.state_converter.StateConverter`
like the Hugging Face mappings in :mod:`olmo_core.nn.hf.convert`.

OLMo Core state uses ``named_parameters()`` names. MaxText state is the model's ``params``
collection flattened to ``/``-joined paths in the unscanned (``scan_layers=False``) layout, e.g.
``decoder/layers_3/mlp/wi_0/kernel``; see :func:`olmo_core.nn.maxtext.checkpoint.scan_params` for
the scanned layout.

Every mapping is a pure re-layout (transpose, reshape, split, concatenate), so converting there
and back is bit-exact. The mappings for a layer depend on the MaxText decoder and on the layer's
sequence mixer and feed-forward, since layers of a hybrid model share OLMo Core names
(``blocks.{i}.attention.*``) but not MaxText ones. They are built per layer from the
``*_TO_MAXTEXT`` / ``*_FROM_MAXTEXT`` builders below, which you may change or extend.
"""

import logging
from typing import Callable, Dict, List, Mapping, MutableMapping, Tuple

import torch

from olmo_core.doc_utils import beta_feature
from olmo_core.nn.conversion.state_converter import StateConverter
from olmo_core.nn.conversion.state_mapping import StateMappingTemplate

from .config import LayerSpec, MaxTextDecoderBlock, MaxTextModelConfig

__all__ = [
    "get_converter_to_maxtext",
    "get_converter_from_maxtext",
    "convert_state_to_maxtext",
    "convert_state_from_maxtext",
    "olmo_core_shapes",
    "maxtext_shapes",
    "normalize_olmo_core_key",
]

log = logging.getLogger(__name__)

#: Builds the mappings for one part of the model from the model config, the OLMo Core prefix
#: (e.g. ``blocks.3``) and the MaxText prefix (e.g. ``decoder/layers_3``).
MappingBuilder = Callable[[MaxTextModelConfig, str, str], List[StateMappingTemplate]]


def _rename(source: str, dest: str) -> StateMappingTemplate:
    return StateMappingTemplate(source, dest)


def _transpose(source: str, dest: str) -> StateMappingTemplate:
    """``nn.Linear`` weights are ``[out, in]``; MaxText kernels are ``[in, out]``."""
    return StateMappingTemplate(source, dest, dims_permutation=(1, 0))


# ------------------------------------------------------------------------------------------------
# Embeddings and LM head
# ------------------------------------------------------------------------------------------------


def _model_to_maxtext(c: MaxTextModelConfig, b: str, m: str) -> List[StateMappingTemplate]:
    del b, m
    mappings = [
        _rename("embeddings.weight", "token_embedder/embedding"),
        _rename("lm_head.norm.weight", "decoder/decoder_norm/scale"),
        _transpose("lm_head.w_out.weight", "decoder/logits_dense/kernel"),
    ]
    if c.decoder_block == MaxTextDecoderBlock.olmoe3:
        mappings.append(_rename("embedding_norm.weight", "decoder/embedding_norm/scale"))
    return mappings


def _model_from_maxtext(c: MaxTextModelConfig, b: str, m: str) -> List[StateMappingTemplate]:
    return [_invert_simple(t) for t in _model_to_maxtext(c, b, m)]


def _invert_simple(t: StateMappingTemplate) -> StateMappingTemplate:
    """Inverse of a :func:`_rename` or :func:`_transpose` mapping."""
    assert isinstance(t.source_template_keys, str) and isinstance(t.dest_template_keys, str)
    assert t.unflatten_dim is None and t.flatten_dims is None
    return StateMappingTemplate(
        t.dest_template_keys, t.source_template_keys, dims_permutation=t.dims_permutation
    )


# ------------------------------------------------------------------------------------------------
# Norms
# ------------------------------------------------------------------------------------------------

#: OLMo Core block norm -> MaxText layer norm, per decoder.
BLOCK_NORM_MAPPINGS: Dict[str, Dict[str, str]] = {
    # OLMo Core's reordered-norm block normalizes the attention and feed-forward outputs.
    MaxTextDecoderBlock.olmo3: {
        "attention_norm": "post_self_attention_layer_norm",
        "feed_forward_norm": "post_mlp_layer_norm",
    },
    # Peri-norm: each sublayer's input and output.
    MaxTextDecoderBlock.olmoe3: {
        "attention_input_norm": "attn_in_norm",
        "attention_norm": "attn_out_norm",
        "feed_forward_input_norm": "ffn_in_norm",
        "feed_forward_norm": "ffn_out_norm",
    },
}


def _norms_to_maxtext(c: MaxTextModelConfig, b: str, m: str) -> List[StateMappingTemplate]:
    return [
        _rename(f"{b}.{norm}.weight", f"{m}/{mt_norm}/scale")
        for norm, mt_norm in BLOCK_NORM_MAPPINGS[c.decoder_block].items()
    ]


# ------------------------------------------------------------------------------------------------
# Attention
# ------------------------------------------------------------------------------------------------


def _attention_to_maxtext(c: MaxTextModelConfig, b: str, m: str) -> List[StateMappingTemplate]:
    a, hd = f"{b}.attention", c.attention.head_dim
    if c.decoder_block == MaxTextDecoderBlock.olmoe3:
        m = f"{m}/mixer"
    mt = f"{m}/attention"
    mappings = [
        # [kv_heads * head_dim, d_model] -> [d_model, kv_heads, head_dim]
        StateMappingTemplate(
            f"{a}.w_k.weight",
            f"{mt}/key/kernel",
            unflatten_dim=(0, (-1, hd)),
            dims_permutation=(2, 0, 1),
        ),
        StateMappingTemplate(
            f"{a}.w_v.weight",
            f"{mt}/value/kernel",
            unflatten_dim=(0, (-1, hd)),
            dims_permutation=(2, 0, 1),
        ),
    ]
    if c.attention.gated:
        # MaxText widens the query projection to 2 * head_dim per head and splits off the second
        # half as the elementwise output gate; OLMo Core keeps a separate w_g.
        # [2 * heads * head_dim, d_model] -> [2, heads, head_dim, d_model] -> [d_model, heads, 2 * head_dim]
        mappings.append(
            StateMappingTemplate(
                (f"{a}.w_q.weight", f"{a}.w_g.weight"),
                f"{mt}/query/kernel",
                unflatten_dim=(0, (2, -1, hd)),
                dims_permutation=(3, 1, 0, 2),
                flatten_dims=(2, 3),
            )
        )
    else:
        mappings.append(
            StateMappingTemplate(
                f"{a}.w_q.weight",
                f"{mt}/query/kernel",
                unflatten_dim=(0, (-1, hd)),
                dims_permutation=(2, 0, 1),
            )
        )
    if c.decoder_block == MaxTextDecoderBlock.olmoe3:
        mappings.append(_transpose(f"{a}.w_out.weight", f"{mt}/out/kernel"))
    else:
        # [d_model, heads * head_dim] -> [heads, head_dim, d_model]
        mappings.append(
            StateMappingTemplate(
                f"{a}.w_out.weight",
                f"{mt}/out/kernel",
                unflatten_dim=(1, (-1, hd)),
                dims_permutation=(1, 2, 0),
            )
        )
    if c.attention.qk_norm:
        mappings += [
            _rename(f"{a}.q_norm.weight", f"{mt}/query_norm/scale"),
            _rename(f"{a}.k_norm.weight", f"{mt}/key_norm/scale"),
        ]
    if c.attention.scalable_softmax:
        mappings.append(_rename(f"{a}.ssmax_scale", f"{mt}/ssmax_scale"))
    return mappings


def _attention_from_maxtext(c: MaxTextModelConfig, b: str, m: str) -> List[StateMappingTemplate]:
    a, hd = f"{b}.attention", c.attention.head_dim
    if c.decoder_block == MaxTextDecoderBlock.olmoe3:
        m = f"{m}/mixer"
    mt = f"{m}/attention"
    mappings = [
        StateMappingTemplate(
            f"{mt}/key/kernel", f"{a}.w_k.weight", dims_permutation=(1, 2, 0), flatten_dims=(0, 1)
        ),
        StateMappingTemplate(
            f"{mt}/value/kernel", f"{a}.w_v.weight", dims_permutation=(1, 2, 0), flatten_dims=(0, 1)
        ),
    ]
    if c.attention.gated:
        # [d_model, heads, 2 * head_dim] -> [2, heads, head_dim, d_model], then split in two.
        mappings.append(
            StateMappingTemplate(
                f"{mt}/query/kernel",
                (f"{a}.w_q.weight", f"{a}.w_g.weight"),
                unflatten_dim=(2, (2, hd)),
                dims_permutation=(2, 1, 3, 0),
                flatten_dims=(0, 2),
            )
        )
    else:
        mappings.append(
            StateMappingTemplate(
                f"{mt}/query/kernel",
                f"{a}.w_q.weight",
                dims_permutation=(1, 2, 0),
                flatten_dims=(0, 1),
            )
        )
    if c.decoder_block == MaxTextDecoderBlock.olmoe3:
        mappings.append(_transpose(f"{mt}/out/kernel", f"{a}.w_out.weight"))
    else:
        mappings.append(
            StateMappingTemplate(
                f"{mt}/out/kernel",
                f"{a}.w_out.weight",
                dims_permutation=(2, 0, 1),
                flatten_dims=(1, 2),
            )
        )
    if c.attention.qk_norm:
        mappings += [
            _rename(f"{mt}/query_norm/scale", f"{a}.q_norm.weight"),
            _rename(f"{mt}/key_norm/scale", f"{a}.k_norm.weight"),
        ]
    if c.attention.scalable_softmax:
        mappings.append(_rename(f"{mt}/ssmax_scale", f"{a}.ssmax_scale"))
    return mappings


# ------------------------------------------------------------------------------------------------
# Kimi Delta Attention
# ------------------------------------------------------------------------------------------------

#: KDA projections that are plain ``nn.Linear`` weights on the OLMo Core side, with the same names
#: in MaxText.
KDA_KERNELS = ("w_q", "w_k", "w_v", "f_proj_1", "f_proj_2", "w_b", "g_proj_1", "g_proj_2", "w_out")

#: KDA parameters with the same layout on both sides: OLMo Core suffix -> MaxText suffix.
KDA_RENAMES = {
    "g_proj_2.bias": "g_proj_2/bias",
    "A_log": "A_log",
    "dt_bias": "dt_bias",
    "o_norm.weight": "o_norm/scale",
}


def _kda_to_maxtext(c: MaxTextModelConfig, b: str, m: str) -> List[StateMappingTemplate]:
    a, mt = f"{b}.attention", f"{m}/mixer"
    mappings = [_transpose(f"{a}.{w}.weight", f"{mt}/{w}/kernel") for w in KDA_KERNELS]
    mappings += [_rename(f"{a}.{k}", f"{mt}/{v}") for k, v in KDA_RENAMES.items()]
    # Depthwise torch Conv1d weights are [C, 1, K]; both sides put the current token on the last tap.
    mappings += [
        StateMappingTemplate(f"{a}.{x}_conv1d.weight", f"{mt}/{x}_conv", flatten_dims=(1, 2))
        for x in "qkv"
    ]
    return mappings


def _kda_from_maxtext(c: MaxTextModelConfig, b: str, m: str) -> List[StateMappingTemplate]:
    a, mt = f"{b}.attention", f"{m}/mixer"
    mappings = [_transpose(f"{mt}/{w}/kernel", f"{a}.{w}.weight") for w in KDA_KERNELS]
    mappings += [_rename(f"{mt}/{v}", f"{a}.{k}") for k, v in KDA_RENAMES.items()]
    mappings += [
        StateMappingTemplate(f"{mt}/{x}_conv", f"{a}.{x}_conv1d.weight", unflatten_dim=(1, (1, -1)))
        for x in "qkv"
    ]
    return mappings


# ------------------------------------------------------------------------------------------------
# Feed-forward
# ------------------------------------------------------------------------------------------------


def _feed_forward_to_maxtext(c: MaxTextModelConfig, b: str, m: str) -> List[StateMappingTemplate]:
    # OLMo Core computes w2(silu(w1(x)) * w3(x)); MaxText silu(x @ wi_0) * (x @ wi_1) @ wo.
    return [
        _transpose(f"{b}.feed_forward.w1.weight", f"{m}/mlp/wi_0/kernel"),
        _transpose(f"{b}.feed_forward.w3.weight", f"{m}/mlp/wi_1/kernel"),
        _transpose(f"{b}.feed_forward.w2.weight", f"{m}/mlp/wo/kernel"),
    ]


def _feed_forward_from_maxtext(c: MaxTextModelConfig, b: str, m: str) -> List[StateMappingTemplate]:
    return [_invert_simple(t) for t in _feed_forward_to_maxtext(c, b, m)]


def _shared_expert_to_maxtext(c: MaxTextModelConfig, b: str, m: str) -> List[StateMappingTemplate]:
    # x @ w_up_gate gives [up | gate] along the columns, and the block computes up * silu(gate).
    # MaxText computes silu(wi_0) * wi_1, so wi_0 is the gate and wi_1 is up.
    return [
        StateMappingTemplate(
            f"{b}.shared_experts.w_up_gate",
            (f"{m}/shared_ffn/wi_1/kernel", f"{m}/shared_ffn/wi_0/kernel"),
            dest_chunk_dim=1,
        ),
        # [1, hidden, d_model] -> [hidden, d_model]
        StateMappingTemplate(
            f"{b}.shared_experts.w_down", f"{m}/shared_ffn/wo/kernel", flatten_dims=(0, 1)
        ),
    ]


def _shared_expert_from_maxtext(
    c: MaxTextModelConfig, b: str, m: str
) -> List[StateMappingTemplate]:
    return [
        StateMappingTemplate(
            (f"{m}/shared_ffn/wi_1/kernel", f"{m}/shared_ffn/wi_0/kernel"),
            f"{b}.shared_experts.w_up_gate",
            source_concat_dim=1,
        ),
        StateMappingTemplate(
            f"{m}/shared_ffn/wo/kernel", f"{b}.shared_experts.w_down", unflatten_dim=(0, (1, -1))
        ),
    ]


def _routed_experts_to_maxtext(c: MaxTextModelConfig, b: str, m: str) -> List[StateMappingTemplate]:
    assert c.moe is not None
    return [
        _transpose(f"{b}.latent_down_proj.weight", f"{m}/latent_down/kernel"),
        _transpose(f"{b}.latent_up_proj.weight", f"{m}/latent_up/kernel"),
        # The router stores its [experts, d_model] weight flattened and computes x @ W.T.
        StateMappingTemplate(
            f"{b}.routed_experts_router.weight",
            f"{m}/moe_block/gate/kernel",
            unflatten_dim=(0, (c.moe.num_experts, -1)),
            dims_permutation=(1, 0),
        ),
        # Routed experts compute x @ w_up_gate[e].T with up in rows [:hidden] and the gate in rows
        # [hidden:]. MaxText's wi_* are [experts, latent, hidden].
        StateMappingTemplate(
            f"{b}.routed_experts.w_up_gate",
            (f"{m}/moe_block/wi_1", f"{m}/moe_block/wi_0"),
            dims_permutation=(0, 2, 1),
            dest_chunk_dim=2,
        ),
        # Both are [experts, hidden, latent].
        _rename(f"{b}.routed_experts.w_down", f"{m}/moe_block/wo"),
    ]


def _routed_experts_from_maxtext(
    c: MaxTextModelConfig, b: str, m: str
) -> List[StateMappingTemplate]:
    return [
        _transpose(f"{m}/latent_down/kernel", f"{b}.latent_down_proj.weight"),
        _transpose(f"{m}/latent_up/kernel", f"{b}.latent_up_proj.weight"),
        StateMappingTemplate(
            f"{m}/moe_block/gate/kernel",
            f"{b}.routed_experts_router.weight",
            dims_permutation=(1, 0),
            flatten_dims=(0, 1),
        ),
        StateMappingTemplate(
            (f"{m}/moe_block/wi_1", f"{m}/moe_block/wi_0"),
            f"{b}.routed_experts.w_up_gate",
            source_concat_dim=2,
            dims_permutation=(0, 2, 1),
        ),
        _rename(f"{m}/moe_block/wo", f"{b}.routed_experts.w_down"),
    ]


#: Sequence mixer type (as in OLMo Core's config) -> (to MaxText, from MaxText).
SEQUENCE_MIXER_MAPPINGS: Dict[str, Tuple[MappingBuilder, MappingBuilder]] = {
    "attention": (_attention_to_maxtext, _attention_from_maxtext),
    "kimi_delta_attention": (_kda_to_maxtext, _kda_from_maxtext),
}


def _layer_builders(
    c: MaxTextModelConfig, layer: LayerSpec
) -> List[Tuple[MappingBuilder, MappingBuilder]]:
    builders = [
        (_norms_to_maxtext, _norms_from_maxtext),
        SEQUENCE_MIXER_MAPPINGS[layer.sequence_mixer],
    ]
    if c.decoder_block == MaxTextDecoderBlock.olmo3:
        builders.append((_feed_forward_to_maxtext, _feed_forward_from_maxtext))
    else:
        builders.append((_shared_expert_to_maxtext, _shared_expert_from_maxtext))
        if layer.moe:
            builders.append((_routed_experts_to_maxtext, _routed_experts_from_maxtext))
    return builders


def _norms_from_maxtext(c: MaxTextModelConfig, b: str, m: str) -> List[StateMappingTemplate]:
    return [_invert_simple(t) for t in _norms_to_maxtext(c, b, m)]


def _prefixes(layer: int) -> Tuple[str, str]:
    return f"blocks.{layer}", f"decoder/layers_{layer}"


@beta_feature
def get_converter_to_maxtext(
    config: MaxTextModelConfig, layer: int | None = None
) -> StateConverter:
    """
    The converter from OLMo Core to MaxText state for one layer, or for the parameters outside the
    layers (embeddings and LM head) if ``layer`` is ``None``.
    """
    if layer is None:
        return StateConverter(_model_to_maxtext(config, "", ""))
    b, m = _prefixes(layer)
    return StateConverter(
        [
            t
            for to_mt, _ in _layer_builders(config, config.layers[layer])
            for t in to_mt(config, b, m)
        ]
    )


@beta_feature
def get_converter_from_maxtext(
    config: MaxTextModelConfig, layer: int | None = None
) -> StateConverter:
    """The inverse of :func:`get_converter_to_maxtext`."""
    if layer is None:
        return StateConverter(_model_from_maxtext(config, "", ""))
    b, m = _prefixes(layer)
    return StateConverter(
        [
            t
            for _, from_mt in _layer_builders(config, config.layers[layer])
            for t in from_mt(config, b, m)
        ]
    )


# ------------------------------------------------------------------------------------------------
# Shapes
# ------------------------------------------------------------------------------------------------


def olmo_core_shapes(c: MaxTextModelConfig) -> Dict[str, Tuple[int, ...]]:
    """Unsharded OLMo Core parameter shapes, keyed by ``named_parameters()`` name."""
    D, V = c.d_model, c.vocab_size
    shapes: Dict[str, Tuple[int, ...]] = {
        "embeddings.weight": (V, D),
        "lm_head.norm.weight": (D,),
        "lm_head.w_out.weight": (V, D),
    }
    if c.decoder_block == MaxTextDecoderBlock.olmoe3:
        shapes["embedding_norm.weight"] = (D,)
    for i, layer in enumerate(c.layers):
        b = f"blocks.{i}"
        for norm in BLOCK_NORM_MAPPINGS[c.decoder_block]:
            shapes[f"{b}.{norm}.weight"] = (D,)
        if layer.sequence_mixer == "attention":
            shapes.update(_attention_shapes(c, f"{b}.attention"))
        else:
            shapes.update(_kda_shapes(c, f"{b}.attention"))
        F = layer.hidden_size
        if c.decoder_block == MaxTextDecoderBlock.olmo3:
            shapes[f"{b}.feed_forward.w1.weight"] = (F, D)
            shapes[f"{b}.feed_forward.w2.weight"] = (D, F)
            shapes[f"{b}.feed_forward.w3.weight"] = (F, D)
        else:
            shapes[f"{b}.shared_experts.w_up_gate"] = (D, 2 * F)
            shapes[f"{b}.shared_experts.w_down"] = (1, F, D)
        if layer.moe:
            assert c.moe is not None
            E, H, L = c.moe.num_experts, c.moe.hidden_size, c.moe.latent_dim
            shapes[f"{b}.routed_experts_router.weight"] = (E * D,)
            shapes[f"{b}.routed_experts.w_up_gate"] = (E, 2 * H, L)
            shapes[f"{b}.routed_experts.w_down"] = (E, H, L)
            shapes[f"{b}.latent_down_proj.weight"] = (L, D)
            shapes[f"{b}.latent_up_proj.weight"] = (D, L)
    return shapes


def _attention_shapes(c: MaxTextModelConfig, a: str) -> Dict[str, Tuple[int, ...]]:
    D, at = c.d_model, c.attention
    q_dim, kv_dim = at.n_heads * at.head_dim, at.n_kv_heads * at.head_dim
    shapes: Dict[str, Tuple[int, ...]] = {
        f"{a}.w_q.weight": (q_dim, D),
        f"{a}.w_k.weight": (kv_dim, D),
        f"{a}.w_v.weight": (kv_dim, D),
        f"{a}.w_out.weight": (D, q_dim),
    }
    if at.gated:
        shapes[f"{a}.w_g.weight"] = (q_dim, D)
    if at.qk_norm and at.per_head_qk_norm:
        shapes[f"{a}.q_norm.weight"] = (at.n_heads, at.head_dim)
        shapes[f"{a}.k_norm.weight"] = (at.n_kv_heads, at.head_dim)
    elif at.qk_norm:
        shapes[f"{a}.q_norm.weight"] = (q_dim,)
        shapes[f"{a}.k_norm.weight"] = (kv_dim,)
    if at.scalable_softmax:
        shapes[f"{a}.ssmax_scale"] = (at.n_heads,)
    return shapes


def _kda_shapes(c: MaxTextModelConfig, a: str) -> Dict[str, Tuple[int, ...]]:
    assert c.kda is not None
    D, k = c.d_model, c.kda
    key_dim, value_dim = k.n_heads * k.head_dim, k.n_heads * k.head_v_dim
    return {
        f"{a}.w_q.weight": (key_dim, D),
        f"{a}.w_k.weight": (key_dim, D),
        f"{a}.w_v.weight": (value_dim, D),
        f"{a}.f_proj_1.weight": (k.head_v_dim, D),
        f"{a}.f_proj_2.weight": (key_dim, k.head_v_dim),
        f"{a}.w_b.weight": (k.n_heads, D),
        f"{a}.g_proj_1.weight": (k.head_v_dim, D),
        f"{a}.g_proj_2.weight": (value_dim, k.head_v_dim),
        f"{a}.w_out.weight": (D, value_dim),
        f"{a}.g_proj_2.bias": (value_dim,),
        f"{a}.A_log": (k.n_heads,),
        f"{a}.dt_bias": (key_dim,),
        f"{a}.o_norm.weight": (k.head_v_dim,),
        f"{a}.q_conv1d.weight": (key_dim, 1, k.conv_size),
        f"{a}.k_conv1d.weight": (key_dim, 1, k.conv_size),
        f"{a}.v_conv1d.weight": (value_dim, 1, k.conv_size),
    }


def maxtext_shapes(c: MaxTextModelConfig) -> Dict[str, Tuple[int, ...]]:
    """Unscanned MaxText parameter shapes, keyed by ``/``-joined path."""
    meta = {k: torch.empty(s, device="meta") for k, s in olmo_core_shapes(c).items()}
    return {k: tuple(v.shape) for k, v in _convert(c, meta, to_maxtext=True, check=False).items()}


# ------------------------------------------------------------------------------------------------
# Converting
# ------------------------------------------------------------------------------------------------


def normalize_olmo_core_key(key: str) -> str:
    """
    Strip the prefixes and suffixes that OLMo Core checkpoints put around parameter names:
    ``model.`` (train module), ``module.`` (DDP wrapper) and ``.main`` (the fp32 master copy that
    the OLMoDDP train module saves).
    """
    for prefix in ("model.", "module."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    return key[: -len(".main")] if key.endswith(".main") else key


def _check_shapes(
    state: Mapping[str, torch.Tensor], expected: Mapping[str, Tuple[int, ...]], what: str
):
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    if missing or extra:
        raise KeyError(
            f"{what} keys don't match the model: missing={missing[:10]} extra={extra[:10]}"
        )
    for k, shape in expected.items():
        if tuple(state[k].shape) != tuple(shape):
            raise ValueError(f"{what} {k}: shape {tuple(state[k].shape)} != expected {shape}")


def _layer_of(key: str, to_maxtext: bool) -> int | None:
    prefix, sep = ("blocks.", ".") if to_maxtext else ("decoder/layers_", "/")
    if not key.startswith(prefix):
        return None
    return int(key[len(prefix) :].split(sep, 1)[0])


def _convert(
    c: MaxTextModelConfig,
    state: MutableMapping[str, torch.Tensor],
    *,
    to_maxtext: bool,
    check: bool = True,
    consume: bool = False,
    round_trip: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    :param consume: Delete each layer from ``state`` once it's converted, so the input is freed as
        the output grows.
    :param round_trip: Check that converting each layer back reproduces it exactly.
    """
    if check:
        _check_shapes(
            state,
            olmo_core_shapes(c) if to_maxtext else maxtext_shapes(c),
            "OLMo Core" if to_maxtext else "MaxText",
        )
    by_layer: Dict[int | None, List[str]] = {}
    for k in state:
        by_layer.setdefault(_layer_of(k, to_maxtext), []).append(k)
    get_converter = get_converter_to_maxtext if to_maxtext else get_converter_from_maxtext
    get_inverse = get_converter_from_maxtext if to_maxtext else get_converter_to_maxtext

    out: Dict[str, torch.Tensor] = {}
    for layer in [None, *range(c.n_layers)]:
        keys = by_layer.get(layer, [])
        part = {k: state[k] for k in keys}
        converted = get_converter(c, layer).convert(part, {})
        if round_trip:
            back = get_inverse(c, layer).convert(converted, {})
            mismatched = [k for k in part if not torch.equal(part[k], back[k])]
            if mismatched:
                raise RuntimeError(f"converting back doesn't reproduce {mismatched}")
        out.update(converted)
        if consume:
            for k in keys:
                del state[k]
    if check:
        _check_shapes(
            out,
            maxtext_shapes(c) if to_maxtext else olmo_core_shapes(c),
            "MaxText" if to_maxtext else "OLMo Core",
        )
    return out


@beta_feature
def convert_state_to_maxtext(
    config: MaxTextModelConfig, olmo_core_state: Mapping[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Convert an unsharded OLMo Core model state dict (``named_parameters()`` names, full shapes) to
    MaxText's unscanned parameter tree, flattened to ``/``-joined paths. Every key must be converted
    and every shape is checked, in and out.
    """
    return _convert(config, dict(olmo_core_state), to_maxtext=True)


@beta_feature
def convert_state_from_maxtext(
    config: MaxTextModelConfig, maxtext_state: Mapping[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """The inverse of :func:`convert_state_to_maxtext`."""
    return _convert(config, dict(maxtext_state), to_maxtext=False)
