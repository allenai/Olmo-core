"""
Weight mappings between OLMo Core and MaxText, built on :class:`~olmo_core.nn.conversion.state_converter.StateConverter`
like the Hugging Face mappings in :mod:`olmo_core.nn.hf.convert`.

OLMo Core state uses ``named_parameters()`` names. MaxText state is the model's ``params``
collection flattened to ``/``-joined paths in the unscanned (``scan_layers=False``) layout, e.g.
``decoder/layers_3/mlp/wi_0/kernel``; see :func:`olmo_core.nn.maxtext.checkpoint.scan_params` for
the scanned layout.

Every mapping is a pure re-layout (transpose, reshape, split, concatenate), so it's inverted
mechanically and converting there and back is bit-exact. The mappings for a layer depend on the
MaxText decoder and on the layer's sequence mixer and feed-forward, since layers of a hybrid model
share OLMo Core names (``blocks.{i}.attention.*``) but not MaxText ones. They are built per layer
from the ``*_mappings`` builders below, which you may change or extend.
"""

import logging
from typing import Callable, Dict, List, Mapping, MutableMapping, Optional, Tuple

import torch

from olmo_core.doc_utils import beta_feature
from olmo_core.nn.conversion.state_converter import StateConverter
from olmo_core.nn.conversion.state_mapping import StateMappingTemplate

from .config import MaxTextDecoderBlock, MaxTextModelConfig

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

#: Builds the OLMo Core -> MaxText mappings for one part of a layer from the model config, the
#: OLMo Core prefix (e.g. ``blocks.3``) and the MaxText prefix (e.g. ``decoder/layers_3``).
MappingBuilder = Callable[[MaxTextModelConfig, str, str], List[StateMappingTemplate]]


def _rename(source: str, dest: str) -> StateMappingTemplate:
    return StateMappingTemplate(source, dest)


def _transpose(source: str, dest: str) -> StateMappingTemplate:
    """``nn.Linear`` weights are ``[out, in]``; MaxText kernels are ``[in, out]``."""
    return StateMappingTemplate(source, dest, dims_permutation=(1, 0))


# ------------------------------------------------------------------------------------------------
# Embeddings and LM head
# ------------------------------------------------------------------------------------------------


def _model_mappings(c: MaxTextModelConfig) -> List[StateMappingTemplate]:
    mappings = [
        _rename("embeddings.weight", "token_embedder/embedding"),
        _rename("lm_head.norm.weight", "decoder/decoder_norm/scale"),
        _transpose("lm_head.w_out.weight", "decoder/logits_dense/kernel"),
    ]
    if c.decoder_block == MaxTextDecoderBlock.olmoe3:
        mappings.append(_rename("embedding_norm.weight", "decoder/embedding_norm/scale"))
    return mappings


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


def _norm_mappings(c: MaxTextModelConfig, b: str, m: str) -> List[StateMappingTemplate]:
    return [
        _rename(f"{b}.{norm}.weight", f"{m}/{mt_norm}/scale")
        for norm, mt_norm in BLOCK_NORM_MAPPINGS[c.decoder_block].items()
    ]


# ------------------------------------------------------------------------------------------------
# Attention
# ------------------------------------------------------------------------------------------------


def _attention_mappings(c: MaxTextModelConfig, b: str, m: str) -> List[StateMappingTemplate]:
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


def _kda_mappings(c: MaxTextModelConfig, b: str, m: str) -> List[StateMappingTemplate]:
    a, mt = f"{b}.attention", f"{m}/mixer"
    mappings = [_transpose(f"{a}.{w}.weight", f"{mt}/{w}/kernel") for w in KDA_KERNELS]
    mappings += [_rename(f"{a}.{k}", f"{mt}/{v}") for k, v in KDA_RENAMES.items()]
    # Depthwise torch Conv1d weights are [C, 1, K]; both sides put the current token on the last tap.
    mappings += [
        StateMappingTemplate(f"{a}.{x}_conv1d.weight", f"{mt}/{x}_conv", flatten_dims=(1, 2))
        for x in "qkv"
    ]
    return mappings


# ------------------------------------------------------------------------------------------------
# Feed-forward
# ------------------------------------------------------------------------------------------------


def _feed_forward_mappings(c: MaxTextModelConfig, b: str, m: str) -> List[StateMappingTemplate]:
    # OLMo Core computes w2(silu(w1(x)) * w3(x)); MaxText silu(x @ wi_0) * (x @ wi_1) @ wo.
    return [
        _transpose(f"{b}.feed_forward.w1.weight", f"{m}/mlp/wi_0/kernel"),
        _transpose(f"{b}.feed_forward.w3.weight", f"{m}/mlp/wi_1/kernel"),
        _transpose(f"{b}.feed_forward.w2.weight", f"{m}/mlp/wo/kernel"),
    ]


def _shared_expert_mappings(c: MaxTextModelConfig, b: str, m: str) -> List[StateMappingTemplate]:
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


def _routed_expert_mappings(c: MaxTextModelConfig, b: str, m: str) -> List[StateMappingTemplate]:
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


#: Sequence mixer type (as in OLMo Core's config) -> its mapping builder.
SEQUENCE_MIXER_MAPPINGS: Dict[str, MappingBuilder] = {
    "attention": _attention_mappings,
    "kimi_delta_attention": _kda_mappings,
}


def _layer_mappings(c: MaxTextModelConfig, layer: int) -> List[StateMappingTemplate]:
    spec = c.layers[layer]
    builders = [_norm_mappings, SEQUENCE_MIXER_MAPPINGS[spec.sequence_mixer]]
    if c.decoder_block == MaxTextDecoderBlock.olmo3:
        builders.append(_feed_forward_mappings)
    else:
        builders.append(_shared_expert_mappings)
        if spec.moe:
            builders.append(_routed_expert_mappings)
    b, m = f"blocks.{layer}", f"decoder/layers_{layer}"
    return [t for build in builders for t in build(c, b, m)]


def _mappings(c: MaxTextModelConfig, layer: Optional[int]) -> List[StateMappingTemplate]:
    return _model_mappings(c) if layer is None else _layer_mappings(c, layer)


def _invert(
    t: StateMappingTemplate, source_shapes: Mapping[str, Tuple[int, ...]]
) -> StateMappingTemplate:
    """
    The mapping that undoes ``t``. :class:`StateConverter` concatenates, unflattens, permutes,
    flattens and chunks, in that order, so the inverse undoes each step in reverse. Undoing a
    flatten needs the flattened dims' sizes, which come from ``source_shapes``.
    """
    sources = (
        t.source_template_keys
        if isinstance(t.source_template_keys, tuple)
        else (t.source_template_keys,)
    )
    x = torch.cat(
        [torch.empty(source_shapes[k], device="meta") for k in sources], dim=t.source_concat_dim
    )
    flatten_dims = unflatten_dim = dims_permutation = None
    if t.unflatten_dim is not None:
        dim, shape = t.unflatten_dim
        x = x.unflatten(dim, tuple(int(s) for s in shape))
        flatten_dims = (dim, dim + len(shape) - 1)
    if t.dims_permutation is not None:
        x = x.permute(*t.dims_permutation)
        dims_permutation = tuple(
            sorted(range(len(t.dims_permutation)), key=t.dims_permutation.__getitem__)
        )
    if t.flatten_dims is not None:
        first, last = t.flatten_dims
        unflatten_dim = (first, tuple(x.shape[first : last + 1]))
    return StateMappingTemplate(
        t.dest_template_keys,
        t.source_template_keys,
        source_concat_dim=t.dest_chunk_dim,
        unflatten_dim=unflatten_dim,
        dims_permutation=dims_permutation,
        flatten_dims=flatten_dims,
        dest_chunk_dim=t.source_concat_dim,
    )


@beta_feature
def get_converter_to_maxtext(
    config: MaxTextModelConfig, layer: Optional[int] = None
) -> StateConverter:
    """
    The converter from OLMo Core to MaxText state for one layer, or for the parameters outside the
    layers (embeddings and LM head) if ``layer`` is ``None``.
    """
    return StateConverter(_mappings(config, layer))


@beta_feature
def get_converter_from_maxtext(
    config: MaxTextModelConfig, layer: Optional[int] = None
) -> StateConverter:
    """The inverse of :func:`get_converter_to_maxtext`."""
    shapes = olmo_core_shapes(config)
    return StateConverter([_invert(t, shapes) for t in _mappings(config, layer)])


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
    ``model.`` (train module), ``module.`` (DDP wrapper), ``_orig_mod.`` (``torch.compile``) and
    ``.main`` (the fp32 master copy that the OLMoDDP train module saves).
    """
    key = key.removesuffix(".main")
    for prefix in ("model.", "module.", "_orig_mod."):
        key = key.removeprefix(prefix)
    return key


def split_layer_key(key: str, prefix: str, sep: str) -> Optional[Tuple[int, str]]:
    """Split ``{prefix}{i}{sep}{rest}`` into ``(i, rest)``, or ``None`` if ``key`` isn't of that form."""
    if not key.startswith(prefix):
        return None
    i, _, rest = key[len(prefix) :].partition(sep)
    return (int(i), rest) if i.isdigit() else None


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
    prefix, sep = ("blocks.", ".") if to_maxtext else ("decoder/layers_", "/")
    by_layer: Dict[Optional[int], List[str]] = {}
    for k in state:
        split = split_layer_key(k, prefix, sep)
        by_layer.setdefault(None if split is None else split[0], []).append(k)
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
