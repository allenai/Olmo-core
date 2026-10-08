"""
Weight mapping between OLMo Core's OLMoE3 (OLMo 3.5) model and MaxText's ``olmoe3`` decoder.

Everything here is a pure re-layout (transpose, reshape, split, concatenate, stack), so a round trip
in either direction is bit-exact. This module only depends on numpy; reading and writing the
actual checkpoint formats (torch DCP and Orbax) lives in ``src/scripts/maxtext/``.

MaxText parameter paths are ``/``-joined, relative to the model's ``params`` collection, e.g.
``decoder/layers_3/mixer/w_q/kernel``. With ``scan_layers=True`` MaxText keeps the first
:data:`MAXTEXT_CYCLE` layers unrolled under ``decoder/layers_0/layer_{j}`` and stacks every later
cycle under ``decoder/scanned_blocks/layer_{j}`` along axis :data:`MAXTEXT_SCAN_AXIS`;
:func:`scan_maxtext` and :func:`unscan_maxtext` convert between that and the unscanned
``decoder/layers_{i}`` layout that the mapping functions use.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

__all__ = [
    "MAXTEXT_CYCLE",
    "MAXTEXT_SCAN_AXIS",
    "OLMoE3LayerSpec",
    "OLMoE3Geometry",
    "olmo_core_shapes",
    "maxtext_shapes",
    "maxtext_overrides",
    "normalize_olmo_core_key",
    "olmo_core_to_maxtext",
    "maxtext_to_olmo_core",
    "scan_maxtext",
    "unscan_maxtext",
]

#: MaxText's ``inhomogeneous_layer_cycle_interval`` for olmoe3: one full-attention layer per cycle.
MAXTEXT_CYCLE = 8

#: MaxText's ``param_scan_axis``: scanned cycles are stacked along this axis of every parameter.
MAXTEXT_SCAN_AXIS = 1


@dataclass(frozen=True)
class OLMoE3LayerSpec:
    attention: bool
    """Full (NoPE, gated, SSMax) attention if true, otherwise Kimi Delta Attention."""

    moe: bool
    """Latent routed experts alongside the shared expert if true, otherwise a dense shared expert."""

    shared_hidden: int


@dataclass(frozen=True)
class OLMoE3Geometry:
    d_model: int
    vocab_size: int
    layers: Tuple[OLMoE3LayerSpec, ...]

    kda_heads: int
    kda_head_dim: int
    kda_head_v_dim: int
    kda_conv_size: int

    attn_heads: int
    attn_kv_heads: int
    attn_head_dim: int

    num_experts: int
    expert_hidden: int
    latent_dim: int
    top_k: int

    emo_pools: Optional[Tuple[int, int, int]] = None
    """EMO document expert pool sizes (min, max, eval), if EMO routing is on."""

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @classmethod
    def from_olmo_core_config(cls, model_config: Mapping[str, Any]) -> "OLMoE3Geometry":
        """
        Build the geometry from the ``model`` section of an OLMo Core ``config.json``
        (i.e. ``TransformerConfig.as_config_dict()``).
        """
        d_model = model_config["d_model"]
        n_layers = model_config["n_layers"]
        default = model_config["block"]
        overrides = model_config.get("block_overrides") or {}

        kda: Optional[Mapping[str, Any]] = None
        attn: Optional[Mapping[str, Any]] = None
        routed: Optional[Mapping[str, Any]] = None
        router: Optional[Mapping[str, Any]] = None
        latent: Optional[Mapping[str, Any]] = None
        layers = []
        for i in range(n_layers):
            block = overrides.get(str(i), overrides.get(i, default))
            mixer = block["sequence_mixer"]
            is_attention = mixer.get("type") == "attention"
            if is_attention:
                attn = _same(attn, mixer, ("n_heads", "n_kv_heads", "head_dim"), "attention")
                if not mixer.get("use_head_qk_norm") or not mixer.get("qk_norm_per_head_gains"):
                    raise ValueError(f"block {i}: MaxText olmoe3 needs per-head QK norm with gains")
                if not mixer.get("scalable_softmax") or mixer.get("rope") is not None:
                    raise ValueError(f"block {i}: MaxText olmoe3 needs NoPE + scalable softmax")
                if (mixer.get("gate") or {}).get("granularity") != "elementwise":
                    raise ValueError(f"block {i}: MaxText olmoe3 needs an elementwise output gate")
            elif mixer.get("type") == "kimi_delta_attention":
                kda = _same(
                    kda, mixer, ("n_heads", "n_v_heads", "head_dim", "expand_v", "conv_size"), "KDA"
                )
                if mixer.get("conv_bias"):
                    raise ValueError(f"block {i}: MaxText olmoe3 KDA has no conv bias")
            else:
                raise ValueError(f"block {i}: unsupported sequence mixer {mixer.get('type')!r}")

            shared = block["shared_experts"]
            if shared["num_experts"] != 1 or shared.get("bias"):
                raise ValueError(f"block {i}: expected a single bias-free shared expert")
            is_moe = block.get("routed_experts") is not None
            if is_moe:
                routed = _same(
                    routed, block["routed_experts"], ("num_experts", "hidden_size"), "MoE"
                )
                latent = _same(latent, block["latent_moe"], ("latent_dim",), "latent MoE")
                router = _same(
                    router, block["routed_experts_router"], ("top_k", "emo"), "MoE router"
                )
                if block["routed_experts"].get("bias") or block["latent_moe"].get("bias"):
                    raise ValueError(f"block {i}: MaxText olmoe3 experts have no bias")
                if block["routed_experts"].get("activation", "swiglu") != "swiglu":
                    raise ValueError(f"block {i}: MaxText olmoe3 experts use swiglu")
            if not block.get("use_peri_norm"):
                raise ValueError(f"block {i}: MaxText olmoe3 uses peri-norm blocks")
            layers.append(
                OLMoE3LayerSpec(
                    attention=is_attention, moe=is_moe, shared_hidden=shared["hidden_size"]
                )
            )

        if kda is None or attn is None or routed is None or latent is None or router is None:
            raise ValueError("expected KDA, full-attention, and MoE blocks in an OLMoE3 model")
        head_v_dim = int(kda["head_dim"] * kda["expand_v"])
        if kda["n_v_heads"] != kda["n_heads"]:
            raise ValueError("MaxText olmoe3 KDA needs n_v_heads == n_heads")
        return cls(
            d_model=d_model,
            vocab_size=model_config["vocab_size"],
            layers=tuple(layers),
            kda_heads=kda["n_heads"],
            kda_head_dim=kda["head_dim"],
            kda_head_v_dim=head_v_dim,
            kda_conv_size=kda["conv_size"],
            attn_heads=attn["n_heads"],
            attn_kv_heads=attn["n_kv_heads"],
            attn_head_dim=attn["head_dim"],
            num_experts=routed["num_experts"],
            expert_hidden=routed["hidden_size"],
            latent_dim=latent["latent_dim"],
            top_k=router["top_k"],
            emo_pools=None
            if router.get("emo") is None
            else (
                router["emo"]["min_document_expert_pool"],
                router["emo"]["max_document_expert_pool"],
                router["emo"]["eval_document_expert_pool"],
            ),
        )

    def validate_maxtext_layout(self) -> None:
        """
        Check that the layer pattern is the one MaxText's olmoe3 decoder builds: one dense layer
        first, then MoE everywhere, with full attention on the last layer of every cycle.
        """
        if self.n_layers % MAXTEXT_CYCLE:
            raise ValueError(f"n_layers={self.n_layers} must be a multiple of {MAXTEXT_CYCLE}")
        for i, layer in enumerate(self.layers):
            want_attention = (i + 1) % MAXTEXT_CYCLE == 0
            if layer.attention != want_attention:
                raise ValueError(
                    f"layer {i}: MaxText puts full attention on layers where (i + 1) % "
                    f"{MAXTEXT_CYCLE} == 0, but this model has attention={layer.attention}"
                )
            if layer.moe != (i > 0):
                raise ValueError(f"layer {i}: MaxText olmoe3 has exactly one dense layer, layer 0")
            if layer.moe and layer.shared_hidden != self.expert_hidden:
                raise ValueError(
                    f"layer {i}: shared expert hidden {layer.shared_hidden} != routed expert "
                    f"hidden {self.expert_hidden} (MaxText's shared_expert_mlp_dim default)"
                )


def maxtext_overrides(g: OLMoE3Geometry) -> Tuple[str, ...]:
    """
    MaxText ``key=value`` overrides that size ``model_name=olmo35-tiny`` (plus
    ``override_model_config=True``) to this geometry, so the converted weights load into it.
    """
    g.validate_maxtext_layout()
    dense_hidden = g.layers[0].shared_hidden
    overrides = [
        f"base_emb_dim={g.d_model}",
        f"vocab_size={g.vocab_size}",
        f"base_num_decoder_layers={g.n_layers}",
        f"base_num_query_heads={g.attn_heads}",
        f"base_num_kv_heads={g.attn_kv_heads}",
        f"head_dim={g.attn_head_dim}",
        "first_num_dense_layers=1",
        f"base_mlp_dim={dense_hidden}",
        f"base_moe_mlp_dim={g.expert_hidden}",
        f"moe_expert_input_dim={g.latent_dim}",
        f"num_experts={g.num_experts}",
        f"num_experts_per_tok={g.top_k}",
        f"inhomogeneous_layer_cycle_interval={MAXTEXT_CYCLE}",
        f"gdn_num_key_heads={g.kda_heads}",
        f"gdn_num_value_heads={g.kda_heads}",
        f"gdn_key_head_dim={g.kda_head_dim}",
        f"gdn_value_head_dim={g.kda_head_v_dim}",
        f"gdn_conv_kernel_dim={g.kda_conv_size}",
    ]
    if g.emo_pools is None:
        overrides.append("emo_enabled=False")
    else:
        lo, hi, ev = g.emo_pools
        overrides += [
            "emo_enabled=True",
            f"emo_min_document_expert_pool={lo}",
            f"emo_max_document_expert_pool={hi}",
            f"emo_eval_document_expert_pool={ev}",
        ]
    return tuple(overrides)


def _same(prev, cfg: Mapping[str, Any], keys: Tuple[str, ...], what: str):
    if prev is not None and any(prev.get(k) != cfg.get(k) for k in keys):
        raise ValueError(f"all {what} blocks must share {keys}")
    return cfg


def _kda_weight_shapes(g: OLMoE3Geometry) -> Dict[str, Tuple[int, ...]]:
    key_dim = g.kda_heads * g.kda_head_dim
    value_dim = g.kda_heads * g.kda_head_v_dim
    return {
        "w_q": (key_dim, g.d_model),
        "w_k": (key_dim, g.d_model),
        "w_v": (value_dim, g.d_model),
        "f_proj_1": (g.kda_head_v_dim, g.d_model),
        "f_proj_2": (key_dim, g.kda_head_v_dim),
        "w_b": (g.kda_heads, g.d_model),
        "g_proj_1": (g.kda_head_v_dim, g.d_model),
        "g_proj_2": (value_dim, g.kda_head_v_dim),
        "w_out": (g.d_model, value_dim),
    }


def olmo_core_shapes(g: OLMoE3Geometry) -> Dict[str, Tuple[int, ...]]:
    """Unsharded OLMo Core parameter shapes, keyed by ``named_parameters()`` name."""
    D, V = g.d_model, g.vocab_size
    shapes: Dict[str, Tuple[int, ...]] = {
        "embeddings.weight": (V, D),
        "embedding_norm.weight": (D,),
        "lm_head.norm.weight": (D,),
        "lm_head.w_out.weight": (V, D),
    }
    for i, layer in enumerate(g.layers):
        b = f"blocks.{i}"
        for norm in _NORMS:
            shapes[f"{b}.{norm}.weight"] = (D,)
        a = f"{b}.attention"
        if layer.attention:
            q_dim = g.attn_heads * g.attn_head_dim
            kv_dim = g.attn_kv_heads * g.attn_head_dim
            shapes[f"{a}.w_q.weight"] = (q_dim, D)
            shapes[f"{a}.w_g.weight"] = (q_dim, D)
            shapes[f"{a}.w_k.weight"] = (kv_dim, D)
            shapes[f"{a}.w_v.weight"] = (kv_dim, D)
            shapes[f"{a}.w_out.weight"] = (D, q_dim)
            shapes[f"{a}.q_norm.weight"] = (g.attn_heads, g.attn_head_dim)
            shapes[f"{a}.k_norm.weight"] = (g.attn_kv_heads, g.attn_head_dim)
            shapes[f"{a}.ssmax_scale"] = (g.attn_heads,)
        else:
            for name, shape in _kda_weight_shapes(g).items():
                shapes[f"{a}.{name}.weight"] = shape
            value_dim = g.kda_heads * g.kda_head_v_dim
            key_dim = g.kda_heads * g.kda_head_dim
            shapes[f"{a}.g_proj_2.bias"] = (value_dim,)
            shapes[f"{a}.A_log"] = (g.kda_heads,)
            shapes[f"{a}.dt_bias"] = (key_dim,)
            shapes[f"{a}.o_norm.weight"] = (g.kda_head_v_dim,)
            for conv, dim in (("q", key_dim), ("k", key_dim), ("v", value_dim)):
                shapes[f"{a}.{conv}_conv1d.weight"] = (dim, 1, g.kda_conv_size)
        F = layer.shared_hidden
        shapes[f"{b}.shared_experts.w_up_gate"] = (D, 2 * F)
        shapes[f"{b}.shared_experts.w_down"] = (1, F, D)
        if layer.moe:
            E, H, L = g.num_experts, g.expert_hidden, g.latent_dim
            shapes[f"{b}.routed_experts_router.weight"] = (E * D,)
            shapes[f"{b}.routed_experts.w_up_gate"] = (E, 2 * H, L)
            shapes[f"{b}.routed_experts.w_down"] = (E, H, L)
            shapes[f"{b}.latent_down_proj.weight"] = (L, D)
            shapes[f"{b}.latent_up_proj.weight"] = (D, L)
    return shapes


#: OLMo Core peri-norm names and their MaxText counterparts, in block order.
_NORMS = ("attention_input_norm", "attention_norm", "feed_forward_input_norm", "feed_forward_norm")
_MAXTEXT_NORMS = ("attn_in_norm", "attn_out_norm", "ffn_in_norm", "ffn_out_norm")

#: KDA projections that are plain ``nn.Linear`` weights on the OLMo Core side.
_KDA_KERNELS = ("w_q", "w_k", "w_v", "f_proj_1", "f_proj_2", "w_b", "g_proj_1", "g_proj_2", "w_out")


def maxtext_shapes(g: OLMoE3Geometry) -> Dict[str, Tuple[int, ...]]:
    """Unscanned MaxText parameter shapes, keyed by ``/``-joined path."""
    return {k: v.shape for k, v in olmo_core_to_maxtext(_zeros(olmo_core_shapes(g)), g).items()}


def _zeros(shapes: Mapping[str, Tuple[int, ...]]) -> Dict[str, np.ndarray]:
    # Zero-strided views: shape bookkeeping without allocating the model.
    return {k: np.broadcast_to(np.zeros((), np.float32), s) for k, s in shapes.items()}


_OLMO_CORE_KEY_RE = re.compile(r"^(?:model\.)?(?:module\.)?(?P<name>.+?)(?:\.main)?$")


def normalize_olmo_core_key(key: str) -> str:
    """
    Strip the prefixes and suffixes that OLMo Core checkpoints put around parameter names:
    ``model.`` (FSDP train module), ``module.`` (DDP wrapper) and ``.main`` (fp32 master copy
    in the OLMoDDP train module).
    """
    m = _OLMO_CORE_KEY_RE.match(key)
    assert m is not None
    return m.group("name")


def _check_complete(state: Mapping[str, Any], expected: Mapping[str, Tuple[int, ...]], what: str):
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    if missing or extra:
        raise KeyError(f"{what} keys don't match the geometry: missing={missing} extra={extra}")
    for k, shape in expected.items():
        if tuple(state[k].shape) != tuple(shape):
            raise ValueError(f"{what} {k}: shape {tuple(state[k].shape)} != expected {shape}")


def olmo_core_to_maxtext(
    state: Mapping[str, np.ndarray], g: OLMoE3Geometry
) -> Dict[str, np.ndarray]:
    """
    Map an unsharded OLMo Core state dict (normalized names, full shapes) to MaxText's unscanned
    parameter tree. Every input key must be consumed and every output shape is checked.
    """
    g.validate_maxtext_layout()
    _check_complete(state, olmo_core_shapes(g), "OLMo Core")
    out: Dict[str, np.ndarray] = {
        "token_embedder/embedding": state["embeddings.weight"],
        "decoder/embedding_norm/scale": state["embedding_norm.weight"],
        "decoder/decoder_norm/scale": state["lm_head.norm.weight"],
        "decoder/logits_dense/kernel": state["lm_head.w_out.weight"].T,
    }
    for i, layer in enumerate(g.layers):
        b, pre = f"blocks.{i}", f"decoder/layers_{i}"
        for norm, mt_norm in zip(_NORMS, _MAXTEXT_NORMS):
            out[f"{pre}/{mt_norm}/scale"] = state[f"{b}.{norm}.weight"]

        a, mixer = f"{b}.attention", f"{pre}/mixer"
        if layer.attention:
            H, KV, hd = g.attn_heads, g.attn_kv_heads, g.attn_head_dim
            # MaxText widens the query projection to 2 * head_dim per head and splits off the
            # second half as the elementwise output gate; OLMo Core keeps a separate w_g.
            w_q = state[f"{a}.w_q.weight"].T.reshape(-1, H, hd)
            w_g = state[f"{a}.w_g.weight"].T.reshape(-1, H, hd)
            out[f"{mixer}/attention/query/kernel"] = np.concatenate([w_q, w_g], axis=-1)
            out[f"{mixer}/attention/key/kernel"] = state[f"{a}.w_k.weight"].T.reshape(-1, KV, hd)
            out[f"{mixer}/attention/value/kernel"] = state[f"{a}.w_v.weight"].T.reshape(-1, KV, hd)
            out[f"{mixer}/attention/out/kernel"] = state[f"{a}.w_out.weight"].T
            out[f"{mixer}/attention/query_norm/scale"] = state[f"{a}.q_norm.weight"]
            out[f"{mixer}/attention/key_norm/scale"] = state[f"{a}.k_norm.weight"]
            out[f"{mixer}/attention/ssmax_scale"] = state[f"{a}.ssmax_scale"]
        else:
            for w in _KDA_KERNELS:
                out[f"{mixer}/{w}/kernel"] = state[f"{a}.{w}.weight"].T
            out[f"{mixer}/g_proj_2/bias"] = state[f"{a}.g_proj_2.bias"]
            out[f"{mixer}/A_log"] = state[f"{a}.A_log"]
            out[f"{mixer}/dt_bias"] = state[f"{a}.dt_bias"]
            out[f"{mixer}/o_norm/scale"] = state[f"{a}.o_norm.weight"]
            # Depthwise torch Conv1d weights are [C, 1, K]; both sides put the current token on
            # the last tap.
            for conv in ("q", "k", "v"):
                out[f"{mixer}/{conv}_conv"] = state[f"{a}.{conv}_conv1d.weight"][:, 0, :]

        # Shared expert: x @ w_up_gate gives [up | gate] along the columns, and the block computes
        # up * silu(gate). MaxText computes silu(wi_0) * wi_1, so wi_0 is gate and wi_1 is up.
        F = layer.shared_hidden
        up_gate = state[f"{b}.shared_experts.w_up_gate"]
        out[f"{pre}/shared_ffn/wi_0/kernel"] = up_gate[:, F:]
        out[f"{pre}/shared_ffn/wi_1/kernel"] = up_gate[:, :F]
        out[f"{pre}/shared_ffn/wo/kernel"] = state[f"{b}.shared_experts.w_down"][0]

        if layer.moe:
            E, D, H = g.num_experts, g.d_model, g.expert_hidden
            out[f"{pre}/latent_down/kernel"] = state[f"{b}.latent_down_proj.weight"].T
            out[f"{pre}/latent_up/kernel"] = state[f"{b}.latent_up_proj.weight"].T
            # The router stores its [E, D] weight flattened and computes x @ W.T.
            router = state[f"{b}.routed_experts_router.weight"].reshape(E, D)
            out[f"{pre}/moe_block/gate/kernel"] = router.T
            # Routed experts compute x @ w_up_gate[e].T with up in rows [:H] and gate in rows
            # [H:], then h @ w_down[e]. MaxText's wi_* are [E, latent, H] and wo is [E, H, latent].
            up_gate = state[f"{b}.routed_experts.w_up_gate"]
            out[f"{pre}/moe_block/wi_0"] = up_gate[:, H:, :].transpose(0, 2, 1)
            out[f"{pre}/moe_block/wi_1"] = up_gate[:, :H, :].transpose(0, 2, 1)
            out[f"{pre}/moe_block/wo"] = state[f"{b}.routed_experts.w_down"]
    return out


def maxtext_to_olmo_core(
    params: Mapping[str, np.ndarray], g: OLMoE3Geometry
) -> Dict[str, np.ndarray]:
    """
    Inverse of :func:`olmo_core_to_maxtext`: map MaxText's unscanned parameter tree to an
    unsharded OLMo Core state dict with ``named_parameters()`` names and full shapes.
    """
    g.validate_maxtext_layout()
    _check_complete(params, maxtext_shapes(g), "MaxText")
    out: Dict[str, np.ndarray] = {
        "embeddings.weight": params["token_embedder/embedding"],
        "embedding_norm.weight": params["decoder/embedding_norm/scale"],
        "lm_head.norm.weight": params["decoder/decoder_norm/scale"],
        "lm_head.w_out.weight": params["decoder/logits_dense/kernel"].T,
    }
    for i, layer in enumerate(g.layers):
        b, pre = f"blocks.{i}", f"decoder/layers_{i}"
        for norm, mt_norm in zip(_NORMS, _MAXTEXT_NORMS):
            out[f"{b}.{norm}.weight"] = params[f"{pre}/{mt_norm}/scale"]

        a, mixer = f"{b}.attention", f"{pre}/mixer"
        if layer.attention:
            hd = g.attn_head_dim
            query = params[f"{mixer}/attention/query/kernel"]
            out[f"{a}.w_q.weight"] = query[..., :hd].reshape(g.d_model, -1).T
            out[f"{a}.w_g.weight"] = query[..., hd:].reshape(g.d_model, -1).T
            out[f"{a}.w_k.weight"] = (
                params[f"{mixer}/attention/key/kernel"].reshape(g.d_model, -1).T
            )
            out[f"{a}.w_v.weight"] = (
                params[f"{mixer}/attention/value/kernel"].reshape(g.d_model, -1).T
            )
            out[f"{a}.w_out.weight"] = params[f"{mixer}/attention/out/kernel"].T
            out[f"{a}.q_norm.weight"] = params[f"{mixer}/attention/query_norm/scale"]
            out[f"{a}.k_norm.weight"] = params[f"{mixer}/attention/key_norm/scale"]
            out[f"{a}.ssmax_scale"] = params[f"{mixer}/attention/ssmax_scale"]
        else:
            for w in _KDA_KERNELS:
                out[f"{a}.{w}.weight"] = params[f"{mixer}/{w}/kernel"].T
            out[f"{a}.g_proj_2.bias"] = params[f"{mixer}/g_proj_2/bias"]
            out[f"{a}.A_log"] = params[f"{mixer}/A_log"]
            out[f"{a}.dt_bias"] = params[f"{mixer}/dt_bias"]
            out[f"{a}.o_norm.weight"] = params[f"{mixer}/o_norm/scale"]
            for conv in ("q", "k", "v"):
                out[f"{a}.{conv}_conv1d.weight"] = params[f"{mixer}/{conv}_conv"][:, None, :]

        gate = params[f"{pre}/shared_ffn/wi_0/kernel"]
        up = params[f"{pre}/shared_ffn/wi_1/kernel"]
        out[f"{b}.shared_experts.w_up_gate"] = np.concatenate([up, gate], axis=1)
        out[f"{b}.shared_experts.w_down"] = params[f"{pre}/shared_ffn/wo/kernel"][None]

        if layer.moe:
            out[f"{b}.latent_down_proj.weight"] = params[f"{pre}/latent_down/kernel"].T
            out[f"{b}.latent_up_proj.weight"] = params[f"{pre}/latent_up/kernel"].T
            out[f"{b}.routed_experts_router.weight"] = params[
                f"{pre}/moe_block/gate/kernel"
            ].T.reshape(-1)
            gate = params[f"{pre}/moe_block/wi_0"].transpose(0, 2, 1)
            up = params[f"{pre}/moe_block/wi_1"].transpose(0, 2, 1)
            out[f"{b}.routed_experts.w_up_gate"] = np.concatenate([up, gate], axis=1)
            out[f"{b}.routed_experts.w_down"] = params[f"{pre}/moe_block/wo"]
    _check_complete(out, olmo_core_shapes(g), "OLMo Core")
    return out


_UNSCANNED_RE = re.compile(r"^decoder/layers_(?P<layer>\d+)/(?P<rest>.+)$")
_SCANNED_RE = re.compile(
    r"^decoder/(?P<group>layers_0|scanned_blocks)/layer_(?P<slot>\d+)/(?P<rest>.+)$"
)


def scan_maxtext(params: Mapping[str, np.ndarray], n_layers: int) -> Dict[str, np.ndarray]:
    """
    Convert an unscanned MaxText tree (``decoder/layers_{i}``) to the ``scan_layers=True`` layout:
    the first cycle unrolled under ``decoder/layers_0/layer_{j}``, the rest stacked under
    ``decoder/scanned_blocks/layer_{j}`` along :data:`MAXTEXT_SCAN_AXIS`.
    """
    n_cycles = n_layers // MAXTEXT_CYCLE
    out: Dict[str, np.ndarray] = {}
    per_slot: Dict[Tuple[int, str], list] = {}
    for key, value in params.items():
        m = _UNSCANNED_RE.match(key)
        if m is None:
            out[key] = value
            continue
        layer = int(m["layer"])
        cycle, slot = divmod(layer, MAXTEXT_CYCLE)
        if cycle == 0:
            out[f"decoder/layers_0/layer_{slot}/{m['rest']}"] = value
        else:
            per_slot.setdefault((slot, m["rest"]), [None] * (n_cycles - 1))[cycle - 1] = value
    for (slot, rest), values in per_slot.items():
        if any(v is None for v in values):
            raise KeyError(f"layer slot {slot} {rest} is missing from some cycles")
        out[f"decoder/scanned_blocks/layer_{slot}/{rest}"] = np.stack(
            values, axis=MAXTEXT_SCAN_AXIS
        )
    return out


def unscan_maxtext(params: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Inverse of :func:`scan_maxtext`."""
    out: Dict[str, np.ndarray] = {}
    for key, value in params.items():
        m = _SCANNED_RE.match(key)
        if m is None:
            if _UNSCANNED_RE.match(key):
                raise ValueError(f"{key} is already unscanned")
            out[key] = value
            continue
        slot = int(m["slot"])
        if m["group"] == "layers_0":
            out[f"decoder/layers_{slot}/{m['rest']}"] = value
        else:
            for c in range(value.shape[MAXTEXT_SCAN_AXIS]):
                layer = (c + 1) * MAXTEXT_CYCLE + slot
                out[f"decoder/layers_{layer}/{m['rest']}"] = np.take(
                    value, c, axis=MAXTEXT_SCAN_AXIS
                )
    return out
