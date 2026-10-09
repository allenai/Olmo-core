"""
Describe an OLMo Core transformer in MaxText's terms: which MaxText ``decoder_block`` builds the
same model, the structure of each layer, how MaxText lays the layers out when ``scan_layers=True``,
and the MaxText config overrides that size a MaxText model to match.

Everything here works from the ``model`` section of an OLMo Core ``config.json``, without building
the model, so it runs on a CPU machine without the GPU kernel packages some sequence mixers need.
"""

import copy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from olmo_core.config import StrEnum
from olmo_core.doc_utils import beta_feature

__all__ = [
    "MaxTextDecoderBlock",
    "ScanLayout",
    "AttentionSpec",
    "KDASpec",
    "MoESpec",
    "LayerSpec",
    "MaxTextModelConfig",
    "get_maxtext_config",
]


class MaxTextDecoderBlock(StrEnum):
    """
    The MaxText decoders that OLMo Core models can be converted to, named by MaxText's
    ``decoder_block`` setting.
    """

    olmo3 = "olmo3"
    """
    Dense attention + SwiGLU with norms on the sublayer outputs (OLMo Core's ``reordered_norm``
    block), RoPE, global QK norm, and an optional 3:1 sliding-window pattern.
    """

    olmoe3 = "olmoe3"
    """
    Kimi Delta Attention with one gated, NoPE full-attention layer per cycle, peri-norm blocks,
    a dense first layer, and latent routed experts with a shared expert everywhere else.
    """


@dataclass(frozen=True)
class ScanLayout:
    """
    Where MaxText puts layer parameters when ``scan_layers=True``.

    Layer ``i`` is slot ``i % cycle`` of cycle ``i // cycle``. Cycles are stacked along ``axis`` of
    every parameter under ``{stacked}/{slot}``. If ``unrolled`` is set, the first cycle is not
    stacked and lives under ``{unrolled}/{slot}`` instead.
    """

    cycle: int
    stacked: str
    slot: str
    """Format string for a slot's name, e.g. ``"layer_{}"``."""
    unrolled: Optional[str] = None
    axis: int = 1


_SCAN_LAYOUTS = {
    MaxTextDecoderBlock.olmo3: dict(stacked="decoder/layers", slot="layers_{}"),
    MaxTextDecoderBlock.olmoe3: dict(
        stacked="decoder/scanned_blocks", slot="layer_{}", unrolled="decoder/layers_0"
    ),
}

#: The MaxText model config each decoder's overrides start from. MaxText switches some behavior
#: on the model name (e.g. global vs. per-head QK norm), so this has to be a config for the same
#: decoder.
_BASE_MODEL_NAMES = {
    MaxTextDecoderBlock.olmo3: "olmo3-7b",
    MaxTextDecoderBlock.olmoe3: "olmoe3-30m",
}

#: MaxText's olmo3 decoder: three sliding-window layers, then one global layer.
_OLMO3_CYCLE = 4


@dataclass(frozen=True)
class AttentionSpec:
    n_heads: int
    n_kv_heads: int
    head_dim: int
    gated: bool = False
    """An elementwise sigmoid output gate (``w_g``)."""
    per_head_qk_norm: bool = False
    """QK norm per head with ``[heads, head_dim]`` gains, rather than over the whole projection."""
    qk_norm: bool = True
    scalable_softmax: bool = False


@dataclass(frozen=True)
class KDASpec:
    n_heads: int
    head_dim: int
    head_v_dim: int
    conv_size: int


@dataclass(frozen=True)
class MoESpec:
    num_experts: int
    hidden_size: int
    latent_dim: int
    top_k: int
    emo_pools: Optional[Tuple[int, int, int]] = None
    """EMO document expert pool sizes (min, max, eval), if EMO routing is on."""


@dataclass(frozen=True)
class LayerSpec:
    sequence_mixer: str
    """``"attention"`` or ``"kimi_delta_attention"``."""
    hidden_size: int
    """Hidden size of the dense feed-forward or the shared expert."""
    moe: bool = False
    """Routed experts alongside the shared expert."""


@beta_feature
@dataclass(frozen=True)
class MaxTextModelConfig:
    """
    An OLMo Core transformer described in MaxText's terms. Build it with :func:`get_maxtext_config`.
    """

    decoder_block: MaxTextDecoderBlock
    d_model: int
    vocab_size: int
    layers: Tuple[LayerSpec, ...]
    norm_eps: float
    attention: AttentionSpec
    kda: Optional[KDASpec] = None
    moe: Optional[MoESpec] = None
    rope_theta: Optional[float] = None
    sliding_window: Optional[int] = None

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def model_name(self) -> str:
        """The MaxText ``model_name`` that :meth:`overrides` apply to."""
        return _BASE_MODEL_NAMES[self.decoder_block]

    @property
    def scan_layout(self) -> ScanLayout:
        cycle = _OLMO3_CYCLE if self.decoder_block == MaxTextDecoderBlock.olmo3 else self._cycle
        return ScanLayout(cycle=cycle, **_SCAN_LAYOUTS[self.decoder_block])  # type: ignore[arg-type]

    @property
    def _cycle(self) -> int:
        # The hybrid decoder puts full attention on the last layer of every cycle.
        return (
            next(i for i, layer in enumerate(self.layers) if layer.sequence_mixer == "attention")
            + 1
        )

    def overrides(self) -> Tuple[str, ...]:
        """
        MaxText ``key=value`` overrides that size :attr:`model_name` (with
        ``override_model_config=True``) to this model, so converted weights load into it.
        """
        a = self.attention
        common = [
            f"base_emb_dim={self.d_model}",
            f"vocab_size={self.vocab_size}",
            f"base_num_decoder_layers={self.n_layers}",
            f"base_num_query_heads={a.n_heads}",
            f"base_num_kv_heads={a.n_kv_heads}",
            f"head_dim={a.head_dim}",
            f"normalization_layer_epsilon={self.norm_eps}",
            f"use_qk_norm={a.qk_norm}",
            "logits_via_embedding=False",
        ]
        if self.decoder_block == MaxTextDecoderBlock.olmo3:
            return tuple(common + self._olmo3_overrides())
        return tuple(common + self._olmoe3_overrides())

    def _olmo3_overrides(self) -> list:
        assert self.rope_theta is not None
        return [
            f"base_mlp_dim={self.layers[0].hidden_size}",
            f"inhomogeneous_layer_cycle_interval={_OLMO3_CYCLE}",
            # Without a sliding window every layer is global; a window as long as MaxText allows
            # makes MaxText's sliding layers global too.
            f"sliding_window_size={self.sliding_window or 2**30}",
            "rope_type=default",
            f"rope_max_timescale={self.rope_theta:g}",
            "rope_interleave=False",
        ]

    def _olmoe3_overrides(self) -> list:
        assert self.kda is not None and self.moe is not None
        k, m = self.kda, self.moe
        overrides = [
            # Embeddings are scaled, then normalized.
            "scale_embeddings_by_sqrt_emb_dim=True",
            "use_embedding_norm=True",
            f"qk_norm_per_head={self.attention.per_head_qk_norm}",
            f"use_scalable_softmax={self.attention.scalable_softmax}",
            "first_num_dense_layers=1",
            f"base_mlp_dim={self.layers[0].hidden_size}",
            f"base_moe_mlp_dim={m.hidden_size}",
            f"moe_expert_input_dim={m.latent_dim}",
            f"num_experts={m.num_experts}",
            f"num_experts_per_tok={m.top_k}",
            f"inhomogeneous_layer_cycle_interval={self._cycle}",
            f"gdn_num_key_heads={k.n_heads}",
            f"gdn_num_value_heads={k.n_heads}",
            f"gdn_key_head_dim={k.head_dim}",
            f"gdn_value_head_dim={k.head_v_dim}",
            f"gdn_conv_kernel_dim={k.conv_size}",
        ]
        if m.emo_pools is None:
            overrides.append("emo_enabled=False")
        else:
            lo, hi, ev = m.emo_pools
            overrides += [
                "emo_enabled=True",
                f"emo_min_document_expert_pool={lo}",
                f"emo_max_document_expert_pool={hi}",
                f"emo_eval_document_expert_pool={ev}",
            ]
        return overrides


@beta_feature
def get_maxtext_config(model_config: Mapping[str, Any]) -> MaxTextModelConfig:
    """
    Describe an OLMo Core model in MaxText's terms, and check that a MaxText decoder can build it.

    :param model_config: The ``model`` section of an OLMo Core ``config.json``, i.e.
        ``TransformerConfig.as_config_dict()``.

    :raises NotImplementedError: If no supported MaxText decoder builds this model.
    :raises ValueError: If the model is close to a supported decoder but differs in a way the
        decoder can't express (e.g. the sliding-window pattern).
    """
    # Imported here: olmo_core.nn.hf pulls in transformers.
    from olmo_core.nn.hf.convert_checkpoint import _normalize_legacy_latent_moe_config

    model_config = copy.deepcopy(dict(model_config))
    _normalize_legacy_latent_moe_config(model_config)
    if model_config.get("tie_word_embeddings"):
        raise NotImplementedError("tied embeddings aren't supported yet")
    blocks = _blocks(model_config)
    first = blocks[0]
    if first.get("use_peri_norm") and first.get("shared_experts") is not None:
        return _hybrid_moe_config(model_config, blocks)
    if first.get("name") == "reordered_norm" and first.get("feed_forward") is not None:
        return _dense_config(model_config, blocks)
    raise NotImplementedError(
        f"no MaxText decoder for blocks of type {first.get('name')!r}; supported: "
        f"{', '.join(_BASE_MODEL_NAMES)}"
    )


def _blocks(model_config: Mapping[str, Any]) -> list:
    if (
        not isinstance(model_config["block"], Mapping)
        or "sequence_mixer" not in model_config["block"]
    ):
        raise NotImplementedError("named block configs aren't supported")
    overrides = model_config.get("block_overrides") or {}
    return [
        overrides.get(str(i), overrides.get(i, model_config["block"]))
        for i in range(model_config["n_layers"])
    ]


def _same(prev: Optional[Dict[str, Any]], cfg: Mapping[str, Any], keys: Tuple[str, ...], what: str):
    if prev is not None and any(prev.get(k) != cfg.get(k) for k in keys):
        raise ValueError(f"all {what} blocks must share {keys}")
    return dict(cfg)


def _head_dim(d_model: int, mixer: Mapping[str, Any]) -> int:
    return mixer.get("head_dim") or d_model // mixer["n_heads"]


def _norm_eps(norms: list) -> float:
    eps = {n["eps"] for n in norms if n is not None}
    if len(eps) != 1:
        raise ValueError(f"MaxText uses one epsilon for every norm, but this model has {eps}")
    if any(n is not None and (n.get("name", "rms") != "rms" or n.get("bias")) for n in norms):
        raise ValueError("MaxText's decoders use bias-free RMS norms")
    return eps.pop()


def _dense_config(model_config: Mapping[str, Any], blocks: list) -> MaxTextModelConfig:
    d_model = model_config["d_model"]
    mixer = blocks[0]["sequence_mixer"]
    ff = blocks[0]["feed_forward"]
    for i, block in enumerate(blocks):
        if block != blocks[0]:
            raise ValueError(
                f"block {i}: MaxText's {MaxTextDecoderBlock.olmo3} decoder needs identical blocks"
            )
    if mixer.get("type", "attention") != "attention" or mixer.get("name", "default") != "default":
        raise ValueError(f"unsupported sequence mixer {mixer.get('type')!r}/{mixer.get('name')!r}")
    if mixer.get("bias") or ff.get("bias"):
        raise ValueError("MaxText's olmo3 decoder has no projection biases")
    if mixer.get("use_head_qk_norm") or mixer.get("gate") or mixer.get("scalable_softmax"):
        raise ValueError("MaxText's olmo3 decoder has global QK norm, no output gate, and no SSMax")
    if ff.get("activation", "silu") != "silu" or ff.get("name", "default") != "default":
        raise ValueError("MaxText's olmo3 decoder uses a SwiGLU feed-forward")
    if model_config.get("embedding_norm") is not None or model_config.get("embed_scale"):
        raise ValueError("MaxText's olmo3 decoder has no embedding norm or scale")
    if (model_config.get("lm_head") or {}).get("bias"):
        raise ValueError("MaxText's olmo3 decoder has no LM head bias")

    rope = mixer.get("rope")
    if rope is None or rope.get("name", "default") != "default" or rope.get("scaling") is not None:
        raise NotImplementedError("only unscaled default RoPE is supported for the olmo3 decoder")
    if rope.get("partial_rotary_factor", 1.0) != 1.0:
        raise NotImplementedError("partial RoPE isn't supported")

    n_layers = len(blocks)
    window = None
    swa = mixer.get("sliding_window")
    if swa is not None:
        pattern = list(swa["pattern"])
        windows = set(pattern[:-1])
        if len(pattern) != _OLMO3_CYCLE or pattern[-1] != -1 or len(windows) != 1 or -1 in windows:
            raise ValueError(
                f"MaxText's olmo3 decoder has a [w, w, w, -1] sliding-window pattern, not {pattern}"
            )
        if swa.get("force_full_attention_on_first_layer"):
            raise ValueError(
                "MaxText's olmo3 decoder can't force full attention on the first layer"
            )
        if swa.get("force_full_attention_on_last_layer") and (n_layers - 1) % _OLMO3_CYCLE != 3:
            raise ValueError("MaxText's olmo3 decoder can't force full attention on the last layer")
        # Both count the current token: a query sees at most `window` positions.
        window = windows.pop()

    qk_norm = mixer.get("qk_norm")
    lm_head_norm = (model_config.get("lm_head") or {}).get("layer_norm")
    eps = _norm_eps([blocks[0]["layer_norm"], qk_norm, lm_head_norm])
    return MaxTextModelConfig(
        decoder_block=MaxTextDecoderBlock.olmo3,
        d_model=d_model,
        vocab_size=model_config["vocab_size"],
        layers=tuple(
            LayerSpec(sequence_mixer="attention", hidden_size=ff["hidden_size"])
            for _ in range(n_layers)
        ),
        norm_eps=eps,
        attention=AttentionSpec(
            n_heads=mixer["n_heads"],
            n_kv_heads=mixer.get("n_kv_heads") or mixer["n_heads"],
            head_dim=_head_dim(d_model, mixer),
            qk_norm=qk_norm is not None,
        ),
        rope_theta=rope.get("theta", 500_000),
        sliding_window=window,
    )


def _hybrid_moe_config(model_config: Mapping[str, Any], blocks: list) -> MaxTextModelConfig:
    d_model = model_config["d_model"]
    kda: Optional[Dict[str, Any]] = None
    attn: Optional[Dict[str, Any]] = None
    routed: Optional[Dict[str, Any]] = None
    router: Optional[Dict[str, Any]] = None
    latent: Optional[Dict[str, Any]] = None
    layers = []
    norms = []
    for i, block in enumerate(blocks):
        mixer = block["sequence_mixer"]
        if mixer.get("type") == "attention":
            attn = _same(attn, mixer, ("n_heads", "n_kv_heads", "head_dim"), "attention")
            if not mixer.get("use_head_qk_norm") or not mixer.get("qk_norm_per_head_gains"):
                raise ValueError(
                    f"block {i}: MaxText's olmoe3 decoder needs per-head QK norm with gains"
                )
            if not mixer.get("scalable_softmax") or mixer.get("rope") is not None:
                raise ValueError(
                    f"block {i}: MaxText's olmoe3 decoder needs NoPE + scalable softmax"
                )
            if (mixer.get("gate") or {}).get("granularity") != "elementwise":
                raise ValueError(
                    f"block {i}: MaxText's olmoe3 decoder needs an elementwise output gate"
                )
        elif mixer.get("type") == "kimi_delta_attention":
            kda = _same(
                kda, mixer, ("n_heads", "n_v_heads", "head_dim", "expand_v", "conv_size"), "KDA"
            )
            if mixer.get("conv_bias"):
                raise ValueError(f"block {i}: MaxText's olmoe3 KDA has no conv bias")
        else:
            raise ValueError(f"block {i}: unsupported sequence mixer {mixer.get('type')!r}")

        shared = block["shared_experts"]
        if shared["num_experts"] != 1 or shared.get("bias"):
            raise ValueError(f"block {i}: expected a single bias-free shared expert")
        is_moe = block.get("routed_experts") is not None
        if is_moe:
            routed = _same(routed, block["routed_experts"], ("num_experts", "hidden_size"), "MoE")
            latent = _same(latent, block["latent_moe"], ("latent_dim",), "latent MoE")
            router = _same(router, block["routed_experts_router"], ("top_k", "emo"), "MoE router")
            if block["routed_experts"].get("bias") or block["latent_moe"].get("bias"):
                raise ValueError(f"block {i}: MaxText's olmoe3 experts have no bias")
            if block["routed_experts"].get("activation", "swiglu") != "swiglu":
                raise ValueError(f"block {i}: MaxText's olmoe3 experts use swiglu")
        if not block.get("use_peri_norm"):
            raise ValueError(f"block {i}: MaxText's olmoe3 decoder uses peri-norm blocks")
        norms.append(block["layer_norm"])
        layers.append(
            LayerSpec(sequence_mixer=mixer["type"], hidden_size=shared["hidden_size"], moe=is_moe)
        )

    if kda is None or attn is None or routed is None or latent is None or router is None:
        raise ValueError("MaxText's olmoe3 decoder needs KDA, full-attention, and MoE blocks")
    if kda["n_v_heads"] != kda["n_heads"]:
        raise ValueError("MaxText's olmoe3 KDA needs n_v_heads == n_heads")
    # MaxText scales embeddings by sqrt(d_model) before this norm; the RMS norm undoes any scale,
    # so OLMo Core's embed_scale doesn't have to match.
    if model_config.get("embedding_norm") is None:
        raise ValueError("MaxText's olmoe3 decoder normalizes the embeddings")
    emo = router.get("emo")
    config = MaxTextModelConfig(
        decoder_block=MaxTextDecoderBlock.olmoe3,
        d_model=d_model,
        vocab_size=model_config["vocab_size"],
        layers=tuple(layers),
        norm_eps=_norm_eps(norms + [model_config["embedding_norm"]]),
        attention=AttentionSpec(
            n_heads=attn["n_heads"],
            n_kv_heads=attn["n_kv_heads"],
            head_dim=attn["head_dim"],
            gated=True,
            per_head_qk_norm=True,
            scalable_softmax=True,
        ),
        kda=KDASpec(
            n_heads=kda["n_heads"],
            head_dim=kda["head_dim"],
            head_v_dim=int(kda["head_dim"] * kda["expand_v"]),
            conv_size=kda["conv_size"],
        ),
        moe=MoESpec(
            num_experts=routed["num_experts"],
            hidden_size=routed["hidden_size"],
            latent_dim=latent["latent_dim"],
            top_k=router["top_k"],
            emo_pools=(
                None
                if emo is None
                else (
                    emo["min_document_expert_pool"],
                    emo["max_document_expert_pool"],
                    emo["eval_document_expert_pool"],
                )
            ),
        ),
    )
    _validate_hybrid_layout(config)
    return config


def _validate_hybrid_layout(config: MaxTextModelConfig) -> None:
    """
    The layer pattern MaxText's olmoe3 decoder builds: a dense first layer then MoE everywhere,
    with full attention on the last layer of every cycle.
    """
    cycle = config._cycle
    assert config.moe is not None
    if config.n_layers % cycle:
        raise ValueError(f"n_layers={config.n_layers} must be a multiple of the cycle ({cycle})")
    for i, layer in enumerate(config.layers):
        want_attention = (i + 1) % cycle == 0
        if (layer.sequence_mixer == "attention") != want_attention:
            raise ValueError(
                f"layer {i}: MaxText's olmoe3 decoder puts full attention on layers where "
                f"(i + 1) % {cycle} == 0"
            )
        if layer.moe != (i > 0):
            raise ValueError(
                f"layer {i}: MaxText's olmoe3 decoder has exactly one dense layer, layer 0"
            )
        if layer.moe and layer.hidden_size != config.moe.hidden_size:
            raise ValueError(
                f"layer {i}: shared expert hidden {layer.hidden_size} != routed expert hidden "
                f"{config.moe.hidden_size} (MaxText's shared_expert_mlp_dim default)"
            )
