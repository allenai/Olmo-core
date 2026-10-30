"""
Describe an OLMo Core transformer in MaxText's terms: which MaxText ``decoder_block`` builds the
same model, the structure of each layer, how MaxText lays the layers out when ``scan_layers=True``,
and the MaxText config overrides that size a MaxText model to match.

Everything here works from the ``model`` section of an OLMo Core ``config.json``, without building
the model, so it runs on a CPU machine without the GPU kernel packages some sequence mixers need.
"""

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from olmo_core.config import StrEnum
from olmo_core.doc_utils import beta_feature
from olmo_core.nn.transformer.config import normalize_legacy_config_dict

__all__ = [
    "MaxTextDecoderBlock",
    "ScanLayout",
    "AttentionSpec",
    "KDASpec",
    "EMOSpec",
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

    Layer ``i`` is slot ``i % cycle`` of cycle ``i // cycle``. Cycles are stacked along axis 1 of
    every parameter under ``{stacked}/{slot}{i % cycle}``. If ``unrolled`` is set, the first cycle
    is not stacked and lives under ``{unrolled}/{slot}{i % cycle}`` instead.
    """

    cycle: int
    stacked: str
    slot: str
    """Prefix of a slot's name, e.g. ``"layer_"``."""
    unrolled: Optional[str] = None


#: The MaxText model config each decoder's overrides start from. MaxText switches some behavior
#: on the model name (e.g. global vs. per-head QK norm), so this has to be a config for the same
#: decoder.
_BASE_MODEL_NAMES = {
    MaxTextDecoderBlock.olmo3: "olmo3-7b",
    MaxTextDecoderBlock.olmoe3: "olmoe3-30m",
}

#: MaxText's olmo3 decoder: three sliding-window layers, then one global layer.
_OLMO3_CYCLE = 4

#: MaxText's olmoe3 decoder hardcodes this epsilon for the KDA output norm (``_KDA_NORM_EPS``),
#: separately from ``normalization_layer_epsilon``.
_OLMOE3_KDA_NORM_EPS = 1e-5


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
    allow_neg_eigval: bool = False
    """Scale the delta gate so the state transition's eigenvalues can be negative."""


@dataclass(frozen=True)
class EMOSpec:
    """
    EMO routing: every EOS-delimited document routes within its own pool of experts.
    """

    eos_token_id: int
    """The token that ends a document."""
    min_pool: int
    max_pool: int
    """Training samples each document's pool size uniformly from ``[min_pool, max_pool]``."""
    eval_pool: int
    """The pool size outside training."""


@dataclass(frozen=True)
class MoESpec:
    num_experts: int
    hidden_size: int
    latent_dim: int
    top_k: int
    emo: Optional[EMOSpec] = None


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
    cycle: int
    """The decoder repeats its layer pattern (e.g. sliding/global attention) with this period."""
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
    def n_dense_layers(self) -> int:
        """Layers without routed experts before the first MoE layer."""
        return next((i for i, layer in enumerate(self.layers) if layer.moe), self.n_layers)

    @property
    def model_name(self) -> str:
        """The MaxText ``model_name`` that :meth:`overrides` apply to."""
        return _BASE_MODEL_NAMES[self.decoder_block]

    @property
    def scan_layout(self) -> ScanLayout:
        if self.decoder_block == MaxTextDecoderBlock.olmo3:
            return ScanLayout(self.cycle, stacked="decoder/layers", slot="layers_")
        return ScanLayout(
            self.cycle, stacked="decoder/scanned_blocks", slot="layer_", unrolled="decoder/layers_0"
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
            f"inhomogeneous_layer_cycle_interval={self.cycle}",
            "logits_via_embedding=False",
        ]
        if self.decoder_block == MaxTextDecoderBlock.olmo3:
            return tuple(common + self._olmo3_overrides())
        return tuple(common + self._olmoe3_overrides())

    def _olmo3_overrides(self) -> list:
        assert self.rope_theta is not None
        return [
            f"base_mlp_dim={self.layers[0].hidden_size}",
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
            # Embeddings are scaled by sqrt(d_model), then normalized.
            "scale_embeddings_by_sqrt_emb_dim=True",
            "use_embedding_norm=True",
            f"qk_norm_per_head={self.attention.per_head_qk_norm}",
            f"use_scalable_softmax={self.attention.scalable_softmax}",
            f"first_num_dense_layers={self.n_dense_layers}",
            f"base_mlp_dim={self.layers[0].hidden_size}",
            f"base_moe_mlp_dim={m.hidden_size}",
            f"moe_expert_input_dim={m.latent_dim}",
            f"num_experts={m.num_experts}",
            f"num_experts_per_tok={m.top_k}",
            f"gdn_num_key_heads={k.n_heads}",
            f"gdn_num_value_heads={k.n_heads}",
            f"gdn_key_head_dim={k.head_dim}",
            f"gdn_value_head_dim={k.head_v_dim}",
            f"gdn_conv_kernel_dim={k.conv_size}",
            f"kda_allow_neg_eigval={k.allow_neg_eigval}",
        ]
        if m.emo is None:
            overrides.append("emo_enabled=False")
        else:
            overrides += [
                "emo_enabled=True",
                f"emo_min_document_expert_pool={m.emo.min_pool}",
                f"emo_max_document_expert_pool={m.emo.max_pool}",
                f"emo_eval_document_expert_pool={m.emo.eval_pool}",
                # OLMo Core's EMO router splits documents on the EOS token itself. MaxText's
                # router takes document boundaries from the segment ids, so its OLMo data
                # pipeline (``dataset_type=olmo_grain``) has to number documents by the same
                # token, and only the router may see that split: attention and KDA still see
                # the whole instance, as in OLMo Core.
                f"olmo_segment_eos_token_id={m.emo.eos_token_id}",
                "olmoe3_segments_for_emo_only=True",
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
    model_config = copy.deepcopy(dict(model_config))
    normalize_legacy_config_dict(model_config)
    if model_config.get("tie_word_embeddings"):
        raise NotImplementedError("tied embeddings aren't supported yet")
    if model_config.get("n_extra_vocab"):
        raise NotImplementedError(
            "split vocabularies (n_extra_vocab > 0) aren't supported yet: the extra input-only "
            "embedding rows have no MaxText counterpart"
        )
    blocks = _blocks(model_config)
    for i, block in enumerate(blocks):
        # The overrides carry no dropout rate (MaxText defaults to none), so a model trained with
        # dropout would silently continue training without it.
        dropout = {
            k: v
            for k, v in (
                ("dropout", block.get("dropout")),
                ("masked_dropout", block.get("masked_dropout")),
                ("sequence_mixer.dropout", block["sequence_mixer"].get("dropout")),
            )
            if v
        }
        if dropout:
            raise ValueError(
                f"block {i}: the MaxText decoders are built without dropout: {dropout}"
            )
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
    # Keys are ints from ``as_config_dict()`` and strings from JSON.
    overrides = {int(i): b for i, b in (model_config.get("block_overrides") or {}).items()}
    return [overrides.get(i, model_config["block"]) for i in range(model_config["n_layers"])]


def _same(prev: Optional[Dict[str, Any]], cfg: Mapping[str, Any], keys: Tuple[str, ...], what: str):
    if prev is not None and any(prev.get(k) != cfg.get(k) for k in keys):
        raise ValueError(f"all {what} blocks must share {keys}")
    return dict(cfg)


def _head_dim(d_model: int, mixer: Mapping[str, Any]) -> int:
    return mixer.get("head_dim") or d_model // mixer["n_heads"]


def _norm_eps(norms: list) -> float:
    """
    The one epsilon MaxText applies to every norm, after checking that each of ``norms`` is a
    plain RMS norm with a gain and no bias. ``as_config_dict()`` drops unset fields, so the
    RMSNorm constructor's defaults fill them in here.
    """
    eps = set()
    for n in norms:
        if n is None:
            continue
        if n.get("name", "default") != "rms":
            raise ValueError(f"MaxText's decoders use RMS norms, not {n.get('name', 'default')!r}")
        if not n.get("elementwise_affine", True):
            raise ValueError("MaxText's decoders' norms have a gain (elementwise_affine=True)")
        if n.get("bias", True):
            raise ValueError("MaxText's decoders' norms have no bias")
        eps.add(n.get("eps", 1e-5))
    if len(eps) != 1:
        raise ValueError(f"MaxText uses one epsilon for every norm, but this model has {eps}")
    return eps.pop()


def _lm_head_norm(model_config: Mapping[str, Any], decoder: MaxTextDecoderBlock) -> Dict[str, Any]:
    """
    The LM head's norm config. Both decoders normalize the final hidden state with a bias-free
    RMS norm (``decoder_norm``) and have no LM head bias.
    """
    lm_head = model_config.get("lm_head") or {}
    # ``as_config_dict()`` drops None fields, and the default LM head then builds with a bias.
    if lm_head.get("name", "default") != "default" or lm_head.get("bias", True):
        raise ValueError(f"MaxText's {decoder} decoder has a default LM head without bias")
    if lm_head.get("layer_norm") is None:
        raise ValueError(f"MaxText's {decoder} decoder normalizes the LM head input (layer_norm)")
    return lm_head["layer_norm"]


def _check_router(i: int, router: Mapping[str, Any]) -> None:
    """
    MaxText's olmoe3 router has exactly one weighting: pick the top-k experts by logit, softmax
    over their logits, and multiply by ``top_k``. That's OLMo Core's softmax gating with
    ``normalize_expert_weights=1.0`` and ``restore_weight_scale=True`` (the L1-normalized gather
    of the full softmax equals the softmax over the top-k logits), or ``topk_softmax`` gating
    with ``restore_weight_scale=True``.
    """
    gating = router.get("gating_function", "softmax")
    normalize = router.get("normalize_expert_weights")
    weighting = {
        "gating_function": gating,
        "normalize_expert_weights": normalize,
        "restore_weight_scale": bool(router.get("restore_weight_scale")),
    }
    if (gating, normalize) not in (("softmax", 1.0), ("topk_softmax", None), ("topk_softmax", 1.0)):
        raise ValueError(
            f"block {i}: MaxText's olmoe3 router weights experts by top_k * softmax(top-k logits), "
            f"i.e. gating_function='softmax' with normalize_expert_weights=1.0, not {weighting}"
        )
    if not router.get("restore_weight_scale"):
        raise ValueError(
            f"block {i}: MaxText's olmoe3 router scales expert weights by top_k "
            f"(restore_weight_scale=True), not {weighting}"
        )
    if router.get("expert_weight_scale") is not None or router.get("original_top_k") not in (
        None,
        router["top_k"],
    ):
        raise ValueError(
            f"block {i}: MaxText's olmoe3 router can't rescale expert weights "
            "(expert_weight_scale, original_top_k)"
        )
    if (
        router.get("bias")
        or router.get("bias_gamma") is not None
        or router.get("score_correction_bias")
    ):
        raise ValueError(
            f"block {i}: MaxText's olmoe3 router has no bias and no score-correction bias "
            "(bias, bias_gamma, score_correction_bias)"
        )
    if (
        router.get("uniform_expert_assignment")
        or router.get("random_expert_assignment")
        or router.get("n_group") is not None
        or router.get("topk_group") is not None
    ):
        raise ValueError(
            f"block {i}: MaxText's olmoe3 router picks the top-k experts by score, without expert "
            "groups or forced assignment"
        )


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
    alphas = {
        k: blocks[0].get(k)
        for k in ("attention_residual_alpha", "feed_forward_residual_alpha")
        if blocks[0].get(k) not in (None, 1.0)
    }
    if alphas:
        raise ValueError(f"MaxText's olmo3 decoder doesn't scale the residual branches: {alphas}")
    # ``as_config_dict()`` drops None fields, and both modules then build with bias=True.
    if mixer.get("bias", True) or ff.get("bias", True):
        raise ValueError("MaxText's olmo3 decoder has no projection biases")
    if mixer.get("clip_qkv") is not None:
        raise ValueError("MaxText's olmo3 decoder doesn't clip Q/K/V (clip_qkv)")
    if mixer.get("use_head_qk_norm") or mixer.get("gate") or mixer.get("scalable_softmax"):
        raise ValueError("MaxText's olmo3 decoder has global QK norm, no output gate, and no SSMax")
    if ff.get("activation", "silu") != "silu" or ff.get("name", "default") != "default":
        raise ValueError("MaxText's olmo3 decoder uses a SwiGLU feed-forward")
    if (
        model_config.get("embedding_norm") is not None
        or model_config.get("embed_scale") is not None
    ):
        raise ValueError("MaxText's olmo3 decoder has no embedding norm or scale")
    lm_head_norm = _lm_head_norm(model_config, MaxTextDecoderBlock.olmo3)

    rope = mixer.get("rope")
    if rope is None or rope.get("name", "default") != "default" or rope.get("scaling") is not None:
        raise NotImplementedError("only unscaled default RoPE is supported for the olmo3 decoder")
    if rope.get("partial_rotary_factor", 1.0) != 1.0:
        raise NotImplementedError("partial RoPE isn't supported")
    if rope.get("no_global_rope"):
        raise ValueError(
            "MaxText's olmo3 decoder applies RoPE on every layer, so it can't express "
            "no_global_rope=True"
        )

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
    eps = _norm_eps([blocks[0]["layer_norm"], qk_norm, lm_head_norm])
    return MaxTextModelConfig(
        decoder_block=MaxTextDecoderBlock.olmo3,
        d_model=d_model,
        vocab_size=model_config["vocab_size"],
        layers=tuple(
            LayerSpec(sequence_mixer="attention", hidden_size=ff["hidden_size"])
            for _ in range(n_layers)
        ),
        cycle=_OLMO3_CYCLE,
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
            # Resolve the optional head settings first, so blocks that spell the same model
            # differently (unset vs. explicit) compare equal.
            mixer = {
                **mixer,
                "n_kv_heads": mixer.get("n_kv_heads") or mixer["n_heads"],
                "head_dim": _head_dim(d_model, mixer),
            }
            attn = _same(attn, mixer, ("n_heads", "n_kv_heads", "head_dim"), "attention")
            # ``as_config_dict()`` drops None fields, and attention then builds with bias=True.
            if mixer.get("bias", True):
                raise ValueError(f"block {i}: MaxText's olmoe3 attention has no projection bias")
            if mixer.get("clip_qkv") is not None:
                raise ValueError(
                    f"block {i}: MaxText's olmoe3 attention doesn't clip Q/K/V (clip_qkv)"
                )
            if mixer.get("qk_norm") is None:
                raise ValueError(f"block {i}: MaxText's olmoe3 decoder needs a QK norm")
            if not mixer.get("use_head_qk_norm") or not mixer.get("qk_norm_per_head_gains"):
                raise ValueError(
                    f"block {i}: MaxText's olmoe3 decoder needs per-head QK norm with gains"
                )
            norms.append(mixer["qk_norm"])
            if not mixer.get("scalable_softmax") or mixer.get("rope") is not None:
                raise ValueError(
                    f"block {i}: MaxText's olmoe3 decoder needs NoPE + scalable softmax"
                )
            if (mixer.get("gate") or {}).get("granularity") != "elementwise":
                raise ValueError(
                    f"block {i}: MaxText's olmoe3 decoder needs an elementwise output gate"
                )
        elif mixer.get("type") == "kimi_delta_attention":
            mixer = {
                **mixer,
                "n_v_heads": mixer.get("n_v_heads") or mixer["n_heads"],
                "head_dim": _head_dim(d_model, mixer),
            }
            kda = _same(
                kda,
                mixer,
                ("n_heads", "n_v_heads", "head_dim", "expand_v", "conv_size", "allow_neg_eigval"),
                "KDA",
            )
            if mixer.get("conv_bias"):
                raise ValueError(f"block {i}: MaxText's olmoe3 KDA has no conv bias")
            # OLMo Core's default; the config dict always carries it since it isn't None.
            if mixer.get("norm_eps", 1e-5) != _OLMOE3_KDA_NORM_EPS:
                raise ValueError(
                    f"block {i}: MaxText's olmoe3 KDA output norm has a fixed epsilon of "
                    f"{_OLMOE3_KDA_NORM_EPS}, not norm_eps={mixer.get('norm_eps')}"
                )
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
            _check_router(i, block["routed_experts_router"])
            if block["routed_experts"].get("bias") or block["latent_moe"].get("bias"):
                raise ValueError(f"block {i}: MaxText's olmoe3 experts have no bias")
            if block["latent_moe"].get("up_proj_input_norm_enabled"):
                raise ValueError(
                    f"block {i}: MaxText's olmoe3 decoder doesn't normalize the latent up "
                    "projection's input (up_proj_input_norm_enabled)"
                )
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
    # MaxText scales embeddings by sqrt(d_model) before this norm. The norm's epsilon keeps a
    # different scale from cancelling out, so OLMo Core's embed_scale has to match.
    if model_config.get("embedding_norm") is None:
        raise ValueError("MaxText's olmoe3 decoder normalizes the embeddings")
    embed_scale = model_config.get("embed_scale")
    if embed_scale is None or not math.isclose(embed_scale, math.sqrt(d_model), rel_tol=1e-6):
        raise ValueError(
            f"MaxText's olmoe3 decoder scales embeddings by sqrt(d_model) = {math.sqrt(d_model):g} "
            f"before the embedding norm, so it can't express embed_scale={embed_scale}"
        )
    lm_head_norm = _lm_head_norm(model_config, MaxTextDecoderBlock.olmoe3)
    emo = router.get("emo")
    kda_head_dim = _head_dim(d_model, kda)
    config = MaxTextModelConfig(
        decoder_block=MaxTextDecoderBlock.olmoe3,
        d_model=d_model,
        vocab_size=model_config["vocab_size"],
        layers=tuple(layers),
        # The decoder puts full attention on the last layer of every cycle.
        cycle=next(i for i, layer in enumerate(layers) if layer.sequence_mixer == "attention") + 1,
        norm_eps=_norm_eps(norms + [model_config["embedding_norm"], lm_head_norm]),
        attention=AttentionSpec(
            n_heads=attn["n_heads"],
            n_kv_heads=attn.get("n_kv_heads") or attn["n_heads"],
            head_dim=_head_dim(d_model, attn),
            gated=True,
            per_head_qk_norm=True,
            scalable_softmax=True,
        ),
        kda=KDASpec(
            n_heads=kda["n_heads"],
            head_dim=kda_head_dim,
            head_v_dim=int(kda_head_dim * kda["expand_v"]),
            conv_size=kda["conv_size"],
            allow_neg_eigval=bool(kda.get("allow_neg_eigval")),
        ),
        moe=MoESpec(
            num_experts=routed["num_experts"],
            hidden_size=routed["hidden_size"],
            latent_dim=latent["latent_dim"],
            top_k=router["top_k"],
            emo=None if emo is None else _emo_spec(emo),
        ),
    )
    _validate_hybrid_layout(config)
    return config


def _emo_spec(emo: Mapping[str, Any]) -> EMOSpec:
    lo, hi = emo["min_document_expert_pool"], emo["max_document_expert_pool"]
    # The eval pool is optional; ``EmoRouterConfig.eval_pool_size()`` falls back to the midpoint.
    eval_pool = emo.get("eval_document_expert_pool")
    return EMOSpec(
        eos_token_id=emo["eos_token_id"],
        min_pool=lo,
        max_pool=hi,
        eval_pool=(lo + hi) // 2 if eval_pool is None else eval_pool,
    )


def _validate_hybrid_layout(config: MaxTextModelConfig) -> None:
    """
    The layer pattern MaxText's olmoe3 decoder builds: dense layers within the first cycle, then
    MoE everywhere, with full attention on the last layer of every cycle.
    """
    cycle, n_dense = config.cycle, config.n_dense_layers
    assert config.moe is not None
    if config.n_layers % cycle:
        raise ValueError(f"n_layers={config.n_layers} must be a multiple of the cycle ({cycle})")
    if not 0 < n_dense <= cycle:
        raise ValueError(
            f"MaxText's olmoe3 decoder needs 1 to {cycle} leading dense layers, not {n_dense}"
        )
    for i, layer in enumerate(config.layers):
        want_attention = (i + 1) % cycle == 0
        if (layer.sequence_mixer == "attention") != want_attention:
            raise ValueError(
                f"layer {i}: MaxText's olmoe3 decoder puts full attention on layers where "
                f"(i + 1) % {cycle} == 0"
            )
        if layer.moe != (i >= n_dense):
            raise ValueError(
                f"layer {i}: MaxText's olmoe3 decoder has routed experts on every layer after "
                f"the first {n_dense}"
            )
        if layer.moe and layer.hidden_size != config.moe.hidden_size:
            raise ValueError(
                f"layer {i}: shared expert hidden {layer.hidden_size} != routed expert hidden "
                f"{config.moe.hidden_size} (MaxText's shared_expert_mlp_dim default)"
            )
