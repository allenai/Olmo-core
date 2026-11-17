"""
Research-arm variants of the training recipes, in ONE place.

Every research arm (document-chunked masks, sparse landmark, pooled-doc KV, the nested-FFN /
KV / block-skip routers, pooled soft tokens, FFN token drop) is a :class:`Variant` registered in
:data:`VARIANTS`. A variant says exactly what it changes relative to the dense recipe, through hooks
that both trainers call:

* ``recipes/train/research_trainer/train_ctc_suite.py`` (torchrun + argparse; the research
  launchers under ``research/*``) builds the variant's options from its argparse namespace
  (:meth:`Variant.options_from_namespace` -- the option fields ARE the argparse dest names).
* ``recipes/train/setA/`` (the canonical mixed-task SFT recipe) gets them from
  ``launch_setA_sft.py --variant <name> --variant-opts k=v,...`` via the env vars
  ``SETA_VARIANT`` / ``SETA_VARIANT_OPTS`` (:func:`options_from_kv`).

Hook map (all optional; the base class is the dense no-op):

==========================  =================================================================
``factory_kwargs``          extra kwargs for the model factory (chunked: ``document_chunked``)
``apply_model_config``      mutate the :class:`TransformerConfig` (mixer swap,
                            ``document_chunk_attention``) -- idempotent w.r.t. factory_kwargs
``train_module_overrides``  fields of the train-module config (``microbatch_sort_pad_id``)
``optim_group_overrides``   extra :class:`OptimGroupOverride` (router LR groups)
``data_overrides``          replacement instance source (sparse landmark's landmark packing)
``post_build_model``        ``enable_*`` calls on the built (meta-device) model, ``budget_attach``
``post_build_train_module`` forward pre-hooks that need the wrapped model (gold keep sets)
``extra_callbacks``         trainer callbacks (router / drop schedule callbacks)
``export_config``           the eval-side record written into the exported ``config.json``
==========================  =================================================================

Variants with ``new_params=True`` add parameters the base lacks, so the base must be loaded with
:func:`tolerant_base_load` instead of the trainer's strict load.

This module is stdlib-only at import time (olmo_core is imported inside the hooks), so the setA
launcher can validate ``--variant`` / ``--variant-opts`` on a login node without torch.
"""

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type

#: Sparse-landmark block = 64 (63 content + 1 landmark), matches every landmark lineage.
MEM_FREQ_SPARSE = 63



# ----------------------------------------------------------------------------------------------
# Context + options
# ----------------------------------------------------------------------------------------------


@dataclass
class VariantContext:
    """
    Run facts the hooks need that are not variant options.

    :param model_family: ``qwen3_5`` / ``qwen3`` / ... (keys :data:`RESERVED_IDS`).
    :param seed: The run seed (router / keep-set / drop seeds derive from it).
    :param rank: Global rank (FFN token drop decorrelates its draws per rank).
    :param total_calls: Forwards per rank over the whole run (= steps x grad-accum); the horizon
        every annealing schedule is a fraction of.
    :param calls_per_step: Forwards per optimizer step (grad-accum), for the router callbacks.
    :param seq_len: Training window.
    :param mix_total_forwards: chunked-mix only: the mask-mix anneal horizon in forwards per rank.
    :param vocab_size: Embedding rows (soft-token tables).
    :param data_dir: Shard dir (soft-token stop set / gold sidecar default).
    :param meta: The shard's ``metadata.json``.
    :param tokenizer_id: HF tokenizer id of the family (soft-token stop set / piece tables).
    """

    model_family: str = "qwen3_5"
    seed: int = 0
    rank: int = 0
    total_calls: int = 1
    calls_per_step: int = 1
    seq_len: int = 0
    mix_total_forwards: int = 0
    vocab_size: int = 0
    data_dir: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    tokenizer_id: Optional[str] = None

    @property
    def ids(self):
        from olmo_core.data.document_chunk_landmark import RESERVED_IDS

        return RESERVED_IDS[self.model_family]


@dataclass
class VariantOptions:
    """Base (dense) options: none. Field names equal train_ctc_suite's argparse dest names."""


@dataclass
class ChunkedMixOptions(VariantOptions):
    """chunked-mix: curriculum mask mixing p_standard ``mix_start_p`` -> ``mix_end_p``."""

    mix_start_p: float = 0.80
    mix_end_p: float = 0.0
    mix_seed: int = 42
    mix_log_interval: int = 5


@dataclass
class PooledKVOptions(VariantOptions):
    """pooledkv: pooled-doc KV on the full-attention blocks (+ optional gold keep-set hook)."""

    pooled_keep_prob: float = 0.1
    pooled_no_len_bias: bool = False
    pooled_gold_sidecar: Optional[str] = None
    pooled_n_random: int = 2
    pooled_keep_mode: str = "gold_plus_random"
    pooled_n_gold: int = 0


@dataclass
class RouterOptions(VariantOptions):
    """Shared by the learned routers: their own (livelier) LR group."""

    router_lr: float = 1e-3


@dataclass
class FFNMoEOptions(RouterOptions):
    """ffnmoe: nested-width FFN router under a budget hinge."""

    ffn_moe_start_layer: int = 0
    ffn_moe_divisors: str = "1,2.4,4.9,10,32"
    ffn_moe_width_multiple: int = 8
    ffn_moe_no_null: bool = False
    ffn_moe_target: float = 0.01
    ffn_moe_budget_weight: float = 1.0
    ffn_moe_hinge_power: int = 1
    ffn_moe_trainable_width: int = 0
    ffn_moe_two_sided: bool = False
    ffn_moe_target_anneal_frac: float = 0.3
    ffn_moe_explore: float = 0.1
    ffn_moe_explore_anneal_frac: float = 0.3
    ffn_moe_recon_frac: float = 0.02
    ffn_moe_recon_weight: float = 0.0
    ffn_moe_entropy_weight: float = 0.0
    ffn_moe_layer_curriculum_frac: float = 0.0


@dataclass
class KVRouteOptions(RouterOptions):
    """kvroute: learned per-token KV keep/drop router on the attention blocks."""

    kv_route_start_layer: int = 0
    kv_route_target: float = 0.5
    kv_route_budget_weight: float = 1.0
    kv_route_one_sided: bool = False
    kv_route_target_anneal_frac: float = 0.3
    kv_route_explore: float = 0.0
    kv_route_explore_anneal_frac: float = 0.3


@dataclass
class FlexComputeOptions(FFNMoEOptions, KVRouteOptions):
    """flexcompute: FFN-width router + KV router (unless ``kv_route_off``) + optional block skip
    (``block_skip_target``) + optional joint FFN/attention budget (``flex_joint_target``)."""

    kv_route_off: bool = False
    block_skip_target: Optional[float] = None
    block_skip_start_layer: int = 0
    block_skip_capacity: Optional[float] = None
    block_skip_compact: bool = False
    flex_share_seq_len: int = 8192
    flex_joint_target: Optional[float] = None


@dataclass
class FFNDropOptions(VariantOptions):
    """ffndrop (drop-CPT): random null-FFN token drop, no router."""

    ffn_drop_max_rate: float = 0.0
    ffn_drop_layer_prob: float = 0.0
    ffn_drop_start_layer: int = 1

    @property
    def active(self) -> bool:
        return self.ffn_drop_max_rate > 0 or self.ffn_drop_layer_prob > 0


@dataclass
class LayerDropOptions(VariantOptions):
    """layerdrop: train-time WHOLE-block skipping (olmo_core.nn.layer_drop), a fixed set and/or
    random LayerDrop per optimizer step. No params; eval runs every block."""

    train_skip_layers: str = ""
    layer_drop_p: float = 0.0
    layer_drop_schedule: str = "uniform"
    layer_drop_exclude_first: int = 0
    layer_drop_exclude_last: int = 0
    layer_drop_rescale: bool = False

    @property
    def active(self) -> bool:
        return bool(self.train_skip_layers.strip()) or self.layer_drop_p > 0


@dataclass
class SoftTokenOptions(VariantOptions):
    """softtoken: pooled-document soft tokens (compacted rows) + keep policy."""

    st_n_random: int = 128
    st_n_random_range: str = ""
    st_keep_frac: Optional[float] = None
    st_keep_mode: str = "gold_plus_random"
    st_n_gold: int = 0
    st_keep_smallcat: int = 3
    st_keep_decoy_cats: int = 1
    st_keep_decoy_max_mult: float = 2.0
    st_keep_cats: int = 3
    st_keep_cats_random: bool = False
    st_cat_threshold_rule: str = "gap"
    st_cat_log_every: int = 50
    st_keep_prob: float = 0.1
    st_gold_blind: bool = False
    st_gold_sidecar: Optional[str] = None
    freeze_backbone: bool = False
    st_no_detach_soft_kv: bool = False
    st_no_detach_soft_gdn: bool = False
    st_len_bias: bool = False
    st_header_stop_id: Optional[int] = None
    st_header_stop_count: int = 1
    st_header_extra_tokens: int = 0
    st_keep_token_rule: str = "none"
    st_keep_token_k: float = 0
    st_keep_token_weights: Optional[str] = None
    st_keep_token_log_every: int = 50
    st_drop_slots: bool = False
    st_block_size: int = 0
    # position-aware keep rules (positional-block mode) and token exit (olmo3_cptdrop, 2026-10-22)
    st_keep_recent: int = 0
    st_keep_ramp: str = ""
    st_tokexit_layer: int = -1
    st_slot_mode: str = "mean"
    st_slot_stop_topk: int = 100
    st_slot_stop_rows: int = 512
    st_slot_tokenizer: Optional[str] = None
    st_neighbour_runs: int = 0
    st_distill_prob: float = 0.0
    st_distill_weight: float = 1.0
    st_aux_weight: float = 0.0
    st_mix_start_p: float = 0.0
    st_mix_end_p: float = 0.0
    st_mix_anneal_frac: float = 1.0
    # (2026-10-22) these four were read with getattr(..., default) while missing from this dataclass,
    # so options_from_namespace dropped them and every run silently used the defaults
    st_pad_multiple: int = 256
    st_span_len: int = 12
    st_kept_doc_k: Optional[float] = None
    st_keep_token_min: int = 0
    # online learned keep scorer (st_keep_token_rule="online"; olmo_core.nn.online_token_router)
    st_otr_keep_frac: float = 0.3
    st_otr_explore_frac: float = 0.1
    st_otr_doc_share: float = 1.0
    st_otr_keep_first: int = 0
    st_otr_early_nograd: bool = False
    st_otr_dense_tail_frac: float = 0.0
    st_otr_stages: str = ""
    st_otr_merge_dropped: bool = False
    st_otr_freeze_early: bool = False
    st_otr_min_per_doc: int = 0
    st_otr_freeze_after: int = 0
    st_otr_exit_loss_weight: float = 0.0
    st_otr_dense_keep_frac: float = 1.0
    st_otr_score: str = "router"
    st_otr_feature_layer: int = 0
    st_otr_drop_layer: int = 0
    st_otr_no_hidden: bool = False
    st_otr_no_query: bool = False
    st_otr_use_gold: bool = False
    st_otr_target: str = "gnorm"
    st_otr_ridge: float = 0.1
    st_otr_decay: float = 0.9
    st_otr_warmup_steps: int = 1
    st_otr_dense_every: int = 0
    st_otr_samples_per_row: int = 2048
    st_otr_proj_dim: int = 0
    st_otr_freeze_schedule: str = ""
    st_otr_target_layer: int = 0
    st_otr_early_lora: int = 0
    st_otr_early_lora_scale: float = 32.0
    st_otr_mass_target: float = 0.0
    st_otr_max_keep_frac: float = 1.0
    st_otr_mod_layers: str = ""
    st_otr_per_task_key_tokens: int = 0
    st_otr_log_every: int = 10


# ----------------------------------------------------------------------------------------------
# Options parsing
# ----------------------------------------------------------------------------------------------


def _coerce(value: str, typ: Any) -> Any:
    """Convert a ``--variant-opts`` string to the dataclass field's type (``Optional`` aware)."""
    s = value.strip()
    args = getattr(typ, "__args__", None)
    if args and type(None) in args:
        if s.lower() in ("none", "null", ""):
            return None
        typ = next(a for a in args if a is not type(None))
    if typ is bool:
        if s.lower() in ("1", "true", "yes", "on"):
            return True
        if s.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"not a bool: {value!r}")
    return typ(s)


def parse_kv(spec: str) -> Dict[str, str]:
    """
    Parse ``k=v,k=v`` (``;`` also separates, so a value may hold commas: ``a=1,2;b=3``).

    :param spec: The option string; empty means no options.

    :returns: Raw string values keyed by option name (dashes normalized to underscores).
    """
    out: Dict[str, str] = {}
    if not spec or not spec.strip():
        return out
    sep = ";" if ";" in spec else ","
    for part in spec.split(sep):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"variant option {part!r} is not k=v")
        k, v = part.split("=", 1)
        out[k.strip().lstrip("-").replace("-", "_")] = v
    return out


# ----------------------------------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------------------------------


def _attn_mixer(model_config):
    blk = model_config.block
    return blk["attn"].sequence_mixer if isinstance(blk, dict) else blk.sequence_mixer


class Variant:
    """
    One research arm, as a set of hooks relative to the dense recipe. The base class IS dense.

    Class attributes:

    * ``name`` -- the ``--variant`` value.
    * ``options_cls`` -- its typed options dataclass.
    * ``new_params`` -- adds parameters the base lacks (tolerant base load required).
    * ``budget_attach`` -- route budget-loss gradients through block outputs.
    * ``data_path`` -- ``"any"``, ``"padded"`` (one example per row; refuses packing),
      ``"packed"`` (requires packing) or ``"landmark"`` (landmark-packed windows).
    * ``setA_blocker`` -- why the arm cannot run on the canonical setA recipe, or ``None``.
    * ``needs_horizon`` -- its schedules anneal over ``ctx.total_calls``.
    * ``cp_blocker`` -- why the arm cannot run under (Ulysses) context parallelism, or ``None``.
    """

    name: str = "full"
    options_cls: Type[VariantOptions] = VariantOptions
    new_params: bool = False
    budget_attach: bool = False
    data_path: str = "any"
    setA_blocker: Optional[str] = None
    cp_blocker: Optional[str] = None
    #: Uses ``ctx.total_calls`` (annealing schedules), so the run horizon must be known.
    needs_horizon: bool = False

    # -- options ----------------------------------------------------------------------------
    def default_options(self) -> VariantOptions:
        return self.options_cls()

    def options_from_namespace(self, ns: Any) -> VariantOptions:
        """Pick this variant's option fields off an argparse namespace (same dest names)."""
        kw = {
            f.name: getattr(ns, f.name)
            for f in dataclasses.fields(self.options_cls)
            if hasattr(ns, f.name)
        }
        return self.options_cls(**kw)

    def options_from_kv(self, spec: str) -> VariantOptions:
        """Build typed options from a ``k=v,...`` string, refusing unknown keys."""
        return self.options_from_dict(parse_kv(spec))

    def options_from_dict(self, raw: Mapping[str, str]) -> VariantOptions:
        """Build typed options from raw string values, refusing unknown keys."""
        fields = {f.name: f for f in dataclasses.fields(self.options_cls)}
        unknown = sorted(set(raw) - set(fields))
        if unknown:
            raise ValueError(
                f"variant {self.name!r} has no option(s) {unknown}; valid: {sorted(fields)}"
            )
        kw = {k: _coerce(v, fields[k].type) for k, v in raw.items()}
        return self.options_cls(**kw)

    def validate(self, o: VariantOptions) -> None:
        """Raise ``ValueError`` on an option combination that would be silently ignored."""

    # -- hooks -------------------------------------------------------------------------------
    def factory_kwargs(self, o: VariantOptions) -> Dict[str, Any]:
        return {}

    def apply_model_config(self, cfg, o: VariantOptions, ctx: VariantContext) -> None:
        pass

    def train_module_overrides(self, o: VariantOptions, ctx: VariantContext) -> Dict[str, Any]:
        return {}

    def optim_group_overrides(self, o: VariantOptions) -> List[Any]:
        return []

    def data_overrides(
        self,
        o: VariantOptions,
        ctx: VariantContext,
        *,
        source_paths: List[str],
        label_mask_paths: List[str],
        tokenizer,
        best_fit: bool = False,
    ) -> Optional[Any]:
        return None

    def post_build_model(self, model, o: VariantOptions, ctx: VariantContext) -> None:
        if self.budget_attach:
            # budget gradients through the block outputs (olmo_core.nn.budget_attach): the separate
            # loss term recomputed every checkpointed block at once (68 GB vs 26 GB dense)
            model.budget_attach = True

    def post_build_train_module(self, train_module, o: VariantOptions, ctx: VariantContext):
        pass

    def extra_callbacks(self, o: VariantOptions, ctx: VariantContext) -> Dict[str, Any]:
        return {}

    def export_config(self, o: VariantOptions) -> Dict[str, Any]:
        return {}


class Full(Variant):
    """Plain causal attention (dense); the box markers are ordinary tokens."""

    name = "full"


class Chunked(Variant):
    """Document-chunked mask on the full-attention blocks (GDN blocks ignore ``chunk_ids``)."""

    name = "chunked"
    cp_blocker = "chunk_ids are not sequence-sharded (Transformer refuses document_chunk_attention + CP)"

    def factory_kwargs(self, o):
        return dict(document_chunked=True, cross_doc_mode="chunked")

    def _mix_keys(self, o, ctx) -> Dict[str, Any]:
        return {}

    def apply_model_config(self, cfg, o, ctx):
        from olmo_core.nn.attention import AttentionType

        mixer = _attn_mixer(cfg)
        if self.factory_kwargs(o).get("document_chunked"):
            # Same result as the factory kwargs (idempotent when the factory already applied
            # them): the setA builder gets its model config from a fixed factory call.
            mixer.name = AttentionType.document_chunked
            mixer.cross_doc_mode = "chunked"
        ids = ctx.ids
        cfg.document_chunk_attention = {
            "doc_start_id": ids.doc_start,
            "doc_end_id": ids.doc_end,
            "eos_id": ids.eos,
            "mode": "chunked",
            **self._mix_keys(o, ctx),
        }


class ChunkedMix(Chunked):
    """chunked + curriculum mask mixing (the project default; eval uses the pure chunked mask)."""

    name = "chunked-mix"
    options_cls = ChunkedMixOptions
    needs_horizon = True

    def _mix_keys(self, o, ctx):
        assert isinstance(o, ChunkedMixOptions)
        return dict(
            mix_start_p=o.mix_start_p,
            mix_end_p=o.mix_end_p,
            mix_total_forwards=ctx.mix_total_forwards,
            mix_seed=o.mix_seed,
            mix_log_interval=o.mix_log_interval,
        )


class SparseLandmark(Variant):
    """Sparse landmark attention on the full-attention blocks + landmark-packed data."""

    name = "sparselandmark"
    data_path = "landmark"
    # Ulysses CP is supported (SparseLandmarkAttention.apply_cp). The base checkpoint must have
    # its landmark embedding row (ids.landmark) repaired -- it is untrained in the raw conversion.

    def apply_model_config(self, cfg, o, ctx):
        from olmo_core.nn.attention import AttentionType

        # Full attention within a 64-token block, past blocks only through their landmark token.
        # The qwen3_5 elementwise output gate is KEPT; GDN blocks are untouched.
        mixer = _attn_mixer(cfg)
        mixer.name = AttentionType.sparse_landmark
        mixer.mem_freq = MEM_FREQ_SPARSE
        mixer.num_landmarks = 1

    def data_overrides(self, o, ctx, *, source_paths, label_mask_paths, tokenizer, best_fit=False):
        from olmo_core.data.composable import (
            LandmarkPackingInstanceSourceConfig,
            NumpyDocumentSourceConfig,
        )
        from olmo_core.data.types import LandmarkPackingStrategy

        if ctx.seq_len % (MEM_FREQ_SPARSE + 1) != 0:
            raise SystemExit(
                f"--variant sparselandmark requires --seq-len divisible by the landmark block "
                f"({MEM_FREQ_SPARSE + 1}); got {ctx.seq_len}."
            )
        ids = ctx.ids
        # best_fit: packing-match a dense arm built with PackingInstanceSource (BFD); the
        # research trainer keeps the historical next_fit default.
        extra = dict(packing_strategy=LandmarkPackingStrategy.best_fit_decreasing) if best_fit else {}
        return LandmarkPackingInstanceSourceConfig(
            source=NumpyDocumentSourceConfig(
                source_paths=list(source_paths),
                tokenizer=tokenizer,
                label_mask_paths=list(label_mask_paths),
                expand_glob=True,
            ),
            sequence_length=ctx.seq_len,
            mem_freq=MEM_FREQ_SPARSE,
            mem_id=ids.landmark,
            pad_id=ids.pad,
            **extra,
        )


def _install_fingerprint_keep(train_module, gold_table, ctx, **kw):
    from olmo_core.nn.attention.pooled_doc_kv import (
        install_pooled_doc_keep,
        make_fingerprint_keep_docs_fn,
    )

    ids = ctx.ids
    keep_fn = make_fingerprint_keep_docs_fn(
        gold_table, doc_start_id=ids.doc_start, doc_end_id=ids.doc_end, eos_id=ids.eos, **kw
    )
    return install_pooled_doc_keep(train_module.model, keep_fn)


class PooledKV(Chunked):
    """Pooled-doc KV on the full-attention blocks (train-time pooling; inference = full attn)."""

    name = "pooledkv"
    options_cls = PooledKVOptions
    data_path = "padded"
    setA_blocker = (
        "the keep-set fingerprint / role reconstruction assume ONE example per padded row; setA "
        "rows are BFD-packed"
    )

    def factory_kwargs(self, o):
        return {}

    def apply_model_config(self, cfg, o, ctx):
        from olmo_core.nn.attention import AttentionType

        assert isinstance(o, PooledKVOptions)
        mixer = _attn_mixer(cfg)
        mixer.name = AttentionType.pooled_doc_kv
        mixer.pooled_keep_prob = o.pooled_keep_prob
        mixer.pooled_keep_seed = ctx.seed
        mixer.pooled_len_bias = not o.pooled_no_len_bias
        super().apply_model_config(cfg, o, ctx)  # document_chunk_attention (no mix keys)

    def post_build_train_module(self, train_module, o, ctx):
        assert isinstance(o, PooledKVOptions)
        if not o.pooled_gold_sidecar:
            return
        with open(o.pooled_gold_sidecar) as f:
            gold_table = json.load(f)
        holder = _install_fingerprint_keep(
            train_module,
            gold_table,
            ctx,
            n_random=o.pooled_n_random,
            mode=o.pooled_keep_mode,
            seed=ctx.seed,
            n_gold=o.pooled_n_gold,
        )
        if holder.n_attached == 0:
            raise SystemExit(
                "[variants] --pooled-gold-sidecar given but no PooledDocKVAttention layers were "
                "found on the model (variant/config mismatch?)."
            )
        print(
            f"[variants] pooledkv: gold keep-set hook installed on {holder.n_attached} layers "
            f"({len(gold_table)} fingerprints, mode={o.pooled_keep_mode}, "
            f"n_random={o.pooled_n_random})",
            flush=True,
        )


def _router_group(params: List[str], lr: float):
    from olmo_core.optim import OptimGroupOverride

    # The router decides discrete routing from a cold start and wants a livelier LR than the
    # pretrained backbone; gains/routers are small and must not be decayed.
    return OptimGroupOverride(params=params, opts=dict(lr=lr, weight_decay=0.0))


class FFNMoE(Variant):
    """Nested-width FFN router (per-token prefix width or null) under a budget hinge."""

    name = "ffnmoe"
    options_cls = FFNMoEOptions
    new_params = True
    needs_horizon = True
    budget_attach = True

    def _ffn_routed(self, o) -> bool:
        return True

    def optim_group_overrides(self, o):
        return [
            _router_group(
                ["blocks.*.feed_forward._nffn_router.*", "blocks.*.feed_forward._nffn_gain"],
                o.router_lr,
            )
        ]

    def _enable_ffn(self, model, o, ctx):
        tc = ctx.total_calls
        model.enable_nested_ffn_moe(
            start_layer=o.ffn_moe_start_layer,
            divisors=[float(x) for x in o.ffn_moe_divisors.split(",")],
            include_null=not o.ffn_moe_no_null,
            target_cost=o.ffn_moe_target,
            budget_weight=o.ffn_moe_budget_weight,
            hinge_power=o.ffn_moe_hinge_power,
            two_sided=o.ffn_moe_two_sided,
            target_anneal_calls=int(tc * o.ffn_moe_target_anneal_frac),
            explore_prob=o.ffn_moe_explore,
            explore_anneal_calls=int(tc * o.ffn_moe_explore_anneal_frac),
            recon_frac=o.ffn_moe_recon_frac,
            recon_weight=o.ffn_moe_recon_weight,
            entropy_weight=o.ffn_moe_entropy_weight,
            seed=ctx.seed,
            layer_curriculum_calls=int(tc * o.ffn_moe_layer_curriculum_frac),
            trainable_width=o.ffn_moe_trainable_width,
            width_multiple=o.ffn_moe_width_multiple,
        )
        print(
            f"[variants] ffnmoe: routed from layer {o.ffn_moe_start_layer}, rungs="
            f"{model._nested_ffn_moe['widths']}, target {o.ffn_moe_target} annealed over "
            f"{int(tc * o.ffn_moe_target_anneal_frac)}/{tc} calls",
            flush=True,
        )

    def post_build_model(self, model, o, ctx):
        self._enable_ffn(model, o, ctx)
        super().post_build_model(model, o, ctx)

    def extra_callbacks(self, o, ctx):
        from olmo_core.train.callbacks import NestedFFNMoECallback

        return {"ffn_moe": NestedFFNMoECallback(calls_per_step=ctx.calls_per_step)}

    def export_config(self, o):
        return {
            "ffn_moe": {
                "start_layer": o.ffn_moe_start_layer,
                "divisors": o.ffn_moe_divisors,
                "include_null": not o.ffn_moe_no_null,
                "width_multiple": o.ffn_moe_width_multiple,
                "trainable_width": o.ffn_moe_trainable_width,
            }
        }


class KVRoute(Variant):
    """Learned KV keep/drop router on the attention blocks (root-located routers)."""

    name = "kvroute"
    options_cls = KVRouteOptions
    needs_horizon = True
    cp_blocker = "kv_route does not support context parallelism (single-rank attention only)"
    new_params = True
    budget_attach = True
    data_path = "packed"

    def kv_route_enabled(self, o) -> bool:
        return True

    def optim_group_overrides(self, o):
        return [_router_group(["kvr_routers.*"], o.router_lr)] if self.kv_route_enabled(o) else []

    def _enable_kv(self, model, o, ctx):
        tc = ctx.total_calls
        model.enable_kv_route(
            start_layer=o.kv_route_start_layer,
            target=o.kv_route_target,
            budget_weight=o.kv_route_budget_weight,
            two_sided=not o.kv_route_one_sided,
            target_anneal_calls=int(tc * o.kv_route_target_anneal_frac),
            explore_prob=o.kv_route_explore,
            explore_anneal_calls=int(tc * o.kv_route_explore_anneal_frac),
            seed=ctx.seed,
        )
        print(
            f"[variants] kvroute: routed attention layers {model._kv_route['routed']}, keep target "
            f"{o.kv_route_target} annealed over {int(tc * o.kv_route_target_anneal_frac)}/{tc} calls",
            flush=True,
        )

    def post_build_model(self, model, o, ctx):
        Variant.post_build_model(self, model, o, ctx)  # budget_attach
        self._enable_kv(model, o, ctx)

    def extra_callbacks(self, o, ctx):
        from olmo_core.train.callbacks import KVRouteCallback

        return {"kv_route": KVRouteCallback(calls_per_step=ctx.calls_per_step)}

    def export_config(self, o):
        return {
            "kv_route": {
                "start_layer": o.kv_route_start_layer,
                "target": o.kv_route_target,
                "router_location": "root",
            }
        }


class FlexCompute(FFNMoE, KVRoute):
    """FFN-width router + KV router (unless ``kv_route_off``) + block skip + joint budget."""

    name = "flexcompute"
    options_cls = FlexComputeOptions
    cp_blocker = "kv_route / compacted block skip are single-rank (unverified under CP)"
    new_params = True
    budget_attach = True
    data_path = "packed"

    def kv_route_enabled(self, o) -> bool:
        # --kv-route-off measures the FFN-width + block-skip routers without the KV router: a
        # COMPACTED block skip already removes a skipped token as a key (5.650x -> 5.402x).
        return not o.kv_route_off

    def _block_skip(self, o) -> bool:
        return o.block_skip_target is not None

    def optim_group_overrides(self, o):
        out = FFNMoE.optim_group_overrides(self, o) + KVRoute.optim_group_overrides(self, o)
        if self._block_skip(o):
            out.append(_router_group(["bskip_routers.*"], o.router_lr))
        return out

    def post_build_model(self, model, o, ctx):
        tc = ctx.total_calls
        self._enable_ffn(model, o, ctx)
        Variant.post_build_model(self, model, o, ctx)  # budget_attach
        if self.kv_route_enabled(o):
            self._enable_kv(model, o, ctx)
        if self._block_skip(o):
            model.enable_block_skip(
                start_layer=o.block_skip_start_layer,
                target=o.block_skip_target,
                budget_weight=o.kv_route_budget_weight,
                two_sided=not o.kv_route_one_sided,
                target_anneal_calls=int(tc * o.kv_route_target_anneal_frac),
                seed=ctx.seed,
                compact=o.block_skip_compact,
                capacity_frac=o.block_skip_capacity,
            )
            print(
                f"[variants] block-skip: routed blocks {model._block_skip['routed']}, run target "
                f"{o.block_skip_target} compact={o.block_skip_compact}",
                flush=True,
            )
        if o.flex_joint_target is not None:
            # One budget over BOTH routers, weighted by each block's share of dense FLOPs.
            from olmo_core.nn.joint_budget import install_joint_budget

            install_joint_budget(
                model,
                target=o.flex_joint_target,
                seq_len=o.flex_share_seq_len or ctx.seq_len,
                anneal_calls=int(tc * o.ffn_moe_target_anneal_frac),
            )
            print(
                f"[variants] flexcompute: JOINT FFN+attention budget target {o.flex_joint_target}",
                flush=True,
            )

    def extra_callbacks(self, o, ctx):
        out = dict(FFNMoE.extra_callbacks(self, o, ctx))
        if self.kv_route_enabled(o):
            out.update(KVRoute.extra_callbacks(self, o, ctx))
        if self._block_skip(o):
            from olmo_core.train.callbacks import BlockSkipCallback

            out["block_skip"] = BlockSkipCallback(calls_per_step=ctx.calls_per_step)
        return out

    def export_config(self, o):
        out = dict(FFNMoE.export_config(self, o))
        if self.kv_route_enabled(o):
            out.update(KVRoute.export_config(self, o))
        if self._block_skip(o):
            out["block_skip"] = {
                "start_layer": o.block_skip_start_layer,
                "target": o.block_skip_target,
                "compact": o.block_skip_compact,
            }
        return out


class FFNDrop(Variant):
    """Drop-CPT: random null-FFN token drop (no router). In train_ctc_suite this is a modifier
    (``--ffn-drop-*`` on any non-routed arm); on setA it is ``--variant ffndrop``."""

    name = "ffndrop"
    options_cls = FFNDropOptions

    def validate(self, o):
        assert isinstance(o, FFNDropOptions)
        if not o.active:
            raise ValueError("ffndrop needs ffn_drop_max_rate > 0 or ffn_drop_layer_prob > 0")

    def post_build_model(self, model, o, ctx):
        assert isinstance(o, FFNDropOptions)
        model.enable_ffn_token_drop(
            max_rate=o.ffn_drop_max_rate,
            layer_prob=o.ffn_drop_layer_prob,
            start_layer=o.ffn_drop_start_layer,
            seed=ctx.seed,
            rank=ctx.rank,
        )
        print(
            f"[variants] ffn-drop: layers >= {o.ffn_drop_start_layer}, per-row token rate "
            f"U[0, {o.ffn_drop_max_rate}], whole-layer prob {o.ffn_drop_layer_prob}",
            flush=True,
        )

    def extra_callbacks(self, o, ctx):
        from olmo_core.train.callbacks import FFNTokenDropCallback

        return {"ffn_drop": FFNTokenDropCallback()}

    def export_config(self, o):
        return {
            "ffn_token_drop": {
                "max_rate": o.ffn_drop_max_rate,
                "layer_prob": o.ffn_drop_layer_prob,
                "start_layer": o.ffn_drop_start_layer,
            }
        }


class LayerDrop(Variant):
    """Train-time whole-block skipping (fixed set and/or LayerDrop). In train_ctc_suite this is a
    modifier (``--train-skip-layers`` / ``--layer-drop-*`` on any arm, dense or softtoken); on
    setA it is ``--variant layerdrop``. A skipped block's forward is never called, so it saves
    wall-clock, not just FLOPs; the draw is seeded from (seed, global step), identical on all
    ranks and constant across a step's micro-batches (LayerDropCallback)."""

    name = "layerdrop"
    options_cls = LayerDropOptions

    def validate(self, o):
        assert isinstance(o, LayerDropOptions)
        if not o.active:
            raise ValueError("layerdrop needs train_skip_layers or layer_drop_p > 0")

    def post_build_model(self, model, o, ctx):
        assert isinstance(o, LayerDropOptions)
        from olmo_core.nn.layer_drop import parse_layer_list

        model.enable_layer_drop(
            fixed=set(parse_layer_list(o.train_skip_layers, len(model.blocks))),
            p=o.layer_drop_p,
            schedule=o.layer_drop_schedule,
            exclude_first=o.layer_drop_exclude_first,
            exclude_last=o.layer_drop_exclude_last,
            rescale=o.layer_drop_rescale,
            seed=ctx.seed,
        )
        plan = model._layer_drop
        print(
            f"[variants] layer-drop: fixed skip {sorted(plan.fixed)}, p={o.layer_drop_p} "
            f"({o.layer_drop_schedule}) on {len(plan.probs)} eligible blocks, rescale="
            f"{o.layer_drop_rescale}; expected blocks run {plan.expected_run_frac():.3f}",
            flush=True,
        )

    def extra_callbacks(self, o, ctx):
        from olmo_core.train.callbacks import LayerDropCallback

        return {"layer_drop": LayerDropCallback()}


# -- soft tokens -------------------------------------------------------------------------------


def resolve_keep_frac(o: SoftTokenOptions) -> Optional[float]:
    """The soft-token non-gold keep FRACTION (``st_keep_frac``; ``gold_pooled_random`` falls back
    to ``st_keep_prob``, the flag the gold-blind arms spell it with), or ``None``."""
    if o.st_keep_frac is not None:
        return o.st_keep_frac
    if o.st_keep_mode == "gold_pooled_random":
        print(
            f"[variants] softtoken: gold_pooled_random without --st-keep-frac; using "
            f"--st-keep-prob {o.st_keep_prob} as the non-gold keep fraction",
            flush=True,
        )
        return o.st_keep_prob
    return None


def resolve_cat_keep(o: SoftTokenOptions) -> Optional[Dict[str, Any]]:
    """The WHOLE-CATEGORY keep config for ``enable_pooled_soft_tokens`` (``smallcat_keep`` /
    ``gold_plus_wholecats``), or ``None``. Whole categories only, more than one, so "which category
    is entirely real?" names nothing (records/ds64-overnight-2026-09-14.md)."""
    if o.st_keep_mode not in ("smallcat_keep", "gold_plus_wholecats"):
        return None
    return {
        "mode": o.st_keep_mode,
        "n_small": o.st_keep_smallcat,
        "n_decoy": o.st_keep_decoy_cats,
        "decoy_max_size_mult": o.st_keep_decoy_max_mult,
        "n_cats": o.st_keep_cats,
        "cats_random": o.st_keep_cats_random,
        "threshold_rule": o.st_cat_threshold_rule,
        "log_every": o.st_cat_log_every,
    }


def shard_head_token_ids(
    data_dir: str, meta: Dict[str, Any], seq_len: int, n_rows: int, why: str
) -> Tuple[Any, str]:
    """Read the head of the shard's first (raw, headerless) token-id part via ``np.memmap``.

    :returns: ``(1-D int array, basename of the part read)``.
    """
    import glob

    import numpy as np

    parts = sorted(glob.glob(os.path.join(data_dir, "token_ids_part_*.npy")))
    if not parts:
        raise SystemExit(f"{why} needs {data_dir}/token_ids_part_*.npy")
    dtype = np.dtype(meta.get("dtype") or "uint32")
    n_total = os.path.getsize(parts[0]) // dtype.itemsize
    arr = np.memmap(parts[0], dtype=dtype, mode="r", shape=(n_total,))
    row_len = int(meta.get("max_example_len") or seq_len)
    n_tok = min(n_total, max(1, int(n_rows)) * max(1, row_len))
    return np.asarray(arr[:n_tok]), os.path.basename(parts[0])


def build_keep_token_tables(o: SoftTokenOptions, ctx: VariantContext) -> Any:
    """The ``st_keep_token_rule=rule`` feature tables (IDF over the shard head + tokenizer piece
    tables); refused rather than degraded if the tokenizer will not load."""
    import numpy as np
    import torch

    from olmo_core.nn.pooled_soft_token import (
        KeepTokenTables,
        build_token_idf,
        build_token_piece_tables,
    )

    vocab = int(ctx.vocab_size)
    assert ctx.data_dir is not None
    head, name = shard_head_token_ids(
        ctx.data_dir, ctx.meta, ctx.seq_len, o.st_slot_stop_rows, "--st-keep-token-rule rule"
    )
    idf = build_token_idf(head, vocab)
    tok_id = o.st_slot_tokenizer or ctx.tokenizer_id
    try:
        from transformers import AutoTokenizer

        _tk = AutoTokenizer.from_pretrained(tok_id)
        pieces = _tk.convert_ids_to_tokens(list(range(vocab)))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"--st-keep-token-rule rule needs a tokenizer for its piece features, but "
            f"{tok_id!r} did not load ({exc}); pass --st-slot-tokenizer pointing at a local copy"
        )
    sent_end, is_cap, is_dig, tok_len = build_token_piece_tables(pieces)
    print(
        f"[variants] keep-token rule tables: idf over {head.size} tokens of {name} "
        f"(mean {float(np.mean(idf)):.2f}), pieces from {tok_id} "
        f"(sent_end {int(sent_end.sum())}, cap {int(is_cap.sum())}, digit {int(is_dig.sum())} "
        f"of {vocab} ids)",
        flush=True,
    )
    return KeepTokenTables(
        idf=torch.from_numpy(idf),
        sent_end=torch.from_numpy(sent_end),
        is_cap=torch.from_numpy(is_cap),
        is_dig=torch.from_numpy(is_dig),
        tok_len=torch.from_numpy(tok_len),
    )


def build_online_router(o: SoftTokenOptions, ctx: VariantContext, d_model: int) -> Any:
    """The ``st_keep_token_rule=online`` router (:mod:`olmo_core.nn.online_token_router`). With
    ``st_otr_use_gold`` it reads the shard's gold-fingerprint sidecar as a per-document 0/1 feature
    (no document is ever kept whole because of it)."""
    from olmo_core.nn.online_token_router import OnlineRouterConfig, OnlineTokenRouter

    cfg = OnlineRouterConfig(
        keep_frac=o.st_otr_keep_frac,
        explore_frac=o.st_otr_explore_frac,
        doc_share=o.st_otr_doc_share,
        keep_first=o.st_otr_keep_first,
        early_nograd=o.st_otr_early_nograd,
        dense_tail_frac=o.st_otr_dense_tail_frac,
        stages=o.st_otr_stages,
        merge_dropped=o.st_otr_merge_dropped,
        freeze_early=o.st_otr_freeze_early,
        min_per_doc=o.st_otr_min_per_doc,
        freeze_after=o.st_otr_freeze_after,
        exit_loss_weight=o.st_otr_exit_loss_weight,
        dense_keep_frac=o.st_otr_dense_keep_frac,
        score=o.st_otr_score,
        feature_layer=o.st_otr_feature_layer,
        drop_layer=o.st_otr_drop_layer,
        use_hidden=not o.st_otr_no_hidden,
        use_query=not o.st_otr_no_query,
        use_gold=o.st_otr_use_gold,
        target=o.st_otr_target,
        ridge=o.st_otr_ridge,
        decay=o.st_otr_decay,
        warmup_steps=o.st_otr_warmup_steps,
        dense_every=o.st_otr_dense_every,
        samples_per_row=o.st_otr_samples_per_row,
        proj_dim=o.st_otr_proj_dim,
        freeze_schedule=o.st_otr_freeze_schedule,
        target_layer=o.st_otr_target_layer,
        early_lora=o.st_otr_early_lora,
        mass_target=o.st_otr_mass_target,
        max_keep_frac=o.st_otr_max_keep_frac,
        mod_layers=o.st_otr_mod_layers,
        per_task_key_tokens=o.st_otr_per_task_key_tokens,
        seed=ctx.seed,
        log_every=o.st_otr_log_every,
    )
    # the gold table is always loaded when the shard has one: it feeds the online leak diagnostic
    # (online_router/leak_auc); it is a FEATURE only with --st-otr-use-gold
    gold_table = None
    sidecar = o.st_gold_sidecar or os.path.join(ctx.data_dir or "", "gold_fingerprints.json")
    if o.st_otr_use_gold or os.path.exists(sidecar):
        with open(sidecar) as f:
            gold_table = json.load(f)
    ids = ctx.ids
    router = OnlineTokenRouter(
        cfg,
        doc_start_id=ids.doc_start,
        doc_end_id=ids.doc_end,
        eos_id=ids.eos,
        d_model=d_model,
        gold_table=gold_table,
        rank=ctx.rank,
    )
    print(
        f"[variants] softtoken online router: {cfg} -> {router.n_features} features"
        f"{f', gold sidecar {len(gold_table)} fingerprints' if gold_table is not None else ''}",
        flush=True,
    )
    return router


def build_slot_stop_set(o: SoftTokenOptions, ctx: VariantContext) -> List[int]:
    """The ``st_slot_mode`` stop set: the ``st_slot_stop_topk`` most frequent ids over the shard
    head + marker/pad/landmark/eos ids + (if a tokenizer loads) non-alphanumeric pieces."""
    from olmo_core.nn.pooled_soft_token import build_slot_stop_ids

    ids = ctx.ids
    assert ctx.data_dir is not None
    head, name = shard_head_token_ids(
        ctx.data_dir, ctx.meta, ctx.seq_len, o.st_slot_stop_rows, o.st_slot_mode
    )
    n_tok = int(head.size)
    decode = None
    tok_id = o.st_slot_tokenizer or ctx.tokenizer_id
    try:
        from transformers import AutoTokenizer

        _tk = AutoTokenizer.from_pretrained(tok_id)

        def decode(t: int) -> str:  # noqa: F811
            return _tk.decode([t])

    except Exception as exc:  # noqa: BLE001
        print(
            f"[variants] slot stop set: tokenizer {tok_id!r} did not load ({exc}); "
            "using FREQUENCY-ONLY stop ids (the top-K already covers common punctuation)",
            flush=True,
        )
    stop, shown = build_slot_stop_ids(
        head,
        top_k=o.st_slot_stop_topk,
        extra_ids=(ids.doc_start, ids.doc_end, ids.eos, ids.landmark, ids.pad),
        decode=decode,
    )
    print(
        f"[variants] slot stop set: {len(stop)} ids from {n_tok} tokens of "
        f"{name} (top-{o.st_slot_stop_topk} + markers"
        f"{' + punctuation/whitespace' if decode is not None else ''}); "
        f"most frequent dropped: {' '.join(shown)}",
        flush=True,
    )
    return stop


def parse_keep_ramp(spec: str) -> Optional[Tuple[float, float]]:
    """``"a,b"`` -> ``(a, b)`` (``--st-keep-ramp``); empty -> ``None``."""
    if not spec:
        return None
    parts = [p.strip() for p in str(spec).split(",")]
    if len(parts) != 2:
        raise ValueError(f"--st-keep-ramp wants 'a,b' (two fractions), got {spec!r}")
    return (float(parts[0]), float(parts[1]))


class SoftToken(Variant):
    """Pooled-document soft tokens: non-kept documents collapse to slots on compacted rows."""

    name = "softtoken"
    options_cls = SoftTokenOptions
    new_params = True
    needs_horizon = True
    data_path = "padded"
    cp_blocker = "unverified under CP"
    setA_blocker = (
        "per-row gold fingerprints and the compacted-row keep policy assume ONE example per "
        "padded row; setA rows are BFD-packed"
    )

    def train_module_overrides(self, o, ctx):
        return {"microbatch_sort_pad_id": ctx.ids.eos}  # length-homogeneous micro-batches

    def post_build_model(self, model, o, ctx):
        assert isinstance(o, SoftTokenOptions)
        from olmo_core.nn.pooled_soft_token import parse_keep_token_weights

        ids = ctx.ids
        tc = ctx.total_calls
        online = o.st_keep_token_rule == "online"
        if online and (o.st_header_extra_tokens or o.st_header_stop_id is not None or not o.st_drop_slots):
            raise SystemExit(
                "[variants] --st-keep-token-rule online is slot-less and learns its own headers: "
                "pass --st-drop-slots and no --st-header-extra-tokens / --st-header-stop-id"
            )
        _w, _sd = (
            parse_keep_token_weights(o.st_keep_token_weights)
            if o.st_keep_token_rule == "rule"
            else (None, None)
        )
        model.enable_pooled_soft_tokens(
            ids.doc_start,
            ids.doc_end,
            ids.eos,
            placeholder_id=ids.landmark,
            keep_prob=o.st_keep_prob,
            keep_seed=ctx.seed,
            aux_match_weight=o.st_aux_weight,
            detach_soft_kv=not o.st_no_detach_soft_kv,
            len_bias=o.st_len_bias,
            distill_prob=o.st_distill_prob,
            distill_weight=o.st_distill_weight,
            header_stop_id=o.st_header_stop_id,
            header_stop_count=o.st_header_stop_count,
            header_extra_tokens=o.st_header_extra_tokens,
            keep_token_rule=o.st_keep_token_rule,
            keep_token_k=(o.st_otr_keep_frac if online else o.st_keep_token_k),
            online_router=(
                build_online_router(o, ctx, int(model.embeddings.weight.shape[1])) if online else None
            ),
            keep_token_weights=_w,
            keep_token_sd=_sd,
            keep_token_tables=(
                build_keep_token_tables(o, ctx) if o.st_keep_token_rule == "rule" else None
            ),
            keep_token_log_every=o.st_keep_token_log_every,
            detach_soft_gdn=not o.st_no_detach_soft_gdn,
            drop_slots=o.st_drop_slots,
            pad_multiple=o.st_pad_multiple,
            span_len=o.st_span_len,
            kept_doc_k=o.st_kept_doc_k,
            keep_token_min=o.st_keep_token_min,
            positional_block_size=o.st_block_size,
            keep_recent=o.st_keep_recent,
            keep_ramp=parse_keep_ramp(o.st_keep_ramp),
            # Compression-mixing curriculum: with st_gold_blind there is NO keep hook, so the model
            # applies it to the seeded keep_prob draw itself; otherwise the hook carries it.
            mix_start_p=o.st_mix_start_p,
            mix_end_p=o.st_mix_end_p,
            mix_total_calls=int(tc * o.st_mix_anneal_frac),
            slot_mode=o.st_slot_mode,
            slot_stop_ids=(None if o.st_slot_mode == "mean" else build_slot_stop_set(o, ctx)),
            cat_keep=resolve_cat_keep(o),
        )
        if online and o.st_otr_early_lora > 0:
            from olmo_core.nn.early_lora import install_early_lora

            r = model._online_token_router
            mods = install_early_lora(model, r.cfg.drop_layer, o.st_otr_early_lora, o.st_otr_early_lora_scale)
            print(f"[variants] online router early LoRA: blocks [0, {r.cfg.drop_layer}) + embeddings frozen, "
                  f"{len(mods)} linears get rank-{o.st_otr_early_lora} adapters (scale {o.st_otr_early_lora_scale})",
                  flush=True)
        if online and o.st_otr_freeze_early:
            from olmo_core.nn.online_token_router import freeze_early_blocks

            r = model._online_token_router
            n = freeze_early_blocks(model, r.cfg.drop_layer)
            print(f"[variants] online router freeze_early: embeddings + blocks [0, {r.cfg.drop_layer}) frozen "
                  f"({n / 1e6:.0f}M params: no grads, no optimizer state)", flush=True)
        if o.st_tokexit_layer >= 0:
            # token exit: the same keep rule, dropped tokens leave at the input of this block
            model.enable_token_exit(o.st_tokexit_layer)
        if o.freeze_backbone:
            # learned-slot screening arm: only the soft-token projector trains
            n_train = 0
            for pname, prm in model.named_parameters():
                trainable = pname.startswith("pooled_projector")
                prm.requires_grad_(trainable)
                n_train += int(trainable) * prm.numel()
            print(
                f"[variants] --freeze-backbone: {n_train:,} trainable params (pooled_projector only)",
                flush=True,
            )
        print(
            f"[variants] softtoken: slot_mode={o.st_slot_mode} header_stop_id={o.st_header_stop_id} "
            f"(count {o.st_header_stop_count}, extra {o.st_header_extra_tokens}) "
            f"detach_gdn={not o.st_no_detach_soft_gdn} drop_slots={o.st_drop_slots} "
            f"block_size={o.st_block_size} keep_recent={o.st_keep_recent} "
            f"keep_ramp={o.st_keep_ramp or None} tokexit_layer={o.st_tokexit_layer} "
            f"detach={not o.st_no_detach_soft_kv} len_bias={o.st_len_bias} "
            f"distill_prob={o.st_distill_prob} keep_mode={o.st_keep_mode} "
            f"n_random={o.st_n_random_range or o.st_n_random} keep_frac={o.st_keep_frac} "
            f"keep_token={o.st_keep_token_rule}/k={o.st_keep_token_k} keep_token_weights={_w} "
            f"gold_blind={o.st_gold_blind} keep_prob={o.st_keep_prob} "
            f"mix=({o.st_mix_start_p}->{o.st_mix_end_p} over "
            f"{int(tc * o.st_mix_anneal_frac)}/{tc} calls)",
            flush=True,
        )

    def post_build_train_module(self, train_module, o, ctx):
        assert isinstance(o, SoftTokenOptions)
        # smallcat_keep is gold-blind by construction; gold_plus_wholecats installs the hook with
        # n_random=0 so the holder's mask is exactly the gold set for the model to grow. The online
        # router never keeps a document whole; it reads gold (if asked) from the sidecar itself.
        if o.st_gold_blind or o.st_keep_mode == "smallcat_keep" or o.st_keep_token_rule == "online":
            return
        whole = o.st_keep_mode == "gold_plus_wholecats"
        sidecar = o.st_gold_sidecar or os.path.join(ctx.data_dir or "", "gold_fingerprints.json")
        with open(sidecar) as f:
            gold_table = json.load(f)
        holder = _install_fingerprint_keep(
            train_module,
            gold_table,
            ctx,
            n_random=(0 if whole else o.st_n_random),
            n_random_range=(
                None
                if whole
                else (
                    tuple(int(x) for x in o.st_n_random_range.split(","))
                    if o.st_n_random_range
                    else None
                )
            ),
            n_random_frac=(None if whole else resolve_keep_frac(o)),
            mode=("gold_plus_random" if whole else o.st_keep_mode),
            n_gold=o.st_n_gold,
            seed=ctx.seed,
            mix_start_p=o.st_mix_start_p,
            mix_end_p=o.st_mix_end_p,
            mix_total_calls=int(ctx.total_calls * o.st_mix_anneal_frac),
            neighbour_runs=o.st_neighbour_runs,
        )
        if holder.n_attached == 0:
            raise SystemExit("[variants] softtoken: gold keep hook attached to nothing")
        print(
            f"[variants] softtoken: gold keep hook on {holder.n_attached} module(s), "
            f"{len(gold_table)} fingerprints from {sidecar}",
            flush=True,
        )

    def extra_callbacks(self, o, ctx):
        assert isinstance(o, SoftTokenOptions)
        if o.st_keep_token_rule != "online":
            return {}
        from olmo_core.train.callbacks.online_router import OnlineRouterCallback

        return {"online_router": OnlineRouterCallback()}

    def export_config(self, o):
        out: Dict[str, Any] = {}
        if o.st_keep_token_rule == "online" and o.st_otr_early_lora > 0:
            # the eval re-installs the adapters before loading (nested_ffn_moe.post_build_hook_from_config); a DCP load
            # only reads the keys the model asks for, so without this block the adapters would be silently dropped
            from olmo_core.nn.online_token_router import parse_mod_layers

            L = parse_mod_layers(o.st_otr_mod_layers)[0] if o.st_otr_mod_layers else o.st_otr_drop_layer
            out["early_lora"] = {"n_blocks": L, "rank": o.st_otr_early_lora, "scale": o.st_otr_early_lora_scale}
        return {
            **out,
            "softtoken": {
                "n_random": o.st_n_random,
                "n_random_range": o.st_n_random_range,
                "keep_frac": o.st_keep_frac,
                "keep_prob": o.st_keep_prob,
                "keep_mode": o.st_keep_mode,
                "gold_blind": o.st_gold_blind,
                "len_bias": o.st_len_bias,
                "mix_start_p": o.st_mix_start_p,
                "mix_end_p": o.st_mix_end_p,
                "mix_anneal_frac": o.st_mix_anneal_frac,
            }
        }


# ----------------------------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------------------------

VARIANTS: Dict[str, Variant] = {
    v.name: v
    for v in (
        Full(),
        Chunked(),
        ChunkedMix(),
        SparseLandmark(),
        PooledKV(),
        FFNMoE(),
        SoftToken(),
        KVRoute(),
        FlexCompute(),
        FFNDrop(),
        LayerDrop(),
    )
}
#: setA spells the dense arm ``dense``.
VARIANTS["dense"] = VARIANTS["full"]

#: Every export key, ``None`` unless an active variant fills it (train_ctc_suite's config.json).
EXPORT_KEYS = ("ffn_token_drop", "ffn_moe", "kv_route", "block_skip", "softtoken", "early_lora")


def get_variant(name: str) -> Variant:
    """:returns: The registered variant, or raises ``ValueError`` naming the valid ones."""
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; choose from {sorted(VARIANTS)}")
    return VARIANTS[name]


def setA_variant_names() -> List[str]:
    """:returns: The variants that run on the canonical setA recipe."""
    return sorted(n for n, v in VARIANTS.items() if v.setA_blocker is None)


#: Recipe-level keys ``--variant-opts`` may carry besides the variant's own options (setA):
#: ``total_steps`` (the optimizer-step horizon the schedules anneal over) and ``seed`` (router /
#: keep / drop seeds; train_ctc_suite's ``--seed``, default 0).
RECIPE_KEYS = ("total_steps", "seed")


def parse_variant_spec(name: str, spec: str) -> Tuple[Variant, VariantOptions, Dict[str, int]]:
    """
    Resolve ``--variant <name> --variant-opts <spec>`` for the setA recipe.

    :param name: The variant name (``dense`` == ``full``).
    :param spec: ``k=v,...`` (``;``-separated if a value holds commas).

    :returns: ``(variant, typed options, recipe keys)``; the recipe keys are ints from
        :data:`RECIPE_KEYS` that were given.

    :raises ValueError: Unknown variant / option, or an option combination it would ignore.
    """
    variant = get_variant(name)
    raw = parse_kv(spec)
    recipe = {k: int(raw.pop(k)) for k in RECIPE_KEYS if k in raw}
    opts = variant.options_from_dict(raw)
    variant.validate(opts)
    return variant, opts, recipe


def export_record(active: List[Tuple[Variant, VariantOptions]]) -> Dict[str, Any]:
    """The eval-side routing record for ``config.json`` (every :data:`EXPORT_KEYS` key present)."""
    out: Dict[str, Any] = {k: None for k in EXPORT_KEYS}
    for v, o in active:
        out.update(v.export_config(o))
    return out


def tolerant_base_load(base_checkpoint: str, model, save_folder: str) -> None:
    """Load ``base_checkpoint`` into ``model`` allowing ONLY the keys a variant adds (router,
    gains, projector), and REFUSE if any backbone key is missing (would silently train from init).

    DCP rejects a state dict asking for keys the checkpoint lacks before any ``strict`` option
    applies, so the state dict is FILTERED to the checkpoint's keys, loaded, and set back
    non-strictly; the new keys are then re-initialized deterministically. Resumes are unaffected:
    ``fit()`` loads a step checkpoint from the save folder afterwards if one exists.
    """
    import torch.distributed.checkpoint as dist_cp
    import torch.distributed.checkpoint.state_dict as dist_cp_sd

    from olmo_core.distributed.checkpoint import (
        RemoteFileSystemReader,
        _prepare_state_dict,
    )
    from olmo_core.utils import gc_cuda

    reader = RemoteFileSystemReader(base_checkpoint)
    ckpt_keys = set(reader.read_metadata().state_dict_metadata)
    state_dict = _prepare_state_dict(model, None)
    model_sd = state_dict["model"]
    missing = sorted(k for k in model_sd if f"model.{k}" not in ckpt_keys)
    allowed = (
        "._nffn_router.",
        "._nffn_gain",
        "._nffnp_",
        "._kvr_router.",
        "kvr_routers.",
        "bskip_routers.",
        "pooled_projector.",
        "._pooled_projector",
        ".lora_A",
        ".lora_B",
    )
    bad = [k for k in missing if not any(a in k for a in allowed)]
    if bad:
        raise SystemExit(
            f"[variants] base {base_checkpoint} lacks {len(bad)} backbone keys (first: {bad[:3]}); "
            "refusing to train from init"
        )
    filtered = {"model": {k: v for k, v in model_sd.items() if k not in set(missing)}}
    dist_cp.state_dict_loader.load(filtered, checkpoint_id=base_checkpoint, storage_reader=reader)
    dist_cp_sd.set_model_state_dict(
        model, filtered["model"], options=dist_cp_sd.StateDictOptions(strict=False)
    )
    gc_cuda()
    # New keys start from their DETERMINISTIC init: re-run each owner's reset for every missing
    # key (only missing keys -- a stage-2 warm start LOADS its router/gains from stage 1).
    reset_owners: dict = {}
    for k in missing:
        if "._nffn_router." in k:
            reset_owners.setdefault(k.split("._nffn_router.")[0], set()).add("router")
        elif "._nffn_gain" in k:
            reset_owners.setdefault(k.split("._nffn_gain")[0], set()).add("gain")
        elif "._nffnp_" in k:
            reset_owners.setdefault(k.split("._nffnp_")[0], set()).add("prefix")
        elif "._kvr_router." in k:
            reset_owners.setdefault(k.split("._kvr_router.")[0], set()).add("kv_router")
        elif k.startswith("bskip_routers."):
            reset_owners.setdefault(k.rsplit(".w.", 1)[0], set()).add("bskip_router")
        elif k.startswith("kvr_routers."):
            reset_owners.setdefault(k.rsplit(".w.", 1)[0], set()).add("kv_router")
        elif "pooled_projector" in k:
            reset_owners.setdefault(
                k.rsplit(".", 2)[0] if k.count(".") >= 2 else "pooled_projector", set()
            )
    stats = []
    for owner in sorted(reset_owners):
        mod = model.get_submodule(owner)
        if "kv_router" in reset_owners[owner]:
            from olmo_core.nn.attention.kv_route import reset_kv_route_extras

            reset_kv_route_extras(mod)
            stats.append(f"{owner}: kv router reset (keep-all)")
            continue
        if "bskip_router" in reset_owners[owner]:
            from olmo_core.nn.block_skip import reset_block_skip_extras

            reset_block_skip_extras(mod)
            stats.append(f"{owner}: block-skip router reset (run-all)")
            continue
        if hasattr(mod, "_nffn_gain"):
            from olmo_core.nn.nested_ffn_moe import reset_nested_ffn_extras

            reset_nested_ffn_extras(mod, parts=reset_owners[owner])
            # Under FSDP these are DTensors; one rank's local shard can be EMPTY (.min() raises
            # there and hangs every other rank). Gather before reading.
            g = mod._nffn_gain.detach()
            b = mod._nffn_router.w.bias.detach()
            g = g.full_tensor() if hasattr(g, "full_tensor") else g
            b = b.full_tensor() if hasattr(b, "full_tensor") else b
            g, b = g.float(), b.float()
            stats.append(
                f"{owner}: gain={g.min().item():.2f}..{g.max().item():.2f} bias0={b[0].item():.1f}"
            )
        elif hasattr(mod, "reset_parameters"):
            mod.reset_parameters()
            stats.append(f"{owner}: reset_parameters()")
    print(
        f"[variants] tolerant base load from {base_checkpoint}: {len(filtered['model'])} keys "
        f"loaded, {len(missing)} new keys re-initialized deterministically (e.g. {missing[:2]}); "
        f"{stats[:2]}{' ...' if len(stats) > 2 else ''}",
        flush=True,
    )


def options_to_kv(o: VariantOptions) -> str:
    """Inverse of :meth:`Variant.options_from_kv` for the NON-default fields (``;``-joined)."""
    default = type(o)()
    parts = [
        f"{f.name}={getattr(o, f.name)}"
        for f in dataclasses.fields(o)
        if getattr(o, f.name) != getattr(default, f.name)
    ]
    return ";".join(parts)


def options_as_dict(o: VariantOptions) -> Mapping[str, Any]:
    return dataclasses.asdict(o)
