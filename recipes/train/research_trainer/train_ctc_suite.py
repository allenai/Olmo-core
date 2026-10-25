"""Task-agnostic LOCAL (torchrun, no Beaker/weka) SFT trainer for the CTC suite experiment
(``records/ctc-suite-scaling-plan.md`` §4/§11): one joint run per (task, arm) on the hybrid
Qwen3.5 models (Gated DeltaNet + full attention, 3:1), full vs document-chunked arms.

Generalized from ``attn_explore/Qwen3.5-0.8B-docchunk-mask-mix-contradiction-SFT-local.py`` +
the curriculum/hard-fail logic of its Qwen3-0.6B sibling. What it keeps:

  * distcp base loading (``load_path`` -> the shared per-scale converted base; model-only), with a
    ``.metadata`` guard so a mis-staged base cannot silently train from scratch;
  * the mask-mix curriculum derivation with BOTH divisor fixes (world_size AND
    micro-batch-instances) and the hard-fail check that the anneal actually lands on
    ``mix_end_p`` (see the mask-mix-ngpu-anneal bug);
  * ``metadata.json`` as a hard requirement of the shard dir (``num_instances`` drives the
    curriculum; ``max_example_len`` guards ``--seq-len``; ``marker_set`` is cross-checked).

Arms (``--variant``):

  * ``full``        -> plain causal attention (no ``document_chunk_attention``); the box markers in
                       the shard are just ordinary tokens. Upper-bound / O(N^2) arm.
  * ``chunked``     -> document-chunked mask on the full-attention blocks (GDN blocks are linear
                       attention and ignore ``chunk_ids``), no mixing. Eval-matched arm.
  * ``chunked-mix`` -> chunked + curriculum mask mixing 0.8 -> 0.0 (the project default recipe;
                       eval still uses the pure chunked mask).

Marker ids come from ``RESERVED_IDS["qwen3_5"]`` in :mod:`olmo_core.data.document_chunk_landmark`
-- never retyped. The shard must be built with ``--marker-set qwen3_5``
(``convert_unified_to_document_landmark.py``); this script hard-fails on a mismatch.

CPU-checkable dry run (no CUDA, nothing written)::

    PYTHONPATH=src python recipes/train/research_trainer/train_ctc_suite.py \\
      --task contradiction --data <shard-dir> --variant chunked-mix --model-scale 0.8b \\
      --dry-run --dry-run-world-size 8 --dry-run-n-examples 20000

Real run (via ``run_ctc_local.sbatch``; ``--variant chunked-mix`` forces ``--no-compile``)::

    torchrun --nproc_per_node=8 recipes/train/research_trainer/train_ctc_suite.py \\
      --task contradiction --data <shard-dir> --variant chunked-mix --model-scale 0.8b \\
      --run-name ctc-contra-chunkedmix-08b --save-checkpoint --no-compile
"""

import argparse
import datetime
import functools
import json
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    PackingInstanceSourceConfig,
    PadToLengthInstanceSourceConfig,
)
from olmo_core.data.document_chunk_landmark import (
    RESERVED_IDS,  # canonical ids -- never retype
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.attention.chunked_mask import mask_mix_standard_prob
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import WSD, LinearWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    LoadStrategy,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    FlopMeterCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

try:  # package import (repo root on sys.path) or same-directory fallback (torchrun on the file path)
    from recipes.train.research_trainer.llama_configs import (
        LLAMA_MARKER_TOKENIZER,
        llama3_1_8B,
        llama3_2_3B,
    )
except ImportError:  # pragma: no cover
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from llama_configs import (  # type: ignore[no-redef]
        LLAMA_MARKER_TOKENIZER,
        llama3_1_8B,
        llama3_2_3B,
    )

try:  # package import (repo root on sys.path) or same-directory fallback (torchrun on the file path)
    from recipes.train.research_trainer.olmo3_configs import (
        OLMO3_MARKER_TOKENIZER,
        OLMO3_VOCAB_SIZE,
        olmo3_7B_ctc,
        olmo3_7B_ctc_swa,
    )
    from recipes.train.research_trainer.olmo_hybrid_configs import (
        OLMO_HYBRID_MARKER_TOKENIZER,
        OLMO_HYBRID_VOCAB_SIZE,
        olmo_hybrid_7B_ctc,
    )
except ImportError:  # pragma: no cover
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from olmo3_configs import (  # type: ignore[no-redef]
        OLMO3_MARKER_TOKENIZER,
        OLMO3_VOCAB_SIZE,
        olmo3_7B_ctc,
        olmo3_7B_ctc_swa,
    )
    from olmo_hybrid_configs import (  # type: ignore[no-redef]
        OLMO_HYBRID_MARKER_TOKENIZER,
        OLMO_HYBRID_VOCAB_SIZE,
        olmo_hybrid_7B_ctc,
    )

try:  # the per-variant logic lives in recipes/train/variants.py (shared with the setA recipe)
    from recipes.train import variants as V
except ImportError:  # pragma: no cover
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import variants as V  # type: ignore[no-redef]

try:  # start-of-run guards shared with the setA recipe (recipes/train/run_guards.py)
    from recipes.train import run_guards as RG
except ImportError:  # pragma: no cover
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import run_guards as RG  # type: ignore[no-redef]

#: Supported model families. The trainer is family-agnostic: everything below (marker ids,
#: embedding size, tokenizer, model factory) is keyed on the family, which is auto-detected from
#: the shard's ``marker_set`` (or forced via ``--model-family``). ``qwen3_5`` is the GDN+attn
#: hybrid; ``qwen3`` is the plain dense/causal family. A shard built for one family MUST train with
#: that family -- a wrong-tokenizer shard produces plausible numbers, not a crash.
#: Per-family embedding-matrix size (rows), i.e. the converted base checkpoint's embedding size.
FAMILY_VOCAB_SIZE = {
    "qwen3_5": 248320,  # Qwen3.5-{0.8B,4B,9B}-Base embedding size
    "qwen3": 151936,  # Qwen3-{0.6B,1.7B,4B,8B}-Base embedding size
    "gemma": 262208,  # Gemma-3-{1B,4B,12B,27B} embedding size (262144 real ids + 64 padding rows)
    "llama": 128256,  # Llama-3.x embedding size (no padding rows: vocab == matrix rows)
    "olmo3": OLMO3_VOCAB_SIZE,  # dolma2 padded embedding size (100278 real ids + 74 padding rows)
}

#: Per-family HF tokenizer identifier (all sizes within a family share one tokenizer).
FAMILY_TOKENIZER = {
    "qwen3_5": "Qwen/Qwen3.5-0.8B-Base",
    "qwen3": "Qwen/Qwen3-4B",
    "gemma": "google/gemma-3-4b-pt",
    # Llama 3 has no ``<|box_start|>``/``<|box_end|>`` tokens, so the suite uses a local tokenizer
    # copy in which reserved slots 128002/128003 are RENAMED to those strings (ids unchanged) --
    # built by ``src/scripts/data/make_llama_marker_tokenizer.py``. Point at the patched copy, not
    # the stock repo, or the converter's marker-id verification fails.
    "llama": LLAMA_MARKER_TOKENIZER,
    # Same story for OLMo: dolma2 has no ``<|box_start|>``/``<|box_end|>``, so the suite uses a
    # local copy in which the unused ``<|extra_id_1|>``/``<|extra_id_2|>`` slots (100266/100267)
    # are RENAMED to those strings, ids unchanged.
    "olmo3": OLMO3_MARKER_TOKENIZER,
}

#: Per-family model factories, keyed by ``--model-scale``.
MODEL_FACTORIES = {
    "qwen3_5": {
        "0.8b": TransformerConfig.qwen3_5_0_8B,
        "2b": TransformerConfig.qwen3_5_2B,
        "4b": TransformerConfig.qwen3_5_4B,
        "9b": TransformerConfig.qwen3_5_9B,
        "27b": TransformerConfig.qwen3_5_27B,
    },
    "qwen3": {
        "0.6b": TransformerConfig.qwen3_0_6B,
        "1.7b": TransformerConfig.qwen3_1_7B,
        "4b": TransformerConfig.qwen3_4B,
        "8b": TransformerConfig.qwen3_8B,
    },
    # Gemma 3 (local/global sliding-window hybrid). ``gemma3_4B``'s factory defaults do NOT match
    # the released ``google/gemma-3-4b-pt`` checkpoint (8 query heads / head_dim 256, and the
    # global layers carry HF ``rope_scaling={"rope_type":"linear","factor":8}``), so the exact HF
    # dims are pinned here rather than inherited -- a silently mismatched shape loads the weights
    # wrong and produces plausible garbage instead of crashing.
    "gemma": {
        "4b": functools.partial(
            TransformerConfig.gemma3_4B,
            n_heads=8,
            head_dim=256,
            global_rope_linear_scaling_factor=8.0,
        ),
    },
    # Llama 3.x (plain dense/causal, GQA). olmo-core ships no 3B factory, so ``llama_configs``
    # builds one with the generic ``llama_like`` using dims read off ``meta-llama/Llama-3.2-3B``'s
    # own config.json, and hard-asserts the resulting parameter count against it. 8b maps onto
    # olmo-core's own ``llama3_8B`` but still overrides ``rope_scaling``: the class default factor
    # (32.0) is Llama-3.2's, and Llama-3.1-8B's is 8.0 -- see llama_configs.LLAMA3_1_8B_HF_SHAPE.
    # 8b is the SIZE CONTROL for the 3b contradiction result (0.362 at 2k vs ~0.83 for Qwen3.5-4B
    # and OLMo-3-7B): until a bigger Llama is measured on the identical task, "Llama is weak at
    # N^2 pair-finding" and "3B is too small for it" are indistinguishable.
    "llama": {
        "3b": llama3_2_3B,
        "8b": llama3_1_8B,
    },
    # OLMo 3 (``allenai/Olmo-3-1025-7B``), this repo's native family. ``olmo3_configs`` wraps the
    # stock ``olmo3_7B`` factory to disable sliding-window attention (DocumentChunkedAttention
    # refuses it, so the chunked arm cannot have it -- disabled in BOTH arms to keep the mask the
    # only manipulated variable) and to apply the checkpoint's YaRN scaling to every layer.
    "olmo3": {
        # Keeps Olmo 3's native 3:1 sliding:full backbone and chunks ONLY the 8 full-attention
        # layers -- the exact counterpart of chunking Qwen3.5's non-GDN blocks.
        "7b": olmo3_7B_ctc_swa,
        # Superseded no-sliding-window variant: disabling the windows costs the base model ~41x CE
        # before training (olmo3_swa_ablation.py). Kept so the first wave of runs reproduces.
        "7b-noswa": olmo3_7B_ctc,
        # ``allenai/Olmo-Hybrid-7B``: same size and same training data as Olmo-3-1025-7B, but a 3:1
        # LINEAR (Gated-DeltaNet):full backbone instead of 3:1 sliding:full. It belongs here rather
        # than in a family of its own because it shares dolma2, the patched marker tokenizer, the
        # marker ids and the 100352-row embedding -- so it trains on the SAME olmo3 shards, and the
        # shard's ``marker_set`` cross-check passes without an exemption. That shared everything is
        # the whole point: pairing 7b against 7b-hybrid varies the attention backbone and nothing
        # else, which no other entry in this table can do. Chunks the 8 full-attention layers only,
        # exactly as ``olmo3_7B_ctc_swa`` and the Qwen3.5 arms do.
        "7b-hybrid": olmo_hybrid_7B_ctc,
    },
}


def resolve_family(opts: argparse.Namespace) -> str:
    """Resolve the model family: explicit ``--model-family`` or auto-detect from the shard.

    Auto-detection reads ``metadata.json:marker_set`` (written by
    ``convert_unified_to_document_landmark.py --marker-set ...``), so the tokenizer that built the
    shard picks the model/marker ids that train on it -- they cannot silently mismatch. Falls back
    to ``qwen3_5`` only when the shard predates the ``marker_set`` field (all current shards write
    it), preserving the original Qwen3.5-only behavior.

    :param opts: Parsed CLI options (uses ``model_family`` and ``data``).

    :returns: A key of :data:`MODEL_FACTORIES` (``"qwen3_5"`` or ``"qwen3"``).

    :raises SystemExit: If the resolved family is unknown.
    """
    fam = opts.model_family
    if fam == "auto":
        fam = "qwen3_5"
        meta_path = os.path.join(opts.data, "metadata.json")
        if os.path.exists(meta_path):
            ms = json.load(open(meta_path)).get("marker_set")
            if ms:
                fam = ms
    if fam not in MODEL_FACTORIES:
        raise SystemExit(
            f"unknown model family {fam!r}; known: {sorted(MODEL_FACTORIES)}. "
            "Pass --model-family explicitly or rebuild the shard with a known --marker-set."
        )
    return fam


# Per-scale converted olmo distcp bases (model-only). Only 0.8B exists today; 4B/9B must be passed
# explicitly (--base-checkpoint / BASE_SRC) once converted -- repeat the marker audit first (§4).
DEFAULT_BASE_CHECKPOINT = {
    "0.8b": "/scratch/users/prasann/cpt_mix_ckpts/q35-08b-base-modelonly/model_and_optim",
}
SAVE_ROOT = "/scratch/users/prasann/olmo_ckpts"
WORK_DIR = "/scratch/users/prasann/longctx_sft_qwen/dataset-cache-ctc-suite"

# Hyperparams inherited from the source attn_explore scripts (NOT contradiction-specific).
LR = 5e-5
MEM_FREQ_SPARSE = V.MEM_FREQ_SPARSE  # 63 content + 1 landmark = 64-token block
NUM_EPOCHS = 3
WANDB_PROJECT = "memory-networks"


def read_shard_metadata(data_dir: str, seq_len: int, family: str) -> Dict[str, Any]:
    """Read and validate the shard's ``metadata.json``.

    :param data_dir: Shard dir from ``convert_unified_to_document_landmark.py``.
    :param seq_len: The training sequence length, checked against ``max_example_len``.
    :param family: The resolved model family; the shard's ``marker_set`` must match it.

    :returns: The parsed metadata dict.

    :raises SystemExit: If the metadata is missing, the marker set disagrees with ``family``, or
        ``seq_len`` is too short for the shard.
    """
    ids = RESERVED_IDS[family]
    meta_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise SystemExit(
            f"{meta_path} missing. The docchunk trainers REQUIRE the converter's metadata.json "
            "(num_instances drives the mask-mix anneal; max_example_len guards --seq-len)."
        )
    meta = json.load(open(meta_path))
    shard_marker_set = meta.get("marker_set")
    if shard_marker_set is not None and shard_marker_set != family:
        raise SystemExit(
            f"shard {data_dir} was built with --marker-set {shard_marker_set!r}, but this run "
            f"resolved to family {family!r}. A wrong-tokenizer shard produces plausible numbers, "
            "not a crash -- rebuild the shard or pass the matching --model-family, do not override."
        )
    for key, want in [("doc_start_id", ids.doc_start), ("doc_end_id", ids.doc_end)]:
        got = meta.get(key)
        if got is not None and int(got) != want:
            raise SystemExit(f"shard metadata {key}={got} != canonical {want} ({family})")
    if meta.get("max_example_len", 0) > seq_len:
        raise SystemExit(
            f"--seq-len={seq_len} < max example length {meta['max_example_len']} "
            f"(PadToLength would SKIP the long examples). Raise --seq-len."
        )
    return meta


def derive_batch_geometry(global_batch: int, world_size: int, micro_batch_instances: int) -> int:
    """Validate the batch geometry and return instances-per-rank-per-step.

    :param global_batch: Instances per optimizer step (across all ranks).
    :param world_size: Number of DP ranks.
    :param micro_batch_instances: Instances per rank per FORWARD.

    :returns: ``global_batch // world_size`` (each rank's share of a step).

    :raises SystemExit: If a rank's share is < 1 or ``micro_batch_instances`` exceeds it --
        olmo-core would otherwise silently reinterpret the batch (quiet optimization drift).
    """
    per_rank = global_batch // world_size
    if per_rank < 1 or micro_batch_instances > per_rank:
        raise SystemExit(
            f"--micro-batch-instances={micro_batch_instances} invalid: with --global-batch="
            f"{global_batch} and world_size={world_size} each rank holds {per_rank} instance(s). "
            f"Use --global-batch >= world_size and --micro-batch-instances in [1, {per_rank}]."
        )
    return per_rank


def derive_mask_mix_curriculum(
    *,
    n_examples: int,
    epochs: int,
    global_batch: int,
    world_size: int,
    micro_batch_instances: int,
    mix_start_p: float,
    mix_end_p: float,
) -> Dict[str, Any]:
    """Derive the curriculum's ``mix_total_forwards`` and hard-check the anneal lands.

    ``mix_total_forwards`` must be the number of FORWARDS a rank actually performs, because that
    is what the curriculum's counter counts. Two divisors, both learned the hard way:

      * ``world_size`` -- data is sharded across DP ranks (the original mask-mix-ngpu-anneal bug:
        p stalled at ``mix_start_p * (1 - 1/world_size)``, worse the more GPUs).
      * ``micro_batch_instances`` -- a forward carries this many instances, so raising it CUTS the
        forward count by the same factor (observed 2026-07-16: p ended at 0.601 instead of 0.0).

    The invariant: forwards_per_rank == optimizer steps * (per-rank instances / micro-batch).

    :param n_examples: Instances in the shard (``metadata.json: num_instances``).
    :param epochs: Training epochs.
    :param global_batch: Instances per optimizer step.
    :param world_size: Number of DP ranks.
    :param micro_batch_instances: Instances per rank per forward.
    :param mix_start_p: Curriculum start collapse probability.
    :param mix_end_p: Curriculum end collapse probability.

    :returns: Dict with ``mix_total_forwards``, ``steps``, ``forwards_per_rank``, ``final_p``.

    :raises SystemExit: If the predicted final ``p_standard`` misses ``mix_end_p`` by > 0.01 --
        a silent shortfall does not crash and does not look wrong in the loss, it just means the
        model never trained predominantly on its target mask, voiding the run.
    """
    per_rank = derive_batch_geometry(global_batch, world_size, micro_batch_instances)
    mix_total_forwards = max(1, (n_examples * epochs) // (world_size * micro_batch_instances))
    # Independently re-derive the forward count from the STEP count and assert the curriculum
    # will actually land on mix_end_p, instead of trusting one formula to agree with itself.
    steps = max(1, (n_examples * epochs) // global_batch)
    forwards_per_rank = steps * (per_rank // micro_batch_instances)
    final_p = mask_mix_standard_prob(
        forwards_per_rank - 1,
        mix_start_p=mix_start_p,
        mix_end_p=mix_end_p,
        mix_total_forwards=mix_total_forwards,
    )
    # Tolerance 0.01, not 1e-3: integer-division rounding legitimately leaves the last forward a
    # hair off. We are catching gross shortfalls like the observed 0.601, not rounding dust.
    if abs(final_p - mix_end_p) > 0.01:
        raise SystemExit(
            f"mask-mix curriculum would end at p_standard={final_p:.4f}, not mix_end_p="
            f"{mix_end_p}: mix_total_forwards={mix_total_forwards} disagrees with the "
            f"{forwards_per_rank} forwards this rank will actually run. The model would keep "
            f"training on PLAIN CAUSAL for {final_p:.0%} of forwards forever, silently voiding "
            "the full-vs-chunked comparison. Check the world_size / micro_batch_instances "
            "divisors."
        )
    return dict(
        mix_total_forwards=mix_total_forwards,
        steps=steps,
        forwards_per_rank=forwards_per_rank,
        final_p=final_p,
    )


def variant_and_options(opts: argparse.Namespace) -> Tuple["V.Variant", "V.VariantOptions"]:
    """The ``--variant`` arm from :mod:`recipes.train.variants` with its typed options, read off the
    argparse namespace (the option fields are the argparse dest names).

    :param opts: Parsed CLI options.

    :returns: ``(variant, options)``.
    """
    variant = V.get_variant(opts.variant)
    return variant, variant.options_from_namespace(opts)


def active_variants(opts: argparse.Namespace) -> List[Tuple["V.Variant", "V.VariantOptions"]]:
    """Every variant whose hooks run, in hook order: the ``--ffn-drop-*`` modifier (drop-CPT, any
    non-routed arm) first, then the ``--variant`` arm, then the ``--train-skip-layers`` /
    ``--layer-drop-*`` modifier (whole-block skipping, any arm).

    :param opts: Parsed CLI options.

    :returns: ``[(variant, options), ...]``.
    """
    out = []
    drop = V.VARIANTS["ffndrop"]
    drop_opts = drop.options_from_namespace(opts)
    if drop_opts.active:  # type: ignore[attr-defined]
        out.append((drop, drop_opts))
    out.append(variant_and_options(opts))
    ldrop = V.VARIANTS["layerdrop"]
    ldrop_opts = ldrop.options_from_namespace(opts)
    if ldrop_opts.active:  # type: ignore[attr-defined]
        out.append((ldrop, ldrop_opts))
    return out


def variant_context(opts: argparse.Namespace, **kw: Any) -> "V.VariantContext":
    """The :class:`~recipes.train.variants.VariantContext` for this run.

    :param opts: Parsed CLI options.
    :param kw: Run facts known only at the call site (``total_calls``, ``calls_per_step``,
        ``rank``, ``meta``, ``mix_total_forwards``).
    """
    return V.VariantContext(
        model_family=opts.model_family,
        seed=opts.seed,
        seq_len=opts.seq_len,
        vocab_size=opts.vocab_size or 0,
        data_dir=opts.data,
        tokenizer_id=opts.st_slot_tokenizer or FAMILY_TOKENIZER.get(opts.model_family),
        **kw,
    )


def build_model_config(opts: argparse.Namespace) -> TransformerConfig:
    """Build the per-scale :class:`TransformerConfig` for the requested family/variant.

    :param opts: Parsed CLI options (``model_family``, ``model_scale``, ``variant``, mix knobs).

    :returns: The model config, with ``document_chunk_attention`` set for the chunked arms.
    """
    factory = MODEL_FACTORIES[opts.model_family][opts.model_scale]
    # Pin flash_2 by default so the saved config.json supports KV-cached generation at eval (the
    # qwen3_5 factories default attn_backend=None otherwise; matches the source attn_explore
    # scripts). ``--attn-backend torch`` is an escape hatch for clusters where flash-attn isn't
    # importable/compatible (e.g. a fresh env on an unverified GPU arch) -- SDPA works everywhere
    # but does not support KV-cached generation, so eval on a torch-backend checkpoint needs the
    # full-precompute path, not incremental decoding.
    qwen_kwargs: Dict[str, Any] = dict(
        vocab_size=opts.vocab_size, attn_backend=AttentionBackendName(opts.attn_backend)
    )
    # RoPE theta override (NTK-style context extension): raise the base rope_theta so the model's
    # RoPE frequencies cover the longer context. A plain factory param (applies to the base block,
    # no block_overrides), so it unambiguously takes effect at runtime -- unlike YaRN, whose
    # with_rope_scaling path writes per-layer overrides. For Qwen3-4B (native 32k, theta 1M) at
    # 256k, the NTK-aware value is ~8M. Off by default (0). Prefer this over --rope-yarn-factor
    # for large (>=8x) extensions where YaRN was found to plateau.
    if opts.rope_theta and opts.rope_theta > 0:
        qwen_kwargs["rope_theta"] = opts.rope_theta
    # Per-variant model changes live in recipes/train/variants.py (shared with the setA recipe):
    # factory kwargs (chunked: document_chunked + cross_doc_mode) and the config mutations (mixer
    # swap for sparselandmark / pooledkv, ``document_chunk_attention`` for the chunked family).
    # Applied BEFORE the YaRN copy below so per-layer rope overrides see the swapped mixer.
    variant, vopts = variant_and_options(opts)
    qwen_kwargs.update(variant.factory_kwargs(vopts))
    # OLMo 3 7B applies its own YaRN inside the factory (default x8, the released stage-3 config),
    # so --rope-yarn-factor is passed THROUGH to it instead of stacking a second with_rope_scaling
    # below (olmo3_cptdrop: stage-2 checkpoint, x4 for 32k).
    if opts.model_family == "olmo3" and opts.rope_yarn_factor:
        if factory is not olmo3_7B_ctc_swa:
            raise SystemExit(
                f"--rope-yarn-factor is only wired for olmo3 scale '7b' (got {opts.model_scale!r})"
            )
        qwen_kwargs["yarn_factor"] = float(opts.rope_yarn_factor)
    model_config = factory(**qwen_kwargs)
    variant.apply_model_config(
        model_config, vopts, variant_context(opts, mix_total_forwards=opts._mix_total_forwards)
    )
    # YaRN RoPE context extension for long-context runs (base native ctx is 32k for Qwen3; the
    # 256k rung needs factor ~8). Off by default (--rope-yarn-factor 0) so short-rung runs are
    # byte-identical. Mirrors sft_longctx/Qwen3-4B-dense-longctx-SFT.py (factor=2 for 64k).
    if opts.rope_yarn_factor and opts.rope_yarn_factor > 1 and opts.model_family != "olmo3":
        if opts.model_family == "qwen3_5":
            # with_rope_scaling refuses hybrid (named-block) models; the GDN-hybrid needs a
            # per-block/theta-based extension instead. Skip here and warn -- the hybrid's RoPE
            # extension to >native is a separate (open) modeling choice; the SFT still adapts.
            print(
                "[ctc-suite] WARNING: --rope-yarn-factor ignored for qwen3_5 hybrid "
                "(with_rope_scaling unsupported for named-block models); training at native RoPE.",
                flush=True,
            )
        else:
            model_config = model_config.with_rope_scaling(
                YaRNRoPEScalingConfig(
                    factor=float(opts.rope_yarn_factor),
                    beta_fast=32,
                    beta_slow=1,
                    old_context_len=opts.rope_old_context,
                )
            )
    # Fused-linear CE: never materialize float logits over the full 248k vocab. At seq-len
    # 40960 the unfused path needs ~38 GiB per rank just for logits.float() and OOMs H200s
    # (same setting as the proven 40k-seq sft_docchunk Beaker scripts).
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear
    # Run the head only where the loss is defined. Our SFT shards supervise 0.2-5% of positions,
    # so the dense head throws away most of its work; the head is also the largest term no
    # compute router touches, which is what caps total routed speedup (see research/kv_route/flexcompute_40x).
    model_config.lm_head.sparse_loss_positions = opts.sparse_loss_positions
    return model_config


#: Scales that need FULL parameter sharding + activation checkpointing to fit 40960 on an 80GB
#: H100 (proven OOM on jupiter: 76.71 GiB allocated then OOM in the first dry-run batch at
#: shard_degree=1 / no AC). 0.8B already fits at shard_degree=1 / no AC on 141GB H200s -- keep it
#: on that path so its proven runs don't change. 2B is grouped with the large scales: it is 2.4B
#: params and the model-scale study runs it on 80GB A100s, where even 0.8B needs full sharding +
#: AC at 40960 (lambda_cluster.md).
_LARGE_SCALES = ("2b", "4b", "9b", "27b")


def kv_route_enabled(opts: argparse.Namespace) -> bool:
    """Whether the learned KV keep/drop router runs (``kvroute``; ``flexcompute`` unless
    ``--kv-route-off``). Thin wrapper over :mod:`recipes.train.variants`."""
    variant, vopts = variant_and_options(opts)
    return bool(getattr(variant, "kv_route_enabled", lambda o: False)(vopts))


def resolve_keep_frac(opts: argparse.Namespace) -> Optional[float]:
    """Soft-token non-gold keep fraction; see :func:`recipes.train.variants.resolve_keep_frac`."""
    return V.resolve_keep_frac(V.VARIANTS["softtoken"].options_from_namespace(opts))


def resolve_cat_keep(opts: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Whole-category keep config; see :func:`recipes.train.variants.resolve_cat_keep`."""
    return V.resolve_cat_keep(V.VARIANTS["softtoken"].options_from_namespace(opts))


def resolve_activation_checkpointing(opts: argparse.Namespace) -> str:
    """Resolve ``--activation-checkpointing`` ("auto" -> scale-dependent default).

    :param opts: Parsed CLI options (``model_scale``, ``activation_checkpointing``).

    :returns: ``"full"`` or ``"none"``.
    """
    if opts.activation_checkpointing != "auto":
        return opts.activation_checkpointing
    return "full" if opts.model_scale in _LARGE_SCALES else "none"


def resolve_shard_degree(opts: argparse.Namespace, world_size: int) -> int:
    """Resolve ``--shard-degree`` ("auto" -> scale-dependent default).

    0.8B keeps its proven ``shard_degree=1`` default (single-GPU-resident params fit 141GB
    H200s); 4B/9B default to full FSDP over ``world_size`` (matches the proven 40k-seq docchunk
    pattern in ``_docchunk_5task_32k_nocpt_common.py``, needed to fit an 80GB H100).

    :param opts: Parsed CLI options (``model_scale``, ``shard_degree``).
    :param world_size: DP world size (real, or hypothetical under ``--dry-run``).

    :returns: The resolved ``shard_degree``.
    """
    if opts.shard_degree:
        return opts.shard_degree
    return world_size if opts.model_scale in _LARGE_SCALES else 1


def _build_ac_config(ac_mode: str, ac_budget: float):
    """Build the activation-checkpointing config for the resolved mode.

    ``full`` = checkpoint every block (most memory-frugal, but breaks under compile+CP -- see
    build_train_module_config). ``budget`` = FLOP-optimal checkpointing to hit ``ac_budget`` (the
    compile+CP-safe path, from the proven long-context recipe). ``none`` = no AC.
    """
    if ac_mode == "full":
        return TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full
        )
    if ac_mode == "budget":
        return TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=ac_budget,
        )
    return None


def build_scheduler(opts: argparse.Namespace):
    """LR schedule from ``--scheduler`` (default: linear warmup then linear decay to 0).

    ``wsd`` = warmup, constant, then linear decay to 0 over the last ``--wsd-decay-fraction`` of
    ``max_duration``. With ``--save-interval`` at 0.9 x a smaller budget, that run's checkpoint is
    still at peak LR, and ``--branch-from`` it with the smaller ``--max-tokens`` gives that budget's
    decayed point (olmo3_cptdrop decay branches).
    """
    if opts.scheduler == "wsd":
        return WSD(warmup_fraction=opts.warmup_fraction, decay_fraction=opts.wsd_decay_fraction)
    return LinearWithWarmup(warmup_fraction=opts.warmup_fraction, alpha_f=0.0)


def build_train_module_config(
    opts: argparse.Namespace, world_size: int
) -> TransformerTrainModuleConfig:
    """Build the :class:`TransformerTrainModuleConfig` (FSDP + activation checkpointing).

    Shared by ``dry_run`` (to print the resolved config) and ``build_and_fit`` so the two never
    drift apart.

    :param opts: Parsed CLI options.
    :param world_size: DP world size (real, or hypothetical under ``--dry-run``).

    :returns: The train-module config, unbuilt (no CUDA needed to construct it).
    """
    ac_mode = resolve_activation_checkpointing(opts)
    # Context parallelism (long-context arms): CP shards the SEQUENCE/activations across cp_degree
    # ranks; params/optimizer are sharded by the FSDP dp dimension, which is world_size//cp_degree.
    # Ulysses only (ring flash-attn isn't guaranteed in the image, and GDN blocks reject ring), so
    # cp_degree is bounded by n_kv_heads (dense=8, hybrid=4). Off by default (--cp-degree 0/1).
    cp_degree = opts.cp_degree if opts.cp_degree and opts.cp_degree > 1 else 1
    if world_size % cp_degree != 0:
        raise SystemExit(f"world_size {world_size} not divisible by --cp-degree {cp_degree}")
    dp_world_size = world_size // cp_degree
    shard_degree = resolve_shard_degree(opts, dp_world_size)
    cp_config = (
        TransformerContextParallelConfig.ulysses(degree=cp_degree) if cp_degree > 1 else None
    )
    variant, vopts = variant_and_options(opts)
    tm_overrides = variant.train_module_overrides(vopts, variant_context(opts))
    assert set(tm_overrides) <= {"microbatch_sort_pad_id"}, tm_overrides
    return TransformerTrainModuleConfig(
        rank_microbatch_size=opts.micro_batch_instances * opts.seq_len,
        # length-homogeneous micro-batches (softtoken); variants.Variant.train_module_overrides
        microbatch_sort_pad_id=tm_overrides.get("microbatch_sort_pad_id"),
        max_sequence_length=opts.seq_len,
        optim=SkipStepAdamWConfig(
            lr=opts.lr,
            weight_decay=opts.weight_decay,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ]
            # router LR groups (ffnmoe / kvroute / flexcompute): variants.Variant.optim_group_overrides
            + variant.optim_group_overrides(vopts),
        ),
        scheduler=build_scheduler(opts),
        compile_model=opts.compile,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp if cp_degree > 1 else DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=shard_degree,
        ),
        cp_config=cp_config,
        # Activation checkpointing. FULL-block AC fits 4B/9B at seq_len=40960 on 80GB H100s.
        # BUT full-block AC is INCOMPATIBLE with torch.compile + Ulysses CP: the block boundary
        # straddles the CP all-to-all, so the recomputed activation has a different shape/dtype
        # than the saved one ("recomputed metadata != saved metadata" crash at the dry-run). The
        # proven long-context CP recipe (sft_longctx/Qwen3-4B-dense-longctx-SFT.py) uses BUDGET AC
        # instead, which places checkpoints FLOP-optimally and avoids the all-to-all boundary.
        ac_config=_build_ac_config(ac_mode, opts.ac_budget),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=opts.z_loss,
        max_grad_norm=1.0,
    )


def build_provenance(opts: argparse.Namespace, world_size: int) -> Dict[str, Any]:
    """Assemble the run's ``provenance.json`` payload (results-hub-ingestible; never fabricated).

    :param opts: Parsed CLI options.
    :param world_size: DP world size of the run.

    :returns: The provenance dict.
    """
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        git_commit = "unknown"
    return {
        "experiment": "ctc_suite",
        "script": os.path.abspath(__file__),
        "git_commit": git_commit or "unknown",
        "task": opts.task,
        "variant": opts.variant,
        "model_family": opts.model_family,
        "model_scale": opts.model_scale,
        "data": os.path.abspath(opts.data),
        "marker_set": {"family": opts.model_family, **RESERVED_IDS[opts.model_family]._asdict()},
        "world_size": world_size,
        "start_time": os.environ.get("CTC_LAUNCH_TS")
        or datetime.datetime.now().isoformat(timespec="seconds"),
        "args": {k: v for k, v in vars(opts).items() if not k.startswith("_")},
    }


def announce_wandb(opts: argparse.Namespace) -> None:
    """Print the wandb group URL at launch (project directive: always surface it)."""
    if not opts.wandb:
        return
    group = opts.wandb_group or opts.run_name
    entity = opts.wandb_entity or os.environ.get("WANDB_ENTITY")
    if entity:
        print(
            f"[ctc-suite] wandb group: https://wandb.ai/{entity}/{WANDB_PROJECT}/groups/{group}",
            flush=True,
        )
    else:
        print(
            f"[ctc-suite] wandb project={WANDB_PROJECT} group={group} "
            "(set --wandb-entity / WANDB_ENTITY for a direct group URL)",
            flush=True,
        )


def resolve_plan(opts: argparse.Namespace, world_size: int) -> Dict[str, Any]:
    """Resolve everything derivable without CUDA: metadata, batch geometry, curriculum, configs.

    :param opts: Parsed CLI options. ``opts._mix_total_forwards`` is filled as a side effect for
        the chunked-mix variant.
    :param world_size: DP world size (real, or hypothetical under ``--dry-run``).

    :returns: Dict with ``meta``, ``n_examples``, ``per_rank``, ``curriculum`` (or None),
        ``model_config``, ``instance_source_config``, ``data_loader_config``.
    """
    ids = RESERVED_IDS[opts.model_family]
    meta = read_shard_metadata(opts.data, opts.seq_len, opts.model_family)
    n_examples = int(opts.dry_run_n_examples or meta["num_instances"])
    if meta.get("task") and meta["task"] != opts.task:
        print(
            f"[ctc-suite] WARNING: --task {opts.task} != shard metadata task {meta['task']!r}",
            flush=True,
        )
    # Under context parallelism, instances are distributed over the DP dimension (world_size/cp),
    # not all ranks -- each CP group cooperatively processes ONE instance's sequence.
    cp_degree = opts.cp_degree if opts.cp_degree and opts.cp_degree > 1 else 1
    dp_world_size = world_size // cp_degree
    per_rank = derive_batch_geometry(opts.global_batch, dp_world_size, opts.micro_batch_instances)
    curriculum: Optional[Dict[str, Any]] = None
    if opts.variant == "chunked-mix":
        curriculum = derive_mask_mix_curriculum(
            n_examples=n_examples,
            epochs=opts.epochs,
            global_batch=opts.global_batch,
            world_size=dp_world_size,
            micro_batch_instances=opts.micro_batch_instances,
            mix_start_p=opts.mix_start_p,
            mix_end_p=opts.mix_end_p,
        )
        opts._mix_total_forwards = curriculum["mix_total_forwards"]

    # Family-specific tokenizer (its OWN vocab; pad == eos). The Qwen3.5 path keeps its own vocab
    # (NOT TokenizerConfig.qwen3()); the plain-Qwen3 path uses the Qwen3 vocab/eos.
    tokenizer_config = TokenizerConfig(
        vocab_size=opts.vocab_size,
        eos_token_id=ids.eos,
        pad_token_id=ids.eos,
        bos_token_id=None,
        identifier=FAMILY_TOKENIZER[opts.model_family],
    )
    # Instance source. Default (--no-pack): one example per instance, padded to seq_len (the proven
    # short-rung path). With --pack: greedily bin-pack WHOLE examples into seq_len windows (no
    # cross-example splitting) and emit cu_doc_lens for intra-example masking -- the only tractable
    # option for a mixed 8k..256k length distribution, where padding every short example up to a
    # 256k seq_len would be a ~32x compute waste. Packing composes with CP (proven in the
    # sft_longctx -packed- scripts: "Packing is supported under CP"). Same npy format either way.
    variant, vopts = variant_and_options(opts)
    # Landmark-packed data path for sparselandmark (variants.SparseLandmark.data_overrides):
    # first-fit bin-packs whole SFT examples into block-aligned landmark windows and inserts the
    # landmark token every MEM_FREQ_SPARSE content tokens, emitting doc_lens.
    override_source = variant.data_overrides(
        vopts,
        variant_context(opts),
        source_paths=[f"{opts.data}/token_ids_part_*.npy"],
        label_mask_paths=[f"{opts.data}/labels_mask_*.npy"],
        tokenizer=tokenizer_config,
    )
    if override_source is not None:
        instance_source_config = override_source
    elif opts.pack:
        # HARD GUARD: the packer builds a SegmentTree over max_sequence_length, which asserts
        # log2(N) is an integer. A non-power-of-2 --seq-len (e.g. 40960) therefore dies with a bare
        # "N should be a power of 2" -- but only AFTER the base checkpoint loads and the mesh is
        # built (~15 min at 4B), and only on rank 0, so every other rank then hangs until the 900 s
        # gloo timeout. One bad argument costs a full node-hour and reports itself as a distributed
        # timeout. Fail here instead, before anything expensive happens.
        if opts.seq_len & (opts.seq_len - 1) != 0:
            lo = 1 << (opts.seq_len.bit_length() - 1)
            raise SystemExit(
                f"--pack requires a power-of-2 --seq-len (the instance packer's SegmentTree "
                f"asserts it); got {opts.seq_len}. Use {lo} or {lo * 2} "
                f"(and keep --seq-len >= the shard's max_example_len)."
            )
        instance_source_config = PackingInstanceSourceConfig.from_npy(
            f"{opts.data}/token_ids_part_*.npy",
            tokenizer=tokenizer_config,
            sequence_length=opts.seq_len,
            max_sequence_length=opts.seq_len,
            label_mask_paths=[f"{opts.data}/labels_mask_*.npy"],
            expand_glob=True,
        )
    else:
        instance_source_config = PadToLengthInstanceSourceConfig.from_npy(
            f"{opts.data}/token_ids_part_*.npy",
            tokenizer=tokenizer_config,
            sequence_length=opts.seq_len,
            label_mask_paths=[f"{opts.data}/labels_mask_*.npy"],
            expand_glob=True,
        )
    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=opts.work_dir or WORK_DIR,
        global_batch_size=opts.global_batch * opts.seq_len,
        seed=34521 + opts.seed,
        num_workers=opts.num_workers,
    )
    return dict(
        meta=meta,
        n_examples=n_examples,
        per_rank=per_rank,
        curriculum=curriculum,
        model_config=build_model_config(opts),
        tokenizer_config=tokenizer_config,
        instance_source_config=instance_source_config,
        data_loader_config=data_loader_config,
    )


def dry_run(opts: argparse.Namespace) -> None:
    """Build all configs without CUDA / distributed init and print the resolved plan."""
    world_size = opts.dry_run_world_size
    plan = resolve_plan(opts, world_size)
    model_config = plan["model_config"]
    print("[ctc-suite] DRY RUN -- configs only, no CUDA, nothing trained, nothing written")
    print(
        f"  task={opts.task} variant={opts.variant} scale={opts.model_scale} "
        f"run_name={opts.run_name}"
    )
    print(
        f"  data={os.path.abspath(opts.data)} n_examples={plan['n_examples']}"
        + (
            f" (OVERRIDE; shard has {plan['meta']['num_instances']})"
            if opts.dry_run_n_examples
            else " (metadata)"
        )
    )
    ids = RESERVED_IDS[opts.model_family]
    print(
        f"  family={opts.model_family} marker_set={opts.model_family} "
        f"doc_start={ids.doc_start} doc_end={ids.doc_end} eos={ids.eos} "
        f"vocab_size={opts.vocab_size}"
    )
    print(
        f"  model: {opts.model_family}_{opts.model_scale} n_layers={model_config.n_layers} "
        f"params={model_config.num_params:,} "
        f"document_chunk_attention={model_config.document_chunk_attention}"
    )
    print(
        f"  seq_len={opts.seq_len} epochs={opts.epochs} lr={opts.lr} "
        f"global_batch={opts.global_batch} instances "
        f"({opts.global_batch * opts.seq_len:,} tokens) "
        f"micro_batch_instances={opts.micro_batch_instances}"
    )
    print(f"  world_size={world_size} (hypothetical) -> per-rank instances/step={plan['per_rank']}")
    train_module_config = build_train_module_config(opts, world_size)
    ac_config = train_module_config.ac_config
    print(
        f"  train_module: dp_name={train_module_config.dp_config.name} "
        f"shard_degree={train_module_config.dp_config.shard_degree} "
        f"AC={ac_config.mode if ac_config is not None else 'none'} "
        f"rank_microbatch_size={train_module_config.rank_microbatch_size:,} tokens"
    )
    if plan["curriculum"] is not None:
        c = plan["curriculum"]
        print(
            f"  mask-mix curriculum: p {opts.mix_start_p} -> {opts.mix_end_p} over "
            f"mix_total_forwards={c['mix_total_forwards']} | steps={c['steps']} "
            f"forwards_per_rank={c['forwards_per_rank']} "
            f"predicted final p_standard={c['final_p']:.4f}  [HARD-CHECK PASSED]"
        )
    else:
        print(f"  mask-mix: none (variant={opts.variant})")
    print(f"  data_loader: work_dir={plan['data_loader_config'].work_dir}")
    base = opts.base_checkpoint or DEFAULT_BASE_CHECKPOINT.get(opts.model_scale)
    print(f"  base_checkpoint={base or 'MISSING (required for a real run at this scale)'}")
    announce_wandb(opts)
    prov = build_provenance(opts, world_size)
    print(f"  provenance preview: {json.dumps(prov, indent=2, default=str)}")


#: Tolerant base load (new router / gain / projector keys keep their init); see
#: :func:`recipes.train.variants.tolerant_base_load`.
_tolerant_base_load = V.tolerant_base_load


def _guard_save_folder_and_markers(
    opts: argparse.Namespace, save_folder: str, base_checkpoint: str
) -> Dict[str, Any]:
    """Start-of-run guards (recipes/train/run_guards.py), checked on rank 0 and broadcast so every
    rank fails together instead of the others hanging at the first collective.

    * silent auto-resume: refuse a NEW launch into a save folder holding another launch's step
      checkpoints unless ``--resume`` (a preemption restart of the same launch passes);
    * unrepaired marker base: the marker rows the shard feeds the model must be distinguishable and
      in-distribution in norm, unless ``--skip-marker-audit``.

    :returns: The run identity to record in the save folder.
    """
    from olmo_core.distributed.utils import broadcast_object, is_distributed

    errors: List[str] = []
    identity: Optional[Dict[str, Any]] = None
    if get_rank() == 0:
        try:  # any rank-0 failure must reach the broadcast, or the other ranks hang
            try:
                identity = RG.check_run_identity(
                    save_folder, opts.run_name, allow_resume=opts.resume, resume_flag="--resume"
                )
            except RuntimeError as exc:
                errors.append(str(exc))
            meta_path = os.path.join(opts.data, "metadata.json")
            meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
            markers = ["doc_start", "doc_end"] if meta.get("doc_markers", True) else []
            if variant_and_options(opts)[0].data_path == "landmark":
                markers.append("landmark")
            if opts.skip_marker_audit:
                print("[ctc-suite] --skip-marker-audit: marker embedding audit NOT run", flush=True)
            elif markers:
                res = RG.audit_base_checkpoint(base_checkpoint, opts.model_family, markers)
                if res["ok"]:
                    r = res["report"]
                    print(
                        f"[ctc-suite] marker audit PASS ({','.join(markers)}): norm/median "
                        f"{ {k: round(v, 3) for k, v in r['marker_norm_ratios'].items()} } "
                        f"cos { {k: round(v, 4) for k, v in r['pairwise_cosine'].items()} }",
                        flush=True,
                    )
                else:
                    errors.append(
                        f"{res['error']}\n(pass --skip-marker-audit only if you know the markers "
                        "are irrelevant to this run)"
                    )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"start-of-run guard crashed: {type(exc).__name__}: {exc}")
    if is_distributed():
        errors, identity = broadcast_object((errors, identity))
    if errors:
        raise SystemExit("[ctc-suite] refusing to start:\n  " + "\n  ".join(errors))
    assert identity is not None
    return identity


def build_and_fit(opts: argparse.Namespace) -> None:
    """Build the model / data loader / trainer and run the fit (real training entrypoint).

    :param opts: Parsed CLI options.
    """
    save_folder = opts.save_folder or f"{SAVE_ROOT}/{opts.run_name}"
    base_checkpoint = opts.base_checkpoint or DEFAULT_BASE_CHECKPOINT.get(opts.model_scale)
    if not base_checkpoint:
        raise SystemExit(
            f"--base-checkpoint required for --model-scale {opts.model_scale} (no default base "
            "converted yet; run the marker audit on it first, see the plan §4)."
        )
    # Guard the distcp marker: staging a base via cp can drop the hidden .metadata, after which
    # olmo-core silently trains FROM SCRATCH (CE ~ ln(vocab)) instead of from the base.
    if not os.path.exists(os.path.join(base_checkpoint, ".metadata")):
        raise SystemExit(
            f"base checkpoint {base_checkpoint} missing .metadata -> would silently train from "
            "scratch. Point at the model_and_optim SUBDIR of the converted base."
        )

    world_size = max(1, get_world_size())
    run_identity = _guard_save_folder_and_markers(opts, save_folder, base_checkpoint)
    plan = resolve_plan(opts, world_size)
    print(
        f"[ctc-suite] task={opts.task} variant={opts.variant} scale={opts.model_scale} "
        f"n_examples={plan['n_examples']} epochs={opts.epochs} seq_len={opts.seq_len} "
        f"global_batch={opts.global_batch} world_size={world_size} "
        f"curriculum={plan['curriculum']}",
        flush=True,
    )
    announce_wandb(opts)

    if get_rank() == 0:
        os.makedirs(save_folder, exist_ok=True)
        RG.write_run_identity(save_folder, run_identity)
        with open(os.path.join(save_folder, "provenance.json"), "w") as f:
            json.dump(build_provenance(opts, world_size), f, indent=2, default=str)

    train_module_config = build_train_module_config(opts, world_size)

    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            # ffnmoe / softtoken add NEW parameters (router+gains / projector) that the plain
            # base does not have; the trainer's own load is strict, so those arms load the base
            # themselves below with strict=False (new keys keep their init: router -> full rung,
            # projector -> identity). load_path stays None for them so fit() does not re-load.
            # --branch-from: a decay branch resumes the stable run's checkpoint WITH optimizer and
            # trainer state (step, tokens seen, data-loader position), so it continues on unseen data.
            load_path=(
                opts.branch_from
                if opts.branch_from
                else (None if variant_and_options(opts)[0].new_params else base_checkpoint)
            ),
            # ...and "always" would then demand a checkpoint that no longer exists: those arms
            # resume from the save folder only if a step checkpoint is there.
            load_strategy=(
                LoadStrategy.if_available
                if variant_and_options(opts)[0].new_params and not opts.branch_from
                else LoadStrategy.always
            ),
            # --branch-optim-only (arm C's 8k -> 32k switch): the next phase reads a DIFFERENT shard,
            # so the data-loader position must not be restored; step/token counters restart and the
            # phase's own --max-tokens / --warmup-fraction 0 carry the schedule on from the stable LR.
            load_trainer_state=bool(opts.branch_from) and not opts.branch_optim_only,
            load_optim_state=bool(opts.branch_from),
            metrics_collect_interval=1,
            cancel_check_interval=1,
            # --max-tokens: a SHORTER budget on the same arm with a complete LR schedule (warmup
            # fraction + linear decay are relative to max_duration), i.e. a clean data-scaling
            # point below the arm's size. --max-steps only hard-stops mid-schedule.
            max_duration=(
                Duration.tokens(opts.max_tokens)
                if opts.max_tokens
                else Duration.epochs(opts.epochs)
            ),
            hard_stop=Duration.steps(opts.max_steps) if opts.max_steps else None,
            # no_checkpoints=True would ALSO skip the base-checkpoint LOAD block (trainer.fit
            # gates loading on `not no_checkpoints`) -> silent train-from-scratch. Keep False;
            # mid-run saving is controlled by the explicit CheckpointerCallback below, and the
            # --save-checkpoint block after fit() writes the model-only checkpoint for eval.
            no_checkpoints=False,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        # Register the checkpointer EXPLICITLY. Leaving it out does not disable it -- Trainer does
        # `callbacks.setdefault("checkpointer", CheckpointerCallback())` (trainer.py), so omitting it
        # silently activates the DEFAULT save_interval=250. At 4B that wrote a full
        # model+optim+train-state checkpoint (~55G) every 250 steps: a 750-step run produced 165G of
        # step*/ dirs that nothing reads, since eval loads the model-only `model_and_optim` saved
        # after fit(). Default to no mid-run checkpoints; pass --save-interval N to get resume
        # points back for a long run on a preemptible queue.
        # ``enabled=False`` when --no-final-checkpoint: CheckpointerCallback.post_train()
        # otherwise saves a FULL model+optim+train-state checkpoint unconditionally at the end —
        # 82G and ~25 min over lambda's ~60MB/s NFS, for state nothing downstream reads (eval and
        # rebasing both use the model-only export from the --save-checkpoint block after fit()).
        # The base-checkpoint LOAD is unaffected: trainer.fit gates it on ``no_checkpoints``, not
        # on this callback (see the note above).
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=opts.save_interval,
                enabled=not opts.no_final_checkpoint,
                # default 3 = keep only the newest three permanent checkpoints; 0 = keep all (decay-branch
                # points must survive the rest of the run)
                max_checkpoints=(opts.max_checkpoints or None),
            ),
        )
    )
    if opts.wandb:
        trainer_config = trainer_config.with_callback(
            "wandb",
            WandBCallback(
                name=opts.run_name,
                group=opts.wandb_group or opts.run_name,
                entity=opts.wandb_entity,
                project=WANDB_PROJECT,
                enabled=True,
                cancel_check_interval=10,
                config={
                    "experiment": "ctc_suite",
                    "task": opts.task,
                    "variant": opts.variant,
                    "model_scale": opts.model_scale,
                    "seq_len": opts.seq_len,
                    "epochs": opts.epochs,
                    "lr": opts.lr,
                    "global_batch": opts.global_batch,
                    "seed": opts.seed,
                    "n_examples": plan["n_examples"],
                    "mix_start_p": opts.mix_start_p if opts.variant == "chunked-mix" else None,
                    "mix_end_p": opts.mix_end_p if opts.variant == "chunked-mix" else None,
                    "base_checkpoint": base_checkpoint,
                    "ffn_drop_max_rate": opts.ffn_drop_max_rate,
                    "ffn_drop_layer_prob": opts.ffn_drop_layer_prob,
                    "train_skip_layers": opts.train_skip_layers,
                    "layer_drop_p": opts.layer_drop_p,
                    "layer_drop_schedule": opts.layer_drop_schedule,
                    "layer_drop_exclude_first": opts.layer_drop_exclude_first,
                    "layer_drop_exclude_last": opts.layer_drop_exclude_last,
                    "layer_drop_rescale": opts.layer_drop_rescale,
                    "data": opts.data,
                },
            ),
        )

    seed_all(12536 + opts.seed)
    model = plan["model_config"].build(init_device="meta")
    if os.environ.get("OLMO_FLA_LENGTH_AUTOTUNE") != "1":
        # FLA's GDN kernels key their Triton autotune on a length-derived NB, so every unseen
        # sequence width costs ~25-50 s of compiles (A100, measured 2026-10-20). Fixed-width dense
        # rows never notice; data-dependent widths (softtoken compaction) stalled on it.
        from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune

        print(f"[ctc-suite] FLA length autotune frozen on {freeze_fla_length_autotune()} kernel(s)")
    ids = RESERVED_IDS[opts.model_family]
    # Forwards per rank over the whole run = steps x grad-accum: the horizon the ffnmoe schedules
    # (target / exploration / layer curriculum) and the softtoken mixing curriculum anneal over.
    gbs_examples = max(1, opts.global_batch)
    if opts.pack:
        # Packed rows hold many examples: the step count is tokens / (rows x seq_len), not
        # examples / rows. Using the example count here over-estimated the schedule horizon
        # ~20x on the first Qwen3.5 FFN runs (target still 0.84 at the last step).
        n_tok = int(plan["meta"].get("num_tokens") or plan["meta"].get("total_tokens") or 0)
        if opts.max_tokens:
            n_tok = min(n_tok, opts.max_tokens) if n_tok else opts.max_tokens
        steps_total = (
            max(1, -(-n_tok * opts.epochs // (gbs_examples * opts.seq_len)))
            if n_tok
            else -(-plan["n_examples"] * opts.epochs // gbs_examples)
        )
    else:
        steps_total = -(-plan["n_examples"] * opts.epochs // gbs_examples)
    accum = max(1, opts.global_batch // (world_size * opts.micro_batch_instances))
    total_calls = max(1, steps_total * accum)
    if opts.ffn_drop_max_rate > 0 or opts.ffn_drop_layer_prob > 0:
        if opts.variant in ("ffnmoe", "flexcompute"):
            raise SystemExit("--ffn-drop-* (drop-CPT) cannot be combined with a routed variant")
    if opts.kv_route_debug:
        os.environ["KV_ROUTE_DEBUG"] = opts.kv_route_debug
        print(f"[ctc-suite] KV_ROUTE_DEBUG={opts.kv_route_debug}", flush=True)
    # Every per-variant model / train-module / callback change lives in recipes/train/variants.py
    # (shared with the canonical setA recipe): enable_* calls, budget_attach, gold keep hooks,
    # router / drop callbacks.
    active = active_variants(opts)
    vctx = variant_context(
        opts, rank=get_rank(), total_calls=total_calls, calls_per_step=accum, meta=plan["meta"]
    )
    for variant, vopts in active:
        variant.post_build_model(model, vopts, vctx)
    train_module = train_module_config.build(model)
    for variant, vopts in active:
        variant.post_build_train_module(train_module, vopts, vctx)
    for variant, vopts in active:
        for cb_name, cb in variant.extra_callbacks(vopts, vctx).items():
            trainer_config = trainer_config.with_callback(cb_name, cb)
    # Method-aware training FLOPs for every arm (records/flop-scaling-ffn-kv-plan.md §5).
    trainer_config = trainer_config.with_callback(
        "flop_meter",
        FlopMeterCallback(
            seq_len=opts.seq_len, pad_id=ids.eos, per_example=opts.flop_per_example
        ),  # rows are padded with EOS (see pad_token_id above)
    )
    source = plan["instance_source_config"].build(plan["data_loader_config"].work_dir)
    data_loader = plan["data_loader_config"].build(
        source, dp_process_group=train_module.dp_process_group
    )
    if opts.mem_snapshot:
        # Peak-memory attribution from inside the real (FSDP) trainer: record the allocator
        # history through the dry-run + first step, replay it to the peak, print the live set at
        # the peak grouped by allocation site, then exit. Debug only (2026-09-05, Qwen3 routers).
        from olmo_core.train.callbacks.callback import Callback as _CB

        class _MemSnapshot(_CB):
            def pre_train(self):
                import torch

                torch.cuda.memory._record_memory_history(max_entries=400000)

            def post_step(self):
                import torch

                if self.step >= 1:
                    from olmo_core.nn.mem_attribution import summarize_peak

                    summarize_peak(torch.cuda.memory._snapshot(), top=16)
                    torch.cuda.memory._record_memory_history(enabled=None)
                    raise SystemExit(0)

        trainer_config = trainer_config.with_callback("mem_snapshot", _MemSnapshot())
    trainer = trainer_config.build(train_module, data_loader)
    if variant_and_options(opts)[0].new_params and not opts.branch_from:
        _tolerant_base_load(base_checkpoint, train_module.model, save_folder)
    if opts.mem_snapshot:
        import torch

        try:
            trainer.fit()
        except torch.OutOfMemoryError:
            # the attribution is most useful exactly when the step OOMs: print it, then re-raise
            from olmo_core.nn.mem_attribution import summarize_peak

            print(
                "[mem-snapshot] OOM during the step; attribution of the trace so far:", flush=True
            )
            summarize_peak(torch.cuda.memory._snapshot(), top=16)
            raise
    elif opts.torch_profile:
        # Where does a step go? torch.profiler over steps 3-4 (2 warm-up), rank 0 prints the top
        # ops by CUDA and by CPU time (2026-09-08: soft-token steps were ~7x slower than dense at
        # 0.18x the FLOPs on Beaker/FSDP while a single local GPU was fine).
        import torch
        from torch.profiler import ProfilerActivity, profile, schedule

        from olmo_core.train.callbacks import Callback

        def _report(prof):
            if int(os.environ.get("RANK", "0")) == 0:
                ka = prof.key_averages()
                print(
                    "[torch-profile] top ops by CUDA time:\n"
                    + ka.table(sort_by="cuda_time_total", row_limit=40),
                    flush=True,
                )
                print(
                    "[torch-profile] top ops by CPU time:\n"
                    + ka.table(sort_by="cpu_time_total", row_limit=30),
                    flush=True,
                )

        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=1, active=2, repeat=1),
            on_trace_ready=_report,
        )

        class _ProfStep(Callback):
            def post_step(self):
                prof.step()

        trainer.add_callback("torch_profile", _ProfStep())
        prof.start()
        try:
            trainer.fit()
        finally:
            prof.stop()
    else:
        trainer.fit()

    # Save a model-only checkpoint in the eval loader's expected layout (config.json +
    # model_and_optim/) so the docchunk eval scripts load it directly.
    if opts.save_checkpoint:
        from olmo_core.distributed.checkpoint import save_model_and_optim_state

        save_model_and_optim_state(
            f"{save_folder}/model_and_optim", train_module.model, save_overwrite=True
        )
        if get_rank() == 0:
            model_dict = plan["model_config"].as_config_dict()
            if opts.attn_backend == "torch":
                # The soft-token arm trains with the SDPA backend (the flash_2 path is ~20x slower
                # on its compacted rows), but SDPA has no KV-cached generation: record flash_2 in
                # the export so the evaluators build the (identical-weight) model for fast decode.
                def _swap(o):
                    if isinstance(o, dict):
                        for k, v in o.items():
                            if k == "backend" and v == "torch":
                                o[k] = "flash_2"
                            else:
                                _swap(v)
                    elif isinstance(o, list):
                        for v in o:
                            _swap(v)

                _swap(model_dict)
            experiment = {
                "model": model_dict,
                "dataset": {"tokenizer": plan["tokenizer_config"].as_config_dict()},
                # Recorded so the eval scores with the routing the run trained with (the
                # evaluator reads these blocks and enables the router before loading); every key
                # is present, None unless an active variant fills it (variants.export_record).
                **V.export_record(active),
            }
            with open(f"{save_folder}/config.json", "w") as f:
                json.dump(experiment, f)
            print(f"[ctc-suite] saved model-only checkpoint -> {save_folder}", flush=True)


def parse_args() -> argparse.Namespace:
    """Build the CLI parser and parse arguments.

    :returns: Parsed options (with the private ``_mix_total_forwards`` slot pre-initialized).
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task", required=True, help="suite task name (provenance + wandb; see plan)")
    ap.add_argument(
        "--data",
        required=True,
        help="shard dir from convert_unified_to_document_landmark.py --marker-set qwen3_5 "
        "(token_ids_part_*.npy + labels_mask_*.npy + metadata.json)",
    )
    ap.add_argument(
        "--model-family",
        choices=["auto", *sorted(MODEL_FACTORIES)],
        default="auto",
        help="auto (default) = detect from the shard's metadata marker_set; or force qwen3 "
        "(plain dense/causal) / qwen3_5 (GDN+attn hybrid). Determines the model factory, marker "
        "ids, tokenizer, and embedding size.",
    )
    ap.add_argument(
        "--model-scale",
        default="0.8b",
        help="scale key within the resolved family: qwen3_5={0.8b,2b,4b,9b,27b}; qwen3={0.6b,1.7b,4b,8b}",
    )
    ap.add_argument(
        "--variant",
        choices=[
            "full",
            "chunked",
            "chunked-mix",
            "sparselandmark",
            "pooledkv",
            "ffnmoe",
            "softtoken",
            "kvroute",
            "flexcompute",
        ],
        required=True,
        help="full = plain causal (no document_chunk_attention); chunked = pure document-chunked "
        "mask; chunked-mix = chunked + curriculum mask mixing (mix_start_p -> mix_end_p); "
        "sparselandmark = AttentionType.sparse_landmark on the full-attn blocks + landmark-packed "
        "data; pooledkv = AttentionType.pooled_doc_kv on the full-attn blocks (train-time "
        "per-document KV pooling: gold + a few random docs keep real KV, the rest collapse to a "
        "mean-pooled slot; inference is ordinary full attention); ffnmoe = nested-width FFN "
        "router (olmo_core.nn.nested_ffn_moe: per-token choice of a prefix-sliced FFN width or "
        "null under a budget hinge); softtoken = pooled-doc soft tokens (the whole stack runs on "
        "a compacted sequence: gold + n_random docs keep real tokens, every other doc collapses "
        "to ONE projected soft token; inference is ordinary full attention). Both add small new "
        "parameters (router+gains / projector) that are initialized at load, so they run from "
        "the plain base -- no baked base needed (tolerant load, see build_and_fit).",
    )
    # ---- ffnmoe (records/flop-scaling-ffn-kv-plan.md; recipe = ffnmoe/README.md v10/v12) ----
    ap.add_argument(
        "--sparse-loss-positions",
        action="store_true",
        help="run the LM head only at supervised positions (exact; SFT masked-label runs only)",
    )
    # 0 = every layer routed. Matches the KV and block-skip routers (both default 0) and what the
    # trained v11/v12 checkpoints used. The old default of 12 left a third of the FFN unrouted, so
    # an all-null arm only reached a 2.66x FFN cut instead of the ~20x the component sweeps show.
    ap.add_argument(
        "--ffn-moe-start-layer", type=int, default=0, help="first routed layer (0 = all)"
    )
    # Rung ladder. Default chosen from the MEASURED speed knee (research/kv_route/flexcompute_40x): below
    # width ~300 at 4B, narrowing a rung buys FLOPs but no time -- a narrow token still reads x,
    # writes a full-width output and launches its own kernels. The old default
    # (1,16,64,256,1024,9728 -> widths 9728/608/152/38/9/1) put FOUR of six rungs below that knee,
    # and the trained routers put 52-65% of their working tokens there: full fixed cost for ~1e-4
    # of the budget. These divisors keep every working rung at or above the knee.
    ap.add_argument(
        "--ffn-moe-divisors", default="1,2.4,4.9,10,32", help="rung ladder (see note above)"
    )
    ap.add_argument(
        "--ffn-moe-width-multiple",
        type=int,
        default=8,
        help="8 keeps rungs GEMM-friendly; 1 allows a width-1 rung (measured to be pure overhead)",
    )
    ap.add_argument("--ffn-moe-no-null", action="store_true")
    ap.add_argument(
        "--ffn-moe-target", type=float, default=0.01, help="budget: mean FFN cost on routed layers"
    )
    ap.add_argument("--ffn-moe-budget-weight", type=float, default=1.0)
    ap.add_argument("--ffn-moe-hinge-power", type=int, default=1)
    ap.add_argument(
        "--ffn-moe-trainable-width",
        type=int,
        default=0,
        help="'train what you route to': freeze the routed FFNs beyond this many hidden units "
        "(only the prefix + router + gains train; 0 = whole FFN trains)",
    )
    ap.add_argument(
        "--ffn-moe-two-sided",
        action="store_true",
        help="budget = |mean_cost - target| (pulls an undershooting router back up to the target)",
    )
    ap.add_argument(
        "--ffn-moe-target-anneal-frac",
        type=float,
        default=0.3,
        help="0 = hinge active from step 0 (stage 2 of the two-stage recipe)",
    )
    ap.add_argument("--ffn-moe-explore", type=float, default=0.1)
    ap.add_argument("--ffn-moe-explore-anneal-frac", type=float, default=0.3)
    ap.add_argument("--ffn-moe-recon-frac", type=float, default=0.02)
    ap.add_argument("--ffn-moe-recon-weight", type=float, default=0.0)
    ap.add_argument("--ffn-moe-entropy-weight", type=float, default=0.0)
    ap.add_argument("--ffn-moe-layer-curriculum-frac", type=float, default=0.0)
    ap.add_argument(
        "--ffn-drop-max-rate",
        type=float,
        default=0.0,
        help="drop-CPT (olmo_core.nn.ffn_token_drop): each row draws r ~ U[0, this] and every token "
        "skips each droppable layer's FFN with prob r (training only; no params; 0 = off)",
    )
    ap.add_argument(
        "--ffn-drop-layer-prob",
        type=float,
        default=0.0,
        help="drop-CPT: probability a (row, layer) drops the layer's FFN for EVERY token",
    )
    ap.add_argument(
        "--ffn-drop-start-layer", type=int, default=1, help="drop-CPT: first droppable layer"
    )
    # ---- whole-block skipping (olmo_core.nn.layer_drop; any arm, training only) ----
    ap.add_argument(
        "--train-skip-layers",
        default="",
        help="skip these blocks (e.g. '10,14,18' or '10-12'; negative = from the end) on EVERY "
        "training forward: the block's forward is not called (identity on the residual stream). "
        "Eval runs every block.",
    )
    ap.add_argument(
        "--layer-drop-p",
        type=float,
        default=0.0,
        help="LayerDrop: every optimizer step each eligible block is skipped whole with this "
        "probability (seeded from --seed and the global step: same blocks on every rank and "
        "micro-batch; 0 = off)",
    )
    ap.add_argument(
        "--layer-drop-schedule",
        choices=["uniform", "linear"],
        default="uniform",
        help="uniform: p_i = p; linear: p_i = p * (i + 1) / n_layers (stochastic depth)",
    )
    ap.add_argument(
        "--layer-drop-exclude-first", type=int, default=0, help="never drop the first K blocks"
    )
    ap.add_argument(
        "--layer-drop-exclude-last", type=int, default=0, help="never drop the last K blocks"
    )
    ap.add_argument(
        "--layer-drop-rescale",
        action="store_true",
        help="scale a RUN block's residual update by 1/(1-p_i) (inverted-dropout expectation)",
    )
    ap.add_argument(
        "--router-lr", type=float, default=1e-3, help="ffnmoe: router/gain LR (backbone uses --lr)"
    )
    ap.add_argument(
        "--kv-route-off",
        action="store_true",
        help="flexcompute: do NOT install the KV keep/drop router (FFN + block-skip routers only). "
        "Ignored for --variant kvroute. See kv_route_enabled().",
    )
    ap.add_argument(
        "--kv-route-start-layer", type=int, default=0, help="kvroute: first routed attention layer"
    )
    ap.add_argument(
        "--kv-route-target",
        type=float,
        default=0.5,
        help="kvroute: mean KEEP fraction budget (1 = keep all)",
    )
    ap.add_argument("--kv-route-budget-weight", type=float, default=1.0)
    ap.add_argument(
        "--kv-route-one-sided", action="store_true", help="hinge instead of |mean-target|"
    )
    ap.add_argument("--kv-route-target-anneal-frac", type=float, default=0.3)
    ap.add_argument("--kv-route-explore", type=float, default=0.0)
    ap.add_argument("--kv-route-explore-anneal-frac", type=float, default=0.3)
    ap.add_argument(
        "--block-skip-target",
        type=float,
        default=None,
        help="flexcompute: enable per-token block skipping (olmo_core.nn.block_skip) with this mean RUN "
        "fraction budget (ignored under --flex-joint-target, which owns the budget)",
    )
    ap.add_argument("--block-skip-start-layer", type=int, default=0)
    ap.add_argument(
        "--block-skip-capacity",
        type=float,
        default=None,
        help="flexcompute: FIXED per-block capacity (mixture-of-depths top-k) instead of a "
        "threshold, e.g. 0.25. Only useful with --block-skip-compact, where it makes the "
        "compacted shape static and removes one host sync per block",
    )
    ap.add_argument(
        "--block-skip-compact",
        action="store_true",
        help="flexcompute: GATHER the kept tokens instead of masking them, so a skipped token's "
        "work is really not done. Measured on a 4B step at 8k (research/kv_route/flexcompute_40x/"
        "router_matrix_4B_8192_compact.json): keep=0.5 goes from 0.90x (masked, i.e. SLOWER than "
        "dense) to 1.34x, keep=0.25 to 1.64x. Changes what 'skipped' means for the GatedDeltaNet "
        "short convolution -- see the caveat in olmo_core.nn.block_skip",
    )
    ap.add_argument(
        "--kv-route-debug", default="", help="memory-diagnostic ablations: no_router | no_holder"
    )
    ap.add_argument(
        "--mem-snapshot",
        action="store_true",
        help="debug: print the live allocations at the peak of the first step, then exit",
    )
    ap.add_argument(
        "--flex-share-seq-len",
        type=int,
        default=8192,
        help="flexcompute: sequence length the FFN/attention FLOP shares of the joint budget are evaluated "
        "at (real example length, NOT the padded window: at 65k attention-score FLOPs are ~50%% of "
        "dense vs ~12%% at real 2k-32k lengths, which would steer every saving into eviction)",
    )
    ap.add_argument(
        "--flex-joint-target",
        type=float,
        default=None,
        help="flexcompute: ONE budget on total (FFN + attention-score) FLOPs as a fraction of dense; "
        "replaces the two per-router budgets",
    )
    # ---- softtoken (records/pooled-doc-kv-handoff.md; v20 = --st-n-random 128, v22 = 256) ----
    ap.add_argument(
        "--st-n-random", type=int, default=128, help="random non-gold docs kept real per example"
    )
    ap.add_argument(
        "--st-n-random-range",
        default="",
        help="lo,hi log-uniform breadth per call (overrides --st-n-random)",
    )
    ap.add_argument(
        "--st-keep-frac",
        type=float,
        default=None,
        help="keep a FIXED FRACTION of each example's non-gold docs real (gold always kept); "
        "context-length invariant -- the FLOP-scaling study's KV arms (overrides n-random/range)",
    )
    ap.add_argument(
        "--st-keep-mode",
        default="gold_plus_random",
        help="softtoken + gold sidecar: the select_keep_docs policy. gold_plus_random (default) "
        "keeps every gold doc real; gold_pooled_random is its inverse -- gold is ALWAYS pooled and "
        "--st-keep-frac (or --st-keep-prob) of the NON-gold docs stay real, so a visible id is "
        "never an answer and the slots are the only route to the loss. Also gold_subsample / "
        "random_only / random_nongold / gold_pair / gold_halves. The WHOLE-CATEGORY policies "
        "smallcat_keep (gold-blind) and gold_plus_wholecats are handled by resolve_cat_keep()",
    )
    ap.add_argument("--st-n-gold", type=int, default=0)
    ap.add_argument(
        "--st-keep-smallcat",
        type=int,
        default=3,
        help="--st-keep-mode smallcat_keep: how many of the SMALLEST document categories to keep "
        "fully real (a category is kept or pooled as a UNIT, never partially)",
    )
    ap.add_argument(
        "--st-keep-decoy-cats",
        type=int,
        default=1,
        help="--st-keep-mode smallcat_keep: additional LARGE categories kept whole as decoys, so "
        "'kept whole' does not by itself mean 'small category'. Costs real FLOPs -- watch the "
        "[cat-keep] log's real_token_frac",
    )
    ap.add_argument(
        "--st-keep-decoy-max-mult",
        type=float,
        default=2.0,
        help="a decoy category is eligible only if its size is <= this multiple of the largest "
        "kept small one; when nothing is eligible the row keeps no decoy (logged as decoy_skipped)",
    )
    ap.add_argument(
        "--st-keep-cats",
        type=int,
        default=3,
        help="--st-keep-mode gold_plus_wholecats: how many NON-gold categories to keep whole "
        "alongside the gold document's own category (the SMALLEST ones by default)",
    )
    ap.add_argument(
        "--st-keep-cats-random",
        action="store_true",
        help="--st-keep-mode gold_plus_wholecats: pick those categories at random instead of "
        "taking the smallest",
    )
    ap.add_argument(
        "--st-cat-threshold-rule",
        default="gap",
        choices=["gap", "mean_std"],
        help="how the document-similarity graph is cut (olmo_core.nn.attention.doc_categories)",
    )
    ap.add_argument(
        "--st-cat-log-every",
        type=int,
        default=50,
        help="[cat-keep] log cadence (categories found, kept sizes, real-token fraction)",
    )
    ap.add_argument(
        "--st-keep-prob",
        type=float,
        default=0.1,
        help="gold-blind fallback keep prob (no sidecar / --st-gold-blind)",
    )
    ap.add_argument(
        "--st-gold-blind",
        action="store_true",
        help="ignore the gold sidecar: keep docs by --st-keep-prob only (oolong)",
    )
    ap.add_argument("--st-gold-sidecar", default=None, help="default <data>/gold_fingerprints.json")
    ap.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="softtoken: freeze every parameter except pooled_projector (learned-slot screening arm)",
    )
    ap.add_argument(
        "--st-no-detach-soft-kv",
        action="store_true",
        help="the winning recipe DETACHES; this is the ablation",
    )
    ap.add_argument(
        "--st-no-detach-soft-gdn",
        action="store_true",
        help="ablation: let the slots' GDN state writes carry gradient (pre-2026-09-08 behaviour). Default: "
        "slots are fully detached -- attention K/V AND GDN writes -- so they play no role in training",
    )
    ap.add_argument(
        "--st-len-bias",
        action="store_true",
        help="softtoken: add +log(doc_len) to every pooled slot's attention logit (log-mass trick; "
        "uses the additive-bias SDPA path, so pair with --attn-backend torch)",
    )
    ap.add_argument(
        "--st-header-stop-id",
        type=int,
        default=None,
        help="softtoken: keep each doc's HEADER real (tokens after doc_start through the "
        "--st-header-stop-count-th occurrence of this token id; ':' = 25 on Qwen3.5) and pool "
        "only the body -- the eval-side parity construction (contradiction: count 1; oolong: count 3)",
    )
    ap.add_argument("--st-header-stop-count", type=int, default=1)
    ap.add_argument(
        "--st-header-extra-tokens",
        type=int,
        default=0,
        help="softtoken: ALSO keep the next K body tokens (right after the header) real, so a "
        "pooled doc's compacted row is <header><first K body tokens><SLOT>. 0 (default) is "
        "bit-identical to header-only behaviour. Works even without --st-header-stop-id (K is "
        "then counted from the doc's first token after doc_start). A doc whose remaining body "
        "has fewer than K tokens is kept entirely real (no slot at all) -- see "
        "olmo_core.nn.attention.chunked_mask.mark_doc_headers_free.",
    )
    ap.add_argument(
        "--st-keep-token-rule",
        choices=["none", "first", "first_last", "rule", "spans", "online"],
        default="none",
        help="softtoken: WHICH body tokens of a pooled document stay real. 'none' (default) = off, "
        "bit-identical to every run before 2026-09-15. 'first' = the first --st-keep-token-k body "
        "tokens, i.e. exactly --st-header-extra-tokens K (same code path, bit-identical; the two "
        "flags are mutually exclusive). 'first_last' = the first K//2 and last K-K//2 body tokens "
        "-- same cost as 'first' and better on every rung of the frozen-model probe (dCE +0.044 vs "
        "+0.049 and R@gold 0.927 vs 0.818 at 2k, genF1 0.800 vs 0.787 at 32k, "
        "records/outlier-realtoken-parity-probe.md), and it needs no tokenizer and no shard read. "
        "'rule' = the K highest-scoring body tokens PER DOCUMENT "
        "under a linear score over cheap token features (idf, relative position in the body, "
        "first-sentence, capitalised, digit, piece length) -- kept at their ORIGINAL positions, so "
        "non-contiguous. Ties go to the earlier position; a doc with <= K body tokens is kept "
        "whole. Which tokens are kept is the whole effect: at K=8 on outlier the eval-side probe "
        "measures rand 0.558 CE (WORSE than keeping nothing, 0.470), first 0.263, and this rule "
        "0.071 -- the gradient oracle's 0.072 (records/outlier-saliency-preview-probe.md 5d)",
    )
    ap.add_argument(
        "--st-keep-token-k",
        type=float,
        default=0,
        help="softtoken: tokens kept real per pooled document for --st-keep-token-rule; a value in "
        "(0, 1) is a FRACTION of each document body (per-document budget, e.g. 0.2)",
    )
    ap.add_argument(
        "--st-keep-token-weights",
        default=None,
        help="softtoken: feature weights for --st-keep-token-rule rule -- inline JSON or a path to "
        'a JSON file. Flat ({"idf": 0.4, ...}) = weights on RAW features; the ridge shape '
        '({"weights": {...}, "sd": {...}}) = weights on STANDARDISED features (mu/bias are '
        "accepted and ignored: they shift every token equally and cannot change a per-doc top-k). "
        "Default: olmo_core.nn.pooled_soft_token.DEFAULT_KEEP_TOKEN_WEIGHTS, a PLACEHOLDER whose "
        "signs come from the measured within-document gradient profile (probe record 5a(c)), not "
        "from the probe's fitted ridge -- pass the fitted vector here once it is harvested",
    )
    ap.add_argument(
        "--st-keep-token-log-every",
        type=int,
        default=50,
        help="softtoken: log the realised keep-token cost (real tokens per pooled doc, real-token "
        "fraction) every N compactions; the first 5 are always logged",
    )
    ap.add_argument(
        "--st-drop-slots",
        action="store_true",
        help="softtoken: emit NO slot token for pooled documents -- their tokens simply vanish from "
        "the compacted row and the kept tokens keep their original positions. Slot-less compaction "
        "has no slot K/V / GDN write to detach (detach flags become no-ops). Motivated by the "
        "eval-side grid, where gold_fl20p8_noslot == gold_fl20p8 on every cell (2026-09-22).",
    )
    ap.add_argument(
        "--st-block-size",
        type=int,
        default=0,
        help="softtoken: > 0 = positional-block mode for MARKER-FREE rows (e.g. the olmo3_cptdrop "
        "CPT shards, EOS + one 32,767-token window): a row's 'documents' are consecutive blocks of "
        "this many positions after the leading EOS (last partial block included) instead of "
        "<|doc_start|>/<|doc_end|> spans; every keep rule applies per block. Needs --st-gold-blind. "
        "0 = marker documents (historical).",
    )
    ap.add_argument(
        "--st-keep-recent",
        type=int,
        default=0,
        help="softtoken positional-block mode: the last N content tokens of every row are always "
        "kept (dense recent context); the keep rule applies to the tokens before them. 0 = off.",
    )
    ap.add_argument(
        "--st-keep-ramp",
        default="",
        help="softtoken positional-block mode: 'a,b' = the per-block keep fraction ramps linearly "
        "from a on the row's first block to b on its last (far context compressed hardest when "
        "a < b). Replaces --st-keep-token-k when a token rule (first_last / first / rule / spans) "
        "is set, else --st-keep-prob (whole blocks). Empty = off.",
    )
    ap.add_argument(
        "--tokexit-layer",
        dest="st_tokexit_layer",
        type=int,
        default=-1,
        help="softtoken positional-block mode, token exit (olmo_core.nn.token_exit): the tokens "
        "the keep rule drops run blocks [0, L) on the full row and are compacted out (original "
        "positions kept) at the input of block L; kept tokens run every block, dropped tokens' "
        "labels get no loss. 0 = identical to the input drop. -1 (default) = off. Needs "
        "--st-block-size > 0 and --st-drop-slots.",
    )
    ap.add_argument(
        "--st-span-len",
        type=int,
        default=12,
        help="softtoken --st-keep-token-rule spans: length of the random phrase-sized spans kept "
        "(re-drawn every forward, so every position of a document is eventually seen)",
    )
    ap.add_argument(
        "--st-keep-token-min",
        type=int,
        default=0,
        help="softtoken: floor for a fractional --st-keep-token-k -- every document keeps at least this "
        "many body tokens (max(min, k * |body|)): short documents stay whole, long ones keep the fraction",
    )
    ap.add_argument(
        "--st-kept-doc-k",
        type=float,
        default=None,
        help="softtoken two-tier keep: documents in the keep set (gold + random decoys, or the "
        "gold-blind random set) get the keep-token rule at THIS k instead of staying whole; every "
        "other document gets --st-keep-token-k. Unset = kept documents stay whole (historical).",
    )
    # ---- online learned keep scorer (--st-keep-token-rule online; olmo_core.nn.online_token_router)
    ap.add_argument("--st-otr-keep-frac", type=float, default=0.3,
                    help="online router: body tokens kept per row (fraction of the row's body tokens)")
    ap.add_argument("--st-otr-explore-frac", type=float, default=0.1,
                    help="online router: share of the kept budget drawn uniformly at random (exploration)")
    ap.add_argument("--st-otr-early-nograd", action="store_true",
                    help="online router, mid-stack: on routed steps run blocks [0, drop_layer) forward-only (they learn "
                    "only on the dense steps; charged forward-only)")
    ap.add_argument("--st-otr-dense-tail-frac", type=float, default=0.0,
                    help="online router: train the last FRAC of the steps uncompressed (e.g. 0.1, with "
                    "--st-otr-dense-every 0 for routed-then-dense-tail)")
    ap.add_argument("--st-otr-dense-keep-frac", type=float, default=1.0,
                    help="online router: keep share on the periodic --st-otr-dense-every steps (1.0 = full context; e.g. "
                    "0.5 routes them at 50%%, bounding the peak memory)")
    ap.add_argument("--st-otr-exit-loss-weight", type=float, default=0.0,
                    help="online router, mid-stack: add W x the LM loss of the DROPPED tokens read out at the split layer "
                    "(early exit; 1.0 = every label counts once; CPT rows -- answer-only SFT has no dropped labels)")
    ap.add_argument("--st-otr-freeze-after", type=int, default=0,
                    help="online router: train normally for N steps, then freeze the embeddings + blocks [0, drop_layer) "
                    "and free their optimizer state (warmup-then-freeze)")
    ap.add_argument("--st-otr-min-per-doc", type=int, default=0,
                    help="online router: per-document token floor -- every document keeps max(its share, K) body tokens "
                    "(its top ones by score), counted against the budget")
    ap.add_argument("--st-otr-freeze-early", action="store_true",
                    help="online router, mid-stack: freeze the embeddings and blocks [0, drop_layer) (no grads, no "
                    "optimizer state; forward-only on every step)")
    ap.add_argument("--st-otr-merge-dropped", action="store_true",
                    help="online router, mid-stack: average each document's dropped tokens into one summary token "
                    "(no label) carried with the kept row past the split, instead of discarding them")
    ap.add_argument("--st-otr-stages", default="",
                    help="online router pyramid drop, e.g. '3:0.5,7:0.1,16:0.03' (layer:keep of the body, absolute and "
                    "non-increasing; each stage scored from its own layer by its own router); overrides "
                    "--st-otr-drop-layer / --st-otr-keep-frac")
    ap.add_argument("--st-otr-early-lora", type=int, default=0,
                    help="online router: LOW-RANK early stack -- blocks [0, drop_layer) + embeddings frozen from step 0, "
                    "every linear in those blocks gets a rank-R LoRA adapter (no base grads / Adam state). 0 = off")
    ap.add_argument("--st-otr-early-lora-scale", type=float, default=32.0,
                    help="early LoRA output scale (acts as the adapters' LR multiplier under Adam; default 32)")
    ap.add_argument("--st-otr-target-layer", type=int, default=0,
                    help="online router: read the gradient targets at this deeper block (e.g. 16) instead of the drop "
                    "layer; with the drop only the full-context (--st-otr-dense-every) steps fit, with "
                    "--st-otr-mod-layers every step does. 0 = the drop layer")
    ap.add_argument("--st-otr-mass-target", type=float, default=0.0,
                    help="online router ADAPTIVE keep: per document, keep the fewest top-scored tokens whose predicted "
                    "||dL/dh||^2 covers this share (e.g. 0.8); then --st-otr-min-per-doc, then the per-row "
                    "--st-otr-max-keep-frac cap. --st-otr-keep-frac then only sizes exploration. 0 = fixed keep")
    ap.add_argument("--st-otr-max-keep-frac", type=float, default=1.0,
                    help="online router, with --st-otr-mass-target: cap on each row's kept body share (peak memory)")
    ap.add_argument("--st-otr-freeze-schedule", default="",
                    help="online router PROGRESSIVE freeze, e.g. '4:0.25,7:0.5,12:0.75': freeze embeddings + blocks "
                    "[0, 4) after 25%% of the steps, [0, 7) after 50%%, [0, 12) after 75%% (gradients and optimizer state "
                    "released at each stage); exclusive with --st-otr-freeze-after / --st-otr-freeze-early")
    ap.add_argument("--st-otr-mod-layers", default="",
                    help="online router MIXTURE-OF-DEPTHS, e.g. '8-31:2' (blocks 8, 10, ..., 30): at the first listed "
                    "block the router picks --st-otr-keep-frac of each document's tokens; the listed blocks run on "
                    "those (+ every non-document token) only and the rest pass the residual through; every token "
                    "reaches the head (no label dropped). Sets --st-otr-drop-layer to the first listed block")
    ap.add_argument("--st-otr-per-task-key-tokens", type=int, default=0,
                    help="online router: a separate ridge router per task in a mixed batch, keyed on each row's first "
                    "N tokens (its prompt template; 11 identifies every CTC mix task). 0 = one shared router")
    ap.add_argument("--st-otr-keep-first", type=int, default=0,
                    help="online router: always keep each document's first K body tokens (its header / id span); "
                    "they count against the budget and the router picks the rest")
    ap.add_argument("--st-otr-doc-share", type=float, default=1.0,
                    help="online router: share of the budget spread over documents in proportion to length "
                    "(each keeps its own top tokens by score); the rest goes to the row-level top scores. "
                    "1.0 = within-document choice only (per-document density cannot name gold)")
    ap.add_argument("--st-otr-score", choices=["router", "random", "first_last", "dh"], default="router",
                    help="online router: who picks the kept tokens -- the learned scorer, or a control with the same "
                    "budget, allocation and drop point (random; each document's first and last tokens; dh = fit-free "
                    "rule, top tokens by residual change ||h^L - h^0||, mid-stack only)")
    ap.add_argument("--st-otr-feature-layer", type=int, default=0,
                    help="online router: hidden features at this layer (0 = input embedding, free; L > 0 = a "
                    "no-grad prefix pass through blocks [0, L) on the full row, charged to the FLOP meter)")
    ap.add_argument("--st-otr-drop-layer", type=int, default=0,
                    help="online router: 0 = drop at the input; L > 0 = every token runs blocks [0, L) on the "
                    "full row, then dropped tokens skip blocks [L, n) (features = the forward's own h at L, no "
                    "prefix pass; targets read at L). Charged per block by the FLOP meter")
    ap.add_argument("--st-otr-no-hidden", action="store_true",
                    help="online router: structure + lexical features only")
    ap.add_argument("--st-otr-no-query", action="store_true",
                    help="online router: drop the h*q query-interaction features")
    ap.add_argument("--st-otr-use-gold", action="store_true",
                    help="online router: per-document 0/1 gold flag (gold sidecar) as a feature")
    ap.add_argument("--st-otr-target", choices=["gnorm", "gxh", "dh", "mix"], default="gnorm",
                    help="online router target: gnorm log ||dL/dh_t||; gxh log |<dL/dh_t, h_t>|; dh log ||h_t^L - h_t^0|| "
                    "(residual change through the first L blocks, mid-stack only); mix = average of the gnorm and dh ranks")
    ap.add_argument("--st-otr-ridge", type=float, default=0.1,
                    help="online router: ridge strength relative to the mean Gram diagonal (offline probe: 0.01 > 0.1 > 1.0 at 1M samples)")
    ap.add_argument("--st-otr-decay", type=float, default=0.9, help="online router: per-step forgetting")
    ap.add_argument("--st-otr-warmup-steps", type=int, default=1,
                    help="online router: leading keep-all steps (targets over every token, no heuristic init)")
    ap.add_argument("--st-otr-dense-every", type=int, default=0,
                    help="online router: every N-th step trains uncompressed (0 = never after warmup)")
    ap.add_argument("--st-otr-samples-per-row", type=int, default=2048)
    ap.add_argument("--st-otr-proj-dim", type=int, default=0,
                    help="online router: random projection of hidden features (0 = all d_model dims)")
    ap.add_argument("--st-otr-log-every", type=int, default=10)
    ap.add_argument(
        "--st-pad-multiple",
        type=int,
        default=256,
        help="softtoken: round each compacted micro-batch's width up to a multiple of this. The "
        "width is data-dependent and FLA's GDN kernels re-autotune (~25-50 s on an A100) at every "
        "width they have not seen -- the 2026-10-20 fast8k Lambda stalls. 1 = exact width.",
    )
    ap.add_argument(
        "--st-slot-mode",
        choices=["mean", "cmean", "cent_cmean"],
        default="mean",
        help="softtoken: how a pooled doc's slot vector is built. 'mean' (default) = the plain "
        "mean input embedding, bit-identical to every run before 2026-09-15. 'cmean' = the mean "
        "over CONTENT tokens only (drop punctuation/whitespace, the markers/pad, and the top-K "
        "most frequent training-shard ids). 'cent_cmean' = that, minus the row centroid and "
        "rescaled to the row's mean real-token embedding norm. A document's plain mean is ~94% "
        "shared common-word mass -- every doc sits at cosine 0.93-0.94 from the corpus centroid "
        "-- which is why the plain slot is unreadable for outlier; removing it is free and lifts "
        "the oracle readout 3.2x/4.5x over the k/n floor at 2k/8k "
        "(records/outlier-richer-slot-probe.md). It has to happen HERE, not in training: with "
        "detach_soft_kv the projector gets no gradient on pooled slots",
    )
    ap.add_argument(
        "--st-slot-stop-topk",
        type=int,
        default=100,
        help="softtoken: how many of the most frequent training-shard token ids go in the "
        "--st-slot-mode stop set (K=100 matches the eval-side cmean100 probe)",
    )
    ap.add_argument(
        "--st-slot-stop-rows",
        type=int,
        default=512,
        help="softtoken: how many shard rows' worth of tokens to take the stop-set frequencies "
        "over (read once at startup from the first token_ids_part_*.npy)",
    )
    ap.add_argument(
        "--st-slot-tokenizer",
        default=None,
        help="softtoken: tokenizer dir/identifier used ONLY to (a) also drop ids whose decoded "
        "piece has no alphanumeric character and (b) log a decoded sample of the stop set. "
        "Defaults to the family identifier; if it will not load the stop set is frequency-only "
        "(the top-K already covers the common punctuation) and says so",
    )
    ap.add_argument(
        "--torch-profile",
        action="store_true",
        help="profile steps 3-4 with torch.profiler and print the top ops (rank 0)",
    )
    ap.add_argument(
        "--st-neighbour-runs",
        type=int,
        default=0,
        help="softtoken: keep every selected doc (gold AND random) with its K neighbours on each side "
        "(leak-free runs; random budget divided by 2K+1). K=1 reaches full-attention parity on "
        "contradiction at --st-keep-frac 0.0278 (~11x) without any header",
    )
    ap.add_argument("--st-distill-prob", type=float, default=0.0)
    ap.add_argument("--st-distill-weight", type=float, default=1.0)
    ap.add_argument("--st-aux-weight", type=float, default=0.0)
    ap.add_argument("--st-mix-start-p", type=float, default=0.0)
    ap.add_argument("--st-mix-end-p", type=float, default=0.0)
    ap.add_argument("--st-mix-anneal-frac", type=float, default=1.0)
    ap.add_argument(
        "--pooled-gold-sidecar",
        default=None,
        help="pooledkv only: gold sidecar JSON ({content_fingerprint: gold ids or pairs}, e.g. "
        "from build_gold_sidecar_from_shard.py). Installs the gold-aware keep-set hook; without "
        "it the keep set is a gold-blind random --pooled-keep-prob fraction (control arm).",
    )
    ap.add_argument(
        "--pooled-n-random",
        type=int,
        default=2,
        help="pooledkv + sidecar: random non-gold docs kept real per example",
    )
    ap.add_argument(
        "--pooled-keep-mode",
        default="gold_plus_random",
        help="pooledkv + sidecar: select_keep_docs policy (gold_plus_random / gold_subsample / "
        "random_only / random_nongold / gold_pair / gold_halves)",
    )
    ap.add_argument(
        "--pooled-n-gold",
        type=int,
        default=0,
        help="pooledkv + sidecar: n_gold for the gold_subsample / gold_halves policies",
    )
    ap.add_argument(
        "--pooled-keep-prob",
        type=float,
        default=0.1,
        help="pooledkv without sidecar: per-doc probability of keeping real KV (seeded hash)",
    )
    ap.add_argument(
        "--pooled-no-len-bias",
        action="store_true",
        help="pooledkv: drop the +log(doc_len) slot logit bias (ablation; the biased form is the "
        "principled 'L copies of the mean KV entry' equivalence)",
    )
    ap.add_argument(
        "--seq-len", type=int, default=40960, help="fits the 32k rung + prompt/CoT overhead"
    )
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument(
        "--weight-decay", type=float, default=0.0, help="AdamW weight decay (embeddings stay at 0)"
    )
    ap.add_argument(
        "--z-loss", type=float, default=None, help="z-loss multiplier (OLMo 3 recipe: 1e-5); off by default"
    )
    ap.add_argument("--warmup-fraction", type=float, default=0.03)
    ap.add_argument("--scheduler", choices=["linear", "wsd"], default="linear")
    ap.add_argument(
        "--wsd-decay-fraction",
        type=float,
        default=0.1,
        help="--scheduler wsd: final fraction of max_duration that decays linearly to 0",
    )
    ap.add_argument(
        "--branch-from",
        default=None,
        help="a stepN checkpoint dir of a stable (wsd) run: load model + optimizer + trainer state "
        "from it (instead of --base-checkpoint) and train to --max-tokens in a FRESH save folder",
    )
    ap.add_argument(
        "--branch-optim-only",
        action="store_true",
        help="with --branch-from: load model + optimizer but NOT trainer state (fresh step counter and "
        "data-loader position), for continuing on a different shard / sequence length",
    )
    ap.add_argument(
        "--global-batch",
        type=int,
        default=8,
        help="instances per optimizer step across all ranks (global batch tokens = this * seq-len)",
    )
    ap.add_argument("--save-folder", default=None, help=f"default {SAVE_ROOT}/<run-name>")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="allow continuing step checkpoints that ANOTHER launch left in the save folder. "
        "Without it a new launch into such a folder fails (the trainer would silently resume); "
        "a preemption restart of the same Beaker experiment / slurm job resumes without it.",
    )
    ap.add_argument(
        "--skip-marker-audit",
        action="store_true",
        help="skip the start-of-run check that the base's marker embedding rows are repaired "
        "(recipes/data/fix_marker_embeddings.py)",
    )
    ap.add_argument("--run-name", default="ctc-suite-smoke")
    ap.add_argument("--wandb-group", default=None, help="default: the run name")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument(
        "--no-wandb",
        dest="wandb",
        action="store_false",
        help="disable wandb (ON by default; needs WANDB_API_KEY)",
    )
    ap.add_argument(
        "--base-checkpoint",
        default=None,
        help="model_and_optim distcp subdir of the converted per-scale base (default only for "
        f"0.8b: {DEFAULT_BASE_CHECKPOINT['0.8b']})",
    )
    ap.add_argument(
        "--work-dir",
        default=None,
        help=f"data-loader cache dir (default {WORK_DIR}); use node-local /data for speed",
    )
    ap.add_argument(
        "--micro-batch-instances",
        type=int,
        default=1,
        help="instances per rank per FORWARD (rank_microbatch_size = this * seq-len). Pure "
        "throughput knob (grad accumulation is exact); cannot exceed --global-batch / world_size",
    )
    ap.add_argument("--num-workers", type=int, default=2, help="dataloader workers per rank")
    ap.add_argument("--max-steps", type=int, default=0, help="hard-stop after N steps (0 = full)")
    ap.add_argument(
        "--no-flop-per-example",
        dest="flop_per_example",
        action="store_false",
        help="flop_meter: charge attention at --seq-len for every token (pre-2026-10-20 accounting) "
        "instead of each example at its own (compacted) length",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="train on the first N tokens of the arm with a full schedule (0 = whole arm); a sub-budget point",
    )
    ap.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="embedding-matrix size; default is the resolved family's base size "
        "(qwen3_5=248320, qwen3=151936). Override only to match a non-standard base.",
    )
    ap.add_argument(
        "--activation-checkpointing",
        choices=["auto", "full", "budget", "none"],
        default="auto",
        help="activation checkpointing. auto (default) = full for 4b/9b, none for 0.8b. "
        "USE 'budget' for context-parallel (--cp-degree) runs: full-block AC is incompatible with "
        "compile+CP (recompute-metadata mismatch at the dry-run); budget is the proven CP recipe.",
    )
    ap.add_argument(
        "--ac-budget",
        type=float,
        default=0.7,
        help="activation-memory budget for --activation-checkpointing budget (proven CP value 0.7; "
        "lower = more checkpointing / less memory if a long-context run OOMs).",
    )
    ap.add_argument(
        "--shard-degree",
        type=int,
        default=0,
        help="FSDP shard_degree. 0 (default/auto) = 1 for 0.8b (proven working), world_size for "
        "4b/9b (full parameter sharding, needed to fit seq_len=40960 on an 80GB H100). Override "
        "to force a specific value. Under CP this shards over the dp dimension (world_size/cp).",
    )
    ap.add_argument(
        "--cp-degree",
        type=int,
        default=0,
        help="context-parallel (Ulysses) degree for long-context arms. 0/1 = no CP (default). "
        "Shards the sequence over cp_degree GPUs; bounded by n_kv_heads (dense qwen3=8, hybrid "
        "qwen3_5=4). world_size must be divisible by it; the dp/FSDP dim becomes world_size/cp.",
    )
    ap.add_argument(
        "--rope-yarn-factor",
        type=float,
        default=0.0,
        help="YaRN RoPE context-extension factor (0/1 = off). E.g. 8 extends Qwen3's native 32k "
        "to 256k; 2 for 64k. Required whenever --seq-len exceeds the base's native context.",
    )
    ap.add_argument(
        "--rope-old-context",
        type=int,
        default=32768,
        help="base native context length for YaRN scaling (Qwen3 = 32768).",
    )
    ap.add_argument(
        "--rope-theta",
        type=float,
        default=0.0,
        help="override base rope_theta for NTK-style context extension (0 = factory default). "
        "For Qwen3-4B (native 32k) at 256k use ~8e6. Cleaner + more robust than --rope-yarn-factor "
        "for large extensions; do not combine with --rope-yarn-factor.",
    )
    ap.add_argument(
        "--pack",
        action="store_true",
        help="bin-pack whole examples into seq_len windows (PackingInstanceSource) instead of one "
        "padded example per instance. Required for a mixed-length (e.g. 8k..256k) shard so short "
        "examples don't waste compute padded up to the max seq_len. Composes with CP.",
    )
    ap.add_argument(
        "--attn-backend",
        choices=["flash_2", "torch"],
        default="flash_2",
        help="attention backend for the full-attention blocks (default flash_2; use torch/SDPA "
        "on a cluster where flash-attn is not importable or not verified for the GPU arch)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="run-to-run seed offset (shifts data order + torch seed) for noise-floor runs",
    )
    ap.add_argument("--mix-start-p", type=float, default=0.80)
    ap.add_argument("--mix-end-p", type=float, default=0.0)
    ap.add_argument("--mix-seed", type=int, default=42)
    ap.add_argument("--mix-log-interval", type=int, default=5)
    ap.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="after fit, save model-only checkpoint (config.json + model_and_optim) for eval",
    )
    ap.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="steps between mid-run model+optim checkpoints. Default None = none written (eval only "
        "needs the post-fit model-only save, and at 4B each mid-run checkpoint costs ~55G). Set an "
        "integer to get resume points for a long run on a preemptible queue.",
    )
    ap.add_argument(
        "--max-checkpoints",
        type=int,
        default=3,
        help="permanent checkpoints kept in the save folder (CheckpointerCallback.max_checkpoints); 0 = keep all. "
        "The default 3 silently deletes older --save-interval checkpoints, e.g. a decay-branch point.",
    )
    ap.add_argument(
        "--no-final-checkpoint",
        action="store_true",
        help="skip the checkpointer's end-of-training FULL (model+optim+train-state) save; the "
        "post-fit model-only export from --save-checkpoint still runs. Use on lambda, where the "
        "full save costs ~82G of user quota and ~25 min of NFS writes that nothing reads. "
        "Incompatible with resuming the run later.",
    )
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build configs (model, data loader, curriculum math) WITHOUT CUDA/distributed and "
        "print the resolved plan; trains nothing, writes nothing",
    )
    ap.add_argument(
        "--dry-run-world-size",
        type=int,
        default=8,
        help="hypothetical DP world size for --dry-run curriculum math",
    )
    ap.add_argument(
        "--dry-run-n-examples",
        type=int,
        default=0,
        help="--dry-run only: pretend the shard has this many examples (0 = use metadata)",
    )
    opts = ap.parse_args()
    opts._mix_total_forwards = 0
    # Resolve the model family (explicit or auto from the shard) and rebind opts.model_family to
    # the concrete family, so every downstream function keys off it deterministically.
    opts.model_family = resolve_family(opts)
    if opts.model_scale not in MODEL_FACTORIES[opts.model_family]:
        ap.error(
            f"--model-scale {opts.model_scale!r} is not defined for family "
            f"{opts.model_family!r}; choose from {sorted(MODEL_FACTORIES[opts.model_family])}"
        )
    if opts.vocab_size is None:
        opts.vocab_size = FAMILY_VOCAB_SIZE[opts.model_family]
    try:
        RG.check_token_budget(opts.max_tokens, opts.global_batch, opts.seq_len)
    except ValueError as exc:
        ap.error(str(exc))
    if not opts.dry_run and opts.dry_run_n_examples:
        ap.error("--dry-run-n-examples is only valid with --dry-run")
    if not 0.0 <= opts.layer_drop_p < 1.0:
        ap.error(f"--layer-drop-p must be in [0, 1), got {opts.layer_drop_p}")
    if opts.train_skip_layers:
        from olmo_core.nn.layer_drop import parse_layer_list

        try:
            parse_layer_list(opts.train_skip_layers)
        except ValueError as exc:
            ap.error(f"--train-skip-layers: {exc}")
    if opts.variant == "chunked-mix" and opts.compile:
        # The python-seeded mix coin + counter are not torch.compile-capturable.
        print("[ctc-suite] chunked-mix: forcing --no-compile (mix coin is not compilable)")
        opts.compile = False
    if opts.variant == "sparselandmark" and opts.compile:
        # SparseLandmarkAttention is a Triton custom-autograd Function; keep compile off (same
        # posture as every sparse-landmark CPT/SFT script) rather than risk AC+compile metadata
        # mismatches mid-sweep.
        print("[ctc-suite] sparselandmark: forcing --no-compile (triton custom-autograd mixer)")
        opts.compile = False
    if opts.variant == "pooledkv":
        if opts.pack:
            # The keep-set fingerprint + role reconstruction assume ONE example per padded row
            # (everything after the first EOS is PAD); packed rows would fingerprint-miss every
            # example and silently keep all docs real.
            ap.error("--variant pooledkv requires the padded (no --pack) data path")
        if opts.compile:
            # Data-dependent shapes (per-batch doc count) + host-side keep-set resolution.
            print("[ctc-suite] pooledkv: forcing --no-compile (data-dependent pooled-KV shapes)")
            opts.compile = False
    if opts.variant != "pooledkv" and (opts.pooled_gold_sidecar or opts.pooled_no_len_bias):
        ap.error("--pooled-* options are only valid with --variant pooledkv")
    if opts.variant == "softtoken":
        if opts.pack:
            # Packed rows run in SEGMENT mode: the model unpacks each row to one example per row
            # (per-example gold fingerprints), compacts, and re-packs the compacted examples into
            # one varlen row. Defined only for slot-less compaction with no extra attention bias.
            bad = [
                flag
                for flag, on in (
                    ("--st-drop-slots is required", not opts.st_drop_slots),
                    ("--st-aux-weight", opts.st_aux_weight > 0),
                    ("--st-len-bias", opts.st_len_bias),
                    ("--st-distill-prob", opts.st_distill_prob > 0),
                )
                if on
            ]
            if bad:
                ap.error(f"--variant softtoken with --pack (segment mode): {', '.join(bad)}")
        if opts.st_block_size > 0 and (not opts.st_gold_blind or opts.st_header_stop_id is not None):
            ap.error("--st-block-size needs --st-gold-blind and no --st-header-stop-id")
        if (opts.st_keep_recent > 0 or opts.st_keep_ramp) and opts.st_block_size <= 0:
            ap.error("--st-keep-recent / --st-keep-ramp need --st-block-size > 0")
        if opts.st_tokexit_layer >= 0 and (opts.st_block_size <= 0 or not opts.st_drop_slots):
            ap.error("--tokexit-layer needs --st-block-size > 0 and --st-drop-slots")
        if opts.st_tokexit_layer >= 0 and opts.pack:
            ap.error("--tokexit-layer runs on padded one-example rows (no --pack)")
        if opts.compile:
            print("[ctc-suite] softtoken: forcing --no-compile (data-dependent compacted shapes)")
            opts.compile = False
    # The compression-mixing curriculum is only implemented on the softtoken path (gold-aware hook
    # or gold-blind keep draw). It used to be parsed everywhere and applied nowhere else -- and, up
    # to 2026-09-14, nowhere at all under --st-gold-blind, which turned the ds64 `kvgbmix` arm into
    # a byte-identical re-run of `kvgb` with no warning. Refuse the flags instead of ignoring them.
    mix_requested = (
        opts.st_mix_start_p > 0.0 or opts.st_mix_end_p > 0.0 or opts.st_mix_anneal_frac != 1.0
    )
    if opts.st_block_size > 0 and opts.variant != "softtoken":
        ap.error(f"--st-block-size is only honoured by --variant softtoken (got {opts.variant!r})")
    if opts.variant != "softtoken" and (
        opts.st_keep_recent > 0 or opts.st_keep_ramp or opts.st_tokexit_layer >= 0
    ):
        ap.error("--st-keep-recent / --st-keep-ramp / --tokexit-layer need --variant softtoken")
    if opts.st_slot_mode != "mean" and opts.variant != "softtoken":
        ap.error(
            f"--st-slot-mode is only honoured by --variant softtoken (got {opts.variant!r}); "
            "it would be ignored."
        )
    if opts.st_header_extra_tokens > 0 and opts.variant != "softtoken":
        ap.error(
            "--st-header-extra-tokens is only honoured by --variant softtoken "
            f"(got --variant {opts.variant!r}); it would be ignored."
        )
    if opts.st_keep_token_rule != "none":
        if opts.variant != "softtoken":
            ap.error(
                "--st-keep-token-rule is only honoured by --variant softtoken "
                f"(got --variant {opts.variant!r}); it would be ignored."
            )
        if opts.st_keep_token_rule == "online":
            # the online router's budget is --st-otr-keep-frac, not --st-keep-token-k
            if not 0.0 < opts.st_otr_keep_frac <= 1.0:
                ap.error("--st-keep-token-rule online needs 0 < --st-otr-keep-frac <= 1.")
        elif opts.st_keep_token_k <= 0:
            ap.error(f"--st-keep-token-rule {opts.st_keep_token_rule} needs --st-keep-token-k > 0.")
        # first_last / rule ON TOP of a fixed real prefix is the devloss grid's gold_fl20p8
        # construction (the prefix keeps the `Document [N]:` id; the K budget then applies to the
        # rest of the body). Only `first` duplicates the prefix flag.
        if opts.st_header_extra_tokens > 0 and opts.st_keep_token_rule == "first":
            ap.error(
                "--st-keep-token-rule first IS --st-header-extra-tokens; pass only one "
                "(first_last / rule may be combined with a --st-header-extra-tokens prefix)."
            )
        if opts.st_keep_token_rule == "rule":
            from olmo_core.nn.pooled_soft_token import parse_keep_token_weights

            try:
                parse_keep_token_weights(opts.st_keep_token_weights)
            except ValueError as exc:
                ap.error(f"--st-keep-token-weights: {exc}")
        elif opts.st_keep_token_weights is not None:
            ap.error(
                f"--st-keep-token-weights only applies to --st-keep-token-rule rule (got "
                f"{opts.st_keep_token_rule!r}); it would be ignored."
            )
    elif opts.st_keep_token_k > 0 or opts.st_keep_token_weights is not None:
        ap.error(
            "--st-keep-token-k/--st-keep-token-weights were given but --st-keep-token-rule is "
            "'none', so they would be ignored."
        )
    if opts.st_kept_doc_k is not None and opts.st_keep_token_rule == "none":
        ap.error("--st-kept-doc-k (two-tier keep) needs a --st-keep-token-rule for the tiers")
    if mix_requested and opts.variant != "softtoken":
        ap.error(
            "--st-mix-start-p/--st-mix-end-p/--st-mix-anneal-frac are only honoured by "
            f"--variant softtoken (got --variant {opts.variant!r}); they would be ignored."
        )
    if mix_requested and opts.st_mix_start_p <= 0.0 and opts.st_mix_end_p <= 0.0:
        ap.error(
            "--st-mix-anneal-frac was given but both --st-mix-start-p and --st-mix-end-p are 0, "
            "so the compression-mixing curriculum would never fire."
        )
    if mix_requested and opts.st_mix_anneal_frac <= 0.0:
        ap.error("--st-mix-anneal-frac must be > 0 (it scales the anneal horizon in forwards).")
    if opts.variant in ("ffnmoe", "flexcompute") and opts.compile:
        print("[ctc-suite] ffnmoe: forcing --no-compile (data-dependent per-rung GEMM shapes)")
        opts.compile = False
    if kv_route_enabled(opts) and opts.compile:
        print("[ctc-suite] kvroute: forcing --no-compile (FlexAttention compiles its own kernel)")
        opts.compile = False
    if opts.variant in ("kvroute", "flexcompute") and not opts.pack:
        ap.error("--variant kvroute/flexcompute expects the packed (--pack) data path")
    return opts


def main() -> None:
    """CLI entrypoint: dry-run without distributed init, otherwise train under torchrun."""
    # `kill -USR1 <pid>` dumps every thread's stack into the log (hung-job diagnosis, no py-spy).
    RG.install_stack_dump_handler()
    opts = parse_args()
    if opts.dry_run:
        dry_run(opts)
        return
    prepare_training_environment()
    try:
        build_and_fit(opts)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
