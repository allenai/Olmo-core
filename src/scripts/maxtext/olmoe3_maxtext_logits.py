"""
Run a converted OLMoE3 (OLMo 3.5) checkpoint in MaxText on the tokens of a reference logit summary
(from ``olmoe3_reference_logits.py``) and compare.

Weights load through MaxText's own ``load_parameters_path`` path, so this also checks that MaxText
accepts the converted checkpoint. Runs in fp32 with ``matmul_precision=highest``. On CPU::

    JAX_PLATFORMS=cpu python src/scripts/maxtext/olmoe3_maxtext_logits.py \\
        --checkpoint /tmp/olmoe3-maxtext/0/items --reference ref.npz

Anything after ``--`` is passed to MaxText as extra ``key=value`` overrides (e.g. sharding
settings on TPU). Exits non-zero if a threshold fails.
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
from cached_path import cached_path

from olmo_core.io import normalize_path
from olmo_core.nn.hf.convert_checkpoint import _normalize_legacy_latent_moe_config
from olmo_core.nn.maxtext.checkpoint import load_maxtext_params
from olmo_core.nn.maxtext.olmoe3 import OLMoE3Geometry, maxtext_overrides
from olmo_core.nn.maxtext.parity import (
    LogitSummary,
    compare_summaries,
    concat,
    summarize_logits,
)
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


def _load_config(args) -> dict:
    path = args.olmo_core_config
    if path is None:
        root = normalize_path(args.checkpoint).rstrip("/").rsplit("/", 2)[0]
        path = f"{root}/olmo_core_config.json"
    with cached_path(path).open("r", encoding="utf-8") as f:
        config = json.load(f)
    model_config = config.get("model", config)
    _normalize_legacy_latent_moe_config(model_config)
    return model_config


def _check_loaded(model, checkpoint: str) -> None:
    """
    Make sure MaxText is running the checkpoint's weights: a model that silently fell back to
    its own init would otherwise fail (or worse, pass a self-comparison) for the wrong reason.
    """
    import jax  # type: ignore
    from flax import nnx  # type: ignore

    stored = load_maxtext_params(checkpoint)
    loaded = {
        "/".join(str(p) for p in path): v
        for path, v in nnx.to_flat_state(nnx.state(model, nnx.Param))
    }
    if loaded.keys() != stored.keys():
        raise KeyError(
            f"model/checkpoint params differ: only in model {sorted(loaded.keys() - stored.keys())[:5]}, "
            f"only in checkpoint {sorted(stored.keys() - loaded.keys())[:5]}"
        )
    for k, v in loaded.items():
        value = np.asarray(jax.experimental.multihost_utils.process_allgather(v.value, tiled=True))
        if not np.array_equal(value, stored[k]):
            raise ValueError(f"{k}: the model's weights are not the checkpoint's")
    log.info(f"All {len(loaded)} MaxText params match the checkpoint")


def main() -> None:
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        extra = argv[argv.index("--") + 1 :]
        argv = argv[: argv.index("--")]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="MaxText checkpoint items dir")
    parser.add_argument(
        "--olmo-core-config",
        help="OLMo Core config.json for the model; defaults to the copy convert_olmoe3.py wrote",
    )
    parser.add_argument("--reference", required=True, help="reference .npz summary")
    parser.add_argument("--unscanned", action="store_true", help="checkpoint is scan_layers=False")
    parser.add_argument("--output", help="also save MaxText's summary here")
    parser.add_argument("--max-rel-logit-err", type=float, default=1e-3)
    parser.add_argument("--max-mean-kl", type=float, default=1e-4)
    parser.add_argument("--min-top1-agreement", type=float, default=0.999)
    parser.add_argument("--max-mean-abs-dloss", type=float, default=1e-3)
    args = parser.parse_args(argv)
    # MaxText code paths may hand sys.argv to absl, which rejects this script's flags.
    sys.argv = sys.argv[:1]
    prepare_cli_environment()

    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
    from maxtext.common.common_types import MODEL_MODE_TRAIN  # type: ignore
    from maxtext.configs import pyconfig  # type: ignore
    from maxtext.utils import maxtext_utils, model_creation_utils  # type: ignore
    from maxtext.utils.globals import MAXTEXT_PKG_DIR  # type: ignore

    geometry = OLMoE3Geometry.from_olmo_core_config(_load_config(args))
    reference = LogitSummary.load(args.reference)
    B, S = reference.tokens.shape

    # MaxText samples EMO pool sizes whenever the call isn't in inference mode, regardless of
    # enable_dropout, while OLMo Core samples only in train(). Pin the pool to the eval size.
    emo = []
    if geometry.emo_pools is not None:
        pool = geometry.emo_pools[2]
        emo = [f"emo_min_document_expert_pool={pool}", f"emo_max_document_expert_pool={pool}"]

    config = pyconfig.initialize(
        [
            "",
            os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
            "model_name=olmo35-tiny",
            "override_model_config=True",
            *maxtext_overrides(geometry),
            *emo,
            "run_name=olmoe3_parity",
            "enable_checkpointing=True",
            f"load_parameters_path={args.checkpoint}",
            f"scan_layers={not args.unscanned}",
            "skip_jax_distributed_system=True",
            f"max_target_length={S}",
            "dtype=float32",
            "weight_dtype=float32",
            "matmul_precision=highest",
            "use_tokamax_kda=False",
            "per_device_batch_size=1",
            *extra,
        ]
    )
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(config), config.mesh_axes)
    model = model_creation_utils.from_pretrained(config, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
    _check_loaded(model, args.checkpoint)

    global_batch = config.global_batch_size_to_load
    positions = jnp.broadcast_to(jnp.arange(S, dtype=jnp.int32), (global_batch, S))
    segments = jnp.ones((global_batch, S), dtype=jnp.int32)
    summaries = []
    for row in reference.tokens:
        # One sequence per call, replicated across the global batch.
        ids = jnp.broadcast_to(jnp.asarray(row, dtype=jnp.int32), (global_batch, S))
        logits = model(
            decoder_input_tokens=ids,
            decoder_positions=positions,
            decoder_segment_ids=segments,
            enable_dropout=False,
        )
        logits = np.asarray(
            jax.experimental.multihost_utils.process_allgather(logits, tiled=True)[:1],
            dtype=np.float32,
        )
        summaries.append(summarize_logits(row[None], logits, reference.positions))
    actual = concat(summaries)
    if args.output:
        actual.save(args.output)

    metrics = compare_summaries(reference, actual)
    for k, v in metrics.items():
        log.info(f"{k:>20}: {v:.3e}")
    failures = [
        name
        for name, ok in (
            ("max_rel_logit_err", metrics["max_rel_logit_err"] <= args.max_rel_logit_err),
            ("mean_kl", metrics["mean_kl"] <= args.max_mean_kl),
            ("top1_agreement", metrics["top1_agreement"] >= args.min_top1_agreement),
            ("mean_abs_dloss", metrics["mean_abs_dloss"] <= args.max_mean_abs_dloss),
        )
        if not ok
    ]
    if failures:
        log.error(f"FAILED: {', '.join(failures)}")
        sys.exit(1)
    log.info(f"PASSED on {B} x {S} tokens")


if __name__ == "__main__":
    main()
