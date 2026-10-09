"""
Run a converted checkpoint in MaxText on the tokens of a reference logit summary (from
``reference_logits.py``) and compare the two.

Weights load through MaxText's own ``load_parameters_path``, so this also checks that MaxText
accepts the converted checkpoint. Runs in fp32 with ``matmul_precision=highest``. On CPU::

    JAX_PLATFORMS=cpu python src/examples/maxtext/maxtext_logits.py \\
        --checkpoint /tmp/maxtext/0/items --reference ref.npz

Anything after ``--`` is passed to MaxText as extra ``key=value`` overrides (e.g. sharding
settings on TPU). Exits non-zero if a threshold fails.
"""

import argparse
import json
import logging
import sys

from cached_path import cached_path

from olmo_core.nn.maxtext import get_maxtext_config, load_olmo_core_config_for_maxtext
from olmo_core.nn.maxtext.parity import (
    LogitSummary,
    compare_summaries,
    maxtext_logit_summary,
)
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


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
        help="OLMo Core config.json for the model; defaults to the copy saved next to the checkpoint",
    )
    parser.add_argument("--reference", required=True, help="reference .npz summary")
    parser.add_argument("--unscanned", action="store_true", help="checkpoint is scan_layers=False")
    parser.add_argument("--output", help="also save MaxText's summary here")
    parser.add_argument("--max-rel-logit-err", type=float, default=1e-3)
    parser.add_argument("--max-mean-kl", type=float, default=1e-4)
    parser.add_argument("--min-top1-agreement", type=float, default=0.999)
    parser.add_argument("--max-mean-abs-dloss", type=float, default=1e-3)
    args = parser.parse_args(argv)
    prepare_cli_environment()

    if args.olmo_core_config:
        with cached_path(args.olmo_core_config).open("r", encoding="utf-8") as f:
            experiment_config = json.load(f)
    else:
        experiment_config = load_olmo_core_config_for_maxtext(args.checkpoint)
    config = get_maxtext_config(experiment_config["model"])
    reference = LogitSummary.load(args.reference)
    actual = maxtext_logit_summary(
        config,
        args.checkpoint,
        reference.tokens,
        reference.positions,
        scan_layers=not args.unscanned,
        extra_overrides=extra,
    )
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
    B, S = reference.tokens.shape
    log.info(f"PASSED on {B} x {S} tokens")


if __name__ == "__main__":
    main()
