"""
Convert OLMoE3 (OLMo 3.5) checkpoints between OLMo Core and MaxText.

Weights only: optimizer state is not converted, so the target framework starts its optimizer
fresh. Every parameter is converted with a pure re-layout, so converting there and back
reproduces the original weights bit for bit (check with ``compare``).

Requires the MaxText checkout with the ``olmoe3`` decoder installed in the same environment (CPU
JAX is fine)::

    # OLMo Core -> MaxText; load in MaxText with load_parameters_path=<out>/0/items
    python src/scripts/maxtext/convert_olmoe3.py to-maxtext \\
        gs://bucket/run/step1000 gs://bucket/maxtext/run-step1000

    # MaxText -> OLMo Core (config.json comes from --olmo-core-config, or the copy that
    # to-maxtext leaves next to the checkpoint)
    python src/scripts/maxtext/convert_olmoe3.py to-olmo-core \\
        gs://bucket/maxtext-run/checkpoints/50/items gs://bucket/run/maxtext-step50 \\
        --olmo-core-config gs://bucket/run/step1000/config.json

    # Bit-exact comparison of two OLMo Core checkpoints' weights
    python src/scripts/maxtext/convert_olmoe3.py compare gs://bucket/run/step1000 /tmp/round-trip

Full-size checkpoints are held in memory in fp32 (about 50 GB for the 810m / 12.5B-total model),
so run this on a high-memory machine.
"""

import argparse
import copy
import json
import logging
import sys
from typing import Any, Dict

import numpy as np
from cached_path import cached_path

from olmo_core.io import file_exists, normalize_path
from olmo_core.nn.hf.convert_checkpoint import _normalize_legacy_latent_moe_config
from olmo_core.nn.maxtext.checkpoint import (
    load_maxtext_params,
    load_olmo_core_config,
    load_olmo_core_weights,
    save_maxtext_params,
    save_olmo_core_weights,
    write_json,
)
from olmo_core.nn.maxtext.olmoe3 import (
    OLMoE3Geometry,
    maxtext_to_olmo_core,
    olmo_core_to_maxtext,
    scan_maxtext,
    unscan_maxtext,
)
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)

#: Written next to a converted MaxText checkpoint so it can be converted back without a template.
SIDECAR_CONFIG = "olmo_core_config.json"


def _geometry(experiment_config: Dict[str, Any]) -> OLMoE3Geometry:
    model_config = copy.deepcopy(experiment_config["model"])
    _normalize_legacy_latent_moe_config(model_config)
    return OLMoE3Geometry.from_olmo_core_config(model_config)


def _read_json(path: str) -> Dict[str, Any]:
    with cached_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def to_maxtext(args: argparse.Namespace) -> None:
    config = load_olmo_core_config(args.input)
    geometry = _geometry(config)
    state = load_olmo_core_weights(args.input, geometry)
    params = olmo_core_to_maxtext(state, geometry)
    del state
    if not args.unscanned:
        params = scan_maxtext(params, geometry.n_layers)
    save_maxtext_params(args.output, params)

    write_json(f"{normalize_path(args.output)}/{SIDECAR_CONFIG}", config, save_overwrite=True)
    layout = "unscanned (scan_layers=False)" if args.unscanned else "scanned (scan_layers=True)"
    log.info(
        f"Wrote {layout} MaxText checkpoint; load it with load_parameters_path={args.output}/0/items"
    )


def to_olmo_core(args: argparse.Namespace) -> None:
    config_path = args.olmo_core_config
    if config_path is None:
        # <out>/0/items -> <out>/olmo_core_config.json, as written by to-maxtext.
        root = normalize_path(args.input).rstrip("/").rsplit("/", 2)[0]
        config_path = f"{root}/{SIDECAR_CONFIG}"
        if not file_exists(config_path):
            raise FileNotFoundError(
                f"no {SIDECAR_CONFIG} next to {args.input}; pass --olmo-core-config"
            )
    config = _read_json(config_path)
    geometry = _geometry(config)
    params = load_maxtext_params(args.input)
    if any("scanned_blocks/" in k or "/layer_" in k for k in params):
        params = unscan_maxtext(params)
    state = maxtext_to_olmo_core(params, geometry)
    del params
    save_olmo_core_weights(args.output, state, config, save_overwrite=args.save_overwrite)
    log.info(f"Wrote OLMo Core checkpoint to {args.output}")


def compare(args: argparse.Namespace) -> None:
    geometry = _geometry(load_olmo_core_config(args.a))
    a = load_olmo_core_weights(args.a, geometry)
    b = load_olmo_core_weights(args.b, geometry)
    mismatched = [k for k in a if not np.array_equal(a[k], b[k])]
    for k in mismatched[:20]:
        diff = np.abs(a[k].astype(np.float64) - b[k].astype(np.float64))
        log.error(f"{k}: max |diff| {diff.max():.3e}, {np.count_nonzero(diff)} elements differ")
    if mismatched:
        log.error(f"{len(mismatched)} / {len(a)} parameters differ")
        sys.exit(1)
    log.info(f"All {len(a)} parameters are bit-identical")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("to-maxtext", help="OLMo Core checkpoint -> MaxText checkpoint")
    p.add_argument(
        "input", help="OLMo Core checkpoint step dir (holds model_and_optim/ and config.json)"
    )
    p.add_argument("output", help="MaxText output dir; weights land in <output>/0/items")
    p.add_argument(
        "--unscanned", action="store_true", help="write the scan_layers=False layout instead"
    )
    p.set_defaults(func=to_maxtext)

    p = sub.add_parser("to-olmo-core", help="MaxText checkpoint -> OLMo Core checkpoint")
    p.add_argument("input", help="MaxText checkpoint items dir, e.g. .../checkpoints/50/items")
    p.add_argument("output", help="OLMo Core output dir")
    p.add_argument(
        "--olmo-core-config",
        help="OLMo Core config.json describing the model; defaults to the copy to-maxtext wrote",
    )
    p.add_argument("--save-overwrite", action="store_true")
    p.set_defaults(func=to_olmo_core)

    p = sub.add_parser("compare", help="check two OLMo Core checkpoints hold identical weights")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=compare)

    args = parser.parse_args()
    prepare_cli_environment()
    args.func(args)


if __name__ == "__main__":
    main()
