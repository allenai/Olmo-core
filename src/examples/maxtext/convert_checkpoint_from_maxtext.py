"""
Example script to convert a MaxText checkpoint to an OLMo Core model checkpoint.

The conversion logic lives in :mod:`olmo_core.nn.maxtext.convert_checkpoint`; this script is a
thin CLI wrapper. It needs a MaxText checkout that has the model's decoder installed in the same
environment (CPU JAX is enough).

Warnings:
    - Only model weights are converted. Without optimizer state, OLMo Core starts the optimizer
      fresh, so this is not a true resume.
    - MaxText checkpoints don't describe the OLMo Core model, so this needs an OLMo Core
      ``config.json``: the copy that ``convert_checkpoint_to_maxtext.py`` saved next to the
      checkpoint, or one passed with ``--olmo-core-config``.

Usage::

    python convert_checkpoint_from_maxtext.py \\
        -i gs://bucket/maxtext-run/checkpoints/50/items -o gs://bucket/run/maxtext-step50 \\
        --olmo-core-config gs://bucket/run/step1000/config.json
"""

import json
from argparse import ArgumentParser

from cached_path import cached_path

from olmo_core.nn.maxtext import convert_checkpoint_from_maxtext
from olmo_core.utils import prepare_cli_environment


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--maxtext-input-path",
        required=True,
        help="A MaxText checkpoint step's items directory, e.g. .../checkpoints/50/items.",
    )
    parser.add_argument(
        "-o",
        "--checkpoint-output-dir",
        required=True,
        help="Local or remote directory for the OLMo Core checkpoint.",
    )
    parser.add_argument(
        "--olmo-core-config",
        help="OLMo Core config.json describing the model. Defaults to the copy saved next to the checkpoint.",
    )
    parser.add_argument(
        "--skip-validation",
        dest="validate",
        action="store_false",
        help="Skip checking that converting back reproduces the MaxText weights.",
    )
    parser.add_argument("--save-overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_config = None
    if args.olmo_core_config:
        with cached_path(args.olmo_core_config).open("r", encoding="utf-8") as f:
            experiment_config = json.load(f)
    convert_checkpoint_from_maxtext(
        args.maxtext_input_path,
        args.checkpoint_output_dir,
        experiment_config=experiment_config,
        validate=args.validate,
        save_overwrite=args.save_overwrite,
    )


if __name__ == "__main__":
    prepare_cli_environment()
    main()
