"""
Example script to convert a MaxText checkpoint to an OLMo Core model checkpoint.

The conversion logic lives in :mod:`olmo_core.nn.maxtext.convert_checkpoint`; this script is a
thin CLI wrapper. It needs a MaxText checkout that has the model's decoder installed in the same
environment (CPU JAX is enough).

Warnings:
    - By default only model weights are converted, and OLMo Core starts the optimizer fresh. Pass
      ``--optimizer-state`` to also convert the AdamW moments and step (MaxText's
      ``opt_type=adamw``). They're written for the OLMoDDP train module; the FSDP train module
      loads only the weights.
    - MaxText checkpoints don't describe the OLMo Core model, so this needs an OLMo Core
      ``config.json``: the copy that ``convert_checkpoint_to_maxtext.py`` saved next to the
      checkpoint, or one passed with ``--olmo-core-config``.

Usage::

    python convert_checkpoint_from_maxtext.py \\
        -i gs://bucket/maxtext-run/checkpoints/50/items -o gs://bucket/run/maxtext-step50 \\
        --olmo-core-config gs://bucket/run/step1000/config.json
"""

from argparse import ArgumentParser

from olmo_core.nn.maxtext import convert_checkpoint_from_maxtext
from olmo_core.nn.maxtext.checkpoint import read_json
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
        "--optimizer-state",
        action="store_true",
        help="Also convert the AdamW moments and step, as OLMoDDP optimizer state.",
    )
    parser.add_argument(
        "--optim-key-prefix",
        default="module.",
        help="Prefix the OLMoDDP train module puts before parameter names in its optimizer state.",
    )
    parser.add_argument(
        "--skip-validation",
        dest="validate",
        action="store_false",
        help="Skip checking that converting back reproduces the MaxText tensors.",
    )
    parser.add_argument("--save-overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    convert_checkpoint_from_maxtext(
        args.maxtext_input_path,
        args.checkpoint_output_dir,
        experiment_config=read_json(args.olmo_core_config) if args.olmo_core_config else None,
        optimizer_state=args.optimizer_state,
        optim_key_prefix=args.optim_key_prefix,
        validate=args.validate,
        save_overwrite=args.save_overwrite,
    )


if __name__ == "__main__":
    prepare_cli_environment()
    main()
