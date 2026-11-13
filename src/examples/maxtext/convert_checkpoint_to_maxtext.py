"""
Example script to convert an OLMo Core model checkpoint to a MaxText checkpoint.

The conversion logic lives in :mod:`olmo_core.nn.maxtext.convert_checkpoint`; this script is a
thin CLI wrapper. It needs a MaxText checkout that has the model's decoder installed in the same
environment (CPU JAX is enough). Only weights are converted.

Usage::

    python convert_checkpoint_to_maxtext.py -i gs://bucket/run/step1000 -o gs://bucket/maxtext/run-step1000

Then load it in MaxText with ``load_parameters_path=<output>/0/items``; the script logs the MaxText
``model_name`` and overrides that match the model.
"""

from argparse import ArgumentParser

from olmo_core.nn.maxtext import convert_checkpoint_to_maxtext
from olmo_core.utils import prepare_cli_environment


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--checkpoint-input-path",
        required=True,
        help="Local or remote OLMo Core checkpoint step directory (with model_and_optim/ and config.json).",
    )
    parser.add_argument(
        "-o",
        "--maxtext-output-dir",
        required=True,
        help="Local or remote directory for the MaxText checkpoint. Weights land in <output>/0/items.",
    )
    parser.add_argument(
        "--unscanned",
        dest="scan_layers",
        action="store_false",
        help="Write the layout for MaxText's scan_layers=False instead of scan_layers=True.",
    )
    parser.add_argument(
        "--skip-validation",
        dest="validate",
        action="store_false",
        help="Skip checking that converting back reproduces the original weights.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    convert_checkpoint_to_maxtext(
        args.checkpoint_input_path,
        args.maxtext_output_dir,
        scan_layers=args.scan_layers,
        validate=args.validate,
    )


if __name__ == "__main__":
    prepare_cli_environment()
    main()
