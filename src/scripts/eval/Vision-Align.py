"""Evaluate vision-alignment checkpoints independently of training."""

import argparse
import importlib
import sys


def main() -> None:
    """Dispatch to a maintained text, decoded-vision, or academic panel runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("fast-text", "decoded", "academic"))
    args = parser.parse_args(sys.argv[1:2])
    module = importlib.import_module(f"olmo_core.eval.vision_{args.suite.replace('-', '_')}")
    module.main(sys.argv[2:])


if __name__ == "__main__":
    main()
