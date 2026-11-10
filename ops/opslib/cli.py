"""Argparse dispatcher: each module exposes ``add_parser(subparsers)`` and sets ``func`` defaults."""
import argparse
import importlib
import sys

MODULES = ["registry", "results", "backfill", "jobs", "capacity", "guard", "experiments"]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ops", description="Research-ops toolkit (see ops/README.md)")
    sub = parser.add_subparsers(dest="cmd")
    for name in MODULES:
        try:
            mod = importlib.import_module(f"opslib.{name}")
        except ImportError:
            continue
        if hasattr(mod, "add_parser"):
            mod.add_parser(sub)
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
