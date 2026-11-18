"""LeakySFT contradiction-only Qwen3.5-4B: a10; see the shared builder."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _qwen35_contradiction_common import build_leaky_experiment  # noqa: E402
from olmo_core.internal.experiment import main  # noqa: E402


def build_experiment_config(cli_context):
    return build_leaky_experiment(cli_context, arm="a10")


if __name__ == "__main__":
    main(config_builder=build_experiment_config)
