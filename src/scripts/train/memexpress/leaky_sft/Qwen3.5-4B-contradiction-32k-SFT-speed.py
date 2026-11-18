"""Regular SFT vs LeakySFT speed, with the same two-GPU contradiction recipe.

Readout: wall-clock seconds per step over steps 51–300 against the same steps
of the LeakySFT arms; exclude checkpoint intervals for steady-state timing.
Only the loss path and hard stop differ. Keep the one-epoch LR schedule and
checkpoint cadence; the hard stop does not rescale the scheduler to 300 steps.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _qwen35_contradiction_common import build_leaky_experiment  # noqa: E402
from olmo_core.internal.experiment import main  # noqa: E402
from olmo_core.train import Duration  # noqa: E402


def build_experiment_config(cli_context):
    config = build_leaky_experiment(cli_context, arm="a00")
    # None uses the original SFT path: no context CE or LeakySFT diagnostics.
    config.train_module.cpt_loss_scaler = None
    config.trainer.hard_stop = Duration.steps(300)
    return config


if __name__ == "__main__":
    main(config_builder=build_experiment_config)
