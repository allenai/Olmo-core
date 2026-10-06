"""Run mixed midtraining from a joint-alignment checkpoint."""

from olmo_core.internal.experiment import main
from olmo_core.internal.vision_midtraining import build_config

if __name__ == "__main__":
    main(config_builder=build_config)
