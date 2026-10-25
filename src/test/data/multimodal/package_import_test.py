import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_import_check(code: str):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(Path(__file__).parents[3]) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )


@pytest.mark.parametrize(
    "module",
    [
        "olmo_core.data.multimodal",
        "olmo_core.data.multimodal.alignment",
        "olmo_core.internal.vision_alignment",
        "olmo_core.internal.vision_alignment_pipeline",
    ],
)
def test_clean_import_does_not_load_legacy_registries(module):
    _run_import_check(
        f"""
import importlib
import sys

importlib.import_module({module!r})
legacy = (
    'native_text_replay',
    'vision_alignment_sources',
    'vision_alignment_perception_sources',
    'vision_alignment_joint_sources',
    'vision_alignment_perception_provenance',
    'vision_alignment_joint_provenance',
    'mixtures.vision_alignment',
)
loaded = [name for name in legacy if 'olmo_core.data.multimodal.' + name in sys.modules]
assert not loaded, loaded
"""
    )


def test_native_configs_are_public_and_keep_serialization_names():
    from olmo_core.config import Config
    from olmo_core.data.multimodal import (
        MultimodalMixtureConfig,
        MultimodalSourceConfig,
        PretrainingReplayConfig,
    )
    from olmo_core.data.tokenizer import TokenizerConfig

    configs = [
        MultimodalMixtureConfig(tokenizer=TokenizerConfig.dolma2()),
        MultimodalSourceConfig(dataset=PretrainingReplayConfig(checkpoint="/unused")),
        PretrainingReplayConfig(checkpoint="/unused"),
    ]
    for config in configs:
        saved = config.as_config_dict()
        assert (
            saved[Config.CLASS_NAME_FIELD] == f"{type(config).__module__}.{type(config).__name__}"
        )
        assert Config.from_dict(saved) == config


def test_star_import_preserves_target_and_native_dataset_exports():
    _run_import_check(
        """
import olmo_core.data.multimodal as package
from olmo_core.data.multimodal.tulu import Tulu4DatasetConfig
from olmo_core.data.multimodal.pixmo_points import PixMoPointsDatasetConfig
from olmo_core.data.multimodal.pretraining_replay import PretrainingReplayConfig

namespace = {}
exec('from olmo_core.data.multimodal import *', namespace)
assert all(name in namespace for name in package.__all__)
for cls in (Tulu4DatasetConfig, PixMoPointsDatasetConfig, PretrainingReplayConfig):
    assert namespace[cls.__name__] is cls
"""
    )
