"""Load academic datasets via mm_olmo configs (interim bridge for parity work).

This module imports mm_olmo only to reuse ``format_example`` and raw data loading
while olmo-core owns tokenization. Replace with native formatters incrementally.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Any, Dict

import numpy as np

# HARDCODED personal checkout (donovanc's mm_olmo clone on weka): this module is a
# debug/parity bridge only — nothing in the training data path imports it. Override with
# the MM_OLMO_ROOT env var to point at your own mm_olmo checkout.
_MM_OLMO_ROOT = os.environ.get(
    "MM_OLMO_ROOT", "/weka/oe-training-default/donovanc/molmofication/mm_olmo"
)


@lru_cache(maxsize=1)
def _ensure_mm_olmo() -> None:
    if _MM_OLMO_ROOT not in sys.path:
        sys.path.insert(0, _MM_OLMO_ROOT)
    if "MOLMO_DATA_DIR" not in os.environ:
        os.environ["MOLMO_DATA_DIR"] = "/weka/oe-training-default/mm-olmo"


def get_mm_olmo_config(dataset_name: str):
    _ensure_mm_olmo()
    from olmo.data.get_dataset import get_dataset_config_by_name

    return get_dataset_config_by_name(dataset_name)


def get_mm_olmo_formatted_example(
    dataset_name: str, index: int, seed: int, split: str = "train"
) -> Dict[str, Any]:
    """Return the intermediate dict from mm_olmo ``format_example``."""
    cfg = get_mm_olmo_config(dataset_name)
    data = cfg.build(split=split)
    rng = np.random.RandomState(seed)
    if hasattr(data, "__getitem__"):
        row = data[index]
    else:
        row = data.get(index, rng)
    return cfg.format_example(row, rng, split)


def get_mm_olmo_tokenized_example(
    dataset_name: str,
    index: int,
    seed: int,
    *,
    seq_len: int = 16384,
    split: str = "train",
) -> Dict[str, Any]:
    """Full mm_olmo SFT pipeline (formatter + Molmo2 preprocessor)."""
    _ensure_mm_olmo()
    from olmo.data.data_formatter import DataFormatter
    from olmo.data.dataset import FormattedData
    from olmo.models.molmo2.example_preprocessor import Molmo2ExamplePreprocessor
    from olmo.models.molmo2.grounding_formatter import GroundingPreprocessor
    from olmo.preprocessing.multicrop_preprocessor import MultiCropConfig
    from olmo.preprocessing.text_preprocessor import MessageWeight
    from transformers import AutoTokenizer

    cfg = get_mm_olmo_config(dataset_name)
    data = cfg.build(split=split)
    tokenizer = AutoTokenizer.from_pretrained("allenai/Molmo2-4B", trust_remote_code=True)
    formatter = DataFormatter(
        prompt_templates="uber_model_v2",
        system_prompt="demo_or_style_v2",
        select_answer="best",
    )
    preprocessor = Molmo2ExamplePreprocessor(
        formatter,
        tokenizer,
        grounding_preprocessor=GroundingPreprocessor(),
        image_preprocessor=MultiCropConfig().build_image_preprocessor(tokenizer, None, None)[0],
        video_preprocessor=None,
        max_sequence_len=seq_len,
        message_format="qwen3",
        default_message_weight=MessageWeight.from_string("root_subsegments_root_tokens"),
        is_training=True,
    )
    formatted = FormattedData(data, formatter, preprocessor)
    return formatted.get(index, seed)
