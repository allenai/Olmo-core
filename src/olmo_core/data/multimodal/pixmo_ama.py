"""PixMo Ask Model Anything demo dataset for SFT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from PIL import Image

from olmo_core.config import Config

from .detect_counting_question import is_pixmo_point_and_count_question
from .message_sequence import encode_sft_example
from .paths import PIXMO_DATASETS
from .sequence_builder import example_rng

__all__ = ["PixMoAmaDatasetConfig", "PixMoAmaDataset"]

NO_POINT_PREFIX = (
    "No pointing: ",
    "No pointing: ",
    "no pointing:\n",
    "No pointing:\n",
    "Not pointing:\n",
    "No Points: ",
    "No Points: ",
    "NO POINTING\n",
    "No pontiing\n",
    "No Points:\n ",
    "No pointing\n",
    "Do not point. ",
    "Refrain from pointing. ",
    "Avoid generating points . ",
    "For this question, do not use points. ",
    "Refrain from using points:\n",
    "Don't include points in your response. ",
    "Don't point. ",
    "Don't use points. ",
    "Please don't use points.\n\n",
    "Please don't use points.\n\n",
    "Respond without using points. ",
    "Respond without pointing:\n",
    "Do not generate ponits: ",
    "Do not point. ",
    "Do not point\n",
    "no pointing\n\n",
    "Answer without points: ",
    "Answer this question without pointing: ",
    "Answer without poiints. ",
    "answer without points: ",
    "answer with text only, do not points\n",
)


@dataclass
class PixMoAmaDatasetConfig(Config):
    max_crops: int = 8
    loss_token_weighting: str = "root_subsegments_root_tokens"
    seed: int = 0

    def build(self, tokenizer) -> "PixMoAmaDataset":
        return PixMoAmaDataset(self, tokenizer)


class PixMoAmaDataset:
    def __init__(self, config: PixMoAmaDatasetConfig, tokenizer):
        from .dataset_compat import load_from_disk_compat

        self.config = config
        self.tokenizer = tokenizer
        ds = load_from_disk_compat(f"{PIXMO_DATASETS}/ask-model-anything")
        self._data = ds["train"] if "train" in ds else ds

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Dict:
        rng = example_rng(self.config.seed, index)
        row = self._data[index]
        turns: List[Tuple[str, str]] = []
        for q, a in zip(row["question"], row["answer"]):
            q = q.strip()
            if is_pixmo_point_and_count_question(q, a):
                q = NO_POINT_PREFIX[rng.randint(len(NO_POINT_PREFIX))] + q
            turns.append((q, a))
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        return encode_sft_example(
            self.tokenizer,
            image,
            turns,
            max_crops=self.config.max_crops,
            loss_token_weighting=self.config.loss_token_weighting,
            shuffle_rng=rng,
        )
