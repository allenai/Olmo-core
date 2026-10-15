"""PixMo Cap-QA as user-QA demo dataset for SFT (mm_olmo ``PixMoCapQaConfig``).

Each row carries several synthetic conversations about one image. mm_olmo keeps every
conversation as ONE ``{"messages": [...]}`` annotation branch (turn 2 attends turn 1)
with ``style="user_qa"`` (registry name ``pixmo_cap_qa_as_user_qa``), and prefixes
counting questions with a ``NO_POINT_PREFIX`` instruction so the model is not trained
to point (``pixmo_datasets.py:594-608``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from PIL import Image

from olmo_core.config import Config

from .detect_counting_question import is_pixmo_point_and_count_question
from .message_sequence import encode_sft_example
from .paths import PIXMO_DATASETS
from .pixmo_ama import NO_POINT_PREFIX
from .sequence_builder import example_rng
from .sft_formatter import SftFormatter

__all__ = ["PixMoCapQaDatasetConfig", "PixMoCapQaDataset"]


def _parse_cap_qa_messages(question: str, answer: str) -> List[str]:
    """Split the ``[USER]``/``[ASSISTANT]``-marked transcript into alternating messages."""
    parts = re.split(r"\s*(\[USER\]|\[ASSISTANT\])\s*", question)
    assert parts[0] == "" and parts[-1] == ""
    parts = parts[1:-1]
    messages: List[str] = []
    for part_ix, part in enumerate(parts):
        if part_ix % 4 in (1, 3):
            messages.append(part)
    messages.append(answer)
    return messages


@dataclass
class PixMoCapQaDatasetConfig(Config):
    style: str = "user_qa"
    prefix_how_many: bool = True
    max_crops: int = 8
    loss_token_weighting: str = "root_subsegments_root_tokens"
    seed: int = 0

    def build(self, tokenizer) -> "PixMoCapQaDataset":
        return PixMoCapQaDataset(self, tokenizer)


class PixMoCapQaDataset:
    def __init__(self, config: PixMoCapQaDatasetConfig, tokenizer):
        from .dataset_compat import load_from_disk_compat

        self.config = config
        self.tokenizer = tokenizer
        self._formatter = SftFormatter(seed=config.seed)
        ds = load_from_disk_compat(f"{PIXMO_DATASETS}/cap-qa")
        self._data = ds["train"] if "train" in ds else ds

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Dict:
        rng = example_rng(self.config.seed, index)
        row = self._data[index]
        message_list = []
        for qs, ans in zip(row["question"], row["answer"]):
            if not ans or not str(ans).strip():
                continue
            messages = _parse_cap_qa_messages(qs, ans)
            message_list.append(dict(messages=messages, style=self.config.style))
        if not message_list:
            raise ValueError(f"No valid QA pairs at index {index}")

        # mm_olmo prefix_how_many: mark counting questions "no pointing" (one rng draw
        # per counting question, before the formatter runs).
        if self.config.prefix_how_many:
            for conv in message_list:
                conv_messages: Any = conv["messages"]
                for user_ix in range(0, len(conv_messages), 2):
                    q, a = conv_messages[user_ix], conv_messages[user_ix + 1]
                    if is_pixmo_point_and_count_question(q, a):
                        prefix = NO_POINT_PREFIX[rng.randint(0, len(NO_POINT_PREFIX))]
                        conv_messages[user_ix] = prefix + conv_messages[user_ix]

        branches = self._formatter.format_branches(
            {"message_list": message_list}, index=index, rng=rng
        )
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        return encode_sft_example(
            self.tokenizer,
            image,
            branches,
            max_crops=self.config.max_crops,
            loss_token_weighting=self.config.loss_token_weighting,
            shuffle_rng=rng,
        )
