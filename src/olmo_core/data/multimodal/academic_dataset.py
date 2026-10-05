"""Academic VQA dataset wrappers: image-only-v9 (stage 2) and the stage-1 academic group."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError

from .academic.registry import (
    ACADEMIC_REGISTRY,
    STAGE2_EVAL_TRAIN_SETS,
    build_academic_data,
    format_academic_example,
)
from .message_sequence import encode_sft_example
from .pixmo_points_v2 import STAGE1_PROMPT_FAMILY
from .sequence_builder import example_rng
from .sft_common import EpochSeededExamples
from .sft_formatter import SftFormatter

__all__ = [
    "AcademicDatasetConfig",
    "AcademicDataset",
    "ACADEMIC_DATASET_NAMES",
    "Stage1AcademicDatasetConfig",
    "Stage1AcademicDataset",
    "STAGE1_DEFAULT_MAX_QUESTIONS",
]

ACADEMIC_DATASET_NAMES = sorted(ACADEMIC_REGISTRY.keys())

#: Questions per image a stage-1 source trains per epoch when its config leaves ``max_questions``
#: unset. 20 questions of a PlotQA image fit a 2,560-token sequence: over 300 sampled images, p99
#: 2,360 tokens and max 2,437 with the Qwen3 tokenizer and 8 crops (at 24, 1.3% overflow). Its
#: images average 131 questions, all of which would be tail-truncated past the sequence length.
STAGE1_DEFAULT_MAX_QUESTIONS: Dict[str, int] = {"plot_qa": 20}


@dataclass
class AcademicDatasetConfig(Config):
    name: str
    max_crops: int = 8
    loss_token_weighting: str = "root_subsegments_root_tokens"
    message_weight: float | None = None
    seed: int = 0

    def build(self, tokenizer) -> "AcademicDataset":
        return AcademicDataset(self, tokenizer)


class AcademicDataset:
    def __init__(self, config: AcademicDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._formatter = SftFormatter(seed=config.seed)
        self._data = build_academic_data(config.name, split="train")
        self._len = len(self._data)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # One mm_olmo-derived rng threads the dataset formatter, prompt templating,
        # and branch shuffle (mm_olmo dataset.py:68) — including pixmo_clocks, whose
        # augmentation draws must vary per example.
        rng = example_rng(self.config.seed, index)
        row = self._data[index]
        formatted = format_academic_example(self.config.name, row, rng)
        turns = self._formatter.format_branches(formatted, index=index, rng=rng)
        example_weight = formatted.get("weight")
        message_weight = (
            example_weight if example_weight is not None else self.config.message_weight
        )
        return encode_sft_example(
            self.tokenizer,
            formatted["image"],
            turns,
            max_crops=self.config.max_crops,
            loss_token_weighting=self.config.loss_token_weighting,
            message_weight=message_weight,
            shuffle_rng=rng,
        )


@dataclass
class Stage1AcademicDatasetConfig(Config):
    """One academic QA source, formatted for stage 1.

    The prompt family is fixed to the stage-1 one (:data:`~.pixmo_points_v2.STAGE1_PROMPT_FAMILY`):
    the user turn is the bare question behind the source's ``"<style>:"`` tag (``dv_qa: What is
    the label of the third bar?``), with no template, instruction or chain-of-thought request. A
    ``*_exp`` source still answers ``"<explanation> Answer: <answer>"``; its tag is what asks for
    the explanation.

    The training set of a benchmark the stage-2 checkpoints are evaluated on
    (:data:`~.academic.registry.STAGE2_EVAL_TRAIN_SETS`) is refused: it stays stage-2 data.
    """

    name: str
    """Registry name (:data:`ACADEMIC_DATASET_NAMES`)."""
    max_crops: int = 8
    max_questions: Optional[int] = None
    """Most questions trained per image per epoch. ``None`` takes the source's default from
    :data:`STAGE1_DEFAULT_MAX_QUESTIONS` (20 for PlotQA, whose images carry 131 questions on
    average and do not fit a stage-1 sequence) and otherwise trains all of them. Which ones are
    drawn moves with the epoch, so over several epochs the rest are reached too."""
    loss_token_weighting: str = "none"
    """Every response token weighted equally, as for the other stage-1 sources."""
    seed: int = 0

    def build(self, tokenizer) -> "Stage1AcademicDataset":
        return Stage1AcademicDataset(self, tokenizer)


class Stage1AcademicDataset(EpochSeededExamples):
    """See :class:`Stage1AcademicDatasetConfig`.

    :raises OLMoConfigurationError: If the source is a stage-2 benchmark's training set, is not
        in the registry, or ``max_questions`` is below 1. Checked before any data is read.
    """

    def __init__(self, config: Stage1AcademicDatasetConfig, tokenizer):
        if config.name in STAGE2_EVAL_TRAIN_SETS:
            raise OLMoConfigurationError(
                f"{config.name!r} is the training set of a stage-2 eval benchmark "
                f"({STAGE2_EVAL_TRAIN_SETS[config.name]}), so it is not trained in stage 1"
            )
        if config.name not in ACADEMIC_REGISTRY:
            raise OLMoConfigurationError(
                f"Unknown academic dataset {config.name!r}; "
                f"expected one of {ACADEMIC_DATASET_NAMES}"
            )
        if config.max_questions is not None and config.max_questions < 1:
            raise OLMoConfigurationError(f"max_questions={config.max_questions} must be >= 1")
        self.config = config
        self.tokenizer = tokenizer
        self._max_questions = (
            config.max_questions
            if config.max_questions is not None
            else STAGE1_DEFAULT_MAX_QUESTIONS.get(config.name)
        )
        self._formatter = SftFormatter(
            seed=config.seed,
            prompt_templates=STAGE1_PROMPT_FAMILY["prompt_templates"],
            system_prompt=STAGE1_PROMPT_FAMILY["system_prompt"],
        )
        self._data = build_academic_data(config.name, split="train")

    def __len__(self) -> int:
        return len(self._data)

    def format_row(
        self, index: int, rng: np.random.RandomState
    ) -> Tuple[Any, List[List[Tuple[str, str]]], Optional[float]]:
        """The image, the branches trained this epoch, and the example weight of one row."""
        formatted = format_academic_example(self.config.name, self._data[index], rng)
        branches = self._formatter.format_branches(formatted, index=index, rng=rng)
        cap = self._max_questions
        if cap is not None and len(branches) > cap:
            keep = np.sort(rng.choice(len(branches), size=cap, replace=False))
            branches = [branches[i] for i in keep]
        return formatted["image"], branches, formatted.get("weight")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rng = self.epoch_rng(index)
        image, branches, weight = self.format_row(index, rng)
        return encode_sft_example(
            self.tokenizer,
            image,
            branches,
            max_crops=self.config.max_crops,
            loss_token_weighting=self.config.loss_token_weighting,
            message_weight=weight,
            shuffle_rng=rng,
        )
