"""Academic vision evaluation on saved validation selections.

Uses document prompts, fixed image controls, and native expert-parallel inference.
Selection manifests define the benchmark; checkpoint metadata and runtime settings
identify an execution. Training-source inventories are not replayed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import string
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from olmo_core.data.multimodal.academic import registry as academic_registry
from olmo_core.data.multimodal.document_layout import document_prompt_ids, response_ids
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.vision.molmo2_image_processor import (
    _select_tiling,
    preprocess_image_molmo2,
)
from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds, build_image_token_ids
from olmo_core.utils import gc_cuda

log = logging.getLogger(__name__)

DEFAULT_TASKS = ("vqav2", "textvqa", "docvqa", "chartqa", "ai2d", "a_okvqa_mc")
CONTROLS = ("correct", "shuffled", "blank")
EMPTY_OPTION_DISPLAY = "<empty>"
DEFAULT_MAX_CROPS = 8
RESULT_FORMAT = "olmo_core_vision_academic_v1"
TOKENIZER_ID = "allenai/dolma2-tokenizer"
TOKENIZER_REVISION = "5292e5d6c0f40b67cc765fe41bec991cf4345b5c"
TOKENIZER_FINGERPRINT = "8fec2af8c372f4c72a1a665ad8e70517625f94f041dbfcb7db4932071380f9a7"
TOKEN_IDS = Molmo2TokenIds(
    im_start_id=100278,
    im_end_id=100279,
    im_patch_id=100280,
    im_col_id=100281,
    low_res_im_start_id=100282,
    image_placeholder_id=100283,
    im_end_turn_id=100265,
)
OVERLAP_SCOPE = (
    "Nonoverlap aggregates retain the selection manifest's training-image inventory "
    "annotations. They do not establish nonoverlap with this checkpoint's training data."
)


@dataclass(frozen=True)
class AcademicExample:
    """Canonical runtime representation of one academic validation question."""

    task: str
    example_id: str
    source_position: str
    visual: Any
    image_reference: Any
    question: str
    answers: tuple[str, ...] = ()
    options: tuple[str, ...] = ()
    answer_index: int | None = None
    stratum: str | None = None

    def annotation(self) -> dict[str, Any]:
        """Return the image-independent canonical annotation projection."""
        return {
            "task": self.task,
            "example_id": self.example_id,
            "source_position": self.source_position,
            "question": self.question,
            "answers": list(self.answers),
            "options": list(self.options),
            "answer_index": self.answer_index,
            "stratum": self.stratum,
        }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _image_reference_sha256(reference: Any) -> str:
    if isinstance(reference, Mapping):
        embedded = reference.get("bytes")
        if isinstance(embedded, (bytes, bytearray, memoryview)) and embedded:
            return hashlib.sha256(bytes(embedded)).hexdigest()
        reference = reference.get("path")
    if isinstance(reference, (bytes, bytearray, memoryview)):
        if not reference:
            raise ValueError("Image byte reference is empty")
        return hashlib.sha256(bytes(reference)).hexdigest()
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"Unsupported exact image reference {type(reference)!r}")
    path = Path(reference).expanduser().resolve()
    return _file_sha256(path)


def _image_dimensions(reference: Any) -> tuple[int, int]:
    if isinstance(reference, Image.Image):
        width, height = reference.size
    elif isinstance(reference, np.ndarray):
        if reference.ndim < 2:
            raise ValueError("Image array has fewer than two dimensions")
        height, width = reference.shape[:2]
    else:
        if isinstance(reference, Mapping):
            embedded = reference.get("bytes")
            if isinstance(embedded, (bytes, bytearray, memoryview)) and embedded:
                import io

                with Image.open(io.BytesIO(bytes(embedded))) as image:
                    width, height = image.size
            else:
                reference = reference.get("path")
                if not isinstance(reference, (str, os.PathLike)):
                    raise ValueError("Image mapping has neither encoded bytes nor a path")
                with Image.open(reference) as image:
                    width, height = image.size
        elif isinstance(reference, (str, os.PathLike)):
            with Image.open(reference) as image:
                width, height = image.size
        else:
            raise TypeError(f"Unsupported image reference {type(reference)!r}")
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    return int(height), int(width)


def _molmo2_grid_signature(
    reference: Any, *, max_crops: int = DEFAULT_MAX_CROPS
) -> tuple[int, ...]:
    """Derive Molmo2's pooled grid from dimensions without materializing crop tensors."""
    height, width = _image_dimensions(reference)
    patch_size = 14
    crop_patch_size = 378 // patch_size
    overlap_patches = 8
    crop_window_patches = crop_patch_size - overlap_patches
    crop_window_size = crop_window_patches * patch_size
    tiling = _select_tiling(
        height - overlap_patches * patch_size,
        width - overlap_patches * patch_size,
        crop_window_size,
        max_crops,
    )
    high_resolution_height = int(tiling[0]) * crop_window_patches + overlap_patches
    high_resolution_width = int(tiling[1]) * crop_window_patches + overlap_patches
    pooled_low_resolution = (crop_patch_size + 1) // 2
    return (
        pooled_low_resolution,
        pooled_low_resolution,
        (high_resolution_height + 1) // 2,
        (high_resolution_width + 1) // 2,
    )


def _validate_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-blank string")
    return value.strip()


def _option_text(value: Any, *, name: str, allow_blank: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    output = value.strip()
    if not output and not allow_blank:
        raise ValueError(f"{name} must be non-blank")
    return output


def _answers(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Sequence[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        raise TypeError(f"{name} must be a string or string sequence")
    output = tuple(_validate_text(item, name=f"{name} answer") for item in values)
    if not output:
        raise ValueError(f"{name} contains no answers")
    return output


def _validate_examples(task: str, examples: Sequence[AcademicExample]) -> None:
    if not examples:
        raise ValueError(f"Academic task {task!r} is empty")
    ids = [example.example_id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Academic task {task!r} contains duplicate example IDs")
    for example in examples:
        if example.task != task:
            raise ValueError(f"Academic task projection drifted for {example.example_id!r}")
        _validate_text(example.question, name=f"{task}/{example.example_id} question")
        if task in ("ai2d", "a_okvqa_mc"):
            if not 2 <= len(example.options) <= len(string.ascii_uppercase):
                raise ValueError(f"{task}/{example.example_id} has an invalid option count")
            if example.answer_index is None or not 0 <= example.answer_index < len(example.options):
                raise ValueError(f"{task}/{example.example_id} has an invalid answer index")
        elif not example.answers:
            raise ValueError(f"{task}/{example.example_id} contains no gold answers")


def _ai2d_examples_from_raw(raw: Any) -> list[AcademicExample]:
    """Project AI2D rows without eagerly decoding and retaining validation images."""
    output = []
    for index in range(len(raw)):
        row = raw[index]
        output.append(
            AcademicExample(
                task="ai2d",
                example_id=_validate_text(row["question_id"], name="AI2D question_id"),
                source_position=str(index),
                visual=row["image"],
                image_reference=row["image"],
                question=_validate_text(row["question"], name="AI2D question"),
                options=tuple(
                    _option_text(value, name="AI2D option", allow_blank=True)
                    for value in row["answer_texts"]
                ),
                answer_index=int(row["correct_answer"]),
                stratum=("transparent" if bool(row.get("has_transparent_box")) else "standard"),
            )
        )
    return output


def _ai2d_examples() -> list[AcademicExample]:
    from datasets import Image as HFImage

    from olmo_core.data.multimodal.dataset_compat import load_from_disk_compat

    root = Path(academic_registry.ACADEMIC_DATASETS) / "ai2d"
    raw = load_from_disk_compat(root)["validation"].cast_column("image", HFImage(decode=False))
    return _ai2d_examples_from_raw(raw)


def _load_task_examples(task: str) -> list[AcademicExample]:
    output: list[AcademicExample] = []
    if task == "vqav2":
        rows = academic_registry.ACADEMIC_REGISTRY["coco_2014_vqa_multi"].loader("validation")
        for image_index, row in enumerate(rows):
            for question_index, message in enumerate(row["messages"]):
                output.append(
                    AcademicExample(
                        task=task,
                        example_id=str(message["question_id"]),
                        source_position=f"{image_index}:{question_index}",
                        visual=row["image"],
                        image_reference=row["image"],
                        question=_validate_text(message["question"], name="VQAv2 question"),
                        answers=_answers(message["answers"], name="VQAv2"),
                    )
                )
    elif task in ("textvqa", "docvqa", "chartqa", "a_okvqa_mc"):
        registry_name = {
            "textvqa": "text_vqa",
            "docvqa": "doc_qa",
            "chartqa": "chart_qa_weighted",
            "a_okvqa_mc": "a_okvqa_mc",
        }[task]
        rows = academic_registry.ACADEMIC_REGISTRY[registry_name].loader("validation")
        for index, row in enumerate(rows):
            metadata = row.get("metadata", {})
            raw_id = metadata.get("example_id")
            if task == "chartqa":
                kind = "human" if bool(metadata.get("is_human")) else "augmented"
                example_id = f"{kind}:{index:06d}"
            else:
                example_id = str(raw_id)
                if not example_id or example_id == "None":
                    raise ValueError(f"{task} row {index} lacks an example ID")
            common = {
                "task": task,
                "example_id": example_id,
                "source_position": str(index),
                "visual": row["image"],
                "image_reference": row["image"],
                "question": _validate_text(row["question"], name=f"{task} question"),
            }
            if task == "a_okvqa_mc":
                output.append(
                    AcademicExample(
                        **common,
                        options=tuple(
                            _option_text(value, name="A-OKVQA option", allow_blank=True)
                            for value in row["options"]
                        ),
                        answer_index=int(row["answer_idx"]),
                    )
                )
            else:
                output.append(
                    AcademicExample(
                        **common,
                        answers=_answers(row["answers"], name=task),
                        stratum=(
                            ("human" if bool(metadata.get("is_human")) else "augmented")
                            if task == "chartqa"
                            else None
                        ),
                    )
                )
    elif task == "ai2d":
        output = _ai2d_examples()
    else:
        raise ValueError(f"Unsupported academic task {task!r}")
    _validate_examples(task, output)
    return output


_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}


_NUMBER_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


_ARTICLES = frozenset(("a", "an", "the"))


_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")


_COMMA_STRIP = re.compile(r"(?<=\d)(,)+(?=\d)")


_PUNCTUATION = (
    ";",
    "/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
)


def _normalize_vqa_answer(value: str) -> str:
    text = value.replace("\n", " ").replace("\t", " ").strip()
    processed = text
    for punctuation in _PUNCTUATION:
        if (
            punctuation + " " in text
            or " " + punctuation in text
            or _COMMA_STRIP.search(text) is not None
        ):
            processed = processed.replace(punctuation, "")
        else:
            processed = processed.replace(punctuation, " ")
    processed = _PERIOD_STRIP.sub("", processed)
    tokens = []
    for token in processed.lower().split():
        token = _NUMBER_MAP.get(token, token)
        if token not in _ARTICLES:
            tokens.append(_CONTRACTIONS.get(token, token))
    return " ".join(tokens)


def _normalize_textvqa_answer(value: str) -> str:
    """Apply TextVQA's word-tokenize pass before VQA answer normalization."""
    tokenized = value.lower().replace(",", "").replace("?", "").replace("'s", " 's").strip()
    return _normalize_vqa_answer(tokenized)


def _vqa_consensus_accuracy(
    prediction: str,
    answers: Sequence[str],
    *,
    normalizer: Any,
) -> float:
    predicted = normalizer(prediction)
    normalized = [normalizer(answer) for answer in answers]
    if not normalized:
        raise ValueError("VQA accuracy requires at least one reference answer")
    scores = []
    for index in range(len(normalized)):
        matches = sum(
            answer == predicted
            for other_index, answer in enumerate(normalized)
            if other_index != index
        )
        scores.append(min(1.0, float(matches) / 3.0))
    return float(sum(scores) / len(scores))


def _vqa_accuracy(prediction: str, answers: Sequence[str]) -> float:
    return _vqa_consensus_accuracy(
        prediction,
        answers,
        normalizer=_normalize_vqa_answer,
    )


def _textvqa_accuracy(prediction: str, answers: Sequence[str]) -> float:
    return _vqa_consensus_accuracy(
        prediction,
        answers,
        normalizer=_normalize_textvqa_answer,
    )


def _levenshtein_distance(left: str, right: str) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row, right_character in enumerate(right, start=1):
        current = [row]
        for column, left_character in enumerate(left, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _anls(prediction: str, answers: Sequence[str], *, threshold: float = 0.5) -> float:
    predicted = prediction.lower().strip()
    if not predicted:
        return 0.0
    similarities = []
    for answer in answers:
        target = answer.lower().strip()
        denominator = max(len(predicted), len(target))
        similarity = (
            1.0
            if denominator == 0
            else 1.0 - _levenshtein_distance(predicted, target) / denominator
        )
        similarities.append(similarity)
    score = max(similarities)
    return float(score if score > threshold else 0.0)


def _chartqa_relaxed_accuracy(
    prediction: str,
    target: str,
    *,
    max_relative_change: float = 0.05,
) -> float:
    def to_float(text: str) -> float | None:
        text = text.strip()
        try:
            return float(text[:-1]) / 100.0 if text.endswith("%") else float(text)
        except ValueError:
            return None

    predicted_float = to_float(prediction)
    target_float = to_float(target)
    if predicted_float is not None and target_float is not None and target_float != 0.0:
        return float(abs(predicted_float - target_float) / abs(target_float) <= max_relative_change)
    return float(prediction.strip().lower() == target.strip().lower())


def _metric_name(task: str) -> str:
    if task in ("vqav2", "textvqa"):
        return "vqa_accuracy"
    if task == "docvqa":
        return "anls"
    if task == "chartqa":
        return "relaxed_accuracy"
    if task in ("ai2d", "a_okvqa_mc"):
        return "multiple_choice_accuracy"
    raise ValueError(f"Unsupported academic task {task!r}")


def _score_prediction(
    example: AcademicExample,
    *,
    prediction: str,
    predicted_index: int | None,
) -> float:
    if example.task == "vqav2":
        return _vqa_accuracy(prediction, example.answers)
    if example.task == "textvqa":
        return _textvqa_accuracy(prediction, example.answers)
    if example.task == "docvqa":
        return _anls(prediction, example.answers)
    if example.task == "chartqa":
        return _chartqa_relaxed_accuracy(prediction, example.answers[0])
    if example.task == "ai2d":
        if predicted_index is None or example.answer_index is None:
            raise ValueError("Multiple-choice scoring requires prediction and target indices")
        return float(predicted_index == example.answer_index)
    if example.task == "a_okvqa_mc":
        if predicted_index is None or example.answer_index is None:
            raise ValueError("Multiple-choice scoring requires prediction and target indices")
        return float(example.options[predicted_index] == example.options[example.answer_index])
    raise ValueError(f"Unsupported academic task {example.task!r}")


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _aggregate_task_outputs(task: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_name = _metric_name(task)
    controls: dict[str, Any] = {}
    for control in CONTROLS:
        all_values = [float(row["controls"][control]["score"]) for row in rows]
        exact_byte_nonoverlap_values = [
            float(row["controls"][control]["score"])
            for row in rows
            if not (
                row["shuffled_alignment_train_image_overlap"]
                if control == "shuffled"
                else row["alignment_train_image_overlap"]
            )
        ]
        metrics: dict[str, Any] = {
            metric_name: _mean(all_values),
            f"exact_byte_nonoverlap_{metric_name}": _mean(exact_byte_nonoverlap_values),
            "examples": len(all_values),
            "exact_byte_nonoverlap_examples": len(exact_byte_nonoverlap_values),
        }
        strata = (
            ("human", "augmented")
            if task == "chartqa"
            else (("standard", "transparent") if task == "ai2d" else ())
        )
        for stratum in strata:
            stratum_values = [
                float(row["controls"][control]["score"])
                for row in rows
                if row["stratum"] == stratum
            ]
            exact_byte_nonoverlap_stratum_values = [
                float(row["controls"][control]["score"])
                for row in rows
                if row["stratum"] == stratum
                and not (
                    row["shuffled_alignment_train_image_overlap"]
                    if control == "shuffled"
                    else row["alignment_train_image_overlap"]
                )
            ]
            metrics[f"{metric_name}_{stratum}"] = _mean(stratum_values)
            metrics[f"exact_byte_nonoverlap_{metric_name}_{stratum}"] = _mean(
                exact_byte_nonoverlap_stratum_values
            )
        controls[control] = metrics
    correct = controls["correct"]
    exact_byte_nonoverlap_pair_values = {
        control: [
            float(row["controls"][control]["score"])
            for row in rows
            if not row["alignment_train_image_overlap"]
            and not row["shuffled_alignment_train_image_overlap"]
        ]
        for control in ("correct", "shuffled")
    }
    exact_byte_nonoverlap_blank_values = {
        control: [
            float(row["controls"][control]["score"])
            for row in rows
            if not row["alignment_train_image_overlap"]
        ]
        for control in ("correct", "blank")
    }
    exact_pair_correct = _mean(exact_byte_nonoverlap_pair_values["correct"])
    exact_pair_shuffled = _mean(exact_byte_nonoverlap_pair_values["shuffled"])
    exact_blank_correct = _mean(exact_byte_nonoverlap_blank_values["correct"])
    exact_blank = _mean(exact_byte_nonoverlap_blank_values["blank"])
    return {
        "metric": metric_name,
        "controls": controls,
        "image_control_deltas": {
            f"{metric_name}_correct_minus_shuffled": (
                correct[metric_name] - controls["shuffled"][metric_name]
            ),
            f"{metric_name}_correct_minus_blank": (
                correct[metric_name] - controls["blank"][metric_name]
            ),
            f"exact_byte_nonoverlap_{metric_name}_correct_minus_shuffled": (
                exact_pair_correct - exact_pair_shuffled
                if exact_pair_correct is not None and exact_pair_shuffled is not None
                else None
            ),
            f"exact_byte_nonoverlap_{metric_name}_correct_minus_blank": (
                exact_blank_correct - exact_blank
                if exact_blank_correct is not None and exact_blank is not None
                else None
            ),
            "exact_byte_nonoverlap_correct_minus_shuffled_examples": len(
                exact_byte_nonoverlap_pair_values["correct"]
            ),
            "exact_byte_nonoverlap_correct_minus_blank_examples": len(
                exact_byte_nonoverlap_blank_values["correct"]
            ),
        },
    }


def _generation_stop_counts(task: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if task in ("ai2d", "a_okvqa_mc"):
        return {}
    return {
        control: {
            "eos": sum(row["controls"][control]["stop_reason"] == "eos" for row in rows),
            "max_tokens": sum(
                row["controls"][control]["stop_reason"] == "max_tokens" for row in rows
            ),
        }
        for control in CONTROLS
    }


def _as_rgb_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB").copy()
    if isinstance(value, np.ndarray):
        return Image.fromarray(value.astype("uint8")).convert("RGB")
    if isinstance(value, Mapping):
        embedded = value.get("bytes")
        if isinstance(embedded, (bytes, bytearray, memoryview)) and embedded:
            import io

            with Image.open(io.BytesIO(bytes(embedded))) as image:
                return image.convert("RGB").copy()
        value = value.get("path")
    if isinstance(value, (str, os.PathLike)):
        with Image.open(value) as image:
            return image.convert("RGB").copy()
    raise ValueError(f"Unsupported runtime image value {type(value)!r}")


def _build_mc_prompt(question: str, options: Sequence[str]) -> str:
    if not 2 <= len(options) <= len(string.ascii_uppercase):
        raise ValueError("Multiple-choice prompt requires between 2 and 26 options")
    option_text = "\n".join(
        f"{letter}. {option if option else EMPTY_OPTION_DISPLAY}"
        for letter, option in zip(string.ascii_uppercase, options)
    )
    return (
        f"Question: {question}\nOptions:\n{option_text}\n"
        "Answer with only the option letter.\nAnswer:"
    )


def _free_answer_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


class _NativeAcademicInference:
    """Greedy and option-letter inference with identical inputs on every EP rank."""

    def __init__(
        self,
        train_module: Any,
        tokenizer: Any,
        token_ids: Molmo2TokenIds,
        *,
        max_sequence_length: int,
        max_crops: int,
        max_new_tokens: int,
        sequence_bucket_size: int,
    ) -> None:
        self.train_module = train_module
        self.model = train_module.model_parts[0]
        self.tokenizer = tokenizer
        self.token_ids = token_ids
        self.max_sequence_length = max_sequence_length
        self.max_crops = max_crops
        self.max_new_tokens = max_new_tokens
        self.sequence_bucket_size = sequence_bucket_size
        self.text_vocab_size = min(token_ids.image_token_ids)

    @property
    def device(self) -> torch.device:
        """Device hosting the evaluation train module."""
        return self.train_module.device

    def _buffer_length(self, required: int) -> int:
        rounded = (
            (required + self.sequence_bucket_size - 1) // self.sequence_bucket_size
        ) * self.sequence_bucket_size
        return min(rounded, self.max_sequence_length)

    def _prepare_visual(
        self, image: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], tuple[int, ...]]:
        images, pooling, grid = preprocess_image_molmo2(
            image,
            dtype=torch.float32,
            device=torch.device("cpu"),
            max_crops=self.max_crops,
            is_training=False,
        )
        resized_h, resized_w, height, width = (int(grid[index]) for index in range(4))
        image_ids = build_image_token_ids(
            resized_h,
            resized_w,
            height,
            width,
            token_ids=self.token_ids,
        )
        return images, pooling, image_ids, (resized_h, resized_w, height, width)

    def _inputs(
        self,
        prompt_ids: Sequence[int],
        *,
        required: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if required > self.max_sequence_length:
            raise ValueError(
                f"Academic request needs {required} tokens, exceeding the frozen "
                f"maximum {self.max_sequence_length}"
            )
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Academic evaluator tokenizer has no pad or EOS token")
        buffer_length = self._buffer_length(required)
        input_ids = torch.full(
            (1, buffer_length),
            int(pad_token_id),
            dtype=torch.long,
            device=self.device,
        )
        input_ids[0, : len(prompt_ids)] = torch.tensor(prompt_ids, device=self.device)
        token_type_ids = torch.zeros_like(input_ids)
        prompt_tensor = input_ids[0, : len(prompt_ids)]
        for image_token_id in self.token_ids.image_token_ids:
            token_type_ids[0, : len(prompt_ids)] |= prompt_tensor.eq(image_token_id)
        position_ids = torch.arange(buffer_length, device=self.device).unsqueeze(0)
        return input_ids, token_type_ids, position_ids

    def _consensus_token(self, token_id: int) -> int:
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return token_id
        extrema = torch.tensor([token_id, token_id], dtype=torch.int64, device=self.device)
        dist.all_reduce(extrema[:1], op=dist.ReduceOp.MIN)
        dist.all_reduce(extrema[1:], op=dist.ReduceOp.MAX)
        if int(extrema[0]) != int(extrema[1]):
            raise RuntimeError(
                "Native EP ranks predicted different tokens: "
                f"min={int(extrema[0])}, max={int(extrema[1])}"
            )
        return int(extrema[0])

    @torch.inference_mode(False)
    def predict(
        self,
        example: AcademicExample,
        visual: Image.Image,
    ) -> dict[str, Any]:
        """Score option letters or generate a bounded free answer for one image."""
        images, pooling, image_ids, grid_signature = self._prepare_visual(visual)
        is_multiple_choice = bool(example.options)
        prompt = (
            _build_mc_prompt(example.question, example.options)
            if is_multiple_choice
            else _free_answer_prompt(example.question)
        )
        prompt_ids = document_prompt_ids(self.tokenizer, prompt, image_ids=image_ids)
        # Reuse the eager, grad-enabled attention regime from native evaluation. The
        # decorator also keeps inputs ordinary tensors under an enclosing inference mode.
        # Detach projected features so autoregressive forwards do not retain the vision graph.
        with self.train_module._eval_batch_context():
            encoded_features = self.model.encode_images(images, pooling).detach()
        if is_multiple_choice:
            candidate_encodings = [
                response_ids(self.tokenizer, letter)
                for letter in string.ascii_uppercase[: len(example.options)]
            ]
            if any(len(ids) != 1 for ids in candidate_encodings):
                raise ValueError("Every academic option letter must encode to one response token")
            input_ids, token_type_ids, position_ids = self._inputs(
                prompt_ids,
                required=len(prompt_ids),
            )
            logits_position = torch.tensor(
                [[len(prompt_ids) - 1]], dtype=torch.long, device=self.device
            )
            with self.train_module._eval_batch_context():
                logits = self.train_module.model_forward_no_pipeline(
                    input_ids,
                    encoded_image_features=encoded_features,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    logits_to_keep=logits_position,
                )
            if not isinstance(logits, torch.Tensor):
                raise TypeError(f"Expected academic logits tensor, got {type(logits).__name__}")
            logits = logits.detach()
            candidate_ids = torch.tensor(
                [ids[0] for ids in candidate_encodings],
                dtype=torch.long,
                device=self.device,
            )
            candidate_logits = logits[0, 0, candidate_ids].float()
            predicted_index = self._consensus_token(int(candidate_logits.argmax().item()))
            candidate_log_probabilities = candidate_logits.log_softmax(dim=-1).cpu().tolist()
            return {
                "prediction": string.ascii_uppercase[predicted_index],
                "predicted_index": predicted_index,
                "candidate_log_probabilities": {
                    letter: float(value)
                    for letter, value in zip(
                        string.ascii_uppercase[: len(example.options)],
                        candidate_log_probabilities,
                    )
                },
                "image_grid_signature": list(grid_signature),
                "image_token_count": len(image_ids),
                "image_token_ids_sha256": _canonical_sha256(image_ids),
                "input_tokens": len(prompt_ids),
                "output_tokens": 1,
            }

        required = len(prompt_ids) + self.max_new_tokens
        input_ids, token_type_ids, position_ids = self._inputs(prompt_ids, required=required)
        generated: list[int] = []
        stop_reason = "max_tokens"
        with self.train_module._eval_batch_context():
            for _ in range(self.max_new_tokens):
                current_length = len(prompt_ids) + len(generated)
                logits_position = torch.tensor(
                    [[current_length - 1]], dtype=torch.long, device=self.device
                )
                logits = self.train_module.model_forward_no_pipeline(
                    input_ids,
                    encoded_image_features=encoded_features,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    logits_to_keep=logits_position,
                )
                if not isinstance(logits, torch.Tensor):
                    raise TypeError(f"Expected academic logits tensor, got {type(logits).__name__}")
                logits = logits.detach()
                next_token = int(logits[0, 0, : self.text_vocab_size].argmax().item())
                next_token = self._consensus_token(next_token)
                generated.append(next_token)
                if next_token == self.tokenizer.eos_token_id:
                    stop_reason = "eos"
                    break
                input_ids[0, current_length] = next_token
        prediction = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return {
            "prediction": prediction,
            "predicted_index": None,
            "generated_token_ids": generated,
            "stop_reason": stop_reason,
            "image_grid_signature": list(grid_signature),
            "image_token_count": len(image_ids),
            "image_token_ids_sha256": _canonical_sha256(image_ids),
            "input_tokens": len(prompt_ids),
            "output_tokens": len(generated),
        }


def _evaluate_manifest(
    inference: _NativeAcademicInference,
    manifest: Mapping[str, Any],
    loaded: Mapping[str, Mapping[str, AcademicExample]],
) -> dict[str, Any]:
    task_results: dict[str, Any] = {}
    task_names = manifest["selection"]["tasks"]
    for task in task_names:
        started = time.monotonic()
        manifest_task = manifest["tasks"][task]
        records = manifest_task["records"]
        examples = loaded[task]
        output_rows = []
        for row_index, record in enumerate(records):
            example = examples[record["example_id"]]
            donor = examples[record["shuffled_donor_id"]]
            control_outputs: dict[str, Any] = {}
            for control in CONTROLS:
                if control == "correct":
                    visual = _as_rgb_image(example.visual)
                elif control == "shuffled":
                    visual = _as_rgb_image(donor.visual)
                else:
                    recipient = _as_rgb_image(example.visual)
                    visual = Image.new("RGB", recipient.size, color=(0, 0, 0))
                    recipient.close()
                try:
                    prediction = inference.predict(example, visual)
                finally:
                    visual.close()
                if (
                    prediction["image_grid_signature"] != record["image_grid_signature"]
                    or prediction["image_token_count"] != record["image_token_count"]
                ):
                    raise ValueError(
                        f"{task}/{example.example_id}/{control} image-token layout differs"
                    )
                score = _score_prediction(
                    example,
                    prediction=prediction["prediction"],
                    predicted_index=prediction["predicted_index"],
                )
                control_outputs[control] = {**prediction, "score": score}
            if len({output["image_token_ids_sha256"] for output in control_outputs.values()}) != 1:
                raise ValueError(
                    f"{task}/{example.example_id} correct/shuffled/blank image IDs differ"
                )
            output_rows.append(
                {
                    "example_id": example.example_id,
                    "source_position": example.source_position,
                    "annotation_sha256": record["annotation_sha256"],
                    "image_sha256": record["image_sha256"],
                    "image_grid_signature": record["image_grid_signature"],
                    "image_token_count": record["image_token_count"],
                    "alignment_train_image_overlap": record["alignment_train_image_overlap"],
                    "shuffled_donor_id": record["shuffled_donor_id"],
                    "shuffled_image_sha256": record["shuffled_image_sha256"],
                    "shuffled_image_grid_signature": record["shuffled_image_grid_signature"],
                    "shuffled_alignment_train_image_overlap": record[
                        "shuffled_alignment_train_image_overlap"
                    ],
                    "question": example.question,
                    "gold_answers": list(example.answers),
                    "options": list(example.options),
                    "gold_answer_index": example.answer_index,
                    "stratum": example.stratum,
                    "controls": control_outputs,
                }
            )
            if get_rank() == 0 and (row_index == 0 or (row_index + 1) % 100 == 0):
                log.info("[%s] evaluated %d/%d examples", task, row_index + 1, len(records))
        aggregates = _aggregate_task_outputs(task, output_rows)
        task_results[task] = {
            "source": manifest_task["source"],
            "selection_count": len(records),
            "selection_sha256": manifest_task["selection_sha256"],
            "alignment_train_image_overlap_count": sum(
                bool(row["alignment_train_image_overlap"]) for row in output_rows
            ),
            "generation_stop_counts": _generation_stop_counts(task, output_rows),
            "elapsed_seconds": time.monotonic() - started,
            **aggregates,
            "examples": output_rows,
        }
        if get_rank() == 0:
            log.info("Finished %s: %s", task, aggregates)
        if dist.is_initialized():
            dist.barrier()
        gc_cuda()
    return task_results


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    """Read saved selections without replaying source inventories or changing donors."""
    manifest = _read_json(path)
    if manifest.get("format") != "vision_alignment_external_academic_manifest":
        raise ValueError("Unsupported academic selection manifest")
    content = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if manifest.get("content_sha256") != _canonical_sha256(content):
        raise ValueError("Academic manifest content checksum differs")
    tasks = manifest["selection"]["tasks"]
    if (
        not tasks
        or len(tasks) != len(set(tasks))
        or set(tasks) - set(DEFAULT_TASKS)
        or set(tasks) != set(manifest["tasks"])
        or manifest["selection"]["split"] != "validation"
        or manifest["controls"]["names"] != list(CONTROLS)
    ):
        raise ValueError("Academic manifest tasks, split, or controls differ")
    for task in tasks:
        payload = manifest["tasks"][task]
        records = payload["records"]
        by_id = {record["example_id"]: record for record in records}
        if (
            not records
            or len(by_id) != len(records)
            or payload["selection_count"] != len(records)
            or payload["selection_sha256"] != _canonical_sha256(records)
        ):
            raise ValueError(f"Invalid saved selection for {task}")
        for record in records:
            donor = by_id.get(record["shuffled_donor_id"])
            if donor is None or donor["image_sha256"] == record["image_sha256"]:
                raise ValueError(f"Missing or identical-image donor for {task}")
            if (
                donor["image_sha256"] != record["shuffled_image_sha256"]
                or donor["image_grid_signature"] != record["image_grid_signature"]
                or donor["image_grid_signature"] != record["shuffled_image_grid_signature"]
                or donor["alignment_train_image_overlap"]
                != record["shuffled_alignment_train_image_overlap"]
                or record["image_token_count"]
                != len(build_image_token_ids(*record["image_grid_signature"]))
            ):
                raise ValueError(f"Donor or image layout differs for {task}")
    return manifest


def load_selected_examples(
    manifest: Mapping[str, Any], tasks: Sequence[str]
) -> dict[str, dict[str, AcademicExample]]:
    """Load annotations and check only the selected images, not the training corpus."""
    loaded = {}
    image_hashes: dict[str, str] = {}
    for task in tasks:
        examples = {example.example_id: example for example in _load_task_examples(task)}
        selected = {}
        for record in manifest["tasks"][task]["records"]:
            example = examples[record["example_id"]]
            if (
                example.source_position != record["source_position"]
                or _canonical_sha256(example.annotation()) != record["annotation_sha256"]
            ):
                raise ValueError(f"Selected annotation differs: {task}/{example.example_id}")
            reference = example.image_reference
            cache_key = reference if isinstance(reference, str) else f"{task}/{example.example_id}"
            if cache_key not in image_hashes:
                image_hashes[cache_key] = _image_reference_sha256(reference)
            if image_hashes[cache_key] != record["image_sha256"]:
                raise ValueError(f"Selected image differs: {task}/{example.example_id}")
            if list(_molmo2_grid_signature(reference)) != record["image_grid_signature"]:
                raise ValueError(f"Selected image dimensions differ: {task}/{example.example_id}")
            selected[example.example_id] = example
        loaded[task] = selected
    return loaded


def _checkpoint_tokenizer_config(saved: Mapping[str, Any]) -> tuple[int, str | None]:
    dataset, artifacts = saved.get("dataset", {}), saved.get("artifacts", {})
    tokenizer = dataset.get("tokenizer", {})
    if (
        tokenizer.get("identifier", artifacts.get("tokenizer_id")) != TOKENIZER_ID
        or dataset.get("tokenizer_revision", artifacts.get("tokenizer_revision"))
        != TOKENIZER_REVISION
        or artifacts.get("tokenizer_fingerprint", TOKENIZER_FINGERPRINT) != TOKENIZER_FINGERPRINT
    ):
        raise ValueError("Checkpoint tokenizer differs from the academic benchmark")
    model = saved["model"]
    if (
        "vision" not in model
        or "lm" not in model
        or model.get("image_patch_token_id") != 100280
        or model["lm"]["vocab_size"] != 100352
        or tokenizer.get("pad_token_id", saved.get("collator", {}).get("pad_token_id")) != 100277
        or tokenizer.get("eos_token_id", 100257) != 100257
    ):
        raise ValueError("Checkpoint vocabulary or image/EOS/padding tokens differ")
    return model["lm"]["vocab_size"], saved.get("recipe", {}).get(
        "hf_cache_dir", artifacts.get("hf_cache_dir")
    )


def _load_tokenizer(cache_dir: str, model_vocab_size: int) -> tuple[Any, Molmo2TokenIds]:
    from huggingface_hub import snapshot_download
    from transformers import GPT2Tokenizer

    from olmo_core.nn.vision import prepare_molmo2_tokenizer

    snapshot = Path(
        snapshot_download(
            TOKENIZER_ID,
            revision=TOKENIZER_REVISION,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    )
    digest = hashlib.sha256()
    for name in (
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ):
        digest.update(f"{_file_sha256(snapshot / name)}  {name}\n".encode())
    if digest.hexdigest() != TOKENIZER_FINGERPRINT:
        raise ValueError("Academic tokenizer snapshot differs")
    tokenizer = GPT2Tokenizer.from_pretrained(snapshot, local_files_only=True)
    tokens = prepare_molmo2_tokenizer(tokenizer, model_vocab_size=model_vocab_size)
    if tokens != TOKEN_IDS:
        raise ValueError("Academic image-token layout differs")
    return tokenizer, tokens


def benchmark_definition(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Describe scoring and selected rows independently of allocation and checkpoint."""
    return {
        "protocol": "academic_document_v1",
        "selection_sha256": {
            task: manifest["tasks"][task]["selection_sha256"]
            for task in manifest["selection"]["tasks"]
        },
        "split": "validation",
        "controls": list(CONTROLS),
        "tokenizer": {"identifier": TOKENIZER_ID, "revision": TOKENIZER_REVISION},
        "message_format": "document",
        "free_answer_prompt": _free_answer_prompt("{question}"),
        "multiple_choice_prompt": _build_mc_prompt("{question}", ["{option_1}", "{option_2}"]),
        "max_new_tokens": 24,
        "max_crops": 8,
        "max_sequence_length": 8192,
        "metrics": {task: _metric_name(task) for task in manifest["selection"]["tasks"]},
        "overlap_scope": OVERLAP_SCOPE,
    }


def _definition(checkpoint: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    saved = _read_json(checkpoint / "config.json")
    _checkpoint_tokenizer_config(saved)
    state_metadata = (checkpoint / "model_and_optim" / ".metadata").stat()
    return {
        "benchmark": benchmark_definition(manifest),
        "execution": {
            "checkpoint": str(checkpoint),
            "config_sha256": _canonical_sha256(saved),
            "checkpoint_metadata": _read_json(checkpoint / ".metadata.json"),
            "state_metadata": {
                "bytes": state_metadata.st_size,
                "mtime_ns": state_metadata.st_mtime_ns,
            },
            "world_size": 8,
            "expert_parallel_path": "sync_1d",
            "attention_backend": "flex",
            "forward_context": "eager_grad_enabled",
            "sequence_bucket_size": 128,
            "torch_version": torch.__version__,
        },
    }


def validate_task_result(task: str, result: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Rescore completed rows and verify selection, controls, and aggregate consistency."""
    records = manifest["tasks"][task]["records"]
    rows = result["examples"]
    if len(rows) != len(records) or result["selection_count"] != len(records):
        raise ValueError(f"Incomplete academic task: {task}")
    if result["selection_sha256"] != manifest["tasks"][task]["selection_sha256"]:
        raise ValueError(f"Academic task selection differs: {task}")
    for row, record in zip(rows, records):
        if any(row.get(key) != value for key, value in record.items()):
            raise ValueError(f"Academic task row differs: {task}/{record['example_id']}")
        example = AcademicExample(
            task=task,
            example_id=row["example_id"],
            source_position=row["source_position"],
            visual=None,
            image_reference=None,
            question=row["question"],
            answers=tuple(row["gold_answers"]),
            options=tuple(row["options"]),
            answer_index=row["gold_answer_index"],
            stratum=row["stratum"],
        )
        if _canonical_sha256(example.annotation()) != record["annotation_sha256"]:
            raise ValueError(f"Academic result annotation differs: {task}")
        if set(row["controls"]) != set(CONTROLS):
            raise ValueError(f"Academic result controls differ: {task}")
        for control in CONTROLS:
            prediction = row["controls"][control]
            if example.options:
                letters = string.ascii_uppercase[: len(example.options)]
                probabilities = prediction["candidate_log_probabilities"]
                if set(probabilities) != set(letters) or any(
                    not math.isfinite(value) for value in probabilities.values()
                ):
                    raise ValueError(f"Invalid option probabilities: {task}/{control}")
                winner = max(range(len(letters)), key=lambda index: probabilities[letters[index]])
                if (
                    prediction["predicted_index"] != winner
                    or prediction["prediction"] != letters[winner]
                    or prediction["output_tokens"] != 1
                ):
                    raise ValueError(f"Academic option argmax differs: {task}/{control}")
            score = _score_prediction(
                example,
                prediction=prediction["prediction"],
                predicted_index=prediction["predicted_index"],
            )
            if not math.isfinite(score) or prediction["score"] != score:
                raise ValueError(f"Academic result score differs: {task}/{control}")
            if (
                prediction["image_grid_signature"] != record["image_grid_signature"]
                or prediction["image_token_count"] != record["image_token_count"]
                or prediction["image_token_ids_sha256"]
                != _canonical_sha256(
                    build_image_token_ids(*record["image_grid_signature"], token_ids=TOKEN_IDS)
                )
            ):
                raise ValueError(f"Academic result image layout differs: {task}/{control}")
            if not example.options and (
                prediction["stop_reason"] not in ("eos", "max_tokens")
                or not 1 <= len(prediction["generated_token_ids"]) <= 24
                or prediction["output_tokens"] != len(prediction["generated_token_ids"])
                or any(
                    type(token) is not int or not 0 <= token < 100278
                    for token in prediction["generated_token_ids"]
                )
                or 100257 in prediction["generated_token_ids"][:-1]
                or (prediction["stop_reason"] == "eos")
                != (prediction["generated_token_ids"][-1] == 100257)
                or (prediction["stop_reason"] == "max_tokens" and prediction["output_tokens"] != 24)
            ):
                raise ValueError(f"Academic generation limits differ: {task}/{control}")
    expected = {
        **_aggregate_task_outputs(task, rows),
        "generation_stop_counts": _generation_stop_counts(task, rows),
        "alignment_train_image_overlap_count": sum(
            row["alignment_train_image_overlap"] for row in rows
        ),
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Academic task aggregates differ: {task}")


def _write_result(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = _read_json(path)
        if (
            previous.get("format") != RESULT_FORMAT
            or previous.get("definition") != payload["definition"]
        ):
            raise ValueError(f"Output belongs to another evaluation: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".json", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(dict(payload), stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def _task_path(output: Path, task: str) -> Path:
    return output.parent / f"{output.stem}.tasks" / f"{task}.json"


def _load_task_result(
    path: Path, task: str, manifest: Mapping[str, Any], definition: Mapping[str, Any]
) -> dict[str, Any]:
    value = _read_json(path)
    if (
        value.get("format") != RESULT_FORMAT
        or value.get("definition") != definition
        or value.get("task") != task
        or value.get("completed") is not True
    ):
        raise ValueError(f"Incomplete or mismatched academic task: {path}")
    validate_task_result(task, value["result"], manifest)
    return value["result"]


def merge_results(
    output: Path, manifest: Mapping[str, Any], definition: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and merge all completed task files on CPU without loading datasets or weights."""
    tasks = {
        task: _load_task_result(_task_path(output, task), task, manifest, definition)
        for task in manifest["selection"]["tasks"]
    }
    receipt = {
        "format": RESULT_FORMAT,
        "definition": definition,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed": True,
        "tasks": tasks,
    }
    _write_result(output, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> None:
    """Evaluate selected academic tasks or validate/merge their persisted results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=DEFAULT_TASKS)
    parser.add_argument("--hf-cache")
    parser.add_argument("--checkpoint-load-threads", type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--check-complete", action="store_true")
    mode.add_argument("--merge", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    checkpoint, output = (
        args.checkpoint.expanduser().resolve(),
        args.output.expanduser().resolve(),
    )
    if output.is_relative_to(checkpoint) or checkpoint.is_relative_to(output):
        raise ValueError("Evaluation output must be separate from checkpoint files")
    manifest = load_manifest(args.manifest)
    definition = _definition(checkpoint, manifest)
    tasks = args.tasks or manifest["selection"]["tasks"]
    if len(tasks) != len(set(tasks)) or set(tasks) - set(manifest["selection"]["tasks"]):
        raise ValueError("Requested tasks must be a unique subset of the saved manifest")
    if args.dry_run:
        print(json.dumps({**definition, "requested_tasks": tasks}, indent=2))
        return
    if args.merge:
        merge_results(output, manifest, definition)
        return
    pending = []
    for task in tasks:
        path = _task_path(output, task)
        if path.exists() or args.check_complete:
            _load_task_result(path, task, manifest, definition)
        else:
            pending.append(task)
    if args.check_complete or (not pending and int(os.environ.get("WORLD_SIZE", "1")) == 1):
        if not args.check_complete and all(
            _task_path(output, task).is_file() for task in manifest["selection"]["tasks"]
        ):
            merge_results(output, manifest, definition)
        print(f"Validated completed academic tasks: {', '.join(tasks)}")
        return
    if args.checkpoint_load_threads <= 0 or any(
        int(os.environ.get(key, "1")) != 8 for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE")
    ):
        raise ValueError(
            "Academic inference requires one node with eight GPU ranks and positive load threads"
        )
    _evaluate(checkpoint, output, manifest, definition, tasks, args)


def _evaluate(
    checkpoint: Path,
    output: Path,
    manifest: Mapping[str, Any],
    definition: Mapping[str, Any],
    tasks: Sequence[str],
    args: argparse.Namespace,
) -> None:
    from olmo_core.eval.multimodal_checkpoint import (
        build_model_and_module_config,
        checkpoint_state_dir,
        native_checkpoint_load_coverage_distributed,
    )
    from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath
    from olmo_core.train import (
        prepare_training_environment,
        teardown_training_environment,
    )

    saved = _read_json(checkpoint / "config.json")
    vocab_size, default_cache = _checkpoint_tokenizer_config(saved)
    cache = args.hf_cache or default_cache
    if not cache:
        raise ValueError("A local tokenizer cache is required")
    prepare_training_environment()
    try:
        completion: list[Any] = [None]
        if get_rank() == 0:
            try:
                pending = []
                for task in tasks:
                    path = _task_path(output, task)
                    if path.exists():
                        _load_task_result(path, task, manifest, definition)
                    else:
                        pending.append(task)
                completion[0] = (pending, None)
            except Exception as error:
                completion[0] = ([], str(error))
        dist.broadcast_object_list(completion, src=0)
        tasks, error = completion[0]
        if error is not None:
            raise ValueError(error)
        if not tasks:
            if get_rank() == 0 and all(
                _task_path(output, task).is_file() for task in manifest["selection"]["tasks"]
            ):
                merge_results(output, manifest, definition)
            dist.barrier()
            return
        tokenizer, token_ids = _load_tokenizer(cache, vocab_size)
        loaded = load_selected_examples(manifest, tasks)
        model, config, kind = build_model_and_module_config(
            saved,
            ep_degree=8,
            max_sequence_length=8192,
            rank_batch_size=8192,
            ep_path=ExpertParallelPath.sync_1d,
        )
        if kind != "multimodal_stage1":
            raise ValueError("Academic evaluation requires a native multimodal checkpoint")
        module = config.build(model, eval_only=True)
        state_dir = checkpoint_state_dir(checkpoint)
        coverage = native_checkpoint_load_coverage_distributed(module, state_dir)
        module.load_state_dict_direct(
            state_dir,
            process_group=dist.group.WORLD,
            thread_count=args.checkpoint_load_threads,
            load_optim_state=False,
        )
        for part in module.model_parts:
            part.eval()
        inference = _NativeAcademicInference(
            module,
            tokenizer,
            token_ids,
            max_sequence_length=8192,
            max_crops=8,
            max_new_tokens=24,
            sequence_bucket_size=128,
        )
        for task in tasks:
            selected = {
                **manifest,
                "selection": {**manifest["selection"], "tasks": [task]},
            }
            result = _evaluate_manifest(inference, selected, loaded)[task]
            validate_task_result(task, result, manifest)
            if get_rank() == 0:
                _write_result(
                    _task_path(output, task),
                    {
                        "format": RESULT_FORMAT,
                        "definition": definition,
                        "completed": True,
                        "task": task,
                        "result": result,
                        "native_checkpoint_load": coverage,
                    },
                )
            dist.barrier()
        if _definition(checkpoint, manifest) != definition:
            raise ValueError("Checkpoint metadata changed during evaluation")
        if get_rank() == 0 and all(
            _task_path(output, task).is_file() for task in manifest["selection"]["tasks"]
        ):
            merge_results(output, manifest, definition)
        dist.barrier()
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
