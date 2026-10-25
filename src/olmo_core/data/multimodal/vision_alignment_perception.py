"""OCR and quality-filtered alignment datasets in document format.

OCR examples use ``Question: ...\nAnswer:`` prompts. FineVision examples retain their
single-turn prompts and responses after quality filtering. Content fingerprints identify
the ordered annotations and encoding configuration, not external image-file contents.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from olmo_core.config import Config
from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds

from .academic.registry import build_academic_data
from .finevision import FINEVISION_ROOT, FineVisionDataset, FineVisionDatasetConfig
from .message_sequence import encode_sft_example
from .sft_common import (
    SftMessageFormat,
    decode_pil_image,
    get_example_with_skip,
    sft_example_rng,
    truncate_example,
    validate_sft_message_format,
)

__all__ = [
    "VISION_ALIGNMENT_OCR_SOURCES",
    "VisionAlignmentAuditedAlignmentDataset",
    "VisionAlignmentAuditedAlignmentDatasetConfig",
    "VisionAlignmentOcrDocumentDataset",
    "VisionAlignmentOcrDocumentDatasetConfig",
]

VISION_ALIGNMENT_OCR_SOURCES: Tuple[str, ...] = (
    "text_vqa",
    "doc_qa",
    "info_qa",
    "chart_qa_weighted",
)
"""Direct-answer OCR, document, infographic, and chart sources."""

_OCR_FINGERPRINT_VERSION = "vision-alignment-ocr-document-v1"
_ALIGNMENT_FINGERPRINT_VERSION = "vision-alignment-audited-alignment-v1"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _clean_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _answer_candidates(row: Mapping[str, Any]) -> List[str]:
    raw: Any
    if "answers" in row:
        raw = row.get("answers")
    else:
        raw = row.get("answer")
    if isinstance(raw, str):
        candidates: Sequence[Any] = [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        candidates = raw
    elif raw is None:
        candidates = []
    else:
        candidates = [raw]

    answers: List[str] = []
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            candidate = candidate.get("answer", candidate.get("raw_answer"))
        text = _clean_text(candidate)
        if text:
            answers.append(text)
    return answers


def _consensus_answer(row: Mapping[str, Any]) -> str:
    """Choose a deterministic modal direct answer, preserving first-seen tie order."""
    answers = _answer_candidates(row)
    if not answers:
        return ""
    normalized = [answer.casefold() for answer in answers]
    counts = Counter(normalized)
    best_count = max(counts.values())
    winner = next(value for value in normalized if counts[value] == best_count)
    return next(answer for answer, value in zip(answers, normalized) if value == winner)


def _image_identity(image: Any) -> str:
    if isinstance(image, str):
        return os.path.realpath(image)
    if isinstance(image, Mapping):
        data = image.get("bytes")
        if isinstance(data, (bytes, bytearray)) and data:
            return f"embedded-sha256:{hashlib.sha256(data).hexdigest()}"
        path = image.get("path")
        if isinstance(path, str) and path:
            return os.path.realpath(path)
    # In-memory objects have no stable file or byte identity.
    return f"object:{type(image).__module__}.{type(image).__qualname__}"


@dataclass
class VisionAlignmentOcrDocumentDatasetConfig(Config):
    """Configuration for deterministic OCR/document continued pretraining.

    :param source_names: Direct-answer academic source names, concatenated in order.
    :param split: Requested raw-data split.
    :param max_crops: Maximum Molmo2 crops for the single image.
    :param max_sequence_length: Maximum serialized length before packing.
    :param loss_token_weighting: Response-token weighting policy.
    :param message_weight: Optional positive multiplier for response-token loss weights.
    :param token_ids: Model-specific Molmo2 image token identities.
    :param message_format: Supported serialization format; must be ``document``.
    :param prompt_prefix: Stable prefix placed before the source question.
    :param answer_prefix: Stable delimiter placed after the source question.
    :param seed: Deterministic image-augmentation seed.
    :param require_existing_images: Check every path-backed image during validation.
    """

    source_names: Tuple[str, ...] = VISION_ALIGNMENT_OCR_SOURCES
    split: str = "train"
    max_crops: int = 8
    max_sequence_length: int = 2560
    loss_token_weighting: str = "root_subsegments_root_tokens"
    message_weight: Optional[float] = None
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)
    message_format: SftMessageFormat = "document"
    prompt_prefix: str = "Question: "
    answer_prefix: str = "\nAnswer:"
    seed: int = 0
    require_existing_images: bool = True
    skip_bad_rows: bool = False
    """Whether unusable rows may be replaced by a later row."""

    def build(self, tokenizer: Any) -> "VisionAlignmentOcrDocumentDataset":
        """Build the OCR/document dataset.

        :param tokenizer: Tokenizer compatible with :attr:`token_ids`.
        :returns: The configured map-style dataset.
        """
        return VisionAlignmentOcrDocumentDataset(self, tokenizer)


class VisionAlignmentOcrDocumentDataset:
    """Concatenated direct-answer OCR/document source in native document layout."""

    content_fingerprint_version = _OCR_FINGERPRINT_VERSION

    def __init__(self, config: VisionAlignmentOcrDocumentDatasetConfig, tokenizer: Any):
        validate_sft_message_format(config.message_format)
        if config.message_format != "document":
            raise ValueError("Vision-alignment OCR data requires message_format='document'")
        if not config.source_names or len(set(config.source_names)) != len(config.source_names):
            raise ValueError("OCR source_names must be non-empty and unique")
        if any(name not in VISION_ALIGNMENT_OCR_SOURCES for name in config.source_names):
            raise ValueError(
                f"OCR source_names must be selected from {VISION_ALIGNMENT_OCR_SOURCES}"
            )
        if (
            isinstance(config.max_crops, bool)
            or not isinstance(config.max_crops, int)
            or config.max_crops <= 0
        ):
            raise ValueError("Vision-alignment OCR max_crops must be a positive integer")
        if config.max_sequence_length <= 0:
            raise ValueError("Vision-alignment OCR max_sequence_length must be positive")
        if config.message_weight is not None and (
            isinstance(config.message_weight, bool)
            or not isinstance(config.message_weight, (int, float))
            or not math.isfinite(config.message_weight)
            or config.message_weight <= 0
        ):
            raise ValueError("Vision-alignment OCR message_weight must be finite and positive")

        self.config = config
        self.tokenizer = tokenizer
        self._sources = [
            build_academic_data(name, split=config.split) for name in config.source_names
        ]
        self._offsets: List[int] = [0]
        for source in self._sources:
            self._offsets.append(self._offsets[-1] + len(source))
        if self._offsets[-1] == 0:
            raise ValueError("Vision-alignment OCR dataset is empty")
        self.content_fingerprint = self._build_content_fingerprint()
        self._warned = 0

    def __len__(self) -> int:
        return self._offsets[-1]

    def _locate(self, index: int) -> Tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source_index = int(np.searchsorted(self._offsets, index, side="right") - 1)
        return source_index, index - self._offsets[source_index]

    def _row(self, index: int) -> Tuple[str, Mapping[str, Any]]:
        source_index, local_index = self._locate(index)
        row = self._sources[source_index][local_index]
        if not isinstance(row, Mapping):
            raise ValueError(f"OCR row {index} must be a mapping")
        return self.config.source_names[source_index], row

    @staticmethod
    def _descriptor(source_name: str, row: Mapping[str, Any]) -> Dict[str, str]:
        return {
            "source": source_name,
            "image": _image_identity(row.get("image")),
            "question": _clean_text(row.get("question")),
            "answer": _consensus_answer(row),
        }

    def _build_content_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for index in range(len(self)):
            source_name, row = self._row(index)
            digest.update(
                json.dumps(
                    {"index": index, **self._descriptor(source_name, row)},
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
        identity = {
            "version": self.content_fingerprint_version,
            "source_names": list(self.config.source_names),
            "split": self.config.split,
            "rows": len(self),
            "ordered_annotations_sha256": digest.hexdigest(),
            "max_crops": self.config.max_crops,
            "max_sequence_length": self.config.max_sequence_length,
            "loss_token_weighting": self.config.loss_token_weighting,
            "message_format": self.config.message_format,
            "prompt_prefix": self.config.prompt_prefix,
            "answer_prefix": self.config.answer_prefix,
            "seed": self.config.seed,
            "skip_bad_rows": self.config.skip_bad_rows,
        }
        if self.config.message_weight is not None:
            identity["message_weight"] = float(self.config.message_weight)
        return _canonical_sha256(identity)

    def raw_image_references(self, index: int) -> Tuple[Any, ...]:
        """Return the exact path-backed image reference for one logical OCR row."""
        _, row = self._row(index)
        image = row.get("image")
        if image is None:
            raise ValueError(f"OCR row {index} has no image reference")
        return (image,)

    def validate_required_annotations(self) -> None:
        """Validate all questions, answers, and path-backed images without decoding them.

        :raises ValueError: If any row is unusable or a required image path is absent.
        """
        invalid_count = 0
        first_invalid: List[Tuple[int, str]] = []
        for index in range(len(self)):
            _, row = self._row(index)
            reason = ""
            if not _clean_text(row.get("question")):
                reason = "blank question"
            elif not _consensus_answer(row):
                reason = "blank answer"
            elif row.get("image") is None:
                reason = "missing image"
            elif (
                self.config.require_existing_images
                and isinstance(row.get("image"), str)
                and not os.path.isfile(str(row["image"]))
            ):
                reason = "image path is not a file"
            if reason:
                invalid_count += 1
                if len(first_invalid) < 8:
                    first_invalid.append((index, reason))
        if invalid_count:
            raise ValueError(
                f"Vision-alignment OCR source has {invalid_count} invalid rows "
                f"(first: {first_invalid})"
            )

    def _build(self, index: int, epoch: int = 0) -> Dict[str, np.ndarray]:
        _, row = self._row(index)
        question = _clean_text(row.get("question"))
        answer = _consensus_answer(row)
        if not question or not answer:
            raise ValueError(f"OCR row {index} has blank supervision")
        prompt = f"{self.config.prompt_prefix}{question}{self.config.answer_prefix}"
        rng = sft_example_rng(self.config.seed, index, epoch, self.config.message_format)
        sequence = encode_sft_example(
            self.tokenizer,
            decode_pil_image(row.get("image")),
            [(prompt, answer)],
            max_crops=self.config.max_crops,
            max_images=1,
            loss_token_weighting=self.config.loss_token_weighting,
            token_ids=self.config.token_ids,
            message_format=self.config.message_format,
            message_weight=self.config.message_weight,
            shuffle_rng=rng,
        )
        original_length = len(sequence["input_ids"])
        example = truncate_example(
            sequence,
            self.config.max_sequence_length,
            image_token_ids=self.config.token_ids.image_token_ids,
        )
        example["metadata"] = {
            **example.get("metadata", {}),
            "original_length": original_length,
            "truncated": original_length > self.config.max_sequence_length,
        }
        return example

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Build one epoch-zero OCR/document example."""
        return self.get(index, 0)

    def get(self, index: int, epoch: int = 0) -> Dict[str, np.ndarray]:
        """Build one epoch-aware example, optionally skipping unusable rows."""
        if self.config.skip_bad_rows:
            return get_example_with_skip(self, index, len(self), epoch)
        return self._build(index, epoch)


@dataclass
class VisionAlignmentAuditedAlignmentDatasetConfig(Config):
    """Configuration for the quality-filtered FineVision alignment pool.

    The source concatenates ``visualwebinstruct(filtered)`` and ``geo170k(align)`` after
    requiring formatting, visual dependency, and relevance ratings of at least four.
    Arrow ``DatasetDict`` paths may provide a validation split. Raw local parquet datasets
    contain only ``train`` and reject ``split='validation'``.
    """

    root: str = FINEVISION_ROOT
    visualweb_path: Optional[str] = None
    geo170k_path: Optional[str] = None
    visualweb_fingerprint: Optional[str] = None
    geo170k_fingerprint: Optional[str] = None
    split: str = "train"
    min_formatting: int = 4
    min_visual_dependency: int = 4
    min_relevance: int = 4
    max_crops: int = 8
    max_sequence_length: int = 2560
    loss_token_weighting: str = "root_subsegments_root_tokens"
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)
    message_format: SftMessageFormat = "document"
    seed: int = 0

    def build(self, tokenizer: Any) -> "VisionAlignmentAuditedAlignmentDataset":
        """Build the combined audited-alignment dataset."""
        return VisionAlignmentAuditedAlignmentDataset(self, tokenizer)


class VisionAlignmentAuditedAlignmentDataset:
    """Quality-filtered single-turn visual alignment examples in document layout."""

    content_fingerprint_version = _ALIGNMENT_FINGERPRINT_VERSION

    def __init__(
        self,
        config: VisionAlignmentAuditedAlignmentDatasetConfig,
        tokenizer: Any,
    ):
        validate_sft_message_format(config.message_format)
        if config.message_format != "document":
            raise ValueError("Audited alignment requires message_format='document'")
        if (
            isinstance(config.max_crops, bool)
            or not isinstance(config.max_crops, int)
            or config.max_crops <= 0
        ):
            raise ValueError("Audited-alignment max_crops must be a positive integer")
        if min(config.min_formatting, config.min_visual_dependency, config.min_relevance) < 1:
            raise ValueError("Audited-alignment quality thresholds must be positive")

        self.config = config
        self.tokenizer = tokenizer
        self._datasets: List[FineVisionDataset] = []
        for config_name, explicit_path, expected_fingerprint in (
            (
                "visualwebinstruct(filtered)",
                config.visualweb_path,
                config.visualweb_fingerprint,
            ),
            ("geo170k(align)", config.geo170k_path, config.geo170k_fingerprint),
        ):
            child_config = FineVisionDatasetConfig(
                config_name=config_name,
                root=config.root,
                dataset_path=explicit_path,
                expected_materialized_fingerprint=expected_fingerprint,
                split=config.split,
                min_formatting=config.min_formatting,
                min_visual_dependency=config.min_visual_dependency,
                min_relevance=config.min_relevance,
                require_quality_columns=True,
                strict_annotations=True,
                skip_bad_rows=False,
                max_crops=config.max_crops,
                max_images=1,
                max_sequence_length=config.max_sequence_length,
                loss_token_weighting=config.loss_token_weighting,
                token_ids=config.token_ids,
                message_format=config.message_format,
                seed=config.seed,
            )
            child = child_config.build(tokenizer)
            if len(child) == 0:
                raise ValueError(f"Audited-alignment child {config_name!r} is empty")
            self._datasets.append(child)
        self._offsets = [0]
        for dataset in self._datasets:
            self._offsets.append(self._offsets[-1] + len(dataset))
        self.content_fingerprint = _canonical_sha256(
            {
                "version": self.content_fingerprint_version,
                "split": config.split,
                "child_fingerprints": [dataset.content_fingerprint for dataset in self._datasets],
                "child_lengths": [len(dataset) for dataset in self._datasets],
                "max_crops": config.max_crops,
                "max_sequence_length": config.max_sequence_length,
                "loss_token_weighting": config.loss_token_weighting,
                "message_format": config.message_format,
                "seed": config.seed,
            }
        )

    def __len__(self) -> int:
        return self._offsets[-1]

    def _locate(self, index: int) -> Tuple[FineVisionDataset, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        dataset_index = int(np.searchsorted(self._offsets, index, side="right") - 1)
        return self._datasets[dataset_index], index - self._offsets[dataset_index]

    def validate_required_annotations(self) -> None:
        """Run each child's strict single-image/single-turn annotation scan."""
        for dataset in self._datasets:
            dataset.validate_required_annotations()

    def raw_image_references(self, index: int) -> Tuple[Any, ...]:
        """Return the exact encoded image reference for one logical alignment row."""
        dataset, local_index = self._locate(index)
        return dataset.raw_image_references(local_index)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Build one epoch-zero quality-filtered alignment example."""
        return self.get(index, 0)

    def get(self, index: int, epoch: int = 0) -> Dict[str, np.ndarray]:
        """Build one epoch-aware alignment example."""
        dataset, local_index = self._locate(index)
        return dataset.get(local_index, epoch)
