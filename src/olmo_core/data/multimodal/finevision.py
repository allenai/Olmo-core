"""FineVision instruction-tuning datasets (``HuggingFaceM4/FineVision``).

One loader for any FineVision config, since they all share the same row schema:

* ``images`` — a list of images.
* ``texts`` — a list of ``{"user": ..., "assistant": ...}`` turns; the user side holds the
  instruction, the assistant side the supervision target.
* ``source`` plus four quality signals with per-turn ratings and a per-row minimum:
  ``formatting``, ``visual_dependency``, ``image_correspondence`` and ``relevance``. The
  ``*_min`` columns are exposed as optional filters on :class:`FineVisionDatasetConfig`.

A row is assembled as ONE sequential conversation branch (loss on every assistant turn,
one EOS target at the end) by
:func:`~olmo_core.data.multimodal.message_sequence.encode_sft_example`, using the selected
Qwen or native-document layout. ``Image {i+1}`` prefixes are retained
when a row carries several images.

Configs verified against the copies on weka (see :data:`FINEVISION_ROOT`):

===========================  =========  ================  =========================
config                       rows       ``<image>`` mark  notes
===========================  =========  ================  =========================
``visualwebinstruct(filtered)``  263,581  leading, 1/row  web visual instruction
``mavis_math_rule_geo``           99,986  leading, 1/row  synthetic geometry w/ CoT
``mavis_math_metagen``            87,348  **none**        synthetic math w/ CoT
``geo170k(align)``                35,297  **none**        geometry caption/alignment
``geo170k(qa)``                   12,101  **none**        geometry multiple-choice
===========================  =========  ================  =========================

Every one of those is single-turn with exactly one image per row. Note that three of them
carry **no** ``<image>`` marker at all: the image block is positioned from the ``images``
column, not from the marker, so both layouts work identically (any marker present is
stripped so it is never tokenized as literal text).

.. warning::
    ``image_correspondence_min`` is very low on the synthetic-geometry configs — 75% of
    ``geo170k(qa)`` and 63% of ``mavis_math_metagen`` rows score 1 — so a naive
    ``min_image_correspondence=4`` discards most of them. Prefer
    ``min_visual_dependency``, which is 4-5 for ~95% of those rows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from olmo_core.config import Config
from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds

from .fs_cache import maybe_cache_env
from .message_sequence import encode_sft_example
from .sft_common import (
    SftMessageFormat,
    count_image_placeholders,
    decode_pil_image,
    get_example_with_skip,
    load_hf_dataset,
    sft_example_rng,
    strip_image_placeholders,
    truncate_example,
    validate_sft_message_format,
)

__all__ = [
    "FineVisionDatasetConfig",
    "FineVisionDataset",
    "VisualWebInstructDatasetConfig",
    "FINEVISION_ROOT",
]

log = logging.getLogger(__name__)

FINEVISION_ROOT = "/weka/oe-training-default/mm-olmo/hf_datasets/HuggingFaceM4___FineVision"
"""Directory holding one subdirectory of parquet shards per downloaded FineVision config."""

_QUALITY_COLUMNS = {
    "min_formatting": "formatting_min",
    "min_visual_dependency": "visual_dependency_min",
    "min_image_correspondence": "image_correspondence_min",
    "min_relevance": "relevance_min",
}


def _file_identity(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _materialized_file_identities(path: str) -> dict[Path, tuple[int, ...]]:
    root = Path(path).resolve()
    return {
        file: _file_identity(file)
        for file in [root / "dataset_info.json", *sorted(root.glob("data-*.arrow"))]
    }


@maybe_cache_env("OLMO_CORE_DATA_VERIFICATION_CACHE_DIR")
def _verify_file(path: str, identity: tuple[int, ...], expected_sha256: str) -> None:
    """Verify bytes, caching only when the data-verification cache is configured."""
    if _file_identity(Path(path)) != identity:
        raise ValueError(f"FineVision file changed before verification: {path}")
    with open(path, "rb") as stream:
        observed = hashlib.file_digest(stream, "sha256").hexdigest()
    if _file_identity(Path(path)) != identity:
        raise ValueError(f"FineVision file changed during verification: {path}")
    if observed != expected_sha256:
        raise ValueError(f"FineVision file checksum differs: {path}")


def _verify_materialization(config: FineVisionDatasetConfig, dataset: Any) -> None:
    """Validate the materialization content pin, independently of HF transformation hashes."""
    root = Path(config.resolved_path()).resolve()
    manifest_path = root.parent / "vision-alignment-finevision-materialization.json"
    with manifest_path.open() as stream:
        manifest = json.load(stream)
    if (
        manifest.get("format") != "vision_alignment_finevision_materialization"
        or manifest.get("version") != 1
        or manifest.get("status") != "verified"
    ):
        raise ValueError(f"Invalid FineVision materialization manifest: {manifest_path}")
    outputs = [
        output
        for output in manifest["outputs"]
        if output["name"] == config.config_name and output["path"] == root.name
    ]
    if len(outputs) != 1:
        raise ValueError(f"FineVision manifest does not identify {root}")
    output = outputs[0]
    payload = {
        "version": "vision-alignment-finevision-arrow-content-v1",
        "source_name": config.config_name,
        "rows": output["rows"],
        "physical_schema_sha256": output["physical_schema_sha256"],
        "shards": [
            {"rows": shard["rows"], "sha256": shard["sha256"]} for shard in output["shards"]
        ],
        "dataset_info_sha256": output["dataset_info_sha256"],
    }
    observed = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()
    if (
        observed != config.expected_materialized_fingerprint
        or observed != output["dataset_fingerprint"]
    ):
        raise ValueError(f"FineVision materialization fingerprint differs for {root}")
    shards = [root.parent / shard["path"] for shard in output["shards"]]
    if (
        not shards
        or any(path.parent != root for path in shards)
        or shards != sorted(root.glob("data-*.arrow"))
        or shards != [Path(item["filename"]).resolve() for item in dataset.cache_files]
        or len(dataset) != output["rows"]
        or sum(shard["rows"] for shard in output["shards"]) != len(dataset)
    ):
        raise ValueError(f"FineVision loaded shard inventory differs for {root}")
    files = [(root / "dataset_info.json", output["dataset_info_sha256"])] + [
        (path, shard["sha256"]) for path, shard in zip(shards, output["shards"])
    ]
    identities = [_file_identity(path) for path, _ in files]
    for (path, checksum), identity in zip(files, identities):
        _verify_file(str(path), identity, checksum)
    if identities != [_file_identity(path) for path, _ in files]:
        raise ValueError(f"FineVision materialization changed during verification: {root}")


def _parse_turn(turn: Any, *, first: bool, has_image: bool) -> tuple[str, str] | None:
    """Normalize a usable turn, including an explicitly image-only first prompt."""
    if not isinstance(turn, dict):
        return None
    raw_user, raw_assistant = turn.get("user"), turn.get("assistant")
    if not isinstance(raw_user, str) or not isinstance(raw_assistant, str):
        return None
    user = strip_image_placeholders(raw_user)
    assistant = raw_assistant.strip()
    image_only = first and has_image and count_image_placeholders(raw_user) > 0
    if not assistant or (not user and not image_only):
        return None
    return user, assistant


@dataclass
class FineVisionDatasetConfig(Config):
    """Configuration for :class:`FineVisionDataset`."""

    config_name: str = "visualwebinstruct(filtered)"
    """FineVision config to load, e.g. ``"mavis_math_rule_geo"`` or ``"geo170k(qa)"``.
    Resolved against :attr:`root` unless :attr:`dataset_path` is set."""

    root: str = FINEVISION_ROOT
    """Directory containing one subdirectory per FineVision config."""

    dataset_path: Optional[str] = None
    """Explicit parquet directory / glob / file, or a ``save_to_disk`` Arrow directory.
    Overrides :attr:`root` + :attr:`config_name` when set."""

    expected_materialized_fingerprint: Optional[str] = None
    """Content SHA-256 from a FineVision materialization manifest.

    When set, verify the loaded Arrow files against the adjacent
    ``vision-alignment-finevision-materialization.json`` manifest. This is not a Hugging
    Face dataset fingerprint. Set ``OLMO_CORE_DATA_VERIFICATION_CACHE_DIR`` to cache byte verification
    across processes and launches; file replacement or modification invalidates the cache.
    When unset, use the live Arrow fingerprint for resume identity.
    """

    split: str = "train"

    texts_column: str = "texts"
    """Column holding the list of ``{"user", "assistant"}`` turns."""

    images_column: str = "images"

    min_formatting: Optional[int] = None
    """Keep rows with ``formatting_min >=`` this (1-5; how well-formed the answer is)."""

    min_visual_dependency: Optional[int] = None
    """Keep rows with ``visual_dependency_min >=`` this (1-5; does the answer need the
    image?). Usually the most effective filter for image-grounded training."""

    min_image_correspondence: Optional[int] = None
    """Keep rows with ``image_correspondence_min >=`` this (1-5). See the module-level
    warning before using this on the synthetic-geometry configs."""

    min_relevance: Optional[int] = None
    """Keep rows with ``relevance_min >=`` this (1-5)."""

    require_quality_columns: bool = False
    """Fail when a configured quality column is absent instead of ignoring that filter."""

    strict_annotations: bool = False
    """Require every selected row to contain usable text and exactly one image/turn.

    Enabled for the single-image, single-turn alignment subset.
    """

    skip_bad_rows: bool = True
    """Deterministically advance over bad rows.

    Disable this when logical indices must identify exact source rows and images.
    """

    max_crops: int = 8
    """Max high-res crops *per image*. Rows with several images cost a multiple of this."""

    max_images: int = 5
    """Max images per row (extra images are dropped, matching the stage-2 budget)."""

    max_sequence_length: int = 4096
    loss_token_weighting: str = "root_subsegments"
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)
    message_format: SftMessageFormat = "qwen3"
    seed: int = 0

    def resolved_path(self) -> str:
        """The directory this config will read.

        :returns: :attr:`dataset_path` if set, else ``root/config_name``.
        """
        if self.dataset_path is not None:
            return self.dataset_path
        return os.path.join(self.root, self.config_name)

    def build(self, tokenizer) -> "FineVisionDataset":
        """Construct the dataset.

        :param tokenizer: A Molmo2 chat tokenizer.
        """
        return FineVisionDataset(self, tokenizer)


@dataclass
class VisualWebInstructDatasetConfig(FineVisionDatasetConfig):
    """:class:`FineVisionDatasetConfig` pinned to ``visualwebinstruct(filtered)``."""

    config_name: str = "visualwebinstruct(filtered)"


class FineVisionDataset:
    """Map-style dataset yielding packed FineVision instruction examples."""

    content_fingerprint_version = "finevision-runtime-v1"

    def __init__(self, config: FineVisionDatasetConfig, tokenizer):
        validate_sft_message_format(config.message_format)
        self.config = config
        self.tokenizer = tokenizer

        identities = (
            _materialized_file_identities(config.resolved_path())
            if config.expected_materialized_fingerprint is not None
            else None
        )
        self._data = load_hf_dataset(
            config.resolved_path(),
            config.split,
            keep_columns=[config.texts_column, config.images_column]
            + list(_QUALITY_COLUMNS.values()),
        )
        if config.expected_materialized_fingerprint is not None:
            _verify_materialization(config, self._data)
            if identities != _materialized_file_identities(config.resolved_path()):
                raise ValueError("FineVision materialization changed while loading")
        self._index = self._build_index()
        if config.strict_annotations:
            self.content_fingerprint = self._build_content_fingerprint()
        self._warned = 0

    def _build_index(self) -> Optional[np.ndarray]:
        """Row positions kept after the quality filters (``None`` when none are active)."""
        cfg = self.config
        active = {
            column: getattr(cfg, attr)
            for attr, column in _QUALITY_COLUMNS.items()
            if getattr(cfg, attr) is not None
        }
        if not active:
            return None

        keep = np.ones(len(self._data), dtype=bool)
        for column, threshold in active.items():
            if column not in self._data.column_names:
                if cfg.require_quality_columns:
                    raise ValueError(
                        f"FineVision[{cfg.config_name}] lacks required quality column {column!r}"
                    )
                log.warning("FineVision: no %r column; ignoring that filter", column)
                continue
            values = np.array(
                [-np.inf if v is None else float(v) for v in self._data[column]],
                dtype=np.float64,
            )
            keep &= values >= threshold
        index = np.nonzero(keep)[0]
        log.info(
            "FineVision[%s]: kept %d / %d rows after quality filtering (%s)",
            cfg.config_name,
            len(index),
            len(self._data),
            ", ".join(f"{c}>={t}" for c, t in active.items()),
        )
        return index

    def _build_content_fingerprint(self) -> str:
        """Build a stable identity for source rows, filtering, and serialization policy."""
        cfg = self.config
        arrow_fingerprint = (
            cfg.expected_materialized_fingerprint
            if cfg.expected_materialized_fingerprint is not None
            else getattr(self._data, "_fingerprint", None)
        )
        if not isinstance(arrow_fingerprint, str) or not arrow_fingerprint:
            raise ValueError(
                f"FineVision[{cfg.config_name}] lacks a stable Arrow dataset fingerprint"
            )
        selected = (
            np.arange(len(self._data), dtype="<i8")
            if self._index is None
            else np.asarray(self._index, dtype="<i8")
        )
        selection_sha256 = hashlib.sha256(selected.tobytes()).hexdigest()
        payload = {
            "version": self.content_fingerprint_version,
            "arrow_fingerprint": arrow_fingerprint,
            "config_name": cfg.config_name,
            "dataset_path": os.path.realpath(cfg.resolved_path()),
            "split": cfg.split,
            "texts_column": cfg.texts_column,
            "images_column": cfg.images_column,
            "quality_filters": {
                field_name: getattr(cfg, field_name) for field_name in _QUALITY_COLUMNS
            },
            "require_quality_columns": cfg.require_quality_columns,
            "strict_annotations": cfg.strict_annotations,
            "skip_bad_rows": cfg.skip_bad_rows,
            "max_crops": cfg.max_crops,
            "max_images": cfg.max_images,
            "max_sequence_length": cfg.max_sequence_length,
            "loss_token_weighting": cfg.loss_token_weighting,
            "message_format": cfg.message_format,
            "seed": cfg.seed,
            "selected_rows": len(selected),
            "selection_sha256": selection_sha256,
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def validate_required_annotations(self) -> None:
        """Validate the fail-closed single-image/single-turn annotation contract.

        The scan reads Arrow list lengths and text structs but does not decode image bytes.
        It is intentionally opt-in through :attr:`FineVisionDatasetConfig.strict_annotations`
        so other sources can retain their permissive row handling.

        :raises ValueError: If a selected row lacks exactly one image or one usable
            ``(user, assistant)`` turn. An image-only first prompt must explicitly contain
            an image marker and have a non-empty answer.
        """
        if not self.config.strict_annotations:
            return

        import pyarrow.compute as pc

        table = self._data.data
        try:
            images = table.column(self.config.images_column)
            image_lengths = pc.list_value_length(images)
            texts = table.column(self.config.texts_column)
        except (KeyError, ValueError) as error:
            raise ValueError("FineVision strict annotation columns are unavailable") from error

        positions = (
            range(len(self._data))
            if self._index is None
            else (int(position) for position in self._index)
        )
        invalid_count = 0
        first_invalid: List[int] = []
        for dataset_index, position in enumerate(positions):
            image_count = image_lengths[position].as_py()
            has_image = image_count == 1 and images[position].values[0].is_valid
            row_turns = texts[position].as_py()
            valid_turn = (
                isinstance(row_turns, list)
                and len(row_turns) == 1
                and _parse_turn(row_turns[0], first=True, has_image=has_image) is not None
            )
            if not has_image or not valid_turn:
                invalid_count += 1
                if len(first_invalid) < 8:
                    first_invalid.append(dataset_index)
        if invalid_count:
            raise ValueError(
                "FineVision strict mode requires exactly one image and one non-empty turn; "
                f"found {invalid_count} invalid selected rows (first indices: {first_invalid})"
            )

    def __len__(self) -> int:
        return len(self._data) if self._index is None else len(self._index)

    def _row(self, i: int) -> Dict:
        pos = int(i if self._index is None else self._index[i])
        return self._data[pos]

    def raw_image_references(self, index: int) -> Tuple[Any, ...]:
        """Return the encoded image cells for one logical row without decoding them.

        :param index: Logical filtered-dataset index.
        :returns: The row's exact image structs/paths in serialized order.
        """
        position = int(index if self._index is None else self._index[index])
        try:
            raw = self._data.data.column(self.config.images_column)[position].as_py()
        except (KeyError, ValueError) as error:
            raise ValueError("FineVision image column is unavailable") from error
        if not isinstance(raw, list):
            raise ValueError(f"FineVision image row {index} is not a list")
        return tuple(raw)

    def _build(self, i: int, epoch: int = 0) -> Dict[str, np.ndarray]:
        cfg = self.config
        row = self._row(i)
        raw_images = row.get(cfg.images_column) or []
        has_image = any(image is not None for image in raw_images)

        turns: List[Tuple[str, str]] = []
        for turn_index, turn in enumerate(row[cfg.texts_column] or []):
            parsed = _parse_turn(turn, first=turn_index == 0, has_image=has_image)
            if parsed is not None:
                turns.append(parsed)
        if not turns:
            raise ValueError("no usable (user, assistant) turn in row")

        pil_images = [decode_pil_image(im) for im in raw_images if im is not None]

        # One sequential conversation branch (turn k attends earlier turns).
        seq = encode_sft_example(
            self.tokenizer,
            pil_images,
            [turns],
            max_crops=cfg.max_crops,
            max_images=cfg.max_images,
            loss_token_weighting=cfg.loss_token_weighting,
            token_ids=cfg.token_ids,
            message_format=cfg.message_format,
            shuffle_rng=sft_example_rng(cfg.seed, i, epoch, cfg.message_format),
        )
        original_length = len(seq["input_ids"]) if cfg.strict_annotations else None
        example = truncate_example(
            seq,
            cfg.max_sequence_length,
            image_token_ids=cfg.token_ids.image_token_ids,
        )
        if cfg.strict_annotations:
            assert original_length is not None
            example["metadata"] = {
                **example.get("metadata", {}),
                "original_length": original_length,
                "truncated": original_length > cfg.max_sequence_length,
            }
        return example

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Build the example at ``index``, skipping ahead over unusable rows.

        See :func:`~olmo_core.data.multimodal.sft_common.get_example_with_skip`.
        """
        return self.get(index, 0)

    def get(self, index: int, epoch: int = 0) -> Dict[str, np.ndarray]:
        """Build one example with epoch-aware augmentation and deterministic bad-row skips."""
        if self.config.skip_bad_rows:
            return get_example_with_skip(self, index, len(self), epoch)
        return self._build(index, epoch)


# Backwards-compatible alias: the loader used to be VisualWebInstruct-specific.
VisualWebInstructDataset = FineVisionDataset
