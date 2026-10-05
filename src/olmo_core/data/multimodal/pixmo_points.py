"""PixMo pointing / counting + CoSyn pointing datasets for Molmo2 stage-1.

Ports mm_olmo's pointing data sources (``olmo/data/pixmo_datasets.py``):

* :class:`PixMoPointsDataset` — ``points-pointing`` / ``points-counting`` (or both);
  each row has several ``(label, points)`` annotations → a multi-branch example, each
  branch a ``pointing`` or ``point_count`` Q/A over the shared image.
* :class:`PixMoCountDataset` — ``count``; single-annotation, alternating ``point_count``
  / ``pointing`` style; points are pixel-space (normalized by image size).
* :class:`CoSynPointDataset` — ``cosyn-point``; each row has several ``(question, points,
  name)`` annotations → multi-branch pointing.

All answers use the html-v2 grounding format (see :mod:`.grounding`). Sequences are
assembled with :func:`~olmo_core.data.multimodal.sequence_builder.build_branched_sequence`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional

import numpy as np

from olmo_core.config import Config
from olmo_core.nn.vision.molmo2_tokens import (
    N_PATCHES_SQ,
    PATCH_DIM,
    POOL_H,
    POOL_W,
    Molmo2TokenIds,
)

from .grounding import normalize_points, pointing_answer
from .message_sequence import encode_sft_example
from .qwen3_layout import branch_context_ids, image_prefix_ids
from .sequence_builder import build_branched_sequence
from .sft_common import (
    SftMessageFormat,
    sft_example_rng,
    truncate_example,
    validate_sft_message_format,
)
from .sft_formatter import SftFormatter

__all__ = [
    "CoSynPointDataset",
    "COSYN_POINT_STYLE",
    "COSYN_POINT_V2_PATH",
    "FAILED_AUDIT_RESULTS",
    "CoSynPointDatasetConfig",
    "PixMoCountDataset",
    "PixMoCountDatasetConfig",
    "PixMoPointsDataset",
    "PixMoPointsDatasetConfig",
    "select_annotation",
]

from olmo_core.exceptions import OLMoConfigurationError

from .paths import PIXMO_DATASETS

#: ``audit_result`` values mm_olmo treats as a failed audit (``PixMoPointV2._keep``,
#: ``CoSynPointConfigV2.FAILED``).
FAILED_AUDIT_RESULTS = frozenset({"error", "clear_error"})
_CONTENT_FINGERPRINT_VERSION = "pixmo-perception-adapter-v1"
_CONTENT_FINGERPRINT_DOMAIN = b"pixmo-perception-adapter-v1\0"
_SCALAR_COUNT_PROMPT = "How many {label} are there?"
_TOKEN_FIELDS = ("input_ids", "labels", "loss_masks", "position_ids", "token_type_ids")
_PERCENT_POINT_CLAMP_TOLERANCE = 2.0
_ANNOTATION_SAMPLINGS = ("all", "one")
_ANNOTATION_STREAM = 0x0A11  # spawn-key salt keeping the selection off the augmentation stream

AnnotationSampling = Literal["all", "one"]


def _validate_annotation_sampling(value: str) -> None:
    if value not in _ANNOTATION_SAMPLINGS:
        raise ValueError(
            f"annotation_sampling must be one of {_ANNOTATION_SAMPLINGS}, got {value!r}"
        )


def select_annotation(seed: int, index: int, epoch: int, count: int) -> int:
    """
    Pick one of an example's ``count`` annotations for ``annotation_sampling="one"``.

    The choice is deterministic per ``(seed, index, epoch)`` and drawn from its own stream, so
    it does not perturb the example's augmentation stream and successive epochs can pick
    different annotations of the same image.

    :param seed: The dataset seed.
    :param index: The example index.
    :param epoch: The source epoch.
    :param count: The number of annotations to choose from (positive).

    :returns: The selected annotation's position, in ``[0, count)``.
    """
    from .rng import make_random_state

    if count <= 0:
        raise ValueError("count must be positive")
    if count == 1:
        return 0
    return int(make_random_state(seed, index, epoch, _ANNOTATION_STREAM).randint(count))


def _explicit_grounding_prompt(prompt: str, *, counting: bool = False) -> str:
    instruction = "Return point coordinates for every match"
    instruction += " and give the total count." if counting else "."
    return f"{prompt}\n{instruction}"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _require_arrow_fingerprint(dataset: Any, *, source_name: str, split: str) -> dict[str, Any]:
    """Return the stable identity of one selected Arrow source split."""
    fingerprint = getattr(dataset, "_fingerprint", None)
    if callable(fingerprint):
        fingerprint = fingerprint()
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(
            f"{source_name} split {split!r} does not expose a stable Arrow fingerprint"
        )
    return {
        "arrow_fingerprint": fingerprint,
        "num_rows": len(dataset),
        "source_name": source_name,
        "split": split,
    }


def _index_sha256(index: Sequence[tuple[int, list[int]]]) -> str:
    digest = hashlib.sha256()
    for row_index, label_indices in index:
        digest.update(
            _canonical_bytes(
                {"label_indices": [int(i) for i in label_indices], "row_index": int(row_index)}
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _adapter_fingerprint(
    adapter_name: str,
    config: Config,
    source_descriptors: Sequence[Mapping[str, Any]],
    *,
    derived_index_sha256: str | None = None,
) -> str:
    fingerprint_config = asdict(config)
    # The v1 identity omits default-valued prompt-family controls.
    for key, default in (
        ("prompt_templates", "uber_model_v2"),
        ("system_prompt", "demo_or_style_v2"),
        ("annotation_sampling", "all"),
    ):
        if fingerprint_config.get(key) == default:
            fingerprint_config.pop(key)
    payload: dict[str, Any] = {
        "adapter": adapter_name,
        "config": fingerprint_config,
        "sources": list(source_descriptors),
        "version": _CONTENT_FINGERPRINT_VERSION,
    }
    if derived_index_sha256 is not None:
        payload["derived_index_sha256"] = derived_index_sha256
    return hashlib.sha256(_CONTENT_FINGERPRINT_DOMAIN + _canonical_bytes(payload)).hexdigest()


def _available_columns(dataset: Any) -> set[str] | None:
    columns = getattr(dataset, "column_names", None)
    if columns is None:
        return None
    return {str(column) for column in columns}


def _annotation_rows(dataset: Any, required_columns: Sequence[str]):
    """Iterate annotations without decoding the source's image column."""
    columns = _available_columns(dataset)
    if columns is not None:
        missing = sorted(set(required_columns) - columns)
        if missing:
            raise ValueError(f"Dataset lacks required annotation columns: {missing}")
    selected = dataset
    select_columns = getattr(dataset, "select_columns", None)
    if callable(select_columns):
        selected = select_columns(list(required_columns))
    for index in range(len(selected)):
        row = selected[index]
        if not isinstance(row, Mapping):
            raise TypeError(f"Annotation row {index} must be a mapping, got {type(row)}")
        missing = [column for column in required_columns if column not in row]
        if missing:
            raise ValueError(f"Annotation row {index} lacks required columns: {missing}")
        yield index, row


def _require_columns(dataset: Any, required_columns: Sequence[str]) -> None:
    columns = _available_columns(dataset)
    if columns is None:
        return
    missing = sorted(set(required_columns) - columns)
    if missing:
        raise ValueError(f"Dataset lacks required columns: {missing}")


def _require_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    return value


def _require_nonnegative_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field_name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return value


def _require_sequence(value: Any, *, field_name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence")
    return value


def _require_percent_point(point: Any, *, field_name: str) -> None:
    if not isinstance(point, Mapping) or "x" not in point or "y" not in point:
        raise ValueError(f"{field_name} must contain x/y coordinates")
    try:
        x, y = float(point["x"]), float(point["y"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} coordinates must be numeric") from error
    low, high = -_PERCENT_POINT_CLAMP_TOLERANCE, 100.0 + _PERCENT_POINT_CLAMP_TOLERANCE
    if not np.isfinite([x, y]).all() or not (low <= x <= high and low <= y <= high):
        raise ValueError(
            f"{field_name} coordinates must be finite percentages within the preprocessing "
            f"clamp tolerance [{low:g}, {high:g}]"
        )


def _require_xy_mapping(
    points: Any,
    *,
    field_name: str,
    percent_coordinates: bool,
) -> np.ndarray:
    if not isinstance(points, Mapping) or "x" not in points or "y" not in points:
        raise ValueError(f"{field_name} must contain x/y coordinate arrays")
    xs = _require_sequence(points["x"], field_name=f"{field_name}.x")
    ys = _require_sequence(points["y"], field_name=f"{field_name}.y")
    if len(xs) != len(ys):
        raise ValueError(f"{field_name}.x and {field_name}.y must have the same length")
    try:
        xy = np.asarray([xs, ys], dtype=np.float64).T.reshape(-1, 2)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} coordinates must be numeric") from error
    if not np.isfinite(xy).all():
        raise ValueError(f"{field_name} coordinates must be finite")
    if percent_coordinates and xy.size and np.any((xy < 0.0) | (xy > 100.0)):
        raise ValueError(f"{field_name} coordinates must be percentages in [0, 100]")
    return xy


def _validate_dataset_rows(dataset_name: str, rows, validate_row: Callable[[Any], None]) -> None:
    invalid_count = 0
    first_errors: list[str] = []
    row_count = 0
    for row_index, row in rows:
        row_count += 1
        try:
            validate_row(row)
        except (TypeError, ValueError) as error:
            invalid_count += 1
            if len(first_errors) < 8:
                first_errors.append(f"{row_index}: {error}")
    if invalid_count:
        raise ValueError(
            f"{dataset_name} has {invalid_count} invalid annotation rows out of {row_count}; "
            f"first errors: {first_errors}"
        )


def _validate_serialized_example(
    example: dict[str, np.ndarray],
    *,
    token_ids: Molmo2TokenIds,
    max_sequence_length: int | None,
    max_output_crops: int,
) -> None:
    """Fail before packing when a serialized image example violates model geometry."""
    missing = sorted((set(_TOKEN_FIELDS) | {"images", "pooled_patches_idx"}) - set(example))
    if missing:
        raise ValueError(f"Serialized example lacks required fields: {missing}")
    n_tokens: int | None = None
    for field_name in _TOKEN_FIELDS:
        value = example[field_name]
        if not isinstance(value, np.ndarray) or value.ndim != 1:
            raise ValueError(f"{field_name} must be a rank-1 NumPy array")
        if n_tokens is None:
            n_tokens = len(value)
        elif len(value) != n_tokens:
            raise ValueError("All serialized token fields must have identical lengths")
    assert n_tokens is not None
    if n_tokens <= 0:
        raise ValueError("Serialized examples must contain at least one token")
    if max_sequence_length is not None and n_tokens > max_sequence_length:
        raise ValueError(
            f"Serialized example has {n_tokens} tokens, exceeding {max_sequence_length}"
        )
    for field_name in ("input_ids", "labels", "position_ids", "token_type_ids"):
        if not np.issubdtype(example[field_name].dtype, np.integer):
            raise ValueError(f"{field_name} must have an integer dtype")
    loss_masks = example["loss_masks"]
    if not np.issubdtype(loss_masks.dtype, np.floating):
        raise ValueError("loss_masks must have a floating dtype")
    if not np.isfinite(loss_masks).all() or np.any(loss_masks < 0):
        raise ValueError("loss_masks must be finite and nonnegative")
    if not np.any(loss_masks > 0):
        raise ValueError("Serialized example contains no supervised tokens")
    if np.any(example["position_ids"] < 0):
        raise ValueError("position_ids must be nonnegative")
    expected_token_types = np.isin(
        example["input_ids"], np.fromiter(token_ids.image_token_ids, dtype=np.int64)
    ).astype(np.int64)
    if not np.array_equal(example["token_type_ids"], expected_token_types):
        raise ValueError("token_type_ids do not exactly mark the configured image tokens")
    if "subsegment_ids" in example:
        subsegments = example["subsegment_ids"]
        if (
            not isinstance(subsegments, np.ndarray)
            or subsegments.ndim != 1
            or len(subsegments) != n_tokens
            or not np.issubdtype(subsegments.dtype, np.integer)
        ):
            raise ValueError("subsegment_ids must be a rank-1 integer array matching tokens")

    images = example["images"]
    if (
        not isinstance(images, np.ndarray)
        or images.dtype != np.float32
        or images.ndim != 3
        or images.shape[1:] != (N_PATCHES_SQ, PATCH_DIM)
    ):
        raise ValueError(f"images must be float32 with shape (crops, {N_PATCHES_SQ}, {PATCH_DIM})")
    if not 1 <= images.shape[0] <= max_output_crops:
        raise ValueError(
            f"images has {images.shape[0]} crops; expected between 1 and {max_output_crops}"
        )
    if not np.isfinite(images).all():
        raise ValueError("images must contain only finite values")

    pooled = example["pooled_patches_idx"]
    if (
        not isinstance(pooled, np.ndarray)
        or pooled.dtype != np.int64
        or pooled.ndim != 2
        or pooled.shape[1] != POOL_H * POOL_W
    ):
        raise ValueError(
            f"pooled_patches_idx must be int64 with shape (pooled_tokens, {POOL_H * POOL_W})"
        )
    valid = pooled >= 0
    if pooled.shape[0] == 0 or not np.all(valid.any(axis=1)):
        raise ValueError("Every pooled-patch row must contain at least one valid patch index")
    if np.any(pooled < -1) or np.any(pooled[valid] >= images.shape[0] * N_PATCHES_SQ):
        raise ValueError("pooled_patches_idx contains an out-of-range patch index")
    n_image_tokens = int(np.count_nonzero(example["input_ids"] == token_ids.im_patch_id))
    if n_image_tokens != pooled.shape[0]:
        raise ValueError(
            f"Serialized example has {n_image_tokens} <im_patch> tokens but "
            f"{pooled.shape[0]} pooled rows"
        )


def _finalize_example(
    example: dict[str, np.ndarray],
    *,
    strict_validation: bool,
    max_sequence_length: int | None,
    max_crops: int,
    high_res_max_crops: int,
    p_high_res: float,
    loss_token_weighting: str,
    token_ids: Molmo2TokenIds,
) -> dict[str, np.ndarray]:
    if not strict_validation:
        return example
    original_length = len(example["input_ids"])
    if max_sequence_length is not None:
        example = truncate_example(
            example,
            max_sequence_length,
            image_patch_token_id=token_ids.im_patch_id,
            image_token_ids=token_ids.image_token_ids,
            recompute_root_subsegments=loss_token_weighting
            in ("root_subsegments", "root_subsegments_root_tokens"),
        )
    effective_high_res = max_crops if p_high_res <= 0 else max(max_crops, high_res_max_crops)
    # The processor returns one global crop in addition to at most ``max_crops`` tiles.
    _validate_serialized_example(
        example,
        token_ids=token_ids,
        max_sequence_length=max_sequence_length,
        max_output_crops=effective_high_res + 1,
    )
    example["metadata"] = {
        **example.get("metadata", {}),
        "original_length": original_length,
        "truncated": max_sequence_length is not None and original_length > max_sequence_length,
    }
    return example


def _build_example(
    tokenizer,
    pil_image,
    branches: Callable[[np.random.RandomState], list[tuple[str, str]]] | Sequence[tuple[str, str]],
    *,
    max_crops: int,
    loss_token_weighting: str,
    strict_validation: bool = False,
    high_res_max_crops: int = 24,
    max_sequence_length: int | None = None,
    token_ids: Molmo2TokenIds | None = None,
    message_weight: float | None = None,
    p_high_res: float = 0.0,
    message_format: SftMessageFormat = "qwen3",
    rng: np.random.RandomState | None = None,
    shuffle_rng: np.random.RandomState | None = None,
    seed: int = 0,
    branch_weights: Optional[Sequence[Optional[float]]] = None,
) -> Dict[str, np.ndarray]:
    """Preprocess the image and assemble a (possibly multi-branch) pointing example.

    Two calling conventions are supported, and they consume the example's RNG stream in
    different orders, so each keeps the exact sequences its callers were built against:

    * ``branches`` is a **list** of ``(user_question, assistant_answer)`` strings (the Molmo2
      stage-1 sources, e.g. :mod:`.pixmo_points_v2`): the branches are shuffled first and the
      image is augmented second, on the qwen3 layout, exactly as mm_olmo does.
    * ``branches`` is a **callable** ``rng -> [(question, answer), ...]`` (the alignment
      adapters in this module): the branches are built, then the example is serialized with
      :func:`~olmo_core.data.multimodal.message_sequence.encode_sft_example` in the requested
      ``message_format`` and validated by :func:`_finalize_example`.

    :param branch_weights: optional per-branch loss multipliers parallel to the branch list
        (``None`` entries mean 1). mm_olmo's per-message ``AssistantMessage.weight``: the
        branch's response tokens are scaled by it on top of ``loss_token_weighting`` and
        ``message_weight``, which stay example-wide. Only the list convention supports it.
    :param rng: The example's RNG stream for the callable convention. ``shuffle_rng`` is its
        name in the list convention; ``seed`` seeds a fresh stream when neither is given.
    """
    if rng is None:
        rng = shuffle_rng if shuffle_rng is not None else np.random.RandomState(seed)
    resolved_token_ids = token_ids if token_ids is not None else Molmo2TokenIds()

    if callable(branches):
        if branch_weights is not None:
            raise ValueError("branch_weights are not supported with a branch-building callable")
        branches_text = list(branches(rng))
        example = encode_sft_example(
            tokenizer,
            pil_image,
            branches_text,
            max_crops=max_crops,
            high_res_max_crops=high_res_max_crops,
            p_high_res=p_high_res,
            loss_token_weighting=loss_token_weighting,
            token_ids=resolved_token_ids,
            message_format=message_format,
            message_weight=message_weight,
            shuffle_rng=rng,
        )
        return _finalize_example(
            example,
            strict_validation=strict_validation,
            max_sequence_length=max_sequence_length,
            max_crops=max_crops,
            high_res_max_crops=high_res_max_crops,
            p_high_res=p_high_res,
            loss_token_weighting=loss_token_weighting,
            token_ids=resolved_token_ids,
        )

    if message_format != "qwen3":
        raise ValueError("The branch-list convention serializes the qwen3 layout only")
    import torch

    from olmo_core.nn.vision.molmo2_image_processor import preprocess_image_molmo2

    branches_text = list(branches)
    weights = None if branch_weights is None else list(branch_weights)
    if weights is not None and len(weights) != len(branches_text):
        raise ValueError(
            f"branch_weights has {len(weights)} entries for {len(branches_text)} branches"
        )
    if len(branches_text) > 1:
        order = np.arange(len(branches_text))
        rng.shuffle(order)
        branches_text = [branches_text[i] for i in order]
        if weights is not None:
            weights = [weights[i] for i in order]

    images_t, pooling_t, image_grid = preprocess_image_molmo2(
        pil_image,
        dtype=torch.float32,
        device=torch.device("cpu"),
        max_crops=max_crops,
        p_high_res=p_high_res,
        is_training=True,
        rng=rng,
    )
    prefix = image_prefix_ids(tokenizer, image_grid)
    multi_branch = len(branches_text) > 1
    encoded_branches = [
        (
            branch_context_ids(tokenizer, q, branch_index=i, multi_branch=multi_branch),
            tokenizer.encode(a, add_special_tokens=False),
        )
        for i, (q, a) in enumerate(branches_text)
    ]
    from olmo_core.data.multimodal.message_weight import (
        MessageWeight,
        apply_message_weight_to_loss_masks,
    )

    seq = build_branched_sequence(
        prefix,
        encoded_branches,
        eos_id=tokenizer.eos_token_id,
        loss_token_weighting=loss_token_weighting,
    )
    subsegment_ids = seq.get("subsegment_ids")
    mw = MessageWeight.from_string(loss_token_weighting).with_overrides(message_weight)
    seq["loss_masks"] = apply_message_weight_to_loss_masks(
        seq["loss_masks"], subsegment_ids, mw, branch_scaling_already_applied=True
    )
    if weights is not None:
        seq["loss_masks"] = _apply_branch_weights(seq["loss_masks"], subsegment_ids, weights)
    seq["images"] = images_t[0].numpy()
    seq["pooled_patches_idx"] = pooling_t[0].numpy()
    return _finalize_example(
        seq,
        strict_validation=strict_validation,
        max_sequence_length=max_sequence_length,
        max_crops=max_crops,
        high_res_max_crops=high_res_max_crops,
        p_high_res=p_high_res,
        loss_token_weighting=loss_token_weighting,
        token_ids=resolved_token_ids,
    )


def _apply_branch_weights(
    loss_masks: np.ndarray,
    subsegment_ids: Optional[np.ndarray],
    weights: Sequence[Optional[float]],
) -> np.ndarray:
    """Scale each branch's loss weights by its multiplier (``None`` / 1 leave it alone).

    ``loss_masks`` is aligned with ``labels`` (shifted one position left), but every position
    that carries loss for branch ``b`` -- the token before each of its response tokens, and its
    segment-end token -- itself belongs to ``b``, so selecting by ``subsegment_ids`` is exact.
    A single-branch example has no ``subsegment_ids``; its one weight applies to the whole mask.
    """
    out = loss_masks.astype(np.float32, copy=True)
    if subsegment_ids is None:
        w = weights[0]
        if w is not None and w != 1.0:
            out *= float(w)
        return out
    for branch_idx, w in enumerate(weights):
        if w is not None and w != 1.0:
            out[subsegment_ids == branch_idx] *= float(w)
    return out


def _load_split(path: str, split: str, *, require_split: bool = False):
    from .dataset_compat import load_from_disk_compat

    ds = load_from_disk_compat(path)
    if hasattr(ds, "keys") and split in ds:
        return ds[split]
    if require_split:
        raise ValueError(f"Dataset {path!r} lacks required split {split!r}")
    return ds


def _open_image(p):
    from PIL import Image

    return p if isinstance(p, Image.Image) else Image.open(p)


# ---------------------------------------------------------------------------
# PixMo points (pointing / counting)
# ---------------------------------------------------------------------------


@dataclass
class PixMoPointsDatasetConfig(Config):
    """Configure PixMo's multi-annotation pointing/counting source.

    ``kind="basic"`` selects ``points-pointing`` and ``kind="high_frequency"`` selects
    ``points-counting``. ``kind="both"`` concatenates counting before pointing. When
    ``max_sequence_length`` is set, examples are safely truncated and validated before
    they can reach packing or collation.
    """

    split: str = "train"
    require_split: bool = False
    """Require ``split`` to exist in the saved Arrow ``DatasetDict``."""
    kind: str = "both"  # "basic" (points-pointing) | "high_frequency" (points-counting) | "both"
    counting: str | bool = "both"  # "both" randomly selects; bool fixes one style
    both_mode: Literal["per_annotation", "duplicate"] = "per_annotation"
    """Sample one style per annotation, or expose both styles as separate examples."""
    annotation_sampling: AnnotationSampling = "all"
    """``"all"`` packs every annotation of an image as sibling branches of one example;
    ``"one"`` keeps a single annotation per example, chosen per ``(seed, index, epoch)``.
    Language models that can only isolate whole documents (no attention masks, e.g. Kimi Delta
    Attention) cannot train on sibling branches and need ``"one"``."""
    explicit_grounding_prompts: bool = False
    """Explicitly request point coordinates, including in count-style prompts."""
    max_points: int = 60
    max_total_points_per_example: int = 60
    max_crops: int = 8
    high_res_max_crops: int = 24
    max_sequence_length: int | None = None
    loss_token_weighting: str = "root_subsegments"
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)
    message_weight: float | None = None
    p_high_res: float = 0.0
    message_format: SftMessageFormat = "qwen3"
    seed: int = 0
    prompt_templates: str = "uber_model_v2"
    """Question template family; ``none`` uses the bare label."""
    system_prompt: str = "demo_or_style_v2"
    """Style-prefix family; ``style_and_length_v2`` prefixes pointing styles."""

    def build(self, tokenizer) -> PixMoPointsDataset:
        return PixMoPointsDataset(self, tokenizer)


class PixMoPointsDataset:
    """Map-style PixMo points adapter with audited annotations and stable identity."""

    content_fingerprint_version = _CONTENT_FINGERPRINT_VERSION

    def __init__(self, config: PixMoPointsDatasetConfig, tokenizer):
        if config.kind not in ("basic", "high_frequency", "both"):
            raise ValueError(f"Unknown PixMo points kind {config.kind!r}")
        if config.counting not in (False, True, "both"):
            raise ValueError(f"Unknown PixMo points counting mode {config.counting!r}")
        if config.both_mode not in ("per_annotation", "duplicate"):
            raise ValueError(f"Unknown PixMo points both_mode {config.both_mode!r}")
        _validate_annotation_sampling(config.annotation_sampling)
        if not isinstance(config.split, str) or not config.split:
            raise ValueError("PixMo points split must be a non-empty string")
        if config.max_points < 0 or config.max_total_points_per_example <= 0:
            raise ValueError("PixMo point limits must be nonnegative with a positive total")
        if config.max_crops <= 0 or config.high_res_max_crops <= 0:
            raise ValueError("PixMo crop limits must be positive")
        if not 0.0 <= config.p_high_res <= 1.0:
            raise ValueError("p_high_res must be in [0, 1]")
        if config.max_sequence_length is not None and config.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive when set")
        validate_sft_message_format(config.message_format)
        self.config = config
        self.tokenizer = tokenizer
        sub = {
            "basic": ["points-pointing"],
            "high_frequency": ["points-counting"],
            "both": ["points-counting", "points-pointing"],
        }[config.kind]
        from datasets import concatenate_datasets

        self._sources = [
            _load_split(
                f"{PIXMO_DATASETS}/{source_name}",
                config.split,
                require_split=config.require_split,
            )
            for source_name in sub
        ]
        self._data = concatenate_datasets(self._sources)
        # Pre-split each row's labels into sub-batches with <= max_total_points (mm_olmo).
        self._index = self._build_sub_index()
        if config.require_split:
            source_descriptors = [
                _require_arrow_fingerprint(source, source_name=name, split=config.split)
                for name, source in zip(sub, self._sources)
            ]
            self.content_fingerprint = _adapter_fingerprint(
                type(self).__name__,
                config,
                source_descriptors,
                derived_index_sha256=_index_sha256(self._index),
            )
        self._annotations_validated = False

    def _build_sub_index(self) -> list[tuple[int, list[int]]]:
        cfg = self.config
        counts = self._data["count"]
        labels = self._data["label"] if cfg.require_split else None
        index: list[tuple[int, list[int]]] = []
        skipped_blank_labels = 0
        skipped_over_max_points = 0
        for row, point_counts in enumerate(counts):
            row_labels = labels[row] if labels is not None else None
            if row_labels is not None and len(point_counts) != len(row_labels):
                raise ValueError(
                    f"PixMo points row {row} has {len(point_counts)} counts but "
                    f"{len(row_labels)} labels"
                )
            on: list[int] = []
            total = 0
            for li, n in enumerate(point_counts):
                if row_labels is not None and (
                    not isinstance(row_labels[li], str) or not row_labels[li].strip()
                ):
                    skipped_blank_labels += 1
                    continue
                if n > cfg.max_points:
                    skipped_over_max_points += 1
                    continue
                if on and total + n > cfg.max_total_points_per_example:
                    index.append((row, on))
                    on, total = [], 0
                on.append(li)
                total += n
            if on:
                index.append((row, on))
        self.annotation_filter_stats = {
            "blank_labels": skipped_blank_labels,
            "over_max_points": skipped_over_max_points,
        }
        return index

    def __len__(self) -> int:
        size = len(self._index)
        if self.config.counting == "both" and self.config.both_mode == "duplicate":
            return size * 2
        return size

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        return self.get(i, 0)

    def raw_image_references(self, index: int) -> tuple[Any, ...]:
        """Return the source image reference for one expanded logical example."""
        example_index = (
            index // 2
            if self.config.counting == "both" and self.config.both_mode == "duplicate"
            else index
        )
        row_index, _ = self._index[example_index]
        return (self._data[row_index]["image"],)

    def validate_required_annotations(self) -> None:
        """Exhaustively validate the non-image annotations used by this adapter.

        Images are deliberately not decoded during this scan. Their serialized crop and
        pooling geometry is checked on every built example before packing.

        :raises ValueError: If any row has missing, malformed, or out-of-range annotations.
        """
        if self._annotations_validated:
            return
        _require_columns(
            self._data,
            ("image", "label", "points", "count", "collection_method"),
        )

        retained_by_row: dict[int, set[int]] = {}
        for row_index, label_indices in self._index:
            retained_by_row.setdefault(row_index, set()).update(label_indices)

        def retained_rows():
            rows = _annotation_rows(
                self._data,
                ("label", "points", "count", "collection_method"),
            )
            for row_index, row in rows:
                retained = sorted(retained_by_row.get(row_index, ()))
                if not retained:
                    continue
                yield row_index, {
                    field_name: [row[field_name][index] for index in retained]
                    for field_name in ("label", "points", "count", "collection_method")
                }

        def validate_row(row: Mapping[str, Any]) -> None:
            labels = _require_sequence(row["label"], field_name="label")
            point_groups = _require_sequence(row["points"], field_name="points")
            counts = _require_sequence(row["count"], field_name="count")
            methods = _require_sequence(row["collection_method"], field_name="collection_method")
            sizes = {len(labels), len(point_groups), len(counts), len(methods)}
            if len(sizes) != 1 or not labels:
                raise ValueError(
                    "label, points, count, and collection_method must be equally sized and nonempty"
                )
            for annotation_index, (label, points, count, method) in enumerate(
                zip(labels, point_groups, counts, methods)
            ):
                prefix = f"annotation {annotation_index}"
                _require_text(label, field_name=f"{prefix}.label")
                count = _require_nonnegative_integer(count, field_name=f"{prefix}.count")
                points = _require_sequence(points, field_name=f"{prefix}.points")
                if len(points) != count:
                    raise ValueError(
                        f"{prefix}.count={count} does not match {len(points)} point annotations"
                    )
                method = _require_text(method, field_name=f"{prefix}.collection_method")
                if method not in ("pointing", "counting"):
                    raise ValueError(f"{prefix}.collection_method has unknown value {method!r}")
                for point_index, point in enumerate(points):
                    _require_percent_point(
                        point,
                        field_name=f"{prefix}.points[{point_index}]",
                    )

        _validate_dataset_rows(
            "PixMo points",
            retained_rows(),
            validate_row,
        )
        self._annotations_validated = True

    def get(self, i: int, epoch: int = 0) -> dict[str, np.ndarray]:
        """Build one deterministically augmented example for a source epoch."""
        fixed_style = None
        if self.config.counting == "both" and self.config.both_mode == "duplicate":
            example_idx = i // 2
            fixed_style = "point_count" if i % 2 == 0 else "pointing"
        else:
            example_idx = i
        row_idx, label_idxs = self._index[example_idx]
        rng = sft_example_rng(self.config.seed, i, epoch, self.config.message_format)
        row = self._data[row_idx]
        fmt = SftFormatter(
            seed=self.config.seed,
            prompt_templates=self.config.prompt_templates,
            system_prompt=self.config.system_prompt,
        )
        specs: list[tuple[str, str, Any]] = []
        for li in label_idxs:
            label = row["label"][li]
            pts = row["points"][li]
            if fixed_style is not None:
                style = fixed_style
            elif self.config.counting == "both":
                style = rng.choice(["point_count", "pointing"])
            else:
                style = "point_count" if self.config.counting else "pointing"
            specs.append((style, label, pts))
        if self.config.annotation_sampling == "one" and len(specs) > 1:
            specs = [specs[select_annotation(self.config.seed, i, epoch, len(specs))]]

        def build_branches(branch_rng: np.random.RandomState) -> list[tuple[str, str]]:
            branches: list[tuple[str, str]] = []
            for branch_style, label, points in specs:
                sub = {
                    "style": branch_style,
                    "label": label,
                    "points": points,
                    "point_scale": 100,
                }
                prompt, answer = fmt.format_turns(sub, index=i, rng=branch_rng)[0]
                if self.config.explicit_grounding_prompts:
                    prompt = _explicit_grounding_prompt(
                        prompt, counting=branch_style == "point_count"
                    )
                branches.append((prompt, answer))
            return branches

        return _build_example(
            self.tokenizer,
            _open_image(row["image"]),
            build_branches,
            max_crops=self.config.max_crops,
            strict_validation=self.config.require_split,
            high_res_max_crops=self.config.high_res_max_crops,
            max_sequence_length=self.config.max_sequence_length,
            loss_token_weighting=self.config.loss_token_weighting,
            token_ids=self.config.token_ids,
            message_weight=self.config.message_weight,
            p_high_res=self.config.p_high_res,
            message_format=self.config.message_format,
            rng=rng,
        )


# ---------------------------------------------------------------------------
# PixMo count (single annotation, alternating point_count / pointing)
# ---------------------------------------------------------------------------


@dataclass
class PixMoCountDatasetConfig(Config):
    """Configure PixMo Count grounding or scalar-count document continuations."""

    dataset_path: str = f"{PIXMO_DATASETS}/count"
    """Arrow ``DatasetDict`` containing the requested count split."""
    split: str = "train"
    require_split: bool = False
    """Require ``split`` to exist in the saved Arrow ``DatasetDict``."""
    mode: Literal["grounded", "scalar_count"] = "grounded"
    """``scalar_count`` always supervises the declared integer using document layout."""
    counting: str | bool = "both"  # "both" interleaves point_count (even) / pointing (odd)
    explicit_grounding_prompts: bool = False
    """Explicitly request coordinates for grounded answers; scalar prompts are unchanged."""
    scalar_count_replay: bool = False
    """Replay the scalar integer interface once per grounded raw row.

    With ``counting="both"`` the scalar branch is attached only to the deterministic
    ``point_count`` variant, so duplicated grounding styles do not silently double its dose.
    A single-style grounded dataset attaches the branch to its sole variant.
    """
    max_crops: int = 8
    high_res_max_crops: int = 24
    max_sequence_length: int | None = None
    loss_token_weighting: str = "root_subsegments"
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)
    message_weight: float | None = None
    p_high_res: float = 0.0
    message_format: SftMessageFormat = "qwen3"
    seed: int = 0
    prompt_templates: str = "uber_model_v2"
    """Question template family; ``none`` uses the bare label."""
    system_prompt: str = "demo_or_style_v2"
    """Style-prefix family; ``style_and_length_v2`` prefixes pointing styles."""

    def build(self, tokenizer) -> PixMoCountDataset:
        return PixMoCountDataset(self, tokenizer)


class PixMoCountDataset:
    """Map-style PixMo Count adapter with an explicit scalar-count mode."""

    content_fingerprint_version = _CONTENT_FINGERPRINT_VERSION

    def __init__(self, config: PixMoCountDatasetConfig, tokenizer):
        if config.mode not in ("grounded", "scalar_count"):
            raise ValueError(f"Unknown PixMo Count mode {config.mode!r}")
        if config.counting not in (False, True, "both"):
            raise ValueError(f"Unknown PixMo Count counting mode {config.counting!r}")
        if config.mode == "scalar_count":
            if config.message_format != "document":
                raise ValueError("PixMo scalar_count mode requires message_format='document'")
            if config.counting != "both":
                raise ValueError(
                    "PixMo scalar_count mode does not use grounded counting styles; leave "
                    "counting='both'"
                )
            if config.scalar_count_replay:
                raise ValueError("scalar_count_replay is redundant in scalar_count mode")
        if not isinstance(config.split, str) or not config.split:
            raise ValueError("PixMo Count split must be a non-empty string")
        if config.max_crops <= 0 or config.high_res_max_crops <= 0:
            raise ValueError("PixMo crop limits must be positive")
        if not 0.0 <= config.p_high_res <= 1.0:
            raise ValueError("p_high_res must be in [0, 1]")
        if config.max_sequence_length is not None and config.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive when set")
        validate_sft_message_format(config.message_format)
        self.config = config
        self.tokenizer = tokenizer
        self._data = _load_split(
            config.dataset_path,
            config.split,
            require_split=config.require_split,
        )
        self._n = len(self._data)
        if config.require_split:
            descriptor = _require_arrow_fingerprint(
                self._data,
                source_name="count",
                split=config.split,
            )
            self.content_fingerprint = _adapter_fingerprint(
                type(self).__name__,
                config,
                [descriptor],
            )
        self._annotations_validated = False

    def __len__(self) -> int:
        if self.config.mode == "scalar_count":
            return self._n
        return self._n * 2 if self.config.counting == "both" else self._n

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        return self.get(i, 0)

    def raw_image_references(self, index: int) -> tuple[Any, ...]:
        """Return the source image reference for one scalar or grounded logical example."""
        row_index = (
            index // 2
            if self.config.mode == "grounded" and self.config.counting == "both"
            else index
        )
        return (self._data[row_index]["image"],)

    def validate_required_annotations(self) -> None:
        """Validate every declared count, label, and optional grounding coordinate array.

        The scalar-count mode intentionally treats ``count`` as authoritative on every
        split; validation rows may therefore carry empty point arrays for positive counts.

        :raises ValueError: If a required annotation is absent or malformed.
        """
        if self._annotations_validated:
            return
        _require_columns(self._data, ("image", "label", "count", "points"))

        def validate_row(row: Mapping[str, Any]) -> None:
            _require_text(row["label"], field_name="label")
            _require_nonnegative_integer(row["count"], field_name="count")
            _require_xy_mapping(
                row["points"],
                field_name="points",
                percent_coordinates=False,
            )

        _validate_dataset_rows(
            "PixMo Count",
            _annotation_rows(self._data, ("label", "count", "points")),
            validate_row,
        )
        self._annotations_validated = True

    def get(self, i: int, epoch: int = 0) -> dict[str, np.ndarray]:
        """Build one deterministically augmented example for a source epoch."""
        if self.config.mode == "scalar_count":
            row_idx, style = i, "scalar_count"
        elif self.config.counting == "both":
            row_idx, style = i // 2, ("point_count" if i % 2 == 0 else "pointing")
        else:
            row_idx, style = i, ("point_count" if self.config.counting else "pointing")
        row = self._data[row_idx]
        label = row["label"]
        count = int(row["count"])
        pil = _open_image(row["image"])
        pts = row.get("points") or {"x": [], "y": []}
        rng = sft_example_rng(self.config.seed, i, epoch, self.config.message_format)
        fmt = SftFormatter(
            seed=self.config.seed,
            prompt_templates=self.config.prompt_templates,
            system_prompt=self.config.system_prompt,
        )
        if self.config.require_split:
            xy = _require_xy_mapping(
                pts,
                field_name=f"row {row_idx}.points",
                percent_coordinates=False,
            )
            if style != "scalar_count" and xy.size:
                width, height = pil.size
                if width <= 0 or height <= 0:
                    raise ValueError(f"row {row_idx} image has invalid size {pil.size!r}")
                if np.any(xy[:, 0] < 0) or np.any(xy[:, 0] > width):
                    raise ValueError(f"row {row_idx} contains an x coordinate outside the image")
                if np.any(xy[:, 1] < 0) or np.any(xy[:, 1] > height):
                    raise ValueError(f"row {row_idx} contains a y coordinate outside the image")
        else:
            xy = np.array([pts["x"], pts["y"]], dtype=np.float64).T.reshape(-1, 2)
        sub = {
            "style": style,
            "label": label,
            "points": xy,
            "point_scale": None,
            "image_size": pil.size,
        }

        def build_branches(branch_rng: np.random.RandomState) -> list[tuple[str, str]]:
            scalar_branch = (_SCALAR_COUNT_PROMPT.format(label=label), str(count))
            if style == "scalar_count":
                return [scalar_branch]
            # PixMo Count validation/test retain the declared count but omit point
            # annotations. Those rows support count-only evaluation, not grounding.
            if len(xy) == 0 and count > 0:
                return [scalar_branch]
            prompt, answer = fmt.format_turns(sub, index=i, rng=branch_rng)[0]
            if self.config.explicit_grounding_prompts:
                prompt = _explicit_grounding_prompt(prompt, counting=style == "point_count")
            branches = [(prompt, answer)]
            replay_on_this_variant = self.config.counting != "both" or style == "point_count"
            if self.config.scalar_count_replay and replay_on_this_variant:
                branches.append(scalar_branch)
            return branches

        return _build_example(
            self.tokenizer,
            pil,
            build_branches,
            max_crops=self.config.max_crops,
            strict_validation=self.config.require_split,
            high_res_max_crops=self.config.high_res_max_crops,
            max_sequence_length=self.config.max_sequence_length,
            loss_token_weighting=self.config.loss_token_weighting,
            token_ids=self.config.token_ids,
            message_weight=self.config.message_weight,
            p_high_res=self.config.p_high_res,
            message_format=self.config.message_format,
            rng=rng,
        )


# ---------------------------------------------------------------------------
# CoSyn point (document pointing; multi-branch, prompt = the question)
# ---------------------------------------------------------------------------

#: The style CoSyn pointing is tagged with, as in mm_olmo. Its question is an English request stored
#: in the data (e.g. "Highlight the period that shows the largest five-year increase..."), not an
#: object name: the ``names`` column is a short summary written for the answer's label and drops
#: most of what the request asks. A tag of its own keeps ``pointing:`` meaning "an object name
#: follows".
COSYN_POINT_STYLE = "cosyn_point"

#: mm_olmo's audited CoSyn build (``CoSynPointConfigV2``): the v1 build's images, questions,
#: points and names unchanged (all 68,051 train rows match), plus a per-question ``audit_result``
#: from a VLM audit (81.7% ``correct``, 17.3% ``clear_error``, 1.0% ``error`` on a 1-in-20
#: sample) and agent masks. The masks feed segmentation messages, which this repo does not
#: train, so they are ignored.
COSYN_POINT_V2_PATH = f"{PIXMO_DATASETS}/cosyn-point-v2-masks"


@dataclass
class CoSynPointDatasetConfig(Config):
    """Configure CoSyn pointing with an explicit Arrow split and sequence bound."""

    dataset_path: str = f"{PIXMO_DATASETS}/cosyn-point"
    """HF dataset with ``image``, ``questions``, ``answer_points`` and ``names`` columns."""

    split: str = "train"
    require_split: bool = False
    """Require ``split`` to exist in the saved Arrow ``DatasetDict``."""
    explicit_grounding_prompts: bool = False
    """Explicitly request point coordinates after each source question."""
    max_crops: int = 8
    high_res_max_crops: int = 24
    max_sequence_length: int | None = None
    loss_token_weighting: str = "root_subsegments"
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)
    message_weight: float | None = None
    p_high_res: float = 0.0
    message_format: SftMessageFormat = "qwen3"
    seed: int = 0
    prompt_templates: str = "uber_model_v2"
    """Unused: the question is always the one stored in the data. Kept so the source takes the
    same kwargs as the other pointing sources."""
    system_prompt: str = "demo_or_style_v2"
    """Prompt family for the style prefix; stage 1 uses ``"style_and_length_v2"``, which prefixes
    the question with ``"cosyn_point:"`` (:data:`COSYN_POINT_STYLE`)."""
    audit_style: Optional[str] = None
    """Style for the questions that failed the VLM audit (:data:`FAILED_AUDIT_RESULTS`), e.g.
    ``"aux_cosyn_point"``: they are kept, behind a tag of their own, so the model learns them apart
    from the questions that passed. Needs the audited build (:data:`COSYN_POINT_V2_PATH`). ``None``
    treats every question alike, which is all the v1 build allows."""
    annotation_sampling: AnnotationSampling = "all"
    """``"all"`` packs every question of an image as sibling branches of one example; ``"one"``
    keeps a single question per example, chosen per ``(seed, index, epoch)`` (needed by language
    models that can only isolate whole documents, e.g. Kimi Delta Attention)."""

    def build(self, tokenizer) -> CoSynPointDataset:
        return CoSynPointDataset(self, tokenizer)


class CoSynPointDataset:
    """Map-style CoSyn pointing adapter with audited percent-coordinate geometry."""

    content_fingerprint_version = _CONTENT_FINGERPRINT_VERSION

    def __init__(self, config: CoSynPointDatasetConfig, tokenizer):
        if not isinstance(config.split, str) or not config.split:
            raise ValueError("CoSyn Point split must be a non-empty string")
        if config.max_crops <= 0 or config.high_res_max_crops <= 0:
            raise ValueError("CoSyn crop limits must be positive")
        _validate_annotation_sampling(config.annotation_sampling)
        if not 0.0 <= config.p_high_res <= 1.0:
            raise ValueError("p_high_res must be in [0, 1]")
        if config.max_sequence_length is not None and config.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive when set")
        validate_sft_message_format(config.message_format)
        self.config = config
        self.tokenizer = tokenizer
        self._data = _load_split(
            config.dataset_path, config.split, require_split=config.require_split
        )
        if config.audit_style is not None and "audit_result" not in self._data.column_names:
            raise OLMoConfigurationError(
                f"audit_style={config.audit_style!r} needs an audited CoSyn build with an "
                f"`audit_result` column, such as {COSYN_POINT_V2_PATH!r}; {config.dataset_path!r} "
                f"has {self._data.column_names}"
            )
        if config.require_split:
            descriptor = _require_arrow_fingerprint(
                self._data,
                source_name="cosyn-point",
                split=config.split,
            )
            self.content_fingerprint = _adapter_fingerprint(
                type(self).__name__,
                config,
                [descriptor],
            )
        self._annotations_validated = False

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        return self.get(i, 0)

    def raw_image_references(self, index: int) -> tuple[Any, ...]:
        """Return the source image reference for one CoSyn logical example."""
        return (self._data[index]["image"],)

    def validate_required_annotations(self) -> None:
        """Validate every question, answer point, and object name without decoding images.

        :raises ValueError: If branches are missing, misaligned, or outside percent geometry.
        """
        if self._annotations_validated:
            return
        _require_columns(self._data, ("image", "questions", "answer_points", "names"))

        def validate_row(row: Mapping[str, Any]) -> None:
            questions = _require_sequence(row["questions"], field_name="questions")
            answer_points = _require_sequence(row["answer_points"], field_name="answer_points")
            names = _require_sequence(row["names"], field_name="names")
            sizes = {len(questions), len(answer_points), len(names)}
            if len(sizes) != 1 or not questions:
                raise ValueError(
                    "questions, answer_points, and names must be equally sized and nonempty"
                )
            for branch_index, (question, points, name) in enumerate(
                zip(questions, answer_points, names)
            ):
                prefix = f"branch {branch_index}"
                _require_text(question, field_name=f"{prefix}.question")
                _require_text(name, field_name=f"{prefix}.name")
                xy = _require_xy_mapping(
                    points,
                    field_name=f"{prefix}.answer_points",
                    percent_coordinates=True,
                )
                if len(xy) == 0:
                    raise ValueError(f"{prefix}.answer_points must be nonempty")

        _validate_dataset_rows(
            "CoSyn Point",
            _annotation_rows(self._data, ("questions", "answer_points", "names")),
            validate_row,
        )
        self._annotations_validated = True

    def get(self, i: int, epoch: int = 0) -> dict[str, np.ndarray]:
        """Build one deterministically augmented example for a source epoch."""
        row = self._data[i]
        cfg = self.config
        if cfg.require_split:
            questions = _require_sequence(row["questions"], field_name=f"row {i}.questions")
            answer_points = _require_sequence(
                row["answer_points"], field_name=f"row {i}.answer_points"
            )
            names = _require_sequence(row["names"], field_name=f"row {i}.names")
            if len({len(questions), len(answer_points), len(names)}) != 1 or not questions:
                raise ValueError(
                    f"row {i} questions, answer_points, and names must be equally sized and "
                    "nonempty"
                )
        else:
            questions = row["questions"]
            answer_points = row["answer_points"]
            names = row["names"]
        # The prefix follows the checkpoint's family exactly as for the PixMo sources.
        fmt = SftFormatter(
            seed=cfg.seed, prompt_templates=cfg.prompt_templates, system_prompt=cfg.system_prompt
        )
        prefix = fmt.style_prefix(COSYN_POINT_STYLE)
        failed_prefix = fmt.style_prefix(cfg.audit_style) if cfg.audit_style else prefix
        audits = row["audit_result"] if cfg.audit_style else [None] * len(questions)
        branches: list[tuple[str, str]] = []
        for branch_index, (question, points, name, audit) in enumerate(
            zip(questions, answer_points, names, audits)
        ):
            if cfg.require_split:
                question = _require_text(question, field_name=f"row {i}.question[{branch_index}]")
                name = _require_text(name, field_name=f"row {i}.name[{branch_index}]")
                xy = _require_xy_mapping(
                    points,
                    field_name=f"row {i}.answer_points[{branch_index}]",
                    percent_coordinates=True,
                )
                if len(xy) == 0:
                    raise ValueError(f"row {i}.answer_points[{branch_index}] must be nonempty")
            else:
                xy = np.array([points["x"], points["y"]], dtype=np.float64).T.reshape(-1, 2)
            norm = normalize_points(xy, point_scale=100, image_size=None)
            # cosyn_point uses the "pointing" answer (just the points tag), label = name.
            answer = pointing_answer(norm, name.lower(), "pointing", count=len(norm))
            if cfg.explicit_grounding_prompts:
                question = _explicit_grounding_prompt(question)
            tag = failed_prefix if audit in FAILED_AUDIT_RESULTS else prefix
            branches.append((f"{tag} {question}" if tag else question, answer))
        if cfg.annotation_sampling == "one" and len(branches) > 1:
            branches = [branches[select_annotation(cfg.seed, i, epoch, len(branches))]]
        return _build_example(
            self.tokenizer,
            _open_image(row["image"]),
            lambda branch_rng: branches,
            max_crops=self.config.max_crops,
            strict_validation=self.config.require_split,
            high_res_max_crops=self.config.high_res_max_crops,
            max_sequence_length=self.config.max_sequence_length,
            loss_token_weighting=self.config.loss_token_weighting,
            token_ids=self.config.token_ids,
            message_weight=self.config.message_weight,
            p_high_res=self.config.p_high_res,
            message_format=self.config.message_format,
            rng=sft_example_rng(self.config.seed, i, epoch, self.config.message_format),
        )
