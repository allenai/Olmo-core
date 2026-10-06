"""Shared building blocks for image + instruction (SFT) datasets read from parquet.

:mod:`~olmo_core.data.multimodal.mmfinereason` and
:mod:`~olmo_core.data.multimodal.finevision` both turn a HuggingFace row of
``(image(s), user prompt, assistant response)`` into a packed Molmo2 training example.
They differ only in *which columns hold the text* and *how the supervision target is
extracted*; everything shared lives here:

* :func:`load_hf_dataset` — open a parquet shard directory / glob / file, or a
  ``datasets.save_to_disk`` Arrow directory, as a map-style ``datasets.Dataset``.
* :func:`strip_image_placeholders` — drop the ``<image>`` markers that these corpora put
  in the prompt text (the image is instead supplied as an explicit token block inside the
  first user turn — see
  :func:`~olmo_core.data.multimodal.qwen3_layout.multi_image_prefix_ids`).
* :func:`truncate_example` — right-truncate a built example, refusing to cut image tokens
  or to drop every loss token.
* :func:`get_example_with_skip` — deterministic bad-row skipping for ``__getitem__``.
* :class:`EpochSeededExamples` — per-example RNG streams that rotate with the training epoch.

Sequence assembly itself goes through
:func:`~olmo_core.data.multimodal.message_sequence.encode_sft_example`, with a selectable
released-Qwen or native-document serializer. Multi-turn conversations remain one
sequential branch in either layout.
"""

from __future__ import annotations

import glob as _glob
import logging
import os
import re
from typing import (
    Any,
    Collection,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)

import numpy as np

from olmo_core.nn.vision.molmo2_tokens import IM_PATCH_ID

from .message_weight import ATTEND_ALL_SUBSEGMENT_ID
from .sequence_builder import example_rng

SftMessageFormat = Literal["qwen3", "document"]
SFT_MESSAGE_FORMATS: Tuple[SftMessageFormat, ...] = ("qwen3", "document")

__all__ = [
    "heldout_ids",
    "IMAGE_PLACEHOLDER",
    "MAX_ROW_SKIP",
    "EpochSeededExamples",
    "SFT_MESSAGE_FORMATS",
    "SftMessageFormat",
    "load_hf_dataset",
    "strip_image_placeholders",
    "count_image_placeholders",
    "decode_pil_image",
    "truncate_example",
    "truncate_for_format",
    "get_example_with_skip",
    "sft_example_rng",
    "validate_sft_message_format",
]

log = logging.getLogger(__name__)

MAX_ROW_SKIP = 32
"""How many following rows :func:`get_example_with_skip` tries before giving up."""


def heldout_ids(paths: Sequence[str], column: str) -> Set[str]:
    """The identifiers (image urls / hashes, row ids) a training source must not contain.

    Each path is a ``save_to_disk`` dataset. For a ``DatasetDict`` every split except ``train`` is
    held out; a flat ``Dataset`` (an eval set) is held out entirely.

    :param paths: Held-out dataset directories.
    :param column: The identifier column to read from each (``image_url`` / ``image_sha256`` /
        ``id``).

    :returns: The union of the held-out identifiers.

    :raises OLMoConfigurationError: If a path is missing or lacks ``column``. A guard that cannot
        be checked fails the build rather than passing silently; pass no paths to skip it on
        purpose.
    """
    import os

    from olmo_core.exceptions import OLMoConfigurationError

    from .dataset_compat import load_from_disk_compat

    ids: Set[str] = set()
    for path in paths:
        if not os.path.isdir(path):
            raise OLMoConfigurationError(
                f"held-out set {path!r} not found, so train/eval image overlap cannot be checked; "
                "fix the path, or pass heldout_paths=() to skip the check deliberately"
            )
        ds = load_from_disk_compat(path)
        splits = [ds[k] for k in ds.keys() if k != "train"] if hasattr(ds, "keys") else [ds]
        for split in splits:
            if column not in split.column_names:
                raise OLMoConfigurationError(
                    f"held-out set {path!r} has no {column!r} column (it has {split.column_names})"
                )
            ids.update(x for x in split.data.column(column).to_pylist() if x)
    return ids


class EpochSeededExamples:
    """Mixin for map-style datasets whose ``__getitem__`` *samples* something.

    A source that draws a subset per example — a few of an image's negatives, a render size —
    must advance its RNG stream with the training epoch, or it redraws the same subset forever
    and the remainder of each pool is never trained on. mm_olmo does this by folding the epoch
    into the per-example seed (``dataset.py:70-73``); the data loader calls :meth:`set_epoch`
    from ``reshuffle`` so the same happens here.

    Requires the host class to expose ``config.seed`` and ``__len__``. The epoch is restored
    with the loader's state, so a resumed run replays the epoch it was in.

    :meth:`get` instead takes the epoch per call, as the continuous-stream mixture loader
    passes each source's own epoch. It never touches the shared epoch, so concurrent loader
    threads cannot race on it. It requires the host class to define ``_build(index, epoch)``
    and skips unusable rows deterministically (:func:`get_example_with_skip`).
    """

    _epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to derive per-example RNG streams (called by the data loader)."""
        self._epoch = int(epoch)

    def resolve_epoch(self, epoch: Optional[int] = None) -> int:
        """``epoch`` when given, else the epoch last set by :meth:`set_epoch`."""
        return self._epoch if epoch is None else int(epoch)

    def epoch_rng(self, index: int, epoch: Optional[int] = None) -> np.random.RandomState:
        """This row's RNG stream for ``epoch`` (default: the current epoch)."""
        return example_rng(
            self.config.seed,  # type: ignore[attr-defined]
            index,
            epoch=self.resolve_epoch(epoch),
            dataset_len=len(self),  # type: ignore[arg-type]
        )

    def get(self, index: int, epoch: int = 0) -> Dict[str, Any]:
        """Build row ``index`` for source epoch ``epoch``, skipping unusable rows.

        :param index: Row index.
        :param epoch: Source epoch; it seeds the row's RNG stream in place of the shared epoch.
        """
        return get_example_with_skip(self, index, len(self), epoch)  # type: ignore[arg-type]


IMAGE_PLACEHOLDER = "<image>"
"""Inline marker used by these corpora to indicate where an image belongs in the prompt."""

_PLACEHOLDER_RE = re.compile(r"<image>\s*")


def validate_sft_message_format(message_format: str) -> SftMessageFormat:
    """Validate the released Molmo2 or native pretraining-document serializer."""
    if message_format not in SFT_MESSAGE_FORMATS:
        raise ValueError(
            f"Unknown message_format {message_format!r}; expected one of {SFT_MESSAGE_FORMATS}"
        )
    return cast(SftMessageFormat, message_format)


def sft_example_rng(
    seed: int,
    index: int,
    epoch: int,
    message_format: SftMessageFormat,
) -> np.random.RandomState:
    """Return the augmentation stream for a Stage 1 document or Stage 2 SFT example.

    Qwen examples retain the seeded per-example stream used by the Molmo2 adapters.
    Native document examples additionally vary their stream with the source epoch.

    :param seed: Dataset seed.
    :param index: Example index.
    :param epoch: Source epoch.
    :param message_format: Selected document or chat serializer.

    :returns: A deterministic NumPy random-state stream.
    """
    validate_sft_message_format(message_format)
    from .rng import make_random_state
    from .sequence_builder import example_rng

    if message_format == "document":
        return make_random_state(seed + index, epoch)
    return example_rng(seed, index)


def load_hf_dataset(path: str, split: str = "train", *, keep_columns: Optional[List[str]] = None):
    """Open a dataset for map-style (random-access) reading.

    Three layouts are accepted:

    * a directory of ``.parquet`` shards (as produced by ``hf download`` of a parquet
      dataset repo) — searched non-recursively, then recursively;
    * an explicit glob such as ``".../data/train-*.parquet"``, or a single ``.parquet`` file;
    * a ``datasets.save_to_disk`` Arrow directory (what the PixMo sources on weka use).

    .. note::
        Loading parquet builds a one-time memory-mapped Arrow cache under
        ``HF_DATASETS_CACHE``; for image corpora that is roughly the size of the parquet
        itself. Point ``path`` at a ``save_to_disk`` directory to avoid the extra copy.

    :param path: Directory, glob, or file as described above.
    :param split: Split to select when the source is a ``DatasetDict``.
    :param keep_columns: If given, drop every other column right after loading. This keeps
        row access cheap (notably: it avoids carrying long unused reasoning-trace columns).

    :returns: A ``datasets.Dataset``.

    :raises FileNotFoundError: If no parquet files match a glob / directory.
    """
    from datasets import load_dataset

    from .dataset_compat import load_from_disk_compat

    files: List[str] = []
    if any(ch in path for ch in "*?["):
        files = sorted(_glob.glob(path))
    elif path.endswith(".parquet"):
        files = [path]
    elif os.path.isdir(path):
        files = sorted(_glob.glob(os.path.join(path, "*.parquet")))
        if not files:
            files = sorted(_glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))

    if files:
        ds = load_dataset("parquet", data_files=files, split=split)
    else:
        if any(ch in path for ch in "*?[") or path.endswith(".parquet"):
            raise FileNotFoundError(f"No parquet files matched {path!r}")
        loaded = load_from_disk_compat(path)
        ds = loaded[split] if hasattr(loaded, "keys") and split in loaded else loaded

    if keep_columns is not None:
        extra = [c for c in ds.column_names if c not in keep_columns]
        if extra:
            ds = ds.remove_columns(extra)
    return ds


def strip_image_placeholders(text: Optional[str]) -> str:
    """Remove ``<image>`` markers (and the whitespace that follows them) from ``text``.

    The image is presented to the model as an explicit token block *before* the prompt
    text, so the inline marker would otherwise be tokenized as literal text. Markers that
    sit mid-sentence are simply dropped, which reorders the image to the front of the turn.

    :param text: Raw prompt text (``None`` is treated as empty).

    :returns: The text with every marker removed, stripped of surrounding whitespace.
    """
    if not text:
        return ""
    return _PLACEHOLDER_RE.sub("", text).strip()


def count_image_placeholders(text: Optional[str]) -> int:
    """Return how many ``<image>`` markers appear in ``text``."""
    return (text or "").count(IMAGE_PLACEHOLDER)


def decode_pil_image(obj: Any):
    """Coerce a HuggingFace image cell into a PIL image.

    Handles an already-decoded ``PIL.Image``, the ``{"bytes": ..., "path": ...}`` struct that
    parquet round-trips when the ``Image`` feature is not decoded, and a plain path/URL str.

    :param obj: The image cell.

    :returns: A ``PIL.Image.Image``.

    :raises TypeError: If the cell is not one of the supported forms.
    :raises ValueError: If an image struct carries neither bytes nor a path.
    """
    import io

    from PIL import Image

    if isinstance(obj, Image.Image):
        return obj
    if isinstance(obj, dict):
        data = obj.get("bytes")
        if data:
            return Image.open(io.BytesIO(data))
        path = obj.get("path")
        if path:
            return Image.open(path)
        raise ValueError("Image struct has neither 'bytes' nor 'path'")
    if isinstance(obj, str):
        return Image.open(obj)
    raise TypeError(f"Unsupported image cell type: {type(obj)}")


def truncate_for_format(
    seq: Dict[str, Any],
    max_len: Optional[int],
    token_ids: Molmo2TokenIds,
    message_format: str,
) -> Dict[str, Any]:
    """:func:`truncate_example` for a source's selected tokenizer and serializer.

    The released Qwen layout protects only ``<im_patch>`` (the stage-1 sources' existing
    behaviour); the document layout protects every structural image token.

    :param seq: A built example.
    :param max_len: Maximum token count; ``None`` leaves the example unchanged.
    :param token_ids: The source's image token IDs.
    :param message_format: ``"qwen3"`` or ``"document"``.
    """
    if max_len is None:
        return seq
    return truncate_example(
        seq,
        max_len,
        image_patch_token_id=token_ids.im_patch_id,
        image_token_ids=token_ids.image_token_ids if message_format == "document" else None,
    )


def truncate_example(
    seq: Dict[str, Any],
    max_len: int,
    *,
    image_patch_token_id: int = IM_PATCH_ID,
    image_token_ids: Optional[Collection[int]] = None,
    recompute_root_subsegments: bool = False,
) -> Dict[str, Any]:
    """Right-truncate every per-token field of ``seq`` to ``max_len``.

    :param seq: A built example (as returned by the sequence builders).
    :param max_len: Maximum token count.
    :param image_patch_token_id: ``<im_patch>`` ID for the selected language model.
    :param image_token_ids: All structural image-token IDs. When supplied, truncation
        rejects any image structure in the discarded suffix, matching the vendor collator.
    :param recompute_root_subsegments: Drop a trailing branch whose response would be
        entirely truncated, then rescale surviving branch weights as mm_olmo does.

    :returns: The example with per-token fields truncated (other fields untouched).

    :raises ValueError: If truncation would drop image-structural tokens or every loss token.
    """
    if len(seq["input_ids"]) <= max_len:
        return seq
    original_branch_count = 0
    if recompute_root_subsegments and "subsegment_ids" in seq:
        subsegment_ids = seq["subsegment_ids"]
        original_branch_ids = [
            int(branch_id)
            for branch_id in np.unique(subsegment_ids)
            if branch_id != ATTEND_ALL_SUBSEGMENT_ID
        ]
        original_branch_count = len(original_branch_ids)
        # Vendor preprocessing stops before a branch whose first supervised token would
        # fall outside the context. Avoid retaining that branch's prompt without its answer.
        for branch_id in original_branch_ids:
            branch_positions = np.flatnonzero(subsegment_ids == branch_id)
            if branch_positions.size == 0 or branch_positions[0] >= max_len:
                continue
            retained_positions = branch_positions[branch_positions < max_len]
            if not np.any(seq["loss_masks"][retained_positions] > 0):
                max_len = min(max_len, int(branch_positions[0]))
                break
    protected_ids = image_token_ids or (image_patch_token_id,)
    if np.any(np.isin(seq["input_ids"][max_len:], list(protected_ids))):
        raise ValueError(
            "max_sequence_length too small: truncation would drop <im_patch> or other "
            "image-structural tokens (the complete image block must fit within the sequence)."
        )
    if not np.any(seq["loss_masks"][:max_len] > 0):
        raise ValueError(f"Truncation to {max_len} removed all loss tokens")
    n = len(seq["input_ids"])
    out = {
        k: (v[:max_len] if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == n else v)
        for k, v in seq.items()
    }
    if original_branch_count > 1 and "subsegment_ids" in out:
        surviving_branch_ids = np.unique(out["subsegment_ids"][out["loss_masks"] > 0])
        surviving_branch_ids = surviving_branch_ids[
            surviving_branch_ids != ATTEND_ALL_SUBSEGMENT_ID
        ]
        if 0 < len(surviving_branch_ids) < original_branch_count:
            out["loss_masks"] = out["loss_masks"] * np.sqrt(
                original_branch_count / len(surviving_branch_ids)
            )
    return out


def get_example_with_skip(
    dataset,
    index: int,
    size: int,
    epoch: int | None = None,
) -> Dict[str, np.ndarray]:
    """Build ``dataset._build(index)``, deterministically skipping over unusable rows.

    A corrupt image or a row whose supervision text parses to nothing must never raise out
    of ``__getitem__``: that kills the data worker and, under distributed packing, hangs the
    other ranks' collectives into a NCCL watchdog abort. Up to :data:`MAX_ROW_SKIP`
    subsequent rows are tried instead (wrapping at the end), which keeps the substitution
    deterministic and therefore identical across ranks and resumes.

    The first few failures per dataset instance are logged (tracked on ``dataset._warned``)
    so a systematically broken source stays visible without flooding the log.

    :param dataset: Object exposing ``_build(i)`` and a mutable ``_warned`` counter.
    :param index: Requested row.
    :param size: Number of rows in the dataset.
    :param epoch: Optional source epoch. When set, call ``dataset._build(i, epoch)``;
        otherwise preserve the single-argument dataset interface.

    :returns: The built example.

    :raises RuntimeError: If :data:`MAX_ROW_SKIP` consecutive rows all failed.
    """
    last: Optional[Exception] = None
    for step in range(MAX_ROW_SKIP):
        i = (index + step) % max(size, 1)
        try:
            return dataset._build(i) if epoch is None else dataset._build(i, epoch)
        except Exception as exc:  # noqa: BLE001 - never crash training on one bad row
            last = exc
            if getattr(dataset, "_warned", 0) < 5:
                dataset._warned = getattr(dataset, "_warned", 0) + 1
                log.warning(
                    "%s: skipping row %d (%s: %s)",
                    type(dataset).__name__,
                    i,
                    type(exc).__name__,
                    exc,
                )
    raise RuntimeError(
        f"{type(dataset).__name__}: {MAX_ROW_SKIP} consecutive rows from {index} were unusable"
    ) from last
