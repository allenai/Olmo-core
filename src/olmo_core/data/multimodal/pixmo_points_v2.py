"""Audited PixMo pointing / counting sources for Molmo2 stage-1.

Ports mm_olmo's second-generation pointing data (``olmo/data/pixmo_datasets.py``
``PixMoPointV2`` and ``PixMoCountConfigV2``), the sources its molmo3 stage-1 mixture uses in
place of ``pixmo_points_train`` / ``pixmo_points_high_freq_train`` / ``pixmo_count_train``
(``launch_scripts/train_molmo3_stage1.py``, ``_base_mixture``). Compared to the v1 builds in
:mod:`.pixmo_points`, every row is one *image* carrying all of its annotations, and:

* every point set has an ``audit_result`` -- a VLM re-evaluation: ``correct`` / ``unsure`` /
  ``error`` / ``clear_error`` (``n/a`` for empty sets). A failed audit (``error`` or
  ``clear_error``) can either be dropped (``filter_audit=True``) or trained on behind a marker
  style (``audit_style``, e.g. ``aux_pointing``) that differs from the base style only in its
  ``"<style>:"`` token, so those less reliable targets do not dilute the primary
  ``pointing`` / ``point_count`` distributions. Both work, per the mm_olmo team.
* :class:`PixMoPointsV2Dataset` rows also ship absence queries for training the
  ``"There are none."`` refusal: ``easy_negatives`` (unrelated labels) and
  ``paired_negatives`` / ``paired_negatives_v2`` (hard, near-miss labels for the image's own
  objects). These are meant to be *sub-sampled* per epoch (``n_easy_samples``,
  ``p_paired_negatives``); training on all of them makes the model over-refuse. The draw is
  seeded per (row, epoch) via :class:`~.sft_common.EpochSeededExamples`, so successive epochs
  work through different negatives instead of re-showing one fixed subset.

Both datasets format through :class:`~.sft_formatter.SftFormatter` and the html-v2 grounding
format like their v1 siblings, and assemble with the shared ``_build_example`` (image
preprocessing + branched sequence). Not ported from ``PixMoPointV2``: the segmentation-mask
branches (``include_masks`` / ``pixmo_seg``) and the 3D / surface-normal renderings, none of
which Molmo2 can emit.

Two consequences of dropping the mask branches, both measured against mm_olmo's ``_base_mixture``
(which leaves ``include_masks=True``):

* **Row count is all but unchanged.** mm_olmo's ``_keep`` also admits an annotation through the
  mask path, so its filter is the looser one, but on the v17 build that is 220,314 rows against
  our 220,291 (+23, +0.010%) -- the mask path only rescues annotations with more than
  ``max_points`` points. Under ``pointing_data="v2"`` the group splits *linearly* by row count,
  so this shifts the split within the group by a hundredth of a percent.
* **Every refusal here is a pointing refusal.** mm_olmo sends a negative to the segmentation
  head with probability ``1 - zero_points_as_segmentation`` (0.5 by default), and that applies to
  all of them -- annotated absences, easy and paired negatives alike. With no segmentation path
  to send them to, this port renders all of them as ``"There are none."`` pointing answers, so at
  equal ``n_easy_samples`` / ``p_paired_negatives`` the model sees roughly **twice** mm_olmo's
  density of pointing refusals. Halve those knobs to match mm_olmo's, and read the mm_olmo team's
  over-refusal warning above with that factor in mind.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds

from .paths import PIXMO_DATASETS, PIXMO_POINTS_V2
from .pixmo_points import (
    FAILED_AUDIT_RESULTS,
    AnnotationSampling,
    _build_example,
    _load_split,
    _open_image,
    _validate_annotation_sampling,
    select_annotation,
)
from .sft_common import (
    EpochSeededExamples,
    SftMessageFormat,
    heldout_ids,
    truncate_for_format,
)
from .sft_formatter import SftFormatter

__all__ = [
    "STAGE1_PROMPT_FAMILY",
    "POINTS_V2_HELDOUT_PATHS",
    "COUNT_V2_HELDOUT_PATHS",
    "FAILED_AUDIT_RESULTS",
    "PixMoPointsV2DatasetConfig",
    "PixMoPointsV2Dataset",
    "PixMoCountV2DatasetConfig",
    "PixMoCountV2Dataset",
]

log = logging.getLogger(__name__)

#: The prompt family these sources are formatted with, fixed rather than configurable: the
#: question is the bare lowercased label behind a ``"<style>:"`` prefix, as recorded in the
#: released ``Molmo2-4B-Pretrain`` ``data_formatter`` (``prompt_templates: none``,
#: ``system_prompt: style_and_length_v2``). mm_olmo's molmo3 stage 1 (``style_and_length_v3``)
#: renders pointing styles identically. These are stage-1 sources, so no other family is
#: reachable from here.
STAGE1_PROMPT_FAMILY = {"prompt_templates": "none", "system_prompt": "style_and_length_v2"}

#: Held-out sets the v2 points source must share no image with: the validation splits of the
#: two v1 PixMo-Points builds it was regrouped from, and the pointing eval set. See
#: :func:`~.sft_common.heldout_ids` for how a path is read.
POINTS_V2_HELDOUT_PATHS: Tuple[str, ...] = (
    f"{PIXMO_DATASETS}/points-pointing",
    f"{PIXMO_DATASETS}/points-counting",
    f"{PIXMO_DATASETS}/pixmo-points-eval",
)
#: Held-out sets for the v2 count source: PixMo-Count's validation and test splits. The
#: ``validation`` / ``test`` splits shipped *inside* ``count-v2`` are copies of its train split,
#: not held-out data, so they are neither read nor usable as a guard.
COUNT_V2_HELDOUT_PATHS: Tuple[str, ...] = (f"{PIXMO_DATASETS}/count",)


# mm_olmo ``PixMoPointV2.kind`` -> the ``source`` column value it keeps.
_KIND_TO_SOURCE = {"basic": "pointing", "high_frequency": "counting", "both": None}


def _failed_audit(audit_result: Optional[str]) -> bool:
    return audit_result in FAILED_AUDIT_RESULTS


def _choose(rng: np.random.RandomState, options: Sequence[str]) -> str:
    """``rng.choice`` over a style tuple, as a plain ``str``."""
    return str(rng.choice(list(options)))


def _rows_with_any(column, keep_fn) -> np.ndarray:
    """Per-row ``any(keep_fn(annotation))`` over a list-of-struct Arrow column.

    Works chunk by chunk on the raw Arrow buffers so that, unlike materialising the column
    with ``dataset["annotations"]``, the (large) mask RLE strings are never decoded: the
    full 224k-row PixMo-Points scan takes ~0.1 s. ``keep_fn`` receives the flattened
    struct array of one chunk and returns a boolean numpy mask over it.
    """
    import pyarrow.compute as pc

    out: List[np.ndarray] = []
    for chunk in column.chunks:
        flat = pc.list_flatten(chunk)
        keep = np.asarray(keep_fn(flat), dtype=bool)
        parents = pc.list_parent_indices(chunk).to_numpy(zero_copy_only=False)
        row_any = np.zeros(len(chunk), dtype=bool)
        if len(parents):
            row_any[np.unique(parents[keep])] = True
        out.append(row_any)
    return np.concatenate(out) if out else np.zeros(0, dtype=bool)


def _not_heldout(table, column: str, heldout: Set[str]) -> np.ndarray:
    """Per-row mask that is False where ``column``'s value is a held-out identifier."""
    import pyarrow as pa
    import pyarrow.compute as pc

    if not heldout:
        return np.ones(table.num_rows, dtype=bool)
    return ~_bool_np(pc.is_in(table.column(column), value_set=pa.array(sorted(heldout))))


def _bool_np(arr) -> np.ndarray:
    """Arrow boolean array -> numpy bool, nulls as False."""
    import pyarrow.compute as pc

    return pc.fill_null(arr, False).to_numpy(zero_copy_only=False).astype(bool)


# ---------------------------------------------------------------------------
# PixMo points v2: audited, image-grouped pointing + counting with absence queries
# ---------------------------------------------------------------------------


@dataclass
class PixMoPointsV2DatasetConfig(Config):
    """mm_olmo ``PixMoPointV2`` (its pointing leg). Field names follow mm_olmo's.

    Rows are images; each carries several ``(label, points, audit_result)`` annotations plus
    per-image negative labels. Every annotation kept by :meth:`PixMoPointsV2Dataset.keep`
    becomes one branch of a multi-branch example; annotated empty point sets and the
    sub-sampled negatives become ``"There are none."`` branches.
    """

    dataset_path: str = PIXMO_POINTS_V2
    """A flat HF ``Dataset`` saved with ``save_to_disk`` (columns ``image``, ``image_url``,
    ``source``, ``annotations``, ``easy_negatives``, ``paired_negatives``,
    ``paired_negatives_v2``). The build has no splits: it is the regrouped *train* rows of the
    two v1 PixMo-Points sources."""

    heldout_paths: Tuple[str, ...] = POINTS_V2_HELDOUT_PATHS
    """Held-out sets (see :func:`~.sft_common.heldout_ids`); a row whose ``image_url`` appears in
    one is never trained on. The build overlaps them by a single image today, but it is rebuilt
    independently of them, so this is checked on every build."""

    kind: str = "both"
    """``"basic"`` keeps the PixMo-Points rows (``source == "pointing"``), ``"high_frequency"``
    the PixMo-Count-style rows (``"counting"``), ``"both"`` everything."""

    style: Tuple[str, ...] = ("point_count", "pointing")
    """Styles drawn uniformly per annotation and per negative (mm_olmo ``style``)."""

    max_points: int = 60
    """Annotations with more points are skipped (they are also the least reliable)."""

    min_points: int = 0
    """Annotations with fewer points are skipped; ``0`` keeps the annotated empty sets, which
    render as ``"There are none."``."""

    filter_audit: bool = False
    """Drop annotations whose ``audit_result`` is ``error`` / ``clear_error``."""

    audit_style: Optional[Tuple[str, ...]] = None
    """Styles drawn for audit-failed annotations that are kept, e.g.
    ``("aux_point_count", "aux_pointing")``. ``None`` renders them like everything else."""

    n_easy_samples: int = 2
    """Easy negatives sampled per image and epoch (mm_olmo samples a fixed 2 regardless of
    this field's value; we honour the field, which is identical at the default)."""

    n_hard_negatives: int = 0
    """Negatives sampled from ``easy_negatives + paired_negatives`` (the v1 pool); mm_olmo's
    stage 1 leaves this at 0."""

    p_paired_negatives: float = 0.0
    """Expected fraction of the image's paired (hard) negatives used per epoch; the count is
    the stochastically-rounded ``len * p``. mm_olmo's stage 1 uses 0.25."""

    v2_paired_negatives: bool = True
    """Sample from ``paired_negatives_v2`` rather than ``paired_negatives``."""

    negative_weight: Optional[float] = None
    """Loss multiplier for the paired and annotated negatives. Easy negatives always weigh 1
    (mm_olmo ``PixMoPointV2.format_example``)."""

    max_crops: int = 8
    loss_token_weighting: str = "none"
    """``"none"`` weights every response token equally, as the released Molmo2 pretrain does for
    its pointing sources (``loss_token_weighting: None``)."""
    message_weight: Optional[float] = None
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)
    """Image token IDs of the selected language-model tokenizer (set by the mixture)."""
    message_format: SftMessageFormat = "qwen3"
    """``"qwen3"`` (the released Molmo2 chat layout) or ``"document"`` (plain pretraining
    documents, for a language model trained without a chat template)."""
    annotation_sampling: AnnotationSampling = "all"
    """``"all"`` trains every branch of an image (its kept annotations and sampled negatives) as
    sibling branches of one example; ``"one"`` keeps a single branch per example, chosen per
    ``(seed, index, epoch)`` (for language models that isolate packed examples only by document
    boundaries, e.g. Kimi Delta Attention)."""
    max_sequence_length: Optional[int] = None
    """Tail-truncate the built sequence to this many tokens. Applies outside the default
    ``qwen3`` / ``"all"`` path, which keeps the released stage-1 example exactly."""
    seed: int = 0

    def validate(self):
        if self.kind not in _KIND_TO_SOURCE:
            raise OLMoConfigurationError(
                f"kind={self.kind!r} is not one of {tuple(_KIND_TO_SOURCE)}"
            )
        if not self.style:
            raise OLMoConfigurationError("style must name at least one style")
        if self.audit_style is not None and not self.audit_style:
            raise OLMoConfigurationError("audit_style must be None or name at least one style")
        if not 0.0 <= self.p_paired_negatives <= 1.0:
            raise OLMoConfigurationError("p_paired_negatives must be in [0, 1]")
        if self.min_points < 0 or self.max_points < self.min_points:
            raise OLMoConfigurationError("need 0 <= min_points <= max_points")
        if self.n_easy_samples < 0 or self.n_hard_negatives < 0:
            raise OLMoConfigurationError("n_easy_samples / n_hard_negatives must be >= 0")
        _validate_annotation_sampling(self.annotation_sampling)

    def build(self, tokenizer) -> "PixMoPointsV2Dataset":
        self.validate()
        return PixMoPointsV2Dataset(self, tokenizer)


def _released_layout(cfg: Any) -> bool:
    """Whether a v2 source builds the released stage-1 example (every branch, qwen3 layout,
    Qwen image tokens) through :func:`~.pixmo_points._build_example`'s branch-list convention."""
    return (
        cfg.annotation_sampling == "all"
        and cfg.message_format == "qwen3"
        and cfg.token_ids == Molmo2TokenIds()
    )


def _build_selected_example(
    dataset: EpochSeededExamples,
    pil_image: Any,
    fmt: SftFormatter,
    messages: List[Dict[str, Any]],
    weights: Sequence[Optional[float]],
    index: int,
    epoch: Optional[int],
    rng: np.random.RandomState,
) -> Dict[str, np.ndarray]:
    """Build a v2 example in the configured layout and tokenizer, optionally keeping one branch
    (``annotation_sampling="one"``, chosen on its own stream by
    :func:`~.pixmo_points.select_annotation`). A kept branch's loss multiplier is folded into the
    example's ``message_weight``; several branches with differing multipliers are refused, since
    the serializer here weights the example as a whole."""
    cfg = dataset.config  # type: ignore[attr-defined]
    if cfg.annotation_sampling == "one" and len(messages) > 1:
        k = select_annotation(cfg.seed, index, dataset.resolve_epoch(epoch), len(messages))
        messages, weights = [messages[k]], [weights[k]]
    multipliers = {1.0 if w is None else float(w) for w in weights}
    if len(multipliers) > 1:
        raise ValueError(
            "Per-branch loss multipliers need the released qwen3 layout with every branch; "
            "use annotation_sampling='one' or leave negative_weight unset"
        )
    (multiplier,) = multipliers
    message_weight = cfg.message_weight
    if multiplier != 1.0:
        message_weight = multiplier * (1.0 if message_weight is None else message_weight)
    seq = _build_example(
        dataset.tokenizer,  # type: ignore[attr-defined]
        pil_image,
        lambda branch_rng: [
            fmt.format_turns(msg, index=index, rng=branch_rng)[0] for msg in messages
        ],
        max_crops=cfg.max_crops,
        loss_token_weighting=cfg.loss_token_weighting,
        token_ids=cfg.token_ids,
        message_weight=message_weight,
        message_format=cfg.message_format,
        rng=rng,
    )
    return truncate_for_format(seq, cfg.max_sequence_length, cfg.token_ids, cfg.message_format)


class PixMoPointsV2Dataset(EpochSeededExamples):
    """Map-style dataset over the images of :class:`PixMoPointsV2DatasetConfig` that have at
    least one trainable annotation."""

    def __init__(self, config: PixMoPointsV2DatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._data = _load_split(config.dataset_path, "train")
        self._index = self._build_index()
        log.info(
            "PixMoPointsV2 (%s): %d of %d images have a trainable annotation "
            "(%d dropped for appearing in a held-out set)",
            config.dataset_path,
            len(self._index),
            len(self._data),
            self.n_heldout_dropped,
        )

    # -- selection -------------------------------------------------------------------------

    def keep(self, anno: Dict[str, Any]) -> bool:
        """Whether to train on this annotation (mm_olmo ``PixMoPointV2._keep``, pointing leg).

        Must agree with :meth:`_build_index`\'s vectorized copy: a row is admitted when *any* of
        its annotations passes here, and every annotation the row carries is then re-tested
        during formatting. A null ``points`` is rejected on both paths (the index maps it to
        ``-1``, which fails the ``>= min_points`` bound), so it cannot reach ``format_row`` and
        raise there.
        """
        cfg = self.config
        label = anno.get("label")
        if not label or not label.strip():
            return False
        if cfg.filter_audit and _failed_audit(anno.get("audit_result")):
            return False
        points = anno.get("points")
        if points is None:
            return False
        return cfg.min_points <= len(points) <= cfg.max_points

    def _build_index(self) -> np.ndarray:
        """Row indices with ``kind``'s source and at least one annotation passing :meth:`keep`
        -- mm_olmo's build-time ``ds.filter``, computed in memory instead of as an Arrow cache
        file next to the (shared) data."""
        import pyarrow as pa
        import pyarrow.compute as pc

        cfg = self.config
        table = self._data.data

        def _keep_flat(flat) -> np.ndarray:
            n_points = pc.list_value_length(flat.field("points")).to_numpy(zero_copy_only=False)
            n_points = np.nan_to_num(n_points.astype(np.float64), nan=-1)
            has_label = _bool_np(
                pc.invert(pc.equal(pc.utf8_trim_whitespace(flat.field("label")), ""))
            )
            keep = has_label & (n_points >= cfg.min_points) & (n_points <= cfg.max_points)
            if cfg.filter_audit:
                failed = _bool_np(
                    pc.is_in(
                        flat.field("audit_result"),
                        value_set=pa.array(sorted(FAILED_AUDIT_RESULTS)),
                    )
                )
                keep &= ~failed
            return keep

        rows = _rows_with_any(table.column("annotations"), _keep_flat)
        source = _KIND_TO_SOURCE[cfg.kind]
        if source is not None:
            rows &= _bool_np(pc.equal(table.column("source"), source))
        train_only = _not_heldout(table, "image_url", heldout_ids(cfg.heldout_paths, "image_url"))
        self.n_heldout_dropped = int((rows & ~train_only).sum())
        return np.flatnonzero(rows & train_only)

    def __len__(self) -> int:
        return len(self._index)

    # -- formatting ------------------------------------------------------------------------

    def format_row(
        self, row: Dict[str, Any], rng: np.random.RandomState
    ) -> Tuple[List[Dict[str, Any]], List[Optional[float]]]:
        """One image's branch messages and per-branch loss weights (mm_olmo
        ``PixMoPointV2.format_example``, in its draw order: annotation styles, then the hard,
        easy and paired negative samples, then the negatives' styles).

        Each message is a formatter sub-example: ``style``, ``label``, ``points`` (``(N, 2)``
        in 0-100 percent coordinates, clipped) -- ``N == 0`` renders ``"There are none."``.
        """
        cfg = self.config
        messages: List[Dict[str, Any]] = []
        weights: List[Optional[float]] = []
        negatives: List[str] = []

        for anno in row["annotations"]:
            if not self.keep(anno):
                continue
            label = anno["label"].strip()
            raw = anno["points"]
            if len(raw) == 0:
                negatives.append(label)  # an annotated absence
                continue
            # Rows carry (x, y, depth) per point; only x, y are rendered.
            points = np.asarray(raw, dtype=np.float64).reshape(len(raw), -1)[:, :2]
            if cfg.audit_style and _failed_audit(anno.get("audit_result")):
                style = _choose(rng, cfg.audit_style)
            else:
                style = _choose(rng, cfg.style)
            messages.append(
                dict(style=style, label=label, points=points, point_scale=100, clip_points=True)
            )
            weights.append(None)

        easy = [str(x) for x in row.get("easy_negatives") or []]
        if cfg.n_hard_negatives > 0:
            pool = easy + [str(x) for x in row.get("paired_negatives") or []]
            if len(pool) > cfg.n_hard_negatives:
                pool = [str(x) for x in rng.choice(pool, size=cfg.n_hard_negatives, replace=False)]
            negatives += pool
        if cfg.n_easy_samples > 0:
            sample = easy
            if len(sample) > cfg.n_easy_samples:
                sample = [
                    str(x) for x in rng.choice(sample, size=cfg.n_easy_samples, replace=False)
                ]
            negatives += sample
        if cfg.p_paired_negatives > 0:
            key = "paired_negatives_v2" if cfg.v2_paired_negatives else "paired_negatives"
            paired = [str(x) for x in row.get(key) or []]
            expected = len(paired) * cfg.p_paired_negatives
            n_paired = int(expected) + int((expected - int(expected)) > rng.random())
            if len(paired) > n_paired:
                paired = [str(x) for x in rng.choice(paired, size=n_paired, replace=False)]
            negatives += paired

        for label in negatives:
            messages.append(
                dict(
                    style=_choose(rng, cfg.style),
                    label=label,
                    points=np.zeros((0, 2), dtype=np.float64),
                    point_scale=100,
                    clip_points=True,
                )
            )
            weights.append(1.0 if label in easy else cfg.negative_weight)

        if not messages:
            raise ValueError("PixMoPointsV2 row has no trainable annotation")
        return messages, weights

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        return self._build(i)

    def _build(self, i: int, epoch: Optional[int] = None) -> Dict[str, np.ndarray]:
        cfg = self.config
        row = self._data[int(self._index[i])]
        # Per (row, epoch): the negative sub-sampling in `format_row` has to rotate across epochs.
        rng = self.epoch_rng(i, epoch)
        messages, weights = self.format_row(row, rng)
        fmt = SftFormatter(seed=cfg.seed, **STAGE1_PROMPT_FAMILY)
        if _released_layout(cfg):
            branches = [fmt.format_turns(msg, index=i, rng=rng)[0] for msg in messages]
            return _build_example(
                self.tokenizer,
                _open_image(row["image"]),
                branches,
                max_crops=cfg.max_crops,
                loss_token_weighting=cfg.loss_token_weighting,
                message_weight=cfg.message_weight,
                shuffle_rng=rng,
                branch_weights=weights,
            )
        return _build_selected_example(
            self, _open_image(row["image"]), fmt, messages, weights, i, epoch, rng
        )


# ---------------------------------------------------------------------------
# PixMo count v2: audited PixMo-Count grouped by image
# ---------------------------------------------------------------------------


@dataclass
class PixMoCountV2DatasetConfig(Config):
    """mm_olmo ``PixMoCountConfigV2``: PixMo-Count re-grouped by image with a VLM audit per
    point set. Points are pixel coordinates (normalised by the image size, as in the v1
    ``count`` build); unlike v1 the two styles are drawn per annotation rather than doubling
    the dataset.
    """

    dataset_path: str = f"{PIXMO_DATASETS}/count-v2"
    """HF ``DatasetDict`` saved with ``save_to_disk``. Only its ``train`` split is ever read:
    this is a training source, and the build's ``validation`` / ``test`` splits are copies of
    ``train`` rather than held-out data."""

    heldout_paths: Tuple[str, ...] = COUNT_V2_HELDOUT_PATHS
    """Held-out sets (see :func:`~.sft_common.heldout_ids`); a row whose ``image_sha256`` appears
    in one is never trained on. PixMo-Count's own train split shares 15 images with its test
    split, and this build inherits them."""

    style: Tuple[str, ...] = ("point_count", "pointing")
    """Styles drawn uniformly per annotation (mm_olmo ``style="both"``)."""

    filter_audit: bool = False
    """Drop point sets whose ``audit_result`` is ``error`` / ``clear_error``."""

    audit_style: Optional[Tuple[str, ...]] = None
    """Styles drawn for audit-failed point sets that are kept, e.g.
    ``("aux_point_count", "aux_pointing")``."""

    max_crops: int = 8
    loss_token_weighting: str = "none"
    """``"none"`` weights every response token equally, as the released Molmo2 pretrain does for
    its pointing sources (``loss_token_weighting: None``)."""
    message_weight: Optional[float] = None
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)
    """Image token IDs of the selected language-model tokenizer (set by the mixture)."""
    message_format: SftMessageFormat = "qwen3"
    """``"qwen3"`` (the released Molmo2 chat layout) or ``"document"`` (plain pretraining
    documents, for a language model trained without a chat template)."""
    annotation_sampling: AnnotationSampling = "all"
    """``"all"`` trains every branch of an image (its kept annotations and sampled negatives) as
    sibling branches of one example; ``"one"`` keeps a single branch per example, chosen per
    ``(seed, index, epoch)`` (for language models that isolate packed examples only by document
    boundaries, e.g. Kimi Delta Attention)."""
    max_sequence_length: Optional[int] = None
    """Tail-truncate the built sequence to this many tokens. Applies outside the default
    ``qwen3`` / ``"all"`` path, which keeps the released stage-1 example exactly."""
    seed: int = 0

    def validate(self):
        if not self.style:
            raise OLMoConfigurationError("style must name at least one style")
        if self.audit_style is not None and not self.audit_style:
            raise OLMoConfigurationError("audit_style must be None or name at least one style")
        _validate_annotation_sampling(self.annotation_sampling)

    def build(self, tokenizer) -> "PixMoCountV2Dataset":
        self.validate()
        return PixMoCountV2Dataset(self, tokenizer)


class PixMoCountV2Dataset(EpochSeededExamples):
    """Map-style dataset over the images of :class:`PixMoCountV2DatasetConfig` that keep at
    least one point set."""

    def __init__(self, config: PixMoCountV2DatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._data = _load_split(config.dataset_path, "train")
        self._index = self._build_index()
        log.info(
            "PixMoCountV2 (%s/train): %d of %d images have a trainable point set "
            "(%d dropped for appearing in a held-out set)",
            config.dataset_path,
            len(self._index),
            len(self._data),
            self.n_heldout_dropped,
        )

    def keep(self, anno: Dict[str, Any]) -> bool:
        """Whether to train on this point set (mm_olmo ``PixMoCountConfigV2``).

        Beyond mm_olmo\'s audit filter this also drops unusable annotations, which mm_olmo never
        had to: a blank ``label`` is not an error here, it silently trains ``"pointing:"`` against
        an empty-named ``<points>`` tag, and a null ``label`` / ``points`` instead raises deep in
        formatting. :meth:`_build_index` applies the same rules, so a row survives only while it
        still has a usable point set.
        """
        if self.config.filter_audit and _failed_audit(anno.get("audit_result")):
            return False
        label = anno.get("label")
        if not label or not label.strip():
            return False
        return anno.get("points") is not None

    def _build_index(self) -> np.ndarray:
        """Rows with at least one kept point set (mm_olmo's ``filter_audit`` build-time
        filter; without it every row qualifies)."""
        import pyarrow as pa
        import pyarrow.compute as pc

        def _keep_flat(flat) -> np.ndarray:
            # Mirrors `keep`: a usable label and non-null points, then the audit filter.
            keep = _bool_np(
                pc.invert(pc.equal(pc.utf8_trim_whitespace(flat.field("label")), ""))
            ) & ~_bool_np(pc.is_null(flat.field("points")))
            if self.config.filter_audit:
                keep &= ~_bool_np(
                    pc.is_in(
                        flat.field("audit_result"),
                        value_set=pa.array(sorted(FAILED_AUDIT_RESULTS)),
                    )
                )
            return keep

        table = self._data.data
        rows = _rows_with_any(table.column("points"), _keep_flat)
        train_only = _not_heldout(
            table,
            "image_sha256",
            heldout_ids(self.config.heldout_paths, "image_sha256"),
        )
        self.n_heldout_dropped = int((rows & ~train_only).sum())
        return np.flatnonzero(rows & train_only)

    def __len__(self) -> int:
        return len(self._index)

    def format_row(
        self, row: Dict[str, Any], rng: np.random.RandomState, image_size: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """One image's branch messages (mm_olmo ``PixMoCountConfigV2.format_example``)."""
        cfg = self.config
        messages: List[Dict[str, Any]] = []
        for anno in row["points"]:
            if not self.keep(anno):
                continue
            if cfg.audit_style and _failed_audit(anno.get("audit_result")):
                style = _choose(rng, cfg.audit_style)
            else:
                style = _choose(rng, cfg.style)
            xy = np.asarray(anno["points"], dtype=np.float64).reshape(-1, 2)
            messages.append(
                dict(
                    style=style,
                    label=anno["label"],
                    points=xy,
                    point_scale=None,
                    image_size=image_size,
                )
            )
        if not messages:
            raise ValueError("PixMoCountV2 row has no trainable point set")
        return messages

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        return self._build(i)

    def _build(self, i: int, epoch: Optional[int] = None) -> Dict[str, np.ndarray]:
        cfg = self.config
        row = self._data[int(self._index[i])]
        rng = self.epoch_rng(i, epoch)
        pil = _open_image(row["image"])
        messages = self.format_row(row, rng, pil.size)
        fmt = SftFormatter(seed=cfg.seed, **STAGE1_PROMPT_FAMILY)
        if _released_layout(cfg):
            branches = [fmt.format_turns(msg, index=i, rng=rng)[0] for msg in messages]
            return _build_example(
                self.tokenizer,
                pil,
                branches,
                max_crops=cfg.max_crops,
                loss_token_weighting=cfg.loss_token_weighting,
                message_weight=cfg.message_weight,
                shuffle_rng=rng,
            )
        return _build_selected_example(
            self, pil, fmt, messages, [None] * len(messages), i, epoch, rng
        )
