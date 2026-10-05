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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from olmo_core.config import Config

from .grounding import normalize_points, pointing_answer
from .qwen3_layout import branch_context_ids, image_prefix_ids
from .sequence_builder import build_branched_sequence, example_rng
from .sft_formatter import SftFormatter

__all__ = [
    "PixMoPointsDatasetConfig",
    "PixMoPointsDataset",
    "PixMoCountDatasetConfig",
    "PixMoCountDataset",
    "CoSynPointDatasetConfig",
    "CoSynPointDataset",
    "COSYN_POINT_STYLE",
    "COSYN_POINT_V2_PATH",
    "FAILED_AUDIT_RESULTS",
]

from olmo_core.exceptions import OLMoConfigurationError

from .paths import PIXMO_DATASETS

#: ``audit_result`` values mm_olmo treats as a failed audit (``PixMoPointV2._keep``,
#: ``CoSynPointConfigV2.FAILED``).
FAILED_AUDIT_RESULTS = frozenset({"error", "clear_error"})


def _build_example(
    tokenizer,
    pil_image,
    branches_text: List[Tuple[str, str]],
    *,
    max_crops: int,
    loss_token_weighting: str,
    message_weight: float | None = None,
    p_high_res: float = 0.0,
    shuffle_rng: np.random.RandomState | None = None,
    seed: int = 0,
    branch_weights: Optional[Sequence[Optional[float]]] = None,
) -> Dict[str, np.ndarray]:
    """Preprocess the image and assemble a (possibly multi-branch) pointing example.

    :param branches_text: list of ``(user_question, assistant_answer)`` strings.
    :param branch_weights: optional per-branch loss multipliers parallel to ``branches_text``
        (``None`` entries mean 1). mm_olmo's per-message ``AssistantMessage.weight``: the
        branch's response tokens are scaled by it on top of ``loss_token_weighting`` and
        ``message_weight``, which stay example-wide.
    """
    import torch

    from olmo_core.nn.vision.molmo2_image_processor import preprocess_image_molmo2

    branches_text = list(branches_text)
    weights = None if branch_weights is None else list(branch_weights)
    if weights is not None and len(weights) != len(branches_text):
        raise ValueError(
            f"branch_weights has {len(weights)} entries for {len(branches_text)} branches"
        )
    if len(branches_text) > 1:
        order = np.arange(len(branches_text))
        rng = shuffle_rng if shuffle_rng is not None else np.random.RandomState(seed)
        rng.shuffle(order)
        branches_text = [branches_text[i] for i in order]
        if weights is not None:
            weights = [weights[i] for i in order]

    preprocess_rng = shuffle_rng if shuffle_rng is not None else np.random.RandomState(seed)
    images_t, pooling_t, image_grid = preprocess_image_molmo2(
        pil_image,
        dtype=torch.float32,
        device=torch.device("cpu"),
        max_crops=max_crops,
        p_high_res=p_high_res,
        is_training=True,
        rng=preprocess_rng,
    )
    prefix = image_prefix_ids(tokenizer, image_grid)
    multi_branch = len(branches_text) > 1
    branches = [
        (
            branch_context_ids(tokenizer, q, branch_index=i, multi_branch=multi_branch),
            tokenizer.encode(a, add_special_tokens=False),
        )
        for i, (q, a) in enumerate(branches_text)
    ]
    from olmo_core.data.multimodal.message_weight import (
        apply_message_weight_to_loss_masks,
    )

    seq = build_branched_sequence(
        prefix,
        branches,
        eos_id=tokenizer.eos_token_id,
        loss_token_weighting=loss_token_weighting,
    )
    subsegment_ids = seq.get("subsegment_ids")
    from olmo_core.data.multimodal.message_weight import MessageWeight

    mw = MessageWeight.from_string(loss_token_weighting).with_overrides(message_weight)
    seq["loss_masks"] = apply_message_weight_to_loss_masks(
        seq["loss_masks"], subsegment_ids, mw, branch_scaling_already_applied=True
    )
    if weights is not None:
        seq["loss_masks"] = _apply_branch_weights(seq["loss_masks"], subsegment_ids, weights)
    seq["images"] = images_t[0].numpy()
    seq["pooled_patches_idx"] = pooling_t[0].numpy()
    return seq


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


def _load_split(path: str, split: str):
    from .dataset_compat import load_from_disk_compat

    ds = load_from_disk_compat(path)
    return ds[split] if hasattr(ds, "keys") and split in ds else ds


def _open_image(p):
    from PIL import Image

    return p if isinstance(p, Image.Image) else Image.open(p)


# ---------------------------------------------------------------------------
# PixMo points (pointing / counting)
# ---------------------------------------------------------------------------


@dataclass
class PixMoPointsDatasetConfig(Config):
    """``pixmo_points_train`` (kind=basic) / ``pixmo_points_high_freq_train`` (high_frequency)."""

    kind: str = "both"  # "basic" (points-pointing) | "high_frequency" (points-counting) | "both"
    counting: str = "both"  # "both" -> random pointing/point_count per branch
    max_points: int = 60
    max_total_points_per_example: int = 60
    max_crops: int = 8
    loss_token_weighting: str = "root_subsegments"
    message_weight: float | None = None
    p_high_res: float = 0.0
    seed: int = 0
    prompt_templates: str = "uber_model_v2"
    """Prompt family for the question text; stage 1 uses ``"none"`` (bare label)."""
    system_prompt: str = "demo_or_style_v2"
    """Prompt family for the style prefix; stage 1 uses ``"style_and_length_v2"``."""

    def build(self, tokenizer) -> "PixMoPointsDataset":
        return PixMoPointsDataset(self, tokenizer)


class PixMoPointsDataset:
    def __init__(self, config: PixMoPointsDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        sub = {"basic": ["points-pointing"], "high_frequency": ["points-counting"]}.get(
            config.kind, ["points-counting", "points-pointing"]
        )
        from datasets import concatenate_datasets

        self._data = concatenate_datasets(
            [_load_split(f"{PIXMO_DATASETS}/{s}", "train") for s in sub]
        )
        # Pre-split each row's labels into sub-batches with <= max_total_points (mm_olmo).
        self._index = self._build_sub_index()

    def _build_sub_index(self) -> List[Tuple[int, List[int]]]:
        cfg = self.config
        counts = self._data["count"]
        index: List[Tuple[int, List[int]]] = []
        for row, point_counts in enumerate(counts):
            on: List[int] = []
            total = 0
            for li, n in enumerate(point_counts):
                if n > cfg.max_points:
                    continue
                if on and total + n > cfg.max_total_points_per_example:
                    index.append((row, on))
                    on, total = [], 0
                on.append(li)
                total += n
            if on:
                index.append((row, on))
        return index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        row_idx, label_idxs = self._index[i]
        rng = example_rng(self.config.seed, i)
        row = self._data[row_idx]
        fmt = SftFormatter(
            seed=self.config.seed,
            prompt_templates=self.config.prompt_templates,
            system_prompt=self.config.system_prompt,
        )
        specs: List[Tuple[str, str, Any]] = []
        for li in label_idxs:
            label = row["label"][li]
            pts = row["points"][li]
            if self.config.counting == "both":
                style = rng.choice(["point_count", "pointing"])
            else:
                style = "point_count" if self.config.counting else "pointing"
            specs.append((style, label, pts))
        branches: List[Tuple[str, str]] = []
        for style, label, pts in specs:
            sub = {
                "style": style,
                "label": label,
                "points": pts,
                "point_scale": 100,
            }
            branches.append(fmt.format_turns(sub, index=i, rng=rng)[0])
        return _build_example(
            self.tokenizer,
            _open_image(row["image"]),
            branches,
            max_crops=self.config.max_crops,
            loss_token_weighting=self.config.loss_token_weighting,
            message_weight=self.config.message_weight,
            p_high_res=self.config.p_high_res,
            shuffle_rng=rng,
        )


# ---------------------------------------------------------------------------
# PixMo count (single annotation, alternating point_count / pointing)
# ---------------------------------------------------------------------------


@dataclass
class PixMoCountDatasetConfig(Config):
    counting: str = "both"  # "both" interleaves point_count (even) / pointing (odd)
    max_crops: int = 8
    loss_token_weighting: str = "root_subsegments"
    message_weight: float | None = None
    p_high_res: float = 0.0
    seed: int = 0
    prompt_templates: str = "uber_model_v2"
    """Prompt family for the question text; stage 1 uses ``"none"`` (bare label)."""
    system_prompt: str = "demo_or_style_v2"
    """Prompt family for the style prefix; stage 1 uses ``"style_and_length_v2"``."""

    def build(self, tokenizer) -> "PixMoCountDataset":
        return PixMoCountDataset(self, tokenizer)


class PixMoCountDataset:
    def __init__(self, config: PixMoCountDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._data = _load_split(f"{PIXMO_DATASETS}/count", "train")
        self._n = len(self._data)

    def __len__(self) -> int:
        return self._n * 2 if self.config.counting == "both" else self._n

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        if self.config.counting == "both":
            row_idx, style = i // 2, ("point_count" if i % 2 == 0 else "pointing")
        else:
            row_idx, style = i, ("point_count" if self.config.counting else "pointing")
        row = self._data[row_idx]
        label = row["label"]
        count = int(row["count"])
        pil = _open_image(row["image"])
        pts = row.get("points") or {"x": [], "y": []}
        rng = example_rng(self.config.seed, i)
        fmt = SftFormatter(
            seed=self.config.seed,
            prompt_templates=self.config.prompt_templates,
            system_prompt=self.config.system_prompt,
        )
        xy = np.array([pts["x"], pts["y"]], dtype=np.float64).T.reshape(-1, 2)
        sub = {
            "style": style,
            "label": label,
            "points": xy,
            "point_scale": None,
            "image_size": pil.size,
            "count": count,
        }
        prompt, answer = fmt.format_turns(sub, index=i, rng=rng)[0]
        return _build_example(
            self.tokenizer,
            pil,
            [(prompt, answer)],
            max_crops=self.config.max_crops,
            loss_token_weighting=self.config.loss_token_weighting,
            message_weight=self.config.message_weight,
            p_high_res=self.config.p_high_res,
            shuffle_rng=rng,
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
    dataset_path: str = f"{PIXMO_DATASETS}/cosyn-point"
    """HF dataset with ``image``, ``questions``, ``answer_points`` and ``names`` columns."""

    max_crops: int = 8
    loss_token_weighting: str = "root_subsegments"
    message_weight: float | None = None
    p_high_res: float = 0.0
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

    def build(self, tokenizer) -> "CoSynPointDataset":
        return CoSynPointDataset(self, tokenizer)


class CoSynPointDataset:
    def __init__(self, config: CoSynPointDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._data = _load_split(config.dataset_path, "train")
        if config.audit_style is not None and "audit_result" not in self._data.column_names:
            raise OLMoConfigurationError(
                f"audit_style={config.audit_style!r} needs an audited CoSyn build with an "
                f"`audit_result` column, such as {COSYN_POINT_V2_PATH!r}; {config.dataset_path!r} "
                f"has {self._data.column_names}"
            )

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        row = self._data[i]
        cfg = self.config
        # The prefix follows the checkpoint's family exactly as for the PixMo sources.
        fmt = SftFormatter(
            seed=cfg.seed, prompt_templates=cfg.prompt_templates, system_prompt=cfg.system_prompt
        )
        prefix = fmt.style_prefix(COSYN_POINT_STYLE)
        failed_prefix = fmt.style_prefix(cfg.audit_style) if cfg.audit_style else prefix
        audits = row["audit_result"] if cfg.audit_style else [None] * len(row["questions"])
        branches: List[Tuple[str, str]] = []
        for question, points, name, audit in zip(
            row["questions"], row["answer_points"], row["names"], audits
        ):
            tag = failed_prefix if audit in FAILED_AUDIT_RESULTS else prefix
            xy = np.array([points["x"], points["y"]], dtype=np.float64).T.reshape(-1, 2)
            norm = normalize_points(xy, point_scale=100, image_size=None)
            # cosyn_point uses the "pointing" answer (just the points tag), label = name.
            answer = pointing_answer(norm, name.lower(), "pointing", count=len(norm))
            branches.append((f"{tag} {question}" if tag else question, answer))
        return _build_example(
            self.tokenizer,
            _open_image(row["image"]),
            branches,
            max_crops=self.config.max_crops,
            loss_token_weighting=self.config.loss_token_weighting,
            message_weight=self.config.message_weight,
            p_high_res=self.config.p_high_res,
            shuffle_rng=example_rng(self.config.seed, i),
        )
