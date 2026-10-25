"""Sequence packing for Molmo2 multimodal training.

Most stage-1 examples are far shorter than the training sequence length (caption/pointing
real lengths are ~1.1-1.5k tokens vs a 4096 pad length → ~65% of every sequence is
padding, i.e. wasted compute). Packing concatenates several whole examples into one
sequence so there is little/no padding, recovering that compute as useful tokens — the
OLMo-core analogue of mm_olmo's dynamic packer.

Cross-example isolation reuses the same machinery as intra-example branch isolation:

* a per-token ``example_ids`` vector marks which packed example each token belongs to;
  :class:`~olmo_core.nn.vision.MultimodalLM` ANDs ``example_ids[q] == example_ids[k]`` into
  the attention keep-mask so a token never attends across an example boundary.
* per-example ``position_ids`` are preserved (each example keeps its own 0-based RoPE
  positions / branch overlap), so packing is invisible to RoPE.
* ``subsegment_ids`` are concatenated (examples without branches get a constant id, which
  is unrestrictive within the example); the existing ``subseg[q] <= subseg[k]`` rule then
  applies *within* each example, gated by the example-id equality above.
* image crops are concatenated along the crop axis and each example's
  ``pooled_patches_idx`` is offset by the running crop-patch count so the connector still
  gathers the right patches.

Examples are never split, so the ``#<im_patch> == #pooled-features`` invariant is preserved
per example (and hence for the pack).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np

__all__ = [
    "pack_examples",
    "greedy_pack_indices",
    "iter_packs",
    "example_has_images",
    "select_subset_2d_knapsack",
    "PackingConstraint",
    "DynamicPacker",
    "iter_dynamic_packs",
]


@dataclass(frozen=True)
class _PackedImageParts:
    """Float32 crop arrays copied directly into the collator's final allocation."""

    parts: tuple[np.ndarray, ...]
    shape: tuple[int, int, int]


def example_has_images(ex: Dict[str, np.ndarray]) -> bool:
    """True when the example carries real image crops (not text-only NLP)."""
    return ex["images"].shape[0] > 0


def example_crop_count(ex: Dict[str, np.ndarray]) -> int:
    """Number of image crops in an example (0 for text-only)."""
    return int(ex["images"].shape[0])


def greedy_pack_indices(
    lengths: Sequence[int],
    seq_len: int,
    *,
    crop_counts: Optional[Sequence[int]] = None,
    max_crops_per_pack: Optional[int] = None,
) -> List[List[int]]:
    """Greedily group example indices so each group's total length ``<= seq_len``.

    A simple next-fit over the given order keeps the sampling order stable and remains the
    default packing policy for backwards compatibility.

    :param lengths: real token length of each example, in the order they should be packed.
    :param seq_len: maximum packed length.
    :param crop_counts: per-example crop counts (required when ``max_crops_per_pack`` is set).
    :param max_crops_per_pack: maximum total crops in one pack (mm_olmo ``image_weight`` parity).
    :returns: a list of groups, each a list of indices into ``lengths``.
    """
    if max_crops_per_pack is not None and crop_counts is None:
        raise ValueError("crop_counts is required when max_crops_per_pack is set")
    groups: List[List[int]] = []
    cur: List[int] = []
    cur_len = 0
    cur_crops = 0
    for i, n in enumerate(lengths):
        n = int(n)
        crops = int(crop_counts[i]) if crop_counts is not None else 0
        over_tokens = cur and cur_len + n > seq_len
        over_crops = (
            max_crops_per_pack is not None and cur and cur_crops + crops > max_crops_per_pack
        )
        if over_tokens or over_crops:
            groups.append(cur)
            cur, cur_len, cur_crops = [], 0, 0
        cur.append(i)
        cur_len += n
        cur_crops += crops
    if cur:
        groups.append(cur)
    return groups


def _select_buffered_pack_indices(
    lengths: Sequence[int],
    crop_counts: Sequence[int],
    seq_len: int,
    max_crops_per_pack: int,
    image_weight: float = 1.0,
) -> List[int]:
    """Select one pack with Molmo2's two-constraint dynamic program."""
    if not np.isfinite(image_weight) or image_weight < 0:
        raise ValueError("image_weight must be finite and non-negative")
    token_granularity = max(1, seq_len // 512)
    token_values = np.asarray(
        [(int(n) + token_granularity - 1) // token_granularity for n in lengths],
        dtype=np.int64,
    )
    crop_values = np.asarray(crop_counts, dtype=np.int64)
    # The released 2,560-token recipe is evenly divisible by this granularity and therefore
    # matches Molmo2 exactly. For custom unaligned lengths, floor division is deliberately
    # conservative so quantization can never select a pack whose real tokens exceed seq_len.
    max_tokens = seq_len // token_granularity

    # Match Molmo2's objective: useful text tokens plus weighted image crops. Stage 1 uses
    # image_weight=1 while Stage 2 uses image_weight=30.
    objective = np.asarray(lengths, dtype=np.float32) + (
        crop_values.astype(np.float32) * float(image_weight)
    )
    return select_subset_2d_knapsack(
        token_values.tolist(),
        crop_values.tolist(),
        max_tokens,
        max_crops_per_pack,
        objective.tolist(),
    )


def _pop_buffered_pack(
    buffer: List[Dict[str, np.ndarray]],
    seq_len: int,
    max_crops_per_pack: int,
    image_weight: float = 1.0,
) -> List[Dict[str, np.ndarray]]:
    selected = _select_buffered_pack_indices(
        [len(ex["input_ids"]) for ex in buffer],
        [example_crop_count(ex) for ex in buffer],
        seq_len,
        max_crops_per_pack,
        image_weight,
    )
    if not selected:
        raise RuntimeError("Buffered packer could not select an example within its constraints")
    # Emit in buffer order, like Molmo2's DynamicSolver and :class:`DynamicPacker`.
    packed = [buffer[i] for i in sorted(selected)]
    for i in sorted(selected, reverse=True):
        buffer.pop(i)
    return packed


def pack_examples(
    examples: List[Dict[str, np.ndarray]], *, defer_image_copy: bool = False
) -> Dict[str, Any]:
    """Concatenate several example dicts into one packed example.

    Each input is a dict as produced by the stage-1 datasets (``input_ids``, ``labels``,
    ``loss_masks``, ``position_ids``, ``token_type_ids``, optional ``subsegment_ids``,
    ``images`` ``(n_crops, n_patches, patch_dim)``, ``pooled_patches_idx``
    ``(n_pooled, pool_size)``). The result is the concatenation along the token axis (and
    crop axis for images), plus an ``example_ids`` vector. It is **not** padded — the
    collator pads to the batch/seq length.

    :param defer_image_copy: Retain float32 image parts for direct final-batch assembly by
        :class:`~olmo_core.data.multimodal.collator.MultimodalCollator`. Other image dtypes
        retain the ordinary concatenation path and its dtype-promotion behavior. Input
        image arrays must remain unmodified until collation when this option is enabled.
    :raises ValueError: if ``examples`` is empty.
    """
    if not examples:
        raise ValueError("pack_examples requires at least one example")

    tok_keys = ["input_ids", "labels", "loss_masks", "position_ids", "token_type_ids"]
    out: Dict[str, Any] = {}
    for k in tok_keys:
        out[k] = np.concatenate([ex[k] for ex in examples], axis=0)

    # Subsegment ids: keep each example's own (branch isolation); examples without branches
    # get a constant id so `subseg[q] <= subseg[k]` is unrestrictive within them. The values
    # need not be globally unique — example-id equality gates cross-example attention.
    subseg_parts = []
    for ex in examples:
        n = len(ex["input_ids"])
        if "subsegment_ids" in ex:
            subseg_parts.append(ex["subsegment_ids"])
        else:
            subseg_parts.append(np.zeros(n, dtype=np.int64))
    out["subsegment_ids"] = np.concatenate(subseg_parts, axis=0)

    # Per-token example id (0, 1, 2, ...).
    out["example_ids"] = np.concatenate(
        [np.full(len(ex["input_ids"]), i, dtype=np.int64) for i, ex in enumerate(examples)],
        axis=0,
    )

    if any("_source_name" in ex for ex in examples):
        out["pack_source_names"] = [ex.get("_source_name", "?") for ex in examples]

    # Images: concatenate crops; offset each example's pooled indices by the running
    # crop-patch count so they index into the concatenated (total_crops * n_patches) axis.
    images = [ex["images"] for ex in examples]
    n_patches = next((im.shape[1] for im in images if im.shape[0]), images[0].shape[1])
    patch_dim = next((im.shape[2] for im in images if im.shape[0]), images[0].shape[2])
    pool_size = examples[0]["pooled_patches_idx"].shape[1]

    pooled_parts: List[np.ndarray] = []
    crop_offset = 0
    for ex in examples:
        im = ex["images"]
        pp = ex["pooled_patches_idx"].copy()
        if pp.shape[0]:
            valid = pp >= 0
            pp[valid] = pp[valid] + crop_offset * n_patches
            pooled_parts.append(pp)
        crop_offset += im.shape[0]

    image_parts = tuple(im for im in images if im.shape[0])
    if (
        defer_image_copy
        and image_parts
        and all(
            im.dtype == np.float32 and im.ndim == 3 and im.shape[1:] == (n_patches, patch_dim)
            for im in image_parts
        )
    ):
        out["images"] = _PackedImageParts(image_parts, (crop_offset, n_patches, patch_dim))
    else:
        out["images"] = (
            np.concatenate(image_parts, axis=0)
            if image_parts
            else np.zeros((0, n_patches, patch_dim), dtype=np.float32)
        )
    out["pooled_patches_idx"] = (
        np.concatenate(pooled_parts, axis=0)
        if pooled_parts
        else np.full((0, pool_size), -1, dtype=np.int64)
    )
    return out


def iter_packs(
    examples: Iterable[Dict[str, np.ndarray]],
    seq_len: int,
    *,
    max_crops_per_pack: Optional[int] = None,
    buffer_size: int = 0,
    image_weight: float = 1.0,
    defer_image_copy: bool = False,
) -> Iterator[Dict[str, Any]]:
    """Pack a stream of example dicts into ``<= seq_len`` sequences.

    ``examples`` is an iterator of example dicts — typically an infinite, cycled (and
    optionally prefetched) stream, in which case this yields indefinitely and the caller
    caps the number of batches. Cycling in the caller keeps every data-parallel rank
    yielding the same number of batches regardless of how its examples pack (no collective
    desync). An example longer than ``seq_len`` is emitted alone (the collator tail-truncates
    it; the image block at the front is preserved). A finite stream flushes all examples.

    The default ``buffer_size=0`` retains the original deterministic next-fit behavior.
    A positive buffer enables the same two-constraint knapsack policy used by Molmo2: it
    selects examples that maximize useful tokens while respecting the token and crop
    budgets. Buffered packing may safely mix image and text-only examples because selected
    packs are guaranteed not to be tail-truncated.

    :param examples: iterator of per-example dicts (the heavy loading happens upstream, so it
        can be prefetched off the training thread).
    :param seq_len: target packed length.
    :param max_crops_per_pack: cap total image crops per pack (mm_olmo crop-budget parity).
    :param buffer_size: number of examples considered by the dynamic packing solver; zero
        keeps the original next-fit policy.
    :param image_weight: objective value assigned to each image crop by the buffered
        solver. Molmo2 Stage 1 uses 1 and Stage 2 uses 30.
    :param defer_image_copy: Retain float32 image parts for direct final-batch collation.
    """
    if buffer_size < 0:
        raise ValueError("buffer_size must be non-negative")
    if buffer_size:
        if max_crops_per_pack is None:
            raise ValueError("max_crops_per_pack is required when buffer_size is positive")
        if max_crops_per_pack <= 0:
            raise ValueError("max_crops_per_pack must be positive")

        buffer: List[Dict[str, np.ndarray]] = []
        for ex in examples:
            length = len(ex["input_ids"])
            crops = example_crop_count(ex)
            token_granularity = max(1, seq_len // 512)
            at_token_capacity = (length + token_granularity - 1) // token_granularity >= (
                seq_len + token_granularity - 1
            ) // token_granularity
            # Preserve the existing handling of an individually oversized example. Stage-1
            # datasets are bounded by these constraints, but emitting alone is safer than
            # allowing an invalid item to stall an infinite stream.
            if at_token_capacity or crops > max_crops_per_pack:
                yield pack_examples([ex], defer_image_copy=defer_image_copy)
                continue
            if len(buffer) < buffer_size:
                buffer.append(ex)
                continue
            packed = _pop_buffered_pack(
                buffer, seq_len, max_crops_per_pack, image_weight=image_weight
            )
            buffer.append(ex)
            yield pack_examples(packed, defer_image_copy=defer_image_copy)

        while buffer:
            yield pack_examples(
                _pop_buffered_pack(buffer, seq_len, max_crops_per_pack, image_weight=image_weight),
                defer_image_copy=defer_image_copy,
            )
        return

    cur: List[Dict[str, np.ndarray]] = []
    cur_len = 0
    cur_crops = 0
    for ex in examples:
        length = len(ex["input_ids"])
        crops = example_crop_count(ex)
        # Do not mix text-only NLP (e.g. Tulu4) with image-bearing examples in one pack:
        # head-truncating a cross-modal pack can orphan <im_patch> tokens from their
        # pooled rows (mm_olmo uses separate packing constraints when NLP is enabled).
        over_tokens = cur and cur_len + length > seq_len
        over_crops = (
            max_crops_per_pack is not None
            and cur
            and example_has_images(cur[0])
            and cur_crops + crops > max_crops_per_pack
        )
        if cur and (
            over_tokens or over_crops or example_has_images(ex) != example_has_images(cur[0])
        ):
            yield pack_examples(cur, defer_image_copy=defer_image_copy)
            cur, cur_len, cur_crops = [], 0, 0
        cur.append(ex)
        cur_len += length
        cur_crops += crops
    if cur:
        yield pack_examples(cur, defer_image_copy=defer_image_copy)


# ---------------------------------------------------------------------------
# 2D-knapsack dynamic packer (port of mm_olmo ``olmo/data/dynamic_packer.py``)
# ---------------------------------------------------------------------------


def select_subset_2d_knapsack(
    t_values: Sequence[int],
    i_values: Sequence[int],
    max_t: int,
    max_i: int,
    obj_vals: Sequence[float],
) -> List[int]:
    """Vectorized 2D knapsack dynamic-program solver (verbatim mm_olmo port).

    Selects the subset of items maximizing ``sum(obj_vals)`` subject to
    ``sum(t_values) <= max_t`` and ``sum(i_values) <= max_i``.

    :returns: selected indices (descending).
    """
    M = len(t_values)

    # DP table with quantized dimensions.
    dp = np.zeros((M + 1, max_t + 1, max_i + 1), dtype=np.float32)

    for item in range(1, M + 1):
        t_val_q = t_values[item - 1]
        i_val_q = i_values[item - 1]
        obj_val = obj_vals[item - 1]

        # Copy previous layer.
        dp[item] = dp[item - 1]

        # Vectorized update where the item can fit.
        if t_val_q <= max_t and i_val_q <= max_i:
            take_val = dp[item - 1, : max_t + 1 - t_val_q, : max_i + 1 - i_val_q] + obj_val
            dp[item, t_val_q:, i_val_q:] = np.maximum(dp[item, t_val_q:, i_val_q:], take_val)

    # Backtrack to find the solution.
    selected_indices: List[int] = []
    t_rem_q, i_rem_q = max_t, max_i
    for item in range(M, 0, -1):
        t_val_q = t_values[item - 1]
        i_val_q = i_values[item - 1]
        if (
            t_val_q <= t_rem_q
            and i_val_q <= i_rem_q
            and dp[item, t_rem_q, i_rem_q] != dp[item - 1, t_rem_q, i_rem_q]
        ):
            selected_indices.append(item - 1)
            t_rem_q -= t_val_q
            i_rem_q -= i_val_q
    return selected_indices


class PackingConstraint:
    """One capacity dimension of the dynamic packer (mm_olmo ``Constraint``).

    :param key: example-dict key whose ``len()`` is constrained (``len(images)`` is the
        crop count for our ``(n_crops, n_patches, patch_dim)`` arrays).
    :param max_len: max total ``len(example[key])`` in one pack.
    :param allow_shortcut: emit an example alone (without buffering) when it already
        reaches ``max_len`` on its own.
    :param weight: objective value per unit of this dimension in the knapsack.
    :param granularity: quantization step for the DP table.
    """

    def __init__(
        self, key: str, max_len: int, allow_shortcut: bool, weight: float, granularity: int
    ):
        self.key = key
        self.max_len = max_len
        self.allow_shortcut = allow_shortcut
        self.weight = weight
        self.granularity = max(1, int(granularity))

    def get_quantized_value(self, val: int) -> int:
        return (val + self.granularity - 1) // self.granularity

    def get_quantized_max_len(self) -> int:
        return self.get_quantized_value(self.max_len)


class DynamicPacker:
    """Buffered 2D-knapsack packer (port of mm_olmo ``DynamicSolver``).

    Buffers up to ``max_buffer_size`` examples; once full, a 2D knapsack picks the
    highest-value subset that fits both capacities (text tokens and image crops) and
    emits it as one pack. Text-only and image examples pack together, exactly like
    mm_olmo (``pack()`` gives text-only rows an empty crop array).

    Deviation from mm_olmo (documented): an example that *alone* exceeds a
    non-shortcut constraint (e.g. a >max-crops example) is emitted as its own pack
    instead of being buffered — mm_olmo would keep it in the buffer forever (the
    knapsack can never select it), eventually starving the buffer.
    """

    def __init__(
        self,
        max_buffer_size: int,
        constraints: Sequence[PackingConstraint],
        pack_fn=None,
    ):
        if len(constraints) != 2:
            raise NotImplementedError("DynamicPacker currently supports exactly 2 constraints")
        self.max_buffer_size = max_buffer_size
        self.constraints = list(constraints)
        self.pack_fn = pack_fn or pack_examples
        # Parallel lists: quantized lens per constraint, objective values, examples.
        self._buffer: List[Dict[str, np.ndarray]] = []
        self._buffer_qlens: List[Dict[str, int]] = []
        self._buffer_values: List[float] = []

    def __len__(self) -> int:
        return len(self._buffer)

    def _quantized_lens_and_value(self, example):
        qlens: Dict[str, int] = {}
        value = 0.0
        for c in self.constraints:
            if c.key in example:
                n = len(example[c.key])
                qlens[c.key] = c.get_quantized_value(n)
                value += n * c.weight
            else:
                qlens[c.key] = 0
        return qlens, value

    def __call__(self, example: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        """Offer one example; returns a finished pack or ``None`` (buffered)."""
        for c in self.constraints:
            n = len(example[c.key]) if c.key in example else 0
            at_capacity = c.get_quantized_value(n) >= c.get_quantized_max_len()
            over_capacity = n > c.max_len
            if (c.allow_shortcut and at_capacity) or over_capacity:
                # Already fills (or exceeds) a capacity on its own — emit alone.
                return self.pack_fn([example])

        qlens, value = self._quantized_lens_and_value(example)
        if len(self._buffer) < self.max_buffer_size:
            self._buffer.append(example)
            self._buffer_qlens.append(qlens)
            self._buffer_values.append(value)
            return None

        # Buffer full: solve over the buffered examples, emit the best-fitting subset.
        c1, c2 = self.constraints
        indices = select_subset_2d_knapsack(
            [q[c1.key] for q in self._buffer_qlens],
            [q[c2.key] for q in self._buffer_qlens],
            c1.get_quantized_max_len(),
            c2.get_quantized_max_len(),
            self._buffer_values,
        )
        if len(indices) == 0:
            raise RuntimeError("No indices returned by select_subset_2d_knapsack")

        out = self.pack_fn([self._buffer[i] for i in sorted(indices)])
        for ix in sorted(indices, reverse=True):
            self._buffer.pop(ix)
            self._buffer_qlens.pop(ix)
            self._buffer_values.pop(ix)
        self._buffer.append(example)
        self._buffer_qlens.append(qlens)
        self._buffer_values.append(value)
        return out

    def flush(self) -> Iterator[Dict[str, np.ndarray]]:
        """Drain the buffer (finite streams / end of epoch)."""
        c1, c2 = self.constraints
        while self._buffer:
            indices = select_subset_2d_knapsack(
                [q[c1.key] for q in self._buffer_qlens],
                [q[c2.key] for q in self._buffer_qlens],
                c1.get_quantized_max_len(),
                c2.get_quantized_max_len(),
                self._buffer_values,
            )
            if not indices:  # nothing fits (shouldn't happen; be safe)
                indices = [len(self._buffer) - 1]
            yield self.pack_fn([self._buffer[i] for i in sorted(indices)])
            for ix in sorted(indices, reverse=True):
                self._buffer.pop(ix)
                self._buffer_qlens.pop(ix)
                self._buffer_values.pop(ix)


def iter_dynamic_packs(
    examples: Iterable[Dict[str, np.ndarray]],
    seq_len: int,
    *,
    max_crops_per_pack: int,
    buffer_size: int = 48,
    text_weight: float = 1.0,
    image_weight: float = 30.0,
    flush: bool = True,
) -> Iterator[Dict[str, np.ndarray]]:
    """2D-knapsack-pack a stream of example dicts (mm_olmo SFT packing parity).

    mm_olmo stage-2 uses ``PackingConfig(buffer_size=48, image_weight=30,
    shortcut_max_len_images=False)`` with capacities = (sequence length, per-sequence
    crop capacity) and quantization ``(seq_len // 512, 1)``. Text-only examples pack
    together with image examples.

    :param seq_len: token capacity per pack.
    :param max_crops_per_pack: crop capacity per pack. Use the max crops a single
        example can produce (global crop + locals; the high-res budget if any dataset
        uses ``p_high_res > 0``) or more.
    :param buffer_size: candidate buffer size (mm_olmo: 48).
    :param flush: drain the buffer once ``examples`` is exhausted (set False to match
        mm_olmo's ``packed_iterator``, which drops the tail of infinite streams).
    """
    packer = DynamicPacker(
        buffer_size,
        [
            PackingConstraint("input_ids", seq_len, True, text_weight, max(1, seq_len // 512)),
            PackingConstraint("images", max_crops_per_pack, False, image_weight, 1),
        ],
    )
    for ex in examples:
        out = packer(ex)
        if out is not None:
            yield out
    if flush:
        yield from packer.flush()
