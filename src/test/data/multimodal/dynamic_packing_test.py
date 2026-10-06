"""Tests for the 2D-knapsack dynamic packer (mm_olmo ``dynamic_packer.py`` port)."""

from __future__ import annotations

import itertools

import numpy as np

from olmo_core.data.multimodal.packing import (
    DynamicPacker,
    PackingConstraint,
    iter_dynamic_packs,
    select_subset_2d_knapsack,
)

_PATCH_DIM = 8
_N_PATCHES = 4


def _example(n_text: int, n_crops: int, tag: int):
    L = n_text
    ex = dict(
        input_ids=np.full(L, tag, dtype=np.int64),
        labels=np.full(L, tag, dtype=np.int64),
        loss_masks=np.ones(L, dtype=np.float32),
        position_ids=np.arange(L, dtype=np.int64),
        token_type_ids=np.zeros(L, dtype=np.int64),
        images=np.full((n_crops, _N_PATCHES, _PATCH_DIM), float(tag), dtype=np.float32),
        pooled_patches_idx=(
            np.arange(n_crops * _N_PATCHES, dtype=np.int64).reshape(n_crops, _N_PATCHES)
            if n_crops
            else np.full((0, _N_PATCHES), -1, dtype=np.int64)
        ),
    )
    return ex


def _brute_force_2d(t_values, i_values, max_t, max_i, obj_vals):
    best_val, best = -1.0, set()
    n = len(t_values)
    for r in range(n + 1):
        for combo in itertools.combinations(range(n), r):
            if sum(t_values[i] for i in combo) > max_t:
                continue
            if sum(i_values[i] for i in combo) > max_i:
                continue
            v = sum(obj_vals[i] for i in combo)
            if v > best_val:
                best_val, best = v, set(combo)
    return best_val, best


def test_knapsack_matches_brute_force():
    rng = np.random.RandomState(0)
    for _ in range(25):
        n = int(rng.randint(1, 9))
        t = rng.randint(1, 8, size=n).tolist()
        i = rng.randint(0, 4, size=n).tolist()
        v = (rng.randint(1, 50, size=n).astype(np.float64)).tolist()
        max_t, max_i = int(rng.randint(4, 16)), int(rng.randint(1, 8))
        picked = select_subset_2d_knapsack(t, i, max_t, max_i, v)
        assert sum(t[j] for j in picked) <= max_t
        assert sum(i[j] for j in picked) <= max_i
        best_val, _ = _brute_force_2d(t, i, max_t, max_i, v)
        got_val = sum(v[j] for j in picked)
        assert abs(got_val - best_val) < 1e-4, (got_val, best_val)


def _packer(seq_len=32, max_crops=4, buffer_size=4):
    return DynamicPacker(
        buffer_size,
        [
            PackingConstraint("input_ids", seq_len, True, 1.0, max(1, seq_len // 512)),
            PackingConstraint("images", max_crops, False, 30.0, 1),
        ],
    )


def test_packer_buffers_until_full_then_respects_capacities():
    packer = _packer(seq_len=16, max_crops=4, buffer_size=4)
    outs = []
    for k in range(8):
        out = packer(_example(n_text=6, n_crops=1, tag=k))
        if out is not None:
            outs.append(out)
    # First 4 offers only fill the buffer.
    assert len(outs) >= 1
    for pack in outs:
        assert len(pack["input_ids"]) <= 16
        assert pack["images"].shape[0] <= 4


def test_packer_mixes_text_only_and_image_examples():
    packer = _packer(seq_len=32, max_crops=8, buffer_size=2)
    packer(_example(n_text=4, n_crops=2, tag=1))
    packer(_example(n_text=4, n_crops=0, tag=2))  # text-only NLP
    out = packer(_example(n_text=4, n_crops=1, tag=3))
    assert out is not None
    # Both buffered examples fit both budgets -> packed together (mm_olmo mixes
    # text-only and image examples; text-only rows contribute an empty crop array).
    assert len(out["input_ids"]) == 8
    assert out["images"].shape[0] == 2
    # Cross-example isolation ids present.
    assert "example_ids" in out and len(np.unique(out["example_ids"])) == 2


def test_packer_emits_oversized_example_alone():
    packer = _packer(seq_len=64, max_crops=2, buffer_size=4)
    out = packer(_example(n_text=4, n_crops=5, tag=9))  # 5 crops > capacity 2
    assert out is not None
    assert out["images"].shape[0] == 5  # emitted alone, not buffered forever


def test_packer_shortcuts_token_full_example():
    packer = _packer(seq_len=16, max_crops=8, buffer_size=4)
    out = packer(_example(n_text=16, n_crops=1, tag=7))
    assert out is not None and len(out["input_ids"]) == 16


def test_iter_dynamic_packs_flush_and_determinism():
    def stream():
        rng = np.random.RandomState(3)
        for k in range(40):
            yield _example(int(rng.randint(2, 10)), int(rng.randint(0, 3)), tag=k)

    packs1 = list(iter_dynamic_packs(stream(), 24, max_crops_per_pack=4, buffer_size=6))
    packs2 = list(iter_dynamic_packs(stream(), 24, max_crops_per_pack=4, buffer_size=6))
    # Everything is emitted (flush drains the buffer) and the result is deterministic.
    total = sum(len(p["input_ids"]) for p in packs1)
    assert total == sum(len(p["input_ids"]) for p in packs2)
    assert [p["input_ids"].tolist() for p in packs1] == [p["input_ids"].tolist() for p in packs2]
    n_examples = sum(len(np.unique(p["example_ids"])) for p in packs1)
    assert n_examples == 40
    for p in packs1:
        assert len(p["input_ids"]) <= 24
        assert p["images"].shape[0] <= 4 or len(np.unique(p["example_ids"])) == 1
