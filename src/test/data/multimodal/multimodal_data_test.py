"""CPU tests for the Molmo2 stage-1 data pipeline: grounding format, branched-sequence
assembly, text-only handling, and the weighted mixture loader."""

import itertools
import json

import numpy as np
import pytest
import torch

from olmo_core.data.multimodal import (
    MixtureDataLoader,
    MultimodalCollatorConfig,
    build_branched_sequence,
)
from olmo_core.data.multimodal.grounding import format_points_tag, normalize_points
from olmo_core.data.multimodal.packing import (
    greedy_pack_indices,
    iter_packs,
    pack_examples,
)
from olmo_core.data.multimodal.prefetch import prefetch_map
from olmo_core.data.multimodal.rng import make_random_state
from olmo_core.exceptions import OLMoConfigurationError

_SEQ = 8
_PATCH_DIM = 14 * 14 * 3


# ---------------------------------------------------------------------------
# Grounding / point format
# ---------------------------------------------------------------------------


def test_format_points_tag_html_v2():
    # Two points, normalized; expect 0-1000 3-digit coords, sorted by (x, y), image idx 1.
    pts = [[0.075, 0.812], [0.0, 0.5]]
    tag = format_points_tag(pts, "cat")
    assert tag == '<points coords="1 1 000 500 2 075 812">cat</points>'


def test_normalize_points():
    # percent (point_scale=100) -> /100
    np.testing.assert_allclose(
        normalize_points(np.array([[50.0, 10.0]]), point_scale=100, image_size=None),
        [[0.5, 0.1]],
    )
    # pixel (point_scale=None) -> /image_size (w, h)
    np.testing.assert_allclose(
        normalize_points(np.array([[100.0, 50.0]]), point_scale=None, image_size=(200, 100)),
        [[0.5, 0.5]],
    )


# ---------------------------------------------------------------------------
# Branched (per-branch user turn) sequence assembly
# ---------------------------------------------------------------------------


def test_build_branched_sequence_two_branches():
    # prefix = BOS + 2 image tokens; 2 branches, each (user ctx 2 toks, answer 2 toks).
    out = build_branched_sequence(
        [100, 151938, 151937],
        [([10, 11], [20, 21]), ([12, 13], [30, 31])],
        eos_id=1,
    )
    assert out["input_ids"].tolist() == [100, 151938, 151937, 10, 11, 20, 21, 12, 13, 30, 31]
    # prefix 0,1,2 ; both branches start at position 3 (overlap, no carry-over)
    assert out["position_ids"].tolist() == [0, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6]
    assert out["subsegment_ids"].tolist() == [10000, 10000, 10000, 0, 0, 0, 0, 1, 1, 1, 1]
    # loss only where a response (or its EOS) is predicted, scaled by 1/sqrt(2)
    nz = out["loss_masks"] > 0
    assert nz.tolist() == [False, False, False, False, True, True, True, False, True, True, True]
    np.testing.assert_allclose(out["loss_masks"][nz], 1.0 / np.sqrt(2), rtol=1e-3)
    # segment ends predict EOS
    assert out["labels"][6] == 1 and out["labels"][10] == 1


def test_build_branched_sequence_single():
    out = build_branched_sequence([100, 151938], [([10, 11], [20, 21])], eos_id=1)
    assert "subsegment_ids" not in out  # single branch -> no subsegments
    assert out["position_ids"].tolist() == [0, 1, 2, 3, 4, 5]  # sequential
    assert out["loss_masks"].tolist() == [0, 0, 0, 1, 1, 1]  # loss on response + its EOS target


# ---------------------------------------------------------------------------
# Mixture loader
# ---------------------------------------------------------------------------


class _FakeDataset:
    """Tiny in-memory text-only dataset emitting the collator example dict."""

    def __init__(self, n: int, tag: int):
        self.n, self.tag = n, tag

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        L = 6
        return dict(
            input_ids=np.full(L, self.tag, dtype=np.int64),
            labels=np.full(L, -100, dtype=np.int64),
            loss_masks=np.ones(L, dtype=np.float32),
            position_ids=np.arange(L, dtype=np.int64),
            token_type_ids=np.zeros(L, dtype=np.int64),
            images=np.zeros((0, 729, _PATCH_DIM), dtype=np.float32),
            pooled_patches_idx=np.full((0, 4), -1, dtype=np.int64),
        )


class _CountingFakeDataset(_FakeDataset):
    """Fake dataset that makes each ref observable and counts preprocessing calls."""

    def __init__(self, n: int, tag: int):
        super().__init__(n, tag)
        self.loads = 0

    def __getitem__(self, i):
        self.loads += 1
        out = super().__getitem__(i)
        out["input_ids"] = np.full(len(out["input_ids"]), self.tag + i, dtype=np.int64)
        return out


class _FingerprintedFakeDataset(_FakeDataset):
    """Fake source implementing the preferred content-fingerprint protocol."""

    content_fingerprint_version = "fake-content-v1"

    def __init__(self, n: int, tag: int, content_fingerprint: str):
        super().__init__(n, tag)
        self.content_fingerprint = content_fingerprint
        # Ensure the explicit content protocol wins over the backwards-compatible fallback.
        self.fingerprint = "must-not-be-used"


class _FailingFakeDataset(_CountingFakeDataset):
    def __init__(self, n: int, tag: int):
        super().__init__(n, tag)
        self.fail_indices = set()

    def __getitem__(self, i):
        if i in self.fail_indices:
            self.loads += 1
            raise ValueError(f"synthetic failure for {i}")
        return super().__getitem__(i)


class _EpochAwareFakeDataset(_FakeDataset):
    def __init__(self, n: int, tag: int):
        super().__init__(n, tag)
        self.requests = []

    def get(self, i: int, epoch: int):
        self.requests.append((i, epoch))
        return super().__getitem__(i)


def test_mixture_data_loader_weighted_sampling(tmp_path):
    ds = [_FakeDataset(1000, 10), _FakeDataset(500, 20), _FakeDataset(200, 30)]
    weights = [0.6, 0.3, 0.1]
    coll = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()
    dl = MixtureDataLoader(
        ds, weights, coll, work_dir=str(tmp_path), global_batch_size=4 * _SEQ, seed=0
    )
    dl.reshuffle(epoch=1)
    order = dl._order
    counts = np.bincount([source for source, _, _ in order], minlength=3) / len(order)
    np.testing.assert_allclose(counts, weights, atol=0.05)

    batch = next(iter(dl))
    assert tuple(batch["input_ids"].shape) == (4, _SEQ)
    # all sources here are text-only -> the batch still emits a single dummy zero crop
    # (with all-(-1) pooled indices) so the vision/connector path runs on every rank,
    # keeping FSDP collectives in lockstep. Nothing is spliced (no <im_patch> tokens).
    assert tuple(batch["images"].shape) == (4, 1, 729, _PATCH_DIM)
    assert (batch["pooled_patches_idx"] == -1).all()


def test_mixture_data_loader_tracks_source_epoch_for_augmentation(tmp_path):
    dataset = _EpochAwareFakeDataset(2, 10)
    collator = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()
    loader = MixtureDataLoader(
        [dataset],
        [1.0],
        collator,
        work_dir=str(tmp_path),
        global_batch_size=2 * _SEQ,
        seed=3,
        pack=True,
        pack_max_crops=1,
        pack_buffer_size=4,
        continuous_stream=True,
    )
    refs = list(itertools.islice(loader._rank_refs_from_cursor(), 6))

    assert [source_epoch for _, _, source_epoch in refs] == [0, 0, 1, 1, 2, 2]
    for ref in refs:
        loader._try_load_example(ref)
    assert [epoch for _, epoch in dataset.requests] == [0, 0, 1, 1, 2, 2]


def test_mixture_data_loader_default_mode_loads_with_getitem(tmp_path):
    # The epoch-shuffled default mode reproduces vision's loader: ``dataset[index]``, no
    # source epochs, even for datasets that also expose ``get(index, epoch)``.
    dataset = _EpochAwareFakeDataset(2, 10)
    collator = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()
    loader = MixtureDataLoader(
        [dataset],
        [1.0],
        collator,
        work_dir=str(tmp_path),
        global_batch_size=2 * _SEQ,
        epoch_instances=6,
        seed=3,
    )
    loader.reshuffle(epoch=1)

    assert [source_epoch for _, _, source_epoch in loader._order] == [0] * 6
    for ref in loader._order:
        loader._try_load_example(ref)
    assert dataset.requests == []
    assert (loader.pack_buffer_size, loader.pack_image_weight) == (48, 30.0)
    assert loader.continuous_stream is False and loader.total_batches == 3


# ---------------------------------------------------------------------------
# Sequence packing
# ---------------------------------------------------------------------------


def _img_example(n_text: int, n_crops: int, tag: int):
    L = n_crops + n_text  # n_crops <im_patch> tokens (id 1) + text
    return dict(
        input_ids=np.array([1] * n_crops + [tag] * n_text, dtype=np.int64),
        labels=np.full(L, tag, dtype=np.int64),
        loss_masks=np.ones(L, dtype=np.float32),
        position_ids=np.arange(L, dtype=np.int64),
        token_type_ids=np.array([1] * n_crops + [0] * n_text, dtype=np.int64),
        images=np.full((n_crops, 729, _PATCH_DIM), float(tag), dtype=np.float32),
        pooled_patches_idx=np.arange(n_crops * 4).reshape(n_crops, 4).astype(np.int64),
    )


def _text_example(n_text: int = 4, tag: int = 3):
    return dict(
        input_ids=np.full(n_text, tag, dtype=np.int64),
        labels=np.full(n_text, tag, dtype=np.int64),
        loss_masks=np.ones(n_text, dtype=np.float32),
        position_ids=np.arange(n_text, dtype=np.int64),
        token_type_ids=np.zeros(n_text, dtype=np.int64),
        images=np.zeros((0, 729, _PATCH_DIM), dtype=np.float32),
        pooled_patches_idx=np.full((0, 4), -1, dtype=np.int64),
    )


def test_multimodal_collator_marks_only_retained_real_tokens_for_routing():
    collator = MultimodalCollatorConfig(
        pad_token_id=0, pad_sequence_length=8, batch_metadata=True
    ).build()
    batch = collator([_text_example(n_text=3), _text_example(n_text=10)])

    assert batch["router_token_mask"].dtype == torch.bool
    assert batch["router_token_mask"].tolist() == [
        [True, True, True, False, False, False, False, False],
        [True, True, True, True, True, True, True, True],
    ]
    assert batch["image_crop_counts"].tolist() == [0, 0]
    assert batch["pooled_token_counts"].tolist() == [0, 0]


def test_greedy_pack_indices():
    # next-fit: [3,3]->ok(6), +5 overflows 8 -> new group, +2 fits(7)
    assert greedy_pack_indices([3, 3, 5, 2], seq_len=8) == [[0, 1], [2, 3]]
    assert greedy_pack_indices([10], seq_len=8) == [[0]]  # over-length example alone


def test_iter_packs_keeps_image_and_text_examples_separate():
    image = _img_example(n_text=2, n_crops=1, tag=5)
    text = _text_example(n_text=3, tag=7)
    packs = list(iter_packs([image, text, image], seq_len=32))
    assert len(packs) == 3
    assert packs[0]["images"].shape[0] == 1
    assert packs[1]["images"].shape[0] == 0
    assert packs[2]["images"].shape[0] == 1


def test_iter_packs_buffered_solver_improves_fill_and_flushes():
    examples = [_text_example(n_text=n, tag=i + 2) for i, n in enumerate([6, 6, 4, 4])]
    packs = list(
        iter_packs(
            examples,
            seq_len=10,
            max_crops_per_pack=1,
            buffer_size=4,
        )
    )

    assert [len(pack["input_ids"]) for pack in packs] == [10, 10]
    assert sum(len(np.unique(pack["example_ids"])) for pack in packs) == len(examples)
    np.testing.assert_array_equal(
        np.sort(np.concatenate([pack["input_ids"] for pack in packs])),
        np.sort(np.concatenate([example["input_ids"] for example in examples])),
    )


def test_iter_packs_buffered_solver_respects_crop_budget_and_mixes_modalities():
    image_a = _img_example(n_text=2, n_crops=2, tag=5)
    text = _text_example(n_text=6, tag=7)
    image_b = _img_example(n_text=2, n_crops=2, tag=9)
    image_a["_source_name"] = "image-a"
    text["_source_name"] = "text"
    image_b["_source_name"] = "image-b"

    packs = list(
        iter_packs(
            [image_a, text, image_b],
            seq_len=10,
            max_crops_per_pack=2,
            buffer_size=3,
        )
    )

    assert all(len(pack["input_ids"]) <= 10 for pack in packs)
    assert all(pack["images"].shape[0] <= 2 for pack in packs)
    assert any(set(pack["pack_source_names"]) == {"image-a", "text"} for pack in packs)


def test_pack_examples_concat_and_offsets():
    a = _img_example(n_text=3, n_crops=1, tag=5)  # len 4, 1 crop
    b = _img_example(n_text=2, n_crops=2, tag=7)  # len 4, 2 crops
    a["_source_name"] = "pixmo_points_train"
    b["_source_name"] = "pixmo_count_train"
    packed = pack_examples([a, b])

    assert packed["pack_source_names"] == ["pixmo_points_train", "pixmo_count_train"]

    assert packed["input_ids"].tolist() == [1, 5, 5, 5, 1, 1, 7, 7]
    assert packed["position_ids"].tolist() == [
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        3,
    ]  # positions reset per example
    assert packed["example_ids"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    # images concatenated along the crop axis (1 + 2 = 3 crops)
    assert packed["images"].shape == (3, 729, _PATCH_DIM)
    # b's pooled indices are offset by a's crop-patch count (1 crop * 729 patches)
    np.testing.assert_array_equal(packed["pooled_patches_idx"][0], [0, 1, 2, 3])  # a
    np.testing.assert_array_equal(packed["pooled_patches_idx"][1], np.arange(4) + 729)  # b crop 0
    np.testing.assert_array_equal(
        packed["pooled_patches_idx"][2], np.arange(4, 8) + 729
    )  # b crop 1


def test_prefetch_map_order_and_completeness():
    import time

    def slow(x):
        time.sleep(0.001 * ((x * 7) % 5))  # uneven work so threads finish out of order
        return x * x

    items = list(range(50))
    for workers in (0, 1, 4):
        out = list(prefetch_map(slow, iter(items), num_workers=workers, max_in_flight=8))
        assert out == [x * x for x in items]  # order preserved, nothing dropped


def test_mixture_data_loader_packs(tmp_path):
    ds = [_FakeDataset(200, 10), _FakeDataset(100, 20)]
    coll = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()
    dl = MixtureDataLoader(
        ds, [0.5, 0.5], coll, work_dir=str(tmp_path), global_batch_size=2 * _SEQ, seed=0, pack=True
    )
    dl.reshuffle(epoch=1)
    batch = next(iter(dl))
    # _FakeDataset emits length-6 text-only examples; with _SEQ=8 only one fits per pack.
    assert tuple(batch["input_ids"].shape) == (2, _SEQ)
    assert "example_ids" in batch  # packing marks example membership


def test_mixture_data_loader_buffered_packing(tmp_path):
    ds = [_FakeDataset(200, 10), _FakeDataset(100, 20)]
    coll = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()
    dl = MixtureDataLoader(
        ds,
        [0.5, 0.5],
        coll,
        work_dir=str(tmp_path),
        global_batch_size=2 * _SEQ,
        seed=0,
        pack=True,
        pack_max_crops=1,
        pack_buffer_size=4,
        continuous_stream=True,
    )
    dl.reshuffle(epoch=1)
    assert dl.total_batches is None
    batch = next(iter(dl))
    assert tuple(batch["input_ids"].shape) == (2, _SEQ)
    assert "example_ids" in batch


def test_mixture_data_loader_buffered_packing_resumes_exactly(tmp_path):
    datasets = [_FakeDataset(200, 10), _FakeDataset(100, 20)]
    collator = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()

    def build_loader(work_dir, prefetch_workers):
        return MixtureDataLoader(
            datasets,
            [0.5, 0.5],
            collator,
            work_dir=work_dir,
            global_batch_size=2 * _SEQ,
            seed=17,
            pack=True,
            pack_max_crops=1,
            pack_buffer_size=4,
            continuous_stream=True,
            prefetch_workers=prefetch_workers,
        )

    for original_workers, restored_workers in ((0, 0), (0, 4), (4, 0), (4, 4)):
        original = build_loader(tmp_path / f"original-{original_workers}", original_workers)
        original.reshuffle(epoch=3)
        original_iter = iter(original)
        next(original_iter)
        state = original.state_dict()
        expected = next(original_iter)
        original_iter.close()

        restored = build_loader(tmp_path / f"restored-{restored_workers}", restored_workers)
        restored.load_state_dict(state)
        restored.reshuffle()
        restored_iter = iter(restored)
        actual = next(restored_iter)
        restored_iter.close()

        for key in ("input_ids", "example_ids", "loss_masks", "position_ids"):
            np.testing.assert_array_equal(actual[key], expected[key])


def test_mixture_data_loader_v5_validates_source_content_fingerprints(tmp_path):
    collator = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()

    def build_loader(work_dir, fingerprint):
        return MixtureDataLoader(
            [_FingerprintedFakeDataset(200, 1000, fingerprint), _FakeDataset(100, 2000)],
            [0.5, 0.5],
            collator,
            work_dir=work_dir,
            global_batch_size=2 * _SEQ,
            seed=29,
            pack=True,
            pack_max_crops=1,
            pack_buffer_size=4,
            continuous_stream=True,
            dataset_names=["native-replay", "caption"],
        )

    original = build_loader(tmp_path / "original", "content-a")
    original.reshuffle(epoch=2)
    original_iter = iter(original)
    next(original_iter)
    state = original.state_dict()
    expected = next(original_iter)
    original_iter.close()

    fingerprints = state["packing_state"]["dataset_fingerprints"]
    assert fingerprints[0]["type"].endswith("._FingerprintedFakeDataset")
    assert fingerprints[0]["version"] == "fake-content-v1"
    assert fingerprints[0]["value"] == "content-a"
    assert fingerprints[1] is None

    restored = build_loader(tmp_path / "restored", "content-a")
    restored.load_state_dict(state)
    restored.reshuffle()
    restored_iter = iter(restored)
    actual = next(restored_iter)
    restored_iter.close()
    np.testing.assert_array_equal(actual["input_ids"], expected["input_ids"])

    changed = build_loader(tmp_path / "changed", "content-b")
    changed.load_state_dict(state)
    changed.reshuffle()
    with pytest.raises(
        OLMoConfigurationError,
        match="dataset content fingerprint changed for source 'native-replay'",
    ):
        next(iter(changed))


def test_mixture_data_loader_prefetch_skips_errors_in_reference_order(tmp_path):
    collator = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()

    def build_loader(work_dir, workers):
        dataset = _FailingFakeDataset(200, 1000)
        loader = MixtureDataLoader(
            [dataset],
            [1.0],
            collator,
            work_dir=work_dir,
            global_batch_size=2 * _SEQ,
            seed=31,
            pack=True,
            pack_max_crops=1,
            pack_buffer_size=4,
            continuous_stream=True,
            prefetch_workers=workers,
        )
        loader.reshuffle(epoch=2)
        refs = list(itertools.islice(loader._rank_refs_from_cursor(), 3))
        dataset.fail_indices = {refs[0][1], refs[2][1]}
        return loader

    sync = build_loader(tmp_path / "sync", 0)
    sync_iter = iter(sync)
    expected = next(sync_iter)
    sync_state = sync.state_dict()
    sync_iter.close()

    threaded = build_loader(tmp_path / "threaded", 4)
    threaded_iter = iter(threaded)
    actual = next(threaded_iter)
    threaded_state = threaded.state_dict()
    threaded_iter.close()

    np.testing.assert_array_equal(actual["input_ids"], expected["input_ids"])
    assert sync_state["total_data_errors"] == threaded_state["total_data_errors"] == 2
    assert (
        sync_state["packing_state"]["refs_consumed"]
        == threaded_state["packing_state"]["refs_consumed"]
    )


@pytest.mark.parametrize(
    ("max_consecutive", "max_total"),
    [(2, 10), (10, 2)],
)
def test_mixture_data_loader_error_limits_allow_n_and_fail_on_n_plus_one(
    tmp_path, max_consecutive, max_total
):
    collator = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()
    loader = MixtureDataLoader(
        [_FakeDataset(10, 1)],
        [1.0],
        collator,
        work_dir=tmp_path,
        global_batch_size=2 * _SEQ,
        max_consecutive_data_errors=max_consecutive,
        max_total_data_errors=max_total,
        dataset_names=["academic-test"],
    )

    limit = min(max_consecutive, max_total)
    for index in range(limit):
        loader._handle_data_error((0, index, 0), ValueError(f"failure {index}"))
    with pytest.raises(ValueError, match=f"failure {limit}"):
        loader._handle_data_error((0, limit, 0), ValueError(f"failure {limit}"))

    assert loader.total_data_errors == limit + 1


def test_mixture_data_loader_normalizes_weights(tmp_path):
    ds = [_FakeDataset(10, 1), _FakeDataset(10, 2)]
    coll = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()
    dl = MixtureDataLoader(
        ds, [3.0, 1.0], coll, work_dir=str(tmp_path), global_batch_size=2 * _SEQ, seed=0
    )
    np.testing.assert_allclose(dl.weights, [0.75, 0.25])


def test_buffered_mixture_reference_stream_matches_molmo2(tmp_path):
    sizes = [7, 5, 11]
    weights = [0.6, 0.3, 0.1]
    seed = 95818
    world_size = 4
    n_per_rank = 80

    def molmo2_refs():
        rates = np.asarray(weights, dtype=np.float64)
        rates = np.asarray(rates / rates.sum(), dtype=np.float32)
        rng = np.random.RandomState(seed)
        counts = np.zeros(len(sizes), dtype=np.int64)
        shuffled = [(None, None) for _ in sizes]
        while True:
            source = int(rng.choice(len(sizes), p=rates))
            source_count = int(counts[source])
            counts[source] += 1
            source_epoch = source_count // sizes[source]
            shuffled_for, order = shuffled[source]
            if shuffled_for != source_epoch:
                order = np.arange(sizes[source], dtype=np.int32)
                make_random_state(seed, source_epoch, 1).shuffle(order)
                shuffled[source] = (source_epoch, order)
            yield source, int(order[source_count % sizes[source]]), source_epoch

    expected = list(itertools.islice(molmo2_refs(), world_size * n_per_rank))
    collator = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()
    actual_by_rank = []
    for rank in range(world_size):
        loader = MixtureDataLoader(
            [_FakeDataset(size, 10 + i) for i, size in enumerate(sizes)],
            weights,
            collator,
            work_dir=tmp_path / str(rank),
            global_batch_size=world_size * _SEQ,
            seed=seed,
            pack=True,
            pack_max_crops=1,
            pack_buffer_size=4,
            continuous_stream=True,
            dp_world_size=world_size,
            dp_rank=rank,
        )
        actual_by_rank.append(list(itertools.islice(loader._rank_refs_from_cursor(), n_per_rank)))
        resumed = list(itertools.islice(loader._rank_refs_from_cursor(17), 10))
        assert resumed == expected[rank + 17 * world_size :: world_size][:10]

    actual = [actual_by_rank[rank][i] for i in range(n_per_rank) for rank in range(world_size)]
    assert actual == expected


# ---------------------------------------------------------------------------
# PixMoCap style_and_length_v2 conditioning (Gap 1 vs mm_olmo)
# ---------------------------------------------------------------------------


class _FakeTok:
    """Minimal tokenizer for CPU tests: records the prompts it templates."""

    eos_token_id = 1
    bos_token_id = 0

    def __init__(self, record_encoded: bool = True):
        # The qwen3 layout templates prompts through ``apply_chat_template``; the document
        # layout encodes them as plain text, so those tests also record ``encode`` calls.
        self.prompts = []
        self.record_encoded = record_encoded

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        prompt = messages[0]["content"]
        if prompt:
            self.prompts.append(prompt)
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def encode(self, text, add_special_tokens=False):
        if self.record_encoded and not text.startswith(" "):
            self.prompts.append(text)
        return [(ord(c) % 90) + 10 for c in text]


def _pixmo_cap(mode, **kw):
    from olmo_core.data.multimodal.pixmo_cap import PixMoCapDatasetConfig

    kw.setdefault("message_format", "document")
    cfg = PixMoCapDatasetConfig(
        dataset_path="synthetic", mode=mode, max_sequence_length=4096, seed=0, **kw
    )
    return cfg.build(_FakeTok(record_encoded=kw["message_format"] == "document"))


def test_pixmo_cap_can_fail_closed_when_named_split_is_required():
    with pytest.raises(ValueError, match="does not provide named splits"):
        _pixmo_cap("caption", split="validation", require_split=True)


@pytest.mark.parametrize("mode", ["caption", "transcript_and_caption"])
@pytest.mark.parametrize("message_format", ["document"])
def test_caption_message_weight_only_scales_response_masks(monkeypatch, mode, message_format):
    from PIL import Image

    baseline = _pixmo_cap(mode, message_format=message_format)
    weighted = _pixmo_cap(mode, message_format=message_format, message_weight=2.0)
    if mode == "sft_demo":
        for dataset in (baseline, weighted):
            monkeypatch.setattr(dataset, "_get_row", lambda _index: {"caption": "A red square."})
            monkeypatch.setattr(dataset, "_load_image", lambda _row: Image.new("RGB", (32, 32)))
    base_example, weighted_example = baseline[0], weighted[0]
    for key, value in base_example.items():
        if isinstance(value, np.ndarray):
            expected = value * 2 if key == "loss_masks" else value
            np.testing.assert_array_equal(weighted_example[key], expected)


def test_pixmo_cap_select_branches_styles():
    from olmo_core.data.multimodal.pixmo_cap import CAPTION_STYLE, TRANSCRIPT_STYLE

    row = {"caption": "a cat", "transcripts": ["spoken one", "spoken two"]}
    rng = np.random.RandomState(0)
    assert [s for s, _ in _pixmo_cap("caption")._select_branches(row, rng)] == [CAPTION_STYLE]
    assert [s for s, _ in _pixmo_cap("transcript")._select_branches(row, rng)] == [TRANSCRIPT_STYLE]
    both = _pixmo_cap("transcript_and_caption")._select_branches(row, rng)
    assert [s for s, _ in both] == [CAPTION_STYLE, TRANSCRIPT_STYLE]


def test_pixmo_cap_tag_is_the_whole_user_turn():
    ds = _pixmo_cap("transcript_and_caption", style_tag=True, message_format="qwen3")
    seq = ds[0]
    # two branches -> subsegment ids present, two distinct annotations
    assert "subsegment_ids" in seq
    assert len(set(seq["subsegment_ids"].tolist())) == 3  # prefix + 2 branches
    assert ds.tokenizer.prompts == ["long_caption:", "transcript:"]


def test_pixmo_cap_transcript_fallback_is_backward_compatible_and_can_be_disabled():
    from olmo_core.data.multimodal.pixmo_cap import CAPTION_STYLE

    row = {"caption": "caption fallback", "transcripts": []}
    rng = np.random.RandomState(0)

    assert _pixmo_cap("transcript")._select_branches(row, rng) == [
        (CAPTION_STYLE, "caption fallback")
    ]
    strict = _pixmo_cap("transcript", require_transcript=True)
    with pytest.raises(ValueError, match="requires at least one non-blank transcript"):
        strict._select_branches(row, rng)
    with pytest.raises(ValueError, match="requires at least one non-blank transcript"):
        strict._select_branches({"caption": "caption", "transcripts": ["", "  "]}, rng)


def test_pixmo_cap_validates_strict_transcript_completeness_without_loading_images(tmp_path):
    jsonl_path = tmp_path / "pixmo-cap.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "image": "missing-image-a.png",
                        "caption": "caption a",
                        "transcripts": ["spoken a"],
                    }
                ),
                json.dumps(
                    {
                        "image": "missing-image-b.png",
                        "caption": "caption b",
                        "transcripts": ["", "  "],
                    }
                ),
                json.dumps({"image": "missing-image-c.png", "caption": "caption c"}),
            ]
        )
        + "\n"
    )
    from olmo_core.data.multimodal.pixmo_cap import PixMoCapDatasetConfig

    dataset = PixMoCapDatasetConfig(
        dataset_path=str(jsonl_path),
        mode="transcript",
        require_transcript=True,
        message_format="document",
    ).build(_FakeTok())

    with pytest.raises(ValueError, match=r"has 2 invalid annotation rows out of 3.*1:.*2:"):
        dataset.validate_required_annotations()


def test_pixmo_cap_fixed_prompt_overrides_the_tag():
    ds = _pixmo_cap("caption", fixed_prompt="Describe this image.")
    _ = ds[0]
    assert ds.tokenizer.prompts == ["Describe this image."]  # verbatim, no style prefix


# ---------------------------------------------------------------------------
# Truncated/corrupt image tolerance (multi-node run robustness)
# ---------------------------------------------------------------------------


def test_truncated_image_preprocesses_without_raising():
    """A truncated PixMo image must not raise (PIL OSError) — it would crash a data-worker
    thread and, under distributed packing, hang the other ranks into a NCCL watchdog abort."""
    import io

    import torch
    from PIL import Image

    from olmo_core.nn.vision.molmo2_image_processor import preprocess_image_molmo2

    buf = io.BytesIO()
    Image.fromarray((np.random.rand(64, 96, 3) * 255).astype("uint8")).save(buf, format="JPEG")
    truncated = Image.open(io.BytesIO(buf.getvalue()[:-200]))  # drop trailing bytes
    crops, pooled, grid = preprocess_image_molmo2(
        truncated, dtype=torch.float32, device=torch.device("cpu"), max_crops=8
    )
    assert crops.shape[0] == 1 and grid.shape == (4,)
