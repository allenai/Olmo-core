"""CPU tests for the Molmo2 stage-1 data pipeline: grounding format, branched-sequence
assembly, text-only handling, and the weighted mixture loader."""


import numpy as np

from olmo_core.data.multimodal import (
    MixtureDataLoader,
    MultimodalCollatorConfig,
    build_branched_sequence,
)
from olmo_core.data.multimodal.grounding import (
    format_points_tag,
    normalize_points,
    pointing_answer,
)
from olmo_core.data.multimodal.packing import (
    greedy_pack_indices,
    iter_packs,
    pack_examples,
)
from olmo_core.data.multimodal.prefetch import prefetch_map

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


def test_format_points_tag_sorted_and_clamped():
    pts = [[0.9, 0.1], [0.1, 0.9], [1.5, -0.2]]  # last is out of range -> clamped
    tag = format_points_tag(pts, "x")
    # sorted by (x, y): (0.1,0.9)->100 900, (0.9,0.1)->900 100, (1.0,0.0)->1000 000
    assert tag == '<points coords="1 1 100 900 2 900 100 3 1000 000">x</points>'


def test_pointing_answer_styles():
    pts = [[0.1, 0.2], [0.3, 0.4]]
    assert pointing_answer(pts, "cats", "pointing") == format_points_tag(pts, "cats")
    assert pointing_answer(pts, "cats", "point_count") == (
        f"Counting the {format_points_tag(pts, 'cats')} shows a total of 2."
    )
    assert pointing_answer([], "dogs", "pointing") == "There are none."
    assert pointing_answer([], "dogs", "point_count") == "There are none."


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


def test_mixture_data_loader_weighted_sampling(tmp_path):
    ds = [_FakeDataset(1000, 10), _FakeDataset(500, 20), _FakeDataset(200, 30)]
    weights = [0.6, 0.3, 0.1]
    coll = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()
    dl = MixtureDataLoader(
        ds, weights, coll, work_dir=str(tmp_path), global_batch_size=4 * _SEQ, seed=0
    )
    dl.reshuffle(epoch=1)
    order = dl._order
    assert order is not None
    counts = np.bincount([s for s, _ in order], minlength=3) / len(order)
    np.testing.assert_allclose(counts, weights, atol=0.05)

    batch = next(iter(dl))
    assert tuple(batch["input_ids"].shape) == (4, _SEQ)
    # all sources here are text-only -> the batch still emits a single dummy zero crop
    # (with all-(-1) pooled indices) so the vision/connector path runs on every rank,
    # keeping FSDP collectives in lockstep. Nothing is spliced (no <im_patch> tokens).
    assert tuple(batch["images"].shape) == (4, 1, 729, _PATCH_DIM)
    assert (batch["pooled_patches_idx"] == -1).all()


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


def _text_example(n_text=4, tag=3):
    L = n_text
    return dict(
        input_ids=np.array([1] + [tag] * (L - 1), dtype=np.int64),
        labels=np.full(L, tag, dtype=np.int64),
        loss_masks=np.ones(L, dtype=np.float32),
        position_ids=np.arange(L, dtype=np.int64),
        token_type_ids=np.zeros(L, dtype=np.int64),
        images=np.zeros((0, 729, _PATCH_DIM), dtype=np.float32),
        pooled_patches_idx=np.full((0, 4), -1, dtype=np.int64),
    )


def test_iter_packs_homogeneous_image_text():
    """Text-only and image examples must not share a pack (stage-1 NLP + vision)."""
    img = _img_example(n_text=2, n_crops=1, tag=5)  # len 3
    txt = _text_example(n_text=3, tag=9)  # len 3
    packs = list(iter_packs([txt, img, txt], seq_len=16))
    assert len(packs) == 3
    assert len(packs[0]["input_ids"]) == 3
    assert len(packs[1]["input_ids"]) == 3  # image example alone
    assert len(packs[2]["input_ids"]) == 3


def test_greedy_pack_indices():
    # next-fit: [3,3]->ok(6), +5 overflows 8 -> new group, +2 fits(7)
    assert greedy_pack_indices([3, 3, 5, 2], seq_len=8) == [[0, 1], [2, 3]]
    assert greedy_pack_indices([10], seq_len=8) == [[0]]  # over-length example alone


def test_greedy_pack_indices_crop_budget():
    # token budget allows all four, crop budget splits after two crops.
    assert greedy_pack_indices(
        [4, 4, 4, 4],
        seq_len=20,
        crop_counts=[1, 1, 1, 1],
        max_crops_per_pack=2,
    ) == [[0, 1], [2, 3]]


def test_iter_packs_crop_budget():
    """Packed sequences must not exceed max_crops_per_pack."""
    exs = [_img_example(n_text=2, n_crops=2, tag=i) for i in range(4)]  # len 4 each
    packs = list(iter_packs(exs, seq_len=32, max_crops_per_pack=4))
    assert len(packs) == 2
    for pack in packs:
        assert pack["images"].shape[0] <= 4
    assert packs[0]["images"].shape[0] == 4
    assert packs[1]["images"].shape[0] == 4


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


def test_prefetch_does_not_change_packing():
    # Packing output must be identical whether examples are loaded sync or prefetched.
    exs = [_img_example(n_text=3 + (i % 4), n_crops=1, tag=i + 2) for i in range(12)]
    get = lambda i: exs[i]  # noqa: E731
    sync = list(iter_packs((get(i) for i in range(12)), seq_len=12))
    pref = list(iter_packs(prefetch_map(get, iter(range(12)), num_workers=4), seq_len=12))
    assert len(sync) == len(pref)
    for a, b in zip(sync, pref):
        np.testing.assert_array_equal(a["input_ids"], b["input_ids"])
        np.testing.assert_array_equal(a["example_ids"], b["example_ids"])


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


def test_mixture_data_loader_normalizes_weights(tmp_path):
    ds = [_FakeDataset(10, 1), _FakeDataset(10, 2)]
    coll = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()
    dl = MixtureDataLoader(
        ds, [3.0, 1.0], coll, work_dir=str(tmp_path), global_batch_size=2 * _SEQ, seed=0
    )
    np.testing.assert_allclose(dl.weights, [0.75, 0.25])


# ---------------------------------------------------------------------------
# PixMoCap style_and_length_v2 conditioning (Gap 1 vs mm_olmo)
# ---------------------------------------------------------------------------


class _FakeTok:
    """Minimal tokenizer for CPU tests: records the prompts it templates."""

    eos_token_id = 1
    bos_token_id = 0

    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        prompt = messages[0]["content"]
        if prompt:
            self.prompts.append(prompt)
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def encode(self, text, add_special_tokens=False):
        return [(ord(c) % 90) + 10 for c in text]


def _pixmo_cap(mode, **kw):
    from olmo_core.data.multimodal.pixmo_cap import PixMoCapDatasetConfig

    cfg = PixMoCapDatasetConfig(
        dataset_path="synthetic", mode=mode, max_sequence_length=4096, seed=0, **kw
    )
    return cfg.build(_FakeTok())


def test_pixmo_cap_select_branches_styles():
    from olmo_core.data.multimodal.pixmo_cap import CAPTION_STYLE, TRANSCRIPT_STYLE

    row = {"caption": "a cat", "transcripts": ["spoken one", "spoken two"]}
    rng = np.random.RandomState(0)
    assert [s for s, _ in _pixmo_cap("caption")._select_branches(row, rng)] == [CAPTION_STYLE]
    assert [s for s, _ in _pixmo_cap("transcript")._select_branches(row, rng)] == [TRANSCRIPT_STYLE]
    both = _pixmo_cap("transcript_and_caption")._select_branches(row, rng)
    assert [s for s, _ in both] == [CAPTION_STYLE, TRANSCRIPT_STYLE]


def test_pixmo_cap_tag_is_the_whole_user_turn():
    ds = _pixmo_cap("transcript_and_caption", style_tag=True)
    seq = ds[0]
    # two branches -> subsegment ids present, two distinct annotations
    assert "subsegment_ids" in seq
    assert len(set(seq["subsegment_ids"].tolist())) == 3  # prefix + 2 branches
    assert ds.tokenizer.prompts == ["long_caption:", "transcript:"]


def test_pixmo_cap_tag_carries_no_instruction_and_no_length():
    """Train/test consistency: the stage-1 caption eval sends the bare ``long_caption:``, so
    training must send exactly that on every example -- no sampled instruction after it, and no
    length number in it."""
    from olmo_core.data.multimodal.pixmo_cap import CAPTION_PROMPTS, TRANSCRIPT_PROMPTS

    ds = _pixmo_cap("transcript_and_caption", style_tag=True)
    sentences = set(CAPTION_PROMPTS) | set(TRANSCRIPT_PROMPTS)
    seen = set()
    for i in range(24):
        ds.tokenizer.prompts.clear()
        _ = ds[i]
        for p in ds.tokenizer.prompts:
            assert not any(s in p for s in sentences), p
            seen.add(p)
    assert seen == {"long_caption:", "transcript:"}


def test_pixmo_cap_fixed_prompt_overrides_the_tag():
    ds = _pixmo_cap("caption", fixed_prompt="Describe this image.")
    _ = ds[0]
    assert ds.tokenizer.prompts == ["Describe this image."]  # verbatim, no style prefix


def test_pixmo_cap_tag_off():
    from olmo_core.data.multimodal.pixmo_cap import CAPTION_PROMPTS

    ds = _pixmo_cap("caption", style_tag=False)
    _ = ds[0]
    # prompt is sampled from the pool verbatim, with no "long_caption ...:" prefix
    assert all(not p.startswith("long_caption") for p in ds.tokenizer.prompts)
    assert all(p in CAPTION_PROMPTS for p in ds.tokenizer.prompts)


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
