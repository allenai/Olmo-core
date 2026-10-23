"""CPU tests for the audited PixMo pointing sources (mm_olmo ``PixMoPointV2`` /
``PixMoCountConfigV2`` ports): the ``aux_*`` marker styles, build-time row selection, the
per-image message assembly with sub-sampled absence queries, per-branch negative weights,
pixel-space count points, and the Molmo2-Stage1 ``pointing_data`` selector."""

import importlib.util
import sys
from typing import Any, cast

import numpy as np
import pytest
from datasets import Dataset, DatasetDict

from olmo_core.data.multimodal import (
    PixMoCountV2DatasetConfig,
    PixMoPointsV2DatasetConfig,
)
from olmo_core.data.multimodal.message_weight import ATTEND_ALL_SUBSEGMENT_ID
from olmo_core.data.multimodal.pixmo_points_v2 import STAGE1_PROMPT_FAMILY
from olmo_core.data.multimodal.sft_formatter import SftFormatter, base_pointing_style
from olmo_core.exceptions import OLMoConfigurationError

STAGE1 = STAGE1_PROMPT_FAMILY
POINTS = [{"x": 10.5, "y": 20.5}, {"x": 30.0, "y": 40.0}]


class _FakeTok:
    """Minimal tokenizer for CPU tests (chat template + char-level encode)."""

    eos_token_id = 1
    bos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def encode(self, text, add_special_tokens=False):
        return [(ord(c) % 90) + 10 for c in text]


REFUSAL_IDS = _FakeTok().encode("There are none.")


def _contains(seq, sub) -> bool:
    seq, sub = list(seq), list(sub)
    return any(seq[i : i + len(sub)] == sub for i in range(len(seq) - len(sub) + 1))


# ---------------------------------------------------------------------------
# aux_* marker styles in the formatter
# ---------------------------------------------------------------------------


def _turn(style, points, **family):
    fmt = SftFormatter(seed=0, **family)
    ex = {"style": style, "label": "Leather Earmuff", "points": points, "point_scale": 100}
    return fmt.format_turns(ex, index=3, rng=np.random.RandomState(0))[0]


@pytest.mark.parametrize("style", ["pointing", "point_count"])
def test_aux_style_differs_from_base_only_in_the_style_token(style):
    """mm_olmo data_formatter.py:1189 / 1824-1825: same prompt body, same answer."""
    user, target = _turn(f"aux_{style}", POINTS, **STAGE1)
    base_user, base_target = _turn(style, POINTS, **STAGE1)
    assert user == f"aux_{style}: leather earmuff"
    assert base_user == f"{style}: leather earmuff"
    assert target == base_target
    assert base_pointing_style(f"aux_{style}") == style


def test_aux_point_count_answer_counts():
    _, target = _turn("aux_point_count", POINTS, **STAGE1)
    assert target == (
        'Counting the <points coords="1 1 105 205 2 300 400">leather earmuff</points> '
        "shows a total of 2."
    )


def test_aux_style_with_no_points_abstains():
    _, target = _turn("aux_pointing", [], **STAGE1)
    assert target == "There are none."


# ---------------------------------------------------------------------------
# Fixtures: tiny on-disk copies of the two v2 layouts
# ---------------------------------------------------------------------------


def _image(tmp_path):
    from PIL import Image

    path = tmp_path / "img.png"
    Image.new("RGB", (64, 48), color=(200, 30, 30)).save(path)
    return str(path)


def _anno(label, points, audit):
    return {
        "audit_result": audit,
        "label": label,
        "mask_example_id": None,
        "mask_f1": None,
        "masks": None,
        "points": points,
    }


def _write_points_v2(tmp_path):
    """Four images: a pointing row with a passed, a failed and an empty annotation plus
    negatives; a counting row whose only annotation failed the audit; a row whose only
    annotation has a blank label (never trainable); and a pointing row pairing a good
    annotation with a null-``points`` one (the row filter admits it, so ``keep`` must reject
    the null annotation rather than let ``format_row`` raise on it)."""
    image = _image(tmp_path)
    rows = {
        "image": [image] * 4,
        "image_url": ["u0", "u1", "u2", "u3"],
        "source": ["pointing", "counting", "pointing", "pointing"],
        "annotations": [
            [
                _anno("Red Cup", [[10.0, 20.0, 1.5], [30.0, 40.0, -1.0]], "correct"),
                _anno("Spoon", [[50.0, 50.0, 2.0]], "clear_error"),
                _anno("Fork", [], "n/a"),
            ],
            [_anno("Books", [[1.0, 2.0, 0.5], [3.0, 4.0, 0.5], [5.0, 6.0, 0.5]], "error")],
            [_anno("   ", [[1.0, 1.0, 0.0]], "correct")],
            [
                _anno("Kettle", [[11.0, 12.0, 1.0]], "correct"),
                _anno("Ghost", None, "correct"),
            ],
        ],
        "min_points": [0, 3, 1, 1],
        "max_points": [2, 3, 1, 1],
        "min_masks": [0, 0, 0, 0],
        "paired_negatives": [["Mug"], ["Magazines"], [], []],
        "negatives": [[], [], [], []],
        "easy_negatives": [["giraffe", "piano", "sailboat"], ["kite"], [], []],
        "rare_negatives": [[], [], [], []],
        "paired_negatives_v2": [["Mug", "Glass"], ["Magazines"], [], []],
    }
    path = tmp_path / "points-v2"
    Dataset.from_dict(rows).save_to_disk(str(path))
    return str(path)


def _write_count_v2(tmp_path):
    image = _image(tmp_path)

    def row(label, points, audit):
        count = 0 if points is None else len(points)
        return [{"audit_result": audit, "count": count, "label": label, "points": points}]

    rows = {
        "image_url": ["u0", "u1", "u2", "u3", "u4", "u5"],
        "image_sha256": ["s0", "s1", "s2", "s3", "s4", "s5"],
        "image": [image] * 6,
        "points": [
            row("ties", [[32.0, 24.0], [16.0, 12.0]], "correct"),
            row("Cats", [[10.0, 10.0]], "clear_error"),
            row("dogs", [], "n/a"),
            # Unusable annotations: a blank label would silently train an empty-named <points>
            # tag, and null label / points raise deep in formatting. All three are dropped.
            row("   ", [[1.0, 2.0]], "correct"),
            row(None, [[3.0, 4.0]], "correct"),
            row("kites", None, "correct"),
        ],
    }
    path = tmp_path / "count-v2"
    DatasetDict({"train": Dataset.from_dict(rows)}).save_to_disk(str(path))
    return str(path)


def _points_cfg(path, **kw):
    kw.setdefault("style", ("pointing",))
    kw.setdefault("heldout_paths", ())  # the defaults are weka paths
    return PixMoPointsV2DatasetConfig(dataset_path=path, max_crops=1, **kw)


def _count_cfg(path, **kw):
    kw.setdefault("style", ("pointing",))
    kw.setdefault("heldout_paths", ())  # the defaults are weka paths
    return PixMoCountV2DatasetConfig(dataset_path=path, max_crops=1, **kw)


def _write_heldout(tmp_path, name, column, *, train=(), validation=(), test=(), flat=None):
    """A held-out dataset fixture: a ``DatasetDict`` of id-only splits, or a flat eval set."""
    path = tmp_path / name
    if flat is not None:
        Dataset.from_dict({column: list(flat)}).save_to_disk(str(path))
    else:
        splits = {"train": train, "validation": validation, "test": test}
        DatasetDict(
            {k: Dataset.from_dict({column: list(v)}) for k, v in splits.items() if v}
        ).save_to_disk(str(path))
    return str(path)


def _labels(messages):
    return [m["label"] for m in messages]


# ---------------------------------------------------------------------------
# PixMoPointsV2: selection
# ---------------------------------------------------------------------------


def test_points_v2_index_keeps_rows_with_a_trainable_annotation(tmp_path):
    path = _write_points_v2(tmp_path)
    tok = _FakeTok()
    assert len(_points_cfg(path).build(tok)) == 3  # the blank-label row is dropped
    assert len(_points_cfg(path, kind="basic").build(tok)) == 2
    assert len(_points_cfg(path, kind="high_frequency").build(tok)) == 1
    # Row 1's only annotation failed the audit, so the audit filter drops the whole row.
    assert len(_points_cfg(path, filter_audit=True).build(tok)) == 2
    # min/max point bounds apply per annotation.
    assert len(_points_cfg(path, min_points=3).build(tok)) == 1
    assert len(_points_cfg(path, max_points=2, min_points=1).build(tok)) == 2


def test_points_v2_null_points_annotation_is_never_formatted(tmp_path):
    """The row filter tolerates a null ``points`` (it maps to -1), so ``keep`` has to reject it
    too -- otherwise the row is admitted and ``format_row`` raises ``TypeError`` on ``len(None)``,
    which the loader absorbs as a data error and counts against its abort threshold."""
    path = _write_points_v2(tmp_path)
    ds = _points_cfg(path, n_easy_samples=0, p_paired_negatives=0.0).build(_FakeTok())
    row = next(ds._data[int(i)] for i in ds._index if ds._data[int(i)]["image_url"] == "u3")
    assert [a["label"] for a in row["annotations"]] == ["Kettle", "Ghost"]
    assert ds.keep(row["annotations"][1]) is False
    messages, _ = ds.format_row(row, np.random.RandomState(0))  # must not raise
    assert _labels(messages) == ["Kettle"]


def test_points_v2_config_validation(tmp_path):
    with pytest.raises(OLMoConfigurationError):
        PixMoPointsV2DatasetConfig(kind="nope").validate()
    with pytest.raises(OLMoConfigurationError):
        PixMoPointsV2DatasetConfig(style=()).validate()
    with pytest.raises(OLMoConfigurationError):
        PixMoPointsV2DatasetConfig(p_paired_negatives=1.5).validate()


@pytest.mark.parametrize("config_cls", [PixMoPointsV2DatasetConfig, PixMoCountV2DatasetConfig])
@pytest.mark.parametrize("knob", ["prompt_templates", "system_prompt", "p_high_res"])
def test_v2_configs_are_stage1_only(config_cls, knob):
    """The prompt family is fixed to the released pretrain's (bare label behind ``"<style>:"``),
    so no default or override can change it."""
    assert STAGE1_PROMPT_FAMILY == {
        "prompt_templates": "none",
        "system_prompt": "style_and_length_v2",
    }
    assert knob not in config_cls.__dataclass_fields__
    assert config_cls().loss_token_weighting == "none"
    with pytest.raises(Exception):
        config_cls().merge([f"{knob}=x"])


def test_points_v2_config_tuple_fields_merge_from_cli():
    cfg = PixMoPointsV2DatasetConfig().merge(
        ["audit_style=[aux_point_count,aux_pointing]", "style=[pointing]", "filter_audit=true"]
    )
    assert cfg.audit_style == ("aux_point_count", "aux_pointing")
    assert cfg.style == ("pointing",)
    assert cfg.filter_audit is True


# ---------------------------------------------------------------------------
# Train-only guard: no image of a held-out set is ever trained on
# ---------------------------------------------------------------------------


def test_points_v2_drops_images_of_heldout_sets(tmp_path):
    path = _write_points_v2(tmp_path)
    tok = _FakeTok()
    # `u0` is another source's validation image, `u3` is in a flat eval set. A held-out
    # dataset's own `train` split is not held out, so `u1` stays.
    v1 = _write_heldout(tmp_path, "v1", "image_url", train=["u1"], validation=["u0"])
    ev = _write_heldout(tmp_path, "eval", "image_url", flat=["u3", "elsewhere"])
    ds = _points_cfg(path, heldout_paths=(v1, ev)).build(tok)
    assert [ds._data[int(i)]["image_url"] for i in ds._index] == ["u1"]
    assert ds.n_heldout_dropped == 2
    assert len(_points_cfg(path).build(tok)) == 3  # unguarded: the same build keeps all three


def test_count_v2_drops_images_of_heldout_sets(tmp_path):
    path = _write_count_v2(tmp_path)
    tok = _FakeTok()
    heldout = _write_heldout(
        tmp_path, "count", "image_sha256", train=["s1"], validation=["s0"], test=["s2"]
    )
    ds = _count_cfg(path, heldout_paths=(heldout,)).build(tok)
    assert [ds._data[int(i)]["image_sha256"] for i in ds._index] == ["s1"]
    assert ds.n_heldout_dropped == 2


def test_count_v2_reads_only_the_train_split(tmp_path):
    """`count-v2` ships `validation` / `test` splits that are copies of `train`; there is no
    knob to read them."""
    assert "split" not in PixMoCountV2DatasetConfig.__dataclass_fields__


@pytest.mark.parametrize("config_cls", [PixMoPointsV2DatasetConfig, PixMoCountV2DatasetConfig])
def test_heldout_guard_cannot_be_skipped_by_a_bad_path(tmp_path, config_cls):
    """A guard that cannot be checked fails the build instead of passing silently."""
    writer = _write_points_v2 if config_cls is PixMoPointsV2DatasetConfig else _write_count_v2
    cfg = config_cls(dataset_path=writer(tmp_path), heldout_paths=(str(tmp_path / "missing"),))
    with pytest.raises(OLMoConfigurationError, match="held-out set"):
        cfg.build(_FakeTok())


def test_heldout_guard_requires_the_id_column(tmp_path):
    path = _write_count_v2(tmp_path)
    wrong = _write_heldout(tmp_path, "count", "image_url", validation=["u0"])
    with pytest.raises(OLMoConfigurationError, match="image_sha256"):
        _count_cfg(path, heldout_paths=(wrong,)).build(_FakeTok())


def test_default_heldout_sets_are_the_real_validation_and_eval_sets():
    assert [p.rsplit("/", 1)[1] for p in PixMoPointsV2DatasetConfig().heldout_paths] == [
        "points-pointing",
        "points-counting",
        "pixmo-points-eval",
    ]
    assert [p.rsplit("/", 1)[1] for p in PixMoCountV2DatasetConfig().heldout_paths] == ["count"]


# ---------------------------------------------------------------------------
# No segmentation: masks in the data never turn into anything
# ---------------------------------------------------------------------------


def test_masks_in_the_data_never_become_segmentation(tmp_path):
    """The v17 build ships masks; this port has no segmentation path. mm_olmo keeps an
    annotation that is over ``max_points`` when it carries masks and renders it as ``pixmo_seg``.
    Here a mask changes nothing: that annotation is dropped and every branch is a pointing one."""
    mask = "rle-string-the-loader-must-never-decode"
    rows = {
        "image": [_image(tmp_path)],
        "image_url": ["u0"],
        "source": ["pointing"],
        "annotations": [
            [
                dict(_anno("Cup", [[10.0, 20.0, 1.0]], "correct"), masks=[mask]),
                dict(_anno("Crowd", [[1.0, 1.0, 0.0]] * 3, "correct"), masks=[mask] * 3),
            ]
        ],
        "min_points": [1],
        "max_points": [3],
        "min_masks": [1],
        "paired_negatives": [[]],
        "negatives": [[]],
        "easy_negatives": [[]],
        "rare_negatives": [[]],
        "paired_negatives_v2": [[]],
    }
    path = tmp_path / "with-masks"
    Dataset.from_dict(rows).save_to_disk(str(path))
    ds = _points_cfg(str(path), max_points=2, style=("point_count", "pointing")).build(_FakeTok())
    messages, _ = ds.format_row(_row0(ds), np.random.RandomState(0))
    assert _labels(messages) == ["Cup"]  # "Crowd" is over max_points; its masks do not rescue it
    assert {m["style"] for m in messages} <= {"pointing", "point_count"}
    assert "masks" not in messages[0] and "segmentations" not in messages[0]
    for config_cls in (PixMoPointsV2DatasetConfig, PixMoCountV2DatasetConfig):
        fields = set(cast(Any, config_cls).__dataclass_fields__)
        assert not fields & {"include_masks", "include_pointing", "zero_points_as_segmentation"}


# ---------------------------------------------------------------------------
# PixMoPointsV2: message assembly (mm_olmo PixMoPointV2.format_example)
# ---------------------------------------------------------------------------


def _row0(ds):
    return ds._data[int(ds._index[0])]


def test_points_v2_audit_style_marks_failed_annotations(tmp_path):
    path = _write_points_v2(tmp_path)
    ds = _points_cfg(
        path, audit_style=("aux_pointing",), n_easy_samples=0, p_paired_negatives=0.0
    ).build(_FakeTok())
    messages, weights = ds.format_row(_row0(ds), np.random.RandomState(0))
    by_label = {m["label"]: m for m in messages}
    assert set(by_label) == {"Red Cup", "Spoon", "Fork"}
    assert by_label["Red Cup"]["style"] == "pointing"
    assert by_label["Spoon"]["style"] == "aux_pointing"  # failed the audit, kept + marked
    # Only x, y are rendered (rows carry a depth column); percent coords, clipped.
    np.testing.assert_array_equal(by_label["Red Cup"]["points"], [[10.0, 20.0], [30.0, 40.0]])
    assert by_label["Red Cup"]["point_scale"] == 100
    # The annotated empty set is a negative ("There are none."), weighted like a paired one.
    assert by_label["Fork"]["points"].shape == (0, 2)
    assert weights == [None, None, None]  # negative_weight unset -> no per-branch scaling


def test_points_v2_filter_audit_drops_failed_annotations(tmp_path):
    path = _write_points_v2(tmp_path)
    ds = _points_cfg(path, filter_audit=True, n_easy_samples=0).build(_FakeTok())
    messages, _ = ds.format_row(_row0(ds), np.random.RandomState(0))
    assert _labels(messages) == ["Red Cup", "Fork"]


def test_points_v2_subsamples_easy_and_paired_negatives(tmp_path):
    path = _write_points_v2(tmp_path)
    tok = _FakeTok()
    row = _row0(_points_cfg(path).build(tok))

    # Default: 2 of the 3 easy negatives, no paired ones.
    ds = _points_cfg(path, n_easy_samples=2).build(tok)
    messages, weights = ds.format_row(row, np.random.RandomState(0))
    labels = _labels(messages)
    assert labels[:3] == ["Red Cup", "Spoon", "Fork"]
    easy = labels[3:]
    assert len(easy) == 2 and set(easy) < {"giraffe", "piano", "sailboat"}
    assert weights[3:] == [1.0, 1.0]  # easy negatives always weigh 1

    # p_paired_negatives=1 takes every v2 paired negative; the v1 pool is a different list.
    ds = _points_cfg(path, n_easy_samples=0, p_paired_negatives=1.0).build(tok)
    messages, _ = ds.format_row(row, np.random.RandomState(0))
    assert _labels(messages)[3:] == ["Mug", "Glass"]
    ds = _points_cfg(
        path, n_easy_samples=0, p_paired_negatives=1.0, v2_paired_negatives=False
    ).build(tok)
    messages, _ = ds.format_row(row, np.random.RandomState(0))
    assert _labels(messages)[3:] == ["Mug"]

    # A fractional rate is stochastically rounded: with 2 candidates and p=0.5 we get
    # exactly one per draw, and which one varies with the stream.
    ds = _points_cfg(path, n_easy_samples=0, p_paired_negatives=0.5).build(tok)
    picked = set()
    for seed in range(20):
        messages, _ = ds.format_row(row, np.random.RandomState(seed))
        paired = _labels(messages)[3:]
        assert len(paired) == 1
        picked.add(paired[0])
    assert picked == {"Mug", "Glass"}

    # Hard negatives draw from the v1 pool (easy + paired_negatives).
    ds = _points_cfg(path, n_easy_samples=0, n_hard_negatives=4).build(tok)
    messages, _ = ds.format_row(row, np.random.RandomState(0))
    assert set(_labels(messages)[3:]) == {"giraffe", "piano", "sailboat", "Mug"}


def test_points_v2_negative_weight_targets_paired_and_annotated_negatives(tmp_path):
    path = _write_points_v2(tmp_path)
    ds = _points_cfg(path, n_easy_samples=1, p_paired_negatives=1.0, negative_weight=0.5).build(
        _FakeTok()
    )
    messages, weights = ds.format_row(_row0(ds), np.random.RandomState(0))
    labels = _labels(messages)
    assert labels[:3] == ["Red Cup", "Spoon", "Fork"] and labels[4:] == ["Mug", "Glass"]
    assert weights == [None, None, 0.5, 1.0, 0.5, 0.5]


def test_points_v2_example_applies_branch_weights_after_shuffle(tmp_path):
    """End to end: the negatives' 0.5 follows them through the branch shuffle, easy negatives
    and positives stay at 1, and every branch is isolated by its own subsegment id."""
    path = _write_points_v2(tmp_path)
    ds = _points_cfg(
        path,
        n_easy_samples=1,
        p_paired_negatives=1.0,
        negative_weight=0.5,
        loss_token_weighting="none",
    ).build(_FakeTok())
    ex = ds[0]
    for key in ("input_ids", "labels", "loss_masks", "position_ids", "subsegment_ids", "images"):
        assert key in ex
    branch_ids = sorted(set(ex["subsegment_ids"].tolist()) - {ATTEND_ALL_SUBSEGMENT_ID})
    assert branch_ids == list(range(6))
    per_branch = {}
    for b in branch_ids:
        sel = ex["subsegment_ids"] == b
        w = set(np.round(ex["loss_masks"][sel][ex["loss_masks"][sel] > 0], 4).tolist())
        assert len(w) == 1, f"branch {b} has mixed weights {w}"
        per_branch[b] = (w.pop(), _contains(ex["input_ids"][sel], REFUSAL_IDS))
    assert sorted(w for w, _ in per_branch.values()) == [0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
    # Every down-weighted branch is a refusal; exactly one refusal (the easy negative) is not.
    assert all(is_refusal for w, is_refusal in per_branch.values() if w == 0.5)
    assert sum(1 for w, is_refusal in per_branch.values() if w == 1.0 and is_refusal) == 1
    # Prefix carries no loss.
    assert not ex["loss_masks"][ex["subsegment_ids"] == ATTEND_ALL_SUBSEGMENT_ID].any()


def test_points_v2_example_is_deterministic_per_index(tmp_path):
    path = _write_points_v2(tmp_path)
    ds = _points_cfg(path, p_paired_negatives=0.5).build(_FakeTok())
    a, b = ds[0], ds[0]
    np.testing.assert_array_equal(a["input_ids"], b["input_ids"])
    np.testing.assert_array_equal(a["loss_masks"], b["loss_masks"])
    other = _points_cfg(path, p_paired_negatives=0.5, seed=1).build(_FakeTok())[0]
    assert not np.array_equal(a["input_ids"], other["input_ids"])


def test_negatives_rotate_across_epochs(tmp_path):
    """``p_paired_negatives`` / ``n_easy_samples`` are documented as per-epoch sub-sampling, so
    the draw must move with the epoch. Pinned at epoch 0 the same subset is redrawn forever and
    the rest of each pool is never trained on."""
    path = _write_points_v2(tmp_path)
    ds = _points_cfg(path, n_easy_samples=1, p_paired_negatives=0.5).build(_FakeTok())
    row = ds._data[int(ds._index[0])]

    seen = set()
    for epoch in range(6):
        ds.set_epoch(epoch)
        messages, _ = ds.format_row(row, ds.epoch_rng(0))
        seen.add(tuple(_labels(messages)[3:]))  # the sampled negatives
    assert len(seen) > 1, f"negatives never rotated across epochs: {seen}"

    # Deterministic within an epoch, and identical for a resumed run that replays that epoch.
    ds.set_epoch(3)
    a = ds[0]
    ds.set_epoch(0)
    ds.set_epoch(3)
    np.testing.assert_array_equal(a["input_ids"], ds[0]["input_ids"])
    ds.set_epoch(4)
    assert not np.array_equal(a["input_ids"], ds[0]["input_ids"])


def test_mixture_loader_hands_the_epoch_to_sources(tmp_path):
    """The epoch only reaches a dataset if the loader passes it: ``reshuffle`` permutes
    ``(source, index)`` references, which cannot rotate a draw made *inside* an example."""
    from olmo_core.data.multimodal import MixtureDataLoader, MultimodalCollatorConfig

    class _Recording:
        def __init__(self):
            self.epochs = []

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

        def __len__(self):
            return 4

        def __getitem__(self, i):
            raise AssertionError("not iterated in this test")

    seq = 8
    sources = [_Recording(), _Recording()]
    loader = MixtureDataLoader(
        sources,
        [0.5, 0.5],
        MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=seq).build(),
        work_dir=str(tmp_path),
        global_batch_size=2 * seq,
        seed=0,
    )
    loader.reshuffle(epoch=1)
    loader.reshuffle(epoch=2)
    assert [s.epochs for s in sources] == [[1, 2], [1, 2]]


def test_points_v2_stage1_prompt_family(tmp_path):
    """Stage-1 form: ``"<style>: <lowercased label>"`` and the html-v2 answer."""
    path = _write_points_v2(tmp_path)
    ds = _points_cfg(
        path, audit_style=("aux_pointing",), n_easy_samples=0, p_paired_negatives=0.0
    ).build(_FakeTok())
    rng = np.random.RandomState(0)
    messages, _ = ds.format_row(_row0(ds), rng)
    fmt = SftFormatter(seed=0, **STAGE1)
    turns = {m["label"]: fmt.format_turns(m, index=0, rng=rng)[0] for m in messages}
    assert turns["Red Cup"] == (
        "pointing: red cup",
        '<points coords="1 1 100 200 2 300 400">red cup</points>',
    )
    assert turns["Spoon"] == (
        "aux_pointing: spoon",
        '<points coords="1 1 500 500">spoon</points>',
    )
    assert turns["Fork"] == ("pointing: fork", "There are none.")


# ---------------------------------------------------------------------------
# PixMoCountV2
# ---------------------------------------------------------------------------


def test_count_v2_selection_and_pixel_points(tmp_path):
    path = _write_count_v2(tmp_path)
    tok = _FakeTok()
    ds = _count_cfg(path, audit_style=("aux_pointing",)).build(tok)
    # 6 rows, but blank-label / null-label / null-points rows are not trainable.
    assert len(ds) == 3
    assert len(_count_cfg(path, filter_audit=True).build(tok)) == 2

    fmt = SftFormatter(seed=0, **STAGE1)
    rng = np.random.RandomState(0)
    rows = [ds._data[int(i)] for i in ds._index]
    # Pixel coordinates are normalised by the image size (64 x 48) and sorted by (x, y).
    (msg,) = ds.format_row(rows[0], rng, image_size=(64, 48))
    assert msg["point_scale"] is None and msg["image_size"] == (64, 48)
    assert fmt.format_turns(msg, index=0, rng=rng)[0] == (
        "pointing: ties",
        '<points coords="1 1 250 250 2 500 500">ties</points>',
    )
    # Failed audit -> marker style; empty set -> refusal.
    (msg,) = ds.format_row(rows[1], rng, image_size=(64, 48))
    assert msg["style"] == "aux_pointing"
    assert fmt.format_turns(msg, index=0, rng=rng)[0][0] == "aux_pointing: cats"
    (msg,) = ds.format_row(rows[2], rng, image_size=(64, 48))
    assert fmt.format_turns(msg, index=0, rng=rng)[0] == ("pointing: dogs", "There are none.")


def test_count_v2_drops_unusable_annotations(tmp_path):
    """mm_olmo never had to guard these; here a blank label trains ``"pointing:"`` against an
    empty-named ``<points>`` tag, and a null label / points raises during formatting."""
    path = _write_count_v2(tmp_path)
    ds = _count_cfg(path).build(_FakeTok())
    kept = [ds._data[int(i)]["image_url"] for i in ds._index]
    assert kept == ["u0", "u1", "u2"]  # u3 blank, u4 null label, u5 null points
    assert ds.keep({"label": "ties", "points": [], "audit_result": "n/a"}) is True
    for bad in (
        {"label": "   ", "points": [[1.0, 2.0]], "audit_result": "correct"},
        {"label": None, "points": [[1.0, 2.0]], "audit_result": "correct"},
        {"label": "kites", "points": None, "audit_result": "correct"},
    ):
        assert ds.keep(bad) is False, bad


def test_count_v2_example_end_to_end(tmp_path):
    path = _write_count_v2(tmp_path)
    ex = _count_cfg(path, style=("point_count", "pointing")).build(_FakeTok())[0]
    assert "subsegment_ids" not in ex  # one point set per image -> single branch
    assert ex["loss_masks"].sum() > 0
    assert ex["images"].shape[0] == 2  # max_crops=1 -> the global crop + one local crop


# ---------------------------------------------------------------------------
# Molmo2-Stage1 wiring
# ---------------------------------------------------------------------------


def _load_stage1_module():
    try:
        import olmo_core.internal.common  # noqa: F401  (needs a recent beaker-py)
    except ImportError as e:  # pragma: no cover - env-dependent
        pytest.skip(f"Molmo2-Stage1.py imports fail here: {e}")
    spec = importlib.util.spec_from_file_location(
        "_stage1_v2", "src/scripts/train/Molmo2-Stage1.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_v2"] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


def test_stage1_pointing_data_selector():
    mod = _load_stage1_module()
    assert mod.POINTING_DATA == "v1"
    assert mod.POINTING_DATA_CHOICES == ("v1", "v2")
    fields = {f.name for f in mod.ExperimentConfig.__dataclass_fields__.values()}
    assert {"pointing_data", "pointing_v2", "count_v2"} <= fields
    # mm_olmo `_base_mixture`: audit-failed points are marked, negatives are sub-sampled.
    assert mod.POINTING_V2_AUDIT_STYLE == ("aux_point_count", "aux_pointing")
    assert mod.POINTING_V2_FILTER_AUDIT is False
    assert mod.POINTING_V2_P_PAIRED_NEGATIVES == 0.25
    assert mod.POINTING_V2_N_EASY_NEGATIVES == 2


def test_stage1_pointing_group_split_rule():
    """v1 splits the group's rate by sqrt(size) (mm_olmo captioner), v2 linearly
    (mm_olmo molmo3 stage 1 ``size_weighted=1``)."""
    mod = _load_stage1_module()
    sizes = [100, 400, 1600]
    np.testing.assert_allclose(
        mod._pointing_group_fractions(sizes, "v1"), np.array([10, 20, 40]) / 70
    )
    np.testing.assert_allclose(
        mod._pointing_group_fractions(sizes, "v2"), np.array([100, 400, 1600]) / 2100
    )
    with pytest.raises(OLMoConfigurationError):
        mod._pointing_group_fractions(sizes, "v3")


def test_stage1_rejects_empty_mixture_sources():
    """A size-0 source gets weight 0 from any size-proportional split and is then skipped by the
    loader, so a mistyped path would quietly train the rest at renormalized rates; all-empty
    divides by zero into NaN weights."""
    mod = _load_stage1_module()
    with pytest.raises(OLMoConfigurationError) as e:
        mod._size_fractions([100, 0, 50], "sqrt", ["a", "empty_one", "c"])
    assert "empty_one" in str(e.value)
    with pytest.raises(OLMoConfigurationError):
        mod._size_fractions([0, 0], "linear", ["a", "b"])
    with pytest.raises(OLMoConfigurationError):
        mod._pointing_group_fractions([10, 0], "v2", ["a", "b"])
    # Names are optional; the index is reported instead.
    with pytest.raises(OLMoConfigurationError) as e:
        mod._size_fractions([0, 3], "sqrt")
    assert "['0']" in str(e.value)
