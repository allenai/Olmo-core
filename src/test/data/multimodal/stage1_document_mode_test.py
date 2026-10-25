"""Stage-1 v3 sources in document mode: the language model's image tokens and layout, one sampled
annotation per example, and per-call source epochs (``get(index, epoch)``)."""

import threading

import numpy as np
import pytest
from datasets import Dataset, DatasetDict

from olmo_core.data.multimodal import (
    PixMoPointsV2DatasetConfig,
    TextRichCaptionDatasetConfig,
)
from olmo_core.data.multimodal.text_rich_caption import CAPTION_LEVELS
from olmo_core.nn.vision import Molmo2TokenIds

# Image token rows of an LM with a 100,352-row embedding table (the dolma2 layout).
DOLMA2_IDS = Molmo2TokenIds(
    im_start_id=100278,
    im_end_id=100279,
    im_patch_id=100280,
    im_col_id=100281,
    low_res_im_start_id=100282,
    image_placeholder_id=100283,
    im_end_turn_id=100264,
)
_QWEN_IMAGE_IDS = set(Molmo2TokenIds().image_token_ids)


class _Tok:
    eos_token_id = 1
    bos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"

    def encode(self, text, add_special_tokens=False):
        return [(ord(c) % 90) + 10 for c in text]


def _image(tmp_path, relpath="img.png"):
    from PIL import Image

    path = tmp_path / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color=(200, 30, 30)).save(path)
    return str(path)


def _text_rich(tmp_path, **kw):
    rows = []
    for i in range(6):
        relpath = f"chart/ex{i}/ex{i}.png"
        _image(tmp_path, relpath)
        rows.append(
            {
                "id": f"ex{i}",
                "category": "chart",
                "image_relpath": relpath,
                "high_level": f"high {i}",
                "mid_level": f"mid level caption {i}",
                "low_level": f"low level dense caption number {i}",
            }
        )
    DatasetDict({"train": Dataset.from_list(rows)}).save_to_disk(str(tmp_path / "hf" / "chart"))
    return TextRichCaptionDatasetConfig(dataset_path=str(tmp_path), max_crops=1, **kw).build(_Tok())


def _points_v2(tmp_path, **kw):
    image = _image(tmp_path)
    anno = [
        {"audit_result": "correct", "label": label, "points": [[10.0 * k, 20.0, 1.0]]}
        for k, label in enumerate(("cup", "spoon", "plate"), start=1)
    ]
    Dataset.from_dict(
        {
            "image": [image] * 4,
            "image_url": [f"u{i}" for i in range(4)],
            "source": ["pointing"] * 4,
            "annotations": [anno] * 4,
            "easy_negatives": [["giraffe"]] * 4,
            "paired_negatives": [[]] * 4,
            "paired_negatives_v2": [[]] * 4,
        }
    ).save_to_disk(str(tmp_path / "points-v2"))
    return PixMoPointsV2DatasetConfig(
        dataset_path=str(tmp_path / "points-v2"),
        heldout_paths=(),
        style=("pointing",),
        max_crops=1,
        **kw,
    ).build(_Tok())


def _supervised_levels(example, row) -> list:
    """The caption levels of ``row`` whose response tokens the example supervises."""
    supervised = np.asarray(example["labels"])[np.asarray(example["loss_masks"]) > 0].tolist()

    def contains(ids):
        return any(supervised[i : i + len(ids)] == ids for i in range(len(supervised)))

    return [lvl for lvl in CAPTION_LEVELS if contains(_Tok().encode(" " + row[lvl]))]


def _assert_document_example(example):
    ids = set(np.asarray(example["input_ids"]).tolist())
    assert DOLMA2_IDS.im_patch_id in ids and DOLMA2_IDS.im_start_id in ids
    assert not ids & _QWEN_IMAGE_IDS
    assert max(ids) < 100352
    assert example["input_ids"][0] == _Tok.eos_token_id  # the document boundary
    assert (
        "subsegment_ids" not in example
        or len(set(example["subsegment_ids"].tolist()) - {10000}) <= 1
    )


@pytest.mark.parametrize("build", [_text_rich, _points_v2])
def test_document_mode_one_annotation_uses_the_lm_image_tokens(tmp_path, build):
    ds = build(
        tmp_path,
        token_ids=DOLMA2_IDS,
        message_format="document",
        annotation_sampling="one",
        max_sequence_length=4096,
    )
    for index in range(len(ds)):
        _assert_document_example(ds.get(index, 0))


def test_one_annotation_rotates_with_the_source_epoch(tmp_path):
    ds = _text_rich(
        tmp_path, token_ids=DOLMA2_IDS, message_format="document", annotation_sampling="one"
    )
    chosen = set()
    for epoch in range(4):
        for index in range(len(ds)):
            levels = _supervised_levels(ds.get(index, epoch), ds._data[index])
            assert len(levels) == 1, levels
            chosen.add(levels[0])
    assert chosen == set(CAPTION_LEVELS)


@pytest.mark.parametrize("build", [_text_rich, _points_v2])
def test_get_epoch_matches_set_epoch_without_mutating_it(tmp_path, build):
    """``get(index, epoch)`` is the example ``set_epoch(epoch); ds[index]`` builds, and leaves the
    shared epoch alone, so concurrent loader threads at different source epochs agree."""
    ds = build(tmp_path)  # defaults: qwen3, every annotation
    expected = {}
    for epoch in range(3):
        ds.set_epoch(epoch)
        for index in range(len(ds)):
            expected[index, epoch] = ds[index]
    ds.set_epoch(7)
    results = {}

    def work(epoch):
        for index in range(len(ds)):
            results[index, epoch] = ds.get(index, epoch)

    threads = [threading.Thread(target=work, args=(epoch,)) for epoch in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert ds._epoch == 7
    for key, example in expected.items():
        assert set(results[key]) == set(example)
        for name, value in example.items():
            np.testing.assert_array_equal(results[key][name], value)
