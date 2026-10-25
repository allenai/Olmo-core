"""``annotation_sampling`` on the multi-annotation pointing adapters (PixMo points, CoSyn)."""

import hashlib

import numpy as np
import pytest
import torch

from olmo_core.data.multimodal import message_sequence, pixmo_points
from olmo_core.nn.vision.multimodal import has_sibling_branches

_ANNOTATIONS = {
    "apple": {"x": [10.0], "y": [20.0]},
    "boat": {"x": [30.0, 50.0], "y": [40.0, 60.0]},
    "cat": {"x": [70.0], "y": [80.0]},
}
_INDICES = range(4)
_EPOCHS = range(3)
# Digests of every (index, epoch) example of the three-annotation fixture, recorded before
# ``annotation_sampling`` existed: the default must keep examples bitwise unchanged.
_LEGACY_DIGESTS = {
    "points": "8a3a6a8935c2a4709b924f6fc78623687a4690daeaa0e1e629e24cdb9676b12a",
    "cosyn": "8ac5ada118efa1f425e793a438d8752d0776bdef00252e2b7c084881a3c9ac77",
}
_LEGACY_FINGERPRINT = "f9058a1acfac3788cc7910286822e021854847b947be94be811724fa48910d4c"


class _Tok:
    eos_token_id = 1
    bos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"

    def encode(self, text, add_special_tokens=False):
        return [(ord(c) % 90) + 10 for c in text]


def _fake_preprocess(*args, rng, **kwargs):
    # Image features come from the example's stream, so digests also pin its augmentation draws.
    generator = torch.Generator().manual_seed(int(rng.randint(2**31)))
    return (
        torch.rand(1, 1, 729, 14 * 14 * 3, generator=generator),
        torch.zeros(1, 1, 4, dtype=torch.long),
        np.array([1, 1, 1, 1], dtype=np.int32),
    )


@pytest.fixture(autouse=True)
def _fake_images(monkeypatch):
    monkeypatch.setattr(message_sequence, "preprocess_image_molmo2", _fake_preprocess)
    monkeypatch.setattr(pixmo_points, "_open_image", lambda _path: object())


def _points(names, **config):
    dataset = object.__new__(pixmo_points.PixMoPointsDataset)
    dataset.config = pixmo_points.PixMoPointsDatasetConfig(
        message_format="document", loss_token_weighting="root_subsegments_root_tokens", **config
    )
    dataset.tokenizer = _Tok()
    points = [
        [{"x": x, "y": y} for x, y in zip(_ANNOTATIONS[name]["x"], _ANNOTATIONS[name]["y"])]
        for name in names
    ]
    dataset._data = [{"image": "unused", "label": list(names), "points": points}]
    dataset._index = [(0, list(range(len(names))))] * len(_INDICES)
    return dataset


def _cosyn(names, **config):
    dataset = object.__new__(pixmo_points.CoSynPointDataset)
    dataset.config = pixmo_points.CoSynPointDatasetConfig(
        message_format="document", loss_token_weighting="root_subsegments_root_tokens", **config
    )
    dataset.tokenizer = _Tok()
    row = {
        "image": "unused",
        "questions": [f"Where is the {name}?" for name in names],
        "answer_points": [_ANNOTATIONS[name] for name in names],
        "names": list(names),
    }
    dataset._data = [row] * len(_INDICES)
    return dataset


_BUILDERS = {"points": _points, "cosyn": _cosyn}


def _digest(examples) -> str:
    digest = hashlib.sha256()
    for example in examples:
        for key in sorted(example):
            value = np.ascontiguousarray(np.asarray(example[key]))
            digest.update(f"{key}:{value.dtype}:{value.shape}".encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def _all_examples(dataset):
    return [dataset.get(index, epoch) for index in _INDICES for epoch in _EPOCHS]


def _siblings(example) -> bool:
    if "subsegment_ids" not in example:
        return False
    return has_sibling_branches(torch.as_tensor(example["subsegment_ids"])[None])


@pytest.mark.parametrize("kind", sorted(_BUILDERS))
def test_default_sampling_is_bitwise_unchanged(kind):
    build = _BUILDERS[kind]
    default = _all_examples(build(list(_ANNOTATIONS)))
    assert all(_siblings(example) for example in default)
    assert _digest(default) == _LEGACY_DIGESTS[kind]
    explicit = build(list(_ANNOTATIONS), annotation_sampling="all")
    assert _digest(_all_examples(explicit)) == _LEGACY_DIGESTS[kind]


def test_default_sampling_keeps_the_content_fingerprint():
    def fingerprint(**config):
        return pixmo_points._adapter_fingerprint(
            "PixMoPointsDataset",
            pixmo_points.PixMoPointsDatasetConfig(**config),
            [],
            derived_index_sha256="0" * 64,
        )

    assert fingerprint() == fingerprint(annotation_sampling="all") == _LEGACY_FINGERPRINT
    assert fingerprint(annotation_sampling="one") != _LEGACY_FINGERPRINT


@pytest.mark.parametrize("kind", sorted(_BUILDERS))
def test_one_sampled_annotation_is_a_single_annotation_example(kind):
    build = _BUILDERS[kind]
    names = list(_ANNOTATIONS)
    # Fixed pointing style: no per-annotation style draws precede the selection.
    config = {"counting": False} if kind == "points" else {}
    one = build(names, annotation_sampling="one", **config)
    for index in _INDICES:
        for epoch in _EPOCHS:
            example = one.get(index, epoch)
            assert not _siblings(example)
            chosen = names[pixmo_points.select_annotation(0, index, epoch, len(names))]
            single = build([chosen], **config).get(index, epoch)
            assert _digest([example]) == _digest([single])


@pytest.mark.parametrize("kind", sorted(_BUILDERS))
def test_one_sampled_annotation_is_deterministic_per_seed_index_and_epoch(kind):
    build = _BUILDERS[kind]
    first = _all_examples(build(list(_ANNOTATIONS), annotation_sampling="one"))
    again = _all_examples(build(list(_ANNOTATIONS), annotation_sampling="one"))
    assert _digest(first) == _digest(again)
    assert not any(_siblings(example) for example in first)
    reseeded = _all_examples(build(list(_ANNOTATIONS), annotation_sampling="one", seed=1))
    assert _digest(reseeded) != _digest(first)


def test_select_annotation_covers_every_annotation_across_epochs():
    picks = [pixmo_points.select_annotation(0, 5, epoch, 3) for epoch in range(64)]
    assert set(picks) == {0, 1, 2}
    assert picks == [pixmo_points.select_annotation(0, 5, epoch, 3) for epoch in range(64)]
    assert [pixmo_points.select_annotation(0, index, 0, 3) for index in range(64)] != picks
    assert pixmo_points.select_annotation(7, 5, 2, 1) == 0
    with pytest.raises(ValueError, match="count must be positive"):
        pixmo_points.select_annotation(0, 0, 0, 0)


@pytest.mark.parametrize(
    "config_type, dataset_type",
    [
        (pixmo_points.PixMoPointsDatasetConfig, pixmo_points.PixMoPointsDataset),
        (pixmo_points.CoSynPointDatasetConfig, pixmo_points.CoSynPointDataset),
    ],
)
def test_annotation_sampling_is_validated(config_type, dataset_type):
    with pytest.raises(ValueError, match="annotation_sampling must be one of"):
        dataset_type(config_type(annotation_sampling="some"), _Tok())
