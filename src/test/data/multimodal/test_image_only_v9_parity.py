"""Registry smoke tests for image-only-v9 datasets."""

import pytest

from olmo_core.data.multimodal.mixtures.image_only_v9 import (
    IMAGE_ONLY_V9_SUBMIXTURES,
    build_image_only_v9_dataset,
)


class _FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return f"user:{messages[0]['content']}\nassistant:"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 1000 for c in text[:32]]


@pytest.mark.parametrize(
    "name",
    [src.name for g in IMAGE_ONLY_V9_SUBMIXTURES for src in g.datasets],
)
def test_build_image_only_v9_dataset_registered(name):
    try:
        ds = build_image_only_v9_dataset(name, _FakeTokenizer())
    except FileNotFoundError:
        pytest.skip(f"{name} data not available")
    assert hasattr(ds, "__len__")
    assert hasattr(ds, "__getitem__")


def _fake_sources(monkeypatch):
    """Replace the dataset builder by one that records which sources get opened."""
    from olmo_core.data.multimodal.mixtures import image_only_v9 as mod

    names = [src.name for group in mod.IMAGE_ONLY_V9_SUBMIXTURES for src in group.datasets]
    sizes = {name: 100 + 37 * i for i, name in enumerate(names)}
    opened: list = []

    class _Fake:
        def __init__(self, name):
            self.name = name

        def __len__(self):
            return sizes[self.name]

    def build(name, tokenizer, seed=0, *, max_sequence_length=None):
        opened.append(name)
        return _Fake(name)

    monkeypatch.setattr(mod, "build_image_only_v9_dataset", build)
    return mod, sizes, opened


@pytest.mark.parametrize("requested", [["tulu4"], ["pixmo_points_train"], ["tulu4", "pixmo_cap"]])
def test_subset_mixture_opens_only_the_requested_groups_and_keeps_the_weights(
    monkeypatch, requested
):
    from olmo_core.data.multimodal.mixture_weights import compute_flat_mixture_weights

    mod, sizes, opened = _fake_sources(monkeypatch)
    datasets, weights = mod.build_image_only_v9_mixture(None, dataset_names=requested)

    # The weights are what the all-sources computation gives, renormalized over the subset.
    full = compute_flat_mixture_weights(mod.IMAGE_ONLY_V9_SUBMIXTURES, sizes)
    kept = [(name, w) for name, w in full if name in requested]
    expected = {name: w / sum(w for _, w in kept) for name, w in kept}
    assert {d.name: w for d, w in zip(datasets, weights)} == pytest.approx(expected)
    assert sum(weights) == pytest.approx(1.0)

    # Only the groups that contain a requested source were opened.
    wanted_groups = [
        {src.name for src in group.datasets}
        for group in mod.IMAGE_ONLY_V9_SUBMIXTURES
        if any(src.name in requested for src in group.datasets)
    ]
    assert set(opened) == set().union(*wanted_groups)
    assert len(set(opened)) < len(sizes)


def test_full_mixture_still_opens_every_source_and_unknown_names_are_rejected(monkeypatch):
    mod, sizes, opened = _fake_sources(monkeypatch)
    datasets, weights = mod.build_image_only_v9_mixture(None)
    assert set(opened) == set(sizes)
    assert sum(weights) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="No mixture sources matched"):
        mod.build_image_only_v9_mixture(None, dataset_names=["not_a_source"])
