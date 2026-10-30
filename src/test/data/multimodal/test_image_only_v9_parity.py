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
    ds = build_image_only_v9_dataset(name, _FakeTokenizer())
    assert hasattr(ds, "__len__")
    assert hasattr(ds, "__getitem__")
