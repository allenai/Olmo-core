"""The collator's compact image layout: real crops only, plus crops per example."""

import numpy as np
import pytest
import torch

from olmo_core.config import DType
from olmo_core.data.multimodal.collator import (
    MultimodalCollator,
    MultimodalCollatorConfig,
)
from olmo_core.data.multimodal.packing import _PackedImageParts

_N_PATCHES, _PATCH_DIM = 4, 6


def _example(n_crops: int, seed: int, length: int = 8):
    rng = np.random.default_rng(seed)
    return dict(
        input_ids=np.arange(length, dtype=np.int64),
        labels=np.arange(length, dtype=np.int64),
        loss_masks=np.ones(length, dtype=np.float32),
        position_ids=np.arange(length, dtype=np.int64),
        token_type_ids=np.zeros(length, dtype=np.int64),
        images=rng.standard_normal((n_crops, _N_PATCHES, _PATCH_DIM)).astype(np.float32),
        pooled_patches_idx=(
            np.arange(n_crops * _N_PATCHES, dtype=np.int64).reshape(n_crops, _N_PATCHES)
            if n_crops
            else np.full((0, _N_PATCHES), -1, dtype=np.int64)
        ),
    )


def _real_crops(padded: torch.Tensor, counts) -> torch.Tensor:
    return torch.cat([padded[b, :n] for b, n in enumerate(counts)])


@pytest.mark.parametrize("batch_metadata", [False, True])
@pytest.mark.parametrize("image_dtype", [None, DType.bfloat16])
def test_compact_images_are_the_padded_batch_without_its_padding(batch_metadata, image_dtype):
    examples = [_example(2, 0), _example(0, 1), _example(3, 2)]
    padded = MultimodalCollatorConfig(
        pad_token_id=0, batch_metadata=batch_metadata, image_dtype=image_dtype
    ).build()(examples)
    compact = MultimodalCollatorConfig(
        pad_token_id=0, batch_metadata=batch_metadata, image_dtype=image_dtype, compact_images=True
    ).build()(examples)

    assert padded["images"].shape == (3, 3, _N_PATCHES, _PATCH_DIM)
    assert compact["images"].shape == (5, _N_PATCHES, _PATCH_DIM)
    assert compact["images"].dtype == padded["images"].dtype
    torch.testing.assert_close(
        compact["images"], _real_crops(padded["images"], [2, 0, 3]), rtol=0, atol=0
    )
    # The crop counts always come with compact images; the pooled counts stay metadata.
    assert compact["image_crop_counts"].tolist() == [2, 0, 3]
    assert ("pooled_token_counts" in compact) is batch_metadata
    assert ("image_crop_counts" in padded) is batch_metadata
    # Every other field is the same: the pooled indices still address each example's crops.
    for key in padded:
        if key not in ("images", "image_crop_counts"):
            torch.testing.assert_close(compact[key], padded[key], rtol=0, atol=0)
    assert set(compact) - set(padded) <= {"image_crop_counts"}


def test_text_only_batch_collates_no_crops():
    examples = [_example(0, 0), _example(0, 1)]
    compact = MultimodalCollator(pad_token_id=0, compact_images=True, image_dtype=torch.bfloat16)(
        examples
    )
    assert compact["images"].shape == (0, _N_PATCHES, _PATCH_DIM)
    assert compact["images"].dtype == torch.bfloat16
    assert compact["image_crop_counts"].tolist() == [0, 0]
    assert compact["pooled_patches_idx"].shape == (2, 1, _N_PATCHES)
    assert bool((compact["pooled_patches_idx"] == -1).all())


def test_packed_image_parts_are_concatenated_in_order():
    first, second = _example(2, 0), _example(1, 1)
    packed = dict(first)
    packed["images"] = _PackedImageParts(
        parts=(first["images"], second["images"]), shape=(3, _N_PATCHES, _PATCH_DIM)
    )
    packed["pooled_patches_idx"] = np.arange(3 * _N_PATCHES, dtype=np.int64).reshape(3, _N_PATCHES)
    batch = MultimodalCollator(pad_token_id=0, compact_images=True)([packed, _example(1, 2)])
    assert batch["image_crop_counts"].tolist() == [3, 1]
    expected = np.concatenate([first["images"], second["images"], _example(1, 2)["images"]])
    torch.testing.assert_close(batch["images"], torch.from_numpy(expected), rtol=0, atol=0)


def test_config_default_keeps_the_padded_layout():
    assert MultimodalCollatorConfig(pad_token_id=0).compact_images is False
    batch = MultimodalCollatorConfig(pad_token_id=0).build()([_example(1, 0), _example(0, 1)])
    assert batch["images"].ndim == 4 and "image_crop_counts" not in batch
