import numpy as np
import pytest
import torch

from olmo_core.config import DType
from olmo_core.data.multimodal.collator import (
    MultimodalCollator,
    MultimodalCollatorConfig,
)

_N_PATCHES, _PATCH_DIM = 4, 6


def _example(n_crops: int, seed: int):
    rng = np.random.default_rng(seed)
    length = 8
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


def test_images_keep_float32_by_default():
    batch = MultimodalCollator(pad_token_id=0)([_example(2, 0), _example(0, 1)])
    assert batch["images"].dtype == torch.float32


@pytest.mark.parametrize("via_config", [False, True])
def test_images_are_emitted_in_the_requested_dtype(via_config):
    examples = [_example(2, 0), _example(1, 1), _example(0, 2)]
    if via_config:
        collator = MultimodalCollatorConfig(pad_token_id=0, image_dtype=DType.bfloat16).build()
    else:
        collator = MultimodalCollator(pad_token_id=0, image_dtype=torch.bfloat16)
    batch = collator(examples)
    reference = MultimodalCollator(pad_token_id=0)(examples)["images"]
    assert batch["images"].dtype == torch.bfloat16
    assert batch["images"].shape == reference.shape
    # The same pixels, rounded once to bfloat16; the padding stays exactly zero.
    torch.testing.assert_close(batch["images"], reference.to(torch.bfloat16), rtol=0, atol=0)
    assert batch["images"][2].abs().sum() == 0
    assert batch["pooled_patches_idx"].dtype == torch.int64
