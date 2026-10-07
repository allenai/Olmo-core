"""The OLMoDDP train module's asynchronous, pinned device copy of the batch's pixels."""

import pytest
import torch

from olmo_core.testing import requires_gpu
from olmo_core.train.train_module.transformer.multimodal_train_module import (
    MultimodalOLMoDDPTrainModule,
)


def _module(device: str, enabled: bool = True) -> MultimodalOLMoDDPTrainModule:
    module = object.__new__(MultimodalOLMoDDPTrainModule)
    module.device = torch.device(device)
    module._pp_config = None
    module.response_logits_only = True
    module.trim_microbatch_image_padding = False
    module.pinned_image_transfer = enabled
    module._pinned_images = None
    module._image_copy_stream = None
    module._image_copy_event = None
    return module


def _batch(size: int = 2, crops: int = 3) -> dict:
    inputs = torch.randint(0, 50, (size, 8))
    return {
        "input_ids": inputs,
        "labels": inputs.clone(),
        "loss_masks": torch.ones_like(inputs, dtype=torch.float32),
        "images": torch.randn(size, crops, 4, 14 * 14 * 3).to(torch.bfloat16),
        "pooled_patches_idx": torch.arange(size * 4).reshape(size, 1, 4),
    }


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_staging_is_a_no_op_when_disabled_or_without_images(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    batch = _batch()
    original = dict(batch)
    assert _module(device, enabled=False)._stage_images_on_device(batch) is None
    assert all(batch[key] is original[key] for key in original)
    text_only = {"input_ids": batch["input_ids"]}
    assert _module(device)._stage_images_on_device(text_only) is None
    assert text_only == {"input_ids": batch["input_ids"]}


def test_staging_leaves_host_batches_alone_on_a_cpu_module():
    batch = _batch()
    images = batch["images"]
    assert _module("cpu")._stage_images_on_device(batch) is None
    assert batch["images"] is images


@requires_gpu
def test_model_receives_device_images_and_host_indices():
    module = _module("cuda")
    batch = _batch()
    host = {key: value.clone() for key, value in batch.items()}

    seconds = module._stage_images_on_device(batch)
    assert seconds is not None and seconds >= 0
    assert module._pinned_images is not None and module._pinned_images.is_pinned()
    staged = batch["images"]
    assert staged.is_cuda and staged.dtype == torch.bfloat16
    torch.testing.assert_close(staged.cpu(), host["images"])
    for key in ("input_ids", "labels", "loss_masks", "pooled_patches_idx"):
        assert batch[key].device.type == "cpu"
        torch.testing.assert_close(batch[key], host[key])

    # The model gets the pre-moved pixels and the host-side indices from ``_prepare_batch``.
    input_ids, labels, kwargs = module._prepare_batch(dict(batch))
    assert kwargs["images"] is staged
    assert kwargs["pooled_patches_idx"].device.type == "cpu"
    assert input_ids.device.type == "cpu" and labels is not None and labels.device.type == "cpu"
    assert kwargs["loss_masks"].device.type == "cpu"

    # Already-on-device pixels are left as they are (no second copy, no new buffer).
    buffer = module._pinned_images
    assert module._stage_images_on_device(batch) is None and batch["images"] is staged
    assert module._pinned_images is buffer


@requires_gpu
def test_pinned_buffer_is_reused_and_grows_to_the_largest_batch():
    module = _module("cuda")
    small = _batch(size=1, crops=2)
    module._stage_images_on_device(small)
    buffer = module._pinned_images
    assert buffer is not None and buffer.numel() == small["images"].numel()
    first_copy = module._image_copy_event

    same = _batch(size=1, crops=2)
    expected = same["images"].clone()
    module._stage_images_on_device(same)
    assert module._pinned_images is buffer and module._image_copy_event is not first_copy
    torch.testing.assert_close(same["images"].cpu(), expected)

    large = _batch(size=2, crops=3)
    expected = large["images"].clone()
    module._stage_images_on_device(large)
    assert module._pinned_images is not buffer
    assert module._pinned_images.numel() == large["images"].numel()
    torch.testing.assert_close(large["images"].cpu(), expected)

    # A smaller batch after the growth reuses the larger buffer.
    module._stage_images_on_device(_batch(size=1, crops=2))
    assert module._pinned_images.numel() == large["images"].numel()
