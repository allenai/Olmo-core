"""Compact images through the OLMoDDP multimodal train module: micro-batching by crop counts,
the model kwargs, the data metrics, and parity with the padded layout on a tiny model."""

import importlib

import numpy as np
import pytest
import torch

from olmo_core.data.multimodal.collator import MultimodalCollator
from olmo_core.data.utils import split_batch
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.train.train_module.transformer.multimodal_train_module import (
    CompactImages,
    MultimodalOLMoDDPTrainModule,
    MultimodalTransformerTrainModule,
    _trim_microbatch_image_padding,
)

_N_PATCHES, _PATCH_DIM, _SEQ_LEN = 4, 14 * 14 * 3, 12
_IMAGE_PATCH_TOKEN = 1  # the tiny model's <im_patch> id (see test.nn.vision.multimodal_test)


def _example(n_crops: int, seed: int):
    """One example with one pooled ``<im_patch>`` token per crop, at the front of the sequence."""
    rng = np.random.default_rng(seed)
    input_ids = rng.integers(2, 200, size=_SEQ_LEN).astype(np.int64)
    input_ids[:n_crops] = _IMAGE_PATCH_TOKEN
    loss_masks = np.zeros(_SEQ_LEN, dtype=np.float32)
    loss_masks[_SEQ_LEN // 2 :] = 1.0
    return dict(
        input_ids=input_ids,
        labels=input_ids.copy(),
        loss_masks=loss_masks,
        position_ids=np.arange(_SEQ_LEN, dtype=np.int64),
        token_type_ids=(input_ids == _IMAGE_PATCH_TOKEN).astype(np.int64),
        images=rng.standard_normal((n_crops, _N_PATCHES, _PATCH_DIM)).astype(np.float32),
        pooled_patches_idx=(
            np.arange(n_crops * _N_PATCHES, dtype=np.int64).reshape(n_crops, _N_PATCHES)
            if n_crops
            else np.full((0, _N_PATCHES), -1, dtype=np.int64)
        ),
    )


def _batches(counts):
    examples = [_example(n, seed) for seed, n in enumerate(counts)]
    padded = MultimodalCollator(pad_token_id=0, batch_metadata=True)(examples)
    compact = MultimodalCollator(pad_token_id=0, batch_metadata=True, compact_images=True)(examples)
    return padded, compact


def _module(*, trim: bool = False) -> MultimodalOLMoDDPTrainModule:
    module = object.__new__(MultimodalOLMoDDPTrainModule)
    module._pp_config = None
    module.response_logits_only = True
    module.trim_microbatch_image_padding = trim
    return module


def _real_crops(padded: torch.Tensor, counts) -> torch.Tensor:
    return torch.cat([padded[b, :n] for b, n in enumerate(counts)])


@pytest.mark.parametrize("size", [1, 2, 3, 5])
def test_compact_images_are_split_by_cumulative_crop_counts(size):
    counts = [2, 0, 3, 1, 0]
    padded, compact = _batches(counts)
    batch = dict(compact)
    batch["images"] = CompactImages(compact["images"], counts)
    micro_batches = split_batch(batch, size)
    assert len(micro_batches) == -(-len(counts) // size)
    for index, micro_batch in enumerate(micro_batches):
        rows = range(index * size, min((index + 1) * size, len(counts)))
        images = micro_batch["images"]
        assert isinstance(images, CompactImages)
        assert images.crop_counts == [counts[b] for b in rows]
        assert images.crop_counts == micro_batch["image_crop_counts"].tolist()
        # The micro-batch's crops are a view of the collated tensor, not a copy.
        expected = _real_crops(padded["images"], counts)[
            sum(counts[: rows.start]) : sum(counts[: rows.stop])
        ]
        torch.testing.assert_close(images.tensor(), expected, rtol=0, atol=0)
        assert (
            images.tensor().untyped_storage().data_ptr()
            == compact["images"].untyped_storage().data_ptr()
        )


def test_compact_images_reject_mismatched_counts():
    _, compact = _batches([1, 2])
    with pytest.raises(OLMoConfigurationError, match="do not match"):
        CompactImages(compact["images"], [1, 1])
    with pytest.raises(OLMoConfigurationError, match="do not match"):
        CompactImages(compact["images"][None], [1, 2])


@pytest.mark.parametrize("trim", [False, True])
def test_prepare_batch_keeps_the_crop_counts_with_compact_images(trim):
    counts = [2, 0]
    _, compact = _batches(counts)
    batch = dict(compact)
    batch["images"] = CompactImages(compact["images"], counts)
    micro_batch = split_batch(batch, 2)[0]
    inputs, labels, kwargs = _module(trim=trim)._prepare_batch(micro_batch)
    assert inputs is compact["input_ids"] and labels is compact["labels"]
    assert isinstance(kwargs["images"], torch.Tensor)
    assert kwargs["images"].shape == (2, _N_PATCHES, _PATCH_DIM)
    assert kwargs["image_crop_counts"].tolist() == counts
    assert "pooled_token_counts" not in kwargs
    # Trimming leaves compact images alone and still drops padded pooled rows.
    assert kwargs["pooled_patches_idx"].shape[1] == (2 if trim else 2)
    # The original batch dict is not altered.
    assert isinstance(micro_batch["images"], CompactImages)


def test_prepare_batch_accepts_a_plain_compact_tensor_and_requires_counts():
    counts = [1, 1]
    _, compact = _batches(counts)
    _, _, kwargs = _module()._prepare_batch(dict(compact))
    assert kwargs["images"].shape == (2, _N_PATCHES, _PATCH_DIM)
    assert kwargs["image_crop_counts"].tolist() == counts
    without_counts = {k: v for k, v in compact.items() if k != "image_crop_counts"}
    with pytest.raises(OLMoConfigurationError, match="image_crop_counts"):
        _module()._prepare_batch(without_counts)


def test_trim_passes_compact_images_through_and_trims_pooled_rows():
    counts = [0, 3, 1]
    _, compact = _batches(counts)
    batch = dict(compact)
    batch["pooled_patches_idx"] = torch.cat(
        [batch["pooled_patches_idx"], torch.full((3, 2, _N_PATCHES), -1)], dim=1
    )
    result = _trim_microbatch_image_padding(batch)
    assert result["images"] is batch["images"]
    assert result["pooled_patches_idx"].shape == (3, 3, _N_PATCHES)
    bad = dict(batch)
    bad["image_crop_counts"] = torch.tensor([0, 2, 1])
    with pytest.raises(OLMoConfigurationError, match="image_crop_counts"):
        _trim_microbatch_image_padding(bad)


class _MetricTrainerStub:
    def __init__(self):
        self.global_step = 1
        self.metrics = {}

    def record_metric(self, name, value, *args, namespace=None, **kwargs):
        del args, kwargs
        self.metrics[f"{namespace}/{name}" if namespace else name] = value


@pytest.mark.parametrize("compact_layout", [False, True])
def test_data_metrics_report_real_crops_for_compact_images(compact_layout):
    counts = [2, 0, 3, 1]
    padded, compact = _batches(counts)
    batch = compact if compact_layout else padded
    module = _module()
    module._trainer = _MetricTrainerStub()  # type: ignore[assignment]
    module.source_loss_mass_targets = None
    module._record_data_metrics(batch)
    metrics = module._trainer.metrics
    assert float(metrics["data/real crops per sequence"]) == pytest.approx(1.5)
    if compact_layout:
        assert float(metrics["data/padded crops per sequence"]) == pytest.approx(1.5)
        assert float(metrics["data/crop utilization"]) == pytest.approx(1.0)
    else:
        assert float(metrics["data/padded crops per sequence"]) == pytest.approx(3.0)
        assert float(metrics["data/crop utilization"]) == pytest.approx(6 / 12)


def test_extra_flops_count_the_real_crops_of_compact_images():
    helpers = importlib.import_module("test.nn.vision.multimodal_test")
    model = helpers._tiny_multimodal_cfg().build(init_device="cpu")
    counts = [2, 0, 3]
    padded, compact = _batches(counts)
    module = _module()
    module.model_parts = [model]
    flops_padded = module.extra_flops_per_batch(padded)
    flops_compact = module.extra_flops_per_batch(compact)
    pooled = int((padded["input_ids"] == _IMAGE_PATCH_TOKEN).sum())
    assert flops_padded == model.image_encoder_flops(3 * 3, _N_PATCHES, pooled)
    assert flops_compact == model.image_encoder_flops(5, _N_PATCHES, pooled)
    as_list = dict(compact)
    as_list["images"] = CompactImages(compact["images"], counts)
    assert module.extra_flops_per_batch(as_list) == flops_compact


@pytest.mark.parametrize("counts", [[2, 0, 3, 1, 0], [0, 0, 0], [1]])
@pytest.mark.parametrize("size", [2, 5])
def test_micro_batches_of_compact_and_padded_batches_give_the_same_outputs(counts, size):
    """The collator's two layouts, micro-batched the way the train module does it, drive the
    tiny model to identical logits per micro-batch (an all-text micro-batch included)."""
    helpers = importlib.import_module("test.nn.vision.multimodal_test")
    torch.manual_seed(0)
    model = helpers._tiny_multimodal_cfg().build(init_device="cpu").eval()
    padded, compact = _batches(counts)
    compact_batch = dict(compact)
    compact_batch["images"] = CompactImages(compact["images"], counts)
    padded_micro = split_batch(dict(padded), size)
    compact_micro = split_batch(compact_batch, size)
    assert len(padded_micro) == len(compact_micro)
    for padded_mb, compact_mb in zip(padded_micro, compact_micro):
        outputs = []
        for module, micro_batch in ((_module(trim=True), padded_mb), (_module(), compact_mb)):
            input_ids, _, kwargs = module._prepare_batch(dict(micro_batch))
            for key in ("response_logits_only", "router_token_mask", "loss_masks"):
                kwargs.pop(key, None)
            outputs.append(model(input_ids, **kwargs))
        assert ("image_crop_counts" in kwargs) is True
        torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)


def test_the_stage1_train_module_rejects_compact_images():
    _, compact = _batches([1, 0])
    module = object.__new__(MultimodalTransformerTrainModule)
    with pytest.raises(OLMoConfigurationError, match="OLMoDDP"):
        module.train_batch(dict(compact))
