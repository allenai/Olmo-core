import numpy as np
import pytest
import torch

from olmo_core.eval import MultimodalBlankImageEvaluator, MultimodalLMEvaluator
from olmo_core.eval.multimodal_image_pairing import (
    MultimodalImagePairDataset,
    build_bounded_image_pairs,
)
from olmo_core.exceptions import OLMoConfigurationError


def _make_evaluator() -> MultimodalLMEvaluator:
    return MultimodalLMEvaluator(
        name="pixmo-cap-validation",
        batches=iter([]),
        device=torch.device("cpu"),
    )


def test_multimodal_lm_evaluator_normalizes_summed_weighted_loss():
    evaluator = _make_evaluator()
    batch = {
        "labels": torch.tensor([[1, 2, 3, -100]]),
        "loss_masks": torch.tensor([[0.0, 1.0, 0.5, 1.0]]),
    }
    # Weighted token losses: 2 * 1.0 + 4 * 0.5 = 4.0. The ignored final label
    # contributes neither loss nor denominator weight.
    evaluator.update_metrics(batch, ce_loss=torch.tensor(4.0), logits=None)

    metrics = evaluator.compute_metrics()

    torch.testing.assert_close(metrics["CE loss"], torch.tensor(4.0 / 1.5))
    torch.testing.assert_close(metrics["PPL"], torch.exp(torch.tensor(4.0 / 1.5)))


def test_multimodal_lm_evaluator_accumulates_by_loss_weight():
    evaluator = _make_evaluator()
    evaluator.update_metrics(
        {"labels": torch.tensor([[1]]), "loss_masks": torch.tensor([[1.0]])},
        ce_loss=torch.tensor(2.0),
        logits=None,
    )
    evaluator.update_metrics(
        {"labels": torch.tensor([[1]]), "loss_masks": torch.tensor([[3.0]])},
        ce_loss=torch.tensor(12.0),
        logits=None,
    )

    metrics = evaluator.compute_metrics()

    torch.testing.assert_close(metrics["CE loss"], torch.tensor(3.5))


def test_multimodal_blank_image_control_changes_only_images():
    images = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
    batch = {
        "images": images,
        "input_ids": torch.tensor([[1, 2]]),
        "labels": torch.tensor([[2, -100]]),
        "loss_masks": torch.tensor([[1.0, 0.0]]),
    }
    evaluator = MultimodalBlankImageEvaluator(
        name="blank-image",
        batches=[batch],
        device=torch.device("cpu"),
    )

    [transformed] = list(evaluator)

    torch.testing.assert_close(transformed["images"], torch.zeros_like(images))
    torch.testing.assert_close(transformed["input_ids"], batch["input_ids"])
    torch.testing.assert_close(batch["images"], images)


def test_bounded_image_pairs_use_distinct_exact_geometry_and_no_extra_rows():
    rows = [_row(i) for i in range(8)]
    rows[2] = _row(2, image_value=0)
    rows[3] = _row(3, pooled=(1, 0))
    rows[4] = _row(4, pooled=(1, 0))
    rows[5] = _row(5, pooled=(2, 3))

    class BoundedDataset(_MutableMultimodalDataset):
        def get(self, index, epoch=0):
            assert index < 6, "Read beyond the configured candidate bound"
            return super().get(index, epoch)

    dataset = BoundedDataset(rows)
    pairs = build_bounded_image_pairs(dataset, examples=4, max_candidates=6, seed=19)
    assert pairs == build_bounded_image_pairs(dataset, examples=4, max_candidates=6, seed=19)
    assert len({recipient for recipient, _ in pairs}) == 4
    assert len({donor for _, donor in pairs}) == 4
    assert {recipient for recipient, _ in pairs} == {0, 1, 3, 4}
    correct = MultimodalImagePairDataset(dataset, pairs, wrong_images=False)
    wrong = MultimodalImagePairDataset(dataset, pairs, wrong_images=True)
    for i, (recipient, donor) in enumerate(pairs):
        assert recipient != donor
        assert np.array_equal(correct[i]["pooled_patches_idx"], wrong[i]["pooled_patches_idx"])
        assert not np.array_equal(correct[i]["images"], wrong[i]["images"])
        for name in rows[recipient]:
            if name != "images":
                assert np.array_equal(correct[i][name], wrong[i][name])
    with pytest.raises(OLMoConfigurationError, match="requested 5"):
        build_bounded_image_pairs(dataset, examples=5, max_candidates=6)
    rows[pairs[0][1]]["pooled_patches_idx"] += 100
    with pytest.raises(OLMoConfigurationError, match="geometry or distinctness changed"):
        wrong[0]


class _MutableMultimodalDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def get(self, index, epoch=0):
        assert epoch == 0
        return self.rows[index]


def _row(index: int, *, pooled=(0, 1), image_value=None):
    value = float(index if image_value is None else image_value)
    return {
        "images": np.full((1, 2, 3), value, dtype=np.float32),
        "pooled_patches_idx": np.asarray([pooled], dtype=np.int64),
        "input_ids": np.asarray([10, 20, 30 + index], dtype=np.int64),
        "labels": np.asarray([-100, 20, 30 + index], dtype=np.int64),
        "loss_masks": np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
    }


@pytest.mark.parametrize("drift", ["dtype", "geometry", "duplicate"])
def test_bounded_pairs_reject_changed_donors(drift):
    rows = [_row(index) for index in range(4)]
    dataset = _MutableMultimodalDataset(rows)
    pairs = build_bounded_image_pairs(dataset, examples=4, max_candidates=4)
    wrong = MultimodalImagePairDataset(dataset, pairs, wrong_images=True)
    recipient, donor = pairs[0]
    if drift == "dtype":
        rows[donor]["images"] = rows[donor]["images"].astype(np.float64)
    elif drift == "geometry":
        rows[donor]["pooled_patches_idx"] += 10
    else:
        rows[donor]["images"] = rows[recipient]["images"].copy()
    with pytest.raises(OLMoConfigurationError, match="geometry or distinctness changed"):
        wrong[0]
