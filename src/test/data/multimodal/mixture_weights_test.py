import pytest

from olmo_core.data.multimodal.mixture_weights import sampling_weights_from_loss_mass


def test_sampling_weights_preserve_order_without_mutating_inputs():
    targets = {"text": 7.0, "caption": 2.0, "count": 1.0}
    means = {"count": 5.0, "text": 10.0, "caption": 4.0}
    original_targets = targets.copy()
    original_means = means.copy()

    actual = sampling_weights_from_loss_mass(targets, means)

    assert actual == pytest.approx({"text": 0.5, "caption": 5 / 14, "count": 1 / 7})
    assert list(actual) == list(targets)
    assert sum(actual.values()) == pytest.approx(1.0)
    assert targets == original_targets
    assert means == original_means


@pytest.mark.parametrize(
    "targets,means,error",
    [
        ({}, {}, "target_loss_mass must not be empty"),
        ({"a": 1.0}, {}, "mean_loss_weight must not be empty"),
        ({"a": 1.0}, {"b": 1.0}, "source mismatch"),
        ({"a": 1.0, "b": 1.0}, {"a": 1.0}, "source mismatch"),
        ({"a": 1.0}, {"a": 1.0, "b": 1.0}, "source mismatch"),
        ({"a": 0.0}, {"a": 1.0}, "target_loss_mass values must be positive"),
        ({"a": -1.0}, {"a": 1.0}, "target_loss_mass values must be positive"),
        ({"a": float("nan")}, {"a": 1.0}, "target_loss_mass values must be positive"),
        ({"a": float("inf")}, {"a": 1.0}, "target_loss_mass values must be positive"),
        ({"a": 1.0}, {"a": 0.0}, "mean_loss_weight values must be positive"),
        ({"a": 1.0}, {"a": -1.0}, "mean_loss_weight values must be positive"),
        ({"a": 1.0}, {"a": float("nan")}, "mean_loss_weight values must be positive"),
        ({"a": 1.0}, {"a": float("inf")}, "mean_loss_weight values must be positive"),
        ({"a": 1e308, "b": 1e308}, {"a": 1.0, "b": 1.0}, "non-positive total mass"),
    ],
)
def test_sampling_weights_reject_invalid_inputs(targets, means, error):
    with pytest.raises(ValueError, match=error):
        sampling_weights_from_loss_mass(targets, means)
