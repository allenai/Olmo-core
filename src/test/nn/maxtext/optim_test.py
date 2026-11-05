"""
Tests for converting AdamW optimizer state between OLMo Core and MaxText. These need neither JAX
nor a built model; ``maxtext_test.py`` checks the MaxText layout against MaxText itself.
"""

from typing import Any, Dict

import numpy as np
import pytest
import torch

from olmo_core.distributed.checkpoint import (
    get_checkpoint_metadata,
    load_keys,
    save_state_dict,
)
from olmo_core.nn.maxtext import convert_state_to_maxtext
from olmo_core.nn.maxtext.checkpoint import (
    AdamState,
    _flatten,
    _maxtext_adam_state,
    load_olmo_core_optim_state,
    load_olmo_core_weights,
    maxtext_checkpoint_items,
    save_olmo_core_weights,
    scan_params,
    unscan_params,
)
from olmo_core.nn.maxtext.convert import maxtext_shapes, olmo_core_shapes
from olmo_core.nn.maxtext.convert_checkpoint import _convert_optim, _map_moments

from .convert_test import CONFIGS, assert_states_equal, distinct

EXPERIMENT_CONFIG = {"model": {}}


def adam_state(shapes, step: int = 7) -> AdamState:
    """Moments with values distinct from each other and from :func:`distinct`'s default weights."""
    return AdamState(
        step=step, exp_avg=distinct(shapes, seed=10), exp_avg_sq=distinct(shapes, seed=11)
    )


def assert_adam_states_equal(a: AdamState, b: AdamState):
    assert a.step == b.step
    assert_states_equal(a.exp_avg, b.exp_avg)
    assert_states_equal(a.exp_avg_sq, b.exp_avg_sq)


@pytest.mark.parametrize("name", CONFIGS)
def test_olmo_core_round_trip_through_maxtext(name):
    c = CONFIGS[name]()
    optim = adam_state(olmo_core_shapes(c))
    expected = AdamState(optim.step, dict(optim.exp_avg), dict(optim.exp_avg_sq))

    maxtext = _convert_optim(c, optim, to_maxtext=True, validate=True)
    assert maxtext.step == expected.step
    for moments in maxtext.moments().values():
        assert {k: tuple(v.shape) for k, v in moments.items()} == maxtext_shapes(c)
    # The moments are laid out like the weights, so they map the same way.
    assert_states_equal(
        maxtext.exp_avg, convert_state_to_maxtext(c, distinct(olmo_core_shapes(c), seed=10))
    )

    scanned = _map_moments(maxtext, lambda m: scan_params(m, c.scan_layout, c.n_layers))
    unscanned = _map_moments(scanned, lambda m: unscan_params(m, c.scan_layout))
    back = _convert_optim(c, unscanned, to_maxtext=False, validate=True)
    assert_adam_states_equal(back, expected)


def save_fsdp_checkpoint(path, weights, optim: AdamState):
    """
    A checkpoint laid out like the FSDP train module's: the optimizer state as
    ``get_optimizer_state_dict(..., flatten_optimizer_state_dict=True)`` returns it.
    """
    optim_sd: Dict[str, Any] = {}
    for name in weights:
        optim_sd[f"state.{name}.exp_avg"] = optim.exp_avg[name]
        optim_sd[f"state.{name}.exp_avg_sq"] = optim.exp_avg_sq[name]
        optim_sd[f"state.{name}.step"] = torch.tensor(float(optim.step))
        optim_sd[f"param_groups.{name}.lr"] = 1e-3
    save_state_dict(path / "model_and_optim", {"model": weights, "optim": optim_sd})


@pytest.mark.parametrize("name", CONFIGS)
def test_olmo_ddp_checkpoint_round_trip(tmp_path, name):
    c = CONFIGS[name]()
    weights = distinct(olmo_core_shapes(c))
    optim = adam_state(olmo_core_shapes(c), step=123)
    save_olmo_core_weights(tmp_path, weights, EXPERIMENT_CONFIG, optim=optim)

    keys = set(get_checkpoint_metadata(tmp_path / "model_and_optim").state_dict_metadata)
    name0 = next(iter(weights))
    # OLMoDDP's flat entries, plus the model weights the FSDP train module loads.
    for key in (
        f"model.{name0}",
        *(f"module.{name0}.{s}" for s in ("main", "exp_avg", "exp_avg_sq", "step")),
    ):
        assert key in keys

    assert_states_equal(load_olmo_core_weights(tmp_path, c), weights)
    loaded = load_olmo_core_optim_state(tmp_path, c)
    assert loaded is not None
    assert_adam_states_equal(loaded, optim)


def test_fsdp_checkpoint_optim_state(tmp_path):
    c = CONFIGS["dense"]()
    weights = distinct(olmo_core_shapes(c))
    optim = adam_state(olmo_core_shapes(c), step=42)
    save_fsdp_checkpoint(tmp_path, weights, optim)

    loaded = load_olmo_core_optim_state(tmp_path, c)
    assert loaded is not None
    assert_adam_states_equal(loaded, optim)


def test_weights_only_checkpoint_has_no_optim_state(tmp_path):
    c = CONFIGS["dense"]()
    save_olmo_core_weights(tmp_path, distinct(olmo_core_shapes(c)), EXPERIMENT_CONFIG)
    assert load_olmo_core_optim_state(tmp_path, c) is None


def test_mismatched_steps_fail(tmp_path):
    c = CONFIGS["dense"]()
    weights = distinct(olmo_core_shapes(c))
    save_olmo_core_weights(
        tmp_path, weights, EXPERIMENT_CONFIG, optim=adam_state(olmo_core_shapes(c))
    )
    # Overwrite with one parameter a step behind the rest.
    path = tmp_path / "model_and_optim"
    name0 = next(iter(weights))
    keys = list(get_checkpoint_metadata(path).state_dict_metadata)
    state = dict(zip(keys, load_keys(path, keys)))
    state[f"module.{name0}.step"] = torch.tensor(6.0)
    save_state_dict(path, state, save_overwrite=True)
    with pytest.raises(ValueError, match="different numbers of optimizer steps"):
        load_olmo_core_optim_state(tmp_path, c)


def test_non_adam_state_fails(tmp_path):
    c = CONFIGS["dense"]()
    weights = distinct(olmo_core_shapes(c))
    name0 = next(iter(weights))
    save_state_dict(
        tmp_path / "model_and_optim",
        {"model": weights, f"module.{name0}.muon_momentum": torch.zeros(3)},
    )
    with pytest.raises(ValueError, match="non-AdamW"):
        load_olmo_core_optim_state(tmp_path, c)


def maxtext_shaped_state(name: str):
    c = CONFIGS[name]()
    shapes = maxtext_shapes(c)
    return distinct(shapes), adam_state(shapes, step=99)


@pytest.mark.parametrize("name", CONFIGS)
def test_maxtext_items_round_trip(name):
    params, optim = maxtext_shaped_state(name)
    items = maxtext_checkpoint_items(params, optim)
    assert set(items) == {"params", "step", "opt_state"}
    assert int(items["step"]) == optim.step
    # optax.adamw's chain: Adam state, the (empty) weight decay state, then the schedule's count.
    adam, decay, schedule = items["opt_state"]
    assert decay is None
    assert int(adam["count"]) == int(schedule["count"]) == optim.step
    assert set(adam["mu"]) == set(adam["nu"]) == {"params"}

    found = _maxtext_adam_state(_flatten(items["opt_state"]), "test")
    assert_adam_states_equal(found, optim)
    assert_states_equal(
        {k: torch.from_numpy(v) for k, v in _flatten(items["params"]["params"]).items()}, params
    )


def test_maxtext_adam_state_inside_skip_step_wrapper():
    params, optim = maxtext_shaped_state("dense")
    chain = maxtext_checkpoint_items(params, optim)["opt_state"]
    wrapped: Dict[str, Any] = {
        "inner_state": chain,
        "losses": np.zeros(128, np.float32),
        "grad_norms": np.zeros(128, np.float32),
        "count": np.asarray(5, np.int32),
        "is_skipped": np.asarray(False),
    }
    assert_adam_states_equal(_maxtext_adam_state(_flatten(wrapped), "test"), optim)


def test_maxtext_adam_pax_state_fails():
    params, optim = maxtext_shaped_state("dense")
    adam = maxtext_checkpoint_items(params, optim)["opt_state"][0]
    with pytest.raises(ValueError, match="only opt_type=adamw"):
        _maxtext_adam_state(_flatten(adam), "test")


def test_maxtext_state_without_second_moments_fails():
    params, optim = maxtext_shaped_state("dense")
    adam = maxtext_checkpoint_items(params, optim)["opt_state"][0]
    del adam["nu"]
    with pytest.raises(ValueError, match="no second moments"):
        _maxtext_adam_state(_flatten([adam]), "test")


def test_maxtext_items_reject_mismatched_moments():
    params, optim = maxtext_shaped_state("dense")
    optim.exp_avg.pop(next(iter(optim.exp_avg)))
    with pytest.raises(KeyError, match="first moments"):
        maxtext_checkpoint_items(params, optim)
