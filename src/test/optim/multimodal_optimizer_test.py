"""
Unit tests for :mod:`olmo_core.optim.multimodal_optimizer`: per-scheduler-group clipping,
component gradient norms and partial master synchronization. The distributed machinery is
stubbed so the logic runs on CPU; the shared optimizer is exercised by its own tests.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.distributed.tensor import Replicate

from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.optim import OLMoDDPOptimizerConfig
from olmo_core.optim.adamw import foreach_adamw_step
from olmo_core.optim.multimodal_optimizer import (
    MultimodalOLMoDDPOptimizer,
    MultimodalOLMoDDPOptimizerConfig,
)


def _stub_optimizer(*, by_group: bool, grads: dict, groups: list) -> MultimodalOLMoDDPOptimizer:
    optim = object.__new__(MultimodalOLMoDDPOptimizer)
    optim.param_groups = groups
    optim.states = {f"{name}.main": SimpleNamespace(placements=[Replicate()]) for name in grads}
    optim.main_grad = grads
    optim.max_grad_norm = 1.0
    optim.check_nan_inf_grad = False
    optim.clip_grad_norm_by_scheduler_group = by_group
    optim.skip_nonfinite_grad_steps = False
    optim.latest_component_grad_norms = {}
    optim.latest_clip_group_grad_norms = {}
    optim.latest_clip_group_coefficients = {}
    optim._component_grad_norm_patterns = None
    # Single-rank stand-ins for the mesh reductions.
    optim._compute_total_grad_norm = lambda *parts: torch.linalg.vector_norm(
        torch.cat([g.reshape(-1) for part in parts for g in part] or [torch.zeros(1)])
    )
    optim._maybe_debug_nan_inf_grad_norm = lambda *args, **kwargs: None
    return optim


def _groups(params):
    return [
        {
            "pg": "dp",
            "named_params": {name: params[name] for name in ("lm.a", "lm.b")},
        },
        {
            "pg": "dp",
            "scheduler_name": "connector",
            "named_params": {"connector.w": params["connector.w"]},
        },
    ]


def _params_and_grads():
    params = {
        "lm.a": torch.nn.Parameter(torch.zeros(2)),
        "lm.b": torch.nn.Parameter(torch.zeros(2)),
        "connector.w": torch.nn.Parameter(torch.zeros(2)),
    }
    grads = {
        "lm.a": torch.tensor([3.0, 0.0]),
        "lm.b": torch.tensor([0.0, 4.0]),  # LM norm = 5
        "connector.w": torch.tensor([0.3, 0.4]),  # connector norm = 0.5
    }
    return params, grads


def test_logical_clip_groups_merge_by_scheduler_name():
    params, grads = _params_and_grads()
    params["lm.b"].requires_grad_(False)
    optim = _stub_optimizer(by_group=True, grads=grads, groups=_groups(params))
    groups = optim._logical_grad_clip_groups()
    assert list(groups) == [optim.DEFAULT_CLIP_GROUP_NAME, "connector"]
    assert groups[optim.DEFAULT_CLIP_GROUP_NAME] == ["lm.a"]
    assert groups["connector"] == ["connector.w"]


def test_clipping_by_scheduler_group_scales_each_group_independently():
    params, grads = _params_and_grads()
    optim = _stub_optimizer(by_group=True, grads=grads, groups=_groups(params))
    total = optim._clip_grad()
    # LM norm 5 is clipped to 1, the connector's 0.5 is left alone.
    torch.testing.assert_close(total, torch.tensor((5.0**2 + 0.5**2) ** 0.5))
    torch.testing.assert_close(grads["lm.a"], torch.tensor([3.0, 0.0]) / (5.0 + 1e-6))
    torch.testing.assert_close(grads["lm.b"], torch.tensor([0.0, 4.0]) / (5.0 + 1e-6))
    torch.testing.assert_close(grads["connector.w"], torch.tensor([0.3, 0.4]))
    assert set(optim.latest_clip_group_grad_norms) == {optim.DEFAULT_CLIP_GROUP_NAME, "connector"}
    torch.testing.assert_close(optim.latest_clip_group_coefficients["connector"], torch.tensor(1.0))
    assert optim.latest_component_grad_norms == {}


def test_component_grad_norms_are_reported_without_changing_clipping():
    params, grads = _params_and_grads()
    optim = _stub_optimizer(by_group=True, grads=grads, groups=_groups(params))
    optim.set_component_grad_norm_patterns({"LM": ("lm.*",), "connector": ("connector.*",)})
    optim._clip_grad()
    torch.testing.assert_close(optim.latest_component_grad_norms["LM"], torch.tensor(5.0))
    torch.testing.assert_close(optim.latest_component_grad_norms["connector"], torch.tensor(0.5))
    with pytest.raises(ValueError, match="No trainable optimizer parameters match"):
        optim.set_component_grad_norm_patterns({"vision": ("vision.*",)})
        optim._compute_component_grad_norms()
    with pytest.raises(ValueError, match="non-empty"):
        optim.set_component_grad_norm_patterns({"": ("lm.*",)})


def _history(optim, n=80):
    optim.rolling_interval_length = 128
    optim.sigma_factor = 6
    optim._losses = [torch.tensor(1.0 + 0.01 * (i % 3)) for i in range(n)]
    optim._grad_norms = [torch.tensor(0.1 + 0.001 * (i % 3)) for i in range(n)]


def test_nonfinite_grad_norm_skips_the_step_instead_of_aborting():
    params, grads = _params_and_grads()
    grads["lm.a"] = torch.tensor([float("inf"), 1.0])
    optim = _stub_optimizer(by_group=True, grads=grads, groups=_groups(params))
    optim.check_nan_inf_grad = True  # the loss check stays; the grad-norm assert must not fire
    optim.skip_nonfinite_grad_steps = True
    total = optim._clip_grad()
    assert not torch.isfinite(total)
    # The bad group is zeroed (no inf * 0 = NaN); the connector keeps its unclipped gradient.
    torch.testing.assert_close(grads["lm.a"], torch.zeros(2))
    torch.testing.assert_close(grads["lm.b"], torch.zeros(2))
    torch.testing.assert_close(grads["connector.w"], torch.tensor([0.3, 0.4]))

    _history(optim)
    optim._losses.append(torch.tensor(1.0))
    optim._grad_norms.append(total)
    assert optim.get_step_factor().item() == 0.0
    # The skipped step's norm no longer poisons the statistics the next step is compared against.
    assert all(bool(torch.isfinite(g)) for g in optim._grad_norms)
    optim._losses.append(torch.tensor(1.01))
    optim._grad_norms.append(torch.tensor(0.1))
    assert optim.get_step_factor().item() == 1.0


def test_skipped_step_leaves_weights_and_moments_unchanged():
    param, grad = torch.ones(3), torch.zeros(3)  # zeroed by the skip path
    exp_avg, exp_avg_sq, step = torch.full((3,), 0.5), torch.full((3,), 0.25), torch.tensor(7.0)
    foreach_adamw_step(
        [param],
        [grad],
        [exp_avg],
        [exp_avg_sq],
        [step],
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        step_factor=torch.tensor(0.0),
        step_increment_bugfix=True,
    )
    torch.testing.assert_close(param, torch.ones(3))
    torch.testing.assert_close(exp_avg, torch.full((3,), 0.5))
    torch.testing.assert_close(exp_avg_sq, torch.full((3,), 0.25))
    assert step.item() == 7.0


def test_finite_steps_are_unchanged_by_the_skip_option():
    params, grads = _params_and_grads()
    optim = _stub_optimizer(by_group=True, grads=grads, groups=_groups(params))
    optim.skip_nonfinite_grad_steps = True
    torch.testing.assert_close(optim._clip_grad(), torch.tensor((5.0**2 + 0.5**2) ** 0.5))
    torch.testing.assert_close(grads["lm.a"], torch.tensor([3.0, 0.0]) / (5.0 + 1e-6))
    _history(optim)
    optim._losses.append(torch.tensor(5.0))  # a loss spike is still skipped by the parent rule
    optim._grad_norms.append(torch.tensor(0.1))
    assert optim.get_step_factor().item() == 0.0
    assert optim._losses[-1].item() == 5.0


def test_skip_nonfinite_requires_scheduler_group_clipping():
    with patch.object(MultimodalOLMoDDPOptimizer.__mro__[1], "__init__", return_value=None):
        with pytest.raises(OLMoConfigurationError, match="requires clip_grad_norm_by_scheduler"):
            MultimodalOLMoDDPOptimizer(skip_nonfinite_grad_steps=True)


def test_global_clipping_path_is_the_parent_one():
    params, grads = _params_and_grads()
    optim = _stub_optimizer(by_group=False, grads=grads, groups=_groups(params))
    with patch.object(
        MultimodalOLMoDDPOptimizer.__mro__[1], "_clip_grad", return_value=torch.tensor(7.0)
    ) as parent:
        assert optim._clip_grad() == torch.tensor(7.0)
    parent.assert_called_once()
    assert optim.latest_clip_group_grad_norms == {}


def test_partial_master_sync_touches_only_the_requested_parameters():
    params, grads = _params_and_grads()
    optim = _stub_optimizer(by_group=False, grads=grads, groups=_groups(params))
    optim.should_maintain_fp32_main_param = True
    optim._copy_main_params_to_mxfp8_weights = lambda: None
    optim._refresh_rowwise_fp8_caches_from_model_params = lambda: None
    copied = []
    with patch(
        "olmo_core.optim.multimodal_optimizer.assign_full_tensor_to_dtensor",
        lambda dst, src: copied.append(dst),
    ):
        optim._copy_model_params_to_main_params({"connector.w"})
    assert copied == [optim.states["connector.w.main"]]
    with pytest.raises(KeyError, match="vision.w"):
        optim._copy_model_params_to_main_params({"vision.w"})


def test_partial_master_check_compares_only_the_requested_parameters():
    params, grads = _params_and_grads()
    optim = _stub_optimizer(by_group=False, grads=grads, groups=_groups(params))
    with torch.no_grad():
        params["connector.w"].copy_(torch.tensor([1.0, 2.0]))
    optim.states["connector.w.main"] = SimpleNamespace(
        full_tensor=lambda: torch.tensor([1.0, 2.0]), placements=[Replicate()]
    )
    optim.states["lm.a.main"] = SimpleNamespace(
        full_tensor=lambda: torch.tensor([9.0, 9.0]), placements=[Replicate()]
    )
    optim._check_model_param_main_param_the_same({"connector.w"})
    with pytest.raises(ValueError, match="not close"):
        optim._check_model_param_main_param_the_same({"lm.a"})
    with pytest.raises(KeyError, match="missing"):
        optim._check_model_param_main_param_the_same({"missing"})


def test_config_builds_the_subclass_and_round_trips():
    config = MultimodalOLMoDDPOptimizerConfig(
        lr=1e-3, clip_grad_norm_by_scheduler_group=True, foreach_chunk_size=32
    )
    assert config.optimizer() is MultimodalOLMoDDPOptimizer
    assert OLMoDDPOptimizerConfig.optimizer() is not MultimodalOLMoDDPOptimizer
    restored = Config.from_dict(config.as_config_dict())
    assert restored == config
    assert set(config.as_dict()) >= set(OLMoDDPOptimizerConfig().as_dict()) | {
        "clip_grad_norm_by_scheduler_group",
        "foreach_chunk_size",
    }
