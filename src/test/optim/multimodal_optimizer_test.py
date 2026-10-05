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
from olmo_core.optim import OLMoDDPOptimizerConfig
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
