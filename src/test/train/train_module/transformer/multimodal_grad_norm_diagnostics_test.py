"""
CPU tests for the gradient-norm diagnostics protocol between
:class:`~olmo_core.train.train_module.MultimodalOLMoDDPTrainModule` and
:class:`~olmo_core.optim.multimodal_optimizer.MultimodalOLMoDDPOptimizer`: component norms are
requested only on diagnostics steps, and the component patterns are matched against the
trainable parameters once. The distributed machinery is stubbed; the GPU train-module test covers
the real optimizer step.
"""

from types import SimpleNamespace
from typing import Dict
from unittest.mock import patch

import torch
from torch.distributed.tensor import Replicate

import olmo_core.train.train_module.transformer.multimodal_train_module as train_module_module
from olmo_core.optim.multimodal_optimizer import MultimodalOLMoDDPOptimizer
from olmo_core.train.train_module.transformer.ddp_train_module import OLMoDDPTrainModule
from olmo_core.train.train_module.transformer.multimodal_train_module import (
    COMPONENT_GRAD_NORM_PATTERNS,
    MultimodalOLMoDDPTrainModule,
)

PARAM_NAMES = ("lm.embeddings.weight", "lm.blocks.0.attention.w_q", "connector.w")


def _stub_optimizer() -> MultimodalOLMoDDPOptimizer:
    params = {name: torch.nn.Parameter(torch.zeros(2)) for name in PARAM_NAMES}
    optim = object.__new__(MultimodalOLMoDDPOptimizer)
    optim.param_groups = [
        {"pg": "dp", "named_params": {name: params[name] for name in PARAM_NAMES[:2]}},
        {
            "pg": "dp",
            "scheduler_name": "connector",
            "named_params": {"connector.w": params["connector.w"]},
        },
    ]
    optim.states = {f"{name}.main": SimpleNamespace(placements=[Replicate()]) for name in params}
    optim.main_grad = {name: torch.tensor([3.0, 4.0]) for name in PARAM_NAMES}
    optim.max_grad_norm = 1.0
    optim.check_nan_inf_grad = False
    optim.clip_grad_norm_by_scheduler_group = True
    optim.latest_component_grad_norms = {}
    optim.latest_clip_group_grad_norms = {}
    optim.latest_clip_group_coefficients = {}
    optim._component_grad_norm_patterns = None
    optim._grad_norm_plan_cache = None
    optim._step_local_norms = None
    optim._maybe_debug_nan_inf_grad_norm = lambda *args, **kwargs: None  # type: ignore[method-assign]
    optim._compute_total_grad_norm = lambda *parts: torch.linalg.vector_norm(  # type: ignore[method-assign]
        torch.cat([g.reshape(-1) for part in parts for g in part] or [torch.zeros(1)])
    )
    return optim


def _stub_train_module(optim: MultimodalOLMoDDPOptimizer, diagnostics_interval: int):
    module = object.__new__(MultimodalOLMoDDPTrainModule)
    module.optim = optim
    module.diagnostics_interval = diagnostics_interval
    module._component_grad_norm_patterns_cache = None
    module._trainer = SimpleNamespace(global_step=0)  # type: ignore[assignment]
    metrics: Dict[str, torch.Tensor] = {}

    def record_metric(name, value, reduce_type=None, namespace=None, **kwargs):
        metrics[f"{namespace}/{name}"] = value

    module.record_metric = record_metric  # type: ignore[method-assign, assignment]
    return module, metrics


def test_component_norms_only_on_diagnostics_steps_with_patterns_matched_once():
    optim = _stub_optimizer()
    module, metrics = _stub_train_module(optim, diagnostics_interval=2)

    def fake_parent_step(self):
        # The optimizer step is the only consumer of the patterns set for this step.
        optim._clip_grad()

    expected_components = {"input embeddings", "LM attention", "connector"}
    with patch.object(OLMoDDPTrainModule, "optim_step", fake_parent_step), patch.object(
        train_module_module, "fnmatch", wraps=train_module_module.fnmatch
    ) as matcher, patch.object(
        optim,
        "set_component_grad_norm_patterns",
        wraps=optim.set_component_grad_norm_patterns,
    ) as setter:
        for step in range(4):
            module._trainer.global_step = step
            metrics.clear()
            optim.main_grad = {name: torch.tensor([3.0, 4.0]) for name in PARAM_NAMES}
            module.optim_step()
            if step == 0:
                assert matcher.call_count > 0
                first_pass_calls = matcher.call_count
            if step % 2 == 0:
                assert set(optim.latest_component_grad_norms) == expected_components
                assert {f"optim/{component} grad norm" for component in expected_components} <= set(
                    metrics
                )
                assert "optim/connector clip group grad norm" in metrics
                assert "optim/language model clip coefficient" in metrics
            else:
                assert optim.latest_component_grad_norms == {}
                assert not any(name.endswith("grad norm") for name in metrics)
            # Patterns are set for the diagnostics step and cleared afterwards.
            assert optim._component_grad_norm_patterns is None
        assert [args[0] for args, _ in setter.call_args_list] == [
            {name: COMPONENT_GRAD_NORM_PATTERNS[name] for name in expected_components},
            None,
            {name: COMPONENT_GRAD_NORM_PATTERNS[name] for name in expected_components},
            None,
        ]
        # The second diagnostics step reused the first step's matching.
        assert matcher.call_count == first_pass_calls
    assert module._component_grad_norm_patterns_cache is not None
    assert module._component_grad_norm_patterns_cache[0] == PARAM_NAMES
