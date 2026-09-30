"""Fresh balancing counts must never change replay dispatch or its policy gradient."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.moe.loss import MoELoadBalancingLossGranularity
from olmo_core.nn.moe.router import MoERouterGatingFunction
from olmo_core.nn.moe.v2.router import MoERouterConfigV2


def build(source="executed", **kwargs):
    router = MoERouterConfigV2(
        d_model=3,
        num_experts=3,
        top_k=1,
        dtype=DType.float32,
        lb_loss_weight=0.1,
        lb_loss_count_source=source,
        **kwargs,
    ).build()
    with torch.no_grad():
        router.weight.copy_(torch.eye(3).reshape_as(router.weight))
    return router


def run(source, replay=True, recompute=False, **kwargs):
    router = build(source, **kwargs)
    x = torch.tensor(
        [[[3.0, 1.0, 0.0], [2.0, 0.0, 1.0]], [[1.0, 3.0, 0.0], [2.0, 1.0, 0.0]]], requires_grad=True
    )
    if replay:
        router.replay_expert_indices = torch.full((2, 2, 1), 2)

    def forward(x):
        weights, indices, counts, info = router(x, False, loss_div_factor=4.0)
        aux = router.compute_aux_loss(*info, accumulate_metrics=False)
        return weights, indices, counts, aux

    weights, indices, counts, aux = (
        checkpoint(forward, x, use_reentrant=False) if recompute else forward(x)
    )
    policy = torch.autograd.grad(weights.square().sum(), (router.weight, x), retain_graph=True)
    auxiliary = torch.autograd.grad(aux, (router.weight, x))
    return weights, indices, counts, aux, policy, auxiliary


@pytest.mark.parametrize("granularity", list(MoELoadBalancingLossGranularity))
@pytest.mark.parametrize("recompute", [False, True])
def test_router_selected_counts_preserve_dispatch_and_policy_and_match_fresh_reference(
    granularity, recompute
):
    kwargs = dict(lb_loss_granularity=granularity, recompute=recompute)
    old = run("executed", **kwargs)
    current = run("router_selected", **kwargs)
    reference = run("executed", replay=False, **kwargs)
    for a, b in zip(old[:3], current[:3]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.equal(current[1], torch.full_like(current[1], 2))
    for a, b in zip(old[4], current[4]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(current[3], reference[3])
    for a, b in zip(current[5], reference[5]):
        torch.testing.assert_close(a, b)
    assert not torch.allclose(old[5][0], current[5][0])
    # The dense probabilities and detached fresh counts give the analytic loss.
    scores = torch.tensor(
        [[[3.0, 1.0, 0.0], [2.0, 0.0, 1.0]], [[1.0, 3.0, 0.0], [2.0, 1.0, 0.0]]]
    ).softmax(-1)
    if granularity == MoELoadBalancingLossGranularity.local_batch:
        expected = 0.1 * 3 * (scores.sum((0, 1)) * torch.tensor([3, 1, 0])).sum() / 16
        torch.testing.assert_close(current[3], expected)


def test_no_replay_is_identical():
    old, current = run("executed", replay=False), run("router_selected", replay=False)
    for a, b in zip(old[:4], current[:4]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(old[5], current[5]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_z_loss_unchanged_and_metrics_still_measure_dispatch():
    results = []
    for source in ("executed", "router_selected"):
        router = build(source, z_loss_weight=0.2)
        router.lb_loss_weight = None
        router.replay_expert_indices = torch.full((1, 2, 1), 2)
        x = torch.tensor([[[3.0, 1.0, 0.0], [2.0, 0.0, 1.0]]], requires_grad=True)
        _, _, counts, info = router(x, False, loss_div_factor=2.0)
        loss = router.compute_aux_loss(*info)
        results.append((loss, torch.autograd.grad(loss, (router.weight, x))))
        torch.testing.assert_close(router.batch_size_per_expert, counts.float())
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for a, b in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"global_load_balancing": True},
        {"gating_function": MoERouterGatingFunction.sigmoid},
        {"bias_gamma": 0.1},
        {"score_correction_bias": True},
        {"n_group": 1},
        {"uniform_expert_assignment": True},
        {"random_expert_assignment": True},
    ],
)
def test_unsupported_router_selected_count_modes_fail_explicitly(kwargs):
    with pytest.raises(OLMoConfigurationError):
        build("router_selected", **kwargs)


def test_default_and_invalid_source():
    assert MoERouterConfigV2(d_model=3, num_experts=3, top_k=1).lb_loss_count_source == "executed"
    with pytest.raises(OLMoConfigurationError):
        build("unknown")


def test_existing_positional_balancing_argument_keeps_its_meaning():
    config = MoERouterConfigV2(
        8,
        4,
        2,
        False,
        None,
        None,
        False,
        False,
        None,
        MoERouterGatingFunction.softmax,
        None,
        False,
        0.1,
        MoELoadBalancingLossGranularity.instance,
    )
    router = config.build()
    assert router.lb_loss_granularity == MoELoadBalancingLossGranularity.instance
    assert router.lb_loss_count_source == "executed"
