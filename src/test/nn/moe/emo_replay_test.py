"""EMO replay and full-pool execution against independent differentiable weights."""

from unittest import mock

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint

from olmo_core.nn.moe import emo
from olmo_core.nn.moe.v2 import replay, router


def build_router(*, full_pool=False, gating="softmax", normalize=None):
    model_router = router.MoERouterConfigV2(
        d_model=4,
        num_experts=4,
        top_k=2,
        gating_function=gating,
        normalize_expert_weights=normalize,
        restore_weight_scale=True,
        emo=emo.EmoRouterConfig(
            eos_token_id=0,
            min_document_expert_pool=2,
            max_document_expert_pool=3,
            eval_document_expert_pool=4,
            full_pool=full_pool,
        ),
    ).build()
    with torch.no_grad():
        model_router.weight.copy_(torch.arange(16).float().sin())
    return model_router


@pytest.mark.parametrize("gating", ["softmax", "topk_softmax"])
@pytest.mark.parametrize("normalize", [None, 1.0])
@pytest.mark.parametrize("recompute", [None, "reentrant", "non_reentrant"])
@pytest.mark.parametrize("full_pool", [False, True])
def test_replay_matches_independent_weights_and_gradients(gating, normalize, recompute, full_pool):
    torch.manual_seed(18)
    model_router = build_router(full_pool=full_pool, gating=gating, normalize=normalize)
    model = nn.Module()
    model.add_module("routed_experts_router", model_router)
    x = torch.randn(1, 3, 4, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    reference_weight = model_router.weight.detach().clone().requires_grad_()
    ids = torch.tensor([[[0, 3], [1, 2], [3, 2]]])
    coefficients = torch.tensor([[[1.0, 3.0], [2.0, 7.0], [5.0, 4.0]]])
    seen = []

    def forward(value):
        seen.append(model_router.replay_expert_indices.detach().clone())
        weights, actual, counts, _ = model_router(value, scores_only=False)
        assert torch.equal(actual, ids)
        assert counts.sum() == 6
        return weights

    with (
        mock.patch.object(model_router, "_pool_sizes", side_effect=AssertionError("resampled")),
        replay.replay_routes(model, {"routed_experts_router": ids}),
    ):
        assert not model_router.requires_segment_ids
        weights = (
            checkpoint.checkpoint(forward, x, use_reentrant=recompute == "reentrant")
            if recompute
            else forward(x)
        )
        (weights * coefficients).sum().backward()

    logits = F.linear(reference_x, reference_weight.view(4, 4))
    expected = (
        logits.gather(-1, ids).softmax(-1)
        if gating == "topk_softmax"
        else logits.softmax(-1).gather(-1, ids)
    )
    if normalize is not None:
        expected = expected / expected.abs().sum(-1, keepdim=True)
    expected = expected * 2
    (expected * coefficients).sum().backward()
    assert len(seen) == (2 if recompute else 1)
    assert all(torch.equal(actual, ids) for actual in seen)
    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(x.grad, reference_x.grad)
    torch.testing.assert_close(model_router.weight.grad, reference_weight.grad)
    assert torch.isfinite(model_router.weight.grad).all()
    assert model_router.weight.grad.abs().sum() > 0
    assert model_router.replay_expert_indices is None
    assert model_router.requires_segment_ids == (not full_pool)


def test_full_pool_is_token_local_and_independent_of_training_mode():
    model_router = build_router(full_pool=True)
    x = torch.arange(12).float().cos().reshape(1, 3, 4)
    with mock.patch.object(model_router, "_pool_sizes", side_effect=AssertionError("resampled")):
        training = model_router(x, False)
        model_router.eval()
        evaluation = model_router(x, False)
        changed = x.clone()
        changed[:, 1:] *= -100
        perturbed = model_router(changed, False)
    logits = F.linear(x, model_router.weight.view(4, 4))
    expected_weights, expected_ids = logits.softmax(-1).topk(2)
    torch.testing.assert_close(training[0], expected_weights * 2)
    assert torch.equal(training[1], expected_ids)
    torch.testing.assert_close(training[:3], evaluation[:3])
    torch.testing.assert_close(training[0][:, :1], perturbed[0][:, :1])
    assert torch.equal(training[1][:, :1], perturbed[1][:, :1])


def test_emo_nested_replay_restores_native_routing_after_exception():
    model_router = build_router()
    model = nn.Module()
    model.add_module("routed_experts_router", model_router)
    outer = torch.tensor([[[0, 1]]])
    inner = torch.tensor([[[2, 3]]])
    with replay.replay_routes(model, {"routed_experts_router": outer}):
        with pytest.raises(RuntimeError, match="injected"):
            with replay.replay_routes(model, {"routed_experts_router": inner}):
                assert model_router.replay_expert_indices is inner
                raise RuntimeError("injected")
        assert model_router.replay_expert_indices is outer
    assert model_router.replay_expert_indices is None
    assert model_router.requires_segment_ids
