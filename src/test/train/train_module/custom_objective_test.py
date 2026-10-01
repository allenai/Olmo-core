"""Numerical checks against real Core modules and HF reference models."""

import contextlib
from typing import Any, cast

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.moe.router import MoERouter, MoERouterConfig
from olmo_core.nn.moe.v2.replay import replay_routes
from olmo_core.nn.moe.v2.router import MoERouterConfigV2, MoERouterV2
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig
from olmo_core.train.train_module.transformer import TransformerTrainModuleConfig
from olmo_core.train.train_module.transformer.objective import train_batch_with_loss


class ObjectiveModule:
    def __init__(self, model):
        self.model = model
        self.contexts = []

    @contextlib.contextmanager
    def _train_microbatch_context(self, index, count):
        self.contexts.append((index, count))
        yield


def test_custom_objective_accumulation_matches_full_batch():
    torch.manual_seed(17)
    model = nn.Linear(4, 3)
    reference = nn.Linear(4, 3)
    reference.load_state_dict(model.state_dict())
    x = torch.randn(5, 4)
    labels = torch.tensor([0, 1, 2, 0, 2])
    module = ObjectiveModule(model)

    def objective(module, batch):
        loss = F.cross_entropy(module.model(batch["x"]), batch["y"], reduction="sum") / 5
        return loss, {"loss": loss}

    metrics = train_batch_with_loss(
        module, [{"x": x[:2], "y": labels[:2]}, {"x": x[2:], "y": labels[2:]}], objective
    )
    F.cross_entropy(reference(x), labels).backward()
    torch.testing.assert_close(model.weight.grad, reference.weight.grad)
    assert module.contexts == [(0, 2), (1, 2)]
    assert all(not metric["loss"].requires_grad for metric in metrics)


@pytest.mark.parametrize("recompute", [False, True])
def test_router_replay_keeps_experts_and_router_gradients(recompute):
    router = MoERouterConfigV2(d_model=8, num_experts=4, top_k=2).build()
    (
        router.reset_parameters()
        if hasattr(router, "reset_parameters")
        else nn.init.normal_(router.weight, std=0.1)
    )
    model = nn.Module()
    model.add_module("routed_experts_router", router)
    indices = torch.tensor([[[0, 2], [1, 3], [2, 3]]])
    x = torch.randn(1, 3, 8, requires_grad=True)
    seen = []

    def forward(x):
        weights, actual, _, _ = router(x, scores_only=False)
        seen.append(actual.detach().clone())
        return weights

    with replay_routes(model, {"routed_experts_router": indices}):
        weights = checkpoint(forward, x, use_reentrant=True) if recompute else forward(x)
        (weights * torch.tensor([1.0, 2.0])).sum().backward()
    assert seen and all(torch.equal(actual, indices) for actual in seen)
    assert torch.isfinite(router.weight.grad).all() and router.weight.grad.abs().sum() > 0
    assert router.replay_expert_indices is None


def test_replay_rejects_emo_before_setting_router_state():
    from olmo_core.nn.moe.emo import EmoRouterConfig

    model = nn.Module()
    standard = MoERouterConfigV2(d_model=4, num_experts=4, top_k=2).build()
    emo = MoERouterConfigV2(
        d_model=4,
        num_experts=4,
        top_k=2,
        emo=EmoRouterConfig(eos_token_id=0, min_document_expert_pool=4, max_document_expert_pool=4),
    ).build()
    model.add_module("first_routed_experts_router", standard)
    model.add_module("emo_routed_experts_router", emo)
    routes = {name: torch.tensor([[[0, 1]]]) for name, _ in model.named_children()}
    with pytest.raises(ValueError, match="does not support EmoRouterV2"):
        with replay_routes(model, routes):
            pytest.fail("Unsupported replay must fail before entering the context")
    assert getattr(standard, "replay_expert_indices", None) is None
    assert getattr(emo, "replay_expert_indices", None) is None


def test_nested_replay_restores_routes_after_failure():
    model = nn.Module()
    router = MoERouterConfigV2(d_model=4, num_experts=4, top_k=2).build()
    model.add_module("routed_experts_router", router)
    outer, inner = torch.tensor([[[0, 1]]]), torch.tensor([[[2, 3]]])
    with replay_routes(model, {"routed_experts_router": outer}):
        with pytest.raises(RuntimeError, match="injected"):
            with replay_routes(model, {"routed_experts_router": inner}):
                assert router.replay_expert_indices is inner
                raise RuntimeError("injected")
        assert router.replay_expert_indices is outer
    assert router.replay_expert_indices is None


@pytest.mark.parametrize("objective_fails", [False, True])
def test_eval_after_custom_objective_restores_eval_mode(objective_fails):
    model = TransformerConfig.llama_like(
        d_model=16,
        vocab_size=32,
        n_layers=1,
        n_heads=2,
        feed_forward=FeedForwardConfig(hidden_size=32, bias=False),
        attn_backend=AttentionBackendName.torch,
    ).build(init_device="cpu")
    module = TransformerTrainModuleConfig(
        rank_microbatch_size=4,
        max_sequence_length=4,
        optim=AdamWConfig(),
    ).build(model, device=torch.device("cpu"))
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
    forward_modes = []
    hook = module.model.register_forward_pre_hook(
        lambda model, _args: forward_modes.append(model.training)
    )

    def objective(module, batch):
        logits = module.model_forward(batch["input_ids"])
        assert isinstance(logits, torch.Tensor)
        if objective_fails:
            raise RuntimeError("injected objective failure")
        loss = logits.square().mean()
        return loss, {"loss": loss}

    try:
        before = module.eval_batch(dict(batch))
        module.zero_grads()
        with (
            pytest.raises(RuntimeError, match="injected objective failure")
            if objective_fails
            else contextlib.nullcontext()
        ):
            module.train_batch_with_loss([batch], objective)
        after = module.eval_batch(dict(batch))
    finally:
        hook.remove()

    # Check the mode at the actual forwards, including after a failed objective.
    assert forward_modes == [False, True, False]
    assert not module.model.training
    # No optimizer step occurred, so evaluation must still give the same logits.
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    if not objective_fails:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize("generation", [1, 2])
@pytest.mark.parametrize("part_index", [None, 0, 1])
def test_custom_objective_rejects_bias_gamma_before_training(generation, part_index, monkeypatch):
    router: MoERouter | MoERouterV2 = (
        MoERouterConfig(bias_gamma=1e-3).build(d_model=4, num_experts=4)
        if generation == 1
        else MoERouterConfigV2(d_model=4, num_experts=4, top_k=1, bias_gamma=1e-3).build()
    )
    model = nn.Module()
    model.add_module("routed_experts_router", router)
    models = [model]
    if part_index is not None:
        models = [nn.Linear(4, 4), nn.Linear(4, 4)]
        models[part_index] = model
    for part in models:
        part.eval()
    module = ObjectiveModule(models[0])
    if part_index is not None:
        monkeypatch.setattr(module, "model_parts", models, raising=False)

    def objective(module, batch):
        pytest.fail("Unsupported balancing must be rejected before the objective runs")

    with pytest.raises(
        NotImplementedError,
        match=rf"bias_gamma is set on model part {part_index or 0}, module routed_experts_router",
    ):
        train_batch_with_loss(module, [{}], objective)
    assert module.contexts == []
    assert all(not part.training for part in models)
    assert all(parameter.grad is None for part in models for parameter in part.parameters())
    assert torch.count_nonzero(router.score_bias) == 0
    assert torch.count_nonzero(router.score_bias_batch_size_per_expert) == 0


@pytest.mark.parametrize("aux_weight", [0.0, 0.1])
def test_custom_objective_supports_auxiliary_losses_including_zero(aux_weight):
    router = MoERouterConfigV2(
        d_model=4, num_experts=4, top_k=2, lb_loss_weight=aux_weight, z_loss_weight=aux_weight
    ).build()
    model = nn.Module()
    model.add_module("routed_experts_router", router)
    module = ObjectiveModule(model)
    x = torch.randn(1, 3, 4)

    def objective(module, batch):
        weights, _, _, info = module.model.routed_experts_router(
            batch["x"], scores_only=False, loss_div_factor=3.0
        )
        aux = router.compute_aux_loss(*info)
        assert aux is not None
        loss = weights.square().sum() + aux
        return loss, {"loss": loss, "aux": aux}

    metrics = train_batch_with_loss(module, [{"x": x}], objective)
    assert torch.isfinite(router.weight.grad).all()
    assert router.weight.grad.abs().sum() > 0
    if aux_weight == 0:
        assert metrics[0]["aux"].item() == 0
    else:
        assert metrics[0]["aux"].item() > 0


class AuxiliaryMetricModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.routed_experts_router = MoERouterConfigV2(
            d_model=4, num_experts=4, top_k=2, lb_loss_weight=0.1, z_loss_weight=0.01
        ).build(init_device="cpu")
        nn.init.normal_(self.routed_experts_router.weight, std=0.1)

    def forward(self, x):
        router = self.routed_experts_router
        weights, _, _, info = router(x, scores_only=False, loss_div_factor=3.0)
        return weights.square().sum() + router.compute_aux_loss(*info)

    def reset_auxiliary_metrics(self):
        self.routed_experts_router.reset_metrics()


@pytest.mark.parametrize("entrypoint", ["standard", "ddp"])
@pytest.mark.parametrize("reset", [None, False, True])
def test_custom_objective_metric_ownership(entrypoint, reset, monkeypatch):
    import copy

    from olmo_core.train.train_module.transformer.ddp_train_module import (
        OLMoDDPTrainModule,
    )
    from olmo_core.train.train_module.transformer.train_module import (
        TransformerTrainModule,
    )

    model = AuxiliaryMetricModel()
    module = ObjectiveModule(model)
    models = [model]
    method = TransformerTrainModule.train_batch_with_loss
    if entrypoint == "ddp":
        # Exercise cleanup of every model part, not only module.model.
        models.append(copy.deepcopy(model))
        monkeypatch.setattr(module, "model_parts", models, raising=False)
        method = OLMoDDPTrainModule.train_batch_with_loss
    references = copy.deepcopy(models)
    for part in models:
        part.routed_experts_router.batch_size_per_expert.fill_(9)
    x = torch.randn(1, 3, 4)

    def objective(module, batch):
        loss = sum(part(batch["x"]) for part in models)
        router = model.routed_experts_router
        return loss, {"counts": router.batch_size_per_expert, "lb": router.load_balancing_loss}

    options = {} if reset is None else {"reset_auxiliary_metrics": reset}
    previous_metrics = None
    for iteration in range(2):
        for part in models + references:
            part.zero_grad()
        metrics = method(cast(Any, module), [{"x": x}, {"x": x}], objective, **options)
        for _ in range(2):
            sum(part(x) for part in references).backward()
        for part, reference in zip(models, references):
            for parameter, expected in zip(part.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter.grad, expected.grad, rtol=0, atol=0)
            router = part.routed_experts_router
            assert router.load_balancing_loss is not None
            assert router.z_loss is not None
            if reset:
                assert router.batch_size_per_expert.sum() == 0
                assert router.load_balancing_loss == 0
                assert router.z_loss == 0
            else:
                assert router.batch_size_per_expert.sum() == 36 + 12 * (iteration + 1)
                assert router.load_balancing_loss > 0
                assert router.z_loss > 0
        if reset:
            # Returned counter views must survive cleanup and subsequent batches.
            for values, assignments in zip(metrics, [6, 12]):
                assert values["counts"].sum() == assignments
                assert values["lb"] > 0
                assert not values["lb"].requires_grad
            if previous_metrics is not None:
                for actual, previous in zip(metrics, previous_metrics):
                    for name in actual:
                        torch.testing.assert_close(actual[name], previous[name], rtol=0, atol=0)
            previous_metrics = metrics


@pytest.mark.parametrize("failure_phase", ["objective", "backward", "finalize"])
def test_custom_objective_metric_cleanup_on_failure(failure_phase):
    model = AuxiliaryMetricModel()
    module = ObjectiveModule(model)

    def fail(*args):
        raise RuntimeError("injected custom-objective failure")

    def objective(module, batch):
        loss = model(batch["x"])
        assert model.routed_experts_router.batch_size_per_expert.sum() == 6
        if failure_phase == "objective":
            fail()
        if failure_phase == "backward":
            loss.register_hook(fail)
        return loss, {}

    if failure_phase == "finalize":
        model.finalize_grad_reduce = fail
    with pytest.raises(RuntimeError, match="injected custom-objective failure"):
        train_batch_with_loss(
            module, [{"x": torch.randn(1, 3, 4)}], objective, reset_auxiliary_metrics=True
        )
    router = model.routed_experts_router
    assert router.batch_size_per_expert.sum() == 0
    assert router.load_balancing_loss == 0
    assert router.z_loss == 0
