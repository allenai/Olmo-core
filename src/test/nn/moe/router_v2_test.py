import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

import olmo_core.nn.moe.v2.router as router_v2
from olmo_core.config import DType
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.moe.loss import MoELoadBalancingLossGranularity
from olmo_core.nn.moe.router import MoERouterConfig, MoERouterGatingFunction
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.testing import requires_multi_gpu, run_distributed_test


def _build(*, top_k=2, num_experts=8, d_model=16, **kwargs):
    return MoERouterConfigV2(
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k,
        dtype=DType.float32,
        **kwargs,
    ).build(init_device="cpu")


def test_router_config_num_params():
    cfg = MoERouterConfigV2(d_model=16, num_experts=8, top_k=2)
    # The router weight is (num_experts * d_model).
    assert cfg.num_params() == 16 * 8


def test_router_forward_shapes_and_invariants():
    torch.manual_seed(0)
    B, S, D, E, K = 2, 4, 16, 8, 2
    router = _build(top_k=K, num_experts=E, d_model=D)

    x = torch.randn(B, S, D)
    weights, indices, batch_size_per_expert, aux = router(x, False)

    assert weights.shape == (B, S, K)
    assert indices.shape == (B, S, K)
    assert batch_size_per_expert.shape == (E,)
    # Every selected expert index is valid.
    assert int(indices.min()) >= 0 and int(indices.max()) < E
    # Each of B*S tokens is routed to exactly top_k experts.
    assert int(batch_size_per_expert.sum()) == B * S * K
    # Forward returns the auxiliary-loss inputs (not the reduced losses).
    assert aux is not None


def test_router_scores_only_short_circuits():
    B, S, D, E = 2, 4, 16, 8
    router = _build(num_experts=E, d_model=D)

    scores, indices, batch_size_per_expert, aux = router(torch.randn(B, S, D), True)

    assert scores.shape == (B, S, E)
    assert indices is None and batch_size_per_expert is None and aux is None


def test_router_top1_and_sigmoid_gating():
    router = _build(top_k=1, gating_function=MoERouterGatingFunction.sigmoid)

    weights, indices, _, _ = router(torch.randn(2, 4, 16), False)

    assert weights.shape == (2, 4, 1)
    assert indices.shape == (2, 4, 1)


def test_router_uniform_expert_assignment_balances_experts():
    B, S, D, E, K = 2, 8, 16, 4, 2
    router = _build(top_k=K, num_experts=E, d_model=D, uniform_expert_assignment=True)

    _, indices, batch_size_per_expert, _ = router(torch.randn(B, S, D), False)

    # Uniform assignment spreads the B*S*K routing slots evenly across experts.
    assert int(indices.min()) >= 0 and int(indices.max()) < E
    expected = B * S * K // E
    torch.testing.assert_close(
        batch_size_per_expert,
        torch.full_like(batch_size_per_expert, expected),
    )


def test_router_restore_weight_scale_multiplies_by_top_k():
    torch.manual_seed(0)
    router = _build(top_k=4)
    x = torch.randn(2, 4, 16)

    base_weights, _, _, _ = router(x, False)
    router.restore_weight_scale = True
    scaled_weights, _, _, _ = router(x, False)

    torch.testing.assert_close(scaled_weights, base_weights * router.top_k)


def test_router_original_top_k_rescales_weights():
    torch.manual_seed(0)
    router = _build(top_k=2, original_top_k=8)
    x = torch.randn(2, 4, 16)

    scaled_weights, _, _, _ = router(x, False)
    router.original_top_k = None
    base_weights, _, _, _ = router(x, False)

    torch.testing.assert_close(scaled_weights, base_weights * (8 / 2) ** 0.5)


def test_router_normalize_expert_weights_unit_norm():
    router = _build(num_experts=8, normalize_expert_weights=2.0)

    # The scores-only path normalizes the score vector to unit Lp norm per token.
    scores, _, _, _ = router(torch.randn(2, 4, 16), True)

    norms = scores.norm(p=2.0, dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms))


def test_router_bias_gamma_creates_buffer_and_biases_routing():
    router = _build(top_k=1, num_experts=4, bias_gamma=0.01)
    assert router.score_bias is not None
    assert tuple(router.score_bias.shape) == (4,)

    # Strongly bias expert 3; with top_k=1 every token must select it (selection uses
    # scores + score_bias, while the returned weights still come from the raw scores).
    with torch.no_grad():
        router.score_bias.copy_(torch.tensor([-10.0, -10.0, -10.0, 10.0]))
    _, indices, _, _ = router(torch.randn(2, 4, 16), False)
    assert torch.equal(indices, torch.full_like(indices, 3))

    # No bias buffer when bias_gamma is unset.
    assert _build(bias_gamma=None).score_bias is None


def test_global_load_balancing_averages_counts_without_mutating_local_counts(
    monkeypatch: pytest.MonkeyPatch,
):
    router = _build(
        top_k=1,
        num_experts=2,
        lb_loss_weight=1.0,
        global_load_balancing=True,
    )
    router.lb_process_group = object()  # type: ignore[assignment]
    local_counts = torch.tensor([8, 0])
    captured = {}

    def fake_all_reduce(counts, *, op, group):
        assert op == dist.ReduceOp.SUM
        assert group is router.lb_process_group
        counts.add_(torch.tensor([0.0, 8.0]))

    def fake_load_balancing_loss(**kwargs):
        captured["counts"] = kwargs["batch_size_per_expert"].clone()
        return kwargs["expert_scores"].sum() * 0

    monkeypatch.setattr(router_v2.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(router_v2.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(router_v2, "load_balancing_loss", fake_load_balancing_loss)

    scores = torch.full((1, 8, 2), 0.5, requires_grad=True)
    router.compute_aux_loss(
        scores,
        scores.log(),
        local_counts,
        local_counts.unsqueeze(0),
        8.0,
    )

    torch.testing.assert_close(captured["counts"], torch.tensor([4.0, 4.0]))
    torch.testing.assert_close(local_counts, torch.tensor([8, 0]))
    torch.testing.assert_close(router.batch_size_per_expert, torch.tensor([8.0, 0.0]))
    assert router.global_batch_size_per_expert is not None
    torch.testing.assert_close(router.global_batch_size_per_expert, torch.tensor([4.0, 4.0]))

    metrics = router.compute_metrics(reset=False)
    torch.testing.assert_close(metrics["load imbalance"][0], torch.tensor(2.0))
    torch.testing.assert_close(metrics["global load imbalance"][0], torch.tensor(1.0))

    router.reset_metrics()
    torch.testing.assert_close(router.batch_size_per_expert, torch.zeros(2))
    torch.testing.assert_close(router.global_batch_size_per_expert, torch.zeros(2))


def test_global_load_balancing_requires_process_group():
    router = _build(lb_loss_weight=1.0, global_load_balancing=True)
    _, _, _, aux = router(torch.randn(1, 4, 16), False)
    assert aux is not None
    with pytest.raises(RuntimeError, match="requires a load-balancing process group"):
        router.compute_aux_loss(*aux)


def test_global_load_balancing_rejects_instance_granularity():
    with pytest.raises(OLMoConfigurationError, match="instance-granularity"):
        _build(
            lb_loss_weight=1.0,
            global_load_balancing=True,
            lb_loss_granularity=MoELoadBalancingLossGranularity.instance,
        )


def _run_global_load_balancing_matches_concatenated_reference(compiled=False):
    world_size = get_world_size()
    rank = get_rank()
    group = dist.group.WORLD
    torch.manual_seed(7)

    local_tokens = 8
    full_x = torch.randn(world_size, local_tokens, 4)
    router = _build(
        d_model=4,
        num_experts=4,
        top_k=1,
        lb_loss_weight=1.0,
        global_load_balancing=True,
    )
    router.set_load_balancing_process_group(group)

    def forward(x):
        _, _, _, aux = router(x, False, loss_div_factor=local_tokens)
        assert aux is not None
        return router.compute_aux_loss(*aux, accumulate_metrics=False)

    if compiled:
        # A full graph must include the count collective; graph breaks fail the test.
        forward = torch.compile(forward, fullgraph=True)
    loss = forward(full_x[rank : rank + 1])
    assert loss is not None
    loss.backward()
    assert router.weight.grad is not None
    distributed_grad = router.weight.grad.clone()
    dist.all_reduce(distributed_grad, group=group)
    distributed_grad.div_(world_size)

    reference = _build(
        d_model=4,
        num_experts=4,
        top_k=1,
        lb_loss_weight=1.0,
    )
    with torch.no_grad():
        reference.weight.copy_(router.weight)
    _, _, _, reference_aux = reference(
        full_x,
        False,
        loss_div_factor=world_size * local_tokens,
    )
    assert reference_aux is not None
    reference_loss = reference.compute_aux_loss(*reference_aux, accumulate_metrics=False)
    assert reference_loss is not None
    reference_loss.backward()
    assert reference.weight.grad is not None

    torch.testing.assert_close(distributed_grad, reference.weight.grad)


@pytest.mark.parametrize("compiled", [False, True])
def test_global_load_balancing_matches_concatenated_reference_cpu(compiled):
    run_distributed_test(
        _run_global_load_balancing_matches_concatenated_reference,
        world_size=2,
        backend="gloo",
        start_method="spawn",
        func_args=(compiled,),
    )


@pytest.mark.parametrize(
    "gating", [MoERouterGatingFunction.softmax, MoERouterGatingFunction.sigmoid]
)
@pytest.mark.parametrize("top_k", [1, 2])
def test_v2_router_matches_v1_with_defaults(top_k: int, gating: MoERouterGatingFunction):
    torch.manual_seed(0)
    D, E = 16, 8

    v1 = MoERouterConfig(top_k=top_k, gating_function=gating, dtype=DType.float32).build(
        d_model=D, num_experts=E, init_device="cpu"
    )
    v2 = MoERouterConfigV2(
        d_model=D, num_experts=E, top_k=top_k, gating_function=gating, dtype=DType.float32
    ).build(init_device="cpu")

    # Same (flat num_experts*d_model) weights -> identical routing with default settings.
    with torch.no_grad():
        v2.weight.copy_(v1.weight)

    x = torch.randn(2, 4, D)
    w1, i1, bspe1, _ = v1(x)
    w2, i2, bspe2, _ = v2(x, False)

    torch.testing.assert_close(w2, w1)
    assert torch.equal(i2, i1)
    torch.testing.assert_close(bspe2, bspe1)


# NOTE: ``use_recompute_fp32_cast`` is exercised end-to-end on GPU. It routes the fp32 cast
# through OutputDiscardCheckpoint, whose storage sharing relies on a C++ extension (covered
# by the OutputDiscardCheckpoint test suite); the Python fallback cannot recompute through
# autograd on a plain CPU host, so it is not unit-tested at the router level here.


def _run_router_tp(device: torch.device):
    tp_mesh = dist.init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("tp",))

    router = MoERouterConfigV2(d_model=16, num_experts=4, top_k=2, dtype=DType.float32).build(
        init_device=device.type
    )
    router.apply_tp(tp_mesh)

    # apply_tp replicates the router weight across the TP mesh.
    assert isinstance(router.weight, DTensor)
    assert router.weight.placements == (Replicate(),)

    # PrepareModuleInput shards the input on the sequence dim; forward sees the local shard.
    B, S, D, K = 2, 4 * get_world_size(), 16, 2
    x = torch.randn(B, S, D, device=device)
    local_x = distribute_tensor(x, tp_mesh, [Shard(1)]).to_local()

    # scores_only must be passed by keyword: PrepareModuleInput maps positional inputs to
    # input_layouts, so the lone sharded positional arg is the activation tensor.
    weights, indices, _, _ = router(local_x, scores_only=False)

    assert weights.shape == (B, S // get_world_size(), K)
    assert indices.shape == (B, S // get_world_size(), K)


@requires_multi_gpu
def test_router_tp_replicates_weight_and_runs():
    run_distributed_test(
        _run_router_tp,
        world_size=2,
        backend="nccl",
        func_args=(torch.device("cuda"),),
    )


def _run_router_tp_accumulates_aux_losses(device: torch.device):
    tp_mesh = dist.init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("tp",))

    router = MoERouterConfigV2(
        d_model=16,
        num_experts=4,
        top_k=2,
        dtype=DType.float32,
        lb_loss_weight=0.01,
        z_loss_weight=0.001,
    ).build(init_device=device.type)
    router.apply_tp(tp_mesh)

    B, S, D = 2, 4 * get_world_size(), 16
    x = torch.randn(B, S, D, device=device)
    local_x = distribute_tensor(x, tp_mesh, [Shard(1)]).to_local()

    # Under TP the aux losses are replicated-scalar DTensors; accumulating them into the
    # plain-tensor metric accumulators must not raise (regression: plain += DTensor in-place).
    loss_div_factor = float(B * S)
    router(local_x, scores_only=False, loss_div_factor=loss_div_factor)
    router(local_x, scores_only=False, loss_div_factor=loss_div_factor)

    metrics = router.compute_metrics(reset=True)
    lb = metrics["load balancing loss unscaled"][0]
    z = metrics["router Z loss unscaled"][0]
    assert not isinstance(lb, DTensor)
    assert not isinstance(z, DTensor)
    assert torch.isfinite(lb).all()
    assert torch.isfinite(z).all()
    # compute_metrics(reset=True) should have zeroed the accumulators.
    assert router.load_balancing_loss is not None
    assert router.z_loss is not None
    assert float(router.load_balancing_loss) == 0.0
    assert float(router.z_loss) == 0.0


def test_router_tp_accumulates_aux_losses_cpu():
    run_distributed_test(
        _run_router_tp_accumulates_aux_losses,
        world_size=2,
        backend="gloo",
        func_args=(torch.device("cpu"),),
        start_method="spawn",
    )


def test_step_load_balancing_rejects_instance_granularity():
    with pytest.raises(OLMoConfigurationError, match="step_load_balancing"):
        _build(
            lb_loss_weight=1.0,
            step_load_balancing=True,
            lb_loss_granularity=MoELoadBalancingLossGranularity.instance,
        )


def _capture_lb_counts(monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
    captured: list[torch.Tensor] = []

    def fake_load_balancing_loss(**kwargs):
        captured.append(kwargs["batch_size_per_expert"].clone())
        return kwargs["expert_scores"].sum() * 0

    monkeypatch.setattr(router_v2, "load_balancing_loss", fake_load_balancing_loss)
    return captured


def _aux(router, counts: torch.Tensor):
    scores = torch.full((1, int(counts.sum()), 2), 0.5, requires_grad=True)
    router.compute_aux_loss(scores, scores.log(), counts, counts.unsqueeze(0), 8.0)


def test_step_load_balancing_uses_previous_step_counts(monkeypatch: pytest.MonkeyPatch):
    router = _build(top_k=1, num_experts=2, lb_loss_weight=1.0, step_load_balancing=True)
    captured = _capture_lb_counts(monkeypatch)

    # Step 1: no previous step yet, so each micro-batch uses its own counts.
    _aux(router, torch.tensor([8, 0]))
    _aux(router, torch.tensor([0, 8]))
    torch.testing.assert_close(captured[0], torch.tensor([8.0, 0.0]))
    torch.testing.assert_close(captured[1], torch.tensor([0.0, 8.0]))
    assert router.step_batch_size_per_expert is not None
    torch.testing.assert_close(router.step_batch_size_per_expert, torch.tensor([8.0, 8.0]))
    metrics = router.compute_metrics(reset=False)
    torch.testing.assert_close(metrics["step load imbalance"][0], torch.tensor(0.0))

    # The metric reset at the end of a step must not touch the step accumulator.
    router.reset_metrics()
    torch.testing.assert_close(router.step_batch_size_per_expert, torch.tensor([8.0, 8.0]))

    router.post_batch()
    assert router.prev_step_batch_size_per_expert is not None
    torch.testing.assert_close(router.prev_step_batch_size_per_expert, torch.tensor([8.0, 8.0]))
    torch.testing.assert_close(router.step_batch_size_per_expert, torch.zeros(2))

    # Step 2: a lopsided micro-batch is balanced against the whole previous step,
    # rescaled to this micro-batch's slot count (4).
    _aux(router, torch.tensor([4, 0]))
    torch.testing.assert_close(captured[2], torch.tensor([2.0, 2.0]))
    # Local per-step metrics still see the real micro-batch counts.
    torch.testing.assert_close(router.batch_size_per_expert, torch.tensor([4.0, 0.0]))
    torch.testing.assert_close(router.step_batch_size_per_expert, torch.tensor([4.0, 0.0]))
    metrics = router.compute_metrics(reset=False)
    torch.testing.assert_close(metrics["step load imbalance"][0], torch.tensor(1.0))

    # A dry run clears the accumulator without promoting it to the previous step.
    router.post_batch(dry_run=True)
    torch.testing.assert_close(router.prev_step_batch_size_per_expert, torch.tensor([8.0, 8.0]))
    torch.testing.assert_close(router.step_batch_size_per_expert, torch.zeros(2))

    # Eval mode neither accumulates nor rotates.
    router.eval()
    _aux(router, torch.tensor([4, 0]))
    router.post_batch()
    assert len(captured) == 3
    torch.testing.assert_close(router.prev_step_batch_size_per_expert, torch.tensor([8.0, 8.0]))


def test_step_load_balancing_composes_with_global_load_balancing(
    monkeypatch: pytest.MonkeyPatch,
):
    router = _build(
        top_k=1,
        num_experts=2,
        lb_loss_weight=1.0,
        global_load_balancing=True,
        step_load_balancing=True,
    )
    router.lb_process_group = object()  # type: ignore[assignment]
    captured = _capture_lb_counts(monkeypatch)

    def fake_all_reduce(counts, *, op, group):
        counts.add_(torch.tensor([0.0, 8.0]))

    monkeypatch.setattr(router_v2.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(router_v2.dist, "get_world_size", lambda group: 2)

    # Step 1 accumulates the DP-averaged counts, not the local ones.
    _aux(router, torch.tensor([8, 0]))
    torch.testing.assert_close(captured[0], torch.tensor([4.0, 4.0]))
    assert router.step_batch_size_per_expert is not None
    torch.testing.assert_close(router.step_batch_size_per_expert, torch.tensor([4.0, 4.0]))
    router.post_batch()

    # Step 2 balances against last step's DP-averaged, whole-step counts.
    _aux(router, torch.tensor([8, 0]))
    torch.testing.assert_close(captured[1], torch.tensor([4.0, 4.0]))
