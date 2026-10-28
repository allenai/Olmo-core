"""Train-time whole-block skipping (olmo_core.nn.layer_drop): CPU tests on a tiny model."""

from types import SimpleNamespace

import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.layer_drop import LayerDropPlan, flops_per_token_ran, parse_layer_list
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train.callbacks import LayerDropCallback
from olmo_core.utils import seed_all


def _model(n_layers=4):
    cfg = TransformerConfig.olmo2_190M(
        vocab_size=1000, n_layers=n_layers, fused_ops=False, dtype=DType.float32
    )
    return cfg.build(init_device="cpu")


def _count_calls(model):
    calls = {i: 0 for i in range(len(model.blocks))}
    for k, blk in model.blocks.items():

        def hook(mod, args, i=int(k)):
            calls[i] += 1

        blk.register_forward_pre_hook(hook)
    return calls


def test_parse_layer_list():
    assert parse_layer_list("10,14,18") == {10, 14, 18}
    assert parse_layer_list("3-5, 7") == {3, 4, 5, 7}
    assert parse_layer_list("-1", n_layers=32) == {31}
    assert parse_layer_list("") == frozenset()
    with pytest.raises(ValueError):
        parse_layer_list("32", n_layers=32)


def test_plan_is_a_pure_function_of_seed_and_step():
    a = LayerDropPlan(n_layers=32, p=0.25, seed=3)
    b = LayerDropPlan(n_layers=32, p=0.25, seed=3)
    c = LayerDropPlan(n_layers=32, p=0.25, seed=4)
    draws = [a.skipped_for_step(s) for s in range(1, 200)]
    assert draws == [b.skipped_for_step(s) for s in range(1, 200)]  # "other rank"
    assert draws != [c.skipped_for_step(s) for s in range(1, 200)]
    assert len(set(draws)) > 100  # varies step to step
    mean = sum(len(d) for d in draws) / len(draws) / 32
    assert 0.2 < mean < 0.3
    # constant within a step (every micro-batch reads the cached set)
    a.step = 17
    assert a.skipped() == a.skipped() == a.skipped_for_step(17)


def test_plan_exclusions_fixed_and_linear_schedule():
    p = LayerDropPlan(n_layers=8, fixed=frozenset({3}), p=0.5, exclude_first=2, exclude_last=1)
    assert sorted(p.probs) == [2, 4, 5, 6]
    for s in range(100):
        d = p.skipped_for_step(s)
        assert 3 in d and not d & {0, 1, 7}
    assert p.skipped_for_step(None) == {3}  # no step yet (dry-run batch): fixed set only
    lin = LayerDropPlan(n_layers=4, p=0.4, schedule="linear")
    assert lin.probs == pytest.approx({0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4})
    with pytest.raises(ValueError):
        LayerDropPlan(n_layers=4, fixed=frozenset({0, 1, 2, 3}))
    with pytest.raises(ValueError):
        LayerDropPlan(n_layers=4, p=1.0)


def test_skipped_block_forward_is_not_called_in_training_only():
    seed_all(0)
    m = _model()
    m.enable_layer_drop(fixed={1, 2})
    calls = _count_calls(m)
    ids = torch.randint(0, 1000, (2, 16))
    m.train()
    m(ids)
    assert calls == {0: 1, 1: 0, 2: 0, 3: 1}
    m.eval()
    with torch.no_grad():
        m(ids)
    assert calls == {0: 2, 1: 1, 2: 1, 3: 2}


def test_skip_equals_model_without_those_blocks_and_grads_are_none():
    seed_all(1)
    m = _model()
    ref = _model()
    ref.load_state_dict(m.state_dict())
    m.enable_layer_drop(fixed={2})
    ids = torch.randint(0, 1000, (2, 16))
    m.train()
    out = m(ids)
    # reference: run the blocks by hand, leaving out block 2
    h = ref.embeddings(ids)
    for k in ("0", "1", "3"):
        h = ref.blocks[k](h)
    torch.testing.assert_close(out, ref.lm_head(h))
    out.sum().backward()
    assert all(p.grad is None for p in m.blocks["2"].parameters())
    assert all(p.grad is not None for p in m.blocks["3"].parameters())


def test_random_drop_follows_the_callback_step():
    seed_all(2)
    m = _model(n_layers=8)
    m.enable_layer_drop(p=0.5, seed=0)
    calls = _count_calls(m)
    cb = LayerDropCallback(log_every=0)
    recorded = {}
    cb._trainer = SimpleNamespace(  # type: ignore[assignment]
        global_step=0,
        train_module=SimpleNamespace(model=m),
        record_metric=lambda k, v: recorded.__setitem__(k, v),
    )
    ids = torch.randint(0, 1000, (1, 8))
    m.train()
    for step in (1, 2, 3):
        cb._trainer.global_step = step
        cb.pre_step({})
        before = dict(calls)
        for _ in range(3):  # three micro-batches: same skip set
            m(ids)
        ran = {i for i in calls if calls[i] - before[i] == 3}
        assert all(calls[i] - before[i] in (0, 3) for i in calls)
        assert ran == set(range(8)) - m._layer_drop.skipped_for_step(step)
        cb.post_step()
        assert recorded["layer_drop/n_skipped"] == 8 - len(ran)


def test_rescale_scales_the_residual_update():
    seed_all(3)
    m = _model(n_layers=2)
    m.enable_layer_drop(p=0.5, rescale=True, exclude_first=1)  # only block 1 is eligible
    plan = m._layer_drop
    step = next(s for s in range(100) if 1 not in plan.skipped_for_step(s))
    plan.step = step
    ids = torch.randint(0, 1000, (1, 8))
    m.train()
    out = m(ids)
    h0 = m.blocks["0"](m.embeddings(ids))
    h1 = h0 + (m.blocks["1"](h0) - h0) * 2.0
    torch.testing.assert_close(out, m.lm_head(h1))


def test_flops_per_token_ran():
    m = _model()
    full = m.num_flops_per_token(128)
    blk = m.blocks["1"].num_flops_per_token(128)
    assert flops_per_token_ran(m, 128, ()) == full
    assert flops_per_token_ran(m, 128, {1, 2}) == full - 2 * blk


def test_early_lora_call_check_ignores_skipped_blocks():
    """The early-LoRA 'every adapted linear ran' check must not count blocks layer drop skipped on purpose."""
    from olmo_core.nn.early_lora import install_early_lora

    seed_all(4)
    m = _model()
    install_early_lora(m, 4, 4, scale=8.0)
    m.enable_layer_drop(fixed={1, 2})
    m.train()
    out = m(torch.randint(0, 1000, (1, 8)))  # raised "adapted linears never ran their forward" before the fix
    assert m._early_lora["checked"] and torch.isfinite(out).all()
