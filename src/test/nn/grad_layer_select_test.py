"""Gradient-ranked layer selection (olmo_core.nn.grad_layer_select): CPU tests on a tiny model."""

from types import SimpleNamespace

import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.grad_layer_select import GradLayerSelector
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train.callbacks import GradLayerSelectCallback
from olmo_core.train.callbacks.flop_meter import FlopMeterCallback
from olmo_core.utils import seed_all


def _model(n_layers=4, vocab=1000):
    cfg = TransformerConfig.olmo2_190M(vocab_size=vocab, n_layers=n_layers, fused_ops=False, dtype=DType.float32)
    return cfg.build(init_device="cpu")


def _trainer(model, step=1, metrics=None):
    metrics = {} if metrics is None else metrics
    return SimpleNamespace(
        global_step=step,
        global_batch_size=2 * 256,
        save_folder="/nonexistent",
        train_module=SimpleNamespace(model=model),
        record_metric=lambda k, v: metrics.__setitem__(k, v),
    )


# -- selector ---------------------------------------------------------------------------------------
def test_online_ranking_measures_every_block_first_then_keeps_the_top():
    s = GradLayerSelector(n_layers=8, keep_frac=0.25, explore_frac=0.0)
    assert s.k == 2
    seen = set()
    for step in range(1, 5):  # optimistic init: 4 steps x 2 slots measure all 8 blocks once
        sel = s.begin_step(step)
        seen |= sel
        s.update_energies({i: float(i) for i in sel})  # energy = block index
    assert seen == set(range(8))
    assert s.begin_step(5) == {6, 7}
    s.update_energies({6: 0.0, 7: 0.0})  # EMA moves toward the new (zero) measurements
    assert s.ema[6] == pytest.approx(0.9 * 6) and s.ema[7] == pytest.approx(0.9 * 7)


def test_exploration_is_seeded_and_identical_across_ranks():
    a = GradLayerSelector(n_layers=32, keep_frac=0.5, explore_frac=0.1, seed=3)
    b = GradLayerSelector(n_layers=32, keep_frac=0.5, explore_frac=0.1, seed=3)
    for s_ in (a, b):
        s_.ema = [float(i) for i in range(32)]
    assert a.n_explore() == 2  # 10% of 16 slots
    picks = [a.select_for_step(t) for t in range(1, 30)]
    assert picks == [b.select_for_step(t) for t in range(1, 30)]
    top = set(range(18, 32))  # the 14 exploited blocks
    assert all(top <= p and len(p) == 16 for p in picks)
    assert len({frozenset(p - top) for p in picks}) > 5  # the explored pair varies


def test_static_ranking_always_and_cut():
    energy = [5.0, 1.0, 4.0, 0.5, 3.0, 0.1, 0.2, 2.0]
    s = GradLayerSelector(n_layers=8, keep_frac=0.5, ranking="static", static_energy=energy, always=[7])
    s.begin_step(1)
    assert s.selected() == {0, 2, 4, 7}
    assert s.n_explore() == 0 and s.skipped() == frozenset()  # update mode skips nothing
    assert s.cut() == 0
    e2 = [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0]
    s2 = GradLayerSelector(n_layers=8, keep_frac=0.25, ranking="static", static_energy=e2)
    s2.begin_step(1)
    assert s2.selected() == {3, 7} and s2.cut() == 3
    f = s2.block_factors()
    assert f[0] == f[2] == pytest.approx(1 / 3) and f[4] == f[6] == pytest.approx(2 / 3) and 3 not in f
    s3 = GradLayerSelector(n_layers=8, keep_frac=0.25, ranking="static", static_energy=e2, max_cut=2)
    s3.begin_step(1)
    assert s3.cut() == 2  # capped below the lowest selected block (e.g. at a router drop layer)
    f = s3.block_factors()
    assert f[1] == pytest.approx(1 / 3) and f[2] == pytest.approx(2 / 3)
    sk = GradLayerSelector(n_layers=8, keep_frac=0.5, mode="skip", ranking="static", static_energy=energy)
    sk.begin_step(1)
    assert sk.skipped() == {1, 3, 5, 6} and sk.cut() == 0
    assert sk.select_for_step(None) == set(range(8))  # dry-run batch: everything


# -- model ------------------------------------------------------------------------------------------
def test_update_mode_trains_only_selected_blocks_and_matches_dense_forward():
    seed_all(0)
    m = _model()
    ref = _model()
    ref.load_state_dict(m.state_dict())
    energy = [0.1, 3.0, 0.2, 2.0]  # top-2 = blocks 1 and 3 -> cut at block 1
    m.enable_grad_layer_select(keep_frac=0.5, ranking="static", static_energy=energy)
    cb = GradLayerSelectCallback(log_every=0)
    cb._trainer = _trainer(m)  # type: ignore[assignment]
    m.train()
    ref.train()
    cb.pre_train()
    cb.pre_step({})
    ids = torch.randint(0, 1000, (2, 16))
    out = m(ids)
    torch.testing.assert_close(out, ref(ids))  # every block runs forward: no train/eval mismatch
    out.sum().backward()
    for i in (0, 2):
        assert all(p.grad is None for p in m.blocks[str(i)].parameters()), i
    for i in (1, 3):
        assert all(p.grad is not None for p in m.blocks[str(i)].parameters()), i
    assert m.embeddings.weight.grad is None  # below the cut
    assert m.lm_head.w_out.weight.grad is not None
    cb.pre_optim_step()
    sel = m._grad_select
    assert sel.ema[1] is not None and sel.ema[3] is not None and sel.ema[0] is None
    want = sum(float(p.grad.pow(2).sum()) for p in m.blocks["1"].parameters())
    assert sel.ema[1] == pytest.approx(want, rel=1e-5)
    # the selected blocks' gradients equal the full model's
    ref(ids).sum().backward()
    for i in (1, 3):
        for (n, p), (_, rp) in zip(m.blocks[str(i)].named_parameters(), ref.blocks[str(i)].named_parameters()):
            torch.testing.assert_close(p.grad, rp.grad, msg=n)
    cb.post_train()
    assert all(p.requires_grad for p in m.parameters())


def test_skip_mode_does_not_call_unselected_blocks_and_eval_runs_all():
    seed_all(1)
    m = _model()
    m.enable_grad_layer_select(keep_frac=0.5, mode="skip", ranking="static", static_energy=[1.0, 0.0, 2.0, 0.5])
    calls = {i: 0 for i in range(4)}
    for k, blk in m.blocks.items():
        blk.register_forward_pre_hook(lambda mod, a, i=int(k): calls.__setitem__(i, calls[i] + 1))
    cb = GradLayerSelectCallback(log_every=0)
    cb._trainer = _trainer(m)  # type: ignore[assignment]
    cb.pre_train()
    cb.pre_step({})
    ids = torch.randint(0, 1000, (1, 8))
    m.train()
    m(ids).sum().backward()
    assert calls == {0: 1, 1: 0, 2: 1, 3: 0}
    m.eval()
    with torch.no_grad():
        m(ids)
    assert calls == {0: 2, 1: 1, 2: 2, 3: 1}


def test_grad_select_excludes_layer_drop():
    m = _model()
    m.enable_layer_drop(fixed={1})
    with pytest.raises(Exception):
        m.enable_grad_layer_select(keep_frac=0.5)


# -- flop meter ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["update", "skip"])
def test_flop_meter_charges_selected_full_unselected_two_thirds_below_cut_one_third(mode):
    m = _model()
    m.enable_grad_layer_select(keep_frac=0.5, mode=mode, ranking="static", static_energy=[0.1, 3.0, 0.2, 2.0])
    m._grad_select.begin_step(1)  # selected {1, 3}; cut 1 in update mode
    cb = FlopMeterCallback(seq_len=256, pad_id=0, per_example=True)
    metrics = {}
    cb._trainer = _trainer(m, metrics=metrics)  # type: ignore[assignment]
    lens = [100, 256]
    m._flop_example_lens = {"dense": list(lens), "actual": list(lens)}
    cb.post_step()
    want = 0.0
    for L in lens:
        f = m.blocks["0"].num_flops_per_token(L)  # identical blocks
        if mode == "update":
            blocks = f * (1 / 3 + 1 + 2 / 3 + 1)  # block 0 below the cut, 2 unselected above it
        else:
            blocks = f * 2  # only 1 and 3 ran
        want += L * (m.num_flops_per_token(L) - 4 * f + blocks)
    assert metrics["flop_meter/actual_pflops"] == pytest.approx(want / 1e15)


# -- online router ------------------------------------------------------------------------------------
def test_update_mode_with_router_keeps_the_target_gradient_and_composes_factors():
    from olmo_core.nn.online_token_router import OnlineRouterConfig, OnlineTokenRouter
    from .online_token_router_test import DE, DS, EOS, PH, _row

    torch.manual_seed(0)
    model = _model(n_layers=4, vocab=9100)
    D = int(model.embeddings.weight.shape[1])
    router = OnlineTokenRouter(
        OnlineRouterConfig(keep_frac=0.25, explore_frac=0.2, samples_per_row=64, drop_layer=1),
        doc_start_id=DS, doc_end_id=DE, eos_id=EOS, d_model=D,
    )
    model.enable_pooled_soft_tokens(
        DS, DE, EOS, placeholder_id=PH, keep_prob=0.0, keep_seed=0, detach_soft_kv=True,
        drop_slots=True, keep_token_rule="online", keep_token_k=0.25, online_router=router,
    )
    # top-1 = block 3; the router's drop layer (1) caps the cut and keeps an input gradient there
    model.enable_grad_layer_select(keep_frac=0.25, ranking="static", static_energy=[0.0, 0.1, 0.2, 9.0],
                                   max_cut=1, cut_input_grad=True)
    router._grad_select = model._grad_select
    cb = GradLayerSelectCallback(log_every=0)
    cb._trainer = _trainer(model)  # type: ignore[assignment]
    cb.pre_train()
    model.train()
    ids, cid, labels = _row(B=2)
    for step in (1, 2):
        cb._trainer.global_step = step
        cb.pre_step({})
        router.begin_step(step, model=model)
        model._flop_extra_actual = 0.0
        out = model(ids, labels=labels)
        out.loss.backward()
        assert router._pending[-1]["target"] is not None  # dL/dh reached the drop layer
        assert all(p.grad is None for p in model.blocks["0"].parameters())  # below the cut
        assert all(p.grad is None for p in model.blocks["2"].parameters())  # unselected above it
        assert all(p.grad is not None for p in model.blocks["3"].parameters())
        model.zero_grad(set_to_none=True)
    assert router._block_factors(4) == pytest.approx([1 / 3, 2 / 3, 2 / 3, 1.0])
