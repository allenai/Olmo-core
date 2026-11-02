"""Selective backprop decisions (olmo_core.train.selective_backprop)."""
import pytest
import torch

from olmo_core.train.selective_backprop import SelectiveBackprop, SelectiveBackpropConfig


def _steps(sbp, scores_per_step):
    runs = []
    for scores in scores_per_step:
        sbp.begin_batch()
        runs.append([sbp.decide(s, i, len(scores)) for i, s in enumerate(scores)])
    return runs


def test_top_backprops_the_high_loss_units_after_warmup():
    sbp = SelectiveBackprop(SelectiveBackpropConfig(mode="top", q=0.25, window=64, warmup_steps=1))
    g = torch.Generator().manual_seed(0)
    steps = [torch.rand(16, generator=g).tolist() for _ in range(40)]
    runs = _steps(sbp, steps)
    assert all(r for r, _ in runs[0])  # warmup: everything
    later = [(s, r) for sc, rr in zip(steps[5:], runs[5:]) for s, (r, _) in zip(sc, rr)]
    frac = sum(r for _, r in later) / len(later)
    assert frac == pytest.approx(0.25, abs=0.07)
    ran = [s for s, r in later if r]
    skipped = [s for s, r in later if not r]
    assert min(ran) > 0.5 and max(skipped) < 0.9 and sum(ran) / len(ran) > sum(skipped) / len(skipped) + 0.4
    assert all(w == 1.0 for rr in runs for r, w in rr if r)


def test_a_step_always_runs_at_least_one_backward():
    sbp = SelectiveBackprop(SelectiveBackpropConfig(mode="top", q=0.1, window=8, warmup_steps=0))
    _steps(sbp, [[10.0] * 8])
    runs = _steps(sbp, [[0.0] * 4])[0]
    assert [r for r, _ in runs] == [False, False, False, True]
    assert sbp.metrics()["sbp/forced"] == 1.0


def test_sample_is_unbiased_and_hits_q():
    cfg = SelectiveBackpropConfig(mode="sample", q=0.3, window=1000, min_p=0.05, warmup_steps=0)
    sbp = SelectiveBackprop(cfg)
    g = torch.Generator().manual_seed(1)
    scores = (torch.rand(4000, generator=g) ** 3 * 4).tolist()
    _steps(sbp, [scores[:1000]])  # fill the window
    tot_w, tot, ran = 0.0, 0.0, 0
    for k in range(1, 4):
        sc = scores[1000 * k: 1000 * (k + 1)]
        for s, (r, w) in zip(sc, _steps(sbp, [sc])[0]):
            tot_w += s * w
            tot += s
            ran += int(r)
    assert tot_w / tot == pytest.approx(1.0, abs=0.08)  # E[1{run} w s] = s
    assert ran / 3000 == pytest.approx(0.3, abs=0.05)


def test_decisions_are_identical_for_identical_inputs():
    a = SelectiveBackprop(SelectiveBackpropConfig(mode="sample", seed=3, warmup_steps=0))
    b = SelectiveBackprop(SelectiveBackpropConfig(mode="sample", seed=3, warmup_steps=0))
    sc = [[float(i % 5) for i in range(8)] for _ in range(6)]
    assert _steps(a, sc) == _steps(b, sc)


def test_validate():
    for bad in (dict(mode="x"), dict(q=0.0), dict(window=1), dict(min_p=0.0)):
        with pytest.raises(ValueError):
            SelectiveBackpropConfig(**bad).validate()
