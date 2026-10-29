"""Partial-MLP training (olmo_core.nn.ffn_width): step-seeded neuron subsets, exact gathered forward, unbiased in
expectation with the 1/f rescale, full width at eval, and the FLOP credit."""
import pytest
import torch

from olmo_core.nn.feed_forward import FeedForward
from olmo_core.nn.ffn_width import FFNWidthHolder, install_ffn_width, parse_layer_range


class _M(torch.nn.Module):
    def __init__(self, n=3, d=16, hid=64):
        super().__init__()
        self.blocks = torch.nn.ModuleDict({str(i): torch.nn.Module() for i in range(n)})
        for b in self.blocks.values():
            b.feed_forward = FeedForward(d_model=d, hidden_size=hid, bias=False, init_device="cpu")


def test_parse_layer_range():
    assert parse_layer_range("", 4) == [0, 1, 2, 3]
    assert parse_layer_range("0-2", 8) == [0, 1, 2]
    assert parse_layer_range("1,5-6", 8) == [1, 5, 6]
    with pytest.raises(ValueError):
        parse_layer_range("9", 8)


@pytest.mark.parametrize("mode", ["random", "blocked"])
def test_indices_deterministic_per_step_and_layer(mode):
    a = FFNWidthHolder(0.5, [0, 1], mode=mode, block=8, seed=3)
    b = FFNWidthHolder(0.5, [0, 1], mode=mode, block=8, seed=3)  # another rank
    for step in (1, 2):
        a.set_step(step)
        b.set_step(step)
        assert torch.equal(a.indices(0, 64, "cpu"), b.indices(0, 64, "cpu"))
        assert torch.equal(a.indices(0, 64, "cpu"), a.indices(0, 64, "cpu"))  # recompute: same subset
    a.set_step(1)
    i1 = a.indices(0, 64, "cpu")
    a.set_step(2)
    assert not torch.equal(i1, a.indices(0, 64, "cpu"))
    assert not torch.equal(a.indices(0, 64, "cpu"), a.indices(1, 64, "cpu"))
    assert a.indices(0, 64, "cpu").numel() == 32
    if mode == "blocked":
        idx = a.indices(0, 64, "cpu")
        blocks = set((idx // 8).tolist())
        assert len(blocks) == 4 and all(((idx // 8) == bb).sum() == 8 for bb in blocks)


def test_partial_forward_exact_grads_and_eval_full_width():
    torch.manual_seed(0)
    m = _M()
    ff = m.blocks["0"].feed_forward
    x = torch.randn(2, 5, 16)
    full = ff(x).detach()
    h = install_ffn_width(m, 0.25, "0-1", seed=1)
    h.set_step(7)
    ff.train()
    y = ff(x)
    idx = h.indices(0, 64, "cpu")
    ref = torch.nn.functional.linear(
        torch.nn.functional.silu(x @ ff.w1.weight[idx].T) * (x @ ff.w3.weight[idx].T), ff.w2.weight[:, idx]) * 4.0
    assert torch.allclose(y, ref, atol=1e-6)
    y.sum().backward()
    rows = ff.w1.weight.grad.abs().sum(1) > 0
    assert torch.equal(rows.nonzero().flatten(), idx)
    assert (ff.w2.weight.grad[:, ~rows] == 0).all()
    ff.eval()
    assert torch.allclose(ff(x), full, atol=1e-6)
    # an unlisted layer is untouched
    ff2 = m.blocks["2"].feed_forward
    ff2.train()
    assert not hasattr(ff2, "_ffnw")


def test_rescaled_partial_ffn_is_unbiased_in_expectation():
    torch.manual_seed(0)
    m = _M(n=1, hid=32)
    ff = m.blocks["0"].feed_forward
    x = torch.randn(1, 3, 16)
    full = ff(x).detach()
    h = install_ffn_width(m, 0.5, seed=2)
    ff.train()
    acc = torch.zeros_like(full)
    n = 2000
    with torch.no_grad():
        for s in range(n):
            h.set_step(s)
            acc += ff(x)
    err = (acc / n - full).abs().max() / full.abs().max()
    assert err < 0.05, err
    h2 = install_ffn_width(_M(n=1, hid=32), 0.5, seed=2, rescale=False)  # ablation: biased low
    assert h2.rescale is False


def test_router_charges_partial_ffn_credit_cpu():
    """With the mid-stack router, blocks below the drop are credited -(1-f) x FFN on the full row, blocks above on the
    kept row."""
    from .online_token_router_test import PH, _mid_model, _row

    model, router = _mid_model(4, drop_layer=2, dense_every=0)
    blocks = list(model.blocks.values())
    h = install_ffn_width(model, 0.5, "1-3", seed=0)
    ids, cid, labels = _row(B=1)
    flops = []
    orig = router._flop
    router._flop = lambda m_, v: (flops.append(float(v)), orig(m_, v))
    router.begin_step(1, model=model)
    model(ids, labels=labels).loss.backward()
    router.begin_step(2, model=model)
    h.set_step(2)
    flops.clear()
    router.metrics()
    model(ids, labels=labels).loss.backward()
    real = int((ids != PH).sum())
    kept = int(round(router.metrics()["online_router/mid_row_frac"] * real))
    head = int(model.lm_head.num_flops_per_token(0))
    f = lambda b, n: int(b.num_flops_per_token(int(n)))
    cr = h.credits_per_block(4)
    assert cr[0] == 0 and cr[1] < 0
    expect = (real * (f(blocks[0], real) + f(blocks[1], real)) + kept * (f(blocks[2], kept) + f(blocks[3], kept) + head)
              - real * (sum(f(b, real) for b in blocks) + head) + real * (cr[0] + cr[1]) + kept * (cr[2] + cr[3]))
    assert any(abs(v - expect) < 1e-6 * max(1.0, abs(expect)) for v in flops), (expect, flops)


def test_partial_ffn_applies_early_lora_adapters():
    """An FFN with LoRA adapters (early_lora) under partial width: the adapters fire (no silent no-op) on the same
    neuron subset, and only the subset's adapter rows / columns get gradients."""
    from olmo_core.nn.early_lora import install_early_lora

    torch.manual_seed(0)
    m = _M(n=2)
    m.embeddings = torch.nn.Embedding(10, 16)
    install_early_lora(m, 1, 4, scale=2.0, check_calls=False)
    ff = m.blocks["0"].feed_forward
    with torch.no_grad():
        for lin in (ff.w1, ff.w2, ff.w3):
            lin.lora_B.normal_()
    x = torch.randn(1, 4, 16)
    h = install_ffn_width(m, 0.5, "0", seed=4)
    h.set_step(3)
    ff.train()
    y = ff(x)
    idx = h.indices(0, 64, "cpu")
    def lin(mod, inp, rows=None, cols=None):
        w = mod.weight if rows is None else mod.weight[rows]
        w = w if cols is None else w[:, cols]
        A = mod.lora_A if cols is None else mod.lora_A[:, cols]
        B = mod.lora_B if rows is None else mod.lora_B[rows]
        return inp @ w.T + 2.0 * (inp @ A.T) @ B.T
    z = torch.nn.functional.silu(lin(ff.w1, x, rows=idx)) * lin(ff.w3, x, rows=idx)
    ref = lin(ff.w2, z, cols=idx) * 2.0
    assert torch.allclose(y, ref, atol=1e-5)
    assert all(lin_._lora_calls > 0 for lin_ in (ff.w1, ff.w2, ff.w3))
    y.sum().backward()
    assert ff.w1.weight.grad is None
    gB = ff.w1.lora_B.grad.abs().sum(1) > 0
    assert torch.equal(gB.nonzero().flatten(), idx)


def test_topk_refresh_is_full_width_then_keeps_the_most_active_neurons():
    torch.manual_seed(0)
    m = _M(n=1, hid=64)
    ff = m.blocks["0"].feed_forward
    with torch.no_grad():  # make neurons 0..15 by far the most active
        ff.w1.weight[:16] *= 20.0
        ff.w3.weight[:16] *= 20.0
    x = torch.randn(2, 5, 16)
    full = ff(x).detach()
    h = install_ffn_width(m, 0.5, mode="topk", rescale=False, explore=0.25, refresh_every=10, seed=1)
    ff.train()
    h.set_step(1)
    assert h.is_refresh() and h.frac_of(0) == 1.0
    assert torch.allclose(ff(x), full, atol=1e-5)  # refresh = exact full width
    ff(x)  # a second micro-batch / AC recompute does not double-count
    assert h._pending[0][1] == 10.0
    h.set_step(2)
    assert not h.is_refresh() and 0 in h.importance and not h._pending
    idx = h.indices(0, 64, "cpu")
    assert idx.numel() == 32
    assert set(range(16)) <= set(idx.tolist())  # the top neurons are kept
    k_top = 32 - 8  # explore 0.25 of the kept set
    imp_order = torch.argsort(h.importance[0], descending=True)[:k_top]
    assert set(imp_order.tolist()) <= set(idx.tolist())
    assert h.frac_of(0) == 0.5
    y = ff(x)  # partial, no rescale: exact on the kept neurons
    ref = torch.nn.functional.linear(
        torch.nn.functional.silu(x @ ff.w1.weight[idx].T) * (x @ ff.w3.weight[idx].T), ff.w2.weight[:, idx])
    assert torch.allclose(y, ref, atol=1e-5)
    assert (y - full).norm() < 0.2 * full.norm()  # the dropped half carries little of the output
    h.set_step(10)
    assert h.is_refresh() and h.frac_of(0) == 1.0  # periodic refresh, charged at full width
    assert h.credits_per_block(1) == [0.0]
    h.set_step(11)
    assert h.credits_per_block(1)[0] == pytest.approx(-0.5 * h.ffn_fpt[0])


def test_topk_rejects_bad_args():
    with pytest.raises(ValueError):
        FFNWidthHolder(0.5, [0], mode="topk", explore=1.0)
    with pytest.raises(ValueError):
        FFNWidthHolder(0.5, [0], mode="bogus")
