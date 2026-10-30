"""olmo_core.nn.sparse_backward: exact forward, compressed VJP of frozen-base + LoRA linears (CPU)."""
import math

import torch
import torch.nn as nn

from olmo_core.nn.early_lora import install_early_lora
from olmo_core.nn.sparse_backward import (
    SparseBackwardConfig,
    install_sparse_backward,
    select_rows,
    sparse_backward_metrics,
)


class _Blk(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.up = nn.Linear(d, h, bias=False)
        self.down = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return x + self.down(torch.tanh(self.up(x)))


class _Tiny(nn.Module):
    def __init__(self, d=16, h=24, n=3):
        super().__init__()
        self.embeddings = nn.Embedding(10, d)
        self.blocks = nn.ModuleDict({str(i): _Blk(d, h) for i in range(n)})
        self.lm_head = nn.Linear(d, 5, bias=False)

    def forward(self, x):
        for b in self.blocks.values():
            x = b(x)
        return self.lm_head(x)


def _model(seed=0):
    torch.manual_seed(seed)
    m = _Tiny().double()
    install_early_lora(m, 3, rank=4, scale=2.0)
    with torch.no_grad():
        for mod in m._early_lora["modules"]:
            mod.lora_B.normal_(0, 0.1)  # nonzero B so dA is nonzero too
    m._flop_extra_actual = 0.0
    return m


def _grads(m, x, w):
    for p in m.parameters():
        p.grad = None
    xx = x.clone().requires_grad_(True)
    loss = (m(xx) * w).sum()
    loss.backward()
    g = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    return float(loss), g, xx.grad.clone()


def _ref():
    m = _model()
    torch.manual_seed(1)
    x = torch.randn(2, 7, 16, dtype=torch.float64)
    w = torch.randn(2, 7, 5, dtype=torch.float64)
    w[:, -2:] = 0  # trailing rows get no gradient (like pads / post-answer tokens)
    return m, x, w


def test_forward_exact_and_full_selection_exact():
    m0, x, w = _ref()
    l0, g0, gx0 = _grads(m0, x, w)
    for cfg in (SparseBackwardConfig("topk", frac=1.0), SparseBackwardConfig("energy", energy=1.0),
                SparseBackwardConfig("sample", frac=1.0), SparseBackwardConfig("sketch", rank=64, sketch=0),
                SparseBackwardConfig("topk", frac=1.0, lora_all_rows=True)):
        m = _model()
        install_sparse_backward(m, cfg)
        l1, g1, gx1 = _grads(m, x, w)
        assert abs(l1 - l0) < 1e-10, cfg
        assert set(g1) == set(g0)
        for n in g0:
            torch.testing.assert_close(g1[n], g0[n], rtol=1e-8, atol=1e-10, msg=f"{cfg.mode} {n}")
        torch.testing.assert_close(gx1, gx0, rtol=1e-8, atol=1e-10)
        if cfg.mode != "sketch":
            # every nonzero row computed: FLOP delta = the row-norm overhead only (no saving claimed)
            assert m._flop_extra_actual > 0
        mt = sparse_backward_metrics(m)
        assert abs(mt["sparse_bw/row_frac"] - 1.0) < 1e-9


def test_topk_drops_rows_and_charges_less():
    _, x, w = _ref()
    m = _model()
    install_sparse_backward(m, SparseBackwardConfig("topk", frac=0.25))
    _grads(m, x, w)
    mt = sparse_backward_metrics(m)
    assert mt["sparse_bw/row_frac"] < 0.5
    assert m._flop_extra_actual < 0  # charged less than dense


def test_sample_unbiased():
    m0, x, w = _ref()
    _, g0, gx0 = _grads(m0, x, w)
    m = _model()
    install_sparse_backward(m, SparseBackwardConfig("sample", frac=0.4))
    torch.manual_seed(3)
    acc, accx, K = None, None, 3000
    for _ in range(K):
        _, g, gx = _grads(m, x, w)
        acc = g if acc is None else {n: acc[n] + g[n] for n in acc}
        accx = gx if accx is None else accx + gx
    for n in g0:
        if "lora" not in n:
            continue
        err = (acc[n] / K - g0[n]).norm() / g0[n].norm()
        assert err < 0.1, (n, float(err))
    assert float((accx / K - gx0).norm() / gx0.norm()) < 0.1


def test_sketch_unbiased():
    m0, x, w = _ref()
    _, _, gx0 = _grads(m0, x, w)
    m = _model()
    install_sparse_backward(m, SparseBackwardConfig("sketch", rank=4, sketch=8))
    torch.manual_seed(4)
    accx, K = None, 3000
    for _ in range(K):
        _, _, gx = _grads(m, x, w)
        accx = gx if accx is None else accx + gx
    assert float((accx / K - gx0).norm() / gx0.norm()) < 0.1


def test_select_rows_sample_budget():
    torch.manual_seed(0)
    n2 = torch.rand(1000).pow(4)
    n2[:100] = 0
    cfg = SparseBackwardConfig("sample", frac=0.2)
    ks = [select_rows(n2, cfg)[0].numel() for _ in range(200)]
    assert abs(sum(ks) / len(ks) - 0.2 * 900) < 5
    idx, wt, nz = select_rows(n2, SparseBackwardConfig("energy", energy=0.9))
    assert nz == 900
    assert float(n2[idx].sum()) >= 0.9 * float(n2.sum()) - 1e-6
    assert math.isfinite(float(wt.sum())) if wt is not None else True


def test_eval_no_grad_uses_hook():
    m0, x, w = _ref()
    m = _model()
    install_sparse_backward(m, SparseBackwardConfig("topk", frac=0.1))
    with torch.no_grad():
        torch.testing.assert_close(m(x), m0(x))


def test_lora_all_rows_exact_adapter_grads_at_top_block():
    # the TOP block's adapters see the exact dy, so with lora_all_rows their gradients are exact even at frac 0.2
    m0, x, w = _ref()
    _, g0, _ = _grads(m0, x, w)
    m = _model()
    install_sparse_backward(m, SparseBackwardConfig("topk", frac=0.2, lora_all_rows=True))
    _, g1, _ = _grads(m, x, w)
    for n in ("blocks.2.down.lora_A", "blocks.2.down.lora_B"):
        torch.testing.assert_close(g1[n], g0[n], rtol=1e-8, atol=1e-10)
    assert (g1["blocks.0.up.lora_B"] - g0["blocks.0.up.lora_B"]).abs().max() > 0  # lower blocks: approximate


def test_dense_rows_registered_real_count():
    m = _model()
    install_sparse_backward(m, SparseBackwardConfig("topk", frac=0.5))
    st = m._sparse_backward
    st.real_rows = {14: 10.0}  # 2 x 7 rows, 10 real
    assert st.dense_rows(14, 8) == 10.0
    assert st.dense_rows(15, 8) == 8.0  # unknown call: the nonzero rows (conservative)
    st.assume_no_pad = True
    assert st.dense_rows(15, 8) == 15.0
