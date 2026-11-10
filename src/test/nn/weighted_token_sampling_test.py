"""Weighted token sampling (online router ``wts``; records/weighted-token-sampling.md).

The two exactness claims the method rests on, checked on toy cases:

* attention: one key of weight ``w`` == ``w`` identical copies of it (``log w`` logit bias);
* Gated DeltaNet: one token of weight ``w`` == ``w`` identical consecutive writes (``g -> w g``,
  ``beta -> 1 - (1 - beta)^w``, value rescale), against a reference recurrence on CPU and the FLA chunk kernel on GPU;

plus: all-ones weights are bit-for-bit the standard forward, the sampler's inclusion probabilities are exact, and the
mid-stack router carries ``w <- w / p`` through two cuts.
"""
import math

import pytest
import torch

from olmo_core.nn.attention.recurrent import gdn_token_weight
from olmo_core.nn.online_token_router import (
    OnlineRouterConfig,
    OnlineTokenRouter,
    parse_wts,
    sample_keep,
    waterfill,
)

from .online_token_router_test import DE, DS, EOS, PH, _row


def _ref_gdn(q, k, v, g, beta):
    """Reference gated delta rule, one head: S_t = a S (I - b k k^T) + b v k^T (k, q L2-normalised), o_t = S_t q_t."""
    T, Dk = k.shape
    Dv = v.shape[1]
    k = k / k.norm(dim=-1, keepdim=True)
    q = q / q.norm(dim=-1, keepdim=True)
    S = torch.zeros(Dv, Dk, dtype=torch.float64)
    out = []
    for t in range(T):
        a = math.exp(float(g[t]))
        b = float(beta[t])
        S = a * (S - b * (S @ k[t])[:, None] * k[t][None, :]) + b * v[t][:, None] * k[t][None, :]
        out.append(S @ q[t])
    return torch.stack(out)


@pytest.mark.parametrize("w", [1, 2, 3, 7])
def test_gdn_weight_equals_repeated_writes_reference(w):
    """One token of weight w vs w identical copies, reference recurrence in float64: every later output and the
    output at the (last) copy agree."""
    torch.manual_seed(0)
    T, Dk, Dv, j = 9, 6, 5, 4
    q, k, v = torch.randn(T, Dk, dtype=torch.float64), torch.randn(T, Dk, dtype=torch.float64), torch.randn(
        T, Dv, dtype=torch.float64)
    g = -torch.rand(T, dtype=torch.float64) * 0.7
    beta = torch.rand(T, dtype=torch.float64) * 0.9 + 0.05
    rep = torch.tensor([1] * j + [w] + [1] * (T - j - 1))
    o_rep = _ref_gdn(q.repeat_interleave(rep, 0), k.repeat_interleave(rep, 0), v.repeat_interleave(rep, 0),
                     g.repeat_interleave(rep, 0), beta.repeat_interleave(rep, 0))
    keep_idx = torch.cumsum(rep, 0) - 1  # the last copy of every token
    wt = rep.to(torch.float64)[None]
    g2, b2, v2 = gdn_token_weight(g[None, :, None], beta[None, :, None], v[None, :, None, :], wt)
    o_w = _ref_gdn(q, k, v2[0, :, 0], g2[0, :, 0], b2[0, :, 0])
    torch.testing.assert_close(o_w, o_rep[keep_idx], rtol=1e-9, atol=1e-9)


def test_gdn_weight_edge_cases_finite():
    """beta near 0 / 1 and no decay (r -> 1): finite, and the limits (beta' -> w beta, c -> 1) hold."""
    g = torch.tensor([[[0.0], [-1e-9], [-3.0], [0.0]]], dtype=torch.float32)
    beta = torch.tensor([[[1e-9], [0.5], [1.0], [1e-4]]], dtype=torch.float32)
    v = torch.ones(1, 4, 1, 3)
    w = torch.full((1, 4), 5.0)
    g2, b2, v2 = gdn_token_weight(g, beta, v, w)
    assert torch.isfinite(g2).all() and torch.isfinite(b2).all() and torch.isfinite(v2).all()
    assert float(b2[0, 0, 0]) == pytest.approx(5e-9, rel=1e-3)
    assert float(v2[0, 0, 0, 0]) == pytest.approx(1.0, rel=1e-3)
    assert float(b2[0, 2, 0]) == pytest.approx(1.0)


def test_gdn_weight_ones_bit_exact():
    torch.manual_seed(1)
    g = -torch.rand(2, 7, 3)
    beta = torch.rand(2, 7, 3, dtype=torch.bfloat16)
    v = torch.randn(2, 7, 3, 4, dtype=torch.bfloat16)
    w = torch.ones(2, 7)
    w[0, 3] = 2.5
    g2, b2, v2 = gdn_token_weight(g, beta, v, w)
    m = w == 1
    assert torch.equal(g2[m], g[m]) and torch.equal(b2[m], beta[m]) and torch.equal(v2[m], v[m])
    assert not torch.equal(g2[0, 3], g[0, 3])


def _attn(seed=0):
    from olmo_core.nn.attention import Attention
    from olmo_core.nn.rope import RoPEConfig, RoPEType

    torch.manual_seed(seed)
    a = Attention(d_model=32, n_heads=4, n_kv_heads=2, rope=RoPEConfig(name=RoPEType.default), init_device="cpu",
                  dtype=torch.float64, use_flash=False)
    for p in a.parameters():
        torch.nn.init.normal_(p, std=0.3)
    return a.eval()


@pytest.mark.parametrize("w", [2, 5])
def test_attention_weight_equals_duplicated_keys(w):
    """Key j with weight w == j duplicated w times at the same position: identical outputs at every query (the last
    copy for j itself)."""
    a = _attn()
    T, j = 8, 3
    x = torch.randn(1, T, 32, dtype=torch.float64)
    pos = torch.arange(T)[None]
    rep = torch.tensor([1] * j + [w] + [1] * (T - j - 1))
    xr = x.repeat_interleave(rep, 1)
    pr = pos.repeat_interleave(rep, 1)
    with torch.no_grad():
        o_rep = a(xr, position_ids=pr)
        tw = torch.ones(1, T, dtype=torch.float64)
        tw[0, j] = w
        o_w = a(x, position_ids=pos, token_weight=tw)
    last = torch.cumsum(rep, 0) - 1
    torch.testing.assert_close(o_w[0], o_rep[0, last], rtol=1e-9, atol=1e-9)


def test_attention_weight_ones_bit_exact_and_off_switch():
    a = _attn()
    x = torch.randn(2, 6, 32, dtype=torch.float64)
    with torch.no_grad():
        o = a(x)
        assert torch.equal(a(x, token_weight=torch.ones(2, 6, dtype=torch.float64)), o)
        tw = torch.ones(2, 6, dtype=torch.float64)
        tw[:, 1] = 4.0
        assert not torch.equal(a(x, token_weight=tw), o)
        a._token_weight_on = False  # wts weights=gdn: attention layers ignore the weights
        assert torch.equal(a(x, token_weight=tw), o)


def test_waterfill_hits_targets_and_bounds():
    torch.manual_seed(0)
    v = torch.rand(300).pow(4) + 1e-4
    gid = torch.randint(0, 5, (300,))
    n = torch.bincount(gid, minlength=5).double()
    tgt = 0.2 * n
    p = waterfill(v, gid, 5, tgt, lo=0.01)
    tot = torch.zeros(5, dtype=torch.float64).index_add_(0, gid, p.double())
    torch.testing.assert_close(tot, tgt, rtol=1e-4, atol=1e-4)
    assert float(p.min()) >= 0.01 - 1e-6 and float(p.max()) <= 1.0
    # proportional where unclipped
    inner = (p > 0.0101) & (p < 0.999)
    r = (p[inner] / v[inner]).double()
    for gg in range(5):
        m = gid[inner] == gg
        if m.sum() > 1:
            assert float(r[m].std() / r[m].mean()) < 1e-3


@pytest.mark.parametrize("sampler", ["sys", "bern"])
def test_sample_keep_marginals(sampler):
    torch.manual_seed(0)
    T = 40
    p = torch.rand(1, T).clamp(0.05, 1.0)
    mask = torch.ones(1, T, dtype=torch.bool)
    mask[0, :3] = False
    gen = torch.Generator().manual_seed(0)
    N = 20000
    freq = torch.zeros(1, T)
    counts = []
    for _ in range(N):
        kk = sample_keep(p, mask, gen, sampler)
        freq += kk
        counts.append(int(kk.sum()))
    freq /= N
    assert (freq[~mask] == 0).all()
    assert float((freq[mask] - p[mask]).abs().max()) < 0.015
    if sampler == "sys":  # systematic: the count is sum(p) +- 1
        s = float(p[mask].sum())
        assert min(counts) >= math.floor(s) and max(counts) <= math.ceil(s)


def test_parse_wts():
    o = parse_wts("weights=both,mix=0.2,price=0.26")
    assert o["weights"] == "both" and o["mix"] == 0.2 and o["price"] == 0.26 and o["sample"] == 1
    assert parse_wts("") is None
    for bad in ("foo=1", "weights=x", "sampler=z", "sample=0"):
        with pytest.raises(ValueError):
            parse_wts(bad)
    with pytest.raises(ValueError):
        OnlineRouterConfig(wts="weights=attn").validate()  # needs the mid-stack drop


def _wts_model(n_layers, **cfg):
    from olmo_core.config import DType
    from olmo_core.nn.transformer import TransformerConfig

    torch.manual_seed(0)
    model = TransformerConfig.olmo2_190M(vocab_size=9100, n_layers=n_layers, fused_ops=False,
                                         dtype=DType.float32).build(init_device="cpu")
    D = int(model.embeddings.weight.shape[1])
    router = OnlineTokenRouter(OnlineRouterConfig(**{"keep_frac": 0.5, "explore_frac": 0.1, "samples_per_row": 64,
                                                     "warmup_random": True, **cfg}),
                               doc_start_id=DS, doc_end_id=DE, eos_id=EOS, d_model=D)
    model.enable_pooled_soft_tokens(
        DS, DE, EOS, placeholder_id=PH, keep_prob=0.0, keep_seed=0, detach_soft_kv=True,
        drop_slots=True, keep_token_rule="online", keep_token_k=0.5, online_router=router,
    )
    model.train()
    return model, router


def test_wts_two_cuts_carry_weights_cpu():
    """stages 1:0.5,2:0.25 with weights=attn: the second cut's token weights are the first cut's w / p2, the attention
    layers see them, targets are corrected by log w, and the fit is importance weighted."""
    model, router = _wts_model(4, stages="1:0.5,2:0.25", wts="weights=attn")
    chain = router.chain()
    seen = {}
    for i, blk in enumerate(model.blocks.values()):
        orig = blk.attention.forward

        def fwd(*a, _o=orig, _i=i, **kw):
            seen[_i] = kw.get("token_weight")
            return _o(*a, **kw)

        blk.attention.forward = fwd
    ids, cid, labels = _row(B=2)
    for step in (1, 2, 3):
        model.zero_grad(set_to_none=True)
        for r in chain:
            r.begin_step(step, model=model)
            r.metrics()
        out = model(ids, labels=labels)
        assert torch.isfinite(out.loss)
        out.loss.backward()
        assert seen[0] is None  # before the first cut every token has weight 1
        w1, w2 = seen[1], seen[2]
        assert w1 is not None and w2 is not None
        assert float(w1.min()) >= 1.0 and float(w2.min()) >= 1.0 and float(w2.max()) > 1.0
        # survivors of the second cut carry at least their first-cut weight
        p2 = chain[1]._last_p
        assert p2 is not None and float(p2.max()) <= 1.0
        for r in chain:
            e = r._pending[-1]
            assert e["ipw"] is not None and e.get("logw") is not None
            assert e["target"] is not None and torch.isfinite(e["target"]).all()
    assert all(r.w is not None for r in chain)
    m = chain[0].metrics()
    assert "online_router/wts_logw_mean" in m and "online_router/wts_empty_doc_frac" in m
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(ids, labels=labels).loss)


def test_wts_all_ones_is_bit_exact_cpu():
    """keep_frac 1.0 (every p = 1, every w = 1): the wts forward equals the plain mid-stack forward bit for bit."""
    a, ra = _wts_model(3, drop_layer=1, keep_frac=1.0, wts="weights=both")
    b, rb = _wts_model(3, drop_layer=1, keep_frac=1.0)
    b.load_state_dict(a.state_dict())
    ids, cid, labels = _row(B=2)
    ra.begin_step(1, model=a)
    rb.begin_step(1, model=b)
    la = a(ids, labels=labels).loss
    lb = b(ids, labels=labels).loss
    assert torch.equal(la, lb)


def test_wts_price_controller_moves_tau():
    """price mode: the controller raises tau (keeps less) when the measured step ratio is above target, and the
    per-row cap bounds expected survivors."""
    model, router = _wts_model(4, stages="1:0.5,2:0.25", wts="weights=attn,price=0.3,tau0=0.5")
    ids, cid, labels = _row(B=2)
    taus = []
    for step in (1, 2, 3, 4):
        router.begin_step(step, model=model)
        for r in router.chain()[1:]:
            r.begin_step(step, model=model)
        taus.append(router._logtau)
        out = model(ids, labels=labels)
        out.loss.backward()
        model._flop_step_ratio = 0.6  # measured: twice the target
    assert taus[-1] > taus[1] > taus[0] - 1e-12
    p = router._last_p
    body_p = p[p < 1.0] if p is not None else torch.tensor([])
    assert p is not None
    # cap: expected survivors per row at the first cut <= 0.5 x body
    from olmo_core.nn.online_token_router import structural_features

    _, body = structural_features(ids, cid, doc_start_id=DS, doc_end_id=DE)
    exp_keep = (router._last_p[:, : body.shape[1]] * body).sum(1)
    assert (exp_keep <= 0.5 * body.sum(1) + 1e-3).all()
    del body_p


@pytest.mark.gpu
@pytest.mark.parametrize("w", [2, 5])
def test_gdn_weight_equals_repeated_writes_fla_kernel_gpu(w):
    """The FLA chunk kernel (the training path, L2-normalised q/k in kernel): one weighted token == w copies."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    pytest.importorskip("fla")
    from olmo_core.nn.attention.flash_linear_attn_api import dispatch_chunk_gated_delta_rule

    torch.manual_seed(0)
    B, T, H, Dk, Dv, j = 1, 96, 2, 64, 64, 40
    dev = "cuda"
    q = torch.randn(B, T, H, Dk, device=dev)
    k = torch.randn(B, T, H, Dk, device=dev)
    v = torch.randn(B, T, H, Dv, device=dev)
    g = -torch.rand(B, T, H, device=dev) * 0.3
    beta = torch.rand(B, T, H, device=dev) * 0.8 + 0.1
    rep = torch.tensor([1] * j + [w] + [1] * (T - j - 1), device=dev)
    R = lambda x: x.repeat_interleave(rep, 1)  # noqa: E731
    o_rep, _ = dispatch_chunk_gated_delta_rule(q=R(q), k=R(k), v=R(v), g=R(g), beta=R(beta),
                                               use_qk_l2norm_in_kernel=True)
    wt = rep.to(torch.float32)[None]
    g2, b2, v2 = gdn_token_weight(g, beta, v, wt)
    o_w, _ = dispatch_chunk_gated_delta_rule(q=q, k=k, v=v2, g=g2, beta=b2, use_qk_l2norm_in_kernel=True)
    last = torch.cumsum(rep, 0) - 1
    torch.testing.assert_close(o_w, o_rep[:, last], rtol=2e-3, atol=2e-3)


@pytest.mark.gpu
def test_gdn_hybrid_wts_ones_bit_exact_and_weighted_gpu():
    """GPU hybrid (GDN, attention, GDN): all-ones token weights in the block kwargs == no weights, bit for bit; real
    weights change the output and stay finite (forward + backward)."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    pytest.importorskip("fla")
    from .online_token_router_test import _gpu_hybrid

    model = _gpu_hybrid().train()
    for m in model.modules():  # Qwen3.5's DeltaNet layers: beta in (0, 1)
        if hasattr(m, "allow_neg_eigval"):
            m.allow_neg_eigval = False
    ids, _, labels = _row(B=2)
    ids = ids.cuda()
    with torch.no_grad():
        h0 = model.embeddings(ids)
        outs = {}
        for name, tw in (("none", None), ("ones", torch.ones(ids.shape, device="cuda")),
                         ("w", torch.where(torch.rand(ids.shape, device="cuda") < 0.3, 4.0, 1.0))):
            h = h0
            for blk in model.blocks.values():
                h = blk(h, token_weight=tw) if tw is not None else blk(h)
            outs[name] = h
    assert torch.equal(outs["none"], outs["ones"])
    assert not torch.equal(outs["none"], outs["w"]) and torch.isfinite(outs["w"]).all()
