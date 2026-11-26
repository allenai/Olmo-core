"""Correctness of edge_attn on toy sequences (CPU or GPU, seconds):

1. ``full``: output and q/k/v gradients match autograd through an explicit causal softmax attention (GQA).
2. The streamed per-edge quantities match autograd: |dL/ds_ij| (s.grad of the explicit logits), a_ij |<g_i, v_j>|
   and a_ij, aggregated to (query segment, key segment) blocks per head, to distance bins and in the raw-edge
   histogram totals.
3. ``detach`` with every edge kept reproduces ``full`` bit for bit; ``detach`` with a pattern equals the explicit
   truncated decomposition (dS * keep with the full D_i, dV from a * keep).
4. ``mask`` equals autograd through an explicitly masked softmax.

    python -m pytest research/token_routing/attn_edge_grad/test_edge_attn.py -q     (on a compute node)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edge_attn as E  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False


def toy(T=53, H=4, Hk=2, D=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, T, H, D, generator=g).to(DEV)
    k = torch.randn(1, T, Hk, D, generator=g).to(DEV)
    v = torch.randn(1, T, Hk, D, generator=g).to(DEV)
    G = torch.randn(1, T, H, D, generator=g).to(DEV)
    seg = torch.tensor(sorted(torch.randint(0, 7, (T,), generator=g).tolist()), device=DEV)
    seg = torch.unique(seg, return_inverse=True)[1]
    return q, k, v, G, seg


def ref_attn(q, k, v, keep=None):
    """explicit attention; returns o and the logits s (retain_grad)."""
    B, T, H, D = q.shape
    rep = H // k.shape[2]
    kk = k.repeat_interleave(rep, dim=2)
    vv = v.repeat_interleave(rep, dim=2)
    s = torch.einsum("bihd,bjhd->bhij", q, kk) * D ** -0.5
    s.retain_grad()
    causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    m = causal if keep is None else causal & keep
    a = torch.softmax(s.masked_fill(~m, float("-inf")), -1)
    return torch.einsum("bhij,bjhd->bihd", a, vv), s, a, vv


def leafs(*xs):
    return [x.detach().clone().requires_grad_(True) for x in xs]


def run_edge(q, k, v, G, mode, pattern=None, stats=None, chunk_elems=4 * 16 * 53):
    st = E.EdgeState()
    st.mode, st.pattern, st.stats, st.chunk_elems = mode, pattern, stats, chunk_elems
    q2, k2, v2 = leafs(q, k, v)
    o = E.edge_attention(q2, k2, v2, layer=3, state=st)
    (o * G).sum().backward()
    return o.detach(), q2.grad, k2.grad, v2.grad


def test_full_matches_autograd():
    q, k, v, G, seg = toy()
    q1, k1, v1 = leafs(q, k, v)
    o_ref, s, a, _ = ref_attn(q1, k1, v1)
    (o_ref * G).sum().backward()
    o, dq, dk, dv = run_edge(q, k, v, G, "full")
    assert torch.allclose(o, o_ref.detach(), atol=1e-5)
    for a_, b_ in ((dq, q1.grad), (dk, k1.grad), (dv, v1.grad)):
        assert torch.allclose(a_, b_, atol=1e-4), (a_ - b_).abs().max()


def test_edge_quantities_match_autograd():
    q, k, v, G, seg = toy()
    T, H = q.shape[1], q.shape[2]
    q1, k1, v1 = leafs(q, k, v)
    o_ref, s, a, vv = ref_attn(q1, k1, v1)
    (o_ref * G).sum().backward()
    ds = s.grad[0]  # (H, T, T)
    u = torch.einsum("ihd,jhd->hij", G[0], vv[0].detach())
    val = a[0].detach() * u.abs()
    att = a[0].detach()
    nseg = int(seg.max()) + 1
    stats = E.EdgeStats(seg=seg, nseg=nseg, per_head=True, raw_hist=True)
    run_edge(q, k, v, G, "full", stats=stats)
    onehot = torch.nn.functional.one_hot(seg, nseg).double()
    for name, X in (("logit", ds.abs()), ("value", val), ("attn", att)):
        ref_blocks = torch.einsum("ia,hij,jb->hab", onehot, X.double(), onehot)
        got = stats.blocks[3][name]
        assert torch.allclose(got, ref_blocks, rtol=1e-4, atol=1e-6), (name, (got - ref_blocks).abs().max())
        i = torch.arange(T, device=DEV)
        d = (i[:, None] - i[None, :]).clamp_min(0)
        b = torch.floor(torch.log2(d.double() + 1)).long().clamp_max(E.NDIST - 1)
        causal = d.new_ones(T, T, dtype=torch.bool).tril()
        ref_d = torch.stack([torch.bincount(b[causal], weights=X[h][causal].double(), minlength=E.NDIST)
                             for h in range(H)]).flatten()
        assert torch.allclose(stats.dist[3][name], ref_d, rtol=1e-4, atol=1e-6), name
        tot = float(X.double().tril().sum())
        assert abs(float(stats.hist[3][name][1].sum()) - tot) < 1e-4 * max(1.0, tot), name
        assert int(stats.hist[3][name][0].sum()) == H * T * (T + 1) // 2
    # the value-path edge effects sum to <g_i, o_i> per query (first-order identity)
    assert torch.allclose((att * u).sum(-1), torch.einsum("ihd,ihd->hi", G[0], o_ref[0].detach()), atol=1e-4)


def test_detach_all_kept_is_full_exactly():
    q, k, v, G, seg = toy()
    nseg = int(seg.max()) + 1
    allk = E.Pattern("all", seg, K=torch.ones(nseg, nseg, dtype=torch.bool, device=DEV))
    full = run_edge(q, k, v, G, "full")
    det = run_edge(q, k, v, G, "detach", pattern=allk)
    for a_, b_ in zip(full, det):
        assert torch.equal(a_, b_)
    msk = run_edge(q, k, v, G, "mask", pattern=allk)
    for a_, b_ in zip(full, msk):
        assert torch.equal(a_, b_)


def _some_pattern(seg):
    nseg = int(seg.max()) + 1
    g = torch.Generator().manual_seed(1)
    K = (torch.rand(nseg, nseg, generator=g) < 0.4).to(DEV)
    T = seg.numel()
    rowall = torch.zeros(T, dtype=torch.bool, device=DEV)
    rowall[-3:] = True
    keyall = torch.zeros(T, dtype=torch.bool, device=DEV)
    keyall[:2] = True
    return E.Pattern("p", seg, K=K, rowall=rowall, keyall=keyall, window=3)


def test_detach_pattern_matches_truncated_decomposition():
    q, k, v, G, seg = toy()
    T, H, D = q.shape[1], q.shape[2], q.shape[3]
    pat = _some_pattern(seg)
    keep = pat.keep(0, T, 3)
    q1, k1, v1 = leafs(q, k, v)
    o_ref, s, a, vv = ref_attn(q1, k1, v1)
    (o_ref * G).sum().backward()
    ds = s.grad[0] * keep  # (H, T, T)
    A = a[0].detach() * keep
    rep = H // k.shape[2]
    kk = k.repeat_interleave(rep, dim=2)[0]
    dq = torch.einsum("hij,jhd->ihd", ds, kk) * D ** -0.5
    dk = torch.einsum("hij,ihd->jhd", ds, q[0]) * D ** -0.5
    dv = torch.einsum("hij,ihd->jhd", A, G[0])
    dk = dk.view(T, k.shape[2], rep, D).sum(2)
    dv = dv.view(T, k.shape[2], rep, D).sum(2)
    o, dq2, dk2, dv2 = run_edge(q, k, v, G, "detach", pattern=pat)
    assert torch.allclose(o, o_ref.detach(), atol=1e-5)
    for x, y in ((dq2[0], dq), (dk2[0], dk), (dv2[0], dv)):
        assert torch.allclose(x, y, atol=1e-4), (x - y).abs().max()


def test_mask_matches_masked_autograd():
    q, k, v, G, seg = toy()
    T = q.shape[1]
    pat = _some_pattern(seg)
    keep = pat.keep(0, T, 3)
    q1, k1, v1 = leafs(q, k, v)
    o_ref, s, a, _ = ref_attn(q1, k1, v1, keep=keep)
    (o_ref * G).sum().backward()
    o, dq, dk, dv = run_edge(q, k, v, G, "mask", pattern=pat)
    assert torch.allclose(o, o_ref.detach(), atol=1e-5)
    for a_, b_ in ((dq, q1.grad), (dk, k1.grad), (dv, v1.grad)):
        assert torch.allclose(a_, b_, atol=1e-4), (a_ - b_).abs().max()


def test_kept_fraction():
    q, k, v, G, seg = toy()
    T = seg.numel()
    pat = _some_pattern(seg)
    fr = pat.kept_fraction(T, [3], chunk=16)[3]
    assert abs(fr - float(pat.keep(0, T, 3).sum()) / (T * (T + 1) / 2)) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
