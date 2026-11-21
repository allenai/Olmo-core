"""Group-partitioned attention (olmo_core.nn.attention.group_attention): exactness vs full attention / masked
reference, the chunked-prefix (q_start > 0) path, the Attention-module wiring and the FLOP pair count."""
import pytest
import torch

from olmo_core.nn.attention.group_attention import (
    HUB,
    PAD,
    STRIP,
    group_attention,
    group_attention_pairs,
    group_mask_reference,
)

gpu = pytest.mark.gpu
DEV = "cuda"


def _toy_gid(B, T, n_docs, doc_len, groups, hubs_per_row, pre=5, pad=7, full_row=None, seed=0):
    """Rows: preamble (HUB) + n_docs docs of doc_len tokens + strip + pad; docs assigned to hubs / groups."""
    g = torch.Generator().manual_seed(seed)
    gid = torch.full((B, T), PAD, dtype=torch.int32)
    for b in range(B):
        L = T - pad
        gid[b, :pre] = HUB
        perm = torch.randperm(n_docs, generator=g).tolist()
        hubs = set(perm[:hubs_per_row])
        rest = perm[hubs_per_row:]
        cls = {}
        for i, d in enumerate(rest):
            cls[d] = 1 + i % groups
        for d in range(n_docs):
            s = pre + d * doc_len
            gid[b, s:s + doc_len] = HUB if d in hubs else cls[d]
        gid[b, pre + n_docs * doc_len:L] = STRIP
        if full_row is not None and b == full_row:
            gid[b, :L] = STRIP
    return gid


def _qkv(B, T, Hq=4, Hk=2, D=64, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, T, Hq, D, device=DEV, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, Hk, D, device=DEV, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, Hk, D, device=DEV, dtype=dtype, requires_grad=True)
    return q, k, v


def _masked_ref(q, k, v, allowed):
    """fp32 reference: SDPA with a boolean mask (GQA expanded)."""
    Hq, Hk = q.shape[2], k.shape[2]
    qf, kf, vf = (t.float().transpose(1, 2) for t in (q, k, v))
    kf = kf.repeat_interleave(Hq // Hk, dim=1)
    vf = vf.repeat_interleave(Hq // Hk, dim=1)
    o = torch.nn.functional.scaled_dot_product_attention(qf, kf, vf, attn_mask=allowed[:, None].to(DEV))
    return o.transpose(1, 2)


def _grads(out, real, q, k, v, w):
    loss = (out.float() * w)[real].sum()
    return torch.autograd.grad(loss, (q, k, v))


@gpu
def test_g1_all_docs_one_group_equals_full_attention():
    from flash_attn import flash_attn_func

    B, n_docs, doc_len = 2, 12, 37
    T = 5 + n_docs * doc_len + 20 + 7
    gid = _toy_gid(B, T, n_docs, doc_len, groups=1, hubs_per_row=0).to(DEV)
    q, k, v = _qkv(B, T)
    real = (gid != PAD)
    w = torch.randn(B, T, 4, 64, device=DEV)
    o_g = group_attention(q, k, v, gid)
    o_f = flash_attn_func(q, k, v, causal=True)
    d_out = (o_g.float() - o_f.float())[real].abs().max().item()
    gg = _grads(o_g, real, q, k, v, w)
    gf = _grads(o_f, real, q, k, v, w)
    d_g = [(a.float() - b.float()).abs().max().item() for a, b in zip(gg, gf)]
    print(f"G=1 vs full flash: out {d_out:.3e} grads q/k/v {d_g}")
    assert d_out < 1e-2
    assert max(d_g) < 5e-2


@gpu
@pytest.mark.parametrize("groups", [2, 4, 8])
def test_groups_equal_masked_reference(groups):
    B, n_docs, doc_len = 3, 40, 23
    T = 5 + n_docs * doc_len + 30 + 7
    gid = _toy_gid(B, T, n_docs, doc_len, groups=groups, hubs_per_row=6, full_row=2, seed=groups).to(DEV)
    q, k, v = _qkv(B, T, seed=groups)
    real = gid != PAD
    w = torch.randn(B, T, 4, 64, device=DEV)
    o_g = group_attention(q, k, v, gid)
    o_r = _masked_ref(q, k, v, group_mask_reference(gid))
    d_out = (o_g.float() - o_r)[real].abs().max().item()
    gg = _grads(o_g, real, q, k, v, w)
    gr = _grads(o_r, real, q, k, v, w)
    d_g = [(a.float() - b.float()).abs().max().item() for a, b in zip(gg, gr)]
    # bf16 kernel vs fp32 reference: compare against the same reference run on full causal attention
    o_fr = _masked_ref(q, k, v, torch.ones(T, T, dtype=torch.bool, device=DEV).tril()[None].expand(B, T, T))
    from flash_attn import flash_attn_func

    base = (flash_attn_func(q, k, v, causal=True).float() - o_fr)[real].abs().max().item()
    gbase = [(a.float() - b.float()).abs().max().item() for a, b in
             zip(_grads(flash_attn_func(q, k, v, causal=True), real, q, k, v, w), _grads(o_fr, real, q, k, v, w))]
    print(f"G={groups} vs masked fp32 ref: out {d_out:.3e} (flash full vs fp32 full {base:.3e}) grads {d_g} "
          f"(flash full vs fp32 full {gbase})")
    assert d_out < 3 * base + 1e-3
    assert all(d < 3 * b + 1e-2 for d, b in zip(d_g, gbase))
    # and the grouping actually changes the output (not silently full attention)
    o_full = flash_attn_func(q, k, v, causal=True)
    assert (o_g.float() - o_full.float())[real].abs().max().item() > 0.05


@gpu
def test_chunked_prefix_tail_queries_match_whole_row():
    B, n_docs, doc_len = 2, 30, 19
    T = 5 + n_docs * doc_len + 25 + 7
    gid = _toy_gid(B, T, n_docs, doc_len, groups=4, hubs_per_row=4, seed=3).to(DEV)
    q, k, v = _qkv(B, T, seed=3)
    real = gid != PAD
    whole = group_attention(q, k, v, gid)
    cuts = [0, 97, 250, 401, T]
    parts = [group_attention(q[:, s:e], k[:, :e], v[:, :e], gid[:, :e], q_start=s) for s, e in zip(cuts, cuts[1:])]
    d = (torch.cat(parts, 1).float() - whole.float())[real].abs().max().item()
    print(f"chunked vs whole: {d:.3e}")
    assert d < 1e-2


@gpu
def test_attention_module_wiring_g1_and_grouped():
    from olmo_core.nn.attention import Attention, AttentionBackendName

    torch.manual_seed(0)
    att = Attention(d_model=256, n_heads=4, n_kv_heads=2, head_dim=64, bias=False,
                    backend=AttentionBackendName.flash_2, init_device=DEV, dtype=torch.bfloat16)
    B, n_docs, doc_len = 2, 10, 21
    T = 5 + n_docs * doc_len + 15 + 7
    x = torch.randn(B, T, 256, device=DEV, dtype=torch.bfloat16)
    gid1 = _toy_gid(B, T, n_docs, doc_len, groups=1, hubs_per_row=0).to(DEV)
    real = gid1 != PAD
    y0 = att(x)
    y1 = att(x, attn_group_ids=gid1)
    d1 = (y1.float() - y0.float())[real].abs().max().item()
    gid4 = _toy_gid(B, T, n_docs, doc_len, groups=4, hubs_per_row=2).to(DEV)
    y4 = att(x, attn_group_ids=gid4)
    d4 = (y4.float() - y0.float())[gid4 != PAD].abs().max().item()
    print(f"module: G=1 vs plain {d1:.3e}; G=4 vs plain {d4:.3e}")
    assert d1 < 1e-2 and d4 > 1e-2


def test_pair_count_hand():
    # one row: 2 preamble (HUB), hub doc 3 tokens, group1 doc 4, group2 doc 2, group1 doc 1, strip 3, pad 2
    g = [HUB] * 2 + [HUB] * 3 + [1] * 4 + [2] * 2 + [1] * 1 + [STRIP] * 3 + [PAD] * 2
    gid = torch.tensor([g], dtype=torch.int32)
    actual, charged = group_attention_pairs(gid)
    L = 15
    n_hub = 5
    hand = n_hub * 6 / 2  # hub seq 5x6/2 = 15
    hand += (5 + 5) * 11 / 2  # group 1 seq: 5 hub + 5 group tokens = 55
    hand += (5 + 2) * 8 / 2  # group 2: 28
    hand += 3 * 15 - 3 * 2 / 2  # strip: 3 queries over the 15-token prefix = 42
    assert charged == L * (L + 1) / 2 == 120
    assert actual == hand == 140
    # full row (all strip) charges exactly the meter's triangle
    full = torch.tensor([[STRIP] * 9 + [PAD]], dtype=torch.int32)
    a, c = group_attention_pairs(full)
    assert a == c == 45


def test_charge_attn_groups_hand():
    """FLOP charge on a toy 4-block model (attention in blocks 1 and 3, a non-attention mixer elsewhere)."""
    from olmo_core.nn.attention import Attention
    from olmo_core.nn.attention.group_attention import charge_attn_groups

    class Blk(torch.nn.Module):
        def __init__(self, att):
            super().__init__()
            self.attention = att if att is not None else torch.nn.Identity()

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            mk = lambda: Attention(d_model=64, n_heads=4, n_kv_heads=2, head_dim=16, bias=False)  # noqa: E731
            self.blocks = torch.nn.ModuleDict({"0": Blk(None), "1": Blk(mk()), "2": Blk(None), "3": Blk(mk())})

    m = M()
    g = [HUB] * 5 + [1] * 4 + [2] * 2 + [1] * 1 + [STRIP] * 3 + [PAD] * 2
    gid = torch.tensor([g], dtype=torch.int32)
    tot = charge_attn_groups(m, gid, 0)
    per_layer = 12 * 4 * 16 * (140 - 120)
    assert tot == 2 * per_layer and m._flop_extra_actual == 2 * per_layer
    # from block 2 on: one attention layer
    assert charge_attn_groups(m, gid, 2, sign=-1.0) == -per_layer
