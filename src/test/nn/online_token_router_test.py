"""Online train-time token router (``keep_token_rule="online"``, olmo_core.nn.online_token_router)."""
import math

import pytest
import torch

from olmo_core.nn.attention.chunked_mask import FREE_CHUNK_ID, build_chunk_ids_from_tokens
from olmo_core.nn.online_token_router import (
    N_STRUCT,
    OnlineRouterConfig,
    OnlineTokenRouter,
    query_mask,
    structural_features,
)

DS, DE, EOS, PH = 9001, 9002, 9000, 9003
QTOK = 1005  # a body token of doc 0 that also occurs in the query


def _row(B=1):
    def doc(n, base):
        return [DS] + [base + i for i in range(n)] + [DE]

    one = [1, 2, 3] + doc(10, 1000) + doc(40, 2000) + doc(3, 3000) + [7, QTOK, 9] + [50, 51, EOS]
    ids = torch.tensor([one] * B)
    cid = build_chunk_ids_from_tokens(ids, doc_start_id=DS, doc_end_id=DE, eos_id=EOS, mode="chunked")
    labels = torch.full_like(ids, -100)
    T = ids.shape[1]
    labels[:, T - 4 : T - 1] = ids[:, T - 3 :]  # answer = [50, 51, EOS]
    return ids, cid, labels


class _Emb(torch.nn.Module):
    """Minimal stand-in for the model attributes the router reads at feature_layer 0."""

    def __init__(self, V=9100, D=16):
        super().__init__()
        self.embeddings = torch.nn.Embedding(V, D)
        self.embed_scale = None
        self.embedding_norm = None


def test_structural_features_offsets_and_body():
    ids, cid, _ = _row()
    f, body = structural_features(ids, cid, doc_start_id=DS, doc_end_id=DE)
    assert f.shape == (1, ids.shape[1], N_STRUCT)
    assert int(body.sum()) == 10 + 40 + 3
    # first body token of doc 0 (position 4): start one-hot 0 and end one-hot 9 (10 body tokens)
    assert f[0, 4, 1 + 0] == 1 and f[0, 4, 1 + 27 + 9] == 1
    # markers and free tokens carry no features
    assert f[0, 3].abs().sum() == 0 and f[0, 0].abs().sum() == 0
    # doc 1 offset 20 falls in the first log2 bucket (16..31)
    p = 3 + 12 + 1 + 20
    assert ids[0, p] == 2020 and f[0, p, 1 + 16 + 0] == 1


def test_query_mask_excludes_answer():
    ids, cid, labels = _row()
    qm = query_mask(cid, labels)
    assert ids[0][qm[0]].tolist() == [7, QTOK, 9]


def _router(**kw):
    cfg = OnlineRouterConfig(**{"keep_frac": 0.2, "explore_frac": 0.0, "samples_per_row": 1000, **kw})
    return OnlineTokenRouter(cfg, doc_start_id=DS, doc_end_id=DE, eos_id=EOS, d_model=16)


def test_warmup_keeps_everything_then_learns_lexical_match():
    """Step 1 is keep-all; after a fit on targets that equal the in-query flag, the router keeps the
    query-matching token first (a learned rule, no heuristic init)."""
    torch.manual_seed(0)
    m = _Emb()
    ids, cid, labels = _row(B=2)
    r = _router(use_hidden=False, doc_share=0.0)
    r.begin_step(1, model=m)
    keep = r.select(m, ids, cid, labels)
    body = (cid >= 0) & (ids != DS) & (ids != DE)
    assert torch.equal(keep, body)
    # synthetic backward: target = log(1 + 9 * in_query)
    e = r._current
    e["ok"] = torch.ones_like(e["rows"], dtype=torch.bool)
    tgt = torch.log1p(9.0 * (ids[e["rows"], e["pos"]] == QTOK).float())
    e["target"] = tgt
    r._pending.append(e)
    r._current = None
    r.begin_step(2, model=m)
    assert r.w is not None and not r.dense_now()
    keep2 = r.select(m, ids, cid, labels)
    assert keep2[(ids == QTOK) & body].all()
    n_body = int(body[0].sum())
    assert int(keep2[0].sum()) == math.ceil(0.2 * n_body)
    assert m._flop_extra_actual > 0  # scoring + fit were charged


def test_exploration_share_and_dense_every():
    torch.manual_seed(0)
    m = _Emb()
    ids, cid, labels = _row()
    r = _router(explore_frac=0.5, dense_every=3, doc_share=0.0)
    r.w = torch.zeros(r.n_features)
    r.step = 2
    keep = r.select(m, ids, cid, labels)
    n = math.ceil(0.2 * 53)
    assert int(keep.sum()) == n
    assert int(r._current["explore"].sum()) == round(0.5 * n)
    r.step = 3
    assert r.dense_now()


def test_target_hook_maps_compacted_columns():
    """Targets are read at the compacted embedding output and mapped back via position_ids."""
    torch.manual_seed(0)
    m = _Emb()
    ids, cid, labels = _row()
    r = _router(use_hidden=False)
    r.step = 1
    keep = r.select(m, ids, cid, labels)  # warmup: all body tokens
    # compact: drop the first 3 prompt tokens only (positions shift by 3)
    pos = torch.arange(3, ids.shape[1])[None]
    h = torch.randn(1, pos.shape[1], 16, requires_grad=True)
    out = r.attach_target_hook(m, h, pos)
    wts = torch.arange(pos.shape[1], dtype=torch.float32)[None, :, None]
    (out * wts).sum().backward()
    e = r._pending[-1]
    assert e["ok"].all()
    # grad at compacted column c is wts[c] * ones(16): norm = c * 4
    exp = torch.log((e["pos"] - 3).float() * 4.0 + 1e-12)
    assert torch.allclose(e["target"], exp, atol=1e-4)


def test_ridge_recovers_hidden_direction():
    torch.manual_seed(0)
    m = _Emb()
    ids, cid, labels = _row(B=2)
    r = _router(ridge=1e-3, decay=1.0)
    r.step = 1
    r.select(m, ids, cid, labels)
    e = r._current
    w_true = torch.randn(16)
    h = m.embeddings(ids).detach()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    e["target"] = (h[e["rows"], e["pos"]] @ w_true)
    e["ok"] = torch.ones_like(e["rows"], dtype=torch.bool)
    r._pending.append(e)
    r._current = None
    r.begin_step(2, model=m)
    hd = r.w[r.n_scalar : r.n_scalar + 16]
    cos = torch.nn.functional.cosine_similarity(hd, w_true, dim=0)
    assert cos > 0.9, cos


def test_model_training_step_with_online_rule_cpu():
    """End to end on a tiny CPU model: warmup step keeps every body token, the hook captures targets,
    the refit routes the next step at the budget, and FLOP overhead is recorded."""
    from olmo_core.config import DType
    from olmo_core.nn.transformer import TransformerConfig

    torch.manual_seed(0)
    V = 9100
    cfg = TransformerConfig.olmo2_190M(vocab_size=V, n_layers=2, fused_ops=False, dtype=DType.float32)
    model = cfg.build(init_device="cpu")
    D = int(model.embeddings.weight.shape[1])
    router = OnlineTokenRouter(
        OnlineRouterConfig(keep_frac=0.25, explore_frac=0.2, samples_per_row=64),
        doc_start_id=DS, doc_end_id=DE, eos_id=EOS, d_model=D,
    )
    model.enable_pooled_soft_tokens(
        DS, DE, EOS, placeholder_id=PH, keep_prob=0.0, keep_seed=0, detach_soft_kv=True,
        drop_slots=True, keep_token_rule="online", keep_token_k=0.25, online_router=router,
    )
    model.train()
    ids, cid, labels = _row(B=2)
    for step in (1, 2):
        router.begin_step(step, model=model)
        out = model(ids, labels=labels)
        loss = out.loss
        assert torch.isfinite(loss)
        loss.backward()
        assert router._pending and router._pending[-1]["target"] is not None
        if step == 1:
            assert router.dense_now()
            comp = model._soft_token_compaction
            # all 53 body tokens + 6 markers + 9 free kept: only the right pad goes
            assert comp["tokens_out"] == comp["tokens_in"]
        model._soft_token_compaction = None
    router.begin_step(3, model=model)
    assert router.w is not None
    cb = model._compact_pooled_soft_tokens(ids, labels, -100)[0]
    # 25% of 53 body tokens (14; exploration tokens can overfill a short document's share by <= 3)
    # + 6 markers + 9 free (prompt, query, answer, EOS)
    n_kept = int(cb.row_lens[0]) - 6 - 9
    assert 14 <= n_kept <= 14 + 3, cb.row_lens
    assert model._flop_extra_actual > 0
    mets = router.metrics()
    assert "online_router/keep_frac" in mets and "online_router/rank_corr_kept" in mets


def test_online_rule_requires_router():
    from olmo_core.config import DType
    from olmo_core.exceptions import OLMoConfigurationError
    from olmo_core.nn.transformer import TransformerConfig

    cfg = TransformerConfig.olmo2_190M(vocab_size=9100, n_layers=1, fused_ops=False, dtype=DType.float32)
    model = cfg.build(init_device="cpu")
    with pytest.raises(OLMoConfigurationError):
        model.enable_pooled_soft_tokens(
            DS, DE, EOS, placeholder_id=PH, drop_slots=True, keep_token_rule="online", keep_token_k=0.3
        )


def _gpu_hybrid():
    """3 blocks: GDN, attention, GDN (the Qwen3.5 hybrid's shape in miniature)."""
    from olmo_core.nn.attention import AttentionConfig, GatedDeltaNetConfig
    from olmo_core.nn.feed_forward import FeedForwardConfig
    from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
    from olmo_core.nn.lm_head import LMHeadConfig
    from olmo_core.nn.rope import RoPEConfig, RoPEType
    from olmo_core.nn.transformer import TransformerBlockConfig, TransformerBlockType, TransformerConfig

    torch.manual_seed(0)
    ln = LayerNormConfig(name=LayerNormType.rms, bias=False)
    ff = FeedForwardConfig(hidden_size=512, bias=False)
    gdn = TransformerBlockConfig(name=TransformerBlockType.reordered_norm, sequence_mixer=GatedDeltaNetConfig(n_heads=8),
                                 layer_norm=ln, feed_forward=ff)
    attn = TransformerBlockConfig(name=TransformerBlockType.reordered_norm,
                                  attention=AttentionConfig(n_heads=8, rope=RoPEConfig(name=RoPEType.default)),
                                  layer_norm=ln, feed_forward=ff)
    cfg = TransformerConfig(d_model=256, vocab_size=9100, n_layers=3, block=gdn, block_overrides={1: attn},
                            lm_head=LMHeadConfig(layer_norm=ln, bias=False))
    model = cfg.build(init_device="cuda")
    model.init_weights()
    return model


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["input_prefix", "mid_stack"])
def test_online_rule_on_gdn_hybrid_gpu(mode):
    """GPU, GDN + attention: features from a no-grad prefix pass through blocks [0, 2) (GDN and
    attention on the FULL row), compaction to the budget, targets at the compacted embedding output,
    a refit, and the prefix pass charged as forward-only FLOPs."""
    from olmo_core.testing import requires_gpu  # noqa: F401
    from olmo_core.testing.utils import requires_fla  # noqa: F401

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    pytest.importorskip("fla")
    model = _gpu_hybrid()
    mid = mode == "mid_stack"
    router = OnlineTokenRouter(
        OnlineRouterConfig(keep_frac=0.3, explore_frac=0.2, samples_per_row=64,
                           feature_layer=0 if mid else 2, drop_layer=2 if mid else 0),
        doc_start_id=DS, doc_end_id=DE, eos_id=EOS, d_model=256,
    )
    model.enable_pooled_soft_tokens(
        DS, DE, EOS, placeholder_id=PH, keep_prob=0.0, keep_seed=0, detach_soft_kv=True,
        drop_slots=True, keep_token_rule="online", keep_token_k=0.3, online_router=router,
    )
    model.train()
    ids, cid, labels = _row(B=2)
    ids, labels = ids.cuda(), labels.cuda()
    prefix_flops = sum(int(b.num_flops_per_token(ids.shape[1])) for b in list(model.blocks.values())[:2]) / 3.0
    for step in (1, 2, 3):
        router.begin_step(step, model=model)
        model._flop_extra_actual = 0.0
        out = model(ids, labels=labels)
        assert torch.isfinite(out.loss)
        out.loss.backward()
        e = router._pending[-1]
        assert e["target"] is not None and e["ok"].all() and torch.isfinite(e["target"]).all()
        comp = model._soft_token_compaction
        if mid:
            assert comp["tokens_out"] == comp["tokens_in"]
            if step >= 2:
                assert model._flop_extra_actual < 0  # blocks [2, 3) + head ran on the kept row only
        else:
            # the prefix pass ran on the full (trimmed) row and was charged
            assert model._flop_extra_actual >= prefix_flops * ids.numel() * 0.99
            if step >= 2:
                assert comp["tokens_out"] < comp["tokens_in"]
        model._soft_token_compaction = None
    assert router.w is not None and torch.isfinite(router.w).all()


def test_doc_share_spreads_budget_over_documents():
    """doc_share=1: every document keeps ceil(keep_frac * |body|) of its own top tokens, even when the
    scorer prefers one document; doc_share=0 lets the scorer take the whole budget from it."""
    m = _Emb()
    ids, cid, labels = _row()
    for share, expect in ((1.0, {0: 2, 1: 8, 2: 1}), (0.0, {0: 10, 1: 1, 2: 0})):
        r = _router(use_hidden=False, doc_share=share, keep_frac=0.2)
        r.w = torch.zeros(r.n_features)
        r.w[r.feature_names().index("start_off0")] = 0.0
        r.step = 2
        # scorer: doc 0 tokens high (via the doc-index feature, negative weight = earlier docs first)
        r.w[r.feature_names().index("doc_rel_idx")] = -5.0
        keep = r.select(m, ids, cid, labels)
        got = {d: int((keep[0] & (cid[0] == d)).sum()) for d in range(3)}
        assert sum(got.values()) == math.ceil(0.2 * 53) or share == 1.0
        if share == 1.0:
            assert got == expect, got
        else:
            assert got[0] == 10 and got[2] == 0, got


def test_mid_stack_drop_cpu():
    """drop_layer=1 on a 2-block model: the input row is kept whole, block 0 runs on every token, the
    router keeps its budget at block 1, labels follow the gather, targets come from block 1's input,
    and the FLOP adjustment is negative (block 1 + head ran on fewer tokens)."""
    from olmo_core.config import DType
    from olmo_core.nn.transformer import TransformerConfig

    torch.manual_seed(0)
    cfg = TransformerConfig.olmo2_190M(vocab_size=9100, n_layers=2, fused_ops=False, dtype=DType.float32)
    model = cfg.build(init_device="cpu")
    D = int(model.embeddings.weight.shape[1])
    router = OnlineTokenRouter(
        OnlineRouterConfig(keep_frac=0.25, explore_frac=0.2, samples_per_row=64, drop_layer=1),
        doc_start_id=DS, doc_end_id=DE, eos_id=EOS, d_model=D,
    )
    model.enable_pooled_soft_tokens(
        DS, DE, EOS, placeholder_id=PH, keep_prob=0.0, keep_seed=0, detach_soft_kv=True,
        drop_slots=True, keep_token_rule="online", keep_token_k=0.25, online_router=router,
    )
    model.train()
    ids, cid, labels = _row(B=2)
    seen = {}
    h1 = model.blocks["1"].register_forward_pre_hook(lambda m, a: seen.__setitem__("w", a[0].shape[1]))
    for step in (1, 2):
        router.begin_step(step, model=model)
        model._flop_extra_actual = 0.0
        out = model(ids, labels=labels)
        assert torch.isfinite(out.loss)
        out.loss.backward()
        assert router._pending[-1]["target"] is not None and router._pending[-1]["ok"].all()
        comp = model._soft_token_compaction
        assert comp["tokens_out"] == comp["tokens_in"]  # nothing dropped at the input
        model._soft_token_compaction = None
        if step == 1:
            assert seen["w"] == ids.shape[1]  # warmup: every token through block 1
        else:
            # routed: block 1 saw 25% of the body + markers + free tokens
            assert seen["w"] < ids.shape[1] and seen["w"] >= 14 + 6 + 9
            assert model._flop_extra_actual < 0
    h1.remove()
    # the answer CE is unchanged by the gather when everything is kept (dense step): same loss as full
    router.cfg.dense_every = 1
    router.begin_step(3, model=model)
    model.eval()
    ref = model(ids, labels=labels).loss
    model.train()
    out = model(ids, labels=labels)
    assert torch.allclose(out.loss, ref, atol=1e-5), (out.loss, ref)


def test_fidelity_mass_logged_on_dense_steps():
    """On a dense (full-context) step with fitted weights, the router logs the share of full-pass
    gradient mass its top keep_frac would keep, plus the oracle share."""
    torch.manual_seed(0)
    m = _Emb()
    ids, cid, labels = _row(B=2)
    r = _router(use_hidden=False, doc_share=0.0, dense_every=2)
    r.w = torch.zeros(r.n_features)
    r.w[r.feature_names().index("in_query")] = 1.0
    r.step = 2
    assert r.dense_now()
    r.select(m, ids, cid, labels)
    e = r._current
    e["ok"] = torch.ones_like(e["rows"], dtype=torch.bool)
    e["target"] = torch.log1p(9.0 * (ids[e["rows"], e["pos"]] == QTOK).float())
    r._pending.append(e)
    r._current = None
    r.begin_step(3, model=m)
    met = r.metrics()
    assert met["online_router/fid_mass_router"] == pytest.approx(met["online_router/fid_mass_oracle"])
    assert met["online_router/fid_mass_router"] > met["online_router/fid_mass_random"]
    assert met["online_router/rank_corr_fullctx"] > 0.5  # held-out correlation on the full-context step


def test_random_control_keeps_doc_share_without_router():
    """explore_frac >= 1: per-document share of uniformly random tokens, whatever the weights say."""
    m = _Emb()
    ids, cid, labels = _row()
    r = _router(use_hidden=False, doc_share=1.0, keep_frac=0.2, explore_frac=1.0)
    r.w = torch.zeros(r.n_features)
    r.w[r.feature_names().index("doc_rel_idx")] = -50.0
    r.step = 2
    keep = r.select(m, ids, cid, labels)
    got = {d: int((keep[0] & (cid[0] == d)).sum()) for d in range(3)}
    assert got == {0: 2, 1: 8, 2: 1}, got


def test_first_last_control_keeps_document_edges():
    m = _Emb()
    ids, cid, labels = _row()
    r = _router(use_hidden=False, doc_share=1.0, keep_frac=0.2, score="first_last")
    r.w = torch.zeros(r.n_features)
    r.step = 2
    keep = r.select(m, ids, cid, labels)
    # doc 1 (40 body tokens, ids 2000..2039) keeps 8: its first 4 and last 4
    kept = sorted(int(t) for t in ids[0][keep[0] & (cid[0] == 1)])
    assert kept == [2000, 2001, 2002, 2003, 2036, 2037, 2038, 2039], kept


@pytest.mark.parametrize("target,score", [("gxh", "router"), ("dh", "router"), ("mix", "router"), ("gnorm", "dh")])
def test_mid_stack_targets_cpu(target, score):
    """Every target trains end to end in mid-stack mode, records the gradient norm too, and refits."""
    from olmo_core.config import DType
    from olmo_core.nn.transformer import TransformerConfig

    torch.manual_seed(0)
    cfg = TransformerConfig.olmo2_190M(vocab_size=9100, n_layers=2, fused_ops=False, dtype=DType.float32)
    model = cfg.build(init_device="cpu")
    D = int(model.embeddings.weight.shape[1])
    router = OnlineTokenRouter(
        OnlineRouterConfig(keep_frac=0.25, explore_frac=0.2, samples_per_row=64, drop_layer=1, target=target,
                           dense_every=2, score=score),
        doc_start_id=DS, doc_end_id=DE, eos_id=EOS, d_model=D,
    )
    model.enable_pooled_soft_tokens(
        DS, DE, EOS, placeholder_id=PH, keep_prob=0.0, keep_seed=0, detach_soft_kv=True,
        drop_slots=True, keep_token_rule="online", keep_token_k=0.25, online_router=router,
    )
    model.train()
    ids, cid, labels = _row(B=2)
    for step in (1, 2, 3):
        router.begin_step(step, model=model)
        out = model(ids, labels=labels)
        out.loss.backward()
        e = router._pending[-1]
        assert e["target"] is not None and torch.isfinite(e["target"]).all()
        assert e["gn"] is not None and torch.isfinite(e["gn"]).all()
        if target == "mix":
            assert (e["target"] >= 0).all() and (e["target"] <= 1).all()
        if score == "dh" and step == 3:
            # fit-free rule: no exploration, and every document keeps its own top tokens by the residual change
            assert not bool(e["explore"].any())
    router.begin_step(4, model=model)
    assert router.w is not None and torch.isfinite(router.w).all()
    assert "online_router/fid_mass_router" in router.metrics()


def test_dh_target_needs_mid_stack():
    with pytest.raises(ValueError):
        OnlineRouterConfig(target="dh").validate()


def test_dh_score_needs_mid_stack():
    with pytest.raises(ValueError):
        OnlineRouterConfig(score="dh").validate()


def test_mid_stack_composes_with_layer_drop_cpu():
    """Mid-stack drop + whole-block skip of a later block: the skipped block's forward is never called, and the
    FLOP adjustment charges exactly blocks [0, L) at the full length + the blocks that ran past L at the kept length."""
    from olmo_core.config import DType
    from olmo_core.nn.transformer import TransformerConfig

    torch.manual_seed(0)
    cfg = TransformerConfig.olmo2_190M(vocab_size=9100, n_layers=4, fused_ops=False, dtype=DType.float32)
    model = cfg.build(init_device="cpu")
    D = int(model.embeddings.weight.shape[1])
    router = OnlineTokenRouter(
        OnlineRouterConfig(keep_frac=0.25, explore_frac=0.2, samples_per_row=64, drop_layer=1, dense_every=0),
        doc_start_id=DS, doc_end_id=DE, eos_id=EOS, d_model=D,
    )
    model.enable_pooled_soft_tokens(
        DS, DE, EOS, placeholder_id=PH, keep_prob=0.0, keep_seed=0, detach_soft_kv=True,
        drop_slots=True, keep_token_rule="online", keep_token_k=0.25, online_router=router,
    )
    model.enable_layer_drop(fixed={3})
    calls = []
    model.blocks["3"].register_forward_hook(lambda *a: calls.append(1))
    flops = []
    orig = router._flop
    router._flop = lambda m, v: (flops.append(float(v)), orig(m, v))
    model.train()
    ids, cid, labels = _row(B=1)
    for step in (1, 2):
        router.begin_step(step, model=model)
        flops.clear()
        router.metrics()  # reset the per-forward stats: mid_row_frac below is step 2's alone
        out = model(ids, labels=labels)
        out.loss.backward()
    assert not calls, "a layer-drop-skipped block ran"
    st = router._pending[-1]
    real = int((ids != PH).sum())
    kept = int(round(router.metrics().get("online_router/mid_row_frac", 0.0) * real))
    blocks = list(model.blocks.values())
    head = int(model.lm_head.num_flops_per_token(0))
    f = lambda b, n: int(b.num_flops_per_token(int(n)))
    expect = (real * f(blocks[0], real) + kept * (f(blocks[1], kept) + f(blocks[2], kept) + head)
              - real * (f(blocks[0], real) + f(blocks[1], real) + f(blocks[2], real) + head))
    assert any(abs(v - expect) < 1e-6 * max(1.0, abs(expect)) for v in flops), (expect, flops)
    assert st["target"] is not None


def test_mid_stack_drop_layer_cannot_be_skipped():
    from olmo_core.config import DType
    from olmo_core.nn.transformer import TransformerConfig

    cfg = TransformerConfig.olmo2_190M(vocab_size=9100, n_layers=3, fused_ops=False, dtype=DType.float32)
    model = cfg.build(init_device="cpu")
    D = int(model.embeddings.weight.shape[1])
    router = OnlineTokenRouter(OnlineRouterConfig(drop_layer=1, dense_every=0), doc_start_id=DS, doc_end_id=DE,
                               eos_id=EOS, d_model=D)
    model.enable_pooled_soft_tokens(
        DS, DE, EOS, placeholder_id=PH, keep_prob=0.0, keep_seed=0, detach_soft_kv=True,
        drop_slots=True, keep_token_rule="online", keep_token_k=0.3, online_router=router,
    )
    model.enable_layer_drop(fixed={1})
    model.train()
    ids, cid, labels = _row(B=1)
    router.begin_step(1, model=model)
    with pytest.raises(RuntimeError):
        model(ids, labels=labels)


@pytest.mark.parametrize("doc_share", [1.0, 0.0])
def test_keep_first_always_keeps_document_headers(doc_share):
    """keep_first: every document's first K body tokens are kept whatever the scores, and (per-document share)
    they count against their document's share; with doc_share 0 the rest of the budget goes anywhere by score."""
    m = _Emb()
    ids, cid, labels = _row()
    r = _router(use_hidden=False, doc_share=doc_share, keep_frac=0.2, score="random", keep_first=3)
    r.w = torch.zeros(r.n_features)
    r.step = 2
    keep = r.select(m, ids, cid, labels)
    for d, base, n in ((0, 1000, 10), (1, 2000, 40), (2, 3000, 3)):
        kept = {int(t) for t in ids[0][keep[0] & (cid[0] == d)]}
        assert {base, base + 1, base + 2} <= kept, (d, sorted(kept))
    if doc_share == 1.0:
        # doc 1's share is ceil(0.2 * 40) = 8, of which 3 are its forced header tokens
        assert int((keep[0] & (cid[0] == 1)).sum()) == 8
    assert "online_router/forced_frac" in r.metrics()



def _mid_model(n_layers, **cfg):
    from olmo_core.config import DType
    from olmo_core.nn.transformer import TransformerConfig

    torch.manual_seed(0)
    model = TransformerConfig.olmo2_190M(vocab_size=9100, n_layers=n_layers, fused_ops=False, dtype=DType.float32).build(
        init_device="cpu")
    D = int(model.embeddings.weight.shape[1])
    router = OnlineTokenRouter(OnlineRouterConfig(**{"keep_frac": 0.25, "explore_frac": 0.2, "samples_per_row": 64,
                                                     **cfg}), doc_start_id=DS, doc_end_id=DE, eos_id=EOS, d_model=D)
    model.enable_pooled_soft_tokens(
        DS, DE, EOS, placeholder_id=PH, keep_prob=0.0, keep_seed=0, detach_soft_kv=True,
        drop_slots=True, keep_token_rule="online", keep_token_k=0.25, online_router=router,
    )
    model.train()
    return model, router


def test_early_nograd_blocks_learn_only_on_dense_steps_cpu():
    """early_nograd: on a routed step blocks [0, L) get no gradient, blocks >= L do, the target hook at the split still
    fires, and the FLOP adjustment charges [0, L) forward-only; on a dense step every block gets a gradient."""
    model, router = _mid_model(4, drop_layer=2, early_nograd=True, dense_every=0)
    blocks = list(model.blocks.values())
    flops = []
    orig = router._flop
    router._flop = lambda m, v: (flops.append(float(v)), orig(m, v))
    ids, cid, labels = _row(B=1)

    def step(n):
        model.zero_grad(set_to_none=True)
        router.begin_step(n, model=model)
        flops.clear()
        router.metrics()
        model(ids, labels=labels).loss.backward()

    step(1)  # warmup: dense
    assert all(p.grad is not None for p in blocks[0].parameters())
    step(2)  # routed
    assert torch.is_grad_enabled()
    assert all(p.grad is None for b in blocks[:2] for p in b.parameters())
    assert all(p.grad is not None for b in blocks[2:] for p in b.parameters())
    e = router._pending[-1]
    assert e["target"] is not None and torch.isfinite(e["target"]).all()
    real = int((ids != PH).sum())
    m = router.metrics()
    kept = int(round(m["online_router/mid_row_frac"] * real))
    assert m["online_router/early_nograd"] == 1.0
    head = int(model.lm_head.num_flops_per_token(0))
    f = lambda b, n: int(b.num_flops_per_token(int(n)))
    expect = (real * (f(blocks[0], real) + f(blocks[1], real)) / 3.0 + kept * (f(blocks[2], kept) + f(blocks[3], kept) + head)
              - real * (sum(f(b, real) for b in blocks) + head))
    assert any(abs(v - expect) < 1e-6 * max(1.0, abs(expect)) for v in flops), (expect, flops)


def test_dense_tail_frac():
    r = _router(dense_tail_frac=0.2, dense_every=0)
    r.w = torch.zeros(r.n_features)
    with pytest.raises(RuntimeError):
        r.step = 5
        r.dense_now()
    r.total_steps = 10
    dense = []
    for s_ in range(2, 11):
        r.step = s_
        dense.append(r.dense_now())
    assert dense == [False] * 7 + [True, True], dense


def test_early_nograd_needs_mid_stack():
    with pytest.raises(ValueError):
        OnlineRouterConfig(early_nograd=True).validate()



def test_pyramid_stages_cpu():
    """stages "1:0.5,2:0.25": two routers in a chain, each fit from its own layer's targets; each stage's FLOP adjustment
    is exact, the labels are gathered through both stages, and eval runs every token."""
    model, router = _mid_model(4, stages="1:0.5,2:0.25", dense_every=0)
    chain = router.chain()
    assert [r.cfg.drop_layer for r in chain] == [1, 2] and chain[1].cfg.keep_frac == pytest.approx(0.5)
    blocks = list(model.blocks.values())
    flops = []
    for r in chain:
        orig = r._flop
        r._flop = (lambda o: (lambda m, v: (flops.append(float(v)), o(m, v))))(orig)
    ids, cid, labels = _row(B=1)
    for n in (1, 2, 3):
        model.zero_grad(set_to_none=True)
        for r in chain:
            r.begin_step(n, model=model)
            r.metrics()
        flops.clear()
        out = model(ids, labels=labels)
        assert torch.isfinite(out.loss)
        out.loss.backward()
    assert all(r.w is not None for r in chain)
    for r in chain:
        e = r._pending[-1]
        assert e["target"] is not None and torch.isfinite(e["target"]).all()
    real = int((ids != PH).sum())
    m0, m1 = chain[0].metrics(), chain[1].metrics()
    k1 = int(round(m0["online_router/mid_row_frac"] * real))
    k2 = int(round(m1["online_router/s1_mid_row_frac"] * k1))
    assert k2 < k1 < real
    head = int(model.lm_head.num_flops_per_token(0))
    f = lambda b, n: int(b.num_flops_per_token(int(n)))
    e1 = (real * f(blocks[0], real) + k1 * (sum(f(b, k1) for b in blocks[1:]) + head)
          - real * (sum(f(b, real) for b in blocks) + head))
    e2 = k2 * (f(blocks[2], k2) + f(blocks[3], k2) + head) - k1 * (f(blocks[2], k1) + f(blocks[3], k1) + head)
    for e in (e1, e2):
        assert any(abs(v - e) < 1e-6 * max(1.0, abs(e)) for v in flops), (e, flops)
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(ids, labels=labels).loss)


def test_pyramid_stages_validate():
    for bad in ("7:0.1,3:0.5", "3:0.1,7:0.5", "0:0.5"):
        with pytest.raises(ValueError):
            OnlineRouterConfig(stages=bad).validate()


def test_marker_free_offsets_on_positional_blocks():
    """CPT rows (positional blocks, no markers): offsets count from each block's own first / last token."""
    from olmo_core.nn.attention.chunked_mask import build_positional_block_chunk_ids

    ids = torch.arange(100, 100 + 20)[None]  # 20 tokens, no leading EOS -> blocks of 8, 8, 4
    cid = build_positional_block_chunk_ids(ids, block_size=8, eos_id=EOS)
    offs = {}
    f, body = structural_features(ids, cid, doc_start_id=-1, doc_end_id=-1, offsets=offs, marker_free=True)
    assert int(body.sum()) == 20
    assert offs["j"][0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7] * 2 + [0, 1, 2, 3]
    assert offs["r"][0].tolist() == [7, 6, 5, 4, 3, 2, 1, 0] * 2 + [3, 2, 1, 0]
    # the marker-assuming default would merge the first two tokens at offset 0
    offs2 = {}
    structural_features(ids, cid, doc_start_id=-1, doc_end_id=-1, offsets=offs2)
    assert offs2["j"][0, :2].tolist() == [0, 0]


def test_mid_stack_drop_on_positional_blocks_cpu():
    """olmo3_cptdrop CPT geometry: marker-free rows in positional-block mode (block 8), every token has an
    LM label, no query. The mid-stack router keeps 25% of each block for block 1; dropped tokens get no
    loss (labels follow the gather); a dense step reproduces the full-row loss."""
    from olmo_core.config import DType
    from olmo_core.nn.transformer import TransformerConfig

    torch.manual_seed(0)
    cfg = TransformerConfig.olmo2_190M(vocab_size=9100, n_layers=2, fused_ops=False, dtype=DType.float32)
    model = cfg.build(init_device="cpu")
    D = int(model.embeddings.weight.shape[1])
    router = OnlineTokenRouter(
        OnlineRouterConfig(keep_frac=0.25, explore_frac=0.2, samples_per_row=64, drop_layer=1, use_query=False),
        doc_start_id=DS, doc_end_id=DE, eos_id=EOS, d_model=D,
    )
    model.enable_pooled_soft_tokens(
        DS, DE, EOS, placeholder_id=PH, keep_prob=0.0, keep_seed=0, detach_soft_kv=True,
        drop_slots=True, keep_token_rule="online", keep_token_k=0.25, online_router=router,
        positional_block_size=8,
    )
    model.train()
    T = 64
    ids = torch.randint(10, 9000, (2, T))
    ids[:, -1] = EOS  # EOS-last row, as the *_ee shards
    labels = torch.full_like(ids, -100)
    labels[:, :-1] = ids[:, 1:]
    seen = {}
    h1 = model.blocks["1"].register_forward_pre_hook(lambda m, a: seen.__setitem__("w", a[0].shape[1]))
    for step in (1, 2, 3):
        router.begin_step(step, model=model)
        model._flop_extra_actual = 0.0
        out = model(ids, labels=labels)
        assert torch.isfinite(out.loss)
        out.loss.backward()
        assert router._marker_free
        model._soft_token_compaction = None
        if step == 1:
            assert seen["w"] == T
        else:
            # 8 blocks x ceil(0.25 * 8) = 16 kept tokens per row reach block 1
            assert seen["w"] == 16, seen["w"]
            assert model._flop_extra_actual < 0
    h1.remove()
    router.cfg.dense_every = 1
    router.begin_step(4, model=model)
    model.eval()
    ref = model(ids, labels=labels).loss
    model.train()
    out = model(ids, labels=labels)
    assert torch.allclose(out.loss, ref, atol=1e-5), (out.loss, ref)



def test_merge_dropped_summary_tokens_cpu():
    """merge_dropped: past the split each document gets one summary slot = the mean of its dropped tokens' hidden
    states, placed after its last dropped token, with no label; the step trains and the kept width includes them."""
    model, router = _mid_model(3, drop_layer=1, dense_every=0, merge_dropped=True, explore_frac=0.0)
    seen = {}
    model.blocks["0"].register_forward_hook(lambda m, a, out: seen.__setitem__("h0", out.detach()))
    model.blocks["1"].register_forward_pre_hook(lambda m, a: seen.__setitem__("h1", a[0].detach()))
    ids, cid, labels = _row(B=1)
    for n in (1, 2):
        model.zero_grad(set_to_none=True)
        router.begin_step(n, model=model)
        router.metrics()
        out = model(ids, labels=labels)
        assert torch.isfinite(out.loss)
        out.loss.backward()
    m = router.metrics()
    assert m["online_router/merge_tokens"] == 3.0  # three documents, each with dropped tokens
    order, valid, n_extra = router._mid_gather if router._mid_gather is not None else (None, None, 0)
    h0, h1 = seen["h0"][0], seen["h1"][0]
    # doc 1 (40 body tokens): its summary slot equals the mean of its dropped tokens at the split input
    keep_cols = set()
    for v in h1:
        hit = (h0 - v[None]).abs().max(dim=-1).values < 1e-5
        keep_cols.update(torch.nonzero(hit).flatten().tolist())
    d1 = [c for c in range(ids.shape[1]) if int(cid[0, c]) == 1 and int(ids[0, c]) >= 2000 and c not in keep_cols]
    target = h0[d1].mean(0)
    assert ((h1 - target[None]).abs().max(dim=-1).values < 1e-4).any(), "no summary slot equals doc 1's dropped mean"


def test_merge_dropped_validate():
    with pytest.raises(ValueError):
        OnlineRouterConfig(merge_dropped=True).validate()
    with pytest.raises(ValueError):
        OnlineRouterConfig(merge_dropped=True, drop_layer=1, stages="1:0.5,2:0.25").validate()


def test_freeze_early_cpu():
    """freeze_early: embeddings + blocks [0, L) never get a gradient (warmup / dense steps included), later blocks do,
    and even the dense warmup step charges blocks [0, L) forward-only."""
    from olmo_core.nn.online_token_router import freeze_early_blocks

    model, router = _mid_model(3, drop_layer=1, dense_every=0, freeze_early=True)
    assert freeze_early_blocks(model, 1) > 0
    blocks = list(model.blocks.values())
    flops = []
    orig = router._flop
    router._flop = lambda m, v: (flops.append(float(v)), orig(m, v))
    ids, cid, labels = _row(B=1)
    real = int((ids != PH).sum())
    head = int(model.lm_head.num_flops_per_token(0))
    f = lambda b, n: int(b.num_flops_per_token(int(n)))
    for n in (1, 2):
        model.zero_grad(set_to_none=True)
        router.begin_step(n, model=model)
        router.metrics()
        flops.clear()
        out = model(ids, labels=labels)
        assert torch.isfinite(out.loss)
        out.loss.backward()
        assert all(p.grad is None for p in model.embeddings.parameters())
        assert all(p.grad is None for p in blocks[0].parameters())
        assert all(p.grad is not None for b in blocks[1:] for p in b.parameters())
        assert router._pending[-1]["target"] is not None
        if n == 1:  # dense warmup: every token kept, blocks [0, 1) still forward-only
            e = real * f(blocks[0], real) / 3.0 - real * f(blocks[0], real)
            assert any(abs(v - e) < 1e-6 * max(1.0, abs(e)) for v in flops), (e, flops)



def test_min_per_doc_floor():
    """min_per_doc: every document keeps at least K body tokens (all of a shorter one), by score, on top of its share."""
    m = _Emb()
    ids, cid, labels = _row()
    r = _router(use_hidden=False, doc_share=1.0, keep_frac=0.1, score="random", min_per_doc=5)
    r.w = torch.zeros(r.n_features)
    r.step = 2
    keep = r.select(m, ids, cid, labels)
    per_doc = [int((keep[0] & (cid[0] == d)).sum()) for d in range(3)]
    assert per_doc[0] == 5 and per_doc[1] >= 5 and per_doc[2] == 3, per_doc  # doc 2 has only 3 body tokens
    assert "online_router/floor_extra_frac" in r.metrics()



def test_freeze_and_release_frees_optimizer_state_cpu():
    """freeze_after: the early stack trains, then freeze_and_release drops its grads and AdamW state, and later steps
    give it no gradient."""
    from olmo_core.nn.online_token_router import freeze_and_release

    model, router = _mid_model(3, drop_layer=1, dense_every=0, freeze_after=1)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    ids, cid, labels = _row(B=1)
    router.begin_step(1, model=model)
    model(ids, labels=labels).loss.backward()
    opt.step()
    early = list(model.embeddings.parameters()) + list(list(model.blocks.values())[0].parameters())
    assert all(p in opt.state for p in early)
    assert freeze_and_release(model, 1, opt) > 0
    router.cfg.freeze_early = True
    assert not any(p in opt.state for p in early) and all(p.grad is None for p in early)
    opt.zero_grad(set_to_none=True)
    router.begin_step(2, model=model)
    model(ids, labels=labels).loss.backward()
    assert all(p.grad is None for p in early)
    assert all(p.grad is not None for p in list(model.blocks.values())[1].parameters())


def test_exit_loss_scores_every_label_once_cpu():
    """exit_loss_weight: on a fully-labelled row, the main LM head call plus the early-exit call together score every
    label exactly once (W = 1 = the full LM loss); the exit term is positive and its head FLOPs are charged."""
    model, router = _mid_model(3, drop_layer=1, dense_every=0, exit_loss_weight=1.0, explore_frac=0.0)
    ids, cid, _ = _row(B=1)
    labels = torch.full_like(ids, -100)
    labels[:, :-1] = ids[:, 1:]  # CPT-style: every position predicts the next token
    total = int((labels != -100).sum())
    seen, flops = [], []
    orig_fwd = model.lm_head.forward

    def spy(x, **kw):
        if kw.get("labels") is not None:
            seen.append(int((kw["labels"] != -100).sum()))
        return orig_fwd(x, **kw)

    model.lm_head.forward = spy
    orig_flop = router._flop
    router._flop = lambda m, v: (flops.append(float(v)), orig_flop(m, v))
    for n in (1, 2):
        seen.clear(); flops.clear()
        router.begin_step(n, model=model)
        router.metrics()
        out = model(ids, labels=labels, loss_reduction="sum")
        out.loss.backward()
    assert len(seen) == 2 and sum(seen) == total, (seen, total)  # routed step: main call + exit call
    m = router.metrics()
    n_drop = int(round(m["online_router/exit_tokens_frac"] * int((ids != PH).sum())))
    head = int(model.lm_head.num_flops_per_token(0))
    assert any(abs(v - head * n_drop) < 1e-6 * max(1.0, head * n_drop) for v in flops), (head * n_drop, flops)
    # W = 0: only the kept labels are scored
    router.cfg.exit_loss_weight = 0.0
    seen.clear()
    router.begin_step(3, model=model)
    model(ids, labels=labels, loss_reduction="sum").loss.backward()
    assert len(seen) == 1 and seen[0] < total


def test_exit_loss_validate():
    with pytest.raises(ValueError):
        OnlineRouterConfig(exit_loss_weight=1.0).validate()
    with pytest.raises(ValueError):
        OnlineRouterConfig(exit_loss_weight=1.0, stages="3:0.5,7:0.1").validate()



def test_dense_keep_frac_routes_periodic_steps():
    """dense_keep_frac: the periodic dense step keeps that share (not everything); warmup still keeps everything."""
    m = _Emb()
    ids, cid, labels = _row()
    r = _router(use_hidden=False, doc_share=1.0, keep_frac=0.1, score="random", dense_every=4, dense_keep_frac=0.5)
    r.step = 1
    assert bool(r.select(m, ids, cid, labels).sum() == r.token_features(m, ids, cid, labels, -100, hidden=False)["body"].sum())
    r.w = torch.zeros(r.n_features)
    n_body = int(r.token_features(m, ids, cid, labels, -100, hidden=False)["body"].sum())
    r.step = 4
    k4 = int(r.select(m, ids, cid, labels).sum())
    r.step = 5
    k5 = int(r.select(m, ids, cid, labels).sum())
    assert k5 < k4 < n_body, (k5, k4, n_body)
    assert abs(k4 / n_body - 0.5) < 0.1


def test_early_nograd_trains_early_blocks_on_dense_keep_frac_steps_cpu():
    """early_nograd + dense_keep_frac < 1: the periodic steps route at F but stay GRAD steps for blocks [0, L) (charged
    forward + backward); the other routed steps are forward-only there."""
    model, router = _mid_model(3, drop_layer=1, dense_every=2, early_nograd=True, dense_keep_frac=0.5, keep_frac=0.25)
    blocks = list(model.blocks.values())
    ids, cid, labels = _row(B=1)
    got = {}
    for n in (1, 2, 3):
        model.zero_grad(set_to_none=True)
        router.begin_step(n, model=model)
        router.metrics()
        model(ids, labels=labels).loss.backward()
        m = router.metrics()
        got[n] = (all(p.grad is not None for p in blocks[0].parameters()), m.get("online_router/mid_row_frac"),
                  m.get("online_router/early_nograd"))
    assert got[2][0] and got[2][2] == 0.0 and got[2][1] < 1.0, got[2]   # periodic step: routed at F, early blocks train
    assert not got[3][0] and got[3][2] == 1.0, got[3]                   # ordinary routed step: early blocks forward-only
    assert got[2][1] > got[3][1]                                        # F = 0.5 keeps more than keep_frac = 0.25


def test_parse_mod_layers():
    from olmo_core.nn.online_token_router import parse_mod_layers

    assert parse_mod_layers("8-31:2") == list(range(8, 31, 2))
    assert parse_mod_layers("8,12,16") == [8, 12, 16]
    assert parse_mod_layers("2-5:2,9") == [2, 4, 9]
    with pytest.raises(ValueError):
        parse_mod_layers("0,2")


def test_mixture_of_depths_cpu():
    """mod_layers '2-5:2' on a 6-block model: drop_layer becomes 2; on a routed step blocks 2 and 4 run on the kept
    tokens only while 0, 1, 3, 5 and the head see the full row (no label dropped); every block trains; the targets at
    block 2 sample every body token (none censored); the FLOP adjustment charges blocks 2 and 4 at the kept length;
    and a dense step reproduces the full model's loss."""
    model, router = _mid_model(6, mod_layers="2-5:2", dense_every=0)
    assert router.cfg.drop_layer == 2 and router.mod_layers == [2, 4]
    blocks = list(model.blocks.values())
    widths = {}
    hooks = [b.register_forward_pre_hook(lambda m, a, i=i: widths.__setitem__(i, a[0].shape[1]))
             for i, b in enumerate(blocks)]
    flops = []
    orig = router._flop
    router._flop = lambda m, v: (flops.append(float(v)), orig(m, v))
    ids, cid, labels = _row(B=1)
    T = ids.shape[1]
    body = int(((cid >= 0) & (ids != DS) & (ids != DE)).sum())

    def step(n):
        model.zero_grad(set_to_none=True)
        router.begin_step(n, model=model)
        flops.clear()
        widths.clear()
        router.metrics()
        out = model(ids, labels=labels)
        assert torch.isfinite(out.loss)
        out.loss.backward()

    step(1)  # warmup: dense
    assert all(widths[i] == T for i in range(6))
    step(2)  # routed
    assert widths[2] < T and widths[4] < T and widths[2] == widths[4]
    assert all(widths[i] == T for i in (0, 1, 3, 5))
    assert all(p.grad is not None for b in blocks for p in b.parameters())
    e = router._pending[-1]
    assert e["target"] is not None and torch.isfinite(e["target"]).all() and e["ok"].all()
    assert int(e["rows"].numel()) == body  # samples_per_row 64 >= body: every body token, kept or not
    m = router.metrics()
    kept = int(round(m["online_router/mod_row_frac"] * T))
    assert kept == widths[2]
    f = lambda b, n: int(b.num_flops_per_token(int(n)))
    expect = sum(kept * f(blocks[i], kept) - T * f(blocks[i], T) for i in (2, 4))
    assert any(abs(v - expect) < 1e-6 * max(1.0, abs(expect)) for v in flops), (expect, flops)
    for h in hooks:
        h.remove()
    router.cfg.dense_every = 1
    router.begin_step(3, model=model)
    model.eval()
    ref = model(ids, labels=labels).loss
    model.train()
    assert torch.allclose(model(ids, labels=labels).loss, ref, atol=1e-5)


def test_mod_block_scatters_kept_tokens_only():
    """mod_block runs the block on the gathered kept columns (at their original positions) and leaves every other
    column, including the padded gather slots, untouched."""
    r = _router(mod_layers="1")
    h = torch.randn(2, 6, 4)
    order = torch.tensor([[1, 4, 0, 2], [0, 5, 1, 2]])
    valid = torch.tensor([[True, True, False, False], [True, True, True, False]])
    pos = torch.tensor([[1, 4, 5, 5], [0, 5, 1, 5]])
    r._mod = {"order": order, "valid": valid, "pos": pos}
    seen = {}

    def block(x, **kw):
        seen["pos"] = kw["position_ids"]
        return x + 1.0

    out = r.mod_block(block, h, {})
    assert torch.equal(seen["pos"], pos)
    ran = torch.zeros(2, 6, dtype=torch.bool)
    ran[0, [1, 4]] = True
    ran[1, [0, 5, 1]] = True
    assert torch.allclose(out[ran], h[ran] + 1.0)
    assert torch.equal(out[~ran], h[~ran])


def test_mod_layers_validate():
    with pytest.raises(ValueError):
        OnlineRouterConfig(mod_layers="4,6", drop_layer=2).validate()
    with pytest.raises(ValueError):
        _router(mod_layers="4,6", stages="4:0.5")
    with pytest.raises(ValueError):
        _router(mod_layers="4,6", exit_loss_weight=1.0)
    assert _router(mod_layers="4,6").cfg.drop_layer == 4


def test_per_task_routers_learn_opposite_rules():
    """per_task_key_tokens: two rows with different prompt prefixes (two tasks) whose targets reward the query-matching
    token in one task and penalise it in the other. A shared router would average the rules away; per-task routers
    keep it in task A and drop it in task B."""
    torch.manual_seed(0)
    m = _Emb()
    ids, cid, labels = _row(B=4)
    ids[2:, :3] = torch.tensor([4, 5, 6])  # rows 2-3: task B's prompt template
    r = _router(use_hidden=False, doc_share=0.0, per_task_key_tokens=3)
    keys = r.task_keys(ids)
    assert keys[0] == keys[1] and keys[2] == keys[3] and keys[0] != keys[2]
    r.begin_step(1, model=m)
    r.select(m, ids, cid, labels)
    e = r._current
    e["ok"] = torch.ones_like(e["rows"], dtype=torch.bool)
    q = torch.log1p(9.0 * (ids[e["rows"], e["pos"]] == QTOK).float())
    e["target"] = torch.where(e["rows"] < 2, q, -q)
    r._pending.append(e)
    r._current = None
    r.metrics()  # reset the per-step means (step 1 had no task fits yet)
    r.begin_step(2, model=m)
    assert set(r.task_w) == set(keys.tolist())
    keep = r.select(m, ids, cid, labels)
    body = (cid >= 0) & (ids != DS) & (ids != DE)
    assert keep[:2][(ids[:2] == QTOK) & body[:2]].all()
    assert not keep[2:][(ids[2:] == QTOK) & body[2:]].any()
    assert r.metrics()["online_router/task_routers"] == 2.0


def test_freeze_schedule_validate():
    with pytest.raises(ValueError):
        _router(drop_layer=3, freeze_schedule="4:0.5,2:0.75")
    with pytest.raises(ValueError):
        _router(drop_layer=3, freeze_schedule="2:0.5", freeze_after=4)
    with pytest.raises(ValueError):
        _router(freeze_schedule="2:0.5")  # needs the mid-stack drop
    _router(drop_layer=3, freeze_schedule="1:0.25,4:0.5")


@pytest.mark.parametrize("frozen", [1, 4])
def test_progressive_freeze_grads_and_flops_cpu(frozen):
    """drop_layer 3 on a 6-block model with blocks [0, frozen) frozen: blocks below min(frozen, 3) run forward-only
    (no grad, charged 1/3), frozen blocks >= 3 get no weight gradient but pass the activation gradient to the target
    hook (charged 2/3 at the kept length), the rest train; the target hook still fires."""
    from olmo_core.nn.online_token_router import freeze_and_release

    model, router = _mid_model(6, drop_layer=3, dense_every=0, freeze_schedule="1:0.25,4:0.5")
    blocks = list(model.blocks.values())
    flops = []
    orig = router._flop
    router._flop = lambda m, v: (flops.append(float(v)), orig(m, v))
    ids, cid, labels = _row(B=1)
    router.begin_step(1, model=model)
    model(ids, labels=labels).loss.backward()  # warmup, nothing frozen
    freeze_and_release(model, frozen)
    router.frozen_blocks = frozen
    model.zero_grad(set_to_none=True)
    router.begin_step(2, model=model)
    flops.clear()
    router.metrics()
    model(ids, labels=labels).loss.backward()
    assert all(p.grad is None for b in blocks[:frozen] for p in b.parameters())
    assert all(p.grad is not None for b in blocks[frozen:] for p in b.parameters())
    e = router._pending[-1]
    assert e["target"] is not None and torch.isfinite(e["target"]).all()
    real = int((ids != PH).sum())
    m = router.metrics()
    kept = int(round(m["online_router/mid_row_frac"] * real))
    head = int(model.lm_head.num_flops_per_token(0))
    f = lambda b, n: int(b.num_flops_per_token(int(n)))
    nu = min(frozen, 3)
    fac = [1 / 3 if i < nu else (2 / 3 if 3 <= i < frozen else 1.0) for i in range(6)]
    expect = (sum(fac[i] * real * f(blocks[i], real) for i in range(3))
              + sum(fac[i] * kept * f(blocks[i], kept) for i in range(3, 6)) + kept * head
              - real * (sum(f(b, real) for b in blocks) + head))
    assert any(abs(v - expect) < 1e-6 * max(1.0, abs(expect)) for v in flops), (expect, flops)


def test_freeze_schedule_callback_stages():
    """The callback freezes the deepest stage whose step share has passed, once, and releases optimizer state."""
    from types import SimpleNamespace

    from olmo_core.train.callbacks.online_router import OnlineRouterCallback

    model, router = _mid_model(6, drop_layer=3, dense_every=0, freeze_schedule="1:0.25,4:0.5")
    opt = torch.optim.AdamW(model.parameters())
    model(_row()[0], labels=_row()[2]).loss.backward()
    opt.step()
    tr = SimpleNamespace(global_step=0, max_steps=8, train_module=SimpleNamespace(model=model, optim=opt))
    cb = OnlineRouterCallback()
    cb.trainer = tr
    cb.pre_train()
    blocks = list(model.blocks.values())
    seen = []
    for step in range(1, 9):
        tr.global_step = step
        cb.pre_step({})
        seen.append(router.frozen_blocks)
    assert seen == [0, 0, 1, 1, 4, 4, 4, 4]
    assert not any(p.requires_grad for b in blocks[:4] for p in b.parameters())
    assert all(p.requires_grad for b in blocks[4:] for p in b.parameters())
    assert not any(p in opt.state for b in blocks[:4] for p in b.parameters())


def test_mass_keep_minimal_cover_per_document():
    """One dominant token carries a document's predicted mass -> it alone is kept; a uniform document keeps
    ceil(share x n)."""
    ids, cid, _ = _row()
    body = (cid >= 0) & (ids != DS) & (ids != DE)
    score = torch.zeros(ids.shape)
    p0 = int((ids[0] == 1004).nonzero()[0])  # a token of doc 0 (10 body tokens)
    score[0, p0] = 5.0
    keep = OnlineTokenRouter._mass_keep(score, body, cid, 0.8)
    d0 = body[0] & (cid[0] == cid[0, p0])
    assert keep[0][d0].sum() == 1 and keep[0, p0]
    assert int(keep[0].sum()) == 1 + 32 + 3  # docs of 40 and 3 uniform: ceil(0.8 n)
    assert not keep[~body].any()


@pytest.mark.parametrize("cap,expect", [(1.0, 8 + 32 + 3), (0.1, 6)])
def test_mass_target_keep_and_cap(cap, expect):
    torch.manual_seed(0)
    m = _Emb()
    ids, cid, labels = _row()
    r = _router(use_hidden=False, mass_target=0.8, max_keep_frac=cap)
    r.w = torch.zeros(r.n_features)  # uniform scores
    r.step = 2
    keep = r.select(m, ids, cid, labels)
    assert int(keep.sum()) == expect


def test_mass_target_validate():
    with pytest.raises(ValueError):
        _router(mass_target=1.5)
    with pytest.raises(ValueError):
        _router(mass_target=0.8, score="random")
    with pytest.raises(ValueError):
        _router(mass_target=0.8, max_keep_frac=0.0)


def test_target_layer_validate():
    with pytest.raises(ValueError):
        _router(drop_layer=3, target_layer=2, dense_every=4)
    with pytest.raises(ValueError):
        _router(drop_layer=3, target_layer=5, explore_frac=0.0)  # the drop fits on the exploration tokens
    with pytest.raises(ValueError):
        _router(drop_layer=3, target_layer=5, explore_frac=0.2, target="dh")
    _router(drop_layer=3, target_layer=5, explore_frac=0.2)
    _router(mod_layers="3,5", target_layer=4)


def test_target_layer_reads_deeper_gradient_cpu():
    """target_layer 4 with the drop at 2: the target is log ||dL/dh|| at block 4's INPUT (for the sampled tokens); on
    a routed step only the exploration tokens count (uncensored)."""
    model, router = _mid_model(6, drop_layer=2, target_layer=4, dense_every=0, explore_frac=0.5)
    ids, cid, labels = _row(B=1)
    cap = {}

    def pre(m, a):
        a[0].retain_grad()
        cap["h"] = a[0]

    hk = list(model.blocks.values())[4].register_forward_pre_hook(pre)
    router.begin_step(1, model=model)  # warmup = full context
    model(ids, labels=labels).loss.backward()
    e = router._pending[-1]
    g = cap["h"].grad[e["rows"], e["cols"]]
    assert torch.allclose(e["target"], torch.log(g.norm(dim=-1) + 1e-12), atol=1e-5)
    assert e["ok"].all()  # warmup = full context: every sampled token
    router.begin_step(2, model=model)  # routed
    model(ids, labels=labels).loss.backward()
    e = router._pending[-1]
    assert torch.equal(e["ok"], e["explore"]) and e["explore"].any() and not e["explore"].all()
    g = cap["h"].grad[e["rows"], e["cols"]]
    assert torch.allclose(e["target"][e["ok"]], torch.log(g.norm(dim=-1) + 1e-12)[e["ok"]], atol=1e-5)
    hk.remove()


def test_target_layer_with_mixture_of_depths_fits_every_step_cpu():
    model, router = _mid_model(6, mod_layers="2-5:2", target_layer=3, dense_every=0)
    ids, cid, labels = _row(B=1)
    for step in (1, 2, 3):
        router.begin_step(step, model=model)
        model(ids, labels=labels).loss.backward()
        assert len(router._pending) == 1 and torch.isfinite(router._pending[0]["target"]).all()


def test_early_lora_identity_init_grads_and_flops_cpu():
    """early_lora 4 on blocks [0, 2) of a 4-block model: the untrained adapters reproduce the base exactly; the base
    weights of blocks 0-1 and the embeddings get no gradient while their adapters and blocks 2-3 do; the FLOP
    adjustment charges blocks 0-1 at 2/3 plus the adapters' 6 r (in + out) per token."""
    from olmo_core.nn.early_lora import early_lora_flops_per_token, install_early_lora

    model, router = _mid_model(4, drop_layer=2, dense_every=0, early_lora=4)
    ids, cid, labels = _row(B=1)
    model.eval()
    ref = model(ids, labels=labels).loss
    mods = install_early_lora(model, 2, 4, scale=8.0)
    assert torch.allclose(model(ids, labels=labels).loss, ref, atol=1e-6)
    model.train()
    blocks = list(model.blocks.values())
    flops = []
    orig = router._flop
    router._flop = lambda m, v: (flops.append(float(v)), orig(m, v))
    router.begin_step(1, model=model)
    model(ids, labels=labels).loss.backward()
    model.zero_grad(set_to_none=True)
    router.begin_step(2, model=model)
    flops.clear()
    router.metrics()
    model(ids, labels=labels).loss.backward()
    assert all(m.weight.grad is None and m.lora_A.grad is not None and m.lora_B.grad is not None for m in mods)
    assert all(p.grad is None for p in model.embeddings.parameters())
    assert all(p.grad is not None for b in blocks[2:] for p in b.parameters())
    real = int((ids != PH).sum())
    kept = int(round(router.metrics()["online_router/mid_row_frac"] * real))
    head = int(model.lm_head.num_flops_per_token(0))
    f = lambda b, n: int(b.num_flops_per_token(int(n)))
    expect = (early_lora_flops_per_token(model) * real
              + real * (2 / 3) * (f(blocks[0], real) + f(blocks[1], real))
              + kept * (f(blocks[2], kept) + f(blocks[3], kept) + head)
              - real * (sum(f(b, real) for b in blocks) + head))
    assert any(abs(v - expect) < 1e-6 * max(1.0, abs(expect)) for v in flops), (expect, flops)


def test_early_lora_reset_and_dead_adapter_check():
    from olmo_core.nn.early_lora import install_early_lora

    model, router = _mid_model(3, drop_layer=1, dense_every=0, early_lora=2)
    mods = install_early_lora(model, 1, 2)
    m = mods[0]
    with torch.no_grad():
        m.lora_B.fill_(7.0)
    m.reset_parameters()  # what init_weights calls on a meta-built model
    assert torch.equal(m.lora_B, torch.zeros_like(m.lora_B)) and torch.isfinite(m.lora_A).all()
    # an adapted linear that the forward never calls -> loud failure, not a silent no-op adapter
    model2, _ = _mid_model(3, drop_layer=1, dense_every=0, early_lora=2)
    list(model2.blocks.values())[0].unused = torch.nn.Linear(8, 8)
    install_early_lora(model2, 1, 2)
    ids, cid, labels = _row()
    with pytest.raises(RuntimeError, match="never ran"):
        model2(ids, labels=labels)


def test_early_lora_validate_and_eval_reinstall(tmp_path):
    import json

    from olmo_core.nn.nested_ffn_moe import post_build_hook_from_config

    with pytest.raises(ValueError):
        _router(drop_layer=2, early_lora=4, freeze_early=True)
    _router(drop_layer=2, early_lora=4, freeze_after=10)  # LoRA, then freeze the adapters too
    with pytest.raises(ValueError):
        _router(early_lora=4)  # needs the mid-stack drop
    (tmp_path / "config.json").write_text(json.dumps({"early_lora": {"n_blocks": 1, "rank": 2, "scale": 8.0}}))
    hook = post_build_hook_from_config(str(tmp_path))
    model, _ = _mid_model(3, drop_layer=1, dense_every=0)
    hook(model)
    assert model._early_lora["rank"] == 2 and all(hasattr(m, "lora_A") for m in model._early_lora["modules"])
    assert any(k.endswith(".lora_A") for k in model.state_dict())


def test_lora_block_from_checkpoint(tmp_path):
    """A checkpoint holding adapter keys but no config.json block is rebuilt from provenance.json, or refused --
    never loaded as a plain model."""
    import json

    import torch.distributed.checkpoint as dcp

    from olmo_core.nn.early_lora import install_early_lora, lora_block_from_checkpoint

    model, _ = _mid_model(3, drop_layer=2, dense_every=0, early_lora=2)
    install_early_lora(model, 2, 2, scale=8.0)
    dcp.save({"model": model.state_dict()}, checkpoint_id=str(tmp_path / "model_and_optim"))
    with pytest.raises(RuntimeError, match="refusing"):
        lora_block_from_checkpoint(str(tmp_path))
    (tmp_path / "provenance.json").write_text(json.dumps({"args": {"st_otr_early_lora": 2,
                                                                   "st_otr_early_lora_scale": 8.0}}))
    assert lora_block_from_checkpoint(str(tmp_path)) == {"n_blocks": 2, "rank": 2, "scale": 8.0}
    plain, _ = _mid_model(3, drop_layer=2, dense_every=0)
    dcp.save({"model": plain.state_dict()}, checkpoint_id=str(tmp_path / "plain" / "model_and_optim"))
    assert lora_block_from_checkpoint(str(tmp_path / "plain")) is None


def test_early_lora_then_freeze_after_cpu():
    """early_lora + freeze_after: after the freeze the adapters get no gradient, blocks [0, L) run forward-only (1/3)
    and the adapters are charged forward-only too."""
    from olmo_core.nn.early_lora import early_lora_flops_per_token, install_early_lora
    from olmo_core.nn.online_token_router import freeze_and_release

    model, router = _mid_model(4, drop_layer=2, dense_every=0, early_lora=4, freeze_after=1)
    mods = install_early_lora(model, 2, 4, scale=8.0)
    ids, cid, labels = _row(B=1)
    router.begin_step(1, model=model)
    model(ids, labels=labels).loss.backward()
    assert all(m.lora_A.grad is not None for m in mods)
    freeze_and_release(model, 2)  # what the callback does after freeze_after steps
    router.cfg.freeze_early = True
    model.zero_grad(set_to_none=True)
    blocks = list(model.blocks.values())
    flops = []
    orig = router._flop
    router._flop = lambda m, v: (flops.append(float(v)), orig(m, v))
    router.begin_step(2, model=model)
    flops.clear()
    router.metrics()
    model(ids, labels=labels).loss.backward()
    assert all(m.lora_A.grad is None and m.lora_B.grad is None for m in mods)
    assert all(p.grad is not None for b in blocks[2:] for p in b.parameters())
    real = int((ids != PH).sum())
    kept = int(round(router.metrics()["online_router/mid_row_frac"] * real))
    head = int(model.lm_head.num_flops_per_token(0))
    f = lambda b, n: int(b.num_flops_per_token(int(n)))
    expect = (early_lora_flops_per_token(model) / 3 * real
              + real * (1 / 3) * (f(blocks[0], real) + f(blocks[1], real))
              + kept * (f(blocks[2], kept) + f(blocks[3], kept) + head)
              - real * (sum(f(b, real) for b in blocks) + head))
    assert any(abs(v - expect) < 1e-6 * max(1.0, abs(expect)) for v in flops), (expect, flops)
