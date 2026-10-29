"""Online router mid-stack drop on PACKED rows (segment mode, cu_doc_lens): the packed forward must train exactly what
the padded one-example-per-row forward trains (same kept tokens, positions, loss and FLOP charge), and
``route_min_len`` must route only the long documents."""

import pytest
import torch

from olmo_core.nn.online_token_router import OnlineRouterConfig, OnlineTokenRouter
from olmo_core.testing import requires_gpu
from olmo_core.testing.utils import requires_fla

from .pooled_soft_token_packed_test import DOC_END, DOC_START, EOS, EXAMPLES, PLACEHOLDER, _packed, _padded


def _model(**cfg):
    from olmo_core.nn.attention import AttentionConfig, GatedDeltaNetConfig
    from olmo_core.nn.attention.backend import AttentionBackendName
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
    attn = TransformerBlockConfig(
        name=TransformerBlockType.reordered_norm,
        attention=AttentionConfig(n_heads=8, rope=RoPEConfig(name=RoPEType.default),
                                  backend=AttentionBackendName.flash_2),
        layer_norm=ln, feed_forward=ff)
    model = TransformerConfig(d_model=256, vocab_size=1024, n_layers=3, block=gdn, block_overrides={1: attn},
                              lm_head=LMHeadConfig(layer_norm=ln, bias=False)).build(init_device="cuda")
    model.init_weights()
    model = model.to(torch.bfloat16)
    router = OnlineTokenRouter(
        OnlineRouterConfig(**{"keep_frac": 0.3, "explore_frac": 0.0, "samples_per_row": 64, "drop_layer": 1,
                              "score": "first_last", "use_hidden": False, **cfg}),
        doc_start_id=DOC_START, doc_end_id=DOC_END, eos_id=EOS, d_model=256)
    model.enable_pooled_soft_tokens(DOC_START, DOC_END, EOS, placeholder_id=PLACEHOLDER, keep_prob=0.0,
                                    drop_slots=True, keep_token_rule="online", keep_token_k=0.3,
                                    online_router=router, pad_multiple=16)
    model._flop_pad_id = EOS
    router.w = torch.zeros(router.n_features, device="cuda")  # past warmup: a routed step
    router.step = 2
    model.train()
    return model, router


def _run(model, router, packed):
    flops = []
    orig = router._flop
    router._flop = lambda m, v: (flops.append(float(v)), orig(m, v))
    router.metrics()
    if packed:
        ids, lab, dl = _packed()
        out = model(ids.cuda(), labels=lab.cuda(), loss_reduction="sum", doc_lens=dl.cuda(),
                    max_doc_lens=[int(dl.max())])
    else:
        ids, lab = _padded()
        out = model(ids.cuda(), labels=lab.cuda(), loss_reduction="sum")
    router._flop = orig
    return out, flops, router.metrics()


@requires_gpu
@requires_fla
def test_packed_mid_stack_drop_matches_padded():
    model, router = _model()
    out_pad, fl_pad, m_pad = _run(model, router, packed=False)
    out_pk, fl_pk, m_pk = _run(model, router, packed=True)
    assert m_pk["online_router/mid_row_frac"] == pytest.approx(m_pad["online_router/mid_row_frac"])
    assert m_pk["online_router/mid_row_frac"] < 1.0
    assert torch.allclose(out_pk.loss.float(), out_pad.loss.float(), rtol=2e-2, atol=2e-2), (out_pk.loss, out_pad.loss)
    assert sum(fl_pk) == pytest.approx(sum(fl_pad), rel=1e-6)
    out_pk.loss.backward()
    e = router._pending[-1]
    assert e["target"] is not None and torch.isfinite(e["target"]).all()


@requires_gpu
@requires_fla
def test_packed_route_min_len_routes_long_documents_only():
    lens = sorted(len(e) for e in EXAMPLES)
    model, router = _model(route_min_len=lens[-1])  # only the longest example is routed
    out, _, m = _run(model, router, packed=True)
    assert m["online_router/routed_row_frac"] == pytest.approx(1 / len(EXAMPLES))
    full = _model(route_min_len=10**9)
    out_full, fl_full, m_full = _run(full[0], full[1], packed=True)
    assert m_full["online_router/mid_row_frac"] == pytest.approx(1.0)
    assert m["online_router/mid_row_frac"] < 1.0
    out.loss.backward()
