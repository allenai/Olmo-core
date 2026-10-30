"""Multi-view training from a cached split-layer state (olmo_core.nn.multiview, ``--st-otr-views``)."""
import contextlib

import pytest
import torch

from olmo_core.nn.multiview import _mv_train_batch, enable_multiview
from olmo_core.nn.online_token_router import OnlineRouterConfig, OnlineTokenRouter

from .online_token_router_test import EOS, PH, _row


def _model(n_layers=4, stages="1:0.5,2:0.25", **cfg):
    from olmo_core.config import DType
    from olmo_core.nn.transformer import TransformerConfig

    from .online_token_router_test import DE, DS

    torch.manual_seed(0)
    model = TransformerConfig.olmo2_190M(vocab_size=9100, n_layers=n_layers, fused_ops=False,
                                         dtype=DType.float32).build(init_device="cpu")
    D = int(model.embeddings.weight.shape[1])
    router = OnlineTokenRouter(OnlineRouterConfig(**{"keep_frac": 0.25, "explore_frac": 0.2, "samples_per_row": 64,
                                                     "stages": stages, "dense_every": 0, "score": "random", **cfg}),
                               doc_start_id=DS, doc_end_id=DE, eos_id=EOS, d_model=D)
    model.enable_pooled_soft_tokens(
        DS, DE, EOS, placeholder_id=PH, keep_prob=0.0, keep_seed=0, detach_soft_kv=True,
        drop_slots=True, keep_token_rule="online", keep_token_k=0.25, online_router=router,
    )
    model.train()
    return model, router


def _warm(model, router, ids, labels, steps=(1, 2)):
    for n in steps:
        model.zero_grad(set_to_none=True)
        for r in router.chain():
            r.begin_step(n, model=model)
            r.metrics()
        model(ids, labels=labels).loss.backward()
    for r in router.chain():
        r.begin_step(steps[-1] + 1, model=model)
        r.metrics()
    assert not router.dense_now()


def _calls(router):
    return [r._calls for r in router.chain()]


def _set_calls(router, c):
    for r, x in zip(router.chain(), c):
        r._calls = x


def test_cached_split_is_exact_cpu():
    """View 1 (graph split at the drop layer) and a repeat view from the cached state reproduce the plain forward's
    loss for the same draw, and the two-part backward (late, then early with dL/dh_L) reproduces its gradients."""
    model, router = _model()
    ids, _, labels = _row(B=2)
    _warm(model, router, ids, labels)
    c0 = _calls(router)
    model.zero_grad(set_to_none=True)
    ref = model(ids, labels=labels).loss
    ref.backward()
    g_ref = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    _set_calls(router, c0)
    model.zero_grad(set_to_none=True)
    ctx = {"layer": 1, "repeat": False, "h7": None}
    model._mv_ctx = ctx
    l1 = model(ids, labels=labels).loss
    model._mv_ctx = None
    assert torch.allclose(l1, ref, atol=1e-6)
    l1.backward()
    blocks = list(model.blocks.values())
    assert all(p.grad is None for p in blocks[0].parameters())  # the early stack waits for dL/dh_L
    torch.autograd.backward(ctx["h7_attached"], ctx["leaf"].grad)
    for n, p in model.named_parameters():
        if n in g_ref:
            assert p.grad is not None and torch.allclose(p.grad, g_ref[n], atol=1e-5, rtol=1e-4), n

    _set_calls(router, c0)
    rep = {"layer": 1, "repeat": True, "h7": ctx["h7"]}
    model._mv_ctx = rep
    lens_before = len((getattr(model, "_flop_example_lens", None) or {"dense": []})["dense"])
    with torch.no_grad():
        l2 = model(ids, labels=labels).loss
    model._mv_ctx = None
    assert torch.allclose(l2, ref, atol=1e-6)
    lens_after = len((getattr(model, "_flop_example_lens", None) or {"dense": []})["dense"])
    assert lens_after == lens_before


def test_repeat_view_flops_charge_late_stack_only_cpu():
    """A repeat view's stage-0 FLOP adjustment is the late stack + head at the kept length (nothing below the split,
    no dense subtraction); the later stage keeps its relative formula; the meter logs no example for it."""
    model, router = _model()
    model._flop_pad_id = EOS
    ids, _, labels = _row(B=1)
    _warm(model, router, ids, labels)
    chain = router.chain()
    flops = []
    for r in chain:
        orig = r._flop
        r._flop = (lambda o: (lambda m, v: (flops.append(float(v)), o(m, v))))(orig)
    ctx = {"layer": 1, "repeat": False, "h7": None}
    model._mv_ctx = ctx
    model(ids, labels=labels)
    model._flop_example_lens = {"dense": [], "actual": []}
    for r in chain:
        r.metrics()
    flops.clear()
    model._mv_ctx = {"layer": 1, "repeat": True, "h7": ctx["h7"]}
    model(ids, labels=labels)
    model._mv_ctx = None
    assert model._flop_example_lens == {"dense": [], "actual": []}
    blocks = list(model.blocks.values())
    real = int((ids != PH).sum())
    m0, m1 = chain[0].metrics(), chain[1].metrics()
    k1 = int(round(m0["online_router/mid_row_frac"] * real))
    k2 = int(round(m1["online_router/s1_mid_row_frac"] * k1))
    head = int(model.lm_head.num_flops_per_token(0))
    f = lambda b, n: int(b.num_flops_per_token(int(n)))
    e1 = k1 * (sum(f(b, k1) for b in blocks[1:]) + head)
    e2 = k2 * (f(blocks[2], k2) + f(blocks[3], k2) + head) - k1 * (f(blocks[2], k1) + f(blocks[3], k1) + head)
    for e in (e1, e2):
        assert any(abs(v - e) < 1e-6 * max(1.0, abs(e)) for v in flops), (e, flops)


class _TM:
    """The attributes of TransformerTrainModule that the multi-view train_batch reads (no trainer, no FSDP)."""

    label_ignore_index = -100
    microbatch_sort_pad_id = None
    device = torch.device("cpu")
    z_loss_multiplier = None
    max_grad_norm = 1.0
    scheduler = None

    def __init__(self, model, rows_per_mb, seq_len):
        self.model = model
        self.rank_microbatch_size = rows_per_mb * seq_len
        self.optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        self.metrics = {}
        self.train_batch = lambda b, dry_run=False: (_ for _ in ()).throw(AssertionError("orig called"))

    def _set_model_mode(self, mode):
        pass

    @contextlib.contextmanager
    def _train_microbatch_context(self, i, n):
        yield

    def _prepare_batch(self, b):
        return b.pop("input_ids"), b.pop("labels", None), b

    def model_forward(self, input_ids, labels=None, **kw):
        return self.model(input_ids, labels=labels, **kw)

    def _clip_grad_norm(self, m):
        return torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.grad is not None], m)

    def record_metric(self, name, value, reduce_type=None, namespace=None):
        self.metrics[name] = value

    def record_ce_loss(self, value, reduce_type=None):
        self.metrics["ce"] = value


@pytest.mark.parametrize("k", [2, 3])
def test_multiview_train_batch_cpu(k):
    """k views: the late stack takes k - 1 inner optimizer steps (the trainer takes the k-th), the early stack none
    until the trainer's step (its gradient = mean over views), each row is logged for the FLOP meter once, and the
    views draw different token subsets."""
    model, router = _model()
    model._flop_pad_id = EOS
    ids, _, labels = _row(B=2)
    _warm(model, router, ids, labels)
    model.zero_grad(set_to_none=True)
    tm = _TM(model, rows_per_mb=1, seq_len=ids.shape[1])
    enable_multiview(tm, k)
    model._flop_example_lens = {"dense": [], "actual": []}
    blocks = list(model.blocks.values())
    w_early = [p.detach().clone() for p in blocks[0].parameters()]
    keep_fracs = []
    orig_stat = router._stat
    router._stat = lambda key, v: (keep_fracs.append(v) if key == "mid_row_frac" else None, orig_stat(key, v))
    tm.train_batch({"input_ids": ids.clone(), "labels": labels.clone()})
    assert tm._multiview["last_views"] == k
    assert len(model._flop_example_lens["dense"]) == 2  # one entry per row, not per view
    late = [p for b in blocks[1:] for p in b.parameters()]
    assert all(int(tm.optim.state[p]["step"]) == k - 1 for p in late)
    early = list(blocks[0].parameters())
    assert all(len(tm.optim.state[p]) == 0 for p in early)
    assert all(torch.equal(p, w) for p, w in zip(early, w_early))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in early)
    assert all(p.grad is not None for p in late)
    assert torch.isfinite(torch.as_tensor(tm.metrics["ce"]))
    assert "multiview/ce_last_view" in tm.metrics
    assert len(keep_fracs) == 2 * k  # 2 micro-batches x k views reached the stage-0 drop
    tm.optim.step()  # the trainer's step
    assert all(int(tm.optim.state[p]["step"]) == 1 for p in early)


def test_multiview_dense_step_runs_once_cpu():
    """A step the router trains dense (warmup) goes through the ordinary train_batch."""
    model, router = _model()
    tm = _TM(model, rows_per_mb=1, seq_len=8)
    called = []
    tm.train_batch = lambda b, dry_run=False: called.append(dry_run)
    enable_multiview(tm, 2)
    tm.train_batch({}, dry_run=True)
    router.begin_step(1, model=model)
    assert router.dense_now()
    tm.train_batch({})
    assert called == [True, False]


def test_multiview_validate():
    model, router = _model(stages="", drop_layer=2, early_nograd=True)
    with pytest.raises(ValueError):
        enable_multiview(_TM(model, 1, 8), 2)
