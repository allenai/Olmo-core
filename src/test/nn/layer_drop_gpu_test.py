"""Layer drop under FSDP2 (2 ranks) and on GatedDeltaNet blocks. GPU only."""

import copy

import torch
import torch.distributed as dist
import torch.nn.functional as F

from olmo_core.config import DType
from olmo_core.distributed.utils import get_full_tensor, get_rank
from olmo_core.nn.attention import GatedDeltaNetConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.transformer import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)
from olmo_core.testing import requires_gpu, requires_multi_gpu, run_distributed_test
from olmo_core.testing.utils import requires_fla
from olmo_core.utils import get_default_device, seed_all

VOCAB = 1000


def _ce(model, ids):
    logits = model(ids[:, :-1])
    return F.cross_entropy(logits.reshape(-1, VOCAB).float(), ids[:, 1:].reshape(-1))


def run_fsdp2_layer_drop_matches_single_process():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = get_default_device()
    seed_all(0)
    cfg = TransformerConfig.olmo2_190M(
        vocab_size=VOCAB, n_layers=4, fused_ops=False, dtype=DType.float32
    )
    model = cfg.build(init_device=dev)
    model.init_weights(max_seq_len=64, device=dev)
    ref = copy.deepcopy(model)
    model.apply_fsdp()
    for m in (model, ref):
        m.enable_layer_drop(p=0.5, seed=0)
        m.train()
    # SGD, not Adam: Adam's first step is ~lr * sign(g), which turns reduction-order noise in
    # near-zero grads into lr-sized parameter differences. The grads are compared directly below.
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    ref_opt = torch.optim.SGD(ref.parameters(), lr=0.05)
    rank, world = get_rank(), dist.get_world_size()
    gen = torch.Generator().manual_seed(1)
    seen = []
    for step in range(1, 7):
        model._layer_drop.step = ref._layer_drop.step = step
        skipped = model._layer_drop.skipped()
        seen.append(skipped)
        batch = torch.randint(0, VOCAB, (2 * world, 33), generator=gen).to(dev)  # 2 micro-batches/rank
        local = batch[2 * rank : 2 * rank + 2]
        for mb in range(2):
            (_ce(model, local[mb : mb + 1]) / 2).backward()
        ref_loss = sum(_ce(ref, batch[i : i + 1]) for i in range(batch.shape[0])) / batch.shape[0]
        ref_loss.backward()
        for k, blk in model.blocks.items():
            has = [p.grad is not None for p in blk.parameters()]
            if int(k) in skipped:
                assert not any(has), (step, k)
            else:
                assert all(has), (step, k)
        for (n, p), (_, rp) in zip(model.named_parameters(), ref.named_parameters()):
            if p.grad is None:
                assert rp.grad is None, n  # skipped in both
                continue
            g = get_full_tensor(p.grad)
            assert torch.isfinite(g).all(), n
            torch.testing.assert_close(g, rp.grad, atol=1e-6, rtol=1e-3, msg=lambda m: f"{n} step {step}: {m}")
        opt.step()
        ref_opt.step()
        opt.zero_grad(set_to_none=True)
        ref_opt.zero_grad(set_to_none=True)
        for (n, p), (_, rp) in zip(model.named_parameters(), ref.named_parameters()):
            torch.testing.assert_close(
                get_full_tensor(p).detach(), rp.detach(), atol=1e-5, rtol=1e-4, msg=lambda m: f"{n} step {step}: {m}"
            )
    assert any(seen) and len(set(seen)) > 1, seen  # the draw skipped something and varied
    # eval runs every block: FSDP output equals the reference model with no skipping
    model.eval()
    ref.eval()
    with torch.no_grad():
        ids = torch.randint(0, VOCAB, (1, 16), generator=gen).to(dev)
        torch.testing.assert_close(model(ids), ref(ids), atol=2e-5, rtol=1e-4)


@requires_multi_gpu
def test_fsdp2_layer_drop_matches_single_process():
    run_distributed_test(
        run_fsdp2_layer_drop_matches_single_process, backend="nccl", start_method="spawn"
    )


@requires_gpu
@requires_fla
def test_gdn_blocks_are_skipped_whole():
    torch.manual_seed(0)
    ln = LayerNormConfig(name=LayerNormType.rms, bias=False)
    cfg = TransformerConfig(
        d_model=256,
        vocab_size=1024,
        n_layers=3,
        block=TransformerBlockConfig(
            name=TransformerBlockType.reordered_norm,
            sequence_mixer=GatedDeltaNetConfig(n_heads=8),
            layer_norm=ln,
            feed_forward=FeedForwardConfig(hidden_size=512, bias=False),
        ),
        lm_head=LMHeadConfig(layer_norm=ln, bias=False),
    )
    m = cfg.build(init_device="cuda")
    m.init_weights()
    m.enable_layer_drop(fixed={1})
    m.train()
    ids = torch.randint(0, 1024, (2, 64), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(ids)
        h = m.embeddings(ids)
        h = m.blocks["2"](m.blocks["0"](h))
        want = m.lm_head(h)
    torch.testing.assert_close(out, want)
    out.float().sum().backward()
    assert all(p.grad is None for p in m.blocks["1"].parameters())
    assert all(p.grad is not None for p in m.blocks["2"].parameters())


def run_fsdp2_grad_select_update_mode_matches_single_process():
    """Grad layer select, update mode, under FSDP2: requires_grad toggled on the sharded params each step reaches the
    unsharded ones (FSDP2 copies it on every unshard); unselected blocks get no grad on any rank; the selected
    blocks' grads and params match a single-process reference that freezes the same blocks; the online EMA is
    identical on both ranks."""
    from olmo_core.train.callbacks import GradLayerSelectCallback
    from types import SimpleNamespace

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = get_default_device()
    seed_all(0)
    cfg = TransformerConfig.olmo2_190M(vocab_size=VOCAB, n_layers=6, fused_ops=False, dtype=DType.float32)
    model = cfg.build(init_device=dev)
    model.init_weights(max_seq_len=64, device=dev)
    ref = copy.deepcopy(model)
    model.apply_fsdp()
    for m in (model, ref):
        m.enable_grad_layer_select(keep_frac=0.5, explore_frac=0.34, seed=0)  # k=3, 1 explored slot
        m.train()
    cbs = []
    for m in (model, ref):
        cb = GradLayerSelectCallback(log_every=0)
        cb._trainer = SimpleNamespace(global_step=0, save_folder="/nonexistent",  # type: ignore[assignment]
                                      train_module=SimpleNamespace(model=m), record_metric=lambda k, v: None)
        cb.pre_train()
        cbs.append(cb)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    ref_opt = torch.optim.SGD(ref.parameters(), lr=0.05)
    rank, world = get_rank(), dist.get_world_size()
    gen = torch.Generator().manual_seed(1)
    seen = []
    for step in range(1, 8):
        for cb in cbs:
            cb._trainer.global_step = step
            cb.pre_step({})
        sel = model._grad_select.selected()
        assert sel == ref._grad_select.selected(), step  # ref EMA is built from the full-batch grads
        seen.append(sel)
        batch = torch.randint(0, VOCAB, (2 * world, 33), generator=gen).to(dev)
        local = batch[2 * rank : 2 * rank + 2]
        for mb in range(2):
            (_ce(model, local[mb : mb + 1]) / 2).backward()
        (sum(_ce(ref, batch[i : i + 1]) for i in range(batch.shape[0])) / batch.shape[0]).backward()
        for k, blk in model.blocks.items():
            has = [p.grad is not None for p in blk.parameters()]
            assert all(has) if int(k) in sel else not any(has), (step, k, sel)
        for (n, p), (_, rp) in zip(model.named_parameters(), ref.named_parameters()):
            if p.grad is None:
                assert rp.grad is None, n
                continue
            torch.testing.assert_close(get_full_tensor(p.grad), rp.grad, atol=1e-6, rtol=1e-3,
                                       msg=lambda m_, n=n: f"{n} step {step}: {m_}")
        for cb in cbs:
            cb.pre_optim_step()
        opt.step()
        ref_opt.step()
        opt.zero_grad(set_to_none=True)
        ref_opt.zero_grad(set_to_none=True)
        e = torch.tensor([x if x is not None else -1.0 for x in model._grad_select.ema], device=dev)
        e0 = e.clone()
        dist.broadcast(e0, 0)
        torch.testing.assert_close(e, e0)  # same ranking state on every rank
        # the reference's callback also all-reduces (SUM) its full, replicated grads over the 2 ranks: 2x the energy
        r = torch.tensor([x / world if x is not None else -1.0 for x in ref._grad_select.ema], device=dev)
        torch.testing.assert_close(torch.where(e < 0, e, e), torch.where(r < 0, e, r), rtol=1e-3, atol=0.0)
        for (n, p), (_, rp) in zip(model.named_parameters(), ref.named_parameters()):
            torch.testing.assert_close(get_full_tensor(p).detach(), rp.detach(), atol=1e-5, rtol=1e-4,
                                       msg=lambda m_, n=n: f"{n} step {step}: {m_}")
    assert len(set(seen)) > 1, seen
    assert all(x is not None for x in model._grad_select.ema)  # every block measured


@requires_multi_gpu
def test_fsdp2_grad_select_update_mode_matches_single_process():
    run_distributed_test(
        run_fsdp2_grad_select_update_mode_matches_single_process, backend="nccl", start_method="spawn"
    )
