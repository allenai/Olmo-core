"""FlopMeterCallback charges only the blocks that ran under layer drop (olmo_core.nn.layer_drop)."""

from types import SimpleNamespace

import pytest

from olmo_core.config import DType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train.callbacks.flop_meter import FlopMeterCallback


def _setup(per_example, fixed=None, p=0.0):
    m = TransformerConfig.olmo2_190M(
        vocab_size=1000, n_layers=4, fused_ops=False, dtype=DType.float32
    ).build(init_device="cpu")
    if fixed or p:
        m.enable_layer_drop(fixed=fixed, p=p, seed=0)
    cb = FlopMeterCallback(seq_len=256, pad_id=0 if per_example else None, per_example=per_example)
    metrics = {}
    cb._trainer = SimpleNamespace(  # type: ignore[assignment]
        global_step=1,
        global_batch_size=2 * 256,
        train_module=SimpleNamespace(model=m),
        record_metric=lambda k, v: metrics.__setitem__(k, v),
    )
    return m, cb, metrics


def _blk(m, L):
    return m.blocks["0"].num_flops_per_token(L)


@pytest.mark.parametrize("fixed", [None, {1}, {0, 3}])
def test_per_example_charges_only_blocks_that_ran(fixed):
    m, cb, metrics = _setup(per_example=True, fixed=fixed)
    lens = [100, 37, 256]
    m._flop_example_lens = {"dense": list(lens), "actual": list(lens)}
    cb.post_step()
    n_skip = len(fixed or ())
    want_dense = sum(L * m.num_flops_per_token(L) for L in lens)
    want_actual = sum(L * (m.num_flops_per_token(L) - n_skip * _blk(m, L)) for L in lens)
    assert metrics["flop_meter/dense_pflops"] == pytest.approx(want_dense / 1e15)
    assert metrics["flop_meter/actual_pflops"] == pytest.approx(want_actual / 1e15)
    if fixed:
        assert metrics["flop_meter/layers_run_frac"] == pytest.approx(1 - n_skip / 4)
        assert cb.summary()["layer_drop_run_frac"] == pytest.approx(1 - n_skip / 4)


def test_random_drop_is_charged_per_step():
    m, cb, metrics = _setup(per_example=True, p=0.5)
    plan = m._layer_drop
    total = 0.0
    for step in range(1, 6):
        plan.step = step
        cb._trainer.global_step = step
        m._flop_example_lens = {"dense": [64, 128], "actual": [64, 128]}
        cb.post_step()
        k = len(plan.skipped_for_step(step))
        total += sum(L * (m.num_flops_per_token(L) - k * _blk(m, L)) for L in (64, 128))
    assert metrics["flop_meter/actual_pflops"] == pytest.approx(total / 1e15)


def test_padded_dense_path_charges_only_blocks_that_ran():
    m, cb, metrics = _setup(per_example=False, fixed={2})
    cb.post_step()
    want = 2 * 256 * (m.num_flops_per_token(256) - _blk(m, 256))
    assert metrics["flop_meter/actual_pflops"] == pytest.approx(want / 1e15)
    assert metrics["flop_meter/dense_pflops"] == pytest.approx(2 * 256 * m.num_flops_per_token(256) / 1e15)


def test_soft_token_compaction_path_charges_only_blocks_that_ran():
    m, cb, metrics = _setup(per_example=False, fixed={2})
    m._soft_token_compaction = {"tokens_in": 512, "tokens_out": 128, "rows": 2}
    cb.post_step()
    want = 128 * (m.num_flops_per_token(64) - _blk(m, 64))
    assert metrics["flop_meter/actual_pflops"] == pytest.approx(want / 1e15)


def test_lora_without_router_charges_frozen_base_and_adapters():
    """--lora-rank modifier (no online router): blocks [0, B) forward-only, [B, K) frozen base + adapters."""
    from olmo_core.nn.early_lora import early_lora_flops_per_block, install_early_lora

    m, cb, metrics = _setup(per_example=False)
    install_early_lora(m, 3, 4, scale=8.0, start=1)
    cb.post_step()
    blocks = list(m.blocks.values())
    f = [b.num_flops_per_token(256) for b in blocks]
    ab = early_lora_flops_per_block(m)
    adj = -(2 / 3) * f[0] + sum(-(1 / 3) * f[i] + ab[i] for i in (1, 2))
    want = 2 * 256 * (m.num_flops_per_token(256) + adj)
    assert ab[0] == 0.0 and ab[3] == 0.0 and ab[1] > 0
    assert metrics["flop_meter/actual_pflops"] == pytest.approx(want / 1e15)


def test_sparse_loss_positions_charges_the_head_only_at_supervised_positions():
    import torch

    m, cb, metrics = _setup(per_example=False)
    m.lm_head._sparse_loss_positions = True
    ids = torch.randint(1, 1000, (2, 256))
    mask = torch.zeros(2, 256, dtype=torch.bool)
    mask[:, -10:] = True  # 10 labelled tokens per row -> 10 supervised positions after the shift
    cb.pre_step({"input_ids": ids, "label_mask": mask})
    cb.post_step()
    sup = 2 * 10
    head = m.lm_head.num_flops_per_token(256)
    want = 2 * 256 * m.num_flops_per_token(256) - head * (2 * 256 - sup)
    assert metrics["flop_meter/actual_pflops"] == pytest.approx(want / 1e15)
    assert metrics["flop_meter/supervised_frac"] == pytest.approx(sup / 512)


def test_dry_run_flop_adjustments_are_not_charged_to_step_one():
    import torch

    m, cb, metrics = _setup(per_example=False)
    m._flop_extra_actual = -1e15  # logged by the dry-run batch before step 1
    cb._trainer.global_step = 1
    cb.pre_step({"input_ids": torch.ones(2, 256, dtype=torch.long)})
    cb.post_step()
    assert metrics["flop_meter/actual_pflops"] == pytest.approx(2 * 256 * m.num_flops_per_token(256) / 1e15)


def test_lora_frozen_head_is_charged_at_two_thirds():
    from olmo_core.nn.early_lora import early_lora_flops_per_block, install_early_lora, lora_flop_adjust_per_token

    m, cb, metrics = _setup(per_example=False)
    install_early_lora(m, 4, 4, scale=8.0, freeze_head=True)
    assert not any(p.requires_grad for p in m.lm_head.parameters())
    cb.post_step()
    head = m.lm_head.num_flops_per_token(256)
    want = 2 * 256 * (m.num_flops_per_token(256) + lora_flop_adjust_per_token(m, 256) - head / 3)
    assert metrics["flop_meter/actual_pflops"] == pytest.approx(want / 1e15)
