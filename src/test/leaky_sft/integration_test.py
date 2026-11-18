"""Exercise the real transformer and train_batch, including gradient accumulation."""
import copy

import pytest
import torch
import torch.nn.functional as F

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig
from olmo_core.train.train_module.transformer import TransformerTrainModuleConfig


def make_model():
    return TransformerConfig.llama_like(d_model=16, vocab_size=32, n_layers=1, n_heads=2,
        feed_forward=FeedForwardConfig(hidden_size=32, bias=False),
        attn_backend=AttentionBackendName.torch).build()


@pytest.mark.parametrize("alpha", [0.0, 0.1, 1.0])
@pytest.mark.parametrize("microbatch", [8, 16])
def test_train_batch_matches_full_batch_reference(alpha, microbatch):
    torch.manual_seed(12)
    model = make_model()
    cfg = TransformerTrainModuleConfig(rank_microbatch_size=microbatch, max_sequence_length=8,
                                        optim=AdamWConfig(lr=1e-4), cpt_loss_scaler=alpha)
    module = cfg.build(model, device=torch.device("cpu"))
    reference = copy.deepcopy(module.model)
    metrics = {}
    module.record_metric = lambda name, value, *a, **kw: metrics.update({name: value})
    module.record_ce_loss = lambda value, *a, **kw: metrics.update({"CE": value})
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]])
    output = torch.tensor([[False]*6+[True]*2, [False]+[True]*7])
    batch = dict(input_ids=ids.clone(), label_mask=output.clone(), loss_valid_mask=torch.ones_like(output))
    logits = reference(ids)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, 32), ids[:, 1:].reshape(-1), reduction="none").view(2, 7)
    weights = torch.where(output[:, 1:], 1.0, alpha)
    expected = (ce * weights).sum() / weights.sum()
    expected.backward()
    module.train_batch(batch)
    torch.testing.assert_close(metrics["CE"], expected.detach())
    for p, q in zip(module.model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-6)
    assert "leaky/output_ce" in metrics and "leaky/context_ce" in metrics
    torch.testing.assert_close(metrics["leaky/output_ce"], ce[output[:, 1:]].mean().detach())
    torch.testing.assert_close(metrics["leaky/context_ce"], ce[~output[:, 1:]].mean().detach())


def test_legacy_config_is_opt_in():
    cfg = TransformerTrainModuleConfig(rank_microbatch_size=8, max_sequence_length=8, optim=AdamWConfig())
    assert cfg.cpt_loss_scaler is None


@pytest.mark.parametrize("alpha", [-1.0, float("nan"), 1.1])
def test_invalid_config_fails_before_training(alpha):
    with pytest.raises(ValueError, match="cpt_loss_scaler"):
        TransformerTrainModuleConfig(rank_microbatch_size=8, max_sequence_length=8,
            optim=AdamWConfig(), cpt_loss_scaler=alpha).build(make_model(), device=torch.device("cpu"))


def test_context_parallel_shards_output_mask_with_labels():
    class Balancer:
        def batch_shard(self, *, inputs, seq_dims, pad_values):
            # Noncontiguous sequence partition: catches naive contiguous slicing.
            return [x.index_select(dim, torch.tensor([0, 2, 5, 7]))
                    for x, dim in zip(inputs, seq_dims)]

    model = make_model()
    model._cp_load_balancer = Balancer()
    ids = torch.arange(8).view(1, 8)
    output = torch.tensor([[False, True, False, True, True, False, False, True]])
    _, labels, _, _, kwargs = model._prepare_inputs(ids, ids.clone(),
                       output_mask=output, cpt_loss_scaler=.1)
    torch.testing.assert_close(labels, ids[:, [0, 2, 5, 7]])
    torch.testing.assert_close(kwargs["output_mask"], output[:, [0, 2, 5, 7]])
    assert kwargs["cpt_loss_scaler"] == .1
