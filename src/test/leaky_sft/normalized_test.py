"""Frozen contract for NormalizedLeakySFT, written before implementation."""
import copy

import pytest
import torch
import torch.nn.functional as F

from olmo_core.optim import AdamWConfig
from olmo_core.train.train_module.transformer import TransformerTrainModuleConfig
from integration_test import make_model


@pytest.mark.parametrize('alpha', [0., .1, .3, .5, .7, 1.])
@pytest.mark.parametrize('microbatch', [8, 16])
def test_normalized_full_batch_loss_and_gradients(alpha, microbatch):
    torch.manual_seed(125)
    module = TransformerTrainModuleConfig(rank_microbatch_size=microbatch,
        max_sequence_length=8, optim=AdamWConfig(),
        normalized_leaky_sft_alpha=alpha).build(make_model(), device=torch.device('cpu'))
    ref = copy.deepcopy(module.model)
    ids = torch.arange(1, 17).reshape(2, 8)
    output = torch.zeros_like(ids, dtype=torch.bool)
    output[0, -1:] = True
    output[1, -5:] = True
    valid = torch.ones_like(output)
    valid[:, 0] = False
    valid[1, 2] = False  # excluded structural target must not enter either mean
    ce = F.cross_entropy(ref(ids)[:, :-1].reshape(-1, 32), ids[:, 1:].reshape(-1),
                         reduction='none').reshape(2, 7)
    answers = output[:, 1:] & valid[:, 1:]
    expected = (1-alpha)*ce[answers].mean() + alpha*ce[valid[:, 1:]].mean()
    expected.backward()
    metrics = {}
    module.record_metric = lambda name, value, *a, **kw: metrics.update({name: value})
    module.record_ce_loss = lambda value, *a, **kw: metrics.update(CE=value)
    module.train_batch(dict(input_ids=ids, label_mask=output, loss_valid_mask=valid))
    torch.testing.assert_close(metrics['CE'], expected.detach())
    for p, q in zip(module.model.parameters(), ref.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-6)
    context_contribution = alpha*ce[valid[:, 1:] & ~answers].sum()/valid[:, 1:].sum()
    torch.testing.assert_close(metrics['normalized_leaky/context_loss_fraction'],
                               (context_contribution/expected).detach())


@pytest.mark.parametrize('alpha', [0., .1, 1.])
@pytest.mark.parametrize('all_invalid', [False, True])
def test_normalized_empty_answers_are_finite(alpha, all_invalid):
    module = TransformerTrainModuleConfig(rank_microbatch_size=8, max_sequence_length=8,
        optim=AdamWConfig(), normalized_leaky_sft_alpha=alpha).build(make_model(), device=torch.device('cpu'))
    module.record_metric = lambda *a, **kw: None
    got = []
    module.record_ce_loss = lambda value, *a, **kw: got.append(value)
    ids = torch.arange(1,9).reshape(1,8)
    module.train_batch(dict(input_ids=ids, label_mask=torch.zeros_like(ids,dtype=torch.bool),
                           loss_valid_mask=torch.full_like(ids,not all_invalid,dtype=torch.bool)))
    assert torch.isfinite(got[0])
    for p in module.model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        if all_invalid or alpha == 0:
            assert torch.count_nonzero(p.grad) == 0


@pytest.mark.parametrize('alpha', [-.1, 1.1, float('nan'), float('inf')])
def test_normalized_invalid_alpha(alpha):
    with pytest.raises(ValueError, match='normalized_leaky_sft_alpha'):
        TransformerTrainModuleConfig(rank_microbatch_size=8,max_sequence_length=8,
            optim=AdamWConfig(),normalized_leaky_sft_alpha=alpha).build(make_model(),device=torch.device('cpu'))


def test_methods_are_mutually_exclusive():
    with pytest.raises(ValueError, match='mutually exclusive'):
        TransformerTrainModuleConfig(rank_microbatch_size=8,max_sequence_length=8,
            optim=AdamWConfig(),cpt_loss_scaler=.1,normalized_leaky_sft_alpha=.1).build(make_model(),device=torch.device('cpu'))
