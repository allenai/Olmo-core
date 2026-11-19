"""Additional audit cases; original frozen tests are unchanged."""
import copy
import pytest
import torch
import torch.nn.functional as F
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation


@pytest.mark.parametrize('alpha', [0.0, 0.1, 0.3, 0.5, 0.7, 1.0])
def test_sparse_answer_weight_mass(alpha):
    answers, context = 30, 30000
    answer_share = answers / (answers + alpha * context)
    if alpha == 0:
        assert answer_share == 1
    else:
        assert answer_share < 0.01  # alpha=.1 is already >99% context weight.


def test_zero_endpoint_rank_normalization_is_not_legacy_sft():
    # Two DP ranks with 1 and 9 answer tokens, with unequal mean losses.
    sums = torch.tensor([2., 9.], requires_grad=True)
    counts = torch.tensor([1., 9.])
    legacy = (sums / counts).mean()
    leaky = sums.sum() / counts.sum()
    assert legacy == 1.5 and leaky == 1.1
    assert not torch.equal(torch.autograd.grad(legacy, sums)[0],
                           torch.autograd.grad(leaky, sums)[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/Liger required')
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('alpha', [0.0, 0.1, 0.3, 0.5, 0.7, 1.0])
def test_sparse_fused_gradients(dtype, alpha):
    torch.manual_seed(918)
    head = LMHeadConfig(loss_implementation=LMLossImplementation.fused_linear).build(
        d_model=64, vocab_size=4096, init_device='cuda').to(dtype)
    ref = copy.deepcopy(head)
    x = torch.randn(1, 2048, 64, device='cuda', dtype=dtype, requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    labels = torch.randint(0, 4096, (1, 2048), device='cuda'); labels[:, -10:] = -100
    outputs = torch.zeros_like(labels, dtype=torch.bool); outputs[:, 2000:2030] = True
    got = head(x, labels=labels, output_mask=outputs, cpt_loss_scaler=alpha, return_logits=False)
    logits = ref.w_out(rx).float()
    ce = F.cross_entropy(logits.flatten(0,1), labels.flatten(), reduction='none')
    weights = torch.where(outputs, 1., alpha) * (labels != -100)
    expected = (ce * weights.flatten()).sum() / weights.sum()
    got.loss.backward(); expected.backward()
    torch.testing.assert_close(got.loss, expected, rtol=.002, atol=.002)
    for actual, wanted in [(x.grad, rx.grad)]+[(p.grad,q.grad) for p,q in zip(head.parameters(),ref.parameters())]:
        rel = (actual.float()-wanted.float()).norm()/wanted.float().norm().clamp_min(1e-20)
        assert torch.isfinite(actual).all() and rel < (.025 if dtype==torch.bfloat16 else .001), rel.item()
