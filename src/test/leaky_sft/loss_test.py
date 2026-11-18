"""Frozen LeakySFT contract: independently compute the weighted token objective."""
import copy

import pytest
import torch
import torch.nn.functional as F

from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation


def reference(head, x, labels, output, alpha):
    logits = head.w_out(x)
    ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none").view_as(labels)
    valid = labels != -100
    weights = torch.where(output, 1.0, alpha) * valid
    return (ce * weights).sum() / weights.sum().clamp_min(1e-30), ce


@pytest.mark.parametrize("alpha", [0.0, 0.001, 0.1, 1.0])
@pytest.mark.parametrize("case", ["mixed", "outputs", "context", "empty"])
def test_loss_gradients_and_diagnostics(alpha, case):
    torch.manual_seed(71)
    head = LMHeadConfig().build(d_model=4, vocab_size=9).double()
    control = copy.deepcopy(head)
    x = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    labels = torch.tensor([[1, 2, 3, -100, 4], [5, 6, 7, 8, -100]])
    output = torch.tensor([[False, True, True, True, False], [False, False, True, True, False]])
    if case == "outputs":
        output.fill_(True)
    elif case == "context":
        output.fill_(False)
    elif case == "empty":
        labels.fill_(-100)
    expected, ce = reference(control, ref_x, labels, output, alpha)
    actual = head(x, labels=labels, output_mask=output, cpt_loss_scaler=alpha, return_logits=False)
    torch.testing.assert_close(actual.loss, expected)
    torch.testing.assert_close(actual.ce_loss, expected.detach())
    torch.testing.assert_close(actual.output_ce_sum, ce[output & (labels != -100)].sum().detach())
    torch.testing.assert_close(actual.context_ce_sum, ce[~output & (labels != -100)].sum().detach())
    assert not actual.context_ce_sum.requires_grad
    actual.loss.backward()
    expected.backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    for p, q in zip(head.parameters(), control.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


@pytest.mark.parametrize("alpha", [0.0, 0.1, 1.0])
def test_accumulation_and_uneven_rank_partition(alpha):
    torch.manual_seed(18)
    head = LMHeadConfig().build(d_model=4, vocab_size=9).double()
    control = copy.deepcopy(head)
    x = torch.randn(3, 5, 4, dtype=torch.float64, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    labels = torch.tensor([[1, 2, 3, 4, 5], [1, 2, -100, -100, -100], [1, 2, 3, -100, -100]])
    output = torch.tensor([[False]*4+[True], [True]*5, [False, True, True, False, False]])
    denominator = ((labels != -100) * torch.where(output, 1.0, alpha)).sum()
    expected, _ = reference(control, ref_x, labels, output, alpha)
    # Represents uneven microbatches, or summed rank numerators with a common global denominator.
    total = 0
    for lo, hi in [(0, 1), (1, 3)]:
        out = head(x[lo:hi], labels=labels[lo:hi], output_mask=output[lo:hi],
                   cpt_loss_scaler=alpha, loss_reduction="sum", loss_div_factor=denominator)
        total = total + out.loss
    torch.testing.assert_close(total, expected)
    total.backward()
    expected.backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    for p, q in zip(head.parameters(), control.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_scaler_rejected(alpha):
    head = LMHeadConfig().build(d_model=4, vocab_size=9)
    with pytest.raises(ValueError, match="cpt_loss_scaler"):
        head(torch.randn(1, 2, 4), labels=torch.ones(1, 2, dtype=torch.long),
             output_mask=torch.ones(1, 2, dtype=torch.bool), cpt_loss_scaler=alpha)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real Liger kernel requires CUDA")
@pytest.mark.parametrize("alpha", [0.0, 0.1, 1.0])
@pytest.mark.parametrize("case", ["mixed", "context", "empty"])
def test_real_fused_kernel(alpha, case):
    pytest.importorskip("liger_kernel")
    torch.manual_seed(82)
    head = LMHeadConfig(loss_implementation=LMLossImplementation.fused_linear).build(
        d_model=64, vocab_size=128, init_device="cuda")
    control = LMHeadConfig().build(d_model=64, vocab_size=128, init_device="cuda")
    control.load_state_dict(head.state_dict())
    x = torch.randn(2, 16, 64, device="cuda", requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    labels = torch.randint(0, 128, (2, 16), device="cuda")
    labels[:, -2:] = -100
    output = torch.rand(2, 16, device="cuda") > .6
    if case == "context":
        output.fill_(False)
    if case == "empty":
        labels.fill_(-100)
    got = head(x, labels=labels, output_mask=output, cpt_loss_scaler=alpha, return_logits=False)
    expected, _ = reference(control, rx, labels, output, alpha)
    torch.testing.assert_close(got.loss, expected, rtol=2e-4, atol=2e-5)
    got.loss.backward()
    expected.backward()
    torch.testing.assert_close(x.grad, rx.grad, rtol=2e-4, atol=2e-5)
    for p, q in zip(head.parameters(), control.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-5)
