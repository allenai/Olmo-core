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


@pytest.mark.parametrize('alpha', [0.0, 0.1, 1.0])
def test_nonfused_fp16_sum_does_not_overflow(alpha):
    head = LMHeadConfig().build(d_model=4,vocab_size=32).half()
    with torch.no_grad():
        head.w_out.weight.zero_()
        if head.w_out.bias is not None:head.w_out.bias.zero_()
    x=torch.zeros(1,32768,4,dtype=torch.float16,requires_grad=True)
    labels=torch.ones(1,32768,dtype=torch.long)
    output=torch.zeros_like(labels,dtype=torch.bool);output[:,-30:]=True
    result=head(x,labels=labels,output_mask=output,cpt_loss_scaler=alpha,return_logits=False)
    assert torch.isfinite(result.context_ce_sum)
    torch.testing.assert_close(result.loss.float(),torch.tensor(32.).log(),rtol=1e-5,atol=1e-5)
    result.loss.backward()
    assert torch.isfinite(head.w_out.weight.grad).all()


@pytest.mark.parametrize('enabled,scale,expected', [(False,1e-4,0.),(True,1e-4,1e-4),(True,0.,0.)])
def test_fused_wrapper_disables_z_loss_in_kernel(monkeypatch,enabled,scale,expected):
    import importlib
    mod=importlib.import_module('olmo_core.nn.functional.cross_entropy_loss')
    seen={}
    def fake(*args):
        seen['scale']=args[6]
        return torch.tensor(1.),torch.tensor(0.),None
    monkeypatch.setattr(mod,'_fused_linear_cross_entropy_loss',fake)
    mod.fused_linear_cross_entropy_loss(torch.zeros(2,4),torch.zeros(3,4),torch.ones(2,dtype=torch.long),compute_z_loss=enabled,z_loss_multiplier=scale)
    assert seen['scale']==expected


def test_explicit_zero_z_loss_stays_zero():
    head=LMHeadConfig().build(d_model=4,vocab_size=16)
    x=torch.randn(1,8,4);labels=torch.ones(1,8,dtype=torch.long)
    result=head(x,labels=labels,z_loss_multiplier=0.,return_logits=False)
    assert result.z_loss.item()==0.


def test_empty_fp16_groups_do_not_overflow_zero_anchor():
    head=LMHeadConfig().build(d_model=4,vocab_size=32768).half()
    with torch.no_grad():head.w_out.weight.fill_(1.)
    x=torch.ones(1,4,4,dtype=torch.float16,requires_grad=True)
    labels=torch.full((1,4),-100,dtype=torch.long)
    result=head(x,labels=labels,output_mask=torch.zeros_like(labels,dtype=torch.bool),cpt_loss_scaler=.1,return_logits=False)
    assert result.loss.item()==0.
    result.loss.backward()
    assert torch.count_nonzero(head.w_out.weight.grad)==0
