"""Contracts for additive and claim-boundary LeakySFT, before implementation."""
import copy
import pytest
import torch
import torch.nn.functional as F
from integration_test import make_model
from olmo_core.optim import AdamWConfig
from olmo_core.train.train_module.transformer import TransformerTrainModuleConfig

class CharacterTokenizer:
    def decode(self, ids, **kwargs): return ''.join(chr(i) for i in ids)
    def __call__(self, text, **kwargs):
        return {'input_ids':[ord(c) for c in text], 'offset_mapping':[(i,i+1) for i in range(len(text))]}


def test_claim_boundaries_are_target_aligned_and_answer_safe():
    from olmo_core.data.contradiction import claim_boundary_mask
    text='Task\n\nClaim 1: Café. Mention Claim 9: inline.\n\nClaim 2: Other.\nAnswer\nClaim 3: answer text'
    ids=torch.tensor([[ord(c) for c in text]])
    outputs=torch.zeros_like(ids,dtype=torch.bool);outputs[:,text.index('Answer'):]=True
    valid=torch.ones_like(outputs)
    got=claim_boundary_mask(ids,outputs,valid,CharacterTokenizer())
    assert got.nonzero().tolist()==[[0,text.index('Claim 1:')],[0,text.index('Claim 2:')]]
    assert not (got & outputs).any()
    assert torch.equal(ids,torch.tensor([[ord(c) for c in text]]))


def test_boundary_detector_rejects_misalignment_and_missing_headers():
    from olmo_core.data.contradiction import claim_boundary_mask
    ids=torch.tensor([[ord(c) for c in 'Claim 1: text']]);valid=torch.ones_like(ids,dtype=torch.bool)
    class Bad(CharacterTokenizer):
        def __call__(self,text,**kwargs):
            data=super().__call__(text,**kwargs);data['input_ids'][0]=999;return data
    with pytest.raises(ValueError,match='round.trip'):
        claim_boundary_mask(ids,~valid,valid,Bad())
    missing=torch.tensor([[ord(c) for c in 'no claim headings']])
    with pytest.raises(ValueError,match='claim'):
        claim_boundary_mask(missing,torch.zeros_like(missing,dtype=torch.bool),torch.ones_like(missing,dtype=torch.bool),CharacterTokenizer())


@pytest.mark.parametrize('additive',[False,True])
@pytest.mark.parametrize('alpha',[0.,.1,.7,1.])
@pytest.mark.parametrize('masked',[False,True])
@pytest.mark.parametrize('microbatch',[8,16])
def test_variant_loss_and_gradients(monkeypatch,additive,alpha,masked,microbatch):
    torch.manual_seed(291)
    kwargs={'additive_leaky_sft_alpha' if additive else 'normalized_leaky_sft_alpha':alpha}
    if masked:kwargs['cpt_claim_boundary_tokenizer']='test-tokenizer'
    module=TransformerTrainModuleConfig(rank_microbatch_size=microbatch,max_sequence_length=8,
        optim=AdamWConfig(),**kwargs).build(make_model(),device=torch.device('cpu'))
    ref=copy.deepcopy(module.model)
    ids=torch.arange(1,17).reshape(2,8);output=torch.zeros_like(ids,dtype=torch.bool)
    output[0,-1:]=True;output[1,-4:]=True
    valid=torch.ones_like(output);valid[0,2]=False
    boundaries=torch.zeros_like(output);boundaries[0,3]=True;boundaries[1,2]=True
    if masked:
        monkeypatch.setattr(module,'_claim_boundary_mask',lambda batch: boundaries)
    ce=F.cross_entropy(ref(ids)[:,:-1].reshape(-1,32),ids[:,1:].reshape(-1),reduction='none').reshape(2,7)
    answers=output[:,1:] & valid[:,1:]
    cpt=valid[:,1:] & (~boundaries[:,1:] if masked else True)
    expected=(1 if additive else 1-alpha)*ce[answers].mean()+alpha*ce[cpt].mean()
    expected.backward();metrics={}
    module.record_metric=lambda name,value,*a,**kw:metrics.update({name:value})
    module.record_ce_loss=lambda value,*a,**kw:metrics.update(CE=value)
    module.train_batch(dict(input_ids=ids,label_mask=output,loss_valid_mask=valid))
    torch.testing.assert_close(metrics['CE'],expected.detach())
    for p,q in zip(module.model.parameters(),ref.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=3e-4,atol=3e-6)


@pytest.mark.parametrize('alpha',[-.1,float('nan'),float('inf')])
def test_bad_additive_alpha(alpha):
    with pytest.raises(ValueError,match='additive_leaky_sft_alpha'):
        TransformerTrainModuleConfig(rank_microbatch_size=8,max_sequence_length=8,
            optim=AdamWConfig(),additive_leaky_sft_alpha=alpha).build(make_model(),device=torch.device('cpu'))


def test_additive_supports_alpha_above_one():
    module=TransformerTrainModuleConfig(rank_microbatch_size=8,max_sequence_length=8,
        optim=AdamWConfig(),additive_leaky_sft_alpha=2.).build(make_model(),device=torch.device('cpu'))
    assert module.additive_leaky_sft_alpha==2.


def test_variant_selection_is_explicit():
    with pytest.raises(ValueError,match='mutually exclusive'):
        TransformerTrainModuleConfig(rank_microbatch_size=8,max_sequence_length=8,
            optim=AdamWConfig(),additive_leaky_sft_alpha=.1,normalized_leaky_sft_alpha=.1).build(make_model(),device=torch.device('cpu'))
    with pytest.raises(ValueError,match='boundary'):
        TransformerTrainModuleConfig(rank_microbatch_size=8,max_sequence_length=8,
            optim=AdamWConfig(),cpt_claim_boundary_tokenizer='unused').build(make_model(),device=torch.device('cpu'))
