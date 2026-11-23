"""Two-GPU FSDP gradient audit against a full-batch, explicit weighted-CE reference."""
import os,json
import torch
import torch.distributed as dist
import torch.nn.functional as F
from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.distributed.utils import init_distributed
from olmo_core.optim import AdamWConfig
from olmo_core.train.train_module import TransformerTrainModuleConfig,TransformerDataParallelConfig

init_distributed()
rank=dist.get_rank(); device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
bf16=os.environ.get("AUDIT_BF16")=="1"
normalized=os.environ.get("AUDIT_NORMALIZED")=="1"
additive=os.environ.get("AUDIT_ADDITIVE")=="1"
boundary_masked=os.environ.get("AUDIT_CLAIM_MASK")=="1"
assert not (additive or boundary_masked) or normalized
results=[]
legacy_grad=None
for alpha in [None,0.,.1,.3,.5,.7,1.]:
    torch.manual_seed(123)
    cfg=TransformerConfig.llama_like(d_model=64,vocab_size=128,n_layers=1,n_heads=2,
        feed_forward=FeedForwardConfig(hidden_size=128,bias=False),attn_backend=AttentionBackendName.torch)
    cfg.lm_head.loss_implementation=LMLossImplementation.fused_linear
    if alpha is None:
        module=TransformerTrainModuleConfig(rank_microbatch_size=32,max_sequence_length=32,
            optim=AdamWConfig(),cpt_loss_scaler=alpha,
            dp_config=TransformerDataParallelConfig(name=DataParallelType.fsdp,shard_degree=2,num_replicas=1,
                param_dtype=DType.bfloat16 if bf16 else DType.float32,reduce_dtype=DType.float32)).build(cfg.build(),device=device)
    else:
        module.optim.zero_grad(set_to_none=True)
        if normalized:
            if additive:module.additive_leaky_sft_alpha=alpha
            else:module.normalized_leaky_sft_alpha=alpha
            if boundary_masked:module.cpt_claim_boundary_tokenizer="audit"
        else:
            module.cpt_loss_scaler=alpha
    cfg.lm_head.loss_implementation=LMLossImplementation.default
    ref=cfg.build().to(device)
    ref.load_state_dict({k:v.full_tensor() if hasattr(v,'full_tensor') else v
                        for k,v in module.model.state_dict().items()})
    if bf16:ref.bfloat16()
    torch.manual_seed(456)
    ids=torch.randint(0,128,(2,32),device=device)
    output=torch.zeros_like(ids,dtype=torch.bool);output[0,-2:]=True;output[1,-20:]=True
    if normalized:
        # Two microbatches per rank, with unequal counts within and between ranks.
        ids=torch.cat([ids,ids.roll(1,dims=1)])
        output=torch.zeros_like(ids,dtype=torch.bool)
        for i,n in enumerate([1,3,15,25]):output[i,-n:]=True
    labels=F.pad(ids[:,1:],(0,1),value=-100)
    mask=F.pad(output[:,1:],(0,1),value=False)
    logits=ref(ids)
    ce=F.cross_entropy(logits.flatten(0,1).float(),labels.flatten(),reduction='none')
    if normalized:
        answer_sums=(ce.view_as(mask)*mask).reshape(2,-1).sum(1)
        answer_counts=mask.reshape(2,-1).sum(1)
        sft=(answer_sums/answer_counts).mean()
        cpt_valid=labels!=-100
        if boundary_masked:
            boundary=torch.zeros_like(output);boundary[:,3]=True
            cpt_valid &= ~F.pad(boundary[:,1:],(0,1),value=False)
            module._claim_boundary_mask=lambda batch: boundary[:len(batch['input_ids'])]
        cpt=((ce.view_as(mask)*cpt_valid).reshape(2,-1).sum(1)/cpt_valid.reshape(2,-1).sum(1)).mean()
        loss=sft if alpha is None else (1 if additive else 1-alpha)*sft+alpha*cpt
    elif alpha is None:
        loss=((ce.view_as(mask)*mask).sum(1)/mask.sum(1)).mean()
    else:
        weights=torch.where(mask,1.,alpha)*(labels!=-100)
        loss=(ce*weights.flatten()).sum()/weights.sum()
    loss.backward()
    metrics={};module.record_metric=lambda *a,**kw:None
    module.record_ce_loss=lambda value,*a,**kw:metrics.update(loss=float(value))
    per_rank=2 if normalized else 1
    sl=slice(rank*per_rank,(rank+1)*per_rank)
    module.train_batch(dict(input_ids=ids[sl],label_mask=output[sl],
        loss_valid_mask=torch.ones_like(output[sl])))
    worst=0.
    actual_grads=[]
    for (name,p),(other,q) in zip(module.model.named_parameters(),ref.named_parameters()):
        assert name==other
        grad=p.grad.full_tensor() if hasattr(p.grad,'full_tensor') else p.grad
        actual_grads.append(grad.detach().float().flatten())
        rel=(grad.float()-q.grad.float()).norm()/q.grad.float().norm().clamp_min(1e-20)
        worst=max(worst,rel.item())
    assert worst<(.04 if bf16 else .001),(alpha,worst)
    if alpha is None:
        # Legacy AdamW defers mean-reduction of logged CE to Trainer.
        logged=torch.tensor(metrics['loss'],device=device)
        dist.all_reduce(logged);metrics['loss']=float(logged/dist.get_world_size())
    assert abs(metrics['loss']-float(loss))<(.02 if bf16 else 1e-4),(alpha,metrics['loss'],float(loss))
    flat=torch.cat([p.grad.flatten() for p in ref.parameters()])
    if alpha is None:legacy_grad=flat.detach().clone()
    difference=float((flat-legacy_grad).norm()/legacy_grad.norm())
    actual_flat=torch.cat(actual_grads)
    if alpha is None:legacy_actual=actual_flat.clone()
    if normalized and alpha == 0:
        endpoint_error=float((actual_flat-legacy_actual).norm()/legacy_actual.norm())
        assert endpoint_error < (.01 if bf16 else 1e-5), endpoint_error
    results.append(dict(alpha=alpha,max_relative_grad_error=worst,loss=metrics['loss'],relative_to_legacy_sft=difference))
    del ref
    torch.cuda.empty_cache()
if rank==0:
    print(json.dumps(results,indent=2))
    open('/results/distributed-audit'+('-additive' if additive else '-normalized' if normalized else '')+('-boundary' if boundary_masked else '')+('-bf16' if bf16 else '-fp32')+'.json','w').write(json.dumps(results,indent=2))
dist.destroy_process_group()
