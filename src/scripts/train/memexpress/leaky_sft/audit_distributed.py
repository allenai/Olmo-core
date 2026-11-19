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
results=[]
legacy_grad=None
for alpha in [None,0.,.1,.3,.5,.7,1.]:
    torch.manual_seed(123)
    cfg=TransformerConfig.llama_like(d_model=64,vocab_size=128,n_layers=1,n_heads=2,
        feed_forward=FeedForwardConfig(hidden_size=128,bias=False),attn_backend=AttentionBackendName.torch)
    cfg.lm_head.loss_implementation=LMLossImplementation.fused_linear
    module=TransformerTrainModuleConfig(rank_microbatch_size=32,max_sequence_length=32,
        optim=AdamWConfig(),cpt_loss_scaler=alpha,
        dp_config=TransformerDataParallelConfig(name=DataParallelType.fsdp,shard_degree=2,num_replicas=1,
            param_dtype=DType.bfloat16 if bf16 else DType.float32,reduce_dtype=DType.float32)).build(cfg.build(),device=device)
    cfg.lm_head.loss_implementation=LMLossImplementation.default
    ref=cfg.build().to(device)
    ref.load_state_dict({k:v.full_tensor() if hasattr(v,'full_tensor') else v
                        for k,v in module.model.state_dict().items()})
    if bf16:ref.bfloat16()
    torch.manual_seed(456)
    ids=torch.randint(0,128,(2,32),device=device)
    output=torch.zeros_like(ids,dtype=torch.bool);output[0,-2:]=True;output[1,-20:]=True
    labels=F.pad(ids[:,1:],(0,1),value=-100)
    mask=F.pad(output[:,1:],(0,1),value=False)
    logits=ref(ids)
    ce=F.cross_entropy(logits.flatten(0,1).float(),labels.flatten(),reduction='none')
    if alpha is None:
        loss=((ce.view_as(mask)*mask).sum(1)/mask.sum(1)).mean()
    else:
        weights=torch.where(mask,1.,alpha)*(labels!=-100)
        loss=(ce*weights.flatten()).sum()/weights.sum()
    loss.backward()
    metrics={};module.record_metric=lambda *a,**kw:None
    module.record_ce_loss=lambda value,*a,**kw:metrics.update(loss=float(value))
    module.train_batch(dict(input_ids=ids[rank:rank+1],label_mask=output[rank:rank+1],
        loss_valid_mask=torch.ones_like(output[rank:rank+1])))
    worst=0.
    for (name,p),(other,q) in zip(module.model.named_parameters(),ref.named_parameters()):
        assert name==other
        grad=p.grad.full_tensor() if hasattr(p.grad,'full_tensor') else p.grad
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
    results.append(dict(alpha=alpha,max_relative_grad_error=worst,loss=metrics['loss'],relative_to_legacy_sft=difference))
    del module,ref
    torch.cuda.empty_cache()
if rank==0:
    print(json.dumps(results,indent=2))
    open('/results/distributed-audit'+('-bf16' if bf16 else '-fp32')+'.json','w').write(json.dumps(results,indent=2))
dist.destroy_process_group()
