"""Measure exact loss-token weights on the packed contradiction stream used by the sweep."""
import json
import numpy as np
import torch
from _qwen35_contradiction_common import build_leaky_experiment
from olmo_core.internal.experiment import CliContext,SubCmd
from olmo_core.data.utils import get_leaky_labels,get_labels
c=build_leaky_experiment(CliContext(script=__file__,cmd=SubCmd.prep,run_name='leaky-audit',cluster='ai2/jupiter-cirrascale-2',overrides=[]),arm='a00')
packed=c.dataset[0].build(c.data_loader.work_dir);loader=c.data_loader.build(packed);loader.reshuffle(epoch=1)
counts=[];different=0
for batch in loader:
 labels,outputs=get_leaky_labels(batch);valid=labels!=-100
 answer=(valid & outputs).sum(-1).tolist();context=(valid & ~outputs).sum(-1).tolist()
 assert torch.equal(labels.masked_fill(~outputs,-100),get_labels(batch)), 'alpha-zero label mismatch'
 counts.extend(zip(answer,context));different+=answer[0]!=answer[1]
a=np.asarray(counts);total=a.sum(0);report=dict(packed_windows=len(packed),answer_tokens=int(total[0]),context_tokens=int(total[1]),context_per_answer=float(total[1]/total[0]),two_rank_batches_unequal_answer_counts=different,total_batches=len(loader),answer_tokens_per_window_quantiles=np.quantile(a[:,0],[0,.25,.5,.75,1]).tolist(),scalers={})
for alpha in [0,.001,.01,.1,.3,.5,.7,1]:
 report['scalers'][str(alpha)]=dict(answer_weight_fraction=float(total[0]/(total[0]+alpha*total[1])),context_weight_fraction=float(alpha*total[1]/(total[0]+alpha*total[1])))
open('/results/data-audit.json','w').write(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
try:
 import wandb
 api=wandb.Api(); summaries={}
 for arm,rid in dict(a00='q6a4ec5d',a01='ij79yb5u',a03='orqcau6i',a05='whvyweyi',a07='6d9u105k',a10='2xklj09p').items():
  run=api.run('ai2-llm/memory-networks/'+rid)
  summaries[arm]={k:v for k,v in dict(run.summary).items() if 'leaky' in k or k in ['train/CE loss','optim/step skipped']}
 open('/results/run-summaries.json','w').write(json.dumps(summaries,indent=2));print(json.dumps(summaries,indent=2))
except Exception as e:print('WandB summary fetch unavailable:',type(e).__name__,str(e))
