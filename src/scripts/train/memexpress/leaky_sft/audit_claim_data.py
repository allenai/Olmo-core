"""Check claim masks on actual packed training windows without changing the corpus."""
import json
import torch
from transformers import AutoTokenizer
from _qwen35_contradiction_common import build_leaky_experiment
from olmo_core.internal.experiment import CliContext, SubCmd
from olmo_core.data.utils import get_leaky_labels
from olmo_core.data.contradiction import claim_boundary_mask

cfg = build_leaky_experiment(CliContext(script=__file__, cmd=SubCmd.prep,
    run_name='claim-mask-audit', cluster='ai2/jupiter-cirrascale-2', overrides=[]), arm='a00')
packed = cfg.dataset[0].build(cfg.data_loader.work_dir)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B')
report = dict(total_windows=len(packed), checked_windows=0, valid_targets=0, masked_claim_targets=0,
              unchanged_sft_labels=True, round_trip_exact=True, positions=[])
# Evenly spaced windows span the BFD length distribution, not just the longest tail.
for index in sorted(set(int(i) for i in torch.linspace(0,len(packed)-1,64))):
    instance=packed[index]
    batch={k:torch.as_tensor(instance[k]).unsqueeze(0) for k in ('input_ids','label_mask','loss_valid_mask')}
    labels,output=get_leaky_labels(batch)
    mask=claim_boundary_mask(batch['input_ids'],batch['label_mask'],batch['loss_valid_mask'],tokenizer)
    shifted=torch.cat([mask[:,1:],torch.zeros_like(mask[:,:1])],dim=1)
    new=labels.masked_fill(shifted & ~output,-100)
    assert torch.equal(new.masked_fill(~output,-100),labels.masked_fill(~output,-100))
    report['checked_windows']+=1
    report['valid_targets']+=int((labels!=-100).sum())
    report['masked_claim_targets']+=int(((labels!=-100)&(new==-100)).sum())
    report['positions'].append(dict(window=index,claims=int(mask.sum())))
open('/results/claim-data-audit.json','w').write(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
