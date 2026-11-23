"""Queue both requested sweeps, reusing the existing NormalizedLeakySFT alpha=0 control."""
import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from launch_sweep import make_plan as original_plan
from _variant_common import VARIANTS

CONTROL = '/weka/oe-training-default/ai2-llm/checkpoints/amandab/q35-normalized-leakysft-contra-filter32k-1ep-2gpu-20261119-a00/step2493'
FAMILY = 'src/scripts/train/memexpress/leaky_sft'


def make_plan(prefix, ref, output_dir):
    plan = []
    for variant, method in VARIANTS.items():
        for row in original_plan(prefix+'-'+variant, ref, output_dir, normalized=True):
            if row['arm'] == 'a00':
                continue
            alpha = row['normalized_leaky_sft_alpha']
            row.update(method=method,variant=variant,reused_alpha_zero_control=CONTROL,
                       additive_leaky_sft_alpha=alpha if variant=='additive' else None,
                       normalized_leaky_sft_alpha=alpha if variant=='boundary' else None,
                       cpt_claim_boundary_tokenizer='Qwen/Qwen3.5-0.8B' if variant=='boundary' else None)
            command = row['command']
            command[command.index('--')+1] = f'{FAMILY}/Qwen3.5-4B-contradiction-32k-1ep-{method}-{row["arm"]}.py'
            plan.append(row)
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref',required=True)
    parser.add_argument('--prefix',default='q35-leakysft-variants-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.prefix):parser.error('Invalid prefix')
    if not re.fullmatch(r'[0-9a-f]{40}',args.ref):parser.error('Full pushed source SHA required')
    if args.submit:
        subprocess.run(['git','diff','--exit-code',args.ref,'--','src/olmo_core',FAMILY],check=True)
        assert not subprocess.check_output(['git','ls-files','--others','--exclude-standard','--','src/olmo_core',FAMILY]).strip(), 'Commit all required source first'
    args.output_dir.mkdir(parents=True,exist_ok=False)
    plan=make_plan(args.prefix,args.ref,args.output_dir)
    (args.output_dir/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    if args.submit:
        with (args.output_dir/'launch-ledger.jsonl').open('a') as ledger:
            for row in plan:
                row['status']='submitting';ledger.write(json.dumps(row)+'\n');ledger.flush()
                log=args.output_dir/(row['run_name']+'.log')
                with log.open('w') as f:result=subprocess.run(row['command'],stdout=f,stderr=subprocess.STDOUT)
                ids=re.findall(r'https://beaker.org/ex/([A-Z0-9]+)',log.read_text())
                row.update(status='submitted' if result.returncode==0 else 'submission_failed',submission_log=str(log),experiment_id=ids[-1] if ids else None)
                ledger.write(json.dumps(row)+'\n');ledger.flush()
                print(row['variant'],row['arm'],row['status'],row['experiment_id'],flush=True)
                result.check_returncode()

if __name__=='__main__':main()
