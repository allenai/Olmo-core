import json,os,re,collections,sys,datetime
idx=os.path.expanduser('~/.cache/storage-audit/weka.json')
d=json.load(open(idx)); now=d['generated']; TiB=2**40
R='/weka/oe-training-default/ai2-llm/checkpoints/prasanns/'; C=R+'ctc_suite/ckpts/'
ROOTREPO='/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/'
runs=json.load(open('runs.json'))
# experiment status from INDEX
st={};sec=None
for line in open('/accounts/projects/berkeleynlp/prasann/ops/experiments/INDEX.md'):
    m=re.match(r'## (\w+)',line); sec=m.group(1) if m else sec
    m=re.search(r'\[(\w+)\]\(',line)
    if m and sec: st[m.group(1)]=sec
RETIRED_PREFIX=r'^(lm3-|q3-|q4b-|q06b-)'   # legacy landmark, retired 2026-10-13
PROTECT=[r"/ctc_suite/bases(/|$)",r"/ctc_olmo3/bases(/|$)",r"xlong5_2k256k_qwen35",r"contra_mix_",r"_eval_bundle",r"ctc-olmo3-7b-hpqaret-",r"vllm_serving",r"/shards(/|$)",r"_inventory",r"_olmoeval_ctc",r"_eval_results",r"cr_suite_data",r"ctc_sft_sets",r"hf_models",r"_transfer"]
exp_of=collections.defaultdict(set); last=collections.defaultdict(str)
cut=(datetime.datetime.fromtimestamp(now)-datetime.timedelta(days=14)).strftime('%Y-%m-%d')
recent=set()
for r in runs:
    cp=(r.get('ckpt_path') or '').rstrip('/'); nm=os.path.basename(cp) if cp else None
    for k in {r.get('run_id') if r.get('kind')=='train' else None, nm}-{None,''}:
        exp_of[k].add(r.get('experiment'))
    la=r.get('launched_at') or ''
    if la>=cut and not str(r.get('source','')).startswith('backfill'):
        for m in re.findall(r'/prasanns/(?:ctc_suite/ckpts/)?([A-Za-z0-9_.\-]+)',json.dumps(r)): recent.add(m)
# active code / STATUS text (not ledgers)
act=[e for e,s in st.items() if s=='Active']
texts=[]
for e in act:
    p=f'/accounts/projects/berkeleynlp/prasann/ops/experiments/{e}/STATUS.md'
    if os.path.exists(p): texts.append(open(p,errors='ignore').read())
for top in ['research','recipes']:
    for dp,dn,fn in os.walk(ROOTREPO+top):
        dn[:]=[x for x in dn if not x.startswith(('results','runs','weights','.'))]
        for f in fn:
            if f.endswith(('.py','.sh','.sbatch','.yaml','.json')) and os.path.getsize(os.path.join(dp,f))<3e6:
                texts.append(open(os.path.join(dp,f),errors='ignore').read())
blob='\n'.join(texts)
items=[]
CONT=('ctc_suite','outlier_lengthmix','ctc_length_mix','ctc_olmo3','olmo3_cptdrop')
for p,b,n,new,dep,_ in d['dirs']:
    if not p.startswith(R): continue
    rel=p[len(R):]; parts=rel.split('/')
    if p.startswith(C) and dep==3: name=parts[-1]
    elif dep==1 and parts[0] not in CONT: name=rel
    elif dep==2 and parts[0] in CONT and rel!='ctc_suite/ckpts' and parts[0]!='olmo3_cptdrop': name=rel
    else: continue
    base=os.path.basename(name); idle=(now-new)/86400
    exps={e for e in exp_of.get(base,set()) if e}
    sts={st.get(e,'?') for e in exps}
    if any(re.search(x,p) for x in PROTECT): t,why='keep','protected'
    elif base in recent: t,why='keep','used in last 14d'
    elif idle<14: t,why='keep',f'modified {idle:.0f}d ago'
    elif 'Active' in sts: t,why='decide',f'active: {"/".join(sorted(exps))}'
    elif len(base)>8 and base in blob: t,why='keep','named in active code/STATUS'
    elif exps and sts<= {'Done','Paused'}: t,why='cand',f'{"/".join(sorted(exps))} ({"/".join(sorted(sts))})'
    elif not exps and re.match(RETIRED_PREFIX,base): t,why='cand','retired landmark'
    else: t,why='keep',f'unclassified ({"/".join(sorted(exps)) or "no registry entry"})'
    items.append(dict(path=p,bytes=b,idle=round(idle),tier=t,why=why))
json.dump(items,open('tiers2.json','w'),indent=0)
tot=d['totals'][R.rstrip('/')]['bytes']
print(f'weka total {tot/TiB:.2f} TiB, index {datetime.datetime.fromtimestamp(now)}')
agg=collections.defaultdict(lambda:[0,0])
for it in items:
    k=(it['tier'],it['why'] if it['tier']!='keep' or it['why'].startswith(('unclass','active')) else it['why'].split(' ')[0])
    agg[k][0]+=it['bytes'];agg[k][1]+=1
for (t,w),(b,n) in sorted(agg.items(),key=lambda x:(x[0][0],-x[1][0])): print(f'{t:6s} {b/TiB:6.2f} TiB {n:4d}  {w}')
