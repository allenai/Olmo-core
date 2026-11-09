import json,os,re,collections,sys
idx=sys.argv[1] if len(sys.argv)>1 else os.path.expanduser('~/.cache/storage-audit/weka.json')
d=json.load(open(idx)); now=d['generated']; TiB=2**40
R='/weka/oe-training-default/ai2-llm/checkpoints/prasanns/'; C=R+'ctc_suite/ckpts/'
runs=json.load(open('runs.json'))
PROTECT=[r"/ctc_suite/bases(/|$)",r"/ctc_olmo3/bases/.*-fixmark",r"xlong5_2k256k_qwen35",r"contra_mix_.*_10k_128k",r"_eval_bundle",r"ctc-olmo3-7b-hpqaret-",r"vllm_serving",r"_olmoeval_ctc",r"_eval_results",r"_inventory",r"cr_suite_data",r"ctc_sft_sets",r"hf_models",r"/shared(/|$)"]
# experiment of each run name (train launches + eval ckpt paths)
exp_of={}
recent=set()
for r in runs:
    cp=(r.get('ckpt_path') or '').rstrip('/'); nm=os.path.basename(cp) if cp else None
    for k in [r.get('run_id') if r.get('kind')=='train' else None, nm]:
        if k: exp_of.setdefault(k,r.get('experiment'))
    la=r.get('launched_at') or ''
    if la>='2026-10-09' and not str(r.get('source','')).startswith('backfill'):
        blob=json.dumps(r)
        for m in re.findall(r'/prasanns/(?:ctc_suite/ckpts/)?([A-Za-z0-9_.\-]+)',blob): recent.add(m.split('/')[-1])
DONE={'flop_scaling','ds64','softdetach','ds64_fast2k','ctc_olmo_hybrid','olmo3_vs_hybrid_wave2','pooled_kv','kv_route','nested_ffn_moe'}
PAUSED={'outlier_lengthmix','ctc_modelscale','hybridish_sft','ctc_data_fixes'}
def prefix_tier(name):
    if re.match(r'(lmx|tsl)',name): return 2,'outlier_lengthmix (paused)'
    if re.match(r'(ctc-4b|ctcfam|ctcms|ctc-olmohyb|ctc-olmo3nsc|ctc-llama8b|ctc-xabs|ctc-s5|ctc-contra|ctc-outlier)',name): return 2,'ctc_modelscale/olmo-hybrid era (paused)'
    if re.match(r'(lm3|q3-)',name): return 1,'retired landmark'
    if re.match(r'(q4b-|q06b-|single_task|cptmix_|docchunk_5task|longctx_sft|rag_jsonl)',name): return 1,'retired landmark (top-level)'
    return None
items=[]
for p,b,n,new,dep,_ in d['dirs']:
    CONT=('ctc_suite','outlier_lengthmix','ctc_length_mix','ctc_olmo3')
    rel=p[len(R):] if p.startswith(R) else None
    if rel is None: continue
    parts=rel.split('/')
    if p.startswith(C) and dep==3: name=parts[-1]
    elif dep==1 and parts[0] not in CONT: name=rel
    elif dep==2 and parts[0] in CONT and rel!='ctc_suite/ckpts': name=rel
    else: continue
    idle=(now-new)/86400
    e=exp_of.get(name)
    if any(re.search(x,p) for x in PROTECT): t,why=0,'PROTECTED'
    elif name in recent or os.path.basename(name) in recent: t,why=0,'used since 10-09'
    elif idle<21: t,why=0,f'modified {idle:.0f}d ago (active work)'
    elif name.split('/')[0] in('outlier_lengthmix','ctc_length_mix'): t,why=2,'outlier_lengthmix (paused)'
    elif e in DONE: t,why=1,f'{e} (done)'
    elif e in PAUSED: t,why=2,f'{e} (paused)'
    elif prefix_tier(name): t,why=prefix_tier(name)
    else: t,why=3,f'unclassified ({e})'
    items.append(dict(path=p,bytes=b,idle=round(idle),tier=t,why=why))
json.dump(items,open('tiers.json','w'),indent=0)
tot=d['totals'][R.rstrip('/')]['bytes']
print(f'weka prasanns total {tot/TiB:.2f} TiB  (index {os.path.basename(idx)}, {(os.path.getmtime(idx))})')
agg=collections.defaultdict(lambda:[0,0])
for it in items: x=agg[(it['tier'],it['why'])];x[0]+=it['bytes'];x[1]+=1
for (t,w),(b,n) in sorted(agg.items()): print(f'tier{t} {b/TiB:6.2f} TiB {n:4d}  {w}')
for t in range(4):
    print(f'TIER {t}: {sum(i["bytes"] for i in items if i["tier"]==t)/TiB:.2f} TiB')
cum=tot
for t in (1,2,3):
    cum-=sum(i["bytes"] for i in items if i["tier"]==t); print(f'after deleting tiers 1..{t}: {cum/TiB:.2f} TiB')
