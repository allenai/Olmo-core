import json,os,collections,re,sys
idx=sys.argv[1] if len(sys.argv)>1 else os.path.expanduser('~/.cache/storage-audit/weka.json')
d=json.load(open(idx)); now=d['generated']; T=1e12
runs=json.load(open('runs.json'))
st={}
for line in open('/accounts/projects/berkeleynlp/prasann/ops/experiments/INDEX.md'):
    pass
sec=None
for line in open('/accounts/projects/berkeleynlp/prasann/ops/experiments/INDEX.md'):
    m=re.match(r'## (\w+)',line)
    if m: sec=m.group(1)
    m=re.search(r'\[(\w+)\]\(',line)
    if m and sec: st[m.group(1)]=sec
R='/weka/oe-training-default/ai2-llm/checkpoints/prasanns/'
C=R+'ctc_suite/ckpts/'
exp_of={};nev=collections.Counter();lastev={}
for r in runs:
    cp=(r.get('ckpt_path') or '').rstrip('/')
    name=os.path.basename(cp) if cp.startswith(C) else None
    if r.get('kind')=='train':
        for k in [r.get('run_id'), name]:
            if k: exp_of.setdefault(k,r.get('experiment'))
    if r.get('kind')=='eval' and name:
        nev[name]+=1; lastev[name]=max(lastev.get(name,''), r.get('launched_at') or r.get('ts') or '')
        exp_of.setdefault(name,r.get('experiment'))
rows=[]
for p,b,n,new,dep,_ in d['dirs']:
    if dep==3 and p.startswith(C):
        k=p[len(C):]; rows.append((k,b,int((now-new)/86400),exp_of.get(k),nev[k],lastev.get(k,'')))
json.dump(rows,open('ckpt_rows.json','w'))
agg=collections.defaultdict(lambda:[0,0,0,999])
for k,b,a,e,ne,le in rows:
    x=agg[e];x[0]+=b;x[1]+=1;x[2]+= (ne>0);x[3]=min(x[3],a)
tot=collections.Counter()
print(f"{'TB':>6} {'runs':>5} {'eval':>5} {'minidle':>7}  experiment [status]")
for e,(b,n,ev,a) in sorted(agg.items(),key=lambda x:-x[1][0]):
    s=st.get(e,'?'); tot[s]+=b
    print(f'{b/T:6.2f} {n:5d} {ev:5d} {a:7d}  {e} [{s}]')
print({k:round(v/T,2) for k,v in tot.items()})
