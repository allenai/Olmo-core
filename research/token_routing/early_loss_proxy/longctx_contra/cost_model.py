import numpy as np
BLK=2.217e10/32; HEAD=3.8e9; Q=24576.0  # per full-attn layer: 12*16*256/2 = 24576 per (q,k) pair
s1,s2=0.247,0.129; OVH=1.06
def dense(L, lora=True):
    lin=(32*BLK+HEAD)*L; quad=8*Q*L*L
    f=2/3 if lora else 1
    return f*lin, f*quad
def recipe(l):
    lin=BLK*l*(7+9*s1+16*s2)+HEAD*s2*l; quad=Q*l*l*(1+3*s1**2+4*s2**2)
    return OVH*(2/3)*lin, OVH*(2/3)*quad
def gdd(L, lo=0.05, hi=0.5, whole=0.15, n=200000, extra=0.0):
    rng=np.random.default_rng(0); r=rng.uniform(lo,hi,n); k=np.where(rng.random(n)<whole,1.0,r+extra)
    a=np.array([sum(recipe(x*L)) for x in k[:20000]]); return a.mean(), k[:20000]
for name,L in [("16k",16384*0.95),("64k",58586),("256k",249000),("1m",1010000)]:
    dl,dq=dense(L); D=dl+dq
    rl,rq=recipe(L); R=rl+rq
    print(f"{name}: denseLoRA {D/1e15:.2f} PF/row (lin {dl/1e15:.2f}, attn-quad {dq/1e15:.2f} = {dq/D:.0%}); recipe {R/1e15:.3f} ({R/D:.3f}x)")
    for lo,hi,w in [(0.05,0.5,0.15),(0.05,0.5,0.0),(0.05,0.25,0.0),(0.05,0.5,0.05)]:
        g,_=gdd(L,lo,hi,w); print(f"    gdd keep {lo}-{hi} whole {w}: {g/1e15:.3f} PF/row = {g/D:.4f}x dense LoRA ({D/g:.1f}x)")
    # pure-gdd (no router) expectation: E[k] lin + E[k^2] quad
    for lo,hi,w in [(0.05,0.5,0.15)]:
        Ek=w+(1-w)*(lo+hi)/2; Ek2=w+(1-w)*(hi**3-lo**3)/(3*(hi-lo))
        print(f"    gdd alone (dense LoRA forward on kept row) E[k]={Ek:.3f} E[k^2]={Ek2:.3f}: {(Ek*dl+Ek2*dq)/D:.3f}x")
