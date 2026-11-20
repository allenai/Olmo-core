"""A/B/C/D/E comparison at 16k: per-probe PF and CE, speedup vs dft and vs the recipe's two-seed mean curve (at the
arm's lowest CE, or the reference's best if the arm gets below it), CE at matched PF, paired final-probe differences.
    python3 ab_compare.py "A mean:a_sd0.json,a_sd1.json" "--pair=a_sd0.json:b_sd0.json"
"""
# usage: ab2.py ARMNAME:json1,json2[,...] ... ; then pairs A:B per seed via --pair a.json:b.json
import json, sys, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from curves_lib import curve, pf_to_reach, ce_at, T, RF
def avg(cs):
    n = min(len(c) for c in cs)
    return [(cs[0][i][0], float(np.mean([c[i][1] for c in cs])), float(np.mean([c[i][2] for c in cs])),
             {t: float(np.mean([c[i][3][t] for c in cs])) for t in T}) for i in range(n)]
dft = curve(f"{RF}/kl16k/elp-kl16k-dft-sd0.json")[0]
rec = avg([curve(f"{RF}/kl16k/elp-kl16k-k20-sd{s}.json")[0] for s in (0, 1)])
def show(name, c):
    print(f"== {name}")
    for s, pf, ce, bt in c:
        print(f"  step {s:4d} PF {pf:7.1f}  mean {ce:.4f}  " + " / ".join(f"{bt[t]:.4f}" for t in T))
    i = int(np.argmin([x[2] for x in c])); ce = c[i][2]
    print(f"  lowest {ce:.4f} at step {c[i][0]} ({c[i][1]:.1f} PF)")
    for rn, R in (("dft", dft), ("recipe 2-seed mean", rec)):
        best = min(x[2] for x in R); lvl = max(ce, best)
        pfr, pfa = pf_to_reach(R, lvl), pf_to_reach(c, lvl)
        print(f"  x vs {rn}: {pfr / pfa:.2f} (level {lvl:.4f}: ref {pfr:.1f} / arm {pfa:.1f} PF)")
    for b in (59, 138, 200, 250, 300, 333):
        v = ce_at(c, b)
        print(f"  CE@{b}PF arm {'-' if v is None else f'{v:.4f}'}  recipe-mean {ce_at(rec, b):.4f}  dft {ce_at(dft, b):.4f}")
def rows_at(f, pf):
    """per-row CE linearly interpolated between the two probes bracketing pf (no running min); seed-averaged if f has commas"""
    out = []
    for g in f.split(","):
        d = json.load(open(g)); P = [(q["flops"]["actual_pf"], np.array(q["ce"])) for q in d["probes"]]
        if pf > P[-1][0] * 1.02: return None, d["rows"]
        for i in range(1, len(P)):
            if P[i][0] >= pf or i == len(P) - 1:
                (p0, c0), (p1, c1) = P[i - 1], P[i]; w = min(1.0, (pf - p0) / (p1 - p0)); out.append(c0 + w * (c1 - c0)); break
    return np.mean(out, axis=0), d["rows"]
for a in sys.argv[1:]:
    if a.startswith("--pairpf="):  # --pairpf=PF:a1.json[,a2.json]:b1.json[,b2.json]
        pf, x, y = a[9:].split(":"); pf = float(pf)
        (u, r), (v, _) = rows_at(x, pf), rows_at(y, pf)
        d = u - v; o = f"paired @ {pf:g} PF {x.rsplit('/',1)[-1]} - {y.rsplit('/',1)[-1]}: mean {d.mean():+.4f}±{d.std(ddof=1) / np.sqrt(len(d)):.4f}"
        for t in T:
            m = np.array([q["task"] == t for q in r]); o += f" | {t} {d[m].mean():+.4f}±{d[m].std(ddof=1) / np.sqrt(m.sum()):.4f}"
        print(o); continue
    if a.startswith("--pair="):
        x, y = a[7:].split(":")
        A, B = json.load(open(x)), json.load(open(y))
        u, v = np.array(A["probes"][-1]["ce"]), np.array(B["probes"][-1]["ce"]); r = A["rows"]; d = u - v
        o = f"paired {x.rsplit('/',1)[-1]} - {y.rsplit('/',1)[-1]}: mean {d.mean():+.4f}±{d.std(ddof=1) / np.sqrt(len(d)):.4f}"
        for t in T:
            m = np.array([q["task"] == t for q in r]); o += f" | {t} {d[m].mean():+.4f}±{d[m].std(ddof=1) / np.sqrt(m.sum()):.4f}"
        print(o)
        continue
    name, files = a.split(":", 1)
    cs = [curve(f)[0] for f in files.split(",")]
    show(name, cs[0] if len(cs) == 1 else avg(cs))
