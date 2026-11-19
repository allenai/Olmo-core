"""Per-task FLOPs-to-match: ftm.py's frontier logic applied to each task's 16k held-out CE (gen16k + rung16k of heldout_v3,
80 rows per task: far below 500, so read the ratios as rough). Finished runs only, seeds averaged; dense frontier = dense
full fine-tune + dense LoRA at every budget.
    python3 ftm_task.py [--json hub_tab/ftm_task.json] [arm substrings to print; default: lo-p50-20-7 recipe lo-r50to20]"""
import glob, json, math, os, sys
import numpy as np

SRC = [p for pat in ("v3", "w[0-9]*", "x4", "bud", "x16", "ds", "tb") for p in sorted(glob.glob("/tmp/claude-3018/elp_beaker_results/" + pat))]
SRC += sorted(glob.glob("/net/sneetches/data/prasann/early_loss_proxy/results/w[0-9]*"))
P = []
for src in SRC:
    for p in sorted(glob.glob(src + "/elp-*.json")):
        d = json.load(open(p))
        if not d.get("done") or not d.get("probes"): continue
        S = np.array([r["set"] for r in d["rows"]]); m = np.isin(S, ["gen16k", "rung16k"])
        if not m.any(): continue
        T = np.array([r.get("task", "?") for r in d["rows"]]); q = d["probes"][-1]; ce = np.asarray(q["ce"]); pf = (q.get("flops") or {}).get("actual_pf", 0)
        k = os.path.basename(p)[4:-5]; arm = k.split("-", 1)[1].rsplit("-sd", 1)[0]
        if "mix" in arm or arm == "denselora2e5" or not pf: continue
        kind = "dense" if arm.startswith("dense") else "drop"
        P.append(dict(arm=arm, kind=kind, steps=q["step"], pf=pf, by={**{t: float(ce[m & (T == t)].mean()) for t in sorted(set(T[m]))}, "mean": float(ce[m].mean())}))
G = {}
for q in P: G.setdefault((q["arm"], q["steps"]), []).append(q)
P = [dict(arm=a, kind=g[0]["kind"], steps=s, pf=sum(x["pf"] for x in g) / len(g), seeds=len(g), by={t: sum(x["by"][t] for x in g) / len(g) for t in g[0]["by"]}) for (a, s), g in G.items()]
TASKS = [t for t in P[0]["by"]]


def ratio(p, t):
    F = []
    for q in sorted((q for q in P if q["kind"] == "dense"), key=lambda q: q["pf"]):
        if not F or q["by"][t] < F[-1][1]: F.append((q["pf"], q["by"][t]))
    ce = p["by"][t]
    if ce >= F[0][1]: return None, "<"
    for (p0, c0), (p1, c1) in zip(F, F[1:]):
        if c1 <= ce <= c0: return math.exp(math.log(p0) + (c0 - ce) / (c0 - c1) * (math.log(p1) - math.log(p0))) / p["pf"], ""
    return F[-1][0] / p["pf"], ">"


want = [a for a in sys.argv[1:] if not a.startswith("--") and not a.endswith(".json")] or ["lo-p50-20-7", "recipe", "lo-r50to20"]
out = []
print(f"{'arm':26s} steps    PF sd | " + " | ".join(f"{t[:13]:>19s}" for t in TASKS))
for p in sorted(P, key=lambda p: (p["kind"], p["arm"].split("-b")[0].replace("-x4", ""), p["pf"])):
    row = {t: ratio(p, t) for t in TASKS}
    out.append(dict(arm=p["arm"], kind=p["kind"], steps=p["steps"], pf=p["pf"], seeds=p["seeds"], ce=p["by"], x={t: dict(x=row[t][0], bound=row[t][1]) for t in TASKS}))
    if p["kind"] == "dense" or any(p["arm"].startswith(w) and "fz7" not in p["arm"] for w in want):
        print(f"{p['arm'][:26]:26s} {p['steps']:5d} {p['pf']:5.0f} {p['seeds']:2d} | " + " | ".join(
            f"{p['by'][t]:.3f} " + ("          " if p["kind"] == "dense" else (f"{row[t][1]:>1s}x{row[t][0]:4.1f}    " if row[t][0] else "  <x1     ")) for t in TASKS))
if "--json" in sys.argv: json.dump(dict(tasks=TASKS, runs=out, eval_size_per_task=80), open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
