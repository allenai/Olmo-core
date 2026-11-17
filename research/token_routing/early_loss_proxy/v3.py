"""16k-first table on heldout_v3 runs (Beaker tag v3 + any extra tags given): per run, mean CE on the 16k sets
(gen16k + rung16k), each 16k set, and 4k/8k, at each probe with PF; and the FLOP ratio vs v3/denselora at the
16k loss levels dense LoRA reaches at its probes.   python3 v3.py [extra tags...] [--json out]"""
import json, glob, os, sys
import numpy as np
B = "/tmp/claude-3018/elp_beaker_results"
tags = ["v3"] + [a for a in sys.argv[1:] if not a.startswith("--") and not a.endswith(".json")]
def load(p):
    d = json.load(open(p)); f = lambda q: (q.get("flops") or {})
    S = np.array([r["set"] for r in d["rows"]])
    g = {"16k": np.isin(S, ["gen16k", "rung16k"]), "gen": S == "gen16k", "rung": S == "rung16k", "8k": S == "suite8k", "4k": S == "suite4k"}
    if not g["16k"].any(): return None
    c = [dict(step=q["step"], pf=f(q).get("actual_pf", 0.0), **{k: float(np.asarray(q["ce"])[m].mean()) for k, m in g.items() if m.any()}) for q in d["probes"]]
    return dict(name=os.path.basename(p)[4:-5], curve=c, done=bool(d.get("done")), xd=f(d["probes"][-1]).get("actual_pf", 0) / max(1e-9, f(d["probes"][-1]).get("dense_pf", 0)), n16=int(g["16k"].sum()))
def reach(c, lvl, k="16k"):
    best = 9
    for a, b in zip(c, c[1:]):
        l0 = min(best, a[k]); best = l0
        if b[k] <= lvl:
            return b["pf"] if l0 <= b[k] else a["pf"] + (l0 - lvl) / (l0 - b[k]) * (b["pf"] - a["pf"])
    return None
runs = [r for t in tags for p in sorted(glob.glob(f"{B}/{t}/elp-*.json")) for r in [load(p)] if r]
ref = next((r for r in runs if r["name"].startswith("v3-denselora-")), None)
print(f"16k sets eval_size {runs[0]['n16'] if runs else 0}. Columns per probe: step:16k/gen16k/rung16k@PF")
for r in runs:
    print(f"{r['name'][:22]:22s} x/d {r['xd']:.3f} {'D' if r['done'] else '.'} | " + " ".join(f"{c['step']}:{c['16k']:.3f}/{c['gen']:.3f}/{c['rung']:.3f}@{c['pf']:.0f}" for c in r["curve"][1:]))
out = {"runs": runs, "ref": ref and ref["name"], "table": []}
if ref and len(ref["curve"]) > 2:
    lv = [(c["step"], min(x["16k"] for x in ref["curve"] if x["step"] <= c["step"])) for c in ref["curve"][1:]]
    print("\nFLOP ratio vs dense LoRA at the 16k loss it reaches at step: " + " ".join(f"{s}({l:.3f})" for s, l in lv))
    for r in runs:
        cells = []
        for s, l in lv:
            a, b = reach(r["curve"], l), reach(ref["curve"], l)
            cells.append(f"{b / a:5.2f}" if a and b else "    -")
            out["table"].append({"run": r["name"], "ref_step": s, "level": l, "pf": a, "x": (b / a) if a and b else None})
        print(f"{r['name'][:22]:22s} " + " ".join(cells))
if "--json" in sys.argv: json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
