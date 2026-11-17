"""Iso-loss FLOP ratios at each run's LAST probe: vs the dense-LoRA reference and vs the recipe (>1 = fewer FLOPs
than that curve needs for the same held-out loss). For runs stopped by the time cap before reaching a reference level.
    python3 iso.py [board json]"""
import json, sys
sys.path.insert(0, ".")
from score import pf_to_reach
b = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "hub_tab/board_v2.json"))
runs = b["runs"]
cur = lambda r: [{"step": s, "pf": p, "all": c} for s, p, c in r["curve"]]
ref = [cur(r) for r in runs if r["tag"] == "v2" and r["arm"] == "denselora"]
rec = [cur(r) for r in runs if r["tag"] == "v2" and r["arm"] == "recipe"]
out = []
for r in runs:
    c = cur(r); la, fa = c[-1]["all"], c[-1]["pf"]
    def iso(curves):
        need = [pf_to_reach(x, la) for x in curves]
        need = [n for n in need if n is not None]
        return (min(need) / fa) if need and fa > 0 else None
    out.append((iso(rec) or 0, r["tag"], r["arm"], r["steps"], la, fa, r.get("pf_per_step_vs_dense") or 0, iso(ref), iso(rec),
                {k.split("/")[1][:6]: round(v[-1], 3) for k, v in r["by"].items() if k.startswith("task/")}))
print(f"{'tag/arm':30s} {'steps':>5s} {'CE':>6s} {'PF':>7s} {'step/d':>6s} {'iso/ref':>7s} {'iso/recipe':>10s}  per-task CE")
f = lambda v: f"{v:7.2f}" if v else "   past"
for o in sorted(out, reverse=True):
    print(f"{(o[1] + '/' + o[2])[:30]:30s} {o[3]:5d} {o[4]:6.3f} {o[5]:7.1f} {o[6]:6.3f} {f(o[7])} {f(o[8]):>10s}  {o[9]}")
