"""Matched-loss FLOP ratios on ONE held-out subset (default set/suite16k) for every board run: vs the dense-LoRA
references and vs the recipe, at each run's own final loss on that subset.   python3 iso_set.py [set/suite16k] [N]"""
import json, sys
key = sys.argv[1] if len(sys.argv) > 1 else "set/suite16k"
top = int(sys.argv[2]) if len(sys.argv) > 2 else 45
b = json.load(open("hub_tab/board_v2.json"))["runs"]
def cur(r): return [(p, c) for (s, p, _), c in zip(r["curve"], r["by"][key])]
def reach(c, lvl):
    best = 9
    for (p0, l0), (p1, l1) in zip(c, c[1:]):
        l0 = min(best, l0); best = l0
        if l1 <= lvl:
            return p1 if l0 <= l1 else p0 + (l0 - lvl) / (l0 - l1) * (p1 - p0)
    return None
ref = [cur(r) for r in b if r["tag"] == "v2" and r["arm"] in ("denselora", "denselora2e5")]
rec = [cur(r) for r in b if r["arm"] == "recipe" and r["tag"] in ("v2", "full", "v2s1")]
rows = []
for r in b:
    if key not in r["by"]: continue
    c = cur(r); la, fa = min(l for _, l in c), c[-1][0]
    fa = next(p for p, l in c if l == la)  # PF at its best loss on this subset
    def iso(cs):
        n = [reach(x, la) for x in cs]; n = [v for v in n if v]
        return (min(n) / fa) if n and fa > 0 else None
    rows.append((iso(rec) or -1, f"{r['tag']}/{r['arm']}/sd{r['seed']}", r["steps"], la, fa, r.get("pf_per_step_vs_dense") or 0, iso(ref), iso(rec)))
print(f"{key}: {'run':34s} {'steps':>5s} {'bestCE':>6s} {'PF':>6s} {'x/dns':>5s} {'vs ref':>6s} {'vs recipe':>9s}")
f = lambda v: f"{v:6.2f}" if v else "  past"
for o in sorted(rows, reverse=True)[:top]:
    print(f"   {o[1][:40]:40s} {o[2]:5d} {o[3]:6.3f} {o[4]:6.0f} {o[5]:5.2f} {f(o[6])} {f(o[7]):>9s}")
