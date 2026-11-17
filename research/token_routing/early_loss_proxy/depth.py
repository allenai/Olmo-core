"""FLOPs (PF) each run needs to reach fixed held-out CE levels, and the ratio to the full-epoch dense-LoRA curve.
    python3 depth.py [--json out.json]"""
import json, glob, os, sys
import numpy as np
B = "/tmp/claude-3018/elp_beaker_results"
def curve(p):
    d = json.load(open(p)); f = lambda q: (q.get("flops") or {})
    c = [(f(q).get("actual_pf", 0.0), float(np.mean(q["ce"]))) for q in d["probes"]]
    st = d.get("steps", [])
    return c, f(d["probes"][-1]).get("actual_pf", 0) / max(1e-9, f(d["probes"][-1]).get("dense_pf", 0)), len(st), \
        max([s.get("transient_gib", 0) for s in st] or [0]), {t: float(np.mean([c for r, c in zip(d["rows"], d["probes"][-1]["ce"]) if r["task"] == t])) for t in sorted({r["task"] for r in d["rows"]})}
def reach(c, lvl):
    best = 9
    for (p0, l0), (p1, l1) in zip(c, c[1:]):
        l0 = min(best, l0); best = l0
        if l1 <= lvl:
            return p1 if l0 <= l1 else p0 + (l0 - lvl) / (l0 - l1) * (p1 - p0)
    return None
ref, *_ = curve(f"{B}/full/elp-full-denselora-sd0.json")
LV = [0.160, 0.140, 0.125, 0.115, 0.105, 0.095]
runs = [("recipe sd0", "full/elp-full-recipe-sd0"), ("recipe sd1", "full/elp-full-recipe-sd1"),
        ("rc-mix50 sd0", "deep/elp-deep-rc-mix50-sd0"), ("rc-mix50 sd1", "deep/elp-deep-rc-mix50-sd1"),
        ("rc-mix80 sd0", "deep/elp-deep-rc-mix80-sd0"), ("rc-mix80 sd1", "deep/elp-deep-rc-mix80-sd1"),
        ("r_cut5k", "rowsbk/elp-rowsbk-r_cut5k-sd0"), ("r_mix50curr", "rowsbk/elp-rowsbk-r_mix50curr-sd0"),
        ("a16ema11", "alloc/elp-alloc-a16ema11-sd0"), ("m50ema11", "alloc/elp-alloc-m50ema11-sd0"),
        ("m50ema10", "alloc/elp-alloc-m50ema10-sd0"), ("rc-mq", "mq/elp-mq-rc-mq-sd0"), ("rc-mix50mq", "mq/elp-mq-rc-mix50mq-sd0"),
        ("dl_mix50pack", "rowsbk/elp-rowsbk-dl_mix50pack-sd0"), ("dense LoRA 2e-5 sd1", "v2s1/elp-v2s1-denselora2e5-sd1")]
runs += [(os.path.basename(p)[4:-5], p[len(B) + 1:-5]) for t in sys.argv[1:] if not t.startswith("--") for p in sorted(glob.glob(f"{B}/{t}/elp-*.json"))]
out = []
print(f"{'run':22s} {'steps':>5s} {'x/dense':>7s} {'last CE':>7s} " + " ".join(f"{'@' + format(l, '.3f'):>11s}" for l in LV) + "   (PF to reach, x vs dense LoRA)")
print(f"{'dense LoRA (full)':22s} {488:5d} {0.741:7.3f} {ref[-1][1]:7.3f} " + " ".join(f"{reach(ref, l) or 0:6.0f}     " for l in LV))
for name, rel in runs:
    p = f"{B}/{rel}.json"
    if not os.path.exists(p): continue
    c, xd, n, tr, by = curve(p)
    cells = []
    rec = {"run": name, "steps": n, "x_dense": xd, "last_ce": c[-1][1], "pf": c[-1][0], "transient_gib": tr, "by_task": by, "levels": {}}
    for l in LV:
        a, r = reach(c, l), reach(ref, l)
        cells.append(f"{a:6.0f} {r / a:4.2f}" if a and r else "     -     ")
        rec["levels"][f"{l:.3f}"] = {"pf": a, "x_vs_dense_lora": (r / a) if a and r else None}
    out.append(rec)
    print(f"{name[:22]:22s} {n:5d} {xd:7.3f} {c[-1][1]:7.3f} " + " ".join(cells))
if "--json" in sys.argv: json.dump({"levels": LV, "ref": ref, "runs": out}, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
