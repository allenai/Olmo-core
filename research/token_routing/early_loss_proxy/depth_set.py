"""Like depth.py but on one held-out length set (suite4k / suite8k / suite16k) or task: FLOP ratio vs the full-epoch
dense-LoRA curve at the loss levels that curve reaches at steps 100 / 150 / 200 / 300 on THAT subset.
    python3 depth_set.py suite16k"""
import json, os, sys
import numpy as np
B = "/tmp/claude-3018/elp_beaker_results"
key = sys.argv[1]
def curve(p):
    d = json.load(open(p)); f = lambda q: (q.get("flops") or {})
    ix = [i for i, r in enumerate(d["rows"]) if r["set"] == key or r["task"] == key]
    return [(q["step"], f(q).get("actual_pf", 0.0), float(np.mean(np.asarray(q["ce"])[ix]))) for q in d["probes"]], len(ix)
def reach(c, lvl):
    best = 9
    for (s0, p0, l0), (s1, p1, l1) in zip(c, c[1:]):
        l0 = min(best, l0); best = l0
        if l1 <= lvl:
            return p1 if l0 <= l1 else p0 + (l0 - lvl) / (l0 - l1) * (p1 - p0)
    return None
ref, n = curve(f"{B}/full/elp-full-denselora-sd0.json")
steps = [100, 150, 200, 300]
lv = {s: min(l for st, _, l in ref if st <= s) for s in steps}
pfr = {s: reach(ref, lv[s]) for s in steps}
print(f"{key}: eval_size {n} rows. dense LoRA CE on this subset at steps {steps}: " + ", ".join(f"{lv[s]:.3f} ({pfr[s]:.0f} PF)" for s in steps) + f"; final {ref[-1][2]:.3f}")
runs = [("recipe sd0", "full/elp-full-recipe-sd0"), ("recipe sd1", "full/elp-full-recipe-sd1"),
        ("rc-mix50 sd0", "deep/elp-deep-rc-mix50-sd0"), ("rc-mix50 sd1", "deep/elp-deep-rc-mix50-sd1"),
        ("rc-mix80 sd0", "deep/elp-deep-rc-mix80-sd0"), ("rc-mix80 sd1", "deep/elp-deep-rc-mix80-sd1"),
        ("r_cut5k (rows<=5k)", "rowsbk/elp-rowsbk-r_cut5k-sd0"), ("a16ema11", "alloc/elp-alloc-a16ema11-sd0"),
        ("m50ema11", "alloc/elp-alloc-m50ema11-sd0"), ("dl_mix50pack", "rowsbk/elp-rowsbk-dl_mix50pack-sd0")]
print(f"{'run':20s} {'last CE':>7s} {'at PF':>6s} " + " ".join(f"{'x@lvl' + str(s):>9s}" for s in steps))
for name, rel in runs:
    p = f"{B}/{rel}.json"
    if not os.path.exists(p): continue
    c, _ = curve(p)
    cells = []
    for s in steps:
        a = reach(c, lv[s])
        cells.append(f"{pfr[s] / a:9.2f}" if a else "        -")
    print(f"{name:20s} {c[-1][2]:7.3f} {c[-1][1]:6.0f} " + " ".join(cells))
