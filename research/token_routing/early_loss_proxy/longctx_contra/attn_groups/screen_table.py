"""Screen table: per config (dense_probe JSONs from screen_local.sh) the mean per-step actual PF, s/step and peak GiB over
steps >= SKIP (the router's warmup step and kernel autotune excluded), and the paired per-step ratio vs the same rows
with attention off (same seed, no shuffle: identical rows and gold-drop draws in every config).
    python3 screen_table.py <dir with drop_*.json / whole_*.json> [SKIP]"""
import json
import os
import sys

d = sys.argv[1]
skip = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def load(cfg):
    p = os.path.join(d, cfg + ".json")
    if not os.path.exists(p):
        return None
    st = json.load(open(p))["steps"]
    pf, prev = [], 0.0
    for s in st:
        a = s["flops"]["actual_pf"]
        pf.append(a - prev)
        prev = a
    return [(s["step"], p_, s["train_s"], s["peak_gib"]) for s, p_ in zip(st, pf) if s["step"] >= skip]


for mode in ("drop", "whole"):
    base = load(f"{mode}_off")
    if base is None:
        continue
    bmap = {s: (p, t) for s, p, t, _ in base}
    print(f"\n{mode} rows (steps >= {skip}, {len(base)} steps):")
    print("| G | PF/step | ratio vs off | s/step | ratio vs off | peak GiB |")
    print("|---|---|---|---|---|---|")
    for g in ("off", "g1", "g2", "g4", "g8"):
        r = load(f"{mode}_{g}")
        if not r:
            continue
        n = len(r)
        mp = sum(x[1] for x in r) / n
        mt = sum(x[2] for x in r) / n
        pk = max(x[3] for x in r)
        rp = sum(x[1] for x in r if x[0] in bmap) / max(1e-9, sum(bmap[x[0]][0] for x in r if x[0] in bmap))
        rt = sum(x[2] for x in r if x[0] in bmap) / max(1e-9, sum(bmap[x[0]][1] for x in r if x[0] in bmap))
        print(f"| {g} | {mp:.3f} | {rp:.3f} | {mt:.1f} | {rt:.3f} | {pk:.1f} |")
