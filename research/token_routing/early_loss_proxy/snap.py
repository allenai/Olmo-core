"""One-line-per-run snapshot of every new-pool proxy run (also unfinished ones): python3 snap.py [substr]"""
import json, glob, os, sys
import numpy as np
pat = sys.argv[1] if len(sys.argv) > 1 else ""
roots = ["/net/sneetches/data/prasann/early_loss_proxy/results", "/net/mooney/data/prasann/early_loss_proxy/results",
         "/tmp/claude-3018/elp_lambda_results", "/tmp/claude-3018/elp_beaker_results"]
skip = ("/v1/", "/s50/", "smoke", "8k/", "v1pool", "/t/")
for root in roots:
    for p in sorted(glob.glob(root + "/*/elp-*.json")):
        if any(s in p for s in skip) or pat not in p:
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        st, pr = d.get("steps", []), d["probes"]
        if len(d["rows"]) < 100:
            continue
        ts = [s["train_s"] for s in st]
        f = lambda q: (q.get("flops") or {})
        cur = " ".join(f"{q['step']}:{np.mean(q['ce']):.3f}" for q in pr)
        last = pr[-1]
        print(f"{os.path.basename(os.path.dirname(p))[:9]:9s} {os.path.basename(p)[4:-5][:30]:30s} st {len(st):3d} s/st {np.mean(ts[3:]) if len(ts) > 3 else 0:5.1f} "
              f"tr {max([s.get('transient_gib', 0) for s in st] or [0]):4.1f} x/dense {f(last).get('actual_pf', 0) / max(1e-9, f(last).get('dense_pf', 0)):.3f} "
              f"{'D' if d.get('done') else '.'} | {cur}")
