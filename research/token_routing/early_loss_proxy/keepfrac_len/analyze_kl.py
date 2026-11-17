"""Score keepfrac_len runs: FLOPs to matched dense held-out CE vs the dense-LoRA reference and vs the 20%/7% recipe.

Per length L the runs are ``elp-kl<L>-<arm>-sd<seed>.json`` (fetch_beaker_results.sh kl<L>). Curves: (training PF
actually spent, mean held-out CE of the dense forward) at every probe, made monotone (running min), PF to reach a CE
level interpolated linearly between probes (depth.py's ``reach``).

Levels (per length): the reference's own mean CE at its probes from ``--level-from`` steps on (the region where the
reference has made most of its drop), so every level is one the reference provably reaches. Per arm and level:
x_ref = PF_ref(level) / PF_arm(level), x_rec = PF_k20(level) / PF_arm(level). Summary = geometric mean over the
levels the arm reaches (``n_lv`` of them); ``deep`` = the same at the deepest level both reach. Worst task: the same
with per-task curves and per-task levels (the reference's per-task CE at those probe steps), min over tasks of the
per-task geometric mean.

    python3 analyze_kl.py --root /tmp/claude-3018/elp_beaker_results --lens 16k,32k,64k --out keepfrac_len/kl_results.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

import numpy as np

KEEP = {"k40": (0.40, 0.14), "k20": (0.20, 0.07), "k10": (0.10, 0.035), "k5": (0.05, 0.0175),
        "k2p5": (0.025, 0.00875), "k1p25": (0.0125, 0.0045), "c1k10": (0.10, 0.07), "c2k3p5": (0.20, 0.035),
        "c1k5": (0.05, 0.05), "c2k1p75": (0.20, 0.0175), "grJ": (0.20, 0.07), "grJc": (0.20, 0.07)}


def load(p):
    d = json.load(open(p))
    tasks = sorted({r["task"] for r in d["rows"]})
    pr = [q for q in d["probes"] if q.get("ce")]
    pf = [float((q.get("flops") or {}).get("actual_pf", 0.0)) for q in pr]
    mean = [float(np.mean(q["ce"])) for q in pr]
    by = {t: [float(np.mean([c for r, c in zip(d["rows"], q["ce"]) if r["task"] == t])) for q in pr] for t in tasks}
    last = (pr[-1].get("flops") or {})
    st = d.get("steps", [])
    steps_s = [s["train_s"] for s in st[3:]] or [s["train_s"] for s in st]
    return {"steps": [q["step"] for q in pr], "pf": pf, "ce": mean, "by_task": by,
            "x_dense_step": last.get("actual_pf", 0) / max(1e-9, last.get("dense_pf", 0)),
            "peak_gib": max([s.get("peak_gib", 0) or 0 for s in st] or [0]), "n_steps": len(st),
            "s_per_step": float(np.median(steps_s)) if steps_s else None, "eval_rows": len(d["rows"]),
            "rows_per_task": {t: sum(r["task"] == t for r in d["rows"]) for t in tasks}, "done": d.get("done")}


def reach(pf, ce, lvl):
    best = 9e9
    pts = list(zip(pf, ce))
    for (p0, l0), (p1, l1) in zip(pts, pts[1:]):
        l0 = min(best, l0)
        best = l0
        if l0 <= lvl:
            return p0
        if l1 <= lvl:
            return p0 + (l0 - lvl) / (l0 - l1) * (p1 - p0)
    return None


def gmean(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def score(run, ref, rec, levels, key=None):
    pfr, cer = (run["pf"], run["ce"]) if key is None else (run["pf"], run["by_task"][key])
    out = {}
    for lv in levels:
        a = reach(pfr, cer, lv)
        r = reach(ref["pf"], ref["ce"] if key is None else ref["by_task"][key], lv)
        c = reach(rec["pf"], rec["ce"] if key is None else rec["by_task"][key], lv) if rec else None
        out[f"{lv:.4f}"] = {"pf": a, "x_ref": (r / a) if a and r else None, "x_rec": (c / a) if a and c else None}
    return out


def summarize(lv):
    xr = [v["x_ref"] for v in lv.values() if v["x_ref"]]
    xc = [v["x_rec"] for v in lv.values() if v["x_rec"]]
    deep = [v for v in lv.values() if v["x_ref"]]
    return {"x_ref": gmean(xr), "x_rec": gmean(xc), "n_lv": len(xr), "deep_x_ref": deep[-1]["x_ref"] if deep else None,
            "deep_x_rec": deep[-1]["x_rec"] if deep else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/tmp/claude-3018/elp_beaker_results")
    ap.add_argument("--lens", default="16k,32k,64k")
    ap.add_argument("--level-from", type=float, default=0.25, help="levels = ref CE at probes past this share of its run")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = {"lens": {}}
    for L in a.lens.split(","):
        runs = {}
        for p in sorted(glob.glob(f"{a.root}/kl{L}*/elp-kl{L}*-sd*.json")):
            m = re.match(rf"elp-kl{L}\w*?-(\w+)-sd(\d+)\.json", os.path.basename(p))
            if m:
                runs.setdefault(m.group(1), {})[int(m.group(2))] = load(p)
        if "ref" not in runs or "k20" not in runs:
            print(f"[kl {L}] missing ref or k20: have {sorted(runs)}")
            continue
        ref, rec = runs["ref"][0], runs["k20"][0]
        last = ref["steps"][-1]
        idx = [i for i, s in enumerate(ref["steps"]) if s >= a.level_from * last and s > 0]
        levels = sorted({round(ref["ce"][i], 4) for i in idx}, reverse=True)
        tasks = sorted(ref["by_task"])
        t_levels = {t: sorted({round(ref["by_task"][t][i], 4) for i in idx}, reverse=True) for t in tasks}
        Lres = {"levels": levels, "ref_steps": ref["steps"], "ref_ce": ref["ce"], "task_levels": t_levels, "arms": {}}
        print(f"\n=== {L}: eval rows {ref['eval_rows']} {ref['rows_per_task']}; ref steps {ref['steps']}; levels {levels}")
        print(f"{'arm':9s} sd {'steps':>5s} {'x/dense':>7s} {'s/step':>6s} {'peakGiB':>7s} {'lastCE':>7s} "
              f"{'x_ref':>6s} {'x_rec':>6s} {'n_lv':>4s} {'deep_ref':>8s} | worst-task x_ref (task) | per task x_ref")
        for arm, seeds in sorted(runs.items(), key=lambda kv: (kv[0] != "ref", -KEEP.get(kv[0], (1, 1))[0], kv[0])):
            for sd, run in sorted(seeds.items()):
                lv = score(run, ref, rec if arm != "k20" or sd != 0 else None, levels)
                s = summarize(lv)
                per_t = {}
                for t in tasks:
                    per_t[t] = summarize(score(run, ref, rec, t_levels[t], key=t))
                wt = min(tasks, key=lambda t: per_t[t]["x_ref"] or 0)
                rec_ = {"seed": sd, "keep": KEEP.get(arm), **{k: run[k] for k in ("steps", "pf", "ce", "x_dense_step", "peak_gib",
                        "s_per_step", "n_steps", "by_task", "done")}, "levels": lv, **s,
                        "per_task": per_t, "worst_task": wt, "worst_x_ref": per_t[wt]["x_ref"],
                        "worst_x_rec": min((per_t[t]["x_rec"] for t in tasks if per_t[t]["x_rec"]), default=None),
                        "x_ref_step": run["x_dense_step"] / max(1e-9, ref["x_dense_step"])}
                Lres["arms"].setdefault(arm, {})[sd] = rec_
                f = lambda x, n=6, d=2: f"{x:{n}.{d}f}" if isinstance(x, (int, float)) and x is not None else " " * (n - 1) + "-"
                print(f"{arm:9s} {sd:2d} {run['n_steps']:5d} {f(run['x_dense_step'], 7, 3)} {f(run['s_per_step'], 6, 1)} "
                      f"{f(run['peak_gib'], 7, 1)} {f(run['ce'][-1], 7, 4)} {f(s['x_ref'])} {f(s['x_rec'])} {s['n_lv']:4d} "
                      f"{f(s['deep_x_ref'], 8)} | {f(per_t[wt]['x_ref'])} ({wt}) | "
                      + " ".join(f"{t[:6]}={f(per_t[t]['x_ref'], 4)}" for t in tasks))
        res["lens"][L] = Lres
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
