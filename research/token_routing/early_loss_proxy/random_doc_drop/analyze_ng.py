"""Score the ng16k runs (random doc drop on no-gold tasks): per arm and seed, step cost vs dense, final / lowest held-out
loss (mean and per task), FLOPs saved vs dft and vs the recipe (k20, same seed) at the arm's own lowest loss, loss at
matched FLOPs, and a loss-vs-PF plot (mean + per task).

PF = the FLOP meter's actual_pf (every arm charged for the rows it actually ran; on a drop pool that is the shortened
rows). "x/dense step" = arm PF / dft PF at the same step (same rows, same order), NOT the meter's dense_pf, which on a
drop pool would be dense-on-shortened-rows.

    python3 analyze_ng.py --root /tmp/claude-3018/elp_beaker_results/ng16k --out ng_results.json --png ng_loss_vs_pf.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "keepfrac_len"))
from analyze_kl import load, reach  # noqa: E402

TASKS = ["reorder", "oolong", "grouping"]


def ce_at(pf, ce, x):
    """running-min CE at PF x (linear between probes); None past the curve's end."""
    m = np.minimum.accumulate(np.asarray(ce))
    if x > pf[-1] * 1.0001:
        return None
    return float(np.interp(x, pf, m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--png", default="")
    a = ap.parse_args()
    runs = {}
    for p in sorted(glob.glob(f"{a.root}/elp-ng16k-*-sd*.json")):
        m = re.match(r"elp-ng16k-(\w+?)-sd(\d+)\.json", os.path.basename(p))
        if m:
            r = load(p)
            if len(r["steps"]) >= 2:
                runs.setdefault(m.group(1), {})[int(m.group(2))] = r
    dft = runs["dft"][0]
    out = {"eval_size": dft["eval_rows"], "rows_per_task": dft["rows_per_task"], "arms": {}}
    print(f"eval_size {dft['eval_rows']} {dft['rows_per_task']} (under 500: flag)")
    hdr = f"{'arm':6s} sd {'n':>3s} {'x/dft':>6s} {'s/step':>6s} {'finalCE':>7s} " + " ".join(f"{t[:7]:>7s}" for t in TASKS) \
        + f" {'minCE':>7s} {'xdft@min':>8s} {'xk20@min':>8s} {'k20CE@PF':>8s} {'dftCE@PF':>8s}"
    print(hdr)
    for arm in sorted(runs, key=lambda k: ["dft", "k20", "rddB", "rddA", "dftA"].index(k) if k in ["dft", "k20", "rddB", "rddA", "dftA"] else 9):
        for sd, r in sorted(runs[arm].items()):
            rec = runs.get("k20", {}).get(sd) or runs.get("k20", {}).get(0)
            n = min(len(r["pf"]), len(dft["pf"]))
            step_x = r["pf"][n - 1] / dft["pf"][n - 1] if r["steps"][n - 1] == dft["steps"][n - 1] else None
            mn = float(np.min(r["ce"]))
            pf_mn = reach(r["pf"], r["ce"], mn)
            pd, pk = reach(dft["pf"], dft["ce"], mn), (reach(rec["pf"], rec["ce"], mn) if rec else None)
            res = {"steps": r["steps"], "pf": r["pf"], "ce": r["ce"], "by_task": r["by_task"], "done": r["done"],
                   "x_dft_step": step_x, "s_per_step": r["s_per_step"], "final_ce": r["ce"][-1],
                   "final_by_task": {t: r["by_task"][t][-1] for t in TASKS}, "min_ce": mn, "pf_at_min": pf_mn,
                   "x_dft_at_min": (pd / pf_mn) if pd and pf_mn else None,
                   "x_k20_at_min": (pk / pf_mn) if pk and pf_mn and arm != "k20" else None,
                   # matched FLOPs: the others' running-min CE at this arm's final PF
                   "k20_ce_at_final_pf": ce_at(rec["pf"], rec["ce"], r["pf"][-1]) if rec else None,
                   "dft_ce_at_final_pf": ce_at(dft["pf"], dft["ce"], r["pf"][-1]),
                   "per_task_min": {}, "per_task_matched": {}}
            for t in TASKS:
                mt = float(np.min(r["by_task"][t]))
                pa = reach(r["pf"], r["by_task"][t], mt)
                pdt = reach(dft["pf"], dft["by_task"][t], mt)
                pkt = reach(rec["pf"], rec["by_task"][t], mt) if rec else None
                res["per_task_min"][t] = {"min_ce": mt, "x_dft": (pdt / pa) if pdt and pa else None,
                                          "x_k20": (pkt / pa) if pkt and pa and arm != "k20" else None}
                res["per_task_matched"][t] = {"self": float(np.min(r["by_task"][t])),
                                              "k20": ce_at(rec["pf"], rec["by_task"][t], r["pf"][-1]) if rec else None,
                                              "dft": ce_at(dft["pf"], dft["by_task"][t], r["pf"][-1])}
            out["arms"].setdefault(arm, {})[sd] = res
            f = lambda x, w=7, d=4: f"{x:{w}.{d}f}" if isinstance(x, (int, float)) else " " * (w - 1) + "-"  # noqa: E731
            print(f"{arm:6s} {sd:2d} {r['steps'][-1]:3d} {f(step_x, 6, 3)} {f(r['s_per_step'], 6, 1)} {f(r['ce'][-1])} "
                  + " ".join(f(r["by_task"][t][-1]) for t in TASKS)
                  + f" {f(mn)} {f(res['x_dft_at_min'], 8, 2)} {f(res['x_k20_at_min'], 8, 2)} "
                  f"{f(res['k20_ce_at_final_pf'], 8)} {f(res['dft_ce_at_final_pf'], 8)}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        col = {"dft": "#555555", "k20": "#2a6fdb", "rddA": "#d9480f", "rddB": "#2f9e44", "dftA": "#9c36b5"}
        fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
        for i, key in enumerate(["mean"] + TASKS):
            for arm, seeds in out["arms"].items():
                for sd, r in seeds.items():
                    y = r["ce"] if key == "mean" else r["by_task"][key]
                    ax[i].plot(r["pf"], y, "-o", ms=3, color=col.get(arm, "k"), alpha=1.0 if sd == 0 else 0.5,
                               ls="-" if sd == 0 else "--", label=f"{arm} sd{sd}")
            ax[i].set_xscale("log")
            ax[i].set_yscale("log")
            ax[i].set_title(f"{key} held-out CE (eval_size {'144' if key == 'mean' else '48'})")
            ax[i].set_xlabel("training PF (actual)")
            ax[i].grid(alpha=0.3, which="both")
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(a.png, dpi=120)
        print("wrote", a.png)


if __name__ == "__main__":
    main()
