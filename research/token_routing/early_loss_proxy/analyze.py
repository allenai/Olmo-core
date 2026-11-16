"""Analyze early dense-loss proxy runs: per-arm dense eval CE curves, drops, rank correlation with the final 8k score.

    python research/token_routing/early_loss_proxy/analyze.py --results <dir with elp-<tag>-<arm>-sd<seed>.json> \\
        --tag v1 --out research/token_routing/early_loss_proxy/results_v1.json --fig research/.../proxy_v1.png

Definitions (per held-out set S in {suite8k, suite2k, train} and task t):
  CE_S,t(step)   mean over the set's rows of the row's mean answer-token CE (dense forward, teacher forced)
  CE_S(step)     4-task mean of CE_S,t
  drop(step)     CE(0) - CE(step)            (positive = the dense eval loss went down)
  frac_step      drop_arm(step) / drop_dense(step)          (dense full FT at the same step)
  frac_flops     drop_arm(step) / drop_dense(F)  where F = the arm's cumulative training FLOPs at that step and the
                 dense curve is interpolated linearly in cumulative FLOPs (dense probe points 0,1,2,5,10,...)
Noise: SE of the arm-vs-dense paired per-row CE difference at each step (same rows for every arm), and the seed-to-seed
difference of each arm's drop. Spearman over arms of the seed-mean drop vs the known final 8k 4-task mean.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

TASKS = ["contradiction", "nq", "outlier", "oolong"]
#: known final 8k scores at 1,952 rows/task (eval_size 500 per task, no-cot; `ops result show --experiment tokdrop_lr`;
#: oracle / random from records/oracle-threshold-sweep.md, tail from records/tail-rule-recipe-ablation.md).
#: order: contradiction, nq, outlier, oolong. recipe = mean of 2 seeds.
FINAL = {
    "dense": (0.973, 0.924, 0.817, 0.656),
    "denselora": (0.974, 0.922, 0.858, 0.659),
    "recipe": (0.958, 0.917, 0.8605, 0.6495),
    "rnd19": (0.945, 0.926, 0.857, 0.634),
    "rnd17": (0.928, 0.904, 0.783, 0.637),
    "rnd11": (0.931, 0.878, 0.740, 0.637),
    "rnd9": (0.915, 0.862, 0.689, 0.609),
    "wtsS1": (0.942, 0.930, 0.670, 0.620),
    "wtsS0wr": (0.956, 0.920, 0.838, 0.633),
    "ddp": (0.696, 0.886, 0.001, 0.635),
    "ddpg": (0.715, 0.896, 0.421, 0.627),
    "tail20to7": (0.921, 0.856, 0.711, 0.637),
    "tail10": (0.919, 0.810, 0.617, 0.640),
    "orc95": (0.973, 0.930, 0.828, 0.669),
    "orc90": (0.957, 0.924, 0.760, 0.656),
    "orc85": (0.959, 0.936, 0.820, 0.633),
    "orc80": (0.898, 0.932, 0.409, 0.644),
    "orc70": (0.820, 0.916, 0.254, 0.601),
    "orc60": (0.861, 0.922, 0.210, 0.581),
}


def final_mean(arm):
    return float(np.mean(FINAL[arm]))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    # average ranks for ties
    for arr, r in ((x, rx), (y, ry)):
        for v in np.unique(arr):
            m = arr == v
            if m.sum() > 1:
                r[m] = r[m].mean()
    return float(np.corrcoef(rx, ry)[0, 1])


def load(results, tag):
    runs = {}
    for p in sorted(glob.glob(f"{results}/elp-{tag}-*.json")):
        m = re.match(rf"elp-{re.escape(tag)}-(.+)-sd(\d+)\.json$", os.path.basename(p))
        if not m:
            continue
        d = json.load(open(p))
        runs[(m.group(1), int(m.group(2)))] = d
    return runs


def curves(d):
    """{step: {set: {task: mean CE}}} plus the per-row CE arrays keyed (step, set, task)."""
    rows = d["rows"]
    out, per_row = {}, {}
    for pr in d["probes"]:
        st = pr["step"]
        by = {}
        for r, c in zip(rows, pr["ce"]):
            by.setdefault(r["set"], {}).setdefault(r["task"], []).append(c)
        out[st] = {s: {t: float(np.mean(v)) for t, v in tv.items()} for s, tv in by.items()}
        for s, tv in by.items():
            for t, v in tv.items():
                per_row[(st, s, t)] = np.asarray(v)
    return out, per_row


def mean4(c):
    return float(np.mean([c[t] for t in TASKS]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", default="")
    ap.add_argument("--fig-set", default="suite8k")
    ap.add_argument("--fig-step", type=int, default=10)
    a = ap.parse_args()
    runs = load(a.results, a.tag)
    arms = sorted({k[0] for k in runs}, key=lambda x: -final_mean(x) if x in FINAL else 0)
    C = {k: curves(d) for k, d in runs.items()}
    steps = sorted(set.intersection(*[set(c[0]) for c in C.values()]))
    sets = sorted(next(iter(C.values()))[0][0].keys())
    res = {"tag": a.tag, "steps": steps, "sets": sets, "arms": {}, "corr": {}, "step0": {}, "noise": {}}

    # step 0 agreement (same base, same rows): max abs spread across all runs per set/task
    for s in sets:
        for t in TASKS:
            v = [C[k][0][0][s][t] for k in C]
            res["step0"][f"{s}/{t}"] = {"min": min(v), "max": max(v), "spread": max(v) - min(v)}

    # timing / LR / FLOPs per run
    def run_meta(d):
        st = d["steps"]
        pr = d["probes"]
        return {"t_load_s": d.get("t_pre_train"), "process_wall_s": d.get("process_wall_s"),
                "train_s_per_step": [round(x["train_s"], 2) for x in st],
                "probe_s": [round(p["probe_s"], 1) for p in pr],
                "lr_at_probe": {p["step"]: p["lr"][0] for p in pr},
                "pf_at_probe": {p["step"]: (p["flops"] or {}).get("actual_pf") for p in pr},
                "dense_pf_at_probe": {p["step"]: (p["flops"] or {}).get("dense_pf") for p in pr},
                "batch_hashes": [x["batch_hash_rank0"] for x in st]}

    meta = {f"{k[0]}/sd{k[1]}": run_meta(d) for k, d in runs.items()}
    res["runs"] = meta
    # data-order check: same seed -> same batch hashes for arms on the same rows (ddp/ddpg also use mt4f1952 rows)
    order = {}
    for (arm, sd), d in runs.items():
        order.setdefault(sd, {})[arm] = tuple(x["batch_hash_rank0"] for x in d["steps"])
    res["order_identical"] = {sd: len(set(v.values())) == 1 for sd, v in order.items()}

    dense_keys = [k for k in C if k[0] == "dense"]
    for arm in arms:
        ks = [k for k in C if k[0] == arm]
        A = {"final_8k": final_mean(arm) if arm in FINAL else None,
             "final_8k_tasks": dict(zip(TASKS, FINAL.get(arm, (None,) * 4))), "seeds": sorted(k[1] for k in ks),
             "ce": {}, "drop": {}, "drop_task": {}, "frac_step": {}, "frac_flops": {}, "seed_diff": {},
             "paired_se_vs_dense": {}}
        for s in sets:
            A["ce"][s] = {st: float(np.mean([mean4(C[k][0][st][s]) for k in ks])) for st in steps}
            dr = {st: [mean4(C[k][0][0][s]) - mean4(C[k][0][st][s]) for k in ks] for st in steps}
            A["drop"][s] = {st: float(np.mean(v)) for st, v in dr.items()}
            A["seed_diff"][s] = {st: (float(abs(v[0] - v[1])) if len(v) > 1 else None) for st, v in dr.items()}
            A["drop_task"][s] = {t: {st: float(np.mean([C[k][0][0][s][t] - C[k][0][st][s][t] for k in ks]))
                                     for st in steps} for t in TASKS}
            # paired per-row difference vs dense at the same step and seed (4-task mean of per-task means)
            se = {}
            for st in steps:
                diffs = []
                for k in ks:
                    dk = ("dense", k[1])
                    if dk not in C:
                        continue
                    # SE of the 4-task mean difference: per task var/n, averaged over 4 tasks
                    var = 0.0
                    for t in TASKS:
                        dd = C[k][1][(st, s, t)] - C[dk][1][(st, s, t)]
                        var += dd.var(ddof=1) / len(dd)
                    diffs.append(np.sqrt(var) / len(TASKS))
                se[st] = float(np.mean(diffs)) if diffs else None
            A["paired_se_vs_dense"][s] = se
        res["arms"][arm] = A

    # fractions vs dense (same step; same FLOPs)
    if "dense" in res["arms"]:
        D = res["arms"]["dense"]
        dense_pf = {st: np.mean([meta[f"dense/sd{k[1]}"]["pf_at_probe"][st] or 0.0 for k in dense_keys])
                    for st in steps}
        xs = np.array([dense_pf[st] for st in steps])
        for arm, A in res["arms"].items():
            ks = [k for k in C if k[0] == arm]
            for s in sets:
                ys = np.array([D["drop"][s][st] for st in steps])
                A["frac_step"][s] = {st: (A["drop"][s][st] / D["drop"][s][st] if D["drop"][s][st] else None)
                                     for st in steps}
                fr = {}
                for st in steps:
                    pf = np.mean([meta[f"{arm}/sd{k[1]}"]["pf_at_probe"][st] or 0.0 for k in ks])
                    dd = float(np.interp(pf, xs, ys)) if pf <= xs.max() else None
                    fr[st] = {"pf": float(pf), "dense_drop_same_pf": dd,
                              "frac": (A["drop"][s][st] / dd if dd else None)}
                A["frac_flops"][s] = fr

    # rank correlations: over arms with a known outcome
    known = [arm for arm in res["arms"] if arm in FINAL]
    for s in sets:
        for st in steps:
            if st == 0:
                continue
            x = [res["arms"][arm]["drop"][s][st] for arm in known]
            y = [final_mean(arm) for arm in known]
            res["corr"][f"{s}/step{st}/mean"] = {"spearman": spearman(x, y), "n_arms": len(known)}
            for ti, t in enumerate(TASKS):
                x = [res["arms"][arm]["drop_task"][s][t][st] for arm in known]
                y = [FINAL[arm][ti] for arm in known]
                res["corr"][f"{s}/step{st}/{t}"] = {"spearman": spearman(x, y), "n_arms": len(known)}
            # same, method arms only (dense and dense LoRA have different LRs/parameterisation)
            meth = [arm for arm in known if arm not in ("dense", "denselora")]
            if len(meth) >= 3:
                x = [res["arms"][arm]["drop"][s][st] for arm in meth]
                res["corr"][f"{s}/step{st}/mean_methods_only"] = {
                    "spearman": spearman(x, [final_mean(arm) for arm in meth]), "n_arms": len(meth)}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")

    # console table
    s = a.fig_set
    print(f"\n{s}: 4-task mean dense eval CE by step (seed mean); drop = CE0 - CE(step)")
    print(f"{'arm':12s} {'final':>6s} " + " ".join(f"{'CE'+str(st):>7s}" for st in steps)
          + " " + " ".join(f"{'d'+str(st):>7s}" for st in steps if st))
    for arm, A in res["arms"].items():
        print(f"{arm:12s} {A['final_8k'] or float('nan'):6.3f} " + " ".join(f"{A['ce'][s][st]:7.4f}" for st in steps)
              + " " + " ".join(f"{A['drop'][s][st]:7.4f}" for st in steps if st))
    for k, v in res["corr"].items():
        if k.startswith(s) and (k.endswith("/mean") or k.endswith("methods_only")):
            print(f"  spearman {k}: {v['spearman']:+.3f} (n_arms={v['n_arms']})")

    if a.fig:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        cmap = plt.get_cmap("viridis")
        fm = [res["arms"][arm]["final_8k"] or 0.6 for arm in res["arms"]]
        lo, hi = min(fm), max(fm)
        for arm, A in res["arms"].items():
            col = cmap((A["final_8k"] - lo) / max(1e-9, hi - lo)) if A["final_8k"] else "grey"
            ls = "--" if arm in ("dense", "denselora") else "-"
            ax[0].plot(steps, [A["ce"][s][st] for st in steps], ls, marker="o", ms=3, color=col,
                       label=f"{arm} ({A['final_8k']:.3f})" if A["final_8k"] else arm)
        ax[0].set_xlabel("training step")
        ax[0].set_ylabel(f"dense eval answer CE ({s}, 4-task mean)")
        ax[0].set_title("Dense-forward held-out CE during early training")
        ax[0].legend(fontsize=7, ncol=2)
        st = a.fig_step
        for arm, A in res["arms"].items():
            if not A["final_8k"]:
                continue
            ks = [k for k in C if k[0] == arm]
            per = [mean4(C[k][0][0][s]) - mean4(C[k][0][st][s]) for k in ks]
            ax[1].errorbar(A["drop"][s][st], A["final_8k"],
                           xerr=(max(per) - min(per)) / 2 if len(per) > 1 else None, fmt="o", ms=5,
                           color="C3" if arm in ("dense", "denselora") else "C0")
            ax[1].annotate(arm, (A["drop"][s][st], A["final_8k"]), fontsize=7, xytext=(3, 3),
                           textcoords="offset points")
        c = res["corr"].get(f"{s}/step{st}/mean", {})
        ax[1].set_xlabel(f"dense eval CE drop by step {st} ({s}; bar = seed range)")
        ax[1].set_ylabel("final 8k 4-task mean (eval_size 500/task)")
        ax[1].set_title(f"Proxy vs outcome: Spearman {c.get('spearman', float('nan')):+.2f} "
                        f"over {c.get('n_arms')} arms")
        fig.tight_layout()
        fig.savefig(a.fig, dpi=130)
        print(f"wrote {a.fig}")


if __name__ == "__main__":
    main()
