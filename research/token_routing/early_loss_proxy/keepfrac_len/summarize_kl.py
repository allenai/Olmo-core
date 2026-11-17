"""Per-length summary + charts for keepfrac_len from analyze_kl.py's JSON (--out).

Headline metric per length L and arm: FLOP speedup vs dense full fine-tune (dft) at a FIXED matched held-out CE,
x = PF_dft(level) / PF_arm(level). Two fixed levels per length (dft's own mean CE at two probe steps):
  mid  = dft's CE at the probe nearest 50% of its run   (the headline: a level the arm must actually reach)
  deep = dft's CE at its last probe                      (end of the epoch)
An arm that never reaches a level gets x = None ("not reached") and is ranked below every arm that reaches it;
its final CE is reported instead. Same per task (dft's per-task CE at the same probe step); worst task = min.
Seeds: x per seed, mean and spread reported.

    python3 summarize_kl.py --scored kl_results.json --out-json kl_summary.json --png keepfrac_vs_length.png
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

ORDER = ["dft", "ref", "k70", "k40", "k20", "k10", "k5", "k2p5", "k1p25", "grJ", "grJc"]
LENS = {"2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768, "64k": 65536}


def reach(pf, ce, lvl):
    best = 9e9
    for (p0, l0), (p1, l1) in zip(zip(pf, ce), list(zip(pf, ce))[1:]):
        l0 = min(best, l0)
        best = l0
        if l0 <= lvl:
            return p0
        if l1 <= lvl:
            return p0 + (l0 - lvl) / (l0 - l1) * (p1 - p0)
    return None


def xat(run, ref, lvl, task=None):
    c = (lambda r: r["ce"]) if task is None else (lambda r: r["by_task"][task])
    a, r = reach(run["pf"], c(run), lvl), reach(ref["pf"], c(ref), lvl)
    return (r / a) if a and r else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--png", default="")
    ap.add_argument("--ref", default="dft")
    a = ap.parse_args()
    S = json.load(open(a.scored))["lens"]
    out = {}
    for L, R in S.items():
        arms = R["arms"]
        ref = arms[a.ref].get("0") or arms[a.ref].get(0)
        steps = ref["steps"]
        last = steps[-1]
        i_mid = min(range(1, len(steps)), key=lambda i: abs(steps[i] - 0.5 * last))
        lv = {"mid": ref["ce"][i_mid], "deep": ref["ce"][-1]}
        tasks = sorted(ref["by_task"])
        tl = {t: {"mid": ref["by_task"][t][i_mid], "deep": ref["by_task"][t][-1]} for t in tasks}
        rows = {}
        for arm in sorted(arms, key=lambda k: ORDER.index(k) if k in ORDER else 99):
            per_seed = []
            for sd, run in sorted(arms[arm].items()):
                e = {"seed": int(sd), "final_ce": run["ce"][-1], "final_step": run["steps"][-1], "pf": run["pf"][-1],
                     "x_dense_step": run["x_dense_step"], "peak_gib": run["peak_gib"], "done": run.get("done"),
                     "keep": run.get("keep")}
                for k in ("mid", "deep"):
                    e[f"x_{k}"] = xat(run, ref, lv[k])
                    e[f"task_x_{k}"] = {t: xat(run, ref, tl[t][k], t) for t in tasks}
                    tx = e[f"task_x_{k}"]
                    e[f"worst_x_{k}"] = None if any(v is None for v in tx.values()) else min(tx.values())
                    e[f"worst_task_{k}"] = min(tasks, key=lambda t: tx[t] if tx[t] is not None else -1)
                e["task_final_ce"] = {t: run["by_task"][t][-1] for t in tasks}
                per_seed.append(e)
            xs = [e["x_mid"] for e in per_seed]
            rows[arm] = {"seeds": per_seed, "n_seeds": len(per_seed),
                         "x_mid_mean": float(np.mean(xs)) if all(x is not None for x in xs) else None,
                         "x_mid_range": [min(xs), max(xs)] if all(x is not None for x in xs) else None,
                         "final_ce_mean": float(np.mean([e["final_ce"] for e in per_seed]))}
        router = {k: v for k, v in rows.items() if k.startswith("k")}
        best = max(router, key=lambda k: (rows[k]["x_mid_mean"] is not None, rows[k]["x_mid_mean"] or -rows[k]["final_ce_mean"]))
        out[L] = {"levels": lv, "level_step_mid": steps[i_mid], "task_levels": tl, "rows": rows, "best_router_arm": best,
                  "eval_size": R.get("eval_size"), "rows_per_task": R.get("rows_per_task")}
        print(f"\n=== {L}: eval_size {R.get('eval_size')}; levels mid(step {steps[i_mid]})={lv['mid']:.4f} deep(step {last})={lv['deep']:.4f}; best router arm {best}")
        f = lambda x: "  -  " if x is None else f"{x:5.2f}"  # noqa: E731
        for arm, r in rows.items():
            for e in r["seeds"]:
                print(f"{arm:6s} sd{e['seed']} step {e['final_step']:3d} CE {e['final_ce']:.4f} x/dense {e['x_dense_step']:.3f} "
                      f"x_mid {f(e['x_mid'])} x_deep {f(e['x_deep'])} worst_mid {f(e['worst_x_mid'])} ({e['worst_task_mid']}) | "
                      + " ".join(f"{t[:6]}={f(e['task_x_mid'][t])}" for t in tasks)
                      + " | CE " + " ".join(f"{t[:6]}={e['task_final_ce'][t]:.3f}" for t in tasks))
    if a.out_json:
        json.dump(out, open(a.out_json, "w"), indent=1)
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
        cmap = plt.get_cmap("viridis")
        Ls = [L for L in LENS if L in out]
        for i, L in enumerate(Ls):
            pts = [(r["seeds"][0]["keep"][0], r["x_mid_mean"]) for k, r in out[L]["rows"].items()
                   if k.startswith("k") and r["seeds"][0]["keep"] and r["x_mid_mean"] is not None]
            miss = [r["seeds"][0]["keep"][0] for k, r in out[L]["rows"].items()
                    if k.startswith("k") and r["seeds"][0]["keep"] and r["x_mid_mean"] is None]
            pts.sort()
            col = cmap(i / max(1, len(Ls) - 1))
            if pts:
                ax[0].plot([p[0] * 100 for p in pts], [p[1] for p in pts], "o-", color=col, label=L)
            if miss:
                ax[0].plot([m * 100 for m in miss], [0.3] * len(miss), "x", color=col)
        ax[0].set_xscale("log")
        ax[0].set_xlabel("keep at cut 1 (%; cut 2 = 0.35x)")
        ax[0].set_ylabel("FLOP speedup vs dense full FT at matched held-out CE")
        ax[0].axhline(1, color="gray", lw=0.6)
        ax[0].set_title("Speedup at dft's mid-epoch CE (x at 0.3 = never reached)")
        ax[0].legend(title="length")
        xs = [LENS[L] for L in Ls]
        bk = [out[L]["rows"][out[L]["best_router_arm"]]["seeds"][0]["keep"][0] for L in Ls]
        ax[1].plot(xs, [b * 100 for b in bk], "o-", label="best keep fraction (%)")
        ax[1].set_xscale("log", base=2)
        ax[1].set_yscale("log")
        ax[1].set_xticks(xs, Ls)
        ax[1].set_xlabel("context length")
        ax[1].set_ylabel("best keep at cut 1 (%)")
        ax2 = ax[1].twinx()
        ax2.plot(xs, [b * x for b, x in zip(bk, xs)], "s--", color="C1", label="kept tokens at cut 1")
        ax2.set_yscale("log")
        ax2.set_ylabel("kept tokens at cut 1", color="C1")
        ax[1].set_title("Best keep vs length")
        fig.tight_layout()
        fig.savefig(a.png, dpi=130)


if __name__ == "__main__":
    main()
