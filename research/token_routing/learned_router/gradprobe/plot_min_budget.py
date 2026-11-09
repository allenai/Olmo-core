"""Smallest lossless keep budget vs context length, from grad_importance.py JSONs.

    python plot_min_budget.py --rules gold_ga,a0.5_abs --tau 0.05 --out min_budget.png \\
        "per-task:nq:8k=a.json" "per-task:nq:16k=b.json" ... "setA:nq:8k=c.json" ...

Per row, rho* = the smallest grid keep ratio rho such that dCE <= tau at rho AND at every larger grid rho (a lucky
isolated grid point does not count); a row that never qualifies gets rho* = 1. Panels: one column per task; top row
rho* (fraction of document tokens, log), bottom row rho* x n_body (absolute document tokens kept, log). Lines = model x
rule; markers = per-row median, whiskers = per-row min..max. The grid floor (smallest rho) is drawn dotted.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
RUNG_TOK = {"2k": 2048, "8k": 8192, "16k": 16384, "32k": 32768}
LABEL = {"gold_ga": "gold first, then |grad| @ g=0.5", "a0.5_abs": "|grad| @ g=0.5 (no gold)", "ig_abs": "|IG| 16 steps",
         "grad_abs": "|grad| @ g=1"}


def rho_star(row, rule, tau):
    pts = sorted(row["curves"].get(rule, []), key=lambda q: q["rho"])
    if not pts:
        return None
    best = 1.0
    for i in range(len(pts) - 1, -1, -1):  # walk down from the largest rho while every point so far passes
        if pts[i]["dce"] <= tau:
            best = pts[i]["rho"]
        else:
            break
    return best


def rho_star_agg(rows, rule, tau, idx=None):
    """rho* of the across-row MEDIAN dCE curve (same strict walk-down rule); idx = bootstrap row indices."""
    rows = rows if idx is None else [rows[i] for i in idx]
    by = defaultdict(list)
    for r in rows:
        for q in r["curves"].get(rule, []):
            by[round(q["rho"], 6)].append(q["dce"])
    if not by:
        return None
    best = 1.0
    for rho in sorted(by, reverse=True):
        if np.median(by[rho]) <= tau:
            best = rho
        else:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="model:task:rung=json")
    ap.add_argument("--rules", default="gold_ga,a0.5_abs")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--agg", choices=["row", "median_curve"], default="median_curve",
                    help="row: per-row rho* (median, min..max); median_curve: rho* of the across-row median dCE curve "
                         "(robust to single-row spikes), whiskers = 10-90%% bootstrap over rows")
    ap.add_argument("--boot", type=int, default=400)
    a = ap.parse_args()
    data = defaultdict(dict)  # (model, task) -> rung -> rows
    tasks, models = [], []
    for spec in a.inputs:
        key, path = spec.split("=", 1)
        model, task, rung = key.split(":")
        data[(model, task)][rung] = json.load(open(path))["rows"]
        tasks += [task] if task not in tasks else []
        models += [model] if model not in models else []
    rules = a.rules.split(",")
    fig, axes = plt.subplots(2, len(tasks), figsize=(4.6 * len(tasks), 7.2), squeeze=False)
    floor = None
    table = []
    for ci, task in enumerate(tasks):
        for ri in range(2):
            ax = axes[ri][ci]
            ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(colors=INK2, labelsize=9)
            ax.set_xscale("log", base=2); ax.set_yscale("log")
        si = 0
        for mi, model in enumerate(models):
            for rule in rules:
                rungs = sorted(data[(model, task)], key=lambda g: RUNG_TOK[g])
                xs, med, lo, hi, tmed, tlo, thi, ns = [], [], [], [], [], [], [], []
                for g in rungs:
                    rows = data[(model, task)][g]
                    if a.agg == "median_curve":
                        r0 = rho_star_agg(rows, rule, a.tau)
                        if r0 is None:
                            continue
                        rng = np.random.default_rng(0)
                        bs = [rho_star_agg(rows, rule, a.tau, rng.integers(0, len(rows), len(rows))) for _ in range(a.boot)]
                        nb = float(np.median([r["n_body"] for r in rows]))
                        rs = [r0, float(np.percentile(bs, 10)), float(np.percentile(bs, 90))]
                        toks = [v * nb for v in rs]
                        xs.append(RUNG_TOK[g]); ns.append(len(rows))
                        med.append(rs[0]); lo.append(min(rs)); hi.append(max(rs))
                        tmed.append(toks[0]); tlo.append(min(toks)); thi.append(max(toks))
                        grid_min = min(q["rho"] for r in rows for q in r["curves"][rule])
                        floor = grid_min if floor is None else min(floor, grid_min)
                        table.append((task, model, rule, g, len(rows), rs[0], rs[1], rs[2], toks[0]))
                        continue
                    rs = [rho_star(r, rule, a.tau) for r in rows]
                    if any(v is None for v in rs):
                        continue
                    toks = [v * r["n_body"] for v, r in zip(rs, rows)]
                    grid_min = min(q["rho"] for r in rows for q in r["curves"][rule])
                    floor = grid_min if floor is None else min(floor, grid_min)
                    xs.append(RUNG_TOK[g]); ns.append(len(rows))
                    med.append(np.median(rs)); lo.append(min(rs)); hi.append(max(rs))
                    tmed.append(np.median(toks)); tlo.append(min(toks)); thi.append(max(toks))
                    table.append((task, model, rule, g, len(rows), np.median(rs), min(rs), max(rs), np.median(toks)))
                if not xs:
                    continue
                col = SERIES[si % len(SERIES)]
                ls = "-" if mi == 0 else "--"
                mk = "o" if rule == rules[0] else "s"
                lab = f"{model}: {LABEL.get(rule, rule)}"
                jit = 1.0 + 0.04 * (si - 1.5)  # small x offset so whiskers do not overlap
                for ri, (m_, l_, h_) in enumerate(((med, lo, hi), (tmed, tlo, thi))):
                    ax = axes[ri][ci]
                    xj = np.array(xs) * jit
                    ax.errorbar(xj, m_, yerr=[np.array(m_) - np.array(l_), np.array(h_) - np.array(m_)], color=col, lw=2, ls=ls,
                                marker=mk, ms=6, mec="white", mew=0.8, capsize=3, elinewidth=1, label=lab if ci == 0 and ri == 0 else None)
                si += 1
        for ri in range(2):
            ax = axes[ri][ci]
            ax.set_xlim(6800, 39000); ax.set_xticks([8192, 16384, 32768]); ax.set_xticklabels(["8k", "16k", "32k"])
            ax.minorticks_off()
        axes[0][ci].set_title(task, fontsize=11, color=INK, loc="left")
        if floor:
            axes[0][ci].axhline(floor, color=INK2, lw=1, ls=":", zorder=1)
        axes[1][ci].set_xlabel("context length (tokens)", fontsize=9, color=INK2)
    axes[0][0].set_ylabel(f"smallest lossless keep ratio ρ*\n(ΔCE ≤ {a.tau:g} at ρ* and above)", fontsize=9.5, color=INK2)
    axes[1][0].set_ylabel("document tokens kept at ρ*", fontsize=9.5, color=INK2)
    for ci in range(len(tasks)):
        ax = axes[0][ci]
        ax.set_yticks([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
        ax.set_yticklabels(["0.05%", "0.1%", "0.2%", "0.5%", "1%", "2%", "5%", "10%", "20%", "50%", "100%"])
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(4, len(l)), frameon=False, fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, -0.005))
    if a.title:
        fig.suptitle(a.title, fontsize=12, color=INK, x=0.01, ha="left", y=0.995)
    how = ("ρ* of the across-row median ΔCE curve, whiskers = 10–90% bootstrap over rows" if a.agg == "median_curve"
           else "markers = per-row median ρ*, whiskers = min..max over rows")
    fig.text(0.01, 0.955 if a.title else 0.985, how + " · dotted = smallest keep ratio on the grid (values there are an upper bound) · "
             "16 rows per cell ⚠", ha="left", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94 if a.title else 0.97))
    fig.savefig(a.out, dpi=160, facecolor="white")
    print("wrote", a.out)
    print(f"{'task':14s}{'model':10s}{'rule':10s}{'rung':5s} eval_size  rho* median [min..max]   tokens@median")
    for t, m, ru, g, n, md, mn, mx, tk in table:
        print(f"{t:14s}{m:10s}{ru:10s}{g:5s} {n:3d}        {md:.3f} [{mn:.3f}..{mx:.3f}]     {tk:7.0f}")


if __name__ == "__main__":
    main()
