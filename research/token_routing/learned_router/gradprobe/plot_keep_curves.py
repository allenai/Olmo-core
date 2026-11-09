"""Keep ratio vs answer CE (dev loss) from grad_importance.py JSONs.

    # one panel per task, one line per rule (all JSONs at one rung)
    python research/token_routing/learned_router/gradprobe/plot_keep_curves.py --by rule --out keep_8k.png \\
        nq=out/nq_8k_r16.json outlier=out/outlier_8k_r16.json contradiction=out/contradiction_8k_r16.json
    # one panel per task, one line per rung (one rule)
    python .../plot_keep_curves.py --by rung --rule ig_abs --out keep_len.png nq:8k=a.json nq:16k=b.json ...

Mean over rows with ±1 SE bands; rho = 1 is the full-context CE (dashed line). Markers: the bar
(gold_fl20p8_noslot, at its own keep ratio) and gold-only. Stdlib + numpy + matplotlib (login-node safe).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# validated categorical slots (dataviz reference palette, light), fixed order; random = neutral gray baseline
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GRAY, INK, INK2, GRID = "#8a8985", "#0b0b0b", "#52514e", "#e6e5e1"
MARKERS = ["o", "s", "^", "D", "v"]
LABEL = {"gold_rand": "gold first, then random", "gold_ga": "gold first, then |grad| @ g=0.5", "a0.5_abs": "|grad| @ g=0.5 (1 pass)",
         "rand": "random", "grad_abs": "|grad| (1 pass)", "iter_abs_c": "|grad| re-scored", "iter2_abs": "|grad| re-scored, 2 steps",
         "iter3_abs": "|grad| re-scored, 3 steps", "ig_abs": "|IG| 16 steps", "ig8_abs": "|IG| 8 steps", "ig4_abs": "|IG| 4 steps",
         "ig3_abs": "|IG| 3 steps", "ig2_abs": "|IG| 2 steps", "ig1_abs": "|IG| 1 step (g=0.5)", "ig": "IG signed 16"}


DELTA = False  # --delta: plot dCE = CE - CE_full per row (compares row sets with different full-context CE)


def curve(rows, rule):
    """{rho: (mean CE, SE)} for one rule (rand pooled over seeds per row); rho = 1 -> CE_full."""
    by = defaultdict(list)
    for r in rows:
        acc = defaultdict(list)
        for n, pts in r["curves"].items():
            if n == rule or (rule == "rand" and n.startswith("rand_s")):
                for q in pts:
                    acc[round(q["rho"], 4)].append(q["dce"] if DELTA else q["ce"])
        for rho, v in acc.items():
            by[rho].append(float(np.mean(v)))
        by[1.0].append(0.0 if DELTA else r["ce_full"])
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0)
            for k, v in sorted(by.items())}


def ref(rows, name):
    pts = [r["curves"][name][0] for r in rows if name in r["curves"]]
    return (float(np.mean([q["rho"] for q in pts])), float(np.mean([q["ce"] for q in pts]))) if pts else None


def style(ax):
    ax.set_xscale("log")
    ax.set_xlim(0.0004, 1.15)
    ax.set_xticks([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    ax.set_xticklabels(["0.05%", "0.1%", "0.2%", "0.5%", "1%", "2%", "5%", "10%", "20%", "50%", "100%"])
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def plot_alpha(panels, tasks, a):
    """One panel per task: x = gate value alpha of the one-pass gradient (aX_abs), y = CE, one line per keep ratio;
    horizontal references = |grad| at g = 1 (grad_abs) at the same keep ratios (dotted, same color)."""
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 3.7), squeeze=False)
    rhos = [float(v) for v in a.alpha_rhos.split(",")]
    for ax, task in zip(axes[0], tasks):
        rows = panels[task]["_"]
        alphas = sorted({float(n[1:-4]) for r in rows for n in r["curves"] if n.startswith("a") and n.endswith("_abs") and n[1].isdigit()})
        ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=9)
        full = float(np.mean([r["ce_full"] for r in rows]))
        ax.axhline(full, color=INK2, lw=1, ls="--", zorder=1)
        for ci, rho in enumerate(rhos):
            col, mk = SERIES[ci % len(SERIES)], MARKERS[ci % len(MARKERS)]
            ms_, ss_ = [], []
            for al in alphas:
                c = curve(rows, f"a{al:g}_abs")
                ms_.append(c[round(rho, 4)][0]); ss_.append(c[round(rho, 4)][1])
            m, se = np.array(ms_), np.array(ss_)
            ax.fill_between(alphas, m - se, m + se, color=col, alpha=0.13, lw=0)
            ax.plot(alphas, m, color=col, lw=2, marker=mk, ms=6, mec="white", mew=0.8, label=f"keep {rho:.0%}")
            g1 = curve(rows, "grad_abs").get(round(rho, 4))
            if g1:
                ax.axhline(g1[0], color=col, lw=1.2, ls=":", zorder=1)
        ax.set_xticks(alphas)
        ax.set_xlabel("gate value g where the single gradient pass is taken", fontsize=9, color=INK2)
        ax.set_title(f"{task}   ⚠ eval_size={len(rows)}", fontsize=11, color=INK, loc="left")
        ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper center", ncol=3)
    axes[0][0].set_ylabel("answer CE (dev loss)", fontsize=10, color=INK2)
    if a.title:
        fig.suptitle(a.title, fontsize=12, color=INK, x=0.01, ha="left", y=0.99)
    fig.text(0.01, 0.905 if a.title else 0.97, "dotted = |grad| at g = 1 (plain gradient) at the same keep ratio · dashed = full-context CE · bands = ±1 SE",
             ha="left", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.88 if a.title else 0.95))
    fig.savefig(a.out, dpi=160, facecolor="white")
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="task=json (--by rule) or task:rung=json (--by rung)")
    ap.add_argument("--by", choices=["rule", "rung", "alpha"], default="rule")
    ap.add_argument("--alpha-rhos", default="0.2,0.1,0.05", help="--by alpha: one line per keep ratio")
    ap.add_argument("--rules", default="rand,grad_abs,iter_abs_c,ig4_abs,ig_abs")
    ap.add_argument("--rule", default="ig_abs", help="the rule plotted with --by rung")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--delta", action="store_true", help="y = dCE vs each row's full-context CE")
    a = ap.parse_args()
    global DELTA
    DELTA = a.delta
    panels = defaultdict(dict)  # task -> {series key: rows}
    for spec in a.inputs:
        key, path = spec.split("=", 1)
        d = json.load(open(path))
        task, sub = (key.split(":", 1) + [d["rung"]])[:2] if a.by == "rung" else (key, None)
        panels[task][sub or "_"] = d["rows"]
    tasks = list(panels)
    if a.by == "alpha":
        return plot_alpha(panels, tasks, a)
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 3.9), squeeze=False)
    handles = {}
    for ax, task in zip(axes[0], tasks):
        style(ax)
        series = panels[task]
        if a.by == "rule":
            rows = series["_"]
            keys = a.rules.split(",")
            eval_size = len(rows)
            full = float(np.mean([r["ce_full"] for r in rows]))
            ax.axhline(full, color=INK2, lw=1, ls="--", zorder=1)
            ci = 0
            for rule in keys:
                c = curve(rows, rule)
                if len(c) <= 1:
                    continue
                x = np.array(list(c)); m = np.array([v[0] for v in c.values()]); se = np.array([v[1] for v in c.values()])
                if rule == "rand":
                    col, mk, ls = GRAY, "x", ":"
                else:
                    col, mk, ls = SERIES[ci % len(SERIES)], MARKERS[ci % len(MARKERS)], "-"
                    ci += 1
                ax.fill_between(x, m - se, m + se, color=col, alpha=0.13, lw=0, zorder=2)
                h, = ax.plot(x, m, color=col, lw=2, ls=ls, marker=mk, ms=6, mec="white", mew=0.8, zorder=3, label=LABEL.get(rule, rule))
                handles.setdefault(rule, h)
            for name, mk in (("bar", "*"), ("gold_only", "P")):
                rr = ref(rows, name)
                if rr:
                    h = ax.scatter([rr[0]], [rr[1]], marker=mk, s=110, color=INK, edgecolor="white", lw=0.8, zorder=4,
                                   label={"bar": "heuristic bar (own ρ)", "gold_only": "gold docs only"}[name])
                    handles.setdefault(name, h)
            ax.set_title(f"{task}   ⚠ eval_size={eval_size}", fontsize=11, color=INK, loc="left")
        else:
            for ci, (rung, rows) in enumerate(sorted(series.items(), key=lambda kv: int(kv[0].rstrip("k")))):
                c = curve(rows, a.rule)
                x = np.array(list(c)); m = np.array([v[0] for v in c.values()]); se = np.array([v[1] for v in c.values()])
                col, mk = SERIES[ci % len(SERIES)], MARKERS[ci % len(MARKERS)]
                ax.fill_between(x, m - se, m + se, color=col, alpha=0.13, lw=0, zorder=2)
                h, = ax.plot(x, m, color=col, lw=2, marker=mk, ms=6, mec="white", mew=0.8, zorder=3,
                             label=f"{rung} (eval_size={len(rows)})")
                ax.axhline(0.0 if DELTA else float(np.mean([r["ce_full"] for r in rows])), color=col if not DELTA else INK2,
                           lw=1, ls="--", zorder=1)
                ax.plot([], [], " ", label="")  # spacer keeps legend order stable
                ax.legend(fontsize=8, frameon=False, loc="upper right", labelcolor=INK2)
            ax.set_title(f"{task} — {LABEL.get(a.rule, a.rule)}", fontsize=11, color=INK, loc="left")
        ax.set_xlabel("keep ratio ρ (fraction of document tokens kept; log scale)", fontsize=9, color=INK2)
    axes[0][0].set_ylabel("ΔCE vs full context (nats)" if DELTA else "answer CE (dev loss)", fontsize=10, color=INK2)
    if a.by == "rule":
        fig.legend(list(handles.values()), [h.get_label() for h in handles.values()], loc="lower center",
                   ncol=min(7, len(handles)), frameon=False, fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, -0.02))
    if a.title:
        fig.suptitle(a.title, fontsize=12, color=INK, x=0.01, ha="left", y=0.99)
    fig.text(0.01, 0.905 if a.title else 0.97, ("dashed line = no change from full context" if DELTA else
             "dashed line = full-context CE (ρ = 100%)") + " · bands = ±1 SE over rows",
             ha="left", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.07 if a.by == "rule" else 0, 1, 0.88 if a.title else 0.95))
    fig.savefig(a.out, dpi=160, facecolor="white")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
