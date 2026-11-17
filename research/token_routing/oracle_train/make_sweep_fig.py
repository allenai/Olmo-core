"""Pareto figure for the oracle threshold sweep, from collect_sweep.py's sweep_table.json.

Panels (a) 8k 4-task mean and (b) outlier@8k vs training FLOPs relative to dense at 1,952 rows/task (FLOP meter
`flop_meter/actual_pflops`, log x; for the 1,952-row arms this is the routed step cost, since steps are equal; dense at
488 / 976 / 1,952 rows/task sits at 0.25 / 0.5 / 1.0, i.e. dense at matched total FLOPs). Panel (c): 8k mean vs the
realised body keep fraction, oracle vs random per document, the like-for-like comparison.

    python3 research/token_routing/oracle_train/make_sweep_fig.py [--out research/token_routing/oracle_train/sweep_pareto.png]
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORACLE, RANDOM, RECIPE = "#2a78d6", "#eb6834", "#1baf7a"  # reference palette slots 1-3 (all-pairs validated)


def style(ax, title):
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def label(ax, x, y, txt, dx, dy, ha="center"):
    ax.annotate(txt, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha, va="center", fontsize=8, color=INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=os.path.join(HERE, "sweep_table.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "sweep_pareto.png"))
    a = ap.parse_args()
    T = json.load(open(a.table))
    dense = sorted((d for d in T["dense"] if d["pf"] and d["y8"] is not None), key=lambda d: d["pf"])
    ref = next(d["pf"] for d in T["dense"] if d["rows"] == 1952)
    orc = sorted((r for r in T["arms"] if r["kind"] == "oracle" and r["step_cost"] and r["y8"] is not None),
                 key=lambda r: -r["target"])
    rnd = sorted((r for r in T["arms"] if r["kind"] == "random" and r["step_cost"] and r["y8"] is not None),
                 key=lambda r: r["keep"])
    row = [r for r in T["arms"] if r["kind"] == "oracle_row" and r["step_cost"] and r["y8"] is not None]
    rec = [r for r in T["arms"] if r["kind"] == "recipe" and r["step_cost"] and r["y8"] is not None]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    panels = ((axes[0], lambda r: r["y8"], lambda d: d["y8"], "(a) 8k rung, 4-task mean"),
              (axes[1], lambda r: r["t8"].get("outlier"), lambda d: d["out8"], "(b) outlier @ 8k"))
    for ax, val, dv, title in panels:
        ds = [(d["pf"] / ref, dv(d), d["rows"]) for d in dense if dv(d) is not None]
        ax.plot([x for x, _, _ in ds], [y for _, y, _ in ds], "-", color=INK2, lw=2, marker="o", ms=5, mfc="white",
                mec=INK2, label="dense, LR 5e-6 (fewer rows = matched total FLOPs)", zorder=2)
        for x, y, n in ds:
            label(ax, x, y, f"{n:,}", 0, -11)
        ax.plot([r["step_cost"] for r in orc], [val(r) for r in orc], "-", color=ORACLE, lw=1.5, marker="o", ms=8,
                mec="white", mew=1.5, label="oracle per document (label = attribution-cos target)", zorder=4)
        for r in orc:
            label(ax, r["step_cost"], val(r), f"{r['target']:.2f}", -7, 0, ha="right")
        ax.plot([r["step_cost"] for r in rnd], [val(r) for r in rnd], "s", color=RANDOM, ms=8, mec="white", mew=1.5,
                label="random per document (label = realised keep)", zorder=4)
        for r in rnd:
            label(ax, r["step_cost"], val(r), f"{100 * r['keep']:.0f}%", 7, 0, ha="left")
        for r in row:  # a different selection scope: its own marker, not joined to the per-document curve
            ax.plot([r["step_cost"]], [val(r)], "^", color="white", ms=9, mec=ORACLE, mew=2, zorder=4,
                    label="oracle per ROW 0.90 + random x1.15 (other scope)")
            label(ax, r["step_cost"], val(r), f"row {100 * r['keep']:.0f}%", 8, 0, ha="left")
        for r in rec:
            ax.plot([r["step_cost"]], [val(r)], "D", color=RECIPE, ms=8, mec="white", mew=1.5, zorder=5,
                    label="recipe (learned router, cuts 20% @7 / 7% @16)")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1, 2, 4]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(["0.15", "0.2", "0.25", "0.3", "0.4", "0.5", "0.7", "1", "2", "4"])
        ax.set_xlim(0.14, 4.6)
        ax.set_xlabel("training FLOPs / dense f1952 (log; routed pass only for oracle arms)", fontsize=9, color=INK2)
        style(ax, title)
    axes[0].set_ylabel("f1 (oolong: score), eval_size 500/cell, no-cot", fontsize=9, color=INK2)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    # (c) like-for-like: 8k mean vs realised keep
    ax = axes[2]
    od = sorted(orc, key=lambda r: r["keep"])
    ax.plot([100 * r["keep"] for r in od], [r["y8"] for r in od], "-", color=ORACLE, lw=1.5, marker="o", ms=8,
            mec="white", mew=1.5, label="oracle per document", zorder=4)
    for r in od:
        label(ax, 100 * r["keep"], r["y8"], f"{r['target']:.2f}", -7, 0, ha="right")
    ax.plot([100 * r["keep"] for r in rnd], [r["y8"] for r in rnd], "-", color=RANDOM, lw=1.5, marker="s", ms=8,
            mec="white", mew=1.5, label="random per document", zorder=4)
    d1952 = next(d for d in dense if d["rows"] == 1952)
    ax.axhline(d1952["y8"], color=INK2, lw=1, ls="--")
    label(ax, 3, d1952["y8"], "dense f1952 (1.0x FLOPs)", 0, 7, ha="left")
    ax.set_xlabel("realised body keep fraction (%)", fontsize=9, color=INK2)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    style(ax, "(c) 8k mean at matched keep (1 seed each)")
    fig.suptitle("Oracle threshold sweep: 1,952 rows/task, single cut at block 7, 1 seed per arm "
                 "(4-task mean SE ~0.008; outlier cell SE ~0.02)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, facecolor="#fcfcfb")
    print(a.out)


if __name__ == "__main__":
    main()
