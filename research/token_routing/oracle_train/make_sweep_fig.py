"""Pareto figure for the oracle threshold sweep, from collect_sweep.py's sweep_table.json.

x = training FLOPs relative to dense at 1,952 rows/task (FLOP meter `flop_meter/actual_pflops`; for the 1,952-row
arms this is the routed step cost, since steps are equal; dense at 488 / 976 / 1,952 rows/task sits at 0.25 / 0.5 / 1.0,
i.e. dense at matched total FLOPs). Left: 8k 4-task mean. Right: outlier@8k.

    python3 research/token_routing/oracle_train/make_sweep_fig.py [--out research/token_routing/oracle_train/sweep_pareto.png]
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORACLE, RANDOM, RECIPE = "#2a78d6", "#eb6834", "#1baf7a"  # reference palette slots 1-3 (all-pairs validated)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=os.path.join(HERE, "sweep_table.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "sweep_pareto.png"))
    a = ap.parse_args()
    T = json.load(open(a.table))
    dense = [d for d in T["dense"] if d["pf"] and d["y8"] is not None]
    ref = next(d["pf"] for d in T["dense"] if d["rows"] == 1952)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for ax, key, tkey, title in ((axes[0], "y8", None, "8k rung, 4-task mean"), (axes[1], "out8", "outlier", "outlier @ 8k")):
        ds = sorted((d["pf"] / ref, d[key]) for d in dense if d[key] is not None)
        ax.plot([x for x, _ in ds], [y for _, y in ds], "-", color=INK2, lw=2, marker="o", ms=5, mfc="white", mec=INK2,
                label="dense (fewer rows = matched total FLOPs)", zorder=2)
        for (x, y), d in zip(ds, sorted((d for d in dense if d[key] is not None), key=lambda d: d["pf"])):
            ax.annotate(f"{d['rows']:,}", (x, y), textcoords="offset points", xytext=(5, -11), fontsize=8, color=INK2)
        val = (lambda r: r["y8"]) if tkey is None else (lambda r: r["t8"].get(tkey))
        orc = sorted((r for r in T["arms"] if r["kind"] == "oracle" and r["step_cost"] and val(r) is not None),
                     key=lambda r: r["step_cost"])
        ax.plot([r["step_cost"] for r in orc], [val(r) for r in orc], "-", color=ORACLE, lw=2, marker="o", ms=8,
                mec="white", mew=1.5, label="oracle, per document (attribution-cos target)", zorder=4)
        for r in orc:
            ax.annotate(f"{r['target']:.2f}", (r["step_cost"], val(r)), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=INK)
        rnd = [r for r in T["arms"] if r["kind"] == "random" and r["step_cost"] and val(r) is not None]
        if rnd:
            ax.plot([r["step_cost"] for r in rnd], [val(r) for r in rnd], "s", color=RANDOM, ms=8, mec="white", mew=1.5,
                    label="random, per document (matched keep)", zorder=4)
            for r in rnd:
                ax.annotate(f"{100 * r['keep']:.0f}%" if r.get("keep") else "", (r["step_cost"], val(r)),
                            textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color=INK)
        row = [r for r in T["arms"] if r["kind"] == "oracle_row" and r["step_cost"] and val(r) is not None]
        if row:  # a different selection scope: its own marker, not joined to the per-document curve
            ax.plot([r["step_cost"] for r in row], [val(r) for r in row], "^", color="white", ms=9, mec=ORACLE, mew=2,
                    label="oracle per ROW 0.90 + random x1.15 (different scope)", zorder=4)
            for r in row:
                ax.annotate(f"row {100 * r['keep']:.0f}%" if r.get("keep") else "row", (r["step_cost"], val(r)),
                            textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8, color=INK)
        rec = [r for r in T["arms"] if r["kind"] == "recipe" and r["step_cost"] and val(r) is not None]
        if rec:
            ax.plot([r["step_cost"] for r in rec], [val(r) for r in rec], "D", color=RECIPE, ms=8, mec="white", mew=1.5,
                    label="recipe (router, two cuts 20%/7%)", zorder=4)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.set_xlabel("training FLOPs / dense at 1,952 rows/task  (= routed step cost for the 1,952-row arms)",
                      fontsize=9, color=INK2)
    axes[0].set_ylabel("f1 (oolong: score), eval_size 500/cell, no-cot", fontsize=9, color=INK2)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Oracle threshold sweep: quality vs step cost (1,952 rows/task, single cut at block 7, 1 seed each)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, facecolor="#fcfcfb")
    print(a.out)


if __name__ == "__main__":
    main()
