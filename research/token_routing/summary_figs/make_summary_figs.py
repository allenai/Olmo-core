"""Two summary figures drawn from already-recorded numbers (no new runs).

1. cpt_long_range_gradient.png  -- continued pre-training: how much of one step's gradient comes
   from dependencies at each distance, and how few context tokens carry a late chunk's gradient.
   Source: branch claude/cpt-gradmaps-34a768, research/token_routing/grad_concentration/gradmaps_cpt/
   (distance_decomp.json, span_oracle_table_dn.md). Record: records/cpt-gradient-maps.md.
2. mt11_64k_length_generalization.png -- 11-task mix trained on rows <= 64k, scored at 64k/128k/256k:
   token-dropping arms vs dense at the same training FLOPs.
   Source: worktree mt11-lengen-bed337, research/token_routing/mt11_64k/results_lengen/lengen.json.
   Record: records/mt11-64k-length-generalization.md (branch claude/mt11-lengen-bed337).

Run from the repo root on the login node (stdlib + matplotlib, < 5 s):
    python3 research/token_routing/summary_figs/make_summary_figs.py
"""
import json
import math
import os
import subprocess
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CPT_REF = "claude/cpt-gradmaps-34a768"
CPT_DIR = "research/token_routing/grad_concentration/gradmaps_cpt"
LENGEN_JSON = os.path.join(
    REPO, ".claude/worktrees/mt11-lengen-bed337/research/token_routing/mt11_64k/results_lengen/lengen.json"
)

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e2da"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
    }
)


def git_json(path):
    out = subprocess.run(["git", "-C", REPO, "show", f"{CPT_REF}:{path}"], capture_output=True, check=True)
    return json.loads(out.stdout)


def cpt_figure():
    d = git_json(f"{CPT_DIR}/distance_decomp.json")
    base, cpt = d["dist_base_32k"], d["dist_cpt_32k"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw={"width_ratios": [1.25, 1, 1]})

    # A: share of the step gradient per 1,024-token distance bin
    ax = axes[0]
    for run, color, name in ((base, BLUE, "before CPT (base)"), (cpt, ORANGE, "after dense CPT")):
        g = run["gap_share_whole"]
        xs = [k * 1024 for k in range(1, len(g))]
        # one row has 32 chunks, so a gap of k chunks has (32 - k) (target chunk, context chunk) pairs;
        # dividing by that count gives the contribution of ONE context chunk at that distance
        ys = [100 * v / (len(g) - k) for k, v in enumerate(g) if k >= 1]
        print(name, "per-pair % at 1k/4k/16k/29k-31k mean", round(ys[0], 3), round(ys[3], 3), round(ys[15], 3), round(sum(ys[-3:]) / 3, 3))
        pts = [(x, y) for x, y in zip(xs, ys) if y > 0]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=2, marker="o", ms=4, label=name)
        print(name, "same-chunk share", round(g[0], 3), "min per-gap %", round(min(ys), 3), "neg bins", sum(y <= 0 for y in ys))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1024, 2048, 4096, 8192, 16384, 31744])
    ax.set_xticklabels(["1k", "2k", "4k", "8k", "16k", "31k"])
    ax.minorticks_off()
    ax.set_yticks([0.01, 0.03, 0.1, 0.3])
    ax.set_yticklabels(["0.01%", "0.03%", "0.1%", "0.3%"])
    ax.set_xlabel("distance from the target chunk back to the context chunk (tokens)")
    ax.set_ylabel("share of the step gradient from one\n1k context chunk, per 1k target chunk")
    ax.set_title("A context chunk counts ~20x less at 16k back than at 1k")
    ax.legend(loc="upper right")
    ax.text(
        0.03,
        0.05,
        "Not drawn: same 1k chunk (incl. own label)\n= %.0f%% before CPT, %.0f%% after"
        % (100 * base["gap_share_whole"][0], 100 * cpt["gap_share_whole"][0]),
        transform=ax.transAxes,
        fontsize=8.5,
        color=MUTED,
    )

    # B: share of the gradient from > 4k back, by block band
    ax = axes[1]
    bands = ["blocks\n0-6", "blocks\n7-15", "blocks\n16-23", "blocks\n24-31"]
    far = lambda run: [100 * (a + b) for a, b in zip(run["step"]["~4k-16k back"]["band"], run["step"]["> ~16k back"]["band"])]
    w = 0.38
    for off, run, color, name in ((-w / 2, base, BLUE, "before CPT (base)"), (w / 2, cpt, ORANGE, "after dense CPT")):
        vals = far(run)
        bars = ax.bar([i + off for i in range(4)], vals, width=w - 0.03, color=color, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, "%.0f%%" % v if v >= 9.5 else "%.1f%%" % v, ha="center", fontsize=8.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(bands)
    ax.set_ylabel("share of that band's gradient\nfrom more than ~4k tokens back")
    ax.set_ylim(0, 32)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_yticklabels(["0%", "10%", "20%", "30%"])
    ax.grid(axis="x", visible=False)
    ax.set_title("Long-range credit sits in the early blocks")
    ax.legend(loc="upper right")

    # C: context tokens needed to keep a late chunk's context gradient at cosine 0.95 (dense CPT ckpt)
    ax = axes[2]
    settings = ["32k doc,\nlast 1k", "32k doc,\nlast 4k", "64k doc,\nlast 4k"]
    rules = (("oracle selection", [19.9, 24.1, 31.0], AQUA), ("most recent context", [60.2, 66.3, 77.8], VIOLET), ("random", [75.5, 87.4, 71.3], "#b9b7ae"))
    w = 0.27
    for j, (name, vals, color) in enumerate(rules):
        bars = ax.bar([i + (j - 1) * w for i in range(3)], vals, width=w - 0.03, color=color, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, "%.0f%%" % v, ha="center", fontsize=8.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(settings)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("context tokens needed for\ncontext-gradient cosine 0.95")
    ax.grid(axis="x", visible=False)
    ax.set_title("A late chunk needs 20-31% of its context")
    ax.legend(loc="upper left", ncol=3, fontsize=8, columnspacing=0.9, handlelength=1.2)

    fig.suptitle(
        "Continued pre-training: long-range gradient by distance (OLMo 3 7B, frozen checkpoints, one step = 16 rows x 32k; right panel 16 x 32k and 8 x 64k)",
        fontsize=10.5,
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, "cpt_long_range_gradient.png")
    fig.savefig(out, dpi=170)
    print("wrote", out)


def lengen_figure():
    d = json.load(open(LENGEN_JSON))
    rungs = ["r64k", "r128k", "r256k"]
    cell = defaultdict(dict)  # (run, rung) -> task -> (value, se)
    for r in d["rows"]:
        cell[(r["run"], r["rung"])][r["task"]] = (r["value"], r["se"])
    matched = defaultdict(dict)  # (run, rung) -> task -> (score, dense_at_pf, pf)
    for r in d["matched"]:
        matched[(r["run"], r["rung"])][r["task"]] = (r["score"], r["dense_at_pf"], r["pf"])

    def mean_se(run, rung):
        v = cell[(run, rung)]
        assert len(v) == 6, (run, rung, len(v))
        return sum(x[0] for x in v.values()) / 6, math.sqrt(sum(x[1] ** 2 for x in v.values())) / 6

    arms = (
        ("m64-lora64r20to7f16-f4-lr5e-06", "Recipe (LoRA + 20%→7% keep)", ORANGE),
        ("m64-otr10M7-f2-lr5e-06", "Router, 10% keep", AQUA),
        ("m64-otr10M7-f1-lr5e-06", "Router, 10% keep, full epoch", VIOLET),
    )
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    xs = [0, 1, 2]
    for ax, (run, name, color) in zip(axes[:3], arms):
        m = [matched[(run, g)] for g in rungs]
        pf = next(iter(m[0].values()))[2]
        arm = [sum(x[0] for x in mm.values()) / 6 for mm in m]
        den = [sum(x[1] for x in mm.values()) / 6 for mm in m]
        se = [mean_se(run, g)[1] for g in rungs]
        ax.errorbar(xs, arm, yerr=se, color=color, lw=2, marker="o", ms=7, capsize=3, label="token dropping")
        ax.plot(xs, den, color=INK, lw=2, ls="--", marker="s", ms=6, label="dense, same training FLOPs")
        for x, a, b in zip(xs, arm, den):
            ax.text(x, max(a, b) + 0.028, "%+.3f" % (a - b), ha="center", fontsize=9, color=INK)
        ax.axvspan(-0.35, 0.3, color="#f3f1ea", zorder=0)
        ax.text(-0.3, 0.262, "longest\ntraining rows", fontsize=8, color=MUTED, va="bottom")
        ax.set_xticks(xs)
        ax.set_xticklabels(["64k", "128k", "256k"])
        ax.set_xlim(-0.35, 2.3)
        ax.set_ylim(0.25, 0.72)
        ax.set_xlabel("eval context length")
        ax.set_title("%s\n%s PF of training" % (name, format(round(pf), ",")))
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right", fontsize=8.5)
        print(name, pf, ["%.3f" % v for v in arm], ["%.3f" % v for v in den], ["%.3f" % v for v in se])
    axes[0].set_ylabel("6-task mean score")

    # Pareto view: score vs training FLOPs at 128k and 256k
    ax = axes[3]
    dense_runs = ["m64-dense-f16-lr5e-06", "m64-dense-f8-lr5e-06", "m64-dense-f4-lr5e-06", "m64-dense-f2-lr5e-06", "m64-dense-f1-lr5e-06"]
    for g, ls, lab in (("r128k", "-", "dense, eval 128k"), ("r256k", ":", "dense, eval 256k")):
        px = [d["pf"][r] for r in dense_runs]
        py = [mean_se(r, g)[0] for r in dense_runs]
        ax.plot(px, py, color=INK, lw=2, ls=ls, marker="s", ms=5, label=lab)
        print(lab, list(zip([round(p) for p in px], ["%.3f" % v for v in py])))
        for run, name, color in arms:
            pf = next(iter(matched[(run, g)].values()))[2]
            ax.scatter([pf], [mean_se(run, g)[0]], s=70, color=color, zorder=3, marker="o" if g == "r128k" else "^", edgecolor="white", linewidth=1.2)
    ax.scatter([], [], s=60, color=MUTED, marker="o", label="token dropping, eval 128k")
    ax.scatter([], [], s=60, color=MUTED, marker="^", label="token dropping, eval 256k")
    ax.set_xscale("log")
    ax.set_xticks([1000, 2000, 5000, 10000, 20000])
    ax.set_xticklabels(["1k", "2k", "5k", "10k", "20k"])
    ax.minorticks_off()
    ax.set_xlabel("training FLOPs (PF, log scale); point colors as in the left panels")
    ax.set_ylabel("6-task mean score")
    ax.set_title("Past the training length the points\nstay on or above the dense curve")
    ax.legend(loc="upper left", fontsize=8.5)

    fig.suptitle(
        "Length generalization, 11-task mix trained on rows ≤ 64k (Qwen3.5-4B, single seed, no-cot; eval_size 500 per cell at 64k/128k, 125 at 256k; bars = ±1 SE)",
        fontsize=10.5,
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, "mt11_64k_length_generalization.png")
    fig.savefig(out, dpi=170)
    print("wrote", out)


if __name__ == "__main__":
    cpt_figure()
    lengen_figure()
