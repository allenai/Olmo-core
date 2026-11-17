"""6-task mean vs eval length (2k..256k), one panel per FLOP budget, three lines: trained on rows <= 16k / <= 32k / <= 64k.

Six tasks = the ones with 128k/256k suite rows (nq, hotpotqa, qdmatch_nq, outlier, oolong, contradiction).
2k..32k: native docchunk ladder (eval_size 500 per cell). 64k/128k/256k: olmo-eval path (r64k/r128k eval_size 500,
r256k eval_size 125 per cell, YaRN x2 + chunked prefill). The 64k-data line is the real 64k-data checkpoint nearest the
budget (its PF in the legend); the delta tables interpolate in log PF instead.

    python3 research/token_routing/mt11_64k/lengen_chart.py   # -> lengen_rows16_32_64.png (+ .json) beside this file
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_lengen")
POINTS = ("/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.claude/worktrees/mt11-64k/research/token_routing/"
          "mt11_64k/results/points.json")
TASKS = ["nq", "hotpotqa", "qdmatch_nq", "outlier", "oolong", "contradiction"]
X = [("2k", "2k"), ("8k", "8k"), ("16k", "16k"), ("32k", "32k"), ("r64k", "64k"), ("r128k", "128k"), ("r256k", "256k")]
FULL = {"r64k": 500, "r128k": 500, "r256k": 125}
#: budget -> (<=16k run, <=32k run, <=64k run)
PANELS = json.load(open(os.path.join(HERE, "lengen_chart_panels.json")))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pf = json.load(open(os.path.join(RES, "lengen.json")))["pf"]
    pf.update(json.load(open(os.path.join(HERE, "extra_pf.json"))))
    sc = {}
    for s in json.load(open(POINTS))["scores"]:
        if s.get("eval_size") == 500:
            sc[(s["train_run"], s["task"], s["rung"])] = s["value"]
    sh = json.load(open(os.path.join(RES, "short16.json")))
    for c in sh["native"]:
        sc[(c["run"], c["task"], c["rung"])] = c["score"]
    for r in json.load(open(os.path.join(RES, "lengen.json")))["rows"]:
        if r["eval_size"] == FULL.get(r["rung"]):
            sc[(r["run"], r["task"], r["rung"])] = r["value"]

    def curve(run):
        ys = []
        for rung, _ in X:
            v = [sc.get((run, t, rung)) for t in TASKS]
            ys.append(sum(v) / len(v) if all(x is not None for x in v) else None)
        return ys

    out = {}
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.6 * len(PANELS), 3.9), sharey=True)
    colors = {"16k": "#0e7490", "32k": "#c2410c", "64k": "#334155"}
    for ax, (title, runs) in zip(axes, PANELS.items()):
        out[title] = {}
        for win, run in zip(("16k", "32k", "64k"), runs):
            ys = curve(run)
            out[title][run] = ys
            pts = [(i, y) for i, y in enumerate(ys) if y is not None]
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=colors[win], lw=2, ms=4,
                    label=f"rows ≤{win} ({pf.get(run, float('nan')):,.0f} PF)")
        ax.axvspan(4.5, 6.5, color="#94a3b8", alpha=0.12, lw=0)
        ax.set_xticks(range(len(X)))
        ax.set_xticklabels([x[1] for x in X])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("eval length (256k: eval_size 125)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower left", frameon=False)
    axes[0].set_ylabel("6-task mean score")
    fig.suptitle("mt11: training on rows ≤16k / ≤32k / ≤64k at matched FLOPs (single seed, no-cot)", fontsize=11)
    fig.tight_layout()
    png = os.path.join(HERE, "lengen_rows16_32_64.png")
    fig.savefig(png, dpi=150)
    json.dump(dict(x=[x[1] for x in X], panels=out), open(os.path.join(RES, "chart_rows16_32_64.json"), "w"), indent=1)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
