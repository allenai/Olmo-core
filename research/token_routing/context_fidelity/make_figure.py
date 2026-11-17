"""Fidelity vs step cost from analyze_context.py output: one row per checkpoint; left = expected gradient cos (Adam),
right = answer CE minus dense CE. x = analytic step cost (fraction of a dense step). Color = selector.

    python research/token_routing/context_fidelity/make_figure.py results/ctx.json figs/context_fidelity.png
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = {"rtr": "#2a78d6", "rnd": "#eb6834", "od": "#1baf7a"}
NAME = {"rtr": "router (ridge emulation)", "rnd": "random", "od": "attribution top-k"}
SHORT = {"drop": "drop", "frozen": "frozen", "refresh15": "refr15", "kvfresh": "kvfresh*", "mean1": "mean1",
         "span64": "span64", "lin7": "lin7", "lin7r": "lin7r", "frozen_all": "stale_all", "fwdonly": "fwdonly*",
         "stale20": "stale20", "stale50": "stale50"}

res = json.load(open(sys.argv[1]))
labs = list(res)
fig, axes = plt.subplots(len(labs), 2, figsize=(12, 4.6 * len(labs)), squeeze=False)
for i, lab in enumerate(labs):
    o = res[lab]
    for j, (metric, title) in enumerate((("exp_cos_adam", "expected gradient cos with dense (Adam)"),
                                         ("ce_excess", "answer CE minus dense CE (lower is better)"))):
        ax = axes[i][j]
        for c, fl in o["flops"].items():
            s, v = c.split("_", 1)
            x = fl["frac"]
            if metric == "exp_cos_adam":
                y = o["adam"][c]["exp_cos"][0]
                e = o["adam"][c]["exp_cos"][1]
            else:
                y = o["ce"][c]["pooled"]["routed"][0] - o["ce"][c]["pooled"]["dense"][0]
                e = o["ce"][c]["pooled"]["routed"][1]
            ax.errorbar(x, y, yerr=e, fmt="o", ms=7, color=COL[s], mec="white", mew=1.5, elinewidth=1, capsize=0)
            if s == "rtr" and v in ("drop", "frozen", "stale20", "stale50", "frozen_all", "kvfresh", "fwdonly", "mean1",
                                    "lin7r"):
                ax.annotate(SHORT.get(v, v), (x, y), textcoords="offset points", xytext=(6, -3), fontsize=7.5, color="#52514e")
        ax.set_xlim(0.355, 0.575)
        if metric == "ce_excess":
            ax.axhline(0, color="#bbb", lw=1)
        ax.set_xlabel("analytic step cost (fraction of a dense step = 1.0)")
        ax.set_title(f"{lab} ({o['rows']} rows): {title}", fontsize=10)
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
handles = [plt.Line2D([], [], marker="o", ls="", color=COL[s], label=NAME[s]) for s in COL]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
fig.text(0.5, 0.005, "labels: router points only; stale_all = stale context in every layer.  * diagnostic, not a method: kvfresh = dense K/V for dropped tokens at attention layers; "
         "fwdonly = exact dense forward, no gradient through dropped tokens",
         ha="center", fontsize=7, color="#52514e")
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
fig.savefig(sys.argv[2], dpi=140)
print("wrote", sys.argv[2])
