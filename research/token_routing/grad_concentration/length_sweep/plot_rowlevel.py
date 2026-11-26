"""Row-level comparison across checkpoints: each row's own per-row-quota update vs that row's own full update.
Left: share of rows needing no body tokens (their gradient sits in prompt / query / answer tokens). Right: geometric
mean share of body tokens needed for cos 0.95 over the other rows, with a bootstrap band over rows.

    python plot_rowlevel.py --series "mt11 dense f1=data/sweep_mt11_dense_f1.json" ... --out grad_sparsity_rowlevel.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", action="append", required=True, help="label=summary.json")
    ap.add_argument("--t", type=float, default=0.95)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for i, item in enumerate(a.series):
        lab, path = item.rsplit("=", 1)
        S = json.load(open(path))
        x, z, g, lo, hi = [], [], [], [], []
        for s in S["runs"]:
            k = np.array(s[f"_rows_rowself_cos{a.t}"], float)
            k = k[np.isfinite(k)]
            pos = np.log(k[k > 0])
            x.append(s["T_mean"])
            z.append(100 * np.mean(k == 0))
            g.append(100 * np.exp(pos.mean()))
            bs = [np.exp(pos[rng.integers(0, len(pos), len(pos))].mean()) for _ in range(1000)]
            lo.append(100 * np.percentile(bs, 2.5))
            hi.append(100 * np.percentile(bs, 97.5))
        axes[0].plot(x, z, color=COLORS[i], marker=MARKERS[i], lw=2, ms=7, label=lab)
        axes[1].plot(x, g, color=COLORS[i], marker=MARKERS[i], lw=2, ms=7, label=lab)
        axes[1].fill_between(x, lo, hi, color=COLORS[i], alpha=0.15, lw=0)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("row length (tokens, mean per length bin)")
        ax.grid(True, color="#dddddd", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[1].set_yscale("log")
    axes[0].set_ylabel("% of rows")
    axes[1].set_ylabel(f"% of body tokens for cos {a.t} (geometric mean)")
    axes[0].set_title("Rows whose own gradient needs no body tokens")
    axes[1].set_title("Body share needed, rows that need any (row vs its own gradient)")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(a.series), frameon=False, fontsize=9)
    fig.suptitle("Row-level gradient sparsity vs context length, by checkpoint (blocks 7-31; bands = 95% bootstrap over rows)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
