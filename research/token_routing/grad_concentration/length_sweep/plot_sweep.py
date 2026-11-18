"""Plot analyze_sweep.py output: x = mean row length (log), y = share of body tokens needed for cos 0.95 (log), one
line per selection rule; second panel = absolute body tokens (per row: step tokens / 16, and the row-level
self-cosine). Optional dashed overlay of a second checkpoint series.

    python plot_sweep.py --summary sweep_mt11.json [--overlay sweep_f7808.json] --out grad_sparsity_vs_length.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RULES = [("down7_body", "global oracle (16-row step)", "#2a78d6", "o"),
         ("rowq7_body", "per-row quota", "#eb6834", "s"),
         ("docq7_body", "per-document quota", "#1baf7a", "^"),
         ("rand7_body", "random", "#eda100", "D")]


def series(S, r, t, what):
    xs, ys, lo, hi = [], [], [], []
    for s in S["runs"]:
        e = s["rules"][r]
        if what == "frac":
            v, ci = e[f"k_cos{t}"], e[f"k_cos{t}_ci"]
        else:
            v, ci = e[f"tokens_row_cos{t}"], [c / 16 for c in e[f"tokens_step_cos{t}_ci"]]
        if not np.isfinite(v):
            continue
        xs.append(s["T_mean"])
        ys.append(v * (100 if what == "frac" else 1))
        lo.append(ci[0] * (100 if what == "frac" else 1))
        hi.append(ci[1] * (100 if what == "frac" else 1))
    return np.array(xs), np.array(ys), np.array(lo), np.array(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--overlay", default="")
    ap.add_argument("--label", default="mt11 dense f1 (rows <= 64k)")
    ap.add_argument("--overlay-label", default="mt4 dense 7,808 rows/task (8k)")
    ap.add_argument("--overlay2", default="", help="third series, dotted with x markers (e.g. the untrained base)")
    ap.add_argument("--overlay2-label", default="base Qwen3.5-4B (no fine-tuning)")
    ap.add_argument("--t", type=float, default=0.95)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = json.load(open(a.summary))
    O = json.load(open(a.overlay)) if a.overlay else None
    O2 = json.load(open(a.overlay2)) if a.overlay2 else None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, what in zip(axes, ("frac", "tok")):
        for r, name, col, mk in RULES:
            x, y, lo, hi = series(S, r, a.t, what)
            ax.plot(x, y, color=col, marker=mk, lw=2, ms=7, label=name)
            ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
            if O:
                x2, y2, _, _ = series(O, r, a.t, what)
                ax.plot(x2, y2, color=col, marker=mk, lw=1.5, ms=6, ls="--", mfc="white")
            if O2:
                x3, y3, _, _ = series(O2, r, a.t, what)
                ax.plot(x3, y3, color=col, marker="x", lw=1.5, ms=7, ls=(0, (1, 1.5)))
        if what == "frac":  # guide: what a constant kept-token count (the per-row quota's 2k count) would look like
            x, y, _, _ = series(S, "rowq7_body", a.t, "frac")
            ax.plot(x, y[0] * x[0] / x, color="#999999", ls=":", lw=1.5, label="constant token count (slope -1)")
        if what == "tok":
            xr = [s["T_mean"] for s in S["runs"] if f"rowself_tokens_cos{a.t}" in s]
            yr = [s[f"rowself_tokens_cos{a.t}"] for s in S["runs"] if f"rowself_tokens_cos{a.t}" in s]
            if xr:
                ax.plot(xr, yr, color="#e87ba4", marker="v", lw=2, ms=7, label="row vs its own gradient")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, which="major", color="#dddddd", lw=0.6)
        ax.set_xlabel("row length (tokens, mean per length bin)")
        ax.set_ylabel(f"% of body tokens kept for cos {a.t}" if what == "frac"
                      else f"body tokens kept per row for cos {a.t}")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_title("Share of context tokens needed")
    axes[1].set_title("Absolute tokens needed (step total / 16)")
    h, l = axes[1].get_legend_handles_labels()
    h0, l0 = axes[0].get_legend_handles_labels()
    h, l = h + h0[-1:], l + l0[-1:]
    if O:
        h.append(plt.Line2D([], [], color="#666666", ls="--", mfc="white", marker="o"))
        l.append(f"dashed: {a.overlay_label}")
    if O2:
        h.append(plt.Line2D([], [], color="#666666", ls=(0, (1, 1.5)), marker="x"))
        l.append(f"dotted x: {a.overlay2_label}")
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fit = S["fits"].get(f"rowq7_body|cos{a.t}")
    sub = f"solid: {a.label}; bands = 95% bootstrap over steps"
    if fit:
        sub += f"; per-row quota slope {fit['slope']:.2f} [{fit['ci95'][0]:.2f}, {fit['ci95'][1]:.2f}]"
    fig.suptitle(f"Gradient sparsity vs context length (blocks 7-31 weight gradient, cos {a.t})\n{sub}", fontsize=10)
    fig.tight_layout(rect=(0, 0.14, 1, 0.93))
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
