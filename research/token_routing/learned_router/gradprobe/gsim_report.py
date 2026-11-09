"""Tables + plots for grad_similarity.py outputs.

    python3 research/token_routing/learned_router/gradprobe/gsim_report.py results/json/gsim/gsim_*_r32.json --out-dir results

Prints a markdown table per (model, task): median global cosine(g_full, g_drop) [IQR], median log10 norm ratio and
mean dCE per (rule, rho), plus the rerun / keepall / cross-row references. Writes gsim_cos_vs_rho.png,
gsim_norm_vs_rho.png and gsim_cos_by_depth.png (per-module cosine vs layer for one rho).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np

RULES = ["goldhfl", "hfl", "rand", "a0.5_abs", "gold_ga", "bar"]
COLORS = {"rand": "#2a78d6", "a0.5_abs": "#eb6834", "gold_ga": "#1baf7a", "bar": "#eda100", "goldhfl": "#1baf7a", "hfl": "#e87ba4"}
LABEL = {"rand": "random", "a0.5_abs": "|grad|@g=0.5", "gold_ga": "gold first + |grad|@0.5", "bar": "bar (gold_fl20p8)",
         "goldhfl": "bar family: gold + hdr8 + first/last x%", "hfl": "gold-blind: hdr8 + first/last x%"}


def pkey(q):
    """Group key of a point: 'own' for the bar, ('x', x) for the bar family and its matched-budget references (each row has
    its own keep ratio there), else the fixed keep ratio."""
    if q["rule"] == "bar":
        return "own"
    if q.get("x") is not None:
        return ("x", q["x"])
    return round(q["rho"], 4)


def model_of(path, argv):
    ck = os.path.basename(argv[argv.index("--ckpt") + 1].rstrip("/"))
    if ck.startswith("ctc-4b-") and ck.endswith("-full"):
        return "per-task SFT (Qwen3.5)"  # one row for the per-task checkpoints, each scored on its own task
    return {"setA32k": "setA32k (Qwen3.5 SFT)", "q3setA": "Qwen3 setA SFT"}.get(ck, ck)


def load(paths):
    out = {}
    for p in paths:
        d = json.load(open(p))
        out[(model_of(p, d["argv"]), d["task"])] = d
    return out


def points(d, floor=None):
    """(rule, rho) -> list of point dicts, optionally only rows with ce_full >= floor."""
    pts = defaultdict(list)
    for r in d["rows"]:
        if floor is not None and r["ce_full"] < floor:
            continue
        for q in r["points"]:
            pts[(q["rule"], pkey(q))].append(q)
    return pts


def table(key, d, floor=None):
    rows = [r for r in d["rows"] if floor is None or r["ce_full"] >= floor]
    if not rows:
        return f"### {key[0]} / {key[1]}: no rows with CE_full >= {floor}\n"
    ref = {k: np.median([r[k]["cos"] for r in rows if k in r]) for k in ("rerun", "keepall", "xrow", "xrow_centered", "xrow_proj") if any(k in r for r in rows)}
    ce = np.median([r["ce_full"] for r in rows])
    s = [f"### {key[0]} / {key[1]}  (⚠ eval_size={len(rows)}{'' if floor is None else f', rows with CE_full >= {floor}'};"
         f" median CE_full {ce:.3g})",
         "refs: " + ", ".join(f"{k} cos {v:.3f}" for k, v in ref.items()), "",
         "| rule | ρ | cos median [IQR] | centered cos median | proj cos median [IQR] | log10 ‖g_drop‖/‖g_full‖ | ΔCE mean | gate cos | gate Spearman | gate top-10% overlap |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    pts = points(d, floor)
    for rule in RULES:
        for (r_, rho), v in sorted(((k, v) for k, v in pts.items() if k[0] == rule),
                                   key=lambda kv: -1 if kv[0][1] == "own" else (-kv[0][1][1] if isinstance(kv[0][1], tuple) else -kv[0][1])):
            c = np.array([q["cos"] for q in v])
            lr = np.log10([q["norm_ratio"] for q in v])
            g = [q["gate"] for q in v if "gate" in q]
            rho_s = (f"own ({np.mean([q['rho'] for q in v]):.3f})" if rho == "own" else
                     f"x={rho[1]:g} ({np.mean([q['rho'] for q in v]):.3f})" if isinstance(rho, tuple) else f"{rho:g}")
            gs = (f"{np.median([x['cos_signed'] for x in g]):+.2f} | {np.nanmedian([x['spearman_abs'] for x in g]):+.2f} | "
                  f"{np.median([x['top10pct_overlap'] for x in g]):.2f}") if g else "– | – | –"
            cc = [q["centered"]["cos"] for q in v if "centered" in q]
            ccs = f"{np.median(cc):+.3f}" if cc else "–"
            pc = [q["proj"]["cos"] for q in v if "proj" in q]
            pcs = f"{np.median(pc):+.3f} [{np.quantile(pc, .25):+.2f}, {np.quantile(pc, .75):+.2f}]" if pc else "–"
            s.append(f"| {LABEL[rule].replace('|', chr(92) + '|')} | {rho_s} | {np.median(c):+.3f} [{np.quantile(c, .25):+.2f}, {np.quantile(c, .75):+.2f}] | {ccs} | {pcs} | "
                     f"{np.median(lr):+.2f} | {np.mean([q['dce'] for q in v]):+.3f} | {gs} |")
    return "\n".join(s) + "\n"


def plots(data, out_dir, floor=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if floor is not None and not any(r["ce_full"] >= floor for d in data.values() for r in d["rows"]):
        return
    models = sorted({k[0] for k in data})
    tasks = [t for t in ("nq", "outlier", "contradiction") if any(k[1] == t for k in data)]
    for metric, ylab, fname in (("cos", "median cosine(g_full, g_drop)", "gsim_cos_vs_rho.png"),
                                ("ccos", "median centered cosine\n(leave-one-out mean full grad removed)", "gsim_ccos_vs_rho.png"),
                                ("pcos", "median projected cosine\n(leave-one-out mean direction projected out)", "gsim_pcos_vs_rho.png"),
                                ("lognorm", "median log10 ‖g_drop‖ / ‖g_full‖", "gsim_norm_vs_rho.png")):
        fig, axes = plt.subplots(len(models), len(tasks), figsize=(4.2 * len(tasks), 3.3 * len(models)), squeeze=False,
                                 sharex=True)
        for i, m in enumerate(models):
            for j, t in enumerate(tasks):
                ax = axes[i][j]
                d = data.get((m, t))
                if d is None:
                    ax.set_visible(False)
                    continue
                pts = points(d, floor)
                for rule in RULES:
                    ks = sorted([k for k in pts if k[0] == rule], key=lambda k: np.mean([q["rho"] for q in pts[k]]))
                    if not ks:
                        continue
                    xs = [np.mean([q["rho"] for q in pts[k]]) for k in ks]
                    if metric == "cos":
                        ys = [np.median([q["cos"] for q in pts[k]]) for k in ks]
                    elif metric == "pcos":
                        if not all("proj" in q for k in ks for q in pts[k]):
                            continue
                        ys = [np.median([q["proj"]["cos"] for q in pts[k]]) for k in ks]
                    elif metric == "ccos":
                        if not all("centered" in q for k in ks for q in pts[k]):
                            continue
                        ys = [np.median([q["centered"]["cos"] for q in pts[k]]) for k in ks]
                    else:
                        ys = [np.median(np.log10([q["norm_ratio"] for q in pts[k]])) for k in ks]
                    if rule == "bar":
                        ax.plot(xs, ys, "D", ms=8, color=COLORS[rule], label=LABEL[rule], mec="white", mew=1.5)
                    else:
                        ax.plot(xs, ys, "-o", lw=2, ms=6, color=COLORS[rule], label=LABEL[rule], mec="white", mew=1)
                if metric in ("cos", "ccos", "pcos"):
                    rk = {"cos": "xrow", "ccos": "xrow_centered", "pcos": "xrow_proj"}[metric]
                    ref = [r[rk]["cos"] for r in d["rows"] if rk in r]
                    if ref:
                        ax.axhline(np.median(ref), color="#8a8984", lw=1, ls="--")
                        ax.text(0.5, np.median(ref), "other row's g_full ", va="bottom", ha="right", fontsize=7, color="#52514e")
                    ax.set_ylim(min(-0.1, ax.get_ylim()[0]), 1.02)
                else:
                    ax.axhline(0, color="#8a8984", lw=1, ls="--")
                ax.set_xscale("log")
                ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5], ["2%", "5%", "10%", "20%", "50%"])
                ax.minorticks_off()
                ax.grid(alpha=0.25, lw=0.6)
                ax.spines[["top", "right"]].set_visible(False)
                n = len(d["rows"]) if floor is None else sum(r["ce_full"] >= floor for r in d["rows"])
                ax.set_title(f"{m} · {t} (eval_size={n})", fontsize=9)
                if j == 0:
                    ax.set_ylabel(ylab, fontsize=8)
                if i == len(models) - 1:
                    ax.set_xlabel("keep ratio ρ (body tokens)", fontsize=8)
        h, l = [], []
        for ax_ in [a_ for row_ in axes for a_ in row_]:
            for hh, ll in zip(*ax_.get_legend_handles_labels()):
                if ll not in l:
                    h.append(hh); l.append(ll)
        if not l:  # this metric has no data in these files (e.g. ccos for --demean 2 runs)
            plt.close(fig)
            continue
        nc = len(l) if len(tasks) >= 2 else 2
        fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=nc)
        fig.tight_layout(rect=(0, (0.06 if nc == len(l) else 0.12) / len(models), 1, 1))
        fig.savefig(os.path.join(out_dir, fname.replace(".png", f"_ce{floor}.png" if floor is not None else ".png")), dpi=130)
        plt.close(fig)

    if floor is not None:
        return
    # per-module cosine by depth, one rho per rule, aggregated over tasks (median over rows of per-group cosine)
    rho_pick = 0.1
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 3.4), squeeze=False)
    for i, m in enumerate(models):
        ax = axes[0][i]
        for rule in RULES:
            per_layer = defaultdict(list)
            for (mm, t), d in data.items():
                if mm != m:
                    continue
                groups = d["groups"]
                for r in d["rows"]:
                    nf = np.array(r["full_group_norm2"])
                    for q in r["points"]:
                        sel = (abs(q["x"] - 20) < 1e-9) if q.get("x") is not None else abs(q["rho"] - rho_pick) < 1e-6
                        if q["rule"] != rule or (rule != "bar" and not sel):
                            continue
                        dot, nd = np.array(q["group_dot"]), np.array(q["group_norm2_drop"])
                        layer_dot, layer_a, layer_b = defaultdict(float), defaultdict(float), defaultdict(float)
                        for gname, a_, b_, c_ in zip(groups, nf, nd, dot):
                            mt = re.match(r"blocks\.(\d+)\.", gname)
                            if mt:
                                L = int(mt.group(1))
                                layer_dot[L] += c_
                                layer_a[L] += a_
                                layer_b[L] += b_
                        for L in layer_dot:
                            per_layer[L].append(layer_dot[L] / np.sqrt(max(layer_a[L] * layer_b[L], 1e-300)))
            if per_layer:
                Ls = sorted(per_layer)
                ax.plot(Ls, [np.median(per_layer[L]) for L in Ls], "-o", ms=4, lw=2, color=COLORS[rule],
                        label=LABEL[rule] + ("" if rule in ("bar", "goldhfl", "hfl") else f" ρ={rho_pick} or x=20 budget"), mec="white", mew=0.8)
        mix = next(d["mixers"] for (mm, _), d in data.items() if mm == m)
        for L, ch in mix.items():
            if "GatedDeltaNet" not in ch.get("attention", ""):
                ax.axvspan(int(L) - 0.5, int(L) + 0.5, color="#e5e4df", lw=0, zorder=0)
        ax.set_title(f"{m}: per-layer cosine, median over rows + tasks\n(shaded = full-attention layers, rest GDN)", fontsize=9)
        ax.set_xlabel("layer", fontsize=8)
        ax.set_ylabel("cosine(g_full, g_drop) within layer", fontsize=8)
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    h, l = [], []
    for ax_ in axes[0]:
        for hh, ll in zip(*ax_.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    nc = min(len(l), 3) if l else 1
    fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=nc)
    fig.tight_layout(rect=(0, 0.08 if nc == len(l) else 0.14, 1, 1))
    fig.savefig(os.path.join(out_dir, "gsim_cos_by_depth.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--floor", type=float, default=0.05)
    a = ap.parse_args()
    data = load(a.paths)
    for k in sorted(data):
        print(table(k, data[k]))
        print(table(k, data[k], a.floor))
    os.makedirs(a.out_dir, exist_ok=True)
    plots(data, a.out_dir)
    plots(data, a.out_dir, a.floor)


if __name__ == "__main__":
    main()
