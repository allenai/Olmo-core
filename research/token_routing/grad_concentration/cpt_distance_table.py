"""Shares of the CPT gradient by dependency distance, from cpt_distance_decomp.py outputs (small; any CPU).

    python research/token_routing/grad_concentration/cpt_distance_table.py --in-dir <cpt_distance> \
        --labels dist_base_32k,dist_cpt_32k --out-dir research/token_routing/grad_concentration/gradmaps_cpt

For each label the rows form one step: G = sum over rows and gaps of the band x gap sketches. A distance class X (a set of
gaps; gap 0 split into own label and same-chunk context) gets
  share(X) = <X, G> / ||G||^2     (projection share; the classes sum to exactly 1, a class can be negative)
  norm(X)  = ||X|| / ||G||
per block band and for the whole model (inner products summed over bands). Per-row shares (each row's own G) give the
mean and its SE over rows.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

BAND_NAMES = ("blocks 0-6", "blocks 7-15", "blocks 16-23", "blocks 24-31")


def classes(W, nch):
    """(name, list of gaps or 'own' / 'ctx0')."""
    def gaps(lo_tok, hi_tok):
        return [g for g in range(1, nch) if lo_tok <= g * W < hi_tok]
    return [("own label (distance 0)", "own"), (f"same {W:,}-token chunk, other tokens", "ctx0"),
            ("~1k-4k back", gaps(1, 4096)), ("~4k-16k back", gaps(4096, 16384)), ("> ~16k back", gaps(16384, 1 << 30))]


def vecs(bins, own, spec):
    if spec == "own":
        return own
    if spec == "ctx0":
        return bins[:, 0] - own
    return bins[:, spec].sum(1) if len(spec) else np.zeros_like(own)


def shares(bins, own, cls):
    """bins (nb, nch, m), own (nb, m) -> dict name -> (per band share list, whole share, whole norm ratio)."""
    G = bins.sum(1)
    gg = (G * G).sum(-1)  # (nb,)
    out = {}
    for name, spec in cls:
        X = vecs(bins, own, spec)
        ip = (X * G).sum(-1)
        out[name] = dict(band=(ip / np.maximum(gg, 1e-300)).tolist(), whole=float(ip.sum() / gg.sum()),
                         norm=float(np.sqrt((X * X).sum() / gg.sum())))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    res, lines = {}, []
    for lab in a.labels.split(","):
        meta = json.load(open(sorted(glob.glob(os.path.join(a.in_dir, lab, "meta_*.json")))[0]))
        W = meta["chunk"]
        files = sorted(glob.glob(os.path.join(a.in_dir, lab, "r*.npz")))
        rows = [dict(np.load(f)) for f in files]
        nch = rows[0]["bins"].shape[1]
        cls = classes(W, nch)
        B = sum(r["bins"].astype(np.float64) for r in rows)
        O = sum(r["own"].astype(np.float64) for r in rows)
        step = shares(B, O, cls)
        per = [shares(r["bins"].astype(np.float64), r["own"].astype(np.float64), cls) for r in rows]
        chk = [float((r["bins"].astype(np.float64).sum(1) ** 2).sum() / r["exact_band"].sum()) for r in rows]
        e = {"rows": len(rows), "T": meta["T"], "chunk": W, "step": step, "sketch_check": chk,
             "per_row": {n: {"mean": float(np.mean([p[n]["whole"] for p in per])),
                             "se": float(np.std([p[n]["whole"] for p in per], ddof=1) / np.sqrt(len(per)))}
                         for n, _ in cls}}
        # cumulative projection share by gap (whole model), own label at gap 0
        G = B.sum(1)
        ipg = np.array([(B[:, g] * G).sum() for g in range(nch)]) / (G * G).sum()
        e["cum_by_gap"] = np.cumsum(ipg).tolist()
        # per-gap projection share, gap 0 split into own label / other same-chunk tokens; per band and whole model
        own_ip = (O * G).sum(-1)
        gg = (G * G).sum(-1)
        per_band = np.stack([(B[:, g] * G).sum(-1) for g in range(B.shape[1])], 1)  # (nb, nch)
        e["gap_share_band"] = (per_band / np.maximum(gg, 1e-300)[:, None]).tolist()
        e["own_share_band"] = (own_ip / np.maximum(gg, 1e-300)).tolist()
        e["gap_share_whole"] = (per_band.sum(0) / gg.sum()).tolist()
        e["own_share_whole"] = float(own_ip.sum() / gg.sum())
        # per-row whole-model per-gap shares (for an SE band)
        prs = []
        for r in rows:
            b_ = r["bins"].astype(np.float64)
            o_ = r["own"].astype(np.float64)
            G_ = b_.sum(1)
            g2 = (G_ * G_).sum()
            v = np.array([(b_[:, g] * G_).sum() for g in range(b_.shape[1])]) / g2
            prs.append(np.concatenate([[float((o_ * G_).sum() / g2)], v]))
        prs = np.array(prs)  # (rows, 1 + nch): own, then gaps (gap 0 includes own)
        e["row_gap_share_mean"] = prs.mean(0).tolist()
        e["row_gap_share_se"] = (prs.std(0, ddof=1) / np.sqrt(len(prs))).tolist()
        res[lab] = e
    json.dump(res, open(os.path.join(a.out_dir, "distance_decomp.json"), "w"), indent=1)
    labs = list(res)
    cls_names = [n for n, _ in classes(res[labs[0]]["chunk"], len(res[labs[0]]["cum_by_gap"]))]
    lines.append(f"Projection share of the full-loss gradient by dependency distance (one step = "
                 f"{res[labs[0]]['rows']} rows x {res[labs[0]]['T'] // 1024}k; shares sum to 1; in brackets the per-row "
                 f"mean ± SE)\n")
    lines.append("| distance class | " + " | ".join(f"{lab} whole model" for lab in labs) + " | "
                 + " | ".join(f"{lab} {b}" for lab in labs for b in BAND_NAMES) + " |")
    lines.append("|" + "---|" * (1 + len(labs) * 5))
    for n in cls_names:
        cells = [n]
        for lab in labs:
            s = res[lab]["step"][n]
            pr = res[lab]["per_row"][n]
            cells.append(f"{s['whole']:.3f} ({pr['mean']:.3f} ± {pr['se']:.3f})")
        for lab in labs:
            cells += [f"{v:.3f}" for v in res[lab]["step"][n]["band"]]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("\nNorm of each class relative to ||G|| (whole model): " + "; ".join(
        f"{lab}: " + ", ".join(f"{n.split(' (')[0]} {res[lab]['step'][n]['norm']:.2f}" for n in cls_names) for lab in labs))
    txt = "\n".join(lines)
    open(os.path.join(a.out_dir, "distance_decomp.md"), "w").write(txt + "\n")
    print(txt)
    for lab in labs:
        c = res[lab]["sketch_check"]
        print(lab, "sketch/exact norm^2 of the summed chunk gradients", f"{min(c):.3f}-{max(c):.3f}")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    col = {"base": "#4c72b0", "cpt": "#dd8452"}
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.0))
    x = np.arange(len(BAND_NAMES) + 1)
    w = 0.8 / len(labs)
    ccol = ["#4c4c4c", "#8da0cb", "#66c2a5", "#fc8d62", "#e78ac3"]
    for j, lab in enumerate(labs):
        bottom_p, bottom_n = np.zeros(len(x)), np.zeros(len(x))
        for ci, n in enumerate(cls_names):
            s = res[lab]["step"][n]
            v = np.array(s["band"] + [s["whole"]])
            pos, neg = np.clip(v, 0, None), np.clip(v, None, 0)
            axs[0].bar(x + (j - (len(labs) - 1) / 2) * w, pos, w * 0.95, bottom=bottom_p, color=ccol[ci],
                       edgecolor="white", lw=0.5, label=n if j == 0 else None)
            axs[0].bar(x + (j - (len(labs) - 1) / 2) * w, neg, w * 0.95, bottom=bottom_n, color=ccol[ci],
                       edgecolor="white", lw=0.5, hatch="//")
            bottom_p += pos
            bottom_n += neg
        for xi in x:
            axs[0].text(xi + (j - (len(labs) - 1) / 2) * w, 1.02, lab.split("_")[1], ha="center", fontsize=6.5)
    axs[0].axhline(0, color="k", lw=0.6)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(list(BAND_NAMES) + ["whole model"], fontsize=8)
    axs[0].set_ylabel("projection share of the step gradient")
    axs[0].legend(fontsize=7, loc="lower left")
    axs[0].set_title("Gradient by dependency distance, per block band (hatched = negative share)", fontsize=9)
    for lab in labs:
        W = res[lab]["chunk"]
        cg = np.array(res[lab]["cum_by_gap"])
        d = (np.arange(len(cg)) + 0.5) * W
        axs[1].plot(d, cg, marker="o", ms=3, color=col.get(lab.split("_")[1], "k"), label=lab)
    for D in (4096, 16384):
        axs[1].axvline(D, color="grey", lw=0.6)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("target-to-token distance (chunk gap x chunk size, midpoint)")
    axs[1].set_ylabel("cumulative projection share (whole model)")
    axs[1].legend(fontsize=7)
    axs[1].set_title("Cumulative share of the gradient from dependencies up to distance d", fontsize=9)
    fig.suptitle(f"Short- vs long-range credit in the CPT gradient (exact additive decomposition, sketched; "
                 f"{res[labs[0]]['rows']} rows x {res[labs[0]]['T'] // 1024}k = one step)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "sum_distance_decomp.png"), dpi=110)
    plt.close(fig)
    # percentage of the gradient as a function of distance
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.4))
    bcol = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    for lab in labs:
        e = res[lab]
        W = e["chunk"]
        c = col.get(lab.split("_")[1], "k")
        nch = len(e["gap_share_whole"])
        # x: own label at a nominal distance, then gap g at its midpoint distance; gap 0 minus own = same chunk
        xs = [0.5, W / 2] + [g * W for g in range(1, nch)]
        ws = [e["own_share_whole"], e["gap_share_whole"][0] - e["own_share_whole"]] + e["gap_share_whole"][1:]
        axs[0].plot(xs[1:], 100 * np.array(ws[1:]), marker="o", ms=3, color=c, label=f"{lab}")
        axs[0].scatter([xs[0]], [100 * ws[0]], color=c, marker="*", s=80, zorder=3)
        m = np.array(e["row_gap_share_mean"])
        se = np.array(e["row_gap_share_se"])
        npair = np.array([nch - g for g in range(1, nch)], dtype=float)  # (target chunk, token chunk) pairs at gap g
        axs[1].plot(xs[2:], 100 * m[2:] / npair, color=c, marker="o", ms=3, label=lab)
        axs[1].fill_between(xs[2:], 100 * (m[2:] - se[2:]) / npair, 100 * (m[2:] + se[2:]) / npair, color=c, alpha=0.2)
        for bi, bn in enumerate(BAND_NAMES):
            sb = e["gap_share_band"][bi]
            ls = "-" if lab.split("_")[1] == "cpt" else ":"
            axs[2].plot(xs[2:], 100 * np.array(sb[1:]), color=bcol[bi], ls=ls, lw=1.2,
                        label=f"{lab.split('_')[1]} {bn}")
    for ax in axs:
        ax.set_xscale("log")
        ax.axhline(0, color="k", lw=0.5)
        for D in (4096, 16384):
            ax.axvline(D, color="grey", lw=0.6)
        ax.set_xlabel("distance from target back to the token (tokens; 1k-chunk gaps)")
        ax.tick_params(labelsize=7.5)
    axs[0].set_ylabel("% of the step gradient (projection share)")
    axs[0].set_title("Whole model: % of the gradient at each distance\n(star = own label, distance 0; first dot = rest of the "
                     "same 1k chunk)", fontsize=8.5)
    axs[1].set_ylabel("% per (target chunk, token chunk) pair")
    axs[1].set_yscale("log")
    axs[1].set_title("Per dependency: share / number of chunk pairs at that gap\n(per-row mean ± SE; removes the fewer-far-pairs "
                     "effect)", fontsize=8.5)
    axs[2].set_ylabel("% of that band's gradient")
    axs[2].set_title("By block band, gaps >= 1 chunk (solid: dense CPT, dotted: base)", fontsize=8.5)
    axs[0].legend(fontsize=7)
    axs[1].legend(fontsize=7)
    axs[2].legend(fontsize=6, ncol=2)
    fig.suptitle(f"Gradient percentage as a function of dependency distance (exact additive decomposition, sketched; "
                 f"{res[labs[0]]['rows']} rows x {res[labs[0]]['T'] // 1024}k = one step; grey: 4k / 16k)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "sum_distance_pct.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
