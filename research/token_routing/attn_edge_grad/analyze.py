"""Aggregate probe_edges outputs per condition -> summary JSON + CSVs + figures.

    python analyze.py --root RESULTS_DIR --conds c1,c2,... --out OUTDIR

Per condition (a directory of rows_s*.jsonl + row_*.pt):
* Q1 category table: share of each (query class -> key class) category in each mass (logit / value / attn), pooled
  over layers with equal weight per layer, next to its share of causal edges (= attention FLOPs); per layer too.
* concentration: edge fraction (greedy by mass density) and block fraction (greedy by mass) for 50/80/90/95/99% of
  the mass, per layer and pooled; raw-edge fractions from the histograms (rows with T <= 70k); distance profile.
* Q2: per pattern and mode, the kept edge fraction and per-row global cosine / norm ratio vs the full gradient
  (mean +- SE over rows), the batch cosine (all rows summed, from CountSketches), per-block cosine, MASK dCE/token,
  and the smallest kept fraction reaching cos 0.9 / 0.95 per pattern family.
All means are over rows; SE = std / sqrt(rows).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re

import numpy as np
import torch

CLS = ("pre", "gold", "dist", "question", "answer")
KEY = ("pre", "gold_same", "gold_other", "dist_same", "dist_other", "question", "answer")
QTY = ("logit", "value", "attn")
LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99)
NAMED = {  # the brief's categories -> (query classes, key classes)
    "answer->gold": (("answer",), ("gold_same", "gold_other")),
    "answer->distractor": (("answer",), ("dist_same", "dist_other")),
    "answer->question/instr": (("answer",), ("question", "pre", "answer")),
    "question->gold": (("question",), ("gold_same", "gold_other")),
    "question->distractor": (("question",), ("dist_same", "dist_other")),
    "question->question/instr": (("question",), ("question", "pre")),
    "gold->same doc": (("gold",), ("gold_same",)),
    "gold->other gold": (("gold",), ("gold_other",)),
    "gold->distractor": (("gold",), ("dist_other",)),
    "gold->instr": (("gold",), ("pre",)),
    "distractor->gold": (("dist",), ("gold_other",)),
    "distractor->same doc": (("dist",), ("dist_same",)),
    "distractor->other distractor": (("dist",), ("dist_other",)),
    "distractor->instr": (("dist",), ("pre",)),
    "instr->instr": (("pre",), ("pre",)),
}


def se(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def mse(x):
    x = np.asarray(x, float)
    return {"mean": float(np.nanmean(x)), "se": se(x), "n": int(np.isfinite(x).sum())}


def cat_id(qc, kc):
    return CLS.index(qc) * len(KEY) + KEY.index(kc)


def load(cdir):
    recs = {}
    for f in sorted(glob.glob(os.path.join(cdir, "rows_s*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            recs[(r["row"])] = r  # last write wins (resumed shards)
    return list(recs.values())


def q1_tables(recs):
    layers = sorted(recs[0]["q1"]["layers"], key=int)
    out = {"layers": layers, "categories": {}, "named": {}, "per_layer_named": {}, "conc": {}, "raw": {}, "dist": {}}
    ncat = len(CLS) * len(KEY)
    edges = np.array([np.array(r["q1"]["cat_edges"]) / np.sum(r["q1"]["cat_edges"]) for r in recs])
    for q in QTY:
        per_layer = np.zeros((len(recs), len(layers), ncat))
        for i, r in enumerate(recs):
            for j, L in enumerate(layers):
                d = r["q1"]["layers"][L][q]
                per_layer[i, j] = np.array(d["cat_mass_per_head"]).sum(0) / max(d["total"], 1e-300)
        pooled = per_layer.mean(1)
        out["categories"][q] = {f"{CLS[c // 7]}->{KEY[c % 7]}": {"mass": mse(pooled[:, c]), "edges": mse(edges[:, c])}
                                for c in range(ncat) if edges[:, c].mean() > 0 or pooled[:, c].mean() > 1e-6}
        out["named"][q] = {}
        out["per_layer_named"][q] = {}
        for nm, (qs, ks) in NAMED.items():
            ids = [cat_id(a, b) for a in qs for b in ks]
            out["named"][q][nm] = {"mass": mse(pooled[:, ids].sum(1)), "edges": mse(edges[:, ids].sum(1))}
            out["per_layer_named"][q][nm] = [float(per_layer[:, j, ids].sum(1).mean()) for j in range(len(layers))]
        # concentration (doc-pair blocks)
        c = {}
        for k, key in (("edge_frac_for_mass", "edge"), ("block_frac_for_mass", "block")):
            c[f"pooled_{key}"] = {str(l): mse([r["q1"]["pooled_conc"][q][k][i] for r in recs]) for i, l in enumerate(LEVELS)}
            c[f"layer_{key}"] = {L: {str(l): mse([r["q1"]["layers"][L][q]["conc"][k][i] for r in recs])
                                     for i, l in enumerate(LEVELS)} for L in layers}
        c["pooled_mass_at_edge_frac"] = np.mean([r["q1"]["pooled_conc"][q]["mass_at_edge_frac"] for r in recs], 0).tolist()
        out["conc"][q] = c
        # raw edges (histograms are log10-value bins; greedy = descending value)
        raw_rows = [r for r in recs if "raw_hist" in r["q1"]["layers"][layers[0]][q]]
        if raw_rows:
            ef = {L: [] for L in layers}
            curves = []
            for r in raw_rows:
                for L in layers:
                    h = np.array(r["q1"]["layers"][L][q]["raw_hist"])
                    cnt, mass = h[0][::-1], h[1][::-1]
                    cc, cm = np.cumsum(cnt) / cnt.sum(), np.cumsum(mass) / max(mass.sum(), 1e-300)
                    ef[L].append([float(np.interp(l, np.concatenate([[0], cm]), np.concatenate([[0], cc]))) for l in LEVELS])
            out["raw"][q] = {"rows": len(raw_rows),
                             "layer_edge_frac": {L: {str(l): mse(np.array(ef[L])[:, i]) for i, l in enumerate(LEVELS)} for L in layers},
                             "mean_over_layers": {str(l): mse(np.mean([np.array(ef[L])[:, i] for L in layers], 0))
                                                  for i, l in enumerate(LEVELS)}}
        # distance
        dc = np.array([r["q1"]["dist_cnt"] for r in recs])
        dm = np.array([np.mean([np.array(r["q1"]["layers"][L][q]["dist_mass"]).sum(0) / max(r["q1"]["layers"][L][q]["total"], 1e-300)
                                for L in layers], 0) for r in recs])
        out["dist"][q] = {"edge_share": (dc / dc.sum(1, keepdims=True)).mean(0).tolist(), "mass_share": dm.mean(0).tolist(),
                          "mass_share_se": [se(dm[:, b]) for b in range(dm.shape[1])]}
    return out


def q2_tables(recs, cdir):
    by = {}
    for r in recs:
        for x in r["q2"]:
            k = (x["mode"], x["pattern"])
            d = np.array(x["dots"]); n1 = np.array(x["n1"]); n2 = np.array(x["n2"])
            c = d.sum() / math.sqrt(n1.sum() * n2.sum())
            nr = math.sqrt(n2.sum() / n1.sum())
            blk = d / np.sqrt(np.maximum(n1 * n2, 1e-300))
            dce = (x["ce_sum"] - r["ce_sum"]) / r["n_ans"] if "ce_sum" in x else float("nan")
            by.setdefault(k, []).append((x["frac"], c, nr, dce, blk, r["row"]))
    # batch cosines from sketches
    sk_sum = {}
    for f in glob.glob(os.path.join(cdir, "row_*.pt")):
        s = torch.load(f, map_location="cpu")["sketch"]
        for k, v in s.items():
            sk_sum[k] = sk_sum.get(k, 0) + v.double()
    out = {}
    for (mode, pat), v in by.items():
        fr = np.array([x[0] for x in v]); c = np.array([x[1] for x in v]); nr = np.array([x[2] for x in v])
        dce = np.array([x[3] for x in v]); blk = np.array([x[4] for x in v])
        bc = float("nan")
        key = f"{mode}/{pat}"
        if key in sk_sum and "ref" in sk_sum:
            a, b = sk_sum["ref"], sk_sum[key]
            bc = float(a @ b / math.sqrt(float(a @ a) * float(b @ b)))
        out[key] = {"mode": mode, "pattern": pat, "frac": mse(fr), "cos": mse(c), "norm_ratio": mse(nr),
                    "dce_tok": mse(dce) if mode == "mask" else None, "batch_cos": bc,
                    "block_cos": blk.mean(0).tolist(), "rows": len(v)}
    # sketch accuracy check: per-row sketch cos vs exact (first rows)
    return out


def family(p):
    m = re.match(r"([a-z_]+?)(\d+)$", p)
    return m.group(1) if m and m.group(1) in ("oracle", "random", "attn", "local", "topk") else p


def thresholds(q2):
    out = {}
    for mode in ("detach", "mask"):
        fam = {}
        for k, v in q2.items():
            if v["mode"] != mode:
                continue
            fam.setdefault(family(v["pattern"]), []).append((v["frac"]["mean"], v["cos"]["mean"], v["batch_cos"]))
        for f, pts in fam.items():
            pts.sort()
            for th in (0.9, 0.95):
                ok = [p[0] for p in pts if p[1] >= th]
                okb = [p[0] for p in pts if p[2] >= th]
                out.setdefault(mode, {}).setdefault(f, {})[f"min_frac_cos{th}"] = min(ok) if ok else None
                out[mode][f][f"min_frac_batchcos{th}"] = min(okb) if okb else None
    return out


def write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) for c in cols])


def plot_cond(name, q1, q2, outdir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = {"oracle": "#1f77b4", "random": "#7f7f7f", "attn": "#2ca02c", "local": "#9467bd", "topk": "#8c564b"}
    for ax, mode in zip(axes, ("detach", "mask")):
        fam = {}
        for v in q2.values():
            if v["mode"] == mode:
                fam.setdefault(family(v["pattern"]), []).append(v)
        for f, vs in sorted(fam.items()):
            vs.sort(key=lambda v: v["frac"]["mean"])
            x = [v["frac"]["mean"] for v in vs]
            y = [v["cos"]["mean"] for v in vs]
            e = [v["cos"]["se"] for v in vs]
            if len(vs) > 1:
                ax.errorbar(x, y, yerr=e, marker="o", ms=4, label=f, color=colors.get(f), capsize=2)
            else:
                ax.errorbar(x, y, yerr=e, marker="s", ms=6, ls="none", label=f, capsize=2)
        ax.set_xscale("log")
        ax.axhline(0.9, ls=":", c="k", lw=0.8)
        ax.axhline(0.95, ls=":", c="k", lw=0.8)
        ax.set_xlabel("kept edge fraction (= attention FLOP fraction)")
        ax.set_title(f"{mode.upper()}: LoRA-gradient cosine vs full")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("per-row global cosine (mean +- SE)")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"cos_vs_frac_{name}.png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--conds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    allsum = {}
    curve_rows, cat_rows = [], []
    for c in a.conds.split(","):
        cdir = os.path.join(a.root, c)
        recs = load(cdir)
        if not recs:
            print("no rows", c)
            continue
        q1 = q1_tables(recs)
        q2 = q2_tables(recs, cdir)
        th = thresholds(q2)
        meta = {"rows": len(recs), "T": mse([r["T"] for r in recs]), "ndocs": mse([r["nd"] for r in recs]),
                "ce_tok": mse([r["ce_sum"] / r["n_ans"] for r in recs]), "adapter": recs[0]["adapter"],
                "drop_r": mse([r["drop_r"] for r in recs if r.get("drop_r") is not None]) if recs[0].get("drop_r") else None,
                "numbered_ok": float(np.mean([r["numbered_ok"] for r in recs])),
                "n_gold": mse([r["n_gold"] for r in recs]), "t_row_s": mse([r["t_row"] for r in recs])}
        allsum[c] = {"meta": meta, "q1": q1, "q2": q2, "thresholds": th}
        for k, v in q2.items():
            curve_rows.append(dict(cond=c, mode=v["mode"], pattern=v["pattern"], family=family(v["pattern"]),
                                   frac=v["frac"]["mean"], cos=v["cos"]["mean"], cos_se=v["cos"]["se"],
                                   batch_cos=v["batch_cos"], norm_ratio=v["norm_ratio"]["mean"],
                                   dce_tok=(v["dce_tok"] or {}).get("mean"), rows=v["rows"]))
        for q in QTY:
            for nm, d in q1["named"][q].items():
                cat_rows.append(dict(cond=c, qty=q, category=nm, mass=d["mass"]["mean"], mass_se=d["mass"]["se"],
                                     edges=d["edges"]["mean"], ratio=d["mass"]["mean"] / max(d["edges"]["mean"], 1e-12)))
        if not a.no_plots:
            plot_cond(c, q1, q2, a.out)
        print(f"{c}: {len(recs)} rows")
    json.dump(allsum, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    write_csv(os.path.join(a.out, "cos_vs_frac.csv"), curve_rows,
              ["cond", "mode", "pattern", "family", "frac", "cos", "cos_se", "batch_cos", "norm_ratio", "dce_tok", "rows"])
    write_csv(os.path.join(a.out, "categories.csv"), cat_rows,
              ["cond", "qty", "category", "mass", "mass_se", "edges", "ratio"])


if __name__ == "__main__":
    main()
