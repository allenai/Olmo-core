"""Draw gradmap.py's per-token x per-layer gradient maps and summaries (run on a compute node: it reads ~GBs of npz).

    python research/token_routing/grad_concentration/gradmap_plot.py --in-dir /data/prasann/gradmap/out \
        --labels base,f1952,f7808,recipe --out-dir <repo>/research/token_routing/grad_concentration/gradmaps

Per row (picked by the dense 1,952 rows/task checkpoint's answer CE per token; --rung 2k: per task the best- and
worst-solved row; --rung 8k: per task one median-CE row among the longest quarter): map_<task>_<kind>_{E,A}.png (one
panel per checkpoint), zoom_<task>_<kind>.png (gold document and query + answer at token resolution),
routed_<task>_<kind>.png (recipe: dense vs routed). Summaries over every row of the set (eval_size in each title):
sum_{layer,type,cumshare,layer_corr,routed}_<rung>.png and summary_<rung>.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

NB = 32
CUTS = (7, 16)
#: 0.95-attribution oracle budgets at block 7 (records/oracle-tokens-under-routing.md, eval_size 512 rows per ckpt):
#: (global, per row, per document) shares of body tokens. base / recipe use the 1,952 rows/task numbers.
BUDGET = {"f7808": (0.0063, 0.056, 0.159)}
BUDGET_DEFAULT = (0.0198, 0.090, 0.195)
NAMES = {"base": "base", "f1952": "dense 1,952 rows/task", "f7808": "dense 7,808 rows/task",
         "recipe": "recipe (lora64all32r20to7, 1,952 rows/task)"}
#: segment codes: 0 instruction / markers, 1 distractor doc, 2 gold doc, 3 query, 4 answer
SEG_COL = ["#bdbdbd", "#9ecae1", "#e6550d", "#756bb1", "#d62728"]
SEG_NAME = ["instruction / markers", "distractor doc", "gold doc", "query", "answer"]
TY = ("answer", "prompt+query", "gold doc", "distractor doc")
GRID = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
DPI = 110


def segs(d):
    ty, qm, cid = d["ty"], d["qm"].astype(bool), d["cid"]
    s = np.zeros(ty.shape[0], dtype=np.int8)
    s[ty == 3] = 1
    s[ty == 2] = 2
    s[qm & (ty == 1)] = 3
    s[ty == 0] = 4
    return s


def doc_bounds(cid):
    return np.flatnonzero((cid[1:] != cid[:-1]) & (cid[1:] >= 0)) + 0.5


def topk_row(score, body, frac):
    keep = np.zeros_like(body)
    idx = np.flatnonzero(body)
    n = int(math.ceil(frac * idx.size))
    if n:
        keep[idx[np.argsort(-score[idx])[:n]]] = True
    return keep


def topk_doc(score, body, cid, frac):
    """routed_probe.doc_topk: each document its own top ceil(frac |doc|), then the row budget by score."""
    keep = np.zeros_like(body)
    idx = np.flatnonzero(body)
    for d in np.unique(cid[idx]):
        di = idx[cid[idx] == d]
        n = int(math.ceil(frac * di.size))
        keep[di[np.argsort(-score[di])[:n]]] = True
    left = int(math.ceil(frac * idx.size)) - int(keep.sum())
    if left > 0:
        rest = idx[~keep[idx]]
        keep[rest[np.argsort(-score[rest])[:left]]] = True
    return keep


def cumshare(v, grid):
    v = np.sort(v[np.isfinite(v)])[::-1]
    if v.size == 0 or v.sum() <= 0:
        return np.full(grid.size, np.nan), np.nan
    c = np.cumsum(v) / v.sum()
    at = c[np.clip(np.ceil(grid * v.size).astype(int) - 1, 0, v.size - 1)]
    f95 = (np.searchsorted(c, 0.95) + 1) / v.size
    return at, f95


def ranks(x):
    r = np.empty_like(x)
    r[np.argsort(x)] = np.arange(x.size)
    return r


def row_stats(d, item, routed):
    E = 10.0 ** d["E"].astype(np.float64)
    A = 10.0 ** d["A"].astype(np.float64)
    ty, body = d["ty"], d["body"].astype(bool)
    o = {}
    tot = E.sum()
    o["e_layer"] = E.sum(1) / tot
    o["a_layer_body"] = np.median(A[:, body], axis=1) if body.any() else np.full(NB, np.nan)
    o["a_layer_ans"] = np.median(A[:, ty == 0], axis=1)
    for nm, sl in (("all", slice(0, NB)), ("down7", slice(7, NB))):
        e = E[sl].sum(0)
        o[f"type_{nm}"] = np.array([e[ty == 0].sum(), e[ty == 1].sum(), e[ty == 2].sum(), e[ty == 3].sum()]) / e.sum()
        o[f"tokshare_{nm}"] = np.array([(ty == k).mean() for k in range(4)])
    d7 = E[7:].sum(0)
    o["down7_body"] = d7[body]
    gold, dis = d7[ty == 2], d7[ty == 3]
    o["gold_vs_dis"] = float(gold.mean() / dis.mean()) if gold.size and dis.size else np.nan
    o["cum_tok"], o["f95_tok"] = cumshare(d7[body], GRID)
    cells = E[7:][:, body].reshape(-1)
    o["cum_cell"], o["f95_cell"] = cumshare(cells, GRID)
    # spearman across body tokens between blocks (subsample for speed)
    bi = np.flatnonzero(body)
    if bi.size > 2000:
        bi = bi[np.linspace(0, bi.size - 1, 2000).astype(int)]
    Rk = np.stack([ranks(E[b, bi]) for b in range(NB)])
    o["layer_corr"] = np.corrcoef(Rk)
    if routed:
        Er = 10.0 ** d["E_routed"].astype(np.float64)
        Ar = 10.0 ** d["A_routed"].astype(np.float64)
        kb = np.isfinite(Ar) & body[None]
        lr = np.log10(Ar / A)
        o["routed_lr_body"] = np.array([np.nanmedian(np.where(kb[b], lr[b], np.nan)) if kb[b].any() else np.nan
                                        for b in range(NB)])
        o["routed_lr_ans"] = np.nanmedian(lr[:, ty == 0], axis=1)
        e = np.nansum(Er[7:], 0)
        o["type_routed_down7"] = np.array([e[ty == k].sum() for k in range(4)]) / e.sum()
        o["routed_e_layer"] = np.nansum(Er, 1) / np.nansum(Er)
        o["routed_total_ratio"] = float(np.nansum(Er) / tot)
        o["e_tot"], o["er_tot"] = float(tot), float(np.nansum(Er))
    return o


def load_label(in_dir, lab):
    meta = json.load(open(os.path.join(in_dir, lab, "rows.json")))
    rows = meta["rows"]
    return rows


def panel(ax, M, title, vmin, vmax, cmap="magma", cut=True):
    T = M.shape[1]
    im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
                   extent=(-0.5, T - 0.5, -0.5, NB - 0.5))
    if cut:
        for L in CUTS:
            ax.axhline(L - 0.5, color="#00e5ff", lw=0.8, ls="--")
    ax.set_ylabel("layer")
    ax.set_yticks([0, 7, 16, 31])
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.set_xticks([])
    return im


def strip(ax, rows, labels, T, colors):
    """rows: list of bool (T,) masks drawn as tick strips."""
    img = np.zeros((len(rows), T))
    for i, m in enumerate(rows):
        img[i, m] = i + 1
    cm = ListedColormap(["white"] + colors[:len(rows)])
    ax.imshow(img, aspect="auto", cmap=cm, vmin=0, vmax=len(rows), interpolation="nearest",
              extent=(-0.5, T - 0.5, len(rows) - 0.5, -0.5))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xticks([])


def segbar(ax, d, T, x0=0, x1=None):
    x1 = T if x1 is None else x1
    s = segs(d)[x0:x1]
    ax.imshow(s[None], aspect="auto", cmap=ListedColormap(SEG_COL), vmin=-0.5, vmax=4.5, interpolation="nearest",
              extent=(x0 - 0.5, x1 - 0.5, 0, 1))
    for b in doc_bounds(d["cid"]):
        if x0 <= b <= x1:
            ax.axvline(b, color="black", lw=0.4)
    ax.set_yticks([])
    ax.set_xlabel("token position")


def legend(fig):
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(SEG_COL, SEG_NAME)], loc="lower center", ncol=5,
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.0))


def oracle_masks(lab, d, batch_thr):
    E = 10.0 ** d["E"].astype(np.float64)
    d7 = E[7:].sum(0)
    body, cid = d["body"].astype(bool), d["cid"]
    fg, fr, fd = BUDGET.get(lab, BUDGET_DEFAULT)
    return [body & (d7 >= batch_thr), topk_row(d7, body, fr), topk_doc(d7, body, cid, fd)], (fg, fr, fd)


def title_of(task, item, d, extra=""):
    nd = int(d["cid"].max()) + 1
    gold = item["gold_docs"]
    return (f"{task}  n={nd} docs  no-cot  {item['T']:,} tokens  {item['n_answer']} answer tokens"
            f"  gold doc(s) {[g + 1 for g in gold] if gold else 'none (counting task)'}{extra}")


def draw_maps(lab_list, data, items, thr, task, kind, out, qty):
    d0 = data[lab_list[0]]
    T = d0["ty"].shape[0]
    n = len(lab_list)
    hr = []
    for lab in lab_list:
        hr += [3.0, 0.45 + (0.3 if "keep7" in data[lab] else 0)]
    H = n * 2.9 + 1.5
    fig, axs = plt.subplots(2 * n + 1, 1, figsize=(12.7, H),
                            gridspec_kw=dict(height_ratios=hr + [0.25], hspace=0.22))
    ces = "  ".join(f"{lab} {items[lab]['ce_tok']:.3f}" for lab in lab_list)
    fig.suptitle(title_of(task, items[lab_list[0]], d0) + f"\nanswer CE per token: {ces}   "
                 + ("color: log10 weight-gradient energy per (block, token) = sum over the block's linears of "
                    "||dL/dy_t||^2 ||x_t||^2" if qty == "E" else "color: log10 ||dL/dh_l[t]|| at block l's input"),
                 fontsize=9, y=0.995)
    for i, lab in enumerate(lab_list):
        d = data[lab]
        M = d[qty].astype(np.float64)
        f = M[np.isfinite(M)]
        vmin, vmax = np.percentile(f, 1), np.percentile(f, 99.8)
        im = panel(axs[2 * i], M, f"{NAMES.get(lab, lab)}   (color range per panel: log10 {vmin:.1f} .. {vmax:.1f}; "
                   f"dashed = blocks 7 / 16 cuts)", vmin, vmax)
        plt.colorbar(im, ax=axs[2 * i], pad=0.005, fraction=0.012)
        ms, (fg, fr, fd) = oracle_masks(lab, d, thr[lab])
        labels = [f"oracle global {fg:.1%}", f"oracle per row {fr:.1%}", f"oracle per doc {fd:.1%}"]
        cols = ["#2ca02c", "#1f77b4", "#9467bd"]
        if "keep7" in d:
            ms += [d["keep7"].astype(bool), d["keep16"].astype(bool)]
            labels += ["router keep @7 (20%)", "router keep @16 (7%)"]
            cols += ["#ff7f0e", "#d62728"]
        strip(axs[2 * i + 1], ms, labels, T, cols)
        cb = axs[2 * i + 1].inset_axes([1.005, 0, 0.012, 1])
        cb.axis("off")
    segbar(axs[-1], d0, T)
    legend(fig)
    fig.subplots_adjust(left=0.13, right=0.95, top=1 - 0.75 / H, bottom=0.8 / H)
    fig.savefig(os.path.join(out, f"map_{task}_{kind}_{qty}.png"), dpi=DPI)
    plt.close(fig)


def draw_zoom(lab_list, data, items, task, kind, out, tok):
    """token-resolution windows: the gold doc (or the first doc) and the query + answer tail."""
    d0 = data[lab_list[0]]
    T = d0["ty"].shape[0]
    cid, ty = d0["cid"], d0["ty"]
    gold = items[lab_list[0]]["gold_docs"]
    gd = gold[0] if gold else 0
    gi = np.flatnonzero(cid == gd)
    w1 = (max(0, gi[0] - 20), min(T, gi[0] + 140)) if gi.size else (0, 160)
    ai = np.flatnonzero(ty == 0)
    w2 = (max(0, ai[0] - 110), min(T, ai[-1] + 3))
    fig, axs = plt.subplots(len(lab_list) * 2 + 1, 2, figsize=(12.7, 2.0 * len(lab_list) + 1.8),
                            gridspec_kw=dict(height_ratios=[1.6, 1.6] * len(lab_list) + [0.3], hspace=0.25,
                                             width_ratios=[w1[1] - w1[0], w2[1] - w2[0]], wspace=0.06))
    fig.suptitle(title_of(task, items[lab_list[0]], d0) + f"\nzoom: gold doc {gd + 1} window (left) and query + answer "
                 "(right); top = log10 weight-gradient energy, bottom = log10 ||dL/dh||", fontsize=9, y=0.995)
    for i, lab in enumerate(lab_list):
        d = data[lab]
        for j, qty in enumerate(("E", "A")):
            M = d[qty].astype(np.float64)
            f = M[np.isfinite(M)]
            vmin, vmax = np.percentile(f, 1), np.percentile(f, 99.8)
            for c, (a0, a1) in enumerate((w1, w2)):
                ax = axs[2 * i + j, c]
                ax.imshow(M[:, a0:a1], aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax,
                          interpolation="nearest", extent=(a0 - 0.5, a1 - 0.5, -0.5, NB - 0.5))
                for L in CUTS:
                    ax.axhline(L - 0.5, color="#00e5ff", lw=0.7, ls="--")
                ax.set_xticks([])
                ax.set_yticks([0, 7, 16, 31] if c == 0 else [])
                if c == 0:
                    ax.set_ylabel(f"{lab}\n{qty}", fontsize=7.5)
    for c, (a0, a1) in enumerate((w1, w2)):
        ax = axs[-1, c]
        segbar(ax, d0, T, a0, a1)
        if tok is not None:
            ids = d0["ids"][a0:a1]
            step = max(1, (a1 - a0) // 70)
            ax.set_xticks(np.arange(a0, a1, step))
            ax.set_xticklabels([tok.decode([int(t)]).replace("\n", "\\n")[:8] for t in ids[::step]], rotation=90,
                               fontsize=5.5)
    legend(fig)
    H = 2.0 * len(lab_list) + 1.8
    fig.subplots_adjust(left=0.07, right=0.99, top=1 - 0.6 / H, bottom=(1.5 if tok is not None else 0.8) / H)
    fig.savefig(os.path.join(out, f"zoom_{task}_{kind}.png"), dpi=DPI)
    plt.close(fig)


def draw_routed(d, item, task, kind, out):
    T = d["ty"].shape[0]
    fig, axs = plt.subplots(7, 1, figsize=(12.7, 11.5), gridspec_kw=dict(height_ratios=[3, 3, 3, 3, 3, 3, 0.25 * 2],
                                                                         hspace=0.18))
    fig.suptitle(title_of(task, item, d, f"\nrecipe checkpoint, same row: full forward vs ROUTED forward (router keep "
                 f"{item['keep7']:.0%} of body at block 7, {item['keep16']:.0%} at 16; dropped cells grey). Answer CE/tok "
                 f"dense {item['ce_tok']:.3f}, routed {item['ce_routed_tok']:.3f}"), fontsize=9, y=0.995)
    for j, qty in enumerate(("E", "A")):
        M, Mr = d[qty].astype(np.float64), d[f"{qty}_routed"].astype(np.float64)
        f = np.concatenate([M[np.isfinite(M)], Mr[np.isfinite(Mr)]])
        vmin, vmax = np.percentile(f, 1), np.percentile(f, 99.8)
        nm = "weight-grad energy" if qty == "E" else "||dL/dh||"
        cm = plt.get_cmap("magma").copy()
        cm.set_bad("#d9d9d9")
        for k, (MM, tt) in enumerate(((M, f"dense: log10 {nm}"), (Mr, f"routed: log10 {nm}"))):
            im = panel(axs[3 * j + k], np.ma.masked_invalid(MM), tt + f"  (shared range {vmin:.1f} .. {vmax:.1f})",
                       vmin, vmax, cmap=cm)
            plt.colorbar(im, ax=axs[3 * j + k], pad=0.005, fraction=0.012)
        cmr = plt.get_cmap("RdBu_r").copy()
        cmr.set_bad("#d9d9d9")
        p_ = 1 if qty == "E" else 2  # E is a sum of squares; A is a norm (compare squared shares)
        tot, totr = np.log10(np.nansum(10 ** (p_ * M))) / p_, np.log10(np.nansum(10 ** (p_ * Mr))) / p_
        im = panel(axs[3 * j + 2], np.ma.masked_invalid((Mr - totr) - (M - tot)),
                   f"log10 routed / dense {nm}, each map normalised to its own total (kept cells); totals: routed / dense "
                   f"= {10 ** (p_ * (totr - tot)):.3g}" + (" (squared)" if p_ == 2 else "")
                   + " -- sums of squared per-token terms of THIS row; not a gradient-norm ratio (see sum_routed)",
                   -1.5, 1.5, cmap=cmr)
        plt.colorbar(im, ax=axs[3 * j + 2], pad=0.005, fraction=0.012)
    segbar(axs[-1], d, T)
    legend(fig)
    fig.subplots_adjust(left=0.07, right=0.95, top=1 - 0.75 / 11.5, bottom=0.8 / 11.5)
    fig.savefig(os.path.join(out, f"routed_{task}_{kind}.png"), dpi=DPI)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--labels", default="base,f1952,f7808,recipe")
    ap.add_argument("--pick-by", default="f1952")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--rung", default="8k", choices=("2k", "8k"), help="which rows to draw (summaries: every row)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    labs = a.labels.split(",")
    rows = {lab: load_label(a.in_dir, lab) for lab in labs}
    n = min(len(v) for v in rows.values())
    for lab in labs:
        assert [r["md5"] for r in rows[lab][:n]] == [r["md5"] for r in rows[labs[0]][:n]], lab
    tasks = sorted({r["task"] for r in rows[labs[0]]}, key=["contradiction", "nq", "outlier", "oolong"].index)
    # ------------------------------------------------------------------ per-row stats (streaming)
    S = {lab: [] for lab in labs}
    for lab in labs:
        for r in rows[lab][:n]:
            d = np.load(os.path.join(a.in_dir, lab, f"r{r['idx']:04d}.npz"))
            S[lab].append(row_stats(d, r, "E_routed" in d.files))
        print(f"stats {lab}: {n} rows", flush=True)
    # batch-global oracle thresholds (per 16-row batch, by this checkpoint's own down7 and global budget)
    thr = {lab: {} for lab in labs}
    for lab in labs:
        fg = BUDGET.get(lab, BUDGET_DEFAULT)[0]
        for J in sorted({r["batch"] for r in rows[lab][:n]}):
            v = np.concatenate([S[lab][i]["down7_body"] for i, r in enumerate(rows[lab][:n]) if r["batch"] == J])
            k = max(1, int(round(fg * v.size)))
            thr[lab][J] = np.sort(v)[::-1][k - 1]
    # ------------------------------------------------------------------ pick rows
    P = rows[a.pick_by][:n]
    picks = []
    for t in tasks:
        ix = [i for i, r in enumerate(P) if r["task"] == t]
        ce = lambda i: P[i]["ce_tok"]  # noqa: E731
        if a.rung == "2k":  # the best- and worst-solved row (by answer CE per token)
            picks += [(t, "2k-solved", min(ix, key=ce)), (t, "2k-unsolved", max(ix, key=ce))]
        else:  # one 8k row: the median-CE row among the longest quarter
            Ts = np.array([P[i]["T"] for i in ix])
            long_ = sorted([i for i in ix if P[i]["T"] >= np.percentile(Ts, 75)], key=ce)
            picks += [(t, "8k", long_[len(long_) // 2])]
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    except Exception as exc:  # noqa: BLE001
        print(f"no tokenizer ({exc}); zoom plots without token text", flush=True)
    picked = []
    for t, kind, i in picks:
        data = {lab: dict(np.load(os.path.join(a.in_dir, lab, f"r{rows[lab][i]['idx']:04d}.npz"))) for lab in labs}
        items = {lab: rows[lab][i] for lab in labs}
        th = {lab: thr[lab][rows[lab][i]["batch"]] for lab in labs}
        for qty in ("E", "A"):
            draw_maps(labs, data, items, th, t, kind, a.out_dir, qty)
        draw_zoom(labs, data, items, t, kind, a.out_dir, tok)
        for lab in labs:
            if "E_routed" in data[lab]:
                draw_routed(data[lab], items[lab], t, kind, a.out_dir)
        picked.append(dict(task=t, kind=kind, idx=i, T=P[i]["T"], gold_docs=P[i]["gold_docs"],
                           ce_tok={lab: items[lab]["ce_tok"] for lab in labs},
                           ce_routed_tok={lab: items[lab].get("ce_routed_tok") for lab in labs}))
        print(f"drew {t} {kind} row {i} T={P[i]['T']}", flush=True)
    summarize(labs, rows, S, n, tasks, a.out_dir, picked, a.rung)


def ms(x):
    x = np.asarray([v for v in x if v is not None and np.all(np.isfinite(v))])
    if x.size == 0:
        return np.nan, np.nan
    return x.mean(0), x.std(0, ddof=1) / math.sqrt(len(x)) if len(x) > 1 else np.nan


def summarize(labs, rows, S, n, tasks, out, picked, rung):
    cols = {"base": "#7f7f7f", "f1952": "#1f77b4", "f7808": "#d62728", "recipe": "#2ca02c"}
    cols = {lab: cols.get(lab, "#2ca02c") for lab in labs}
    rl = [lab for lab in labs if "routed_lr_body" in S[lab][0]]
    per_task = {t: [i for i, r in enumerate(rows[labs[0]][:n]) if r["task"] == t] for t in tasks}
    es = {t: len(v) for t, v in per_task.items()}
    J = dict(rung=rung, eval_size_rows=n, eval_size_per_task=es, picked=picked, by_label={})
    # -- gradient mass vs layer
    fig, axs = plt.subplots(2, len(tasks), figsize=(12.7, 6.2), sharex=True)
    for c, t in enumerate(tasks):
        for lab in labs:
            m, se = ms([S[lab][i]["e_layer"] for i in per_task[t]])
            axs[0, c].plot(range(NB), m, color=cols[lab], label=lab)
            axs[0, c].fill_between(range(NB), m - 2 * se, m + 2 * se, color=cols[lab], alpha=0.2)
            a_ = np.array([S[lab][i]["a_layer_body"] for i in per_task[t]])
            axs[1, c].plot(range(NB), np.nanmedian(np.log10(a_), 0), color=cols[lab])
            a2 = np.array([S[lab][i]["a_layer_ans"] for i in per_task[t]])
            axs[1, c].plot(range(NB), np.nanmedian(np.log10(a2), 0), color=cols[lab], ls=":")
        for L in CUTS:
            for r_ in range(2):
                axs[r_, c].axvline(L - 0.5, color="k", lw=0.6, ls="--")
        axs[0, c].set_title(f"{t} (eval_size {es[t]} rows)", fontsize=9)
        axs[1, c].set_xlabel("layer")
    axs[0, 0].set_ylabel("share of weight-grad energy\n(mean over rows, ±2 SE)")
    axs[1, 0].set_ylabel("median log10 ||dL/dh||\nsolid = doc body, dotted = answer")
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(f"Gradient mass vs layer, frozen checkpoints, no-cot, eval_size {n} rows ({n // len(tasks)} per task)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"sum_layer_{rung}.png"), dpi=DPI)
    plt.close(fig)
    # -- by token type + gold vs distractor
    fig, axs = plt.subplots(1, 2, figsize=(12.7, 4.6), gridspec_kw=dict(width_ratios=[2.2, 1]))
    tcol = ["#d62728", "#756bb1", "#e6550d", "#9ecae1"]
    xs, xl = [], []
    x = 0
    for t in tasks:
        for lab in labs:
            m, _ = ms([S[lab][i]["type_down7"] for i in per_task[t]])
            bottom = 0
            for k in range(4):
                axs[0].bar(x, m[k], bottom=bottom, color=tcol[k], width=0.8, label=TY[k] if x == 0 else None)
                bottom += m[k]
            tk, _ = ms([S[lab][i]["tokshare_down7"] for i in per_task[t]])
            axs[0].plot([x - 0.4, x + 0.4], [tk[0] + tk[1]] * 2, color="k", lw=0.8)
            xs.append(x)
            xl.append(f"{t[:6]}\n{lab}")
            x += 1
        x += 0.6
    axs[0].set_xticks(xs)
    axs[0].set_xticklabels(xl, fontsize=6.5)
    axs[0].set_ylabel("share of blocks 7-31 weight-grad energy")
    axs[0].set_title(f"energy by token type (black tick: answer+prompt share of TOKENS); eval_size {n} rows", fontsize=9)
    axs[0].legend(fontsize=7, loc="lower right")
    for j, lab in enumerate(labs):
        for c, t in enumerate([t for t in tasks if t != "oolong"]):
            v = np.log10([S[lab][i]["gold_vs_dis"] for i in per_task[t] if np.isfinite(S[lab][i]["gold_vs_dis"])])
            if v.size:
                axs[1].errorbar(c + 0.18 * j - 0.27, v.mean(), yerr=2 * v.std(ddof=1) / math.sqrt(v.size),
                                fmt="o", color=cols[lab], label=lab if c == 0 else None, ms=4)
    gt = [t for t in tasks if t != "oolong"]
    axs[1].set_xticks(range(len(gt)))
    axs[1].set_xticklabels(gt, fontsize=8)
    axs[1].axhline(0, color="k", lw=0.5)
    axs[1].set_ylabel("log10 (gold / distractor)\nmean energy per body token, blocks 7-31")
    axs[1].set_title(f"gold vs distractor per token (±2 SE; eval_size {es.get('nq', 0)} rows/task)", fontsize=9)
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"sum_type_{rung}.png"), dpi=DPI)
    plt.close(fig)
    # -- cumulative share curves, tokens vs cells
    fig, axs = plt.subplots(1, len(tasks), figsize=(12.7, 3.9), sharey=True)
    for c, t in enumerate(tasks):
        for lab in labs:
            ct = np.array([S[lab][i]["cum_tok"] for i in per_task[t]])
            cc = np.array([S[lab][i]["cum_cell"] for i in per_task[t]])
            axs[c].plot(GRID, np.nanmedian(ct, 0), color=cols[lab], label=f"{lab} tokens")
            axs[c].fill_between(GRID, np.nanpercentile(ct, 25, 0), np.nanpercentile(ct, 75, 0), color=cols[lab],
                                alpha=0.15)
            axs[c].plot(GRID, np.nanmedian(cc, 0), color=cols[lab], ls="--", label=f"{lab} cells")
        axs[c].set_xscale("log")
        axs[c].axhline(0.95, color="k", lw=0.5)
        axs[c].set_title(f"{t} (eval_size {es[t]} rows)", fontsize=9)
        axs[c].set_xlabel("top fraction of body tokens (solid) / cells (dashed)")
    axs[0].set_ylabel("share of blocks 7-31 energy\n(median row, IQR band)")
    axs[0].legend(fontsize=6)
    fig.suptitle("Cumulative energy share within a row: whole tokens (ranked by down7) vs token x block cells", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"sum_cumshare_{rung}.png"), dpi=DPI)
    plt.close(fig)
    # -- layer-to-layer rank agreement
    fig, axs = plt.subplots(1, len(labs), figsize=(12.7, 3.6), squeeze=False)
    axs = axs[0]
    for c, lab in enumerate(labs):
        Cm = np.nanmean(np.array([S[lab][i]["layer_corr"] for i in range(n)]), 0)
        im = axs[c].imshow(Cm, origin="lower", cmap="viridis", vmin=0, vmax=1)
        axs[c].set_title(f"{lab}", fontsize=9)
        axs[c].set_xlabel("layer")
        J["by_label"].setdefault(lab, {})["layer_corr_7_vs_16"] = float(Cm[7, 16])
        J["by_label"][lab]["layer_corr_7_vs_31"] = float(Cm[7, 31])
        J["by_label"][lab]["layer_corr_adjacent_mean"] = float(np.mean([Cm[b, b + 1] for b in range(NB - 1)]))
    axs[0].set_ylabel("layer")
    plt.colorbar(im, ax=axs, fraction=0.015)
    fig.suptitle(f"Spearman rank correlation of body-token energy between layers (mean over eval_size {n} rows)",
                 fontsize=10)
    fig.savefig(os.path.join(out, f"sum_layer_corr_{rung}.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    # -- routed (recipe)
    if rl:
        RL = rl[0]
        fig, axs = plt.subplots(1, 4, figsize=(12.7, 3.9), gridspec_kw=dict(width_ratios=[1, 1, 1.15, 1]))
        batches = sorted({r["batch"] for r in rows[RL][:n]})
        bratio = []
        for Jb in batches:
            ii = [i for i, r in enumerate(rows[RL][:n]) if r["batch"] == Jb]
            bratio.append(sum(S[RL][i]["er_tot"] for i in ii) / sum(S[RL][i]["e_tot"] for i in ii))
        bratio = np.array(bratio)
        J["routed_energy_ratio_per_batch"] = dict(median=float(np.median(bratio)), q25=float(np.percentile(bratio, 25)),
                                                  q75=float(np.percentile(bratio, 75)), n_batches=len(bratio))
        rr_ = np.array([S[RL][i]["routed_total_ratio"] for i in range(n)])
        J["routed_energy_ratio_per_row"] = dict(median=float(np.median(rr_)), share_above_10=float((rr_ > 10).mean()),
                                                share_below_0p1=float((rr_ < 0.1).mean()))
        for t in tasks:
            I = per_task[t]
            axs[3].scatter([rows[RL][i]["ce_tok"] for i in I], [S[RL][i]["routed_total_ratio"] for i in I], s=5,
                           label=t)
        axs[3].set_xscale("log")
        axs[3].set_yscale("log")
        axs[3].axhline(1, color="k", lw=0.5)
        axs[3].set_xlabel("dense answer CE per token (row)")
        axs[3].set_title(f"per-row routed / dense energy total\n16-row batch: median {np.median(bratio):.2f} "
                         f"(IQR {np.percentile(bratio, 25):.2f}-{np.percentile(bratio, 75):.2f})", fontsize=8.5)
        axs[3].legend(fontsize=6, markerscale=2)
        for t in tasks:
            lr = np.array([S[RL][i]["routed_lr_body"] for i in per_task[t]])
            la = np.array([S[RL][i]["routed_lr_ans"] for i in per_task[t]])
            p = axs[0].plot(range(NB), np.nanmedian(lr, 0), label=t)
            axs[0].plot(range(NB), np.nanmedian(la, 0), color=p[0].get_color(), ls=":")
            re_ = np.array([S[RL][i]["routed_e_layer"] for i in per_task[t]])
            de_ = np.array([S[RL][i]["e_layer"] for i in per_task[t]])
            axs[1].plot(range(NB), re_.mean(0), color=p[0].get_color())
            axs[1].plot(range(NB), de_.mean(0), color=p[0].get_color(), ls=":")
        for L in CUTS:
            axs[0].axvline(L - 0.5, color="k", lw=0.6, ls="--")
            axs[1].axvline(L - 0.5, color="k", lw=0.6, ls="--")
        axs[0].axhline(0, color="k", lw=0.5)
        axs[0].set_title("median log10 routed/dense ||dL/dh||\nkept body tokens (solid), answer (dotted)", fontsize=9)
        axs[0].legend(fontsize=7)
        axs[0].set_xlabel("layer")
        axs[1].set_title("energy share by layer: routed (solid) vs dense (dotted)", fontsize=9)
        axs[1].set_xlabel("layer")
        xs = np.arange(len(tasks))
        for k in range(4):
            dm = [ms([S[RL][i]["type_down7"] for i in per_task[t]])[0][k] for t in tasks]
            rm = [ms([S[RL][i]["type_routed_down7"] for i in per_task[t]])[0][k] for t in tasks]
            axs[2].bar(xs * 2.5 + k * 0.5, dm, 0.22, color=tcol[k], alpha=0.45)
            axs[2].bar(xs * 2.5 + k * 0.5 + 0.23, rm, 0.22, color=tcol[k], label=TY[k])
        axs[2].set_xticks(xs * 2.5 + 0.8)
        axs[2].set_xticklabels(tasks, fontsize=8)
        axs[2].set_title("blocks 7-31 energy share by type\nfaded = dense, solid = routed", fontsize=9)
        axs[2].legend(fontsize=6.5)
        fig.suptitle(f"Recipe checkpoint: what the routed forward does to the gradient (eval_size {n} rows)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"sum_routed_{rung}.png"), dpi=DPI)
        plt.close(fig)
    # -- numbers
    for lab in labs:
        b = J["by_label"].setdefault(lab, {})
        for t in tasks:
            I = per_task[t]
            q = {}
            q["ce_tok_median"] = float(np.median([rows[lab][i]["ce_tok"] for i in I]))
            for nm in ("type_down7", "type_all", "tokshare_down7"):
                q[nm] = dict(zip(TY, map(float, ms([S[lab][i][nm] for i in I])[0])))
            e = ms([S[lab][i]["e_layer"] for i in I])[0]
            q["energy_share_blocks_0_6"] = float(e[:7].sum())
            q["energy_share_blocks_7_15"] = float(e[7:16].sum())
            q["energy_share_blocks_16_31"] = float(e[16:].sum())
            q["peak_layer"] = int(np.argmax(e))
            g = [S[lab][i]["gold_vs_dis"] for i in I if np.isfinite(S[lab][i]["gold_vs_dis"])]
            q["gold_vs_dis_per_token_geomean"] = float(10 ** np.mean(np.log10(g))) if g else None
            q["f95_tokens_median"] = float(np.nanmedian([S[lab][i]["f95_tok"] for i in I]))
            q["f95_cells_median"] = float(np.nanmedian([S[lab][i]["f95_cell"] for i in I]))
            if "routed_lr_body" in S[lab][I[0]]:
                q["routed_ce_tok_median"] = float(np.median([rows[lab][i]["ce_routed_tok"] for i in I]))
                q["routed_lr_body_median_by_layer"] = [float(v) for v in
                                                       np.nanmedian([S[lab][i]["routed_lr_body"] for i in I], 0)]
                q["routed_total_energy_ratio_median"] = float(np.median([S[lab][i]["routed_total_ratio"] for i in I]))
                q["type_routed_down7"] = dict(zip(TY, map(float, ms([S[lab][i]["type_routed_down7"] for i in I])[0])))
                q["router_spearman7_mean"] = float(np.mean([rows[lab][i]["router_spearman7"] for i in I]))
                q["router_spearman16_mean"] = float(np.mean([rows[lab][i]["router_spearman16"] for i in I]))
            q["eval_size_rows"] = len(I)
            b[t] = q
    json.dump(J, open(os.path.join(out, f"summary_{rung}.json"), "w"), indent=1)
    print(json.dumps({lab: {t: {k: v for k, v in J["by_label"][lab][t].items() if not isinstance(v, list)}
                            for t in tasks} for lab in labs}, indent=0)[:6000])


if __name__ == "__main__":
    main()
