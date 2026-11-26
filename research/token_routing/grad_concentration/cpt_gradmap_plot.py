"""Draw and summarize cpt_gradmap.py's CPT gradient maps (run on the node that holds the npz: ~100 MB per 128k row).

    python research/token_routing/grad_concentration/cpt_gradmap_plot.py --in-dir /data/prasann/grad_composition/cpt_gradmap \
        --sets 32k:base_32k:cpt_32k,64k:base_64k:cpt_64k --out-dir <repo>/research/token_routing/grad_concentration/gradmaps_cpt

Per row (document) and set: map_<set>_r<row>_{E,G}[_lin|_linclip].png (full CPT loss; base and dense-CPT panels with a
per-token loss strip), span_<set>_r<row>_{G,E}[...].png (loss of the last / mid 1,024 tokens only),
ctx_<set>_r<row>.png (context share of each cell's hidden-state gradient from the K/V-detached pass).
Tokens are binned to 512 columns (bin = T / 512 tokens): E is summed per bin, ||dL/dh|| is the bin RMS.
Summaries over all rows of a set: sum_{conc,band,distance,corr,kinds,ctx,flops}_<set>.png and summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NB = 32
NCOL = 512
DPI = 110
BANDS = ((0, 7), (7, 16), (16, 24), (24, 32))
BAND_NAMES = ("0-6", "7-15", "16-23", "24-31")
FRACS = (0.7, 0.9, 0.95)
DIST = (1024, 4096, 16384)
CUTS = (2, 4, 8, 12, 16, 20, 24, 28)
FULL_ATTN = (3, 7, 11, 15, 19, 23, 27, 31)
COL = {"base": "#4c72b0", "cpt": "#dd8452", "stage3": "#55a868"}
SCALES = ("log", "lin", "linclip")
SUFFIX = {"log": "", "lin": "_lin", "linclip": "_linclip"}


# ---------------------------------------------------------------------------------------------------------- helpers
def binsum(M, B):
    T = M.shape[-1] // B * B
    return M[..., :T].reshape(*M.shape[:-1], -1, B).sum(-1)


def binmean(v, B):
    T = v.shape[-1] // B * B
    return np.nanmean(v[..., :T].reshape(*v.shape[:-1], -1, B), -1)


def need(v, frac):
    """Share of entries (largest first) needed to hold `frac` of sum(v)."""
    s = np.sort(v.reshape(-1))[::-1]
    c = np.cumsum(s)
    if c[-1] <= 0:
        return np.nan
    return (np.searchsorted(c, frac * c[-1]) + 1) / s.size


def ranks(x):
    r = np.empty(x.size)
    r[np.argsort(x, kind="stable")] = np.arange(x.size)
    return r


def spearman(a, b):
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def ms(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan)
    return (float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan)


def scaled(M, scale):
    """M raw nonnegative values. Returns (matrix, vmin, vmax, text)."""
    if scale == "log":
        L = np.log10(np.maximum(M, 1e-30))
        f = L[M > 0]
        lo, hi = np.percentile(f, 1), np.percentile(f, 99.8)
        return L, lo, hi, f"log10, {lo:.1f} .. {hi:.1f}"
    mx = float(M.max())
    if scale == "lin":
        return M, 0.0, mx, f"linear, 0 .. max {mx:.2e}"
    cap = float(np.percentile(M, 99))
    return M, 0.0, cap, f"linear, 0 .. 99th pct {cap:.2e} (max {mx:.2e})"


# ------------------------------------------------------------------------------------------------------------- load
def load(in_dir, lab):
    meta = json.load(open(os.path.join(in_dir, lab, "rows.json")))
    rows = {r["row"]: r for r in meta["rows"]}
    return meta, rows


def arrays(in_dir, lab, row):
    with np.load(os.path.join(in_dir, lab, f"r{row:04d}.npz")) as z:
        return {k: z[k] for k in z.files}


def topics(dev):
    p = os.path.join(dev, "rows.tsv")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return {int(r["row"]): (r["topic"], int(r["doc_tokens"]), int(r["window"])) for r in csv.DictReader(f, delimiter="\t")}


def load_vocab(path):
    if path and os.path.exists(path):
        return json.load(open(path))
    try:
        import glob as _g

        from tokenizers import Tokenizer

        cands = sorted(_g.glob("/data/prasann/olmo3_cptdrop/hf_cache/**/models--allenai--dolma2-tokenizer/snapshots/*/tokenizer.json",
                               recursive=True))
        tok = Tokenizer.from_file(cands[0])
        voc = [tok.decode([i]) for i in range(tok.get_vocab_size())]
        if path:
            json.dump(voc, open(path, "w"))
        return voc
    except Exception as e:  # noqa: BLE001
        print(f"no tokenizer ({e}); token classes by text are skipped")
        return None


# ------------------------------------------------------------------------------------------------------- token kinds
def ngram_repeats(ids, n=4):
    """label_rep[t]: the n-gram ending at token t+1 (ids[t-n+2 .. t+1]) occurred earlier in the row (its label is
    copyable); source[t]: token t lies in an n-gram that occurs again later (it is a copy source)."""
    T = ids.size
    lab = np.zeros(T, bool)
    src = np.zeros(T, bool)
    first = {}
    last_end = {}
    keys = [tuple(ids[i - n + 1:i + 1]) for i in range(n - 1, T)]
    for j, k in enumerate(keys):
        e = j + n - 1  # end index of the n-gram
        if k in first:
            lab[e - 1] = True  # the logit at e-1 predicts ids[e]
            s0 = first[k]
            src[s0 - n + 1:s0 + 1] = True
        else:
            first[k] = e
        last_end[k] = e
    return lab, src


def kinds(d, voc):
    ids = d["ids"].astype(np.int64)
    T = ids.size
    ce = d["ce"]
    lab_ce = np.full(T, np.nan)
    lab_ce[:-1] = ce[1:]  # token t's own label CE (logit at t scores token t+1)
    ok = np.isfinite(lab_ce)
    thr_hi = np.nanpercentile(lab_ce, 90)
    thr_lo = np.nanpercentile(lab_ce, 50)
    rep, src = ngram_repeats(ids)
    K = {"high-loss label (top 10%)": ok & (lab_ce >= thr_hi), "low-loss label (bottom 50%)": ok & (lab_ce <= thr_lo),
         "label copyable (4-gram seen earlier)": rep, "copy source (4-gram reappears later)": src,
         "first 4 tokens (attention sink)": np.arange(T) < 4,
         "tokens 4-255 of the row": (np.arange(T) >= 4) & (np.arange(T) < 256)}
    if voc is not None:
        txt = [voc[i] if i < len(voc) else "" for i in range(max(ids.max() + 1, 1))]
        nl = np.array([("\n" in txt[i]) for i in ids])
        pun = np.array([(txt[i].strip() != "" and all(not c.isalnum() for c in txt[i].strip())) for i in ids])
        dig = np.array([any(c.isdigit() for c in txt[i]) for i in ids])
        K.update({"newline token": nl, "punctuation-only token": pun & ~nl, "token with a digit": dig})
    return K


# ------------------------------------------------------------------------------------------------------- statistics
def row_stats(d, item):
    E, G = d["E"].astype(np.float64), d["G"].astype(np.float64)
    T = E.shape[1]
    G2 = G ** 2
    o = {}
    Et = E.sum(0)
    o["tok_need_E"] = {f: need(Et, f) for f in FRACS}
    o["tok_need_G2"] = {f: need(G2.sum(0), f) for f in FRACS}
    o["cell_need_E"] = {f: need(E, f) for f in FRACS}
    o["blk_need_E"] = {f: [need(E[b], f) for b in range(NB)] for f in FRACS}
    o["blk_need_G2"] = {f: [need(G2[b], f) for b in range(NB + 1)] for f in FRACS}
    o["E_block_share"] = (E.sum(1) / E.sum()).tolist()
    pg = np.array(item["pgrad"])  # (33, 3)
    tot = pg.sum()
    o["pgrad_block_share"] = (pg[:NB].sum(1) / tot).tolist()
    o["pgrad_head_share"] = float(pg[NB].sum() / tot)
    o["coherence"] = (pg[:NB].sum(1) / np.maximum(E.sum(1), 1e-30)).tolist()  # ||sum_t g_t||^2 / sum_t ||g_t||^2
    for nm in ("late", "mid"):
        p = np.array(item[f"pgrad_{nm}"])
        o[f"pgrad_block_share_{nm}"] = (p[:NB].sum(1) / p.sum()).tolist()
        Es = d[f"E_{nm}"].astype(np.float64)
        o[f"E_block_share_{nm}"] = (Es.sum(1) / Es.sum()).tolist()
    # cross-block rank agreement (log values; all tokens but the last, whose label is absent)
    idx = np.linspace(0, T - 2, min(T - 1, 8192)).astype(int)
    for q, M in (("E", E), ("G", G)):
        R = np.stack([ranks(np.log(M[b, idx] + 1e-30)) for b in range(M.shape[0])])
        o[f"corr_{q}"] = np.corrcoef(R)
    # distance profile for the target spans (energy at tokens BEFORE the span, by distance back from the span start)
    for nm in ("late", "mid"):
        s = item[f"s_{nm}"]
        Es = d[f"E_{nm}"].astype(np.float64)
        Gs2 = d[f"G_{nm}"].astype(np.float64) ** 2
        pre = np.arange(T) < s
        dist = s - np.arange(T)  # 1 = the token just before the span
        res = {}
        for q, M in (("E", Es), ("G2", Gs2)):
            tot_b = M.sum(1)
            pre_b = M[:, pre].sum(1)
            res[f"{q}_pre_share"] = (pre_b / np.maximum(tot_b, 1e-30)).tolist()
            cum = {}
            for D in DIST:
                m = pre & (dist <= D)
                cum[D] = (M[:, m].sum(1) / np.maximum(pre_b, 1e-30)).tolist()
            res[f"{q}_within"] = cum
            # cumulative curve over log-spaced distances (per block band and overall)
            grid = np.unique(np.round(np.logspace(0, np.log10(max(s, 2)), 60)).astype(int))
            Mp = M[:, pre][:, ::-1]  # column 0 = distance 1
            cs = np.cumsum(Mp, 1)
            res[f"{q}_curve_grid"] = grid.tolist()
            bands = [(b0, min(b1, M.shape[0])) for b0, b1 in BANDS] + [(0, M.shape[0])]
            res[f"{q}_curve"] = [(cs[b0:b1].sum(0)[np.minimum(grid, Mp.shape[1]) - 1]
                                  / max(cs[b0:b1, -1].sum(), 1e-30)).tolist() for b0, b1 in bands]
        o[f"dist_{nm}"] = res
    # context share (K/V-detached pass)
    if "C" in d:
        C = d["C"].astype(np.float64)
        Go2 = d["G_own"].astype(np.float64) ** 2
        o["ctx_share_block"] = (C.sum(1) / np.maximum(G2.sum(1), 1e-30)).tolist()
        o["own_share_block"] = (Go2.sum(1) / np.maximum(G2.sum(1), 1e-30)).tolist()
        o["ctx_frac_block"] = (C.sum(1) / np.maximum(C.sum(1) + Go2.sum(1), 1e-30)).tolist()
    # FLOP bound from attribution: keep the top tokens from block L on (nested), FLOP ratio (L + k (32 - L)) / 32
    tot_E = E.sum()
    fl = {}
    for f in (0.9, 0.95):
        best = []
        for L in CUTS:
            head = E[:L].sum()
            tail = E[L:].sum(0)
            needed = (f * tot_E - head) / max(tail.sum(), 1e-30)
            k = 0.0 if needed <= 0 else need(tail, min(needed, 1.0))
            best.append(((L + k * (NB - L)) / NB, k))
        fl[f] = best
    o["flop_tok"] = fl
    o["flop_cell"] = {f: need(E, f) for f in (0.9, 0.95)}
    return o


def kind_stats(d, K):
    E = d["E"].astype(np.float64)
    out = {}
    tot_band = [E[b0:b1].sum() for b0, b1 in BANDS] + [E.sum()]
    C = d["C"].astype(np.float64) if "C" in d else None
    for nm, m in K.items():
        if m.sum() == 0:
            continue
        r = {"tok_share": float(m.mean())}
        r["E_share_band"] = [float(E[b0:b1, m].sum() / tb) for (b0, b1), tb in zip(BANDS + ((0, NB),), tot_band)]
        if C is not None:
            r["C_share"] = float(C[:, m].sum() / C.sum())
        out[nm] = r
    return out


# ----------------------------------------------------------------------------------------------------------- figures
def loss_strip(ax, ds, labs, names, B, T):
    for lab, d in zip(labs, ds):
        lc = binmean(np.where(np.isfinite(d["ce"]), d["ce"], np.nan), B)
        ax.plot(np.arange(lc.size) * B + B / 2, lc, lw=0.7, color=COL[lab.split("_")[0]], label=names[lab])
    ax.set_xlim(0, T)
    ax.set_ylabel(f"CE / tok\n(bin mean)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="upper right", ncol=2, frameon=False)
    ax.set_xticks([])


def map_panel(ax, M, T, title, vmin, vmax, nrow, cmap="magma", marks=()):
    im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
                   extent=(0, T, -0.5, nrow - 0.5))
    for x in marks:
        ax.axvline(x, color="#00e5ff", lw=0.8, ls="--")
    ax.set_yticks([0, 7, 16, 24, 31] + ([32] if nrow == 33 else []))
    ax.set_ylabel("block" if nrow == NB else "h_b (b=32: final)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8, loc="left")
    return im


def xaxis(ax, T):
    tk = np.linspace(0, T, 9)
    ax.set_xticks(tk)
    ax.set_xticklabels([f"{int(t) // 1024}k" for t in tk], fontsize=7)
    ax.set_xlabel("token position", fontsize=7.5)


def draw_full(sets_d, labs, names, setn, row, head, out, qty, scale, items):
    d0 = sets_d[0]
    T = d0["ce"].size
    B = T // NCOL
    fig = plt.figure(figsize=(14, 2.0 + 3.1 * len(labs)))
    gs = fig.add_gridspec(1 + len(labs), 2, height_ratios=[1.0] + [3.0] * len(labs), width_ratios=[60, 1],
                          hspace=0.35, wspace=0.02)
    loss_strip(fig.add_subplot(gs[0, 0]), sets_d, labs, names, B, T)
    for i, (lab, d) in enumerate(zip(labs, sets_d)):
        if qty == "E":
            M = binsum(d["E"].astype(np.float64), B)
            q = f"weight-gradient energy E (sum over the block's linears of ||dL/dy_t||^2 ||x_t||^2), summed per {B}-token bin"
        else:
            M = np.sqrt(binsum(d["G"].astype(np.float64) ** 2, B) / B)
            q = f"||dL/dh_b|| (RMS per {B}-token bin)"
        Ms, lo, hi, txt = scaled(M, scale)
        ax = fig.add_subplot(gs[1 + i, 0])
        im = map_panel(ax, Ms, T, f"{names[lab]}   CE {items[i]['loss']:.3f}   {q}   [{txt}]", lo, hi, M.shape[0])
        if i == len(labs) - 1:
            xaxis(ax, T)
        else:
            ax.set_xticks([])
        fig.colorbar(im, cax=fig.add_subplot(gs[1 + i, 1]))
    fig.suptitle(f"{head}\nfull CPT loss (mean next-token CE over all {T:,} targets), one dense forward + backward, "
                 f"frozen weights", fontsize=9, x=0.01, ha="left")
    fig.savefig(os.path.join(out, f"map_{setn}_r{row}_{qty}{SUFFIX[scale]}.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def draw_span(sets_d, labs, names, setn, row, head, out, qty, scale, items, span):
    T = sets_d[0]["ce"].size
    B = T // NCOL
    fig = plt.figure(figsize=(14, 1.2 + 2.7 * 2 * len(labs)))
    gs = fig.add_gridspec(2 * len(labs), 2, width_ratios=[60, 1], hspace=0.45, wspace=0.02)
    r = 0
    for nm in ("late", "mid"):
        for lab, d, it in zip(labs, sets_d, items):
            s = it[f"s_{nm}"]
            if qty == "E":
                M = binsum(d[f"E_{nm}"].astype(np.float64), B)
            else:
                M = np.sqrt(binsum(d[f"G_{nm}"].astype(np.float64) ** 2, B) / B)
            Ms, lo, hi, txt = scaled(np.where(M > 0, M, 0.0), scale)
            if scale == "log":
                Ms = np.where(M > 0, Ms, np.nan)
            ax = fig.add_subplot(gs[r, 0])
            cm = plt.get_cmap("magma").copy()
            cm.set_bad("#e8e8e8")
            im = map_panel(ax, Ms, T, f"{names[lab]}  loss = CE of the {nm} {span:,}-token span [{s // 1024}k, "
                           f"{(s + span) / 1024:.0f}k) only ({it[f'loss_{nm}']:.3f})   [{txt}]", lo, hi, M.shape[0],
                           cmap=cm, marks=(s, s + span))
            if s > 4096:
                ax.axvline(s - 4096, color="white", lw=0.8, ls=":")
            if r == 2 * len(labs) - 1:
                xaxis(ax, T)
            else:
                ax.set_xticks([])
            fig.colorbar(im, cax=fig.add_subplot(gs[r, 1]))
            r += 1
    q = "weight-gradient energy E per bin (sum)" if qty == "E" else "||dL/dh_b|| (bin RMS)"
    fig.suptitle(f"{head}\ntarget-span maps: where the gradient of a late / mid span's loss lands over the earlier "
                 f"context; {q}; grey = no gradient (after the span); dashed = span; white dotted = 4,096 tokens before the span (sliding-window reach of 24 of the 32 blocks)", fontsize=9, x=0.01, ha="left")
    fig.savefig(os.path.join(out, f"span_{setn}_r{row}_{qty}{SUFFIX[scale]}.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def draw_ctx(sets_d, labs, names, setn, row, head, out, items):
    T = sets_d[0]["ce"].size
    B = T // NCOL
    fig = plt.figure(figsize=(14, 2.0 + 3.1 * len(labs)))
    gs = fig.add_gridspec(1 + len(labs), 2, height_ratios=[1.0] + [3.0] * len(labs), width_ratios=[60, 1],
                          hspace=0.35, wspace=0.02)
    loss_strip(fig.add_subplot(gs[0, 0]), sets_d, labs, names, B, T)
    for i, (lab, d) in enumerate(zip(labs, sets_d)):
        C = binsum(d["C"].astype(np.float64), B)
        O = binsum(d["G_own"].astype(np.float64) ** 2, B)
        F = C / np.maximum(C + O, 1e-30)
        ax = fig.add_subplot(gs[1 + i, 0])
        im = map_panel(ax, F, T, f"{names[lab]}   context share ||g_ctx||^2 / (||g_own||^2 + ||g_ctx||^2) per bin "
                       f"(g_own: keys/values detached = own label only; g_ctx = g_full - g_own)", 0, 1, F.shape[0],
                       cmap="RdBu_r")
        if i == len(labs) - 1:
            xaxis(ax, T)
        else:
            ax.set_xticks([])
        fig.colorbar(im, cax=fig.add_subplot(gs[1 + i, 1]))
    fig.suptitle(f"{head}\nown-label vs context split of the full-loss hidden-state gradient (red = gradient from "
                 f"serving as context to later tokens, blue = from the token's own label)", fontsize=9, x=0.01, ha="left")
    fig.savefig(os.path.join(out, f"ctx_{setn}_r{row}.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def summary_figs(setn, labs, names, S, KS, out, n_rows, T):
    tag = f"{setn}, {n_rows} documents (rows) per checkpoint"
    # concentration per block
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    for lab in labs:
        c = COL[lab.split("_")[0]]
        for f, ls in zip(FRACS, (":", "--", "-")):
            v = np.nanmean([s["blk_need_E"][f] for s in S[lab]], 0)
            axs[0].plot(range(NB), v, ls=ls, color=c, label=f"{names[lab]} {int(f * 100)}%")
            v = np.nanmean([s["blk_need_G2"][f] for s in S[lab]], 0)
            axs[1].plot(range(NB + 1), v, ls=ls, color=c)
        axs[2].plot(range(NB), np.mean([s["pgrad_block_share"] for s in S[lab]], 0), color=c, label=f"{names[lab]} exact ||grad W||^2")
        axs[2].plot(range(NB), np.mean([s["E_block_share"] for s in S[lab]], 0), color=c, ls="--", label=f"{names[lab]} attribution sum E")
    axs[0].set_title("share of tokens holding 70 / 90 / 95% of block-b E", fontsize=8.5)
    axs[1].set_title("share of tokens holding 70 / 90 / 95% of ||dL/dh_b||^2", fontsize=8.5)
    axs[2].set_title("gradient energy share by block (full loss)", fontsize=8.5)
    for a in axs[:2]:
        a.set_yscale("log")
    for a in axs:
        a.set_xlabel("block")
        a.tick_params(labelsize=7.5)
    axs[0].legend(fontsize=6.5)
    axs[2].legend(fontsize=6.5)
    fig.suptitle(f"Concentration, {tag}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"sum_conc_{setn}.png"), dpi=DPI)
    plt.close(fig)
    # band shares
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for ai, (key, ttl) in enumerate((("", "full loss"), ("_late", "late-span loss"), ("_mid", "mid-span loss"))):
        x = np.arange(len(BANDS))
        for j, lab in enumerate(labs):
            v = np.array([s[f"pgrad_block_share{key}"] for s in S[lab]])
            vb = np.stack([v[:, b0:b1].sum(1) for b0, b1 in BANDS], 1)
            e = np.array([s[f"E_block_share{key}"] for s in S[lab]])
            eb = np.stack([e[:, b0:b1].sum(1) for b0, b1 in BANDS], 1)
            w = 0.8 / len(labs)
            axs[ai].bar(x + (j - (len(labs) - 1) / 2) * w, vb.mean(0), w * 0.95, yerr=vb.std(0, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else None,
                        color=COL[lab.split("_")[0]], label=f"{names[lab]} exact ||grad W||^2")
            axs[ai].plot(x + (j - (len(labs) - 1) / 2) * w, eb.mean(0), "k_", ms=14, mew=2, label="attribution sum E" if j == 0 else None)
        axs[ai].set_xticks(x)
        axs[ai].set_xticklabels([f"blocks {b}" for b in BAND_NAMES], fontsize=7.5)
        axs[ai].set_title(ttl, fontsize=8.5)
    axs[0].set_ylabel("share of parameter-gradient energy\n(incl. LM head in the total)")
    axs[0].legend(fontsize=6.5)
    fig.suptitle(f"Gradient energy by block band, {tag} (bars: exact; black ticks: sum of per-token E)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"sum_band_{setn}.png"), dpi=DPI)
    plt.close(fig)
    # distance profile
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
    for ai, (nm, q) in enumerate((("late", "E"), ("late", "G2"), ("mid", "E"), ("mid", "G2"))):
        ax = axs[ai]
        for lab in labs:
            grid = S[lab][0][f"dist_{nm}"][f"{q}_curve_grid"]
            cur = np.array([s[f"dist_{nm}"][f"{q}_curve"] for s in S[lab] if s[f"dist_{nm}"][f"{q}_curve_grid"] == grid])
            for bi, ls in zip((0, 1, 2, 3, 4), (":", "-.", "--", (0, (1, 3)), "-")):
                lbl = (f"{names[lab]}: " if bi == 4 else "") + ("all blocks" if bi == 4 else f"blocks {BAND_NAMES[bi]}")
                ax.plot(grid, cur[:, bi].mean(0), ls=ls, color=COL[lab.split("_")[0]], lw=1.6 if bi == 4 else 0.9, label=lbl)
        for D in DIST:
            ax.axvline(D, color="grey", lw=0.5)
        ax.axvline(4096, color="grey", lw=1.0)
        ax.set_xscale("log")
        ax.set_xlabel("distance back from the span start (tokens)", fontsize=7.5)
        ax.set_title(f"{nm} span, {'weight-grad E' if q == 'E' else '||dL/dh||^2'}", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
    axs[0].set_ylabel("cumulative share of the pre-span energy")
    axs[0].legend(fontsize=6, loc="lower right")
    fig.suptitle(f"How far back the gradient of a 1,024-token span's loss reaches, {tag} (grey: 1k / 4k (sliding window) / 16k)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"sum_distance_{setn}.png"), dpi=DPI)
    plt.close(fig)
    # rank agreement
    fig, axs = plt.subplots(1, 2 * len(labs), figsize=(4.2 * 2 * len(labs), 3.8))
    for j, lab in enumerate(labs):
        for k, q in enumerate(("E", "G")):
            Cm = np.mean([s[f"corr_{q}"] for s in S[lab]], 0)
            ax = axs[2 * j + k]
            im = ax.imshow(Cm, origin="lower", cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"{names[lab]}\nSpearman of per-token {'E' if q == 'E' else '||dL/dh||'} between blocks", fontsize=7.5)
            ax.set_xlabel("block")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Cross-block rank agreement of token importance (full loss), {tag}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"sum_corr_{setn}.png"), dpi=DPI)
    plt.close(fig)
    # token kinds
    kn = [k for k in KS[labs[0]][0] if all(k in ks for lab in labs for ks in KS[lab])]
    fig, axs = plt.subplots(1, len(BANDS) + 1, figsize=(18, 0.6 + 0.42 * len(kn) + 1.2), sharey=True)
    y = np.arange(len(kn))
    for bi in range(len(BANDS) + 1):
        ax = axs[bi]
        for j, lab in enumerate(labs):
            enr = np.array([[ks[k]["E_share_band"][bi] / ks[k]["tok_share"] for k in kn] for ks in KS[lab]])
            w = 0.8 / len(labs)
            ax.barh(y + (j - (len(labs) - 1) / 2) * w, enr.mean(0), w * 0.95, color=COL[lab.split("_")[0]], label=names[lab])
        ax.axvline(1, color="k", lw=0.6)
        ax.set_xscale("log")
        ax.set_title(f"blocks {BAND_NAMES[bi]}" if bi < len(BANDS) else "all blocks", fontsize=8.5)
        ax.set_xlabel("E share / token share", fontsize=7.5)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([f"{k} ({np.mean([ks[k]['tok_share'] for ks in KS[labs[0]]]) * 100:.1f}% of tokens)" for k in kn], fontsize=7)
    axs[0].legend(fontsize=6.5)
    fig.suptitle(f"Which tokens carry the weight-gradient energy (enrichment > 1 = more than their share), {tag}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"sum_kinds_{setn}.png"), dpi=DPI)
    plt.close(fig)
    # context share
    if all("ctx_share_block" in s for lab in labs for s in S[lab]):
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.6))
        for lab in labs:
            v = np.array([s["ctx_frac_block"] for s in S[lab]])
            ax.plot(range(NB + 1), np.median(v, 0), color=COL[lab.split("_")[0]], label=f"{names[lab]} (median over rows)")
            ax.fill_between(range(NB + 1), v.min(0), v.max(0), color=COL[lab.split("_")[0]], alpha=0.15)
        for L in FULL_ATTN:
            ax.axvline(L, color="grey", lw=0.4)
        ax.set_xlabel("block input h_b (grey: full-attention blocks)")
        ax.set_ylabel("context share of ||dL/dh_b||^2")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
        ax.set_title(f"own-label vs context, {tag}\n(shaded: min..max over rows)", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"sum_ctx_{setn}.png"), dpi=DPI)
        plt.close(fig)
    # FLOP bound
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.6))
    for lab in labs:
        for f, ls in ((0.9, "--"), (0.95, "-")):
            v = np.array([[x[0] for x in s["flop_tok"][f]] for s in S[lab]])
            ax.plot(CUTS, 1 / v.mean(0), ls=ls, color=COL[lab.split("_")[0]], marker="o", ms=3,
                    label=f"{names[lab]}, keep {int(f * 100)}% of E")
            cell = np.mean([s["flop_cell"][f] for s in S[lab]])
            ax.axhline(1 / cell, ls=ls, color=COL[lab.split("_")[0]], lw=0.6)
    ax.set_xlabel("split block L (all tokens below L; attribution top-k tokens from L on)")
    ax.set_ylabel("best-case per-step FLOP speedup")
    ax.set_yscale("log")
    ax.legend(fontsize=6.5)
    ax.set_title(f"Attribution-based bound (NOT a routing result), {tag}\nthin lines: cell-level (block x token) bound",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"sum_flops_{setn}.png"), dpi=DPI)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------- main
def agg(S, key, f=None):
    v = [s[key] if f is None else s[key][f] for s in S]
    return ms(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--sets", required=True, help="set:label_base:label_cpt,...")
    ap.add_argument("--names", default="", help="label=display name,...")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vocab", default="")
    ap.add_argument("--maps-rows", default="", help="set:row;row,... rows to draw maps for (default: all)")
    ap.add_argument("--scales", default="log,lin,linclip")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    names = dict(kv.split("=", 1) for kv in a.names.split(",") if kv)
    voc = load_vocab(a.vocab)
    maps_rows = {}
    for kv in a.maps_rows.split(","):
        if kv:
            s_, r_ = kv.split(":")
            maps_rows[s_] = [int(x) for x in r_.split(";")]
    summary = {}
    for spec in a.sets.split(","):
        setn, *labs = spec.split(":")
        for lab in labs:
            names.setdefault(lab, lab)
        metas = {lab: load(a.in_dir, lab) for lab in labs}
        rows = sorted(set.intersection(*[set(m[1]) for m in metas.values()]))
        top = topics(metas[labs[0]][0]["dev"])
        T = metas[labs[0]][0]["T"]
        span = metas[labs[0]][0]["span"]
        S = {lab: [] for lab in labs}
        KS = {lab: [] for lab in labs}
        per_row = []
        for row in rows:
            ds = [arrays(a.in_dir, lab, row) for lab in labs]
            items = [metas[lab][1][row] for lab in labs]
            K = kinds(ds[0], voc)
            for lab, d, it in zip(labs, ds, items):
                S[lab].append(row_stats(d, it))
                KS[lab].append(kind_stats(d, K))
            tp = top.get(row, ("?", 0, 0))
            head = (f"OLMo 3 7B, Longmino dev {setn} row {row}: topic {tp[0]}, document {tp[1]:,} tokens, window {tp[2]} "
                    f"(this row = tokens [{tp[2] * T:,}, {(tp[2] + 1) * T:,}) of it), no-cot (teacher-forced CE)")
            per_row.append(dict(row=row, topic=tp[0], doc_tokens=tp[1], **{f"loss_{lab}": it["loss"] for lab, it in zip(labs, items)}))
            if not maps_rows or row in maps_rows.get(setn, []):
                for sc in a.scales.split(","):
                    for qty in ("E", "G"):
                        draw_full(ds, labs, names, setn, row, head, a.out_dir, qty, sc, items)
                        draw_span(ds, labs, names, setn, row, head, a.out_dir, qty, sc, items, span)
                if all("C" in d for d in ds):
                    draw_ctx(ds, labs, names, setn, row, head, a.out_dir, items)
            print(f"[{setn}] row {row} done", flush=True)
            del ds
        n = len(rows)
        summary_figs(setn, labs, names, S, KS, a.out_dir, n, T)
        res = {"eval_size_rows": n, "T": T, "span": span, "rows": per_row, "ckpts": {}}
        for lab in labs:
            s = S[lab]
            e = {"name": names[lab], "loss": ms([it["loss"] for it in [metas[lab][1][r] for r in rows]])}
            for f in FRACS:
                e[f"tok_need_E_{f}"] = agg(s, "tok_need_E", f)
                e[f"tok_need_G2_{f}"] = agg(s, "tok_need_G2", f)
                e[f"cell_need_E_{f}"] = agg(s, "cell_need_E", f)
                e[f"blk_need_E_{f}"] = np.nanmean([x["blk_need_E"][f] for x in s], 0).tolist()
                e[f"blk_need_G2_{f}"] = np.nanmean([x["blk_need_G2"][f] for x in s], 0).tolist()
            for key in ("", "_late", "_mid"):
                v = np.array([x[f"pgrad_block_share{key}"] for x in s])
                ev = np.array([x[f"E_block_share{key}"] for x in s])
                e[f"band_pgrad{key}"] = [ms(v[:, b0:b1].sum(1)) for b0, b1 in BANDS]
                e[f"band_E{key}"] = [ms(ev[:, b0:b1].sum(1)) for b0, b1 in BANDS]
            e["pgrad_head_share"] = ms([x["pgrad_head_share"] for x in s])
            e["coherence_block"] = np.exp(np.mean(np.log([x["coherence"] for x in s]), 0)).tolist()
            C = np.mean([x["corr_E"] for x in s], 0)
            Cg = np.mean([x["corr_G"] for x in s], 0)
            e["spearman_E"] = {"7v16": float(C[7, 16]), "7v31": float(C[7, 31]), "16v31": float(C[16, 31]),
                               "0v7": float(C[0, 7]), "adjacent_mean": float(np.mean([C[b, b + 1] for b in range(NB - 1)]))}
            e["spearman_G"] = {"7v16": float(Cg[7, 16]), "7v31": float(Cg[7, 31]), "16v31": float(Cg[16, 31]),
                               "0v7": float(Cg[0, 7]), "7v32": float(Cg[7, 32])}
            for nm in ("late", "mid"):
                for q in ("E", "G2"):
                    pre = np.array([x[f"dist_{nm}"][f"{q}_pre_share"] for x in s])
                    e[f"dist_{nm}_{q}_pre_share_band"] = [ms(pre[:, b0:b1].mean(1)) for b0, b1 in BANDS]
                    for D in DIST:
                        w = np.array([x[f"dist_{nm}"][f"{q}_within"][D] for x in s])
                        e[f"dist_{nm}_{q}_within{D}_band"] = [ms(w[:, b0:b1].mean(1)) for b0, b1 in BANDS]
                    curve = np.array([x[f"dist_{nm}"][f"{q}_curve"][4] for x in s
                                      if x[f"dist_{nm}"][f"{q}_curve_grid"] == s[0][f"dist_{nm}"][f"{q}_curve_grid"]])
                    grid = np.array(s[0][f"dist_{nm}"][f"{q}_curve_grid"])
                    e[f"dist_{nm}_{q}_allblocks_within"] = {D: float(np.interp(np.log(D), np.log(grid), curve.mean(0)))
                                                            for D in DIST}
            if "ctx_frac_block" in s[0]:
                v = np.array([x["ctx_frac_block"] for x in s])
                e["ctx_frac_block_median"] = np.median(v, 0).tolist()
                e["ctx_frac_b0_8_16_24_32"] = [float(np.median(v[:, b])) for b in (0, 8, 16, 24, 32)]
            for f in (0.9, 0.95):
                v = np.array([[x2[0] for x2 in x["flop_tok"][f]] for x in s])
                k = np.array([[x2[1] for x2 in x["flop_tok"][f]] for x in s])
                e[f"flop_tok_{f}"] = {L: {"flop_ratio": float(v[:, i].mean()), "keep_from_L": float(k[:, i].mean())}
                                      for i, L in enumerate(CUTS)}
                bi = int(np.argmin(v.mean(0)))
                e[f"flop_tok_{f}_best"] = {"L": CUTS[bi], "flop_ratio": float(v[:, bi].mean()),
                                           "speedup": float(1 / v[:, bi].mean()), "keep_from_L": float(k[:, bi].mean())}
                e[f"flop_cell_{f}"] = ms([x["flop_cell"][f] for x in s])
            kn = KS[lab][0].keys()
            e["kinds"] = {k: {"tok_share": ms([ks[k]["tok_share"] for ks in KS[lab] if k in ks]),
                              "enrich_band": [ms([ks[k]["E_share_band"][bi] / ks[k]["tok_share"] for ks in KS[lab] if k in ks])
                                              for bi in range(len(BANDS) + 1)],
                              "C_enrich": ms([ks[k]["C_share"] / ks[k]["tok_share"] for ks in KS[lab] if k in ks and "C_share" in ks[k]])}
                          for k in kn}
            res["ckpts"][lab] = e
        summary[setn] = res
        json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1,
                  default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
        print(f"[{setn}] summary done ({n} rows)", flush=True)


if __name__ == "__main__":
    main()
