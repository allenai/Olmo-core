"""Aggregate probe.py outputs (results/<label>/probe.json + probe_stats.npz + example rows) into
results/summary.json (per task x checkpoint means), results/heatmaps.json (downsampled token x block arrays for the
artifact) and figures in results/fig/.

    python research/token_routing/grad_composition/analyze.py [--labels base,mt4-st31,...]
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SERIES = {
    "mt4 dense (LR 5e-6)": ["base", "mt4-st31", "mt4-st61", "mt4-st122", "mt4-st366"],
    "mt11 dense (LR 5e-6)": ["base", "mt11-st165", "mt11-st330"],
}
ROUTERS = ["mt4-otr10M8-st366", "mt4-st366-lr2e-6", "mt11-otr20M8-st330"]
MASSES = ["g2", "pw", "attr", "dh", "am", "amall"]
XC = ["g", "pw", "attr", "dh", "am"]
AGREE = ["g_dh", "g_attr", "g_pw", "g_h", "g_dh0", "g_am", "dh_am", "pw_am"]
HEAT_Q = {"G": "log10 ||dL/dh_b||", "pw": "log10 param-grad share (w_out, w2)", "dh": "log10 ||h_{b+1}-h_b||",
          "attr": "log10 |first-order skip attribution|"}
BINS = 320


def load(label):
    d = os.path.join(RES, label)
    if not os.path.exists(os.path.join(d, "probe.json")):
        return None
    meta = json.load(open(os.path.join(d, "probe.json")))
    st = dict(np.load(os.path.join(d, "probe_stats.npz")))
    return meta, st, d


def mean_rows(x, idx):
    return np.nanmean(x[idx], axis=0) if len(idx) else None


def summarize(meta, st):
    cats = meta["cats"]
    rows = meta["rows"]
    tasks = meta["tasks"]
    out = {}
    T = np.array([r["T"] for r in rows], dtype=float)
    cnt = st["cat_cnt"]  # (R, ncat)
    for task in tasks + ["all"]:
        idx = np.array([i for i, r in enumerate(rows) if task == "all" or r["task"] == task])
        if not len(idx):
            continue
        s = {"eval_size": int(len(idx)), "ce": float(np.mean([rows[i]["ce"] for i in idx])),
             "ce_median": float(np.median([rows[i]["ce"] for i in idx])),
             "T": float(T[idx].mean()), "n_body": float(np.mean([rows[i]["n_body"] for i in idx])),
             "cnt_share": (cnt[idx] / T[idx, None]).mean(0).tolist()}
        for q in MASSES:
            cs = st[f"cat_sum/{q}"][idx]  # (R, L, ncat)
            share = cs / np.clip(cs.sum(-1, keepdims=True), 1e-30, None)
            s[f"share/{q}"] = share.mean(0).tolist()
            # per-token enrichment: mass share / token-count share
            cshare = cnt[idx] / T[idx, None]
            enr = share / np.clip(cshare[:, None, :], 1e-12, None)
            enr[:, :, cnt[idx].min(0) == 0] = np.nan
            s[f"enrich/{q}"] = np.nanmedian(enr, axis=0).tolist()
            # total mass per layer (row mean of the per-layer sum) and the layer share of the row's total
            tot = cs.sum(-1)
            s[f"layer_total/{q}"] = tot.mean(0).tolist()
            s[f"layer_share/{q}"] = (tot / np.clip(tot.sum(-1, keepdims=True), 1e-30, None)).mean(0).tolist()
            t10 = st[f"top10_cat/{q}"][idx]
            s[f"top10/{q}"] = (t10 / np.clip(t10.sum(-1, keepdims=True), 1, None)).mean(0).tolist()
            s[f"conc/{q}"] = st[f"conc/{q}"][idx].mean(0).tolist()
            s[f"docconc10/{q}"] = st[f"docconc10/{q}"][idx].mean(0).tolist()
        for q in XC:
            s[f"xcorr/{q}"] = st[f"xcorr/{q}"][idx].mean(0).tolist()
        s["xjacc/g"] = st["xjacc/g"][idx].mean(0).tolist()
        for q in AGREE:
            s[f"agree/{q}"] = st[f"agree/{q}"][idx].mean(0).tolist()
        s["split"] = st["split"][idx].mean(0).tolist()
        s["abl_dce_mean"] = st["abl_dce"][idx].mean(0).tolist()
        s["abl_dce_median"] = np.median(st["abl_dce"][idx], axis=0).tolist()
        s["abl_dce_se"] = (st["abl_dce"][idx].std(0) / np.sqrt(len(idx))).tolist()
        s["abl_overlap_g7"] = st["abl_overlap_g7"][idx].mean(0).tolist()
        s["am_missed_by_g7"] = st["am_missed_by_g7"][idx].mean(0).tolist()
        if "pgrad" in st:
            pgr = st["pgrad"][idx]  # (R, 33, 3) parameter-gradient energy per block (mixer, ffn, norms) + head
            blk = pgr[:, :32].sum(-1)
            tot = pgr.sum((1, 2))
            s["pgrad_block_share"] = (blk / tot[:, None]).mean(0).tolist()  # per-row energy share per block
            s["pgrad_part_share"] = (pgr.sum(1) / tot[:, None]).mean(0).tolist()  # mixer / ffn / norms (+head in col 0)
            s["pgrad_head_share"] = float((pgr[:, 32].sum(-1) / tot).mean())
            # norm (not energy) share of blocks 0-6 vs 7-31: sqrt of the energy in each part over the total norm
            s["pgrad_energy_share_0_6"] = float((blk[:, :7].sum(1) / tot).mean())
            s["pgrad_energy_share_7_31"] = float((blk[:, 7:].sum(1) / tot).mean())
            s["pgrad_norm_ratio_0_6"] = float(np.sqrt(blk[:, :7].sum(1) / tot).mean())
            s["pgrad_norm_ratio_7_31"] = float(np.sqrt(blk[:, 7:].sum(1) / tot).mean())
        for k in ("pyr_contain_cnt", "pyr_contain_mass", "pyr_nested", "drop_conc"):
            if k in st:
                s[k] = st[k][idx].mean(0).tolist()
        out[task] = s
    return out


def pack(a):
    """float array -> {shape, lo, hi, b64 of uint8 quantization} (NaN -> 0)."""
    a = np.asarray(a, dtype=np.float32)
    fin = np.isfinite(a)
    lo, hi = (float(np.percentile(a[fin], 1)), float(np.percentile(a[fin], 99.5))) if fin.any() else (0.0, 1.0)
    q = np.clip((np.where(fin, a, lo) - lo) / max(hi - lo, 1e-9), 0, 1)
    return {"shape": list(a.shape), "lo": lo, "hi": hi,
            "b64": base64.b64encode((q * 255).round().astype(np.uint8).tobytes()).decode()}


def heat_row(npz, tokf):
    z = np.load(npz)
    T = z["G"].shape[1]
    nb = min(BINS, T)
    edges = np.linspace(0, T, nb + 1).astype(int)
    pw = np.sqrt(z["pwA"].astype(np.float32) ** 2 + z["pwF"].astype(np.float32) ** 2)
    attr = np.abs(z["attrA"].astype(np.float32)) + np.abs(z["attrF"].astype(np.float32))
    dh = np.sqrt(z["UA"].astype(np.float32) ** 2 + z["UF"].astype(np.float32) ** 2)
    arrs = {"G": z["G"].astype(np.float32), "pw": pw, "dh": dh, "attr": attr}
    out = {"T": int(T), "edges": edges.tolist()}
    for k, v in arrs.items():
        # max-pool energy within each bin (keeps single hot tokens visible), then log10
        m = np.stack([v[:, edges[i]:edges[i + 1]].max(1) for i in range(nb)], 1)
        out[k] = pack(np.log10(m + 1e-12))
    am = z["AMans"].astype(np.float32)
    out["am"] = pack(np.log10(np.stack([am[:, edges[i]:edges[i + 1]].sum(1) for i in range(nb)], 1) + 1e-6))
    cat = z["cat"].astype(int)
    # majority category per bin; gold flag share per bin
    out["cat"] = [int(np.bincount(cat[edges[i]:edges[i + 1]], minlength=13).argmax()) for i in range(nb)]
    toks = json.load(open(tokf)) if os.path.exists(tokf) else None
    if toks is not None:
        # the 12 hottest tokens by G at the block-7 input and at the block-24 input, with their text
        G = z["G"].astype(np.float32)
        for b in (7, 24):
            top = np.argsort(-G[b])[:12]
            out[f"top_g{b}"] = [{"t": int(t), "tok": toks[t], "cat": int(cat[t]), "g": float(G[b, t])} for t in top]
    return out


def figures(summ, labels, fig_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    s0 = next(iter(summ.values()))
    tasks = [t for t in s0 if t != "all"]
    cats = CATS_ORDER
    # 1. cross-layer Spearman of ||dL/dh_b|| per task, base vs last mt4 vs last mt11
    for lab in labels:
        if lab not in summ:
            continue
        fig, ax = plt.subplots(2, (len(tasks) + 1) // 2, figsize=(3 * ((len(tasks) + 1) // 2), 6))
        for a_, t in zip(ax.flat, tasks):
            a_.imshow(np.array(summ[lab][t]["xcorr/g"]), vmin=0, vmax=1, cmap="viridis")
            a_.set_title(t, fontsize=9)
            a_.set_xticks([0, 8, 16, 24, 32]), a_.set_yticks([0, 8, 16, 24, 32])
        fig.suptitle(f"{lab}: Spearman of body-token ||dL/dh_b|| between blocks")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"xcorr_g_{lab}.png"), dpi=110)
        plt.close(fig)
    # 2. category share of gradient energy by layer
    for lab in labels:
        if lab not in summ:
            continue
        fig, ax = plt.subplots(2, (len(tasks) + 1) // 2, figsize=(3.2 * ((len(tasks) + 1) // 2), 6), sharey=True)
        for a_, t in zip(ax.flat, tasks):
            sh = np.array(summ[lab][t]["share/g2"])
            bot = np.zeros(sh.shape[0])
            for ci, c in enumerate(cats):
                a_.bar(np.arange(sh.shape[0]), sh[:, ci], bottom=bot, width=1.0, label=c)
                bot += sh[:, ci]
            a_.set_title(t, fontsize=9)
        ax.flat[0].legend(fontsize=6, ncol=2)
        fig.suptitle(f"{lab}: share of ||dL/dh_b||^2 by token category")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"share_g2_{lab}.png"), dpi=110)
        plt.close(fig)


CATS_ORDER = []


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="")
    a = ap.parse_args()
    labels = a.labels.split(",") if a.labels else sorted(os.path.basename(os.path.dirname(p)) for p in
                                                         glob.glob(os.path.join(RES, "*", "probe.json")))
    summ, metas, heat, taskpg = {}, {}, {}, {}
    for lab in labels:
        L = load(lab)
        if L is None:
            print(f"[missing] {lab}")
            continue
        meta, st, d = L
        global CATS_ORDER
        CATS_ORDER = meta["cats"]
        summ[lab] = summarize(meta, st)
        if meta.get("task_summed_pgrad"):
            # the per-task SUMMED gradient (minibatch direction): energy share per block, 0-6 vs 7-31
            taskpg[lab] = {}
            for t, e in meta["task_summed_pgrad"].items():
                e = np.array(e)
                tot = e.sum()
                blk = e[:32].sum(-1)
                taskpg[lab][t] = {"block_share": (blk / tot).tolist(), "share_0_6": float(blk[:7].sum() / tot),
                                  "share_7_31": float(blk[7:].sum() / tot), "head_share": float(e[32].sum() / tot),
                                  "part_share": (e.sum(0) / tot).tolist()}
        metas[lab] = {k: meta[k] for k in ("ckpt_resolved", "seen_in_training", "tasks", "rows_per_task", "row_offset")}
        heat[lab] = {}
        for f in sorted(glob.glob(os.path.join(d, "probe_ex_*.npz"))):
            key = os.path.basename(f)[len("probe_ex_"):-4]
            heat[lab][key] = heat_row(f, f[:-4] + ".tok.json")
        print(f"{lab}: {len(meta['rows'])} rows, {len(heat[lab])} example rows")
    meta0 = json.load(open(os.path.join(RES, labels[0], "probe.json")))
    json.dump({"labels": labels, "series": SERIES, "routers": ROUTERS, "cats": meta0["cats"], "ks": meta0["ks"],
               "splits": meta0["splits"], "split_cols": meta0["split_cols"], "abl_rules": meta0["abl_rules"],
               "full_blocks": meta0["full_blocks"], "meta": metas, "summary": summ, "task_summed_pgrad": taskpg,
               **{k: meta0.get(k) for k in ("pyr", "pyr_ka", "pyr_kb", "pyr_sets", "pyr_tgts", "drop_conc_cols")}},
              open(os.path.join(RES, "summary.json"), "w"))
    json.dump(heat, open(os.path.join(RES, "heatmaps.json"), "w"))
    figures(summ, labels, os.path.join(RES, "fig"))
    print(f"wrote {RES}/summary.json, heatmaps.json, fig/")


if __name__ == "__main__":
    main()
