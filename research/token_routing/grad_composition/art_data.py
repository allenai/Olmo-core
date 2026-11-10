"""Compact data extract for the claude.ai artifact: results/artifact_data.json (per label x task profiles, rounded)
and results/artifact_heat.json (one example row per task for a few labels, uint8-quantized token x block maps).

    python research/token_routing/grad_composition/art_data.py
"""
from __future__ import annotations

import base64
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
#: label -> (display name, row set, steps, kind)
LABELS = {
    # Beaker wave (primary): one source for each series. Lambda copies of base / dense 366 / router 366 agree
    # (base exactly; dense 366 is a separate training run: blocks 0-6 share 0.20 vs 0.23, ablation +0.31 vs +0.23).
    "bk-mt4rows-base": ("base", "mt4", 0, "base"),
    "bk-mt4rows-mt4-st31": ("dense 31", "mt4", 31, "dense"),
    "bk-mt4rows-mt4-st61": ("dense 61", "mt4", 61, "dense"),
    "bk-mt4rows-mt4-st122": ("dense 122", "mt4", 122, "dense"),
    "bk-mt4rows-mt4-st366": ("dense 366", "mt4", 366, "dense"),
    "bk-mt4rows-mt4-st366-lr2e-6": ("dense 366, LR 2e-6", "mt4", 366, "dense-lr2e-6"),
    "bk-mt4rows-mt4-otr10M8-st366": ("router otr10M8, 366", "mt4", 366, "router"),
    "bk-mt11rows-base": ("base", "mt11", 0, "base"),
    "bk-mt11rows-mt11-st165": ("dense 165", "mt11", 165, "dense"),
    "bk-mt11rows-mt11-st330": ("dense 330", "mt11", 330, "dense"),
    "bk-mt11rows-mt11-st495": ("dense 495", "mt11", 495, "dense"),
    "bk-mt11rows-mt11-otr10M8-st330": ("router otr10M8, 330", "mt11", 330, "router"),
    "bk-mt11rows-mt11-otr10M8-st495": ("router otr10M8, 495", "mt11", 495, "router2"),
}
HEAT_LABELS = ["bk-mt4rows-base", "bk-mt4rows-mt4-st366", "bk-mt4rows-mt4-otr10M8-st366", "bk-mt11rows-base",
               "bk-mt11rows-mt11-st495", "bk-mt11rows-mt11-otr10M8-st495"]
BINS = 256


def clean_rows(rows, rowset):
    """mt11 rows: keep only rows that no mt11 training shard contains (the mt11 shards are shuffled supersets, so most
    rows of mt11s960 past 720 are still training rows of some mt11 run); the same subset for every checkpoint. mt4
    rows (mt4s1952 rows 1464+, prefix shards): all clean."""
    if rowset != "mt11":
        return [True] * len(rows)
    return [not any(s_.startswith("mt11") for s_ in x["seen_in"]) for x in rows]


def r(x, d=3):
    return np.round(np.asarray(x, dtype=float), d).tolist()


def pack(a):
    a = np.asarray(a, dtype=np.float32)
    fin = np.isfinite(a)
    lo, hi = float(np.percentile(a[fin], 1)), float(np.percentile(a[fin], 99.5))
    q = np.clip((np.where(fin, a, lo) - lo) / max(hi - lo, 1e-9), 0, 1)
    return {"shape": list(a.shape), "lo": round(lo, 3), "hi": round(hi, 3),
            "b64": base64.b64encode((q * 255).round().astype(np.uint8).tobytes()).decode()}


def main():
    out = {"labels": {}, "data": {}}
    meta0 = None
    for lab, (name, rowset, steps, kind) in LABELS.items():
        d = os.path.join(RES, lab)
        if not os.path.exists(os.path.join(d, "probe.json")):
            continue
        meta = json.load(open(os.path.join(d, "probe.json")))
        meta0 = meta0 or meta
        st = dict(np.load(os.path.join(d, "probe_stats.npz")))
        cats = meta["cats"]
        body = [cats.index(c) for c in cats if c[-2:] in ("_g", "_n")]
        gold = [cats.index(c) for c in cats if c.endswith("_g")]
        rows = meta["rows"]
        out["labels"][lab] = {"name": name, "rowset": rowset, "steps": steps, "kind": kind,
                              "ckpt": meta["ckpt_resolved"], "rows_per_task": meta["rows_per_task"]}
        out["data"][lab] = {}
        ok = clean_rows(rows, rowset)
        for t in meta["tasks"] + ["all"]:
            idx = np.array([i for i, x in enumerate(rows) if (t == "all" or x["task"] == t) and ok[i]])
            if not len(idx):
                continue
            e = {"eval_size": int(len(idx)), "ce": round(float(np.mean([rows[i]["ce"] for i in idx])), 4),
                 "n_docs": round(float(np.mean([rows[i]["n_docs"] for i in idx])), 1),
                 "n_body": round(float(np.mean([rows[i]["n_body"] for i in idx])), 0),
                 "T": round(float(np.mean([rows[i]["T"] for i in idx])), 0)}
            for q in ("g2", "pw", "dh", "attr", "am"):
                cs = st[f"cat_sum/{q}"][idx]
                sh = cs / np.clip(cs.sum(-1, keepdims=True), 1e-30, None)
                if q in ("g2", "pw"):
                    e[f"catshare/{q}"] = r(sh.mean(0), 3)  # (L, ncat)
                e[f"body/{q}"] = r(sh[:, :, body].sum(-1).mean(0))
                e[f"gold/{q}"] = r(sh[:, :, gold].sum(-1).mean(0))
                e[f"conc10/{q}"] = r(st[f"conc/{q}"][idx][:, :, 3].mean(0))
                if q == "g2":
                    e[f"conc/{q}"] = r(st[f"conc/{q}"][idx].mean(0))
                e[f"docconc10/{q}"] = r(st[f"docconc10/{q}"][idx].mean(0))
                if q == "g2":
                    t10 = st[f"top10_cat/{q}"][idx]
                    e[f"top10/{q}"] = r((t10 / np.clip(t10.sum(-1, keepdims=True), 1, None)).mean(0))
            e["cnt_share"] = r((st["cat_cnt"][idx] / st["cat_cnt"][idx].sum(-1, keepdims=True)).mean(0))
            e["xcorr/g"] = r(st["xcorr/g"][idx].mean(0), 2)
            e["xjacc/g"] = r(st["xjacc/g"][idx].mean(0), 2)
            for q in ("g_dh", "g_attr", "g_pw", "g_dh0", "g_am", "dh_am"):
                e[f"agree/{q}"] = r(st[f"agree/{q}"][idx].mean(0))
            e["split"] = r(st["split"][idx].mean(0))
            a = st["abl_dce"][idx]
            e["abl_mean"], e["abl_se"] = r(a.mean(0)), r(a.std(0) / np.sqrt(len(idx)))
            e["abl_overlap_g7"] = r(st["abl_overlap_g7"][idx].mean(0))
            e["am_missed_by_g7"] = r(st["am_missed_by_g7"][idx].mean(0))
            pg = st["pgrad"][idx]
            tot = pg.sum((1, 2))
            e["pgrad_block"] = r((pg[:, :32].sum(-1) / tot[:, None]).mean(0), 4)
            e["pgrad_head"] = round(float((pg[:, 32].sum(-1) / tot).mean()), 4)
            e["pgrad_parts"] = r((pg[:, :32].sum(1) / tot[:, None]).mean(0))  # mixer / ffn / norms over blocks
            if t != "all":
                s = np.array(meta["task_summed_pgrad"][t])
                e["pgrad_block_summed"] = r(s[:32].sum(-1) / s.sum(), 4)
                e["pgrad_head_summed"] = round(float(s[32].sum() / s.sum()), 4)
            e["pyr_contain_mass"] = r(st["pyr_contain_mass"][idx].mean(0))
            e["pyr_contain_cnt"] = r(st["pyr_contain_cnt"][idx].mean(0))
            e["pyr_nested"] = r(st["pyr_nested"][idx].mean(0))
            e["drop_conc"] = r(st["drop_conc"][idx].mean(0))
            out["data"][lab][t] = e
    for k in ("cats", "ks", "splits", "split_cols", "abl_rules", "full_blocks", "pyr", "pyr_ka", "pyr_kb", "pyr_sets",
              "pyr_tgts", "drop_conc_cols"):
        out[k] = meta0[k]
    json.dump(out, open(os.path.join(RES, "artifact_data.json"), "w"), separators=(",", ":"))
    heat = {}
    for lab in HEAT_LABELS:
        d = os.path.join(RES, lab)
        for f in sorted(glob.glob(os.path.join(d, "probe_ex_*.npz"))):
            key = os.path.basename(f)[len("probe_ex_"):-4]
            task = key.rsplit("_", 1)[0]
            if f"{lab}|{task}" in heat:  # first example row per task only
                continue
            z = np.load(f)
            T = z["G"].shape[1]
            nb = min(BINS, T)
            ed = np.linspace(0, T, nb + 1).astype(int)
            pool = lambda v, fn=np.max: np.stack([fn(v[:, ed[i]:ed[i + 1]], 1) for i in range(nb)], 1)  # noqa: E731
            dh = np.sqrt(z["UA"].astype(np.float32) ** 2 + z["UF"].astype(np.float32) ** 2)
            cat = z["cat"].astype(int)
            toks = None
            tf = f[:-4] + ".tok.json"
            if os.path.exists(tf):
                toks = json.load(open(tf))
            G = z["G"].astype(np.float32)
            top = {}
            if toks:
                for b in (7, 16, 24):
                    ii = np.argsort(-G[b])[:10]
                    top[b] = [[int(t), toks[t], int(cat[t])] for t in ii]
            heat[f"{lab}|{task}"] = {
                "T": int(T), "nb": nb,
                "G": pack(np.log10(pool(G) + 1e-12)),
                "dh": pack(np.log10(pool(dh) + 1e-12)),
                "am": pack(np.log10(pool(z["AMans"].astype(np.float32), np.sum) + 1e-6)),
                "cat": [int(np.bincount(cat[ed[i]:ed[i + 1]], minlength=13).argmax()) for i in range(nb)],
                "top": top,
            }
    json.dump(heat, open(os.path.join(RES, "artifact_heat.json"), "w"), separators=(",", ":"))
    print(f"labels: {list(out['labels'])}; heat rows: {len(heat)}; sizes "
          f"{os.path.getsize(os.path.join(RES, 'artifact_data.json')) / 1e6:.1f} MB / "
          f"{os.path.getsize(os.path.join(RES, 'artifact_heat.json')) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
