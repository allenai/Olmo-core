"""Aggregate cpt_probe.py outputs (results/cpt/<label>/cpt.json, cpt_stats.npz, cpt_ce.npz) into
results/cpt/summary.json and print the router-design numbers for the CPT addendum.

Long-context gain: per target token, CE(base) - CE(checkpoint), by position bin and by a repeat category of the target
(does the 4-gram ending at the target occur earlier in the row: last occurrence within 4096 tokens = near repeat; only
further back = far-only repeat, reachable only through the 8 full-attention layers; never = novel).

    python research/token_routing/grad_composition/cpt_analyze.py
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "cpt")
ORDER = ["base", "dlr20-12", "dlr20-48", "dlr20-120", "dlr20-240", "dlr20-480", "otr10m7-310", "stage3"]
STEPS = {"base": 0, "dlr20-12": 12, "dlr20-48": 48, "dlr20-120": 120, "dlr20-240": 240, "dlr20-480": 480,
         "otr10m7-310": 310, "stage3": None}
FAR = 4096
N = 4


def repeat_category(ids):
    """(L,) int8: 0 novel, 1 near repeat (last earlier occurrence of the n-gram ending here <= FAR back), 2 far-only."""
    L = len(ids)
    cat = np.zeros(L, dtype=np.int8)
    last = {}
    for j in range(N - 1, L):
        key = tuple(ids[j - N + 1:j + 1])
        p = last.get(key)
        if p is not None:
            cat[j] = 1 if j - p <= FAR else 2
        last[key] = j
    return cat


def load(lab):
    d = os.path.join(RES, lab)
    if not os.path.exists(os.path.join(d, "cpt.json")):
        return None
    return (json.load(open(os.path.join(d, "cpt.json"))), dict(np.load(os.path.join(d, "cpt_stats.npz"))),
            dict(np.load(os.path.join(d, "cpt_ce.npz"))))


def se(v):
    v = np.asarray(v, dtype=float)
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def main():
    labs = [l for l in ORDER if load(l) is not None]
    D = {l: load(l) for l in labs}
    m0, _, c0 = D[labs[0]]
    pos_edges = m0["pos_edges"]
    ids = c0["ids"]
    rep = np.stack([repeat_category(r) for r in ids])
    out = {"labels": labs, "steps": STEPS, "pos_edges": pos_edges, "doc_edges": m0["doc_edges"], "ks": m0["ks"],
           "pyr": m0["pyr"], "pyr_ka": m0["pyr_ka"], "pyr_sets": m0["pyr_sets"], "pyr_tgts": m0["pyr_tgts"],
           "full_blocks": m0["full_blocks"], "rep_share": [float((rep == c).mean()) for c in range(3)], "data": {}}
    base_ce = D["base"][2]["ce"].astype(np.float32) if "base" in D else None
    for l in labs:
        m, st, c = D[l]
        n = len(m["rows"])
        e = {"eval_size_rows": n, "loss": float(np.mean([r["loss"] for r in m["rows"]])),
             "loss_se": se([r["loss"] for r in m["rows"]])}
        pg = st["pgrad"]
        tot = pg.sum((1, 2))
        blk = pg[:, :32].sum(-1) / tot[:, None]
        e["pgrad_block"] = blk.mean(0).tolist()
        e["pgrad_head"] = float((pg[:, 32].sum(-1) / tot).mean())
        e["pgrad_0_6"], e["pgrad_0_6_se"] = float(blk[:, :7].sum(1).mean()), se(blk[:, :7].sum(1))
        e["pgrad_0_15"] = float(blk[:, :16].sum(1).mean())
        e["pgrad_parts"] = (pg[:, :32].sum(1) / tot[:, None]).mean(0).tolist()
        s = np.array(m["summed_pgrad"])
        e["pgrad_block_summed"] = (s[:32].sum(-1) / s.sum()).tolist()
        e["pgrad_0_6_summed"] = float(s[:7].sum() / s.sum())
        for k in ("conc", "pos_share", "doc_share", "xcorr", "contain_mass", "pyr_nested", "own_share", "ctx_share",
                  "own_cos", "own_rank_corr", "am_far_frac", "am_far_pos", "am_far_doc", "agree_g_amfar", "agree_g_dh",
                  "ctx_share_pos", "ce_pos", "doc_cnt"):
            # own / context shares are per-row energy ratios; a few rows with a tiny full gradient at block 0 make
            # the mean unstable (own and context parts can be anti-aligned), so use the median for those
            agg = np.nanmedian if k in ("own_share", "ctx_share", "own_cos", "ctx_share_pos", "own_rank_corr") else np.nanmean
            e[k] = agg(st[k], 0).tolist()
        e["conc_se_b24_10"] = se(st["conc"][:, 24, 3])
        e["xcorr_se_7_24"] = se(st["xcorr"][:, 7, 24])
        ce = c["ce"].astype(np.float32)
        e["ce_overall"] = float(np.nanmean(ce))
        if base_ce is not None and l != "base":
            g = base_ce - ce  # > 0: improved over base
            pos = np.arange(ce.shape[1])
            gain = {}
            for bi in range(len(pos_edges) - 1):
                pm = (pos >= pos_edges[bi]) & (pos < pos_edges[bi + 1])
                for cc, cn in enumerate(("novel", "near", "far_only")):
                    msk = pm[None, :] & (rep == cc) & ~np.isnan(g)
                    rows_g = [np.nanmean(g[r][msk[r]]) for r in range(n) if msk[r].any()]
                    gain[f"{bi}/{cn}"] = {"gain": float(np.nanmean(g[msk])) if msk.any() else None, "se": se(rows_g),
                                          "tokens": int(msk.sum())}
                msk = pm[None, :] & ~np.isnan(g)
                gain[f"{bi}/all"] = {"gain": float(np.nanmean(g[msk])), "se": se([np.nanmean(g[r][msk[r]]) for r in range(n)]),
                                     "tokens": int(msk.sum())}
            e["ce_gain"] = gain
            # share of the total CE gain at positions >= 4096 carried by each repeat category
            far_pos = (pos >= 4096)[None, :] & ~np.isnan(g)
            tot_g = g[far_pos].sum()
            e["gain_share_ge4k"] = {cn: float(g[far_pos & (rep == cc)].sum() / tot_g) if tot_g else None
                                    for cc, cn in enumerate(("novel", "near", "far_only"))}
        out["data"][l] = e
    json.dump(out, open(os.path.join(RES, "summary.json"), "w"))
    # ---- print
    P = list(m0["pyr"])
    print(f"rows: {[out['data'][l]['eval_size_rows'] for l in labs]} (⚠ eval_size < 500 rows; each row 32k tokens)")
    print(f"repeat categories of targets: novel / near / far-only = {[round(x, 3) for x in out['rep_share']]}")
    hdr = ("label", "loss", "pg0-6", "(sum)", "pg0-15", "head", "conc10@7/16/24/31", "conc30@24", "rho7~16", "rho7~24",
           "top10@7 holds G24", "rnd10", "nest3/7/16 G24", "own@0/7/16/24", "ctx@0/7/16/24", "amfar@7/15/31")
    print(" | ".join(hdr))
    for l in labs:
        e = out["data"][l]
        cm = np.array(e["contain_mass"])
        nest = np.array(e["pyr_nested"])
        sets, tg = list(m0["pyr_sets"]), list(m0["pyr_tgts"])
        conc = np.array(e["conc"])
        xc = np.array(e["xcorr"])
        fb = m0["full_blocks"]
        print(f"{l:12s} | {e['loss']:.3f} | {e['pgrad_0_6']:.3f} | {e['pgrad_0_6_summed']:.3f} | {e['pgrad_0_15']:.2f} | {e['pgrad_head']:.3f} | "
              f"{conc[7, 3]:.2f}/{conc[16, 3]:.2f}/{conc[24, 3]:.2f}/{conc[31, 3]:.2f} | {conc[24, 5]:.2f} | {xc[7, 16]:.2f} | {xc[7, 24]:.2f} | "
              f"{cm[P.index(7), P.index(24), 2]:.2f} | {nest[sets.index('random_10pct'), tg.index('g24')]:.2f} | "
              f"{nest[sets.index('nest_3_7_16'), tg.index('g24')]:.2f} | "
              f"{e['own_share'][0]:.2f}/{e['own_share'][7]:.2f}/{e['own_share'][16]:.2f}/{e['own_share'][24]:.2f} | "
              f"{e['ctx_share'][0]:.2f}/{e['ctx_share'][7]:.2f}/{e['ctx_share'][16]:.2f}/{e['ctx_share'][24]:.2f} | "
              f"{e['am_far_frac'][fb.index(7)]:.2f}/{e['am_far_frac'][fb.index(15)]:.2f}/{e['am_far_frac'][fb.index(31)]:.2f}")
    print("\nG^2 share by position bin (0-4k / 4-8k / 8-16k / 16-32k; token shares 0.125/0.125/0.25/0.5) at blocks 7 / 24:")
    for l in labs:
        ps = np.array(out["data"][l]["pos_share"])
        print(f"  {l:12s} b7 {' '.join(f'{v:.2f}' for v in ps[7])} | b24 {' '.join(f'{v:.2f}' for v in ps[24])} | "
              f"ctx share by pos @b16 {' '.join(f'{v:.2f}' for v in np.array(out['data'][l]['ctx_share_pos'])[16])}")
    print("\nCE gain vs base by position bin (all / novel / near / far-only targets):")
    for l in labs:
        g = out["data"][l].get("ce_gain")
        if not g:
            continue
        print(f"  {l:12s} " + " | ".join(
            f"{b}: {g[f'{b}/all']['gain']:+.3f} ({g[f'{b}/novel']['gain']:+.3f} / {g[f'{b}/near']['gain']:+.3f} / "
            f"{(g[f'{b}/far_only']['gain'] if g[f'{b}/far_only']['gain'] is not None else float('nan')):+.3f})"
            for b in range(len(pos_edges) - 1)) + f" | share of >=4k gain novel/near/far {out['data'][l]['gain_share_ge4k']}")
    print(f"\nwrote {RES}/summary.json")


if __name__ == "__main__":
    main()
