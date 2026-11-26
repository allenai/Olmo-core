"""Summaries of tokgrad_probe.py outputs: concentration per format, alignment of gold drop / router with the whole-row
gradient mass, FLOP ratios. Mean over rows (draws averaged within a row first), SE over rows.

    python analyze_tokgrad.py <results dir> [...] --out summary.json
"""
import argparse
import glob
import json
import os

import numpy as np

SPLIT = 7
FR = (0.5, 0.9, 0.99)


def n_for(e):
    s = np.sort(e.astype(np.float64))[::-1]
    if s.size == 0 or s.sum() <= 0:
        return [np.nan] * len(FR)
    c = np.cumsum(s) / s.sum()
    return [float((np.searchsorted(c, q) + 1) / s.size) for q in FR]


def ms(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"), int(len(v))]


def analyze(d):
    meta = json.load(open(f"{d}/rows.json"))
    acc = {}

    def put(k, v):
        acc.setdefault(k, []).append(v)

    for rec in meta["rows"]:
        z = np.load(f"{d}/{rec['set']}_r{rec['row']:03d}.npz")
        body, gold = z["body"], z["gold_tok"]
        A = z["A"].astype(np.float64)
        sA = A.sum(0)
        nb = int(body.sum())
        # A: concentration over body tokens (body-normalised), all tokens, and per block
        for q, v in zip(FR, n_for(sA[body])):
            put(f"A|body|n{q}", v)
        for q, v in zip(FR, n_for(sA)):
            put(f"A|all|n{q}", v)
        put("A|body_share", sA[body].sum() / sA.sum())
        put("A|gold_share", sA[gold].sum() / sA.sum())
        put("A|gold_tok_share_of_body", gold.sum() / nb)
        for b in range(A.shape[0]):
            for q, v in zip(FR, n_for(A[b][body])):
                put(f"A|blk{b + SPLIT}|body|n{q}", v)
        put("ceA", rec["ceA"])
        ds = {}
        for di, dr in enumerate(rec["draws"]):
            kp = z[f"kp{di}"]
            keep = np.zeros(len(sA), bool)
            keep[kp] = True
            kb = keep & body
            x = {}
            x["drop_r"] = dr["r"]
            x["drop|tok_share_body"] = kb.sum() / nb
            x["drop|tok_share_all"] = keep.mean()
            x["drop|A_mass_kept"] = sA[keep].sum() / sA.sum()
            x["drop|A_body_mass_kept"] = sA[kb].sum() / sA[body].sum()
            x["drop|A_body_mass_kept_distractors_only"] = sA[kb & ~gold].sum() / max(sA[body & ~gold].sum(), 1e-300)
            # oracle: the same number of body tokens chosen by A's own mass
            k = int(kb.sum())
            x["drop|A_body_mass_oracle_same_tokens"] = np.sort(sA[body])[::-1][:k].sum() / sA[body].sum()
            x["flopB/A"] = dr["flopsB"] / rec["flopsA"]
            B = z[f"B{di}"].astype(np.float64)
            sB = B.sum(0)
            for q, v in zip(FR, n_for(sB[kb])):
                x[f"B|body_present|n{q}"] = v
                x[f"B|body_of_orig|n{q}"] = v * kb.sum() / nb
            for q, v in zip(FR, n_for(sB[keep])):
                x[f"B|all_present|n{q}"] = v
            x["ceB"] = dr["ceB"]
            if f"C{di}" in z.files:
                C = z[f"C{di}"].astype(np.float64)
                k7, k16 = z[f"k7_{di}"], z[f"k16_{di}"]
                sC = C.sum(0)
                b7 = k7 & body
                for q, v in zip(FR, n_for(sC[b7])):  # tokens that entered blocks 7+ (pooled over blocks 7-31)
                    x[f"C|body_present7|n{q}"] = v
                    x[f"C|body_of_orig|n{q}"] = v * b7.sum() / nb
                for q, v in zip(FR, n_for(sC[k7])):
                    x[f"C|all_present7|n{q}"] = v
                lo, hi = slice(0, 16 - SPLIT), slice(16 - SPLIT, None)
                b16 = k16 & body
                for q, v in zip(FR, n_for(C[hi].sum(0)[b16])):
                    x[f"C|blk16+|body_present16|n{q}"] = v
                for q, v in zip(FR, n_for(C[lo].sum(0)[b7])):
                    x[f"C|blk7-15|body_present7|n{q}"] = v
                x["C|tok7_share_body"] = b7.sum() / nb
                x["C|tok16_share_body"] = b16.sum() / nb
                x["C|tok7_share_of_dropped_body"] = b7.sum() / max(kb.sum(), 1)
                x["C|tok16_share_of_dropped_body"] = b16.sum() / max(kb.sum(), 1)
                # router alignment: whole-row (A) and post-drop (B) mass at blocks 7-15 / 16-31 on router-kept tokens
                a_lo, a_hi = A[lo].sum(0), A[hi].sum(0)
                x["router|A_mass7-15_on_kept7"] = a_lo[k7].sum() / a_lo.sum()
                x["router|A_mass16-31_on_kept16"] = a_hi[k16].sum() / a_hi.sum()
                x["router|A_body_mass7-15_on_kept7"] = a_lo[b7].sum() / a_lo[body].sum()
                x["router|A_body_mass16-31_on_kept16"] = a_hi[b16].sum() / a_hi[body].sum()
                bl, bh = B[lo].sum(0), B[hi].sum(0)
                x["router|B_body_mass7-15_on_kept7"] = bl[b7].sum() / bl[kb].sum()
                x["router|B_body_mass16-31_on_kept16"] = bh[b16].sum() / bh[kb].sum()
                x["router|B_body_mass7-15_oracle_same_tokens"] = np.sort(bl[kb])[::-1][:int(b7.sum())].sum() / bl[kb].sum()
                x["router|B_body_mass16-31_oracle_same_tokens"] = np.sort(bh[kb])[::-1][:int(b16.sum())].sum() / bh[kb].sum()
                x["router|B_body_mass7-15_random_same_tokens"] = b7.sum() / kb.sum()
                x["flopC/A"] = dr["flopsC"] / rec["flopsA"]
                x["ceC"] = dr["ceC"]
                x["C|ans_lost"] = dr["ans_lost"]
            for k_, v in x.items():
                ds.setdefault(k_, []).append(v)
        for k_, v in ds.items():
            put(k_, float(np.mean(v)))
    out = {k: ms(v) for k, v in acc.items()}
    out["_meta"] = dict(label=meta["label"], router_src=meta.get("router_src"), rows=len(meta["rows"]),
                        draws=len(meta["rows"][0]["draws"]), T_mean=float(np.mean([r["T"] for r in meta["rows"]])))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {os.path.basename(d.rstrip("/")): analyze(d) for d in a.dirs if os.path.exists(f"{d}/rows.json")}
    json.dump(res, open(a.out, "w"), indent=1)
    for k, v in res.items():
        print(k, v["_meta"])
