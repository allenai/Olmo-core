"""Tables from support_probe.py sketches: per selection, attribution cos(A, D), true routed cos(R, D) per step, debiased
expected cos(E[R], E[D]) raw / Adam, expected norm ratio, routed answer CE per task, keep fraction, energy / gold share.

    python research/token_routing/grad_concentration/analyze_support.py --dir <save-dir> --labels f1952,recipe --out x.json
Expected cosines: analyze_gap.metrics_from_blocks (cross-batch terms only). Errors: SE over batches.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_gap as AG  # noqa: E402


def deq(z, key, g0=0):
    return AG.deq(z, key)[g0:].reshape(-1)  # (NG, m), group = 2 * block + part


def split_of(s):
    return 7 if s.startswith("q3_") else int(s.split("_L")[1].split("_")[0])


def sel_metrics(zs, s, pre, g0=0):
    """g0 = first group kept: 0 = whole model, 2 * L = only the weights of blocks >= L."""
    B = len(zs)
    V = torch.stack([torch.stack([deq(z, f"{pre}D", g0), deq(z, f"{pre}A_{s}", g0), deq(z, f"{pre}R_{s}", g0)])
                     for z in zs])
    V = V.reshape(B * 3, -1).to(AG.dev()).double()
    G = (V @ V.T).cpu().numpy()
    M = {}
    for xi, X in enumerate("DAR"):
        for yi, Y in enumerate("DAR"):
            M[(X, Y)] = G[np.ix_([b * 3 + xi for b in range(B)], [b * 3 + yi for b in range(B)])]
    return AG.metrics_from_blocks(M, B)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--labels", default="f1952,recipe")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    allout = {}
    for lab in a.labels.split(","):
        d = os.path.join(a.dir, lab)
        files = sorted(glob.glob(os.path.join(d, "b*.npz")))
        metas = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "meta_shard*.json")))]
        if not files:
            print(f"{lab}: no batches")
            continue
        meta = metas[0]
        rows = [r for m in metas for r in m["rows"]]
        bst = [b for m in metas for b in m["batch_stats"]]
        done = {b["batch"] for b in bst}
        files = [f for f in files if int(os.path.basename(f)[1:4]) in done]
        zs = [np.load(f) for f in files]
        B = len(zs)
        tasks = meta["tasks"]
        res = {"label": lab, "n_batches": B, "eval_size_rows": len(rows),
               "small_flag": f"eval_size={len(rows)} rows < 500" if len(rows) < 500 else "", "sels": {}}
        print(f"== {lab}: {B} batches, {len(rows)} rows")
        has_adam = "adam_tot_D" in zs[0]
        for s in meta["sels"]:
            o = {"raw": sel_metrics(zs, s, "tot_")}
            if split_of(s) > 0:
                o["raw_dn"] = sel_metrics(zs, s, "tot_", 2 * split_of(s))
            if has_adam:
                o["adam"] = sel_metrics(zs, s, "adam_tot_")
            ce = {}
            for t in tasks + ["all"]:
                rr = [r for r in rows if (t == "all" or r["task"] == t) and s in r["ce_routed"]]
                if rr:
                    ce[t] = {"dense": AG.ms([r["ce_dense"] / r["n_answer"] for r in rr]),
                             "routed": AG.ms([r["ce_routed"][s] / r["n_answer"] for r in rr])}
            o["ce"] = ce
            for key in ("keep", "energy_share", "gold_share", "overlap_en"):
                v = [r[key][s] for r in rows if s in r.get(key, {})]
                if v:
                    o[key] = AG.ms(v)
            gk = ("en_" + s[3:]) if s.startswith("q1_") else (s[3:] if s.startswith("jt_") else None)
            if gk is not None:
                o["fstar"] = AG.ms([b["grid"][gk]["fstar"] for b in bst])
                o["keep_batch"] = AG.ms([b["grid"][gk]["keep_frac"] for b in bst])
                o["attr_cos_at_fstar"] = AG.ms([b["grid"][gk]["attr_cos"] for b in bst])
                for sc in ("wm", "dn"):
                    if f"attr_cos_{sc}" in bst[0]["grid"][gk]:
                        o[f"attr_cos_{sc}_at_fstar"] = AG.ms([b["grid"][gk][f"attr_cos_{sc}"] for b in bst])
                if bst[0]["grid"][gk].get("g95_frac") is not None:
                    o["g95_frac"] = AG.ms([b["grid"][gk]["g95_frac"] for b in bst])
            o["step"] = {k_: AG.ms([b["step"][s][k_] for b in bst]) for k_ in ("keep", "ce_routed")}
            for sn in ("raw", "raw_dn"):
                if sn in bst[0]["step"][s]:
                    for k_ in ("routed_cos", "attr_cos"):
                        o["step"][f"{sn}_{k_}"] = AG.ms([b["step"][s][sn][k_] for b in bst])
            res["sels"][s] = o
            r_ = o["raw"]
            ad = o.get("adam", {})
            print(f"  {s:22s} keep {o.get('keep', [float('nan')])[0]:.4f} attr {r_['attr_cos'][0]:.3f} | routed "
                  f"{r_['routed_cos'][0]:.3f}±{r_['routed_cos'][1]:.3f} | exp {r_.get('exp_cos_routed', float('nan')):.3f}"
                  f" / adam {ad.get('exp_cos_routed', float('nan')):.3f} | norm {r_.get('exp_norm_ratio_routed', float('nan')):.2f}"
                  + (f" | dn: attr {o['raw_dn']['attr_cos'][0]:.3f} routed {o['raw_dn']['routed_cos'][0]:.3f}"
                     f" exp {o['raw_dn'].get('exp_cos_routed', float('nan')):.3f}" if "raw_dn" in o else "")
                  + f" | CE {ce.get('all', {}).get('routed', [float('nan')])[0]:.3f} (dense "
                  f"{ce.get('all', {}).get('dense', [float('nan')])[0]:.3f})")
        res["dense"] = {"dense_cos_E": res["sels"][meta["sels"][0]]["raw"].get("dense_cos_E"),
                        "pair_cos_dense": res["sels"][meta["sels"][0]]["raw"].get("pair_cos_dense")}
        sk = sorted({k_ for r in rows for k_ in r["spearman_gate_en"]})
        res["spearman_gate_en"] = {k_: AG.ms([r["spearman_gate_en"][k_] for r in rows if k_ in r["spearman_gate_en"]])
                                   for k_ in sk}
        grid = meta["grid"]

        def first(c, thr):
            return next((grid[j] for j, v in enumerate(c) if v >= thr), 1.0)
        res["conc"] = {}
        for key in bst[0].get("conc", {}):
            o = {}
            for kind in ("plain", "sq"):
                cs = [b["conc"][key][kind] for b in bst]
                o[kind] = {"curve": np.mean(cs, 0).tolist(),
                           **{f"f{int(t * 100)}": AG.ms([first(c, t) for c in cs]) for t in (0.7, 0.9, 0.95)}}
            res["conc"][key] = o
        res["grid_curves"] = {key: {"raw": np.mean([b["grid"][key]["curve_raw"] for b in bst], 0).tolist(),
                                    "f95": AG.ms([b["grid"][key]["fstar"] for b in bst]),
                                    "keep": AG.ms([b["grid"][key]["keep_frac"] for b in bst])}
                              for key in bst[0]["grid"]}
        res["jaccard"] = {key: np.mean([b["jaccard"][key] for b in bst], 0).tolist() for key in bst[0].get("jaccard", {})}
        res["gs_ks"] = meta.get("gs_ks")
        res["ce_dense_step"] = AG.ms([b["ce_dense"] for b in bst])
        res["selfcheck"] = [b.get("selfcheck_0") for b in bst if "selfcheck_0" in b]
        res["grid"] = meta["grid"]
        allout[lab] = res
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(allout, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
