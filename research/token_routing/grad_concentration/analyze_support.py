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


def deq(z, key):
    return AG.deq(z, key).reshape(-1)


def sel_metrics(zs, s, pre):
    B = len(zs)
    V = torch.stack([torch.stack([deq(z, f"{pre}D"), deq(z, f"{pre}A_{s}"), deq(z, f"{pre}R_{s}")]) for z in zs])
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
            if s.startswith("q1_"):
                k = s[3:]
                o["fstar"] = AG.ms([b["q1"][k]["fstar"] for b in bst])
                o["keep_batch"] = AG.ms([b["q1"][k]["keep_frac"] for b in bst])
                o["attr_cos_at_fstar"] = AG.ms([b["q1"][k]["attr_cos"] for b in bst])
                o["curve_raw"] = np.mean([b["q1"][k]["curve_raw"] for b in bst], 0).tolist()
                if bst[0]["q1"][k].get("curve_adam"):
                    o["curve_adam"] = np.mean([b["q1"][k]["curve_adam"] for b in bst], 0).tolist()
            res["sels"][s] = o
            r_ = o["raw"]
            ad = o.get("adam", {})
            print(f"  {s:22s} keep {o.get('keep', [float('nan')])[0]:.4f} attr {r_['attr_cos'][0]:.3f} | routed "
                  f"{r_['routed_cos'][0]:.3f}±{r_['routed_cos'][1]:.3f} | exp {r_.get('exp_cos_routed', float('nan')):.3f}"
                  f" / adam {ad.get('exp_cos_routed', float('nan')):.3f} | norm {r_.get('exp_norm_ratio_routed', float('nan')):.2f}"
                  f" | CE {ce.get('all', {}).get('routed', [float('nan')])[0]:.3f} (dense "
                  f"{ce.get('all', {}).get('dense', [float('nan')])[0]:.3f})")
        res["dense"] = {"dense_cos_E": res["sels"][meta["sels"][0]]["raw"].get("dense_cos_E"),
                        "pair_cos_dense": res["sels"][meta["sels"][0]]["raw"].get("pair_cos_dense")}
        res["spearman_gate_en"] = {L: AG.ms([r["spearman_gate_en"][L] for r in rows if L in r["spearman_gate_en"]])
                                   for L in ("7", "16")}
        res["selfcheck"] = [b.get("selfcheck_0") for b in bst if "selfcheck_0" in b]
        res["grid"] = meta["grid"]
        allout[lab] = res
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(allout, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
