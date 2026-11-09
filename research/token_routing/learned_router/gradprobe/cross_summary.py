"""Tables for cross_score.py outputs: base-scored vs self-scored drop rules, all evaluated on the finetuned model.

    python3 cross_summary.py out/cross_setA_nq_8k_r64.json [...]

Per file: median (and mean +- SE) dCE vs keep ratio per rule; the smallest lossless keep ratio (median dCE <= tau,
walking down from the top ratio); paired per-row dCE difference vs a reference rule at every ratio and at the bar's
budget (@bar); gold AUROC. Stdlib + numpy only.
"""
import argparse
import json
import math

import numpy as np

RULES = ["self_a1_abs", "self_a0.5_abs", "base_a1_abs", "base_a0.75_abs", "base_a0.5_abs", "base_a0.25_abs",
         "base_a1_harm", "self_gold_ga", "base_gold_ga", "rand"]


def per_row(rows, name):
    """rule -> {rho: [dce per row]} (rand: mean over seeds per row)."""
    out = {}
    for r in rows:
        if name == "rand":
            by = {}
            for n, v in r["curves"].items():
                if n.startswith("rand_s"):
                    for q in v:
                        by.setdefault(round(q["rho"], 4), []).append(q["dce"])
            for k, v in by.items():
                out.setdefault(k, []).append(float(np.mean(v)))
        else:
            for q in r["curves"].get(name, []):
                out.setdefault(round(q["rho"], 4) if ("@bar" not in name and name != "bar") else "bar", []).append(q["dce"])
    return out


def min_budget(curve, tau):
    best = None
    for rho in sorted(curve, reverse=True):
        if np.median(curve[rho]) <= tau:
            best = rho
        else:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--ref", default="self_a0.5_abs")
    a = ap.parse_args()
    for f in a.files:
        d = json.load(open(f))
        rows = d["rows"]
        n = len(rows)
        print(f"\n=== {f}  ({d['task']}@{d['rung']}, eval_size={n}{'  ⚠ <500' if n < 500 else ''}; "
              f"CE_full eval mean {np.mean([r['ce_full'] for r in rows]):.4f}, score model {np.mean([r['ce_full_score_model'] for r in rows]):.3f})")
        curves = {nm: per_row(rows, nm) for nm in RULES}
        rhos = sorted({k for c in curves.values() for k in c}, reverse=True)
        print("median dCE   " + " ".join(f"{r:>7g}" for r in rhos) + f"   rho*(tau={a.tau})")
        for nm in RULES:
            c = curves[nm]
            if not c:
                continue
            print(f"{nm:15s}" + " ".join(f"{np.median(c[r]):+7.3f}" if r in c else "      -" for r in rhos)
                  + f"   {min_budget(c, a.tau)}")
        print("mean dCE (±SE)")
        for nm in RULES:
            c = curves[nm]
            if c:
                print(f"{nm:15s}" + " ".join(f"{np.mean(c[r]):+.2f}±{np.std(c[r], ddof=1) / math.sqrt(len(c[r])):.2f}" if r in c else "-"
                                             for r in rhos))
        ref = curves[a.ref]
        print(f"paired mean(rule - {a.ref}) ±SE, by rho (negative = rule better)")
        for nm in RULES:
            if nm in (a.ref, "rand") or not curves[nm]:
                continue
            c = curves[nm]
            cells = []
            for r in rhos:
                if r in c and r in ref and len(c[r]) == len(ref[r]):
                    dd = np.array(c[r]) - np.array(ref[r])
                    cells.append(f"{dd.mean():+.2f}±{dd.std(ddof=1) / math.sqrt(len(dd)):.2f}")
                else:
                    cells.append("-")
            print(f"{nm:15s}" + " ".join(cells))
        bar = per_row(rows, "bar")["bar"]
        refb = per_row(rows, a.ref + "@bar")["bar"]
        print(f"@bar budget (rho mean {np.mean([r['curves']['bar'][0]['rho'] for r in rows]):.3f}): bar dCE median {np.median(bar):+.3f}")
        for nm in RULES:
            if nm == "rand":
                continue
            b = per_row(rows, nm + "@bar").get("bar")
            if b:
                dd = np.array(b) - np.array(refb)
                print(f"  {nm + '@bar':18s} median {np.median(b):+.3f} mean {np.mean(b):+.3f}  vs ref {dd.mean():+.3f}±{dd.std(ddof=1) / math.sqrt(len(dd)):.3f}")
        au = {k: np.nanmean([r["auroc_gold"][k] for r in rows]) for k in rows[0]["auroc_gold"]}
        print("gold AUROC: " + " ".join(f"{k} {v:.3f}" for k, v in au.items() if "harm" not in k))
        sp = d.get("summary", {}).get("spearman_base_vs_self", {})
        if sp:
            print("spearman base~self: " + " ".join(f"{k} {v:.2f}" for k, v in sp.items() if "harm" not in k))


if __name__ == "__main__":
    main()
