"""Tables from grad_importance.py JSONs: per task, mean dCE +- SE per (rule, rho), scoring cost (passes), gold kept,
and PAIRED per-row differences of every rule against a reference rule (default iter_abs_c).

    python research/token_routing/learned_router/gradprobe/compare.py out/nq_8k.json out/outlier_8k.json ... [--ref iter_abs_c]

Stdlib + numpy only (login-node safe). Prints markdown.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

import numpy as np

SHOW = ["gold_rand", "gold_ga", "grad_abs", "iter2_abs", "iter3_abs", "iter_abs_c", "iter_abs", "ig1_abs", "ig1", "ig2_abs", "ig2", "ig3_abs", "ig3",
        "ig4_abs", "ig4", "ig8_abs", "ig8", "ig_abs", "ig", "grad", "gxe", "rand"]


def per_rule(rows):
    """{rule: {rho: [(row_i, point)]}} with the rand seeds pooled per row (mean over seeds)."""
    out = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        rand = defaultdict(list)
        for n, pts in r["curves"].items():
            for q in pts:
                if n.startswith("rand_s"):
                    rand[round(q["rho"], 4)].append(q)
                elif n not in ("bar", "gold_only"):
                    out[n][round(q["rho"], 4)].append((i, q))
        for rho, qs in rand.items():
            out["rand"][rho].append((i, {"dce": float(np.mean([q["dce"] for q in qs])), "passes": 0,
                                         "gold_kept": float(np.mean([q["gold_kept"] for q in qs]))}))
    return out


def ms(v):
    v = np.asarray(v, dtype=np.float64)
    return float(v.mean()), (float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--ref", default="iter_abs_c")
    a = ap.parse_args()
    for path in a.jsons:
        d = json.load(open(path))
        rows = d["rows"]
        R = per_rule(rows)
        rhos = sorted({rho for n in R for rho in R[n]}, reverse=True)
        n_rows = len(rows)
        ce_full = ms([r["ce_full"] for r in rows])
        print(f"\n## {d['task']} @ {d['rung']}  ⚠ eval_size={n_rows}  (CE_full {ce_full[0]:.4f} ± {ce_full[1]:.4f})\n")
        refs = []
        for n in ("bar", "gold_only"):
            v = [r["curves"][n][0] for r in rows if n in r["curves"]]
            if v:
                m, se = ms([q["dce"] for q in v])
                refs.append(f"{n}: rho {np.mean([q['rho'] for q in v]):.3f}, dCE {m:+.4f} ± {se:.4f}")
        print("references — " + "; ".join(refs) + "\n")
        print("mean dCE ± SE (passes) [gold kept]\n")
        rules = [n for n in SHOW if n in R] + sorted(n for n in R if n not in SHOW)
        print("| rule | " + " | ".join(f"ρ={r:g}" for r in rhos) + " |")
        print("|---|" + "---|" * len(rhos))
        for n in rules:
            cells = []
            for rho in rhos:
                pts = [q for _, q in R[n].get(rho, [])]
                if not pts:
                    cells.append("")
                    continue
                m, se = ms([q["dce"] for q in pts])
                cells.append(f"{m:+.3f} ± {se:.3f} ({int(np.mean([q.get('passes', 0) for q in pts]))}) [{np.nanmean([q['gold_kept'] for q in pts]):.2f}]")
            print(f"| {n} | " + " | ".join(cells) + " |")
        if a.ref in R:
            print(f"\npaired dCE(rule) − dCE({a.ref}), mean ± SE over rows (negative = rule better)\n")
            print("| rule | " + " | ".join(f"ρ={r:g}" for r in rhos) + " |")
            print("|---|" + "---|" * len(rhos))
            for n in rules:
                if n == a.ref:
                    continue
                cells = []
                for rho in rhos:
                    ref = dict(R[a.ref].get(rho, []))
                    diffs = [q["dce"] - ref[i]["dce"] for i, q in R[n].get(rho, []) if i in ref]
                    if not diffs:
                        cells.append("")
                        continue
                    m, se = ms(diffs)
                    cells.append(f"{m:+.3f} ± {se:.3f}")
                print(f"| {n} | " + " | ".join(cells) + " |")
        au = defaultdict(list)
        for r in rows:
            for k, v in r["auroc_gold"].items():
                if k != "gold_token_frac" and isinstance(v, float) and not math.isnan(v):
                    au[k].append(v)
        print("\ngold-token AUROC: " + ", ".join(f"{k} {np.mean(v):.3f}" for k, v in au.items()))
        ck = [r["checks"]["ig_completeness"] for r in rows]
        print(f"IG completeness (sum harm_ig / actual drop-all): {np.mean([c['sum_harm_ig'] for c in ck]):+.3f} / "
              f"{np.mean([c['actual_drop_all'] for c in ck]):+.3f};  soft-vs-hard max |diff| "
              f"{max(abs(r['checks']['soft_binary_vs_hard_rho0.3']) for r in rows):.1e}")


if __name__ == "__main__":
    main()
