"""Markdown tables (lengths as columns) from analyze_sweep.py summaries, for the record.

    python make_tables.py data/sweep_mt11_dense_f1.json [data/sweep_mt4_dense_f7808.json ...]
"""
from __future__ import annotations

import json
import sys

RULES = [("down7_body", "Global oracle (16-row step)"), ("rowq7_body", "Per-row quota"),
         ("docq7_body", "Per-document quota"), ("rand7_body", "Random")]


def flag(n):
    return "" if n >= 500 else f" ⚠"


def table(S, t):
    runs = S["runs"]
    head = "| rule | " + " | ".join(f"{s['label']} (T≈{s['T_mean'] / 1000:.0f}k)" for s in runs) + " |"
    out = [head, "|---|" + "---|" * len(runs)]
    out.append("| eval_size rows (steps) | " + " | ".join(
        f"{s['eval_size_rows']} ({s['steps']}){flag(s['eval_size_rows'])}" for s in runs) + " |")
    for r, name in RULES:
        cells = []
        for s in runs:
            e = s["rules"][r]
            lo, hi = e[f"k_cos{t}_ci"]
            cells.append(f"{100 * e[f'k_cos{t}']:.2f}% [{100 * lo:.2f}, {100 * hi:.2f}]")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    for r, name in RULES[:3]:
        out.append(f"| {name}: tokens per row | " + " | ".join(
            f"{s['rules'][r][f'tokens_row_cos{t}']:,.0f}" for s in runs) + " |")
    out.append("| Row vs its own gradient (mean / median) | " + " | ".join(
        f"{100 * s[f'rowself_k_cos{t}']:.2f}% / {100 * s[f'rowself_k_cos{t}_median']:.2f}%" for s in runs) + " |")
    import numpy as _np
    out.append("| Row vs its own gradient: geometric mean over rows with k > 0 | " + " | ".join(
        f"{100 * _np.exp(_np.mean(_np.log([k for k in s[f'_rows_rowself_cos{t}'] if k > 0]))):.2f}%" for s in runs) + " |")
    out.append("| Row vs its own gradient: tokens per row | " + " | ".join(
        f"{s[f'rowself_tokens_cos{t}']:,.0f}" for s in runs) + " |")
    out.append("| body tokens per row | " + " | ".join(f"{s['body_tokens_per_step'] / 16:,.0f}" for s in runs) + " |")
    out.append("| body tokens' share of the down7 energy score | " + " | ".join(
        f"{100 * s.get('body_energy_share_mean', float('nan')):.0f}%" for s in runs) + " |")
    out.append("| rows needing 0 body tokens (row vs own, this cos) | " + " | ".join(
        f"{100 * sum(1 for k in s[f'_rows_rowself_cos{t}'] if k == 0) / len(s[f'_rows_rowself_cos{t}']):.0f}%" for s in runs) + " |")
    out.append("| answer CE (mean) | " + " | ".join(f"{s['ce_mean']:.3f}" for s in runs) + " |")
    return "\n".join(out)


def task_table(S, t):
    runs = S["runs"]
    tasks = sorted(runs[0]["tasks"])
    out = ["| task | " + " | ".join(s["label"] for s in runs) + " |", "|---|" + "---|" * len(runs)]
    for tk in tasks:
        out.append(f"| {tk}: row vs own gradient | " + " | ".join(
            f"{100 * s['tasks'][tk][f'rowself_k_cos{t}']:.2f}% ({s['tasks'][tk][f'rowself_tokens_cos{t}']:,.0f} tok)"
            for s in runs) + " |")
    for tk in tasks:
        out.append(f"| {tk}: documents per row (CE) | " + " | ".join(
            f"{s['tasks'][tk]['n_docs_mean']:,.0f} ({s['tasks'][tk]['ce_mean']:.3f})" for s in runs) + " |")
    out.append("| rows per task | " + " | ".join(str(s["tasks"][tasks[0]]["rows"]) for s in runs) + " |")
    return "\n".join(out)


def fits(S):
    out = ["| quantity | slope of log(share) vs log(T) [95% CI] | implied tokens ∝ T^ |", "|---|---|---|"]
    for k, v in S["fits"].items():
        if v:
            rng_ = f"{v['lengths'][0] / 1000:.0f}k-{v['lengths'][-1] / 1000:.0f}k"
            out.append(f"| {k} ({rng_}) | {v['slope']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | {1 + v['slope']:.2f} |")
    for k, v in S.get("row_level_regression", {}).items():
        if v:
            out.append(f"| row-level log k ~ log T + log CE, {k} | {v['slope_logT']:.3f} [{v['slope_logT_ci'][0]:.3f}, "
                       f"{v['slope_logT_ci'][1]:.3f}] (log CE coef {v['slope_logCE']:.2f}) | {1 + v['slope_logT']:.2f} |")
            nc = v["all_rows_no_ce"]
            out.append(f"| row-level log k ~ log T (no CE term, rows with k > 0, n={nc['n_rows']}), {k} | "
                       f"{nc['slope_logT']:.3f} [{nc['ci'][0]:.3f}, {nc['ci'][1]:.3f}] | {1 + nc['slope_logT']:.2f} |")
            sr = v["solved_rows"]
            out.append(f"| row-level, solved rows only (CE<0.01, n={sr['n_rows']}), {k} | {sr['slope_logT']:.3f} "
                       f"[{sr['ci'][0]:.3f}, {sr['ci'][1]:.3f}] | {1 + sr['slope_logT']:.2f} |")
            for tk, x in v["per_task"].items():
                out.append(f"| row-level, {tk}, {k} | {x['slope_logT']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}] | "
                           f"{1 + x['slope_logT']:.2f} |")
    return "\n".join(out)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        S = json.load(open(p))
        print(f"## {p}\n\n### cos 0.95\n\n{table(S, 0.95)}\n\n### cos 0.9\n\n{table(S, 0.9)}\n\n"
              f"### per task, cos 0.95\n\n{task_table(S, 0.95)}\n\n### fits\n\n{fits(S)}\n")
