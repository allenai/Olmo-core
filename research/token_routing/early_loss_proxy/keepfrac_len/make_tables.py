"""Markdown + HTML tables for the record and the hub block from summarize_kl.py's kl_summary.json.

    python3 make_tables.py --summary kl_summary.json --md tables.md --html hub_keeplen_block.html
"""
from __future__ import annotations

import argparse
import html
import json

import numpy as np

LENS = ["2k", "4k", "8k", "16k", "32k", "64k"]
ARMS = ["dft", "ref", "k70", "k40", "k20", "k10", "k5", "k2p5", "k1p25", "grJ", "grJc"]
NAME = {"dft": "dense full FT", "ref": "dense LoRA", "k70": "70% / 24.5%", "k40": "40% / 14%", "k20": "20% / 7% (recipe)",
        "k10": "10% / 3.5%", "k5": "5% / 1.75%", "k2p5": "2.5% / 0.9%", "k1p25": "1.25% / 0.45%",
        "grJ": "grJ (gold-aware stochastic keep)", "grJc": "grJc (grJ, doc drop on contradiction only)"}


def fx(v):
    return "not reached" if v is None else f"{v:.2f}"


def cell(r, key):
    xs = [e[key] for e in r["seeds"]]
    if len(xs) == 1:
        return fx(xs[0])
    if any(x is None for x in xs):
        return " / ".join(fx(x) for x in xs)
    return f"{np.mean(xs):.2f} ({min(xs):.2f}-{max(xs):.2f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--md", default="")
    ap.add_argument("--html", default="")
    a = ap.parse_args()
    S = json.load(open(a.summary))
    md = []
    for L in [l for l in LENS if l in S]:
        R = S[L]
        tasks = sorted(R["task_levels"])
        md.append(f"\n**{L}** (eval_size {R['eval_size']} ⚠ < 500, 48 rows/task; matched level = dft's mean held-out CE at step "
                  f"{R['level_step_mid']} = {R['levels']['mid']:.4f}; end of epoch {R['levels']['deep']:.4f})\n")
        md.append("| arm | seeds | FLOPs/step vs dense | ΔCE vs recipe @ matched FLOPs | x dft @ mid CE | x dft @ end CE | worst task x @ mid (task) | "
                  + " | ".join(f"{t} x @ mid" for t in tasks) + " | final CE | final CE " + " / ".join(t[:6] for t in tasks) + " |")
        md.append("|" + "---|" * (8 + len(tasks) + 1))
        for arm in [x for x in ARMS if x in R["rows"]]:
            r = R["rows"][arm]
            e0 = r["seeds"][0]
            tx = " | ".join(" / ".join(fx(e["task_x_mid"][t]) for e in r["seeds"]) for t in tasks)
            wt = " / ".join(f"{fx(e['worst_x_mid'])} ({e['worst_task_mid']})" for e in r["seeds"])
            fce = " / ".join(f"{e['final_ce']:.4f}" for e in r["seeds"])
            tce = " ; ".join(" / ".join(f"{e['task_final_ce'][t]:.3f}" for t in tasks) for e in r["seeds"])
            dce = " / ".join("-" if e.get("dce_vs_k20") is None else f"{e['dce_vs_k20']:+.4f}" for e in r["seeds"])
            md.append(f"| {NAME.get(arm, arm)} | {','.join(str(e['seed']) for e in r['seeds'])} | {e0['x_dense_step']:.3f} | {dce} | "
                      f"{cell(r, 'x_mid')} | {cell(r, 'x_deep')} | {wt} | {tx} | {fce} | {tce} |")
    if a.md:
        open(a.md, "w").write("\n".join(md) + "\n")
    if a.html:
        # hub "Keep vs length" block: one row per arm, one column per length. Cell = ΔCE vs the recipe at matched
        # training FLOPs (per seed) and, below it, x dft at dft's mid-epoch CE (seed mean) with the worst task.
        Ls = [l for l in LENS if l in S]
        rows = []
        for arm in ARMS:
            if arm in ("dft",) or not any(arm in S[L]["rows"] for L in Ls):
                continue
            tds = []
            for L in Ls:
                r = S[L]["rows"].get(arm)
                if r is None:
                    tds.append('<td class="na">-</td>')
                    continue
                d = " / ".join("-" if e.get("dce_vs_k20") is None else f"{e['dce_vs_k20']:+.3f}" for e in r["seeds"])
                ds = [e.get("dce_vs_k20") for e in r["seeds"] if e.get("dce_vs_k20") is not None]
                cls = "" if arm == "k20" or not ds else (" good" if max(ds) < -0.005 else (" bad" if min(ds) > 0.02 else ""))
                w = " / ".join(fx(e["worst_x_mid"]) for e in r["seeds"])
                tds.append(f'<td class="c{cls}"><b>{html.escape(d)}</b><br><small>{html.escape(cell(r, "x_mid"))}x; worst '
                           f'{html.escape(w)}</small></td>')
            rows.append(f"<tr><th>{html.escape(NAME.get(arm, arm))}</th>{''.join(tds)}</tr>")
        head = "".join(f"<th>{L}</th>" for L in Ls)
        css = (".keeplen{border-collapse:collapse;font-size:12px}.keeplen th,.keeplen td{border:1px solid var(--line,#ccc);"
               "padding:3px 6px;text-align:center;vertical-align:top}.keeplen tbody th{text-align:left;font-weight:500}"
               ".keeplen td.good{background:var(--good-bg,rgba(40,160,80,.15))}.keeplen td.bad{background:var(--bad-bg,"
               "rgba(220,60,60,.13))}.keeplen small{color:var(--muted,#777)}")
        notes = ("<p class=\"keeplen-note\"><small>Cell: held-out CE minus the 20%/7% recipe's at matched training FLOPs, "
                 "per seed (negative = better; green &lt; -0.005 on every seed, red &gt; +0.02); below: FLOP speedup vs "
                 "dense full fine-tune at dft's mid-epoch held-out CE (seed mean; 'not reached' = never gets there) and "
                 "the worst task's speedup. Early dense-loss proxy, one epoch (225 steps), eval_size 144 per length "
                 "(48 rows/task, below 500), 2k/4k saturated (CE ~0.01). Rank only: the proxy cannot resolve gaps "
                 "under ~0.03-0.05 CE.</small></p>")
        open(a.html, "w").write(
            f"<style>{css}</style>\n<div class=\"keeplen-block\">\n<table class=\"keeplen\">\n<thead><tr><th>arm "
            f"(keep cut 1 / cut 2)</th>{head}</tr></thead>\n<tbody>\n" + "\n".join(rows) + f"\n</tbody>\n</table>\n{notes}\n</div>\n")

if __name__ == "__main__":
    main()
