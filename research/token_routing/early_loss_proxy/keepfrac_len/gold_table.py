"""Gold-drop arms (grJ, grJc) vs dft / k20 / k40 at 8k, 32k, 64k from kl_summary.json, per seed.

    python3 gold_table.py --summary kl_summary.json [--md out.md]
"""
from __future__ import annotations

import argparse
import json

LS = ["8k", "32k", "64k"]
T = ["contradiction", "outlier", "qdmatch_nq"]


def f(x, d=2):
    return "not reached" if x is None else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    S = json.load(open(a.summary))
    out = []
    out.append("| length | arm | seed | FLOPs/step vs dense | x dft @ mid CE (mean) | x dft @ end CE (mean) | x @ mid: "
               + " / ".join(T) + " | x @ end: " + " / ".join(T) + " | final CE mean | final CE " + " / ".join(T) + " |")
    out.append("|" + "---|" * 10)
    for L in LS:
        if L not in S:
            continue
        R = S[L]
        for arm in ["dft", "k40", "k20", "grJ", "grJc"]:
            r = R["rows"].get(arm)
            if r is None:
                continue
            for e in r["seeds"]:
                out.append(f"| {L} | {arm} | {e['seed']} | {e['x_dense_step']:.3f} | {f(e['x_mid'])} | {f(e['x_deep'])} | "
                           + " / ".join(f(e['task_x_mid'][t]) for t in T) + " | "
                           + " / ".join(f(e['task_x_deep'][t]) for t in T) + f" | {e['final_ce']:.4f} | "
                           + " / ".join(f"{e['task_final_ce'][t]:.3f}" for t in T) + " |")
        out.append(f"| {L} | levels | | | mid (dft step {R['level_step_mid']}) {R['levels']['mid']:.4f} | end {R['levels']['deep']:.4f} | "
                   + " / ".join(f"{R['task_levels'][t]['mid']:.3f}" for t in T) + " | "
                   + " / ".join(f"{R['task_levels'][t]['deep']:.3f}" for t in T) + " | | |")
    s = "\n".join(out) + "\n"
    print(s)
    if a.md:
        open(a.md, "w").write(s)


if __name__ == "__main__":
    main()
