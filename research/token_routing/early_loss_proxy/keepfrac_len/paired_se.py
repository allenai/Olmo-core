"""Paired per-row held-out CE differences between arms at matched steps (same held-out rows), with SE.

    python3 paired_se.py --root <score_all dest> --len 32k --pairs k40:k20,k70:k40 [--step 225] [--prefix kl]
Also prints seed-to-seed |diff| of the mean CE for arms with two seeds (the run-to-run noise floor).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def ce_at(path, step):
    d = json.load(open(path))
    pr = {q["step"]: q for q in d["probes"] if q.get("ce")}
    s = step if step in pr else max(pr)
    return np.array(pr[s]["ce"]), [r["task"] for r in d["rows"]], s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--len", required=True)
    ap.add_argument("--pairs", default="k40:k20")
    ap.add_argument("--step", type=int, default=225)
    ap.add_argument("--prefix", default="kl")
    a = ap.parse_args()
    T = f"{a.prefix}{a.len}"
    files = {os.path.basename(p)[len(f"elp-{T}-"):-5]: p for p in glob.glob(f"{a.root}/{T}/elp-{T}-*-sd*.json")}
    for pair in a.pairs.split(","):
        x, y = pair.split(":")
        for sd in ("sd0", "sd1"):
            if f"{x}-{sd}" in files and f"{y}-{sd}" in files:
                cx, tasks, s1 = ce_at(files[f"{x}-{sd}"], a.step)
                cy, _, s2 = ce_at(files[f"{y}-{sd}"], a.step)
                d = cx - cy
                per = {t: (d[[i for i, u in enumerate(tasks) if u == t]].mean()) for t in sorted(set(tasks))}
                print(f"{a.len} {x}-{y} {sd} step {s1}/{s2}: mean diff {d.mean():+.4f} +- {d.std(ddof=1) / len(d) ** .5:.4f} "
                      f"(eval_size {len(d)}) | " + " ".join(f"{t[:6]} {v:+.4f}" for t, v in per.items()))
    for k in sorted({f.rsplit("-sd", 1)[0] for f in files}):
        if f"{k}-sd0" in files and f"{k}-sd1" in files:
            c0, _, _ = ce_at(files[f"{k}-sd0"], a.step)
            c1, _, _ = ce_at(files[f"{k}-sd1"], a.step)
            print(f"{a.len} {k} seed0 vs seed1 mean CE {c0.mean():.4f} vs {c1.mean():.4f} (|diff| {abs(c0.mean() - c1.mean()):.4f})")


if __name__ == "__main__":
    main()
