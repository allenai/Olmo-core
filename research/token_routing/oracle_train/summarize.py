"""Summarise an oracle run's ``oracle_meta/steps.jsonl``: per-step table (every N-th) and windowed means.

    python research/token_routing/oracle_train/summarize.py <run save folder or steps.jsonl> [--every 25] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "oracle_meta", "steps.jsonl") if os.path.isdir(os.path.join(path, "oracle_meta")) \
            else os.path.join(path, "steps.jsonl")
    return [json.loads(line) for line in open(path) if line.strip()]


def row(r):
    g = r["grad"]
    out = {"step": r["step"], "k_s0": r["chosen_k"][0], "k_s1": r["chosen_k"][1],
           "keep_s0": r["keep_frac_body"][0], "keep_s1": r["keep_frac_body"][1],
           "attr_s0": r["attr_cos_at_k"][0], "attr_s1": r["attr_cos_at_k"][1],
           "gcos_all": g["all"]["cos"], "gcos_adam_all": g["all"]["cos_adam"],
           "ce_dense": r["ce_dense"], "ce_routed": r["ce_routed"],
           "routed_over_dense": r["flops"]["routed_over_dense"],
           "actual_over_dense": r["flops"]["actual_3pass"] / max(1.0, r["flops"]["dense_pass"]),
           "gold_keep_s1": r["gold_keep_frac"][1], "distr_keep_s1": r["distractor_keep_frac"][1]}
    for k in ("lora_s0", "lora_s1", "lora_s2", "head"):
        if k in g:
            out[f"gcos_{k}"] = g[k]["cos"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rs = [row(r) for r in load(a.path)]
    if not rs:
        raise SystemExit("no steps")
    keys = list(rs[0])
    print(" ".join(f"{k:>9.9}" for k in keys))
    for r in rs:
        if r["step"] == 1 or r["step"] % a.every == 0 or r is rs[-1]:
            print(" ".join(f"{r.get(k, float('nan')):9.4f}" if isinstance(r.get(k), float) else f"{r.get(k):>9}"
                           for k in keys))
    n = len(rs)
    wins = [(0, n)] + ([(0, n // 4), (n // 4, n // 2), (n // 2, 3 * n // 4), (3 * n // 4, n)] if n >= 8 else [])
    summ = {}
    for lo, hi in wins:
        sub = rs[lo:hi]
        m = {k: st.fmean(r[k] for r in sub if isinstance(r.get(k), (int, float))) for k in keys if k != "step"}
        summ[f"steps{sub[0]['step']}-{sub[-1]['step']}"] = m
        print(f"mean steps {sub[0]['step']}-{sub[-1]['step']}: " +
              " ".join(f"{k}={v:.4f}" for k, v in m.items()))
    if a.json:
        json.dump({"rows": rs, "windows": summ}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
