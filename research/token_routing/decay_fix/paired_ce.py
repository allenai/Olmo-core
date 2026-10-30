"""Paired answer-CE differences between checkpoints (same 500 rows per cell), with the paired SE of the 8-cell mean.

    python research/token_routing/decay_fix/paired_ce.py results/ce/*.json --pairs A:B,C:D
Each pair prints CE(A) - CE(B) per cell (mean of per-row mean CE) and for the 8-cell mean, with paired SEs.
"""
import argparse
import json
import math

CELLS = [f"{t}_{r}" for t in ("contradiction", "nq", "oolong", "outlier") for r in ("8k", "2k")]


def per_row(cell):
    return [None if (r is None or r["ans_n"] == 0) else r["ans_sum"] / r["ans_n"] for r in cell["rows"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--pairs", required=True)
    a = ap.parse_args()
    ce = {}
    for f in a.files:
        for k, v in json.load(open(f)).items():
            if not k.startswith("_"):
                ce[k] = v
    for pair in a.pairs.split(","):
        x, y = pair.split(":")
        diffs_by_row = None
        line = []
        cell_d = []
        for c in CELLS:
            rx, ry = per_row(ce[x][c]), per_row(ce[y][c])
            d = [u - v for u, v in zip(rx, ry) if u is not None and v is not None]
            m = sum(d) / len(d)
            se = math.sqrt(sum((t - m) ** 2 for t in d) / (len(d) - 1) / len(d))
            cell_d.append((m, se, d))
            line.append(f"{c}={m:+.4f}+-{se:.4f}")
        # 8-cell mean: rows are independent across cells (different files), so var = sum(se^2)/64
        mean = sum(m for m, _, _ in cell_d) / 8
        se8 = math.sqrt(sum(se ** 2 for _, se, _ in cell_d)) / 8
        unsat = [i for i, c in enumerate(CELLS) if c.startswith(("oolong", "outlier"))]
        mu = sum(cell_d[i][0] for i in unsat) / 4
        seu = math.sqrt(sum(cell_d[i][1] ** 2 for i in unsat)) / 4
        print(f"{x} - {y}: 8cell {mean:+.4f}+-{se8:.4f} | unsat4 {mu:+.4f}+-{seu:.4f}\n    " + " ".join(line))


if __name__ == "__main__":
    main()
