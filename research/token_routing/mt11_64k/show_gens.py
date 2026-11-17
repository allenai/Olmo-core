"""Print a few raw generations (+ length, repetition score, gold) from ladder ``*.generations.jsonl`` files, per rung.

    python show_gens.py --files <a.generations.jsonl> [...] --rungs 8k 16k --n 3
"""
import argparse
import json
from collections import Counter


def rep_score(text):
    """Fraction of the generation covered by its most frequent 8-word window (1.0 = a pure loop)."""
    w = text.split()
    if len(w) < 16:
        return 0.0
    grams = Counter(tuple(w[i:i + 8]) for i in range(len(w) - 7))
    return grams.most_common(1)[0][1] * 8 / len(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--rungs", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    for f in a.files:
        rows = [json.loads(l) for l in open(f)]
        print(f"== {f}: {len(rows)} rows; keys {sorted(rows[0]) if rows else []}")
        by = {}
        for r in rows:
            by.setdefault(str(r.get("rung") or r.get("_rung") or r.get("ladder_rung") or "?"), []).append(r)
        for rung, rs in by.items():
            if a.rungs and not any(x in rung for x in a.rungs):
                continue
            gens = [str(r.get("generation") or r.get("response") or r.get("pred") or r.get("output") or "") for r in rs]
            lens = sorted(len(g) for g in gens)
            reps = [rep_score(g) for g in gens]
            print(f"-- rung {rung}: {len(rs)} rows, gen chars p50 {lens[len(lens)//2]} max {lens[-1]}, "
                  f"empty {sum(not g.strip() for g in gens)}, loop(rep>0.5) {sum(x > 0.5 for x in reps)}")
            for r, g in list(zip(rs, gens))[: a.n]:
                gold = r.get("gold") or r.get("answer") or r.get("target")
                print(f"   GEN[{len(g)}c]: {g[:500]!r}")
                print(f"   ...tail: {g[-300:]!r}")
                print(f"   GOLD: {str(gold)[:300]!r}")


if __name__ == "__main__":
    main()
