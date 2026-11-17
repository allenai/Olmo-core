"""Where an oracle run's selected tokens sit, per task, from its saved selections (oracle_meta/sel/step*_rank*.npz):
keep fraction, share of documents with zero kept tokens, share of kept tokens inside gold documents (vs the gold share of
body tokens), and the position of kept tokens inside their document (relative position bins and the first-8-token share).

A per-document UNIFORM random control at the same keep (rnd1doc<pct>) is the reference: it keeps ceil(frac * len) body
tokens of every document uniformly at random, so its expected values are printed next to the oracle's: zero-kept docs
0, kept-in-gold share = gold share of body, relative position uniform (20% per fifth), first-8 share = E[min(8,len)/len].

    python sel_stats.py <run dir or oracle_meta/sel dir> [--tag NAME] [--every 1] [--json out.json]
Row -> task: the shard's instance index (mt4: 0-1951 contradiction, -3903 nq, -5855 outlier, -7807 oolong).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

TASKS = ["contradiction", "nq", "outlier", "oolong"]
PER_TASK = 1952
NBIN = 5
HEAD = 8


def task_of(index: int) -> str:
    return TASKS[min(int(index) // PER_TASK, len(TASKS) - 1)]


def unpack(a, n):
    return np.unpackbits(a, axis=1)[:, :n].astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--tag", default="")
    ap.add_argument("--every", type=int, default=1, help="use every N-th saved step")
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    d = a.path
    if os.path.isdir(os.path.join(d, "oracle_meta", "sel")):
        d = os.path.join(d, "oracle_meta", "sel")
    files = sorted(glob.glob(os.path.join(d, "step*_rank*.npz")))
    steps = sorted({int(os.path.basename(f)[4:9]) for f in files})[:: a.every]
    keep_steps = set(steps)
    acc = defaultdict(lambda: defaultdict(float))
    for f in files:
        if int(os.path.basename(f)[4:9]) not in keep_steps:
            continue
        z = np.load(f)
        mbs = sorted({k.split("_")[0] for k in z.files})
        for mb in mbs:
            if f"{mb}_orc_s{a.stage}" not in z.files:
                continue
            cid = z[f"{mb}_cid"].astype(np.int32)
            B, L = cid.shape
            body = unpack(z[f"{mb}_body"], L)
            orc = unpack(z[f"{mb}_orc_s{a.stage}"], L)
            rnd = unpack(z[f"{mb}_rnd_s{a.stage}"], L) if f"{mb}_rnd_s{a.stage}" in z.files else None
            gold = unpack(z[f"{mb}_gold"], L) if f"{mb}_gold" in z.files else np.zeros_like(body)
            idx = z[f"{mb}_index"] if f"{mb}_index" in z.files else np.full(B, -1)
            for b in range(B):
                t = task_of(idx[b]) if idx[b] >= 0 else "all"
                A = acc[t]
                for doc in np.unique(cid[b][body[b]]):
                    if doc < 0:
                        continue
                    m = body[b] & (cid[b] == doc)
                    pos = np.nonzero(m)[0]
                    n = len(pos)
                    if n == 0:
                        continue
                    sel = orc[b][pos]
                    k = int(sel.sum())
                    g = bool(gold[b][pos].any())
                    A["docs"] += 1
                    A["body"] += n
                    A["kept"] += k
                    A["zero_docs"] += k == 0
                    A["gold_body"] += n * g
                    A["gold_kept"] += k * g
                    A["gold_docs"] += g
                    A["gold_zero_docs"] += g and k == 0
                    rel = np.arange(n) / max(1, n)
                    kept_rel = rel[sel]
                    for i in range(NBIN):
                        A[f"bin{i}"] += int(((kept_rel >= i / NBIN) & (kept_rel < (i + 1) / NBIN)).sum())
                    A["head_kept"] += int(sel[:HEAD].sum())
                    A["head_exp"] += k * min(HEAD, n) / n  # uniform reference at the same k
                    if rnd is not None:
                        A["rnd_kept"] += int(rnd[b][pos].sum())
                        A["rnd_gold_kept"] += int(rnd[b][pos].sum()) * g
    out = {}
    for t in TASKS + ["all"]:
        if t not in acc:
            continue
        A = acc[t]
        kept = max(1.0, A["kept"])
        r = {"docs": int(A["docs"]), "keep_frac": A["kept"] / max(1, A["body"]),
             "zero_docs": A["zero_docs"] / max(1, A["docs"]),
             "gold_zero_docs": A["gold_zero_docs"] / max(1, A["gold_docs"]),
             "kept_in_gold": A["gold_kept"] / kept, "gold_body_share": A["gold_body"] / max(1, A["body"]),
             "gold_keep": A["gold_kept"] / max(1, A["gold_body"]),
             "distr_keep": (A["kept"] - A["gold_kept"]) / max(1, A["body"] - A["gold_body"]),
             "pos_fifths": [A[f"bin{i}"] / kept for i in range(NBIN)],
             "head8_share": A["head_kept"] / kept, "head8_uniform": A["head_exp"] / kept}
        if A["rnd_kept"]:
            r["rnd_kept_in_gold"] = A["rnd_gold_kept"] / A["rnd_kept"]
        out[t] = r
        print(f"{a.tag or d} {t:13s} docs={r['docs']:6d} keep={r['keep_frac']:.3f} zero_docs={r['zero_docs']:.3f} "
              f"kept_in_gold={r['kept_in_gold']:.3f} (gold body share {r['gold_body_share']:.3f}; gold keep "
              f"{r['gold_keep']:.3f} vs distr {r['distr_keep']:.3f}) pos_fifths="
              + "/".join(f"{x:.2f}" for x in r["pos_fifths"])
              + f" first8={r['head8_share']:.3f} (uniform {r['head8_uniform']:.3f})")
    if a.json:
        json.dump({"tag": a.tag, "steps": len(steps), "by_task": out}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
