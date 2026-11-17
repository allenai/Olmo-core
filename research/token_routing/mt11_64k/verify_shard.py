"""Verify a merged mt11 64k shard before any training reads it.

Per source part (``token_ids_part_t<i>_*`` = one task, ``metadata.json`` ``merged_from`` order) it recomputes
from the raw bytes: row count vs metadata, length histogram over the length-mix bins, max row length vs
``--seq-len``, rows with no loss token, and the GOLD SIDECAR lookup exactly as the trainer does it
(``content_fingerprint_from_row`` -> ``gold_fingerprints.json``): the fraction of rows that hit, and of those the
fraction with a non-empty gold list. Writes a JSON report; exits 1 on any hard failure.

    PYTHONPATH=src python research/token_routing/mt11_64k/verify_shard.py --shard <dir> --seq-len 65536 --out r.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

EDGES = [0, 3072, 6144, 12288, 24576, 49152, 65536]
#: tasks whose rows legitimately carry no gold documents (empty sidecar entries); others only WARN if empty
NO_GOLD_OK = {"oolong"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--seq-len", type=int, default=65536)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from olmo_core.nn.attention.gold_grad_mask import content_fingerprint_from_row

    meta = json.load(open(os.path.join(a.shard, "metadata.json")))
    gold = json.load(open(os.path.join(a.shard, "gold_fingerprints.json")))
    eos = 248044
    rep, bad, warn = {"shard": a.shard, "tasks": {}}, [], []
    tot_rows = 0
    for i, src in enumerate(meta["merged_from"]):
        task = os.path.basename(src["src"].rstrip("/"))
        toks = sorted(glob.glob(os.path.join(a.shard, f"token_ids_part_t{i}_*.npy")))
        masks = sorted(glob.glob(os.path.join(a.shard, f"labels_mask_t{i}_*.npy")))
        tok = np.concatenate([np.fromfile(t, dtype=meta.get("dtype", "uint32")) for t in toks])
        msk = np.concatenate([np.fromfile(m, dtype=meta.get("mask_dtype", "bool")) for m in masks])
        ends = np.flatnonzero(tok == eos)
        starts = np.concatenate([[0], ends[:-1] + 1])
        lens = ends - starts + 1
        loss = np.add.reduceat(msk.astype(np.int64), starts) if len(starts) else np.zeros(0)
        hit = nonempty = 0
        for s, e in zip(starts, ends):
            fp = content_fingerprint_from_row(tok[s:e + 1].tolist(), eos)
            g = gold.get(fp)
            if g is not None:
                hit += 1
                nonempty += bool(g)
        hist = [{"lo": lo, "hi": hi, "rows": int(((lens > lo) & (lens <= hi)).sum()),
                 "tokens": int(lens[(lens > lo) & (lens <= hi)].sum())} for lo, hi in zip(EDGES[:-1], EDGES[1:])]
        r = {"rows": int(len(lens)), "meta_rows": src.get("num_instances"), "tokens": int(lens.sum()),
             "loss_tokens": int(msk.sum()), "max_len": int(lens.max()), "min_len": int(lens.min()),
             "rows_no_loss": int((loss == 0).sum()), "sidecar_hit": hit, "gold_nonempty": nonempty,
             "tail_after_last_eos": int(len(tok) - (ends[-1] + 1)) if len(ends) else int(len(tok)), "hist": hist}
        rep["tasks"][task] = r
        tot_rows += r["rows"]
        if r["rows"] != r["meta_rows"]:
            bad.append(f"{task}: {r['rows']} rows vs metadata {r['meta_rows']}")
        if r["max_len"] > a.seq_len:
            bad.append(f"{task}: max_len {r['max_len']} > seq_len {a.seq_len}")
        if r["rows_no_loss"]:
            bad.append(f"{task}: {r['rows_no_loss']} rows without a loss token")
        if r["tail_after_last_eos"]:
            bad.append(f"{task}: {r['tail_after_last_eos']} tokens after the last EOS")
        if hit != r["rows"]:
            bad.append(f"{task}: sidecar hit {hit}/{r['rows']}")
        if task not in NO_GOLD_OK and nonempty < 0.99 * r["rows"]:
            warn.append(f"{task}: non-empty gold on only {nonempty}/{r['rows']} rows")
        print(f"{task:14} rows {r['rows']:6d} tok {r['tokens']/1e6:6.1f}M loss {r['loss_tokens']/1e6:5.2f}M "
              f"max {r['max_len']:6d} sidecar {hit}/{r['rows']} gold {nonempty} | "
              + " ".join(f"{h['hi']//1024}k:{h['rows']}/{h['tokens']/1e6:.1f}M" for h in hist), flush=True)
    rep["total_rows"] = tot_rows
    rep["total_tokens"] = sum(t["tokens"] for t in rep["tasks"].values())
    rep["meta_num_instances"] = meta.get("num_instances")
    if tot_rows != meta.get("num_instances"):
        bad.append(f"total rows {tot_rows} vs metadata {meta.get('num_instances')}")
    rep["failures"], rep["warnings"] = bad, warn
    for w in warn:
        print("WARN " + w)
    print(f"TOTAL rows {tot_rows} tokens {rep['total_tokens']/1e6:.1f}M max_example_len {meta.get('max_example_len')}")
    print("VERIFY " + ("FAILED: " + "; ".join(bad) if bad else "PASSED"))
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
