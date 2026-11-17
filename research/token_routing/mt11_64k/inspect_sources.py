"""Row-length histogram + gold-sidecar coverage of candidate source shards for the mt11 64k mix.

For every ``<root>/<task>`` shard (converter output: raw uint32 token parts + bool mask parts, rows end at
EOS, ``metadata.json``, optional ``gold_fingerprints.json``) prints rows / tokens per length bin, how many
rows fit in 65,536 tokens, and the sidecar size. Read-only, numpy only.

    python inspect_sources.py --roots <set>/shards_qwen3_5_gold <set>/shards_qwen35_256k --out report.json
"""
import argparse
import glob
import json
import os

import numpy as np

TASKS = ["nq", "hotpotqa", "qdmatch_nq", "outlier", "oolong", "contradiction", "xabsence", "reorder",
         "rerank", "strmatch", "grouping"]
EDGES = [0, 3072, 6144, 12288, 24576, 49152, 65536, 131072, 1 << 40]
EOS_CANDS = (248044, 248046, 151643, 151645)


def row_lengths(src):
    meta = json.load(open(os.path.join(src, "metadata.json")))
    toks = sorted(glob.glob(os.path.join(src, "token_ids_part_*.npy")))
    want = int(meta["num_instances"])
    lens, eos = [], None
    for t in toks:
        a = np.fromfile(t, dtype=meta.get("dtype", "uint32"))
        if eos is None:
            for c in EOS_CANDS:
                if int((a == c).sum()) > 0:
                    eos = c
                    break
        ends = np.flatnonzero(a == eos)
        starts = np.concatenate([[0], ends[:-1] + 1])
        lens.append(ends - starts + 1)
    lens = np.concatenate(lens) if lens else np.zeros(0, dtype=np.int64)
    return meta, lens, want, eos, len(toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rep = {}
    for root in a.roots:
        for t in TASKS:
            src = os.path.join(root, t)
            if not os.path.exists(os.path.join(src, "metadata.json")):
                print(f"[missing] {src}")
                continue
            meta, lens, want, eos, nparts = row_lengths(src)
            side = os.path.join(src, "gold_fingerprints.json")
            nside = len(json.load(open(side))) if os.path.exists(side) else -1
            bins = []
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                m = (lens > lo) & (lens <= hi)
                bins.append({"lo": lo, "hi": hi, "rows": int(m.sum()), "tokens": int(lens[m].sum())})
            r = {"rows": int(len(lens)), "num_instances": want, "eos": eos, "parts": nparts, "sidecar": nside,
                 "rows_le_65536": int((lens <= 65536).sum()), "tokens_le_65536": int(lens[lens <= 65536].sum()),
                 "max_len": int(lens.max()) if len(lens) else 0, "bins": bins,
                 "meta": {k: meta.get(k) for k in ("query_position", "marker_set", "max_example_len",
                                                     "num_tokens", "seq_len", "task")}}
            rep[src] = r
            print(f"== {src}: rows {r['rows']} (meta {want}) eos {eos} parts {nparts} sidecar {nside} "
                  f"<=64k rows {r['rows_le_65536']} tokens {r['tokens_le_65536']/1e6:.1f}M max {r['max_len']} "
                  f"qp {meta.get('query_position')}")
            print("   " + " | ".join(f"<={b['hi'] if b['hi'] < 1<<40 else 'inf'}: {b['rows']} r {b['tokens']/1e6:.1f}M"
                                    for b in bins))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
