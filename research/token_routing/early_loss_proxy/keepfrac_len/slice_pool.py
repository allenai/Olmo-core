"""Quick single-length pool: the rows of one rung of an existing pool (e.g. mtb64k's 32k rows -> mtb64k_r32k).

Same layout / sidecars as the source; row order within each part is kept. Runs on the node that holds the pool.
    $PY slice_pool.py --src /data/prasann/early_loss_proxy/shards/mtb64k --rung 32k
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--rung", required=True)
    a = ap.parse_args()
    out = f"{a.src}_r{a.rung}"
    tmp = out + "_tmp"
    os.makedirs(tmp, exist_ok=True)
    meta = json.load(open(f"{a.src}/metadata.json"))
    info = [json.loads(x) for x in open(f"{a.src}/row_info.jsonl")]
    gold = json.load(open(f"{a.src}/gold_fingerprints.json"))
    pairs = json.load(open(f"{a.src}/gold_pairs_fingerprints.json"))
    parts = sorted({x["part"] for x in info})
    new_info, g2, p2, lens, contra = [], {}, {}, [], []
    tot = loss = 0
    for pt in parts:
        rows = sorted([x for x in info if x["part"] == pt], key=lambda x: x["row"])
        ids = np.fromfile(f"{a.src}/token_ids_part_t{pt}_000.npy", dtype=meta["dtype"])
        mk = np.fromfile(f"{a.src}/labels_mask_t{pt}_000.npy", dtype=meta["mask_dtype"])
        off = np.concatenate([[0], np.cumsum([x["len"] for x in rows])])
        assert off[-1] == ids.size == mk.size, (pt, off[-1], ids.size)
        keep = [i for i, x in enumerate(rows) if x["rung"] == a.rung]
        bt = np.concatenate([ids[off[i]:off[i + 1]] for i in keep])
        bm = np.concatenate([mk[off[i]:off[i + 1]] for i in keep])
        bt.tofile(f"{tmp}/token_ids_part_t{pt}_000.npy")
        bm.tofile(f"{tmp}/labels_mask_t{pt}_000.npy")
        tot += bt.size
        loss += int(bm.sum())
        for k, i in enumerate(keep):
            x = dict(rows[i], row=k)
            new_info.append(x)
            lens.append(x["len"])
            g2[x["fingerprint"]] = gold[x["fingerprint"]]
            if x["fingerprint"] in pairs:
                p2[x["fingerprint"]] = pairs[x["fingerprint"]]
            if x["task"] == "contradiction":
                contra.append(x["fingerprint"])
    with open(f"{tmp}/row_info.jsonl", "w") as f:
        for x in new_info:
            f.write(json.dumps(x) + "\n")
    json.dump(g2, open(f"{tmp}/gold_fingerprints.json", "w"))
    json.dump(p2, open(f"{tmp}/gold_pairs_fingerprints.json", "w"))
    json.dump(contra, open(f"{tmp}/gdd_only_contradiction.json", "w"))
    meta.update({"task": os.path.basename(out), "num_instances": len(lens), "num_tokens": int(tot),
                 "num_loss_tokens": loss, "max_example_len": int(max(lens)), "min_example_len": int(min(lens)),
                 "sliced_from": a.src, "rung": a.rung})
    json.dump(meta, open(f"{tmp}/metadata.json", "w"), indent=1)
    os.rename(tmp, out)
    print(f"[slice] {out}: {len(lens)} rows, {tot} tokens, max {max(lens)}", flush=True)


if __name__ == "__main__":
    main()
