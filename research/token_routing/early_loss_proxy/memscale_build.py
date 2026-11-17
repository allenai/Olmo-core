"""Synthetic long rows for memory-scaling measurements (memscale loop idea, 2026-11-17).

For each target length L, builds a pool of ``--rows`` rows of at most L - 64 tokens from one task of ``mtb16k``:
the prompt prefix and the query + answer suffix of a 16k row, with the documents (marker-wrapped spans) of that row
and of the following rows of the same task concatenated in between until the length is reached. The answer span
(label mask) is the original row's, so the row has a valid loss span; the content is not a consistent task (document
IDs repeat), which does not matter for memory. Output: ``<out>/L<L>/`` in the shard layout the trainer reads
(raw uint32 ``token_ids_part_t0_000.npy`` + bool ``labels_mask_t0_000.npy`` + ``metadata.json``). No gold sidecar
(the online router only reads one if present).

    python memscale_build.py --src /data/prasann/early_loss_proxy/shards/mtb16k \\
        --out /data/prasann/early_loss_proxy/shards/memscale --lengths 16384,32768,65536,131072,262144
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data/prasann/early_loss_proxy/shards/mtb16k")
    ap.add_argument("--out", default="/data/prasann/early_loss_proxy/shards/memscale")
    ap.add_argument("--lengths", default="16384,32768,65536,131072,262144")
    ap.add_argument("--part", type=int, default=0, help="task part (0 = contradiction)")
    ap.add_argument("--rows", type=int, default=4)
    a = ap.parse_args()
    info = [json.loads(ln) for ln in open(f"{a.src}/row_info.jsonl")]
    info = sorted((d for d in info if d["part"] == a.part), key=lambda d: d["row"])
    ids = np.memmap(f"{a.src}/token_ids_part_t{a.part}_000.npy", dtype=np.uint32, mode="r")
    msk = np.memmap(f"{a.src}/labels_mask_t{a.part}_000.npy", dtype=np.bool_, mode="r")
    offs = np.concatenate([[0], np.cumsum([d["len"] for d in info])])
    assert offs[-1] == ids.shape[0], (offs[-1], ids.shape)
    meta0 = json.load(open(f"{a.src}/metadata.json"))
    long_rows = [i for i, d in enumerate(info) if d["rung"] == "16k"]
    for L in [int(x) for x in a.lengths.split(",")]:
        toks, masks, lens = [], [], []
        cursor = 0  # next donor row
        for r in range(a.rows):
            A = long_rows[r]
            o, d = offs[A], info[A]
            sp = d["doc_spans"]
            prefix = np.asarray(ids[o:o + sp[0][0]])
            suffix = np.asarray(ids[o + sp[-1][1]:o + d["len"]])
            suffix_m = np.asarray(msk[o + sp[-1][1]:o + d["len"]])
            assert suffix[-1] == meta0["eos_token_id"] and suffix_m.sum() > 0
            budget = L - 64 - len(prefix) - len(suffix)
            body, n = [], 0
            first = True
            while True:
                if first:
                    D, first = A, False
                else:
                    D = cursor % len(info)
                    cursor += 1
                    if D in long_rows[:a.rows]:
                        continue
                od = offs[D]
                full = False
                for s, e in info[D]["doc_spans"]:
                    if n + (e - s) > budget:
                        full = True
                        break
                    body.append(np.asarray(ids[od + s:od + e]))
                    n += e - s
                if full:
                    break
            row = np.concatenate([prefix] + body + [suffix]).astype(np.uint32)
            m = np.concatenate([np.zeros(len(row) - len(suffix), np.bool_), suffix_m])
            assert row.max() < 248320 and (row == meta0["eos_token_id"]).sum() == 1
            toks.append(row)
            masks.append(m)
            lens.append(len(row))
        od = f"{a.out}/L{L}"
        os.makedirs(od, exist_ok=True)
        np.concatenate(toks).tofile(f"{od}/token_ids_part_t0_000.npy")
        np.concatenate(masks).tofile(f"{od}/labels_mask_t0_000.npy")
        meta = dict(meta0, task=f"memscale_L{L}", num_instances=len(lens), num_tokens=int(sum(lens)),
                    num_loss_tokens=int(sum(int(x.sum()) for x in masks)), max_example_len=max(lens),
                    min_example_len=min(lens), subset_max_len=L, merged_from=[],
                    memscale={"src": a.src, "part": a.part, "lens": lens,
                              "note": "synthetic: 16k row prefix/suffix + docs of following rows; memory only"})
        json.dump(meta, open(f"{od}/metadata.json", "w"), indent=1)
        print(f"L={L}: rows {lens}, loss tokens {meta['num_loss_tokens']} -> {od}", flush=True)


if __name__ == "__main__":
    main()
