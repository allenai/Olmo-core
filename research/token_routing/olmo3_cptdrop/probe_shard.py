"""
Synthetic long-row shard for MEMORY probes only (max trainable T per GPU): each row joins the bodies of k consecutive
train32k_ee rows (their EOS dropped) and ends with one EOS, so it has the EOS-last layout the loader and the Beaker
data check expect. Content spans unrelated documents -- never use it for a quality number.

    python probe_shard.py --src .../data/train32k_ee --out .../data/probe128k_ee --row-len 131072 --rows 16
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--row-len", type=int, required=True)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--start", type=int, default=0, help="first source row")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.src, "metadata.json")))
    L, eos = meta["max_example_len"], meta["eos_token_id"]
    tok = np.memmap(os.path.join(a.src, "token_ids_part_0000.npy"), dtype=np.uint32, mode="r").reshape(-1, L)
    k = -(-(a.row_len - 1) // (L - 1))  # source rows per output row
    out = np.empty((a.rows, a.row_len), dtype=np.uint32)
    for i in range(a.rows):
        s = a.start + i * k
        body = np.concatenate([np.asarray(tok[s + j, : L - 1]) for j in range(k)])[: a.row_len - 1]
        out[i, :-1], out[i, -1] = body, eos
    assert int((out[:, :-1] == eos).sum()) == 0
    os.makedirs(a.out, exist_ok=True)
    out.tofile(os.path.join(a.out, "token_ids_part_0000.npy"))
    (out != eos).tofile(os.path.join(a.out, "labels_mask_0000.npy"))
    m = dict(meta)
    m.update(num_instances=a.rows, num_tokens=int(out.size), num_loss_tokens=int((out != eos).sum()),
             max_example_len=a.row_len, min_example_len=a.row_len, eos_position="last",
             synthetic_memory_probe=True, source_rows=f"train32k_ee rows {a.start}..{a.start + a.rows * k - 1}, {k} per row")
    json.dump(m, open(os.path.join(a.out, "metadata.json"), "w"), indent=1)
    print(f"wrote {a.out}: {a.rows} x {a.row_len} (k={k})")


if __name__ == "__main__":
    main()
