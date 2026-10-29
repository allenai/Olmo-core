"""Build a LONG-ROW shard for memory-vs-length probes: concatenate consecutive examples of a shard into rows of up to
``--target`` tokens (inner EOS tokens become ``--sep`` so each row stays ONE example with many documents and several
query / answer spans; labels carry over). Content is real, so the router sees real document structure.

    python make_long_rows.py --src /tmp/prasann/tdlr/shards/mt4s1464 --out /tmp/prasann/tdlr/shards/mt4L65536 \
        --target 65536 --rows 8
"""
import argparse
import glob
import json
import os

import numpy as np


def examples(src, eos, dtype, mdtype):
    # the shards are RAW memmaps (OLMo-core numpy format: no .npy header), dtypes from metadata.json
    for f in sorted(glob.glob(os.path.join(src, "token_ids_part_*.npy"))):
        ids = np.memmap(f, dtype=dtype, mode="r")
        lab = np.memmap(f.replace("token_ids_part_", "labels_mask_"), dtype=mdtype, mode="r")
        ends = np.nonzero(ids == eos)[0]
        start = 0
        for e in ends:
            yield np.asarray(ids[start:e + 1]), np.asarray(lab[start:e + 1])
            start = e + 1


def build(src, out, target, rows=8, sep=198):
    """Write ``rows`` rows of up to ``target`` tokens built from ``src`` into ``out``; returns the row lengths."""
    meta = json.load(open(os.path.join(src, "metadata.json")))
    eos = int(meta["eos_token_id"])
    dtype, mdtype = np.dtype(meta.get("dtype", "uint32")), np.dtype(meta.get("mask_dtype", "bool"))
    got, cur_i, cur_l = [], [], []
    for ids, lab in examples(src, eos, dtype, mdtype):
        n = sum(len(x) for x in cur_i)
        if cur_i and n + len(ids) > target - 64:
            got.append((cur_i, cur_l))
            cur_i, cur_l = [], []
            if len(got) == rows:
                break
        cur_i.append(ids)
        cur_l.append(lab)
    if len(got) < rows:
        raise SystemExit(f"only {len(got)} rows of <= {target} tokens could be built")
    out_ids, out_lab, lens = [], [], []
    for ci, cl in got:
        ids = np.concatenate(ci).astype(dtype)
        lab = np.concatenate(cl).astype(mdtype)
        inner = np.nonzero(ids[:-1] == eos)[0]
        ids[inner] = sep
        out_ids.append(ids)
        out_lab.append(lab)
        lens.append(len(ids))
    os.makedirs(out, exist_ok=True)
    np.concatenate(out_ids).tofile(os.path.join(out, "token_ids_part_t0_000.npy"))
    np.concatenate(out_lab).tofile(os.path.join(out, "labels_mask_t0_000.npy"))
    m = {k: v for k, v in meta.items() if k not in ("merged_from",)}
    m.update(num_instances=len(got), num_tokens=int(sum(lens)), max_example_len=int(max(lens)),
             min_example_len=int(min(lens)), num_loss_tokens=int(sum(int(x.sum()) for x in out_lab)),
             long_rows={"src": src, "target": target, "sep": sep, "row_lens": lens})
    json.dump(m, open(os.path.join(out, "metadata.json"), "w"), indent=1)
    return lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--sep", type=int, default=198, help="token replacing the inner EOS tokens")
    a = ap.parse_args()
    lens = build(a.src, a.out, a.target, a.rows, a.sep)
    print(f"wrote {len(lens)} rows, lengths {lens} -> {a.out}")


if __name__ == "__main__":
    main()
