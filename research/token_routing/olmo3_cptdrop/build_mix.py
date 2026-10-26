"""
Mixed-length CPT shard for the length-mix / hybrid arms: full 32k rows (L) and short windows (S) cut from DISJOINT
train32k_ee rows, every document EOS-LAST (as the *_ee shards), shuffled together into one EOS-delimited stream for
the packing loader (train_ctc_suite --pack). rho = token share of L rows.

  L rows: train32k_ee rows [start, start + n_long), kept whole (32,767 body tokens + EOS).
  S rows: train32k_ee rows [start + n_long, start + n_long + n_short), each cut into windows of (window - 1) body
          tokens + EOS (8 x 4k per 32k row; the few tail tokens of each row are dropped).
Labels: every body token, never EOS (as the source shard).

    python build_mix.py --src .../data/train32k_ee --out .../data/mix4k_r25 --n-long 1000 --n-short 3000
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
    ap.add_argument("--n-long", type=int, required=True, help="32k rows kept whole")
    ap.add_argument("--n-short", type=int, required=True, help="32k rows cut into short windows")
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--start", type=int, default=0, help="first source row")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.src, "metadata.json")))
    L, eos = meta["max_example_len"], meta["eos_token_id"]
    assert meta.get("eos_position") == "last", "source must be an EOS-last (*_ee) shard"
    tok = np.memmap(os.path.join(a.src, "token_ids_part_0000.npy"), dtype=np.uint32, mode="r").reshape(-1, L)
    lo, mid, hi = a.start, a.start + a.n_long, a.start + a.n_long + a.n_short
    assert hi <= tok.shape[0], f"needs rows up to {hi}, shard has {tok.shape[0]}"
    w = a.window - 1
    k = (L - 1) // w
    docs = [np.asarray(tok[i]) for i in range(lo, mid)]  # whole rows: body + EOS
    for i in range(mid, hi):
        body = np.asarray(tok[i, : L - 1])
        for j in range(k):
            docs.append(np.concatenate([body[j * w : (j + 1) * w], np.array([eos], dtype=np.uint32)]))
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(docs))
    stream = np.concatenate([docs[i] for i in order]).astype(np.uint32)
    mask = stream != eos
    for d in docs:
        assert d[-1] == eos and int((d[:-1] == eos).sum()) == 0
    n_long_tok = a.n_long * L
    os.makedirs(a.out, exist_ok=True)
    stream.tofile(os.path.join(a.out, "token_ids_part_0000.npy"))
    mask.tofile(os.path.join(a.out, "labels_mask_0000.npy"))
    out = dict(meta)
    out.update(
        num_instances=len(docs), num_tokens=int(stream.size), num_loss_tokens=int(mask.sum()),
        max_example_len=L if a.n_long else a.window, min_example_len=a.window if a.n_short else L,
        eos_position="last", packed_mixed_lengths=True,
        rho_long_token_share=round(n_long_tok / stream.size, 4),
        source_rows=f"train32k_ee rows {lo}..{mid - 1} whole (L), {mid}..{hi - 1} cut into {k} x (EOS-last {a.window}) (S)",
    )
    json.dump(out, open(os.path.join(a.out, "metadata.json"), "w"), indent=1)
    print(f"wrote {a.out}: {a.n_long} L rows + {(hi - mid) * k} S windows = {stream.size:,} tokens, rho {out['rho_long_token_share']}")


if __name__ == "__main__":
    main()
