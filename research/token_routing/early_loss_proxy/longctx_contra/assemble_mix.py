"""Assemble a mixed-length contradiction pool from existing single-length pools (CPU, node-local /data).

    $PY assemble_mix.py --out lc256kmix_dense --src lc256kL:1134 --src lc16kS:272 --src lc8kS:272 ...

Each ``--src POOL:N`` takes the first N contradiction rows of POOL in its row_info order (pools are written shuffled, so
the first N are a random subset, and a smaller N is a subset of a larger one: the dense arm's rows are a subset of the
gold-drop arm's). One shard part per source (``token_ids_part_t<i>_000.npy``), gold sidecars merged, every row listed in
``gdd_only_contradiction.json``. The trainer shuffles the combined pool, so the bins are interleaved across the run.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

SH = "/data/prasann/early_loss_proxy/shards"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", action="append", required=True)
    ap.add_argument("--root", default=SH)
    ap.add_argument("--min-len", type=int, default=0, help="take only rows at least this long (e.g. the long rows of a mixed pool, in its order)")
    ap.add_argument("--ordered", action="store_true",
                    help="write ONE part with the rows in a planned order (train with --no-data-shuffle): every --src is "
                    "shuffled into the head, except --tail sources, which go (shuffled, with a proportional share of the "
                    "head's short rows) into a final stretch")
    ap.add_argument("--tail", action="append", default=[], help="--ordered: pool name(s) whose rows come last")
    ap.add_argument("--long", default="", help="--ordered: comma list of the long pools (the rest count as short)")
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--bucket", type=int, default=0,
                    help="--ordered: group rows into length-homogeneous blocks of this many rows (= the global batch: "
                    "every step is all-long or all-short, so data-parallel ranks do not wait on one long row), blocks shuffled")
    a = ap.parse_args()
    tmp, final = f"{a.root}/{a.out}_tmp", f"{a.root}/{a.out}"
    assert not os.path.exists(f"{final}/metadata.json"), f"{final} exists"
    os.system(f"rm -rf {tmp}")
    os.makedirs(tmp)
    gold, pairs, fps, lens, bins = {}, {}, [], [], []
    tot = loss = 0
    meta0 = None
    held, info_rows = [], []
    with open(f"{tmp}/row_info.jsonl", "w") as info:
        for pi, spec in enumerate(a.src):
            pool, _, n = spec.partition(":")
            n = int(n)
            d = f"{a.root}/{pool}"
            meta0 = meta0 or json.load(open(f"{d}/metadata.json"))
            g = json.load(open(f"{d}/gold_fingerprints.json"))
            gp = json.load(open(f"{d}/gold_pairs_fingerprints.json"))
            ri = [json.loads(x) for x in open(f"{d}/row_info.jsonl")]
            off, start = {}, {}
            for r in ri:  # row offsets inside each part (rows of a part are concatenated in row order)
                start[(r["part"], r["row"])] = off.get(r["part"], 0)
                off[r["part"]] = off.get(r["part"], 0) + r["len"]
            sel = [r for r in ri if r["task"] == "contradiction" and r["len"] >= a.min_len][:n]
            assert len(sel) == n, f"{pool}: only {len(sel)} contradiction rows < {n}"
            ids, msk = [], []
            mm = {}
            for r in sel:
                p = r["part"]
                if p not in mm:  # sequential full read (random memmap page-ins crawl on a busy ZFS node)
                    mm[p] = (np.fromfile(f"{d}/token_ids_part_t{p}_000.npy", dtype=np.uint32),
                             np.fromfile(f"{d}/labels_mask_t{p}_000.npy", dtype=np.bool_))
                s0 = start[(p, r["row"])]
                ids.append(np.asarray(mm[p][0][s0:s0 + r["len"]]))
                msk.append(np.asarray(mm[p][1][s0:s0 + r["len"]]))
                assert ids[-1][-1] == meta0.get("eos_token_id", ids[-1][-1])
                fp = r["fingerprint"]
                assert fp not in gold, f"duplicate row {fp}"
                gold[fp] = g[fp]
                if fp in gp:
                    pairs[fp] = gp[fp]
                fps.append(fp)
                lens.append(r["len"])
                rec = dict(r, part=pi, row=len(ids) - 1, src_pool=pool)
                info_rows.append(rec)
                if not a.ordered:
                    info.write(json.dumps(rec) + "\n")
            ntok = int(sum(len(x) for x in ids))
            if a.ordered:
                held.extend((pool, x, y, info_rows[-n:][k]) for k, (x, y) in enumerate(zip(ids, msk)))
            else:
                bt, bm = np.concatenate(ids).astype("uint32"), np.concatenate(msk).astype("bool")
                bt.tofile(f"{tmp}/token_ids_part_t{pi}_000.npy")
                bm.tofile(f"{tmp}/labels_mask_t{pi}_000.npy")
                loss += int(bm.sum())
            tot += ntok
            bins.append({"src": pool, "rows": n, "tokens": ntok, "len_mean": float(ntok / n)})
            print(f"[mix] {a.out} part {pi}: {pool} x {n}: {ntok} tokens", flush=True)
    order_info = None
    if a.ordered:
        rng = np.random.default_rng(a.seed)
        longs = set(x for x in a.long.split(",") if x)
        tail_long = [h for h in held if h[0] in a.tail]
        head_long = [h for h in held if h[0] in longs and h[0] not in a.tail]
        short = [h for h in held if h[0] not in longs and h[0] not in a.tail]
        short = [short[j] for j in rng.permutation(len(short))]
        frac = len(tail_long) / max(1, len(tail_long) + len(head_long))
        n_tail_short = int(round(frac * len(short)))
        head = head_long + short[n_tail_short:]
        tail = tail_long + short[:n_tail_short]
        def arrange(rows_):
            if not a.bucket:
                return [rows_[j] for j in rng.permutation(len(rows_))], []
            lo = [r for r in rows_ if r[0] in longs]
            sh = [r for r in rows_ if r[0] not in longs]
            lo = [lo[j] for j in rng.permutation(len(lo))]
            sh = [sh[j] for j in rng.permutation(len(sh))]
            b = a.bucket
            blocks = [lo[i:i + b] for i in range(0, len(lo) - len(lo) % b, b)] + \
                     [sh[i:i + b] for i in range(0, len(sh) - len(sh) % b, b)]
            rest = lo[len(lo) - len(lo) % b:] + sh[len(sh) - len(sh) % b:]  # < 2 blocks of leftovers, mixed, at the end
            blocks = [blocks[j] for j in rng.permutation(len(blocks))]
            return [r for blk in blocks for r in blk], rest
        head, head_rest = arrange(head)
        tail, tail_rest = arrange(tail)
        tail = tail + head_rest + tail_rest  # leftovers (< 2 blocks each) last, so every block stays step-aligned
        rows_o = head + tail
        with open(f"{tmp}/row_info.jsonl", "w") as info:
            for k, (pool, x, y, rec) in enumerate(rows_o):
                info.write(json.dumps(dict(rec, part=0, row=k, src_pool=pool, tail=k >= len(head))) + "\n")
        loss = 0
        with open(f"{tmp}/token_ids_part_t0_000.npy", "wb") as ft, open(f"{tmp}/labels_mask_t0_000.npy", "wb") as fm:
            for h in rows_o:
                ft.write(np.ascontiguousarray(h[1], dtype=np.uint32).tobytes())
                fm.write(np.ascontiguousarray(h[2], dtype=np.bool_).tobytes())
                loss += int(h[2].sum())
        lens = [len(h[1]) for h in rows_o]
        order_info = {"head_rows": len(head), "tail_rows": len(tail), "tail_long": len(tail_long),
                      "tail_short": n_tail_short, "train_with": "--no-data-shuffle"}
        print(f"[mix] ordered: head {len(head)} rows, tail {len(tail)} rows ({len(tail_long)} {a.tail} + {n_tail_short} short)",
              flush=True)
    meta = dict(meta0)
    meta.update({"task": a.out, "num_instances": len(lens), "num_tokens": int(tot), "num_loss_tokens": int(loss),
                 "max_example_len": int(max(lens)), "min_example_len": int(min(lens)),
                 "merged_from": [{"src": b["src"], "task": "contradiction", "converter_task": "contradiction",
                                  "num_instances": b["rows"]} for b in bins],
                 "tasks_order": [b["src"] for b in bins], "order": order_info})
    json.dump(meta, open(f"{tmp}/metadata.json", "w"), indent=1)
    json.dump(gold, open(f"{tmp}/gold_fingerprints.json", "w"))
    json.dump(pairs, open(f"{tmp}/gold_pairs_fingerprints.json", "w"))
    json.dump(fps, open(f"{tmp}/gdd_only_contradiction.json", "w"))
    json.dump({"pool": a.out, "bins": bins, "rows": len(lens), "tokens": int(tot), "order": order_info}, open(f"{a.root}/{a.out}_stats.json", "w"), indent=1)
    os.rename(tmp, final)
    print(f"[mix] wrote {final}: {len(lens)} rows, {tot} tokens", flush=True)


if __name__ == "__main__":
    main()
