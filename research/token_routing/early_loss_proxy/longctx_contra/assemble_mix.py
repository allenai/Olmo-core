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
    a = ap.parse_args()
    tmp, final = f"{a.root}/{a.out}_tmp", f"{a.root}/{a.out}"
    assert not os.path.exists(f"{final}/metadata.json"), f"{final} exists"
    os.system(f"rm -rf {tmp}")
    os.makedirs(tmp)
    gold, pairs, fps, lens, bins = {}, {}, [], [], []
    tot = loss = 0
    meta0 = None
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
            sel = [r for r in ri if r["task"] == "contradiction"][:n]
            assert len(sel) == n, f"{pool}: only {len(sel)} contradiction rows < {n}"
            ids, msk = [], []
            mm = {}
            for r in sel:
                p = r["part"]
                if p not in mm:
                    mm[p] = (np.memmap(f"{d}/token_ids_part_t{p}_000.npy", dtype=np.uint32, mode="r"),
                             np.memmap(f"{d}/labels_mask_t{p}_000.npy", dtype=np.bool_, mode="r"))
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
                info.write(json.dumps(dict(r, part=pi, row=len(ids) - 1, src_pool=pool)) + "\n")
            bt, bm = np.concatenate(ids).astype("uint32"), np.concatenate(msk).astype("bool")
            bt.tofile(f"{tmp}/token_ids_part_t{pi}_000.npy")
            bm.tofile(f"{tmp}/labels_mask_t{pi}_000.npy")
            tot += bt.size
            loss += int(bm.sum())
            bins.append({"src": pool, "rows": n, "tokens": int(bt.size), "len_mean": float(bt.size / n)})
            print(f"[mix] {a.out} part {pi}: {pool} x {n}: {bt.size} tokens", flush=True)
    meta = dict(meta0)
    meta.update({"task": a.out, "num_instances": len(lens), "num_tokens": int(tot), "num_loss_tokens": int(loss),
                 "max_example_len": int(max(lens)), "min_example_len": int(min(lens)),
                 "merged_from": [{"src": b["src"], "task": "contradiction", "converter_task": "contradiction",
                                  "num_instances": b["rows"]} for b in bins],
                 "tasks_order": [b["src"] for b in bins]})
    json.dump(meta, open(f"{tmp}/metadata.json", "w"), indent=1)
    json.dump(gold, open(f"{tmp}/gold_fingerprints.json", "w"))
    json.dump(pairs, open(f"{tmp}/gold_pairs_fingerprints.json", "w"))
    json.dump(fps, open(f"{tmp}/gdd_only_contradiction.json", "w"))
    json.dump({"pool": a.out, "bins": bins, "rows": len(lens), "tokens": int(tot)}, open(f"{a.root}/{a.out}_stats.json", "w"), indent=1)
    os.rename(tmp, final)
    print(f"[mix] wrote {final}: {len(lens)} rows, {tot} tokens", flush=True)


if __name__ == "__main__":
    main()
