"""Bigger single-length pools that contain an existing kl pool (kl16k_x6 / _x3p2 / _x2p4 for the 16k gold-drop ablation).

kl16k's rows come from the same 16k generator cells (seed 42, n_docs 324 / 96 / 174) as mtb16kx16's 16k rows, and all
3,600 of them are in mtb16kx16 (checked by fingerprint). A grown pool = for each task (part t0..t2 in kl order): the
base pool's rows in the base's part order, then mtb16kx16's other 16k rows of that task in mtb16kx16's order, up to
--rows-per-task. Held-out hygiene vs the base's held-out set (kl16k = heldout_v3 gen16k): mtb16kx16_overlap.json
reports 0 prompt hits and 0 gold-as-gold hits for gen16k; on top, as build_kl_pools does, outlier rows are dropped
if ANY document is a held-out outlier passage (sha1 of the whitespace-normalised passage text, decoded from the
tokens with the "Document [k]: " header stripped; verified equal to the source-text hash on 1,920 kl16k documents).
Runs on the node holding both pools (no network I/O).

    $PY grow_pool.py --base kl16k --src mtb16kx16 --rung 16k --rows 7200:kl16k_x6,3840:kl16k_x3p2,2880:kl16k_x2p4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re

import numpy as np

SH = "/data/prasann/early_loss_proxy/shards"
TASKS = ["contradiction", "outlier", "qdmatch_nq"]
TOKENIZER = "/data/prasann/hf_models/Qwen3.5-0.8B-Base/tokenizer.json"
HERE = os.path.dirname(os.path.abspath(__file__))


def load(pool):
    meta = json.load(open(f"{SH}/{pool}/metadata.json"))
    info = [json.loads(x) for x in open(f"{SH}/{pool}/row_info.jsonl")]
    parts = {}
    for pt in sorted({x["part"] for x in info}):
        rows = sorted([x for x in info if x["part"] == pt], key=lambda x: x["row"])
        ids = np.fromfile(f"{SH}/{pool}/token_ids_part_t{pt}_000.npy", dtype=meta["dtype"])
        mk = np.fromfile(f"{SH}/{pool}/labels_mask_t{pt}_000.npy", dtype=meta["mask_dtype"])
        off = np.concatenate([[0], np.cumsum([x["len"] for x in rows])])
        assert off[-1] == ids.size == mk.size, (pool, pt, off[-1], ids.size)
        parts[pt] = (rows, ids, mk, off)
    return meta, parts, json.load(open(f"{SH}/{pool}/gold_fingerprints.json")), \
        json.load(open(f"{SH}/{pool}/gold_pairs_fingerprints.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="kl16k")
    ap.add_argument("--src", default="mtb16kx16")
    ap.add_argument("--rung", default="16k")
    ap.add_argument("--rows", default="7200:kl16k_x6,3840:kl16k_x3p2,2880:kl16k_x2p4")
    ap.add_argument("--outlier-hashes", default=f"{HERE}/kl_heldout_outlier_passage_hashes.json")
    a = ap.parse_args()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(TOKENIZER)
    bad = set(json.load(open(a.outlier_hashes)))
    norm = lambda t: re.sub(r"\s+", " ", t).strip()  # noqa: E731
    bmeta, bparts, bgold, bpairs = load(a.base)
    smeta, sparts, sgold, spairs = load(a.src)
    DS, DE = smeta["doc_start_id"], smeta["doc_end_id"]
    cand = {}  # task -> list of (fingerprint, ids, mask, info)
    stats = {"base": a.base, "src": a.src, "rung": a.rung, "tasks": {}}
    for ti, t in enumerate(TASKS):
        rows, ids, mk, off = bparts[ti]
        assert {x["task"] for x in rows} == {t}, (ti, t)
        base_rows = [(x["fingerprint"], ids[off[i]:off[i + 1]], mk[off[i]:off[i + 1]], x) for i, x in enumerate(rows)]
        base_fp = {x[0] for x in base_rows}
        spt = [p for p, v in sparts.items() if v[0] and v[0][0]["task"] == t]
        assert len(spt) == 1, (t, spt)
        rows2, ids2, mk2, off2 = sparts[spt[0]]
        extra, d = [], {"src_rows_at_rung": 0, "in_base": 0, "outlier_heldout_passage": 0}
        for i, x in enumerate(rows2):
            if x["rung"] != a.rung:
                continue
            d["src_rows_at_rung"] += 1
            if x["fingerprint"] in base_fp:
                d["in_base"] += 1
                continue
            r, m = ids2[off2[i]:off2[i + 1]], mk2[off2[i]:off2[i + 1]]
            if t == "outlier":
                st, en = np.flatnonzero(r == DS), np.flatnonzero(r == DE)
                hs = set()
                for s, e in zip(st, en):
                    txt = re.sub(r"^\s*Document \[\d+\]:\s*", "", tok.decode([int(v) for v in r[s + 1:e]], skip_special_tokens=False))
                    hs.add(hashlib.sha1(norm(txt).encode()).hexdigest())
                if hs & bad:
                    d["outlier_heldout_passage"] += 1
                    continue
            extra.append((x["fingerprint"], r, m, x))
        assert d["in_base"] == len(base_rows), (t, d, len(base_rows))
        d["base_rows"], d["extra_eligible"] = len(base_rows), len(extra)
        stats["tasks"][t] = d
        cand[t] = base_rows + extra
        print(f"[grow] {t}: {d}", flush=True)
    for spec in a.rows.split(","):
        n, name = spec.split(":")
        n = int(n)
        out, tmp = f"{SH}/{name}", f"{SH}/{name}_tmp"
        if os.path.exists(f"{out}/metadata.json"):
            print(f"[grow] {out} exists, skip", flush=True)
            continue
        os.system(f"rm -rf {tmp}")
        os.makedirs(tmp)
        info_f = open(f"{tmp}/row_info.jsonl", "w")
        g2, p2, contra, lens, merged = {}, {}, [], [], []
        tot = loss = 0
        for ti, t in enumerate(TASKS):
            sel = cand[t][:n]
            assert len(sel) == n, (t, len(sel), n)
            bt = np.concatenate([x[1] for x in sel]).astype(bmeta["dtype"])
            bm = np.concatenate([x[2] for x in sel]).astype(bmeta["mask_dtype"])
            bt.tofile(f"{tmp}/token_ids_part_t{ti}_000.npy")
            bm.tofile(f"{tmp}/labels_mask_t{ti}_000.npy")
            tot += bt.size
            loss += int(bm.sum())
            for k, (fp, r, m, x) in enumerate(sel):
                y = dict(x, part=ti, row=k, from_pool=a.base if k < stats["tasks"][t]["base_rows"] else a.src)
                info_f.write(json.dumps(y) + "\n")
                lens.append(int(len(r)))
                g2[fp] = bgold.get(fp, sgold.get(fp))
                pr = bpairs.get(fp, spairs.get(fp))
                if pr is not None:
                    p2[fp] = pr
                if t == "contradiction":
                    contra.append(fp)
            merged.append({"task": t, "rows": n, "from_base": stats["tasks"][t]["base_rows"]})
        info_f.close()
        assert all(v is not None for v in g2.values())
        meta = dict(bmeta)
        meta.update({"task": name, "num_instances": len(lens), "num_tokens": int(tot), "num_loss_tokens": loss,
                     "max_example_len": int(max(lens)), "min_example_len": int(min(lens)), "grown_from": [a.base, a.src],
                     "merged_from": merged, "tasks_order": TASKS})
        json.dump(meta, open(f"{tmp}/metadata.json", "w"), indent=1)
        json.dump(g2, open(f"{tmp}/gold_fingerprints.json", "w"))
        json.dump(p2, open(f"{tmp}/gold_pairs_fingerprints.json", "w"))
        json.dump(contra, open(f"{tmp}/gdd_only_contradiction.json", "w"))
        st = dict(stats, pool=name, rows_per_task=n, num_instances=len(lens), num_tokens=int(tot),
                  mean_row_len=float(np.mean(lens)), max_example_len=int(max(lens)))
        json.dump(st, open(f"{SH}/{name}_stats.json", "w"), indent=1)
        os.rename(tmp, out)
        print(f"[grow] wrote {out}: {len(lens)} rows, {tot} tokens, mean len {np.mean(lens):.0f}, max {max(lens)}", flush=True)


if __name__ == "__main__":
    main()
