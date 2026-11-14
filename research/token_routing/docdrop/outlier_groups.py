"""Recover each outlier passage's source article (its topic group) for a mt4 shard. Runs on a node (~100 GB RAM).

The outlier generator (ctc/src/ctc/tasks/outlier/generate.py) builds each row from whole topic groups: every
majority article contributes >= num_outliers + 1 passages and the outlier article exactly num_outliers, so "the
outlier" is the uniquely SMALLEST group. Random document dropping breaks that (majority groups shrink below the
outlier group, which then becomes the largest), so the group-aware drop keeps whole articles. Titles are not
rendered (use_titles False), so this maps every decoded passage back to its article through the seed pool's
``(title, [passages])`` lists, by the passage's first 160 characters.

Writes ``{"rows": [[group id per document] per outlier row], "titles": [...], "stats": {...}}``; group ids are
indices into ``titles``. Refuses a row with an unmatched passage; rows that break the invariant (gold not exactly one article, a
majority group not larger than the outlier group) are listed under ``violations``.

    $PY research/token_routing/docdrop/outlier_groups.py --shard /data/prasann/oracle/shards/mt4f3904 \\
        --seed-pool /data/prasann/ctc_seed_pools/outlier.seed.jsonl.gz --out /data/prasann/docdrop/outlier_groups_f3904.json
"""
import argparse
import gzip
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import docdrop as DD  # noqa: E402

TOKENIZER = "/data/prasann/hf_models/Qwen3.5-4B-Base/tokenizer.json"
KEY = 160
_HEAD = re.compile(r"^\s*Document \[(\d+)\]:\s?")


def key(text):
    return " ".join(text.split())[:KEY]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--seed-pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", type=int, default=2, help="outlier part index in the merged shard")
    a = ap.parse_args()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(TOKENIZER)
    meta = json.load(open(os.path.join(a.shard, "metadata.json")))
    assert meta["merged_from"][a.part]["task"] == "outlier", meta["merged_from"][a.part]
    DS, DE, EOS = meta["doc_start_id"], meta["doc_end_id"], meta["eos_token_id"]
    ids = np.fromfile(f"{a.shard}/token_ids_part_t{a.part}_000.npy", dtype=meta.get("dtype", "uint32"))
    msk = np.fromfile(f"{a.shard}/labels_mask_t{a.part}_000.npy", dtype=meta.get("mask_dtype", "bool"))
    ends = np.flatnonzero(ids == EOS)
    starts = np.concatenate([[0], ends[:-1] + 1])
    t0 = time.time()
    rows = []
    need = set()
    for s, e in zip(starts, ends + 1):
        row = ids[s:e]
        _, spans, _ = DD.split_row(row, DS, DE)
        texts = []
        for a_, b in spans:
            t = tok.decode(row[a_ + 1:b - 1].tolist(), skip_special_tokens=False)
            texts.append(key(_HEAD.sub("", t, count=1)))
        gold = [g - 1 for g in DD.gold_ids_from_answer(tok.decode(row[msk[s:e]].tolist()), "outlier")]
        rows.append((texts, gold))
        need.update(texts)
    print(f"[groups] decoded {len(rows)} rows, {len(need)} distinct passage keys ({time.time() - t0:.0f}s)", flush=True)
    with gzip.open(a.seed_pool, "rt") as f:
        header = json.loads(f.readline())
        body = json.loads(f.readline())
    print(f"[groups] seed pool {header.get('ladder')} / {header.get('pool')}: {len(body['articles'])} articles "
          f"({time.time() - t0:.0f}s)", flush=True)
    k2t = {}
    for title, passages in body["articles"]:
        for p in passages:
            k = key(p)
            if k in need:
                k2t.setdefault(k, set()).add(title)
    del body
    clash = sum(1 for v in k2t.values() if len(v) > 1)
    titles, tid = [], {}
    out_rows, violations = [], []
    min_major, gold_sizes = [], []
    for r, (texts, gold) in enumerate(rows):
        miss = [i for i, k in enumerate(texts) if k not in k2t]
        if miss:
            raise SystemExit(f"row {r}: {len(miss)} passages not found in the seed pool, e.g. {texts[miss[0]]!r}")
        # a key shared by several articles (identical opening text): take the candidate most of the row agrees on
        votes = {}
        for k in texts:
            if len(k2t[k]) == 1:
                t = next(iter(k2t[k]))
                votes[t] = votes.get(t, 0) + 1
        names = [max(sorted(k2t[k]), key=lambda t: votes.get(t, 0)) for k in texts]
        g = []
        for t in names:
            if t not in tid:
                tid[t] = len(titles)
                titles.append(t)
            g.append(tid[t])
        counts = {}
        for x in g:
            counts[x] = counts.get(x, 0) + 1
        gold_groups = {g[i] for i in gold}
        problem = None
        if len(gold_groups) != 1:
            problem = f"gold spans {len(gold_groups)} articles"
        else:
            gg = next(iter(gold_groups))
            major = [c for x, c in counts.items() if x != gg]
            if sorted(i for i, x in enumerate(g) if x == gg) != sorted(gold):
                problem = "a non-gold passage shares the outlier article"
            elif min(major) <= counts[gg]:
                problem = f"smallest majority group {min(major)} <= outlier group {counts[gg]}"
            else:
                min_major.append(min(major))
                gold_sizes.append(counts[gg])
        if problem:
            violations.append({"row": r, "problem": problem,
                               "groups": {titles[x]: c for x, c in counts.items()},
                               "gold_titles": [titles[g[i]] for i in gold]})
        out_rows.append(g)
    stats = {"rows": len(out_rows), "key_clashes": clash, "rows_violating_invariant": len(violations), "min_majority_group": DD.summarize(min_major),
             "outlier_group_size": DD.summarize(gold_sizes),
             "majority_groups_per_row": DD.summarize([len(set(g)) - 1 for g in out_rows])}
    json.dump({"rows": out_rows, "titles": titles, "stats": stats, "shard": a.shard, "violations": violations}, open(a.out, "w"))
    print(json.dumps(stats, indent=1))
    for v in violations[:5]:
        print("[violation]", json.dumps(v)[:600])
    print(f"[groups] wrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
