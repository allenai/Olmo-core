"""Which topic groups a model's outlier answers come from, on an outlier eval rung (records/docdrop-nongold-baseline.md).
Aligns each response to its rung row by gold tuple (the eval scores a shuffled
500-row subset). Maps every eval passage to its seed-pool article (as outlier_groups.py) and, per run, reports the share of predicted
IDs in the gold (outlier) article vs a majority article, how often all of a row's predictions come from ONE majority
article, and that article's size rank among the row's majority articles. Runs on a node (~100 GB RAM).

    $PY research/token_routing/docdrop/outlier_eval_groups.py --rung /data/prasann/f8k_eval/eval/outlier_lengthmix/eval_rungs/outlier/rung_8192.jsonl \\
        --seed-pool /data/prasann/ctc_seed_pools/outlier.seed.jsonl.gz --responses <run>=<responses.jsonl> ...
"""
import argparse
import gzip
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outlier_groups import key  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True)
    ap.add_argument("--seed-pool", required=True)
    ap.add_argument("--responses", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    ex = [json.loads(l) for l in open(a.rung)]
    need = {key(d["text"]) for e in ex for d in e["documents"]}
    with gzip.open(a.seed_pool, "rt") as f:
        f.readline()
        body = json.loads(f.readline())
    k2t = {}
    for title, passages in body["articles"]:
        for p in passages:
            k = key(p)
            if k in need:
                k2t.setdefault(k, set()).add(title)
    del body
    groups = []
    for e in ex:
        ks = [key(d["text"]) for d in e["documents"]]
        votes = {}
        for k in ks:
            if len(k2t.get(k, ())) == 1:
                t = next(iter(k2t[k]))
                votes[t] = votes.get(t, 0) + 1
        groups.append([max(sorted(k2t[k]), key=lambda t: votes.get(t, 0)) if k in k2t else None for k in ks])
    out = {"unmapped_passages": sum(g.count(None) for g in groups)}
    for spec in a.responses:
        name, path = spec.split("=", 1)
        resp = [json.loads(l) for l in open(path)]
        # the eval scores a shuffled subset of the rung: align each response to its rung row by its gold tuple
        by_gold = {}
        for j, e in enumerate(ex):
            by_gold.setdefault(tuple(e["gold_doc_indices"]), []).append(j)
        ambiguous = sum(1 for r in resp if len(by_gold.get(tuple(r["gold_doc_indices"]), [])) != 1)
        resp = [r for r in resp if len(by_gold.get(tuple(r["gold_doc_indices"]), [])) == 1]
        n_pred = in_gold = in_major = one_major = 0
        size_rank = []
        for r in resp:
            i = by_gold[tuple(r["gold_doc_indices"])][0]
            g, gold = groups[i], set(ex[i]["gold_doc_indices"])
            txt = r["response"].split("Outliers:", 1)[1] if "Outliers:" in r["response"] else ""
            pred = [int(x) - 1 for x in re.findall(r"\d+", txt) if 0 < int(x) <= len(g)]
            n_pred += len(pred)
            in_gold += sum(p in gold for p in pred)
            maj = [g[p] for p in pred if p not in gold]
            in_major += len(maj)
            gold_t = {g[j] for j in gold}
            if pred and len(maj) == len(pred) and len(set(maj)) == 1:
                one_major += 1
                counts = {}
                for t in g:
                    if t not in gold_t:
                        counts[t] = counts.get(t, 0) + 1
                sizes = sorted(set(counts.values()))
                size_rank.append({"group_size": counts[maj[0]], "min_majority": min(counts.values()),
                                  "covers_whole_group": len(set(pred)) == counts[maj[0]]})
        out[name] = {"rows": len(resp), "rows_skipped_ambiguous": ambiguous, "pred_ids": n_pred, "share_in_outlier_article": in_gold / max(1, n_pred),
                     "share_in_majority_articles": in_major / max(1, n_pred),
                     "rows_all_preds_one_majority_article": one_major / max(1, len(resp)),
                     "of_those_group_is_a_smallest_majority": (sum(x["group_size"] == x["min_majority"] for x in size_rank)
                                                               / max(1, len(size_rank))),
                     "of_those_preds_cover_whole_group": (sum(x["covers_whole_group"] for x in size_rank)
                                                          / max(1, len(size_rank)))}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
