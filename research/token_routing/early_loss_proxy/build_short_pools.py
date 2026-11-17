"""Short-row pools for the early dense-loss proxy (POOL_V2.md, "Short-row pools"). Runs on a node (mooney).

Two new cells per task (``1k`` ~1,000 tokens/row, ``2k`` ~2,000) built exactly like ``mtb16k``'s cells
(build_pool_v2.py: ctc ``--split train`` seed 42 via ctc_calibrated.py, same converter flags, same exclusion rules),
then three pools in mtb16k's layout, 1,952 rows per task:

- ``mtbshort``: 976 x 1k + 976 x 2k
- ``mtbmix80``: 781 x 1k + 781 x 2k + 130 / 130 / 130 rows of mtb16k's 4k / 8k / 16k rows
- ``mtbmix50``: 488 x 1k + 488 x 2k + 325 / 325 / 326 rows of mtb16k's 4k / 8k / 16k rows

Documents per row: the eval calibration's own count where it has one (2k), else the tokens-per-document line fit
through the tokenized 4k / 8k cells (1k). Exclusion: a short row is dropped if any gold document (qdmatch: the query
or document of a gold pair; reorder: any passage, or the source book) occurs in ANY eval rung file (the suite's
2k / 4k / 8k / 16k rungs; the 2k files are fetched into ``v2build/suite_rows_r2k``). Long rows are mtb16k's own rows
(filtered against the 4k-16k eval rungs when mtb16k was built), taken in mtb16k's (shuffled) order.

    $PY build_short_pools.py gen ; $PY build_short_pools.py tokenize ; $PY build_short_pools.py pools
    $PY build_pool_v2.py order --out <pool>   # per pool
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_pool_v2 as P  # noqa: E402

SHORT = ["1k", "2k"]
LONG = ["4k", "8k", "16k"]
TARGET = {"1k": 1000.0, "2k": 2000.0}
#: rows generated per (task, rung). qdmatch_nq: ~32% survive the eval-gold exclusion. The 1k cells were regenerated
#: larger (2026-11-17, same seed: a prefix-stable stream) for mtbmix80h, which needs 1,562 rows at 1k per task.
#: Regenerated again (same seed, prefix-stable) for mtbmix80x3, which needs 2,342 / 2,343 rows at 1k / 2k per task.
GEN_ROWS = {("qdmatch_nq", "2k"): 8500, ("qdmatch_nq", "1k"): 8500}
for _t in ("contradiction", "outlier", "reorder"):
    GEN_ROWS[(_t, "1k")] = GEN_ROWS[(_t, "2k")] = 2700
GEN_DEFAULT = 1000
SHORT_N_DOCS_FILE = f"{P.WORK}/n_docs_short.json"
R2K = f"{P.WORK}/suite_rows_r2k/suite"
R2K_FILE = {"contradiction": "contradiction_iid/rung_2560.jsonl", "outlier": "outlier/rung_2048.jsonl",
            "reorder": "reorder/rung_2048.jsonl", "qdmatch_nq": "qdmatch_nq/rung_2048.jsonl"}
POOLS = {
    "mtbshort": {"1k": 976, "2k": 976},
    "mtbmix80": {"1k": 781, "2k": 781, "4k": 130, "8k": 130, "16k": 130},
    "mtbmix50": {"1k": 488, "2k": 488, "4k": 325, "8k": 325, "16k": 326},
    "mtbmix90": {"1k": 878, "2k": 878, "4k": 65, "8k": 65, "16k": 66},
    "mtbmix95": {"1k": 927, "2k": 927, "4k": 33, "8k": 33, "16k": 32},
    "mtbmix80L": {"1k": 781, "2k": 781, "16k": 390},
    "mtbmix80h": {"1k": 1562, "4k": 130, "8k": 130, "16k": 130},
    "mtbmix80x3": {"1k": 2342, "2k": 2343, "4k": 390, "8k": 390, "16k": 391},
}
SH = "/data/prasann/early_loss_proxy/shards"


def eval_files(task):
    return [f"{R2K}/{R2K_FILE[task]}"] + [P.eval_rung_path(task, r) for r in LONG]


def n_docs_for(task, rung, cal):
    if rung in cal[task]:
        return int(cal[task][rung]["n_docs"]), {"why": "eval calibration"}
    n, fit = P.refit_n_docs(task, cal, TARGET[rung])
    n_r = max(1, int(round((TARGET[rung] - fit["overhead"]) / fit["tok_per_doc"])))
    if task == "qdmatch_nq" and n_r % 2:  # separate layout: as many queries as documents
        n_r += 1
    if task == "outlier":  # 3 outliers need a strictly larger majority
        n_r = max(n_r, 7)
    return n_r, {"why": "tokens-per-document fit", "target_mean": TARGET[rung], **fit}


def gen(a):
    cal = json.load(open(f"{P.REPO}/recipes/data/ctc_sft/eval_calibration.json"))["tasks"]
    used = {}
    jobs = []
    for t in P.TASKS:
        for r in SHORT:
            n, why = n_docs_for(t, r, cal)
            used[f"{t}:{r}"] = {"n_docs": n, **why}
            jobs.append((t, r, n, GEN_ROWS.get((t, r), GEN_DEFAULT)))
            print(f"[short gen] {t}@{r}: {n} documents ({why['why']}), rows {jobs[-1][3]}", flush=True)
    json.dump(used, open(SHORT_N_DOCS_FILE, "w"), indent=1)
    with cf.ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(P._gen_cell, jobs):
            print(f"[short gen] {r[0]}@{r[1]} {r[2]} rows={r[3]} ({r[4]:.0f}s)", flush=True)


def tokenize(a):
    with cf.ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(P._tok_cell, [(t, r) for t in P.TASKS for r in SHORT]):
            print(f"[short tok] {r[0]}@{r[1]} {r[2]}", flush=True)


def short_rows(task, rung, eg, eb):
    """Eligible rows of a short cell, in generation order: (ids, mask, info, ex) + drop counts."""
    src = f"{P.cell_dir(task, rung)}/{task}/train.jsonl"
    meta, rows = P.load_shard_rows(f"{P.WORK}/tok/{task}_{rung}")
    exs = P.read_jsonl(src)
    assert len(rows) == len(exs) and meta["num_dropped"] == 0, (task, rung, len(rows), len(exs))
    kept, d = [], {"generated": len(exs), "over_16384": 0, "gold_in_eval": 0, "book_in_eval": 0}
    for i, ((ids, m), ex) in enumerate(zip(rows, exs)):
        g, pairs = P.gold_info(task, ex)
        if len(ids) > P.MAX_LEN:
            d["over_16384"] += 1
            continue
        if any(P.thash(ex["documents"][j].get("text")) in eg for j in g):
            d["gold_in_eval"] += 1
            continue
        if task == "reorder" and P.book_of(ex) in eb:
            d["book_in_eval"] += 1
            continue
        kept.append((ids, m, {"rung": rung, "src": src, "src_idx": i, "gold": g, "pairs": pairs}, ex))
    d["eligible"] = len(kept)
    return kept, d


def long_rows(task):
    """mtb16k's rows of ``task`` per rung, in mtb16k's order."""
    pool = f"{SH}/mtb16k"
    meta = json.load(open(f"{pool}/metadata.json"))
    t = P.TASKS.index(task)
    ids = np.fromfile(f"{pool}/token_ids_part_t{t}_000.npy", dtype=meta["dtype"])
    msk = np.fromfile(f"{pool}/labels_mask_t{t}_000.npy", dtype=meta["mask_dtype"])
    info = [json.loads(x) for x in open(f"{pool}/row_info.jsonl")]
    info = [x for x in info if x["part"] == t]
    ends = np.flatnonzero(ids == P.EOS)
    starts = np.concatenate([[0], ends[:-1] + 1])
    assert len(ends) == len(info)
    out = {r: [] for r in LONG}
    for x, s, e in zip(info, starts, ends):
        assert e + 1 - s == x["len"]
        out[x["rung"]].append((ids[s:e + 1], msk[s:e + 1],
                               {"rung": x["rung"], "src": x["src"], "src_idx": x["src_idx"], "gold": x["gold"],
                                "pairs": x["gold_pairs"]}, None))
    return out


def gen16k_keys(task):
    """Gold hashes / reorder books of heldout_v3's gen16k rows (source JSONL rows named by src_idx)."""
    import build_heldout_v3 as H

    h = np.load("/data/prasann/early_loss_proxy/heldout_v3.npz")
    idx = {int(i) for i, s_, t_ in zip(h["src_idx"], h["sets"], h["tasks"]) if s_ == "gen16k" and t_ == task}
    src = H.CONTRA_EVAL if task == "contradiction" else f"{H.gcell(task)}/{task}/train.jsonl"
    g, b = set(), set()
    with open(src) as f:
        for i, line in enumerate(f):
            if i in idx:
                ex = json.loads(line)
                g |= H.gold_hashes(task, ex)
                if task == "reorder" and P.book_of(ex):
                    b.add(P.book_of(ex))
    assert len(idx) == 48, (task, len(idx))
    return g, b


def pools(a):
    from tokenizers import Tokenizer

    sys.path.insert(0, f"{P.REPO}/src")
    from olmo_core.nn.attention.gold_grad_mask import content_fingerprint

    tok = Tokenizer.from_file(f"{P.TOKENIZER}/tokenizer.json")
    dec = lambda x: tok.decode([int(i) for i in x], skip_special_tokens=False)  # noqa: E731
    n_docs_short = json.load(open(SHORT_N_DOCS_FILE))
    n_docs_long = json.load(open(P.N_DOCS_FILE))
    cells, drops, longs = {}, {}, {}
    for task in P.TASKS:
        eg, eb = set(), set()
        for f in eval_files(task):
            g2, _, b2 = P.eval_content_keys(task, P.read_jsonl(f))
            eg |= g2
            eb |= b2
        # heldout_v3's gen16k rows are held-out too (added 2026-11-17 for mtbmix80x3; earlier pools predate gen16k,
        # which was itself built disjoint from them)
        g3, b3 = gen16k_keys(task)
        eg |= g3
        eb |= b3
        for r in SHORT:
            cells[(task, r)], drops[f"{task}/{r}"] = short_rows(task, r, eg, eb)
            print(f"[pools] {task}@{r}: {json.dumps(drops[f'{task}/{r}'])}", flush=True)
        longs[task] = long_rows(task)
    for name in a.pools.split(","):
        spec = POOLS[name]
        final, tmp = f"{SH}/{name}", f"{SH}/{name}_tmp"
        if os.path.exists(f"{final}/metadata.json"):
            print(f"[pools] {final} exists, skipping", flush=True)
            continue
        os.system(f"rm -rf {tmp}")
        os.makedirs(tmp)
        stats = {"pool": final, "spec_rows_per_task": spec, "shuffle_seed": a.shuffle_seed, "max_len": P.MAX_LEN,
                 "n_docs_used": {**{k: v for k, v in n_docs_long.items()}, **n_docs_short},
                 "short_cell_drops": drops, "eval_files_excluded": {t: eval_files(t) for t in P.TASKS}, "tasks": {}}
        gold_side, pair_side, merged_from, lens_all = {}, {}, [], []
        tot_tok = tot_loss = 0
        info_f = open(f"{tmp}/row_info.jsonl", "w")
        for t, task in enumerate(P.TASKS):
            rows = []
            stats["tasks"][task] = {}
            for r, k in spec.items():
                src = cells[(task, r)] if r in SHORT else longs[task][r]
                if len(src) < k:
                    raise SystemExit(f"{name}: {task}@{r} has {len(src)} rows < {k}")
                sel = src[:k]
                L = np.array([len(x[0]) for x in sel])
                A = np.array([int(x[1].sum()) for x in sel])
                stats["tasks"][task][r] = {"rows": k, "len_mean": float(L.mean()), "len_min": int(L.min()),
                                           "len_max": int(L.max()), "tokens": int(L.sum()),
                                           "answer_tokens_mean": float(A.mean()),
                                           "n_docs": (n_docs_short.get(f"{task}:{r}") or n_docs_long.get(f"{task}:{r}"))["n_docs"],
                                           "source": "new short cell" if r in SHORT else f"{SH}/mtb16k rows"}
                rows += sel
            perm = np.random.default_rng(a.shuffle_seed + t).permutation(len(rows))
            rows = [rows[j] for j in perm]
            ver = {"rows_checked": 0, "span_text_match": 0, "printed_id_is_index_plus_1": 0, "answer_ids_match_gold": 0}
            ids_out, msk_out = [], []
            for i, (ids, m, inf, ex) in enumerate(rows):
                spans = P.doc_spans(ids)
                fp = content_fingerprint(ids.tolist())
                assert fp not in gold_side, f"duplicate fingerprint {task} row {i}"
                gold_side[fp] = inf["gold"]
                if inf["pairs"] is not None:
                    pair_side[fp] = inf["pairs"]
                if ex is not None and ver["rows_checked"] < a.verify_rows:
                    n = len(ex["documents"])
                    assert len(spans) == n
                    ok_text = all(P.norm(ex["documents"][j].get("text"))[:200] in P.norm(dec(ids[spans[j][0] + 1:spans[j][1] - 1]))
                                  for j in set(inf["gold"]) | {0})
                    pid = [P.printed_id(dec(ids[s + 1:e - 1])[:40]) for s, e in spans]
                    nums = [int(x) for x in __import__("re").findall(r"\d+", dec(ids[m]))]
                    if task == "reorder":
                        ok_ans = nums == [int(x) for x in ex["gold_order"]]
                    elif inf["pairs"] is not None:
                        ok_ans = sorted(nums) == sorted(x + 1 for p in inf["pairs"] for x in p)
                    else:
                        ok_ans = sorted(nums) == [x + 1 for x in inf["gold"]]
                    ver["rows_checked"] += 1
                    ver["span_text_match"] += ok_text
                    ver["printed_id_is_index_plus_1"] += pid == list(range(1, n + 1))
                    ver["answer_ids_match_gold"] += ok_ans
                info_f.write(json.dumps({"part": t, "row": i, "task": task, "rung": inf["rung"], "src": inf["src"],
                                         "src_idx": inf["src_idx"], "n_docs": len(spans), "len": int(len(ids)),
                                         "answer_tokens": int(m.sum()), "fingerprint": fp, "gold": inf["gold"],
                                         "gold_pairs": inf["pairs"], "doc_spans": spans}) + "\n")
                ids_out.append(ids)
                msk_out.append(m)
                lens_all.append(len(ids))
            stats["tasks"][task]["gold_verification_short_rows"] = ver
            bt = np.concatenate(ids_out).astype("uint32")
            bm = np.concatenate(msk_out).astype("bool")
            bt.tofile(f"{tmp}/token_ids_part_t{t}_000.npy")
            bm.tofile(f"{tmp}/labels_mask_t{t}_000.npy")
            tot_tok += bt.size
            tot_loss += int(bm.sum())
            merged_from.append({"src": [f"{P.WORK}/tok/{task}_{r}" if r in SHORT else f"{SH}/mtb16k" for r in spec],
                                "task": task, "converter_task": P.SPEC[task], "num_instances": len(rows),
                                "rung_counts": dict(spec)})
            print(f"[pools] {name} t{t} {task}: {len(rows)} rows; verify {json.dumps(ver)}", flush=True)
        info_f.close()
        meta = json.load(open(f"{SH}/mtb16k/metadata.json"))
        meta.update({"task": name, "num_instances": len(lens_all), "num_tokens": int(tot_tok),
                     "num_loss_tokens": int(tot_loss), "max_example_len": int(max(lens_all)),
                     "min_example_len": int(min(lens_all)), "merged_from": merged_from,
                     "shuffle_seed": a.shuffle_seed, "rows_per_rung": dict(spec), "rows_over_max_len_dropped": 0})
        meta.pop("rows_per_rung_target", None)
        json.dump(gold_side, open(f"{tmp}/gold_fingerprints.json", "w"))
        json.dump(pair_side, open(f"{tmp}/gold_pairs_fingerprints.json", "w"))
        stats.update({"num_instances": len(lens_all), "num_tokens": int(tot_tok), "mean_row_len": float(np.mean(lens_all)),
                      "max_example_len": int(max(lens_all))})
        json.dump(stats, open(f"{final}_stats.json", "w"), indent=1)
        json.dump(meta, open(f"{tmp}/metadata.json", "w"), indent=1)
        os.rename(tmp, final)
        print(f"[pools] wrote {final}: {len(lens_all)} rows, {tot_tok} tokens, mean {np.mean(lens_all):.0f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["gen", "tokenize", "pools"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--pools", default="mtbshort,mtbmix80,mtbmix50")
    ap.add_argument("--shuffle-seed", type=int, default=20261117)
    ap.add_argument("--verify-rows", type=int, default=20)
    a = ap.parse_args()
    {"gen": gen, "tokenize": tokenize, "pools": pools}[a.stage](a)


if __name__ == "__main__":
    main()
