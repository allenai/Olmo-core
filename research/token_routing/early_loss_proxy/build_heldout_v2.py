"""Build the v2 held-out probe rows for the early dense-loss proxy (pool ``mtb16k``, build_pool_v2.py). Runs on a node.

Sets ``suite4k`` / ``suite8k`` / ``suite16k``: per task (contradiction, outlier, reorder, qdmatch_nq) the FIRST rows of
the CTC-suite eval rung files that the final generation-graded evals read (HF ``PrasannSinghal/ctc-suite-eval`` at
olmo-eval's pinned revision, fetched by ``ds64_fast8k/fetch_suite_eval_rows.py`` into
``/data/prasann/early_loss_proxy/v2build/suite_rows/suite/<subset>/rung_<N>.jsonl``; contradiction = the suite's
``contradiction_iid`` row), tokenized exactly like the training rows (``convert_unified_to_document_landmark.py
--emit dense``, qwen3_5 markers, query AFTER, no CoT, documents wrapped), gold answer as the loss span. A row longer
than 16,384 tokens is skipped (the next eval row is taken) and counted; ``src_idx`` is the row's index in the eval file.

Same npz format as build_heldout.py: ``ids, mask, offsets, sets, tasks, src_idx``.

Overlap checks against every training row of the pool:
- prompt hash (tokens before the answer) and question hash (tokens after the last document marker), as v1. The
  question span of these four tasks is a fixed instruction, so the question hash matches by construction; it is
  reported, not a contamination signal.
- document level: per held-out row, the share of its documents (normalized text) found among the training rows'
  documents, and whether any gold document (qdmatch: the query or document of a gold pair; reorder: any passage) or, for
  reorder, the source book occurs in training.

    $PY research/token_routing/early_loss_proxy/build_heldout_v2.py --out /data/prasann/early_loss_proxy/heldout_v2.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_pool_v2 as P  # noqa: E402

SETS = {"4k": "suite4k", "8k": "suite8k", "16k": "suite16k"}


def keys(row, mask):
    a0 = int(np.nonzero(mask)[0][0])
    prompt = row[:a0]
    last_end = np.nonzero(prompt == P.DE)[0]
    q = prompt[int(last_end[-1]) + 1:] if len(last_end) else prompt[-64:]
    h = lambda x: hashlib.sha1(np.ascontiguousarray(x, dtype=np.int64).tobytes()).hexdigest()  # noqa: E731
    return h(prompt), h(q)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default=P.OUT)
    ap.add_argument("--rows", type=int, default=32, help="rows per (task, set)")
    ap.add_argument("--max-scan", type=int, default=500, help="eval rows tokenized per file to find --rows that fit")
    a = ap.parse_args()

    # ---- training rows: token hashes + document content --------------------------------------------------------
    meta = json.load(open(f"{a.pool}/metadata.json"))
    info = [json.loads(x) for x in open(f"{a.pool}/row_info.jsonl")]
    tr_p, tr_q = set(), set()
    n_train = 0
    for t in range(4):
        ids = np.fromfile(f"{a.pool}/token_ids_part_t{t}_000.npy", dtype=meta["dtype"])
        mk = np.fromfile(f"{a.pool}/labels_mask_t{t}_000.npy", dtype=meta["mask_dtype"])
        ends = np.flatnonzero(ids == P.EOS)
        s = 0
        for e in ends:
            p, q = keys(ids[s:e + 1], mk[s:e + 1])
            tr_p.add(p)
            tr_q.add(q)
            s = e + 1
            n_train += 1
    assert n_train == len(info) == meta["num_instances"], (n_train, len(info), meta["num_instances"])
    print(f"[heldout] {n_train} training rows hashed", flush=True)
    tr_docs = {t: set() for t in P.TASKS}
    tr_gold = {t: set() for t in P.TASKS}
    gold_of = {(x["src"], x["src_idx"]): x["gold"] for x in info}
    tr_books = set()
    by_src = {}
    for x in info:
        by_src.setdefault(x["src"], set()).add(x["src_idx"])
    for src, idxs in by_src.items():
        task = next(t for t in P.TASKS if f"/{t}_" in src)
        with open(src) as f:
            for i, line in enumerate(f):
                if i in idxs:
                    ex = json.loads(line)
                    tr_docs[task].update(P.thash(d.get("text")) for d in ex["documents"])
                    tr_gold[task].update(P.thash(ex["documents"][j].get("text")) for j in gold_of[(src, i)])
                    if task == "reorder" and P.book_of(ex):
                        tr_books.add(P.book_of(ex))
    print(f"[heldout] training document hashes: { {t: len(v) for t, v in tr_docs.items()} }", flush=True)

    ids_all, mask_all, offs, sets, tasks, src_idx = [], [], [0], [], [], []
    stats = {"pool": a.pool, "train_rows_hashed": n_train, "sets": {}}
    os.makedirs("/data/prasann/tmp", exist_ok=True)
    with tempfile.TemporaryDirectory(dir="/data/prasann/tmp") as tmp:
        for rung, set_name in SETS.items():
            for task in P.TASKS:
                path = P.eval_rung_path(task, rung)
                exs = P.read_jsonl(path, a.max_scan)
                head = f"{tmp}/{task}_{rung}.jsonl"
                with open(head, "w") as g:
                    for ex in exs:
                        g.write(json.dumps(ex) + "\n")
                d = f"{tmp}/{task}_{rung}"
                os.makedirs(d)
                P.convert(head, P.SPEC[task], d)
                _, rows = P.load_shard_rows(d)
                assert len(rows) == len(exs), f"{path}: {len(rows)} tokenized vs {len(exs)} rows (converter dropped some)"
                picked, skipped_long, scanned = [], [], len(rows)
                for i, ((r, m), ex) in enumerate(zip(rows, exs)):
                    if len(r) > P.MAX_LEN:
                        skipped_long.append(i)
                        continue
                    assert m.any(), f"{path} row {i}: no answer tokens"
                    if len(picked) < a.rows:
                        picked.append((i, r, m, ex))
                hp = hq = n_gold_hit = n_gold_gold = n_book_hit = 0
                shares = []
                for i, r, m, ex in picked:
                    p, q = keys(r, m)
                    hp += p in tr_p
                    hq += q in tr_q
                    g, _ = P.gold_info(task, ex)
                    gs = set(g)
                    hs = [P.thash(dd.get("text")) for dd in ex["documents"]]
                    shares.append(float(np.mean([h in tr_docs[task] for h in hs])))
                    n_gold_hit += any(hs[j] in tr_docs[task] for j in gs)
                    n_gold_gold += any(hs[j] in tr_gold[task] for j in gs)
                    n_book_hit += task == "reorder" and P.book_of(ex) in tr_books
                    ids_all.append(r.astype(np.uint32))
                    mask_all.append(m)
                    offs.append(offs[-1] + len(r))
                    sets.append(set_name)
                    tasks.append(task)
                    src_idx.append(i)
                if not picked:
                    stats["sets"][f"{set_name}/{task}"] = {"rows": 0, "source_file": path, "eval_rows_fitting_16384": 0,
                                                          "scanned_eval_rows": scanned}
                    print(f"[heldout] WARNING {set_name}/{task}: no eval row fits in {P.MAX_LEN}", flush=True)
                    continue
                lens = np.array([len(x[1]) for x in picked])
                ans = np.array([int(x[2].sum()) for x in picked])
                all_lens = np.array([len(r) for r, _ in rows[:scanned]])
                stats["sets"][f"{set_name}/{task}"] = {
                    "rows": len(picked), "source_file": path, "src_idx_max": int(picked[-1][0]) if picked else None,
                    "eval_rows_fitting_16384": scanned - len(skipped_long), "skipped_over_16384": len(skipped_long), "skipped_src_idx": skipped_long[:50],
                    "scanned_eval_rows": scanned, "scanned_len_mean": float(all_lens.mean()),
                    "scanned_len_max": int(all_lens.max()),
                    "n_docs": sorted({len(x[3]["documents"]) for x in picked}),
                    "len_mean": float(lens.mean()), "len_min": int(lens.min()), "len_max": int(lens.max()),
                    "answer_tokens_mean": float(ans.mean()), "answer_tokens_min": int(ans.min()),
                    "answer_tokens_max": int(ans.max()),
                    "prompt_in_train": hp, "question_in_train": hq,
                    "rows_with_gold_doc_in_train": n_gold_hit, "rows_with_gold_doc_gold_in_train": n_gold_gold, "rows_with_book_in_train": n_book_hit,
                    "doc_share_in_train_mean": float(np.mean(shares)), "doc_share_in_train_max": float(np.max(shares))}
                print(f"[heldout] {set_name}/{task}: {len(picked)} rows ({scanned - len(skipped_long)}/{scanned} eval rows fit, {len(skipped_long)} > "
                      f"{P.MAX_LEN}), len {lens.mean():.0f} [{lens.min()}, {lens.max()}], answer {ans.mean():.1f} tok; "
                      f"prompt in train {hp}, question in train {hq}, gold doc in train {n_gold_hit} (as train gold {n_gold_gold}), "
                      f"book in train {n_book_hit}, doc share in train {np.mean(shares):.4f} (max {np.max(shares):.4f})",
                      flush=True)
                if len(picked) < a.rows:
                    print(f"[heldout] WARNING {set_name}/{task}: only {len(picked)} rows fit in {P.MAX_LEN}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, ids=np.concatenate(ids_all), mask=np.concatenate(mask_all), offsets=np.asarray(offs),
             sets=np.asarray(sets), tasks=np.asarray(tasks), src_idx=np.asarray(src_idx))
    json.dump(stats, open(a.out.replace(".npz", "_stats.json"), "w"), indent=1)
    print(f"[heldout] wrote {a.out} ({len(sets)} rows)", flush=True)


if __name__ == "__main__":
    main()
