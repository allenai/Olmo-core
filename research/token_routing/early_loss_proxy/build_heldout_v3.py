"""Held-out probe rows v3 (POOL_V2.md, "Held-out v3"): more and cleaner 16k rows. Runs on a node (mooney).

Sets (npz format of build_heldout.py: ``ids, mask, offsets, sets, tasks, src_idx``):
- ``gen16k`` (contradiction: from ctc's EVAL split, see CONTRA_EVAL): 48 rows per task of FRESH rows from the training generator at the training 16k cell sizes
  (n_docs_used.json: contradiction 324, outlier 96, reorder 90, qdmatch_nq 174), ``--split train`` with seed 7777
  (every training cell used seed 42; ctc keys its streams by (seed, split, rung)), rows <= 16,384 tokens, and none
  sharing a gold document (qdmatch: the query or document of a gold pair) or a reorder book with ANY row used by any
  training pool (union of the pools' row_info.jsonl). ``src_idx`` = row index in the generated file.
- ``rung16k``: the first 32 rows per task of the suite's 16k rung files (heldout_v2's source files), converted the same
  way, capped only at 40,960 tokens (rows over the cap are skipped and counted). ``src_idx`` = eval row index.
- ``suite16k`` / ``suite8k`` / ``suite4k``: copied unchanged from heldout_v2.npz.
Overlap checks against every row of every training pool: prompt hash (tokens before the answer) and document level
(any document / any gold document of the held-out row found among training rows' documents / gold documents).

    $PY build_heldout_v3.py gen ; $PY build_heldout_v3.py build --out /data/prasann/early_loss_proxy/heldout_v3.npz
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_pool_v2 as P  # noqa: E402

SH = "/data/prasann/early_loss_proxy/shards"
POOLS = ["mtb16k", "mtbshort", "mtbmix80", "mtbmix50", "mtbmix90", "mtbmix95", "mtbmix80L", "mtbmix80h"]
HV2 = "/data/prasann/early_loss_proxy/heldout_v2.npz"
GSEED = 7777
GEN_ROWS = {"contradiction": 400, "outlier": 400, "reorder": 1500, "qdmatch_nq": 20000}
GDIR = f"{P.WORK}/gen_heldout16k"
RUNG_CAP = 40960
CONTRA_EVAL = f"{P.WORK}/gen_heldout16k_eval/contradiction/contradiction/eval_16k.jsonl"


def gcell(task):
    return f"{GDIR}/{task}"


def _gen(task):
    n_docs = json.load(open(P.N_DOCS_FILE))[f"{task}:16k"]["n_docs"]
    out = gcell(task)
    target = f"{out}/{task}/train.jsonl"
    if os.path.exists(target) and os.path.getsize(target):
        return task, "cached", n_docs
    cmd = [sys.executable, f"{P.REPO}/recipes/data/ctc_sft/ctc_calibrated.py", "build", "--task", task, "--split", "train",
           "--rungs", "16k", "--train", str(GEN_ROWS[task]), "--seed", str(GSEED), "--out", out, "--pool", P.POOL[task]]
    e = P.env()
    e["CTC_DOCS_OVERRIDE"] = json.dumps({f"{task}:16k": n_docs})
    p = subprocess.run(cmd, env=e, capture_output=True, text=True)
    open(f"{out}.log", "w").write(" ".join(cmd) + "\n" + p.stdout + p.stderr)
    return task, ("built" if p.returncode == 0 and os.path.exists(target) else "FAILED " + p.stderr[-500:]), n_docs


def gen(a):
    os.makedirs(GDIR, exist_ok=True)
    with cf.ProcessPoolExecutor(max_workers=4) as ex:
        for r in ex.map(_gen, P.TASKS):
            print(f"[gen16k] {r}", flush=True)


def keys(row, mask):
    a0 = int(np.nonzero(mask)[0][0])
    return hashlib.sha1(np.ascontiguousarray(row[:a0], dtype=np.int64).tobytes()).hexdigest()


def gold_hashes(task, ex, g=None):
    g = P.gold_info(task, ex)[0] if g is None else g
    return {P.thash(ex["documents"][j].get("text")) for j in g}


def build(a):
    # ---- training side: every row of every pool ------------------------------------------------------------------
    tr_prompt = set()
    used = {}  # (src, src_idx) -> gold indices
    pool_rows = {}
    for p in POOLS:
        d = f"{SH}/{p}"
        meta = json.load(open(f"{d}/metadata.json"))
        n = 0
        for t in range(4):
            ids = np.fromfile(f"{d}/token_ids_part_t{t}_000.npy", dtype=meta["dtype"])
            mk = np.fromfile(f"{d}/labels_mask_t{t}_000.npy", dtype=meta["mask_dtype"])
            ends = np.flatnonzero(ids == P.EOS)
            s = 0
            for e in ends:
                tr_prompt.add(keys(ids[s:e + 1], mk[s:e + 1]))
                s = e + 1
                n += 1
        for line in open(f"{d}/row_info.jsonl"):
            x = json.loads(line)
            used[(x["src"], x["src_idx"])] = x["gold"]
        pool_rows[p] = n
    print(f"[v3] training pools {pool_rows}; {len(tr_prompt)} distinct prompts; {len(used)} distinct source rows", flush=True)
    tr_docs = {t: set() for t in P.TASKS}
    tr_gold = {t: set() for t in P.TASKS}
    tr_books = set()
    by_src = {}
    for (src, i), g in used.items():
        by_src.setdefault(src, {})[i] = g
    for src, idx in by_src.items():
        task = next(t for t in P.TASKS if f"/{t}_" in src)
        with open(src) as f:
            for i, line in enumerate(f):
                if i in idx:
                    ex = json.loads(line)
                    tr_docs[task].update(P.thash(d.get("text")) for d in ex["documents"])
                    tr_gold[task] |= gold_hashes(task, ex, idx[i])
                    if task == "reorder" and P.book_of(ex):
                        tr_books.add(P.book_of(ex))
    print(f"[v3] training gold hashes { {t: len(v) for t, v in tr_gold.items()} }, books {len(tr_books)}", flush=True)

    ids_all, mask_all, offs, sets, tasks, src_idx = [], [], [0], [], [], []
    stats = {"pools_checked": pool_rows, "sets": {}}

    def add(set_name, task, picked, extra):
        hp = hg = hgg = hb = 0
        shares = []
        for i, r, m, ex in picked:
            hp += keys(r, m) in tr_prompt
            if ex is not None:
                gh = gold_hashes(task, ex)
                hg += bool(gh & tr_docs[task])
                hgg += bool(gh & tr_gold[task])
                hb += task == "reorder" and P.book_of(ex) in tr_books
                shares.append(float(np.mean([P.thash(d.get("text")) in tr_docs[task] for d in ex["documents"]])))
            ids_all.append(r.astype(np.uint32))
            mask_all.append(m)
            offs.append(offs[-1] + len(r))
            sets.append(set_name)
            tasks.append(task)
            src_idx.append(i)
        L = np.array([len(x[1]) for x in picked])
        A = np.array([int(x[2].sum()) for x in picked])
        st = {"rows": len(picked), "len_mean": float(L.mean()), "len_min": int(L.min()), "len_max": int(L.max()),
              "len_p50": float(np.percentile(L, 50)), "len_p90": float(np.percentile(L, 90)),
              "rows_over_16384": int((L > P.MAX_LEN).sum()), "answer_tokens_mean": float(A.mean()),
              "n_docs": sorted({len(x[3]["documents"]) for x in picked if x[3] is not None}),
              "prompt_in_train": hp, "rows_with_gold_doc_in_train_docs": hg, "rows_with_gold_doc_in_train_gold": hgg,
              "rows_with_book_in_train": hb, "doc_share_in_train_mean": float(np.mean(shares)) if shares else None, **extra}
        stats["sets"][f"{set_name}/{task}"] = st
        print(f"[v3] {set_name}/{task}: {json.dumps({k: v for k, v in st.items() if k not in ('skipped_src_idx',)})}", flush=True)

    # ---- gen16k --------------------------------------------------------------------------------------------------
    for task in P.TASKS:
        src = f"{gcell(task)}/{task}/train.jsonl"
        if task == "contradiction":
            # every --split train row of seed 7777 reused a training gold pair (400/400), so contradiction comes from ctc's
            # EVAL corpus split instead (`--split eval --rungs 16k --eval-seed 77`, 324 documents, 500 rows): 0/100 hits
            src = CONTRA_EVAL
        exs = P.read_jsonl(src)
        # content filter on the JSONL first (cheap), tokenize only candidates. Strict (held-out gold absent from every
        # training document) except qdmatch_nq, whose queries recur as training distractors: there gold-vs-gold.
        drop = {"generated": len(exs), "gold_in_train": 0, "book_in_train": 0, "over_16384": 0,
                "rule": "gold vs training gold" if task == "qdmatch_nq" else "gold vs every training document"}
        cand = []
        for i, ex in enumerate(exs):
            gh = gold_hashes(task, ex)
            if gh & tr_gold[task] or (task != "qdmatch_nq" and gh & tr_docs[task]):
                drop["gold_in_train"] += 1
                continue
            if task == "reorder" and P.book_of(ex) in tr_books:
                drop["book_in_train"] += 1
                continue
            cand.append(i)
        drop["content_eligible"] = len(cand)
        drop["content_eligible_gold_vs_gold_only"] = sum(
            1 for ex in exs if not (gold_hashes(task, ex) & tr_gold[task])
            and not (task == "reorder" and P.book_of(ex) in tr_books))
        print(f"[v3] gen16k/{task} content filter: {drop}", flush=True)
        if not cand:
            print(f"[v3] WARNING gen16k/{task}: no candidate rows", flush=True)
            continue
        cand = cand[:a.gen_rows + 24]
        with tempfile.TemporaryDirectory(dir="/data/prasann/tmp") as tmp:
            with open(f"{tmp}/c.jsonl", "w") as g:
                for i in cand:
                    g.write(json.dumps(exs[i]) + "\n")
            os.makedirs(f"{tmp}/o")
            P.convert(f"{tmp}/c.jsonl", P.SPEC[task], f"{tmp}/o", num_proc=8)
            _, rows = P.load_shard_rows(f"{tmp}/o")
        assert len(rows) == len(cand)
        picked = []
        for i, (r, m) in zip(cand, rows):
            if len(r) > P.MAX_LEN:
                drop["over_16384"] += 1
                continue
            picked.append((i, r, m, exs[i]))
        if len(picked) < a.gen_rows:
            print(f"[v3] WARNING gen16k/{task}: only {len(picked)} eligible rows", flush=True)
        add("gen16k", task, picked[:a.gen_rows], {"source_file": src, "seed": GSEED, "drops": drop})

    # ---- rung16k -------------------------------------------------------------------------------------------------
    with tempfile.TemporaryDirectory(dir="/data/prasann/tmp") as tmp:
        for task in P.TASKS:
            path = P.eval_rung_path(task, "16k")
            exs = P.read_jsonl(path, a.rung_scan)
            head = f"{tmp}/{task}.jsonl"
            with open(head, "w") as g:
                for ex in exs:
                    g.write(json.dumps(ex) + "\n")
            d = f"{tmp}/{task}"
            os.makedirs(d)
            P.convert(head, P.SPEC[task], d, num_proc=8)
            _, rows = P.load_shard_rows(d)
            assert len(rows) == len(exs)
            picked, skipped = [], []
            for i, ((r, m), ex) in enumerate(zip(rows, exs)):
                if len(r) > RUNG_CAP:
                    skipped.append(i)
                    continue
                if len(picked) < a.rung_rows:
                    picked.append((i, r, m, ex))
            add("rung16k", task, picked, {"source_file": path, "skipped_over_40960": len(skipped),
                                          "skipped_src_idx": skipped[:50]})

    # ---- copies from heldout_v2 ----------------------------------------------------------------------------------
    h = np.load(HV2)
    o = h["offsets"]
    for set_name in ("suite16k", "suite8k", "suite4k"):
        for task in P.TASKS:
            picked = []
            for j in np.flatnonzero((h["sets"] == set_name) & (h["tasks"] == task)):
                picked.append((int(h["src_idx"][j]), h["ids"][o[j]:o[j + 1]], h["mask"][o[j]:o[j + 1]], None))
            if picked:
                add(set_name, task, picked, {"source": HV2})

    np.savez(a.out + ".tmp.npz", ids=np.concatenate(ids_all), mask=np.concatenate(mask_all), offsets=np.asarray(offs),
             sets=np.asarray(sets), tasks=np.asarray(tasks), src_idx=np.asarray(src_idx))
    json.dump(stats, open(a.out.replace(".npz", "_stats.json"), "w"), indent=1)
    os.replace(a.out + ".tmp.npz", a.out)
    print(f"[v3] wrote {a.out} ({len(sets)} rows)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["gen", "build"])
    ap.add_argument("--out", default="/data/prasann/early_loss_proxy/heldout_v3.npz")
    ap.add_argument("--gen-rows", type=int, default=48)
    ap.add_argument("--rung-rows", type=int, default=32)
    ap.add_argument("--rung-scan", type=int, default=40)
    a = ap.parse_args()
    {"gen": gen, "build": build}[a.stage](a)


if __name__ == "__main__":
    main()
