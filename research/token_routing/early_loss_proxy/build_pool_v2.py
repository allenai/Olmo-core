"""Build the v2 training pool ``mtb16k`` for the early dense-loss proxy (records/early-dense-loss-proxy.md). Runs on a node.

Tasks (t0..t3): contradiction, outlier, reorder, qdmatch_nq. Three length rungs per task (4k / 8k / 16k), equal row
counts per rung, every row <= 16,384 tokens (the trainer's --seq-len; longer rows are dropped and counted).

Stages (run in order; each is idempotent on its own outputs):

``gen``       task JSONL per (task, rung) with the vendored ``ctc`` generator, ``--split train``, sized exactly like the
              IID setA build (``recipes/data/ctc_sft/build_ctc_sft.py --calibration``): ``ctc_calibrated.py`` with the
              eval ladder's own document count per rung (``recipes/data/ctc_sft/eval_calibration.json``).
``tokenize``  ``convert_unified_to_document_landmark.py --emit dense --marker-set qwen3_5 --query-position after
              --cot-mode none --chunk-by document`` per cell (``--seq-len 65536`` so nothing is dropped there; the
              16,384 cap is applied in ``assemble`` with the source row index kept, see shard-src-positional-zip-bug).
``assemble``  per cell: drop rows > 16,384 tokens and rows whose gold content occurs in the eval rung files (all 500
              rows of each of the 3 rungs), take the first ``--rows-per-rung`` survivors, shuffle each task's rows with a
              fixed seed, write the merged shard (mt4f1952's layout) + ``gold_fingerprints.json`` (the
              ``--st-gold-sidecar`` format: {content fingerprint: sorted 0-based document indices}) +
              ``gold_pairs_fingerprints.json`` (contradiction / qdmatch_nq: 0-based index pairs) + ``row_info.jsonl``
              (per row: task, rung, source file + row, n_docs, gold, document token spans) + ``<out>_stats.json``;
              verifies gold against the source JSONL by decoding document spans.
``order``     the first rows a seed-0 trainer run sees (the real ComposableDataLoader global order), by task and rung.

    PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
    $PY research/token_routing/early_loss_proxy/build_pool_v2.py gen
    $PY research/token_routing/early_loss_proxy/build_pool_v2.py tokenize
    $PY research/token_routing/early_loss_proxy/build_pool_v2.py assemble   # needs build_heldout_v2.py fetch first
    $PY research/token_routing/early_loss_proxy/build_pool_v2.py order
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
WORK = "/data/prasann/early_loss_proxy/v2build"
OUT = "/data/prasann/early_loss_proxy/shards/mtb16k"
TOKENIZER = "/data/prasann/hf_models/Qwen3.5-0.8B-Base"
SEEDS = "/data/prasann/ctc_seed_pools"
EOS, DS, DE = 248044, 248049, 248050
MAX_LEN = 16384
TASKS = ["contradiction", "outlier", "reorder", "qdmatch_nq"]  # t0..t3
RUNGS = ["4k", "8k", "16k"]
RUNG_TOKENS = {"4k": 4096, "8k": 8192, "16k": 16384}
#: converter --task (build_ctc_sft.SET_A spec names)
SPEC = {"contradiction": "contradiction", "outlier": "outlier", "reorder": "reorder", "qdmatch_nq": "qdmatch"}
#: seed pool per task (local file when the node has one, else the published pool from the Hub)
POOL = {"contradiction": "auto", "outlier": f"{SEEDS}/outlier.seed.jsonl.gz", "reorder": "auto",
        "qdmatch_nq": f"{SEEDS}/qdmatch_nq.seed.jsonl.gz"}
#: rows generated per cell (more at the 16k cells of tasks whose eval-sized rows straddle 16,384 tokens)
#: qdmatch_nq: the suite eval reuses the NQ-train query pool (19,177 distinct queries over the 3 rungs), so ~60% of
#: generated rows share a gold query/document with an eval gold pair and are dropped; generate 2,000 per cell
GEN_ROWS = {("qdmatch_nq", "4k"): 2000, ("qdmatch_nq", "8k"): 2000, ("qdmatch_nq", "16k"): 2000}
GEN_DEFAULT = 1000
SEED = 42  # the setA build's pinned seed (ctc keys streams by (seed, split, rung))
#: suite eval rung files (fetched by build_heldout_v2.py fetch from the HF dataset at olmo-eval's pinned revision)
SUITE = f"{WORK}/suite_rows/suite"
EVAL_SUBSET = {"contradiction": "contradiction_iid", "outlier": "outlier", "reorder": "reorder",
               "qdmatch_nq": "qdmatch_nq"}


def env():
    e = dict(os.environ, PYTHONPATH=f"{REPO}/src:{REPO}/ctc/src", TOKENIZERS_PARALLELISM="false",
             HF_HOME=f"{WORK}/hf", HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONWARNINGS="ignore")
    return e


def cell_dir(task, rung):
    return f"{WORK}/gen/{task}_{rung}"


def eval_rung_path(task, rung):
    return f"{SUITE}/{EVAL_SUBSET[task]}/rung_{RUNG_TOKENS[rung]}.jsonl"


# ------------------------------------------------------------------------------------------------ gen
def _gen_cell(job):
    task, rung, n_docs, rows = job
    out = cell_dir(task, rung)
    target = f"{out}/{task}/train.jsonl"
    if os.path.exists(target) and os.path.getsize(target):
        return task, rung, "cached", sum(1 for _ in open(target)), 0.0
    t0 = time.time()
    cmd = [sys.executable, f"{REPO}/recipes/data/ctc_sft/ctc_calibrated.py", "build", "--task", task, "--split",
           "train", "--rungs", rung, "--train", str(rows), "--seed", str(SEED), "--out", out, "--pool", POOL[task]]
    e = env()
    e["CTC_DOCS_OVERRIDE"] = json.dumps({f"{task}:{rung}": n_docs})
    p = subprocess.run(cmd, env=e, capture_output=True, text=True)
    open(f"{out}.log", "w").write(" ".join(cmd) + "\n" + p.stdout + p.stderr)
    if p.returncode != 0 or not os.path.exists(target):
        return task, rung, "FAILED: " + " | ".join((p.stdout + p.stderr).strip().splitlines()[-3:])[:800], 0, 0.0
    return task, rung, "built", sum(1 for _ in open(target)), time.time() - t0


N_DOCS_FILE = f"{WORK}/n_docs_used.json"


def _cell_mean_len(task, rung):
    m = json.load(open(f"{WORK}/tok/{task}_{rung}/metadata.json"))
    return m["num_tokens"] / m["num_instances"]


def refit_n_docs(task, cal, target_mean):
    """Documents per row for ``task``'s 16k cell so the mean tokenized row is ``target_mean``: a line through the
    measured mean row lengths of the tokenized 4k and 8k cells (tokens per document = slope, fixed overhead =
    intercept)."""
    n4, n8 = int(cal[task]["4k"]["n_docs"]), int(cal[task]["8k"]["n_docs"])
    m4, m8 = _cell_mean_len(task, "4k"), _cell_mean_len(task, "8k")
    slope = (m8 - m4) / (n8 - n4)
    icpt = m4 - slope * n4
    return int((target_mean - icpt) // slope), {"tok_per_doc": slope, "overhead": icpt, "mean_4k": m4, "mean_8k": m8}


def gen(a):
    cal = json.load(open(f"{REPO}/recipes/data/ctc_sft/eval_calibration.json"))["tasks"]
    used = json.load(open(N_DOCS_FILE)) if os.path.exists(N_DOCS_FILE) else {}
    for t in (a.refit_16k.split(",") if a.refit_16k else []):
        n, fit = refit_n_docs(t, cal, a.target_mean)
        used[f"{t}:16k"] = {"n_docs": n, "eval_n_docs": int(cal[t]["16k"]["n_docs"]), "why": "refit", "target_mean": a.target_mean, **fit}
        print(f"[gen] refit {t}@16k: {n} documents (eval has {cal[t]['16k']['n_docs']}); {fit}", flush=True)
    jobs = []
    for t in TASKS:
        for r in RUNGS:
            k = f"{t}:{r}"
            if k not in used:
                used[k] = {"n_docs": int(cal[t][r]["n_docs"]), "eval_n_docs": int(cal[t][r]["n_docs"]), "why": "eval calibration"}
            jobs.append((t, r, used[k]["n_docs"], GEN_ROWS.get((t, r), GEN_DEFAULT)))
    os.makedirs(WORK, exist_ok=True)
    json.dump(used, open(N_DOCS_FILE, "w"), indent=1)
    for j in jobs:
        print(f"[gen] {j[0]}@{j[1]}: n_docs {j[2]} ({used[j[0] + ':' + j[1]]['why']}), rows {j[3]}, pool {POOL[j[0]]}", flush=True)
    # one serial cell per task first: warms the Hub seed-pool cache before the parallel pass
    first = [j for j in jobs if j[1] == "4k"]
    rest = [j for j in jobs if j[1] != "4k"]
    with cf.ProcessPoolExecutor(max_workers=len(first)) as ex:
        for r in ex.map(_gen_cell, first):
            print(f"[gen] {r[0]}@{r[1]} {r[2]} rows={r[3]} ({r[4]:.0f}s)", flush=True)
    with cf.ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(_gen_cell, rest):
            print(f"[gen] {r[0]}@{r[1]} {r[2]} rows={r[3]} ({r[4]:.0f}s)", flush=True)


# ------------------------------------------------------------------------------------------------ tokenize
def convert(jsonl, task_spec, out_dir, num_proc=4):
    cmd = [sys.executable, f"{REPO}/recipes/data/convert_unified_to_document_landmark.py", "--input-jsonl", jsonl,
           "--task", task_spec, "--out-dir", out_dir, "--emit", "dense", "--marker-set", "qwen3_5",
           "--tokenizer", TOKENIZER, "--seq-len", "65536", "--query-position", "after", "--cot-mode", "none",
           "--chunk-by", "document", "--num-proc", str(num_proc)]
    p = subprocess.run(cmd, cwd=REPO, env=env(), capture_output=True, text=True)
    open(f"{out_dir}.log", "w").write(" ".join(cmd) + "\n" + p.stdout[-20000:] + p.stderr[-20000:])
    if p.returncode != 0:
        raise RuntimeError(f"convert {jsonl} failed: {(p.stdout + p.stderr)[-1500:]}")


def _tok_cell(job):
    task, rung = job
    src = f"{cell_dir(task, rung)}/{task}/train.jsonl"
    out = f"{WORK}/tok/{task}_{rung}"
    if os.path.exists(f"{out}/metadata.json"):
        return task, rung, "cached"
    os.makedirs(out, exist_ok=True)
    convert(src, SPEC[task], out)
    return task, rung, "built"


def tokenize(a):
    jobs = [(t, r) for t in TASKS for r in RUNGS]
    with cf.ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(_tok_cell, jobs):
            print(f"[tok] {r[0]}@{r[1]} {r[2]}", flush=True)


# ------------------------------------------------------------------------------------------------ shared helpers
def load_shard_rows(d):
    """All rows of a converter shard dir, in order (rows end in EOS). Checks part counts and byte sizes
    (qdmatch-nq-shard-truncated) and that the shard has exactly num_instances rows."""
    meta = json.load(open(f"{d}/metadata.json"))
    tf = sorted(glob.glob(f"{d}/token_ids_part_*.npy"))
    mf = sorted(glob.glob(f"{d}/labels_mask_*.npy"))
    assert len(tf) == len(mf) and tf, (d, len(tf), len(mf))
    ids = np.concatenate([np.fromfile(f, dtype=meta["dtype"]) for f in tf])
    msk = np.concatenate([np.fromfile(f, dtype=meta["mask_dtype"]) for f in mf])
    assert ids.size == msk.size == meta["num_tokens"], (d, ids.size, msk.size, meta["num_tokens"])
    ends = np.flatnonzero(ids == EOS)
    assert len(ends) == meta["num_instances"] and ends[-1] == ids.size - 1, (d, len(ends), meta["num_instances"])
    starts = np.concatenate([[0], ends[:-1] + 1])
    return meta, [(ids[s:e + 1], msk[s:e + 1]) for s, e in zip(starts, ends)]


def doc_spans(row):
    """[(start, end)) token spans of each wrapped document, marker to marker inclusive."""
    st = np.flatnonzero(row == DS)
    en = np.flatnonzero(row == DE)
    assert len(st) == len(en) and np.all(st < en) and np.all(st[1:] > en[:-1]), "unbalanced doc markers"
    return [(int(s), int(e) + 1) for s, e in zip(st, en)]


def norm(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def thash(text):
    return hashlib.sha1(norm(text).encode()).hexdigest()


def gold_info(task, ex):
    """(sorted 0-based gold document indices, 0-based pairs or None) from the SOURCE row, with the per-task base
    (gold-doc-indices-per-task-base): contradiction gold_doc_indices are 1-based Claim ids (pairs), outlier
    gold_doc_indices are 0-based, qdmatch gold_pairs are 1-based [query, document] positions in ``documents``,
    reorder: every document is gold."""
    n = len(ex["documents"])
    if task == "reorder":
        return list(range(n)), None
    if task == "outlier":
        g = sorted(int(x) for x in ex["gold_doc_indices"])
        return g, None
    if task == "contradiction":
        pairs = [[int(a) - 1, int(b) - 1] for a, b in ex["gold_doc_indices"]]
    elif task == "qdmatch_nq":
        pairs = [[int(a) - 1, int(b) - 1] for a, b in ex["gold_pairs"]]
    else:
        raise ValueError(task)
    flat = sorted({i for p in pairs for i in p})
    assert all(0 <= i < n for i in flat), (task, pairs, n)
    return flat, pairs


def eval_content_keys(task, rows):
    """Content hashes the training rows must not share with the eval: gold documents (contradiction: the gold
    claims; outlier: the outlier passages; qdmatch: the query and document of every gold pair -- distractor queries
    are NOT filtered, the suite eval reuses the whole NQ-train query pool), and for reorder every passage (all gold)
    plus the source book id; plus all document hashes (reported, not filtered)."""
    gold, alld, books = set(), set(), set()
    for ex in rows:
        g, _ = gold_info(task, ex)
        docs = ex["documents"]
        for i, d in enumerate(docs):
            h = thash(d.get("text"))
            alld.add(h)
            if i in set(g):
                gold.add(h)
        if task == "reorder" and book_of(ex):
            books.add(book_of(ex))
    return gold, alld, books


def book_of(ex):
    """reorder source book id: eval rows carry ``source_id``, ctc train rows ``_meta.book``."""
    meta = ex.get("_meta") or {}
    return ex.get("source_id") or (meta.get("book") if isinstance(meta, dict) else None)


def read_jsonl(p, n=None):
    out = []
    with open(p) as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            out.append(json.loads(line))
    return out


def printed_id(text):
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


# ------------------------------------------------------------------------------------------------ assemble
def assemble(a):
    from tokenizers import Tokenizer

    sys.path.insert(0, f"{REPO}/src")
    from olmo_core.nn.attention.gold_grad_mask import content_fingerprint

    tok = Tokenizer.from_file(f"{TOKENIZER}/tokenizer.json")
    dec = lambda x: tok.decode([int(i) for i in x], skip_special_tokens=False)  # noqa: E731
    if os.path.exists(f"{a.out}/metadata.json"):
        raise SystemExit(f"{a.out} already has metadata.json; refusing to overwrite")
    os.makedirs(a.out, exist_ok=True)
    n_docs_used = json.load(open(N_DOCS_FILE))
    per_rung = [a.rows_per_task // 3 + (1 if i < a.rows_per_task % 3 else 0) for i in range(3)]
    stats = {"out": a.out, "max_len": MAX_LEN, "rows_per_rung_target": dict(zip(RUNGS, per_rung)),
             "shuffle_seed": a.shuffle_seed, "n_docs_used": n_docs_used, "tasks": {}}
    gold_side, pair_side, merged_from = {}, {}, []
    info_f = open(f"{a.out}/row_info.jsonl", "w")
    tot_tok = tot_loss = 0
    lens_all = []
    verify = {}
    for t, task in enumerate(TASKS):
        rows_t = []  # (ids, mask, info)
        stats["tasks"][task] = {}
        for ri, rung in enumerate(RUNGS):
            src = f"{cell_dir(task, rung)}/{task}/train.jsonl"
            meta, rows = load_shard_rows(f"{WORK}/tok/{task}_{rung}")
            exs = read_jsonl(src)
            assert len(rows) == len(exs), f"{task}@{rung}: {len(rows)} shard rows vs {len(exs)} source rows"
            assert meta["num_dropped"] == 0
            ev = read_jsonl(eval_rung_path(task, rung))
            eg, ea, eb = set(), set(), set()
            for r2 in RUNGS:  # a training row must be clean against every eval rung, not only its own
                g2, a2, b2 = eval_content_keys(task, read_jsonl(eval_rung_path(task, r2)))
                eg |= g2
                ea |= a2
                eb |= b2
            n_long = n_gold_hit = n_book_hit = 0
            doc_hit_frac = []
            kept = []
            for i, ((ids, m), ex) in enumerate(zip(rows, exs)):
                g, pairs = gold_info(task, ex)
                docs = ex["documents"]
                hs = [thash(d.get("text")) for d in docs]
                doc_hit_frac.append(float(np.mean([h in ea for h in hs])))
                if len(ids) > MAX_LEN:
                    n_long += 1
                    continue
                gset = set(g)
                gold_hit = any(hs[j] in eg for j in gset)
                if gold_hit:
                    n_gold_hit += 1
                    continue
                if task == "reorder" and book_of(ex) in eb:
                    n_book_hit += 1
                    continue
                kept.append((i, ids, m, ex, g, pairs))
            if len(kept) < per_rung[ri]:
                raise SystemExit(f"[assemble] {task}@{rung}: only {len(kept)} eligible rows < {per_rung[ri]}")
                print(f"[assemble] WARNING {task}@{rung}: only {len(kept)} eligible rows < {per_rung[ri]}", flush=True)
            sel = kept[:per_rung[ri]]
            L = np.array([len(x[1]) for x in sel])
            A = np.array([int(x[2].sum()) for x in sel])
            stats["tasks"][task][rung] = {
                "rows": len(sel), "source_file": src, "source_rows": len(exs), "tokenized_dir": f"{WORK}/tok/{task}_{rung}",
                "eval_file": eval_rung_path(task, rung), "n_docs": sorted({len(x[3]["documents"]) for x in sel}),
                "n_docs_used": n_docs_used.get(f"{task}:{rung}"),
                "dropped_over_16384": n_long, "dropped_gold_in_eval": n_gold_hit, "dropped_book_in_eval": n_book_hit,
                "eligible_rows": len(kept),
                "len_mean": float(L.mean()), "len_min": int(L.min()), "len_max": int(L.max()),
                "len_all_generated_max": int(max(len(r[0]) for r in rows)),
                "answer_tokens_mean": float(A.mean()), "answer_tokens_min": int(A.min()), "answer_tokens_max": int(A.max()),
                "doc_share_in_eval_mean_all_generated": float(np.mean(doc_hit_frac)),
            }
            print(f"[assemble] {task}@{rung}: {len(exs)} generated, {n_long} > {MAX_LEN} tok, {n_gold_hit} gold-in-eval, "
                  f"{n_book_hit} book-in-eval -> {len(sel)} kept; len {L.mean():.0f} [{L.min()}, {L.max()}], "
                  f"answer {A.mean():.1f} tok; eval-doc share {np.mean(doc_hit_frac):.4f}", flush=True)
            for (i, ids, m, ex, g, pairs) in sel:
                rows_t.append((ids, m, {"rung": rung, "src": src, "src_idx": i, "gold": g, "pairs": pairs, "ex": ex}))
        # ---- shuffle within task (fixed seed), verify gold, write --------------------------------------------
        perm = np.random.default_rng(a.shuffle_seed + t).permutation(len(rows_t))
        rows_t = [rows_t[j] for j in perm]
        vstat = {"rows_checked": 0, "span_text_match": 0, "printed_id_is_index_plus_1": 0, "answer_ids_match_gold": 0,
                 "examples": []}
        ids_out, msk_out = [], []
        for r, (ids, m, inf) in enumerate(rows_t):
            ex = inf.pop("ex")
            spans = doc_spans(ids)
            n = len(ex["documents"])
            assert len(spans) == n, f"{task} row {r}: {len(spans)} doc spans vs {n} source documents"
            a0 = int(np.flatnonzero(m)[0])
            assert a0 > spans[-1][1] - 1, f"{task} row {r}: loss tokens before the last document"
            fp = content_fingerprint(ids.tolist())
            assert fp not in gold_side, f"duplicate fingerprint {task} row {r}"
            gold_side[fp] = inf["gold"]
            if inf["pairs"] is not None:
                pair_side[fp] = inf["pairs"]
            if r < a.verify_rows:  # decode every gold document span and the answer
                vstat["rows_checked"] += 1
                ok_text = ok_id = True
                pid_all = []
                for j, (s, e) in enumerate(spans):
                    body = dec(ids[s + 1:e - 1])
                    pid = printed_id(body[:40])
                    pid_all.append(pid)
                    if j in set(inf["gold"]) or j == 0:
                        src_text = norm(ex["documents"][j].get("text"))
                        if src_text[:200] not in norm(body):
                            ok_text = False
                if pid_all != list(range(1, n + 1)):
                    ok_id = False
                ans = dec(ids[m])
                nums = [int(x) for x in re.findall(r"\d+", ans)]
                if task == "reorder":
                    ok_ans = sorted(nums) == list(range(1, n + 1)) and nums == [int(x) for x in ex["gold_order"]]
                elif inf["pairs"] is not None:
                    want = [x + 1 for p in inf["pairs"] for x in p]
                    ok_ans = sorted(nums) == sorted(want)
                else:
                    ok_ans = sorted(nums) == [x + 1 for x in inf["gold"]]
                vstat["span_text_match"] += ok_text
                vstat["printed_id_is_index_plus_1"] += ok_id
                vstat["answer_ids_match_gold"] += ok_ans
                if r < 2 or not (ok_text and ok_id and ok_ans):
                    gj = inf["gold"][0]
                    s, e = spans[gj]
                    vstat["examples"].append({"row": r, "rung": inf["rung"], "gold0": gj, "answer": ans[:300],
                                              "gold_doc_decoded_head": dec(ids[s:e])[:200],
                                              "src_doc_head": norm(ex["documents"][gj].get("text"))[:160],
                                              "ok": [ok_text, ok_id, ok_ans]})
            info = {"part": t, "row": r, "task": task, "rung": inf["rung"], "src": inf["src"], "src_idx": inf["src_idx"],
                    "n_docs": n, "len": int(len(ids)), "answer_tokens": int(m.sum()), "fingerprint": fp,
                    "gold": inf["gold"], "gold_pairs": inf["pairs"], "doc_spans": spans}
            info_f.write(json.dumps(info) + "\n")
            ids_out.append(ids)
            msk_out.append(m)
            lens_all.append(len(ids))
        verify[task] = vstat
        bt = np.concatenate(ids_out).astype("uint32")
        bm = np.concatenate(msk_out).astype("bool")
        bt.tofile(f"{a.out}/token_ids_part_t{t}_000.npy")
        bm.tofile(f"{a.out}/labels_mask_t{t}_000.npy")
        tot_tok += bt.size
        tot_loss += int(bm.sum())
        merged_from.append({"src": [stats["tasks"][task][r]["tokenized_dir"] for r in RUNGS], "task": task,
                            "converter_task": SPEC[task], "num_instances": len(rows_t),
                            "rung_counts": {r: stats["tasks"][task][r]["rows"] for r in RUNGS}})
        print(f"[assemble] t{t} {task}: {len(rows_t)} rows written; verify {json.dumps({k: v for k, v in vstat.items() if k != 'examples'})}",
              flush=True)
    info_f.close()
    meta0 = json.load(open(f"{WORK}/tok/{TASKS[0]}_{RUNGS[0]}/metadata.json"))
    meta = dict(meta0)
    meta.update({"task": "mtb16k", "tokenizer": TOKENIZER, "num_instances": sum(m["num_instances"] for m in merged_from),
                 "num_dropped": 0, "num_tokens": int(tot_tok), "num_loss_tokens": int(tot_loss),
                 "max_example_len": int(max(lens_all)), "min_example_len": int(min(lens_all)),
                 "subset_eos_id": EOS, "rows_over_max_len_dropped": sum(stats["tasks"][t][r]["dropped_over_16384"]
                                                                       for t in TASKS for r in RUNGS),
                 "subset_max_len": MAX_LEN, "merged_from": merged_from, "shuffle_seed": a.shuffle_seed,
                 "rows_per_rung": dict(zip(RUNGS, per_rung)),
                 "generator": "ctc (vendored) build --split train via recipes/data/ctc_sft/ctc_calibrated.py, "
                              "n_docs per rung = recipes/data/ctc_sft/eval_calibration.json, seed 42",
                 "gold_sidecar": "gold_fingerprints.json (0-based document indices); gold_pairs_fingerprints.json; "
                                 "row_info.jsonl (rung, source, doc token spans)"})
    for k in ("input_jsonl", "num_proc"):
        meta.pop(k, None)
    json.dump(meta, open(f"{a.out}/metadata.json", "w"), indent=1)
    json.dump(gold_side, open(f"{a.out}/gold_fingerprints.json", "w"))
    json.dump(pair_side, open(f"{a.out}/gold_pairs_fingerprints.json", "w"))
    stats["gold_verification"] = verify
    stats["num_instances"] = meta["num_instances"]
    stats["max_example_len"] = meta["max_example_len"]
    json.dump(stats, open(f"{a.out}_stats.json", "w"), indent=1)
    print(f"[assemble] wrote {a.out}: {meta['num_instances']} rows, max len {meta['max_example_len']}", flush=True)


# ------------------------------------------------------------------------------------------------ order
def order(a):
    """Global instance order of a seed-``a.seed`` run, built with the trainer's own source + loader classes."""
    sys.path.insert(0, f"{REPO}/src")
    from olmo_core.data import TokenizerConfig
    from olmo_core.data.composable import ComposableDataLoaderConfig, PadToLengthInstanceSourceConfig

    tc = TokenizerConfig(vocab_size=248320, eos_token_id=EOS, pad_token_id=EOS, bos_token_id=None)
    src_cfg = PadToLengthInstanceSourceConfig.from_npy(f"{a.out}/token_ids_part_*.npy", tokenizer=tc,
                                                       sequence_length=MAX_LEN,
                                                       label_mask_paths=[f"{a.out}/labels_mask_*.npy"], expand_glob=True)
    wd = f"{WORK}/loader_work"
    source = src_cfg.build(wd)
    dl = ComposableDataLoaderConfig(tokenizer=tc, work_dir=wd, global_batch_size=16 * MAX_LEN,
                                    seed=34521 + a.seed, num_workers=0).build(source)
    dl.reshuffle(epoch=1)
    gi = np.asarray(dl.get_global_indices())
    info = [json.loads(x) for x in open(f"{a.out}/row_info.jsonl")]
    # instance i of the source = row i of the concatenated parts in glob (t0..t3) order; check it by content
    lens = np.array([x["len"] for x in info])
    for k in (0, 1, len(info) // 2, len(info) - 1):
        inst = source[int(k)]
        ids = np.asarray(inst["input_ids"])
        n_real = int(np.flatnonzero(ids == EOS)[0]) + 1
        assert n_real == lens[k], (k, n_real, lens[k])
    res = {"seed": a.seed, "loader_seed": 34521 + a.seed, "total_instances": int(len(gi)), "first": {}}
    for n in (160, 400, 800):
        c = {}
        for i in gi[:n]:
            x = info[int(i)]
            c[f"{x['task']}/{x['rung']}"] = c.get(f"{x['task']}/{x['rung']}", 0) + 1
        res["first"][str(n)] = dict(sorted(c.items()))
        print(f"[order] first {n} rows: {dict(sorted(c.items()))}", flush=True)
    json.dump(res, open(f"{a.out}_order_seed{a.seed}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["gen", "tokenize", "assemble", "order"])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--rows-per-task", type=int, default=1952)
    ap.add_argument("--shuffle-seed", type=int, default=20261116)
    ap.add_argument("--verify-rows", type=int, default=40, help="rows per task whose gold spans are decoded")
    ap.add_argument("--seed", type=int, default=0, help="order: the trainer --seed")
    ap.add_argument("--refit-16k", default="", help="gen: tasks whose 16k cell is re-sized from the 4k/8k token fit")
    ap.add_argument("--target-mean", type=float, default=14500.0, help="gen --refit-16k: target mean row tokens")
    a = ap.parse_args()
    {"gen": gen, "tokenize": tokenize, "assemble": assemble, "order": order}[a.stage](a)


if __name__ == "__main__":
    main()
