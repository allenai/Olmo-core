"""Long-row pool ``mtb64k`` + held-out ``heldout_v4.npz`` (POOL_V2.md, "Long-row pool"). Runs on a node (mooney).

Tasks contradiction, outlier, qdmatch_nq (reorder has no rows past 16k in the suite and is left out); rungs ``32k``
(mean ~29k tokens) and ``64k`` (~58k), documents per row from the tokens-per-document line through the tokenized 4k / 8k
cells (n_docs_used.json); 320 rows per task per rung; every row <= 65,536 tokens. Generator / split / converter flags /
layout / sidecars as mtb16k (ctc ``--split train`` seed 42 via ctc_calibrated.py; the converter runs with --seq-len
131072 so nothing is dropped inside it). Pool parts are t0..t2 = contradiction, outlier, qdmatch_nq.

Exclusions for training rows (a row is dropped if any gold item occurs in): every suite eval rung r2k-r64k of its task
(HF ctc-suite-eval at olmo-eval's revision; r128k+ not fetched), heldout_v2/v3 rows (suite rows + gen16k) and every
heldout_v4 candidate row. Held-out ``gen32k`` / ``gen64k``: 24 rows per task from ctc's EVAL split (eval seed 99, same
document counts), <= 65,536 tokens, gold absent from every training pool's gold (outlier: from every training
document). ``gen16k``: copied from heldout_v3.

    $PY build_long_pools.py fetch ; ... gen ; ... tokenize ; ... build
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_pool_v2 as P  # noqa: E402
import build_short_pools as SP  # noqa: E402

TASKS = ["contradiction", "outlier", "qdmatch_nq"]
RUNGS = ["32k", "64k"]
TARGET = {"32k": 29000.0, "64k": 58000.0}
CAP = 65536
ROWS = 320
GEN_ROWS = {"contradiction": 450, "outlier": 450, "qdmatch_nq": 2400}  # qdmatch_nq: ~17% eligible
HO_ROWS, HO_GEN = 24, 120
EVAL_SEED = 99
SH = "/data/prasann/early_loss_proxy/shards"
LW = f"{P.WORK}/long"
SUB = {"contradiction": "contradiction_iid", "outlier": "outlier", "qdmatch_nq": "qdmatch_nq"}
HF_RUNGS = ["r2k", "r4k", "r8k", "r16k", "r32k", "r64k"]
REV = "04ab86006124b3f34c566f44bbc82d4fec98fcda"
ALL_POOLS = ["mtb16k", "mtbshort", "mtbmix80", "mtbmix50", "mtbmix90", "mtbmix95", "mtbmix80L", "mtbmix80h", "mtbmix80x3"]


def n_docs_for(task, rung):
    cal = json.load(open(f"{P.REPO}/recipes/data/ctc_sft/eval_calibration.json"))["tasks"]
    _, fit = P.refit_n_docs(task, cal, TARGET[rung])
    n = int(round((TARGET[rung] - fit["overhead"]) / fit["tok_per_doc"]))
    if task == "qdmatch_nq" and n % 2:
        n += 1
    return n, fit


def fetch(a):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    os.makedirs(f"{LW}/suite", exist_ok=True)
    got = {}
    for t in TASKS:
        for r in HF_RUNGS:
            out = f"{LW}/suite/{SUB[t]}_{r}.jsonl"
            if not os.path.exists(out):
                pqp = hf_hub_download("PrasannSinghal/ctc-suite-eval", f"data/{SUB[t]}/{r}.parquet", repo_type="dataset",
                                      revision=REV, local_dir=f"{LW}/hf_parquet")
                with open(out, "w") as f:
                    for b in pq.ParquetFile(pqp).iter_batches(batch_size=64):
                        for row in b.to_pylist():
                            f.write(json.dumps(SP_norm(row)) + "\n")
            got[f"{t}/{r}"] = sum(1 for _ in open(out))
    print(f"[fetch] {got}", flush=True)


def SP_norm(ex):
    for k in ("meta", "_meta"):
        if isinstance(ex.get(k), str):
            try:
                ex[k] = json.loads(ex[k])
            except ValueError:
                pass
    return ex


def _gen_eval(job):
    task, rung, n = job
    out = f"{LW}/heldout_gen/{task}_{rung}"
    target = f"{out}/{task}/eval_{rung}.jsonl"
    if os.path.exists(target):
        return task, rung, "cached"
    cmd = [sys.executable, f"{P.REPO}/recipes/data/ctc_sft/ctc_calibrated.py", "build", "--task", task, "--split", "eval",
           "--rungs", rung, "--eval-size", str(HO_GEN), "--allow-small-eval", "--eval-seed", str(EVAL_SEED), "--out", out,
           "--pool", P.POOL[task]]
    e = P.env()
    e["CTC_DOCS_OVERRIDE"] = json.dumps({f"{task}:{rung}": n})
    p = subprocess.run(cmd, env=e, capture_output=True, text=True)
    open(f"{out}.log", "w").write(" ".join(cmd) + "\n" + p.stdout + p.stderr)
    return task, rung, "built" if os.path.exists(target) else "FAILED " + (p.stdout + p.stderr)[-400:]


def gen(a):
    used = {}
    jobs, ejobs = [], []
    for t in TASKS:
        for r in RUNGS:
            n, fit = n_docs_for(t, r)
            used[f"{t}:{r}"] = {"n_docs": n, "why": "tokens-per-document fit", "target_mean": TARGET[r], **fit}
            jobs.append((t, r, n, GEN_ROWS[t]))
            ejobs.append((t, r, n))
            print(f"[long gen] {t}@{r}: {n} documents", flush=True)
    os.makedirs(LW, exist_ok=True)
    json.dump(used, open(f"{LW}/n_docs_long.json", "w"), indent=1)
    with cf.ProcessPoolExecutor(max_workers=6) as ex:
        for r in ex.map(P._gen_cell, jobs):
            print(f"[long gen] train {r[0]}@{r[1]} {r[2]} rows={r[3]} ({r[4]:.0f}s)", flush=True)
        for r in ex.map(_gen_eval, ejobs):
            print(f"[long gen] heldout {r}", flush=True)


def convert(jsonl, task, out_dir):
    cmd = [sys.executable, f"{P.REPO}/recipes/data/convert_unified_to_document_landmark.py", "--input-jsonl", jsonl,
           "--task", P.SPEC[task], "--out-dir", out_dir, "--emit", "dense", "--marker-set", "qwen3_5",
           "--tokenizer", P.TOKENIZER, "--seq-len", "131072", "--query-position", "after", "--cot-mode", "none",
           "--chunk-by", "document", "--num-proc", "4"]
    p = subprocess.run(cmd, cwd=P.REPO, env=P.env(), capture_output=True, text=True)
    open(f"{out_dir}.log", "w").write(" ".join(cmd) + "\n" + p.stdout[-5000:] + p.stderr[-5000:])
    if p.returncode != 0:
        raise RuntimeError(f"convert {jsonl}: {(p.stdout + p.stderr)[-800:]}")


def _tok(job):
    kind, t, r = job
    if kind == "train":
        src, out = f"{P.cell_dir(t, r)}/{t}/train.jsonl", f"{P.WORK}/tok/{t}_{r}"
    else:
        src, out = f"{LW}/heldout_gen/{t}_{r}/{t}/eval_{r}.jsonl", f"{LW}/heldout_tok/{t}_{r}"
    if not os.path.exists(f"{out}/metadata.json"):
        os.makedirs(out, exist_ok=True)
        convert(src, t, out)
    return job


def tokenize(a):
    jobs = [(k, t, r) for k in ("train", "heldout") for t in TASKS for r in RUNGS]
    with cf.ProcessPoolExecutor(max_workers=6) as ex:
        for j in ex.map(_tok, jobs):
            print(f"[long tok] {j}", flush=True)


def gold_h(task, ex, g=None):
    g = P.gold_info(task, ex)[0] if g is None else g
    return {P.thash(ex["documents"][j].get("text")) for j in g}


def build(a):
    from tokenizers import Tokenizer

    sys.path.insert(0, f"{P.REPO}/src")
    from olmo_core.nn.attention.gold_grad_mask import content_fingerprint
    import build_heldout_v3 as H

    tok = Tokenizer.from_file(f"{P.TOKENIZER}/tokenizer.json")
    dec = lambda x: tok.decode([int(i) for i in x], skip_special_tokens=False)  # noqa: E731
    n_docs = json.load(open(f"{LW}/n_docs_long.json"))
    stats = {"tasks_left_out": {"reorder": "no suite rows past 16k"}, "cap": CAP, "n_docs_used": n_docs, "cells": {},
             "heldout": {}}
    # ---- held-out candidates (eval split) ------------------------------------------------------------------------
    ho = {}
    for t in TASKS:
        for r in RUNGS:
            src = f"{LW}/heldout_gen/{t}_{r}/{t}/eval_{r}.jsonl"
            exs = P.read_jsonl(src)
            _, rows = P.load_shard_rows(f"{LW}/heldout_tok/{t}_{r}")
            assert len(rows) == len(exs)
            ho[(t, r)] = (src, exs, rows)
    # ---- exclusion keys for training rows ------------------------------------------------------------------------
    excl = {}
    for t in TASKS:
        eg = set()
        for f in [f"{LW}/suite/{SUB[t]}_{r}.jsonl" for r in HF_RUNGS] + SP.eval_files(t):
            g2, _, _ = P.eval_content_keys(t, P.read_jsonl(f))
            eg |= g2
        eg |= SP.gen16k_keys(t)[0]
        for r in RUNGS:
            for ex in ho[(t, r)][1]:
                eg |= gold_h(t, ex)
        excl[t] = eg
    # ---- training cells ------------------------------------------------------------------------------------------
    os.system(f"rm -rf {SH}/mtb64k_tmp")
    tmp, final = f"{SH}/mtb64k_tmp", f"{SH}/mtb64k"
    if os.path.exists(f"{final}/metadata.json"):
        raise SystemExit("mtb64k exists")
    os.makedirs(tmp)
    gold_side, pair_side, merged_from, lens_all = {}, {}, [], []
    tot_tok = tot_loss = 0
    info_f = open(f"{tmp}/row_info.jsonl", "w")
    tr_gold = {t: set() for t in TASKS}
    tr_docs = {t: set() for t in TASKS}
    for ti, t in enumerate(TASKS):
        rows_t = []
        for r in RUNGS:
            src = f"{P.cell_dir(t, r)}/{t}/train.jsonl"
            meta, rows = P.load_shard_rows(f"{P.WORK}/tok/{t}_{r}")
            exs = P.read_jsonl(src)
            assert len(rows) == len(exs) and meta["num_dropped"] == 0
            d = {"generated": len(exs), "over_cap": 0, "gold_in_eval_or_heldout": 0}
            kept = []
            for i, ((ids, m), ex) in enumerate(zip(rows, exs)):
                if len(ids) > CAP:
                    d["over_cap"] += 1
                    continue
                if gold_h(t, ex) & excl[t]:
                    d["gold_in_eval_or_heldout"] += 1
                    continue
                kept.append((ids, m, ex, i, src))
            d["eligible"] = len(kept)
            if len(kept) < ROWS:
                raise SystemExit(f"{t}@{r}: only {len(kept)} eligible rows < {ROWS}: {d}")
            sel = kept[:ROWS]
            ver = {"rows_checked": 0, "span_text_match": 0, "printed_id_is_index_plus_1": 0, "answer_ids_match_gold": 0}
            for ids, m, ex, i, _ in sel[:a.verify_rows]:
                g, pairs = P.gold_info(t, ex)
                spans = P.doc_spans(ids)
                n = len(ex["documents"])
                ver["rows_checked"] += 1
                ver["span_text_match"] += all(
                    P.norm(ex["documents"][j].get("text"))[:200] in P.norm(dec(ids[spans[j][0] + 1:spans[j][1] - 1]))
                    for j in set(g) | {0})
                ver["printed_id_is_index_plus_1"] += [P.printed_id(dec(ids[s + 1:e - 1])[:40]) for s, e in spans] == list(range(1, n + 1))
                nums = [int(x) for x in re.findall(r"\d+", dec(ids[m]))]
                ver["answer_ids_match_gold"] += (sorted(nums) == sorted(x + 1 for p in pairs for x in p)) if pairs is not None \
                    else (sorted(nums) == [x + 1 for x in g])
            L = np.array([len(x[0]) for x in sel])
            A = np.array([int(x[1].sum()) for x in sel])
            stats["cells"][f"{t}/{r}"] = {"rows": ROWS, "n_docs": n_docs[f"{t}:{r}"]["n_docs"], "source_file": src,
                                          "len_mean": float(L.mean()), "len_min": int(L.min()), "len_max": int(L.max()),
                                          "tokens": int(L.sum()), "answer_tokens_mean": float(A.mean()), "drops": d,
                                          "gold_verification": ver}
            print(f"[long] {t}@{r}: {json.dumps(stats['cells'][f'{t}/{r}'])}", flush=True)
            for ids, m, ex, i, s in sel:
                g, pairs = P.gold_info(t, ex)
                tr_gold[t] |= gold_h(t, ex, g)
                tr_docs[t].update(P.thash(dd.get("text")) for dd in ex["documents"])
                rows_t.append((ids, m, {"rung": r, "src": s, "src_idx": i, "gold": g, "pairs": pairs}))
        perm = np.random.default_rng(a.shuffle_seed + ti).permutation(len(rows_t))
        rows_t = [rows_t[j] for j in perm]
        for k, (ids, m, inf) in enumerate(rows_t):
            spans = P.doc_spans(ids)
            fp = content_fingerprint(ids.tolist())
            assert fp not in gold_side
            gold_side[fp] = inf["gold"]
            if inf["pairs"] is not None:
                pair_side[fp] = inf["pairs"]
            info_f.write(json.dumps({"part": ti, "row": k, "task": t, "rung": inf["rung"], "src": inf["src"],
                                     "src_idx": inf["src_idx"], "n_docs": len(spans), "len": int(len(ids)),
                                     "answer_tokens": int(m.sum()), "fingerprint": fp, "gold": inf["gold"],
                                     "gold_pairs": inf["pairs"], "doc_spans": spans}) + "\n")
            lens_all.append(len(ids))
        bt = np.concatenate([x[0] for x in rows_t]).astype("uint32")
        bm = np.concatenate([x[1] for x in rows_t]).astype("bool")
        bt.tofile(f"{tmp}/token_ids_part_t{ti}_000.npy")
        bm.tofile(f"{tmp}/labels_mask_t{ti}_000.npy")
        tot_tok += bt.size
        tot_loss += int(bm.sum())
        merged_from.append({"src": [f"{P.WORK}/tok/{t}_{r}" for r in RUNGS], "task": t, "converter_task": P.SPEC[t],
                            "num_instances": len(rows_t), "rung_counts": {r: ROWS for r in RUNGS}})
    info_f.close()
    meta = json.load(open(f"{SH}/mtb16k/metadata.json"))
    meta.update({"task": "mtb64k", "num_instances": len(lens_all), "num_tokens": int(tot_tok), "num_loss_tokens": int(tot_loss),
                 "max_example_len": int(max(lens_all)), "min_example_len": int(min(lens_all)), "merged_from": merged_from,
                 "shuffle_seed": a.shuffle_seed, "rows_per_rung": {r: ROWS for r in RUNGS}, "subset_max_len": CAP,
                 "rows_over_max_len_dropped": sum(c["drops"]["over_cap"] for c in stats["cells"].values()),
                 "tasks_order": TASKS})
    json.dump(gold_side, open(f"{tmp}/gold_fingerprints.json", "w"))
    json.dump(pair_side, open(f"{tmp}/gold_pairs_fingerprints.json", "w"))
    stats.update({"num_instances": len(lens_all), "num_tokens": int(tot_tok), "mean_row_len": float(np.mean(lens_all)),
                  "max_example_len": int(max(lens_all))})
    json.dump(meta, open(f"{tmp}/metadata.json", "w"), indent=1)
    # ---- training side of every pool, for the held-out checks ---------------------------------------------------
    tr_prompt = set()
    used = {}
    for p in ALL_POOLS:
        for line in open(f"{SH}/{p}/row_info.jsonl"):
            x = json.loads(line)
            used[(x["src"], x["src_idx"])] = (x["task"], x["gold"])
    by_src = {}
    for (s, i), (t, g) in used.items():
        if t in TASKS:
            by_src.setdefault(s, {})[i] = (t, g)
    for s, idx in by_src.items():
        with open(s) as f:
            for i, line in enumerate(f):
                if i in idx:
                    t, g = idx[i]
                    ex = json.loads(line)
                    tr_gold[t] |= gold_h(t, ex, g)
                    tr_docs[t].update(P.thash(dd.get("text")) for dd in ex["documents"])
    for p in ALL_POOLS + ["mtb64k_tmp"]:
        d = f"{SH}/{p}"
        mt = json.load(open(f"{d}/metadata.json"))
        for f in sorted(glob.glob(f"{d}/token_ids_part_t*_000.npy")):
            ids = np.fromfile(f, dtype=mt["dtype"])
            mk = np.fromfile(f.replace("token_ids_part_", "labels_mask_"), dtype=mt["mask_dtype"])
            s = 0
            for e in np.flatnonzero(ids == P.EOS):
                tr_prompt.add(H.keys(ids[s:e + 1], mk[s:e + 1]))
                s = e + 1
    # ---- held-out v4 ------------------------------------------------------------------------------------------------
    ids_all, mask_all, offs, sets, tasks, src_idx = [], [], [0], [], [], []

    def put(set_name, t, i, r_, m_):
        ids_all.append(np.asarray(r_, dtype=np.uint32))
        mask_all.append(np.asarray(m_, dtype=bool))
        offs.append(offs[-1] + len(r_))
        sets.append(set_name)
        tasks.append(t)
        src_idx.append(int(i))

    for t in TASKS:
        for r in RUNGS:
            src, exs, rows = ho[(t, r)]
            d = {"generated": len(exs), "over_cap": 0, "gold_in_train": 0}
            picked = []
            for i, ((ids, m), ex) in enumerate(zip(rows, exs)):
                if len(ids) > CAP:
                    d["over_cap"] += 1
                    continue
                gh = gold_h(t, ex)
                if gh & tr_gold[t] or (t == "outlier" and gh & tr_docs[t]):
                    d["gold_in_train"] += 1
                    continue
                picked.append((i, ids, m, ex))
            sel = picked[:HO_ROWS]
            L = np.array([len(x[1]) for x in sel])
            st = {"rows": len(sel), "source_file": src, "split": "eval", "eval_seed": EVAL_SEED, "drops": d,
                  "len_mean": float(L.mean()), "len_min": int(L.min()), "len_max": int(L.max()),
                  "answer_tokens_mean": float(np.mean([int(x[2].sum()) for x in sel])),
                  "prompt_in_train": sum(H.keys(x[1], x[2]) in tr_prompt for x in sel),
                  "rows_with_gold_in_train_gold": sum(bool(gold_h(t, x[3]) & tr_gold[t]) for x in sel),
                  "rows_with_gold_in_train_docs": sum(bool(gold_h(t, x[3]) & tr_docs[t]) for x in sel),
                  "doc_share_in_train_mean": float(np.mean([np.mean([P.thash(dd.get("text")) in tr_docs[t]
                                                                      for dd in x[3]["documents"]]) for x in sel]))}
            stats["heldout"][f"gen{r}/{t}"] = st
            print(f"[long] heldout gen{r}/{t}: {json.dumps(st)}", flush=True)
            for i, ids, m, _ in sel:
                put(f"gen{r}", t, i, ids, m)
    h3 = np.load("/data/prasann/early_loss_proxy/heldout_v3.npz")
    o = h3["offsets"]
    for j in np.flatnonzero(h3["sets"] == "gen16k"):
        put("gen16k", str(h3["tasks"][j]), h3["src_idx"][j], h3["ids"][o[j]:o[j + 1]], h3["mask"][o[j]:o[j + 1]])
    stats["heldout"]["gen16k"] = {"rows": int((h3["sets"] == "gen16k").sum()), "source": "heldout_v3.npz"}
    out = "/data/prasann/early_loss_proxy/heldout_v4.npz"
    np.savez(out + ".tmp.npz", ids=np.concatenate(ids_all), mask=np.concatenate(mask_all), offsets=np.asarray(offs),
             sets=np.asarray(sets), tasks=np.asarray(tasks), src_idx=np.asarray(src_idx))
    json.dump(stats, open(f"{final}_stats.json", "w"), indent=1)
    json.dump(stats["heldout"], open(out.replace(".npz", "_stats.json"), "w"), indent=1)
    os.rename(tmp, final)
    os.replace(out + ".tmp.npz", out)
    print(f"[long] wrote {final} ({len(lens_all)} rows, {tot_tok} tokens) and {out} ({len(sets)} rows)", flush=True)


QD_SEED, QD_ROWS = 7777, 6000


def hofix(a):
    """Rebuild heldout_v4's qdmatch_nq gen32k / gen64k: ctc's qdmatch EVAL split re-uses gold pairs of the training
    pools (only 3 / 6 of 120 rows were clean), so draw them like heldout_v3's gen16k instead: ``--split train`` seed
    7777, gold pairs absent from every training pool's gold (incl. mtb64k). Other rows of heldout_v4 are kept as is."""
    import tempfile

    import build_heldout_v3 as H

    t = "qdmatch_nq"
    n_docs = json.load(open(f"{LW}/n_docs_long.json"))
    tr_gold, tr_docs, tr_prompt = set(), set(), set()
    by_src = {}
    for p in ALL_POOLS + ["mtb64k"]:
        for line in open(f"{SH}/{p}/row_info.jsonl"):
            x = json.loads(line)
            if x["task"] == t:
                by_src.setdefault(x["src"], {})[x["src_idx"]] = x["gold"]
        d = f"{SH}/{p}"
        mt = json.load(open(f"{d}/metadata.json"))
        for f in sorted(glob.glob(f"{d}/token_ids_part_t*_000.npy")):
            ids = np.fromfile(f, dtype=mt["dtype"])
            mk = np.fromfile(f.replace("token_ids_part_", "labels_mask_"), dtype=mt["mask_dtype"])
            s0 = 0
            for e in np.flatnonzero(ids == P.EOS):
                tr_prompt.add(H.keys(ids[s0:e + 1], mk[s0:e + 1]))
                s0 = e + 1
    for s_, idx in by_src.items():
        with open(s_) as f:
            for i, line in enumerate(f):
                if i in idx:
                    ex = json.loads(line)
                    tr_gold.update(gold_h(t, ex, idx[i]))
                    tr_docs.update(P.thash(dd.get("text")) for dd in ex["documents"])
    out = "/data/prasann/early_loss_proxy/heldout_v4.npz"
    h = np.load(out)
    o = h["offsets"]
    keep = [j for j in range(len(h["sets"])) if not (h["tasks"][j] == t and h["sets"][j] in ("gen32k", "gen64k"))]
    rows = [(str(h["sets"][j]), str(h["tasks"][j]), int(h["src_idx"][j]), h["ids"][o[j]:o[j + 1]], h["mask"][o[j]:o[j + 1]])
            for j in keep]
    hs = json.load(open(out.replace(".npz", "_stats.json")))
    for r in RUNGS:
        n = n_docs[f"{t}:{r}"]["n_docs"]
        outd = f"{LW}/heldout_gen_tr/{t}_{r}"
        src = f"{outd}/{t}/train.jsonl"
        if not os.path.exists(src):
            cmd = [sys.executable, f"{P.REPO}/recipes/data/ctc_sft/ctc_calibrated.py", "build", "--task", t, "--split",
                   "train", "--rungs", r, "--train", str(QD_ROWS), "--seed", str(QD_SEED), "--out", outd, "--pool", P.POOL[t]]
            e = P.env()
            e["CTC_DOCS_OVERRIDE"] = json.dumps({f"{t}:{r}": n})
            subprocess.run(cmd, env=e, check=True, capture_output=True, text=True)
        exs = P.read_jsonl(src)
        cand = [i for i, ex in enumerate(exs) if not (gold_h(t, ex) & tr_gold)]
        d = {"generated": len(exs), "gold_in_train_gold": len(exs) - len(cand), "over_cap": 0}
        with tempfile.TemporaryDirectory(dir="/data/prasann/tmp") as tmp:
            with open(f"{tmp}/c.jsonl", "w") as g:
                for i in cand[:HO_ROWS + 8]:
                    g.write(json.dumps(exs[i]) + "\n")
            os.makedirs(f"{tmp}/o")
            convert(f"{tmp}/c.jsonl", t, f"{tmp}/o")
            _, trows = P.load_shard_rows(f"{tmp}/o")
        sel = []
        for i, (ids, m) in zip(cand[:HO_ROWS + 8], trows):
            if len(ids) > CAP:
                d["over_cap"] += 1
                continue
            sel.append((i, ids, m, exs[i]))
        sel = sel[:HO_ROWS]
        L = np.array([len(x[1]) for x in sel])
        st = {"rows": len(sel), "source_file": src, "split": "train", "seed": QD_SEED, "drops": d,
              "content_eligible": len(cand), "len_mean": float(L.mean()), "len_min": int(L.min()), "len_max": int(L.max()),
              "answer_tokens_mean": float(np.mean([int(x[2].sum()) for x in sel])),
              "prompt_in_train": sum(H.keys(x[1], x[2]) in tr_prompt for x in sel),
              "rows_with_gold_in_train_gold": sum(bool(gold_h(t, x[3]) & tr_gold) for x in sel),
              "rows_with_gold_in_train_docs": sum(bool(gold_h(t, x[3]) & tr_docs) for x in sel),
              "doc_share_in_train_mean": float(np.mean([np.mean([P.thash(dd.get("text")) in tr_docs
                                                                  for dd in x[3]["documents"]]) for x in sel])),
              "note": "eval split re-used training gold pairs (3 / 6 of 120 clean); drawn from --split train seed 7777"}
        hs[f"gen{r}/{t}"] = st
        print(f"[hofix] gen{r}/{t}: {json.dumps(st)}", flush=True)
        rows += [(f"gen{r}", t, i, ids, m) for i, ids, m, _ in sel]
    order_ = {"gen32k": 0, "gen64k": 1, "gen16k": 2}
    tord = {x: i for i, x in enumerate(P.TASKS)}
    rows.sort(key=lambda x: (order_[x[0]], tord[x[1]]))
    offs = np.concatenate([[0], np.cumsum([len(x[3]) for x in rows])])
    np.savez(out + ".tmp.npz", ids=np.concatenate([np.asarray(x[3], dtype=np.uint32) for x in rows]),
             mask=np.concatenate([np.asarray(x[4], dtype=bool) for x in rows]), offsets=offs,
             sets=np.asarray([x[0] for x in rows]), tasks=np.asarray([x[1] for x in rows]),
             src_idx=np.asarray([x[2] for x in rows]))
    json.dump(hs, open(out.replace(".npz", "_stats.json"), "w"), indent=1)
    st_pool = json.load(open(f"{SH}/mtb64k_stats.json"))
    st_pool["heldout"] = hs
    json.dump(st_pool, open(f"{SH}/mtb64k_stats.json", "w"), indent=1)
    os.replace(out + ".tmp.npz", out)
    print(f"[hofix] wrote {out}: {len(rows)} rows", flush=True)


def order(a):
    P.MAX_LEN = CAP
    a.out = f"{SH}/mtb64k"
    P.order(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["fetch", "gen", "tokenize", "build", "order", "hofix"])
    ap.add_argument("--shuffle-seed", type=int, default=20261119)
    ap.add_argument("--verify-rows", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"fetch": fetch, "gen": gen, "tokenize": tokenize, "build": build, "order": order, "hofix": hofix}[a.stage](a)


if __name__ == "__main__":
    main()
