"""Single-length pools ``kl16k`` / ``kl32k`` / ``kl64k`` and held-out ``heldout_kl.npz`` (keepfrac_len/README.md).

Same generator, split, seed, converter flags, layout and sidecars as ``mtb16k`` / ``mtb64k`` (build_pool_v2.py,
build_long_pools.py); only the size differs: every pool holds ``ROWS`` rows per task of ONE length cell, tasks
contradiction / outlier / qdmatch_nq (t0..t2; reorder has no suite rows past 16k, so it is left out at every length,
16k included, to keep the task mix fixed across lengths). Documents per row: 16k = mtb16k's refit cells (324 / 96 / 174),
32k / 64k = mtb64k's cells (651 / 194 / 324 and 1,304 / 388 / 650). ctc's train stream is prefix-stable, so the first
rows of each cell are mtb16k's / mtb64k's rows.

Held-out (sets ``kl16k`` / ``kl32k`` / ``kl64k``, ``HO_ROWS`` rows per task, picked BEFORE the training rows):
  16k: heldout_v3 ``gen16k`` (48 rows / task, the three tasks);
  32k / 64k: heldout_v4's candidates with heldout_v4's rule (contradiction / outlier: ctc EVAL split eval seed 99, the
  first rows whose gold is absent from every existing pool's gold (outlier: from every document); qdmatch_nq: ``--split
  train`` seed 7777, gold pairs absent from every existing pool's gold). The first 24 rows per task are heldout_v4's.
Training rows are dropped if a gold item occurs in: any suite eval rung r2k-r64k of the task, heldout_v2/v3 rows,
gen16k, any heldout_v4 candidate (all 120 eval-split rows), any kl held-out row; outlier rows also if ANY of their
documents is a held-out outlier passage.

    $PY build_kl_pools.py gen ; ... heldout ; ... tokenize ; ... build ; ... order
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ELP = os.path.dirname(HERE)
sys.path.insert(0, ELP)
import build_pool_v2 as P  # noqa: E402
import build_short_pools as SP  # noqa: E402
import build_long_pools as LP  # noqa: E402

TASKS = ["contradiction", "outlier", "qdmatch_nq"]
#: row cap (= the runs' --seq-len) per length; non-power-of-2 caps are fine (padded, no packing)
ALL_LENS = {"2k": 4096, "4k": 6144, "8k": 12288, "16k": 16384, "32k": 32768, "64k": 65536}
GROUPS = {"long": ["16k", "32k", "64k"], "short": ["2k", "4k", "8k"]}
LENS = {r: ALL_LENS[r] for r in GROUPS["long"]}  # set by --group
ROWS = 1200
HO_ROWS = 48
GEN = {"contradiction": 1400, "outlier": 1400,
       "qdmatch_nq": {"2k": 5000, "4k": 5000, "8k": 5000, "16k": 4500, "32k": 9000, "64k": 9000}}
KW = f"{P.WORK}/kl"
SH = "/data/prasann/early_loss_proxy/shards"
HO_FILES = {"long": "/data/prasann/early_loss_proxy/heldout_kl.npz", "short": "/data/prasann/early_loss_proxy/heldout_kls.npz"}
EXCL_FILES = {"long": f"{KW}/heldout_excl.json", "short": f"{KW}/heldout_excl_short.json"}
HO_OUT = HO_FILES["long"]
EXCL = EXCL_FILES["long"]
HO_SEED_EVAL, HO_GEN_EVAL, QD_SEED, QD_ROWS = 99, 120, 7777, 6000  # as build_long_pools (heldout_v4)
EXISTING = LP.ALL_POOLS + ["mtb64k"]


def n_docs(t, r):
    if r == "2k":
        return int(json.load(open(f"{P.WORK}/n_docs_short.json"))[f"{t}:2k"]["n_docs"])
    if r in ("4k", "8k", "16k"):
        return int(json.load(open(P.N_DOCS_FILE))[f"{t}:{r}"]["n_docs"])
    return int(json.load(open(f"{LP.LW}/n_docs_long.json"))[f"{t}:{r}"]["n_docs"])


def gen_rows(t, r):
    g = GEN[t]
    return g[r] if isinstance(g, dict) else g


def gen_src(t, r):
    return f"{KW}/gen/{t}_{r}/{t}/train.jsonl"


# ---------------------------------------------------------------------------------------------------------------- gen
def _gen(job):
    t, r = job
    out = f"{KW}/gen/{t}_{r}"
    if os.path.exists(gen_src(t, r)) and os.path.getsize(gen_src(t, r)):
        return t, r, "cached", 0.0
    t0 = time.time()
    cmd = [sys.executable, f"{P.REPO}/recipes/data/ctc_sft/ctc_calibrated.py", "build", "--task", t, "--split", "train",
           "--rungs", r, "--train", str(gen_rows(t, r)), "--seed", str(P.SEED), "--out", out, "--pool", P.POOL[t]]
    e = P.env()
    e["CTC_DOCS_OVERRIDE"] = json.dumps({f"{t}:{r}": n_docs(t, r)})
    p = subprocess.run(cmd, env=e, capture_output=True, text=True)
    open(f"{out}.log", "w").write(" ".join(cmd) + "\n" + p.stdout + p.stderr)
    ok = p.returncode == 0 and os.path.exists(gen_src(t, r))
    return t, r, "built" if ok else "FAILED " + (p.stdout + p.stderr)[-600:], time.time() - t0


def gen(a):
    os.makedirs(f"{KW}/gen", exist_ok=True)
    jobs = [(t, r) for t in TASKS for r in LENS]
    # largest cells first so they do not set the tail
    jobs.sort(key=lambda j: -gen_rows(*j) * n_docs(*j))
    with cf.ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for res in ex.map(_gen, jobs):
            print(f"[kl gen] {res[0]}@{res[1]} n_docs={n_docs(res[0], res[1])} {res[2]} ({res[3]:.0f}s)", flush=True)


# ------------------------------------------------------------------------------------------------------------ helpers
def gold_h(t, ex, g=None):
    return LP.gold_h(t, ex, g)


def existing_train_sets():
    """Per task: gold hashes and all-document hashes of every row of every existing pool, and prompt keys."""
    import build_heldout_v3 as H

    tr_gold = {t: set() for t in TASKS}
    tr_docs = {t: set() for t in TASKS}
    by_src = {}
    for p in EXISTING:
        for line in open(f"{SH}/{p}/row_info.jsonl"):
            x = json.loads(line)
            if x["task"] in TASKS:
                by_src.setdefault(x["src"], {})[x["src_idx"]] = (x["task"], x["gold"])
    for s_, idx in by_src.items():
        with open(s_) as f:
            for i, line in enumerate(f):
                if i in idx:
                    t, g = idx[i]
                    ex = json.loads(line)
                    tr_gold[t] |= gold_h(t, ex, g)
                    tr_docs[t].update(P.thash(dd.get("text")) for dd in ex["documents"])
    return tr_gold, tr_docs, H


def tokenize_rows(exs, t, tmpdir):
    with open(f"{tmpdir}/c.jsonl", "w") as g:
        for ex in exs:
            g.write(json.dumps(ex) + "\n")
    os.makedirs(f"{tmpdir}/o")
    LP.convert(f"{tmpdir}/c.jsonl", t, f"{tmpdir}/o")
    _, rows = P.load_shard_rows(f"{tmpdir}/o")
    assert len(rows) == len(exs)
    return rows


# ------------------------------------------------------------------------------------------------------------ hogen
def _hogen(job):
    """Held-out candidates for a short length, generated as build_long_pools did for 32k / 64k (heldout_v4)."""
    t, r = job
    e = P.env()
    e["CTC_DOCS_OVERRIDE"] = json.dumps({f"{t}:{r}": n_docs(t, r)})
    if t == "qdmatch_nq":
        out = f"{KW}/heldout_gen_tr/{t}_{r}"
        target = f"{out}/{t}/train.jsonl"
        cmd = ["--split", "train", "--train", str(QD_ROWS), "--seed", str(QD_SEED)]
    else:
        out = f"{KW}/heldout_gen/{t}_{r}"
        target = f"{out}/{t}/eval_{r}.jsonl"
        cmd = ["--split", "eval", "--eval-size", str(HO_GEN_EVAL), "--allow-small-eval", "--eval-seed", str(HO_SEED_EVAL)]
    if not (os.path.exists(target) and os.path.getsize(target)):
        cmd = [sys.executable, f"{P.REPO}/recipes/data/ctc_sft/ctc_calibrated.py", "build", "--task", t, "--rungs", r] + cmd \
            + ["--out", out, "--pool", P.POOL[t]]
        p = subprocess.run(cmd, env=e, capture_output=True, text=True)
        open(f"{out}.log", "w").write(" ".join(cmd) + "\n" + p.stdout + p.stderr)
        if p.returncode != 0 or not os.path.exists(target):
            return t, r, "FAILED " + (p.stdout + p.stderr)[-600:]
    if t != "qdmatch_nq":
        tok = f"{KW}/heldout_tok/{t}_{r}"
        if not os.path.exists(f"{tok}/metadata.json"):
            os.makedirs(tok, exist_ok=True)
            LP.convert(target, t, tok)
    return t, r, "ok"


def hogen(a):
    jobs = [(t, r) for t in TASKS for r in LENS if r not in ("16k", "32k", "64k")]
    with cf.ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for res in ex.map(_hogen, jobs):
            print(f"[kl hogen] {res}", flush=True)


# ------------------------------------------------------------------------------------------------------------ heldout
def heldout(a):
    tr_gold, tr_docs, H = existing_train_sets()
    sel_rows = []  # (set, task, src_idx, ids, mask)
    excl = {t: set() for t in TASKS}  # gold hashes training rows must not contain as gold
    ho_outlier_gold = set()  # outlier: training rows must not contain these anywhere
    stats = {}
    # 16k: heldout_v3 gen16k
    h3 = np.load("/data/prasann/early_loss_proxy/heldout_v3.npz")
    o = h3["offsets"]
    for t in (TASKS if "16k" in LENS else []):
        js = [j for j in np.flatnonzero(h3["sets"] == "gen16k") if h3["tasks"][j] == t]
        assert len(js) == HO_ROWS, (t, len(js))
        for j in js:
            sel_rows.append(("kl16k", t, int(h3["src_idx"][j]), h3["ids"][o[j]:o[j + 1]], h3["mask"][o[j]:o[j + 1]]))
        g16, _ = SP.gen16k_keys(t)
        excl[t] |= g16
        if t == "outlier":
            ho_outlier_gold |= g16
        L = [o[j + 1] - o[j] for j in js]
        stats[f"kl16k/{t}"] = {"rows": len(js), "source": "heldout_v3.npz gen16k", "len_mean": float(np.mean(L)),
                               "len_min": int(min(L)), "len_max": int(max(L))}
    for r in [x for x in LENS if x != "16k"]:
        cap = LENS[r]
        hgen, htok = (f"{LP.LW}/heldout_gen", f"{LP.LW}/heldout_tok") if r in ("32k", "64k") else \
            (f"{KW}/heldout_gen", f"{KW}/heldout_tok")
        hgen_tr = f"{LP.LW}/heldout_gen_tr" if r in ("32k", "64k") else f"{KW}/heldout_gen_tr"
        for t in TASKS:
            d = {"over_cap": 0, "gold_in_train": 0}
            if t == "qdmatch_nq":
                src = f"{hgen_tr}/{t}_{r}/{t}/train.jsonl"
                exs = P.read_jsonl(src)
                cand = [i for i, ex in enumerate(exs) if not (gold_h(t, ex) & tr_gold[t])]
                d["generated"], d["content_eligible"] = len(exs), len(cand)
                d["gold_in_train"] = len(exs) - len(cand)
                take = cand[:HO_ROWS + 12]
                with tempfile.TemporaryDirectory(dir="/data/prasann/tmp") as tmp:
                    rows = tokenize_rows([exs[i] for i in take], t, tmp)
                pairs = list(zip(take, rows))
            else:
                src = f"{hgen}/{t}_{r}/{t}/eval_{r}.jsonl"
                exs = P.read_jsonl(src)
                _, rows = P.load_shard_rows(f"{htok}/{t}_{r}")
                assert len(rows) == len(exs)
                d["generated"] = len(exs)
                pairs = list(enumerate(rows))
                for ex in exs:  # every eval-split candidate is excluded from training (as build_long_pools did)
                    excl[t] |= gold_h(t, ex)
            picked = []
            for i, (ids, m) in pairs:
                ex = exs[i]
                if len(ids) > cap:
                    d["over_cap"] += 1
                    continue
                gh = gold_h(t, ex)
                if gh & tr_gold[t] or (t == "outlier" and gh & tr_docs[t]):
                    d["gold_in_train"] += 1
                    continue
                picked.append((i, ids, m, ex))
            sel = picked[:HO_ROWS]
            assert len(sel) == HO_ROWS, (t, r, len(sel), d)
            for i, ids, m, ex in sel:
                excl[t] |= gold_h(t, ex)
                if t == "outlier":
                    ho_outlier_gold |= gold_h(t, ex)
                sel_rows.append((f"kl{r}", t, i, ids, m))
            L = np.array([len(x[1]) for x in sel])
            stats[f"kl{r}/{t}"] = {
                "rows": len(sel), "source_file": src, "drops": d, "len_mean": float(L.mean()), "len_min": int(L.min()),
                "len_max": int(L.max()), "answer_tokens_mean": float(np.mean([int(x[2].sum()) for x in sel])),
                "n_docs": int(np.mean([len(x[3]["documents"]) for x in sel])),
                "rows_with_gold_in_existing_train_docs": sum(bool(gold_h(t, x[3]) & tr_docs[t]) for x in sel),
                "doc_share_in_existing_train_mean": float(np.mean([np.mean([P.thash(dd.get("text")) in tr_docs[t]
                                                                            for dd in x[3]["documents"]]) for x in sel]))}
            print(f"[kl heldout] kl{r}/{t}: {json.dumps(stats[f'kl{r}/{t}'])}", flush=True)
    order_ = {f"kl{r}": i for i, r in enumerate(ALL_LENS)}
    sel_rows.sort(key=lambda x: (order_[x[0]], TASKS.index(x[1])))
    offs = np.concatenate([[0], np.cumsum([len(x[3]) for x in sel_rows])])
    np.savez(HO_OUT + ".tmp.npz", ids=np.concatenate([np.asarray(x[3], dtype=np.uint32) for x in sel_rows]),
             mask=np.concatenate([np.asarray(x[4], dtype=bool) for x in sel_rows]), offsets=offs,
             sets=np.asarray([x[0] for x in sel_rows]), tasks=np.asarray([x[1] for x in sel_rows]),
             src_idx=np.asarray([x[2] for x in sel_rows]))
    os.replace(HO_OUT + ".tmp.npz", HO_OUT)
    json.dump(stats, open(HO_OUT.replace(".npz", "_stats.json"), "w"), indent=1)
    json.dump({"excl": {t: sorted(v) for t, v in excl.items()}, "outlier_anywhere": sorted(ho_outlier_gold)},
              open(EXCL, "w"))
    print(f"[kl heldout] wrote {HO_OUT}: {len(sel_rows)} rows", flush=True)


# ----------------------------------------------------------------------------------------------------------- tokenize
def train_excl():
    ex = json.load(open(EXCL))
    excl = {t: set(v) for t, v in ex["excl"].items()}
    for t in TASKS:
        for f in [f"{LP.LW}/suite/{LP.SUB[t]}_{r}.jsonl" for r in LP.HF_RUNGS] + SP.eval_files(t):
            g2, _, _ = P.eval_content_keys(t, P.read_jsonl(f))
            excl[t] |= g2
        excl[t] |= SP.gen16k_keys(t)[0]
    return excl, set(ex["outlier_anywhere"])


def _tok(job):
    t, r, idx = job
    out = f"{KW}/tok/{t}_{r}"
    if os.path.exists(f"{out}/metadata.json"):
        return t, r, "cached"
    exs = P.read_jsonl(gen_src(t, r))
    os.makedirs(f"{KW}/elig", exist_ok=True)
    with open(f"{KW}/elig/{t}_{r}.jsonl", "w") as g:
        for i in idx:
            g.write(json.dumps(exs[i]) + "\n")
    os.makedirs(out, exist_ok=True)
    LP.convert(f"{KW}/elig/{t}_{r}.jsonl", t, out)
    return t, r, "built"


def tokenize(a):
    excl, out_any = train_excl()
    jobs, drops = [], {}
    for t in TASKS:
        for r in LENS:
            exs = P.read_jsonl(gen_src(t, r))
            d = {"generated": len(exs), "gold_in_eval_or_heldout": 0, "heldout_outlier_anywhere": 0}
            idx = []
            for i, ex in enumerate(exs):
                if gold_h(t, ex) & excl[t]:
                    d["gold_in_eval_or_heldout"] += 1
                    continue
                if t == "outlier" and {P.thash(dd.get("text")) for dd in ex["documents"]} & out_any:
                    d["heldout_outlier_anywhere"] += 1
                    continue
                idx.append(i)
            d["eligible"] = len(idx)
            need = ROWS + max(30, ROWS // 20)
            if len(idx) < ROWS:
                raise SystemExit(f"{t}@{r}: only {len(idx)} eligible < {ROWS}: {d}")
            d["tokenized"] = min(len(idx), need)
            drops[f"{t}/{r}"] = d
            json.dump(idx[:need], open(f"{KW}/elig_{t}_{r}_src_idx.json", "w"))
            jobs.append((t, r, idx[:need]))
            print(f"[kl tok] {t}@{r}: {d}", flush=True)
    json.dump(drops, open(f"{KW}/drops.json", "w"), indent=1)
    with cf.ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for res in ex.map(_tok, jobs):
            print(f"[kl tok] {res}", flush=True)


# -------------------------------------------------------------------------------------------------------------- build
def build(a):
    from tokenizers import Tokenizer

    sys.path.insert(0, f"{P.REPO}/src")
    from olmo_core.nn.attention.gold_grad_mask import content_fingerprint
    import build_heldout_v3 as H

    tok = Tokenizer.from_file(f"{P.TOKENIZER}/tokenizer.json")
    dec = lambda x: tok.decode([int(i) for i in x], skip_special_tokens=False)  # noqa: E731
    drops = json.load(open(f"{KW}/drops.json"))
    hk = np.load(HO_OUT)
    ho_keys = {}
    ho_o = hk["offsets"]
    for j in range(len(hk["sets"])):
        ho_keys.setdefault(str(hk["sets"][j]), set()).add(H.keys(hk["ids"][ho_o[j]:ho_o[j + 1]], hk["mask"][ho_o[j]:ho_o[j + 1]]))
    for r, cap in LENS.items():
        if a.lens and r not in a.lens.split(","):
            continue
        name = f"kl{r}"
        tmp, final = f"{SH}/{name}_tmp", f"{SH}/{name}"
        if os.path.exists(f"{final}/metadata.json"):
            print(f"[kl build] {final} exists, skip", flush=True)
            continue
        os.system(f"rm -rf {tmp}")
        os.makedirs(tmp)
        stats = {"pool": name, "cap": cap, "tasks": TASKS, "rows_per_task": ROWS, "cells": {}}
        gold_side, pair_side, lens_all, merged_from = {}, {}, [], []
        tot_tok = tot_loss = 0
        contra_fps, tr_keys = [], set()
        info_f = open(f"{tmp}/row_info.jsonl", "w")
        for ti, t in enumerate(TASKS):
            src = gen_src(t, r)
            idx = json.load(open(f"{KW}/elig_{t}_{r}_src_idx.json"))
            exs_all = P.read_jsonl(src)
            meta, rows = P.load_shard_rows(f"{KW}/tok/{t}_{r}")
            assert len(rows) == len(idx) and meta["num_dropped"] == 0, (t, r, len(rows), len(idx))
            kept, over = [], 0
            for (ids, m), i in zip(rows, idx):
                if len(ids) > cap:
                    over += 1
                    continue
                kept.append((ids, m, exs_all[i], i))
            if len(kept) < ROWS:
                raise SystemExit(f"{t}@{r}: {len(kept)} rows under the cap < {ROWS}")
            sel = kept[:ROWS]
            ver = {"rows_checked": 0, "span_text_match": 0, "printed_id_is_index_plus_1": 0, "answer_ids_match_gold": 0}
            for ids, m, ex, i in sel[:a.verify_rows]:
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
            stats["cells"][t] = {"rows": ROWS, "n_docs": n_docs(t, r), "source_file": src, "len_mean": float(L.mean()),
                                 "len_min": int(L.min()), "len_max": int(L.max()), "tokens": int(L.sum()),
                                 "answer_tokens_mean": float(A.mean()), "drops": dict(drops[f"{t}/{r}"], over_cap=over),
                                 "gold_verification": ver}
            print(f"[kl build] {name} {t}: {json.dumps(stats['cells'][t])}", flush=True)
            perm = np.random.default_rng(a.shuffle_seed + ti).permutation(ROWS)
            sel = [sel[j] for j in perm]
            for k, (ids, m, ex, i) in enumerate(sel):
                g, pairs = P.gold_info(t, ex)
                spans = P.doc_spans(ids)
                fp = content_fingerprint(ids.tolist())
                assert fp not in gold_side
                gold_side[fp] = g
                if pairs is not None:
                    pair_side[fp] = pairs
                if t == "contradiction":
                    contra_fps.append(fp)
                tr_keys.add(H.keys(ids, m))
                info_f.write(json.dumps({"part": ti, "row": k, "task": t, "rung": r, "src": src, "src_idx": i,
                                         "n_docs": len(spans), "len": int(len(ids)), "answer_tokens": int(m.sum()),
                                         "fingerprint": fp, "gold": g, "gold_pairs": pairs, "doc_spans": spans}) + "\n")
                lens_all.append(len(ids))
            bt = np.concatenate([x[0] for x in sel]).astype("uint32")
            bm = np.concatenate([x[1] for x in sel]).astype("bool")
            bt.tofile(f"{tmp}/token_ids_part_t{ti}_000.npy")
            bm.tofile(f"{tmp}/labels_mask_t{ti}_000.npy")
            tot_tok += bt.size
            tot_loss += int(bm.sum())
            merged_from.append({"src": f"{KW}/tok/{t}_{r}", "task": t, "converter_task": P.SPEC[t], "num_instances": ROWS})
        info_f.close()
        meta = json.load(open(f"{SH}/mtb64k/metadata.json"))
        meta.update({"task": name, "num_instances": len(lens_all), "num_tokens": int(tot_tok), "num_loss_tokens": int(tot_loss),
                     "max_example_len": int(max(lens_all)), "min_example_len": int(min(lens_all)),
                     "merged_from": merged_from, "shuffle_seed": a.shuffle_seed, "subset_max_len": cap,
                     "tasks_order": TASKS})
        json.dump(meta, open(f"{tmp}/metadata.json", "w"), indent=1)
        json.dump(gold_side, open(f"{tmp}/gold_fingerprints.json", "w"))
        json.dump(pair_side, open(f"{tmp}/gold_pairs_fingerprints.json", "w"))
        json.dump(contra_fps, open(f"{tmp}/gdd_only_contradiction.json", "w"))
        stats.update({"num_instances": len(lens_all), "num_tokens": int(tot_tok), "mean_row_len": float(np.mean(lens_all)),
                      "max_example_len": int(max(lens_all)),
                      "heldout_prompt_hits": {s: len(v & tr_keys) for s, v in ho_keys.items()}})
        json.dump(stats, open(f"{SH}/{name}_stats.json", "w"), indent=1)
        os.rename(tmp, final)
        print(f"[kl build] wrote {final}: {len(lens_all)} rows, {tot_tok} tokens; prompt hits {stats['heldout_prompt_hits']}",
              flush=True)


def order(a):
    for r, cap in LENS.items():
        if a.lens and r not in a.lens.split(","):
            continue
        P.MAX_LEN = cap
        a.out = f"{SH}/kl{r}"
        P.order(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["gen", "hogen", "heldout", "tokenize", "build", "order"])
    ap.add_argument("--group", choices=list(GROUPS), default="long")
    ap.add_argument("--jobs", type=int, default=9)
    ap.add_argument("--shuffle-seed", type=int, default=20261121)
    ap.add_argument("--verify-rows", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lens", default="", help="build/order only these lengths (e.g. 32k,64k)")
    a = ap.parse_args()
    global LENS, HO_OUT, EXCL
    LENS = {r: ALL_LENS[r] for r in GROUPS[a.group]}
    HO_OUT, EXCL = HO_FILES[a.group], EXCL_FILES[a.group]
    {"gen": gen, "hogen": hogen, "heldout": heldout, "tokenize": tokenize, "build": build, "order": order}[a.stage](a)


if __name__ == "__main__":
    main()
