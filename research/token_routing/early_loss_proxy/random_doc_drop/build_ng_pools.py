"""16k no-gold pool ``ng16k`` (reorder / oolong / grouping, t0..t2, 1,200 rows each), held-out ``heldout_ng.npz`` set
``ng16k`` (48 rows per task), and random-document-drop VARIANT pools of it (README.md in this folder).

Base rows per task (kl-pool conventions: one length cell, cap 16,384 tokens, converter --emit dense, qwen3_5 markers,
query after, no CoT, documents wrapped; no gold sidecar since these tasks have no small gold set):
* reorder: ctc ``reorder`` train split, seed 42, 90 passages (= mtb16k's 16k refit), 1,500 generated; rows whose book
  is the book of a held-out row are dropped. Held-out = heldout_v3 ``gen16k`` reorder rows (48, already tokenized).
* oolong: ctc's oolong generator called directly (``build_example``) over the seed pool's TRAIN item split, budget
  ``U(12000, 15500)`` (Qwen3 tokens), seed 42; held-out from the pool's item-disjoint EVAL split, seed 99.
  Converter ``--chunk-by line`` (each ``||`` item line is a chunk).
* grouping: the suite's unlabeled OpenAlex partition rows (``openalex_grouping_n88_levels_train_4000.jsonl``, the
  generator ctc_grouping's ladder came from). 88 abstracts overshoot 16,384 tokens, so each row is cut to a fixed
  random 72-abstract subset with the exact relabelling (``relabel.grouping_subset``): rows 0..1599 -> training
  candidates, rows 3700..3999 -> held-out candidates.
Training / held-out rows: first ROWS / HO_ROWS candidates per task at most CAP tokens. The builder checks that no
held-out prompt equals a training prompt.

Variant pool (``variants``): for every row slot of ng16k, with prob ``whole`` the row is copied unchanged, else a
random share r ~ U(lo, hi) of its documents (oolong: items) is kept and the answer rewritten (relabel.subset;
degenerate oolong subsets are redrawn up to 20 times, then the row stays whole and is counted). Same slots, same
parts, same order -> the trainer's loader permutation is the same as on ng16k.

    $PY build_ng_pools.py gen | base | variants --name rddA_s0 --lo 0.05 --hi 0.5 --whole 0.15 --seed 0 | order --pool P
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ELP = os.path.dirname(HERE)
sys.path.insert(0, ELP)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(ELP))), "ctc", "src"))
import build_pool_v2 as P  # noqa: E402
import relabel as R  # noqa: E402

TASKS = ["reorder", "oolong", "grouping"]
CONV = {"reorder": ("reorder", "document"), "oolong": ("oolong", "line"), "grouping": ("grouping", "document")}
CAP = 16384
ROWS, HO_ROWS = 1200, 48
W = "/data/prasann/early_loss_proxy/v2build/ng"
SH = "/data/prasann/early_loss_proxy/shards"
HO_OUT = "/data/prasann/early_loss_proxy/heldout_ng.npz"
GROUP_SRC = "/data/prasann/ctc_suite_data/grouping/openalex_grouping_n88_levels_train_4000.jsonl"
GROUP_DOCS = 72
OOLONG_POOL = "/data/prasann/ctc_seed_pools/oolong.seed.jsonl.gz"
REORDER_DOCS, REORDER_GEN = 90, 1500
OOLONG_GEN, OOLONG_HO_GEN = 1500, 90


def env():
    return P.env()


def oolong_pool():
    from pathlib import Path

    from ctc.data import seeds

    return seeds.load(Path(OOLONG_POOL), "oolong")


def ctx_for(task):
    return R.OolongCtx(oolong_pool()) if task == "oolong" else None


def convert(jsonl, task, out_dir, num_proc=8):
    spec, chunk = CONV[task]
    cmd = [sys.executable, f"{P.REPO}/recipes/data/convert_unified_to_document_landmark.py", "--input-jsonl", jsonl,
           "--task", spec, "--out-dir", out_dir, "--emit", "dense", "--marker-set", "qwen3_5",
           "--tokenizer", P.TOKENIZER, "--seq-len", "65536", "--query-position", "after", "--cot-mode", "none",
           "--chunk-by", chunk, "--num-proc", str(num_proc)]
    p = subprocess.run(cmd, cwd=P.REPO, env=env(), capture_output=True, text=True)
    open(f"{out_dir}.log", "w").write(" ".join(cmd) + "\n" + p.stdout[-20000:] + p.stderr[-20000:])
    if p.returncode != 0:
        raise RuntimeError(f"convert {jsonl} failed: {(p.stdout + p.stderr)[-1500:]}")
    meta, rows = P.load_shard_rows(out_dir)
    assert meta.get("num_dropped", 0) == 0, (out_dir, meta.get("num_dropped"))
    return rows


def write_jsonl(p, exs):
    with open(p, "w") as f:
        for ex in exs:
            f.write(json.dumps(ex) + "\n")


def tok_answer_check(task, exs, rows, n=None):
    """Decoded loss span of each tokenized row contains the example's answer (row order = example order)."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(f"{P.TOKENIZER}/tokenizer.json")
    bad = 0
    for ex, (ids, m) in list(zip(exs, rows))[:n]:
        ans = tok.decode([int(i) for i in ids[m]], skip_special_tokens=True)
        if ex["answers"][0].strip() not in ans:
            bad += 1
    return bad


# ---------------------------------------------------------------------------------------------------------------- gen
def gen(a):
    os.makedirs(W, exist_ok=True)
    # reorder via the ctc CLI (prefix-stable stream: the first 1,000 rows are mtb16k's 16k cell)
    out = f"{W}/gen/reorder_16k"
    tgt = f"{out}/reorder/train.jsonl"
    if not (os.path.exists(tgt) and sum(1 for _ in open(tgt)) >= REORDER_GEN):
        cmd = [sys.executable, f"{P.REPO}/recipes/data/ctc_sft/ctc_calibrated.py", "build", "--task", "reorder",
               "--split", "train", "--rungs", "16k", "--train", str(REORDER_GEN), "--seed", str(P.SEED), "--out", out,
               "--pool", P.POOL["reorder"]]
        e = env()
        e["CTC_DOCS_OVERRIDE"] = json.dumps({"reorder:16k": REORDER_DOCS})
        os.makedirs(os.path.dirname(out), exist_ok=True)
        p = subprocess.run(cmd, env=e, capture_output=True, text=True)
        open(f"{out}.log", "w").write(" ".join(cmd) + "\n" + p.stdout + p.stderr)
        assert p.returncode == 0, (p.stdout + p.stderr)[-2000:]
    print(f"[ng gen] reorder rows {sum(1 for _ in open(tgt))}", flush=True)
    # oolong: ctc's generator over the seed pool's train / eval item splits
    from ctc.tasks.oolong.generate import build_example

    pool = oolong_pool()
    for split, n, seed in (("train", OOLONG_GEN, 42), ("eval", OOLONG_HO_GEN, 99)):
        p_ = f"{W}/gen/oolong_{split}.jsonl"
        if os.path.exists(p_):
            continue
        sp = pool.for_split(split)
        rng = random.Random(seed)
        exs = []
        while len(exs) < n:
            ex = build_example(rng, corpus=sp, target_tokens=15500, min_tokens=12000)
            if ex is not None:
                exs.append(ex)
        write_jsonl(p_, exs)
        print(f"[ng gen] oolong {split}: {len(exs)} rows, items {np.mean([e['_meta']['num_items'] for e in exs]):.0f}",
              flush=True)


# --------------------------------------------------------------------------------------------------------------- base
def base_candidates(task):
    """(train candidates, held-out candidates or None) as unified examples."""
    if task == "reorder":
        sys.path.insert(0, ELP)
        import build_short_pools as SP

        _, ho_books = SP.gen16k_keys("reorder")
        exs = P.read_jsonl(f"{W}/gen/reorder_16k/reorder/train.jsonl")
        tr = [e for e in exs if P.book_of(e) not in ho_books]
        print(f"[ng base] reorder: {len(exs)} generated, {len(exs) - len(tr)} share a held-out book", flush=True)
        return tr, None
    if task == "oolong":
        return P.read_jsonl(f"{W}/gen/oolong_train.jsonl"), P.read_jsonl(f"{W}/gen/oolong_eval.jsonl")
    tr, ho = [], []
    with open(GROUP_SRC) as f:
        for i, line in enumerate(f):
            if i < 1600 or i >= 3700:
                ex = json.loads(line)
                R.selfcheck("grouping", ex)
                keep = sorted(random.Random(1000 + i).sample(range(len(ex["documents"])), GROUP_DOCS))
                ex = R.grouping_subset(ex, keep)
                ex.pop("rdd_kept", None)
                ex["ng_src_idx"] = i
                (tr if i < 1600 else ho).append(ex)
    return tr, ho


def base(a):
    import build_heldout_v3 as H

    os.makedirs(f"{W}/base", exist_ok=True)
    ctxs = {t: ctx_for(t) for t in TASKS}
    sel, ho_sel, stats = {}, [], {"cap": CAP, "tasks": TASKS}
    for t in TASKS:
        tr, ho = base_candidates(t)
        for ex in tr[:50] + (ho or [])[:20]:
            R.selfcheck(t, ex, ctxs[t])
        for kind, exs in (("train", tr), ("ho", ho)):
            if exs is None:
                continue
            d = f"{W}/base/tok_{t}_{kind}"
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
            write_jsonl(f"{W}/base/{t}_{kind}_cand.jsonl", exs)
            rows = convert(f"{W}/base/{t}_{kind}_cand.jsonl", t, d)
            assert len(rows) == len(exs)
            bad = tok_answer_check(t, exs, rows, 200)
            assert bad == 0, (t, kind, bad)
            fit = [(e, r) for e, r in zip(exs, rows) if len(r[0]) <= CAP]
            need = ROWS if kind == "train" else HO_ROWS
            assert len(fit) >= need, (t, kind, len(fit), len(exs))
            L = np.array([len(r[0]) for _, r in fit[:need]])
            stats[f"{t}/{kind}"] = {"candidates": len(exs), "fit_cap": len(fit), "rows": need, "len_mean": float(L.mean()),
                                    "len_min": int(L.min()), "len_max": int(L.max()),
                                    "units_mean": float(np.mean([R.n_units(t, e) for e, _ in fit[:need]])),
                                    "answer_tokens_mean": float(np.mean([int(r[1].sum()) for _, r in fit[:need]]))}
            print(f"[ng base] {t}/{kind}: {json.dumps(stats[f'{t}/{kind}'])}", flush=True)
            if kind == "train":
                sel[t] = fit[:need]
            else:
                ho_sel += [("ng16k", t, i, r[0], r[1]) for i, (e, r) in enumerate(fit[:need])]
        if t == "reorder":  # heldout_v3 gen16k reorder rows (already tokenized with the same converter flags)
            h = np.load("/data/prasann/early_loss_proxy/heldout_v3.npz")
            o = h["offsets"]
            js = [j for j in np.flatnonzero(h["sets"] == "gen16k") if h["tasks"][j] == "reorder"]
            assert len(js) == HO_ROWS
            ho_sel += [("ng16k", "reorder", int(h["src_idx"][j]), h["ids"][o[j]:o[j + 1]], h["mask"][o[j]:o[j + 1]])
                       for j in js]
    ho_sel.sort(key=lambda x: TASKS.index(x[1]))
    # pool: per task shuffled (fixed seed), parts t0..t2
    tmp, final = f"{SH}/ng16k_tmp", f"{SH}/ng16k"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    tr_keys, lens = set(), []
    info = open(f"{tmp}/row_info.jsonl", "w")
    tot = totl = 0
    for ti, t in enumerate(TASKS):
        perm = np.random.default_rng(a.shuffle_seed + ti).permutation(ROWS)
        rows = [sel[t][j] for j in perm]
        write_jsonl(f"{W}/base/{t}_pool_rows.jsonl", [e for e, _ in rows])  # slot order: variants read this
        for k, (e, (ids, m)) in enumerate(rows):
            tr_keys.add(H.keys(ids, m))
            info.write(json.dumps({"part": ti, "row": k, "task": t, "rung": "16k", "n_units": R.n_units(t, e),
                                   "len": int(len(ids)), "answer_tokens": int(m.sum())}) + "\n")
            lens.append(len(ids))
        bt = np.concatenate([r[0] for _, r in rows]).astype("uint32")
        bm = np.concatenate([r[1] for _, r in rows]).astype("bool")
        bt.tofile(f"{tmp}/token_ids_part_t{ti}_000.npy")
        bm.tofile(f"{tmp}/labels_mask_t{ti}_000.npy")
        tot += bt.size
        totl += int(bm.sum())
    info.close()
    hits = sum(H.keys(x[3], x[4]) in tr_keys for x in ho_sel)
    assert hits == 0, hits
    meta = json.load(open(f"{SH}/kl16k/metadata.json"))
    meta.update({"task": "ng16k", "num_instances": len(lens), "num_tokens": int(tot), "num_loss_tokens": int(totl),
                 "max_example_len": int(max(lens)), "min_example_len": int(min(lens)), "tasks_order": TASKS,
                 "merged_from": [{"task": t, "converter_task": CONV[t][0], "chunk_by": CONV[t][1], "num_instances": ROWS}
                                 for t in TASKS], "shuffle_seed": a.shuffle_seed, "subset_max_len": CAP})
    json.dump(meta, open(f"{tmp}/metadata.json", "w"), indent=1)
    stats.update({"num_instances": len(lens), "num_tokens": int(tot), "mean_row_len": float(np.mean(lens)),
                  "heldout_prompt_hits": hits})
    shutil.rmtree(final, ignore_errors=True)
    os.rename(tmp, final)
    json.dump(stats, open(f"{SH}/ng16k_stats.json", "w"), indent=1)
    offs = np.concatenate([[0], np.cumsum([len(x[3]) for x in ho_sel])])
    np.savez(HO_OUT + ".tmp.npz", ids=np.concatenate([np.asarray(x[3], dtype=np.uint32) for x in ho_sel]),
             mask=np.concatenate([np.asarray(x[4], dtype=bool) for x in ho_sel]), offsets=offs,
             sets=np.asarray([x[0] for x in ho_sel]), tasks=np.asarray([x[1] for x in ho_sel]),
             src_idx=np.asarray([x[2] for x in ho_sel]))
    os.replace(HO_OUT + ".tmp.npz", HO_OUT)
    print(f"[ng base] wrote {final} ({len(lens)} rows, {tot} tokens) and {HO_OUT} ({len(ho_sel)} rows); prompt hits 0",
          flush=True)
    if a.overlap:
        overlap_report()


def overlap_report():
    """Share of held-out documents whose text occurs in some training row (rows themselves are disjoint)."""
    out = {}
    for t in TASKS:
        tr = P.read_jsonl(f"{W}/base/{t}_pool_rows.jsonl")
        trd = {P.thash(d.get("text")) for e in tr for d in e["documents"]} if t != "oolong" else \
            {ln for e in tr for ln in e["documents"][0]["text"].split("\n")}
        if t == "reorder":
            ho = []  # heldout_v3 gen16k rows: chosen with no shared book / passage (POOL_V2.md)
            out[t] = "heldout_v3 gen16k (no shared book with training)"
            continue
        ho = P.read_jsonl(f"{W}/base/{t}_ho_cand.jsonl")[:HO_ROWS]
        if t == "oolong":
            sh = [np.mean([ln in trd for ln in e["documents"][0]["text"].split("\n")[2:]]) for e in ho]
        else:
            sh = [np.mean([P.thash(d.get("text")) in trd for d in e["documents"]]) for e in ho]
        out[t] = {"heldout_doc_share_in_train_mean": float(np.mean(sh)), "max": float(np.max(sh))}
    print(f"[ng overlap] {json.dumps(out)}", flush=True)
    json.dump(out, open(f"{SH}/ng16k_overlap.json", "w"), indent=1)


# ----------------------------------------------------------------------------------------------------------- variants
def _variant_task(job):
    t, name, lo, hi, whole, seed = job
    ctx = ctx_for(t)
    rows = P.read_jsonl(f"{W}/base/{t}_pool_rows.jsonl")
    rng = random.Random(f"{name}:{t}:{seed}")
    out, st = [], {"rows": len(rows), "whole": 0, "dropped": 0, "redraws": 0, "fallback_whole": 0, "units_in": 0,
                   "units_out": 0}
    min_k = 4 if t == "oolong" else 2
    for ex in rows:
        n = R.n_units(t, ex, ctx)
        st["units_in"] += n
        if rng.random() < whole:
            st["whole"] += 1
            out.append(ex)
            st["units_out"] += n
            continue
        r = rng.uniform(lo, hi)
        new = None
        for _ in range(20):
            k = max(min_k, int(round(r * n)))
            keep = sorted(rng.sample(range(n), k))
            try:
                new = R.subset(t, ex, keep, ctx)
                break
            except R.Degenerate:
                st["redraws"] += 1
        if new is None:
            st["fallback_whole"] += 1
            out.append(ex)
            st["units_out"] += n
            continue
        st["dropped"] += 1
        st["units_out"] += k
        out.append(new)
    d = f"{W}/var/{name}/tok_{t}"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    write_jsonl(f"{W}/var/{name}/{t}.jsonl", out)
    toks = convert(f"{W}/var/{name}/{t}.jsonl", t, d)
    assert len(toks) == len(out)
    st["answer_mismatch"] = tok_answer_check(t, out, toks)
    assert st["answer_mismatch"] == 0, st
    assert max(len(x[0]) for x in toks) <= CAP
    st["unit_keep"] = st["units_out"] / st["units_in"]
    return t, st, toks


def variants(a):
    name = a.name
    os.makedirs(f"{W}/var/{name}", exist_ok=True)
    base_meta = json.load(open(f"{SH}/ng16k/metadata.json"))
    _, base_rows = P.load_shard_rows(f"{SH}/ng16k")
    base_len = np.array([len(r[0]) for r in base_rows])
    jobs = [(t, name, a.lo, a.hi, a.whole, a.seed) for t in TASKS]
    res = {}
    with cf.ProcessPoolExecutor(max_workers=3) as ex:
        for t, st, toks in ex.map(_variant_task, jobs):
            res[t] = (st, toks)
    tmp, final = f"{SH}/ng16k_{name}_tmp", f"{SH}/ng16k_{name}"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    stats = {"name": name, "lo": a.lo, "hi": a.hi, "whole": a.whole, "seed": a.seed, "tasks": {}}
    lens, tot, totl = [], 0, 0
    info = open(f"{tmp}/row_info.jsonl", "w")
    for ti, t in enumerate(TASKS):
        st, toks = res[t]
        L = np.array([len(x[0]) for x in toks])
        bl = base_len[ti * ROWS:(ti + 1) * ROWS]
        st["token_keep"] = float(L.sum() / bl.sum())
        st["len_mean"] = float(L.mean())
        stats["tasks"][t] = st
        print(f"[ng var {name}] {t}: {json.dumps(st)}", flush=True)
        for k, (ids, m) in enumerate(toks):
            info.write(json.dumps({"part": ti, "row": k, "task": t, "len": int(len(ids)), "base_len": int(bl[k]),
                                   "answer_tokens": int(m.sum())}) + "\n")
        lens += L.tolist()
        bt = np.concatenate([x[0] for x in toks]).astype("uint32")
        bm = np.concatenate([x[1] for x in toks]).astype("bool")
        bt.tofile(f"{tmp}/token_ids_part_t{ti}_000.npy")
        bm.tofile(f"{tmp}/labels_mask_t{ti}_000.npy")
        tot += bt.size
        totl += int(bm.sum())
    info.close()
    meta = dict(base_meta)
    meta.update({"task": f"ng16k_{name}", "num_instances": len(lens), "num_tokens": int(tot), "num_loss_tokens": int(totl),
                 "max_example_len": int(max(lens)), "min_example_len": int(min(lens)),
                 "random_doc_drop": {k: stats[k] for k in ("lo", "hi", "whole", "seed")}})
    json.dump(meta, open(f"{tmp}/metadata.json", "w"), indent=1)
    stats["token_keep_all"] = float(tot / base_meta["num_tokens"])
    shutil.rmtree(final, ignore_errors=True)
    os.rename(tmp, final)
    json.dump(stats, open(f"{SH}/ng16k_{name}_stats.json", "w"), indent=1)
    print(f"[ng var {name}] wrote {final}: token keep {stats['token_keep_all']:.3f}", flush=True)


# -------------------------------------------------------------------------------------------------------------- order
def order(a):
    """Loader global order of trainer seed a.seed on each named pool: must be identical across pools."""
    P.MAX_LEN = CAP
    res = {}
    for pool in a.pools.split(","):
        sys.path.insert(0, f"{P.REPO}/src")
        from olmo_core.data import TokenizerConfig
        from olmo_core.data.composable import ComposableDataLoaderConfig, PadToLengthInstanceSourceConfig

        tc = TokenizerConfig(vocab_size=248320, eos_token_id=P.EOS, pad_token_id=P.EOS, bos_token_id=None)
        d = f"{SH}/{pool}"
        src_cfg = PadToLengthInstanceSourceConfig.from_npy(f"{d}/token_ids_part_*.npy", tokenizer=tc, sequence_length=CAP,
                                                           label_mask_paths=[f"{d}/labels_mask_*.npy"], expand_glob=True)
        wd = tempfile.mkdtemp(dir="/data/prasann/tmp")
        source = src_cfg.build(wd)
        dl = ComposableDataLoaderConfig(tokenizer=tc, work_dir=wd, global_batch_size=16 * CAP, seed=34521 + a.seed,
                                        num_workers=0).build(source)
        dl.reshuffle(epoch=1)
        res[pool] = np.asarray(dl.get_global_indices())
        shutil.rmtree(wd, ignore_errors=True)
    ref = list(res)[0]
    for p_, gi in res.items():
        print(f"[ng order] seed {a.seed} {p_}: {len(gi)} instances, identical to {ref}: {np.array_equal(gi, res[ref])}",
              flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["gen", "base", "variants", "order", "overlap"])
    ap.add_argument("--shuffle-seed", type=int, default=20261118)
    ap.add_argument("--name", default="")
    ap.add_argument("--lo", type=float, default=0.05)
    ap.add_argument("--hi", type=float, default=0.5)
    ap.add_argument("--whole", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pools", default="ng16k")
    ap.add_argument("--overlap", action="store_true")
    a = ap.parse_args()
    if a.stage == "overlap":
        return overlap_report()
    {"gen": gen, "base": base, "variants": variants, "order": order}[a.stage](a)


if __name__ == "__main__":
    main()
