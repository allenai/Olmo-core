"""Contradiction-only long-context pools ``lc128k`` / ``lc256k`` / ``lc1m`` + held-out ``heldout_lc.npz`` (README here).

Same generator, split, seed (42), seed pool (PubMed pairs + fillers), converter flags, layout and sidecars as the
keepfrac_len pools ``kl<L>`` (``../keepfrac_len/build_kl_pools.py``), contradiction only (3 gold pairs = 6 gold claims per
row). Documents per row are set from the measured kl64k density (1,304 docs -> mean 58,586 tokens, 44.9 tokens/doc) so the
longest row stays under the cap: 128k 2,720 docs (cap 131,072), 256k 5,550 (cap 262,144: also the Gated DeltaNet
single-call limit, records/grad-sparsity-vs-length.md), 1m 22,500 (cap 1,048,576).

Held-out (set ``lc<L>``): ctc EVAL split, eval seed 99 (as heldout_v4 / heldout_kl), the first rows whose gold is absent
from every training pool. Training rows are dropped if a gold claim occurs in any suite eval rung, any existing held-out
set (``../keepfrac_len`` train_excl) or any lc held-out row.

    $PY build_lc_pools.py gen ; ... heldout ; ... tokenize ; ... build      (CPU; build_lc.sbatch)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ELP = os.path.dirname(HERE)
sys.path.insert(0, ELP)
sys.path.insert(0, os.path.join(ELP, "keepfrac_len"))
import build_pool_v2 as P  # noqa: E402
import build_kl_pools as K  # noqa: E402

T = "contradiction"
# short bins (2k-16k) use the keepfrac_len kl<L> cells' document counts and caps, so their first rows are kl<L>'s
LENS = {"128k": 131072, "256k": 262144, "1m": 1048576, "2k": 4096, "4k": 6144, "8k": 12288, "16k": 16384}
NDOCS = {"128k": 2720, "256k": 5550, "1m": 22500, "2k": 56, "4k": 92, "8k": 187, "16k": 324}
GEN = {"128k": 1100, "256k": 1800, "1m": 420, "2k": 4300, "4k": 4300, "8k": 4300, "16k": 4300}  # train rows generated
ROWS = {"128k": 1000, "256k": 1600, "1m": 360, "2k": 4000, "4k": 4000, "8k": 4000, "16k": 4000}  # rows kept
HO_GEN = {"128k": 96, "256k": 200, "1m": 80, "2k": 16, "4k": 16, "8k": 16, "16k": 200}  # eval-split candidates
HO_ROWS = {"128k": 64, "256k": 160, "1m": 64, "2k": 0, "4k": 0, "8k": 0, "16k": 160}  # held-out rows kept
KW = f"{P.WORK}/lc"
SH = "/data/prasann/early_loss_proxy/shards"
HO_OUT = "/data/prasann/early_loss_proxy/heldout_lc.npz"
SUFFIX = [""]


def env(r):
    e = P.env()
    e["CTC_DOCS_OVERRIDE"] = json.dumps({f"{T}:{r}": NDOCS[r]})
    return e


def convert(jsonl, out_dir, cap):
    cmd = [sys.executable, f"{P.REPO}/recipes/data/convert_unified_to_document_landmark.py", "--input-jsonl", jsonl,
           "--task", P.SPEC[T], "--out-dir", out_dir, "--emit", "dense", "--marker-set", "qwen3_5",
           "--tokenizer", P.TOKENIZER, "--seq-len", str(int(cap * 1.1)), "--query-position", "after", "--cot-mode", "none",
           "--chunk-by", "document", "--num-proc", "8"]
    p = subprocess.run(cmd, cwd=P.REPO, env=P.env(), capture_output=True, text=True)
    open(f"{out_dir}.log", "w").write(" ".join(cmd) + "\n" + p.stdout[-5000:] + p.stderr[-5000:])
    if p.returncode != 0:
        raise RuntimeError(f"convert {jsonl}: {(p.stdout + p.stderr)[-800:]}")


def _gen(job):
    kind, r = job
    t0 = time.time()
    if kind == "train":
        out, target = f"{KW}/gen/{r}", f"{KW}/gen/{r}/{T}/train.jsonl"
        args = ["--split", "train", "--train", str(GEN[r]), "--seed", str(P.SEED)]
    else:
        out, target = f"{KW}/hogen/{r}", f"{KW}/hogen/{r}/{T}/eval_{r}.jsonl"
        args = ["--split", "eval", "--eval-size", str(HO_GEN[r]), "--allow-small-eval", "--eval-seed", "99"]
    if os.path.exists(target) and os.path.getsize(target):
        return kind, r, "cached", 0.0
    cmd = [sys.executable, f"{P.REPO}/recipes/data/ctc_sft/ctc_calibrated.py", "build", "--task", T, "--rungs", r] + args \
        + ["--out", out, "--pool", P.POOL[T]]
    p = subprocess.run(cmd, env=env(r), capture_output=True, text=True)
    open(f"{out}.log", "w").write(" ".join(cmd) + "\n" + p.stdout + p.stderr)
    ok = p.returncode == 0 and os.path.exists(target)
    return kind, r, "built" if ok else "FAILED " + (p.stdout + p.stderr)[-600:], time.time() - t0


def lens_of(a):
    return [r for r in (a.lens.split(",") if a.lens else ["128k", "256k", "1m"])]


def gen(a):
    os.makedirs(f"{KW}/gen", exist_ok=True)
    os.makedirs(f"{KW}/hogen", exist_ok=True)
    jobs = [(k, r) for r in lens_of(a) for k in ("train", "eval")]
    with cf.ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for res in ex.map(_gen, jobs):
            print(f"[lc gen] {res[0]}@{res[1]} n_docs={NDOCS[res[1]]} {res[2]} ({res[3]:.0f}s)", flush=True)


def _tok(job):
    src, out, cap = job
    if os.path.exists(f"{out}/metadata.json"):
        return out, "cached"
    os.makedirs(out, exist_ok=True)
    convert(src, out, cap)
    return out, "built"


def tokenize(a):
    """Held-out candidates and training rows (training rows whose gold hits an exclusion set are dropped first)."""
    excl, _ = K.train_excl()  # suite eval rungs + every existing held-out set's gold, per task
    jobs, drops = [], {}
    ho_gold = set()
    for r in lens_of(a):
        for ex in P.read_jsonl(f"{KW}/hogen/{r}/{T}/eval_{r}.jsonl"):
            ho_gold |= K.gold_h(T, ex)
    json.dump(sorted(ho_gold), open(f"{KW}/ho_gold.json", "w"))
    for r in lens_of(a):
        jobs.append((f"{KW}/hogen/{r}/{T}/eval_{r}.jsonl", f"{KW}/hotok/{r}", LENS[r]))
        exs = P.read_jsonl(f"{KW}/gen/{r}/{T}/train.jsonl")
        d = {"generated": len(exs), "gold_in_eval_or_heldout": 0, "gold_in_lc_heldout": 0}
        idx = []
        for i, ex in enumerate(exs):
            g = K.gold_h(T, ex)
            if g & excl[T]:
                d["gold_in_eval_or_heldout"] += 1
            elif g & ho_gold:
                d["gold_in_lc_heldout"] += 1
            else:
                idx.append(i)
        d["eligible"] = len(idx)
        drops[r] = d
        os.makedirs(f"{KW}/elig", exist_ok=True)
        with open(f"{KW}/elig/{r}.jsonl", "w") as g:
            for i in idx:
                g.write(json.dumps(exs[i]) + "\n")
        json.dump(idx, open(f"{KW}/elig_{r}_src_idx.json", "w"))
        jobs.append((f"{KW}/elig/{r}.jsonl", f"{KW}/tok/{r}", LENS[r]))
        print(f"[lc tok] {r}: {d}", flush=True)
    json.dump(drops, open(f"{KW}/drops.json", "w"), indent=1)
    with cf.ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for res in ex.map(_tok, jobs):
            print(f"[lc tok] {res}", flush=True)


def build(a):
    sys.path.insert(0, f"{P.REPO}/src")
    from olmo_core.nn.attention.gold_grad_mask import content_fingerprint
    import build_heldout_v3 as H

    drops = json.load(open(f"{KW}/drops.json"))
    ho_rows, ho_stats, ho_keys = [], {}, set()
    for r in ([] if SUFFIX[0] else lens_of(a)):  # held-out first (main build only; a suffixed build reuses it)
        exs = P.read_jsonl(f"{KW}/hogen/{r}/{T}/eval_{r}.jsonl")
        _, rows = P.load_shard_rows(f"{KW}/hotok/{r}")
        assert len(rows) == len(exs)
        sel = [(i, ids, m) for i, (ids, m) in enumerate(rows) if len(ids) <= LENS[r]][:HO_ROWS[r]]
        L = np.array([len(x[1]) for x in sel])
        ho_stats[f"lc{r}"] = {"rows": len(sel), "candidates": len(exs), "len_mean": float(L.mean()), "len_min": int(L.min()),
                              "len_max": int(L.max()), "n_docs": NDOCS[r],
                              "answer_tokens_mean": float(np.mean([int(x[2].sum()) for x in sel]))}
        for i, ids, m in sel:
            ho_rows.append((f"lc{r}", T, i, ids, m))
            ho_keys.add(H.keys(ids, m))
        print(f"[lc heldout] lc{r}: {json.dumps(ho_stats[f'lc{r}'])}", flush=True)
    if SUFFIX[0]:
        hz = np.load(HO_OUT)
        for j in range(len(hz["sets"])):
            ho_keys.add(H.keys(hz["ids"][hz["offsets"][j]:hz["offsets"][j + 1]], hz["mask"][hz["offsets"][j]:hz["offsets"][j + 1]]))
    offs = np.concatenate([[0], np.cumsum([len(x[3]) for x in ho_rows])]) if ho_rows else None
    if ho_rows:
        np.savez(HO_OUT + ".tmp.npz", ids=np.concatenate([np.asarray(x[3], dtype=np.uint32) for x in ho_rows]),
                 mask=np.concatenate([np.asarray(x[4], dtype=bool) for x in ho_rows]), offsets=offs,
                 sets=np.asarray([x[0] for x in ho_rows]), tasks=np.asarray([x[1] for x in ho_rows]),
                 src_idx=np.asarray([x[2] for x in ho_rows]))
        os.replace(HO_OUT + ".tmp.npz", HO_OUT)
        json.dump(ho_stats, open(HO_OUT.replace(".npz", "_stats.json"), "w"), indent=1)

    for r in lens_of(a):
        cap, name = LENS[r], f"lc{r}{SUFFIX[0]}"
        tmp, final = f"{SH}/{name}_tmp", f"{SH}/{name}"
        if os.path.exists(f"{final}/metadata.json"):
            print(f"[lc build] {final} exists, skip", flush=True)
            continue
        os.system(f"rm -rf {tmp}")
        os.makedirs(tmp)
        src = f"{KW}/gen/{r}/{T}/train.jsonl"
        idx = json.load(open(f"{KW}/elig_{r}_src_idx.json"))
        exs_all = P.read_jsonl(src)
        meta, rows = P.load_shard_rows(f"{KW}/tok/{r}")
        assert len(rows) == len(idx) and meta["num_dropped"] == 0, (r, len(rows), len(idx))
        kept = [(ids, m, exs_all[i], i) for (ids, m), i in zip(rows, idx) if len(ids) <= cap]
        over = len(rows) - len(kept)
        sel = kept[:ROWS[r]]
        if len(sel) < ROWS[r]:
            print(f"[lc build] WARNING {r}: only {len(sel)} rows under the cap (< {ROWS[r]})", flush=True)
        perm = np.random.default_rng(a.shuffle_seed).permutation(len(sel))
        sel = [sel[j] for j in perm]
        gold_side, pair_side, fps, tr_keys = {}, {}, [], set()
        with open(f"{tmp}/row_info.jsonl", "w") as info_f:
            for k, (ids, m, ex, i) in enumerate(sel):
                g, pairs = P.gold_info(T, ex)
                spans = P.doc_spans(ids)
                assert len(spans) == len(ex["documents"]), (r, k, len(spans), len(ex["documents"]))
                fp = content_fingerprint(ids.tolist())
                assert fp not in gold_side
                gold_side[fp] = g
                pair_side[fp] = pairs
                fps.append(fp)
                tr_keys.add(H.keys(ids, m))
                info_f.write(json.dumps({"part": 0, "row": k, "task": T, "rung": r, "src": src, "src_idx": i,
                                         "n_docs": len(spans), "len": int(len(ids)), "answer_tokens": int(m.sum()),
                                         "fingerprint": fp, "gold": g, "gold_pairs": pairs,
                                         "gold_tokens": int(sum(spans[j][1] - spans[j][0] for j in g))}) + "\n")
        bt = np.concatenate([x[0] for x in sel]).astype("uint32")
        bm = np.concatenate([x[1] for x in sel]).astype("bool")
        bt.tofile(f"{tmp}/token_ids_part_t0_000.npy")
        bm.tofile(f"{tmp}/labels_mask_t0_000.npy")
        L = np.array([len(x[0]) for x in sel])
        meta = json.load(open(f"{SH}/kl64k/metadata.json"))
        meta.update({"task": name, "num_instances": len(sel), "num_tokens": int(bt.size), "num_loss_tokens": int(bm.sum()),
                     "max_example_len": int(L.max()), "min_example_len": int(L.min()),
                     "merged_from": [{"src": f"{KW}/tok/{r}", "task": T, "converter_task": P.SPEC[T],
                                      "num_instances": len(sel)}],
                     "shuffle_seed": a.shuffle_seed, "subset_max_len": cap, "tasks_order": [T]})
        json.dump(meta, open(f"{tmp}/metadata.json", "w"), indent=1)
        json.dump(gold_side, open(f"{tmp}/gold_fingerprints.json", "w"))
        json.dump(pair_side, open(f"{tmp}/gold_pairs_fingerprints.json", "w"))
        json.dump(fps, open(f"{tmp}/gdd_only_contradiction.json", "w"))
        st = {"pool": name, "cap": cap, "task": T, "rows": len(sel), "n_docs": NDOCS[r], "len_mean": float(L.mean()),
              "len_p50": float(np.median(L)), "len_min": int(L.min()), "len_max": int(L.max()), "tokens": int(L.sum()),
              "answer_tokens_mean": float(bm.sum() / len(sel)), "gold_docs_per_row": 6,
              "drops": dict(drops[r], over_cap=over), "heldout_prompt_hits": len(ho_keys & tr_keys)}
        json.dump(st, open(f"{SH}/{name}_stats.json", "w"), indent=1)
        os.rename(tmp, final)
        print(f"[lc build] wrote {final}: {json.dumps(st)}", flush=True)


def addho(a):
    """Append (or replace) held-out sets lc<L> for --lens from this work dir's eval-split rows into heldout_lc.npz."""
    z = np.load(HO_OUT)
    keep = [j for j in range(len(z["sets"])) if str(z["sets"][j]) not in {f"lc{r}" for r in lens_of(a)}]
    o = z["offsets"]
    rows = [(str(z["sets"][j]), str(z["tasks"][j]), int(z["src_idx"][j]), z["ids"][o[j]:o[j + 1]], z["mask"][o[j]:o[j + 1]])
            for j in keep]
    st = json.load(open(HO_OUT.replace(".npz", "_stats.json")))
    for r in lens_of(a):
        exs = P.read_jsonl(f"{KW}/hogen/{r}/{T}/eval_{r}.jsonl")
        _, toks = P.load_shard_rows(f"{KW}/hotok/{r}")
        assert len(toks) == len(exs)
        sel = [(i, ids, m) for i, (ids, m) in enumerate(toks) if len(ids) <= LENS[r]][:HO_ROWS[r]]
        assert len(sel) == HO_ROWS[r], (r, len(sel))
        rows += [(f"lc{r}", T, i, ids, m) for i, ids, m in sel]
        L = np.array([len(x[1]) for x in sel])
        st[f"lc{r}"] = {"rows": len(sel), "candidates": len(exs), "len_mean": float(L.mean()), "len_min": int(L.min()),
                        "len_max": int(L.max()), "n_docs": NDOCS[r], "source": f"{KW}/hogen/{r}",
                        "answer_tokens_mean": float(np.mean([int(x[2].sum()) for x in sel]))}
        print(f"[lc addho] lc{r}: {json.dumps(st[f'lc{r}'])}", flush=True)
    offs = np.concatenate([[0], np.cumsum([len(x[3]) for x in rows])])
    np.savez(HO_OUT + ".tmp.npz", ids=np.concatenate([np.asarray(x[3], dtype=np.uint32) for x in rows]),
             mask=np.concatenate([np.asarray(x[4], dtype=bool) for x in rows]), offsets=offs,
             sets=np.asarray([x[0] for x in rows]), tasks=np.asarray([x[1] for x in rows]),
             src_idx=np.asarray([x[2] for x in rows]))
    os.replace(HO_OUT + ".tmp.npz", HO_OUT)
    json.dump(st, open(HO_OUT.replace(".npz", "_stats.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["gen", "tokenize", "build", "addho"])
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--shuffle-seed", type=int, default=20261118)
    ap.add_argument("--lens", default="")
    ap.add_argument("--sizes", default="", help="override train rows per length, e.g. 256k=18000:18000,1m=5000:5000 "
                    "(generated:kept); with --suffix the pools are lc<L><suffix> and the work dir v2build/lc<suffix>")
    ap.add_argument("--suffix", default="", help="pool-name / work-dir suffix (e.g. L); held-out stays heldout_lc.npz")
    a = ap.parse_args()
    global KW
    for kv in [x for x in a.sizes.split(",") if x]:
        r, _, v = kv.partition("=")
        g, _, k = v.partition(":")
        GEN[r], ROWS[r] = int(g), int(k or g)
    if a.suffix:
        KW = f"{P.WORK}/lc{a.suffix}"
        SUFFIX[0] = a.suffix
    {"gen": gen, "tokenize": tokenize, "build": build, "addho": addho}[a.stage](a)


if __name__ == "__main__":
    main()
