"""Merge several fast8k task shards into ONE multi-task shard for the research trainer.

    python3 research/token_routing/ds64_fast8k/merge_tasks.py --out <shards>/mt4_g64M \
        --src <shards>/contradiction_g16M <shards>/nq_g16M <shards>/outlier_g16M <shards>/oolong_g16M

``train_ctc_suite.py`` reads one ``--data`` dir (``token_ids_part_*.npy`` + ``labels_mask_*.npy`` by
glob, ``metadata.json``, ``gold_fingerprints.json``). This copies every source's npy parts under a
per-source prefix (``token_ids_part_t<i>_<orig>.npy`` / ``labels_mask_t<i>_<orig>.npy``, so the two
globs still sort into aligned pairs), unions the gold sidecars (keyed by content fingerprint; a key
present in two sources with different gold is an error), and writes a merged ``metadata.json``.
Rows from different tasks interleave at train time because the data loader shuffles instances.
Stdlib only.
"""

import argparse
import glob
import json
import os
import shutil

MUST_AGREE = ("marker_set", "doc_start_id", "doc_end_id", "query_position")
SUMMED = ("num_instances", "num_tokens", "num_loss_tokens", "num_dropped",
          "rows_over_max_len_dropped")


def _subset(toks, masks, meta, n_rows, dst_tok, dst_mask, max_len=0, seed=0, length_mix=None, tokens_per_src=0):
    """First ``n_rows`` rows of a source (all parts concatenated; 0 = all), keeping only rows of at
    most ``max_len`` tokens when ``max_len`` > 0, written as one part each."""
    import numpy as np

    # converter parts are RAW `ndarray.tofile` dumps (no .npy header): dtype comes from metadata.json
    tok = np.concatenate([np.fromfile(t, dtype=meta.get("dtype", "uint32")) for t in toks])
    msk = np.concatenate([np.fromfile(m, dtype=meta.get("mask_dtype", "bool")) for m in masks])
    if tok.shape != msk.shape:
        raise SystemExit(f"token/mask length mismatch {tok.shape} vs {msk.shape}")
    want = int(meta["num_instances"])
    eos = None
    for cand in (248044, 248046, 151643, 151645):  # Qwen3.5 / Qwen3 endoftext, im_end
        if int((tok == cand).sum()) == want:
            eos = cand
            break
    if eos is None:
        raise SystemExit(f"no EOS candidate occurs exactly num_instances={want} times; refusing to split")
    ends = np.flatnonzero(tok == eos)
    starts = np.concatenate([[0], ends[:-1] + 1])
    lens = ends - starts + 1
    keep_rows = np.ones(len(ends), dtype=bool) if not max_len else lens <= max_len
    idx = np.flatnonzero(keep_rows)
    mix_report = None
    if length_mix:
        # length-stratified sample: bin (lo, hi] gets share * tokens_per_src tokens, drawn uniformly (seeded)
        # from the eligible rows in that bin; a bin with fewer tokens than its quota contributes all it has
        # (the deficit is NOT moved to other bins, so the realised shares are reported, not forced)
        rng = np.random.default_rng(seed)
        picked, mix_report, lo = [], [], 0
        for hi, share in length_mix:
            cand = idx[(lens[idx] > lo) & (lens[idx] <= hi)]
            cand = cand[rng.permutation(len(cand))]
            want_tok = share * tokens_per_src
            cum = np.cumsum(lens[cand])
            take = int(np.searchsorted(cum, want_tok, side="left") + 1) if len(cand) else 0
            take = min(take, len(cand))
            picked.append(cand[:take])
            mix_report.append({"lo": int(lo), "hi": int(hi), "share": share, "target_tokens": int(want_tok),
                               "rows": int(take), "tokens": int(cum[take - 1]) if take else 0,
                               "available_rows": int(len(cand)), "available_tokens": int(cum[-1]) if len(cand) else 0})
            lo = hi
        idx = np.sort(np.concatenate(picked)) if picked else idx[:0]
    elif n_rows:
        if n_rows > len(idx):
            raise SystemExit(f"--rows-per-src {n_rows} > {len(idx)} rows")
        if seed:  # uniform sample of the eligible rows (kept in source order), not the first n
            idx = np.sort(np.random.default_rng(seed).choice(idx, size=n_rows, replace=False))
        else:
            idx = idx[:n_rows]
    sel = np.zeros(len(tok), dtype=bool)
    if len(idx) == len(ends) and not n_rows and not length_mix:
        sel[:] = True
    else:
        # mark selected rows' token spans via a +1/-1 difference array
        d = np.zeros(len(tok) + 1, dtype=np.int32)
        np.add.at(d, starts[idx], 1)
        np.add.at(d, ends[idx] + 1, -1)
        sel = np.cumsum(d[:-1]) > 0
    np.ascontiguousarray(tok[sel]).tofile(dst_tok)
    np.ascontiguousarray(msk[sel]).tofile(dst_mask)
    kl = lens[idx]
    return {"num_instances": int(len(idx)), "num_tokens": int(sel.sum()),
            "num_loss_tokens": int(np.asarray(msk[sel]).sum()),
            "max_example_len": int(kl.max()), "min_example_len": int(kl.min()), "subset_eos_id": eos,
            "rows_over_max_len_dropped": int((~keep_rows).sum()), "subset_max_len": int(max_len),
            **({"length_mix": mix_report, "tokens_per_src": int(tokens_per_src), "sample_seed": int(seed)}
               if mix_report is not None else {})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--name", default="mt4", help="task name written to metadata.json")
    ap.add_argument("--rows-per-src", type=int, default=0,
                    help="keep only the first N rows of each source (a small screening subset); rows are "
                    "split at the EOS id, which must occur exactly num_instances times per source")
    ap.add_argument("--sample-seed", type=int, default=0,
                    help="with --rows-per-src: sample the rows uniformly (this seed) instead of taking the first N "
                    "(a CTC shard is ordered by rung, so the first N are all the shortest rung)")
    ap.add_argument("--max-row-len", type=int, default=0,
                    help="drop rows longer than this many tokens (e.g. 32768 for a 32k window; the "
                    "trainer refuses a --seq-len below a shard's max_example_len)")
    ap.add_argument("--length-mix", default="",
                    help="length-stratified sample per source: 'HI:SHARE,...' bins (prev HI, HI] in tokens, e.g. "
                    "'3072:0.30,6144:0.20,12288:0.15,24576:0.13,49152:0.12,65536:0.10' (the ds64 short-heavy "
                    "shares); each bin gets SHARE * --tokens-per-src tokens, sampled with --sample-seed")
    ap.add_argument("--tokens-per-src", type=int, default=0, help="with --length-mix: token target per source")
    a = ap.parse_args()
    mix = [(int(h), float(f)) for h, f in (kv.split(":") for kv in a.length_mix.split(","))] if a.length_mix else None
    if mix and (not a.tokens_per_src or a.rows_per_src):
        raise SystemExit("--length-mix needs --tokens-per-src and excludes --rows-per-src")
    if os.path.exists(os.path.join(a.out, "metadata.json")):
        raise SystemExit(f"{a.out} already has metadata.json; refusing to overwrite")
    os.makedirs(a.out, exist_ok=True)
    metas, gold = [], {}
    for i, src in enumerate(a.src):
        meta = json.load(open(os.path.join(src, "metadata.json")))
        metas.append(meta)
        toks = sorted(glob.glob(os.path.join(src, "token_ids_part_*.npy")))
        masks = sorted(glob.glob(os.path.join(src, "labels_mask_*.npy")))
        if len(toks) != len(masks) or not toks:
            raise SystemExit(f"{src}: {len(toks)} token parts vs {len(masks)} mask parts")
        for t, m in zip(toks, masks):
            if t.rsplit("_", 1)[-1] != m.rsplit("_", 1)[-1]:
                raise SystemExit(f"unaligned parts {t} / {m}")
        dt = os.path.join(a.out, f"token_ids_part_t{i}_000.npy")
        dm = os.path.join(a.out, f"labels_mask_t{i}_000.npy")
        if a.rows_per_src or a.max_row_len or mix:
            meta = dict(meta)
            meta.update(_subset(toks, masks, meta, a.rows_per_src, dt, dm, a.max_row_len, a.sample_seed, mix,
                                a.tokens_per_src))
        else:
            for k, (t, m) in enumerate(zip(toks, masks)):
                shutil.copy2(t, dt.replace("_000.npy", f"_{k:03d}.npy"))
                shutil.copy2(m, dm.replace("_000.npy", f"_{k:03d}.npy"))
        metas[-1] = meta
        side = json.load(open(os.path.join(src, "gold_fingerprints.json")))
        clash = [k for k in side if k in gold and gold[k] != side[k]]
        if clash:
            raise SystemExit(f"{src}: {len(clash)} fingerprints with conflicting gold, e.g. {clash[0]}")
        gold.update(side)
        print(f"[merge] t{i} {src}: {meta.get('num_instances')} rows, {len(toks)} parts, "
              f"{len(side)} gold rows, max_example_len {meta.get('max_example_len')}")
    for k in MUST_AGREE:
        vals = {json.dumps(m.get(k)) for m in metas}
        if len(vals) > 1:
            raise SystemExit(f"sources disagree on {k}: {vals}")
    merged = dict(metas[0])
    for k in SUMMED:
        if all(k in m for m in metas):
            merged[k] = sum(int(m[k]) for m in metas)
    merged["max_example_len"] = max(int(m.get("max_example_len", 0)) for m in metas)
    merged["min_example_len"] = min(int(m.get("min_example_len", 0)) for m in metas)
    merged["task"] = a.name
    merged["merged_from"] = [{"src": s, "task": m.get("task"), "num_instances": m.get("num_instances"),
                              **({"num_tokens": m.get("num_tokens"), "length_mix": m["length_mix"]}
                                 if "length_mix" in m else {})}
                             for s, m in zip(a.src, metas)]
    merged.pop("length_mix", None)
    json.dump(gold, open(os.path.join(a.out, "gold_fingerprints.json"), "w"))
    json.dump(merged, open(os.path.join(a.out, "metadata.json"), "w"), indent=1)
    print(f"[merge] wrote {a.out}: {merged.get('num_instances')} rows, {len(gold)} gold rows, "
          f"max_example_len {merged['max_example_len']}")


if __name__ == "__main__":
    main()
