"""Build the doc-drop training shard from a fast8k mt4 shard (records/docdrop-nongold-baseline.md). Runs on a node.

For every contradiction / nq / outlier row: keep every gold document (gold = the IDs the answer cites, matched to
the IDs the documents print) and a random share ``q`` of the non-gold documents, remove the rest from the row,
leave every surviving document's text (and so its ID) untouched. oolong rows stay whole (their label aggregates over
every item, so it cannot be recomputed from the tokens). ``q`` is chosen by bisection so the run's training FLOPs
(the trainer's FLOP meter formula: per example ``L * (fpt(L) + lora_adjust(L))``, whole-model LoRA r64) hit
``--target-pf``.

Writes into ``--out``: the four ``token_ids_part_t<i>_000.npy`` / ``labels_mask_t<i>_000.npy`` parts (raw
``tofile`` dumps, like the source), ``metadata.json``, ``docdrop_stats.json`` (realised keep shares, gold share of
visible documents, FLOP model), ``docdrop_examples.txt`` (decoded rows before / after), and
``keep_sidecar.json`` ({content fingerprint of the FULL source row: kept 0-based document indices}, every
document for oolong) for the original-position arm, which compacts the full rows in the model instead.

    $PY research/token_routing/docdrop/build_docdrop_shard.py --src /data/prasann/oracle/shards/mt4f1952 \\
        --out /data/prasann/docdrop/shards/mt4f1952_dd --target-pf 344
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import docdrop as DD  # noqa: E402

TOKENIZER = "/data/prasann/hf_models/Qwen3.5-4B-Base/tokenizer.json"


def load_parts(src, meta):
    parts = []
    for t in range(4):
        tok = np.fromfile(f"{src}/token_ids_part_t{t}_000.npy", dtype=meta.get("dtype", "uint32"))
        msk = np.fromfile(f"{src}/labels_mask_t{t}_000.npy", dtype=meta.get("mask_dtype", "bool"))
        assert tok.shape == msk.shape, (t, tok.shape, msk.shape)
        ends = np.flatnonzero(tok == meta["eos_token_id"])
        starts = np.concatenate([[0], ends[:-1] + 1])
        assert ends[-1] == len(tok) - 1, f"part {t} does not end in EOS"
        parts.append((tok, msk, starts, ends + 1))
    return parts


class FlopModel:
    """Per-example training FLOPs with the trainer's formulas (FlopMeterCallback, per_example=True):
    dense full fine-tune ``L * fpt(L)``; whole-model LoRA ``L * (fpt(L) + lora_flop_adjust_per_token(L))``."""

    def __init__(self, lora_rank=64, lora_scale=32.0):
        from olmo_core.nn.early_lora import install_early_lora, lora_flop_adjust_per_token
        from olmo_core.nn.transformer import TransformerConfig

        self.m = TransformerConfig.qwen3_5_4B(vocab_size=248320).build(init_device="meta")
        self.dense_m = TransformerConfig.qwen3_5_4B(vocab_size=248320).build(init_device="meta")
        install_early_lora(self.m, len(self.m.blocks), lora_rank, lora_scale)
        self._adj = lora_flop_adjust_per_token
        self.cache = {}

    def per_token(self, L):
        v = self.cache.get(L)
        if v is None:
            f = float(self.dense_m.num_flops_per_token(int(L)))
            v = self.cache[L] = (f, f + float(self._adj(self.m, int(L))))
        return v

    def total(self, lens, lora=True):
        return sum(L * self.per_token(int(L))[1 if lora else 0] for L in lens)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-pf", type=float, default=344.0, help="training PF of the whole run (recipe 343-345)")
    ap.add_argument("--keep-frac", type=float, default=None, help="fix q instead of bisecting on --target-pf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-sidecar", action="store_true")
    ap.add_argument("--outlier-groups", default="",
                    help="outlier_groups.py output: drop WHOLE topic groups on outlier rows (group-aware arm)")
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, "metadata.json")):
        raise SystemExit(f"{a.out} already has metadata.json; refusing to overwrite")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(TOKENIZER)
    meta = json.load(open(os.path.join(a.src, "metadata.json")))
    tasks = [m["task"] for m in meta["merged_from"]]
    assert tasks == ["contradiction", "retrieval", "outlier", "oolong"], tasks
    DS, DE, EOS = meta["doc_start_id"], meta["doc_end_id"], meta["eos_token_id"]
    t0 = time.time()
    parts = load_parts(a.src, meta)
    ogroups = json.load(open(a.outlier_groups))["rows"] if a.outlier_groups else None
    if ogroups is not None and len(ogroups) < len(parts[2][2]):
        raise SystemExit(f"--outlier-groups has {len(ogroups)} rows < {len(parts[2][2])} outlier rows")
    print(f"[docdrop] loaded {sum(len(p[2]) for p in parts)} rows in {time.time() - t0:.0f}s", flush=True)

    # ---- parse every row: spans, printed IDs, gold ----------------------------------------------------------
    rows = []  # per part: list of dicts
    issues = {"id_not_index_plus_1": 0, "gold_id_missing": 0, "no_gold": 0}
    for p, (ids, msk, st, en) in enumerate(parts):
        task = DD.PART_TASK[p]
        out = []
        for r, (s, e) in enumerate(zip(st, en)):
            row, m = ids[s:e], msk[s:e]
            pe, spans, ss = DD.split_row(row, DS, DE)
            info = {"s": int(s), "e": int(e), "pe": pe, "spans": spans, "ss": ss,
                    "span_lens": np.array([b - a_ for a_, b in spans], dtype=np.int64)}
            if task != "oolong":
                if m[:pe].any() or any(m[a_:b].any() for a_, b in spans):
                    raise SystemExit(f"{task} row {r}: loss tokens outside the suffix")
                # each document's printed ID (decode the first 16 tokens after the opening marker)
                pid = [DD.doc_id_from_text(tok.decode(row[a_ + 1:a_ + 17].tolist(), skip_special_tokens=False), task)
                       for a_, _ in spans]
                if any(x is None for x in pid):
                    raise SystemExit(f"{task} row {r}: a document without a printed ID")
                if pid != list(range(1, len(spans) + 1)):
                    issues["id_not_index_plus_1"] += 1
                id2idx = {x: i for i, x in enumerate(pid)}
                ans = tok.decode(row[m].tolist(), skip_special_tokens=False)
                gold_ids = DD.gold_ids_from_answer(ans, task)
                miss = [g for g in gold_ids if g not in id2idx]
                if miss:
                    issues["gold_id_missing"] += 1
                    raise SystemExit(f"{task} row {r}: answer cites IDs {miss} that no document prints ({ans!r})")
                if not gold_ids:
                    issues["no_gold"] += 1
                info["gold"] = sorted(id2idx[g] for g in gold_ids)
                info["answer"] = ans
            out.append(info)
        rows.append(out)
        print(f"[docdrop] parsed {task}: {len(out)} rows ({time.time() - t0:.0f}s)", flush=True)

    # ---- FLOP model and q ------------------------------------------------------------------------------------
    fm = FlopModel()
    full_lens = [r["e"] - r["s"] for part in rows for r in part]
    dense_pf = fm.total(full_lens, lora=False) / 1e15
    lora_full_pf = fm.total(full_lens, lora=True) / 1e15
    print(f"[docdrop] full rows: dense full-FT {dense_pf:.1f} PF (Beaker dense f1952 run: 1531), "
          f"dense LoRA {lora_full_pf:.1f} PF", flush=True)

    def plan(q):
        lens, keeps = [], []
        for p, part in enumerate(rows):
            task = DD.PART_TASK[p]
            kp = []
            for r, info in enumerate(part):
                n = len(info["spans"])
                if task == "oolong":
                    keep = np.arange(n)
                elif task == "outlier" and ogroups is not None:
                    if len(ogroups[r]) != n:
                        raise SystemExit(f"outlier row {r}: {len(ogroups[r])} group ids vs {n} documents")
                    keep = DD.choose_keep_groups(ogroups[r], info["gold"], q, DD.row_seed(a.seed, p, r))
                else:
                    keep = DD.choose_keep(n, info["gold"], q, DD.row_seed(a.seed, p, r), task)
                L = (info["e"] - info["s"]) - int(info["span_lens"].sum()) + int(info["span_lens"][keep].sum())
                lens.append(L)
                kp.append(keep)
            keeps.append(kp)
        return lens, keeps

    if a.keep_frac is not None:
        q = a.keep_frac
    else:
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2
            pf = fm.total(plan(mid)[0]) / 1e15
            lo, hi = (mid, hi) if pf < a.target_pf else (lo, mid)
        q = lo
    lens, keeps = plan(q)
    pf = fm.total(lens) / 1e15
    print(f"[docdrop] q = {q:.4f} -> {pf:.1f} PF (target {a.target_pf}); {pf / dense_pf:.3f}x dense full-FT",
          flush=True)

    # ---- write -----------------------------------------------------------------------------------------------
    os.makedirs(a.out, exist_ok=True)
    stats = {"keep_frac_nongold": q, "seed": a.seed, "outlier_groups": a.outlier_groups or None, "target_pf": a.target_pf, "model_pf": pf,
             "dense_fullft_pf_model": dense_pf, "dense_lora_full_rows_pf_model": lora_full_pf,
             "step_cost_vs_dense_fullft": pf / dense_pf, "issues": issues, "tasks": {}}
    examples = []
    sidecar = {}
    if not a.no_sidecar:
        from olmo_core.nn.attention.gold_grad_mask import content_fingerprint_from_row
    new_lens_all = []
    for p, part in enumerate(rows):
        task = DD.PART_TASK[p]
        ids, msk = parts[p][0], parts[p][1]
        out_t, out_m = [], []
        tok_in = tok_out = doc_in = doc_out = 0
        gs_before, gs_after, keep_share_rows = [], [], []
        n_nonctx = []
        maj_groups, min_kept_major = [], []
        for r, info in enumerate(part):
            row, m = ids[info["s"]:info["e"]], msk[info["s"]:info["e"]]
            keep = keeps[p][r]
            nr, nm = DD.drop_docs(row, m, info["pe"], info["spans"], info["ss"], keep)
            assert int(nm.sum()) == int(m.sum()) and np.array_equal(nr[nm], row[m]), "labels changed"
            assert nr[-1] == EOS and int((nr == EOS).sum()) == 1
            out_t.append(nr)
            out_m.append(nm)
            new_lens_all.append(len(nr))
            n = len(info["spans"])
            tok_in += len(row)
            tok_out += len(nr)
            doc_in += n
            doc_out += len(keep)
            n_nonctx.append(len(row) - int(info["span_lens"].sum()))
            if task != "oolong":
                g = len(info["gold"])
                gs_before.append(DD.gold_share(g, n))
                gs_after.append(DD.gold_share(g, len(keep)))
                keep_share_rows.append((len(keep) - g) / max(1, n - g))
                if task == "outlier":
                    grp = ogroups[r] if ogroups is not None else None
                    if grp is not None:
                        kg = [grp[k] for k in keep]
                        gg = {grp[i] for i in info["gold"]}
                        sizes = [kg.count(x) for x in set(kg) if x not in gg]
                        maj_groups.append(len(sizes))
                        min_kept_major.append(min(sizes))
                        assert min(sizes) > len(info["gold"]), "outlier invariant broken"
            if not a.no_sidecar:
                sidecar[content_fingerprint_from_row(row.tolist(), EOS)] = [int(k) for k in keep]
            if r < 2:
                def dec(x):
                    return tok.decode(x.tolist(), skip_special_tokens=False)
                kept_ids = ([int(DD.doc_id_from_text(dec(row[info["spans"][k][0] + 1:info["spans"][k][0] + 17]), task))
                             for k in keep] if task != "oolong" else "all")
                examples.append(f"===== {task} row {r}: {n} docs -> {len(keep)} kept, {len(row)} -> {len(nr)} tokens\n"
                                f"gold idx {info.get('gold')} ; kept printed IDs {kept_ids}\n"
                                f"ANSWER: {info.get('answer', dec(row[m]))!r}\n--- AFTER (head 1200 chars):\n"
                                f"{dec(nr)[:1200]}\n--- AFTER (tail 1500 chars):\n{dec(nr)[-1500:]}\n")
        bt = np.concatenate(out_t).astype(meta.get("dtype", "uint32"))
        bm = np.concatenate(out_m).astype(meta.get("mask_dtype", "bool"))
        bt.tofile(os.path.join(a.out, f"token_ids_part_t{p}_000.npy"))
        bm.tofile(os.path.join(a.out, f"labels_mask_t{p}_000.npy"))
        stats["tasks"][task] = {
            "rows": len(part), "tokens_in": tok_in, "tokens_out": tok_out, "token_keep_share": tok_out / tok_in,
            "docs_in": doc_in, "docs_out": doc_out, "doc_keep_share": doc_out / doc_in,
            "nongold_keep_share_per_row": DD.summarize(keep_share_rows),
            "gold_share_of_visible_before": DD.summarize(gs_before),
            "gold_share_of_visible_after": DD.summarize(gs_after),
            "non_document_tokens_per_row": DD.summarize(n_nonctx),
            "row_len_after": DD.summarize([len(x) for x in out_t]),
        }
        if maj_groups:
            stats["tasks"][task]["kept_majority_groups"] = DD.summarize(maj_groups)
            stats["tasks"][task]["smallest_kept_majority_group"] = DD.summarize(min_kept_major)
        print(f"[docdrop] {task}: tokens {tok_in} -> {tok_out} ({tok_out / tok_in:.3f}), docs {doc_in} -> {doc_out} "
              f"({doc_out / doc_in:.3f})", flush=True)
    tin = sum(v["tokens_in"] for v in stats["tasks"].values())
    tout = sum(v["tokens_out"] for v in stats["tasks"].values())
    stats["token_keep_share_all"] = tout / tin
    stats["doc_keep_share_all"] = (sum(v["docs_out"] for v in stats["tasks"].values())
                                   / sum(v["docs_in"] for v in stats["tasks"].values()))
    nm = dict(meta)
    nm.update({"num_tokens": int(tout), "max_example_len": int(max(new_lens_all)),
               "min_example_len": int(min(new_lens_all)), "task": "mt4",
               "docdrop": {"src": a.src, "keep_frac_nongold": q, "seed": a.seed, "oolong": "whole",
                           "outlier": "whole topic groups" if ogroups is not None else "random documents",
                           "gold": "IDs cited by the answer span, matched to the IDs the documents print",
                           "target_pf": a.target_pf, "model_pf": pf}})
    json.dump(nm, open(os.path.join(a.out, "metadata.json"), "w"), indent=1)
    json.dump(stats, open(os.path.join(a.out, "docdrop_stats.json"), "w"), indent=1)
    open(os.path.join(a.out, "docdrop_examples.txt"), "w").write("\n".join(examples))
    if not a.no_sidecar:
        assert len(sidecar) == sum(len(p) for p in rows), "duplicate fingerprints"
        json.dump(sidecar, open(os.path.join(a.out, "keep_sidecar.json"), "w"))
    print(json.dumps({k: v for k, v in stats.items() if k != "tasks"}, indent=1))
    print(f"[docdrop] wrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
