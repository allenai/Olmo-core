"""Multi-question pools (POOL_V2.md, "Multi-question pools"): the SAME rows as a source pool, same order, each
keeping its prompt + original answer byte for byte, then K extra chat turns (user sub-question / assistant
sub-answer) derived from the row's gold, before the EOS. The labels mask covers the original answer and every
sub-answer (each answer + ``<|im_end|>\\n``, like the original); sub-questions and the assistant prefix are unmasked.
Runs on a node.

Sub-questions (printed IDs as the row prints them; every task prints document i as i+1):
- contradiction: "Which claim contradicts Claim i?" -> partner's ID (claims in a gold pair) or "none".
- outlier: "Is Document [i] about a different topic than the rest of these passages?" -> yes / no.
- reorder: "Which passage comes immediately after Passage [x] in the original order?" -> successor ID or "none"
  (last passage), alternating with "Put passages [a, b, c, d] in their original order." -> their gold relative order.
- qdmatch_nq: "Which document answers query [q]?" / "Which query does document [d] answer?" -> the partner's ID for a
  gold pair, "none" otherwise.
K per rung: 1k 2, 2k 4, 4k 8, 8k 16, 16k 32. Positives (gold-pair members / outliers) take up to half of K; negatives
fill the rest but at most 2x the positives used (contradiction has 6 gold claims, outlier 3, qdmatch 6), so K is
smaller where the row has few positives. Order of the K turns shuffled per row (fixed seed). A row that would exceed
16,384 tokens loses sub-questions from the end; the original answer is never touched.

Writes ``<src>_mq``: token / mask parts, metadata.json, gold_fingerprints.json / gold_pairs_fingerprints.json
(re-keyed to the new fingerprints, same gold), row_info.jsonl (updated len / answer_tokens / fingerprint),
mq_info.jsonl (per row: n_sub, orig_answer [s, e), sub_answers [[s, e), ...], questions, answers) and
``<src>_mq_stats.json`` (loss tokens per row before / after, K per rung, verification on --verify-rows rows per task).

    $PY build_mq_pools.py --src mtb16k ; $PY build_mq_pools.py --src mtbmix50
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_pool_v2 as P  # noqa: E402

SH = "/data/prasann/early_loss_proxy/shards"
K_RUNG = {"1k": 2, "2k": 4, "4k": 8, "8k": 16, "16k": 32}
IM_START, IM_END = 248045, 248046


def subquestions(task, ex, n, k, rng):
    """[(question, answer)] for one row from its SOURCE JSONL example (independent of the sidecar)."""
    if task == "reorder":
        order = [int(x) for x in ex["gold_order"]]  # printed passage IDs in original order
        pos = {x: i for i, x in enumerate(order)}
        ids = list(range(1, n + 1))
        out = []
        for j, x in enumerate(rng.sample(ids, min(k, n))):
            if j % 2 == 0 or n < 4:
                nxt = order[pos[x] + 1] if pos[x] + 1 < n else None
                out.append((f"Which passage comes immediately after Passage [{x}] in the original order? Output its "
                            f"passage ID, or none if it is the last passage.", str(nxt) if nxt else "none"))
            else:
                sub = rng.sample(ids, 4)
                out.append((f"Put passages {sub} in their original order. Write your answer as [id1, id2, ...]",
                            str(sorted(sub, key=lambda y: pos[y]))))
        return out
    if task == "contradiction":
        partner = {}
        for a, b in ex["gold_doc_indices"]:  # 1-based Claim IDs = printed IDs
            partner[int(a)] = int(b)
            partner[int(b)] = int(a)
        posl = sorted(partner)
        negl = [i for i in range(1, n + 1) if i not in partner]
        mk = lambda i: (f"Which claim contradicts Claim {i}? Output the claim ID, or none if no claim contradicts it.",  # noqa: E731
                        str(partner[i]) if i in partner else "none")
    elif task == "outlier":
        gold = {int(g) + 1 for g in ex["gold_doc_indices"]}  # 0-based -> printed
        posl = sorted(gold)
        negl = [i for i in range(1, n + 1) if i not in gold]
        mk = lambda i: (f"Is Document [{i}] about a different topic than the rest of these passages? Answer yes or no.",  # noqa: E731
                        "yes" if i in gold else "no")
    elif task == "qdmatch_nq":
        partner = {}
        for q, d in ex["gold_pairs"]:  # 1-based positions = printed IDs
            partner[int(q)] = int(d)
            partner[int(d)] = int(q)
        typ = {i + 1: dd.get("type") for i, dd in enumerate(ex["documents"])}
        posl = sorted(partner)
        negl = [i for i in range(1, n + 1) if i not in partner]

        def mk(i):
            if typ[i] == "query":
                return (f"Which document answers query [{i}]? Output the document's number, or none if no document "
                        f"answers it.", str(partner[i]) if i in partner else "none")
            return (f"Which query does document [{i}] answer? Output the query's number, or none if it answers no "
                    f"query.", str(partner[i]) if i in partner else "none")
    else:
        raise ValueError(task)
    n_pos = min(len(posl), (k + 1) // 2)
    n_neg = min(len(negl), k - n_pos, 2 * n_pos)
    picks = rng.sample(posl, n_pos) + rng.sample(negl, n_neg)
    rng.shuffle(picks)
    return [mk(i) for i in picks]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="pool name under shards/, e.g. mtb16k")
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--verify-rows", type=int, default=20)
    a = ap.parse_args()
    from tokenizers import Tokenizer

    sys.path.insert(0, f"{P.REPO}/src")
    from olmo_core.nn.attention.gold_grad_mask import content_fingerprint

    tok = Tokenizer.from_file(f"{P.TOKENIZER}/tokenizer.json")
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids  # noqa: E731
    dec = lambda x: tok.decode([int(i) for i in x], skip_special_tokens=False)  # noqa: E731
    assert enc("<|im_start|>user\n")[0] == IM_START and enc("<|im_end|>\n")[0] == IM_END
    src, final, tmp = f"{SH}/{a.src}", f"{SH}/{a.src}_mq", f"{SH}/{a.src}_mq_tmp"
    if os.path.exists(f"{final}/metadata.json"):
        raise SystemExit(f"{final} exists")
    os.system(f"rm -rf {tmp}")
    os.makedirs(tmp)
    meta = json.load(open(f"{src}/metadata.json"))
    info = [json.loads(x) for x in open(f"{src}/row_info.jsonl")]
    # source JSONL rows the pool uses
    need = {}
    for x in info:
        need.setdefault(x["src"], set()).add(x["src_idx"])
    exs = {}
    for f, idxs in need.items():
        with open(f) as fh:
            for i, line in enumerate(fh):
                if i in idxs:
                    exs[(f, i)] = json.loads(line)
    gold_side, pair_side = {}, {}
    stats = {"src": src, "out": final, "seed": a.seed, "K_per_rung": K_RUNG, "tasks": {}}
    info_out = open(f"{tmp}/row_info.jsonl", "w")
    mq_out = open(f"{tmp}/mq_info.jsonl", "w")
    tot_tok = tot_loss = 0
    lens_all = []
    k_ptr = 0
    for t, task in enumerate(P.TASKS):
        ids = np.fromfile(f"{src}/token_ids_part_t{t}_000.npy", dtype=meta["dtype"])
        msk = np.fromfile(f"{src}/labels_mask_t{t}_000.npy", dtype=meta["mask_dtype"])
        ends = np.flatnonzero(ids == P.EOS)
        starts = np.concatenate([[0], ends[:-1] + 1])
        rows_info = [x for x in info if x["part"] == t]
        assert len(rows_info) == len(ends)
        out_t, out_m = [], []
        st = {"rows": 0, "loss_before": [], "loss_after": [], "n_sub": {}, "len_before": [], "len_after": [],
              "subq_dropped_for_cap": 0, "rungs": {}}
        ver = {"rows_checked": 0, "prefix_identical": 0, "sub_answers_correct": 0, "mask_exact": 0, "examples": []}
        for r, (x, s, e) in enumerate(zip(rows_info, starts, ends)):
            row, m = ids[s:e + 1], msk[s:e + 1]
            assert len(row) == x["len"] and row[-1] == P.EOS
            ex = exs[(x["src"], x["src_idx"])]
            n = len(ex["documents"])
            assert n == x["n_docs"]
            a0 = int(np.flatnonzero(m)[0])
            a1 = int(np.flatnonzero(m)[-1]) + 1
            assert a1 == len(row) - 1 and m[a0:a1].all(), "original answer must be one span ending before EOS"
            im = np.flatnonzero(row[:a0] == IM_START)
            aprefix = row[int(im[-1]):a0].tolist()  # <|im_start|>assistant\n<think>\n\n</think>\n\n
            rng = random.Random(f"{a.seed}:{task}:{x['src']}:{x['src_idx']}")
            qa = subquestions(task, ex, n, K_RUNG[x["rung"]], rng)
            new = row[:a1].tolist()
            mask = m[:a1].tolist()
            spans, kept_qa = [], []
            for q, ans in qa:
                u = enc(f"<|im_start|>user\n{q}<|im_end|>\n") + aprefix
                av = enc(f"{ans}<|im_end|>\n")
                if len(new) + len(u) + len(av) + 1 > P.MAX_LEN:
                    st["subq_dropped_for_cap"] += len(qa) - len(kept_qa)
                    break
                new += u
                mask += [False] * len(u)
                spans.append([len(new), len(new) + len(av)])
                new += av
                mask += [True] * len(av)
                kept_qa.append((q, ans))
            new.append(P.EOS)
            mask.append(False)
            new = np.asarray(new, dtype=np.uint32)
            mask = np.asarray(mask, dtype=bool)
            fp = content_fingerprint(new.tolist())
            assert fp not in gold_side
            gold_side[fp] = x["gold"]
            if x["gold_pairs"] is not None:
                pair_side[fp] = x["gold_pairs"]
            if r < a.verify_rows:  # decode-based checks
                ver["rows_checked"] += 1
                ver["prefix_identical"] += bool(np.array_equal(new[:a1], row[:a1]))
                exp_mask = np.zeros(len(new), dtype=bool)
                exp_mask[a0:a1] = True
                ok = True
                for (q, ans), (ss, ee) in zip(kept_qa, spans):
                    exp_mask[ss:ee] = True
                    got = dec(new[ss:ee])
                    if got != f"{ans}<|im_end|>\n":
                        ok = False
                    pat = {"contradiction": r"Claim (\d+)\?", "outlier": r"Document \[(\d+)\]",
                           "qdmatch_nq": r"(?:query|document) \[(\d+)\]", "reorder": r"after Passage \[(\d+)\]"}[task]
                    mt = re.search(pat, q)
                    num = int(mt.group(1)) if mt else None
                    # independent re-derivation of the answer from the source gold
                    if task == "contradiction":
                        pr = {int(b): int(c) for b, c in ex["gold_doc_indices"]} | {int(c): int(b) for b, c in ex["gold_doc_indices"]}
                        ok &= got.startswith(str(pr[num]) if num in pr else "none")
                    elif task == "outlier":
                        ok &= got.startswith("yes" if (num - 1) in [int(g) for g in ex["gold_doc_indices"]] else "no")
                    elif task == "qdmatch_nq":
                        pr = {int(b): int(c) for b, c in ex["gold_pairs"]} | {int(c): int(b) for b, c in ex["gold_pairs"]}
                        ok &= got.startswith(str(pr[num]) if num in pr else "none")
                    else:
                        go = [int(y) for y in ex["gold_order"]]
                        if num is not None:
                            i = go.index(num)
                            ok &= got.startswith(str(go[i + 1]) if i + 1 < n else "none")
                        else:
                            sub = [int(y) for y in re.search(r"passages \[([\d, ]+)\]", q).group(1).split(",")]
                            ok &= got.startswith(str([y for y in go if y in sub]))
                ver["sub_answers_correct"] += ok
                ver["mask_exact"] += bool(np.array_equal(exp_mask, mask))
                if r < 1:
                    ver["examples"].append({"rung": x["rung"], "first_turn": dec(new[a1:spans[0][1]]) if spans else "",
                                            "n_sub": len(kept_qa)})
            st["loss_before"].append(int(m.sum()))
            st["loss_after"].append(int(mask.sum()))
            st["len_before"].append(len(row))
            st["len_after"].append(len(new))
            st["n_sub"][x["rung"]] = st["n_sub"].get(x["rung"], []) + [len(kept_qa)]
            out_t.append(new)
            out_m.append(mask)
            lens_all.append(len(new))
            x2 = dict(x, len=int(len(new)), answer_tokens=int(mask.sum()), fingerprint=fp,
                      orig_len=int(len(row)), orig_answer_tokens=int(m.sum()))
            info_out.write(json.dumps(x2) + "\n")
            mq_out.write(json.dumps({"part": t, "row": r, "task": task, "rung": x["rung"], "n_sub": len(kept_qa),
                                     "orig_answer": [a0, a1], "sub_answers": spans,
                                     "questions": [q for q, _ in kept_qa], "answers": [y for _, y in kept_qa]}) + "\n")
        bt = np.concatenate(out_t).astype("uint32")
        bm = np.concatenate(out_m).astype("bool")
        bt.tofile(f"{tmp}/token_ids_part_t{t}_000.npy")
        bm.tofile(f"{tmp}/labels_mask_t{t}_000.npy")
        tot_tok += bt.size
        tot_loss += int(bm.sum())
        lb, la = np.array(st["loss_before"]), np.array(st["loss_after"])
        Lb, La = np.array(st["len_before"]), np.array(st["len_after"])
        stats["tasks"][task] = {
            "rows": len(out_t), "loss_tokens_per_row_before": float(lb.mean()), "loss_tokens_per_row_after": float(la.mean()),
            "len_mean_before": float(Lb.mean()), "len_mean_after": float(La.mean()), "len_max_after": int(La.max()),
            "extra_token_share": float((La - Lb).sum() / Lb.sum()),
            "n_sub_mean_per_rung": {k: float(np.mean(v)) for k, v in st["n_sub"].items()},
            "n_sub_min_per_rung": {k: int(np.min(v)) for k, v in st["n_sub"].items()},
            "subq_dropped_for_cap": st["subq_dropped_for_cap"], "verification": ver}
        print(f"[mq] {a.src} {task}: loss tok/row {lb.mean():.1f} -> {la.mean():.1f}, len {Lb.mean():.0f} -> {La.mean():.0f} "
              f"(max {La.max()}), n_sub {stats['tasks'][task]['n_sub_mean_per_rung']}, cap drops {st['subq_dropped_for_cap']}; "
              f"verify {json.dumps({k: v for k, v in ver.items() if k != 'examples'})}", flush=True)
        print(f"[mq]   example: {ver['examples'][0] if ver['examples'] else None!r}", flush=True)
    info_out.close()
    mq_out.close()
    m2 = dict(meta)
    m2.update({"task": f"{meta.get('task')}_mq", "num_tokens": int(tot_tok), "num_loss_tokens": int(tot_loss),
               "max_example_len": int(max(lens_all)), "min_example_len": int(min(lens_all)),
               "multi_question": {"src": src, "K_per_rung": K_RUNG, "seed": a.seed, "sidecar": "mq_info.jsonl"}})
    json.dump(gold_side, open(f"{tmp}/gold_fingerprints.json", "w"))
    json.dump(pair_side, open(f"{tmp}/gold_pairs_fingerprints.json", "w"))
    stats.update({"num_instances": len(lens_all), "num_tokens": int(tot_tok), "num_loss_tokens": int(tot_loss),
                  "max_example_len": int(max(lens_all))})
    json.dump(stats, open(f"{final}_stats.json", "w"), indent=1)
    json.dump(m2, open(f"{tmp}/metadata.json", "w"), indent=1)
    os.rename(tmp, final)
    print(f"[mq] wrote {final}: {len(lens_all)} rows, {tot_tok} tokens, {tot_loss} loss tokens", flush=True)


if __name__ == "__main__":
    main()
