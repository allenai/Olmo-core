"""
GPU-free validation of the CTC-suite tasks in ``eval_lc_native_docchunk_ladder.py`` (SUITE_TASKS),
and (``--base-tasks``) of nq / outlier / oolong / contradiction graded by the ctc spec
(``--ctc-grade``, CTC_GRADED_BASE) on their suite rows (contradiction = ``contradiction_iid``, whose
2k rung is ``rung_2560``) or on any rung file given with ``--files``.

For the first ``--rows`` rows of each fetched rung file (``fetch_suite_eval_rows.py`` layout):

1. **Prefix parity.** The ladder's prefill (``segment_prompt_to_chunks(include_answer=False)`` +
   dense emitter, exactly as ``build_prefill``) must be an exact token prefix of the converter's
   training sequence (``convert_unified_to_document_landmark.tokenize_example``), and every training
   token after it must be label-masked answer up to the appended EOS, with ``<|im_end|>`` in it.
   Also counts documents the wrapper could not find verbatim (they would stay FREE / unisolated).
   ``--train-query-position`` / ``--train-tokenizer`` render the training side as the SHARDS were
   built (setA / mt11: ``both`` + ``Qwen/Qwen3.5-4B``; read them from the shard's metadata.json),
   so a ladder ``--query-position`` that differs from the training data fails parity.
2. **Gold scores max.** The training tokens after the prefill -- what a perfect model would emit --
   run through the ladder's decode stopping rule (``finish_tokens``: stop before EOS/<|im_end|>,
   preset early stop at newlines) and the ladder's score_fn must give the maximum (1.0) on the
   primary metric with parse_rate 1.0.
3. **Garbage scores low.** An empty answer and a junk sentence must score 0 on the primary metric.
   Base tasks also report the native ``ctc_eval`` grader on gold / junk and its mean |difference|
   from the ctc grader over perturbed gold answers (drop one id, add a wrong id).
4. **Budget.** Gold answer token counts over ALL rows vs the decode budget the ladder uses.

Runs on CPU (imports torch through the ladder module, so use a compute node):

    PYTHONPATH=<src>:<recipes/eval/legacy_native> python validate_suite_ladder_tasks.py \
        --rows-root <out>/suite --tokenizer Qwen/Qwen3.5-0.8B-Base --query-position after
"""

import argparse
import importlib.util
import json
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Count(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.n = 0

    def emit(self, record):
        if "not found verbatim" in record.getMessage():
            self.n += 1


#: ladder base task -> (suite subset, {rung tokens: file tokens}) for --base-tasks defaults
BASE_SUBSETS = {"nq": ("nq", {}), "outlier": ("outlier", {}), "oolong": ("oolong", {}),
                "contradiction": ("contradiction_iid", {"2048": "2560"}), "rerank": ("rerank", {})}


def _perturb(text):
    """Near-miss variants of a gold answer: last integer +1, first integer +1, and the gold answer
    followed by a rambling second (wrong) answer line (tests the stop preset / parser)."""
    import re

    nums = list(re.finditer(r"\d+", text))
    if not nums:
        return [text + " xyz", "xyz", text + "\n" + "xyz"]
    last, first = nums[-1], nums[0]
    bump_last = text[: last.start()] + str(int(last.group()) + 1) + text[last.end():]
    bump_first = text[: first.start()] + str(int(first.group()) + 1) + text[first.end():]
    return [bump_last, bump_first, text + "\n" + bump_last]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows-root", required=True, help="<out>/suite from fetch_suite_eval_rows.py")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B-Base", help="eval tokenizer")
    ap.add_argument("--train-tokenizer", default=None, help="default: --tokenizer")
    ap.add_argument("--query-position", default="after", help="the ladder's (eval) query position")
    ap.add_argument("--train-query-position", default=None,
                    help="the training shards' query position (default: --query-position)")
    ap.add_argument("--rows", type=int, default=20)
    ap.add_argument("--rungs", nargs="+", default=["2048", "8192"])
    ap.add_argument("--tasks", nargs="+", default=None, help="SUITE_TASKS subset (default all)")
    ap.add_argument("--base-tasks", nargs="*", default=[],
                    help="nq/outlier/oolong/contradiction/rerank: graded by the ctc spec (--ctc-grade)")
    ap.add_argument("--files", default=None,
                    help='JSON {task: {rung: path}} replacing the default rung files of a task')
    ap.add_argument("--out", default=None, help="write the summary JSON here")
    args = ap.parse_args()
    train_qp = args.train_query_position or args.query_position

    sys.path.insert(0, os.path.join(REPO, "recipes", "eval", "legacy_native"))
    sys.path.insert(0, os.path.join(REPO, "src"))
    L = _load(os.path.join(REPO, "recipes/eval/legacy_native/ctc_eval/eval/"
                                 "eval_lc_native_docchunk_ladder.py"), "ladder")
    C = _load(os.path.join(REPO, "recipes/data/convert_unified_to_document_landmark.py"), "conv")
    from ctc_eval.eval import evaluate as E
    from ctc_eval.eval.evaluate import load_unified_examples
    from transformers import AutoTokenizer

    from olmo_core.data.corpus_reasoning_prompts.data_format import build_prompt
    from olmo_core.data.document_chunk_landmark import (
        RESERVED_IDS,
        emit_document_chunk_dense,
        segment_prompt_to_chunks,
    )

    counter = _Count()
    logging.getLogger("olmo_core.data.document_chunk_landmark").addHandler(counter)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ttok = AutoTokenizer.from_pretrained(args.train_tokenizer) if args.train_tokenizer else tok
    ids = RESERVED_IDS["qwen3_5"]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    newline = tok("\n", add_special_tokens=False).input_ids[-1]
    native_fns = {"contradiction": E._eval_contradiction, "nq": E._eval_retrieval,
                  "outlier": E._eval_outlier, "oolong": E._eval_oolong}
    native_stop = {"contradiction": "contradicting pairs:", "oolong": "answer:"}
    suite = [t for t in L.SUITE_TASKS if not args.tasks or t in args.tasks]
    base = list(args.base_tasks)
    # the ladder's own spec table + defaults (no GPU needed)
    targs = argparse.Namespace(tasks=",".join(suite), suite_max_new_tokens="",
                               ctc_grade=",".join(base),
                               contra_max_new_tokens=96, nq_max_new_tokens=64,
                               oolong_max_new_tokens=200, rerank_max_new_tokens=256,
                               outlier_max_new_tokens=200)
    tspec = L.build_task_spec(targs)
    files = json.loads(args.files) if args.files else {}

    summary, bad = {}, 0
    for task in suite + base:
        if task in L.SUITE_TASKS:
            grader, sub, alias = L.SUITE_TASKS[task][1], L.SUITE_TASKS[task][2], {}
        else:
            grader, (sub, alias) = L.CTC_GRADED_BASE[task], BASE_SUBSETS[task]
        loadtask, chunk_by, item_regex, primary, max_new, stopkind = tspec[task]
        score_fn = L.make_ctc_score_fn(grader)
        ac = L.make_ctc_answer_complete(grader, lambda t: tok.decode(t, skip_special_tokens=True))
        spec = L.ctc_spec(grader)

        def finish(gen_ids, ac=ac):  # == the ladder's finish_tokens (EOS/im_end stop, early stop)
            out = []
            for t in gen_ids:
                if t in (ids.eos, im_end):
                    break
                out.append(t)
                if t == newline and ac is not None and ac(out):
                    break
            text = tok.decode(out, skip_special_tokens=True)
            return text.split("</think>", 1)[1] if "</think>" in text else text

        nkey = native_stop.get(task)
        native_ac = (lambda o, k=nkey: k in tok.decode(o, skip_special_tokens=True).lower()) \
            if nkey else None

        for rung in args.rungs:
            path = files.get(task, {}).get(rung) or os.path.join(
                args.rows_root, sub, f"rung_{alias.get(rung, rung)}.jsonl")
            exs = load_unified_examples(path, 500, task=loadtask,
                                        query_position=args.query_position, use_alpaca=True)
            sel = exs[: args.rows]
            n_par = n_unmatched_rows = 0
            golds, golds_native, prefill_lens, problems = [], [], [], []
            for i, e in enumerate(sel):
                raw = e["ex"]
                before = counter.n
                segs, _, _ = segment_prompt_to_chunks(
                    tok, raw, loadtask, query_position=args.query_position, cot_mode="none",
                    chunk_by=chunk_by, item_regex=item_regex, include_answer=False,
                    doc_start_id=ids.doc_start, doc_end_id=ids.doc_end)
                pre, _ = emit_document_chunk_dense(segs)
                n_unmatched_rows += counter.n > before
                prefill_lens.append(len(pre))
                tr = C.tokenize_example(
                    ttok, raw, loadtask, emit="dense", query_position=train_qp,
                    cot_mode="none", mem_freq=63, seq_len=10 ** 9, chunk_by=chunk_by,
                    item_regex=item_regex, use_titles=False, ids_set=ids)
                tr_ids, tr_mask = [int(x) for x in tr[0]], [bool(x) for x in tr[1]]
                rest, rest_mask = tr_ids[len(pre):], tr_mask[len(pre):]
                ok = (tr_ids[: len(pre)] == pre and rest and rest[-1] == ids.eos
                      and all(rest_mask[:-1]) and not rest_mask[-1] and im_end in rest)
                if ok:
                    n_par += 1
                else:
                    k = next((j for j, (a, b) in enumerate(zip(pre, tr_ids)) if a != b),
                             min(len(pre), len(tr_ids)))
                    problems.append(f"#{i}: first diff at {k}/{len(pre)}: eval "
                                    f"{tok.decode(pre[k:k + 10])!r} | train "
                                    f"{ttok.decode(tr_ids[k:k + 10])!r}")
                # the gold continuation: the training answer tokens (from the eval prefill on, or
                # from the training prompt's end when parity failed)
                if not ok:
                    first = next((j for j, m in enumerate(tr_mask) if m), len(tr_ids))
                    rest = tr_ids[first:]
                golds.append(finish(rest))
                golds_native.append(finish(rest, ac=native_ac))
            res_gold, rows_gold = score_fn(sel, golds)
            res_empty, _ = score_fn(sel, [""] * len(sel))
            res_junk, _ = score_fn(sel, ["I am not sure what the answer is."] * len(sel))
            # gold answer length over ALL rows (what the decode budget must hold)
            # (the converter's target string: build_prompt(use_alpaca=False, cot_mode="none"); +1 for
            # the <|im_end|> the model must also emit)
            ans_tok = [len(tok(build_prompt(e["ex"], task=loadtask,
                                            query_position=train_qp, use_alpaca=False,
                                            cot_mode="none", use_titles=False)[1],
                               add_special_tokens=False).input_ids) + 1 for e in exs]
            key = f"{task}@{rung}"
            rec = {
                "grader": f"ctc:{grader}", "primary": primary, "stop": spec.stop,
                "file": path, "eval_query_position": args.query_position,
                "train_query_position": train_qp,
                "max_new_tokens": max_new, "rows_checked": len(sel),
                "prefix_parity": f"{n_par}/{len(sel)}",
                "rows_with_unwrapped_docs": n_unmatched_rows,
                "gold_primary": res_gold[primary], "gold_parse_rate": res_gold["parse_rate"],
                "empty_primary": res_empty[primary], "junk_primary": res_junk[primary],
                "gold_answer_tokens_max_all_rows": max(ans_tok),
                "gold_answer_tokens_mean_all_rows": round(sum(ans_tok) / len(ans_tok), 1),
                "prefill_tokens_max": max(prefill_lens), "eval_size": len(exs),
                "gold_text_example": golds[0][:160],
            }
            if task in native_fns:  # native grader beside the ctc one
                nf = native_fns[task]
                ng, _ = nf(sel, golds_native)
                nj, _ = nf(sel, ["I am not sure what the answer is."] * len(sel))
                diffs, cmeans, nmeans = [], [], []
                for v in range(3):
                    pert = [_perturb(g)[v] for g in golds]
                    rc, rows_c = score_fn(sel, pert)
                    rn, rows_n = nf(sel, pert)
                    cmeans.append(round(rc[primary], 4))
                    nmeans.append(round(rn[primary], 4))
                    diffs += [abs(a[primary] - b[primary]) for a, b in zip(rows_c, rows_n)]
                rec.update({"native_gold_primary": ng[primary], "native_junk_primary": nj[primary],
                            "perturbed_ctc_primary": cmeans, "perturbed_native_primary": nmeans,
                            "perturbed_mean_abs_diff": round(sum(diffs) / len(diffs), 4)})
            fail = []
            if n_par != len(sel):
                fail.append("parity")
            if abs(res_gold[primary] - 1.0) > 1e-9 or res_gold["parse_rate"] != 1.0:
                fail.append("gold<max")
            if res_empty[primary] > 0 or res_junk[primary] > 0:
                fail.append("garbage>0")
            if max(ans_tok) > max_new:
                fail.append("budget")
            if n_unmatched_rows:
                fail.append("unwrapped-docs")
            rec["status"] = "OK" if not fail else "FAIL:" + ",".join(fail)
            bad += bool(fail)
            summary[key] = rec
            print(f"[{rec['status']}] {key}: {json.dumps(rec)}", flush=True)
            for p in problems[:3]:
                print(f"    {p}", flush=True)
            if rec["gold_primary"] < 1.0:
                worst = sorted(range(len(rows_gold)), key=lambda j: rows_gold[j][primary])[:2]
                for j in worst:
                    print(f"    low gold #{j}: {rows_gold[j]} text={golds[j][:200]!r}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
    print(f"\n{len(summary) - bad}/{len(summary)} task@rung cells pass")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
