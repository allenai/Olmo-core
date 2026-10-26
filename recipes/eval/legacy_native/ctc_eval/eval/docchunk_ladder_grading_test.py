"""CPU tests for the docchunk ladder's ctc grading additions (2026-10-25).

* ``reorder``: the parse-tolerant partial-credit metrics (``kendall_tau_tolerant``,
  ``pairwise_acc_tolerant``) beside the unchanged strict ``kendall_tau``.
* ``rerank``: graded by the vendored ctc ``rerank`` spec under ``--ctc-grade`` (``ce_pos_recall``,
  primary mean over CE-scorable rows only, as olmo-eval does).
* the opt-in repetition stop, and the log-line format ``record_beaker_evals.py`` parses.

The ladder imports torch inside ``main()`` only, so this needs no GPU and no torch. Run on a compute
node with the node-local env:

    python -m pytest -q recipes/eval/legacy_native/ctc_eval/eval/docchunk_ladder_grading_test.py
"""

import argparse
import importlib.util
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, *[os.pardir] * 5))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


L = _load("docchunk_ladder", os.path.join(HERE, "eval_lc_native_docchunk_ladder.py"))


def _row(gold_order):
    return {"ex": {"documents": [{"text": f"p{i}"} for i in gold_order], "gold_order": gold_order}}


# ------------------------------------------------------------------ reorder: tolerant metric


GOLD = [4, 1, 6, 2, 5, 3]  # display ids in source order (a shuffled permutation, like the suite rows)
T = len(GOLD) * (len(GOLD) - 1) // 2


def test_tolerant_perfect_order():
    s = L.reorder_tolerant_scores(str(GOLD), GOLD)
    assert s == {"kendall_tau_tolerant": 1.0, "pairwise_acc_tolerant": 1.0, "id_coverage": 1.0,
                 "dup_rate": 0.0}


def test_tolerant_reversed_order():
    s = L.reorder_tolerant_scores(str(GOLD[::-1]), GOLD)
    assert s["kendall_tau_tolerant"] == -1.0
    assert s["pairwise_acc_tolerant"] == 0.0
    assert s["id_coverage"] == 1.0


def test_tolerant_duplicates_keep_first_occurrence():
    # every id emitted, then 4 and 1 repeated in the wrong place: the repeats are ignored
    text = "[4, 1, 4, 6, 2, 1, 5, 3]"
    s = L.reorder_tolerant_scores(text, GOLD)
    assert s["kendall_tau_tolerant"] == 1.0
    assert s["dup_rate"] == pytest.approx(2 / 8)
    # a duplicate whose FIRST occurrence is out of place costs exactly those pairs
    s = L.reorder_tolerant_scores("[1, 4, 1, 6, 2, 5, 3]", GOLD)
    assert s["pairwise_acc_tolerant"] == pytest.approx((T - 1) / T)
    assert s["kendall_tau_tolerant"] == pytest.approx((T - 2) / T)


def test_tolerant_repetition_loop():
    # the first four ids in order, then a loop that never reaches 5 and 3 (what 8k generations do)
    text = "[4, 1, 6, 2, " + "6, 2, " * 200 + "6"
    s = L.reorder_tolerant_scores(text, GOLD)
    # the 6 pairs among the 4 emitted ids are concordant; the 9 pairs touching 5 or 3 are missing
    assert s["kendall_tau_tolerant"] == pytest.approx(6 / T)
    assert s["pairwise_acc_tolerant"] == pytest.approx(6 / T)
    assert s["id_coverage"] == pytest.approx(4 / 6)
    assert s["dup_rate"] > 0.9


def test_tolerant_missing_ids_and_out_of_range():
    # only two ids, in the right order; 99 and 0 are not display ids and are ignored
    s = L.reorder_tolerant_scores("[99, 4, 0, 3]", GOLD)
    assert s["kendall_tau_tolerant"] == pytest.approx(1 / T)
    assert s["pairwise_acc_tolerant"] == pytest.approx(1 / T)
    assert s["id_coverage"] == pytest.approx(2 / 6)


def test_tolerant_empty_and_garbage_score_zero():
    for text in ("", "I am not sure.", None):
        s = L.reorder_tolerant_scores(text, GOLD)
        assert s["kendall_tau_tolerant"] == 0.0
        assert s["pairwise_acc_tolerant"] == 0.0
        assert s["id_coverage"] == 0.0


def test_reorder_score_fn_keeps_strict_and_adds_tolerant():
    score_fn = L.make_ctc_score_fn("reorder")
    rows = [_row(GOLD)] * 4
    looped = "[4, 1, 6, 2, " + "6, 2, " * 50
    res, per_row = score_fn(rows, [str(GOLD), looped, "", "<think>4, 1, 6"])
    # strict: only the exact permutation parses (1.0); the loop, the empty answer and the
    # unclosed think score 0 -- unchanged from before the tolerant metric was added
    assert [r["kendall_tau"] for r in per_row] == [1.0, 0.0, 0.0, 0.0]
    assert res["kendall_tau"] == pytest.approx(0.25)
    assert res["parse_rate"] == pytest.approx(0.25)
    assert res["eval_size"] == 4
    # tolerant: the loop gets partial credit; an unclosed think stays a failure (0)
    assert [r["kendall_tau_tolerant"] for r in per_row] == pytest.approx([1.0, 6 / T, 0.0, 0.0])
    assert res["kendall_tau_tolerant"] == pytest.approx((1.0 + 6 / T) / 4)
    for k in L.REORDER_TOLERANT_KEYS:
        assert k in res


def test_strict_kendall_tau_is_not_the_ordering_tau():
    """Documents the vendored strict metric's semantics: it correlates id VALUES by position, so an
    exact reversal of a non-identity gold is not -1. The tolerant metric is the ordering tau."""
    gold = [3, 1, 2]
    _, per_row = L.make_ctc_score_fn("reorder")([_row(gold)], ["[2, 1, 3]"])
    assert per_row[0]["kendall_tau"] == pytest.approx(1 / 3)
    assert per_row[0]["kendall_tau_tolerant"] == -1.0


# ------------------------------------------------------------------ rerank: ctc grading


def _rerank_row(ce_scores, gold=(0,)):
    return {"ex": {"documents": [{"text": f"d{i}"} for i in range(len(ce_scores))],
                   "queries": ["q"], "answers": [""], "gold_doc_indices": list(gold),
                   "ce_scores": list(ce_scores)}}


def test_rerank_graded_by_ctc_spec_with_gate():
    score_fn = L.make_ctc_score_fn("rerank")
    # CE-positive docs (1-based): row A -> {1, 4}; row B -> {2}; row C -> none (ungradable)
    rows = [_rerank_row([5.0, None, -3.0, 2.0, -10.0]),
            _rerank_row([-1.0, 7.5, -2.0, None], gold=(1,)),
            _rerank_row([-4.0, -6.0, None], gold=(2,))]
    responses = ["Ranking: [4], [3], [1]", "Ranking: [3], [1]\n[2]", "Ranking: [3]"]
    res, per_row = score_fn(rows, responses)
    # row B's [2] sits after the newline, which the rerank stop preset cuts: recall 0
    assert [r["ce_pos_recall"] for r in per_row] == [1.0, 0.0, 0.0]
    assert [r["ce_ref_available"] for r in per_row] == [1.0, 1.0, 0.0]
    # primary mean over the 2 CE-scorable rows only (olmo-eval skip_when_zero)
    assert res["ce_pos_recall"] == pytest.approx(0.5)
    assert res["eval_size_gated"] == 2
    assert res["eval_size"] == 3
    assert res["parse_rate"] == 1.0
    # mrr@10 against the qrel gold, over all rows: 1/3, 0, 1
    assert res["mrr@10"] == pytest.approx((1 / 3 + 0 + 1) / 3)


def test_rerank_partial_and_unparsed():
    score_fn = L.make_ctc_score_fn("rerank")
    rows = [_rerank_row([5.0, None, -3.0, 2.0, -10.0])] * 2
    res, per_row = score_fn(rows, ["[4], [4], [2]", "no ranking here"])
    assert [r["ce_pos_recall"] for r in per_row] == [0.5, 0.0]
    assert res["parse_rate"] == 0.5


def _spec_args(**kw):
    base = dict(tasks="rerank,reorder", suite_max_new_tokens="", ctc_grade="rerank",
                contra_max_new_tokens=96, nq_max_new_tokens=64, oolong_max_new_tokens=200,
                rerank_max_new_tokens=256, outlier_max_new_tokens=200)
    base.update(kw)
    return argparse.Namespace(**base)


def test_rerank_ctc_grade_task_spec():
    assert "rerank" in L.ctc_graded_tasks(_spec_args(ctc_grade="all"))
    spec = L.build_task_spec(_spec_args())
    loadtask, chunk_by, _, primary, max_new, stopkind = spec["rerank"]
    assert (loadtask, chunk_by, primary, stopkind) == ("rerank", "document", "ce_pos_recall",
                                                       "ctc:rerank")
    assert max_new == 512  # olmo-eval's budget: max(newline preset 64, spec 512)
    # without --ctc-grade, rerank keeps the native grader spec (unchanged)
    assert L.build_task_spec(_spec_args(ctc_grade=""))["rerank"][3] is None
    # the newline early stop is wired
    ac = L.make_ctc_answer_complete("rerank", lambda ids: "".join(ids))
    assert ac(["Ranking: [1]", "\n"]) and not ac(["Ranking: [1]"])


# ------------------------------------------------------------------ repetition stop


def test_repetition_stop_id():
    pieces = ["[", "4", ",", " 1", ",", " 6", ",", " 1", ",", " 2"]
    rs = L.RepetitionStop("id", piece=lambda t: pieces[t])
    fired = [i for i in range(len(pieces)) if rs.update(i)]
    assert fired[0] == 8  # the "," that terminates the second "1"
    # multi-digit ids split across tokens are assembled before comparing
    pieces = ["[1", "2", ", 2", ", 1", "2", "]"]
    rs = L.RepetitionStop("id", piece=lambda t: pieces[t])
    assert [rs.update(i) for i in range(len(pieces))] == [False] * 5 + [True]


def test_repetition_stop_ngram():
    toks = [7, 8, 9, 10, 8, 9, 10]
    rs = L.RepetitionStop("ngram", ngram=3)
    assert [rs.update(t) for t in toks] == [False] * 6 + [True]
    rs.reset()
    assert not any(rs.update(t) for t in [1, 2, 3, 4, 5, 6])


def test_repetition_stop_off_by_default():
    assert L.make_repetition_stop("off", 16, None) is None
    assert L.make_repetition_stop("", 16, None) is None
    with pytest.raises(ValueError):
        L.make_repetition_stop("bogus", 16, lambda t: "")


# ------------------------------------------------------------------ log format


def test_log_lines_match_record_beaker_evals_regex():
    rec = _load("record_beaker_evals",
                os.path.join(REPO, "research", "token_routing", "ds64_fast8k", "record_beaker_evals.py"))
    log = ("[ladder:reorder@8k] kendall_tau=0.001 (n=500, skipped_too_long=0)\n"
           "[ladder:reorder@8k] kendall_tau_tolerant=-0.042 (n=500, skipped_too_long=0)\n"
           "[ladder:rerank@2k] ce_pos_recall=0.712 (n=500, skipped_too_long=0)\n"
           "[ladder:rerank@2k] ce_pos_recall averaged over 486/500 rows with a CE-positive document\n")
    got = [m.groups() for m in rec.LINE.finditer(log)]
    assert got == [("reorder", "8k", "kendall_tau", "0.001", "500"),
                   ("reorder", "8k", "kendall_tau_tolerant", "-0.042", "500"),
                   ("rerank", "2k", "ce_pos_recall", "0.712", "500")]
