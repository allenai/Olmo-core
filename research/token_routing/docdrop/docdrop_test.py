"""CPU unit test for docdrop.py: the kept set is exactly all gold plus a random share of non-gold, surviving
documents keep their printed IDs, labels are untouched. Pure numpy (no torch / tokenizer):

    python -m pytest research/token_routing/docdrop/docdrop_test.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import docdrop as DD  # noqa: E402

DS, DE, EOS = 900, 901, 999
PROMPT, QUERY, ANSWER = [1, 2, 3], [4, 5], [7, 8, 9]


def make_row(n_docs, rng):
    """prefix, n documents [DS, id, body..., DE] (the 'printed ID' is the token right after DS), query, answer, EOS."""
    toks, spans = list(PROMPT), []
    for i in range(n_docs):
        body = list(rng.integers(10, 800, size=int(rng.integers(3, 9))))
        s = len(toks)
        toks += [DS, 1000 + (i + 1)] + body + [DE]
        spans.append((s, len(toks)))
    toks += QUERY
    a0 = len(toks)
    toks += ANSWER + [EOS]
    ids = np.array(toks, dtype=np.int64)
    mask = np.zeros(len(ids), dtype=bool)
    mask[a0:a0 + len(ANSWER)] = True
    return ids, mask, spans


def printed_ids(ids):
    return [int(ids[i + 1]) - 1000 for i in np.flatnonzero(ids == DS)]


@pytest.mark.parametrize("task", ["contradiction", "nq", "outlier"])
@pytest.mark.parametrize("q", [0.0, 0.1, 0.19, 0.5, 1.0])
def test_keep_is_all_gold_plus_random_share_of_nongold(task, q):
    rng = np.random.default_rng(0)
    for row in range(40):
        n = int(rng.integers(5, 120))
        gold = sorted(rng.choice(n, size=int(rng.integers(0, min(6, n))), replace=False).tolist())
        keep = DD.choose_keep(n, gold, q, DD.row_seed(0, 1, row), task)
        assert list(keep) == sorted(set(keep.tolist())), "sorted, no duplicates"
        assert set(gold) <= set(keep.tolist()), "every gold doc kept"
        n_ng = n - len(gold)
        want = DD.keep_count(n_ng, len(gold), q, task)
        assert len(keep) - len(gold) == want
        assert want == min(n_ng, max(1, round(q * n_ng), (2 * len(gold) + 1) if task == "outlier" else 0))


def test_nongold_choice_is_uniform_and_nested():
    n, gold, q = 50, [3, 17], 0.2
    counts = np.zeros(n)
    for row in range(4000):
        keep = DD.choose_keep(n, gold, q, DD.row_seed(7, 0, row), "nq")
        counts[keep] += 1
        small = DD.choose_keep(n, gold, 0.1, DD.row_seed(7, 0, row), "nq")
        assert set(small.tolist()) <= set(keep.tolist()), "kept sets nested in q"
    ng = np.array([d for d in range(n) if d not in gold])
    freq = counts[ng] / 4000
    k = DD.keep_count(len(ng), 2, q, "nq")
    assert counts[gold].min() == 4000
    assert np.allclose(freq, k / len(ng), atol=0.03), "each non-gold doc survives with the same probability"


def test_row_seed_is_deterministic():
    a = DD.choose_keep(80, [5], 0.3, DD.row_seed(0, 2, 11), "outlier")
    b = DD.choose_keep(80, [5], 0.3, DD.row_seed(0, 2, 11), "outlier")
    c = DD.choose_keep(80, [5], 0.3, DD.row_seed(1, 2, 11), "outlier")
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_drop_preserves_ids_order_prompt_and_labels():
    rng = np.random.default_rng(1)
    for row in range(30):
        n = int(rng.integers(4, 60))
        ids, mask, spans = make_row(n, rng)
        pe, sp, ss = DD.split_row(ids, DS, DE)
        assert sp == spans and pe == len(PROMPT)
        gold = sorted(rng.choice(n, size=3, replace=False).tolist())
        keep = DD.choose_keep(n, gold, 0.25, DD.row_seed(0, 0, row), "contradiction")
        nr, nm = DD.drop_docs(ids, mask, pe, sp, ss, keep)
        # surviving documents print the SAME IDs as in the full row (1-based position there), in order
        assert printed_ids(nr) == [k + 1 for k in keep]
        assert set(g + 1 for g in gold) <= set(printed_ids(nr))
        # each kept document is byte-identical to its full-row span
        _, sp2, _ = DD.split_row(nr, DS, DE)
        for (a, b), k in zip(sp2, keep):
            s, e = spans[k]
            assert np.array_equal(nr[a:b], ids[s:e])
        assert np.array_equal(nr[:pe], ids[:pe])  # prompt untouched
        assert np.array_equal(nr[nm], ids[mask]) and nm.sum() == mask.sum()  # answer/labels untouched
        assert np.array_equal(nr[-len(QUERY) - len(ANSWER) - 1:], ids[-len(QUERY) - len(ANSWER) - 1:])
        assert len(nr) == len(ids) - sum(spans[d][1] - spans[d][0] for d in range(n) if d not in set(keep))


def test_refuses_to_drop_a_span_with_loss_tokens_and_bad_rows():
    rng = np.random.default_rng(2)
    ids, mask, spans = make_row(5, rng)
    mask[spans[1][0] + 2] = True
    pe, sp, ss = DD.split_row(ids, DS, DE)
    with pytest.raises(ValueError):
        DD.drop_docs(ids, mask, pe, sp, ss, [0, 2, 3, 4])
    bad = ids.copy()
    bad = np.insert(bad, spans[1][1], 5)  # a token between two documents
    with pytest.raises(ValueError):
        DD.split_row(bad, DS, DE)
    with pytest.raises(ValueError):
        DD.choose_keep(5, [5], 0.5, DD.row_seed(0, 0, 0), "nq")


def test_gold_and_id_parsers():
    assert DD.gold_ids_from_answer("[[22, 131], [110, 152]]<|im_end|>\n", "contradiction") == [22, 110, 131, 152]
    assert DD.gold_ids_from_answer("[]<|im_end|>\n", "contradiction") == []
    assert DD.gold_ids_from_answer("[25]<|im_end|>\n", "nq") == [25]
    assert DD.gold_ids_from_answer("The majority ... 5 stars.\nOutliers: [18], [39], [57]<|im_end|>", "outlier") == [18, 39, 57]
    assert DD.doc_id_from_text("\n\nClaim 37: The increased", "contradiction") == 37
    assert DD.doc_id_from_text("\n\nDocument [12]: \"Title\"", "nq") == 12
    assert DD.doc_id_from_text("\n\nsomething else", "outlier") is None
