"""Exact label rewriting for a SUBSET of a row's documents, for tasks with no small gold set (every document feeds
the answer): reorder, oolong, grouping (the suite's unlabeled OpenAlex partition).

``subset(task, ex, keep, ctx)`` returns a new unified-format example holding only the documents (oolong: item lines)
at the 0-based positions ``keep`` (shown order preserved, display ids renumbered 1..m by the renderer) with the
answer recomputed for them, or raises ``Degenerate`` when the subset has no well-defined answer under the generator's
own rules (oolong: a tied argmax, or the question's label / user / month is gone). ``subset(task, ex, all)`` returns
the row unchanged in every rendered field (checked by ``selfcheck``).

* reorder: ``gold_order[i]`` = display id of the passage at source position i. Restricted: the kept display ids in
  source order, each mapped to its new display id.
* grouping: clusters (``gold_doc_indices``, 0-based) restricted to the kept docs, empty clusters dropped (cluster
  order kept), k = non-empty clusters; the query is re-rendered in the same one of the generator's three phrasings
  with the new k (and n); answer JSON exactly as ``generate_arxiv_grouping_data.build_example`` writes it.
* oolong: one document = preamble + "\\n\\n" + item lines. The kept lines are looked up in the seed pool (each line
  carries date / user / label there); the question is rebuilt by ctc's own ``oolong.generate.build_variant`` with the
  SAME variant key and the SAME sampled parameters (label / user / month / n, replayed through a fixed-choice RNG),
  so question text (incl. its label list) and gold are what the generator computes for that item set; the preamble's
  N is re-rendered.
"""
from __future__ import annotations

import copy
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Sequence


class Degenerate(Exception):
    pass


# ------------------------------------------------------------------------------------------------------------ reorder
def reorder_subset(ex: Dict, keep: Sequence[int]) -> Dict:
    keep = sorted(set(int(k) for k in keep))
    n = len(ex["documents"])
    if len(keep) < 2:
        raise Degenerate("reorder needs >= 2 passages")
    new_id = {old + 1: j + 1 for j, old in enumerate(keep)}  # display id (1-based) -> new display id
    go = [new_id[d] for d in ex["gold_order"] if d in new_id]
    assert sorted(go) == list(range(1, len(keep) + 1))
    out = copy.deepcopy(ex)
    out["documents"] = [ex["documents"][k] for k in keep]
    out["gold_order"] = go
    out["answers"] = [json.dumps(go)]
    m = out.setdefault("_meta", {})
    if len(keep) != n:
        m["n"] = len(keep)
        m.pop("passage_word_lens", None)
        m["rdd_kept"] = keep
    return out


# ----------------------------------------------------------------------------------------------------------- grouping
GROUP_PHRASINGS = [
    "Group these documents into {k} groups.",
    "Partition the following abstracts into {k} categories.",
    "Cluster these {n} papers into {k} groups.",
]


def _phrasing(q: str, k: int, n: int) -> int:
    for i, p in enumerate(GROUP_PHRASINGS):
        if p.format(k=k, n=n) == q:
            return i
    raise ValueError(f"unknown grouping query {q!r} (k={k}, n={n})")


def grouping_subset(ex: Dict, keep: Sequence[int]) -> Dict:
    keep = sorted(set(int(k) for k in keep))
    n = len(ex["documents"])
    k0 = len(ex["gold_doc_indices"])
    ph = _phrasing(ex["queries"][0], k0, n)
    if len(keep) < 2:
        raise Degenerate("grouping needs >= 2 documents")
    new = {old: j for j, old in enumerate(keep)}
    clusters, labels = [], []
    lab0 = ex.get("cluster_labels") or [None] * k0
    for c, lb in zip(ex["gold_doc_indices"], lab0):
        cc = sorted(new[i] for i in c if i in new)
        if cc:
            clusters.append(cc)
            labels.append(lb)
    k = len(clusters)
    out = copy.deepcopy(ex)
    out["documents"] = [ex["documents"][i] for i in keep]
    out["gold_doc_indices"] = clusters
    if ex.get("cluster_labels") is not None:
        out["cluster_labels"] = labels
    out["k"] = k
    out["queries"] = [GROUP_PHRASINGS[ph].format(k=k, n=len(keep))]
    out["answers"] = [json.dumps({"groups": [{"doc_ids": [i + 1 for i in idxs]} for idxs in clusters]})]
    if len(keep) != n:
        out["rdd_kept"] = keep
    return out


# ------------------------------------------------------------------------------------------------------------- oolong
_KEY = {("counting", "ANSWER_TYPE.LABEL"): "count_mostfreq_label", ("counting", "ANSWER_TYPE.NUMERIC"): "count_numeric",
        ("counting", "ANSWER_TYPE.COMPARISON"): "count_comparison", ("user", "ANSWER_TYPE.USER"): "user_mostfreq_user",
        ("user", "ANSWER_TYPE.LABEL"): "user_mostfreq_label", ("user", "ANSWER_TYPE.NUMERIC"): "user_numeric",
        ("user", "ANSWER_TYPE.COMPARISON"): "user_comparison", ("timeline", "ANSWER_TYPE.NUMERIC"): "timeline_repr_n",
        ("timeline", "ANSWER_TYPE.LABEL"): "timeline_month_label"}
_MONTHS = {datetime(2000, m, 1).strftime("%B"): m for m in range(1, 13)}


def oolong_params(key: str, q: str) -> List[Any]:
    """The values build_variant drew from its RNG, in draw order, parsed back from the question text."""
    def one(rx):
        m = re.search(rx, q, re.S)
        if not m:
            raise ValueError(f"{key}: cannot parse {rx!r} from {q[:300]!r}")
        return m.groups()
    if key in ("count_mostfreq_label", "user_mostfreq_user"):
        p: List[Any] = []
    elif key == "count_numeric":
        p = [one(r"classified as label '(.*?)'\? Give")[0]]
    elif key == "count_comparison":
        a, b = one(r"is label '(.*?)' more common, less common, or the same frequency as label '(.*?)'\? Give")
        p = [("sample", [a, b])]
    elif key.startswith("user_"):
        u = one(r"associated with user IDs (.*?)\. Among")[0]
        p = [u]
        if key == "user_numeric":
            p.append(one(r"classified as label '(.*?)'\? Give")[0])
        elif key == "user_comparison":
            a, b = one(r"is label '(.*?)' more common, less common, or the same frequency as label '(.*?)'\? Give")
            p.append(("sample", [a, b]))
    elif key == "timeline_repr_n":
        p = [int(one(r"represented exactly (\d+) time")[0])]
    elif key == "timeline_month_label":
        p = [_MONTHS[one(r"occur in (\w+) of any year")[0]]]
    else:
        raise ValueError(key)
    return p


class FixedRng:
    """Replays build_variant's draws; a value no longer available in the subset -> Degenerate."""

    def __init__(self, vals):
        self.v = list(vals)

    def choice(self, seq):
        x = self.v.pop(0)
        if isinstance(x, tuple) or x not in list(seq):
            raise Degenerate(f"choice {x!r} not available")
        return x

    def sample(self, seq, k):
        tag, xs = self.v.pop(0)
        assert tag == "sample" and len(xs) == k
        if any(x not in list(seq) for x in xs):
            raise Degenerate(f"sample {xs!r} not available")
        return list(xs)


class OolongCtx:
    """Seed pool lookup: unlabelled item line -> Item, per sub-dataset (lines are deduplicated per dataset)."""

    def __init__(self, pool):
        self.pool = pool
        self.by_line = {ds: {it.line: it for it in items} for ds, items in pool.items.items()}


def oolong_split(ex: Dict, ctx: OolongCtx):
    ds = ex["_meta"]["dataset"]
    text = ex["documents"][0]["text"]
    pre_n = ex["_meta"]["num_items"]
    pre = ctx.pool.preamble[ds].format(N=pre_n)
    assert text.startswith(pre + "\n\n"), "preamble mismatch"
    lines = text[len(pre) + 2:].split("\n")
    items = [ctx.by_line[ds][ln] for ln in lines]
    assert len(items) == pre_n
    return ds, items


def oolong_subset(ex: Dict, keep: Sequence[int], ctx: OolongCtx) -> Dict:
    from ctc.tasks.oolong.generate import build_variant, MIN_ITEMS

    keep = sorted(set(int(k) for k in keep))
    ds, items = oolong_split(ex, ctx)
    if len(keep) < MIN_ITEMS:
        raise Degenerate("too few items")
    m = ex["_meta"]
    key = _KEY[(m["task_group"], m["answer_type"])]
    drawn = [items[i] for i in keep]
    res = build_variant(key, drawn, ctx.pool.labels[ds], FixedRng(oolong_params(key, ex["queries"][0])))
    if res is None:
        raise Degenerate("tie / empty")
    q, gold = res
    out = copy.deepcopy(ex)
    pre = ctx.pool.preamble[ds].format(N=len(drawn))
    out["documents"][0]["text"] = pre + "\n\n" + "\n".join(it.line for it in drawn)
    out["queries"] = [q]
    out["answers"] = [gold]
    mm = out["_meta"]
    mm["gold_list"] = [gold]
    mm["num_items"] = len(drawn)
    mm["context_len"] = ctx.pool.preamble_tokens[ds] + sum(it.tokens + 1 for it in drawn)
    if len(keep) != len(items):
        mm["rdd_kept"] = keep
    return out


# ------------------------------------------------------------------------------------------------------------- common
def n_units(task: str, ex: Dict, ctx=None) -> int:
    if task == "oolong":
        return int(ex["_meta"]["num_items"])
    return len(ex["documents"])


def subset(task: str, ex: Dict, keep: Sequence[int], ctx=None) -> Dict:
    if task == "reorder":
        return reorder_subset(ex, keep)
    if task == "grouping":
        return grouping_subset(ex, keep)
    if task == "oolong":
        return oolong_subset(ex, keep, ctx)
    raise ValueError(task)


RENDERED = ("documents", "queries", "answers", "gold_order", "gold_doc_indices")


def selfcheck(task: str, ex: Dict, ctx=None) -> None:
    """subset over every unit must reproduce the row in every rendered field."""
    full = subset(task, ex, range(n_units(task, ex, ctx)), ctx)
    for f in RENDERED:
        assert full.get(f) == ex.get(f), (task, f)
