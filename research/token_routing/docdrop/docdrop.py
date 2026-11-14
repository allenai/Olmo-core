"""Random non-gold document dropping for the 4-task fast8k mix (records/docdrop-nongold-baseline.md).

Pure numpy + stdlib (no torch, no tokenizer) so the selection logic is unit-testable on CPU
(``docdrop_test.py``). The builder (``build_docdrop_shard.py``) adds the tokenizer (to read each document's
printed ID and the answer's gold IDs) and the FLOP model.

A training row is ``prefix + doc_1 + ... + doc_n + suffix``; each document is one
``<|box_start|> ... <|box_end|>`` span (markers included) and prints its own ID in its text
("Claim 37: ..." for contradiction, "Document [37]: ..." for nq and outlier). Dropping removes whole spans:
the surviving documents keep their text, so document 37 is still "37" when 30-36 are gone, and the answer
(which cites gold IDs) stays valid without relabelling.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: merged-shard part index -> task (mt4f1952 metadata.json ``merged_from`` order)
PART_TASK = {0: "contradiction", 1: "nq", 2: "outlier", 3: "oolong"}

_DOC_ID = {
    "contradiction": re.compile(r"^\s*Claim (\d+):"),
    "nq": re.compile(r"^\s*Document \[(\d+)\]:"),
    "outlier": re.compile(r"^\s*Document \[(\d+)\]:"),
}


def split_row(ids: Sequence[int], doc_start: int, doc_end: int) -> Tuple[int, List[Tuple[int, int]], int]:
    """``(prefix_end, spans, suffix_start)``: ``ids[:prefix_end]`` is the prefix, ``spans`` the half-open
    ``[s, e)`` document spans (both markers inside), ``ids[suffix_start:]`` the suffix.

    Raises if markers are unbalanced or nested, or if any token sits BETWEEN two documents (dropping a span would
    then also have to decide about that token; none of the mt4 tasks has any)."""
    a = np.asarray(ids)
    starts = np.flatnonzero(a == doc_start)
    ends = np.flatnonzero(a == doc_end)
    if len(starts) != len(ends):
        raise ValueError(f"unbalanced markers: {len(starts)} starts vs {len(ends)} ends")
    if len(starts) == 0:
        return len(a), [], len(a)
    if not np.all(starts < ends) or not np.all(ends[:-1] < starts[1:]):
        raise ValueError("nested or interleaved document markers")
    gaps = starts[1:] - (ends[:-1] + 1)
    if np.any(gaps != 0):
        raise ValueError(f"{int((gaps != 0).sum())} non-empty gaps between documents")
    spans = [(int(s), int(e) + 1) for s, e in zip(starts, ends)]
    return int(starts[0]), spans, int(ends[-1]) + 1


def doc_id_from_text(text: str, task: str) -> Optional[int]:
    """The ID a document prints at its start (``None`` if the pattern is absent)."""
    m = _DOC_ID[task].match(text)
    return int(m.group(1)) if m else None


def gold_ids_from_answer(text: str, task: str) -> List[int]:
    """Gold document IDs cited by the answer span (the label-mask tokens).

    contradiction: every integer in ``[[a, b], ...]`` (``[]`` -> no gold); nq: ``[id]``; outlier: the integers
    after ``Outliers:``."""
    if task == "outlier":
        if "Outliers:" not in text:
            raise ValueError(f"outlier answer without 'Outliers:': {text!r}")
        text = text.split("Outliers:", 1)[1]
    elif task not in ("contradiction", "nq"):
        raise ValueError(f"no gold parser for task {task}")
    return sorted({int(x) for x in re.findall(r"\d+", text)})


def keep_count(n_nongold: int, n_gold: int, keep_frac: float, task: str) -> int:
    """How many NON-gold documents survive: ``round(keep_frac * n_nongold)``, at least 1; for outlier at least
    ``2 * n_gold + 1``, so the majority the outliers are defined against stays a clear majority of what is visible."""
    k = max(1, int(round(keep_frac * n_nongold)))
    if task == "outlier":
        k = max(k, 2 * n_gold + 1)
    return min(n_nongold, k)


def choose_keep(n_docs: int, gold_idx: Sequence[int], keep_frac: float, rng: np.random.Generator,
                task: str) -> np.ndarray:
    """Sorted 0-based indices of the documents to keep: every gold document plus a uniformly random
    ``keep_count`` of the non-gold ones. The non-gold order is ONE permutation drawn from ``rng`` regardless of
    ``keep_frac``, so kept sets are nested in ``keep_frac`` (the FLOP bisection is monotone)."""
    gold = sorted(set(int(g) for g in gold_idx))
    if any(g < 0 or g >= n_docs for g in gold):
        raise ValueError(f"gold index out of range: {gold} with {n_docs} docs")
    gset = set(gold)
    nongold = np.array([d for d in range(n_docs) if d not in gset], dtype=np.int64)
    perm = rng.permutation(len(nongold))
    k = keep_count(len(nongold), len(gold), keep_frac, task)
    return np.sort(np.concatenate([np.array(gold, dtype=np.int64), nongold[perm[:k]]]))


def choose_keep_groups(groups: Sequence[int], gold_idx: Sequence[int], keep_frac: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Group-aware keep set (outlier): every document of the gold group(s) plus WHOLE non-gold groups, in one random
    group order, as many as bring the kept non-gold count closest to ``round(keep_frac * n_nongold)`` (at least one
    group). Whole groups keep the generator's invariant -- every visible majority topic stays larger than the outlier
    topic -- and every visible topic is a cluster, so "the passages that share a topic" is not a tell."""
    groups = [int(g) for g in groups]
    gold = sorted(set(int(g) for g in gold_idx))
    gold_groups = {groups[i] for i in gold}
    other = sorted(set(groups) - gold_groups)
    sizes = {g: 0 for g in other}
    for g in groups:
        if g in sizes:
            sizes[g] += 1
    order = [other[i] for i in rng.permutation(len(other))]
    n_ng = sum(sizes.values())
    target = int(round(keep_frac * n_ng))
    cum, best_m, best_err = 0, 1, None
    for m, g in enumerate(order, 1):
        cum += sizes[g]
        err = abs(cum - target)
        if best_err is None or err < best_err:
            best_m, best_err = m, err
    kept = gold_groups | set(order[:best_m])
    return np.array([i for i, g in enumerate(groups) if g in kept], dtype=np.int64)


def drop_docs(ids: np.ndarray, mask: np.ndarray, prefix_end: int, spans: Sequence[Tuple[int, int]],
              suffix_start: int, keep: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """The row without the dropped documents: prefix + kept spans in their original order + suffix, with the
    label mask cut identically. Refuses to drop a span that carries loss tokens."""
    keep_set = set(int(k) for k in keep)
    pieces = [np.arange(0, prefix_end)]
    for d, (s, e) in enumerate(spans):
        if d in keep_set:
            pieces.append(np.arange(s, e))
        elif mask[s:e].any():
            raise ValueError(f"document {d} carries loss tokens; refusing to drop it")
    pieces.append(np.arange(suffix_start, len(ids)))
    sel = np.concatenate(pieces)
    return ids[sel], mask[sel]


def row_seed(seed: int, part: int, row: int) -> np.random.Generator:
    """The per-row generator (stable across builds: the same seed, part and row index always draw the same order)."""
    return np.random.default_rng([int(seed), int(part), int(row)])


def gold_share(n_gold: int, n_visible: int) -> float:
    return n_gold / n_visible if n_visible else 0.0


def summarize(vals: Sequence[float]) -> Dict[str, float]:
    v = np.asarray(vals, dtype=float)
    if v.size == 0:
        return {}
    return {"mean": float(v.mean()), "p10": float(np.percentile(v, 10)), "p50": float(np.median(v)),
            "p90": float(np.percentile(v, 90)), "min": float(v.min()), "max": float(v.max())}
