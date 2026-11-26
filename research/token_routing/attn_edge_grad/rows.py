"""Held-out rows -> token segments (preamble / one per document / question / answer), gold documents, optional
gold-document drop (as in training: every gold document + a random share r ~ U(lo, hi) of the distractors, dropped
documents removed and the row left-shifted)."""
from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np

DOC_START, DOC_END = 248049, 248050
DIGIT0 = 15  # Qwen3.5: '0'..'9' are the single-digit tokens 15..24
PRE, GOLD, DIST, QUES, ANS = 0, 1, 2, 3, 4
CLS_NAMES = ("pre", "gold", "dist", "question", "answer")


def load_set(npz: str, set_name: str, task: Optional[str], n: int, offset: int = 0) -> List[Dict]:
    z = np.load(npz)
    ids, mask, off, sets, tasks = z["ids"], z["mask"], z["offsets"], z["sets"], z["tasks"]
    out = []
    for r in range(len(sets)):
        if str(sets[r]) != set_name or (task and str(tasks[r]) != task):
            continue
        out.append(dict(row=r, set=set_name, task=str(tasks[r]), ids=ids[off[r]:off[r + 1]].astype(np.int64),
                        lmask=mask[off[r]:off[r + 1]].astype(bool)))
    return out[offset:offset + n]


def digit_runs(ids) -> List[int]:
    out, cur = [], ""
    for t in ids.tolist():
        if DIGIT0 <= t < DIGIT0 + 10:
            cur += str(t - DIGIT0)
        elif cur:
            out.append(int(cur))
            cur = ""
    if cur:
        out.append(int(cur))
    return out


def doc_spans(ids):
    ds = np.flatnonzero(ids == DOC_START)
    de = np.flatnonzero(ids == DOC_END)
    assert len(ds) == len(de) and (de > ds).all(), "unbalanced document markers"
    return ds, de


def gold_docs(ids, lmask, ds, de) -> List[int]:
    ans = set(digit_runs(ids[lmask]))
    g = []
    for d in range(len(ds)):
        r = digit_runs(ids[ds[d]:min(de[d], ds[d] + 16)])
        if r and r[0] in ans:
            g.append(d)
    return g


def gold_drop(row: Dict, lo: float, hi: float, seed: int) -> Dict:
    ids, lm = row["ids"], row["lmask"]
    ds, de = doc_spans(ids)
    gold = set(gold_docs(ids, lm, ds, de))
    rng = random.Random(seed)
    r = rng.uniform(lo, hi)
    dist = [d for d in range(len(ds)) if d not in gold]
    keep_d = set(rng.sample(dist, max(1, round(r * len(dist))))) | gold
    keep = np.ones(len(ids), dtype=bool)
    for d in range(len(ds)):
        if d not in keep_d:
            end = ds[d + 1] if d + 1 < len(ds) else de[d] + 1
            keep[ds[d]:end] = False
    out = dict(row)
    out.update(ids=ids[keep], lmask=lm[keep], drop_r=r, orig_T=len(ids), kept_docs=len(keep_d), orig_docs=len(ds))
    return out


def segments(row: Dict) -> Dict:
    """seg (T,) int64: 0 = preamble, 1..nd = documents (the separator after a document belongs to it), nd+1 =
    question (after the last document, before the answer), nd+2 = answer (from the last prompt position, which
    predicts the first answer token, to the end). cls (nseg,) = PRE/GOLD/DIST/QUES/ANS."""
    ids, lm = row["ids"], row["lmask"]
    T = len(ids)
    ds, de = doc_spans(ids)
    nd = len(ds)
    gold = gold_docs(ids, lm, ds, de)
    first_lab = int(np.flatnonzero(lm)[0])
    a0 = first_lab - 1
    seg = np.zeros(T, dtype=np.int64)
    for d in range(nd):
        end = ds[d + 1] if d + 1 < nd else de[d] + 1
        seg[ds[d]:end] = d + 1
    assert a0 > de[-1], "answer must follow the last document"
    seg[de[-1] + 1:a0] = nd + 1
    seg[a0:] = nd + 2
    cls = np.full(nd + 3, DIST, dtype=np.int64)
    cls[0], cls[nd + 1], cls[nd + 2] = PRE, QUES, ANS
    for g in gold:
        cls[g + 1] = GOLD
    nums = [(digit_runs(ids[ds[d]:min(de[d], ds[d] + 16)]) or [-1])[0] for d in range(nd)]
    return dict(seg=seg, cls=cls, nseg=nd + 3, nd=nd, gold=gold, T=T, numbered_ok=nums == list(range(1, nd + 1)),
                seg_len=np.bincount(seg, minlength=nd + 3))


def block_edges(seg_len: np.ndarray) -> np.ndarray:
    """causal edges per (query segment, key segment) block (segments are contiguous and ordered)."""
    L = seg_len.astype(np.float64)
    E = np.tril(np.outer(L, L), -1)
    E[np.diag_indices_from(E)] = L * (L + 1) / 2
    return E


# key classes for the category table (query class x key class, with same-document distinctions)
KEY_NAMES = ("pre", "gold_same", "gold_other", "dist_same", "dist_other", "question", "answer")


def category_index(cls: np.ndarray) -> np.ndarray:
    """(nseg, nseg) category id = qcls * 7 + kcls_ext (kcls_ext over KEY_NAMES)."""
    n = len(cls)
    kc = np.zeros((n, n), dtype=np.int64)
    same = np.eye(n, dtype=bool)
    kcl = np.broadcast_to(cls[None, :], (n, n))
    kc[kcl == PRE] = 0
    kc[(kcl == GOLD) & same] = 1
    kc[(kcl == GOLD) & ~same] = 2
    kc[(kcl == DIST) & same] = 3
    kc[(kcl == DIST) & ~same] = 4
    kc[kcl == QUES] = 5
    kc[kcl == ANS] = 6
    return cls[:, None] * len(KEY_NAMES) + kc
