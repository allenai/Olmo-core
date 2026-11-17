"""
Long-range position structure for cheap training rows (training forwards only; idea tag ``longpos``).

Two opt-in transforms of a padded one-example-per-row training batch, applied by
:meth:`Transformer.forward` before anything else sees the row (eval / probe forwards are untouched):

* **position stretching** (``stretch_max > 0``): a SHORT row (real length <= ``stretch_short_max``)
  keeps its tokens but gets ``position_ids`` spanning ``[0, S)`` with ``S ~ U[L, stretch_max]``: the
  prompt prefix stays at 0.., and the ``S - L`` extra positions are split at random (uniform
  stick-breaking) into gaps before each document and one gap after the last document, so documents
  sit at random offsets of a long row while query and answer stay contiguous. FLOPs unchanged.
* **position-preserving document subsampling** (``docsub_keep >= 0``): a LONG row (real length >=
  ``docsub_min_len``) drops a random subset of its NON-gold documents at the input (gold from the
  fingerprint sidecar; every gold document kept; a share ``r ~ U[keep/2, min(1, 1.5 keep)]`` of the
  non-gold documents kept). Surviving tokens keep their ORIGINAL position ids, so the row still spans
  its full length. Rows whose fingerprint misses the sidecar stay whole; reorder rows (every
  document gold) stay whole by construction.

Returns ``(input_ids, labels, position_ids)``; ``position_ids`` is ``None`` when no row changed.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import torch

from .attention.gold_grad_mask import content_fingerprint


def _doc_spans(row: List[int], ds: int, de: int) -> List[Tuple[int, int]]:
    spans, open_at = [], None
    for i, t in enumerate(row):
        if t == ds:
            open_at = i
        elif t == de and open_at is not None:
            spans.append((open_at, i))
            open_at = None
    return spans


def apply_longpos(
    input_ids: torch.Tensor, labels: Optional[torch.Tensor], cfg: Dict[str, Any], ignore_index: int
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    eos, ds, de = int(cfg["eos_id"]), int(cfg["doc_start_id"]), int(cfg["doc_end_id"])
    max_pos = int(cfg["max_pos"])
    cnt = cfg["_calls"] = cfg.get("_calls", 0) + 1
    rng = random.Random(f"longpos:{cfg['seed']}:{cfg['rank']}:{cnt}")
    st = cfg.setdefault("_stats", {"rows": 0, "stretched": 0, "span_sum": 0.0, "docsub": 0,
                                   "docsub_miss": 0, "tok_in": 0, "tok_out": 0, "docs_in": 0, "docs_out": 0})
    B, T = input_ids.shape
    ids_cpu = input_ids.detach().cpu()
    lab_cpu = labels.detach().cpu() if labels is not None else None
    out_ids, out_lab, out_pos = [], [], []
    changed = False
    for b in range(B):
        row = ids_cpu[b].tolist()
        try:
            L = row.index(eos, 1) + 1
        except ValueError:
            L = T
        st["rows"] += 1
        keep_cols = list(range(L))
        pos = list(range(L))
        spans = _doc_spans(row[:L], ds, de)
        if cfg["docsub_keep"] >= 0 and L >= cfg["docsub_min_len"] and spans:
            gold = cfg["gold_table"].get(content_fingerprint(row[:L]))
            if gold is None:
                st["docsub_miss"] += 1
            else:
                gold_set = {int(g) for g in gold}
                non = [i for i in range(len(spans)) if i not in gold_set]
                k = float(cfg["docsub_keep"])
                r = rng.uniform(k / 2.0, min(1.0, 1.5 * k))
                kept_non = set(rng.sample(non, int(round(r * len(non))))) if non else set()
                drop = [spans[i] for i in range(len(spans)) if i not in gold_set and i not in kept_non]
                if drop:
                    dmask = [False] * L
                    for s, e in drop:
                        for j in range(s, e + 1):
                            dmask[j] = True
                    keep_cols = [j for j in range(L) if not dmask[j]]
                    pos = keep_cols  # original positions
                    changed = True
                st["docsub"] += 1
                st["tok_in"] += L
                st["tok_out"] += len(keep_cols)
                st["docs_in"] += len(spans)
                st["docs_out"] += len(spans) - len(drop)
        elif cfg["stretch_max"] > 0 and L <= cfg["stretch_short_max"] and spans and rng.random() < cfg["stretch_prob"]:
            S = rng.randint(L, int(cfg["stretch_max"]))
            G = S - L
            w = [rng.random() for _ in range(len(spans) + 1)]
            tot = sum(w) or 1.0
            gaps = [int(G * x / tot) for x in w]
            inc = [0] * (L + 1)
            for i, (s, _e) in enumerate(spans):
                inc[s] += gaps[i]
            inc[spans[-1][1] + 1] += gaps[-1]
            acc, pos = 0, []
            for j in range(L):
                acc += inc[j]
                pos.append(j + acc)
            changed = True
            st["stretched"] += 1
            st["span_sum"] += pos[-1] + 1
        out_ids.append([row[j] for j in keep_cols])
        if lab_cpu is not None:
            lr = lab_cpu[b].tolist()
            out_lab.append([lr[j] for j in keep_cols])
        out_pos.append(pos)
    if cfg.get("log_every", 0) and (cnt in (1, 2, 5, 10) or cnt % cfg["log_every"] == 0) and cfg["rank"] == 0:
        msg = f"[longpos] call {cnt}: rows={st['rows']} stretched={st['stretched']}"
        if st["stretched"]:
            msg += f" mean_span={st['span_sum'] / st['stretched']:.0f}"
        if st["docsub"] or st["docsub_miss"]:
            msg += (f" docsub={st['docsub']} miss={st['docsub_miss']} tok_kept={st['tok_out'] / max(1, st['tok_in']):.3f}"
                    f" docs_kept={st['docs_out'] / max(1, st['docs_in']):.3f}")
        print(msg, flush=True)
    if not changed:
        return input_ids, labels, None
    W = max(len(r) for r in out_ids)
    W = min(T, ((W + 63) // 64) * 64)
    ids_t = torch.full((B, W), eos, dtype=input_ids.dtype)
    pos_t = torch.zeros((B, W), dtype=torch.long)
    lab_t = torch.full((B, W), ignore_index, dtype=labels.dtype) if labels is not None else None
    for b in range(B):
        n = len(out_ids[b])
        ids_t[b, :n] = torch.tensor(out_ids[b], dtype=input_ids.dtype)
        p = torch.tensor(out_pos[b], dtype=torch.long)
        pos_t[b, :n] = p
        if W > n:  # pad: after every real token, no label
            pos_t[b, n:] = torch.clamp(torch.arange(W - n) + int(p[-1]) + 1, max=max_pos - 1)
        if lab_t is not None:
            lab_t[b, :n] = torch.tensor(out_lab[b], dtype=labels.dtype)
    dev = input_ids.device
    return ids_t.to(dev), (lab_t.to(dev) if lab_t is not None else None), pos_t.to(dev)
