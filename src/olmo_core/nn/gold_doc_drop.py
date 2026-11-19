"""Gold-aware WHOLE-DOCUMENT drop before block 0 (loop idea ``goldrand``).

Applied to each training forward's padded rows before the soft-token compaction (so before the
online router and before block 0): the dropped documents' tokens (markers included) are REMOVED
from the row and the row is left-shifted and right-padded with EOS (labels ``ignore_index``), so
no block, the LM head or the router ever sees them and the FLOP meter charges the shorter row
(its ``dense`` length stays the original row, recorded before this runs).

Per row (looked up by the content fingerprint in the online router's gold table):

* ``whole_frac`` of the rows are left untouched (no drop, and the router's gold prior is off for
  them: the inference-like distribution);
* rows whose gold share of the documents is at most ``max_gold_frac`` (contradiction / outlier:
  a few gold documents) keep EVERY gold document plus a per-row random fraction
  ``r ~ U(keep_lo, keep_hi)`` of the distractors, so the kept set is never just the gold set.
  The answer stays valid: those tasks name documents by the display id written inside each
  document, which is kept with it, and every gold document is kept;
* the other rows (reorder / qdmatch_nq: nearly every document is gold, dropping one would change
  the answer) keep every document; with ``thin_lo/thin_hi`` set, a per-row random fraction
  ``q ~ U(thin_lo, thin_hi)`` of each document's body tokens past its first ``thin_keep_first``
  is kept (uniform draw), the rest removed.

The new row's fingerprint is registered in the router's gold table with the gold indices
renumbered over the kept documents, so the router's gold feature / prior keeps working, and a
whole row's fingerprint is registered in ``router.noprior_fps``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class GoldDocDropConfig:
    keep_lo: float = 1.0
    keep_hi: float = 1.0
    max_gold_frac: float = 0.5
    whole_frac: float = 0.0
    #: skew of the per-row distractor keep share: r = keep_lo + (keep_hi - keep_lo) * u**keep_power, u ~ U(0,1).
    #: 1.0 = uniform (unchanged); > 1 biases toward keep_lo (more rows dropped hard), < 1 toward keep_hi
    keep_power: float = 1.0
    thin_lo: float = 1.0
    thin_hi: float = 1.0
    thin_keep_first: int = 8
    seed: int = 0
    log_every: int = 20
    #: optional JSON list of row content fingerprints; when set, ONLY those rows get the document drop / thinning
    #: (e.g. a pool's ``gdd_only_contradiction.json``), every other row is passed through whole
    only_fps_path: str = ""
    #: optional JSON ``{row fingerprint: [group id per document]}`` (or ``{"groups": {...}}``); when set, a dropped row
    #: keeps every gold document plus a share r of the NON-gold GROUPS (whole groups, at least ``min_groups``) instead
    #: of a share r of individual distractors (label-valid drop for outlier: each kept topic keeps all its passages).
    #: A row mapped to ``null`` (or a list of the wrong length) is passed through whole (label cannot be kept valid).
    #: Rows absent from the sidecar get the ordinary per-document drop unless ``groups_absent_whole`` (then whole).
    #: Empty = the per-document drop above (unchanged).
    groups_path: str = ""
    min_groups: int = 2
    groups_absent_whole: bool = False


class GoldDocDrop:
    def __init__(self, cfg: GoldDocDropConfig, *, router: Any, doc_start_id: int, doc_end_id: int, eos_id: int,
                 rank: int = 0):
        if router is None or getattr(router, "gold", None) is None:
            raise ValueError("gold doc drop needs the online router with a gold table (gold_fingerprints.json)")
        self.cfg = cfg
        self.router = router
        if not hasattr(router, "noprior_fps"):
            router.noprior_fps = set()
        self.ds, self.de, self.eos = int(doc_start_id), int(doc_end_id), int(eos_id)
        self.rng = random.Random(cfg.seed * 1_000_003 + rank * 7919 + 17)
        self.calls = 0
        self.st: Dict[str, float] = {}
        self.only_fps = None
        if cfg.only_fps_path:
            import json

            with open(cfg.only_fps_path) as f:
                self.only_fps = set(json.load(f))
        self.groups = None
        if cfg.groups_path:
            import json

            with open(cfg.groups_path) as f:
                g = json.load(f)
            self.groups = g.get("groups", g) if isinstance(g, dict) else g

    def _acc(self, k: str, v: float) -> None:
        self.st[k] = self.st.get(k, 0.0) + v

    @torch.no_grad()
    def apply(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor], ignore_index: int
              ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        from .attention.gold_grad_mask import content_fingerprint_from_row

        c = self.cfg
        self.calls += 1
        B, S = input_ids.shape
        ids_l = input_ids.detach().to("cpu").tolist()
        lab_l = labels.detach().to("cpu").tolist() if labels is not None else None
        gold_tab = self.router.gold
        out_ids: List[List[int]] = []
        out_lab: List[List[int]] = []
        changed = False
        for b in range(B):
            row = ids_l[b]
            lab = lab_l[b] if lab_l is not None else None
            fp = content_fingerprint_from_row(row, self.eos)
            gold = gold_tab.get(fp)
            # document spans (start marker .. end marker inclusive), in order = chunk index
            spans: List[Tuple[int, int]] = []
            s0 = -1
            for i, t in enumerate(row):
                if t == self.ds:
                    s0 = i
                elif t == self.de and s0 >= 0:
                    spans.append((s0, i))
                    s0 = -1
            nd = len(spans)
            self._acc("rows", 1)
            self._acc("tok_in", float(sum(e - s + 1 for s, e in spans)))
            if self.only_fps is not None and fp not in self.only_fps and nd > 0:
                self._acc("not_selected", 1)
                out_ids.append(row), out_lab.append(lab)
                self._acc("tok_out", float(sum(e - s + 1 for s, e in spans)))
                continue
            if gold is None or nd == 0:
                self._acc("miss", 1)
                out_ids.append(row), out_lab.append(lab)
                self._acc("tok_out", float(sum(e - s + 1 for s, e in spans)))
                continue
            grp = None
            if self.groups is not None:
                listed = fp in self.groups
                grp = self.groups.get(fp)
                if listed and grp is not None and len(grp) == nd:
                    self._acc("grouped", 1)
                elif listed or c.groups_absent_whole:
                    self._acc("nogroup", 1)
                    out_ids.append(row), out_lab.append(lab)
                    self._acc("tok_out", float(sum(e - s + 1 for s, e in spans)))
                    continue
            if self.rng.random() < c.whole_frac:
                self.router.noprior_fps.add(fp)
                self._acc("whole", 1)
                out_ids.append(row), out_lab.append(lab)
                self._acc("tok_out", float(sum(e - s + 1 for s, e in spans)))
                continue
            gset = {g for g in gold if 0 <= g < nd}
            remove = [False] * S
            new_gold: List[int] = list(sorted(gset))
            if len(gset) / nd <= c.max_gold_frac:
                if c.keep_power == 1.0:
                    r = self.rng.uniform(c.keep_lo, c.keep_hi)
                else:
                    r = c.keep_lo + (c.keep_hi - c.keep_lo) * self.rng.random() ** c.keep_power
                if grp is not None:
                    ggrp = {grp[g] for g in gset}
                    topics = sorted({grp[d] for d in range(nd)} - ggrp)
                    self.rng.shuffle(topics)
                    n_t = min(len(topics), max(c.min_groups, int(round(r * len(topics)))))
                    kt = set(topics[:n_t])
                    keep_docs = gset | {d for d in range(nd) if grp[d] in kt}
                    self._acc("groups_in", len(topics))
                    self._acc("groups_out", n_t)
                else:
                    ng = [d for d in range(nd) if d not in gset]
                    self.rng.shuffle(ng)
                    n_keep = int(round(r * len(ng)))
                    keep_docs = gset | set(ng[:n_keep])
                for d, (s, e) in enumerate(spans):
                    if d not in keep_docs:
                        for i in range(s, e + 1):
                            remove[i] = True
                kept_sorted = sorted(keep_docs)
                remap = {d: j for j, d in enumerate(kept_sorted)}
                new_gold = sorted(remap[g] for g in gset)
                self._acc("dropped_rows", 1)
                self._acc("docs_in", nd)
                self._acc("docs_out", len(keep_docs))
            elif c.thin_lo < 1.0 or c.thin_hi < 1.0:
                q = self.rng.uniform(c.thin_lo, c.thin_hi)
                for (s, e) in spans:
                    for i in range(s + 1 + c.thin_keep_first, e):
                        if self.rng.random() >= q:
                            remove[i] = True
                self._acc("thin_rows", 1)
            else:
                self._acc("full_rows", 1)
            n_rm = sum(remove)
            self._acc("tok_out", float(sum(e - s + 1 for s, e in spans) - n_rm))
            if n_rm == 0:
                out_ids.append(row), out_lab.append(lab)
                continue
            changed = True
            nr = [t for t, rm in zip(row, remove) if not rm] + [self.eos] * n_rm
            out_ids.append(nr)
            if lab is not None:
                out_lab.append([t for t, rm in zip(lab, remove) if not rm] + [ignore_index] * n_rm)
            gold_tab[content_fingerprint_from_row(nr, self.eos)] = new_gold
        if c.log_every and self.calls % c.log_every == 1 and self.router.rank == 0:
            s = self.st
            print(f"[gold-doc-drop] call {self.calls}: rows {s.get('rows', 0):.0f} miss {s.get('miss', 0):.0f} "
                  f"whole {s.get('whole', 0):.0f} dropped {s.get('dropped_rows', 0):.0f} "
                  f"(docs {s.get('docs_out', 0):.0f}/{s.get('docs_in', 0):.0f}) thin {s.get('thin_rows', 0):.0f} "
                  f"full {s.get('full_rows', 0):.0f} not-selected {s.get('not_selected', 0):.0f}"
                  + (f" nogroup {s.get('nogroup', 0):.0f} grouped {s.get('grouped', 0):.0f} (groups {s.get('groups_out', 0):.0f}/{s.get('groups_in', 0):.0f})"
                     if self.groups is not None else "")
                  + f"; doc-token keep {s.get('tok_out', 0) / max(1.0, s.get('tok_in', 1)):.3f}",
                  flush=True)
        if not changed:
            return input_ids, labels
        new_ids = torch.tensor(out_ids, dtype=input_ids.dtype, device=input_ids.device)
        new_lab = (torch.tensor(out_lab, dtype=labels.dtype, device=labels.device) if labels is not None else None)
        return new_ids, new_lab

    def metrics(self) -> Dict[str, float]:
        s = self.st
        return {"gold_doc_drop/doc_token_keep": s.get("tok_out", 0) / max(1.0, s.get("tok_in", 1)),
                "gold_doc_drop/miss_frac": s.get("miss", 0) / max(1.0, s.get("rows", 1))}
