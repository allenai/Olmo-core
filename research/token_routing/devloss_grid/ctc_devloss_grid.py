"""
Dev-loss grid driver: answer-token CE of ONE frozen checkpoint under a list of token-selection
schemes, on the standard olmo-eval CTC suite rows (plus the ``cpt80`` continued-pretraining
pseudo-task). One GPU, one (task, rung) per invocation, one JSON out.

    python research/token_routing/devloss_grid/ctc_devloss_grid.py --task ctc_nq --rung 8k --rows 48 \\
        --ckpt /data/prasann/ctc_suite/vllm_serving_4b/ctc-4b-nq-full --ckpt-format hf \\
        --data-root /data/prasann/ctc_1m_ladders --schemes full,k0,first16,fl64 --out out.json

What is measured. Every row is rendered EXACTLY the way the suite's own native/vLLM eval driver
renders it (``olmo_core.data.document_chunk_landmark.segment_prompt_to_chunks`` -> the same
``build_prompt`` the olmo-eval branch vendors, ``query_position="both"``, Qwen chat template, each
document wrapped in ``<|doc_start|>``/``<|doc_end|>``), with the gold answer appended as the
assistant turn. The loss tokens are the assistant-turn tokens (teacher forced). Each scheme is a
different *input construction* for the same checkpoint (``Transformer._compact_pooled_soft_tokens``
driven through ``model._pooled_soft_tokens`` + ``model._pooled_keep_holder`` -- the training-side
code path, bit-for-bit), scored at the answer positions after mapping them into the compacted
sequence. ``full`` is the plain-causal reference on the uncompacted row.

See README.md in this folder for the scheme table, gold conventions and deviations.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time
import types
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
OLMO_EVAL_VENDOR = os.path.expanduser("~/projects/olmo-eval/src/olmo_eval/evals/tasks/ctc_suite/_vendor")

from olmo_core.data.document_chunk_landmark import RESERVED_IDS  # noqa: E402
from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens  # noqa: E402
from olmo_core.nn.attention.pooled_doc_kv import PooledDocKeepHolder, resolve_keep_docs  # noqa: E402
from olmo_core.nn.pooled_soft_token import (  # noqa: E402
    KeepTokenTables,
    build_slot_stop_ids,
    build_token_idf,
    build_token_piece_tables,
)

FAMILY = os.environ.get("PROBE_FAMILY", "qwen3_5")
VOCAB_BY_FAMILY = {"qwen3_5": 248320, "qwen3": 151936}
TOKENIZER_BY_FAMILY = {"qwen3_5": "Qwen/Qwen3.5-0.8B-Base", "qwen3": "Qwen/Qwen3-4B"}
IGN = -100

# --------------------------------------------------------------------------------------------
# The 22-row roster, copied from olmo-eval ``ctc_suite/__init__.py:ROSTER`` (branch
# prasann/ctc-suite). ``subset`` is the HF config name == local ladder directory; ``seg_task`` is
# the canonical segmentation task (``corpus_reasoning.eval.eval_lc_native_docchunk.TASK_CFG`` key,
# reached via its TASK_ALIASES) -- the name the marker scaffold + prompt builder dispatch on;
# ``spec`` is the vendored ctc spec that knows the gold convention.
# --------------------------------------------------------------------------------------------
ROSTER: Dict[str, Dict[str, Any]] = {
    "ctc_fiqa": dict(subset="fiqa", spec="retrieval", seg_task="retrieval"),
    "ctc_nq": dict(subset="nq", spec="retrieval", seg_task="retrieval"),
    "ctc_hpqa": dict(subset="hotpotqa", spec="retrieval", seg_task="retrieval"),
    "ctc_qdmatch_fiqa": dict(subset="qdmatch_fiqa", spec="qdmatch", seg_task="qdmatch"),
    "ctc_qdmatch_nq": dict(subset="qdmatch_nq", spec="qdmatch", seg_task="qdmatch"),
    "ctc_qdmatch_hpqa": dict(subset="qdmatch_hpqa", spec="qdmatch", seg_task="qdmatch"),
    "ctc_outlier_amzn": dict(subset="outlier_amzn", spec="outlier", seg_task="outlier"),
    "ctc_outlier": dict(subset="outlier", spec="outlier", seg_task="outlier"),
    "ctc_outlier_fixedm": dict(subset="outlier_fixedM", spec="outlier", seg_task="outlier"),
    "ctc_oolong": dict(subset="oolong", spec="oolong", seg_task="oolong"),
    "ctc_grouping": dict(subset="grouping", spec="grouping", seg_task="grouping"),
    "ctc_absence": dict(subset="absence_gutenberg", spec="absence", seg_task="absence"),
    "ctc_xabsence": dict(subset="xabsence", spec="xabsence", seg_task="xabsence"),
    "ctc_rerank": dict(subset="rerank", spec="rerank", seg_task="rerank"),
    "ctc_msmarco": dict(subset="msmarco", spec="retrieval", seg_task="retrieval"),
    "ctc_reorder": dict(subset="reorder", spec="reorder", seg_task="reorder"),
    "ctc_obliq": dict(subset="obliq_twitter", spec="retrieval", seg_task="retrieval"),
    "ctc_niah": dict(subset="niah", spec="retrieval", seg_task="retrieval"),
    "ctc_contradiction": dict(
        subset="contradiction_iid", spec="contradiction", seg_task="contradiction", rung_alias={"2k": 2560}
    ),
    "ctc_strmatch": dict(subset="strmatch", spec="strmatch", seg_task="strmatch"),
    "ctc_textgroups": dict(subset="textgroups", spec="textgroups", seg_task="textgroups"),
    "ctc_scifact": dict(subset="scifact", spec="retrieval", seg_task="retrieval"),
}
RUNG_TOKENS = {"2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768, "64k": 65536}

#: Segmentation config per canonical task: mirrors ``eval_lc_native_docchunk.TASK_CFG`` (chunk_by /
#: cot). Copied so the driver imports no eval module; ``check_task_cfg()`` asserts agreement.
SEG_CFG: Dict[str, Dict[str, str]] = {
    "retrieval": dict(chunk_by="document", cot="none"),
    "qdmatch": dict(chunk_by="document", cot="none"),
    "outlier": dict(chunk_by="document", cot="none"),
    "oolong": dict(chunk_by="line", cot="plan"),
    "grouping": dict(chunk_by="document", cot="none"),
    "absence": dict(chunk_by="document", cot="none"),
    "xabsence": dict(chunk_by="document", cot="none"),
    "rerank": dict(chunk_by="document", cot="none"),
    "reorder": dict(chunk_by="document", cot="none"),
    "contradiction": dict(chunk_by="document", cot="none"),
    "strmatch": dict(chunk_by="document", cot="none"),
    "textgroups": dict(chunk_by="document", cot="none"),
}

#: Gold-document convention per spec: (field, index base, kind). ``kind`` "flat" = list of doc
#: indices, "nested" = pairs/triples of doc indices, "pairs_shared" = qdmatch (query, doc) pairs
#: over the shared item index, None = the task has no gold SUBSET (every document is load-bearing:
#: reorder / oolong / grouping), so gold-forced schemes degrade to their gold-blind twin.
#: Values are what ``ctc/tasks/*/spec.py`` declares (``gold_index_base``, ``extra["gold_field"]``);
#: ``check_gold_conventions()`` asserts against the vendored registry when it is importable.
GOLD_CONVENTION: Dict[str, Optional[Tuple[str, int, str]]] = {
    "retrieval": ("gold_doc_indices", 0, "flat"),
    "qdmatch": ("gold_pairs", 1, "pairs_shared"),
    "outlier": ("gold_doc_indices", 0, "flat"),
    "oolong": None,
    "grouping": None,
    "absence": ("gold_doc_indices", 0, "flat"),
    "xabsence": ("gold_doc_indices", 0, "flat"),
    "rerank": ("gold_doc_indices", 0, "flat"),  # plus every CE-positive doc, see gold_docs()
    "reorder": None,
    "contradiction": ("gold_doc_indices", 1, "nested"),
    "strmatch": ("gold_doc_indices", 1, "nested"),
    "textgroups": ("gold_doc_indices", 1, "nested"),
}

# --------------------------------------------------------------------------------------------
# Schemes. Every entry is a construction for ``Transformer._compact_pooled_soft_tokens``:
#   keep      : "blind" (seeded random whole documents, keep_prob) | "gold" (gold docs real, plus
#               random whole docs up to ``frac`` of all docs)
#   slot      : "mean" | "cent_cmean"
#   extra     : first-K body tokens real per document (``--st-header-extra-tokens`` semantics)
#   rule/k    : keep_token_rule ("first_last" | "rule") with its K
#   weights   : keep_token_weights for rule="rule" (idf-only = pure IDF top-k)
# No header stop id anywhere: K counts from the document's first token (nq-style), which is the
# only header rule that is the same for every task.
# --------------------------------------------------------------------------------------------
SCHEMES: Dict[str, Optional[Dict[str, Any]]] = {
    "full": None,
    "k0": dict(keep="blind", keep_prob=0.0, slot="cent_cmean"),
    "first16": dict(keep="blind", keep_prob=0.0, slot="cent_cmean", extra=16),
    "first64": dict(keep="blind", keep_prob=0.0, slot="cent_cmean", extra=64),
    "fl16": dict(keep="blind", keep_prob=0.0, slot="cent_cmean", rule="first_last", k=16),
    "fl64": dict(keep="blind", keep_prob=0.0, slot="cent_cmean", rule="first_last", k=64),
    "idf16": dict(keep="blind", keep_prob=0.0, slot="cent_cmean", rule="rule", k=16, weights={"idf": 1.0}),
    "idf64": dict(keep="blind", keep_prob=0.0, slot="cent_cmean", rule="rule", k=64, weights={"idf": 1.0}),
    "rand33": dict(keep="blind", keep_prob=1.0 / 3.0, slot="mean"),
    "gold_rand33": dict(keep="gold", frac=1.0 / 3.0, slot="mean"),
    "gold_fl64": dict(keep="gold", frac=0.0, slot="cent_cmean", rule="first_last", k=64),
    # fractional budget (2026-09-21, Prasann): 20% of EVERY document body, first_last split -- compaction
    # controlled per document instead of a fixed 64 that keeps short suite documents entirely real
    "fl20": dict(keep="blind", keep_prob=0.0, slot="cent_cmean", rule="first_last", k=0.2),
    "gold_fl20": dict(keep="gold", frac=0.0, slot="cent_cmean", rule="first_last", k=0.2),
    # --- CPT screening schemes (2026-09-21, records/softdetach-cpt-plan.md): training-free strategies at
    # ~x0.25 compaction, ranked on dev loss before any 4B CPT run is spent ---
    # random-chunk pooling: 20% of documents stay whole, the rest collapse to ONE slot (k0 inside)
    "rand20": dict(keep="blind", keep_prob=0.2, slot="cent_cmean"),
    # first-K prefix per document at a K that is ~x0.25 on 512-token CPT blocks
    "first128": dict(keep="blind", keep_prob=0.0, slot="cent_cmean", extra=128),
    # cheap token-feature saliency (idf / position / first-sentence / caps / digits / piece length),
    # DEFAULT_KEEP_TOKEN_WEIGHTS -- the deployable proxy of the gradient oracle, 20% per document
    "rule20": dict(keep="blind", keep_prob=0.0, slot="cent_cmean", rule="rule", k=0.2, weights=None),
    # ORACLE: top-20% body tokens per document by ||d(answer CE)/d(embedding)|| from one backward on
    # the full row (uses the label; selection cost ~3 forward-equivalents, reported per scheme)
    "grad20": dict(keep="blind", keep_prob=0.0, slot="cent_cmean", rule="custom", k=0.2, sel="grad", sel_cost=3.0),
    # layer-3 attention budget allocator (records/outlier-saliency-preview-probe.md `attnrow`): the
    # attention mass the last 32 prompt positions put on each document sets that document's budget
    # (total = 20% of body tokens), spent on its FIRST tokens; needs a dense forward through the
    # first softmax layer (charged as a full forward here; 4/32 of one if truncated at layer 3)
    "attnrow20": dict(keep="blind", keep_prob=0.0, slot="cent_cmean", rule="custom", k=0.2, sel="attnrow", sel_cost=1.0),
    # + a fixed 8-token real prefix per document (uniform across tasks; keeps the `Document [N]:` id
    # that a pure 20% budget pools away on one-sentence documents, cf. niah) before the 20% first_last
    "fl20p8": dict(keep="blind", keep_prob=0.0, slot="cent_cmean", extra=8, rule="first_last", k=0.2),
    "gold_fl20p8": dict(keep="gold", frac=0.0, slot="cent_cmean", extra=8, rule="first_last", k=0.2),
    # gold docs whole, EVERY other document a bare slot (no fragments): separates "ids not visible"
    # from "fragment layout is harmful" on the one-sentence tasks (Prasann, 2026-09-21)
    "gold_k0": dict(keep="gold", frac=0.0, slot="cent_cmean"),
    # first 20% (contiguous prefix, fractional `first` rule) -- does the prefix alone carry the id,
    # without the extra 8? (Prasann, 2026-09-22)
    "first20": dict(keep="blind", keep_prob=0.0, slot="cent_cmean", rule="first", k=0.2),
    "gold_first20": dict(keep="gold", frac=0.0, slot="cent_cmean", rule="first", k=0.2),
    # gold_fl20p8 with the pooled remainder DROPPED (no slot token at all)
    "fl20p8_noslot": dict(keep="blind", keep_prob=0.0, slot="cent_cmean", extra=8, rule="first_last", k=0.2, noslot=True),
    "gold_fl20p8_noslot": dict(keep="gold", frac=0.0, slot="cent_cmean", extra=8, rule="first_last", k=0.2, noslot=True),
    # --- LAYER SKIPPING on top of gold_fl20p8 (Prasann, 2026-09-22): every column of a NON-KEPT
    # document (its kept fragment tokens AND its slot) bypasses the block at each listed layer --
    # the block output at that position is overwritten with the block input (residual copy).
    # Output-copy semantics: the block still runs on the full row, so other tokens see K/V (and the
    # GDN state) computed from the masked tokens' stale hidden states; the saving is reported as
    # ``layer_frac`` = masked columns x skipped layers / (columns x layers). Qwen3.5-4B stacks 32
    # blocks in the pattern gdn,gdn,gdn,attn (attention at 3,7,...,31):
    #   skipodd  = blocks 1,3,5,...,31  -> skips EVERY attention block (and half the GDN blocks)
    #   skipeven = blocks 0,2,4,...,30  -> GDN blocks only, attention never skipped
    #   skipgdn2 = every other GDN block in stack order (0,2,5,8,10,13,16,18,21,24,26,29)
    # gold docs whole + the 8-token id prefix of every other document + a RANDOM 20% of the rest of
    # its body (scattered, original positions; seeded per row), NO slot for the pooled remainder --
    # the random-position twin of gold_fl20p8_noslot at the same per-document budget (Prasann, 2026-09-22)
    "gold_rand20p8_noslot": dict(keep="gold", frac=0.0, slot="cent_cmean", extra=8, rule="custom", k=0.2, sel="random", sel_cost=0.0, noslot=True),
    "gold_fl20p8_skipodd": dict(keep="gold", frac=0.0, slot="cent_cmean", extra=8, rule="first_last", k=0.2, skip_layers=list(range(1, 32, 2))),
    "gold_fl20p8_skipeven": dict(keep="gold", frac=0.0, slot="cent_cmean", extra=8, rule="first_last", k=0.2, skip_layers=list(range(0, 32, 2))),
    "gold_fl20p8_skipgdn2": dict(keep="gold", frac=0.0, slot="cent_cmean", extra=8, rule="first_last", k=0.2, skip_layers=[i for j, i in enumerate([b for b in range(32) if (b + 1) % 4 != 0]) if j % 2 == 0]),
}
# --- LEARNED LINEAR TOKEN ROUTER (research/token_routing/learned_router/, 2026-09-23): per-task linear router over
# [RMS-normed input embedding, length-agnostic position features, gold-doc flag], trained with
# REINFORCE/RLOO on the frozen checkpoint. Slot-less (Prasann): routed = every document body token,
# gold docs included; doc-level keep = none; dropped tokens vanish, NO slot for any document;
# markers + everything outside documents always kept; original RoPE positions. `router` names the
# weights file ROUTER_WEIGHTS.format(task=<manifest key>, name=router); `sample` = seeded Bernoulli
# draw instead of the deterministic p > 0.5; `keep_markers=False` drops the markers of every
# partially kept document (the gold_*_noslot schemes' marker semantics; eval-only probe).
# gold docs whole, EVERY other document dropped outright (no fragments, no slot, no markers): the
# rule a gold-aware router can collapse to -- the reference point for the learned router (2026-09-23)
SCHEMES["gold_only_noslot"] = dict(keep="gold", frac=0.0, slot="cent_cmean", noslot=True)
# learned-router weights stay out of git: ROUTER_WEIGHTS_DIR (default = the original in-tree location)
ROUTER_WEIGHTS_DIR = os.environ.get(
    "ROUTER_WEIGHTS_DIR", "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/learned_router/weights"
)
ROUTER_WEIGHTS = os.path.join(ROUTER_WEIGHTS_DIR, "{task}", "{name}.pt")
for _lam in ("0.05", "0.2", "0.5", "1.0", "2.0"):
    for _var in ("", "nogold_", "noemb_"):
        for _samp in (False, True):
            SCHEMES[f"router_{_var}l{_lam}{'_samp' if _samp else ''}"] = dict(
                keep="none", slot="cent_cmean", rule="custom", k=0.0, sel="router", sel_cost=0.0, noslot=True,
                keep_markers=True, router=f"{_var}l{_lam}", sample=_samp)
SCHEMES["router_l0.2_nomark"] = dict(SCHEMES["router_l0.2"], keep_markers=False)
# DIFFERENTIABLE router (train_diff_router.py, 2026-09-27): same linear router and features, trained
# through a relaxed removal under a task-relative tolerance tau (mean dCE <= tau * mean CE_full, floor
# 0.005 nats); evaluated here with the SAME exact hard removal (deterministic gate logit > 0).
for _tau in ("0.05", "0.1", "0.2", "0.4"):
    SCHEMES[f"router_diff_tau{_tau}"] = dict(SCHEMES["router_l0.2"], router=f"diff_tau{_tau}")
# END-TO-END router (train_e2e_router.py, 2026-09-29): target-keep Lagrangian (CoFi-style), task loss
# CE + KL to the full model, no rule init; rho = target keep, tau = post-hoc val selection
for _r in ("0.05", "0.1", "0.2", "0.3", "0.5"):
    SCHEMES[f"router_e2e_rho{_r}"] = dict(SCHEMES["router_l0.2"], router=f"e2e_rho{_r}")
for _tau in ("0.05", "0.1", "0.2"):
    SCHEMES[f"router_e2e_tau{_tau}"] = dict(SCHEMES["router_l0.2"], router=f"e2e_tau{_tau}")
# PLAIN ATTENTION TOP-K (Prasann, 2026-09-29): gold-blind, slot-less; the top 10/20/30% of all
# document body tokens by layer-3 (first softmax layer) attention mass from the last 32 prompt
# positions, then a second compacted forward. Markers kept (router semantics). sel_cost = 4/32 of a
# full-row forward: the selection pass only needs blocks 0-3.
for _p in (10, 20, 30):
    SCHEMES[f"attntop{_p}_noslot"] = dict(keep="none", slot="cent_cmean", rule="custom", k=_p / 100.0, sel="attntop",
                                          sel_cost=4.0 / 32.0, noslot=True, keep_markers=True)
GOLD_SCHEME_TWIN = {"gold_rand33": "rand33", "gold_fl64": "fl64", "gold_fl20": "fl20", "gold_fl20p8": "fl20p8", "gold_k0": "k0", "gold_first20": "first20", "gold_fl20p8_noslot": "fl20p8_noslot"}
GOLD_SCHEME_TWIN["gold_only_noslot"] = "k0"  # no gold set: every document dropped (keep none, no slot)


def log(m: str) -> None:
    print(f"[devloss] {m}", flush=True)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------------------------
# Consistency checks against the shared code (run at startup; never silently diverge)
# --------------------------------------------------------------------------------------------
def check_task_cfg() -> None:
    try:
        _common = os.path.join(REPO, "research", "common")  # corpus_reasoning package (eval/, lib/)
        if _common not in sys.path:
            sys.path.insert(0, _common)
        from corpus_reasoning.eval.eval_lc_native_docchunk import TASK_CFG
    except Exception as e:  # noqa: BLE001
        log(f"TASK_CFG not importable ({e!r}); using the copied SEG_CFG unverified")
        return
    for t, c in SEG_CFG.items():
        ref = TASK_CFG[t]
        assert (ref["chunk_by"], ref["cot"]) == (c["chunk_by"], c["cot"]), (t, ref, c)
    log("SEG_CFG matches eval_lc_native_docchunk.TASK_CFG")


def check_gold_conventions() -> None:
    """Assert GOLD_CONVENTION against the vendored ctc registry (olmo-eval prasann/ctc-suite)."""
    if not os.path.isdir(OLMO_EVAL_VENDOR):
        log(f"vendored ctc not found at {OLMO_EVAL_VENDOR}; GOLD_CONVENTION unverified")
        return
    if OLMO_EVAL_VENDOR not in sys.path:
        sys.path.insert(0, OLMO_EVAL_VENDOR)
    try:
        from ctc.format import registry
        from ctc.tasks import load_all

        load_all()
    except Exception as e:  # noqa: BLE001
        log(f"vendored ctc import failed ({e!r}); GOLD_CONVENTION unverified")
        return
    for spec_name, conv in GOLD_CONVENTION.items():
        if spec_name == "grouping" and spec_name not in registry.names():
            continue  # registered locally by olmo-eval from the factory; base 0, no gold subset
        spec = registry.get(spec_name)
        field = spec.extra.get("gold_field", "gold_doc_indices")
        if conv is None:
            continue
        assert conv[0] == field, (spec_name, conv, field)
        assert conv[1] == spec.gold_index_base, (spec_name, conv, spec.gold_index_base)
    log("GOLD_CONVENTION matches the vendored ctc specs")


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------
def load_examples(data_root: str, row: Dict[str, Any], rung: str, n: int) -> List[dict]:
    """``<root>/<subset>/rung_<tokens>.jsonl`` (the olmo-eval CTC_SUITE_DATA_ROOT layout), or the
    HF mirror ``<root>/data/<subset>/r<rung>.parquet``."""
    tokens = row.get("rung_alias", {}).get(rung, RUNG_TOKENS[rung])
    jsonl = os.path.join(data_root, row["subset"], f"rung_{tokens}.jsonl")
    out: List[dict] = []
    if os.path.exists(jsonl):
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                if "ex" in ex and "documents" not in ex:
                    ex = ex["ex"]
                for mk in ("meta", "_meta"):  # the HF parquet export stores outlier/absence `meta` as a JSON string
                    if isinstance(ex.get(mk), str):
                        try:
                            ex[mk] = json.loads(ex[mk])
                        except ValueError:
                            pass
                out.append(ex)
                if len(out) >= n:
                    break
        log(f"loaded {len(out)} rows from {jsonl}")
        return out
    pq = os.path.join(data_root, "data", row["subset"], f"r{rung}.parquet")
    if os.path.exists(pq):
        import pandas as pd

        df = pd.read_parquet(pq).head(n)
        for rec in df.to_dict("records"):
            out.append({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in rec.items()})
        log(f"loaded {len(out)} rows from {pq}")
        return out
    raise SystemExit(f"no rung file: {jsonl} or {pq}")


def render_ctc_row(tok, example: dict, seg_task: str, ids) -> Tuple[List[int], List[bool], int]:
    """Marker-scaffolded token row + loss mask, exactly the suite's train/eval layout.

    :returns: ``(ids, mask, n_spans)`` -- ids end with the EOS the training shards append (mask
        False); ``n_spans`` = number of ``<|doc_start|>..<|doc_end|>`` spans found.
    """
    from olmo_core.data.document_chunk_landmark import find_chunk_spans, segment_prompt_to_chunks

    cfg = SEG_CFG[seg_task]
    _segs, row_ids, mask = segment_prompt_to_chunks(
        tok,
        example,
        seg_task,
        query_position="both",
        cot_mode=cfg["cot"],
        chunk_by=cfg["chunk_by"],
        item_regex=r"\|\|",
        include_answer=True,
        doc_start_id=ids.doc_start,
        doc_end_id=ids.doc_end,
    )
    row_ids = list(row_ids) + [ids.eos]
    mask = list(mask) + [False]
    return row_ids, mask, len(find_chunk_spans(row_ids, ids.doc_start, ids.doc_end))


def gold_docs(spec: str, example: dict) -> Optional[Set[int]]:
    """0-based positions (in ``documents`` order == chunk order) of the gold documents, or ``None``
    when the task has no gold subset."""
    conv = GOLD_CONVENTION[spec]
    if conv is None:
        return None
    field, base, kind = conv
    raw = example.get(field) or []
    out: Set[int] = set()

    def _walk(v):
        if isinstance(v, (list, tuple)):
            for u in v:
                _walk(u)
        elif v is not None:
            out.add(int(v) - base)

    _walk(raw)
    if spec == "rerank":
        for i, v in enumerate(example.get("ce_scores") or []):
            if v is not None and v > 0:
                out.add(i)
    n = len(example.get("documents") or [])
    return {g for g in out if 0 <= g < n}


# --------------------------------------------------------------------------------------------
# cpt80: continued-pretraining pseudo-task (amandab's setup): loss on the last 20% of a long
# document; the first 80% is split into 512-token pseudo-documents wrapped in doc markers.
# --------------------------------------------------------------------------------------------
def _iter_cpt_texts(source: str):
    paths = [source] if os.path.isfile(source) else sorted(
        glob.glob(os.path.join(source, "**", "*"), recursive=True)
    )
    n_files = 0
    for p in paths:
        if not os.path.isfile(p):
            continue
        if p.endswith((".jsonl", ".json", ".jsonl.gz", ".json.gz")):
            import gzip

            opener = gzip.open if p.endswith(".gz") else open
            with opener(p, "rt") as f:
                n_files += 1
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    t = d.get("text") if isinstance(d, dict) else None
                    if t:
                        yield t
        elif p.endswith((".parquet", ".arrow")):
            import pandas as pd

            n_files += 1
            df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_feather(p)
            col = "text" if "text" in df.columns else df.columns[0]
            for t in df[col].tolist():
                if t:
                    yield str(t)
        elif p.endswith(".txt"):
            n_files += 1
            yield open(p).read()
    if n_files == 0:
        # last resort: a saved HF dataset directory
        try:
            from datasets import load_from_disk

            ds = load_from_disk(source)
            for rec in ds:
                if rec.get("text"):
                    yield rec["text"]
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"cpt source {source}: no jsonl/parquet/txt files and not an HF dataset ({e!r})")


def render_cpt_rows(tok, source: str, rung: str, n: int, ids, block: int = 512, loss_frac: float = 0.2):
    L = RUNG_TOKENS[rung]
    rows, masks, scanned = [], [], 0
    for text in _iter_cpt_texts(source):
        scanned += 1
        enc = tok(text, add_special_tokens=False)["input_ids"]
        if len(enc) < L:
            continue
        enc = enc[:L]
        n_ctx = int(round(L * (1.0 - loss_frac)))
        ctx, tail = enc[:n_ctx], enc[n_ctx:]
        row: List[int] = []
        for s in range(0, len(ctx), block):
            row += [ids.doc_start] + ctx[s : s + block] + [ids.doc_end]
        # The first tail token is predicted FROM the last pseudo-doc's <|doc_end|>, which every
        # compaction scheme pools away (KeyError in the posmap, cpt80@2k 2026-09-21) -- so the loss
        # starts at tail[1] (predicted from the real tail[0]); one token of ~L/5 is dropped.
        mask = [False] * len(row) + [False] + [True] * (len(tail) - 1) + [False]
        row += tail + [ids.eos]
        rows.append(row)
        masks.append(mask)
        if len(rows) >= n:
            break
    log(f"cpt80: scanned {scanned} texts from {source}, kept {len(rows)} with >= {L} tokens "
        f"(block {block}, loss on the last {int(loss_frac * 100)}%)")
    if not rows:
        raise SystemExit("cpt80: no document long enough for this rung")
    return rows, masks


# --------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------
def build_cfg(vocab: int, attn_backend: str = "flash_2"):
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.lm_head import LMLossImplementation
    from olmo_core.nn.transformer import TransformerConfig

    # Never fall back silently: an unknown PROBE_FAMILY used to build qwen3_4B, which loads a
    # wrong-family checkpoint into the wrong architecture (or dies far from the cause).
    factories = {"qwen3_5": TransformerConfig.qwen3_5_4B, "qwen3": TransformerConfig.qwen3_4B}
    if FAMILY not in factories:
        raise SystemExit(f"PROBE_FAMILY={FAMILY!r} has no model config here (known: {sorted(factories)}); "
                         "for OLMo 3 use research/token_routing/olmo3_cptdrop/eval_devce.py")
    cfg = factories[FAMILY](vocab_size=vocab, attn_backend=AttentionBackendName(attn_backend))
    cfg.lm_head.loss_implementation = LMLossImplementation.default
    return cfg


def find_distcp(root: str) -> str:
    if root.endswith("model_and_optim") and os.path.isdir(root):
        return root
    cands = sorted(glob.glob(f"{root}/model_and_optim") + glob.glob(f"{root}/step*/model_and_optim"))
    if not cands:
        raise SystemExit(f"no model_and_optim under {root}")
    return cands[-1]


def load_hf_into(model, ckpt: str) -> None:
    """HF safetensors (text-only export OR a vLLM serving copy with ``model.language_model.*`` keys
    and a ``visual.*`` tower) -> olmo-core state, via ``convert_state_from_hf`` (Qwen3.5 path).
    Bypasses ``AutoModelForCausalLM`` on purpose: the env's transformers (4.57) has no
    ``qwen3_5_text`` class."""
    from safetensors.torch import load_file

    from olmo_core.nn.hf.convert import convert_state_from_hf

    raw = json.load(open(os.path.join(ckpt, "config.json")))
    text = raw.get("text_config", raw)
    mt = text.get("model_type", raw.get("model_type"))
    if mt != "qwen3_5_text":
        raise SystemExit(f"{ckpt}: model_type {mt!r}; this loader handles qwen3_5_text only")
    hf_state: Dict[str, torch.Tensor] = {}
    shards = sorted(glob.glob(os.path.join(ckpt, "*.safetensors")))
    if not shards:
        raise SystemExit(f"{ckpt}: no *.safetensors")
    for s in shards:
        hf_state.update(load_file(s))
    n_vis = sum(1 for k in hf_state if "visual" in k)
    if text.get("tie_word_embeddings") and not any(k.endswith("lm_head.weight") for k in hf_state):
        emb_key = [k for k in hf_state if k.endswith("embed_tokens.weight")][0]
        hf_state["lm_head.weight"] = hf_state[emb_key].clone()
    cfg_obj = types.SimpleNamespace(**text)
    converted = convert_state_from_hf(cfg_obj, hf_state, model_type="qwen3_5_text")
    del hf_state
    missing, unexpected = model.load_state_dict(converted, strict=False)
    missing = [k for k in missing if "rope" not in k and "inv_freq" not in k and "pooled_projector" not in k]
    if missing or unexpected:
        raise SystemExit(f"HF load mismatch: missing={missing[:8]} unexpected={list(unexpected)[:8]}")
    log(f"loaded HF export {ckpt}: {len(converted)} tensors ({n_vis} visual keys dropped)")


def load_model(ckpt: str, fmt: str, vocab: int, ids, seed: int, stop_ids: Sequence[int], attn_backend: str = "flash_2"):
    from olmo_core.distributed.checkpoint import load_model_and_optim_state

    cfg = build_cfg(vocab, attn_backend)
    model = cfg.build(init_device="cpu")
    t0 = time.time()
    if fmt == "hf":
        load_hf_into(model, ckpt)
    elif fmt == "olmo_st":  # one model-only safetensors file, olmo keys (gradprobe/resave_model_only.py)
        from safetensors.torch import load_file

        f = ckpt if ckpt.endswith(".safetensors") else os.path.join(ckpt, "model.safetensors")
        model.load_state_dict(load_file(f), strict=True)
        log(f"loaded olmo safetensors {f}")
    else:
        ck = find_distcp(ckpt)
        load_model_and_optim_state(ck, model)
        log(f"loaded distcp {ck}")
    log(f"checkpoint load took {time.time() - t0:.0f}s")
    attach_soft_tokens(model, ids, seed, stop_ids)
    model = model.cuda().to(torch.bfloat16)
    # every construction gives new sequence lengths; FLA re-autotunes per length (~20 s each). Speed
    # only, results bit-identical (memory fla-autotune-per-length-eval-slowdown). 2026-09-23.
    from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune

    log(f"froze length-derived autotune keys on {freeze_fla_length_autotune()} FLA kernel(s)")
    return model


def attach_soft_tokens(model, ids, seed: int, stop_ids: Sequence[int]) -> None:
    """Exactly ``trained_parity_check.py``'s call: detached slot K/V (and GDN writes), keep 0."""
    model.enable_pooled_soft_tokens(
        ids.doc_start,
        ids.doc_end,
        ids.eos,
        placeholder_id=ids.landmark,
        keep_prob=0.0,
        keep_seed=seed,
        detach_soft_kv=True,
        slot_mode="cent_cmean",
        slot_stop_ids=list(stop_ids),
    )
    wout = float(model.pooled_projector.w_out.weight.detach().abs().max())
    log(f"pooled_projector max|w_out| = {wout:.3e} (0 => identity slot, the detach_soft_kv default)")


def build_tables(all_ids: np.ndarray, vocab: int, pieces: Sequence[Optional[str]], ids, decode) -> Tuple[List[int], List[str], KeepTokenTables]:
    """Stop set (for ``cent_cmean``) and IDF/piece tables (for ``rule``) from the rows being scored."""
    stop, shown = build_slot_stop_ids(
        all_ids, top_k=100, extra_ids=(ids.doc_start, ids.doc_end, ids.eos, ids.landmark, ids.pad), decode=decode
    )
    idf = build_token_idf(all_ids, vocab)
    sent_end, is_cap, is_dig, tok_len = build_token_piece_tables(pieces)
    tables = KeepTokenTables(
        idf=torch.tensor(idf),
        sent_end=torch.tensor(sent_end),
        is_cap=torch.tensor(is_cap),
        is_dig=torch.tensor(is_dig),
        tok_len=torch.tensor(tok_len),
    )
    return stop, shown, tables


# --------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------
def gold_keep_mask(n_docs: int, gold: Optional[Set[int]], frac: float, seed: int, row_idx: int) -> torch.Tensor:
    """(1, n_docs) bool: gold docs + seeded random non-gold docs up to ``frac`` of all docs."""
    keep = torch.zeros(1, n_docs, dtype=torch.bool)
    g = sorted(x for x in (gold or ()) if x < n_docs)
    keep[0, g] = True
    target = int(round(frac * n_docs))
    need = max(0, target - len(g))
    if need:
        rng = np.random.default_rng(seed * 1000003 + row_idx)
        pool = np.array([d for d in range(n_docs) if d not in set(g)])
        pick = rng.choice(pool, size=min(need, len(pool)), replace=False)
        keep[0, torch.as_tensor(pick, dtype=torch.long)] = True
    return keep


class LayerSkip:
    """Residual-copy layer skipping for a subset of compacted columns, via block hooks.

    ``install(skip)`` registers a forward-pre-hook (stash the block input) and a forward hook
    (overwrite masked positions of the output with the input) on ``model.blocks[str(i)]`` for every
    ``i`` in ``skip`` that exists. ``mask`` is a ``(T2,)`` bool tensor on the model device, set per
    row; ``None`` disables. ``remove()`` unregisters. Output-copy semantics (see SCHEMES)."""

    def __init__(self, model):
        self.model = model
        self.handles: list = []
        self.mask: Optional[torch.Tensor] = None
        self._h: Dict[int, torch.Tensor] = {}
        self.layers: List[int] = []

    def install(self, skip: Sequence[int]) -> None:
        self.remove()
        n = len(self.model.blocks)
        self.layers = [int(i) for i in skip if int(i) < n]
        for i in self.layers:
            blk = self.model.blocks[str(i)]

            def pre(mod, args, _i=i):
                self._h[_i] = args[0]

            def post(mod, args, out, _i=i):
                if self.mask is None:
                    return out
                h_in = self._h.pop(_i)
                m = self.mask.view(1, -1, 1)
                if m.shape[1] != out.shape[1]:
                    raise RuntimeError(f"layer-skip mask length {m.shape[1]} != block sequence length {out.shape[1]}")
                return torch.where(m, h_in.to(out.dtype), out)

            self.handles.append(blk.register_forward_pre_hook(pre))
            self.handles.append(blk.register_forward_hook(post))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles, self.layers, self._h = [], [], {}
        self.mask = None


def skip_mask_for(cb, cid0: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """(T2,) bool: True for every compacted column that belongs to a document NOT kept whole
    (fragment tokens and slot columns alike -- a slot sits at its document's centre position, so
    the chunk id at ``position_ids`` resolves it to its document). Query/answer/instruction columns
    (chunk id < 0) and kept documents are False."""
    pos = cb.position_ids[0].cpu().long()
    doc = cid0[0].long()[pos]
    kept = keep[0].bool()
    in_doc = doc >= 0
    m = torch.zeros_like(in_doc)
    m[in_doc] = ~kept[doc[in_doc]]
    return m


def configure_scheme(model, pst: Dict[str, Any], scheme: Dict[str, Any], tables: KeepTokenTables, stop_ids: Sequence[int]) -> None:
    pst["header_stop_id"] = None
    pst["header_stop_count"] = 1
    pst["header_cap"] = 32
    pst["header_extra_tokens"] = int(scheme.get("extra", 0))
    pst["keep_token_rule"] = scheme.get("rule", "none")
    pst["keep_token_k"] = float(scheme.get("k", 0))
    pst["keep_token_weights"] = scheme.get("weights")
    pst["keep_token_sd"] = None
    pst["keep_token_tables"] = tables
    pst["keep_token_log_every"] = 0
    pst["slot_mode"] = scheme["slot"]
    pst["slot_stop_ids"] = list(stop_ids)
    pst["slot_stop_mask"] = None
    pst["keep_prob"] = float(scheme.get("keep_prob", 0.0))
    pst["drop_slots"] = bool(scheme.get("noslot", False))
    pst["keep_token_mask_markers"] = bool(scheme.get("keep_markers", False))



class AttnCapture:
    """Attention mass received by every key position from a chosen set of query positions, for one
    softmax layer, re-computed from post-RoPE q/k in a forward PRE-hook on the attention backend
    (ported from research/token_routing/pooled_kv/outlier_probe/outlier_saliency_preview_probe.py)."""

    def __init__(self, model):
        self.handles, self.layers = [], []
        self.state = {"on": False, "qsel": None, "acc": {}, "only": None}
        for key, block in model.blocks.items():
            attn = getattr(block, "attention", None)
            if attn is None or not hasattr(attn, "backend"):
                continue
            self.layers.append(int(key))
            self.handles.append(attn.backend.register_forward_pre_hook(self._make(int(key)), with_kwargs=True))

    def _make(self, li):
        def hook(mod, args, kwargs):
            st = self.state
            if not st["on"] or (st["only"] is not None and li not in st["only"]):
                return None
            qkv = kwargs.get("qkv", args[0] if args else None)
            if not isinstance(qkv, (tuple, list)) or len(qkv) != 3:
                return None
            q, k = qkv[0], qkv[1]
            if q is None or k is None or q.dim() != 4 or q.shape[0] != 1:
                return None
            qs = st["qsel"]
            T = k.shape[1]
            if qs is None or int(qs.max()) >= q.shape[1]:
                return None
            n_rep = max(1, q.shape[2] // k.shape[2])
            kk = k if n_rep == 1 else k.repeat_interleave(n_rep, dim=2)
            scale = getattr(mod, "scale", None) or float(q.shape[-1]) ** -0.5
            sc = torch.einsum("qhd,thd->hqt", q[0, qs], kk[0]).float() * float(scale)
            causal = torch.arange(T, device=sc.device)[None, :] > qs[:, None]
            sc = sc.masked_fill(causal[None], float("-inf"))
            st["acc"][li] = st["acc"].get(li, 0.0) + sc.softmax(-1).sum(dim=(0, 1)).detach().float()
            return None
        return hook

    def run(self, model, x, qsel, only=None):
        self.state.update(on=True, qsel=qsel, acc={}, only=only)
        model.eval()
        model._pooled_keep_holder = None
        model(x, logits_to_keep=1)
        self.state["on"] = False
        return dict(self.state["acc"])


def grad_saliency(model, x, pred_pos, targets) -> torch.Tensor:
    """(S,) ||d CE(answer) / d embedding_output|| from ONE backward on the full row (oracle: uses the
    label). Parameters do not require grad, so only activation gradients are materialised."""
    store = {}

    def hook(mod, inp, out):
        out = out.detach().requires_grad_(True)
        store["h"] = out
        return out

    hd = model.embeddings.register_forward_hook(hook)
    model.eval()
    model._pooled_keep_holder = None
    try:
        with torch.enable_grad():
            lg = model(x, logits_to_keep=pred_pos[None])[0].float()
            F.cross_entropy(lg, targets).backward()
    except (torch.OutOfMemoryError, RuntimeError) as e:  # Triton reports CUDA OOM as RuntimeError
        if not isinstance(e, torch.OutOfMemoryError) and "out of memory" not in str(e).lower():
            raise
        # a full-row backward does not fit (32k rows on one GPU): the caller falls back to first-k
        hd.remove()
        store.clear()
        torch.cuda.empty_cache()
        return None
    finally:
        hd.remove()
    g = store["h"].grad
    out = g[0].float().norm(dim=-1).detach()
    store.clear()
    torch.cuda.empty_cache()
    return out


_ROUTER_CACHE: Dict[str, Any] = {}


def router_keep_mask(model, x, cid0, n_docs, ids, gold, scheme: Dict[str, Any], task_key: str, rng_seed: int):
    """(1, S) bool body-token keep mask from the learned linear router (``research/token_routing/learned_router``);
    returns (mask, keep fraction of routed tokens)."""
    rdir = os.path.join(REPO, "research", "token_routing", "learned_router")
    if rdir not in sys.path:
        sys.path.insert(0, rdir)
    from router_lib import LinearRouter, keep_mask_from, rms_embed, routed_features

    path = ROUTER_WEIGHTS.format(task=task_key, name=scheme["router"])
    if path not in _ROUTER_CACHE:
        if not os.path.exists(path):
            raise SystemExit(f"router weights missing: {path}")
        st = torch.load(path, map_location="cpu")
        r_ = LinearRouter.from_state(st).to(x.device)
        r_.keep_rule, r_.rho = st.get("keep_rule", "p>0.5"), st.get("rho")
        _ROUTER_CACHE[path] = r_
        log(f"router: loaded {path} (keep rule {r_.keep_rule}{'' if r_.rho is None else f', rho {r_.rho}'})")
    router = _ROUTER_CACHE[path]
    feats = routed_features(x[0], cid0[0], ids.doc_start, ids.doc_end, n_docs, sorted(gold) if gold else None,
                            route_markers=getattr(router, "route_markers", False))
    e = rms_embed(model.embeddings.weight, feats["tok"]) if router.use_emb else None
    z = router.logits(feats, e)
    p = torch.sigmoid(z)
    N = int(z.numel())
    k = None
    if scheme.get("topk_pair"):
        # PAIRED budget (2026-09-30): keep as many routed tokens on this row as gold_fl20p8_noslot keeps on
        # it (its realised per-row T2/T, not the mean): rows where the bar keeps more (bigger / more gold
        # documents) get the same larger budget
        from check_hand_fl import heuristic_real

        real = heuristic_real(x[0].cpu(), cid0[0].cpu(), gold, ids)
        k = int(real[feats["idx"].cpu()].sum())
        if getattr(router, "marker_follow", False):  # markers follow their doc: the budget counts them too
            k += int(real[feats["mk_idx"].cpu()].sum())
    elif scheme.get("topk_comp") is not None:
        # per-row budget that makes T2/T hit the target: T2 = (T - N non-routed, always kept) + k
        T = x.shape[1]
        k = int(round(float(scheme["topk_comp"]) * T - (T - N)))
        if getattr(router, "marker_follow", False):  # markers are not always kept: they are paid from the budget
            k += int(feats["mk_idx"].numel())
    elif scheme.get("topk_rho") is not None or getattr(router, "keep_rule", "p>0.5") == "topk":
        rho = float(scheme["topk_rho"]) if scheme.get("topk_rho") is not None else float(router.rho)
        k = int(math.ceil(rho * N))
    if k is not None:
        # span routers select whole units (spans / markers), rounding the budget to the nearest span
        from router_lib import topk_keep as _topk_keep

        if getattr(router, "marker_follow", False):
            from router_lib import follow_keep_budget as _fkb

            keep = _fkb(z, feats, k, "whole" if router.marker_follow == "whole" else "any")
        else:
            keep = _topk_keep(z, feats, k, getattr(router, "span", 0))
    elif scheme.get("sample"):
        gen = torch.Generator().manual_seed(int(rng_seed))
        keep = torch.rand(p.shape, generator=gen).to(p.device) < p
    else:
        keep = p > 0.5
    return keep_mask_from(feats, keep, x.shape[1])[None], float(keep.float().mean()) if keep.numel() else 1.0


def custom_keep_mask(model, x, cid0, n_docs, ids, sel: str, frac: float, pred_pos, targets, capture,
                     skip_first: int = 0, rng_seed: int = 0) -> Tuple[torch.Tensor, Optional[str]]:
    """(1, S) bool of body tokens to keep real under a saliency selector. Returns (mask, warning).

    ``sel="random"``: per document, a uniformly random ``ceil(frac * n)`` of the body tokens AFTER
    the first ``skip_first`` (the header prefix ``mark_doc_headers_free`` frees first), with
    ``n`` = that remainder -- the same budget ``mark_doc_topk_tokens_free`` gives first_last on top
    of the same prefix. Seeded by ``rng_seed`` so a row's selection is reproducible."""
    S = x.shape[1]
    cid = cid0[0].to(x.device)
    markers = (x[0] == ids.doc_start) | (x[0] == ids.doc_end)
    body = (cid >= 0) & ~markers
    warn = None
    if sel == "random":
        gen = torch.Generator().manual_seed(int(rng_seed))
        mask = torch.zeros(S, dtype=torch.bool, device=x.device)
        for d in range(n_docs):
            pos = torch.nonzero(body & (cid == d)).flatten()[skip_first:]
            if pos.numel() == 0:
                continue
            kd = min(int(pos.numel()), max(1, int(math.ceil(frac * pos.numel()))))
            pick = torch.randperm(int(pos.numel()), generator=gen)[:kd].to(pos.device)
            mask[pos[pick]] = True
        return mask[None], warn
    if sel == "grad":
        sal = grad_saliency(model, x, pred_pos, targets)
        if sal is None:
            warn = "grad: backward OOM on this row length -> first-k fallback (cell NOT an oracle)"
        # per-document top ceil(frac*|body_d|) by saliency, at original positions
        mask = torch.zeros(S, dtype=torch.bool, device=x.device)
        for d in range(n_docs):
            pos = torch.nonzero(body & (cid == d)).flatten()
            if pos.numel() == 0:
                continue
            kd = max(1, int(math.ceil(frac * pos.numel())))
            top = pos[torch.topk(sal[pos], kd).indices] if sal is not None else pos[:kd]
            mask[top] = True
        return mask[None], warn
    if sel == "attnrow":
        ans_start = int(pred_pos[0]) + 1
        qsel = torch.arange(max(0, ans_start - 32), ans_start, device=x.device)
        first_attn = capture.layers[0] if capture.layers else None
        acc = capture.run(model, x, qsel, only={first_attn} if first_attn is not None else None)
        mass = acc.get(first_attn) if first_attn is not None else None
        mask = torch.zeros(S, dtype=torch.bool, device=x.device)
        body_pos = [torch.nonzero(body & (cid == d)).flatten() for d in range(n_docs)]
        total = sum(int(p.numel()) for p in body_pos)
        B = int(round(frac * total))
        if mass is None or B <= 0:
            warn = "attnrow: no attention mass captured -> uniform first-k fallback"
            budgets = [max(1, int(math.ceil(frac * p.numel()))) if p.numel() else 0 for p in body_pos]
        else:
            m = torch.stack([mass[p].sum() if p.numel() else mass.new_zeros(()) for p in body_pos]).double()
            m = m / m.sum() if float(m.sum()) > 0 else torch.full_like(m, 1.0 / max(1, n_docs))
            raw = m * B
            budgets = [min(int(p.numel()), int(r)) for p, r in zip(body_pos, raw.floor().tolist())]
            # largest-remainder rounding so the total budget is spent
            rem = B - sum(budgets)
            order = torch.argsort(-(raw - raw.floor()))
            for d in order.tolist():
                if rem <= 0:
                    break
                if budgets[d] < body_pos[d].numel():
                    budgets[d] += 1
                    rem -= 1
        for p, b in zip(body_pos, budgets):
            if b > 0:
                mask[p[:b]] = True
        return mask[None], warn
    if sel == "attntop":
        # plain token-level attention ranking: the top ``frac`` of ALL document body tokens (pooled
        # across documents, no per-document budget) by the layer-3 attention mass the last 32 prompt
        # positions put on them; the caller then runs a second, compacted forward on the survivors
        ans_start = int(pred_pos[0]) + 1
        qsel = torch.arange(max(0, ans_start - 32), ans_start, device=x.device)
        first_attn = capture.layers[0] if capture.layers else None
        acc = capture.run(model, x, qsel, only={first_attn} if first_attn is not None else None)
        mass = acc.get(first_attn) if first_attn is not None else None
        mask = torch.zeros(S, dtype=torch.bool, device=x.device)
        pos = torch.nonzero(body).flatten()
        if pos.numel() == 0:
            return mask[None], warn
        B = min(int(pos.numel()), max(1, int(round(frac * pos.numel()))))
        if mass is None:
            warn = "attntop: no attention mass captured -> first-k fallback"
            mask[pos[:B]] = True
        else:
            mask[pos[torch.topk(mass[pos], B).indices]] = True
        return mask[None], warn
    raise ValueError(sel)


@torch.no_grad()
def score_rows(model, rows, masks, golds, schemes: Dict[str, Optional[Dict[str, Any]]], ids, tables, stop_ids, seed: int, pieces_decode, task_key: str = ""):
    """Per-scheme, per-row answer CE (and CE on digit tokens, top-1 agreement / KL vs full,
    compaction ratio). ``golds[i]`` = gold doc set or None."""
    pst = model._pooled_soft_tokens
    dev = next(model.parameters()).device
    tables = tables.to(dev)
    acc: Dict[str, Dict[str, list]] = {
        s: {"ce": [], "ce_digit": [], "top1": [], "kl": [], "compaction": [], "kept_docs": [], "n_docs": [], "sec": [], "sel_cost": [], "layer_frac": [], "route_keep": []}
        for s in schemes
    }
    skipper = LayerSkip(model) if any((sch or {}).get("skip_layers") for sch in schemes.values()) else None
    degenerate = {s: [] for s in schemes}
    capture = AttnCapture(model) if any((sch or {}).get("sel") in ("attnrow", "attntop") for sch in schemes.values()) else None
    warned: set = set()
    max_pos = max(len(r) for r in rows) + 8
    n_warm = 0
    for mod in model.modules():
        rope = getattr(mod, "rope", None)
        if rope is not None and hasattr(rope, "warmup_cache"):
            rope.warmup_cache(max_pos, dev)
            n_warm += 1
    log(f"warmed {n_warm} RoPE caches to {max_pos} positions")
    t_start = time.time()
    for ri, (row, rmask) in enumerate(zip(rows, masks)):
        x = torch.tensor(np.asarray(row)[None], device=dev)
        ans_pos = torch.tensor(np.nonzero(np.asarray(rmask))[0], device=dev)
        pred_pos = ans_pos - 1
        targets = x[0, ans_pos]
        digit_sel = torch.tensor(
            [i for i, t in enumerate(targets.tolist()) if any(ch.isdigit() for ch in pieces_decode(t))],
            device=dev,
            dtype=torch.long,
        )
        cid0 = build_chunk_ids_from_tokens(x.cpu(), doc_start_id=ids.doc_start, doc_end_id=ids.doc_end, eos_id=ids.eos, mode="chunked")
        n_docs = int(cid0.max()) + 1
        full_lg = None
        for name, sch in schemes.items():
            t0 = time.time()
            r = acc[name]
            layer_frac = 0.0
            route_keep = float("nan")
            if sch is None:
                model.eval()
                model._pooled_keep_holder = None
                lg = model(x, logits_to_keep=pred_pos[None])[0].float()
                full_lg = lg
                comp, kept = 1.0, n_docs
            else:
                if n_docs <= 0:
                    # nothing to compact (no documents in the row): the construction IS full
                    lg = full_lg
                    comp, kept = 1.0, 0
                else:
                    configure_scheme(model, pst, sch, tables, stop_ids)
                    if sch.get("sel") == "router":
                        mask, route_keep = router_keep_mask(model, x, cid0, n_docs, ids, golds[ri], sch, task_key,
                                                            rng_seed=seed * 100003 + ri)
                        pst["keep_token_mask"] = mask
                        _r = _ROUTER_CACHE.get(ROUTER_WEIGHTS.format(task=task_key, name=sch["router"]))
                        if getattr(_r, "route_markers", False):
                            pst["keep_token_mask_markers"] = "mask"  # routed markers: kept iff the router keeps them
                        elif getattr(_r, "marker_follow", False):
                            # markers follow their document (any kept body token), or only a WHOLE kept body
                            pst["keep_token_mask_markers"] = False if _r.marker_follow == "whole" else "doc"
                    elif sch.get("rule") == "custom":
                        mask, warn = custom_keep_mask(model, x, cid0, n_docs, ids, sch["sel"], float(sch["k"]), pred_pos, targets, capture,
                                                      skip_first=int(sch.get("extra", 0)), rng_seed=seed * 100003 + ri)
                        pst["keep_token_mask"] = mask
                        if warn and warn not in warned:
                            warned.add(warn)
                            log(f"WARNING {name}: {warn}")
                    if sch["keep"] == "none":
                        keep = torch.zeros(1, n_docs, dtype=torch.bool)  # token mask decides everything
                    elif sch["keep"] == "blind":
                        keep = resolve_keep_docs(cid0, n_docs, holder=None, keep_prob=float(sch["keep_prob"]), keep_seed=seed).cpu()
                    else:
                        g = golds[ri]
                        if g is None:
                            degenerate[name].append(ri)
                            twin = schemes.get(GOLD_SCHEME_TWIN.get(name, ""), None)
                            keep = resolve_keep_docs(cid0, n_docs, holder=None, keep_prob=float(twin["keep_prob"]) if twin else 0.0, keep_seed=seed).cpu()
                        else:
                            keep = gold_keep_mask(n_docs, g, float(sch.get("frac", 0.0)), seed, ri)
                    model.train()
                    model._pooled_keep_holder = PooledDocKeepHolder(keep_docs=keep.clone())
                    cb = model._compact_pooled_soft_tokens(x, None, IGN)[0]
                    posmap = {int(p): c for c, p in enumerate(cb.position_ids[0].tolist())}
                    cols = torch.tensor([posmap[int(p)] for p in pred_pos.tolist()], device=dev)
                    assert int(cols.max()) < cb.input_ids.shape[1], "compaction/column mismatch"
                    layer_frac = 0.0
                    if sch.get("skip_layers") is not None:
                        skipper.install(sch["skip_layers"])
                        smask = skip_mask_for(cb, cid0, keep)
                        skipper.mask = smask.to(dev)
                        layer_frac = float(smask.float().mean()) * len(skipper.layers) / len(model.blocks)
                    try:
                        lg = model(x, logits_to_keep=cols[None])[0].float()
                    finally:
                        if sch.get("skip_layers") is not None:
                            skipper.remove()
                    comp = cb.input_ids.shape[1] / x.shape[1]
                    kept = int(keep[0, :n_docs].sum())
                    model.eval()
                    model._pooled_keep_holder = None
            r["ce"].append(float(F.cross_entropy(lg, targets)))
            r["ce_digit"].append(float(F.cross_entropy(lg[digit_sel], targets[digit_sel])) if digit_sel.numel() else float("nan"))
            r["top1"].append(float((lg.argmax(-1) == full_lg.argmax(-1)).float().mean()))
            r["kl"].append(float(F.kl_div(F.log_softmax(lg, -1), F.log_softmax(full_lg, -1), log_target=True, reduction="batchmean")))
            r["compaction"].append(comp)
            r["kept_docs"].append(kept)
            r["n_docs"].append(n_docs)
            r["sel_cost"].append(float((sch or {}).get("sel_cost", 0.0)))
            r["layer_frac"].append(layer_frac)
            r["route_keep"].append(route_keep)
            r["sec"].append(time.time() - t0)
        done = ri + 1
        if done in (1, 2, 5, 10) or done % 10 == 0 or done == len(rows):
            el = time.time() - t_start
            eta = el / done * (len(rows) - done)
            summ = "  ".join(f"{s}={np.mean(acc[s]['ce']):.3f}@{np.mean(acc[s]['compaction']):.2f}" for s in schemes)
            log(f"row {done}/{len(rows)}  T={x.shape[1]} docs={n_docs}  {el:.0f}s elapsed, ETA {eta:.0f}s | {summ}")
    acc['_warnings'] = sorted(warned)
    return acc, degenerate


def summarize(r: Dict[str, list]) -> Dict[str, float]:
    out = {}
    for k, v in r.items():
        a = np.asarray(v, dtype=np.float64)
        a = a[~np.isnan(a)]
        out[k] = float(a.mean()) if a.size else float("nan")
        out[f"{k}_se"] = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else float("nan")
    out["n"] = len(r["ce"])
    return out


# --------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="roster row (ctc_nq ...) or cpt80")
    ap.add_argument("--rung", default="8k", help="one rung, or a comma list scored in ONE process (model loaded once)")
    ap.add_argument("--rows", default="48", help="rows per rung: one int, or a comma list aligned with --rung")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-format", default="hf", choices=["hf", "distcp", "olmo_st"])
    ap.add_argument("--data-root", default=os.environ.get("CTC_SUITE_DATA_ROOT", ""))
    ap.add_argument("--cpt-source", default=None, help="cpt80: file/dir of jsonl|parquet|txt rows with a 'text' field (else manifest.json cpt_source)")
    ap.add_argument("--schemes", default=",".join(SCHEMES))
    ap.add_argument("--tokenizer", default=os.environ.get("DEVLOSS_TOKENIZER", TOKENIZER_BY_FAMILY[FAMILY]))
    ap.add_argument("--out", required=True, help="output JSON; with several rungs it must contain {rung}")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpt-block", type=int, default=512)
    ap.add_argument("--attn-backend", default="flash_2", choices=["torch", "flash_2"],
                    help="softmax-attention backend; flash_2 is what the ds64 soft-token arms trained with")
    a = ap.parse_args()
    rungs = [r for r in a.rung.split(",") if r]
    for r in rungs:
        if r not in RUNG_TOKENS:
            raise SystemExit(f"unknown rung {r}")
    nrows = [int(x) for x in str(a.rows).split(",")]
    nrows = nrows * len(rungs) if len(nrows) == 1 else nrows
    if len(nrows) != len(rungs) or (len(rungs) > 1 and "{rung}" not in a.out):
        raise SystemExit("--rows must align with --rung, and --out must contain {rung} for several rungs")
    t_main = time.time()
    try:
        import psutil

        log(f"process start -> main: {t_main - psutil.Process().create_time():.1f}s (imports)")
    except Exception:  # noqa: BLE001
        pass
    if os.environ.get("DEVLOSS_SKIP_CHECKS") != "1":  # both import heavy eval packages; verified at every grid run
        check_task_cfg()
        check_gold_conventions()
        log(f"consistency checks done ({time.time() - t_main:.1f}s)")
    ids = RESERVED_IDS[FAMILY]
    vocab = VOCAB_BY_FAMILY[FAMILY]
    # any `router_<name>` resolves to the weights file $ROUTER_WEIGHTS_DIR/<task>/<name>.pt
    # (same slot-less, markers-kept, deterministic-gate semantics as the named router schemes)
    # `router_<name>@k<rho>` keeps the top ceil(rho*N) routed tokens by logit; `router_<name>@c<x>` picks the
    # per-row budget so that T2/T = x (matched compaction); a router saved with keep_rule "topk" defaults to
    # top-k at its training rho
    def _router_scheme(s):
        name, _, spec = s[len("router_"):].partition("@")
        extra = {}
        if spec.startswith("k"):
            extra["topk_rho"] = float(spec[1:])
        elif spec.startswith("c"):
            extra["topk_comp"] = float(spec[1:])
        elif spec == "pair":
            extra["topk_pair"] = True
        return dict(SCHEMES["router_l0.2"], router=name, **extra)

    schemes = {s: SCHEMES[s] if s in SCHEMES else _router_scheme(s)
               for s in a.schemes.split(",") if s and (s in SCHEMES or s.startswith("router_"))}
    if "full" not in schemes:
        schemes = {"full": None, **schemes}

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    pieces = tok.convert_ids_to_tokens(list(range(vocab)))
    decode = lambda t: tok.decode([int(t)])  # noqa: E731
    log(f"tokenizer loaded ({time.time() - t_main:.1f}s)")
    model = None
    for rung, n_rows in zip(rungs, nrows):
        a.rung, a.rows = rung, n_rows
        model = run_cell(a, model, tok, pieces, decode, ids, vocab, schemes, t_main,
                         a.out.replace("{rung}", rung))


def _cell_argv(argv, rung, rows, out_path):
    """argv as if this cell had been run on its own (collect_grid.py parses --rows from it)."""
    out, skip = [], False
    for i, x in enumerate(argv):
        if skip:
            skip = False
            continue
        if x in ("--rung", "--rows", "--out") and i + 1 < len(argv):
            out += [x, {"--rung": rung, "--rows": str(rows), "--out": out_path}[x]]
            skip = True
        else:
            out.append(x)
    return out


def run_cell(a, model, tok, pieces, decode, ids, vocab, schemes, t_main, out_path):
    """Score one (task, rung) cell; loads the model on first use and returns it for the next rung."""
    t0 = time.time()
    golds: List[Optional[Set[int]]]
    if a.task == "cpt80":
        src = a.cpt_source
        if src is None:
            mf = os.path.join(_HERE, "manifest.json")
            src = json.load(open(mf)).get("cpt_source") if os.path.exists(mf) else None
            if isinstance(src, dict):  # manifest.json stores {path, format, origin}
                src = src.get("path")
        if not src:
            raise SystemExit("cpt80 needs --cpt-source or manifest.json:cpt_source")
        rows, masks = render_cpt_rows(tok, src, a.rung, a.rows, ids, block=a.cpt_block)
        golds = [None] * len(rows)
        meta: Dict[str, Any] = {"cpt_source": src, "cpt_block": a.cpt_block}
    else:
        row = ROSTER[a.task]
        if not a.data_root:
            raise SystemExit("--data-root (or CTC_SUITE_DATA_ROOT) is required")
        examples = load_examples(a.data_root, row, a.rung, a.rows)
        rows, masks, golds, n_bad = [], [], [], 0
        for ex in examples:
            r_ids, r_mask, n_spans = render_ctc_row(tok, ex, row["seg_task"], ids)
            n_docs = len(ex.get("documents") or [])
            if SEG_CFG[row["seg_task"]]["chunk_by"] == "document" and n_spans != n_docs:
                # A document with EMPTY text (fiqa r2k row 29, doc 0) renders without markers, so the
                # gold->chunk map would be off by one for every later document: DROP the row.
                n_bad += 1
                if n_bad <= 3:
                    log(f"WARNING: {n_spans} marker spans for {n_docs} documents -- row dropped (empty document text?)")
                continue
            rows.append(r_ids)
            masks.append(r_mask)
            golds.append(gold_docs(row["spec"], ex))
        meta = {"subset": row["subset"], "spec": row["spec"], "seg_task": row["seg_task"], "span_count_mismatch_rows_dropped": n_bad}
    lens = [len(r) for r in rows]
    log(f"rendered {len(rows)} rows in {time.time() - t0:.0f}s; lengths min/median/max = "
        f"{min(lens)}/{int(np.median(lens))}/{max(lens)}; answer tokens median = {int(np.median([sum(m) for m in masks]))}")
    n_gold = [len(g) for g in golds if g is not None]
    log(f"gold docs per row: {'none (gold schemes degrade to their gold-blind twin)' if not n_gold else f'median {int(np.median(n_gold))}'}")

    all_ids = np.concatenate([np.asarray(r, dtype=np.int64) for r in rows])
    stop_ids, shown, tables = build_tables(all_ids, vocab, pieces, ids, decode)
    log(f"tables built ({time.time() - t_main:.1f}s)")
    log(f"slot stop set: {len(stop_ids)} ids from {all_ids.size} tokens of the scored rows; top dropped: {' '.join(shown[:12])}")

    if model is None:  # later rungs reuse it; configure_scheme resets the stop set per scheme
        model = load_model(a.ckpt, a.ckpt_format, vocab, ids, a.seed, stop_ids, a.attn_backend)
    task_key = a.task[4:] if a.task.startswith("ctc_") else a.task
    log(f"model ready ({time.time() - t_main:.1f}s)")
    acc, degenerate = score_rows(model, rows, masks, golds, schemes, ids, tables, stop_ids, a.seed, decode, task_key=task_key)

    out = {
        "task": a.task,
        "rung": a.rung,
        "eval_size": len(rows),
        "ckpt": a.ckpt,
        "ckpt_format": a.ckpt_format,
        "seed": a.seed,
        "tokenizer": a.tokenizer,
        "git_commit": git_commit(),
        "argv": _cell_argv(sys.argv, a.rung, a.rows, out_path),
        "meta": meta,
        "schemes": {s: (sch or {}) for s, sch in schemes.items()},
        "gold_degenerate_rows": {s: v for s, v in degenerate.items() if v},
        "summary": {s: summarize(acc[s]) for s in schemes},
        "warnings": acc.get("_warnings", []),
        "per_row": {s: acc[s] for s in schemes},
        "row_lens": lens,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"\n=== {a.task} @ {a.rung}  eval_size={len(rows)}  ckpt={os.path.basename(a.ckpt)} ===", flush=True)
    print(f"{'scheme':12} {'CE':>7} {'±se':>6} {'CEdig':>7} {'top1':>6} {'KL':>7} {'compact':>8} {'kept':>5}", flush=True)
    for s in schemes:
        m = out["summary"][s]
        print(f"{s:12} {m['ce']:7.3f} {m['ce_se']:6.3f} {m['ce_digit']:7.3f} {m['top1']:6.3f} {m['kl']:7.3f} {m['compaction']:8.3f} {m['kept_docs']:5.1f}", flush=True)
    log(f"wrote {out_path} (total {time.time() - t_main:.1f}s)")
    return model


if __name__ == "__main__":
    main()
