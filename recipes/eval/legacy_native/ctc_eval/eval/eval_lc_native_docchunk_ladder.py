"""
NATIVE olmo_core eval for **document-chunked** Qwen3-4B models across the FULL long-context LADDER,
emitting a flat-keyed ``<task>_<rung>`` JSON (parsed directly by ``viz/collect_headline.py``).

This is the docchunk analogue of the standard native ladder eval ``ctc_eval/eval/eval_lc_native.py``:
it reuses the box-marker chunked prefill + bs=1 KV-cached greedy decode machinery of the single-task
doc-chunk evals (``eval_lc_native_docchunk.py``), but loops every task over its length ladder using the
**SAME rung -> eval-JSONL mapping + graders as ``eval_lc_native.py``** (v1/v2 LADDERS + LSPEC), so the
resulting JSON is directly comparable to the dense/landmark ladder JSONs.

Tasks (9): the 5 in-distribution v2 tasks
    contradiction, nq, oolong, rerank, outlier
plus the 4 OOD generalization ladders (graded by REUSING the in-distribution graders):
    fiqa           = retrieval-f1  (held-out BEIR)
    scifact        = retrieval-f1  (held-out BEIR)
    outlier_review = outlier-f1    (Amazon-review passages, different source than wiki100w)
    contra_fever   = contradiction-f1 (FEVER claims, different source than pubmed)

Per-task chunk granularity (matches the training converter):
    contradiction / contra_fever = (task=contradiction, chunk_by=document)
    nq / fiqa / scifact          = (task=retrieval,     chunk_by=document)
    outlier / outlier_review     = (task=outlier,       chunk_by=document)
    oolong                       = (task=oolong,        chunk_by=line, item_regex='\\|\\|')
    rerank                       = (task=rerank,        chunk_by=document)
``cot_mode='none'`` (NO-CoT) throughout.

Plus 6 CTC-suite tasks (see ``SUITE_TASKS``): hotpotqa, qdmatch_nq, xabsence, reorder, strmatch,
grouping -- rendered like the setA converter and graded by the vendored ctc spec exactly as olmo-eval
grades the suite row. Their rung files come from ``DC_RUNG_FILES`` or ``CTC_SUITE_ROWS_ROOT``.
``--ctc-grade`` (env ``LADDER_CTC_GRADE``) grades nq / outlier / oolong / contradiction / rerank the
same way (``CTC_GRADED_BASE``), for scoring those five on the suite's own rows (rerank: the HF
``rerank`` subset, primary ``ce_pos_recall`` averaged over CE-scorable rows as olmo-eval does).
reorder also reports ``kendall_tau_tolerant`` / ``pairwise_acc_tolerant`` (parse-tolerant partial
credit, :func:`reorder_tolerant_scores`) beside the unchanged strict ``kendall_tau``.
``--repetition-stop id|ngram`` (env ``LADDER_REPETITION_STOP``; default off) stops a generation at a
repeated id / token n-gram; it changes what is scored, so it is off unless asked for.

The rung ladder files resolve against ``EVAL500_ROOT`` (env), exactly like ``eval_lc_native.py`` -- on
Beaker the runner sets ``EVAL500_ROOT=$PRASANNS/_eval_bundle_eval500_v2`` (v2) so ``{E5}/<sub>/<file>``
lands on the weka bundle. The relative base-rung / v1-oolong files resolve via ``--root`` (chdir to the
weka $BUNDLE). *** KEEP build_ladders()/LSPEC IN SYNC WITH eval_lc_native.py. ***

Variants:
  * ``dense``        -> DocumentChunkedAttention, dense emitter (box markers, no landmark tokens).
  * ``hierarchical`` -> loaded EXACTLY like dense (config.json drives the attention class).
  * ``full``         -> plain full attention baseline (no chunked mask; box markers ignored).
  * ``landmark``     -> DocumentLandmarkAttention, landmark emitter (a landmark token every --mem-freq).

Decode is the bs=1 greedy KV-cached loop from the single-task scripts. To stay inside a contended
Beaker wall-clock budget, SPLIT with ``--tasks`` and ``--rungs`` (the launcher fans out one Beaker job
per task); every invocation MERGES its ``<task>_<rung>`` keys into ``--out``.

    PYTHONPATH=<repo>/recipes/eval/legacy_native:<repo>/src torchrun --nproc_per_node=8 \
      recipes/eval/legacy_native/ctc_eval/eval/eval_lc_native_docchunk_ladder.py \
      --variant dense --model-path <step_dir> --ladder-version v2 \
      --out outputs/eval_results/<name>_ladder.json
"""

import argparse
import json
import math
import os
import re
import sys
import time

# torch is imported inside main() (2026-10-25), so the grading helpers below (ctc_score_one,
# make_ctc_score_fn, reorder_tolerant_scores, RepetitionStop) import torch-free: CPU tests and
# research/token_routing/ds64_fast8k/rescore_saved_generations.py load this module on any node.

# Reserved ids. DEFAULTS are the Qwen3 family; main() overwrites every one of them from
# olmo_core.data.document_chunk_landmark.RESERVED_IDS[family] (+ the tokenizer's <|im_end|>) once
# the family is known -- see resolve_reserved_ids(). Before 2026-09-02 they were hardcoded, so a
# Qwen3.5 checkpoint (markers 248049/248050, eos 248044) was prompted with Qwen3's ids: documents
# wrapped in arbitrary vocab tokens and generation never hit a stop id (fs35 KV evals: f1 0.18).
EOS_TOKEN_ID = 151643
# The SFT model ends its assistant turn with <|im_end|> (chat template), and 151643 is appended by the
# converter UNSUPERVISED (not in the loss mask) -- so the model reliably emits 151645 but often never
# emits 151643. Stopping ONLY on 151643 made the (stopkind=None) tasks (outlier/nq/...) decode the full
# max_new budget every example (huge slowdown; outlier@3k ~40min). Stop on 151645 too -> stops at the
# real answer end (answer is fully emitted before <|im_end|>), ~10x faster, results unchanged.
IM_END_ID = 151645  # <|im_end|>
LANDMARK_TOKEN_ID = 151860
DOC_START_ID = 151648  # <|box_start|>
DOC_END_ID = 151649  # <|box_end|>
PAD_TOKEN_ID = 151863

# Full task order (5 in-distribution v2 tasks + 4 OOD ladders + the CTC-suite tasks below).
ALL_TASKS = [
    "contradiction",
    "nq",
    "oolong",
    "rerank",
    "outlier",
    "fiqa",
    "scifact",
    "outlier_review",
    "contra_fever",
]

# ----------------------------------------------------------------------------------------------------
# CTC-suite tasks (2026-10-21): every setA training task the ladder did not already grade. Each is
# RENDERED exactly as the setA converter renders its training rows (recipes/data/ctc_sft/
# build_ctc_sft.py SET_A: olmo_core build_prompt spec + chunk_by="document") and GRADED exactly as
# olmo-eval grades the matching 22-task-suite roster row: the vendored ctc spec's stop preset ->
# parse(text, n_docs) -> score(parsed, gold) (olmo-eval ctc_suite._score_output). The default decode
# budget is olmo-eval's (_decode_budget: max(stop-preset budget, spec budget)). They have no default
# rung files: pass them with DC_RUNG_FILES (fetch them with
# research/token_routing/ds64_fast8k/fetch_suite_eval_rows.py), or set CTC_SUITE_ROWS_ROOT to a
# <subset>/rung_<tokens>.jsonl tree.
#   ladder task: (build_prompt spec, ctc grader spec, HF subset of PrasannSinghal/ctc-suite-eval)
SUITE_TASKS = {
    "hotpotqa": ("retrieval", "retrieval", "hotpotqa"),  # suite row ctc_hpqa
    "qdmatch_nq": ("qdmatch", "qdmatch", "qdmatch_nq"),  # ctc_qdmatch_nq
    "xabsence": ("xabsence", "xabsence", "xabsence"),  # ctc_xabsence (exact-copy Gutenberg build)
    "reorder": ("reorder", "reorder", "reorder"),  # ctc_reorder (kendall_tau in [-1, 1])
    "strmatch": ("strmatch", "strmatch", "strmatch"),  # ctc_strmatch
    "grouping": ("grouping", "grouping", "grouping"),  # ctc_grouping (unlabeled OpenAlex)
}
ALL_TASKS += list(SUITE_TASKS)
#: rung label -> token count in the fetched <subset>/rung_<tokens>.jsonl names (olmo-eval RUNG_TOKENS)
SUITE_RUNG_TOKENS = {"2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768}
#: Opt-in (``--ctc-grade`` / env ``LADDER_CTC_GRADE``, 2026-10-22): grade these four ladder tasks with
#: the vendored ctc spec olmo-eval uses for their roster row (ctc_nq / ctc_outlier / ctc_oolong /
#: ctc_contradiction) instead of the native ``ctc_eval`` graders -- for scoring the SUITE rows
#: (``<subset>/rung_*.jsonl``; contradiction = ``contradiction_iid``) the setA training mix is IID with.
#: Prompt rendering / segmentation stay the ladder's (loadtask, chunk_by, item_regex unchanged); the
#: primary metric, decode budget and stop preset become the spec's, as for SUITE_TASKS.
#:   ladder task: ctc grader spec
#: rerank (2026-10-25) is graded by the ctc ``rerank`` spec on its suite rows (HF subset ``rerank``,
#: olmo-eval roster row ctc_rerank): primary ``ce_pos_recall``, newline stop, 512-token budget, and
#: the primary mean taken over the rows that HAVE a CE-positive document only (olmo-eval's
#: ``skip_when_zero="ce_ref_available"``; 14/500 rows at r2k and r8k have none).
CTC_GRADED_BASE = {"nq": "retrieval", "outlier": "outlier", "oolong": "oolong",
                   "contradiction": "contradiction", "rerank": "rerank"}
#: ctc specs whose primary-metric mean skips rows where this sub-metric is 0 (metric undefined there,
#: not failed) -- olmo-eval ctc_suite._SKIP_WHEN_ZERO, so the ladder number equals olmo-eval's.
CTC_SKIP_WHEN_ZERO = {"rerank": "ce_ref_available"}


def _ctc_registry():
    """
    The vendored ``ctc`` task registry with the six suite graders (and the four CTC_GRADED_BASE
    graders) registered.

    ``ctc`` is stdlib-only. It is imported from the environment when installed, else from this
    repo's ``ctc/src`` (the eval runners put only ``src`` and ``recipes/eval/legacy_native`` on the
    path). Plain ``grouping`` is not in ctc's canonical set; it is registered here from the vendored
    factory with the same arguments olmo-eval's ``ctc_suite`` uses.

    Probe ``ctc.format``, not ``ctc``: run from the repo root, the ``ctc/`` folder (no
    ``__init__.py``) imports as an empty namespace package and ``import ctc`` succeeds.
    """
    try:
        import ctc.format  # noqa: F401
    except ImportError:
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.normpath(os.path.join(here, *[os.pardir] * 5, "ctc", "src")))
        for name in [m for m in sys.modules if m == "ctc" or m.startswith("ctc.")]:
            del sys.modules[name]
    import importlib

    from ctc.format import registry
    from ctc.format.prompts import GROUPING_INSTRUCTION
    from ctc.tasks._grouping import make_grouping_spec

    for mod in ("retrieval", "qdmatch", "xabsence", "reorder", "strmatch", "outlier", "oolong",
                "contradiction", "rerank"):
        importlib.import_module(f"ctc.tasks.{mod}")  # registers on first import
    if "grouping" not in registry.names():
        registry.register(
            make_grouping_spec(
                name="grouping",
                description="Partition abstracts into their (unnamed) field clusters.",
                instruction=GROUPING_INSTRUCTION,
                rungs=("2k", "4k", "8k", "16k", "32k"),
                query_builder=lambda ex: ex["queries"][0] if ex.get("queries") else "",
                sources=("openalex",),
            )
        )
    return registry


def ctc_spec(name):
    """:returns: The vendored ctc :class:`TaskSpec` named ``name``."""
    return _ctc_registry().get(name)


def ctc_decode_budget(spec):
    """olmo-eval's decode budget for a suite row with no override: the larger of the stop preset's
    and the spec's budgets (none of the six rows here carries a roster override)."""
    from ctc.eval.stopping import STOP_PRESETS

    return max(STOP_PRESETS[spec.stop].max_new_tokens, spec.max_new_tokens)


def ctc_score_one(spec, example, text):
    """
    Grade one generation exactly as olmo-eval's ``ctc_suite._score_output`` does: the spec's stop
    preset cleans it up (strip think, truncate at the first firing text stop), a generation still
    inside an unclosed ``<think>`` is a parse failure, then ``parse(text, n_docs)`` and
    ``score(parsed, gold)`` with the spec's declared gold field.

    :returns: ``(parsed, scored)``.
    """
    from ctc.eval.stopping import STOP_PRESETS
    from ctc.eval.stopping import apply as apply_stop
    from ctc.eval.stopping import in_unclosed_think

    cleaned = apply_stop(text or "", STOP_PRESETS[spec.stop])
    parsed = None if in_unclosed_think(cleaned) else spec.parse(cleaned, len(example["documents"]))
    if spec.extra.get("score_takes_example") or spec.name == "oolong":
        gold = example
    else:
        gold = example.get(spec.extra.get("gold_field", "gold_doc_indices"))
        if gold is None:
            gold = example
    return parsed, {k: float(v) for k, v in spec.score(parsed, gold).items()}


_INT_RE = re.compile(r"\d+")

#: The reorder metrics added beside the strict ``kendall_tau`` (2026-10-25). See reorder_tolerant_scores.
REORDER_TOLERANT_KEYS = ("kendall_tau_tolerant", "pairwise_acc_tolerant", "id_coverage", "dup_rate")


def reorder_tolerant_scores(text, gold_order):
    """
    Parse-tolerant partial credit for ``reorder``, next to the ctc spec's strict ``kendall_tau``
    (which scores 0 unless the output contains an EXACT permutation of ``1..n``; at 8k nearly every
    generation repeats ids or loops, so the strict metric reads ~0 for every arm, dense included).

    Parsing: every integer in ``text`` in emission order; an id outside the gold set is ignored, a
    repeated id keeps its FIRST occurrence (so a repetition loop costs nothing beyond the ids it
    never reached). Scoring is over all ``T = n(n-1)/2`` gold pairs ``(a, b)``, ``a`` before ``b`` in
    ``gold_order``: concordant ``C`` = both emitted, ``a`` first; discordant ``D`` = both emitted,
    ``b`` first; a pair with a never-emitted id is "missing".

    :returns: ``kendall_tau_tolerant = (C - D) / T`` in ``[-1, 1]`` -- missing pairs count 0, the
        chance value, so an empty answer scores 0 like the strict metric's parse failure, a full
        exact permutation scores its true Kendall tau, and an exact reversal scores -1;
        ``pairwise_acc_tolerant = C / T`` in ``[0, 1]`` -- missing pairs count WRONG (a random full
        permutation scores 0.5, an empty answer 0); ``id_coverage`` = share of gold ids emitted;
        ``dup_rate`` = share of emitted in-range ids that were repeats.

    .. note::
       On an exact permutation this is NOT the strict ``kendall_tau``: the vendored
       ``ctc.format.metrics.kendall_tau(pred, gold)`` correlates the id VALUES position by position
       (``scipy.stats.kendalltau(pred, gold)``), not the order of each id pair, so it equals the
       ordering tau only when ``gold_order`` is the identity (a reversed answer scores 1/3 on
       ``gold=[3,1,2]``). Both give 1.0 on a perfect answer.
    """
    gold = [int(g) for g in gold_order]
    n = len(gold)
    valid = set(gold)
    pos, n_in_range, n_dup = {}, 0, 0
    for tok in _INT_RE.findall(text or ""):
        i = int(tok)
        if i not in valid:
            continue
        n_in_range += 1
        if i in pos:
            n_dup += 1
        else:
            pos[i] = len(pos)
    total = n * (n - 1) // 2
    seq = [pos[g] for g in gold if g in pos]  # emitted positions, in gold order
    conc = sum(1 for a in range(len(seq)) for b in range(a + 1, len(seq)) if seq[a] < seq[b])
    disc = len(seq) * (len(seq) - 1) // 2 - conc
    return {
        "kendall_tau_tolerant": (conc - disc) / total if total else 0.0,
        "pairwise_acc_tolerant": conc / total if total else 0.0,
        "id_coverage": len(pos) / n if n else 0.0,
        "dup_rate": n_dup / n_in_range if n_in_range else 0.0,
    }


def ctc_clean_text(spec, text):
    """The text :func:`ctc_score_one` parses (stop preset applied), or ``None`` for a generation
    still inside an unclosed ``<think>`` (a parse failure there)."""
    from ctc.eval.stopping import STOP_PRESETS
    from ctc.eval.stopping import apply as apply_stop
    from ctc.eval.stopping import in_unclosed_think

    cleaned = apply_stop(text or "", STOP_PRESETS[spec.stop])
    return None if in_unclosed_think(cleaned) else cleaned


def make_ctc_score_fn(spec_name):
    """
    A ladder ``score_fn(examples, responses) -> (metrics, per_row)`` for one suite grader: the mean
    of every metric the spec returns over ALL rows (a too-long row is an empty response, scored 0),
    plus ``parse_rate`` and ``eval_size``. The primary key is ``spec.primary_metric``.

    ``reorder`` also gets :data:`REORDER_TOLERANT_KEYS` (:func:`reorder_tolerant_scores` on the same
    stop-cleaned text; the strict ``kendall_tau`` is unchanged). A spec in
    :data:`CTC_SKIP_WHEN_ZERO` (rerank) takes its PRIMARY mean over the rows where the gate metric
    is > 0 only, as olmo-eval does, and reports ``eval_size_gated`` (that row count); every other
    metric, ``parse_rate`` and ``eval_size`` stay over all rows.
    """
    spec = ctc_spec(spec_name)
    gate = CTC_SKIP_WHEN_ZERO.get(spec.name)

    def score_fn(examples, responses):
        rows, sums, n_parsed = [], {}, 0
        for ex, resp in zip(examples, responses):
            raw = ex.get("ex", ex)
            parsed, scored = ctc_score_one(spec, raw, resp or "")
            if spec.name == "reorder":
                cleaned = ctc_clean_text(spec, resp or "")
                scored.update(reorder_tolerant_scores(cleaned or "", raw["gold_order"]))
            n_parsed += parsed is not None
            rows.append(scored)
            for k, v in scored.items():
                if k != "parsed":
                    sums[k] = sums.get(k, 0.0) + v
        n = max(1, len(rows))
        res = {k: v / n for k, v in sums.items()}
        res["parse_rate"] = n_parsed / n
        res["eval_size"] = len(rows)
        if gate is not None:
            counted = [r for r in rows if r.get(gate, 0.0) > 0.0]
            prim = spec.primary_metric
            res[prim] = sum(r[prim] for r in counted) / len(counted) if counted else 0.0
            res["eval_size_gated"] = len(counted)
        return res, rows

    return score_fn


class RepetitionStop:
    """
    Opt-in decode stop for runaway repetition (``--repetition-stop``, default ``off``).

    * ``id``: stop once an integer id the generation already emitted is emitted again (checked when
      the repeated number is terminated by a non-digit, e.g. ``", 23,"``). Meant for ``reorder``
      (and ``rerank``), whose answer lists each id once.
    * ``ngram``: stop once the last ``ngram`` generated TOKENS repeat an earlier ``ngram``-token
      window of the same generation.

    The token that triggers the stop is kept, so the text ends with the start of the repeat. Feed
    tokens with :meth:`update`, which returns True when decoding should stop. ``piece`` decodes ONE
    token id to text (``id`` mode only).
    """

    def __init__(self, kind, ngram=16, piece=None):
        if kind not in ("id", "ngram"):
            raise ValueError(f"repetition stop kind {kind!r} not in (id, ngram)")
        if kind == "id" and piece is None:
            raise ValueError("repetition stop 'id' needs a piece(token_id) -> str decoder")
        self.kind, self.ngram, self.piece = kind, int(ngram), piece
        self.reset()

    def reset(self):
        self._toks, self._grams, self._seen, self._buf = [], set(), set(), ""

    def update(self, token_id):
        if self.kind == "ngram":
            self._toks.append(int(token_id))
            if len(self._toks) < self.ngram:
                return False
            gram = tuple(self._toks[-self.ngram:])
            if gram in self._grams:
                return True
            self._grams.add(gram)
            return False
        for ch in self.piece(token_id):
            if ch.isdigit():
                self._buf += ch
                continue
            if self._buf:
                num, self._buf = int(self._buf), ""
                if num in self._seen:
                    return True
                self._seen.add(num)
        return False


def make_repetition_stop(kind, ngram, piece):
    """:returns: A fresh :class:`RepetitionStop` factory (``None`` when ``kind`` is off/empty)."""
    if not kind or kind == "off":
        return None
    RepetitionStop(kind, ngram, piece)  # validate now, not mid-eval
    return lambda: RepetitionStop(kind, ngram, piece)


def make_ctc_answer_complete(spec_name, decode):
    """
    Early stop for a suite task: True once the spec's stop preset would end the generation (the
    ladder checks this at newline tokens only). ``None`` for an EOS-only preset. Whatever this
    leaves on, :func:`ctc_score_one` re-applies the same preset before parsing, so the bs=1 and
    batched paths score the same text.
    """
    from ctc.eval.stopping import STOP_PRESETS, should_stop

    cond = STOP_PRESETS[ctc_spec(spec_name).stop]
    if not cond.text_stops:
        return None

    def _f(content_ids):
        return should_stop(decode(content_ids), cond) is not None

    return _f


# ----------------------------------------------------------------------------------------------------
# LADDER definition -- rung -> eval-JSONL mapping. This MIRRORS eval_lc_native.py's v1/v2 LADDERS EXACTLY
# (so the docchunk ladder JSON is directly comparable to the dense/landmark ladder JSONs). Keep in sync.
# ----------------------------------------------------------------------------------------------------
def resolve_reserved_ids(args, tok):
    """
    Point the module-level marker/stop ids at the checkpoint's tokenizer family.

    Family = ``--family`` if given, else ``model_family`` from ``<model_path>/config.json`` (the
    train_ctc_suite export writes it), else inferred from the tokenizer name (``3.5``/``3_5`` ->
    ``qwen3_5``), else ``qwen3``.
    """
    global EOS_TOKEN_ID, IM_END_ID, LANDMARK_TOKEN_ID, DOC_START_ID, DOC_END_ID, PAD_TOKEN_ID
    from olmo_core.data.document_chunk_landmark import RESERVED_IDS

    fam = args.family
    if fam == "auto":
        fam = None
        for _fn in ("config.json", "provenance.json"):  # train_ctc_suite export writes both
            cfg_p = os.path.join(args.model_path, _fn)
            if not fam and os.path.exists(cfg_p):
                try:
                    fam = json.load(open(cfg_p)).get("model_family")
                except Exception:
                    fam = None
        if not fam:
            name = args.tokenizer.lower()
            fam = "qwen3_5" if ("3.5" in name or "3_5" in name) else "qwen3"
    ids = RESERVED_IDS[fam]
    EOS_TOKEN_ID, LANDMARK_TOKEN_ID = ids.eos, ids.landmark
    DOC_START_ID, DOC_END_ID, PAD_TOKEN_ID = ids.doc_start, ids.doc_end, ids.pad
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0 and im_end != tok.unk_token_id:
        IM_END_ID = im_end
    print(f"[ids] family={fam} eos={EOS_TOKEN_ID} im_end={IM_END_ID} doc_start={DOC_START_ID} "
          f"doc_end={DOC_END_ID} landmark={LANDMARK_TOKEN_ID} pad={PAD_TOKEN_ID}", flush=True)
    return fam


def build_ladders(args):
    E5 = os.environ.get("EVAL500_ROOT", "/scratch/users/prasann/cpt_data/eval500")
    # v1 ladders are DISABLED (2026-07-29). Each v1 rung drew its OWN questions, so every
    # rung-to-rung delta carried eval-set resampling noise on top of the length effect it was
    # meant to isolate. v2 fixes the question set across rungs and varies only the distractors.
    if args.ladder_version != "v2":
        raise NotImplementedError(
            f"--ladder-version {args.ladder_version!r} is no longer supported: v2 is the only "
            "ladder. Build what you need as v2 (build_v2_eval_ladders.py for 2k-32k, "
            "build_xlong_rungs.py for 64k-2M) and point EVAL500_ROOT at a v2 bundle."
        )
    if args.ladder_version == "v2":
        # v2: every rung of a task shares the SAME >=500 questions; only distractor docs differ. ALL
        # rungs live under $EVAL500_ROOT/<task>/ (point EVAL500_ROOT at the v2 bundle).
        ladders_v2 = {
            "contradiction": [
                ("2k", f"{E5}/contra/contradiction_eval_pubmed_both_n100_k3.jsonl"),
                ("8k", f"{E5}/contra/contradiction_eval_pubmed_both_n190_k3.jsonl"),
                ("16k", f"{E5}/contra/contradiction_eval_pubmed_both_n385_k3.jsonl"),
                ("32k", f"{E5}/contra/contradiction_eval_pubmed_both_n765_k3.jsonl"),
            ],
            "nq": [
                ("3k", f"{E5}/nq/nq_validation_k20_600.jsonl"),
                ("8k", f"{E5}/nq/nq_validation_k50_600.jsonl"),
                ("16k", f"{E5}/nq/nq_validation_k100_600.jsonl"),
                ("32k", f"{E5}/nq/nq_validation_k200_600.jsonl"),
            ],
            "outlier": [
                ("3k", f"{E5}/outlier/outlier_wiki100w_n22_k3_eval_600.jsonl"),
                ("8k", f"{E5}/outlier/outlier_wiki100w_n55_k3_eval_600.jsonl"),
                ("16k", f"{E5}/outlier/outlier_wiki100w_n110_k3_eval_600.jsonl"),
                ("32k", f"{E5}/outlier/outlier_wiki100w_n220_k3_eval_600.jsonl"),
                # 128k transfer-test rung (n=880 docs); see corpus_reasoning/eval/eval_lc_native.py.
                ("128k", f"{E5}/outlier/outlier_wiki100w_n880_k3_eval_600.jsonl"),
            ],
            "rerank": [
                ("3k", f"{E5}/rerank/msmarco_trainhn_eval_k20_500.jsonl"),
                ("8k", f"{E5}/rerank/msmarco_trainhn_eval_k50_500.jsonl"),
                ("16k", f"{E5}/rerank/msmarco_trainhn_eval_k100_500.jsonl"),
            ],
            "oolong": [
                ("8k", f"{E5}/oolong/oolong_test_synth_ctx8192_spliteval.jsonl"),
                ("16k", f"{E5}/oolong/oolong_test_synth_ctx16384_spliteval.jsonl"),
                ("32k", f"{E5}/oolong/oolong_test_synth_ctx32768_spliteval.jsonl"),
            ],
            # OOD generalization (held-out BEIR retrieval, graded as retrieval f1). Version-agnostic.
            "fiqa": [
                ("2k", f"{E5}/beir/beir_fiqa_ce_ladder_k10_648.jsonl"),
                ("4k", f"{E5}/beir/beir_fiqa_ce_ladder_k20_648.jsonl"),
                ("8k", f"{E5}/beir/beir_fiqa_ce_ladder_k40_648.jsonl"),
                ("16k", f"{E5}/beir/beir_fiqa_ce_ladder_k80_648.jsonl"),
            ],
            "scifact": [
                ("4k", f"{E5}/beir/beir_scifact_ladder_k11_299.jsonl"),
                ("8k", f"{E5}/beir/beir_scifact_ladder_k22_299.jsonl"),
                ("16k", f"{E5}/beir/beir_scifact_ladder_k44_299.jsonl"),
                ("32k", f"{E5}/beir/beir_scifact_ladder_k88_299.jsonl"),
            ],
            # OOD outlier + contradiction (review / FEVER source; graded identically via gold indices).
            "outlier_review": [
                ("3k", f"{E5}/outlier/outlier_review_matched_n30_k3_eval_600.jsonl"),
                ("8k", f"{E5}/outlier/outlier_review_matched_n75_k3_eval_600.jsonl"),
                ("16k", f"{E5}/outlier/outlier_review_matched_n150_k3_eval_600.jsonl"),
                ("32k", f"{E5}/outlier/outlier_review_matched_n300_k3_eval_600.jsonl"),
            ],
            "contra_fever": [
                ("2k", f"{E5}/contra/contradiction_eval_fever_plain_n100_k3.jsonl"),
                ("8k", f"{E5}/contra/contradiction_eval_fever_plain_n408_k3.jsonl"),
                ("16k", f"{E5}/contra/contradiction_eval_fever_plain_n820_k3.jsonl"),
                ("32k", f"{E5}/contra/contradiction_eval_fever_plain_n1642_k3.jsonl"),
            ],
        }
        # ---- OPT-IN ultra-long rungs (64k/128k/256k), OFF by default (v2 only) ----
        # KEEP IN SYNC with eval_lc_native.py: size-labelled glob so the calibrated doc-count in the
        # filename can drift without editing this file. Files staged on weka under EVAL500_ROOT/<sub>/
        # (NB contra subdir is "contra"). contra|nq|outlier only. Runner forces bs=1 + raises max_length.
        if getattr(args, "xlong", False):
            import glob as _glob

            _XL = {
                "contradiction": ("contra", "contradiction_eval_pubmed_both_n*_k3_xlong_{s}.jsonl"),
                "nq": ("nq", "nq_validation_k*_xlong_{s}.jsonl"),
                "outlier": ("outlier", "outlier_wiki100w_n*_k3_eval_xlong_{s}.jsonl"),
            }
            for _t, (_sub, _pat) in _XL.items():
                for _s in ("64k", "128k", "256k"):
                    _hits = sorted(_glob.glob(os.path.join(E5, _sub, _pat.format(s=_s))))
                    if _hits:
                        ladders_v2[_t].append((_s, _hits[0]))
            print(f"[xlong] appended ultra-long rungs where files exist under {E5}", flush=True)
        # CTC-suite tasks: no default files. CTC_SUITE_ROWS_ROOT=<dir> with <subset>/rung_<tokens>.jsonl
        # (fetch_suite_eval_rows.py writes <out>/suite/<subset>/...) supplies every rung present.
        _suite_root = os.environ.get("CTC_SUITE_ROWS_ROOT")
        if _suite_root:
            for _t, (_, _, _sub) in SUITE_TASKS.items():
                _r = [(lab, os.path.join(_suite_root, _sub, f"rung_{tok}.jsonl"))
                      for lab, tok in SUITE_RUNG_TOKENS.items()]
                ladders_v2[_t] = [(lab, p) for lab, p in _r if os.path.exists(p)]
        # FLOP-scaling study: explicit rung files per task, e.g. to score a marker-trained model
        # on EXACTLY the rung files a prior dense campaign used (outlier_lengthmix/eval_rungs/...).
        # DC_RUNG_FILES = JSON {"outlier": {"8k": "/weka/.../rung_8192.jsonl", ...}, ...}; a task
        # present here REPLACES its default rung list. (Until 2026-09-02 this sat after the v2
        # return below, i.e. never ran: every "override" eval scored the default v2 files.)
        _ovr = os.environ.get("DC_RUNG_FILES")
        if _ovr:
            for _task, _rungs in json.loads(_ovr).items():
                ladders_v2[_task] = [(lab, path) for lab, path in _rungs.items()]
            print(f"[ladder] DC_RUNG_FILES override for {sorted(json.loads(_ovr))}: "
                  f"{ {t: [p for _, p in ladders_v2[t]] for t in json.loads(_ovr)} }", flush=True)
        return ladders_v2
    _ladders = {
        "contradiction": [
            ("2k", args.contra_data),
            ("8k", f"{E5}/contra/contradiction_eval_pubmed_both_n190_k3.jsonl"),
            ("16k", f"{E5}/contra/contradiction_eval_pubmed_both_n385_k3.jsonl"),
            ("32k", f"{E5}/contra/contradiction_eval_pubmed_both_n765_k3.jsonl"),
        ],
        "nq": [
            ("3k", args.nq_data),
            ("8k", f"{E5}/nq/nq_validation_k50_hn5_600.jsonl"),
            ("16k", f"{E5}/nq/nq_validation_k100_hn10_600.jsonl"),
            ("32k", f"{E5}/nq/nq_validation_k200_hn20_600.jsonl"),
        ],
        "oolong": [
            ("1k", "data/oolong_test_synth_ctx1024_spliteval.jsonl"),
            ("2k", "data/oolong_test_synth_ctx2048_spliteval.jsonl"),
            ("4k", "data/oolong_test_synth_ctx4096_spliteval.jsonl"),
            ("8k", "data/oolong_test_synth_ctx8192_spliteval.jsonl"),
            ("16k", "data/oolong_test_synth_ctx16384_spliteval.jsonl"),
            ("32k", "data/oolong_test_synth_ctx32768_spliteval.jsonl"),
        ],
        "rerank": [
            ("2k", args.rerank_data),
            ("8k", f"{E5}/rerank/msmarco_validation_rerank_k80_600.jsonl"),
            ("16k", f"{E5}/rerank/msmarco_validation_rerank_k158_597.jsonl"),
            ("32k", f"{E5}/rerank/msmarco_validation_rerank_k315_599.jsonl"),
        ],
        "outlier": [
            ("3k", args.outlier_data),
            ("8k", f"{E5}/outlier/outlier_wiki100w_n55_k3_eval_600.jsonl"),
            ("16k", f"{E5}/outlier/outlier_wiki100w_n110_k3_eval_600.jsonl"),
            ("32k", f"{E5}/outlier/outlier_wiki100w_n220_k3_eval_600.jsonl"),
        ],
        "fiqa": [
            ("2k", f"{E5}/beir/beir_fiqa_ce_ladder_k10_648.jsonl"),
            ("4k", f"{E5}/beir/beir_fiqa_ce_ladder_k20_648.jsonl"),
            ("8k", f"{E5}/beir/beir_fiqa_ce_ladder_k40_648.jsonl"),
            ("16k", f"{E5}/beir/beir_fiqa_ce_ladder_k80_648.jsonl"),
        ],
        "scifact": [
            ("4k", f"{E5}/beir/beir_scifact_ladder_k11_299.jsonl"),
            ("8k", f"{E5}/beir/beir_scifact_ladder_k22_299.jsonl"),
            ("16k", f"{E5}/beir/beir_scifact_ladder_k44_299.jsonl"),
            ("32k", f"{E5}/beir/beir_scifact_ladder_k88_299.jsonl"),
        ],
        "outlier_review": [
            ("3k", f"{E5}/outlier/outlier_review_matched_n30_k3_eval_600.jsonl"),
            ("8k", f"{E5}/outlier/outlier_review_matched_n75_k3_eval_600.jsonl"),
            ("16k", f"{E5}/outlier/outlier_review_matched_n150_k3_eval_600.jsonl"),
            ("32k", f"{E5}/outlier/outlier_review_matched_n300_k3_eval_600.jsonl"),
        ],
        "contra_fever": [
            ("2k", f"{E5}/contra/contradiction_eval_fever_plain_n100_k3.jsonl"),
            ("8k", f"{E5}/contra/contradiction_eval_fever_plain_n408_k3.jsonl"),
            ("16k", f"{E5}/contra/contradiction_eval_fever_plain_n820_k3.jsonl"),
            ("32k", f"{E5}/contra/contradiction_eval_fever_plain_n1642_k3.jsonl"),
        ],
    }
    return _ladders


# Per-task segmentation + scoring spec. loadtask/chunk_by/item_regex match the training converter; the
# grader + primary-metric key match eval_lc_native.py's LSPEC (OOD tasks reuse the base graders).
#   primary_key=None -> rerank: take the first "mrr*" metric.
def build_task_spec(args):
    spec = {
        # task:            (loadtask,        chunk_by,   item_regex, primary_key, max_new_tokens,            stopkind)
        "contradiction": (
            "contradiction",
            "document",
            r"\|\|",
            "f1",
            args.contra_max_new_tokens,
            "contra",
        ),
        "nq": ("retrieval", "document", r"\|\|", "f1", args.nq_max_new_tokens, None),
        "oolong": ("oolong", "line", r"\|\|", "score", args.oolong_max_new_tokens, "oolong"),
        "rerank": ("rerank", "document", r"\|\|", None, args.rerank_max_new_tokens, None),
        "outlier": ("outlier", "document", r"\|\|", "f1", args.outlier_max_new_tokens, None),
        # ---- OOD ladders: reuse the in-distribution graders + segmentation of their base task ----
        "fiqa": ("retrieval", "document", r"\|\|", "f1", args.nq_max_new_tokens, None),
        "scifact": ("retrieval", "document", r"\|\|", "f1", args.nq_max_new_tokens, None),
        "outlier_review": ("outlier", "document", r"\|\|", "f1", args.outlier_max_new_tokens, None),
        "contra_fever": (
            "contradiction",
            "document",
            r"\|\|",
            "f1",
            args.contra_max_new_tokens,
            "contra",
        ),
    }
    # CTC-suite tasks (only those asked for, so a run of the tasks above never imports ctc): the
    # setA converter's spec + chunk_by="document", the ctc spec's primary metric, olmo-eval's
    # decode budget unless --suite-max-new-tokens overrides it, and the spec's stop preset.
    asked = {t.strip() for t in args.tasks.split(",") if t.strip()}
    override = {}
    for kv in filter(None, (s.strip() for s in (args.suite_max_new_tokens or "").split(","))):
        k, v = kv.split("=")
        override[k.strip()] = int(v)
    for task, (prompt_spec, grader, _sub) in SUITE_TASKS.items():
        if task not in asked:
            continue
        sp = ctc_spec(grader)
        spec[task] = (
            prompt_spec,
            "document",
            r"\|\|",
            sp.primary_metric,
            override.get(task, ctc_decode_budget(sp)),
            f"ctc:{grader}",
        )
    # --ctc-grade: the four base tasks keep their rendering, take the ctc spec's grading knobs
    for task in ctc_graded_tasks(args):
        loadtask, chunk_by, item_regex = spec[task][:3]
        sp = ctc_spec(CTC_GRADED_BASE[task])
        spec[task] = (loadtask, chunk_by, item_regex, sp.primary_metric,
                      override.get(task, ctc_decode_budget(sp)), f"ctc:{CTC_GRADED_BASE[task]}")
    return spec


def ctc_graded_tasks(args):
    """:returns: The CTC_GRADED_BASE tasks ``--ctc-grade`` names (``all`` = every one)."""
    names = [t.strip() for t in (getattr(args, "ctc_grade", "") or "").split(",") if t.strip()]
    if names == ["all"]:
        names = list(CTC_GRADED_BASE)
    bad = [t for t in names if t not in CTC_GRADED_BASE]
    if bad:
        raise SystemExit(f"--ctc-grade: {bad} not in {sorted(CTC_GRADED_BASE)}")
    return names



def _tokenizer_with_retry(name, tries=8):
    """``AutoTokenizer.from_pretrained`` with exponential backoff: with dozens of evals starting
    at once the Hub answers 429 (rate limit) and the whole job died at startup (2026-09-02)."""
    from transformers import AutoTokenizer

    delay = 20.0
    for i in range(tries):
        try:
            return AutoTokenizer.from_pretrained(name)
        except Exception as e:  # HfHubHTTPError / OSError wrap the 429
            if i == tries - 1:
                raise
            print(f"[tokenizer] load failed ({type(e).__name__}); retry {i + 1}/{tries - 1} in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 300.0)

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--variant",
        required=True,
        choices=["dense", "landmark", "full", "hierarchical"],
        help="hierarchical loads EXACTLY like dense (DocumentChunkedAttention, dense "
        "doc-chunk data, no landmarks) -- config.json drives the attention class.",
    )
    ap.add_argument("--model-path", required=True, help="step dir: config.json + model_and_optim/")
    ap.add_argument(
        "--out",
        required=True,
        help="ladder JSON. MERGES <task>_<rung> keys into an existing file if present, so "
        "per-task/per-rung split invocations accumulate into one ladder JSON.",
    )
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    ap.add_argument("--family", default="auto",
                    help="reserved-id family (qwen3|qwen3_5|olmo3|llama|gemma); auto = checkpoint "
                    "config.json model_family, else inferred from the tokenizer name")
    ap.add_argument(
        "--ladder-version",
        choices=["v2"],
        default="v2",
        help="v2 (DEFAULT): every rung of a task shares the SAME >=500 questions, only "
        "distractors vary (reads the _eval_bundle_eval500_v2 bundle via EVAL500_ROOT).",
    )
    ap.add_argument(
        "--mem-freq",
        type=int,
        default=63,
        help="landmark variant: a landmark token every mem_freq content tokens (window=64).",
    )
    ap.add_argument("--max-test-samples", type=int, default=400)
    ap.add_argument("--gen-batch", type=int, default=int(os.environ.get("LADDER_GEN_BATCH", "1")),
                    help="non-landmark models: decode this many examples per batch (1 = the bs=1 loop)")
    ap.add_argument("--gen-batch-tokens", type=int, default=int(os.environ.get("LADDER_GEN_BATCH_TOKENS", "98304")),
                    help="cap on padded prompt tokens per decode batch")
    ap.add_argument("--gen-compare", type=int, default=int(os.environ.get("LADDER_GEN_COMPARE", "0")),
                    help="also decode the first K rows bs=1 and report agreement + speed")
    ap.add_argument(
        "--max-length",
        type=int,
        default=40960,
        help="KV-cache length budget (covers the 32k rung + box/landmark token overhead).",
    )

    # ---- task/rung filters (let the launcher split a slow ladder to dodge wall-clock timeouts) ----
    ap.add_argument(
        "--tasks",
        default=",".join(t for t in ALL_TASKS if t not in SUITE_TASKS),
        help="comma list of tasks to run: default = the 9 ladder tasks; the CTC-suite tasks "
        "(SUITE_TASKS) run only when named.",
    )
    ap.add_argument(
        "--rungs",
        default=None,
        help="comma list restricting rungs (e.g. '16k,32k'); applied across tasks.",
    )
    ap.add_argument(
        "--xlong",
        action="store_true",
        default=os.environ.get("LADDER_XLONG") == "1",
        help="OPT-IN (v2 only): append the ultra-long 64k/128k/256k rungs (contra|nq|outlier) "
        "by size-labelled glob under EVAL500_ROOT. Combine with --rungs 64k,128k to pick "
        "which; the runner forces bs=1 and raises --max-length. Honors env LADDER_XLONG=1.",
    )

    # ---- per-task max-new-tokens knobs ----
    ap.add_argument(
        "--contra-max-new-tokens",
        type=int,
        default=96,
        help="NO-CoT contradiction: the 'Contradicting pairs: [[...]]' line is short.",
    )
    ap.add_argument("--nq-max-new-tokens", type=int, default=64)
    ap.add_argument("--oolong-max-new-tokens", type=int, default=200)
    ap.add_argument("--rerank-max-new-tokens", type=int, default=256)
    ap.add_argument("--outlier-max-new-tokens", type=int, default=200)
    ap.add_argument(
        "--ctc-grade",
        default=os.environ.get("LADDER_CTC_GRADE", ""),
        help="comma list of nq/outlier/oolong/contradiction/rerank (or 'all') to grade with the ctc "
        "spec olmo-eval uses for the suite row (CTC_GRADED_BASE), e.g. when scoring suite rows.",
    )
    ap.add_argument(
        "--repetition-stop",
        default=os.environ.get("LADDER_REPETITION_STOP", "off"),
        choices=["off", "id", "ngram"],
        help="OPT-IN decode stop for repetition loops (default off = scores comparable with every "
        "earlier number): 'id' stops once an already-emitted integer id repeats, 'ngram' once the "
        "last --repetition-ngram tokens repeat an earlier window. Applies to "
        "--repetition-stop-tasks only. When on, the log's metric names get a '_repstop_<kind>' "
        "suffix and _meta.repetition_stop records it. Env LADDER_REPETITION_STOP.",
    )
    ap.add_argument("--repetition-stop-tasks",
                    default=os.environ.get("LADDER_REPETITION_STOP_TASKS", "reorder"),
                    help="comma list of tasks the --repetition-stop applies to (default reorder)")
    ap.add_argument("--repetition-ngram", type=int,
                    default=int(os.environ.get("LADDER_REPETITION_NGRAM", "16")),
                    help="window length in tokens for --repetition-stop ngram")
    ap.add_argument(
        "--suite-max-new-tokens",
        default=os.environ.get("LADDER_SUITE_MAX_NEW_TOKENS", ""),
        help="CTC-suite tasks: 'task=N,...' decode-budget overrides. Default = olmo-eval's budget "
        "(hotpotqa 64, qdmatch_nq 512, xabsence 200, reorder 2048, strmatch 512, grouping 4096).",
    )

    # ---- per-task default (2k/3k) base eval JSONLs used only for --ladder-version v1 ----
    ap.add_argument("--contra-data", default="data/contradiction_eval_pubmed_both_n100_k3.jsonl")
    ap.add_argument("--nq-data", default="data/nq_validation_k20_hn2_600.jsonl")
    ap.add_argument("--rerank-data", default="data/msmarco_dev_rerank_k20_1000.jsonl")
    ap.add_argument("--outlier-data", default="data/outlier_wiki100w_n20_k3_eval_100.jsonl")
    ap.add_argument(
        "--root",
        default=None,
        help="chdir here before resolving relative data paths (on-cluster: the weka $BUNDLE, "
        "so relative data/... base + v1-oolong files resolve).",
    )
    # accepted for launcher parity; docchunk builds its own prefill via segment_prompt_to_chunks.
    ap.add_argument(
        "--query-position",
        default="both",
        choices=["before", "after", "both"],
        help="prompt layout; MUST match the SFT shards the model trained on (query-after shards -> "
        "after). Was hardcoded to 'both' before the FLOP-scaling study.",
    )
    ap.add_argument(
        "--cot-mode",
        default="none",
        choices=["none", "plan"],
        help="OOLONG CoT mode used to build the prefill prompt. 'none' (default) keeps the "
        "no-CoT behavior byte-identical; 'plan' matches a CoT-trained checkpoint so the "
        "model can externalize cross-item aggregation into (free/global) generated tokens.",
    )
    ap.add_argument(
        "--prompt-format",
        default="chat",
        choices=["chat", "raw", "alpaca"],
        help="accepted for runner parity; docchunk prefill mirrors the training converter.",
    )

    # ---- landmark-only top-k inference knobs (exact if unset; mirror the single-task scripts) ----
    ap.add_argument(
        "--landmark-top-k-blocks",
        type=int,
        default=None,
        help="landmark variant: keep only top-k landmark BLOCKS per query (exact if unset).",
    )
    ap.add_argument(
        "--landmark-top-k-fraction",
        type=float,
        default=None,
        help="landmark variant: top-k = ceil(fraction * num_prompt_blocks) (reference: 0.1).",
    )
    args = ap.parse_args()
    import torch

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.root:
        os.chdir(args.root)

    from ctc_eval.eval.evaluate import (
        _eval_contradiction,
        _eval_oolong,
        _eval_outlier,
        _eval_rerank,
        _eval_retrieval,
        load_unified_examples,
    )
    from transformers import AutoTokenizer

    from olmo_core.config import DType
    from olmo_core.data.document_chunk_landmark import DOC_END_ID as _DE
    from olmo_core.data.document_chunk_landmark import DOC_START_ID as _DS
    from olmo_core.data.document_chunk_landmark import (
        emit_document_chunk_dense,
        emit_document_chunk_landmark,
        segment_prompt_to_chunks,
    )
    from olmo_core.generate.generation_module.config import GenerationConfig
    from olmo_core.generate.generation_module.transformer import (
        TransformerGenerationModuleConfig,
    )

    # (module defaults equal the library's Qwen3 defaults; resolve_reserved_ids() re-points them)
    assert (_DS, _DE) == (151648, 151649)

    # Grader per task. OOD ladders reuse their base-task grader (fiqa/scifact->retrieval,
    # outlier_review->outlier, contra_fever->contradiction), matching eval_lc_native.py's LSPEC.
    SCORE_FNS = {
        "contradiction": _eval_contradiction,
        "nq": _eval_retrieval,
        "oolong": _eval_oolong,
        "rerank": _eval_rerank,
        "outlier": _eval_outlier,
        "fiqa": _eval_retrieval,
        "scifact": _eval_retrieval,
        "outlier_review": _eval_outlier,
        "contra_fever": _eval_contradiction,
    }

    # ---- data-parallel across ranks (torchrun) ----
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        torch.distributed.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()
    is_main = rank == 0
    if not is_main:
        sys.stdout = open(os.devnull, "w")

    tok = _tokenizer_with_retry(args.tokenizer)
    resolve_reserved_ids(args, tok)
    NEWLINE_ID = tok("\n", add_special_tokens=False).input_ids[-1]
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    is_landmark = args.variant == "landmark"
    is_full = args.variant == "full"
    use_dense_emit = not is_landmark

    t0 = time.time()
    # GenerationConfig requires pad != eos; Qwen3 has no pad and we decode bs=1 (pad unused).
    pad_id = tok.pad_token_id if tok.pad_token_id not in (None, EOS_TOKEN_ID) else IM_END_ID
    gen_cfg = GenerationConfig(
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=pad_id,
        max_length=args.max_length,
        use_cache=True,
    )
    from olmo_core.nn.nested_ffn_moe import post_build_hook_from_config

    gm = TransformerGenerationModuleConfig(
        gen_cfg, float8_config=None, dtype=DType("bfloat16"), compile_model=False
    ).build(
        checkpoint_dir=args.model_path,
        device=device,
        post_build_hook=post_build_hook_from_config(args.model_path),
    )
    if not is_full:
        gm.model.enable_document_chunk_attention(
            doc_start_id=DOC_START_ID,
            doc_end_id=DOC_END_ID,
            eos_id=EOS_TOKEN_ID,
            mode="chunked",
            pad_id=PAD_TOKEN_ID if is_landmark else None,
        )
    print(
        f"[docchunk-ladder-{args.variant}] built from {args.model_path} in {time.time()-t0:.1f}s",
        flush=True,
    )

    if is_landmark and args.landmark_top_k_blocks is not None:
        n_set = gm.model.set_landmark_eval_top_k(args.landmark_top_k_blocks)
        print(
            f"[topk] fixed top_k={args.landmark_top_k_blocks} on {n_set} landmark layers",
            flush=True,
        )

    block_size = args.mem_freq + 1  # landmark window (64); eager landmark forward needs T % 64 == 0

    def build_prefill(raw_example, loadtask, chunk_by, item_regex):
        segs, _ids, _ = segment_prompt_to_chunks(
            tok,
            raw_example,
            loadtask,
            query_position=args.query_position,
            cot_mode=args.cot_mode,
            chunk_by=chunk_by,
            item_regex=item_regex,
            include_answer=False,
            doc_start_id=DOC_START_ID,
            doc_end_id=DOC_END_ID,
        )
        if use_dense_emit:
            out, _ = emit_document_chunk_dense(
                segs
            )  # box markers present; full attention ignores them
        else:
            out, _ = emit_document_chunk_landmark(
                segs, mem_freq=args.mem_freq, mem_id=LANDMARK_TOKEN_ID, pad_id=PAD_TOKEN_ID
            )
        return out

    def strip_think(text):
        return text.split("</think>", 1)[1] if "</think>" in text else text

    def make_answer_complete(stopkind):
        if stopkind == "oolong":

            def _f(content_ids):
                return "answer:" in tok.decode(content_ids, skip_special_tokens=True).lower()

            return _f
        if stopkind == "contra":

            def _f(content_ids):
                txt = tok.decode(
                    [t for t in content_ids if t != LANDMARK_TOKEN_ID], skip_special_tokens=True
                ).lower()
                return "contradicting pairs:" in txt

            return _f
        if stopkind is not None and stopkind.startswith("ctc:"):
            return make_ctc_answer_complete(
                stopkind[len("ctc:"):],
                lambda ids: tok.decode(
                    [t for t in ids if t != LANDMARK_TOKEN_ID], skip_special_tokens=True
                ),
            )
        return None

    # FLA's GDN kernels autotune per length-derived NB: every new prompt-length bucket cost ~25-50 s
    # of Triton compiles here (this loop calls gm.model directly, so the generation module's own
    # freeze in generate_batch never ran). OLMO_FLA_LENGTH_AUTOTUNE=1 keeps per-length tuning.
    if os.environ.get("OLMO_FLA_LENGTH_AUTOTUNE") != "1":
        from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune

        print(f"[docchunk-ladder] FLA length autotune frozen on {freeze_fla_length_autotune()} kernel(s)", flush=True)

    @torch.inference_mode()  # (not no_grad: generate_batch builds its caches as inference tensors)
    def generate_one(prefill, max_new_tokens, answer_complete, rep_stop=None):
        # Size the KV cache to THIS example's actual need (prefill + decode budget), not the raised
        # --max-length. At the xlong 64k/128k rungs the runner raises --max-length to cover the longest
        # rung, so a full-max_length cache wastes ~10 GiB at the 64k rung and starves the FlexAttention
        # prefill kernel (the OOM that killed the docchunk xlong jobs). The landmark decode also inserts
        # a landmark token every mem_freq generated tokens, so budget for those extra cache slots. The
        # cache manager only reallocates when it needs to GROW (is_reusable keeps a larger buffer), so
        # this never shrinks a cache mid-rung. Capped at args.max_length as a hard ceiling.
        lm_overhead = (max_new_tokens // args.mem_freq + 2) if is_landmark else 0
        cache_len = min(len(prefill) + max_new_tokens + lm_overhead + 1, args.max_length)
        gm.prepare_inference_cache(1, cache_len)  # (re)set the cache cursor to 0 per example
        leftpad = torch.zeros(1, dtype=torch.int32, device=device)
        if not is_landmark:
            # Dense / hierarchical / full: prefill once (chunked mask applied + K,V cached for the
            # chunked variants), then single-token greedy decode over the cache (plain causal).
            logits = gm.model(
                torch.tensor([prefill], device=device), logits_to_keep=1, cache_leftpad=leftpad
            )
            nxt = int(logits[0, -1].argmax().item())
            new_content = []
            rs = rep_stop() if rep_stop is not None else None
            for _ in range(max_new_tokens):
                if nxt == EOS_TOKEN_ID or nxt == IM_END_ID:
                    break
                new_content.append(nxt)
                if (
                    nxt == NEWLINE_ID
                    and answer_complete is not None
                    and answer_complete(new_content)
                ):
                    break
                if rs is not None and rs.update(nxt):
                    break
                logits = gm.model(torch.tensor([[nxt]], device=device), logits_to_keep=1)
                nxt = int(logits[0, -1].argmax().item())
            return strip_think(tok.decode(new_content, skip_special_tokens=True))

        # Landmark: KV-cached decode. Prefill (block-aligned) runs once with the chunked grouped-softmax
        # mask + K,V cached; then each generated token is fed incrementally and decoded as a plain
        # landmark query (FREE -> chunk mask is a no-op). A real landmark token is inserted after every
        # mem_freq generated content tokens so the periodic is_mem structure matches the eager loop.
        logits = gm.model(
            torch.tensor([prefill], device=device), logits_to_keep=1, cache_leftpad=leftpad
        )
        nxt = int(logits[0, -1].argmax().item())
        new_content = []
        since_landmark = 0
        rs = rep_stop() if rep_stop is not None else None
        for _ in range(max_new_tokens):
            if nxt == EOS_TOKEN_ID or nxt == IM_END_ID:
                break
            new_content.append(nxt)
            if rs is not None and rs.update(nxt):
                break
            logits = gm.model(torch.tensor([[nxt]], device=device), logits_to_keep=1)
            since_landmark += 1
            if since_landmark == args.mem_freq:
                logits = gm.model(
                    torch.tensor([[LANDMARK_TOKEN_ID]], device=device), logits_to_keep=1
                )
                since_landmark = 0
            if nxt == NEWLINE_ID and answer_complete is not None and answer_complete(new_content):
                break
            nxt = int(logits[0, -1].argmax().item())
        return strip_think(tok.decode(new_content, skip_special_tokens=True))

    def finish_tokens(gen_ids, answer_complete, rep_stop=None):
        """The bs=1 loop's stopping rule applied to an already generated token list: stop BEFORE
        EOS / im_end, after a newline once answer_complete(prefix) holds (newline kept), and after
        the token that fires the opt-in repetition stop (kept)."""
        new_content = []
        rs = rep_stop() if rep_stop is not None else None
        for t in gen_ids:
            if t == EOS_TOKEN_ID or t == IM_END_ID:
                break
            new_content.append(t)
            if t == NEWLINE_ID and answer_complete is not None and answer_complete(new_content):
                break
            if rs is not None and rs.update(t):
                break
        return strip_think(tok.decode(new_content, skip_special_tokens=True))

    @torch.no_grad()
    def generate_many(items, max_new_tokens, answer_complete, rep_stop=None):
        """Batched greedy decode for non-landmark models: length-sorted, left-padded batches of at
        most --gen-batch rows and --gen-batch-tokens padded prompt tokens, via
        generate_batch (one decode step serves the whole batch, where the bs=1 loop is
        kernel-launch bound). Returns {gi: text}. The answer_complete stop is applied afterwards on
        the generated tokens, which yields the same text as stopping early."""
        out = {}
        items = sorted(items, key=lambda it: len(it[1]))
        i = 0
        while i < len(items):
            bs = 1
            while (i + bs < len(items) and bs < args.gen_batch
                   and (bs + 1) * len(items[i + bs][1]) <= args.gen_batch_tokens):
                bs += 1
            batch = items[i : i + bs]
            L = max(len(p) for _, p in batch)
            ids = torch.full((bs, L), pad_id, dtype=torch.long, device=device)
            mask = torch.zeros((bs, L), dtype=torch.bool, device=device)
            for j, (_, p) in enumerate(batch):
                ids[j, L - len(p):] = torch.tensor(p, device=device)
                mask[j, L - len(p):] = True
            gen, _, _ = gm.generate_batch(
                ids, attention_mask=mask, completions_only=True, log_timing=False,
                max_new_tokens=max_new_tokens, stop_token_ids=[EOS_TOKEN_ID, IM_END_ID],
            )
            for j, (gi, _) in enumerate(batch):
                # (the batched decode runs to EOS / max_new_tokens; the repetition stop truncates
                # afterwards, which yields the bs=1 loop's text, not its speed)
                out[gi] = finish_tokens(gen[j].tolist(), answer_complete, rep_stop)
            i += bs
        return out

    # ---- output: MERGE into an existing ladder JSON so per-task/rung splits accumulate ----
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    if os.path.exists(args.out):
        try:
            with open(args.out) as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError):
            summary = {}
    summary["model_path"] = args.model_path
    summary["variant"] = args.variant
    summary["ladder_version"] = args.ladder_version
    # Provenance (eval_size per key, eval job id, commit, ckpt, cot label) inside _meta -- see
    # ctc_eval/eval/provenance.py. Merged like the scores. Best effort: never costs the scores.
    try:
        from ctc_eval.eval.provenance import COT_MODE_LABELS, eval_meta, maybe_record_results

        meta = summary["_meta"] if isinstance(summary.get("_meta"), dict) else {}
        sizes = meta["eval_size"] if isinstance(meta.get("eval_size"), dict) else {}
        meta.update(eval_meta(args.model_path, sizes, cot_label=COT_MODE_LABELS.get(args.cot_mode),
                              suite=args.tasks.replace(",", "+"), variant=f"docchunk-{args.variant}"))
        summary["_meta"] = meta
    except Exception as e:  # noqa: BLE001
        maybe_record_results = None
        print(f"[docchunk-ladder] WARNING: provenance not recorded ({type(e).__name__}: {e})", flush=True)

    def flush():
        if is_main:
            with open(args.out, "w") as f:
                json.dump(summary, f, indent=2)

    ladders = build_ladders(args)
    task_spec = build_task_spec(args)
    task_filter = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for _t, (_, _grader, _) in SUITE_TASKS.items():
        if _t in task_filter:
            SCORE_FNS[_t] = make_ctc_score_fn(_grader)
    ctc_graded = set(ctc_graded_tasks(args))
    for _t in ctc_graded:
        SCORE_FNS[_t] = make_ctc_score_fn(CTC_GRADED_BASE[_t])
        print(f"[ladder] {_t}: graded by the ctc '{CTC_GRADED_BASE[_t]}' spec (--ctc-grade)", flush=True)
    if ctc_graded and "_meta" in summary:
        summary["_meta"]["ctc_graded"] = {t: CTC_GRADED_BASE[t] for t in sorted(ctc_graded)}
    # Opt-in repetition stop (default off): one fresh RepetitionStop per generation.
    rep_tasks = {t.strip() for t in args.repetition_stop_tasks.split(",") if t.strip()}
    rep_factory = make_repetition_stop(args.repetition_stop, args.repetition_ngram,
                                       lambda t: tok.decode([t], skip_special_tokens=True))
    rep_suffix = f"_repstop_{args.repetition_stop}" if rep_factory is not None else ""
    if rep_factory is not None:
        print(f"[ladder] repetition stop '{args.repetition_stop}' (ngram={args.repetition_ngram}) ON "
              f"for {sorted(rep_tasks)}: metric names carry '{rep_suffix}'", flush=True)
        if "_meta" in summary:
            summary["_meta"]["repetition_stop"] = {"kind": args.repetition_stop,
                                                   "ngram": args.repetition_ngram,
                                                   "tasks": sorted(rep_tasks)}
    rung_filter = set(r.strip() for r in args.rungs.split(",")) if args.rungs else None

    for task in ALL_TASKS:
        if task not in task_filter or task not in ladders:
            continue
        loadtask, chunk_by, item_regex, primary_key, max_new_tokens, stopkind = task_spec[task]
        answer_complete = make_answer_complete(stopkind)
        rep_stop = rep_factory if task in rep_tasks else None
        msfx = rep_suffix if rep_stop is not None else ""
        score_fn = SCORE_FNS[task]
        cap = args.max_length - max_new_tokens
        rungs = ladders[task]
        if rung_filter:
            rungs = [(lab, p) for (lab, p) in rungs if lab in rung_filter]

        for label, path in rungs:
            if not path or not os.path.exists(path):
                print(f"[ladder:{task}@{label}] MISSING {path}, skipping", flush=True)
                continue
            examples = load_unified_examples(
                path,
                args.max_test_samples,
                task=loadtask,
                query_position=args.query_position,
                use_alpaca=True,
            )
            # bs=1 DP shard: this rank decodes global indices [rank, rank+world, ...].
            my_gidx = list(range(rank, len(examples), world))
            local = []
            skipped = 0
            if args.gen_batch > 1 and not is_landmark:
                todo = []
                for gi in my_gidx:
                    raw = examples[gi].get("ex", examples[gi])
                    prefill = build_prefill(raw, loadtask, chunk_by, item_regex)
                    if len(prefill) > cap:
                        skipped += 1
                        local.append((gi, ""))
                    else:
                        todo.append((gi, prefill))
                t_b = time.time()
                got = generate_many(todo, max_new_tokens, answer_complete, rep_stop)
                t_b = time.time() - t_b
                local += sorted(got.items())
                if args.gen_compare > 0:  # same rows one at a time: agreement + speed check
                    sub = todo[: args.gen_compare]
                    t_1 = time.time()
                    one = {gi: generate_one(p, max_new_tokens, answer_complete, rep_stop) for gi, p in sub}
                    t_1 = time.time() - t_1
                    agree = sum(one[gi] == got[gi] for gi, _ in sub)
                    print(f"[gen-compare] {task}@{label} rank{rank}: batched == bs1 on {agree}/{len(sub)}; "
                          f"batched {t_b / max(1, len(todo)):.3f}s/ex vs bs1 {t_1 / max(1, len(sub)):.3f}s/ex", flush=True)
                    for gi, _ in sub:
                        if one[gi] != got[gi]:
                            print(f"[gen-compare]   diff gi={gi}: bs1={one[gi][:120]!r} | batched={got[gi][:120]!r}", flush=True)
                my_gidx = []
            for gi in my_gidx:
                raw = examples[gi].get("ex", examples[gi])
                prefill = build_prefill(raw, loadtask, chunk_by, item_regex)
                if len(prefill) > cap:
                    skipped += 1
                    local.append((gi, ""))  # too long for this max_length -> empty (scored wrong)
                    continue
                if is_landmark and args.landmark_top_k_fraction is not None:
                    n_blocks = max(1, len(prefill) // block_size)
                    gm.model.set_landmark_eval_top_k(
                        max(1, math.ceil(args.landmark_top_k_fraction * n_blocks))
                    )
                local.append((gi, generate_one(prefill, max_new_tokens, answer_complete, rep_stop)))

            full = [None] * len(examples)
            if world > 1:
                parts = [None] * world
                torch.distributed.all_gather_object(parts, local)
                for part in parts:
                    for gi, resp in part:
                        full[gi] = resp
            else:
                for gi, resp in local:
                    full[gi] = resp

            if is_main:
                # Keep every generation next to the result (project default: eval outputs are kept).
                resp_path = f"{os.path.splitext(args.out)[0]}_{task}_{label}_responses.jsonl"
                with open(resp_path, "w") as rf:
                    for gi, resp in enumerate(full):
                        raw = examples[gi].get("ex", examples[gi])
                        gold = {k: raw[k] for k in ("gold_doc_indices", "gold_pairs", "answers", "gold_order",
                                                    "ce_scores") if k in raw}
                        rf.write(json.dumps({"i": gi, "response": resp, **gold}) + "\n")
                res, _ = score_fn(examples, full)
                if primary_key is not None:
                    prim = res.get(primary_key)
                else:  # rerank -> first mrr* metric
                    prim = next((v for k, v in res.items() if k.startswith("mrr")), None)
                summary[f"{task}_{label}"] = float(prim) if prim is not None else None
                if "_meta" in summary:  # too-long examples are scored (as wrong), so all count
                    summary["_meta"]["eval_size"][f"{task}_{label}"] = len(examples)
                    if task in SUITE_TASKS or task in ctc_graded:  # ctc graders: parse rate too
                        summary["_meta"].setdefault("parse_rate", {})[f"{task}_{label}"] = res["parse_rate"]
                        # every metric the grader returned (tolerant reorder, gated rerank, ...)
                        summary["_meta"].setdefault("metrics", {})[f"{task}_{label}"] = {
                            k: float(v) for k, v in res.items()}
                print(
                    f"[ladder:{task}@{label}] {primary_key or 'mrr'}{msfx}="
                    f"{prim if prim is None else round(float(prim), 3)} "
                    f"(n={len(examples)}, skipped_too_long={skipped})",
                    flush=True,
                )
                # Secondary metrics in the SAME line format record_beaker_evals.py parses, after the
                # primary line: reorder's parse-tolerant tau; rerank's gated row count.
                if task in SUITE_TASKS or task in ctc_graded:
                    for k in ("kendall_tau_tolerant", "pairwise_acc_tolerant"):
                        if k in res:
                            print(f"[ladder:{task}@{label}] {k}{msfx}={round(float(res[k]), 3)} "
                                  f"(n={len(examples)}, skipped_too_long={skipped})", flush=True)
                    if "eval_size_gated" in res:
                        print(f"[ladder:{task}@{label}] {primary_key} averaged over "
                              f"{res['eval_size_gated']}/{len(examples)} rows with a CE-positive "
                              f"document (olmo-eval's skip_when_zero)", flush=True)
                if task in SUITE_TASKS or task in ctc_graded:
                    print(f"[ladder:{task}@{label}] ctc metrics: "
                          f"{ {k: round(v, 4) for k, v in res.items()} }", flush=True)
                flush()
            if world > 1:
                torch.distributed.barrier()

    flush()
    if is_main:
        print(f"\n[docchunk-ladder] WROTE {args.out} in {time.time()-t0:.1f}s", flush=True)
        if maybe_record_results is not None:
            maybe_record_results(args.out)
        keys = sorted(k for k in summary if any(k.startswith(t + "_") for t in ALL_TASKS))
        print(f"[docchunk-ladder] ladder keys present: {keys}", flush=True)
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
