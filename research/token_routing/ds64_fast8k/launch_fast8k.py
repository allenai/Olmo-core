"""
FAST **8k** screening loop for soft-token (pooled-doc KV) training recipes -- the length sibling of
``research/token_routing/ds64_fast2k/launch_fast2k.py``, whose arm vocabulary and helpers it imports rather than
copies.

WHY AN 8k SCREEN EXISTS AT ALL. fast2k screens a recipe in under an hour, and it is decisive about
whether a recipe can learn the task. It is blind to the one thing outlier actually fails at:
**length generalisation**. Measured on the ds64 ladder (records/ds64-overnight-2026-09-14.md,
09-15): ``cc00`` -- keep 0, header real, ``cent_cmean`` slot -- scores 0.80 at the 2k rung and
**0.17 at 8k**, and ``kv33`` scores 0.890 at 2k and 0.018 at 32k. A 2k screen calls both of those a
success. fast8k moves the whole loop up one rung so the failure is INSIDE the screen: training rows
come from the ds64 **8k** pool (n ~ 56 documents), and the decisive eval is the 8k rung.

WHAT IS HELD FIXED vs fast2k: same task, same generator, same repaired Qwen3.5-4B base, same
``--st-*`` flag strings, same ds64 pools, same unpacked matched-step construction, same FLOP meter,
the same eval files the ds64 ladder uses. What differs: the 8k rung instead of the 2k rung, seq-len
sized to the 8k rows, 16 rows/step instead of 32 (so an 8k step carries a comparable token count),
and an eval on the **8k rung first**, with the 2k rung kept for continuity with the fast2k table.

THE PRIMARY METRIC IS EVAL-TIME CE PARITY, NOT f1 (2026-09-15). For each trained checkpoint the
``parity`` mode runs ONE checkpoint on TWO inputs -- full real text (what the ladder eval feeds it)
vs its own soft construction -- and reports ``CE_full / CE_soft / dCE``, the same on answer DIGITS
only, and generative F1 both ways. That separates "the recipe cannot do the task" from "the recipe
can do the task but its compaction throws away the answer", which an f1 number alone cannot.

    python research/token_routing/ds64_fast8k/launch_fast8k.py --arms dense --budgets 8M,16M,20M dry_run
    python research/token_routing/ds64_fast8k/launch_fast8k.py --arms cc00,cpi17,cpi33,cpi50,kvgb50 --budgets 16M launch
    python research/token_routing/ds64_fast8k/launch_fast8k.py status
    python research/token_routing/ds64_fast8k/launch_fast8k.py eval        # 8k + 2k rungs, 500 rows each
    python research/token_routing/ds64_fast8k/launch_fast8k.py parity      # eval-time CE parity, the primary signal

mt11 runs (``--task mt11``) are scored on Beaker per task (10 jobs; rerank's suite rows are opt-in with
``--eval-rerank`` until staged, see RERANK_SUITE_ROWS_STAGED) on the
CTC-suite r8k + r2k rows staged on weka (stage_suite_rows_beaker.sh), query position ``both`` (the setA
shards' layout) and ctc-spec grading -- the same eval the Lambda path (run_eval_lambda.sbatch,
EVAL_ROWS=suite) runs. mt4 runs keep the ds64 files at query position ``after``. Preview the submit
commands without touching state.json:

    python research/token_routing/ds64_fast8k/launch_fast8k.py --task mt11 --runs <run> --eval-dry-run eval
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
D = f"{REPO}/research/token_routing/ds64_fast8k"
sys.path.insert(0, f"{REPO}/research/token_routing/ds64_fast2k")

# Reused verbatim from the fast2k loop (path-independent helpers + the arm flag vocabulary, so a
# fast2k screen and a fast8k screen of the same arm NAME are the same recipe at two lengths).
from launch_fast2k import (  # noqa: E402
    ARM_EXTRA as F2K_ARM_EXTRA,
    BASE,
    CKPTS,
    SCALE,
    TOKENIZER,
    W,
    _XHDR,
    _XSLOT,
    beaker_status,
    parse_id,
    sh,
)

LAUNCHER = f"{REPO}/recipes/train/research_trainer/beaker_ctc_suite.py"
EVAL_LAUNCHER = f"{REPO}/recipes/eval/legacy_native/run_q4b_beaker_multirung_eval.py"
PROBE_LAUNCHER = f"{REPO}/research/ffn_compute/flop_scaling/beaker_bench_launch.py"
PROBE = "research/token_routing/pooled_kv/outlier_probe/outlier_slot_probe.py"
SHARDS = f"{W}/ds64/fast8k/shards"
LEDGER = f"{D}/LAUNCH_LEDGER.tsv"
STATE = os.environ.get("F8K_STATE", f"{D}/state.json")  # F8K_STATE: drive another checkout's campaign state

#: Model family (env F8K_FAMILY, 2026-10-29 Qwen3-4B recipe study). qwen3_5 = Qwen3.5-4B (32 blocks, GDN + full
#: attention every 4th block), the campaign default. qwen3 = Qwen3-4B (36 blocks, ALL full attention): the
#: marker-repaired base (bases/qwen3-4b-base-trainedmark, donor-row repair, checked by the trainer's marker audit),
#: the NOTHINK Qwen3-4B tokenizer (stock template puts <think></think> in the answer loss span), and the same fresh
#: pools re-tokenized for Qwen3 (build_mt4_fresh_beaker.sh FAMILY=qwen3 -> mt4f<rows>q3).
FAMILY = os.environ.get("F8K_FAMILY", "qwen3_5")
SHARD_SFX = {"qwen3_5": "", "qwen3": "q3"}[FAMILY]
if FAMILY == "qwen3":
    BASE = f"{W}/ctc_suite/bases/qwen3-{SCALE}-base-trainedmark/model_and_optim"
    TOKENIZER = f"{W}/hf_tokenizers/Qwen3-4B-nothink"
#: GPUs per eval job (the tokdrop campaign used 2; 1-GPU jobs schedule faster)
EVAL_NGPU = os.environ.get("F8K_EVAL_NGPU", "2")

TASK = os.environ.get("F8K_TASK", "outlier")
BUDGETS = ["8M", "16M", "20M"]
#: rows per budget/slice -- ``compose_fast8k.py`` writes these; the build job prints them and the
#: shard's ``fast8k_stats.json`` carries the realised ``shard_num_instances``. P1/P2 are the
#: two-phase arm's DISJOINT windows: 85% then the remaining 15% of the 16M budget.
EXAMPLES = {"8M": 977, "16M": 1953, "20M": 2441, "P1": 1660, "P2": 293,
            "64M": 4 * 1953}  # mt4 = the four 16M task shards merged (merge_tasks.py)
#: Multi-task shard (task name "mt4"): one training job covers all four tasks; eval runs per task.
MT4_TASKS = ["contradiction", "nq", "outlier", "oolong"]
#: 16, not fast2k's 32: an 8k row carries ~4x the tokens of a 2k row, so 16 rows/step is ~141k real
#: tokens/step -- the same order as fast2k's 32 x 2.2k -- and it keeps the smallest budget at 61
#: steps rather than 30. Keep GLOBAL_BATCH % (micro * ngpu) == 0.
GLOBAL_BATCH = int(os.environ.get("F8K_GB", "16"))
#: 2 GPUs is enough: 4b resolves to shard_degree=world_size + FULL activation checkpointing
#: (train_ctc_suite.resolve_activation_checkpointing), so a 2-rank FSDP job holds ~1/2 of an 8 GB
#: bf16 param set, its grads and its ~32 GB fp32 AdamW state, and 2 x 12288 tokens of checkpointed
#: activations is far below the 4-GPU x 65536 the ds64 ladder already runs. Memory is not the
#: binding constraint here; queue turnaround is.
GPUS = int(os.environ.get("F8K_NGPU", "2"))
NODES = 1
#: Train-time padded row length; MUST be >= the shard's max_example_len or PadToLength silently
#: SKIPS the long rows. Set from the build job's metadata (an 8k-rung outlier row tokenizes to
#: ~1.1-1.4x its nominal length once the markers and the query are added).
SEQ_LEN = int(os.environ.get("F8K_SEQ_LEN", "12288"))
SOFT_BACKEND = os.environ.get("F8K_SOFT_BACKEND", "flash_2")
# jupiter only (2026-10-19): saturn is A100 (ask Prasann first), and Beaker GPU work goes to jupiter.
CLUSTER = os.environ.get("F8K_CLUSTER", "ai2/jupiter-cirrascale-2")
EVAL_CLUSTER = os.environ.get("F8K_EVAL_CLUSTER", CLUSTER)
WANDB_GROUP = os.environ.get("F8K_WANDB_GROUP", f"f8k-{'q35' if FAMILY == 'qwen3_5' else 'q3'}-{SCALE}")

# ---------------------------------------------------------------------------------------------
# Arm vocabulary. fast2k's arms carry over unchanged (same names, same flags); fast8k adds the
# gold_pooled_random family.
#
# ``cpi<p>`` = **gold always POOLED, p of the NON-gold bodies real** (--st-keep-mode
# gold_pooled_random, this repo's new keep mode). It is the middle ground between the two regimes
# the 2k screen separated:
#   * cc00 (keep 0) reads the slots -- 0.482 vs a 0.239 same-FLOP control -- because with no real
#     body anywhere, guessing an id is exactly chance and the slots are the only gradient left. But
#     a model that never saw a real body scores 0.17 at 8k;
#   * cc03 / cc08 / cc17 (1/36, 1/12, 1/6 of bodies real, GOLD-BLIND) all revert to id-guessing,
#     because a gold body that happens to be real still pays off.
# Pooling GOLD removes the payoff without removing real text: a visible id is never an answer, so
# copying one earns exactly chance, while the real non-gold bodies keep the model in the
# distribution it meets at eval (where every document is real).
# NB the fraction knob on the gold-sidecar path is --st-keep-frac; --st-keep-prob is accepted as an
# alias for this mode (train_ctc_suite.resolve_keep_frac) since that is how the gold-blind arms
# spell the same quantity.
_CPI = f"--st-keep-mode gold_pooled_random {_XHDR} --st-slot-mode cent_cmean {_XSLOT}"
#
# ``sc<C>`` / ``gw<K>`` = the WHOLE-CATEGORY policies (olmo_core.nn.attention.doc_categories).
# They come from a sharper reading of why gold-forcing collapsed: under ``gold_plus_random`` the
# gold category was the only category kept WHOLE -- every gold outlier real -- while other
# categories showed up as one or two scattered real documents, so "which category is complete?" was
# a content-free detector of the answer. Both policies keep whole categories ONLY, and both keep
# more than one, so completeness names nothing.
#   sc<C>  smallcat_keep, GOLD-BLIND: the C smallest categories (by the documents' own cent_cmean
#          vectors) kept whole + 1 decoy large category. The rule never reads the label.
#   gw<K>  gold_plus_wholecats: the gold document's category kept whole (as gold_plus_random did)
#          PLUS the K smallest non-gold categories, also whole.
# ⚠ NOT LAUNCHED. Held until the frozen-model probe reports whether the rule reaches eval-loss
# parity; the real-token fraction (and therefore the FLOP cost) is data-dependent and is printed
# every 50 steps by the trainer's ``[cat-keep]`` line.
_CAT = f"{_XHDR} --st-slot-mode cent_cmean {_XSLOT}"
#
# ``rk<K>`` = cc00's recipe (gold-blind, keep 0, header real, cent_cmean slot) PLUS
# ``--st-keep-token-rule rule --st-keep-token-k K``: the K highest-scoring BODY tokens of every
# pooled document stay real, chosen by a linear score over cheap token features (idf, relative
# position in the body, first-sentence, capitalisation, digit, piece length) -- no model forward,
# no gold, kept at their ORIGINAL positions so the selection is non-contiguous. This is the
# training-side port of the eval-side probe's ``rule{k}``
# (``records/outlier-saliency-preview-probe.md`` 5d), where at 8 tokens per document the SAME
# budget spent at RANDOM scores CE 0.558 -- worse than cc00's 0.470 -- the first 8 score 0.263,
# and the rule scores 0.071, matching the gradient ORACLE's 0.072. ``ck08``/``ck16`` (first-K, the
# fast2k/ds64 arms) are the exact controls: same cost, different tokens.
# ⚠ The weight vector is the documented PLACEHOLDER in
# olmo_core.nn.pooled_soft_token.DEFAULT_KEEP_TOKEN_WEIGHTS (signs and rough magnitudes from the
# measured within-document gradient profile, probe record 5a(c)), NOT the probe's fitted ridge,
# which is not reachable from this repo. Pass the fitted vector with --st-keep-token-weights once
# it is harvested off the probe job.
_RULE = f"--st-gold-blind --st-keep-prob 0.0 {_XHDR} --st-slot-mode cent_cmean {_XSLOT}"
ARM_EXTRA = dict(F2K_ARM_EXTRA)
ARM_EXTRA.update({
    "rk08": f"{_RULE} --st-keep-token-rule rule --st-keep-token-k 8",
    "rk16": f"{_RULE} --st-keep-token-rule rule --st-keep-token-k 16",
    "cpi17": f"--st-keep-frac 0.1667 {_CPI}",
    "cpi33": f"--st-keep-frac 0.3333 {_CPI}",
    "cpi50": f"--st-keep-frac 0.5 {_CPI}",
    "sc3": f"--st-keep-mode smallcat_keep --st-keep-smallcat 3 --st-keep-decoy-cats 1 {_CAT}",
    "sc5": f"--st-keep-mode smallcat_keep --st-keep-smallcat 5 --st-keep-decoy-cats 1 {_CAT}",
    "gw3": f"--st-keep-mode gold_plus_wholecats --st-keep-cats 3 {_CAT}",
    "gw5": f"--st-keep-mode gold_plus_wholecats --st-keep-cats 5 {_CAT}",
    # The devloss grid's task-general rule `gold_fl20p8_noslot` as a TRAINING construction
    # (2026-10-19, flopdrop final-figures plan, LR ablation): gold docs whole, an 8-token real prefix
    # per document (keeps the `Document [N]:` id), first_last 20% of the rest of every body, and the
    # pooled remainder DROPPED (no slot). Same model path and config keys as the grid's scheme.
    # --st-n-random 0, NOT --st-keep-frac 0.0: the fraction path floors at one random non-gold doc
    # kept whole (pooled_doc_kv: max(1, round(frac * n_non_gold))); the grid keeps gold only.
    "gfl20p8ns": ("--st-keep-mode gold_plus_random --st-n-random 0 --st-header-extra-tokens 8 "
                  "--st-keep-token-rule first_last --st-keep-token-k 0.2 --st-drop-slots"),
})
# tokdrop_lr universal-rule screen (2026-10-21/22, Lambda rounds 3-7; flags verbatim from those launches).
# tier<D>... = gold + D% random docs form the keep set, which gets --st-kept-doc-k of its body (two-tier)
# instead of staying whole; every other body keeps --st-keep-token-k, floored at --st-keep-token-min tokens.
_P8 = "--st-header-extra-tokens 8 --st-drop-slots"
_FL = "--st-keep-token-rule first_last"
_GB = "--st-gold-blind --st-keep-prob 0.0"
def _gr(frac):
    return f"--st-keep-mode gold_plus_random --st-keep-frac {frac}"
ARM_EXTRA.update({
    "tier67a50m32": f"{_gr(0.6667)} {_FL} --st-keep-token-k 0.05 --st-kept-doc-k 0.5 --st-keep-token-min 32 {_P8}",
    "tier67flm16": f"{_gr(0.6667)} {_FL} --st-keep-token-k 0.05 --st-kept-doc-k 0.3 --st-keep-token-min 16 {_P8}",
    "tier67a50m64": f"{_gr(0.6667)} {_FL} --st-keep-token-k 0.05 --st-kept-doc-k 0.5 --st-keep-token-min 64 {_P8}",
    # pad-free dense wall-clock reference: the soft-token path (strips padding, length-sorts micro-batches) with
    # EVERY document kept whole -- same tokens and maths as dense; its scores double as a check that it IS dense
    "densepf": "--st-gold-blind --st-keep-prob 1.0 --st-drop-slots",
    # ds64's per-task Pareto winners, verbatim from research/token_routing/ds64/launch_ds64.py (2026-10-22 per-task
    # reproduction in the fast8k setting: single-task g16M shards, 8k rows, unpacked, 16 rows/step)
    "chk16": ("--st-gold-blind --st-keep-prob 0.0 --st-header-stop-id 25 --st-header-stop-count 1 --st-header-extra-tokens 16 "
              f"--st-slot-mode cent_cmean --st-slot-tokenizer {W}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    "chk64": ("--st-gold-blind --st-keep-prob 0.0 --st-header-stop-id 25 --st-header-stop-count 1 --st-header-extra-tokens 64 "
              f"--st-slot-mode cent_cmean --st-slot-tokenizer {W}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    "flk64": ("--st-gold-blind --st-keep-prob 0.0 --st-header-stop-id 5491 --st-header-stop-count 1 --st-keep-token-rule first_last "
              f"--st-keep-token-k 64 --st-slot-mode cent_cmean --st-slot-tokenizer {W}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    "ck64ds": ("--st-gold-blind --st-keep-prob 0.0 --st-header-stop-id 5491 --st-header-stop-count 1 --st-header-extra-tokens 64 "
               f"--st-slot-mode cent_cmean --st-slot-tokenizer {W}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    "ohdr33": "--st-gold-blind --st-keep-prob 0.3333 --st-header-stop-id 25 --st-header-stop-count 3",
    # online train-time router (olmo_core.nn.online_token_router, merged 2026-10-22): keep<%> M<split block>;
    # gradient-norm target, 10% exploration, per-document allocation, a charged full-context step every 8
    **{f"otr{k:02d}M{L}": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 "
                          f"--st-otr-keep-frac {k / 100} --st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 "
                          f"--st-otr-dense-every 8 --st-otr-drop-layer {L}")
       for k in (10, 30) for L in (7, 8)},
    "otr10M7gold": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.1 "
                    "--st-otr-explore-frac 0.1 --st-otr-doc-share 0.5 --st-otr-dense-every 8 --st-otr-drop-layer 7 "
                    "--st-otr-use-gold"),
    "otr10M7de16": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.1 "
                    "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 16 --st-otr-drop-layer 7"),
    # warmup-then-freeze, no full steps: fa<P>M7 freezes embeddings + blocks 0-6 after P% of the run's steps
    # (FA_STEPS fills in the step count per budget at launch)
    **{f"fa{pc}M7": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.1 "
                     "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
                     "--st-otr-freeze-after {FA_STEPS}") for pc in (50, 75)},
    # overnight 10-24 ideas runnable on router 67a28d605+ (see the summary page): two-stage rescoring (20% from block 7,
    # rescored to 7% at block 16), bounded-peak periodic steps (every-8 / every-16 steps keep 50%, not full context),
    # and the router + fixed layer skip in the routed stack
    "r20to7M7": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
                 "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 16 --st-otr-drop-layer 7 "
                 "--st-otr-stages 7:0.2,16:0.07"),
    **{f"otr10M7de{e}dk50": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.1 "
                             f"--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every {e} --st-otr-drop-layer 7 "
                             "--st-otr-dense-keep-frac 0.5") for e in (8, 16)},
    "otr10M7de16sk3": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.1 "
                       "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 16 --st-otr-drop-layer 7 "
                       "--train-skip-layers 20,24,28"),
    # per-task router (one ridge fit per prompt template; router 56c96377d) and mixture-of-depths (alternate blocks
    # 8..30 run on c of each document's tokens, the rest pass the residual through; full-row loss)
    **{f"otr10M7de16f16{t}": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.1 "
                              "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 16 --st-otr-drop-layer 7 "
                              "--st-otr-min-per-doc 16" + (" --st-otr-per-task-key-tokens 11" if t else ""))
       for t in ("", "pt")},
    "otr10M7de16pt": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.1 "
                      "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 16 --st-otr-drop-layer 7 "
                      "--st-otr-per-task-key-tokens 11"),
    **{f"mod{c}": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 "
                   f"--st-otr-keep-frac {c / 100} --st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 "
                   "--st-otr-drop-layer 8 --st-otr-mod-layers 8-31:2") for c in (25, 50)},
    # progressive freeze (router bf671531e): embeddings + blocks [0,3) frozen at 25% of steps, [0,7) at 50%; no full steps
    "pfz37M7": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.1 "
                "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
                "--st-otr-freeze-schedule 3:0.25,7:0.5"),
    # NO full-context steps anywhere (Prasann 2026-10-24: "avoid mixing with dense steps completely"): de0 versions of
    # the overnight ideas
    "r20to7M7de0": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
                    "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
                    "--st-otr-stages 7:0.2,16:0.07"),
    "otr10M7de0sk3": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.1 "
                      "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
                      "--train-skip-layers 20,24,28"),
    **{f"otr10M7de0{v}": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.1 "
                          "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7"
                          + x) for v, x in (("pt", " --st-otr-per-task-key-tokens 11"), ("f16", " --st-otr-min-per-doc 16"),
                                             ("f16pt", " --st-otr-min-per-doc 16 --st-otr-per-task-key-tokens 11"))},
    # later-depth target: with plain drop + de0 the target layer never fits (only full steps do), so pair it with MoD
    "mod25t16": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.25 "
                 "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 8 "
                 "--st-otr-mod-layers 8-31:2 --st-otr-target-layer 16"),
    # router 076464355: adaptive keep by predicted gradient mass (80%, per-row cap 20%) and a LoRA early stack
    "mass80c20M7": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.1 "
                    "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
                    "--st-otr-mass-target 0.8 --st-otr-max-keep-frac 0.2"),
    "lora16M7": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.1 "
                 "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
                 "--st-otr-early-lora 16 --st-otr-early-lora-scale 32"),
    "lora64M7": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.1 "
                 "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
                 "--st-otr-early-lora 64 --st-otr-early-lora-scale 32"),
    # LoRA r64 from step 0, then adapters + blocks 0-6 frozen at 75% of the run (router 5e1202e4b+; best 10-24 point)
    "lora64fa75M7": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.1 "
                     "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
                     "--st-otr-early-lora 64 --st-otr-early-lora-scale 32 --st-otr-freeze-after {FA_STEPS}"),
    # best 10-24 combination: LoRA r64 + freeze at 75% + two-stage keep (20% at 7 -> 7% at 16), no full steps
    "lora64fa75r20to7": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
                         "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
                         "--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
                         "--st-otr-freeze-after {FA_STEPS}"),
    **{f"lora64fa75{n}": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
                          "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
                          f"--st-otr-stages {st} --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
                          "--st-otr-freeze-after {FA_STEPS}")
       for n, st in (("r20to10", "7:0.2,16:0.10"), ("r3s", "7:0.2,16:0.07,24:0.03"))},
    # universal recipe (10-24): LoRA r64 on blocks 0-6, NO freeze, two-stage keep, no full steps
    **{f"lora64{n}": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
                      "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
                      f"--st-otr-stages {st} --st-otr-early-lora 64 --st-otr-early-lora-scale 32")
       for n, st in (("r20to7", "7:0.2,16:0.07"), ("r20to10", "7:0.2,16:0.10"))},
    # partial MLP (router 2cfcd92d1+) on top of the universal recipe; whole-model LoRA + router (--st-otr-early-lora-blocks)
    **{f"lora64r20to7ffn{f}{tag}": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
                                    "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
                                    "--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
                                    f"--ffn-width-frac {f / 100}" + (f" --ffn-width-layers {lay}" if lay else ""))
       for f in (50, 25) for tag, lay in (("e", "0-6"), ("a", ""))},
    "lora64all32r20to7": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
                          "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
                          "--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
                          "--st-otr-early-lora-blocks 32"),
    # hill-climb winner (router session, 3 seeds mt4s1464: 0.864 vs 0.852): stage-2 adaptive budget = each doc's top
    # tokens covering 70% of predicted gradient mass, capped at 50% of the stage input (needs router 8ce5fd22a+)
    **{f"lora64all32m70c50{t}": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
                                 "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
                                 "--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
                                 "--st-otr-early-lora-blocks 32 --st-otr-later-mass-target 0.7 --st-otr-later-max-keep-frac 0.5"
                                 + x) for t, x in (("", ""), ("f16", " --st-otr-min-per-doc 16"))},
    # early first stage (Prasann: drop before the first full-attention layer at block 3), recipe otherwise; and a
    # pyramid that keeps 50% from block 3 before the usual 20% / 7%
    **{f"lora64all32{tag}": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
                             "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
                             f"--st-otr-stages {st} --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
                             "--st-otr-early-lora-blocks 32")
       for tag, st in (("r20at1", "1:0.2,16:0.07"), ("r20at3", "3:0.2,16:0.07"), ("r20at5", "5:0.2,16:0.07"),
                       ("p50at3", "3:0.5,7:0.2,16:0.07"))},
    # Qwen3-4B (36 blocks, all full attention; F8K_FAMILY=qwen3): the recipe with LoRA on all 36 blocks and the drop
    # points at proportional depth (7/32 -> 8/36, 16/32 -> 18/36); q3r20at{4,12} move the first stage (split sweep)
    **{f"q3lora64all36r20at{a}": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
                                  f"--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer {a} "
                                  f"--st-otr-stages {a}:0.2,18:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
                                  "--st-otr-early-lora-blocks 36")
       for a in (4, 8, 12)},
    # Qwen3-0.6B / 1.7B (28 blocks): the 4B winner 12/24 of 36 scaled by depth -> 9/19 of 28, LoRA on all 28 blocks
    "q3lora64all28r20at9s19": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
                               "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 9 "
                               "--st-otr-stages 9:0.2,19:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
                               "--st-otr-early-lora-blocks 28"),
    # deeper split pairs (q3lora64all36r20at<a>s<b>: 20% from block a, 7% from block b); at12 beat at8/at4 on f1952
    **{f"q3lora64all36r20at{a}s{b}": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
                                      f"--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer {a} "
                                      f"--st-otr-stages {a}:0.2,{b}:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
                                      "--st-otr-early-lora-blocks 36")
       for a, b in ((12, 24), (16, 24), (16, 27))},
    # 2x2 control on Qwen3: the best split (12 -> 24) WITHOUT LoRA (full fine-tune under routing), cf. r20to7M7de0
    "q3r20at12s24": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
                     "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 12 "
                     "--st-otr-stages 12:0.2,24:0.07"),
    # second stage at block 12 instead of 16 (same budgets)
    "lora64r20to7s12": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
                        "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
                        "--st-otr-stages 7:0.2,12:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32"),
    # review control: whole-model LoRA + two-stage with RANDOM token choice (same budgets / allocation / drop points)
    "lora64all32rnd20to7": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
                            "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
                            "--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
                            "--st-otr-early-lora-blocks 32 --st-otr-score random"),
    # 2x2 ablation (LoRA x routing): two-stage keep without LoRA + per-doc floor 16 (11-task)
    "r20to7M7de0f16": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
                       "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
                       "--st-otr-stages 7:0.2,16:0.07 --st-otr-min-per-doc 16"),
    "lora64all32r20to7f16": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
                             "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
                             "--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
                             "--st-otr-early-lora-blocks 32 --st-otr-min-per-doc 16"),
    "otr10M7de4": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.1 "
                   "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 4 --st-otr-drop-layer 7"),
    # no full-context steps (peak memory = routed steps), optionally + early-exit LM loss on dropped tokens
    # (xl<W*10>; needs router commit 8537ebd2f+)
    **{f"otr10M7de0{s}": ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.1 "
                          "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7"
                          + (f" --st-otr-exit-loss-weight {w}" if w else ""))
       for s, w in (("", 0), ("xl3", 0.3), ("xl10", 1.0))},
    "tier67a60m48": f"{_gr(0.6667)} {_FL} --st-keep-token-k 0.05 --st-kept-doc-k 0.6 --st-keep-token-min 48 {_P8}",
    "tier67flm32": f"{_gr(0.6667)} {_FL} --st-keep-token-k 0.05 --st-kept-doc-k 0.3 --st-keep-token-min 32 {_P8}",
    "tier50flm32": f"{_gr(0.5)} {_FL} --st-keep-token-k 0.05 --st-kept-doc-k 0.4 --st-keep-token-min 32 {_P8}",
    "tier33fl": f"{_gr(0.3333)} {_FL} --st-keep-token-k 0.1 --st-kept-doc-k 0.5 {_P8}",
    "gbflk32p8": f"{_GB} {_FL} --st-keep-token-k 32 {_P8}",
    "gbflk64p8": f"{_GB} {_FL} --st-keep-token-k 64 {_P8}",
    "gbfl10m32p8": f"{_GB} {_FL} --st-keep-token-k 0.1 --st-keep-token-min 32 {_P8}",
})
# Train-time whole-block skipping (olmo_core.nn.layer_drop, 2026-10-22): a skipped block's forward is never
# called during TRAINING (eval runs every block); the flop meter charges only blocks that ran. Qwen3.5-4B has 32
# blocks, full attention at 3,7,...,31; the sets below are GDN blocks in the back half, where the 2-GPU smoke
# showed skipping 20,24,28 costs nothing in train CE (0.347 vs 0.351) while LayerDrop p=0.25 uniform wrecked it.
_SK3 = "--train-skip-layers 20,24,28"
_SK8 = "--train-skip-layers 16,17,18,20,21,22,24,25"
_LDL = "--layer-drop-p 0.2 --layer-drop-schedule linear --layer-drop-exclude-first 4 --layer-drop-exclude-last 2"
#: dense + a train-time modifier (variant full with extra flags)
DENSE_EXTRA = {
    # dense with the LM head only at supervised positions (exact; meter charges it so from router d74ad1194)
    "densesp": "--sparse-loss-positions --sparse-loss-positions",  # repeated: a single dash-token string is parsed as a launcher option
    # review controls (2026-10-26): dense + whole-model LoRA, no router (is the LoRA-all gain LoRA or routing?)
    "denselora64all": "--lora-rank 64 --lora-blocks 0 --lora-scale 32",
    "denseffn50": "--ffn-width-frac 0.5",
    "denseffn25": "--ffn-width-frac 0.25","dense-sk3": _SK3, "dense-sk8": _SK8, "dense-ldl20": _LDL}
for _r in ("tier67a50m32", "tier67flm16", "gbflk64p8"):
    ARM_EXTRA[f"{_r}-sk3"] = f"{ARM_EXTRA[_r]} {_SK3}"
    ARM_EXTRA[f"{_r}-sk8"] = f"{ARM_EXTRA[_r]} {_SK8}"
    ARM_EXTRA[f"{_r}-ldl20"] = f"{ARM_EXTRA[_r]} {_LDL}"
ARM_MICRO = {}          # every arm at micro 2: the FLOP audit (09-15 10:30) requires identical
DEFAULT_MICRO = 2       # gpus/micro across any pair that will be compared at matched FLOPs.

#: Two-phase arms. ``<arm>-warm`` trains ``<arm>``'s recipe from ANOTHER fast8k run's exported
#: weights: the trainer loads --base-checkpoint with load_optim_state=False /
#: load_trainer_state=False, so phase 2 gets a fresh LR schedule and a fresh budget and phase 1's
#: flags do not carry over. No trainer support beyond --base-checkpoint is needed.
#: ``kvgb50-warm`` on the P2 slice is the "cc00 for 85% of the budget, then real bodies for the
#: last 15%" arm, and P1/P2 are disjoint windows of the 16M rows so phase 2 is not a second epoch.
WARM_FROM = {"kvgb50-warm": ("kvgb50", f"{TASK}-cc00-P1"),
             "cpi33-warm": ("cpi33", f"{TASK}-cc00-P1")}

# --- eval-time CE parity -----------------------------------------------------------------------
# The soft construction the probe runs beside FULL real text, written as the TRAINER's own flags
# with commas for spaces (outlier_slot_probe.py --construction; commas so the spec survives a
# launcher that splits argv on whitespace).
#
# ⚠ These are the GOLD-BLIND analogues of the cpi arms, not their literal training construction:
# parse_construction REFUSES --st-keep-frac, because the probe implements the gold-blind keep_prob
# path and the gold-only keep, not the gold-sidecar policies. That is the right thing to measure
# anyway -- at eval there is no gold sidecar, so "keep p of the bodies real, blind" is the
# construction a cpi arm would actually be DEPLOYED under, and the pooled-set shortcut a cpi arm
# could learn in training (see collect_fast8k.pooled_floor) is not available here either.
_SLOT = "--st-header-stop-id,5491,--st-header-stop-count,1,--st-slot-mode,cent_cmean"
CONSTRUCTIONS = {
    # everything pooled -- the COMMON maximal-compaction reference, run for every arm so dCE is
    # comparable across arms and across budgets
    "cc00": f"--st-gold-blind,--st-keep-prob,0.0,{_SLOT}",
    "gb17c": f"--st-gold-blind,--st-keep-prob,0.1667,{_SLOT}",
    "gb33c": f"--st-gold-blind,--st-keep-prob,0.3333,{_SLOT}",
    "gb50c": f"--st-gold-blind,--st-keep-prob,0.5,{_SLOT}",
    "kvgb50": "--st-gold-blind,--st-keep-prob,0.5",     # plain mean slot, no header
}
#: arm -> the constructions to score it under. First entry is the one promoted into results.csv.
PARITY_ARMS = {"dense": ["cc00"], "cc00": ["cc00"], "kvgb50": ["kvgb50", "cc00"],
               "cpi17": ["gb17c", "cc00"], "cpi33": ["gb33c", "cc00"], "cpi50": ["gb50c", "cc00"],
               # a whole-CATEGORY construction is not reproducible by outlier_slot_probe's
               # keep_prob/gold-only conditions, so these are scored against the common
               # maximal-compaction reference only.
               "sc3": ["cc00"], "sc5": ["cc00"], "gw3": ["cc00"], "gw5": ["cc00"],
               # the keep-token rule selects WITHIN a document, which is not one of
               # outlier_slot_probe's conditions either -- score against the common reference only
               "rk08": ["cc00"], "rk16": ["cc00"],
               # gold-forced + within-document selection: not an outlier_slot_probe condition either
               "gfl20p8ns": ["cc00"]}
PARITY_RUNGS = os.environ.get("F8K_PARITY_RUNGS", "8k,32k")
PARITY_ROWS = os.environ.get("F8K_PARITY_ROWS", "200")
PARITY_GEN_ROWS = os.environ.get("F8K_PARITY_GEN_ROWS", "32")


#: Default LR of every fast2k/fast8k/ds64 run. A run at any other --lr carries an `-lr<value>` tag,
#: so an LR ablation never collides with (or silently resumes into) the default-LR run.
DEFAULT_LR = 5e-6
LR_TAG = ""
#: Campaign suffix (env F8K_SUFFIX, e.g. "-lra"): gives a campaign fresh run names AND fresh weka
#: save folders, so it can never auto-resume into an older run's checkpoint. When set, the LR tag is
#: always added (default LR included), so every run in the campaign is named alike.
SUFFIX = os.environ.get("F8K_SUFFIX", "")


def run_name(task, arm, budget):
    return f"f8k-{task}-{arm}-{budget.replace('@', 'st')}{SUFFIX}{LR_TAG}"  # s488@31 -> s488st31


def resolve(arm):
    """(base recipe arm, base checkpoint) for ``arm`` -- resolves the ``-warm`` suffix."""
    if arm in WARM_FROM:
        recipe, src = WARM_FROM[arm]
        return recipe, f"{CKPTS}/f8k-{src}/model_and_optim"
    return arm, BASE


#: tasks per multi-task shard (merge_tasks.py), for the ``s<rows>`` budgets below.
MT_TASKS = {"mt4": MT4_TASKS,
            "mt11": ["nq", "hotpotqa", "qdmatch_nq", "outlier", "oolong", "contradiction", "xabsence",
                     "reorder", "rerank", "strmatch", "grouping"]}


def parse_budget(task, budget):
    """``8M``/``16M``/... -> the ``<task>_g<budget>`` pool shard (``16M@31``: cut at 31 steps; ``16M_eg``: the oolong eg pool). ``s<rows>[@<steps>]`` -> the merged
    first-N shard ``<task>s<rows>`` (rows per task; mt4s488 is a prefix of mt4s976, of mt4s1464), optionally
    cut at ``<steps>`` optimizer steps (the dense matched-FLOP curve). Returns (data dir, examples, max_steps)."""
    if budget.startswith("f") and budget[1:2].isdigit():  # f<rows>: first-N of the FRESH pools (build_mt4_fresh_beaker.sh)
        rows, _, steps = budget[1:].partition("@")
        return f"{SHARDS}/{task}f{rows}{SHARD_SFX}", int(rows) * len(MT_TASKS.get(task, [task])), int(steps or 0)
    if budget.startswith("s"):
        rows, _, steps = budget[1:].partition("@")
        rows = rows.split("x")[0]  # s1952x2 = two epochs over mt4s1952 (see budget_epochs)
        n = int(rows) * len(MT_TASKS.get(task, [task]))
        return f"{SHARDS}/{task}s{rows}", n, int(steps or 0)
    base, _, steps = budget.partition("@")  # 16M@31 = the g16M pool cut at 31 steps (per-task dense curve)
    return f"{SHARDS}/{task}_g{base}", EXAMPLES.get(base.split("_")[0], 0), int(steps or 0)


def budget_epochs(budget):
    """``s<rows>x<E>`` trains E epochs over the first-N shard (more data for cheap arms once fresh rows run out)."""
    m = re.fullmatch(r"s\d+x(\d+)(?:@\d+)?", budget)
    return int(m.group(1)) if m else 1


def arm_args(task, arm, budget):
    recipe, base = resolve(arm)
    data, _, max_steps = parse_budget(task, budget)
    micro = ARM_MICRO.get(recipe, DEFAULT_MICRO)
    if GLOBAL_BATCH % (micro * GPUS * NODES):
        raise SystemExit(f"global batch {GLOBAL_BATCH} not divisible by micro {micro} x {GPUS * NODES} GPUs")
    common = ["--seq-len", str(SEQ_LEN), "--global-batch", str(GLOBAL_BATCH),
              "--micro-batch-instances", str(micro), "--base-checkpoint", base]
    if max_steps:
        common += ["--max-steps", str(max_steps)]
    if recipe == "dense":
        # NOT --pack: every arm trains unpacked so rows/step and step counts match exactly.
        return "full", data, common, ""
    if recipe in DENSE_EXTRA:
        return "full", data, common, DENSE_EXTRA[recipe]
    if recipe in ARM_EXTRA:
        extra = ARM_EXTRA[recipe]
        if "{FA_STEPS}" in extra:  # freeze after a fixed share of this budget's optimizer steps
            _, n, cut = parse_budget(task, budget)
            total = cut or budget_epochs(budget) * -(-n // GLOBAL_BATCH)
            extra = extra.replace("{FA_STEPS}", str(total * int(re.search(r"fa(\d+)", recipe).group(1)) // 100))
        return "softtoken", data, common, f"{extra} --attn-backend {SOFT_BACKEND}"
    raise SystemExit(f"unknown arm {arm} (known: dense, {', '.join(ARM_EXTRA)}, "
                     f"plus the -warm variants {', '.join(WARM_FROM)})")


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}, "evals": {}, "parity": {}}


def save_state(st):
    """Write state, MERGED with what is on disk under a lock: two launcher processes (e.g. a launch and a watcher's
    eval sweep) each hold an older copy, and a plain overwrite silently dropped the other's new runs (2026-10-30)."""
    import fcntl
    st.setdefault("parity", {})
    with open(STATE + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        disk = json.load(open(STATE)) if os.path.exists(STATE) else {}
        for kind in ("runs", "evals", "parity"):
            merged = dict(disk.get(kind, {}))
            merged.update(st.get(kind, {}))  # this process's entries win for keys it holds
            st[kind] = merged
        json.dump(st, open(STATE + ".tmp", "w"), indent=1)
        os.replace(STATE + ".tmp", STATE)


def _log(name, mode, out):
    os.makedirs(f"{D}/launch_logs", exist_ok=True)
    open(f"{D}/launch_logs/{name}.{mode}.log", "w").write(out)


def do_launch(args, mode):
    st = load_state()
    rows = []
    for arm in args.arms.split(","):
        for budget in args.budgets.split(","):
            variant, data, largs, extra = arm_args(args.task, arm, budget)
            # F8K_EXTRA: trainer args appended to every arm, e.g. "--save-interval 250" so a preempted Beaker job resumes
            # (a retry restarts from scratch without a checkpoint: 4B dense f7808 lost 1884/1952 steps, 2026-10-30)
            extra = f"{extra} {os.environ.get('F8K_EXTRA', '')}".strip()
            # --seed N: a seed replicate named <arm>sd<N> (collectors strip sd<N> and average seed copies)
            label = f"{arm}sd{args.seed}" if args.seed else arm
            name = run_name(args.task, label, budget)
            if name in st["runs"] and mode == "launch" and not args.force:
                print(f"[skip] {name} already launched ({st['runs'][name].get('ex')})")
                continue
            cmd = [sys.executable, "-u", LAUNCHER, "--task", args.task, "--variant", variant,
                   "--model-family", FAMILY, "--model-scale", SCALE, "--data-root", data,
                   "--run-name", name, "--exact-run-name", "--num-nodes", str(NODES),
                   "--num-gpus", str(GPUS), "--epochs", str(budget_epochs(budget)), "--lr", str(args.lr),
                   "--cluster", CLUSTER, "--wandb-group", args.wandb_group, "--priority", "urgent",
                   "--no-follow", "--no-compile"] + (["--seed", str(args.seed)] if args.seed else []) + largs + (["--extra-args", extra] if extra else []) + [mode]
            print(" ".join(cmd), flush=True)
            rc, out = sh(cmd)
            _log(name, mode, out)
            ex = parse_id(out)
            tail = "\n".join(out.strip().splitlines()[-2:]).replace("\t", " ")[:200]
            print(f"  -> rc={rc} ex={ex}: {tail[:200]}", flush=True)
            if mode == "launch":
                st["runs"][name] = {"ex": ex, "task": args.task, "arm": arm, "budget": budget,
                                    "steps_expected": (parse_budget(args.task, budget)[2]
                                                       or budget_epochs(budget) * -(-parse_budget(args.task, budget)[1] // GLOBAL_BATCH)),
                                    "gpus": GPUS * NODES, "state": "S" if ex else "LAUNCH-FAILED"}
                save_state(st)
                rows.append((args.task, arm, budget, name, rc, (ex or "") + " " + tail))
    if rows:
        with open(LEDGER, "a") as f:
            w = csv.writer(f, delimiter="\t")
            for task, arm, budget, name, rc, tail in rows:
                w.writerow(["fast8k", SCALE, task, arm, budget, name, CLUSTER,
                            dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "LAUNCHED" if rc == 0 else "LAUNCH-FAILED", tail])
    print(f"\nwandb: https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/{args.wandb_group}")


#: 8k FIRST (the decisive rung -- it is where the outlier soft arms die) and 2k for continuity with
#: the fast2k table. Same files the ds64 orchestrator's ladder uses.
RUNGS = {"outlier": {"8k": f"{W}/outlier_lengthmix/eval_rungs/outlier/rung_8192.jsonl",
                     "2k": f"{W}/outlier_lengthmix/eval_rungs/outlier/rung_2048.jsonl"},
         "nq": {"8k": f"{W}/outlier_lengthmix/eval_rungs/nq/rung_8192.jsonl",
                "2k": f"{W}/outlier_lengthmix/eval_rungs/nq/rung_2048.jsonl"},
         # 2026-10-19 (LR ablation): the ds64 orchestrator's files for the other two tasks.
         "contradiction": {
             "8k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n190_k3.jsonl",
             "2k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl"},
         "oolong": {
             "8k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx8192_spliteval.jsonl",
             "2k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx2048_spliteval.jsonl"}}
TASK_KEY = {"nq": "nq", "outlier": "outlier", "oolong": "oolong", "contradiction": "contra"}
#: mt11: tasks scored on the CTC-suite rows (run_q4b_beaker_multirung_eval.py --suite-rows)
SUITE_EVAL_TASKS = ["nq", "hotpotqa", "qdmatch_nq", "outlier", "oolong", "contradiction", "xabsence",
                    "reorder", "strmatch", "grouping"]
#: rerank's suite rows (2026-10-25, ctc-graded ce_pos_recall) join SUITE_EVAL_TASKS once they are staged
#: on weka (stage_suite_rows_beaker.sh, whose fetch now includes rerank). Until then a rerank eval job
#: fails at REQUIRE_EVAL_SIZE, so it is opt-in: --eval-rerank (or flip this after staging).
RERANK_SUITE_ROWS_STAGED = False
SUITE_ROW_RUNS = {"mt11"}
#: batched greedy decode, as run_eval_lambda.sbatch (validated vs bs=1: contradiction@8k f1 0.641 vs 0.639)
SUITE_EVAL_ENV = ["LADDER_GEN_BATCH=32", "LADDER_GEN_BATCH_TOKENS=196608"]


def eval_cmd(ename, task, ckpt, suite):
    """The run_q4b_beaker_multirung_eval.py command scoring ``task`` of checkpoint ``ckpt``.

    ``suite`` (mt11): the CTC-suite rows at query position ``both`` with ctc-spec grading; else the ds64
    files (RUNGS) at query position ``after`` (mt4 / single-task runs, unchanged)."""
    cmd = [sys.executable, "-u", EVAL_LAUNCHER, ename, EVAL_CLUSTER,
           "--task", TASK_KEY.get(task, task), "--variant", "docchunk", "--ckpt", ckpt,
           "--query-position", "both" if suite else "after", "--cot-mode", "none",
           "--tokenizer", TOKENIZER, "--ngpu", EVAL_NGPU, "--max-test", "500", "--priority", "urgent",
           # >= the 8k rung's prompt + generation, with headroom: a --max-length BELOW the
           # prompt truncates it away and the run scores exactly 0.000 (records/ memory
           # "Goldgrad eval MAXLEN truncation bug"). 20000 covers an 8k-rung outlier row.
           "--max-length", "20000", "--batch-size", "8"]
    if suite:
        return cmd + ["--suite-rows"] + [a for kv in SUITE_EVAL_ENV for a in ("--env", kv)]
    return cmd + ["--dc-rung-files", json.dumps({task: RUNGS[task]}), "--dc-rungs", ",".join(RUNGS[task])]


def do_eval(args):
    st = load_state()
    names = args.runs.split(",") if args.runs else list(st["runs"])
    dry = args.eval_dry_run
    for name in names:
        r = st["runs"].get(name)
        if r is None and dry:  # preview a run that is not (yet) in state.json
            r = {"task": args.task, "state": "DONE", "steps_expected": 0}
        if r is None:
            print(f"[skip] {name} not in state.json")
            continue
        if st["evals"].get(name, {}).get("ex") and not args.force and not dry:
            print(f"[skip] eval {name} already launched ({st['evals'][name]['ex']})")
            continue
        # Only score a FINISHED run: the eval reads the model-only export written after fit(), so
        # launching against a still-training save folder finds no checkpoint and burns a GPU slot
        # failing.
        if (r.get("state") != "DONE" and not args.force and not dry
                and not (args.eval_step_dir and r.get("state") == "FAILED")):
            print(f"[wait] {name} train={r.get('state')} -- eval deferred")
            continue
        # A multi-task (mt4/mt11) checkpoint is scored once PER TASK, as `<run>-ev<task>`.
        # mt11 trains on setA rows (IID with the CTC suite, query position both): score it on the suite
        # rows (rerank only with --eval-rerank / RERANK_SUITE_ROWS_STAGED: its rows must be staged first).
        suite = r["task"] in SUITE_ROW_RUNS
        tasks = SUITE_EVAL_TASKS if suite else MT_TASKS.get(r["task"], [r["task"]])
        if suite and (RERANK_SUITE_ROWS_STAGED or args.eval_rerank):
            tasks = tasks + ["rerank"]
        if args.eval_tasks:
            tasks = [t for t in tasks if t in args.eval_tasks.split(",")]
        for task in tasks:
            ename = f"{name}-ev{task}" if r["task"] in MT_TASKS else name
            if ename != name and st["evals"].get(ename, {}).get("ex") and not args.force and not dry:
                print(f"[skip] eval {ename} already launched ({st['evals'][ename]['ex']})")
                continue
            # --eval-step-dir: score the trainer's final distcp step dir instead of the post-fit export (the
            # 10-22 tdb runs finished every step, then died in wandb.finish before the export was written)
            ckpt = f"{CKPTS}/{name}/step{r['steps_expected']}" if args.eval_step_dir else f"{CKPTS}/{name}"
            cmd = eval_cmd(ename, task, ckpt, suite)
            if dry:
                rc, out = sh(cmd + ["--dry-run"])
                print(f"eval {ename} [dry-run rc={rc}]: " + " ".join(
                    ln.strip() for ln in out.splitlines() if ln.strip().startswith("cmd:")), flush=True)
                continue
            rc, out = sh(cmd)
            ex = parse_id(out)
            _log(ename, "eval", out)
            st["evals"][ename] = {"ex": ex, "state": "S" if ex else "LAUNCH-FAILED", "train_run": name,
                                  "task": task}
            save_state(st)
            print(f"eval {ename} -> {ex or 'FAILED: ' + out[-300:]}", flush=True)


def do_parity(args):
    """Eval-time CE parity: ONE trained checkpoint, TWO inputs (full real text vs its own soft
    construction), on the 8k rung and -- where affordable -- 32k.

    This is the primary signal. A ladder f1 conflates "the checkpoint cannot do outlier" with "the
    checkpoint can do outlier but the compaction destroys the answer"; running the same weights on
    both inputs separates them, and ``CE on the answer DIGITS`` isolates the id choice from the
    output format. Submitted through the shared 1-GPU bench launcher so it uses the same image and
    the same PUSHED commit the training jobs do.
    """
    st = load_state()
    names = args.runs.split(",") if args.runs else list(st["runs"])
    for name in names:
        r = st["runs"].get(name)
        if r is None:
            print(f"[skip] {name} not in state.json")
            continue
        if r.get("state") != "DONE" and not args.force:
            print(f"[wait] {name} train={r.get('state')} -- parity deferred")
            continue
        arm = r["arm"]
        conds = args.parity_arms.split(",") if args.parity_arms else PARITY_ARMS.get(
            arm.replace("-warm", ""), ["cc00"])
        for cond in conds:
            key = f"{name}::{cond}"
            if st.setdefault("parity", {}).get(key, {}).get("ex") and not args.force:
                print(f"[skip] parity {key} already launched ({st['parity'][key]['ex']})")
                continue
            spec = CONSTRUCTIONS.get(cond, cond)
            extra = [
                # --trained-parity implies --no-reset-projector and scores every rung in ONE
                # process, so the 4B checkpoint is loaded once.
                # ``--construction=<spec>``, never ``--construction <spec>``: the spec itself
                # starts with "--", and argparse refuses an option value that looks like another
                # option ("expected one argument"). The "=" form always parses.
                "--trained-parity", "--arm", "none", f"--construction={spec}",
                "--rungs", args.parity_rungs, "--rows", args.parity_rows,
                "--gen-rows", args.parity_gen_rows,
                "--ckpt", f"{CKPTS}/{name}", "--ckpt-name", name,
                "--tokenizer", TOKENIZER, "--tag", f"f8k-{cond}",
                "--work", "/tmp/f8k_probe_work",
                # the stop set MUST come from the TRAINING shard, or the eval-side cent_cmean slot
                # is not the slot the arm trained with
                "--slot-stop-shard", f"{SHARDS}/{r['task']}_g{r['budget']}",
            ]
            cmd = [sys.executable, "-u", PROBE_LAUNCHER, "--cluster", EVAL_CLUSTER,
                   "--script", PROBE, "--gpus", "1", "--priority", "urgent",
                   "--extra", " ".join(extra)]
            rc, out = sh(cmd)
            ex = parse_id(out)
            _log(f"{name}.{cond}", "parity", out)
            st["parity"][key] = {"ex": ex, "run": name, "cond": cond,
                                 "state": "S" if ex else "LAUNCH-FAILED"}
            save_state(st)
            print(f"parity {key} -> {ex or 'FAILED: ' + out[-400:]}", flush=True)


def do_status(args):
    st = load_state()
    changed = False
    for kind in ("runs", "evals", "parity"):
        for name, r in st.get(kind, {}).items():
            if not r.get("ex") or r.get("state") in ("DONE", "FAILED"):
                continue
            s, rc = beaker_status(r["ex"])
            if s == "F":
                r["state"] = "DONE" if rc == 0 else "FAILED"
                r["rc"] = rc
            elif s != "?":
                r["state"] = s
            changed = True
    if changed:
        save_state(st)
    for name, r in sorted(st["runs"].items()):
        e = st["evals"].get(name, {})
        p = [f"{k.split('::')[1]}={v.get('state')}" for k, v in sorted(st.get("parity", {}).items())
             if v.get("run") == name]
        print(f"{name:32} train={str(r.get('state')):12} {r.get('ex','')}  "
              f"eval={str(e.get('state','-')):12} {e.get('ex','')}  parity[{' '.join(p)}]")
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--arms", default="dense")
    ap.add_argument("--budgets", default=",".join(BUDGETS))
    ap.add_argument("--lr", type=float, default=DEFAULT_LR,
                    help="peak LR; any non-default value tags run names with -lr<value> (all modes)")
    ap.add_argument("--wandb-group", default=WANDB_GROUP)
    ap.add_argument("--runs", default="", help="eval/parity: comma run names (default = all)")
    ap.add_argument("--parity-arms", default="", help="override the PARITY_ARMS conditions")
    ap.add_argument("--parity-rungs", default=PARITY_RUNGS)
    ap.add_argument("--parity-rows", default=PARITY_ROWS)
    ap.add_argument("--parity-gen-rows", default=PARITY_GEN_ROWS)
    ap.add_argument("--force", action="store_true", help="relaunch even if state.json has it")
    ap.add_argument("--eval-rerank", action="store_true",
                    help="eval (mt11): also score rerank on its suite rows (needs them staged on weka: "
                    "stage_suite_rows_beaker.sh); default once RERANK_SUITE_ROWS_STAGED is True")
    ap.add_argument("--eval-tasks", default="",
                    help="eval: comma list restricting which per-task eval jobs are launched (e.g. rerank)")
    ap.add_argument("--eval-dry-run", action="store_true",
                    help="eval: build each job with the launcher's --dry-run (nothing submitted, state.json "
                    "untouched; a run not in state.json is previewed as a --task run)")
    ap.add_argument("--eval-step-dir", action="store_true",
                    help="eval: read <run>/step<steps_expected> (no post-fit export); also scores FAILED runs")
    ap.add_argument("--seed", type=int, default=0, help="launch: training seed (run named <arm>sd<seed>; 0 = default)")
    ap.add_argument("mode", choices=["launch", "dry_run", "eval", "parity", "status"])
    args = ap.parse_args()
    global LR_TAG
    LR_TAG = "" if (args.lr == DEFAULT_LR and not SUFFIX) else f"-lr{args.lr:g}"
    if args.mode in ("launch", "dry_run"):
        do_launch(args, args.mode)
    elif args.mode == "eval":
        do_eval(args)
    elif args.mode == "parity":
        do_parity(args)
    else:
        do_status(args)


if __name__ == "__main__":
    main()
