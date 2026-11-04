---
title: Oracle token selection in TRAINING — per-step gradient-chosen token sets as an upper bound for token dropping
date: 2026-11-04
updated: 2026-11-04
experiment: tokdrop_lr
area: token_routing
kind: result
status: headline single-cut doc + batch done; row, random-2x, routed-target and two-cut arms in progress
superseded_by: 
artifact: 
summary: Per-step oracle token selection in training (dense backward picks the smallest top-k body set with exact LoRA-gradient attribution cos >= 0.95, routed pass applied), single cut at block 7, f1952. Per-document oracle (~36% of body tokens kept) ties dense at 8k (0.850 vs 0.843); batch-global oracle (~19%) loses outlier (8k 0.537 vs 0.817; mean 0.756). Routed-gradient cos only ~0.5-0.6 for both.
---
# Oracle token selection in training

Branch `claude/oracle-train-bd7414`. Code: `src/olmo_core/nn/oracle_token_select.py` (3-pass step),
`--st-otr-score oracle` in `src/olmo_core/nn/online_token_router.py`, flags `--oracle-*` in
`recipes/train/research_trainer/train_ctc_suite.py`; launcher + README + summariser in
`research/token_routing/oracle_train/`.

## Question
Prasann (2026-11-04): oracle gradient upper bounds at document, row and batch level. Each step: dense backward,
use it to pick the tokens (a selection that keeps cos 0.95 with the gradient signal), rerun forward/backward with
only that computation, and apply that. Record as much metadata as possible. One or two runs also keep 2x extra
random computation: are "negatives" needed for the right gradient?

## Method (per optimizer step, 16 rows)
1. Dense pass A (not applied). Per-token gradient energy in the trainable LoRA space,
   `s^2 (|delta_t|^2 |A x_t|^2 + |B^T delta_t|^2 |x_t|^2)`, summed over the adapted linears of blocks [7, 16)
   (stage 1) and [16, 32) (stage 2). The dense gradient is saved.
2. Dense pass B. Each token's exact rank-1 LoRA gradient term is added into the accumulator of its energy-rank
   bin (grid of body-token shares 0.05% ... 100%, ranked within each document / row / the whole batch).
   Prefix sums give the exact attribution gradient of every top-k set. Check: the summed bins equal the true
   dense LoRA gradient (cos 0.99997-0.999999 in every smoke).
3. Per stage, the smallest grid k with attribution cos >= 0.95 (stage 1 = its own top-k ∪ stage 2, nested).
4. Routed pass C: the router keeps exactly that set at the block-7 and block-16 cuts (non-body tokens -- prompt,
   markers, query, answer -- always kept). This gradient is applied (Adam, clipping, LR schedule unchanged).

Base: the recipe `lora64all32r20to7` (repaired Qwen3.5-4B, whole-model LoRA r64 scale 32, LR 1e-5, 2 GPUs,
16 rows/step, mt4 fresh pool f1952 = 488 steps). Only the selection differs.

**Two attribution views.** `all`: G = the whole gradient. `body` (default): only the body tokens' part of the
gradient (the always-kept tokens' terms left out of both sides). Reason (smoke, step 1-3): the always-kept
non-body tokens (4.4% of real tokens) ALONE give attribution cos 0.94 (stage 1) / 0.98 (stage 2) with the whole
LoRA gradient, so under `all` the rule is met at the grid floor with ~0.25% of body tokens -- a selection that
says nothing about which context tokens matter. `orcbatchall` runs the literal rule anyway. Correction from the full
run (Beaker, step 7): the domination is an early-training effect (LoRA B starts at zero, so only the dL/dB terms
exist); by step 7 the `all` view already needs 10% / 1.5% of body tokens (batch scope), close to the body view's
10% / 3% at step 4 of `orcbatch`.

## Arms (f1952, 488 steps, 2 GPUs)
HEADLINE (parent session / Prasann 2026-11-04): ONE cut at block 7 (a token is kept or dropped in every block from 7 on; blocks
0-6 dense), cos over the blocks after the cut. Secondary: the recipe's two-cut shape (7 + 16 nested).

| arm | cut | scope | view | extra | where |
|---|---|---|---|---|---|
| orc1doc | 7 | each document's top k% | body | | jsteinhardt mooney (H200; local mt4f1952 copy, no gold sidecar) |
| orc1batch | 7 | batch-global top k% | body | | mooney |
| orc1row | 7 | each row's top k% | body | | Beaker (+ insurance copy on mooney) |
| orc1batchr2 | 7 | batch | body | + random body tokens = 2x the oracle set per row | Beaker (+ Lambda copy) |
| orc1docr2 | 7 | doc | body | + random 2x | Beaker (+ Lambda copy) |
| orc1batchrc | 7 | batch | body | grow k (~2x per try) until ROUTED LoRA-grad cos >= 0.95 | Beaker (+ mooney copy) |
| orcdoc / orcbatch | 7 + 16 | doc / batch | body | | Beaker (both preempted once, restarted from step 0) |
| orcbatchall | 7 + 16 | batch | all | the literal whole-gradient rule | Beaker (preempted once) |

Jupiter preempted 3 runs after their 1 h minRuntime; with no mid-run checkpoint they restart from step 0 (their
steps.jsonl holds both passes; `summarize.py` keeps the last). Insurance copies run where nothing preempts them.

## Smoke findings (jsteinhardt mooney, 1 H200, steps 1-3 from the base; mt4s1952 rows, no gold sidecar)
- Exactness: bins-vs-gradient cos >= 0.99997.
- Cost per step on 1 H200 (16 rows): dense A 16 s, dense B + selection 21 s, routed 8-10 s (routed-target arm
  ~20 s for 2 tries). Routed pass = 0.18-0.37 of a dense pass's FLOPs.
- Body-view k for cos 0.95 at step 1 (stage 1 / stage 2): per doc 22% / 5%, per row 13% / 5%, batch 10% / 4%.
- Routed gradient vs dense gradient (LoRA): the per-doc attribution set (22/5%) gives routed cos 0.86; growing to
  55% / 13% reaches 0.97; at steps 2-3 the routed-target search settled at 75% / 17%.
- Literal `all` view: k = grid floor; routed cos 0.62-0.68 raw at ~1% kept.

## Results

### Downstream (4-task 8k + 2k rungs, eval_size 500 per cell, no-cot, query after; f1)
| run | contra 8k | nq 8k | outlier 8k | oolong 8k | **mean 8k** | contra 2k | nq 2k | outlier 2k | oolong 2k | mean 2k |
|---|---|---|---|---|---|---|---|---|---|---|
| dense (f1952, lr 5e-6) | 0.973 | 0.924 | 0.817 | 0.656 | **0.843** | 0.987 | 0.980 | 0.911 | 0.853 | 0.933 |
| recipe lora64all32r20to7 (router) | 0.959 | 0.912 | 0.862 | 0.648 | **0.845** | 0.977 | 0.970 | 0.880 | 0.856 | 0.921 |
| orc1doc (oracle, per document) | 0.973 | 0.930 | 0.828 | 0.669 | **0.850** | 0.985 | 0.978 | 0.789 | 0.856 | 0.902 |
| orc1batch (oracle, batch-global) | 0.963 | 0.900 | 0.537 | 0.626 | **0.756** | 0.982 | 0.966 | 0.442 | 0.828 | 0.804 |

SE per cell ±0.007-0.022 (binomial, eval_size 500). Single seed each.

### Training-time metadata (means over 488 steps; `summarize.py`)
| run | body tokens kept | attr cos (body) | routed-grad cos raw / Adam | routed/dense grad norm | train CE dense -> routed | routed-pass FLOPs / dense | actual 3-pass / dense |
|---|---|---|---|---|---|---|---|
| orc1doc | 35.5% (39% -> 30% over training) | 0.959 | 0.58 / 0.66 | 1.25 | 0.148 -> 0.151 | 0.38 | 2.38 |
| orc1batch | 19.1% (13% -> 26%) | 0.960 | 0.52 / 0.55 | 1.24 | 0.166 -> 0.121 | 0.29 | 2.29 |

### Reading
- **Per-document allocation is what matters, not the token count.** The per-document oracle ties dense at 8k
  (0.850 vs 0.843, within noise; outlier 2k is lower, 0.789 vs 0.911). The batch-global oracle -- half the tokens,
  the same attribution cos 0.96 on the body gradient -- breaks outlier (8k 0.537, 2k 0.442) and drops oolong.
  Batch-global k concentrates on the few rows / documents with the largest gradient energy and leaves most
  documents with nothing, which outlier (find the off-topic document) cannot survive.
- **Matching the attribution gradient does not match the routed gradient**, in either arm: routed cos ~0.5-0.6
  all run, routed grad norm 1.25x dense, no loss spikes. The batch oracle's routed train CE falls far BELOW dense
  (0.121 vs 0.166): the model fits the oracle-chosen subset, an easier task than the full context.
- **Not a speedup**: picking the set costs two dense passes (2.3-2.4x dense per step). The upper bound says a
  per-document 36% keep set exists that trains as well as dense; the recipe's router (20% / 7%) already ties dense
  at this budget.
- Concentration: these keep shares are much larger than the frozen-probe table's (1.96 / 8.1 / 16.9% batch / row /
  doc at 1,952). Two reasons: the cos here is on the body tokens' part of the LoRA gradient only (the probe's
  includes the always-kept tokens, which alone give ~0.96 here), and the oracle is measured on the training
  trajectory it shapes.

## Metadata (per run: `<weka>/ctc_suite/ckpts/<run>/oracle_meta/`)
`steps.jsonl` (one line per step) + `sel/step*_rank*.npz` (replayable selections, energies); fields listed in
`research/token_routing/oracle_train/README.md`.
