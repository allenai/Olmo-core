---
title: Oracle token selection in TRAINING — per-step gradient-chosen token sets as an upper bound for token dropping
date: 2026-11-04
updated: 2026-11-04
experiment: tokdrop_lr
area: token_routing
kind: result
status: in progress (7 arms on Beaker at f1952; smoke findings below)
superseded_by: 
artifact: 
summary: Every step a dense backward picks, per stage of the recipe's 7:0.2/16:0.07 drop, the smallest top-k body-token set whose EXACT LoRA-gradient attribution cos >= 0.95 (per doc / row / batch), and the routed pass on that set is applied. Smoke: non-body tokens alone give cos ~0.96 of the whole gradient; on the body part the set is ~10-22% / 4-5% of body tokens, yet its routed gradient has cos only ~0.86.
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

## Arms (Beaker jupiter, urgent, 2 GPUs, f1952)
| arm | scope | view | extra |
|---|---|---|---|
| orcdoc | each document's top k% | body | |
| orcrow | each row's top k% | body | |
| orcbatch | batch-global top k% | body | |
| orcbatchr2 | batch | body | + random body tokens = 2x the oracle set per row and stage |
| orcdocr2 | doc | body | + random 2x |
| orcbatchall | batch | all | the literal whole-gradient rule |
| orcbatchrc | batch | body | grow k (~2x per try) until the ROUTED LoRA-gradient cos >= 0.95 |

## Smoke findings (jsteinhardt mooney, 1 H200, steps 1-3 from the base; mt4s1952 rows, no gold sidecar)
- Exactness: bins-vs-gradient cos >= 0.99997.
- Cost per step on 1 H200 (16 rows): dense A 16 s, dense B + selection 21 s, routed 8-10 s (routed-target arm
  ~20 s for 2 tries). Routed pass = 0.18-0.37 of a dense pass's FLOPs.
- Body-view k for cos 0.95 at step 1 (stage 1 / stage 2): per doc 22% / 5%, per row 13% / 5%, batch 10% / 4%.
- Routed gradient vs dense gradient (LoRA): the per-doc attribution set (22/5%) gives routed cos 0.86; growing to
  55% / 13% reaches 0.97; at steps 2-3 the routed-target search settled at 75% / 17%.
- Literal `all` view: k = grid floor; routed cos 0.62-0.68 raw at ~1% kept.

## Results
(in progress)

## Metadata (per run: `<weka>/ctc_suite/ckpts/<run>/oracle_meta/`)
`steps.jsonl` (one line per step) + `sel/step*_rank*.npz` (replayable selections, energies); fields listed in
`research/token_routing/oracle_train/README.md`.
