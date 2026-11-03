---
title: Cross-row token allocation for token-drop training (step-level threshold, selective backprop)
date: 2026-11-02
updated: 2026-11-02
experiment: tokdrop_lr
area: token_routing
kind: result
status: current
superseded_by:
artifact:
summary: Two ways to let the token budget move across rows in the token-drop recipe -- one router threshold per optimizer step, and skipping the backward for low-answer-loss rows (selective backprop). RESULTS PENDING (runs in flight).
---
# Cross-row token allocation (tokdrop_lr, 2026-11-02)

Branch `claude/cross-row-alloc` (cut from `claude/online-router-1bc01a` @ `8ce5fd22a`). Launcher
`research/token_routing/cross_row/launch_xrow.py`, state `research/token_routing/cross_row/state.json`.

## Why

The gradient-concentration probe (branch `claude/grad-concentration-d91a37`) found the one-step update concentrates
across ROWS: for gradient cosine 0.95 a batch-global token pick needs 0.47% of body tokens, a per-row quota 4.6%, a
per-document quota 13.7%. Late in training most rows are solved (at 7,808 rows/task the median row answer CE is 0.001
and the top 10% of rows hold 81% of the loss). The recipe (whole-model LoRA r64 + online router, keep 20% of each
document's body after block 7 and 7% after block 16) allocates per document; the older `keep_mode threshold` spans
only one 2-row micro-batch and tied the recipe.

## Design

Both arms are the recipe (`lora64all32r20to7`: LR 1e-5, 16 rows/step, Qwen3.5-4B markerfix base, unpacked seq 12288)
with one change.

**A. Step-level threshold (`--st-otr-keep-mode step`, arm `xstep`).** One router-score threshold per optimizer step,
shared by all 16 rows on both ranks, in each pyramid stage.
- Threshold: after each refit, the new weights score a uniform body-token sample from the previous step (its
  exploration tokens; on the full-context warmup step every sampled token, stage 1 only). The sample is pooled over
  ranks (all-gather, 4,096 per rank) and the threshold is its (1 - a) quantile.
- Budget: `a` (the admitted share) is an integral controller, `a += keep_frac - realised` per step (realised =
  kept / body over the step, all ranks). The accumulated error is bounded, so the run-average keep matches 20% / 7% and
  FLOPs match the recipe's while hard rows keep more and easy ones less.
- Always kept: the exploration draw (10% of each row's budget), a per-document floor of 2 tokens. Per-row cap 3x the
  row's budget (forced tokens first, then by score).
- Fit: targets centred over the whole step on all ranks (not per row), so scores can carry across-row difficulty.
- Diagnostic `step_row_corr`: Spearman, across this rank's rows in a step, of the row-mean router score vs the
  row-mean measured log gradient norm on the uniform tokens (does the router rank ROWS at all?).

**B. Selective backprop (`--st-otr-sbp-mode top|sample`, arms `sbptop30`, `sbpsmp30`).** Every row runs the recipe's
routed forward; only rows with high answer loss run the backward (`olmo_core.train.selective_backprop`).
- Unit = one micro-batch index across ranks (FSDP all-gathers parameters in every backward, so all ranks must run
  or skip the same backward). Run on 1 GPU with micro-batch 1, so the unit is one row (16 decisions per step).
- Score = the row's summed answer CE.
- `top`: back-propagate iff the score is >= the 0.7 quantile of the last 48 rows' scores (biased: assumes skipped rows
  contribute ~0 gradient). `sample`: back-propagate with p = clip(0.3 x score / window mean, 0.05, 1) and weight 1/p
  (unbiased; seeded identically on every rank). Warmup step 1 back-propagates every row; a step that reaches its last
  row with no backward runs that row.
- FLOPs: the router computes, per forward, the cost if the backward is skipped (every block, adapter and the head at
  1/3 of forward + backward); a skipped row is charged that. With a frozen LoRA base the backward is ~one forward, so
  a routed step costs about (1 + q) / 2 of the recipe's: 0.65x at q = 0.3.

## Smoke tests (jsteinhardt cubbins, H200, 2026-11-02)

| run | steps | GPUs | actual PF | dense-equiv PF | notes |
|---|---|---|---|---|---|
| xstep (first version) | 6 | 2 | 5.73 | 19.51 | stage 2 kept 91% of its input on the first routed step (bug, below) |
| xstep (fixed) | 12 | 2 | 9.68 | 38.16 | stage 1 keep 0.18-0.22, stage 2 0.49 -> 0.22 -> 0.39 by step 10 (target 0.35) |
| sbptop30 | 6 | 1 | 4.51 | 19.51 | 31% of rows back-propagated per routed step, 0 forced |
| sbpsmp30 | 6 | 1 | 4.56 | 19.51 | 31% back-propagated, mean 1/p weight 2.8 |

Same 96 rows in every smoke. Excluding the full-context warmup step (~2.2 PF), a selective-backprop step costs ~0.63x
a recipe step (expected 0.65x).

Bug found and fixed (`bc82f3324`): a later stage's threshold reference must be the tokens ENTERING it on a routed step;
the full-context warmup step sends every token into stage 2, so the first routed step kept 91% instead of 35%. Later
stages now ignore full-context samples, and their first routed micro-batch sets the threshold.

Unit tests (`src/test/nn/online_token_router_test.py` step-mode tests, `src/test/train/selective_backprop_test.py`):
98 passed (4 GPU-only skipped) on horton CPU.

## Results

Recipe references (fresh pools, LR 1e-5, eval_size 500 per cell, no-cot): at 1,952 rows/task two seeds, 8k mean
0.845 / 0.847 (2k 0.921 / 0.913) at 345 PF, FTM 4.6x; at 3,904 rows/task 0.882 / 0.889 (2k 0.949 / 0.945) at 688 PF.
Fresh dense 8k means 0.843 / 0.899 / 0.901 / 0.904 at 1531 / 3063 / 4595 / 6128 PF.

### A. Step-level threshold, 1,952 rows/task (Beaker jupiter, 2 H100, micro-batch 2)

| run | Beaker | PF | contra 8k | nq 8k | outlier 8k | oolong 8k | **8k mean** | 2k mean | FTM (8k) |
|---|---|---|---|---|---|---|---|---|---|
| xstep sd1 | 01M6F4EK4CTMX299D8HP7B159N | 333.8 | 0.964 | 0.904 | 0.821 | 0.647 | **0.834** | 0.905 | 4.0x |
| xstep sd2 | 01M6F4FDS9B2RE08SVEY28RPQ8 | 332.9 | 0.962 | 0.888 | 0.809 | 0.663 | **0.831** | 0.924 | 4.0x |
| recipe (2 seeds) | | 345 | 0.957 | 0.917 | 0.861 | 0.651 | 0.846 | 0.917 | 4.6x |

At matched FLOPs (0.97x the recipe's PF) the step threshold is 0.014 below the recipe at 8k, against a recipe seed
spread of 0.002; almost all of it is outlier at 8k (-0.04 to -0.05). Diagnostics (seed 2, steps 20-310, logged every 10
steps): the budget does move across rows (per-row keep CV 0.80 at stage 1, 0.49 at stage 2) but the router barely
ranks rows: Spearman of row-mean score vs row-mean measured log gradient norm is 0.13 at stage 1, 0.30 at stage 2. So
budget moves on noise, and losing the recipe's per-document share (each document keeps 20%) costs outlier, which needs
every document represented.
