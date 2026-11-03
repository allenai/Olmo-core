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
summary: Cross-row allocation does not reach 5x. At 1,952 rows/task both variants lose to the recipe at the 8k rung (step threshold -0.014 at equal FLOPs; selective backprop at q 0.3, 0.65x FLOPs: -0.10 to -0.18 vs the recipe at the same rows, ~-0.04 (sample) / ~-0.10 (top) at matched FLOPs). At 3,904 rows/task the step threshold ties (+0.006, 0.97x FLOPs) and top-50% selective backprop is +0.008 over the recipe interpolated to its 0.74x FLOPs (FTM 4.5x vs 3.8x, within noise). The router barely ranks rows (row-level Spearman 0.1-0.3).
---
# Cross-row token allocation (tokdrop_lr, 2026-11-02)

Branch `claude/cross-row-alloc` (cut from `claude/online-router-1bc01a` @ `8ce5fd22a`). Launcher
`research/token_routing/cross_row/launch_xrow.py`, state `research/token_routing/cross_row/state.json`.

## Verdict (8k rung primary; 2 seeds per cell, eval_size 500 per task/rung, no-cot)

- **Neither variant beats the recipe at 1,952 rows/task.** Step threshold: 8k mean 0.832 vs 0.846 at 0.97x its PF
  (FTM 4.0x vs 4.6x); the loss is outlier at 8k. Selective backprop at q 0.3 (0.65x PF): 0.68 (top) / 0.74 (sample)
  vs ~0.78 for the recipe interpolated to the same PF: skipping 70% of backwards discards gradient that still matters
  at this data size.
- **At 3,904 rows/task both are at best ties.** Step threshold 0.891 vs 0.885 at 0.97x PF (FTM 4.2x vs 3.8x).
  Top-50% selective backprop 0.876 at 0.74x PF (510 PF) vs the recipe interpolated to 510 PF ~0.868 (+0.008) and vs
  the recipe at the same rows 0.885 (-0.009); FTM 4.5x vs 3.8x; 2k 0.944 vs 0.947. Differences are within seed + eval
  noise (~+-0.006 per 2-seed mean).
- The selective-backprop gain grows with data (at 1,952 rows/task far below the recipe; at 3,904 a tie at 26% fewer
  FLOPs), consistent with "most rows are solved late" holding only once the data is large. It is the variant worth a
  larger-data test (7,808 rows/task), not a >=5x result.
- The step threshold's premise fails on the scorer: router scores barely rank rows by measured gradient norm
  (row-level Spearman 0.10-0.13 at stage 1, 0.30 at stage 2), so budget moves across rows mostly on noise.

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

### B. Selective backprop, 1,952 rows/task (Beaker jupiter, 1 H100, micro-batch 1)

| run | Beaker | PF | x recipe PF | contra 8k | nq 8k | outlier 8k | oolong 8k | **8k mean** | 2k mean |
|---|---|---|---|---|---|---|---|---|---|
| sbptop30 sd1 | 01M6FERDDQ0Z68VSGRBPZMVN4F | 225.2 | 0.65 | 0.926 | 0.632 | 0.610 | 0.495 | **0.666** | 0.818 |
| sbptop30 sd2 | 01M6F48N10CYB3MZ5WPW9QSJN1 | 225.0 | 0.65 | 0.924 | 0.726 | 0.509 | 0.645 | **0.701** | 0.773 |
| sbpsmp30 sd1 | 01M6FES5AY6CX28TXHH6N4W5AP | 222.8 | 0.65 | 0.901 | 0.900 | 0.644 | 0.551 | **0.749** | 0.900 |
| sbpsmp30 sd2 | 01M6FESYFQNADZYK82FJ3N1GN2 | 222.0 | 0.64 | 0.929 | 0.856 | 0.569 | 0.559 | **0.728** | 0.847 |
| recipe (2 seeds) | | 345 | 1.00 | 0.957 | 0.917 | 0.861 | 0.651 | 0.846 | 0.917 |

Both modes cost 0.65x the recipe (as designed: the FLOP meter charges skipped rows forward-only) and both lose badly.
Their 8k means (0.67-0.75) fall below the whole fresh-dense curve, so FTM is an extrapolation (0.8x-2.2x by the
collector) and not meaningful; against the recipe itself at matched FLOPs (recipe at 1,464 rows/task, older pool, 8k
mean 0.804 over 4 runs at ~259 PF; log-PF interpolation to 222 PF ~0.78), sample mode is ~-0.04 and top mode ~-0.10.
The unbiased `sample` mode beats `top` by ~0.05 at 8k and ~0.08 at 2k, so the deterministic cut's bias costs
something, but the main loss is reason (1): with 70% of backwards skipped the model trains on ~590 rows/task of
gradient, and at this data size those rows were not solved. Training CE on fresh rows (each row seen once, routed
forward, every row counted) lags from the first 50 steps: steps 150-200 recipe 0.161, xstep 0.185, top 0.228-0.238,
sample 0.243-0.253; steps 400-489 recipe 0.090, xstep 0.120, top 0.147.

Preemption note: 3 of the 4 selective-backprop runs were preempted on jupiter right after their 1 h minRuntime, twice
each, restarting from step 0 (no mid-run checkpoints); relaunched with minRuntime 2h (`XROW_MIN_RUNTIME`). The table
lists the final attempts' Beaker ids.

### 3,904 rows/task (Beaker jupiter, H100)

| run | Beaker | GPUs | PF | contra 8k | nq 8k | outlier 8k | oolong 8k | **8k mean** | 2k mean | FTM (8k) |
|---|---|---|---|---|---|---|---|---|---|---|
| xstep sd1 | 01M6FJBRK6MKFA4AD4XTXZGPXK | 2 | 666.3 | 0.972 | 0.922 | 0.979 | 0.680 | **0.888** | 0.948 | 4.0x |
| xstep sd2 | 01M6FJCPJAAXMJDQCXHT06QAHQ | 2 | 665.2 | 0.982 | 0.932 | 0.973 | 0.688 | **0.894** | 0.948 | 4.3x |
| recipe | 01M5VH6FK7J23RHG272XB796M5 | 2 | 688.1 | 0.977 | 0.918 | 0.970 | 0.663 | 0.882 | 0.949 | 3.6x |
| recipe sd1 | 01M62CQ8YR3Y2GHHRG0WENVP88 | 2 | 686.0 | 0.973 | 0.926 | 0.967 | 0.689 | 0.889 | 0.945 | 3.9x |

At 3,904 rows/task the step threshold ties the recipe: 2-seed 8k mean 0.891 vs 0.885 (+0.006) at 0.97x its PF, inside
the seed spread (recipe seeds differ by 0.007; eval SE of a 4-cell mean ~0.006); 2k identical (0.948 vs 0.947). The
outlier deficit at 1,952 rows/task is gone (0.976 vs 0.969). Row-level router predictiveness stays weak
(step_row_corr 0.10 / 0.13 at stage 1). FTM 4.0x / 4.3x vs the recipe's 3.6x / 3.9x: a hint, not a result.

| run | Beaker | GPUs | PF | x recipe PF | rows back-propagated | contra 8k | nq 8k | outlier 8k | oolong 8k | **8k mean** | 2k mean | FTM (8k) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sbptop50 sd1 | 01M6FJDXCT92FFGXBSC7JB35C3 | 1 | 512.1 | 0.74 | 49.9% | 0.967 | 0.902 | 0.966 | 0.658 | **0.873** | 0.946 | 4.35x |
| sbptop50 sd2 | 01M6FJFF6Y39384X13NDKFS2YN | 1 | 509.1 | 0.74 | 49.5% | 0.976 | 0.920 | 0.973 | 0.648 | **0.879** | 0.942 | 4.71x |

Top-50% selective backprop at 3,904 rows/task: 8k 0.876 (2 seeds) at 510 PF. The recipe interpolated in log PF between
its 1,952 and 3,904 rows/task points (0.846 @ 345, 0.885 @ 687) gives ~0.868 at 510 PF, so +0.008 at matched FLOPs;
against the recipe at the same rows it is -0.009 for 26% fewer FLOPs. 2k: 0.944 vs 0.947. Training CE on fresh rows
over steps 500-600: recipe 0.088, top-50% 0.111 / 0.095, step threshold 0.097.

## Costs

| arm | realised keep (stage 1 / stage 2 of its input) | rows back-propagated per step | actual / dense-equivalent FLOPs |
|---|---|---|---|
| recipe | 0.21-0.22 / 0.39-0.40 (log means) | 16 of 16 | ~0.22 |
| xstep | 0.19-0.21 / 0.38-0.45 (log means; PF 0.97x the recipe) | 16 of 16 | 0.213-0.214 |
| sbptop30 / sbpsmp30 | recipe's | 30.5% / 29.8-30.5% (~4.8 of 16) | 0.143-0.145 |
| sbptop50 | recipe's | 49.5-49.9% (8 of 16) | 0.163-0.164 |

Realised backprop shares (run totals incl. the full warmup step) from each run's `wandb-summary.json`
(`read_summaries_beaker.sh`, job 01M6G3YH60GWAEARN55F7VMSH5); keep shares are unweighted means of the router's
every-10-step log lines (`collect_xrow.py`); PF from each run's `flops.json`.

## Where things are

- Code: `src/olmo_core/nn/online_token_router.py` (`keep_mode="step"`, `_step_keep`, `_step_threshold`, forward-only
  pricing in `mid_compact`), `src/olmo_core/train/selective_backprop.py`, hook in
  `src/olmo_core/train/train_module/transformer/train_module.py`, flags in `train_ctc_suite.py` / `variants.py`.
- Launch / collect: `research/token_routing/cross_row/{launch_xrow.py,collect_xrow.py,read_summaries_beaker.sh,state.json}`;
  results JSON in `research/token_routing/cross_row/results/`; scores in ops (`ops result show --train-run <run>`).
- Smokes ran on jsteinhardt cubbins (H200) at `/data/prasann/ctc_suite/ckpts/*tdxsmk*` (bf16 copy of the base); all
  full runs on Beaker jupiter (H100), checkpoints at `ctc_suite/ckpts/<run>` on weka.
