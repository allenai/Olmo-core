---
title: Multi-view training from a cached block-7 state (k late-stack views per row, one early-stack pass)
date: 2026-10-30
updated: 2026-10-30
experiment: tokdrop_lr
area: token_routing
kind: result
status: in progress (2 of 5 runs scored; mv3-f1952, mv2/mv3-f3904 training)
superseded_by:
artifact: https://claude.ai/artifact/Y15xcuaVDVhz737nw9g5VZ
summary: Run blocks 0-6 once per row, cache h7, take k optimizer steps on blocks 7-31 each with a fresh random 20%/7% token subset, then one early-stack backward on the mean dL/dh7. Cost per row 0.10 + 0.12k x dense; goal = beat the recipe at matched FLOPs by >= 0.01 (mt4 8-cell mean).
---

# Multi-view training from a cached block-7 state

Overnight idea (2026-10-30, Prasann: "try some ambitious ideas ... much higher FLOP speedups relative to dense").
Branch `claude/multiview-h7cache` (off `claude/online-router-1bc01a`).

## Idea

The recipe (`lora64all32r20to7`: whole-model LoRA r64 over a frozen Qwen3.5-4B base + two-stage online router, keep
20% after block 7 and 7% after block 16, LR 1e-5) costs 0.225x dense per step, and blocks 0-6 at full length are about
45% of that. Random token choice ties the router (0.914 vs 0.915 at f3904), so different random views of a row are
cheap to draw. Multi-view (`--st-otr-views k`):

1. View 1: the ordinary routed forward, with the graph split at block 7's input: h7 is cached detached, the late
   stack runs from a fresh leaf; its backward reaches blocks 7-31 + head and dL/dh7. The early graph stays alive.
2. Views 2..k: blocks 7-31 + head only, from the cached h7, each with a fresh random subset at both router stages;
   an optimizer step on the late parameters between views (clip, LR schedule, AdamW).
3. One backward through blocks 0-6 with the MEAN of the k views' dL/dh7, then the trainer's ordinary step (early
   adapters from that mean, late ones from view k).

**Early stack: mean of the views, not last view only.** Each view's dL/dh7 is an unbiased draw of the early-stack
gradient for a different token subset; averaging k of them cuts the subset-sampling variance at no extra FLOPs (the
early backward is paid once either way), while the last view alone throws k - 1 of them away. The early parameters
do not move during the k views, so the view-1 graph is exact for them; the only staleness is that view v's dL/dh7 was
taken with late parameters v - 1 steps newer, the same order as gradient-accumulation staleness at LR 1e-5.

Cost per row (x dense per row): 0.10 + 0.12k (k = 1 is the recipe, 0.225). Risk: the k views share answer labels, so
the late stack sees each row's labels k times, a partial multi-epoch.

## Implementation (`src/olmo_core/nn/multiview.py`)

- `enable_multiview(train_module, k)` swaps `train_batch` (wired from `SoftTokenOptions.st_otr_views`, CLI
  `--st-otr-views`). Dense (warmup) router steps and the dry-run batch take the ordinary single-view path.
- Model (`Transformer._run_blocks`): `model._mv_ctx` per forward; the first view splits at the router's first drop
  layer, a repeat view skips blocks [0, L) and starts from the cached state.
- FLOP meter: the repeat views log no per-example lengths (so `dense_pflops` counts every row once) and the router's
  stage-0 `mid_compact` charges them `kept x (late blocks + head)` with nothing below the split and no dense
  subtraction; later stages keep their relative formula. Unit test `src/test/nn/multiview_test.py` checks the
  split/repeat forwards reproduce the plain forward's loss and gradients for the same draw, the repeat-view FLOP
  charge, one meter entry per row, and k - 1 inner steps for the late parameters / none for the early ones.
- **FSDP2 trap (the reason the first three smokes died):** a backward that starts from an intermediate block output
  instead of the loss never fires the ROOT module's FSDP pre-backward (its hook sits on the model output), and the
  root's own parameters are in the early graph: the soft-token `pooled_projector` runs before block 0 and its output
  is written into the embeddings, so block 0's input requires grad. Its saved weight view was resharded, and the
  deferred backward died with `setStorage: ... out of bounds for storage of size 0`. Reproduced on CPU (gloo,
  world 1, tiny olmo2 + AC + FSDP2: `research/token_routing/multiview/fsdp_cpu_repro.py`; even a plain model with no
  multi-view fails the same way when backpropagating from block 0's output alone). Fix (1f57a5feb): a tensor hook on
  h_L runs `FSDPState._pre_backward` for the root and every early block at the start of the deferred backward
  (idempotent for modules whose own hook fires); their post-backward / the final callback reshard and reduce-scatter.
  The AC wrapper sits outside the fully_shard-ed block, so `model.blocks[i]` is not itself an FSDPModule.

## Smokes (2026-10-30, Beaker jupiter, 2 H100, 6 steps of mt4f1952, step 1 dense warmup)

| run | views | actual PF | dense-equiv PF | peak mem / rank |
|---|---|---|---|---|
| mvsmoke-bkr-k1-1f57a5feb (01M6609ZM49GJBEFVA5MPEVNDY) | 1 | 5.981 | 19.694 | - |
| mvsmoke-bkr-1f57a5feb (01M65ZKX3574TAZZFNMCTCNN9M) | 2 | 7.438 | 19.694 | 29.2 GiB |

Same data, same dense-equivalent (every row charged once); the second view adds 1.456 PF over the 5 routed steps =
0.089x dense per routed step, vs 0.165x for a whole k = 1 routed step on these rows, i.e. the late stack is ~54% of a
routed step, as planned (0.12 of 0.225). Expected full-run cost: k = 2 about 1.54x the recipe's PF, k = 3 about 2.08x
(f1952: ~530 / ~715 PF; f3904: ~1,060 / ~1,430 PF).

## Runs (Beaker jupiter, 2 H100, commit 1f57a5feb, state `research/token_routing/multiview/state.json`)

| run | views | rows/task | actual PF | 8-cell mean | recipe @ same PF | delta |
|---|---|---|---|---|---|---|
| f8k-mt4-lora64all32rnd20to7mv2-f1952-tdmv-lr1e-05 (01M660BDJE2ZQA3YE2AQQS20FA) | 2 | 1,952 | 483.0 | 0.885 | 0.899 | -0.014 |
| ...mv3-f1952 (01M660CBN2GSBT9AQ954ZB1HGK) | 3 | 1,952 | running | | | |
| ...mv1-f1952 (01M6662Z9T213KKGJC8M3ZA215), same-code-path k = 1 control | 1 | 1,952 | 343.4 | 0.853 | 0.883 (extrap.) | -0.030 |
| ...mv2-f3904 (01M660D44KG044R94YW2CYDC8T) | 2 | 3,904 | running | | | |
| ...mv3-f3904 (01M660DX5J61G6CSZ21MXAF1X3) | 3 | 3,904 | running | | | |

eval_size 500 per cell, no-cot, query after; cells (8k / 2k) for mv2-f1952: contradiction 0.954 / 0.984, nq 0.930 /
0.978, outlier 0.887 / 0.867, oolong 0.639 / 0.844 (recipe f1952: 0.959 / 0.977, 0.912 / 0.970, 0.862 / 0.880,
0.648 / 0.856). Single seed; run-to-run noise of the recipe is about +-0.01 on the mean.

mv1-f1952 cells (8k / 2k): contradiction 0.954 / 0.977, nq 0.926 / 0.974, outlier 0.812 / 0.690, oolong
0.645 / 0.848.

First read (2 of 5 runs): vs the router recipe, k = 2 trails by 0.014 at matched FLOPs. Vs its own same-code-path
k = 1 control (random choice) it gains +0.032 for 1.41x the FLOPs, which is about the recipe's gain from doubling
rows (+0.032 for 2x). The two readings disagree because the random-choice k = 1 control is 0.030 below the router
recipe at the same rows, nearly all on outlier (2k 0.690 vs 0.880), which looks like single-seed noise or a
random-choice weakness at f1952 (random tied the router at f3904). A seed replicate of mv1/mv2 at f1952 would decide.

Earlier note: the second view of each row is worth almost nothing (+0.002 over the recipe at the same rows for 1.40x
the FLOPs), so at matched FLOPs it trails the recipe's fresh rows.

## Comparison target

mt4 8-cell mean (4 tasks x {8k, 2k} rungs, 500 rows each, no-cot). Recipe 0.883 / 0.915 / 0.924 at 345 / 688 /
1,373 PF (f1952 / f3904 / f7808); fresh dense 0.888 / 0.926 / 0.928 / 0.931 at 1,531 / 3,063 / 4,595 / 6,128 PF.
Decisive: beat the recipe, interpolated (linear in log PF) at the multi-view run's own `actual_pflops`, by >= 0.01.
