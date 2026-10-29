---
title: Train-time layer skipping and gradient-ranked layer selection on the fast8k mt4 screen
date: 2026-10-28
updated: 2026-10-28
experiment: tokdrop_lr
area: token_routing
kind: result
status: complete (screen); fresh-pool FLOP curves not run (awaiting Prasann)
superseded_by: 
artifact: https://claude.ai/artifact/SPSBxHFs2PUn5CPjJX3Wgt
summary: Fixed / random / gradient-ranked whole-block skipping on Qwen3.5-4B, mt4 fast8k, matched FLOPs. Dense - skip 20-27, LayerDrop on 16-29 and update-only top-25% ties dense at 0.78-0.89x FLOPs over 2 seeds (s1464 and f3904; no clear win). Forward-skip of the top 25-50% collapses. On the recipe (whole-model LoRA + two-stage router) layer skipping never helps - it saves at most 8% FLOPs there.
---
# Layer skipping on the fast8k mt4 screen

Prasann 2026-10-28: "could you just try layer skipping?", then "keep only the x% of layers with the most gradient
signal, with and without token skipping".

## Verdict (seed 0 / seed 1; Δ = score minus dense at matched training PF)

- **Dense + layer skipping ties dense at 0.78-0.89x FLOPs; nothing is clearly better.** Best two-seed means:
  skip 20-27 (0.79x) +0.024 / -0.034 on s1464, +0.004 / +0.009 on f3904; LayerDrop on 16-29 (0.89x) +0.007 / +0.021;
  update-only top-25% static (0.79x) +0.031 / +0.003. Noise is about ±0.02-0.03 per cell.
- **Forward-skip of the gradient-selected top 25-50% collapses** (contradiction@8k → 0.00): training through 8-16 of
  32 blocks and evaluating through all 32 breaks the model. The static 50% set is -0.11 to -0.12, the online one -0.5.
- **The recipe gains nothing from layer skipping.** Its base is frozen behind LoRA and its later blocks already run on
  7-20% of tokens, so skipping saves 3-8% FLOPs, while skip 20-27 costs -0.13 on s1464 (ties on f3904) and
  update-only selection costs 0.04-0.07.
- **Wall-clock**: a skipped block is not called, so dense + skip saves time about in proportion. Update-only saves
  much less than its FLOPs once block 0 is selected (backward through every block): same-node H200, dense 8.53 s/step
  vs update-only-25% static 7.70 (0.90x at 0.79x FLOPs), vs 4.60 (0.54x at 0.57x) when the lowest selected block is 24.
  The online ranking picks block 0 in 98-99% of steps.

## Setup

Qwen3.5-4B `q35-4b-base-markerfix`, mt4 (contradiction, nq, outlier, oolong) at 8k, `launch_fast8k.py`, `-tde`,
16 rows/step, unpacked, no full-context steps; dense arms LR 5e-6, recipe arms (`lora64all32r20to7`: whole-model LoRA
r64 + two-stage router 7:0.2,16:0.07) LR 1e-5. Data: original pool s1464 (1,464 rows/task) and fresh f3904. Evals:
8k + 2k rungs per task, eval_size 500 each, no-cot, olmo-core native docchunk ladder. Dense reference per cell = the
better of the LR 2e-6 / 5e-6 dense curves of the same pool at the arm's PF (log-PF interpolation,
`research/token_routing/layer_drop/layerskip_screen.py`). Seed 0 trained on 2 GPUs, seed 1 on 4 (same global batch);
s/step is only comparable within a GPU count.

Arms:
- `sk3` = `--train-skip-layers 20,24,28`, `sk2027` = blocks 20-27 (the low-gradient-energy span of the
  grad-composition study), `ldu40` = `--layer-drop-p 0.4 --layer-drop-schedule linear --layer-drop-exclude-first 16
  --layer-drop-exclude-last 2` (blocks 16-29, ~4.1 skipped per step).
- `gsu<x>` / `gss<x>` = `--grad-select-frac x --grad-select-mode update|skip`
  (`olmo_core.nn.grad_layer_select`): online EMA of each block's ||g||^2 over its trainable params (the adapters for
  LoRA), all-reduced so ranks agree, never-measured blocks first, 10% exploration; `st` = static ranking from the
  grad-composition per-block parameter-gradient share (top 8 = 31,0,10,11,14,15,6,7). With the router, skip mode
  always runs its drop layers 7/16 and update mode stops backward at block 7.

## Results
| arm | data | PF ÷ dense | mean Δ vs dense (s0 / s1) | outlier@8k Δ | contra@8k Δ | mean Δ vs recipe | s/step (s0 / s1) |
|---|---|---|---|---|---|---|---|
| dense + skip 20,24,28 | s1464 | 0.92 | -0.023 / -0.040 | -0.015 / -0.104 | -0.010 / -0.003 | – | 10.3 / 5.1 |
| dense + skip 20-27 | s1464 | 0.79 | +0.024 / -0.034 | +0.168 / -0.052 | +0.006 / -0.005 | – | 8.8 / 4.4 |
| dense + LayerDrop 16-29 (linear, p<=0.4) | s1464 | 0.89 | +0.007 / +0.021 | +0.050 / +0.076 | +0.000 / -0.005 | – | 9.9 / 5.0 |
| dense + update-only top 25% (online) | s1464 | 0.78 | +0.005 / -0.002 | -0.054 / -0.028 | +0.011 / -0.007 | – | 10.1 / 5.0 |
| dense + update-only top 50% (online) | s1464 | 0.86 | +0.009 / +0.009 | +0.013 / +0.128 | +0.004 / -0.000 | – | 10.4 / 5.3 |
| dense + update-only top 25% (static) | s1464 | 0.79 | +0.031 / +0.003 | +0.142 / +0.027 | +0.008 / -0.006 | – | 10.0 / 5.0 |
| dense + forward-skip top 25% (online) | s1464 | 0.36 | -0.541 / -0.514 | -0.216 / -0.219 | -0.914 / -0.915 | – | 4.8 / 2.4 |
| dense + forward-skip top 50% (online) | s1464 | 0.57 | -0.549 / -0.503 | -0.315 / -0.315 | -0.934 / -0.934 | – | 7.2 / 3.6 |
| dense + forward-skip top 50% (static) | s1464 | 0.57 | -0.124 / -0.109 | -0.233 / -0.118 | -0.279 / -0.281 | – | 6.9 / 3.5 |
| recipe (baseline) | s1464 | 0.22 | +0.082 / +0.083 | +0.406 / +0.398 | +0.077 / +0.081 | – | 4.9 / 2.7 |
| recipe + skip 20,24,28 | s1464 | 0.22 | +0.056 / +0.070 | +0.352 / +0.321 | +0.045 / +0.067 | -0.027 / -0.013 | 2.7 / 2.6 |
| recipe + skip 20-27 | s1464 | 0.21 | -0.048 / -0.041 | +0.050 / -0.012 | +0.023 / -0.014 | -0.132 / -0.126 | 2.5 / 2.5 |
| recipe + update-only top 25% | s1464 | 0.22 | +0.033 / +0.045 | +0.106 / +0.152 | +0.049 / +0.051 | -0.049 / -0.039 | 4.8 / 2.7 |
| recipe + update-only top 50% | s1464 | 0.22 | +0.010 / +0.090 | +0.052 / +0.393 | +0.079 / +0.079 | -0.072 / +0.007 | 5.0 / 2.8 |
| recipe + forward-skip top 25% | s1464 | 0.07 | -0.469 / -0.526 | -0.231 / -0.249 | -0.603 / -0.617 | -0.650 / -0.703 | 3.1 / 1.8 |
| recipe + forward-skip top 50% | s1464 | 0.12 | -0.524 / -0.529 | -0.259 / -0.267 | -0.752 / -0.752 | -0.647 / -0.654 | 4.0 / 2.3 |
| dense (seed-1 check) | f3904 | 1.00 | … / +0.002 | … / -0.004 | … / -0.001 | – | … / 5.5 |
| dense + skip 20,24,28 | f3904 | 0.92 | +0.005 / +0.002 | +0.020 / +0.015 | +0.001 / -0.011 | – | 10.3 / 5.1 |
| dense + skip 20-27 | f3904 | 0.79 | +0.004 / +0.009 | +0.049 / +0.054 | -0.008 / -0.008 | – | 8.8 / 4.4 |
| dense + LayerDrop 16-29 (linear, p<=0.4) | f3904 | 0.89 | +0.002 / +0.000 | +0.023 / +0.016 | -0.001 / -0.003 | – | 5.0 / 5.0 |
| recipe (baseline) | f3904 | 0.22 | … / +0.073 | … / +0.334 | … / +0.012 | – | … / 2.5 |
| recipe + skip 20,24,28 | f3904 | 0.22 | +0.061 / +0.072 | +0.324 / +0.341 | +0.006 / -0.003 | -0.012 / -0.003 | 2.5 / 2.5 |
| recipe + skip 20-27 | f3904 | 0.21 | +0.064 / +0.060 | +0.353 / +0.335 | +0.008 / -0.017 | -0.012 / -0.018 | 2.4 / 2.4 |

s/step: seed 0 trained on 2 GPUs and seed 1 on 4 (same global batch), except the relaunches after a bad-node NaN or
repeated preemption, which ran at 4 GPUs (recipe + skip 20-27 s1464 seed 0, dense + LayerDrop f3904 seed 0, dense f3904
seed 1). Compare s/step only within one GPU count, and across nodes only roughly (H100 node-to-node variance is ~10%).

## Selected-layer histograms (share of steps each block was selected; blocks >= 0.3 shown)

- dense update-only 25% (online): 0: 0.99, 11: 0.83, 10: 0.77, 14: 0.61, 6: 0.40, 12: 0.39, 15: 0.36; seed 1 the same
  set (0, 10, 11, 14, 6, 7, 15). The online ranking converges on the static grad-composition set minus block 31.
- dense update-only 50%: 0, 6, 10, 11, 14 ≈ 1.0; 2, 5, 7, 12, 15 ≈ 0.9; 1, 3, 4 ≈ 0.7; blocks 20-29 ≈ never.
- recipe update-only 25%: 0, 6, 7, 11 ≈ 0.9; 5, 3 ≈ 0.6.
- forward-skip runs select the same early blocks; blocks 20-29 are selected 30-40% of steps (exploration + stale
  estimates). Full per-step histories: `[grad-select]` lines in each training log, and `grad_select.json` in the save
  folder.

## Limits

- Two seeds at one data size per pool; the Pareto curves (f1952 / f3904 / f7808) are not run yet. A dense seed-1 run on
  f3904 lands +0.002 off the seed-0 dense curve, so f3904 differences of ±0.01 are ties.
- outlier@8k swings ±0.1-0.2 between seeds for the skip arms; the means are steadier.
- Some evals ran on Berkeley H200 (`record_local_evals.py`, batched greedy decode, validated equal to bs=1 on Lambda)
  while Beaker was backed up; the rest on Beaker.

## Where things are

- Code: `src/olmo_core/nn/layer_drop.py`, `src/olmo_core/nn/grad_layer_select.py`,
  `src/olmo_core/train/callbacks/{layer_drop,grad_layer_select,flop_meter}.py`, trainer flags `--train-skip-layers`,
  `--layer-drop-*`, `--grad-select-*`; launcher arms in `research/token_routing/ds64_fast8k/launch_fast8k.py`.
- Analysis: `research/token_routing/layer_drop/layerskip_screen.py`; local scoring path `local_eval_node.py`,
  `record_local_evals.py`, `ckpts_to_s3_beaker.sh`.
- Scores: `ops result show --train-run <run>` (experiment tokdrop_lr); runs registered in ops.
- Branch `claude/train-layerskip-35010b`.
