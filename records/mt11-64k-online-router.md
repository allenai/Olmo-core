---
title: The mid-stack online token router on the 11-task CTC mix at 64k with ~64M tokens per task
date: 2026-10-23
updated: 2026-10-25
experiment: mt11_64k
area: token_routing
kind: result
status: complete for otr10M7 (two seeds at 1/4 epoch, one at 1/2 and full epoch; otr10M8 not run)
superseded_by: 
artifact: https://claude.ai/artifact/5LcLKC8bT6MDfz6JcLLQ9n
summary: 11-task 64k mix (662M tokens) - the otr10M7 online router (0.40x dense FLOPs/step) beats the better-LR dense curve at matched FLOPs and LR 5e-6 by +0.025 to +0.032 (task-average) at every budget (1/4, 1/2, full epoch; 2 seeds at 1/4), ahead on 8-9/10 tasks; largest xabsence +0.10-0.12, outlier +0.04-0.13, contradiction +0.03-0.06. Same data it trails dense by ~0.03. Wall-clock 0.79-0.82x on 4 GPUs, ~1x on 8. eval_size 500/cell.
---

# The mid-stack online token router at 64k, 11 tasks, ~64M tokens per task

Experiment state: `ops/experiments/mt11_64k/STATUS.md`. Code: branch `prasann/mt11-64k`,
`research/token_routing/mt11_64k/`. Page with the per-rung Pareto plots: the artifact above.

## Question

Prasann (2026-10-22): "for the 11-task mix try to go up to like 64k and have a larger amount of data (maybe similar
data to the individual task experiments from before for each task), since it seems like these methods maybe scale
better with more flops and longer contexts?" On the 8k fast8k 11-task mixes (480-720 rows/task) the router failed
strmatch (-0.12 to -0.25) and grouping (-0.05 to -0.07).

## Setup

- **Data `mt11L64`.** setA seed-42 gold shards (the generator the CTC suite was calibrated against; query position
  both; full gold sidecars), rows <= 65,536 tokens, ds64's short-heavy token shares 30/20/15/13/12/10 over
  <=3k/6k/12k/24k/48k/64k of a 64M-token target per task (`merge_tasks.py --length-mix`). 156,587 rows, 661.7M
  tokens, 12.3k-15.8k rows and 49-64M tokens per task. `verify_shard.py`: every row hits its gold sidecar; max row
  65,536. The top bin holds ~100 rows/task for 6 tasks, 12 for reorder, none for outlier/oolong/rerank (their 64k
  bucket rows exceed 65,536 tokens) or strmatch (max 20k).
- **Training.** Qwen3.5-4B markerfix base, 8 H100s, unpacked rows at seq 65,536, 128 rows/step, micro-batch 1,
  flash_2. Dense = the pad-free soft-token path with every document kept (`densepf`; on the mt4 8k screen it scored
  within noise of padded dense). Budgets f16/f8/f4/f2/f1 = 77/153/306/612/1,223 steps, each a complete LR schedule
  (`--max-tokens`).
- **Router `otr10M7`.** Blocks 0-6 on the full row; a ridge router refit every step on per-token gradient norms
  (10% exploration) keeps 10% of each document's tokens for blocks 7-31; a charged full-context step every 8; 512
  target samples per row.
- **Eval.** CTC-suite rows (HF `PrasannSinghal/ctc-suite-eval` at olmo-eval's pinned revision), 500 per rung,
  query position both, ctc-spec grading, native docchunk ladder on Beaker, no-cot. Rungs 2k/8k/16k/32k/64k where the
  suite has them (reorder to 16k; xabsence/strmatch/grouping to 32k). rerank is trained but not scored (no ladder
  grader for its suite rows). The 25 long cells were validated on all 500 rows before use (eval/train prefix parity
  500/500, gold scores 1.000, junk 0; longest prompt 70,035 tokens -> MAX_LENGTH 73,728, nothing skipped).
- **Comparison.** FLOPs = `flop_meter/actual_pflops`. Dense reference = per task x rung the better of LR 2e-6 and
  5e-6, interpolated linearly in log PF. No-signal cells excluded from means: reorder 8k/16k (every arm fails the
  exact-permutation parser: duplicate ids at 8k, repetition loops in 150/500 rows at 16k) and grouping 32k (every arm
  0.001-0.005).

## Results (single seed; eval_size 500 per cell; no-cot)

Cost: dense ~15 PF/step (f16 1,175 PF, f8 2,287 PF). Router f4 (306 steps): 1,872 PF actual vs 4,625 PF
dense-equivalent = 0.405x, i.e. between dense f16 and f8 (interpolated, not extrapolated).

Mean over each task's rungs of (router - dense at the router's PF):

| task | router LR 5e-6 vs best-LR dense | router LR 2e-6 vs best-LR dense | router LR 2e-6 vs dense LR 2e-6 |
|---|---|---|---|
| xabsence | **+0.125** (8k/16k/32k +0.14/+0.20/+0.15) | -0.015 | +0.082 |
| outlier | +0.051 (8k +0.11) | -0.051 | +0.056 |
| contradiction | +0.056 (16k/32k/64k +0.07/+0.08/+0.105) | -0.034 | +0.085 |
| oolong | +0.023 | -0.006 | +0.021 |
| qdmatch_nq | +0.022 | -0.014 | +0.060 |
| hotpotqa | +0.017 (32k/64k +0.04) | -0.009 | +0.032 |
| grouping | +0.027 (8k/16k +0.04/+0.05) | -0.027 | +0.020 |
| nq | +0.005 | -0.008 | +0.013 |
| strmatch | -0.018 | -0.090 | -0.041 |
| reorder (2k only) | +0.031 | -0.082 | +0.004 |

Task-average of the LR 5e-6 column: **+0.034**, ahead on 9/10 tasks.

Binomial SE per cell is about ±0.01-0.02; run-to-run noise on the 8k screens was ±0.02-0.03 (up to ±0.13 on
outlier). Only the large long-rung wins (xabsence, contradiction 16k-64k, outlier 8k) clear a single seed; the rest
need the replicates.

- **The learning rate decides the sign.** At matched LR the router is ahead on 9/10 tasks; at 2e-6 it is ahead of
  2e-6 dense on 9/10 but behind 5e-6 dense on all. Dense 5e-6 beats dense 2e-6 at every measured budget on most cells.
- **Gains grow with length** on xabsence, contradiction, hotpotqa and qdmatch_nq; nq and 2k cells sit at parity.
- **strmatch**, the 8k screen's failure, is now within about one noise unit (-0.018) at matched LR.
- **Training on the mix fixes most of the length falloff:** nq f1 at 64k 0.17 (10-step smoke) -> 0.86 after 153
  dense steps.
- **No wall-clock saving** at micro-batch 1 on 8 GPUs: a routed step took 17-25 s vs ~20 s dense. The FLOP saving
  is real (0.40x) but this configuration is not compute-bound per micro-batch.
- Router health at 64k: keep_frac 0.105-0.108, leak_auc 0.47-0.52 (does not pick gold by count), gradient-mass
  fidelity at the full-context steps 0.62-0.66 (oracle 0.86-0.88, random 0.10).

## Seed replicate (router f4 LR 5e-6, seeds 0 and 1; dense f8/f4 LR 5e-6 also replicated)

Each router seed vs the seed-averaged better-LR dense curve (f16 seed 0 only), mean over rungs (eval_size 500/cell):

| task | seed 0 | seed 1 | mean |
|---|---|---|---|
| xabsence | +0.108 | +0.118 | +0.113 |
| outlier | +0.071 | +0.035 | +0.053 |
| contradiction | +0.050 | +0.033 | +0.042 |
| hotpotqa | +0.018 | +0.024 | +0.021 |
| oolong | +0.021 | +0.021 | +0.021 |
| qdmatch_nq | +0.011 | +0.014 | +0.012 |
| nq | +0.002 | +0.007 | +0.004 |
| reorder (2k) | +0.010 | -0.009 | +0.001 |
| strmatch | -0.017 | -0.016 | -0.016 |
| grouping | +0.027 | pending | |

Ahead on 8/9 tasks with both seeds, mean +0.028; the seeds agree in sign everywhere except reorder (a tie). strmatch is
a small, consistent loss.

## More data: 1/2 and full epoch (LR 5e-6; router and dense on 4 GPUs)

Router minus the better-LR dense curve (seed-averaged where replicated), interpolated in log PF, mean over rungs:

| router run | PF | nq | hotpotqa | qdmatch | outlier | oolong | contra | xabsence | reorder | strmatch | grouping | task-avg | ahead |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f4 seed 0 | 1,872 | +0.002 | +0.018 | +0.011 | +0.071 | +0.021 | +0.050 | +0.108 | +0.010 | -0.017 | +0.043 | +0.032 | 9/10 |
| f4 seed 1 | 1,862 | +0.007 | +0.024 | +0.014 | +0.035 | +0.021 | +0.033 | +0.118 | -0.009 | -0.016 | +0.024 | +0.025 | 8/10 |
| f2 | 3,725 | -0.003 | +0.021 | -0.001 | +0.069 | +0.011 | +0.044 | +0.124 | +0.009 | +0.001 | +0.031 | +0.031 | 8/10 |
| f1 | 7,439 | -0.012 | +0.001 | +0.011 | +0.125 | -0.004 | +0.031 | +0.099 | +0.001 | +0.002 | +0.023 | +0.028 | 8/10 |

Every router point lies inside the measured dense range (dense f16 1,175 / f8 2,287 / f4 ~4,575 / f2 9,294 / f1 18,592 PF):
no extrapolation. The advantage is steady at about +0.03 rather than growing with data. outlier at f1 (+0.125;
16k 0.899, 32k 0.599, above every dense run) is single-seed and outlier's run-to-run noise reaches ±0.13: replicate
before quoting it alone.

**Same data** (router - dense at LR 5e-6, same rows): -0.035 (f4), -0.031 (f2), -0.034 (f1). The router gives up
~0.03 for 0.40x the FLOPs; dense needs ~2.5x the FLOPs to buy that 0.03 back, hence the matched-FLOP win.

**Wall-clock**, like for like on 4 H100s: f2 router 18,824 s vs dense 22,979 s (0.82x); f1 38,479 s vs 49,008 s (0.79x).
On 8 GPUs the per-step times were about equal (~20 s): with fewer GPUs per step the compute share grows and part of the
FLOP saving shows up as time.

## The adopted universal recipe at f4 (2026-10-25, requested by the LR-sweep session)

`lora64r20to7f16` (launch_otr.py, verbatim): LoRA r64 (scale 32) on blocks 0-6, two-stage router keeping 20% after
block 7 and 7% after block 16, per-document floor 16 tokens, no full-context steps, no freeze, LR 5e-6, 4 GPUs.
1,695.8 PF = 0.365x dense-equivalent (between dense f16 and f8). Matched-FLOP delta vs the better-LR dense curve, mean
over rungs: nq +0.003, hotpotqa +0.022, qdmatch_nq +0.016, outlier +0.063, oolong +0.023, contradiction +0.068 (+0.119 at
64k), xabsence +0.110, reorder (2k) +0.017, strmatch +0.020; task-average **+0.038, ahead 9/9** (grouping eval re-running
after an NCCL timeout; its 2k/8k put it near +0.006). Beats otr10M7 at f4 (+0.025 / +0.032 over two seeds) at less
compute, and turns strmatch from -0.017 into +0.020. Peak GPU memory 37.2 / 44.2 GiB active / reserved vs dense 48.7 /
52.8 (-24% / -16%); otr10M7 matched dense (its full-context step every 8). Single seed. Eval log confirms the LoRA
adapters were rebuilt from the checkpoint (eval commit contains 984825862).

## Traps hit

- Urgent Beaker jobs were preempted right after their 1 h minRuntime and restarted from scratch (no resume point
  before step 300): minRuntime now covers the whole run (capped at Beaker's 8 h) with a resume point every 50 steps.
- One node (`01JRYCX2HZQ2QE60PMHBB7ZM8J`) produced a NaN gradient at step 1; the same batch gives CE 0.9751 / grad norm
  26.82 elsewhere. Check the step-1 grad norm before suspecting the code.
- A routed step costs ~0.40x a dense step (not ~0.3x from the block count); the 10-step smoke's 0.52x included 2
  dense steps and overstated it.

## Open

- otr10M8 (split after block 8) was not run on this mix; launch_mt11_64k.py has it.
- Seeds at f2/f1 (single seed there; the f1 outlier jump in particular).
- A wall-clock-optimal configuration (micro-batch > 1 for short rows, fewer GPUs per run) was not explored.
