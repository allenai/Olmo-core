---
title: Length generalization of the mt11 64k runs - router/recipe vs dense at 128k and 256k, and training on rows <= 16k
date: 2026-11-04
updated: 2026-11-04
experiment: mt11_64k
area: token_routing
kind: result
status: complete (single seed)
superseded_by: 
artifact: https://claude.ai/artifact/5LcLKC8bT6MDfz6JcLLQ9n
summary: mt11 64k checkpoints scored at 64k/128k/256k (olmo-eval, r256k YaRN x2; agrees with the native ladder at 64k to +-0.002). Universal recipe f4 matched-FLOP gain persists past the 64k train length (+0.046/+0.033/+0.054 6-task mean); router f2 +0.040/+0.042/+0.023; router f1 fades (+0.025/+0.013/-0.005). Training on rows <= 16k at the same FLOPs: >= 0 at 2k-32k, then -0.06 to -0.10 (dense, at f4 and f2 FLOPs) and -0.05 to -0.08 (recipe) from 64k to 256k. Single seed; r256k eval_size 125.
---

# Length generalization of the mt11 64k runs

Experiment state: `ops/experiments/mt11_64k/STATUS.md`. Base results: `records/mt11-64k-online-router.md`. Code: branch
`claude/mt11-lengen-bed337`, `research/token_routing/mt11_64k/lengen_*`. Page: the artifact above (section "Length
generalization").

## Question

Prasann (2026-11-04): "For 11-task 64k runs, can you get length-generalization evals at 128k and 256k (for a few select
relevant high-FLOP pairs, comparing dense vs recipe train), and see if pareto gains persist to length generalization
setting as well. You can do on a subset of tasks" and "could you get a few points where you have train data only going
up to 16k, but more train data (such that FLOPs match the train run with up to 64k data) as a test of length
generalization. I want to see how much worse this is for the longer contexts at evaluation time".

## Setup

- **Checkpoints** (all LR 5e-6, seed 0, trained on mt11L64, rows <= 65,536 tokens): dense f16/f8/f4/f2/f1 (1,175 /
  2,287 / 4,625 / 9,294 / 18,592 PF), router otr10M7 f2 (3,725 PF) and f1 (7,439 PF), universal recipe
  lora64r20to7f16 f4 (1,696 PF).
- **Tasks**: the six trained tasks with 128k/256k suite rows: nq, hotpotqa, qdmatch_nq, outlier, oolong, contradiction.
- **Eval path**: olmo-eval `scripts/ctc_suite/launch_ctc_suite.py`, branch `prasann/ctc-suite-mt11lengen` (=
  `prasann/ctc-suite-lengen`), `--arm full --doc-markers`, tokenizer Qwen3.5-0.8B (chat template closes the think
  block), OLMo-core `claude/mt11-lengen-bed337` installed `--no-deps` on the torch 2.9.1 image. r256k runs with YaRN x2
  and 131,072-token chunked prefill (the suite's r256k prompts exceed 262,144 positions). eval_size 500 at r64k/r128k,
  125 at r256k (all the suite ships). The 2k-64k numbers in the base record came from the native docchunk ladder,
  which OOMs past ~77k tokens on one H100 and has no YaRN, so r64k was re-scored on the new path for every checkpoint.
- **Matched FLOPs**: arm score minus dense interpolated linearly in log PF between the two neighbouring dense budgets
  (both must have the cell). Dense uses LR 5e-6 throughout (the better LR at most cells of the base experiment; f2/f1
  dense exist only at 5e-6).
- **Rows <= 16k (mt11L16)**: every setA s42 gold row of at most 16,384 tokens, same 11 tasks (`merge_tasks.py`
  `--max-row-len 16384`; `build_beaker.sh MAXLEN=16384`): 192,554 rows, 716.1M tokens, VERIFY PASSED. Most tasks'
  16k-rung rows exceed 16,384 tokens, so the 12-16k bin holds 0-1,220 rows per task and training rows are mostly at or
  below the 8k rung. Seq 16,384, 128 rows/step, 4 GPUs, LR 5e-6. Steps set from a fitted PF/token (25.3-26.7 GFLOP vs
  28.1 at 64k): m16-dense-s380 (4,839 PF, ~f4), m16-lora64r20to7f16-s380 (1,836 PF), m16-dense-s770 (~f2, training).
  No repetition (the f4-matched run used ~25% of the pool).

## Trap fixed on the way

olmo-eval's olmo_core provider builds the model through `TransformerGenerationModule.from_checkpoint` without a
`post_build_hook`, so the universal recipe's early-stack LoRA adapters were not installed and a DCP load would have
silently scored the base early stack. `from_checkpoint` now applies `post_build_hook_from_config` by default
(2011b86d5; `OLMO_CONFIG_BUILD_HOOK=0` disables), which also adds the checkpoint-key guard. Every recipe eval log shows
`[early-lora] adapters re-installed from config.json`.

## Results: matched-FLOP delta past the training length

6-task mean of (arm - dense at the arm's PF), tasks ahead in brackets. 64k/128k eval_size 500 per cell; 256k
eval_size 125 per cell (binomial SE +-0.03-0.045). No-cot. Single seed.

| arm (PF) | 64k | 128k | 256k (YaRN x2) |
|---|---|---|---|
| recipe f4 (1,696) | +0.046 (5/6) | +0.033 (6/6) | +0.054 (5/6) |
| otr10M7 f2 (3,725) | +0.040 (4/6) | +0.042 (6/6) | +0.023 (5/6) |
| otr10M7 f1 (7,439) | +0.025 (5/6) | +0.013 (4/6) | -0.005 (3/6) |

Per task, recipe f4: contradiction +0.126 / +0.055 / +0.116, hotpotqa +0.046 / +0.065 / +0.095, nq +0.023 / +0.033 /
+0.070, oolong +0.036 / +0.029 / +0.044, outlier -0.007 / +0.013 / -0.005, qdmatch_nq +0.048 / +0.006 / +0.005.
Router f2: contradiction +0.101 / +0.132 / +0.104, outlier +0.085 at 64k, nq -0.019 / +0.002 / -0.011.

- The recipe's matched-FLOP gain does not shrink past the 64k training length; on the native ladder at 64k it was
  +0.037 over 10 tasks.
- outlier and qdmatch_nq are at the floor at 128k/256k for every checkpoint (outlier 0.000-0.13, qdmatch 0.01-0.13):
  they carry no signal there. Parse rate 1.000 and no empty generations on those cells (dense f16 outlier r256k =
  0.000 is a real floor).
- The full-epoch router (f1) loses its edge with length: +0.025 / +0.013 / -0.005 (oolong -0.060, contradiction -0.027 at
  256k). Its native-ladder gain was +0.028 at 2k-64k.
- **Anchor:** the two eval paths agree at 64k on all 8 checkpoints: 6-task mean native vs olmo-eval within +-0.002, every
  task within +-0.01 (e.g. dense f1 0.708 / 0.707, recipe f4 0.538 / 0.538). The long-rung numbers are on the same scale
  as the base record's 64k numbers, and the recipe's LoRA adapters load identically on both paths.

## Results: training on rows <= 16k at matched FLOPs

Mean delta vs the 64k-data model at the same PF (dense: 64k-data dense interpolated at 4,839 PF; recipe: the 64k-data
recipe f4 directly). 2k-64k: native ladder, 10 tasks (no-signal cells reorder 8k/16k, grouping 32k excluded). 128k/256k:
olmo-eval path, 6 tasks.

| rung | rows<=16k dense (4,839 PF) vs 64k-data dense | rows<=16k recipe (1,836 PF) vs 64k-data recipe (1,696 PF) |
|---|---|---|
| 2k | +0.001 | +0.003 |
| 8k | +0.023 | +0.018 |
| 16k | +0.019 | +0.016 |
| 32k | +0.004 | -0.012 |
| 64k native / olmo-eval | -0.081 / -0.079 (contradiction -0.20, hotpotqa -0.15) | -0.055 / -0.053 |
| 128k | -0.080 (contradiction -0.18, hotpotqa -0.19, nq -0.07) | -0.064 (contradiction -0.12, hotpotqa -0.06) |
| 256k (eval_size 125) | -0.071 (hotpotqa -0.17, contradiction -0.10, nq -0.10) | -0.083 (hotpotqa -0.14, contradiction -0.06) |

At dense f2's FLOPs, m16-dense-s770 (9,792 PF) vs 64k-data dense at the same PF: 2k +0.007, 8k +0.015, 16k +0.039,
32k +0.012, 64k -0.064 (native; contradiction -0.25), 128k -0.100 (contradiction -0.35, hotpotqa -0.16), 256k -0.076
(contradiction -0.20, hotpotqa -0.19). Twice the data does not shrink the long-rung loss.

More short rows at the same FLOPs help a little at the short rungs (xabsence +0.08..+0.25 at 8k-32k) and cost a
roughly constant 0.05-0.08 from 64k to 256k, mostly on contradiction and hotpotqa. The gap appears at 64k (past the
16k-data model's training length but within the 64k-data model's) and does not keep growing at 128k/256k, where both
models are extrapolating.

## Traps hit

- titan (B200) accepts our unallocated jobs only as preemptible filler (minRuntime 0); several bins were preempted 1-3
  times. Stalled cells were rerun on jupiter with minRuntime 3h; `lengen_pool.py` takes each (checkpoint, task, rung)
  cell from one complete source launch. B200 and H100 smoke scores were identical.
- Seven router-f1 bins failed together at 09:32 at data prep ("Failed to prepare 1 task", likely unauthenticated HF
  requests); resubmitted.
- olmo-eval jobs print nothing until the end (`tail -80`): no output for 1-3 h is normal. r128k rows took ~20-40 s and
  r256k ~85 s each on one GPU, 2-3x the launcher's planner estimate.
- An interpolation that skips a missing middle dense budget gives wrong router deltas; the pool and the page now
  interpolate only between true neighbours with complete cells.

## Open

- Seeds; r256k is capped at the 125 rows the suite ships.
- Not tested: a length curriculum or a small share of long rows (does a few-% 64k tail recover the long rungs?).
- Single seed throughout; r256k at eval_size 125.
