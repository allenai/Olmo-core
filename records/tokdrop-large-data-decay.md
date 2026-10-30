---
title: Large-data decay of the token-drop recipe -- diagnosis (answer CE, unsaturated cells) and two training fixes (ReLoRA, late keep-widening)
date: 2026-10-30
updated: 2026-10-30
experiment: tokdrop_lr
area: token_routing
kind: result
status: diagnosis done; 4 fix runs + 1 recipe seed-1 control training on Beaker (2026-10-30)
superseded_by: 
artifact: (pending -- add to the tokdrop_lr overview artifact)
summary: Token-drop recipe FTM decay (f1 3.7 -> 2.2x, answer CE 2.7 -> 1.8x from 3,904 to 7,808 rows/task) is mostly the dense plateau: dense CE is flat from 3,904 rows (-0.002 +- 0.004) while the recipe's matched-data CE gap stays ~0.01 nats (no trend) and it keeps improving; fixes ReLoRA / late keep-widening in flight.
---

# Large-data decay of the token-drop recipe

Branch `claude/tokdrop-decay-fix` (from `claude/online-router-1bc01a` @ 8ce5fd22a). Code in
`research/token_routing/decay_fix/`; trainer changes in `src/olmo_core/nn/early_lora.py`
(`merge_and_reinit_lora`), `src/olmo_core/train/callbacks/relora.py`, `src/olmo_core/nn/online_token_router.py`
(`late_keep`), flags `--st-otr-relora-frac / --st-otr-relora-rewarm / --st-otr-late-keep` in
`recipes/train/research_trainer/train_ctc_suite.py` (+ `recipes/train/variants.py`).

## Question

The recipe (`lora64all32r20to7`: whole-model rank-64 LoRA over a frozen base + two-stage online router keeping
20% after block 7 and 7% after block 16, LR 1e-5, fresh mt4 pools) matches dense at 4.1 / 3.7 / 2.2x fewer FLOPs at
1,952 / 3,904 / 7,808 fresh rows/task. Is that decay the dense ceiling (eval saturation) or the method?

## Part 1: diagnosis

### f1 view (existing evals; eval_size 500 per cell, no-cot)

`analyze_decay.py` on `research/token_routing/ds64_fast8k/results/pareto_points.json`:

| run | PFLOPs | 8-cell f1 | FTM | unsat-4 f1 (oolong+outlier, 8k+2k) | FTM |
|---|---|---|---|---|---|
| dense f1952 | 1531 | 0.8876 | | 0.8093 | |
| dense f3904 | 3063 | 0.9257 | | 0.8767 | |
| dense f5856 | 4595 | 0.9280 | | 0.8832 | |
| dense f7808 | 6128 | 0.9310 | | 0.8822 | |
| recipe f1952 | 345 | 0.8830 | < 4.4x (below dense f1952) | 0.8115 | 4.5x |
| recipe f3904 | 688 | 0.9153 | 3.7x | 0.8658 | 4.0x |
| recipe f7808 | 1373 | 0.9243 | 2.2x | 0.8745 | 2.2x |

Dense gains +0.005 on the 8-cell mean from 3,904 to 7,808 rows and nothing on the unsaturated cells (0.877 ->
0.882): the "unsaturated" cells are saturated for dense too (oolong 8k sits at 0.69 from f3904 on). On a flat dense
curve any small deficit maps to a large FLOP ratio: recipe f7808 is -0.007 behind dense f7808 on the 8-cell mean
(SE of the mean ~0.004 per run, before seed noise), and most of that is ONE cell, outlier@2k (0.938 vs 0.979), which
moved 0.880 -> 0.972 -> 0.938 over the three recipe budgets (the status page's run-to-run noise on outlier is +-0.03).

### Answer-CE view (eval_size 500 per cell, no-cot, teacher forced; mean over rows of per-row answer CE, nats)

`score_answer_ce.py` (7 one-GPU Beaker jobs `dfce-*`; JSONs in `research/token_routing/decay_fix/results/ce/`, recorded
with `ops result add --check`, metric `answer_ce` / `answer_top1`). FTM read off the dense CE curve the same way.

| run | PFLOPs | CE 8-cell | FTM | CE unsat-4 | FTM | CE token-weighted | FTM |
|---|---|---|---|---|---|---|---|
| dense f1952 | 1531 | 0.1581 | | 0.2842 | | 0.1422 | |
| dense f3904 | 3063 | 0.1390 | | 0.2496 | | 0.1256 | |
| dense f5856 | 4595 | 0.1372 | | 0.2495 | | 0.1246 | |
| dense f7808 | 6128 | 0.1374 | | 0.2508 | | 0.1223 | |
| recipe f1952 | 345 | 0.1677 | < 4.4x | 0.2952 | < 4.4x | 0.1529 | < 4.4x |
| recipe f3904 | 688 | 0.1527 | 2.7x | 0.2746 | 2.7x | 0.1370 | 2.8x |
| recipe f7808 | 1373 | 0.1457 | 1.8x | 0.2637 | 1.7x | 0.1299 | 1.9x |

Paired differences (same rows; SE of the 8-cell mean = sqrt(sum of per-cell paired SE^2) / 8), `paired_ce.py`:

| comparison | CE 8-cell | CE unsat-4 |
|---|---|---|
| dense f3904 - dense f1952 | -0.0190 +- 0.0031 | -0.0345 +- 0.0055 |
| dense f5856 - dense f3904 | -0.0018 +- 0.0028 | -0.0001 +- 0.0052 |
| dense f7808 - dense f3904 | -0.0017 +- 0.0037 | +0.0011 +- 0.0070 |
| recipe - dense, f1952 (matched rows) | +0.0096 +- 0.0023 | +0.0111 +- 0.0041 |
| recipe - dense, f3904 | +0.0136 +- 0.0027 | +0.0250 +- 0.0049 |
| recipe - dense, f7808 | +0.0083 +- 0.0031 | +0.0129 +- 0.0055 |
| recipe f7808 - recipe f3904 | -0.0070 +- 0.0034 | -0.0109 +- 0.0062 |
| recipe f7808 - dense f3904 | +0.0066 +- 0.0037 | +0.0141 +- 0.0068 |

Per cell, the recipe-f7808 gap to dense f7808 sits in oolong (8k +0.024 +- 0.016, 2k +0.016 +- 0.015), nq@8k
(+0.014 +- 0.010) and outlier@2k (+0.013 +- 0.003); contradiction and outlier@8k are within +-0.003.

### Verdict

**Both, but mostly the dense ceiling.** Dense stops improving at 3,904 rows/task on BOTH metrics (CE -0.0017 +-
0.0037 from 3,904 to 7,808; f1 +0.005), so FTM past that point is a ratio against a flat curve. The method does NOT
get worse with data: its CE gap to dense at matched rows is +0.010 / +0.014 / +0.008 nats at 1,952 / 3,904 / 7,808
(no trend), and it keeps improving (-0.007 +- 0.003 from 3,904 to 7,808). What makes FTM fall is that a CONSTANT
~0.01-nat gap must now be closed against a plateau: at 7,808 rows the recipe is still +0.007 +- 0.004 above dense's
3,904-row plateau, i.e. it has not reached the dense asymptote at 2x the data. CE puts the decay at 2.7x -> 1.8x
(f1: 3.7x -> 2.2x), the same shape. So the method has a small asymptotic deficit (in oolong and outlier@2k), and the
saturated eval converts it into a large FTM drop; the f1 numbers additionally carry one noisy cell (outlier@2k
0.880 / 0.972 / 0.938 across recipe budgets). A fix has to shrink the constant ~0.008-nat gap, which is what the two
Part-2 arms target (rank capacity: ReLoRA; late compute: keep-widening).

## Part 2: fixes

| arm | change | FLOPs |
|---|---|---|
| `rl25` | ReLoRA: after 25/50/75% of the steps merge `scale * B A` into the frozen base, re-init A (per-rank seeded kaiming), B = 0, zero the adapters' AdamW moments and step, clear the skip-step history, re-warm every group's LR linearly over 20 steps | +~2 r x params per merge (charged; negligible) |
| `lw35` | late keep-widening: from step round(0.667 T) the stages keep 35% / 15% of the body instead of 20% / 7% | charged at the realised kept lengths |

Verification: CPU unit tests (`src/test/nn/online_token_router_test.py -k 'late_keep or relora'`, 23 pass with the
pyramid / early-LoRA tests), a 2-rank gloo DTensor check of the sharded merge (`check_relora_dtensor.py`: merged base
== dense W + sBA, uneven Shard(0) chunks, per-rank A draws differ, optimizer state zeroed, FLOPs sum right), and a
1-GPU H200 trainer smoke on horton (12 steps, merges at 4/8/12: |dW|/|W| 4.9e-4 / 9.9e-4 / 1.5e-3, fp32 base, LR
re-warm 1/3, 2/3, 1; late keep on from step 6: keep_frac 0.357, stage-2 relative 0.43; no skipped steps).

Runs (2 x H100 jupiter, urgent, minRuntime 1h, gb 16, micro 2, seq 12288, fresh mt4 pools, 1 seed):

| run | Beaker |
|---|---|
| f8k-mt4-lora64all32r20to7rl25-f3904-tdf-lr1e-05 | 01M65WY3GPRRDX43CGDNE3ECWT |
| f8k-mt4-lora64all32r20to7rl25-f7808-tdf-lr1e-05 | 01M65WZ0WNJV5BJZ7CNXE1X85N |
| f8k-mt4-lora64all32r20to7lw35-f3904-tdf-lr1e-05 | 01M65WZZZS1KHWQ0VZ2TJ27S3M |
| f8k-mt4-lora64all32r20to7lw35-f7808-tdf-lr1e-05 | 01M65X11FJ4QX8WAPXVA7CG8BD |
| f8k-mt4-lora64all32r20to7-f7808-tdf-sd1-lr1e-05 (recipe, seed 1: run-to-run noise control) | 01M65X26E7PMTRG4FGHRVJDDT4 |

Decisive: +0.006 on the 8-cell mean at 7,808 rows over the recipe, or answer CE matching dense at >= 1.5x its data.
| f8k-mt4-lora64all32r25to10-f3904-tdf-lr1e-05 (uniform 25% / 9.7%: compute-matched control for lw35) | 01M66624MPFG8E4P970ZY5TAJ1 |
| f8k-mt4-lora64all32r25to10-f7808-tdf-lr1e-05 | 01M6662WV3DFQGMBTSCZXDJ0SC |
| f8k-mt4-lora64all32r20to7lw50-f3904-tdf-lr1e-05 (late 50% / 25%) | 01M6663P9JR9G66X25SWRP6KH4 |

The three f7808 runs launched first (rl25, lw35, recipe sd1) were PREEMPTED at step ~1,315 / 1,952 (10:19 UTC) and
requeued by Beaker; the trainer saves no mid-run checkpoint, so each restarts from step 0.

### Results at 3,904 rows/task (eval_size 500 per cell, no-cot; 1 seed)

| run | PFLOPs | 8-cell f1 | f1 FTM | answer CE 8-cell | CE FTM | CE vs recipe (paired) | CE vs dense f3904 (paired) |
|---|---|---|---|---|---|---|---|
| recipe (tde) | 688 | 0.9153 | 3.7x | 0.1527 | 2.7x | | +0.0136 +- 0.0027 |
| rl25 (ReLoRA) | 686 | 0.9171 | 3.8x | 0.1553 | 2.5x | +0.0026 +- 0.0020 | +0.0162 +- 0.0029 |
| lw35 (late widen) | 745 | 0.9157 | 3.4x | 0.1458 | 3.2x | **-0.0069 +- 0.0022** | +0.0067 +- 0.0024 |

- ReLoRA does nothing (CE +0.003 +- 0.002, f1 +0.002): the rank-64 adapter is not the binding constraint at this scale.
  Merges ran as designed (|dW|/|W| per segment 3.0e-2 / 2.1e-2 / 1.3e-2, no loss spike, LR re-warm 1/20 ... 1).
- Late keep-widening halves the CE gap to dense (+0.014 -> +0.007) for +8% FLOPs, mostly on oolong (8k -0.037 +-
  0.012, 2k -0.018 +- 0.009): lw35 at 3,904 rows has the CE of the recipe at 7,808 rows (0.1458 vs 0.1457) at 54% of
  its FLOPs. f1 does not move (0.9157 vs 0.9153; one 8-cell f1 run has SE ~0.004), so it does not yet clear the f1 bar.
  The uniform compute-matched control (uw25) decides whether "late" matters or only "more tokens".

### Second wave (2026-10-30 afternoon; eval_size 500 per cell, no-cot, 1 seed)

f1 now read from ops results (`analyze_decay.py --f1-ops`, `run_analysis.sh`): `ds64_fast8k/results/pareto_points.json`
was regenerated after the answer_ce rows landed, and its collector (`collect_pareto_points._ops_scores`) keeps any
metric per (task, rung), so the dense/recipe "scores" there became answer CE. **Fix needed in flopdrop_clean:** skip
metrics starting with `answer_` there (as it already skips `gradcomp_`).

| run | PFLOPs | 8-cell f1 | f1 FTM | CE 8-cell | CE FTM | CE vs recipe, same rows (paired) |
|---|---|---|---|---|---|---|
| uw25 f3904 (uniform 25%/9.7%) | 745 | 0.9140 | 3.3x | 0.1518 | 2.6x | -0.0009 +- 0.0022 |
| lw35 f3904 (late 35%/15%) | 745 | 0.9157 | 3.4x | 0.1458 | 3.2x | -0.0069 +- 0.0022 |
| lw50 f3904 (late 50%/25%) | 812 | 0.9195 | 3.4x | 0.1482 | 2.7x | -0.0045 +- 0.0023 |
| rl25 f7808 (ReLoRA) | 1371 | 0.9245 | 2.2x | 0.1426 | 2.0x | -0.0030 +- 0.0022 |

- At equal FLOPs (745 vs 745 PF) late widening beats the uniform wider keep by 0.0060 +- 0.0021 CE: it is the TIMING
  (full-ish context in the last third), not the extra tokens. Widening further (lw50) buys nothing more.
- ReLoRA at f7808: -0.003 +- 0.002 CE vs the recipe (unsat-4 -0.008 +- 0.004), f1 +0.000; gap to dense f7808 shrinks
  from +0.008 to +0.005 +- 0.003. Within the recipe's seed noise until the seed-1 control lands.

### Seed / commit noise at f7808 (recipe seed 1, this branch)

| run | PFLOPs | 8-cell f1 | f1 FTM | unsat-4 f1 | CE 8-cell | CE FTM |
|---|---|---|---|---|---|---|
| recipe f7808 (tde, seed 0) | 1373 | 0.9243 | 2.2x | 0.8745 | 0.1457 | 1.8x |
| recipe f7808 (tdf, seed 1) | 1371 | 0.9270 | 2.8x | 0.8827 | 0.1421 | 2.0x |
| rl25 f7808 | 1371 | 0.9245 | 2.2x | 0.8805 | 0.1426 | 2.0x |

Seed 1 - seed 0: CE -0.0035 +- 0.0026, f1 +0.0027; so the f1 FTM at 7,808 rows is 2.2-2.8x across two seeds and part
of the "4.1 -> 2.2x" decay is one unlucky seed. rl25 - recipe seed 1: CE +0.0005 +- 0.0023 -> ReLoRA = no effect.
Recipe (seed 1) - dense f7808: CE +0.0048 +- 0.0031.

Preemption: lw35-f7808 and uw25-f7808 were preempted 4 and 3 times (restart from 0 each time). Added resume support
(commit def4dce00: `--ephemeral-save-interval`, FLOP-meter + online-router fit in the trainer state, checkpointer
`save_final`; verified on horton: resumed at step 3 with the router fit and FLOP total intact, no full-context step)
and relaunched both as `...-f7808-tdf-lr1e-05-rs` (`--resume-every 150`); the original attempts keep running too.

### Status at hand-back (2026-10-30 ~19:40 UTC)

- The original lw35-f7808 / uw25-f7808 attempts (non-resumable, preempted 4x / 3x) were CANCELLED; the resumable
  copies `f8k-mt4-lora64all32r20to7lw35-f7808-tdf-lr1e-05-rs` (01M66S2ZS1HFBF47J70H978ZGX) and
  `f8k-mt4-lora64all32r25to10-f7808-tdf-lr1e-05-rs` (01M66S3SBB0C7GJAM2M2VXX72T) reached ~step 980 / 1952, wrote a
  step-900 resume checkpoint, were preempted at 17:38 UTC and are PENDING on jupiter (they resume from step 900).
- Not moved to jsteinhardt: the Qwen3.5 mt4f7808 shard, the eval rung files and the step-900 resume checkpoints are on
  weka only (sneetches holds the Qwen3 `mt4f*q3` pools, not these), so a local run means a weka -> S3 -> /data staging
  job and a restart from step 0.

**Verdict so far:** at 3,904 rows late keep-widening is the only fix that beats the recipe (answer CE -0.007 +- 0.002
at +8% FLOPs; it also beats the compute-matched uniform keep by 0.006 +- 0.002), but no arm moves the 8-cell f1 beyond
noise (|delta f1| <= 0.004). ReLoRA does nothing at either size. The f7808 verdict (lw35 / uw25 vs the two recipe
seeds, CE 0.1457 / 0.1421, f1 0.9243 / 0.9270) is pending on the two -rs runs.
