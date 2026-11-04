---
title: Position-id reset after each cut on the token-drop recipe (lora64all32r20to7pr)
date: 2026-11-04
updated: 2026-11-04
experiment: tokdrop_lr
area: token_routing
kind: result
status: done (1 seed at 1,952 and 3,904 rows/task)
superseded_by: 
artifact: https://claude.ai/artifact/SPSBxHFs2PUn5CPjJX3Wgt
summary: Renumbering the kept tokens' position ids 0..k-1 after each router cut (instead of keeping their original positions) costs 0.026 / 0.015 on the 8k 4-task mean at 1,952 / 3,904 rows/task (0.820 vs 0.846; 0.870 vs 0.885), almost all on outlier (8k -0.10 / -0.05, 2k -0.14 / -0.18); contra -0.01, nq and oolong unchanged; training CE matches the recipe.
---
# Position-id reset ablation on the recipe

**Verdict.** Keeping original position ids matters, but mostly for one task. Resetting the kept row's positions to
0..k-1 after each cut lowers the 8k 4-task mean by 0.026 at 1,952 rows/task (0.820 vs recipe 0.845 / 0.847) and by
0.015 at 3,904 (0.870 vs 0.882 / 0.889). Outlier carries nearly all of it: at 8k it loses 0.10 / 0.05 (vs a recipe
seed spread of 0.001-0.003 and ±0.012-0.019 binomial error). The 2k rung is hit harder: outlier loses 0.14 / 0.18 and
the 2k mean 0.039 / 0.051. Contradiction is about 0.01 lower in all 4 cells (about 1 SE each, same sign every time).
NQ and oolong do not move. So it is clearly worse, as expected, but not "much worse" on 3 of the 4 tasks. Training CE
is the same as the recipe's, so the cost is the train/eval position mismatch, not worse fitting. One seed per budget.

## Setup
- Recipe `lora64all32r20to7` (see `records/perrow-floor-recipe-ablation.md`):
  - Qwen3.5-4B (marker-repaired base), whole-model LoRA r64 (scale 32), online router.
  - Stages `7:0.2,16:0.07`, `--st-otr-doc-share 1.0`, LR 1e-5, 16 rows/step, mt4 fast8k shard, suffix `-tde`.
  - Kept tokens keep their original position ids, so RoPE in the full-attention blocks (every 4th block) sees gaps.
- Ablation `lora64all32r20to7pr` = the recipe + `--st-otr-reset-positions`:
  - After each cut, the kept row's `position_ids` are renumbered contiguously (0..k-1 in row order).
  - Stage 2 renumbers stage 1's already-compacted row.
  - Packed rows are renumbered per document.
  - Blocks 0-6 and the dense warmup step are unchanged.
  - Eval is dense with true positions (the usual docchunk ladder), so the model meets positions at eval that it never
    saw in blocks 7+ during training.
- Code (branch `claude/posid-reset-0abac3`, cut from `claude/otr-perrow-minfloor` 635264704, the recipe sd1 control's
  code):
  - `OnlineTokenRouterConfig.reset_positions` in `src/olmo_core/nn/online_token_router.py` (`mid_compact`,
    `_mid_compact_packed`).
  - Flag plumbed through `recipes/train/variants.py` and `train_ctc_suite.py`.
  - Tests:
    - `src/test/nn/online_token_router_test.py` (CPU): positions 0..W-1 after each cut; with the default flag off the
      loss is bit-identical.
    - `src/test/nn/online_router_packed_test.py` (GPU): packed matches padded under reset.
  - Launcher `research/token_routing/online_router/launch_posid_reset.py`; collector `collect_posid_reset.py`.
- Smoke (mooney 1x H200, slurm 4150009-11 / 4150013-14):
  - Step 2 selects the same tokens under all settings.
  - Step-2 CE is 1.237 / 1.239 for the new default, against 1.226 / 1.236 for the old code (that spread is GPU
    nondeterminism).
  - Reset gives 1.348.
- Training runs (Beaker, urgent):
  - f1952: 01M6JTWWY7PDKMQ3YZ4TT7V6K6
  - f3904: 01M6JTY2EEQ82843KR8DQ87FYB
- Evals: 8 jobs, `launch_posid_reset.py eval`, eval_size 500 per cell, no-cot.
  - Same launcher and outlier rung files as the recipe evals.
  - Every eval log has the "adapters re-installed" line.

## Results (eval_size 500 per cell, no-cot; recipe = two seeds s0 / sd1)

| rows/task | arm | mean@8k | contra 8k | nq 8k | outlier 8k | oolong 8k | mean@2k | contra 2k | nq 2k | outlier 2k | oolong 2k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1,952 | recipe | .845 / .847 | .959 / .956 | .912 / .922 | .862 / .861 | .648 / .649 | .921 / .920 | .977 / .975 | .970 / .970 | .880 / .881 | .856 / .852 |
| 1,952 | **reset** | **.820** | .947 | .914 | **.764** | .653 | **.881** | .963 | .970 | **.737** | .854 |
| 3,904 | recipe | .882 / .889 | .977 / .973 | .918 / .926 | .970 / .967 | .663 / .689 | .949 / .945 | .988 / .985 | .976 / .966 | .972 / .973 | .858 / .856 |
| 3,904 | **reset** | **.870** | .965 | .920 | **.919** | .677 | **.896** | .982 | .958 | **.795** | .849 |

Binomial SE at eval_size 500: outlier ±0.019 at 0.76, ±0.012 at 0.92, ±0.018 at 0.80; contradiction ±0.010 at 0.95.

Training (same data, steps and FLOPs; keep fractions identical: 0.2056 overall, 0.3845 of stage 1 at stage 2):

| rows/task | arm | PF | CE first 50 steps | CE last 25% | CE last 20 steps | 8k FTM |
|---|---|---|---|---|---|---|
| 1,952 | recipe | 345.3 / 343.2 | .452 / .447 | .0956 / .0979 | .083 / .093 | 4.56x / 4.69x |
| 1,952 | reset | 343.4 | .468 | .1005 | .092 | >=4.46x |
| 3,904 | recipe | 688.1 / 686.0 | .464 / .471 | .0580 / .0639 | .049 / .059 | 3.61x / 3.93x |
| 3,904 | reset | 685.9 | .501 | .0587 | .050 | 3.13x |

## Reading
- Training under renumbered positions is self-consistent, so CE reaches the recipe's level by the end. Only the first
  ~50 steps run slightly higher, as the model adapts to the new position statistics.
- The penalty shows up only when scoring with true positions. It is concentrated on outlier, which asks for every
  document whose topic or rating differs from the majority. That means comparing all documents with one another and
  reporting their IDs. NQ and oolong do not move. This ablation does not show why outlier, and not oolong (also an
  aggregation task), is the position-sensitive one.
- The model gets much of its order information from the GDN blocks (no positions) and from the partial RoPE (25% of
  head dims, theta 1e7). That is likely why the damage is so localized.
- Not explained: the 2k rung is hurt more than 8k, and at 3,904 rows outlier@2k barely improves with data (0.737 ->
  0.795, vs recipe 0.880 -> 0.972).

## Limits
- One seed per budget, though the outlier gaps are 3-10x the recipe seed spread and binomial error.
- Recorded rows carry no `git_commit` (an `ops result add --check` provenance warning). The commits are:
  - training: 6fc2fc48f (the flag);
  - evals: 7d1763487, printed in each eval JSON's `_meta`.

## Where things are
- Numbers: `ops result show --train-run f8k-mt4-lora64all32r20to7pr-f1952-tde-lr1e-05` (and `-f3904-`), experiment
  `tokdrop_lr`.
- State: `research/token_routing/online_router/posid_reset_state.json`.
- Checkpoints: `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts/f8k-mt4-lora64all32r20to7pr-{f1952,f3904}-tde-lr1e-05`.
- Eval JSONs: `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/<run>-ev<task>/eval/<task>_multirung.json`.
