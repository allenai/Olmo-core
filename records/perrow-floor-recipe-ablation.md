---
title: Per-row token budget + per-document floor on the token-drop recipe (lora64all32r20to7)
date: 2026-11-03
updated: 2026-11-09
experiment: tokdrop_lr
area: token_routing
kind: result
status: done (1,952 rows/task x 2 seeds; best arm at 3,904 rows/task x 1 seed)
superseded_by: 
artifact: https://claude.ai/artifact/SPSBxHFs2PUn5CPjJX3Wgt
summary: Giving the row (not each document) the online router's 20%/7% token budget, with or without a 1-2 token per-document floor, does not beat the recipe's per-document budget at 8k (1,952 rows/task, 2 seeds: no floor 0.843 vs recipe 0.846, within noise; floors 0.831/0.840, losing on outlier; 3,904 rows/task, 1 seed: no floor 0.880 vs recipe 0.882/0.889), though the router then concentrates on gold documents (absolute leak_auc 0.73-0.75 vs 0.48).
---
# Per-row budget + per-document floor on the recipe

**Verdict.** At 1,952 rows/task the per-row budget does not beat the per-document recipe at the 8k rung, at
matched data or matched FLOPs. With no floor it ties (0.843 vs 0.846, 2 seeds each, within noise). A floor of
1 or 2 tokens per document is slightly worse (0.831 / 0.840), and almost all of the loss is on outlier. The
per-row arms use 2.3% fewer FLOPs, because the recipe's per-document rounding keeps 20.6% / 7.9% instead of
20.0% / 7.0%, which is far too small to move FLOPs-to-match (4.55x vs 4.62x). At 3,904 rows/task the no-floor arm
(one seed) gives 0.880 at 8k against the recipe's 0.882 / 0.889 (2 seeds), so it still ties or trails slightly
(FLOPs-to-match 3.58x vs 3.61x for the recipe's seed 0).

## Setup
- Recipe `lora64all32r20to7`: Qwen3.5-4B (marker-repaired base), whole-model LoRA r64 (scale 32), online
  token router.
  - Every token goes through blocks 0-6.
  - Block 7 keeps 20% of the body tokens; block 16 keeps 7% (absolute). Exploration share is 10%.
  - Fast8k 4-task mix (contradiction, nq, outlier, oolong), fresh pool, 16 rows/step, seq 12,288, LR 1e-5,
    2 GPUs.
- Arms (all start from the recipe's flags):
  - **recipe**: `--st-otr-doc-share 1.0`. Each document gets its own share of the budget.
  - **ds0**: `--st-otr-doc-share 0`. The row's budget is spread over documents by router score.
  - **ds0f1 / ds0f2**: ds0 plus `--st-otr-min-per-doc 1|2`. Each document keeps at least K tokens, counted
    against the budget, at both stages (the later stage inherits the floor via `dataclasses.replace`).
- Budget: 1,952 rows/task, seeds 0 and 1. The recipe's seed 0 is its original run, 01M5VH5APW70Y48FW01Y7W3BVF.
- Eval: native ladder, docchunk, query after, no-cot, 8k and 2k rungs, eval_size 500 per task and rung.
  - Metric: f1, except oolong, which uses score.
  - SE of a single run's 4-task mean is about 0.008; the seed-to-seed spread is up to 0.012.
- Hardware: training and eval on Beaker jupiter H100, urgent, 2 GPUs. Smoke test on jsteinhardt cubbins
  H200 (slurm 4145927).

## Results (1,952 rows/task)
Two-seed means; the per-seed 8k means are in brackets.

| arm | 8k mean | 2k mean | PF | FTM (8k) | contra 8k | nq 8k | outlier 8k | oolong 8k |
|---|---|---|---|---|---|---|---|---|
| recipe (per-document) | **0.846** [0.845, 0.847] | 0.920 | 344 | 4.62x | 0.958 | 0.917 | 0.861 | 0.649 |
| per-row, no floor | 0.843 [0.838, 0.848] | 0.921 | 337 | 4.55x | 0.956 | 0.919 | 0.858 | 0.639 |
| per-row, floor 1 | 0.831 [0.837, 0.824] | 0.914 | 337 | ≥4.55x | 0.956 | 0.922 | 0.816 | 0.629 |
| per-row, floor 2 | 0.840 [0.846, 0.835] | 0.905 | 337 | ≥4.55x | 0.956 | 0.929 | 0.835 | 0.643 |

FTM = the dense-curve PF needed to reach the arm's 8k mean, divided by the arm's PF. The dense fresh curve is
1,531 / 3,063 / 4,595 / 6,128 PF → 0.843 / 0.899 / 0.901 / 0.904. "≥" marks a mean below the first dense
point.

Router diagnostics, averaged over steps 50 and later (the two seeds agree to ±0.03):

| arm | keep after block 7 | keep after block 16 (absolute) | leak_auc stage 1 | leak_auc after both cuts | floor tokens (stage 1 / stage 2) |
|---|---|---|---|---|---|
| recipe | 0.206 | 0.079 | 0.48 | 0.48 | - |
| per-row, no floor | 0.200 | 0.070 | 0.60-0.62 | 0.73 | - |
| per-row, floor 1 | 0.200 | 0.070 | 0.62-0.63 | 0.75 | 0.5% / 4.5% |
| per-row, floor 2 | 0.200 | 0.070 | 0.62-0.64 | 0.73-0.75 | 1.4% / 11% |

leak_auc is the AUC of a document's kept fraction as a predictor of the gold flag; 0.5 means no signal.
"After both cuts" is the absolute kept fraction after block 16, a new diagnostic (`s1_leak_auc_abs`). With
the per-row budget the router sends tokens to the gold documents, but the answers do not improve. Outlier
needs coverage of every document, so it is the task that loses when the floor's forced tokens displace
scored ones.

## Runs
| run | Beaker |
|---|---|
| f8k-mt4-lora64all32r20to7ds0-f1952-tde-lr1e-05 | 01M6F3THAA49E0WV59CG38PZTW |
| f8k-mt4-lora64all32r20to7ds0f1-f1952-tde-lr1e-05 | 01M6F3VA0RJX2X94Z7ZY0QBE0F |
| f8k-mt4-lora64all32r20to7ds0f2-f1952-tde-lr1e-05 | 01M6F3W357QSXWEN0FAQ71YEHY |
| f8k-mt4-lora64all32r20to7ds0sd1-f1952-tde-lr1e-05 | 01M6F3WYDK7BY2P70WBA53SYS5 |
| f8k-mt4-lora64all32r20to7ds0f1sd1-f1952-tde-lr1e-05 | 01M6F3XQTP4EH8NCR8DBY47PC7 |
| f8k-mt4-lora64all32r20to7ds0f2sd1-f1952-tde-lr1e-05 | 01M6F3YGRVME6VS9PZG5R3GY1Y |
| f8k-mt4-lora64all32r20to7sd1-f1952-tde-lr1e-05 (recipe seed 1) | 01M6F3ZAR3PENM133YFNVHDTB5 |
| f8k-mt4-lora64all32r20to7ds0-f3904-tde-lr1e-05 | 01M6HCRGTD1AZB2YKX2FAQ53TR (evals 01M6HWBW6AXCAD4JZ4NN0GBT1C, 01M6HWDBD4QEXPS4XZJ6P407CJ, 01M6HWEB4RS65714RSND0PCP6S, 01M6HWF9NA94Z1PN1KKEKQPP57) |

The 28 eval job ids are in `research/token_routing/online_router/perrow_floor_state.json` (`evals`).

## Results (3,904 rows/task, best arm only)
Per-row, no floor, seed 0 only (Beaker 01M6HCRGTD1AZB2YKX2FAQ53TR, 976 steps, 672.5 PF). Recipe: seed-0 per-task
numbers (01M5VH6FK7J23RHG272XB796M5, 688 PF); its 8k/2k means are given for both seeds.

| arm | 8k mean | 2k mean | PF | FTM (8k) | contra 8k / 2k | nq 8k / 2k | outlier 8k / 2k | oolong 8k / 2k |
|---|---|---|---|---|---|---|---|---|
| recipe (per-document), 2 seeds | **0.886** (0.882, 0.889) | 0.947 (0.949, 0.945) | 688 | 3.61x (seed 0) | 0.977 / 0.988 (seed 0) | 0.918 / 0.976 | 0.970 / 0.972 | 0.663 / 0.858 |
| per-row, no floor, 1 seed | 0.880 | 0.950 | 672.5 | 3.58x | 0.979 / 0.992 | 0.928 / 0.976 | 0.950 / 0.959 | 0.661 / 0.872 |

eval_size 500 per task and rung. The 2.3% FLOP saving holds (672.5 vs 688 PF). The 8k gap to the recipe's 2-seed
mean (-0.006) is under one SE of a single run's 4-task mean (~0.008). Outlier is again the per-row arm's weakest task
(0.950 vs 0.970). For context, position reset at 3,904 rows/task scored 0.870 at 8k. Router diagnostics: keep 0.200 /
0.070 (absolute), leak_auc 0.64 at stage 1 and 0.76 after both cuts, so the gold concentration grows slightly
with data and still does not raise the scores.

## Where things live
- Code is on branch `claude/otr-perrow-minfloor`; training ran commit dac2b81d5 (router code as in be4c1a1c8).
  - Router diagnostics and the CPU test (`test_pyramid_per_row_budget_with_floor_cpu`):
    `src/olmo_core/nn/online_token_router.py` and `src/test/nn/online_token_router_test.py`.
  - Launcher, including eval mode: `research/token_routing/online_router/launch_perrow_floor.py`.
  - Collector (PF, router diagnostics, FTM): `collect_perrow_floor.py` in the same folder.
  - Local jsteinhardt wrapper: `run_perrow_floor_local.sh` in the same folder.
- Results are in `ops result` (train_run = the run names above). They were recorded with the 3dcae5917
  version of `ds64_fast8k/record_beaker_evals.py`
  (`--state research/token_routing/online_router/perrow_floor_state.json --suffix=-tde`).

## Limits
- Two seeds each. The no-floor vs recipe gap (-0.003) is well inside the noise. The floor-1 gap (-0.016) is
  about 2 SE and comes from one task (outlier).
- 3,904 rows/task has one seed of the per-row arm only, and no floor arms.
- The native ladder logs report no parse rate. Every score is 0.66-0.99 with no zeros, and every LoRA eval log has
  the 'adapters re-installed' line and no MISSING lines.
