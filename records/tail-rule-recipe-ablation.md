---
title: Fixed per-document tail rule instead of the router on the token-drop recipe (lora64all32t10, lora64all32t20to7)
date: 2026-11-04
updated: 2026-11-04
experiment: tokdrop_lr
area: token_routing
kind: result
status: done (1 seed per arm at 1,952 rows/task; no 2nd seed, both far from the recipe)
superseded_by: 
artifact: https://claude.ai/artifact/SPSBxHFs2PUn5CPjJX3Wgt
summary: Replacing the router with a content-blind "keep each document's last tokens" rule costs 0.065 on the 8k 4-task mean at the recipe's own FLOP-matched shape (tail20to7 0.781 vs recipe 0.845 / 0.847) and 0.10 with one 10% cut (tail10 0.747); both are below dense at 1,952 rows (0.843). Outlier carries most of the loss, then nq at 8k. The model does not simply adapt to the drop shape.
---
# Tail-rule ablation on the recipe

**Verdict.** The model does not just adapt to whatever drop shape it trains with. With the router swapped for a fixed
rule (each document keeps its last body tokens), at the recipe's own two-cut shape, keep fractions and FLOPs
(`tail20to7`), the 8k 4-task mean drops from 0.845 / 0.847 to 0.781 (-0.065). The 2k mean drops from 0.921 / 0.920 to
0.851. A single cut keeping the last 10% at block 7 (`tail10`) is worse still: 0.747 at 8k, 0.815 at 2k. Both arms are
below dense at the same 1,952 rows/task (0.843), so their FLOPs-to-match is under 4.47-4.56x and cannot be pinned (no
smaller dense point). Outlier carries most of the loss (8k -0.15 / -0.25, 2k -0.22 / -0.38), then nq at 8k
(-0.06 / -0.11). Contradiction loses about 0.04 at 8k; oolong barely moves. One seed per arm, but the gaps are 3-20x
the recipe's seed spread and the binomial error.

## Setup
- Recipe `lora64all32r20to7` (see `records/perrow-floor-recipe-ablation.md`): Qwen3.5-4B (marker-repaired base),
  whole-model LoRA r64 (scale 32), online router, stages `7:0.2,16:0.07`, `--st-otr-doc-share 1.0`, original position
  ids, LR 1e-5, 16 rows/step, mt4 fast8k shard, 1,952 rows/task, suffix `-tde`.
- `--st-otr-score tail` (new): the router score is replaced by minus each token's distance to its document's end, so
  the existing per-document top-k (ceil(k * |body|), at least 1) keeps each document's last body tokens. Non-body
  tokens (prompt, markers, query, answer) are always kept, as in the recipe. No exploration. A second cut ranks by the
  original distance, so the kept sets are nested. The router is still fit (it gates the dense warmup), but none of its
  scoring / target / fit FLOPs are charged.
  - `tail10` (`lora64all32t10`): one cut at block 7, last 10%.
  - `tail20to7` (`lora64all32t20to7`): last 20% at block 7, last 7% (nested) at block 16. FLOP-matched to the recipe.
- Code (branch `claude/posid-reset-0abac3`):
  - Rule: b471298dc, `OnlineTokenRouter._choose` and `_flop_router` in `src/olmo_core/nn/online_token_router.py`;
    `--st-otr-score tail` in `train_ctc_suite.py`.
  - Test: `test_tail_rule_keeps_exactly_each_documents_tail_cpu` in `src/test/nn/online_token_router_test.py`. It
    checks that the kept set equals each document's exact tail after 1 and 2 cuts, and that no router FLOPs are
    charged. All 90 router tests pass (mooney, slurm 4152635).
  - Launcher `research/token_routing/online_router/launch_tail_rule.py` (state `tail_rule_state.json`); collector
    `collect_posid_reset.py --state tail_rule_state.json`.
- Smoke: mooney 1x H200, 4 steps, slurm 4152752-3. Keep 0.107 / 0.103 for tail10. tail20to7 kept 0.2057, then
  0.0796 absolute at block 16 (= the recipe). PF 0.538 / 0.544 vs the recipe smoke's 0.545.
- Training runs: Beaker, urgent, commit b076f2571. tail10 01M6MC1WN4S0QGCFFYMV89820T; tail20to7 01M6MC2T4E8HMVRZAHRM38X939.
- Evals: 8 jobs, `launch_tail_rule.py eval`, eval_size 500 per cell, no-cot, same rung files as the recipe evals.
  Every eval log has the "adapters re-installed" line.

## Results (eval_size 500 per cell, no-cot; recipe = two seeds s0 / sd1)

| arm | mean@8k | contra 8k | nq 8k | outlier 8k | oolong 8k | mean@2k | contra 2k | nq 2k | outlier 2k | oolong 2k |
|---|---|---|---|---|---|---|---|---|---|---|
| recipe | .845 / .847 | .959 / .956 | .912 / .922 | .862 / .861 | .648 / .649 | .921 / .920 | .977 / .975 | .970 / .970 | .880 / .881 | .856 / .852 |
| **tail20to7** | **.781** | .921 | .856 | **.711** | .637 | **.851** | .952 | .956 | **.658** | .838 |
| **tail10** | **.747** | .919 | .810 | **.617** | .640 | **.815** | .958 | .966 | **.497** | .839 |

Other points (8k mean, 1,952 rows/task unless noted): dense 0.843; posid-reset 0.820; recipe with random token choice
0.782 at 1,464 rows/task. Binomial SE at eval_size 500: ±0.020 at 0.71, ±0.022 at 0.62, ±0.022 at 0.50, ±0.016 at
0.86, ±0.012 at 0.92.

Training (same data and steps):

| arm | PF | keep frac | stage-2 keep (of stage 1) | CE first 50 steps | CE last 25% | CE last 20 steps | 8k FTM |
|---|---|---|---|---|---|---|---|
| recipe | 345.3 / 343.2 | .2056 | .3845 | .452 / .447 | .0956 / .0979 | .083 / .093 | 4.56x / 4.69x |
| tail20to7 | 342.5 | .2056 | .3845 | .660 | .1166 | .110 | <=4.47x (below dense f1952) |
| tail10 | 335.6 | .1063 | - | .764 | .1199 | .115 | <=4.56x (below dense f1952) |

Train CE is measured on the routed forward, so it is the loss under each arm's own drop.

## Reading
- At identical keep fractions and FLOPs, which tokens are kept matters: the router beats the fixed tail by 0.065 at 8k
  and 0.07 at 2k. The model trains under the tail rule but stays about 0.02 CE above the recipe through the end.
- The loss is concentrated on outlier, which needs every review's rating or category. A document's tail does not
  always carry it, so the model has nothing to compare at blocks 7+. NQ at 8k (one gold passage among many) also
  loses 0.06-0.11, while NQ at 2k is nearly intact. Oolong, already the weakest task, barely moves.
- tail10 keeps half as many tokens in blocks 7-15 and twice as many in blocks 16+, compared with tail20to7. It is
  worse on outlier (-0.09 at 8k, -0.16 at 2k) and nq@8k (-0.05); on the other cells the two are within noise. The
  extra late tokens do not make up for the deeper early cut.
- tail20to7 lands at the 8k mean of recipe-random (0.782), but random ran at 1,464 rows/task, so it is not a matched
  comparison.

## Limits
- One seed per arm. Not run: a second seed (both arms are over 0.06 below the recipe) or larger budgets.
- Recorded rows carry no `git_commit` (an `ops result add --check` provenance warning). Training commit is b076f2571;
  each eval JSON's `_meta` prints its commit.

## Where things are
- Numbers: `ops result show --train-run f8k-mt4-lora64all32t10-f1952-tde-lr1e-05` (and `...t20to7...`), experiment
  `tokdrop_lr`.
- State: `research/token_routing/online_router/tail_rule_state.json`.
- Checkpoints: `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts/f8k-mt4-lora64all32{t10,t20to7}-f1952-tde-lr1e-05`.
- Eval JSONs: `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/<run>-ev<task>/eval/<task>_multirung.json`.
