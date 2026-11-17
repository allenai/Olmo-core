---
title: Early dense-eval loss as a cheap predictor of a FLOP-drop method's final score
date: 2026-11-16
updated: 2026-11-16
experiment: tokdrop_lr
area: token_routing
kind: result
status: partly (11-16): step-10/20 dense eval CE ranks 19 arms with Spearman 0.83-0.84 vs the final 8k mean and flags every document-drop / tail / low-keep failure; 30-50 steps rank a 9-arm subset at 0.95 and the step-50 outlier CE also catches wtsS1; cannot resolve gaps under ~0.03-0.05 final
superseded_by: 
artifact: 
summary: Train each FLOP-drop method with its own forward, optimiser and real LR schedule for 10-20 steps from the repaired Qwen3.5-4B base on the same mt4 batches, and score the DENSE forward's answer CE on fixed held-out eval-rung rows. Over 19 arms x 2 seeds with known final 8k means (0.554-0.853), the step-10 and step-20 CE rank arms with Spearman 0.83 / 0.84 (0.82 / 0.85 with only 16 rows per task); steps 1-5 are uninformative (warmup and answer format). A step-20 threshold separates every arm that ended >= 0.84 from every arm that ended < 0.80 except wtsS1 (outlier collapse not visible). Extending to 30-50 steps (9 arms) raises it to 0.95 and the step-50 outlier CE ranks the outlier collapses (0.97). One check costs ~50 s (10 steps), ~95 s (20), ~130 s (30) or ~215 s (50) of 2-GPU time after a ~49 s model load.
---
# Early dense-eval loss as a cheap predictor (`early_loss_proxy`)

Prasann (2026-11-16): "a sanity-check for a cheap metric for predicting success: when doing recipe training, does the
dense eval loss (eval with dense format) go down as much as it normally does in the first X (e.g. 5-10) steps, and is
there a relationship between that and how well a given approach will work? ... this check should take less than a
minute to run" (later: "I want a check that can run in less than a minute ... carefully track runtime").
Code: `research/token_routing/early_loss_proxy/` (branch `claude/early-loss-proxy-d6e886`).

## Verdict

**Partly.**
- **Steps 1-5 are uninformative.** They are dominated by LR warmup and answer-format learning: every arm's
  suite8k CE falls from 1.69 to about 0.4 by step 5, and arms at LR 1e-5 fall about twice as fast as dense full FT
  at 5e-6.
- **From step 10, the dense eval CE ranks the 19 arms like their final 8k mean.** Spearman is **+0.83 at step 10**
  and **+0.84 at step 20** (19 arms, 2 seeds each, final means 0.554-0.853). With only the 17 method arms it is
  +0.77 / +0.79.
- **It holds with 16 held-out rows per task** (+0.82 / +0.85), and on train-distribution held-out rows (+0.75 / +0.83).
- **It separates good arms from failures, with one miss at step 20.** At step 20, every arm that ended **≥ 0.84**
  (dense, dense LoRA, recipe, oracle 0.95, random 19%) has CE ≤ 0.337. Every arm that ended **< 0.80** has CE ≥ 0.357,
  except **wtsS1** (0.319, final 0.790). At step 10 **oracle 0.80** (0.347, final 0.721) also slips through.
  - Pairwise AUC between the two groups: 0.96 at step 10, 0.98 at step 20.
  - Both misses are **outlier collapses** (outlier 0.67 and 0.41 at 8k).
  - The outlier CE flags oracle 0.80 by step 20 (outlier drop 0.685 vs ≥ 0.795 for the good arms) but not wtsS1 (0.811).
- **It would have caught the document-drop failure.** ddp and ddpg are at step-20 excess CE +0.146 / +0.147 over
  dense LoRA, among the worst four arms. Almost all of it is contradiction (drop 0.28 / 0.11 vs 0.53-0.58).
- **30-50 steps help** (9-arm subset, both seeds; same arms at step 10 / 20: Spearman 0.93 / 0.85):
  - Spearman is **+0.95 at both step 30 and step 50**.
  - At step 30 the 4-task CE puts every ≥ 0.84 arm below every < 0.80 arm. wtsS1 is only 0.011 behind random 19%,
    inside the noise.
  - At step 50 the per-task **outlier** CE ranks the 9 arms at **+0.97**; wtsS1's outlier drop is the lowest of the
    non-doc-drop arms (0.818 vs 0.842-0.880). That is the late signal steps 10-20 lacked.
  - Cost: ~130 s (30 steps) or ~215 s (50 steps) per recipe-like check, over the 1-minute target.
- **What it cannot see:**
  - **Gaps under about 0.03-0.05 in the final mean.** The decent methods (recipe, wtsS0wr, oracle 0.85/0.90,
    random 19%) sit within 0.05 CE of each other, and the same arm's two seeds differ by a median 0.027 (max 0.071).
  - **Failures that develop later in training** (wtsS1's outlier collapse).
  - **nq and oolong.** Per-task Spearman is 0.26-0.31 for nq and about 0 for oolong at steps 10-20; their final
    scores barely differ across arms (nq 0.81-0.94, oolong 0.58-0.67), so there is little to rank.
- **Cost.** After a ~49 s model load (2 H200s), a recipe-like check costs **~51-54 s for 10 steps** (4.2 s/step) plus
  one 64-row probe (≈ 7.5 s, measured from the 256-row probe at ≈ 24 s). 20 steps cost **~92-96 s**. The step-0 probe
  is the same base model for every arm and can be cached. Oracle arms cost about 22 s/step (≈ 4 min per check);
  dense full FT costs 10.2 s/step.

## Setup

- **Arms.** Each arm is its full-run training command, flag for flag (`launch_proxy.ARMS`):
  - repaired base `q35-4b-base-markerfix`;
  - fresh 4-task pool `mt4f1952`;
  - 16 rows/step, micro 2, 2 GPUs, seq 12288;
  - the arm's own LR (dense full FT 5e-6; everything else 1e-5, whole-model LoRA r64 scale 32);
  - the real schedule of the 488-step run: linear warmup over 15 steps, so the LR at step 10 is 2/3 of peak and at
    step 20 is 0.99 of peak.

  The trainer is unmodified; `dense_probe.py` attaches a callback and hard-stops at the last probe step. Nothing is saved.
- **Held-out rows** (`build_heldout.py`, built once per node):
  - `suite8k` / `suite2k`: the first rows of the four 8k / 2k eval rung files, tokenized like the training shards
    (query after, no CoT, documents wrapped in markers, oolong chunked by line), with the gold answer as the loss span.
  - `train`: pool rows 1952+ of `mt4f3904` (f1952 is a byte prefix of f3904, checked).
  - Prompt-hash overlap with the 7,808 training rows: **0**. Only templated question strings repeat.
  - This sweep: suite8k 32 rows/task (eval_size 128 rows ⚠ well under 500; it is a probe, not a score), suite2k and
    train 16/task.
- **Probe.** `model.eval()` turns off every routing / compaction / drop path, which are all gated on
  `self.training`. Each row is scored at B=1, unpadded, with positions 0..T-1; LoRA adapters stay in the forward.
  The metric is the teacher-forced answer CE, averaged per row over the answer tokens.
- **Measured steps:** 0, 1, 2, 5, 10, 20 (50 in a follow-up job, below).
- **Checks passed:**
  - Step 0 is identical across all 38 runs (max spread 0.0 on every set and task).
  - The rank-0 batch hashes are identical across arms at each seed (data seed 34521 + seed), so every arm sees the
    same batches in the same order.
  - The recipe's step 1 is dense: its online-router log shows `dense=1` at step 1 and keep 0.205 from step 2. So step
    1 is before dropping starts and steps 2+ are after. The wts arms route from step 1 (`--st-otr-warmup-random`).
  - The oracle and tail-rule code was ported from `claude/oracle-thresh-sweep-69bc7d` and `claude/posid-reset-0abac3`
    (minimal hunks, see the README). The random-keep arms use the nominal flags (19 / 17 / 11 / 9%) that realised
    19.7 / 17.7 / 11.7 / 9.7%.
- **Noise.**
  - Paired SE of an arm's 4-task CE difference against dense on the same rows: median 0.013 (0.007-0.020) with 32
    rows/task.
  - Seed-to-seed |difference| of an arm's drop: median 0.027, max 0.071 (orc90). Seed noise is the larger term.

## Results (suite8k, 4-task mean of per-task CE, seed mean; final = known 8k 4-task mean at 1,952 rows/task, eval_size 500/task, no-cot)

"excess vs dense LoRA" = CE(arm) − CE(dense LoRA) at the same step (dense LoRA uses the same LR and
parameterisation as every method arm). "frac, same FLOPs" = the arm's drop ÷ the dense full-FT drop interpolated at
the arm's cumulative FLOPs.

| arm | final 8k | CE step 10 | CE step 20 | excess vs dense LoRA 10 / 20 | outlier drop 20 | contra drop 20 | frac of dense drop, same step 10 / 20 | frac, same FLOPs 10 | s / step |
|---|---|---|---|---|---|---|---|---|---|
| denselora | 0.853 | 0.271 | 0.269 | +0.000 / +0.000 | 0.845 | 0.576 | 1.03 / 1.00 | 1.2 | 10.4 |
| orc95 | 0.850 | 0.281 | 0.299 | +0.010 / +0.030 | 0.800 | 0.537 | 1.03 / 0.98 | 2.8 | 22.0 |
| recipe | 0.846 | 0.316 | 0.289 | +0.045 / +0.020 | 0.839 | 0.533 | 1.00 / 0.99 | 4.1 | 4.2 |
| dense | 0.843 | 0.318 | 0.271 | +0.047 / +0.002 | 0.849 | 0.558 | 1.00 / 1.00 | 1.0 | 10.2 |
| rnd19 | 0.840 | 0.352 | 0.337 | +0.081 / +0.068 | 0.795 | 0.413 | 0.98 / 0.95 | 2.9 | 3.9 |
| orc85 | 0.837 | 0.334 | 0.331 | +0.063 / +0.062 | 0.809 | 0.458 | 0.99 / 0.96 | 11.8 | 21.7 |
| wtsS0wr | 0.837 | 0.327 | 0.313 | +0.056 / +0.044 | 0.821 | 0.474 | 0.99 / 0.97 | 7.1 | 4.2 |
| orc90 | 0.824 | 0.322 | 0.318 | +0.051 / +0.049 | 0.784 | 0.482 | 1.00 / 0.97 | 6.7 | 21.7 |
| rnd17 | 0.813 | 0.363 | 0.381 | +0.093 / +0.112 | 0.765 | 0.364 | 0.97 / 0.92 | 3.1 | 4.0 |
| rnd11 | 0.797 | 0.392 | 0.406 | +0.122 / +0.137 | 0.643 | 0.373 | 0.95 / 0.91 | 3.8 | 4.2 |
| wtsS1 | 0.790 | 0.333 | 0.319 | +0.063 / +0.050 | 0.811 | 0.405 | 0.99 / 0.97 | 7.6 | 4.2 |
| tail20to7 | 0.781 | 0.371 | 0.418 | +0.101 / +0.149 | 0.555 | 0.355 | 0.96 / 0.90 | 3.9 | 4.2 |
| rnd9 | 0.769 | 0.385 | 0.361 | +0.115 / +0.092 | 0.714 | 0.369 | 0.95 / 0.94 | 4.2 | 4.2 |
| tail10 | 0.747 | 0.385 | 0.391 | +0.115 / +0.122 | 0.613 | 0.342 | 0.95 / 0.92 | 4.0 | 4.2 |
| orc80 | 0.721 | 0.347 | 0.357 | +0.076 / +0.088 | 0.685 | 0.407 | 0.98 / 0.94 | 14.5 | 21.7 |
| ddpg | 0.665 | 0.373 | 0.416 | +0.102 / +0.147 | 0.716 | 0.114 | 0.96 / 0.90 | 6.2 | 4.3 |
| orc70 | 0.648 | 0.418 | 0.389 | +0.147 / +0.120 | 0.687 | 0.300 | 0.93 / 0.92 | 15.3 | 21.7 |
| orc60 | 0.643 | 0.468 | 0.471 | +0.197 / +0.202 | 0.439 | 0.297 | 0.89 / 0.86 | 15.3 | 21.7 |
| ddp | 0.554 | 0.380 | 0.415 | +0.110 / +0.146 | 0.584 | 0.282 | 0.95 / 0.90 | 6.2 | 4.3 |

| held-out set | step | Spearman, 4-task mean (all arms) | methods only | contradiction | nq | outlier | oolong |
|---|---|---|---|---|---|---|---|
| suite8k | 1 | +0.50 (n=19) | +0.60 | +0.38 | -0.26 | +0.56 | -0.48 |
| suite8k | 2 | +0.49 (n=19) | +0.59 | +0.45 | -0.28 | +0.78 | -0.12 |
| suite8k | 5 | +0.68 (n=19) | +0.84 | +0.54 | -0.31 | +0.50 | +0.21 |
| suite8k | 10 | +0.83 (n=19) | +0.77 | +0.93 | +0.26 | +0.71 | -0.07 |
| suite8k | 20 | +0.84 (n=19) | +0.79 | +0.94 | +0.31 | +0.78 | +0.01 |
| suite2k | 1 | +0.13 (n=19) | +0.17 | +0.14 | -0.45 | -0.30 | -0.54 |
| suite2k | 2 | -0.14 (n=19) | -0.12 | +0.22 | -0.56 | -0.29 | -0.19 |
| suite2k | 5 | +0.31 (n=19) | +0.33 | +0.25 | -0.37 | +0.33 | +0.07 |
| suite2k | 10 | +0.75 (n=19) | +0.65 | +0.91 | +0.07 | +0.51 | +0.26 |
| suite2k | 20 | +0.85 (n=19) | +0.80 | +0.93 | +0.12 | +0.46 | +0.61 |
| train | 1 | +0.46 (n=19) | +0.64 | +0.11 | -0.18 | +0.53 | -0.65 |
| train | 2 | +0.48 (n=19) | +0.57 | +0.39 | -0.27 | +0.77 | +0.07 |
| train | 5 | +0.69 (n=19) | +0.86 | +0.52 | +0.04 | +0.54 | +0.31 |
| train | 10 | +0.75 (n=19) | +0.69 | +0.90 | +0.36 | +0.67 | +0.23 |
| train | 20 | +0.83 (n=19) | +0.78 | +0.93 | +0.38 | +0.75 | +0.48 |

How to read the "fraction of the dense drop" views:
- **Same step:** every LoRA-at-1e-5 arm drops faster than dense full FT through step 5 (×1.22-1.33), because its LR
  is twice dense's. By step 10-20 the fractions converge to 0.86-1.03. The ordering inside that narrow band is what
  carries the signal.
- **Same FLOPs:** every method looks 3-15× better than dense. This mostly measures dense's slow warmup at a low LR,
  not method quality.
- **Oracle FLOPs are undercounted.** The flop meter charges only their routed pass, not the dense pass and attribution
  each step runs; their wall clock is 22 s/step. Their same-FLOP fractions are not comparable.

## 30 and 50 steps (9 arms x 2 seeds; jobs 5149886/5149887)

Arms: recipe, ddp, ddpg, wtsS1, rnd19, rnd9, tail20to7, dense LoRA, dense. Probes at 0, 30, 50. LR at step 30 and 50:
0.97 and 0.93 of peak.

| arm | final 8k | CE step 30 | CE step 50 | excess vs dense LoRA 30 / 50 | outlier drop 30 / 50 | contra drop 50 | outlier final |
|---|---|---|---|---|---|---|---|
| denselora | 0.853 | 0.218 | 0.210 | +0.000 / +0.000 | 0.884 / 0.880 | 0.693 | 0.858 |
| recipe | 0.846 | 0.269 | 0.244 | +0.051 / +0.034 | 0.850 / 0.872 | 0.612 | 0.861 |
| dense | 0.843 | 0.243 | 0.224 | +0.025 / +0.014 | 0.883 / 0.856 | 0.702 | 0.817 |
| rnd19 | 0.840 | 0.277 | 0.264 | +0.059 / +0.054 | 0.855 / 0.847 | 0.573 | 0.857 |
| wtsS1 | 0.790 | 0.289 | 0.263 | +0.070 / +0.053 | 0.853 / 0.818 | 0.576 | 0.670 |
| tail20to7 | 0.781 | 0.438 | 0.297 | +0.220 / +0.087 | 0.682 / 0.845 | 0.508 | 0.711 |
| rnd9 | 0.769 | 0.315 | 0.270 | +0.096 / +0.060 | 0.793 / 0.842 | 0.499 | 0.689 |
| ddpg | 0.665 | 0.462 | 0.329 | +0.244 / +0.119 | 0.674 / 0.799 | 0.448 | 0.421 |
| ddp | 0.554 | 0.452 | 0.378 | +0.234 / +0.168 | 0.608 / 0.638 | 0.442 | 0.001 |

- Spearman (suite8k, 4-task mean):
  - 30-50 steps: 0.95 / 0.95. The same 9 arms at steps 5 / 10 / 20 of the v1 sweep: 0.47 / 0.93 / 0.85.
  - Methods only (7 arms): 30-50 steps 0.93 / 0.93, vs 10 / 20 steps 0.86 / 0.71.
  - On train-distribution rows: 0.92 / 0.95.
  - With 16 rows per task: 0.97 / 0.88.
- Per task at step 50: contradiction 0.97, outlier **0.97**, nq 0.60, oolong −0.17.
- Leaving oolong out of the mean (and out of the final mean) does not help: 0.92 / 0.90 at 30 / 50 steps, 0.83 / 0.79
  at 10 / 20 on all 19 arms.
- Seed noise is still the limit: the |difference| between the two seeds' step-50 drops is 0.003-0.123 (ddpg).

## What the proxy is and is not

- **It ranks well** when a method's failure is a loss of signal it can learn in the first ~20 steps: contradiction
  (per-task Spearman 0.93-0.94) and the outlier collapses of the low-keep / tail / document-drop arms (0.71-0.78).
- **It is blind** to failures that build later (wtsS1) and to gaps smaller than seed noise (≈ 0.03 CE at step 20,
  about 0.03-0.05 in the final mean).
- **Practical use:**
  - Under a minute: run a candidate for 10 steps (≈ 51-54 s after load) and compare its suite8k CE with the recipe's
    (0.316 at step 10, seed range ±0.02). This catches the gross failures: document drop, low keep, tail rules.
  - For outlier-type failures, run 50 steps (≈ 215 s) and check the per-task outlier CE against the recipe's
    (0.149-0.181).
  - More than about +0.05 means a likely < 0.80 method; within ±0.03 is undetermined, so run it fully. A second seed
    halves the seed noise and doubles the cost.

## Not done / limits

- **Correlations rest on 19 arms whose final scores are single-seed** (recipe two seeds); seed noise in the final
  means is about ±0.03.
- **The suite8k probe uses the first eval rows.** That is fine for a sanity check, but any policy *tuned* on this
  signal should use the `train` rows (Spearman +0.83 at step 20) to keep the eval set untouched.
- **No constant-LR variant was run.** The real schedule already moves every arm a lot by step 10, so the test was not
  uninformative.

## Files

- Code, README, `results_v1.json`, `results_v1_r16.json` (16 rows/task), `cost_v1.json`, `proxy_v1.png`,
  `experiment_log_entry.json`: `research/token_routing/early_loss_proxy/`.
- Raw per-run JSONs and logs: `mooney:/data/prasann/early_loss_proxy/results/{v1,s50}/`.
- Held-out rows: `mooney:/data/prasann/early_loss_proxy/heldout_v1.npz`.
- Slurm jobs: 5149824 / 5149825 (v1), 5149886 / 5149887 (50 steps), registered with `ops` (`elp-*`).
