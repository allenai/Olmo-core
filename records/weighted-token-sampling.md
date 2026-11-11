---
title: Weighted token sampling — sampled token survival with weights w = 1/p in attention and DeltaNet, one FLOP price instead of the recipe's hand-set two-stage top-k
date: 2026-11-10
updated: 2026-11-10
experiment: tokdrop_lr
area: token_routing
kind: result
status: in progress (S1-S3 scored at 1,952 rows: all trail the recipe on outlier; round 2 running)
superseded_by: 
artifact: 
summary: Replaces the recipe's fixed top-k token drop (20% after block 7, 7% after 16) with sampled survival (exact inclusion probabilities), token weights w = 1/p carried into attention (log w key bias) and Gated DeltaNet (exact w-fold write), an IPW router fit, and a price rule p = clip(s / (tau sqrt(c)), pmin, 1) with tau set by a FLOP controller. Staged S0-S5; results tables below.
---
# Weighted token sampling (`wts`)

Branch `claude/weighted-token-sampling-041da9` (base: `claude/online-router-1bc01a` 8ce5fd22a + the preemption-resume
commit 8980cfc55 from `claude/oracle-thresh-sweep-69bc7d`). Experiment `tokdrop_lr`. Run tags start with `wts`.

## Goal

The recipe `lora64all32r20to7` keeps the top 20% of each document's body tokens after block 7 and the top 7% after
block 16, deterministically. Prasann wants an end-to-end rule instead of two hand-set thresholds, optimised for FLOP
Pareto optimality, with no dense steps in training and peak memory bounded. Judged by best-case per-step FLOP speedup
and empirical 8k FLOPs-to-match (FTM) against the dense curve.

## Method

1. **Token weights.** Every token carries `w` (how many tokens it stands for), 1 at the input. At a cut, token `t`
   survives with probability `p_t`; a survivor's weight becomes `w / p_t`; the rest leave the stack. Prompt,
   markers, query and answer have `p = 1`.
2. **Weights act only where tokens mix.**
   * Full attention: key `j` gets the additive logit bias `log w_j` for every query. Exact: `w` identical copies of a
     key multiply its softmax mass by `w`. Runs the masked-SDPA path (flash-attn has no additive bias).
   * Gated DeltaNet: one token of weight `w` replaces `w` identical consecutive writes. With unit-norm `k` (the
     kernel L2-normalises it) one step is `S' = a S (I - b k k^T) + b v k^T`, `a = exp(g)`. On the part of `S`
     orthogonal to `k` the state only decays, so `w` steps give `a^w`. Along `k` the read maps `S k -> r S k + b v`
     with `r = a (1 - b)`, so `w` steps give `r^w S k + b (1 - r^w) / (1 - r) v`. A single step with
     `g' = w g`, `b' = 1 - (1 - b)^w` and `v' = c v`, `c = b (1 - r^w) / ((1 - r) b')`, reproduces both exactly.
     The brief's guess (`1 - (1 - b)^w` and `w g`) is exact only without decay: with decay the value also needs the
     rescale `c`. The short causal conv still sees one copy (approximation). Code: `gdn_token_weight` in
     `src/olmo_core/nn/attention/recurrent.py`. Needs `beta` in (0, 1) (Qwen3.5: `allow_neg_eigval=False`).
   * FFN, norms, LM head: per token, unchanged.
3. **Survival probabilities.**
   * Fixed-budget mode (S1, S2): each document keeps its recipe share in expectation:
     `p = mix * kf + (1 - mix) * min(1, lam_d exp(score / temp))`, water-filled so `sum_d p = kf |d|`; `mix = 0.1`
     mirrors the recipe's 10% exploration share. Drawn by systematic sampling along the row (one uniform offset per
     row): exact marginals `p_t`, counts within 1 of their expectation.
   * Price mode (S3+): `p_t = clip(v_t / (tau sqrt(c_l)), pmin, 1)`, `v_t = w_prev exp(score_t)`, `c_l` the
     per-token cost of blocks `l..31` + head as a share of a dense token, one `tau` per step for every cut. A
     controller moves `log tau` by `gain * log(measured step FLOP ratio / target)` (the FLOP meter's ratio of the
     previous step, overhead included). `tau` is calibrated whenever the score source changes (uniform warmup ->
     fitted router). Each stage's keep in `--st-otr-stages` becomes a hard per-row cap on expected survivors
     (fraction of the row's original body), which bounds peak memory.
4. **Router fit.** Same ridge router (layer-7 / layer-16 features, gradient-norm targets from the training
   backward), fitted on survivors with importance weights `1/p` (normalised per row, capped at 50x). With weights
   on, the target is `log ||dL/dh_t|| - log w_t` (a token of weight `w` carries about `w` tokens' gradient).
   `center=none` fits the absolute gradient scale, so one price compares cuts (and rows).
5. **No dense steps.** Every wts arm routes its warmup step at the budget with uniform scores
   (`--st-otr-warmup-random`). The recipe's step 1 is a dense step.

Flags: `--st-otr-wts "k=v,..."` (keys and defaults: `WTS_DEFAULTS` in `src/olmo_core/nn/online_token_router.py`).
Launcher `research/token_routing/wts/launch_wts.py` (Beaker or local `run_wts_local.sbatch`), collector
`collect_wts.py`.

## Checks (tests: `src/test/nn/weighted_token_sampling_test.py`, 19 pass on an H200, cubbins)

* DeltaNet: one weighted token vs `w` repeated writes, reference recurrence in float64 (`w` = 1, 2, 3, 7): max
  abs error < 1e-9; the FLA chunk kernel (the training path) for `w` = 2, 5: within 2e-3 (bf16-free fp32 inputs).
* Attention: a key of weight `w` vs that key duplicated `w` times at the same position: equal to 1e-9.
* All-ones weights: bit-for-bit the standard forward (attention short-circuits to the normal backend; DeltaNet keeps
  `w == 1` positions untouched), at layer level and for the whole router path (keep 1.0).
* Sampler: inclusion frequencies within 0.015 of `p` over 20k draws; systematic counts within 1 of `sum p`.
* Two cuts: second-cut weights compound (`w1 / p2`), targets corrected, fit importance-weighted; the price controller
  moves `tau` by exactly `0.5 log 2` when the measured ratio is twice the target, and the per-row cap holds.

## Baseline (S0)

The recipe at 1,952 rows/task (existing runs on the router branch, code paths unchanged here; the router tests pass
on this branch): 8k mean 0.845 (`f8k-mt4-lora64all32r20to7-f1952-tde-lr1e-05`) / 0.847 (`...sd1...`), 2k 0.921 /
0.919. Measured from the seed-0 Beaker log (2 H100): cumulative FLOP ratio **0.221** of dense (step 1 is dense;
routed steps 0.22), peak allocated memory **26.9 GiB** per GPU (reserved 31.9 GiB). The brief's "0.26x, 49 GiB" do
not match the log; comparisons here use the measured values.

## Results

### 8k scores, 1,952 rows/task, seed 0 (eval_size 500 per cell, no-cot; local slurm eval path, 11-11)

SE of the 4-task 8k mean is about 0.008; per task about 0.01-0.02 (outlier at 0.8: 0.018).

| arm | step cost (x dense) | peak GiB | contra | nq | outlier | oolong | 8k mean | 2k mean | outlier 8k: share with none right |
|---|---|---|---|---|---|---|---|---|---|
| recipe `lora64all32r20to7` (seed 0 / seed 1) | 0.221 / 0.220 | 26.9 / 27.2 | 0.960 / 0.956 | 0.912 / 0.922 | 0.860 / 0.861 | 0.650 / 0.649 | **0.846 / 0.847** | 0.921 / 0.919 | - |
| dense (LR 5e-6, f488 point of the curve) | 1.0 | - | | | | | 0.843 | | - |
| wtsS0wr (recipe top-k, warmup routed) | 0.219 | 24.2 | 0.956 | 0.920 | 0.838 | 0.633 | 0.837 | 0.907 | 0.158 |
| wtsS1 (sampled, no weights) | 0.215 | 24.1* | 0.942 | 0.930 | 0.670 | 0.620 | 0.790 | 0.895 | 0.326 |
| wtsS2a (weights: attention) | 0.215 | 24.2 | 0.945 | 0.914 | 0.811 | 0.638 | 0.827 | 0.903 | 0.184 |
| wtsS2g (weights: DeltaNet) | 0.215 | 24.2 | 0.941 | 0.902 | 0.775 | 0.641 | 0.815 | 0.924 | 0.220 |
| wtsS2b (weights: both) | 0.215 | 24.1* | 0.937 | 0.920 | 0.794 | 0.637 | 0.822 | 0.918 | 0.202 |
| wtsS3 (price, cuts 7 / 16) | 0.214 | 24.6 | 0.934 | 0.914 | 0.725 | 0.609 | 0.795 | 0.929 | 0.274 |
| wtsS3d (price, alloc=doc) | 0.216 | 24.2 | 0.919 | 0.914 | 0.710 | 0.638 | 0.795 | 0.922 | 0.288 |
| wtsS3m (price, 6 cuts) | 0.213 | 25.8 | 0.953 | 0.910 | 0.665 | 0.629 | 0.789 | 0.910 | 0.332 |

\* S1 / S2b logged 30.4 GiB because of the diag leak; the same config without diag (memory probe) peaks at 24.1.
The recipe's 26.9 includes its dense step 1: routed from step 1 (S0wr) it peaks at 24.2. FTM vs the dense curve is an
upper bound for every wts arm (below the lowest dense point, 0.843 at 1,531 PF): `<=` 4.9-5.0x; recipe `=` 4.6-4.7x.

Reading. The gap is almost all outlier (recipe 0.86; wts 0.67-0.81), with contra about -0.02 and oolong about -0.01.
Outlier errors are all-or-nothing (an example gets all three outlier documents or none), so they are not a parsing or
partial-recall effect (outputs checked: no empty or malformed generations). Order: top-k (S0wr) > weights in attention
(S2a) > both (S2b) > DeltaNet (S2g) > no weights (S1). Weights recover most of what sampling loses, attention weights
more than DeltaNet weights. The price rule (S3*) loses outlier again: it keeps less after block 16 (4.5% of the body vs
7%) and uses DeltaNet weights. Per-document allocation (S3d) does not help (0.710 vs 0.725), so empty documents
(15% in S3, 0.2% in S3d) are not the cause. With the oracle sweep (uniform per-document random at 19.7% keep after
block 7 alone: outlier 0.857; at 9.7%: 0.689), the deep cut is the likely place sampling hurts: the recipe's top-k at
block 16 keeps 7% well, a soft sample keeps 7% close to random.

Training-side detail:

Training-side, 1,952 rows/task, 488 steps, 2 H200 (mooney), seed 0:

| arm | step cost (x dense, cumulative) | peak alloc GiB/GPU | train CE steps 1-20 / 21-40 / 41-60 / 61-160 |
|---|---|---|---|
| recipe (Beaker H100) | 0.221 (step 1 dense) | 26.9 | - |
| wtsS1 (sampled, no weights) | 0.215 | 24.1 (memory probe, no diag) | 0.760 / 0.376 / 0.340 / 0.226 |
| wtsS2b (weights attn + gdn) | 0.215 | (diag leak inflated; rerun pending) | 0.775 / 0.390 / 0.351 / 0.242 |
| wtsS3 smoke (price rule, 60 steps) | 0.205 (target 0.216) | 24.25 | 0.758 / 0.386 / 0.330 / - |
| wtsS3 (price, cuts 7 / 16, center=none) | 0.214 | 24.6 | 488 steps: 0.500 / 0.242 / 0.166 / 0.127 (steps 1-60 / 61-160 / 161-325 / 326-488) |
| wtsS3d (as S3, alloc=doc) | 0.216 | 24.2 | 0.523 / 0.263 / 0.174 / 0.128 |
| wtsS3m (price, cuts 7/11/15/19/23/27) | 0.213 | 25.8 | 0.537 / 0.219 / 0.160 / 0.123 |
| wtsS2a (weights attn only) | 0.215 | 24.2 | 0.511 / 0.259 / 0.160 / 0.112 |
| wtsS2g (weights gdn only) | 0.215 | 24.2 | 0.490 / 0.243 / 0.164 / 0.118 |
| wtsS0wr (recipe top-k, warmup routed) | 0.219 | 24.2 | 0.420 / 0.206 / 0.148 / 0.104 |
| (S1, S2b on the same windows) | | | S1 0.492 / 0.226 / 0.150 / 0.112; S2b 0.505 / 0.242 / 0.167 / 0.120 |

Train CE is on survivors, so it is only a coarse screen across arms that keep different tokens; the 8k evals decide.
S3 over the last 200 steps: expected keep 20% of the body after block 7 (recipe 20%) and 22% of the survivors after
block 16 (about 4.5% of the body; recipe 7%), so the price spends less at block 16 than the recipe does. About 15% of
documents have no survivor after block 7 with alloc=row; alloc=doc (S3d) brings that to 0.2%. Caps bind on about 1% of
rows at block 16, never at block 7.

S3 smoke: per-step ratio 0.19-0.22 after step 1 (0.266 at the uniform-score warmup), tau calibrated once at the
uniform -> fitted switch; tau keeps falling (1.6e-3 -> 2.9e-4 over 60 steps) because the fitted score scale drifts
down, and the controller tracks it with a lag (cumulative 0.205, slightly under target). Expected keep after block 7
about 14% of body (recipe 20%), after block 16 about 13% of the survivors; per-row caps never bind after step 0. Up to
15-20% of documents have no survivor after block 7 (alloc=row lets the price drop whole low-gradient documents; the
alloc=doc arm wtsS3d keeps every document's share).

## Hypothesis log

| date | hypothesis | outcome |
|---|---|---|
| 11-10 | `1 - (1-b)^w`, `w g` alone reproduce w writes | false with decay: value needs the rescale `c`; with it exact (tests) |
| 11-11 | sampling loses only through IPW noise; weights fix it | partly: weights recover 0.67 -> 0.81 outlier, still under top-k 0.84-0.86 |
| 11-11 | price rule needs alloc=doc to keep every document (outlier) | no: S3d 0.710 vs S3 0.725; empty documents are not the cause |
| 11-11 | soft selection at the deep cut costs outlier; sharper scores (temp 0.5, 0.2) close the gap | round 2 running (S2aT5, S2aT2, S1T2, S3aT5) |
| 11-10 | peak memory creeps 26.9 -> 30.5 GiB in S1/S2b | diag (`--st-otr-diag-every`) shallow-copied router links, leaking stale router copies each diag; fixed in the callback, probe without diag flat at 24.1 GiB |
| 11-10 | row-centred targets let one price trade off two cuts | no: centring erases the gradient scale between cuts; added `center=none` and tau calibration |
