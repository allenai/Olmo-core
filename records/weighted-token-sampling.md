---
title: Weighted token sampling — sampled token survival with weights w = 1/p in attention and DeltaNet, one FLOP price instead of the recipe's hand-set two-stage top-k
date: 2026-11-10
updated: 2026-11-10
experiment: tokdrop_lr
area: token_routing
kind: result
status: in progress (S1/S2 runs training)
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

(scores filled in as evals land; Beaker evals for S1 / S2b queued since 11-10 afternoon)

Training-side, 1,952 rows/task, 488 steps, 2 H200 (mooney), seed 0:

| arm | step cost (x dense, cumulative) | peak alloc GiB/GPU | train CE steps 1-20 / 21-40 / 41-60 / 61-160 |
|---|---|---|---|
| recipe (Beaker H100) | 0.221 (step 1 dense) | 26.9 | - |
| wtsS1 (sampled, no weights) | 0.215 | 24.1 (memory probe, no diag) | 0.760 / 0.376 / 0.340 / 0.226 |
| wtsS2b (weights attn + gdn) | 0.215 | (diag leak inflated; rerun pending) | 0.775 / 0.390 / 0.351 / 0.242 |
| wtsS3 smoke (price rule, 60 steps) | 0.205 (target 0.216) | 24.25 | 0.758 / 0.386 / 0.330 / - |

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
| 11-10 | peak memory creeps 26.9 -> 30.5 GiB in S1/S2b | diag (`--st-otr-diag-every`) shallow-copied router links, leaking stale router copies each diag; fixed in the callback, probe without diag flat at 24.1 GiB |
| 11-10 | row-centred targets let one price trade off two cuts | no: centring erases the gradient scale between cuts; added `center=none` and tau calibration |
