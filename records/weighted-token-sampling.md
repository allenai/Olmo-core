---
title: Weighted token sampling — sampled token survival with weights w = 1/p in attention and DeltaNet, one FLOP price instead of the recipe's hand-set two-stage top-k
date: 2026-11-10
updated: 2026-11-11
experiment: tokdrop_lr
area: token_routing
kind: result
status: negative (11-11): sampled survival with 1/p weights never matched top-k: about -0.02 at the recipe's budget (1,952 rows), -0.007 at 3,904 rows, -0.016 at half the keeps (S4, 2 seeds). Side result: the recipe without its dense step 1 (S0wr) matches the recipe at 24.2 GiB instead of 26.9
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
| dense (LR 5e-6, same 1,952 rows) | 1.0 | - | | | | | 0.843 | | - |
| wtsS0wr (recipe top-k, warmup routed) | 0.219 | 24.2 | 0.956 | 0.920 | 0.838 | 0.633 | 0.837 | 0.907 | 0.158 |
| wtsS1 (sampled, no weights) | 0.215 | 24.1* | 0.942 | 0.930 | 0.670 | 0.620 | 0.790 | 0.895 | 0.326 |
| wtsS2a (weights: attention) | 0.215 | 24.2 | 0.945 | 0.914 | 0.811 | 0.638 | 0.827 | 0.903 | 0.184 |
| wtsS2g (weights: DeltaNet) | 0.215 | 24.2 | 0.941 | 0.902 | 0.775 | 0.641 | 0.815 | 0.924 | 0.220 |
| wtsS2b (weights: both) | 0.215 | 24.1* | 0.937 | 0.920 | 0.794 | 0.637 | 0.822 | 0.918 | 0.202 |
| wtsS3 (price, cuts 7 / 16) | 0.214 | 24.6 | 0.934 | 0.914 | 0.725 | 0.609 | 0.795 | 0.929 | 0.274 |
| wtsS3d (price, alloc=doc) | 0.216 | 24.2 | 0.919 | 0.914 | 0.710 | 0.638 | 0.795 | 0.922 | 0.288 |
| wtsS3m (price, 6 cuts) | 0.213 | 25.8 | 0.953 | 0.910 | 0.665 | 0.629 | 0.789 | 0.910 | 0.332 |
| *round 2: sharper scores* | | | | | | | | | |
| **wtsS2aT5 (attention weights, temp 0.5)** | 0.215 | 24.2 | 0.954 | 0.920 | 0.841 | 0.644 | **0.840** | 0.899 | 0.156 |
| wtsS2aT2 (attention weights, temp 0.2) | 0.215 | 24.2 | 0.945 | 0.908 | 0.835 | 0.626 | 0.829 | 0.907 | 0.158 |
| wtsS1T2 (no weights, temp 0.2) | 0.215 | 24.2 | 0.946 | 0.914 | 0.843 | 0.628 | 0.833 | 0.899 | 0.152 |
| wtsS3aT5 (price, attention weights, temp 0.5) | 0.211 | 24.4 | 0.962 | 0.910 | 0.778 | 0.630 | 0.820 | 0.914 | 0.220 |
| *round 3* | | | | | | | | | |
| wtsS2aT5 seed 1 | 0.215 | 24.2 | 0.944 | 0.888 | 0.852 | 0.662 | 0.837 | 0.920 | 0.144 |
| **wtsS2aH (sampled at block 7, top-k at block 16, attention weights)** | 0.218 | 24.2 | 0.956 | 0.934 | 0.851 | 0.648 | **0.847** | 0.918 | 0.148 |
| wtsS2aT5H (as S2aH, temp 0.5) | 0.218 | 24.2 | 0.948 | 0.910 | 0.757 | 0.626 | 0.810 | 0.920 | 0.238 |
| wtsS3aT2 (price, attention weights, temp 0.2) | 0.203 | 24.2 | 0.963 | 0.910 | 0.815 | 0.639 | 0.832 | 0.924 | 0.176 |
| *round 4: seed 1* | | | | | | | | | |
| wtsS2aH seed 1 | 0.218 | 24.2 | 0.941 | 0.914 | 0.815 | 0.634 | 0.826 | 0.898 | 0.178 |
| wtsS2aT5H seed 1 | 0.218 | 24.2 | 0.949 | 0.918 | 0.908 | 0.680 | 0.864 | 0.917 | 0.088 |
| *round 5* | | | | | | | | | |
| wtsS0wr seed 1 | 0.219 | 24.2 | 0.943 | 0.940 | 0.897 | 0.656 | 0.859 | 0.923 | 0.100 |
| wtsS1H (sampled at block 7, no weights, top-k at 16) | 0.218 | 24.2 | 0.932 | 0.936 | 0.625 | 0.599 | 0.773 | 0.843 | 0.368 |
| wtsS2aT2H (as S2aH, temp 0.2) | 0.218 | 24.2 | 0.945 | 0.902 | 0.782 | 0.664 | 0.823 | 0.922 | 0.214 |
| *round 6 (all `hard=1`)* | | | | | | | | | |
| wtsS2aHc16 (clip log w 1.6) | 0.218 | 24.2 | 0.943 | 0.926 | 0.800 | 0.597 | 0.817 | 0.919 | 0.196 |
| wtsS2aHc07 (clip log w 0.7) | 0.218 | 24.2 | 0.949 | 0.906 | 0.774 | 0.646 | 0.819 | 0.910 | 0.220 |
| wtsS2aHm02 (uniform share 0.02) | 0.218 | 24.2 | 0.950 | 0.924 | 0.734 | 0.617 | 0.806 | 0.913 | 0.262 |
| wtsS2aHm02c16 (both) | 0.218 | 24.2 | 0.948 | 0.934 | 0.825 | 0.657 | 0.841 | 0.913 | 0.172 |
| *S4: half the recipe's keeps (7:0.1, 16:0.035), seed 0 / seed 1* | | | | | | | | | |
| wtsS4k (top-k, no dense step) | 0.189 | 23.9 | 0.942 / 0.936 | 0.930 / 0.902 | 0.777 / 0.784 | 0.626 / 0.668 | **0.819 / 0.823** | 0.877 / 0.895 | 0.218 / 0.210 |
| wtsS4aH (sampled at block 7, top-k at 16) | 0.188 | 23.9 | 0.922 / 0.924 | 0.876 / 0.886 | 0.818 / 0.743 | 0.638 / 0.633 | 0.814 / 0.796 | 0.913 / 0.885 | 0.174 / 0.250 |

3,904 rows/task (976 steps, local `mt4f3904` shard), seed 0, eval_size 500 per cell:

| arm | step cost | peak GiB | contra | nq | outlier | oolong | 8k mean | 2k mean | outlier none right | FTM 8k |
|---|---|---|---|---|---|---|---|---|---|---|
| recipe (seed 0 / seed 1) | 0.220 / 0.220 | 26.9 / 20.2 | 0.977 / 0.973 | 0.918 / 0.926 | 0.970 / 0.967 | 0.663 / 0.689 | **0.882 / 0.889** | 0.949 / 0.945 | - | 3.61x / 3.94x |
| wtsS2aH | 0.218 | 24.2 | 0.967 | 0.918 | 0.945 | 0.679 | 0.877 | 0.939 | 0.054 | 3.55x |
| wtsS2aT5 | 0.215 | 24.2 | 0.971 | 0.908 | 0.966 | 0.659 | 0.876 | 0.937 | 0.028 | 3.54x |
| wtsS0wr (recipe, no dense step) | 0.219 | 24.2 | 0.972 | 0.936 | 0.973 | 0.655 | 0.884 | 0.945 | 0.024 | 3.82x |

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

Round 2 (11-11). Sharper scores close the outlier gap: temp 0.5 lifts S2a's outlier 0.811 -> 0.841 and its 8k mean
0.827 -> 0.840 (recipe 0.846 / 0.847: -0.006, inside the 0.008 SE, one seed); temp 0.2 lifts even the unweighted S1
0.670 -> 0.843, so at temp 1 the sample was simply too flat. Going sharper than 0.5 does not help (S2aT2 0.829, within
noise of S2aT5 but not above it). The none-right share falls to the top-k level (0.15-0.16). The price rule gains too
(S3 0.795 -> S3aT5 0.820, outlier 0.725 -> 0.778) but still trails, consistent with it keeping less after block 16.
Cost 0.211-0.215 and peak 24.2-24.4 GiB, under the recipe's 0.221 / 26.9. Round 3: S2aT5 seed 1 (the bar), S2aT5H
and S2aH (`hard=1`: the recipe's top-k at block 16, sampling only at block 7), S3aT2 (price, temp 0.2).

Round 3 (11-11; every run resumed from its step-400 checkpoint for the last 88 steps because the first launch saved
no final model; the resume restores the router fit, FLOP meter and data position). S2aT5 holds on seed 1 (0.837;
2-seed mean 0.8385 vs recipe 0.8465, -0.008, about one SE). Top-k at the deep cut (`hard=1`) with the default temp
(S2aH) reaches 0.847, the recipe's number, at 0.218x and 24.2 GiB: sampling at block 7 plus weights costs nothing once
block 16 keeps the recipe's top-k, so the residual gap in S2aT5 is the soft deep cut. Combining temp 0.5 with
`hard=1` (S2aT5H) drops outlier to 0.757 (none right 0.238); one seed, so a seed-1 rerun is running before reading
anything into it. The price rule keeps improving with sharper scores (S3aT2 0.832 at 0.203x, the cheapest arm). The
3,904-row shard was built on mooney the same way as f1952 (`merge_tasks.py --rows-per-src 3904 --allow-no-gold` over
the node-local `*_fresh` pools; the f1952 rebuild is byte-identical to the shard every 1,952-row arm used).

Round 4 (11-11). Seed 1 moves outlier by ±0.05 in both `hard=1` arms in opposite directions: S2aH 0.847 -> 0.826,
S2aT5H 0.810 -> 0.864. Their 2-seed means are 0.8365 and 0.837, and S2aT5's is 0.8385, all about 0.009 under the
recipe (0.8465), whose two seeds agree to 0.001 (outlier 0.860 / 0.861). So the sampled arms are noisier across seeds
and sit about one SE of a 2-seed mean (~0.006) plus seed spread under the bar. At 3,904 rows the gap is the same
size: S2aH 0.877, S2aT5 0.876 vs recipe 0.882 / 0.889, at 0.215-0.218x and 24.2 GiB. The routed-warmup recipe S0wr
(top-k at both cuts, no dense step 1, as every wts arm) scored the same 0.837 in round 1, so the gap may come from
dropping the dense step and not from sampling. Round 5 runs S0wr seed 1 and S0wr at 3,904 rows to test that, plus S1H
(no weights, `hard=1`) and S2aT2H (temp 0.2, `hard=1`).

Round 5 (11-11). The dense-step hypothesis is refuted: S0wr seed 1 scores 0.859 (2-seed mean 0.848 vs recipe
0.8465) and 0.884 at 3,904 rows (recipe 0.882 / 0.889), at 0.219x and 24.2 GiB. So the recipe needs no dense step,
and the best wts arms' gap (S2aH 2-seed 0.8365, 0.877 at 3,904) is the sampling itself: -0.012 and -0.007 against
S0wr. S0wr also swings on outlier across seeds (0.838 / 0.897), so the recipe's 0.001 seed agreement is partly luck.
Weights are required: without them (S1H) outlier falls to 0.625. Sharper scores at block 7 do not help (S2aT2H
0.823). Logged weights in S2aH: mean log w ~1.1 (w ~3), per-step max ~2.9 (w ~18). Round 6 trims the tail (clip
log w at 1.6 or 0.7) and the uniform share (mix 0.02), all with `hard=1`.

Round 6 (11-11). Trimming the weight tail or the uniform share does not close the gap: clip 1.6 0.817, clip 0.7
0.819, mix 0.02 0.806, both 0.841 (one seed each). The screens are now noise-limited. Across seeds the 8k mean moves
by ~0.02 (S2aH 0.847 / 0.826, S2aT5H 0.810 / 0.864, S0wr 0.837 / 0.859), so single-seed differences under ~0.03 mean
little. Pooled, the 9 single-seed runs that sample at block 7 with top-k at block 16 average 0.828 (sd 0.019); the 4
top-k runs (recipe x2, S0wr x2) average 0.847 (sd 0.009). At the recipe's operating point sampling at block 7 costs
about 0.02 at 1,952 rows and 0.007 at 3,904, and no variant tried removes it. S4 asks whether sampling wins once the
budget binds: half the recipe's keeps (7:0.1, 16:0.035, ~0.18x), top-k (S4k) vs S2aH-style (S4aH), 2 seeds each.

S4 (11-11). At half the recipe's keeps (0.189x per step, 23.9 GiB), top-k scores 0.819 / 0.823 (mean 0.821) and
sampling 0.814 / 0.796 (0.805): sampling loses here too (-0.016), so a tighter budget does not turn it into a win.
Halving the keeps costs top-k 0.026 for 14% fewer step FLOPs; both arms are below the lowest dense point, so their
FTM is only an upper bound (<= 5.6x).

**Verdict.** Sampled survival with w = 1/p weights does not match the recipe's top-k at any budget tried: about -0.02
at 1,952 rows (pooled over 9 single-seed variants), -0.007 at 3,904 rows, -0.016 at half the keeps (2 seeds each).
Weights are necessary for sampling to work at all (without them outlier falls to 0.63-0.67), and sharper scores help
at temp 1 -> 0.5. But no variant beats keeping the highest-scored tokens. The price rule (S3) trails further (best
0.832 at 0.203x, one seed). S5 (row-level sampling) was not run. Side result: the recipe with its dense step 1
replaced by a routed uniform warmup (S0wr) matches the recipe (2-seed 0.848 vs 0.8465; 0.884 at 3,904 vs 0.882 /
0.889) at 24.2 GiB peak instead of 26.9, with no dense step.

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
| 11-11 | soft selection costs outlier; sharper scores (temp 0.5, 0.2) close the gap | yes: S2aT5 0.840 (outlier 0.841), S1T2 0.833 (0.843); temp 1 was too flat. Price rule gains (0.795 -> 0.820) but trails |
| 11-11 | the remaining gap is the soft deep cut; top-k there (`hard=1`) matches the recipe | yes on seed 0: S2aH 0.847 (recipe 0.846 / 0.847); seed 1 and 3,904 rows running. With temp 0.5 (S2aT5H) 0.810: seed 1 running |
| 11-11 | S2aH matches the recipe over 2 seeds and at 3,904 rows | no: 2-seed 0.8365 (seed 1 0.826); 3,904 rows 0.877 vs 0.882 / 0.889. S2aT5H seed 1 0.864, so its 0.810 was seed noise (2-seed 0.837). Sampled arms swing ±0.05 on outlier across seeds; the recipe does not |
| 11-11 | the ~0.009 gap is the missing dense step 1, not the sampling (S0wr = 0.837 too) | no: S0wr seed 1 0.859 (2-seed 0.848), 0.884 at 3,904 rows; the routed warmup matches the recipe |
| 11-11 | weights matter once the deep cut is top-k | yes: S1H (no weights) 0.773, outlier 0.625 vs S2aH 0.8365 |
| 11-11 | the gap is the high-weight tail / uniform share (low-p random survivors with large w) | no: clip 1.6 0.817, clip 0.7 0.819, mix 0.02 0.806, both 0.841 (one seed each, inside seed noise) |
| 11-11 | sampling beats top-k once the budget binds (half the recipe's keeps) | no: top-k 0.819 / 0.823 vs sampled 0.814 / 0.796 at 0.19x |
| 11-10 | peak memory creeps 26.9 -> 30.5 GiB in S1/S2b | diag (`--st-otr-diag-every`) shallow-copied router links, leaking stale router copies each diag; fixed in the callback, probe without diag flat at 24.1 GiB |
| 11-10 | row-centred targets let one price trade off two cuts | no: centring erases the gradient scale between cuts; added `center=none` and tau calibration |
