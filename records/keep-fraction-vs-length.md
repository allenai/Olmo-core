---
title: Does the recipe's best keep fraction fall as context grows? (keep 70%..1.25% at 2k-64k, dense held-out-loss proxy)
date: 2026-11-17
updated: 2026-11-17
experiment: tokdrop_lr
area: token_routing
kind: result
status: current (11-17; the 64k 70% arm is still running)
superseded_by: 
artifact: 
summary: No. On single-length pools at 2k-64k scored by the early dense held-out-loss proxy (one epoch, eval_size 144 per length), the best router keep is 20-40% at every length the proxy resolves and does not fall with length (8k 40% tied with 20%; 16k 20-40% tie; 32k 40% beats the 20%/7% recipe at matched FLOPs by 0.01-0.02 CE on both seeds, 70% ties; 64k 20%, 40% +0.015 on both seeds). Low keeps break harder the longer the rows (10%/3.5%: +0.003 CE vs the recipe at 8k, +0.08 at 64k; 2.5% and 1.25% collapse at 32k/64k), so constant fraction fits and a constant kept-token count is rejected; 2k/4k saturate. Gold-aware keep grJ has the biggest mean speedups (6-8x vs dense full FT) but stalls on outlier; grJc is outlier-safe at 32k, not at 64k.
---
# Does the recipe's best keep fraction fall as context grows? (keepfrac_len)

Prasann (2026-11-17): does the recipe's best keep fraction get lower as context grows? Lengths 2k-64k, each with a
single-length training pool and a same-length held-out set, scored on the cheap dense held-out-loss proxy. Code:
`research/token_routing/early_loss_proxy/keepfrac_len/` (branch `claude/keepfrac-vs-length-4bf10c`, built on
`claude/early-loss-proxy-d6e886`). Brief: `records/tokdrop-handoff-2026-11-17.md` section 2a.

## Verdict

**No. The best keep stays at 20-40% from 8k to 64k (40% at 32k, 20% at 64k) and does not fall with length; low
keeps break harder the longer the rows.** Constant fraction fits; constant kept-token count is rejected.

- **Best keep per length** (headline = held-out CE difference vs the 20%/7% recipe at matched training FLOPs, plus
  FLOP speedup vs dense full fine-tune `dft` at dft's mid-epoch CE):
  - 2k, 4k: **unresolvable**. Every arm drives held-out CE to 0.006-0.036 by the end of the epoch; differences at
    matched FLOPs are ≤ 0.02 and every keep from 70% to 2.5% reaches dft's mid-epoch CE at 2.6-4.1x. Saturated.
  - 8k: **40%/14%**, tied with 20%/7%: ΔCE vs recipe -0.004 / -0.004 (2 seeds), 3.34x vs 3.30x dft (seed means);
    70% is worse (+0.007 / +0.018); 10% and 5% slightly worse (+0.003, +0.012); 2.5% fails (+0.069).
  - 16k: **20-40% tie**: 40% ΔCE -0.001 / +0.014, 10% +0.002 / +0.014, 70% +0.007; speedups 3.0-3.2x (seed means)
    for 10-40%. 5% +0.010, 2.5% +0.036 (never reaches dft's mid-epoch CE).
  - 32k: **40%/14%** beats the recipe at matched FLOPs on both seeds (-0.019 / -0.010; paired SE of a final-CE
    difference 0.005); 70% ties it (-0.022 / -0.002); 10% +0.011 / +0.031, 5% +0.040, 2.5% +0.296 (collapse). Only
    20-70% reach dft's mid-epoch CE (seed means 40% 3.24x, 70% 3.06x, 20% 2.86x); 40% and 70% reach dft's
    end-of-epoch CE on seed 0 (3.95x / 4.09x).
  - 64k: **20%/7%**: 40% is +0.015 / +0.017 at matched FLOPs (both seeds; at equal steps +0.008 / -0.007, a tie),
    10% +0.080 / +0.033, 5% +0.103, 2.5% +0.278, 1.25% +0.602 (collapse). Only 20% and 40% reach dft's mid-epoch CE
    (seed means 3.17x and 2.58x); no router arm reaches dft's end-of-epoch CE (0.211; recipe 0.238 / 0.265). A 70%
    arm is still running.
- **Margin vs seed noise.** Seed-to-seed |Δ mean CE| at the end of the epoch is 0.001-0.018 per arm at 8k-32k
  (k20 at 32k 0.0006, k40 0.005, k10 0.015, k70 0.018) and up to 0.027 at 64k (k20); the speedup at the mid level
  moves by up to 1.0x between seeds (32k k70: 3.57 vs 2.54; 64k k20: 3.61 vs 2.72). So 40% vs 20% at 8k/16k and
  40% vs 70% at 32k are inside the noise; 40% > 20% at 32k (0.010-0.019) and 20% > 40% at 64k (0.015-0.017) have
  the same sign on both seeds but are no larger than the seed spread; the low-keep penalties at 32k/64k (0.03-0.6
  CE, both seeds where run) are far outside it.
- **Which rule fits.** A constant kept-token count anchored at 8k (40% x 8k ≈ 3.3k tokens) predicts best keeps of
  20% / 10% / 5% at 16k / 32k / 64k. Measured: 20-40% / 40-70% / 20% (5% at 64k is +0.10 CE worse than the recipe).
  Constant fraction in the 20-40% band fits every resolvable length. The penalty of keeping 10% grows with length
  (ΔCE vs recipe +0.003 at 8k, +0.002/+0.014 at 16k, +0.011/+0.031 at 32k, +0.080/+0.033 at 64k), so if anything longer rows
  need *more* kept tokens per row, not a fixed count.
- **Why low keeps buy little.** Below 20%/7% the per-step FLOP saving flattens (FLOPs/step vs dense: 64k 0.185 at
  20% -> 0.160 at 10% -> 0.140 at 1.25%; 32k 0.201 -> 0.174 -> 0.154) because blocks 0-6 run dense on every token
  and dominate; the loss penalty grows instead.
- **Per task / worst task.** Contradiction is where low keeps break first at every length (e.g. 32k final CE
  contradiction: dft 0.063, 40% 0.072, 20% 0.122, 10% 0.155, 5% 0.224); outlier is the worst task for most arms at
  64k (no router arm reaches dft's outlier mid-epoch CE there). 70% at 32k seed 0 is the only router run below dft on
  outlier (0.169 vs 0.187; seed 1 0.195).
- **Gold-aware stochastic keep (`grJ`) and the outlier-safe variant (`grJc`)**, 32k and 64k:
  - `grJ` has the largest mean speedups of the study (6.2x at 32k, 8.4x at 64k vs dft at the mid level; 0.080 /
    0.067 of dense FLOPs per step) but **outlier never reaches dft's mid-epoch CE** (final outlier CE 0.249 vs
    0.187 at 32k, 0.284 vs 0.220 at 64k) and at 64k contradiction does not either. The mean hides an outlier stall,
    as in the 4k-16k mixed-pool screen.
  - `grJc` (document drop on contradiction rows only) at 32k: 4.16x / 4.15x on the mean (2 seeds); worst task
    2.82x (contradiction) on seed 0, but on seed 1 contradiction never reaches dft's mid-epoch CE. 40%/14% is 3.24x
    (worst 2.72x / 2.41x). At equal steps `grJc` ends 0.015 / 0.019 CE above 40%/14% (contradiction +0.036 /
    +0.038), so its mean advantage is per-step cheapness, and contradiction is its weak task. At 64k `grJc` is 4.10x
    on the mean but contradiction (0.191 vs dft 0.086) and outlier do not reach dft's mid-epoch level: **outlier-
    safe at 32k, not at 64k** (one seed).

The proxy understates gains relative to generation evals (recipe 1.9-2.2x on the proxy vs 3.3x on generation evals,
`records/early-dense-loss-proxy.md`) and cannot resolve final gaps below about 0.03-0.05 CE: use these numbers to
rank keeps, not to quote speedups.

## Setup

- Base `q35-4b-base-markerfix` (Qwen3.5-4B, marker-repaired). Training pools `kl{2k,4k,8k,16k,32k,64k}`
  (`build_kl_pools.py`): three tasks (contradiction, outlier, qdmatch_nq), 1,200 rows each, one length per pool.
  Row lengths (mean / max): 2k 2,215 / 3,367; 4k 4,092 / 5,354; 8k 8,246 / 9,819; 16k 14,932 / 16,308; 32k
  29,210 / 31,483; 64k 58,549 / 62,307 (the short pools are rung-labelled and overshoot their label; trainer
  --seq-len 4,096 / 6,144 / 10,240 there).
- Held-out: `heldout_kls.npz` (sets kl2k/4k/8k) and `heldout_kl.npz` (kl16k/32k/64k), 48 rows per task (eval_size
  144 per length, ⚠ below 500: paired SE of a CE difference between two arms ≈ 0.003-0.006 at the end of the epoch),
  gold not in training (builder checks).
- One epoch = 3,600 rows / 16 rows per step = 225 steps, the trainer's real LR schedule over that epoch; dense
  probes at steps 0, 10, 25, 50, ..., 225 (teacher-forced answer CE of the DENSE forward, no generation).
- Arms: `dft` (dense full fine-tune, LR 5e-6, the reference), `ref` (dense LoRA r64, LR 1e-5), router keeps
  `k70/k40/k20/k10/k5/k2p5/k1p25` = the recipe's online router with cut 1 (after block 7) keeping 70/40/20/10/5/2.5/
  1.25% and cut 2 (after block 16) keeping 0.35x that (24.5/14/7/3.5/1.75/0.9/0.45%), `grJ` (recipe + gold-aware
  stochastic whole-document keep, `--st-gdd-keep-range 0.05,0.5 --st-gdd-whole-frac 0.15`), `grJc` (`grJ` with the
  document drop on contradiction rows only). LR 1e-5 for all LoRA arms.
- Compute: Beaker 2 x H100 (jupiter/ceres), micro-batch 2 (1 at 64k and for `dft` at 32k; 2 GPUs at micro 2 ran
  out of memory there); `dft` and `ref` at 64k on 8 x H100 (2 x H200 reserves 78 GiB and takes 76 s/step for `dft`).
  Quick reads `q32k` / `q64k` on mooney 2 x H200. Beaker vs mooney agree: 64k `dft` step-10 CE 0.528 (8 x H100)
  vs 0.527 (2 x H200), `k10` 0.569 vs 0.572.
- Metrics (`analyze_kl.py`, `summarize_kl.py`): PF = training FLOPs actually spent (the trainer's meter).
  (1) **ΔCE vs recipe at matched FLOPs**: mean held-out CE of the arm minus the recipe's, both read (running-min
  curves, linear interpolation) at the smaller of the two runs' total training FLOPs; negative = better than the
  recipe. (2) **x dft @ mid CE**: PF_dft / PF_arm to reach dft's own mean CE at step 100 (the "mid-epoch" level),
  "not reached" if the arm never gets there. (3) the same at dft's end-of-epoch CE. Worst task = min over the three
  tasks of the per-task speedup at dft's per-task mid-epoch CE.

## Quick reads (40 steps, one seed)

- `q32k` (pool `mtb64k_r32k`, 960 rows = 60-step epoch, held-out `heldout_v4` gen32k, eval_size 72): 40%/14% and
  20%/7% tie (CE 0.370 vs 0.380 ±0.015); 10% and below plateau higher. Full table `keepfrac_len/INTERIM_q32k.md`.
- `q64k` (pool kl64k, held-out kl64k, eval_size 144; 225-step schedule, stopped at 40): held-out CE at step 40: dft
  0.404, dense LoRA 0.370, 40% 0.452, 20% 0.471, 10% 0.496, 5% 0.499, 2.5% 1.052, 1.25% 1.051 (both diverged
  after step 10). Same ordering as the full 64k epoch. Slurm jobs 5201918-5201921 (mooney, completed); scored
  `keepfrac_len/q64k_scored.json`.

## Memory and time (for reference only; micro-batch and GPU count differ between dense arms)

Peak GiB (max over ranks) for router arms is flat in keep at each length (2k 24-25, 8k 27-28, 16k 32, 32k 46.5,
64k 46 on 2 x H100), i.e. it is set by the dense early blocks and full-length activations, not by the kept share.
Train s/step on 2 x H100: 16k recipe 6.1, 40% 7.4, 70% 8.4, dft 13.9; 32k recipe 9.8, 40% 12.3, 70% 15.3.

## Per-length tables

Seed values are separated by " / ". ⚠ eval_size 144 per length (48 rows per task), below 500: the paired SE of a CE
difference between two arms on the same rows is 0.003-0.006 at the end of the epoch; seed-to-seed spread is up to
0.015 CE and 0.6x in the mid-level speedup. "not reached" = the arm never gets down to that dft CE level.

<!-- tables:begin (finalize.sh) -->

**2k** (eval_size 144 ⚠ < 500, 48 rows/task; matched level = dft's mean held-out CE at step 100 = 0.0233; end of epoch 0.0103)

| arm | seeds | FLOPs/step vs dense | ΔCE vs recipe @ matched FLOPs | x dft @ mid CE | x dft @ end CE | worst task x @ mid (task) | contradiction x @ mid | outlier x @ mid | qdmatch_nq x @ mid | final CE | final CE contra / outlie / qdmatc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense full FT | 0 | 1.000 | +0.0204 | 1.00 | 1.00 | 1.00 (contradiction) | 1.00 | 1.00 | 1.00 | 0.0103 | 0.017 / 0.003 / 0.010 |
| dense LoRA | 0 | 0.746 | +0.0068 | 1.85 | 1.75 | 1.71 (outlier) | 1.90 | 1.71 | 1.79 | 0.0070 | 0.015 / 0.002 / 0.004 |
| 70% / 24.5% | 0 | 0.407 | +0.0001 | 2.64 | 3.15 | 1.91 (qdmatch_nq) | 2.65 | 2.90 | 1.91 | 0.0057 | 0.007 / 0.003 / 0.007 |
| 40% / 14% | 0 | 0.319 | +0.0008 | 4.05 | not reached | 3.18 (qdmatch_nq) | 4.05 | 4.81 | 3.18 | 0.0130 | 0.021 / 0.008 / 0.010 |
| 20% / 7% (recipe) | 0 | 0.262 | +0.0000 | 2.78 | not reached | 2.57 (qdmatch_nq) | 2.95 | 3.95 | 2.57 | 0.0138 | 0.012 / 0.020 / 0.009 |
| 10% / 3.5% | 0 | 0.233 | -0.0066 | 3.22 | not reached | 2.82 (qdmatch_nq) | 3.62 | 4.29 | 2.82 | 0.0110 | 0.015 / 0.012 / 0.006 |
| 5% / 1.75% | 0 | 0.219 | -0.0033 | 3.06 | not reached | not reached (qdmatch_nq) | 2.47 | 5.06 | not reached | 0.0144 | 0.027 / 0.005 / 0.011 |
| 2.5% / 0.9% | 0 | 0.212 | -0.0030 | 2.88 | not reached | not reached (qdmatch_nq) | 2.86 | 3.98 | not reached | 0.0147 | 0.019 / 0.016 / 0.009 |

**4k** (eval_size 144 ⚠ < 500, 48 rows/task; matched level = dft's mean held-out CE at step 100 = 0.0429; end of epoch 0.0129)

| arm | seeds | FLOPs/step vs dense | ΔCE vs recipe @ matched FLOPs | x dft @ mid CE | x dft @ end CE | worst task x @ mid (task) | contradiction x @ mid | outlier x @ mid | qdmatch_nq x @ mid | final CE | final CE contra / outlie / qdmatc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense full FT | 0 | 1.000 | +0.0403 | 1.00 | 1.00 | 1.00 (contradiction) | 1.00 | 1.00 | 1.00 | 0.0129 | 0.012 / 0.012 / 0.014 |
| dense LoRA | 0 | 0.745 | +0.0318 | 1.49 | not reached | 1.13 (qdmatch_nq) | 1.71 | 1.43 | 1.13 | 0.0169 | 0.016 / 0.016 / 0.018 |
| 70% / 24.5% | 0 | 0.390 | +0.0190 | 2.63 | not reached | not reached (qdmatch_nq) | 2.95 | 1.44 | not reached | 0.0270 | 0.027 / 0.024 / 0.031 |
| 40% / 14% | 0 | 0.299 | +0.0059 | 3.25 | not reached | 2.16 (qdmatch_nq) | 2.40 | 2.31 | 2.16 | 0.0214 | 0.029 / 0.014 / 0.021 |
| 20% / 7% (recipe) | 0 | 0.240 | +0.0000 | 3.63 | not reached | not reached (qdmatch_nq) | 3.02 | 2.22 | not reached | 0.0187 | 0.018 / 0.014 / 0.025 |
| 10% / 3.5% | 0 | 0.210 | +0.0048 | 3.82 | not reached | not reached (qdmatch_nq) | 3.30 | 2.02 | not reached | 0.0258 | 0.026 / 0.027 / 0.024 |
| 5% / 1.75% | 0 | 0.195 | +0.0033 | 3.09 | not reached | not reached (qdmatch_nq) | 2.70 | 2.08 | not reached | 0.0266 | 0.033 / 0.022 / 0.024 |
| 2.5% / 0.9% | 0 | 0.188 | +0.0119 | 2.99 | not reached | not reached (qdmatch_nq) | 2.90 | 2.15 | not reached | 0.0364 | 0.033 / 0.047 / 0.030 |

**8k** (eval_size 144 ⚠ < 500, 48 rows/task; matched level = dft's mean held-out CE at step 100 = 0.0808; end of epoch 0.0445)

| arm | seeds | FLOPs/step vs dense | ΔCE vs recipe @ matched FLOPs | x dft @ mid CE | x dft @ end CE | worst task x @ mid (task) | contradiction x @ mid | outlier x @ mid | qdmatch_nq x @ mid | final CE | final CE contra / outlie / qdmatc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense full FT | 0 | 1.000 | +0.0749 | 1.00 | 1.00 | 1.00 (contradiction) | 1.00 | 1.00 | 1.00 | 0.0445 | 0.022 / 0.068 / 0.044 |
| dense LoRA | 0 | 0.743 | +0.0571 | 1.15 | 1.62 | 0.84 (contradiction) | 0.84 | 1.63 | 1.06 | 0.0365 | 0.021 / 0.053 / 0.036 |
| 70% / 24.5% | 0,1 | 0.377 | +0.0065 / +0.0178 | 2.76 (2.53-2.99) | 3.35 (2.98-3.72) | 2.69 (contradiction) / 2.16 (outlier) | 2.69 / 2.83 | 3.53 / 2.16 | 3.39 / 2.98 | 0.0306 / 0.0396 | 0.017 / 0.043 / 0.031 ; 0.020 / 0.069 / 0.030 |
| 40% / 14% | 0,1 | 0.284 | -0.0035 / -0.0043 | 3.34 (3.11-3.57) | 4.21 (4.01-4.41) | 2.50 (contradiction) / 2.92 (contradiction) | 2.50 / 2.92 | 4.86 / 7.02 | 3.66 / 3.67 | 0.0324 / 0.0390 | 0.022 / 0.042 / 0.033 ; 0.023 / 0.061 / 0.033 |
| 20% / 7% (recipe) | 0,1 | 0.224 | +0.0000 / +0.0000 | 3.30 (3.23-3.37) | not reached / not reached | 2.47 (contradiction) / 2.78 (qdmatch_nq) | 2.47 / 2.96 | 5.98 / 4.05 | 3.21 / 2.78 | 0.0476 / 0.0519 | 0.033 / 0.067 / 0.043 ; 0.023 / 0.085 / 0.048 |
| 10% / 3.5% | 0 | 0.195 | +0.0030 | 3.21 | not reached | not reached (contradiction) | not reached | 5.75 | 2.83 | 0.0584 | 0.052 / 0.077 / 0.046 |
| 5% / 1.75% | 0 | 0.180 | +0.0119 | 3.30 | not reached | not reached (contradiction) | not reached | 4.35 | not reached | 0.0712 | 0.057 / 0.092 / 0.065 |
| 2.5% / 0.9% | 0 | 0.173 | +0.0690 | not reached | not reached | not reached (contradiction) | not reached | 2.87 | not reached | 0.1295 | 0.130 / 0.127 / 0.132 |

**16k** (eval_size 144 ⚠ < 500, 48 rows/task; matched level = dft's mean held-out CE at step 100 = 0.1455; end of epoch 0.1230)

| arm | seeds | FLOPs/step vs dense | ΔCE vs recipe @ matched FLOPs | x dft @ mid CE | x dft @ end CE | worst task x @ mid (task) | contradiction x @ mid | outlier x @ mid | qdmatch_nq x @ mid | final CE | final CE contra / outlie / qdmatc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense full FT | 0 | 1.000 | +0.0971 | 1.00 | 1.00 | 1.00 (contradiction) | 1.00 | 1.00 | 1.00 | 0.1230 | 0.057 / 0.213 / 0.099 |
| dense LoRA | 0 | 0.739 | +0.0325 | 1.30 | 1.79 | 0.94 (outlier) | 1.37 | 0.94 | 1.73 | 0.1138 | 0.039 / 0.215 / 0.088 |
| 70% / 24.5% | 0 | 0.367 | +0.0067 | 2.94 | 3.56 | 2.21 (qdmatch_nq) | 3.10 | 2.86 | 2.21 | 0.1107 | 0.047 / 0.207 / 0.079 |
| 40% / 14% | 0,1 | 0.273 | -0.0014 / +0.0138 | 3.01 (2.76-3.27) | 4.27 (4.13-4.41) | 2.80 (contradiction) / 2.59 (outlier) | 2.80 / 2.96 | 4.02 / 2.59 | 3.57 / 2.88 | 0.1143 / 0.1188 | 0.050 / 0.204 / 0.089 ; 0.051 / 0.206 / 0.100 |
| 20% / 7% (recipe) | 0,1 | 0.214 | +0.0000 / +0.0000 | 3.19 (2.89-3.49) | not reached / 5.14 | 2.72 (contradiction) / 3.15 (contradiction) | 2.72 / 3.15 | 4.23 / 8.61 | 3.03 / 3.34 | 0.1265 / 0.1203 | 0.066 / 0.207 / 0.105 ; 0.061 / 0.211 / 0.089 |
| 10% / 3.5% | 0,1 | 0.185 | +0.0015 / +0.0140 | 2.97 (2.80-3.13) | not reached / not reached | not reached (contradiction) / not reached (contradiction) | not reached / not reached | 6.07 / 3.29 | 3.06 / 2.64 | 0.1335 / 0.1386 | 0.088 / 0.210 / 0.102 ; 0.095 / 0.213 / 0.108 |
| 5% / 1.75% | 0 | 0.170 | +0.0099 | 2.59 | not reached | not reached (contradiction) | not reached | 3.54 | 2.68 | 0.1450 | 0.109 / 0.216 / 0.110 |
| 2.5% / 0.9% | 0 | 0.163 | +0.0357 | not reached | not reached | not reached (contradiction) | not reached | not reached | 2.11 | 0.1738 | 0.173 / 0.223 / 0.125 |

**32k** (eval_size 144 ⚠ < 500, 48 rows/task; matched level = dft's mean held-out CE at step 100 = 0.1922; end of epoch 0.1440)

| arm | seeds | FLOPs/step vs dense | ΔCE vs recipe @ matched FLOPs | x dft @ mid CE | x dft @ end CE | worst task x @ mid (task) | contradiction x @ mid | outlier x @ mid | qdmatch_nq x @ mid | final CE | final CE contra / outlie / qdmatc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense full FT | 0 | 1.000 | +0.0917 | 1.00 | 1.00 | 1.00 (contradiction) | 1.00 | 1.00 | 1.00 | 0.1440 | 0.063 / 0.187 / 0.182 |
| dense LoRA | 0 | 0.733 | +0.0377 | 1.84 | 1.65 | 1.08 (outlier) | 1.47 | 1.08 | 2.32 | 0.1329 | 0.061 / 0.184 / 0.154 |
| 70% / 24.5% | 0,1 | 0.354 | -0.0220 / -0.0022 | 3.06 (2.54-3.57) | 3.51 (2.93-4.09) | 2.52 (contradiction) / 2.11 (contradiction) | 2.52 / 2.11 | 2.61 / 2.99 | 4.39 / 3.45 | 0.1236 / 0.1413 | 0.054 / 0.169 / 0.148 ; 0.071 / 0.195 / 0.158 |
| 40% / 14% | 0,1 | 0.259 | -0.0190 / -0.0104 | 3.24 (2.98-3.50) | 3.95 / not reached | 2.72 (contradiction) / 2.41 (contradiction) | 2.72 / 2.41 | 2.89 / 2.81 | 4.22 / 3.95 | 0.1438 / 0.1487 | 0.072 / 0.184 / 0.175 ; 0.079 / 0.181 / 0.186 |
| 20% / 7% (recipe) | 0,1 | 0.201 | +0.0000 / +0.0000 | 2.85 (2.70-3.01) | not reached / not reached | not reached (contradiction) / not reached (contradiction) | not reached / not reached | 2.77 / 4.22 | 3.95 / 3.80 | 0.1722 / 0.1728 | 0.122 / 0.192 / 0.202 ; 0.119 / 0.189 / 0.211 |
| 10% / 3.5% | 0,1 | 0.174 | +0.0114 / +0.0306 | not reached / not reached | not reached / not reached | not reached (contradiction) / not reached (contradiction) | not reached / not reached | not reached / not reached | 4.16 / 3.86 | 0.1938 / 0.2090 | 0.155 / 0.211 / 0.215 ; 0.185 / 0.227 / 0.216 |
| 5% / 1.75% | 0 | 0.160 | +0.0400 | not reached | not reached | not reached (contradiction) | not reached | not reached | 3.18 | 0.2336 | 0.224 / 0.222 / 0.255 |
| 2.5% / 0.9% | 0 | 0.154 | +0.2961 | not reached | not reached | not reached (contradiction) | not reached | not reached | not reached | 0.4955 | 0.608 / 0.404 / 0.474 |
| grJ (gold-aware stochastic keep) | 0 | 0.080 | -0.1110 | 6.17 | not reached | not reached (outlier) | 5.80 | not reached | 12.87 | 0.1860 | 0.106 / 0.249 / 0.204 |
| grJc (grJ, doc drop on contradiction only) | 0,1 | 0.158 | -0.0362 / -0.0175 | 4.16 (4.15-4.16) | not reached / not reached | 2.82 (contradiction) / not reached (contradiction) | 2.82 / not reached | 4.46 / 3.64 | 6.47 / 5.55 | 0.1590 / 0.1673 | 0.108 / 0.197 / 0.172 ; 0.117 / 0.199 / 0.185 |

**64k** (eval_size 144 ⚠ < 500, 48 rows/task; matched level = dft's mean held-out CE at step 100 = 0.2785; end of epoch 0.2110)

| arm | seeds | FLOPs/step vs dense | ΔCE vs recipe @ matched FLOPs | x dft @ mid CE | x dft @ end CE | worst task x @ mid (task) | contradiction x @ mid | outlier x @ mid | qdmatch_nq x @ mid | final CE | final CE contra / outlie / qdmatc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense full FT | 0 | 1.000 | +0.1634 | 1.00 | 1.00 | 1.00 (contradiction) | 1.00 | 1.00 | 1.00 | 0.2110 | 0.086 / 0.220 / 0.327 |
| dense LoRA | 0 | 0.723 | +0.0961 | 1.34 | 1.84 | 1.34 (qdmatch_nq) | 1.71 | 1.83 | 1.34 | 0.1994 | 0.089 / 0.212 / 0.297 |
| 40% / 14% | 0,1 | 0.240 | +0.0145 / +0.0170 | 2.58 (2.30-2.87) | not reached / not reached | not reached (outlier) / not reached (outlier) | 2.82 / 2.94 | not reached / not reached | 2.98 / 1.98 | 0.2464 / 0.2578 | 0.117 / 0.240 / 0.382 ; 0.124 / 0.232 / 0.418 |
| 20% / 7% (recipe) | 0,1 | 0.185 | +0.0000 / +0.0000 | 3.17 (2.72-3.61) | not reached / not reached | not reached (outlier) / not reached (outlier) | 3.57 / 2.67 | not reached / not reached | 4.02 / 2.85 | 0.2382 / 0.2650 | 0.129 / 0.243 / 0.343 ; 0.166 / 0.234 / 0.395 |
| 10% / 3.5% | 0,1 | 0.160 | +0.0797 / +0.0329 | not reached / not reached | not reached / not reached | not reached (contradiction) / not reached (contradiction) | not reached / not reached | not reached / not reached | not reached / not reached | 0.3311 / 0.3128 | 0.285 / 0.253 / 0.456 ; 0.237 / 0.255 / 0.446 |
| 5% / 1.75% | 0 | 0.148 | +0.1030 | not reached | not reached | not reached (contradiction) | not reached | not reached | not reached | 0.3548 | 0.304 / 0.278 / 0.482 |
| 2.5% / 0.9% | 0 | 0.142 | +0.2778 | not reached | not reached | not reached (contradiction) | not reached | not reached | not reached | 0.5318 | 0.638 / 0.425 / 0.533 |
| 1.25% / 0.45% | 0 | 0.140 | +0.6016 | not reached | not reached | not reached (contradiction) | not reached | not reached | not reached | 1.0407 | 1.201 / 0.839 / 1.083 |
| grJ (gold-aware stochastic keep) | 0 | 0.067 | -0.1207 | 8.41 | not reached | not reached (contradiction) | not reached | not reached | 21.13 | 0.2652 | 0.185 / 0.284 / 0.327 |
| grJc (grJ, doc drop on contradiction only) | 0 | 0.144 | +0.0018 | 4.10 | not reached | not reached (contradiction) | not reached | not reached | 6.84 | 0.2544 | 0.191 / 0.235 / 0.337 |

<!-- tables:end -->

## Still running (2026-11-17 19:30 PT)

- 70%/24.5% at 64k, seed 0 (Beaker `elp-kl64k-bkBX8XWK0M`, 01M7P6T8EEKK6T00K0BX8XWK0M; expected done ~21:30 PT).
- Re-score: `bash research/token_routing/early_loss_proxy/keepfrac_len/finalize.sh`, then update the verdict lines.

## Limits

- Proxy, not generation evals: teacher-forced answer CE of the dense forward on 48 held-out rows per task; it ranks
  arms (Spearman 0.83-0.95 vs final scores, `records/early-dense-loss-proxy.md`), understates the recipe's gain vs
  generation evals and cannot resolve final gaps below ~0.03-0.05 CE. Every 8k/16k keep difference between 20% and
  40% is below that.
- One epoch of 3,600 rows; budget-dependence (does 40% vs 20% flip with more data?) is untested here.
- Two seeds only on the deciding arms; `dft` (the reference) has one seed per length.
- The keep schedule is two cuts with cut 2 = 0.35 x cut 1; one-cut-at-a-time arms (`c1k10`, `c2k3p5`, ...) exist in
  `launch_kl.py` but were not run.
- Memory: router-arm peaks are flat in keep at every length here; this study says nothing about flat-memory
  training at 256k+.

## Files

- Code: `research/token_routing/early_loss_proxy/keepfrac_len/` — `launch_kl.py` (arms; `--local` for jsteinhardt),
  `analyze_kl.py` (matched-loss scoring), `summarize_kl.py` (headline metrics + chart), `paired_se.py`,
  `make_tables.py`, `record_results.py` (`ops result add` rows), `score_all.sh` / `finalize.sh`.
- Scored: `keepfrac_len/kl_results.json` (all curves and levels), `kl_summary.json` (the tables above),
  `q32k_scored.json`, `q64k_scored.json`, `INTERIM_q32k.md`; chart `keepfrac_len/keepfrac_vs_length.png`; hub block
  `keepfrac_len/hub_keeplen_block.html`; log entries `keepfrac_len/experiment_log_entry.json`.
- Raw run JSONs + logs: weka `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/early_loss_proxy/results/kl<L>/
  elp-kl<L>-<arm>-sd<seed>.json`, mirrored to `s3://ai2-llm/checkpoints/prasanns/early_loss_proxy/results/kl<L>/`
  (`../fetch_beaker_results.sh kl<L>`); quick reads on mooney `/data/prasann/early_loss_proxy/results/q{32,64}k/`.
- Pools / held-out: mooney `/data/prasann/early_loss_proxy/shards/kl*`, `heldout_kl{,s}.npz`; weka copies under
  `.../prasanns/early_loss_proxy/` (staged by Beaker `elp-stage-weka-b47e`, 01M7NW1V51Y38BC14Q2KXR6MXX).
- Numbers recorded with `ops result add --check` (experiment `tokdrop_lr`, metrics `heldout_ce_dense_final`,
  `flop_x_vs_dft_at_mid_ce`, `flop_x_vs_dft_at_mid_ce_worst_task`; run ids `elp-kl<L>-<arm>-sd<seed>-probe`). `--check` warnings, all investigated: every row has eval_size 48
  / 144 (flagged above); the "deviation vs prior pass" warnings on q32k / q64k rows are two different arms that ran
  back to back in one slurm job (e.g. 5201920 = 40% then 2.5%), which ops reads as two passes of one training run.
