---
title: Gradient sparsity vs context length - how many context tokens carry the training gradient, 2k to 256k
date: 2026-11-17
updated: 2026-11-18
experiment: grad_composition
area: token_routing
kind: result
status: current
superseded_by: 
artifact: 
summary: Under the gradient-concentration table's protocol (frozen Qwen3.5-4B, blocks 7-31 weight gradient, 16-row steps, cos 0.95), the share of context tokens a per-row quota must keep does not fall with length on fine-tuned checkpoints; it rises slowly (mt11 dense f1: 6.1% at 2k, 8.6% at 64k, 17% at 256k; slope +0.19 [0.15, 0.23], tokens ~T^1.19; global oracle slope +0.47), same on mt4 f7808. On the untrained base model the rise does not survive inside the context: for rows that need any context tokens the share falls with length (geometric mean 1.7% at 2k to 0.15% at 256k, row slope -0.49 [-0.56, -0.41], tokens ~T^0.5), and its step-level numbers are ~0 because the base gradient sits in prompt/query/answer tokens. So the trained-model rise comes from fine-tuning spreading the gradient over the context, mostly on contradiction and outlier. Longest valid row 256k (8 H200s fit 524,288 tokens, but the forward breaks past ~262k). eval_size 512 per length to 64k, 192 at 128k, 80 at 256k.
---
# Gradient sparsity vs context length

**Question (Prasann, 2026-11-17, via the main agent).** "A similar table which shows something similar but for
increasing context lengths (whatever the maximum is that can fit on 8 GPUs). Maybe it can be a plot showing how
row-level gradient sparsity scales with context length." The table extended is the hub's gradient-concentration
table: on frozen dense SFT checkpoints, the share of body (context-document) tokens that must be kept for the
kept-token gradient to reach cos 0.95 / 0.9 with the full gradient (7,808 rows/task at 8k: global oracle 0.47%,
per-row quota 4.6%, per-document quota 13.7%, random 53.7%).

## Verdict

**Row-level gradient sparsity does not scale like 1/T. The kept share stays roughly constant and creeps up, so the
kept token count grows slightly faster than the row.** On the mt11 dense f1 checkpoint (trained on rows up to 64k),
the per-row quota share for cos 0.95 is 6.1% at 2k, 8.7-9.8% from 8k to 64k, 11.6% at 128k and 17.0% at 256k
(⚠ eval_size 192 / 80 at those two, and both are past the 64k training length). Log-log slope of share vs length:
**+0.19 [0.15, 0.23]** (95% bootstrap over steps), so **kept tokens per row ∝ T^1.19** (124 tokens at 2k, 5,600 at
64k, 42,000 at 256k). A constant kept-token count (slope -1) is far outside the interval, and so is an exactly constant
fraction (slope 0). Without the two long, small points the slope is the same: +0.14 [0.10, 0.20] over 2k-64k
(in distribution, eval_size 512 each) and +0.15 [0.11, 0.20] over 2k-128k. At cos 0.9 the per-row slope is +0.23 [0.13, 0.32].

- The other rules: the **global oracle** (top-k across all 16 rows of a step) rises fastest, 0.64% -> 4.5% (slope
  +0.47 [0.42, 0.51], tokens ∝ T^1.47): at short lengths most of a step's gradient sits in a few rows, and longer
  rows spread it. The **per-document quota** rises 12% -> 22-25% (slope +0.17 [0.12, 0.21]). **Random** stays at
  53-77% (slope +0.06). The gap between per-row and per-document quota narrows from 2x to 1.3x at 256k.
- **Per row against its own gradient** (each row's own top-k vs its own full gradient, 1 row instead of 16): the
  share of rows whose gradient is already carried by the always-kept non-body tokens (prompt, query, answer; 0 body
  tokens needed) falls from 74% at 2k to 5% at 256k. Among the remaining rows the geometric-mean share is 2.0% at
  2k and settles at 3.0-3.5% from 16k to 256k. A row-level regression over those rows (log share on log T and log
  CE) gives a length slope of -0.01 [-0.06, 0.05]: for a row that needs any body tokens, the share is
  length-independent once difficulty is held fixed. The mean (5-7% at 32k-256k) is pulled up by a tail of rows.
- **Per task (row vs own gradient, cos 0.95)**: contradiction rises steadily, 0.7% -> 16.5% of body tokens (row
  slope +0.64); outlier jumps from 0.7% at 2k to 10% at 8k and then stays at 6.8-11%; oolong is flat at 2.6-4.2%;
  **nq is the only task whose share falls**, 0.71% -> 0.41% (tokens ∝ T^0.76). So "document-drop style" budgets
  that look fine on the mean are tightest on contradiction and outlier, as in the 11-17 handoff trap.
- **Second checkpoint**: the mt4 dense 7,808 rows/task model (trained at 8k only) shows the same rise from 8k to 64k
  on its own layout (per-row 5.1% -> 9.1%, global 0.47% -> 3.9%), and on newly generated 8k rows reproduces the
  published table (0.47 / 5.1 / 10.8 / 53.5%). The rise is therefore not specific to mt11 or to its query layout.
- **Confound that is not removed**: answer CE rises with length (mt11: 0.034 at 2k -> 0.19 at 64k -> 0.36 at 256k),
  and within a length harder rows need a larger share (row-level coefficient on log CE +0.19). Part of the rise is
  "long rows are harder for this model", not "long rows are intrinsically less sparse". The length-generalization
  part (128k/256k past mt11's 64k training length) adds to this; the slope over 2k-64k alone (in-distribution) is
  per-row +0.14 [0.10, 0.20], global +0.55 [0.49, 0.61] (6.1% -> 8.6% and 0.64% -> 3.4%).

**Third checkpoint, the untrained base (added 2026-11-18; details below).** The rise with length does not survive
on the base model where it matters:
- Inside the context, the base gets *sparser* with length. Restrict to rows that need any context tokens, and score
  each row against its own gradient. The base's geometric-mean share falls from 1.66% at 2k to 0.38% at 64k and 0.15%
  at 256k: row-level slope **-0.49 [-0.56, -0.41]**, tokens ∝ T^0.51. On the same rows, mt11 gives
  **+0.13 [0.08, 0.18]** (2.0% → 3.2-3.5%) and mt4 f7808 gives **+0.28 [0.19, 0.36]**.
- The base's step-level numbers are near zero and not comparable as a trend. Its gradient sits almost entirely in the
  always-kept prompt / query / answer tokens: body tokens hold 9-18% of the energy score, and answer CE is 1.6-2.0. So
  cos 0.95 needs about 0 body tokens up to 8k. The per-row quota then rises to 1.6% at 64k and falls back to 0.27% at
  256k as the body's share of the energy drops again (18% → 9%).
- 27-52% of base rows need no body tokens at any length; on mt11 this falls from 74% to 5%.
- With log CE in the row-level regression the base slope is +0.43 [0.32, 0.54], but with a log-CE coefficient of -4.
  On the base, CE separates tasks (oolong 2.6-3.3, nq 1.7-2.4, contradiction 0.6-1.6) rather than row difficulty, so
  that coefficient is a task effect and this regression is not a usable control there.
- Per task, the base is flat or falling everywhere except contradiction (row slopes: contradiction +0.18, nq -0.18,
  oolong -0.05, outlier -0.84). On mt11 they are +0.64, -0.24, -0.03 and +0.16.

**Reading.** The rise on fine-tuned checkpoints is not a property of long context as such. Fine-tuning moves the
gradient into the context (body share of the energy goes from about 10% to about 40% on mt11), and the trained model
spreads that context gradient over more tokens as rows grow, mostly on contradiction and outlier. The base model, which
has not learned the tasks, concentrates its small context gradient on fewer and fewer tokens. This does not separate
"in-distribution" from "learned the task": both fine-tuned models rise, including mt4 well past its 8k training length
(8k → 64k). Difficulty within a fine-tuned model still matters (mt11's log-CE coefficient is +0.19).

**What this means for token dropping.** A router that keeps a fixed fraction of each row's context (the recipe's
20%/7%) is consistent with this; a router that keeps a fixed token count would fall below the 0.95 attribution
bound well before 64k. This agrees with the keep-fraction thread (`records/keep-fraction-vs-length.md`, branch
`claude/keepfrac-vs-length-4bf10c`: the best keep stays at 20-40% from 8k to 64k and low keeps break harder at long
rows). Attribution is an upper bound only (`records/oracle-tokens-under-routing.md`: forward removal breaks it).

## Results: base Qwen3.5-4B (q35-4b-base-markerfix, no fine-tuning; added 2026-11-18)

Same rows (seed 20261117, query "both" layout as mt11), protocol, rules and eval sizes as the mt11 run. Checkpoint:
the weka `ctc_suite/bases/q35-4b-base-markerfix` (the base mt11 was trained from), via its existing model-only bf16 S3
copy `_transfer/gradconc/q35-4b-base-markerfix`.

cos 0.95:

| rule | 2k (T≈2k) | 4k (T≈4k) | 8k (T≈8k) | 16k (T≈17k) | 32k (T≈34k) | 64k (T≈67k) | 128k (T≈134k) | 256k (T≈253k) |
|---|---|---|---|---|---|---|---|---|
| eval_size rows (steps) | 512 (32) | 512 (32) | 512 (32) | 512 (32) | 512 (32) | 512 (32) | 192 (12) ⚠ | 80 (5) ⚠ |
| Global oracle (16-row step) | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.01] | 0.01% [0.00, 0.02] | 0.14% [0.10, 0.18] | 0.32% [0.24, 0.40] | 0.71% [0.60, 0.82] | 0.54% [0.30, 0.86] | 0.15% [0.00, 0.45] |
| Per-row quota | 0.00% [0.00, 0.00] | 0.01% [0.00, 0.04] | 0.01% [0.00, 0.03] | 0.19% [0.12, 0.28] | 0.49% [0.34, 0.64] | 1.56% [1.25, 1.88] | 1.45% [0.39, 2.98] | 0.27% [0.00, 0.82] |
| Per-document quota | 0.00% [0.00, 0.00] | 0.04% [0.00, 0.12] | 0.06% [0.03, 0.10] | 0.53% [0.36, 0.74] | 1.42% [1.07, 1.81] | 2.90% [2.45, 3.36] | 1.89% [0.92, 3.20] | 0.12% [0.02, 0.28] |
| Random | 0.00% [0.00, 0.00] | 0.95% [0.00, 2.45] | 2.51% [1.08, 4.21] | 19.28% [15.83, 22.79] | 29.34% [26.03, 32.92] | 40.65% [36.25, 44.83] | 28.69% [20.50, 36.24] | 5.29% [1.07, 12.48] |
| Global oracle (16-row step): tokens per row | 0 | 0 | 1 | 22 | 104 | 464 | 704 | 370 |
| Per-row quota: tokens per row | 0 | 1 | 1 | 31 | 158 | 1,011 | 1,882 | 674 |
| Per-document quota: tokens per row | 0 | 2 | 5 | 86 | 463 | 1,885 | 2,454 | 284 |
| Row vs its own gradient (mean / median) | 1.54% / 0.00% | 1.50% / 0.19% | 1.33% / 0.29% | 1.59% / 0.26% | 1.94% / 0.10% | 2.47% / 0.07% | 2.62% / 0.04% | 2.16% / 0.01% |
| Row vs its own gradient: geometric mean over rows with k > 0 | 1.66% | 1.43% | 1.10% | 0.73% | 0.44% | 0.38% | 0.25% | 0.15% |
| Row vs its own gradient: tokens per row | 35 | 58 | 106 | 257 | 628 | 1,551 | 3,296 | 5,317 |
| body tokens per row | 2,043 | 3,909 | 8,001 | 16,282 | 32,574 | 64,983 | 130,036 | 246,306 |
| body tokens' share of the down7 energy score | 11% | 12% | 14% | 17% | 18% | 18% | 15% | 9% |
| rows needing 0 body tokens (row vs own, this cos) | 52% | 45% | 41% | 32% | 29% | 27% | 27% | 29% |
| answer CE (mean) | 1.599 | 1.662 | 1.721 | 1.789 | 1.908 | 2.036 | 2.004 | 2.015 |

cos 0.9:

| rule | 2k (T≈2k) | 4k (T≈4k) | 8k (T≈8k) | 16k (T≈17k) | 32k (T≈34k) | 64k (T≈67k) | 128k (T≈134k) | 256k (T≈253k) |
|---|---|---|---|---|---|---|---|---|
| eval_size rows (steps) | 512 (32) | 512 (32) | 512 (32) | 512 (32) | 512 (32) | 512 (32) | 192 (12) ⚠ | 80 (5) ⚠ |
| Global oracle (16-row step) | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.01] | 0.04% [0.00, 0.12] | 0.00% [0.00, 0.00] |
| Per-row quota | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 0.01% [0.00, 0.02] | 0.16% [0.00, 0.46] | 0.00% [0.00, 0.00] |
| Per-document quota | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 0.03% [0.01, 0.06] | 0.26% [0.15, 0.41] | 0.29% [0.00, 0.83] | 0.00% [0.00, 0.00] |
| Random | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 1.77% [0.51, 3.38] | 7.09% [4.48, 10.17] | 4.24% [0.00, 9.92] | 0.00% [0.00, 0.00] |
| Global oracle (16-row step): tokens per row | 0 | 0 | 0 | 0 | 0 | 3 | 56 | 0 |
| Per-row quota: tokens per row | 0 | 0 | 0 | 0 | 0 | 6 | 207 | 0 |
| Per-document quota: tokens per row | 0 | 0 | 0 | 0 | 9 | 172 | 377 | 0 |
| Row vs its own gradient (mean / median) | 0.18% / 0.00% | 0.17% / 0.00% | 0.15% / 0.00% | 0.25% / 0.00% | 0.41% / 0.00% | 0.70% / 0.00% | 0.81% / 0.00% | 0.51% / 0.00% |
| Row vs its own gradient: geometric mean over rows with k > 0 | 0.32% | 0.24% | 0.17% | 0.18% | 0.22% | 0.21% | 0.26% | 0.19% |
| Row vs its own gradient: tokens per row | 4 | 7 | 12 | 40 | 133 | 440 | 1,016 | 1,254 |
| body tokens per row | 2,043 | 3,909 | 8,001 | 16,282 | 32,574 | 64,983 | 130,036 | 246,306 |
| body tokens' share of the down7 energy score | 11% | 12% | 14% | 17% | 18% | 18% | 15% | 9% |
| rows needing 0 body tokens (row vs own, this cos) | 69% | 65% | 59% | 52% | 57% | 58% | 64% | 74% |
| answer CE (mean) | 1.599 | 1.662 | 1.721 | 1.789 | 1.908 | 2.036 | 2.004 | 2.015 |

Per task (row vs its own gradient, cos 0.95):

| task | 2k | 4k | 8k | 16k | 32k | 64k | 128k | 256k |
|---|---|---|---|---|---|---|---|---|
| contradiction: row vs own gradient | 5.27% (121 tok) | 4.98% (189 tok) | 4.21% (331 tok) | 5.36% (865 tok) | 7.17% (2,324 tok) | 9.50% (5,962 tok) | 10.21% (12,837 tok) | 8.55% (21,007 tok) |
| nq: row vs own gradient | 0.02% (0 tok) | 0.05% (2 tok) | 0.04% (3 tok) | 0.17% (28 tok) | 0.23% (76 tok) | 0.17% (109 tok) | 0.10% (135 tok) | 0.06% (160 tok) |
| oolong: row vs own gradient | 0.04% (1 tok) | 0.05% (2 tok) | 0.26% (21 tok) | 0.03% (5 tok) | 0.03% (10 tok) | 0.12% (77 tok) | 0.11% (148 tok) | 0.02% (46 tok) |
| outlier: row vs own gradient | 0.85% (18 tok) | 0.91% (38 tok) | 0.81% (68 tok) | 0.79% (130 tok) | 0.31% (103 tok) | 0.08% (56 tok) | 0.05% (64 tok) | 0.02% (55 tok) |
| contradiction: documents per row (CE) | 56 (0.555) | 92 (0.669) | 187 (0.785) | 379 (0.953) | 762 (1.144) | 1,461 (1.311) | 2,907 (1.449) | 5,646 (1.578) |
| nq: documents per row (CE) | 11 (1.808) | 23 (1.728) | 48 (1.749) | 104 (1.748) | 208 (1.858) | 413 (2.102) | 825 (2.362) | 1,615 (2.304) |
| oolong: documents per row (CE) | 39 (3.307) | 78 (3.323) | 151 (3.251) | 303 (3.168) | 584 (3.199) | 1,136 (3.195) | 2,237 (2.583) | 4,851 (2.642) |
| outlier: documents per row (CE) | 14 (0.726) | 28 (0.930) | 57 (1.099) | 111 (1.286) | 220 (1.432) | 451 (1.535) | 901 (1.623) | 1,680 (1.536) |
| rows per task | 128 | 128 | 128 | 128 | 128 | 128 | 48 | 20 |

Fits. The step-level fits are unreliable on the base: shares are about 0 up to 8k and fall again past 64k. A length
whose mean share is below 0.1% is left out of the log fit, and the range used is shown.

| quantity | slope of log(share) vs log(T) [95% CI] | implied tokens ∝ T^ |
|---|---|---|
| docq7_body|cos0.9 (67k-134k) | 0.133 [-4.955, 1.803] | 1.13 |
| rand7_body|cos0.9 (34k-134k) | 0.633 [-3.945, 1.777] | 1.63 |
| rowself|cos0.9 (2k-253k) | 0.353 [0.227, 0.433] | 1.35 |
| down7_body|cos0.95 (17k-253k) | 0.117 [-0.765, 0.462] | 1.12 |
| rowq7_body|cos0.95 (17k-253k) | 0.277 [-0.843, 0.643] | 1.28 |
| docq7_body|cos0.95 (17k-253k) | -0.393 [-0.897, -0.109] | 0.61 |
| rand7_body|cos0.95 (4k-253k) | 0.568 [0.254, 1.339] | 1.57 |
| rowself|cos0.95 (2k-253k) | 0.127 [0.053, 0.184] | 1.13 |
| row-level log k ~ log T + log CE, cos0.9 | 1.017 [0.883, 1.162] (log CE coef -4.33) | 2.02 |
| row-level log k ~ log T (no CE term, rows with k > 0, n=1326), cos0.9 | -0.040 [-0.144, 0.065] | 0.96 |
| row-level, solved rows only (CE<0.01, n=0), cos0.9 | 0.000 [0.000, 0.000] | 1.00 |
| row-level, contradiction, cos0.9 | 0.404 [0.321, 0.488] | 1.40 |
| row-level, nq, cos0.9 | -0.468 [-0.707, -0.212] | 0.53 |
| row-level, oolong, cos0.9 | -1.003 [-1.662, -0.493] | -0.00 |
| row-level, outlier, cos0.9 | -0.672 [-0.871, -0.464] | 0.33 |
| row-level log k ~ log T + log CE, cos0.95 | 0.428 [0.322, 0.537] (log CE coef -3.96) | 1.43 |
| row-level log k ~ log T (no CE term, rows with k > 0, n=2106), cos0.95 | -0.486 [-0.558, -0.409] | 0.51 |
| row-level, solved rows only (CE<0.01, n=0), cos0.95 | 0.000 [0.000, 0.000] | 1.00 |
| row-level, contradiction, cos0.95 | 0.177 [0.149, 0.205] | 1.18 |
| row-level, nq, cos0.95 | -0.184 [-0.311, -0.051] | 0.82 |
| row-level, oolong, cos0.95 | -0.053 [-0.582, 0.392] | 0.95 |
| row-level, outlier, cos0.95 | -0.835 [-0.912, -0.758] | 0.16 |

mt11, extra rows for comparison (cos 0.95):

| quantity | 2k | 4k | 8k | 16k | 32k | 64k | 128k | 256k |
|---|---|---|---|---|---|---|---|---|
| Row vs its own gradient: geometric mean over rows with k > 0 | 2.01% | 1.97% | 2.82% | 3.02% | 3.01% | 3.44% | 3.45% | 3.15% |
| body tokens' share of the down7 energy score | 10% | 16% | 30% | 36% | 39% | 41% | 43% | 39% |
| rows needing 0 body tokens (row vs own, this cos) | 74% | 58% | 30% | 15% | 10% | 6% | 6% | 5% |

Figures: `grad_sparsity_rowlevel.png` (three checkpoints: rows needing no body tokens; geometric-mean share for the
rest). `grad_sparsity_vs_length.png` now shows the base as dotted x lines; zero shares are not drawn on the log axis.

**Query layout check (base, 8k, 512 rows each):** the query-after layout gives somewhat larger step-level shares than
query-both. Per-row quota at cos 0.95 is 0.19% vs 0.01%, global oracle 0.05% vs 0.01%, random 9.8% vs 2.5%. Both are
tiny next to mt11's 8.7%. Row-level numbers barely move: 40% vs 41% of rows need no body tokens, geometric mean 0.99%
vs 1.10%, CE 1.722 vs 1.721 (`data/base_8k_query_layout.json`). The layout does not change the picture.

## Max length on 8 GPUs

| setup | row length | peak memory per GPU | result |
|---|---|---|---|
| 1 H200, all blocks | 262,144 (tiled row) | 109.6 GiB | fits (pass 1 86 s, pass 2 141 s) |
| 1 H200 | 327,680 | 122.5 GiB | fits (chunked energy pass) |
| 1 H200 | 393,216 / 524,288 | 135-138 GiB | OOM |
| 2 H200 (blocks split) | 524,288 | 87 / 135 GiB | OOM on GPU 1 |
| **8 H200 (blocks 7-31 spread, sneetches)** | 393,216 | 86-92 GiB | fits (300 + 260 s per row) |
| **8 H200** | **524,288** | **115-121 GiB** | **fits (313 + 411 s per row)** |
| 8 H200 | 589,824 | - | CUDA illegal memory access in the backward |
| 8 H200 | 655,360 | 135 GiB on GPU 7 | OOM |

**Memory max on 8 H200s = 524,288 tokens** (per-GPU peak 121 GiB; the limit is one block's recompute peak, not the
stored activations, so more GPUs would not help much). Above 524,288 a kernel fails with an illegal memory access
(asynchronous error, kernel not localized; consistent with a 32-bit index overflow at 2^31 / 4096).

**But the dense forward is not valid past ~262k tokens, so the measured sweep stops at 256k.** Forward-only answer
CE on real rows (`--ce-only`, 1 GPU, `data/ce_*.json`): 256k rows give CE 0.002-0.69 (native RoPE and YaRN x2 agree
within 0.1); 306-350k rows give CE 1.9-4.5 for every task, native or YaRN x2; 368k+ rows give NaN, first non-finite
output in block 0 (and in block 20's Gated DeltaNet output norm on a 433k oolong row). YaRN does not fix it, so it
is not positional. It is consistent with a numerical / indexing failure in the Gated DeltaNet kernels past 2^18
tokens. A first 512k attempt produced NaN losses on every row and was cancelled. Gradients past 262k would be
meaningless, so no point is reported there. Finding and fixing that kernel would let the sweep reach 512k on 8 GPUs.
**Every reported row is at most 260,491 tokens and uses native RoPE**, so YaRN is not used anywhere in the tables.

## Setup

- **Protocol = `probe.py` / `analyze.py` unchanged** (the source of the published table, branch
  `claude/grad-concentration-d91a37`; reused by import). Gradient = summed answer-CE weight gradient of every
  `nn.Linear` in blocks 7-31 (teacher forced, no-cot, frozen bf16 checkpoint). Token t's term per linear is rank 1
  (delta_t x_t^T); score = energy |delta_t|^2 |x_t|^2 summed over the linears of blocks >= 7; only document-body tokens
  are ranked, every other token is always kept. Rank-1 terms go into TensorSketch accumulators per rank bin (dim
  32,768, the same hash seeds); cos is summed over the (block, mixer|mlp) groups of blocks 7-31; k by linear
  interpolation on the realized kept-body share, mean over steps. One step = 16 rows, 4 per task, interleaved as in
  `probe.py`. Rules: global oracle (top-k over the step), per-row quota, per-document quota (document = marker
  chunk; oolong lines), random.
- **Changes that do not alter the quantity**: k grid extended down to 0.01% (the global oracle needs < 0.05% at
  some lengths); only the four block-7 rankings; blocks 0-6 run without autograd; chunked energy pass; blocks
  optionally spread over GPUs; added per-row self accumulators ("row vs its own gradient").
- **Smoke (reproduction)**: the new code on the original 512 rows of the f7808 checkpoint gives 0.46 / 4.55 / 13.67
  / 53.8% (cos 0.95) and 0.12 / 1.31 / 5.75 / 38.4% (cos 0.9), vs the published 0.47 / 4.6 / 13.7 / 53.7% and
  0.13 / 1.4 / 5.8 / 38.5% (`data/smoke8k_f7808_original_rows.json`).
- **Checkpoints**: primary `m64-dense-f1-lr5e-06` (mt11 64k dense, LR 5e-6, full epoch of mt11L64 = the largest-data
  dense arm, 18,592 PF; rows <= 65,536 tokens, query position both; final `model_and_optim` from weka, staged
  model-only bf16 via S3, Beaker 01M7NZBRDSC2D26QFTSWQVY135). Secondary `f8k-mt4-dense-f7808-tde-lr5e-06` (the
  original table's 7,808 rows/task checkpoint, 8k rows, query after).
- **Rows**: fresh ctc rows for the table's tasks (contradiction, nq, outlier, oolong), generated per length with the
  setA generator (`ctc_calibrated.py build --split train`, eval-calibrated documents per row) but **seed 20261117**
  (seed 42 would reproduce mt11's training rows); converter as setA (dense, qwen3_5 markers, no-cot, oolong chunked
  by line), tokenized twice: query both (mt11's layout) and after (mt4's). Mean row length 2.3k / 4.2k / 8.4k / 16.9k
  / 33.6k / 66.9k / 134k / 253k. Rows per task 128 at 2k-64k (eval_size 512), 48 at 128k (192 ⚠), 20 at 256k (80 ⚠);
  mt4: 64 per task (256 ⚠). Documents per row grow with the rung (table below; contradiction 56 -> 5,646).
- **Error bars**: 95% bootstrap over steps for step-level numbers, over rows for row-level numbers, and steps
  resampled within each length for the fitted slopes. Seed-to-seed (model) variation is not included.
- **Compute**: jsteinhardt H200 (mooney, 1 GPU per job, `--qos=preemptive` above the cap; sneetches for the 8-GPU
  probe). About 8 min per 16-row step at 64k, 25 min at 128k, 57 min at 256k on one GPU.

## Results: mt11 dense f1 (primary)

Share of body tokens for cos 0.95 [95% CI over steps]; tokens per row = step tokens / 16. ⚠ = eval_size < 500.

| rule | 2k (T≈2k) | 4k (T≈4k) | 8k (T≈8k) | 16k (T≈17k) | 32k (T≈34k) | 64k (T≈67k) | 128k (T≈134k) | 256k (T≈253k) |
|---|---|---|---|---|---|---|---|---|
| eval_size rows (steps) | 512 (32) | 512 (32) | 512 (32) | 512 (32) | 512 (32) | 512 (32) | 192 (12) ⚠ | 80 (5) ⚠ |
| Global oracle (16-row step) | 0.64% [0.50, 0.77] | 0.66% [0.51, 0.82] | 1.27% [1.10, 1.46] | 1.92% [1.67, 2.21] | 2.75% [2.48, 3.02] | 3.42% [3.05, 3.76] | 4.50% [3.98, 5.02] | 4.53% [3.54, 5.80] |
| Per-row quota | 6.06% [4.83, 7.21] | 5.67% [4.54, 6.77] | 8.70% [7.65, 9.77] | 9.10% [8.27, 9.92] | 9.75% [9.04, 10.47] | 8.64% [8.10, 9.18] | 11.64% [10.54, 12.79] | 17.02% [14.12, 19.20] |
| Per-document quota | 12.16% [10.05, 14.49] | 10.29% [8.58, 12.05] | 15.67% [13.73, 17.42] | 17.84% [16.15, 19.43] | 18.50% [17.16, 20.04] | 20.55% [19.17, 21.95] | 24.95% [22.98, 27.00] | 22.47% [16.95, 26.93] |
| Random | 53.01% [47.54, 58.37] | 61.34% [54.47, 67.98] | 70.13% [66.26, 73.44] | 74.37% [71.21, 77.44] | 77.33% [74.89, 79.71] | 74.42% [72.66, 76.30] | 73.25% [71.34, 75.71] | 74.19% [67.94, 80.72] |
| Global oracle (16-row step): tokens per row | 13 | 26 | 102 | 313 | 897 | 2,221 | 5,853 | 11,165 |
| Per-row quota: tokens per row | 124 | 221 | 696 | 1,483 | 3,176 | 5,611 | 15,138 | 41,916 |
| Per-document quota: tokens per row | 248 | 402 | 1,253 | 2,904 | 6,025 | 13,359 | 32,450 | 55,335 |
| Row vs its own gradient (mean / median) | 1.17% / 0.00% | 1.78% / 0.00% | 3.98% / 1.89% | 4.70% / 3.44% | 5.02% / 4.41% | 5.31% / 5.47% | 6.00% / 6.65% | 6.96% / 5.54% |
| Row vs its own gradient: tokens per row | 24 | 72 | 329 | 767 | 1,634 | 3,456 | 7,777 | 17,082 |
| body tokens per row | 2,043 | 3,909 | 8,001 | 16,282 | 32,574 | 64,983 | 130,036 | 246,306 |
| answer CE (mean) | 0.034 | 0.046 | 0.052 | 0.093 | 0.125 | 0.192 | 0.258 | 0.356 |

cos 0.9:

| rule | 2k (T≈2k) | 4k (T≈4k) | 8k (T≈8k) | 16k (T≈17k) | 32k (T≈34k) | 64k (T≈67k) | 128k (T≈134k) | 256k (T≈253k) |
|---|---|---|---|---|---|---|---|---|
| eval_size rows (steps) | 512 (32) | 512 (32) | 512 (32) | 512 (32) | 512 (32) | 512 (32) | 192 (12) ⚠ | 80 (5) ⚠ |
| Global oracle (16-row step) | 0.12% [0.08, 0.16] | 0.14% [0.09, 0.18] | 0.35% [0.29, 0.42] | 0.55% [0.44, 0.68] | 0.86% [0.73, 0.99] | 0.84% [0.70, 0.99] | 1.12% [0.89, 1.37] | 1.46% [0.70, 2.37] |
| Per-row quota | 1.23% [0.83, 1.71] | 1.34% [0.86, 1.85] | 2.89% [2.28, 3.46] | 2.91% [2.41, 3.41] | 3.26% [2.77, 3.73] | 2.14% [1.81, 2.47] | 2.81% [2.10, 3.59] | 5.39% [2.45, 8.08] |
| Per-document quota | 3.87% [2.81, 5.04] | 3.55% [2.70, 4.52] | 6.27% [5.34, 7.23] | 7.53% [6.54, 8.57] | 7.69% [6.89, 8.58] | 8.51% [7.70, 9.33] | 11.45% [9.98, 13.06] | 10.16% [5.97, 13.77] |
| Random | 27.64% [21.98, 33.14] | 39.80% [31.89, 47.60] | 50.43% [45.48, 54.62] | 56.23% [51.92, 60.39] | 60.58% [57.57, 63.78] | 56.69% [53.89, 59.57] | 54.31% [51.05, 58.15] | 54.66% [46.74, 63.14] |
| Global oracle (16-row step): tokens per row | 2 | 5 | 28 | 90 | 281 | 547 | 1,458 | 3,603 |
| Per-row quota: tokens per row | 25 | 53 | 231 | 474 | 1,062 | 1,390 | 3,648 | 13,261 |
| Per-document quota: tokens per row | 79 | 139 | 502 | 1,225 | 2,504 | 5,530 | 14,894 | 25,015 |
| Row vs its own gradient (mean / median) | 0.27% / 0.00% | 0.45% / 0.00% | 1.21% / 0.11% | 1.31% / 0.46% | 1.28% / 0.53% | 1.17% / 0.50% | 1.30% / 0.56% | 2.03% / 0.53% |
| Row vs its own gradient: tokens per row | 6 | 18 | 101 | 214 | 415 | 769 | 1,699 | 4,988 |
| body tokens per row | 2,043 | 3,909 | 8,001 | 16,282 | 32,574 | 64,983 | 130,036 | 246,306 |
| answer CE (mean) | 0.034 | 0.046 | 0.052 | 0.093 | 0.125 | 0.192 | 0.258 | 0.356 |

Per task (row vs its own gradient, cos 0.95; tokens per row in brackets) and documents per row:

| task | 2k | 4k | 8k | 16k | 32k | 64k | 128k | 256k |
|---|---|---|---|---|---|---|---|---|
| contradiction: row vs own gradient | 0.72% (16 tok) | 0.50% (20 tok) | 1.79% (141 tok) | 4.29% (691 tok) | 6.01% (1,948 tok) | 7.60% (4,775 tok) | 10.73% (13,501 tok) | 16.50% (40,585 tok) |
| nq: row vs own gradient | 0.71% (12 tok) | 0.66% (24 tok) | 0.85% (64 tok) | 0.90% (147 tok) | 0.93% (302 tok) | 0.89% (575 tok) | 0.66% (854 tok) | 0.41% (1,055 tok) |
| oolong: row vs own gradient | 2.56% (54 tok) | 2.61% (108 tok) | 3.12% (257 tok) | 2.72% (446 tok) | 2.74% (899 tok) | 3.52% (2,303 tok) | 3.73% (4,886 tok) | 4.15% (9,704 tok) |
| outlier: row vs own gradient | 0.69% (14 tok) | 3.36% (138 tok) | 10.17% (856 tok) | 10.90% (1,782 tok) | 10.41% (3,388 tok) | 9.24% (6,170 tok) | 8.87% (11,868 tok) | 6.78% (16,985 tok) |
| contradiction: documents per row (CE) | 56 (0.006) | 92 (0.001) | 187 (0.004) | 379 (0.008) | 762 (0.016) | 1,461 (0.043) | 2,907 (0.154) | 5,646 (0.357) |
| nq: documents per row (CE) | 11 (0.004) | 23 (0.005) | 48 (0.007) | 104 (0.016) | 208 (0.031) | 413 (0.065) | 825 (0.063) | 1,615 (0.102) |
| oolong: documents per row (CE) | 39 (0.124) | 78 (0.172) | 151 (0.170) | 303 (0.283) | 584 (0.308) | 1,136 (0.432) | 2,237 (0.527) | 4,851 (0.550) |
| outlier: documents per row (CE) | 14 (0.001) | 28 (0.004) | 57 (0.026) | 111 (0.064) | 220 (0.144) | 451 (0.227) | 901 (0.289) | 1,680 (0.415) |
| rows per task | 128 | 128 | 128 | 128 | 128 | 128 | 48 | 20 |

Fitted slopes (log share vs log mean row length, 2k-256k; tokens ∝ T^(1 + slope)):

| quantity | slope of log(share) vs log(T) [95% CI] | implied tokens ∝ T^ |
|---|---|---|
| down7_body|cos0.9 | 0.544 [0.446, 0.628] | 1.54 |
| rowq7_body|cos0.9 | 0.230 [0.126, 0.316] | 1.23 |
| docq7_body|cos0.9 | 0.237 [0.167, 0.303] | 1.24 |
| rand7_body|cos0.9 | 0.117 [0.083, 0.156] | 1.12 |
| rowself|cos0.9 | 0.333 [0.273, 0.390] | 1.33 |
| down7_body|cos0.95 | 0.467 [0.422, 0.513] | 1.47 |
| rowq7_body|cos0.95 | 0.190 [0.155, 0.229] | 1.19 |
| docq7_body|cos0.95 | 0.168 [0.125, 0.207] | 1.17 |
| rand7_body|cos0.95 | 0.060 [0.041, 0.082] | 1.06 |
| rowself|cos0.95 | 0.340 [0.297, 0.382] | 1.34 |
| row-level log k ~ log T + log CE, cos0.9 | -0.143 [-0.214, -0.073] (log CE coef 0.13) | 0.86 |
| row-level, solved rows only (CE<0.01, n=773), cos0.9 | -0.479 [-0.591, -0.363] | 0.52 |
| row-level, contradiction, cos0.9 | 0.386 [0.256, 0.510] | 1.39 |
| row-level, nq, cos0.9 | -0.402 [-0.499, -0.308] | 0.60 |
| row-level, oolong, cos0.9 | -0.328 [-0.449, -0.206] | 0.67 |
| row-level, outlier, cos0.9 | 0.013 [-0.071, 0.102] | 1.01 |
| row-level log k ~ log T + log CE, cos0.95 | -0.006 [-0.062, 0.051] (log CE coef 0.19) | 0.99 |
| row-level, solved rows only (CE<0.01, n=942), cos0.95 | -0.021 [-0.143, 0.100] | 0.98 |
| row-level, contradiction, cos0.95 | 0.637 [0.529, 0.749] | 1.64 |
| row-level, nq, cos0.95 | -0.239 [-0.339, -0.140] | 0.76 |
| row-level, oolong, cos0.95 | -0.030 [-0.128, 0.071] | 0.97 |
| row-level, outlier, cos0.95 | 0.157 [0.079, 0.237] | 1.16 |

The row-level regressions use only rows that need at least one body token (k > 0); rows carried entirely by
non-body tokens (74% at 2k, 5% at 256k) cannot enter a log fit.

## Results: mt4 dense 7,808 rows/task (secondary, trained at 8k)

| rule | 2k (T≈2k) | 4k (T≈4k) | 8k (T≈8k) | 16k (T≈17k) | 32k (T≈34k) | 64k (T≈67k) |
|---|---|---|---|---|---|---|
| eval_size rows (steps) | 256 (16) ⚠ | 256 (16) ⚠ | 256 (16) ⚠ | 256 (16) ⚠ | 256 (16) ⚠ | 256 (16) ⚠ |
| Global oracle (16-row step) | 0.66% [0.46, 0.87] | 0.25% [0.14, 0.37] | 0.47% [0.26, 0.70] | 1.31% [0.97, 1.69] | 3.14% [2.78, 3.53] | 3.93% [3.55, 4.31] |
| Per-row quota | 6.14% [4.06, 8.22] | 3.08% [1.60, 4.78] | 5.08% [2.92, 7.31] | 8.47% [7.42, 9.57] | 8.65% [7.97, 9.34] | 9.08% [8.25, 9.90] |
| Per-document quota | 12.32% [8.37, 16.39] | 9.97% [5.22, 14.92] | 10.84% [6.69, 15.00] | 18.59% [15.73, 21.83] | 19.98% [18.44, 21.58] | 18.18% [17.06, 19.38] |
| Random | 58.29% [46.71, 68.94] | 42.52% [27.31, 57.59] | 53.51% [38.39, 67.89] | 75.60% [72.83, 78.42] | 75.67% [73.55, 77.92] | 69.77% [67.08, 73.47] |
| Global oracle (16-row step): tokens per row | 13 | 10 | 37 | 214 | 1,024 | 2,554 |
| Per-row quota: tokens per row | 125 | 121 | 405 | 1,377 | 2,815 | 5,894 |
| Per-document quota: tokens per row | 251 | 392 | 865 | 3,025 | 6,508 | 11,809 |
| Row vs its own gradient (mean / median) | 2.14% / 0.24% | 1.31% / 0.00% | 1.56% / 0.00% | 3.78% / 1.77% | 5.12% / 4.48% | 5.55% / 5.36% |
| Row vs its own gradient: tokens per row | 44 | 53 | 127 | 616 | 1,664 | 3,580 |
| body tokens per row | 2,039 | 3,907 | 7,980 | 16,272 | 32,562 | 64,948 |
| answer CE (mean) | 0.049 | 0.083 | 0.072 | 0.144 | 0.309 | 0.514 |

| task | 2k | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|---|
| contradiction: row vs own gradient | 1.03% (24 tok) | 0.65% (26 tok) | 1.27% (100 tok) | 4.96% (800 tok) | 9.10% (2,948 tok) | 11.10% (6,976 tok) |
| nq: row vs own gradient | 1.35% (23 tok) | 0.99% (36 tok) | 0.82% (61 tok) | 0.85% (138 tok) | 0.98% (320 tok) | 1.08% (699 tok) |
| oolong: row vs own gradient | 1.28% (28 tok) | 1.22% (51 tok) | 1.71% (141 tok) | 2.09% (342 tok) | 1.66% (545 tok) | 2.88% (1,888 tok) |
| outlier: row vs own gradient | 4.92% (102 tok) | 2.40% (98 tok) | 2.43% (205 tok) | 7.24% (1,184 tok) | 8.74% (2,844 tok) | 7.14% (4,758 tok) |
| contradiction: documents per row (CE) | 56 (0.005) | 92 (0.005) | 187 (0.001) | 379 (0.032) | 762 (0.177) | 1,461 (0.391) |
| nq: documents per row (CE) | 11 (0.003) | 23 (0.009) | 48 (0.008) | 104 (0.041) | 208 (0.129) | 413 (0.162) |
| oolong: documents per row (CE) | 38 (0.180) | 78 (0.308) | 145 (0.268) | 303 (0.399) | 568 (0.675) | 1,129 (0.854) |
| outlier: documents per row (CE) | 14 (0.009) | 28 (0.009) | 57 (0.012) | 111 (0.102) | 220 (0.256) | 451 (0.649) |
| rows per task | 64 | 64 | 64 | 64 | 64 | 64 |

Fitted per-row quota slope 2k-64k: +0.24 [0.15, 0.36]; global oracle +0.74 [0.65, 0.85]. Its 2k-8k points are
noisier (eval_size 256) and its 4k point is unusually low on every rule.

## Figure

`research/token_routing/grad_concentration/length_sweep/grad_sparsity_vs_length.png` (cos 0.95; `_cos0.9.png` for 0.9). Left: share of body tokens needed vs row length
(log-log), one line per rule, solid = mt11 dense f1 with 95% bands, dashed = mt4 f7808, dotted = what a constant
token count would look like. Right: absolute body tokens kept per row (step total / 16) plus the row-vs-own-gradient
count.

## Caveats

- Difficulty confound (above): CE rises with length on both checkpoints and share rises with CE.
- 128k and 256k are beyond mt11's training length and under eval_size 500 (192 and 80 rows; 12 and 5 steps). The 256k
  per-row quota (17.0% [14.1, 19.2]) is the highest point; without it the slope is +0.15 [0.11, 0.20] (2k-128k), and
  +0.14 [0.10, 0.20] over 2k-64k.
- Attribution, not routing: these are shares of the dense gradient's attribution, an upper bound on what a router
  can drop (forward removal changes the kept tokens' own gradients).
- Single model seed per checkpoint; the bootstrap covers eval sampling only.
- Lengths are generator rungs; rows within a rung vary by ±5-10% in length (oolong is shorter at 256k: 235-247k).
- The 8-GPU memory limit is real, but its kernel failure above 524,288 and the forward breakdown past ~262k are
  diagnosed only to the module (Gated DeltaNet), not to a specific kernel line.

## Where things are

- Code: `research/token_routing/grad_concentration/length_sweep/` on branch `claude/grad-sparsity-vs-length`: `sweep_probe.py` (probe, multi-GPU, YaRN option,
  `--memprobe`, `--ce-only`), `gen_rows.py` (+ `.sbatch`), `run_sweep.sbatch` (1 node-local GPU), `run_sweep8.sbatch`
  / `run_memprobe8.sbatch` (8 GPUs, self-staging from S3), `analyze_sweep.py`, `plot_sweep.py`, `make_tables.py`,
  `export_results.py`, `stage_rows_weka.sh`, `HANDOFF.md`, `experiment_log_entry.json`.
- Data: `research/token_routing/grad_concentration/length_sweep/data/` (summaries `sweep_mt11_dense_f1.json`, `sweep_mt4_dense_f7808.json`, `sweep_base_q35_4b_markerfix.json`, `base_8k_query_layout.json`,
  `smoke8k_f7808_original_rows.json`; memory probes; CE scans; `results_*.jsonl` = the `ops result` rows). Per-step
  arrays: mooney `/data/prasann/gradlen/results/<label>/step_XXXX.npz` + `rows_XXXX.json`. Rows: mooney
  `/data/prasann/gradlen/tok/{after,both}/`, S3 `_transfer/gradlen/tok/`, weka `prasanns/gradlen/tok/`.
- `ops result` rows: experiment `grad_composition`, run ids `gsl-m11f1-<len>` / `gsl-f7808-<len>` / `gsl-base-<len>`. The base rows include exact 0.0 shares that `--check` flags; they are real (cos already reached with 0 body tokens), not a bug.
