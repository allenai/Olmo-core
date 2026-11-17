---
title: Why gradient sparsity does not predict routing sparsity (attribution vs the true routed gradient)
date: 2026-10-30
updated: 2026-10-30
experiment: tokdrop_lr
area: token_routing
kind: result
status: f1952 + recipe complete (eval_size 512 each); early-phase run (7 checkpoints, eval_size 256 each) complete; s488 / f7808 not run
superseded_by: 
artifact: 
summary: Gradient attribution (cos 0.999) says nothing about routing - forward removal breaks it. On dense f1952 the true routed gradient at the recipe shape keeps cos(E[R],E[D]) 0.12 (oracle) / 0.03-0.04 (random, router), >97% of the loss is kept tokens' terms changing. On the routed-trained recipe checkpoint the router-routed gradient is near-unbiased: cos(E[R],E[D]) 0.87 raw / 0.83 Adam, norm ratio 0.89, routed CE ~ dense; random stays biased (0.18). Block-16 7% with the router: 0.98. eval_size 512.
---
# Why gradient sparsity does not predict routing sparsity

**Question (Prasann, 2026-10-30).** The grad-concentration probe (branch `claude/grad-concentration-d91a37`) finds
that the top 1-4% of tokens by per-token gradient contribution reach cos 0.95 with the minibatch update, and an oracle
shaped like the recipe (keep 20% of body tokens past block 7, 7% past block 16) reaches cos ≥ 0.996, i.e. a 11-20x
first-order FLOP bound. Training gets 4.1x / 3.7x / 2.2x dense FLOPs-to-match, random ≈ router, earlier/more
aggressive keeps fail, and sparse backward (all tokens forward, backprop only the top rows; cos 0.85-0.96) stalls.
Which link in the chain breaks?

**Answer, in one line.** Forward removal. The attribution partial sum is measured under the DENSE forward; the routed
forward changes the kept tokens' activations and, above all, the gradient they receive, and that change is the lost
update (>97% of it), not the dropped tokens' own terms.

## Setup

* Code: `research/token_routing/grad_concentration/` on branch `claude/grad-routing-gap`:
  `routed_probe.py` (probe), `analyze_gap.py` (waterfall), `compare_gap.py` (side-by-side), `launch_gap.py` (Beaker),
  `run_gap_local.sbatch` / `submit_local.py` / `run_gap_analyze.sbatch` / `run_gap_sync.sbatch` (jsteinhardt),
  `record_gap.py` (ops results).
* Rows: 128 unseen fresh-pool rows per task (contradiction, nq, outlier, oolong; rows ≥ 7,808, md5-checked against
  mt4f7808 / mt4s1952) = 512 rows, eval_size=512, no-cot answer CE. 32 batches of 16 rows (4 per task) = one
  optimizer step each.
* Vectors per batch, as TensorSketches (dim 8192 per block x {mixer, mlp} group, fixed hashes; 304 block linears,
  embeddings / LM head excluded):
  * D: the dense batch gradient;
  * A_S: attribution partial sum = dense terms of kept tokens (+ every non-body token; every token before the split);
  * R_S: the TRUE routed gradient: kept tokens gathered at the split with their original RoPE positions (the online
    router's mid_compact for one row), same answer CE.
* Selections S: block-7 split at k = 3 / 7 / 20%, block-16 split at k = 1 / 3 / 7%, and the recipe's nested
  7:0.2 → 16:0.07. Selectors: `og` oracle (batch-global top-k by the probe's down<L> energy), `od` the same score
  allocated per document (doc_share 1, as the router), `rnd` uniform per document (the training random control),
  `rtr` a ridge router emulating the online router (structural + lexical + cos(h,q) + h + h*q features, row-centred
  log ||dL/dh|| target from the dense pass, fit on earlier batches + the other half of the batch, 10% exploration;
  Spearman with its target 0.54 at block 7, 0.57 at block 16; the recipe's stage 2 re-scores the ROUTED h16, overlap
  0.76 with the dense-h16 choice).
* E: leave-one-out expected dense gradient (the other 31 batches = 496 rows). Debiased expected cosines use cross-batch
  terms only: <E X, E Y> ≈ mean_{b≠c} <X_b, Y_c>.
* Adam: neither checkpoint saved optimizer state, so the second moment is ESTIMATED from 32 dense 16-row batches of the
  same rows (Adafactor-style row/column sums of E[G²]); the preconditioner D ≈ R_i^-1/2 C_j^-1/2 keeps every token
  term rank 1.
* Sketch check (exact fp32 on b16 attention.w_out and b20 feed_forward.w2, batch 0): A~D 0.9978 exact / 0.9977
  sketch; R_og~D 0.401 / 0.379; R_rnd~D 0.008 / 0.006.
* Errors: ± = SE over the 32 batches.

## Results: dense f1952 (`f8k-mt4-dense-f1952-tde-lr5e-06`)

Plot data: `research/token_routing/grad_concentration/results/gap_f1952.json` (gitignored; also
`/data/prasann/gradgap/gap_f1952.json` on cubbins); sketches on cubbins `/data/prasann/gradgap/sketches/f1952/`.

### Waterfall at the recipe shape (whole model)

| selector | attribution cos(A,D) | routed cos(R,D) | cos(D,E) → cos(R,E) | exp cos A / R (debiased) | Adam: routed cos | Adam: exp cos R | Adam: proj E[R] on E[D] |
|---|---|---|---|---|---|---|---|
| og (oracle, global) | 0.999 | 0.385±0.031 | 0.047 → 0.021 | 0.996 / 0.124 | 0.290 | 0.120 | 0.82 |
| od (oracle, per doc) | 0.978 | 0.098±0.015 | 0.047 → 0.026 | 0.987 / 0.156 | 0.131 | 0.041 | 0.28 |
| rnd (random, per doc) | 0.855 | 0.023±0.005 | 0.047 → 0.004 | 0.856 / 0.029 | 0.052 | 0.114 | 0.78 |
| rtr (ridge router) | 0.935 | 0.054±0.006 | 0.047 → 0.007 | 0.980 / 0.044 | 0.077 | 0.096 | 0.60 |

* Single block-16 split at 7%: og routed 0.781 (Adam 0.609), exp cos 0.293 (Adam 0.433), Adam projection 1.25,
  routed CE barely moves (contradiction 0.016 → 0.048); rtr 0.387 / exp 0.086 (Adam 0.254); rnd 0.196 / −0.031 (Adam 0.114).
* Routed answer CE on this dense checkpoint at the recipe shape (per answer token; dense in brackets): og
  contradiction 0.60 (0.016), nq 0.32 (0.062), outlier 0.13 (0.077), oolong 0.35 (0.28); rnd and rtr similar.
  The routed forward is far off the dense model's distribution.

### Which effect is largest

1. **Forward removal (largest).** Attribution is near-perfect per step (0.999) and in expectation (0.996); the routed
   gradient at the SAME oracle selection keeps 0.385 per step and 0.124 in expectation. With random or router choice,
   0.02-0.05 per step and 0.03-0.04 in expectation.
2. **Noise is not it.** The dense step itself is ~99% noise: cos(D_b, E) = 0.047, cross-batch cos(D,D) = 0.012,
   bias share 0.007. Attribution keeps its cos to E (ratio ≈ 1.0-1.07); routing loses it.
3. **Bias accumulates, but in a routing-specific direction.** Routed gradients are 5-20x more self-consistent than
   dense ones (pair cos R-R 0.07-0.26 vs D-D 0.012; bias share 0.06-0.25 vs 0.007), with 7-12x the expected norm,
   mostly orthogonal to E[D]. The residual D − R is consistent too (bias share 0.06-0.24): it compounds.
4. **Adam narrows random vs oracle.** In the preconditioned space the expected routed direction keeps 60-82% of the
   dense expected progress (og 0.82, rnd 0.78, rtr 0.60), and random ≈ oracle in expected cosine (0.114 vs 0.120).
   This matches "random trains about as well as the router": per-step cos(R, D) is the wrong metric.
5. **Router vs oracle.** The ridge router is not better than random in expectation (0.096 vs 0.114 Adam).

### Where the lost update D − R lives (recipe shape)

Share of ||D − R||² (component inner product with D − R; ± SE):

| component | og | od | rnd | rtr |
|---|---|---|---|---|
| dropped tokens' own dense terms | 0.1% | 0.8% | 2.9% | 1.8% |
| kept prompt / query: term shift | 38% | 18% | 24% | 15% |
| kept gold-document body: term shift | 22% | 32% | 31% | 33% |
| kept other body: term shift | 26% | 47% | 36% | 45% |
| answer positions: term shift | 14% | 3% | 6% | 6% |

By task, contradiction carries 66-78% of the lost update (outlier 18-28%, nq 3-6%, oolong ≤ 1%). By group, early
blocks dominate: og_recipe b0 mixer 9.7%, b7 mixer 8.2%, b6 mlp 5.1%, b6 mixer 5.0% (the backward signal into the
pre-split blocks changes once the late stack no longer reads the dropped tokens).

## Results: routed-trained recipe checkpoint (`f8k-mt4-lora64all32r20to7-f1952-tde-lr1e-05`)

Whole-model LoRA r64 (merged, W += 32·BA, 304 adapters) trained with the online router at 7:0.2 → 16:0.07 on the same
1,952 rows/task as f1952. Same 512 rows, same probe. Adam estimated the same way on this checkpoint. Plot data:
`research/token_routing/grad_concentration/results/gap_recipe.json`; sketches on mooney
`/data/prasann/gradgap/sketches/recipe/`. Exact checks (b16 attention.w_out, batches 1-3): A~D 0.994-0.997 exact = sketch;
R_og~D 0.19 / 0.34 / 0.48 exact vs sketch within 0.006.

### Side by side (recipe shape and block-16 7%; `compare_gap.py`)

| sel | ckpt | routed cos raw | routed cos Adam | exp cos raw | exp cos Adam | Adam proj E[R]·E[D] | pair cos R-R raw (D-D) | bias share R / D raw | routed CE contra/nq/outlier/oolong (dense) |
|---|---|---|---|---|---|---|---|---|---|
| og_recipe | f1952 | 0.385±0.031 | 0.290±0.025 | 0.124 | 0.120 | 0.82 | 0.129 (0.012) | 0.124 / 0.007 | 0.60/0.32/0.13/0.35 (0.016/0.062/0.077/0.28) |
| og_recipe | recipe | 0.525±0.042 | 0.512±0.039 | 0.415 | 0.323 | 0.97 | 0.044 (0.025) | 0.044 / 0.016 | 0.11/0.18/0.10/0.30 (0.025/0.081/0.095/0.29) |
| rnd_recipe | f1952 | 0.023±0.005 | 0.052±0.005 | 0.029 | 0.114 | 0.78 | 0.116 (0.012) | 0.108 / 0.007 | 0.61/0.25/0.34/0.37 |
| rnd_recipe | recipe | 0.112±0.010 | 0.135±0.010 | 0.179 | 0.142 | 1.00 | 0.225 (0.025) | 0.217 / 0.016 | 0.08/0.25/0.39/0.33 |
| rtr_recipe | f1952 | 0.054±0.006 | 0.077±0.007 | 0.044 | 0.096 | 0.60 | 0.067 (0.012) | 0.060 / 0.007 | 0.56/0.21/0.16/0.35 |
| rtr_recipe | recipe | 0.395±0.027 | 0.422±0.026 | **0.874** | **0.831** | 0.78 | 0.014 (0.025) | 0.007 / 0.016 | 0.047/0.093/0.089/0.32 |
| og_L16_k0.07 | f1952 | 0.781±0.027 | 0.609±0.038 | 0.293 | 0.433 | 1.25 | 0.047 (0.012) | 0.041 / 0.007 | 0.048/0.064/0.085/0.29 |
| og_L16_k0.07 | recipe | 0.952±0.018 | 0.937±0.020 | 0.976 | 0.951 | 0.93 | 0.024 (0.025) | 0.015 / 0.016 | 0.024/0.091/0.097/0.29 |
| rnd_L16_k0.07 | f1952 | 0.196±0.025 | 0.166±0.014 | −0.031 | 0.114 | 0.97 | 0.263 (0.012) | 0.250 / 0.007 | 0.14/0.12/0.20/0.31 |
| rnd_L16_k0.07 | recipe | 0.238±0.026 | 0.232±0.018 | 0.124 | 0.117 | 0.76 | 0.220 (0.025) | 0.206 / 0.016 | 0.029/0.26/0.35/0.31 |
| rtr_L16_k0.07 | f1952 | 0.387±0.022 | 0.275±0.015 | 0.086 | 0.254 | 1.27 | 0.115 (0.012) | 0.102 / 0.007 | 0.088/0.074/0.11/0.31 |
| rtr_L16_k0.07 | recipe | 0.855±0.015 | 0.818±0.018 | **0.983** | **0.944** | 0.92 | 0.021 (0.025) | 0.013 / 0.016 | 0.025/0.090/0.096/0.30 |

* Recipe checkpoint, dense step noise: cos(D_b, E) 0.091 raw / 0.096 Adam; pair cos D-D 0.025. Attribution at the
  recipe shape: og 0.997, rnd 0.896, rtr 0.957 per step.
* rtr_recipe on the recipe checkpoint: ||E[R]|| / ||E[D]|| = 0.89 (f1952: 8.3), adjacent-batch cos R-R 0.009
  (≈ dense 0.019): the router-routed gradient is a near-unbiased, dense-like estimator of the dense direction.
* Random on the same checkpoint stays biased (exp cos 0.18, pair cos R-R 0.23, norm ratio 9.9) and its routed CE on
  outlier stays high (0.39 vs dense 0.095). The checkpoint co-adapted with the ROUTER's kind of selection.
* Lost update at the recipe shape is still >95% kept-token term shifts (prompt 27-48%, gold body 18-29%, other body
  22-37%, answer 7-11%; dropped tokens' own terms ≤ 6%).

### What routed training changes (verdict)

1. On the dense checkpoint, forward removal makes the routed gradient a different, consistent, 8-12x larger direction
   (exp cos with the dense direction 0.03-0.12): gradient attribution (0.996) is not a valid proxy at all.
2. After routed training, the forward under the trained kind of selection is back on-distribution (routed CE ≈ dense)
   and the router-routed gradient is nearly the expected dense gradient: 0.87 at the recipe shape, 0.98 at block 16 / 7%.
   Per-step cos to D (0.40) is low only because the dense step itself is noisy (cos(D, E) 0.09).
3. So the gap between the attribution bound (11-20x) and achieved FLOPs-to-match (2-4x) is not a per-step gradient
   mismatch at the end state: the co-adapted router recovers the expected gradient. The loss is in the transient
   (early steps, when the routed forward is off-distribution and the update is biased), in the early blocks the
   change concentrates in, and in random-vs-router only on a router-co-adapted model.
4. Achievable sparsity from this evidence: a block-16 split keeping ~7% of body tokens preserves the expected
   gradient (0.94-0.98 with a router, on a co-adapted model); the block-7 / 20% stage is the binding one (0.83-0.87).
   Untested: a run that starts at the late split (no block-7 stage), and the cumulative effect of the early biased phase.


## Early phase: when does the routed gradient become unbiased? (2026-10-31)

**Run.** `f8k-mt4-lora64all32r20to7-f1952st244-tdgap-lr1e-05` (Beaker `01M67X8WFPB07FPQW79X9CNRXX`, branch
`claude/grad-routing-gap-train` @ `92201511f`): the f1952 recipe config (launch_fast8k.py arm `lora64all32r20to7`,
LR 1e-5, seed 0, its own online router; the full 488-step schedule hard-stopped at step 244) with permanent checkpoints
at steps 0, 8, 16, 31, 61, 122, 244 (new `--save-fixed-steps`; ephemeral `--resume-every 20`). Probe: one Beaker job per
checkpoint, ⚠ eval_size=256 rows (16 x 16-row steps) per checkpoint, in-job Adam estimate, selections rtr / rnd / og at
the recipe shape plus the aggressive budgets rtr at block 7 / 10% and block 16 / 3%. Plot data:
`research/token_routing/grad_concentration/results/step_table.json` (+ `results/gap_early.json`).
⚠ With 16 batches and a dense cross-batch cos of only 0.03-0.13 after step 0, the debiased expected cosines are noisy
(swings of ±0.2 between neighbouring checkpoints are within noise); read trends, not single cells.

| sel | step | exp cos raw | exp cos Adam | Adam proj | ‖E[R]‖/‖E[D]‖ | per-step cos | pair cos R-R (D-D) | routed CE contra / nq / outlier / oolong (dense) |
|---|---|---|---|---|---|---|---|---|
| rtr_recipe | 0 | 0.943 | 0.948 | 0.93 | 0.95 | 0.825 | 0.753 (0.689) | 1.23/2.13/1.18/3.24 (0.86/2.03/1.09/3.16) |
| rtr_recipe | 8 | 0.209 | 0.413 | 0.48 | 1.64 | 0.180 | 0.120 (0.131) | 0.75/0.27/0.31/0.51 (0.46/0.16/0.29/0.44) |
| rtr_recipe | 16 | 0.652 | 0.678 | 0.79 | 1.07 | 0.309 | 0.075 (0.109) | 0.50/0.21/0.26/0.45 (0.34/0.12/0.24/0.48) |
| rtr_recipe | 31 | 0.841 | 0.809 | 1.07 | 1.28 | 0.282 | 0.151 (0.116) | 0.44/0.20/0.23/0.58 (0.30/0.10/0.23/0.52) |
| rtr_recipe | 61 | 0.809 | 0.852 | 1.11 | 1.52 | 0.209 | 0.065 (0.054) | 0.34/0.20/0.20/0.52 (0.22/0.16/0.20/0.52) |
| rtr_recipe | 122 | 0.654 | 0.702 | 0.88 | 1.77 | 0.282 | 0.034 (0.049) | 0.17/0.12/0.20/0.37 (0.11/0.07/0.19/0.37) |
| rtr_recipe | 244 | 0.886 | 0.882 | 0.75 | 1.00 | 0.387 | 0.016 (0.031) | 0.07/0.10/0.18/0.35 (0.05/0.08/0.17/0.35) |
| rnd_recipe | 0 | 0.870 | 0.850 | 0.84 | 0.93 | 0.751 | 0.814 (0.689) | 1.31/2.56/1.31/3.26 |
| rnd_recipe | 8 | 0.067 | 0.172 | 0.32 | 2.55 | 0.068 | 0.319 (0.131) | 0.94/0.79/0.73/0.51 |
| rnd_recipe | 31 | 0.319 | 0.345 | 0.89 | 2.83 | 0.094 | 0.224 (0.116) | 0.83/0.92/0.75/0.59 |
| rnd_recipe | 122 | 0.217 | 0.154 | 0.77 | 5.68 | 0.093 | 0.121 (0.049) | 0.29/0.62/0.82/0.38 |
| rnd_recipe | 244 | 0.043 | 0.164 | 0.62 | 6.19 | 0.146 | 0.118 (0.031) | 0.13/0.22/0.31/0.36 |
| og_recipe | 0 | 0.979 | 0.967 | 0.96 | 1.00 | 0.927 | 0.679 (0.689) | 0.86/1.96/1.06/3.13 |
| og_recipe | 8 | 0.754 | 0.709 | 0.89 | 1.11 | 0.720 | 0.125 (0.131) | 0.48/0.36/0.29/0.47 |
| og_recipe | 31 | 0.917 | 0.891 | 1.05 | 1.15 | 0.785 | 0.123 (0.116) | 0.31/0.38/0.42/0.55 |
| og_recipe | 122 | 0.427 | 0.662 | 1.03 | 2.50 | 0.660 | 0.122 (0.049) | 0.11/0.33/0.24/0.37 |
| og_recipe | 244 | 0.717 | 0.742 | 0.93 | 1.34 | 0.687 | 0.025 (0.031) | 0.06/0.29/0.18/0.36 |
| rtr_L7_k0.1 | 0 | 0.908 | 0.912 | 0.88 | 0.94 | 0.760 | 0.741 (0.689) | 1.30/2.21/1.24/3.24 |
| rtr_L7_k0.1 | 8 | 0.154 | 0.295 | 0.35 | 1.54 | 0.114 | 0.111 (0.131) | 0.92/0.34/0.44/0.53 |
| rtr_L7_k0.1 | 16 | 0.378 | 0.482 | 0.81 | 1.98 | 0.171 | 0.134 (0.109) | 0.58/0.33/0.34/0.45 |
| rtr_L7_k0.1 | 31 | 0.815 | 0.784 | 1.05 | 1.35 | 0.229 | 0.139 (0.116) | 0.49/0.34/0.24/0.59 |
| rtr_L7_k0.1 | 61 | 0.664 | 0.704 | 1.19 | 2.47 | 0.125 | 0.081 (0.054) | 0.41/0.47/0.32/0.49 |
| rtr_L7_k0.1 | 122 | 0.482 | 0.510 | 0.91 | 2.56 | 0.170 | 0.053 (0.049) | 0.25/0.27/0.24/0.39 |
| rtr_L7_k0.1 | 244 | 0.446 | 0.378 | 0.58 | 2.16 | 0.224 | 0.022 (0.031) | 0.13/0.23/0.20/0.36 |
| rtr_L16_k0.03 | 0 | 0.972 | 0.959 | 0.96 | 0.98 | 0.886 | 0.710 (0.689) | 1.10/2.09/1.15/3.11 |
| rtr_L16_k0.03 | 8 | 0.248 | 0.285 | 0.56 | 2.22 | 0.427 | 0.233 (0.131) | 0.62/0.20/0.31/0.50 |
| rtr_L16_k0.03 | 16 | 0.425 | 0.455 | 0.73 | 1.65 | 0.520 | 0.139 (0.109) | 0.46/0.13/0.28/0.51 |
| rtr_L16_k0.03 | 31 | 0.746 | 0.700 | 1.12 | 1.45 | 0.536 | 0.204 (0.116) | 0.38/0.42/0.25/0.53 |
| rtr_L16_k0.03 | 61 | 0.628 | 0.747 | 1.02 | 1.29 | 0.536 | 0.073 (0.054) | 0.27/0.18/0.25/0.52 |
| rtr_L16_k0.03 | 122 | 0.529 | 0.487 | 0.92 | 1.73 | 0.568 | 0.077 (0.049) | 0.12/0.12/0.20/0.38 |
| rtr_L16_k0.03 | 244 | 0.237 | 0.386 | 0.82 | 3.25 | 0.565 | 0.068 (0.031) | 0.06/0.12/0.21/0.36 |

(rnd / og rows at steps 16 and 61 are in `step_table.json`.)

**Verdict.**
1. Step 0 is trivially aligned (every selection 0.87-0.98): the base model's gradient is one shared, format-learning
   direction (pair cos D-D 0.69), which any context subset reproduces.
2. The bias is an early-phase effect: right after that direction is learned (step 8) the router-routed expected gradient
   drops to 0.21 raw / 0.41 Adam (random 0.07 / 0.17), with ‖E[R]‖ 1.6-2.6x ‖E[D]‖. The router-routed gradient is back
   above 0.8 by step 31 (0.84 / 0.81) and stays mostly there (0.65-0.89, noisy), ending at 0.89 / 0.88 with norm ratio
   1.00 at step 244 (the f1952-trained recipe checkpoint: 0.87 / 0.83). Random never recovers (0.04-0.32; norm ratio
   grows to 6): the co-adaptation is to the router's kind of selection.
3. The aggressive budgets never get there to stay: rtr at block 7 / 10% touches 0.82 / 0.78 at step 31 and then decays
   to 0.45 / 0.38 by step 244 (norm ratio 2.2); rtr at block 16 / 3% peaks at 0.75 / 0.70 (step 31) and decays to
   0.24 / 0.39 (norm ratio 3.3), even though its per-step cosine (0.57) is higher than the recipe's. They were not the
   budgets the model trained at, so this measures how far a recipe-co-adapted model is from supporting them, not a
   model trained at those budgets.

## Limits

* Adam is an estimate (no optimizer state saved), rank-1 per linear, from the same rows.
* Sketch noise: per-linear exact checks agree within ~0.02 on cosines of 0.4 and better on ≥ 0.99.
* The ridge router is an emulation of the online router (same features and target, not its exact online state).
* One dense stage (f1952) and one routed checkpoint (recipe f1952); s488 / f7808 not run. The recipe checkpoint was trained with the ONLINE router; the probe's `rtr` is a ridge emulation fit on this checkpoint's own dense pass (Spearman 0.54 / 0.64 with its target), not the run's router state.
* Gradients exclude embeddings and the LM head.

## Next

* Item 8 (oracle-routed training ceiling) only if the recipe comparison leaves the question open.
