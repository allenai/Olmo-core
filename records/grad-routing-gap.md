---
title: Why gradient sparsity does not predict routing sparsity (attribution vs the true routed gradient)
date: 2026-10-30
updated: 2026-10-30
experiment: tokdrop_lr
area: token_routing
kind: result
status: f1952 + recipe complete (eval_size 512 each); s488 / f7808 not run
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


## Limits

* Adam is an estimate (no optimizer state saved), rank-1 per linear, from the same rows.
* Sketch noise: per-linear exact checks agree within ~0.02 on cosines of 0.4 and better on ≥ 0.99.
* The ridge router is an emulation of the online router (same features and target, not its exact online state).
* One dense stage (f1952) and one routed checkpoint (recipe f1952); s488 / f7808 not run. The recipe checkpoint was trained with the ONLINE router; the probe's `rtr` is a ridge emulation fit on this checkpoint's own dense pass (Spearman 0.54 / 0.64 with its target), not the run's router state.
* Gradients exclude embeddings and the LM head.

## Next

* Item 8 (oracle-routed training ceiling) only if the recipe comparison leaves the question open.
