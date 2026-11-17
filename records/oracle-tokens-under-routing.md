---
title: The gradient-concentration oracle under true routing (token level, frozen dense checkpoints)
date: 2026-11-04
updated: 2026-11-04
experiment: tokdrop_lr
area: token_routing
kind: result
status: complete for dense 1,952 and 7,808 rows/task (eval_size 512 rows each); gate-gradient selection only a 16-row quick pass (unconfirmed); full gate sweep on hold (Prasann 11-03)
superseded_by: 
artifact: 
summary: The 0.95 gradient-concentration oracle (already token-level: a token is kept or dropped in all of blocks 7-31) does not survive forward removal. On frozen dense checkpoints the oracle budgets reproduce (1,952 / 7,808 rows/task, block 7, global 2.0 / 0.6%, per row 9.0 / 5.6%, per doc 19.5 / 15.9%), but the true routed gradient keeps per-step cos 0.08-0.19 with the dense gradient (whole model 0.07-0.15) and is 2-25x its norm (raw, per step); at attribution 0.9, 0.04-0.07. Dropping the tokens from every block (split 0, whole-model ranking) is no better (0.03-0.13); block 16 is the best case (0.04-0.40). Expected (debiased) routed cos at 1,952 rows/task is about 0 at block 7 (-0.25 to 0.11) while attribution keeps 0.94-0.98. eval_size 512 rows per checkpoint. Frozen-model reference for the oracle-train session.
---
# The gradient-concentration oracle under true routing

**Question (Prasann, 2026-11-04, via the parent session).** "Can you validate whether the 0.95 oracle results hold
up if you drop all cells per token". The gradient-concentration oracle (branch `claude/grad-concentration-d91a37`,
hub table "Reading the concentration numbers") says that 0.47% (global) / 4.6% (per row) / 13.7% (per document) of
body tokens carry 95% of one optimizer step's update for blocks 7-31 at 7,808 rows/task (1.96 / 8.1 / 16.9% at
1,952). That is **attribution**: the kept tokens' weight-gradient terms under the unchanged dense forward. Does the
same token set still give the dense gradient when the other tokens are actually **removed from the forward**?

**Reading confirmed with Prasann ("version A").** The oracle numbers were already token-level: `probe.py` ranks
each body token by its energy summed over all linears of blocks >= 7 and keeps or drops it in all of them (no
token x layer cells). So the open part is the routed gradient: tokens dropped in blocks 7-31 by compaction at
block 7, blocks 0-6 dense. As side variants: the whole-model cosine scope, threshold 0.9, split 16, and split 0
(dropped from every block, ranked by whole-model energy).

## Verdict

**No.** On frozen dense checkpoints the oracle budgets reproduce, but removing everything else from the forward
leaves the true routed gradient almost orthogonal to the dense one. At block 7 and attribution 0.95 (the table's
convention) the per-step cosine is 0.08-0.19 over blocks 7-31 and 0.07-0.15 over the whole model, with a gradient
2-25x the dense norm and answer CE 2-40x dense. Lowering the bar to 0.9 makes it worse (0.04-0.07). The
concentration numbers measure where the dense gradient lives, not which tokens a routed forward can do without:
removing 80-99% of the body changes the kept tokens' own gradient terms. This is the frozen-model reference; whether
a model trained under the oracle selection adapts (as the router-trained recipe does: exp cos 0.87, see
`records/grad-routing-gap.md`) is the oracle-train session's question.

## Setup

- Code: `research/token_routing/grad_concentration/support_probe.py --parts q1 --splits 0,7,16 --q1-variants
  t95wm,t95dn,t90wm,t90dn` (branch `claude/grad-support-tokens`, commit d68125b5a), `analyze_support.py`,
  `launch_support.py`; reuses `routed_probe.py` (compaction with original RoPE positions = the online router's
  mid-stack compaction) and `probe.py` (per-token rank-1 energies, TensorSketch, dim 8192 per (block, mixer|mlp)).
- Checkpoints (frozen): dense `f8k-mt4-dense-f1952-tde-lr5e-06` and `f8k-mt4-dense-f7808-tde-lr5e-06`
  (Qwen3.5-4B, 1,952 / 7,808 rows/task, LR 5e-6). Rows: 128 unseen fresh rows per task (contradiction, nq, outlier,
  oolong; rows >= 7,808 of `<task>_fresh`, md5-checked against the training shards) = 512 rows = 32 steps of 16.
- Per step: rank body tokens by own energy summed over blocks >= L (global across the 16 rows / per row / per
  document); pick the smallest grid fraction whose attribution cosine with the dense step gradient reaches the
  threshold, in scope `dn` (weights of blocks >= L only, the oracle-table convention) or `wm` (whole model, blocks
  < L dense). Then one routed pass per set: keep the set (+ every non-body token) from block L on, backprop, compare
  with the dense step gradient. Raw gradient (no Adam preconditioning), answer CE summed over the answer tokens.
- Compute: Beaker jupiter, 16 one-GPU H100 jobs (8 per checkpoint, ~25 min each); 64-row first pass on mooney
  (H200, slurm 4149985-8 and 4149999-4150002). Numbers below are the 512-row Beaker runs.

## Results (eval_size 512 rows per checkpoint, 32 steps; ± = SE over steps)

Block-7 split, threshold 0.95 on blocks 7-31 (the table's numbers in brackets):

| checkpoint | scope | body kept | routed cos, blocks 7-31 | routed cos, whole model | norm ratio | routed CE (dense) |
|---|---|---|---|---|---|---|
| 1,952 rows/task | global | 1.98% (1.96%) | 0.137 ± 0.010 | 0.122 ± 0.012 | 3.3 | 0.723 (0.058) |
| | per row | 9.0% (8.1%) | 0.187 ± 0.018 | 0.146 ± 0.014 | 2.2 | 0.140 |
| | per document | 19.5% (16.9%) | 0.164 ± 0.021 | 0.126 ± 0.019 | 3.8 | 0.201 |
| 7,808 rows/task | global | 0.63% (0.47%) | 0.080 ± 0.012 | 0.084 ± 0.012 | 17.1 | 0.982 (0.026) |
| | per row | 5.6% (4.6%) | 0.144 ± 0.028 | 0.131 ± 0.025 | 14.7 | 0.249 |
| | per document | 15.9% (13.7%) | 0.081 ± 0.016 | 0.068 ± 0.014 | 24.8 | 0.331 |

The budgets land one grid step above the table (the fraction grid is 0.5 / 0.7 / 1 / 1.4 / 2 / 3 / 4 / 5 / 7 / 10 /
14 / 20 / 30% ...), on different rows of the same fresh pools.

Other variants, routed cos over blocks >= L / whole model (keep):

| variant | 1,952 global / per row / per document | 7,808 global / per row / per document |
|---|---|---|
| block 7, threshold 0.9 (blocks 7-31) | 0.058 / 0.067 / 0.053 (keep 0.64 / 3.1 / 8.7%) | 0.051 / 0.044 / 0.035 (0.19 / 1.6 / 7.1%) |
| block 7, 0.95 whole-model scope | 0.062 / 0.074 / 0.062 (0.66 / 3.4 / 9.6%) | 0.060 / 0.056 / 0.059 (0.26 / 2.2 / 8.8%) |
| block 0 (every block), 0.95 whole-model ranking | 0.131 / 0.090 / 0.049 (3.3 / 16.7 / 33.1%) | 0.065 / 0.054 / 0.028 (1.2 / 9.9 / 26.7%) |
| block 0, 0.9 | 0.057 / 0.030 / 0.024 (1.2 / 6.7 / 19.1%) | 0.050 / 0.028 / 0.024 (0.4 / 3.9 / 14.0%) |
| block 16, 0.95 (blocks 16-31) | 0.259 / 0.398 / 0.188, whole 0.378 / 0.466 / 0.283 (0.15 / 0.77 / 2.3%) | 0.038 / 0.094 / 0.079, whole 0.048 / 0.100 / 0.121 (0.08 / 0.31 / 2.0%) |

**Expected (debiased) cosine, 1,952 rows/task** (`analyze_support.py`, cross-step terms over the 32 steps;
Beaker 01M6JXS9ZTGE4EW0TK4SXDEN9G). It is the cosine of the expected routed gradient with the expected dense
gradient, i.e. what accumulates over many steps. The dense gradient is noise-dominated per step (a single step's cos
with its own expectation is 0.045), so these are noisier than the per-step numbers above. There is no SE; treat
|cos| < ~0.15 as zero. Columns are global / per row / per document.

| variant | exp cos, attribution (blocks >= L) | exp cos, routed (blocks >= L) | exp cos, routed (whole model) | exp norm ratio, routed (whole) |
|---|---|---|---|---|
| block 7, 0.95 (blocks 7-31) | 0.94 / 0.94 / 0.98 | 0.03 / -0.27 / 0.13 | 0.01 / -0.25 / 0.11 | 12 / 4.5 / 13 |
| block 7, 0.9 | 0.84 / 0.85 / 0.96 | 0.03 / -0.11 / 0.07 | 0.01 / -0.08 / 0.06 | 14 / 9 / 13 |
| block 0, 0.95 whole-model | 0.94 / 0.95 / 0.99 | = whole | -0.01 / -0.20 / 0.09 | 8 / 8 / 15 |
| block 16, 0.95 (blocks 16-31) | 0.91 / 0.91 / 0.97 | 0.22 / 0.28 / 0.05 | 0.10 / 0.17 / 0.01 | 7 / 4 / 11 |

The attribution sets keep their 0.95 in expectation, so the oracle is not a per-step artifact. The routed gradient's
expected direction, though, is unrelated to the dense one (about 0 at block 7) and an order of magnitude larger.

**7,808 rows/task: the expected dense gradient is not resolvable over 32 steps** (Beaker
01M6K82NTBWYYN2K0H9R70EY9D, after fix c3df16e02). The debiased ||E[D]||^2 is negative for the whole model and for
blocks 7-31, i.e. the mean dense gradient is below the step-to-step noise: a step's cos with the mean of the other
31 is 0.009 ± 0.004 (0.045 at 1,952). Expected cosines there are undefined. Only blocks 16-31 resolve: attribution
0.98-0.99 and routed 0.16 / 0.15 / 0.14 (global / per row / per document, threshold 0.95), with routed norm 50-160x
(inflated by the small dense mean). So the per-step numbers are the usable view at 7,808; both views agree that
routed sets do not reproduce the dense gradient.

Reading:
- **Larger budgets do not fix it within this range.** The per-document budget (16-20%, near the recipe's 20% at
  block 7) is no better than the global one; dropping from every block needs up to a third of the body for
  attribution 0.95 and routes worst.
- **It gets worse with more training data.** At 7,808 rows/task the routed gradient is 11-41x the dense norm (2-5x
  at 1,952) and the routed CE is near 1 for the global set: the dense model has fit these rows (dense CE 0.026), so a
  forward without most of the context produces large, mostly unrelated loss gradients.
- **Block 16 is the best case** at 1,952 (whole-model 0.28-0.47) and collapses at 7,808 (0.05-0.12).

## Gate-gradient selection (quick pass only, unconfirmed)

16-row quick pass, dense 1,952 rows/task, block 7, across-batch allocation, H200 mooney (slurm 4145986), ⚠
eval_size 16 rows (one step, no error bars; run-to-run noise of per-step cos up to ±0.035). Per-step routed cos with
the dense gradient at k = 3 / 7 / 20% of body tokens: own-gradient oracle 0.115 / 0.477 / 0.328; gate |dCE/dg| at
g = 0.5: 0.001 / 0.008 / 0.125; at g = 1: 0.047 / 0.137 / 0.257; random 0.008 / 0.012 / 0.045. Gate sets keep the
answer CE near the oracle's (0.42 vs 0.37 at 7%, dense 0.050) but not the gradient. ⚠ 4-row smoke: a joint set
(95% gate mass + attribution 0.95) keeps ~39% of body tokens at block 7, routed cos 0.97. The soft-gate path at
g = 1 matches dense to <= 5e-4 in whole-model fp32; bf16 attention kernels differ by up to ~10% on CE, so the gate
passes use fp32 attention. The full gate sweep (`launch_support.py --labels f1952,recipe --preset trimmed`,
~23 H100-h per checkpoint) is on hold.

## Limits

- Frozen checkpoints: a model trained under the selection may adapt (the router-trained recipe's routed gradient is
  near-unbiased, `records/grad-routing-gap.md`). The oracle-train session (full training) answers that.
- Raw gradient only (no Adam view for these runs). Headline cosines are per step (16 rows), averaged over 32
  steps. The debiased expected cosine is in for 1,952 rows/task; at 7,808 rows/task it is only defined for blocks
  16-31 (the mean dense gradient is below the noise over 32 steps elsewhere; more steps would be needed).
- The fraction grid is coarse, so budgets are rounded up by up to one grid step.
- Sketched gradients (TensorSketch dim 8192 per group): self-check cos(D summed, D binned) 0.99995.

## Where things are

- Code: branch `claude/grad-support-tokens` (`research/token_routing/grad_concentration/`).
- Beaker jobs: `research/token_routing/grad_concentration/state_gap.json` keys `q1tok512-*` (ops runs
  `gradsupp-f1952-s*of8`, `gradsupp-f7808-s*of8`); sketches on weka
  `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/gradgap/support/q1tok512-{f1952,f7808}/`.
- Per-step JSONs: `results/q1tok512/` (gitignored); 64-row pass on mooney `/data/prasann/gradgap/results/q1tok64-*`.
- Results log: `ops result show --run-id gradsupp-q1tok512-f1952` / `-f7808` (metrics `q1tok_<alloc>_<L>_<thr><scope>_*`,
  218 rows). `--check` flags the f7808-vs-f1952 differences (expected: different checkpoints, and the growth with data
  is part of the result) and a provenance gap (per-step aggregates over 8 Beaker jobs, no single eval job).
