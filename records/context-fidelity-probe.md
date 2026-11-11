---
title: Context-fidelity probe - can cheap stand-ins for dropped document tokens keep the kept tokens' states and gradients dense-like?
date: 2026-11-10
updated: 2026-11-10
experiment: tokdrop_lr
area: token_routing
kind: result
status: f1952 complete (eval_size 512); recipe complete (eval_size 512)
superseded_by: 
artifact: 
summary: No-training probe of cheap stand-ins for the tokens dropped after block 7 (keep 20% per document), on the dense f1952 and recipe checkpoints, eval_size 512. Dense ckpt: plain drop keeps expected gradient cos (Adam) 0.02-0.08; an exact forward with no gradient through dropped tokens restores 0.96-1.00 (step cost 0.55, 1.8x), so the loss is forward context only. The best cheap stand-in is the dropped tokens' stale block-7 state fed to EVERY layer incl. DeltaNet: 0.19-0.42 and 45-74% of the CE gap at 0.43 (2.35x) vs drop 0.37 (2.70x); attention-only stale state, refresh, per-doc / per-span means and linear corrections close little or nothing; 20% / 50% of the stale tokens give ~30% / ~49% of the CE gap (router) but little gradient. Recipe ckpt: drop already 0.88-0.91 (router / attribution), stale context slightly hurts; only random selection gains.
---
# Context-fidelity probe

**Question (Prasann, 2026-11-10).** The gradient-gap probe (`records/grad-routing-gap.md`) found that what dropped
tokens give the kept tokens is context, not gradient: >97% of the routed-gradient error is the kept tokens' own terms
shifting once their context is gone. Can the dropped tokens' computation be cheaply simulated, so the kept tokens
behave as if the full context were there? No training: score cheap stand-ins on frozen checkpoints by how well they
preserve the kept tokens' hidden states, answer CE and gradient, next to each one's best-case per-step FLOP cost.

**Answer, in one line.** Only full context in every layer fixes the gradient: the exact forward (no gradient through dropped tokens) restores it completely, but costs 1.8x per step instead of 2.7x. The cheap stand-ins recover part of the CE and less of the gradient on the dense checkpoint (the best, stale block-7 state in every layer including DeltaNet, reaches expected cos 0.19-0.42 at 2.35x), and on the routed-trained recipe checkpoint there is no gap left for them to close with the router (0.88). None changes the best-case speedup ceiling; the one worth a screen is stale-context-everywhere as a way to make MORE aggressive keeps (5-10%) trainable, which this probe did not test.

## Setup

* Code: `research/token_routing/context_fidelity/` (branch `claude/gallant-robinson-adc54a`): `context_probe.py`
  (probe; reuses `grad_concentration/routed_probe.py` for rows, router emulation, TensorSketch gradients),
  `analyze_context.py`, `make_figure.py`, `record_ctx.py`, `run_ctx_local.sbatch` (jsteinhardt), `launch_ctx.py`
  (Beaker; launched but cancelled after >20 min pending, everything ran on cubbins).
* Checkpoints: dense `f8k-mt4-dense-f1952-tde-lr5e-06` and the recipe `f8k-mt4-lora64all32r20to7-f1952-tde-lr1e-05`
  (LoRA merged at staging). Qwen3.5-4B: 32 blocks, full attention at 3, 7, ..., 31, Gated DeltaNet elsewhere.
* Rows: the same 512 unseen fresh-pool rows as the gradient-gap probe (128 per task, rows >= 7,808, md5-checked
  against mt4f7808 / mt4s1952), no-cot answer CE, 32 batches of 16 rows (4 per task) = one optimizer step each.
  **eval_size = 512 pooled; ⚠ 128 per task** wherever a per-task number appears (CE SE ±0.01-0.04).
  The linear corrections were fitted on 16 further held-out rows per task (64 rows, ~115k kept tokens).
* One cut at the input of block 7 (blocks 0-6 see every token), keep 20% of each document's body tokens
  (prompt, markers, query, answer always kept; original RoPE positions). Selections, per document:
  `rtr` the ridge emulation of the online router (the recipe checkpoint saved no router state, so this is the same
  emulation as the gradient-gap probe; Spearman 0.54 with its target), `rnd` uniform, `od` gradient-attribution
  top-k (dense down7 energy).
* Variants above the cut (nothing is frozen in the PARAMETERS: every weight gets its full gradient through the kept
  tokens; "stale" / "frozen" refers to the dropped tokens' activations):

  | variant | what the kept tokens see of the dropped tokens |
  |---|---|
  | drop | nothing (the recipe; reproduces the earlier probe's `<sel>_L7_k0.2`) |
  | frozen | their block-7 input state, projected by each full-attention block's own norm / K / V; constants, no gradient; DeltaNet blocks skip them |
  | refresh15 | frozen, but the dropped tokens also run block 15 once (no grad); blocks 19-31 read the refreshed state |
  | mean1 / span64 | one stand-in per document / per 64-token span = mean of its dropped tokens' block-7 states, logit bias log(count), at the last dropped position; attention only |
  | lin7 / lin7r | nothing, plus a fixed ridge-fitted linear correction added to the kept tokens after block 15, from [h7] / [h7; h16_drop] to h16_dense - h16_drop |
  | frozen_all ("stale, all layers") | frozen, but in EVERY block >= 7 including DeltaNet: the stale block-7 state is the dropped tokens' input to each mixer, outputs discarded |
  | stale20 / stale50 | stale, all layers, for a random 20% / 50% of each document's dropped tokens (per-row seed); the rest dropped (Prasann, 11-10) |
  | kvfresh (diagnostic) | their DENSE input state at each full-attention block (needs their full forward); DeltaNet skips them |
  | fwdonly (diagnostic bound) | the exact dense forward; the dropped tokens are detached at every block input >= 7 and excluded from the gradient: the best any no-gradient context can do |

* Metrics. Gradient: TensorSketch of every block linear's weight gradient (kept tokens' terms only), raw and
  Adam-preconditioned (estimated second moment, as before). Per-step cos(R_b, D_b) with the dense step (mean ± SE over
  32 batches); expected cos(E[R], E[D]) and norm ratio ||E[R]|| / ||E[D]|| from cross-batch terms only, jackknife SE.
  Per-task expected cosines were also computed and are NOT usable (one task's 128 rows make the debiased dense norm
  too small; values above 1), so per task only CE and per-step cos are reported. Hidden error: mean over kept tokens
  (position 0 excluded) of ||h - h_dense|| / ||h_dense|| at the outputs of blocks 15 / 23 / 31. CE: mean over rows of
  per-answer-token CE. Gap closed vs plain drop: (x - x_drop) / (x_dense - x_drop) for cos (x_dense = 1) and CE,
  1 - err / err_drop for hidden error.
* Step cost: analytic, the flop_meter.py / token_exit convention: 6 x params per token run with gradient, 12 n_heads
  head_dim per attended (query, key) pair (fwd+bwd), 2 x params per token run forward-only (4 n_heads head_dim per
  pair), LM head per token run; the dense row is T x num_flops_per_token(T). Context keys are charged as fwd+bwd pairs
  (conservative: their dK / dV are never computed). Router overhead is not charged (selector-independent).
  Best-case per-step speedup = dense / variant, pooled over rows. Mean row length 7,284 tokens, 1,748 kept.
* Checks: plain drop reproduces the earlier probe at the same cut (f1952 expected cos Adam: rtr 0.061 vs 0.057,
  rnd 0.081 vs 0.106, od 0.024 vs 0.021; norm ratios 7.6 / 13.4 / 12.8 vs 8.6 / 14.0 / 15.3); `fwdonly` CE equals dense
  on every row and its hidden error is 0; `frozen_all` in the stale20/50 run reproduces the main run (rtr 0.418 vs 0.419);
  `span1` (every dropped token its own stand-in, count 1, through the masked-SDPA path) reproduces `frozen` (flash path) on 16 rows: mean CE 0.3646 vs 0.3640 (rtr), 0.1917 vs 0.1919 (od), block-31 error within 0.0005.

## Results: dense f1952 (eval_size 512)

Expected gradient cos with the dense step (± jackknife SE), CE (dense 0.109), hidden error at block 31, step cost:

| variant | step cost (speedup) | router: exp cos Adam / raw, CE, err31, CE gap closed | random: exp cos Adam / raw, CE, err31, CE gap closed | attribution: exp cos Adam / raw, CE, err31, CE gap closed |
|---|---|---|---|---|
| drop | 0.371 (2.70x) | 0.06±0.05 / 0.03, 0.313, 0.44, 0% | 0.08±0.03 / 0.03, 0.373, 0.64, 0% | 0.02±0.03 / 0.12, 0.176, 0.47, 0% |
| frozen (stale, attention only) | 0.382 (2.62x) | 0.14±0.05 / 0.25, 0.266, 0.42, 23% | 0.20±0.05 / 0.17, 0.298, 0.60, 28% | 0.45±0.06 / 0.66, 0.150, 0.44, 39% |
| refresh15 | 0.389 (2.57x) | 0.11±0.05 / 0.24, 0.275, 0.42, 19% | 0.14±0.04 / 0.16, 0.314, 0.61, 22% | 0.25±0.07 / 0.47, 0.153, 0.45, 34% |
| mean1 | 0.371 (2.70x) | 0.07±0.05 / 0.01, 0.317, 0.45, -2% | 0.09±0.03 / 0.05, 0.411, 0.65, -15% | 0.06±0.04 / 0.12, 0.185, 0.47, -14% |
| span64 | 0.371 (2.70x) | 0.07±0.05 / 0.04, 0.315, 0.44, -1% | 0.09±0.03 / 0.04, 0.409, 0.65, -14% | 0.06±0.04 / 0.14, 0.186, 0.47, -15% |
| lin7 | 0.371 (2.70x) | -0.07±0.05 / 0.03, 0.320, 0.38, -4% | -0.02±0.03 / 0.01, 0.416, 0.49, -16% | -0.11±0.03 / 0.09, 0.203, 0.39, -41% |
| lin7r | 0.371 (2.69x) | -0.06±0.04 / 0.03, 0.317, 0.36, -2% | -0.01±0.02 / 0.04, 0.435, 0.45, -24% | -0.12±0.03 / 0.10, 0.204, 0.36, -42% |
| stale, all layers | 0.426 (2.35x) | 0.42±0.07 / 0.46, 0.184, 0.37, 63% | 0.19±0.05 / 0.09, 0.253, 0.42, 45% | 0.39±0.09 / 0.49, 0.126, 0.35, 74% |
| kvfresh (diag.) | 0.553 (1.81x) | 0.31±0.06 / 0.13, 0.160, 0.37, 75% | 0.85±0.05 / 0.44, 0.116, 0.54, 97% | 0.59±0.07 / 0.55, 0.135, 0.39, 61% |
| fwdonly (diag. bound) | 0.553 (1.81x) | 0.99±0.00 / 0.95, 0.109, 0.00, 100% | 0.96±0.01 / 0.62, 0.109, 0.00, 100% | 1.00±0.00 / 0.96, 0.109, 0.00, 100% |

Stale context with only part of the dropped tokens (separate run, `f1952st`, same rows; its rnd selections differ
from the main run, so rnd is compared within the run):

| variant | step cost (speedup) | router: exp cos Adam / raw, CE, err31, CE gap closed | random: exp cos Adam / raw, CE, err31, CE gap closed | attribution: exp cos Adam / raw, CE, err31, CE gap closed |
|---|---|---|---|---|
| drop | 0.371 (2.70x) | 0.06±0.05 / 0.03, 0.313, 0.44, 0% | 0.09±0.03 / 0.01, 0.395, 0.64, 0% | 0.02±0.03 / 0.11, 0.177, 0.47, 0% |
| stale20 | 0.382 (2.62x) | 0.10±0.06 / 0.14, 0.251, 0.39, 30% | 0.08±0.04 / 0.08, 0.319, 0.52, 27% | -0.02±0.04 / 0.13, 0.165, 0.40, 17% |
| stale50 | 0.399 (2.51x) | 0.20±0.06 / 0.26, 0.212, 0.38, 49% | 0.09±0.03 / 0.05, 0.274, 0.45, 42% | 0.03±0.04 / 0.20, 0.147, 0.37, 44% |
| stale, all layers | 0.426 (2.35x) | 0.42±0.07 / 0.46, 0.184, 0.37, 63% | 0.22±0.04 / 0.13, 0.245, 0.42, 52% | 0.38±0.09 / 0.49, 0.127, 0.35, 74% |

Per task (⚠ eval_size 128 per task): answer CE, router selection (dense: contradiction 0.016, nq 0.062,
outlier 0.077, oolong 0.281):

| variant | contradiction | nq | outlier | oolong | per-step cos raw (contra / nq / outlier / oolong) |
|---|---|---|---|---|---|
| drop | 0.587 | 0.199 | 0.124 | 0.342 | 0.01 / 0.15 / 0.18 / 0.62 |
| frozen (attention only) | 0.450 | 0.167 | 0.120 | 0.327 | 0.01 / 0.19 / 0.23 / 0.65 |
| stale20 | 0.401 | 0.151 | 0.116 | 0.338 | 0.01 / 0.23 / 0.27 / 0.67 |
| stale50 | 0.270 | 0.143 | 0.112 | 0.323 | 0.03 / 0.29 / 0.30 / 0.69 |
| stale, all layers | 0.185 | 0.126 | 0.120 | 0.308 | 0.05 / 0.30 / 0.25 / 0.69 |
| kvfresh | 0.186 | 0.077 | 0.092 | 0.285 | 0.03 / 0.53 / 0.33 / 0.85 |
| fwdonly | 0.016 | 0.062 | 0.077 | 0.281 | 0.73 / 0.78 / 0.72 / 0.95 |

## Results: recipe checkpoint

Same rows and probe; plain drop reproduces the earlier probe (expected cos Adam rtr 0.88 vs 0.87, od 0.91 vs
0.91; rnd 0.16 vs 0.15). The dense step's own cross-batch cos is 0.027 (Adam), so per-step cosines are noise-bound.

| variant | step cost (speedup) | router: exp cos Adam / raw, CE, err31, CE gap closed | random: exp cos Adam / raw, CE, err31, CE gap closed | attribution: exp cos Adam / raw, CE, err31, CE gap closed |
|---|---|---|---|---|
| drop | 0.371 (2.70x) | 0.88±0.06 / 0.81, 0.135, 0.32, 0% | 0.16±0.04 / 0.15, 0.256, 0.48, 0% | 0.91±0.03 / 0.86, 0.120, 0.35, -0% |
| frozen (stale, attention only) | 0.382 (2.62x) | 0.81±0.06 / 0.75, 0.136, 0.30, -3% | 0.19±0.05 / 0.22, 0.225, 0.45, 23% | 0.82±0.04 / 0.76, 0.121, 0.33, 68% |
| refresh15 | 0.389 (2.57x) | 0.79±0.07 / 0.77, 0.136, 0.30, -7% | 0.15±0.04 / 0.18, 0.245, 0.45, 8% | 0.83±0.04 / 0.77, 0.120, 0.33, 19% |
| mean1 | 0.371 (2.70x) | 0.67±0.09 / 0.66, 0.144, 0.31, -62% | 0.15±0.05 / 0.20, 0.291, 0.47, -26% | 0.79±0.06 / 0.77, 0.133, 0.34, 903% |
| span64 | 0.371 (2.70x) | 0.66±0.09 / 0.66, 0.144, 0.31, -61% | 0.15±0.05 / 0.21, 0.290, 0.47, -26% | 0.79±0.06 / 0.77, 0.128, 0.34, 590% |
| lin7 | 0.371 (2.70x) | 0.89±0.06 / 0.82, 0.136, 0.28, -6% | 0.16±0.05 / 0.17, 0.270, 0.42, -10% | 0.92±0.02 / 0.85, 0.121, 0.30, 55% |
| lin7r | 0.371 (2.69x) | 0.86±0.06 / 0.80, 0.137, 0.26, -11% | 0.14±0.05 / 0.18, 0.272, 0.39, -12% | 0.91±0.02 / 0.84, 0.122, 0.29, 125% |
| stale20 | 0.382 (2.62x) | 0.90±0.03 / 0.81, 0.136, 0.28, -6% | 0.24±0.05 / 0.24, 0.236, 0.42, 14% | 0.92±0.02 / 0.88, 0.124, 0.31, 257% |
| stale50 | 0.399 (2.51x) | 0.86±0.04 / 0.77, 0.138, 0.28, -20% | 0.24±0.04 / 0.25, 0.238, 0.39, 13% | 0.90±0.02 / 0.91, 0.125, 0.30, 380% |
| stale, all layers | 0.426 (2.35x) | 0.81±0.04 / 0.76, 0.142, 0.28, -46% | 0.23±0.04 / 0.26, 0.240, 0.39, 12% | 0.83±0.02 / 0.84, 0.129, 0.30, 650% |
| kvfresh (diag.) | 0.553 (1.81x) | 0.91±0.04 / 0.87, 0.123, 0.26, 85% | 0.80±0.04 / 0.64, 0.128, 0.41, 95% | 0.95±0.02 / 0.90, 0.120, 0.29, -15% |
| fwdonly (diag. bound) | 0.553 (1.81x) | 0.98±0.01 / 0.85, 0.121, 0.00, 100% | 0.91±0.01 / 0.70, 0.121, 0.00, 100% | 0.98±0.01 / 0.91, 0.121, 0.00, 100% |

* Dense CE 0.121 (contradiction 0.025, nq 0.081, outlier 0.095, oolong 0.285; ⚠ 128 rows per task). Router drop:
  0.042 / 0.094 / 0.087 / 0.317; stale everywhere 0.049 / 0.087 / 0.116 / 0.314 (outlier worse); kvfresh
  0.040 / 0.075 / 0.086 / 0.293. Random drop 0.071 / 0.204 / 0.391 / 0.356; random kvfresh 0.037 / 0.080 / 0.109 / 0.289.
* For attribution (od) the plain-drop CE gap is ~0 (0.120 vs 0.121), so its "% gap closed" column is meaningless
  (divides by ~0); read its CE directly. Router CE gap is 0.014; random's 0.135.
* Router norm ratio ||E[R]|| / ||E[D]||: drop 0.82, stale everywhere 1.24, fwdonly 0.75; random drop 5.98 (biased, as
  in the earlier probe), random kvfresh 0.89.

## Figure

`research/token_routing/context_fidelity/figs/context_fidelity.png`: x = step cost (fraction of a dense step),
left = expected gradient cos (Adam), right = answer-CE gap closed vs plain drop; one row per checkpoint.

## What it means

1. **What the dropped tokens contribute is forward context, and it is enough.** On the dense checkpoint the
   exact forward with NO gradient through the dropped tokens (`fwdonly`) restores the expected gradient almost
   completely (Adam 0.96-1.00, raw 0.62-0.96) and the CE exactly. The dropped tokens' own gradient paths (including the
   backward signal into blocks 0-6 through them) are worth little in expectation; per step they matter more (raw step
   cos 0.48-0.88), so they add noise, not bias. This sharpens the gradient-gap finding: a method only has to supply the
   right forward context.
2. **The DeltaNet layers are where the context matters.** Stale block-7 state seen only by the 7 full-attention
   blocks closes 23-39% of the CE gap and little of the gradient; the same state fed to every block (DeltaNet too)
   closes 45-74% and reaches expected cos 0.19-0.42. Even FRESH dense keys / values at the attention blocks
   (`kvfresh`, which costs a full forward) leave the router selection at 0.31 / 75% of the CE gap: 24 of the 32
   blocks are recurrent and their state is built from every token.
3. **Staleness of the attention context is not the problem; summaries do not work.** Refreshing the stale state once
   at block 15 does not help (slightly worse). One mean vector per document or per 64-token span with a log-count
   bias closes nothing (documents are short: ~92 per row, ~60 body tokens each, so span64 ≈ per-document); the
   `span1` check (each dropped token its own stand-in) reproduces `frozen` to 0.001 CE, so this is not a code bug.
4. **A fixed linear correction lowers hidden-state error but hurts the gradient.** It removes 18-42% of the block-15
   error held out and 15-29% at block 31, yet CE gets slightly worse and the expected gradient goes negative (Adam
   -0.12 to -0.01). Matching hidden states in L2 is not the same as preserving what the loss reads; a learned
   correction would need a loss-aware target, i.e. training.
5. **Partial stale context.** Keeping 20% / 50% of each document's dropped tokens as stale context gives, for the
   router, 30% / 49% of the CE gap (all of them: 63%) but expected cos only 0.10 / 0.20 (all: 0.42): CE recovers about
   linearly in the stale share, the gradient needs nearly all of it. For attribution selection the partial versions
   give ~0 gradient gain.
6. **On the routed-trained (recipe) checkpoint there is little to close.** Plain drop with the router (or attribution)
   is already 0.88-0.91 expected cos, and the bound is 0.98. Stale context makes the router selection slightly WORSE
   (0.81, CE +0.007, outlier +0.03): the model co-adapted to not seeing those tokens. Random selection is still biased
   (0.16) and gains little from stale context (0.23) but a lot from fresh context (kvfresh 0.80).
7. **Costs.** Every cheap stand-in costs <= 0.43 of a dense step at this cut (plain drop 0.371, stale everywhere
   0.426: the dropped tokens' forward-only mixer projections in 24 blocks); exact context costs 0.553. None of them
   raises the best-case speedup above plain drop's 2.70x at this cut, so they can only help by letting a training run
   reach dense quality in fewer steps, or by making a more aggressive keep trainable.

**Which deserve a training screen.** Only "stale context in every layer" (best on the dense checkpoint, +15% step
cost), and only where the recipe is weak: the early phase (the gradient-gap early-phase run found the routed gradient
biased until step ~31) and aggressive keeps (5-10%, where the router-routed gradient never stays unbiased). Before
training, the cheapest next step is this probe at keep 5% / 10% (and at the recipe's block-16 cut), to see whether
stale context keeps an aggressive keep's gradient faithful on the early-phase checkpoints (`...-f1952st244-tdgap`,
steps 8-31). Mean stand-ins and fixed linear corrections are not worth a screen.

## Limits

* One cut (block 7, 20% per document). The recipe's second cut (block 16, 7%) is not probed here.
* Frozen checkpoints only: a training screen is what tells whether a stand-in also helps co-adaptation.
* The router is the ridge emulation, not the online router's exact state.
* Adam is the estimated preconditioner (no optimizer state saved), rank 1 per linear.
* Gradients exclude embeddings and the LM head, and keep only the kept tokens' weight terms.
* The linear corrections are fixed least-squares maps, not trained end to end.
* Mean / span stand-ins use the raw mean of block-7 residual states (then each layer's RMSNorm); a learned pooling or
  a mean of keys / values was not tried.
