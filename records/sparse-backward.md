---
title: Backward compression with an exact forward (token-sparse / sketched VJPs of the frozen-base LoRA recipe)
date: 2026-10-30
updated: 2026-10-30
experiment: tokdrop_lr
area: token_routing
kind: result
status: negative so far (row-sparse backward fails in training; de2/de4 and some evals pending 10-30 04:00)
superseded_by: 
artifact: (pending: the tokdrop_lr hub, https://claude.ai/artifact/SPSBxHFs2PUn5CPjJX3Wgt, gets a section when results land)
summary: Recipe with each frozen linear's input-gradient + LoRA VJPs computed only for the token rows of dy that matter (exact forward). Cuts the step to 0.110-0.125x dense (recipe 0.204-0.221) and keeps cos 0.85-0.96 to the exact LoRA gradient, but fails in training - contradiction / outlier collapse, 8-cell mean 0.51-0.63 vs 0.88 at f1952; unbiased sampler and residual sketch diverge, truncated SVD is poor. Periodic exact steps (de2/de4) pending.
---

# Backward compression with an exact forward

Branch `claude/sparse-backward` (from `claude/recipe-scale-ablation-71d007`). Code:

- `src/olmo_core/nn/sparse_backward.py`: custom autograd function for every LoRA-adapted frozen linear. Forward is
  bit-identical to the LoRA hook path. Backward modes: `topk` (top k% rows by ||dy_t||), `energy` (fewest rows
  covering e% of sum ||dy_t||^2), `sample` (p_t ~ ||dy_t||, 1/p reweighting, unbiased), `sketch` (top-r SVD + Gaussian
  sketch of the residual, unbiased). `lora_all_rows`: adapter weight gradients from every nonzero row.
- FLOP accounting: each backward call adds `actual - dense` to `model._flop_extra_actual`; `dense` = the real-token
  rows the meter charged (registered per router stage), `actual` = selected rows x (2 in out + 4 r (in+out)) + row
  norms. The answer-only LM head (`--sparse-loss-positions`) is now priced by the router (`charge_sparse_head`)
  instead of being refused by the meter.
- Trainer flags `--st-otr-sbw-{mode,frac,energy,rank,sketch,start,lora-all-rows}`; fast8k arms
  `lora64all32r20to7sl{,sbt<k>,sbe<e>,sbt<k>a,sbe<e>a,sbs<k>,sk<r>}` (`sl` = recipe + answer-only head).
- Probe: `research/token_routing/online_router/sparse_bw_fidelity.py`.

Results: see the final sections (filled in as they land).

## Probe smoke (horton H200, base markerfix + fresh LoRA, mt4s1464 rows 1000-1001 per task)

⚠ eval_size 8 rows only (2 per task): a smoke, not a result. The router-routed exact forward (7:0.2, 16:0.07). FLOPs by
the router's formula (frozen blocks 2/3, adapters, head at the kept length); the router's own overhead is excluded.

Step cost vs a dense step: recipe 0.223, recipe + answer-only head 0.206. The frozen-linear input-gradient + LoRA VJPs
are 0.098 of a dense step, so the floor with the whole VJP removed is ~0.107.

| variant | rows kept (of nonzero-dy rows) | VJP FLOPs kept | step / dense | cos (1 row) | cos (8 rows) |
|---|---|---|---|---|---|
| topk 5% | 5% | 1.0% | 0.108 | 0.45 | 0.61 |
| topk 10% | 10% | 2.4% | 0.110 | 0.66 | 0.86 |
| topk 20% | 20% | 6.8% | 0.114 | 0.85 | 0.95 |
| topk 40% | 40% | 20.7% | 0.128 | 0.94 | 0.99 |
| energy 90% | 1.9% | 0.5% | 0.108 | 0.66 | 0.77 |
| energy 95% | 4.6% | 1.3% | 0.109 | 0.82 | 0.85 |
| energy 99% | 17.9% | 6.1% | 0.115 | 0.93 | 0.95 |
| energy 95% + adapter grads on all rows | 4.6% | 3.1% | 0.111 | 0.83 | 0.87 |
| sample 10 / 20 / 40% (1/p reweighted) | 10 / 20 / 40% | 3 / 9 / 24% | 0.111-0.131 | ~0.00 | ~0.00 (gradient norm 1e12 / 2e8 / 7e3 x exact) |

- The VJP kept is far below the row share: once upper linears drop a row, the dropped tokens of blocks 0-6 (80% of
  the row, which reach the loss only through K/V) often get no gradient at all below, and those rows cost nothing.
- Adapter gradients from all rows ("a") add little: the error comes from the truncated dx propagated down, not
  from the adapter weight-gradient at the layer itself.
- The unbiased 1/p sampler fails at every rate: ~300 sequential sampled VJPs compound the variance (gradient norm
  1e12 x exact at 10%, still 7e3 x at 40%). Not usable without per-layer variance control.

## Training smoke (horton, 2 x H200, 3 steps from the 4B markerfix base on mt4s1464, commit 24196f0b7)

All three arms (recipe + answer-only head `sl`, `sbt20`, `sbe95a`) trained 3 steps with finite loss (CE 1.02 / 1.29-1.35 /
0.96-1.02), peak 21 GiB. On the dense warmup step 1 sbt20 skipped 92% of the VJP FLOPs (0.83 PF saved), sbe95a 98%.
The answer-only head ran on 1.4-1.8% of the kept rows. Caveat seen: the router's ranking target (dL/dh at the drop
layer) is read from the compressed backward, so the router fits a distorted target (rank corr on kept tokens 0.36 /
0.20 / 0.02 at step 3 for sl / sbt20 / sbe95a; early-step noise, but watch it). Follow-up variant if it matters:
compress only blocks below the drop layer (`--st-otr-sbw-end 7`, arm `sbt20e7`),
which keeps the router target exact and still covers ~70% of the VJP FLOPs.

## Probe on trained checkpoints (Beaker, 1 GPU each; held-out fresh rows mt4f7808 rows 4000-4031 per task)

eval_size 128 rows per checkpoint (32 per task; a probe). Checkpoints: base markerfix + fresh LoRA; "mid" = recipe
f1952 final (`f8k-mt4-lora64all32r20to7-f1952-tde-lr1e-05`, 488 steps; the recipe saves no mid-run checkpoint);
"end" = recipe f3904 final (976 steps). Router fit on 24 rows per task (offset 6000). Jobs: sbwfid-{base,mid,end}-{rows,sketch}.

| variant | step/dense | VJP kept | cos 1 row (base / mid / end) | cos 8 rows (base / mid / end) | cos 32 rows (base / mid / end) | rel. sq. error (end) |
|---|---|---|---|---|---|---|
| topk5 | 0.109 | 1.0% | 0.20 / 0.38 / 0.44 | 0.47 / 0.39 / 0.39 | 0.55 / 0.36 / 0.36 | 0.83 |
| topk10 | 0.110 | 2.5% | 0.48 / 0.60 / 0.60 | 0.76 / 0.61 / 0.64 | 0.82 / 0.59 / 0.65 | 0.66 |
| topk20 | 0.114 | 6.9% | 0.74 / 0.83 / 0.78 | 0.90 / 0.85 / 0.85 | 0.94 / 0.86 / 0.86 | 0.40 |
| topk40 | 0.128 | 20.5% | 0.91 / 0.95 / 0.92 | 0.96 / 0.95 / 0.95 | 0.98 / 0.96 / 0.96 | 0.15 |
| energy90 | 0.109 | 1.1% | 0.65 / 0.70 / 0.76 | 0.77 / 0.64 / 0.73 | 0.80 / 0.61 / 0.74 | 0.45 |
| energy95 | 0.111 | 2.5% | 0.79 / 0.82 / 0.86 | 0.85 / 0.80 / 0.84 | 0.88 / 0.78 / 0.84 | 0.28 |
| energy99 | 0.123 | 13.1% | 0.93 / 0.96 / 0.96 | 0.95 / 0.95 / 0.96 | 0.96 / 0.95 / 0.96 | 0.07 |
| topk5L | 0.110 | 2.8% | 0.24 / 0.43 / 0.48 | 0.48 / 0.43 / 0.44 | 0.55 / 0.41 / 0.41 | 0.78 |
| topk10L | 0.112 | 4.5% | 0.48 / 0.63 / 0.63 | 0.76 / 0.64 / 0.67 | 0.82 / 0.61 / 0.68 | 0.63 |
| topk20L | 0.117 | 9.2% | 0.74 / 0.85 / 0.81 | 0.89 / 0.86 / 0.86 | 0.93 / 0.87 / 0.87 | 0.36 |
| energy90L | 0.111 | 2.9% | 0.67 / 0.73 / 0.78 | 0.79 / 0.67 / 0.76 | 0.82 / 0.63 / 0.76 | 0.41 |
| energy95L | 0.113 | 4.5% | 0.80 / 0.84 / 0.88 | 0.87 / 0.82 / 0.86 | 0.89 / 0.80 / 0.86 | 0.24 |
| sample10 | 0.111 | 3.3% | 0.00 / 0.00 / 0.00 | -0.00 / -0.00 / 0.00 | -0.00 / -0.00 / -0.00 | 7868210023716712611840.00 |
| sample20 | 0.117 | 9.0% | -0.00 / 0.00 / 0.00 | -0.00 / 0.00 / -0.00 | -0.01 / 0.00 / -0.00 | 983882103239187.50 |
| sample40 | 0.131 | 24.0% | 0.00 / 0.00 / 0.00 | -0.00 / 0.00 / 0.00 | 0.00 / 0.00 / -0.00 | 3204264.62 |
| svd128 | 0.121 | 13.2% | 0.12 / 0.21 / 0.28 | 0.11 / 0.16 / 0.17 | 0.11 / 0.15 / 0.15 | 0.94 |
| svd256 | 0.127 | 19.4% | 0.15 / 0.26 / 0.34 | 0.14 / 0.23 / 0.23 | 0.14 / 0.21 / 0.21 | 0.90 |
| svd512 | 0.139 | 32.0% | 0.21 / 0.37 / 0.42 | 0.18 / 0.33 / 0.35 | 0.18 / 0.31 / 0.33 | 0.84 |
| sk128 | 0.119 | 10.4% | nan / nan / nan | nan / nan / nan | nan / nan / nan | nan |
| sk256 | 0.149 | 39.4% | nan / nan / nan | nan / nan / nan | nan / nan / nan | nan |
| sk512 | 0.195 | 81.0% | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | inf |

- Row-sparse top-k / energy: fidelity is stable across checkpoints. topk40 / energy99 keep cos >= 0.95 at 1, 8 and 32
  rows for a step of 0.128 / 0.123 x dense; topk20 / energy95 ~0.85 at 0.114 / 0.111.
- Unbiased variants diverge: the 1/p sampler (gradient norm 4e10 / 2e7 / 9e2 x exact at 10 / 20 / 40%) and the
  Gaussian residual sketch (overflows to inf/NaN in bf16) both compound variance over ~300 sequential VJPs.
- Truncated SVD of the frozen weights is poor (cos 0.15-0.42 even at r = 512): the spectra are flat.

## Training (Beaker jupiter, 2 GPUs, LR 1e-5, eval 8k + 2k rungs x 4 tasks, eval_size 500 each, no-cot)

- Equivalence control: `sbt100` (top-k at 100%, i.e. the custom backward computing every row) tracks the recipe
  step for step over 40 steps (CE 1.331/1.369/1.171/0.648 vs 1.331/1.368/1.171/0.651; steps 10-40 mean 0.307 vs
  0.319), so the FSDP2 + activation-checkpointing path is correct and every deficit below is the compression.
- Training CE (routed, answer tokens; same data order for seed 0): the recipe falls 0.32 -> 0.20 -> 0.11 (steps 10-40 /
  50-100 / 200-488); every compressed arm stalls at 0.35-0.40 from ~step 40 (sbt20, sbe95a, and also the
  high-fidelity sbt40 / sbe99); sbt20e7 (blocks 0-6 only) reaches 0.21.
Results so far (03:55 10-30; 8-cell mean = 4 tasks x {8k, 2k}, every cell eval_size 500, no-cot, binomial SE per
cell 0.005-0.02; step/dense = the finished run's `actual_over_dense` from flops.json, router overhead included):

| arm (mt4, LR 1e-5) | rows | step / dense | train CE last 20 | 8-cell mean | contra 8k/2k | nq 8k/2k | outlier 8k/2k | oolong 8k/2k |
|---|---|---|---|---|---|---|---|---|
| recipe (tde, seed 0, reference) | f1952 | 0.221 | - | 0.883 | 0.959/0.977 | 0.912/0.970 | 0.862/0.880 | 0.648/0.856 |
| recipe + answer-only head `sl` sd1 | f1952 | 0.204 | 0.078 | 0.880 | 0.948/0.970 | 0.914/0.972 | 0.865/0.874 | 0.641/0.856 |
| sbt20 | f1952 | 0.113 | 0.333 | 0.514 | 0.173/0.279 | 0.832/0.930 | 0.203/0.284 | 0.567/0.840 |
| sbt20 sd1 | f1952 | 0.113 | 0.248 | 0.630 | 0.646/0.729 | 0.776/0.928 | 0.227/0.303 | 0.594/0.835 |
| sbe95a | f1952 | 0.110 | 0.310 | 0.534 | 0.131/0.192 | 0.852/0.942 | 0.185/0.594 | 0.564/0.811 |
| sbe95a sd1 | f1952 | 0.110 | 0.298 | 0.514 | 0.165/0.261 | 0.824/0.936 | 0.221/0.263 | 0.599/0.840 |
| sbt20e7 (blocks 0-6 only) | f1952 | 0.147 | 0.176 | 0.614 | 0.557/0.709 | 0.830/0.932 | 0.159/0.262 | 0.615/0.848 |
| sbt40 | f1952 | 0.125 | 0.145 | pending (4/8) | pending | 0.870/0.952 | pending | 0.582/0.831 |
| sbe99 | f1952 | 0.116 | 0.151 | pending (4/8) | pending | 0.914/0.960 | pending | 0.582/0.819 |
| sbt20, exact every 2nd step (de2) | f1952 | 0.158 | 0.110 | pending | | | | |
| sbt20, exact every 4th step (de4) | f1952 | 0.136 | 0.152 | pending | | | | |
| recipe (tde, reference) | f3904 | 0.221 | - | 0.915 | 0.977/0.988 | 0.918/0.976 | 0.970/0.972 | 0.663/0.858 |
| sbt20 | f3904 | 0.113 | 0.195 | pending (6/8) | 0.798/0.823 | 0.834/0.922 | pending | 0.612/0.829 |
| sbe95a | f3904 | 0.110 | 0.236 | pending | | | | |

**Verdict so far: the row-sparse backward does not survive training.** It cuts the step to 0.110-0.125x dense as
designed (1.6-1.9x cheaper than the recipe, measured), and per-step it keeps cos 0.85-0.96 to the exact LoRA gradient,
but the deterministic truncation is the same every step: training CE stalls at 0.3 (recipe 0.08), and the tasks that
need the early layers to re-encode context tokens collapse (contradiction 0.96 -> 0.13-0.65, outlier 0.86 -> 0.16-0.23
at 8k), while nq / oolong lose 0.03-0.08. Matched-data deficit at f1952 is 0.25-0.37 on the 8-cell mean, and at f3904
sbt20 is already 0.18 behind on contradiction 8k: the kill criterion (> 0.01 at 3,904 rows) is hit. Higher fidelity
(sbt40 / sbe99, cos 0.95-0.96) trains much better (CE 0.15) but still loses 0.07 on oolong 8k (beyond the 0.021 SE).
Periodic exact steps (de2 / de4) are the open thread: de2 trains to CE 0.11 at 0.158x.

- Per-step gradient fidelity did not predict training here either: cos 0.85 per 32 rows (sbt20) collapses
  contradiction, the unbiased estimators that would fix the bias diverge, and a 0.96-cosine variant still loses.
- Free exact saving found on the way: the custom backward skips rows whose dy is exactly zero (`sbt100`: identical
  gradients, 0.206 vs 0.213 step over 40 steps), ~3%.
