---
title: Backward compression with an exact forward (token-sparse / sketched VJPs of the frozen-base LoRA recipe)
date: 2026-10-30
updated: 2026-10-30
experiment: tokdrop_lr
area: token_routing
kind: result
status: in progress (probe smoke done; Beaker probes + training screens launched overnight 10-30)
superseded_by: 
artifact: (pending: the tokdrop_lr hub, https://claude.ai/artifact/SPSBxHFs2PUn5CPjJX3Wgt, gets a section when results land)
summary: Recipe (LoRA-all r64 + two-stage router) with each frozen linear's input-gradient + LoRA VJPs computed only for the token rows of dy that matter. Exact forward. Row-sparse backward brings the step from 0.223 (0.206 with the answer-only head) to 0.108-0.128x dense; deterministic top-k / energy keep cos 0.85-0.99 to the exact LoRA gradient over 8 rows (base ckpt, eval_size 8 probe); the unbiased 1/p sampler diverges (variance compounds over ~300 sequential VJPs). Training screens at f1952 / f3904 in flight.
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
- RESULTS_PLACEHOLDER
