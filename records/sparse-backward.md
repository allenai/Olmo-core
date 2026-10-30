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
