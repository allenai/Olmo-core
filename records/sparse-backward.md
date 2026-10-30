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
