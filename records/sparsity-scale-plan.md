---
title: Plan — does token-drop + layer-skip headroom grow with model scale? (frozen trained checkpoints, 0.8B-27B)
date: 2026-10-21
updated: 2026-10-21
experiment: sparsity_scale
area: token_routing
kind: plan
status: current
superseded_by:
artifact: https://claude.ai/artifact/DeDhCkJLGyMug1cNW5znGs
summary: Frozen sweep of token keep × attention-block skip on the dense fs35 contradiction/oolong SFT checkpoints at 0.8B/2B/9B/27B; headline = max FLOP cut within 0.05 nats ΔCE vs parameter count.
---
# Plan — sparsity headroom vs model scale

Nothing here is launched.

## Question

On the same task, how much forward compute can a bigger model shed (tokens dropped + layers skipped)
before its answer CE moves? If the headroom grows with scale, that motivates a trained dense vs
token-drop comparison at 2-3 sizes; if it is flat, the scale question is closed cheaply.

Prior hint (trained, separate methods, fixed operating points; from `records/flop-scaling-ffn-kv-plan.md`
§(b), not yet re-derived from the eval JSONs): on contradiction the KV gold+1/3 FLOP multiplier vs dense
went 0.87 (2B) → 1.00 (4B) → 1.66 (9B), 1.59 at 27B*; routed FFN 0.76 → 0.81 → 0.91, then 0.67 at
27B*. On oolong, KV 1/6 wins at 0.8B/2B and loses by 9B.

## Design (Prasann approved the simple version, 2026-10-21)

- **Checkpoints (trained, frozen):** the dense fs35 SFT checkpoints, one recipe across sizes, all on weka
  under `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts/`:
  - `fs35s{08b,2b,9b,27b}-contradiction-dense-s56M`
  - `fs35s{08b,2b,9b,27b}-oolong-dense-s80M`
  - 4B is missing at these budgets (only `fs35-contradiction-dense-s7M` / `fs35-oolong-dense-s10M`
    exist); optional: two 4B retrains with the fs35 recipe to fill the gap.
- **Rows:** 128 eval rows per cell at 8k (⚠ eval_size<500; report paired SE).
- **Sweep (no training):** token keep κ ∈ {5, 10, 20, 50%} × attention-block skip ∈ {0, ¼, ½ of the
  full-attention blocks}, plus κ = 100% and skip = 0 as the full-context reference.
  - Token rule: `gold_fl20p8_noslot` (the devloss-grid bar). oolong has no gold docs, so it is `fl20p8`.
  - Skip is a fraction of depth because layer counts differ (24 / 24 / 32 / 64 at 0.8B / 2B / 9B / 27B).
    GDN blocks are never skipped (not skippable at 4B: devloss_grid skipgdn2).
- **Metric:** per model, the largest FLOP reduction (analytic FLOPs at that size and length) whose
  median ΔCE vs full context is ≤ 0.05 nats. Also report ΔCE as a fraction of the context benefit,
  ΔCE / (CE_no-context − CE_full), because bigger models reach lower CE.
- **Headline plot:** max FLOP cut vs parameter count, one line per task.
- **Generation check:** at each size's winning point, one generation-graded eval at eval_size 500
  (CE parity does not imply right answers).

## Where / cost

One 1-GPU job per (size, task), 8 jobs in all (every size fits one H200 for forward;
27B bf16 ≈ 54 GB); about half a GPU-day in total on jsteinhardt H200s (not berkeleynlp: debug only). Checkpoints go
weka → S3 → node `/data` as bf16, ~200 GB in total (27B dominates; it fits one H200 for forward).
Harness: `research/token_routing/devloss_grid/ctc_devloss_grid.py` (keep schemes + skip layers).

## Caveats

- 27B is fine-tuned from the post-trained Qwen3.5-27B (no -Base release): plot it as a separate marker.
- 2B shares the 0.8B hyperparameters and is under-tuned on oolong (ctc_modelscale).
- One high-CTC (contradiction) and one low-CTC (oolong) task only.
