---
experiment: sparsity_scale
title: Does token-drop + layer-skip headroom grow with model scale? Frozen sweep on the dense fs35 contradiction/oolong SFT checkpoints, Qwen3.5 0.8B/2B/9B/27B
status: active
updated: 2026-10-21
branch: prasann/sparsity-scale (pushed 7fa42b801; code on top of origin/prasann/flopdrop_clean)
launcher: research/token_routing/sparsity_scale/launch_beaker.py
records:
  - records/sparsity-scale-plan.md
results:
artifacts:
  - https://claude.ai/artifact/DeDhCkJLGyMug1cNW5znGs
wandb_group:
next_step: Marker gate passed on all 8 checkpoints. Next: extend gradprobe grad_importance.py (model-size flag, attention skip {0, ½, all} for kept tokens of non-gold docs, per-size analytic FLOPs), smoke at 0.8B, then the sweep (bar + gradient oracle × κ × skip) as 1-GPU Beaker jobs.
---

# sparsity_scale

## Question
On the same task, how much forward compute (tokens dropped + attention blocks skipped) can a bigger
trained model shed before its answer CE moves by more than 0.05 nats? Does that headroom grow from
0.8B to 27B?

## Decisions
- 2026-10-21 (agent): marker gate passed on all 8 checkpoints (|mean ΔCE| ≤ 0.010); two script bugs fixed on the way
  (27B OOM from a fp32 .cuda; an exact-length assert that oolong's line wrapping breaks), commit 656f4cad6.
- 2026-10-21 (Prasann): run the simple version: a frozen sweep on existing trained checkpoints, one
  headline plot (max FLOP cut vs parameter count). Design in `records/sparsity-scale-plan.md`.
- 2026-10-21 (agent): checkpoints = dense fs35 SFT, `fs35s{08b,2b,9b,27b}-contradiction-dense-s56M` and
  `-oolong-dense-s80M` (same recipe at every size). 4B rung skipped for now (not on weka at these
  budgets). Token rule `gold_fl20p8_noslot` (oolong: `fl20p8`), κ ∈ {5,10,20,50%} × attention-block
  skip ∈ {0, ¼, ½}; GDN never skipped. 128 rows/cell at 8k.
- 2026-10-21 (Prasann): 1-GPU jobs are fine if a job fits on one GPU. Every size fits one H200 for
  forward-only CE (27B bf16 ≈ 54 GB), so run one (size, task) per 1-GPU job.
- 2026-10-21 (Prasann): run as 1-GPU BEAKER jobs (weka mounted, no checkpoint transfer). Code goes out on
  its own pushed branch `prasann/sparsity-scale` (one commit on origin/prasann/flopdrop_clean, so other
  sessions' unpushed commits are not published); launched from a scratch worktree at that commit
  because BeakerLaunchConfig refuses an unpushed HEAD.
- 2026-10-21 (Prasann): two token rules: the bar `gold_fl20p8_noslot` (deployable) AND the gradprobe
  oracle gold-first + |grad|@g=0.5 (headroom; uses the answer). oolong has no gold: gradient only.
- 2026-10-21 (agent): attention-block skip settings changed to {0, ½, all} for kept tokens of documents
  not kept whole: the devloss grid found skipping every attention block for those tokens ≈ free at 4B,
  so {0, ¼, ½} could not find the edge.
- 2026-10-21 (agent → Prasann, option 1 chosen): the fs35 dense arms trained on MARKER-FREE shards
  (`taskscale_lengthmix/arms_tokenized/*`; only the kv arms got `_mk` re-tokenizations), while both
  harnesses find documents via markers. Prasann chose: keep the fs35 ladder, use its in-domain eval rows
  (`_eval_bundle_eval500_v3` contradiction n190 = 8k, `_v2_clean` oolong ctx8192) rendered with markers,
  gated by a marker check FIRST (`marker_check.py`: per-row answer CE with vs without markers).

## Current best numbers
Marker gate PASSED on all 8 (2026-10-21): every |mean ΔCE| ≤ 0.010 nats, under the 0.02 bar; medians ≈ 0.
\* 27B is fine-tuned from the post-trained Qwen3.5-27B. ⚠ eval_size = 128 rows per cell.

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| marker check 0.8B | contradiction / 8k | ΔCE (markers − none) +0.0098; CE none 0.137, markers 0.147 | 128 ⚠ | 0.0042 | no-cot (TF CE) | ops result show --experiment sparsity_scale (marker_dCE_mean) |
| marker check 2B | contradiction / 8k | ΔCE (markers − none) +0.0091; CE none 0.047, markers 0.057 | 128 ⚠ | 0.0030 | no-cot (TF CE) | ops result show --experiment sparsity_scale (marker_dCE_mean) |
| marker check 9B | contradiction / 8k | ΔCE (markers − none) +0.0020; CE none 0.020, markers 0.022 | 128 ⚠ | 0.0020 | no-cot (TF CE) | ops result show --experiment sparsity_scale (marker_dCE_mean) |
| marker check 27B* | contradiction / 8k | ΔCE (markers − none) −0.0016; CE none 0.011, markers 0.009 | 128 ⚠ | 0.0021 | no-cot (TF CE) | ops result show --experiment sparsity_scale (marker_dCE_mean) |
| marker check 0.8B | oolong / 8k | ΔCE (markers − none) +0.0006; CE none 0.318, markers 0.319 | 128 ⚠ | 0.0012 | no-cot (TF CE) | ops result show --experiment sparsity_scale (marker_dCE_mean) |
| marker check 2B | oolong / 8k | ΔCE (markers − none) −0.0010; CE none 0.317, markers 0.316 | 128 ⚠ | 0.0015 | no-cot (TF CE) | ops result show --experiment sparsity_scale (marker_dCE_mean) |
| marker check 9B | oolong / 8k | ΔCE (markers − none) +0.0051; CE none 0.256, markers 0.261 | 128 ⚠ | 0.0038 | no-cot (TF CE) | ops result show --experiment sparsity_scale (marker_dCE_mean) |
| marker check 27B* | oolong / 8k | ΔCE (markers − none) −0.0026; CE none 0.215, markers 0.212 | 128 ⚠ | 0.0034 | no-cot (TF CE) | ops result show --experiment sparsity_scale (marker_dCE_mean) |

## Next step
1. ~~Marker check~~ PASSED 2026-10-21 on all 8 (table above).
2. Extend gradprobe `grad_importance.py`: model size flag, attention-skip settings, per-size analytic FLOPs.
3. Sweep: bar + gradient oracle × κ × skip, one 1-GPU Beaker job per (size, task); generation check after.

## Open issues / traps
- Verified 2026-10-21 (peek job 01M5GE87M3CKNKAQKCK4BT73FC): the dense `arms_tokenized/{contradiction_mix_s56M,
  oolong_mix_s80M}` shards contain ZERO box markers, and their rendering is otherwise identical to the kv
  `_mk` rebuild of the same JSONL (cot none, no titles, same prompt text), so `tokenize_example(doc_markers=False)`
  reproduces the training format. One difference: the dense labels also cover the trailing EOS (one extra
  loss token per row); marker_check scores the answer tokens only, the same for both renderings.
  The contradiction eval rows carry no `answers`: the converter renders the gold pairs as the answer.
- gantry/BeakerLaunchConfig: `ops update` takes key=value (final_state=..., notes=...); beaker CLI is not
  installed here, cancel via beaker-py `Beaker.from_env().workload.cancel(...)`.
- 27B is fine-tuned from the post-trained Qwen3.5-27B (no -Base): plot it separately.
- 2B is under-tuned on oolong (shares 0.8B hyperparameters).
- eval_size 128 per cell: flag inline; report paired SE.
- Prior scale-ladder multipliers (records/flop-scaling-ffn-kv-plan.md §(b)) come from a report whose
  CSVs are not at their listed paths; re-derive from `OLMo-core-legacy/results/flop_scaling/evals35/`
  before quoting.

## Verified vs unverified
- Verified 2026-10-21: the 8 checkpoint dirs and the q35-{08b,2b,9b,27b}-base-markerfix bases appear
  in the weka storage index (built 2026-10-12); layer counts 24/24/32/32/64 from
  `src/olmo_core/nn/transformer/config.py`.
- UNVERIFIED: the checkpoints still exist on weka today (index is 9 days old); whether the devloss
  harness's skip-layer option loads non-4B configs without changes.
