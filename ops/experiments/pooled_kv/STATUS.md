---
experiment: pooled_kv
title: Pooled-doc-KV / soft-token compressed training with full-attention inference (contradiction 32k) + eval-side soft-KV slot-construction probes
status: done
updated: 2026-09-15
branch: prasann/landmark
launcher: research/token_routing/pooledkv_train/run_q4b_pooledkv_horton.sbatch
records:
  - records/pooled-doc-kv-handoff.md
  - records/pooled-doc-kv-attention.md
  - records/soft-kv-slot-probe-handoff.md
  - records/layer-soft-probe.md
  - records/landmark-slot-probe.md
  - records/outlier-slot-probe.md
  - records/outlier-richer-slot-probe.md
  - records/outlier-cc00-evaltime-probe.md
  - records/outlier-realtoken-parity-probe.md
  - records/outlier-saliency-preview-probe.md
results:
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/results/pooled_kv/slot_probe_results.csv
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/results/pooled_kv/slot_tracker.html
  - research/token_routing/pooled_kv/
wandb_group:
next_step: None here. Superseded by ds64 (trained soft-token arms on 2k-64k ladders); the 64k/128k pooled-KV length campaign and the learned null-expert FFN router were never run in this form
---

# pooled_kv

## Question
Can a model be trained on compressed contexts, where most documents collapse to one pooled KV slot or
soft token, and still transfer zero-shot to ordinary full attention at eval? Can it do so without any
full-context forward in training? A second, eval-side question: which slot construction lets a
dense-trained model read pooled documents at answer-CE parity?

## Decisions
- 2026-08-27 (agent): `PooledDocKVAttention` (mean-pooled post-RoPE K/V + `log(doc_len)` bias), exactly equivalent to full attention over a perturbed corpus. Flex block-sparse is no faster than cuDNN, so v1 is a transfer experiment.
- 2026-08-29 (agent): zero-shot transfer collapsed (0.156 vs 0.985). Mechanism: CO-DRIFT, where compressed training drags the full-attention pathway (L16-L28).
- 2026-08-30 (agent): paired consistency distillation fixed transfer (0.933 at 6k). gold_subsample made the task unsolvable (Prasann caught it) → `gold_plus_random`.
- **2026-10-16 (Prasann): the 2026-08-31 rule below is REVERSED.** Full-context forwards in training are allowed again (`memory:no-full-attention-32k-training`).
- **2026-08-31 (Prasann, HARD RULE): never run a full-attention forward at 32k in training.** No distill teachers and no full-row mixing, even if they help. Full baseline and full-attention eval are allowed. See `memory:no-full-attention-32k-training`. Distill arms v17-v19 were cancelled.
- 2026-08-31 (agent): breadth of kept negatives is the protection mechanism. v25 (`--n-random-range 128,512`, softtoken B1, `--detach-soft-kv`, no distill) reaches parity.
  - Closed: oracle slot cache (v23 hurt) and role-blind FFN gate (v24/v26 dead).
- 2026-09-08 (Prasann / agent): eval-side slot probe. Keep each doc's HEADER real (exact-match ids) and pool the body: parity on all four tasks. Wired in as `--st-header-stop-id 25 --st-header-stop-count {1|3}` (aa0b7db2a) and `--st-neighbour-runs 1` (bb71f1f46).
- 2026-09-08 evening (Prasann): slots play no role in training. The GDN write detach was added, and the work moved to the ds64 campaign.
- 2026-09-14/15 (agent): eval-only follow-up probes.
  - Landmark slot: NO-GO.
  - Soft-every-layer: no fidelity gain, ~2 points of FLOPs.
  - Outlier plain mean slot: unreadable (hard limit). `cent_cmean` is oracle-readable. First-k real tokens give parity. These fed ds64.

## Current best numbers
| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| full-attention baseline (3-epoch) | contradiction / 32k | f1=0.939 | 488 (whole held-out file) | 0.011 | no-cot | horton pooledkv_logs/eval_3485312.log |
| **v25 softtoken rb128-512, zero full-attn (~2.8x train TPS)** | contradiction / 32k | f1=0.927 (parity) | 488 | 0.012 | no-cot | horton pooledkv_logs/eval_3487685.log |
| v22 n_random=256 (~3.8x) | contradiction / 32k | f1=0.916 | 488 | 0.013 | no-cot | eval_3487481.log |
| v21 rb16-256 (~7.3x) | contradiction / 32k | f1=0.857 | 488 | 0.016 | no-cot | eval_3487459.log |
| full baseline at 1/3 budget | contradiction / 32k | f1=0.542 | 488 | 0.023 | no-cot | eval_3487451.log |
| v23 oracle slots / v26 FFN gate@12 | contradiction / 32k | f1=0.809 / 0.316 (closed axes) | 488 | 0.018 / 0.021 | no-cot | eval_3487528 / 3487694.log |
| eval-side header-real slot, keep 1/3 | contradiction / 32k | answer CE 0.064 vs full 0.071 | 24 rows ⚠ eval_size=24 | n/a | no-cot (TF CE) | results/pooled_kv/slot_probe_results.csv |

The 488-example contradiction held-out set is the whole file and is accepted as-is (CLAUDE.md).

## Next step
None in this experiment. The open threads in the 08-31 handoff were:
- 64k/128k length campaign, where fixed absolute breadth projects to ~4-5x at 64k and ~7-10x at 128k;
- a learned null-expert FFN router.

Neither was run as pooled-KV. The soft-token line continued as `ds64` (2k-64k ladders, header-real /
first-k / cent_cmean). The FFN line continued as `ffndrop`, and earlier as nested-ffn-moe and flop_scaling.

## Open issues / traps
- The 2026-08-31 HARD RULE was REVERSED on 2026-10-16. Distill / full-row-mixing arms are allowed again; charge their FLOPs (`memory:no-full-attention-32k-training`).
- The softtoken arm refuses to run on an un-baked base. Run `bake_projector_into_base.py` first (`q4b-dense-cpt-fixmark-b1`).
- `--detach-soft-kv` with aux weight 0 freezes the projector at identity. That is the working configuration; do not "fix" it.
- Sneetches probe rows ≠ Beaker eval-bundle rows (oolong full 0.507 vs 0.466). Never compare across sources.
- The plain mean slot is NOT at parity: contra EM 0.42 vs 0.86 at keep 1/12. Parity numbers are header-real numbers.
- The eval driver hardcodes `query_position="both"`, so the shard must be qboth. MAXLEN below the prompt length silently zeroes f1.
- mooney /data filled with fp32 optimizer checkpoints. v15/v16 SIGSEGVs were node hardware.

## Verified vs unverified
- Verified 2026-10-12 from the eval logs at `/net/horton/data/prasann/from_scratch/pooledkv_logs/` (read-only, dated 2026-08-31): v20 0.797, v21 0.857, v22 0.916, v23 0.809, v25 0.927, v26 0.316, 1/3-budget baseline 0.542, baseline 0.939. All have eval_size 488.
- Verified: the contradiction header-real row in `results/pooled_kv/slot_probe_results.csv` (1640 probe rows).
- UNVERIFIED: the train-speedup multipliers (2.8x / 3.8x / 7.3x) come only from the handoff record.
- UNVERIFIED — confirm with Prasann: should `done` be `abandoned` for the 64k/128k pooled-KV campaign, or does ds64 count as its completion?
