---
experiment: lengen_rope_hash
title: Length generalization from 32k setA SFT to 64k/128k — RoPE reset per document/query vs hashed doc IDs vs a full-attention base (Qwen3-4B-Base)
status: done
updated: 2026-10-13
branch: prasann/lengen (OLMo-core worktree projects/OLMo-core-lengen); olmo-eval prasann/ctc-suite-lengen (worktree projects/olmo-eval-lengen)
launcher: recipes/train/setA/launch_setA_sft.py
records:
results:
wandb_group:
next_step: Done for the long-rung question (Prasann 2026-10-19: neither trick generalizes better; long evals cancelled). Optional: eval_size-500 rerun of contradiction r128k (base/rope/hash); short-rung (<=32k) bins + Qwen3 <=32k still queued on Beaker as sanity — cancel if not needed.
---

# lengen_rope_hash

## Question
Trained on setA at ≤32k (canonical recipe), which change length-generalizes better to the r64k/r128k
CTC rungs: (a) RoPE positions restarted at every document and at the trailing query, or (b)
document IDs as random per-example 5-digit IDs instead of 1..N? Plus (c): the same recipe on a
full-attention base (Qwen3-4B-Base) with neither change. Baseline = the existing dense 32k run.

## Arms
| arm | base | change | run |
|---|---|---|---|
| baseline | q35-4b-dense-256k-fix/step2385 | none | `q35-4b-dense-ctc-setA-32k-20260923T1537/step888` (exists); re-eval `lengen-baseline-step888-p2` (jupiter) + `-ceres` duplicate QUEUED |
| (a) rope-reset | same | `--variant rope-reset` | `q35-4b-rope-reset-ctc-setA-32k-20261013T2145` ([Beaker](https://beaker.org/ex/01M4WBPPGP625FT5CY5ZF4YDE0)) — DONE step888 (preempted mid-save at step300; resumed from step200 via --resume-overwrite, ef0fabf51). Eval `lengen-ropereset-step888` QUEUED |
| (b) hash-ID | same | `--data-root .../setA_max20_evaliid_hashid --shards-dir shards_qwen35_q35le32k --window 65536` (baseline's exact 203,156 examples, hashed; +4.1% tokens; max 34,411) | `q35-4b-hashid-ctc-setA-32k-20261013T2202` ([Beaker](https://beaker.org/ex/01M4WCNDTK0P6AKGXEMX27DNXS), 375b6bb83, 922 steps) — DONE step922. Eval `lengen-hashid-step922` (--doc-id-style hash) QUEUED |
| (c) full-attn | `prasanns/ctc_suite/bases/qwen3-4b-base-trainedmark` (marker audit PASS) | `--model qwen3-4b`; shards `shards_qwen3_q35le32k` (baseline's exact 203,156 examples; q3/q35 tokens 1.002; max 33,002), window 65,536 | `q3-4b-dense-ctc-setA-32k-20261013T2244` ([Beaker](https://beaker.org/ex/01M4WF16GXA34WYCV01AHE06A7), f66c6d624, 886 steps) — DONE step886. Evals `lengen-qwen3-step886-{le32k,r64k-yarn2,r128k-yarn4}` QUEUED |

## Decisions
- 2026-10-13 (Prasann): 3 SFT runs (a, b, c); baseline reused. Eval r64k/r128k (+ r2k-r32k sanity).
- 2026-10-13 (Prasann): rope-reset trailing query + answer restart at P like a document (P = the example's first `<|box_start|>` position); preamble keeps 0..P-1.
- 2026-10-13 (Prasann): hashed IDs = distinct random ints in [10000, 99999] per example, deterministically seeded.
- 2026-10-13 (Prasann): full-attention base = Qwen3-4B-Base (native 32k; needs YaRN at r64k/r128k at eval — a known confound).
- 2026-10-13 (Prasann): eval sizes = launcher default (100 at r64k/r128k) — every long-rung cell is sub-500 and must be flagged.
- 2026-10-13 (agent): prasann/lengen sits on upstream v3.0.0 (torch>=2.10 pin). Training installs it --no-deps with the baseline's resolved env (torch 2.9.1, fla 0.4.1; bd3832cc8); evals use olmo-eval `--olmo-core-ref prasann/lengen --olmo-core-no-deps` for ALL four checkpoints incl. the baseline re-eval, so every arm is scored on one stack.
- 2026-10-13 (agent): dense smoke on prasann/lengen reproduces the baseline: CE step1 1.327 / step10 0.4278 vs baseline 1.327 / 0.4277. rope-reset smoke: 1.313 / 0.441 / step20 0.380 (baseline step20 0.346).
- 2026-10-13 (agent): hash arm keeps the baseline's 32k window; hashed IDs lengthen examples (+1.9-9.3% tokens/task), so a few hundred near-32k examples newly exceed the cap and are dropped (exact count from the re-verify job). The converter's 262,144 cap also drops ~135 r256k rows (irrelevant at 32k). rerank targets get 1.3-2x longer.
- 2026-10-13 (agent): Qwen3 arm: nothink tokenizer copy (`prasanns/hf_tokenizers/Qwen3-4B-Base-nothink`) is required at eval (stock template leaves the prompt one think block short: 0/96 prefix parity vs 96/96); eval needs `--eos-token-id 151645 --pad-token-id 151643 --yarn-old-len 32768` (olmo-eval 2af3d960) and YaRN x2 at r64k, x4 at r128k (Qwen3 prompts run ~0-3% past factor x 32,768).
- 2026-10-13 (agent): hash arm + Qwen3 arm train on EXACTLY the baseline's 203,156 pieces (kept iff baseline Qwen3.5 piece length <= 32,768; per-task counts match), at a 65,536 window (tokens/step unchanged), instead of re-applying the 32k cap (which would drop ~1k of the hash arm's longest examples, 546 in qdmatch_nq). Shipped rerank shard holds 25 ~256k rows the current converter rejects (all >32k, irrelevant).
- 2026-10-13 (agent): eval plan per checkpoint: 12 setA rows, policy r2k-r32k:300, r64k:100, r128k:100, no r256k; `--olmo-core-ref prasann/lengen --olmo-core-no-deps --doc-markers`; hash arm adds `--doc-id-style hash`; Qwen3 adds nothink tokenizer + eos/pad + `--yarn-old-len 32768`, YaRN x2 at r64k / x4 at r128k as separate launches. Rows without r64k/r128k data (reorder, grouping, xabsence, strmatch, textgroups) are scored at <=32k only.
- 2026-10-13 (agent): work in separate worktrees/branches to avoid clobbering other sessions. rope-reset is a model-config field (`TransformerConfig.rope_reset`), so the checkpoint's config.json carries it to the olmo-eval native provider (generation module computes positions for prefill chunks and decode).

## Current best numbers
Long rungs, all four checkpoints scored on one stack (olmo-eval `prasann/ctc-suite-lengen`, olmo_core flash_3, OLMo-core `prasann/lengen` --no-deps, horton H200; Beaker H100 baseline agrees: contradiction 0.760/0.350). ⚠ eval_size=100 per cell (±~0.04-0.05). no-cot. Recorded with `ops result add --check` (runs `lengen-local-*`, `lengen-*-g1`).

| task (metric) | r64k base | r64k rope | r64k hash | r64k qwen3 (YaRN x2) | r128k base | r128k rope | r128k hash | r128k qwen3 (YaRN x4) |
|---|---|---|---|---|---|---|---|---|
| contradiction f1 | 0.757 | 0.700 | 0.743 | 0.556 | 0.350 | 0.425 | 0.447 | 0.093 |
| hotpotqa f1 | 0.870 | 0.865 | 0.850 | 0.835 | 0.795 | 0.817 | 0.800 | 0.749 |
| nq f1 | 0.920 | 0.940 | 0.920 | 0.910 | 0.910 | 0.910 | 0.880 | 0.875 |
| qdmatch_nq f1 | 0.340 | 0.363 | 0.353 | 0.337 | 0.117 | 0.143 | 0.130 | 0.087 |
| outlier f1 | 0.393 | 0.353 | 0.316 | 0.457 | 0.097 | 0.083 | 0.073 | 0.100 |
| oolong score | 0.663 | 0.601 | 0.589 | 0.633 | 0.561 | 0.552 | 0.476 | 0.485 |
| rerank ce_pos_recall | 0.927 | 0.951 | 0.942 | 0.806 | 0.861 | 0.833 | 0.917 | 0.646 |
| mean | 0.696 | 0.682 | 0.673 | 0.648 | 0.527 | 0.538 | 0.532 | 0.434 |

Verdict (2026-10-19): neither rope-reset nor hashed IDs improves 64k/128k generalization overall (means within ±0.02 of baseline = noise). Only hint: contradiction r128k 0.35 -> 0.43/0.45 in both arms (~1.2-1.4 sigma; would need eval_size 500 to confirm). Full-attention Qwen3-4B-Base is clearly worse at 128k (confounded: native 32k + YaRN x4 vs 256k-CPT base).

## Next step
rope-reset run training (~3.2h). Launch hash-ID run after re-verify + prep; Qwen3 run after shards + prep + smoke; eval-smoke the rope-reset step20 checkpoint; then evals (r2k-r128k) for all 4 checkpoints.

## Open issues / traps
- Eval jobs need `--min-runtime 2h` on Beaker (minRuntime 0 = preempted within minutes; up to 56 attempts/job). The <=32k bs=8 bin OOMs on H100 (use --batch-size 4).
- lorax: slurm GRES says 7 GPUs, node has 8; an unmanaged GPU (PCI 41:00, 63.6 GB used) leaks into every job and CUDA init fails (cuInit=3). Admin issue; not drained.
- berkeleynlp is debugging-only at --qos=normal since 2026-10-16 (ops hook).
- Training runs from `prasann/lengen` (off flopdrop_clean, which owes GPU validation); the dense smoke must reproduce the baseline's first steps.
- Qwen3.5-4B has RoPE only on its 8 full-attention layers (25% partial rotary); GDN layers have no positions, so rope-reset changes less than on a pure-attention model.
- Arm (c) differs from the baseline in family, tokenizer and long-context pretraining (32k native + YaRN at eval vs a 256k-CPT base) — it is not a clean GDN-vs-attention test.
- Longer CTC rungs have more documents (larger n), not just longer ones.
- Beaker jupiter was saturated overnight: 1-GPU urgent evals pended 2h+, training was preempted 3x (rope-reset at step~300 mid-save -> FileExistsError on retry; fixed by --resume-overwrite). lorax CUDA init is broken (cuInit=3), local evals use horton.
- Fixed in passing (8357f40d9): the explicit position_ids RoPE path sized its sin/cos cache from the first call and indexed out of bounds later.

## Verified vs unverified
- Verified 2026-10-13: rope-reset unit/GPU tests (141 passed on horton); check_variant_parity vs 187104b18 passes (only new key rope_reset=None).
- UNVERIFIED: everything about training/eval outcomes.
