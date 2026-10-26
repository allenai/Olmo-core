---
title: Result — OLMo 3 7B long-context CPT: token dropping loses, whole-model LoRA wins at 0.70x FLOPs
date: 2026-10-25
updated: 2026-10-25
experiment: olmo3_cptdrop
area: token_routing
kind: result
status: current (K32 at 960 steps pending)
superseded_by:
artifact: https://claude.ai/artifact/4tZKEXtpSpg5ax6B1YeoLk
summary: OLMo 3 7B stage-2 → 32k/64k CPT on Longmino. Every token-drop / router / MoD / partial-MLP / backprop-stop arm loses to dense; whole-model LoRA r64 (frozen base) beats dense at 0.70x FLOPs/step and 29% less memory, m 1.7–3.9 by budget, edge ~1.5x larger at 64k.
---
# Result — OLMo 3 7B long-context CPT: what saves FLOPs

Plan: [`olmo3-cptdrop-screen-plan.md`](olmo3-cptdrop-screen-plan.md). Live state: `ops/experiments/olmo3_cptdrop/STATUS.md`.
Code: `research/token_routing/olmo3_cptdrop/`. Every number below is in the ops results log (`ops result show --experiment olmo3_cptdrop`).

## Verdict

- **Recipe: whole-model LoRA ("K32").** Freeze the embeddings and every block, and train rank-64 adapters (scale 32) on
  every linear in all 32 blocks, at LR 1e-5 to 2e-5, 16 rows/step, no token dropping.
  - Cost: 0.700x dense FLOPs/step (measured by the FLOP meter); peak 25.0 vs 35.2 GiB per GPU.
  - Quality: below dense at the same step count at every budget tested, at 32k and 64k.
- **The win is real but modest:** FLOPs-to-match (m) peaks at about 3.9x near 48–120 steps and decays to about 1.8x at
  480 steps. That is well short of the 10–20x target. At large budgets most of the value is the cheaper step and the
  lower memory.
- **The edge grows with context length:** at 64k it is about 1.5x the 32k gap at every budget.
- **Everything that removes work per token loses, alone or stacked on K32:**
  - token drops, slots and routers;
  - mixture-of-depths;
  - partial MLP (random or top-k, frozen or trainable base);
  - an adapter-less frozen bottom stack;
  - smaller batches (a tie).

## Setup

- **Base and data:**
  - Start: OLMo 3 7B `stage2-step47684` (8k, YaRN x4 for 32k, x8 for 64k).
  - Train: Dolma 3 Longmino pool, 2^15 / 2^16 buckets, fresh rows only (single epoch).
  - Batch: 524,288 tokens/step (16 x 32k or 8 x 64k), WSD schedule, peak LR 2e-5 (the LR probe beat the recipe's 2.07e-4).
- **Metric:** full-context dev CE.
  - `dev32k_1024`: 1,024 rows from 2^17+ documents, never in the stage-3 mix. `dev64k_512`: 512 rows.
  - Paired per-row SE on deltas; seed noise on overall CE is ±0.0001 (K16 and K32 seed replicates).
  - Short-document guard `short4k_v2`: 256 rows ⚠ eval_size=256 only (relative forgetting check).
- **m** = PF dense needs to reach the arm's CE ÷ the arm's PF. Both come from the measured FLOP meter; dense is ~27.5 PF/step
  plus a ~27 PF constant. Read piecewise log-linear between measured dense points; beyond the last point it is a bound.
- **The meter does not charge activation-checkpoint recompute.** It counts forward + backward (≈ 6N per token plus
  attention). Every run here uses full AC, which recomputes each block's forward once more. True cost is then ≈ 8N per token
  for dense and ≈ 6N for K32 (no base weight grads), so **K32 is ≈ 0.75× dense in real FLOPs, not the metered 0.70×.**
  The m values below use the metered convention for both arms; with recompute charged, K32's m shrinks by ~7%.

## Dense reference (32k, dev32k_1024)

| steps | 12 | 24 | 48 | 120 | 240 | 480 | 960 |
|---|---|---|---|---|---|---|---|
| PF | 356 | 685 | 1,344 | 3,318 | 6,606 | 13,188 | 26,349 |
| CE | 1.6972 | 1.6912 | 1.6880 | 1.6853 | 1.6827 | 1.6800 | 1.6762 |

## LoRA depth: freezing more of the stack is cheaper and better

"K" = blocks [0, K) frozen behind rank-64 adapters; blocks K..31 train fully.

| K | FLOPs/step | peak GiB | CE @48 | CE @120 | Δ vs dense @48 / @120 |
|---|---|---|---|---|---|
| dense | 1.000x | 35.2 | 1.6880 | 1.6853 | — |
| 16 | 0.849x | 30.9 | 1.6864 | 1.6834 | −0.0015 / −0.0019 |
| 24 | 0.774x | 28.7 | 1.6856 | 1.6827 | −0.0023 / −0.0026 |
| 28 | 0.737x | — | 1.6854 | 1.6824 | −0.0025 / −0.0029 |
| 30 | 0.721x | — | 1.6852 | 1.6823 | −0.0028 / −0.0031 |
| **32** | **0.700x** | **25.0** | **1.6849** | **1.6819** | **−0.0031 / −0.0034** |

Further checks:
- K32 seed 1 at 120 steps: 1.6821, reproducing seed 0.
- LR 1e-5 at 120: 1.6820, a tie with 2e-5. LR 4e-5 at 120: 1.6843, worse.
- Rank 128 on K16 (scale 32) is no better than r64. A K32 r128 run at scale 64 doubled the adapter LR (scale is a direct
  multiplier, not alpha/r), so it reads as an LR result, not a rank result.

## K32 budget curve

| steps | 12 | 24 | 48 | 120 | 240 | 480 |
|---|---|---|---|---|---|---|
| 32k CE | 1.6926 | 1.6880 | 1.6849 | 1.6819 | 1.6804 | 1.6788 |
| 32k Δ vs dense | −0.0046 | −0.0032 | −0.0031 | −0.0034 | −0.0023 | −0.0012 |
| 32k m | 1.65 | 2.75 | 3.9 | 3.45 | 2.5 | ~1.8 |
| 64k Δ vs dense (dev64k_512, 512 rows) | | | | −0.0049 | −0.0033 | −0.0020 |
| 64k m | | | | ~3.4 | ~2.7 | >1.44 (bound) |

- **Context gain** (CE with an 8k window minus CE with full 32k, on each row's last 4k): K32 matches dense.
  - K32 at 12/24/48/120 steps: 0.0681 / 0.0700 / 0.0725 / 0.0739. Dense: 0.0671 / 0.0700 / 0.0726 / 0.0750. Per-run SE ±0.0036.
- **Short guard:** level with dense throughout ⚠ (256 rows).
- **64k PF:** the dense-64k meter files are partial counts, so 64k dense PF is taken as K32 PF / 0.696.

## What lost (best m in each family)

| family | example | result |
|---|---|---|
| input token drops (random 20% blocks, first+last 25% / 10% / 5%), sd20 slots | I-rand 260 steps | m ≤ 0.8; I-fl gets worse with training |
| online mid-stack router, all variants (pyramid, forward-only early stack, merged summary, early-exit loss, periodic 50%-keep) | keep 10% @ block 7 | m ~0.6–0.8 |
| router keep 0.5 on top of K32 | 48 / 120 steps | 1.7247 / 1.7120: worse than dense at 4 steps (m < 0.24) |
| mixture-of-depths 50% / 25% | | m ≤ 0.47 |
| all-local 4096 windows | | context gain collapses (0.044) |
| warmup-then-freeze, progressive freeze | | ties dense only at 120 |
| partial MLP, frozen base (K32 + random f 0.5 / 0.75 / 0.9; top-k f 0.5 / 0.75) | | 1.7700 / 1.7124 / 1.6927; top-k 1.7342 / 1.6999 at 48: all worse than dense |
| partial MLP, trainable base (dense f 0.5 / 0.25) | | gets WORSE with steps: 1.7402 → 1.7647, 1.7480 → 1.8663 (24 → 120) |
| backprop-stop: [0, B) frozen without adapters, LoRA on [B, 32) | B = 4 / 8 / 16 | m 2.9 / 1.4 / <0.5 at 120 vs K32 3.45 |
| K32 at 8 rows/step (same tokens) | | ties 16 rows/step |
| packed length-mix / hybrid rows | | not candidates: packing undercounts FLOPs |

## Traps hit (also in STATUS.md)

- **LoRA adapters silently dropped at eval:** config.json lacked `early_lora`, so the distcp load skipped the adapter keys.
  Fixed: `eval_devce.py` rebuilds the adapters from checkpoint keys, carries `start`, and refuses checkpoints with keys the
  model lacks.
- **Intermittent end-of-run crash after mid-run saves:** gloo "Connection reset by peer" after the last step, seen on
  5 Beaker runs. Relaunching into the same directory resumes at the last step and exports. See
  `records/online-token-router.md`.
- **Beaker preemption restarts runs from step 0 without `SAVE`:** use `SAVE=60` on runs over an hour.
- **balrog (A100):** the env's flash-attn 2.8.2 was built for sm90/100 only; the balrog /tmp copy was rebuilt for sm80;90.
- **FLOP basis:** the table PF (36.6/step) flattered drop arms by ~1.2x; always use the measured meter.

## Not done

- **Generation-graded long-context eval.** The CTC suite needs chat-template SFT checkpoints, so comparing K32 vs dense
  bases needs a matched SFT stage first.
- **128k:** needs a new train/dev build from 2^17+ documents (disjoint from dev32k/dev64k) and a CTX=128k arm.
