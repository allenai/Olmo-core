---
title: Result — LoRA vs token routing, a 2×2 ablation at matched LRs (Qwen3.5-4B SFT, 4-task and 11-task CTC mixes)
date: 2026-10-27
updated: 2026-10-27
experiment: tokdrop_lr
area: token_routing
kind: result
status: current
superseded_by:
artifact: https://claude.ai/artifact/RFpSfNnqnz8M9ygi8kK1U5 (tokdrop_lr overview; section from make_lora_routing_section.py)
summary: Routing does most of the work. At matched FLOPs, two-stage routing alone saves 2.5-3.5x dense FLOPs on the 4-task mix and 1.6-1.8x on the 11-task mix; whole-model LoRA alone saves 1.0-1.5x; both 3.8-4.8x and 1.8-2.3x. In log-FLOP terms routing is ~53-96% of the gain, LoRA-alone 7-25% (54% at the smallest 11-task size). At matched DATA all four cells score within ~0.03; the gain is FLOP cuts (PF 0.74x LoRA, 0.31-0.43x routing, 0.23-0.32x both). The LR confound is not the story - dense and routing-only are best at 5e-6, LoRA arms mostly at 1e-5.
---
# Result — how much of the cheap-method gain is LoRA, how much is routing?

**Question (from the code review).** Our cheap methods (two-stage online gradient-norm routing,
optionally with LoRA) beat dense at matched training FLOPs, but no dense + LoRA control existed,
the LoRA arms ran at LR 1e-5 while dense ran only up to 5e-6, and the router barely beat random
token choice at matched structure. So the gain might be mostly LoRA + LR. This 2×2 runs every cell
at **both** LRs and attributes the matched-FLOP gain.

## Verdict

**Routing is most of it; LoRA is a smaller, additive-ish extra; LR is not the explanation.**

- **Matched data, quality is a wash.** On the same rows the four cells score within ≈0.01–0.03 of
  each other (4-task 8-cell mean; on the 11-task mix routing costs 0.02–0.04 on the same data). The
  matched-FLOP gains are therefore almost entirely **FLOP cuts**: per step PF ÷ dense is 0.74× for
  LoRA-all (frozen base, no weight gradients), 0.31× (4-task) / 0.43× (11-task) for two-stage
  routing, and 0.23× / 0.32× for both.
- **Matched FLOPs.** Dense FLOPs needed to match ÷ the cell's FLOPs (dense at its best LR,
  2e-6 included):

  | | 4-task (1,464 / 1,952 / 3,904 rows/task) | 11-task (240 / 480 / 720 rows/task) |
  |---|---|---|
  | B. dense + LoRA-all | 1.48× / 1.41× / 1.17× | 1.39× / 1.15× / 1.06× |
  | C. routing, no LoRA | 3.45× / 2.85× / 2.89× | 1.79× / 1.67× / 1.56× |
  | D. routing + LoRA-all | 4.82× / 4.63× / 3.81× | 1.83× / 2.09×¹ / 2.30× |

- **Share of the gain** (log2 of the FLOP multiplier, D = total): routing alone accounts for
  65–79% on the 4-task mix and 53–96% on the 11-task mix; LoRA alone for 12–25% (4-task) and
  7–54% (11-task, shrinking with data: 54% at 240 → 19% at 480 → 7% at 720 rows/task). The
  remainder is interaction: positive at 11-task 720 (LoRA helps more on top of routing: D vs C
  1.47×) and on 4-task f1952, roughly zero to negative elsewhere.
- **LoRA's value on top of routing** (D vs C at matched FLOPs) is 1.3–1.6× on the 4-task mix and
  1.0× → 1.25× → 1.47× on the 11-task mix as data grows. On matched DATA, LoRA on top of routing is
  worth −0.02…+0.02 — i.e. it is a FLOP saving, not a quality gain.
- **LR.** Dense is best at 5e-6 at every size (1e-5 costs 0.01–0.05), and so is routing without
  LoRA in 5 of 6 sizes; the LoRA arms prefer 1e-5: B in 4 of the 5 sizes run at both LRs, D in 4 of 6 (the 11-task 240 pick is a 0.002 tie). At the smallest 4-task sizes (488 / 976 rows/task) dense is better still at 2e-6, which is why the reference curve takes every LR. So the original
  "LoRA at 1e-5, dense at 5e-6" choices were each arm's better LR, not a handicap on dense.
- **Router vs random token choice** (whole-model LoRA + two-stage, LR 1e-5): the gradient-norm
  router beats random choice by +0.027 at 1,464 rows/task (router 4 seeds vs random 1) and ties at
  3,904 (0.915 vs 0.914). So "routing" here is mostly the **drop structure** (keep 20% after
  block 7, 7% after block 16) rather than *which* tokens the router keeps.

## Setup

- **Model / data.** Qwen3.5-4B (marker-repaired base), CTC SFT, fast8k unpacked rows, 16
  rows/step, 2× H100 on Beaker jupiter, single pass over fresh rows (no repeated epochs).
  4-task mix = contradiction / nq / outlier / oolong: fresh pools `mt4f1952`, `mt4f3904` and the
  original pool `mt4s1464`. 11-task mix: `mt11s240/s480/s720`.
- **Cells** (`research/token_routing/ds64_fast8k/launch_fast8k.py`):
  - A `dense`;
  - B `denselora64all` = `--lora-rank 64 --lora-blocks 0 --lora-scale 32` (frozen base, rank-64
    adapters on all 32 blocks, no routing);
  - C `r20to7M7de0` (4-task) / `r20to7M7de0f16` (11-task, + 16-token per-document floor; the router
    session's `r20to7f16` is the same cell): online gradient-norm router, keep 20% after block 7,
    7% after block 16, no full-context steps (`--st-otr-dense-every 0`);
  - D `lora64all32r20to7` / `lora64all32r20to7f16` (router session: `lora64K32r20to7[f16]`) = C
    plus whole-model LoRA.
- **LRs.** Every cell at 5e-6 and 1e-5; per (mix, data size, cell) the better LR (seed copies
  averaged first) is used and the other is shown.
- **Seeds.** Second seeds (`--seed 1`, run names `<arm>sd1`) for the cells whose comparisons were
  within ~0.02. Seed counts are in the table.
- **Scores.** 4-task: 8-cell mean (4 tasks × 8k + 2k rungs, ds64 eval files, query position
  after). 11-task: mean of the 9 graded tasks at 8k on the CTC-suite rows (nq, hotpotqa,
  qdmatch_nq, outlier, oolong, contradiction, xabsence, strmatch, grouping; reorder and rerank
  excluded). eval_size 500 per task and rung, no-cot. Every LoRA eval log was checked for
  `[early-lora] adapters re-installed` (`record_beaker_evals.py` refuses one without it).
- **FLOPs.** Training PF from the trainer's FLOP meter (`flop_meter/actual_pflops`).
- **Matched-FLOP reference.** Dense curve per mix = per data size the best LR of every dense run
  (2e-6, 5e-6, 1e-5), seed-averaged, all pools of the mix, interpolated linearly in log PF.
  "vs dense" = score − dense(PF); "× dense" = dense PF reaching the same score ÷ the cell's PF.
  Attribution: LoRA effect = B−A and D−C; routing effect = C−A and D−B; interaction = D−C−B+A
  (A ≡ 0 at matched FLOPs since A is on the curve), both in score and in log2 of the multiplier.

## Tables

Dense reference = best LR per data size (incl. 2e-6). ⚠ Most comparisons rest on 1–2 seeds: the
8-cell / 9-task mean moves ≈0.005–0.025 between seeds (sd column), and **outlier@8k moves up to
±0.1–0.19** between seeds (e.g. dense f1952: 0.817 vs 0.909), so do not read single-task columns
as effects.

### Cells

| data | cell | best LR | other LR | seeds | PF | mean | outlier@8k | contra@8k | vs dense | × dense |
|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| 11-task, 240 rows/task | A. dense | 5e-06 | 1e-05: 0.675 | 2 (sd 0.012) | 231 | 0.689 | 0.482 | 0.806 | +0.000 | 1.00× |
|  | B. dense + LoRA-all | 1e-05 | 5e-06: 0.690 | 2 (sd 0.025) | 170 | 0.691 | 0.536 | 0.775 | +0.027 | 1.39× |
|  | C. two-stage routing | 5e-06 | 1e-05: 0.638 | 2 (sd 0.017) | 100 | 0.669 | 0.558 | 0.799 | +0.047 | 1.79× |
|  | D. routing + LoRA-all | 5e-06 | 1e-05: 0.646 | 2 (sd 0.002) | 75 | 0.648 | 0.484 | 0.733 | +0.066 | 1.83× |
| 11-task, 480 rows/task | A. dense | 5e-06 | 1e-05: 0.744 | 2 (sd 0.007) | 482 | 0.755 | 0.635 | 0.883 | +0.000 | 1.00× |
|  | B. dense + LoRA-all | 1e-05 | 5e-06: 0.739 | 1 | 356 | 0.740 | 0.671 | 0.875 | +0.013 | 1.15× |
|  | C. two-stage routing | 1e-05 | 5e-06: 0.720 | 1 | 208 | 0.726 | 0.647 | 0.890 | +0.045 | 1.67× |
|  | D. routing + LoRA-all | 1e-05 | 5e-06: 0.702 | 1 | 155 | 0.720 | 0.666 | 0.869 | +0.062 | 2.09× |
| 11-task, 720 rows/task | A. dense | 5e-06 | 1e-05: 0.771 | 2 (sd 0.009) | 714 | 0.782 | 0.697 | 0.915 | +0.000 | 1.00× |
|  | B. dense + LoRA-all | 5e-06 | 1e-05: 0.755 | 2 (sd 0.011) | 528 | 0.765 | 0.611 | 0.901 | +0.004 | 1.06× |
|  | C. two-stage routing | 5e-06 | 1e-05: 0.755 | 2 (sd 0.010) | 309 | 0.755 | 0.700 | 0.906 | +0.040 | 1.56× |
|  | D. routing + LoRA-all | 1e-05 | 5e-06: 0.740 | 2 (sd 0.010) | 232 | 0.762 | 0.654 | 0.899 | +0.072 | 2.30× |
| 4-task, fresh 1,952 rows/task | A. dense | 5e-06 | 1e-05: 0.870 | 2 (sd 0.000) | 1530 | 0.887 | 0.863 | 0.975 | +0.009 | 1.14× |
|  | B. dense + LoRA-all | 1e-05 | 5e-06: 0.877 | 2 (sd 0.002) | 1142 | 0.882 | 0.808 | 0.969 | +0.030 | 1.41× |
|  | C. two-stage routing | 5e-06 | 1e-05: 0.852 | 2 (sd 0.008) | 467 | 0.866 | 0.839 | 0.962 | +0.084 | 2.85× |
|  | D. routing + LoRA-all | 1e-05 | 5e-06: 0.865 | 2 (sd 0.002) | 344 | 0.881 | 0.861 | 0.957 | +0.114 | 4.63× |
| 4-task, fresh 3,904 rows/task | A. dense | 5e-06 | – | 1 | 3063 | 0.926 | 0.976 | 0.983 | +0.000 | 1.00× |
|  | B. dense + LoRA-all | 1e-05 | – | 1 | 2286 | 0.917 | 0.966 | 0.987 | +0.011 | 1.17× |
|  | C. two-stage routing | 5e-06 | 1e-05: 0.900 | 1 | 936 | 0.917 | 0.983 | 0.977 | +0.088 | 2.89× |
|  | D. routing + LoRA-all | 1e-05 | 5e-06: 0.914 | 1 | 688 | 0.915 | 0.970 | 0.977 | +0.114 | 3.81× |
| 4-task, original pool 1,464 rows/task | A. dense | 5e-06 | 1e-05: 0.802 | 3 (sd 0.013) | 1154 | 0.854 | 0.729 | 0.962 | +0.000 | 1.00× |
|  | B. dense + LoRA-all | 1e-05 | 5e-06: 0.851 | 2 (sd 0.005) | 861 | 0.862 | 0.749 | 0.962 | +0.043 | 1.48× |
|  | C. two-stage routing | 5e-06 | 1e-05: 0.786 | 2 (sd 0.017) | 355 | 0.859 | 0.791 | 0.956 | +0.090 | 3.45× |
|  | D. routing + LoRA-all | 5e-06 | 1e-05: 0.852 | 2 (sd 0.001) | 260 | 0.861 | 0.700 | 0.941 | +0.106 | 4.82× |

### Attribution at matched FLOPs (Δ score / FLOP multiplier)

| data | LoRA, no routing (B−A) | LoRA on routing (D−C) | routing, no LoRA (C−A) | routing on LoRA (D−B) | interaction | total (D−A) | PF ÷ dense PF, same data (B · C · D) |
|---|--:|--:|--:|--:|--:|--:|---|
| 11-task, 240 rows/task | +0.027 / 1.39× | +0.020 / 1.02× | +0.047 / 1.79× | +0.040 / 1.32× | -0.007 / 0.74× | +0.066 / 1.83× | 0.74 · 0.43 · 0.33 |
| 11-task, 480 rows/task | +0.013 / 1.15× | +0.017 / 1.25× | +0.045 / 1.67× | +0.050 / 1.82× | +0.005 / 1.09× | +0.062 / 2.09× | 0.74 · 0.43 · 0.32 |
| 11-task, 720 rows/task | +0.004 / 1.06× | +0.033 / 1.47× | +0.040 / 1.56× | +0.069 / 2.17× | +0.029 / 1.39× | +0.072 / 2.30× | 0.74 · 0.43 · 0.32 |
| 4-task, fresh 1,952 rows/task | +0.021 / 1.24× | +0.030 / 1.62× | +0.075 / 2.50× | +0.084 / 3.28× | +0.009 / 1.31× | +0.105 / 4.06× | 0.75 · 0.30 · 0.23 |
| 4-task, fresh 3,904 rows/task | +0.011 / 1.17× | +0.026 / 1.32× | +0.088 / 2.89× | +0.103 / 3.26× | +0.015 / 1.13× | +0.114 / 3.81× | 0.75 · 0.31 · 0.23 |
| 4-task, original pool 1,464 rows/task | +0.043 / 1.48× | +0.016 / 1.40× | +0.090 / 3.45× | +0.063 / 3.26× | -0.027 / 0.94× | +0.106 / 4.82× | 0.74 · 0.31 · 0.23 |

¹ 11-task 480, cell D: the second seed (`lora64all32r20to7f16sd1`, LR 1e-5) has no grouping score: its grouping eval died twice at the same point with an NCCL all-gather timeout (one rank stuck ~10 min, most likely a runaway generation), so the cell is one seed. On the 8 other tasks seed 1 scores 0.779 vs seed 0's 0.770, so the single-seed number, if anything, understates D.

## Limits

- **Seeds.** 1–3 seeds per cell; differences under ≈0.02 in the mean are within seed noise.
- **Low-FLOP end of the dense curve.** The cheap cells' PF (75–470) falls at the bottom of the
  dense curve, where it is built from the smallest full runs (4-task s488 / s976; 11-task s240) and
  hard-stopped cuts (4-task s488 at 31 / 61 steps; 11-task s240 at 32 / 64 steps) whose LR never
  decayed. Those cuts under-state dense somewhat, which inflates the multipliers of C and D more
  than B's.
- **Pools.** For the fresh 4-task cells (f1952 / f3904) dense below 1,530 PF comes from the
  original pool (s-pool); dense f1952 vs s1952 differ by 0.017, within seed noise.
- **Scope.** Matched data costs routing 0.02–0.04 on the 11-task mix; the cheap methods win on
  FLOPs, not on data efficiency. The router-vs-random comparison is two sizes, one random seed.
- **Two-stage only.** The recipe's adapters-on-blocks-0–6 variant is not a cell here; LoRA-all is
  the version the review questioned.

## Where things are

- Launcher: `research/token_routing/ds64_fast8k/launch_fast8k.py` (`F8K_SUFFIX=-tdx`, `--seed`);
  run state `research/token_routing/ds64_fast8k/state.json` on branch
  `claude/vibrant-williams-4b0f66` (untracked).
- Analysis: `research/token_routing/ds64_fast8k/lora_routing_2x2.py` →
  `results/lora_routing_2x2.json` (gitignored; tables above are `make_lora_routing_section.py
  --md`). Overview-page section: `make_lora_routing_section.section()`.
- Recording: `record_beaker_evals.py --suffix=-tdx` (finalized jobs only, adapter check). All
  numbers are in `ops result show --experiment tokdrop_lr` under the `-tdx` / `-tde` / `-tdb` /
  `-lro` / `-lrg` train runs.
- Related: `records/online-token-router.md` (router), `records/olmo3-cptdrop-results.md` (CPT,
  where whole-model LoRA is the only win).
