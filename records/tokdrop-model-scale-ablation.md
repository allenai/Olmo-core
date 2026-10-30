---
title: Result — token-drop recipe vs dense across Qwen3.5 0.8B / 2B / 4B / 9B (fast8k mt4, fresh pools)
date: 2026-10-30
updated: 2026-10-30
experiment: tokdrop_lr
area: token_routing
kind: result
status: current (9B dense f7808 in flight; not needed for the headline)
superseded_by:
artifact: https://claude.ai/artifact/SPSBxHFs2PUn5CPjJX3Wgt (hub; this ablation is a section there, no separate page)
summary: At matched data the recipe's dense-FLOPs-to-match grows with scale (1,952 / 3,904 fresh rows/task - 0.8B 1.7 / 1.9x, 2B 2.4 / 3.2x, 4B 4.1 / 3.7x, 9B 4.8 / 4.1x). Small-model dense saturates early (0.8B plateaus at 0.743), and at 7,808 rows/task the 0.8B recipe passes it (0.806, 2 seeds) by routing, not LoRA (dense+LoRA 0.757). The recipe costs 0.23-0.25x dense FLOPs/step and ~0.5-0.6x peak memory at every scale; it is less seed-stable below 4B.
---

# Token-drop recipe vs dense across model scale

**Verdict.** On the fresh 4-task pools, the recipe's gain over dense **grows with model size at a fixed
data budget**: dense needs 1.7 / 1.9× the FLOPs to match it at 0.8B, 2.4 / 3.2× at 2B, 4.1 / 3.7× at 4B and 4.8 / 4.1× at
9B (1,952 / 3,904 rows/task); at 9B the recipe beats the full dense f1952 run at 0.23× its FLOPs (two seeds,
0.902 / 0.903 vs 0.900). At 7,808 rows the order reverses (>4.2× / 2.9× / 2.2×) because dense saturates. Smaller models' lower ceiling does **not** cap the recipe: it caps small-model *dense*.
0.8B dense plateaus at 0.743 from 3,904 rows on, while the 0.8B recipe keeps learning and reaches 0.806 at
7,808 rows (two seeds, 0.806 / 0.807). That crossing comes from routing (outlier@8k 0.51 vs 0.30), not from
LoRA (dense + whole-model LoRA, no routing: 0.757). The recipe is less seed-stable below 4B (0.8B
contradiction@8k 0.38–0.76 over 3 seeds; dense seeds agree within 0.02).

## Setup

- Task mix `mt4` (contradiction / nq / outlier / oolong), fresh pools `mt4f1952 / f3904 / f5856 / f7808`
  (`build_mt4_fresh_beaker.sh`, campaign `-tde` at 4B; this ablation `-tds08b / -tds2b / -tds9b`). Same
  trainer path as 4B: unpacked, seq 12,288, 16 rows/step, micro 2, full activation checkpointing (forced at
  0.8B, whose scale default is none), no full-context steps.
- Score = 8-cell mean, {contradiction, nq, outlier, oolong} × {2k, 8k}, **eval_size 500 per cell**, no-cot,
  query position `after` (ds64 eval files), 1-GPU evals. PF = `flop_meter/actual_pflops`.
- Bases: marker-repaired with `fix_marker_embeddings_qwen35.py` (cosine + norm gates pass; trainer audit
  norm ≈1.0× median). 0.8B/2B/9B repaired locally on sneetches from the `*-modelonly` bases; Beaker runs use
  the weka `q35-{08b,2b,9b}-base-markerfix` copies built by the same script on 2026-09-02.
- Recipe = rank-64 LoRA (scale 32) on **all** blocks over a frozen base + online ridge router, keep 20% of
  each document's tokens at stage 1 and 7% at stage 2. Stage depths keep each layer type:
  - 4B / 9B (32 blocks, full attention at 3, 7, …): stages `7:0.2,16:0.07` (arm `lora64all32r20to7`).
  - 0.8B / 2B (24 blocks, same pattern): stage 1 at the full-attention block nearest 22% of depth (5.3 →
    **7**, i.e. 29%), stage 2 at **12**, the GDN block after full-attention 11 (50%), as 16 follows 15 at 4B
    (arm `lora64all24r20to7s12`).
  - LoRA rank 64 kept at every scale (64/1024 = 1/16 of d_model at 0.8B, 1/64 at 9B; 46.0M adapter
    params at 0.8B).
- LR: dense 5e-6 (the 4B choice) plus 2e-5 at 0.8B / 2B; recipe 1e-5 plus 3e-5 at 0.8B / 2B.
- Compute: 0.8B / 2B core runs and all seeds on jsteinhardt sneetches (2×H200 per run, preemptive_high,
  `scale_run_local.sbatch`); the dense tail, LR variants and 9B on Beaker jupiter (2×H100, 9B on 4×H100;
  `launch_fast8k.py`). Same commit for both paths (`6173899e6`).

## FLOPs-to-match

Dense PF at the recipe's score, piecewise log-linear along that size's own dense LR 5e-6 curve (seed
replicates averaged first), divided by the recipe's PF. ">" = the recipe beats dense's best point.

| rows/task | 0.8B | 2B | 4B | 9B |
|---|---|---|---|---|
| 1,952 | 1.7× (3 seeds) | 2.4× (2 seeds) | 4.1× (extrapolated below the dense range) | 4.8× (2 seeds) |
| 3,904 | 1.9× (2 seeds) | 3.2× (3 seeds) | 3.7× | 4.1× |
| 7,808 | **>4.2×** (2 seeds, above dense's plateau) | 2.9× | 2.2× | — |

The hub's 4B numbers (3.9 / 3.0 / 2.0×) used a different dense aggregation; the 4B column above is the same
method as the other columns.

## Per-run numbers

Mean8 = 8-cell mean (eval_size 500 per cell; per-cell binomial SE ≈ ±0.02 at f1 0.7, the 8-cell mean's
≈ ±0.008; run-to-run seed noise is larger, see below). GiB = peak reserved per GPU, rank 0.

| size | arm | rows/task (steps) | LR | seeds | PF | PF/step | GiB | mean8 | outlier@8k | contra@8k |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.8B | dense | 1952 (122) | 5e-6 | 1 | 74 | 0.61 | 16.6 | 0.483 | 0.047 | 0.265 |
| 0.8B | dense | 1952 (244) | 5e-6 | 1 | 147 | 0.60 | 16.6 | 0.687 | 0.297 | 0.753 |
| 0.8B | dense | 1952 | 5e-6 | 2 | 294 | 0.60 | 16.6 | 0.708 / 0.721 | 0.246 / 0.295 | 0.844 / 0.855 |
| 0.8B | dense | 3904 | 5e-6 | 1 | 588 | 0.60 | 16.6 | 0.740 | 0.274 | 0.918 |
| 0.8B | dense | 5856 | 5e-6 | 1 | 883 | 0.60 | 16.6 | 0.743 | 0.333 | 0.936 |
| 0.8B | dense | 7808 | 5e-6 | 1 | 1178 | 0.60 | 16.6 | 0.743 | 0.313 | 0.954 |
| 0.8B | dense | 1952 / 3904 | 2e-5 | 1 | 294 / 588 | 0.60 | 16.6 | 0.562 / 0.673 | 0.240 / 0.244 | 0.427 / 0.879 |
| 0.8B | dense + LoRA-all (no routing) | 7808 | 1e-5 | 1 | 961 | 0.49 | — | 0.757 | 0.299 | 0.952 |
| 0.8B | recipe | 1952 | 1e-5 | 3 | 71 | 0.145 | 10.0 | 0.655 / 0.590 / 0.655 | 0.211 / 0.310 / 0.284 | 0.759 / 0.382 / 0.648 |
| 0.8B | recipe | 3904 | 1e-5 | 2 | 141 | 0.145 | 10.0 | 0.706 / 0.715 | 0.265 / 0.297 | 0.867 / 0.831 |
| 0.8B | recipe | 7808 | 1e-5 | 2 | 283 | 0.145 | 11.2 | 0.806 / 0.807 | 0.511 / 0.517 | 0.938 / 0.927 |
| 0.8B | recipe | 1952 | 3e-5 | 1 | 71 | 0.145 | 11.2 | 0.518 | 0.194 | 0.378 |
| 2B | dense | 1952 (122) | 5e-6 | 1 | 170 | 1.40 | 34.0 | 0.615 | 0.072 | 0.618 |
| 2B | dense | 1952 (244) | 5e-6 | 1 | 339 | 1.39 | 34.0 | 0.718 | 0.313 | 0.854 |
| 2B | dense | 1952 | 5e-6 | 2 | 676 | 1.39 | 38.0 | 0.768 / 0.782 | 0.356 / 0.372 | 0.929 / 0.917 |
| 2B | dense | 3904 | 5e-6 | 2 | 1354 | 1.39 | 38.0 | 0.825 / 0.828 | 0.520 / 0.524 | 0.954 / 0.951 |
| 2B | dense | 5856 | 5e-6 | 1 | 2032 | 1.39 | 34.0 | 0.880 | 0.887 | 0.971 |
| 2B | dense | 7808 | 5e-6 | 1 | 2710 | 1.39 | 34.0 | 0.899 | 0.893 | 0.976 |
| 2B | dense | 1952 / 3904 | 2e-5 | 1 | 676 / 1354 | 1.39 | 34.0 | 0.707 / 0.722 | 0.244 / 0.304 | 0.848 / 0.919 |
| 2B | recipe | 1952 | 1e-5 | 2 | 167 | 0.342 | 21.9 | 0.715 / 0.750 | 0.270 / 0.348 | 0.883 / 0.835 |
| 2B | recipe | 3904 | 1e-5 | 3 | 333 | 0.342 | 21.8 | 0.786 / 0.836 / 0.806 | 0.479 / 0.631 / 0.475 | 0.932 / 0.953 / 0.934 |
| 2B | recipe | 7808 | 1e-5 | 1 | 666 | 0.341 | 21.0 | 0.872 | 0.818 | 0.973 |
| 2B | recipe | 1952 | 3e-5 | 1 | 167 | 0.342 | 21.6 | 0.680 | 0.202 | 0.817 |
| 4B | dense | 1952 / 3904 / 5856 / 7808 | 5e-6 | 1 | 1531 / 3063 / 4595 / 6128 | 3.14 | 67.7 (2 GPU) | 0.888 / 0.926 / 0.928 / 0.931 | 0.817 / 0.976 / 0.981 / 0.987 | 0.973 / 0.983 / 0.983 / 0.990 |
| 4B | recipe | 1952 / 3904 / 7808 | 1e-5 | 1 | 345 / 688 / 1373 | 0.71 | 31.9 | 0.883 / 0.915 / 0.924 | 0.862 / 0.970 / 0.989 | 0.959 / 0.977 / 0.987 |
| 9B | recipe | 1952 | 1e-5 | 2 | 632 | 1.29 | 38.2 (4 GPU) | 0.902 / 0.903 | 0.873 / 0.905 | 0.963 / 0.964 |
| 9B | recipe | 3904 | 1e-5 | 1 | 1262 | 1.29 | 37.8 (4 GPU) | 0.920 | 0.977 | 0.983 |
| 9B | dense | 1952 (122) | 5e-6 | 1 | 702 | 5.76 | 136 (2×H200) | 0.725 | 0.238 | 0.895 |
| 9B | dense | 1952 (244) | 5e-6 | 1 | 1398 | 5.73 | 136 (2×H200) | 0.818 | 0.576 | 0.936 |
| 9B | dense | 1952 / 3904 | 5e-6 | 1 | 2792 / 5590 | 5.72 | 64.2 (4 GPU) | 0.900 / 0.923 | 0.869 / 0.979 | 0.979 / 0.989 |

Metered FLOPs per step, recipe / dense: 0.241 (0.8B), 0.247 (2B), 0.225 (4B), 0.226 (9B). The 24-block
models drop later in relative depth (29% vs 22%) but adapters skip more weight-gradient work, so the ratio
barely moves. Peak memory, recipe / dense: 0.60 (0.8B), 0.57 (2B), 0.47 (4B), 0.60 (9B, both on 4 GPUs).

## Reading it

1. **Fixed data budget: the gain grows with scale.** At 1,952 and 3,904 rows/task FLOPs-to-match goes
   1.7 → 2.4 → 4.1× (1,952) and 1.9 → 3.2 → 3.7× (3,904) from 0.8B to 4B. At the same FLOPs fraction, the recipe trails its own
   same-data dense run (seed means) by 0.08 / 0.03 at 0.8B (1,952 / 3,904 rows), 0.04 / 0.02 at 2B and
   0.005 / 0.010 at 4B.
2. **The ceiling confound runs the other way.** Small dense models saturate early at LR 5e-6 (0.8B: 0.740
   → 0.743 → 0.743 over 3,904 → 7,808 rows; outlier@8k stuck at 0.27–0.33), so a ratio measured against
   their curve can blow up (0.8B f7808 >4.2×). At 4B dense saturates near 0.93, which is what pulls its
   ratio from 4× down to 2× as data grows. FLOPs-to-match therefore depends on where each curve saturates;
   the f1952 / f3904 rows are the like-for-like comparison.
3. **It is routing, not LoRA.** 0.8B dense + whole-model LoRA (no routing) at 7,808 rows scores 0.757, a
   hair above dense (0.743) and with the same outlier@8k (0.30); the recipe reaches 0.806 with outlier@8k
   0.51. This matches the 4B 2×2 (`records/lora-vs-routing-ablation.md`).
4. **Variance shrinks with scale.** Recipe seeds disagree far more than dense seeds below 4B: 0.8B f1952
   0.590–0.655 (contradiction@8k 0.38–0.76), 2B f3904 0.786–0.836 (outlier@8k 0.48–0.63); at 9B two seeds
   give 0.902 / 0.903. Dense seeds agree within 0.015 at every size. Single-seed recipe numbers at 0.8B / 2B
   are ±0.03 on the 8-cell mean.
5. **LR is not the confound.** Higher LRs are worse at both small sizes, for dense (2e-5) and for the recipe
   (3e-5), as at 4B.

## Limits

- One task mix (mt4), the 8k fast screen, not the 32k ladder.
- Dense curves are single-seed except 0.8B/2B f1952 and 2B f3904 (2 seeds).
- 4B has no dense step-cut points, so its f1952 ratio is extrapolated below the dense range.
- Peak memory is per GPU and not comparable across GPU counts (9B on Beaker and 4B dense f5856/f7808 ran on 4
  GPUs; the local 9B cut points and seed ran on 2×H200).
- Scores are read from ops with metric ∈ {f1, score}. Since 2026-10-30 other sessions also record teacher-forced
  `answer_ce` / `answer_top1` rows at eval_size 500 on the same train runs. A collector that keeps "the
  largest eval per (task, rung)" without a metric filter (e.g. the hub's `pareto_points.json` regenerated
  10-30 06:52) mixes them in. 4B 8-cell means then read 0.14–0.17.
- The 24-block stage-1 depth (block 7 = 29%) is not the 22% used at 4B. An earlier stage 1 (block 3) puts a
  full-attention block at 12.5%, which the 4B fidelity study found biased.

## Where things are

- Branch `claude/recipe-scale-ablation-71d007`, which merges `claude/online-router-1bc01a`.
- Launchers:
  - `research/token_routing/ds64_fast8k/launch_fast8k.py` (`F2K_SCALE`, `F8K_SUFFIX`, `F8K_TRAIN_EXTRA`, `F8K_EVAL_NGPU`);
  - `scale_run_local.sbatch` / `scale_smoke_local.sbatch` (jsteinhardt);
  - `sync_fresh_shards_s3_beaker.sh` (weka → S3 for the local copies).
- Collector: `collect_scale_ablation.py` → `results/scale_ablation.json`.
- Recorders:
  - `record_local_scale_evals.py`;
  - `record_beaker_evals.py`, which now records only finished eval jobs, so a mid-run parse can no longer drop the 2k rung.
- Scores: `ops result` (experiment `tokdrop_lr`).
- Local checkpoints and eval outputs: `sneetches:/data/prasann/scale_abl/{runs,out,logs}`. Beaker ones: weka `ctc_suite/ckpts/<run>`.
