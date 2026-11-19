---
title: Random whole-document drop on no-gold tasks (reorder / oolong / grouping), with and without label rewriting
date: 2026-11-18
updated: 2026-11-18
experiment: tokdrop_lr
area: token_routing
kind: result
status: current
superseded_by:
artifact:
summary: "negative (11-18, 16k early-loss proxy, eval_size 144, 2 seeds on the deciding arms): random document drop on top of the 20%/7% recipe cuts the step to 0.088-0.098x dense (recipe 0.221x) but ends 0.03-0.05 higher in held-out CE and is not reliably ahead of the recipe at equal FLOPs (own-lowest-loss savings vs the recipe 1.25x / 0.90x rewritten, 1.51x / 1.09x not rewritten; reorder clearly behind, grouping level, oolong noisy). Rewriting the label is not needed: leaving the full-row answer is slightly better on every task (-0.016 / -0.014 mean CE)."
---
# Random whole-document drop on no-gold tasks

**Verdict.** On tasks where every document feeds the answer, a random whole-document drop at the input (the gold-drop
mechanics, with no gold set) does **not** reliably help the recipe at 16k.
- It cuts the step from 0.221x dense (recipe) to 0.088x (keep U(.05,.5)).
- It ends 0.046 / 0.053 higher in mean held-out CE after one epoch (paired SE ±0.006 / ±0.007, seeds 0 / 1).
- At its own lowest loss it saves 1.25x / 0.90x FLOPs vs the recipe (seeds 0 / 1), so the gain is about a coin flip.
- Per task:
  - reorder is clearly hurt, about +0.02 CE behind the recipe at equal FLOPs on both seeds.
  - grouping is level.
  - oolong is ahead at equal FLOPs, but it is the noisiest task (about 4 answer tokens a row).

Compare gold drop on contradiction / outlier / qdmatch_nq, which keeps the answer evidence and was about 1.6-1.8x
ahead of the recipe.

**Label rewriting is not needed.** The no-rewrite control drops the same documents and keeps the full row's answer,
question and ids. It is better than the rewritten arm on every task: mean CE -0.016 ± 0.006 (seed 0) and
-0.014 ± 0.008 (seed 1), with reorder and grouping significant on both seeds. Its savings vs the recipe at its own
lowest loss are 1.51x / 1.09x. Even so, it ends 0.030 / 0.039 above the recipe.

All numbers are teacher-forced held-out CE ⚠ eval_size=144 (48 rows per task, ⚠ eval_size=48 per task). The paired
SEs quoted cover eval noise only. Seed-to-seed spread on the FLOP ratios is large: 1.25 vs 0.90 for the same arm.

## Setup
- **Proxy**: the early dense held-out-loss proxy, keepfrac_len setup (`research/token_routing/early_loss_proxy/`).
  - Training: Qwen3.5-4B markerfix base, one epoch = 225 steps of 16 rows on a 1,200-row-per-task pool.
  - Probes: dense-forward held-out answer CE at steps 0, 10, 25, 50, ..., 225.
  - Hardware: Beaker 2 x H100 per arm.
  - Arms: dft = dense full FT (LR 5e-6); k20 = whole-model LoRA r64 + online router keeping 20% after block 7 and 7%
    after block 16 (LR 1e-5).
- **Tasks**: reorder (ctc `reorder`, 90 passages), oolong (ctc oolong generator, items = lines, about 246 per row),
  grouping (the suite's unlabeled OpenAlex partition; n88 rows cut to 72 abstracts to fit 16,384 tokens).
  Pool `ng16k`: means 14.4k / 14.2k / 13.6k tokens.
- **Held-out** (`heldout_ng.npz`, 48 rows per task). Every held-out row is a full row with its original answer.
  - reorder: heldout_v3 gen16k, no shared book.
  - oolong: from the item-disjoint eval split; 1.3% of held-out item lines also occur in training.
  - grouping: disjoint n88 rows. **89% of held-out abstracts also occur in some training row**, because the generator
    reuses a small OpenAlex pool. The grouping (k, partition) differs, but this inflates absolute grouping scores.
  - No held-out prompt equals a training prompt.
- **Exact label rewriting** (`relabel.py`, offline): a full-subset relabel reproduces every row exactly, and every
  rewritten row's tokenized loss span equals its recomputed answer.
  - reorder: the original order restricted to the kept passages, renumbered.
  - grouping: clusters restricted to the kept abstracts, empty clusters dropped, k re-rendered in the same query
    phrasing.
  - oolong: ctc's own `build_variant` re-run over the kept items with the same variant and the same sampled
    parameters, plus the preamble's N. Degenerate subsets (ties, or the asked label / user / month gone) are
    redrawn; after 20 tries the row stays whole (47 of 1,200 rows in rddA_s0).
- **Drop arms** (pre-generated per epoch). One epoch sees each row once, so one random variant per row slot equals a
  per-step resample. Slots and loader order are identical to ng16k (checked).
  - In each arm, 15% of rows stay whole; the rest keep a share r of their documents (oolong: items).
  - rddA: r ~ U(0.05, 0.5), the gold-drop grJ setting. Token keep 0.394.
  - rddB: r ~ U(0.25, 0.75). Token keep 0.577.
  - rddAnr: the no-rewrite control, using the SAME kept sets as rddA. Each dropped document's span (marker to
    marker) is cut from the full row's tokens, as `gold_doc_drop` does. Survivors keep their printed ids, and the
    question, answer and oolong preamble N are untouched. What the unrewritten label then means:
    - reorder: the full permutation, including ids no longer shown.
    - oolong: the aggregate over all original items, most of them not shown.
    - grouping: the full partition, including absent ids, with the original k.
    - The ids are positional in the renderer, so they are preserved only because the cut is at token level; no task
      needed re-rendering.
  - FLOPs: the meter charges the rows actually run. Step cost is measured against dft's actual PF on full rows.
- **How this differs from `subproblem`**: those were offline pools of independent short examples mixed with long
  rows. Here each training row's own documents are subsampled, at a random keep share per row, with 15% whole rows,
  on top of the online router. It also differs from `docdrop-nongold-baseline.md`, which dropped random non-gold
  documents from gold tasks.

## Results (seed 0; seed 1 where run)

| arm | step cost vs dft | final mean CE | reorder | oolong | grouping | FLOPs x vs dft at own min | x vs k20 at own min | k20 CE at arm's final PF |
|---|---|---|---|---|---|---|---|---|
| dft | 1.000 | 0.317 | 0.304 | 0.370 | 0.276 | 1.00 | - | - |
| k20 sd0 / sd1 | 0.221 | 0.334 / 0.338 | 0.347 / 0.351 | 0.372 / 0.382 | 0.284 / 0.280 | 3.46 / 3.32 | - | - |
| rddB (U.25-.75) | 0.127 | 0.367 | 0.410 | 0.393 | 0.299 | 3.38 | 1.03 | 0.368 |
| rddA (U.05-.5) sd0 / sd1 | 0.088 / 0.090 | 0.381 / 0.390 | 0.441 / 0.432 | 0.390 / 0.430 | 0.311 / 0.310 | 4.36 / 3.91 | 1.25 / 0.90 | 0.411 / 0.386 |
| rddAnr (no rewrite) sd0 / sd1 | 0.096 / 0.098 | 0.365 / 0.377 | 0.417 / 0.426 | 0.371 / 0.399 | 0.306 / 0.305 | 4.57 / 4.09 | 1.51 / 1.09 | 0.397 / 0.383 |

Mean CE ⚠ eval_size=144; per-task columns ⚠ eval_size=48.

**Per task at matched FLOPs** (the recipe's running-min CE at the drop arm's final PF; seed 0, then seed 1):
- reorder: rddA .441 vs .422 / .432 vs .415 (behind); rddAnr .417 vs .416 / .426 vs .411.
- oolong: rddA .390 vs .477 / .411 vs .436 (ahead, noisy); rddAnr .365 vs .455 / .399 vs .432.
- grouping: rddA .311 vs .314 / .310 vs .308 (level); rddAnr .306 vs .314 / .305 vs .305.

**Paired final-CE differences** (same 144 rows, mean ± paired SE, seed 0 / seed 1):
- rddAnr - rddA: all -0.016±0.006 / -0.014±0.008; reorder -0.024±0.003 / -0.006±0.003; oolong -0.019±0.018 /
  -0.031±0.022; grouping -0.005±0.002 / -0.005±0.002.
- rddA - k20: all +0.046±0.006 / +0.053±0.007; reorder +0.094 / +0.081; oolong +0.017 / +0.047; grouping +0.027 /
  +0.030.

Chart: `research/token_routing/early_loss_proxy/random_doc_drop/ng_loss_vs_pf.png` (log CE vs actual PF; panels for
the mean and each task).

## Why (reading)
- With no gold set, every dropped document removes evidence the full-row answer needs, so the rows trained on are
  easier, different problems. Reorder suffers most: its held-out answer is a 90-way permutation, and the drop arms
  train mostly on 5-45 passage permutations.
- Rewriting moves the training answers further from the held-out ones (shorter permutations, smaller counts, fewer
  groups). That is the likely reason the no-rewrite control, which keeps full-row answers, scores slightly better on
  this full-row held-out loss.
- One epoch is a fixed number of rows. The drop arms spend far fewer FLOPs per row and end shallower, so their
  advantage, where there is one, is only at shallow loss levels. This is the same pattern as `subproblem`.

## Limits
- ⚠ eval_size=48 per task. Oolong's CE rests on about 4 answer tokens per row and swings by up to 0.1 between probes.
- Seed 1 was run only for k20, rddA and rddAnr. rddB is single seed.
- One epoch only. Deeper budgets (more rows) were not run.
- Grouping held-out abstracts overlap training (89%), and grouping rows are 72-abstract cuts of the n88 suite-train
  rows, since no OpenAlex pool is on disk to generate fresh 16k rows.
- Teacher-forced loss only; no generation scores.

## Where things live
- Code: branch `claude/random-doc-drop-nogold`, `research/token_routing/early_loss_proxy/random_doc_drop/`
  (`relabel.py`, `build_ng_pools.py`, `build_ng.sbatch`, `launch_ng.py`, `analyze_ng.py`, `record_ng.py`,
  `HANDOFF.md`).
- Data: mooney `/data/prasann/early_loss_proxy/shards/ng16k*`, `heldout_ng.npz`; weka
  `.../prasanns/early_loss_proxy/` (same paths).
- Results: S3/weka `early_loss_proxy/results/ng16k/`, scored in `random_doc_drop/ng_results.json`.
- ops: run ids `elp-ng16k-bk*`, result rows `elp-ng16k-<arm>-sd<seed>-probe` (52 rows).
