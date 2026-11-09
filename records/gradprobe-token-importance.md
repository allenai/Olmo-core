---
title: Gradient token importance and the lossless keep budget (dev-loss sanity check for token dropping)
date: 2026-10-13
updated: 2026-10-19
experiment: learned_router
area: token_routing
kind: result
status: closed for now (exploratory; eval_size 16 per cell)
superseded_by: 
artifact: https://claude.ai/artifact/BRAYbDr5uLW7KdZzeh2yL1
summary: Which document tokens a frozen CTC model needs: one gradient pass at gate g=0.5 finds them (plain |grad| does not); smallest lossless budget vs 8k/16k/32k.
---
# Gradient token importance and the lossless keep budget

**Question.** On a trained checkpoint, do gradients reveal which context tokens matter, and how far can
dev loss (teacher-forced answer CE) hold if the least important x% of document tokens are dropped from the
forward pass? Token dropping only; layer skipping was deferred.

**Code.** `research/token_routing/learned_router/gradprobe/` (`grad_importance.py` + README). Results JSONs
and plots: `gradprobe/results/` (gitignored, copied there 2026-10-16). Every number below is teacher-forced
answer CE (no-cot), ⚠ **eval_size = 16 rows per (model, task, length)** unless stated. Treat differences
under ~0.1 nats as noise: re-picking the same mask under GPU nondeterminism moves ΔCE by 0.06-0.10.

## Method

* Per document-body token t, a keep gate g_t through the existing relaxed-deletion path (`soft_keep`:
  attention +log g key bias, GDN beta/decay scaled by g, removal-aware conv). g = 1 is the full model
  (matches flash to ≤2e-5), g = 0 is exact deletion (soft binary mask matches hard compaction to ≤1e-2).
  Markers, prompt, question and answer are never dropped.
* Scores: |dCE/dg| at g = 1 ("|grad|"); the same gradient with every body gate at g = α ("gradient at
  g = α", one forward+backward); integrated gradients over g = α·1 (S steps); iterative re-scoring.
* Hard-drop curves: keep the top ρ of body tokens, compact, measure ΔCE vs the full context, ρ from 50%
  down to 0.05%. Gold-first baselines keep every gold-document token first, then fill by random or by the
  g = 0.5 gradient (these use the answer: oracle references, not deployable). The bar `gold_fl20p8_noslot`
  and its x-sweep are recomputed in the same runs (mask matches the grid's bar token-for-token).
* Models: the per-task dense Qwen3.5-4B CTC checkpoints (dev-loss-grid manifest), the setA multi-task SFT
  (`q35-4b-dense-ctc-setA-32k-20260923T1537/step888`), and setA's pre-SFT base
  (`q35-4b-dense-256k-fix/step2385`). Tasks nq, outlier, contradiction; rungs 8k/16k/32k.

## Findings

1. **Where the gradient is taken matters more than how many steps are averaged.** At g = 1 the finetuned
   models are near-certain (CE ≈ 0.001), the loss is flat and |grad| is mostly noise: dropping 90% of tokens
   by |grad| is ≈ random on outlier and contradiction. One pass at **g = 0.5** recovers most of the gap and
   matches 16-step IG at 8k (ΔCE at 10% kept, per-task: outlier +0.68 → +0.08, contradiction +1.56 → +0.33).
   IG variants that include g = 0.5 work; 2- and 4-step IG (no g = 0.5) do not. The best g is sharp on
   outlier/contradiction and drifts to ~0.6 at 32k on contradiction; an adaptive g (steepest CE drop along
   g, forward-only probes) is a wash. Re-scoring |grad| never helps (it re-evaluates at saturated g = 1).
2. **Gold first, then fill by the g = 0.5 gradient is the best rule** (ΔCE ≈ 0 at 10% kept on all three tasks
   at 8k). Gold + *random* filler is worse than gold alone on contradiction: random fragments distract.
3. **The bar heuristic is really "gold + 8 header tokens per doc".** Its first/last-x% part adds nothing:
   gold + headers with x = 0 is already lossless (nq 5.5-7%, outlier 7-11%, contradiction 20-23% kept).
   Without headers, setA collapses (gold docs only: +1.58 on nq 8k). Without gold, the heuristic fails
   (+0.4 to +1.3). At the bar's own per-row budget, the gold-blind g = 0.5 gradient is within about +0.07 of
   the bar; plain |grad| is +0.2 to +0.6 worse.
4. **setA behaves like the per-task models** (same ranking), slightly less compressible. **setA's pre-SFT base
   does not use the context** (full CE 2-5 nats; any dropping lowers CE, random included; gold-token AUROC
   0.50-0.75 vs 0.78-0.99 for SFT models), so it cannot test the saturation story.
5. **Smallest lossless budget vs length** (gold first + g = 0.5 gradient; ΔCE ≤ 0.1 on the across-row median
   curve, walking down from 50%; fraction / absolute document tokens):

   | | 8k | 16k | 32k |
   |---|---|---|---|
   | nq, setA | 5% / 369 | 2% / 323 | 1% / 324 |
   | nq, per-task | 0.1% / 7 | 0.1% / 8 | 0.1% / 16 |
   | outlier, setA | 10% / 807 | 5% / 793 | 5% / 1,579 |
   | outlier, per-task | 1% / 81 | 5% / 793 | 30% / 9,472 (unreliable, below) |
   | contradiction, per-task | 10% / 771 | 10% / 1,582 | 10% / 3,214 |
   | contradiction, setA | 10% / 771 | 10% / 1,582 | 20% / 6,428 |

   nq (a fixed-size needle) needs a roughly constant absolute budget, so the fraction falls as 1/length;
   contradiction keeps a constant fraction (the budget grows with the corpus, plausibly because "no other
   document contradicts" is evidence that scales). Gold-blind, the g = 0.5 gradient's fraction is flat or
   rising on every task. The per-task outlier 32k cell is unreliable: that model is itself weak at 32k (half
   the rows have full CE > 0.1) and one bump at 20% breaks the walk-down; by eye its knee is ~2%.

## Base-model gradients, checkpoint drop + eval (2026-10-19)

**Question (Prasann).** The finetuned checkpoints are near-certain, so their gradients are saturated. Does
scoring tokens with the *base* model's gradient (unsaturated loss), then dropping and evaluating with the
checkpoint, beat scoring with the checkpoint itself?

**Setup.** `gradprobe/cross_score.py` (sbatch: `SCORE_CKPT=...`); both models loaded, same rows, same body
positions; every rule hard-dropped and scored on the finetuned model. Pairs: setAbase → setA32k, and raw
Qwen3.5-4B base (`q35-4b-base-modelonly`, the per-task models' init) → per-task `ctc-4b-<task>-full`. nq /
outlier / contradiction at 8k, ⚠ **eval_size = 64 rows per cell** (all the 8k rows). Results:
`gradprobe/results/json/cross/` (+ `summary_tau0.1.txt` from `cross_summary.py`); ops results
`gradprobe_cross_*`.

Median ΔCE on the checkpoint (keep 10%; nq at 2%, where everything is lossless at 10%):

| eval model / task | ckpt \|grad\| g=1 | ckpt \|grad\| g=0.5 | base \|grad\| g=1 | base \|grad\| g=0.75 | random |
|---|---|---|---|---|---|
| setA / nq (2%) | +0.10 | **+0.00** | +0.30 | +0.36 | +1.44 |
| setA / outlier | +0.87 | **+0.04** | +0.63 | +0.66 | +1.09 |
| setA / contradiction | +1.59 | **+0.55** | +0.97 | +1.24 | +1.63 |
| per-task / nq (2%) | +0.04 | **+0.02** | +0.08 | +0.13 | +1.30 |
| per-task / outlier | +0.65 | **+0.00** | +0.03 | +0.03 | +1.28 |
| per-task / contradiction | +1.73 | **+0.31** | +1.10 | +0.87 | +1.75 |

Paired at the bar's per-row budget, best base rule (g = 0.75) minus checkpoint g = 0.5: setA nq +0.09 ± 0.06,
outlier +0.29 ± 0.04, contradiction +0.72 ± 0.05; per-task nq +0.01 ± 0.01, outlier +0.00 ± 0.04,
contradiction +0.11 ± 0.03.

6. **Base gradients beat the checkpoint's saturated g = 1 gradient on outlier and contradiction** (per-task
   outlier +0.65 → +0.03 at 10% kept), as hypothesized, and the base works best at g = 1-0.75 (its loss is not
   saturated; g ≤ 0.5 and the signed gradient are much worse). On nq the base is slightly worse.
7. **They never beat the checkpoint's own g = 0.5 gradient**: tie on per-task nq/outlier, worse elsewhere.
   Softening the gate already removes the saturation, and the checkpoint knows where the answer is (gold AUROC
   0.77-0.97 vs 0.51-0.84 for the base). Base and checkpoint rankings agree only weakly (Spearman ≈ 0.3).
8. **The raw Qwen3.5 base is the better scorer than setA's pre-SFT base** (consistent with finding 4: setAbase
   ignores the context; full CE 2-5.5 nats vs 0.8-1.6 for the raw base). Gold-first + base fill (`base_gold_ga`)
   ties gold-first + checkpoint fill on per-task nq/outlier and is worse on contradiction.

## Limits and traps

* ⚠ eval_size 16 everywhere (5 for some superseded 16k/32k runs); no generation-graded check.
* The soft-gate attention builds a dense T×T bias: 32k needs per-block activation checkpointing (~55 GB on
  H200, auto above 12k tokens); **64k OOMs** (one 130 GiB allocation at T = 66,001). A 64k point needs a
  memory-efficient soft-keep attention.
* Isolated per-row spikes (+0.1 to +0.7 at one ρ, gone at the next) make a strict per-row ρ* unusable; use
  the median curve and τ = 0.1, or more rows.
* Infra traps hit along the way (Beaker image lacks `dataclass_extensions` and has a conflicting `fla`;
  lorax CUDA init broken; berkeleynlp `preemptive_high_sewonm` cap is group-wide; Lambda `/accounts` quota
  full) are in memory and `gradprobe/README.md`.

## Next steps (not started)

1. A 64k point (memory-efficient soft-keep attention) to check setA's ~330-token nq budget holds.
2. A deployable rule: the g = 0.5 gradient is an oracle (uses the answer); a router trained to imitate it is
   the natural follow-up for `learned_router`.
3. Layer skipping with the same probe (per-(token, layer) gates at g = 0.5).
