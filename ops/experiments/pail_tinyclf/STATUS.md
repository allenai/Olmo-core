---
experiment: pail_tinyclf
title: Smallest AI-vs-human text classifier that keeps recall (size ladder, cascades, data ablations) on WildAI
status: paused
updated: 2026-10-19
branch: main (PAIL; tinyclf merged 2026-10-19 as 0a07b1f, branch deleted)
launcher: /accounts/projects/berkeleynlp/prasann/projects/PAIL/scripts/tinyclf/run.sbatch (Slurm), scripts/tinyclf/beaker.sh (Beaker)
next_step: add 2026-era training data (the main gap) and benchmark optimized Ettin-17m serving for the 50T-token cost estimate
records:
  - https://claude.ai/artifact/WWqKqBFQk1fUpYjxRDPY1b (living report, v12; decision log, charts, trade-off tool)
  - /accounts/projects/berkeleynlp/prasann/projects/PAIL/scripts/tinyclf/README.md
results:
  - s3://ai2-llm/checkpoints/prasanns/PAIL/tinyclf/runs/ (data/, models/<rung>/{info.json,scores.npz,model*}, report*.md/json, cascade.*)
  - s3://ai2-llm/checkpoints/prasanns/PAIL/tinyclf/corpora/ (wildai_1m_strat, wildai_200k_nat, mage, raid, coling, meld_eval, dsewiki)
---
# pail_tinyclf: the smallest classifier that keeps recall

Code: `src/tinyclf/` (prepare, train, report, cascade, models: hashlin, fasttext incl. folded `prune=N`, hf encoders),
`scripts/tinyclf/` (corpora converters, run.sbatch, beaker.sh, s3sync.py, analysis/), `scripts/beaker_shell/bkx` (warm Beaker shell
via S3). All on PAIL main (worktree removed). No jobs running.

## Decisions
- 2026-10-14 (Prasann): two-corpus classifier training with a held-out dev set, aiming for the smallest model that keeps high recall.
- 2026-10-15 (Prasann): non-debug work moves to Beaker/S3. berkeleynlp Slurm is for debugging only and runs at QOS normal (the ops hook enforces this). jsteinhardt may preempt (max 8 GPUs at preemptive_high, never evicting rhys_gould). Beaker A100s only for jobs under 10 min.
- 2026-10-16 (Prasann): use WildAI (/data/prasann/wildai on thidwick) as the main data. Mix in COLING (approved despite no stated licence) and RAID. meld-eval is eval only. DetectRL-X deferred. DseWiki agent and human pages go in dev/test, not train.
- 2026-10-16 (Prasann): penalising missed AI text and cascades: implement with defaults, evaluate later (done since).
- 2026-10-19 (Prasann): FPR at 99% recall is the primary metric everywhere (report.py default `--fix recall --at 0.99`). Stratify results by crawl year.
- 2026-10-17 (Claude, from evidence): reference data is the crawl-stratified sample `wildai_1m_strat`. `wildai_200k_nat` is crawl-confounded and kept only for within-run comparisons.

## Current best numbers (dev = 26,879 docs, test = 27,210, 2026 natural slice = 49,668 / 17.6% AI; run wildai_1m_strat)
FPR at 99% dev recall (dev-tuned threshold). Source: scripts/tinyclf/analysis/fpr99_all.py and tradeoff.py run on the S3 scores.

| model | params | dev FPR@99 | test recall / FPR | 2026 recall / FPR (dev thr) | 2026 FPR if tuned for 99% on 2026 |
|---|---|---|---|---|---|
| Ettin-17m (1.94M docs) | 16.9M | 1.84% | 0.991 / 1.82% | 0.931 / 3.45% | 16.5% |
| cascade fastText-100k (filter @99.5%) → Ettin-17m | | 1.87% | 0.992 / 1.91% | 0.930 / 3.72% | 17.7% (half-split tuned, held-out half) |
| BERT-mini (1.94M) | 11.2M | 2.95% | 0.990 / 2.81% | 0.916 / 5.28% | 28.2% |
| fastText full (lr .1, 10 ep) | 63M | 4.09% | 0.992 / 3.91% | 0.932 / 7.11% | 27.1% |
| fastText pruned 100k | 100k | 4.28% | 0.992 / 4.36% | 0.932 / 7.58% | 27.4% |
| sparse linear L1 1e-6 (300k docs) | 14.4k | 7.59% | 0.988 / 7.17% | 0.922 / 11.96% | 43.5% |
| fastText pruned 10k | 10k | 9.72% | 0.991 / 9.56% | 0.932 / 14.8% | 48.0% |

- Secondary metric (recall at 1% FPR): Ettin-17m .976, cascade .979 (test), BERT-mini .959, fastText .945 / pruned 100k .937 / 10k .868, L1 14k .893.
- Data scaling: FPR@99 falls with data for all models (Ettin-17m 5.4% at 30k → 1.8% at 1.94M).
- Site generalization: domain-disjoint run `wildai_1m_strat_dom` matches or beats the shared-site results.
- Confounded vs stratified data at about 300k docs: 3–10× worse FPR@99 on the confounded sample.
- By year: 2024 ≈ 2025. Dev has 122 AI and 1 human page from 2026. 2026 natural pages need far more FPR for 99% recall, rising through the year (Ettin-17m: Feb 7.7% → Jun 28.9%).
- Mixing ablation (mix_v4, FPR needed for 99% on the slice):
  - Ettin-17m, WildAI-only vs mixed: meld-eval 28.2% vs 7.8%; natural 22.3% vs 27.1%; DseWiki about 100% vs 100%.
  - fastText: mixing hurts on meld-eval and natural.
- Hyperparameters (confounded run, within-run comparisons):
  - encoders: lr 1e-4, batch 32; 2 epochs help BERT-mini at 99% recall; max_len 512 rather than 256;
  - fastText: lr 0.1 × 10 epochs;
  - dense linear: alpha 1e-6, 10 epochs is enough;
  - the missed-AI penalty helps only the dense linear model.
- Cost estimate for 50T tokens (≈50B docs at ~1k tokens), unbenchmarked beyond the dev-set timing:
  - fastText stage: ~2,500 CPU-h.
  - Ettin stage at the 2026-tuned operating point (49% of docs pass the filter): ~11–13.5k H100-h as measured (Python serving), or ~1.4–3.3k H100-h with optimized serving (estimate).

## Open issues / traps
- 2026 gap: dev-tuned thresholds give about 93% recall on 2026 pages. Newer crawls and random (rather than EditLens-preselected) sampling are confounded in this data. Fix: 2026-era training data.
- DseWiki: the WildAI-only Ettin ranks agent pages well (AUROC 0.906), but nothing reaches 99% recall there without flagging nearly all human pages. Small set (193 AI / 67 human). Needs unfiltered-crawl training data or its own threshold.
- Linear models: ≥ about 700k docs exceed 2^31 nonzeros and sklearn SGD rejects int64 indices (the 1M and mix_v3/v4 linear rungs failed). Fix: chunked `partial_fit`.
- Traps hit:
  - first WildAI sampler read whole files in order, giving a crawl confound;
  - the per-class cap shrank held-out sets (now exempt);
  - dedup order moved meld-eval human seeds into RAID (list held-out sources first);
  - fastText `.ftz` probabilities saturate (use folded `prune=N`);
  - a local import shadowed `parse_spec` (UnboundLocalError);
  - one-class `train_groups` selection crashed;
  - `srun --overlap` without `-n 1` runs once per CPU;
  - editing a bash watcher script while it runs breaks it.
- Beaker: CPU clusters (phobos, hammond) are single 180-CPU nodes where big jobs never schedule; GPU clusters allocate a GPU even for 0-GPU jobs; session SSH is unreachable from Berkeley; jobs can't build images.

## Verified vs unverified
- Verified from stored scores: every number in the table above (dev, test, natural, by-year, mixing, cascade).
- UNVERIFIED:
  - H100 ≈ H200 speed, and the optimized-serving speed (0.2–0.4 ms/doc). Only the Python-serving timing was measured, on H200.
  - ~1k tokens/doc for the 50T estimate (dev mean ~3.9k chars).
  - That the natural slice represents raw web pages (it is already FineWeb-style extracted English).
- Labels are Pangram's: "AI" means Pangram calls it AI.
