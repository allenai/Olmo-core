---
experiment: pail_ensemble
title: Ensemble labels (Pangram/WildAI + MELD + ProseLens + EditLens + tinyclf) on ccdev crawl text to retrain unified classifiers and build a high-recall cascade
status: active
updated: 2026-10-23
branch: main (ensemble merged 2026-10-22); v2 work on branch ensemble-v2 (pushed, not merged; worktree /accounts/projects/berkeleynlp/prasann/projects/PAIL-ensemble2)
launcher: scripts/tinyclf/beaker.sh (1-GPU Beaker jobs), scripts/beaker_shell/bkx (warm session `ens`)
records:
  - /accounts/projects/berkeleynlp/prasann/projects/PAIL/docs/classify/ensemble_v1.md (results record, on PAIL main ee0ef62)
  - /accounts/projects/berkeleynlp/prasann/projects/PAIL/scripts/classify/ensemble/README.md (pipeline + reproduce)
results:
  - s3://ai2-llm/checkpoints/prasanns/PAIL/ensemble/eval/teacher.json (all members, clean students, stackers, cascades; eval sites)
  - s3://ai2-llm/checkpoints/prasanns/PAIL/ensemble/silver_v1/ (163,547 ai / 1,484,900 human, train sites)
  - s3://ai2-llm/tinyclf runs ens_wildai_clean (control students)
  - s3://ai2-llm/checkpoints/prasanns/PAIL/tinyclf/corpora/ens_wildai_clean/ (WildAI-1M minus eval sites: 506k AI / 523k human)
  - s3://ai2-llm/checkpoints/prasanns/PAIL/ensemble/eval/baseline.json (evaluate.py, existing scores, eval sites)
  - s3://ai2-llm/checkpoints/prasanns/PAIL/commoncrawl/ccdev_v1/scores/proselens/ (in progress)
artifacts:
  - https://claude.ai/artifact/4iFiw9prUPAgTMLsxW7eYB (summary page, private)
wandb_group:
next_step: Ettin-17m v2 on all 3.78M docs (ens-student-v2-ettin-b running); then try a larger student (Ettin-68m) as the expensive end and per-format weighting; EditLens waits on Pangram approving HF access
---

# pail_ensemble

## Question
Can an ensemble of detectors, calibrated on WildAI's Pangram labels, label enough real Common Crawl text (ccdev_v1) to
retrain small "unified" classifiers that keep very high recall on crawl text, with precision recovered at the expensive
end of a cascade? Everything is measured on ccdev (the real dev set), on sites never used for labeling or training.

## Decisions
- 2026-10-21 (Prasann): build a bigger training set from an ensemble (MELD, WildAI/Pangram, etc.), retrain unified
  classifiers, iterate toward a pipeline with very high recall and solid precision at the expensive end.
- 2026-10-21 (Prasann): evaluate on the real dev set (ccdev_v1).
- 2026-10-21 (Prasann): run on Beaker wherever possible, mostly 1-GPU jobs; not jsteinhardt.
- 2026-10-22 (Prasann): OK to create the PRASANNS_HF_TOKEN Beaker secret (EditLens joins); staying on Beaker despite queue waits is fine.
- 2026-10-21 (Claude): ccdev split by site (URL host minus www., sha1 hash < 0.5 = eval); DseWiki always eval
  (`src/classify/ensemble.py`). Stackers and students only ever see train-site documents.
- 2026-10-21 (Claude): ensemble members: Pangram labels (WildAI, gold where present), ProseLens (Pangram-distilled
  30B-A3B), MELD (1B), EditLens (Pangram's open 3B; needs an HF token on Beaker), tinyclf ettin17m and ft100k.

## Plan
1. Score the ccdev labeled docs + the 2% random-English sample with ProseLens (+ EditLens): every member on the same docs.
2. Fit a stacker on train-site Pangram labels; compare to single members on eval-site gold (natural selection, DseWiki).
3. Scale the teacher over a large train-site pool of crawl English; hard silver labels where confident.
4. Train students (fastText, Ettin-17m, larger Ettin) on WildAI 1M + ccdev silver; score eval sites.
5. Cascades: cheap filter at ~99.5% recall -> student -> expensive end (ProseLens/stacker) for precision; report
   recall vs Pangram, FPR, random-English flag rate, cost per doc.

## Current best numbers
Eval sites of ccdev_v1 (6.15M docs; site-hash split). Thresholds set on train-site WildAI natural AI docs (r0.99 = keep 99%
of them). eval_size: natural 751 AI / 3,576 human; DseWiki 398 agent / 150 human (⚠ human < 500); random English 41,077.
Source: s3://.../ensemble/eval/baseline.json. ⚠ ettin17m/ft100k trained on WildAI, which may include eval-site pages
(ens-corpus-wildai-clean: 49.6% of WildAI-1M docs are on ccdev eval sites; by record id only 424 of 20,752 eval-site
labeled ccdev docs are in it), so their eval-site numbers may be optimistic through site overlap.

| model @ cut | natural recall | natural FPR | DseWiki agent / human | random-EN flagged | AUROC natural / DseWiki |
|---|---|---|---|---|---|
| ettin17m @r0.99 | 99.1 | 30.9 | 97.7 / 83.3 | 50.1 | 0.979 / 0.848 |
| meld @r0.99 | 99.5 | 61.2 | 100 / 98.7 | 80.3 | 0.946 / 0.910 |
| ft100k @r0.99 | 98.8 | 92.0 | 93.7 / 100 | 83.8 | 0.952 / 0.479 |
| stack ettin17m+meld @r0.99 | 99.6 | 28.6 | 100 / 78.7 | 52.3 | 0.982 / 0.929 |
| stack ft100k+ettin17m+meld @r0.99 | 99.5 | 27.4 | 100 / 80.7 | 49.3 | 0.982 / 0.905 |
| cascade ft100k@r0.995 -> ettin17m@r0.99 | 98.5 | 30.5 | 97.2 / 83.3 | 48.6 | |

ProseLens smoke (3 WildAI WARC files, 1,444 Pangram-labeled docs, 390 AI; ⚠ natural only 172 / 29 AI): AUROC 0.994 vs
ettin17m 0.991, meld 0.972; flags 6.3% of 448 random English docs at p_ai >= 0.5. 50-63 docs/s on one H100 (vLLM
0.21 cu129), ~940 prompt tokens/doc.

Full eval sites (teacher.json, 2026-10-22; same eval_size as above). Cut = keep 99% of train-site natural Pangram AI.
| model @r0.99 | natural recall | natural FPR | DseWiki agent / human | random-EN flagged | AUROC natural / DseWiki |
|---|---|---|---|---|---|
| proselens | 98.9 | 21.4 | 100 / 43.3 | 49.2 | 0.987 / 0.948 |
| stack proselens+ettin17m | 98.9 | 16.2 | 99.7 / 34.7 | 42.4 | 0.991 / 0.977 |
| stack proselens+ettin17m+meld | 99.6 | 16.9 | 100 / 32.7 | 43.3 | 0.991 / 0.988 |
| ens_clean_ettin17m (control) | 98.4 | 26.1 | 47.2 / 27.3 | 40.8 | 0.977 / 0.672 |
| ens_clean_ft100k (control) | 98.8 | 90.2 | 96.7 / 100 | 81.2 | 0.949 / 0.484 |
| cascade ft100k@.995 -> ettin17m@.995 -> proselens@.99 | 97.9 | 15.0 | 99.2 / 42.7 | 38.3 | |
| cascade clean ft@.998 -> clean ettin@.998 -> proselens@.99 | 98.4 | 17.8 | 96.5 / 32.0 | 42.2 | |
At r0.98 the stack has FPR 12.8 (random-EN 37.3); proselens alone at f0.01 keeps 81.1% recall. Reading: the last ~1-2%
of Pangram-AI natural docs are missed by every detector; 99% Pangram recall costs flagging ~40-50% of crawl English.
Clean control matches the old ettin17m on natural text (no visible site-leak inflation) but loses DseWiki (0.85 -> 0.67).
silver_v1 students on eval sites (silver_v1.json; corpus = clean WildAI + silver_v1, 974k + 1.65M docs):
| model @ cut | natural recall / FPR | random-EN | AUROC natural / DseWiki |
|---|---|---|---|
| ens_silver_ettin17m @r0.98 / @r0.99 | 98.1 / 25.4 ; 99.5 / 38.9 | 39.6 ; 52.6 | 0.979 / 0.847 |
| ens_clean_ettin17m @r0.98 / @r0.99 | 98.1 / 24.1 ; 98.4 / 26.1 | 37.9 ; 40.8 | 0.977 / 0.672 |
| ens_silver_ft100k @r0.95 / @r0.99 | 95.6 / 17.1 ; 98.8 / 95.9 | 24.9 ; 87.8 | 0.959 / 0.310 |
| cascade silver ettin@.995 -> proselens@.99 | 98.7 / 16.4 | 39.8 | |
| cascade silver ft@.995 -> silver ettin@.995 -> proselens@.99 | 97.9 / 15.5 | 35.7 | |
Reading: silver recovers DseWiki ranking (0.67 -> 0.85) and raises agreement with ProseLens on random crawl (precision
vs teacher at r0.95: 87% vs 80%) but does not move Pangram-natural AUROC; fastText gets worse at high recall.
v2 fastText (ens_v2_ft100k; clean WildAI + silver_v1 + WildAI natural, 3.78M train docs; v2ft.json): r0.95 94.4 / 21.0,
r0.98 97.1 / 67.6, recall@f0.01 70.7 (orig ft100k 60.5), AUROC natural 0.953 / DseWiki 0.185. More data helps fastText
only at the low-FPR end; past ~95% recall its FPR explodes for every training set, so it cannot be a 99%-recall filter.
v2 Ettin-17m (ens_v2_ettin17m_2m5, first 2.5M of ens_v2; v2.json): r0.95 95.3 / 9.4, r0.98 98.5 / 24.9, r0.99 99.5 / 40.5;
AUROC natural 0.981 (best student) / DseWiki 0.705. Stack proselens+v2 ettin r0.99 99.5 / 20.2. Best fully clean cascade:
v2 ettin @r0.995 -> stack(proselens+v2 ettin) @r0.99 = 99.2 / 17.3, DseWiki 97.7 / 42.0, random-EN 40.2.
WildAI natural on train sites (wildai_natural_train): 228,186 ai / 1,080,781 human kept (of 468,510 / 2,202,941).
Recall evenness (dump_v1, eval-site Pangram AI: 5,195 docs on 4,525 sites, 111 sites with >= 3; thresholds = overall
recall quantiles on eval AI; ⚠ mostly editlens_ai docs). Stack: misses spread one per site (52 misses on 50 sites at
99%; dispersion 0.3x binomial; no site fully missed). fastText (orig and clean): misses cluster by site at 99%
(dispersion 8-9x; clean ft misses all of kiltkingdom.com 0/3, sfbay-automaticgates.com 0/3, therunwire.com 0/4).
Multi-AI-page sites are caught more fully than even recall predicts (all-caught 82-86% vs 49% expected at 80%).
By format at 95% overall recall, every model: Product Page ~92 (n=1,462), About ~92-93, News ~91-95 vs Listicle/Blog
98-99. By topic fastText spread is wider (Software Dev./Fashion/Food ~90-91 vs stack 92.6-96.2).
silver_v1 (train sites, stack proselens+ettin17m, p>=.9 ai / <=.1 human): 2,086,594 candidates, 1,648,447 kept;
silver 158,375 ai / 1,469,927 human; gold 5,172 ai / 14,972 human (natural gold held out of training).

## Open issues / traps
- ens-student-v2-ettin OOM (rc=-9) at 128 GiB on 3.78M docs: train.py tokenizes the whole train split; use 200 GiB or n_train <= 2.5M.
- Beaker preempts urgent jobs soon after their min-runtime (ens-student-v2: 4 restarts at ~1-1.5 h). train.py does not
  checkpoint inside a rung, so a rung longer than min-runtime never finishes: pass --min-runtime >= the rung's time
  (Ettin-17m on 3.78M docs ~3 h -> 4h), or add mid-rung checkpointing.
- vLLM 0.21's PyPI wheel is built for CUDA 13; jupiter's driver is CUDA 12.8. Use the cu129 release wheel + torch cu129
  (`beaker.sh --vllm`). score.py initialises CUDA first, so vLLM must spawn (set in proselens.py).
- Beaker 2026-10-21/22: jupiter near-full (100+ urgent ahead); 1-GPU jobs waited 1-2 h. titan accepts only
  min-runtime-0 filler from ai2/flex2.
- ops status shows NO-JOB-FOUND for runs registered with a Beaker experiment id that has not scheduled yet.
- EditLens blocked: HF account PrasannSinghal gets 403 on gated pangram/editlens_Llama-3.2-3B (Llama-3.2-3B itself is fine; beaker.sh passes HF_TOKEN from PRASANNS_HF_TOKEN). Needs Pangram to approve access. The peft/TransformerEngine import crash is fixed (419287e).
- bkx sessions used --min-runtime 0 (never schedules on jupiter); fixed to default 1h (85ea4e4).

## Verified vs unverified
- Verified (2026-10-21): MELD and ProseLens HF repos download anonymously; EditLens needs a token (ours has access).
- UNVERIFIED: ProseLens vLLM throughput on H100; that Pangram labels on FineWeb-filtered docs transfer to raw crawl pages.
