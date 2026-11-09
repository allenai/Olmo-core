---
experiment: pail_ccpeek
title: A few Common Crawl WARC files through resiliparse, scored with the tinyclf classifiers
status: done
updated: 2026-10-19
branch: main (PAIL e5b80e3)
launcher: scripts/tinyclf/beaker.sh (steps scripts/download/commoncrawl/cc_peek.py, scripts/tinyclf/score_docs.py)
records:
results:
  - s3://ai2-llm/checkpoints/prasanns/PAIL/commoncrawl/peek_2025-30_2026-25/extracted/ (jsonl.gz per WARC, peek.json, samples.jsonl)
  - s3://ai2-llm/checkpoints/prasanns/PAIL/commoncrawl/peek_2025-30_2026-25/scores/ (scores.jsonl.gz, summary.json, examples.jsonl)
wandb_group:
next_step: superseded by pail_ccdev (the full dev set)
---

# pail_ccpeek

## Question
What does resiliparse-extracted Common Crawl look like (WARC streamed, never stored), and how do the
tinyclf classifiers (pruned fastText-100k, Ettin-17m, both from run wildai_1m_strat) behave on it?

## Decisions
- 2026-10-19 (Prasann): run on Beaker, only a few WARC files; stream the WARC through resiliparse
  (process_warc in src/download/commoncrawl.py already does: urlopen -> fastwarc ArchiveIterator -> extract_plain_text).
  Beaker jobs set minRuntime (beaker.sh --min-runtime, default 1h).
- 2026-10-19 (Claude): 2 WARC files each from CC-MAIN-2025-30 and CC-MAIN-2026-25 (seed 20261019); CPU job
  (phobos/hammond, 8 CPU, 32 GiB); Ettin-17m on CPU on 4,000 random docs per crawl; thresholds = 99% recall
  on dev (dev99) and on the natural slice (nat99).

## Runs
- ccpeek-10192334, Beaker 01M5B89664Q9MWCQNX8EVYZJDS: failed at once, fastwarc 1.x panics calling seek() on the urllib stream (fixed in 66283a1, ForwardStream).
- ccpeek-10192340, Beaker 01M5B8K8X926PNZ2827DMSD3S3 (relaunch with the fix).

## Current best numbers
4 WARC files (2 x CC-MAIN-2025-30, 2 x 2026-25): 84,668 docs, ~35% English, 16% under 200 chars. Flagged share of English
docs: ettin17m @dev99 6.7% (2025) / 11.6% (2026); ft100k @dev99 11.2% / 16.2% (summary.json). Superseded by pail_ccdev.

## Next step
Same as next_step above.

## Open issues / traps
- The pruned fastText rung stores only (row, weight); score_docs.py rebuilds it with the unpruned rung's
  dictionary (rank-1 trick) and checks against the stored dev scores.

## Verified vs unverified
- Verified: process_warc streams (code read 2026-10-19).
- Verified: the unpruned rung's dictionary reproduces the pruned rung's stored dev scores (max diff 4.6e-5).
