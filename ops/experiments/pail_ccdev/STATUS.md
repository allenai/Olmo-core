---
experiment: pail_ccdev
title: ccdev_v1, a Common Crawl dev set (resiliparse text) with DseWiki and WildAI labels and classifier scores, for cascades on real data
status: done
updated: 2026-10-19
branch: main (ccdev merged 2026-10-20; worktree removed)
launcher: scripts/tinyclf/beaker.sh (steps: scripts/download/commoncrawl/ccdev.py plan|extract, scripts/classify/ccdev/{meta,score,merge,reproduce,dsewiki_check}.py)
records:
results:
  - s3://ai2-llm/checkpoints/prasanns/PAIL/commoncrawl/ccdev_v1/ (manifest, text/, meta/, scores/, ccdev_v1.metadata.parquet, card)
  - s3://ai2-llm/checkpoints/prasanns/PAIL/commoncrawl/ccdev_v1/checks/dsewiki/ (report.json, pages_scored.jsonl)
  - s3://ai2-llm/checkpoints/prasanns/PAIL/commoncrawl/ccdev_v1/checks/eval/report.json (evaluate.py, all classifiers + cascades)
  - s3://ai2-llm/checkpoints/prasanns/PAIL/commoncrawl/ccdev_v1/checks/preproc{,_wildai}/ (preprocessing ablation)
wandb_group:
next_step: use ccdev_v1 to tune cascade thresholds on real crawl text (evaluate.py), retrain/recalibrate on 2026 + unfiltered-crawl data; DseWiki needs its own signal
---

# pail_ccdev

## Question
A dev set of real Common Crawl documents (streamed WARC, resiliparse text, < 50 GB of text) where DseWiki agent/human
pages and WildAI-labeled documents are tagged among ordinary pages, with a metadata document (locator + labels +
classifier scores) from which anyone can rebuild the AI-text subset. First: does MELD flag the DseWiki agent pages on
resiliparse text?

## Decisions
- 2026-10-19 (Prasann): label sources DseWiki, WildAI, MELD (check DseWiki first); preprocessing allowed if recorded
  and lossless of important content; < 50 GB text; Beaker GPU as needed but nothing very long; own worktree.
- 2026-10-19 (Claude): whole WARC files as the unit: all 367 DseWiki-agent files + 150 DseWiki-human files (WildAI
  crawls) + 8 random files per WildAI raw-WARC crawl (12 crawls) from the files WildAI itself took (labels join on
  WARC-Record-ID). ~600+ files, ~13M docs, ~40-45 GB text (peek: ~70 MB text, ~21k docs per WARC).
- 2026-10-19 (Claude): no global text preprocessing; each classifier's own input handling is recorded in the card
  (src/classify/ccdev.py CLASSIFIERS). ft100k on all docs, ettin17m on English, MELD on English labeled docs + 2%
  id-hash sample (MELD is 1B params, fp32 by its contract).
- 2026-10-20 (Claude, from evidence): labeled documents are scored by every classifier whatever their language (the
  English-only subset left most DseWiki pages unscored); no text preprocessing (ablation: no AUROC gain).
- 2026-10-19 (Claude): `length` in extracted records is now the compressed record length (CC index semantics);
  checked 40/40 round trips (range fetch + re-extract gives identical text).

## Runs
- ccdev-plan: manifest had WildAI's abbreviated keys (no segments/); fixed, rerun as ccdev-plan2 (613 files: 367
  dsewiki_agent, 150 dsewiki_human, 96 wildai; 42,339 WildAI label rows).
- ccdev-extract-0..3 on CPU clusters: 1-3 never scheduled (phobos/hammond full: hammond 1 node, 38 jobs); rerun as
  g1-g3 on GPU clusters (1 GPU, 24 CPU, 16 workers, ~10 min each). 609/613 files done; 4 hung on a pathological page
  (resiliparse superlinear), fixed in a6e33a2, ccdev-extract-cleanup (01M5BGPQCECJ1VS5N6KS0050TC) does the rest + meta.
- ccdev-meta-0/1: 12.24M docs, 4.15M English (detect_fast), 545/548 DseWiki captures, 41,742/42,339 WildAI labels matched.
- ccdev-score-0..7 (H100, minRuntime 1h): ft100k ~4.8k docs/s (CPU), ettin17m ~930 docs/s, MELD ~15 docs/s; ~80-100 min each.
- ccdev-dsewiki-check (A100, 4 min): see numbers below.

## Current best numbers
ccdev_v1: 12,294,285 docs (613 WARC files), 4.17M English; text 20.8 GB gz (~41 GB raw); metadata 1.33 GB.
reproduce.py: 516/516 sampled labeled-AI docs rebuilt with matching text_sha1.
Full table: scripts/classify/ccdev/README.md "First results" (source checks/eval/report.json). Headlines (%; rand_en =
flagged share of 82,436 random unlabeled English docs; WildAI natural n = 1,456 AI / 7,311 human; DseWiki 398 / 150):
| classifier / cascade | rand_en | DseWiki agent rec / human FPR | WildAI natural rec / FPR |
|---|---|---|---|
| ettin17m @dev99 | 10.7 | 0 / 0 | 80.6 / 2.3 |
| ettin17m @nat99 | 29.4 | 70.1 / 17.3 | 96.1 / 13.3 |
| meld @fpr_0.01 (shipped) | 29.1 | 65.6 / 10.0 | 89.2 / 12.9 |
| ft100k @nat99 -> ettin17m @nat99 | 19.3 | 37.7 / 12.0 | 92.9 / 9.1 |
| ettin17m @nat99 -> meld @fpr_0.01 | 19.4 | 46.5 / 6.7 | 87.4 / 6.2 |
AUROC WildAI natural: ettin17m 0.979, ft100k 0.952, meld 0.944. Preprocessing (ws, noshort): no AUROC gain -> raw text.

DseWiki check on resiliparse text (latest capture per page; 396 agent, 143 human, 85 agent with natural language;
s3://.../ccdev_v1/checks/dsewiki/report.json). n_human = 143 (< 500: wide intervals).
| classifier | AUROC | agent / NL-agent / human flagged @ threshold |
|---|---|---|
| MELD | 0.936 | 75% / 73% / 9.8% @ shipped fpr_0.01; 100% / 100% / 41% @ fpr_0.05 |
| ettin17m | 0.887 | 0 / 0 / 0 @ dev99; 79% / 71% / 17% @ nat99 |
| ft100k | 0.609 (NL 0.44) | 58% / 39% / 34% @ dev99 |
Earlier, on dsewiki_cc's own page text: MELD AUROC 0.77 (51% agent / 19% human at fpr_0.01).
Caveats: some "human" captures are the wiki's preferences form; agent pages carry the English wiki menu, human pages
the German one (resiliparse keeps it), so part of the separation may be language.

## Next step
Same as next_step above.

## Open issues / traps
- fastwarc 1.x calls tell()/seek() on its input; the HTTP stream is wrapped in ForwardStream (src/download/commoncrawl.py).
- The pail env on mooney lacked fastwarc/resiliparse (installed 2026-10-19); other nodes may too.
- WildAI's FineWeb-era crawls (2024-10..2025-26) have warc_path = null; only its 12 raw-WARC crawls give file keys,
  and those omit the segments/<id>/warc/ part (resolved through warc.paths by file name).
- WildAI `selection`: editlens_ai 36.2M, editlens_human 57.0M, natural 2.8M. Only `natural` is a crawl-rate sample.
- resiliparse extract_plain_text can run unboundedly on some pages (5.2 MB page in CC-MAIN-2026-25); pages >= 1M chars
  now extract in a forked child with a 30 s limit (skip reason 'timeout').
- Beaker CPU clusters are full; CPU work goes to GPU clusters (a 0-GPU job still holds a GPU there).
- ops status shows these Beaker runs as NO-JOB-FOUND (it seems to need the job id, not the experiment id).

## Verified vs unverified
- Verified: record locate/rebuild round trip (mooney, 40/40); folded fastText reproduces stored dev scores (4.6e-5);
  forked extraction == in-process; DseWiki and WildAI joins (545/548, 41,742/42,339 before the 4 missing files).
- Verified: reproduce.py end to end (516/516 sha1 match).
- UNVERIFIED: that WildAI's ~1.4% unmatched labels are extraction skips (not id mismatches); MELD flag rates on the 2%
  random sample generalize (n = 82k, fine) but its DseWiki numbers rest on 150 human pages.
