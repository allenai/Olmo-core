---
experiment: pail_pre2020
title: Pre-2020 human text that Pangram 4.1 calls >= 75% AI, with proof from Common Crawl
status: active
updated: 2026-11-13
branch: PAIL prasann/pre2020-fp (worktree /accounts/projects/berkeleynlp/prasann/projects/PAIL-pre2020), scripts/classify/pre2020/
launcher: scripts/tinyclf/beaker.sh (harvest: --gpus 0 --cpus 16; ranking: --gpus 1)
records:
results: s3://ai2-llm/checkpoints/prasanns/PAIL/pre2020/v1/ (passages/, rank/, pangram/, proofs)
artifacts: https://claude.ai/artifact/U8tdBvrbWe1EDAZQFoK2wj (PAIL Text Workbench, "Challenge" tab = challenge.html; v222 first 20 candidates)
wandb_group:
next_step: Prasann decides: genre-targeted harvest (content mills / press releases / essay mills via cc-index), a small Pangram budget to calibrate the proxies on old text, or stop
---

# pail_pre2020

## Question
Is there text that (1) runs >= 750 consecutive words, (2) Pangram's latest model (4.1) calls >= 75% AI, (3) provably
existed unmodified in 2019 or earlier (a Common Crawl capture), and (4) is ostensibly human (not GPT-2, machine
translation, Markov text or repetition; no HTML)? Prasann asked for candidates in an artifact tab he can copy from.

## Decisions
- 2026-11-13 (Prasann): find such text; use fasttext / our classifiers / others to find candidates; show them in a tab.
- 2026-11-13 (Claude): the repo had no pre-2020 pool (68 FineWeb CC-MAIN-2019-35 docs, Pangram 3.3.2 called none AI), so
  harvest fresh: 375 WARC files, every main crawl 2013-20 .. 2019-51 (8 files each 2017-2019, 3 each before), pail_v9
  extraction, longest prose run per page >= 750 words, English and quality filters. Proof = re-fetch by byte range,
  exact passage, WARC-Date < 2020, Common Crawl index match, earliest Wayback capture (verify.py).
- 2026-11-13 (Claude): pre-filter with our Pangram-4.1 imitators from p41_aug_s (mmBERT-base rung on all passages, a
  Qwen3.5-4B window rung on the top), then Pangram 4.1 (--model pangram-4.1 --require-version 4.1) on the top ~1,000.
- 2026-11-13 (Prasann): spend NO Pangram credits (the key was out of credits anyway; nothing was scored); he pastes candidates
  into Pangram by hand. Tab name "Challenge". Time-sensitive competition: aim for maximum precision, one hit suffices;
  aggressive filtering from much more text is fine.
- 2026-11-13 (Claude): cascade per shard: mmBERT-base (seed234 filter) on all -> top 2,000 (8,000 for pool b) -> Qwen3.5-4B
  (#4) -> top 300 -> 4B and 9B window logits -> trim to the >= 760-word run of whole paragraphs the two models score most
  AI -> 4B and 9B on the exact trimmed text -> rank by the worse of the two rank percentiles (cascade.py merge).
- Smoke (4 files, 2013-2014 crawls): 215k pages -> 3,843 passages (~960 per file), 5.6 min with 4 workers.

## Current best numbers
| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|

## Results so far
| arm | metric | value | eval_size | source |
|---|---|---|---|---|
| pool a (375 files, 2013-2019) | passages / finals | ~170k passages; 1,168 trimmed finals | | rank/0..3 |
| pool a finals | 4B > 0 / 9B > 0 / both | 2 / 0 / 0 (max 4B 1.07, 9B -0.46; dev: ~7% of pages > 0, AI pages ~3 to 6) | 1,168 | rank/*/final.jsonl.gz |
| proofs (pool a top 20) | all checks pass (exact passage, sha1, date < 2020, record id, CC index) | 20 / 20 | 20 | final/proofs0.jsonl |
| pool a+b finals (16 shards, ~1.6M passages) | 4B > 0 / 9B > 0 / both | 25 / 5 / 1 (max of min(4B,9B) 0.05) | 4,670 | rank/*/final.jsonl.gz |
| Prasann's hand check | top 2 proxy candidates in Pangram | both Human | 2 | 2026-11-13 |
| dev check (25k, p41 dev) | P(Pangram AI share >= 75%) at min(4B,9B) >= 0 / 2 / 4 / 6 / 8 | 18% / 23% / 29% / 41% / 72% (n 1596 / 1215 / 867 / 518 / 144) | 25,000 | pre/dev.jsonl.zst + scores.npz |
Dev check reading: the proxies predict "any AI" (Mixed counts), not "Pangram >= 75%"; even at the top cut 1 in 4 misses,
and on pre-2020 text (base rate ~0) the same score means far less. Proxy ranking dropped as the selector.
- 2026-11-13 (Claude, after Prasann's two Human results and his dev-check suggestion): approach 2, oldcopy.py: take
  the 6,693 pages in the 600k labels with a Pangram AI-window run >= 700 words (3,540 labelled AI), find pre-2020 captures
  of the same URL in Common Crawl's columnar index (65 crawls 2013-2019), fetch them, and keep >= 750 words shared exactly
  with the flagged span. Pangram's own windows say which text it calls AI; no credits.
| old-copy approach, round 1 | targets 6,693 (4.1) + 12,817 (WildAI); URLs with a pre-2020 capture 88; >= 750 identical words with the flagged span 1 | 1 candidate (tmhcorp.com, WildAI's older Pangram only) | | oldcopy/candidates_b.jsonl.gz |
| Prasann's hand check | tmhcorp.com passage in Pangram 4.1 | Human (the older Pangram's false positive is fixed in 4.1) | 1 | 2026-11-14 |
- 2026-11-14 (Prasann): Mixed pages are fair game. (Claude): round 2 of the old-copy approach uses every 4.1 AI / Mixed page
  (34,882; >= 563 AI words in total) and any >= 750-word shared window with >= 75% AI-Generated coverage (oldcopy.py
  targets_all, compare coverage mode). In parallel, pool C (5,400 more 2017-2019 WARC files, ~5M passages) through the
  cascade, reported only where both 4B and 9B score >= 2 on the exact passage (dev: 11% Pangram-Human at that cut).
| old-copy round 2 (every 4.1 AI / Mixed page) | 11,072 targets -> 127 URLs with pre-2020 captures -> 0 with >= 750 shared words; 4 with 500-742 | shared text 0-1% AI-Generated in every case (the AI spans are the post-2019 rewrites) | 11,072 | oldcopy/near500_proofs.jsonl |
| pool C (5,400 files 2017-2019, ~5M passages) + all pools (~6.6M) | finals with 4B >= 2 and 9B >= 2 | 0 (both > 0: 2; best min 0.64: 99pinestreet.com 4B 0.64 / 9B 3.25) | 10,533 finals | rank/*/final.jsonl.gz |
Overall reading (2026-11-14): no route finds pre-2020 long prose that Pangram 4.1 likely flags; its false positives there look
extremely rare. Workbench Challenge tab v227 shows the 4 near misses (pre-2020 text, proofs pass) as near misses, not candidates.
Reading: our best Pangram-4.1 proxies find almost no pre-2020 prose that looks AI; hits will be rare, so pool size matters.
Traps: the loop's 4B/9B need --image akshitab/olmo-core-tch2130cu129-sm80-2026-09-11, peft 0.21.2, flash-linear-attention
0.5.2 and the causal_conv1d wheel; window_logits ignored sbs (9B OOM, fixed in score_pages.py); verify.py must not send a
contact email and runs on Beaker (Wayback lookup opt-in).

## Next step
See next_step above.

## Open issues / traps
- Old pages: nav text can run into the first prose line; finalists get trimmed to a clean >= 750-word sub-span (still
  consecutive, unmodified text).

## Verified vs unverified
- Verified (2026-11-13): harvest smoke read by hand (4 passages).
- UNVERIFIED: everything after harvest.
