# v2 proxy data: pool `mtb16k` + held-out `heldout_v2` (built 2026-11-17)

Built by `build_pool_v2.py` (gen / tokenize / assemble / order) and `build_heldout_v2.py`, on mooney.
- **Pool:** `mooney:/data/prasann/early_loss_proxy/shards/mtb16k`, in the same layout as mt4f1952. It holds 7,808 rows; t0..t3 = contradiction, outlier, reorder, qdmatch_nq.
- **Stats:** `shards/mtb16k_stats.json` and `shards/mtb16k_order_seed0.json`.
- **Held-out:** `mooney:/data/prasann/early_loss_proxy/heldout_v2.npz` (360 rows) and `heldout_v2_stats.json`.
- **Build inputs:** `/data/prasann/early_loss_proxy/v2build/` holds the generated JSONL (`gen/`), the per-cell shards (`tok/`) and `n_docs_used.json`.

## Generation and tokenization
- **Generator:** the vendored `ctc` package run as `recipes/data/ctc_sft/ctc_calibrated.py build --split train --seed 42`, sized like the IID setA build. Documents per row match the eval's own count (`recipes/data/ctc_sft/eval_calibration.json`), except the refit 16k cells listed below.
- **Seed pools:** contradiction and reorder come from the Hub; outlier and qdmatch_nq come from `/data/prasann/ctc_seed_pools`.
- **Tokenization:** `convert_unified_to_document_landmark.py --emit dense --marker-set qwen3_5 --query-position after --cot-mode none --chunk-by document`, with the Qwen3.5-0.8B-Base tokenizer.
- **Row length:** at most 16,384 tokens (pool maximum 16,378).
- **Row order:** rows are shuffled within each task (seed 20261116).

## Rows (pool: 651 / 651 / 650 per task at 4k / 8k / 16k)
| task | docs per row 4k / 8k / 16k | mean length (min-max) at 4k / 8k / 16k | answer tokens | rows dropped (gen -> kept) |
|---|---|---|---|---|
| contradiction | 92 / 187 / **324** (eval 379) | 4140 (3357-5006) / 8365 (7197-9565) / 14498 (12827-15941) | 25-30 | none over 16,384 at 324 docs (850/1000 were over at 379) |
| outlier | 28 / 57 / **96** (eval 111) | 4308 / 8635 / 14473 (13560-15435) | 16-17 | 1 gold-in-eval at 16k (2104/2400 were over at 111) |
| reorder | 27 / 57 / **90** (eval 116) | 4402 / 9184 / 14406 (12870-16378) | 101 / 221 / 353 | 16k: 15 over 16,384. Gold-in-eval: 3 at 8k. Same book as an eval row: 13 / 8 / 14 |
| qdmatch_nq | 40 / 84 / 174 | 3823 / 7734 / 15823 | 25-28 | ~1,200 of 2,000 per cell gold-in-eval (see caveats) |

**How the 16k refit was chosen:** a line was fit through the tokenized 4k and 8k mean lengths, and the document count was set so the mean comes out near 14.5k tokens (over-length drops in the final pool: 15, reorder@16k only): contradiction 44.5 tok/doc -> 324 docs; outlier 149.2 -> 96; reorder 159.2 -> 90.

## Gold sidecar (verified)
- **`gold_fingerprints.json`** is the format `--st-gold-sidecar` reads: `{content_fingerprint: sorted 0-based document indices}`. The fingerprint is `content_fingerprint` of the row up to and including EOS.
- **`gold_pairs_fingerprints.json`** holds 0-based index pairs, for contradiction and qdmatch only.
- **`row_info.jsonl`** has one line per row: part, row, task, rung, source JSONL and index, n_docs, length, answer tokens, fingerprint, gold, gold_pairs, and `doc_spans`. The spans are `[start, end)` from marker 248049 to 248050 inclusive.
- **Index bases in the source JSONL:** contradiction: `gold_doc_indices` are 1-based Claim id pairs; outlier: `gold_doc_indices` are 0-based; qdmatch: `gold_pairs` are 1-based `[query, document]` positions; reorder: every document is gold
- **Check:** 40 rows per task were decoded. For every task, 40/40 rows passed all three checks: the span text matches the source, printed IDs are index + 1, and the answer IDs match gold.

## First rows a seed-0 run sees
Measured with the trainer's own `ComposableDataLoader` (global order, seed 34521). The first 800 rows break down as:
- **By task:** contradiction 188, outlier 197, qdmatch_nq 202, reorder 213.
- **By rung:** every task × rung cell gets 59-78 rows (the expected value is 66.7).

## Held-out (`suite4k` / `suite8k` / `suite16k`, 32 rows per task; npz keys `ids, mask, offsets, sets, tasks, src_idx`)
- **Source:** the first eval rows that fit in 16,384 tokens after conversion. The gold answer is the loss span.
- **Rows fitting the cap:** all 500/500 fit at 4k and 8k.
- **suite16k exceptions:** contradiction 70/500 fit; outlier 303/500 fit; **reorder 8/500 fit, so suite16k/reorder has only 8 rows**; qdmatch_nq 497/500 fit
- **suite16k lengths:** about 15.8-16.1k tokens. These rows are the short tail of the 16k eval (379 / 111 / 116 docs), so the 16k probe is biased toward shorter documents.

Eval rung files (HF `PrasannSinghal/ctc-suite-eval` at olmo-eval's pinned revision 04ab8600, 500 rows each). Use them for the generation evals:
`mooney:/data/prasann/early_loss_proxy/v2build/suite_rows/suite/{contradiction_iid,outlier,reorder,qdmatch_nq}/rung_{4096,8192,16384}.jsonl`

## Overlap (held-out vs all 7,808 training rows)
- **Prompt hash:** 0 hits everywhere.
- **Question hash:** matches every row by construction (the question is a fixed instruction), so it carries no information.
- **Held-out gold document present among a training row's GOLD documents:** 0 for every task and set.
- **Held-out gold document present anywhere in a training row:** contradiction 5 / 5 / 9 rows: eval gold claims appear as training filler; outlier 1 / 1 / 1; reorder 0, with 0 shared books; **qdmatch_nq 26 / 29 / 32 rows: eval gold queries appear as training *distractor* queries**
- **Mean share of a held-out row's documents found in training:** contradiction ~0.08, outlier ~0.01, reorder 0, qdmatch_nq ~0.46.

## Caveats
- **qdmatch_nq:** the suite eval reuses the NQ-train query pool (19,177 distinct queries over its 3 rungs), so complete disjointness from the ctc train pool is impossible. Training rows whose gold query or gold document is an eval gold item were dropped (~60% of rows generated). Distractor overlap remains.
- **Contradiction length gap:** at 8k, both training and eval rows average 41.2 body tokens per document, and their mean row lengths are 8,225 vs 8,228 tokens. The earlier "1.2x gap" came from the eval audit's token count, not from the content.
- **Outlier length gap:** at 8k, body tokens per document are 147.7 in training vs 142.6 in eval (raw characters 613 vs 615).
- **16k rung document counts:** the pool's 16k documents per row for contradiction, outlier and reorder are below the eval's 16k rung (because of the 16,384 cap). The held-out suite16k rows use the eval's full document count but only its shortest rows.
- **qdmatch prompt layout:** the converter's qdmatch template puts part of the instruction before the documents even with query AFTER.

## Short-row pools (built 2026-11-17 by `build_short_pools.py`, then `build_pool_v2.py order`)
- **Paths:** `/data/prasann/early_loss_proxy/shards/{mtbshort,mtbmix80,mtbmix50}`, each with `<pool>_stats.json` and `<pool>_order_seed0.json`. They are on mooney and copied to sneetches (same folder).
- **Layout:** same as mtb16k, 7,808 rows (1,952 per task).
- **New cells:** same generator, split, converter flags and exclusion rules as mtb16k, plus the suite's r2k eval rungs (`v2build/suite_rows_r2k`). Long rows are mtb16k's own rows.
- **Gold check:** 20 short rows per task per pool, all 20/20 on every check.

| pool | rows per task (1k / 2k / 4k / 8k / 16k) | total tokens | mean row length |
|---|---|---|---|
| mtbshort | 976 / 976 / - / - / - | 12.59M | 1,612 |
| mtbmix80 | 781 / 781 / 130 / 130 / 130 | 24.35M | 3,118 |
| mtbmix50 | 488 / 488 / 325 / 325 / 326 | 42.06M | 5,386 |
| (mtb16k) | - / - / 651 / 651 / 650 | 71.42M | 9,147 |

**Short cells:** documents per row and tokens per row (mtbshort rows). 2k uses the eval calibration's document count; 1k uses the tokens-per-document fit to ~1,000 tokens (outlier raised to 7 so the majority beats 3 outliers).
- contradiction: 1k = 21 docs, 1,053 tokens (777-1,573); 2k = 56 docs, **2,590** (2,011-3,367). The eval's own 2k rung is labelled 2,560.
- outlier: 1k = 7 docs, 1,175 (1,003-1,453); 2k = 14 docs, 2,201 (1,913-2,904)
- reorder: 1k = 6 docs, 1,044 (874-1,990); 2k = 12 docs, 2,001 (1,694-3,314)
- qdmatch_nq: 1k = 8 docs, 971 (871-1,071); 2k = 18 docs, 1,862 (1,737-2,006)

**Drops (generated -> eligible):**
- reorder: 1k 1,000 -> 983, 2k 1,000 -> 991 (rows sharing a book with an eval rung)
- qdmatch_nq: 1k 4,500 -> 1,458, 2k 4,500 -> 1,410 (gold pair in an eval rung)
- no other task dropped any rows

**First 800 rows of a seed-0 run:** by task, 188 / 197 / 213 / 202 (contradiction / outlier / reorder / qdmatch_nq) in all three pools. By rung:
- mtbshort: 1k 376, 2k 424
- mtbmix80: 1k 303, 2k 347, 4k 57, 8k 53, 16k 40
- mtbmix50: 1k 202, 2k 174, 4k 154, 8k 143, 16k 127

**Caveat:** the long rows (from mtb16k) were filtered against the 4k-16k eval rungs only, not the r2k rung.

## Multi-question pools (built 2026-11-17 by `build_mq_pools.py`)
- **Paths:** `/data/prasann/early_loss_proxy/shards/{mtb16k_mq,mtbmix50_mq}`, with `_stats.json` and `_order_seed0.json`. On mooney + sneetches; staged to weka `.../prasanns/early_loss_proxy/shards/` (Beaker 01M7MEGXE1X2KEC7920N33H1MK).
- **Rows:** the same rows and order as mtb16k / mtbmix50. The prompt plus original answer is byte-identical. Then K chat turns follow before the EOS: `<|im_start|>user\n{q}<|im_end|>\n` + the row's own assistant prefix + `{a}<|im_end|>\n`.
- **Mask:** the original answer plus every sub-answer (each with `<|im_end|>\n`).
- **`mq_info.jsonl`:** one line per row with n_sub, `orig_answer` [s, e), `sub_answers` [[s, e)], and the question and answer texts.
- **Other sidecars:** gold sidecars re-keyed to the new fingerprints.
- **Verification:** 20 rows per task per pool, 20/20 on all three checks: prefix identical, sub-answers re-derived from the source gold, mask exact.

**K per rung:** 1k 2, 2k 4, 4k 8, 8k 16, 16k 32. Positives take at most half of K and negatives at most 2× the positives. Each row has 6 gold claims (contradiction), 3 outliers (outlier) and 6 pair members (qdmatch_nq), so the realised 16k K is lower:
- contradiction ~18
- outlier 9 (also 9 at 8k)
- reorder ~29
- qdmatch_nq ~14

Turns that would push a row past 16,384 tokens are dropped from the end. Cap drops: reorder 2,015 sub-questions, qdmatch_nq 2,840, contradiction 37 (mtb16k).

**Loss tokens per row, before -> after:**

| task | mtb16k_mq | mtbmix50_mq |
|---|---|---|
| contradiction | 28.0 -> 77.3 | 26.1 -> 55.8 |
| outlier | 16.4 -> 42.4 | 15.5 -> 33.0 |
| reorder | 224.9 -> 411.6 | 127.8 -> 234.5 |
| qdmatch_nq | 26.1 -> 68.7 | 23.8 -> 49.9 |

**Row length:** grows 3–10% (mtb16k mean 9,147 -> 9,724 tokens; 75.93M tokens and 1.17M loss tokens in total). mtbmix50_mq has 44.79M tokens.

**Example sub-questions:**
- contradiction: "Which claim contradicts Claim 5? Output the claim ID, or none if no claim contradicts it." -> `15`
- outlier: "Is Document [73] about a different topic than the rest of these passages? Answer yes or no." -> `yes`
- reorder: "Which passage comes immediately after Passage [23] in the original order? Output its passage ID, or none if it is the last passage." -> `4`. These alternate with "Put passages [a, b, c, d] in their original order. Write your answer as [id1, id2, ...]".
- qdmatch_nq: "Which query does document [49] answer? Output the query's number, or none if it answers no query." -> `none`. The pool also asks "Which document answers query [q]?".

## More short-share pools (2026-11-17, `build_short_pools.py --pools mtbmix90,mtbmix95,mtbmix80L,mtbmix80h`)
- **Built like** mtbshort / mtbmix*: same layout, sidecars, exclusions and in-task shuffle. Each pool is 7,808 rows, on mooney + sneetches, and staged to weka by Beaker job 01M7NG7KZ71TVP4ANSNNNSGBP2.
- **Regenerated cells:** the 1k cells were regenerated larger for mtbmix80h (contradiction / outlier / reorder 1,800 rows, qdmatch_nq 5,500). The seed is the same and the stream is prefix-stable: the earlier rows are byte-identical (checked).
  - Eligible rows at 1k: 1,800 / 1,800 / 1,771 / 1,771.
  - Exclusions: reorder dropped 29 rows that share a book with an eval file; qdmatch_nq dropped 3,729 rows with an eval gold pair.

| pool | rows per task (1k / 2k / 4k / 8k / 16k) | total tokens | mean row length | first 800 rows by rung (seed 0) |
|---|---|---|---|---|
| mtbmix90 | 878 / 878 / 65 / 65 / 66 | 18.51M | 2,371 | 336 / 399 / 25 / 19 / 21 |
| mtbmix95 | 927 / 927 / 33 / 33 / 32 | 15.52M | 1,987 | 353 / 415 / 12 / 11 / 9 |
| mtbmix80L | 781 / 781 / - / - / 390 | 33.19M | 4,251 | 303 / 347 / - / - / 150 |
| mtbmix80h | 1,562 / - / 130 / 130 / 130 | 20.90M | 2,676 | 650 / - / 57 / 53 / 40 |

- **Mean row length by rung** (all pools): 1k ~1,061, 2k ~2,164, 4k ~4,170, 8k ~8,480, 16k ~14,780 tokens.
- **First 800 rows by task:** 188 / 197 / 213 / 202 in every pool.
- **Gold check:** 20/20 on every check, with one exception. One mtbmix95 reorder row fails the text match on 3 passages. These passages contain Unicode combining marks (e.g. "K͟hwāja"), which the tokenizer's normalization changes; the content is the same and the printed IDs and answer are correct.

## Held-out v3 (`build_heldout_v3.py`, 2026-11-17)
**Locations:** `heldout_v3.npz` (680 rows) and `heldout_v3_stats.json` are in:
- `/data/prasann/early_loss_proxy/` on mooney and on sneetches
- weka `.../prasanns/early_loss_proxy/` (Beaker job 01M7NGY7571FMJMJFXC8CV1PFC)
- Lambda `/accounts/projects/sewonm/prasann/ctc_suite/elp/`

The npz format is the same as v2.

**Sets:**
- **`gen16k`:** 48 fresh rows per task at the training 16k cell sizes (324 / 96 / 90 / 174 documents for contradiction / outlier / reorder / qdmatch_nq). Every row is at most 16,384 tokens; means are 14.4-15.8k (range 12,963-16,107).
  - outlier, reorder, qdmatch_nq: ctc `--split train` with seed 7777.
  - contradiction: ctc's EVAL split (`--split eval --eval-seed 77`), because all 400/400 train-split rows of seed 7777 reused a training gold pair.
  - Rows are dropped if any gold document appears anywhere in a training row's documents. For qdmatch_nq the check is only against training gold, since its queries recur as distractors in training. Reorder rows are also dropped if their book was used in training.
  - Content-eligible rows: contradiction 71/500, outlier 390/400, reorder 105/1,500, qdmatch_nq 790/20,000.
- **`rung16k`:** the first 32 rows of each suite 16k rung file, uncapped (all of them fit the 40,960 limit).
  - Length range: contradiction 15,840-17,722; outlier 15,414-17,967; reorder 16,597-20,217; qdmatch_nq 15,475-16,088 tokens.
  - Rows over 16,384 tokens: 28 / 9 / 32 / 0.
- **`suite16k`, `suite8k`, `suite4k`:** copied from v2, same rows and order (reorder has 8 rows at 16k).

**Overlap against all 8 training pools (17,960 distinct training rows):**
- Prompt-hash hits: 0 in every set.
- Held-out gold documents found among training gold documents: 0.
- Held-out gold documents found among training distractors: qdmatch_nq queries in gen16k 48/48 and rung16k 32/32; rung16k contradiction 5, outlier 1.
- gen16k contradiction rows share 89% of their non-gold (filler) claims with training, because the filler pool is shared.

## `mtbmix80x3` (2026-11-17): the 80/20 recipe with 3x the rows
- **Rows:** 5,856 per task (23,424 total): 2,342 at 1k, 2,343 at 2k, and 390 / 390 / 391 at 4k / 8k / 16k.
- **Long rows** are the first rows of the mtb16k cells, with no repeats.
- **Locations:** on mooney + sneetches, and on weka via Beaker job 01M7NK5Z16MM5B0FR8QEAC71RA.
- **Size:** 73.10M tokens, mean row 3,121 tokens.
- **Short cells were regenerated larger.** The seed is unchanged and the earlier rows are byte-identical as a prefix:
  - contradiction / outlier / reorder: 2,700 rows each
  - qdmatch_nq: 8,500 rows
- **Eligible rows at 1k / 2k after exclusions:**
  - contradiction 2,700 / 2,700
  - outlier 2,700 / 2,699
  - reorder 2,634 / 2,631
  - qdmatch_nq 2,665 / 2,625 (generated 8,500 each)
- **New exclusion:** short rows are also excluded against heldout_v3's gen16k gold documents and books, in addition to every eval rung.
- **Gold check:** 20/20 rows per task pass.

## Long-row pool `mtb64k` + `heldout_v4` (`build_long_pools.py`, 2026-11-17)
**Pool:** `/data/prasann/early_loss_proxy/shards/mtb64k` (mooney + sneetches + weka, Beaker job 01M7NQ62YQSQ3GCC01YF3GMKCM).
- Parts t0..t2 are contradiction, outlier, qdmatch_nq. Reorder is left out because the suite has no reorder rows past 16k.
- 320 rows per task per rung, 1,920 rows in total, 84.26M tokens. Every row is at most 65,536 tokens.
- The converter ran with `--seq-len 131072`, so it dropped no rows.

| task | 32k: documents per row, mean tokens (range) | 64k: documents per row, mean tokens (range) |
|---|---|---|
| contradiction | 651 docs, 29,126 (27,155-31,483) | 1,304 docs, 58,625 (55,670-61,736) |
| outlier | 194 docs, 29,223 (28,193-30,307) | 388 docs, 58,382 (56,971-60,123) |
| qdmatch_nq | 324 docs, 29,314 (28,763-29,758) | 650 docs, 58,651 (57,998-59,417) |

- **Exclusions:** gold documents are checked against the suite's r2k-r64k eval rungs (HF revision 04ab8600; r128k and longer were not fetched), heldout_v2, heldout_v3 gen16k, and every heldout_v4 candidate.
  - qdmatch_nq: generated 2,400 rows per rung; 391 / 411 were eligible.
  - contradiction and outlier: 450 generated per rung, all eligible.
- **Gold check:** 10 rows per new cell. All checks pass, except the text match in one qdmatch_nq@32k row.
- **First 800 rows of a seed-0 run** (rung counts, read with a 65,536-token sequence length):
  - contradiction 132 / 145
  - outlier 137 / 125
  - qdmatch_nq 126 / 135

**`heldout_v4.npz`** (336 rows; same locations as the pool):
- **`gen32k` / `gen64k`:** 24 rows per task each, at the same document counts as the training cells.
  - contradiction and outlier come from ctc's EVAL split (eval seed 99).
  - qdmatch_nq comes from `--split train` with seed 7777, because ctc's qdmatch eval split re-used training gold pairs (only 3 / 6 of 120 rows were clean). Only 168 / 157 of 6,000 generated rows were gold-disjoint from all training pools.
- **`gen16k`:** copied from v3, 48 rows per task, all four tasks.
- **Overlap:** prompt-hash hits are 0 in every set, and no held-out gold document is a gold document in any training pool.
  - contradiction: 23 of 24 rows have a gold claim that appears among training filler claims, and 99% of their documents occur in training.
  - qdmatch_nq: every row's gold queries appear as training distractors.

## `mtb16kx4` (2026-11-18): mtb16k with 4x the rows
- **What it is:** mtb16k's recipe, tasks, rungs, documents per row, tokenizer, converter flags, 16,384 cap and layout, with 7,808 rows per task (2,603 / 2,603 / 2,602 at 4k / 8k / 16k). That is 31,232 rows, **1,952 steps per epoch at 16 rows/step**. t0..t3 = contradiction, outlier, reorder, qdmatch_nq.
- **Relation to mtb16k:** it contains all 7,808 mtb16k rows plus 23,424 new generated rows. `overlap --contains` found 7,808 / 7,808 mtb16k fingerprints in the pool.
  - The new rows come from the same ctc stream (seed 42, `--split train`), regenerated larger: 3,200 / 3,200 / 3,300 / 8,000 rows per cell (contradiction / outlier / reorder / qdmatch_nq).
  - The stream is prefix-stable: in all 12 cells the first 1,000 rows (2,000 for qdmatch_nq) are byte-identical to the mtb16k cells.
  - Rows are shuffled within each task (seed 20261116, as for mtb16k), so the order differs from mtb16k's.
- **No duplicate prompts:** 31,232 distinct prompt hashes. The dedupe in assemble dropped 0 rows.
- **Size:** 285.9M tokens and 2.31M loss tokens. Mean row length is 9,154 tokens and the maximum is 16,378. On disk: 12 files, 1,492,490,266 bytes.
- **Built by:** `build_pool_v2.py` with `--cell-root /data/prasann/early_loss_proxy/x4build --gen-rows ...`, then `assemble --rows-per-task 7808 --exclude-heldout`, `order`, and `overlap --contains .../mtb16k`. Slurm jobs 5202903 and 5202936 on mooney, CPU only, about 25 minutes in total. The build cells are in `mooney:/data/prasann/early_loss_proxy/x4build/{gen,tok}`.
- **Locations:** the pool, `mtb16kx4_stats.json`, `mtb16kx4_order_seed0.json` and `mtb16kx4_overlap.json` are in:
  - `/data/prasann/early_loss_proxy/shards/mtb16kx4` on mooney and on sneetches (the sneetches copy was piped node to node through srun; same 12 files and bytes, matching sha1)
  - weka `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/early_loss_proxy/shards/mtb16kx4` (Beaker job 01M7Q2DFPVENQMNXQWAQ68YNNQ; S3 copy at `s3://ai2-llm/checkpoints/prasanns/early_loss_proxy/shards/mtb16kx4`)

| task | mean tokens (min-max) at 4k / 8k / 16k | answer tokens | rows dropped (generated -> kept per rung) |
|---|---|---|---|
| contradiction | 4,144 (3,357-5,354) / 8,372 (7,197-9,942) / 14,496 (12,827-16,069) | 25-30 | 16k: 1 over 16,384 |
| outlier | 4,311 / 8,636 / 14,476 (13,560-15,745) | 16-17 | 16k: 1 gold-in-eval |
| reorder | 4,402 / 9,211 / 14,425 (12,570-16,378) | 101 / 221 / 353 | over 16,384: 0 / 0 / 53. Gold-in-eval: 1 / 4 / 3. Same book as an eval row: 49 / 35 / 43. Gold or book in a held-out gen row: 51 / 49 / 63 |
| qdmatch_nq | 3,820 / 7,738 / 15,826 | 25-28 | gold pair in eval: 4,853 / 4,806 / 4,835 of 8,000. Gold in a held-out gen row: 45 / 53 / 47. That leaves 3,102 / 3,141 / 3,118 eligible, so the 2,603 needed per rung was not the limit |

**Exclusions:** as mtb16k (all three 4k-16k eval rung files), plus a new rule (`--exclude-heldout`): a training row is dropped if a gold document of the row, or its reorder book, is gold in heldout_v3 gen16k or heldout_v4 gen32k / gen64k. That set is 426 / 216 / 3,822 / 531 gold hashes (contradiction / outlier / reorder / qdmatch_nq) and 37 reorder books. Rows dropped by the new rule are all new rows; the mtb16k rows were already clean against it.

**Gold sidecars and check:** `gold_fingerprints.json`, `gold_pairs_fingerprints.json` and `row_info.jsonl` are in the same format as mtb16k. 40 rows per task were decoded, and every task passed 40/40 on all three checks (the span text matches the source, printed IDs are index + 1, and the answer IDs match gold).

**First 800 rows of a seed-0 run:**
- By task: contradiction 188, outlier 207, reorder 211, qdmatch_nq 194.
- By rung: every task × rung cell gets 59-78 rows.

**Overlap (held-out vs all 31,232 training rows; `mtb16kx4_overlap.json`).** Prompt-hash hits are 0 in every set of heldout_v2 / v3 / v4, and held-out gold documents among training GOLD documents are 0 in every set. The table counts held-out rows that have a gold document anywhere in a training row, followed by the mean share of a held-out row's documents found in training:

| set | contradiction | outlier | reorder | qdmatch_nq |
|---|---|---|---|---|
| suite4k / suite8k / suite16k (v2 = v3) | 6 / 6 / 9 (share 0.08-0.10) | 4 / 4 / 4 (0.03-0.04) | 0, 0 shared books | 26 / 29 / 32 (0.46) |
| v3 rung16k | 6 (0.10) | 4 (0.04) | 0 | 32 (0.47) |
| v3 gen16k (= v4 gen16k) | **12 / 48** (0.98) | 2 / 48 (0.06) | 0, 0 books | 48 / 48 (1.0) |
| v4 gen32k / gen64k | 23 / 23 of 24 (0.99) | 0 / 0 | - | 24 / 24 (1.0) |

**Caveats:**
- **Gold claims as filler:** with 4x the contradiction rows, 12 of the 48 gen16k contradiction rows now have a gold claim that appears among training *filler* claims (0 for mtb16k). It is never a training gold claim. The filler pool is shared, so nearly all of gen16k's filler occurs in training.
- **Distractors:** outlier gen16k has 2 such rows, and qdmatch_nq gold queries still recur as training distractors, as in mtb16k.
- **Eval rungs covered:** like mtb16k, the exclusion covers the 4k-16k eval rungs, not r2k or r32k+.

## `mtb16kx16` (2026-11-18): mtb16k with 16x the rows
- **What it is:** same recipe, rungs, documents per row, converter flags, exclusions (including `--exclude-heldout`) and layout as mtb16kx4, with 31,232 rows per task (10,411 / 10,411 / 10,410 at 4k / 8k / 16k). That is 124,928 rows, **7,808 steps per epoch at 16 rows/step**. t0..t3 = contradiction, outlier, reorder, qdmatch_nq.
- **Relation to mtb16k and mtb16kx4:** it contains every mtb16kx4 row (`overlap --contains mtb16kx4`: 31,232 / 31,232 found), so it also contains every mtb16k row.
  - The stream is the same seed-42 ctc stream regenerated larger: 11,000 / 11,000 / 11,700 / 28,500 rows per cell (contradiction / outlier / reorder / qdmatch_nq).
  - In all 12 cells the first 3,200 / 3,200 / 3,300 / 8,000 rows are byte-identical to the x4 cells.
  - Rows are shuffled within each task with seed 20261116.
- **No duplicate prompts:** 124,928 distinct prompt hashes; the dedupe dropped 0 rows.
- **Size:** 1,143.6M tokens and 9.23M loss tokens. Mean row length is 9,154 tokens and the maximum is 16,380. On disk: 12 files, 5,970,511,080 bytes.
- **Built by:** slurm job 5203053 on mooney, CPU only, 71 minutes: gen 45, tokenize 9, assemble 12, then order and overlap. It used the same `build_pool_v2.py` arguments as x4, with `--rows-per-task 31232`.
- **Locations:** the pool, `mtb16kx16_stats.json`, `mtb16kx16_order_seed0.json` and `mtb16kx16_overlap.json` are in:
  - `/data/prasann/early_loss_proxy/shards/mtb16kx16` on mooney and on sneetches (copied node to node through srun; same 12 files and bytes, matching sha1)
  - weka `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/early_loss_proxy/shards/mtb16kx16` (Beaker job 01M7QEP5MFB1FY9QAEZ6Y83KVK; S3 copy at `s3://ai2-llm/checkpoints/prasanns/early_loss_proxy/shards/mtb16kx16`)
  - The build dir `mooney:/data/prasann/early_loss_proxy/x16build` had 4.0G of gen and 3.8G of tok. Both were deleted after staging; only `code/` and the job log remain. Regenerate them with the same arguments: the stream is deterministic.
- **Did not fall short:** qdmatch_nq kept about 38% of generated rows (10,936 / 10,932 / 10,922 eligible for the 10,411 needed). The clean rate is flat along the stream (761-832 clean per 2,000-row block on x4), so the finite NQ seed pool limits gold-pair diversity, not row count: each clean gold pair recurs across rows.

| task | mean tokens (min-max) at 4k / 8k / 16k | answer tokens | rows dropped per rung (4k / 8k / 16k) |
|---|---|---|---|
| contradiction | 4,146 / 8,383 / 14,498 (12,827-16,289) | 25-30 | over 16,384: 0 / 0 / 3 |
| outlier | 4,309 / 8,637 / 14,472 (13,499-16,089) | 16-17 | gold-in-eval: 1 / 3 / 5. Gold in a held-out gen row: 0 / 0 / 1 |
| reorder | 4,405 / 9,203 / 14,422 (12,570-16,380) | 101 / 221 / 353 | over 16,384: 0 / 0 / 183. Gold-in-eval: 6 / 13 / 6. Same book as an eval row: 164 / 160 / 145. Gold or book in a held-out gen row: 215 / 222 / 257 |
| qdmatch_nq | 3,820 / 7,737 / 15,822 | 25-28 | gold-in-eval: 17,368 / 17,340 / 17,375 of 28,500. Gold in a held-out gen row: 196 / 228 / 200. Over 16,384: 0 / 0 / 3 |

**Gold sidecars and check:** `gold_fingerprints.json`, `gold_pairs_fingerprints.json` and `row_info.jsonl` are in the same format as before. 40 rows per task were decoded, and every task passed 40/40 on all three checks.

**First 800 rows of a seed-0 run:**
- By task: contradiction 197, outlier 204, reorder 197, qdmatch_nq 202.
- By rung: every task × rung cell gets 52-81 rows.

**Overlap (`mtb16kx16_overlap.json`).** Prompt-hash hits are 0 in every set of heldout_v2 / v3 / v4, and held-out gold documents among training GOLD documents are 0 in every set. The table counts held-out rows with a gold document anywhere in a training row, followed by the mean share of a held-out row's documents found in training:

| set | contradiction | outlier | reorder | qdmatch_nq |
|---|---|---|---|---|
| suite4k / suite8k / suite16k (v2 = v3) | 6 / 6 / 9 (share 0.09-0.11) | **10 / 10 / 9** (0.14) | 0, 0 shared books | 26 / 29 / 32 (0.46) |
| v3 rung16k | 6 (0.10) | 10 (0.14) | 0 | 32 (0.47) |
| v3 gen16k (= v4 gen16k) | **12 / 48** (0.98) | **12 / 48** (0.27) | 0, 0 books | 48 / 48 (1.0) |
| v4 gen32k / gen64k | 23 / 23 of 24 (0.99) | 0 / 0 | - | 24 / 24 (1.0) |

**Held-out gen16k rows whose gold appears as training filler:**

| pool | contradiction | outlier |
|---|---|---|
| mtb16k | 0 | 0 |
| x4 | 12 | 2 |
| x16 | 12 | 12 |

Never as training gold. With 16x the outlier rows, outlier filler overlap with the held-out sets grows: 9-10 of 32 suite rows, and a 0.14 mean document share against 0.03-0.04 on x4. The other caveats are as for x4.
