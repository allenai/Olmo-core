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
