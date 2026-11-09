---
experiment: pail_webgen
title: webgen, a web-native multilingual AI-text test set: LLM pages generated from ccdev_v3 seeds, spliced into the seeds' boilerplate
status: active
updated: 2026-10-27
branch: prasann/webgen-testset (worktree /accounts/projects/berkeleynlp/prasann/projects/PAIL-webgen)
launcher: scripts/tinyclf/beaker.sh (steps: scripts/webgen/{build_seeds,generate,qc}.py; viewer.py builds the artifact)
records: docs/webgen/PLAN.md
results:
  - s3://ai2-llm/checkpoints/prasanns/PAIL/webgen/v0/ (seeds/, gen/{qwen35-9b,qwen38-27b}.jsonl.gz, pages.jsonl.gz, qc_stats.json)
artifacts:
  - https://claude.ai/artifact/8CoCDMMH4qRDZUvQGnrDzE (webgen Viewer: seed vs AI pages, QC, shared reviewer flags in its db collection `flags`)
wandb_group:
next_step: v0.1 done (brief regenerated, audit clean, baselines in the viewer); next: more models with the fixed prompts, multilingual baselines if Prasann wants them
---

# pail_webgen

## Question
Build a high-quality test set of AI-generated web pages that look like what Common Crawl holds (every language, page
type and boilerplate, pail_v9 text), paired with the real pages they were generated from. Test set first; detector
scores (MELD, ProseLens) after.

## Decisions
- 2026-10-26 (Prasann): seeds from ccdev_v3 (2024-26 crawls); label noise on seeds is fine. Two disjoint subsets:
  truly random, and stratified (language x length), reported separately. Modes new_content, brief, translate.
  Splice generated bodies into the seed's header/footer. Shortcut audit only fixes blatant artifacts (AUROC >= 0.95).
  Baselines MELD + ProseLens only, later. Push the branch often. All jobs on Beaker, no local compute.
- 2026-10-26 (Claude): main body = best-scoring contiguous run of pail_v9 lines that resiliparse main-content keeps
  (max subarray); seeds need >= 60 body words and >= 50% of the matched text inside the run.

## Runs
- webgen-seeds-smoke-10262304 (Beaker 01M5X7A74QE8K0JAH7WASANWHM, CPU), done: build_seeds.py 10+10 ->
  s3://ai2-llm/checkpoints/prasanns/PAIL/webgen/smoke/seeds/. 20 kept of 83 candidates (body_short the main loss,
  26 of 53 stratified, mostly the short bucket). Showed ccdev `lang` is wrong on ~4 of 20 pages (pt as lt/gl, en as
  id) and one 41k-word single-line zh body -> fixed in 0614e88 (fastText lid.176 body language, 3000-word body cap).
- webgen-seeds-v0-10262329: cancelled, hung once fetching began (fastText loaded by 16 threads at once; fixed f273c4f).
- webgen-seeds-v0-10262356 (Beaker 01M5XA9J06A1VF392JKW6NE7X1, CPU), done in 12 min: seeds v0 at
  s3://ai2-llm/checkpoints/prasanns/PAIL/webgen/v0/seeds/ (seeds.jsonl.gz, rejects.jsonl.gz, stats.json).
  150 random (29 languages, en 60, ru 12, fr 10, es 8, de 7; short/medium/long 24/82/44) + 150 stratified (16
  languages at 6-9 each plus a tail; 42/55/53). fastText changed ccdev's `lang` on 33 of 300. Random candidates
  dropped: 74 body_short, 11 no_body, 8 body_long, 5 body_scattered of ~250, so "random" = random among pages with a
  60-3000-word body (link-only and very short pages are out).
- webgen-gen-smoke-10262329 / -10262344: failed. (1) the image's flash_attn (built for its torch) is imported by vLLM's
  rotary embedding -> beaker.sh --vllm now removes it; (2) FlashInfer JIT (GDN prefill, sampler) needs nvcc ->
  generate.py uses Triton GDN prefill and VLLM_USE_FLASHINFER_SAMPLER=0.
- webgen-gen-smoke-10262355: failed (the image's deep_gemm is too old; DeepGEMM FP8 warmup) -> VLLM_USE_DEEP_GEMM=0.
- webgen-gen-smoke-10270025 (Beaker 01M5XBYV75QJDVD9PAB24RPAN4), done: Qwen3.5-9B, 6 seeds x 3 modes, 18/18 stop,
  17/18 pass QC (one new_content copied its seed, overlap 0.98). Load 144 s, 18 requests in 35 s. Read all 18:
  translate keeps nav/phones/dates (realistic MT pages); new_content plausible, keeps seed quirks; brief outputs were
  flat essay prose -> briefs now record layout (6f940d2).
- webgen-gen-v0-qwen3-5-9b-10270039 (01M5XCR7AE1YRYPV2DN6Y7C8KZ), done: 900 rows (882 stop, 18 length), 234 s of
  generation on 1 H100 -> .../webgen/v0/gen/qwen35-9b.jsonl.gz
- webgen-gen-v0-qwen3-8-27b-10270039: failed, max_num_seqs 1024 > 335 Mamba state blocks -> --max-num-seqs 256.
- webgen-gen-v0-qwen3-8-27b-10270059 (01M5XDWR6FYT5207CXH1YM8YNS), done: 900 rows (889 stop, 11 length), 979 s.
  Started before e77a087, so 11 of its translations (and the 9B's) target Latin.
- webgen-qc-v0-10270127 (01M5XFH80QNM7AC6XPJFVWMS3G), done: QC over both -> .../webgen/v0/pages.jsonl.gz, qc_stats.json.
- Trap: zsh does not word-split an unquoted $VAR into several arguments: write launcher args out in full.
- 2026-10-28, commit eb57b39: webgen-{audit,meld,proselens}-v0 failed at argparse (zsh does not word-split an unquoted
  $A). Relaunched, running: webgen-audit-v0b (01M627JESP4W6HZHJB2PHJZHZA, CPU, scripts/webgen/audit.py ->
  .../webgen/v0/audit/), webgen-meld-v0b (01M627JYDWWFZWC6J832N20HB5, 1 GPU) and webgen-proselens-v0b
  (01M627KBB0J5FZ78HYPG3FF1FV, 1 GPU, --vllm), both scripts/webgen/baselines.py -> .../webgen/v0/baselines/.
  MELD and ProseLens are English-only (model cards): non-English numbers are out of scope.
  Baselines done (.../v0/baselines/report.md, QC-passing, full page): MELD AUROC 0.66 (brief 0.77, new_content 0.65,
  translate 0.54-0.57), seed FPR at its shipped 1% cut 0.62 (en 0.25); ProseLens AUROC 0.79 (brief 0.89, new_content
  0.78-0.80, translate 0.68-0.69), recall 0.27 at its 1% cut with seed FPR 0.047; ProseLens ranks the AI page above its
  own seed in 85-98% of new_content/brief pairs. Audit v0b page view: no feature >= 0.95 (best: bold/asterisk 0.81 on
  brief); n-gram probe 0.71 all, 0.85 brief. Its body view was invalid (seed span is line indices); audit-v0c
  (01M629FVPE48TTTDJYAE5JCEQ3, befb2c3) reruns it, queued on the CPU clusters.
- 2026-10-29: audit-v0c sat 1.5 h in the CPU queue -> cancelled; webgen-audit-v0d (1 GPU cluster) done: one flag, n-gram
  body|brief 0.957 ('**Label:**' bold on AI vs ###/| on seeds). Fix d59efca: brief writer's format line comes from the
  seed body's own markup. webgen-brief2-v0 (.../webgen/v0_brief2/): flag gone (0.904) but wrong_lang 15 -> 52 (English).
  Fix 42bc4c8: restate the target language last. webgen-brief3-v0 (.../webgen/v0_brief3/): brief pass 0.98 / 0.973,
  wrong_lang 3 / 0, audit nothing flagged (n-gram body 0.908, page 0.786). MELD / ProseLens on brief3: brief AUROC
  0.74-0.76 / 0.88-0.89 (v0: 0.77 / 0.89-0.90).
- v0.1 = v0 new_content + translate, brief from v0_brief3. Viewer republished (version 4) with per-page MELD/ProseLens
  scores, a baseline table computed from them, and both audits (viewer 8c526a1).

## Next step
Read the viewer's `flags` collection when Prasann has reviewed. Optional cheap softenings (not flagged): curly quotes
on seeds vs ASCII quotes on AI pages; '**' still in 8% of brief bodies. Multilingual baselines (Binoculars /
Fast-DetectGPT on a non-Qwen pair; Pangram with OK) await Prasann. Then more models (Qwen3.6-35B-A3B on 2 GPUs, other families, API models).

## Open issues / traps
- Seed filters (body length, body found) bias away from link-only listing pages; rejects.jsonl.gz records the rates.
- Beaker image vs vLLM 0.21: flash_attn, FlashInfer JIT (no nvcc), deep_gemm all break Qwen3.5-family loading; the
  fixes live in beaker.sh --vllm and generate.py (env vars + gdn_prefill_backend=triton).
- Some ccdev seeds are mojibake (a ko page decoded wrong); fastText still labels them; QC flags their translations.

## Verified vs unverified
- Verified 2026-10-26 by reading 20 smoke seeds: body spans mostly right; failures are product/UI pages where
  resiliparse main content is itself junk (exaclean) and generic site blurbs (kanshudo).
- Verified 2026-10-27: brief layout fix (78% of 9B brief pages now have a heading or list line); Qwen3.8-27B runs on 1
  H100 with max_num_seqs 256.
- UNVERIFIED: the viewer's review save from the page (the db read works; no page write tried yet).
