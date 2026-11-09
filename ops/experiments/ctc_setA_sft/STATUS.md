---
experiment: ctc_setA_sft
title: setA dense SFT pair — Qwen3.5-4B on the 11-task CTC setA mix, 32k vs 256k training window (canonical mixed-task recipe)
status: active
updated: 2026-10-12
branch: prasann/ctc-setA-sft
launcher: recipes/train/setA/launch_setA_sft.py
records:
  - records/ctc-fast-suite-eval.md
results:
  - recipes/eval/setA_audit/
wandb_group:
next_step: Pull the per-cell eval JSONs from the weka results root, record the 32k-vs-256k table with `ops result add --check` (results-hub is retired), then use this recipe as the base for flopdrop_clean
---

# ctc_setA_sft

## Question
Two dense Qwen3.5-4B SFTs on setA differ only in maximum training context: 32k drops the longer
examples, 256k keeps everything. Does training at 256k buy long-rung CTC performance, and on which
tasks? The pair also defines the canonical mixed-task train→eval recipe.

## Decisions
- 2026-09-22 (Prasann): "just use set A". setA is 13 tasks as built. textgroups is then excluded from training for a shortest-document shortcut, leaving 11 tasks (README). `qdmatch_hpqa` is dropped so the qdmatch_fiqa/hpqa rows stay clean probes.
- 2026-09-23 (agent): setA is rebuilt IID with the eval (`setA_max20_evaliid`, `build_ctc_sft.py --calibration eval_calibration.json`). The IID audit against real eval rows is 76/77 cells clean (record §2026-09-23).
- 2026-09-23 (Prasann): dense arm. Schedule 32k first. Avoid 2 nodes. Use amandab's validated 256k SFT recipe. The branch is cut from `amandab/contradiction-only-256k-20260914` for its CP fixes.
- 2026-09-23 (Prasann): do NOT re-tokenize the marker-wrapped shards. Eval passes olmo-eval `--doc-markers` instead.
- 2026-09-23 (agent): 256k runs on 1 node with Ulysses CP=8. `256k-2node` (CP=4/DP=4) is the fallback.
- 2026-10-12 (Prasann): this setA recipe IS the canonical mixed-task training recipe. Its docs live in three places: README on `prasann/ctc-setA-sft`, the olmo-eval launcher README, and `records/ctc-fast-suite-eval.md` plus the run-evals skill on landmark. `ctc_suite/train_ctc_suite.py` is the research-variant trainer, not canonical.
- 2026-10-12 (agent): `fd49fc8a8` merges `prasann/ctc-setA-sft` into `prasann/landmark` on `tmp/flopdrop-integration`, the integration for `prasann/flopdrop_clean`.

## Current best numbers
Base `q35-4b-dense-256k-fix/step2385`, 1,048,576 tok/step, LR 4e-5, 1 epoch. Checkpoints: 32k = `q35-4b-dense-ctc-setA-32k-20260923T1537/step888`, 256k = `q35-4b-dense-ctc-setA-256k-20260923T1537/step1595`. Evals use the olmo-eval launcher, `--arm full --doc-markers`. The manifest policy is `r2k-r32k:300, r64k/r128k/r256k:100`. That is below the 500 floor at every rung.

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| 256k | contradiction / r128k | f1≈0.79 | ⚠ 100 | ±0.041 | no-cot | memory ctc-setA-dense-sft-pair (UNVERIFIED) |
| 32k | contradiction / r128k | f1≈0.35 | ⚠ 100 | ±0.048 | no-cot | same |
| 256k | contradiction / r256k (YaRN x2 chunked) | f1≈0.66 | ⚠ 100 | ±0.047 | no-cot | same |
| 32k | contradiction / r256k (YaRN x2 chunked) | f1≈0.15 | ⚠ 100 | ±0.036 | no-cot | same |
| both | all 12 rows / ≤32k | "tie" (no numbers recorded) | ⚠ 300 | — | no-cot | same |

Memory also says the 256k checkpoint wins hpqa and rerank at 256k. No values are recorded for those.

## Next step
Harvest the cells from the weka results root of these launcher runs:
- `q35-4b-dense-ctc-setA-{32k-step888,256k-step1595}-full-dm` for 2k-128k;
- `…-r256k-yarn2-chunked` (the default) and `…-r256k-fit` for r256k.

Record the table with `ops result add` (results-hub is retired), with `eval_size` per cell. Re-run any row Prasann wants to
quote at 500 (125 at 256k).

## Open issues / traps
- `*-full-dm` r256k cells are LEFT-TRUNCATED (43-100% of rows). Never use them. Use `*-r256k-yarn2-chunked` or `*-r256k-fit`.
- The eval MUST pass `--doc-markers` and use the 0.8B tokenizer (memory:qwen35-4b-chat-template-eval-mismatch). The earlier "66/66 doc-marker parity" used the 4B template on both sides. `check_doc_marker_parity.py` was fixed 2026-09-28.
- Run names must carry the variant for the docchunk/landmark eval path. Repair marker embeddings before any non-dense arm (CLAUDE.md, memory:marker-embedding-norm-bug). The prep audit for this base PASSED: cos +0.24, norms 0.57x/0.84x.
- 256k resumed from step1200 after two preemptions. Check memory:trainer-silent-autoresume-trap if relaunching into the same save folder.
- The branch is 272 commits away from landmark (612). It diverged at `d65ff09df`, 2026-07-29. Data is built on landmark (`src/scripts/data/ctc_sft/build_ctc_sft.py`). Training runs on setA-sft.
- Some r256k rows of hpqa, outlier, qdmatch_nq and rerank exceed 262,144 tokens (record).

## Verified vs unverified
- Verified 2026-10-12:
  - `origin/prasann/ctc-setA-sft` @`0c7b3e9ab` (2026-10-12) README: run names, Beaker IDs, final steps 888/1595, the prep table (212,786 / 203,156 examples; 1,595 / 888 steps), and the eval run names;
  - the olmo-eval launch manifests: policy `r2k-r32k:300,…:100`, rows, `--doc-markers`, tokenizer 0.8B, and the r256k-yarn2-chunked manifest (7 rows, `rope_yarn 2.0`, `prefill_chunk_size 131072`, `olmo_core_ref prasann/landmark-chunked-prefill`);
  - local `prasann/ctc-setA-sft` is one commit behind origin (`774a2d8df`).
- 2026-10-16: the full per-cell tables (both checkpoints, 12 rows × 2k–256k, plus the fitted vs YaRN 256k comparison) are in the Claude transcript `~/.claude/projects/-accounts-projects-berkeleynlp-prasann-projects-OLMo-core/1c1c30a5-db8c-4cda-973a-ceaf79899b03.jsonl` (assistant messages 2026-09-29 20:39, 10-03 03:15 / 10:23, 10-12 19:26). They are still not recorded in `ops result`, which needs the weka JSONs.
- UNVERIFIED — confirm with Prasann: every number in the table. They come only from memory (updated 2026-10-12). The result JSONs are on the weka results root, which was not read. No in-repo or results-hub copy was found.
- UNVERIFIED: the hpqa/rerank 256k wins and the "within noise" agreement of YaRN vs fit-window.
