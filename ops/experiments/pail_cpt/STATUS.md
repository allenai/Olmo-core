---
experiment: pail_cpt
title: One-time continued pretraining of PAIL classifier backbones on ~1B tokens of pail_v9 web text
status: active
updated: 2026-11-06
branch: PAIL prasann/ccdev-v3-cpt
launcher: PAIL scripts/cpt/launch.sh (scripts/cpt/clm.py for decoders, scripts/cpt/mlm.py for Ettin)
records: PAIL docs/classify/p41dev_cpt.md (Ettin-1B, done), PAIL docs/classify/p41dev_cpt_large.md (Qwen2.5 7B-Instruct / 14B, running)
results: s3://ai2-llm/checkpoints/prasanns/PAIL/cpt/
artifacts: https://claude.ai/artifact/U8tdBvrbWe1EDAZQFoK2wj (PAIL Text Workbench; CPT results go there when they land)
wandb_group:
next_step: wait for DONE.json of cpt-qwen25-7bi-1b (~15:40 UTC 2026-11-06) and cpt-qwen25-14b-1b (~22:30 UTC); then record held-out loss and hand the aliases pail-qwen25-7bi-cpt / -14b-cpt to the classifier loop
---

# pail_cpt

## Question
Does continued pretraining on in-distribution web text (pail_v9 Common Crawl pages from sites outside p41dev) make
better backbones for the AI-text classifiers? Each CPT runs exactly once (DONE.json guard).

## Decisions
- 2026-11-04 (Prasann): approved a one-time 1B-token CPT of Ettin-1B (done; record p41dev_cpt.md).
- 2026-11-05 (Prasann): "could you look into getting the CPT jobs kicked off? Set appropriate minruntimes but avoid
  excessive storage of checkpoints" -> LoRA CPT of the large-ensemble Qwen2.5 bases, one rolling checkpoint per run.
- 2026-11-05/06 (Prasann): start with the 14B and Qwen2.5-7B-Instruct only; 32B and 72B on hold until he says go.

## Current runs
| run | Beaker | model | tokens/s (4 H100) | peak mem | ETA | out |
|---|---|---|---|---|---|---|
| cpt-qwen25-7bi-1b | 01M6R0BGQCK9QARPTBQ3RAR667 | Qwen2.5-7B-Instruct, LoRA r64, bf16 | 42.4k | 50.3 GiB | 6.5 h train | .../cpt/qwen25_7bi_pailv9_1b/ |
| cpt-qwen25-14b-1b | 01M6R0BBVHH941G6EBN75CV3FX | Qwen2.5-14B, LoRA r64, bf16 | 20.8k | 58.8 GiB | 13.3 h train | .../cpt/qwen25_14b_pailv9_1b/ |

Both: --min-runtime 8h (Beaker max), autoResume, hourly rolling checkpoint.

## Next step
Wait for both DONE.json; record held-out loss start -> end; then classifier runs with the CPT aliases.

## Open issues / traps
- The 14B outlasts the 8 h min-runtime: a preemption costs up to ~1 h (resumes from the hourly checkpoint).
- 32B / 72B (QLoRA, `--q4 --gc`) would take ~1.5-2 and 3-4+ days on 4 H100s (FLOP estimate, not measured).

## Verified vs unverified
- Verified (smokes 2026-11-06): resume after simulated preemption, merge, merged model loads as a classifier (bf16 + 4-bit),
  merged held-out loss equals the trained adapter's.
- UNVERIFIED: whether the CPT checkpoints improve the classifiers.
