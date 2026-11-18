---
experiment: recipe_cpt
title: Continued pre-training of Qwen3.5-4B at 32k with the FLOP-drop recipe (whole-model LoRA r64 + online two-stage router 20%/7%) vs dense full-FT CPT, 1B tokens, checkpoints at 125M/250M/500M/1B as SFT bases
status: active
updated: 2026-11-17
branch: claude/recipe-cpt-q35-4b (pushed; cut from claude/early-loss-proxy-d6e886 @ d52756c12)
launcher: research/token_routing/recipe_cpt/launch_recipe_cpt.py
records:
  - research/token_routing/recipe_cpt/README.md (design, smoke, probes; no record yet)
  - research/token_routing/recipe_cpt/HANDOFF.md (jobs, paths, next commands)
artifacts:
results: ops result show --experiment recipe_cpt
wandb_group: https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/rcpt-q35-4b
next_step: export + dev-CE each checkpoint as it lands (post_ckpt_beaker.sh), then recipe/dense SFT on mt4 s1952 from each base (launch_sft_from_cpt.py)
---

# recipe_cpt

## Question
Does CPT under the recipe make the recipe more Pareto-optimal downstream? CPT Qwen3.5-4B at 32k with the recipe for
1B tokens, keep 125M / 250M / 500M / 1B checkpoints, and run recipe-vs-dense SFT from each. Does the recipe's
FLOPs-to-match-dense improve against the same comparison from the original base? Dense full-FT CPT on the same rows
is the control ("CPT data" vs "CPT with dropping").

## Decisions
- 2026-11-17 (Prasann): run the recipe CPT on Beaker, ~1B tokens, a few checkpoints.
- 2026-11-17 (Prasann): dense full-FT CPT control approved, lower priority, same setup. minRuntime = 1.3 x the
  measured runtime, also above 4 h if Beaker accepts it.
- 2026-11-17 (agent), base and data:
  - base = `q35-4b-base-markerfix` (the recipe SFT base);
  - data = softdetach's cpt_u1B stream re-cut to 32,257-token rows (63 x 512-token pseudo-docs), 1,938 steps x 16
    rows;
  - recipe flags = `lora64all32r20to7` + `--st-gold-blind --st-otr-no-query`.
- 2026-11-17 (agent), schedule and checkpoints:
  - LR 2e-5 for both arms (olmo3_cptdrop), WSD with 3% warmup and 10% decay;
  - permanent checkpoints every 242 steps, ephemeral every 121.
- 2026-11-17 (agent): launched the recipe run despite the dev-CE regression found in the probes. It costs ~15
  GPU-h, and the question is downstream.

## Current best numbers
Teacher-forced full-attention CE on cpt_dev (held-out source part 28, 64k rows), no-cot.

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| base q35-4b-markerfix | cpt_dev / 64k | full CE 1.3199 | 32 ⚠ eval_size=32 | n/a | no-cot (TF CE) | ops result (job 01M7NYFYY2SCBYFF0P6NZFK703) |
| 20-step smokes: dense / loradense / recipe 5e-6 / 1e-5 / 2e-5 | cpt_dev / 64k | rows 1-5 CE 1.095 / 1.080 / 1.237 / 1.292 / 1.288 vs base 1.121 | 8 ⚠ eval_size=8 (direction only) | n/a | no-cot (TF CE) | ops result; README |

Smoke throughput (8 x H100): recipe 3.30 s/step, ~20k tok/s/GPU, 19.1 GiB peak active, 0.179x dense FLOPs/step.
Dense 8.38 s/step, 7.8k tok/s/GPU, 31.2 GiB.

## Next step
Export and dev-CE each checkpoint (`post_ckpt_beaker.sh`), then SFT from each base (`launch_sft_from_cpt.py`). The
commands are in HANDOFF.md.

## Open issues / traps
- Recipe CPT raises full-context dev CE by +0.12 to +0.17 within 20 steps at every LR tried. LoRA alone at the same LR
  lowers it, so the cause is the routing (olmo3_cptdrop's failure, stronger). Read the checkpoint dev CE before
  SFT-ing from it.
- `train/CE` of the recipe arm is the kept-token CE sum divided by all label tokens (~0.24 = ~3.2 nats on 7.4% of
  targets); not comparable to dense CE.
- `beaker_ctc_suite.py` passes `--no-final-checkpoint`, which disables the checkpointer unless
  `--ephemeral-save-interval` is set (`--save-interval` alone writes nothing).
- The post-ckpt eval needs the training image (`akshitab/olmo-core-tch2130cu129-sm80-2026-09-11`). The tylerr image
  lacks `fla`, and `pip install '.[all]'` fails there on `dion`.

## Verified vs unverified
- Verified from job logs: shard metadata, smoke throughput / memory / router keep fractions / resume, merged ==
  unmerged dev CE, base dev CE 1.3199, probe dev CE.
- Not yet: any production checkpoint; the wall-clock estimates are smoke extrapolations.
