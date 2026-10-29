---
experiment: tokdrop_lr
title: LR ablation for token-drop training — dense vs gold_fl20p8_noslot (gfl20p8ns) × LR {5e-6, 1.5e-5, 5e-5}, ONE multi-task job per (arm, LR) on contradiction+nq+outlier+oolong (fast8k mt4 shard), Qwen3.5-4B base
status: active
updated: 2026-10-28
branch: prasann/flopdrop_clean @ eb3c9dd1e (pushed); Beaker -tde runs pinned at origin/claude/online-router-1bc01a 67a28d605
launcher: research/token_routing/ds64_fast8k/launch_fast8k.py (env F8K_SUFFIX=-lra, OPS_EXPERIMENT=tokdrop_lr)
records:
  - records/flopdrop-final-figures-plan.md
  - records/lora-vs-routing-ablation.md (branch claude/vibrant-williams-4b0f66, unmerged)
  - records/layer-skip-screen.md (branch claude/train-layerskip-35010b, unmerged)
  - records/online-token-router.md (router branch claude/online-router-1bc01a: hill-climb + gradient fidelity v2)
results:
experiment_log: research/token_routing/ds64_fast8k/experiment_log.json (standing rule 2026-10-29: keep current, every session; renders in the hub's Experiment log)
artifacts:
  - https://claude.ai/artifact/SPSBxHFs2PUn5CPjJX3Wgt (overview, v20, the summary page)
  - https://claude.ai/artifact/RFpSfNnqnz8M9ygi8kK1U5
wandb_group: https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/f8k-lra-q35-4b
next_step: (10-28) RECIPE = whole-model LoRA + two-stage 7:0.2,16:0.07, no full-context steps: dense FLOPs-to-match 3.0x / 3.0x / 2.0x at fresh 1,952 / 3,904 / 7,808 rows/task (f7808 now a point estimate; dense curve extended to f5856/f7808). 2x2: routing is most of the gain, LoRA 15-35% on 4-task. Router hill-climb (threshold, MLP, ctx feats, task id, per-task, keep sched) all tie; early drop (<block 7) fails by bias even for an oracle (fidelity v2). Layer skip (44 runs): ties dense at 0.78-0.89x, hurts on the recipe. IN FLIGHT: m70c50 (adaptive stage-2 budget, 0.870 on 1 seed vs recipe 0.852) seeds 1-2 + mlprg sd1 (router session, Beaker). PENDING PRASANN: fresh-pool curves for layer-skip arms (recommend skip); early-split sweep of the recipe at blocks 1/3/5/7; per-row drop depth; matched SFT+CTC of CPT ckpts. Hourly cron check-in drives this.
---

# tokdrop_lr

## Question
Every token-drop run so far (ds64, fast2k, fast8k) trained dense AND compressed arms at LR 5e-6
(setA dense SFT uses 4e-5; train_ctc_suite's default is 5e-5), and the LR was never swept. Does a
different LR change token dropping's standing against dense? A compressed arm sees fewer real
tokens per step, so its best LR may differ from dense's. Stage T2-LR of
`records/flopdrop-final-figures-plan.md`.

## Decisions
- 2026-10-19 (Prasann): use the gold + fl20 + p8 heuristic (`gold_fl20p8_noslot`), the devloss grid's
  best task-general rule. One approach for every task; only global knobs are tuned.
- 2026-10-19 (Prasann): go-ahead to push and launch.
- 2026-10-19 (agent): fast8k screen (ds64 8k pool, unpacked, 2 GPUs, 16 rows/step, 16M nominal =
  1953 rows = 123 steps, base q35-4b-base-markerfix), eval 8k + 2k rungs × 500. Grid: {dense,
  gfl20p8ns} × LR {5e-6, 1.5e-5, 5e-5} × {contradiction, nq, outlier, oolong} = 24 runs on jupiter,
  urgent. Pick ONE LR per arm, shared across tasks, by matched-FLOP f1 pooled over tasks.
- 2026-10-19 (Prasann): combine the 4 tasks into multi-task jobs: 6 jobs = {dense, gfl20p8ns} × 3 LRs
  (dense and token-drop are separate trainings, so not 3). Shard `mt4_g64M` = the four 16M task shards
  merged (`merge_tasks.py`, `build_mt4_beaker.sh`): 7,812 rows, shuffled together; gb 16 rows/step
  (the fast8k convention, so 5e-6 stays comparable) = 489 steps; 4 GPUs, micro 2, all six identical
  geometry. Each checkpoint is evaluated per task (`<run>-ev<task>`).
- 2026-10-19 (agent): gfl20p8ns uses `--st-n-random 0`, not `--st-keep-frac 0.0` (the fraction path
  floors at one random non-gold doc kept whole; the grid keeps gold only). oolong has NO gold docs and
  its sidecar was empty -> every oolong row would have been a fingerprint miss = trained UNCOMPRESSED.
  Re-tokenized with the new converter flag `--gold-sidecar-empty-for-missing` (fp -> []).
- 2026-10-19 (Prasann, conventions): every Beaker GPU job sets minRuntime 1h (memory:
  beaker-min-runtime). The first smoke job (minRuntime 0) never scheduled and was cancelled.
- 2026-10-19 (agent): campaign suffix `-lra` on every run name: fresh weka save folders, so nothing
  auto-resumes into the September f8k-outlier-* runs. Names: `f8k-<task>-<arm>-16M-lra-lr<lr>`.
- 2026-10-19 (agent): trainer change (7434103af) lets `--st-header-extra-tokens 8` combine with
  `--st-keep-token-rule first_last --st-keep-token-k 0.2`. CPU check
  `research/token_routing/ds64_fast8k/check_gfl20p8_flags.py` PASSED on horton: the parsed options
  equal the grid scheme field by field.

## Lambda design (2026-10-20, Prasann: "do the LR sweep on lambda ... very small dataset")
- Beaker jupiter would not schedule the 4-GPU job (~3 h pending, twice); moved to Lambda A100s.
- Data: `mt4s488` = first 488 rows of each 16M task shard (1,952 rows; 122 steps at 16 rows/step, the
  September fast8k step count), built by `build_mt4_small_s3_beaker.sh` -> S3 -> horton /data ->
  streamed into node-local /tmp/prasann/tdlr on lambda-hyperplane02 and 05 (Lambda /accounts at quota).
  Base q35-4b-base-markerfix streamed from horton too. SEQ_LEN 10240 (max row 9,757).
- Per job: 2 A100, gb 16, micro 2, shard 2, SAVE/WORK on node /tmp. Evals: `run_eval_lambda.sbatch`
  (same runner and args as the Beaker evals) on the node that holds the checkpoint.
- Lambda fair share: node02 4 jobs at preemptive_high (8 GPUs), node05 2 jobs at preemptive (4 GPUs).

## Runs
| what | id | state |
|---|---|---|
| data build nq 16M | 01M5ATRD655S01EW7V8HSH8F99 | launched 12:38 |
| data build contradiction 16M | 01M5ATRH70B625Z4047D00FANS | launched 12:38 |
| data build oolong 16M | 01M5ATRNCZ0A7PX66AN9E8PZTP | launched 12:38 |
| ~~smoke = f8k-outlier-gfl20p8ns-16M-lra-lr1.5e-05~~ | 01M5ATXYN75PMM399BVP0RPZ1G | CANCELLED: never scheduled (minRuntime 0); superseded by mt4 |
| data build mt4_g64M (+ oolong_g16M_eg re-tokenize) | 01M5AWJYWSMN16X841MGWQKFED | DONE 13:15: 7,812 rows, 7,812 gold rows (oolong 1,953 empty entries), max_example_len 10,162 |
| smoke = f8k-mt4-gfl20p8ns-64M-lra-lr1.5e-05 (4 GPU, minRuntime 1h) | 01M5AWYYKM2N6XWW9RZM8C62MT | FAILED 10-19 23:31 after ~6h queue: marker audit `load_keys` in a distributed context (every multi-GPU train_ctc_suite launch on this branch). Fixed 4d8d22941 (audit read in a child process; verified under 2-proc torchrun on horton; run_guards_test 23 pass) |
| smoke relaunch (same name, --force; no checkpoint was written) | 01M5D7VB5SMN56NGKYJMSVCQK3 | CANCELLED 10-20: still pending after ~3 h; superseded by Lambda |
| data mt4s488 -> S3 | 01M5DJHXA0CY8KAXZVCY44EPGS | DONE: 1,952 rows, max 9,757 tokens, 172 MB incl. eval files |
| Lambda smoke = f8k-mt4s-gfl20p8ns-lra-lr1.5e-05-lam | slurm 113092 (node02) | FAILED at start: model-side enable_pooled_soft_tokens also rejected first_last+prefix; fixed 18f90d7ef (+ unit test) |
| gfl20p8ns lr1.5e-5 (-r2, smoke) | slurm 113093 node02 preemptive_high | RUNNING; marker audit PASS in torchrun; keep-token real frac 0.26-0.37; CE 0.97 -> 0.15 by step 5; ~50 s/step |
| gfl20p8ns lr5e-6 | slurm 113094 node02 preemptive_high | RUNNING (launched 10-20 14:41) |
| gfl20p8ns lr5e-5 | slurm 113095 node02 preemptive_high | RUNNING |
| dense lr5e-6 | slurm 113096 node02 preemptive_high | RUNNING |
| dense lr1.5e-5 | slurm 113097 node05 preemptive | RUNNING |
| dense lr5e-5 | slurm 113098 node05 preemptive | RUNNING |

## Current best numbers
| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|

## Next step
See `next_step` above.

## Universal-rule screen (2026-10-21/22)
- Goal (Prasann): ONE keep rule, optional per-doc 0/1 gold flag, Pareto on all 22 tasks; start with the 4-task mix.
- Recipe: mt4s488 mix (1,952 rows, 122 steps), LR 2e-6 for rules; dense 5e-6 at 31/61/122 steps = matched-FLOP curve.
- Results so far (8k, Δf1 vs dense at matched FLOPs, eval_size 500): gold-blind fraction rules (fl20p8, spans20) fail
  contradiction (−0.17 / −0.39); gold-aware rules with decoys whole fail outlier (bfd33 −0.13, bfd50 −0.07, gold+⅔ whole
  −0.06); gold-blind fixed 64 tokens/doc (gbflk64p8, 0.66× FLOPs) is closest (worst −0.03 nq); gbflk32p8 (0.47×) ties
  contradiction/outlier/oolong, nq −0.07. At 2k several rules tie/beat dense on all four.
- Run-to-run noise (3 accidental gbfl20p8 repeats): ±~0.02 contradiction, ±~0.03 outlier at 8k.
- BUG (fixed 2026-10-22, commit after f889c827e): --st-kept-doc-k / --st-keep-token-min / --st-pad-multiple /
  --st-span-len were not SoftTokenOptions fields → silently defaulted. recipes/train/variants_options_test.py guards it.

## Open issues / traps
- The Beaker image (OLMoCoreBeakerImage.stable) now ships torch 2.13.0+cu129, while this branch was
  validated on torch 2.9.1 / fla 0.4.1. The first smoke never reached model init, so this is untested.
- Outlier with gold kept whole is the known gold-shortcut setting (ds64 had to go gold-blind on
  outlier); expect gfl20p8ns to look bad there at every LR. That is still informative for the
  global gold-whole-rate knob q.
- The contradiction eval file (`_eval_bundle_eval500_v3`, realistic mode) predates the canonical
  ordering of unordered gold pairs (a0ed3898a). Check the grader treats pairs as unordered before
  quoting contradiction numbers.
- An 8k screen is not the ladder: a fast8k win licenses a 32k run, it does not replace one.
- F8K_SEQ_LEN must be ≥ the shard's max_example_len, or rows are silently skipped. Task max lengths:
  nq 8,185 / contradiction 9,722 / oolong 10,051 / outlier ≤12,288 → 12288 for mt4.
- The three task builds exited 1 at the stats step: `ds64_fast2k/fast2k_stats.py` was never ported to
  this branch, so the train-vs-eval overlap check did NOT run for contradiction/nq/oolong. Run it from
  the legacy checkout before quoting.
- FLOPs come from `flop_meter/actual_pflops` of FINISHED runs only.
- Eval rows are the ds64 ladder files, not the final olmo-eval suite rows; check disjointness before
  treating this as dev-set tuning.

## Verified vs unverified
- Verified 2026-10-19: CPU flag check (above); 7434103af pushed to origin/prasann/flopdrop_clean.
- UNVERIFIED: everything about training and eval outcomes.
