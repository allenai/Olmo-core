---
experiment: tokdrop_lr
title: LR ablation for token-drop training — dense vs gold_fl20p8_noslot (gfl20p8ns) × LR {5e-6, 1.5e-5, 5e-5}, ONE multi-task job per (arm, LR) on contradiction+nq+outlier+oolong (fast8k mt4 shard), Qwen3.5-4B base
status: active
updated: 2026-11-09
branch: prasann/flopdrop_clean (pushed; hub/log/launcher); Beaker -tde runs pinned at origin/claude/online-router-1bc01a 67a28d605
launcher: research/token_routing/ds64_fast8k/launch_fast8k.py (env F8K_SUFFIX=-lra, OPS_EXPERIMENT=tokdrop_lr)
records:
  - records/flopdrop-final-figures-plan.md
  - records/lora-vs-routing-ablation.md (branch claude/vibrant-williams-4b0f66, unmerged)
  - records/layer-skip-screen.md (branch claude/train-layerskip-35010b, unmerged)
  - records/online-token-router.md (router branch claude/online-router-1bc01a: hill-climb + gradient fidelity v2)
  - records/multiview-cached-h7.md (branch claude/multiview-h7cache, unmerged: k late-stack views per row from cached h7)
  - records/tokdrop-large-data-decay.md (branch claude/tokdrop-decay-fix, unmerged)
  - records/grad-routing-gap.md (branch claude/grad-routing-gap, unmerged; training flags on claude/grad-routing-gap-train)
  - records/sparse-backward.md (branch claude/sparse-backward, unmerged)
  - records/oracle-selection-training.md (branch claude/oracle-train-bd7414, unmerged)
  - records/tokdrop-model-scale-ablation.md (branch claude/recipe-scale-ablation-71d007, unmerged)
  - records/oracle-tokens-under-routing.md (branch claude/grad-support-tokens, unmerged: 0.95 oracle token sets under true routing, frozen dense)
  - records/posid-reset-recipe-ablation.md (branch claude/posid-reset-0abac3, unmerged: position-id reset after each cut; -0.026/-0.015 8k, outlier only)
  - records/tail-rule-recipe-ablation.md (branch claude/posid-reset-0abac3, unmerged: fixed per-document tail instead of the router; -0.10 / -0.065 8k)
  - records/tokdrop-handoff-2026-11-09.md (START HERE for a new main agent: goal, state locations, results, traps)
  - records/perrow-floor-recipe-ablation.md (branch claude/otr-perrow-minfloor @ 315f37b03, unmerged: per-row budget; ties / trails at 1,952 and 3,904)
  - records/cross-row-allocation.md (branch claude/cross-row-alloc, unmerged: step threshold ties, selective backprop loses)
  - records/mt11-64k-length-generalization.md (branch claude/mt11-lengen-bed337, unmerged: gain persists to 128k/256k)
  - records/lit-review-efficient-long-context-training.md (branch claude/lit-review-efficient-lc-eacf2d, unmerged)
results:
experiment_log: research/token_routing/ds64_fast8k/experiment_log.json (standing rule 2026-10-29: keep current, every session; renders in the hub's Experiment log)
artifacts:
  - https://claude.ai/artifact/SPSBxHFs2PUn5CPjJX3Wgt (overview hub, v97 on 11-09, the summary page; all results incl. side sessions; source research/token_routing/ds64_fast8k/hub_live/hub.html)
  - https://claude.ai/artifact/RFpSfNnqnz8M9ygi8kK1U5
  - https://claude.ai/artifact/Y15xcuaVDVhz737nw9g5VZ (multi-view block-7 cache, 10-30 overnight)
  - https://claude.ai/artifact/9vAYqzh69sXf1RjVFfsnNV (oracle token-selection training, 9 arms, 11-04; record oracle-selection-training.md)
wandb_group: https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/f8k-lra-q35-4b
next_step: (11-09) Handoff: records/tokdrop-handoff-2026-11-09.md. RECIPE unchanged (whole-model LoRA, router 7:0.2,16:0.07, per-document): 8k-only FTM 4.6x / 3.6x / >=4.5x at 1,952 / 3,904 / 7,808 rows/task. Every >=5x idea so far failed (sparse backward, multi-view, KV-only, layer skip, aggressive keep, late widening; per-row / cross-row allocation, position-id reset, fixed tail rule, oracle-selection training all tie or lose). NOTHING RUNNING; all side sessions archived. PENDING PRASANN: larger-data push (15,616 rows, 11-task, 32k-64k); new approach to blocks 0-6; merge side branches; offered: per-layer router-vs-true-gradient check, two-cut orcbatchr2. METRIC (Prasann 11-02): rank by the 8k rung; 2k secondary.
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

## Multi-view from a cached block-7 state (2026-10-30 overnight, branch claude/multiview-h7cache)
- Idea: blocks 0-6 once per row (cache h7), k optimizer steps on blocks 7-31 + head each with a fresh RANDOM 20%/7%
  subset, one early-stack backward on the MEAN of the k views' dL/dh7. Flag `--st-otr-views k`
  (src/olmo_core/nn/multiview.py). Record: records/multiview-cached-h7.md.
- FLOP meter: repeat views log no example lengths and are charged late blocks + head at the kept length (unit-tested;
  Beaker smoke k=1 vs k=2 on identical rows: same dense-equivalent 19.694 PF, actual 5.981 vs 7.438 PF -> the late
  stack is ~54% of a routed step; expected full-run cost ~1.54x (k=2) / ~2.08x (k=3) the recipe's PF).
- FSDP2 trap fixed (1f57a5feb): a backward started from an intermediate block output never fires the ROOT module's
  pre-backward; the pooled projector (root params) feeds block 0 -> `setStorage ... size 0`. CPU repro in
  research/token_routing/multiview/fsdp_cpu_repro.py.
- IN FLIGHT (Beaker jupiter, 2 H100, urgent, launched 10-30 ~09:00Z, commit 1f57a5feb; state
  research/token_routing/multiview/state.json):
  f8k-mt4-lora64all32rnd20to7mv2-f1952-tdmv-lr1e-05 (01M660BDJE2ZQA3YE2AQQS20FA),
  ...mv3-f1952 (01M660CBN2GSBT9AQ954ZB1HGK), ...mv2-f3904 (01M660D44KG044R94YW2CYDC8T),
  ...mv3-f3904 (01M660DX5J61G6CSZ21MXAF1X3); k=1 same-code-path control ...mv1-f1952 (01M6662Z9T213KKGJC8M3ZA215).
  mv3-f1952 and both f3904 runs were preempted on jupiter and restarted from step 0.
- FIRST RESULT (10-30): mv2-f1952 = 0.885 8-cell mean (eval_size 500/cell) at 483.0 PF (0.310x dense). Recipe
  interpolated at 483 PF = 0.899 -> -0.014 at matched FLOPs (single seed; noise ~+-0.01); vs the recipe at the
  same rows (0.883 @ 345 PF) only +0.002. The second view buys almost nothing; not a win so far.
- k=1 CONTROL (same code path, random choice): mv1-f1952 = 0.853 at 343.4 PF (outlier 2k 0.690 drags it; router
  recipe 0.883). vs that control mv2 gains +0.032 for 1.41x FLOPs. Ambiguous: decided by whether random-k1 at f1952
  really is 0.03 below the router (seed replicate needed).
- mv3-f1952 = 0.842 at 622.6 PF (recipe at that PF 0.910 -> -0.068); below k=2 AND the k=1 control (oolong 0.571/0.798,
  outlier@2k 0.651): a third view overfits. Verdict at f1952: multi-view does not beat the recipe.
- mv2-f3904 = 0.909 at 965.7 PF (recipe at that PF 0.919 -> -0.011); below the k=1 recipe at the same rows (0.915
  router / 0.914 random).
- mv3-f3904 = 0.906 at 1245.4 PF (recipe at that PF 0.923 -> -0.017). VERDICT (all 5 scored): multi-view is
  NEGATIVE - no k>1 run beats the recipe at matched FLOPs; extra views act like extra epochs. No seed replicate.
- Collect: when a run finishes, set its state DONE in that state.json, `launch_mv.py eval`, then
  `ds64_fast8k/record_beaker_evals.py --state research/token_routing/multiview/state.json --suffix -tdmv --out rows.jsonl`
  and `multiview/collect_mv.py` (8-cell mean vs the recipe curve interpolated in log PF; decisive = +0.01).

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

## Large-data decay diagnosis + fixes (2026-10-30, side session; branch claude/tokdrop-decay-fix)
- Record: records/tokdrop-large-data-decay.md (on that branch). Code/state: research/token_routing/decay_fix/
  (state.json = every job id; results/ce/*.json answer CE; results/f1_points.json ladder f1).
- Diagnosis (answer CE, eval_size 500/cell, no-cot): dense CE flat from 3,904 rows (f7808 - f3904 = -0.0017 +- 0.0037);
  recipe matched-data CE gap +0.010 / +0.014 / +0.008 at 1,952 / 3,904 / 7,808 (no trend). CE FTM 2.7x -> 1.8x,
  f1 FTM 3.7x -> 2.2x, unsat-4 f1 4.0x -> 2.2x. Verdict: mostly the dense plateau amplifying a constant ~0.01-nat gap.
- Fixes at f3904 (1 seed): ReLoRA rl25 = no effect (CE +0.003 +- 0.002); late keep-widening lw35 (35%/15% last third,
  +8% FLOPs) CE -0.0069 +- 0.0022 vs recipe (CE FTM 3.2x), f1 unchanged (0.9157 vs 0.9153).
- IN FLIGHT (Beaker jupiter): rl25 / lw35 / recipe-sd1 at f7808 (preempted once at step ~1315, restarted from 0),
  uw25 (uniform 25%/9.7%, compute-matched control) f3904+f7808, lw50 f3904. Collect: fetch_ce.sh, record_ce.py,
  record_ladder.py, analyze_decay.py / paired_ce.py (commands in the record).
- 2026-10-30 final (decay fix): lw35 (late keep 35%/15% in the last third) at f7808, 1490 PF: f1 0.9335 (dense f7808
  0.9310), answer CE 0.1365 (dense 0.1374) -> FTM >= 4.1x f1 / >= 3.1x CE vs recipe 2.2-2.8x / 1.8-2.0x; CE vs recipe
  seeds -0.0091 +- 0.0027 / -0.0056 +- 0.0025. Uniform 25%/9.7% at equal FLOPs: f1 0.9319, CE 0.1398 (gain not robust
  across seeds). 1 seed each; next = lw35 seed 2 at f7808. Record: records/tokdrop-large-data-decay.md (branch
  claude/tokdrop-decay-fix @ a3d8f2e50). Hub pareto_points.json is contaminated by answer_ce rows (collector fix needed).
- 2026-10-31 REVISED: lw35 seed 2 at f7808 (1490 PF): f1 0.9276, CE 0.1443 (seed 0 was a lucky draw). 2-seed means
  lw35 0.9306 f1 / 0.1404 CE vs recipe 0.9257 / 0.1439: +0.005 f1 (seed SE ~0.003), -0.0035 CE (+-0.0017 eval-only,
  ~+-0.004 with seed spread) -> within seed noise, below the +0.006 bar; no dense CE parity. Seed variance at f7808 is as
  large as the effects; FTM claims here need >= 3 seeds per arm. Branch claude/tokdrop-decay-fix @ b01502969.

## 2026-10-30 → 11-02 summary (main session + overnight agents; details in the hub and records)
- Large-data decay (claude/tokdrop-decay-fix): dense saturates from 3,904 rows (CE flat); recipe gap ~0.01 nats, no
  trend; ReLoRA null; late widening 35/15% 2-seed mean within noise (f1 +0.005, CE -0.0035; +8.6% FLOPs). Adds
  --ephemeral-save-interval / --resume-every (Beaker resume verified).
- Sparse backward (claude/sparse-backward): 0.11-0.13x/step but -0.2 to -0.37 at matched data. Kept: answer-only head.
- Multi-view (claude/multiview-h7cache): every k>1 trails the recipe at matched FLOPs.
- Aggressive keep r10to3 / r10to5 (this branch's launcher): 3.1x / 2.4x at f1952, 3.9x / 3.7x at f3904.
- Gradient-gap (claude/grad-routing-gap): dense-ckpt routed gradient exp cos 0.04 (router) vs attribution 0.996;
  recipe ckpt 0.87; early-phase probe: unbiased by step ~31; aggressive budgets decay to ~0.38.
- Qwen3 sizes (claude/qwen3-recipe-c876d1 @ 5593eb001): 0.6B 2.9/2.9x, 1.7B 2.9/2.9x, 4B 2.1/3.3x, 8B 2.3/2.6x.
- Scale (claude/recipe-scale-ablation-71d007): 0.8B 1.7x, 2B 2.4x, 4B 4.1x, 9B 4.8x at f1952.
- Gradient concentration (claude/grad-concentration-d91a37): partial, 8/11 stages; signal more concentrated later.
- Collector fix: collect_pareto_points drops answer_ce/top1 and gradgap rows (they share train runs).

## Per-row budget + per-document floor on the recipe (2026-11-02, side session; branch claude/otr-perrow-minfloor)
- Question: does giving the ROW the token budget (`--st-otr-doc-share 0`, spread over documents by router score) with a
  floor of K tokens per document (`--st-otr-min-per-doc K`, counted against the budget, applied at BOTH stages) beat
  the recipe's per-document budget at the 8k rung? Arms: ds0 (no floor), ds0f1, ds0f2; 1,952 rows/task x 2 seeds +
  recipe seed-1 control; best arm then at 3,904 rows/task.
- Code: branch claude/otr-perrow-minfloor (cut from router tip 8ce5fd22a). Floor + per-row path verified (CPU test
  test_pyramid_per_row_budget_with_floor_cpu: floor >= K per doc after each cut, budgets exact). New diagnostics on the
  last stage: online_router/s1_keep_frac_abs, s1_leak_auc_abs (absolute per-document kept fraction after both cuts vs
  gold; stage-1 leak_auc alone cannot see a stage-2 concentration on gold).
- Smoke PASS on cubbins (slurm 4145927, 4 steps): keep 0.200 after block 7, 0.070 absolute after block 16 (the
  recipe's per-document ceil realizes ~0.206 / ~0.079), floor adds 2% (stage 1) / 6-15% (stage 2) of tokens.
- RESULT (1,952 rows/task, 2 seeds each, eval_size 500 per task x rung, no-cot; record
  records/perrow-floor-recipe-ablation.md): **per-row + floor does NOT beat the per-document recipe at 8k.**
  4-task 8k mean (2-seed): recipe 0.846 (0.845/0.847) at 344 PF, FTM 4.62x; per-row no floor 0.843 (0.838/0.848) at
  337 PF, FTM 4.55x; floor 1 0.831 (0.837/0.824); floor 2 0.840 (0.846/0.835). 2k: 0.920 / 0.921 / 0.914 / 0.905.
  SE of one run's 4-task mean ~0.008; seed spread up to 0.012, so no-floor = recipe within noise; floors cost 0.006-0.016,
  almost all on outlier (8k 0.861 recipe vs 0.816 floor 1 / 0.835 floor 2). Per-row arms realize exactly 20.0% / 7.0%
  keep (recipe 20.6% / 7.9%, so -2.3% PF) and concentrate on gold documents (leak_auc 0.60-0.64 stage 1, 0.73-0.75
  absolute after both cuts; recipe 0.48). Runs (Beaker jupiter H100, commit dac2b81d5): ds0 01M6F3THAA49E0WV59CG38PZTW,
  ds0f1 01M6F3VA0RJX2X94Z7ZY0QBE0F, ds0f2 01M6F3W357QSXWEN0FAQ71YEHY, ds0sd1 01M6F3WYDK7BY2P70WBA53SYS5, ds0f1sd1
  01M6F3XQTP4EH8NCR8DBY47PC7, ds0f2sd1 01M6F3YGRVME6VS9PZG5R3GY1Y, recipe sd1 01M6F3ZAR3PENM133YFNVHDTB5.
- 3,904 rows/task (best arm, per-row no floor, 1 seed; Beaker 01M6HCRGTD1AZB2YKX2FAQ53TR, 672.5 PF): 8k 0.880
  (contra 0.979, nq 0.928, outlier 0.950, oolong 0.661), 2k 0.950, FTM 3.58x, vs recipe 0.882 / 0.889 (8k), 0.949 /
  0.945 (2k), FTM 3.61x (seed 0, 688 PF). Ties or trails slightly; thread DONE, no further runs planned.

## Position-id reset ablation on the recipe (2026-11-04, side session; branch claude/posid-reset-0abac3)
- Question (Prasann: "an ablation where you don't fix position ids ... should be much worse but is just a
  confirmation"): the recipe keeps each kept token's ORIGINAL position id after each cut. New flag
  `--st-otr-reset-positions` renumbers the kept row 0..k-1 after each cut (stage 2 renumbers stage 1's row; packed rows
  per document); eval stays dense with true positions. Arm `lora64all32r20to7pr` = the recipe + that flag.
- Code: branch claude/posid-reset-0abac3 (cut from claude/otr-perrow-minfloor 635264704, same router as the recipe sd1
  control). Change is in `OnlineTokenRouter.mid_compact` / `_mid_compact_packed`; CPU tests (positions 0..W-1 at
  blocks after each cut vs original positions with gaps by default; default path bit-identical) + GPU packed test pass.
  Launcher `research/token_routing/online_router/launch_posid_reset.py`, collector `collect_posid_reset.py`.
- DONE (1 seed each; Beaker 01M6JTWWY7PDKMQ3YZ4TT7V6K6 / 01M6JTY2EEQ82843KR8DQ87FYB; eval_size 500/cell, no-cot):
  resetting positions costs 0.026 / 0.015 on the 8k mean at 1,952 / 3,904 rows/task (0.820 vs recipe 0.845/0.847;
  0.870 vs 0.882/0.889), almost all on outlier (8k .764 vs .862/.861, .919 vs .970/.967; 2k .737 vs .880, .795 vs
  .972); contra ~-0.01 in all 4 cells; nq, oolong unchanged. 2k mean 0.881 / 0.896 vs 0.92 / 0.947. Training CE
  matches the recipe (last 25%: .1005 vs .0956/.0979; .0587 vs .058/.064), so the cost is the train/eval position
  mismatch. 8k FTM <=4.46x / 3.13x vs 4.56-4.69x / 3.61-3.93x. Record: records/posid-reset-recipe-ablation.md.
- Smoke PASS (mooney 1x H200, 2 rows/step, 4 steps; slurm 4150009-11, 4150013-14): step 1 (dense warmup) CE 1.588 in
  all 5 runs; step 2 (first routed) picks identical tokens in all (keep 0.2057, leak_auc 0.4896, mid_row 0.2402); CE
  old code 1.226 / 1.236 (repeat = GPU nondeterminism), new default 1.237 / 1.239, reset 1.348.

## Fixed per-document TAIL rule on the recipe (2026-11-04, side session; branch claude/posid-reset-0abac3)
- Question (Prasann): does the model simply adapt to whatever drop shape it trains with? Content-blind rule
  `--st-otr-score tail`: each document keeps its LAST body tokens (ceil, >=1; nested across stages); non-body tokens
  always kept; no exploration; original position ids; router still fit (dense-warmup gate) but its FLOPs uncharged.
  Arms: `lora64all32t10` = one cut at block 7, last 10%; `lora64all32t20to7` = recipe shape (last 20% at 7, last 7%
  nested at 16; FLOP-matched control). 1,952 rows/task, LR 1e-5, mt4; 1 seed first, 2nd seed if within ~0.02 of recipe.
- Code b471298dc (rule + CPU test `test_tail_rule_keeps_exactly_each_documents_tail_cpu`: kept set == exact per-doc
  tail at 1 and 2 stages, no router FLOPs charged); launcher `research/token_routing/online_router/launch_tail_rule.py`
  (state `tail_rule_state.json`); collector `collect_posid_reset.py --state tail_rule_state.json`. Router tests 90 pass
  on mooney (slurm 4152635).
- Smoke PASS (mooney 1x H200, 4 steps, slurm 4152752-3): keep 0.107/0.103 (t10); t20to7 0.2057 + abs 0.0796 at block
  16 (= recipe budget); step-2 CE 1.49/1.48 vs recipe ~1.24 (different tokens); PF 0.5440 vs recipe 0.5452 (router
  overhead uncharged).
- DONE (Beaker, commit b076f2571; t10 01M6MC1WN4S0QGCFFYMV89820T, t20to7 01M6MC2T4E8HMVRZAHRM38X939; 8 evals,
  eval_size 500 per cell, no-cot, all with the adapters-re-installed line). **Both far below the recipe: no 2nd seed.**
  8k mean: tail10 0.747, tail20to7 0.781 vs recipe 0.845/0.847, dense 0.843, recipe-random (1,464 rows) 0.782,
  posid-reset 0.820. 2k mean: 0.815 / 0.851 vs recipe 0.921/0.920.
  - 8k contra/nq/outlier/oolong: t10 .919/.810/.617/.640; t20to7 .921/.856/.711/.637 (recipe .958/.917/.862/.649 avg).
  - 2k: t10 .958/.966/.497/.839; t20to7 .952/.956/.658/.838 (recipe .976/.970/.881/.854 avg).
  - Keep 0.1063 (t10) / 0.2056 then 0.3845 of it (t20to7, = recipe). PF 335.6 / 342.5 (recipe 345.3/343.2). Train CE
    first 50 / last 25%: .764/.120, .660/.117 (recipe .452/.096). FTM <=4.56x / <=4.47x (below dense f1952; no
    smaller dense point), i.e. worse than dense at equal rows.
  - Reading: the model does NOT just adapt to the drop shape. At matched FLOPs and keep fractions the content-blind
    tail loses 0.065 (8k) to the router (it lands at recipe-random's 0.782, but that was 1,464 rows); outlier carries most of it (-0.15 8k, -0.22
    2k), then nq at 8k (-0.06). Record: records/tail-rule-recipe-ablation.md.

## High-gradient vs supporting tokens in the true routed gradient (2026-11-03, side session; branch claude/grad-support-tokens)
- Question: which tokens does the true routed gradient (compaction at a split, original RoPE positions) need? Does
  keeping the top gate-gradient tokens (|dCE/dg| at g=0.5 / g=1) recover cos >= 0.95 with the dense gradient, vs the
  own-gradient-energy oracle and random? Plus 0.95-attribution budgets, gate-mass concentration, joint sets, Jaccard.
- Code: research/token_routing/grad_concentration/{support_probe.py, analyze_support.py, launch_support.py,
  soft_parity.py} on claude/grad-support-tokens @ 9682de9f9 (pushed). No record yet.
- ⚠ NOT YET CONFIRMED (one 16-row step, no error bars; run-to-run per-step cos noise up to +-0.035).
  Quick pass, dense 1,952 rows/task ckpt, block-7 split, across-batch, H200 mooney (slurm 4145986), ⚠ eval_size 16 rows;
  per-step cos(routed, dense) at k = 3 / 7 / 20%: own-gradient 0.115 / 0.477 / 0.328; gate g=0.5 0.001 / 0.008 / 0.125;
  gate g=1 0.047 / 0.137 / 0.257; random 0.008 / 0.012 / 0.045. Routed answer CE at 7%: own-grad 0.372, g0.5 0.419,
  g1 0.428, random 0.938 (dense 0.050). Reading: gate selection keeps CE near the oracle but not the gradient; nothing
  near 0.95 at block 7 up to 20%.
- ⚠ 4-row smoke (slurm 4145974, eval_size 4 rows): a joint set (95% gate mass + 0.95 attribution) keeps ~39% of body
  tokens at block 7, routed cos 0.97; at block 16 / 10%, gate g=0.5 across-batch 0.984 vs own-grad 0.989. 95% gate mass
  is diffuse (joint sets 30-89% of tokens).
- Parity: soft-gate path at g=1 equals dense in whole-model fp32 to <=5e-4; the bf16 gap (up to ~10-16%) is attention
  kernel noise. Gate passes now use fp32 attention.
- HELD (Prasann 11-03: "Hold off on this for now"): full sweep NOT launched. Ready config: `launch_support.py --labels
  f1952,recipe --preset trimmed` (splits 7/16, fewer k; ~23 H100-h per ckpt, 16 one-GPU Beaker jobs each; Beaker smoke
  01M6F50NS04RYSMM1XV7V89JXT passed). Weka SUPPORT_DIR/smoke-recipe is smoke output: ignore before analysis.

## Oracle token sets under true routing (2026-11-04, side session; branch claude/grad-support-tokens)
- Question (Prasann): do the 0.95 gradient-concentration oracle sets hold up when the other tokens are dropped from
  the forward? Reading confirmed ("version A"): the oracle was already token-level (each body token kept/dropped in
  all of blocks 7-31); routed = compaction at block 7, blocks 0-6 dense. Record: records/oracle-tokens-under-routing.md.
- RESULT (frozen dense 1,952 / 7,808 rows/task, eval_size 512 rows each = 32 steps, raw gradient, per-step cos ± SE):
  **no**. Budgets reproduce the table (global 2.0 / 0.6%, per row 9.0 / 5.6%, per doc 19.5 / 15.9% at block 7), but
  routed cos with the dense gradient is 0.137 / 0.187 / 0.164 (1,952) and 0.080 / 0.144 / 0.081 (7,808) over blocks
  7-31 (whole model 0.07-0.15); norm 2-4x (1,952) and 15-25x (7,808) dense; routed answer CE 0.14-0.98 vs dense
  0.058 / 0.026. Attribution 0.9: 0.04-0.07. Dropped from every block (split 0, whole-model ranking): 0.03-0.13 at
  1-33% kept. Block 16 best: 1,952 whole-model 0.28-0.47; 7,808 0.05-0.12.
- Frozen-model reference only: the oracle-train session (local_fb441871) tests whether training under the selection
  adapts. Expected (cross-step debiased) cos, 1,952 rows/task: attribution 0.94-0.98, routed about 0 at block 7
  (-0.27 to 0.13, noisy, no SE), norm 4-13x; block 16 0.05-0.28. 7,808: mean dense gradient NOT resolvable over 32
  steps (debiased ||E[D]||^2 < 0 for whole model and blocks 7-31); blocks 16-31 only: attribution 0.98-0.99, routed
  0.14-0.16. Analyzer fix c3df16e02.
- ops: `ops result show --run-id gradsupp-q1tok512-f1952` / `-f7808` (218 rows, metrics q1tok_*). Beaker jobs in
  research/token_routing/grad_concentration/state_gap.json (q1tok512-*); sketches weka gradgap/support/q1tok512-*.
- Gate-gradient sweep (sections above) still on hold.


## Gradient maps: per-token x per-layer gradient (2026-11-04, side session; branch claude/grad-support-tokens)
- Heatmaps (weight-grad energy and ||dL/dh|| per block x token) on base / dense 1,952 / dense 7,808 / recipe, 8k rows
  (eval_size 512) + fresh 2k rows (⚠ eval_size 128), with oracle 0.95 strips, router keep sets, routed-vs-dense
  pairs and summaries: research/token_routing/grad_concentration/gradmaps/ (findings.md). For the hub "Gradient maps"
  page (parent session publishes; no separate artifact).
- Reading: answer+query tokens (2-10% of tokens) carry 60-95% of blocks 7-31 energy; gold docs 8-55x distractors per
  token (outlier 2.4-3.8x); training moves mass to block 31 and a step at layers ~17-20 drops distractor energy 1-2
  orders; token ranks agree within layers 0-15 and 17-31 but not across (block 7 vs 31 rho 0.33-0.39) -> supports
  per-half (2-stage) selection, not per-layer cells; cells are 2.5-3x more concentrated than tokens in ENERGY, which
  (per oracle-tokens-under-routing) does not translate into routed fidelity.
- The "270x routed/dense" on one nq row = one near-solved row's sum of squared per-token terms (not a norm ratio);
  routed pass matches the gap probe exactly; per 16-row batch energy ratio median 1.97 (~1.4x in norm).
## Oracle-selection TRAINING runs (2026-11-04, side session; branch claude/oracle-train-bd7414)
Record `records/oracle-selection-training.md` (on the branch). Code `src/olmo_core/nn/oracle_token_select.py` (3-pass step:
dense backward -> exact per-token LoRA-gradient attribution bins -> smallest top-k body set with attribution cos >= 0.95 ->
routed pass applied), `--oracle-*` flags, router `--st-otr-score oracle`; launcher / summariser / collector in
`research/token_routing/oracle_train/` (state.json there). Recipe base (lora64all32r20to7, LR 1e-5, f1952 = 488 steps).
- HEADLINE single cut at block 7 (Prasann via parent: token-level, blocks 0-6 dense). Cos is on the BODY tokens' part of
  the LoRA gradient (the always-kept prompt/query/answer/marker tokens alone give cos ~0.96 of the whole gradient early on).
- DONE (8k mean, eval_size 500/cell, no-cot): **orc1doc 0.850** (keeps 35.5% of body tokens; dense 0.843, recipe 0.845);
  **orc1batch 0.756** (keeps 19.1%; outlier 8k 0.537 vs dense 0.817). 2k means 0.902 / 0.804 (dense 0.933). Routed-grad
  cos ~0.5-0.6 in both, routed/dense grad norm 1.25, no loss spikes. Actual cost 2.3-2.4x dense per step (not a method).
  Recorded with `ops result add`.
- DONE 2026-11-04 evening: **orc1row 0.674** (per row, 29% kept; outlier 8k 0.155) and two-cut **orcdoc 0.716**
  (outlier 8k 0.351). Only the per-document single-cut oracle keeps outlier; the other three tasks stay near dense in
  every arm. The router recipe keeps outlier (0.862), so it is the oracle's gradient-energy ranking that fails it.
- DONE: **orc1batchr2 0.806** (8k; 2k 0.922): 13% batch-oracle + 17% random body tokens. Random negatives lift
  outlier 8k 0.537 -> 0.737 and routed-grad cos 0.52 -> 0.74 vs the oracle alone; at a matched ~30% keep the
  per-row oracle alone scores 0.674.
- DONE: **orc1batchrc 0.837** (8k; 2k 0.936 = dense): growing the set until the ROUTED gradient has cos >= 0.95 takes
  71% of body tokens (63% -> 80% over training; the attribution set at 5% gives routed cos 0.30). The only token sets
  that keep the dense gradient are ~70%+ of the body.
- DONE (all 9 arms, 2026-11-04 night): two-cut **orcbatch 0.793** (outlier 0.635), literal whole-gradient rule
  **orcbatchall 0.736**, **orc1docr2 0.867** (keeps ~94%, ~dense control). NOTHING RUNNING. Verdict: attribution-chosen
  sets give routed cos ~0.5 and tie dense only with per-document allocation at one cut; gradient-faithful sets need
  ~70%+ of the body; the 20/7% router works through co-adaptation, not gradient matching. Checkpoints: weka ckpts/
  (Beaker-trained + mooney copies pushed via S3), Lambda node02 /tmp/prasann/oracle/runs/ (orc1docr2, orcbatch-lam),
  mooney /data/prasann/oracle/runs/.
- RUNNING (Beaker unless noted; insurance copies on mooney / Lambda because jupiter preempts after minRuntime and runs
  have no mid-run checkpoint): orc1row, orc1batchr2 + orc1docr2 (random 2x extra), orc1batchrc (grow until ROUTED cos
  >= 0.95), two-cut orcdoc / orcbatch / orcbatchall (all three restarted after preemption).
- Metadata per run: `<weka>/ctc_suite/ckpts/<run>/oracle_meta/{steps.jsonl, sel/*.npz}` (mooney runs: also
  `/data/prasann/oracle/runs/<run>/` on mooney).

## 2026-11-04 → 11-09 close-out (main session)
- Tail rule (claude/posid-reset-0abac3 @ d121b6f6e): 8k 0.781 (recipe shape) / 0.747 (last 10%) vs recipe 0.845/0.847,
  1 seed, eval_size 500/cell; outlier carries it. The router's choice of tokens matters; the model does not just adapt
  to any drop shape.
- Per-row no floor at 3,904 rows/task (315f37b03): 8k 0.880 vs recipe 0.882/0.889, 2k 0.950 vs 0.949/0.945 (1 seed);
  FTM 3.58x vs 3.61x. Evals had finished 11-03 but sat uncollected (registered under ds64_fast8k).
- FTM sign fix: below the first dense point FTM is an upper bound; posid-reset f1952 <=4.46x, per-row floors <=4.55x.
- Hub v97: every log entry can carry tables/charts; new tabs Selection ablations (6c) and Prior work (8b); 128k/256k
  length-generalization and router-accuracy/gradient-norm sections. Live source saved in
  research/token_routing/ds64_fast8k/hub_live/ (the generator make_overview_page.py has drifted; do not regenerate).
- New log entries: gradient maps, router ranking accuracy + gradient magnitude, literature review.
- All 7 side sessions archived; branches pushed; their ignored results copied to
  OLMo-core-archive/worktree_files/<worktree>/.

