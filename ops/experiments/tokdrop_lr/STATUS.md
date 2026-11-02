---
experiment: tokdrop_lr
title: LR ablation for token-drop training — dense vs gold_fl20p8_noslot (gfl20p8ns) × LR {5e-6, 1.5e-5, 5e-5}, ONE multi-task job per (arm, LR) on contradiction+nq+outlier+oolong (fast8k mt4 shard), Qwen3.5-4B base
status: active
updated: 2026-11-02
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
  - records/tokdrop-model-scale-ablation.md (branch claude/recipe-scale-ablation-71d007, unmerged)
results:
experiment_log: research/token_routing/ds64_fast8k/experiment_log.json (standing rule 2026-10-29: keep current, every session; renders in the hub's Experiment log)
artifacts:
  - https://claude.ai/artifact/SPSBxHFs2PUn5CPjJX3Wgt (overview hub, v71, the summary page; all results incl. side sessions)
  - https://claude.ai/artifact/RFpSfNnqnz8M9ygi8kK1U5
  - https://claude.ai/artifact/Y15xcuaVDVhz737nw9g5VZ (multi-view block-7 cache, 10-30 overnight)
wandb_group: https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/f8k-lra-q35-4b
next_step: (11-02) RECIPE unchanged (whole-model LoRA, 7:0.2,16:0.07): FTM 4.1x / 3.7x / 2.2x at fresh 1,952 / 3,904 / 7,808 rows/task; the decay is dense saturating (CE flat from 3,904) + seed noise (recipe sd1 at 7,808 reads 2.8x), not the method degrading. No >=5x idea worked: sparse backward, multi-view, KV-only, layer skip, aggressive keep 10%->3/5% (ties at 3,904; blocks 0-6 dense floor ~0.2x/step), late widening (2 seeds: within noise). Scale: Qwen3.5 gain grows with size (4.8x at 9B, f1952); Qwen3 flat ~2-3x. Gradient-gap: attribution under dense forward is not a proxy; routed-trained model's router gradient is unbiased (exp cos 0.87) from step ~31. NOTHING RUNNING. PENDING PRASANN: larger-data push (15,616 rows, 11-task, 32k-64k); new approach to blocks 0-6; merge side branches. METRIC (Prasann 11-02): rank by the 8k rung (4-task mean at 8k); 2k is secondary (shorter than training rows, near ceiling, it drove most of the 8-cell decay). Recipe 8k-only FTM 4.6x / 3.6x / >=4.5x (2-seed mean at 7,808 ~3.4x).
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
