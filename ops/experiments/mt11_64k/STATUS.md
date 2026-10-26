---
experiment: mt11_64k
title: Long-context, large-data 11-task CTC mix (rows to 64k, ~64M tokens/task) -- dense vs the mid-stack online token router at matched FLOPs
status: active
updated: 2026-10-25
branch: prasann/mt11-64k (pushed; based on prasann/flopdrop_clean 177c76b05; worktree .claude/worktrees/mt11-64k)
launcher: research/token_routing/mt11_64k/launch_mt11_64k.py (state.json beside it)
records:
  - records/mt11-64k-online-router.md
results:
artifacts:
  - https://claude.ai/artifact/5LcLKC8bT6MDfz6JcLLQ9n
wandb_group: https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/m64-q35-4b
next_step: Universal recipe (lora64r20to7f16) f4 done: +0.038 at 0.365x FLOPs, 9/9 tasks, -24% peak memory (single seed). Pending: its grouping eval relaunch (01M5V9STM3FTH5TS1EKBH7F5NR) -> collect_m64.py --record, send the grouping number to local_a8068d93, republish the page. Possible next: a second seed of the recipe at f4; the recipe at f2/f1.
---

# mt11_64k

## Question
Prasann (2026-10-22): "for the 11-task mix try to go up to like 64k and have a larger amount of data (maybe
similar data to the individual task experiments from before for each task), since it seems like these methods
maybe scale better with more flops and longer contexts?" and "64k context length". On the 8k fast8k mixes the
mid-stack online router matched dense at equal FLOPs on 4 tasks but failed strmatch (-0.12 to -0.25) and grouping
(-0.05 to -0.07) on 11 tasks with 480-720 rows/task. Does it hold up with ~10x the data per task and rows up to 64k?

## Design
- Data `mt11L64` (weka `mt11_64k/shards/mt11L64`, 3.1 GB): setA seed-42 gold shards
  (`ctc_sft_sets/setA_max20_evaliid/shards_qwen3_5_gold`, IID with the CTC suite, query position both, full gold
  sidecars), rows <= 65,536 tokens, ds64's short-heavy token shares 30/20/15/13/12/10 over length bins
  <=3k/6k/12k/24k/48k/64k of a 64M-token target per task (`merge_tasks.py --length-mix`). Realised: 156,587 rows,
  661.7M tokens; 12.3k-15.8k rows and 49-64M tokens per task. Top bin (48-64k) is ~100 rows/task for nq, hotpotqa,
  qdmatch_nq, contradiction, xabsence, grouping; 12 for reorder; none for outlier/oolong/rerank (their 64k-bucket
  rows are all > 65,536 tokens) or strmatch (max 20k).
- Geometry, every arm: 8 GPUs (jupiter or ceres, any-of), unpacked, seq 65,536, 128 rows/step, micro 1, flash_2,
  `q35-4b-base-markerfix`, FLOP x-axis `flop_meter/actual_pflops`. Dense = `densepf` (soft-token path, every
  document kept whole = pad-free dense; on mt4 fast8k it scored within noise of padded dense).
- Budgets: f1 = 1 epoch = 1,223 steps; f2/f4/f8 = 612/306/153 steps via `--max-tokens` (complete LR schedule).
- Arms: dense f8/f4/f2/f1 x LR {2e-6, 5e-6} (reference = per-cell better LR); online router `otr10M7`, `otr10M8`
  (launch_fast8k flags + 512 target samples/row) at f1 LR 2e-6, plus otr10M7 f1 LR 5e-6 and otr10M7 f2.
- Memory plan for the router's full-context step every 8: micro 1 means it is one <= 65,536-token row per
  micro-batch, the same as a dense micro-batch. Runs over 300 steps save a resume point every 300 steps (keep 1).
- Eval: CTC-suite rows at every rung the subset has (2k, 8k, 16k, 32k, 64k; reorder to 16k; xabsence/strmatch/
  grouping to 32k), 500 rows each, query position both, ctc-spec grading, native docchunk ladder on Beaker
  (`run_q4b_beaker_multirung_eval.py --suite-rows --suite-rungs`). rerank is trained but not scored (the ladder
  has no ctc grader for its suite rows). Long rows staged add-only at `_eval_bundle/ctc_suite_rows`
  (`MANIFEST_long.json`).

## Decisions
- 2026-10-23 (agent): no new data generation: setA s42 already holds 2k-256k rows with sidecars for all 11 tasks
  (inspect job 01M5KPT3XQPBJVZR1CZ7CC2AAT), and it is the set the CTC suite was calibrated against.
- 2026-10-23 (agent): seq 65,536 (ds64-proven memory), not ~72k to admit the full 64k bucket; documented above.
- 2026-10-23 (agent): dense uses the pad-free soft path; the padded `full` variant would run 128 x 65,536 tokens
  per step for ~540k real ones.
- 2026-10-23 (Prasann, via parent session): "Feel free to validate ideas at smaller scale first, I want results and iteration to come in quickly and bigger runs should be saved for the most promising stuff". -> cancelled dense f4/f2/f1 and router f1/f2 (all still pending); first comparison = otr10M7 at f4 vs dense f16/f8 at both LRs.
- 2026-10-23 (agent): seed replicates and f2 runs moved to 4 GPUs after 4.5 h in the whole-node queue (99 urgent
  whole-node jobs ahead; 1-4 GPU slots schedule in ~2 min). Same rows/step and micro-batch -> same scores and FLOPs;
  wall-clock compared only within a GPU count. Eval minRuntime now 2h (grouping 4h) after a 1h grouping eval was
  preempted at 2.5h.
- 2026-10-23 (agent): scale-up at LR 5e-6 only for f2/f1 dense (5e-6 beat 2e-6 at every measured budget f16/f8/f4 on
  most cells, so 2e-6 at the big budgets would not set the reference); both LRs exist at f16/f8/f4 as required.
- 2026-10-23 (agent): minRuntime now covers each whole run (x1.3 the expected time) + a resume point every 50 steps:
  the urgent 2-3 h runs were preempted right after their 1 h minRuntime and restarted from scratch.
- 2026-10-23 (agent): ceres added beside jupiter: whole 8-GPU nodes on jupiter had 67 urgent jobs ahead.

## Runs
| run | job | state |
|---|---|---|
| inspect setA shards | 01M5KPT3XQPBJVZR1CZ7CC2AAT | done |
| build mt11L64 | 01M5KQ85DJ4H4GD8SYCQ8HBD7C | done (merge ok; in-job verify hit an import error) |
| verify mt11L64 | 01M5KQJTP36TXNJT5VFZ19G99E | done: VERIFY PASSED (sidecar 100%, max 65,536) |
| stage suite r16k/r32k/r64k | 01M5KQGVTAZB700CBTANA5MHEK | done: 25/25 files hash-verified |
| validate long suite cells | 01M5KRA2QA3YNS4XGND5S02B28 | DONE: 25/25 cells pass on all 500 rows (prefix parity 500/500 at query position both, gold 1.000, junk 0); max prefill 70,035 (hotpotqa@64k) -> eval MAX_LENGTH 73,728 skips nothing; reports weka mt11_64k/reports/validate_long_{a,b,c}.json |
| m64-otr10M7-smk-lr2e-06 (smoke) | 01M5KQY681Z0T90XACXXDXFB8R | DONE: 10 steps incl. the full-context step 8 at micro 1, no OOM; ~22 s/step on 8 H100; 88.4 PF actual vs 169.9 dense-equivalent (0.52x incl. 2 dense steps of 10); keep_frac 0.106-0.108; leak_auc 0.46-0.52; fid_mass router 0.66 / oracle 0.87 / random 0.10 at step 9 |
| smoke eval nq 2k..64k | 01M5KTX195Y4976D3FAX1SVTS0 | running |
| m64-otr10M7-f1-lr{2e-06,5e-06}, m64-otr10M8-f1-lr2e-06, m64-otr10M7-f2-lr2e-06 | 01M5KTRS12WZEAGXZRN34XDR8X, 01M5KTSSVKDY09QX28M4A792YG, 01M5KTTZ45A9GGP8JKYS25GPYN, 01M5KTW2XQH82TVDMA1KW3KAF8 | CANCELLED while pending (small-first) |
| m64-dense-f16-lr{2e-06,5e-06} | 01M5KV1BZQXZN0Y24J0P0HSWBR, 01M5KV26E5X06Z0DMN8D9S936N | DONE: 1,175.2 PF, 77 steps, ~2,050 s wall; 10 evals each launched |
| m64-dense-f8-lr{2e-06,5e-06} | 01M5KQZ52Z5MQ3W6N86NKYVW4M, 01M5KR0BQXMHGHHKXQ6QWZGJJR | DONE: 2,287.3 PF, 153 steps, ~4,100 s wall (~20 s/step training); 10 evals each launched |
| m64-dense-f4-lr{2e-06,5e-06} (bracket the router f4 run above dense f8) | 01M5M319PV5CWBK53PMMT1ABBV, 01M5M322Z5QDRF76GSBEQ2WCRH: preempted at ~1h, cancelled; RELAUNCHED 01M5M8P5K3TW03SW2ENAEJX27Z, 01M5M8PY3ZYDNHS7V27WWB2QWQ (minRuntime 4h, resume every 50 steps) | pending |
| m64-otr10M7-f4-lr5e-06 (LR-matched router) | 01M5MMX3TY9H4786BRZ2BNY83P | DONE 1,872.4 PF; 10 evals launched (9 done by 19:40) |
| seed replicates -sd1, LR 5e-6: otr10M7-f4, dense-f8, dense-f4 | 8-GPU 01M5MZK4XJ4JC4Z695P6Q53MJB, 01M5MZN84KF4DX04FTHKJ50HS8, 01M5MZPX8VF998YXPCHVW32N1K queued 4.5 h, cancelled; 4-GPU 01M5NFRD7VMK8B9N7X743DV9Y3 (RUNNING), 01M5NFS7P24KP3KDCY93SJ8779, 01M5NFSZV56883960609SSJVM6 | 4 GPUs: same rows/step, scores and FLOPs unaffected, wall-clock not comparable to 8-GPU runs |
| scale-up LR 5e-6: otr10M7-f2, dense-f2 (4 GPUs: 01M5NFTSY0MJBQNXPNB9HDBERV, 01M5NFVJS7Y4D046KXFE9AFE81); dense-f1, otr10M7-f1 (8 GPUs: 01M5MZYWVPS7TXCFSGKNEQH0V6, 01M5N00AE88VBBGQRQ28F87N6S) | pending (minRuntime capped at Beaker's 8h; resume every 50 steps) |
| m64-otr10M7-f4-lr2e-06 | 01M5KV36S0MTH0VRZT2WK32QKB: preempted twice at ~1h (step ~100 / ~200), restarted from scratch (no resume point before step 300), cancelled; RELAUNCHED 01M5M8NDSDPCS9YNWBQED7NNJE (minRuntime 3h, resume every 50 steps): FAILED at step 2, NaN grad norm already at step 1 on node 01JRYCX2HZQ2QE60PMHBB7ZM8J (same batch gives CE 0.9751 / grad norm 26.82 on other nodes = bad node); RELAUNCHED 01M5MEB6K9S3H6MQCS9FPPQ4G4: DONE 14:50, 1,872.4 PF (0.405x dense-equivalent); 10 evals launched |
| m64-dense-f{8,4,2,1}-lr{2e-06,5e-06} (f4/f2/f1 CANCELLED pending, small-first) | 01M5KQZ52Z5MQ3W6N86NKYVW4M, 01M5KR0BQXMHGHHKXQ6QWZGJJR, 01M5KR1S0JRQ8MJ8C2SV7P3H4P, 01M5KR3V9W8Y669X20SJT0RT1P, 01M5KR60YGENAF5XM27F0FV9PM, 01M5KR6VCQNA4ZRWYNPHQAEQF5, 01M5KR7W5JAD0GYBXK9J1FKKBD, 01M5KR95YM906V56JZK1HVNK4Q | pending |

## Universal recipe f4 RESULT (2026-10-25): m64-lora64r20to7f16-f4-lr5e-06, 1,695.8 PF = 0.365x dense-equivalent,
interpolated between dense f16 and f8. vs better-LR dense (seed-averaged where replicated), mean over rungs:
nq +0.003, hotpotqa +0.022, qdmatch +0.016, outlier +0.063, oolong +0.023, contradiction +0.068 (8k->64k +0.041 ->
+0.119), xabsence +0.110, reorder(2k) +0.017, strmatch +0.020 (otr10M7 had -0.017) -> task-average +0.038, ahead 9/9.
grouping: first eval died at 16k from an NCCL allgather timeout (2k 0.762 / 8k 0.395, ~+0.006 by hand); relaunched
2k/8k/16k 01M5V9STM3FTH5TS1EKBH7F5NR. Peak memory (4 GPUs): 37.22 / 44.22 GiB active/reserved vs dense 48.67 / 52.78
(-24% / -16%). LoRA adapters confirmed loaded in eval ("[early-lora] adapters re-installed from config.json").
Single seed. Reported to the LR-sweep session (local_a8068d93) on 2026-10-25.

## Universal-recipe request (2026-10-25): router branch claude/online-router-1bc01a (tip 7c2eeebe8, >= 5e1202e4b and
984825862) merged into prasann/mt11-64k (31e9a124b); arm lora64r20to7f16 imported verbatim from launch_otr.py.

## FINAL (2026-10-25): router otr10M7 LR 5e-6 vs better-LR dense at matched FLOPs, task-average / tasks ahead:
f4 seed0 +0.032 9/10, f4 seed1 +0.025 8/10, f2 +0.031 8/10, f1 +0.028 8/10 (all interpolated, none extrapolated).
Steady ~+0.03 across 4x data, not growing. Same data: router -0.03 vs dense. Wall-clock (4 GPUs) 0.79-0.82x.
Dense f1 = 18,592 PF; router f1 = 7,439 PF. Full tables: records/mt11-64k-online-router.md.

## Wall-clock, like for like (both f2 runs on 4 H100s, 612 steps): dense 22,979 s (37.5 s/step incl. load) vs router
18,824 s (30.8 s/step) = 0.82x, at 0.40x the FLOPs. On 8 GPUs the two ran at about the same per-step time (~20 s); with
fewer GPUs per step the compute share grows and part of the FLOP saving shows up as time. Dense f2 = 9,294 PF.
(Dense f2's outlier 16k-64k are new experiment bests, 0.745/0.423/0.323: dense keeps improving with data; explained.)

## f2 read (2026-10-24, partial: grouping eval pending): router otr10M7-f2 LR 5e-6, 3,725 PF (0.40x), between dense f8 and
f4. vs better-LR dense (seed-averaged where replicated): nq -0.003, hotpotqa +0.021, qdmatch -0.001, outlier +0.069, oolong
+0.011, contradiction +0.044, xabsence +0.124, reorder(2k) +0.009, strmatch +0.001 -> avg +0.031, the advantage holds at
2x the data and strmatch no longer loses. --check flags explained: xabsence 2k = 1.000 is saturation (every arm
0.999-1.000), xabsence 8k-32k are new experiment bests (most-trained run), reorder 16k 0.000 = the known parse failure.

## Seed replicate (2026-10-24 ~05:00): router f4 LR 5e-6 seed 1 agrees with seed 0. vs seed-averaged better-LR dense:
seed0/seed1 xabsence +0.108/+0.118, outlier +0.071/+0.035, contradiction +0.050/+0.033, hotpotqa +0.018/+0.024,
oolong +0.021/+0.021, qdmatch +0.011/+0.014, nq +0.002/+0.007, reorder(2k) +0.010/-0.009, strmatch -0.017/-0.016;
grouping seed 1 pending. Ahead 8/9 with both seeds, mean +0.028.

## LR-matched read FINAL for the single seed (2026-10-23 ~23:30): all 10 evals in. Task-average +0.034, ahead on 9/10:
nq +0.005, hotpotqa +0.017, qdmatch_nq +0.022, outlier +0.051, oolong +0.023, contradiction +0.056 (64k +0.105),
xabsence +0.125, reorder (2k) +0.031, strmatch -0.018, grouping +0.027. Recorded in ops results.

## LR-matched read (2026-10-23 ~19:30; router otr10M7-f4 at LR 5e-6, 1,872 PF = 0.40x; single seed; eval_size 500/cell; no-cot)
vs the better dense LR per cell, interpolated in log PF between f16 and f8; mean over each task's rungs (no-signal cells
reorder 8k/16k and grouping 32k excluded; grouping and reorder partial when read):
xabsence +0.125 (8k/16k/32k +0.14/+0.20/+0.15), outlier +0.051 (8k +0.11), contradiction +0.043 (16k/32k +0.07/+0.08),
oolong +0.023, qdmatch_nq +0.022, hotpotqa +0.017 (32k/64k +0.04), grouping +0.016, nq +0.005, strmatch -0.018 (was
-0.090 at LR 2e-6). Router ahead on 9/10 tasks; gains mostly grow with length. Single seed: noise ±0.02-0.03
(outlier up to ±0.13) -> seed replicates launched.

## First matched-FLOP read (2026-10-23 ~15:45; single seed; eval_size 500 per cell; no-cot; recorded in ops)
Router otr10M7, f4 data, LR 2e-6, 1,872 PF, vs dense interpolated in log PF between f16 (1,175 PF) and f8 (2,287 PF);
mean over each task's rungs (contradiction, grouping, reorder partial -- evals still running):
- vs the better dense LR per cell (mostly 5e-6): nq -0.008, hotpotqa -0.009, oolong -0.006, qdmatch_nq -0.014,
  xabsence -0.015, contradiction -0.021, grouping -0.027, outlier -0.051 (16k -0.10), strmatch -0.090 (every rung
  -0.07..-0.13). No cell above +0.011.
- vs dense at the SAME LR 2e-6: router ahead on 9/10 tasks (contradiction +0.085, xabsence +0.082, qdmatch +0.060,
  outlier +0.056, hotpotqa +0.032, oolong +0.021, grouping +0.020, nq +0.013, reorder +0.004); strmatch -0.041.
- Reading: at 2e-6 the router is FLOP-efficient, but dense at 5e-6 is better still; the router at 5e-6
  (m64-otr10M7-f4-lr5e-06, queued) decides it. strmatch is the one task the router loses either way.

## Early reads (single seed, 500 rows each, no-cot)
- nq f1 at 2k/8k/16k/32k/64k: 10-step smoke checkpoint (~base) 0.934 / 0.704 / 0.490 / 0.296 / 0.170; dense f8 LR 2e-6
  0.976 / 0.904 / 0.896 / 0.872 / 0.856. The 64k mix removes most of the length falloff.
- Cost: dense ~15 PF/step. Router otr10M7-f4 (306 steps) measured 1,872.4 PF actual vs 4,624.7 dense-equivalent
  = 0.405x (flops.json), i.e. between dense f16 (1,175 PF) and f8 (2,287 PF) -- bracketed. (The smoke's 0.52x
  included 2 dense steps in 10 and overstated it.) Router wall-clock per step ~20-25 s vs dense ~20 s at micro 1 on
  8 H100s: no wall-clock saving in this configuration.

## Current best numbers
| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|

## Next step
See `next_step`.

## Open issues / traps
- Beaker node 01JRYCX2HZQ2QE60PMHBB7ZM8J produced NaN gradients at step 1 (CE matched other nodes exactly). If a run
  NaNs at step 1-2, compare its step-1 `optim/total grad norm` with 26.82 before suspecting the code.
- reorder 8k/16k score 0.000 for dense f8 (parse_rate 0.072 / 0.0; 2k 0.172, parse 0.97). Checked the raw generations
  (gantry 01M5M51J4CQHJTJG8A3S699QMG, `show_gens.py`): 8k answers are well-formed lists with duplicate/missing ids, which
  the ctc reorder parser (exact permutation of 1..n required) scores as unparsed; 16k: 150/500 repetition loops
  ("1, 1, 1, ..."). A genuine model failure under a strict grader, identical for every arm -> reorder 8k/16k carry no
  signal at these budgets; report them as uninformative, not as zeros.
- grouping 32k scores 0.001-0.005 for every dense run (the rung also takes 1.4-1.8 h to decode): no signal at these
  budgets, excluded from the 5-rung means like reorder 8k/16k.
- Evals launched before 10:05 registered as experiment "unassigned" (eval launcher looked up the eval name, not the
  training run); results are recorded under mt11_64k regardless. Fixed: evals now pass --experiment mt11_64k.
- The docchunk FlexAttention eval OOMs near ~77k tokens on one H100 and SKIPS prompts over MAX_LENGTH; r64k
  prompts must be checked against the validation's prefill tail before trusting eval_size there.
- 38-80% of r64k prompts are 65.5k-70k tokens, slightly past the longest training row (65,536), for every arm.
- Outlier/oolong/rerank never see a row > 35k tokens in training; their r64k rung is length extrapolation for
  every arm alike.
- reorder and oolong rows have empty gold sidecars by construction (router allocation is per document, gold only
  feeds diagnostics).

## Verified vs unverified
- Verified 2026-10-23: shard rows/bins/sidecar (verify job above); long suite rows hash-match the HF revision.
- UNVERIFIED: everything about training and eval outcomes.
