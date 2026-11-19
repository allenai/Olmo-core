---
title: Dynamic token dropping against the dense frontier, budget-matched (auto-research loop, 11-18 rules)
date: 2026-11-19
updated: 2026-11-19
experiment: tokdrop_lr
area: token_routing
kind: result
status: complete for this phase (11-19): token dropping with LoRA and a 50% / 20% / 7% keep is worth 2.1-2.4x in FLOPs on two seeds from 495 to 1,980 PF and 1.9x at 3,973 PF; nothing tried pushes it further at large budgets
superseded_by: 
artifact: https://claude.ai/artifact/SPSBxHFs2PUn5CPjJX3Wgt (hub, section "Auto-research loop"; private)
summary: Budget-matched FLOPs-to-match of dynamic token dropping against the dense frontier (dense full fine-tune and dense LoRA at every budget) on a 4-task 4k-16k pool, Qwen3.5-4B; about 2x that holds from 495 to 3,975 PF; schedules, per-task budgets, document drop, backward skipping, frozen early blocks, gradient compensation and cheaper keep shares give nothing durable.
---

# Dynamic token dropping against the dense frontier, budget-matched

Loop session of 2026-11-18 / 19 (branch `claude/early-loss-proxy-d6e886`, folder `research/token_routing/early_loss_proxy/`). Rules set by Prasann on 11-17 / 11-18:
dense FULL fine-tuning is the reference; LoRA and fixed data re-mixes count as the dense baseline, so only dynamic FLOP dropping during training is a method; a method counts only
at the deepest loss dense reaches (extra steps are fine); cheaper arms get proportionally more steps (FLOP-matched budgets).

## Verdict

- **About 2x, steady.** Whole-model LoRA + online router keeping 50% / 20% / 7% of document tokens after blocks 7 / 16 / 24 (0.29x dense per step) needs 2.1-2.4x fewer FLOPs than the
  dense frontier at 495, 991 and 1,980 PF (two seeds each) and 1.9x at 3,973 PF (one seed). Against dense full fine-tuning alone: 2.7-2.9x. The gain does not shrink with data.
- **The old recipe (20% / 7%) fades with data**: 1.8-2.2x up to 1,980 PF, 1.2x at 3,957 PF. Per task it is the best arm on outlier and the worst on reorder; reorder is 60-70% of the mean loss.
- **Nothing durable below about 0.3x per step.** Smaller keep shares, an earlier first cut, more stages, keep schedules, gradient compensation, frozen early blocks, mixture-of-depths,
  sparse backward and backward skipping all lose per FLOP or are level. Document drop on 16k rows and per-task budgets add 0.005-0.008 at <= 991 PF and nothing at about 2,000 PF.
- **Far from the 10-20x target.** The cost floor is blocks 0-6, which must run on (and be trained through) every token.
- **Diagnosis** (sparse-backward arm): with the exact dense forward and gradients only through the kept tokens the loss gap to dense is the same as with forward dropping, so the gap
  comes from the missing gradient of dropped tokens, not from the truncated context.

## Setup

- Qwen3.5-4B (`q35-4b-base-markerfix`), 16 rows per step, sequence 16,384, trainer `recipes/train/research_trainer/train_ctc_suite.py` through `dense_probe.py`.
- Pools (see `POOL_V2.md`): `mtb16k` (7,808 rows; contradiction, outlier, reorder, qdmatch_nq at 4k / 8k / 16k), `mtb16kx4` (31,232 rows), `mtb16kx16` (124,928 rows); each larger pool contains the smaller.
- Metric: dense-forward teacher-forced answer CE on held-out rows, mean over the 16k sets of `heldout_v3.npz` (gen16k + rung16k). **eval_size 320 (80 per task), below 500**; one or two seeds;
  same-seed reruns differ by up to 0.007, so differences under about 0.007 are noise.
- Budget-matched runs: every arm, dense included, gets a fixed FLOP budget with a complete LR schedule over its own steps (`--max-tokens`). FLOPs from the flop meter (PF = petaFLOPs).
- FLOPs-to-match (`ftm.py`): PF the dense frontier needs for the arm's final loss (log-linear interpolation between frontier points, seeds averaged) divided by the arm's PF.

## Dense frontier

| dense run | pool | steps | total PF | final 16k CE | seeds |
|---|---|---|---|---|---|
| bud-dense-b495-sd0+1 | mtb16kx4 | 122 | 491 | 0.191 | 2 |
| bud-denselora-b991-sd0+1 | mtb16kx4 | 323 | 987 | 0.143 | 2 |
| v3-denselora-sd0+1 | mtb16k | 488 | 1,496 | 0.120 | 2 |
| x4-denselora-x4-sd0+1 | mtb16kx4 | 642 | 1,965 | 0.103 | 2 |
| x4-dense-x4-sd0 | mtb16kx4 | 976 | 3,960 | 0.081 | 1 |
| x16-denselora-b3964-sd0 | mtb16kx16 | 1,293 | 3,986 | 0.078 | 1 |
| x4-denselora-x4full-sd0 | mtb16kx4 | 1,952 | 5,990 | 0.068 | 1 |
| x4-dense-x4full-sd0 | mtb16kx4 | 1,952 | 7,934 | 0.064 | 1 |

## Token-drop arms (final 16k CE, eval_size 320)

| arm | pool | steps | total PF | final 16k CE | seeds | FLOPs-to-match vs dense frontier |
|---|---|---|---|---|---|---|
| LoRA + keep 50/20/7 (blocks 7/16/24) | mtb16kx4 | 420 | 494 | 0.134 (0.136 / 0.133) | 2 | 2.34x |
| LoRA + keep 50/20/7 | mtb16kx4 | 841 | 988 | 0.100 (0.100 / 0.100) | 2 | 2.21x |
| LoRA + keep 50/20/7 | mtb16kx4 | 1,683 | 1,980 | 0.074 (0.072 / 0.076) | 2 | 2.38x |
| LoRA + keep 50/20/7 | mtb16kx16 | 3,364 | 3,973 | 0.065 | 1 | 1.88x |
| LoRA + keep 50/20 (blocks 7/16) | mtb16kx4 | 376 | 495 | 0.140 | 1 | 2.09x |
| LoRA + keep 50/20 | mtb16kx4 | 752 | 988 | 0.097 | 1 | 2.37x |
| LoRA + keep 50/20 | mtb16kx4 | 1,525 | 2,006 | 0.078 | 1 | 1.99x |
| LoRA + keep 50/20 | mtb16kx16 | 3,008 | 3,975 | 0.064 | 1 | 1.97x |
| LoRA + keep 20/7 (old recipe) | mtb16kx4 | 547 | 495 | 0.146 | 1 | 1.90x |
| LoRA + keep 20/7 | mtb16kx4 | 1,094 | 986 | 0.101 | 1 | 2.14x |
| LoRA + keep 20/7 | mtb16kx4 | 1,952 | 1,762 | 0.083 | 1 | 2.16x |
| LoRA + keep 20/7 | mtb16kx16 | 2,188 | 1,980 | 0.082 | 1 | 1.92x |
| LoRA + keep 20/7 | mtb16kx16 | 4,377 | 3,957 | 0.074 | 1 | 1.22x |
| LoRA + keep 10/3 | mtb16kx4 | 1,304 | 1,006 | 0.117 | 1 | 1.55x |
| LoRA + keep 10/3 | mtb16kx16 | 2,608 | 2,015 | 0.100 | 1 | 1.06x |
| full FT + keep 50/20 | mtb16kx4 | 557 | 993 | 0.113 | 1 | 1.69x |
| full FT + keep 50/20 | mtb16kx4 | 1,108 | 1,970 | 0.082 | 1 | 1.95x |
| full FT + keep ramp 20->90% (block 7) | mtb16kx4 | 762 | 1,978 | 0.088 | 1 | 1.60x |
| LoRA + keep 50% after block 16 | mtb16kx4 | 905 | 1,973 | 0.088 | 1 | 1.62x |

## Add-ons on top of the 50 / 20 / 7 token drop

| arm | pool | steps | total PF | final 16k CE | seeds | FLOPs-to-match vs dense frontier |
|---|---|---|---|---|---|---|
| 50/20/7 + keep 25% of distractor docs on 16k rows | mtb16kx4 | 554 | 463 | 0.130 (0.128 / 0.132) | 2 | 2.67x |
| same | mtb16kx4 | 1,183 | 986 | 0.093 (0.091 / 0.095) | 2 | 2.77x |
| same | mtb16kx16 | 2,368 | 1,971 | 0.073 | 1 | 2.46x |
| 50/20/7 + keep 10% of distractor docs | mtb16kx4 | 640 | 494 | 0.121 | 1 | 2.93x |
| same | mtb16kx4 | 1,284 | 986 | 0.092 | 1 | 2.85x |
| dense LoRA + keep 25% of distractor docs (control) | mtb16kx4 | 462 | 986 | 0.137 | 1 | 1.11x |
| per-task: reorder 50/20/7, others 20/7 | mtb16kx4 | 1,013 | 986 | 0.095 (0.093 / 0.097) | 2 | 2.59x |
| per-task, online rule (sqrt grad norm) | mtb16kx4 | 1,013 | 993 | 0.095 | 1 | 2.60x |
| per-task: reorder 50/20/7, others 20/7 | mtb16kx16 | 2,032 | 1,984 | 0.077 | 1 | 2.11x |
| 50/20/7 + backward share 100% -> 30% | mtb16kx4 | 1,015 | 992 | 0.097 | 1 | 2.40x |
| 50/20/7 + backward on harder half | mtb16kx4 | 1,114 | 999 | 0.108 | 1 | 1.80x |
| 50/20/7 + backward share 100% -> 30% | mtb16kx16 | 2,027 | 2,025 | 0.080 | 1 | 1.96x |
| 50/20/7, blocks 0-6 forward-only | mtb16kx4 | 604 | 540 | 0.153 | 1 | 1.59x |
| same | mtb16kx4 | 1,209 | 1,079 | 0.105 | 1 | 1.76x |

Other closures (one epoch on `mtb16k`, 16k CE at step 488; dense full fine-tune 0.114 / 0.119, dense LoRA 0.121 / 0.118): keep ramp 20% -> 90% 0.113 / 0.119 / 0.120 on three runs
(the first reading of "reaches dense at 1.6x" was noise); hold-then-ramp 0.120 at 0.46x; gradient compensation 0.151-0.160 at 20/7 (0.138 without); random token choice 0.132 against the
router's 0.119-0.126; mixture-of-depths 0.134-0.137; adaptive keep by predicted gradient mass keeps most tokens (0.79-0.97x cost); first cut at block 3 0.132-0.152; sparse backward
0.136 at 0.54x (forward drop 0.138 at 0.30x).

## Limits

- 16k eval_size 320 and one or two seeds: ratios move by 10-20% with a 0.007 change in loss on the flat part of the dense curve.
- Runs at 1,980 PF and above on the 16x pool used 4 GPUs (the 4x-pool runs 2 GPUs); 12 contradiction and 12 outlier gen16k held-out rows have a gold item appearing as training filler on the 16x pool.
- qdmatch_nq sits on a plateau (0.05-0.06) for every arm past about 2,000 PF; per-task ratios there are not meaningful (`ftm_task.py`).
- Generation evals (branch `loop/validate2` @ 69ec057f4; one seed; F1, no-cot, **eval_size 500 per cell**, parse rate 1.0; reorder excluded, grader still fails): token drop 50% / 20% / 7% (LoRA), 1,683 steps, 1,980 PF: 16k F1 contradiction 0.976, outlier 0.965, qdmatch_nq 0.885; 3-task mean 4k / 8k / 16k 0.992 / 0.971 / 0.942 (SE 0.006 at 16k); dense LoRA, 642 steps, 1,963 PF: 16k F1 contradiction 0.949, outlier 0.726, qdmatch_nq 0.878; 3-task mean 4k / 8k / 16k 0.984 / 0.944 / 0.851 (SE 0.009 at 16k); dense full fine-tune, 976 steps, 3,960 PF: 16k F1 contradiction 0.970, outlier 0.929, qdmatch_nq 0.889; 3-task mean 4k / 8k / 16k 0.993 / 0.977 / 0.929 (SE 0.007 at 16k); dense LoRA, 1,293 steps, 3,986 PF: 16k F1 contradiction 0.972, outlier 0.954, qdmatch_nq 0.887; 3-task mean 4k / 8k / 16k 0.995 / 0.976 / 0.938 (SE 0.006 at 16k). The token-drop arm beats dense LoRA at equal FLOPs and is level with dense at about twice the FLOPs at 16k, so the 2x holds on the task metric.
- 64k check (tag `klb64`, pool `kl64k`, held-out `heldout_kl.npz` set kl64k: **eval_size 144**, 3 tasks; drop arms from the keep-versus-length session): dense full fine-tune 0.385 at 1,457 PF and 0.211 at 7,809 PF;
  dense LoRA 0.325 at 1,475 PF, 0.236 at 2,925 PF, 0.199 at 5,723 PF; keep 20% / 7% 0.238 / 0.265 (two seeds) at 1,463 PF = 2.0x / 1.6x against dense LoRA, 4.1x / 3.2x against dense full fine-tune alone.
  The gain does not grow with context length against the LoRA baseline.

## Where things are

- Scorers: `research/token_routing/early_loss_proxy/{ftm.py, ftm_task.py, ladder.py, eqflop.py}`; outputs `hub_tab/{ftm, ftm_task, ladder, eqflop}.json`; loop state `hub_tab/loop_state.json`; handoff `LOOP_HANDOFF.md`.
- Raw results: weka / S3 `prasanns/early_loss_proxy/results/<tag>/` (tags v3, w1-w3, x4, bud, x16, ks, gc, sbw, ds, tb, sb), fetched with `fetch_beaker_results.sh <tag>`; every final loss recorded with `ops result add` (task `heldout_v3_16k`).
- Code branches (none merged): `loop/keepsched` (`--st-otr-stage-sched`), `loop/sparsebwd` (`--st-otr-sparse-bwd`), `loop/gradcomp` (`--st-otr-grad-comp`), `loop/docsub` (`--longpos-docsub-keep`, `-sched`, `-min-len`),
  `loop/taskbudget` (`--st-otr-task-stages`, `--st-otr-task-adapt`), `loop/selbp` (`--selbp-keep`, `-mode`, `-sched`). Router fix for frozen early blocks with multi-stage cuts: commit 81d0924be on the loop branch.
