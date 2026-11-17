---
title: GPU usage and agent-workflow audit, 2026-10-20 → 2026-11-10
date: 2026-11-10
updated: 2026-11-10
experiment: gpu_audit
area: pipeline
kind: audit
status: current
superseded_by:
artifact: https://claude.ai/artifact/AxWXjBS1jDQy51MtGszd3d
summary: 6,705 GPU-h in 3 weeks (Beaker 74%). The largest loss is preempted Beaker training restarting from step 0 (1,083 GPU-h, 16%): allocation balance preempts us right after the 1 h minRuntime, and the trainer saves no mid-run checkpoint. Beaker jobs spent 60% of their lifetime pending (jupiter-only waits ~4x longer than jupiter+ceres), while jsteinhardt ran at 23% of its 8-GPU cap. Lambda is ~85% busy with other users. Ranked fixes inside.
---
# GPU usage and agent-workflow audit, 2026-10-20 → 2026-11-10

Read-only audit. Nothing was launched, cancelled or reconfigured. Artifact (private):
https://claude.ai/artifact/AxWXjBS1jDQy51MtGszd3d. Scripts and small pulls: `research/common/gpu_audit/`
(its README lists every pull command). Ops page: `ops exp show gpu_audit`.

## Verdict

1. **Preempted Beaker training restarted from step 0: 1,083 GPU-h (16% of all GPU-h; 309 attempts).**
   Since about 10-19, Beaker preempts by *allocation balance*. Our workspace group `ai2/oe-scaling` is about 55% over
   target, so once a job passes minRuntime, other groups' jobs at any priority (even `low`) displace it. 304 of the
   309 attempts had minRuntime 1 h. The median attempt ran 1.26 h; 77% were preempted before 2 h and 91% before 3 h.
   `train_ctc_suite.py` writes no mid-run checkpoint unless `--save-interval` is passed (47 of about 430 logged
   Beaker training commands passed it). 92% of completed Beaker training runs take ≤2 h and 99% ≤4 h, so a
   minRuntime that covers the run avoids most of this.
2. **Beaker jobs spent 60% of their lifetime pending.** Jupiter-only vs jupiter+ceres waits, median / p90 in
   hours: 1 GPU 0.33/1.04 vs 0.05/0.56; 2 GPUs 0.40/2.97 vs 0.10/0.81. 3,759 of 5,563 attempts were pinned to
   jupiter. No job waited more than 7.2 h; the days-long tail of minRuntime-0 jobs is gone (98% set one).
3. **jsteinhardt ran at 23% of the 8-GPU `preemptive_high` cap** (mean 1.8 GPUs). 375 of 520 hours had none of our
   jobs there, and only 7 jobs waited more than 15 min. In the 168 hours when ≥4 of our GPUs were pending on
   Beaker, 691 GPU-h of jsteinhardt headroom went unused (3,215 GPU-h over the window).
4. **Lambda is not idle.** Other users held 20.5 of the 24 working GPUs on average (hyperplane03 down, 04 drained).
   29 of their jobs ended within 3 min of one of our preemptive starts on the same node. Free GPUs without
   preempting: about 2.4 on average, 2–3.5 at every hour of the day, so there is no off-peak window to use.
5. **98% of our Beaker GPU-h was *allocated*** (4,871 of 4,991 GPU-h; before 10-19, 97% unallocated), because
   `ai2/flex2` now belongs to the `ai2/oe-scaling` group. The CLAUDE.md "always unallocated" rule no longer matches
   reality. This is Prasann's decision.

## Data sources: what each covers and misses

| source | covers | misses |
|---|---|---|
| `ops_state/runs.jsonl` (registry) | 9,138 launches in window, with experiment, kind, expected minutes and parent run | 72% of runs never got a final state; 1,175 GPU-h under names with no STATUS page |
| `sacct -u prasann -X -D` Berkeley + Lambda | every attempt with submit/eligible/start/end, state, QOS, node, GPUs. Retention back to 2026-07-21 | in-job utilization |
| `sacct -a` Berkeley (jsteinhardt, berkeleynlp) + Lambda | other users' load for headroom and preemption attribution | node up/down history (Lambda assumed 24 GPUs) |
| `beaker experiment get` (`fetch_beaker.py`) | 6,425 experiments / 7,553 job attempts, with created/started/exited, minRuntime, cluster constraint, preemption reason | experiments in neither the registry nor the ops cache. Beaker's report agrees with our total to within 1% (4,954 vs 4,990 GPU-h) |
| `beaker report gpu-usage --organizations ai2 --users prasanns` | GPU-h by cluster, allocated, workspace group, day | per job |
| STATUS pages + records (subagent, `data/workflow_incidents.csv`) | 77 dated incidents with stated costs | most incidents state no GPU-h |
| Claude transcripts on this machine (`sessions.py`) | 40 sessions (34 OLMo-core, 6 PAIL); idle gaps labeled by what ended them | sessions run from other machines |
| in-job GPU utilization | **not available**: no stored `ops stall` samples, wandb system metrics not pulled | "allocated but idle" is unmeasured |

## Answers

**1. GPU-hours.** Total 6,705: Beaker 4,954 (jupiter 2,971, jupiter+ceres 1,864, titan 119), jsteinhardt 1,190
(`preemptive_high` 945, `preemptive` 245), Lambda 560, berkeleynlp 1. By experiment: tokdrop_lr 2,035,
olmo3_cptdrop 1,585, mt11_64k 1,079, learned_router 616, PAIL (all `pail_*`, all on Beaker) 1,192. By kind:
train 4,847, eval 1,657, smoke 77, probe/bench 52, data 37. Beaker usage peaked at 99 running GPUs (mean 9.5
running, 16.0 pending). Beaker's pending queue peaks 14:00–18:00 PT. jsteinhardt sat at its cap for 92 h.

**2. Queue wait.** Pending share of job lifetime: Beaker 60%, Lambda 57%, jsteinhardt 17%. Median / p90 wait:
jupiter 0.39/2.35 h, jupiter+ceres 0.06/0.60 h, jsteinhardt 0.00/0.14 h, Lambda 0.10/0.78 h. On jupiter, 1-GPU
jobs wait less than 2-GPU jobs (p90 1.04 vs 2.97 h). minRuntime 2–3 h did not wait longer than 1 h (median
0.15–0.28 h vs 0.28 h). The worst waits were a burst of 2-GPU eval waves pinned to jupiter on 10-22/10-23
(4.9–7.2 h), an 8-GPU olmo3_cptdrop run (6.1 h), and a berkeleynlp debug job at QOS normal (35 h).

**3. Waste** (GPU-h over the window; per week ÷3.1):

| cause | GPU-h |
|---|---|
| Beaker training preempted, restarted | 1,083 |
| preempted eval / smoke / probe attempts | 100 |
| slurm / Lambda training preempted | 47 |
| failed attempts (457, 171 of them training) | 233 |
| cancelled while running (includes deliberate kills) | 234 |
| completed evals with no result row after 24 h (597 jobs) | 173 |
| same training run on two pools (33 runs, smaller copy) | 53 |
| unregistered jobs (94 attempts) | 36 |
| smokes, all (154; 71 failed or cancelled) | 77 |
| unused jsteinhardt cap during Beaker backlog *(capacity, not held)* | 691 |

Expected vs actual minutes: the median ratio is 0.21 (p90 0.95). Only 30 jobs ran more than 2× their estimate,
while 2,247 finished in under a quarter of it. The estimates run about 5× long.

**4. Latency.** Training end → first eval launch: median 7 min, 62% within 10 min, 8% over 12 h. Eval end →
result recorded: median 0.3 h, p90 5.6 h, 0.6% over 24 h. The worst case was 144 h, for tokdrop per-row evals
registered under `ds64_fast8k`. Sessions: of 3,110 open session-hours, 225 were active, 1,181 idle until a
job or subagent notification, 671 idle until another session's message or state change, and 873 idle until Prasann wrote
(33 waits longer than 1 h). That last figure is an upper bound, since it includes nights. Items held for Prasann
(subagent, from STATUS pages): the tokdrop larger-data push, blocks 0-6 and branch merges; the gate-gradient
sweep (since 11-03); olmo3_cptdrop's matched SFT (since about 10-27); weka D/D2 and the ffndrop rebuild (since
10-30); mt11_64k's next step (11-04).

**5. Process gaps.** Unregistered: 0.5% of GPU-h (rare). Experiment names with no STATUS page: 17.5% of GPU-h.
Registered runs with no final state: 4,836 of 6,916. Contention for the jsteinhardt cap: 1 of 520 hours (today's
three side sessions). Other users' load is invisible in `ops status`.

## Recommendations (ranked)

| # | change | type | evidence (above) | saves / week | cost |
|---|---|---|---|---|---|
| 1 | minRuntime = min(4 h, 1.3 × expected) on every Beaker training launch; above 4 h, add `--save-interval` with an overwriting checkpoint | launcher default | verdict 1 | ~300 GPU-h plus one requeue and rerun per avoided preemption | small. Watch for a week whether longer minRuntimes queue longer |
| 2 | constrain Beaker jobs to `ai2/jupiter` + `ai2/ceres` by default | launcher default | verdict 2 | ~650 job-hours of waiting; 2-GPU p90 ~3 h → ~0.8 h | small, after a ceres 2-GPU smoke (image, weka, memory) |
| 3 | decide allocated vs unallocated, then update CLAUDE.md, memory and the check | Prasann decides | verdict 5 | unknown | a decision, plus an infra request if moving back |
| 4 | spill ≤2-GPU, ≤2 h jobs to jsteinhardt when the Beaker queue is >30 min and the cap has room | routing rule (where-to-run) | verdict 3 | ~220 GPU-h of throughput during backlogs | medium. Same-job slurm launchers for evals and probes. Preempts lower-QOS users (allowed; Rhys only up to 8 GPUs) |
| 5 | one eval job per checkpoint looping over its tasks (2,665 per-task jobs covered 862 checkpoints; median eval runs 6.6 min) | tooling (launch_fast8k eval) | waits, eval counts | ~580 fewer queue entries; load overhead unmeasured (~60 GPU-h at 3 min × 2 GPUs) | small to medium |
| 6 | `ops status` shows other users' GPUs per node, and launch-run warns before a start would preempt someone | tooling | verdict 4 | other users' GPU-h; avoids the "Lambda idle" mistake | small |
| 7 | `ops orphans` (done >24 h, no result row), run in job-status and close-out; `ops register` refuses an `--experiment` with no STATUS page | tooling + hook | answers 3–5 | ~55 GPU-h of unread evals, and multi-day collection delays | small |
| 8 | expected_minutes defaults to the history median per launcher and arm prefix | tooling | answer 3 | makes #1 automatic; better OVERRUN flags | small |
| 9 | no cross-session GPU lock; keep insurance copies only for runs that gate a decision; no probe packing (smokes plus probes = 2%) | routing rule | answers 3, 5 | — | none |

## What the starting examples turned out to be

- 7 urgent jobs pending >1 h: **common** (27% of jupiter-only attempts waited >1 h). PAIL's 1,192 GPU-h
  shares the same workspace and queue.
- Three sessions contending for the jsteinhardt cap: **rare** (1 hour, today).
- Lambda read as idle, our start preempted another user: **real and recurring** (29 jobs), though no record
  mentions it.
- Preempted after minRuntime, restarted from step 0: **the largest waste**. Confirmed in mt11_64k (10-23),
  tokdrop decay (10-30), cross-row (11-02), oracle (11-04) and olmo3_cptdrop.
- 6-day uncollected evals under the wrong name: confirmed once. The structural cause is the 17.5% of GPU-h
  registered under names with no STATUS page.
- Insurance copies: small (53 GPU-h) and useful when jupiter preempted the original.
- UNREGISTERED jobs: small (0.5%).

## Limits

In-job utilization is unmeasured. Restart-from-scratch is inferred from the trainer default plus the incident
log. Duplicate detection matches names, so renamed copies are missed. "Training runs with no linked eval"
(827 runs) is mostly a registry-linking gap, because PAIL sweeps and tokdrop in-training metrics keep their
numbers elsewhere, so it is not reported as waste. Committed data pseudonymizes other cluster users (the
mapping is in gitignored `raw/user_map.json`).
