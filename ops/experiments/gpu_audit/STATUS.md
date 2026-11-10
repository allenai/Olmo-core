---
experiment: gpu_audit
title: INFRA — GPU usage + agent-workflow audit (Beaker / jsteinhardt / Lambda), 2026-10-20 → 11-10, with ranked fixes
status: active
updated: 2026-11-10
branch: claude/gpu-workflow-audit-8833c4
launcher: research/common/gpu_audit/analyze.py
records:
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/records/gpu-usage-workflow-audit-2026-11.md
results:
  - research/common/gpu_audit/data/summary.json
  - research/common/gpu_audit/data/jobs.csv
  - research/common/gpu_audit/data/workflow_incidents.csv
artifacts:
  - https://claude.ai/artifact/AxWXjBS1jDQy51MtGszd3d
wandb_group:
next_step: Implemented on branch claude/gpu-workflow-audit-8833c4 (not merged): minRuntime auto + resume point, spill rule in where-to-run, `ops capacity`, allocation wording. #2 (jupiter+ceres default) PENDING Prasann ("ask me again later"). After merge, optionally seed runtime history (backfill_runtimes.py --write); re-run the audit scripts ~2026-12-01 to measure.
---

# gpu_audit

## Question
Not an experiment: how well were the GPUs used across Beaker, Berkeley slurm and Lambda over 2026-10-20 → 11-10,
where did time and GPU-hours go (queue waits, preemption, failures, uncollected results), and which tooling,
routing or policy changes would raise throughput.

## Decisions
- 2026-11-10 (agent): read-only audit. Scripts and small pulls live in `research/common/gpu_audit/` (branch
  `claude/gpu-workflow-audit-8833c4`); raw sacct/Beaker dumps are gitignored. Other cluster users are
  pseudonymized in committed data (the repo remote is public).
- 2026-11-10 (agent): the audit itself changed no rule, hook, launcher, skill or CLAUDE.md.
- 2026-11-10 (Prasann, via the main session): #1 accurate minRuntimes: YES. #4 spill small jobs to jsteinhardt
  "or wherever they run": YES ("always a good idea"). #6 other users' load in ops status: YES. #3 allocated:
  "Always allocated is good, this is the updated rule" (unallocated rule retired; CLAUDE.md + memory updated in
  the main checkout by the main session). **#2 jupiter + ceres by default: PENDING — "ask me about this again
  later"; nothing changed.**
- 2026-11-10 (agent, branch claude/gpu-workflow-audit-8833c4, not merged): `opslib/runtime.py` (min(4h, 1.3 x
  expected), 1h fallback; history median of `runtime_minutes`, now written by `ops status --record`);
  `train_ctc_suite.py --resume-interval` (rolling resume point; fixes `--save-interval` being a no-op under
  `--no-final-checkpoint`); `beaker_ctc_suite.py --min-runtime auto` + auto resume past 4h; eval launcher
  `BEAKER_MIN_RUNTIME=auto`; `where-to-run` spill rule + our cap use by experiment; `ops capacity` (+ `--would-preempt`);
  allocation wording in beaker.md / skills / ops README / recipes docs.

## Current best numbers
Infra audit (no eval metrics):

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| all pools | GPU-h 10-20 → 11-10 | 6,705 (Beaker 4,954 / jsteinhardt 1,190 / Lambda 560) | — | — | — | data/summary.json |
| Beaker | training preempted + restarted | 1,083 GPU-h (16%), 309 attempts, 304 at minRuntime 1 h | — | — | — | data/jobs.csv |
| Beaker | pending share of job lifetime | 60%; jupiter-only 2-GPU wait median/p90 0.40/2.97 h vs jupiter+ceres 0.10/0.81 h | — | — | — | data/waits.csv |
| jsteinhardt | preemptive_high use of 8-GPU cap | 23% (mean 1.8 GPUs); 691 GPU-h headroom unused during Beaker backlog | — | — | — | data/summary.json |
| Lambda | other users' load | 20.5 / 24 GPUs; 29 of their jobs preempted at our starts | — | — | — | data/summary.json |
| Beaker | allocated share | 98% (4,871 / 4,991 GPU-h; flex2 ∈ oe-scaling) | — | — | — | data/beaker_report/ |

## Next step
Prasann picks which recommendations to adopt (1 minRuntime ≥ run length for Beaker training, 2 jupiter+ceres
default, 3 allocated vs unallocated decision); nothing changed yet. Re-run the scripts in ~3 weeks to measure the effect.

## Open issues / traps
- RESOLVED (branch, pending merge): allocated is now the rule; `--resume-interval` gives a resume point; `ops capacity`
  shows other users' load.
- The runtime history is empty until `ops status --record` accumulates `runtime_minutes` or someone runs
  `research/common/gpu_audit/backfill_runtimes.py --write` (6,111 runs; writes the LIVE registry). Leave-one-out,
  1.3 x the history median covers the actual runtime of 70% of Beaker training runs (flat 1h: 67%; p75 would give
  76%). Passing `--expected-minutes` is the stronger fix.
- In-job GPU utilization is unmeasured (no stored samples).

## Verified vs unverified
- Verified (2026-11-10): our Beaker total matches `beaker report gpu-usage` to within 1%. Preemption reasons
  quoted verbatim from Beaker job status. Trainer checkpoint default read from source.
- UNVERIFIED — confirm with Prasann: whether a longer minRuntime slows scheduling (no penalty seen at 2–3 h);
  whether ceres can run the 4B training image and data.
