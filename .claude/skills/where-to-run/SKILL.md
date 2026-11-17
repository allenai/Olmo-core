---
name: where-to-run
description: >
  Use this skill in ANY repo whenever you are about to run, launch, schedule or plan experiments
  or any GPU work — training runs, sweeps, evals, smoke tests, benchmarks, GPU data generation,
  sbatch/srun/gantry/Beaker launches, interactive GPU sessions — and before choosing a cluster,
  partition, QOS or node. Also for "where should this run?", "is there capacity?", "how long
  until it schedules?", "is beaker/jupiter busy?", "are the local nodes free?", "can I preempt
  anything?". Gives a ~4 s snapshot of Berkeley slurm (jsteinhardt, berkeleynlp) and AI2 Beaker
  with start ETAs and a ranked, copy-pasteable recommendation.
---

# Where to run

One command from any directory, ~4 s, stdlib python (never use an NFS conda env for it):

```bash
python3 ~/.claude/skills/where-to-run/compute_avail.py                       # 1/2/4/8 GPUs
python3 ~/.claude/skills/where-to-run/compute_avail.py -g 8 -g 16 --hours 12 # sizes + H200 runtime
python3 ~/.claude/skills/where-to-run/compute_avail.py --json                # agents: parse this
python3 ~/.claude/skills/where-to-run/compute_avail.py --debug -g 1          # small debug job: berkeleynlp allowed (normal)
python3 ~/.claude/skills/where-to-run/compute_avail.py --html                # + HTML view
```

`--html` writes `./visualizations/compute_avail.html` when the cwd has a `visualizations/` dir,
else `~/.cache/where-to-run/compute_avail.html`. Source of truth lives in the OLMo-core repo at
`.claude/skills/where-to-run/` (`~/.claude/skills/where-to-run` is a symlink to it; OLMo-core's
`scripts/compute_avail.py` is a convenience symlink) — edit it there.

Only the snapshot is global. Repo-specific launch recipes (env, `/data` paths, sbatch headers,
gantry flags) still come from the repo you are in.

Run it fresh right before launching — capacity moves within minutes. Pass `--hours` with the
job's expected wall-clock on H200s: the ranking minimizes *start ETA + runtime × GPU-speed
factor*.

Scope is deliberately narrow: only **A100-or-better** GPUs (local RTX/A4000 boxes and Beaker
L40S/A6000 clusters like neptune are dropped), and only local QOS **≥ `preemptive`** (`normal`
is never shown — it is preemptible by everything), except on berkeleynlp (below).

**berkeleynlp: small-scale debugging only, `--qos=normal` only (Prasann, 2026-10-16).** Never run
real training/eval work there, and never use `preemptive` / `preemptive_high_sewonm` on it (an
srun/sbatch with no `--qos` defaults to `preemptive_high_sewonm`). The script shows berkeleynlp only
at `normal` and recommends it only with `--debug`. The ops bash hook denies any other QOS there.
jsteinhardt and Lambda are unaffected.

**A100s: check with Prasann before using one.** The script never picks an A100 (local balrog,
Beaker saturn/prometheus) when an H100/H200/B200 can fit at all, and every A100 pick or
"A100 could start sooner" note carries that warning. Don't launch on an A100 without asking.

**Preempting `normal` jobs locally is always fine and expected** — the script counts GPUs held
by `normal` jobs as free (`● NOW`, "preempt normal-QOS … (always fine)"). No need to ask or
hesitate. (Preempting `preemptive` holders at `preemptive_high*` is also fine; it shows as `◐`.)

**Spill rule (Prasann, 2026-11-10): small, short jobs go wherever they start soonest.** A job of
≤ 2 GPUs and ≤ 2 h (`-g 1`/`-g 2`, `--hours ≤ 2`) whose best pick is a Beaker queue with an expected
wait over 30 min goes to jsteinhardt instead (the pick then says `SPILL RULE`): at `preemptive_high`
while our 8-GPU cap has room, else at `preemptive` on GPUs that are free or held at `normal`. The
jsteinhardt cap is **8 GPUs at `preemptive_high` only**; more may run at `preemptive` when capacity is
available (Prasann, 2026-11-10). `preemptive` bumps `normal` and is bumped by `preemptive_high`, so
only restartable work goes there (evals, requeue-safe jobs). The LOCAL SLURM header shows what all of
our sessions hold at each of the two QOS right now, split by experiment from the run registry. The LAMBDA line does the same
for its 8 + 8 caps. Before an actual slurm or Lambda start, `ops capacity --would-preempt ...`
names any other user's job it would preempt (see the launch-run skill). Evidence: over 2026-10-20 →
11-10, jsteinhardt ran at 23% of the cap while Beaker jobs spent 60% of their lifetime pending
(`records/gpu-usage-workflow-audit-2026-11.md`). Tests: `python3 test_spill.py` in this directory.

## Reading it

- **BEST PLACE PER REQUEST SIZE** — the pick plus two runners-up. The `where` string is directly
  usable: `jsteinhardt --qos=preemptive_high -w cubbins` → `#SBATCH --partition / --qos /
  --nodelist`. For Beaker it names the cluster (launch urgent from ai2/flex2, as always).
- Status glyphs: `●` free now · `◐` needs preemption, starts in minutes · `○ ≤X` must wait; X is
  an **upper bound** from the blocking jobs' scheduled end times (slurm jobs usually end early)
  · `⊘` QOS cap (per-user or group) is the blocker · `—` can't fit · `○ no fit` (Beaker) = no
  node can host it even after preempting every lower-priority unallocated job.
- **preempt-risk to you** — whether a higher QOS can bump your job, and how much of that work is
  already pending. `preemptive_high*` = none. Plain `preemptive` can be requeued.
- Node bars: `█` busy and not yours to take · `▒` held by lower-QOS jobs you can preempt at your
  best QOS · `░` free.
- **Beaker ETAs** are gated by structure first: a job only gets a time if some node can host it
  from free slots plus slots held by lower-priority *unallocated* jobs whose min-runtime has
  expired (or will — that expiry is the ETA). On top of that sits the measured created→scheduled
  p50 of recent urgent jobs from never-allocated workspaces (`jobs=` = sample count).
  A trailing `?` (e.g. `◐ ~2m?`) means **unverified**: preemptable capacity exists, but urgent
  jobs of that size are queued and nothing that size has recently scheduled unallocated — treat
  as "plausible, not measured". Workspaces that hold *any* allocated slots are treated as
  allocated wholesale (jobs carry no per-job flag), so their fast waits never count as ours.
  "urgent ahead (oldest …)" is context — days-old pending urgent jobs are usually unsatisfiable.
  Strict-priority clusters (jupiter/titan) let allocated jobs preempt us; eager ones don't.

## Policy it already encodes (don't re-derive)

- A100 options always rank last. Then, ties within 15 min: no preempt-risk > free-now (incl.
  preempting `normal`) over preempting `preemptive` > jsteinhardt > berkeleynlp (only with
  `--debug`) > Beaker, then GPU speed — i.e. the job-routing-policy memory.
- Beaker recommendations are **jupiter only** (standing GPU directive). Other clusters show as
  "(info only)"; `--beaker-any` lets them win. Titan/B200 is untested with our image.
- `rhys_gould`'s jobs are never counted as preemptable; nodes they hold get an "ask first" note —
  ask Prasann before preempting them.
- `--gpu-type H200` restricts local nodes to H200 only (default: any A100+).

## Known limits

- Local placement is single-node only (G > node size shows `—`).
- A node that looks idle for a long time (e.g. lorax with 7/7 free) may be idle because it is
  flaky — `sinfo -R` before betting a long run on it.
- GPU speed factors (`SPEED` in the script) are rough bf16-LM-training guesses; edit there.
- Lambda (A100, `ssh lambda`) is shown as an info-only section (one ssh call; `--no-lambda` skips it): per-node
  free GPUs, holders, and your GPUs vs the fair-share caps (≤8 preemptive_high + ≤8 preemptive, rest normal).
  It is never in the ranked picks (A100: ask Prasann). Launch recipe: OLMo-core's `lambda_cluster.md`.
- User-specific constants (slurm user, Beaker author `prasanns`, oe-other budget id, jupiter-only,
  ask-first users) are at the top of the script.
