# ops — research-ops toolkit

A small, **stdlib-only** command-line toolkit for running the research in this repo: a registry of
every launched run, one status view across Beaker, Berkeley slurm and Lambda, stall diagnosis, an
append-only results log, and one status page per experiment. It runs with any `python3` (≥ 3.9) on
the login node in a few seconds and imports nothing from `src/`.

```bash
OPS=/accounts/projects/berkeleynlp/prasann/bin/ops     # stable symlink; use this path everywhere
$OPS --help
```

## Why it exists

Before this (audit of 2026-10-12): experiment state was spread across memory files, `records/`,
per-campaign `LAUNCH_LEDGER.tsv` files and results-hub, and they disagreed; about 38% of the memory
index was stale point-in-time status; agents re-launched runs and missed assets that were already
recorded; job checks were ad-hoc `squeue`/`beaker` calls and sleep loops. `ops` gives each of
those one home:

| question | answer lives in | command |
|---|---|---|
| what did we launch, where, with which commit? | `~/ops_state/runs.jsonl` | `ops register`, `ops update`, `ops runs` |
| is it running / stuck / done? | live query of Beaker, slurm, Lambda | `ops status`, `ops stall ID` |
| what did it score? | `~/ops_state/results.jsonl` | `ops result add/check/show` |
| where does experiment X stand, what's next? | `experiments/<X>/STATUS.md` | `ops exp show X`, `ops exp index` |
| wrap a local command so a stall can't go unnoticed | — | `ops guard -- CMD` |

## Daily use

```bash
# before launching: what exists already?
$OPS exp show ds64
$OPS runs --experiment ds64 --since 14d

# at launch (every launch, every cluster)
$OPS register --run-id q35-4b-ds64-ck64-32M --experiment ds64 --cluster beaker --kind train \
  --job-id 01K... --expected-minutes 240 --wandb-url https://wandb.ai/... --git /path/to/repo

# checking on things — one call, no sleep loops
$OPS status                       # live + recently finished registered runs, all clusters
$OPS status --experiment ds64 --json
$OPS stall 01K...                 # deep diagnosis of one job: log tail, process state, GPU use

# when a run finishes
$OPS update q35-4b-ds64-ck64-32M final_state=done

# results: validate, then record (results-hub is retired)
$OPS result add --json /data/.../result.json --check
$OPS result show --experiment ds64 --best

# end of a work session on an experiment
$EDITOR ops/experiments/ds64/STATUS.md && $OPS exp index
```

## Commands

### Run registry — `~/ops_state/runs.jsonl`
- `ops register --run-id X --experiment E --cluster beaker|slurm|lambda|local [--kind train|eval|data|other]
  [--job-id ..] [--parent-run ..] [--arm ..] [--task ..] [--node ..] [--ckpt-path ..] [--log-path ..]
  [--wandb-url ..] [--expected-minutes N] [--git REPO_DIR] [--notes ..]` — append a launch event.
  `--git` fills `git_commit`, `git_branch` and `dirty` from that repo. For Beaker, the commit that
  matters is the **pushed** one.
- `ops update RUN_ID key=value ...` — append an update event (e.g. `final_state=done`, `ckpt_path=...`).
- `ops runs [--experiment E] [--cluster C] [--since 7d|2026-09-01] [--json]` — folded view.

Rows are events; readers fold by `run_id` (later non-null fields win). `run_id` is the run name =
save-folder basename = wandb run name; `experiment` matches a directory under `experiments/`.
Writers hold an `flock`, so concurrent sessions can register safely.

### Live status — `ops status`, `ops stall`, `ops guard`
- `ops status [--experiment E] [--all] [--json] [--no-beaker] [--no-lambda] [--fresh] [--record]` —
  queries Beaker, `squeue`/`sacct` and Lambda (over `ssh lambda`) for registered runs, each with a
  timeout and an "unavailable" fallback. Flags runs that are OVERRUN (more than 2x
  `expected_minutes`), stalled, or silent. `--record` writes final states back as update events,
  with `runtime_minutes` for finished runs (the history the launchers' minRuntime default uses).
- `ops capacity [--cluster all|berkeley|lambda] [--json]` — everyone's GPUs per node on
  jsteinhardt / berkeleynlp and Lambda (`ops status` shows only ours), GPU type, free GPUs, and our
  use of the per-user caps (8 at `preemptive_high` on jsteinhardt; 8 + 8 on Lambda) split by
  experiment. `--would-preempt --gpus N --qos Q [--partition P] [--node X]` says whether a new
  single-node job fits in free GPUs or whose jobs it would preempt (exit 3 = another user's).
  Placement skips GPUs below A100 unless `--any-gpu`. About 3 s.
- `ops stall JOB_OR_RUN_ID` — one job in depth: log freshness and tail, last progress line,
  process state (`D` + `nfs_wait_bit_killable` at ~0% CPU is the NFS-interpreter/log trap, not a
  hang), GPU utilisation.
- `ops guard [--first-output 120] [--stall 300] [--log PATH] [--kill] -- CMD ...` — run a local
  command and flag (or kill) it if it prints nothing within the first-output window or goes
  silent for the stall window.
- Lambda reports FAILED for runs that fully succeeded; verify by loss curve and checkpoint.
- Berkeley sbatch payloads: `bash "$REPO/recipes/env/ops_job.sh" CMD ...` = register + `ops guard` +
  final state from the exit code (knobs in its header). Lambda jobs can't reach the registry; register
  them from the login side.
- Claude Code hooks for these checks live in `hooks/` (see `INCORPORATE.d/` for how they are wired).

### Results — `~/ops_state/results.jsonl`
Each eval pass is its own run (`kind=eval`, `parent_run` = the training run): a result row's `run_id`
is the eval pass and `train_run` the training run it scores (filled from the registry; older rows
get it on read). Eval pass ids: the launcher's natural name (Beaker `ev-<task>-<run>`), else
`<train_run>__ev-<suite_or_task>-<variant>-<YYYYmmddHHMM>`.
- `ops result add (--json FILE | --run-id EVAL_RUN [--train-run T] --task .. --metric .. --value .. --eval-size ..) [--check]`
- `ops result check (...same...)` — warnings only (duplicate within one pass, jump vs other passes of
  the same train_run or vs the experiment best, eval_size < 500, missing provenance). Investigate
  every warning before quoting the number.
- `ops result show [--experiment E] [--task T] [--run-id R] [--train-run T] [--best]` — grouped by
  train_run, one line per pass; `--best` = best pass per task/rung.
- `ops backfill ledgers|results [--dry-run]` — import the old `LAUNCH_LEDGER.tsv` files and result
  CSVs (rows are tagged `backfill`).

Conventions: `eval_size` is the number of eval examples (never `n`; `n` is corpus size); `se` is the
binomial SE `sqrt(p(1-p)/eval_size)`; `cot_label` is `long-cot`/`short-cot`/`no-cot`; `task` is the
eval task actually graded.

### Experiments — `experiments/<name>/STATUS.md`
- `ops exp new NAME` — create a page from the template.
- `ops exp index` — regenerate `experiments/INDEX.md` (tables by status, newest first, with next step).
- `ops exp show NAME [--json]`, `ops exp stale [--days 7]` (active pages not updated in N days),
  `ops exp check` (front-matter validation; exits 1 on problems).
- `ops exp records [--write] [--records-dir D]` — regenerate `records/README.md` from each record's
  front-matter header (`title/date/updated/experiment/area/kind/status/superseded_by/summary`; falls
  back to the old README description, then the first paragraph; date from the last git commit).
  Default dir: `$OPS_REPO_ROOT/records`, else the `records/` next to this `ops/` checkout. Every
  record on `prasann/flopdrop_clean` carries a header; new ones copy the template the README prints.

STATUS.md front matter is deliberately simple (`key: value` lines and `- item` lists):

```yaml
---
experiment: ds64            # = directory name = `experiment` in runs/results
title: one line
status: active              # active | paused | done | abandoned
updated: 2026-10-12
branch: prasann/landmark
launcher: research/token_routing/ds64/launch_ds64.py
records:
  - records/ds64-handoff.md
results:
  - results/ds64/
wandb_group: https://wandb.ai/...
next_step: one line
---
```

Paths: repo-relative = this checkout (`prasann/flopdrop_clean` layout: `recipes/`, `research/`);
an absolute `/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/...` path = something that
exists only in the main checkout (old `debug/`, `results/`, unported records) and was not moved.

Body sections: **Question · Decisions** (dated, who decided) **· Current best numbers** (with
eval_size, ±SE, cot label, source path) **· Next step · Open issues / traps · Verified vs
unverified**. STATUS.md is the canonical current state; memory files and `records/` are history.

## Skills (`skills/`)

Claude Code skills that make agents use the toolkit; each calls the stable `ops` path. `skills/` is
their single source: the repo exposes them to project-skill discovery through relative symlinks
`.claude/skills/<name> -> ../../ops/skills/<name>`, and `~/.claude/skills/<name>` points at the same
dirs for every session.

| skill | triggers on | does |
|---|---|---|
| `job-status` | "status?", "is it running", "did X finish" | one `ops status`, `ops stall` on anything flagged, no sleep loops |
| `launch-run` | any training/eval/data launch | existing-runs check, smoke test, urgent + minRuntime, /data interpreter, `ops register` |
| `record-results` | results arrive, before quoting a number | validation checklist, `ops result add --check`, STATUS.md numbers |
| `close-out-experiment` | before /clear, /compact, handoff | STATUS.md + `ops exp index`, final run states, clean up watchers |
| `megaslides` | "add slides for X to the megaslides deck" | markdown spec → one Slides batch in the deck's dash-bullet format, native bar/line charts, stacked-thumbnail check (user-level link only) |

## Layout and rules

```
ops/
  bin/ops                 CLI entry (resolves the ~/bin symlink)
  opslib/                 paths, cli, registry, results, backfill, jobs, guard, experiments
  hooks/                  Claude Code PreToolUse hooks
  skills/<name>/SKILL.md
  experiments/<name>/STATUS.md, experiments/INDEX.md (generated)
  state/                  fallback only — LIVE runs.jsonl / results.jsonl live in
                          /accounts/projects/berkeleynlp/prasann/ops_state (own git repo,
                          outside any checkout; OPS_STATE_DIR overrides)
  tests/                  stdlib unittest; whole suite < 10 s
  INCORPORATE.d/          repo edits to apply when ops/ is merged
```

- Environment overrides: `OPS_STATE_DIR` (state files, used by tests), `OPS_REPO_ROOT` (the
  OLMo-core checkout), `OPS_EXPERIMENTS_DIR`.
- Tests: `cd ops && python3 -m unittest discover -s tests`.
- Never edit `state/*.jsonl` by hand except to append; fix a wrong row with an `update` event.
