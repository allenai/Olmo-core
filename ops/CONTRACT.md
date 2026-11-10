# ops/ interface contract (build-time; folded into README.md when done)

`ops/` is a standalone, **stdlib-only** (python3 >= 3.9, no torch/olmo_core imports) research-ops
toolkit: run registry, cross-cluster job status + stall detection, results log, experiment status
pages. It must run with any `python3` on the login node in < a few seconds, and must not depend on
anything under `src/` (the repo is being refactored; `ops/` merges in later as a pure addition).

## Layout

```
ops/
  bin/ops                 # CLI entry (dispatches to opslib.<module>.add_parser)
  opslib/paths.py         # OPS_ROOT, STATE_DIR (env OPS_STATE_DIR overrides), REPO_ROOT guess
  opslib/cli.py           # argparse dispatcher; each module below exposes add_parser(subparsers)
  opslib/registry.py      # runs.jsonl            (owner: agent A)
  opslib/results.py       # results.jsonl         (owner: agent A)
  opslib/backfill.py      # ledgers/CSVs -> state (owner: agent A)
  opslib/jobs.py          # live status           (owner: agent B)
  opslib/guard.py         # stall-watch wrapper   (owner: agent B)
  opslib/experiments.py   # STATUS.md index       (owner: agent C)
  hooks/                  # Claude Code PreToolUse hook scripts (owner: agent B)
  skills/<name>/SKILL.md  # Claude Code skills    (owner: agent C)
  experiments/<exp>/STATUS.md, experiments/INDEX.md (generated)  (owner: agent C)
  state/runs.jsonl, state/results.jsonl   # tracked in git, append-only, small
  state/cache/            # gitignored scratch cache (beaker/squeue snapshots)
  tests/test_<module>.py  # stdlib unittest; whole suite < 10 s
  INCORPORATE.d/<topic>.md  # each agent: what must change in the REPO after the refactor lands
```

Stable invocation path for skills/hooks/agents: **`/accounts/projects/berkeleynlp/prasann/bin/ops`**
(a symlink, repointed at incorporation). Never hardcode the worktree path in skills.

## state/runs.jsonl — one JSON object per line, append-only

Rows are *events*. Readers fold rows by `run_id` in file order (later non-null fields win).

| field | type | notes |
|---|---|---|
| `event` | `"launch"` \| `"update"` | |
| `run_id` | str | unique key. For training = run_name = save-folder basename = wandb run name. Each eval PASS is its own run (`kind=eval`): the launcher's natural eval name if it has one (e.g. Beaker job `ev-<task>-<run>`, timestamp-suffixed if already taken), else `<train_run>__ev-<suite_or_task>-<variant>-<YYYYmmddHHMM>` |
| `experiment` | str | logical campaign key; matches `ops/experiments/<experiment>/` (e.g. `ds64`, `flop_scaling`, `ffndrop`) |
| `kind` | `"train"`\|`"eval"`\|`"data"`\|`"other"` | |
| `parent_run` | str\|null | eval pass → the train run_id it scores (required for `kind=eval`) |
| `arm`, `task` | str\|null | |
| `cluster` | `"beaker"`\|`"slurm"`\|`"lambda"`\|`"local"` | |
| `job_id` | str\|null | Beaker experiment/job id or slurm job id |
| `node` | str\|null | |
| `ckpt_path`, `log_path`, `wandb_url` | str\|null | |
| `git_commit`, `git_branch` | str\|null | commit actually shipped (Beaker = pushed commit) |
| `dirty` | bool\|null | working tree dirty at launch |
| `command` | str\|null | launcher argv |
| `expected_minutes` | number\|null | declared at launch; status flags OVERRUN at > 2x |
| `final_state` | `"done"`\|`"failed"`\|`"cancelled"`\|null | only via update events |
| `runtime_minutes` | number\|null | actual run time of the finished job, written with `final_state=done` by `ops status --record`; its history median is the default expected runtime for minRuntime (`opslib/runtime.py`) |
| `state_source` | str\|null | where final_state came from |
| `launched_at` / `ts` | ISO-8601 UTC | `ts` = when this row was written |
| `launched_by` | str | session id, `manual`, or `backfill` |
| `source` | str | `launch`, `cli`, or `backfill:<path>` |
| `notes` | str\|null | |

Writers take an exclusive `fcntl.flock` on `state/runs.jsonl.lock`. Python API (registry.py):
`register(**fields) -> dict`, `update(run_id, **fields) -> dict`, `load_runs(path=None) -> dict[run_id, folded_row]`,
`find(experiment=None, since=None, cluster=None) -> list[dict]`.

## state/results.jsonl — append-only

| field | type | notes |
|---|---|---|
| `run_id` | str | the EVAL run (one per eval pass); older/backfilled rows hold the training run here |
| `train_run` | str | the training run scored. `add` fills it from the registry (`parent_run` of a registered eval run, else `run_id`); `load()` fills it on read for rows that predate the field (no rewrite) |
| `experiment`, `task` | str | `task` = eval task name actually graded (not the training task name) |
| `rung` | int\|str\|null | context-length rung label as used in files (e.g. `rung_16384`, `32k`) |
| `metric` | str | e.g. `f1`, `acc`, `recall` |
| `value` | float | |
| `eval_size` | int | number of eval examples (NEVER call it `n`; `n` = corpus size) |
| `se` | float\|null | binomial SE for proportion metrics: sqrt(p(1-p)/eval_size) |
| `cot_label` | `long-cot`\|`short-cot`\|`no-cot`\|null | |
| `eval_backend` | str\|null | e.g. `olmo-eval-vllm`, `native` |
| `ckpt_path`, `git_commit`, `eval_job_id`, `source_json` | str\|null | |
| `recorded_at`, `recorded_by` | str | |
| `flags` | list[str] | e.g. `eval_size<500`, `backfill`, `provenance_incomplete` |

Python API (results.py): `add(**fields) -> dict` (validates; raises on missing run_id/task/metric/value/eval_size
unless `backfill=True`), `load(path=None) -> list[dict]` (with `train_run`), `check(row) -> list[str]`
(warnings vs prior rows: duplicate = same `run_id`+task+rung+metric; deviation vs other passes of the same
`train_run` and vs the experiment best). `ops result show` groups by `train_run` and shows each pass's
`run_id`; `--best` = best pass per (task, rung, metric).

## CLI (all subcommands; `--json` gives machine output)

```
ops register --run-id X --experiment E --cluster C [--kind train] [--job-id ..] [--ckpt-path ..] [--expected-minutes N] ...
ops update RUN_ID key=value [key=value ...]
ops runs [--experiment E] [--since 7d] [--json]
ops status [--experiment E] [--all] [--json] [--no-beaker] [--no-lambda] [--fresh]
ops stall JOB_OR_RUN_ID                 # deep diagnosis of one job
ops guard [--first-output 120] [--stall 300] [--log PATH] [--kill] -- CMD ...
ops result add (--json FILE | --run-id EVAL_RUN [--train-run T] --task .. --metric .. --value .. --eval-size ..) [--check]
ops result check (--json FILE | same fields)
ops result show [--experiment E] [--task T] [--run-id EVAL_OR_TRAIN_RUN] [--train-run T] [--best]
ops backfill ledgers|results [--dry-run]
ops exp index | ops exp show NAME | ops exp new NAME | ops exp stale [--days 7]
```

## Rules for every builder
- Write only the files you own. No git commands that write (no add/commit/checkout/stash); use
  `git --no-optional-locks` for reads. The main checkout and other worktrees are READ-ONLY
  (another agent is refactoring the repo).
- stdlib only; fast; every external command (squeue, sacct, beaker, ssh) gets a timeout and a
  graceful "unavailable" fallback, never a hang.
- Login node = light work only. `python3 -m unittest discover -s ops/tests` must finish < 10 s;
  anything heavier goes to `srun --partition=berkeleynlp --qos=normal --nodelist=horton --cpus-per-task=4 --mem=8G --time=00:10:00`.
- Record repo-side follow-ups (things that must change in launchers/trainer/CLAUDE.md/records after
  the refactor) in `ops/INCORPORATE.d/<your-topic>.md` — precise file + change, not prose.
