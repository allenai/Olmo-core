# Using Beaker (AI2 cluster) from Berkeley

The Beaker counterpart to `local_cluster.md`. Everything here is validated from this machine
(`beaker` CLI at `~/.local/bin/beaker`, authed as **prasanns**; `gantry` lives in the
`corpus-reasoning-olmo` env — prepend its bin dir or use `source recipes/env/local_env.sh`).

**Constants (use these unless you have a reason not to):** workspace `ai2/flex2`, budget
`ai2/oe-other`, weka bucket `oe-training-default`, baked image
`tylerr/olmo-core-tch291cu128-2025-11-25` (== `OLMoCoreBeakerImage.stable`), secrets prefixed
`PRASANNS_` (wandb, AWS, beaker token, claude oauth).

**Allocation:** our jobs from `ai2/flex2` count as *allocated*: flex2 is in the `ai2/oe-scaling`
workspace group (98% of our GPU-h since about 2026-10-19). **That is correct** (Prasann, 2026-11-10;
the old "always unallocated" rule is retired). The side effect is that allocation balance preempts
our jobs once they pass minRuntime whenever `oe-scaling` is over target, hence rule 2's minRuntime.
Check with `beaker report gpu-usage --organizations ai2 --users prasanns --since 7d -g allocated,cluster`.

## The four golden rules

1. **Commit AND push before launching.** Both the internal `launch` command and gantry clone the
   repo at your *pushed* HEAD — a dirty tree errors, and an unpushed commit either fails
   (`RemoteBranchNotFoundError`, sometimes silently after "Launched") or ships stale code.
2. **Priority is ALWAYS `urgent` — exactly `urgent`, not `immediate`, never normal/high**; lower pends for hours
   behind capacity. Bump a queued job in place: `beaker job update-priority <JOB-id> urgent`.
   **And set minRuntime = min(4 h, 1.3 × expected runtime), `1h` when the runtime is unknown**
   (Prasann, 2026-11-10). Beaker preempts our jobs by *allocation balance* as soon as they pass
   minRuntime, at any priority, and a preempted run restarts from step 0 unless it has a resume
   point: 1,083 GPU-h of training was lost that way in 3 weeks at minRuntime 1h, while 99% of our
   training runs finish within 4 h (`records/gpu-usage-workflow-audit-2026-11.md`). Never `0`: urgent
   0s jobs get a days-long wait tail. `beaker_ctc_suite.py` and `run_q4b_beaker_multirung_eval.py`
   do this by default (`--min-runtime auto` / `BEAKER_MIN_RUNTIME=auto`): the estimate is
   `--expected-minutes`, else the registry's median runtime for similar finished runs
   (`ops/opslib/runtime.py`; `ops status --record` records `runtime_minutes`). Other launchers: call
   `opslib.safe.suggest_min_runtime(run_id=..., expected_minutes=...)`, or pass gantry
   `--min-runtime` / `BeakerLaunchConfig.min_runtime` by hand. **Runs expected past 4 h also get a
   rolling resume point** (`train_ctc_suite.py --resume-interval N`, on automatically in
   `beaker_ctc_suite.py`). `--save-interval` alone does nothing under `--no-final-checkpoint`.
3. **GPU jobs → `ai2/jupiter`** (H100s; gantry `--cluster` may need the full
   `ai2/jupiter-cirrascale-2` form). Never neptune for training — L40S, ~5x slower (MFU 4.5%).
4. **CPU jobs → multi-cluster, eager-first:** `--cluster ai2/neptune,ai2/ceres,ai2/saturn,ai2/jupiter`.
   Jupiter alone is strict-priority backfill and can queue a 0-GPU job 10+ min; multi-cluster
   schedules in seconds. (`ai2/holmes` is reserved for other budgets — unusable.)

## Launching a training run

Training scripts built on `ExperimentConfig` (the `research/<family>/` Beaker launchers) use the internal
`launch` pattern (the setA SFT instead goes through `recipes/train/setA/launch_setA_sft.py`):

```bash
cd $REPO && git push origin <branch>                    # rule 1
python -u research/<family>/<script>-SFT.py \
    launch <run-name> ai2/jupiter-cirrascale-2 --launch.priority=urgent
```

- `python -u` matters: `launch` submits then **streams logs until the run finishes** (the process
  staying alive is normal, not a hang), and without `-u` the
  `Experiment: https://beaker.org/ex/<ID>` line sits in the block buffer and never appears in a
  piped/captured shell. Grab the ID, then it's safe to kill the follower — the job runs
  independently. (`--launch.follow=false` does NOT work: soft-timeout requires follow.)
- Dry-run first when touching config: `SCRIPT.py dry_run <run-name> ai2/jupiter-cirrascale-2`.
- Surface the **wandb group link** as soon as the run starts.
- **Register it** once you have the ID: `ops register --run-id <run-name> --experiment <E>
  --cluster beaker --kind train --job-id <EXP-id> --expected-minutes N --wandb-url ... --git $REPO`
  (`ops` = `/accounts/projects/berkeleynlp/prasann/bin/ops`; `launch-run` skill). The shared
  Beaker launcher (`recipes/train/research_trainer/beaker_ctc_suite.py`) is being wired to register
  automatically — check `ops runs --experiment E` before registering twice.
- The auto-resume trap applies on Beaker too: relaunching a run name whose `--save-folder` already
  has a `stepN` checkpoint silently RESUMES it. Fresh experiment ⇒ fresh run name.

## One-off jobs with gantry (data staging, transfers, evals)

Template (this exact shape is validated — it's the S3→weka sync from the eval bundle staging):

```bash
gantry run --name <job-name> -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/neptune,ai2/ceres,ai2/saturn,ai2/jupiter --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --no-python --allow-dirty --timeout 0 --yes -- bash -c '<the work>'
```

- **The baked image is the speed trick**: deps (transformers/numpy/etc.) are pre-installed and the
  image is cached on AI2 nodes → near-zero startup. With `--beaker-image` pass `--install true`
  if the job needs `olmo_core` importable (the literal `true` no-op is REQUIRED — omitting
  `--install` triggers a slow `pip install -e .`); use `--no-python` when it doesn't.
- `--timeout 0` returns immediately — never let gantry block the shell (killing the shell cancels
  the job).
- AWS creds on-node come from the beaker secrets as above (`AWS_PROFILE=S3` = real AWS).

## Monitoring, canceling, priorities

**Check status with `ops status` (all registered runs, Beaker + slurm + Lambda, one cached call) and
`ops stall <EXP-id|run-id>` for anything flagged** — not hand-rolled `beaker experiment get` loops
(the PreToolUse hook denies foreground poll loops). `ops` reads the same JSON field as below
(`jobs[-1].status`, plus CANCELLED/PREEMPTED). The raw commands, for cancels and one-off digging:

```bash
beaker experiment get <EXP-id> --format json   # jobs[-1].status is the ONLY truth:
   # canceled→CANCELED · finalized→DONE (check exitCode) · started→RUNNING · scheduled→SCHEDULED
beaker experiment logs <EXP-id>
beaker job cancel <JOB-id>                     # JOB-id = jobs[-1].id from the JSON
beaker job update-priority <JOB-id> urgent
```

- The `beaker experiment get` **table view lies** (shows "running" for canceled jobs) and
  `beaker experiment stop` **silently no-ops** (exit 0, job keeps running). JSON + `job cancel`
  only, then re-verify `jobs[-1].status`.
- This CLI (v1.5.317) has no `beaker experiment list` — track experiment IDs at launch time.
- The shell here is **zsh: unquoted `$VAR` does not word-split**, so `for id in $IDS` loops over
  one newline-joined blob and every cancel/status loop silently breaks. Wrap list loops in
  `bash -c '...'`.
- `exitCode=0` ≠ success for evals — **grep the logs for `MISSING`** (see next section).

## Data: weka vs S3 (weka is NOT mounted at Berkeley)

- Jobs read/write **weka** (`/weka/oe-training-default/ai2-llm/checkpoints/prasanns/...`); this
  machine can only reach **S3** (`s3://ai2-llm/checkpoints/prasanns/...`, real AWS, the `S3`
  profile; the weka S3-gateway is firewalled from here).
- **Staging new data for on-beaker jobs is a TWO-step**: (1) `aws s3 sync <local> s3://...` from
  here, (2) a gantry job (template above) to `aws s3 sync s3://... /weka/...` on a weka node.
  S3-push alone = every input logs `MISSING`, the job skips everything and **exits 0** looking
  successful.
- Checkpoints weka→here: gantry job does a model-only resave (drop optimizer) + `aws s3 sync` to
  S3, then pull from S3 locally. See the `weka-s3-checkpoint-transfer` pattern
  (`/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/deprecated/scratch_transfer_cptmix32k.sh`
  in the main checkout is a worked example to adapt; not ported to this branch).

## Interactive + agent sessions

These live in the main checkout's `scripts/` (`/accounts/projects/berkeleynlp/prasann/projects/OLMo-core-legacy/scripts/`;
not ported to this branch):

- `interactive_gpu_session.sh` — live dev shell on an AI2 GPU with weka (GPUS=N, LOCAL=1).
- `launch_remote_control_agent.sh` — Claude Code agent on a Beaker GPU, steerable from
  your phone; OAuth token via masked beaker secret.
- `setup_dev_env.sh` — bootstrap the toolchain + sibling repos on a new machine.

## Lessons index (each burned real time)

| Lesson | Symptom if ignored |
|---|---|
| Push before launch | `RemoteBranchNotFoundError` (sometimes silent after "Launched"), or job runs stale code |
| `python -u` + kill follower | Launch "hangs"; experiment URL never appears in captured output |
| urgent priority, always | 1h+ pending; use `job update-priority` to rescue a queued job |
| minRuntime = min(4h, 1.3 × expected), never 0 | at 1h, allocation-balance preemption right after minRuntime restarts runs from step 0 (1,083 GPU-h in 3 weeks); at 0s urgent jobs have a days-long wait tail |
| `--resume-interval` past 4h, not `--save-interval` | under `--no-final-checkpoint` the checkpointer is disabled, so `--save-interval` writes nothing and a preempted run restarts from 0 |
| Multi-cluster for CPU | 0-GPU job unscheduled 10+ min on jupiter (1007/1008 slots) |
| Baked image + `--install true` / `--no-python` | ~3 min image pull + pip per job, dominating short jobs |
| JSON status, `job cancel` | Canceled jobs look "running"; `experiment stop` no-ops; wrong-base jobs kept running |
| `ops register` + `ops status`, no poll loops | Untracked jobs re-launched or forgotten; sleep loops burn the session |
| S3→weka gantry sync + grep `MISSING` | Eval exits 0 with zero results |
| Fresh run name per config | Silent auto-resume from a stale checkpoint (chimera runs, garbage timings) |
| zsh loops via `bash -c` | Cancel/status loops silently operate on one blob argument |
| Freshness guard trips on its own writes | Deleting a `stepN` bumps its **parent** mtime; an inline in-flight check then refuses every later path in that run (74/210 deleted). Resolve each parent once, in a pre-pass. |
