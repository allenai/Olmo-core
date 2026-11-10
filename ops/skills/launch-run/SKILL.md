---
name: launch-run
description: >
  Pre-launch checklist for ANY training, eval, data-build, benchmark or smoke-test job in
  Prasann's research repos — local Berkeley slurm (sbatch/srun), Beaker/gantry, or Lambda.
  Use it before EVERY launch or relaunch ("launch X", "kick off the sweep", "rerun with...",
  "start training", "submit the evals"): it checks for existing runs and assets first, enforces
  the smoke test, priority, interpreter and logging rules, and registers the run with `ops`.
  Pairs with where-to-run (which cluster) and job-status (after launch).
---

# Launch a run

`OPS=/accounts/projects/berkeleynlp/prasann/bin/ops`. Work through every box; most past failures
were one skipped box.

## Before writing anything
- [ ] **Existing runs/assets:** `$OPS runs --experiment E` and `$OPS exp show E`
  (`ops/experiments/INDEX.md` lists all). Has this already run, is a checkpoint/shard/serving copy
  already built, is a twin still live (`$OPS status --experiment E`)? Don't redo recorded work.
- [ ] **Where:** run the `where-to-run` skill (`python3 ~/.claude/skills/where-to-run/compute_avail.py`).
  No A100 without asking Prasann. Local jsteinhardt: preemptive_high, 8-GPU/user cap, never evict
  `rhys_gould`. Lambda: ≤8 GPUs preemptive_high + ≤8 preemptive, the rest at `normal`; use /tmp, not /data.
- [ ] **Preemption check (slurm / Lambda):** `$OPS capacity --cluster lambda|berkeley --would-preempt
  --gpus N --qos Q [--partition jsteinhardt] [--node X]`. `ops status` shows only our jobs, so a node
  that looks idle may be full. Exit 3 = this start would preempt another user's job; the output names
  them. On Lambda prefer free GPUs or wait; preempt only within the fair-share caps, and say so to the
  user. Rhys's jobs only up to 8 GPUs without asking. It also flags a start that would exceed our cap.
- [ ] **Training forward lengths:** no full-attention forward at 32k in training (hard ban).

## Smoke test first
- [ ] A short run of the **exact** launcher and config: **≥2 optimizer steps** (a `--max-tokens` below
  one step's tokens silently becomes a 1-step run), loss is finite, a checkpoint save works if the
  full run saves, the eval parses at least one example.
- [ ] Data/task check: train and eval come from the same generator (in-domain); marker embeddings
  repaired for docchunk/landmark bases (`fix_marker_embeddings.py`).

## Launch mechanics
- [ ] **Repo dir explicit:** Bash cwd resets between calls, so every command uses absolute paths or
  `git -C <repo>` / `cd <repo> && ...` in the same call. Launch from the repo and branch that holds
  the code (OLMo-core vs olmo-eval).
- [ ] **Beaker/gantry:** the commit is **pushed** (gantry runs the pushed commit; `allow_dirty` ships
  nothing). `priority=urgent` exactly (not lower, not `immediate`); **`minRuntime` = min(4 h,
  1.3 × expected runtime), `1h` if unknown, never `0`** (`beaker_ctc_suite.py` and the eval launcher
  do it from `--expected-minutes` or the registry's history: check the `minRuntime=... (source)`
  line they print; elsewhere use `opslib.safe.suggest_min_runtime`). Runs expected past 4 h need a
  resume point (`--resume-interval`, automatic in `beaker_ctc_suite.py`); workspace `ai2/flex2`, budget
  `ai2/oe-other` (allocated via the `ai2/oe-scaling` group, which is correct (Prasann, 2026-11-10)); cluster ai2/jupiter. Final checkpoint not written twice.
- [ ] **Local slurm:** interpreter `/data/prasann/conda/envs/<env>/bin/python` (node-local; never
  bare `torchrun`, use `$PY -m torch.distributed.run`); `--output`, logs, HOME, TMPDIR, triton and
  flashinfer caches and checkpoints on `/data`; stage code with `git archive | srun ... tar -x` from
  the login node.
- [ ] **Progress logging** at items/steps 1, 2, 5, 10 with an ETA; a zero-output stall must show in
  seconds. Keep raw eval generations (KEEP_RESPONSES=1).
- [ ] **Fresh save folder** (relaunching into an old one silently auto-resumes).

## Register and watch
```bash
$OPS register --run-id RUN --experiment E --cluster beaker|slurm|lambda --kind train|eval|data \
  --job-id ID --expected-minutes N --log-path /data/... --ckpt-path ... --wandb-url ... \
  --git /abs/path/to/repo        # fills git_commit/branch/dirty (Beaker: the PUSHED commit)
# local long command not under slurm: wrap it so stalls kill or flag it
$OPS guard --first-output 120 --stall 300 --log /data/.../run.log -- CMD ...
```
- [ ] Share the **wandb group link** with the user at launch.
- [ ] Watch the first minutes until the **first progress line** appears (`$OPS status`, then
  `$OPS stall ID` if nothing shows), then leave it in the background. No sleep loops; use the
  `job-status` skill or the Monitor tool.
- [ ] Add the run and next step to `ops/experiments/E/STATUS.md`.
