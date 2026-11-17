---
name: training-smoke-test
description: >
  This skill should be used when the user asks to "create a verification script",
  "write a test training run", "make a quick training job", "verify my change with a
  short run", "launch a smoke test", or wants a small Beaker job, or a local slurm
  (horton / jsteinhardt) or Lambda smoke run, to validate that a feature works
  end-to-end on GPUs. Also triggers when the user mentions creating a modified 190M
  script to test a specific behavior. Pairs with launch-run (checklist + `ops register`)
  and where-to-run (which cluster).
---

# Training job smoke test

Create short, single-node training scripts that exercise a specific feature in a
real GPU run. These scripts are derived from the 190M base config
(`src/scripts/train/OLMo3/OLMo-3-190M.py`) and train for only ~20 steps with an
eval callback to confirm the feature works.

## When to use

- After implementing a new training feature (parallelism mode, eval type, data
  pipeline change) that needs GPU validation beyond unit tests.
- When a CI test cannot cover the behavior (e.g., multi-GPU distributed features).
- To produce a quick smoke-test script that can be launched on Beaker.

## Workflow

### 1. Identify what to verify

Determine the feature under test and what "success" looks like. Examples:

| Feature | Success signal |
|---------|---------------|
| CP perplexity evals | LM evaluator reports CE loss and PPL without error |
| TP training | Training completes 20 steps, loss decreases |
| New data mix | Data loader produces batches without errors |
| New attention backend | Training completes 20 steps with the new backend, no kernel errors |

### 2. Copy and modify the 190M base script

Start from `src/scripts/train/OLMo3/OLMo-3-190M.py` and place the new script in
`research/<family>/smoke/` (e.g. `research/kv_route/smoke/`). `src/scripts/` is
upstream-only in this repo — the existing `src/scripts/train/smoketests/` files are
upstream's examples, not a place for ours.

Key modifications:

- **File name**: `<feature>-test.py` in `research/<family>/smoke/`.
- **Docstring**: Describe what is being verified and how to run it.
- **Duration**: `Duration.steps(20)` — just enough to confirm the feature works.
  Never fewer than **2 optimizer steps**: when a run is sized by `--max-tokens`, a value
  below one step's tokens silently becomes a 1-step run, which proves nothing about the
  optimizer, the LR schedule or step-2 state.
- **Batch size**: `SEQ_LENGTH * 16` — small but enough for distributed training.
- **Metrics/cancel interval**: 5 steps (frequent enough to catch issues early).
- **Remove unused features**: Drop `Float8Config`, `InstanceFilterConfig`,
  `CheckpointerCallback`, `FOR_BENCHMARKING`, `CHINCHILLA_MULTIPLE`,
  `estimate_lr()` none of these are needed for a quick verification.
- **WandB**: Keep but set `enabled=False`.

### 3. Add the feature-specific config

Depending on what is being verified, add the relevant config. Common patterns:

**Context Parallelism (CP):**
```python
from olmo_core.train.train_module import TransformerContextParallelConfig

# In train_module_config:
cp_config=TransformerContextParallelConfig.ulysses(degree=2),
```

**Tensor Parallelism (TP):**
```python
from olmo_core.train.train_module import TransformerTensorParallelConfig

# In train_module_config:
tp_config=TransformerTensorParallelConfig(degree=2),
```

**New attention backend:**
```python
from olmo_core.nn.attention import AttentionBackendName

# In model_config — override the backend selection:
model_config = TransformerConfig.olmo3_190M(
    vocab_size=tokenizer_config.padded_vocab_size(),
    attn_backend=AttentionBackendName.flash_4,  # or .te, .torch, etc.
)
```
Available backends: `torch`, `flash_2`, `flash_3`, `flash_4`, `te`.
The base script selects between `flash_2` and `flash_3` based on GPU type;
a verification script can hardcode a specific backend to test it.


**PPL evals (LMEvaluator):**
```python
from olmo_core.data import NumpyPaddedFSLDatasetConfig
from olmo_core.train.callbacks import LMEvaluatorCallbackConfig

# In trainer_config:
.with_callback(
    "lm_evaluator",
    LMEvaluatorCallbackConfig(
        eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
            DataMix.v3_small_ppl_validation,
            mix_base_dir=get_root_dir(cli_context.cluster),
            sequence_length=SEQ_LENGTH,
            tokenizer=tokenizer_config,
            work_dir=work_dir,
        ),
        eval_interval=10,
    ),
)
```


**Full recommended evals (PPL + downstream):**
```python
trainer_config = trainer_config.with_recommended_evals(
    tokenizer_config, SEQ_LENGTH, cli_context.cluster, task_set="fast"
)
```
Note: downstream evals require full logits and are incompatible with CP or TP.

### 4. Launch and verify

Always launch at `--launch.priority=urgent` — exactly `urgent`: never lower, never
`immediate`, and do not ask the user. Workspace `ai2/flex2`, budget `ai2/oe-other`
(allocated via the ai2/oe-scaling group, which is correct (Prasann, 2026-11-10)), cluster `ai2/jupiter`. Gantry runs the
**pushed** commit (`allow_dirty` ships nothing) — confirm it is pushed; ask before pushing.

```bash
# Dry run first to check the config renders:
python research/<family>/smoke/<feature>-test.py \
    dry_run test-<feature> ai2/jupiter

# Launch on Beaker:
python research/<family>/smoke/<feature>-test.py \
    launch test-<feature> ai2/jupiter \
    --launch.priority=urgent \
    --launch.workspace=ai2/flex2 \
    --launch.budget=ai2/oe-other \
    --launch.follow=false
```

Always launch with `--launch.follow=false` (or set `--launch.launch_timeout=<seconds>`
to cap how long to wait). This avoids blocking the terminal indefinitely.

By default the `launch` command follows the job logs in the terminal until
completion, which requires `step_soft_timeout` and blocks the session. Using
`follow=false` lets the job run asynchronously.

After launching, **report the Beaker experiment link** to the user so they can
monitor the job, and register it:
`/accounts/projects/berkeleynlp/prasann/bin/ops register --run-id test-<feature> --experiment <E> --cluster beaker --kind train --job-id <id> --expected-minutes 30 --git <repo> --notes smoke`.
Check on it with `ops status` / `ops stall <id>` (the `job-status` skill), never a sleep loop.

To reconnect to a running job later:
```bash
gantry follow <experiment-id>
```

### 5. Confirm success

Check the Beaker logs for:
- Training steps completing without errors.
- Throughput / MFU at expected levels.
- Eval metrics being reported (CE loss, PPL for LM evals).
- No NCCL or distributed errors.

## Local slurm / Lambda smoke tests

For a research arm (anything driven by `recipes/train/research_trainer/train_ctc_suite.py`
or a `research/<family>/` driver) the smoke test is usually a short run of the **exact**
launcher on Berkeley slurm or Lambda, not a 190M Beaker job. Follow the `launch-run`
skill; the rules that bite smoke tests:

- **≥2 optimizer steps.** A `--max-tokens` below one step's tokens is a 1-step run.
- **Node-local interpreter.** `PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python`
  and `$PY -m torch.distributed.run ...` — never bare `torchrun` (it resolves to the NFS
  miniforge) and never an interpreter under `/scratch`, `/accounts` or `/net/...`.
- **Everything the job writes on `/data`.** slurm `--output`/`--error`, logs, `HOME`,
  `TMPDIR`, triton / flashinfer caches and checkpoints go on the node's `/data`; an NFS
  `--output` deadlocks at step 0 and looks like a hang. `recipes/env/local_env.sh` sets the
  cache variables.
- **Stage code from the login node** into a fresh node-local dir:
  `git archive <commit> | srun -p <partition> -w <node> bash -c 'mkdir -p /data/prasann/<fresh> && tar -x -C /data/prasann/<fresh>'`.
- **Fresh save folder** — relaunching into an old one silently auto-resumes.
- **Lambda** (`ssh lambda`, `recipes/train/research_trainer/run_ctc_lambda.sbatch`): node-local
  disk is `/tmp` (no usable `/data`); ≤8 GPUs on `preemptive_high` + ≤8 on `preemptive`;
  `sbatch`/`sacct` FAILED ≠ failed — check the loss curve and checkpoint. See `lambda_cluster.md`.
- **Register** the smoke run (`ops register --kind train --notes smoke ...`) and check it with
  `ops status` / `ops stall` rather than polling `squeue`. Local slurm details: `local_cluster.md`.

## Naming convention

Scripts live in `research/<family>/smoke/` and are named `<feature-slug>-test.py`, e.g.
`research/kv_route/smoke/cp-ppl-eval-test.py`, `research/landmark/smoke/flash4-attn-test.py`.
