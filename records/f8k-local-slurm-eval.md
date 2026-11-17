---
title: fast8k 4-task eval on Berkeley slurm, with parity against Beaker
date: 2026-11-10
updated: 2026-11-10
experiment: gpu_audit
area: pipeline
kind: howto
status: current
superseded_by:
artifact: https://claude.ai/artifact/AxWXjBS1jDQy51MtGszd3d
summary: A local slurm path for the fast8k mt4 evals (contradiction/nq/outlier/oolong x 8k+2k, 500 rows per cell) that runs the same Beaker on-node runner, one jsteinhardt job per checkpoint at --qos=preemptive with a per-cell resume. On the recipe checkpoint all 8 cells match the Beaker scores within 0.002 (binomial SE 0.007-0.021), the 8k mean is 0.846 vs 0.845, and a checkpoint takes ~30 min on 2 H200s with no queue. Command, node requirements and caveats inside.
---

# fast8k 4-task eval on Berkeley slurm

**Why.** On 2026-11-10 our jupiter eval jobs had been pending for more than 4 hours, and every result in two
running tokdrop_lr experiments was waiting on them. Prasann approved recommendation #4 of
`gpu-usage-workflow-audit-2026-11.md` (spill small, short jobs to wherever they start soonest). This is the
fast8k eval made local.

**What it runs.** It is the eval `launch_fast8k.py eval` submits to Beaker as `ev-*` jobs:

- tasks contradiction, nq, outlier and oolong, at the 8k and 2k rungs, 500 rows per cell;
- query position `after`, no-cot, docchunk ladder, dense emitter, bs=1 greedy decode, `--max-length 20000`;
- the same rung files and graders.

It goes through the same on-node runner (`recipes/eval/legacy_native/run_beaker_multirung_eval.sh`), with
the arguments `run_q4b_beaker_multirung_eval.py` passes it. Code lives in
`research/token_routing/ds64_fast8k/local_eval/` (README there).

## Parity on the recipe checkpoint

- **Checkpoint:** `f8k-mt4-lora64all32r20to7-f1952-tde-lr1e-05`, the recipe run.
  - Beaker scored the weka export, re-installing its LoRA adapters from `config.json`.
  - Local scored grad_concentration's model-only copy on S3, with the adapters merged in fp32 as
    W + 32·BA, plus a plain Qwen3.5-4B `config.json` (`--dense-config`).
- **Code:** the same commit as the Beaker evals, `099491837`.
- **Local runs:** slurm 4432702 + 4432790 on mooney, 2 H200s, `--qos=preemptive`.
- **Beaker runs:** jobs 01M5W4X…, 2 H100s.

| task | rung | Beaker | local | diff | binomial SE (eval_size 500) |
|---|---|---|---|---|---|
| contradiction (f1) | 8k | 0.959 | 0.960 | +0.001 | 0.009 |
| nq (f1) | 8k | 0.912 | 0.912 | 0.000 | 0.013 |
| outlier (f1) | 8k | 0.862 | 0.860 | −0.002 | 0.016 |
| oolong (score) | 8k | 0.648 | 0.650 | +0.002 | 0.021 |
| contradiction (f1) | 2k | 0.977 | 0.978 | +0.001 | 0.007 |
| nq (f1) | 2k | 0.970 | 0.972 | +0.002 | 0.007 |
| outlier (f1) | 2k | 0.880 | 0.879 | −0.001 | 0.015 |
| oolong (score) | 2k | 0.856 | 0.857 | +0.001 | 0.016 |
| **8k mean** | | **0.845** | **0.846** | +0.000 | |

Every cell agrees within 0.002, about a fifth of one SE or less. The leftover differences are what you'd
expect from bf16 numerics:

- H200 vs H100;
- the merged vs separately applied LoRA;
- the node env (torch 2.9.1+cu128, flash-attn 2.8.2) vs the Beaker image.

These numbers come from one checkpoint, so they show the pipeline is the same measurement; they don't
bound noise across runs.

**Recorded:** `ops result show --train-run f8k-mt4-lora64all32r20to7-f1952-tde-lr1e-05`, rows with
`run_id …-evslurm-4432702` (nq, outlier, oolong) and `…-evslurm-4432790` (contradiction). The second job also
re-filed the first job's six cells under its own run id, with identical values. That bug is fixed: each cell
now records the job that scored it. The collectors keep one row per (task, rung), so the duplicates change
nothing downstream.

**LoRA export path (unmerged).** Smoke job 4432918 on cubbins, with code from `claude/oracle-thresh-sweep-69bc7d`,
loaded a `train_ctc_suite` export that keeps its adapters as separate keys and re-installed them from
`config.json`. It scored one cell with recording off: `[early-lora] adapters re-installed from config.json:
{'n_blocks': 32, 'rank': 64, 'scale': 32.0}`, then nq@2k f1 0.918 over 500 rows, 4.5 min including the
60 s model build. That is a smoke checkpoint, so there is no Beaker reference to compare against.

## Cost

| | Beaker (4 `ev-*` jobs) | local (1 job) |
|---|---|---|
| GPUs | 2 H100 per task job | 2 H200 |
| time per checkpoint | ~12 min per task job (contradiction: 7 min at 8k, 5 at 2k), plus the queue: >4 h on 11-10 | ~30 min for all 8 cells (2-5 min each, model load included) + ~2 min staging |
| preemption | allocation-balance preemption restarts the job | slurm requeues; finished cells are skipped |

## How to run it (for other sessions)

```bash
python3 research/token_routing/ds64_fast8k/local_eval/submit_eval_local.py \
    --run <train run> --node <node> --ckpt /data/prasann/ctc_suite/ckpts/<train run>
```

- Run it from the checkout whose HEAD should score the checkpoint, or pass `--src`/`--code-ref`. LoRA
  runs need a ref with `olmo_core.nn.early_lora`.
- `--ckpt-s3 <key prefix under s3://ai2-llm/>` stages a copy from the S3 hub onto any H200 node.
- `--dense-config` is only for model-only resaves with no `config.json`.
- `--dry-run` shows the sbatch line and who the start would preempt.
- Results go to `/data/prasann/f8k_eval/runs/<run>/eval/<key>_multirung.json` on the node (the Beaker
  layout), and into ops results under `tokdrop_lr` with the fields `record_beaker_evals.py` writes.

Needed on the node:

- the node-local env `/data/prasann/conda/envs/corpus-reasoning-olmo` (cubbins, mooney, sneetches, mcfuzz);
- `/data/prasann/joblogs/`, which mcfuzz lacks, so mcfuzz is excluded.

The job stages everything else itself:

- the code at the pinned commit;
- the eight rung files from `s3://ai2-llm/checkpoints/prasanns/_transfer/f8k_eval/` (md5-checked);
- the tokenizer, from `/data/prasann/hf_cache`.

## Traps found while building it

- **`DC_RUNG_FILES` keyed by the runner key.** The runner's TASK for contradiction is `contra`, but the
  ladder looks up `contradiction`. Building the override inline, after `TASK=contra` in the same env prefix,
  keyed it `contra` (bash expands later assignment words with the new value). The result was empty, which the
  runner's `REQUIRE_EVAL_SIZE` caught (rc=5). Lambda hit the same bug on 2026-10-20. `run_cells.sh` now builds
  the override first, and `test_cells.py` checks it against a stub runner.
- **Model-only copies have no `config.json`.** grad_concentration's S3 copies are model-only resaves with
  LoRA merged and no `config.json`. They need `--dense-config`. A checkpoint that still holds LoRA keys without
  a config describing them is refused by the loader (`early_lora`), so a wrong `--dense-config` cannot
  silently score the base model.
