# fast8k 4-task eval on Berkeley slurm (no Beaker queue)

The eval the Beaker `ev-*` jobs of `launch_fast8k.py eval` run, on a jsteinhardt H200 node:
contradiction / nq / outlier / oolong × the 8k and 2k rungs, 500 rows per cell, query position `after`,
no-cot, docchunk ladder, dense emitter, bs=1 greedy decode, the same rung files, prompts and graders. It
calls the same on-node runner Beaker and Lambda use (`recipes/eval/legacy_native/run_beaker_multirung_eval.sh`)
with the arguments `run_q4b_beaker_multirung_eval.py` hands it, so a change to the runner reaches all three.

One slurm job per checkpoint. It runs one runner call per (task, rung) cell, every 8k cell first. Slurm
requeues a preempted job (`--requeue`), and the job skips the cells already finished by the same commit.

## Run it

From the checkout whose code should score the checkpoint (its HEAD commit is used, as gantry uses the pushed
commit):

```bash
# a node-local train_ctc_suite export (config.json + model_and_optim/)
python3 research/token_routing/ds64_fast8k/local_eval/submit_eval_local.py \
    --run <train run> --node <mooney|cubbins|sneetches> --ckpt /data/prasann/ctc_suite/ckpts/<train run>
# a model-only copy on the S3 hub (staged to the node; any H200 node with the env)
python3 research/token_routing/ds64_fast8k/local_eval/submit_eval_local.py \
    --run <train run> --ckpt-s3 checkpoints/prasanns/_transfer/<hub>/<train run> [--dense-config]
```

- `--dry-run` prints the sbatch line and who the start would preempt, and submits nothing.
- The submitter registers the job as `<run>-evslurm-<jobid>` (kind eval, parent run `<run>`, experiment
  `tokdrop_lr`; `--experiment` overrides). The job registers again when it starts, through `recipes/env/ops_job.sh`.
- Default QOS is `--qos preemptive`. It runs outside our 8-GPU `preemptive_high` cap, bumps `normal`
  jobs, and can be bumped by `preemptive_high`. That is fine here because evals are short and requeue-safe.
- Other knobs: `--tasks`, `--rungs`, `--gpus` (2), `--code-ref`, `--fresh` (rescore every cell) and
  `--gen-batch N` (opt-in batched decode, as `run_eval_lambda.sbatch`).
- `--dense-config` is only for a checkpoint with no `config.json`, i.e. a plain or LoRA-merged Qwen3.5-4B
  model-only resave such as `grad_concentration/stage_s3_beaker.sh` makes. The loader still refuses a
  directory that holds LoRA keys.
- A `train_ctc_suite` export needs no flag: its `config.json` records the LoRA / router blocks. Use a code
  ref that has `olmo_core.nn.early_lora` for LoRA runs (the router and oracle branches do;
  `prasann/flopdrop_clean` does not).

## What the job needs on the node (it stages everything itself)

| What | Where on the node | From |
|---|---|---|
| interpreter | `/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python` (cubbins, mooney, sneetches, mcfuzz) | already there |
| job log | `/data/prasann/joblogs/f8kev-<run>-<jobid>.out` (mcfuzz has no `joblogs/`, so it is excluded) | slurm |
| code | `/data/prasann/f8k_eval/code/<sha>/` | `git archive <sha>` from `--src`, once per commit |
| eval rows (8 files, ~100 MB) | `/data/prasann/f8k_eval/eval/<same relative paths as weka>` | `s3://ai2-llm/checkpoints/prasanns/_transfer/f8k_eval/` (md5-checked against its `MANIFEST.json`) |
| tokenizer | `/data/prasann/hf_cache` (`Qwen/Qwen3.5-0.8B-Base`; cubbins, mooney, sneetches have it) | HF |
| checkpoint (`--ckpt-s3`) | `/data/prasann/f8k_eval/ckpts/<run>/` | S3 hub (ETag-checked, `olmo3_cptdrop/s3_sync.py`) |
| results | `/data/prasann/f8k_eval/runs/<run>/eval/<key>_multirung.json` + `*_responses.jsonl` (the Beaker layout); copies in `/data/prasann/f8k_eval/results/` | the job |

The S3 copy of the eval rows came from Lambda's `stage_tdlr/eval` (the copy the Lambda path scores). Both are
copies of the weka files, and the parity run below checks them against Beaker end to end.

## Results

When the cells finish, the job files ops result rows from each cell's log, using the regex and fields
`record_beaker_evals.py` uses for Beaker logs: `train_run`, `task`, `rung`, `metric` (f1, or score for
oolong), `eval_size`, `se`, `cot_label=no-cot`, `eval_job_id=slurm:<id>`. It files them with
`ops result add --check` and skips rows it already filed. So `collect_pareto_points._ops_scores`,
`make_pertask_section.py` and `ops result show --train-run <run>` read local and Beaker evals alike. Run
with `RECORD=0` (`--no-record`) to skip this step.

## Files

- `submit_eval_local.py`: checks, sbatch and `ops register` (stdlib, safe on the login node).
- `run_eval_local.sbatch`: staging, env check, the cell loop under `ops_job.sh`, merging, recording.
- `run_cells.sh`: the per-cell runner calls.
- `cells.py`: plan, merge, rows, S3 data staging (stdlib except boto3 for S3).
- `test_cells.py`: `python3 test_cells.py`.

## Parity with Beaker

See `records/f8k-local-slurm-eval.md` for the reference run, the cell-by-cell table and the timing.
