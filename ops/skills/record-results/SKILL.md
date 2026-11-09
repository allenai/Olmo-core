---
name: record-results
description: >
  Use this skill whenever eval or training numbers come in and before ANY number is quoted,
  compared or written down — "pull the results", "did the evals land", "what did <run> score",
  "add these numbers", "update the table", "is X better than Y". Validates the outputs
  (parse rate, empty generations, eval_size, cot label, grader), records them with
  `ops result add --check` (results-hub is RETIRED as of 2026-10-12) and updates the
  experiment's STATUS.md. Replaces the results-hub ingest step of pull-evals.
---

# Record results

`OPS=/accounts/projects/berkeleynlp/prasann/bin/ops`. Nothing is quoted until it passes steps 1–3.

## 1. Find the outputs
**Each eval pass is its own run** (`kind=eval`, `parent_run` = the training run). Result rows record
under the EVAL run's `run_id`; `train_run` (the training run scored) is filled from the registry's
`parent_run`, so all passes over one model line up under it in `ops result show`.
Match every result JSON to a registered run (`$OPS runs --experiment E`, `$OPS status`). A result with
no run is an unrecorded launch: register it (`$OPS register ... --notes "registered after the fact"`)
before recording. A finished job with no result JSON needs a reason (failed, preempted, still writing).

## 2. Validate (read the raw generations, not only the score)
- **Parse rate** and **empty generations**: an exact 0.0000, or a cell of empty outputs, is a bug
  signature (newline stop with an open `<think>`, maxlen below the prompt, wrong chat template), not a result.
- **eval_size** ≥ 500, or flag it inline: `f1=0.83 ⚠ eval_size=100 (±0.038)`. Never call it `n`
  (n = corpus size). SE = sqrt(p(1-p)/eval_size).
- **cot label**: long-cot / short-cot / no-cot.
- **Grader and data are in-domain**: the eval file comes from the same generator and mode as training
  (task name ≠ file: xabsence EXACT vs PARAPHRASE, contradiction realistic vs both); gold indexed
  by its source file, not by position; rung labels are not token counts.
- **Checkpoint and commit** are the ones you think (`git_commit` in the result JSON).
- CE results: compare to the chance floor `ln(C(n,k))/T` and confirm with a generation metric.

## 3. Record
```bash
$OPS result add --json RESULT.json --check          # or the explicit fields:
$OPS result add --run-id EVAL_RUN --train-run TRAIN_RUN --experiment E --task T --rung rung_32768 --metric f1 \
  --value 0.824 --eval-size 500 --cot-label no-cot --eval-backend olmo-eval-vllm \
  --ckpt-path CKPT --git-commit SHA --eval-job-id JOB --check
$OPS result check ...                                 # dry run: warnings only
```
Investigate **every** `--check` warning (a jump against prior rows, a duplicate, missing provenance,
eval_size<500) before you quote the number. If the warning is real, say so next to the number.

## 4. Update state
- `$OPS update EVAL_RUN final_state=done` for the finished eval jobs.
- Edit `ops/experiments/E/STATUS.md` → **Current best numbers** (arm, task/rung, metric, eval_size,
  ±SE, cot, source path), bump `updated:`, revise `next_step`, then `$OPS exp index`.
- Update the experiment's summary artifact (the URL in STATUS.md `artifacts:`; create one if the
  list is empty) with the new numbers: republish to the same URL, never a second page.
- To read numbers back: `$OPS result show --experiment E [--task T] [--train-run T] [--best]` (grouped by
  training run, one line per pass; `--best` names the best pass per task/rung).
- For native multirung sweeps, the `pull-evals` skill does the job-by-job cross-check and validation, then hands off to this skill for recording.
