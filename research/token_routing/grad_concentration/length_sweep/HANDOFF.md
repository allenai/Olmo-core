# HANDOFF: gradient sparsity vs context length

Branch `claude/grad-sparsity-vs-length` (cut from `claude/merge-side-branches-1117`), worktree
`.claude/worktrees/grad-sparsity-vs-length`. Experiment `grad_composition`. Record: `records/grad-sparsity-vs-length.md`.

## Pipeline
1. Rows: `gen_rows.py` (`gen_rows.sbatch`, CPU, mooney): ctc seed 20261117, 4 tasks, rungs 2k-1m, tokenized query
   `after` and `both` -> mooney `/data/prasann/gradlen/tok/{after,both}/<task>_<rung>/`; S3 `_transfer/gradlen/tok/`;
   weka `prasanns/gradlen/tok/` (`stage_rows_weka.sh`).
2. Checkpoints on mooney `/data/prasann/gradconc/ckpts/`: `f8k-mt4-dense-f7808-tde-lr5e-06` (original table) and
   `m64-dense-f1-lr5e-06` (mt11 dense f1, staged 11-17 via `stage_s3_beaker.sh`, Beaker 01M7NZBRDSC2D26QFTSWQVY135).
3. Probe: `sweep_probe.py` (one run = checkpoint x length; per-step npz, resumable); slurm `run_sweep.sbatch`, Beaker
   `beaker_bench_launch.py --script .../sweep_probe.py`.
4. `analyze_sweep.py` -> summary JSON; `plot_sweep.py` -> PNG.

## State (2026-11-17 22:30, all jobs finished; nothing running)
- DONE: smoke (reproduces the published 8k table), mt11 dense f1 at 2k-256k (512 rows per length to 64k, 192 at
  128k, 80 at 256k), mt4 f7808 at 2k-64k (256 rows each), memory probes (1 / 2 / 8 H200), CE scans 256k-512k.
- Results: `data/` (summaries, memprobes, CE scans, ops rows), `grad_sparsity_vs_length{,_cos0.9}.png`,
  `records/grad-sparsity-vs-length.md`, `experiment_log_entry.json`; `ops result` rows `gsl-m11f1-<len>` /
  `gsl-f7808-<len>` (experiment grad_composition; the mt11 block was accidentally appended twice and the duplicate
  block removed under the ops lock).
- Verdict: per-row quota share for cos 0.95 rises slowly with length (slope +0.19 [0.15, 0.23]; tokens ~T^1.19); no
  1/T. See the record.
- Blocked: no valid point past 256k. 8 H200s hold 524,288 tokens, but the dense forward breaks past ~262k (CE 2-4 at
  306-350k, NaN from 368k, first non-finite in Gated DeltaNet / block 0), with or without YaRN; above 524,288 a
  backward kernel hits an illegal memory access. Fixing the GDN kernel path (probably 32-bit indexing in the FLA
  chunk kernels) is the prerequisite for a 512k point (`run_sweep8.sbatch` with TASK_RUNGS="contradiction=448k
  outlier=448k" and RUNG=512k is ready: rows 433-519k on S3).
- Not done: artifact (main agent), a CE-matched comparison across lengths beyond the row-level regression, model
  seeds.

## 2026-11-18: base-model arm (done, nothing running)
- Checkpoint q35-4b-base-markerfix: the S3 model-only copy `_transfer/gradconc/q35-4b-base-markerfix` (the weka base
  mt11 trained from) -> mooney `/data/prasann/gradconc/ckpts/q35-4b-base-markerfix`. A second, older copy exists at
  mooney `/data/prasann/ctc_suite/bases/q35-4b-base-markerfix` (13 GB, Aug 27; not used).
- Runs: `base-<len>` (2k-256k, query both, same eval sizes as mt11) + `base-8k-after` (layout check); slurm
  5202812-5202828 on mooney, `--qos=preemptive`.
- Outputs: `data/sweep_base_q35_4b_markerfix.json`, `data/base_8k_query_layout.json`, `data/results_base.jsonl` (ops
  rows gsl-base-*; exact-0.0 warnings are real), `grad_sparsity_rowlevel.png`, base lines in the main plot.
- Result: the rise does not survive inside the context on the base (row slope -0.49); see the record.

## Traps hit
- `sbatch --export=ALL,X=a,b` splits on commas: export comma lists in the environment, then `--export=ALL`.
- Long time limits showed `ReqNodeNotAvail` on mooney for a while (short ones started); it cleared by itself.
- The results dir name `results/` is gitignored repo-wide (`research/**/results*/`): committed outputs live in `data/`.
- On the base model the CE term in the row-level regression is a task effect (CE separates tasks), not a
  difficulty control; use the no-CE row slope and per-task slopes there.
- `--memprobe` tiles a short row: memory and speed only, its CE means nothing; use `--ce-only` on real rows for
  numerics.
