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

## State
See the bottom of this file (updated as jobs land).
