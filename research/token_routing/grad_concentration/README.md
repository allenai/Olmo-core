# grad_concentration

How many tokens carry the summed (minibatch) parameter update of dense SFT, and does that change over training?
(Prasann, 2026-10-29; experiment `tokdrop_lr`.) Follows `research/token_routing/grad_composition/` (branch
`claude/grad-composition-6f09c7`), which measured per-row hidden-state gradient shares.

* `probe.py`: frozen Qwen3.5-4B checkpoint, 128 unseen fresh-pool rows per task (contradiction, nq, outlier,
  oolong; rows ≥ 7,808 of `<task>_fresh`, md5-checked against `mt4f7808` and `mt4s1952`). Hooks every block
  `nn.Linear`; each token's term of dL/dW is rank 1 (δ_t x_tᵀ, energy ‖δ_t‖²‖x_t‖²). Two passes per 128-row batch:
  energies → oracle rankings (batch-global), then TensorSketch accumulators per rank bin → cos / relative norm of the
  top-k update vs the full batch update, for batch scopes of 16 rows (one optimizer step) and 128 rows (8 steps).
  Two linears also get exact fp32 accumulators to validate the sketch.
* `analyze.py`: k for cos 0.7 / 0.9 / 0.95 and for 70 / 90 / 95% energy, per stage, view (all tokens / body only),
  scope and depth; recipe-shape fidelity; single-split FLOP bound.
* `launch.py`: Beaker (1 GPU, jupiter, urgent, minRuntime 1h). `stage_s3_beaker.sh` + `run_local.sbatch`: the
  jsteinhardt path when Beaker is backed up (weka → S3 → node `/data/prasann/gradconc`).

Results: node-local `/data/prasann/gradconc/results/<label>/` (sneetches) or Beaker `/results`; copied to `results/`
here (gitignored) for `analyze.py`. Writeup: `records/grad-concentration.md`.
