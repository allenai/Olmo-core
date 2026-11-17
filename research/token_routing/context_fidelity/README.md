# context_fidelity

No-training probe (Prasann, 2026-11-10; experiment `tokdrop_lr`): can cheap stand-ins for the document tokens dropped
after block 7 keep the kept tokens' hidden states, answer CE and gradient dense-like, and at what step cost?
Record: `records/context-fidelity-probe.md`.

* `context_probe.py` — the probe (reuses `../grad_concentration/` rows, router emulation, TensorSketch gradients).
  Variants: drop, frozen, refresh15, kvfresh, mean1, spanN, lin7, lin7r, frozen_all, staleNN, fwdonly.
* `analyze_context.py` — per-step / expected gradient cos, CE, hidden error, step cost, gap closed.
* `tables.py`, `make_figure.py` (`figs/context_fidelity.png`), `record_ctx.py` (`ops result add --check`).
* `run_ctx_local.sbatch` (jsteinhardt, 1 GPU; arguments go through the environment, not `--export=K=V`, because
  they contain commas), `launch_ctx.py` (Beaker jupiter, 1 GPU, urgent, minRuntime 1h).

Results: sketches and per-shard JSON on cubbins `/data/prasann/ctxfid/{sketches,results}/` (labels `f1952`,
`f1952st`, `recipe`, smoke / check runs); analysis JSON copied to `results/` here (gitignored).
