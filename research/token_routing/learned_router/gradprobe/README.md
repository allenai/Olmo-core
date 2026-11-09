# gradprobe — gradient token importance and lossless keep budgets on frozen CTC checkpoints

Record: [`records/gradprobe-token-importance.md`](../../../../records/gradprobe-token-importance.md). Experiment
state: `ops exp show learned_router`.

| file | what |
|---|---|
| `grad_importance.py` | the probe: per-row gradients of answer CE w.r.t. per-token keep gates (`soft_keep`), hard-drop curves for every rule, gold-first and bar-family baselines, built-in parity checks. `--help` lists the rule flags (`--alphas`, `--ig-steps`, `--iter-k`, `--adapt-grid`, `--fl-sweep`, `--gold-curves`, `--keeps`, `--row-offset`). Checkpoint formats: `hf`, `distcp`, `olmo_st` (one safetensors file from `resave_model_only.py`). |
| `cross_score.py`, `cross_summary.py` | cross-model probe: gradients from one checkpoint (`--score-ckpt`, e.g. the base), hard drop + CE with another (`--ckpt`); base vs self rules, @bar paired points. Summary: median/mean ΔCE curves, ρ*, paired diffs, AUROC. sbatch: set `SCORE_CKPT=`. |
| `run_gradprobe_local.sbatch` | Berkeley slurm, from a code tree staged on node-local `/data` (`git ls-files \| tar` on the login node, `srun tar -x`). Auto-detects the checkpoint format. |
| `run_gradprobe_lambda.sbatch` | Lambda A100, from inputs streamed into the node's `/tmp/prasann/gp` by `stage_to_lambda.sh` (Lambda's `/accounts` quota is usually full). Pin the node. |
| `stage_to_lambda.sh` | stream code + rows + tokenizer + checkpoints from a Berkeley node's `/data` into a Lambda node's `/tmp` (run with `srun --export=NONE ... bash --noprofile --norc`, see traps). |
| `launch_beaker.py`, `beaker_gradprobe.sh` | Beaker path (code from an S3 tarball, never a public push). **GPU jobs currently fail** on the baked image (conflicting `fla`); `--resave` mode works. |
| `resave_model_only.py`, `fetch_s3_ckpt.py` | weka checkpoint → model-only safetensors on S3 (`unshard_checkpoint`, no model build, no fla) → node `/data` via boto3. |
| `grad_similarity.py`, `run_gradsim_local.sbatch`, `gsim_report.py` | full-context vs token-dropped **parameter** gradient of the answer CE (hard drop, grid semantics): per (rule, ρ) global / per-module cosine and norm ratio, with rerun / keep-all / other-row references, plus gate-gradient stability over the kept tokens; `--demean 1` adds cosines with the leave-one-out mean full gradient removed (base models share a large row-independent direction). One model, several tasks per job; `gsim_report.py` prints tables and writes `gsim_*.png`. Record: [`records/gradprobe-gradient-similarity.md`](../../../../records/gradprobe-gradient-similarity.md). |
| `grad_tokens.py`, `gtok_report.py` | exact per-token / per-layer decomposition of that gradient at each block's `attention.w_out` and `feed_forward.w2` (dCE/dW = Σ_t δ_t x_tᵀ): dropped tokens' share D, kept-token drift K→K′, forward vs backward split, token-category mass, distance-to-drop profile, decoded top tokens. Run via `SCRIPT=grad_tokens.py` in `run_gradsim_local.sbatch`; `PROBE_FAMILY=qwen3` runs Qwen3-4B with the nothink tokenizer copy (`/data/prasann/hf_tokenizers/Qwen3-4B-nothink` on mooney). |
| `compare.py`, `plot_keep_curves.py`, `plot_min_budget.py` | tables (paired diffs vs a reference rule), ΔCE-vs-keep-ratio plots (by rule / length / gate value), smallest-lossless-budget vs length. |

```bash
CODE=/data/prasann/gradprobe/<staged tree> TASK=nq RUNG=8k ROWS=16 OUT=/data/prasann/gradprobe/out/x.json \
  EXTRA="--ig-steps= --alphas 0.5 --iter-fine 0 --iter-coarse 0 --doc-level 0" \
  sbatch --nodelist=horton --gres=gpu:H200:1 research/token_routing/learned_router/gradprobe/run_gradprobe_local.sbatch
```

**Inputs (not in git).** Rows: `/data/prasann/devloss_grid/data/<subset>/rung_<tokens>.jsonl` (16k/64k rows were
fetched from the public HF dataset `PrasannSinghal/ctc-suite-eval`). Checkpoints: `/data/prasann/devloss_grid/ckpts/`
on horton and sneetches (`ctc-4b-<task>-full`, `setA32k`, `setAbase`); setA copies also on S3
`s3://ai2-llm/checkpoints/prasanns/_transfer/gradprobe/ckpts/`.

**Results (not in git).** `results/json/` (all JSONs used by the record, incl. `firm/` = the 16-row 16k/32k
runs), `results/*.png`.

**Traps.** 32k needs per-block activation checkpointing (on automatically above 12k tokens, ~55 GB); 64k OOMs
(dense T×T soft-keep attention bias). Isolated per-row ΔCE spikes: summarize on the median curve, τ = 0.1.
`EXTRA` is expanded unquoted by the sbatch wrappers: write an empty value as `--ig-steps=`, not `''`; on Lambda
pass `EXTRA` through the environment, not `sbatch --export` (commas split). `srun bash -c` can inherit shell hooks
that start `conda` and hang on NFS: use `--export=NONE` and `bash --noprofile --norc`.
