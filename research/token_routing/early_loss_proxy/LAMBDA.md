# Early-loss proxy on Lambda (2 x A100-80GB per run)

Same runs as `run_proxy.sbatch` (16 rows/step, micro-batch 2, 2 GPUs, seq 16,384, same probe flags), so FLOPs match the
Berkeley numbers. Wall-clock does not (A100 vs H200). Read `lambda_cluster.md` (repo root) for the cluster rules.

```bash
# commit first: the launcher stages `git archive HEAD` (it warns about uncommitted changes)
python3 research/token_routing/early_loss_proxy/launch_proxy_lambda.py --arms recipe --seeds 0 --steps 25,50,100 \
    --max-train-s 600 --tag <tag> [--custom 'name=variant,lr,flags'] [--qos preemptive_high|preemptive|normal]
# -> prints the job id, the slurm log path, registers `elp-<tag>-lambda<jobid>` with ops (experiment tokdrop_lr)

# wait for it WITHOUT a foreground loop: run with run_in_background=true, one ssh per 60 s, bounded
timeout 2700 bash -c 'until ssh lambda "grep -aq \"ELP_DONE\|FATAL\" /accounts/projects/sewonm/prasann/ctc_suite/elp/logs/elp-<tag>-<jobid>.out"; do sleep 60; done'

bash research/token_routing/early_loss_proxy/fetch_lambda_results.sh <tag>     # -> /tmp/claude-3018/elp_lambda_results/<tag>/
python3 research/token_routing/early_loss_proxy/score.py --results /net/mooney/data/prasann/early_loss_proxy/results/v2 \
    /tmp/claude-3018/elp_lambda_results/<tag> --out /tmp/.../board.json
```
`ops status` / `ops stall <jobid>` also cover Lambda. sbatch often reports FAILED on runs that succeeded: trust the
result JSON (`"done": true`), not the slurm state.

## Where things live on Lambda (`L=/accounts/projects/sewonm/prasann/ctc_suite/elp`, NFS, counts toward the 1.30T quota)
- `$L/bases/q35-4b-base-markerfix/model_and_optim` (13 GB, copied from mooney; NOT the old `bases/q35-4b-base-modelonly`).
- `$L/shards/mtb16k`, `$L/heldout_v2.npz` (new pool / held-out rows, copied from mooney).
- `$L/code/code_<sha>` staged code, `$L/results/<tag>/` JSONs + logs, `$L/logs/` slurm logs, `$L/step0_cache.json`
  (Lambda's own step-0 cache; never copy the Berkeley one), `$L/jobs/` arm specs.
- Env: `/accounts/projects/sewonm/prasann/olmo_test/env` (conda-pack; reused). Run/work dirs: node-local `/tmp/$USER/elp`.

## Rules
Lambda is shared: at most 8 GPUs `preemptive_high` plus 8 `preemptive` in total, anything more at `normal`. One job =
2 GPUs; keep at most 4 submitted at `preemptive_high`. No internet, no `/data` on Lambda.

## Measured on A100-80GB (2 GPUs, 2026-11-17, jobs 115205 / 115208)
- Startup (env, model load, step-0 probe when uncached) 220-270 s; later jobs reuse `step0_cache.json`.
- `recipe`: ~10-12 s/step steady (first steps 17-24 s: Triton/JIT), peak 13.2 GiB. `denselora`: ~31 s/step, peak 12.4 GiB
  (fits easily; trainer defaults apply full activation checkpointing + FSDP). Probe of 192 held-out rows: ~55 s.
- recipe, 10 steps: process wall 554 s. Step-0 mean CE 0.8668 (cached for later jobs). No v2 Berkeley result existed to compare.
