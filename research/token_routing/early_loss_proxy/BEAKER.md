# early_loss_proxy on Beaker

The same proxy as `run_proxy.sbatch` (one torchrun per arm, arms back to back, 16 rows/step, micro-batch 2, 2 GPUs,
sequence 16,384, same `dense_probe.py` flags, step-0 cache), as ONE 2-GPU H100 Beaker job per call
(jupiter or ceres, any-of; priority `urgent`; minRuntime = max(30 min, 1.3 x the estimate), at most 4 h; workspace
`ai2/flex2`, budget `ai2/oe-other`). Registered with `ops` automatically (`elp-<tag>-bk<last 8 of id>`, experiment
`tokdrop_lr`).

## Paths (weka; nothing to stage for the default pool)

| what | weka |
|---|---|
| base | `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim` |
| pool `mtb16k` (+ gold sidecars) | `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/early_loss_proxy/shards/mtb16k` |
| held-out | `.../prasanns/early_loss_proxy/heldout_v2.npz` (`heldout_v1.npz` beside it) |
| old 8k pool | `.../prasanns/ds64/fast8k/shards/mt4f1952` |
| results | `.../prasanns/early_loss_proxy/results/<tag>/` + mirror `s3://ai2-llm/checkpoints/prasanns/early_loss_proxy/results/<tag>/` |
| step-0 cache | `.../prasanns/early_loss_proxy/step0_cache.json` (separate from mooney's: keys include the held-out path) |

## Launch, wait, fetch

```bash
# 1. gantry runs the PUSHED commit of your current branch: commit, then push YOUR branch (never main / prasann/flopdrop_clean)
git push -u origin <your-branch>
# 2. launch (same args as launch_proxy.py minus --code/--node/--qos; --dry-run prints the job). ~25-40 s on the login node.
python3 research/token_routing/early_loss_proxy/launch_proxy_beaker.py --tag <tag> --seeds 0 \
    --steps 25,50,75,100 --max-train-s 600 --arms myarm --custom 'myarm=softtoken,1e-5,<trainer flags>'
#    -> "[elp-beaker] SUBMITTED id=01M..."
# 3. wait without a foreground loop: `ops status` (one snapshot), or a bounded background wait on the S3 mirror:
timeout 3600 bash -c 'until bash research/token_routing/early_loss_proxy/fetch_beaker_results.sh <tag> | grep -q "done: elp-<tag>-myarm-sd0"; do sleep 60; done'   # run_in_background=true
#    live log: beaker experiment logs <id> | grep -a "dense-probe\|=== \|result\|FATAL"
# 4. fetch (small JSONs + logs) and score
bash research/token_routing/early_loss_proxy/fetch_beaker_results.sh <tag>      # -> /tmp/claude-3018/elp_beaker_results/<tag>/
python3 research/token_routing/early_loss_proxy/score.py --results /net/mooney/data/prasann/early_loss_proxy/results/v2 \
    /tmp/claude-3018/elp_beaker_results/<tag> --out /tmp/.../board.json
```

- Cancel: `beaker experiment get <id> --format json` (`jobs[-1].id`), then `beaker job cancel <job-id>`.
- Arms whose flags name a Berkeley `/data/...` path (`ddp`, `ddpg`) are refused: stage the file with
  `stage_to_weka.sh` and pass a `--custom` arm with the weka path.
- New inputs: `bash stage_to_weka.sh <node> <rel>...` (rel under `/data/prasann/early_loss_proxy`): srun on that
  node uploads to S3, then a CPU Beaker job syncs S3 -> weka (needs your branch pushed; it runs `s3sync.py` from it).
- Larger FLOP-matched training runs (`recipes/train/research_trainer/beaker_ctc_suite.py`) can use the same pool via
  `--data-root /weka/.../early_loss_proxy/shards/mtb16k` and `--base-checkpoint` above.

## Measured (2026-11-17, 2 x H100 on jupiter, pool mtb16k, held-out v2 = 184 rows: 12 sets x 16, suite16k/reorder has 8)

| | recipe (`bksmoke4`, 10 steps, probes 5,10) | denselora (`bksmoke5`, 2 steps) |
|---|---|---|
| submit -> running | 20 s | 29 s |
| process start -> base loaded (`t_pre_train`) | 55 s | 53 s |
| step-0 probe (full, 184 rows; later runs use the cache) | 71 s | 70 s |
| later probes | 24.8 s | 24.8 s |
| train s/step | 13.7, 19.2, then 5.4-10.8 (steps 3-10) | 15.2, 14.5 |
| peak GiB (max over ranks) | 18.2-18.7 | 17.6 |
| flop meter after last step | 10.78 PF actual / 39.99 PF dense-equivalent (0.27x) | 5.74 / 7.75 PF |
| process wall | 386 s | 295 s |

- Queueing varies: the very first job of the session pended 30.5 min on jupiter (+7 min image pull/start) while
  later ones started in 20-30 s. Gantry setup itself is ~6 s. Check `ops status`; a job pending > 15 min with
  minRuntime set is just the queue.
- Step-0 CE on the OLD pool / held-out v1 (suite8k, 8 rows/task, 12,288) vs the Berkeley H200 `results/v1` rows: task
  means agree within 0.003 for contradiction / nq / outlier and differ by 0.024 for oolong (per-row |diff| up to 0.096
  there, signs mixed). Different GPU and kernels; keep Beaker and Berkeley runs in separate comparisons when a
  difference is that small. (No Berkeley step-0 on held-out v2 existed yet to compare.)
