# grad_composition — per-token, per-block gradient / activation composition over CTC fine-tuning

Experiment page: `ops exp show grad_composition`. Record: [`records/grad-composition.md`](../../../records/grad-composition.md).

| file | what |
|---|---|
| `probe.py` | GPU probe on one frozen checkpoint: full forward + backward of the answer CE per held-out row, block split into mixer / FFN halves; per token × block gradient norms, residual updates, first-order skip attributions, output-projection parameter-gradient shares, attention mass at the full-attention blocks; per-row statistics + a mid-stack keep-10% ablation. |
| `launch.py` | one Beaker job (1 H100, jupiter, urgent, minRuntime 1h) per checkpoint through `research/ffn_compute/flop_scaling/beaker_bench_launch.py`. |
| `analyze.py` | aggregates the per-checkpoint `*_stats.npz` + example rows into `results/summary.json` and the figures in `results/fig/`. |

Rows: `mt11s960` rows 720-751 of each task (held out from every mt11 run up to s720; mt11 rows are disjoint from every
mt4 shard, and the probe logs which training shards contain any probe row). Full per-row arrays (float16 npz, ~5 MB
per row) are on weka under `prasanns/grad_composition/dumps/<label>/`; results (not in git) in `results/`.
