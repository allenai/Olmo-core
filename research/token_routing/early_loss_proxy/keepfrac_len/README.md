# keepfrac_len: does the recipe's best keep fraction fall with context length? (+ gold-aware keep at 32k/64k)

Record: [`records/keep-fraction-vs-length.md`](../../../../records/keep-fraction-vs-length.md) (experiment `tokdrop_lr`).
Reuses the early dense-loss proxy (`..`, README there): each arm trains with its own forward, the DENSE forward is
scored on held-out rows by teacher-forced answer CE, and the score is FLOPs to a matched held-out CE.

| file | what |
|---|---|
| `build_kl_pools.py`, `build_kl.sbatch` | single-length pools `kl16k` / `kl32k` / `kl64k` (contradiction, outlier, qdmatch_nq; 1,200 rows each) + `heldout_kl.npz` (sets `kl16k`/`kl32k`/`kl64k`, 48 rows per task); CPU job on mooney |
| `launch_kl.py` | the arms (dense-LoRA reference, router keep 40/20/10/5/2.5/1.25% with the second cut scaled with the first, one-cut-at-a-time arms, `grJ`, `grJc`) -> one 2-GPU Beaker job each via `../launch_proxy_beaker.py` |
| `analyze_kl.py` | FLOPs to matched held-out CE vs the reference and vs the 20%/7% recipe, per task, worst task, peak memory |
| `kl_results.json`, `keepfrac_vs_length.png` | scored results and the chart |
| `experiment_log_entry.json` | entry for the hub's experiment log |

Data: `mooney:/data/prasann/early_loss_proxy/shards/kl{16k,32k,64k}` (+ `_stats.json`, `_order_seed0.json`),
`heldout_kl.npz` (+ `_stats.json`); weka copies under `.../prasanns/early_loss_proxy/`. Build work dir:
`mooney:/data/prasann/early_loss_proxy/v2build/kl/`. Raw run JSONs and logs: weka `.../early_loss_proxy/results/kl*/`,
mirrored to S3 (`../fetch_beaker_results.sh kl32k`).

`grJc` (experiment 2's variant) = `grJ` with the document drop applied to contradiction rows only
(`--st-gdd-only-fps <pool>/gdd_only_contradiction.json`, new flag in `src/olmo_core/nn/gold_doc_drop.py`); outlier and
qdmatch_nq rows run the plain recipe.
