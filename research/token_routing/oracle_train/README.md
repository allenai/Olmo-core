# oracle_train — oracle token selection as a gradient upper bound for token dropping

Experiment `tokdrop_lr` (mechanism track). Record: `records/oracle-selection-training.md`.

Each optimizer step (code: `src/olmo_core/nn/oracle_token_select.py`, flag `--oracle-scope` in
`recipes/train/research_trainer/train_ctc_suite.py`, router mode `--st-otr-score oracle`):

1. **Dense pass A** (forward + backward, not applied): per-token LoRA-gradient energy
   `s^2 (|delta|^2 |Ax|^2 + |B^T delta|^2 |x|^2)` summed over the adapted linears of blocks [7, 16) and [16, 32).
2. **Dense pass B**: every token's exact rank-1 LoRA gradient term is added into the bin of its energy rank
   (grid `KS` of body-token shares, per allocation scope: per document / per row / whole batch). Prefix sums give
   the exact attribution gradient of each top-k set, so cos(G_S(k), G_dense) is known on the whole grid.
   Per stage the smallest grid k with cos >= 0.95 is chosen (stage 1 = its own top-k ∪ stage 2: nested).
3. **Routed pass C** with exactly those tokens kept at the block-7 and block-16 cuts (non-body tokens always
   kept), the recipe's drop structure. This gradient is applied. Optional: + uniformly random extra body tokens,
   `--oracle-random-mult` x the oracle set per row and stage.

Base setup = the recipe `lora64all32r20to7` (whole-model LoRA r64, LR 1e-5, 16 rows/step, mt4 fresh pool).

| file | what |
|---|---|
| `launch_oracle.py` | Beaker launch (`launch`) and 4-task 8k+2k evals (`eval`); arms `orcdoc`, `orcrow`, `orcbatch`, `orcbatchr2`, `orcdocr2`; state in `state.json` |
| `run_oracle_local.sbatch` | jsteinhardt smoke / short runs (base + shard already on the node's /data) |

**Metadata** (per run, in the save folder `.../ctc_suite/ckpts/<run>/oracle_meta/`):
- `steps.jsonl` (rank 0, one line per step): chosen k and attribution cos per stage, exact cos /
  projection / relative-norm curves on the KS grid for every scope, kept fractions per stage (overall, oracle
  vs random part), gold vs distractor keep fractions and energy shares, per-document keep-fraction stats,
  dense vs routed CE (batch and per row), answer tokens per row, cos(dense, routed) per parameter group raw and
  Adam-preconditioned (`grad`), grad norms, FLOPs (dense pass, routed pass, actual 3-pass), timings, and the
  bins-vs-true-gradient check (should be ~1.0).
- `sel/step<NNNNN>_rank<R>.npz`: per micro-batch: dataset `index`, `real_len`, chunk ids, packed body / gold /
  oracle / random masks per stage, fp16 energies per stage — enough to replay every selection.
- wandb `oracle/*` scalars (group `f8k-oracle`).

FLOP meter (`flop_meter/*`) sees the routed pass only; the actual cost is ~2 dense passes + routed.
