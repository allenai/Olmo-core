# longctx_contra: gold-document drop vs dense LoRA on contradiction at 256k / 1M (Beaker)

Phase 1 (2026-11-18): feasibility smokes, pools, length mix, eval plan. Comparison runs NOT launched (needs Prasann's go).
Experiment `tokdrop_lr`. Branch `claude/longctx-contra-golddrop` (off `claude/elp-lambda-kl` + the `--max-pf` commit).

| file | what |
|---|---|
| `build_lc_pools.py`, `build_lc.sbatch` | contradiction-only pools `lc128k` / `lc256k` / `lc1m` (smoke-sized) and `lc256kL` / `lc1mL` (FLOP-matched gold-drop sizes, `--sizes ... --suffix L`) + held-out `heldout_lc.npz` (sets `lc128k` 64, `lc256k` 160, `lc1m` 64 rows; ctc eval split, eval seed 99). CPU job on mooney. |
| `run_lc_smoke.sh`, `launch_lc_smoke.py` | Beaker feasibility smokes: runs back to back in one job; `probe` mode = through `../dense_probe.py` (per-step s / peak GiB / PF), `direct` = bare trainer (needed for `--cp-degree`). |

Data: `mooney:/data/prasann/early_loss_proxy/shards/lc*` (+ `_stats.json`), `heldout_lc.npz` (+ `_stats.json`); weka copies
under `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/early_loss_proxy/`. Work dirs `mooney:/data/prasann/early_loss_proxy/v2build/lc{,L}/`.
Smoke results: weka/S3 `early_loss_proxy/results/lcsmA|lcsmB/` (`../fetch_beaker_results.sh <tag>`).

Generator facts (ctc `contradiction`, PubMed pool): 60,342 claim pairs (90% train split = 54,308 -> at most 18,102 rows with
distinct gold per length; row i uses pairs 3i..3i+2 at every length) and 184,974 filler sentences from 20,000 abstracts
(shared across rows). 3 gold pairs = 6 gold documents per row; ~45.3 tokens per document.

## Smoke results (2026-11-18/19, Beaker H100 80 GB; jobs lcsmB 01M7S8KNCZVXRQPVQWQJVYM5HV, lcsmA5 01M7SA5TA15SZX7CTXSEN3050F)

| run | setup | result |
|---|---|---|
| dense LoRA 256k, 1 GPU | (memscale, H200 141 GB) | OOM |
| `dl256cp2` | dense LoRA, CP2 x DP4, 8 GPUs | OOM |
| `dl256cp4` | dense LoRA, CP4 x DP2, 8 GPUs, rows ~252k | fits: 25.3 s/step (2 rows) = 101 GPU-s/row; 13.3 PF/row actual (19.1 full-FT eq.); meter counts each row x cp_degree under CP (divide by 4) |
| `dl1mcp4` | dense LoRA 1M, CP4 x DP2 | OOM (qwen3_5 CP max = 4) |
| `fwd1m` | dense 1M forward, no grad, 1 GPU | OOM |
| step-0 probe | dense 256k forward, 1 GPU | fits, CE 1.474 |
| `gdd256` | recipe + gold drop 0.05-0.5 (whole 0.15; no whole row drawn), 1 GPU | fits: 8 steps, 0.53 PF/row (0.040x dense LoRA), ~7 s/row steady, peak 55 GiB; padding trimmed (block widths 64k/16k/8k) |
| `rcw256` / `rcw256pc` | whole 256k row on the recipe, without / with `--prefix-chunk 4096` | OOM / fits (2.79 PF, 24 s, 57 GiB) |
| `gdd1m50pc` | 1M, keep 0.5, prefix-chunk | OOM |
| `gdd1m25` | 1M, keep 0.25, no prefix-chunk | OOM |
| `gdd1mpc` | 1M, keep 0.05-0.5, prefix-chunk | step 1 (keep 0.41, ~430k tokens) 6.4 PF / 51 s, step 2 2.9 PF / 22 s, peak 72 GiB; step 3 OOM (r > ~0.45) |

Required on long rows: `--skip-dry-run` (the trainer's mock batch is a full-length dense pass) and `--st-otr-warmup-random`
(the router's warmup step otherwise trains the full context densely). Pooled soft tokens do not support CP, so the gold-drop
arm runs plain DP. `cost_model.py` / `mix_tables.py`: the FLOP model (calibrated on memscale and checked on these smokes) and the
length-mix tables.
