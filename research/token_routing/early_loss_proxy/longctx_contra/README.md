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

## Arms (launched 2026-11-19; `launch_arms.sh`, scoring `launch_arms.sh score <arm> <steps>`)

| arm | Beaker | pool | setup | budget / stop |
|---|---|---|---|---|
| dense LoRA 256k (`dense256b`) | 01M7SJP9PY8W3H8B2JM7SXW87S | `lc256kmix_dense`: 1,091 x 256k + 1,200 each 2k/4k/8k/16k (first rows of the T pools) | CP4 x DP2, `--pack` (doc lengths), 2 windows/step, ~615 steps | 15,206 PF |
| gold drop 256k (`gdd256`) | 01M7SHSBVSZ7HR6KQZ83V6HYW0 | `lc256kmix_gdd` (ordered, bucketed 8-row steps): 18,000 + 1,980 reused-gold 256k rows (tail, from row 83,008) + 71,993 short | DP8, keep 0.05-0.5, whole 0.05, prefix-chunk 4096, `--no-data-shuffle`, 11,496 steps | 15,206 PF |
| dense LoRA 1M (`dense1m`) | 01M7SQ9K76E66V5R1KV24B68AB | `lc1mdense84`: the first 84 1M rows of `lc1mmix_gdd` in its order | CP8, `--gdn-int32-safe --ffn-chunk-tokens 32768 --shard-degree 1`, 1 row/step, 84 steps (168.5 s/row) | 84 rows = ~31.5 GPU-h (~13,600 PF) |
| gold drop 1M (`gdd1m`) | 01M7SKA30E465Z6AZQTDZEJEEA | `lc1mmix_gdd` (ordered, bucketed): 4,971 x 1M + 71,993 short | DP8, keep 0.05-0.4, whole 0, prefix-chunk 4096, `--gdn-int32-safe`, 9,620 steps | max 17,000 PF; read at dense1m's PF |

Checkpoints: weka `.../early_loss_proxy/runs_lc/<arm>/stepN`. Scoring = checkpoint-then-score in the dense-LoRA model class:
256k / 16k held-out on the 1-GPU logits path (both 256k arms), 1M held-out (64 rows) under CP4 with the z-loss-corrected
labels path (matches the 1-GPU path to -0.0035 mean on 256k rows). Live logs: S3 `early_loss_proxy/results/lcA-<arm>/<run>.live.log`.

Findings on the way (details in the smoke table and commits): the trainer's `--pack` emitted no document lengths
(cross-example attention, one FLOP charge per window) -> fixed; the FLOP meter counted every CP rank -> fixed; the
fused-linear CE (liger) always adds 1e-4 * lse^2 (z-loss) to the training loss and to the labels-path CE (+0.05 per token)
-> probe path fixed, training unchanged; Beaker node jupiter-cs-aus-152 fails container start (`--exclude-host`).
