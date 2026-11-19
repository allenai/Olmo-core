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
