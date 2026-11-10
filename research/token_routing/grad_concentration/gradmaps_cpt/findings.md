# CPT gradient maps: per-token x per-block gradient of the next-token loss (2026-11-10)

The continued-pretraining counterpart of `../gradmaps/` (SFT answer loss). Experiment `grad_composition`, branch
`claude/cpt-gradmaps-34a768`. Record: `records/cpt-gradient-maps.md`.

## Verdict

**After dense CPT the gradient has no sparse support, at the token or the cell level.** Ranked by energy, the
smallest token set whose attribution gradient reaches cos 0.95 with the full step gradient is 70% of tokens (32k) and
74% (64k). The SFT oracle needed 0.47-1.96%. (Attribution = the kept tokens' gradient terms under the unchanged dense
forward.) (block, token) cells give the same 71-76%. Below about 65% kept, the energy-ranked set is *worse than
random*. The step gradient is a coherent sum of many small terms, while the high-energy tokens (high-loss labels)
carry large terms that point elsewhere. The best-case attribution bound for token or cell dropping in CPT is about
1.3x per step, against 5-10x targets (SFT: about 4x at the same split). Attribution sparsity is an upper bound, not a routing result: in SFT, sets that
held 95% of the attribution kept only cos 0.08-0.19 once the other tokens were actually removed.

## Setup

- **Model.** OLMo 3 7B, frozen weights, bf16, one H200 per checkpoint (jsteinhardt / cubbins, `--qos=preemptive_high`).
- **Checkpoints** (all on cubbins `/data/prasann/olmo3_cptdrop/`). The CPT start and the dense CPT run trained at each
  length (LR 2e-5, the olmo3_cptdrop recipe):

  | length | base | dense CPT | YaRN |
  |---|---|---|---|
  | 32k | `base/stage2-step47684` | `runs/o3cd-Dlr20-480` (480 steps x 16 rows) | x4 |
  | 64k | stage2 | `runs/o3cd-D64lr20-480` (480 x 8) | x8 |
  | 128k | stage2 | `runs/o3cd-D128lr20-120` (120 x 8) | x16 |

  Reference at 32k: OLMo 3 stage 3 (`base/stage3-step11921`, YaRN x8). It looks like the base on every statistic
  (as in the 10-24 addendum).
- **Documents.** Longmino dev rows: one window of one long document each, from the 2^17+ buckets, disjoint from
  training.
  - 32k: `dev32k_1024` rows 1, 7, 10, 11, 14, 15, 17, 19 (8 documents: art, finance, health, history, literature,
    politics, science, software development).
  - 64k: `dev64k_512`, the same row indices (8 documents).
  - 128k: `dev128k_512` rows 1, 11, 19 (3 documents).
  - **These are small samples** (⚠ 8 / 8 / 3 documents). Differences between checkpoints on a handful of rows are
    readings, not findings, unless they are large and consistent across lengths.
- **Probe** (`cpt_gradmap.py`): four backward passes per row.
  1. Full loss = mean next-token CE over every target.
  2. Same loss with every attention layer's keys / values detached (`cpt_probe.py`'s hook). A token's gradient then
     comes only from its own label; context part = g_full - g_own.
  3. Loss of the last 1,024 tokens only.
  4. Loss of tokens [T/2, T/2 + 1,024) only.

  Per (block b, token t):
  - E = sum over block b's linears of ||dL/dy_t||^2 ||x_t||^2 (the SFT maps' attribution energy);
  - ||dL/dh_b|| at block b's input;
  - exact per-block parameter-gradient energy.
- **Cost.** 31 s / 70 s / 171 s per row at 32k / 64k / 128k. Peak memory 38 / 61 / 108 GiB: 128k fits on one H200.
- **Oracle budgets** (`cpt_oracle_budget.py`, `cpt_oracle_cells.py`).
  - Each token's rank-1 terms dy_t x_t^T, for every linear, are TensorSketched: blocks 7-31 into 32,768 dims, blocks
    0-6 into 8,192.
  - One step = the CPT runs' own global batch: 16 rows x 32k (dev32k rows 1-4, 6-15, 17, 19) or 8 rows x 64k.
  - Self-check: sketched vs exact gradient norm^2 per row is 0.985-1.021 at every checkpoint.

## Maps (linear color scale, 0 .. each panel's 99th percentile; max in the title)

- `map_<len>_r<row>_E_linclip.png` / `_G_linclip.png`: full-loss maps. A per-token CE strip on top (bin mean); one
  panel per checkpoint. Tokens are binned to 512 columns: 64 / 128 / 256 tokens per bin at 32k / 64k / 128k. E is
  summed per bin, ||dL/dh|| is the bin RMS.
- `span_<len>_r<row>_{E,G}_linclip.png`: loss of the late and mid 1,024-token spans only.
  - Dashed lines mark the span; the white dotted line is 4,096 tokens before it (the reach of the 24 sliding-window
    blocks).
  - Everything left of it reaches the span only through the 8 full-attention blocks (3, 7, ..., 31) plus multi-hop
    paths.
- `ctx_<len>_r<row>.png`: context share per cell, ||g_ctx||^2 / (||g_own||^2 + ||g_ctx||^2), on 0 .. 1.
- Log-scale versions of every map are on cubbins `/data/prasann/grad_composition/cpt_gradmap_figs/32k/` (32k only).
  The default is linear (Prasann, 2026-11-10).

## Findings

1. **CPT flattens the gradient over tokens** (`sum_conc_<len>.png`; share of tokens holding 95% of the full-loss E):

   | | 32k (8 docs) | 64k (8 docs) | 128k (⚠ 3 docs) |
   |---|---|---|---|
   | base | 35% (cells 26%) | 27% (cells 18%) | 2.7% (cells 0.9%) |
   | dense CPT | 54% (cells 44%) | 59% (cells 47%) | 49% (cells 36%) |

   70% of E needs 7% → 22% of tokens at 32k. The base at 128k (YaRN x16, never trained that long) is dominated by a
   few tokens, mostly early copy targets.
2. **Base: one shared direction. CPT: many directions** (oracle table below, and the coherence ratio).
   - Coherence ratio = exact ||sum_t g_t||^2 / sum_t ||g_t||^2 per block. Base: 5-33. After CPT: 1.5-4.
   - So in the base almost every token pushes the same way, probably the adaptation to the new RoPE scaling: at 32k
     a random 5.9% of tokens reaches cos 0.95, the same as the oracle.
   - After CPT the per-token terms mostly cancel, and what survives is spread thinly over most tokens.
3. **Oracle budgets: the CPT version of the SFT oracle table** (attribution cos with the step gradient, scope blocks
   7-31, tokens ranked by energy summed over blocks 7-31; `oracle_table.md`, `sum_oracle_cos_{dn,wm}.png`).
   One step per column; random is one draw.

   | selection rule | 32k base, cos 0.95 | 32k dense CPT, cos 0.95 | 32k dense CPT, cos 0.9 | 64k base, cos 0.95 | 64k dense CPT, cos 0.95 | 64k dense CPT, cos 0.9 |
   |---|---|---|---|---|---|---|
   | Global oracle (tokens) | 5.9% | 69.9% | 65.3% | 12.0% | 74.4% | 70.2% |
   | Per-row quota (= per document window) | 11.5% | 77.6% | 72.7% | 17.2% | 80.8% | 76.9% |
   | Per-512-token-block quota | 18.4% | 82.4% | 77.9% | 25.3% | 83.6% | 79.8% |
   | Random (tokens) | 5.9% | 83.0% | 69.2% | 36.2% | 79.3% | 64.1% |
   | Global oracle (cells) | 5.7% | 71.0% | 65.4% | 11.9% | 75.8% | 70.9% |
   | Per-row quota (cells) | 10.4% | 76.8% | 70.1% | 15.0% | 82.0% | 76.5% |
   | Per-(row, block) token quota (cells) | 9.9% | 74.4% | 69.5% | 16.8% | 77.0% | 72.7% |
   | Random (cells) | 8.6% | 83.0% | 69.1% | 30.3% | 79.7% | 64.7% |

   - SFT, for comparison (7,808 rows/task, cos 0.95): global 0.47%, per row 4.6%, per document 13.7%, random 53.7%.
     CPT "sharpening" runs the other way: base → CPT needs 6-12x more tokens.
   - In the curves, every energy-ranked rule sits at cos 0.13-0.3 from 5% to 40% kept, while random climbs steadily.
     Ranking by energy picks the noisiest terms.
   - Cells do not beat whole tokens, unlike SFT, where cells were 2.5-3x more concentrated. The whole-model scope is
     the same: global 70.4% / 75.0% for tokens, 69.7% / 75.0% for cells.
4. **How far back a span's gradient reaches** (`span_*`, `sum_distance_<len>.png`). The pre-span weight-gradient
   energy of the late 1,024-token span's loss lies within 1k / 4k / 16k tokens of the span as follows:

   | | within 1k | within 4k | within 16k |
   |---|---|---|---|
   | 32k base | 0.28 | 0.61 | 0.81 |
   | 32k CPT | 0.28 | 0.65 | 0.87 |
   | 64k base | 0.13 | 0.41 | 0.69 |
   | 64k CPT | 0.35 | 0.57 | 0.69 |
   | 128k CPT (⚠ 3 docs) | 0.16 | 0.48 | 0.70 |

   - So 35-59% of it lies beyond the 4k sliding window and 13-31% beyond 16k: real long-range credit assignment.
   - It sits in the early blocks. Share of the span loss's gradient energy that lands on tokens before the span
     (CPT 32k): blocks 0-6 0.43, 7-15 0.26, 16-23 0.10, 24-31 0.035 (64k: 0.39 / 0.25 / 0.12 / 0.04).
   - Late blocks carry little pre-span energy at all, and what they carry is slightly more local: within 4k, blocks
     24-31 hold 0.78 vs 0.60-0.70 for the lower bands (CPT 32k).
   - In the maps, the far context lights up mainly in blocks 0-15.
5. **Own label vs context** (`ctx_*`, `sum_ctx_<len>.png`). Median context share of ||dL/dh_b||^2 at blocks
   0 / 8 / 16 / 24:

   | | 0 | 8 | 16 | 24 |
   |---|---|---|---|---|
   | 32k CPT | 0.84 | 0.73 | 0.45 | 0.21 |
   | 64k CPT | 0.90 | 0.85 | 0.47 | 0.21 |
   | 128k CPT | 0.89 | 0.85 | 0.47 | 0.19 |
   | 32k base | 0.95 | 0.87 | 0.82 | 0.68 |

   - After CPT the late blocks learn from their own label and the early blocks from serving as context, at every
     length. The base stays context-dominated deep into the stack.
   - Consistent with the 10-24 addendum (0.95 / 0.80 / 0.42 / 0.21, 48 rows, different normalization).
6. **Block bands** (`sum_band_<len>.png`; exact parameter-gradient energy, share including the LM head):
   - Blocks 0-6: 0.43 → 0.16 (32k), 0.40 → 0.17 (64k), 0.52 → 0.29 (128k).
   - LM head: 0.10 → 0.34, 0.07 → 0.32, 0.05 → 0.26.
   - Stage 3 keeps the base pattern (blocks 0-6 0.56).
   - For the late-span loss alone, blocks 0-6 hold 0.20 / 0.14 / 0.18 after CPT.
7. **Rank agreement across blocks** (`sum_corr_<len>.png`; Spearman of per-token E):
   - Block 7 vs 31: 0.38 → 0.51 (32k), 0.28 → 0.44 (64k), 0.22 → 0.46 (128k).
   - Block 7 vs 16: 0.73 → 0.78. Adjacent blocks: 0.96.
   - More stable than SFT (7 vs 31: 0.33-0.39), but stability does not help when nothing is sparse.
8. **Which tokens carry it** (`sum_kinds_<len>.png`; E share / token share, all blocks, 32k):
   - Base: newline tokens 6.1x, labels that continue a 4-gram seen earlier (copyable) 1.75x, high-loss labels (top
     10%) 3.0x, the first 4 tokens 82x.
   - After CPT: newlines 1.1x, copyable labels 0.5x (copying is learned), high-loss labels 2.3x, first 4 tokens 22x.
   - 64k and 128k agree. After CPT the gradient follows the loss and little else.
9. **What it implies for token dropping in CPT** (`sum_flops_<len>.png`; best-case per-step FLOP speedup, linear-layer
   FLOPs, attention neglected).
   - **Energy view** (95% of E kept by a nested top-k from split block L):
     - 32k: base 2.6x, CPT 1.8x;
     - 64k: base 3.3x, CPT 1.6x;
     - 128k: CPT 2.0x.
     - At L = 8: CPT 1.6x / 1.5x / 1.8x.
     - Cell-level energy bound after CPT: 2.2-2.8x.
   - **Cosine view** (the SFT table's criterion), stricter: about 70% of tokens or cells from block 7 →
     (7 + 0.7 x 25) / 32 = 0.77 of the FLOPs, **about 1.3x**.
   - In SFT, 0.5-2% of tokens from block 7 held cos 0.95, a bound of about 4x per step with blocks 0-6 dense. The
     recipe there keeps 20% → 7%.
   - These are **attribution bounds, not routing results**. Removing tokens from the forward changes the kept tokens'
     own terms; in SFT, 95%-attribution sets gave a routed cos of only 0.08-0.19.
   - The reading for CPT: the per-step gradient is an average of many small, nearly independent per-token terms. So
     dropping tokens acts like a smaller batch (random keep traces a smooth subsampling curve). There is no sparse
     set to route to, and energy-based importance selects the noisiest terms.
   - This is consistent with every token-drop arm losing to dense in the olmo3_cptdrop screen. The levers that won
     there (LoRA, frozen LM head) cut per-parameter cost, not tokens.

## Files

| Files | What |
|---|---|
| `map_<len>_r<row>_{E,G}_linclip.png` | full-loss maps: 32k rows 11 / 14 / 19, 64k and 128k rows 11 / 19 (all rows on /data) |
| `span_<len>_r<row>_{E,G}_linclip.png` | late / mid span maps |
| `ctx_<len>_r<row>.png` | own-label vs context split |
| `sum_{conc,band,distance,corr,kinds,ctx,flops}_<len>.png` | summaries over every row of the set |
| `summary_<len>.json` | the numbers above per checkpoint (eval_size_rows, means and SEs over rows) |
| `oracle_table.md`, `oracle_budgets.json`, `sum_oracle_cos_{dn,wm}.png` | oracle budgets, tokens and cells |

## Code, compute, raw data

- **Code** (`research/token_routing/grad_concentration/`):
  - `cpt_gradmap.py` + `run_cpt_gradmap.sbatch` (probe), `cpt_gradmap_plot.py` + `run_cpt_gradmap_plot.sbatch` (draw);
  - `cpt_oracle_budget.py`, `cpt_oracle_cells.py`, `run_cpt_oracle.sbatch`, `cpt_oracle_table.py`.
- **Jobs** (slurm, cubbins, 1 H200 each):
  - smokes 4430500-1 (mooney);
  - maps 4430508-14, 4430525-29, 4430543-4;
  - plots 4430547 / 4430604 / 4430667;
  - token oracle 4430599-602, cell oracle 4430659-62.
- **Raw arrays** (cubbins, node-local):
  - `/data/prasann/grad_composition/cpt_gradmap/<label>/r<row>.npz`: float32 G, E, G_own, C, G_late, E_late, G_mid,
    E_mid, ce, ids; 30-150 MB per row.
  - `/data/prasann/grad_composition/cpt_gradmap_figs/<len>/`: every figure.
  - `/data/prasann/grad_composition/cpt_oracle/`: oracle JSON / npz.
- **Not done.**
  - The Qwen3.5-4B contrast: its CPT checkpoints are on weka only. The `cpt_u1B` dense runs were never staged
    locally, and the K/V-detach hook would need a Gated DeltaNet equivalent.
  - An oracle ranked by projection onto the step gradient instead of by energy (a fairer upper bound; one extra pass).
