# Gradient maps: per-token x per-layer gradient on frozen checkpoints (2026-11-04)

Prasann asked for heatmaps of how each token's gradient is spread over layers, read by eye, plus what they say about
cell-level (token x layer) dropping. Experiment `tokdrop_lr`, branch `claude/grad-support-tokens`.

## What is plotted

- Two quantities per (block l, token t), from one dense forward + backward of the summed answer CE (no-cot):
  - **E** = the token's weight-gradient energy in block l, summed over the block's linears: ||dL/dy_t||^2 ||x_t||^2.
    This is the rank-1 attribution energy the 0.95 oracle ranks tokens by (down7 = E summed over blocks 7-31).
  - **A** = ||dL/dh_l[t]||, the activation-gradient norm at block l's input.
  - Both are shown on a log10 color scale.
- Checkpoints: base (`q35-4b-base-markerfix`), dense 1,952 and 7,808 rows/task (LR 5e-6), and the recipe
  (`lora64all32r20to7`, 1,952 rows/task).
- Rows:
  - **8k**: the 512 unseen fresh-pool rows of the oracle runs (128 per task; md5-checked against training).
  - **2k**: 128 newly generated rows (32 per task), built with the same generator and tokenizer call at `--rungs 2k`
    (seed 9917, `build_2k_rows.sbatch`). These are 2k-rung rows the models never saw; the 8k training pools have no 2k
    rows.
  - Maps use rows picked by the dense 1,952 checkpoint's answer CE per token:
    - 2k: the best- and worst-solved row of each task (on the 2k set the "unsolved" rows are mostly solved by
      f7808; only oolong stays unsolved);
    - 8k: one median-CE row from the longest quarter of each task.
- Overlays:
  - Oracle 0.95 sets at block 7, as tick strips: global (per 16-row batch) / per row / per document.
    - Budgets come from `records/oracle-tokens-under-routing.md`: 2.0 / 9.0 / 19.5% at 1,952 rows/task and
      0.6 / 5.6 / 15.9% at 7,808.
    - Base and recipe have no budgets of their own and use the 1,952 rows/task numbers.
  - The recipe's router keep set at blocks 7 (20%) and 16 (7%).
  - Cut lines at blocks 7 / 16.
  - Segment bar under each map: instruction / distractor doc / gold doc / query / answer, with document boundaries.
  - Gold documents are the ones whose header number appears in the answer (oolong has none: counting task).
- The router is the ridge emulation from `routed_probe.py` (`rtr`), not the trained router: the recipe checkpoint does
  not save its online router.
  - The emulation is fit 2-fold within each 16-row batch, with 10% exploration and per-document allocation; stage 2
    is scored on the routed h16.
  - Its Spearman with the target is 0.52-0.59 at block 7.

## Findings (8k: eval_size 512 rows, 128 per task; 2k: ⚠ eval_size 128 rows, 32 per task)

1. **Where the mass sits by layer** (`sum_layer_8k.png`).
   - The base model's weight-gradient energy peaks at block 0; blocks 0-6 hold 40-48% of it.
   - Training moves it to the top. Block 31 alone holds 18-48% at dense 7,808 (3-38% at 1,952; oolong highest), and
     blocks 16-31 hold 43-71% at dense 7,808 vs 15-22% at base.
   - On document tokens both E and A fall steadily with depth (||dL/dh|| by 3-4 orders of magnitude from layer 0
     to 31). On answer tokens ||dL/dh|| is flat across layers.
   - In trained checkpoints, distractor-document energy drops by a further 1-2 orders of magnitude at about layers
     17-20: a visible dark band in every 2k map (e.g. `map_nq_2k-unsolved_E.png`). The base model has no such step.
2. **Answer and query tokens dominate.** They are 2-10% of tokens but carry 60-95% of the blocks 7-31 energy on
   trained checkpoints (base 73-99%; the one exception is outlier at dense 1,952, 30%). In the maps they are the
   brightest columns at every layer (`zoom_*.png`).
3. **Gold vs distractor** (`sum_type_8k.png`, right panel). Per document token, gold documents carry far more
   blocks 7-31 energy than distractors:
   - nq: 45-55x at the dense checkpoints, 29x at the recipe, 5x at base;
   - contradiction: 8-29x;
   - outlier: 2.4-3.8x (5.3x at base).

   In the maps a gold document is a bright full-height column, while distractors fade above layer ~18. The
   per-row and global oracle sets fall almost entirely inside the gold documents (`map_nq_2k-unsolved_E.png`,
   `map_contradiction_2k-unsolved_A.png`). The router's per-document keep set spreads over every document by
   design (doc_share 1).
4. **Base → 1,952 → 7,808 rows/task.**
   - Gradients shrink as rows get solved: document-token ||dL/dh|| at layer 7 is about 2 orders of magnitude below
     base at 1,952 and about 3 at 7,808.
   - Mass moves to the top layers (the block-31 spike grows) and onto the answer: the answer share at 7,808 is
     0.46-0.77 vs 0.15-0.39 at base.
   - Gold concentration rises from base to 1,952, then partly falls back at 7,808 (contradiction 17x → 8x), because
     the solved model's remaining gradient is mostly on the answer itself.
5. **What the routed forward does** (recipe; `routed_*.png`, `sum_routed_8k.png`).
   - Routed answer CE per token is close to dense (median 0.007 / 0.014 / 0.067 / 0.149 vs 0.006 / 0.009 / 0.073 /
     0.140 for contradiction / nq / outlier / oolong).
   - The layer profile and the split of energy by token type barely move.
   - What changes: the kept document tokens' ||dL/dh|| rises 1.6-4x above block 7 (median log10 +0.2 to +0.6), and
     1.3-1.5x in blocks 0-6 for contradiction and nq. Answer tokens are unchanged.
   - Normalized to their own totals, the routed maps in the example rows shift weight toward some kept documents
     (often gold) in blocks 0-6, and away from the instruction and others.
   - **The "270x" on one nq row (checked against the gradient-gap probe; it is not a bug):**
     - The routed pass is the probe's. Dense CE matches the probe exactly on all 16 shared rows, and routed CE tracks
       its `rtr_recipe` row by row (e.g. 0.115 vs 0.143 summed) at the same keep fractions.
     - The 270x is a sum of squared per-token terms (no cancellation between tokens) for one row the dense model had
       nearly solved (summed CE 0.013 → 0.115 routed). Its dense energy total is about 4 vs about 1e4 for a typical
       row.
     - Over all 512 rows, the per-row ratio has median 1.14; 17% of rows are above 10x and 7% below 0.1x. The large
       ratios are almost all rows with dense CE/token < 0.01 (`sum_routed_8k.png`, right panel).
     - Per 16-row batch, the energy ratio has median 1.97 (IQR 1.0-3.5), i.e. about 1.4x in norm units. That is
       still a per-step noise measure. The probe's 0.89 is ||E[R]|| / ||E[D]|| over 32 steps, where these noisy
       terms cancel.
6. **Cell-level vs token-level dropping** (`sum_cumshare_8k.png`, `sum_layer_corr_8k.png`).
   - *For.* Gradient energy is 2.5-3x more concentrated in cells than in whole tokens. Within a row, 95% of the
     blocks 7-31 energy needs (dense 1,952 / 7,808 rows/task):

     | task | share of cells | share of whole tokens |
     |---|---|---|
     | contradiction | 3.0 / 4.2% | 8.2 / 10.7% |
     | nq | 1.5 / 1.4% | 4.2 / 3.7% |
     | outlier | 6.4 / 7.5% | 18 / 20% |
     | oolong | 12 / 13% | 29 / 35% |

     A cell budget is directly a FLOP share.
   - *For.* Token importance is not one number per token. The rank correlation of a document token's energy between
     layers is 0.92-0.93 for adjacent layers, but 0.57-0.61 for block 7 vs 16 and 0.33-0.39 for block 7 vs 31
     (base: 0.78 / 0.45). Trained checkpoints show two blocks, about layers 0-15 and 17-31, with the step at layers
     17-20 from point 1. A ranking made once at block 7 is a poor ranking for the top half.
   - *Against finer granularity.* Within each half, adjacent layers agree at 0.93, so per-layer selection buys
     little over one selection per half. The recipe's re-selection at block 16 already matches the two-block
     structure. A third stage would only help if it sat at the layer-17-20 step rather than at 24.
   - *Against reading energy as a budget.* These are energy (attribution) shares, not routed fidelity. The frozen
     oracle sets that hold 95% of attribution give a routed per-step cos of only 0.08-0.19 once the other tokens are
     removed (`records/oracle-tokens-under-routing.md`). Cell-level dropping removes context at a finer grain, so the
     same gap applies. The case for it is that late-layer document cells carry little gradient (point 1), which
     supports aggressive late-stage budgets. It is not that a 3% cell set reproduces the update.
7. **2k vs 8k** (⚠ eval_size 128, `sum_*_2k.png`, `summary_2k.json`). Same picture; concentration is weaker per row
   because rows are shorter (95% of energy needs 4.5-55% of tokens and 2.5-35% of cells).

## Linear color scale (`*_lin.png`, `*_linclip.png`; added 2026-11-04 at Prasann's request)

The map_ / zoom_ / routed_ PNGs are redrawn from the same arrays with raw E and ||dL/dh|| (10 ** stored log10). `_lin`
runs each panel from 0 to its max with no clipping; `_linclip` caps it at the panel's 99th percentile (the cap and the max
are in each title). Zoom windows each get their own range and colorbar. The routed ratio panels are the linear ratio
on a diverging map centred at 1, linear on each side (0 .. 1 .. max); a plain 0 .. max map was unreadable because the
max is 50-8,000x. The log versions are unchanged.

- **`_lin` is black except the answer / query columns.** The panel max is 1,000-10,000x the 99th percentile (nq 8k E:
  base 6.8e4 vs 11.6, dense 1,952 1.8 vs 0.0015, dense 7,808 0.24 vs 2.5e-5, recipe 106 vs 0.08). On a linear scale,
  the update of a trained checkpoint is the answer tokens and nothing else; every document cell is under 0.1% of the
  answer spike.
- **`_linclip` shows how sparse the document gradient is, which the log scale hid.** On log, gold documents look like
  uniformly bright full-height columns. On a linear scale:
  - the document energy sits in a handful of documents (on nq 8k, the gold one plus 3-4 distractors);
  - it sits almost entirely in blocks 0-15, with only isolated tokens above 16;
  - inside the gold document it is a few tokens (the answer-bearing span; `zoom_nq_8k_lin.png`).
- **Depth.** On a linear scale, ||dL/dh|| on document tokens is concentrated in blocks 0-7 (`map_oolong_8k_A_linclip.png`).
  The steady 3-4 orders-of-magnitude decay that looks gentle on log means that, in absolute terms, nearly all of it
  sits in the bottom layers.
- **Routed ratio (linear).** In blocks 0-6, the tokens the router drops at block 7 are mostly below 1 (blue). The kept,
  gold and high-energy documents are at or above 1. The large ratios (up to 50x for ||dL/dh||, 8,000x for E) are
  isolated cells whose dense energy was near zero.

## Files

| Files | What |
|---|---|
| `map_<task>_<2k-solved\|2k-unsolved\|8k>_{E,A}.png` | full-row maps, 4 checkpoints stacked, oracle / router strips, segment bar |
| `zoom_<task>_<kind>.png` | token-resolution windows (gold document; query + answer) with token text |
| `routed_<task>_<kind>.png` | recipe: dense vs routed maps and the normalized ratio, for E and A |
| `*_lin.png` / `*_linclip.png` | the same map_ / zoom_ / routed_ PNGs on a linear scale (0 .. max / 0 .. 99th pct) |
| `sum_{layer,type,cumshare,layer_corr,routed}_{8k,2k}.png` | summaries over every row of the set |
| `summary_{8k,2k}.json` | the numbers above, per checkpoint and task, with eval_size |

## Code and provenance

- Code: `research/token_routing/grad_concentration/gradmap.py` (compute) and `gradmap_plot.py` (draw), plus
  `run_gradmap.sbatch`, `build_2k_rows.sbatch` and `run_gradmap_plot.sbatch`.
- Compute: jsteinhardt / mooney H200, `--qos=preemptive_high`, 1 GPU per job.
  - 8k: slurm 4150467-8 (all 4 checkpoints in about 7-10 min each).
  - 2k: slurm 4150478-9 (rows built by 4150469).
  - Plots: slurm 4150485 / 4150488; linear-scale redraw 4150621-4 (CPU, `--scale lin|linclip --maps-only`).
- Raw per-row arrays (log10 E / A, token types, keep sets) are on mooney `/data/prasann/gradmap/{out,out2k}/<label>/`.
