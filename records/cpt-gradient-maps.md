---
title: CPT gradient maps - per-token x per-block gradient of the next-token loss on long documents (OLMo 3 7B, 32k / 64k / 128k), and the CPT oracle-budget table
date: 2026-11-10
updated: 2026-11-10
experiment: grad_composition
area: token_routing
kind: result
status: current
superseded_by:
artifact: https://claude.ai/artifact/BVaPjZdAKjETHdQDdGqqtQ
summary: After dense CPT the next-token-loss gradient has no sparse support. Ranked by energy, the smallest token set whose attribution gradient reaches cos 0.95 with the step gradient is 70% of tokens at 32k and 74% at 64k (SFT oracle 0.5-2%); (block, token) cells give the same 71-76%, and below ~65% kept the energy ranking is worse than random. The base's apparent sparsity (6-12%) is one shared direction that a random 6% also reaches. A late span's gradient reaches far back (35-59% of its pre-span energy beyond the 4k window), mostly through blocks 0-15. The attribution bound for token or cell dropping in CPT is about 1.3x per step. eval_size 8 / 8 / 3 documents (maps) and one 16 x 32k / 8 x 64k step (oracle).
---
# CPT gradient maps

**Brief (parent session, 2026-11-10).** The CPT counterpart of the SFT gradient maps
(`research/token_routing/grad_concentration/gradmaps/`). Per-token x per-block maps of the full CPT loss, maps of a
late / mid span's loss over the earlier context, the own-label vs context split, sparsity / composition statistics, and
what they imply for token dropping in CPT. Prasann added, during the session:
- the CPT version of the SFT oracle-budget table (attribution cos 0.95 / 0.9 per allocation rule);
- the same at the (block, token) cell level;
- linear color scales rather than log.

Full write-up with every table and figure: `research/token_routing/grad_concentration/gradmaps_cpt/findings.md`.
Results page (private): https://claude.ai/artifact/BVaPjZdAKjETHdQDdGqqtQ, section "CPT gradient maps".

## Setup

- **Model and checkpoints.** OLMo 3 7B, frozen weights, bf16; checkpoints on cubbins `/data/prasann/olmo3_cptdrop`.
  - Stage-2 base vs the dense CPT run trained at each length (LR 2e-5): `o3cd-Dlr20-480` (32k, YaRN x4),
    `o3cd-D64lr20-480` (64k, x8), `o3cd-D128lr20-120` (128k, x16).
  - OLMo stage 3 as a 32k reference.
  - Chosen because they are the only CPT checkpoints still on disk with matching dev sets.
  - The Qwen3.5-4B `cpt_u1B` runs are on weka only, and the K/V-detach split would need a Gated DeltaNet hook, so the
    Qwen contrast was not done.
- **Documents.** Longmino dev windows (2^17+ buckets, disjoint from training): ⚠ 8 documents at 32k, 8 at 64k, 3 at
  128k for the maps.
  - Oracle: one optimizer step of the CPT runs' own batch, 16 x 32k or 8 x 64k.
- **Probe** (`cpt_gradmap.py`): full loss, the K/V-detached own-label pass, and late / mid 1,024-token span losses.
  Recorded per (block, token):
  - E = sum over the block's linears of ||dL/dy_t||^2 ||x_t||^2;
  - ||dL/dh_b||;
  - exact per-block parameter-gradient energy.
- **Oracle budgets** (`cpt_oracle_budget.py`, `cpt_oracle_cells.py`).
  - Per-token rank-1 terms TensorSketched (32,768 + 8,192 dims; sketch vs exact norm^2 within ±2.1% on every row).
  - Energy-ranked nested sets under global / per-row / per-512-block / per-(row, block) / random allocation.
  - Measured: attribution cos with the full step gradient.

## Results (no-cot, teacher-forced gradients)

| | 32k base | 32k dense CPT | 64k base | 64k dense CPT | 128k dense CPT |
|---|---|---|---|---|---|
| share of tokens holding 95% of E (⚠ 8 / 8 / 3 docs) | 35% | 54% | 27% | 59% | 49% |
| share of cells holding 95% of E | 26% | 44% | 18% | 47% | 36% |
| oracle: global tokens for attribution cos 0.95, blocks 7-31 (one step) | 5.9% | **69.9%** | 12.0% | **74.4%** | - |
| oracle: global cells for cos 0.95 | 5.7% | **71.0%** | 11.9% | **75.8%** | - |
| oracle: random tokens for cos 0.95 (one draw) | 5.9% | 83.0% | 36.2% | 79.3% | - |
| blocks 0-6 share of exact parameter-gradient energy | 0.43 | 0.16 | 0.40 | 0.17 | 0.29 |
| late 1k span: pre-span E within 4k / 16k | 0.61 / 0.81 | 0.65 / 0.87 | 0.41 / 0.69 | 0.57 / 0.69 | 0.48 / 0.70 |
| context share of dL/dh at blocks 0 / 8 / 16 / 24 (median) | 0.95 / 0.87 / 0.82 / 0.68 | 0.84 / 0.73 / 0.45 / 0.21 | 0.93 / 0.88 / 0.83 / 0.69 | 0.90 / 0.85 / 0.47 / 0.21 | 0.89 / 0.85 / 0.47 / 0.19 |

1. **No sparse support after CPT, at the token or the cell level.**
   - The oracle needs 70-76% of tokens or cells; per-row and per-block quotas need more.
   - Energy-ranked sets sit at cos 0.13-0.3 from 5% to 40% kept, below random. The step gradient is a coherent sum
     of many small terms, while the high-energy tokens (high-loss labels, 2.3x enriched) are the noisy ones.
   - Exact ||sum g||^2 / sum ||g||^2 per block is 1.5-4 after CPT.
2. **The base's sparsity is one shared direction, not a few important tokens.**
   - The same ratio is 5-33 in the base, and a random 5.9% of tokens matches the oracle at 32k.
   - The base's energy also sits on structural tokens (newlines 6x) and copy targets (1.75x); after CPT both are flat
     or below 1.
3. **Long-range credit is real but lives in the early blocks.**
   - 35-59% of a late span's pre-span gradient energy is beyond the 4k sliding window and 13-31% beyond 16k.
   - Blocks 0-6 receive 0.43 of the span loss's pre-span energy (CPT 32k), blocks 24-31 only 0.035.
4. **Own label vs context.** After CPT, the late blocks learn from their own labels (context share 0.19-0.21 at block
   24 at every length), and the early blocks from serving as context (0.84-0.90 at block 0). The base is
   context-dominated throughout.
5. **Cross-block rank agreement rises with CPT** (E, block 7 vs 31: 0.38 → 0.51 at 32k, 0.22 → 0.46 at 128k). It
   doesn't help when nothing is sparse.
6. **Implication for token dropping (attribution bound, NOT a routing result).**
   - About 70% kept from block 7 = (7 + 0.7 x 25) / 32 = 0.77 of the FLOPs: about 1.3x per step. The energy-share
     view (95% of E) gives 1.5-2.0x.
   - SFT: 0.5-2% of tokens held cos 0.95, about 4x at the same split.
   - Even this is optimistic: in SFT, 95%-attribution sets kept only cos 0.08-0.19 when the other tokens were actually
     removed (`records/oracle-tokens-under-routing.md`).
   - In CPT, dropping tokens behaves like a smaller batch (random keep traces a smooth subsampling curve). This fits
     every token-drop arm losing to dense in `records/olmo3-cptdrop-results.md`.

## Limits

- The documents are few (8 / 8 / 3), and each oracle column is one step with random as a single draw.
- These are attribution statistics on frozen checkpoints, not routed gradients or training runs.
- The 128k base runs at YaRN x16 without ever training at that length: it is a degenerate reference, with 95% of E in
  2.7% of tokens.
- Not done:
  - Qwen3.5-4B (no local CPT checkpoints);
  - an oracle ranked by projection onto the step gradient (a fairer upper bound than energy; one extra pass per
    step).

## Where things live

- **Code:** branch `claude/cpt-gradmaps-34a768`, `research/token_routing/grad_concentration/`:
  - `cpt_gradmap.py`, `cpt_gradmap_plot.py`, `cpt_oracle_budget.py`, `cpt_oracle_cells.py`, `cpt_oracle_table.py`;
  - the `run_cpt_*.sbatch` launchers.
- **Figures and summaries:** `gradmaps_cpt/` (linear color scale).
- **Raw arrays** (cubbins, node-local): `/data/prasann/grad_composition/cpt_gradmap/<label>/`, `cpt_gradmap_figs/`,
  `cpt_oracle/`.
- **ops results:** 212 rows, metrics `gradcomp_cptmap_*` and `gradcomp_cptoracle_*`.
