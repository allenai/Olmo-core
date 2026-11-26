# attn_edge_grad: which attention edges carry the fine-tuning gradient (Qwen3.5-4B + LoRA r64, long-context contradiction)

Prasann (2026-11-25), experiment `tokdrop_lr`. Record: `records/attention-edge-gradient-analysis.md`.
Follows the group-attention result in `records/longctx-contradiction-golddrop.md` (training on grouped attention did not
transfer to full-attention eval).

| file | what |
|---|---|
| `edge_attn.py` | exact causal attention, streamed in query chunks (TF32 matmuls, fp32 softmax), as a custom autograd function with three modes: `full`, `mask` (edges removed in forward + backward), `detach` (full forward; backward keeps only the kept edges' terms of the exact gradient, i.e. dS * keep with the full-forward D_i, dV from a * keep). The backward of a full pass accumulates, per layer and head, (a) logit-gradient mass \|dL/ds_ij\|, (b) value-path mass a_ij \|<g_i, v_j>\|, (c) attention mass a_ij, into (query segment, key segment) blocks, log2 distance bins and raw-edge log-histograms. `Pattern` = keep rule from segments (block matrix, per layer or shared), answer/question rows, always-kept preamble keys, local window. |
| `rows.py` | held-out rows -> segments (preamble / one per document incl. its trailing separator / question / answer from the last prompt position), gold documents (the claim numbers in the answer), training-style gold-document drop. |
| `probe_edges.py` | the driver: base `q35-4b-base-markerfix` + LoRA r64 / scale 32 on every linear of every block (fresh: A kaiming, B = 0, so the gradient is all in B, the real step-1 gradient; or a staged adapter + trained LM head). Per row: one forward, a full backward (reference LoRA gradient + Q1 accounting), one extra backward per pattern over the same graph (DETACH), and a fresh forward + backward per pattern (MASK). Writes `rows_s<shard>.jsonl` (Q1 summaries, Q2 per-block dots / norms, MASK CE) and `row_<i>.pt` (CountSketches of every gradient, 2^15 buckets, for batch-level cosines; layer block matrices for the first rows). |
| `analyze.py` | per condition: category table, concentration, distance, cos-vs-fraction curves (per-row mean +- SE and batch cosine), thresholds; `summary.json`, `cos_vs_frac.csv`, `categories.csv`, figures. |
| `test_edge_attn.py` | toy-sequence correctness: full mode vs autograd; streamed per-edge quantities vs autograd (s.grad etc.) per block / distance / histogram; DETACH and MASK with every edge kept = full bit for bit; DETACH pattern = explicit truncated decomposition; MASK = autograd through a masked softmax. |
| `stage_adapters_beaker.sh` | Beaker CPU job: LoRA tensors + LM head of `runs_lc/<arm>/stepN` (weka) -> S3 `_transfer/attn_edge_grad/`. |

Runs: Berkeley slurm (jsteinhardt, cubbins, 1 process per GPU). Node paths: code `/data/prasann/attn_edge_grad/code_*`,
held-out `/data/prasann/attn_edge_grad/data/heldout_{lc,kl}.npz` (from S3 `early_loss_proxy/`), adapters `.../ckpt/`,
outputs `.../out/<condition>/`. Summaries and figures copied to `results/` here.

## Status (2026-11-25): one quick sanity check only (scope cut to < 1 GPU-hour)

Ran one condition: `c16k_gddF` = 16 rows of `heldout_lc.npz` lc16k (contradiction, whole rows, mean 14,871 tokens,
324 docs, 6 gold) on gdd256's final adapter (`runs_lc/gdd256/step11496`, LoRA + trained head), 27 patterns x
{DETACH, MASK}; stopped at 16 of 32 rows. Numbers: `data/summary.json`, `data/cos_vs_frac.csv`,
`data/categories.csv` (`analyze.py` output). ⚠ This checkpoint has essentially zero loss on these rows
(answer CE 4e-6 per token), so the gradient is that of a saturated model; an early checkpoint (`gdd256_step1450.pt`,
staged at `/data/prasann/attn_edge_grad/ckpt/` on cubbins) or the base with a fresh LoRA is the informative next run.
The distance bins of this run used an earlier binning that misplaced pairs at exact powers of 2 (fixed afterwards).
No record was written. Compute: ~0.7 GPU-h on cubbins H200s.

Follow-up (same day): the same 16 rows and patterns on gdd256's step-1450 adapter (~1,821 PF; answer CE 0.020 +- 0.007
per token, not saturated), condition `c16k_gddE`, 2 H200s for 15 min (~0.5 GPU-h). `data/` now holds both conditions
(`analyze.py --conds c16k_gddF,c16k_gddE`). `c16k_gddE` used the corrected distance bins.
